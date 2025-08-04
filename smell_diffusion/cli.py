"""Command-line interface for Smell Diffusion Generator."""

import typer
import json
from pathlib import Path
from typing import Optional, List
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.json import JSON

from .core.smell_diffusion import SmellDiffusion
from .safety.evaluator import SafetyEvaluator
from .multimodal.generator import MultiModalGenerator
from .editing.editor import MoleculeEditor
from .design.accord import AccordDesigner
from .utils.config import get_config, ConfigManager
from .utils.logging import SmellDiffusionLogger
from .utils.validation import ValidationError

app = typer.Typer(
    name="smell-diffusion",
    help="Cross-modal diffusion model for generating fragrance molecules from text descriptions",
    no_args_is_help=True
)

console = Console()
logger = SmellDiffusionLogger("cli")


@app.command()
def generate(
    prompt: str = typer.Argument(..., help="Text description of desired fragrance"),
    num_molecules: int = typer.Option(5, "--num", "-n", help="Number of molecules to generate"),
    model: str = typer.Option("smell-diffusion-base-v1", "--model", "-m", help="Model to use"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output file for results"),
    safety_filter: bool = typer.Option(True, "--safety/--no-safety", help="Enable safety filtering"),
    guidance_scale: float = typer.Option(7.5, "--guidance", "-g", help="Guidance scale for generation"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output")
):
    """Generate fragrance molecules from text description."""
    
    try:
        # Load configuration
        config = get_config()
        
        console.print(f"🌸 Generating {num_molecules} fragrance molecules...")
        console.print(f"Prompt: [bold cyan]'{prompt}'[/bold cyan]")
        
        # Initialize model
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Loading model...", total=None)
            model_instance = SmellDiffusion.from_pretrained(model)
            progress.update(task, description="Generating molecules...")
            
            # Generate molecules
            molecules = model_instance.generate(
                prompt=prompt,
                num_molecules=num_molecules,
                guidance_scale=guidance_scale,
                safety_filter=safety_filter
            )
            
            progress.update(task, description="Complete!")
        
        if not isinstance(molecules, list):
            molecules = [molecules] if molecules else []
        
        # Display results
        _display_molecules(molecules, verbose)
        
        # Save results if requested
        if output:
            _save_results(molecules, output, prompt)
            console.print(f"✅ Results saved to [bold green]{output}[/bold green]")
        
    except ValidationError as e:
        console.print(f"❌ Validation Error: [bold red]{e}[/bold red]")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"❌ Error: [bold red]{e}[/bold red]")
        if verbose:
            console.print_exception()
        raise typer.Exit(1)


@app.command()
def safety_check(
    smiles: str = typer.Argument(..., help="SMILES string to evaluate"),
    comprehensive: bool = typer.Option(False, "--comprehensive", "-c", help="Run comprehensive safety check"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output file for report")
):
    """Evaluate safety of a molecule."""
    
    try:
        from .core.molecule import Molecule
        
        console.print(f"🛡️ Evaluating safety of molecule: [bold cyan]{smiles}[/bold cyan]")
        
        # Create molecule
        molecule = Molecule(smiles)
        if not molecule.is_valid:
            console.print("❌ [bold red]Invalid SMILES structure[/bold red]")
            raise typer.Exit(1)
        
        # Basic safety evaluation
        evaluator = SafetyEvaluator()
        
        if comprehensive:
            # Comprehensive evaluation
            report = evaluator.comprehensive_evaluation(molecule)
            _display_comprehensive_safety(report)
            
            if output:
                with open(output, 'w') as f:
                    json.dump(report.__dict__, f, indent=2, default=str)
        else:
            # Basic evaluation
            safety = evaluator.evaluate(molecule)
            _display_basic_safety(molecule, safety)
        
    except Exception as e:
        console.print(f"❌ Error: [bold red]{e}[/bold red]")
        raise typer.Exit(1)


@app.command()
def create_accord(
    name: str = typer.Argument(..., help="Name of the fragrance"),
    inspiration: str = typer.Option("", "--inspiration", "-i", help="Inspiration for the fragrance"),
    character: List[str] = typer.Option(["balanced"], "--character", "-c", help="Character traits"),
    season: str = typer.Option("all_seasons", "--season", "-s", help="Target season"),
    concentration: str = typer.Option("eau_de_parfum", "--concentration", help="Fragrance concentration"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output file for formula")
):
    """Create a complete fragrance accord."""
    
    try:
        console.print(f"🎨 Creating fragrance accord: [bold cyan]{name}[/bold cyan]")
        
        # Initialize designer
        model = SmellDiffusion.from_pretrained("smell-diffusion-base-v1")
        designer = AccordDesigner(model)
        
        # Create brief
        brief = {
            'name': name,
            'inspiration': inspiration,
            'character': character,
            'season': season
        }
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Creating accord...", total=None)
            
            accord = designer.create_accord(
                brief=brief,
                concentration=concentration
            )
            
            progress.update(task, description="Complete!")
        
        # Display accord
        _display_accord(accord)
        
        # Save formula if requested
        if output:
            formula = designer.export_formula(accord, str(output))
            console.print(f"✅ Formula saved to [bold green]{output}[/bold green]")
        
    except Exception as e:
        console.print(f"❌ Error: [bold red]{e}[/bold red]")
        raise typer.Exit(1)


@app.command()
def config(
    show: bool = typer.Option(False, "--show", help="Show current configuration"),
    create_default: bool = typer.Option(False, "--create-default", help="Create default config file"),
    config_path: Optional[Path] = typer.Option(None, "--path", help="Configuration file path")
):
    """Manage configuration."""
    
    try:
        config_manager = ConfigManager(config_path)
        
        if create_default:
            config_manager.create_default_config(config_path)
            console.print("✅ [bold green]Created default configuration file[/bold green]")
            return
        
        if show:
            current_config = get_config()
            config_dict = {
                "model": {
                    "model_name": current_config.model.model_name,
                    "device": current_config.model.device,
                    "cache_dir": current_config.model.cache_dir
                },
                "generation": {
                    "num_molecules": current_config.generation.num_molecules,
                    "guidance_scale": current_config.generation.guidance_scale,
                    "safety_filter": current_config.generation.safety_filter
                },
                "safety": {
                    "min_safety_score": current_config.safety.min_safety_score,
                    "ifra_compliance": current_config.safety.ifra_compliance
                },
                "logging": {
                    "level": current_config.logging.level,
                    "enable_file_logging": current_config.logging.enable_file_logging
                }
            }
            
            console.print("📋 Current Configuration:")
            console.print(JSON.from_data(config_dict))
        else:
            console.print("Use --show to display configuration or --create-default to create config file")
    
    except Exception as e:
        console.print(f"❌ Error: [bold red]{e}[/bold red]")
        raise typer.Exit(1)


def _display_molecules(molecules: List, verbose: bool = False):
    """Display generated molecules in a table."""
    
    table = Table(title="Generated Molecules")
    table.add_column("ID", style="cyan", no_wrap=True)
    table.add_column("SMILES", style="green")
    table.add_column("MW", justify="right")
    table.add_column("LogP", justify="right")
    table.add_column("Notes", style="yellow")
    table.add_column("Safety", justify="right")
    
    if verbose:
        table.add_column("Longevity")
        table.add_column("Intensity")
    
    for i, mol in enumerate(molecules, 1):
        if mol and mol.is_valid:
            notes = ", ".join(mol.fragrance_notes.top + mol.fragrance_notes.middle + mol.fragrance_notes.base)
            safety = mol.get_safety_profile()
            
            row = [
                str(i),
                mol.smiles[:50] + ("..." if len(mol.smiles) > 50 else ""),
                f"{mol.molecular_weight:.1f}",
                f"{mol.logp:.2f}",
                notes[:30] + ("..." if len(notes) > 30 else ""),
                f"{safety.score:.0f}/100"
            ]
            
            if verbose:
                row.extend([
                    mol.longevity,
                    f"{mol.intensity:.1f}/10"
                ])
            
            table.add_row(*row)
        else:
            table.add_row(str(i), "Invalid", "-", "-", "-", "0/100")
    
    console.print(table)


def _display_basic_safety(molecule, safety):
    """Display basic safety evaluation."""
    
    # Molecule info panel
    mol_info = f"""
SMILES: {molecule.smiles}
Molecular Weight: {molecule.molecular_weight:.1f} g/mol
LogP: {molecule.logp:.2f}
"""
    
    console.print(Panel(mol_info.strip(), title="Molecule Information", border_style="blue"))
    
    # Safety info panel
    safety_color = "green" if safety.score >= 70 else "yellow" if safety.score >= 50 else "red"
    
    safety_info = f"""
Safety Score: {safety.score:.1f}/100
IFRA Compliant: {'✓' if safety.ifra_compliant else '✗'}
"""
    
    if safety.allergens:
        safety_info += f"Allergens: {', '.join(safety.allergens)}\n"
    
    if safety.warnings:
        safety_info += f"Warnings: {'; '.join(safety.warnings)}\n"
    
    console.print(Panel(safety_info.strip(), title="Safety Evaluation", border_style=safety_color))


def _display_comprehensive_safety(report):
    """Display comprehensive safety report."""
    
    # Overall status
    status_color = "green" if report.overall_score >= 70 else "yellow" if report.overall_score >= 50 else "red"
    
    console.print(Panel(
        f"Overall Safety Score: {report.overall_score:.1f}/100\n"
        f"IFRA Compliant: {'✓' if report.ifra_compliant else '✗'}",
        title="Safety Overview",
        border_style=status_color
    ))
    
    # Regulatory status table
    reg_table = Table(title="Regulatory Status")
    reg_table.add_column("Region", style="cyan")
    reg_table.add_column("Status", style="green")
    
    for region, status in report.regulatory_status.items():
        reg_table.add_row(region, status)
    
    console.print(reg_table)
    
    # Allergen analysis
    if report.allergen_analysis["detected"]:
        allergen_table = Table(title="Detected Allergens")
        allergen_table.add_column("Allergen", style="red")
        allergen_table.add_column("Similarity", justify="right")
        
        for allergen in report.allergen_analysis["detected"]:
            allergen_table.add_row(
                allergen["name"],
                f"{allergen['similarity']:.2f}"
            )
        
        console.print(allergen_table)
    
    # Recommendations
    if report.recommendations:
        rec_text = "\n".join(f"• {rec}" for rec in report.recommendations)
        console.print(Panel(rec_text, title="Recommendations", border_style="yellow"))


def _display_accord(accord):
    """Display fragrance accord."""
    
    # Header info
    console.print(Panel(
        f"Inspiration: {accord.inspiration}\n"
        f"Target Audience: {accord.target_audience}\n"
        f"Season: {accord.season}\n"
        f"Character: {', '.join(accord.character)}\n"
        f"Concentration: {accord.concentration.replace('_', ' ').title()}",
        title=f"🌸 {accord.name}",
        border_style="magenta"
    ))
    
    # Fragrance pyramid
    sections = [
        ("Top Notes (0-15 min)", accord.top_notes, "cyan"),
        ("Heart Notes (15 min-3 hrs)", accord.heart_notes, "yellow"),
        ("Base Notes (3+ hrs)", accord.base_notes, "green")
    ]
    
    for section_name, notes, color in sections:
        if notes:
            table = Table(title=section_name)
            table.add_column("Note", style=color)
            table.add_column("Percentage", justify="right")
            table.add_column("Intensity", justify="right")
            
            for note in notes:
                table.add_row(
                    note.name,
                    f"{note.percentage:.1f}%",
                    f"{note.intensity:.1f}/10"
                )
            
            console.print(table)


def _save_results(molecules: List, output_path: Path, prompt: str):
    """Save generation results to file."""
    
    results = {
        "prompt": prompt,
        "timestamp": logger.logger.handlers[0].formatter.formatTime(logger.logger.makeRecord(
            "", 0, "", 0, "", (), None
        )),
        "molecules": []
    }
    
    for i, mol in enumerate(molecules, 1):
        if mol:
            mol_data = mol.to_dict()
            mol_data["id"] = i
            results["molecules"].append(mol_data)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)


def main():
    """Main CLI entry point."""
    app()


if __name__ == "__main__":
    main()