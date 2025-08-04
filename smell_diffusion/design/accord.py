"""Fragrance accord design and pyramid creation."""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from ..core.molecule import Molecule
from ..core.smell_diffusion import SmellDiffusion


@dataclass 
class FragranceNote:
    """Individual fragrance note in an accord."""
    name: str
    smiles: str
    percentage: float
    category: str  # top, heart, base
    intensity: float
    longevity: str
    

@dataclass
class FragranceAccord:
    """Complete fragrance accord with pyramid structure."""
    name: str
    inspiration: str
    top_notes: List[FragranceNote]
    heart_notes: List[FragranceNote] 
    base_notes: List[FragranceNote]
    concentration: str
    target_audience: str
    season: str
    character: List[str]


class AccordDesigner:
    """Designer for creating complete fragrance accords."""
    
    NOTE_CATEGORIES = {
        'top': {
            'volatility_range': (50, 150),  # MW range for top notes
            'longevity': ['very_short', 'short'],
            'typical_notes': ['citrus', 'herbal', 'green', 'fresh']
        },
        'heart': {
            'volatility_range': (150, 250),  # MW range for heart notes  
            'longevity': ['medium', 'long'],
            'typical_notes': ['floral', 'fruity', 'spicy', 'aromatic']
        },
        'base': {
            'volatility_range': (250, 400),  # MW range for base notes
            'longevity': ['long', 'very_long'], 
            'typical_notes': ['woody', 'musky', 'amber', 'vanilla']
        }
    }
    
    CONCENTRATION_PROFILES = {
        'eau_de_cologne': {'strength': 0.05, 'longevity': 'short'},
        'eau_de_toilette': {'strength': 0.10, 'longevity': 'medium'},
        'eau_de_parfum': {'strength': 0.15, 'longevity': 'long'},
        'parfum': {'strength': 0.25, 'longevity': 'very_long'}
    }
    
    def __init__(self, base_model: SmellDiffusion):
        """Initialize with base diffusion model."""
        self.base_model = base_model
    
    def create_accord(self,
                     brief: Dict[str, Any],
                     num_top_notes: int = 3,
                     num_heart_notes: int = 4,
                     num_base_notes: int = 3,
                     concentration: str = 'eau_de_parfum') -> FragranceAccord:
        """Create a complete fragrance accord from a brief."""
        
        # Extract brief information
        name = brief.get('name', 'Untitled Fragrance')
        inspiration = brief.get('inspiration', '')
        target_audience = brief.get('target_audience', 'unisex')
        season = brief.get('season', 'all_seasons')
        character = brief.get('character', ['balanced'])
        
        # Generate notes for each category
        top_notes = self._generate_notes_for_category(
            'top', num_top_notes, brief, concentration
        )
        
        heart_notes = self._generate_notes_for_category(
            'heart', num_heart_notes, brief, concentration
        )
        
        base_notes = self._generate_notes_for_category(
            'base', num_base_notes, brief, concentration
        )
        
        # Balance the accord
        top_notes, heart_notes, base_notes = self._balance_accord(
            top_notes, heart_notes, base_notes, concentration
        )
        
        return FragranceAccord(
            name=name,
            inspiration=inspiration,
            top_notes=top_notes,
            heart_notes=heart_notes,
            base_notes=base_notes,
            concentration=concentration,
            target_audience=target_audience,
            season=season,
            character=character
        )
    
    def _generate_notes_for_category(self, category: str, num_notes: int,
                                   brief: Dict[str, Any], concentration: str) -> List[FragranceNote]:
        """Generate fragrance notes for a specific category."""
        
        # Build prompt for this category
        character = brief.get('character', ['balanced'])
        inspiration = brief.get('inspiration', '')
        season = brief.get('season', 'all_seasons')
        
        category_prompt = self._build_category_prompt(
            category, character, inspiration, season
        )
        
        # Generate molecules for this category
        molecules = self.base_model.generate(
            prompt=category_prompt,
            num_molecules=num_notes * 2,  # Generate extra for selection
            safety_filter=True
        )
        
        if not isinstance(molecules, list):
            molecules = [molecules] if molecules else []
        
        # Filter molecules suitable for this category
        suitable_molecules = []
        for mol in molecules:
            if mol and mol.is_valid:
                if self._is_suitable_for_category(mol, category):
                    suitable_molecules.append(mol)
        
        # Convert to FragranceNote objects
        notes = []
        for i, mol in enumerate(suitable_molecules[:num_notes]):
            note_name = self._generate_note_name(mol, category)
            note = FragranceNote(
                name=note_name,
                smiles=mol.smiles,
                percentage=0.0,  # Will be set in balance_accord
                category=category,
                intensity=mol.intensity,
                longevity=mol.longevity
            )
            notes.append(note)
        
        # Fill remaining slots if needed
        while len(notes) < num_notes:
            # Use fallback molecules from database
            fallback_mol = self._get_fallback_molecule(category)
            if fallback_mol:
                note_name = self._generate_note_name(fallback_mol, category)
                note = FragranceNote(
                    name=note_name,
                    smiles=fallback_mol.smiles,
                    percentage=0.0,
                    category=category,
                    intensity=fallback_mol.intensity,
                    longevity=fallback_mol.longevity
                )
                notes.append(note)
            else:
                break
        
        return notes
    
    def _build_category_prompt(self, category: str, character: List[str],
                             inspiration: str, season: str) -> str:
        """Build text prompt for generating category-specific notes."""
        
        # Base prompts for each category
        base_prompts = {
            'top': 'Fresh, bright opening notes with high impact and volatility',
            'heart': 'Rich, complex middle notes that form the main character',
            'base': 'Deep, lasting base notes that provide foundation and longevity'
        }
        
        prompt = base_prompts[category]
        
        # Add character descriptors
        if character:
            char_str = ', '.join(character)
            prompt += f' with {char_str} character'
        
        # Add seasonal context
        if season != 'all_seasons':
            prompt += f' suitable for {season}'
        
        # Add inspiration
        if inspiration:
            prompt += f' inspired by {inspiration}'
        
        return prompt
    
    def _is_suitable_for_category(self, molecule: Molecule, category: str) -> bool:
        """Check if molecule is suitable for the given category."""
        
        category_info = self.NOTE_CATEGORIES[category]
        mw_min, mw_max = category_info['volatility_range']
        
        # Check molecular weight range
        if not (mw_min <= molecule.molecular_weight <= mw_max):
            return False
        
        # Check longevity
        if molecule.longevity not in category_info['longevity']:
            return False
        
        # Check fragrance notes alignment
        mol_notes = molecule.fragrance_notes.top + molecule.fragrance_notes.middle + molecule.fragrance_notes.base
        typical_notes = category_info['typical_notes']
        
        # Should have at least one matching note type
        has_matching_note = any(note in typical_notes for note in mol_notes)
        
        return has_matching_note
    
    def _generate_note_name(self, molecule: Molecule, category: str) -> str:
        """Generate a descriptive name for a fragrance note."""
        
        # Get dominant fragrance characteristics
        fragrance_notes = molecule.fragrance_notes
        all_notes = fragrance_notes.top + fragrance_notes.middle + fragrance_notes.base
        
        if not all_notes:
            return f"{category.title()} Note"
        
        # Use most prominent note
        primary_note = all_notes[0]
        
        # Add intensity descriptor
        intensity = fragrance_notes.intensity
        if intensity > 8:
            intensity_desc = "Intense"
        elif intensity > 6:
            intensity_desc = "Rich"
        elif intensity > 4:
            intensity_desc = "Moderate"
        else:
            intensity_desc = "Subtle"
        
        return f"{intensity_desc} {primary_note.title()}"
    
    def _get_fallback_molecule(self, category: str) -> Optional[Molecule]:
        """Get fallback molecule for category when generation fails."""
        
        fallback_smiles = {
            'top': "CC(C)=CCCC(C)=CCO",  # Geraniol
            'heart': "CC1=CC=C(C=C1)C=O",  # p-Tolualdehyde  
            'base': "CC12CCC(CC1=CCC2=O)C(C)(C)C"  # Cedrol-like
        }
        
        smiles = fallback_smiles.get(category)
        return Molecule(smiles) if smiles else None
    
    def _balance_accord(self, top_notes: List[FragranceNote],
                       heart_notes: List[FragranceNote], 
                       base_notes: List[FragranceNote],
                       concentration: str) -> tuple:
        """Balance the percentages of notes in the accord."""
        
        # Standard accord balance ratios
        balance_ratios = {
            'eau_de_cologne': {'top': 0.6, 'heart': 0.3, 'base': 0.1},
            'eau_de_toilette': {'top': 0.5, 'heart': 0.35, 'base': 0.15},
            'eau_de_parfum': {'top': 0.3, 'heart': 0.5, 'base': 0.2},
            'parfum': {'top': 0.2, 'heart': 0.4, 'base': 0.4}
        }
        
        ratios = balance_ratios.get(concentration, balance_ratios['eau_de_parfum'])
        
        # Distribute percentages within each category
        if top_notes:
            top_each = ratios['top'] / len(top_notes) * 100
            for note in top_notes:
                note.percentage = top_each
        
        if heart_notes:
            heart_each = ratios['heart'] / len(heart_notes) * 100
            for note in heart_notes:
                note.percentage = heart_each
        
        if base_notes:
            base_each = ratios['base'] / len(base_notes) * 100
            for note in base_notes:
                note.percentage = base_each
        
        return top_notes, heart_notes, base_notes
    
    def export_formula(self, accord: FragranceAccord, filename: str) -> str:
        """Export accord formula to text format."""
        
        formula = f"FRAGRANCE FORMULA: {accord.name}\n"
        formula += "=" * 50 + "\n\n"
        
        formula += f"Inspiration: {accord.inspiration}\n"
        formula += f"Target Audience: {accord.target_audience}\n"
        formula += f"Season: {accord.season}\n"
        formula += f"Character: {', '.join(accord.character)}\n"
        formula += f"Concentration: {accord.concentration.replace('_', ' ').title()}\n\n"
        
        # Top notes
        formula += "TOP NOTES (0-15 minutes):\n"
        formula += "-" * 30 + "\n"
        for note in accord.top_notes:
            formula += f"{note.name:<25} {note.percentage:>6.1f}%\n"
            formula += f"  SMILES: {note.smiles}\n"
            formula += f"  Intensity: {note.intensity:.1f}/10\n\n"
        
        # Heart notes
        formula += "HEART NOTES (15 minutes - 3 hours):\n"
        formula += "-" * 30 + "\n"
        for note in accord.heart_notes:
            formula += f"{note.name:<25} {note.percentage:>6.1f}%\n"
            formula += f"  SMILES: {note.smiles}\n"
            formula += f"  Intensity: {note.intensity:.1f}/10\n\n"
        
        # Base notes
        formula += "BASE NOTES (3+ hours):\n"
        formula += "-" * 30 + "\n"
        for note in accord.base_notes:
            formula += f"{note.name:<25} {note.percentage:>6.1f}%\n"
            formula += f"  SMILES: {note.smiles}\n"
            formula += f"  Intensity: {note.intensity:.1f}/10\n\n"
        
        # Save to file
        with open(filename, 'w') as f:
            f.write(formula)
        
        return formula