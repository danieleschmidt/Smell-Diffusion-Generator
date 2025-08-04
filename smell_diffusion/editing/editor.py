"""Molecule editing and interpolation tools."""

from typing import List, Optional, Dict, Any
import numpy as np
from ..core.molecule import Molecule
from ..core.smell_diffusion import SmellDiffusion


class MoleculeEditor:
    """Tools for editing and interpolating fragrance molecules."""
    
    def __init__(self, base_model: SmellDiffusion):
        """Initialize with base diffusion model."""
        self.base_model = base_model
    
    def edit(self, 
             molecule: str,
             instruction: str,
             preservation_strength: float = 0.7,
             num_steps: int = 50,
             **kwargs) -> Molecule:
        """Edit a molecule based on text instruction."""
        
        base_mol = Molecule(molecule)
        if not base_mol.is_valid:
            raise ValueError(f"Invalid input molecule: {molecule}")
        
        # Analyze the editing instruction
        edit_direction = self._analyze_instruction(instruction)
        
        # Generate variations based on instruction
        edited_smiles = self._apply_molecular_edit(
            molecule, edit_direction, preservation_strength
        )
        
        edited_mol = Molecule(edited_smiles, description=f"Edited: {instruction}")
        
        # Validate edit preserves core structure if requested
        if preservation_strength > 0.5:
            similarity = self._calculate_similarity(molecule, edited_smiles)
            if similarity < preservation_strength:
                # Fall back to more conservative edit
                edited_smiles = self._conservative_edit(molecule, edit_direction)
                edited_mol = Molecule(edited_smiles, description=f"Conservatively edited: {instruction}")
        
        return edited_mol
    
    def interpolate(self,
                   start: str,
                   end: str,
                   steps: int = 10,
                   guided_by: Optional[str] = None) -> List[Molecule]:
        """Interpolate between two molecules."""
        
        start_mol = Molecule(start)
        end_mol = Molecule(end)
        
        if not start_mol.is_valid or not end_mol.is_valid:
            raise ValueError("Both start and end molecules must be valid")
        
        interpolated = []
        
        for i in range(steps + 1):
            alpha = i / steps  # Interpolation factor (0 to 1)
            
            # Generate intermediate molecule
            if alpha == 0:
                intermediate_smiles = start
            elif alpha == 1:
                intermediate_smiles = end
            else:
                intermediate_smiles = self._molecular_interpolation(start, end, alpha)
            
            description = f"Interpolation step {i}/{steps}"
            if guided_by:
                description += f" guided by: {guided_by}"
            
            intermediate_mol = Molecule(intermediate_smiles, description=description)
            interpolated.append(intermediate_mol)
        
        return interpolated
    
    def visualize_transformation(self, molecules: List[Molecule]) -> str:
        """Generate visualization info for molecular transformation."""
        if not molecules:
            return "No molecules to visualize"
        
        viz_info = f"Molecular Transformation ({len(molecules)} steps):\n"
        viz_info += "=" * 50 + "\n"
        
        for i, mol in enumerate(molecules):
            viz_info += f"Step {i}: {mol.smiles}\n"
            viz_info += f"  MW: {mol.molecular_weight:.1f}, LogP: {mol.logp:.2f}\n"
            viz_info += f"  Notes: {', '.join(mol.fragrance_notes.top + mol.fragrance_notes.middle + mol.fragrance_notes.base)}\n"
            viz_info += f"  Safety: {mol.get_safety_profile().score:.0f}/100\n\n"
        
        return viz_info
    
    def _analyze_instruction(self, instruction: str) -> Dict[str, float]:
        """Analyze editing instruction to determine molecular changes."""
        instruction_lower = instruction.lower()
        
        # Map instructions to molecular property changes
        edit_direction = {
            'floral_increase': 0.0,
            'citrus_increase': 0.0,
            'woody_increase': 0.0,
            'fresh_increase': 0.0,
            'sweet_increase': 0.0,
            'intensity_change': 0.0,
            'molecular_weight_change': 0.0,
            'polarity_change': 0.0
        }
        
        # Analyze instruction for scent changes
        if any(word in instruction_lower for word in ['floral', 'flower', 'rose', 'jasmine']):
            edit_direction['floral_increase'] = 1.0 if 'more' in instruction_lower else -1.0
        
        if any(word in instruction_lower for word in ['citrus', 'lemon', 'orange', 'bergamot']):
            edit_direction['citrus_increase'] = 1.0 if 'more' in instruction_lower else -1.0
        
        if any(word in instruction_lower for word in ['woody', 'wood', 'cedar', 'sandalwood']):
            edit_direction['woody_increase'] = 1.0 if 'more' in instruction_lower else -1.0
        
        if any(word in instruction_lower for word in ['fresh', 'clean', 'aquatic']):
            edit_direction['fresh_increase'] = 1.0 if 'more' in instruction_lower else -1.0
        
        if any(word in instruction_lower for word in ['sweet', 'vanilla', 'sugar']):
            edit_direction['sweet_increase'] = 1.0 if 'more' in instruction_lower else -1.0
        
        # Analyze for intensity changes
        if any(word in instruction_lower for word in ['stronger', 'intense', 'powerful']):
            edit_direction['intensity_change'] = 1.0
        elif any(word in instruction_lower for word in ['softer', 'subtle', 'gentle']):
            edit_direction['intensity_change'] = -1.0
        
        return edit_direction
    
    def _apply_molecular_edit(self, molecule: str, edit_direction: Dict[str, float],
                            preservation_strength: float) -> str:
        """Apply molecular modifications based on edit direction."""
        
        # This is a simplified approach - in reality would use proper cheminformatics
        base_mol = Molecule(molecule)
        
        # Start with original molecule
        edited_smiles = molecule
        
        # Apply modifications based on edit direction
        if edit_direction['floral_increase'] > 0:
            # Add/modify to increase floral character
            if 'C=O' not in edited_smiles and preservation_strength < 0.8:
                # Add carbonyl group for floral character
                edited_smiles = edited_smiles.replace('CC', 'CC=O', 1)
        
        if edit_direction['citrus_increase'] > 0:
            # Increase citrus character (more volatile, lighter)
            if 'C(C)' in edited_smiles and preservation_strength < 0.8:
                # Reduce branching for higher volatility
                edited_smiles = edited_smiles.replace('C(C)C', 'CC', 1)
        
        if edit_direction['woody_increase'] > 0:
            # Increase woody character (higher MW, more rings)
            if preservation_strength < 0.7:
                # Add cyclic structure for woody character  
                if '1' not in edited_smiles and len(edited_smiles) < 30:
                    edited_smiles = edited_smiles + 'C1CC1'  # Simple cyclopropyl
        
        # Validate edited molecule
        test_mol = Molecule(edited_smiles)
        if not test_mol.is_valid:
            # Fall back to original if edit created invalid molecule
            edited_smiles = molecule
        
        return edited_smiles
    
    def _conservative_edit(self, molecule: str, edit_direction: Dict[str, float]) -> str:
        """Apply conservative molecular edits that preserve structure."""
        
        # Very minimal changes that are likely to preserve validity
        edited_smiles = molecule
        
        # Only make very safe substitutions
        if edit_direction['floral_increase'] > 0:
            # Replace single methyl with ethyl (minimal change)
            if 'CC' in edited_smiles and 'CCC' not in edited_smiles:
                edited_smiles = edited_smiles.replace('CC', 'CCC', 1)
        
        return edited_smiles
    
    def _molecular_interpolation(self, start: str, end: str, alpha: float) -> str:
        """Perform molecular interpolation between two SMILES."""
        
        # Simplified interpolation - in practice would use proper molecular alignment
        start_mol = Molecule(start)
        end_mol = Molecule(end)
        
        # Property-based interpolation
        start_mw = start_mol.molecular_weight
        end_mw = end_mol.molecular_weight
        target_mw = start_mw + alpha * (end_mw - start_mw)
        
        # Select from database based on target properties
        candidates = []
        for category_molecules in self.base_model.FRAGRANCE_DATABASE.values():
            for smiles in category_molecules:
                mol = Molecule(smiles)
                if mol.is_valid:
                    mw_diff = abs(mol.molecular_weight - target_mw)
                    candidates.append((smiles, mw_diff))
        
        if candidates:
            # Return molecule closest to target molecular weight
            candidates.sort(key=lambda x: x[1])
            return candidates[0][0]
        
        # Fallback to start molecule
        return start
    
    def _calculate_similarity(self, smiles1: str, smiles2: str) -> float:
        """Calculate structural similarity between two molecules."""
        if smiles1 == smiles2:
            return 1.0
        
        # Simple similarity based on common substrings
        common_chars = set(smiles1).intersection(set(smiles2))
        total_chars = set(smiles1).union(set(smiles2))
        
        return len(common_chars) / len(total_chars) if total_chars else 0.0