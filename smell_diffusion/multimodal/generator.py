"""Multi-modal fragrance generation combining text, images, and reference molecules."""

from typing import List, Optional, Dict, Any, Union
import numpy as np
from PIL import Image
from ..core.smell_diffusion import SmellDiffusion
from ..core.molecule import Molecule


class MultiModalGenerator:
    """Multi-modal generator combining text, images, and reference molecules."""
    
    def __init__(self, base_model: SmellDiffusion):
        """Initialize with base diffusion model."""
        self.base_model = base_model
        
    @classmethod
    def from_pretrained(cls, model_name: str) -> "MultiModalGenerator":
        """Load pre-trained multi-modal model."""
        base_model = SmellDiffusion.from_pretrained(model_name)
        return cls(base_model)
    
    def generate(self,
                 text: Optional[str] = None,
                 image: Optional[Image.Image] = None,
                 reference_smiles: Optional[str] = None,
                 interpolation_weights: Optional[Dict[str, float]] = None,
                 num_molecules: int = 1,
                 diversity_penalty: float = 0.5,
                 **kwargs) -> List[Molecule]:
        """Generate molecules using multiple input modalities."""
        
        if interpolation_weights is None:
            interpolation_weights = {'text': 1.0, 'image': 0.0, 'reference': 0.0}
        
        # Generate base molecules from text
        if text and interpolation_weights.get('text', 0) > 0:
            text_molecules = self.base_model.generate(
                prompt=text, 
                num_molecules=num_molecules * 2,  # Generate more for selection
                safety_filter=True
            )
            if not isinstance(text_molecules, list):
                text_molecules = [text_molecules]
        else:
            text_molecules = []
        
        # Process image input
        image_influence = self._process_image(image) if image else {}
        
        # Process reference molecule
        reference_influence = self._process_reference(reference_smiles) if reference_smiles else {}
        
        # Combine influences
        final_molecules = []
        
        for mol in text_molecules[:num_molecules]:
            if mol is None:
                continue
                
            # Apply image influence
            if image and interpolation_weights.get('image', 0) > 0:
                mol = self._apply_image_influence(mol, image_influence, 
                                                interpolation_weights['image'])
            
            # Apply reference influence
            if reference_smiles and interpolation_weights.get('reference', 0) > 0:
                mol = self._apply_reference_influence(mol, reference_influence,
                                                    interpolation_weights['reference'])
            
            final_molecules.append(mol)
        
        # Apply diversity penalty if needed
        if diversity_penalty > 0:
            final_molecules = self._apply_diversity_penalty(final_molecules, diversity_penalty)
        
        return final_molecules[:num_molecules]
    
    def _process_image(self, image: Image.Image) -> Dict[str, Any]:
        """Extract features from image for fragrance generation."""
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Simple color analysis
        colors = image.getcolors(maxcolors=256*256*256)
        if colors:
            # Get dominant colors
            dominant_colors = sorted(colors, key=lambda x: x[0], reverse=True)[:5]
            
            # Map colors to scent categories (simplified)
            color_to_scent = {
                'blue': ['fresh', 'aquatic', 'marine'],
                'green': ['herbal', 'leafy', 'fresh'],
                'yellow': ['citrus', 'bright', 'energetic'],
                'red': ['spicy', 'warm', 'passionate'],
                'purple': ['floral', 'lavender', 'mysterious'],
                'orange': ['citrus', 'warm', 'energetic'],
                'pink': ['floral', 'rose', 'feminine'],
                'brown': ['woody', 'earthy', 'warm']
            }
            
            suggested_scents = []
            for count, rgb in dominant_colors:
                r, g, b = rgb
                # Simple color classification
                if b > r and b > g:
                    suggested_scents.extend(color_to_scent['blue'])
                elif g > r and g > b:
                    suggested_scents.extend(color_to_scent['green'])
                elif r > g and r > b:
                    if g > b:
                        suggested_scents.extend(color_to_scent['orange'])
                    else:
                        suggested_scents.extend(color_to_scent['red'])
            
            return {
                'dominant_colors': dominant_colors[:3],
                'suggested_scents': list(set(suggested_scents))[:5],
                'brightness': sum(rgb[0] + rgb[1] + rgb[2] for _, rgb in dominant_colors[:3]) / (3 * 255 * 3)
            }
        
        return {'suggested_scents': ['neutral']}
    
    def _process_reference(self, reference_smiles: str) -> Dict[str, Any]:
        """Analyze reference molecule for structural features."""
        ref_mol = Molecule(reference_smiles)
        
        if not ref_mol.is_valid:
            return {}
        
        return {
            'molecular_weight': ref_mol.molecular_weight,
            'logp': ref_mol.logp,
            'fragrance_notes': ref_mol.fragrance_notes,
            'structural_features': self._extract_structural_features(reference_smiles)
        }
    
    def _extract_structural_features(self, smiles: str) -> Dict[str, bool]:
        """Extract key structural features from SMILES."""
        features = {
            'aromatic': '1=CC=CC=C1' in smiles or 'c' in smiles.lower(),
            'aldehyde': 'C=O' in smiles,
            'alcohol': 'CO' in smiles and 'C=O' not in smiles,
            'ester': 'COC(=O)' in smiles or 'C(=O)OC' in smiles,
            'ether': 'COC' in smiles and 'C(=O)OC' not in smiles,
            'double_bond': 'C=C' in smiles,
            'triple_bond': 'C#C' in smiles,
            'cyclic': '1' in smiles,
            'branched': '(' in smiles
        }
        return features
    
    def _apply_image_influence(self, molecule: Molecule, image_influence: Dict[str, Any],
                             weight: float) -> Molecule:
        """Apply image-derived influence to molecule."""
        # In a real implementation, this would modify the molecular structure
        # For now, we'll create a description that incorporates image insights
        
        image_scents = image_influence.get('suggested_scents', [])
        if image_scents and molecule.description:
            enhanced_description = f"{molecule.description} with {', '.join(image_scents[:2])} inspiration"
            molecule.description = enhanced_description
        
        return molecule
    
    def _apply_reference_influence(self, molecule: Molecule, reference_influence: Dict[str, Any],
                                 weight: float) -> Molecule:
        """Apply reference molecule influence."""
        # In a real implementation, this would perform structural interpolation
        # For now, we'll blend the descriptions and properties conceptually
        
        ref_notes = reference_influence.get('fragrance_notes')
        if ref_notes and molecule.description:
            ref_note_names = ref_notes.top + ref_notes.middle + ref_notes.base
            if ref_note_names:
                enhanced_description = f"{molecule.description} blended with {', '.join(ref_note_names[:2])}"
                molecule.description = enhanced_description
        
        return molecule
    
    def _apply_diversity_penalty(self, molecules: List[Molecule], penalty: float) -> List[Molecule]:
        """Apply diversity penalty to promote different molecule structures."""
        if len(molecules) <= 1:
            return molecules
        
        # Simple diversity check - in practice would use molecular fingerprints
        diverse_molecules = [molecules[0]]  # Always keep first
        
        for mol in molecules[1:]:
            is_diverse = True
            for existing_mol in diverse_molecules:
                # Simple similarity check
                if self._structural_similarity(mol.smiles, existing_mol.smiles) > (1 - penalty):
                    is_diverse = False
                    break
            
            if is_diverse:
                diverse_molecules.append(mol)
        
        return diverse_molecules
    
    def _structural_similarity(self, smiles1: str, smiles2: str) -> float:
        """Calculate structural similarity between molecules."""
        if smiles1 == smiles2:
            return 1.0
        
        # Simple character-based similarity
        set1 = set(smiles1)
        set2 = set(smiles2)
        
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        return intersection / union if union > 0 else 0.0
    
    def rank_by_multimodal_similarity(self, molecules: List[Molecule],
                                    text: Optional[str] = None,
                                    image: Optional[Image.Image] = None,
                                    reference_smiles: Optional[str] = None) -> List[Molecule]:
        """Rank generated molecules by similarity to multimodal inputs."""
        if not molecules:
            return []
        
        scored_molecules = []
        
        for mol in molecules:
            score = 0.0
            
            # Text similarity (based on predicted notes)
            if text:
                text_score = self._calculate_text_similarity(mol, text)
                score += text_score
            
            # Image similarity (based on color-scent mapping)
            if image:
                image_score = self._calculate_image_similarity(mol, image)
                score += image_score
            
            # Reference similarity (structural)
            if reference_smiles:
                ref_score = self._calculate_reference_similarity(mol, reference_smiles)
                score += ref_score
            
            scored_molecules.append((mol, score))
        
        # Sort by score (descending)
        scored_molecules.sort(key=lambda x: x[1], reverse=True)
        
        return [mol for mol, score in scored_molecules]
    
    def _calculate_text_similarity(self, molecule: Molecule, text: str) -> float:
        """Calculate similarity between molecule and text description."""
        text_lower = text.lower()
        mol_notes = molecule.fragrance_notes.top + molecule.fragrance_notes.middle + molecule.fragrance_notes.base
        
        matches = sum(1 for note in mol_notes if note in text_lower)
        return matches / max(len(mol_notes), 1)
    
    def _calculate_image_similarity(self, molecule: Molecule, image: Image.Image) -> float:
        """Calculate similarity between molecule and image."""
        image_influence = self._process_image(image)
        suggested_scents = image_influence.get('suggested_scents', [])
        mol_notes = molecule.fragrance_notes.top + molecule.fragrance_notes.middle + molecule.fragrance_notes.base
        
        matches = sum(1 for scent in suggested_scents if scent in mol_notes)
        return matches / max(len(suggested_scents), 1)
    
    def _calculate_reference_similarity(self, molecule: Molecule, reference_smiles: str) -> float:
        """Calculate similarity between molecule and reference."""
        return self._structural_similarity(molecule.smiles, reference_smiles)