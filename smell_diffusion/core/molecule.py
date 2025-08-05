"""Molecular representation and utilities."""

from typing import Optional, List, Dict, Any
from dataclasses import dataclass

# Optional imports with fallbacks for demo purposes
try:
    import numpy as np
except ImportError:
    np = None

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors
    from rdkit.Chem.Draw import rdMolDraw2D
    RDKIT_AVAILABLE = True
except ImportError:
    # Mock RDKit for demo purposes
    class MockMol:
        def __init__(self, smiles): self.smiles = smiles
    
    class MockChem:
        Mol = MockMol  # Add Mol class attribute
        @staticmethod
        def MolFromSmiles(smiles): return MockMol(smiles) if smiles else None
        @staticmethod
        def MolToSmiles(mol): return mol.smiles if mol else ""
    
    class MockDescriptors:
        @staticmethod
        def MolWt(mol): return len(mol.smiles) * 12 + 16 if mol else 0  # Simple mock
        @staticmethod
        def MolLogP(mol): return len(mol.smiles) * 0.1 if mol else 0  # Simple mock
    
    Chem = MockChem()
    Descriptors = MockDescriptors()
    RDKIT_AVAILABLE = False

import base64
from io import BytesIO


@dataclass
class FragranceNotes:
    """Fragrance note classification."""
    top: List[str]
    middle: List[str] 
    base: List[str]
    intensity: float
    longevity: str


@dataclass
class SafetyProfile:
    """Basic safety evaluation results."""
    score: float
    ifra_compliant: bool
    allergens: List[str]
    warnings: List[str]


class Molecule:
    """Represents a generated fragrance molecule."""
    
    def __init__(self, smiles: str, description: Optional[str] = None):
        self.smiles = smiles
        self.description = description
        self._mol = None
        self._properties = {}
        self._fragrance_notes = None
        self._safety_profile = None
        
    @property
    def mol(self) -> Optional[Chem.Mol]:
        """Get RDKit molecule object."""
        if self._mol is None and self.smiles:
            self._mol = Chem.MolFromSmiles(self.smiles)
        return self._mol
    
    @property
    def is_valid(self) -> bool:
        """Check if molecule is chemically valid."""
        return self.mol is not None
    
    @property
    def molecular_weight(self) -> float:
        """Calculate molecular weight."""
        if not self.is_valid:
            return 0.0
        return Descriptors.MolWt(self.mol)
    
    @property
    def logp(self) -> float:
        """Calculate LogP (lipophilicity)."""
        if not self.is_valid:
            return 0.0
        return Descriptors.MolLogP(self.mol)
    
    @property
    def fragrance_notes(self) -> FragranceNotes:
        """Get predicted fragrance notes."""
        if self._fragrance_notes is None:
            self._fragrance_notes = self._predict_fragrance_notes()
        return self._fragrance_notes
    
    @property
    def intensity(self) -> float:
        """Predicted scent intensity (0-10)."""
        return self.fragrance_notes.intensity
    
    @property
    def longevity(self) -> str:
        """Predicted longevity category."""
        return self.fragrance_notes.longevity
    
    def _predict_fragrance_notes(self) -> FragranceNotes:
        """Simple fragrance note prediction based on molecular properties."""
        if not self.is_valid:
            return FragranceNotes([], [], [], 0.0, "very_short")
            
        mw = self.molecular_weight
        logp = self.logp
        
        # Simple heuristic classification
        top_notes = []
        middle_notes = []
        base_notes = []
        
        # Light molecules (MW < 150) tend to be top notes
        if mw < 150:
            if logp < 2:
                top_notes = ["citrus", "fresh"]
            else:
                top_notes = ["herbal", "green"]
                
        # Medium molecules (150-250)
        elif mw < 250:
            if logp < 3:
                middle_notes = ["floral", "fruity"]
            else:
                middle_notes = ["spicy", "woody"]
                
        # Heavy molecules (MW > 250) tend to be base notes
        else:
            if logp > 4:
                base_notes = ["woody", "musky"]
            else:
                base_notes = ["amber", "vanilla"]
        
        # Intensity based on volatility (inverse of MW)
        intensity = max(1.0, min(10.0, 300 / mw))
        
        # Longevity based on molecular weight and lipophilicity
        if mw > 250 and logp > 4:
            longevity = "very_long"
        elif mw > 200 and logp > 3:
            longevity = "long"
        elif mw > 150:
            longevity = "medium"
        elif mw > 100:
            longevity = "short"
        else:
            longevity = "very_short"
            
        return FragranceNotes(
            top=top_notes,
            middle=middle_notes,
            base=base_notes,
            intensity=intensity,
            longevity=longevity
        )
    
    def get_safety_profile(self) -> SafetyProfile:
        """Get basic safety evaluation."""
        if self._safety_profile is None:
            self._safety_profile = self._evaluate_safety()
        return self._safety_profile
    
    def _evaluate_safety(self) -> SafetyProfile:
        """Basic safety evaluation using molecular descriptors."""
        if not self.is_valid:
            return SafetyProfile(0.0, False, [], ["Invalid molecule"])
            
        warnings = []
        allergens = []
        
        # Basic safety checks
        mw = self.molecular_weight
        logp = self.logp
        
        # Check for common allergen patterns
        if "C1=CC=C(C=C1)C=O" in self.smiles:  # Benzaldehyde-like
            allergens.append("benzaldehyde")
            
        if "CC1=CC=C(C=C1)C(C)(C)C" in self.smiles:  # Tert-butyl patterns
            allergens.append("tert-butyl compounds")
        
        # Molecular weight warnings
        if mw > 500:
            warnings.append("High molecular weight may affect absorption")
            
        if mw < 50:
            warnings.append("Very low molecular weight - high volatility")
        
        # LogP warnings  
        if logp > 5:
            warnings.append("High lipophilicity - potential skin accumulation")
            
        if logp < -2:
            warnings.append("Very hydrophilic - may not penetrate skin")
        
        # Simple scoring (inverse of warnings)
        base_score = 100.0
        score = base_score - (len(warnings) * 15) - (len(allergens) * 25)
        score = max(0.0, min(100.0, score))
        
        # IFRA compliance (simplified)
        ifra_compliant = len(allergens) == 0 and score > 70
        
        return SafetyProfile(
            score=score,
            ifra_compliant=ifra_compliant,
            allergens=allergens,
            warnings=warnings
        )
    
    def to_svg(self, width: int = 300, height: int = 300) -> str:
        """Generate SVG representation of the molecule."""
        if not self.is_valid:
            return ""
            
        drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
        drawer.DrawMolecule(self.mol)
        drawer.FinishDrawing()
        return drawer.GetDrawingText()
    
    def visualize_3d(self) -> str:
        """Generate basic 3D visualization info."""
        if not self.is_valid:
            return "Invalid molecule - cannot visualize"
            
        return f"3D visualization placeholder for {self.smiles}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert molecule to dictionary representation."""
        return {
            "smiles": self.smiles,
            "description": self.description,
            "is_valid": self.is_valid,
            "molecular_weight": self.molecular_weight,
            "logp": self.logp,
            "fragrance_notes": {
                "top": self.fragrance_notes.top,
                "middle": self.fragrance_notes.middle,
                "base": self.fragrance_notes.base,
                "intensity": self.fragrance_notes.intensity,
                "longevity": self.fragrance_notes.longevity,
            },
            "safety_profile": {
                "score": self.get_safety_profile().score,
                "ifra_compliant": self.get_safety_profile().ifra_compliant,
                "allergens": self.get_safety_profile().allergens,
                "warnings": self.get_safety_profile().warnings,
            }
        }
    
    def __str__(self) -> str:
        """String representation."""
        return f"Molecule(smiles='{self.smiles}', mw={self.molecular_weight:.1f})"
    
    def __repr__(self) -> str:
        """Detailed representation."""
        return self.__str__()