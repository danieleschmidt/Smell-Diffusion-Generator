"""Core SmellDiffusion model for generating fragrance molecules from text."""

import random
from typing import List, Optional, Dict, Any, Union
import numpy as np
from .molecule import Molecule


class SmellDiffusion:
    """Main class for text-to-molecule fragrance generation."""
    
    # Pre-defined fragrance molecule database for demonstration
    FRAGRANCE_DATABASE = {
        "citrus": [
            "CC(C)=CCCC(C)=CCO",  # Geraniol
            "CC1=CCC(CC1)C(C)(C)O",  # Linalool
            "CC(C)CCCC(C)CCO",  # Citronellol
        ],
        "floral": [
            "CC1=CC=C(C=C1)C=O",  # p-Tolualdehyde
            "COC1=CC=C(C=C1)C=O",  # Anisaldehyde
            "CC(C)(C)C1=CC=C(C=C1)C=O",  # Lilial
        ],
        "woody": [
            "CC12CCC(CC1=CCC2=O)C(C)(C)C",  # Cedrol-like
            "CC1CCC2C(C1)C(C(C2)C)(C)C",  # Sandalwood-like
            "CC(C)C1CCC(C)CC1",  # Woody terpene
        ],
        "vanilla": [
            "COC1=C(C=CC(=C1)C=O)O",  # Vanillin
            "COC1=C(C=CC(=C1)C=O)OC",  # Ethyl vanillin
            "CC(=O)C1=CC=C(C=C1)OC",  # Acetovanillone
        ],
        "musky": [
            "CC1CCCC2C1CCCC2(C)C",  # Musk-like macrocycle
            "CCCCCCCCCCCCCC(=O)C",  # Long chain ketone
            "CC(C)(C)C1=CC=C(C=C1)C(C)(C)C",  # Synthetic musk
        ],
        "fresh": [
            "C1=CC=C(C=C1)C(=O)C=C",  # Cinnamaldehyde-like
            "CC(C)=CCC=C(C)C",  # Myrcene
            "CC(C)C1CCC(C)=CC1",  # p-Cymene
        ]
    }
    
    SCENT_KEYWORDS = {
        "citrus": ["lemon", "orange", "lime", "bergamot", "grapefruit", "citrus"],
        "floral": ["rose", "jasmine", "lily", "lavender", "violet", "floral", "flower"],
        "woody": ["sandalwood", "cedar", "oak", "wood", "woody", "forest"],
        "vanilla": ["vanilla", "sweet", "gourmand", "cream", "dessert"],
        "musky": ["musk", "animal", "skin", "sensual", "warm"],
        "fresh": ["fresh", "clean", "aquatic", "sea", "water", "cucumber", "mint"]
    }
    
    def __init__(self, model_name: str = "smell-diffusion-base-v1"):
        """Initialize the SmellDiffusion model."""
        self.model_name = model_name
        self._is_loaded = False
        
    @classmethod
    def from_pretrained(cls, model_name: str) -> "SmellDiffusion":
        """Load a pre-trained model."""
        instance = cls(model_name)
        instance._load_model()
        return instance
        
    def _load_model(self) -> None:
        """Simulate model loading."""
        print(f"Loading model: {self.model_name}")
        self._is_loaded = True
        
    def _analyze_prompt(self, prompt: str) -> Dict[str, float]:
        """Analyze text prompt to identify scent categories."""
        prompt_lower = prompt.lower()
        category_scores = {}
        
        for category, keywords in self.SCENT_KEYWORDS.items():
            score = 0.0
            for keyword in keywords:
                if keyword in prompt_lower:
                    score += 1.0
            category_scores[category] = score / len(keywords)
            
        # Normalize scores
        total_score = sum(category_scores.values())
        if total_score > 0:
            category_scores = {k: v/total_score for k, v in category_scores.items()}
        else:
            # Default to balanced mix if no keywords found
            category_scores = {k: 1.0/len(self.SCENT_KEYWORDS) 
                             for k in self.SCENT_KEYWORDS.keys()}
            
        return category_scores
    
    def _select_molecules(self, category_scores: Dict[str, float], 
                         num_molecules: int) -> List[str]:
        """Select molecules based on category scores."""
        selected_smiles = []
        
        for _ in range(num_molecules):
            # Weighted random selection
            categories = list(category_scores.keys())
            weights = list(category_scores.values())
            
            if sum(weights) == 0:
                weights = [1.0] * len(weights)
                
            selected_category = np.random.choice(categories, p=weights)
            
            # Select random molecule from category
            available_molecules = self.FRAGRANCE_DATABASE[selected_category]
            selected_smiles.append(random.choice(available_molecules))
            
        return selected_smiles
    
    def _add_molecular_variation(self, base_smiles: str) -> str:
        """Add slight variations to base molecules."""
        # Simple variation: occasionally add or modify functional groups
        variations = [
            base_smiles,  # Original
            base_smiles,  # Keep original more likely
            base_smiles,  # Keep original more likely
        ]
        
        # Add some simple variations (placeholder for real molecular editing)
        if "C=O" in base_smiles and random.random() < 0.3:
            # Occasionally convert aldehyde to alcohol
            variations.append(base_smiles.replace("C=O", "CO"))
            
        if "CC" in base_smiles and random.random() < 0.2:
            # Occasionally add methyl group
            variations.append(base_smiles.replace("CC", "C(C)C", 1))
            
        return random.choice(variations)
    
    def generate(self, 
                 prompt: str,
                 num_molecules: int = 1,
                 guidance_scale: float = 7.5,
                 safety_filter: bool = True,
                 **kwargs) -> Union[Molecule, List[Molecule]]:
        """Generate fragrance molecules from text prompt."""
        
        if not self._is_loaded:
            self._load_model()
            
        # Analyze prompt
        category_scores = self._analyze_prompt(prompt)
        
        # Select base molecules
        base_smiles = self._select_molecules(category_scores, num_molecules)
        
        # Add variations
        varied_smiles = [self._add_molecular_variation(smiles) for smiles in base_smiles]
        
        # Create molecule objects
        molecules = []
        for smiles in varied_smiles:
            mol = Molecule(smiles, description=prompt)
            
            # Safety filtering
            if safety_filter:
                safety = mol.get_safety_profile()
                if safety.score < 50:  # Filter out unsafe molecules
                    continue
                    
            molecules.append(mol)
        
        # Ensure we have at least one molecule
        if not molecules and num_molecules > 0:
            # Fallback to safest molecule
            safe_smiles = "CC(C)=CCCC(C)=CCO"  # Geraniol - generally safe
            molecules.append(Molecule(safe_smiles, description=prompt))
        
        # Return single molecule or list based on request
        if num_molecules == 1:
            return molecules[0] if molecules else None
        else:
            return molecules[:num_molecules]
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        return {
            "model_name": self.model_name,
            "is_loaded": self._is_loaded,
            "supported_categories": list(self.SCENT_KEYWORDS.keys()),
            "database_size": sum(len(mols) for mols in self.FRAGRANCE_DATABASE.values())
        }
    
    def __str__(self) -> str:
        """String representation."""
        return f"SmellDiffusion(model='{self.model_name}', loaded={self._is_loaded})"