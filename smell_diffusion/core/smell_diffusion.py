"""Core SmellDiffusion model for generating fragrance molecules from text."""

import random
import logging
import time
from typing import List, Optional, Dict, Any, Union
import traceback
from contextlib import contextmanager

try:
    import numpy as np
except ImportError:
    # Fallback for environments without NumPy
    class MockNumPy:
        def random(self):
            return random
        @staticmethod
        def array(x):
            return x
        @staticmethod
        def choice(items, p=None):
            if p is None:
                return random.choice(items)
            # Simple weighted choice implementation
            total = sum(p)
            r = random.random() * total
            cumulative = 0
            for i, weight in enumerate(p):
                cumulative += weight
                if r <= cumulative:
                    return items[i]
            return items[-1]
    np = MockNumPy()

from .molecule import Molecule
from ..utils.logging import SmellDiffusionLogger, performance_monitor
from ..utils.validation import validate_inputs, ValidationError


class SmellDiffusion:
    """Main class for text-to-molecule fragrance generation.
    
    This class implements robust error handling, comprehensive validation,
    and performance monitoring for production use.
    """
    
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
        """Initialize the SmellDiffusion model with robust error handling."""
        self.model_name = model_name
        self._is_loaded = False
        self._generation_count = 0
        self._error_count = 0
        self.logger = SmellDiffusionLogger(f"smell_diffusion_{model_name}")
        
        # Validate model name
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValidationError("Model name must be a non-empty string")
        
        self.logger.logger.info(f"Initialized SmellDiffusion with model: {model_name}")
        
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
    
    @validate_inputs
    @performance_monitor("molecule_generation")
    def generate(self, 
                 prompt: str,
                 num_molecules: int = 1,
                 guidance_scale: float = 7.5,
                 safety_filter: bool = True,
                 **kwargs) -> Union[Molecule, List[Molecule]]:
        """Generate fragrance molecules from text prompt with comprehensive error handling."""
        
        with self._error_handling_context("generation", 
                                        prompt=prompt, 
                                        num_molecules=num_molecules,
                                        safety_filter=safety_filter):
            
            # Increment generation counter
            self._generation_count += 1
            
            # Validate inputs
            if not isinstance(prompt, str):
                raise ValidationError(f"Prompt must be a string, got {type(prompt)}")
            
            if not prompt.strip():
                raise ValidationError("Prompt cannot be empty")
            
            if not isinstance(num_molecules, int) or num_molecules <= 0:
                raise ValidationError(f"num_molecules must be a positive integer, got {num_molecules}")
            
            if num_molecules > 100:
                raise ValidationError(f"num_molecules cannot exceed 100, got {num_molecules}")
            
            if not isinstance(guidance_scale, (int, float)) or guidance_scale < 0.1 or guidance_scale > 20.0:
                raise ValidationError(f"guidance_scale must be between 0.1 and 20.0, got {guidance_scale}")
            
            # Ensure model is loaded
            if not self._is_loaded:
                self.logger.logger.info("Model not loaded, loading now...")
                self._load_model()
            
            # Log generation request
            request_id = self.logger.log_generation_request(
                prompt, num_molecules, safety_filter, 
                guidance_scale=guidance_scale, **kwargs
            )
            
            start_time = time.time()
            
            try:
                # Analyze prompt with error handling
                category_scores = self._analyze_prompt(prompt)
                self.logger.logger.debug(f"Category analysis complete for request {request_id}")
                
                # Select base molecules with error handling
                base_smiles = self._select_molecules(category_scores, num_molecules)
                self.logger.logger.debug(f"Selected {len(base_smiles)} base molecules for request {request_id}")
                
                # Add variations with error handling
                varied_smiles = []
                for i, smiles in enumerate(base_smiles):
                    try:
                        varied = self._add_molecular_variation(smiles)
                        varied_smiles.append(varied)
                    except Exception as e:
                        self.logger.log_error(f"molecular_variation_{i}", e, {"smiles": smiles})
                        # Use original if variation fails
                        varied_smiles.append(smiles)
                
                # Create molecule objects with error handling
                molecules = []
                failed_molecules = 0
                
                for i, smiles in enumerate(varied_smiles):
                    try:
                        mol = Molecule(smiles, description=prompt)
                        
                        # Validate molecule
                        if not mol.is_valid:
                            self.logger.logger.warning(f"Invalid molecule generated: {smiles}")
                            failed_molecules += 1
                            continue
                        
                        # Safety filtering with error handling
                        if safety_filter:
                            try:
                                safety = mol.get_safety_profile()
                                if safety.score < 50:  # Filter out unsafe molecules
                                    self.logger.logger.debug(f"Filtered unsafe molecule: {smiles} (score: {safety.score})")
                                    continue
                            except Exception as e:
                                self.logger.log_error(f"safety_evaluation_{i}", e, {"smiles": smiles})
                                # If safety evaluation fails, be conservative and skip
                                if safety_filter:
                                    continue
                        
                        molecules.append(mol)
                        
                    except Exception as e:
                        self.logger.log_error(f"molecule_creation_{i}", e, {"smiles": smiles})
                        failed_molecules += 1
                        continue
                
                # Log generation statistics
                generation_time = time.time() - start_time
                self.logger.log_generation_result(request_id, molecules, generation_time)
                
                if failed_molecules > 0:
                    self.logger.logger.warning(f"Failed to create {failed_molecules} molecules in request {request_id}")
                
                # Ensure we have at least one molecule
                if not molecules and num_molecules > 0:
                    self.logger.logger.warning(f"No valid molecules generated, using fallback for request {request_id}")
                    try:
                        # Fallback to safest molecule
                        safe_smiles = "CC(C)=CCCC(C)=CCO"  # Geraniol - generally safe
                        fallback_mol = Molecule(safe_smiles, description=f"Fallback for: {prompt}")
                        molecules.append(fallback_mol)
                    except Exception as e:
                        self.logger.log_error("fallback_molecule_creation", e)
                        raise RuntimeError("Failed to generate any molecules, including fallback")
                
                # Return appropriate format
                if num_molecules == 1:
                    result = molecules[0] if molecules else None
                    if result is None:
                        raise RuntimeError("Failed to generate any molecules")
                    return result
                else:
                    return molecules[:num_molecules]
                    
            except ValidationError:
                # Re-raise validation errors
                raise
            except Exception as e:
                generation_time = time.time() - start_time
                self.logger.log_error("generation_core", e, {
                    "request_id": request_id,
                    "generation_time": generation_time,
                    "prompt": prompt
                })
                raise
    
    
    @contextmanager
    def _error_handling_context(self, operation: str, **context):
        """Context manager for consistent error handling."""
        start_time = time.time()
        try:
            yield
        except ValidationError:
            # Re-raise validation errors as-is
            raise
        except Exception as e:
            self._error_count += 1
            duration = time.time() - start_time
            self.logger.log_error(operation, e, {
                **context,
                "duration": duration,
                "error_count": self._error_count
            })
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information with error handling."""
        try:
            return {
                "model_name": self.model_name,
                "is_loaded": self._is_loaded,
                "generation_count": self._generation_count,
                "error_count": self._error_count,
                "supported_categories": list(self.SCENT_KEYWORDS.keys()),
                "database_size": {cat: len(mols) for cat, mols in self.FRAGRANCE_DATABASE.items()},
                "total_molecules": sum(len(mols) for mols in self.FRAGRANCE_DATABASE.values())
            }
        except Exception as e:
            self.logger.log_error("get_model_info", e)
            return {"error": str(e)}
    
    def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check."""
        try:
            health_status = {
                "status": "healthy",
                "timestamp": time.time(),
                "model_loaded": self._is_loaded,
                "generation_count": self._generation_count,
                "error_count": self._error_count,
                "error_rate": self._error_count / max(self._generation_count, 1),
                "checks": {}
            }
            
            # Check model availability
            health_status["checks"]["model_available"] = self._validate_model_availability()
            
            # Check database integrity
            health_status["checks"]["database_integrity"] = self._check_database_integrity()
            
            # Check memory usage (simplified)
            health_status["checks"]["memory_ok"] = True  # Placeholder
            
            # Overall health determination
            if not all(health_status["checks"].values()):
                health_status["status"] = "degraded"
            
            if health_status["error_rate"] > 0.1:  # 10% error rate threshold
                health_status["status"] = "unhealthy"
            
            return health_status
            
        except Exception as e:
            self.logger.log_error("health_check", e)
            return {
                "status": "error", 
                "error": str(e),
                "timestamp": time.time()
            }
    
    def _validate_model_availability(self) -> bool:
        """Validate that the model is available and functional."""
        try:
            # In a real implementation, this would check:
            # - Model file existence
            # - Model integrity
            # - Compatible format
            # - Hardware requirements
            
            # For demo, just check that we have the fragrance database
            if not self.FRAGRANCE_DATABASE or not self.SCENT_KEYWORDS:
                return False
            
            # Validate database structure
            for category, molecules in self.FRAGRANCE_DATABASE.items():
                if not isinstance(molecules, list) or len(molecules) == 0:
                    return False
                
                for smiles in molecules:
                    if not isinstance(smiles, str) or len(smiles) < 3:
                        return False
            
            return True
            
        except Exception as e:
            self.logger.log_error("model_validation", e)
            return False
    
    def _check_database_integrity(self) -> bool:
        """Check integrity of fragrance database."""
        try:
            if not self.FRAGRANCE_DATABASE:
                return False
            
            for category, molecules in self.FRAGRANCE_DATABASE.items():
                if not isinstance(molecules, list) or len(molecules) == 0:
                    return False
                
                for smiles in molecules:
                    if not isinstance(smiles, str) or len(smiles.strip()) < 3:
                        return False
            
            return True
        except:
            return False

    def __str__(self) -> str:
        """String representation with comprehensive info."""
        return (f"SmellDiffusion(model='{self.model_name}', "
                f"loaded={self._is_loaded}, "
                f"generations={self._generation_count}, "
                f"errors={self._error_count})")
    
    def __repr__(self) -> str:
        """Detailed representation for debugging."""
        return (f"SmellDiffusion(model_name='{self.model_name}', "
                f"_is_loaded={self._is_loaded}, "
                f"_generation_count={self._generation_count}, "
                f"_error_count={self._error_count})")