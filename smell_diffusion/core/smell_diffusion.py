```python
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
from ..utils.logging import SmellDiffusionLogger, performance_monitor, log_molecule_generation
from ..utils.validation import validate_inputs, ValidationError
from ..utils.caching import cached
from ..utils.async_utils import AsyncMoleculeGenerator
from ..utils.config import get_config


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
        self.config = get_config()
        self.logger = SmellDiffusionLogger(f"smell_diffusion_{model_name}")
        self._cache = {}
        self._performance_stats = {
            "generations": 0,
            "cache_hits": 0,
            "avg_generation_time": 0.0
        }
        
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
        """Simulate model loading with optimization."""
        print(f"Loading model: {self.model_name}")
        
        # Optimize based on configuration
        if self.config.model.device == "auto":
            # Auto-detect best device
            try:
                import torch
                if torch.cuda.is_available():
                    self.device = "cuda"
                else:
                    self.device = "cpu"
            except ImportError:
                self.device = "cpu"
        else:
            self.device = self.config.model.device
        
        # Pre-compile molecular patterns for faster lookup
        self._precompile_patterns()
        
        self._is_loaded = True
        print(f"Model loaded on {self.device}")
        
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
                
            selected_category = np.choice(categories, p=weights)
            
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
    
    def _precompile_patterns(self) -> None:
        """Pre-compile molecular patterns for faster matching."""
        self._compiled_patterns = {}
        
        # Pre-compile common substructure patterns
        common_patterns = {
            "aromatic": r"c1ccccc1|C1=CC=CC=C1",
            "aldehyde": r"C=O",
            "alcohol": r"CO(?!C=O)",
            "ester": r"C\(=O\)O",
            "ether": r"COC",
            "ketone": r"C\(=O\)C"
        }
        
        import re
        for name, pattern in common_patterns.items():
            try:
                self._compiled_patterns[name] = re.compile(pattern)
            except:
                pass
    
    @cached(ttl=3600, persist=True)
    def _analyze_prompt_cached(self, prompt: str) -> Dict[str, float]:
        """Cached version of prompt analysis."""
        return self._analyze_prompt(prompt)
    
    @validate_inputs
    @performance_monitor("molecule_generation")
    @log_molecule_generation
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
            self._performance_stats["generations"] += 1
            
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
                # Use caching for repeated prompts
                cache_key = f"{prompt}_{num_molecules}_{guidance_scale}_{safety_filter}"
                
                # Analyze prompt with caching and error handling
                try:
                    category_scores = self._analyze_prompt_cached(prompt)
                    self._performance_stats["cache_hits"] += 1
                    self.logger.logger.debug(f"Category analysis complete for request {request_id}")
                except Exception as e:
                    self.logger.log_error("prompt_analysis", e, {"prompt": prompt})
                    # Fallback to balanced categories on analysis failure
                    category_scores = {k: 1.0/len(self.SCENT_KEYWORDS) 
                                     for k in self.SCENT_KEYWORDS.keys()}
                
                # Select base molecules with retry logic
                attempts = 0
                max_attempts = kwargs.get('max_attempts', 3)
                molecules = []
                failed_molecules = 0
                
                while len(molecules) < num_molecules and attempts < max_attempts:
                    attempts += 1
                    
                    try:
                        # Select base molecules
                        needed_molecules = num_molecules - len(molecules)
                        base_smiles = self._select_molecules(category_scores, needed_molecules * 2)  # Generate extra
                        self.logger.logger.debug(f"Selected {len(base_smiles)} base molecules for request {request_id}")
                        
                        # Add variations with error handling
                        for i, smiles in enumerate(base_smiles):
                            if len(molecules) >= num_molecules:
                                break
                            
                            try:
                                varied_smiles = self._add_molecular_variation(smiles)
                                mol = Molecule(varied_smiles, description=prompt)
                                
                                # Validate molecule
                                if not mol.is_valid:
                                    self.logger.logger.warning(f"Invalid molecule generated: {varied_smiles}")
                                    failed_molecules += 1
                                    continue
                                
                                # Safety filtering with enhanced checks
                                if safety_filter:
                                    try:
                                        safety = mol.get_safety_profile()
                                        min_safety_score = kwargs.get('min_safety_score', 50)
                                        if safety.score < min_safety_score:
                                            self.logger.logger.debug(f"Filtered unsafe molecule: {varied_smiles} (score: {safety.score})")
                                            continue
                                        
                                        # Additional safety checks
                                        if safety.allergens and len(safety.allergens) > 2:
                                            continue
                                    except Exception as e:
                                        self.logger.log_error(f"safety_evaluation_{i}", e, {"smiles": varied_smiles})
                                        # If safety evaluation fails, be conservative and skip
                                        if safety_filter:
                                            continue
                                
                                molecules.append(mol)
                                
                            except Exception as e:
                                self.logger.log_error(f"molecule_creation_{i}", e, {"smiles": smiles})
                                failed_molecules += 1
                                continue
                        
                    except Exception as e:
                        self.logger.log_error("generation_attempt", e, {"attempt": attempts})
                        
                        if attempts >= max_attempts:
                            break
                
                # Log generation statistics
                generation_time = time.time() - start_time
                self.logger.log_generation_result(request_id, molecules, generation_time)
                
                if failed_molecules > 0:
                    self.logger.logger.warning(f"Failed to create {failed_molecules} molecules in request {request_id}")
                
                # Ensure we have at least one molecule with fallback strategies
                if not molecules and num_molecules > 0:
                    self.logger.logger.warning(f"No valid molecules generated, using fallback for request {request_id}")
                    fallback_molecules = self._get_fallback_molecules(prompt, num_molecules)
                    molecules.extend(fallback_molecules)
                
                # Final validation and cleanup
                valid_molecules = []
                for mol in molecules:
                    if mol and mol.is_valid:
                        valid_molecules.append(mol)
                
                # Ensure we don't exceed requested number
                valid_molecules = valid_molecules[:num_molecules]
                
                # Update performance stats
                self._update_performance_stats(generation_time)
                
                # Return appropriate format
                if num_molecules == 1:
                    result = valid_molecules[0] if valid_molecules else None
                    if result is None:
                        raise RuntimeError("Failed to generate any molecules")
                    return result
                else:
                    return valid_molecules
                    
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
    
    def _get_fallback_molecules(self, prompt: str, num_molecules: int) -> List[Molecule]:
        """Get fallback molecules when generation fails."""
        fallback_molecules = []
        
        # Use safest known molecules as fallbacks
        safe_smiles = [
            "CC(C)=CCCC(C)=CCO",      # Geraniol
            "CC(C)CCCC(C)CCO",        # Citronellol  
            "COC1=C(C=CC(=C1)C=O)O",  # Vanillin
            "CC1=CC=C(C=C1)C=O",      # p-Tolualdehyde
            "CC1=CCC(CC1)C(C)(C)O"    # Linalool
        ]
        
        for i, smiles in enumerate(safe_smiles):
            if len(fallback_molecules) >= num_molecules:
                break
            
            try:
                mol = Molecule(smiles, description=f"Fallback for: {prompt}")
                if mol.is_valid:
                    fallback_molecules.append(mol)
            except Exception:
                continue
        
        return fallback_molecules
    
    def _update_performance_stats(self, generation_time: float) -> None:
        """Update performance statistics."""
        # Update average generation time
        current_avg = self._performance_stats["avg_generation_time"]
        total_gens = self._performance_stats["generations"]
        
        if total_gens > 0:
            self._performance_stats["avg_generation_time"] = (
                (current_avg * (total_gens - 1) + generation_time) / total_gens
            )
        else:
            self._performance_stats["avg_generation_time"] = generation_time
    
    def batch_generate(self, prompts: List[str], **kwargs) -> List[List[Molecule]]:
        """Generate molecules for multiple prompts with optimized batching."""
        from ..utils.async_utils import AsyncBatchProcessor
        
        try:
            # Use batch processor for efficient parallel processing
            batch_processor = AsyncBatchProcessor(
                batch_size=kwargs.get('batch_size', 3),
                max_concurrent_batches=kwargs.get('max_concurrent', 2)
            )
            
            def generate_single(prompt: str):
                return self.generate(prompt=prompt, **kwargs)
            
            # Process in batches
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                results = loop.run_until_complete(
                    batch_processor.process_items(prompts, generate_single)
                )
                return results
            finally:
                loop.close()
                
        except Exception as e:
            self.logger.log_error("batch_generation", e)
            
            # Fallback to sequential processing
            results = []
            for prompt in prompts:
                try:
                    result = self.generate(prompt=prompt, **kwargs)
                    results.append([result] if not isinstance(result, list) else result)
                except Exception:
                    results.append([])
            
            return results
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        return {
            **self._performance_stats,
            "model_loaded": self._is_loaded,
            "cache_size": len(self._cache),
            "device": getattr(self, 'device', 'unknown'),
            "error_count": self._error_count,
            "error_rate": self._error_count / max(self._generation_count, 1)
        }
    
    def optimize_for_throughput(self) -> None:
        """Optimize model for high-throughput scenarios."""
        # Pre-warm cache with common patterns
        common_prompts = [
            "fresh citrus", "floral rose", "woody cedar", 
            "vanilla sweet", "musky warm", "aquatic marine"
        ]
        
        for prompt in common_prompts:
            try:
                self._analyze_prompt_cached(prompt)
            except:
                pass
    
    @cached(ttl=1800)  # Cache for 30 minutes
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
                "total_molecules": sum(len(mols) for mols in self.FRAGRANCE_DATABASE.values()),
                "version": "0.1.0",
                "capabilities": ["text_to_molecule", "safety_filtering", "multi_category"],
                "device": getattr(self, 'device', 'unknown')
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
```
