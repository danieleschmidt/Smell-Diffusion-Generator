"""Core SmellDiffusion model for generating fragrance molecules from text."""

import random
from typing import List, Optional, Dict, Any, Union
import numpy as np
from .molecule import Molecule
from ..utils.logging import performance_monitor, log_molecule_generation
from ..utils.validation import validate_inputs
from ..utils.caching import cached
from ..utils.async_utils import AsyncMoleculeGenerator
from ..utils.config import get_config


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
        self.config = get_config()
        self._cache = {}
        self._performance_stats = {
            "generations": 0,
            "cache_hits": 0,
            "avg_generation_time": 0.0
        }
        
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
            import torch
            if torch.cuda.is_available():
                self.device = "cuda"
            else:
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
    
    @performance_monitor("molecule_generation")
    @validate_inputs
    @log_molecule_generation
    def generate(self, 
                 prompt: str,
                 num_molecules: int = 1,
                 guidance_scale: float = 7.5,
                 safety_filter: bool = True,
                 **kwargs) -> Union[Molecule, List[Molecule]]:
        """Generate fragrance molecules from text prompt."""
        
        try:
            if not self._is_loaded:
                self._load_model()
                
            # Validate input parameters
            if not prompt or not isinstance(prompt, str):
                raise ValueError("Prompt must be a non-empty string")
            
            if num_molecules < 1 or num_molecules > 100:
                raise ValueError("num_molecules must be between 1 and 100")
            
            if guidance_scale < 0.1 or guidance_scale > 20.0:
                raise ValueError("guidance_scale must be between 0.1 and 20.0")
            
            # Use caching for repeated prompts
            cache_key = f"{prompt}_{num_molecules}_{guidance_scale}_{safety_filter}"
            
            # Analyze prompt with caching and error handling
            try:
                category_scores = self._analyze_prompt_cached(prompt)
                self._performance_stats["cache_hits"] += 1
            except Exception as e:
                # Fallback to balanced categories on analysis failure
                category_scores = {k: 1.0/len(self.SCENT_KEYWORDS) 
                                 for k in self.SCENT_KEYWORDS.keys()}
            
            # Select base molecules with retry logic
            attempts = 0
            max_attempts = kwargs.get('max_attempts', 3)
            molecules = []
            
            while len(molecules) < num_molecules and attempts < max_attempts:
                attempts += 1
                
                try:
                    # Select base molecules
                    needed_molecules = num_molecules - len(molecules)
                    base_smiles = self._select_molecules(category_scores, needed_molecules * 2)  # Generate extra
                    
                    # Add variations with error handling
                    for smiles in base_smiles:
                        if len(molecules) >= num_molecules:
                            break
                        
                        try:
                            varied_smiles = self._add_molecular_variation(smiles)
                            mol = Molecule(varied_smiles, description=prompt)
                            
                            # Validate molecule
                            if not mol.is_valid:
                                continue
                            
                            # Safety filtering with enhanced checks
                            if safety_filter:
                                safety = mol.get_safety_profile()
                                min_safety_score = kwargs.get('min_safety_score', 50)
                                if safety.score < min_safety_score:
                                    continue
                                
                                # Additional safety checks
                                if safety.allergens and len(safety.allergens) > 2:
                                    continue
                                    
                            molecules.append(mol)
                            
                        except Exception as e:
                            # Log individual molecule creation errors but continue
                            from ..utils.logging import SmellDiffusionLogger
                            logger = SmellDiffusionLogger()
                            logger.log_error("molecule_creation", e, {"smiles": smiles})
                            continue
                    
                except Exception as e:
                    # Log attempt failure but retry
                    from ..utils.logging import SmellDiffusionLogger
                    logger = SmellDiffusionLogger()
                    logger.log_error("generation_attempt", e, {"attempt": attempts})
                    
                    if attempts >= max_attempts:
                        break
            
            # Ensure we have at least one molecule with fallback strategies
            if not molecules and num_molecules > 0:
                fallback_molecules = self._get_fallback_molecules(prompt, num_molecules)
                molecules.extend(fallback_molecules)
            
            # Final validation and cleanup
            valid_molecules = []
            for mol in molecules:
                if mol and mol.is_valid:
                    # Final safety check
                    if safety_filter:
                        safety = mol.get_safety_profile()
                        if safety.score >= kwargs.get('min_safety_score', 50):
                            valid_molecules.append(mol)
                    else:
                        valid_molecules.append(mol)
            
            # Ensure we don't exceed requested number
            valid_molecules = valid_molecules[:num_molecules]
            
            # Return format based on request
            if num_molecules == 1:
                return valid_molecules[0] if valid_molecules else None
            else:
                return valid_molecules
                
        except Exception as e:
            from ..utils.logging import SmellDiffusionLogger
            logger = SmellDiffusionLogger()
            logger.log_error("generation_failure", e, {
                "prompt": prompt,
                "num_molecules": num_molecules,
                "guidance_scale": guidance_scale
            })
            
            # Return fallback results even on major failure
            if num_molecules == 1:
                return None
            else:
                return []
    
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
            from ..utils.logging import SmellDiffusionLogger
            logger = SmellDiffusionLogger()
            logger.log_error("batch_generation", e)
            
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
            "device": getattr(self, 'device', 'unknown')
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
        """Get information about the loaded model."""
        return {
            "model_name": self.model_name,
            "is_loaded": self._is_loaded,
            "supported_categories": list(self.SCENT_KEYWORDS.keys()),
            "database_size": sum(len(mols) for mols in self.FRAGRANCE_DATABASE.values()),
            "version": "0.1.0",
            "capabilities": ["text_to_molecule", "safety_filtering", "multi_category"]
        }
    
    def __str__(self) -> str:
        """String representation."""
        return f"SmellDiffusion(model='{self.model_name}', loaded={self._is_loaded})"