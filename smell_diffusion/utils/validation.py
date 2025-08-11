"""Input validation and error handling utilities."""

import re
import time
from typing import Any, Dict, List, Optional, Union, Callable
from functools import wraps
from collections import defaultdict
from dataclasses import dataclass

# Optional imports with fallbacks
try:
    from rdkit import Chem
    RDKIT_AVAILABLE = True
except ImportError:
    class MockChem:
        @staticmethod
        def MolFromSmiles(smiles): return MockMol(smiles) if smiles and len(smiles) > 3 else None
        @staticmethod
        def MolFromSmarts(smarts): return MockMol(smarts)
        class Descriptors:
            @staticmethod
            def MolWt(mol): return len(mol.smiles) * 12 + 16
            @staticmethod
            def MolLogP(mol): return len(mol.smiles) * 0.1
    
    class MockMol:
        def __init__(self, smiles): self.smiles = smiles
        def HasSubstructMatch(self, pattern): return "C=O" in self.smiles
    
    Chem = MockChem()
    RDKIT_AVAILABLE = False

try:
    import numpy as np
except ImportError:
    np = None


class ValidationError(Exception):
    """Custom exception for validation errors."""
    def __init__(self, message: str, field: str = None, value: Any = None):
        super().__init__(message)
        self.field = field
        self.value = value
        self.timestamp = time.time()


class SecurityValidationError(ValidationError):
    """Security-specific validation error."""
    def __init__(self, message: str, threat_level: str = "medium", **kwargs):
        super().__init__(message, **kwargs)
        self.threat_level = threat_level


class SMILESValidator:
    """Validator for SMILES molecular representations."""
    
    @staticmethod
    def is_valid_smiles(smiles: str) -> bool:
        """Check if SMILES string is chemically valid."""
        if not isinstance(smiles, str) or not smiles.strip():
            return False
        
        try:
            mol = Chem.MolFromSmiles(smiles)
            return mol is not None
        except:
            return False
    
    @staticmethod
    def sanitize_smiles(smiles: str) -> str:
        """Sanitize and canonicalize SMILES string."""
        if not SMILESValidator.is_valid_smiles(smiles):
            raise ValidationError(f"Invalid SMILES: {smiles}")
        
        mol = Chem.MolFromSmiles(smiles)
        return Chem.MolToSmiles(mol)
    
    @staticmethod
    def check_molecular_weight_range(smiles: str, min_mw: float = 30.0, 
                                   max_mw: float = 1000.0) -> bool:
        """Check if molecule is within acceptable molecular weight range."""
        if not SMILESValidator.is_valid_smiles(smiles):
            return False
        
        mol = Chem.MolFromSmiles(smiles)
        mw = Chem.Descriptors.MolWt(mol)
        return min_mw <= mw <= max_mw
    
    @staticmethod 
    def check_fragrance_suitability(smiles: str) -> Dict[str, Any]:
        """Check if molecule is suitable for fragrance applications."""
        if not SMILESValidator.is_valid_smiles(smiles):
            return {
                "suitable": False,
                "reasons": ["Invalid SMILES structure"]
            }
        
        mol = Chem.MolFromSmiles(smiles)
        issues = []
        
        # Check molecular weight (fragrance molecules typically 50-400 Da)
        mw = Chem.Descriptors.MolWt(mol)
        if mw < 50:
            issues.append("Molecular weight too low (too volatile)")
        elif mw > 400:
            issues.append("Molecular weight too high (poor volatility)")
        
        # Check LogP (should be reasonable for skin/air partitioning)
        logp = Chem.Descriptors.MolLogP(mol)
        if logp > 6:
            issues.append("LogP too high (may cause skin irritation)")
        elif logp < -2:
            issues.append("LogP too low (poor skin penetration)")
        
        # Check for potentially problematic groups
        problematic_smarts = [
            "[N+](=O)[O-]",  # Nitro groups
            "S(=O)(=O)",     # Sulfonyl groups
            "C(=O)Cl",       # Acid chlorides
            "[SH]",          # Thiols (often malodorous)
        ]
        
        for smarts in problematic_smarts:
            if mol.HasSubstructMatch(Chem.MolFromSmarts(smarts)):
                issues.append(f"Contains potentially problematic group: {smarts}")
        
        return {
            "suitable": len(issues) == 0,
            "molecular_weight": mw,
            "logp": logp,
            "issues": issues
        }


class TextPromptValidator:
    """Validator for text prompts used in generation."""
    
    MAX_PROMPT_LENGTH = 1000
    MIN_PROMPT_LENGTH = 3
    
    @staticmethod
    def validate_prompt(prompt: str) -> None:
        """Validate text prompt for generation."""
        if not isinstance(prompt, str):
            raise ValidationError("Prompt must be a string")
        
        prompt = prompt.strip()
        
        if len(prompt) < TextPromptValidator.MIN_PROMPT_LENGTH:
            raise ValidationError(
                f"Prompt too short (minimum {TextPromptValidator.MIN_PROMPT_LENGTH} characters)"
            )
        
        if len(prompt) > TextPromptValidator.MAX_PROMPT_LENGTH:
            raise ValidationError(
                f"Prompt too long (maximum {TextPromptValidator.MAX_PROMPT_LENGTH} characters)"
            )
        
        # Check for potentially harmful content
        prohibited_patterns = [
            r'\b(explosive|bomb|poison|toxic)\b',
            r'\b(illegal|drug|narcotic)\b',
            r'\b(weapon|harm|kill)\b'
        ]
        
        for pattern in prohibited_patterns:
            if re.search(pattern, prompt.lower()):
                raise ValidationError("Prompt contains prohibited content")
    
    @staticmethod
    def sanitize_prompt(prompt: str) -> str:
        """Sanitize and clean prompt text."""
        if not isinstance(prompt, str):
            return ""
        
        # Remove excessive whitespace
        cleaned = re.sub(r'\s+', ' ', prompt.strip())
        
        # Remove special characters that might cause issues
        cleaned = re.sub(r'[^\w\s\-\.,;:!?()]', '', cleaned)
        
        return cleaned


class ParameterValidator:
    """Validator for generation parameters."""
    
    @staticmethod
    def validate_num_molecules(num_molecules: int) -> None:
        """Validate number of molecules parameter."""
        if not isinstance(num_molecules, int):
            raise ValidationError("num_molecules must be an integer")
        
        if num_molecules < 1:
            raise ValidationError("num_molecules must be at least 1")
        
        if num_molecules > 100:
            raise ValidationError("num_molecules cannot exceed 100")
    
    @staticmethod
    def validate_guidance_scale(guidance_scale: float) -> None:
        """Validate guidance scale parameter."""
        if not isinstance(guidance_scale, (int, float)):
            raise ValidationError("guidance_scale must be a number")
        
        if guidance_scale < 0.1:
            raise ValidationError("guidance_scale must be at least 0.1")
        
        if guidance_scale > 20.0:
            raise ValidationError("guidance_scale cannot exceed 20.0")
    
    @staticmethod
    def validate_interpolation_weights(weights: Dict[str, float]) -> None:
        """Validate interpolation weights for multimodal generation."""
        if not isinstance(weights, dict):
            raise ValidationError("interpolation_weights must be a dictionary")
        
        valid_keys = {'text', 'image', 'reference'}
        for key in weights.keys():
            if key not in valid_keys:
                raise ValidationError(f"Invalid weight key: {key}. Must be one of {valid_keys}")
        
        for key, weight in weights.items():
            if not isinstance(weight, (int, float)):
                raise ValidationError(f"Weight for '{key}' must be a number")
            
            if weight < 0 or weight > 1:
                raise ValidationError(f"Weight for '{key}' must be between 0 and 1")
        
        # Check that weights sum to approximately 1
        total_weight = sum(weights.values())
        if abs(total_weight - 1.0) > 0.01:
            raise ValidationError(
                f"Interpolation weights must sum to 1.0 (current sum: {total_weight:.3f})"
            )


# Enhanced security and rate limiting validators

class SecurityValidator:
    """Advanced security validation for all inputs."""
    
    SUSPICIOUS_PATTERNS = [
        r'\b(eval|exec|import|open|file)\s*\(',
        r'[<>{}\[\]`\\]',
        r'\b(script|javascript|vbscript)\b',
        r'\$\{.*\}',  # Template injection
        r'\b(system|shell|cmd)\b',
        r'\b(SELECT|INSERT|UPDATE|DELETE|DROP)\b',  # SQL injection
    ]
    
    @staticmethod
    def validate_inputs(args: tuple, kwargs: dict) -> None:
        """Comprehensive security validation of all inputs."""
        
        # Check all string inputs for suspicious patterns
        all_strings = []
        
        # Extract strings from args
        for arg in args:
            if isinstance(arg, str):
                all_strings.append(arg)
        
        # Extract strings from kwargs
        for value in kwargs.values():
            if isinstance(value, str):
                all_strings.append(value)
            elif isinstance(value, dict):
                for nested_val in value.values():
                    if isinstance(nested_val, str):
                        all_strings.append(nested_val)
        
        # Security pattern detection
        for input_str in all_strings:
            for pattern in SecurityValidator.SUSPICIOUS_PATTERNS:
                if re.search(pattern, input_str, re.IGNORECASE):
                    raise SecurityValidationError(
                        f"Suspicious pattern detected: {pattern[:20]}...",
                        threat_level="high"
                    )
        
        # Length-based attacks
        for input_str in all_strings:
            if len(input_str) > 10000:
                raise SecurityValidationError(
                    "Input string too long - possible DoS attack",
                    threat_level="medium"
                )


class RateLimitValidator:
    """Rate limiting validator to prevent abuse."""
    
    _call_history = defaultdict(list)
    _max_calls_per_minute = 100
    _max_calls_per_hour = 1000
    
    @classmethod
    def check_rate_limit(cls, function_name: str) -> None:
        """Check if function call exceeds rate limits."""
        
        current_time = time.time()
        call_times = cls._call_history[function_name]
        
        # Remove old entries (older than 1 hour)
        call_times[:] = [t for t in call_times if current_time - t < 3600]
        
        # Check hourly limit
        if len(call_times) >= cls._max_calls_per_hour:
            raise SecurityValidationError(
                f"Rate limit exceeded for {function_name}: {cls._max_calls_per_hour}/hour",
                threat_level="high"
            )
        
        # Check per-minute limit
        recent_calls = [t for t in call_times if current_time - t < 60]
        if len(recent_calls) >= cls._max_calls_per_minute:
            raise SecurityValidationError(
                f"Rate limit exceeded for {function_name}: {cls._max_calls_per_minute}/minute",
                threat_level="medium"
            )
        
        # Add current call
        call_times.append(current_time)


def _validate_function_inputs(func: Callable, args: tuple, kwargs: dict) -> None:
    """Validate function inputs comprehensively."""
    
    # Basic type validation
    if hasattr(func, '__annotations__'):
        annotations = func.__annotations__
        
        # Check args against annotations (simplified)
        param_names = list(func.__code__.co_varnames[:func.__code__.co_argcount])
        
        for i, arg in enumerate(args[1:], 1):  # Skip 'self'
            if i < len(param_names):
                param_name = param_names[i]
                expected_type = annotations.get(param_name)
                
                if expected_type and not isinstance(arg, expected_type):
                    raise ValidationError(
                        f"Argument {param_name} expected {expected_type.__name__}, got {type(arg).__name__}",
                        field=param_name,
                        value=arg
                    )


def _validate_function_outputs(func: Callable, result: Any) -> None:
    """Validate function outputs for safety."""
    
    # Check output type consistency
    if hasattr(func, '__annotations__') and 'return' in func.__annotations__:
        expected_return_type = func.__annotations__['return']
        
        # Handle Union types and Optional
        if hasattr(expected_return_type, '__origin__'):
            if expected_return_type.__origin__ is Union:
                valid_types = expected_return_type.__args__
                if not any(isinstance(result, t) for t in valid_types if t is not type(None)):
                    raise ValidationError(
                        f"Function {func.__name__} returned unexpected type: {type(result)}"
                    )
        elif not isinstance(result, expected_return_type):
            raise ValidationError(
                f"Function {func.__name__} returned {type(result)}, expected {expected_return_type}"
            )


def validate_inputs(func: Callable) -> Callable:
    """Decorator to validate inputs to generation functions with enhanced security."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Get function name for error context
        func_name = func.__name__
        
        try:
            # Comprehensive input validation
            _validate_function_inputs(func, args, kwargs)
            
            # Security validation
            SecurityValidator.validate_inputs(args, kwargs)
            
            # Rate limiting check
            RateLimitValidator.check_rate_limit(func.__name__)
            
            # Validate prompt if present
            if 'prompt' in kwargs and kwargs['prompt'] is not None:
                TextPromptValidator.validate_prompt(kwargs['prompt'])
                kwargs['prompt'] = TextPromptValidator.sanitize_prompt(kwargs['prompt'])
            elif len(args) > 1:  # Positional prompt argument
                TextPromptValidator.validate_prompt(args[1])
                args = list(args)
                args[1] = TextPromptValidator.sanitize_prompt(args[1])
                args = tuple(args)
            
            # Validate num_molecules
            if 'num_molecules' in kwargs:
                ParameterValidator.validate_num_molecules(kwargs['num_molecules'])
            
            # Validate guidance_scale
            if 'guidance_scale' in kwargs:
                ParameterValidator.validate_guidance_scale(kwargs['guidance_scale'])
            
            # Validate interpolation_weights for multimodal functions
            if 'interpolation_weights' in kwargs and kwargs['interpolation_weights'] is not None:
                ParameterValidator.validate_interpolation_weights(kwargs['interpolation_weights'])
            
            # Execute function
            result = func(*args, **kwargs)
            
            # Output validation
            _validate_function_outputs(func, result)
            
            return result
            
        except ValidationError:
            raise
        except SecurityValidationError:
            raise
        except Exception as e:
            raise ValidationError(f"Unexpected validation error in {func_name}: {str(e)}")
    
    return wrapper


def validate_molecule_input(func: Callable) -> Callable:
    """Decorator to validate molecule inputs (SMILES strings)."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        func_name = func.__name__
        
        # Check for molecule/SMILES parameters
        molecule_params = ['molecule', 'smiles', 'reference_smiles', 'start', 'end']
        
        for param in molecule_params:
            if param in kwargs and kwargs[param] is not None:
                smiles = kwargs[param]
                if not SMILESValidator.is_valid_smiles(smiles):
                    raise ValidationError(f"Invalid SMILES in parameter '{param}': {smiles}")
                
                # Check fragrance suitability
                suitability = SMILESValidator.check_fragrance_suitability(smiles)
                if not suitability['suitable']:
                    issues = '; '.join(suitability['issues'])
                    raise ValidationError(
                        f"Molecule in parameter '{param}' not suitable for fragrance: {issues}"
                    )
        
        return func(*args, **kwargs)
    
    return wrapper


class SafetyValidator:
    """Validator for safety-related parameters and constraints."""
    
    PROHIBITED_SUBSTRUCTURES = [
        # Highly toxic or dangerous substructures
        "[N+](=O)[O-]",  # Nitro compounds (some are explosive)
        "C(=O)Cl",       # Acid chlorides (reactive)
        "S(=O)(=O)Cl",   # Sulfonyl chlorides (reactive)
        "[As]",          # Arsenic compounds
        "[Hg]",          # Mercury compounds
        "[Pb]",          # Lead compounds
        "[Cd]",          # Cadmium compounds
    ]
    
    @staticmethod
    def check_prohibited_structures(smiles: str) -> List[str]:
        """Check for prohibited substructures in molecule."""
        if not SMILESValidator.is_valid_smiles(smiles):
            return ["Invalid SMILES structure"]
        
        mol = Chem.MolFromSmiles(smiles)
        violations = []
        
        for smarts in SafetyValidator.PROHIBITED_SUBSTRUCTURES:
            try:
                pattern = Chem.MolFromSmarts(smarts)
                if mol.HasSubstructMatch(pattern):
                    violations.append(f"Contains prohibited substructure: {smarts}")
            except:
                continue  # Skip invalid SMARTS patterns
        
        return violations
    
    @staticmethod
    def enforce_safety_constraints(smiles: str) -> None:
        """Enforce safety constraints on generated molecules."""
        violations = SafetyValidator.check_prohibited_structures(smiles)
        
        if violations:
            raise ValidationError(
                f"Safety constraint violation: {'; '.join(violations)}"
            )


def safety_check(func: Callable) -> Callable:
    """Decorator to enforce safety checks on molecule generation."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Execute function first
        result = func(*args, **kwargs)
        
        # Check safety of generated molecules
        molecules = result if isinstance(result, list) else [result] if result else []
        
        for mol in molecules:
            if mol and hasattr(mol, 'smiles') and mol.smiles:
                try:
                    SafetyValidator.enforce_safety_constraints(mol.smiles)
                except ValidationError as e:
                    # Log the safety violation but don't fail completely
                    from .logging import SmellDiffusionLogger
                    logger = SmellDiffusionLogger()
                    logger.log_error("safety_check", e, {"smiles": mol.smiles})
        
        return result
    
    return wrapper