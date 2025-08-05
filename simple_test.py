#!/usr/bin/env python3
"""Simple test to verify functionality."""

import sys
sys.path.insert(0, '.')

# Import mocks first
import mock_deps

# Test imports
from smell_diffusion.core.smell_diffusion import SmellDiffusion
from smell_diffusion.core.molecule import Molecule
from smell_diffusion.safety.evaluator import SafetyEvaluator

def test_basic_functionality():
    """Test basic functionality."""
    print("Testing basic functionality...")
    
    # Test molecule creation
    mol = Molecule("CC(C)=CCCC(C)=CCO")
    print(f"Molecule created: {mol.smiles}")
    print(f"Molecule valid: {mol.is_valid}")
    print(f"Molecular weight: {mol.molecular_weight}")
    
    # Test model creation
    model = SmellDiffusion()
    print(f"Model created: {model.model_name}")
    
    # Test generation
    result = model.generate("Fresh citrus", num_molecules=1)
    print(f"Generation result: {result}")
    
    # Test safety evaluation
    evaluator = SafetyEvaluator()
    safety = evaluator.evaluate(mol)
    print(f"Safety score: {safety.score}")
    
    print("All tests passed!")

if __name__ == "__main__":
    test_basic_functionality()