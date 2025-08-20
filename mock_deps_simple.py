#!/usr/bin/env python3
"""Simple dependency mock for testing core functionality."""

class MockModule:
    def __getattr__(self, name):
        return MockModule()
    
    def __call__(self, *args, **kwargs):
        return MockModule()
    
    def __str__(self):
        return "MockModule"
    
    def __repr__(self):
        return "MockModule()"

# Mock commonly missing dependencies
import sys

# Mock rdkit
rdkit = MockModule()
rdkit.Chem = MockModule()
rdkit.Chem.Descriptors = MockModule()
rdkit.DataStructs = MockModule()
sys.modules['rdkit'] = rdkit
sys.modules['rdkit.Chem'] = rdkit.Chem
sys.modules['rdkit.Chem.Descriptors'] = rdkit.Chem.Descriptors
sys.modules['rdkit.DataStructs'] = rdkit.DataStructs

# Mock torch and transformers if needed
try:
    import torch
    print("✅ PyTorch available")
except ImportError:
    torch = MockModule()
    sys.modules['torch'] = torch
    print("⚠️ PyTorch mocked")

try:
    import transformers
    print("✅ Transformers available")
except ImportError:
    transformers = MockModule()
    sys.modules['transformers'] = transformers
    print("⚠️ Transformers mocked")

print("✅ Mock dependencies loaded successfully")