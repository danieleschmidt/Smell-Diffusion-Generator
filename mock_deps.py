"""Mock dependencies for testing."""

# Mock torch
class MockTensor:
    def __init__(self, data=None):
        self.data = data or []
    
    def cpu(self):
        return self
    
    def numpy(self):
        import numpy as np
        return np.array(self.data)

class MockCuda:
    @staticmethod
    def is_available():
        return False

class MockTorch:
    cuda = MockCuda
    
    def tensor(self, data):
        return MockTensor(data)
    
    def randn(self, *args):
        import numpy as np
        return MockTensor(np.random.randn(*args).tolist())

import sys
sys.modules['torch'] = MockTorch()

# Mock transformers
class MockTransformers:
    class AutoTokenizer:
        @staticmethod
        def from_pretrained(model_name):
            return MockTokenizer()
    
    class AutoModel:
        @staticmethod
        def from_pretrained(model_name):
            return MockModel()

class MockTokenizer:
    def __call__(self, text, return_tensors=None, **kwargs):
        return {"input_ids": MockTensor([1, 2, 3])}

class MockModel:
    def __call__(self, **kwargs):
        return MockTensor([0.1, 0.2, 0.3])

sys.modules['transformers'] = MockTransformers()

# Mock RDKit components
class MockMol:
    def HasSubstructMatch(self, pattern):
        return False
    
    def GetNumAtoms(self):
        return 20

class MockDescriptors:
    @staticmethod
    def MolWt(mol):
        return 154.25
    
    @staticmethod
    def MolLogP(mol):
        return 2.3

class MockrdMolDescriptors:
    @staticmethod
    def CalcTPSA(mol):
        return 60.0
    
    @staticmethod
    def CalcNumHBD(mol):
        return 1
    
    @staticmethod
    def CalcNumHBA(mol):
        return 2

class MockDrawer:
    def DrawMolecule(self, mol):
        pass
    
    def FinishDrawing(self):
        pass
    
    def GetDrawingText(self):
        return "<svg>mock molecule</svg>"

class MockMolDraw2D:
    @staticmethod
    def MolDraw2DSVG(width=300, height=300):
        return MockDrawer()

class MockDraw:
    rdMolDraw2D = MockMolDraw2D

class MockChem:
    Mol = MockMol  # Add Mol class reference
    
    @staticmethod
    def MolFromSmiles(smiles):
        if smiles and "INVALID" not in smiles.upper() and smiles.strip() and not smiles.startswith('C[C') and 'XYZ' not in smiles:
            return MockMol()
        return None
    
    @staticmethod
    def MolToSmiles(mol):
        return "CC(C)=CCCC(C)=CCO"
    
    @staticmethod
    def MolFromSmarts(smarts):
        return MockMol()
    
    Descriptors = MockDescriptors
    rdMolDescriptors = MockrdMolDescriptors
    Draw = MockDraw

class MockRDKit:
    Chem = MockChem

# Mock PIL
class MockImageInstance:
    def resize(self, size):
        return self
    
    def convert(self, mode):
        return self
    
    def save(self, path):
        pass

class MockImage:
    Image = MockImageInstance  # Add nested Image class
    
    @staticmethod
    def open(path):
        return MockImageInstance()
    
    @staticmethod
    def new(mode, size, color=None):
        return MockImageInstance()

class MockPIL:
    Image = MockImage

sys.modules['PIL'] = MockPIL()
sys.modules['PIL.Image'] = MockImage

sys.modules['rdkit'] = MockRDKit()
sys.modules['rdkit.Chem'] = MockChem
sys.modules['rdkit.Chem.Descriptors'] = MockDescriptors
sys.modules['rdkit.Chem.rdMolDescriptors'] = MockrdMolDescriptors
sys.modules['rdkit.Chem.Draw'] = MockDraw
sys.modules['rdkit.Chem.Draw.rdMolDraw2D'] = MockMolDraw2D