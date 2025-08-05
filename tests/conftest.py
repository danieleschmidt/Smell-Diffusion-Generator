"""Pytest configuration and shared fixtures."""

import pytest
import tempfile
import shutil
import sys
from pathlib import Path
from unittest.mock import Mock, patch

# Add parent directory to path and import mocks
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

# Import mock dependencies first
import mock_deps

from smell_diffusion.core.smell_diffusion import SmellDiffusion
from smell_diffusion.core.molecule import Molecule
from smell_diffusion.safety.evaluator import SafetyEvaluator
from smell_diffusion.utils.config import ConfigManager


@pytest.fixture
def temp_dir():
    """Create temporary directory for tests."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def sample_smiles():
    """Sample SMILES strings for testing."""
    return {
        'valid': [
            'CC(C)=CCCC(C)=CCO',  # Geraniol
            'CC1=CC=C(C=C1)C=O',  # p-Tolualdehyde
            'COC1=C(C=CC(=C1)C=O)O',  # Vanillin
        ],
        'invalid': [
            'INVALID_SMILES',
            '',
            'C[C',
            'XYZ123'
        ],
        'problematic': [
            'C[N+](=O)[O-]',  # Nitro compound
            'CS(=O)(=O)Cl',   # Sulfonyl chloride
        ]
    }


@pytest.fixture
def sample_molecules(sample_smiles):
    """Sample molecule objects for testing."""
    molecules = {}
    for category, smiles_list in sample_smiles.items():
        molecules[category] = [Molecule(smiles) for smiles in smiles_list]
    return molecules


@pytest.fixture
def mock_model():
    """Mock SmellDiffusion model for testing."""
    model = Mock(spec=SmellDiffusion)
    model.model_name = "test-model-v1"
    model._is_loaded = True
    
    # Mock generation method
    def mock_generate(prompt, num_molecules=1, **kwargs):
        # Return valid molecules for testing
        test_smiles = ['CC(C)=CCCC(C)=CCO', 'CC1=CC=C(C=C1)C=O']
        molecules = [Molecule(smiles) for smiles in test_smiles[:num_molecules]]
        return molecules if num_molecules > 1 else molecules[0] if molecules else None
    
    model.generate = mock_generate
    model.get_model_info.return_value = {
        "model_name": "test-model-v1",
        "is_loaded": True,
        "supported_categories": ["citrus", "floral", "woody"],
        "database_size": 15
    }
    
    return model


@pytest.fixture
def sample_prompts():
    """Sample text prompts for testing."""
    return [
        "Fresh citrus fragrance with bergamot and lemon",
        "Elegant floral bouquet with rose and jasmine",
        "Woody base notes with sandalwood and cedar",
        "Sweet vanilla and amber combination",
        "Marine aquatic scent with cucumber notes"
    ]


@pytest.fixture
def mock_safety_evaluator():
    """Mock safety evaluator for testing."""
    evaluator = Mock(spec=SafetyEvaluator)
    
    def mock_evaluate(molecule):
        # Return mock safety profile based on molecule validity
        if molecule.is_valid:
            from smell_diffusion.core.molecule import SafetyProfile
            return SafetyProfile(
                score=85.0,
                ifra_compliant=True,
                allergens=[],
                warnings=[]
            )
        else:
            from smell_diffusion.core.molecule import SafetyProfile
            return SafetyProfile(
                score=0.0,
                ifra_compliant=False,
                allergens=[],
                warnings=["Invalid molecule"]
            )
    
    evaluator.evaluate = mock_evaluate
    return evaluator


@pytest.fixture
def test_config(temp_dir):
    """Test configuration for isolated testing."""
    config_path = temp_dir / "test_config.yaml"
    config_manager = ConfigManager(config_path)
    
    # Create test configuration
    from smell_diffusion.utils.config import SmellDiffusionConfig
    test_config = SmellDiffusionConfig()
    test_config.model.model_name = "test-model"
    test_config.logging.enable_file_logging = False  # Disable for tests
    
    config_manager.save_config(test_config, config_path)
    return config_manager


@pytest.fixture
def mock_rdkit():
    """Mock RDKit for testing without chemistry dependencies."""
    with patch('smell_diffusion.core.molecule.Chem') as mock_chem:
        # Mock valid molecules
        mock_mol = Mock()
        mock_mol.GetNumAtoms.return_value = 20
        
        def mock_mol_from_smiles(smiles):
            if smiles and 'INVALID' not in smiles.upper() and smiles != '':
                return mock_mol
            return None
        
        mock_chem.MolFromSmiles = mock_mol_from_smiles
        mock_chem.MolToSmiles.return_value = "CC(C)=CCCC(C)=CCO"
        mock_chem.Descriptors.MolWt.return_value = 154.25
        mock_chem.Descriptors.MolLogP.return_value = 2.3
        
        yield mock_chem


@pytest.fixture(autouse=True)
def reset_global_cache():
    """Reset global cache between tests."""
    from smell_diffusion.utils.caching import get_cache
    cache = get_cache()
    cache.clear()
    yield
    cache.clear()


@pytest.fixture
def async_test_client():
    """Async test client for API testing."""
    from httpx import AsyncClient
    from smell_diffusion.api.server import app
    
    return AsyncClient(app=app, base_url="http://testserver")


# Test data constants
VALID_TEST_SMILES = [
    "CC(C)=CCCC(C)=CCO",  # Geraniol
    "CC1=CC=C(C=C1)C=O",  # p-Tolualdehyde  
    "COC1=C(C=CC(=C1)C=O)O",  # Vanillin
    "CC(C)CCCC(C)CCO",  # Citronellol
    "CC1CCC(CC1)C(C)(C)O",  # Linalool
]

INVALID_TEST_SMILES = [
    "INVALID_SMILES",
    "",
    "C[C",
    "XYZ123",
    None,
]

TEST_PROMPTS = [
    "Fresh citrus with bergamot",
    "Floral rose and jasmine",
    "Woody sandalwood base",
    "Sweet vanilla cream",
    "Marine aquatic fresh"
]


class TestDataGenerator:
    """Utility class for generating test data."""
    
    @staticmethod
    def generate_test_molecules(count: int = 5):
        """Generate test molecules."""
        return [Molecule(smiles) for smiles in VALID_TEST_SMILES[:count]]
    
    @staticmethod
    def generate_invalid_molecules(count: int = 3):
        """Generate invalid test molecules."""
        return [Molecule(smiles) for smiles in INVALID_TEST_SMILES[:count]]
    
    @staticmethod
    def generate_test_accord_brief():
        """Generate test accord brief."""
        return {
            'name': 'Test Fragrance',
            'inspiration': 'Spring garden in bloom',
            'character': ['fresh', 'floral'],
            'season': 'spring',
            'target_audience': 'unisex'
        }