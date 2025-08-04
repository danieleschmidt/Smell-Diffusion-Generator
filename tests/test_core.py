"""Tests for core smell diffusion functionality."""

import pytest
from unittest.mock import Mock, patch, MagicMock

from smell_diffusion.core.smell_diffusion import SmellDiffusion
from smell_diffusion.core.molecule import Molecule, FragranceNotes, SafetyProfile


class TestMolecule:
    """Test cases for Molecule class."""
    
    def test_valid_molecule_creation(self, sample_smiles):
        """Test creating valid molecules."""
        for smiles in sample_smiles['valid']:
            mol = Molecule(smiles)
            assert mol.smiles == smiles
            assert mol.is_valid  # Will be True with mock
    
    def test_invalid_molecule_creation(self, sample_smiles):
        """Test creating invalid molecules."""
        for smiles in sample_smiles['invalid']:
            mol = Molecule(smiles)  
            assert mol.smiles == smiles
            assert not mol.is_valid  # Will be False with mock
    
    def test_molecule_properties(self, sample_smiles, mock_rdkit):
        """Test molecule property calculations."""
        mol = Molecule(sample_smiles['valid'][0])
        
        # Test with mocked values
        assert mol.molecular_weight == 154.25
        assert mol.logp == 2.3
        assert isinstance(mol.fragrance_notes, FragranceNotes)
    
    def test_fragrance_notes_prediction(self, sample_smiles, mock_rdkit):
        """Test fragrance note prediction."""
        mol = Molecule(sample_smiles['valid'][0])
        notes = mol.fragrance_notes
        
        assert isinstance(notes.top, list)
        assert isinstance(notes.middle, list)
        assert isinstance(notes.base, list)
        assert isinstance(notes.intensity, float)
        assert notes.intensity >= 0 and notes.intensity <= 10
        assert notes.longevity in ['very_short', 'short', 'medium', 'long', 'very_long']
    
    def test_safety_profile(self, sample_smiles, mock_rdkit):
        """Test safety profile generation."""
        mol = Molecule(sample_smiles['valid'][0])
        safety = mol.get_safety_profile()
        
        assert isinstance(safety, SafetyProfile)
        assert isinstance(safety.score, float)
        assert safety.score >= 0 and safety.score <= 100
        assert isinstance(safety.ifra_compliant, bool)
        assert isinstance(safety.allergens, list)
        assert isinstance(safety.warnings, list)
    
    def test_molecule_svg_generation(self, sample_smiles, mock_rdkit):
        """Test SVG generation."""
        mol = Molecule(sample_smiles['valid'][0])
        with patch('smell_diffusion.core.molecule.rdMolDraw2D') as mock_drawer:
            mock_drawer_instance = Mock()
            mock_drawer_instance.GetDrawingText.return_value = "<svg>test</svg>"
            mock_drawer.MolDraw2DSVG.return_value = mock_drawer_instance
            
            svg = mol.to_svg()
            assert svg == "<svg>test</svg>"
    
    def test_molecule_dict_conversion(self, sample_smiles, mock_rdkit):
        """Test conversion to dictionary."""
        mol = Molecule(sample_smiles['valid'][0])
        mol_dict = mol.to_dict()
        
        required_keys = [
            'smiles', 'description', 'is_valid', 'molecular_weight', 
            'logp', 'fragrance_notes', 'safety_profile'
        ]
        
        for key in required_keys:
            assert key in mol_dict
    
    def test_molecule_string_representation(self, sample_smiles):
        """Test string representations."""
        mol = Molecule(sample_smiles['valid'][0])
        
        str_repr = str(mol)
        assert sample_smiles['valid'][0] in str_repr
        assert "Molecule" in str_repr
        
        repr_str = repr(mol)
        assert sample_smiles['valid'][0] in repr_str


class TestSmellDiffusion:
    """Test cases for SmellDiffusion class."""
    
    def test_model_initialization(self):
        """Test model initialization."""
        model = SmellDiffusion("test-model")
        assert model.model_name == "test-model"
        assert not model._is_loaded
    
    def test_from_pretrained(self):
        """Test loading pretrained model."""
        with patch.object(SmellDiffusion, '_load_model'):
            model = SmellDiffusion.from_pretrained("test-model")
            assert model.model_name == "test-model"
            assert model._is_loaded
    
    def test_prompt_analysis(self):
        """Test text prompt analysis."""
        model = SmellDiffusion()
        
        # Test citrus prompt
        citrus_scores = model._analyze_prompt("Fresh lemon and bergamot citrus")
        assert citrus_scores['citrus'] > 0
        
        # Test floral prompt
        floral_scores = model._analyze_prompt("Beautiful rose and jasmine flowers")
        assert floral_scores['floral'] > 0
        
        # Test mixed prompt
        mixed_scores = model._analyze_prompt("Woody cedar with fresh citrus top notes")
        assert mixed_scores['woody'] > 0
        assert mixed_scores['citrus'] > 0
    
    def test_molecule_selection(self):
        """Test molecule selection from categories."""
        model = SmellDiffusion()
        
        category_scores = {'citrus': 0.7, 'floral': 0.3}
        molecules = model._select_molecules(category_scores, 3)
        
        assert len(molecules) == 3
        assert all(isinstance(mol, str) for mol in molecules)
    
    def test_single_molecule_generation(self, mock_rdkit):
        """Test generating single molecule."""
        model = SmellDiffusion()
        model._is_loaded = True
        
        result = model.generate("Fresh citrus scent", num_molecules=1)
        
        assert result is not None
        assert isinstance(result, Molecule)
    
    def test_multiple_molecule_generation(self, mock_rdkit):
        """Test generating multiple molecules."""
        model = SmellDiffusion()
        model._is_loaded = True
        
        results = model.generate("Fresh citrus scent", num_molecules=3)
        
        assert isinstance(results, list)
        assert len(results) <= 3  # May be fewer due to safety filtering
        assert all(isinstance(mol, Molecule) for mol in results)
    
    def test_generation_with_safety_filter(self, mock_rdkit):
        """Test generation with safety filtering enabled."""
        model = SmellDiffusion()
        model._is_loaded = True
        
        # Mock unsafe molecule
        with patch.object(Molecule, 'get_safety_profile') as mock_safety:
            mock_safety.return_value = SafetyProfile(
                score=30.0,  # Low safety score
                ifra_compliant=False,
                allergens=['test_allergen'],
                warnings=['unsafe']
            )
            
            results = model.generate("Test prompt", num_molecules=3, safety_filter=True)
            
            # Should fallback to safe molecule
            assert len(results) >= 1
    
    def test_generation_without_safety_filter(self, mock_rdkit):
        """Test generation with safety filtering disabled."""
        model = SmellDiffusion()
        model._is_loaded = True
        
        results = model.generate("Test prompt", num_molecules=3, safety_filter=False)
        
        assert isinstance(results, list)
        assert len(results) <= 3
    
    def test_molecular_variation(self):
        """Test molecular variation generation."""
        model = SmellDiffusion()
        
        base_smiles = "CC(C)=CCCC(C)=CCO"
        varied = model._add_molecular_variation(base_smiles)
        
        assert isinstance(varied, str)
        # Variation may return original or modified molecule
    
    def test_model_info(self):
        """Test getting model information."""
        model = SmellDiffusion("test-model")
        model._is_loaded = True
        
        info = model.get_model_info()
        
        assert isinstance(info, dict)
        assert 'model_name' in info
        assert 'is_loaded' in info
        assert 'supported_categories' in info
        assert info['model_name'] == 'test-model'
        assert info['is_loaded'] is True
    
    def test_empty_prompt_handling(self, mock_rdkit):
        """Test handling of empty or invalid prompts."""
        model = SmellDiffusion()
        model._is_loaded = True
        
        # Empty prompt should still generate molecules
        result = model.generate("", num_molecules=1)
        assert result is not None
        
        # Very short prompt
        result = model.generate("a", num_molecules=1)
        assert result is not None
    
    def test_large_molecule_request(self, mock_rdkit):
        """Test handling large molecule generation requests."""
        model = SmellDiffusion()
        model._is_loaded = True
        
        # Request many molecules
        results = model.generate("Test", num_molecules=50)
        
        # Should handle gracefully
        assert isinstance(results, list)
        assert len(results) <= 50
    
    def test_model_string_representation(self):
        """Test string representation of model."""
        model = SmellDiffusion("test-model")
        model._is_loaded = True
        
        str_repr = str(model)
        assert "SmellDiffusion" in str_repr
        assert "test-model" in str_repr
        assert "loaded=True" in str_repr


class TestFragranceDatabase:
    """Test fragrance molecule database."""
    
    def test_database_structure(self):
        """Test database has required structure."""
        from smell_diffusion.core.smell_diffusion import SmellDiffusion
        
        db = SmellDiffusion.FRAGRANCE_DATABASE
        
        assert isinstance(db, dict)
        assert len(db) > 0
        
        # Check required categories
        required_categories = ['citrus', 'floral', 'woody', 'vanilla', 'musky', 'fresh']
        for category in required_categories:
            assert category in db
            assert isinstance(db[category], list)
            assert len(db[category]) > 0
    
    def test_scent_keywords(self):
        """Test scent keyword mapping."""
        from smell_diffusion.core.smell_diffusion import SmellDiffusion
        
        keywords = SmellDiffusion.SCENT_KEYWORDS
        
        assert isinstance(keywords, dict)
        assert len(keywords) > 0
        
        for category, word_list in keywords.items():
            assert isinstance(word_list, list)
            assert len(word_list) > 0
            assert all(isinstance(word, str) for word in word_list)


class TestMolecularProperties:
    """Test molecular property calculations."""
    
    def test_molecular_weight_calculation(self, mock_rdkit):
        """Test molecular weight calculation."""
        mol = Molecule("CC(C)=CCCC(C)=CCO")
        
        # Should use mocked value
        assert mol.molecular_weight == 154.25
    
    def test_logp_calculation(self, mock_rdkit):
        """Test LogP calculation."""
        mol = Molecule("CC(C)=CCCC(C)=CCO")
        
        # Should use mocked value
        assert mol.logp == 2.3
    
    def test_invalid_molecule_properties(self, mock_rdkit):
        """Test properties of invalid molecules."""
        mol = Molecule("INVALID_SMILES")
        
        assert mol.molecular_weight == 0.0
        assert mol.logp == 0.0
        assert not mol.is_valid


@pytest.mark.parametrize("prompt,expected_categories", [
    ("Fresh lemon citrus", ["citrus"]),
    ("Beautiful rose floral", ["floral"]),
    ("Woody cedar forest", ["woody"]),
    ("Sweet vanilla dessert", ["vanilla"]),
    ("Musky sensual skin", ["musky"]),
    ("Clean fresh water", ["fresh"]),
    ("Citrus rose woody", ["citrus", "floral", "woody"]),
])
def test_prompt_category_detection(prompt, expected_categories):
    """Test prompt analysis detects correct categories."""
    model = SmellDiffusion()
    scores = model._analyze_prompt(prompt)
    
    for category in expected_categories:
        assert scores[category] > 0, f"Category '{category}' not detected in prompt '{prompt}'"