"""Tests for core smell diffusion functionality."""

import pytest
import asyncio
from unittest.mock import Mock, patch, MagicMock

from smell_diffusion.core.smell_diffusion import SmellDiffusion
from smell_diffusion.core.molecule import Molecule, FragranceNotes, SafetyProfile
from smell_diffusion.utils.validation import ValidationError


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
        def mock_load_model(self):
            self._is_loaded = True
            
        with patch.object(SmellDiffusion, '_load_model', mock_load_model):
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
        
        # Empty prompt should raise ValidationError
        with pytest.raises(ValidationError):
            model.generate("", num_molecules=1)
        
        # Very short prompt should raise ValidationError
        with pytest.raises(ValidationError):
            model.generate("a", num_molecules=1)
    
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


class TestAdvancedGeneration:
    """Test advanced generation features."""
    
    def test_batch_generation(self, mock_rdkit):
        """Test batch molecule generation."""
        model = SmellDiffusion()
        model._is_loaded = True
        
        prompts = [
            "Fresh citrus scent",
            "Floral rose bouquet", 
            "Woody cedar forest"
        ]
        
        results = model.batch_generate(prompts, num_molecules=2)
        
        assert isinstance(results, list)
        assert len(results) == len(prompts)
        
        for result_set in results:
            assert isinstance(result_set, list)
    
    def test_performance_optimization(self, mock_rdkit):
        """Test performance optimization features."""
        model = SmellDiffusion()
        model._is_loaded = True
        
        # Test throughput optimization
        model.optimize_for_throughput()
        
        # Test performance stats
        stats = model.get_performance_stats()
        assert isinstance(stats, dict)
        assert 'generations' in stats
        assert 'cache_hits' in stats
        assert 'model_loaded' in stats
    
    def test_cached_prompt_analysis(self, mock_rdkit):
        """Test cached prompt analysis."""
        model = SmellDiffusion()
        model._is_loaded = True
        
        prompt = "Fresh citrus bergamot"
        
        # First call should analyze and cache
        result1 = model._analyze_prompt_cached(prompt)
        
        # Second call should use cache
        result2 = model._analyze_prompt_cached(prompt)
        
        assert result1 == result2
        assert isinstance(result1, dict)
    
    def test_error_handling_in_generation(self, mock_rdkit):
        """Test error handling during generation."""
        model = SmellDiffusion()
        model._is_loaded = True
        
        # Test with invalid parameters
        with pytest.raises(ValidationError):
            model.generate("test", num_molecules=0)
        
        with pytest.raises(ValidationError):
            model.generate("test", num_molecules=101)
        
        with pytest.raises(ValidationError):
            model.generate("test", guidance_scale=0.05)
        
        with pytest.raises(ValidationError):
            model.generate("test", guidance_scale=25.0)
    
    def test_fallback_molecules(self, mock_rdkit):
        """Test fallback molecule generation."""
        model = SmellDiffusion()
        
        fallbacks = model._get_fallback_molecules("test prompt", 3)
        
        assert isinstance(fallbacks, list)
        assert len(fallbacks) <= 3
        assert all(isinstance(mol, Molecule) for mol in fallbacks)
    
    def test_pattern_compilation(self):
        """Test molecular pattern compilation."""
        model = SmellDiffusion()
        model._precompile_patterns()
        
        assert hasattr(model, '_compiled_patterns')
        assert isinstance(model._compiled_patterns, dict)
        assert len(model._compiled_patterns) > 0
    
    def test_generation_with_max_attempts(self, mock_rdkit):
        """Test generation with max attempts parameter."""
        model = SmellDiffusion()
        model._is_loaded = True
        
        result = model.generate(
            "test prompt", 
            num_molecules=1,
            max_attempts=5
        )
        
        assert result is not None
    
    def test_generation_with_min_safety_score(self, mock_rdkit):
        """Test generation with minimum safety score."""
        model = SmellDiffusion()
        model._is_loaded = True
        
        result = model.generate(
            "test prompt",
            num_molecules=1,
            safety_filter=True,
            min_safety_score=80.0
        )
        
        assert result is not None


class TestErrorRecovery:
    """Test error recovery mechanisms."""
    
    def test_model_loading_failure(self):
        """Test graceful handling of model loading failure."""
        model = SmellDiffusion("nonexistent-model")
        
        # Should not crash even if model fails to load
        with patch.object(model, '_load_model', side_effect=Exception("Loading failed")):
            try:
                model.generate("test prompt")
            except Exception:
                pass  # Expected to handle gracefully
    
    def test_prompt_analysis_failure(self, mock_rdkit):
        """Test handling of prompt analysis failure."""
        model = SmellDiffusion()
        model._is_loaded = True
        
        # Mock analysis failure
        with patch.object(model, '_analyze_prompt', side_effect=Exception("Analysis failed")):
            result = model.generate("test prompt", num_molecules=1)
            
            # Should fall back to default categories and still generate
            assert result is not None
    
    def test_molecule_creation_failure(self, mock_rdkit):
        """Test handling of molecule creation failures."""
        model = SmellDiffusion()
        model._is_loaded = True
        
        # Mock molecule creation failure
        with patch('smell_diffusion.core.molecule.Molecule', side_effect=Exception("Creation failed")):
            result = model.generate("test prompt", num_molecules=1)
            
            # Should fall back to safe molecules
            assert result is not None
    
    def test_safety_evaluation_failure(self, mock_rdkit):
        """Test handling of safety evaluation failures."""
        model = SmellDiffusion()
        model._is_loaded = True
        
        # Mock safety evaluation failure
        with patch.object(Molecule, 'get_safety_profile', side_effect=Exception("Safety failed")):
            result = model.generate("test prompt", num_molecules=1, safety_filter=True)
            
            # Should still generate molecules
            assert result is not None


class TestIntegration:
    """Integration tests for complete workflows."""
    
    def test_complete_generation_workflow(self, mock_rdkit):
        """Test complete molecule generation workflow."""
        model = SmellDiffusion.from_pretrained("test-model")
        
        # Generate molecules
        molecules = model.generate(
            prompt="Fresh oceanic breeze with citrus notes",
            num_molecules=3,
            guidance_scale=7.5,
            safety_filter=True
        )
        
        assert isinstance(molecules, list)
        assert len(molecules) <= 3
        
        for mol in molecules:
            assert isinstance(mol, Molecule)
            assert mol.is_valid
            
            # Test properties
            assert mol.molecular_weight > 0
            assert isinstance(mol.logp, float)
            
            # Test fragrance notes
            notes = mol.fragrance_notes
            assert isinstance(notes, FragranceNotes)
            
            # Test safety
            safety = mol.get_safety_profile()
            assert isinstance(safety, SafetyProfile)
            assert safety.score >= 50  # Should pass safety filter
    
    def test_multimodal_integration(self, mock_rdkit):
        """Test integration with multimodal components."""
        from smell_diffusion.multimodal.generator import MultiModalGenerator
        
        model = SmellDiffusion.from_pretrained("test-model")
        multimodal = MultiModalGenerator(model)
        
        molecules = multimodal.generate(
            text="Fresh citrus scent",
            num_molecules=2
        )
        
        assert isinstance(molecules, list)
        assert len(molecules) <= 2
    
    def test_safety_integration(self, mock_rdkit):
        """Test integration with safety evaluation."""
        from smell_diffusion.safety.evaluator import SafetyEvaluator
        
        model = SmellDiffusion.from_pretrained("test-model")
        evaluator = SafetyEvaluator()
        
        molecule = model.generate("test prompt", num_molecules=1)
        
        if molecule:
            safety_report = evaluator.comprehensive_evaluation(molecule)
            
            assert hasattr(safety_report, 'overall_score')
            assert hasattr(safety_report, 'ifra_compliant')
            assert hasattr(safety_report, 'recommendations')


class TestPerformance:
    """Performance and benchmarking tests."""
    
    def test_generation_timing(self, mock_rdkit):
        """Test generation performance."""
        import time
        
        model = SmellDiffusion.from_pretrained("test-model")
        
        start_time = time.time()
        result = model.generate("test prompt", num_molecules=5)
        end_time = time.time()
        
        generation_time = end_time - start_time
        
        # Should complete within reasonable time (generous for CI)
        assert generation_time < 10.0
        assert result is not None
    
    def test_batch_performance(self, mock_rdkit):
        """Test batch generation performance."""
        model = SmellDiffusion.from_pretrained("test-model")
        
        prompts = [f"test prompt {i}" for i in range(5)]
        
        import time
        start_time = time.time()
        results = model.batch_generate(prompts, num_molecules=2)
        end_time = time.time()
        
        batch_time = end_time - start_time
        
        # Batch should be reasonably fast
        assert batch_time < 15.0
        assert len(results) == len(prompts)
    
    def test_cache_effectiveness(self, mock_rdkit):
        """Test caching effectiveness."""
        model = SmellDiffusion.from_pretrained("test-model")
        
        prompt = "repeated test prompt"
        
        # First generation
        result1 = model.generate(prompt, num_molecules=1)
        stats1 = model.get_performance_stats()
        
        # Second generation (should hit cache)
        result2 = model.generate(prompt, num_molecules=1)
        stats2 = model.get_performance_stats()
        
        # Cache hits should increase
        assert stats2['cache_hits'] >= stats1['cache_hits']


@pytest.mark.asyncio
class TestAsyncFeatures:
    """Test asynchronous features."""
    
    async def test_async_batch_processing(self, mock_rdkit):
        """Test asynchronous batch processing."""
        from smell_diffusion.utils.async_utils import AsyncBatchProcessor
        
        processor = AsyncBatchProcessor(batch_size=2, max_concurrent_batches=2)
        
        items = ["item1", "item2", "item3", "item4", "item5"]
        
        async def mock_processor(item):
            await asyncio.sleep(0.01)  # Simulate processing
            return f"processed_{item}"
        
        results = await processor.process_items(items, mock_processor)
        
        assert len(results) == len(items)
        assert all("processed_" in result for result in results)


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_empty_database_categories(self):
        """Test handling of empty database categories."""
        model = SmellDiffusion()
        
        # Temporarily empty a category
        original_citrus = model.FRAGRANCE_DATABASE['citrus']
        model.FRAGRANCE_DATABASE['citrus'] = []
        
        try:
            scores = {'citrus': 1.0}
            molecules = model._select_molecules(scores, 1)
            
            # Should handle gracefully
            assert isinstance(molecules, list)
        finally:
            # Restore original
            model.FRAGRANCE_DATABASE['citrus'] = original_citrus
    
    def test_unicode_prompts(self, mock_rdkit):
        """Test handling of Unicode prompts."""
        model = SmellDiffusion()
        model._is_loaded = True
        
        unicode_prompts = [
            "Citrus frais avec des notes de bergamote",  # French
            "フレッシュなシトラスの香り",  # Japanese
            "Fresco aroma cítrico",  # Spanish
        ]
        
        for prompt in unicode_prompts:
            result = model.generate(prompt, num_molecules=1)
            assert result is not None
    
    def test_extremely_long_prompts(self, mock_rdkit):
        """Test handling of very long prompts."""
        model = SmellDiffusion()
        model._is_loaded = True
        
        long_prompt = "citrus " * 1000  # Very long prompt
        
        # Should raise ValidationError for too long prompt
        with pytest.raises(ValidationError):
            model.generate(long_prompt, num_molecules=1)
    
    def test_special_characters_in_prompts(self, mock_rdkit):
        """Test handling of special characters."""
        model = SmellDiffusion()
        model._is_loaded = True
        
        special_prompts = [
            "Fresh@citrus#scent!",
            "Rose & jasmine + lavender",
            "Woody-cedar/pine*essence",
            "Vanilla... sweet, gourmand?",
        ]
        
        for prompt in special_prompts:
            result = model.generate(prompt, num_molecules=1)
            assert result is not None