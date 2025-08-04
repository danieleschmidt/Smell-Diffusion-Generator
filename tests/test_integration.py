"""Integration tests for complete system functionality."""

import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch

from smell_diffusion import SmellDiffusion, SafetyEvaluator, MultiModalGenerator
from smell_diffusion.core.molecule import Molecule
from smell_diffusion.utils.config import ConfigManager
from smell_diffusion.utils.caching import get_cache
from smell_diffusion.utils.logging import SmellDiffusionLogger


class TestEndToEndWorkflow:
    """Test complete end-to-end workflows."""
    
    @pytest.fixture
    def setup_system(self, temp_dir, mock_rdkit):
        """Set up complete system for testing."""
        # Configure test environment
        config_path = temp_dir / "config.yaml"
        config_manager = ConfigManager(config_path)
        config_manager.create_default_config(config_path)
        
        # Initialize components
        model = SmellDiffusion.from_pretrained("test-model")
        safety = SafetyEvaluator()
        
        return {
            'model': model,
            'safety': safety,
            'config_path': config_path,
            'temp_dir': temp_dir
        }
    
    def test_basic_generation_workflow(self, setup_system):
        """Test basic molecule generation workflow."""
        model = setup_system['model']
        safety = setup_system['safety']
        
        # Generate molecules
        prompt = "Fresh citrus fragrance with bergamot and lemon"
        molecules = model.generate(
            prompt=prompt,
            num_molecules=3,
            safety_filter=True
        )
        
        # Verify generation
        assert isinstance(molecules, list)
        assert len(molecules) >= 1  # At least one molecule should be generated
        
        # Evaluate safety for each molecule
        for mol in molecules:
            if mol and mol.is_valid:
                safety_profile = safety.evaluate(mol)
                assert safety_profile.score >= 0
                assert isinstance(safety_profile.ifra_compliant, bool)
    
    def test_multimodal_generation_workflow(self, setup_system, mock_rdkit):
        """Test multimodal generation workflow."""
        model = setup_system['model']
        
        # Create multimodal generator
        multimodal = MultiModalGenerator(model)
        
        # Generate with multiple modalities
        molecules = multimodal.generate(
            text="Fresh floral fragrance",
            reference_smiles="CC(C)=CCCC(C)=CCO",
            interpolation_weights={'text': 0.7, 'reference': 0.3},
            num_molecules=2
        )
        
        assert isinstance(molecules, list)
        assert len(molecules) >= 1
        
        # Verify molecules have descriptions
        for mol in molecules:
            if mol:
                assert mol.description is not None
    
    def test_safety_comprehensive_workflow(self, setup_system, mock_rdkit):
        """Test comprehensive safety evaluation workflow."""
        safety = setup_system['safety']
        
        # Create test molecule
        mol = Molecule("CC(C)=CCCC(C)=CCO")
        
        # Basic safety evaluation
        basic_safety = safety.evaluate(mol)
        assert isinstance(basic_safety.score, float)
        
        # Comprehensive safety evaluation
        comprehensive = safety.comprehensive_evaluation(mol)
        assert comprehensive.molecule_smiles == mol.smiles
        assert isinstance(comprehensive.regulatory_status, dict)
        assert isinstance(comprehensive.recommendations, list)
        
        # Verify consistency
        assert basic_safety.score == comprehensive.overall_score
    
    def test_configuration_workflow(self, setup_system):
        """Test configuration management workflow."""
        config_path = setup_system['config_path']
        
        # Load configuration
        config_manager = ConfigManager(config_path)
        config = config_manager.load_config()
        
        # Modify configuration
        config.generation.num_molecules = 10
        config.safety.min_safety_score = 80.0
        
        # Save configuration
        config_manager.save_config(config, config_path)
        
        # Reload and verify
        reloaded_config = config_manager.load_config()
        assert reloaded_config.generation.num_molecules == 10
        assert reloaded_config.safety.min_safety_score == 80.0
    
    def test_caching_workflow(self, setup_system, mock_rdkit):
        """Test caching integration workflow."""
        model = setup_system['model']
        cache = get_cache()
        
        # Clear cache
        cache.clear()
        
        # First generation (should cache)
        prompt = "Test fragrance for caching"
        molecules1 = model.generate(prompt=prompt, num_molecules=2)
        
        # Second generation (should use cache if implemented)
        molecules2 = model.generate(prompt=prompt, num_molecules=2)
        
        # Results should be consistent
        assert len(molecules1) == len(molecules2)
    
    def test_logging_workflow(self, setup_system, mock_rdkit):
        """Test logging integration workflow."""
        model = setup_system['model']
        logger = SmellDiffusionLogger("integration_test")
        
        # Generate with logging
        try:
            molecules = model.generate(
                prompt="Test logging workflow",
                num_molecules=2
            )
            
            # Log success
            logger.logger.info(f"Generated {len(molecules)} molecules successfully")
            
        except Exception as e:
            # Log error
            logger.log_error("integration_test", e)
            raise
    
    def test_error_handling_workflow(self, setup_system):
        """Test error handling across system components."""
        model = setup_system['model']
        safety = setup_system['safety']
        
        # Test with invalid input
        try:
            molecules = model.generate(
                prompt="",  # Empty prompt
                num_molecules=0  # Invalid count
            )
        except Exception:
            pass  # Expected to handle gracefully
        
        # Test safety with invalid molecule
        invalid_mol = Molecule("INVALID_SMILES")
        safety_result = safety.evaluate(invalid_mol)
        assert safety_result.score == 0.0
    
    def test_performance_workflow(self, setup_system, mock_rdkit):
        """Test system performance characteristics."""
        model = setup_system['model']
        
        import time
        
        # Time single generation
        start_time = time.time()
        molecules = model.generate("Performance test", num_molecules=1)
        single_time = time.time() - start_time
        
        # Time batch generation
        start_time = time.time()
        batch_molecules = model.generate("Performance test", num_molecules=5)
        batch_time = time.time() - start_time
        
        # Verify reasonable performance
        assert single_time < 10.0  # Should complete within 10 seconds
        assert batch_time < 30.0   # Batch should complete within 30 seconds
        
        # Verify outputs
        assert len(molecules) >= 1
        assert len(batch_molecules) >= 1


class TestSystemIntegration:
    """Test integration between different system components."""
    
    def test_model_safety_integration(self, mock_rdkit):
        """Test integration between model and safety components."""
        model = SmellDiffusion()
        model._is_loaded = True
        safety = SafetyEvaluator()
        
        # Generate molecules
        molecules = model.generate("Test integration", num_molecules=3, safety_filter=True)
        
        # Evaluate all molecules for safety
        safety_results = []
        for mol in molecules:
            if mol and mol.is_valid:
                result = safety.evaluate(mol)
                safety_results.append(result)
        
        # Verify safety filtering worked
        assert all(result.score > 0 for result in safety_results)
    
    def test_multimodal_safety_integration(self, mock_rdkit):
        """Test integration between multimodal generation and safety."""
        model = SmellDiffusion()
        model._is_loaded = True
        multimodal = MultiModalGenerator(model)
        safety = SafetyEvaluator()
        
        # Generate with multimodal
        molecules = multimodal.generate(
            text="Safe floral fragrance",
            num_molecules=2
        )
        
        # Evaluate safety
        for mol in molecules:
            if mol and mol.is_valid:
                safety_result = safety.evaluate(mol)
                assert isinstance(safety_result.score, float)
    
    def test_config_logging_integration(self, temp_dir):
        """Test integration between configuration and logging."""
        config_path = temp_dir / "test_config.yaml"
        config_manager = ConfigManager(config_path)
        
        # Create config with logging settings
        from smell_diffusion.utils.config import SmellDiffusionConfig
        config = SmellDiffusionConfig()
        config.logging.level = "DEBUG"
        config.logging.enable_file_logging = True
        config_manager.save_config(config, config_path)
        
        # Initialize logger with config
        logger = SmellDiffusionLogger("config_test", log_level=config.logging.level)
        logger.logger.debug("Test debug message")
        logger.logger.info("Test info message")
    
    def test_caching_performance_integration(self, mock_rdkit):
        """Test integration between caching and performance monitoring."""
        from smell_diffusion.utils.caching import cached, performance_monitor
        
        call_count = 0
        
        @cached(ttl=60)
        @performance_monitor("test_operation")
        def test_expensive_operation(param):
            nonlocal call_count
            call_count += 1
            return f"result_{param}"
        
        # First call
        result1 = test_expensive_operation("test")
        assert result1 == "result_test"
        assert call_count == 1
        
        # Second call (should use cache)
        result2 = test_expensive_operation("test")
        assert result2 == "result_test"
        assert call_count == 1  # Should not increment due to caching


class TestDataFlow:
    """Test data flow through the system."""
    
    def test_prompt_to_molecule_flow(self, mock_rdkit):
        """Test complete flow from prompt to molecule."""
        model = SmellDiffusion()
        model._is_loaded = True
        
        # Input: Text prompt
        prompt = "Fresh citrus with bergamot and lemon zest"
        
        # Process: Analyze prompt
        category_scores = model._analyze_prompt(prompt)
        assert 'citrus' in category_scores
        assert category_scores['citrus'] > 0
        
        # Process: Select molecules
        selected_smiles = model._select_molecules(category_scores, 2)
        assert len(selected_smiles) == 2
        
        # Output: Create molecule objects
        molecules = [Molecule(smiles) for smiles in selected_smiles]
        assert all(isinstance(mol, Molecule) for mol in molecules)
    
    def test_molecule_to_safety_flow(self, mock_rdkit):
        """Test flow from molecule to safety evaluation."""
        # Input: Molecule
        mol = Molecule("CC(C)=CCCC(C)=CCO")
        
        # Process: Calculate properties
        mw = mol.molecular_weight
        logp = mol.logp
        assert isinstance(mw, float)
        assert isinstance(logp, float)
        
        # Process: Predict fragrance notes
        notes = mol.fragrance_notes
        assert hasattr(notes, 'top')
        assert hasattr(notes, 'middle')
        assert hasattr(notes, 'base')
        
        # Output: Safety profile
        safety = mol.get_safety_profile()
        assert isinstance(safety.score, float)
        assert isinstance(safety.ifra_compliant, bool)
    
    def test_configuration_to_behavior_flow(self, temp_dir):
        """Test flow from configuration to system behavior."""
        # Input: Configuration
        config_path = temp_dir / "test_config.yaml"
        config_manager = ConfigManager(config_path)
        
        from smell_diffusion.utils.config import SmellDiffusionConfig
        config = SmellDiffusionConfig()
        config.generation.num_molecules = 7
        config.generation.safety_filter = False
        config_manager.save_config(config, config_path)
        
        # Process: Load configuration
        loaded_config = config_manager.load_config()
        
        # Output: Behavior matches configuration
        assert loaded_config.generation.num_molecules == 7
        assert loaded_config.generation.safety_filter is False


class TestSystemResilience:
    """Test system resilience and error recovery."""
    
    def test_partial_failure_recovery(self, mock_rdkit):
        """Test recovery from partial system failures."""
        model = SmellDiffusion()
        model._is_loaded = True
        
        # Simulate partial failure in molecule generation
        with patch.object(model, '_select_molecules') as mock_select:
            # Return some invalid SMILES
            mock_select.return_value = ["VALID_SMILES", "INVALID_SMILES", "CC(C)=CCCC(C)=CCO"]
            
            molecules = model.generate("Test resilience", num_molecules=3)
            
            # System should recover and return valid molecules
            assert isinstance(molecules, list)
            # Should have at least some valid molecules
            valid_count = sum(1 for mol in molecules if mol and mol.is_valid)
            assert valid_count >= 1
    
    def test_safety_filter_resilience(self, mock_rdkit):
        """Test resilience of safety filtering."""
        model = SmellDiffusion()
        model._is_loaded = True
        
        # Mock unsafe molecules
        with patch.object(Molecule, 'get_safety_profile') as mock_safety:
            from smell_diffusion.core.molecule import SafetyProfile
            mock_safety.return_value = SafetyProfile(
                score=30.0,  # Low safety score
                ifra_compliant=False,
                allergens=['test'],
                warnings=['unsafe']
            )
            
            molecules = model.generate("Test safety", num_molecules=3, safety_filter=True)
            
            # Should fallback to safe molecules
            assert len(molecules) >= 1
    
    def test_cache_failure_resilience(self, mock_rdkit):
        """Test resilience when caching fails."""
        model = SmellDiffusion()
        model._is_loaded = True
        
        # Mock cache failure
        with patch('smell_diffusion.utils.caching.get_cache') as mock_get_cache:
            mock_cache = Mock()
            mock_cache.get.side_effect = Exception("Cache failure")
            mock_cache.set.side_effect = Exception("Cache failure")
            mock_get_cache.return_value = mock_cache
            
            # System should continue working without cache
            molecules = model.generate("Test cache failure", num_molecules=2)
            assert len(molecules) >= 1
    
    def test_logging_failure_resilience(self, mock_rdkit):
        """Test resilience when logging fails."""
        from smell_diffusion.utils.logging import SmellDiffusionLogger
        
        # Mock logger failure
        with patch.object(SmellDiffusionLogger, 'log_generation_request') as mock_log:
            mock_log.side_effect = Exception("Logging failure")
            
            model = SmellDiffusion()
            model._is_loaded = True
            
            # System should continue working without logging
            molecules = model.generate("Test logging failure", num_molecules=1)
            assert len(molecules) >= 1


@pytest.mark.slow
class TestPerformanceBenchmarks:
    """Performance benchmark tests."""
    
    def test_single_generation_benchmark(self, mock_rdkit):
        """Benchmark single molecule generation."""
        model = SmellDiffusion()
        model._is_loaded = True
        
        import time
        
        # Warm up
        model.generate("Warmup", num_molecules=1)
        
        # Benchmark
        start_time = time.time()
        for _ in range(10):
            model.generate("Benchmark test", num_molecules=1)
        end_time = time.time()
        
        avg_time = (end_time - start_time) / 10
        assert avg_time < 1.0  # Should average less than 1 second per generation
    
    def test_batch_generation_benchmark(self, mock_rdkit):
        """Benchmark batch molecule generation."""
        model = SmellDiffusion()
        model._is_loaded = True
        
        import time
        
        # Benchmark batch generation
        start_time = time.time()
        molecules = model.generate("Batch benchmark", num_molecules=10)
        end_time = time.time()
        
        total_time = end_time - start_time
        assert total_time < 5.0  # Should complete within 5 seconds
        assert len(molecules) >= 5  # Should generate reasonable number
    
    def test_safety_evaluation_benchmark(self, mock_rdkit):
        """Benchmark safety evaluation performance."""
        from smell_diffusion.safety.evaluator import SafetyEvaluator
        
        evaluator = SafetyEvaluator()
        molecules = [Molecule("CC(C)=CCCC(C)=CCO") for _ in range(10)]
        
        import time
        
        # Benchmark safety evaluation
        start_time = time.time()
        for mol in molecules:
            evaluator.evaluate(mol)
        end_time = time.time()
        
        total_time = end_time - start_time
        avg_time = total_time / len(molecules)
        assert avg_time < 0.1  # Should average less than 0.1 seconds per evaluation