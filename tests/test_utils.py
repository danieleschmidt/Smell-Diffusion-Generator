"""Tests for utility modules."""

import pytest
import tempfile
import json
import time
import asyncio
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from smell_diffusion.utils.validation import (
    ValidationError, SMILESValidator, TextPromptValidator, 
    ParameterValidator, SafetyValidator, validate_inputs
)
from smell_diffusion.utils.logging import SmellDiffusionLogger, performance_monitor, HealthMonitor
from smell_diffusion.utils.config import ConfigManager, SmellDiffusionConfig, get_config
from smell_diffusion.utils.caching import InMemoryCache, DiskCache, HybridCache, cached
from smell_diffusion.utils.async_utils import (
    AsyncMoleculeGenerator, RateLimiter, AsyncBatchProcessor
)


class TestValidation:
    """Test validation utilities."""
    
    def test_smiles_validator_valid_smiles(self, sample_smiles, mock_rdkit):
        """Test SMILES validation with valid molecules."""
        for smiles in sample_smiles['valid']:
            assert SMILESValidator.is_valid_smiles(smiles)
            
            # Test sanitization
            sanitized = SMILESValidator.sanitize_smiles(smiles)
            assert isinstance(sanitized, str)
    
    def test_smiles_validator_invalid_smiles(self, sample_smiles):
        """Test SMILES validation with invalid molecules."""
        for smiles in sample_smiles['invalid']:
            assert not SMILESValidator.is_valid_smiles(smiles)
            
            # Test sanitization raises error
            with pytest.raises(ValidationError):
                SMILESValidator.sanitize_smiles(smiles)
    
    def test_molecular_weight_range_check(self, mock_rdkit):
        """Test molecular weight range validation."""
        # Mock different molecular weights
        with patch('smell_diffusion.utils.validation.Chem') as mock_chem:
            mock_mol = Mock()
            mock_chem.MolFromSmiles.return_value = mock_mol
            mock_chem.Descriptors.MolWt.return_value = 150.0
            
            assert SMILESValidator.check_molecular_weight_range("CCO")
            
            # Test out of range
            mock_chem.Descriptors.MolWt.return_value = 25.0  # Too low
            assert not SMILESValidator.check_molecular_weight_range("CCO")
            
            mock_chem.Descriptors.MolWt.return_value = 1500.0  # Too high
            assert not SMILESValidator.check_molecular_weight_range("CCO")
    
    def test_fragrance_suitability_check(self, mock_rdkit):
        """Test fragrance suitability assessment."""
        with patch('smell_diffusion.utils.validation.Chem') as mock_chem:
            mock_mol = Mock()
            mock_chem.MolFromSmiles.return_value = mock_mol
            mock_chem.Descriptors.MolWt.return_value = 150.0
            mock_chem.Descriptors.MolLogP.return_value = 2.5
            mock_mol.HasSubstructMatch.return_value = False
            
            result = SMILESValidator.check_fragrance_suitability("CCO")
            
            assert isinstance(result, dict)
            assert 'suitable' in result
            assert 'molecular_weight' in result
            assert 'logp' in result
            assert 'issues' in result
            assert result['suitable'] is True  # Good properties
            
            # Test unsuitable molecule
            mock_chem.Descriptors.MolWt.return_value = 25.0  # Too low
            result = SMILESValidator.check_fragrance_suitability("CCO")
            assert result['suitable'] is False
            assert len(result['issues']) > 0
    
    def test_text_prompt_validator(self):
        """Test text prompt validation."""
        # Valid prompts
        valid_prompts = [
            "Fresh citrus fragrance",
            "Beautiful floral bouquet with rose and jasmine",
            "Woody base notes with cedar and sandalwood"
        ]
        
        for prompt in valid_prompts:
            TextPromptValidator.validate_prompt(prompt)  # Should not raise
            sanitized = TextPromptValidator.sanitize_prompt(prompt)
            assert isinstance(sanitized, str)
            assert len(sanitized) > 0
        
        # Invalid prompts
        with pytest.raises(ValidationError):
            TextPromptValidator.validate_prompt("")  # Too short
        
        with pytest.raises(ValidationError):
            TextPromptValidator.validate_prompt("x" * 1001)  # Too long
        
        with pytest.raises(ValidationError):
            TextPromptValidator.validate_prompt(123)  # Not string
    
    def test_parameter_validator(self):
        """Test parameter validation."""
        # Valid parameters
        ParameterValidator.validate_num_molecules(5)
        ParameterValidator.validate_guidance_scale(7.5)
        
        # Invalid num_molecules
        with pytest.raises(ValidationError):
            ParameterValidator.validate_num_molecules(0)
        
        with pytest.raises(ValidationError):
            ParameterValidator.validate_num_molecules(101)
        
        with pytest.raises(ValidationError):
            ParameterValidator.validate_num_molecules("5")
        
        # Invalid guidance_scale
        with pytest.raises(ValidationError):
            ParameterValidator.validate_guidance_scale(0.05)
        
        with pytest.raises(ValidationError):
            ParameterValidator.validate_guidance_scale(25.0)
    
    def test_interpolation_weights_validation(self):
        """Test interpolation weights validation."""
        # Valid weights
        valid_weights = {'text': 0.5, 'image': 0.3, 'reference': 0.2}
        ParameterValidator.validate_interpolation_weights(valid_weights)
        
        # Invalid weights - wrong keys
        with pytest.raises(ValidationError):
            ParameterValidator.validate_interpolation_weights({'invalid': 1.0})
        
        # Invalid weights - don't sum to 1
        with pytest.raises(ValidationError):
            ParameterValidator.validate_interpolation_weights({'text': 0.5, 'image': 0.8})
        
        # Invalid weights - negative values
        with pytest.raises(ValidationError):
            ParameterValidator.validate_interpolation_weights({'text': -0.1, 'image': 1.1})
    
    def test_safety_validator(self, mock_rdkit):
        """Test safety constraint validation."""
        # Test prohibited structures check
        violations = SafetyValidator.check_prohibited_structures("CCO")
        assert isinstance(violations, list)
        
        # Test with invalid SMILES
        violations = SafetyValidator.check_prohibited_structures("INVALID")
        assert "Invalid SMILES structure" in violations
        
        # Test enforcement
        SafetyValidator.enforce_safety_constraints("CCO")  # Should not raise
    
    def test_validate_inputs_decorator(self):
        """Test validate_inputs decorator."""
        @validate_inputs
        def test_function(prompt, num_molecules=1, guidance_scale=7.5):
            return f"Generated with {prompt}"
        
        # Valid call
        result = test_function("Fresh citrus", num_molecules=3, guidance_scale=8.0)
        assert "Generated with Fresh citrus" in result
        
        # Invalid call
        with pytest.raises(ValidationError):
            test_function("", num_molecules=0)  # Empty prompt, invalid num_molecules


class TestLogging:
    """Test logging utilities."""
    
    def test_logger_initialization(self):
        """Test logger initialization."""
        logger = SmellDiffusionLogger("test_logger")
        assert logger.logger.name == "test_logger"
    
    def test_generation_logging(self):
        """Test generation request/result logging."""
        logger = SmellDiffusionLogger("test")
        
        # Log request
        request_id = logger.log_generation_request(
            "test prompt", 3, True, extra_param="test"
        )
        assert isinstance(request_id, str)
        assert request_id.startswith("gen_")
        
        # Log result
        mock_molecules = [Mock(), Mock()]
        logger.log_generation_result(request_id, mock_molecules, 2.5)
    
    def test_safety_logging(self):
        """Test safety evaluation logging."""
        logger = SmellDiffusionLogger("test")
        
        logger.log_safety_evaluation(
            "CCO", 85.0, ["allergen1"], ["warning1"]
        )
    
    def test_error_logging(self):
        """Test error logging."""
        logger = SmellDiffusionLogger("test")
        
        test_error = Exception("Test error")
        context = {"test": "context"}
        
        logger.log_error("test_operation", test_error, context)
    
    def test_performance_monitor_decorator(self):
        """Test performance monitoring decorator."""
        @performance_monitor("test_operation")
        def test_function():
            time.sleep(0.01)  # Small delay
            return "success"
        
        result = test_function()
        assert result == "success"
        
        # Test with exception
        @performance_monitor("test_operation_error")
        def failing_function():
            raise ValueError("Test error")
        
        with pytest.raises(ValueError):
            failing_function()
    
    def test_health_monitor(self):
        """Test health monitoring."""
        monitor = HealthMonitor()
        
        # Test recording
        monitor.record_generation()
        monitor.record_error()
        
        # Test status
        status = monitor.get_health_status()
        assert isinstance(status, dict)
        assert 'status' in status
        assert 'uptime_seconds' in status
        assert 'total_generations' in status
        assert 'total_errors' in status
        assert status['total_generations'] == 1
        assert status['total_errors'] == 1


class TestConfiguration:
    """Test configuration management."""
    
    def test_config_manager_initialization(self, temp_dir):
        """Test configuration manager initialization."""
        config_path = temp_dir / "test_config.yaml"
        manager = ConfigManager(config_path)
        assert manager.config_path == config_path
    
    def test_default_config_creation(self, temp_dir):
        """Test creating default configuration."""
        config_path = temp_dir / "test_config.yaml"
        manager = ConfigManager()
        
        manager.create_default_config(config_path)
        assert config_path.exists()
    
    def test_config_loading_and_saving(self, temp_dir):
        """Test configuration loading and saving."""
        config_path = temp_dir / "test_config.yaml"
        manager = ConfigManager(config_path)
        
        # Create and save config
        config = SmellDiffusionConfig()
        config.model.model_name = "test-model"
        manager.save_config(config, config_path)
        
        # Load config
        loaded_config = manager.load_config()
        assert loaded_config.model.model_name == "test-model"
    
    def test_environment_variable_override(self, temp_dir):
        """Test environment variable configuration override."""
        with patch.dict('os.environ', {'SMELL_DIFFUSION_MODEL_NAME': 'env-model'}):
            manager = ConfigManager()
            config = manager.load_config()
            assert config.model.model_name == 'env-model'
    
    def test_config_validation(self, temp_dir):
        """Test configuration validation."""
        manager = ConfigManager()
        
        # Valid config
        config = SmellDiffusionConfig()
        manager._validate_config(config)  # Should not raise
        
        # Invalid config
        config.model.device = "invalid_device"
        with pytest.raises(ValueError):
            manager._validate_config(config)
    
    def test_get_config_global(self):
        """Test global configuration getter."""
        config = get_config()
        assert isinstance(config, SmellDiffusionConfig)


class TestCaching:
    """Test caching utilities."""
    
    def test_in_memory_cache(self):
        """Test in-memory cache functionality."""
        cache = InMemoryCache(max_size=3, default_ttl=1)
        
        # Test basic operations
        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"
        assert cache.get("nonexistent") is None
        
        # Test TTL expiration
        cache.set("key2", "value2", ttl=0.1)
        time.sleep(0.2)
        assert cache.get("key2") is None
        
        # Test LRU eviction
        cache.set("key3", "value3")
        cache.set("key4", "value4")
        cache.set("key5", "value5")  # Should evict oldest
        
        stats = cache.get_stats()
        assert stats['size'] <= 3
    
    def test_disk_cache(self, temp_dir):
        """Test disk cache functionality."""
        cache = DiskCache(cache_dir=temp_dir, max_size_mb=1.0)
        
        # Test basic operations
        cache.set("key1", {"data": "value1"})
        result = cache.get("key1")
        assert result == {"data": "value1"}
        
        # Test nonexistent key
        assert cache.get("nonexistent") is None
        
        # Test metadata
        assert "key1" in cache.metadata
    
    def test_hybrid_cache(self, temp_dir):
        """Test hybrid cache functionality."""
        cache = HybridCache(memory_size=2, disk_size_mb=1.0)
        
        # Test basic operations
        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"
        
        # Test persistence
        cache.set("key2", "value2", persist=True)
        assert cache.get("key2") == "value2"
        
        # Test stats
        stats = cache.get_stats()
        assert 'memory' in stats
        assert 'disk' in stats
    
    def test_cached_decorator(self, temp_dir):
        """Test cached decorator functionality."""
        call_count = 0
        
        @cached(ttl=1, persist=False)
        def expensive_function(x):
            nonlocal call_count
            call_count += 1
            return x * 2
        
        # First call
        result1 = expensive_function(5)
        assert result1 == 10
        assert call_count == 1
        
        # Second call (should use cache)
        result2 = expensive_function(5)
        assert result2 == 10
        assert call_count == 1  # Not incremented
        
        # Different argument
        result3 = expensive_function(10)
        assert result3 == 20
        assert call_count == 2


class TestAsyncUtils:
    """Test asynchronous utilities."""
    
    def test_rate_limiter(self):
        """Test rate limiter functionality."""
        limiter = RateLimiter(max_calls=2, time_window=1.0)
        
        # Should allow first two calls
        assert limiter.is_allowed()
        assert limiter.is_allowed()
        
        # Should deny third call
        assert not limiter.is_allowed()
        
        # Test stats
        stats = limiter.get_stats()
        assert stats['current_calls'] == 2
        assert stats['max_calls'] == 2
    
    @pytest.mark.asyncio
    async def test_async_molecule_generator(self, mock_model):
        """Test async molecule generator."""
        async_gen = AsyncMoleculeGenerator(mock_model, max_workers=2)
        
        # Test single generation
        result = await async_gen.generate_async("test prompt", num_molecules=1)
        assert isinstance(result, list)
        
        # Test batch generation
        prompts = ["prompt1", "prompt2", "prompt3"]
        results = await async_gen.batch_generate_async(prompts, num_molecules=1)
        assert len(results) == 3
        assert all(isinstance(result, list) for result in results)
    
    @pytest.mark.asyncio
    async def test_async_batch_processor(self):
        """Test async batch processor."""
        processor = AsyncBatchProcessor(batch_size=2, max_concurrent_batches=2)
        
        # Mock processing function
        async def mock_processor(item):
            await asyncio.sleep(0.01)  # Simulate work
            return f"processed_{item}"
        
        items = [1, 2, 3, 4, 5]
        results = await processor.process_items(items, mock_processor)
        
        assert len(results) == 5
        assert all(result.startswith("processed_") for result in results)
    
    @pytest.mark.asyncio
    async def test_rate_limiter_async(self):
        """Test async rate limiter functionality."""
        limiter = RateLimiter(max_calls=2, time_window=10.0)
        
        # First two calls should be immediate
        start_time = time.time()
        await limiter.wait_if_needed()
        await limiter.wait_if_needed()
        end_time = time.time()
        
        # Should be very fast
        assert (end_time - start_time) < 0.1


class TestUtilityIntegration:
    """Integration tests for utilities."""
    
    def test_logging_with_caching(self, temp_dir):
        """Test logging with caching integration."""
        logger = SmellDiffusionLogger("test")
        cache = HybridCache()
        
        # Log cache operations
        cache.set("test_key", "test_value")
        result = cache.get("test_key")
        
        assert result == "test_value"
    
    def test_validation_with_config(self, temp_dir):
        """Test validation with configuration."""
        config = SmellDiffusionConfig()
        config.generation.num_molecules = 5
        
        # Validate against config
        ParameterValidator.validate_num_molecules(config.generation.num_molecules)
    
    def test_comprehensive_error_handling(self):
        """Test comprehensive error handling across utilities."""
        logger = SmellDiffusionLogger("test")
        
        # Test various error scenarios
        try:
            raise ValidationError("Test validation error")
        except ValidationError as e:
            logger.log_error("validation", e)
        
        try:
            SMILESValidator.sanitize_smiles("INVALID")  
        except ValidationError as e:
            logger.log_error("smiles_validation", e)