"""Configuration management for smell diffusion operations."""

import os
import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, asdict
from .logging import SmellDiffusionLogger


@dataclass
class ModelConfig:
    """Configuration for diffusion models."""
    model_name: str = "smell-diffusion-base-v1"
    cache_dir: Optional[str] = None
    device: str = "auto"  # auto, cpu, cuda
    precision: str = "float32"  # float16, float32
    batch_size: int = 1
    max_sequence_length: int = 512


@dataclass
class GenerationConfig:
    """Configuration for molecule generation."""
    num_molecules: int = 1
    guidance_scale: float = 7.5
    safety_filter: bool = True
    diversity_penalty: float = 0.5
    max_attempts: int = 5
    temperature: float = 1.0


@dataclass
class SafetyConfig:
    """Configuration for safety evaluation."""
    enable_comprehensive_check: bool = True
    ifra_compliance: bool = True
    allergen_screening: bool = True
    toxicity_prediction: bool = True
    environmental_assessment: bool = True
    min_safety_score: float = 70.0


@dataclass
class LoggingConfig:
    """Configuration for logging."""
    level: str = "INFO"
    enable_file_logging: bool = True
    enable_performance_monitoring: bool = True
    log_directory: Optional[str] = None
    max_log_size_mb: float = 100.0
    backup_count: int = 5


@dataclass
class APIConfig:
    """Configuration for API endpoints."""
    huggingface_token: Optional[str] = None
    openai_api_key: Optional[str] = None
    custom_endpoints: Dict[str, str] = None
    timeout_seconds: int = 30
    max_retries: int = 3


@dataclass
class SmellDiffusionConfig:
    """Main configuration class."""
    model: ModelConfig = None
    generation: GenerationConfig = None
    safety: SafetyConfig = None
    logging: LoggingConfig = None
    api: APIConfig = None
    
    def __post_init__(self):
        """Initialize default configurations."""
        if self.model is None:
            self.model = ModelConfig()
        if self.generation is None:
            self.generation = GenerationConfig()
        if self.safety is None:
            self.safety = SafetyConfig()
        if self.logging is None:
            self.logging = LoggingConfig()
        if self.api is None:
            self.api = APIConfig()


class ConfigManager:
    """Manages configuration loading, saving, and environment variable integration."""
    
    DEFAULT_CONFIG_LOCATIONS = [
        Path.home() / ".smell_diffusion" / "config.yaml",
        Path.home() / ".smell_diffusion" / "config.json",
        Path.cwd() / "smell_diffusion_config.yaml",
        Path.cwd() / "smell_diffusion_config.json",
    ]
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        """Initialize configuration manager."""
        self.config_path = Path(config_path) if config_path else None
        self.logger = SmellDiffusionLogger("config_manager")
        self._config = None
    
    def load_config(self) -> SmellDiffusionConfig:
        """Load configuration from file and environment variables."""
        if self._config is not None:
            return self._config
        
        config_data = {}
        
        # Try to load from file
        config_file = self._find_config_file()
        if config_file:
            config_data = self._load_config_file(config_file)
            self.logger.info(f"Loaded configuration from {config_file}")
        else:
            self.logger.info("No configuration file found, using defaults")
        
        # Override with environment variables
        env_overrides = self._load_env_variables()
        config_data = self._merge_configs(config_data, env_overrides)
        
        # Create configuration object
        self._config = self._create_config_object(config_data)
        
        # Validate configuration
        self._validate_config(self._config)
        
        return self._config
    
    def save_config(self, config: SmellDiffusionConfig, 
                   path: Optional[Union[str, Path]] = None) -> None:
        """Save configuration to file."""
        if path is None:
            path = self.config_path or self.DEFAULT_CONFIG_LOCATIONS[0]
        else:
            path = Path(path)
        
        # Ensure directory exists
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to dictionary
        config_dict = asdict(config)
        
        # Save based on file extension
        if path.suffix.lower() == '.json':
            with open(path, 'w') as f:
                json.dump(config_dict, f, indent=2)
        else:  # Default to YAML
            with open(path, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
        
        self.logger.info(f"Configuration saved to {path}")
    
    def _find_config_file(self) -> Optional[Path]:
        """Find configuration file in default locations."""
        if self.config_path and self.config_path.exists():
            return self.config_path
        
        for location in self.DEFAULT_CONFIG_LOCATIONS:
            if location.exists():
                return location
        
        return None
    
    def _load_config_file(self, config_path: Path) -> Dict[str, Any]:
        """Load configuration from file."""
        try:
            with open(config_path, 'r') as f:
                if config_path.suffix.lower() == '.json':
                    return json.load(f)
                else:  # YAML
                    return yaml.safe_load(f) or {}
        except Exception as e:
            self.logger.log_error("config_loading", e, {"path": str(config_path)})
            return {}
    
    def _load_env_variables(self) -> Dict[str, Any]:
        """Load configuration from environment variables."""
        env_config = {}
        
        # Model configuration
        if os.getenv('SMELL_DIFFUSION_MODEL_NAME'):
            env_config.setdefault('model', {})['model_name'] = os.getenv('SMELL_DIFFUSION_MODEL_NAME')
        
        if os.getenv('SMELL_DIFFUSION_DEVICE'):
            env_config.setdefault('model', {})['device'] = os.getenv('SMELL_DIFFUSION_DEVICE')
        
        if os.getenv('SMELL_DIFFUSION_CACHE_DIR'):
            env_config.setdefault('model', {})['cache_dir'] = os.getenv('SMELL_DIFFUSION_CACHE_DIR')
        
        # Generation configuration
        if os.getenv('SMELL_DIFFUSION_NUM_MOLECULES'):
            try:
                env_config.setdefault('generation', {})['num_molecules'] = int(os.getenv('SMELL_DIFFUSION_NUM_MOLECULES'))
            except ValueError:
                pass
        
        if os.getenv('SMELL_DIFFUSION_GUIDANCE_SCALE'):
            try:
                env_config.setdefault('generation', {})['guidance_scale'] = float(os.getenv('SMELL_DIFFUSION_GUIDANCE_SCALE'))
            except ValueError:
                pass
        
        if os.getenv('SMELL_DIFFUSION_SAFETY_FILTER'):
            env_config.setdefault('generation', {})['safety_filter'] = os.getenv('SMELL_DIFFUSION_SAFETY_FILTER').lower() == 'true'
        
        # Safety configuration
        if os.getenv('SMELL_DIFFUSION_MIN_SAFETY_SCORE'):
            try:
                env_config.setdefault('safety', {})['min_safety_score'] = float(os.getenv('SMELL_DIFFUSION_MIN_SAFETY_SCORE'))
            except ValueError:
                pass
        
        # Logging configuration
        if os.getenv('SMELL_DIFFUSION_LOG_LEVEL'):
            env_config.setdefault('logging', {})['level'] = os.getenv('SMELL_DIFFUSION_LOG_LEVEL')
        
        if os.getenv('SMELL_DIFFUSION_LOG_DIR'):
            env_config.setdefault('logging', {})['log_directory'] = os.getenv('SMELL_DIFFUSION_LOG_DIR')
        
        # API configuration
        if os.getenv('HUGGINGFACE_TOKEN'):
            env_config.setdefault('api', {})['huggingface_token'] = os.getenv('HUGGINGFACE_TOKEN')
        
        if os.getenv('OPENAI_API_KEY'):
            env_config.setdefault('api', {})['openai_api_key'] = os.getenv('OPENAI_API_KEY')
        
        return env_config
    
    def _merge_configs(self, base_config: Dict[str, Any], 
                      override_config: Dict[str, Any]) -> Dict[str, Any]:
        """Merge configuration dictionaries."""
        merged = base_config.copy()
        
        for key, value in override_config.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = self._merge_configs(merged[key], value)
            else:
                merged[key] = value
        
        return merged
    
    def _create_config_object(self, config_data: Dict[str, Any]) -> SmellDiffusionConfig:
        """Create configuration object from dictionary."""
        # Create sub-configurations
        model_config = ModelConfig(**(config_data.get('model', {})))
        generation_config = GenerationConfig(**(config_data.get('generation', {})))
        safety_config = SafetyConfig(**(config_data.get('safety', {})))
        logging_config = LoggingConfig(**(config_data.get('logging', {})))
        api_config = APIConfig(**(config_data.get('api', {})))
        
        return SmellDiffusionConfig(
            model=model_config,
            generation=generation_config,
            safety=safety_config,
            logging=logging_config,
            api=api_config
        )
    
    def _validate_config(self, config: SmellDiffusionConfig) -> None:
        """Validate configuration values."""
        # Validate model configuration
        if config.model.device not in ['auto', 'cpu', 'cuda']:
            raise ValueError(f"Invalid device: {config.model.device}")
        
        if config.model.precision not in ['float16', 'float32']:
            raise ValueError(f"Invalid precision: {config.model.precision}")
        
        # Validate generation configuration
        if config.generation.num_molecules < 1 or config.generation.num_molecules > 100:
            raise ValueError(f"num_molecules must be between 1 and 100")
        
        if config.generation.guidance_scale < 0.1 or config.generation.guidance_scale > 20.0:
            raise ValueError(f"guidance_scale must be between 0.1 and 20.0")
        
        # Validate safety configuration
        if config.safety.min_safety_score < 0 or config.safety.min_safety_score > 100:
            raise ValueError(f"min_safety_score must be between 0 and 100")
        
        # Validate logging configuration
        valid_log_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if config.logging.level.upper() not in valid_log_levels:
            raise ValueError(f"Invalid log level: {config.logging.level}")
    
    def create_default_config(self, path: Optional[Union[str, Path]] = None) -> None:
        """Create a default configuration file."""
        default_config = SmellDiffusionConfig()
        
        if path is None:
            path = self.DEFAULT_CONFIG_LOCATIONS[0]
        
        self.save_config(default_config, path)
        self.logger.info(f"Created default configuration at {path}")


# Global configuration manager
config_manager = ConfigManager()


def get_config() -> SmellDiffusionConfig:
    """Get the current configuration."""
    return config_manager.load_config()


def update_config(**kwargs) -> None:
    """Update configuration values."""
    config = get_config()
    
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
    
    # Validate and save
    config_manager._validate_config(config)
    config_manager._config = config