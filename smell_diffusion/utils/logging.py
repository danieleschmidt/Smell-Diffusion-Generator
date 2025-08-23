"""Comprehensive logging and monitoring for smell diffusion operations."""

import logging
import time
import functools
from typing import Any, Dict, Optional, Callable
from pathlib import Path
import json
import sys
from datetime import datetime


class SmellDiffusionLogger:
    """Enhanced logger for smell diffusion operations."""
    
    def __init__(self, name: str = "smell_diffusion", log_level: str = "INFO"):
        """Initialize logger with proper formatting and handlers."""
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, log_level.upper()))
        
        # Prevent duplicate handlers
        if not self.logger.handlers:
            self._setup_handlers()
    
    def _setup_handlers(self) -> None:
        """Set up console and file handlers."""
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Console handler with color support
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # File handler for persistent logging
        log_dir = Path.home() / ".smell_diffusion" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(
            log_dir / f"smell_diffusion_{datetime.now().strftime('%Y%m%d')}.log"
        )
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
    
    def log_generation_request(self, prompt: str, num_molecules: int, 
                             safety_filter: bool, **kwargs) -> str:
        """Log molecule generation request."""
        request_id = f"gen_{int(time.time() * 1000)}"
        
        self.logger.info(
            f"Generation request [{request_id}]: prompt='{prompt}', "
            f"num_molecules={num_molecules}, safety_filter={safety_filter}"
        )
        
        if kwargs:
            self.logger.debug(f"Additional parameters [{request_id}]: {kwargs}")
        
        return request_id
    
    def log_generation_result(self, request_id: str, molecules: list, 
                            generation_time: float) -> None:
        """Log generation results."""
        valid_count = sum(1 for mol in molecules if mol and mol.is_valid)
        
        self.logger.info(
            f"Generation complete [{request_id}]: {valid_count}/{len(molecules)} "
            f"valid molecules in {generation_time:.2f}s"
        )
        
        # Log detailed molecule info at debug level
        for i, mol in enumerate(molecules):
            if mol:
                self.logger.debug(
                    f"Molecule {i+1} [{request_id}]: SMILES={mol.smiles}, "
                    f"MW={mol.molecular_weight:.1f}, valid={mol.is_valid}"
                )
    
    def log_safety_evaluation(self, molecule_smiles: str, safety_score: float,
                            allergens: list, warnings: list) -> None:
        """Log safety evaluation results."""
        self.logger.info(
            f"Safety evaluation: SMILES={molecule_smiles}, score={safety_score:.1f}/100"
        )
        
        if allergens:
            self.logger.warning(f"Allergens detected: {', '.join(allergens)}")
        
        if warnings:
            self.logger.warning(f"Safety warnings: {'; '.join(warnings)}")
    
    def log_error(self, operation: str, error: Exception, context: Dict[str, Any] = None) -> None:
        """Log errors with context."""
        error_msg = f"Error in {operation}: {str(error)}"
        
        if context:
            error_msg += f" | Context: {context}"
        
        self.logger.error(error_msg, exc_info=True)
    
    def log_performance_metrics(self, operation: str, duration: float, 
                              memory_usage: Optional[float] = None,
                              additional_metrics: Dict[str, Any] = None) -> None:
        """Log performance metrics."""
        metrics = {
            "operation": operation,
            "duration_seconds": duration,
            "timestamp": datetime.now().isoformat()
        }
        
        if memory_usage:
            metrics["memory_mb"] = memory_usage
        
        if additional_metrics:
            metrics.update(additional_metrics)
        
        self.logger.info(f"Performance metrics: {json.dumps(metrics)}")


def get_logger(name: str = "smell_diffusion", log_level: str = "INFO") -> SmellDiffusionLogger:
    """Get or create a logger instance."""
    return SmellDiffusionLogger(name, log_level)


def performance_monitor(operation_name: str = None):
    """Decorator to monitor function performance."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            logger = SmellDiffusionLogger()
            op_name = operation_name or f"{func.__module__}.{func.__name__}"
            
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                
                # Log successful completion
                logger.log_performance_metrics(op_name, duration)
                return result
                
            except Exception as e:
                duration = time.time() - start_time
                
                # Log error with timing
                context = {
                    "function": func.__name__,
                    "duration": duration,
                    "args_count": len(args),
                    "kwargs_keys": list(kwargs.keys())
                }
                logger.log_error(op_name, e, context)
                raise
                
        return wrapper
    return decorator


def log_molecule_generation(func: Callable) -> Callable:
    """Decorator specifically for molecule generation functions."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = SmellDiffusionLogger()
        
        # Extract prompt from arguments
        prompt = kwargs.get('prompt') or (args[1] if len(args) > 1 else 'Unknown')
        num_molecules = kwargs.get('num_molecules', 1)
        safety_filter = kwargs.get('safety_filter', True)
        
        # Log request - filter out duplicate kwargs
        filtered_kwargs = {k: v for k, v in kwargs.items() 
                          if k not in ['prompt', 'num_molecules', 'safety_filter']}
        request_id = logger.log_generation_request(
            prompt, num_molecules, safety_filter, **filtered_kwargs
        )
        
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            generation_time = time.time() - start_time
            
            # Handle both single molecule and list results
            molecules = result if isinstance(result, list) else [result] if result else []
            
            # Log results
            logger.log_generation_result(request_id, molecules, generation_time)
            
            return result
            
        except Exception as e:
            generation_time = time.time() - start_time
            context = {
                "request_id": request_id,
                "prompt": prompt,
                "generation_time": generation_time
            }
            logger.log_error("molecule_generation", e, context)
            raise
            
    return wrapper


class HealthMonitor:
    """Monitor system health and performance."""
    
    def __init__(self):
        """Initialize health monitor."""
        self.logger = SmellDiffusionLogger("health_monitor")
        self.start_time = time.time()
        self.generation_count = 0
        self.error_count = 0
        
    def record_generation(self) -> None:
        """Record a successful generation."""
        self.generation_count += 1
        
    def record_error(self) -> None:
        """Record an error."""
        self.error_count += 1
        
    def get_health_status(self) -> Dict[str, Any]:
        """Get current health status."""
        uptime = time.time() - self.start_time
        
        status = {
            "status": "healthy" if self.error_count < 10 else "degraded",
            "uptime_seconds": uptime,
            "total_generations": self.generation_count,
            "total_errors": self.error_count,
            "error_rate": self.error_count / max(self.generation_count, 1),
            "timestamp": datetime.now().isoformat()
        }
        
        return status
    
    def log_health_check(self) -> None:
        """Log current health status."""
        status = self.get_health_status()
        self.logger.info(f"Health check: {json.dumps(status)}")


# Global health monitor instance
health_monitor = HealthMonitor()