"""
Enhanced Error Recovery and Resilience System
Advanced error handling, retry mechanisms, and system recovery for production deployment.
"""

import time
import asyncio
import random
import traceback
from typing import Any, Dict, List, Optional, Callable, Type, Union
from functools import wraps
from dataclasses import dataclass, field
from enum import Enum
import logging
from collections import defaultdict, deque

from .logging import SmellDiffusionLogger


class ErrorSeverity(Enum):
    """Error severity levels for categorized handling."""
    CRITICAL = "critical"
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


class RecoveryStrategy(Enum):
    """Recovery strategy types."""
    RETRY = "retry"
    FALLBACK = "fallback"
    CIRCUIT_BREAK = "circuit_break"
    GRACEFUL_DEGRADE = "graceful_degrade"
    FAIL_FAST = "fail_fast"


@dataclass
class ErrorContext:
    """Context information for error handling."""
    operation: str
    timestamp: float
    attempt: int
    error_type: str
    error_message: str
    stack_trace: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    recovery_strategy: Optional[RecoveryStrategy] = None


@dataclass
class RetryConfig:
    """Configuration for retry mechanisms."""
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    backoff_multiplier: float = 2.0
    jitter: bool = True
    exponential_backoff: bool = True
    retry_on: List[Type[Exception]] = field(default_factory=lambda: [Exception])
    stop_on: List[Type[Exception]] = field(default_factory=list)


class CircuitBreakerState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing fast
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    failure_threshold: int = 5
    recovery_timeout: float = 60.0
    success_threshold: int = 3  # For half-open state
    monitor_window: float = 300.0  # 5 minutes


class EnhancedCircuitBreaker:
    """Advanced circuit breaker with monitoring and statistics."""
    
    def __init__(self, config: CircuitBreakerConfig):
        self.config = config
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = 0
        self.logger = SmellDiffusionLogger("circuit_breaker")
        
        # Statistics
        self.total_calls = 0
        self.total_failures = 0
        self.state_transitions = defaultdict(int)
        self.failure_history = deque(maxlen=1000)
        
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection."""
        self.total_calls += 1
        
        # Check if circuit should be opened
        if self._should_open_circuit():
            self._transition_to_open()
            
        # Handle different states
        if self.state == CircuitBreakerState.OPEN:
            if self._should_attempt_reset():
                self._transition_to_half_open()
            else:
                raise CircuitBreakerOpenError("Circuit breaker is open")
                
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
            
        except Exception as e:
            self._on_failure(e)
            raise
    
    def _should_open_circuit(self) -> bool:
        """Check if circuit should be opened based on failure rate."""
        if self.failure_count >= self.config.failure_threshold:
            return True
            
        # Check failure rate in recent window
        current_time = time.time()
        recent_failures = [
            f for f in self.failure_history 
            if current_time - f['timestamp'] <= self.config.monitor_window
        ]
        
        if len(recent_failures) >= self.config.failure_threshold:
            failure_rate = len(recent_failures) / max(1, self.total_calls)
            return failure_rate > 0.5  # 50% failure rate threshold
            
        return False
    
    def _should_attempt_reset(self) -> bool:
        """Check if we should attempt to reset the circuit."""
        return (time.time() - self.last_failure_time) >= self.config.recovery_timeout
    
    def _transition_to_open(self):
        """Transition circuit to open state."""
        if self.state != CircuitBreakerState.OPEN:
            self.state = CircuitBreakerState.OPEN
            self.state_transitions[CircuitBreakerState.OPEN] += 1
            self.logger.logger.warning("Circuit breaker opened due to failures")
    
    def _transition_to_half_open(self):
        """Transition circuit to half-open state."""
        self.state = CircuitBreakerState.HALF_OPEN
        self.success_count = 0
        self.state_transitions[CircuitBreakerState.HALF_OPEN] += 1
        self.logger.logger.info("Circuit breaker transitioning to half-open")
    
    def _transition_to_closed(self):
        """Transition circuit to closed state."""
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.state_transitions[CircuitBreakerState.CLOSED] += 1
        self.logger.logger.info("Circuit breaker closed - normal operation restored")
    
    def _on_success(self):
        """Handle successful operation."""
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.config.success_threshold:
                self._transition_to_closed()
        elif self.state == CircuitBreakerState.CLOSED:
            # Reset failure count on success
            self.failure_count = max(0, self.failure_count - 1)
    
    def _on_failure(self, error: Exception):
        """Handle failed operation."""
        self.total_failures += 1
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        # Record failure in history
        self.failure_history.append({
            'timestamp': self.last_failure_time,
            'error_type': type(error).__name__,
            'error_message': str(error)
        })
        
        if self.state == CircuitBreakerState.HALF_OPEN:
            self._transition_to_open()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get circuit breaker statistics."""
        return {
            'state': self.state.value,
            'total_calls': self.total_calls,
            'total_failures': self.total_failures,
            'failure_rate': self.total_failures / max(1, self.total_calls),
            'current_failure_count': self.failure_count,
            'state_transitions': dict(self.state_transitions),
            'last_failure_time': self.last_failure_time
        }


class RetryManager:
    """Advanced retry manager with multiple strategies."""
    
    def __init__(self, config: RetryConfig):
        self.config = config
        self.logger = SmellDiffusionLogger("retry_manager")
        
    def retry(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with retry logic."""
        last_exception = None
        
        for attempt in range(1, self.config.max_attempts + 1):
            try:
                result = func(*args, **kwargs)
                if attempt > 1:
                    self.logger.logger.info(f"Function succeeded on attempt {attempt}")
                return result
                
            except Exception as e:
                last_exception = e
                
                # Check if we should stop retrying
                if any(isinstance(e, exc_type) for exc_type in self.config.stop_on):
                    self.logger.logger.info(f"Stopping retry due to {type(e).__name__}")
                    raise e
                
                # Check if we should retry this exception
                if not any(isinstance(e, exc_type) for exc_type in self.config.retry_on):
                    self.logger.logger.info(f"Not retrying {type(e).__name__}")
                    raise e
                
                if attempt < self.config.max_attempts:
                    delay = self._calculate_delay(attempt)
                    self.logger.logger.warning(
                        f"Attempt {attempt} failed with {type(e).__name__}: {str(e)}. "
                        f"Retrying in {delay:.2f}s"
                    )
                    time.sleep(delay)
                else:
                    self.logger.logger.error(
                        f"All {self.config.max_attempts} attempts failed. Last error: {str(e)}"
                    )
        
        raise last_exception
    
    def _calculate_delay(self, attempt: int) -> float:
        """Calculate delay for retry attempt."""
        if self.config.exponential_backoff:
            delay = self.config.base_delay * (self.config.backoff_multiplier ** (attempt - 1))
        else:
            delay = self.config.base_delay
            
        # Apply max delay cap
        delay = min(delay, self.config.max_delay)
        
        # Add jitter to prevent thundering herd
        if self.config.jitter:
            jitter_range = delay * 0.1  # 10% jitter
            delay += random.uniform(-jitter_range, jitter_range)
            
        return max(0, delay)


class ErrorRecoveryManager:
    """Comprehensive error recovery and resilience management."""
    
    def __init__(self):
        self.logger = SmellDiffusionLogger("error_recovery")
        self.circuit_breakers: Dict[str, EnhancedCircuitBreaker] = {}
        self.retry_managers: Dict[str, RetryManager] = {}
        self.error_history = deque(maxlen=10000)
        self.recovery_strategies: Dict[str, RecoveryStrategy] = {}
        
        # Default configurations
        self.default_circuit_config = CircuitBreakerConfig()
        self.default_retry_config = RetryConfig()
        
    def register_circuit_breaker(self, operation: str, config: Optional[CircuitBreakerConfig] = None):
        """Register a circuit breaker for an operation."""
        config = config or self.default_circuit_config
        self.circuit_breakers[operation] = EnhancedCircuitBreaker(config)
        self.logger.logger.info(f"Registered circuit breaker for operation: {operation}")
    
    def register_retry_manager(self, operation: str, config: Optional[RetryConfig] = None):
        """Register a retry manager for an operation."""
        config = config or self.default_retry_config
        self.retry_managers[operation] = RetryManager(config)
        self.logger.logger.info(f"Registered retry manager for operation: {operation}")
    
    def execute_with_recovery(self, operation: str, func: Callable, *args, **kwargs) -> Any:
        """Execute function with comprehensive error recovery."""
        context = ErrorContext(
            operation=operation,
            timestamp=time.time(),
            attempt=1,
            error_type="",
            error_message="",
            stack_trace=""
        )
        
        try:
            # Apply circuit breaker if registered
            if operation in self.circuit_breakers:
                circuit_breaker = self.circuit_breakers[operation]
                
                # Apply retry if registered
                if operation in self.retry_managers:
                    retry_manager = self.retry_managers[operation]
                    return circuit_breaker.call(retry_manager.retry, func, *args, **kwargs)
                else:
                    return circuit_breaker.call(func, *args, **kwargs)
            
            # Apply retry only if registered
            elif operation in self.retry_managers:
                retry_manager = self.retry_managers[operation]
                return retry_manager.retry(func, *args, **kwargs)
            
            # Execute directly if no recovery mechanisms registered
            else:
                return func(*args, **kwargs)
                
        except Exception as e:
            # Record error for analysis
            context.error_type = type(e).__name__
            context.error_message = str(e)
            context.stack_trace = traceback.format_exc()
            
            self._record_error(context)
            
            # Determine recovery strategy
            recovery_strategy = self._determine_recovery_strategy(context)
            context.recovery_strategy = recovery_strategy
            
            self.logger.log_error(f"error_recovery_{operation}", e, {
                "context": context.__dict__,
                "recovery_strategy": recovery_strategy.value if recovery_strategy else None
            })
            
            raise
    
    def _record_error(self, context: ErrorContext):
        """Record error for pattern analysis."""
        self.error_history.append(context)
    
    def _determine_recovery_strategy(self, context: ErrorContext) -> Optional[RecoveryStrategy]:
        """Determine appropriate recovery strategy based on error context."""
        # Analyze error patterns
        recent_errors = [
            err for err in self.error_history 
            if time.time() - err.timestamp <= 300  # Last 5 minutes
        ]
        
        error_types = [err.error_type for err in recent_errors]
        error_counts = defaultdict(int)
        for error_type in error_types:
            error_counts[error_type] += 1
        
        # Strategy determination logic
        if context.error_type in ["ConnectionError", "TimeoutError"]:
            return RecoveryStrategy.RETRY
        elif context.error_type in ["ValidationError", "ValueError"]:
            return RecoveryStrategy.FAIL_FAST
        elif error_counts[context.error_type] > 5:  # Frequent errors
            return RecoveryStrategy.CIRCUIT_BREAK
        else:
            return RecoveryStrategy.GRACEFUL_DEGRADE
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get comprehensive system health metrics."""
        current_time = time.time()
        recent_errors = [
            err for err in self.error_history 
            if current_time - err.timestamp <= 3600  # Last hour
        ]
        
        error_rate = len(recent_errors) / max(1, len(self.error_history))
        
        # Circuit breaker stats
        circuit_stats = {}
        for operation, cb in self.circuit_breakers.items():
            circuit_stats[operation] = cb.get_stats()
        
        return {
            'total_errors_recorded': len(self.error_history),
            'recent_errors_1h': len(recent_errors),
            'error_rate': error_rate,
            'circuit_breakers': circuit_stats,
            'registered_operations': {
                'circuit_breakers': list(self.circuit_breakers.keys()),
                'retry_managers': list(self.retry_managers.keys())
            }
        }


class CircuitBreakerOpenError(Exception):
    """Exception raised when circuit breaker is open."""
    pass


# Decorator for easy error recovery integration
def with_error_recovery(operation: str, 
                       circuit_breaker_config: Optional[CircuitBreakerConfig] = None,
                       retry_config: Optional[RetryConfig] = None):
    """Decorator to add error recovery to functions."""
    
    def decorator(func: Callable) -> Callable:
        # Get or create global error recovery manager
        if not hasattr(with_error_recovery, '_manager'):
            with_error_recovery._manager = ErrorRecoveryManager()
        
        manager = with_error_recovery._manager
        
        # Register recovery mechanisms
        if circuit_breaker_config:
            manager.register_circuit_breaker(operation, circuit_breaker_config)
        if retry_config:
            manager.register_retry_manager(operation, retry_config)
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            return manager.execute_with_recovery(operation, func, *args, **kwargs)
        
        return wrapper
    
    return decorator


# Global error recovery manager instance
global_error_recovery = ErrorRecoveryManager()

# Pre-configure common operations
global_error_recovery.register_circuit_breaker("molecule_generation", CircuitBreakerConfig(
    failure_threshold=3,
    recovery_timeout=30.0
))

global_error_recovery.register_retry_manager("molecule_generation", RetryConfig(
    max_attempts=3,
    base_delay=1.0,
    exponential_backoff=True,
    retry_on=[ConnectionError, TimeoutError, RuntimeError]
))

global_error_recovery.register_circuit_breaker("safety_evaluation", CircuitBreakerConfig(
    failure_threshold=5,
    recovery_timeout=60.0
))

global_error_recovery.register_retry_manager("safety_evaluation", RetryConfig(
    max_attempts=2,
    base_delay=0.5,
    stop_on=[ValueError, TypeError]
))