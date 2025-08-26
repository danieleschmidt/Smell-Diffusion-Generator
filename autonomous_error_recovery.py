#!/usr/bin/env python3
"""
Autonomous Error Recovery System
Self-healing capabilities with circuit breakers, retry logic, and adaptive recovery
"""

import asyncio
import time
import logging
import functools
import threading
import json
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import traceback
from collections import deque, defaultdict

# Mock imports for environments without dependencies
try:
    import numpy as np
except ImportError:
    class MockNumPy:
        @staticmethod
        def mean(x): return sum(x) / len(x) if x else 0
        @staticmethod
        def exponential(scale, size=None): return [scale * 2] * (size or 1)
    np = MockNumPy()


class ErrorSeverity(Enum):
    """Error severity classification"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class ErrorContext:
    """Comprehensive error context information"""
    error_type: str
    error_message: str
    stack_trace: str
    timestamp: float
    severity: ErrorSeverity
    component: str
    retry_count: int = 0
    recovery_attempted: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RecoveryStrategy:
    """Recovery strategy configuration"""
    name: str
    max_retries: int
    backoff_multiplier: float
    timeout: float
    recovery_function: Optional[Callable] = None
    fallback_function: Optional[Callable] = None
    conditions: Dict[str, Any] = field(default_factory=dict)


class AdaptiveCircuitBreaker:
    """
    Self-adapting circuit breaker with machine learning capabilities
    """
    
    def __init__(self, name: str, failure_threshold: int = 5, reset_timeout: float = 60.0):
        self.name = name
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time = 0
        self.success_count = 0
        self.call_count = 0
        self.response_times = deque(maxlen=100)
        self.lock = threading.Lock()
        
        # Adaptive parameters
        self.adaptive_threshold = failure_threshold
        self.learning_rate = 0.1
        
    def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection"""
        with self.lock:
            self.call_count += 1
            
            if self.state == CircuitState.OPEN:
                if time.time() - self.last_failure_time > self.reset_timeout:
                    self.state = CircuitState.HALF_OPEN
                    self.failure_count = 0
                else:
                    raise Exception(f"Circuit breaker {self.name} is OPEN")
            
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                
                # Record success
                execution_time = time.time() - start_time
                self.response_times.append(execution_time)
                self.success_count += 1
                
                if self.state == CircuitState.HALF_OPEN:
                    self.state = CircuitState.CLOSED
                    self.failure_count = 0
                
                # Adaptive learning - decrease threshold if performing well
                if self.success_count > 10:
                    self.adaptive_threshold = max(
                        self.failure_threshold,
                        self.adaptive_threshold - self.learning_rate
                    )
                
                return result
                
            except Exception as e:
                self.failure_count += 1
                self.last_failure_time = time.time()
                
                # Adaptive learning - increase threshold if failures are frequent
                self.adaptive_threshold = min(
                    self.failure_threshold * 2,
                    self.adaptive_threshold + self.learning_rate
                )
                
                if self.failure_count >= self.adaptive_threshold:
                    self.state = CircuitState.OPEN
                
                raise e
    
    def get_stats(self) -> Dict[str, Any]:
        """Get circuit breaker statistics"""
        with self.lock:
            avg_response_time = np.mean(list(self.response_times)) if self.response_times else 0
            
            return {
                "name": self.name,
                "state": self.state.value,
                "failure_count": self.failure_count,
                "success_count": self.success_count,
                "call_count": self.call_count,
                "failure_rate": self.failure_count / max(self.call_count, 1),
                "avg_response_time": avg_response_time,
                "adaptive_threshold": self.adaptive_threshold
            }


class AutonomousErrorRecovery:
    """
    Advanced error recovery system with autonomous learning capabilities
    """
    
    def __init__(self):
        self.logger = self._setup_logging()
        self.error_history: List[ErrorContext] = []
        self.recovery_strategies: Dict[str, RecoveryStrategy] = {}
        self.circuit_breakers: Dict[str, AdaptiveCircuitBreaker] = {}
        self.error_patterns: Dict[str, List[float]] = defaultdict(list)
        self.recovery_success_rates: Dict[str, float] = {}
        
        # Initialize default recovery strategies
        self._initialize_recovery_strategies()
        
        # Performance metrics
        self.metrics = {
            "total_errors": 0,
            "recovered_errors": 0,
            "failed_recoveries": 0,
            "avg_recovery_time": 0.0,
            "circuit_breaker_activations": 0
        }
    
    def _setup_logging(self) -> logging.Logger:
        """Setup error recovery logging"""
        logger = logging.getLogger("AutonomousErrorRecovery")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _initialize_recovery_strategies(self):
        """Initialize default recovery strategies"""
        
        # Exponential backoff strategy
        self.recovery_strategies["exponential_backoff"] = RecoveryStrategy(
            name="exponential_backoff",
            max_retries=5,
            backoff_multiplier=2.0,
            timeout=30.0,
            recovery_function=self._exponential_backoff_recovery,
            conditions={"applicable_errors": ["timeout", "connection", "temporary"]}
        )
        
        # Immediate retry strategy
        self.recovery_strategies["immediate_retry"] = RecoveryStrategy(
            name="immediate_retry",
            max_retries=3,
            backoff_multiplier=1.0,
            timeout=5.0,
            recovery_function=self._immediate_retry_recovery,
            conditions={"applicable_errors": ["transient", "network"]}
        )
        
        # Fallback strategy
        self.recovery_strategies["fallback"] = RecoveryStrategy(
            name="fallback",
            max_retries=1,
            backoff_multiplier=1.0,
            timeout=2.0,
            recovery_function=self._fallback_recovery,
            fallback_function=self._default_fallback,
            conditions={"applicable_errors": ["critical", "data"]}
        )
        
        # Adaptive strategy
        self.recovery_strategies["adaptive"] = RecoveryStrategy(
            name="adaptive",
            max_retries=3,
            backoff_multiplier=1.5,
            timeout=15.0,
            recovery_function=self._adaptive_recovery,
            conditions={"applicable_errors": ["performance", "resource"]}
        )
    
    def with_error_recovery(self, component: str = "default", strategy: str = "exponential_backoff"):
        """Decorator for automatic error recovery"""
        
        def decorator(func: Callable):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                return await self.execute_with_recovery(func, component, strategy, *args, **kwargs)
            
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                return asyncio.run(async_wrapper(*args, **kwargs))
            
            return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
        
        return decorator
    
    async def execute_with_recovery(self, func: Callable, component: str, strategy: str, *args, **kwargs) -> Any:
        """Execute function with autonomous error recovery"""
        recovery_start_time = time.time()
        
        # Get or create circuit breaker
        if component not in self.circuit_breakers:
            self.circuit_breakers[component] = AdaptiveCircuitBreaker(component)
        
        circuit_breaker = self.circuit_breakers[component]
        recovery_strategy = self.recovery_strategies.get(strategy, self.recovery_strategies["exponential_backoff"])
        
        retry_count = 0
        last_error = None
        
        while retry_count <= recovery_strategy.max_retries:
            try:
                # Execute with circuit breaker protection
                if asyncio.iscoroutinefunction(func):
                    result = await circuit_breaker.call(func, *args, **kwargs)
                else:
                    result = circuit_breaker.call(func, *args, **kwargs)
                
                # Success - update metrics
                if retry_count > 0:
                    self.metrics["recovered_errors"] += 1
                    recovery_time = time.time() - recovery_start_time
                    self._update_avg_recovery_time(recovery_time)
                
                return result
                
            except Exception as e:
                self.metrics["total_errors"] += 1
                last_error = e
                
                # Create error context
                error_context = ErrorContext(
                    error_type=type(e).__name__,
                    error_message=str(e),
                    stack_trace=traceback.format_exc(),
                    timestamp=time.time(),
                    severity=self._classify_error_severity(e),
                    component=component,
                    retry_count=retry_count,
                    metadata={"strategy": strategy, "args_count": len(args)}
                )
                
                self.error_history.append(error_context)
                
                # Log error with context
                self.logger.warning(
                    f"Error in {component} (attempt {retry_count + 1}/{recovery_strategy.max_retries + 1}): {str(e)}"
                )
                
                # Attempt recovery if retries remaining
                if retry_count < recovery_strategy.max_retries:
                    retry_count += 1
                    
                    # Apply recovery strategy
                    recovery_successful = await self._apply_recovery_strategy(
                        error_context, recovery_strategy, retry_count
                    )
                    
                    if not recovery_successful:
                        self.logger.error(f"Recovery strategy {strategy} failed for {component}")
                        break
                    
                else:
                    # Max retries reached
                    self.metrics["failed_recoveries"] += 1
                    self.logger.error(f"Max retries exceeded for {component}")
                    break
        
        # Final failure - attempt fallback if available
        if recovery_strategy.fallback_function:
            try:
                self.logger.info(f"Attempting fallback for {component}")
                fallback_result = recovery_strategy.fallback_function(*args, **kwargs)
                self.metrics["recovered_errors"] += 1
                return fallback_result
            except Exception as fallback_error:
                self.logger.error(f"Fallback failed for {component}: {str(fallback_error)}")
        
        # All recovery attempts failed
        raise last_error or Exception(f"All recovery attempts failed for {component}")
    
    async def _apply_recovery_strategy(self, error_context: ErrorContext, strategy: RecoveryStrategy, retry_count: int) -> bool:
        """Apply specific recovery strategy"""
        
        try:
            if strategy.recovery_function:
                await strategy.recovery_function(error_context, retry_count, strategy)
                return True
        except Exception as recovery_error:
            self.logger.error(f"Recovery function failed: {str(recovery_error)}")
            return False
        
        return False
    
    async def _exponential_backoff_recovery(self, error_context: ErrorContext, retry_count: int, strategy: RecoveryStrategy):
        """Exponential backoff recovery strategy"""
        backoff_time = min(
            strategy.backoff_multiplier ** (retry_count - 1),
            strategy.timeout
        )
        
        # Add jitter to prevent thundering herd
        jitter = np.exponential(0.1)[0] if hasattr(np, 'exponential') else 0.1
        total_wait = backoff_time + jitter
        
        self.logger.info(f"Exponential backoff: waiting {total_wait:.2f}s before retry {retry_count}")
        await asyncio.sleep(total_wait)
    
    async def _immediate_retry_recovery(self, error_context: ErrorContext, retry_count: int, strategy: RecoveryStrategy):
        """Immediate retry recovery strategy"""
        # Small delay to avoid immediate retry storms
        await asyncio.sleep(0.1)
        self.logger.info(f"Immediate retry attempt {retry_count}")
    
    async def _fallback_recovery(self, error_context: ErrorContext, retry_count: int, strategy: RecoveryStrategy):
        """Fallback recovery strategy"""
        self.logger.info(f"Applying fallback recovery for {error_context.component}")
        await asyncio.sleep(0.05)  # Brief pause before fallback
    
    async def _adaptive_recovery(self, error_context: ErrorContext, retry_count: int, strategy: RecoveryStrategy):
        """Adaptive recovery strategy that learns from error patterns"""
        
        # Analyze error patterns for this component
        component_errors = [
            err for err in self.error_history 
            if err.component == error_context.component
        ]
        
        # Adjust recovery parameters based on historical success
        success_rate = self.recovery_success_rates.get(error_context.component, 0.5)
        
        # Adaptive backoff based on success rate
        adaptive_backoff = strategy.backoff_multiplier * (2.0 - success_rate)
        backoff_time = min(adaptive_backoff ** (retry_count - 1), strategy.timeout)
        
        self.logger.info(f"Adaptive recovery: waiting {backoff_time:.2f}s (success rate: {success_rate:.2f})")
        await asyncio.sleep(backoff_time)
        
        # Learn from this recovery attempt
        self._update_recovery_learning(error_context.component, retry_count <= strategy.max_retries)
    
    def _default_fallback(self, *args, **kwargs) -> Any:
        """Default fallback implementation"""
        return {
            "status": "fallback_executed",
            "message": "Primary function failed, fallback response provided",
            "timestamp": time.time()
        }
    
    def _classify_error_severity(self, error: Exception) -> ErrorSeverity:
        """Classify error severity based on error type and message"""
        error_type = type(error).__name__
        error_message = str(error).lower()
        
        # Critical errors
        if error_type in ["SystemError", "MemoryError", "KeyboardInterrupt"]:
            return ErrorSeverity.CRITICAL
        
        # High severity errors
        if error_type in ["ValueError", "TypeError", "AttributeError"]:
            return ErrorSeverity.HIGH
        
        # Medium severity errors
        if "timeout" in error_message or "connection" in error_message:
            return ErrorSeverity.MEDIUM
        
        # Default to low severity
        return ErrorSeverity.LOW
    
    def _update_avg_recovery_time(self, recovery_time: float):
        """Update average recovery time metric"""
        current_avg = self.metrics.get("avg_recovery_time", 0.0)
        recovery_count = self.metrics.get("recovered_errors", 1)
        
        # Calculate new average
        self.metrics["avg_recovery_time"] = (
            (current_avg * (recovery_count - 1) + recovery_time) / recovery_count
        )
    
    def _update_recovery_learning(self, component: str, success: bool):
        """Update recovery learning based on success/failure"""
        current_rate = self.recovery_success_rates.get(component, 0.5)
        learning_rate = 0.1
        
        # Update success rate using exponential moving average
        new_rate = current_rate * (1 - learning_rate) + (1.0 if success else 0.0) * learning_rate
        self.recovery_success_rates[component] = new_rate
    
    def get_error_analytics(self) -> Dict[str, Any]:
        """Get comprehensive error analytics"""
        
        # Error distribution by type
        error_types = defaultdict(int)
        error_components = defaultdict(int)
        severity_distribution = defaultdict(int)
        
        for error in self.error_history:
            error_types[error.error_type] += 1
            error_components[error.component] += 1
            severity_distribution[error.severity.value] += 1
        
        # Circuit breaker stats
        circuit_stats = {
            name: breaker.get_stats() 
            for name, breaker in self.circuit_breakers.items()
        }
        
        # Recovery rate calculation
        total_errors = self.metrics["total_errors"]
        recovery_rate = (
            self.metrics["recovered_errors"] / max(total_errors, 1)
        )
        
        return {
            "summary": {
                "total_errors": total_errors,
                "recovered_errors": self.metrics["recovered_errors"],
                "failed_recoveries": self.metrics["failed_recoveries"],
                "recovery_rate": recovery_rate,
                "avg_recovery_time": self.metrics["avg_recovery_time"]
            },
            "error_distribution": {
                "by_type": dict(error_types),
                "by_component": dict(error_components),
                "by_severity": dict(severity_distribution)
            },
            "circuit_breakers": circuit_stats,
            "recovery_success_rates": dict(self.recovery_success_rates),
            "recent_errors": [
                {
                    "error_type": err.error_type,
                    "component": err.component,
                    "severity": err.severity.value,
                    "timestamp": err.timestamp,
                    "retry_count": err.retry_count
                }
                for err in self.error_history[-10:]  # Last 10 errors
            ]
        }
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get system health status based on error patterns"""
        
        recent_errors = [
            err for err in self.error_history 
            if time.time() - err.timestamp < 300  # Last 5 minutes
        ]
        
        critical_errors = [
            err for err in recent_errors 
            if err.severity == ErrorSeverity.CRITICAL
        ]
        
        # Health score calculation
        total_operations = max(
            sum(cb.call_count for cb in self.circuit_breakers.values()),
            1
        )
        error_rate = len(recent_errors) / max(total_operations, 1)
        
        health_score = max(0.0, 1.0 - (error_rate * 2.0))
        
        # Health status
        if critical_errors:
            status = "critical"
        elif health_score < 0.7:
            status = "degraded"
        elif health_score < 0.9:
            status = "warning"
        else:
            status = "healthy"
        
        return {
            "status": status,
            "health_score": health_score,
            "recent_error_count": len(recent_errors),
            "critical_error_count": len(critical_errors),
            "circuit_breakers_open": sum(
                1 for cb in self.circuit_breakers.values() 
                if cb.state == CircuitState.OPEN
            ),
            "recovery_capabilities": "operational"
        }


# Global error recovery instance
global_error_recovery = AutonomousErrorRecovery()


# Convenience decorators
def with_error_recovery(component: str = "default", strategy: str = "exponential_backoff"):
    """Decorator for automatic error recovery"""
    return global_error_recovery.with_error_recovery(component, strategy)


def with_circuit_breaker(component: str = "default"):
    """Decorator for circuit breaker protection"""
    def decorator(func: Callable):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            return await global_error_recovery.execute_with_recovery(
                func, component, "exponential_backoff", *args, **kwargs
            )
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            return asyncio.run(async_wrapper(*args, **kwargs))
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    
    return decorator


async def main():
    """Demo of autonomous error recovery system"""
    
    print("🛡️ Autonomous Error Recovery System Demo")
    print("=" * 50)
    
    # Demo function that fails sometimes
    @with_error_recovery(component="demo_function", strategy="adaptive")
    async def unreliable_function(success_rate: float = 0.3):
        """Function that fails randomly to demonstrate recovery"""
        import random
        
        if random.random() > success_rate:
            raise Exception(f"Random failure (success rate: {success_rate})")
        
        return {"result": "success", "timestamp": time.time()}
    
    # Test error recovery
    print("Testing error recovery capabilities...")
    
    for i in range(5):
        try:
            result = await unreliable_function(success_rate=0.4)
            print(f"✅ Attempt {i+1}: {result}")
        except Exception as e:
            print(f"❌ Attempt {i+1} failed: {str(e)}")
        
        await asyncio.sleep(0.5)
    
    # Display analytics
    analytics = global_error_recovery.get_error_analytics()
    health = global_error_recovery.get_health_status()
    
    print("\n📊 Error Recovery Analytics:")
    print(f"Recovery Rate: {analytics['summary']['recovery_rate']:.1%}")
    print(f"Average Recovery Time: {analytics['summary']['avg_recovery_time']:.2f}s")
    
    print("\n🏥 System Health:")
    print(f"Status: {health['status'].upper()}")
    print(f"Health Score: {health['health_score']:.2f}")
    
    return analytics


if __name__ == "__main__":
    asyncio.run(main())