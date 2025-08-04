"""Asynchronous utilities for concurrent processing."""

import asyncio
import concurrent.futures
import threading
import time
from typing import List, Callable, Any, Optional, Dict, Awaitable
from functools import wraps

from .logging import SmellDiffusionLogger, performance_monitor
from .caching import get_cache


class AsyncMoleculeGenerator:
    """Asynchronous wrapper for molecule generation."""
    
    def __init__(self, base_generator, max_workers: int = 4):
        """Initialize async generator."""
        self.base_generator = base_generator
        self.max_workers = max_workers
        self.logger = SmellDiffusionLogger("async_generator")
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
    
    async def generate_async(self, prompt: str, num_molecules: int = 1, **kwargs) -> List[Any]:
        """Generate molecules asynchronously."""
        loop = asyncio.get_event_loop()
        
        try:
            result = await loop.run_in_executor(
                self.executor,
                self._generate_sync,
                prompt,
                num_molecules,
                kwargs
            )
            return result
        except Exception as e:
            self.logger.log_error("async_generation", e, {"prompt": prompt})
            return []
    
    def _generate_sync(self, prompt: str, num_molecules: int, kwargs: Dict[str, Any]) -> List[Any]:
        """Synchronous generation wrapper."""
        result = self.base_generator.generate(
            prompt=prompt,
            num_molecules=num_molecules,
            **kwargs
        )
        return result if isinstance(result, list) else [result] if result else []
    
    async def batch_generate_async(self, prompts: List[str], **kwargs) -> List[List[Any]]:
        """Generate molecules for multiple prompts concurrently."""
        tasks = []
        
        for prompt in prompts:
            task = self.generate_async(prompt, **kwargs)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions in results
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self.logger.log_error("batch_item_error", result, {"prompt_index": i})
                processed_results.append([])
            else:
                processed_results.append(result)
        
        return processed_results
    
    def __del__(self):
        """Cleanup executor on deletion."""
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)


class ConcurrentSafetyEvaluator:
    """Concurrent safety evaluation for multiple molecules."""
    
    def __init__(self, base_evaluator, max_workers: int = 8):
        """Initialize concurrent evaluator."""
        self.base_evaluator = base_evaluator
        self.max_workers = max_workers
        self.logger = SmellDiffusionLogger("concurrent_safety")
    
    @performance_monitor("concurrent_safety_evaluation")
    def evaluate_batch(self, molecules: List[Any]) -> List[Dict[str, Any]]:
        """Evaluate safety of multiple molecules concurrently."""
        if not molecules:
            return []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all evaluation tasks
            future_to_molecule = {
                executor.submit(self._safe_evaluate, mol): mol
                for mol in molecules if mol is not None
            }
            
            results = []
            for future in concurrent.futures.as_completed(future_to_molecule):
                mol = future_to_molecule[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    self.logger.log_error("safety_evaluation", e, {"molecule": str(mol)})
                    results.append(None)
            
            return results
    
    def _safe_evaluate(self, molecule) -> Optional[Dict[str, Any]]:
        """Safely evaluate a single molecule."""
        try:
            if hasattr(molecule, 'get_safety_profile'):
                safety = molecule.get_safety_profile()
                return {
                    "molecule_smiles": molecule.smiles,
                    "score": safety.score,
                    "ifra_compliant": safety.ifra_compliant,
                    "allergens": safety.allergens,
                    "warnings": safety.warnings
                }
            return None
        except Exception as e:
            self.logger.log_error("single_safety_evaluation", e)
            return None


class AsyncCacheManager:
    """Asynchronous cache operations."""
    
    def __init__(self):
        """Initialize async cache manager."""
        self.cache = get_cache()
        self.logger = SmellDiffusionLogger("async_cache")
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
    
    async def get_async(self, key: str) -> Optional[Any]:
        """Get value from cache asynchronously."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, self.cache.get, key)
    
    async def set_async(self, key: str, value: Any, ttl: int = 3600, persist: bool = True) -> None:
        """Set value in cache asynchronously."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            self.executor,
            lambda: self.cache.set(key, value, ttl, persist)
        )
    
    async def preload_cache(self, keys_and_generators: List[tuple]) -> None:
        """Preload cache with multiple key-value pairs."""
        tasks = []
        
        for key, value_generator in keys_and_generators:
            task = self._preload_single(key, value_generator)
            tasks.append(task)
        
        await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _preload_single(self, key: str, value_generator: Callable) -> None:
        """Preload single cache entry."""
        try:
            # Check if already cached
            existing = await self.get_async(key)
            if existing is not None:
                return
            
            # Generate value
            loop = asyncio.get_event_loop()
            value = await loop.run_in_executor(self.executor, value_generator)
            
            # Cache it
            await self.set_async(key, value)
            
        except Exception as e:
            self.logger.log_error("cache_preload", e, {"key": key})


def async_cached(ttl: int = 3600, persist: bool = True):
    """Decorator for async caching."""
    def decorator(func: Callable) -> Callable:
        cache_manager = AsyncCacheManager()
        
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Generate cache key
            from .caching import cache_key
            key = f"{func.__module__}.{func.__name__}:{cache_key(*args, **kwargs)}"
            
            # Try to get from cache
            cached_result = await cache_manager.get_async(key)
            if cached_result is not None:
                return cached_result
            
            # Execute function
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(None, func, *args, **kwargs)
            
            # Cache result
            await cache_manager.set_async(key, result, ttl, persist)
            
            return result
        
        return wrapper
    return decorator


class RateLimiter:
    """Rate limiter for API calls and resource-intensive operations."""
    
    def __init__(self, max_calls: int = 10, time_window: float = 60.0):
        """Initialize rate limiter."""
        self.max_calls = max_calls
        self.time_window = time_window
        self.calls = []
        self.lock = threading.Lock()
        self.logger = SmellDiffusionLogger("rate_limiter")
    
    def is_allowed(self) -> bool:
        """Check if a call is allowed under rate limits."""
        with self.lock:
            now = time.time()
            
            # Remove old calls outside the time window
            self.calls = [call_time for call_time in self.calls 
                         if now - call_time < self.time_window]
            
            # Check if we can make another call
            if len(self.calls) < self.max_calls:
                self.calls.append(now)
                return True
            
            return False
    
    async def wait_if_needed(self) -> None:
        """Wait until a call is allowed."""
        while not self.is_allowed():
            await asyncio.sleep(0.1)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get rate limiter statistics."""
        with self.lock:
            now = time.time()
            recent_calls = [call_time for call_time in self.calls 
                           if now - call_time < self.time_window]
            
            return {
                "current_calls": len(recent_calls),
                "max_calls": self.max_calls,
                "time_window": self.time_window,
                "utilization": len(recent_calls) / self.max_calls
            }


def rate_limited(max_calls: int = 10, time_window: float = 60.0):
    """Decorator to apply rate limiting to functions."""
    limiter = RateLimiter(max_calls, time_window)
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            await limiter.wait_if_needed()
            
            if asyncio.iscoroutinefunction(func):
                return await func(*args, **kwargs)
            else:
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(None, func, *args, **kwargs)
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            while not limiter.is_allowed():
                time.sleep(0.1)
            return func(*args, **kwargs)
        
        # Return appropriate wrapper based on function type
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator


class AsyncBatchProcessor:
    """Asynchronous batch processing with smart batching."""
    
    def __init__(self, batch_size: int = 5, max_concurrent_batches: int = 3):
        """Initialize async batch processor."""
        self.batch_size = batch_size
        self.max_concurrent_batches = max_concurrent_batches
        self.logger = SmellDiffusionLogger("async_batch")
        self.semaphore = asyncio.Semaphore(max_concurrent_batches)
    
    async def process_items(self, items: List[Any], processor_func: Callable, **kwargs) -> List[Any]:
        """Process items in async batches."""
        if not items:
            return []
        
        # Split into batches
        batches = [items[i:i + self.batch_size] 
                  for i in range(0, len(items), self.batch_size)]
        
        self.logger.logger.info(f"Processing {len(items)} items in {len(batches)} batches")
        
        # Process batches concurrently
        tasks = []
        for batch in batches:
            task = self._process_batch(batch, processor_func, **kwargs)
            tasks.append(task)
        
        batch_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Flatten results
        all_results = []
        for batch_result in batch_results:
            if isinstance(batch_result, Exception):
                self.logger.log_error("batch_processing", batch_result)
                continue
            
            if isinstance(batch_result, list):
                all_results.extend(batch_result)
            else:
                all_results.append(batch_result)
        
        return all_results
    
    async def _process_batch(self, batch: List[Any], processor_func: Callable, **kwargs) -> List[Any]:
        """Process a single batch with semaphore control."""
        async with self.semaphore:
            batch_results = []
            
            # Process batch items
            if asyncio.iscoroutinefunction(processor_func):
                # Async processor
                tasks = [processor_func(item, **kwargs) for item in batch]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                for result in results:
                    if isinstance(result, Exception):
                        self.logger.log_error("batch_item_processing", result)
                        batch_results.append(None)
                    else:
                        batch_results.append(result)
            else:
                # Sync processor - run in executor
                loop = asyncio.get_event_loop()
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    tasks = [
                        loop.run_in_executor(executor, processor_func, item, **kwargs)
                        for item in batch
                    ]
                    results = await asyncio.gather(*tasks, return_exceptions=True)
                    
                    for result in results:
                        if isinstance(result, Exception):
                            self.logger.log_error("batch_item_processing", result)
                            batch_results.append(None)
                        else:
                            batch_results.append(result)
            
            return batch_results


class CircuitBreaker:
    """Circuit breaker pattern for resilient async operations."""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 60.0):
        """Initialize circuit breaker."""
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self.lock = threading.Lock()
        self.logger = SmellDiffusionLogger("circuit_breaker")
    
    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function through circuit breaker."""
        if not self._can_execute():
            raise Exception("Circuit breaker is OPEN")
        
        try:
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(None, func, *args, **kwargs)
            
            self._on_success()
            return result
            
        except Exception as e:
            self._on_failure()
            raise
    
    def _can_execute(self) -> bool:
        """Check if execution is allowed."""
        with self.lock:
            if self.state == "CLOSED":
                return True
            
            if self.state == "OPEN":
                if self.last_failure_time and \
                   time.time() - self.last_failure_time > self.recovery_timeout:
                    self.state = "HALF_OPEN"
                    return True
                return False
            
            if self.state == "HALF_OPEN":
                return True
            
            return False
    
    def _on_success(self) -> None:
        """Handle successful execution."""
        with self.lock:
            self.failure_count = 0
            self.state = "CLOSED"
    
    def _on_failure(self) -> None:
        """Handle failed execution."""
        with self.lock:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.failure_count >= self.failure_threshold:
                self.state = "OPEN"
                self.logger.logger.warning(
                    f"Circuit breaker opened after {self.failure_count} failures"
                )
    
    def get_state(self) -> Dict[str, Any]:
        """Get circuit breaker state."""
        with self.lock:
            return {
                "state": self.state,
                "failure_count": self.failure_count,
                "failure_threshold": self.failure_threshold,
                "last_failure_time": self.last_failure_time
            }