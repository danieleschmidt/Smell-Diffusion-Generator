"""
Advanced Performance Optimization System
High-performance optimizations, caching strategies, and scalability enhancements.
"""

import time
import asyncio
import threading
import functools
import weakref
import gc
from typing import Dict, List, Any, Optional, Callable, Union, Tuple
from collections import defaultdict, OrderedDict
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing
import queue
import pickle
import hashlib
import json

from .logging import SmellDiffusionLogger
from .caching import cached


class PerformanceProfiler:
    """Advanced performance profiler for code optimization."""
    
    def __init__(self):
        self.logger = SmellDiffusionLogger("performance_profiler")
        self.profiles: Dict[str, List[float]] = defaultdict(list)
        self.active_timers: Dict[str, float] = {}
        self.memory_snapshots: Dict[str, List[int]] = defaultdict(list)
        
    def start_timer(self, operation: str):
        """Start timing an operation."""
        self.active_timers[operation] = time.perf_counter()
    
    def end_timer(self, operation: str) -> float:
        """End timing an operation and record duration."""
        if operation in self.active_timers:
            duration = time.perf_counter() - self.active_timers[operation]
            self.profiles[operation].append(duration)
            del self.active_timers[operation]
            return duration
        return 0.0
    
    def record_memory_usage(self, operation: str):
        """Record current memory usage for an operation."""
        try:
            import psutil
            import os
            process = psutil.Process(os.getpid())
            memory_mb = process.memory_info().rss / 1024 / 1024
            self.memory_snapshots[operation].append(int(memory_mb))
        except ImportError:
            # Fallback if psutil not available
            pass
    
    def get_stats(self, operation: Optional[str] = None) -> Dict[str, Any]:
        """Get performance statistics."""
        if operation:
            if operation in self.profiles:
                times = self.profiles[operation]
                return {
                    'operation': operation,
                    'total_calls': len(times),
                    'avg_duration': sum(times) / len(times),
                    'min_duration': min(times),
                    'max_duration': max(times),
                    'total_duration': sum(times)
                }
            return {}
        
        # Return stats for all operations
        stats = {}
        for op, times in self.profiles.items():
            stats[op] = {
                'total_calls': len(times),
                'avg_duration': sum(times) / len(times),
                'min_duration': min(times),
                'max_duration': max(times),
                'total_duration': sum(times)
            }
        
        return stats
    
    def profile(self, operation: str):
        """Decorator for automatic profiling."""
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                self.start_timer(operation)
                self.record_memory_usage(f"{operation}_start")
                
                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    self.end_timer(operation)
                    self.record_memory_usage(f"{operation}_end")
            
            return wrapper
        return decorator


class AdaptiveCacheManager:
    """Adaptive caching with intelligent eviction and preloading."""
    
    def __init__(self, max_size: int = 1000, ttl: float = 3600):
        self.max_size = max_size
        self.ttl = ttl
        self.cache: OrderedDict = OrderedDict()
        self.access_times: Dict[str, float] = {}
        self.access_counts: Dict[str, int] = defaultdict(int)
        self.hit_count = 0
        self.miss_count = 0
        self.logger = SmellDiffusionLogger("adaptive_cache")
        
        # Cache statistics for adaptive optimization
        self.size_history: List[int] = []
        self.hit_rate_history: List[float] = []
        
    def _generate_key(self, *args, **kwargs) -> str:
        """Generate cache key from arguments."""
        key_data = {
            'args': str(args),
            'kwargs': sorted(kwargs.items()) if kwargs else []
        }
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _is_expired(self, key: str) -> bool:
        """Check if cache entry is expired."""
        if key not in self.access_times:
            return True
        return time.time() - self.access_times[key] > self.ttl
    
    def _evict_lru(self):
        """Evict least recently used items."""
        # Sort by access time and frequency
        eviction_candidates = []
        for key in self.cache:
            if key in self.access_times:
                score = self.access_times[key] * self.access_counts[key]
                eviction_candidates.append((score, key))
        
        # Sort by score (lower is more evictable)
        eviction_candidates.sort()
        
        # Evict oldest 25% of cache
        evict_count = max(1, len(self.cache) // 4)
        for _, key in eviction_candidates[:evict_count]:
            self.remove(key)
    
    def get(self, key: str) -> Any:
        """Get item from cache."""
        if key in self.cache and not self._is_expired(key):
            # Move to end (most recently used)
            value = self.cache.pop(key)
            self.cache[key] = value
            self.access_times[key] = time.time()
            self.access_counts[key] += 1
            self.hit_count += 1
            return value
        
        self.miss_count += 1
        return None
    
    def put(self, key: str, value: Any):
        """Put item in cache."""
        # Remove if already exists
        if key in self.cache:
            del self.cache[key]
        
        # Evict if at capacity
        if len(self.cache) >= self.max_size:
            self._evict_lru()
        
        self.cache[key] = value
        self.access_times[key] = time.time()
        self.access_counts[key] = 1
    
    def remove(self, key: str):
        """Remove item from cache."""
        if key in self.cache:
            del self.cache[key]
        if key in self.access_times:
            del self.access_times[key]
        if key in self.access_counts:
            del self.access_counts[key]
    
    def clear(self):
        """Clear all cache."""
        self.cache.clear()
        self.access_times.clear()
        self.access_counts.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.hit_count + self.miss_count
        hit_rate = self.hit_count / max(1, total_requests)
        
        return {
            'size': len(self.cache),
            'max_size': self.max_size,
            'hit_count': self.hit_count,
            'miss_count': self.miss_count,
            'hit_rate': hit_rate,
            'ttl': self.ttl
        }
    
    def cached(self, ttl: Optional[float] = None):
        """Decorator for automatic caching."""
        cache_ttl = ttl or self.ttl
        
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                key = f"{func.__name__}_{self._generate_key(*args, **kwargs)}"
                
                # Try to get from cache
                result = self.get(key)
                if result is not None:
                    return result
                
                # Compute and cache result
                result = func(*args, **kwargs)
                self.put(key, result)
                return result
            
            return wrapper
        return decorator


class AsyncBatchOptimizer:
    """Optimized async batch processing with dynamic batching."""
    
    def __init__(self, 
                 base_batch_size: int = 8,
                 max_batch_size: int = 32,
                 max_concurrent: int = 4,
                 adaptive_sizing: bool = True):
        self.base_batch_size = base_batch_size
        self.max_batch_size = max_batch_size
        self.max_concurrent = max_concurrent
        self.adaptive_sizing = adaptive_sizing
        
        self.logger = SmellDiffusionLogger("batch_optimizer")
        self.performance_history: List[Dict[str, float]] = []
        self.optimal_batch_size = base_batch_size
        
    async def process_batch_optimized(self, 
                                    items: List[Any], 
                                    processor: Callable,
                                    **kwargs) -> List[Any]:
        """Process items in optimized batches."""
        if not items:
            return []
        
        # Determine optimal batch size
        batch_size = self._calculate_optimal_batch_size(len(items))
        
        # Create batches
        batches = [items[i:i + batch_size] for i in range(0, len(items), batch_size)]
        
        # Process batches concurrently
        semaphore = asyncio.Semaphore(self.max_concurrent)
        
        async def process_single_batch(batch):
            async with semaphore:
                start_time = time.perf_counter()
                try:
                    # Process all items in batch concurrently
                    tasks = [self._process_item_async(item, processor, **kwargs) for item in batch]
                    results = await asyncio.gather(*tasks, return_exceptions=True)
                    
                    # Filter out exceptions
                    valid_results = [r for r in results if not isinstance(r, Exception)]
                    
                    duration = time.perf_counter() - start_time
                    self._record_batch_performance(len(batch), duration, len(valid_results))
                    
                    return valid_results
                    
                except Exception as e:
                    self.logger.log_error("batch_processing", e)
                    return []
        
        # Execute all batches
        batch_results = await asyncio.gather(*[process_single_batch(batch) for batch in batches])
        
        # Flatten results
        all_results = []
        for batch_result in batch_results:
            all_results.extend(batch_result)
        
        return all_results
    
    async def _process_item_async(self, item: Any, processor: Callable, **kwargs) -> Any:
        """Process single item asynchronously."""
        if asyncio.iscoroutinefunction(processor):
            return await processor(item, **kwargs)
        else:
            # Run in thread pool for sync functions
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, processor, item, **kwargs)
    
    def _calculate_optimal_batch_size(self, total_items: int) -> int:
        """Calculate optimal batch size based on performance history."""
        if not self.adaptive_sizing or not self.performance_history:
            return min(self.base_batch_size, total_items)
        
        # Analyze recent performance
        recent_performance = self.performance_history[-10:]  # Last 10 batches
        
        # Find best performing batch size
        size_performance = defaultdict(list)
        for perf in recent_performance:
            size_performance[perf['batch_size']].append(perf['throughput'])
        
        best_size = self.base_batch_size
        best_throughput = 0
        
        for size, throughputs in size_performance.items():
            avg_throughput = sum(throughputs) / len(throughputs)
            if avg_throughput > best_throughput:
                best_throughput = avg_throughput
                best_size = size
        
        # Gradually adjust towards optimal size
        self.optimal_batch_size = min(
            self.max_batch_size,
            max(1, int(self.optimal_batch_size * 0.9 + best_size * 0.1))
        )
        
        return min(self.optimal_batch_size, total_items)
    
    def _record_batch_performance(self, batch_size: int, duration: float, success_count: int):
        """Record batch performance for optimization."""
        throughput = success_count / max(duration, 0.001)  # items per second
        
        performance_record = {
            'batch_size': batch_size,
            'duration': duration,
            'success_count': success_count,
            'throughput': throughput,
            'timestamp': time.time()
        }
        
        self.performance_history.append(performance_record)
        
        # Keep only recent history
        if len(self.performance_history) > 100:
            self.performance_history = self.performance_history[-50:]


class ResourcePoolManager:
    """Manage pools of resources for optimal utilization."""
    
    def __init__(self):
        self.logger = SmellDiffusionLogger("resource_pool")
        self.thread_pool: Optional[ThreadPoolExecutor] = None
        self.process_pool: Optional[ProcessPoolExecutor] = None
        self.connection_pools: Dict[str, Any] = {}
        
        # Pool statistics
        self.pool_stats: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        
    def get_thread_pool(self, max_workers: Optional[int] = None) -> ThreadPoolExecutor:
        """Get or create thread pool."""
        if self.thread_pool is None:
            max_workers = max_workers or min(32, (multiprocessing.cpu_count() or 1) + 4)
            self.thread_pool = ThreadPoolExecutor(
                max_workers=max_workers,
                thread_name_prefix='smell_diffusion_'
            )
            self.logger.logger.info(f"Created thread pool with {max_workers} workers")
        
        return self.thread_pool
    
    def get_process_pool(self, max_workers: Optional[int] = None) -> ProcessPoolExecutor:
        """Get or create process pool."""
        if self.process_pool is None:
            max_workers = max_workers or multiprocessing.cpu_count() or 1
            self.process_pool = ProcessPoolExecutor(max_workers=max_workers)
            self.logger.logger.info(f"Created process pool with {max_workers} workers")
        
        return self.process_pool
    
    def register_connection_pool(self, name: str, pool: Any):
        """Register a connection pool."""
        self.connection_pools[name] = pool
        self.logger.logger.info(f"Registered connection pool: {name}")
    
    def get_connection_pool(self, name: str) -> Any:
        """Get a connection pool."""
        return self.connection_pools.get(name)
    
    def record_pool_usage(self, pool_name: str, operation: str):
        """Record pool usage statistics."""
        self.pool_stats[pool_name][operation] += 1
    
    def get_pool_stats(self) -> Dict[str, Any]:
        """Get pool utilization statistics."""
        stats = {}
        
        if self.thread_pool:
            stats['thread_pool'] = {
                'max_workers': self.thread_pool._max_workers,
                'active_threads': len(self.thread_pool._threads),
                'pending_tasks': self.thread_pool._work_queue.qsize()
            }
        
        if self.process_pool:
            stats['process_pool'] = {
                'max_workers': self.process_pool._max_workers,
                'active_processes': len(getattr(self.process_pool, '_processes', {}))
            }
        
        stats['usage_stats'] = dict(self.pool_stats)
        
        return stats
    
    def shutdown(self):
        """Shutdown all pools."""
        if self.thread_pool:
            self.thread_pool.shutdown(wait=True)
            self.thread_pool = None
        
        if self.process_pool:
            self.process_pool.shutdown(wait=True)
            self.process_pool = None
        
        self.logger.logger.info("All resource pools shut down")


class MemoryOptimizer:
    """Memory optimization and garbage collection management."""
    
    def __init__(self):
        self.logger = SmellDiffusionLogger("memory_optimizer")
        self.weak_refs: List[weakref.ref] = []
        self.memory_snapshots: List[Tuple[float, int]] = []
        
    def register_for_cleanup(self, obj: Any):
        """Register object for automatic cleanup."""
        try:
            def cleanup_callback(ref):
                self.logger.logger.debug("Object cleaned up via weak reference")
            
            # Only create weak references for objects that support them
            if hasattr(obj, '__weakref__'):
                self.weak_refs.append(weakref.ref(obj, cleanup_callback))
        except TypeError:
            # Skip objects that don't support weak references (like lists, tuples)
            pass
    
    def force_gc(self):
        """Force garbage collection and record memory usage."""
        before_memory = self._get_memory_usage()
        
        # Force full garbage collection
        collected = gc.collect()
        
        after_memory = self._get_memory_usage()
        freed_mb = before_memory - after_memory
        
        self.memory_snapshots.append((time.time(), after_memory))
        
        self.logger.logger.info(
            f"Garbage collection freed {freed_mb:.1f}MB, "
            f"collected {collected} objects"
        )
        
        return freed_mb
    
    def _get_memory_usage(self) -> int:
        """Get current memory usage in MB."""
        try:
            import psutil
            import os
            process = psutil.Process(os.getpid())
            return int(process.memory_info().rss / 1024 / 1024)
        except ImportError:
            return 0
    
    def optimize_memory_usage(self):
        """Perform memory optimization."""
        # Clean up weak references
        self.weak_refs = [ref for ref in self.weak_refs if ref() is not None]
        
        # Force garbage collection
        freed = self.force_gc()
        
        # Clean old memory snapshots
        current_time = time.time()
        self.memory_snapshots = [
            (timestamp, memory) for timestamp, memory in self.memory_snapshots
            if current_time - timestamp <= 3600  # Keep last hour
        ]
        
        return freed
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory optimization statistics."""
        current_memory = self._get_memory_usage()
        
        stats = {
            'current_memory_mb': current_memory,
            'weak_refs_count': len(self.weak_refs),
            'snapshots_count': len(self.memory_snapshots)
        }
        
        if self.memory_snapshots:
            memories = [memory for _, memory in self.memory_snapshots]
            stats.update({
                'min_memory_mb': min(memories),
                'max_memory_mb': max(memories),
                'avg_memory_mb': sum(memories) / len(memories)
            })
        
        return stats


class PerformanceOptimizer:
    """Comprehensive performance optimization system."""
    
    def __init__(self):
        self.logger = SmellDiffusionLogger("performance_optimizer")
        self.profiler = PerformanceProfiler()
        self.cache_manager = AdaptiveCacheManager()
        self.batch_optimizer = AsyncBatchOptimizer()
        self.resource_pool = ResourcePoolManager()
        self.memory_optimizer = MemoryOptimizer()
        
        # Global optimization settings
        self.optimization_enabled = True
        self.auto_gc_enabled = True
        self.adaptive_batching_enabled = True
        
    def enable_optimizations(self):
        """Enable all performance optimizations."""
        self.optimization_enabled = True
        self.logger.logger.info("Performance optimizations enabled")
    
    def disable_optimizations(self):
        """Disable performance optimizations (for debugging)."""
        self.optimization_enabled = False
        self.logger.logger.info("Performance optimizations disabled")
    
    def optimize_function(self, 
                         cache_ttl: Optional[float] = None,
                         profile: bool = True,
                         memory_optimize: bool = True):
        """Decorator for comprehensive function optimization."""
        
        def decorator(func: Callable) -> Callable:
            # Apply caching
            if self.optimization_enabled:
                func = self.cache_manager.cached(cache_ttl)(func)
            
            # Apply profiling
            if profile and self.optimization_enabled:
                func = self.profiler.profile(func.__name__)(func)
            
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                try:
                    result = func(*args, **kwargs)
                    
                    # Memory optimization
                    if memory_optimize and self.optimization_enabled:
                        self.memory_optimizer.register_for_cleanup(result)
                    
                    return result
                    
                except Exception as e:
                    self.logger.log_error(f"optimized_function_{func.__name__}", e)
                    raise
            
            return wrapper
        
        return decorator
    
    async def optimize_batch_processing(self, 
                                      items: List[Any], 
                                      processor: Callable,
                                      **kwargs) -> List[Any]:
        """Optimize batch processing with all available optimizations."""
        if not self.optimization_enabled:
            # Fallback to simple processing
            return [processor(item, **kwargs) for item in items]
        
        return await self.batch_optimizer.process_batch_optimized(
            items, processor, **kwargs
        )
    
    def periodic_optimization(self):
        """Perform periodic optimization tasks."""
        if not self.optimization_enabled:
            return
        
        # Memory optimization
        if self.auto_gc_enabled:
            freed_memory = self.memory_optimizer.optimize_memory_usage()
            if freed_memory > 10:  # Log if significant memory freed
                self.logger.logger.info(f"Periodic optimization freed {freed_memory:.1f}MB")
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        return {
            'optimization_enabled': self.optimization_enabled,
            'profiler_stats': self.profiler.get_stats(),
            'cache_stats': self.cache_manager.get_stats(),
            'resource_pool_stats': self.resource_pool.get_pool_stats(),
            'memory_stats': self.memory_optimizer.get_memory_stats(),
            'batch_optimizer': {
                'optimal_batch_size': self.batch_optimizer.optimal_batch_size,
                'performance_history_length': len(self.batch_optimizer.performance_history)
            }
        }
    
    def shutdown(self):
        """Shutdown optimizer and clean up resources."""
        self.resource_pool.shutdown()
        self.cache_manager.clear()
        self.memory_optimizer.optimize_memory_usage()
        self.logger.logger.info("Performance optimizer shut down")


# Global performance optimizer instance
global_performance_optimizer = PerformanceOptimizer()

# Convenience decorators
optimize_performance = global_performance_optimizer.optimize_function
profile_performance = global_performance_optimizer.profiler.profile