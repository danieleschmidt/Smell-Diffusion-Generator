"""Caching and performance optimization utilities."""

import hashlib
import pickle
import time
import functools
from pathlib import Path
from typing import Any, Dict, Optional, Callable, Tuple, List
import threading
import json
from datetime import datetime, timedelta
from collections import defaultdict, deque

from .logging import SmellDiffusionLogger, performance_monitor
from .config import get_config


class InMemoryCache:
    """Thread-safe in-memory cache with TTL support."""
    
    def __init__(self, max_size: int = 1000, default_ttl: int = 3600):
        """Initialize cache with size and TTL limits."""
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._cache: Dict[str, Tuple[Any, float]] = {}  # key -> (value, expiry_time)
        self._access_times: Dict[str, float] = {}  # For LRU eviction
        self._lock = threading.RLock()
        self.logger = SmellDiffusionLogger("cache")
        
        # Statistics
        self.hits = 0
        self.misses = 0
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        with self._lock:
            if key not in self._cache:
                self.misses += 1
                return None
            
            value, expiry_time = self._cache[key]
            
            # Check if expired
            if time.time() > expiry_time:
                del self._cache[key]
                if key in self._access_times:
                    del self._access_times[key]
                self.misses += 1
                return None
            
            # Update access time for LRU
            self._access_times[key] = time.time()
            self.hits += 1
            return value
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in cache."""
        with self._lock:
            if ttl is None:
                ttl = self.default_ttl
            
            expiry_time = time.time() + ttl
            
            # Evict if at capacity
            if len(self._cache) >= self.max_size and key not in self._cache:
                self._evict_lru()
            
            self._cache[key] = (value, expiry_time)
            self._access_times[key] = time.time()
    
    def _evict_lru(self) -> None:
        """Evict least recently used item."""
        if not self._access_times:
            return
        
        lru_key = min(self._access_times.keys(), key=lambda k: self._access_times[k])
        del self._cache[lru_key]
        del self._access_times[lru_key]
    
    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
            self._access_times.clear()
            self.hits = 0
            self.misses = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_requests = self.hits + self.misses
            hit_rate = self.hits / total_requests if total_requests > 0 else 0
            
            return {
                "size": len(self._cache),
                "max_size": self.max_size,
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": hit_rate,
                "utilization": len(self._cache) / self.max_size
            }


class DiskCache:
    """Persistent disk-based cache."""
    
    def __init__(self, cache_dir: Optional[Path] = None, max_size_mb: float = 500.0):
        """Initialize disk cache."""
        if cache_dir is None:
            cache_dir = Path.home() / ".smell_diffusion" / "cache"
        
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_size_mb = max_size_mb
        self.logger = SmellDiffusionLogger("disk_cache")
        
        # Metadata file for cache info
        self.metadata_file = self.cache_dir / "metadata.json"
        self._load_metadata()
    
    def _load_metadata(self) -> None:
        """Load cache metadata."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    self.metadata = json.load(f)
            except:
                self.metadata = {}
        else:
            self.metadata = {}
    
    def _save_metadata(self) -> None:
        """Save cache metadata."""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2)
        except Exception as e:
            self.logger.log_error("metadata_save", e)
    
    def _get_cache_path(self, key: str) -> Path:
        """Get file path for cache key."""
        # Use hash to avoid filesystem issues with long keys
        key_hash = hashlib.sha256(key.encode()).hexdigest()
        return self.cache_dir / f"{key_hash}.cache"
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from disk cache."""
        cache_path = self._get_cache_path(key)
        
        if not cache_path.exists():
            return None
        
        try:
            # Check expiry from metadata
            if key in self.metadata:
                expiry_time = self.metadata[key].get('expiry', 0)
                if time.time() > expiry_time:
                    self._remove_cache_file(key, cache_path)
                    return None
            
            # Load and return value
            with open(cache_path, 'rb') as f:
                value = pickle.load(f)
            
            # Update access time
            if key in self.metadata:
                self.metadata[key]['accessed'] = time.time()
                self._save_metadata()
            
            return value
            
        except Exception as e:
            self.logger.log_error("cache_get", e, {"key": key})
            return None
    
    def set(self, key: str, value: Any, ttl: int = 86400) -> None:  # Default 24h TTL
        """Set value in disk cache."""
        try:
            # Check cache size and clean if needed
            self._cleanup_if_needed()
            
            cache_path = self._get_cache_path(key)
            expiry_time = time.time() + ttl
            
            # Save value to disk
            with open(cache_path, 'wb') as f:
                pickle.dump(value, f)
            
            # Update metadata
            self.metadata[key] = {
                'created': time.time(),
                'accessed': time.time(),
                'expiry': expiry_time,
                'size': cache_path.stat().st_size
            }
            self._save_metadata()
            
        except Exception as e:
            self.logger.log_error("cache_set", e, {"key": key})
    
    def _remove_cache_file(self, key: str, cache_path: Path) -> None:
        """Remove cache file and metadata."""
        try:
            if cache_path.exists():
                cache_path.unlink()
            if key in self.metadata:
                del self.metadata[key]
                self._save_metadata()
        except Exception as e:
            self.logger.log_error("cache_remove", e, {"key": key})
    
    def _cleanup_if_needed(self) -> None:
        """Clean up cache if it exceeds size limit."""
        total_size = sum(
            item.get('size', 0) for item in self.metadata.values()
        )
        
        max_size_bytes = self.max_size_mb * 1024 * 1024
        
        if total_size > max_size_bytes:
            # Remove oldest accessed files
            sorted_items = sorted(
                self.metadata.items(),
                key=lambda x: x[1].get('accessed', 0)
            )
            
            removed_size = 0
            target_removal = total_size - (max_size_bytes * 0.8)  # Clean to 80% capacity
            
            for key, meta in sorted_items:
                if removed_size >= target_removal:
                    break
                
                cache_path = self._get_cache_path(key)
                self._remove_cache_file(key, cache_path)
                removed_size += meta.get('size', 0)
    
    def clear(self) -> None:
        """Clear all cache files."""
        try:
            for cache_file in self.cache_dir.glob("*.cache"):
                cache_file.unlink()
            
            self.metadata.clear()
            self._save_metadata()
            
        except Exception as e:
            self.logger.log_error("cache_clear", e)


class HybridCache:
    """Hybrid cache using both memory and disk storage."""
    
    def __init__(self, memory_size: int = 100, disk_size_mb: float = 500.0):
        """Initialize hybrid cache."""
        self.memory_cache = InMemoryCache(max_size=memory_size, default_ttl=3600)
        self.disk_cache = DiskCache(max_size_mb=disk_size_mb)
        self.logger = SmellDiffusionLogger("hybrid_cache")
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache (memory first, then disk)."""
        # Try memory cache first
        value = self.memory_cache.get(key)
        if value is not None:
            return value
        
        # Try disk cache
        value = self.disk_cache.get(key)
        if value is not None:
            # Promote to memory cache
            self.memory_cache.set(key, value)
            return value
        
        return None
    
    def set(self, key: str, value: Any, ttl: int = 3600, persist: bool = True) -> None:
        """Set value in cache."""
        # Always set in memory
        self.memory_cache.set(key, value, ttl)
        
        # Optionally persist to disk
        if persist:
            self.disk_cache.set(key, value, ttl * 24)  # Longer TTL for disk
    
    def clear(self) -> None:
        """Clear both caches."""
        self.memory_cache.clear()
        self.disk_cache.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get combined cache statistics."""
        memory_stats = self.memory_cache.get_stats()
        
        return {
            "memory": memory_stats,
            "disk": {
                "size": len(self.disk_cache.metadata),
                "max_size_mb": self.disk_cache.max_size_mb
            }
        }


# Global cache instance
_global_cache = None


def get_cache() -> HybridCache:
    """Get global cache instance."""
    global _global_cache
    if _global_cache is None:
        config = get_config()
        _global_cache = HybridCache()
    return _global_cache


def cache_key(*args, **kwargs) -> str:
    """Generate cache key from function arguments."""
    # Create a deterministic key from arguments
    key_data = {
        'args': args,
        'kwargs': sorted(kwargs.items())  # Sort for consistency
    }
    
    key_str = json.dumps(key_data, sort_keys=True, default=str)
    return hashlib.sha256(key_str.encode()).hexdigest()


def cached(ttl: int = 3600, persist: bool = True, key_func: Optional[Callable] = None):
    """Decorator to cache function results."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            cache = get_cache()
            
            # Generate cache key
            if key_func:
                key = key_func(*args, **kwargs)
            else:
                key = f"{func.__module__}.{func.__name__}:{cache_key(*args, **kwargs)}"
            
            # Try to get from cache
            cached_result = cache.get(key)
            if cached_result is not None:
                return cached_result
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            cache.set(key, result, ttl=ttl, persist=persist)
            
            return result
        
        # Add cache management methods
        wrapper.cache_clear = lambda: get_cache().clear()
        wrapper.cache_stats = lambda: get_cache().get_stats()
        
        return wrapper
    return decorator


class BatchProcessor:
    """Optimized batch processing for molecule generation."""
    
    def __init__(self, batch_size: int = 10, max_workers: int = 4):
        """Initialize batch processor."""
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.logger = SmellDiffusionLogger("batch_processor")
    
    @performance_monitor("batch_generation")
    def process_batch(self, prompts: list, generation_func: Callable, **kwargs) -> list:
        """Process multiple prompts in optimized batches."""
        if not prompts:
            return []
        
        results = []
        total_batches = (len(prompts) + self.batch_size - 1) // self.batch_size
        
        self.logger.logger.info(f"Processing {len(prompts)} prompts in {total_batches} batches")
        
        for i in range(0, len(prompts), self.batch_size):
            batch = prompts[i:i + self.batch_size]
            batch_results = []
            
            for prompt in batch:
                try:
                    result = generation_func(prompt=prompt, **kwargs)
                    batch_results.append(result)
                except Exception as e:
                    self.logger.log_error("batch_item_processing", e, {"prompt": prompt})
                    batch_results.append(None)
            
            results.extend(batch_results)
        
        return results


class PerformanceOptimizer:
    """System performance optimization utilities."""
    
    def __init__(self):
        """Initialize performance optimizer."""
        self.logger = SmellDiffusionLogger("performance")
        self._memory_monitor = threading.Thread(target=self._monitor_memory, daemon=True)
        self._monitoring = False
    
    def start_monitoring(self) -> None:
        """Start performance monitoring."""
        if not self._monitoring:
            self._monitoring = True
            self._memory_monitor.start()
    
    def stop_monitoring(self) -> None:
        """Stop performance monitoring."""
        self._monitoring = False
    
    def _monitor_memory(self) -> None:
        """Monitor memory usage."""
        try:
            import psutil
            process = psutil.Process()
            
            while self._monitoring:
                memory_info = process.memory_info()
                memory_mb = memory_info.rss / 1024 / 1024
                
                if memory_mb > 1000:  # Log if over 1GB
                    self.logger.logger.warning(f"High memory usage: {memory_mb:.1f} MB")
                
                time.sleep(60)  # Check every minute
                
        except ImportError:
            self.logger.logger.info("psutil not available, skipping memory monitoring")
        except Exception as e:
            self.logger.log_error("memory_monitoring", e)
    
    def optimize_generation_params(self, prompt: str, num_molecules: int) -> Dict[str, Any]:
        """Optimize generation parameters based on system capabilities."""
        optimized_params = {}
        
        # Adjust batch size based on memory and CPU
        try:
            import psutil
            available_memory_gb = psutil.virtual_memory().available / (1024**3)
            cpu_count = psutil.cpu_count()
            
            # Conservative batch sizing
            if available_memory_gb > 8 and cpu_count > 4:
                optimized_params['batch_size'] = min(num_molecules, 5)
            else:
                optimized_params['batch_size'] = min(num_molecules, 2)
                
        except ImportError:
            optimized_params['batch_size'] = 1
        
        # Optimize guidance scale based on prompt complexity
        prompt_complexity = len(prompt.split()) + prompt.count(',') * 2
        if prompt_complexity > 20:
            optimized_params['guidance_scale'] = 8.5  # Higher guidance for complex prompts
        else:
            optimized_params['guidance_scale'] = 7.5  # Standard guidance
        
        return optimized_params


# Global performance optimizer
performance_optimizer = PerformanceOptimizer()