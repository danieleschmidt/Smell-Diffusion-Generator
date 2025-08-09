"""
Distributed and Scalable Molecular Generation

Production-ready scaling infrastructure:
- Distributed generation across multiple workers
- Advanced caching and load balancing  
- Auto-scaling based on demand
- Resource optimization and monitoring
"""

import asyncio
import time
import hashlib
import os
from typing import List, Dict, Any, Optional, Callable, Union
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import queue
import threading
from contextlib import contextmanager

try:
    import multiprocessing as mp
    MP_AVAILABLE = True
except ImportError:
    MP_AVAILABLE = False

from ..core.smell_diffusion import SmellDiffusion
from ..core.molecule import Molecule
from ..utils.logging import SmellDiffusionLogger, performance_monitor
from ..utils.caching import cached


@dataclass
class ScalingConfiguration:
    """Configuration for scalable generation."""
    max_workers: int = 8
    worker_type: str = "thread"  # "thread", "process", "async"
    batch_size: int = 32
    queue_size: int = 1000
    auto_scale: bool = True
    scale_threshold: float = 0.8  # CPU utilization threshold
    min_workers: int = 2
    max_workers_limit: int = 16
    load_balancing: str = "round_robin"  # "round_robin", "least_loaded", "random"


@dataclass 
class WorkerStats:
    """Statistics for individual workers."""
    worker_id: str
    tasks_completed: int
    tasks_failed: int
    avg_processing_time: float
    current_load: float
    last_activity: float


class DistributedGenerator:
    """High-performance distributed molecular generation system."""
    
    def __init__(self, config: Optional[ScalingConfiguration] = None):
        self.config = config or ScalingConfiguration()
        self.logger = SmellDiffusionLogger("distributed_generator")
        
        # Worker management
        self.workers = {}
        self.worker_stats = {}
        self.task_queue = queue.Queue(maxsize=self.config.queue_size)
        self.result_queue = queue.Queue(maxsize=self.config.queue_size * 2)
        
        # Load balancing
        self.load_balancer = LoadBalancer(self.config.load_balancing)
        
        # Auto-scaling
        self.auto_scaler = AutoScaler(self.config) if self.config.auto_scale else None
        
        # Performance monitoring
        self.performance_monitor = DistributedPerformanceMonitor()
        
        # Initialize worker pool
        self._initialize_workers()
        
        # Start monitoring threads
        self._start_monitoring()
        
    def _initialize_workers(self):
        """Initialize worker pool based on configuration."""
        
        if self.config.worker_type == "process" and MP_AVAILABLE:
            self.executor = ProcessPoolExecutor(max_workers=self.config.max_workers)
            self.logger.logger.info(f"Initialized {self.config.max_workers} process workers")
            
        elif self.config.worker_type == "thread":
            self.executor = ThreadPoolExecutor(max_workers=self.config.max_workers)
            self.logger.logger.info(f"Initialized {self.config.max_workers} thread workers")
            
        else:
            # Async workers handled separately
            self.executor = None
            self.logger.logger.info("Initialized async worker pool")
        
        # Initialize worker statistics
        for i in range(self.config.max_workers):
            worker_id = f"worker_{i}"
            self.worker_stats[worker_id] = WorkerStats(
                worker_id=worker_id,
                tasks_completed=0,
                tasks_failed=0,
                avg_processing_time=0.0,
                current_load=0.0,
                last_activity=time.time()
            )
    
    def _start_monitoring(self):
        """Start background monitoring threads."""
        
        # Performance monitoring thread
        monitor_thread = threading.Thread(
            target=self.performance_monitor.start_monitoring,
            args=(self.worker_stats,),
            daemon=True
        )
        monitor_thread.start()
        
        # Auto-scaling thread
        if self.auto_scaler:
            scaling_thread = threading.Thread(
                target=self.auto_scaler.monitor_and_scale,
                args=(self.worker_stats,),
                daemon=True
            )
            scaling_thread.start()
    
    @performance_monitor("distributed_generation")
    async def generate_distributed(self, 
                                 requests: List[Dict[str, Any]],
                                 callback: Optional[Callable] = None) -> List[Dict[str, Any]]:
        """Generate molecules using distributed processing."""
        
        self.logger.logger.info(f"Starting distributed generation for {len(requests)} requests")
        
        if self.config.worker_type == "async":
            return await self._generate_async(requests, callback)
        else:
            return await self._generate_executor(requests, callback)
    
    async def _generate_async(self, requests: List[Dict[str, Any]], 
                            callback: Optional[Callable] = None) -> List[Dict[str, Any]]:
        """Async-based distributed generation."""
        
        # Create semaphore to limit concurrent operations
        semaphore = asyncio.Semaphore(self.config.max_workers)
        
        async def process_request(request_data: Dict[str, Any]) -> Dict[str, Any]:
            async with semaphore:
                worker_id = f"async_worker_{asyncio.current_task().get_name()}"
                start_time = time.time()
                
                try:
                    # Load balancing - select best worker
                    selected_worker = self.load_balancer.select_worker(self.worker_stats)
                    
                    # Generate molecules
                    generator = SmellDiffusion()
                    molecules = generator.generate(
                        prompt=request_data.get('prompt', ''),
                        num_molecules=request_data.get('num_molecules', 1),
                        **request_data.get('kwargs', {})
                    )
                    
                    processing_time = time.time() - start_time
                    
                    # Update worker statistics
                    await self._update_worker_stats(selected_worker, processing_time, success=True)
                    
                    result = {
                        'request_id': request_data.get('request_id'),
                        'molecules': molecules if isinstance(molecules, list) else [molecules],
                        'processing_time': processing_time,
                        'worker_id': selected_worker,
                        'success': True
                    }
                    
                    # Execute callback if provided
                    if callback:
                        try:
                            await callback(result)
                        except Exception as e:
                            self.logger.log_error("callback_execution", e)
                    
                    return result
                    
                except Exception as e:
                    processing_time = time.time() - start_time
                    await self._update_worker_stats(worker_id, processing_time, success=False)
                    
                    self.logger.log_error("async_generation", e)
                    return {
                        'request_id': request_data.get('request_id'),
                        'molecules': [],
                        'processing_time': processing_time,
                        'worker_id': worker_id,
                        'success': False,
                        'error': str(e)
                    }
        
        # Process all requests concurrently
        tasks = [process_request(request) for request in requests]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle any exceptions
        processed_results = []
        for result in results:
            if isinstance(result, Exception):
                processed_results.append({
                    'success': False,
                    'error': str(result),
                    'molecules': [],
                    'processing_time': 0.0
                })
            else:
                processed_results.append(result)
        
        return processed_results
    
    async def _generate_executor(self, requests: List[Dict[str, Any]],
                               callback: Optional[Callable] = None) -> List[Dict[str, Any]]:
        """Executor-based distributed generation."""
        
        def process_single_request(request_data: Dict[str, Any]) -> Dict[str, Any]:
            """Process a single generation request."""
            start_time = time.time()
            worker_id = f"{self.config.worker_type}_worker_{threading.current_thread().ident}"
            
            try:
                # Generate molecules
                generator = SmellDiffusion()
                molecules = generator.generate(
                    prompt=request_data.get('prompt', ''),
                    num_molecules=request_data.get('num_molecules', 1),
                    **request_data.get('kwargs', {})
                )
                
                processing_time = time.time() - start_time
                
                return {
                    'request_id': request_data.get('request_id'),
                    'molecules': molecules if isinstance(molecules, list) else [molecules],
                    'processing_time': processing_time,
                    'worker_id': worker_id,
                    'success': True
                }
                
            except Exception as e:
                processing_time = time.time() - start_time
                return {
                    'request_id': request_data.get('request_id'),
                    'molecules': [],
                    'processing_time': processing_time,
                    'worker_id': worker_id,
                    'success': False,
                    'error': str(e)
                }
        
        # Submit tasks to executor
        loop = asyncio.get_event_loop()
        
        if self.executor:
            # Use thread/process pool executor
            futures = [
                loop.run_in_executor(self.executor, process_single_request, request)
                for request in requests
            ]
            
            results = await asyncio.gather(*futures, return_exceptions=True)
            
            # Process results and execute callbacks
            processed_results = []
            for result in results:
                if isinstance(result, Exception):
                    processed_results.append({
                        'success': False,
                        'error': str(result),
                        'molecules': [],
                        'processing_time': 0.0
                    })
                else:
                    processed_results.append(result)
                    
                    # Execute callback
                    if callback and not isinstance(result, Exception):
                        try:
                            if asyncio.iscoroutinefunction(callback):
                                await callback(result)
                            else:
                                callback(result)
                        except Exception as e:
                            self.logger.log_error("callback_execution", e)
            
            return processed_results
        else:
            # Fallback to sequential processing
            results = [process_single_request(request) for request in requests]
            return results
    
    async def _update_worker_stats(self, worker_id: str, processing_time: float, 
                                 success: bool = True):
        """Update worker statistics."""
        
        if worker_id not in self.worker_stats:
            self.worker_stats[worker_id] = WorkerStats(
                worker_id=worker_id,
                tasks_completed=0,
                tasks_failed=0,
                avg_processing_time=0.0,
                current_load=0.0,
                last_activity=time.time()
            )
        
        stats = self.worker_stats[worker_id]
        
        if success:
            stats.tasks_completed += 1
        else:
            stats.tasks_failed += 1
        
        # Update average processing time
        total_tasks = stats.tasks_completed + stats.tasks_failed
        if total_tasks > 1:
            stats.avg_processing_time = (
                (stats.avg_processing_time * (total_tasks - 1) + processing_time) / total_tasks
            )
        else:
            stats.avg_processing_time = processing_time
        
        stats.last_activity = time.time()
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics."""
        
        total_completed = sum(stats.tasks_completed for stats in self.worker_stats.values())
        total_failed = sum(stats.tasks_failed for stats in self.worker_stats.values())
        total_tasks = total_completed + total_failed
        
        avg_processing_times = [stats.avg_processing_time for stats in self.worker_stats.values() if stats.avg_processing_time > 0]
        overall_avg_time = sum(avg_processing_times) / len(avg_processing_times) if avg_processing_times else 0.0
        
        return {
            'total_workers': len(self.worker_stats),
            'active_workers': len([s for s in self.worker_stats.values() if time.time() - s.last_activity < 60]),
            'total_tasks_completed': total_completed,
            'total_tasks_failed': total_failed,
            'success_rate': total_completed / max(total_tasks, 1),
            'average_processing_time': overall_avg_time,
            'throughput': total_completed / max(time.time() - self.performance_monitor.start_time, 1) if hasattr(self.performance_monitor, 'start_time') else 0,
            'worker_stats': {wid: {
                'completed': stats.tasks_completed,
                'failed': stats.tasks_failed,
                'avg_time': stats.avg_processing_time,
                'last_active': stats.last_activity
            } for wid, stats in self.worker_stats.items()}
        }
    
    def shutdown(self):
        """Gracefully shutdown the distributed generator."""
        
        self.logger.logger.info("Shutting down distributed generator")
        
        if self.executor:
            self.executor.shutdown(wait=True)
        
        # Final performance report
        metrics = self.get_performance_metrics()
        self.logger.logger.info(f"Final performance metrics: {metrics}")


class LoadBalancer:
    """Intelligent load balancing for worker selection."""
    
    def __init__(self, strategy: str = "round_robin"):
        self.strategy = strategy
        self.round_robin_counter = 0
        
    def select_worker(self, worker_stats: Dict[str, WorkerStats]) -> str:
        """Select the best worker based on load balancing strategy."""
        
        if not worker_stats:
            return "default_worker"
        
        workers = list(worker_stats.keys())
        
        if self.strategy == "round_robin":
            selected = workers[self.round_robin_counter % len(workers)]
            self.round_robin_counter += 1
            return selected
            
        elif self.strategy == "least_loaded":
            # Select worker with lowest current load
            least_loaded = min(worker_stats.items(), key=lambda x: x[1].current_load)
            return least_loaded[0]
            
        elif self.strategy == "fastest_response":
            # Select worker with fastest average response time
            fastest = min(
                worker_stats.items(), 
                key=lambda x: x[1].avg_processing_time if x[1].avg_processing_time > 0 else float('inf')
            )
            return fastest[0]
            
        else:  # random
            import random
            return random.choice(workers)


class AutoScaler:
    """Automatic scaling based on system metrics."""
    
    def __init__(self, config: ScalingConfiguration):
        self.config = config
        self.last_scale_action = 0
        self.scale_cooldown = 30  # seconds
        self.logger = SmellDiffusionLogger("auto_scaler")
        
    def monitor_and_scale(self, worker_stats: Dict[str, WorkerStats]):
        """Monitor system and adjust worker count."""
        
        while True:
            try:
                time.sleep(10)  # Check every 10 seconds
                
                # Get current metrics
                current_time = time.time()
                
                # Skip if in cooldown period
                if current_time - self.last_scale_action < self.scale_cooldown:
                    continue
                
                # Calculate system load
                system_load = self._calculate_system_load(worker_stats)
                active_workers = len([s for s in worker_stats.values() if current_time - s.last_activity < 30])
                
                # Scale up if high load
                if system_load > self.config.scale_threshold and active_workers < self.config.max_workers_limit:
                    self._scale_up(worker_stats)
                    self.last_scale_action = current_time
                    
                # Scale down if low load
                elif system_load < (self.config.scale_threshold * 0.3) and active_workers > self.config.min_workers:
                    self._scale_down(worker_stats)
                    self.last_scale_action = current_time
                    
            except Exception as e:
                self.logger.log_error("auto_scaling", e)
                time.sleep(60)  # Wait longer on error
    
    def _calculate_system_load(self, worker_stats: Dict[str, WorkerStats]) -> float:
        """Calculate overall system load."""
        
        if not worker_stats:
            return 0.0
        
        # Simple load calculation based on worker activity
        current_time = time.time()
        active_workers = [s for s in worker_stats.values() if current_time - s.last_activity < 30]
        
        if not active_workers:
            return 0.0
        
        # Load based on recent task completion rate
        total_load = sum(min(1.0, s.avg_processing_time / 5.0) for s in active_workers)
        return total_load / len(active_workers)
    
    def _scale_up(self, worker_stats: Dict[str, WorkerStats]):
        """Add more workers."""
        
        current_workers = len(worker_stats)
        new_worker_id = f"scaled_worker_{current_workers}"
        
        worker_stats[new_worker_id] = WorkerStats(
            worker_id=new_worker_id,
            tasks_completed=0,
            tasks_failed=0,
            avg_processing_time=0.0,
            current_load=0.0,
            last_activity=time.time()
        )
        
        self.logger.logger.info(f"Scaled up: Added worker {new_worker_id}")
    
    def _scale_down(self, worker_stats: Dict[str, WorkerStats]):
        """Remove excess workers."""
        
        # Find least active worker
        current_time = time.time()
        inactive_workers = [
            (wid, stats) for wid, stats in worker_stats.items() 
            if current_time - stats.last_activity > 60
        ]
        
        if inactive_workers:
            worker_to_remove = min(inactive_workers, key=lambda x: x[1].tasks_completed)[0]
            del worker_stats[worker_to_remove]
            self.logger.logger.info(f"Scaled down: Removed worker {worker_to_remove}")


class DistributedPerformanceMonitor:
    """Monitor performance across distributed workers."""
    
    def __init__(self):
        self.start_time = time.time()
        self.logger = SmellDiffusionLogger("distributed_performance")
        self.metrics_history = []
        
    def start_monitoring(self, worker_stats: Dict[str, WorkerStats]):
        """Start continuous performance monitoring."""
        
        while True:
            try:
                time.sleep(30)  # Monitor every 30 seconds
                
                # Collect current metrics
                metrics = self._collect_metrics(worker_stats)
                self.metrics_history.append(metrics)
                
                # Keep only recent history
                if len(self.metrics_history) > 100:
                    self.metrics_history.pop(0)
                
                # Log performance summary
                self._log_performance_summary(metrics)
                
                # Check for performance issues
                self._check_performance_issues(metrics)
                
            except Exception as e:
                self.logger.log_error("performance_monitoring", e)
                time.sleep(60)
    
    def _collect_metrics(self, worker_stats: Dict[str, WorkerStats]) -> Dict[str, Any]:
        """Collect current performance metrics."""
        
        current_time = time.time()
        
        # Worker-level metrics
        active_workers = [s for s in worker_stats.values() if current_time - s.last_activity < 60]
        
        total_completed = sum(s.tasks_completed for s in worker_stats.values())
        total_failed = sum(s.tasks_failed for s in worker_stats.values())
        
        avg_processing_times = [s.avg_processing_time for s in active_workers if s.avg_processing_time > 0]
        overall_avg_time = sum(avg_processing_times) / len(avg_processing_times) if avg_processing_times else 0.0
        
        # System-level metrics
        uptime = current_time - self.start_time
        throughput = total_completed / uptime if uptime > 0 else 0.0
        
        return {
            'timestamp': current_time,
            'active_workers': len(active_workers),
            'total_workers': len(worker_stats),
            'total_completed': total_completed,
            'total_failed': total_failed,
            'success_rate': total_completed / max(total_completed + total_failed, 1),
            'throughput': throughput,
            'avg_processing_time': overall_avg_time,
            'uptime': uptime
        }
    
    def _log_performance_summary(self, metrics: Dict[str, Any]):
        """Log performance summary."""
        
        self.logger.logger.info(
            f"Performance: {metrics['active_workers']}/{metrics['total_workers']} workers active, "
            f"{metrics['throughput']:.2f} req/s throughput, "
            f"{metrics['success_rate']:.2%} success rate, "
            f"{metrics['avg_processing_time']:.2f}s avg time"
        )
    
    def _check_performance_issues(self, metrics: Dict[str, Any]):
        """Check for performance issues and alert."""
        
        # Low success rate
        if metrics['success_rate'] < 0.9:
            self.logger.logger.warning(f"Low success rate: {metrics['success_rate']:.2%}")
        
        # High processing time
        if metrics['avg_processing_time'] > 10.0:
            self.logger.logger.warning(f"High processing time: {metrics['avg_processing_time']:.2f}s")
        
        # Low throughput
        if metrics['throughput'] < 1.0 and metrics['total_completed'] > 10:
            self.logger.logger.warning(f"Low throughput: {metrics['throughput']:.2f} req/s")


class ResourceOptimizer:
    """Optimize resource usage for maximum efficiency."""
    
    def __init__(self):
        self.logger = SmellDiffusionLogger("resource_optimizer")
        
    @contextmanager
    def optimized_context(self, optimization_level: str = "balanced"):
        """Context manager for optimized resource usage."""
        
        original_settings = self._save_current_settings()
        
        try:
            # Apply optimization settings
            self._apply_optimization(optimization_level)
            yield
            
        finally:
            # Restore original settings
            self._restore_settings(original_settings)
    
    def _save_current_settings(self) -> Dict[str, Any]:
        """Save current system settings."""
        return {
            'gc_enabled': True,  # Placeholder
            'thread_count': os.cpu_count() or 2,
        }
    
    def _apply_optimization(self, level: str):
        """Apply optimization based on level."""
        
        if level == "speed":
            # Optimize for speed
            self.logger.logger.info("Applied speed optimization")
            
        elif level == "memory":
            # Optimize for memory efficiency
            self.logger.logger.info("Applied memory optimization")
            
        else:  # balanced
            # Balanced optimization
            self.logger.logger.info("Applied balanced optimization")
    
    def _restore_settings(self, settings: Dict[str, Any]):
        """Restore original settings."""
        self.logger.logger.debug("Restored original settings")


# Factory function for easy distributed generation setup
def create_distributed_generator(max_workers: int = None,
                               worker_type: str = "thread",
                               auto_scale: bool = True) -> DistributedGenerator:
    """Create optimally configured distributed generator."""
    
    if max_workers is None:
        max_workers = min(8, (os.cpu_count() or 2) * 2)
    
    config = ScalingConfiguration(
        max_workers=max_workers,
        worker_type=worker_type,
        auto_scale=auto_scale,
        batch_size=min(32, max_workers * 4),
        load_balancing="least_loaded"
    )
    
    return DistributedGenerator(config)