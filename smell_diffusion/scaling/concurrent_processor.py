"""
Advanced Concurrent Processing System for Molecular Generation

Provides high-performance concurrent processing with:
- Intelligent resource pooling and management
- Adaptive concurrency based on system load
- Circuit breaker patterns for resilience
- Performance monitoring and optimization
- Memory-efficient batch processing
- Error handling and recovery mechanisms
"""

import asyncio
import concurrent.futures
import threading
import time
import psutil
import queue
import gc
from typing import List, Dict, Any, Optional, Callable, Union, Tuple
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from collections import defaultdict, deque
import multiprocessing as mp

from ..utils.logging import SmellDiffusionLogger, performance_monitor
from ..utils.monitoring import get_metrics_collector


@dataclass
class ProcessingTask:
    """Individual processing task."""
    id: str
    prompt: str
    parameters: Dict[str, Any]
    priority: int = 1
    timeout: Optional[float] = None
    retry_count: int = 0
    max_retries: int = 3
    created_at: float = 0.0
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    
    def __post_init__(self):
        if self.created_at == 0.0:
            self.created_at = time.time()


@dataclass 
class ProcessingResult:
    """Processing result container."""
    task_id: str
    success: bool
    result: Any = None
    error: Optional[str] = None
    processing_time: float = 0.0
    worker_id: str = ""
    memory_usage: float = 0.0
    timestamp: float = 0.0
    
    def __post_init__(self):
        if self.timestamp == 0.0:
            self.timestamp = time.time()


@dataclass
class ResourceMetrics:
    """System resource metrics."""
    cpu_percent: float
    memory_percent: float
    memory_available_mb: float
    active_threads: int
    active_processes: int
    queue_size: int
    throughput_per_sec: float
    average_latency: float
    error_rate: float


class ResourcePool:
    """Intelligent resource pool management."""
    
    def __init__(self, max_threads: int = None, max_processes: int = None):
        self.logger = SmellDiffusionLogger("resource_pool")
        
        # Auto-detect optimal resource limits
        cpu_count = psutil.cpu_count(logical=True)
        memory_gb = psutil.virtual_memory().total / (1024**3)
        
        self.max_threads = max_threads or min(32, cpu_count * 4)
        self.max_processes = max_processes or min(8, max(1, cpu_count - 1))
        
        # Resource pools
        self.thread_pool = ThreadPoolExecutor(max_workers=self.max_threads)
        self.process_pool = ProcessPoolExecutor(max_workers=self.max_processes)
        
        # Resource tracking
        self.active_tasks = {}
        self.resource_history = deque(maxlen=1000)
        self.performance_metrics = defaultdict(float)
        
        # Adaptive scaling
        self.scaling_enabled = True
        self.last_scale_time = time.time()
        self.scale_cooldown = 30  # seconds
        
        # Circuit breaker for overload protection
        self.circuit_breaker = ResourceCircuitBreaker()
        
    def submit_task(self, task: ProcessingTask, executor_type: str = 'auto') -> concurrent.futures.Future:
        """Submit task with intelligent executor selection."""
        
        # Check circuit breaker
        if not self.circuit_breaker.allow_request():
            raise RuntimeError("Circuit breaker is open - system overloaded")
        
        # Auto-select executor type
        if executor_type == 'auto':
            executor_type = self._select_optimal_executor(task)
        
        # Submit to appropriate executor
        try:
            if executor_type == 'thread':
                future = self.thread_pool.submit(self._execute_task, task)
            elif executor_type == 'process':
                future = self.process_pool.submit(self._execute_task_in_process, task)
            else:
                raise ValueError(f"Unknown executor type: {executor_type}")
            
            # Track active task
            self.active_tasks[task.id] = {
                'future': future,
                'executor_type': executor_type,
                'submitted_at': time.time(),
                'task': task
            }
            
            # Update circuit breaker
            self.circuit_breaker.record_request()
            
            return future
            
        except Exception as e:
            self.circuit_breaker.record_failure()
            self.logger.log_error("task_submission", e, {"task_id": task.id})
            raise
    
    def _select_optimal_executor(self, task: ProcessingTask) -> str:
        """Select optimal executor based on task characteristics and system state."""
        
        # Get current resource usage
        cpu_percent = psutil.cpu_percent()
        memory_percent = psutil.virtual_memory().percent
        
        # Task complexity analysis
        prompt_complexity = len(task.prompt.split()) + task.prompt.count(',') * 2
        num_molecules = task.parameters.get('num_molecules', 1)
        
        # Decision logic
        if cpu_percent > 80:  # High CPU load - prefer threads
            return 'thread'
        elif memory_percent > 75:  # High memory pressure - prefer processes
            return 'process'
        elif prompt_complexity > 20 or num_molecules > 5:  # Complex tasks - use processes
            return 'process'
        else:  # Simple tasks - use threads
            return 'thread'
    
    def _execute_task(self, task: ProcessingTask) -> ProcessingResult:
        """Execute task in thread."""
        start_time = time.time()
        worker_id = f"thread_{threading.current_thread().ident}"
        
        try:
            # Record start
            task.started_at = start_time
            
            # Import here to avoid circular imports
            from ..core.smell_diffusion import SmellDiffusionGenerator
            
            # Create generator instance
            generator = SmellDiffusionGenerator()
            
            # Execute generation
            molecules = generator.generate(
                prompt=task.prompt,
                **task.parameters
            )
            
            # Record success
            processing_time = time.time() - start_time
            task.completed_at = time.time()
            
            self.circuit_breaker.record_success()
            
            return ProcessingResult(
                task_id=task.id,
                success=True,
                result=molecules,
                processing_time=processing_time,
                worker_id=worker_id,
                memory_usage=self._get_memory_usage()
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.circuit_breaker.record_failure()
            
            self.logger.log_error("task_execution", e, {
                "task_id": task.id,
                "worker_id": worker_id,
                "processing_time": processing_time
            })
            
            return ProcessingResult(
                task_id=task.id,
                success=False,
                error=str(e),
                processing_time=processing_time,
                worker_id=worker_id,
                memory_usage=self._get_memory_usage()
            )
        
        finally:
            # Cleanup task tracking
            if task.id in self.active_tasks:
                del self.active_tasks[task.id]
            
            # Force garbage collection for memory efficiency
            gc.collect()
    
    def _execute_task_in_process(self, task: ProcessingTask) -> ProcessingResult:
        """Execute task in separate process."""
        # This would be called in a separate process
        return self._execute_task(task)
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except:
            return 0.0
    
    def get_resource_metrics(self) -> ResourceMetrics:
        """Get current resource utilization metrics."""
        
        # System metrics
        cpu_percent = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        memory_available_mb = memory.available / 1024 / 1024
        
        # Pool metrics
        active_threads = len([t for t in self.active_tasks.values() if t['executor_type'] == 'thread'])
        active_processes = len([t for t in self.active_tasks.values() if t['executor_type'] == 'process'])
        queue_size = len(self.active_tasks)
        
        # Performance metrics
        throughput = self.performance_metrics.get('throughput_per_sec', 0.0)
        latency = self.performance_metrics.get('average_latency', 0.0)
        error_rate = self.performance_metrics.get('error_rate', 0.0)
        
        return ResourceMetrics(
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            memory_available_mb=memory_available_mb,
            active_threads=active_threads,
            active_processes=active_processes,
            queue_size=queue_size,
            throughput_per_sec=throughput,
            average_latency=latency,
            error_rate=error_rate
        )
    
    def adaptive_scale(self):
        """Adaptively scale resources based on current load."""
        
        if not self.scaling_enabled:
            return
        
        current_time = time.time()
        if current_time - self.last_scale_time < self.scale_cooldown:
            return
        
        metrics = self.get_resource_metrics()
        
        # Scale up conditions
        if (metrics.cpu_percent > 70 and metrics.queue_size > 5 and 
            metrics.error_rate < 0.1):
            
            # Increase thread pool if possible
            current_threads = self.thread_pool._max_workers
            if current_threads < self.max_threads:
                new_size = min(self.max_threads, current_threads + 2)
                self._resize_thread_pool(new_size)
                self.logger.logger.info(f"Scaled up thread pool to {new_size} workers")
        
        # Scale down conditions  
        elif (metrics.cpu_percent < 30 and metrics.queue_size < 2 and
              metrics.active_threads > 4):
            
            # Decrease thread pool
            current_threads = self.thread_pool._max_workers
            new_size = max(4, current_threads - 2)  # Minimum 4 threads
            self._resize_thread_pool(new_size)
            self.logger.logger.info(f"Scaled down thread pool to {new_size} workers")
        
        self.last_scale_time = current_time
    
    def _resize_thread_pool(self, new_size: int):
        """Resize thread pool (simplified implementation)."""
        # In practice, this would need careful coordination
        # For now, we just update the internal tracking
        self.thread_pool._max_workers = new_size
    
    def shutdown(self, wait: bool = True):
        """Shutdown resource pools gracefully."""
        self.logger.logger.info("Shutting down resource pools...")
        
        self.thread_pool.shutdown(wait=wait)
        self.process_pool.shutdown(wait=wait)
        
        # Clear tracking
        self.active_tasks.clear()
        
        self.logger.logger.info("Resource pools shutdown complete")


class ResourceCircuitBreaker:
    """Circuit breaker for resource protection."""
    
    def __init__(self, failure_threshold: int = 10, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        
        self.failure_count = 0
        self.last_failure_time = 0
        self.state = 'CLOSED'  # CLOSED, OPEN, HALF_OPEN
        
    def allow_request(self) -> bool:
        """Check if request should be allowed."""
        
        if self.state == 'CLOSED':
            return True
        elif self.state == 'OPEN':
            # Check if we should try recovery
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = 'HALF_OPEN'
                return True
            return False
        elif self.state == 'HALF_OPEN':
            return True
        
        return False
    
    def record_request(self):
        """Record a request."""
        pass  # Could implement rate limiting here
    
    def record_success(self):
        """Record successful request."""
        if self.state == 'HALF_OPEN':
            self.state = 'CLOSED'
            self.failure_count = 0
    
    def record_failure(self):
        """Record failed request."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = 'OPEN'


class ConcurrentMoleculeProcessor:
    """High-performance concurrent molecule generation processor."""
    
    def __init__(self, max_workers: int = None, enable_monitoring: bool = True):
        self.logger = SmellDiffusionLogger("concurrent_processor")
        
        # Initialize resource pool
        self.resource_pool = ResourcePool(max_threads=max_workers)
        
        # Task management
        self.task_queue = queue.PriorityQueue()
        self.results = {}
        self.task_counter = 0
        
        # Performance monitoring
        self.enable_monitoring = enable_monitoring
        if enable_monitoring:
            self.metrics_collector = get_metrics_collector()
        
        # Statistics
        self.stats = defaultdict(int)
        self.processing_times = deque(maxlen=1000)
        
        # Background monitoring
        self.monitoring_enabled = True
        self.monitoring_thread = threading.Thread(target=self._monitor_resources, daemon=True)
        self.monitoring_thread.start()
    
    @performance_monitor("concurrent_generation")
    def process_batch(self, prompts: List[str], parameters: Dict[str, Any] = None, 
                     priorities: List[int] = None) -> Dict[str, ProcessingResult]:
        """Process multiple prompts concurrently."""
        
        if not prompts:
            return {}
        
        parameters = parameters or {}
        priorities = priorities or [1] * len(prompts)
        
        # Create tasks
        tasks = []
        futures = []
        
        for i, prompt in enumerate(prompts):
            task_id = f"task_{self.task_counter}_{i}"
            self.task_counter += 1
            
            task = ProcessingTask(
                id=task_id,
                prompt=prompt,
                parameters=parameters.copy(),
                priority=priorities[i] if i < len(priorities) else 1
            )
            
            tasks.append(task)
        
        self.logger.logger.info(f"Starting concurrent processing of {len(tasks)} tasks")
        
        # Submit all tasks
        for task in tasks:
            try:
                future = self.resource_pool.submit_task(task)
                futures.append((task.id, future))
                self.stats['tasks_submitted'] += 1
                
            except Exception as e:
                self.logger.log_error("task_submission", e, {"task_id": task.id})
                self.stats['submission_errors'] += 1
        
        # Collect results
        results = {}
        
        for task_id, future in futures:
            try:
                result = future.result(timeout=300)  # 5 minute timeout
                results[task_id] = result
                
                if result.success:
                    self.stats['successful_tasks'] += 1
                    self.processing_times.append(result.processing_time)
                else:
                    self.stats['failed_tasks'] += 1
                
                # Update monitoring
                if self.enable_monitoring:
                    self.metrics_collector.increment_molecules_generated(
                        len(result.result) if result.result else 0
                    )
                
            except concurrent.futures.TimeoutError:
                self.logger.logger.error(f"Task {task_id} timed out")
                self.stats['timeout_errors'] += 1
                
                results[task_id] = ProcessingResult(
                    task_id=task_id,
                    success=False,
                    error="Task timeout"
                )
                
            except Exception as e:
                self.logger.log_error("result_collection", e, {"task_id": task_id})
                self.stats['collection_errors'] += 1
                
                results[task_id] = ProcessingResult(
                    task_id=task_id,
                    success=False,
                    error=str(e)
                )
        
        # Update performance metrics
        self._update_performance_metrics(results)
        
        self.logger.logger.info(
            f"Completed concurrent processing: {len(results)} results, "
            f"{self.stats['successful_tasks']} successful"
        )
        
        return results
    
    def process_single(self, prompt: str, parameters: Dict[str, Any] = None, 
                      priority: int = 1) -> ProcessingResult:
        """Process single prompt with high priority."""
        
        results = self.process_batch([prompt], parameters, [priority])
        return next(iter(results.values()))
    
    async def process_async(self, prompts: List[str], parameters: Dict[str, Any] = None) -> Dict[str, ProcessingResult]:
        """Async processing interface."""
        
        loop = asyncio.get_event_loop()
        
        # Run concurrent processing in thread pool
        return await loop.run_in_executor(
            None, 
            self.process_batch, 
            prompts, 
            parameters
        )
    
    def _update_performance_metrics(self, results: Dict[str, ProcessingResult]):
        """Update performance metrics based on results."""
        
        if not results:
            return
        
        successful_results = [r for r in results.values() if r.success]
        
        if successful_results:
            # Throughput
            total_time = max(r.processing_time for r in successful_results)
            throughput = len(successful_results) / max(total_time, 0.001)
            self.resource_pool.performance_metrics['throughput_per_sec'] = throughput
            
            # Average latency
            avg_latency = sum(r.processing_time for r in successful_results) / len(successful_results)
            self.resource_pool.performance_metrics['average_latency'] = avg_latency
        
        # Error rate
        total_tasks = len(results)
        failed_tasks = sum(1 for r in results.values() if not r.success)
        error_rate = failed_tasks / total_tasks if total_tasks > 0 else 0
        self.resource_pool.performance_metrics['error_rate'] = error_rate
    
    def _monitor_resources(self):
        """Background resource monitoring."""
        
        while self.monitoring_enabled:
            try:
                # Get resource metrics
                metrics = self.resource_pool.get_resource_metrics()
                
                # Log high resource usage
                if metrics.cpu_percent > 90:
                    self.logger.logger.warning(f"High CPU usage: {metrics.cpu_percent:.1f}%")
                
                if metrics.memory_percent > 85:
                    self.logger.logger.warning(f"High memory usage: {metrics.memory_percent:.1f}%")
                
                # Trigger adaptive scaling
                self.resource_pool.adaptive_scale()
                
                # Update monitoring metrics
                if self.enable_monitoring:
                    self.metrics_collector.update_active_connections(metrics.queue_size)
                
                time.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                self.logger.log_error("resource_monitoring", e)
                time.sleep(60)  # Back off on errors
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        
        metrics = self.resource_pool.get_resource_metrics()
        
        # Calculate additional stats
        avg_processing_time = (
            sum(self.processing_times) / len(self.processing_times)
            if self.processing_times else 0.0
        )
        
        success_rate = (
            self.stats['successful_tasks'] / max(self.stats['tasks_submitted'], 1) * 100
        )
        
        return {
            'resource_metrics': asdict(metrics),
            'task_statistics': dict(self.stats),
            'performance': {
                'average_processing_time': avg_processing_time,
                'success_rate_percent': success_rate,
                'total_processed': len(self.processing_times),
                'throughput_per_minute': len(self.processing_times) / max(avg_processing_time / 60, 0.001)
            },
            'circuit_breaker': {
                'state': self.resource_pool.circuit_breaker.state,
                'failure_count': self.resource_pool.circuit_breaker.failure_count
            }
        }
    
    def optimize_configuration(self) -> Dict[str, Any]:
        """Optimize processor configuration based on performance data."""
        
        stats = self.get_performance_stats()
        recommendations = {}
        
        # CPU optimization
        cpu_percent = stats['resource_metrics']['cpu_percent']
        if cpu_percent > 80:
            recommendations['reduce_concurrency'] = True
            recommendations['suggested_max_workers'] = max(1, self.resource_pool.max_threads - 2)
        elif cpu_percent < 40:
            recommendations['increase_concurrency'] = True
            recommendations['suggested_max_workers'] = min(32, self.resource_pool.max_threads + 2)
        
        # Memory optimization
        memory_percent = stats['resource_metrics']['memory_percent']
        if memory_percent > 85:
            recommendations['enable_memory_cleanup'] = True
            recommendations['reduce_batch_size'] = True
        
        # Error rate optimization
        error_rate = stats['resource_metrics']['error_rate']
        if error_rate > 0.1:
            recommendations['enable_retry_mechanism'] = True
            recommendations['reduce_timeout'] = False
        
        return {
            'current_stats': stats,
            'recommendations': recommendations,
            'optimization_timestamp': time.time()
        }
    
    def shutdown(self, wait: bool = True):
        """Shutdown processor gracefully."""
        
        self.logger.logger.info("Shutting down concurrent processor...")
        
        # Stop monitoring
        self.monitoring_enabled = False
        
        # Shutdown resource pool
        self.resource_pool.shutdown(wait=wait)
        
        self.logger.logger.info("Concurrent processor shutdown complete")


# Factory function
def create_concurrent_processor(max_workers: int = None, 
                              enable_monitoring: bool = True) -> ConcurrentMoleculeProcessor:
    """Create optimally configured concurrent processor."""
    
    # Auto-detect optimal worker count if not specified
    if max_workers is None:
        cpu_count = psutil.cpu_count(logical=True)
        memory_gb = psutil.virtual_memory().total / (1024**3)
        
        # Conservative approach for molecular generation
        max_workers = min(
            cpu_count * 2,  # CPU-bound factor
            int(memory_gb),  # Memory constraint
            16  # Practical upper limit
        )
    
    return ConcurrentMoleculeProcessor(
        max_workers=max_workers,
        enable_monitoring=enable_monitoring
    )