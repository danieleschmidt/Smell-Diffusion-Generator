"""
Scalability and Performance Optimization Module

Production-ready scaling infrastructure for the Smell Diffusion Generator:
- Distributed generation across multiple workers
- Advanced load balancing and auto-scaling
- Resource optimization and monitoring
- High-throughput processing capabilities
- Real-time performance orchestration
"""

from .distributed_generation import (
    DistributedGenerator,
    ScalingConfiguration,
    LoadBalancer as BasicLoadBalancer,
    AutoScaler as BasicAutoScaler,
    ResourceOptimizer,
    create_distributed_generator
)

from .concurrent_processor import (
    ConcurrentMoleculeProcessor as ConcurrentProcessor,
    create_concurrent_processor
)

from .advanced_scaling import (
    AdvancedScalingOrchestrator,
    AutoScaler as AdvancedAutoScaler,
    LoadBalancer as AdvancedLoadBalancer,
    ResourceMonitor,
    AdvancedCachingSystem,
    CircuitBreaker,
    ScalingMode,
    ResourceType,
    ScalingMetrics,
    ScalingAction,
    create_advanced_scaling_system
)

__all__ = [
    "DistributedGenerator",
    "ScalingConfiguration", 
    "BasicLoadBalancer",
    "BasicAutoScaler",
    "ResourceOptimizer",
    "create_distributed_generator",
    "ConcurrentProcessor",
    "create_concurrent_processor",
    "AdvancedScalingOrchestrator",
    "AdvancedAutoScaler",
    "AdvancedLoadBalancer",
    "ResourceMonitor",
    "AdvancedCachingSystem",
    "CircuitBreaker",
    "ScalingMode",
    "ResourceType",
    "ScalingMetrics",
    "ScalingAction",
    "create_advanced_scaling_system"
]

# Global scaling system instance
_global_scaling_system = None

def get_global_scaling_system() -> AdvancedScalingOrchestrator:
    """Get or create global scaling system instance"""
    global _global_scaling_system
    if _global_scaling_system is None:
        _global_scaling_system = create_advanced_scaling_system(ScalingMode.CONSERVATIVE)
    return _global_scaling_system