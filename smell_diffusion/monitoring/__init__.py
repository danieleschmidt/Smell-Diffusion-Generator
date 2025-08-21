"""
Advanced Monitoring and Alerting Package

Comprehensive monitoring framework for research-grade molecular generation:
- Real-time performance tracking
- Predictive anomaly detection  
- Multi-channel alerting
- Research analytics
- Auto-scaling recommendations
"""

from .advanced_alerting import (
    AdvancedAlertingSystem,
    Alert,
    AlertSeverity,
    AlertCategory,
    MonitoringMetric,
    AnomalyDetector,
    PerformanceTracker,
    ResearchAnalytics,
    AutoScalingEngine,
    create_advanced_alerting_system
)

__all__ = [
    'AdvancedAlertingSystem',
    'Alert',
    'AlertSeverity', 
    'AlertCategory',
    'MonitoringMetric',
    'AnomalyDetector',
    'PerformanceTracker',
    'ResearchAnalytics',
    'AutoScalingEngine',
    'create_advanced_alerting_system'
]

# Global monitoring instance
_global_alerting_system = None

def get_global_alerting_system() -> AdvancedAlertingSystem:
    """Get or create global alerting system instance"""
    global _global_alerting_system
    if _global_alerting_system is None:
        _global_alerting_system = create_advanced_alerting_system()
    return _global_alerting_system