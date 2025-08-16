"""
Advanced Health Monitoring and Alerting System
Real-time system monitoring, anomaly detection, and automated alerting for production deployment.
"""

import time
import asyncio
import threading
import json
import os
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import statistics
from abc import ABC, abstractmethod
from functools import wraps

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

from .logging import SmellDiffusionLogger
from .error_recovery import ErrorSeverity


class HealthStatus(Enum):
    """System health status levels."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    DOWN = "down"


class MetricType(Enum):
    """Types of metrics to monitor."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


@dataclass
class HealthCheck:
    """Individual health check configuration."""
    name: str
    check_function: Callable[[], bool]
    severity: ErrorSeverity = ErrorSeverity.ERROR
    timeout: float = 10.0
    enabled: bool = True
    description: str = ""


@dataclass
class Metric:
    """Metric data structure."""
    name: str
    value: Union[int, float]
    timestamp: float
    tags: Dict[str, str] = field(default_factory=dict)
    metric_type: MetricType = MetricType.GAUGE


@dataclass
class Alert:
    """Alert data structure."""
    id: str
    title: str
    description: str
    severity: ErrorSeverity
    timestamp: float
    source: str
    tags: Dict[str, str] = field(default_factory=dict)
    resolved: bool = False
    resolution_time: Optional[float] = None


class AlertChannel(ABC):
    """Abstract base class for alert channels."""
    
    @abstractmethod
    async def send_alert(self, alert: Alert) -> bool:
        """Send an alert through this channel."""
        pass


class LogAlertChannel(AlertChannel):
    """Log-based alert channel."""
    
    def __init__(self):
        self.logger = SmellDiffusionLogger("alerts")
    
    async def send_alert(self, alert: Alert) -> bool:
        """Send alert to log."""
        try:
            level_map = {
                ErrorSeverity.CRITICAL: "critical",
                ErrorSeverity.ERROR: "error",
                ErrorSeverity.WARNING: "warning",
                ErrorSeverity.INFO: "info"
            }
            
            level = level_map.get(alert.severity, "info")
            log_method = getattr(self.logger.logger, level, self.logger.logger.info)
            
            log_method(f"ALERT [{alert.severity.value.upper()}] {alert.title}: {alert.description}")
            return True
        except Exception as e:
            self.logger.log_error("alert_logging", e)
            return False


class WebhookAlertChannel(AlertChannel):
    """Webhook-based alert channel."""
    
    def __init__(self, webhook_url: str, headers: Optional[Dict[str, str]] = None):
        self.webhook_url = webhook_url
        self.headers = headers or {}
        self.logger = SmellDiffusionLogger("webhook_alerts")
    
    async def send_alert(self, alert: Alert) -> bool:
        """Send alert via webhook."""
        try:
            import aiohttp
            
            payload = {
                "id": alert.id,
                "title": alert.title,
                "description": alert.description,
                "severity": alert.severity.value,
                "timestamp": alert.timestamp,
                "source": alert.source,
                "tags": alert.tags
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.webhook_url,
                    json=payload,
                    headers=self.headers,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    return response.status < 400
                    
        except ImportError:
            self.logger.logger.warning("aiohttp not available for webhook alerts")
            return False
        except Exception as e:
            self.logger.log_error("webhook_alert", e)
            return False


class SystemMetrics:
    """System-level metrics collector."""
    
    def __init__(self):
        self.logger = SmellDiffusionLogger("system_metrics")
    
    def collect_system_metrics(self) -> Dict[str, float]:
        """Collect current system metrics."""
        try:
            if PSUTIL_AVAILABLE:
                # CPU metrics
                cpu_percent = psutil.cpu_percent(interval=1)
                cpu_count = psutil.cpu_count()
                
                # Memory metrics
                memory = psutil.virtual_memory()
                memory_percent = memory.percent
                memory_available_gb = memory.available / (1024**3)
                
                # Disk metrics
                disk = psutil.disk_usage('/')
                disk_percent = disk.percent
                disk_free_gb = disk.free / (1024**3)
                
                # Network metrics (if available)
                try:
                    network = psutil.net_io_counters()
                    network_bytes_sent = network.bytes_sent
                    network_bytes_recv = network.bytes_recv
                except:
                    network_bytes_sent = 0
                    network_bytes_recv = 0
                
                return {
                    'cpu_percent': cpu_percent,
                    'cpu_count': cpu_count,
                    'memory_percent': memory_percent,
                    'memory_available_gb': memory_available_gb,
                    'disk_percent': disk_percent,
                    'disk_free_gb': disk_free_gb,
                    'network_bytes_sent': network_bytes_sent,
                    'network_bytes_recv': network_bytes_recv
                }
            else:
                # Fallback metrics when psutil is not available
                return {
                    'cpu_percent': 50.0,  # Mock values
                    'cpu_count': os.cpu_count() or 2,
                    'memory_percent': 60.0,
                    'memory_available_gb': 4.0,
                    'disk_percent': 70.0,
                    'disk_free_gb': 10.0,
                    'network_bytes_sent': 0,
                    'network_bytes_recv': 0
                }
            
        except Exception as e:
            self.logger.log_error("system_metrics_collection", e)
            return {}


class AnomalyDetector:
    """Simple anomaly detection using statistical methods."""
    
    def __init__(self, window_size: int = 100, threshold_std: float = 2.0):
        self.window_size = window_size
        self.threshold_std = threshold_std
        self.metric_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window_size))
        self.logger = SmellDiffusionLogger("anomaly_detector")
    
    def add_metric(self, metric_name: str, value: float):
        """Add a metric value for anomaly detection."""
        self.metric_history[metric_name].append(value)
    
    def detect_anomalies(self) -> List[Dict[str, Any]]:
        """Detect anomalies in tracked metrics."""
        anomalies = []
        
        for metric_name, values in self.metric_history.items():
            if len(values) < 10:  # Need minimum history
                continue
            
            try:
                mean = statistics.mean(values)
                std_dev = statistics.stdev(values)
                current_value = values[-1]
                
                # Z-score calculation
                if std_dev > 0:
                    z_score = abs(current_value - mean) / std_dev
                    
                    if z_score > self.threshold_std:
                        severity = ErrorSeverity.WARNING if z_score < 3.0 else ErrorSeverity.CRITICAL
                        
                        anomalies.append({
                            'metric': metric_name,
                            'value': current_value,
                            'expected': mean,
                            'z_score': z_score,
                            'severity': severity.value,
                            'timestamp': time.time()
                        })
                        
            except statistics.StatisticsError:
                continue
        
        return anomalies


class HealthMonitor:
    """Comprehensive health monitoring system."""
    
    def __init__(self):
        self.logger = SmellDiffusionLogger("health_monitor")
        self.health_checks: Dict[str, HealthCheck] = {}
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.alerts: Dict[str, Alert] = {}
        self.alert_channels: List[AlertChannel] = []
        self.system_metrics = SystemMetrics()
        self.anomaly_detector = AnomalyDetector()
        
        # Monitoring state
        self._monitoring_active = False
        self._monitor_thread: Optional[threading.Thread] = None
        
        # Statistics
        self.start_time = time.time()
        self.total_health_checks = 0
        self.failed_health_checks = 0
        
        # Default alert channel
        self.add_alert_channel(LogAlertChannel())
        
        # Register default health checks
        self._register_default_health_checks()
    
    def _register_default_health_checks(self):
        """Register default system health checks."""
        
        def check_cpu_usage():
            """Check if CPU usage is reasonable."""
            try:
                if PSUTIL_AVAILABLE:
                    cpu_percent = psutil.cpu_percent(interval=1)
                    return cpu_percent < 90.0
                else:
                    return True  # Skip check if psutil not available
            except:
                return True
        
        def check_memory_usage():
            """Check if memory usage is reasonable."""
            try:
                if PSUTIL_AVAILABLE:
                    memory = psutil.virtual_memory()
                    return memory.percent < 95.0
                else:
                    return True  # Skip check if psutil not available
            except:
                return True
        
        def check_disk_space():
            """Check if disk space is sufficient."""
            try:
                if PSUTIL_AVAILABLE:
                    disk = psutil.disk_usage('/')
                    return disk.percent < 95.0
                else:
                    return True  # Skip check if psutil not available
            except:
                return True
        
        # Register checks
        self.register_health_check(HealthCheck(
            name="cpu_usage",
            check_function=check_cpu_usage,
            severity=ErrorSeverity.WARNING,
            description="CPU usage should be below 90%"
        ))
        
        self.register_health_check(HealthCheck(
            name="memory_usage",
            check_function=check_memory_usage,
            severity=ErrorSeverity.CRITICAL,
            description="Memory usage should be below 95%"
        ))
        
        self.register_health_check(HealthCheck(
            name="disk_space",
            check_function=check_disk_space,
            severity=ErrorSeverity.CRITICAL,
            description="Disk usage should be below 95%"
        ))
    
    def register_health_check(self, health_check: HealthCheck):
        """Register a health check."""
        self.health_checks[health_check.name] = health_check
        self.logger.logger.info(f"Registered health check: {health_check.name}")
    
    def add_alert_channel(self, channel: AlertChannel):
        """Add an alert channel."""
        self.alert_channels.append(channel)
        self.logger.logger.info(f"Added alert channel: {type(channel).__name__}")
    
    def record_metric(self, name: str, value: Union[int, float], 
                     tags: Optional[Dict[str, str]] = None):
        """Record a metric value."""
        metric = Metric(
            name=name,
            value=value,
            timestamp=time.time(),
            tags=tags or {}
        )
        
        self.metrics[name].append(metric)
        self.anomaly_detector.add_metric(name, value)
    
    async def create_alert(self, title: str, description: str, 
                          severity: ErrorSeverity, source: str = "health_monitor",
                          tags: Optional[Dict[str, str]] = None) -> str:
        """Create and send an alert."""
        alert_id = f"{source}_{int(time.time() * 1000)}"
        
        alert = Alert(
            id=alert_id,
            title=title,
            description=description,
            severity=severity,
            timestamp=time.time(),
            source=source,
            tags=tags or {}
        )
        
        self.alerts[alert_id] = alert
        
        # Send through all channels
        for channel in self.alert_channels:
            try:
                success = await channel.send_alert(alert)
                if not success:
                    self.logger.logger.warning(f"Failed to send alert through {type(channel).__name__}")
            except Exception as e:
                self.logger.log_error("alert_channel_error", e)
        
        return alert_id
    
    async def resolve_alert(self, alert_id: str):
        """Resolve an alert."""
        if alert_id in self.alerts:
            self.alerts[alert_id].resolved = True
            self.alerts[alert_id].resolution_time = time.time()
            self.logger.logger.info(f"Alert resolved: {alert_id}")
    
    async def run_health_checks(self) -> Dict[str, Any]:
        """Run all registered health checks."""
        results = {}
        overall_status = HealthStatus.HEALTHY
        
        for check_name, health_check in self.health_checks.items():
            if not health_check.enabled:
                continue
            
            self.total_health_checks += 1
            start_time = time.time()
            
            try:
                # Run check with timeout
                result = await asyncio.wait_for(
                    asyncio.get_event_loop().run_in_executor(
                        None, health_check.check_function
                    ),
                    timeout=health_check.timeout
                )
                
                duration = time.time() - start_time
                
                results[check_name] = {
                    'status': 'pass' if result else 'fail',
                    'duration': duration,
                    'description': health_check.description
                }
                
                if not result:
                    self.failed_health_checks += 1
                    
                    # Determine overall status based on severity
                    if health_check.severity == ErrorSeverity.CRITICAL:
                        overall_status = HealthStatus.CRITICAL
                    elif health_check.severity == ErrorSeverity.ERROR and overall_status != HealthStatus.CRITICAL:
                        overall_status = HealthStatus.WARNING
                    
                    # Create alert for failed check
                    await self.create_alert(
                        title=f"Health Check Failed: {check_name}",
                        description=f"Health check '{check_name}' failed. {health_check.description}",
                        severity=health_check.severity,
                        source="health_check",
                        tags={'check_name': check_name}
                    )
                
            except asyncio.TimeoutError:
                self.failed_health_checks += 1
                results[check_name] = {
                    'status': 'timeout',
                    'duration': health_check.timeout,
                    'description': f"Health check timed out after {health_check.timeout}s"
                }
                overall_status = HealthStatus.WARNING
                
            except Exception as e:
                self.failed_health_checks += 1
                results[check_name] = {
                    'status': 'error',
                    'duration': time.time() - start_time,
                    'description': f"Health check error: {str(e)}"
                }
                overall_status = HealthStatus.WARNING
                
                self.logger.log_error(f"health_check_{check_name}", e)
        
        return {
            'overall_status': overall_status.value,
            'checks': results,
            'timestamp': time.time()
        }
    
    def start_monitoring(self, interval: float = 30.0):
        """Start continuous monitoring."""
        if self._monitoring_active:
            return
        
        self._monitoring_active = True
        self._monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(interval,),
            daemon=True
        )
        self._monitor_thread.start()
        self.logger.logger.info(f"Health monitoring started with {interval}s interval")
    
    def stop_monitoring(self):
        """Stop continuous monitoring."""
        self._monitoring_active = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5.0)
        self.logger.logger.info("Health monitoring stopped")
    
    def _monitoring_loop(self, interval: float):
        """Main monitoring loop."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            while self._monitoring_active:
                # Collect system metrics
                system_metrics = self.system_metrics.collect_system_metrics()
                for metric_name, value in system_metrics.items():
                    self.record_metric(f"system.{metric_name}", value)
                
                # Run health checks
                health_results = loop.run_until_complete(self.run_health_checks())
                
                # Check for anomalies
                anomalies = self.anomaly_detector.detect_anomalies()
                for anomaly in anomalies:
                    loop.run_until_complete(self.create_alert(
                        title=f"Metric Anomaly: {anomaly['metric']}",
                        description=f"Metric {anomaly['metric']} is {anomaly['value']:.2f}, "
                                  f"expected ~{anomaly['expected']:.2f} (z-score: {anomaly['z_score']:.1f})",
                        severity=ErrorSeverity(anomaly['severity']),
                        source="anomaly_detector",
                        tags={'metric': anomaly['metric'], 'z_score': str(anomaly['z_score'])}
                    ))
                
                time.sleep(interval)
                
        except Exception as e:
            self.logger.log_error("monitoring_loop", e)
        finally:
            loop.close()
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get comprehensive health summary."""
        uptime = time.time() - self.start_time
        
        # Recent metrics
        recent_metrics = {}
        for metric_name, metric_history in self.metrics.items():
            if metric_history:
                recent_metrics[metric_name] = {
                    'current': metric_history[-1].value,
                    'avg_1h': self._calculate_metric_average(metric_name, 3600),
                    'count': len(metric_history)
                }
        
        # Active alerts
        active_alerts = [
            alert for alert in self.alerts.values() 
            if not alert.resolved
        ]
        
        return {
            'uptime_seconds': uptime,
            'monitoring_active': self._monitoring_active,
            'total_health_checks': self.total_health_checks,
            'failed_health_checks': self.failed_health_checks,
            'health_check_success_rate': (
                (self.total_health_checks - self.failed_health_checks) / 
                max(1, self.total_health_checks)
            ),
            'active_alerts': len(active_alerts),
            'total_alerts': len(self.alerts),
            'registered_checks': list(self.health_checks.keys()),
            'recent_metrics': recent_metrics,
            'alert_channels': len(self.alert_channels)
        }
    
    def _calculate_metric_average(self, metric_name: str, time_window: float) -> Optional[float]:
        """Calculate average metric value within time window."""
        if metric_name not in self.metrics:
            return None
        
        current_time = time.time()
        recent_values = [
            metric.value for metric in self.metrics[metric_name]
            if current_time - metric.timestamp <= time_window
        ]
        
        return statistics.mean(recent_values) if recent_values else None


# Global health monitor instance
global_health_monitor = HealthMonitor()


# Decorator for automatic metric recording
def monitor_performance(metric_name: str, record_duration: bool = True):
    """Decorator to automatically record performance metrics."""
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                
                if record_duration:
                    duration = time.time() - start_time
                    global_health_monitor.record_metric(
                        f"{metric_name}.duration", 
                        duration,
                        tags={'status': 'success'}
                    )
                
                global_health_monitor.record_metric(
                    f"{metric_name}.calls",
                    1,
                    tags={'status': 'success'}
                )
                
                return result
                
            except Exception as e:
                if record_duration:
                    duration = time.time() - start_time
                    global_health_monitor.record_metric(
                        f"{metric_name}.duration",
                        duration,
                        tags={'status': 'error'}
                    )
                
                global_health_monitor.record_metric(
                    f"{metric_name}.calls",
                    1,
                    tags={'status': 'error'}
                )
                
                raise
        
        return wrapper
    
    return decorator