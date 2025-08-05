"""Comprehensive monitoring and health check system for production deployment."""

import time
import psutil
import threading
import asyncio
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import defaultdict, deque
import json
from pathlib import Path

from .logging import SmellDiffusionLogger
from .config import get_config


@dataclass
class SystemMetrics:
    """System performance metrics."""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    memory_available_mb: float
    disk_usage_percent: float
    disk_free_gb: float
    load_average_1m: float
    load_average_5m: float
    load_average_15m: float
    network_bytes_sent: int
    network_bytes_recv: int
    process_count: int
    thread_count: int


@dataclass
class ApplicationMetrics:
    """Application-specific metrics."""
    timestamp: float
    total_requests: int
    successful_requests: int
    failed_requests: int
    average_response_time: float
    active_connections: int
    cache_hit_rate: float
    cache_size: int
    molecules_generated: int
    safety_evaluations: int
    error_rate: float
    uptime_seconds: float


@dataclass
class HealthStatus:
    """Overall system health status."""
    status: str  # healthy, degraded, unhealthy, critical
    timestamp: float
    system_metrics: SystemMetrics
    app_metrics: ApplicationMetrics
    checks: Dict[str, bool]
    alerts: List[str]
    recommendations: List[str]


class MetricsCollector:
    """Collects system and application metrics."""
    
    def __init__(self):
        self.logger = SmellDiffusionLogger("metrics_collector")
        self.start_time = time.time()
        self._app_counters = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'molecules_generated': 0,
            'safety_evaluations': 0
        }
        self._response_times = deque(maxlen=1000)
        self._active_connections = 0
        self._cache_stats = {'hits': 0, 'misses': 0, 'size': 0}
        
    def get_system_metrics(self) -> SystemMetrics:
        """Collect current system metrics."""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory metrics
            memory = psutil.virtual_memory()
            memory_used_mb = (memory.total - memory.available) / 1024 / 1024
            memory_available_mb = memory.available / 1024 / 1024
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            disk_usage_percent = (disk.used / disk.total) * 100
            disk_free_gb = disk.free / 1024 / 1024 / 1024
            
            # Load average
            load_avg = psutil.getloadavg()
            
            # Network metrics
            network = psutil.net_io_counters()
            
            # Process metrics
            process_count = len(psutil.pids())
            current_process = psutil.Process()
            thread_count = current_process.num_threads()
            
            return SystemMetrics(
                timestamp=time.time(),
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                memory_used_mb=memory_used_mb,
                memory_available_mb=memory_available_mb,
                disk_usage_percent=disk_usage_percent,
                disk_free_gb=disk_free_gb,
                load_average_1m=load_avg[0],
                load_average_5m=load_avg[1],
                load_average_15m=load_avg[2],
                network_bytes_sent=network.bytes_sent,
                network_bytes_recv=network.bytes_recv,
                process_count=process_count,
                thread_count=thread_count
            )
            
        except Exception as e:
            self.logger.log_error("system_metrics_collection", e)
            # Return default metrics on error
            return SystemMetrics(
                timestamp=time.time(),
                cpu_percent=0.0, memory_percent=0.0, memory_used_mb=0.0,
                memory_available_mb=0.0, disk_usage_percent=0.0, disk_free_gb=0.0,
                load_average_1m=0.0, load_average_5m=0.0, load_average_15m=0.0,
                network_bytes_sent=0, network_bytes_recv=0,
                process_count=0, thread_count=0
            )
    
    def get_application_metrics(self) -> ApplicationMetrics:
        """Collect current application metrics."""
        try:
            total_requests = self._app_counters['total_requests']
            successful_requests = self._app_counters['successful_requests']
            failed_requests = self._app_counters['failed_requests']
            
            # Calculate averages
            avg_response_time = (
                sum(self._response_times) / len(self._response_times)
                if self._response_times else 0.0
            )
            
            # Calculate error rate
            error_rate = (
                failed_requests / total_requests
                if total_requests > 0 else 0.0
            )
            
            # Calculate cache hit rate
            total_cache_requests = self._cache_stats['hits'] + self._cache_stats['misses']
            cache_hit_rate = (
                self._cache_stats['hits'] / total_cache_requests
                if total_cache_requests > 0 else 0.0
            )
            
            return ApplicationMetrics(
                timestamp=time.time(),
                total_requests=total_requests,
                successful_requests=successful_requests,
                failed_requests=failed_requests,
                average_response_time=avg_response_time,
                active_connections=self._active_connections,
                cache_hit_rate=cache_hit_rate,
                cache_size=self._cache_stats['size'],
                molecules_generated=self._app_counters['molecules_generated'],
                safety_evaluations=self._app_counters['safety_evaluations'],
                error_rate=error_rate,
                uptime_seconds=time.time() - self.start_time
            )
            
        except Exception as e:
            self.logger.log_error("app_metrics_collection", e)
            return ApplicationMetrics(
                timestamp=time.time(),
                total_requests=0, successful_requests=0, failed_requests=0,
                average_response_time=0.0, active_connections=0,
                cache_hit_rate=0.0, cache_size=0,
                molecules_generated=0, safety_evaluations=0,
                error_rate=0.0, uptime_seconds=0.0
            )
    
    # Counter update methods
    def increment_requests(self, success: bool = True):
        """Increment request counters."""
        self._app_counters['total_requests'] += 1
        if success:
            self._app_counters['successful_requests'] += 1
        else:
            self._app_counters['failed_requests'] += 1
    
    def record_response_time(self, duration: float):
        """Record response time."""
        self._response_times.append(duration)
    
    def increment_molecules_generated(self, count: int = 1):
        """Increment molecules generated counter."""
        self._app_counters['molecules_generated'] += count
    
    def increment_safety_evaluations(self, count: int = 1):
        """Increment safety evaluations counter."""
        self._app_counters['safety_evaluations'] += count
    
    def update_active_connections(self, count: int):
        """Update active connections count."""
        self._active_connections = count
    
    def update_cache_stats(self, hits: int, misses: int, size: int):
        """Update cache statistics."""
        self._cache_stats = {'hits': hits, 'misses': misses, 'size': size}


class HealthChecker:
    """Performs comprehensive health checks."""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self.logger = SmellDiffusionLogger("health_checker")
        self.config = get_config()
        
        # Health thresholds
        self.thresholds = {
            'cpu_critical': 90.0,
            'cpu_warning': 70.0,
            'memory_critical': 90.0,
            'memory_warning': 70.0,
            'disk_critical': 95.0,
            'disk_warning': 80.0,
            'error_rate_critical': 0.1,  # 10%
            'error_rate_warning': 0.05,  # 5%
            'response_time_critical': 5.0,  # 5 seconds
            'response_time_warning': 2.0,  # 2 seconds
        }
    
    def perform_health_check(self) -> HealthStatus:
        """Perform comprehensive health check."""
        try:
            system_metrics = self.metrics_collector.get_system_metrics()
            app_metrics = self.metrics_collector.get_application_metrics()
            
            checks = self._run_health_checks(system_metrics, app_metrics)
            alerts = self._generate_alerts(system_metrics, app_metrics, checks)
            recommendations = self._generate_recommendations(system_metrics, app_metrics, checks)
            
            # Determine overall status
            status = self._determine_overall_status(checks, alerts)
            
            health_status = HealthStatus(
                status=status,
                timestamp=time.time(),
                system_metrics=system_metrics,
                app_metrics=app_metrics,
                checks=checks,
                alerts=alerts,
                recommendations=recommendations
            )
            
            # Log health status
            self.logger.logger.info(f"Health check completed: {status}")
            if alerts:
                self.logger.logger.warning(f"Health alerts: {'; '.join(alerts)}")
            
            return health_status
            
        except Exception as e:
            self.logger.log_error("health_check_execution", e)
            return HealthStatus(
                status="error",
                timestamp=time.time(),
                system_metrics=SystemMetrics(time.time(), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
                app_metrics=ApplicationMetrics(time.time(), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
                checks={"health_check_failed": False},
                alerts=[f"Health check failed: {str(e)}"],
                recommendations=["Check system logs and restart monitoring system"]
            )
    
    def _run_health_checks(self, sys_metrics: SystemMetrics, app_metrics: ApplicationMetrics) -> Dict[str, bool]:
        """Run individual health checks."""
        checks = {}
        
        # System checks
        checks['cpu_healthy'] = sys_metrics.cpu_percent < self.thresholds['cpu_critical']
        checks['memory_healthy'] = sys_metrics.memory_percent < self.thresholds['memory_critical']
        checks['disk_healthy'] = sys_metrics.disk_usage_percent < self.thresholds['disk_critical']
        checks['load_healthy'] = sys_metrics.load_average_1m < psutil.cpu_count() * 2
        
        # Application checks
        checks['error_rate_healthy'] = app_metrics.error_rate < self.thresholds['error_rate_critical']
        checks['response_time_healthy'] = app_metrics.average_response_time < self.thresholds['response_time_critical']
        checks['uptime_healthy'] = app_metrics.uptime_seconds > 60  # At least 1 minute uptime
        
        # Service-specific checks
        checks['cache_functioning'] = app_metrics.cache_hit_rate > 0.1 or app_metrics.total_requests < 10
        checks['molecules_generating'] = app_metrics.molecules_generated > 0 or app_metrics.total_requests < 5
        
        return checks
    
    def _generate_alerts(self, sys_metrics: SystemMetrics, app_metrics: ApplicationMetrics, checks: Dict[str, bool]) -> List[str]:
        """Generate alerts based on metrics and checks."""
        alerts = []
        
        # Critical system alerts
        if sys_metrics.cpu_percent > self.thresholds['cpu_critical']:
            alerts.append(f"CRITICAL: CPU usage at {sys_metrics.cpu_percent:.1f}%")
        
        if sys_metrics.memory_percent > self.thresholds['memory_critical']:
            alerts.append(f"CRITICAL: Memory usage at {sys_metrics.memory_percent:.1f}%")
        
        if sys_metrics.disk_usage_percent > self.thresholds['disk_critical']:
            alerts.append(f"CRITICAL: Disk usage at {sys_metrics.disk_usage_percent:.1f}%")
        
        # Application alerts
        if app_metrics.error_rate > self.thresholds['error_rate_critical']:
            alerts.append(f"CRITICAL: Error rate at {app_metrics.error_rate:.1%}")
        
        if app_metrics.average_response_time > self.thresholds['response_time_critical']:
            alerts.append(f"CRITICAL: Response time at {app_metrics.average_response_time:.1f}s")
        
        # Warning level alerts
        if sys_metrics.cpu_percent > self.thresholds['cpu_warning']:
            alerts.append(f"WARNING: High CPU usage at {sys_metrics.cpu_percent:.1f}%")
        
        if sys_metrics.memory_percent > self.thresholds['memory_warning']:
            alerts.append(f"WARNING: High memory usage at {sys_metrics.memory_percent:.1f}%")
        
        return alerts
    
    def _generate_recommendations(self, sys_metrics: SystemMetrics, app_metrics: ApplicationMetrics, checks: Dict[str, bool]) -> List[str]:
        """Generate recommendations based on current state."""
        recommendations = []
        
        # Performance recommendations
        if sys_metrics.cpu_percent > self.thresholds['cpu_warning']:
            recommendations.append("Consider scaling up CPU resources or optimizing CPU-intensive operations")
        
        if sys_metrics.memory_percent > self.thresholds['memory_warning']:
            recommendations.append("Consider increasing memory allocation or implementing memory optimization")
        
        if app_metrics.cache_hit_rate < 0.5 and app_metrics.total_requests > 100:
            recommendations.append("Cache hit rate is low - consider increasing cache size or reviewing cache strategy")
        
        if app_metrics.error_rate > self.thresholds['error_rate_warning']:
            recommendations.append("Error rate is elevated - review application logs and error handling")
        
        if app_metrics.average_response_time > self.thresholds['response_time_warning']:
            recommendations.append("Response times are high - consider performance optimization or scaling")
        
        # Maintenance recommendations
        if sys_metrics.disk_usage_percent > self.thresholds['disk_warning']:
            recommendations.append("Disk usage is high - consider cleanup or expanding storage")
        
        if app_metrics.uptime_seconds > 86400 * 7:  # 7 days
            recommendations.append("Consider scheduled maintenance restart for optimal performance")
        
        return recommendations
    
    def _determine_overall_status(self, checks: Dict[str, bool], alerts: List[str]) -> str:
        """Determine overall health status."""
        failed_checks = sum(1 for passed in checks.values() if not passed)
        critical_alerts = sum(1 for alert in alerts if 'CRITICAL' in alert)
        
        if critical_alerts > 0 or failed_checks > 3:
            return "critical"
        elif failed_checks > 1 or any('CRITICAL' in alert for alert in alerts):
            return "unhealthy"
        elif failed_checks > 0 or any('WARNING' in alert for alert in alerts):
            return "degraded"
        else:
            return "healthy"


class MonitoringDashboard:
    """Real-time monitoring dashboard."""
    
    def __init__(self, metrics_collector: MetricsCollector, health_checker: HealthChecker):
        self.metrics_collector = metrics_collector
        self.health_checker = health_checker
        self.logger = SmellDiffusionLogger("monitoring_dashboard")
        
        # Store recent metrics for trending
        self.metrics_history = deque(maxlen=1000)
        self.is_running = False
        self.monitoring_thread = None
    
    def start_monitoring(self, interval: int = 60):
        """Start continuous monitoring."""
        if self.is_running:
            return
        
        self.is_running = True
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(interval,),
            daemon=True
        )
        self.monitoring_thread.start()
        self.logger.logger.info(f"Started monitoring with {interval}s interval")
    
    def stop_monitoring(self):
        """Stop continuous monitoring."""
        self.is_running = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        self.logger.logger.info("Stopped monitoring")
    
    def _monitoring_loop(self, interval: int):
        """Main monitoring loop."""
        while self.is_running:
            try:
                # Collect metrics
                health_status = self.health_checker.perform_health_check()
                
                # Store in history
                self.metrics_history.append(health_status)
                
                # Log significant events
                if health_status.status in ['critical', 'unhealthy']:
                    self.logger.logger.error(f"System status: {health_status.status}")
                elif health_status.status == 'degraded':
                    self.logger.logger.warning(f"System status: {health_status.status}")
                
                # Save metrics to file for external monitoring
                self._save_metrics_snapshot(health_status)
                
                time.sleep(interval)
                
            except Exception as e:
                self.logger.log_error("monitoring_loop", e)
                time.sleep(interval)
    
    def _save_metrics_snapshot(self, health_status: HealthStatus):
        """Save current metrics snapshot to file."""
        try:
            # Create monitoring directory
            monitoring_dir = Path.home() / ".smell_diffusion" / "monitoring"
            monitoring_dir.mkdir(parents=True, exist_ok=True)
            
            # Save current snapshot
            snapshot_file = monitoring_dir / "current_metrics.json"
            with open(snapshot_file, 'w') as f:
                json.dump(asdict(health_status), f, indent=2, default=str)
            
            # Save historical data (keep last 24 hours)
            history_file = monitoring_dir / "metrics_history.json"
            recent_history = [
                asdict(status) for status in list(self.metrics_history)[-1440:]  # Last 24 hours at 1-min intervals
            ]
            with open(history_file, 'w') as f:
                json.dump(recent_history, f, indent=2, default=str)
                
        except Exception as e:
            self.logger.log_error("metrics_snapshot_save", e)
    
    def get_current_status(self) -> Optional[HealthStatus]:
        """Get current health status."""
        return self.metrics_history[-1] if self.metrics_history else None
    
    def get_metrics_summary(self, duration_minutes: int = 60) -> Dict[str, Any]:
        """Get metrics summary for specified duration."""
        cutoff_time = time.time() - (duration_minutes * 60)
        recent_metrics = [
            status for status in self.metrics_history
            if status.timestamp > cutoff_time
        ]
        
        if not recent_metrics:
            return {"error": "No metrics available for specified duration"}
        
        # Calculate averages
        avg_cpu = sum(m.system_metrics.cpu_percent for m in recent_metrics) / len(recent_metrics)
        avg_memory = sum(m.system_metrics.memory_percent for m in recent_metrics) / len(recent_metrics)
        avg_response_time = sum(m.app_metrics.average_response_time for m in recent_metrics) / len(recent_metrics)
        
        # Count status occurrences
        status_counts = defaultdict(int)
        for m in recent_metrics:
            status_counts[m.status] += 1
        
        return {
            "duration_minutes": duration_minutes,
            "sample_count": len(recent_metrics),
            "averages": {
                "cpu_percent": avg_cpu,
                "memory_percent": avg_memory,
                "response_time_seconds": avg_response_time
            },
            "status_distribution": dict(status_counts),
            "latest_status": recent_metrics[-1].status,
            "total_requests": recent_metrics[-1].app_metrics.total_requests,
            "total_molecules_generated": recent_metrics[-1].app_metrics.molecules_generated
        }


# Global monitoring instances
_metrics_collector = None
_health_checker = None
_monitoring_dashboard = None


def get_metrics_collector() -> MetricsCollector:
    """Get global metrics collector instance."""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
    return _metrics_collector


def get_health_checker() -> HealthChecker:
    """Get global health checker instance."""
    global _health_checker
    if _health_checker is None:
        _health_checker = HealthChecker(get_metrics_collector())
    return _health_checker


def get_monitoring_dashboard() -> MonitoringDashboard:
    """Get global monitoring dashboard instance."""
    global _monitoring_dashboard
    if _monitoring_dashboard is None:
        _monitoring_dashboard = MonitoringDashboard(
            get_metrics_collector(),
            get_health_checker()
        )
    return _monitoring_dashboard


def start_monitoring(interval: int = 60):
    """Start global monitoring system."""
    dashboard = get_monitoring_dashboard()
    dashboard.start_monitoring(interval)


def stop_monitoring():
    """Stop global monitoring system."""
    if _monitoring_dashboard:
        _monitoring_dashboard.stop_monitoring()


def get_current_health() -> Optional[HealthStatus]:
    """Get current system health status."""
    dashboard = get_monitoring_dashboard()
    return dashboard.get_current_status()


if __name__ == "__main__":
    # CLI for monitoring system
    import argparse
    
    parser = argparse.ArgumentParser(description="Smell Diffusion Monitoring System")
    parser.add_argument("action", choices=["start", "status", "summary"], 
                       help="Monitoring action")
    parser.add_argument("--interval", type=int, default=60, 
                       help="Monitoring interval in seconds")
    parser.add_argument("--duration", type=int, default=60,
                       help="Summary duration in minutes")
    
    args = parser.parse_args()
    
    if args.action == "start":
        start_monitoring(args.interval)
        print(f"Monitoring started with {args.interval}s interval")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            stop_monitoring()
            print("Monitoring stopped")
    
    elif args.action == "status":
        health_checker = get_health_checker()
        status = health_checker.perform_health_check()
        print(f"System Status: {status.status.upper()}")
        if status.alerts:
            print("Alerts:")
            for alert in status.alerts:
                print(f"  - {alert}")
        if status.recommendations:
            print("Recommendations:")
            for rec in status.recommendations:
                print(f"  - {rec}")
    
    elif args.action == "summary":
        dashboard = get_monitoring_dashboard()
        summary = dashboard.get_metrics_summary(args.duration)
        print(json.dumps(summary, indent=2))
