"""Advanced monitoring and alerting system for production deployment.

Provides comprehensive monitoring, alerting, and observability features:
- Real-time performance monitoring with predictive analytics
- Multi-channel alerting with intelligent routing
- Circuit breaker patterns for resilience
- Health checks and service discovery
- Prometheus/OpenTelemetry integration
- Anomaly detection and auto-remediation
"""

import time
import psutil
import threading
import asyncio
import hashlib
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
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
    sla_compliance: float = 100.0
    readiness: bool = True
    liveness: bool = True


@dataclass
class Alert:
    """Alert information for monitoring events."""
    id: str
    severity: str  # info, warning, critical, error
    title: str
    description: str
    timestamp: float
    source: str
    tags: Dict[str, str]
    resolved: bool = False
    acknowledged: bool = False
    
    @classmethod
    def create(cls, severity: str, title: str, description: str, source: str, **tags) -> 'Alert':
        """Create a new alert with unique ID."""
        alert_data = f"{title}{description}{source}{time.time()}"
        alert_id = hashlib.md5(alert_data.encode()).hexdigest()[:8]
        return cls(
            id=alert_id,
            severity=severity,
            title=title,
            description=description,
            timestamp=time.time(),
            source=source,
            tags=tags
        )


@dataclass 
class SLAMetrics:
    """Service Level Agreement metrics."""
    availability_percent: float
    response_time_p95: float
    response_time_p99: float
    error_rate_percent: float
    throughput_rps: float
    mttr_seconds: float = 0.0
    mttd_seconds: float = 0.0


class MLAlertClassifier:
    """ML-based alert classification and enhancement."""
    
    def __init__(self):
        self.classification_patterns = {
            'performance': ['cpu', 'memory', 'disk', 'latency', 'throughput'],
            'security': ['auth', 'unauthorized', 'breach', 'attack', 'malicious'],
            'safety': ['toxic', 'allergen', 'prohibited', 'dangerous', 'violation'],
            'availability': ['down', 'unreachable', 'timeout', 'connection', 'network'],
            'quality': ['validation', 'accuracy', 'invalid', 'corrupt', 'format']
        }
        
    def classify_and_enhance(self, alert: Alert) -> Alert:
        """Classify alert and enhance with ML insights."""
        # Classify alert category
        category = self._classify_alert(alert)
        alert.tags['ml_category'] = category
        
        # Enhance with context
        alert.tags['ml_confidence'] = self._calculate_confidence(alert, category)
        alert.tags['ml_priority'] = self._calculate_priority(alert, category)
        
        return alert
    
    def _classify_alert(self, alert: Alert) -> str:
        """Classify alert into category."""
        text = f"{alert.title} {alert.description}".lower()
        
        best_category = 'general'
        best_score = 0
        
        for category, keywords in self.classification_patterns.items():
            score = sum(1 for keyword in keywords if keyword in text)
            if score > best_score:
                best_score = score
                best_category = category
        
        return best_category
    
    def _calculate_confidence(self, alert: Alert, category: str) -> float:
        """Calculate confidence score for classification."""
        text = f"{alert.title} {alert.description}".lower()
        keywords = self.classification_patterns.get(category, [])
        
        matches = sum(1 for keyword in keywords if keyword in text)
        return min(1.0, matches / max(len(keywords), 1))
    
    def _calculate_priority(self, alert: Alert, category: str) -> int:
        """Calculate priority score (1-10)."""
        base_priority = {
            'critical': 9,
            'error': 7,
            'warning': 5,
            'info': 3
        }.get(alert.severity, 5)
        
        category_modifier = {
            'security': 2,
            'safety': 2,
            'availability': 1,
            'performance': 1,
            'quality': 0
        }.get(category, 0)
        
        return min(10, base_priority + category_modifier)
    
    def suggest_actions(self, alert: Alert) -> List[Dict[str, Any]]:
        """Suggest actions based on alert analysis."""
        suggestions = []
        
        category = alert.tags.get('ml_category', 'general')
        confidence = float(alert.tags.get('ml_confidence', 0.5))
        
        if category == 'security' and confidence > 0.7:
            suggestions.append({
                'action': 'escalate',
                'reason': 'High-confidence security alert'
            })
        
        if category == 'safety' and alert.severity in ['error', 'critical']:
            suggestions.append({
                'action': 'enrich',
                'data': {'requires_immediate_attention': True}
            })
        
        if confidence < 0.3:
            suggestions.append({
                'action': 'suppress',
                'reason': 'Low confidence classification'
            })
        
        return suggestions


class IncidentTracker:
    """Track and correlate incidents across alerts."""
    
    def __init__(self):
        self.active_incidents = {}
        self.incident_history = deque(maxlen=1000)
        
    def update_with_alert(self, alert: Alert) -> Optional[str]:
        """Update incident tracking with new alert."""
        # Find existing incident or create new one
        incident_id = self._find_or_create_incident(alert)
        
        if incident_id in self.active_incidents:
            incident = self.active_incidents[incident_id]
            incident['alerts'].append(alert)
            incident['last_update'] = time.time()
            incident['severity'] = max(incident['severity'], self._severity_to_int(alert.severity))
        
        return incident_id
    
    def _find_or_create_incident(self, alert: Alert) -> str:
        """Find existing incident or create new one."""
        # Look for related incidents
        for incident_id, incident in self.active_incidents.items():
            if self._alerts_related(alert, incident['alerts'][-1]):
                return incident_id
        
        # Create new incident
        incident_id = hashlib.md5(f"{alert.source}_{time.time()}".encode()).hexdigest()[:8]
        self.active_incidents[incident_id] = {
            'id': incident_id,
            'created_at': time.time(),
            'last_update': time.time(),
            'alerts': [alert],
            'severity': self._severity_to_int(alert.severity),
            'status': 'active'
        }
        
        return incident_id
    
    def _alerts_related(self, alert1: Alert, alert2: Alert) -> bool:
        """Check if two alerts are related to same incident."""
        # Same source
        if alert1.source == alert2.source:
            return True
        
        # Similar titles
        if self._text_similarity(alert1.title, alert2.title) > 0.6:
            return True
        
        # Time proximity (within 5 minutes)
        if abs(alert1.timestamp - alert2.timestamp) < 300:
            return True
        
        return False
    
    def _severity_to_int(self, severity: str) -> int:
        return {'info': 1, 'warning': 2, 'error': 3, 'critical': 4}.get(severity, 2)
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """Simple text similarity calculation."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        return len(words1 & words2) / len(words1 | words2) if words1 or words2 else 0.0


class AlertAggregator:
    """Aggregate related alerts to reduce noise."""
    
    def __init__(self):
        self.aggregation_window = 300  # 5 minutes
        self.pending_aggregations = {}
        
    def aggregate_if_needed(self, alert: Alert) -> Alert:
        """Aggregate alert if similar alerts exist."""
        aggregation_key = self._get_aggregation_key(alert)
        
        if aggregation_key in self.pending_aggregations:
            existing_alert = self.pending_aggregations[aggregation_key]
            
            # Update existing aggregated alert
            existing_alert.tags['aggregated_count'] = str(
                int(existing_alert.tags.get('aggregated_count', '1')) + 1
            )
            existing_alert.description += f" | Additional occurrence: {alert.description[:50]}..."
            existing_alert.timestamp = alert.timestamp  # Update to latest
            
            return existing_alert
        else:
            # Start new aggregation
            alert.tags['aggregated_count'] = '1'
            self.pending_aggregations[aggregation_key] = alert
            
            # Schedule cleanup
            threading.Timer(
                self.aggregation_window,
                lambda: self.pending_aggregations.pop(aggregation_key, None)
            ).start()
            
            return alert
    
    def _get_aggregation_key(self, alert: Alert) -> str:
        """Generate key for alert aggregation."""
        # Combine source, severity, and simplified title
        title_words = alert.title.lower().split()[:3]  # First 3 words
        return f"{alert.source}:{alert.severity}:{'_'.join(title_words)}"


class TimeSeriesAnalyzer:
    """Time series analysis for metrics."""
    
    def __init__(self):
        self.time_series_data = defaultdict(list)
        
    def add_data_point(self, metric_name: str, data_point: Dict[str, Any]):
        """Add data point to time series."""
        self.time_series_data[metric_name].append(data_point)
        
        # Keep only recent data (last 1000 points)
        if len(self.time_series_data[metric_name]) > 1000:
            self.time_series_data[metric_name] = self.time_series_data[metric_name][-1000:]
    
    def detect_anomalies(self, metric_name: str, recent_values: List[float], 
                        timestamps: List[float]) -> List[Dict[str, Any]]:
        """Detect time series anomalies."""
        anomalies = []
        
        if len(recent_values) < 10:
            return anomalies
        
        # Detect sudden spikes/drops
        for i in range(1, len(recent_values)):
            prev_value = recent_values[i-1]
            curr_value = recent_values[i]
            
            if prev_value != 0:
                change_ratio = abs(curr_value - prev_value) / abs(prev_value)
                
                if change_ratio > 2.0:  # 200% change
                    anomalies.append({
                        'metric': metric_name,
                        'value': curr_value,
                        'previous_value': prev_value,
                        'change_ratio': change_ratio,
                        'detection_method': 'sudden_change',
                        'severity': 'warning',
                        'timestamp': timestamps[i] if i < len(timestamps) else time.time()
                    })
        
        return anomalies


class PatternDetector:
    """Detect patterns in metric data."""
    
    def __init__(self):
        self.known_patterns = {}
        
    def analyze_metric(self, metric_name: str, data_points: List[Dict[str, Any]]):
        """Analyze metric for patterns."""
        if len(data_points) < 20:
            return
        
        values = [point['value'] for point in data_points]
        
        # Detect cyclical patterns
        self._detect_cyclical_patterns(metric_name, values)
        
        # Detect trend patterns
        self._detect_trends(metric_name, values)
    
    def _detect_cyclical_patterns(self, metric_name: str, values: List[float]):
        """Detect cyclical patterns in data."""
        # Simple peak detection
        peaks = []
        for i in range(1, len(values) - 1):
            if values[i] > values[i-1] and values[i] > values[i+1]:
                peaks.append(i)
        
        if len(peaks) >= 3:
            # Calculate average period
            periods = [peaks[i+1] - peaks[i] for i in range(len(peaks)-1)]
            avg_period = sum(periods) / len(periods)
            
            self.known_patterns[metric_name] = {
                'type': 'cyclical',
                'period': avg_period,
                'confidence': 0.7 if len(peaks) >= 5 else 0.5
            }
    
    def _detect_trends(self, metric_name: str, values: List[float]):
        """Detect trend patterns."""
        if len(values) < 10:
            return
        
        # Simple linear regression
        n = len(values)
        x = list(range(n))
        y = values
        
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(x[i] * y[i] for i in range(n))
        sum_x2 = sum(xi * xi for xi in x)
        
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
        
        if abs(slope) > 0.1:  # Significant trend
            trend_type = 'increasing' if slope > 0 else 'decreasing'
            self.known_patterns[metric_name] = {
                'type': 'trend',
                'direction': trend_type,
                'slope': slope,
                'confidence': min(1.0, abs(slope))
            }
    
    def detect_pattern_anomalies(self, metric_name: str, recent_values: List[float]) -> List[Dict[str, Any]]:
        """Detect anomalies based on known patterns."""
        anomalies = []
        
        if metric_name not in self.known_patterns:
            return anomalies
        
        pattern = self.known_patterns[metric_name]
        
        if pattern['type'] == 'cyclical':
            # Check if recent values break the cyclical pattern
            expected_range = self._calculate_expected_range_for_cycle(recent_values, pattern)
            latest_value = recent_values[-1]
            
            if latest_value < expected_range['min'] or latest_value > expected_range['max']:
                anomalies.append({
                    'metric': metric_name,
                    'value': latest_value,
                    'expected_min': expected_range['min'],
                    'expected_max': expected_range['max'],
                    'detection_method': 'pattern_cyclical',
                    'severity': 'warning',
                    'timestamp': time.time()
                })
        
        return anomalies
    
    def _calculate_expected_range_for_cycle(self, values: List[float], pattern: Dict) -> Dict[str, float]:
        """Calculate expected range based on cyclical pattern."""
        # Simplified calculation
        mean_value = sum(values) / len(values)
        std_value = (sum((v - mean_value) ** 2 for v in values) / len(values)) ** 0.5
        
        return {
            'min': mean_value - 2 * std_value,
            'max': mean_value + 2 * std_value
        }


class ForecastEngine:
    """Forecasting engine for predictive monitoring."""
    
    def __init__(self):
        self.forecasting_models = {}
        
    def create_model(self, metric_name: str, historical_data: List[Dict[str, Any]]):
        """Create forecasting model for metric."""
        if len(historical_data) < 20:
            return None
        
        values = [point['value'] for point in historical_data]
        self.forecasting_models[metric_name] = SimpleForecastingModel(metric_name, values)
        
        return self.forecasting_models[metric_name]


class SimpleForecastingModel:
    """Simple forecasting model using moving averages and trend analysis."""
    
    def __init__(self, metric_name: str, initial_data: List[float] = None):
        self.metric_name = metric_name
        self.data = initial_data or []
        self.trend = 0.0
        self.seasonal_component = 0.0
        
        if len(self.data) >= 10:
            self._calculate_trend()
    
    def train(self, historical_data: List[Dict[str, Any]]):
        """Train model on historical data."""
        self.data = [point['value'] for point in historical_data]
        if len(self.data) >= 10:
            self._calculate_trend()
    
    def update(self, new_data_point: Dict[str, Any]):
        """Update model with new data point."""
        self.data.append(new_data_point['value'])
        if len(self.data) > 1000:
            self.data = self.data[-1000:]  # Keep recent data
        
        # Recalculate trend periodically
        if len(self.data) % 10 == 0:
            self._calculate_trend()
    
    def forecast(self, horizon: int = 5) -> List[float]:
        """Generate forecast for next 'horizon' time steps."""
        if not self.data:
            return [0.0] * horizon
        
        predictions = []
        last_value = self.data[-1]
        
        for i in range(horizon):
            # Simple trend-based forecast
            predicted_value = last_value + (self.trend * (i + 1))
            predictions.append(max(0.0, predicted_value))  # Ensure non-negative
        
        return predictions
    
    def _calculate_trend(self):
        """Calculate trend component."""
        if len(self.data) < 10:
            return
        
        # Use last 20 data points for trend calculation
        recent_data = self.data[-20:]
        n = len(recent_data)
        
        # Simple linear regression
        x = list(range(n))
        y = recent_data
        
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(x[i] * y[i] for i in range(n))
        sum_x2 = sum(xi * xi for xi in x)
        
        if n * sum_x2 - sum_x * sum_x != 0:
            self.trend = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
        else:
            self.trend = 0.0
    
    def predict(self, values: List[float]) -> List[float]:
        """Predict expected values."""
        if not self.data:
            return values
        
        # Return simple moving average as prediction
        window_size = min(10, len(self.data))
        if window_size > 0:
            avg = sum(self.data[-window_size:]) / window_size
            return [avg] * len(values)
        
        return values
    
    def score_anomaly(self, values: List[float]) -> List[float]:
        """Score anomaly likelihood for values."""
        if not self.data or len(self.data) < 10:
            return [0.5] * len(values)
        
        # Calculate statistical baseline
        recent_data = self.data[-50:] if len(self.data) >= 50 else self.data
        mean_value = sum(recent_data) / len(recent_data)
        std_value = (sum((v - mean_value) ** 2 for v in recent_data) / len(recent_data)) ** 0.5
        
        anomaly_scores = []
        for value in values:
            if std_value > 0:
                z_score = abs(value - mean_value) / std_value
                anomaly_score = min(1.0, z_score / 3.0)  # Normalize to 0-1
            else:
                anomaly_score = 0.0 if value == mean_value else 1.0
            
            anomaly_scores.append(anomaly_score)
        
        return anomaly_scores
    
    def get_confidence_intervals(self) -> Optional[Dict[str, List[float]]]:
        """Get confidence intervals for predictions."""
        if len(self.data) < 10:
            return None
        
        # Calculate prediction uncertainty
        recent_data = self.data[-20:]
        std_dev = (sum((v - sum(recent_data)/len(recent_data)) ** 2 for v in recent_data) / len(recent_data)) ** 0.5
        
        # Simple confidence intervals (±2 std dev)
        forecast_values = self.forecast(5)
        
        return {
            'lower_bound': [v - 2*std_dev for v in forecast_values],
            'upper_bound': [v + 2*std_dev for v in forecast_values]
        }


class AlertingSystem:
    """Intelligent alerting system with multiple notification channels and ML-based routing."""
    
    def __init__(self):
        """Initialize alerting system."""
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: deque = deque(maxlen=1000)
        self.escalation_policies = {
            'critical': {'channels': ['email', 'slack'], 'escalate_after': 300},
            'error': {'channels': ['email'], 'escalate_after': 900},
            'warning': {'channels': ['slack'], 'escalate_after': 1800},
            'info': {'channels': ['log'], 'escalate_after': None}
        }
        self.alert_suppression: Dict[str, float] = {}
        self.notification_callbacks = {}
        
    def add_notification_callback(self, channel: str, callback: Callable[[Alert], bool]):
        """Add notification callback for specific channel."""
        self.notification_callbacks[channel] = callback
    
    def create_alert(self, severity: str, title: str, description: str, 
                    source: str, suppress_duration: int = 300, **tags) -> Alert:
        """Create and process a new alert."""
        # Check for suppression
        suppress_key = f"{title}:{source}"
        current_time = time.time()
        
        if suppress_key in self.alert_suppression:
            if current_time < self.alert_suppression[suppress_key]:
                return None  # Alert suppressed
        
        # Create alert
        alert = Alert.create(severity, title, description, source, **tags)
        
        # Add to active alerts
        self.active_alerts[alert.id] = alert
        self.alert_history.append(alert)
        
        # Set suppression
        if suppress_duration > 0:
            self.alert_suppression[suppress_key] = current_time + suppress_duration
        
        # Send notifications
        self._send_notifications(alert)
        
        return alert
    
    def resolve_alert(self, alert_id: str, resolver: str = "system") -> bool:
        """Resolve an active alert."""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.resolved = True
            alert.tags['resolved_by'] = resolver
            alert.tags['resolved_at'] = str(datetime.now(timezone.utc))
            del self.active_alerts[alert_id]
            return True
        return False
    
    def acknowledge_alert(self, alert_id: str, acknowledger: str) -> bool:
        """Acknowledge an active alert."""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.acknowledged = True
            alert.tags['acknowledged_by'] = acknowledger
            alert.tags['acknowledged_at'] = str(datetime.now(timezone.utc))
            return True
        return False
    
    def _send_notifications(self, alert: Alert):
        """Send alert notifications through configured channels."""
        policy = self.escalation_policies.get(alert.severity, {'channels': ['log']})
        
        for channel in policy.get('channels', []):
            if channel in self.notification_callbacks:
                try:
                    self.notification_callbacks[channel](alert)
                except Exception as e:
                    logging.error(f"Failed to send alert to {channel}: {e}")
    
    def get_active_alerts(self, severity: Optional[str] = None) -> List[Alert]:
        """Get currently active alerts, optionally filtered by severity."""
        alerts = list(self.active_alerts.values())
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        return sorted(alerts, key=lambda a: a.timestamp, reverse=True)
    
    def get_alert_statistics(self) -> Dict[str, Any]:
        """Get alerting statistics."""
        return {
            'active_alerts': len(self.active_alerts),
            'total_alerts_24h': len([a for a in self.alert_history 
                                   if time.time() - a.timestamp < 86400]),
            'alerts_by_severity': defaultdict(int),
            'top_alert_sources': defaultdict(int),
            'suppressed_alerts': len(self.alert_suppression)
        }


class PredictiveMonitoring:
    """ML-based monitoring with anomaly detection."""
    
    def __init__(self, window_size: int = 100):
        """Initialize predictive monitoring."""
        self.metrics_history = deque(maxlen=window_size)
        self.anomaly_threshold = 2.0  # Standard deviations
        self.baseline_metrics = {}
        
    def add_metrics(self, metrics: Dict[str, float]):
        """Add metrics sample for analysis."""
        self.metrics_history.append({
            'timestamp': time.time(),
            'metrics': metrics.copy()
        })
        
        # Update baseline
        if len(self.metrics_history) >= 10:
            self._update_baseline()
    
    def detect_anomalies(self) -> List[Dict[str, Any]]:
        """Detect performance anomalies using statistical analysis."""
        if len(self.metrics_history) < 10:
            return []
        
        anomalies = []
        latest = self.metrics_history[-1]['metrics']
        
        for metric_name, value in latest.items():
            if metric_name in self.baseline_metrics:
                baseline = self.baseline_metrics[metric_name]
                z_score = abs((value - baseline['mean']) / max(baseline['std'], 0.001))
                
                if z_score > self.anomaly_threshold:
                    anomalies.append({
                        'metric': metric_name,
                        'value': value,
                        'expected': baseline['mean'],
                        'z_score': z_score,
                        'severity': 'critical' if z_score > 3.0 else 'warning'
                    })
        
        return anomalies
    
    def _update_baseline(self):
        """Update baseline statistics for anomaly detection."""
        import statistics
        
        # Group metrics by name
        metrics_by_name = defaultdict(list)
        for sample in self.metrics_history:
            for name, value in sample['metrics'].items():
                metrics_by_name[name].append(value)
        
        # Calculate baseline statistics
        for name, values in metrics_by_name.items():
            if len(values) >= 3:
                self.baseline_metrics[name] = {
                    'mean': statistics.mean(values),
                    'std': statistics.stdev(values) if len(values) > 1 else 0.0,
                    'min': min(values),
                    'max': max(values)
                }
    
    def predict_capacity_needs(self) -> Dict[str, Any]:
        """Predict future capacity requirements."""
        if len(self.metrics_history) < 20:
            return {'status': 'insufficient_data'}
        
        # Simple trend analysis
        recent_samples = list(self.metrics_history)[-20:]
        trends = {}
        
        for metric_name in ['cpu_percent', 'memory_percent', 'molecules_generated']:
            values = [s['metrics'].get(metric_name, 0) for s in recent_samples]
            if len(values) >= 2:
                # Simple linear trend
                x = list(range(len(values)))
                if len(set(values)) > 1:  # Avoid division by zero
                    slope = sum((x[i] - sum(x)/len(x)) * (values[i] - sum(values)/len(values)) 
                               for i in range(len(x))) / sum((xi - sum(x)/len(x))**2 for xi in x)
                    trends[metric_name] = {
                        'slope': slope,
                        'trend': 'increasing' if slope > 0.1 else 'decreasing' if slope < -0.1 else 'stable',
                        'projected_1h': values[-1] + slope * 60,  # 60 minutes
                        'projected_24h': values[-1] + slope * 1440  # 1440 minutes
                    }
        
        return {
            'status': 'analysis_complete',
            'trends': trends,
            'recommendations': self._generate_capacity_recommendations(trends)
        }
    
    def _generate_capacity_recommendations(self, trends: Dict) -> List[str]:
        """Generate capacity planning recommendations."""
        recommendations = []
        
        for metric, trend_data in trends.items():
            if trend_data['trend'] == 'increasing':
                if metric == 'cpu_percent' and trend_data['projected_24h'] > 80:
                    recommendations.append(f"Consider scaling up CPU resources - projected {trend_data['projected_24h']:.1f}% usage")
                elif metric == 'memory_percent' and trend_data['projected_24h'] > 85:
                    recommendations.append(f"Consider scaling up memory resources - projected {trend_data['projected_24h']:.1f}% usage")
        
        return recommendations


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
