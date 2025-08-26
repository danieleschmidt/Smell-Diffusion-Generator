#!/usr/bin/env python3
"""
Autonomous Security Hardening System
Self-defending security framework with adaptive threat detection and response
"""

import asyncio
import time
import hashlib
import hmac
import secrets
import json
import logging
import re
import os
from typing import Dict, List, Any, Optional, Callable, Set, Union
from dataclasses import dataclass, field
from enum import Enum
import traceback
from collections import defaultdict, deque
import threading

# Mock imports for environments without dependencies
try:
    import numpy as np
except ImportError:
    class MockNumPy:
        @staticmethod
        def mean(x): return sum(x) / len(x) if x else 0
        @staticmethod
        def std(x):
            mean_val = sum(x) / len(x) if x else 0
            variance = sum((i - mean_val) ** 2 for i in x) / len(x) if x else 0
            return variance ** 0.5
    np = MockNumPy()


class ThreatLevel(Enum):
    """Threat level classification"""
    MINIMAL = "minimal"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class SecurityEvent(Enum):
    """Types of security events"""
    INJECTION_ATTEMPT = "injection_attempt"
    XSS_ATTEMPT = "xss_attempt"
    AUTH_FAILURE = "auth_failure"
    RATE_LIMIT_BREACH = "rate_limit_breach"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"
    DATA_EXFILTRATION = "data_exfiltration"
    MALFORMED_INPUT = "malformed_input"


@dataclass
class SecurityIncident:
    """Security incident record"""
    event_type: SecurityEvent
    threat_level: ThreatLevel
    timestamp: float
    source_ip: str = "unknown"
    user_agent: str = "unknown"
    request_data: Dict[str, Any] = field(default_factory=dict)
    response_action: str = "none"
    blocked: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SecurityPolicy:
    """Security policy configuration"""
    name: str
    enabled: bool = True
    threat_level: ThreatLevel = ThreatLevel.MEDIUM
    action: str = "log"  # log, block, alert, quarantine
    patterns: List[str] = field(default_factory=list)
    whitelist: List[str] = field(default_factory=list)
    rate_limit: Optional[int] = None
    cooldown: float = 60.0


class AdaptiveThreatDetection:
    """
    Machine learning-based threat detection that adapts to new patterns
    """
    
    def __init__(self):
        self.threat_patterns: Dict[str, List[str]] = {}
        self.behavioral_baselines: Dict[str, Dict[str, float]] = {}
        self.anomaly_thresholds: Dict[str, float] = {}
        self.learning_enabled = True
        self.confidence_threshold = 0.7
        
        # Initialize threat patterns
        self._initialize_threat_patterns()
    
    def _initialize_threat_patterns(self):
        """Initialize known threat patterns"""
        self.threat_patterns = {
            "sql_injection": [
                r"(?i)(union|select|insert|update|delete|drop|create|alter)\s",
                r"(?i)(\'\s*(or|and)\s*\d+\s*=\s*\d+)",
                r"(?i)(\'\s*;\s*drop\s+table)",
                r"(?i)(exec\s*\(|execute\s*\()",
                r"(?i)(script\s*>|javascript\s*:)"
            ],
            "xss": [
                r"(?i)<script[^>]*>.*?</script>",
                r"(?i)javascript\s*:",
                r"(?i)on\w+\s*=",
                r"(?i)<iframe[^>]*>",
                r"(?i)eval\s*\("
            ],
            "path_traversal": [
                r"\.\.\/",
                r"\.\.\\",
                r"\/etc\/passwd",
                r"\/windows\/system32"
            ],
            "command_injection": [
                r"(?i)(;|&&|\|\|)\s*(rm|del|cat|type|wget|curl)",
                r"(?i)`[^`]*`",
                r"(?i)\$\([^)]*\)"
            ]
        }
    
    def analyze_threat(self, input_data: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Analyze input for potential threats"""
        if context is None:
            context = {}
        
        threats_detected = []
        confidence_scores = []
        
        for threat_type, patterns in self.threat_patterns.items():
            for pattern in patterns:
                if re.search(pattern, input_data):
                    threats_detected.append(threat_type)
                    # Calculate confidence based on pattern complexity and context
                    confidence = self._calculate_threat_confidence(
                        threat_type, pattern, input_data, context
                    )
                    confidence_scores.append(confidence)
        
        # Behavioral analysis
        behavioral_anomaly = self._detect_behavioral_anomaly(input_data, context)
        
        # Overall threat assessment
        max_confidence = max(confidence_scores) if confidence_scores else 0.0
        overall_threat_level = self._calculate_threat_level(
            threats_detected, max_confidence, behavioral_anomaly
        )
        
        return {
            "threats_detected": list(set(threats_detected)),
            "threat_level": overall_threat_level,
            "confidence": max_confidence,
            "behavioral_anomaly": behavioral_anomaly,
            "recommended_action": self._recommend_action(overall_threat_level, max_confidence)
        }
    
    def _calculate_threat_confidence(self, threat_type: str, pattern: str, input_data: str, context: Dict[str, Any]) -> float:
        """Calculate confidence score for threat detection"""
        base_confidence = 0.6
        
        # Pattern complexity factor
        complexity_factor = min(len(pattern) / 50.0, 0.3)
        
        # Context factors
        context_factor = 0.0
        if context.get("user_agent", "").lower() in ["curl", "wget", "python-requests"]:
            context_factor += 0.2
        
        if context.get("referer") is None:
            context_factor += 0.1
        
        # Input length factor (very long inputs can be suspicious)
        length_factor = min(len(input_data) / 1000.0, 0.1)
        
        total_confidence = min(
            base_confidence + complexity_factor + context_factor + length_factor,
            1.0
        )
        
        return total_confidence
    
    def _detect_behavioral_anomaly(self, input_data: str, context: Dict[str, Any]) -> float:
        """Detect behavioral anomalies using statistical analysis"""
        
        # Input characteristics
        input_length = len(input_data)
        special_char_ratio = sum(1 for c in input_data if not c.isalnum()) / max(len(input_data), 1)
        entropy = self._calculate_entropy(input_data)
        
        # Compare with baseline (simplified)
        baseline_length = 100  # Average expected input length
        baseline_special_ratio = 0.1
        baseline_entropy = 3.5
        
        # Calculate anomaly scores
        length_anomaly = abs(input_length - baseline_length) / baseline_length
        ratio_anomaly = abs(special_char_ratio - baseline_special_ratio) / baseline_special_ratio
        entropy_anomaly = abs(entropy - baseline_entropy) / baseline_entropy
        
        # Weighted average
        overall_anomaly = (
            length_anomaly * 0.3 +
            ratio_anomaly * 0.4 +
            entropy_anomaly * 0.3
        )
        
        return min(overall_anomaly, 1.0)
    
    def _calculate_entropy(self, data: str) -> float:
        """Calculate Shannon entropy of the input data"""
        if not data:
            return 0.0
        
        # Count character frequencies
        char_counts = defaultdict(int)
        for char in data:
            char_counts[char] += 1
        
        # Calculate entropy
        entropy = 0.0
        data_len = len(data)
        
        for count in char_counts.values():
            probability = count / data_len
            if probability > 0:
                entropy -= probability * (probability ** 0.5)  # Simplified log approximation
        
        return entropy
    
    def _calculate_threat_level(self, threats: List[str], confidence: float, behavioral_anomaly: float) -> ThreatLevel:
        """Calculate overall threat level"""
        
        # Base threat level from detected threats
        if not threats:
            base_level = 0
        elif any(threat in ["sql_injection", "command_injection"] for threat in threats):
            base_level = 4
        elif any(threat in ["xss", "path_traversal"] for threat in threats):
            base_level = 3
        else:
            base_level = 2
        
        # Adjust based on confidence and behavioral anomaly
        confidence_adjustment = confidence * 2
        anomaly_adjustment = behavioral_anomaly * 1.5
        
        total_score = base_level + confidence_adjustment + anomaly_adjustment
        
        # Map to threat levels
        if total_score >= 5.0:
            return ThreatLevel.CRITICAL
        elif total_score >= 4.0:
            return ThreatLevel.HIGH
        elif total_score >= 2.5:
            return ThreatLevel.MEDIUM
        elif total_score >= 1.0:
            return ThreatLevel.LOW
        else:
            return ThreatLevel.MINIMAL
    
    def _recommend_action(self, threat_level: ThreatLevel, confidence: float) -> str:
        """Recommend action based on threat assessment"""
        
        if threat_level == ThreatLevel.CRITICAL:
            return "block_and_alert"
        elif threat_level == ThreatLevel.HIGH:
            return "block" if confidence > 0.8 else "log_and_monitor"
        elif threat_level == ThreatLevel.MEDIUM:
            return "log_and_monitor"
        elif threat_level == ThreatLevel.LOW:
            return "log"
        else:
            return "allow"
    
    def learn_from_incident(self, input_data: str, confirmed_threat: bool, threat_type: str = None):
        """Learn from security incidents to improve detection"""
        if not self.learning_enabled:
            return
        
        if confirmed_threat and threat_type:
            # Extract new patterns from confirmed threats
            new_patterns = self._extract_patterns(input_data)
            
            if threat_type not in self.threat_patterns:
                self.threat_patterns[threat_type] = []
            
            # Add unique patterns
            for pattern in new_patterns:
                if pattern not in self.threat_patterns[threat_type]:
                    self.threat_patterns[threat_type].append(pattern)
        
        # Update behavioral baselines
        self._update_behavioral_baseline(input_data, confirmed_threat)
    
    def _extract_patterns(self, input_data: str) -> List[str]:
        """Extract potential threat patterns from input data"""
        patterns = []
        
        # Simple pattern extraction (can be enhanced with ML)
        words = re.findall(r'\w+', input_data.lower())
        special_sequences = re.findall(r'[^\w\s]{2,}', input_data)
        
        # Create patterns from suspicious sequences
        for seq in special_sequences:
            if len(seq) >= 3:
                patterns.append(re.escape(seq))
        
        return patterns
    
    def _update_behavioral_baseline(self, input_data: str, is_threat: bool):
        """Update behavioral baseline statistics"""
        
        characteristics = {
            "length": len(input_data),
            "special_ratio": sum(1 for c in input_data if not c.isalnum()) / max(len(input_data), 1),
            "entropy": self._calculate_entropy(input_data)
        }
        
        # Update baselines using exponential moving average
        alpha = 0.1  # Learning rate
        
        for char_name, value in characteristics.items():
            if char_name not in self.behavioral_baselines:
                self.behavioral_baselines[char_name] = {"mean": value, "variance": 0.0}
            else:
                baseline = self.behavioral_baselines[char_name]
                old_mean = baseline["mean"]
                
                # Update mean
                baseline["mean"] = old_mean * (1 - alpha) + value * alpha
                
                # Update variance
                baseline["variance"] = baseline["variance"] * (1 - alpha) + ((value - old_mean) ** 2) * alpha


class AutonomousSecurityHardening:
    """
    Comprehensive autonomous security hardening system
    """
    
    def __init__(self):
        self.logger = self._setup_logging()
        self.threat_detector = AdaptiveThreatDetection()
        self.security_incidents: List[SecurityIncident] = []
        self.security_policies: Dict[str, SecurityPolicy] = {}
        self.rate_limiters: Dict[str, Dict[str, Any]] = {}
        self.blocked_ips: Set[str] = set()
        self.session_tokens: Dict[str, Dict[str, Any]] = {}
        
        # Security metrics
        self.metrics = {
            "total_requests": 0,
            "blocked_requests": 0,
            "threats_detected": 0,
            "false_positives": 0,
            "response_time": deque(maxlen=1000)
        }
        
        # Initialize default policies
        self._initialize_security_policies()
        
        # Start background tasks
        self._start_security_monitoring()
    
    def _setup_logging(self) -> logging.Logger:
        """Setup security logging"""
        logger = logging.getLogger("AutonomousSecurityHardening")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _initialize_security_policies(self):
        """Initialize default security policies"""
        
        self.security_policies = {
            "input_validation": SecurityPolicy(
                name="input_validation",
                enabled=True,
                threat_level=ThreatLevel.HIGH,
                action="block",
                rate_limit=100,
                cooldown=60.0
            ),
            "rate_limiting": SecurityPolicy(
                name="rate_limiting",
                enabled=True,
                threat_level=ThreatLevel.MEDIUM,
                action="throttle",
                rate_limit=50,
                cooldown=60.0
            ),
            "authentication": SecurityPolicy(
                name="authentication",
                enabled=True,
                threat_level=ThreatLevel.CRITICAL,
                action="block",
                cooldown=300.0
            ),
            "data_sanitization": SecurityPolicy(
                name="data_sanitization",
                enabled=True,
                threat_level=ThreatLevel.MEDIUM,
                action="sanitize"
            )
        }
    
    def _start_security_monitoring(self):
        """Start background security monitoring tasks"""
        # Note: In a real implementation, these would be proper background tasks
        pass
    
    async def secure_request_handler(self, request_data: Dict[str, Any], context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Main security handler for incoming requests
        """
        start_time = time.time()
        self.metrics["total_requests"] += 1
        
        if context is None:
            context = {}
        
        # Extract request information
        user_ip = context.get("remote_addr", "unknown")
        user_agent = context.get("user_agent", "unknown")
        request_method = context.get("method", "GET")
        
        # Rate limiting check
        if await self._check_rate_limit(user_ip):
            self.metrics["blocked_requests"] += 1
            incident = SecurityIncident(
                event_type=SecurityEvent.RATE_LIMIT_BREACH,
                threat_level=ThreatLevel.MEDIUM,
                timestamp=time.time(),
                source_ip=user_ip,
                user_agent=user_agent,
                blocked=True,
                response_action="rate_limited"
            )
            self.security_incidents.append(incident)
            
            return {
                "status": "blocked",
                "reason": "rate_limit_exceeded",
                "retry_after": 60
            }
        
        # IP blacklist check
        if user_ip in self.blocked_ips:
            self.metrics["blocked_requests"] += 1
            return {
                "status": "blocked",
                "reason": "ip_blocked"
            }
        
        # Threat analysis
        request_string = json.dumps(request_data, default=str)
        threat_analysis = self.threat_detector.analyze_threat(request_string, context)
        
        if threat_analysis["threat_level"] != ThreatLevel.MINIMAL:
            self.metrics["threats_detected"] += 1
            
            # Create security incident
            incident = SecurityIncident(
                event_type=self._map_threat_to_event(threat_analysis["threats_detected"]),
                threat_level=threat_analysis["threat_level"],
                timestamp=time.time(),
                source_ip=user_ip,
                user_agent=user_agent,
                request_data=request_data,
                response_action=threat_analysis["recommended_action"],
                blocked=threat_analysis["recommended_action"] in ["block", "block_and_alert"]
            )
            self.security_incidents.append(incident)
            
            # Execute security response
            security_response = await self._execute_security_response(
                threat_analysis, incident, context
            )
            
            if security_response["blocked"]:
                self.metrics["blocked_requests"] += 1
                return security_response
        
        # Input sanitization
        sanitized_data = await self._sanitize_input_data(request_data)
        
        # Record response time
        response_time = time.time() - start_time
        self.metrics["response_time"].append(response_time)
        
        return {
            "status": "allowed",
            "sanitized_data": sanitized_data,
            "security_score": self._calculate_security_score(threat_analysis),
            "response_time": response_time
        }
    
    async def _check_rate_limit(self, user_ip: str) -> bool:
        """Check if user has exceeded rate limits"""
        current_time = time.time()
        
        if user_ip not in self.rate_limiters:
            self.rate_limiters[user_ip] = {
                "requests": [],
                "blocked_until": 0
            }
        
        rate_limiter = self.rate_limiters[user_ip]
        
        # Check if still blocked
        if current_time < rate_limiter["blocked_until"]:
            return True
        
        # Clean old requests (older than 1 minute)
        rate_limiter["requests"] = [
            req_time for req_time in rate_limiter["requests"]
            if current_time - req_time < 60
        ]
        
        # Check rate limit
        rate_limit = self.security_policies["rate_limiting"].rate_limit
        if len(rate_limiter["requests"]) >= rate_limit:
            rate_limiter["blocked_until"] = current_time + 60  # Block for 1 minute
            return True
        
        # Add current request
        rate_limiter["requests"].append(current_time)
        return False
    
    def _map_threat_to_event(self, threats: List[str]) -> SecurityEvent:
        """Map detected threats to security events"""
        if "sql_injection" in threats:
            return SecurityEvent.INJECTION_ATTEMPT
        elif "xss" in threats:
            return SecurityEvent.XSS_ATTEMPT
        elif any(threat in threats for threat in ["command_injection", "path_traversal"]):
            return SecurityEvent.INJECTION_ATTEMPT
        else:
            return SecurityEvent.SUSPICIOUS_ACTIVITY
    
    async def _execute_security_response(self, threat_analysis: Dict[str, Any], incident: SecurityIncident, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute appropriate security response based on threat level"""
        
        action = threat_analysis["recommended_action"]
        user_ip = incident.source_ip
        
        if action == "block_and_alert":
            # Block IP and send alert
            self.blocked_ips.add(user_ip)
            await self._send_security_alert(incident)
            
            self.logger.warning(
                f"CRITICAL THREAT BLOCKED: {incident.event_type.value} from {user_ip}"
            )
            
            return {
                "status": "blocked",
                "reason": "critical_threat_detected",
                "blocked": True,
                "threat_level": incident.threat_level.value
            }
        
        elif action == "block":
            # Temporary block
            self.rate_limiters[user_ip] = {
                "requests": [],
                "blocked_until": time.time() + 300  # 5 minutes
            }
            
            self.logger.warning(
                f"THREAT BLOCKED: {incident.event_type.value} from {user_ip}"
            )
            
            return {
                "status": "blocked",
                "reason": "threat_detected",
                "blocked": True,
                "retry_after": 300
            }
        
        elif action == "log_and_monitor":
            # Enhanced monitoring for this IP
            self.logger.info(
                f"THREAT MONITORED: {incident.event_type.value} from {user_ip}"
            )
            
            return {
                "status": "monitored",
                "reason": "suspicious_activity",
                "blocked": False,
                "enhanced_logging": True
            }
        
        else:
            # Just log
            self.logger.info(
                f"SECURITY EVENT: {incident.event_type.value} from {user_ip}"
            )
            
            return {
                "status": "logged",
                "blocked": False
            }
    
    async def _send_security_alert(self, incident: SecurityIncident):
        """Send security alert for critical incidents"""
        alert_data = {
            "timestamp": incident.timestamp,
            "event_type": incident.event_type.value,
            "threat_level": incident.threat_level.value,
            "source_ip": incident.source_ip,
            "user_agent": incident.user_agent,
            "action_taken": incident.response_action
        }
        
        # In a real implementation, this would send to SIEM, email, Slack, etc.
        self.logger.critical(f"SECURITY ALERT: {json.dumps(alert_data, indent=2)}")
    
    async def _sanitize_input_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize input data to prevent attacks"""
        sanitized = {}
        
        for key, value in data.items():
            if isinstance(value, str):
                # HTML escape
                sanitized_value = (value
                                   .replace("&", "&amp;")
                                   .replace("<", "&lt;")
                                   .replace(">", "&gt;")
                                   .replace('"', "&quot;")
                                   .replace("'", "&#x27;"))
                
                # Remove potential script content
                sanitized_value = re.sub(r'(?i)<script[^>]*>.*?</script>', '', sanitized_value)
                sanitized_value = re.sub(r'(?i)javascript\s*:', '', sanitized_value)
                
                sanitized[key] = sanitized_value
            elif isinstance(value, dict):
                sanitized[key] = await self._sanitize_input_data(value)
            else:
                sanitized[key] = value
        
        return sanitized
    
    def _calculate_security_score(self, threat_analysis: Dict[str, Any]) -> float:
        """Calculate overall security score for the request"""
        base_score = 1.0
        
        # Reduce score based on threat level
        threat_level = threat_analysis["threat_level"]
        if threat_level == ThreatLevel.CRITICAL:
            base_score -= 0.8
        elif threat_level == ThreatLevel.HIGH:
            base_score -= 0.6
        elif threat_level == ThreatLevel.MEDIUM:
            base_score -= 0.3
        elif threat_level == ThreatLevel.LOW:
            base_score -= 0.1
        
        # Factor in confidence
        confidence = threat_analysis.get("confidence", 0.0)
        base_score -= confidence * 0.2
        
        return max(0.0, min(1.0, base_score))
    
    def generate_security_token(self, user_id: str, expiry_minutes: int = 60) -> str:
        """Generate secure session token"""
        token_data = {
            "user_id": user_id,
            "issued_at": time.time(),
            "expires_at": time.time() + (expiry_minutes * 60),
            "nonce": secrets.token_hex(16)
        }
        
        # Create token with HMAC
        secret_key = os.environ.get("SECRET_KEY", "default_secret_key_change_in_production")
        token_string = json.dumps(token_data, sort_keys=True)
        signature = hmac.new(
            secret_key.encode(),
            token_string.encode(),
            hashlib.sha256
        ).hexdigest()
        
        token = {
            "data": token_data,
            "signature": signature
        }
        
        token_id = hashlib.sha256(json.dumps(token, sort_keys=True).encode()).hexdigest()[:16]
        self.session_tokens[token_id] = token
        
        return token_id
    
    def validate_security_token(self, token_id: str) -> Dict[str, Any]:
        """Validate security token"""
        if token_id not in self.session_tokens:
            return {"valid": False, "reason": "token_not_found"}
        
        token = self.session_tokens[token_id]
        current_time = time.time()
        
        # Check expiry
        if current_time > token["data"]["expires_at"]:
            del self.session_tokens[token_id]
            return {"valid": False, "reason": "token_expired"}
        
        # Validate signature
        secret_key = os.environ.get("SECRET_KEY", "default_secret_key_change_in_production")
        token_string = json.dumps(token["data"], sort_keys=True)
        expected_signature = hmac.new(
            secret_key.encode(),
            token_string.encode(),
            hashlib.sha256
        ).hexdigest()
        
        if not hmac.compare_digest(token["signature"], expected_signature):
            return {"valid": False, "reason": "invalid_signature"}
        
        return {
            "valid": True,
            "user_id": token["data"]["user_id"],
            "issued_at": token["data"]["issued_at"],
            "expires_at": token["data"]["expires_at"]
        }
    
    def get_security_analytics(self) -> Dict[str, Any]:
        """Get comprehensive security analytics"""
        
        # Incident analysis
        incident_types = defaultdict(int)
        threat_levels = defaultdict(int)
        recent_incidents = []
        
        for incident in self.security_incidents:
            incident_types[incident.event_type.value] += 1
            threat_levels[incident.threat_level.value] += 1
            
            if time.time() - incident.timestamp < 3600:  # Last hour
                recent_incidents.append({
                    "event_type": incident.event_type.value,
                    "threat_level": incident.threat_level.value,
                    "source_ip": incident.source_ip,
                    "timestamp": incident.timestamp,
                    "blocked": incident.blocked
                })
        
        # Calculate security metrics
        total_requests = self.metrics["total_requests"]
        blocked_requests = self.metrics["blocked_requests"]
        block_rate = blocked_requests / max(total_requests, 1)
        
        avg_response_time = (
            np.mean(list(self.metrics["response_time"]))
            if self.metrics["response_time"] else 0.0
        )
        
        return {
            "summary": {
                "total_requests": total_requests,
                "blocked_requests": blocked_requests,
                "threats_detected": self.metrics["threats_detected"],
                "block_rate": block_rate,
                "avg_response_time": avg_response_time
            },
            "incidents": {
                "by_type": dict(incident_types),
                "by_threat_level": dict(threat_levels),
                "recent_incidents": recent_incidents[-10:]  # Last 10
            },
            "protection_status": {
                "active_policies": len([p for p in self.security_policies.values() if p.enabled]),
                "blocked_ips": len(self.blocked_ips),
                "active_sessions": len(self.session_tokens),
                "rate_limited_ips": len(self.rate_limiters)
            }
        }
    
    def get_security_health(self) -> Dict[str, Any]:
        """Get security system health status"""
        
        recent_threats = [
            incident for incident in self.security_incidents
            if time.time() - incident.timestamp < 3600  # Last hour
        ]
        
        critical_threats = [
            incident for incident in recent_threats
            if incident.threat_level == ThreatLevel.CRITICAL
        ]
        
        # Health scoring
        threat_score = min(len(recent_threats) / 100.0, 1.0)  # Normalized by expected volume
        critical_score = len(critical_threats) * 0.2
        
        health_score = max(0.0, 1.0 - threat_score - critical_score)
        
        # Overall status
        if critical_threats:
            status = "critical"
        elif health_score < 0.7:
            status = "degraded"
        elif health_score < 0.9:
            status = "warning"
        else:
            status = "healthy"
        
        return {
            "status": status,
            "health_score": health_score,
            "recent_threat_count": len(recent_threats),
            "critical_threat_count": len(critical_threats),
            "protection_active": True,
            "learning_enabled": self.threat_detector.learning_enabled
        }


# Global security hardening instance
global_security_hardening = AutonomousSecurityHardening()


# Convenience decorators
def with_security_hardening(component: str = "default"):
    """Decorator for automatic security hardening"""
    def decorator(func: Callable):
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            # Extract request data from arguments
            request_data = {}
            if args:
                request_data = args[0] if isinstance(args[0], dict) else {"args": args}
            
            # Security check
            security_result = await global_security_hardening.secure_request_handler(
                request_data, {"component": component}
            )
            
            if security_result.get("blocked"):
                raise Exception(f"Security block: {security_result.get('reason')}")
            
            # Execute original function with sanitized data
            if security_result.get("sanitized_data"):
                return await func(security_result["sanitized_data"], *args[1:], **kwargs)
            else:
                return await func(*args, **kwargs)
        
        @functools.wraps(func)
        def sync_wrapper(*args, **kwargs):
            return asyncio.run(async_wrapper(*args, **kwargs))
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    
    return decorator


async def main():
    """Demo of autonomous security hardening system"""
    
    print("🔒 Autonomous Security Hardening System Demo")
    print("=" * 50)
    
    # Test various security scenarios
    test_cases = [
        {
            "name": "Normal Request",
            "data": {"user": "alice", "action": "read_data"},
            "context": {"remote_addr": "192.168.1.100", "user_agent": "Mozilla/5.0"}
        },
        {
            "name": "SQL Injection Attempt",
            "data": {"query": "SELECT * FROM users WHERE id = '1' OR '1'='1'"},
            "context": {"remote_addr": "192.168.1.200", "user_agent": "curl/7.68.0"}
        },
        {
            "name": "XSS Attempt",
            "data": {"comment": "<script>alert('XSS')</script>"},
            "context": {"remote_addr": "192.168.1.300", "user_agent": "Python-requests"}
        },
        {
            "name": "Path Traversal",
            "data": {"file": "../../etc/passwd"},
            "context": {"remote_addr": "192.168.1.400", "user_agent": "wget/1.20.3"}
        }
    ]
    
    print("Testing security responses...")
    
    for test_case in test_cases:
        print(f"\n📋 Testing: {test_case['name']}")
        
        result = await global_security_hardening.secure_request_handler(
            test_case["data"],
            test_case["context"]
        )
        
        status = result["status"]
        if status == "blocked":
            print(f"🚫 BLOCKED: {result.get('reason')}")
        elif status == "monitored":
            print(f"👁️ MONITORED: Enhanced logging enabled")
        else:
            print(f"✅ ALLOWED: Security score {result.get('security_score', 'N/A')}")
    
    # Display analytics
    analytics = global_security_hardening.get_security_analytics()
    health = global_security_hardening.get_security_health()
    
    print("\n📊 Security Analytics:")
    print(f"Total Requests: {analytics['summary']['total_requests']}")
    print(f"Blocked Requests: {analytics['summary']['blocked_requests']}")
    print(f"Block Rate: {analytics['summary']['block_rate']:.1%}")
    
    print("\n🏥 Security Health:")
    print(f"Status: {health['status'].upper()}")
    print(f"Health Score: {health['health_score']:.2f}")
    
    return analytics


if __name__ == "__main__":
    asyncio.run(main())