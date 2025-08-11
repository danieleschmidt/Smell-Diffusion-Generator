"""
Advanced Security Module for Smell Diffusion Generator

Provides comprehensive security features for production deployment:
- Input sanitization and validation
- Rate limiting and DDoS protection  
- Authentication and authorization
- Encryption and data protection
- Security audit logging
- Vulnerability scanning
- Compliance monitoring (GDPR, CCPA, etc.)
- Threat detection and response
"""

from .security_scanner import SecurityScanner, VulnerabilityScanner
from .auth_manager import AuthenticationManager, AuthorizationManager
from .encryption import EncryptionManager, DataProtectionManager
from .rate_limiter import RateLimiter, DDOSProtection
from .audit_logger import SecurityAuditLogger
from .compliance_monitor import ComplianceMonitor
from .threat_detector import ThreatDetector, IntrusionDetectionSystem

__all__ = [
    "SecurityScanner",
    "VulnerabilityScanner", 
    "AuthenticationManager",
    "AuthorizationManager",
    "EncryptionManager",
    "DataProtectionManager",
    "RateLimiter",
    "DDOSProtection",
    "SecurityAuditLogger",
    "ComplianceMonitor",
    "ThreatDetector",
    "IntrusionDetectionSystem",
]