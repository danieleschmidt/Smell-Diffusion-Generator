"""
Global Compliance and Regulatory Framework

Comprehensive compliance system supporting:
- Multi-regional regulatory compliance (EU, US, JP, CN, CA, AU, etc.)
- Cultural and religious requirements (Halal, Kosher, Vegan)
- Data protection compliance (GDPR, CCPA, PDPA, LGPD)
- Automated compliance checking and reporting
"""

from .global_compliance import (
    GlobalComplianceChecker,
    ComplianceRegion,
    ComplianceLevel,
    ComplianceResult,
    RegulatoryStandard
)

__all__ = [
    "GlobalComplianceChecker",
    "ComplianceRegion",
    "ComplianceLevel", 
    "ComplianceResult",
    "RegulatoryStandard"
]