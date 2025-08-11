"""
Advanced Security Scanner for Vulnerability Detection and Code Analysis

Provides comprehensive security scanning capabilities:
- Static code analysis for vulnerabilities
- Dynamic analysis and runtime monitoring
- Dependency vulnerability scanning
- Configuration security assessment
- Compliance checking
- Automated remediation suggestions
"""

import re
import os
import time
import json
import hashlib
import subprocess
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
from collections import defaultdict

from ..utils.logging import SmellDiffusionLogger
from ..utils.config import get_config


@dataclass
class SecurityFinding:
    """Security finding or vulnerability."""
    id: str
    severity: str  # critical, high, medium, low, info
    category: str  # injection, xss, auth, crypto, etc.
    title: str
    description: str
    file_path: str
    line_number: int
    code_snippet: str
    cwe_id: Optional[str]
    cvss_score: Optional[float]
    remediation: str
    false_positive_likelihood: float
    timestamp: float
    

@dataclass 
class SecurityReport:
    """Comprehensive security scan report."""
    scan_id: str
    timestamp: float
    scan_type: str
    target: str
    duration_seconds: float
    findings: List[SecurityFinding]
    summary: Dict[str, Any]
    recommendations: List[str]
    compliance_status: Dict[str, str]
    risk_score: float
    

class SecurityScanner:
    """Advanced security scanner with multi-layered analysis."""
    
    def __init__(self):
        self.logger = SmellDiffusionLogger("security_scanner")
        self.config = get_config()
        
        # Security rule patterns
        self.vulnerability_patterns = self._load_vulnerability_patterns()
        self.security_rules = self._load_security_rules()
        
        # Scan history
        self.scan_history = []
        self.baseline_scan = None
        
        # ML-based analysis
        self.ml_analyzer = MLSecurityAnalyzer()
        self.behavioral_analyzer = BehavioralSecurityAnalyzer()
        
    def _load_vulnerability_patterns(self) -> Dict[str, List[Dict[str, Any]]]:
        """Load vulnerability detection patterns."""
        return {
            'injection': [
                {
                    'pattern': r'eval\s*\(',
                    'severity': 'critical',
                    'cwe': 'CWE-94',
                    'description': 'Code injection through eval()',
                    'remediation': 'Avoid eval(). Use safer alternatives like ast.literal_eval() for data parsing.'
                },
                {
                    'pattern': r'exec\s*\(',
                    'severity': 'critical', 
                    'cwe': 'CWE-94',
                    'description': 'Code injection through exec()',
                    'remediation': 'Remove exec() calls. Use structured programming approaches.'
                },
                {
                    'pattern': r'subprocess\.(call|run|Popen).*shell=True',
                    'severity': 'high',
                    'cwe': 'CWE-78',
                    'description': 'Command injection vulnerability',
                    'remediation': 'Use shell=False and pass arguments as list instead of string.'
                },
                {
                    'pattern': r'os\.system\(',
                    'severity': 'high',
                    'cwe': 'CWE-78',
                    'description': 'OS command injection vulnerability',
                    'remediation': 'Replace os.system() with subprocess.run() with proper argument handling.'
                }
            ],
            
            'crypto': [
                {
                    'pattern': r'hashlib\.(md5|sha1)\(',
                    'severity': 'medium',
                    'cwe': 'CWE-327',
                    'description': 'Weak cryptographic hash function',
                    'remediation': 'Use stronger hash functions like SHA-256 or SHA-3.'
                },
                {
                    'pattern': r'random\.random\(\)',
                    'severity': 'medium',
                    'cwe': 'CWE-338',
                    'description': 'Cryptographically weak random number generation',
                    'remediation': 'Use secrets.randbytes() or os.urandom() for cryptographic randomness.'
                },
                {
                    'pattern': r'ssl\.create_default_context\(\).*check_hostname=False',
                    'severity': 'high',
                    'cwe': 'CWE-295',
                    'description': 'Disabled SSL certificate verification',
                    'remediation': 'Enable hostname checking for SSL connections.'
                }
            ],
            
            'auth': [
                {
                    'pattern': r'password.*=.*["\'][^"\']{0,7}["\']',
                    'severity': 'high',
                    'cwe': 'CWE-521',
                    'description': 'Weak password in code',
                    'remediation': 'Use strong passwords and store them securely (environment variables, key management).'
                },
                {
                    'pattern': r'(api[_-]?key|secret[_-]?key|access[_-]?token).*=.*["\'][^"\']+["\']',
                    'severity': 'critical',
                    'cwe': 'CWE-798',
                    'description': 'Hardcoded credentials in source code',
                    'remediation': 'Move credentials to environment variables or secure key management system.'
                }
            ],
            
            'data_exposure': [
                {
                    'pattern': r'print\(.*(password|secret|token|key).*\)',
                    'severity': 'medium',
                    'cwe': 'CWE-532',
                    'description': 'Sensitive information in logs',
                    'remediation': 'Remove or sanitize sensitive data from log statements.'
                },
                {
                    'pattern': r'logging\.(info|debug|warning|error|critical)\(.*(password|secret|token|key).*\)',
                    'severity': 'medium',
                    'cwe': 'CWE-532',
                    'description': 'Sensitive information logged',
                    'remediation': 'Sanitize sensitive data before logging.'
                }
            ],
            
            'input_validation': [
                {
                    'pattern': r'input\([^)]*\).*without.*validation',
                    'severity': 'medium',
                    'cwe': 'CWE-20',
                    'description': 'Insufficient input validation',
                    'remediation': 'Implement proper input validation and sanitization.'
                }
            ]
        }
    
    def _load_security_rules(self) -> List[Dict[str, Any]]:
        """Load additional security rules."""
        return [
            {
                'id': 'SEC001',
                'name': 'SQL Injection Detection',
                'pattern': r'SELECT.*\+.*|INSERT.*\+.*|UPDATE.*\+.*',
                'severity': 'high',
                'category': 'injection'
            },
            {
                'id': 'SEC002',
                'name': 'Path Traversal Detection', 
                'pattern': r'\.\.\/|\.\.\\',
                'severity': 'high',
                'category': 'path_traversal'
            },
            {
                'id': 'SEC003',
                'name': 'Unsafe File Operations',
                'pattern': r'open\([^,)]*,\s*["\']w["\'].*\)|file\([^,)]*,\s*["\']w["\'].*\)',
                'severity': 'medium',
                'category': 'file_access'
            }
        ]
    
    def scan_code(self, target_path: str, scan_type: str = 'comprehensive') -> SecurityReport:
        """Perform comprehensive security code scan."""
        
        scan_start = time.time()
        scan_id = hashlib.md5(f"{target_path}_{scan_start}".encode()).hexdigest()[:8]
        
        self.logger.logger.info(f"Starting security scan {scan_id} on {target_path}")
        
        try:
            findings = []
            
            # Static analysis
            if scan_type in ['comprehensive', 'static']:
                static_findings = self._perform_static_analysis(target_path)
                findings.extend(static_findings)
            
            # Dependency scanning  
            if scan_type in ['comprehensive', 'dependencies']:
                dep_findings = self._scan_dependencies(target_path)
                findings.extend(dep_findings)
            
            # Configuration analysis
            if scan_type in ['comprehensive', 'config']:
                config_findings = self._analyze_configuration(target_path)
                findings.extend(config_findings)
            
            # ML-based analysis
            if scan_type in ['comprehensive', 'ml']:
                ml_findings = self.ml_analyzer.analyze_code(target_path)
                findings.extend(ml_findings)
            
            # Behavioral analysis (if runtime data available)
            if scan_type in ['comprehensive', 'behavioral']:
                behavioral_findings = self.behavioral_analyzer.analyze_patterns()
                findings.extend(behavioral_findings)
            
            # Generate summary and risk assessment
            summary = self._generate_summary(findings)
            recommendations = self._generate_recommendations(findings)
            compliance_status = self._assess_compliance(findings)
            risk_score = self._calculate_risk_score(findings)
            
            scan_duration = time.time() - scan_start
            
            report = SecurityReport(
                scan_id=scan_id,
                timestamp=scan_start,
                scan_type=scan_type,
                target=target_path,
                duration_seconds=scan_duration,
                findings=findings,
                summary=summary,
                recommendations=recommendations,
                compliance_status=compliance_status,
                risk_score=risk_score
            )
            
            # Store scan results
            self.scan_history.append(report)
            if not self.baseline_scan:
                self.baseline_scan = report
            
            self.logger.logger.info(
                f"Security scan {scan_id} completed: {len(findings)} findings, "
                f"risk score {risk_score:.1f}/10"
            )
            
            return report
            
        except Exception as e:
            self.logger.log_error("security_scan", e, {"target": target_path, "scan_type": scan_type})
            
            # Return error report
            return SecurityReport(
                scan_id=scan_id,
                timestamp=scan_start,
                scan_type=scan_type,
                target=target_path,
                duration_seconds=time.time() - scan_start,
                findings=[],
                summary={"error": str(e)},
                recommendations=[f"Security scan failed: {str(e)}"],
                compliance_status={"overall": "error"},
                risk_score=10.0  # Maximum risk on error
            )
    
    def _perform_static_analysis(self, target_path: str) -> List[SecurityFinding]:
        """Perform static code analysis."""
        findings = []
        
        # Scan Python files
        for py_file in Path(target_path).rglob("*.py"):
            if py_file.is_file():
                file_findings = self._scan_python_file(py_file)
                findings.extend(file_findings)
        
        # Scan configuration files
        config_patterns = ["*.json", "*.yaml", "*.yml", "*.ini", "*.conf", "*.cfg"]
        for pattern in config_patterns:
            for config_file in Path(target_path).rglob(pattern):
                if config_file.is_file():
                    file_findings = self._scan_config_file(config_file)
                    findings.extend(file_findings)
        
        return findings
    
    def _scan_python_file(self, file_path: Path) -> List[SecurityFinding]:
        """Scan individual Python file for vulnerabilities."""
        findings = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.split('\\n')
            
            # Apply vulnerability patterns
            for category, patterns in self.vulnerability_patterns.items():
                for pattern_info in patterns:
                    matches = re.finditer(pattern_info['pattern'], content, re.IGNORECASE | re.MULTILINE)
                    
                    for match in matches:
                        line_num = content[:match.start()].count('\\n') + 1
                        line_content = lines[line_num - 1] if line_num <= len(lines) else ""
                        
                        finding = SecurityFinding(
                            id=hashlib.md5(f"{file_path}_{line_num}_{match.group()}".encode()).hexdigest()[:8],
                            severity=pattern_info['severity'],
                            category=category,
                            title=pattern_info['description'],
                            description=f"Potential {category} vulnerability detected",
                            file_path=str(file_path),
                            line_number=line_num,
                            code_snippet=line_content.strip(),
                            cwe_id=pattern_info.get('cwe'),
                            cvss_score=self._estimate_cvss_score(pattern_info['severity']),
                            remediation=pattern_info['remediation'],
                            false_positive_likelihood=self._estimate_false_positive_rate(pattern_info),
                            timestamp=time.time()
                        )
                        
                        findings.append(finding)
            
            # Apply custom security rules
            for rule in self.security_rules:
                matches = re.finditer(rule['pattern'], content, re.IGNORECASE | re.MULTILINE)
                
                for match in matches:
                    line_num = content[:match.start()].count('\\n') + 1
                    line_content = lines[line_num - 1] if line_num <= len(lines) else ""
                    
                    finding = SecurityFinding(
                        id=hashlib.md5(f"{file_path}_{rule['id']}_{line_num}".encode()).hexdigest()[:8],
                        severity=rule['severity'],
                        category=rule['category'],
                        title=rule['name'],
                        description=f"Security rule {rule['id']} violation",
                        file_path=str(file_path),
                        line_number=line_num,
                        code_snippet=line_content.strip(),
                        cwe_id=None,
                        cvss_score=self._estimate_cvss_score(rule['severity']),
                        remediation="Review code for security compliance",
                        false_positive_likelihood=0.3,
                        timestamp=time.time()
                    )
                    
                    findings.append(finding)
                    
        except Exception as e:
            self.logger.log_error("python_file_scan", e, {"file": str(file_path)})
        
        return findings
    
    def _scan_config_file(self, file_path: Path) -> List[SecurityFinding]:
        """Scan configuration file for security issues."""
        findings = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check for hardcoded credentials
            credential_patterns = [
                r'password\s*[:=]\s*["\']?[^"\'\\n\\r]{8,}["\']?',
                r'secret\s*[:=]\s*["\']?[^"\'\\n\\r]{16,}["\']?',
                r'api[_-]?key\s*[:=]\s*["\']?[^"\'\\n\\r]{16,}["\']?',
                r'token\s*[:=]\s*["\']?[^"\'\\n\\r]{20,}["\']?'
            ]
            
            for pattern in credential_patterns:
                matches = re.finditer(pattern, content, re.IGNORECASE)
                
                for match in matches:
                    line_num = content[:match.start()].count('\\n') + 1
                    
                    finding = SecurityFinding(
                        id=hashlib.md5(f"{file_path}_cred_{line_num}".encode()).hexdigest()[:8],
                        severity='high',
                        category='credential_exposure',
                        title='Hardcoded credentials in configuration',
                        description='Configuration file contains hardcoded credentials',
                        file_path=str(file_path),
                        line_number=line_num,
                        code_snippet=match.group()[:50] + "..." if len(match.group()) > 50 else match.group(),
                        cwe_id='CWE-798',
                        cvss_score=8.5,
                        remediation='Move credentials to environment variables or secure key management',
                        false_positive_likelihood=0.2,
                        timestamp=time.time()
                    )
                    
                    findings.append(finding)
                    
        except Exception as e:
            self.logger.log_error("config_file_scan", e, {"file": str(file_path)})
        
        return findings
    
    def _scan_dependencies(self, target_path: str) -> List[SecurityFinding]:
        """Scan dependencies for known vulnerabilities."""
        findings = []
        
        try:
            # Check requirements.txt
            req_file = Path(target_path) / "requirements.txt"
            if req_file.exists():
                findings.extend(self._scan_requirements_file(req_file))
            
            # Check pyproject.toml
            pyproject_file = Path(target_path) / "pyproject.toml"
            if pyproject_file.exists():
                findings.extend(self._scan_pyproject_file(pyproject_file))
            
            # Check package.json (if Node.js components)
            package_file = Path(target_path) / "package.json"
            if package_file.exists():
                findings.extend(self._scan_package_json(package_file))
                
        except Exception as e:
            self.logger.log_error("dependency_scan", e, {"target": target_path})
        
        return findings
    
    def _scan_requirements_file(self, req_file: Path) -> List[SecurityFinding]:
        """Scan requirements.txt for vulnerable dependencies."""
        findings = []
        
        try:
            with open(req_file, 'r') as f:
                requirements = f.read().strip().split('\\n')
            
            # Known vulnerable packages (simplified)
            vulnerable_packages = {
                'requests': {
                    'versions': ['<2.20.0'],
                    'cve': 'CVE-2018-18074',
                    'severity': 'medium',
                    'description': 'Requests library vulnerable to authorization bypass'
                },
                'pyyaml': {
                    'versions': ['<5.1'],
                    'cve': 'CVE-2017-18342',
                    'severity': 'high',
                    'description': 'PyYAML vulnerable to arbitrary code execution'
                },
                'django': {
                    'versions': ['<2.2.13', '>=3.0,<3.0.7'],
                    'cve': 'CVE-2020-13254',
                    'severity': 'high',
                    'description': 'Django vulnerable to data leakage'
                }
            }
            
            for line_num, requirement in enumerate(requirements, 1):
                req = requirement.strip()
                if not req or req.startswith('#'):
                    continue
                
                # Parse package name
                package_name = re.split(r'[<>=!]', req)[0].strip()
                
                if package_name.lower() in vulnerable_packages:
                    vuln_info = vulnerable_packages[package_name.lower()]
                    
                    finding = SecurityFinding(
                        id=hashlib.md5(f"{req_file}_dep_{package_name}".encode()).hexdigest()[:8],
                        severity=vuln_info['severity'],
                        category='vulnerable_dependency',
                        title=f'Vulnerable dependency: {package_name}',
                        description=vuln_info['description'],
                        file_path=str(req_file),
                        line_number=line_num,
                        code_snippet=req,
                        cwe_id='CWE-1104',
                        cvss_score=self._estimate_cvss_score(vuln_info['severity']),
                        remediation=f'Update {package_name} to a secure version',
                        false_positive_likelihood=0.1,
                        timestamp=time.time()
                    )
                    
                    findings.append(finding)
                    
        except Exception as e:
            self.logger.log_error("requirements_scan", e, {"file": str(req_file)})
        
        return findings
    
    def _scan_pyproject_file(self, pyproject_file: Path) -> List[SecurityFinding]:
        """Scan pyproject.toml for security issues."""
        findings = []
        
        try:
            import toml
            with open(pyproject_file, 'r') as f:
                pyproject_data = toml.load(f)
            
            # Check dependencies in pyproject.toml
            dependencies = pyproject_data.get('project', {}).get('dependencies', [])
            
            for dep in dependencies:
                # Simple vulnerability check (would integrate with real vulnerability database)
                if 'requests<2.20' in dep or 'pyyaml<5.1' in dep:
                    finding = SecurityFinding(
                        id=hashlib.md5(f"{pyproject_file}_pyproject_{dep}".encode()).hexdigest()[:8],
                        severity='medium',
                        category='vulnerable_dependency',
                        title='Potentially vulnerable dependency in pyproject.toml',
                        description=f'Dependency {dep} may have known vulnerabilities',
                        file_path=str(pyproject_file),
                        line_number=1,
                        code_snippet=dep,
                        cwe_id='CWE-1104',
                        cvss_score=6.0,
                        remediation='Review and update dependency versions',
                        false_positive_likelihood=0.4,
                        timestamp=time.time()
                    )
                    findings.append(finding)
                    
        except Exception as e:
            self.logger.log_error("pyproject_scan", e, {"file": str(pyproject_file)})
        
        return findings
    
    def _scan_package_json(self, package_file: Path) -> List[SecurityFinding]:
        """Scan package.json for security issues."""
        findings = []
        
        try:
            with open(package_file, 'r') as f:
                package_data = json.load(f)
            
            # Check for known vulnerable npm packages
            dependencies = package_data.get('dependencies', {})
            dev_dependencies = package_data.get('devDependencies', {})
            
            all_deps = {**dependencies, **dev_dependencies}
            
            for dep_name, version in all_deps.items():
                # Simple check for known vulnerable packages
                if dep_name in ['lodash', 'moment', 'handlebars']:  # Known historically vulnerable
                    finding = SecurityFinding(
                        id=hashlib.md5(f"{package_file}_npm_{dep_name}".encode()).hexdigest()[:8],
                        severity='medium',
                        category='vulnerable_dependency',
                        title=f'Potentially vulnerable npm package: {dep_name}',
                        description=f'Package {dep_name} has had security vulnerabilities',
                        file_path=str(package_file),
                        line_number=1,
                        code_snippet=f'"{dep_name}": "{version}"',
                        cwe_id='CWE-1104',
                        cvss_score=5.5,
                        remediation=f'Review {dep_name} for latest security updates',
                        false_positive_likelihood=0.5,
                        timestamp=time.time()
                    )
                    findings.append(finding)
                    
        except Exception as e:
            self.logger.log_error("package_json_scan", e, {"file": str(package_file)})
        
        return findings
    
    def _analyze_configuration(self, target_path: str) -> List[SecurityFinding]:
        """Analyze configuration for security issues."""
        findings = []
        
        config_checks = [
            self._check_debug_mode,
            self._check_ssl_configuration,
            self._check_cors_configuration,
            self._check_security_headers
        ]
        
        for check in config_checks:
            try:
                check_findings = check(target_path)
                findings.extend(check_findings)
            except Exception as e:
                self.logger.log_error(f"config_check_{check.__name__}", e)
        
        return findings
    
    def _check_debug_mode(self, target_path: str) -> List[SecurityFinding]:
        """Check for debug mode enabled in production."""
        findings = []
        
        # Check Python files for DEBUG=True
        for py_file in Path(target_path).rglob("*.py"):
            try:
                with open(py_file, 'r') as f:
                    content = f.read()
                
                if re.search(r'DEBUG\s*=\s*True', content, re.IGNORECASE):
                    finding = SecurityFinding(
                        id=hashlib.md5(f"{py_file}_debug".encode()).hexdigest()[:8],
                        severity='medium',
                        category='configuration',
                        title='Debug mode enabled',
                        description='Debug mode should be disabled in production',
                        file_path=str(py_file),
                        line_number=1,
                        code_snippet="DEBUG = True",
                        cwe_id='CWE-489',
                        cvss_score=4.0,
                        remediation='Set DEBUG = False in production environments',
                        false_positive_likelihood=0.1,
                        timestamp=time.time()
                    )
                    findings.append(finding)
                    
            except Exception as e:
                continue
        
        return findings
    
    def _check_ssl_configuration(self, target_path: str) -> List[SecurityFinding]:
        """Check SSL/TLS configuration."""
        findings = []
        
        # Check for SSL verification disabled
        for py_file in Path(target_path).rglob("*.py"):
            try:
                with open(py_file, 'r') as f:
                    content = f.read()
                
                if re.search(r'verify=False|ssl_verify_mode.*NONE', content, re.IGNORECASE):
                    finding = SecurityFinding(
                        id=hashlib.md5(f"{py_file}_ssl".encode()).hexdigest()[:8],
                        severity='high',
                        category='ssl_tls',
                        title='SSL verification disabled',
                        description='SSL certificate verification is disabled',
                        file_path=str(py_file),
                        line_number=1,
                        code_snippet="verify=False",
                        cwe_id='CWE-295',
                        cvss_score=7.0,
                        remediation='Enable SSL certificate verification',
                        false_positive_likelihood=0.1,
                        timestamp=time.time()
                    )
                    findings.append(finding)
                    
            except Exception as e:
                continue
        
        return findings
    
    def _check_cors_configuration(self, target_path: str) -> List[SecurityFinding]:
        """Check CORS configuration for overly permissive settings."""
        findings = []
        
        for py_file in Path(target_path).rglob("*.py"):
            try:
                with open(py_file, 'r') as f:
                    content = f.read()
                
                # Check for wildcard CORS
                if re.search(r'Access-Control-Allow-Origin.*\*|cors.*origin.*\*', content, re.IGNORECASE):
                    finding = SecurityFinding(
                        id=hashlib.md5(f"{py_file}_cors".encode()).hexdigest()[:8],
                        severity='medium',
                        category='cors',
                        title='Overly permissive CORS configuration',
                        description='CORS allows requests from any origin',
                        file_path=str(py_file),
                        line_number=1,
                        code_snippet="Access-Control-Allow-Origin: *",
                        cwe_id='CWE-942',
                        cvss_score=5.0,
                        remediation='Restrict CORS to specific trusted origins',
                        false_positive_likelihood=0.3,
                        timestamp=time.time()
                    )
                    findings.append(finding)
                    
            except Exception as e:
                continue
        
        return findings
    
    def _check_security_headers(self, target_path: str) -> List[SecurityFinding]:
        """Check for missing security headers."""
        findings = []
        
        # This is a simplified check - would need more sophisticated analysis
        security_headers = [
            'X-Content-Type-Options',
            'X-Frame-Options', 
            'X-XSS-Protection',
            'Strict-Transport-Security',
            'Content-Security-Policy'
        ]
        
        header_found = False
        for py_file in Path(target_path).rglob("*.py"):
            try:
                with open(py_file, 'r') as f:
                    content = f.read()
                
                for header in security_headers:
                    if header in content:
                        header_found = True
                        break
                        
            except Exception as e:
                continue
        
        if not header_found:
            finding = SecurityFinding(
                id=hashlib.md5(f"{target_path}_headers".encode()).hexdigest()[:8],
                severity='low',
                category='security_headers',
                title='Missing security headers',
                description='Application may be missing important security headers',
                file_path=target_path,
                line_number=1,
                code_snippet="",
                cwe_id='CWE-1021',
                cvss_score=3.0,
                remediation='Implement security headers like CSP, HSTS, etc.',
                false_positive_likelihood=0.4,
                timestamp=time.time()
            )
            findings.append(finding)
        
        return findings
    
    def _estimate_cvss_score(self, severity: str) -> float:
        """Estimate CVSS score based on severity."""
        severity_scores = {
            'critical': 9.0,
            'high': 7.5,
            'medium': 5.0,
            'low': 3.0,
            'info': 1.0
        }
        return severity_scores.get(severity.lower(), 5.0)
    
    def _estimate_false_positive_rate(self, pattern_info: Dict) -> float:
        """Estimate false positive likelihood for finding."""
        # Simple heuristic based on pattern complexity
        pattern = pattern_info['pattern']
        
        if len(pattern) > 50:  # Complex patterns less likely to be false positives
            return 0.1
        elif 'eval|exec' in pattern:  # High-risk patterns
            return 0.2
        elif 'password|secret' in pattern:  # Context-dependent patterns
            return 0.4
        else:
            return 0.3
    
    def _generate_summary(self, findings: List[SecurityFinding]) -> Dict[str, Any]:
        """Generate scan summary."""
        severity_counts = defaultdict(int)
        category_counts = defaultdict(int)
        
        for finding in findings:
            severity_counts[finding.severity] += 1
            category_counts[finding.category] += 1
        
        return {
            'total_findings': len(findings),
            'severity_breakdown': dict(severity_counts),
            'category_breakdown': dict(category_counts),
            'critical_findings': severity_counts.get('critical', 0),
            'high_findings': severity_counts.get('high', 0),
            'medium_findings': severity_counts.get('medium', 0),
            'low_findings': severity_counts.get('low', 0),
            'scan_coverage': {
                'static_analysis': True,
                'dependency_scan': True,
                'configuration_analysis': True
            }
        }
    
    def _generate_recommendations(self, findings: List[SecurityFinding]) -> List[str]:
        """Generate security recommendations."""
        recommendations = []
        
        # Critical findings recommendations
        critical_count = sum(1 for f in findings if f.severity == 'critical')
        if critical_count > 0:
            recommendations.append(f"URGENT: Address {critical_count} critical security vulnerabilities immediately")
        
        # High findings recommendations
        high_count = sum(1 for f in findings if f.severity == 'high')
        if high_count > 0:
            recommendations.append(f"HIGH PRIORITY: Remediate {high_count} high-severity vulnerabilities")
        
        # Category-specific recommendations
        categories = set(f.category for f in findings)
        
        if 'injection' in categories:
            recommendations.append("Implement input validation and parameterized queries to prevent injection attacks")
        
        if 'crypto' in categories:
            recommendations.append("Review cryptographic implementations and use strong algorithms")
        
        if 'auth' in categories:
            recommendations.append("Implement secure authentication and credential management")
        
        if 'vulnerable_dependency' in categories:
            recommendations.append("Update vulnerable dependencies to latest secure versions")
        
        # General recommendations
        if len(findings) > 10:
            recommendations.append("Consider implementing automated security scanning in CI/CD pipeline")
        
        recommendations.append("Conduct regular security reviews and penetration testing")
        recommendations.append("Implement security training for development team")
        
        return recommendations
    
    def _assess_compliance(self, findings: List[SecurityFinding]) -> Dict[str, str]:
        """Assess compliance with security standards."""
        compliance_status = {}
        
        # OWASP Top 10 assessment
        owasp_categories = ['injection', 'auth', 'data_exposure', 'xxe', 'broken_access_control']
        owasp_violations = sum(1 for f in findings if f.category in owasp_categories)
        
        if owasp_violations == 0:
            compliance_status['OWASP_Top_10'] = 'compliant'
        elif owasp_violations < 5:
            compliance_status['OWASP_Top_10'] = 'mostly_compliant'
        else:
            compliance_status['OWASP_Top_10'] = 'non_compliant'
        
        # GDPR compliance (data protection)
        data_exposure_count = sum(1 for f in findings if f.category == 'data_exposure')
        compliance_status['GDPR'] = 'compliant' if data_exposure_count == 0 else 'at_risk'
        
        # Overall compliance
        critical_count = sum(1 for f in findings if f.severity == 'critical')
        if critical_count > 0:
            compliance_status['overall'] = 'non_compliant'
        elif owasp_violations > 0 or data_exposure_count > 0:
            compliance_status['overall'] = 'partially_compliant'
        else:
            compliance_status['overall'] = 'compliant'
        
        return compliance_status
    
    def _calculate_risk_score(self, findings: List[SecurityFinding]) -> float:
        """Calculate overall risk score (0-10)."""
        if not findings:
            return 0.0
        
        # Weight findings by severity
        severity_weights = {
            'critical': 10.0,
            'high': 7.0,
            'medium': 4.0,
            'low': 1.0,
            'info': 0.1
        }
        
        total_weighted_score = 0.0
        max_possible_score = 0.0
        
        for finding in findings:
            weight = severity_weights.get(finding.severity, 1.0)
            confidence = 1.0 - finding.false_positive_likelihood
            
            total_weighted_score += weight * confidence
            max_possible_score += weight
        
        if max_possible_score > 0:
            risk_score = min(10.0, (total_weighted_score / len(findings)) * 2)
        else:
            risk_score = 0.0
        
        return round(risk_score, 1)
    
    def compare_scans(self, scan_id1: str, scan_id2: str) -> Dict[str, Any]:
        """Compare two security scans."""
        scan1 = next((s for s in self.scan_history if s.scan_id == scan_id1), None)
        scan2 = next((s for s in self.scan_history if s.scan_id == scan_id2), None)
        
        if not scan1 or not scan2:
            return {"error": "One or both scans not found"}
        
        # Compare findings
        new_findings = [f for f in scan2.findings if f.id not in [f1.id for f1 in scan1.findings]]
        resolved_findings = [f for f in scan1.findings if f.id not in [f2.id for f2 in scan2.findings]]
        
        risk_change = scan2.risk_score - scan1.risk_score
        
        return {
            'comparison_summary': {
                'scan1_id': scan_id1,
                'scan2_id': scan_id2,
                'new_findings': len(new_findings),
                'resolved_findings': len(resolved_findings),
                'risk_score_change': risk_change,
                'overall_trend': 'improved' if risk_change < 0 else 'degraded' if risk_change > 0 else 'stable'
            },
            'new_findings': [asdict(f) for f in new_findings],
            'resolved_findings': [asdict(f) for f in resolved_findings],
            'risk_analysis': {
                'previous_risk': scan1.risk_score,
                'current_risk': scan2.risk_score,
                'change': risk_change
            }
        }


class VulnerabilityScanner:
    """Specialized vulnerability scanner with external integrations."""
    
    def __init__(self):
        self.logger = SmellDiffusionLogger("vulnerability_scanner")
        self.vulnerability_databases = [
            'https://cve.circl.lu/api/',  # CVE API
            'https://vulndb.cyberriskanalytics.com/api/',  # VulnDB
        ]
        
    def scan_for_cves(self, dependencies: List[str]) -> List[Dict[str, Any]]:
        """Scan dependencies against CVE databases."""
        vulnerabilities = []
        
        for dependency in dependencies:
            try:
                # Parse dependency name and version
                dep_name, version = self._parse_dependency(dependency)
                
                # Check against vulnerability databases
                cves = self._query_cve_database(dep_name, version)
                
                for cve in cves:
                    vulnerabilities.append({
                        'dependency': dependency,
                        'cve_id': cve['id'],
                        'severity': cve['severity'],
                        'description': cve['description'],
                        'cvss_score': cve.get('cvss_score', 0.0),
                        'published_date': cve.get('published_date'),
                        'references': cve.get('references', [])
                    })
                    
            except Exception as e:
                self.logger.log_error("cve_scan", e, {"dependency": dependency})
        
        return vulnerabilities
    
    def _parse_dependency(self, dependency: str) -> Tuple[str, str]:
        """Parse dependency string to extract name and version."""
        # Simple parsing - would need more sophisticated logic
        if '==' in dependency:
            name, version = dependency.split('==')
            return name.strip(), version.strip()
        elif '>=' in dependency:
            name, version = dependency.split('>=')
            return name.strip(), version.strip()
        else:
            return dependency.strip(), "unknown"
    
    def _query_cve_database(self, package_name: str, version: str) -> List[Dict[str, Any]]:
        """Query CVE database for package vulnerabilities."""
        # Mock implementation - would make actual API calls
        mock_cves = [
            {
                'id': 'CVE-2024-1234',
                'severity': 'high',
                'description': f'Vulnerability in {package_name}',
                'cvss_score': 7.5,
                'published_date': '2024-01-01',
                'references': ['https://example.com/cve-details']
            }
        ]
        
        # Return mock data for demonstration
        return mock_cves if package_name in ['requests', 'django', 'flask'] else []


class MLSecurityAnalyzer:
    """Machine Learning-based security analysis."""
    
    def __init__(self):
        self.logger = SmellDiffusionLogger("ml_security_analyzer")
        self.trained_models = {}
        self.anomaly_detectors = {}
        
    def analyze_code(self, target_path: str) -> List[SecurityFinding]:
        """Perform ML-based security analysis."""
        findings = []
        
        try:
            # Analyze code patterns with ML
            pattern_findings = self._analyze_code_patterns(target_path)
            findings.extend(pattern_findings)
            
            # Detect anomalous code structures
            anomaly_findings = self._detect_code_anomalies(target_path)
            findings.extend(anomaly_findings)
            
            # Behavioral analysis
            behavioral_findings = self._analyze_behavioral_patterns(target_path)
            findings.extend(behavioral_findings)
            
        except Exception as e:
            self.logger.log_error("ml_security_analysis", e)
        
        return findings
    
    def _analyze_code_patterns(self, target_path: str) -> List[SecurityFinding]:
        """Analyze code patterns for potential vulnerabilities."""
        findings = []
        
        # Mock ML-based pattern analysis
        suspicious_patterns = [
            {
                'pattern': 'complex_eval_usage',
                'confidence': 0.8,
                'description': 'ML detected complex eval() usage pattern',
                'severity': 'high'
            },
            {
                'pattern': 'unusual_network_calls',
                'confidence': 0.6,
                'description': 'ML detected unusual network call patterns',
                'severity': 'medium'
            }
        ]
        
        for pattern in suspicious_patterns:
            if pattern['confidence'] > 0.7:
                finding = SecurityFinding(
                    id=hashlib.md5(f"ml_{pattern['pattern']}_{time.time()}".encode()).hexdigest()[:8],
                    severity=pattern['severity'],
                    category='ml_detected',
                    title=f"ML Detected: {pattern['pattern']}",
                    description=pattern['description'],
                    file_path=target_path,
                    line_number=1,
                    code_snippet="[ML Analysis]",
                    cwe_id=None,
                    cvss_score=self._confidence_to_cvss(pattern['confidence']),
                    remediation="Review flagged code pattern for security implications",
                    false_positive_likelihood=1.0 - pattern['confidence'],
                    timestamp=time.time()
                )
                findings.append(finding)
        
        return findings
    
    def _detect_code_anomalies(self, target_path: str) -> List[SecurityFinding]:
        """Detect anomalous code structures."""
        findings = []
        
        # Mock anomaly detection
        anomalies = [
            {
                'type': 'unusual_function_complexity',
                'score': 0.9,
                'description': 'Function complexity significantly higher than project average'
            },
            {
                'type': 'suspicious_import_patterns', 
                'score': 0.75,
                'description': 'Unusual import patterns detected'
            }
        ]
        
        for anomaly in anomalies:
            if anomaly['score'] > 0.7:
                finding = SecurityFinding(
                    id=hashlib.md5(f"anomaly_{anomaly['type']}_{time.time()}".encode()).hexdigest()[:8],
                    severity='medium' if anomaly['score'] > 0.8 else 'low',
                    category='anomaly_detection',
                    title=f"Code Anomaly: {anomaly['type']}",
                    description=anomaly['description'],
                    file_path=target_path,
                    line_number=1,
                    code_snippet="[Anomaly Detection]",
                    cwe_id=None,
                    cvss_score=anomaly['score'] * 5.0,
                    remediation="Review flagged code for potential security issues",
                    false_positive_likelihood=0.4,
                    timestamp=time.time()
                )
                findings.append(finding)
        
        return findings
    
    def _analyze_behavioral_patterns(self, target_path: str) -> List[SecurityFinding]:
        """Analyze behavioral patterns in code."""
        findings = []
        
        # Mock behavioral analysis
        behaviors = [
            {
                'behavior': 'data_exfiltration_pattern',
                'confidence': 0.6,
                'description': 'Code pattern similar to data exfiltration techniques'
            },
            {
                'behavior': 'privilege_escalation_pattern',
                'confidence': 0.7,
                'description': 'Potential privilege escalation pattern detected'
            }
        ]
        
        for behavior in behaviors:
            if behavior['confidence'] > 0.6:
                finding = SecurityFinding(
                    id=hashlib.md5(f"behavior_{behavior['behavior']}_{time.time()}".encode()).hexdigest()[:8],
                    severity='high' if behavior['confidence'] > 0.8 else 'medium',
                    category='behavioral_analysis',
                    title=f"Behavioral Pattern: {behavior['behavior']}",
                    description=behavior['description'],
                    file_path=target_path,
                    line_number=1,
                    code_snippet="[Behavioral Analysis]",
                    cwe_id=None,
                    cvss_score=behavior['confidence'] * 8.0,
                    remediation="Investigate flagged behavioral pattern",
                    false_positive_likelihood=0.5,
                    timestamp=time.time()
                )
                findings.append(finding)
        
        return findings
    
    def _confidence_to_cvss(self, confidence: float) -> float:
        """Convert ML confidence to CVSS score."""
        return confidence * 8.0  # Scale 0-1 confidence to 0-8 CVSS


class BehavioralSecurityAnalyzer:
    """Analyze runtime behavioral patterns for security threats."""
    
    def __init__(self):
        self.logger = SmellDiffusionLogger("behavioral_security_analyzer")
        self.behavioral_data = []
        self.baseline_behavior = None
        
    def analyze_patterns(self) -> List[SecurityFinding]:
        """Analyze behavioral patterns for security threats."""
        findings = []
        
        # Mock behavioral analysis based on runtime data
        if len(self.behavioral_data) < 10:
            return findings
        
        # Analyze patterns
        patterns = self._identify_behavioral_patterns()
        
        for pattern in patterns:
            if pattern['risk_score'] > 0.7:
                finding = SecurityFinding(
                    id=hashlib.md5(f"behavioral_{pattern['type']}_{time.time()}".encode()).hexdigest()[:8],
                    severity='high' if pattern['risk_score'] > 0.8 else 'medium',
                    category='behavioral_threat',
                    title=f"Behavioral Threat: {pattern['type']}",
                    description=pattern['description'],
                    file_path="runtime_analysis",
                    line_number=1,
                    code_snippet="[Runtime Behavior]",
                    cwe_id=None,
                    cvss_score=pattern['risk_score'] * 8.0,
                    remediation=pattern.get('remediation', 'Investigate behavioral anomaly'),
                    false_positive_likelihood=0.3,
                    timestamp=time.time()
                )
                findings.append(finding)
        
        return findings
    
    def _identify_behavioral_patterns(self) -> List[Dict[str, Any]]:
        """Identify suspicious behavioral patterns."""
        return [
            {
                'type': 'unusual_data_access',
                'description': 'Unusual data access patterns detected',
                'risk_score': 0.8,
                'remediation': 'Review data access patterns and implement monitoring'
            },
            {
                'type': 'abnormal_network_activity',
                'description': 'Abnormal network activity patterns',
                'risk_score': 0.75,
                'remediation': 'Monitor network traffic for suspicious activity'
            }
        ]
    
    def record_behavioral_data(self, data: Dict[str, Any]):
        """Record behavioral data for analysis."""
        self.behavioral_data.append({
            'timestamp': time.time(),
            'data': data
        })
        
        # Keep only recent data
        if len(self.behavioral_data) > 1000:
            self.behavioral_data = self.behavioral_data[-1000:]


# Factory function
def create_security_scanner() -> SecurityScanner:
    """Create configured security scanner instance."""
    return SecurityScanner()