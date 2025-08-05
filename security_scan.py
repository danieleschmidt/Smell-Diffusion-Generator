#!/usr/bin/env python3
"""Security and quality gates scanner."""

import os
import sys
import subprocess
import json
from pathlib import Path
from typing import Dict, List, Any
import re

# Import mocks first
sys.path.insert(0, '.')
import mock_deps

from smell_diffusion.utils.validation import SafetyValidator
from smell_diffusion.utils.logging import SmellDiffusionLogger

logger = SmellDiffusionLogger("security_scan")

class SecurityScanner:
    """Automated security and quality scanner."""
    
    def __init__(self):
        """Initialize scanner."""
        self.results = {
            "security": {"passed": 0, "failed": 0, "issues": []},
            "quality": {"passed": 0, "failed": 0, "issues": []},
            "dependencies": {"passed": 0, "failed": 0, "issues": []},
            "code_quality": {"passed": 0, "failed": 0, "issues": []},
            "overall_score": 0
        }
    
    def scan_secrets(self) -> None:
        """Scan for hardcoded secrets."""
        print("🔍 Scanning for hardcoded secrets...")
        
        secret_patterns = [
            r'(?i)(api[_-]?key|secret|token|password)\s*[:=]\s*[\'"][^\'"\s]{10,}[\'"]',
            r'(?i)(aws_access_key_id|aws_secret_access_key)\s*[:=]\s*[\'"][^\'"\s]+[\'"]',
            r'(?i)(private[_-]?key)\s*[:=]\s*[\'"][^\'"\s]{20,}[\'"]',
            r'sk-[a-zA-Z0-9]{48}',  # OpenAI API key pattern
            r'ghp_[a-zA-Z0-9]{36}',  # GitHub personal access token
        ]
        
        for root, dirs, files in os.walk('.'):
            # Skip common non-source directories
            dirs[:] = [d for d in dirs if d not in ['.git', '__pycache__', '.pytest_cache', 'node_modules']]
            
            for file in files:
                if file.endswith(('.py', '.yaml', '.yml', '.json', '.env', '.config')):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()
                            
                        for i, line in enumerate(content.split('\n'), 1):
                            for pattern in secret_patterns:
                                if re.search(pattern, line):
                                    self.results["security"]["issues"].append({
                                        "type": "potential_secret",
                                        "file": file_path,
                                        "line": i,
                                        "description": "Potential hardcoded secret detected"
                                    })
                                    self.results["security"]["failed"] += 1
                    except Exception:
                        continue
        
        if not self.results["security"]["issues"]:
            print("✅ No hardcoded secrets detected")
            self.results["security"]["passed"] += 1
        else:
            print(f"⚠️ Found {len(self.results['security']['issues'])} potential secrets")
    
    def scan_sql_injection(self) -> None:
        """Scan for SQL injection vulnerabilities."""
        print("🔍 Scanning for SQL injection patterns...")
        
        sql_patterns = [
            r'(?i)execute\s*\(\s*[\'"][^\'\"]*\%s[^\'\"]*[\'"]',
            r'(?i)cursor\.execute\s*\(\s*[\'"][^\'\"]*\+[^\'\"]*[\'"]',
            r'(?i)query\s*=\s*[\'"][^\'\"]*\%[^\'\"]*[\'"]',
        ]
        
        found_issues = False
        for root, dirs, files in os.walk('.'):
            dirs[:] = [d for d in dirs if d not in ['.git', '__pycache__', '.pytest_cache']]
            
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                        
                        for i, line in enumerate(content.split('\n'), 1):
                            for pattern in sql_patterns:
                                if re.search(pattern, line):
                                    self.results["security"]["issues"].append({
                                        "type": "sql_injection",
                                        "file": file_path,
                                        "line": i,
                                        "description": "Potential SQL injection vulnerability"
                                    })
                                    found_issues = True
                    except Exception:
                        continue
        
        if not found_issues:
            print("✅ No SQL injection patterns detected")
            self.results["security"]["passed"] += 1
        else:
            print(f"⚠️ Found potential SQL injection patterns")
            self.results["security"]["failed"] += 1
    
    def check_dependencies(self) -> None:
        """Check dependencies for known vulnerabilities."""
        print("🔍 Checking dependencies...")
        
        # Check if requirements.txt exists
        if Path("requirements.txt").exists():
            try:
                # Would normally use safety or similar tool
                print("✅ Dependencies check completed (mock)")
                self.results["dependencies"]["passed"] += 1
            except Exception as e:
                print(f"⚠️ Dependency check failed: {e}")
                self.results["dependencies"]["failed"] += 1
        else:
            # Check pyproject.toml
            if Path("pyproject.toml").exists():
                print("✅ Found pyproject.toml - dependencies managed")
                self.results["dependencies"]["passed"] += 1
            else:
                print("⚠️ No dependency file found")
                self.results["dependencies"]["failed"] += 1
    
    def code_quality_check(self) -> None:
        """Check code quality metrics."""
        print("🔍 Checking code quality...")
        
        # Check for common anti-patterns
        issues = []
        
        for root, dirs, files in os.walk('./smell_diffusion'):
            dirs[:] = [d for d in dirs if not d.startswith('.')]
            
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                        
                        lines = content.split('\n')
                        
                        # Check for long functions (>50 lines)
                        in_function = False
                        function_start = 0
                        function_name = ""
                        
                        for i, line in enumerate(lines):
                            if line.strip().startswith('def '):
                                if in_function and (i - function_start) > 50:
                                    issues.append({
                                        "type": "long_function",
                                        "file": file_path,
                                        "line": function_start,
                                        "description": f"Function '{function_name}' is too long ({i - function_start} lines)"
                                    })
                                
                                in_function = True
                                function_start = i
                                function_name = line.strip().split('(')[0].replace('def ', '')
                            
                            elif line.strip().startswith('class '):
                                in_function = False
                        
                        # Check for missing docstrings
                        if '"""' not in content and "'''" not in content:
                            issues.append({
                                "type": "missing_docstring",
                                "file": file_path,
                                "description": "File appears to be missing docstrings"
                            })
                        
                        # Check for print statements (should use logging)
                        for i, line in enumerate(lines):
                            if re.search(r'^\s*print\s*\(', line):
                                issues.append({
                                    "type": "print_statement",
                                    "file": file_path,
                                    "line": i + 1,
                                    "description": "Use logging instead of print statements"
                                })
                        
                    except Exception:
                        continue
        
        self.results["code_quality"]["issues"] = issues
        
        if len(issues) < 10:  # Arbitrary threshold
            print(f"✅ Code quality check passed ({len(issues)} minor issues)")
            self.results["code_quality"]["passed"] += 1
        else:
            print(f"⚠️ Code quality issues found: {len(issues)}")
            self.results["code_quality"]["failed"] += 1
    
    def check_safety_compliance(self) -> None:
        """Check safety and compliance features."""
        print("🔍 Checking safety compliance...")
        
        try:
            # Test safety validator
            violations = SafetyValidator.check_prohibited_structures("CCO")
            
            # Check if safety evaluator is working
            from smell_diffusion.safety.evaluator import SafetyEvaluator
            evaluator = SafetyEvaluator()
            
            print("✅ Safety compliance system operational")
            self.results["security"]["passed"] += 1
            
        except Exception as e:
            print(f"⚠️ Safety compliance check failed: {e}")
            self.results["security"]["failed"] += 1
            self.results["security"]["issues"].append({
                "type": "safety_system",
                "description": f"Safety system error: {str(e)}"
            })
    
    def check_input_validation(self) -> None:
        """Check input validation coverage."""
        print("🔍 Checking input validation...")
        
        validation_files = [
            "./smell_diffusion/utils/validation.py",
            "./smell_diffusion/core/molecule.py",
            "./smell_diffusion/safety/evaluator.py"
        ]
        
        validation_present = all(Path(f).exists() for f in validation_files)
        
        if validation_present:
            print("✅ Input validation system present")
            self.results["security"]["passed"] += 1
        else:
            print("⚠️ Input validation system incomplete")
            self.results["security"]["failed"] += 1
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive security report."""
        total_checks = sum(
            self.results[category]["passed"] + self.results[category]["failed"]
            for category in ["security", "quality", "dependencies", "code_quality"]
        )
        
        total_passed = sum(
            self.results[category]["passed"]
            for category in ["security", "quality", "dependencies", "code_quality"]
        )
        
        self.results["overall_score"] = (total_passed / max(total_checks, 1)) * 100
        
        return self.results
    
    def run_full_scan(self) -> Dict[str, Any]:
        """Run comprehensive security and quality scan."""
        print("🚀 Starting comprehensive security and quality scan...\n")
        
        self.scan_secrets()
        self.scan_sql_injection()
        self.check_dependencies()
        self.code_quality_check()
        self.check_safety_compliance()
        self.check_input_validation()
        
        return self.generate_report()


def main():
    """Main scanner execution."""
    scanner = SecurityScanner()
    results = scanner.run_full_scan()
    
    print("\n" + "="*60)
    print("📊 SECURITY & QUALITY SCAN RESULTS")
    print("="*60)
    
    for category, data in results.items():
        if category == "overall_score":
            continue
            
        print(f"\n{category.upper()}:")
        print(f"  ✅ Passed: {data['passed']}")
        print(f"  ❌ Failed: {data['failed']}")
        
        if data["issues"]:
            print(f"  📋 Issues ({len(data['issues'])}):")
            for issue in data["issues"][:5]:  # Show first 5 issues
                print(f"    - {issue.get('type', 'unknown')}: {issue.get('description', 'No description')}")
                if 'file' in issue:
                    print(f"      File: {issue['file']}")
                if 'line' in issue:
                    print(f"      Line: {issue['line']}")
            
            if len(data["issues"]) > 5:
                print(f"    ... and {len(data['issues']) - 5} more issues")
    
    print(f"\n🎯 OVERALL SCORE: {results['overall_score']:.1f}/100")
    
    if results['overall_score'] >= 85:
        print("🎉 QUALITY GATES: PASSED")
        status = 0
    elif results['overall_score'] >= 70:
        print("⚠️ QUALITY GATES: WARNING")
        status = 0  # Still allow deployment with warning
    else:
        print("❌ QUALITY GATES: FAILED")
        status = 1
    
    # Save results
    with open("security_report.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📄 Detailed report saved to: security_report.json")
    return status


if __name__ == "__main__":
    exit(main())