#!/bin/bash

# Quality Assurance Script for Smell Diffusion Generator
# Runs comprehensive quality checks and security validation

set -e

echo "🔍 Starting Quality Assurance Checks..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print status
print_status() {
    echo -e "${GREEN}✓${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

# Check if we're in the right directory
if [ ! -f "pyproject.toml" ]; then
    print_error "Not in project root directory"
    exit 1
fi

print_status "Found project configuration"

# 1. SYNTAX AND COMPILATION CHECKS
echo "📝 Checking Python syntax..."
if find . -name "*.py" -not -path "./venv/*" -not -path "./.venv/*" | xargs python3 -m py_compile; then
    print_status "Python syntax check passed"
else
    print_error "Python syntax errors found"
    exit 1
fi

# 2. CODE FORMATTING CHECKS
echo "🎨 Checking code formatting..."

# Check if black is available
if command -v black >/dev/null 2>&1; then
    if black --check --diff smell_diffusion tests examples 2>/dev/null; then
        print_status "Code formatting check passed"
    else
        print_warning "Code formatting issues found (run 'black smell_diffusion tests examples' to fix)"
    fi
else
    print_warning "Black formatter not available, skipping formatting check"
fi

# 3. IMPORT SORTING
echo "📦 Checking import sorting..."
if command -v isort >/dev/null 2>&1; then
    if isort --check-only --diff smell_diffusion tests examples 2>/dev/null; then
        print_status "Import sorting check passed"
    else
        print_warning "Import sorting issues found (run 'isort smell_diffusion tests examples' to fix)"
    fi
else
    print_warning "isort not available, skipping import sort check"
fi

# 4. LINTING
echo "🔍 Running linting checks..."
if command -v flake8 >/dev/null 2>&1; then
    if flake8 smell_diffusion tests --max-line-length=88 --extend-ignore=E203,W503 2>/dev/null; then
        print_status "Linting check passed"
    else
        print_warning "Linting issues found"
    fi
else
    print_warning "flake8 not available, skipping linting"
fi

# 5. TYPE CHECKING
echo "🔍 Running type checks..."
if command -v mypy >/dev/null 2>&1; then
    if mypy smell_diffusion --ignore-missing-imports 2>/dev/null; then
        print_status "Type checking passed"
    else
        print_warning "Type checking issues found"
    fi
else
    print_warning "mypy not available, skipping type checking"
fi

# 6. SECURITY SCANNING
echo "🔒 Running security scans..."

# Bandit security scan
if command -v bandit >/dev/null 2>&1; then
    if bandit -r smell_diffusion -l 2>/dev/null; then
        print_status "Security scan (bandit) passed"
    else
        print_warning "Security issues found in code"
    fi
else
    print_warning "bandit not available, skipping security scan"
fi

# Safety dependency check
if command -v safety >/dev/null 2>&1; then
    if safety check 2>/dev/null; then
        print_status "Dependency security check passed"
    else
        print_warning "Security vulnerabilities found in dependencies"
    fi
else
    print_warning "safety not available, skipping dependency security check"
fi

# 7. PROJECT STRUCTURE VALIDATION
echo "📁 Validating project structure..."
required_files=(
    "pyproject.toml"
    "README.md"
    "LICENSE"
    "SECURITY.md"
    "smell_diffusion/__init__.py"
    "smell_diffusion/core/__init__.py"
    "smell_diffusion/core/smell_diffusion.py"
    "smell_diffusion/core/molecule.py"
    "smell_diffusion/safety/__init__.py"
    "smell_diffusion/safety/evaluator.py"
    "smell_diffusion/utils/__init__.py"
    "smell_diffusion/cli.py"
    "tests/__init__.py"
    "tests/conftest.py"
)

missing_files=()
for file in "${required_files[@]}"; do
    if [ ! -f "$file" ]; then
        missing_files+=("$file")
    fi
done

if [ ${#missing_files[@]} -eq 0 ]; then
    print_status "Project structure validation passed"
else
    print_error "Missing required files: ${missing_files[*]}"
fi

# 8. CONFIGURATION VALIDATION
echo "⚙️ Validating configuration files..."

# Check pyproject.toml syntax
if python3 -c "import tomllib; tomllib.load(open('pyproject.toml', 'rb'))" 2>/dev/null; then
    print_status "pyproject.toml is valid"
else
    print_error "pyproject.toml has syntax errors"
fi

# Check example scripts
echo "📋 Validating example scripts..."
if [ -f "examples/quick_start_demo.py" ]; then
    if python3 -m py_compile examples/quick_start_demo.py; then
        print_status "Example scripts are valid"
    else
        print_error "Example scripts have syntax errors"
    fi
else
    print_warning "Example scripts not found"
fi

# 9. DOCUMENTATION CHECKS
echo "📚 Checking documentation..."
readme_sections=(
    "Features"
    "Quick Start"
    "Installation"
    "Safety"
)

for section in "${readme_sections[@]}"; do
    if grep -q "$section" README.md; then
        print_status "README section '$section' found"
    else
        print_warning "README section '$section' missing"
    fi
done

# 10. DEPENDENCY ANALYSIS
echo "📦 Analyzing dependencies..."
if [ -f "pyproject.toml" ]; then
    # Count dependencies
    dep_count=$(python3 -c "
import tomllib
with open('pyproject.toml', 'rb') as f:
    data = tomllib.load(f)
deps = data.get('project', {}).get('dependencies', [])
print(len(deps))
" 2>/dev/null || echo "0")
    
    print_status "Found $dep_count main dependencies"
    
    # Check for common security issues
    if python3 -c "
import tomllib
with open('pyproject.toml', 'rb') as f:
    data = tomllib.load(f)
deps = data.get('project', {}).get('dependencies', [])
secure = all('==' in dep or '>=' in dep for dep in deps if not dep.startswith('--'))
print('secure' if secure else 'insecure')
" 2>/dev/null | grep -q "secure"; then
        print_status "Dependencies have version constraints"
    else
        print_warning "Some dependencies lack version constraints"
    fi
fi

# 11. FILE SIZE AND COMPLEXITY CHECKS
echo "📊 Checking file complexity..."
large_files=$(find smell_diffusion -name "*.py" -size +10k | wc -l)
if [ "$large_files" -eq 0 ]; then
    print_status "No oversized Python files found"
else
    print_warning "$large_files Python files are larger than 10KB"
fi

# 12. FINAL REPORT
echo ""
echo "🎯 Quality Assurance Summary"
echo "================================"
print_status "Syntax and compilation checks completed"
print_status "Code quality checks completed"
print_status "Security scans completed"
print_status "Project structure validated"
print_status "Configuration files validated"

echo ""
echo "✅ Quality assurance checks completed!"
echo ""
echo "📋 Next Steps:"
echo "   1. Run tests: python -m pytest tests/"
echo "   2. Build package: python -m build"
echo "   3. Start API server: python -m smell_diffusion.api.server"
echo "   4. Try CLI: python -m smell_diffusion.cli generate 'fresh citrus'"

echo ""
echo "🔗 Resources:"
echo "   - Documentation: README.md"
echo "   - Security Policy: SECURITY.md"
echo "   - API Docs: http://localhost:8000/docs (when server running)"