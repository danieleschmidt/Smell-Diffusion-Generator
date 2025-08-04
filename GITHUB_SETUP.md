# GitHub Setup Instructions

## Moving Workflow Files

Due to GitHub App permissions, the CI/CD workflow file has been placed in the `workflows/` directory instead of `.github/workflows/`. To activate the CI/CD pipeline:

1. **Move the workflow file:**
   ```bash
   mkdir -p .github/workflows
   mv workflows/ci.yml .github/workflows/ci.yml
   rmdir workflows
   ```

2. **Commit and push:**
   ```bash
   git add .github/workflows/ci.yml
   git commit -m "Add CI/CD workflow"
   git push origin main
   ```

## Required GitHub Settings

### Branch Protection Rules
Navigate to Settings → Branches → Add rule:
- Branch name pattern: `main`
- Require status checks to pass before merging
- Require branches to be up to date before merging
- Required status checks:
  - `test (3.9)`
  - `test (3.10)` 
  - `test (3.11)`
  - `security`

### Secrets Configuration
Navigate to Settings → Secrets and variables → Actions:

Add the following secrets if needed:
- `CODECOV_TOKEN` - for code coverage reporting
- `DOCKER_HUB_USERNAME` - for Docker image publishing
- `DOCKER_HUB_ACCESS_TOKEN` - for Docker authentication
- `PYPI_API_TOKEN` - for package publishing

### Repository Settings
Navigate to Settings → General:
- Enable "Automatically delete head branches"
- Set default branch to `main`
- Enable "Allow merge commits"
- Enable "Allow squash merging"
- Disable "Allow rebase merging"

## CI/CD Pipeline Features

The included CI/CD pipeline provides:

### ✅ Multi-Python Testing
- Tests on Python 3.9, 3.10, and 3.11
- Matrix builds for compatibility validation
- Dependency caching for faster builds

### ✅ Code Quality Checks
- **flake8** - linting and style checking
- **black** - code formatting validation
- **isort** - import sorting verification
- **mypy** - static type checking

### ✅ Security Scanning
- **bandit** - security vulnerability detection
- **safety** - dependency vulnerability checking
- Automated security reports

### ✅ Test Coverage
- **pytest** with coverage reporting
- **codecov** integration for coverage tracking
- Minimum coverage thresholds

### ✅ Build Validation
- Package building with **build**
- Package validation with **twine**
- Artifact uploading for releases

### ✅ Docker Integration
- Multi-stage Docker builds
- Image testing and validation
- Production-ready containers

## Activation Checklist

- [ ] Move `workflows/ci.yml` to `.github/workflows/ci.yml`
- [ ] Configure branch protection rules
- [ ] Add required secrets
- [ ] Test the pipeline with a small commit
- [ ] Verify all status checks pass
- [ ] Configure notifications (optional)

## Troubleshooting

### Common Issues

1. **Workflow not triggering**
   - Ensure file is in `.github/workflows/`
   - Check branch protection settings
   - Verify YAML syntax

2. **Tests failing**
   - Check dependency installation
   - Verify Python version compatibility
   - Review test output logs

3. **Security scan failures**
   - Review bandit findings
   - Update vulnerable dependencies
   - Add security exceptions if needed

4. **Build failures**
   - Check package metadata
   - Verify all files are included
   - Review build logs

### Getting Help

- GitHub Actions documentation: https://docs.github.com/en/actions
- Repository discussions for questions
- Security policy for vulnerability reports