# Security Policy

## Supported Versions

We release patches for security vulnerabilities for the following versions:

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

We take the security of Smell Diffusion Generator seriously. If you discover a security vulnerability, please follow these steps:

### 1. Do Not Create a Public Issue

Please **do not** create a public GitHub issue for security vulnerabilities. This could put users at risk.

### 2. Report Privately

Send an email to: `security@terragonlabs.ai`

Please include:
- A description of the vulnerability
- Steps to reproduce the issue
- Potential impact
- Any suggested fixes (if available)

### 3. Response Timeline

- **Initial Response**: Within 48 hours
- **Investigation**: Within 1 week
- **Fix and Disclosure**: Within 2 weeks (depending on complexity)

## Security Measures

### Input Validation

- All user inputs are validated and sanitized
- SMILES strings are validated using RDKit
- Text prompts are checked for malicious content
- API endpoints use Pydantic for input validation

### Safety Constraints

- Prohibited molecular substructures are blocked
- Safety evaluation prevents generation of harmful compounds
- IFRA compliance checking for fragrance applications
- Allergen screening against EU regulations

### Rate Limiting

- API endpoints are rate-limited to prevent abuse
- Circuit breakers protect against cascading failures
- Request size limits prevent DoS attacks

### Data Protection

- No personal data is stored by default
- Generated molecules are not persisted unless explicitly requested
- Logs are sanitized to remove sensitive information

### Dependencies

- All dependencies are regularly updated
- Security scanning with `safety` and `bandit`
- Vulnerability monitoring in CI/CD pipeline

## Security Best Practices for Users

### API Usage

- Use HTTPS for all API communications
- Implement proper authentication in production
- Set appropriate rate limits for your use case
- Validate all responses before processing

### Self-Hosting

- Keep the application updated
- Use a reverse proxy (nginx/Apache) in production
- Configure proper firewall rules
- Monitor for suspicious activity

### Chemical Safety

- Always validate generated molecules experimentally
- Conduct proper safety testing before human exposure
- Follow local regulations for chemical handling
- Consult with qualified chemists for novel compounds

## Disclosure Policy

When a security vulnerability is reported and confirmed:

1. We will work on a fix immediately
2. We will create a security advisory with CVE if applicable
3. We will notify affected users through appropriate channels
4. We will publicly disclose the vulnerability after a fix is available

## Security Updates

Security updates will be released as patch versions (e.g., 0.1.1 → 0.1.2) and will include:

- Clear description of the vulnerability
- Impact assessment
- Migration instructions if needed
- Credits to the reporter (if desired)

## Contact

For security-related questions or concerns:
- Email: security@terragonlabs.ai
- For general questions: issues on GitHub

Thank you for helping keep Smell Diffusion Generator secure!