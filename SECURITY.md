# Security Policy

## Supported Versions

The following versions of Wedding Photo Curator are currently supported with security updates:

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | Supported          |
| < 1.0   | Not supported      |

## Reporting Security Vulnerabilities

We take security seriously. If you discover a security vulnerability, please do not open a public issue. Instead:

1. Email your report to [security contact] with details about the vulnerability
2. Include steps to reproduce and potential impact assessment
3. Allow reasonable time for us to respond and develop a fix
4. Do not disclose the vulnerability publicly until we've had time to address it

We will:
- Acknowledge receipt of your report within 48 hours
- Provide regular updates on our progress
- Credit you appropriately when the fix is released (unless you prefer to remain anonymous)

## Security Best Practices

When using Wedding Photo Curator:

### File Access
- Always verify photo folder permissions
- Do not run with elevated privileges unless necessary
- Ensure BEST_PRINTS folder has appropriate permissions

### Data Storage
- The application processes photos locally - no data is sent externally
- Caching files (photo_analysis_cache.json) are stored locally
- CSV results contain file paths - handle appropriately in shared environments

### Dependencies
- Keep all dependencies updated: `pip install --upgrade -r requirements.txt`
- Review new dependency versions before updating
- Use virtual environments to isolate dependencies

### System Security
- Run in isolated environment (virtual environment)
- Close application when not in use
- Do not store sensitive credentials in configuration
- Monitor disk space to prevent issues during processing

## Dependency Security

We monitor security vulnerabilities in our dependencies using:
- GitHub's dependency scanning
- Regular pip package updates
- Community security advisories

If a vulnerability is found in a dependency:
1. We will attempt to update to a patched version
2. If no patch is available, we'll look for alternatives
3. We'll release a security update promptly
4. Users will be notified of the update

## Known Limitations

This application is designed for local, offline photo curation:
- No network communication features
- No user authentication
- No multi-user access controls
- Designed for single-user environments

## Code Security

- No hardcoded credentials or secrets
- No external API calls
- No telemetry or analytics collection
- Clean git history without sensitive information

## Responsible Disclosure

We practice responsible disclosure and expect the same from security researchers:
- Provide reasonable time for fixes before public disclosure
- Be respectful and constructive in communication
- Avoid accessing or modifying data beyond the scope of the vulnerability
- Do not conduct testing on production systems you don't own

## Security Updates

Security updates will be released as soon as practically possible:
- Critical issues: Within 24-48 hours
- Important issues: Within 1 week
- Moderate issues: Within 2 weeks
- Low severity: Next scheduled release

## Questions?

For questions about security practices or policies, please open a GitHub discussion (not a public issue) or contact via email.

---

Thank you for helping keep Wedding Photo Curator secure!
