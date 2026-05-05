# Contributing to Wedding Photo Curator

Thank you for your interest in contributing to Wedding Photo Curator. This document provides guidelines and instructions for contributing to the project.

## Code of Conduct

This project is committed to providing a welcoming and inclusive environment. All contributors are expected to be respectful and constructive in their interactions.

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally
3. Create a virtual environment: `python3 -m venv venv`
4. Activate it: `source venv/bin/activate`
5. Install development dependencies: `pip install -r requirements.txt`
6. Create a feature branch: `git checkout -b feature/your-feature-name`

## Development Workflow

### Making Changes

1. Write clear, descriptive commit messages
2. Keep commits focused on a single logical change
3. Test your changes before submitting
4. Ensure all Python files compile without syntax errors
5. Maintain consistency with existing code style

### Testing Your Changes

Before submitting a pull request, verify:
- Python syntax: `python3 -m py_compile *.py`
- No emojis or non-professional characters
- Code maintains the project's quality-first philosophy
- CSV output format remains consistent
- README accuracy if documentation is modified

### Commit Message Format

Use clear, descriptive commit messages:

```
Category: Brief description of change

Optional longer explanation of what was changed and why.
- Bullet point 1
- Bullet point 2
```

Example categories:
- Feature: New functionality
- Fix: Bug fix or issue resolution
- Docs: Documentation updates
- Refactor: Code restructuring without feature changes
- Perf: Performance improvements
- Test: Test additions or modifications

## Pull Request Process

1. Push your branch to your fork on GitHub
2. Open a pull request with a clear title and description
3. Explain the purpose and impact of your changes
4. Reference any related issues
5. Ensure your branch is up to date with main
6. Request review from maintainers

### Pull Request Template

```markdown
## Description
Brief description of the changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Code refactoring

## Testing
Describe how you tested these changes

## Related Issues
Fixes #123 (if applicable)

## Checklist
- [ ] Code compiles without errors
- [ ] No new warnings introduced
- [ ] Documentation updated if needed
- [ ] Commit messages are clear
```

## Areas for Contribution

### High Priority
- Performance optimizations for large collections (5000+ photos)
- Memory usage reduction
- Multi-threading or multiprocessing support
- Enhanced error handling and validation

### Medium Priority
- Additional quality metrics (color balance, saturation)
- Metadata extraction (EXIF date, camera model filtering)
- Batch processing improvements
- Configuration file support (YAML or JSON)

### Lower Priority
- UI/UX improvements to Streamlit interface
- Additional documentation or tutorials
- Support for additional image formats
- Integration with cloud storage services

## Code Style Guidelines

### Python Standards
- Use type hints for function parameters and returns
- Follow PEP 8 conventions
- Maximum line length: 100 characters
- Use meaningful variable names
- Include docstrings for all functions and classes

### Example Function
```python
def compute_quality_score(image_path: Path) -> float:
    """
    Compute the overall quality score for an image.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Quality score between 0 and 1
        
    Raises:
        FileNotFoundError: If image file does not exist
    """
    # Implementation here
    pass
```

### Documentation Standards
- No emojis in code or documentation
- Professional tone throughout
- Clear section headers using markdown
- Code examples with proper formatting
- Links use proper markdown syntax

## Performance Considerations

When making changes, consider:
- Impact on analysis speed for large collections
- Memory usage and scalability
- Caching behavior and cache validity
- Disk I/O efficiency

## Reporting Issues

### Bug Reports
Include:
- Python version and OS
- Steps to reproduce
- Expected behavior
- Actual behavior
- Sample error messages or screenshots
- Photo collection size if relevant

### Feature Requests
Include:
- Clear description of the requested feature
- Use cases and benefits
- Potential implementation approach
- Impact on performance

## Questions or Need Help?

- Check the README for common questions
- Review existing issues and discussions
- Ask in a GitHub discussion for guidance
- Provide as much context as possible

## Review Process

Maintainers will:
1. Review your changes within a reasonable timeframe
2. Provide constructive feedback if changes are needed
3. Approve and merge when ready
4. Credit you in the CHANGELOG

## License

By contributing to this project, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to Wedding Photo Curator!
