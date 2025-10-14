# Contributing to Fame2PyGen

We appreciate your interest in contributing! This document outlines how to contribute to the project.

## Development Setup

1. Fork the repository
2. Clone your fork: `git clone https://github.com/yourusername/Fame2PyGen.git`
3. Create a virtual environment: `python -m venv venv`
4. Activate: `source venv/bin/activate` (Linux/Mac) or `venv\Scripts\activate` (Windows)
5. Install dependencies: `pip install -r requirements.txt`
6. Install in development mode: `pip install -e .`

## Code Style

- Follow PEP 8
- Use type hints
- Write docstrings for all functions
- Keep line length under 88 characters

## Testing

- Add unit tests for new features
- Run tests: `python -m pytest`
- Ensure all tests pass before submitting PR

## Submitting Changes

1. Create a feature branch: `git checkout -b feature/your-feature`
2. Make your changes
3. Add tests
4. Run tests and linting
5. Commit with clear messages
6. Push and create a pull request

## Reporting Issues

Use GitHub issues for bugs and feature requests. Include:
- FAME code sample
- Expected output
- Actual output
- Error messages (if any)

## Adding New FAME Patterns

1. Identify the pattern in `formulas_generator.py`
2. Add parsing logic in `parse_fame_formula()`
3. Implement conversion in `fame2py_converter.py`
4. Add helper functions if needed
5. Update tests and documentation
