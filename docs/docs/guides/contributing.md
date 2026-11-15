# Contributing to GEPA

Thank you for your interest in contributing to GEPA! This guide will help you get started.

## Environment Setup

Python 3.10 or later is required.

### Setting Up with uv (Recommended)

[uv](https://github.com/astral-sh/uv) is a fast Python package manager:

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and setup
git clone https://github.com/gepa-ai/gepa
cd gepa
uv sync --extra dev --python 3.11

# Verify installation
uv run pytest tests/
```

!!! note
    Use `uv run` prefix for all Python commands (e.g., `uv run python script.py`).

### Setting Up with conda + pip

```bash
conda create -n gepa-dev python=3.11
conda activate gepa-dev
pip install -e ".[dev]"

# Verify installation
pytest tests/
```

## Code Quality

### Pre-commit Hooks

We use pre-commit hooks for consistent code quality:

```bash
# Install hooks (one-time setup)
uv run pre-commit install

# Hooks run automatically on commit
git add .
git commit -m "your message"

# Or run manually
uv run pre-commit run --all-files
```

### Code Style

We follow the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html) and use `ruff` for linting and formatting.

### Type Checking

Run Pyright before submitting:

```bash
uv run pyright

# Or target specific modules
uv run pyright src/gepa/strategies/
```

## Running Tests

```bash
# Run all tests
uv run pytest tests/

# Run specific test file
uv run pytest tests/test_optimize.py

# Run with coverage
uv run pytest tests/ --cov=gepa
```

## Documentation

### Building Docs Locally

```bash
cd docs
pip install -r requirements.txt
python scripts/generate_api_docs.py
mkdocs serve
```

Then visit `http://localhost:8000`.

### Writing Documentation

- Use Google-style docstrings
- Include examples in docstrings
- Update relevant guides when adding features

## Contribution Types

### Bug Reports

Open an issue with:

- GEPA version
- Python version
- Minimal reproduction code
- Expected vs actual behavior

### Feature Requests

Open an issue describing:

- The use case
- Proposed solution
- Alternatives considered

### Pull Requests

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Make your changes
4. Run tests and linting
5. Submit a PR with a clear description

### Adding Adapters

We welcome new adapters! See [Creating Adapters](adapters.md) for the interface, then:

1. Create a new directory under `src/gepa/adapters/`
2. Implement the `GEPAAdapter` protocol
3. Add a README.md explaining usage
4. Add tests
5. Submit a PR

## Code of Conduct

Be respectful and constructive. We're building something together!

## Questions?

- [Discord](https://discord.gg/WXFSeVGdbW)
- [Slack](https://join.slack.com/t/gepa-ai/shared_invite/zt-3o352xhyf-QZDfwmMpiQjsvoSYo7M1_w)
- [GitHub Issues](https://github.com/gepa-ai/gepa/issues)
