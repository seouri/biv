# Project Setup and Tooling Guide

**Last Updated**: October 15, 2025

This document provides comprehensive setup instructions for Python projects using the `uv` workflow, including environment configuration, dependencies, quality tools, and CI/CD integration.

## Prerequisites

- **Environment**: Python 3.9+ with `uv` installed (latest version recommended).
- **Install uv**: `curl -LsSf https://astral.sh/uv/install.sh | sh`

## Project Initialization

1. **Initialize Project**: `uv init myproject` (creates `pyproject.toml` and virtual environment).
2. **Generate .gitignore**: `curl -sL https://www.gitignore.io/api/python > .gitignore`.
3. **Lock Dependencies**: `uv sync` (for reproducible builds).

## Dependency Management

### Runtime Dependencies

Add production dependencies:

```bash
uv add pandas numpy scipy
```

### Development Dependencies

Add testing, linting, and type checking tools:

```bash
uv add --dev ruff pytest pytest-cov mypy pre-commit hypothesis pytest-asyncio pytest-benchmark
```

## Configuration Files

### pyproject.toml

Configure tools in `pyproject.toml`:

```toml
[tool.pytest.ini_options]
addopts = "--cov --cov-report=html --cov-report=term-missing --cov-fail-under=85"
testpaths = ["tests"]
python_files = ["test_*.py", "*_test.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]

[tool.coverage.run]
branch = true
omit = ["tests/*", "*/venv/*", "*/.venv/*"]

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "def __repr__",
    "raise AssertionError",
    "raise NotImplementedError",
]

[tool.ruff]
line-length = 88
target-version = "py39"

[tool.ruff.format]
quote-style = "double"

[tool.ruff.lint]
select = [
    "E",  # pycodestyle errors
    "W",  # pycodestyle warnings
    "F",  # pyflakes
    "I",  # isort
    "B",  # flake8-bugbear
    "C4", # flake8-comprehensions
    "UP", # pyupgrade
]
ignore = [
    "E501", # line too long, handled by formatter
    "B008", # do not perform function calls in argument defaults
    "C901", # too complex
]

[tool.mypy]
python_version = "3.9"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
disallow_incomplete_defs = true
check_untyped_defs = true
disallow_untyped_decorators = true
no_implicit_optional = true
warn_redundant_casts = true
warn_unused_ignores = true
warn_no_return = true
warn_unreachable = true
strict_equality = true
show_error_codes = true

[[tool.mypy.overrides]]
module = [
    "pandas.*",
    "numpy.*",
]
ignore_missing_imports = true
```

## Pre-commit Hooks

Create `.pre-commit-config.yaml`:

```yaml
repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.6.9
    hooks:
      - id: ruff
        args: [--fix, --exit-non-zero-on-fix]
      - id: ruff-format
  - repo: local
    hooks:
      - id: mypy
        name: mypy
        entry: bash -c 'PYTHONPATH=src uv run mypy src tests'
        language: system
        files: \.py$
        types: [python]
      - id: pytest
        name: pytest
        entry: uv run pytest --cov --cov-fail-under=85 --cov-report=
        language: system
        files: \.py$
        types: [python]
```

Install hooks: `uv run pre-commit install`

## Quality Checks

Run quality checks:

```bash
# Linting and formatting
uv run ruff check --fix .
uv run ruff format .

# Type checking
uv run mypy src tests

# Testing with coverage
uv run pytest --cov --cov-report=html --cov-fail-under=85
```

## CI/CD Integration

Create `.github/workflows/ci.yml`:

```yaml
name: Python CI
on: [push, pull_request]

jobs:
  build:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.9", "3.10", "3.11", "3.12"]

    steps:
      - uses: actions/checkout@v4
      - name: Set up Python ${{ matrix.python-version }}
        uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
      - name: Install uv
        run: curl -LsSf https://astral.sh/uv/install.sh | sh
      - name: Install dependencies
        run: uv sync --frozen
      - name: Run linting and formatting
        run: |
          uv run ruff check .
          uv run ruff format --check .
      - name: Run type checking
        run: uv run mypy src tests
      - name: Run tests
        run: uv run pytest --cov --cov-fail-under=85
      - name: Upload coverage
        uses: codecov/codecov-action@v4
        with:
          file: ./coverage.xml
```

## Additional Tools

### Benchmarking

For performance testing, install `pytest-benchmark`:

```bash
uv run pytest --benchmark-only --benchmark-columns=mean,stddev
```

### Async Testing

For async code, add and use `pytest-asyncio`:

```python
# tests/test_async.py
import pytest
from src.async_module import async_function

@pytest.mark.asyncio
async def test_async_behavior():
    result = await async_function()
    assert result == expected_value
```

### Property-Based Testing

Use `hypothesis` for comprehensive input testing:

```python
# tests/test_property.py
import pytest
from hypothesis import given, strategies as st

@given(st.lists(st.floats(min_value=0, max_value=1000)))
def test_sum_properties(values):
    result = sum(values)
    assert isinstance(result, float)
    assert result >= 0
```

## Troubleshooting

- **Import Errors**: Ensure `PYTHONPATH=src` when running mypy or tests locally.
- **Coverage Issues**: Check `.coveragerc` file if coverage reports are inaccurate.
- **Hook Failures**: Run `pre-commit run --all-files` to test hooks before committing.
