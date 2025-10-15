# Contributing to BIV

Thank you for your interest in contributing to BIV! This guide will help you get started with minimal effort.

## Prerequisites

- Python 3.13 or later
- `uv` package manager (install from [uv documentation](https://docs.astral.sh/uv/getting-started/installation/))

## Development Setup

1. **Clone the repository**:
   ```sh
   git clone https://github.com/seouri/biv.git
   cd biv
   ```

2. **Install dependencies**:
   ```sh
   uv sync
   ```
   This installs all dependencies and activates the virtual environment.

3. **Install pre-commit hooks**:
   ```sh
   uv run pre-commit install
   ```
   This sets up linting, formatting, type checking, and testing hooks that run before each commit.

4. **Verify setup**:
   ```sh
   uv run pytest --cov
   ```
   Ensure all tests pass with coverage.

## Development Workflow

We follow Test-Driven Development (TDD) practices. See [`docs/tdd_protocol.md`](docs/tdd_protocol.md) for details.

- Write tests first (in `tests/` directory)
- Implement code in `src/biv/`
- Run tests: `uv run pytest`
- Pre-commit hooks enforce code quality (Ruff linting/formatting, MyPy type checking)
- Submit pull requests to `main` branch

## Architecture

For an overview, read [`architecture.md`](architecture.md).

## Code Standards

- **Linting & Formatting**: `Ruff` (line length 88, double quotes)
- **Type Checking**: `MyPy`
- **Testing**: `Pytest` with Hypothesis for property testing
- **Import Sorting**: Ensured by `Ruff`

## Reporting Issues

Use [GitHub Issues](https://github.com/seouri/biv/issues) for bugs, features, or questions.

## License

Contributions are under the MIT License.
