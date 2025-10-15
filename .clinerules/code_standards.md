## Code Standards

- **Linting & Formatting**: `Ruff` (line length 88, double quotes)
- **Type Checking**: `MyPy`
- **Testing**: `Pytest` with Hypothesis for property testing
- **Import Sorting**: Ensured by `Ruff`
- **UV Enforcement**: Always prefix tool invocations with `uv run` for the following commands:
  - `pytest` (e.g., `uv run pytest tests --tb=short -v --maxfail=5`, `uv run pytest --cov --cov-fail-under=85`)
  - `mypy` (e.g., `uv run mypy src tests`, `PYTHONPATH=src uv run mypy src tests`)
  - `ruff` (e.g., `uv run ruff check --fix .`, `uv run ruff format .`, `uv run ruff check .`)
  - Coverage via `pytest-cov` (e.g., `uv run pytest --cov=biv --cov-report=html`)
  - Build tools (e.g., `uv build`)
  - Any other Python or project-specific tools (e.g., benchmarks). Never run tools directly without `uv run` unless explicitly for global context (e.g., `uv sync` itself). Align command arguments with `.pre-commit-config.yaml` and CI configurations.
