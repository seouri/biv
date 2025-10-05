# Test-Driven Development (TDD) Guide for Coding Agents in Python with uv Workflow

**Last Updated**: October 5, 2025

This guide provides a structured, step-by-step protocol for coding agents to implement Test-Driven Development (TDD) in Python using the `uv` workflow. TDD is a software development practice where you write automated tests *before* writing the production code. This ensures code is reliable, maintainable, and aligned with requirements from the outset. The `uv` workflow leverages `uv` for fast dependency management, virtual environments, and script execution.

## Why TDD for Coding Agents in Python with uv?

  - **Reliability**: Tests act as a safety net, catching errors early and preventing regressions.
  - **Clarity**: Forces precise specification of behavior, reducing ambiguity in code generation.
  - **Efficiency**: Agents can iterate faster by validating assumptions immediately; `uv`'s speed accelerates installs and runs.
  - **Agent-Specific Benefits**: Reduces hallucination risks by grounding generation in testable specs; enables modular, composable outputs.

TDD follows the **Red-Green-Refactor** cycle:

  - **Red**: Write a failing test to define desired behavior.
  - **Green**: Write minimal production code to make the test pass.
  - **Refactor**: Improve code structure (both test and production code) without changing behavior.

Repeat this cycle for each feature. **Strict Human Oversight**: The agent must complete one full Red-Green-Refactor cycle for a single, atomic behavior, and then **pause** to request explicit human confirmation before committing and advancing. This ensures alignment while maintaining an efficient feedback loop.

-----

## Prerequisites

  - **Environment**: Python 3.13+ with `uv` installed.
  - **Project Setup**:
    1.  Initialize a project: `uv init myproject`.
    2.  Generate a `.gitignore`: `curl -sL https://www.gitignore.io/api/python > .gitignore`.
    3.  Configure `pyproject.toml` for testing, linting, and type checking:
        ```toml
        [tool.pytest.ini_options]
        addopts = "--cov --cov-report=html --cov-fail-under=85"

        [tool.coverage.run]
        branch = true
        omit = ["tests/*", "*/__init__.py"]

        [tool.ruff]
        line-length = 88
        target-version = "py313"

        [tool.ruff.format]
        quote-style = "double"

        [tool.mypy]
        python_version = "3.13"
        warn_return_any = true
        disallow_untyped_defs = true
        ```
    4.  Install dev dependencies: `uv add --dev ruff pytest pytest-cov mypy pre-commit hypothesis`.
    5.  **Lock dependencies** for reproducible builds: `uv pip compile pyproject.toml -o requirements.lock`.
    6.  Install pre-commit hooks for automated quality checks:
          - Create `.pre-commit-config.yaml`:
            ```yaml
            repos:
              - repo: https://github.com/astral-sh/ruff-pre-commit
                rev: v0.13.3 # Use a recent version
                hooks:
                  - id: ruff
                    args: [--fix, --exit-non-zero-on-fix]
                  - id: ruff-format
            ```
          - Install hooks: `uv run pre-commit install`.

-----

## Version Control Workflow

Integrate Git commits at the end of each TDD cycle to maintain a clean, atomic history.

### Creating a Feature Branch

  - **When**: At the start of a new feature, after Step 1 (Understand Requirements).
  - **Agent Action**: Create and switch to a feature branch: `git checkout -b feature/[descriptive-name]`.
  - **Human Confirmation**: "Human: Confirm feature branch `feature/cart-total-calculation` created before starting TDD cycles?"

### Making Git Commits

  - **When**: After each full Red-Green-Refactor cycle is completed and approved by a human.
  - **Agent Action**:
    1.  Run all checks: `uv run ruff check --fix .`, `uv run ruff format .`, and `uv run pytest`.
    2.  Stage changes: `git add .`.
    3.  Commit with a descriptive message using conventional commits: `git commit -m "feat: handle empty cart total"`.
    4.  Push after each commit: `git push`.
  - **Human Confirmation**: This is the primary checkpoint. The agent will pause here after each cycle.

### Squashing and Merging

  - **When**: Upon completion of the entire feature.
  - **Agent Action**: Create a Pull Request on the hosting platform (e.g., GitHub) and use the "Squash and Merge" option. This keeps the `main` branch history clean and feature-focused.
  - **Human Confirmation**: "Human: Confirm feature is complete and ready for squash merge to main?"

-----

## CI/CD Integration

Automate TDD enforcement with GitHub Actions. Create `.github/workflows/ci.yml`:

```yaml
name: Python CI
on: [push, pull_request]
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Set up Python
        uses: actions/setup-python@v5
        with: { python-version: '3.13' }
      - name: Install uv
        run: curl -LsSf https://astral.sh/uv/install.sh | sh
      - name: Install dependencies from lockfile
        run: uv sync --frozen
      - name: Run linting, formatting, and type checks
        run: |
          uv run ruff check .
          uv run ruff format --check .
          uv run mypy .
      - name: Run tests with coverage
        run: uv run pytest --cov --cov-fail-under=85
```

-----

## Step-by-Step TDD Protocol

### Step 1: Understand Requirements

  - Analyze the feature specification and break it down into the smallest possible testable behaviors.
  - For each behavior, define inputs, expected outputs, and side effects.

**Agent Action**: Generate a markdown table of test cases.

| Test Case ID | Description       | Input          | Expected Output   | Edge Case? |
|--------------|-------------------|----------------|-------------------|------------|
| TC001        | Empty cart total  | `[]`           | `0`               | Yes        |
| TC002        | Single item       | `[{"price": 10}]` | `10`              | No         |
| TC003        | Negative price    | `[{"price": -5}]` | `ValueError`      | Yes        |

**Human Confirmation**: "Human: Confirm requirements and test cases before proceeding?" *(Create feature branch after confirmation.)*

### Step 2: Red Phase – Write the Failing Test

  - Write a single test function for one atomic behavior using the Arrange-Act-Assert (AAA) pattern.
  - Ensure the test is descriptive (e.g., `test_total_returns_zero_for_empty_cart`).
  - Run `uv run pytest` to confirm it fails as expected (e.g., `NameError` or `AssertionError`).

**Example (pytest)**:

```python
# tests/test_cart.py
from src.cart import Cart

def test_total_returns_zero_for_empty_cart():
    # Arrange
    cart = Cart()
    # Act
    total = cart.total()
    # Assert
    assert total == 0
```

  - **Agent Action**: After writing the test and confirming it fails, proceed immediately to the Green Phase.

### Step 3: Green Phase – Make the Test Pass

  - Write the **absolute minimum** amount of production code required to make the test pass. Do not add any extra logic.

**Guidelines for Agents**:

  - **Fake It 'Til You Make It**: In early cycles, it's perfectly acceptable to "fake" the implementation (e.g., `return 0`). The goal is simply to make the current test pass. The *next* test you write will force you to replace the fake implementation with more generic logic. This "triangulation" ensures that every line of logic is justified by a test.

**Example (Minimal Implementation)**:

```python
# src/cart.py
class Cart:
    def total(self):
        return 0
```

  - **Agent Action**: Run `uv run pytest` to confirm the test now passes. Proceed immediately to the Refactor Phase.

### Step 4: Refactor Phase

  - With all tests passing, improve the internal structure of the code without changing its external behavior.

**Refactor Checklist**:

1.  **Refactor Test Code First**: Check the test suite for duplication. Is setup logic repeated? Introduce a `pytest` fixture. Are assertions complex? Extract helper functions. A clean test suite is as important as clean production code.
2.  **Refactor Production Code**: Improve readability, eliminate duplication, and apply design principles.
3.  **Run Quality Checks**: After every small refactoring, re-run the full test suite (`uv run pytest`), linter (`uv run ruff check --fix .`), formatter (`uv run ruff format .`), and type checker (`uv run mypy .`) to ensure no behavior has changed and quality standards are met.

**Example (After more tests are added)**:

```python
# src/cart.py (Refactored)
from typing import List

class Cart:
    def __init__(self) -> None:
        self._items: List[float] = []
    
    def add_item(self, price: float) -> None:
        if price < 0:
            raise ValueError("Price cannot be negative")
        self._items.append(price)
    
    def total(self) -> float:
        return sum(self._items)
```

**Human Confirmation (Primary Checkpoint)**: After completing a full Red-Green-Refactor cycle and passing all quality checks, the agent must pause. "Human: The test for [behavior] is now passing and the code has been refactored. Please review the changes. Confirm to commit and proceed to the next test case?" *(Commit after approval.)*

### Step 5: Iterate and Expand

  - Repeat the Red-Green-Refactor cycle for the next atomic behavior from your test case table.
  - Continue until all requirements for the feature are met and code coverage is above 85%.

-----

## Advanced Concepts & Best Practices

### Beyond Unit Tests: The Testing Pyramid

While TDD is most effective at the **unit test** level, a robust application requires multiple layers of testing. Don't rely on unit tests alone.

  - **Unit Tests**: Test a single function or class in isolation. (Your primary TDD focus).
  - **Integration Tests**: Test how modules interact. After unit-testing your `Cart` class, you might write an integration test to ensure it works correctly within a web framework's API endpoint.
  - **End-to-End (E2E) Tests**: Test the entire application flow from the user's perspective.

### Best Practices for Coding Agents

  - **Test First, Always**: Never write production code without a failing test.
  - **One Assertion per Test**: Keep tests focused on a single behavior.
  - **Fast Feedback**: Leverage `uv`'s speed. Ensure tests run in under a second. Mock slow I/O or network calls.
  - **Mock Strategically**: Mock external dependencies to isolate the unit under test, but prefer testing against real objects when possible.
  - **Type Safety**: Use type hints and run `uv run mypy .` in the refactor step.
  - **Style Enforcement**: Use pre-commit hooks with `ruff` to automate formatting and linting.
  - **Property-Based Testing**: For functions with a wide range of inputs, use `hypothesis` to generate test cases automatically and uncover edge cases.
