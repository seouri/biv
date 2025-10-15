# Test-Driven Development (TDD) Protocol and Best Practices

**Last Updated**: October 15, 2025

This document provides the core Test-Driven Development (TDD) protocol, focusing on the Red-Green-Refactor cycle, agent-specific guidelines, and advanced concepts for coding agents implementing TDD in Python.

## What is TDD?

Test-Driven Development (TDD) is a software development practice where you write automated tests *before* writing the production code. This ensures code is reliable, maintainable, and aligned with requirements from the outset.

## Why TDD for Coding Agents?

- **Reliability**: Tests act as a safety net, catching errors early and preventing regressions.
- **Clarity**: Forces precise specification of behavior, reducing ambiguity in code generation.
- **Efficiency**: Agents can iterate faster by validating assumptions immediately.
- **Agent-Specific Benefits**: Reduces hallucination risks by grounding generation in testable specs; enables modular, composable outputs.

## The Red-Green-Refactor Cycle

Repeat this cycle for each feature:

- **Red**: Write a failing test to define desired behavior.
- **Green**: Write minimal production code to make the test pass.
- **Refactor**: Improve code structure (both test and production code) without changing behavior.

**Strict Human Oversight**: The agent must complete one full Red-Green-Refactor cycle for a single, atomic behavior, and then **pause** to request explicit human confirmation before committing and advancing.

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

**Human Confirmation**: "Human: Confirm requirements and test cases before proceeding?"

### Step 2: Red Phase – Write the Failing Test

- Write a single test function for one atomic behavior using the Arrange-Act-Assert (AAA) pattern.
- Ensure the test is descriptive (e.g., `test_total_returns_zero_for_empty_cart`).
- Run tests to confirm it fails as expected (e.g., `NameError` or `AssertionError`).

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

### Step 3: Green Phase – Make the Test Pass

- Write the **absolute minimum** amount of production code required to make the test pass.

**Guidelines for Agents**:

- **Fake It 'Til You Make It**: In early cycles, it's acceptable to fake the implementation (e.g., `return 0`). The next test forces more generic logic through triangulation.

**Example (Minimal Implementation)**:

```python
# src/cart.py
class Cart:
    def total(self):
        return 0
```

### Step 4: Refactor Phase

- Improve structure without changing behavior.

**Refactor Checklist**:

1. **Refactor Test Code First**: Eliminate duplication with fixtures and helpers.
2. **Refactor Production Code**: Improve readability and apply principles.
3. **Run Quality Checks**: Re-run tests, linter, formatter, type checker.

**Example (After more tests)**:

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

**Human Confirmation**: After completing the cycle, pause for review.

### Step 5: Iterate and Expand

- Repeat for next behaviors until requirements met and coverage >85%.

## Advanced Concepts & Best Practices

### Beyond Unit Tests: The Testing Pyramid

A robust application needs multiple testing layers:

- **Unit Tests**: Test functions/classes in isolation (primary TDD focus).
- **Integration Tests**: Test module interactions.
- **End-to-End (E2E) Tests**: Test full application flows.

### Best Practices for Coding Agents

- **Test First, Always**: Never write production code without a failing test.
- **One Assertion per Test**: Focus on a single behavior.
- **Fast Feedback**: Ensure tests run <1 second; mock slow operations.
- **Mock Strategically**: Isolate units, but prefer real objects when possible.
- **Type Safety**: Use type hints and run `mypy`.
- **Property-Based Testing**: Use `hypothesis` for input range testing.

### Additional Guides

- **Error Handling**: Implement robust exceptions and logging patterns.
- **Async Testing**: Use `pytest-asyncio` for async code.
- **ML Integration**: Validate models with TDD for predictable outputs.
- **Debugging Cycles**: Log test failures; use `pdb` or `pytest --pdb`.
- **Test Data Management**: Use fixtures for repeatable data; consider factories for complex scenarios.
- **Parallel Testing**: Run `pytest -n auto` for concurrent execution.
- **Performance Benchmarking**: Use `pytest-benchmark` to measure improvements.
