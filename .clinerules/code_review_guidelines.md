# Code Review Guidelines

## Overview

As a Senior Principal Engineer with extensive experience leading development of high-performance scientific Python packages like NumPy and SciPy, these guidelines ensure code quality, maintainability, and performance in the BIV package. Code reviews should focus on correctness, performance, maintainability, and alignment with NumPy/SciPy ecosystem standards.

## General Principles

### Code Quality & Style
- **Scipy Standards Compliance**: Adhere to [SciPy Developer Guidelines](https://scipy.github.io/devdocs/dev/code_of_conduct.html)
- **PEP 8 Compliance**: Strict adherence to Python style guides, relaxed only when necessary for readability
- **Type Hints**: Full type annotation coverage using typing module, suitable for mypy strict mode
- **Documentation**: Comprehensive docstrings following NumPy/SciPy conventions (parameters, returns, raises sections)

### Performance Considerations
- **Numerical Efficiency**: Prefer NumPy/SciPy vectorized operations over loops
- **Memory Management**: Mindful of memory usage in large dataset processing
- **Complexity Analysis**: O(n) analysis for algorithms, prefer linear or better complexity
- **Profiling**: Consider performance implications, suggest profiling for expensive operations

### Scientific Computing Requirements
- **Numerical Stability**: Ensure algorithms are numerically stable and handle edge cases
- **Reproducibility**: Deterministic behavior with proper seeding for random operations
- **Data Validation**: Robust input validation for scientific data integrity
- **Error Messaging**: Clear, actionable error messages for scientific users

## Specific Review Criteria

### Architecture & Design
- ✅ **SOLID Principles**: Single responsibility, open/closed, Liskov substitution, interface segregation, dependency inversion
- ✅ **Composition over Inheritance**: Prefer composition for code reuse
- ✅ **Dependency Injection**: Clean dependency management
- ✅ **Interface Segregation**: Focused, minimal interfaces
- ✅ **Modular Design**: Clear separation of concerns, logical module boundaries

### Code Structure
- ✅ **Function Length**: Functions ≤ 50 lines, methods ≤ 30 lines
- ✅ **Class Complexity**: Classes with single, well-defined responsibilities
- ✅ **Naming Conventions**: Descriptive names following Python conventions (snake_case functions/variables, PascalCase classes)
- ✅ **Import Organization**: Standard library, third-party, local imports with clear grouping

### Testing Standards
- ✅ **TDD Compliance**: Tests written before implementation following TDD protocol
- ✅ **Coverage Requirements**: ≥90% line coverage, ≥90% branch coverage
- ✅ **Property Testing**: Extensive use of Hypothesis for edge case exploration
- ✅ **Integration Tests**: End-to-end testing for system components
- ✅ **Benchmarking**: Performance regression tests where applicable

### Performance & Optimization
- ✅ **Vectorization**: NumPy array operations preferred over iteration
- ✅ **Memory Efficiency**: Avoid unnecessary copies, use views where possible
- ✅ **Lazy Evaluation**: Consider generators and iterators for large datasets
- ✅ **Parallelization**: Leverage Dask/Numba for computationally intensive operations

### Documentation & Maintainability
- ✅ **API Documentation**: Complete docstrings with examples for public APIs
- ✅ **Code Comments**: Explanatory comments for complex algorithms/logic
- ✅ **Change Logs**: Clear commit messages and PR descriptions
- ✅ **TODOs/Issues**: Proper tracking of technical debt and future improvements

### Scientific Accuracy
- ✅ **Algorithm Validation**: Reference implementations and mathematical correctness
- ✅ **Statistical Methods**: Proper statistical methodology following scientific standards
- ✅ **Unit Consistency**: Careful attention to physical/scientific units
- ✅ **Reference Data**: Validation against known datasets/standards where available

## Review Process

### Automated Checks (Must Pass)
- Ruff linting and formatting (line length 88, double quotes)
- MyPy type checking (strict mode)
- Pytest test suite with coverage requirements
- Pre-commit hooks passing

### Manual Review Checklist
- [ ] **Functionality**: Code works as intended, handles edge cases
- [ ] **Performance**: No obvious bottlenecks, efficient algorithms
- [ ] **Maintainability**: Readable, well-structured, well-documented
- [ ] **Testing**: Comprehensive test coverage, edge cases covered
- [ ] **Security**: No security vulnerabilities in scientific computations
- [ ] **Compatibility**: Backward compatibility maintained

### Review Comments Style
- **Be Specific**: Reference line numbers, suggest concrete fixes
- **Explain Rationale**: Include why the suggestion improves code quality
- **Provide Examples**: Show before/after code snippets for clarity
- **Prioritize Issues**: Critical bugs > Performance > Style > Nitpicks
- **Mentor**: Explain best practices and standards being applied

## Common Issues & Anti-Patterns

### Performance Anti-Patterns
- ❌ Python loops for numerical computations (use NumPy instead)
- ❌ Unnecessary data copying (use views/buffers)
- ❌ Inefficient data structures (prefer arrays over lists for numerical data)
- ❌ Blocking I/O in performance-critical paths

### Code Quality Anti-Patterns
- ❌ Large functions/classes violating single responsibility
- ❌ Tight coupling between modules
- ❌ Missing error handling for edge cases
- ❌ Inconsistent naming or style
- ❌ Missing or inadequate documentation

### Scientific Computing Anti-Patterns
- ❌ Unvalidated algorithms (missing mathematical proofs)
- ❌ Inconsistent units or coordinate systems
- ❌ Lossy data type conversions
- ❌ Insufficient numerical precision handling

## Escalation Criteria
- **Blockers**: Breaking changes, security issues, critical bugs
- **Major Concerns**: Performance regressions >10%, API breaking changes
- **Minor Issues**: Style violations, documentation gaps
- **Nitpicks**: Subjective style preferences

## Tools & Integration
- **CI/CD**: Automated testing with coverage reporting
- **Code Quality**: Ruff for linting, MyPy for typing
- **Performance**: Built-in profiling support, benchmark CI
- **Documentation**: Automated API docs, example validation

## Continuous Improvement
- **Metrics Tracking**: Code quality metrics, performance benchmarks
- **Post-Mortem Reviews**: Learnings from production issues
- **Standards Updates**: Regular review and update of guidelines
- **Team Training**: Ongoing education on best practices
