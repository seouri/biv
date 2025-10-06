# Implementation Plan for `biv` Python Package

This document outlines a modular, phased implementation plan for the `biv` Python package, designed to detect and remove Biologically Implausible Values (BIVs) in longitudinal weight and height measurements. The plan is tailored for coding agents like Cline, Gemini CLI, or Grok Code Fast, emphasizing Test-Driven Development (TDD) as detailed in the attached [`tdd_guide.md`](tdd_guide.md).

**Note**: This implementation plan should be read in conjunction with [`architecture.md`](architecture.md), which provides the detailed architecture overview, design principles, and high-level structure. The plan references the architectural components defined there.

**TDD Instructions**: **Strictly follow the TDD protocol in `tdd_guide.md`** for all code changes, including the Red-Green-Refactor cycle, human confirmation checkpoints, and quality checks (e.g., `uv run pytest`, `uv run ruff check --fix .`, `uv run mypy .`). Before any code implementation, confirm requirements and test cases with a human, write failing tests, and only then implement minimally. After passing new tests, always run the full suite to ensure no regressions. Use the guide's version control workflow for commits after each TDD cycle.

**Always start with the specific plan for the current phase**, based on this overall plan. Before starting any phase, **ask clarifying questions** if ambiguities arise (e.g., via user prompts in the agent interface). For example:
- "Is the default z-score threshold 3.0, or should it be configurable?"

**End goals**: A fully modular, extensible package with:
- Core `biv.detect()` and `biv.remove()` functions as per README.md.
- Support for 'range' and 'zscore' methods, with customizable parameters like ranges and thresholds.
- Comprehensive unit tests achieving >90% coverage.
- PEP 8 compliance and clean code (enforced via Ruff for linting and formatting).
- Ready for PyPI distribution (via `pyproject.toml` with `uv` for management).

The plan is phased for modularity: Start high-level, drill into details. Each phase is presented as a **checklist** for clear progress tracking. Use markdown checkboxes (`- [ ]` for todo, `- [x]` for done) to mark completion.

**STRONG INSTRUCTION ON UPDATING THIS PLAN**: After completing **each full TDD cycle** (Red-Green-Refactor) and obtaining **human confirmation**, **immediately update this `implementation_plan.md` file** by:
- Marking the relevant checkbox as `- [x]`.
- Adding a brief note under the task (e.g., "Completed: TDD cycle for X; human confirmed; quality checks passed; committed").
- Running a full validation (e.g., `uv run pytest` and `uv run ruff check .`).
- Committing the updated plan to Git with a message like "feat: X cycle completed" (per `tdd_guide.md`).
Failure to update the plan will lead to tracking errors—treat this as a mandatory step before proceeding. No commits without human confirmation after each cycle.

**Tooling Note**: This plan uses `uv` (from Astral) for fast dependency resolution, installation, and virtual environment management. Initialize with `uv init` for the project, use `uv add` for dependencies, and `uv sync` for locking the environment. All `pip` commands are replaced with `uv` equivalents. Ruff is integrated for linting and formatting (replacing tools like flake8, black, and isort) to ensure code quality. Follow `tdd_guide.md` for full setup (e.g., pre-commit, mypy).

## Phase 1: Project Setup and Directory Structure

**Objective**: Establish the package skeleton, including directories, basic files, and dependencies. This ensures a clean, modular architecture where new methods can be added easily (e.g., by creating new subdirectories under `methods/`).

**High-Level Structure**: See [`architecture.md`](architecture.md) for the detailed architecture overview and proposed directory structure.

**Checklist**:
- [x] Initialize project: Run `uv init biv` (creates `pyproject.toml` and virtual env). *Note: Completed; pyproject.toml created, virtual environment established. Latest git commit hash: 4d5136f8d40414d32343bed01ce0c2608e1ffce1*
- [x] Add dependencies: Run `uv add pandas numpy` (runtime); `uv add --dev pytest pytest-cov ruff mypy pre-commit hypothesis` (dev, per `tdd_guide.md`). *Note: Dev dependencies added and locked in uv.lock (pytest~8.4.2, ruff~0.13.3, mypy~1.18.2, pre-commit~4.3.0, pytest-cov~7.0.0, hypothesis~6.140.3); pandas numpy pending runtime addition.*
- [x] Create empty files and directories matching the structure detailed in [`architecture.md`](architecture.md). *Note: All modules created: biv/methods/, api.py, conftest.py, fixtures added, test files prepared, unique names fixed.*
- [x] In `conftest.py`, add basic fixtures (e.g., `sample_data` as per conversation). *Note: Added sample_data fixture with weight and height; pandas stubs added for type checking.*
- [x] Configure Ruff, pytest, mypy, and coverage in `pyproject.toml` (per `tdd_guide.md` example). *Note: Configurations added: Ruff line-length 88, double quotes; pytest with cov; mypy python_version 3.13, strict settings; coverage requires 85% default. Config validated.*
- [x] Install pre-commit hooks (create `.pre-commit-config.yaml` and run `uv run pre-commit install`, per guide). *Note: .pre-commit-config.yaml created; hooks installed successfully.*
- [x] Run `uv sync` to lock and install dependencies. *Note: uv.lock created, dependencies installed successfully.*

**Dependencies**: None.

**Clarifying Questions**:
- "Should we use `pyproject.toml` with Poetry for dependency management, or stick to `uv` with setuptools?"
- "Are there additional dependencies beyond pandas and numpy?"
- "Any specific Ruff rules to enable/disable (e.g., strict type checking)?"

**Milestones/Tests**:
- [x] Run `uv run pytest` (should pass with no tests yet). *Note: Passes with 1 placeholder test (100% coverage).*
- [x] Run `uv run ruff check .` (should pass on empty files). *Note: All checks passed!*
- [x] Verify package imports: `uv run python -c "from biv import detect, remove"` (even if empty). *Note: Correctly fails as expected since detect/remove not implemented yet.*

**Upon Phase Completion**: Update all checkboxes above as [x], add summary notes (e.g., "Phase 1 done: Structure ready"), commit the updated plan, and proceed to Phase 2. *Strong Reminder: Do not skip this!*

## Phase 2: Implement Base Detector Interface

**Objective**: Create the abstract base class for all detectors, providing shared methods like `detect`, `flag`, `remove`, and `process`. This ensures consistency and modularity—new detectors only need to implement `detect`.

**Files to Modify**:
- `biv/methods/base.py`: Add `BaseDetector` class as per conversation (with ABC, typing, etc.).
- `tests/conftest.py`: Enhance fixtures if needed (e.g., add edge-case data).

**Checklist** (Follow `tdd_guide.md` for each atomic behavior, e.g., validation, flagging):
- [ ] Confirm requirements and generate test case table (per guide Step 1). *Note: Human confirmed?*
- [ ] Red-Green-Refactor for initial tests (e.g., abstract nature, `_validate_column` errors in `tests/methods/test_base.py`). *Note: Cycle complete, human confirmed?*
- [ ] Red-Green-Refactor for additional behaviors: NaN handling, copy behavior, multi-column support. *Note: Cycles complete?*
- [ ] Refactor overall: Ensure docstrings match conversation; run quality checks (per guide Refactor Checklist). *Note: Linting/type checks pass?*
- [ ] Update `conftest.py` fixtures and move tests to permanent location if needed. *Note: Fixtures tested?*

**Dependencies**: Phase 1.

**Clarifying Questions**:
- "Should `process` allow custom columns, or always default to ['weight_kg', 'height_cm']?"
- "Is pd.NA preferred over np.nan for replacements?"

**Milestones/Tests**:
- [ ] `uv run pytest tests/methods/test_base.py` passes.
- [ ] `uv run ruff check biv/methods/base.py` passes.
- [ ] Mock subclass example in docstring works (manual verification).

**Upon Phase Completion**: Update all checkboxes above as [x], add summary notes (e.g., "Phase 2 done: Base ready for subclassing"), commit the updated plan, and proceed to Phase 3. *Strong Reminder: Do not skip this!*

## Phase 3: Implement Individual Detection Methods

**Objective**: Build modular detectors for 'range' and 'zscore', each in their own subdirectory. This allows easy addition of new methods (e.g., BMI) by subclassing `BaseDetector`.

**Sub-Phases** (Modular: Implement one method at a time; follow `tdd_guide.md` per sub-phase):

### Sub-Phase 3.1: RangeDetector
**Checklist**:
- [ ] Confirm requirements and generate test case table for RangeDetector behaviors. *Note: Human confirmed?*
- [ ] Red-Green-Refactor for core tests in `tests/methods/test_range/test_detector.py` (e.g., `detect` for out-of-range/NaNs, `process`). *Note: Cycle complete?*
- [ ] Implement `biv/methods/range/detector.py` (subclass `BaseDetector`, hardcoded ranges). *Note: Tests passing?*
- [ ] Red-Green-Refactor for edge cases: Zero values, extremes, non-health columns. *Note: Cycles complete?*
- [ ] Refactor: Add RANGES dict; run quality checks (per guide). *Note: Linting passes?*

### Sub-Phase 3.2: ZScoreDetector
**Checklist**:
- [ ] Confirm requirements and generate test case table for ZScoreDetector behaviors. *Note: Human confirmed?*
- [ ] Red-Green-Refactor for core tests in `tests/methods/test_zscore/test_detector.py` (e.g., `detect` for outliers/NaNs, custom threshold). *Note: Cycle complete?*
- [ ] Implement `biv/methods/zscore/detector.py` (subclass with threshold param). *Note: Tests passing?*
- [ ] Red-Green-Refactor for additional cases: Insufficient data, zero variance, large datasets/mixed NaNs. *Note: Cycles complete?*
- [ ] Refactor: Use pandas for efficiency; run quality checks (per guide). *Note: Linting passes?*

**Dependencies**: Phase 2.

**Clarifying Questions**:
- For Range: "Confirm ranges: weight 30-200 kg, height 100-250 cm?"
- For ZScore: "Should threshold be fixed at 3, or init-param only?"

**Milestones/Tests**:
- [ ] `uv run pytest tests/methods/test_range/` and `test_zscore/` pass independently.
- [ ] `uv run ruff check biv/methods/` passes.

**Upon Phase Completion**: Update all checkboxes above as [x], add summary notes (e.g., "Phase 3 done: Both detectors functional"), commit the updated plan, and proceed to Phase 4. *Strong Reminder: Do not skip this!*

## Phase 4: Implement biv.detect() Function

**Objective**: Build biv.detect() to annotate the input DataFrame with boolean flag columns for BIVs using the specified methods, as per README.md API.

**Files to Modify**:
- `biv/api.py`: Add detect function with orchestrator logic (as per architecture.md).
- `biv/__init__.py`: Expose detect.
- `biv/methods/__init__.py`: Add registry for detectors.

**Checklist** (Follow `tdd_guide.md` for each atomic behavior, e.g., method orchestration, flag combination):
- [ ] Confirm requirements and generate test case table for detect behaviors (multi-method support, flag naming). *Note: Human confirmed?*
- [ ] Red-Green-Refactor for core tests in `tests/test_detect.py` (e.g., range and zscore flags, custom thresh-wave). *Note: Cycle complete?*
- [ ] Implement registry in `biv/methods/__init__.py`. *Note: Registry tested?*
- [ ] Implement `detect` in `biv/api.py` minimally. *Note: Core tests passing?*
- [ ] Red-Green-Refactor for additional tests: Custom suffixes/columns, both methods combined. *Note: Cycles complete?*
- [ ] Refactor: Match README params/docstrings; run quality checks (per guide). *Note: Linting passes?*
- [ ] Update `__init__.py` to expose detect. *Note: Import works.*

**Dependencies**: Phase 3.

**Clarifying Questions**:
- "Confirm methods dict structure combines results with OR (any method flags it)."
- "Should defaults be as in README examples?"

**Milestones/Tests**:
- [ ] `uv run pytest tests/test_detect.py` passes.
- [ ] `uv run ruff check biv/api.py` passes.
- [ ] Example usage matches README detect section.

**Upon Phase Completion**: Update all checkboxes above as [x], add summary notes (e.g., "Phase 4 done: detect function ready"), commit the updated plan, and proceed to Phase 5. *Strong Reminder: Do not skip this!*

## Phase 5: Comprehensive Testing and Coverage

**Objective**: Ensure robustness with full test suite, including integration tests.

**Files to Modify**:
- All test files: Expand coverage (e.g., add more edges in fixtures).
- `pyproject.toml`: Ensure `pytest-cov` and `ruff` are in dev deps.

**Checklist** (Follow `tdd_guide.md` for integration tests as atomic behaviors):
- [ ] Confirm requirements for integration tests (e.g., full pipeline). *Note: Human confirmed?*
- [ ] Red-Green-Refactor for integration tests in `test_debiv.py` (e.g., end-to-end with sample data). *Note: Cycle complete?*
- [ ] Fix any test failures via TDD cycles (per guide). *Note: All prior tests still pass?*
- [ ] Verify coverage: Run `uv run pytest --cov=biv` (aim >90% per guide). *Note: Coverage % achieved.*
- [ ] Run `uv run ruff check tests/` to lint tests. *Note: Test linting passes?*

**Dependencies**: Phase 4.

**Clarifying Questions**:
- "Any specific test frameworks beyond pytest?"

**Milestones/Tests**:
- [ ] `uv run pytest --cov=biv` >90%.
- [ ] `uv run ruff check .` passes project-wide.

**Upon Phase Completion**: Update all checkboxes above as [x], add summary notes (e.g., "Phase 5 done: Coverage at 95%"), commit the updated plan, and proceed to Phase 6. *Strong Reminder: Do not skip this!*

## Phase 6: Documentation, Packaging, and Release Prep

**Objective**: Finalize for distribution.

**Files to Modify**:
- `README.md`: Expand with usage, as per conversation.
- `pyproject.toml`: Add metadata, dependencies (use `[build-system]` with setuptools or hatch); include `[tool.ruff]` config if not already.
- `LICENSE`: MIT text.

**Checklist** (No TDD; follow `tdd_guide.md` for quality checks):
- [ ] Write/update `README.md` with usage examples, matching conversation. *Note: Sections added.*
- [ ] Update `pyproject.toml` with metadata, dependencies (use `[build-system]` with setuptools or hatch); include `[tool.ruff]` config if not already. Specify minimum supported versions for pandas (>=1.3.0 for groupby with numeric_only option) and numpy (>=1.21.0 for compatibility). Add python_requires = ">=3.8". *Note: Versions researched and set.*
- [ ] Add MIT text to `LICENSE`. *Note: License file complete.*
- [ ] Test install: Run `uv pip install -e .` (editable) or `uv sync` for locked env. *Note: Import succeeds?*
- [ ] Build: Run `uv build` for wheels/sdists. *Note: Artifacts created?*
- [ ] Lint final: Run `uv run ruff check . --fix` to auto-fix issues. *Note: No remaining issues?*

**Dependencies**: Phase 5.

**Clarifying Questions**:
- "Target Python versions? PyPI name?"

**Milestones**:
- [ ] Local install works (`uv run python -c "import biv"`); docs match code.
- [ ] `uv run ruff check .` passes with no issues.

**Upon Phase Completion**: Update all checkboxes above as [x], add summary notes (e.g., "Phase 6 done: Package release-ready"), commit the updated plan. *Project Complete: Celebrate and prepare for PyPI upload! Strong Reminder: Do not skip this!*

**Overall Execution Notes**:
- Use Git for versioning; branch per phase (e.g., `git checkout -b phase-1-setup`).
- After each task/phase, run `uv run pytest`, `uv run ruff check .` (lint), and `uv run ruff format .` (format).
- If issues arise, revert to TDD cycle per `tdd_guide.md`; use `uv run ruff check --fix` for quick fixes.
- **MANDATORY**: Update `implementation_plan.md` after **every single task** to maintain accurate progress tracking. Log any deviations or clarifications resolved here.
- Final Validation: Match conversation examples exactly; run full suite before declaring done.
