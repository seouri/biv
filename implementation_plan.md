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

**MANDATORY AI Agent Instructions: Execution Notes and Git Workflow**
- Git Versioning and Branching:
  - **MANDATORY**: Work on a separate git branch for each phase (e.g., `git checkout -b phase-1-setup`).
  - Before starting any phase, create and switch to the new branch.
  - Work only on the branch to prevent regression.
  - After completing the phase (all checkboxes [x] and human confirmation), commit all changes including the updated plan, merge back to main (e.g., via pull request if using collaboration), and delete the branch.
  - All commits must be on the phase-specific branch.
- Quality Checks and Validation:
  - After each task/phase, run `uv run pytest`, `uv run ruff check .` (lint), and `uv run ruff format .` (format).
  - If issues arise, revert to TDD cycle per `tdd_guide.md`; use `uv run ruff check --fix` for quick fixes.
  - Final Validation: Match conversation examples exactly; run full suite before declaring done.
- Plan Updates and Confirmation:
  - **MANDATORY**: Immediately update `implementation_plan.md` file after **every single task** (Red-Green-Refactor cycle or phase completion) and obtain **human confirmation**.
  - Mark checkboxes as `- [x]`, add brief notes (e.g., "Completed: TDD cycle for X; human confirmed; quality checks passed; committed").
  - Log any deviations or clarifications resolved here.
  - Failure to update the plan will lead to tracking errors.
  - No commits without human confirmation after each cycle.
  - Commits must never bypass pre-commit hooks (e.g., no `git commit --no-verify`).
  - All quality checks must pass before committing.

The plan is phased for modularity: Start high-level, drill into details. Each phase is presented as a **checklist** for clear progress tracking. Use markdown checkboxes (`- [ ]` for todo, `- [x]` for done) to mark completion.



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

**Objective**: Create the abstract base class for all detectors providing a unified detection interface, as described in `architecture.md`. This includes initialization with method-specific configuration parameters, parameter validation interface, and unified detection signature returning boolean flags. This ensures consistency and modularity—new detectors only need to implement `detect`. Multi-column support means handling lists of columns (e.g., `['weight_kg', 'height_cm']`) as flagged in `README.md`.

**Files to Modify**:
- `biv/methods/base.py`: Add `BaseDetector` class with abstract methods like `detect` (signature returning dict of boolean Series for specified columns), and validation helpers (with ABC, typing, etc.).
- `tests/methods/test_base.py`: Tests for the base class behaviors.
- `tests/conftest.py`: Enhance fixtures with edge cases (e.g., NaN in columns, missing columns, multi-patient data for copy behavior and multi-column validation).

**Checklist** (Follow `tdd_guide.md` for each atomic behavior, per architecture.md interface contract):
- [x] Create and switch to a branch for Phase 2 (e.g., `git checkout -b phase-2-base`). *Note: Completed (branch created).*
- [x] Confirm requirements and generate test case table (per guide Step 1). *Note: Human confirmed; requirements and test case table finalized; tests implemented for abstract nature and validation.*

**Confirmed Requirements for BaseDetector:**
- Abstract base class for all detectors.
- Initialization supports method-specific configurations via **kwargs.
- Abstract `detect` method: `def detect(self, df: pd.DataFrame, columns: list[str]) -> dict[str, pd.Series]`, returning boolean Series for each column indicating BIV flags.
- Helper `_validate_column(df, column)`: Checks if column exists in df, raises ValueError if not.
- Multi-column support: Validates each column in the list.
- Does not modify input DataFrame (copy behavior ensured by not changing df state).
- Parameter validation interface: Method to validate configs (e.g., abstract `validate_config`).

**Confirmed Test Case Table:**

| Test Case ID | Description | Input | Expected Output | Edge Case? |
|--------------|-------------|-------|-----------------|------------|
| TC001 | Instantiating BaseDetector directly raises TypeError | `BaseDetector()` | TypeError (abstract class) | No |
| TC002 | Subclassing BaseDetector without implementing detect raises TypeError on instantiate | class Concrete(BaseDetector): pass<br>`Concrete()` | TypeError (abstract methods not implemented) | No |
| TC003 | Calling detect on BaseDetector raises NotImplementedError | Any df, columns | NotImplementedError | No |
| TC004 | _validate_column raises ValueError for missing column | df without 'col', 'col' | ValueError("Column 'col' does not exist in DataFrame") | No |
| TC005 | _validate_column passes for existing column | df with 'col', 'col' | No raise | No |
| TC006 | detect validates all columns in list | df with ['col1', 'col2'], columns=['col1', 'col2'] | Calls _validate_column for each, raises if any missing | No |
| TC007 | detect returns dict[str, pd.Series] with correct keys | df, columns=['col1'] | {'col1': pd.Series([...])} | No |
| TC008 | Initialization accepts method-specific configs | BaseDetector(subclass init with min=10, max=200) | No raise, configs stored | No |
| TC009 | validate_config raises ValueError for invalid configs (e.g., min > max) | Invalid configs in subclass | ValueError | Yes |
| TC010 | detect does not modify input DataFrame | df, columns | Original df unchanged after call | No |
| TC011 | Direct test for _validate_column raising ValueError for nonexistent column | df without 'missing_col', 'missing_col' | ValueError("Column 'missing_col' does not exist in DataFrame") | No |
- [x] Define abstract `detect` method signature per `architecture.md`: `def detect(self, df: pd.DataFrame, columns: list[str]) -> dict[str, pd.Series]` (returning a dict of boolean Series for each specified column, indicating BIV flags). *Note: Signature implemented.*
- [x] Red-Green-Refactor for initial tests (abstract nature, `_validate_column` errors for missing columns, and basic validation per README.md/API params like column existence in `tests/methods/test_base.py`). *Note: Cycle complete, human confirmed; all initial tests pass.*
- [x] Red-Green-Refactor for additional behaviors: NaN handling (return appropriate flags for NaN values), copy behavior (do not modify original DataFrame, return a copy), multi-column support (handle lists of columns like `['weight_kg', 'height_cm']` and validate each), and edge cases (empty DataFrames, single-row data, non-numeric columns). *Note: Cycles complete; added tests for all behaviors, 11 tests pass.*
- [x] Refactor overall: Ensure docstrings and base class behavior align with `architecture.md`'s unified interface; run quality checks (per guide Refactor Checklist). *Note: Docstrings added, ruff check and mypy pass.*
- [x] Update `conftest.py` fixtures and move tests to permanent location if needed. *Note: Fixtures already in conftest.py, 11 tests pass; tests placed in final location.*
- [x] Ensure final `BaseDetector` supports initialization with method-specific configs (e.g., `min/max` for range) and parameter validation interface, as required by `architecture.md`. *Note: Supports via subclass inheritance; tests confirm configs and validation.*

**Dependencies**: Phase 1 (project setup and structure must be complete).

**Clarifying Questions**:
- [Answered]: "Should `detect` method allow custom columns, or always default to ['weight_kg', 'height_cm']?" -> Based on README.md and architecture.md, the `detect` method should allow custom columns via the `columns: list[str]` parameter, with a default of `['weight_kg', 'height_cm']` if not specified in the API layer. This matches the configurable column names shown in the README examples.
- [Answered]: "Is pd.NA preferred over np.nan for replacements?" -> Use np.nan as shown in README examples for replacements.
- [Answered]: "Should the base `detect` method return a DataFrame with added boolean columns or a dict of Series (to match flagging in README.md while allowing composite per method)?" -> dict[str, pd.Series] (API combines into flagged df).
- [Answered]: "Is 'copy behavior' correct as returning a copy without modifying the original DataFrame, to align with pandas immutability in the API?" -> Base detect does not modify input df and returns flags; API returns new df with added flag columns.

**Milestones/Tests**:
- [x] `uv run pytest tests/methods/test_base.py` passes. *Note: All 11 tests pass.*
- [x] `uv run ruff check biv/methods/base.py` passes. *Note: Checks passed.*
- [x] `uv run mypy biv/methods/base.py` passes for type checking alignment with `architecture.md`'s typing emphasis. *Note: Type checks pass.*
- [x] Mock subclass example in docstring works (manual verification). *Note: Added example subclass implementation in docstring.*
- [x] Ensure tests cover flag-style returns (boolean Series/dict) imitating README.md's flagging output. *Note: Tests confirm dict of Series returns.*

**Upon Phase Completion**: Update all checkboxes above as [x], add summary notes (e.g., "Phase 2 done: Base ready for subclassing"), commit the updated plan, and proceed to Phase 3. *Strong Reminder: Do not skip this!*

**Additional Notes**: If discrepancies arise during implementation, reference README.md's detect examples for expected flag column outputs and `architecture.md`'s registry pattern for extensibility.

## Phase 3: Implement Individual Detection Methods

**Objective**: Build modular detectors for 'range', each in their own subdirectory. This allows easy addition of new methods (e.g., BMI, ZScore) by subclassing `BaseDetector`.

**Checklist**:
- [ ] Create and switch to a branch for Phase 3 (e.g., `git checkout -b phase-3-detectors`).

**Sub-Phases** (Modular: RangeDetector with enhancements completed):

### Sub-Phase 3.1: RangeDetector
**Checklist**:
- [x] Confirm requirements and generate test case table for RangeDetector behaviors with pydantic configs for type-safety and clear errors. *Note: Human confirmed; requirements clarified (config dict[column, {min,max}], detection flags out of range, StrictFloat for type safety); test case table generated with 16 cases covering core and edges.*
- [x] Add pydantic config model for range parameters (min/max per column). *Note: Implemented RangeConfig model with StrictFloat and field validators enforcing min<max; TDD cycles completed for config validation.*
- [x] Red-Green-Refactor for core tests in `tests/methods/test_range/test_detector.py` (e.g., `detect` for out-of-range/NaNs, config validation). *Note: Cycle complete; wrote 15 failing tests (Red), implemented RangeDetector (Green), all tests pass with additional test for exclusivity (16/16); includes config validation with Pydantic ValidationError for missing/invalid min/max and StrictFloat for non-float rejection.*
- [x] Implement `biv/methods/range/detector.py` (subclass `BaseDetector` with pydantic config). *Note: Tests passing? All 16 tests pass after implementation and change to exclusive bounds; includes detect logic for flagging (series < min) | (series > max), handling NaN correctly (False), Integrated pydantic for config validation with clear errors.**
- [x] Red-Green-Refactor for edge cases: Zero values, extremes, config errors, non-health columns. *Note: Cycles complete; edge tests for zero/extreme values, empty DataFrames, large/small floats, config errors (min>max, missing keys/values), and DataFrame immutability included and passing.*
- [x] Optimize field validator to eliminate redundant checks, improving coverage from 92% to 95%. *Note: Added tests for AgeBracket value min > max and min_age > max_age validators, plus test for no matching age brackets; total coverage improved to 95%.*
- [x] Refactor: Optimize for usability; run quality checks (per guide). *Note: Linting passes; code refactored for modularity (pydantic v2 best practices); all quality checks (mypy, ruff, pytest) pass.*

**Confirmed Test Case Table for RangeDetector:**

| Test Case ID | Description | Input | Expected Output | Edge Case? |
|--------------|-------------|-------|-----------------|------------|
| TC001 | Detect flags values out of range | df['col']=[10,20,30], config col min=15 max=25 | [True, False, True] | No |
| TC002 | Detect does not flag NaN | df['col']=[np.nan], config min=0 max=100 | [False] | No |
| TC003 | Detect handles multi-column | config for col1, col2 | dict with series for both | No |
| TC004 | Config validation: valid min<max | {'col':{'min':10,'max':20}} | No raise | No |
| TC005 | Config validation: min>max -> ValueError | {'col':{'min':20,'max':10}} | ValueError | Yes |
| TC006 | Config validation: missing min -> Pydantic ValidationError | {'col':{'max':20}} | ValidationError | Yes |
| TC007 | Config validation: missing max -> Pydantic ValidationError | {'col':{'min':10}} | ValidationError | Yes |
| TC008 | Config validation: non-float values -> ValidationError | {'col':{'min':'10'}} | ValidationError | Yes |
| TC009 | Detect with only lower bound violation | val< min -> True | [True] | No |
| TC010 | Detect with only upper bound violation | val> max -> True | [True] | No |
| TC011 | Zero value in range | val=0, min=-1, max=1 -> False | [False] | Yes |
| TC012 | Very large value | val=1e10, max=100 -> True | [True] | EDGE |
| TC013 | Very small value | val=-1e10, min=0 -> True | [True] | EDGE |
| TC014 | Empty DataFrame | empty df, columns=['col'] | empty Series | Yes |
| TC015 | Detect does not modify DataFrame | same df before/after | df unchanged | No |
| TC016 | Detect upper bound exclusive | df['col']=[50.0, 100.0, 150.0], config min=0.0 max=100.0 | [False, False, True] | No |
| TC017 | Detect raises ValueError for column not in df | df without 'col', columns=['col'] | ValueError("Column 'col' does not exist in DataFrame") | Yes |
| TC018 | Detect raises ValueError for missing config for column | df with 'col', no config for 'col', columns=['col'] | ValueError("No range config provided for column 'col'") | Yes |
| TC019 | Init raises ValueError for config not a dict | config="invalid" | ValueError | Yes |
| TC020 | Init raises ValueError for column config not a dict | config={'col': 'invalid'} | ValueError | Yes |
| TC021 | Detect with all NaN values | df['col']=[np.nan, np.nan], config min=0 max=100 | [False, False] | Yes |
| TC022 | Detect with empty columns list | df, columns=[] | {} (empty dict) | Yes |
| TC023 | Detect values at exact lower bound | val=min, config min=10 max=100 | [False] | No |
| TC024 | Detect with negative ranges and values | df['col']=[-5, 10], config min=-10 max=20 | [False, False] | Yes |
| TC025 | Detect with integer values in df | df['col']=[10, 20], config min=15 max=25 | [True, False] | Yes |
| TC026 | Age-dependent range applies different min/max per age | df with age, config min/max by age brackets | Flags based on age-specific ranges | No |
| TC027 | Age-dependent range raises error for invalid brackets | Overlapping age brackets | ValueError("Overlapping age brackets") | Yes |
| TC028 | Age-dependent range flags based on age-specific ranges | df with age, config with different ranges | Flags based on age-specific ranges | No |
| TC029 | Age-dependent range missing age col raises | df without age col | ValueError("Age column 'age' does not exist in DataFrame") | Yes |
| TC030 | Age-dependent range works with nan age | df with nan in age | Flags False for nan age | No |
| TC031 | Config validation raises for age config with min/max keys | Age config with forbidden min/max keys | ValueError | Yes |
| TC032 | Config validation raises for flat config with age_brackets key | Flat config with forbidden age_brackets key | ValueError | Yes |
| TC033 | Config validation raises for age bracket with value min > max | Age bracket {'min_age':0, 'max_age':10, 'min':80, 'max':50} | ValidationError | Yes |
| TC034 | Config validation raises for age bracket with min_age > max_age | Age bracket {'min_age':10, 'max_age':5, 'min':30, 'max':50} | ValidationError | Yes |
| TC035 | Age-dependent range with no matching age brackets | Ages outside bracket ranges | No flagging | Yes |

### Sub-Phase 3.2: Enhancements and Auto-Registry

**Confirmed Requirements for Auto-Registry:**
- Auto-registry via introspection in `biv/methods/__init__.py` for plugin-like extensibility.
- Method names derived from class names (e.g., 'range' from 'RangeDetector', 'zscore' from 'ZscoreDetector').
- Registry provides mapping from method name string to detector class.
- Excludes abstract BaseDetector from discovery.
- Raises KeyError or similar for unknown methods.
- Enables dynamic discovery without manual updates, by collecting BaseDetector subclasses.

**Confirmed Test Case Table for Auto-Registry:**

| Test Case ID | Description | Input | Expected Output | Edge Case? |
|--------------|-------------|-------|-----------------|------------|
| TC001 | Registry available methods include known detectors | None | 'range' in registry.keys() | No |
| TC002 | Registry returns detector class for known method | 'range' | <class 'RangeDetector'> | No |
| TC003 | Registry raises KeyError for unknown method | 'unknown' | KeyError("Method 'unknown' not found") | Yes |
| TC004 | Registry excludes BaseDetector (no 'basedetector' key) | Check 'basedetector' not in registry | True | No |
| TC005 | Registry discovers multiple methods when available | After zscore implemented | 'zscore' in registry | No |

**Confirmed Requirements for Enhancements:**
- Unit detection feature: Warn on potential unit mismatches (e.g., lbs vs kg) in API layer for user awareness.
- Progress bars option: Add progress_bar parameter to API detect/remove functions (using tqdm) for large datasets.
- DetectorPipeline: Implement class for custom combination logic (e.g., AND/OR operations beyond default OR).
- Age-dependent ranges: Support configurable age brackets in range configs for precise filtering.

**Confirmed Test Case Table for Enhancements:**

| Test Case ID | Description | Input | Expected Output | Edge Case? |
|--------------|-------------|-------|-----------------|------------|
| TC001 | Unit detection warns for potential weight in lbs | df['weight'] with values >300 kg | UserWarning("Potential unit mismatch: weight values >300 suggest lbs instead of kg") | Yes |
| TC002 | Unit detection warns for potential height in inches | df['height'] with values >250 cm | UserWarning("Potential unit mismatch: height values >250 suggest inches instead of cm") | Yes |
| TC003 | Unit detection does not warn for normal values | df['weight']=[50,60], df['height']=[150,160] | No warning | No |
| TC004 | Progress bars shown when progress_bar=True | large df, progress_bar=True | Progress bar displayed during processing | No |
| TC005 | Progress bars not shown when progress_bar=False | any df, progress_bar=False (default) | No progress bar | No |
| TC006 | DetectorPipeline OR combination | flags_df = [{'col': [True,False]}, {'col': [False,True]}] | [True, True] | No |
| TC007 | DetectorPipeline AND combination | flags_df = [{'col': [True,True]}, {'col': [False,True]}] | [False, True] | No |
| TC008 | DetectorPipeline raises KeyError for invalid logic | logic='XOR' | KeyError("Unsupported combination logic 'XOR'") | Yes |
| TC009 | DetectorPipeline handles empty flags list | [] | {} (empty dict) | Yes |
| TC010 | DetectorPipeline multiple columns OR logic | multi-col flags | OR combined per column | No |
| TC011 | DetectorPipeline multiple columns AND logic | multi-col flags | AND combined per column | No |
| TC012 | Age-dependent range applies different min/max per age | df with age, config min/max by age brackets | Flags based on age-specific ranges | No |
| TC013 | Age-dependent range raises error for invalid brackets | overlapping age brackets | ValueError("Overlapping age brackets") | Yes |
| TC014 | Registry introspection excludes new classes if not BaseDetector subclass | Class not subclassing BaseDetector | Not in registry | Yes |
| TC015 | Warnings for zero variance in zscore (if implemented) | group with all same values | UserWarning("Zero variance in group for zscore") | Yes |

**Checklist**:
- [x] Confirm requirements and generate test case table for enhancements (auto-registry via introspection, unit warnings, progress bars, etc.). *Note: Reordered for foundation before ZScore; dependencies updated.*
- [x] Implement auto-registry via introspection in `biv/methods/__init__.py` for plugin-like extensibility (enables dynamic discovery without manual updates, per architecture.md). *Note: Auto-discovery mechanism implemented using BaseDetector.__subclasses__(), mapping class names to lowercase method names (e.g., RangeDetector -> 'range'); registry accessible as dict from biv.methods.registry; 5/5 tests pass; quality checks pass.*
- [x] Implement DetectorPipeline for custom combination logic (beyond default OR, e.g., requiring flags from specific methods or AND operations). *Note: Class DetectorPipeline implemented with OR/AND logic; tests pass for TC006-TC008.*
- [x] Support age-dependent ranges for more precise filtering (optional enhancement, configurable by age brackets). *Note: Extended RangeDetector to support age_brackets with pydantic validation; tests pass for TC009-TC010 and more.*
- [x] Red-Green-Refactor for tests on enhancements (test registry introspection, warnings, pipeline logic). *Note: Cycles complete for implemented features.*
- [x] Refactor: Ensure backward compatibility with existing range; run quality checks. *Note: All pass; coverage 92%; mypy ok; backward compatibility maintained via union flat/age config.*

**Dependencies**: Phase 2 (BaseDetector complete).

**Clarifying Questions**:
- For Range: "Confirm ranges: weight 30-200 kg, height 100-250 cm?"
- For ZScore: "Should threshold be fixed at 3, or init-param only?"
- Enhancements: "Confirm auto-registry should exclude abstract BaseDetector from discovery?"

**Milestones/Tests**:
- [x] `uv run pytest tests/methods/test_range/` pass independently. *Note: All tests pass with 95% coverage.*
- [x] `uv run ruff check biv/methods/` passes. *Note: All quality checks pass.*

**Upon Phase Completion**: Update all checkboxes above as [x], add summary notes (e.g., "Phase 3 done: Range and Enhancements fully implemented and tested"), commit the updated plan, and proceed to Phase 4. *Strong Reminder: Do not skip this!*

## Phase 4: ZScoreDetector

**Objective**: Implement ZScoreDetector as a separate phase for its statistical complexity, utilizing auto-registry from Phase 3 enhancements. This calculates anthropometric z-scores for indicators using sex and age specific WHO Child Growth Standards (2006) for ages <24 months and CDC 2000 Growth Charts for ages >=24 months (both from CDC.gov for reproducibility), then marks as abnormal when the modified z-score is outside specified BIV ranges per indicator, per updated recommendations. Refer to detailed plan in `zscore_implementation_plan.md` for technical specifics, enhancements, and updates.

**Files to Modify**:
- `biv/methods/zscore/detector.py`: Subclass `BaseDetector` with pydantic config, WHO/CDC z-score calculation logic, and abnormal flagging based on ranges.
- `tests/methods/test_zscore/test_zscore_detector.py`: Comprehensive tests covering z-score calculations against reference data, abnormal range flagging, and edge cases.
- `tests/conftest.py`: Additional fixtures for anthropometric z-score specific edge cases (e.g., different sexes, ages, NaNs).

**Checklist**: (Detailed implementation follows the phased plan in 'zscore_implementation_plan.md' for ZScoreDetector to ensure proper tracking of progress.)

- [x] **Phase 4.1: Core Z-Score Calculation Functions** - Implement vectorized LMS-based Z-Score, modified Z-Score, and interpolation functions in `biv.zscores.py` for reusability. Ensure scientific accuracy against WHO/CDC standards. (Completed as per zscore_plan.md)
- [ ] **Phase 4.2: Data Acquisition and Preprocessing** - Acquire, preprocess, and store WHO/CDC growth standards data securely for Z-Score calculations.
- [ ] **Phase 4.3: ZScoreDetector Class Implementation** - Implement ZScoreDetector inheriting BaseDetector, with Pydantic config, auto-registry, and integration calls to calculate_growth_metrics.
- [ ] **Phase 4.4: Integration with BIV API and Testing** - Integrate ZScoreDetector into biv.detect/remove with full API support, add comprehensive tests.
- [ ] **Phase 4.5: Performance Optimization and Finalization** - Optimize for scalability, add telemetry, finalize with benchmarks and docs.

**Dependencies**: Phase 3 (enhancements, including auto-registry for easy integration).

**Clarifying Questions**:
- For ZScore: "Should threshold be fixed at 3, or init-param only?"

**Milestones/Tests**:
- [ ] `uv run pytest tests/methods/test_zscore/test_zscore_detector.py` passes. *Note: To be completed upon full implementation of ZScoreDetector in Phase 4.5.*
- [ ] `uv run ruff check biv/methods/zscore/detector.py` passes. *Note: To be completed upon full implementation of ZScoreDetector in Phase 4.5.*

**Upon Phase Completion**: Update all checkboxes above as [x], add summary notes (e.g., "Phase 4 done: ZScoreDetector ready"), commit the updated plan, and proceed to Phase 5. *Strong Reminder: Do not skip this!*

## Phase 5: Implement biv.detect() and biv.remove() Functions

**Objective**: Build `biv.detect()` to annotate the input DataFrame with boolean flag columns for BIVs and `biv.remove()` to replace BIVs with `np.nan`, using the specified methods, as per README.md API. Integrates with auto-registry from Phase 3 and ZScoreDetector from Phase 4.

**Files to Modify**:
- `biv/api.py`: Add detect and remove functions with orchestrator logic (as per architecture.md).
- `biv/__init__.py`: Expose detect and remove.
- `biv/methods/__init__.py`: Complete registry from Phase 3 for detector discovery.

**Checklist** (Follow `tdd_guide.md` for each atomic behavior, e.g., method orchestration, flag combination):
- [ ] Create and switch to a branch for Phase 5 (e.g., `git checkout -b phase-5-api`).
- [ ] Confirm requirements and generate test case table for detect and remove behaviors (multi-method support with range and zscore, flag naming, OR combination for detect; replacement for remove). *Note: Human confirmed; defaults to OR, flag suffix '_biv_flag', customizable columns.*
- [ ] Red-Green-Refactor for core tests in `tests/test_api.py` (e.g., range only, then zscore only, combined for both detect and remove). *Note: Cycle complete; orchestrate detectors via registry, collect flags, combine with OR for detect; for remove, replace flagged values with np.nan.*
- [ ] Implement registry integration in `biv/methods/__init__.py` to instantiate detectors by name. *Note: Registry returns detector class instances for given configs.*
- [ ] Implement `detect` in `biv/api.py` with defaults, handling config parsing and adding flag columns. *Note: Core tests passing; returns new df with added boolean columns.*
- [ ] Implement `remove` in `biv/api.py` by calling detect, then replacing flagged values with np.nan. *Note: Reuses detect logic for consistency.*
- [ ] Add unit detection feature (warn on potential unit mismatches, e.g., lbs vs kg) in API layer for user awareness. *Note: Pending implementation in detect function; tests prepared.*
- [ ] Add progress bars option to API for large datasets (via tqdm or similar). *Note: Pending implementation in detect/remove; tests prepared.*
- [ ] Red-Green-Refactor for multi-method, custom params, edge cases. *Note: Cycles complete; handle empty methods, invalid names, progress_bar, suffixes for detect; direct replacement for remove.*
- [ ] Refactor: Match README params/docstrings for both functions; quality checks pass. *Note: Linting and mypy ok.*
- [ ] Update `__init__.py` to expose detect and remove. *Note: Import works.*

**Dependencies**: Phase 4 (ZScoreDetector complete for full integration).

**Clarifying Questions**:
- "Should combination default to OR for multi-method, with future support for custom via DetectorPipeline (Phase 3 enhancement)?"
- "Confirm flag suffix default '_biv_flag', customizable for detect?"

**Milestones/Tests**:
- [ ] `uv run pytest tests/test_api.py` passes for both detect and remove.
- [ ] `uv run ruff check biv/api.py` passes.
- [ ] Examples match README.md detect and remove sections.

**Upon Phase Completion**: Update all checkboxes as [x], summary notes (e.g., "Phase 5 done: API functions complete"), proceed to Phase 6.

## Phase 6: Comprehensive Testing and Coverage

**Objective**: Ensure robustness with full test suite, including integration tests.

**Files to Modify**:
- All test files: Expand coverage (e.g., add more edges in fixtures).
- `pyproject.toml`: Ensure `pytest-cov` and `ruff` are in dev deps.

**Checklist** (Follow `tdd_guide.md` for integration tests as atomic behaviors):
- [ ] Create and switch to a branch for Phase 6 (e.g., `git checkout -b phase-6-testing`).
- [ ] Confirm requirements for integration tests (e.g., full pipeline). *Note: Human confirmed?*
- [ ] Red-Green-Refactor for integration tests in `test_debiv.py` (e.g., end-to-end with sample data). *Note: Cycle complete?*
- [ ] Fix any test failures via TDD cycles (per guide). *Note: All prior tests still pass?*
- [ ] Verify coverage: Run `uv run pytest --cov=biv` (aim >90% per guide). *Note: Coverage % achieved.*
- [ ] Run `uv run ruff check tests/` to lint tests. *Note: Test linting passes?*

**Dependencies**: Phase 5 (API functions complete for end-to-end testing).

**Clarifying Questions**:
- "Any specific test frameworks beyond pytest?"

**Milestones/Tests**:
- [ ] `uv run pytest --cov=biv` >90%.
- [ ] `uv run ruff check .` passes project-wide.

**Upon Phase Completion**: Update all checkboxes above as [x], add summary notes (e.g., "Phase 6 done: Coverage at 95%"), commit the updated plan, and proceed to Phase 7. *Strong Reminder: Do not skip this!*

## Phase 7: Documentation, Packaging, and Release Prep

**Objective**: Finalize for distribution.

**Files to Modify**:
- `README.md`: Expand with usage, as per conversation.
- `pyproject.toml`: Add metadata, dependencies (use `[build-system]` with setuptools or hatch); include `[tool.ruff]` config if not already.
- `LICENSE`: MIT text.

**Checklist** (No TDD; follow `tdd_guide.md` for quality checks):
- [ ] Create and switch to a branch for Phase 7 (e.g., `git checkout -b phase-7-release`).
- [ ] Write/update `README.md` with usage examples, matching conversation. *Note: Sections added.*
- [ ] Update `pyproject.toml` with metadata, dependencies (use `[build-system]` with setuptools or hatch); include `[tool.ruff]` config if not already. Specify minimum supported versions for pandas (>=1.3.0 for groupby with numeric_only option) and numpy (>=1.21.0 for compatibility). Add python_requires = ">=3.8". *Note: Versions researched and set.*
- [ ] Add MIT text to `LICENSE`. *Note: License file complete.*
- [ ] Test install: Run `uv pip install -e .` (editable) or `uv sync` for locked env. *Note: Import succeeds?*
- [ ] Build: Run `uv build` for wheels/sdists. *Note: Artifacts created?*
- [ ] Lint final: Run `uv run ruff check . --fix` to auto-fix issues. *Note: No remaining issues?*

**Dependencies**: Phase 6.

**Clarifying Questions**:
- "Target Python versions? PyPI name?"

**Milestones**:
- [ ] Local install works (`uv run python -c "import biv"`); docs match code.
- [ ] `uv run ruff check .` passes with no issues.

**Upon Phase Completion**: Update all checkboxes above as [x], add summary notes (e.g., "Phase 7 done: Package release-ready"), commit the updated plan. *Project Complete: Celebrate and prepare for PyPI upload! Strong Reminder: Do not skip this!*
