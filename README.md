
# BIV: Detect and Remove Biologically Implausible Values

`biv` is a Python package for detecting and removing Biologically Implausible Values (BIVs) in longitudinal weight and height measurements. Designed for researchers and data scientists in public health and epidemiology, `biv` provides a flexible and powerful way to clean biomedical data for reliable analysis.

## Core Features

- **Clear, Verb-Based API**: Separate, intuitive functions for `detect()` and `remove()` operations.

- **Highly Configurable**: Don't get stuck with hardcoded limits. Define your own custom ranges to fit your specific dataset—whether it's pediatric, geriatric, or specialized. Z-score outlier detection requires age and sex columns.

- **Multiple Detection Methods**: Natively supports range checks and z-score outlier detection.

- **Built for Pandas**: Integrates seamlessly into the pandas data analysis ecosystem.
    

## Installation

Install `biv` via pip:

```sh
pip install biv
```

Requires Python 3.8+ and pandas.

## Quick Start

The `biv` package provides two primary functions:

- `biv.detect()`: Adds boolean columns to your DataFrame to flag implausible values.
    
- `biv.remove()`: Replaces implausible values with `NaN`.
    

Here's a typical workflow:

```python
import pandas as pd
import numpy as np
import biv

# Sample longitudinal dataset
data = pd.DataFrame({
    'patient_id': [1, 1, 1, 2, 2, 2],
    'sex': ['M', 'M', 'M', 'F', 'F', 'F'],
    'age': [10, 11, 12, 12, 13, 14],
    'weight_kg': [35, 38, 999, 40, 42, 41],    # 999 is an obvious error
    'height_cm': [140, 150, 152, 155, 50, 159] # 50 is an obvious error
})

# 1. Define your detection methods and their parameters
# Use built-in defaults or customize them as needed.
detection_methods = {
    'range': {
        'weight_kg': {'min': 20, 'max': 200},
        'height_cm': {'min': 100, 'max': 220}
    },
    'zscore': {}  # No additional parameters needed
}

# 2. Detect BIVs, which adds flag columns
flagged_df = biv.detect(
    data,
    methods=detection_methods,
    weight_col='weight_kg',
    height_col='height_cm'
)
print("--- Detected BIVs ---")
print(flagged_df)
# --- Detected BIVs ---
#    patient_id sex  age  weight_kg  height_cm  weight_kg_biv_flag  height_cm_biv_flag
# 0           1   M   10         35        140               False               False
# 1           1   M   11         38        150               False               False
# 2           1   M   12        999        152                True               False
# 3           2   F   12         40        155               False               False
# 4           2   F   13         42         50               False                True
# 5           2   F   14         41        159               False               False


# 3. Remove the detected BIVs for a clean dataset
cleaned_df = biv.remove(
    data,
    methods=detection_methods,
    weight_col='weight_kg',
    height_col='height_cm'
)
print("\n--- Cleaned Data ---")
print(cleaned_df)
# --- Cleaned Data ---
#    patient_id sex  age  weight_kg  height_cm
# 0           1   M   10         35        140
# 1           1   M   11         38        150
# 2           1   M   12        NaN        152
# 3           2   F   12         40        155
# 4           2   F   13         42        NaN
# 5           2   F   14         41        159
```

## Advanced Usage: Full Configuration

The true power of `biv` comes from its flexibility. You can tailor every aspect of the detection logic. This is essential for specialized datasets, such as those in pediatrics.

### Example: Pediatric Data Cleaning

For a children's dataset, adult ranges are inappropriate. Here's how you can specify custom rules.

```python
import pandas as pd
import biv

pediatric_data = pd.DataFrame({
    'subject_id': [101, 101, 102, 102],
    'visit_age': [2, 3, 2, 3],
    'body_weight': [12.5, 14.0, 13.0, 100], # 100kg is a BIV for a 3-year-old
    'body_height': [85, 95, 88, 92]
})

# Define strict, age-appropriate rules
pediatric_methods = {
    'range': {
        'body_weight': {'min': 5, 'max': 30},
        'body_height': {'min': 60, 'max': 110}
    },
    'zscore': {}  # Requires 'visit_age' and 'sex' columns (assuming 'sex' is in data)
}

# Detect BIVs using custom column names and methods
flagged_pediatric_data = biv.detect(
    pediatric_data,
    methods=pediatric_methods,
    patient_id_col='subject_id',
    age_col='visit_age',
    weight_col='body_weight',
    height_col='body_height',
    flag_suffix='_is_implausible' # Customize the flag column suffix
)

print(flagged_pediatric_data)
#    subject_id  visit_age  body_weight  body_height  body_weight_is_implausible  body_height_is_implausible
# 0         101          2         12.5           85                       False                       False
# 1         101          3         14.0           95                       False                       False
# 2         102          2         13.0           88                       False                       False
# 3         102          3        100.0           92                        True                       False
```

By default, when multiple detection methods are used, their flags are combined using logical OR: a value is flagged as BIV if any method detects it as such. Future enhancements may allow custom combination logic, such as requiring flags from specific methods or using AND combinations.

## Unit Handling Warnings

The package currently assumes input numeric columns (weight, height) are in standard units (kg/cm). Mismatched units (e.g., weight in lbs, height in inches) can lead to incorrect range checks and z-score calculations. To handle this:

- Ensure data consistency before use.
- Standardize to kg/cm if possible.
- For zscore method: Uses WHO growth standards (<24 months) or CDC (≥24 months); ages in months; sex as 'M'/'F'. Warnings logged for potential unit issues (e.g., height >250 cm suggests inches) or age >240 mo (set to NaN). Invalid sex raises errors.

For cross-checking, consider if detection results align with expected outliers for healthy population ranges.

## API Reference

### `biv.detect(dataframe, methods, patient_id_col='patient_id', age_col='age', sex_col='sex', weight_col='weight_kg', height_col='height_cm', flag_suffix='_biv_flag', progress_bar=False)`

Identifies BIVs and returns a DataFrame with added boolean flag columns.

- **`dataframe`** (pd.DataFrame): The input data.
    
- **`methods`** (dict): A dictionary defining the detection methods and their parameters.
    
    - **`range`** (dict): Keys are column names (`weight_col`, `height_col`) and values are dictionaries with `'min'` and `'max'` keys.
        
    - **`zscore`** (dict): Defines z-score parameters. Note: Requires 'age' and 'sex' columns in the DataFrame for z-score calculation (defaults to column names specified in function parameters). Computes anthropometric z-scores for weight-for-age (WAZ), height-for-age (HAZ), weight-for-height (WHZ), BMI-for-age (BMIz), and head circumference-for-age (HEADCZ) using WHO/CDC growth standards.

        Z-Score Cutoffs for BIV Flagging:
        - Weight-for-age: <-5 or >8
        - Height-for-age: <-5 or >4
        - Weight-for-height: <-4 or >8
        - BMI-for-age: <-4 or >8
        - Head circumference-for-age: <-5 or >5

        For reproducibility, ZScoreDetector uses WHO/CDC reference data:
        - WHO Child Growth Standards (2006) for ages <24 months: https://www.cdc.gov/growthcharts/who-data-files.htm
        - CDC 2000 Growth Charts for ages ≥24 months: https://www.cdc.gov/growthcharts/cdc-data-files.htm
        Data is cached locally in the `data/` subdirectory as .npz files.

        Optional columns: 'head_circ_cm' for head circumference measurements.
            
- **`..._col`** (str): Column name identifiers for patient ID, age, sex, weight, and height.
    
- **`flag_suffix`** (str): The suffix for the new boolean flag columns.
    
- **`progress_bar`** (bool, default False): Whether to display a progress bar during processing.


**Returns**: A `pandas.DataFrame` with added flag columns indicating BIVs.

### `biv.remove(dataframe, methods, ...)`

Identifies and removes BIVs, replacing them with `numpy.nan`. It accepts the same parameters as `biv.detect()`.

**Returns**: A `pandas.DataFrame` with BIVs replaced by `NaN`.


## Contributing

Contributions are welcome! We follow Test-Driven Development (TDD) practices as outlined in [`tdd_guide.md`](tdd_guide.md). For an overview of the package's architecture, see [`architecture.md`](architecture.md). Please submit an issue or pull request to the [GitHub repository](https://github.com/seouri/biv).

### Development Setup

To set up the development environment for the first time after cloning this repository:

1. **Prerequisites**:
   - Python 3.13 or later installed
   - `uv` installed (see [uv documentation](https://docs.astral.sh/uv/getting-started/installation/) for installation instructions)

2. **Clone the repository**:
   ```sh
   git clone https://github.com/seouri/biv.git
   cd biv
   ```

3. **Install dependencies**:
   ```sh
   uv sync
   ```
   This installs all project dependencies (main and development) and activates the virtual environment.

4. **Install pre-commit hooks**:
   ```sh
   uv run pre-commit install
   ```
   This sets up local pre-commit hooks that match the CI checks (linting, formatting, type checking, and testing). Hooks run automatically on each commit to ensure code quality.

5. **Verify setup**:
   Run the full test suite locally to ensure everything works:
   ```sh
   uv run pytest --cov
   ```
   You should see all tests pass with sufficient coverage.

### Development Workflow

- Follow TDD practices as described in [`tdd_guide.md`](tdd_guide.md).
- Pre-commit hooks will automatically run checks before each commit, matching GitHub Actions CI.
- Push your feature branch and create a pull request. CI will verify all checks pass.


## License

This project is licensed under the MIT License. See the `LICENSE` file for details.


## Contact

For questions or support, please open an [issue on GitHub](https://github.com/seouri/biv/issues).
