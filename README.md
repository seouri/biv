
# BIV: Detect and Remove Biologically Implausible Values

`biv` is a Python package for detecting and removing Biologically Implausible Values (BIVs) in longitudinal weight and height measurements. Designed for researchers and data scientists in public health and epidemiology, `biv` provides a flexible and powerful way to clean biomedical data for reliable analysis.

## Core Features

- **Clear, Verb-Based API**: Separate, intuitive functions for `detect()` and `remove()` operations.
    
- **Highly Configurable**: Don't get stuck with hardcoded limits. Define your own custom ranges, z-score thresholds, and grouping logic to fit your specific dataset—whether it's pediatric, geriatric, or specialized.
    
- **Multiple Detection Methods**: Natively supports range checks and group-based z-score outlier detection.
    
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
    'zscore': {
        'threshold': 3.0,
        'group_by': ['sex'] # Group by sex for z-score calculation
    }
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
    'zscore': {
        'threshold': 2.5,  # Use a tighter z-score threshold
        'group_by': ['visit_age'] # Compare children of the same age
    }
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

## API Reference

### `biv.detect(dataframe, methods, patient_id_col='patient_id', age_col='age', sex_col='sex', weight_col='weight_kg', height_col='height_cm', flag_suffix='_biv_flag')`

Identifies BIVs and returns a DataFrame with added boolean flag columns.

- **`dataframe`** (pd.DataFrame): The input data.
    
- **`methods`** (dict): A dictionary defining the detection methods and their parameters.
    
    - **`range`** (dict): Keys are column names (`weight_col`, `height_col`) and values are dictionaries with `'min'` and `'max'` keys.
        
    - **`zscore`** (dict): Defines z-score parameters.
        
        - `threshold` (float): The number of standard deviations from the mean to use as a cutoff.
            
        - `group_by` (list): A list of column names (e.g., `['sex', 'age']`) to group by before calculating z-scores. This ensures values are compared to their demographic peers.
            
- **`..._col`** (str): Column name identifiers for patient ID, age, sex, weight, and height.
    
- **`flag_suffix`** (str): The suffix for the new boolean flag columns.
    

**Returns**: A `pandas.DataFrame` with added flag columns indicating BIVs.

### `biv.remove(dataframe, methods, ...)`

Identifies and removes BIVs, replacing them with `numpy.nan`. It accepts the same parameters as `biv.detect()`.

**Returns**: A `pandas.DataFrame` with BIVs replaced by `NaN`.


## Contributing

Contributions are welcome! Please feel free to submit an issue or pull request to the [GitHub repository](https://github.com/seouri/biv).


## License

This project is licensed under the MIT License. See the `LICENSE` file for details.


## Contact

For questions or support, please open an [issue on GitHub](https://github.com/seouri/biv/issues).