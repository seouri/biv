"""
Data processing utilities for patient data.
"""

import pandas as pd
from typing import Optional


def filter_data_by_age_range(
    df: pd.DataFrame,
    min_age_days: Optional[float] = None,
    max_age_days: Optional[float] = None
) -> pd.DataFrame:
    """
    Filter patient data by age range.
    
    Parameters
    ----------
    df : pd.DataFrame
        Patient data
    min_age_days : float, optional
        Minimum age in days (inclusive)
    max_age_days : float, optional
        Maximum age in days (inclusive)
        
    Returns
    -------
    pd.DataFrame
        Filtered data
    """
    filtered = df.copy()
    
    if min_age_days is not None:
        filtered = filtered[filtered['age_in_days'] >= min_age_days]
    
    if max_age_days is not None:
        filtered = filtered[filtered['age_in_days'] <= max_age_days]
    
    return filtered


def get_summary_statistics(df: pd.DataFrame) -> dict:
    """
    Calculate summary statistics for patient data.
    
    Parameters
    ----------
    df : pd.DataFrame
        Patient data
        
    Returns
    -------
    dict
        Dictionary of summary statistics
    """
    if df.empty:
        return {}
    
    stats = {
        'num_measurements': len(df),
        'age_range_days': (df['age_in_days'].min(), df['age_in_days'].max()),
        'age_range_years': (
            df['age_in_days'].min() / 365.25,
            df['age_in_days'].max() / 365.25
        ),
        'height_range': (df['height_in'].min(), df['height_in'].max()),
        'mean_height': df['height_in'].mean(),
        'std_height': df['height_in'].std(),
    }
    
    # Add velocity stats if available
    if 'velocity' in df.columns:
        velocity_data = df['velocity'].dropna()
        if not velocity_data.empty:
            stats['mean_velocity'] = velocity_data.mean()
            stats['std_velocity'] = velocity_data.std()
            stats['max_velocity'] = velocity_data.max()
            stats['min_velocity'] = velocity_data.min()
    
    return stats


def detect_potential_errors(
    df: pd.DataFrame,
    zscore_threshold: float = 5.0
) -> list:
    """
    Detect potential errors based on z-score threshold.
    
    Parameters
    ----------
    df : pd.DataFrame
        Patient data with height_zscore column
    zscore_threshold : float, default 5.0
        Z-score threshold for error detection
        
    Returns
    -------
    list
        List of indices that exceed the z-score threshold
    """
    if 'height_zscore' not in df.columns:
        return []
    
    potential_errors = df[
        abs(df['height_zscore']) > zscore_threshold
    ].index.tolist()
    
    return potential_errors


def validate_patient_data(df: pd.DataFrame) -> tuple[bool, list]:
    """
    Validate patient data for required fields and data quality.
    
    Parameters
    ----------
    df : pd.DataFrame
        Patient data
        
    Returns
    -------
    tuple
        (is_valid: bool, errors: list of error messages)
    """
    errors = []
    
    # Check for empty data
    if df.empty:
        errors.append("Data is empty")
        return False, errors
    
    # Check required columns
    required = ['patient_id', 'age_in_days', 'height_in', 'sex']
    missing_cols = set(required) - set(df.columns)
    if missing_cols:
        errors.append(f"Missing required columns: {missing_cols}")
    
    # Check for null values in required columns
    for col in required:
        if col in df.columns and df[col].isnull().any():
            null_count = df[col].isnull().sum()
            errors.append(f"Column '{col}' has {null_count} null values")
    
    # Check for negative ages
    if 'age_in_days' in df.columns and (df['age_in_days'] < 0).any():
        errors.append("Negative ages found in data")
    
    # Check for unrealistic heights
    if 'height_in' in df.columns:
        if (df['height_in'] < 10).any():
            errors.append("Heights below 10 inches found (possibly invalid)")
        if (df['height_in'] > 100).any():
            errors.append("Heights above 100 inches found (possibly invalid)")
    
    # Check for non-monotonic ages (should increase)
    if 'age_in_days' in df.columns:
        ages = df['age_in_days'].values
        if not all(ages[i] <= ages[i+1] for i in range(len(ages)-1)):
            errors.append("Ages are not in chronological order")
    
    is_valid = len(errors) == 0
    return is_valid, errors
