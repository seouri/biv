"""
Utility functions for velocity and metric calculations.
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple
from src.config import DAYS_PER_YEAR
from src.data.growth_standards import (
    calculate_height_for_age_zscore,
    calculate_velocity_for_age_zscore,
    _load_velocity_standards,
)


def detect_peak_errors(
    df: pd.DataFrame,
    patient_col: str = "patient_id",
    age_col: str = "age_in_days",
    height_col: str = "height_in",
) -> pd.DataFrame:
    """
    Adds a 'peak_error_flag' column to the DataFrame marking suspected erroneous height
    measurements based on 3-point sliding windows per patient.

    Logic:
    - Peak patterns: increase->decrease, flat->decrease
    - Trough patterns: decrease->increase, decrease->flat, decrease->decrease

    On detecting a peak:
    - If third > first: mark second as error
    - Else: mark third as error

    On detecting a trough:
    - If third > first: mark second as error
    - Else: mark first as error

    Windows always use the three most recent non-error, non-null points.
    """
    # Ensure a flag column exists
    df = df.copy()
    df["peak_error_flag"] = False

    # Process each patient independently
    for _, group in df.groupby(patient_col):
        # Sort by age
        group_idx = group.sort_values(age_col).index.tolist()
        valid_stack: List[int] = []

        for idx in group_idx:
            height = df.at[idx, height_col]
            # Skip null heights
            if pd.isna(height):
                continue

            valid_stack.append(idx)

            if len(valid_stack) < 3:
                continue

            i1, i2, i3 = valid_stack[-3:]
            h1, h2, h3 = (
                df.at[i1, height_col],
                df.at[i2, height_col],
                df.at[i3, height_col],
            )

            def trend(a, b):
                if b > a:
                    return "increase"
                elif b < a:
                    return "decrease"
                else:
                    return "flat"

            t12 = trend(h1, h2)
            t23 = trend(h2, h3)

            # Identify peak or trough patterns
            is_peak = (t12 == "increase" and t23 == "decrease") or (
                t12 == "flat" and t23 == "decrease"
            )
            is_trough = (
                (t12 == "decrease" and t23 == "increase")
                or (t12 == "decrease" and t23 == "flat")
                or (t12 == "decrease" and t23 == "decrease")
            )

            flagged_idx = None
            if is_peak:
                # Peak: compare third vs first
                flagged_idx = i2 if (h3 > h1) else i3
            elif is_trough:
                # Trough: compare third vs first
                flagged_idx = i2 if (h3 > h1) else i1

            if flagged_idx is not None:
                df.at[flagged_idx, "peak_error_flag"] = True
                valid_stack = [i for i in valid_stack if i != flagged_idx]

    return df

def _get_min_interval_days(age_in_days):
    """
    Returns the minimal intervals (in days) for calculating growth velocity based on age.
    Based on US pediatric guidelines for weight and height velocity calculations.

    Parameters:
        age_in_days (int): Age of the patient in days.

    Returns:
        dict: Dictionary with 'weight' and 'height' keys containing minimum interval days for each measurement type.
    """
    if age_in_days <= 365:  # Infants (0–12 months)
        return {"weight": 30, "height": 90}
    elif age_in_days <= 730:  # Toddlers (1–2 years)
        return {"weight": 90, "height": 180}
    elif age_in_days <= 1825:  # Early Childhood (2–5 years)
        return {"weight": 180, "height": 335}
    elif age_in_days <= 4380:  # School-Age (6–12 years)
        return {"weight": 180, "height": 335}
    else:  # Adolescents (13–18 years)
        return {"weight": 180, "height": 180}


def calculate_growth_velocities_v2(visits: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate growth velocities for weight and height with age-appropriate minimum intervals.
    Replaces simple consecutive-point calculations with clinically appropriate intervals.
    
    Parameters
    ----------
    visits : pd.DataFrame
        Patient visits data. Must contain 'patient_id', 'age_in_days', 'sex' columns.
        Should contain 'weight_kg', 'height_cm', or 'height_in' for calculations.
        
    Returns
    -------
    pd.DataFrame
        DataFrame with added velocity columns
    """
    visits = visits.copy()
    
    velocity_standards = _load_velocity_standards()

    # Initialize columns if they don't exist
    for col in [
        "delta_weight_kg",
        "delta_age_in_days_weight",
        "weight_velocity",
        "delta_height_cm",
        "delta_age_in_days_height",
        "height_velocity",
        "height_velocity_in_per_day",
        "height_velocity_z",
        "high_confidence_error",
        "height_adjacent_velocity",
        "height_next_adjacent_velocity",
    ]:
        if col not in visits.columns:
            visits[col] = np.nan

    if "peak_error_flag" not in visits.columns:
        visits["peak_error_flag"] = False
    else:
        visits["peak_error_flag"] = visits["peak_error_flag"].fillna(False).astype(bool)

    visits["high_confidence_error"] = visits["high_confidence_error"].fillna(False).astype(bool)

    def _height_velocity_z(sex: str, age_days: float, velocity_in_per_day: float):
        if pd.isna(velocity_in_per_day) or pd.isna(age_days) or sex not in ("F", "M"):
            return np.nan
        age_months = age_days / 30.4375
        std = velocity_standards[sex]
        ages = std["Agemos"].values
        idx = int(np.abs(ages - age_months).argmin())
        row = std.iloc[idx]
        L, Mv, Sv = row["L"], row["M"], row["S"]
        if any(pd.isna([L, Mv, Sv])) or Mv <= 0 or Sv <= 0:
            return np.nan
        if abs(L) < 1e-8:
            z = np.log(velocity_in_per_day / Mv) / Sv
        else:
            z = ((velocity_in_per_day / Mv) ** L - 1) / (L * Sv)
        return float(z)

    def _calc_for_measure(patient_data: pd.DataFrame, measure_col: str,
                          delta_col: str, age_delta_col: str, velocity_col: str,
                          measure_type: str):
        cols_needed = [measure_col, "age_in_days"]
        valid_data = patient_data[cols_needed].dropna().copy()
        if len(valid_data) < 2:
            return

        valid_data = valid_data.sort_values("age_in_days")
        ages = valid_data["age_in_days"].values.astype(int)
        measurements = valid_data[measure_col].values.astype(float)

        # Pre-compute adjacent velocities (previous -> current) and (current -> next) for height in inches/day
        if measure_type == "height":
            # Prefer direct height_in if available for clarity in units
            if "height_in" in patient_data.columns:
                heights_in = patient_data.loc[valid_data.index, "height_in"].astype(float).values
            else:
                # Fallback convert from cm
                heights_in = measurements / 2.54  # since measurements is height_cm

            n_local = len(ages)
            backward = np.full(n_local, np.nan)
            forward = np.full(n_local, np.nan)
            for k in range(1, n_local):
                dt = ages[k] - ages[k - 1]
                if dt > 0:
                    backward[k] = (heights_in[k] - heights_in[k - 1]) / dt
            for k in range(0, n_local - 1):
                dt = ages[k + 1] - ages[k]
                if dt > 0:
                    forward[k] = (heights_in[k + 1] - heights_in[k]) / dt
            # Assign to visits DataFrame (these are already inches/day)
            visits.loc[valid_data.index, "height_adjacent_velocity"] = backward
            visits.loc[valid_data.index, "height_next_adjacent_velocity"] = forward

        n = len(valid_data)
        for i in range(1, n):
            current_age = ages[i]
            current_measure = measurements[i]
            idx_current = valid_data.index[i]

            # search backward for prior j meeting min gap and not high_confidence_error
            min_gap = _get_min_interval_days(current_age)[measure_type]
            chosen_j = None
            for j in range(i - 1, -1, -1):
                prev_age = ages[j]
                idx_prev = valid_data.index[j]
                if visits.at[idx_prev, "high_confidence_error"]:
                    continue  # skip flagged prior points
                age_diff = current_age - prev_age
                if age_diff >= min_gap:
                    chosen_j = j
                    break

            if chosen_j is None:
                continue

            prev_measure = measurements[chosen_j]
            age_diff = current_age - ages[chosen_j]
            measure_diff = current_measure - prev_measure
            velocity_per_year = round(measure_diff / age_diff * 365, 2)

            visits.loc[idx_current, delta_col] = round(measure_diff, 2)
            visits.loc[idx_current, age_delta_col] = int(age_diff)
            visits.loc[idx_current, velocity_col] = velocity_per_year

            if measure_type == "height":
                v_in_per_day = velocity_per_year * 0.393701 / 365.25
                visits.loc[idx_current, "height_velocity_in_per_day"] = v_in_per_day
                sex = visits.at[idx_current, "sex"] if "sex" in visits.columns else None
                z = _height_velocity_z(sex, float(current_age), float(v_in_per_day))
                visits.loc[idx_current, "height_velocity_z"] = z
                # Mark HCE inline if criteria met
                if pd.notna(z) and abs(z) > 5 and bool(visits.at[idx_current, "peak_error_flag"]):
                    visits.loc[idx_current, "high_confidence_error"] = True

    # Group by patient and calculate velocities
    if 'patient_id' in visits.columns:
        for _, group in visits.groupby("patient_id"):
            if 'weight_kg' in visits.columns:
                _calc_for_measure(group, "weight_kg", "delta_weight_kg", "delta_age_in_days_weight", "weight_velocity", "weight")
            if 'height_cm' in visits.columns:
                _calc_for_measure(group, "height_cm", "delta_height_cm", "delta_age_in_days_height", "height_velocity", "height")
    else:
        # Single patient mode - treat entire dataframe as one patient
        if 'weight_kg' in visits.columns:
            _calc_for_measure(visits, "weight_kg", "delta_weight_kg", "delta_age_in_days_weight", "weight_velocity", "weight")
        if 'height_cm' in visits.columns:
            _calc_for_measure(visits, "height_cm", "delta_height_cm", "delta_age_in_days_height", "height_velocity", "height")

    # Fix dtypes for age delta columns
    if 'delta_age_in_days_weight' in visits.columns:
        visits["delta_age_in_days_weight"] = visits["delta_age_in_days_weight"].astype("Int16")
    if 'delta_age_in_days_height' in visits.columns:
        visits["delta_age_in_days_height"] = visits["delta_age_in_days_height"].astype("Int16")

    return visits


def calculate_bmi(
    height_in: float,
    weight_oz: float
) -> Optional[float]:
    """
    Calculate BMI from height in inches and weight in ounces.
    
    Parameters
    ----------
    height_in : float
        Height in inches
    weight_oz : float
        Weight in ounces
        
    Returns
    -------
    float or None
        BMI value, or None if inputs are invalid
        
    Notes
    -----
    BMI = (weight_lb / height_in^2) * 703
    """
    if pd.isna(height_in) or pd.isna(weight_oz) or height_in <= 0 or weight_oz <= 0:
        return None
    
    weight_lb = weight_oz / 16.0  # Convert ounces to pounds
    bmi = (weight_lb / (height_in ** 2)) * 703
    
    return round(bmi, 2)


def calculate_age_in_years(age_days: float) -> float:
    """
    Convert age from days to years.
    
    Parameters
    ----------
    age_days : float
        Age in days
        
    Returns
    -------
    float
        Age in years (rounded to 2 decimal places)
    """
    return round(age_days / DAYS_PER_YEAR, 2)


def add_calculated_fields(
    df: pd.DataFrame,
    sex: str
) -> pd.DataFrame:
    """
    Add calculated fields to patient data DataFrame.
    Uses age-appropriate minimum intervals for velocity calculations.
    
    Parameters
    ----------
    df : pd.DataFrame
        Patient data with at least 'age_in_days', 'height_in' columns
    sex : str
        Patient sex for z-score calculations
        
    Returns
    -------
    pd.DataFrame
        DataFrame with additional calculated columns:
        - age_years
        - velocity (inches/day)
        - absolute_change
        - percent_change
        - days_since_previous
        - height_zscore
        - velocity_zscore
        - bmi (if weight_oz column exists)
    """
    df = df.copy()
    
    # Ensure data is sorted by age
    df = df.sort_values('age_in_days').reset_index(drop=True)
    
    # Add sex column for velocity calculation if not present
    if 'sex' not in df.columns:
        df['sex'] = sex
    
    # Age in years
    df['age_years'] = df['age_in_days'].apply(calculate_age_in_years)
    
    # Convert height from inches to cm for the new velocity calculation
    if 'height_cm' not in df.columns and 'height_in' in df.columns:
        df['height_cm'] = df['height_in'] * 2.54
    
    # Use new velocity calculation with age-appropriate intervals
    df = detect_peak_errors(df)
    df = calculate_growth_velocities_v2(df)
    
    # The new function creates height_velocity_in_per_day (inches/day)
    # Rename to 'velocity' for consistency with existing code
    if 'height_velocity_in_per_day' in df.columns:
        df['velocity'] = df['height_velocity_in_per_day']
    else:
        df['velocity'] = np.nan
    
    # The new function creates height_velocity_z
    # Rename to 'velocity_zscore' for consistency
    if 'height_velocity_z' in df.columns:
        df['velocity_zscore'] = df['height_velocity_z']
    else:
        df['velocity_zscore'] = np.nan
    
    # Calculate simple consecutive differences for absolute_change and percent_change
    df['absolute_change'] = df['height_in'].diff()
    df['percent_change'] = (df['absolute_change'] / df['height_in'].shift(1)) * 100
    df['days_since_previous'] = df['age_in_days'].diff()
    
    # Height-for-age z-scores
    df['height_zscore'] = df.apply(
        lambda row: calculate_height_for_age_zscore(
            row['height_in'],
            row['age_in_days'],
            sex
        ),
        axis=1
    )
    
    # Calculate BMI if weight is available
    if 'weight_oz' in df.columns:
        df['bmi'] = df.apply(
            lambda row: calculate_bmi(row['height_in'], row['weight_oz'])
            if 'weight_oz' in row and pd.notna(row['weight_oz']) else None,
            axis=1
        )
    
    return df


def get_point_metrics(
    df: pd.DataFrame,
    index: int
) -> dict:
    """
    Get all metrics for a specific point/measurement.
    
    Parameters
    ----------
    df : pd.DataFrame
        Patient data with calculated fields
    index : int
        Row index of the point
        
    Returns
    -------
    dict
        Dictionary containing all metrics for the point
    """
    if index < 0 or index >= len(df):
        return {}
    
    row = df.iloc[index]
    
    metrics = {
        'age_days': row['age_in_days'],
        'age_years': row.get('age_years', calculate_age_in_years(row['age_in_days'])),
        'height': row['height_in'],
        'height_zscore': row.get('height_zscore', None),
        'velocity': row.get('velocity', None),
        'velocity_zscore': row.get('velocity_zscore', None),
        'absolute_change': row.get('absolute_change', None),
        'percent_change': row.get('percent_change', None),
        'days_since_previous': row.get('days_since_previous', None),
    }
    
    # Add optional fields if available
    if 'weight_oz' in row:
        metrics['weight_oz'] = row['weight_oz']
    if 'bmi' in row:
        metrics['bmi'] = row['bmi']
    if 'head_circ_cm' in row:
        metrics['head_circ_cm'] = row['head_circ_cm']
    if 'encounter_type' in row:
        metrics['encounter_type'] = row['encounter_type']
    
    return metrics
