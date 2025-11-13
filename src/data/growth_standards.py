"""
Functions for CDC/WHO growth standards.

Uses LMS (Lambda-Mu-Sigma) growth chart data.
The LMS method uses the Box-Cox transformation to normalize growth data.
"""

import numpy as np
import pandas as pd
import streamlit as st
from typing import Dict, Tuple
import os


# Global variables to cache loaded data
_WHO_DATA = None
_CDC_DATA = None
_VELOCITY_STANDARDS = None
_DATA_DIR = os.path.join(os.path.dirname(__file__), '../../data/growth_standard')


def _load_who_data() -> pd.DataFrame:
    """Load and preprocess WHO growth standard reference data."""
    global _WHO_DATA
    if _WHO_DATA is None:
        _WHO_DATA = pd.read_csv(os.path.join(_DATA_DIR, "who_growth_standards.csv"))
        _WHO_DATA.set_index(["measure", "sex", "Day"], inplace=True)
    return _WHO_DATA


def _load_cdc_data() -> Dict[str, pd.DataFrame]:
    """Load and preprocess CDC growth chart reference data."""
    global _CDC_DATA
    if _CDC_DATA is None:
        # Load height-for-age data
        hfa = pd.read_csv(os.path.join(_DATA_DIR, "statage_combined.csv"))
        hfa["Sex"] = hfa["Sex"].map({1: "M", 2: "F"})
        hfa["Agemos"] = hfa["Agemos"].astype("float")
        _CDC_DATA = {"height_for_age": hfa}
    return _CDC_DATA


def _calculate_lms_value(L: float, M: float, S: float, z: float) -> float:
    """
    Calculate value at given z-score using LMS method.
    Formula: M * (1 + L*S*z)^(1/L)
    """
    return M * ((1 + L * S * z) ** (1 / L))


def _create_velocity_standards() -> Dict[str, pd.DataFrame]:
    """
    Create velocity LMS standards (inches/day) from raw height LMS standards.
    
    Returns velocity standards for both Male and Female, combining WHO (0-5 years)
    and CDC (5+ years) data.
    """
    
    def _height_to_velocity_lms(df: pd.DataFrame, sex: str) -> pd.DataFrame:
        """Convert height LMS parameters to velocity LMS via numerical differentiation."""
        data = df.sort_values("Agemos").copy()
        ages = data["Agemos"].values
        Ls, Ms, Ss = data["L"].values, data["M"].values, data["S"].values

        def lms_to_value(L, M, S, z):
            """Calculate height values at given z-scores."""
            return np.where(np.abs(L) > 1e-10, M * np.power(1 + L * S * z, 1 / L), M * np.exp(S * z))

        # Calculate heights at different z-scores
        z_set = [-2, -1, 0, 1, 2]
        heights_by_z = [lms_to_value(Ls, Ms, Ss, z) for z in z_set]

        # Numerical differentiation to get velocity
        n = len(ages)
        CM_TO_IN = 0.393701
        velocities_z = []
        
        for h in heights_by_z:
            vel_cm_per_month = np.zeros_like(h)
            if n > 1:
                # First point (forward difference)
                dt = ages[1] - ages[0]
                if dt > 0:
                    vel_cm_per_month[0] = (h[1] - h[0]) / dt
                # Interior points (central difference)
                for i in range(1, n - 1):
                    dt = ages[i + 1] - ages[i - 1]
                    if dt > 0:
                        vel_cm_per_month[i] = (h[i + 1] - h[i - 1]) / dt
                # Last point (backward difference)
                dt = ages[-1] - ages[-2]
                if dt > 0:
                    vel_cm_per_month[-1] = (h[-1] - h[-2]) / dt
            
            # Convert cm/month -> inches/day
            vel_in_per_day = (vel_cm_per_month * CM_TO_IN) / 30.4375
            velocities_z.append(vel_in_per_day)

        # Extract velocities at key z-scores
        vel_median = velocities_z[2]  # z=0
        vel_plus1 = velocities_z[3]   # z=+1
        vel_minus1 = velocities_z[1]  # z=-1
        
        # Estimate S parameter for velocity
        with np.errstate(divide='ignore', invalid='ignore'):
            S_est = np.where(np.abs(vel_median) > 1e-8, 
                           np.abs((vel_plus1 - vel_minus1) / (2 * vel_median)), 
                           0.1)
        S_est = np.clip(S_est, 0.01, 0.5)
        M_vel = np.clip(vel_median, 0, None)

        # Create velocity LMS dataframe
        out = pd.DataFrame({
            "Agemos": ages,
            "L": 1.0,
            "M": M_vel,
            "S": S_est,
            "sex": sex,
        })

        # Add boundary curves (±5 SD) using LMS
        for label, z in [("minus5SD", -5), ("median", 0), ("plus5SD", 5)]:
            out[label] = np.where(np.abs(out["L"]) > 1e-10,
                                out["M"] * np.power(1 + out["L"] * out["S"] * z, 1 / out["L"]),
                                out["M"] * np.exp(out["S"] * z))
        return out

    def _combine_who_cdc(who_df: pd.DataFrame, cdc_df: pd.DataFrame, sex: str) -> pd.DataFrame:
        """Combine WHO (≤60 months) and CDC (>60 months) velocity standards."""
        who_part = who_df[who_df["Agemos"] <= 60].copy()
        cdc_part = cdc_df[cdc_df["Agemos"] > 60].copy()
        return pd.concat([who_part, cdc_part], ignore_index=True).sort_values("Agemos").reset_index(drop=True)

    # Load WHO and CDC data
    who_data = _load_who_data()
    cdc_data = _load_cdc_data()
    
    # Prepare WHO height data
    who_height = who_data.reset_index().query("measure == 'height'").copy()
    who_height['Agemos'] = who_height['Day'] / 30.4375
    who_f = who_height[who_height['sex'] == 'F'][['Agemos', 'L', 'M', 'S']].copy().sort_values('Agemos')
    who_m = who_height[who_height['sex'] == 'M'][['Agemos', 'L', 'M', 'S']].copy().sort_values('Agemos')

    # Prepare CDC height data
    cdc_height = cdc_data['height_for_age'].copy()
    cdc_f = cdc_height[cdc_height['Sex'] == 'F'][['Agemos', 'L', 'M', 'S']].copy().sort_values('Agemos')
    cdc_m = cdc_height[cdc_height['Sex'] == 'M'][['Agemos', 'L', 'M', 'S']].copy().sort_values('Agemos')

    # Convert height LMS to velocity LMS
    who_f_vel = _height_to_velocity_lms(who_f, "F")
    who_m_vel = _height_to_velocity_lms(who_m, "M")
    cdc_f_vel = _height_to_velocity_lms(cdc_f, "F")
    cdc_m_vel = _height_to_velocity_lms(cdc_m, "M")

    # Combine WHO and CDC for each sex
    return {
        "F": _combine_who_cdc(who_f_vel, cdc_f_vel, "F"),
        "M": _combine_who_cdc(who_m_vel, cdc_m_vel, "M"),
    }


def _load_velocity_standards() -> Dict[str, pd.DataFrame]:
    """Load or create velocity standards (cached globally)."""
    global _VELOCITY_STANDARDS
    if _VELOCITY_STANDARDS is None:
        _VELOCITY_STANDARDS = _create_velocity_standards()
    return _VELOCITY_STANDARDS


@st.cache_data
def get_height_zscore_bounds(
    age_days: np.ndarray, 
    sex: str
) -> Dict[str, np.ndarray]:
    """
    Get height z-score bounds for CDC/WHO growth standards.
    
    Uses WHO standards for ages < 1856 days (0-5 years) and 
    CDC standards for ages >= 1856 days (5+ years).
    
    Parameters
    ----------
    age_days : np.ndarray
        Array of ages in days
    sex : str
        Patient sex ('M' or 'F')
        
    Returns
    -------
    dict
        Dictionary containing:
        - 'age_days': Array of ages (same as input)
        - 'median': Median height (50th percentile) in inches
        - 'upper_bound': Height at +3 z-score in inches
        - 'lower_bound': Height at -3 z-score in inches
        - 'z_1': Height at +1 z-score in inches
        - 'z_minus_1': Height at -1 z-score in inches
    """
    who_data = _load_who_data()
    cdc_data = _load_cdc_data()
    
    # Normalize sex
    sex_code = sex.upper()[0] if sex else 'M'
    
    # Initialize result arrays
    median = np.zeros_like(age_days, dtype=float)
    upper_bound = np.zeros_like(age_days, dtype=float)
    lower_bound = np.zeros_like(age_days, dtype=float)
    z_1 = np.zeros_like(age_days, dtype=float)
    z_minus_1 = np.zeros_like(age_days, dtype=float)
    
    # Process each age
    for i, age in enumerate(age_days):
        if age < 1856:  # Use WHO data (0-5 years)
            try:
                # Get WHO data for this age and sex
                row = who_data.loc[("height", sex_code, int(age))]
                L, M, S = row["L"], row["M"], row["S"]
                
                # Convert from cm to inches
                median[i] = M / 2.54
                upper_bound[i] = _calculate_lms_value(L, M, S, 3) / 2.54
                lower_bound[i] = _calculate_lms_value(L, M, S, -3) / 2.54
                z_1[i] = _calculate_lms_value(L, M, S, 1) / 2.54
                z_minus_1[i] = _calculate_lms_value(L, M, S, -1) / 2.54
            except (KeyError, ValueError):
                # If exact day not found, use interpolation fallback
                median[i] = np.nan
                upper_bound[i] = np.nan
                lower_bound[i] = np.nan
                z_1[i] = np.nan
                z_minus_1[i] = np.nan
        else:  # Use CDC data (5+ years)
            try:
                # Convert age to months for CDC lookup
                age_months = age / 30.4375
                hfa_cdc = cdc_data["height_for_age"]
                hfa_cdc = hfa_cdc[hfa_cdc["Sex"] == sex_code].copy()
                
                # Find closest age month
                closest_idx = (hfa_cdc["Agemos"] - age_months).abs().idxmin()
                row = hfa_cdc.loc[closest_idx]
                
                L, M, S = row["L"], row["M"], row["S"]
                
                # Convert from cm to inches
                median[i] = M / 2.54
                upper_bound[i] = _calculate_lms_value(L, M, S, 3) / 2.54
                lower_bound[i] = _calculate_lms_value(L, M, S, -3) / 2.54
                z_1[i] = _calculate_lms_value(L, M, S, 1) / 2.54
                z_minus_1[i] = _calculate_lms_value(L, M, S, -1) / 2.54
            except (KeyError, ValueError, IndexError):
                median[i] = np.nan
                upper_bound[i] = np.nan
                lower_bound[i] = np.nan
                z_1[i] = np.nan
                z_minus_1[i] = np.nan
    
    # Interpolate any NaN values
    mask = ~np.isnan(median)
    if mask.any():
        median = np.interp(age_days, age_days[mask], median[mask])
        upper_bound = np.interp(age_days, age_days[mask], upper_bound[mask])
        lower_bound = np.interp(age_days, age_days[mask], lower_bound[mask])
        z_1 = np.interp(age_days, age_days[mask], z_1[mask])
        z_minus_1 = np.interp(age_days, age_days[mask], z_minus_1[mask])
    
    return {
        'age_days': age_days,
        'median': median,
        'upper_bound': upper_bound,
        'lower_bound': lower_bound,
        'z_1': z_1,
        'z_minus_1': z_minus_1,
    }


@st.cache_data
def get_velocity_zscore_bounds(
    age_days: np.ndarray, 
    sex: str
) -> Dict[str, np.ndarray]:
    """
    Get velocity z-score bounds for CDC/WHO growth standards.
    
    Uses velocity standards derived from WHO (0-5 years) and CDC (5+ years)
    height standards via numerical differentiation.
    
    Parameters
    ----------
    age_days : np.ndarray
        Array of ages in days
    sex : str
        Patient sex ('M' or 'F')
        
    Returns
    -------
    dict
        Dictionary containing:
        - 'age_days': Array of ages (same as input)
        - 'median': Median velocity (50th percentile) in inches/day
        - 'upper_bound': Velocity at +5 z-score in inches/day
        - 'lower_bound': Velocity at -5 z-score in inches/day
        
    Notes
    -----
    Velocity standards are computed by numerical differentiation of height
    standards and combined at 60 months (1826 days) transition point.
    """
    velocity_standards = _load_velocity_standards()
    
    # Normalize sex
    sex_code = sex.upper()[0] if sex else 'M'
    if sex_code not in ('M', 'F'):
        sex_code = 'M'
    
    # Get velocity standard for this sex
    std = velocity_standards[sex_code].copy()
    std["Day"] = std["Agemos"] * 30.4375
    
    # Initialize result arrays
    median = np.zeros_like(age_days, dtype=float)
    upper_bound = np.zeros_like(age_days, dtype=float)
    lower_bound = np.zeros_like(age_days, dtype=float)
    
    # Check if boundary columns exist
    if "minus5SD" not in std.columns or "plus5SD" not in std.columns:
        # Fallback to LMS calculation if precomputed columns missing
        for i, age in enumerate(age_days):
            age_months = age / 30.4375
            # Find closest age
            closest_idx = (std["Agemos"] - age_months).abs().idxmin()
            row = std.loc[closest_idx]
            
            L, M, S = row["L"], row["M"], row["S"]
            median[i] = M
            upper_bound[i] = M * ((1 + L * S * 5) ** (1 / L)) if abs(L) > 1e-10 else M * np.exp(S * 5)
            lower_bound[i] = M * ((1 + L * S * (-5)) ** (1 / L)) if abs(L) > 1e-10 else M * np.exp(S * (-5))
    else:
        # Use precomputed boundary columns (preferred)
        for i, age in enumerate(age_days):
            age_months = age / 30.4375
            # Find closest age
            closest_idx = (std["Agemos"] - age_months).abs().idxmin()
            row = std.loc[closest_idx]
            
            median[i] = row["median"] if "median" in row else row["M"]
            upper_bound[i] = row["plus5SD"]
            lower_bound[i] = row["minus5SD"]
    
    # Ensure non-negative velocities
    median = np.clip(median, 0, None)
    upper_bound = np.clip(upper_bound, 0, None)
    lower_bound = np.clip(lower_bound, 0, None)
    
    return {
        'age_days': age_days,
        'median': median,
        'upper_bound': upper_bound,
        'lower_bound': lower_bound,
    }


def calculate_height_for_age_zscore(
    height_in: float,
    age_days: float,
    sex: str
) -> float:
    """
    Calculate height-for-age z-score using LMS method.
    
    Parameters
    ----------
    height_in : float
        Height in inches
    age_days : float
        Age in days
    sex : str
        Patient sex ('M' or 'F')
        
    Returns
    -------
    float
        Height-for-age z-score
        
    Notes
    -----
    Uses LMS method: z = ((height/M)^L - 1) / (L*S)
    where L, M, S are parameters from CDC/WHO tables
    """
    who_data = _load_who_data()
    cdc_data = _load_cdc_data()
    
    # Normalize sex
    sex_code = sex.upper()[0] if sex else 'M'
    
    # Convert height to cm
    height_cm = height_in * 2.54
    
    try:
        if age_days < 1856:  # Use WHO data
            row = who_data.loc[("height", sex_code, int(age_days))]
            L, M, S = row["L"], row["M"], row["S"]
        else:  # Use CDC data
            age_months = age_days / 30.4375
            hfa_cdc = cdc_data["height_for_age"]
            hfa_cdc = hfa_cdc[hfa_cdc["Sex"] == sex_code].copy()
            closest_idx = (hfa_cdc["Agemos"] - age_months).abs().idxmin()
            row = hfa_cdc.loc[closest_idx]
            L, M, S = row["L"], row["M"], row["S"]
        
        # Calculate z-score using LMS method
        if L != 0:
            z_score = ((height_cm / M) ** L - 1) / (L * S)
        else:
            # For L = 0, use logarithmic transformation
            z_score = np.log(height_cm / M) / S
            
        return z_score
        
    except (KeyError, ValueError, IndexError):
        # Fallback to simple approximation if data not found
        bounds = get_height_zscore_bounds(np.array([age_days]), sex)
        median = bounds['median'][0]
        age_years = age_days / 365.25
        std_dev = 2.5 + (0.1 * age_years)
        z_score = (height_in - median) / std_dev
        return z_score


def calculate_velocity_for_age_zscore(
    velocity: float,
    age_days: float,
    sex: str
) -> float:
    """
    Calculate velocity-for-age z-score using LMS method.
    
    Parameters
    ----------
    velocity : float
        Velocity in inches/day
    age_days : float
        Age in days
    sex : str
        Patient sex ('M' or 'F')
        
    Returns
    -------
    float
        Velocity-for-age z-score
        
    Notes
    -----
    Uses LMS method: z = ((velocity/M)^L - 1) / (L*S)
    where L, M, S are velocity parameters derived from height standards
    """
    velocity_standards = _load_velocity_standards()
    
    # Normalize sex
    sex_code = sex.upper()[0] if sex else 'M'
    if sex_code not in ('M', 'F'):
        sex_code = 'M'
    
    try:
        # Get velocity standard for this sex
        std = velocity_standards[sex_code].copy()
        age_months = age_days / 30.4375
        
        # Find closest age
        closest_idx = (std["Agemos"] - age_months).abs().idxmin()
        row = std.loc[closest_idx]
        
        L, M, S = row["L"], row["M"], row["S"]
        
        # Calculate z-score using LMS method
        if L != 0 and M > 0:
            z_score = ((velocity / M) ** L - 1) / (L * S)
        elif M > 0:
            # For L = 0, use logarithmic transformation
            z_score = np.log(velocity / M) / S
        else:
            z_score = 0.0
            
        return z_score
        
    except (KeyError, ValueError, IndexError, ZeroDivisionError):
        # Fallback to simple approximation if data not found
        bounds = get_velocity_zscore_bounds(np.array([age_days]), sex)
        median = bounds['median'][0]
        std_dev = median * 0.3
        z_score = (velocity - median) / std_dev if std_dev > 0 else 0.0
        return z_score
