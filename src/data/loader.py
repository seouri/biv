"""
Data loading utilities for patient height measurements.
"""

import pandas as pd
import streamlit as st
from pathlib import Path
from typing import List, Optional, Dict
from src.config import RAW_DATA_DIR, REQUIRED_COLUMNS, METADATA_COLUMNS
from src.utils.calculations import add_calculated_fields


@st.cache_data
def load_patient_data(data_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load all patient data from CSV file(s).
    
    Parameters
    ----------
    data_path : Path, optional
        Path to CSV file or directory containing CSV files.
        If None, uses RAW_DATA_DIR from config.
        
    Returns
    -------
    pd.DataFrame
        Combined patient data with columns including:
        patient_id, age_in_days, height_in, sex, and optional fields
        
    Notes
    -----
    Expected CSV format:
    - patient_id: Unique patient identifier
    - age_in_days: Age at measurement (days)
    - height_in: Height in inches
    - sex: M/F or Male/Female
    - weight_oz: Weight in ounces (optional)
    - head_circ_cm: Head circumference in cm (optional)
    - bmi: BMI (optional, calculated if missing)
    - encounter_type: Type of visit (optional)
    - race: Patient race (optional)
    - ethnicity: Patient ethnicity (optional)
    """
    # if data_path is None:
    #     data_path = RAW_DATA_DIR
    
    # # Handle single file vs directory
    # if Path(data_path).is_file():
    #     df = pd.read_csv(data_path)
    # else:
    #     # Load all CSV files from directory
    #     csv_files = list(Path(data_path).glob("*.csv"))
        
    #     if not csv_files:
    #         # Return empty DataFrame with required columns for development
    #         return pd.DataFrame(columns=REQUIRED_COLUMNS)
        
    #     dfs = [pd.read_csv(f) for f in csv_files]
    #     df = pd.concat(dfs, ignore_index=True)

    df = pd.read_csv("data/raw/visits_60_patients.csv")
    
    # Validate required columns
    missing_cols = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Validate data types and values
    _validate_patient_data(df)
    
    # Standardize sex column
    df['sex'] = df['sex'].str.upper().str[0]  # Convert to 'M' or 'F'
    
    return df


def _validate_patient_data(df: pd.DataFrame) -> None:
    """
    Validate patient data structure and values.
    
    Parameters
    ----------
    df : pd.DataFrame
        Patient data to validate
        
    Raises
    ------
    ValueError
        If data validation fails
    """
    if df.empty:
        raise ValueError("Patient data is empty")
    
    # Check for required fields
    if 'patient_id' not in df.columns or df['patient_id'].isna().all():
        raise ValueError("Patient data is missing patient_id column")
    
    # Validate age values
    if 'age_in_days' in df.columns:
        if not pd.api.types.is_numeric_dtype(df['age_in_days']):
            raise ValueError("age_in_days must be numeric")
        
        if (df['age_in_days'] < 0).any():
            raise ValueError("age_in_days contains negative values")
        
        if (df['age_in_days'] > 36500).any():  # > 100 years
            raise ValueError("age_in_days contains unrealistic values (> 100 years)")
    
    # Validate height values
    if 'height_in' in df.columns:
        if not pd.api.types.is_numeric_dtype(df['height_in']):
            raise ValueError("height_in must be numeric")
        
        if (df['height_in'] <= 0).any():
            raise ValueError("height_in contains invalid values (<= 0)")
        
        if (df['height_in'] > 100).any():  # > 100 inches
            raise ValueError("height_in contains unrealistic values (> 100 inches)")
    
    # Validate sex values
    if 'sex' in df.columns:
        valid_sex_values = {'M', 'F', 'MALE', 'FEMALE', 'm', 'f', 'male', 'female'}
        invalid_sex = df[~df['sex'].isin(valid_sex_values) & df['sex'].notna()]
        if not invalid_sex.empty:
            invalid_values = invalid_sex['sex'].unique()
            raise ValueError(
                f"Invalid sex values found: {invalid_values}. "
                f"Must be one of: M, F, Male, Female"
            )


@st.cache_data
def get_patient_list(data: pd.DataFrame) -> List[str]:
    """
    Get sorted list of unique patient IDs.
    
    Parameters
    ----------
    data : pd.DataFrame
        Patient data
        
    Returns
    -------
    list
        Sorted list of patient IDs
    """
    if data.empty:
        return []
    
    return sorted(data['patient_id'].unique().tolist())


@st.cache_data
def load_single_patient(
    patient_id: str,
    data: pd.DataFrame
) -> pd.DataFrame:
    """
    Load and process data for a single patient.
    
    Parameters
    ----------
    patient_id : str
        Patient identifier
    data : pd.DataFrame
        Full patient dataset
        
    Returns
    -------
    pd.DataFrame
        Patient data with calculated fields (velocity, z-scores, etc.)
        Sorted by age_in_days
        
    Raises
    ------
    ValueError
        If patient ID not found or data is invalid
    """
    # Validate inputs
    if not patient_id:
        raise ValueError("Patient ID cannot be empty")
    
    if data is None or data.empty:
        raise ValueError("Patient data is empty")
    
    # Filter for this patient
    patient_data = data[data['patient_id'] == patient_id].copy()
    
    if patient_data.empty:
        raise ValueError(
            f"No data found for patient '{patient_id}'. "
            f"Available patients: {sorted(data['patient_id'].unique().tolist())[:5]}..."
        )
    
    # Validate required fields
    if 'sex' not in patient_data.columns or patient_data['sex'].isna().all():
        raise ValueError(f"Patient {patient_id} is missing sex information")
    
    # Get patient sex for z-score calculations
    sex = patient_data['sex'].iloc[0]
    
    if sex not in ['M', 'F']:
        raise ValueError(
            f"Invalid sex value '{sex}' for patient {patient_id}. Expected 'M' or 'F'"
        )
    
    # Check for minimum data
    if len(patient_data) < 1:
        raise ValueError(f"Patient {patient_id} has no measurements")
    
    # Sort by age
    patient_data = patient_data.sort_values('age_in_days').reset_index(drop=True)
    
    # Add calculated fields
    try:
        patient_data = add_calculated_fields(patient_data, sex)
    except Exception as e:
        raise ValueError(
            f"Error calculating fields for patient {patient_id}: {str(e)}"
        )
    
    return patient_data


def get_patient_metadata(patient_data: pd.DataFrame) -> Dict:
    """
    Extract patient metadata/demographics.
    
    Parameters
    ----------
    patient_data : pd.DataFrame
        Single patient's data
        
    Returns
    -------
    dict
        Dictionary containing:
        - patient_id
        - sex
        - race (if available)
        - ethnicity (if available)
        - num_visits
        - age_span (min to max age in years)
        - first_age_days
        - last_age_days
    """
    if patient_data.empty:
        return {}
    
    first_row = patient_data.iloc[0]
    
    metadata = {
        'patient_id': first_row['patient_id'],
        'sex': first_row['sex'],
        'num_visits': len(patient_data),
        'age_span': (
            patient_data['age_in_days'].min() / 365.25,
            patient_data['age_in_days'].max() / 365.25
        ),
        'first_age_days': patient_data['age_in_days'].min(),
        'last_age_days': patient_data['age_in_days'].max(),
    }
    
    # Add optional metadata if available
    for col in METADATA_COLUMNS:
        if col in patient_data.columns:
            metadata[col] = first_row.get(col, 'Unknown')
    
    return metadata


@st.cache_data
def create_sample_data(num_patients: int = 5, visits_per_patient: int = 45) -> pd.DataFrame:
    """
    Create sample patient data for development/testing.
    
    Parameters
    ----------
    num_patients : int, default 5
        Number of sample patients to generate
    visits_per_patient : int, default 45
        Number of visits per patient
        
    Returns
    -------
    pd.DataFrame
        Sample patient data
    """
    import numpy as np
    
    data = []
    
    for i in range(num_patients):
        patient_id = f"P{i+1:03d}"
        sex = 'M' if i % 2 == 0 else 'F'
        race = ['White', 'Black', 'Asian', 'Other'][i % 4]
        ethnicity = 'Hispanic' if i % 3 == 0 else 'Non-Hispanic'
        
        # Generate ages (roughly quarterly visits from birth to ~12 years)
        ages = np.linspace(30, 4380, visits_per_patient)  # 30 days to 12 years
        ages = ages + np.random.uniform(-10, 10, size=visits_per_patient)  # Add jitter
        ages = np.sort(ages)
        
        # Generate realistic height trajectory with some noise
        age_years = ages / 365.25
        
        # Generate realistic heights based on sex (in inches)
        # Using piecewise linear model with growth spurts for pediatric heights
        if sex == 'M':
            # Male: birth ~20", age 2 ~34", age 10 ~55", age 18 ~69"
            base_heights = np.where(
                age_years < 2,
                20 + (7 * age_years),  # Fast growth: 0-2 years
                np.where(
                    age_years < 12,
                    34 + (2.1 * (age_years - 2)),  # Steady growth: 2-12 years
                    55 + (2.3 * (age_years - 12))  # Adolescent growth: 12-18 years
                )
            )
        else:
            # Female: birth ~19.5", age 2 ~33", age 10 ~54", age 18 ~64"
            base_heights = np.where(
                age_years < 2,
                19.5 + (6.75 * age_years),  # Fast growth: 0-2 years
                np.where(
                    age_years < 10,
                    33 + (2.625 * (age_years - 2)),  # Steady growth: 2-10 years
                    54 + (1.25 * (age_years - 10))  # Slower adolescent: 10-18 years
                )
            )
        
        # Add realistic noise
        heights = base_heights + np.random.normal(0, 1.5, size=visits_per_patient)
        
        # Add 1-2 errors per patient (only if enough visits)
        if visits_per_patient > 10:
            error_indices = np.random.choice(range(5, visits_per_patient-5), size=2, replace=False)
            for idx in error_indices:
                heights[idx] += np.random.choice([-8, 8])  # Significant deviation
        elif visits_per_patient > 5:
            # For smaller datasets, add just 1 error
            error_idx = np.random.choice(range(2, visits_per_patient-2), size=1)[0]
            heights[error_idx] += np.random.choice([-8, 8])
        
        # Generate weights (rough approximation)
        weights_lb = (heights / 2) + np.random.uniform(-5, 5, size=visits_per_patient)
        weights_oz = weights_lb * 16
        
        for j in range(visits_per_patient):
            data.append({
                'patient_id': patient_id,
                'age_in_days': int(ages[j]),
                'height_in': round(heights[j], 2),
                'weight_oz': round(weights_oz[j], 1),
                'sex': sex,
                'race': race,
                'ethnicity': ethnicity,
                'encounter_type': np.random.choice(['Well-child', 'Sick', 'Follow-up']),
                'head_circ_cm': round(40 + (age_years[j] * 2) + np.random.uniform(-1, 1), 1),
            })
    
    return pd.DataFrame(data)
