"""
Patient Information Display Component

This module provides the patient information header showing key demographics
and visit summary statistics.
"""

import streamlit as st
import pandas as pd


def render_patient_info(
    patient_data: pd.DataFrame,
    patient_id: str,
    is_complete: bool = False
) -> None:
    """
    Render patient information card with demographics and visit summary.
    
    Args:
        patient_data: DataFrame containing patient measurements
        patient_id: Current patient identifier
    """
    if patient_data.empty:
        st.warning(f"No data available for patient {patient_id}")
        return
    
    # Extract patient information from first row (same across all visits)
    first_row = patient_data.iloc[0]
    sex = first_row.get('sex', 'Unknown')
    race = first_row.get('race_1', 'Unknown')
    ethnicity = first_row.get('ethnicity', 'Unknown')
    
    # Calculate visit statistics
    n_visits = len(patient_data)
    ages_years = patient_data['age_in_days'] / 365.25
    min_age = ages_years.min()
    max_age = ages_years.max()
    age_span = f"{min_age:.1f} - {max_age:.1f} years"
    age_span_days = f"{patient_data['age_in_days'].min()} - {patient_data['age_in_days'].max()} days"
    
    status_icon = "✅" if is_complete else "⏳"
    status_label = "Complete" if is_complete else "In Progress"

    # Create attractive card layout
    st.markdown(
        f"""
        <div style="
            background-color: #f0f2f6;
            padding: 20px;
            border-radius: 10px;
            border-left: 5px solid #1f77b4;
            margin-bottom: 20px;
        ">
            <div style="display: flex; align-items: center; justify-content: space-between;">
                <h2 style="margin: 0; color: #1f77b4;">{status_icon} Patient {patient_id}</h2>
                <span style="font-weight: bold; color: {'#2ca02c' if is_complete else '#ff7f0e'};">
                    {status_label}
                </span>
            </div>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 15px;">
                <div>
                    <strong>Sex:</strong><br>{sex}
                </div>
                <div>
                    <strong>Race:</strong><br>{race}
                </div>
                <div>
                    <strong>Ethnicity:</strong><br>{ethnicity}
                </div>
                <div>
                    <strong>Visits:</strong><br>{n_visits} measurements
                </div>
                <div>
                    <strong>Age Range:</strong><br>{age_span_days} ({age_span})
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
