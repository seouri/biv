"""
Error marking controls and metrics display.
"""

import streamlit as st
import pandas as pd
from typing import Set, Optional, List
from src.utils.calculations import get_point_metrics
from src.utils.persistence import save_error_labels
from src.utils.state_manager import (
    add_error_index,
    remove_error_index,
    is_error,
    set_point_comment,
    get_point_comment,
    get_general_comment,
    set_general_comment,
    set_patient_complete,
    set_selected_point_index,
)
from src.utils.visit_helpers import compute_height_visit_metadata


def render_selected_point_metrics(
    patient_data: pd.DataFrame,
    selected_index: int
) -> None:
    """
    Display detailed metrics for the selected point.
    
    Parameters
    ----------
    patient_data : pd.DataFrame
        Patient data with calculated fields
    selected_index : int
        Index of selected point
    """
    valid_indices, _missing_indices, visit_number_map = compute_height_visit_metadata(patient_data)
    if selected_index is None or selected_index not in visit_number_map:
        st.info("👆 Select a point with a recorded height measurement to view details")
        return
    
    metrics = get_point_metrics(patient_data, selected_index)
    visit_number = visit_number_map[selected_index]
    
    # Header
    is_error_point = is_error(selected_index)
    status_icon = "♦️" if is_error_point else "✓"
    status_text = "ERROR" if is_error_point else "Normal"

    st.markdown(f"### {status_icon} Visit #{visit_number} - {status_text}")
    
    # Age and Height Metrics
    st.markdown("#### 📏 Height Measurement")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Age",
            f"{metrics['age_years']:.2f} years",
            help=f"{metrics['age_days']:.0f} days"
        )
    
    with col2:
        st.metric(
            "Height",
            f"{metrics['height']:.2f} in"
        )
    
    with col3:
        if metrics['height_zscore'] is not None:
            zscore_val = metrics['height_zscore']
            zscore_color = "normal" if abs(zscore_val) < 3 else "inverse"
            st.metric(
                "Height Z-Score",
                f"{zscore_val:.2f}",
                delta=None,
                delta_color=zscore_color
            )
        else:
            st.metric("Height Z-Score", "N/A")
    
    # Velocity and Change Metrics (if available)
    if visit_number > 1 and metrics['velocity'] is not None:
        st.markdown("#### 📈 Growth Velocity")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Velocity",
                f"{metrics['velocity']:.2f} in/yr"
            )
        
        with col2:
            abs_change = metrics['absolute_change']
            pct_change = metrics['percent_change']
            st.metric(
                "Change from Previous",
                f"{abs_change:+.2f} in",
                f"{pct_change:+.1f}%"
            )
        
        with col3:
            if metrics['velocity_zscore'] is not None:
                vel_zscore = metrics['velocity_zscore']
                vel_color = "normal" if abs(vel_zscore) < 3 else "inverse"
                st.metric(
                    "Velocity Z-Score",
                    f"{vel_zscore:.2f}",
                    delta=None,
                    delta_color=vel_color
                )
            else:
                st.metric("Velocity Z-Score", "N/A")
        
        # Days since previous
        if metrics['days_since_previous'] is not None:
            st.caption(f"⏱️ {metrics['days_since_previous']:.0f} days since previous visit")
    else:
        st.info("？ No velocity data (first recorded height measurement)")
    
    # Additional measurements if available
    additional_metrics = []
    if 'weight_oz' in metrics and metrics['weight_oz'] is not None:
        additional_metrics.append(f"Weight: {metrics['weight_oz']:.1f} oz")
    if 'bmi' in metrics and metrics['bmi'] is not None:
        additional_metrics.append(f"BMI: {metrics['bmi']:.2f}")
    if 'head_circ_cm' in metrics and metrics['head_circ_cm'] is not None:
        additional_metrics.append(f"Head Circ: {metrics['head_circ_cm']:.1f} cm")
    if 'encounter_type' in metrics and metrics['encounter_type']:
        additional_metrics.append(f"Type: {metrics['encounter_type']}")
    
    if additional_metrics:
        st.markdown("**Additional Info:** " + " | ".join(additional_metrics))


def render_error_controls(
    patient_id: str,
    patient_data: pd.DataFrame,
    selected_index: Optional[int],
    error_indices: Set[int]
) -> None:
    """
    Render controls for marking/unmarking errors and adding comments.
    
    Parameters
    ----------
    patient_id : str
        Patient identifier
    patient_data : pd.DataFrame
        Patient data
    selected_index : int, optional
        Currently selected point index
    error_indices : Set[int]
        Set of error indices
    """
    st.markdown("---")
    
    valid_indices, _missing_indices, visit_number_map = compute_height_visit_metadata(patient_data)
    if not valid_indices:
        st.info("No height measurements available to review.")
        return

    if selected_index is None or selected_index not in visit_number_map:
        st.info("👆 Select a measurement with a height value to mark it as an error")
        return

    visit_number = visit_number_map[selected_index]
    valid_index_positions = {idx: pos for pos, idx in enumerate(valid_indices)}
    current_position = valid_index_positions.get(selected_index, 0)

    st.markdown("#### 🏷️ Mark as Error")
    
    is_error_point = is_error(selected_index)
    
    # Comment text area
    current_comment = get_point_comment(selected_index)
    comment = st.text_area(
        f"Comment for Visit #{visit_number}",
        value=current_comment,
        placeholder="Optional: Add a note about why this measurement is an error...",
        height=80,
        key=f"point_comment_{selected_index}"
    )
    
    # Update comment if changed and persist immediately for existing errors
    if comment != current_comment:
        set_point_comment(selected_index, comment)
        if is_error_point:
            _save_current_state(patient_id, error_indices)
    
    # Toggle error button with direct save behaviour
    if is_error_point:
        if st.button(
            "✓ Unmark as Error", 
            type="secondary", 
            use_container_width=True,
            help="Remove error flag from this measurement"
        ):
            remove_error_index(selected_index)
            st.success(f"✓ Visit #{visit_number} unmarked as error")
            # Save immediately
            _save_current_state(patient_id, error_indices - {selected_index})
            st.rerun()
    else:
        if st.button(
            "♦️ Mark as Error", 
            type="primary", 
            use_container_width=True,
            help="Flag this measurement as an error"
        ):
            # Persist the latest comment along with the error flag
            set_point_comment(selected_index, comment)
            add_error_index(selected_index)
            st.success(f"♦️ Visit #{visit_number} marked as error")
            # Save immediately
            _save_current_state(patient_id, error_indices | {selected_index})
            
            # Auto-advance to next valid point
            if valid_indices:
                next_position = (current_position + 1) % len(valid_indices)
                next_index = valid_indices[next_position]
                set_selected_point_index(int(next_index))
            
            st.rerun()


def render_patient_completion_controls(
    patient_id: str,
    error_indices: Set[int],
    patient_list: Optional[List[str]] = None
) -> None:
    """
    Render controls for general patient comments and completion.
    
    Parameters
    ----------
    patient_id : str
        Patient identifier
    error_indices : Set[int]
        Set of error indices
    patient_list : list[str], optional
        Ordered list of patient IDs for automatic navigation
    """
    st.markdown("---")
    st.markdown("#### 📝 Patient Summary")
    
    # General comment
    general_comment = get_general_comment()
    new_general_comment = st.text_area(
        "General notes about this patient",
        value=general_comment,
        placeholder="Optional: Overall observations, patterns, or notes...",
        height=100,
        key="general_comment"
    )
    
    # Update if changed
    if new_general_comment != general_comment:
        set_general_comment(new_general_comment)
    
    # Summary stats
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Errors Marked", len(error_indices))
    with col2:
        comments_count = sum(1 for i in error_indices if get_point_comment(i).strip())
        st.metric("Errors with Comments", comments_count)
    
    # Completion button with immediate save
    st.markdown("---")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("**Mark this patient as reviewed and complete?**")
        st.caption("You can always come back and make changes later.")

    with col2:
        if st.button(
            "✅ Complete Review",
            type="primary",
            width="stretch",
            help="Mark this patient's review as complete"
        ):
            # Persist current state and mark patient as complete
            _save_current_state(patient_id, error_indices, is_complete=True)
            set_patient_complete(patient_id, True)

            next_patient_id = _get_next_patient_id(patient_id, patient_list)
            if next_patient_id:
                st.session_state["pending_patient_id"] = next_patient_id

            st.success(f"Patient {patient_id} marked complete.")
            st.rerun()


def _save_current_state(
    patient_id: str,
    error_indices: Set[int],
    is_complete: bool = False
) -> None:
    """
    Helper function to save current state to disk.
    
    Parameters
    ----------
    patient_id : str
        Patient identifier
    error_indices : Set[int]
        Set of error indices (optional, will use session state if not accurate)
    is_complete : bool
        Whether review is complete
    """
    from src.utils.state_manager import get_point_comment, get_general_comment, get_error_indices
    from src.auth import get_current_user
    
    # Always get the latest error indices from session state to ensure we have the most current data
    current_error_indices = get_error_indices()
    
    # Collect all comments
    point_comments = {}
    for idx in range(1000):  # Arbitrary large number
        comment = get_point_comment(idx)
        if comment.strip():
            point_comments[idx] = comment
    
    general_comment = get_general_comment()
    username = get_current_user()
    
    # Save to disk
    save_error_labels(
        patient_id=patient_id,
        error_indices=current_error_indices,
        point_comments=point_comments,
        general_comment=general_comment,
        is_complete=is_complete,
        username=username
    )


def _get_next_patient_id(
    current_patient_id: str,
    patient_list: Optional[List[str]]
) -> Optional[str]:
    """Determine the next patient in the list, wrapping to the start."""
    if not patient_list:
        return None

    try:
        current_index = patient_list.index(current_patient_id)
    except ValueError:
        return patient_list[0] if patient_list else None

    next_index = (current_index + 1) % len(patient_list)
    return patient_list[next_index]
