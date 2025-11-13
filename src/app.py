"""
Main Streamlit Application for Growth Error Labeling

This application provides an interactive interface for reviewing pediatric growth
measurements and marking errors.
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import streamlit as st
import pandas as pd

# Opt into pandas' future behavior to avoid silent dtype downcasting warnings
pd.set_option("future.no_silent_downcasting", True)
from typing import Optional

# Import components
from src.components.sidebar import render_sidebar
from src.components.patient_info import render_patient_info
from src.components.combined_chart import render_combined_charts
from src.components.data_table import render_data_table
from src.components.error_controls import (
    render_error_controls,
    render_patient_completion_controls
)

# Import utilities
from src.data.loader import load_patient_data, get_patient_list, load_single_patient
from src.utils.state_manager import (
    initialize_session_state,
    get_current_patient_id,
    set_current_patient_id,
    get_selected_point_index,
    set_selected_point_index,
    get_error_indices,
    get_all_point_comments,
    is_patient_complete,
    get_general_comment,
)
from src.utils.persistence import save_error_labels, save_all_labeled_data
from src.utils.visit_helpers import compute_height_visit_metadata

# Import config
from src.config import CHART_CONFIG, PROJECT_ROOT, STATE_KEYS



@st.cache_data
def load_all_data() -> pd.DataFrame:
    """Load and cache all patient data."""
    return load_patient_data()


def load_css() -> None:
    """Load custom CSS styles."""
    import os
    css_file = os.path.join(os.path.dirname(__file__), "styles", "custom.css")
    try:
        with open(css_file, "r") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        # CSS file is optional
        pass


def handle_patient_navigation(new_patient_id: str) -> None:
    """
    Handle patient navigation changes.
    
    Args:
        new_patient_id: The new patient ID to navigate to
    """
    if new_patient_id != get_current_patient_id():
        set_current_patient_id(new_patient_id)
        st.rerun()


def handle_plot_click(clicked_index: Optional[int]) -> None:
    """
    Handle point selection from plots.
    
    Args:
        clicked_index: Index of the clicked point, or None
    """
    current_selection = get_selected_point_index()
    if clicked_index != current_selection:
        set_selected_point_index(clicked_index)
        st.rerun()


def auto_save_labels(patient_id: str) -> None:
    """
    Auto-save error labels for current patient.
    
    Args:
        patient_id: Patient identifier
    """
    try:
        error_indices = get_error_indices()
        point_comments = get_all_point_comments()
        general_comment = get_general_comment()
        save_error_labels(
            patient_id=patient_id,
            error_indices=error_indices,
            point_comments=point_comments,
            general_comment=general_comment,
            is_complete=is_patient_complete(patient_id)
        )
    except Exception as e:
        st.error(f"Error saving labels: {e}")


def main():
    """Main application entry point."""
    
    # Load custom CSS
    load_css()
    
    # Load data with loading indicator
    try:
        with st.spinner("Loading patient data..."):
            all_data = load_all_data()
            patient_list = get_patient_list(all_data)
        
        if not patient_list:
            st.error("No patient data found. Please check data directory.")
            return
            
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return
    
    # Initialize session state
    initialize_session_state(default_patient_id=patient_list[0])
    
    # Get current patient ID
    current_patient_id = get_current_patient_id()

    # Handle pending navigation requests (e.g., after marking complete)
    pending_patient_id = st.session_state.pop("pending_patient_id", None)
    if pending_patient_id and pending_patient_id != current_patient_id:
        set_current_patient_id(pending_patient_id)
        current_patient_id = pending_patient_id
    
    if not current_patient_id:
        st.error("No patient selected")
        return
    
    # Get completion statuses for all patients
    completion_status = {pid: is_patient_complete(pid) for pid in patient_list}

    # Per-patient toggle persistence
    toggle_store = st.session_state.setdefault("_show_missing_height_toggle", {})
    table_toggle_key = f"show_missing_height_widget_{current_patient_id}"
    if table_toggle_key not in st.session_state:
        st.session_state[table_toggle_key] = toggle_store.get(current_patient_id, True)
    else:
        toggle_store[current_patient_id] = st.session_state[table_toggle_key]
    
    # Render sidebar with navigation
    new_patient_id = render_sidebar(
        current_patient_id=current_patient_id,
        patient_list=patient_list,
        completion_status=completion_status
    )
    
    # Handle patient navigation
    if new_patient_id:
        handle_patient_navigation(new_patient_id)
        return
    
    # Load and process current patient data
    try:
        # Load single patient with all calculated fields
        with st.spinner("Processing patient data..."):
            patient_data = load_single_patient(current_patient_id, all_data)
        
        # Get patient metadata
        sex = patient_data['sex'].iloc[0] if 'sex' in patient_data.columns else 'Unknown'
        valid_height_indices, _missing_height_indices, visit_number_map = compute_height_visit_metadata(patient_data)
        valid_index_positions = {idx: pos for pos, idx in enumerate(valid_height_indices)}
        
    except Exception as e:
        st.error(f"Error loading patient data: {e}")
        return
    
    # Get current state
    selected_index = get_selected_point_index()
    error_indices = get_error_indices()

    save_request_key = STATE_KEYS["save_request"]
    if st.session_state.get(save_request_key):
        # Persist current patient state before exporting combined dataset
        auto_save_labels(current_patient_id)
        try:
            point_comments = get_all_point_comments()
            general_comment = get_general_comment()
            output_path = save_all_labeled_data(
                all_data=all_data,
                override_patient_id=current_patient_id,
                override_error_indices=error_indices,
                override_point_comments=point_comments,
                override_general_comment=general_comment,
                override_completed=is_patient_complete(current_patient_id)
            )

            try:
                display_path = output_path.relative_to(PROJECT_ROOT)
            except ValueError:
                display_path = output_path

            # Read the CSV file for download
            with open(output_path, 'rb') as f:
                csv_data = f.read()

            st.session_state[STATE_KEYS["save_feedback"]] = (
                "success",
                f"Saved labeled data to {display_path}",
                csv_data,
                output_path.name
            )
        except Exception as e:
            st.session_state[STATE_KEYS["save_feedback"]] = (
                "error",
                f"Failed to save labeled data: {e}",
                None,
                None
            )
        finally:
            st.session_state[save_request_key] = False
            # Force rerun to immediately display the feedback message
            st.rerun()

    if valid_height_indices:
        if selected_index is None or selected_index not in valid_index_positions:
            set_selected_point_index(valid_height_indices[0])
            st.rerun()
            return
    else:
        if selected_index is not None:
            set_selected_point_index(None)
            st.rerun()
            return
    
    # Render patient information header
    render_patient_info(
        patient_data,
        current_patient_id,
        completion_status.get(current_patient_id, False)
    )
    
    # Create main layout: two columns
    col_left, col_right = st.columns([6, 4])
    
    with col_left:
        st.markdown("### 📊 Growth Charts")
        # Header with navigation controls in a single row
        header_col1, header_col2, header_col3, header_col4 = st.columns([4, 2, 0.5, 0.5])
        
        with header_col1:
            st.info("👇 Click a measurement or use the ◀/▶ arrows to navigate and mark measurements as errors.\n\n* Select area on the plot or use plot control to zoom in")
        
        # Navigation buttons for point selection
        total_visits = len(valid_height_indices)
        current_visit_num = visit_number_map.get(selected_index)
        current_position = valid_index_positions.get(selected_index)

        with header_col2:
            st.write("")  # Spacer for alignment
            if total_visits == 0:
                st.caption("No height measurements")
            else:
                visit_display = current_visit_num if current_visit_num is not None else "—"
                st.caption(f"Visit {visit_display} / {total_visits}")

        with header_col3:
            st.write("")  # Spacer for alignment
            prev_disabled = (total_visits == 0) or (current_position is None) or (current_position <= 0)
            if st.button("◀", key="prev_visit", disabled=prev_disabled, help="Previous visit", use_container_width=True):
                target_index = valid_height_indices[current_position - 1]
                set_selected_point_index(int(target_index))
                st.rerun()

        with header_col4:
            st.write("")  # Spacer for alignment
            next_disabled = (total_visits == 0) or (current_position is None) or (current_position >= total_visits - 1)
            if st.button("▶", key="next_visit", disabled=next_disabled, help="Next visit", use_container_width=True):
                target_index = valid_height_indices[current_position + 1]
                set_selected_point_index(int(target_index))
                st.rerun()
        
        # Render combined chart (height + velocity with synchronized hovering)
        try:
            combined_fig = render_combined_charts(
                patient_data=patient_data,
                sex=sex,
                error_indices=error_indices,
                selected_index=selected_index
            )
            
            # Display chart with click handling
            chart_selection = st.plotly_chart(
                combined_fig,
                width='stretch',
                key="combined_charts",
                config=CHART_CONFIG,
                on_select="rerun",
                selection_mode="points"
            )
            
            # Handle point selection from charts
            if chart_selection and len(chart_selection.selection.points) > 0:
                clicked_point = chart_selection.selection.points[0]
                if 'customdata' in clicked_point and clicked_point['customdata'] is not None:
                    clicked_index = int(clicked_point['customdata'])
                    if clicked_index != selected_index:
                        set_selected_point_index(clicked_index)
                        st.rerun()
            
        except Exception as e:
            st.error(f"Error rendering charts: {e}")
    
    with col_right:
        st.markdown("### 📋 Data Table")

        show_missing_heights = st.toggle(
            "Show visits missing height",
            value=st.session_state[table_toggle_key],
            key=table_toggle_key,
            help="Include visits without a height measurement. These rows are view-only."
        )
        toggle_store[current_patient_id] = show_missing_heights

        # Render data table
        try:
            clicked_index = render_data_table(
                patient_data=patient_data,
                selected_index=selected_index,
                error_indices=error_indices,
                show_missing_heights=show_missing_heights
            )
            
            # Handle table row selection
            if clicked_index is not None and clicked_index != selected_index:
                set_selected_point_index(clicked_index)
                st.rerun()
                
        except Exception as e:
            st.error(f"Error rendering data table: {e}")
        
        # Render error controls
        try:
            render_error_controls(
                patient_id=current_patient_id,
                patient_data=patient_data,
                selected_index=selected_index,
                error_indices=error_indices
            )
            
        except Exception as e:
            st.error(f"Error rendering error controls: {e}")
    
    # Render patient completion controls OUTSIDE and BELOW the two columns
    try:
        render_patient_completion_controls(
            patient_id=current_patient_id,
            error_indices=error_indices,
            patient_list=patient_list
        )
        
    except Exception as e:
        st.error(f"Error rendering completion controls: {e}")
    
    # Auto-save labels periodically
    auto_save_labels(current_patient_id)


if __name__ == "__main__":
    main()
