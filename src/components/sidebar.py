"""
Sidebar Navigation Component

This module provides the sidebar with patient navigation and progress tracking.
"""

import streamlit as st
from typing import List, Dict, Callable, Optional

from src.config import STATE_KEYS


def render_sidebar(
    current_patient_id: str,
    patient_list: List[str],
    completion_status: Dict[str, bool],
    on_patient_change: Optional[Callable[[str], None]] = None
) -> Optional[str]:
    """
    Render the sidebar with patient navigation and status.
    
    Args:
        current_patient_id: Currently selected patient ID
        patient_list: List of all patient IDs
        completion_status: Dictionary mapping patient_id -> completion status
        on_patient_change: Callback function when patient selection changes
        
    Returns:
        Selected patient ID if changed, None otherwise
    """
    with st.sidebar:
        if not patient_list:
            st.warning("No patients available for navigation.")
            return None

        selector_key = STATE_KEYS["patient_selector_widget"]
        selector_value = st.session_state.get(selector_key)
        if selector_value not in patient_list:
            selector_value = current_patient_id if current_patient_id in patient_list else patient_list[0]
            st.session_state[selector_key] = selector_value

        # Title and instructions
        st.title("🩺 Growth Error Labeling")
        st.markdown("""
        **Instructions:**
        - Review each patient's growth data
        - Mark error measurements
        - Add comments as needed
        - Complete review for each patient
        - Save labeled data anytime
        """)

        if st.button(
            "💾 Save Labeled Data",
            use_container_width=True,
            help="Export all patient measurements with error flags and comments"
        ):
            st.session_state[STATE_KEYS["save_request"]] = True
            st.session_state[STATE_KEYS["save_feedback"]] = None

        feedback = st.session_state.get(STATE_KEYS["save_feedback"])
        if feedback:
            if len(feedback) == 4:
                status, message, csv_data, filename = feedback
            else:
                # Backwards compatibility
                status, message = feedback[0], feedback[1]
                csv_data, filename = None, None
                
            if status == "success":
                st.success(message)
                if csv_data and filename:
                    # Use a container to keep download button visible
                    st.download_button(
                        label="📥 Download Copy",
                        data=csv_data,
                        file_name=filename,
                        mime="text/csv",
                        use_container_width=True,
                        help="Download a copy of the labeled data as CSV",
                        key=f"download_{filename}_{hash(csv_data)}"
                    )
            else:
                st.error(message)
                # Only clear error messages immediately
                st.session_state[STATE_KEYS["save_feedback"]] = None
        
        st.divider()
        
        # Progress summary
        completed_count = sum(1 for status in completion_status.values() if status)
        total_count = len(patient_list)
        progress_pct = (completed_count / total_count * 100) if total_count > 0 else 0
        
        st.subheader("📊 Progress")
        st.progress(progress_pct / 100)
        st.markdown(f"**{completed_count} of {total_count} patients completed**")
        
        st.divider()
        
        # Patient navigation
        st.subheader("🔍 Patient Navigation")

        selector_key = STATE_KEYS["patient_selector_widget"]
        active_patient_id = current_patient_id if current_patient_id in patient_list else patient_list[0]

        if selector_key not in st.session_state or st.session_state[selector_key] not in patient_list:
            st.session_state[selector_key] = active_patient_id

        selected_patient = st.session_state[selector_key]
        if selected_patient not in patient_list:
            selected_patient = active_patient_id
            st.session_state[selector_key] = selected_patient

        dropdown_placeholder = st.empty()
        current_index = patient_list.index(active_patient_id)

        # Previous/Next buttons with tooltips
        col1, col2 = st.columns(2)

        with col1:
            prev_disabled = current_index == 0
            if st.button(
                "◀ Previous",
                disabled=prev_disabled,
                use_container_width=True,
                help="Navigate to previous patient (keyboard: ←)"
            ):
                if current_index > 0:
                    selected_patient = patient_list[current_index - 1]
                    st.session_state[selector_key] = selected_patient

        with col2:
            next_disabled = current_index >= len(patient_list) - 1
            if st.button(
                "Next ▶",
                disabled=next_disabled,
                use_container_width=True,
                help="Navigate to next patient (keyboard: →)"
            ):
                if current_index < len(patient_list) - 1:
                    selected_patient = patient_list[current_index + 1]
                    st.session_state[selector_key] = selected_patient

        st.divider()
        st.markdown(
            """
            <style>
            .patient-list {
                max-height: 400px;
                overflow-y: auto;
            }
            .patient-item {
                padding: 8px;
                margin: 4px 0;
                border-radius: 4px;
                cursor: pointer;
                transition: background-color 0.2s;
            }
            .patient-item:hover {
                background-color: #e8eaf6;
            }
            .patient-item.completed {
                background-color: #e8f5e9;
                border-left: 3px solid #4caf50;
            }
            .patient-item.current {
                background-color: #e3f2fd;
                border-left: 3px solid #2196f3;
                font-weight: bold;
            }
            .patient-item.incomplete {
                background-color: #fff3e0;
                border-left: 3px solid #ff9800;
            }
            </style>
            """,
            unsafe_allow_html=True
        )
        
        # Display patient list with clickable items
        for pid in patient_list:
            is_completed = completion_status.get(pid, False)
            is_current = pid == current_patient_id
            
            # Determine styling
            if is_current:
                style_class = "current"
                icon = "👉"
            elif is_completed:
                style_class = "completed"
                icon = "✅"
            else:
                style_class = "incomplete"
                icon = "⏳"
            
            # Create clickable button for each patient
            if st.button(
                f"{icon} Patient {pid}",
                key=f"nav_{pid}",
                use_container_width=True,
                type="primary" if is_current else "secondary"
            ):
                selected_patient = pid
                st.session_state[selector_key] = selected_patient
        
        dropdown_index = patient_list.index(st.session_state[selector_key])
        with dropdown_placeholder:
            st.selectbox(
                "Select Patient:",
                options=patient_list,
                index=dropdown_index,
                format_func=lambda pid: f"{'✓ ' if completion_status.get(pid, False) else '  '}Patient {pid}",
                key=selector_key
            )

        selected_patient = st.session_state[selector_key]

        # Return changed patient ID if different
        if selected_patient != current_patient_id:
            if on_patient_change:
                on_patient_change(selected_patient)
            return selected_patient
        
        return None
