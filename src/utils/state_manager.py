"""
Session state management utilities for Streamlit.
"""

import streamlit as st
from streamlit.errors import StreamlitAPIException
from typing import Any, Optional, Set, Dict
from src.config import STATE_KEYS
from src.utils.persistence import load_error_labels, get_all_patient_statuses


def _get_current_username() -> Optional[str]:
    """Get the current authenticated username from session state."""
    return st.session_state.get("authenticated_user")


def initialize_session_state(default_patient_id: Optional[str] = None) -> None:
    """
    Initialize Streamlit session state with default values.
    
    Parameters
    ----------
    default_patient_id : str, optional
        Default patient ID to display on first load
    """
    selector_key = STATE_KEYS["patient_selector_widget"]

    if STATE_KEYS["initialized"] in st.session_state:
        if STATE_KEYS["save_request"] not in st.session_state:
            st.session_state[STATE_KEYS["save_request"]] = False
        if STATE_KEYS["save_feedback"] not in st.session_state:
            st.session_state[STATE_KEYS["save_feedback"]] = None
        if selector_key not in st.session_state and default_patient_id:
            st.session_state[selector_key] = default_patient_id
        return
    
    # Current patient
    st.session_state[STATE_KEYS["current_patient_id"]] = default_patient_id
    
    # Selected point (None means no selection)
    st.session_state[STATE_KEYS["selected_point_index"]] = None
    
    # Error indices (set of integers)
    st.session_state[STATE_KEYS["error_indices"]] = set()
    
    # Point-specific comments (dict: index -> comment)
    st.session_state[STATE_KEYS["point_comments"]] = {}
    
    # General patient comment
    st.session_state[STATE_KEYS["general_comment"]] = ""
    
    # Patient completion statuses (dict: patient_id -> bool)
    username = _get_current_username()
    st.session_state[STATE_KEYS["patient_statuses"]] = get_all_patient_statuses(username)
    
    if default_patient_id:
        st.session_state[selector_key] = default_patient_id

    # Save/export controls
    st.session_state[STATE_KEYS["save_request"]] = False
    st.session_state[STATE_KEYS["save_feedback"]] = None
    
    # Mark as initialized
    st.session_state[STATE_KEYS["initialized"]] = True
    
    # Load labels for the default patient if provided
    if default_patient_id:
        load_patient_labels(default_patient_id)


def get_current_patient_id() -> Optional[str]:
    """Get the current patient ID from session state."""
    return st.session_state.get(STATE_KEYS["current_patient_id"])


def set_current_patient_id(patient_id: str) -> None:
    """
    Set the current patient ID and load their data.
    
    Parameters
    ----------
    patient_id : str
        Patient identifier to switch to
    """
    st.session_state[STATE_KEYS["current_patient_id"]] = patient_id
    try:
        st.session_state[STATE_KEYS["patient_selector_widget"]] = patient_id
    except StreamlitAPIException:
        # Widget state cannot be mutated after instantiation; defer sync to UI layer
        pass
    
    # Load saved labels for this patient
    load_patient_labels(patient_id)
    
    # Clear selected point when switching patients
    st.session_state[STATE_KEYS["selected_point_index"]] = None


def get_selected_point_index() -> Optional[int]:
    """Get the currently selected point index."""
    return st.session_state.get(STATE_KEYS["selected_point_index"])


def set_selected_point_index(index: Optional[int]) -> None:
    """
    Set the selected point index.
    
    Parameters
    ----------
    index : int or None
        Index of the selected point, or None to clear selection
    """
    st.session_state[STATE_KEYS["selected_point_index"]] = index


def get_error_indices() -> Set[int]:
    """Get the set of error indices for the current patient."""
    return st.session_state.get(STATE_KEYS["error_indices"], set())


def add_error_index(index: int) -> None:
    """
    Mark a point as an error.
    
    Parameters
    ----------
    index : int
        Index of the point to mark as error
    """
    errors = get_error_indices()
    errors.add(index)
    st.session_state[STATE_KEYS["error_indices"]] = errors


def remove_error_index(index: int) -> None:
    """
    Unmark a point as an error.
    
    Parameters
    ----------
    index : int
        Index of the point to unmark as error
    """
    errors = get_error_indices()
    errors.discard(index)
    st.session_state[STATE_KEYS["error_indices"]] = errors


def is_error(index: int) -> bool:
    """
    Check if a point is marked as an error.
    
    Parameters
    ----------
    index : int
        Index to check
        
    Returns
    -------
    bool
        True if the point is an error
    """
    return index in get_error_indices()


def get_point_comment(index: int) -> str:
    """
    Get the comment for a specific point.
    
    Parameters
    ----------
    index : int
        Point index
        
    Returns
    -------
    str
        Comment text, or empty string if no comment
    """
    comments = st.session_state.get(STATE_KEYS["point_comments"], {})
    return comments.get(index, "")


def set_point_comment(index: int, comment: str) -> None:
    """
    Set the comment for a specific point.
    
    Parameters
    ----------
    index : int
        Point index
    comment : str
        Comment text
    """
    comments = st.session_state.get(STATE_KEYS["point_comments"], {})
    if comment.strip():
        comments[index] = comment.strip()
    else:
        # Remove comment if empty
        comments.pop(index, None)
    st.session_state[STATE_KEYS["point_comments"]] = comments


def get_all_point_comments() -> Dict[int, str]:
    """Return a copy of all point comments for the current patient."""
    comments = st.session_state.get(STATE_KEYS["point_comments"], {})
    return {int(idx): text for idx, text in comments.items()}


def get_general_comment() -> str:
    """Get the general comment for the current patient."""
    return st.session_state.get(STATE_KEYS["general_comment"], "")


def set_general_comment(comment: str) -> None:
    """
    Set the general comment for the current patient.
    
    Parameters
    ----------
    comment : str
        Comment text
    """
    st.session_state[STATE_KEYS["general_comment"]] = comment


def is_patient_complete(patient_id: str) -> bool:
    """
    Check if a patient's review is marked as complete.
    
    Parameters
    ----------
    patient_id : str
        Patient identifier
        
    Returns
    -------
    bool
        True if patient review is complete
    """
    statuses = st.session_state.get(STATE_KEYS["patient_statuses"], {})
    return statuses.get(patient_id, False)


def set_patient_complete(patient_id: str, is_complete: bool) -> None:
    """
    Set the completion status for a patient.
    
    Parameters
    ----------
    patient_id : str
        Patient identifier
    is_complete : bool
        Completion status
    """
    statuses = st.session_state.get(STATE_KEYS["patient_statuses"], {})
    statuses[patient_id] = is_complete
    st.session_state[STATE_KEYS["patient_statuses"]] = statuses


def load_patient_labels(patient_id: str) -> None:
    """
    Load saved labels and comments for a patient into session state.
    
    Parameters
    ----------
    patient_id : str
        Patient identifier
    """
    username = _get_current_username()
    labels = load_error_labels(patient_id, username)
    
    st.session_state[STATE_KEYS["error_indices"]] = labels["error_indices"]
    st.session_state[STATE_KEYS["point_comments"]] = labels["point_comments"]
    st.session_state[STATE_KEYS["general_comment"]] = labels["general_comment"]


def get_completion_progress() -> tuple[int, int]:
    """
    Get the number of completed patients vs total.
    
    Returns
    -------
    tuple
        (completed_count, total_count)
    """
    statuses = st.session_state.get(STATE_KEYS["patient_statuses"], {})
    completed = sum(1 for is_complete in statuses.values() if is_complete)
    total = len(statuses) if statuses else 0
    
    return completed, total


def clear_patient_state() -> None:
    """Clear all patient-specific state (useful for testing)."""
    st.session_state[STATE_KEYS["selected_point_index"]] = None
    st.session_state[STATE_KEYS["error_indices"]] = set()
    st.session_state[STATE_KEYS["point_comments"]] = {}
    st.session_state[STATE_KEYS["general_comment"]] = ""
