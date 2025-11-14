"""Persistence layer for saving and loading error labels, comments, and exports."""

import json
from pathlib import Path
from typing import Dict, List, Optional, Set
from datetime import datetime

import pandas as pd

from src.config import LABELS_DIR, PROCESSED_DATA_DIR, get_user_labels_dir, get_user_processed_dir


def get_label_file_path(patient_id: str, username: Optional[str] = None) -> Path:
    """
    Get the file path for a patient's label data.
    
    Parameters
    ----------
    patient_id : str
        Patient identifier
    username : str, optional
        Username for user-specific storage
        
    Returns
    -------
    Path
        Path to the JSON file for this patient
    """
    labels_dir = get_user_labels_dir(username)
    return labels_dir / f"{patient_id}_labels.json"


def save_error_labels(
    patient_id: str,
    error_indices: Set[int],
    point_comments: Dict[int, str] = None,
    general_comment: str = "",
    is_complete: bool = False,
    username: Optional[str] = None
) -> None:
    """
    Save error labels and comments for a patient.
    
    Parameters
    ----------
    patient_id : str
        Patient identifier
    error_indices : Set[int]
        Set of indices marked as errors
    point_comments : Dict[int, str], optional
        Dictionary mapping point indices to comments
    general_comment : str, optional
        General comment about the patient
    is_complete : bool, default False
        Whether the patient review is complete
    username : str, optional
        Username for user-specific storage
        
    Raises
    ------
    ValueError
        If patient_id is empty
    IOError
        If unable to write to file
    """
    if not patient_id or not isinstance(patient_id, str):
        raise ValueError("Patient ID must be a non-empty string")
    
    if point_comments is None:
        point_comments = {}
    
    try:
        # Ensure labels directory exists
        labels_dir = get_user_labels_dir(username)
        labels_dir.mkdir(parents=True, exist_ok=True)
        
        data = {
            "patient_id": patient_id,
            "completed": is_complete,
            "last_updated": datetime.now().isoformat(),
            "error_indices": sorted(list(error_indices)),
            "point_comments": {str(k): v for k, v in point_comments.items() if v and v.strip()},
            "general_comment": general_comment.strip() if general_comment else "",
        }
        
        file_path = get_label_file_path(patient_id, username)
        
        # Write to temporary file first, then rename (atomic operation)
        temp_path = file_path.with_suffix('.tmp')
        with open(temp_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        # Rename to final location
        temp_path.replace(file_path)
        
    except Exception as e:
        raise IOError(f"Failed to save labels for patient {patient_id}: {str(e)}")


def save_all_labeled_data(
    all_data: pd.DataFrame,
    override_patient_id: Optional[str] = None,
    override_error_indices: Optional[Set[int]] = None,
    override_point_comments: Optional[Dict[int, str]] = None,
    override_general_comment: Optional[str] = None,
    override_completed: Optional[bool] = None,
    output_path: Optional[Path] = None,
    username: Optional[str] = None
) -> Path:
    """Persist the entire dataset with error flags and comments appended.

    Parameters
    ----------
    all_data : pd.DataFrame
        Original dataset containing every patient measurement.
    override_patient_id : str, optional
        Patient ID whose in-memory state should override persisted labels.
    override_error_indices : set[int], optional
        Error indices for the override patient.
    override_point_comments : dict[int, str], optional
        Point comments for the override patient.
    override_general_comment : str, optional
        General comment for the override patient.
    override_completed : bool, optional
        Completion status for the override patient.
    output_path : Path, optional
        Destination file path. Defaults to user-specific processed data directory.
    username : str, optional
        Username for user-specific storage

    Returns
    -------
    Path
        Location of the saved CSV file.
    """

    if all_data is None or all_data.empty:
        raise ValueError("All patient data is empty")

    if output_path is None:
        processed_dir = get_user_processed_dir(username)
        destination = processed_dir / "all_patients_labeled.csv"
    else:
        destination = output_path
    
    destination.parent.mkdir(parents=True, exist_ok=True)

    # Preserve patient order as they appear in the source data
    patient_order = list(dict.fromkeys(all_data["patient_id"].tolist()))
    augmented_frames: List[pd.DataFrame] = []

    for patient_id in patient_order:
        patient_rows = all_data[all_data["patient_id"] == patient_id].copy()
        if patient_rows.empty:
            continue

        labels = load_error_labels(patient_id, username)
        error_indices = labels.get("error_indices", set())
        point_comments = labels.get("point_comments", {})
        general_comment = labels.get("general_comment", "")
        completed = bool(labels.get("completed", False))

        if patient_id == override_patient_id:
            if override_error_indices is not None:
                error_indices = set(override_error_indices)
            if override_point_comments is not None:
                point_comments = {
                    int(idx): text
                    for idx, text in override_point_comments.items()
                }
            if override_general_comment is not None:
                general_comment = override_general_comment
            if override_completed is not None:
                completed = override_completed

        # Sort by age to align with saved indices, then restore original order
        sorted_rows = patient_rows.sort_values("age_in_days").copy()
        sorted_rows["_original_position"] = sorted_rows.index
        sorted_rows.reset_index(drop=True, inplace=True)

        normalized_comments = {
            int(idx): (str(text).strip() if str(text).strip() else "")
            for idx, text in point_comments.items()
        }

        sorted_rows["error"] = sorted_rows.index.isin(error_indices)
        sorted_rows["comment"] = sorted_rows.index.map(
            lambda idx: normalized_comments.get(int(idx), "")
        )
        sorted_rows["general_patient_comment"] = (
            general_comment.strip() if general_comment else ""
        )
        sorted_rows["confirmed"] = completed

        restored_rows = sorted_rows.sort_values("_original_position").drop(
            columns="_original_position"
        )
        augmented_frames.append(restored_rows)

    if not augmented_frames:
        raise ValueError("No patient data available to export")

    combined = pd.concat(augmented_frames, ignore_index=True)
    combined.to_csv(destination, index=False)

    return destination


def load_error_labels(patient_id: str, username: Optional[str] = None) -> Dict:
    """
    Load error labels and comments for a patient.
    
    Parameters
    ----------
    patient_id : str
        Patient identifier
    username : str, optional
        Username for user-specific storage
        
    Returns
    -------
    dict
        Dictionary containing:
        - error_indices: Set of error indices
        - point_comments: Dict mapping indices to comments
        - general_comment: General comment string
        - completed: Boolean completion status
        - last_updated: ISO timestamp string
        
        Returns default empty values if file doesn't exist or on error.
        
    Raises
    ------
    ValueError
        If patient_id is empty
    """
    if not patient_id or not isinstance(patient_id, str):
        raise ValueError("Patient ID must be a non-empty string")
    
    default_data = {
        "error_indices": set(),
        "point_comments": {},
        "general_comment": "",
        "completed": False,
        "last_updated": None,
    }
    
    file_path = get_label_file_path(patient_id, username)
    
    if not file_path.exists():
        return default_data
    
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Validate data structure
        if not isinstance(data, dict):
            print(f"Warning: Invalid label data format for {patient_id}, using defaults")
            return default_data
        
        return {
            "error_indices": set(data.get("error_indices", [])),
            "point_comments": {
                int(k): v for k, v in data.get("point_comments", {}).items()
                if isinstance(k, (int, str)) and isinstance(v, str)
            },
            "general_comment": str(data.get("general_comment", "")),
            "completed": bool(data.get("completed", False)),
            "last_updated": data.get("last_updated"),
        }
    except (json.JSONDecodeError, ValueError, TypeError) as e:
        print(f"Error loading labels for {patient_id}: {e}")
        return default_data
    except Exception as e:
        print(f"Unexpected error loading labels for {patient_id}: {e}")
        return default_data


def save_patient_status(patient_id: str, is_complete: bool, username: Optional[str] = None) -> None:
    """
    Update the completion status for a patient.
    
    Parameters
    ----------
    patient_id : str
        Patient identifier
    is_complete : bool
        Whether the patient review is complete
    username : str, optional
        Username for user-specific storage
    """
    # Load existing data
    existing_data = load_error_labels(patient_id, username)
    
    # Update completion status
    save_error_labels(
        patient_id=patient_id,
        error_indices=existing_data["error_indices"],
        point_comments=existing_data["point_comments"],
        general_comment=existing_data["general_comment"],
        is_complete=is_complete,
        username=username
    )


def get_all_patient_statuses(username: Optional[str] = None) -> Dict[str, bool]:
    """
    Get completion status for all patients with saved labels.
    
    Parameters
    ----------
    username : str, optional
        Username for user-specific storage
    
    Returns
    -------
    dict
        Dictionary mapping patient_id to completion status (bool)
    """
    statuses = {}
    
    labels_dir = get_user_labels_dir(username)
    if not labels_dir.exists():
        return statuses
    
    for label_file in labels_dir.glob("*_labels.json"):
        try:
            with open(label_file, 'r') as f:
                data = json.load(f)
                patient_id = data.get("patient_id")
                if patient_id:
                    statuses[patient_id] = data.get("completed", False)
        except (json.JSONDecodeError, ValueError):
            continue
    
    return statuses


def delete_patient_labels(patient_id: str, username: Optional[str] = None) -> bool:
    """
    Delete all saved labels for a patient.
    
    Parameters
    ----------
    patient_id : str
        Patient identifier
    username : str, optional
        Username for user-specific storage
        
    Returns
    -------
    bool
        True if file was deleted, False if it didn't exist
    """
    file_path = get_label_file_path(patient_id, username)
    
    if file_path.exists():
        file_path.unlink()
        return True
    
    return False


def export_all_labels(output_path: Optional[Path] = None, username: Optional[str] = None) -> Path:
    """
    Export all patient labels to a single JSON file.
    
    Parameters
    ----------
    output_path : Path, optional
        Output file path. If None, saves to user-specific labels directory.
    username : str, optional
        Username for user-specific storage
        
    Returns
    -------
    Path
        Path to the exported file
    """
    labels_dir = get_user_labels_dir(username)
    
    if output_path is None:
        output_path = labels_dir / "all_labels_export.json"
    
    all_labels = {}
    
    for label_file in labels_dir.glob("*_labels.json"):
        if label_file.name == "all_labels_export.json":
            continue
            
        try:
            with open(label_file, 'r') as f:
                data = json.load(f)
                patient_id = data.get("patient_id")
                if patient_id:
                    all_labels[patient_id] = data
        except (json.JSONDecodeError, ValueError):
            continue
    
    with open(output_path, 'w') as f:
        json.dump(all_labels, f, indent=2)
    
    return output_path
