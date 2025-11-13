"""
Configuration constants for the error labeling dashboard.
"""

from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
LABELS_DIR = DATA_DIR / "labels"

# Ensure directories exist
LABELS_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

# Data constants
DAYS_PER_YEAR = 365.25

# CDC/WHO Growth Standard constants
Z_SCORE_BOUNDS = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
RIBBON_Z_SCORES = [-5, 5]  # Bounds for ribbon plot

# Plot configuration
PLOT_HEIGHT = 400
PLOT_COLORS = {
    "primary": "#1f77b4",
    "error": "#d62728",
    "selected": "#ff7f0e",
    "ribbon_fill": "rgba(31, 119, 180, 0.2)",
    "ribbon_line": "rgba(31, 119, 180, 0.5)",
    "median": "#2ca02c",
}

# Plot settings - optimized for performance
CHART_CONFIG = {
    "displayModeBar": True,  # Show toolbar with zoom, pan, etc.
    "displaylogo": False,
    "staticPlot": False,  # Keep interactive for point selection
    "responsive": True,
    "modeBarButtonsToRemove": [],  # Keep all buttons available
}

# Data table columns
TABLE_COLUMNS = [
    "age_in_days",
    "height_in",
    "weight_oz",
    "head_circ_cm",
    "bmi",
    "encounter_type",
]

# Patient data required columns
REQUIRED_COLUMNS = [
    "patient_id",
    "age_in_days",
    "height_in",
    "sex",
]

# Optional patient metadata columns
METADATA_COLUMNS = [
    "race",
    "ethnicity",
]

# Session state keys
STATE_KEYS = {
    "current_patient_id": "current_patient_id",
    "selected_point_index": "selected_point_index",
    "error_indices": "error_indices",
    "point_comments": "point_comments",
    "general_comment": "general_comment",
    "patient_statuses": "patient_statuses",
    "initialized": "initialized",
    "patient_selector_widget": "patient_selector",
    "save_request": "save_current_patient_request",
    "save_feedback": "save_current_patient_feedback",
}
