"""Utility helpers for working with visit numbering.

Provides consistent visit numbering that skips measurements lacking
height information so the UI can display sequential visit numbers
aligned with available height data.
"""

from __future__ import annotations

from typing import Dict, List, Tuple, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - used only for type checking
    import pandas as pd  # type: ignore[import]


def compute_height_visit_metadata(patient_data: "pd.DataFrame") -> Tuple[List[int], List[int], Dict[int, int]]:
    """Compute visit metadata for height measurements.

    Returns a tuple of three elements:
      - valid_indices: list of indices with a recorded height measurement
      - missing_indices: list of indices without a recorded height measurement
      - visit_number_map: mapping from index -> sequential visit number (starting at 1)

    Parameters
    ----------
    patient_data : pd.DataFrame
        Patient data containing a ``height_in`` column

    Returns
    -------
    tuple[list[int], list[int], dict[int, int]]
        Metadata describing valid and missing height visits.
    """
    if patient_data.empty:
        return [], [], {}

    if "height_in" not in patient_data.columns:
        indices = [int(idx) for idx in patient_data.index.tolist()]
        visit_number_map = {idx: order for order, idx in enumerate(indices, start=1)}
        return indices, [], visit_number_map

    height_series = patient_data["height_in"]
    valid_indices = [int(idx) for idx in height_series[height_series.notna()].index.tolist()]
    missing_indices = [int(idx) for idx in height_series[height_series.isna()].index.tolist()]
    visit_number_map = {idx: order for order, idx in enumerate(valid_indices, start=1)}
    return valid_indices, missing_indices, visit_number_map
