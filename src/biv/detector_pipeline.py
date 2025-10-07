"""Detector Pipeline for combining flags from multiple detectors."""

from typing import Dict, List
import pandas as pd


class DetectorPipeline:
    """
    Pipeline for combining boolean flags from multiple detectors.

    Supports logical OR and AND combinations.
    """

    def __init__(self, logic: str) -> None:
        """
        Initialize with combination logic.

        Args:
            logic: "OR" or "AND".

        Raises:
            KeyError: If logic is not supported.
        """
        supported = ["OR", "AND"]
        if logic not in supported:
            raise KeyError(f"Unsupported combination logic '{logic}'")
        self.logic = logic

    def combine_flags(
        self, flags_list: List[Dict[str, pd.Series]]
    ) -> Dict[str, pd.Series]:
        """
        Combine flags from multiple detectors.

        Args:
            flags_list: List of dicts from detectors, each with column to Series.

        Returns:
            Combined dict of Series.
        """
        if not flags_list:
            return {}

        # Collect all columns
        all_cols: set[str] = set()
        for flags in flags_list:
            all_cols.update(flags.keys())

        combined = {}
        if flags_list:
            first_index = list(flags_list[0].values())[0].index
            for col in all_cols:
                series_list = [
                    flags.get(col, pd.Series(False, index=first_index))
                    for flags in flags_list
                ]

                if self.logic == "OR":
                    combined[col] = pd.concat(series_list, axis=1).any(axis=1)
                elif self.logic == "AND":
                    combined[col] = pd.concat(series_list, axis=1).all(axis=1)

        return combined
