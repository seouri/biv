from typing import Any, Dict

import pandas as pd
from pydantic import BaseModel, Field, ValidationError, field_validator, StrictFloat

from biv.methods.base import BaseDetector


class RangeConfig(BaseModel):
    """
    Configuration for RangeDetector.

    Defines the lower and upper bounds for allowed values in a column.
    """

    lower_bound: StrictFloat = Field(
        description="Lower bound (inclusive minimum) for allowed values."
    )
    upper_bound: StrictFloat = Field(
        description="Upper bound (exclusive maximum) for allowed values."
    )

    @field_validator("upper_bound", mode="after")
    @classmethod
    def validate_bounds(cls, v: float, info: Any) -> float:
        """Validate that lower_bound < upper_bound."""
        lower_val = info.data.get("lower_bound")
        if lower_val is not None and lower_val >= v:
            raise ValueError("lower_bound must be less than upper_bound")
        return v


class RangeDetector(BaseDetector):
    """
    Detector for Range-based Outliers.

    Flags values as BIV if they fall outside the specified [min, max) range
    for each column. NaN values are not flagged.

    Config example:
        {
            "weight_kg": {"min": 30.0, "max": 200.0},
            "height_cm": {"min": 100.0, "max": 220.0}
        }
    """

    def __init__(self, config: Dict[str, Dict[str, float]]) -> None:
        """
        Initialize with range configs.

        Args:
            config: Dictionary mapping column names to min/max configs.
                   Example: {"weight_kg": {"min": 30.0, "max": 200.0}}

        Raises:
            ValueError: If config is invalid (missing min/max, min>=max, non-float).
        """
        if not isinstance(config, dict):
            raise ValueError("Config must be a dict mapping column names to dicts")
        # Validate config using Pydantic, mapping user-friendly keys to internal fields
        self.config: Dict[str, RangeConfig] = {}
        for col, params in config.items():
            if not isinstance(params, dict):
                raise ValueError(f"Config for column '{col}' must be a dict")
            try:
                self.config[col] = RangeConfig(
                    lower_bound=params["min"], upper_bound=params["max"]
                )
            except (KeyError, TypeError, ValidationError) as e:
                raise ValueError(f"Invalid config for column '{col}': {e}") from e
        self.validate_config()

    def validate_config(self) -> None:
        """Validate the detector configuration."""
        # Additional validation if needed; pydantic handles main validation
        pass

    def detect(self, df: pd.DataFrame, columns: list[str]) -> Dict[str, pd.Series]:
        """Detect BIVs by checking if values are outside configured ranges.

        Args:
            df: DataFrame to check.
            columns: List of column names to check (must have configs).

        Returns:
            Dict of boolean Series, one per column, where True indicates BIV.

        Raises:
            ValueError: If column not in df or no config for column.
        """
        results = {}
        for col in columns:
            self._validate_column(df, col)
            if col not in self.config:
                raise ValueError(f"No range config provided for column '{col}'")
            lower_val = self.config[col].lower_bound
            upper_val = self.config[col].upper_bound
            series = df[col]
            flags = (series < lower_val) | (series > upper_val)
            results[col] = flags.rename(col)
        return results
