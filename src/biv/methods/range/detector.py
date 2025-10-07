from typing import Any, Dict, List

import pandas as pd
from pydantic import BaseModel, field_validator, StrictFloat

from biv.methods.base import BaseDetector


class AgeBracket(BaseModel):
    """
    Configuration for age-dependent ranges.
    """

    min_age: StrictFloat
    max_age: StrictFloat
    min: StrictFloat  # value min for this age bracket
    max: StrictFloat  # value max

    @field_validator("max_age", mode="after")
    @classmethod
    def min_age_lt_max_age(cls, v: float, info: Any) -> float:
        """Validate that min_age < max_age."""
        if info.data.get("min_age", float("inf")) >= v:
            raise ValueError("min_age must be < max_age")
        return v

    @field_validator("max", mode="after")
    @classmethod
    def min_lt_max(cls, v: float, info: Any) -> float:
        """Validate that min < max."""
        if info.data.get("min", float("inf")) >= v:
            raise ValueError("min must be < max")
        return v


class FlatRange(BaseModel):
    """
    Simple min-max range for values.
    """

    min: StrictFloat
    max: StrictFloat

    @field_validator("max", mode="after")
    @classmethod
    def min_lt_max(cls, v: float, info: Any) -> float:
        """Validate that min < max."""
        lower = info.data.get("min", float("inf"))
        if v <= lower:
            raise ValueError("max must be > min")
        return v


class AgeDependentRange(BaseModel):
    """
    Age-dependent ranges using brackets.
    """

    age_col: str
    age_brackets: List[AgeBracket]

    @field_validator("age_brackets", mode="after")
    @classmethod
    def no_overlapping_brackets(cls, v: List[AgeBracket]) -> List[AgeBracket]:
        if not v:
            raise ValueError("At least one age bracket required")
        sorted_brackets = sorted(v, key=lambda x: x.min_age)
        for i in range(len(sorted_brackets) - 1):
            if sorted_brackets[i].max_age > sorted_brackets[i + 1].min_age:
                raise ValueError("Overlapping age brackets")
        return v


class RangeDetector(BaseDetector):
    """
    Detector for Range-based Outliers.

    Supports both flat ranges and age-dependent ranges.

    Config examples:

    Flat ranges:
        {
            "weight_kg": {"min": 30.0, "max": 200.0},
            "height_cm": {"min": 100.0, "max": 220.0}
        }

    Age-dependent:
        {
            "age_col": "age",
            "weight_kg": {"age_brackets": [...]}
        }
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        """
        Initialize with range configs.

        Args:
            config: Dictionary containing configs.

        Raises:
            ValueError: If config is invalid.
        """
        if not isinstance(config, dict):
            raise ValueError("Config must be a dict")
        self.flat_configs: Dict[str, FlatRange] = {}
        self.age_configs: Dict[str, AgeDependentRange] = {}
        if "age_col" in config:
            age_col = config["age_col"]
            for col, params in config.items():
                if col == "age_col":
                    continue
                if not isinstance(params, dict):
                    raise ValueError(f"Config for column '{col}' must be a dict")
                if "age_brackets" not in params:
                    raise ValueError(
                        f"For age-dependent config, column '{col}' must have age_brackets"
                    )
                if "min" in params or "max" in params:
                    raise ValueError(
                        f"For age-dependent config, column '{col}' should not have 'min' or 'max' keys"
                    )
                self.age_configs[col] = AgeDependentRange(
                    age_col=age_col, age_brackets=params["age_brackets"]
                )
        else:
            for col, params in config.items():
                if not isinstance(params, dict):
                    raise ValueError(f"Config for column '{col}' must be a dict")
                if "min" not in params or "max" not in params:
                    raise ValueError(
                        f"Config for column '{col}' must have min and max for flat range"
                    )
                if "age_brackets" in params:
                    raise ValueError(
                        f"For flat range config, column '{col}' should not have 'age_brackets' key"
                    )
                self.flat_configs[col] = FlatRange(min=params["min"], max=params["max"])

    def validate_config(self) -> None:
        """Validate the detector configuration."""
        pass  # Validation done in __init__

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
            has_config = col in self.flat_configs or col in self.age_configs
            if not has_config:
                config_type = "age-dependent" if self.age_configs else "flat"
                raise ValueError(
                    f"No valid range configuration found for column '{col}'. "
                    f"Expected {config_type} format (see RangeDetector docstring)."
                )
            if col in self.flat_configs:
                # Flat range
                lower = self.flat_configs[col].min
                upper = self.flat_configs[col].max
                flags = (df[col] < lower) | (df[col] > upper)
            else:
                # Age-dependent
                age_dependent = self.age_configs[col]
                age_col = age_dependent.age_col
                if age_col not in df.columns:
                    raise ValueError(
                        f"Age column '{age_col}' does not exist in DataFrame"
                    )
                ages = df[age_col]
                flags = pd.Series(False, index=df.index, dtype=bool)
                for bracket in age_dependent.age_brackets:
                    mask = (
                        (ages >= bracket.min_age)
                        & (ages < bracket.max_age)
                        & ages.notna()
                    )
                    # Apply flagging only for non-NaN ages that match bracket
                    if mask.any():
                        bracket_flags = (df.loc[mask, col] < bracket.min) | (
                            df.loc[mask, col] > bracket.max
                        )
                        flags.loc[mask] = bracket_flags
            results[col] = flags.rename(col)
        return results
