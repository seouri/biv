from typing import Any, Dict

import pandas as pd
from pydantic import BaseModel, ValidationError, field_validator, StrictFloat

from biv.methods.base import BaseDetector


class RangeConfig(BaseModel):
    min_: StrictFloat
    max_: StrictFloat

    @field_validator("min_", "max_", mode="after")
    @classmethod
    def validate_min_max(cls, v: float, info: Any) -> float:
        field_name = info.field_name
        if field_name == "min_":
            min_val = v
            max_val = info.data.get("max_")
            if max_val is not None and min_val >= max_val:
                raise ValueError("min_ must be less than max_")
        elif field_name == "max_":
            min_val = info.data.get("min_")
            if min_val is not None and min_val >= v:
                raise ValueError("min_ must be less than max_")
        return v


class RangeDetector(BaseDetector):
    def __init__(self, config: Dict[str, Dict[str, float]]):
        # Validate config using Pydantic
        self.config = {}
        for col, params in config.items():
            try:
                self.config[col] = RangeConfig(min_=params["min"], max_=params["max"])
            except (KeyError, ValidationError) as e:
                raise ValueError(f"Invalid config for column '{col}': {e}") from e
        self.validate_config()

    def validate_config(self) -> None:
        # Additional validation if needed; pydantic handles main validation
        pass

    def detect(self, df: pd.DataFrame, columns: list[str]) -> Dict[str, pd.Series]:
        results = {}
        for col in columns:
            self._validate_column(df, col)
            if col not in self.config:
                raise ValueError(f"No range config provided for column '{col}'")
            min_val = self.config[col].min_
            max_val = self.config[col].max_
            series = df[col]
            flags = (series < min_val) | (series > max_val)
            results[col] = flags.rename(col)
        return results
