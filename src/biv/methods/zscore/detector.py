# ZScoreDetector class

"""
Z-score based detection method for pediatric BIV detection.

Implements detection of Biologically Implausible Values using WHO/CDC growth standards
and z-score thresholds for age- and sex-specific anthropometric measurements.
"""

from typing import Dict, Optional, List
import logging
import numpy as np
import pandas as pd
from pydantic import BaseModel, field_validator, ValidationError

from ..base import BaseDetector
from ...zscores import calculate_growth_metrics


class ZScoreConfig(BaseModel):
    """
    Configuration for Z-score based BIV detection.

    Provides type-safe configuration with validation for column names and parameters
    specific to z-score based methods supporting WHO/CDC growth standards.

    Attributes:
        age_col (str): Name of age column in months ('age' by default). Ages >241mo flagged as NaN.
        sex_col (str): Name of sex column ('sex' by default). Expected values: 'M', 'F'.
        head_circ_col (Optional[str]): Name of head circumference column if present. None by default.
        validate_age_units (bool): Enable validation warning for potential age unit issues (ages <130mo suggesting years instead of months). True by default.
    """

    age_col: str = "age"
    sex_col: str = "sex"
    head_circ_col: Optional[str] = None
    validate_age_units: bool = True

    @field_validator("age_col", "sex_col")
    @classmethod
    def validate_column_names(cls, v: str) -> str:
        """Ensure column names are valid string identifiers."""
        if not isinstance(v, str) or not v.strip():
            raise ValueError("Column name must be a non-empty string")
        return v

    @field_validator("head_circ_col")
    @classmethod
    def validate_optional_column(cls, v: Optional[str]) -> Optional[str]:
        """Ensure optional column name is valid if provided."""
        if v is not None and (not isinstance(v, str) or not v.strip()):
            raise ValueError("Head circumference column name must be a valid string")
        return v


# Column-to-measure mapping per README
COLUMN_MEASURE_MAPPING = {
    "weight_kg": "waz",  # Weight-for-age z-scores → _bivwaz flags
    "height_cm": "haz",  # Height-for-age z-scores → _bivhaz flags
    "bmi": "bmiz",  # BMI-for-age z-scores → _bivbmi flags
    "head_circ_cm": "headcz",  # Head circumference-for-age z-scores → _bivheadcz flags
}

MEASURE_BIV_FLAG_MAPPING = {
    "waz": "_bivwaz",  # WAZ: z < -5 or z > 8
    "haz": "_bivhaz",  # HAZ: z < -5 or z > 4
    "bmiz": "_bivbmi",  # BMIz: z < -4 or z > 8
    "headcz": "_bivheadcz",  # HEADCZ: z < -5 or z > 5
}


class ZScoreDetector(BaseDetector):
    """
    Z-score based detector for Biologically Implausible Values (BIV).

    Detects BIVs using WHO/CDC growth standards and modified z-score thresholds:
    - Weight-for-age z-scores: < -5 or > 8
    - Height-for-age z-scores: < -5 or > 4
    - BMI-for-age z-scores: < -4 or > 8
    - Head circumference-for-age z-scores: < -5 or > 5

    Z-scores calculated using LMS method with WHO data (<24 months) and CDC data (≥24 months).
    Modified z-scores used for asymmetric BIV detection per CDC SAS program methodology.

    Usage:
        detector = ZScoreDetector(age_col='visit_age', sex_col='gender')
        flags = detector.detect(df, ['weight_kg', 'height_cm'])

    Attributes:
        config (ZScoreConfig): Configuration object with column mappings and validation settings.
    """

    def __init__(
        self,
        age_col: str = "age",
        sex_col: str = "sex",
        head_circ_col: Optional[str] = None,
        validate_age_units: bool = True,
    ):
        """
        Initialize ZScoreDetector with configurable column mappings.

        Args:
            age_col: DataFrame column containing age in months
            sex_col: DataFrame column containing sex ('M' or 'F')
            head_circ_col: DataFrame column containing head circumference in cm (optional)
            validate_age_units: Whether to warn about potential age unit issues

        Raises:
            ValueError: If configuration is invalid per ZScoreConfig validation
        """
        try:
            self.config = ZScoreConfig(
                age_col=age_col,
                sex_col=sex_col,
                head_circ_col=head_circ_col,
                validate_age_units=validate_age_units,
            )
        except ValidationError as e:
            raise ValueError(f"Invalid configuration: {e}") from e

        self.validate_config()

    def validate_config(self) -> None:
        """
        Validate detector configuration.

        Performs additional validation beyond Pydantic:
        - Ensures configuration doesn't have conflicting column names
        - Checks for reasonable column name patterns

        Raises:
            ValueError: If configuration violates BIV architectural constraints
        """
        # Basic validation: ensure different columns don't share same name
        columns = [self.config.age_col, self.config.sex_col]
        if self.config.head_circ_col is not None:
            columns.append(self.config.head_circ_col)

        if len(columns) != len(set(columns)):
            raise ValueError("Configuration must specify unique column names")

        # Age validation warnings if enabled
        if self.config.validate_age_units and self.config.age_col == "age":
            # Default column name suggests months, but we'll check later in detect()
            pass

    def detect(self, df: pd.DataFrame, columns: List[str]) -> Dict[str, pd.Series]:
        """
        Detect BIVs in specified columns using z-score thresholds.

        For each requested column, extracts corresponding anthropometric data,
        computes growth metrics via calculate_growth_metrics(), and returns
        BIV flags based on modified z-score thresholds.

        Supported columns and mappings:
        - 'weight_kg' → Weight-for-age BIV flags (_bivwaz)
        - 'height_cm' → Height-for-age BIV flags (_bivhaz)
        - 'bmi' → BMI-for-age BIV flags (_bivbmi)
        - 'head_circ_cm' → Head circumference-for-age BIV flags (_bivheadcz)

        Args:
            df: Input DataFrame with anthropometric measurements
            columns: List of column names to check for BIVs

        Returns:
            Dict mapping column names to boolean Series of BIV flags:
            {column: pd.Series(bool)} where True indicates BIV

        Raises:
            ValueError: If unsupported column requested or required columns missing
        """
        # Step 1: Validate required columns exist
        self._validate_required_columns(df)

        # Step 2: Validate age units if enabled
        if self.config.validate_age_units:
            self._validate_age_units(df)

        # Step 3: Build column data for calculate_growth_metrics
        column_data = self._extract_column_data(df, columns)

        # Step 4: Call calculate_growth_metrics to get BIV flags
        results = {}
        try:
            # Extract arrays for calculate_growth_metrics
            agemos = np.asarray(
                df[self.config.age_col].fillna(np.nan), dtype=np.float64
            )
            sex = np.asarray(
                df[self.config.sex_col].str.upper().fillna(""), dtype=str
            )  # Normalize to uppercase, ensure ndarray

            metrics = calculate_growth_metrics(
                agemos=agemos, sex=sex, measures=None, **column_data
            )

            # Step 5: Extract BIV flags for each requested column
            for col in columns:
                measure = COLUMN_MEASURE_MAPPING[col]
                biv_key = MEASURE_BIV_FLAG_MAPPING[measure]

                # Get BIV flags, default to False if missing
                biv_flags = metrics.get(biv_key, np.full(len(df), False, dtype=bool))

                # Ensure proper boolean Series with index alignment
                results[col] = pd.Series(biv_flags, index=df.index, name=col)

        except Exception as e:
            # Re-raise with more context about the detection failure
            raise ValueError(f"Z-score detection failed: {e}") from e

        return results

    def _validate_required_columns(self, df: pd.DataFrame) -> None:
        """
        Validate that required columns exist in DataFrame.

        Args:
            df: Input DataFrame to validate

        Raises:
            ValueError: If age or sex columns are missing
        """
        self._validate_column(df, self.config.age_col)
        self._validate_column(df, self.config.sex_col)

    def _validate_age_units(self, df: pd.DataFrame) -> None:
        """
        Validate age units and warn about potential issues.

        Following documentation, warns if mean age < 130 months which suggests
        ages might be in years instead of months (common error in pediatric data).

        Args:
            df: Input DataFrame with age data
        """
        if not self.config.validate_age_units:
            return

        try:
            ages = pd.to_numeric(df[self.config.age_col], errors="coerce")
            mean_age = ages.mean()
            max_age = ages.max()

            # Warn if ages suggest years instead of months
            if pd.notna(mean_age) and mean_age < 130:
                logging.warning(
                    f"Mean age {mean_age:.1f} months suggests ages may be in years "
                    f"instead of months. Z-score calculations require ages in months."
                )

            # Also warn for extremely high ages
            if pd.notna(max_age) and max_age > 240:
                logging.warning(
                    f"Maximum age {max_age:.1f} months exceeds CDC reference limit (240 months). "
                    "Z-scores will be NaN for ages >241 months."
                )

        except Exception:
            # If we can't validate, don't fail - let downstream handle invalid data
            pass

    def _extract_column_data(
        self, df: pd.DataFrame, columns: List[str]
    ) -> Dict[str, np.ndarray]:
        """
        Extract anthropometric data for calculate_growth_metrics from requested columns.

        Maps column names to expected parameter names for calculate_growth_metrics().

        Args:
            df: Input DataFrame
            columns: List of column names requested

        Returns:
            Dict with 'height', 'weight', and 'head_circ' arrays (None if not requested)

        Raises:
            ValueError: If unsupported column requested
        """
        column_data = {}

        for col in columns:
            if col not in COLUMN_MEASURE_MAPPING:
                supported = list(COLUMN_MEASURE_MAPPING.keys())
                raise ValueError(
                    f"Unsupported column '{col}' for z-score detection. "
                    f"Supported columns: {supported}"
                )

            # For BIV flag extraction, we need the measure data
            # Map column names to calculate_growth_metrics parameter names
            if col == "weight_kg":
                column_data["weight"] = np.asarray(
                    df[col].fillna(np.nan), dtype=np.float64
                )
            elif col == "height_cm":
                column_data["height"] = np.asarray(
                    df[col].fillna(np.nan), dtype=np.float64
                )
            elif col == "bmi":
                # BMI column directly
                column_data["bmi"] = np.asarray(
                    df[col].fillna(np.nan), dtype=np.float64
                )
            elif col == "head_circ_cm":
                column_data["head_circ"] = np.asarray(
                    df[col].fillna(np.nan), dtype=np.float64
                )

        return column_data
