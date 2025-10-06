"""
Base detector class for all BIV detection methods.
"""

from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict


class BaseDetector(ABC):
    """
    Abstract base class for all BIV detection methods.

    Each detection method should inherit from this class and implement the `detect` and
    `validate_config` methods. The base class provides common validation helpers.

    Example subclass implementation:
        class RangeDetector(BaseDetector):
            def __init__(self, min_val: float = 0, max_val: float = 100):
                self.min_val = min_val
                self.max_val = max_val
                self.validate_config()

            def validate_config(self) -> None:
                if self.min_val >= self.max_val:
                    raise ValueError("min_val must be less than max_val")

            def detect(self, df: pd.DataFrame, columns: list[str]) -> Dict[str, pd.Series]:
                # Implementation here
                return {}
    """

    @abstractmethod
    def detect(self, df: pd.DataFrame, columns: list[str]) -> Dict[str, pd.Series]:
        """
        Detect BIVs in the specified columns of the DataFrame.

        Args:
            df: Input DataFrame.
            columns: List of column names to check for BIVs.

        Returns:
            Dictionary mapping column names to boolean Series indicating BIV flags.
        """
        pass

    @abstractmethod
    def validate_config(self) -> None:
        """
        Validate method-specific configurations.

        Raises:
            ValueError: If configuration is invalid.
        """
        pass

    def _validate_column(self, df: pd.DataFrame, column: str) -> None:
        """
        Validate that a column exists in the DataFrame.

        Args:
            df: DataFrame to validate.
            column: Column name to check.

        Raises:
            ValueError: If column does not exist.
        """
        if column not in df.columns:
            raise ValueError(f"Column '{column}' does not exist in DataFrame")
