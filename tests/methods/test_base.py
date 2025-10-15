# Tests for BaseDetector

import pytest
import pandas as pd
import numpy as np
from biv.methods.base import BaseDetector
from typing import Dict


def test_tc001_instantiating_base_detector_raises_type_error():
    with pytest.raises(TypeError):
        BaseDetector()  # type: ignore[abstract]


def test_tc002_subclass_without_detect_raises_type_error():
    class Concrete(BaseDetector):
        pass

    with pytest.raises(TypeError):
        Concrete()  # type: ignore[abstract]


def test_tc003_validate_column_passes_for_existing_column():
    class Concrete(BaseDetector):
        def validate_config(self):
            pass

        def detect(self, df: pd.DataFrame, columns: list[str]) -> Dict[str, pd.Series]:
            self._validate_column(df, "existing_col")
            return {}

    detector = Concrete()
    df = pd.DataFrame({"existing_col": [1, 2, 3], "other": [4, 5, 6]})
    try:
        detector.detect(df, [])
    except ValueError:
        pytest.fail("Validation should not raise ValueError for existing column")


def test_tc004_detect_validates_multiple_columns():
    class Concrete(BaseDetector):
        def validate_config(self):
            pass

        def detect(self, df: pd.DataFrame, columns: list[str]) -> Dict[str, pd.Series]:
            for col in columns:
                self._validate_column(df, col)
            return {}

    detector = Concrete()
    df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})

    try:
        detector.detect(df, ["col1", "col2"])
    except ValueError:
        pytest.fail("Should not raise for existing columns")

    with pytest.raises(ValueError):
        detector.detect(df, ["col1", "missing"])


def test_tc005_initialization_accepts_method_specific_configs():
    class Concrete(BaseDetector):
        def __init__(self, min_val: int = 0, max_val: int = 100):
            super().__init__()
            self.min_val = min_val
            self.max_val = max_val

        def validate_config(self):
            pass

        def detect(self, df: pd.DataFrame, columns: list[str]) -> Dict[str, pd.Series]:
            return {}

    detector = Concrete(min_val=10, max_val=200)
    assert detector.min_val == 10
    assert detector.max_val == 200


def test_tc006_detect_validates_multiple_columns():
    class Concrete(BaseDetector):
        def __init__(self, min_val: int = 0, max_val: int = 100):
            super().__init__()
            self.min_val = min_val
            self.max_val = max_val
            self.validate_config()

        def validate_config(self):
            if self.min_val >= self.max_val:
                raise ValueError("min_val must be less than max_val")

        def detect(self, df: pd.DataFrame, columns: list[str]) -> Dict[str, pd.Series]:
            return {}

    # Valid config should not raise
    Concrete(min_val=10, max_val=200)

    with pytest.raises(ValueError, match="min_val must be less than max_val"):
        Concrete(min_val=50, max_val=50)  # invalid


def test_detect_does_not_modify_input_dataframe():
    class Concrete(BaseDetector):
        def validate_config(self):
            pass

        def detect(self, df: pd.DataFrame, columns: list[str]) -> Dict[str, pd.Series]:
            return {
                col: pd.Series([False] * len(df), index=df.index) for col in columns
            }

    detector = Concrete()
    original_df = pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})
    copy_df = original_df.copy(deep=True)

    detector.detect(original_df, ["col1", "col2"])  # Call but no result needed

    pd.testing.assert_frame_equal(original_df, copy_df)  # df unchanged


def test_detect_handles_nan_values_appropriately():
    class Concrete(BaseDetector):
        def validate_config(self):
            pass

        def detect(self, df: pd.DataFrame, columns: list[str]) -> Dict[str, pd.Series]:
            result = {}
            for col in columns:
                self._validate_column(df, col)
                series = df[col]
                # For simplicity, flag True If > 10, False if NaN or fine
                result[col] = series.apply(
                    lambda x: x if pd.notna(x) and x > 10 else False
                )
            return result

    detector = Concrete()
    df = pd.DataFrame({"col1": [5, np.nan, 15], "col2": [7, 20, np.nan]})
    result = detector.detect(df, ["col1", "col2"])

    expected_col1 = pd.Series([False, False, 15], index=df.index, name="col1")
    expected_col2 = pd.Series([False, 20, False], index=df.index, name="col2")

    pd.testing.assert_series_equal(result["col1"], expected_col1)
    pd.testing.assert_series_equal(result["col2"], expected_col2)


def test_detect_handles_empty_dataframe():
    class Concrete(BaseDetector):
        def validate_config(self):
            pass

        def detect(self, df: pd.DataFrame, columns: list[str]) -> Dict[str, pd.Series]:
            return dict()

    detector = Concrete()
    df = pd.DataFrame()
    result = detector.detect(df, [])
    assert isinstance(result, dict)
    assert len(result) == 0


def test_detect_handles_single_row_dataframe():
    class Concrete(BaseDetector):
        def validate_config(self):
            pass

        def detect(self, df: pd.DataFrame, columns: list[str]) -> Dict[str, pd.Series]:
            return {col: pd.Series([False], index=df.index) for col in columns}

    detector = Concrete()
    df = pd.DataFrame({"col1": [5]})
    result = detector.detect(df, ["col1"])
    assert len(result["col1"]) == 1
    assert result["col1"].iloc[0] is np.False_  # or == False


def test_detect_handles_non_numeric_columns():
    class Concrete(BaseDetector):
        def validate_config(self):
            pass

        def detect(self, df: pd.DataFrame, columns: list[str]) -> Dict[str, pd.Series]:
            return {col: pd.Series(["flag"], index=df.index) for col in columns}

    detector = Concrete()
    df = pd.DataFrame({"col1": ["a"]})
    result = detector.detect(df, ["col1"])
    assert len(result["col1"]) == 1
    assert result["col1"].iloc[0] == "flag"


def test_validate_column_raises_for_nonexistent_column():
    class Concrete(BaseDetector):
        def validate_config(self):
            pass

        def detect(self, df: pd.DataFrame, columns: list[str]) -> Dict[str, pd.Series]:
            return {}

    detector = Concrete()
    df = pd.DataFrame({"existing_col": [1, 2, 3]})

    with pytest.raises(
        ValueError, match="Column 'missing_col' does not exist in DataFrame"
    ):
        detector._validate_column(df, "missing_col")
