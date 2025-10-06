# Tests for RangeDetector

import pytest
import pandas as pd
import numpy as np
from biv.methods.range.detector import RangeDetector


class TestRangeDetector:
    """Tests for RangeDetector class"""

    @pytest.fixture
    def sample_df(self) -> pd.DataFrame:  # type: ignore
        """Sample DataFrame for testing"""
        return pd.DataFrame(
            {
                "weight_kg": [60.0, 70.0, 150.0, np.nan, 75.0],
                "height_cm": [150.0, 200.0, 250.0, 180.0, np.nan],
            }
        )

    def test_tc001_detect_flags_out_of_range(self, sample_df: pd.DataFrame) -> None:
        # Invalid values: 150 > 120 (max), np.nan (should be False)
        # 150 out of 50-120? Wait, let's define config
        config = {"weight_kg": {"min": 50.0, "max": 120.0}}  # type: ignore
        detector = RangeDetector(config)  # type: ignore
        result = detector.detect(sample_df, ["weight_kg"])
        expected = pd.Series([False, False, True, False, False], name="weight_kg")
        pd.testing.assert_series_equal(result["weight_kg"], expected)

    def test_tc002_detect_nan_not_flagged(self, sample_df: pd.DataFrame) -> None:
        # Only NaN, should return False
        config = {"height_cm": {"min": 100.0, "max": 220.0}}  # type: ignore
        detector = RangeDetector(config)  # type: ignore
        result = detector.detect(sample_df, ["height_cm"])
        expected = pd.Series([False, False, True, False, False], name="height_cm")
        pd.testing.assert_series_equal(result["height_cm"], expected)

    def test_tc003_detect_multi_column(self, sample_df: pd.DataFrame) -> None:
        config = {  # type: ignore
            "weight_kg": {"min": 50.0, "max": 120.0},
            "height_cm": {"min": 100.0, "max": 220.0},
        }
        detector = RangeDetector(config)  # type: ignore
        result = detector.detect(sample_df, ["weight_kg", "height_cm"])
        assert "weight_kg" in result
        assert "height_cm" in result
        assert len(result) == 2

    def test_tc004_config_valid_min_lt_max(self) -> None:
        config = {"col": {"min": 10.0, "max": 20.0}}  # type: ignore
        # Should not raise
        detector = RangeDetector(config)  # type: ignore
        assert detector is not None

    def test_tc005_config_invalid_min_gt_max(self) -> None:
        config = {"col": {"min": 20.0, "max": 10.0}}  # type: ignore
        with pytest.raises(ValueError):
            RangeDetector(config)  # type: ignore

    def test_tc006_config_missing_min(self) -> None:
        config = {"col": {"max": 20.0}}  # type: ignore
        with pytest.raises(Exception):  # Pydantic ValidationError
            RangeDetector(config)  # type: ignore

    def test_tc007_config_missing_max(self) -> None:
        config = {"col": {"min": 10.0}}  # type: ignore
        with pytest.raises(Exception):
            RangeDetector(config)  # type: ignore

    def test_tc008_config_non_float_values(self) -> None:
        config = {"col": {"min": "10", "max": 20.0}}  # type: ignore
        with pytest.raises(Exception):
            RangeDetector(config)  # type: ignore

    def test_tc009_detect_lower_bound_violation(self) -> None:
        df = pd.DataFrame({"col": [5.0]})  # type: ignore
        config = {"col": {"min": 10.0, "max": 100.0}}  # type: ignore
        detector = RangeDetector(config)  # type: ignore
        result = detector.detect(df, ["col"])
        expected = pd.Series([True], name="col")
        pd.testing.assert_series_equal(result["col"], expected)

    def test_tc010_detect_upper_bound_violation(self) -> None:
        df = pd.DataFrame({"col": [150.0]})  # type: ignore
        config = {"col": {"min": 10.0, "max": 100.0}}  # type: ignore
        detector = RangeDetector(config)  # type: ignore
        result = detector.detect(df, ["col"])
        expected = pd.Series([True], name="col")
        pd.testing.assert_series_equal(result["col"], expected)

    def test_tc011_zero_value_in_range(self) -> None:
        df = pd.DataFrame({"col": [0.0]})  # type: ignore
        config = {"col": {"min": -1.0, "max": 1.0}}  # type: ignore
        detector = RangeDetector(config)  # type: ignore
        result = detector.detect(df, ["col"])
        expected = pd.Series([False], name="col")
        pd.testing.assert_series_equal(result["col"], expected)

    def test_tc012_very_large_value(self) -> None:
        df = pd.DataFrame({"col": [1e10]})  # type: ignore
        config = {"col": {"min": 0.0, "max": 100.0}}  # type: ignore
        detector = RangeDetector(config)  # type: ignore
        result = detector.detect(df, ["col"])
        expected = pd.Series([True], name="col")
        pd.testing.assert_series_equal(result["col"], expected)

    def test_tc013_very_small_value(self) -> None:
        df = pd.DataFrame({"col": [-1e10]})  # type: ignore
        config = {"col": {"min": 0.0, "max": 100.0}}  # type: ignore
        detector = RangeDetector(config)  # type: ignore
        result = detector.detect(df, ["col"])
        expected = pd.Series([True], name="col")
        pd.testing.assert_series_equal(result["col"], expected)

    def test_tc014_empty_dataframe(self) -> None:
        df = pd.DataFrame({"col": []})  # type: ignore
        config = {"col": {"min": 0.0, "max": 100.0}}  # type: ignore
        detector = RangeDetector(config)  # type: ignore
        result = detector.detect(df, ["col"])
        expected = pd.Series([], dtype=bool, name="col")
        pd.testing.assert_series_equal(result["col"], expected)

    def test_tc015_detect_does_not_modify_dataframe(
        self, sample_df: pd.DataFrame
    ) -> None:
        df_original = sample_df.copy()
        config = {"weight_kg": {"min": 50.0, "max": 120.0}}  # type: ignore
        detector = RangeDetector(config)  # type: ignore
        detector.detect(sample_df, ["weight_kg"])
        pd.testing.assert_frame_equal(sample_df, df_original)
