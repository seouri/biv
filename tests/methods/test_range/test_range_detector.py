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

    def test_tc001_detect_flags_out_of_range(self, sample_df: pd.DataFrame):
        # Invalid values: 150 > 120 (max), np.nan (should be False)
        # 150 out of 50-120? Wait, let's define config
        config = {"weight_kg": {"min": 50.0, "max": 120.0}}  # type: ignore
        detector = RangeDetector(config)  # type: ignore
        result = detector.detect(sample_df, ["weight_kg"])
        expected = pd.Series([False, False, True, False, False], name="weight_kg")
        pd.testing.assert_series_equal(result["weight_kg"], expected)

    def test_tc002_detect_nan_not_flagged(self, sample_df: pd.DataFrame):
        # Only NaN, should return False
        config = {"height_cm": {"min": 100.0, "max": 220.0}}  # type: ignore
        detector = RangeDetector(config)  # type: ignore
        result = detector.detect(sample_df, ["height_cm"])
        expected = pd.Series([False, False, True, False, False], name="height_cm")
        pd.testing.assert_series_equal(result["height_cm"], expected)

    def test_tc003_detect_multi_column(self, sample_df: pd.DataFrame):
        config = {  # type: ignore
            "weight_kg": {"min": 50.0, "max": 120.0},
            "height_cm": {"min": 100.0, "max": 220.0},
        }
        detector = RangeDetector(config)  # type: ignore
        result = detector.detect(sample_df, ["weight_kg", "height_cm"])
        assert "weight_kg" in result
        assert "height_cm" in result
        assert len(result) == 2

    def test_tc004_config_valid_min_lt_max(self):
        config = {"col": {"min": 10.0, "max": 20.0}}  # type: ignore
        # Should not raise
        detector = RangeDetector(config)  # type: ignore
        assert detector is not None

    def test_tc005_config_invalid_min_gt_max(self):
        config = {"col": {"min": 20.0, "max": 10.0}}  # type: ignore
        with pytest.raises(ValueError):
            RangeDetector(config)  # type: ignore

    def test_tc006_config_missing_min(self):
        config = {"col": {"max": 20.0}}  # type: ignore
        with pytest.raises(Exception):  # Pydantic ValidationError
            RangeDetector(config)  # type: ignore

    def test_tc007_config_missing_max(self):
        config = {"col": {"min": 10.0}}  # type: ignore
        with pytest.raises(Exception):
            RangeDetector(config)  # type: ignore

    def test_tc008_config_non_float_values(self):
        config = {"col": {"min": "10", "max": 20.0}}  # type: ignore
        with pytest.raises(Exception):
            RangeDetector(config)  # type: ignore

    def test_tc009_detect_lower_bound_violation(self):
        df = pd.DataFrame({"col": [5.0]})  # type: ignore
        config = {"col": {"min": 10.0, "max": 100.0}}  # type: ignore
        detector = RangeDetector(config)  # type: ignore
        result = detector.detect(df, ["col"])
        expected = pd.Series([True], name="col")
        pd.testing.assert_series_equal(result["col"], expected)

    def test_tc010_detect_upper_bound_violation(self):
        df = pd.DataFrame({"col": [150.0]})  # type: ignore
        config = {"col": {"min": 10.0, "max": 100.0}}  # type: ignore
        detector = RangeDetector(config)  # type: ignore
        result = detector.detect(df, ["col"])
        expected = pd.Series([True], name="col")
        pd.testing.assert_series_equal(result["col"], expected)

    def test_tc011_zero_value_in_range(self):
        df = pd.DataFrame({"col": [0.0]})  # type: ignore
        config = {"col": {"min": -1.0, "max": 1.0}}  # type: ignore
        detector = RangeDetector(config)  # type: ignore
        result = detector.detect(df, ["col"])
        expected = pd.Series([False], name="col")
        pd.testing.assert_series_equal(result["col"], expected)

    def test_tc012_very_large_value(self):
        df = pd.DataFrame({"col": [1e10]})  # type: ignore
        config = {"col": {"min": 0.0, "max": 100.0}}  # type: ignore
        detector = RangeDetector(config)  # type: ignore
        result = detector.detect(df, ["col"])
        expected = pd.Series([True], name="col")
        pd.testing.assert_series_equal(result["col"], expected)

    def test_tc013_very_small_value(self):
        df = pd.DataFrame({"col": [-1e10]})  # type: ignore
        config = {"col": {"min": 0.0, "max": 100.0}}  # type: ignore
        detector = RangeDetector(config)  # type: ignore
        result = detector.detect(df, ["col"])
        expected = pd.Series([True], name="col")
        pd.testing.assert_series_equal(result["col"], expected)

    def test_tc014_empty_dataframe(self):
        df = pd.DataFrame({"col": []})  # type: ignore
        config = {"col": {"min": 0.0, "max": 100.0}}  # type: ignore
        detector = RangeDetector(config)  # type: ignore
        result = detector.detect(df, ["col"])
        expected = pd.Series([], dtype=bool, name="col")
        pd.testing.assert_series_equal(result["col"], expected)

    def test_tc015_detect_does_not_modify_dataframe(self, sample_df: pd.DataFrame):
        df_original = sample_df.copy()
        config = {"weight_kg": {"min": 50.0, "max": 120.0}}  # type: ignore
        detector = RangeDetector(config)  # type: ignore
        detector.detect(sample_df, ["weight_kg"])
        pd.testing.assert_frame_equal(sample_df, df_original)

    def test_tc016_detect_upper_bound_exclusive(self):
        """Test that upper bound is exclusive: values equal to max are not flagged."""
        df = pd.DataFrame({"col": [50.0, 100.0, 150.0]})  # type: ignore
        config = {"col": {"min": 0.0, "max": 100.0}}  # type: ignore
        detector = RangeDetector(config)  # type: ignore
        result = detector.detect(df, ["col"])
        # 50.0 is in range, 100.0 == max (not flagged), 150.0 > max (flagged)
        expected = pd.Series([False, False, True], name="col")
        pd.testing.assert_series_equal(result["col"], expected)

    def test_tc017_detect_raises_value_error_column_not_in_df(self):
        config = {"col": {"min": 10.0, "max": 100.0}}  # type: ignore
        detector = RangeDetector(config)  # type: ignore
        df = pd.DataFrame({"other_col": [1.0]})  # type: ignore
        with pytest.raises(
            ValueError, match="Column 'col' does not exist in DataFrame"
        ):
            detector.detect(df, ["col"])

    def test_tc018_detect_raises_value_error_missing_config_for_column(self):
        config = {"existing": {"min": 10.0, "max": 100.0}}  # type: ignore
        detector = RangeDetector(config)  # type: ignore
        df = pd.DataFrame({"existing": [1.0], "missing": [2.0]})  # type: ignore
        with pytest.raises(
            ValueError, match="No valid range configuration found for column 'missing'"
        ):
            detector.detect(df, ["existing", "missing"])

    def test_tc019_init_raises_value_error_config_not_dict(self):
        with pytest.raises(ValueError):
            RangeDetector("invalid")  # type: ignore

    def test_tc020_init_raises_value_error_column_config_not_dict(self):
        config = {"col": "invalid"}  # type: ignore
        with pytest.raises(ValueError, match="Config for column 'col' must be a dict"):
            RangeDetector(config)  # type: ignore

    def test_tc021_detect_all_nan_values(self):
        df = pd.DataFrame({"col": [np.nan, np.nan]})  # type: ignore
        config = {"col": {"min": 0.0, "max": 100.0}}  # type: ignore
        detector = RangeDetector(config)  # type: ignore
        result = detector.detect(df, ["col"])
        expected = pd.Series([False, False], name="col")
        pd.testing.assert_series_equal(result["col"], expected)

    def test_tc022_detect_empty_columns_list(self):
        df = pd.DataFrame({"col": [10.0]})  # type: ignore
        config = {"col": {"min": 0.0, "max": 100.0}}  # type: ignore
        detector = RangeDetector(config)  # type: ignore
        result = detector.detect(df, [])
        assert result == {}

    def test_tc023_detect_values_at_exact_lower_bound(self):
        df = pd.DataFrame({"col": [10.0]})  # type: ignore
        config = {"col": {"min": 10.0, "max": 100.0}}  # type: ignore
        detector = RangeDetector(config)  # type: ignore
        result = detector.detect(df, ["col"])
        expected = pd.Series([False], name="col")
        pd.testing.assert_series_equal(result["col"], expected)

    def test_tc024_detect_negative_ranges_and_values(self):
        df = pd.DataFrame({"col": [-5.0, 10.0]})  # type: ignore
        config = {"col": {"min": -10.0, "max": 20.0}}  # type: ignore
        detector = RangeDetector(config)  # type: ignore
        result = detector.detect(df, ["col"])
        expected = pd.Series([False, False], name="col")
        pd.testing.assert_series_equal(result["col"], expected)

    def test_tc025_detect_integer_values_in_df(self):
        df = pd.DataFrame({"col": [10, 20]})  # type: ignore
        config = {"col": {"min": 15.0, "max": 25.0}}  # type: ignore
        detector = RangeDetector(config)  # type: ignore
        result = detector.detect(df, ["col"])
        expected = pd.Series([True, False], name="col")
        pd.testing.assert_series_equal(result["col"], expected)

    def test_tc026_age_dependent_range_applies_different_min_max_per_age(self):
        df = pd.DataFrame(
            {
                "age": [5, 15, 25],
                "weight_kg": [35.0, 60.0, 70.0],  # type: ignore
            }
        )
        config = {
            "age_col": "age",
            "weight_kg": {
                "age_brackets": [
                    {"min_age": 0, "max_age": 10, "min": 30.0, "max": 50.0},
                    {"min_age": 10, "max_age": 20, "min": 50.0, "max": 80.0},
                    {"min_age": 20, "max_age": 30, "min": 60.0, "max": 100.0},
                ]
            },
        }  # type: ignore
        detector = RangeDetector(config)  # type: ignore
        result = detector.detect(df, ["weight_kg"])
        # age 5: 35 in 30-50 -> False, age 15: 60 in 50-80 -> False, age 25: 70 in 60-100 -> False
        expected = pd.Series([False, False, False], name="weight_kg")
        pd.testing.assert_series_equal(result["weight_kg"], expected)

    def test_tc027_age_dependent_range_raises_error_for_invalid_brackets_overlapping(
        self,
    ):
        config = {
            "age_col": "age",
            "weight_kg": {
                "age_brackets": [
                    {"min_age": 0, "max_age": 15, "min": 30.0, "max": 50.0},
                    {
                        "min_age": 10,
                        "max_age": 20,
                        "min": 50.0,
                        "max": 80.0,
                    },  # overlaps
                ]
            },
        }  # type: ignore
        with pytest.raises(ValueError, match="Overlapping age brackets"):
            RangeDetector(config)  # type: ignore

    def test_tc033_age_dependent_range_flags_based_on_age_specific_ranges(self):
        df = pd.DataFrame(
            {
                "age": [5, 15, 25],
                "weight_kg": [25.0, 90.0, 120.0],  # type: ignore
            }
        )
        config = {
            "age_col": "age",
            "weight_kg": {
                "age_brackets": [
                    {"min_age": 0, "max_age": 10, "min": 30.0, "max": 50.0},
                    {"min_age": 10, "max_age": 20, "min": 50.0, "max": 80.0},
                    {"min_age": 20, "max_age": 30, "min": 60.0, "max": 100.0},
                ]
            },
        }  # type: ignore
        detector = RangeDetector(config)  # type: ignore
        result = detector.detect(df, ["weight_kg"])
        # age 5: 25 < 30 -> True, age 15: 90 > 80 -> True, age 25: 120 > 100 -> True
        expected = pd.Series([True, True, True], name="weight_kg")
        pd.testing.assert_series_equal(result["weight_kg"], expected)

    def test_tc034_age_dependent_range_missing_age_col_raises(self):
        df = pd.DataFrame(
            {
                "no_age": [5],
                "weight_kg": [35.0],  # type: ignore
            }
        )
        config = {
            "age_col": "age",
            "weight_kg": {
                "age_brackets": [
                    {"min_age": 0, "max_age": 10, "min": 30.0, "max": 50.0}
                ]
            },
        }  # type: ignore
        detector = RangeDetector(config)  # type: ignore
        with pytest.raises(
            ValueError, match="Age column 'age' does not exist in DataFrame"
        ):
            detector.detect(df, ["weight_kg"])

    def test_tc035_age_dependent_range_works_with_nan_age(self):
        df = pd.DataFrame(
            {
                "age": [5, np.nan],
                "weight_kg": [35.0, 60.0],  # type: ignore
            }
        )
        config = {
            "age_col": "age",
            "weight_kg": {
                "age_brackets": [
                    {"min_age": 0, "max_age": 10, "min": 30.0, "max": 50.0},
                    {"min_age": 10, "max_age": 20, "min": 50.0, "max": 80.0},
                ]
            },
        }  # type: ignore
        detector = RangeDetector(config)  # type: ignore
        result = detector.detect(df, ["weight_kg"])
        # For NaN age, flag as False (not flagged)
        expected = pd.Series([False, False], name="weight_kg")
        pd.testing.assert_series_equal(result["weight_kg"], expected)

    def test_tc036_config_validation_raises_for_age_config_with_min_max_keys(
        self,
    ):
        config = {
            "age_col": "age",
            "weight_kg": {
                "age_brackets": [
                    {"min_age": 0, "max_age": 10, "min": 30.0, "max": 50.0}
                ],
                "min": 30.0,  # Invalid for age-dependent
                "max": 200.0,
            },
        }  # type: ignore
        with pytest.raises(ValueError, match="should not have 'min' or 'max' keys"):
            RangeDetector(config)  # type: ignore

    def test_tc037_config_validation_raises_for_flat_config_with_age_brackets_key(
        self,
    ):
        config = {
            "weight_kg": {
                "min": 30.0,
                "max": 200.0,
                "age_brackets": [],  # Invalid for flat range
            },
        }  # type: ignore
        with pytest.raises(ValueError, match="should not have 'age_brackets' key"):
            RangeDetector(config)  # type: ignore

    def test_tc038_config_validation_raises_for_age_bracket_with_value_min_gt_max(
        self,
    ):
        config = {
            "age_col": "age",
            "weight_kg": {
                "age_brackets": [
                    {"min_age": 0, "max_age": 10, "min": 80, "max": 50}  # min > max
                ],
            },
        }  # type: ignore
        with pytest.raises(
            Exception, match="min must be < max"
        ):  # Pydantic ValidationError or ValueError
            RangeDetector(config)  # type: ignore

    def test_tc039_config_validation_raises_for_age_bracket_with_min_age_gt_max_age(
        self,
    ):
        config = {
            "age_col": "age",
            "weight_kg": {
                "age_brackets": [
                    {
                        "min_age": 10,
                        "max_age": 5,
                        "min": 30,
                        "max": 50,
                    }  # min_age > max_age
                ],
            },
        }  # type: ignore
        with pytest.raises(
            Exception, match="min_age must be < max_age"
        ):  # Pydantic ValidationError or ValueError
            RangeDetector(config)  # type: ignore

    def test_tc040_age_dependent_range_with_no_matching_age_brackets(self):
        """Test when no ages fall into any bracket, mask.any() is False."""
        df = pd.DataFrame(
            {
                "age": [50, 60],  # All outside 0-30 brackets
                "weight_kg": [70.0, 80.0],
            }
        )
        config = {
            "age_col": "age",
            "weight_kg": {
                "age_brackets": [
                    {"min_age": 0, "max_age": 10, "min": 30.0, "max": 50.0},
                    {"min_age": 10, "max_age": 20, "min": 50.0, "max": 80.0},
                    {"min_age": 20, "max_age": 30, "min": 60.0, "max": 100.0},
                ]
            },
        }  # type: ignore
        detector = RangeDetector(config)  # type: ignore
        result = detector.detect(df, ["weight_kg"])
        # No matches, so no flagging
        expected = pd.Series([False, False], name="weight_kg")
        pd.testing.assert_series_equal(result["weight_kg"], expected)
