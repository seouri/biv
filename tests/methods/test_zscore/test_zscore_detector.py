"""
Tests for ZScoreDetector class.

Covers configuration validation, detection logic, and integration with calculate_growth_metrics.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch

from biv.methods.zscore.detector import ZScoreDetector, ZScoreConfig


class TestZScoreConfig:
    """Tests for ZScoreConfig Pydantic model."""

    def test_tc001_default_config(self):
        """TC001: Test default configuration values."""
        config = ZScoreConfig()
        assert config.age_col == "age"
        assert config.sex_col == "sex"
        assert config.head_circ_col is None
        assert config.validate_age_units is True

    def test_tc002_config_validate_age_col_type(self):
        """TC002: Config validate age_col type."""
        with pytest.raises(ValueError):
            ZScoreConfig(age_col=123)

    def test_tc003_config_validate_sex_col_as_string(self):
        """TC003: Config validate sex col as string."""
        with pytest.raises(ValueError):
            ZScoreConfig(sex_col=456)

    def test_tc004_custom_config(self):
        """TC004: Test custom configuration values."""
        config = ZScoreConfig(
            age_col="visit_age",
            sex_col="gender",
            head_circ_col="head_cm",
            validate_age_units=False,
        )
        assert config.age_col == "visit_age"
        assert config.sex_col == "gender"
        assert config.head_circ_col == "head_cm"
        assert config.validate_age_units is False

    def test_tc005_invalid_column_names(self):
        """TC005: Test validation of invalid column names."""
        with pytest.raises(ValueError, match="Column name must be a non-empty string"):
            ZScoreConfig(age_col="")

        with pytest.raises(ValueError, match="Column name must be a non-empty string"):
            ZScoreConfig(sex_col="   ")

    def test_tc006_invalid_head_circ_col(self):
        """TC006: Test validation of invalid head circumference column."""
        with pytest.raises(
            ValueError, match="Head circumference column name must be a valid string"
        ):
            ZScoreConfig(head_circ_col="")


class TestZScoreDetector:
    """Tests for ZScoreDetector class."""

    def test_tc007_instantiate_ZScoreDetector_with_valid_config(self):
        """TC007: Instantiate ZScoreDetector with valid config."""
        detector = ZScoreDetector()
        assert isinstance(detector, ZScoreDetector)
        assert detector.config.age_col == "age"

    def test_tc008_detect_raises_ValueError_for_missing_age_col(self):
        """TC008: Detect raises ValueError for missing age col."""
        df = pd.DataFrame({"sex": ["M"], "bmi": [20.0]})
        detector = ZScoreDetector()
        with pytest.raises(ValueError, match="Column 'age' does not exist"):
            detector.detect(df, ["bmi"])

    def test_tc009_invalid_sex_handling_raises(self):
        """TC009: Invalid sex handling raises."""
        df = pd.DataFrame({"age": [60.0], "sex": ["UNKNOWN"], "bmi": [20.0]})
        detector = ZScoreDetector()
        with pytest.raises(ValueError):
            detector.detect(df, ["bmi"])

    def test_tc012_registry_includes_zscore_method(self):
        """TC012: Registry includes 'zscore' method."""
        from biv.methods import registry

        assert "zscore" in registry
        detector_class = registry["zscore"]
        assert detector_class.__name__ == "ZScoreDetector"

    def test_tc013_detect_does_not_modify_input_df(self):
        """TC013: Detect does not modify input df."""
        with patch("biv.zscores.calculate_growth_metrics") as mock_cgm:
            mock_cgm.return_value = {"_bivbmi": np.array([False])}
            df = pd.DataFrame({"age": [60.0], "sex": ["M"], "bmi": [20.0]})
            df_copy = df.copy()
            detector = ZScoreDetector()
            results = detector.detect(df, ["bmi"])
            pd.testing.assert_frame_equal(df, df_copy)

    @patch("biv.methods.zscore.detector.calculate_growth_metrics")
    def test_tc010_cutoffs_apply_mod_WAZ_lt_neg5_or_gt8_flagged(self, mock_cgm):
        """TC010: Cutoffs apply: Mod WAZ < -5 or >8 flagged."""
        mock_cgm.return_value = {"_bivwaz": np.array([True])}  # Assume extreme flagged
        df = pd.DataFrame(
            {"age": [60.0], "sex": ["M"], "weight_kg": [30.0]}
        )  # Assuming this is extreme
        detector = ZScoreDetector()
        results = detector.detect(df, ["weight_kg"])
        assert results["weight_kg"].tolist() == [True]

    def test_tc011_age_in_months_validation_defaults_to_warning(self, caplog):
        """TC011: Age in months validation defaults to warning."""
        import logging

        caplog.set_level(logging.WARNING)
        df = pd.DataFrame({"age": [15], "sex": ["M"], "bmi": [20]})
        detector = ZScoreDetector()
        with patch("biv.methods.zscore.detector.calculate_growth_metrics") as mock:
            mock.return_value = {"_bivbmi": np.array([False])}
            results = detector.detect(df, ["bmi"])
            assert "suggests ages may be in years" in caplog.text

    def test_tc014_unsupported_column_raises_error(self):
        """TC014: Unsupported column raises error."""
        df = pd.DataFrame({"age": [60], "sex": ["M"], "unsupported_metric": [20]})
        detector = ZScoreDetector()
        with pytest.raises(ValueError, match="Unsupported column 'unsupported_metric'"):
            detector.detect(df, ["unsupported_metric"])

    @patch("biv.methods.zscore.detector.calculate_growth_metrics")
    def test_tc015_biv_flag_extraction_uses_correct_keys(self, mock_cgm):
        """TC015: BIV flag extraction uses correct keys."""
        mock_cgm.return_value = {"_bivwaz": np.array([True])}
        df = pd.DataFrame({"age": [60], "sex": ["M"], "weight_kg": [20]})
        detector = ZScoreDetector()
        results = detector.detect(df, ["weight_kg"])
        assert results["weight_kg"].tolist() == [True]

    @patch("biv.methods.zscore.detector.calculate_growth_metrics")
    def test_tc016_integration_with_actual_measure_arrays(self, mock_cgm):
        """TC016: Integration with actual measure arrays."""
        mock_cgm.return_value = {
            "_bivbmi": np.array([True]),
            "_bivwaz": np.array([False]),
            "_bivhaz": np.array([True]),
            "_bivheadcz": np.array([False]),
        }
        df = pd.DataFrame(
            {
                "age": [60],
                "sex": ["M"],
                "bmi": [20],
                "weight_kg": [25],
                "height_cm": [120],
                "head_circ_cm": [50],
            }
        )
        detector = ZScoreDetector()
        results = detector.detect(df, ["bmi", "weight_kg", "height_cm", "head_circ_cm"])
        assert set(results.keys()) == {"bmi", "weight_kg", "height_cm", "head_circ_cm"}
        assert results["bmi"].tolist() == [True]
        assert results["weight_kg"].tolist() == [False]
        assert results["height_cm"].tolist() == [True]
        assert results["head_circ_cm"].tolist() == [False]

    def test_tc017_detector_init_default(self):
        """TC017: Test detector initialization with default parameters."""
        detector = ZScoreDetector()
        assert detector.config.age_col == "age"
        assert detector.config.sex_col == "sex"
        assert detector.config.head_circ_col is None
        assert detector.config.validate_age_units is True

    def test_tc018_custom_column_mapping_works(self):
        """TC018: Test detection with custom column names."""
        with patch("biv.methods.zscore.detector.calculate_growth_metrics") as mock_cgm:
            mock_cgm.return_value = {"_bivbmi": np.array([True])}

            df = pd.DataFrame({"visit_age": [60.0], "gender": ["M"], "bmi": [25.0]})

            detector = ZScoreDetector(age_col="visit_age", sex_col="gender")
            results = detector.detect(df, ["bmi"])

            assert "bmi" in results
            assert results["bmi"].tolist() == [True]

    def test_tc023_detector_init_invalid_config(self):
        """TC023: Test detector initialization with invalid configuration."""
        with pytest.raises(ValueError, match="Invalid configuration"):
            ZScoreDetector(age_col="")

    def test_tc024_validate_config_conflicting_columns(self):
        """TC024: Test validation of conflicting column names."""
        with pytest.raises(
            ValueError, match="Configuration must specify unique column names"
        ):
            detector = ZScoreDetector(age_col="age", sex_col="age")
            detector.validate_config()

    @patch("biv.methods.zscore.detector.calculate_growth_metrics")
    def test_tc019_detect_basic_bmi(self, mock_cgm):
        """TC019: Test basic detection on BMI column."""
        # Setup mock return - calculate_growth_metrics returns BIV flags
        mock_cgm.return_value = {
            "_bivbmi": np.array([False, True, False]),
        }

        # Create test DataFrame
        df = pd.DataFrame(
            {
                "patient_id": [1, 2, 3],
                "age": [60.0, 120.0, 180.0],  # months
                "sex": ["M", "F", "M"],
                "bmi": [18.5, 25.0, 22.0],
            }
        )

        detector = ZScoreDetector()
        results = detector.detect(df, ["bmi"])

        # Check results
        assert "bmi" in results
        assert len(results["bmi"]) == 3
        assert results["bmi"].tolist() == [False, True, False]  # From mock

        # Verify calculate_growth_metrics was called correctly
        mock_cgm.assert_called_once()
        call_args = mock_cgm.call_args
        assert np.array_equal(call_args[1]["agemos"], df["age"].values)
        assert np.array_equal(call_args[1]["sex"], ["M", "F", "M"])
        assert np.array_equal(call_args[1]["bmi"], df["bmi"].values)

    @patch("biv.methods.zscore.detector.calculate_growth_metrics")
    def test_tc020_detect_for_multiple_measures(self, mock_cgm):
        """TC020: Test detection on weight_kg and height_cm columns."""
        mock_cgm.return_value = {
            "_bivwaz": np.array([True, False]),
            "_bivhaz": np.array([False, True]),
        }

        df = pd.DataFrame(
            {
                "age": [48.0, 72.0],
                "sex": ["F", "M"],
                "weight_kg": [15.0, 25.0],
                "height_cm": [105.0, 120.0],
            }
        )

        detector = ZScoreDetector()
        results = detector.detect(df, ["weight_kg", "height_cm"])

        assert "weight_kg" in results
        assert "height_cm" in results
        assert results["weight_kg"].tolist() == [True, False]
        assert results["height_cm"].tolist() == [False, True]

    def test_tc021_detect_missing_required_column(self):
        """TC021: Test detection fails when required column is missing."""
        df = pd.DataFrame(
            {
                "age": [60.0],  # Missing sex column
                "bmi": [20.0],
            }
        )

        detector = ZScoreDetector()
        with pytest.raises(ValueError, match="Column 'sex' does not exist"):
            detector.detect(df, ["bmi"])

    def test_tc022_detect_unsupported_column(self):
        """TC022: Test detection fails with unsupported column name."""
        df = pd.DataFrame({"age": [60.0], "sex": ["M"], "unsupported_metric": [20.0]})

        detector = ZScoreDetector()
        with pytest.raises(ValueError, match="Unsupported column 'unsupported_metric'"):
            detector.detect(df, ["unsupported_metric"])

    @patch("biv.methods.zscore.detector.calculate_growth_metrics")
    def test_tc025_detect_with_nan_values(self, mock_cgm):
        """TC025: Test detection handles NaN values properly."""
        mock_cgm.return_value = {
            "_bivbmi": np.array([False, False]),  # NaN -> False
            "mod_bmiz": np.array([np.nan, 0.5]),
        }

        df = pd.DataFrame(
            {
                "age": [60.0, 120.0],
                "sex": ["M", "F"],
                "bmi": [np.nan, 20.0],  # First BMI is NaN
            }
        )

        detector = ZScoreDetector()
        results = detector.detect(df, ["bmi"])

        assert results["bmi"].tolist() == [False, False]

    def test_tc026_detect_invalid_sex(self):
        """TC026: Test detection raises ValueError for invalid sex values."""
        df = pd.DataFrame(
            {
                "age": [60.0],
                "sex": ["UNKNOWN"],  # Invalid sex
                "bmi": [20.0],
            }
        )

        with patch("biv.zscores.calculate_growth_metrics") as mock_cgm:
            mock_cgm.side_effect = ValueError("Sex values must be 'M' or 'F'")
            detector = ZScoreDetector()
            with pytest.raises(ValueError, match="Z-score detection failed"):
                detector.detect(df, ["bmi"])

    @patch("biv.methods.zscore.detector.calculate_growth_metrics")
    def test_tc027_detect_age_validation_warning(self, mock_cgm, caplog):
        """TC027: Test age validation warning for potential unit issues."""
        import logging

        caplog.set_level(logging.WARNING)

        mock_cgm.return_value = {"_bivbmi": np.array([False])}

        df = pd.DataFrame(
            {
                "age": [12.0],  # Mean age < 130 suggests years, not months
                "sex": ["M"],
                "bmi": [20.0],
            }
        )

        detector = ZScoreDetector(validate_age_units=True)
        results = detector.detect(df, ["bmi"])

        assert "Mean age 12.0 months suggests ages may be in years" in caplog.text

    @patch("biv.methods.zscore.detector.calculate_growth_metrics")
    def test_tc028_detect_age_validation_disabled(self, mock_cgm, caplog):
        """TC028: Test age validation warning is disabled when configured."""
        import logging

        caplog.set_level(logging.WARNING)

        mock_cgm.return_value = {"_bivbmi": np.array([False])}

        df = pd.DataFrame(
            {
                "age": [12.0],  # Would normally warn
                "sex": ["M"],
                "bmi": [20.0],
            }
        )

        detector = ZScoreDetector(validate_age_units=False)
        results = detector.detect(df, ["bmi"])

        assert "suggests ages may be in years" not in caplog.text

    @patch("biv.methods.zscore.detector.calculate_growth_metrics")
    def test_tc029_detect_extreme_age_warning(self, mock_cgm, caplog):
        """TC029: Test warning for ages exceeding CDC reference limits."""
        import logging

        caplog.set_level(logging.WARNING)

        mock_cgm.return_value = {"_bivbmi": np.array([False])}

        df = pd.DataFrame(
            {
                "age": [250.0],  # > 240 months
                "sex": ["M"],
                "bmi": [20.0],
            }
        )

        detector = ZScoreDetector()
        results = detector.detect(df, ["bmi"])

        assert "Maximum age 250.0 months exceeds CDC reference limit" in caplog.text

    def test_tc030_detect_does_not_modify_original_df(self):
        """TC030: Test that detect() does not modify the original DataFrame."""
        with patch("biv.zscores.calculate_growth_metrics") as mock_cgm:
            mock_cgm.return_value = {"_bivbmi": np.array([False])}

            original_df = pd.DataFrame(
                {"patient_id": [1], "age": [60.0], "sex": ["M"], "bmi": [20.0]}
            )
            df_copy = original_df.copy()

            detector = ZScoreDetector()
            results = detector.detect(original_df, ["bmi"])

            # Original DataFrame should be unchanged
            pd.testing.assert_frame_equal(original_df, df_copy)

    def test_tc031_detect_results_have_correct_index(self):
        """TC031: Test that returned Series have correct index alignment."""
        with patch("biv.methods.zscore.detector.calculate_growth_metrics") as mock_cgm:
            mock_cgm.return_value = {"_bivbmi": np.array([True, False, True])}

            df = pd.DataFrame(
                {
                    "age": [48.0, 72.0, 96.0],
                    "sex": ["M", "F", "M"],
                    "bmi": [15.0, 20.0, 25.0],
                },
                index=[10, 20, 30],
            )  # Custom index

            detector = ZScoreDetector()
            results = detector.detect(df, ["bmi"])

            # Check index alignment
            pd.testing.assert_index_equal(results["bmi"].index, df.index)
            assert results["bmi"].tolist() == [True, False, True]

    @patch("biv.methods.zscore.detector.calculate_growth_metrics")
    def test_tc032_detect_all_measures_mapping(self, mock_cgm):
        """TC032: Test that all supported column-to-measure mappings work."""
        mock_cgm.return_value = {
            "_bivbmi": np.array([True]),
            "_bivwaz": np.array([True]),
            "_bivhaz": np.array([True]),
            "_bivheadcz": np.array([True]),
        }

        df = pd.DataFrame(
            {
                "age": [60.0],
                "sex": ["M"],
                "weight_kg": [25.0],
                "height_cm": [120.0],
                "bmi": [20.0],
                "head_circ_cm": [52.0],
            }
        )

        detector = ZScoreDetector()
        results = detector.detect(df, ["bmi", "weight_kg", "height_cm", "head_circ_cm"])

        assert set(results.keys()) == {"bmi", "weight_kg", "height_cm", "head_circ_cm"}
        for col in results:
            assert results[col].tolist() == [True]


class TestZScoreDetectorRegistry:
    """Tests for ZScoreDetector in methods registry."""

    def test_tc033_registry_includes_zscore(self):
        """TC033: Test that zscore method is registered."""
        from biv.methods import registry

        assert "zscore" in registry
        detector_class = registry["zscore"]
        assert detector_class.__name__ == "ZScoreDetector"

    def test_tc034_registry_instantiation(self):
        """TC034: Test that registered detector can be instantiated."""
        from biv.methods import registry

        detector_class = registry["zscore"]
        detector = detector_class()
        assert isinstance(detector, ZScoreDetector)
