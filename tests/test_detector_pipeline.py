# Tests for DetectorPipeline

import pytest
import pandas as pd
from biv.detector_pipeline import DetectorPipeline


class TestDetectorPipeline:
    """Tests for DetectorPipeline class"""

    def test_tc006_detector_pipeline_or_combination(self):
        flags_df = [
            {"col": pd.Series([True, False])},
            {"col": pd.Series([False, True])},
        ]
        pipeline = DetectorPipeline(logic="OR")
        result = pipeline.combine_flags(flags_df)
        expected = {"col": pd.Series([True, True])}
        pd.testing.assert_series_equal(result["col"], expected["col"])

    def test_tc007_detector_pipeline_and_combination(self):
        flags_df = [
            {"col": pd.Series([True, True])},
            {"col": pd.Series([False, True])},
        ]
        pipeline = DetectorPipeline(logic="AND")
        result = pipeline.combine_flags(flags_df)
        expected = {"col": pd.Series([False, True])}
        pd.testing.assert_series_equal(result["col"], expected["col"])

    def test_tc008_detector_pipeline_raises_keyerror_for_invalid_logic(self):
        with pytest.raises(KeyError, match="Unsupported combination logic 'XOR'"):
            DetectorPipeline(logic="XOR")

    def test_tc009_detector_pipeline_handles_empty_flags_list(self):
        pipeline = DetectorPipeline(logic="OR")
        result = pipeline.combine_flags([])
        assert result == {}

    def test_tc010_detector_pipeline_multiple_columns_or_logic(self):
        flags_df = [
            {"col1": pd.Series([True, False]), "col2": pd.Series([False, True])},
            {"col1": pd.Series([False, True]), "col2": pd.Series([True, False])},
        ]
        pipeline = DetectorPipeline(logic="OR")
        result = pipeline.combine_flags(flags_df)
        expected = {
            "col1": pd.Series([True, True]),
            "col2": pd.Series([True, True]),
        }
        pd.testing.assert_series_equal(result["col1"], expected["col1"])
        pd.testing.assert_series_equal(result["col2"], expected["col2"])

    def test_tc011_detector_pipeline_multiple_columns_and_logic(self):
        flags_df = [
            {"col1": pd.Series([True, True]), "col2": pd.Series([False, True])},
            {"col1": pd.Series([True, False]), "col2": pd.Series([False, True])},
        ]
        pipeline = DetectorPipeline(logic="AND")
        result = pipeline.combine_flags(flags_df)
        expected = {
            "col1": pd.Series([True, False]),
            "col2": pd.Series([False, True]),
        }
        pd.testing.assert_series_equal(result["col1"], expected["col1"])
        pd.testing.assert_series_equal(result["col2"], expected["col2"])
