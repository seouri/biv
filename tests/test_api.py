# Tests for API functions

import pytest
import pandas as pd


class TestApi:
    """Tests for API functions"""

    @pytest.fixture
    def sample_df_large(self) -> pd.DataFrame:
        """Large sample df for progress bar tests"""
        return pd.DataFrame(
            {
                "weight_kg": [60.0] * 100,
                "height_cm": [150.0] * 100,
            }
        )

    def test_tc001_unit_detection_warns_for_potential_weight_in_lbs(self):
        pass
        # with pytest.warns(UserWarning, match="Potential unit mismatch: weight values >300 suggest lbs instead of kg"):
        #     # Implement when API is done

    def test_tc002_unit_detection_warns_for_potential_height_in_inches(self):
        pass

    def test_tc003_unit_detection_does_not_warn_for_normal_values(self):
        pass

    def test_tc004_progress_bars_shown_when_progress_bar_true(self):
        pass

    def test_tc005_progress_bars_not_shown_when_progress_bar_false(self):
        pass
