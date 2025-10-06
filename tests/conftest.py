import pytest
import pandas as pd


@pytest.fixture
def sample_data() -> pd.DataFrame:
    """Sample DataFrame with weight and height measurements."""
    return pd.DataFrame(
        {"weight_kg": [50, 60, 70, 80], "height_cm": [150, 160, 170, 180]}
    )
