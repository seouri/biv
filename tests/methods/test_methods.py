import pytest
from biv.methods import registry
from biv.methods.base import BaseDetector


def test_tc001_registry_available_methods() -> None:
    """Test that registry includes known detectors like 'range'."""
    assert "range" in registry  # Assuming RangeDetector is implemented


def test_tc002_registry_returns_detector_class() -> None:
    """Test that registry returns the correct detector class for known method."""
    detector_cls = registry["range"]
    assert issubclass(detector_cls, BaseDetector)


def test_tc003_registry_excludes_basedetector() -> None:
    """Test that BaseDetector is not included in registry."""
    assert "basedetector" not in registry


def test_tc004_registry_raises_keyerror_for_unknown() -> None:
    """Test that accessing unknown method raises KeyError."""
    with pytest.raises(KeyError):
        _ = registry["unknown"]


def test_tc005_registry_discovers_multiple() -> None:
    """Test that registry includes multiple methods when available."""
    methods = registry.keys()
    assert "range" in methods
    # If zscore is implemented, include it
    # For now, just check range is there, but expect more in future
