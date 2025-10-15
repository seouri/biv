"""
Methods registry for automatic detector discovery.

This module provides automatic registration of detection methods by introspecting
BaseDetector subclasses in the biv.methods submodules.
"""

from typing import Dict, Type
from .base import BaseDetector


# Import detector modules to register subclasses
from . import range
from .zscore import detector as zscore_detector


def _build_registry() -> Dict[str, Type[BaseDetector]]:
    """Build the registry by discovering BaseDetector subclasses."""
    registry = {}
    for cls in BaseDetector.__subclasses__():
        # Derive method name from class name: RangeDetector -> 'range'
        method_name = cls.__name__.replace("Detector", "").lower()
        registry[method_name] = cls
    return registry


# Global registry instance
registry = _build_registry()

__all__ = ["registry", "range", "zscore"]
