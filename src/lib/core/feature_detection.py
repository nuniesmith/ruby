"""
Feature Detection Module

This module provides utilities to detect available features and optional
dependencies within the Ruby Futures framework. It helps with graceful
degradation when optional dependencies are not installed.
"""

import importlib
from collections.abc import Callable
from functools import lru_cache

from lib.core.logging_config import get_logger

logger = get_logger(__name__)

# Registry of features with detection functions
_FEATURE_REGISTRY: dict[str, Callable[[], bool]] = {}

# Cache of detected features
_DETECTED_FEATURES: set[str] = set()
_FEATURES_SCANNED = False


def register_feature(name: str, detection_func: Callable[[], bool]) -> None:
    """
    Register a feature detection function.

    Args:
        name: Name of the feature
        detection_func: Function that returns True if feature is available
    """
    _FEATURE_REGISTRY[name] = detection_func


@lru_cache(maxsize=128)
def has_feature(feature_name: str) -> bool:
    """
    Check if a specific feature is available.

    Args:
        feature_name: Name of the feature to check

    Returns:
        bool: True if the feature is available, False otherwise
    """
    global _FEATURES_SCANNED

    # Scan all features on first use if not done already
    if not _FEATURES_SCANNED:
        _scan_all_features()

    return feature_name in _DETECTED_FEATURES


def _scan_all_features() -> None:
    """Scan all registered features and cache the results."""
    global _FEATURES_SCANNED

    for name, detection_func in _FEATURE_REGISTRY.items():
        try:
            if detection_func():
                _DETECTED_FEATURES.add(name)
                logger.debug("Feature is available", feature=name)
            else:
                logger.debug("Feature is not available", feature=name)
        except Exception as e:
            logger.debug("Error detecting feature", feature=name, error=str(e))

    _FEATURES_SCANNED = True
    logger.debug(
        "Feature detection complete",
        available=len(_DETECTED_FEATURES),
        total=len(_FEATURE_REGISTRY),
    )


def get_available_features() -> set[str]:
    """
    Get all available features.

    Returns:
        Set[str]: Set of available feature names
    """
    global _FEATURES_SCANNED
    if not _FEATURES_SCANNED:
        _scan_all_features()

    return _DETECTED_FEATURES.copy()


# Module detection helpers
def _module_exists(module_name: str) -> bool:
    """Check if a module exists and can be imported."""
    try:
        importlib.import_module(module_name)
        return True
    except ImportError:
        return False


# Feature detection functions
def _has_numpy() -> bool:
    return _module_exists("numpy")


def _has_pandas() -> bool:
    return _module_exists("pandas")


def _has_pydantic() -> bool:
    return _module_exists("pydantic")


def _has_trading_api() -> bool:
    return _module_exists("core.trading.api")


def _has_backtesting() -> bool:
    return _module_exists("core.backtesting")


def _has_ml_features() -> bool:
    return all([_module_exists(m) for m in ["numpy", "pandas", "sklearn"]])


def _has_async_support() -> bool:
    try:
        import asyncio  # noqa: F401

        return True
    except ImportError:
        return False


def _has_database_support() -> bool:
    return _module_exists("sqlalchemy")


def _has_ui() -> bool:
    return _module_exists("core.ui")


def _has_plotting() -> bool:
    return _module_exists("matplotlib")


def _has_web_api() -> bool:
    return _module_exists("fastapi")


def _has_strategy_optimizer() -> bool:
    return _module_exists("core.strategy.optimizer")


def _has_gpu_support() -> bool:
    """Check for GPU support via tensorflow or torch."""
    # First try tensorflow
    if _module_exists("tensorflow"):
        try:
            import tensorflow as tf

            return len(tf.config.list_physical_devices("GPU")) > 0
        except Exception:
            pass

    # Then try torch
    if _module_exists("torch"):
        try:
            import torch

            return torch.cuda.is_available()
        except Exception:
            pass

    return False


# Register all feature detection functions
register_feature("numpy", _has_numpy)
register_feature("pandas", _has_pandas)
register_feature("pydantic", _has_pydantic)
register_feature("trading_api", _has_trading_api)
register_feature("backtesting", _has_backtesting)
register_feature("ml", _has_ml_features)
register_feature("async", _has_async_support)
register_feature("database", _has_database_support)
register_feature("ui", _has_ui)
register_feature("plotting", _has_plotting)
register_feature("web_api", _has_web_api)
register_feature("strategy_optimizer", _has_strategy_optimizer)
register_feature("gpu", _has_gpu_support)

# Add shorthand groups
register_feature("core", lambda: all(has_feature(f) for f in ["numpy", "pandas"]))
register_feature("advanced", lambda: has_feature("ml") and has_feature("strategy_optimizer"))
register_feature("full", lambda: has_feature("core") and has_feature("web_api") and has_feature("ui"))

# Provide feature detection information
__all__ = ["has_feature", "get_available_features", "register_feature"]
