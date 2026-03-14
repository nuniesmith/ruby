"""
Compatibility shims for the indicators package.
Provides lightweight stand-ins for Component, Registry, and validate_dataframe
that were originally from a different project.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd


class Component:
    """Minimal stand-in for the original core.component.base.Component."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self._config = config or {}


class Registry:
    """Minimal stand-in for the original core.registry.base.Registry."""

    def __init__(self, name: str) -> None:
        self.name = name


def validate_dataframe(
    df: pd.DataFrame,
    required_columns: list[str] | None = None,
) -> None:
    """Raise ValueError if any required column is missing from *df*."""
    if required_columns is None:
        return
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ValueError(f"DataFrame is missing required columns: {missing}")


def process_data(
    df: pd.DataFrame,
    enhanced_detection: bool = False,
    asset_type: str = "btc",
) -> pd.DataFrame:
    """
    Minimal stand-in for core.data.processor.process_data.
    Returns the dataframe unmodified (caller is expected to add logic).
    """
    return df
