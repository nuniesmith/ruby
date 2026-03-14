"""
Pre-configured indicator groups for common trading use cases.

Usage:
    from lib.indicators.presets import SCALP_PRESET, SWING_PRESET, REGIME_PRESET
    from lib.indicators import IndicatorManager

    mgr = IndicatorManager()
    for ind in SCALP_PRESET:
        mgr.add_indicator(ind)
    results = mgr.calculate_all(df)

Each preset is a list of (indicator_class, params_dict) tuples so you can
construct them without needing to call the factory.

NOTE on ChoppinessIndex compatibility
--------------------------------------
``ChoppinessIndex`` (lib.indicators.other.choppiness_index) is a **standalone**
class — its constructor signature is ``__init__(self, period=14)`` and it reads
capitalised columns (``High``, ``Low``) rather than the lowercase ``high``/``low``
used by the registry-based ``Indicator`` subclasses.

Because of this, ``ChoppinessIndex`` cannot be instantiated via the
``ind_cls(name=..., params=...)`` pattern that ``build_manager()`` uses.
``REGIME_PRESET`` therefore includes a thin ``_ChoppinessIndexAdapter`` wrapper
that bridges the two APIs so that ``build_manager(REGIME_PRESET)`` works
transparently.  If you ever replace ``ChoppinessIndex`` with a proper
``Indicator`` subclass, simply remove the adapter and update the preset entry.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from lib.indicators.base import Indicator
from lib.indicators.momentum.rsi import RSI
from lib.indicators.other.choppiness_index import ChoppinessIndex
from lib.indicators.trend.macd import MACD
from lib.indicators.trend.moving_average import EMA, VWAP
from lib.indicators.trend.volatility.atr import ATR
from lib.indicators.trend.volatility.bollinger import BollingerBands

# ---------------------------------------------------------------------------
# ChoppinessIndex compatibility adapter
# ---------------------------------------------------------------------------


class _ChoppinessIndexAdapter(Indicator):
    """Wraps the standalone ``ChoppinessIndex`` so it can be used anywhere
    a registry-based ``Indicator`` is expected (e.g. inside IndicatorManager).

    The adapter:
    * Accepts the standard ``(name, params)`` constructor signature.
    * Extracts ``period`` from *params* (default 14).
    * Normalises column names: if the DataFrame has lowercase ``high``/``low``
      columns it renames them temporarily so the inner ``ChoppinessIndex``
      can find its expected ``High``/``Low`` columns.
    * Returns a single-column ``pd.DataFrame`` whose column is ``CHOP_<period>``.
    """

    def __init__(self, name: str = "ChoppinessIndex", params: dict[str, Any] | None = None):
        super().__init__(name, params)
        self.period: int = self.params.get("period", 14)
        self._inner = ChoppinessIndex(period=self.period)

    def calculate(self, data: pd.DataFrame, price_column: str = "close") -> pd.DataFrame:
        col_map: dict[str, str] = {}
        if "high" in data.columns and "High" not in data.columns:
            col_map["high"] = "High"
        if "low" in data.columns and "Low" not in data.columns:
            col_map["low"] = "Low"

        working = data.rename(columns=col_map) if col_map else data
        series = self._inner.calculate(working)
        col_name = f"CHOP_{self.period}"
        return pd.DataFrame({col_name: series}, index=data.index)

    @classmethod
    def required_columns(cls) -> list[str]:
        return ["high", "low"]


# ---------------------------------------------------------------------------
# SCALP_PRESET
# Fast intraday indicators for scalping strategies (1m / 5m bars)
# ---------------------------------------------------------------------------

SCALP_PRESET: list[tuple[type[Indicator], dict[str, Any]]] = [
    (EMA, {"period": 9, "column": "close"}),
    (EMA, {"period": 21, "column": "close"}),
    (RSI, {"period": 14}),
    (ATR, {"period": 14}),
    (VWAP, {}),
]


# ---------------------------------------------------------------------------
# SWING_PRESET
# Swing-trading indicators (15m / 1H bars)
# ---------------------------------------------------------------------------

SWING_PRESET: list[tuple[type[Indicator], dict[str, Any]]] = [
    (EMA, {"period": 21, "column": "close"}),
    (EMA, {"period": 50, "column": "close"}),
    (EMA, {"period": 200, "column": "close"}),
    (MACD, {"fast_period": 12, "slow_period": 26, "signal_period": 9}),
    (RSI, {"period": 14}),
    (ATR, {"period": 14}),
    (BollingerBands, {"period": 20, "std_dev": 2.0}),
]


# ---------------------------------------------------------------------------
# REGIME_PRESET
# Market regime / volatility characterisation indicators
#
# NOTE: ChoppinessIndex is included via ``_ChoppinessIndexAdapter`` because the
# raw ``ChoppinessIndex`` class is standalone and incompatible with the
# ``build_manager`` instantiation pattern.  See module docstring for details.
# ---------------------------------------------------------------------------

REGIME_PRESET: list[tuple[type[Indicator], dict[str, Any]]] = [
    (ATR, {"period": 14}),
    (BollingerBands, {"period": 20, "std_dev": 2.0}),
    (_ChoppinessIndexAdapter, {}),
]


# ---------------------------------------------------------------------------
# Helper: build an IndicatorManager from a preset
# ---------------------------------------------------------------------------


def _make_indicator_name(ind_cls: type[Indicator], params: dict[str, Any]) -> str:
    """Derive a unique indicator name from its class and params.

    Uses ``period`` when present (the most common differentiator), then falls
    back to a sorted list of all param values so that two instances of the same
    class with different params always get distinct names.

    Examples::

        _make_indicator_name(EMA, {"period": 9,  "column": "close"}) -> "EMA_9"
        _make_indicator_name(EMA, {"period": 21, "column": "close"}) -> "EMA_21"
        _make_indicator_name(VWAP, {})                               -> "VWAP"
    """
    base = ind_cls.__name__
    if not params:
        return base
    if "period" in params:
        return f"{base}_{params['period']}"
    # Fallback: append sorted param values joined by underscores
    suffix = "_".join(str(v) for _, v in sorted(params.items()))
    return f"{base}_{suffix}"


def build_manager(preset: list[tuple[type[Indicator], dict[str, Any]]]):
    """Instantiate an IndicatorManager pre-loaded with the given preset.

    Args:
        preset: A preset list of ``(indicator_class, params_dict)`` tuples.
                All indicator classes must follow the standard
                ``Indicator.__init__(name, params)`` signature.  Standalone
                classes (e.g. the raw ``ChoppinessIndex``) must be wrapped in
                an adapter first — see ``_ChoppinessIndexAdapter`` above.

    Returns:
        Configured ``IndicatorManager`` instance ready for ``calculate_all(df)``.

    Example::

        mgr = build_manager(SCALP_PRESET)
        results = mgr.calculate_all(df)
    """
    from lib.indicators.manager import IndicatorManager

    mgr = IndicatorManager()
    for ind_cls, params in preset:
        name = _make_indicator_name(ind_cls, params)
        instance = ind_cls(name=name, params=params)
        mgr.add_indicator(instance)
    return mgr
