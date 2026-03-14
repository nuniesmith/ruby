"""
RB Detector — Unified Range Breakout Detection Facade
======================================================
Provides a clean public API that delegates to the individual range builders
in ``range_builders.py`` and the existing ``detect_range_breakout()`` in
``lib.services.engine.breakout``.

This module is the **canonical entry point** for breakout detection in the
``strategies.rb`` package.  It re-exports the engine's heavy detection
function while adding convenience wrappers that simplify the most common
call patterns.

Phase 1G of the RB System Refactor — incremental migration.  The actual
detection logic still lives in ``lib.services.engine.breakout`` for now;
this facade ensures callers can import from the strategies package and will
automatically benefit when logic migrates here.

Public API::

    from lib.trading.strategies.rb.detector import (
        detect_range_breakout,
        detect_breakout_for_type,
        detect_all_breakout_types,
        compute_atr,
        BreakoutResult,
    )

    # Full detection with engine RangeConfig
    result = detect_range_breakout(bars_1m, symbol="MGC", config=config)

    # Simplified: just pass breakout type, uses default config
    result = detect_breakout_for_type(bars_1m, BreakoutType.PrevDay, symbol="MGC")

    # Scan all 13 types at once
    results = detect_all_breakout_types(bars_1m, symbol="MGC")

Design:
  - Pure detection — no Redis, no publishing, no side-effects.
  - Thread-safe: no shared mutable state.
  - Delegates to ``engine.breakout`` for the heavy lifting (for now).
  - Exposes ``compute_atr`` from ``range_builders`` as the single canonical
    ATR implementation.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd

from lib.core.breakout_types import BreakoutType

from .range_builders import compute_atr

logger = logging.getLogger("strategies.rb.detector")


# ---------------------------------------------------------------------------
# Re-export BreakoutResult from engine (canonical result dataclass)
# ---------------------------------------------------------------------------

from lib.trading.strategies.rb.breakout import BreakoutResult  # noqa: E402

# ---------------------------------------------------------------------------
# Primary detection API
# ---------------------------------------------------------------------------


def detect_range_breakout(
    bars: pd.DataFrame,
    symbol: str = "",
    config: Any = None,
    *,
    prev_day_high: float | None = None,
    prev_day_low: float | None = None,
    ib_high: float | None = None,
    ib_low: float | None = None,
    orb_session_start: Any | None = None,
    orb_session_end: Any | None = None,
    orb_scan_start: Any | None = None,
) -> BreakoutResult:
    """Unified breakout detection for any of the 13 range breakout types.

    This is a thin facade over ``lib.services.engine.breakout.detect_range_breakout()``.
    It exists so callers can import from the ``strategies.rb`` package
    consistently.

    Args:
        bars:   1-minute OHLCV DataFrame (DatetimeIndex, columns: Open, High,
                Low, Close, Volume).
        symbol: Ticker / symbol string (e.g. ``"MGC"``, ``"MES=F"``).
        config: An engine ``RangeConfig`` instance from
                ``lib.services.engine.breakout.DEFAULT_CONFIGS``.
                If ``None``, defaults to the ORB config.
        prev_day_high: Optional override for PDR high level.
        prev_day_low:  Optional override for PDR low level.
        ib_high:       Optional override for IB high level.
        ib_low:        Optional override for IB low level.
        orb_session_start: ORB session start time override.
        orb_session_end:   ORB session end time override.
        orb_scan_start:    ORB scan start time override.

    Returns:
        ``BreakoutResult`` with detection outcome, range levels, direction,
        trigger price, and quality-gate metadata.
    """
    from lib.trading.strategies.rb.breakout import (
        DEFAULT_CONFIGS,
    )
    from lib.trading.strategies.rb.breakout import (
        detect_range_breakout as _engine_detect,
    )

    if config is None:
        config = DEFAULT_CONFIGS[BreakoutType.ORB]

    return _engine_detect(
        bars,
        symbol=symbol,
        config=config,
        prev_day_high=prev_day_high,
        prev_day_low=prev_day_low,
        ib_high=ib_high,
        ib_low=ib_low,
        orb_session_start=orb_session_start,
        orb_session_end=orb_session_end,
        orb_scan_start=orb_scan_start,
    )


def detect_breakout_for_type(
    bars: pd.DataFrame,
    breakout_type: BreakoutType,
    symbol: str = "",
    *,
    prev_day_high: float | None = None,
    prev_day_low: float | None = None,
    ib_high: float | None = None,
    ib_low: float | None = None,
    orb_session_start: Any | None = None,
    orb_session_end: Any | None = None,
    orb_scan_start: Any | None = None,
) -> BreakoutResult:
    """Convenience wrapper: detect a breakout using the default config for a type.

    Equivalent to::

        from lib.trading.strategies.rb.breakout import DEFAULT_CONFIGS
        config = DEFAULT_CONFIGS[breakout_type]
        result = detect_range_breakout(bars, symbol=symbol, config=config)

    but with a cleaner call signature for the common case.

    Args:
        bars:           1-minute OHLCV DataFrame.
        breakout_type:  Which ``BreakoutType`` to detect.
        symbol:         Ticker / symbol string.
        prev_day_high:  Optional PDR high override.
        prev_day_low:   Optional PDR low override.
        ib_high:        Optional IB high override.
        ib_low:         Optional IB low override.
        orb_session_start: ORB session start time override.
        orb_session_end:   ORB session end time override.
        orb_scan_start:    ORB scan start time override.

    Returns:
        ``BreakoutResult`` for the specified breakout type.
    """
    from lib.trading.strategies.rb.breakout import DEFAULT_CONFIGS

    config = DEFAULT_CONFIGS.get(breakout_type)
    if config is None:
        logger.warning(
            "No default config for breakout type %s — returning empty result",
            breakout_type.name,
        )
        return BreakoutResult(symbol=symbol, breakout_type=breakout_type)

    return detect_range_breakout(
        bars,
        symbol=symbol,
        config=config,
        prev_day_high=prev_day_high,
        prev_day_low=prev_day_low,
        ib_high=ib_high,
        ib_low=ib_low,
        orb_session_start=orb_session_start,
        orb_session_end=orb_session_end,
        orb_scan_start=orb_scan_start,
    )


def detect_all_breakout_types(
    bars: pd.DataFrame,
    symbol: str = "",
    *,
    types: list[BreakoutType] | None = None,
    prev_day_high: float | None = None,
    prev_day_low: float | None = None,
    ib_high: float | None = None,
    ib_low: float | None = None,
) -> dict[BreakoutType, BreakoutResult]:
    """Run detection for multiple breakout types and return all results.

    This is a convenience wrapper that calls ``detect_breakout_for_type()``
    for each type in ``types`` and collects the results.

    Args:
        bars:           1-minute OHLCV DataFrame.
        symbol:         Ticker / symbol string.
        types:          List of ``BreakoutType`` to scan.  Defaults to all 13.
        prev_day_high:  Optional PDR high override.
        prev_day_low:   Optional PDR low override.
        ib_high:        Optional IB high override.
        ib_low:         Optional IB low override.

    Returns:
        Dict mapping ``BreakoutType`` → ``BreakoutResult`` for each type
        that was scanned (includes non-detections so callers can inspect
        range levels even when no breakout fired).
    """
    if types is None:
        types = list(BreakoutType)

    results: dict[BreakoutType, BreakoutResult] = {}
    for bt in types:
        try:
            results[bt] = detect_breakout_for_type(
                bars,
                bt,
                symbol=symbol,
                prev_day_high=prev_day_high,
                prev_day_low=prev_day_low,
                ib_high=ib_high,
                ib_low=ib_low,
            )
        except Exception as exc:
            logger.debug("detect_all_breakout_types: %s failed for %s: %s", bt.name, symbol, exc)
            results[bt] = BreakoutResult(
                symbol=symbol,
                breakout_type=bt,
                error=str(exc),
            )

    return results


def detect_breakouts_filtered(
    bars: pd.DataFrame,
    symbol: str = "",
    *,
    types: list[BreakoutType] | None = None,
    prev_day_high: float | None = None,
    prev_day_low: float | None = None,
    ib_high: float | None = None,
    ib_low: float | None = None,
) -> list[BreakoutResult]:
    """Run detection for multiple types and return only confirmed breakouts.

    Like ``detect_all_breakout_types()`` but filters to only results where
    ``breakout_detected is True``, sorted by trigger price quality.

    Args:
        bars:           1-minute OHLCV DataFrame.
        symbol:         Ticker / symbol string.
        types:          List of ``BreakoutType`` to scan.  Defaults to all 13.
        prev_day_high:  Optional PDR high override.
        prev_day_low:   Optional PDR low override.
        ib_high:        Optional IB high override.
        ib_low:         Optional IB low override.

    Returns:
        List of ``BreakoutResult`` where ``breakout_detected is True``,
        ordered by breakout type ordinal.
    """
    all_results = detect_all_breakout_types(
        bars,
        symbol=symbol,
        types=types,
        prev_day_high=prev_day_high,
        prev_day_low=prev_day_low,
        ib_high=ib_high,
        ib_low=ib_low,
    )

    detected = [r for r in all_results.values() if r.breakout_detected]
    detected.sort(key=lambda r: r.breakout_type.value)
    return detected


# ---------------------------------------------------------------------------
# Type-specific convenience functions (backward compat)
# ---------------------------------------------------------------------------


def detect_pdr_breakout(
    bars: pd.DataFrame,
    symbol: str = "",
    *,
    prev_day_high: float | None = None,
    prev_day_low: float | None = None,
) -> BreakoutResult:
    """Convenience: detect a Previous Day Range breakout."""
    return detect_breakout_for_type(
        bars,
        BreakoutType.PrevDay,
        symbol=symbol,
        prev_day_high=prev_day_high,
        prev_day_low=prev_day_low,
    )


def detect_ib_breakout(
    bars: pd.DataFrame,
    symbol: str = "",
    *,
    ib_high: float | None = None,
    ib_low: float | None = None,
) -> BreakoutResult:
    """Convenience: detect an Initial Balance breakout."""
    return detect_breakout_for_type(
        bars,
        BreakoutType.InitialBalance,
        symbol=symbol,
        ib_high=ib_high,
        ib_low=ib_low,
    )


def detect_consolidation_breakout(
    bars: pd.DataFrame,
    symbol: str = "",
) -> BreakoutResult:
    """Convenience: detect a Consolidation/Squeeze breakout."""
    return detect_breakout_for_type(
        bars,
        BreakoutType.Consolidation,
        symbol=symbol,
    )


__all__ = [
    # Primary API
    "detect_range_breakout",
    "detect_breakout_for_type",
    "detect_all_breakout_types",
    "detect_breakouts_filtered",
    # Type-specific convenience
    "detect_pdr_breakout",
    "detect_ib_breakout",
    "detect_consolidation_breakout",
    # Result dataclass
    "BreakoutResult",
    # Shared helper
    "compute_atr",
]
