"""
Generalised Breakout Detection Engine
======================================
Replaces all ORB-specific breakout handlers with a single, type-aware
``detect_range_breakout()`` function that works for every range type:

  - **ORB**  — Opening Range Breakout (existing, session-parameterised)
  - **PDR**  — Previous Day Range breakout (daily high/low anchor)
  - **IB**   — Initial Balance breakout (first 60 min of RTH, 09:30–10:30 ET)
  - **CONS** — Consolidation / Squeeze breakout (ATR/Bollinger contraction)

Design principles
-----------------
1. **Single entry-point**: ``detect_range_breakout(bars, symbol, config)``
   returns a ``BreakoutResult`` regardless of type.  Callers don't need to
   know how each range is built — they set ``BreakoutType`` in the config.

2. **Composable config**: ``RangeConfig`` is a frozen dataclass that holds
   all thresholds (depth, body, OR-size floor/cap, ATR period, etc.) plus
   the ``BreakoutType``.  Callers can start from pre-built ``DEFAULT_CONFIGS``
   and override fields with ``dataclasses.replace()``.

3. **No side-effects**: detection is pure — no Redis, no DB writes.  The
   engine main / handler layer is responsible for publishing, persisting, and
   filtering.

4. **Backward-compatible**: the existing ``ORBResult`` / ``detect_opening_range_breakout``
   pipeline is untouched.  This module adds new types alongside the ORB
   infrastructure rather than replacing it until a later migration step.

Breakout types
--------------
ORB   Opening Range — already handled by orb.py; mirrored here so callers
      can use a single dispatch path via ``detect_range_breakout``.

PDR   Previous Day Range — the prior trading session's high and low define
      the range.  A breakout occurs when price closes beyond either level
      by at least ``min_depth_atr_pct × ATR``.  Strongest at London open
      and US open when yesterday's levels act as magnets / targets.

IB    Initial Balance — the high/low of the first 60 minutes of the RTH
      session (09:30–10:30 ET) defines the range.  Breakouts after 10:30
      are the "B-type" IB breakout from the Dalton/Steidlmayer market-profile
      playbook.  Very high win-rate on trend days when IB is narrow.

CONS  Consolidation / Squeeze — uses a Bollinger Band / ATR contraction
      detector to find periods of range compression, then detects the
      expansion bar.  Parametrised by ``squeeze_atr_mult`` (BB width as a
      fraction of ATR) and ``squeeze_lookback`` (bars to confirm compression).

Public API
----------
    from lib.trading.strategies.rb.breakout import (
        BreakoutType,
        RangeConfig,
        BreakoutResult,
        detect_range_breakout,
        DEFAULT_CONFIGS,
    )

    config = DEFAULT_CONFIGS[BreakoutType.InitialBalance]
    result = detect_range_breakout(bars_1m, symbol="MES=F", config=config)
    if result.breakout_detected:
        print(result.direction, result.trigger_price)
"""

from __future__ import annotations

import datetime as _dt
import logging
from dataclasses import dataclass, field
from datetime import datetime
from datetime import time as dt_time
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from lib.core.breakout_types import BreakoutType, RangeConfig, get_range_config

logger = logging.getLogger("engine.breakout")

_ET = ZoneInfo("America/New_York")
_UTC = ZoneInfo("UTC")


# ===========================================================================
# Phase 1A — BreakoutType is now the canonical IntEnum from src.lib.core.
#
# The old engine StrEnum has been removed.  All code in this module (and
# callers) uses ``lib.core.breakout_types.BreakoutType`` (IntEnum) directly.
#
# Short-name aliases ("ORB", "PDR", "CONS", …) are retained so that
# Redis/SSE payloads still serialise as ``"type": "ORB"`` via ``.name``.
#
# The former bridge functions ``to_training_type()``,
# ``from_training_type()``, and ``breakout_type_ordinal()`` have been
# removed — use the canonical ``BreakoutType`` enum and
# ``get_range_config(bt).breakout_type_ord`` directly.
# ===========================================================================

# Re-export so existing ``from lib.trading.strategies.rb.breakout import BreakoutType``
# keeps working — it now IS the core IntEnum.
__all__ = ["BreakoutType", "RangeConfig", "BreakoutResult", "detect_range_breakout", "DEFAULT_CONFIGS"]


# ---------------------------------------------------------------------------
# Short-name aliases for the IntEnum members so that engine code that
# previously wrote ``BreakoutType.PDR`` (StrEnum value "PDR") can still
# use ``BreakoutType.PDR`` which is now ``BreakoutType.PrevDay``.
# These are module-level constants — not enum members.
# ---------------------------------------------------------------------------
_SHORT_NAME_TO_BREAKOUT_TYPE: dict[str, BreakoutType] = {
    "ORB": BreakoutType.ORB,
    "PDR": BreakoutType.PrevDay,
    "IB": BreakoutType.InitialBalance,
    "CONS": BreakoutType.Consolidation,
    "WEEKLY": BreakoutType.Weekly,
    "MONTHLY": BreakoutType.Monthly,
    "ASIAN": BreakoutType.Asian,
    "BBSQUEEZE": BreakoutType.BollingerSqueeze,
    "VA": BreakoutType.ValueArea,
    "INSIDE": BreakoutType.InsideDay,
    "GAP": BreakoutType.GapRejection,
    "PIVOT": BreakoutType.PivotPoints,
    "FIB": BreakoutType.Fibonacci,
}

_BREAKOUT_TYPE_TO_SHORT_NAME: dict[BreakoutType, str] = {v: k for k, v in _SHORT_NAME_TO_BREAKOUT_TYPE.items()}


def breakout_type_from_short_name(name: str) -> BreakoutType:
    """Resolve a short engine name (``"PDR"``, ``"CONS"``, …) to a ``BreakoutType``.

    Also accepts the canonical ``.name`` form (``"PrevDay"``, ``"Consolidation"``, …).
    """
    upper = name.strip().upper()
    if upper in _SHORT_NAME_TO_BREAKOUT_TYPE:
        return _SHORT_NAME_TO_BREAKOUT_TYPE[upper]
    # Fall back to canonical name lookup
    from lib.core.breakout_types import breakout_type_from_name

    return breakout_type_from_name(name)


def breakout_type_short_name(bt: BreakoutType) -> str:
    """Return the short engine name for a ``BreakoutType`` (e.g. ``"PDR"`` for ``PrevDay``)."""
    return _BREAKOUT_TYPE_TO_SHORT_NAME.get(bt, bt.name)


# ---------------------------------------------------------------------------
# Phase 1A cleanup: ``to_training_type()``, ``from_training_type()``, and
# ``breakout_type_ordinal()`` have been removed.  They were identity
# functions after the StrEnum → IntEnum merge.
#
# Callers should use:
#   - ``BreakoutType`` directly (no conversion needed)
#   - ``get_range_config(bt).breakout_type_ord`` for the normalised ordinal
# ---------------------------------------------------------------------------


# ===========================================================================
# Phase 1B — DEFAULT_CONFIGS now delegates to the core RangeConfig registry.
#
# The engine-side ``RangeConfig`` class has been removed.  ``RangeConfig`` is
# now imported from ``lib.core.breakout_types`` (re-exported at module level).
# ``DEFAULT_CONFIGS`` is a convenience dict that maps each ``BreakoutType``
# to its canonical ``RangeConfig`` from the core registry — callers can also
# use ``get_range_config(bt)`` directly.
# ===========================================================================

DEFAULT_CONFIGS: dict[BreakoutType, RangeConfig] = {bt: get_range_config(bt) for bt in BreakoutType}


# ===========================================================================
# Result dataclass
# ===========================================================================


@dataclass
class BreakoutResult:
    """Result of ``detect_range_breakout()`` for any BreakoutType.

    Mirrors the fields of ``ORBResult`` so the same publishing / filtering /
    persistence pipeline can consume both without branching.

    ``breakout_type`` is the canonical ``lib.core.breakout_types.BreakoutType``
    (IntEnum).  For JSON serialisation, use ``.name`` for human-readable
    strings (``"ORB"``, ``"PrevDay"``, …) and ``.value`` for integer ordinals.
    """

    # --- Identity ---
    symbol: str = ""
    breakout_type: BreakoutType = BreakoutType.ORB  # IntEnum from core
    label: str = ""  # human-readable label from config

    # --- Range ---
    range_high: float = 0.0
    range_low: float = 0.0
    range_size: float = 0.0  # range_high − range_low
    atr_value: float = 0.0
    breakout_threshold: float = 0.0  # threshold = atr × multiplier

    # --- Breakout ---
    breakout_detected: bool = False
    direction: str = ""  # "LONG", "SHORT", or ""
    trigger_price: float = 0.0
    breakout_bar_time: str = ""

    # --- Levels ---
    long_trigger: float = 0.0  # range_high + threshold
    short_trigger: float = 0.0  # range_low  − threshold

    # --- Range formation ---
    range_complete: bool = False  # True once the range window has closed
    range_bar_count: int = 0
    evaluated_at: str = ""
    error: str = ""

    # --- Quality gate results ---
    depth_ok: bool | None = None
    body_ratio_ok: bool | None = None
    range_size_ok: bool | None = None
    breakout_bar_depth: float = 0.0
    breakout_bar_body_ratio: float = 0.0

    # --- Filter enrichment (set by engine after apply_all_filters) ---
    filter_passed: bool | None = None
    filter_summary: str = ""

    # --- MTF enrichment (set by engine after MTF analyzer) ---
    mtf_score: float | None = None  # 0.0 – 1.0 aggregate MTF score
    mtf_direction: str = ""  # "bullish", "bearish", "neutral"
    macd_slope: float | None = None  # MACD histogram slope (signed)
    macd_divergence: bool | None = None  # True if price/MACD diverge

    # --- CNN enrichment (set by engine after inference) ---
    cnn_prob: float | None = None
    cnn_confidence: str = ""
    cnn_signal: bool | None = None

    # --- Squeeze-specific (CONS type only) ---
    squeeze_detected: bool = False
    squeeze_bar_count: int = 0
    squeeze_bb_width: float = 0.0
    bb_upper: float = 0.0
    bb_lower: float = 0.0

    # --- PDR-specific ---
    prev_day_high: float = 0.0
    prev_day_low: float = 0.0
    prev_day_range: float = 0.0

    # --- IB-specific ---
    ib_high: float = 0.0
    ib_low: float = 0.0
    ib_complete: bool = False

    # --- Extra metadata ---
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-compatible dict.

        ``"type"`` is serialised as the human-readable ``.name`` (e.g. ``"ORB"``,
        ``"PrevDay"``) for backward compatibility with Redis/SSE consumers.
        ``"type_ordinal"`` carries the integer value for programmatic use.
        """
        # Use short engine name for backward compat ("PDR" not "PrevDay") if available
        type_str = _BREAKOUT_TYPE_TO_SHORT_NAME.get(self.breakout_type, self.breakout_type.name)
        d: dict[str, Any] = {
            "type": type_str,
            "type_name": self.breakout_type.name,
            "type_ordinal": int(self.breakout_type.value),
            "label": self.label,
            "symbol": self.symbol,
            "range_high": round(self.range_high, 4),
            "range_low": round(self.range_low, 4),
            "range_size": round(self.range_size, 4),
            "atr_value": round(self.atr_value, 4),
            "breakout_threshold": round(self.breakout_threshold, 4),
            "breakout_detected": self.breakout_detected,
            "direction": self.direction,
            "trigger_price": round(self.trigger_price, 4),
            "breakout_bar_time": self.breakout_bar_time,
            "long_trigger": round(self.long_trigger, 4),
            "short_trigger": round(self.short_trigger, 4),
            "range_complete": self.range_complete,
            "range_bar_count": self.range_bar_count,
            "evaluated_at": self.evaluated_at,
            "error": self.error,
            # Quality gates
            "depth_ok": self.depth_ok,
            "body_ratio_ok": self.body_ratio_ok,
            "range_size_ok": self.range_size_ok,
            "breakout_bar_depth": round(self.breakout_bar_depth, 6),
            "breakout_bar_body_ratio": round(self.breakout_bar_body_ratio, 4),
        }
        # Optional enrichment
        if self.filter_passed is not None:
            d["filter_passed"] = bool(self.filter_passed)
            d["filter_summary"] = self.filter_summary
        if self.mtf_score is not None:
            d["mtf_score"] = round(self.mtf_score, 4)
            d["mtf_direction"] = self.mtf_direction
        if self.macd_slope is not None:
            d["macd_slope"] = round(self.macd_slope, 6)
        if self.macd_divergence is not None:
            d["macd_divergence"] = bool(self.macd_divergence)
        if self.cnn_prob is not None:
            d["cnn_prob"] = round(self.cnn_prob, 4)
            d["cnn_confidence"] = self.cnn_confidence
            d["cnn_signal"] = bool(self.cnn_signal) if self.cnn_signal is not None else False
        # Type-specific
        if self.breakout_type in (BreakoutType.Consolidation, BreakoutType.BollingerSqueeze):
            d["squeeze_detected"] = bool(self.squeeze_detected)
            d["squeeze_bar_count"] = int(self.squeeze_bar_count)
            d["squeeze_bb_width"] = round(self.squeeze_bb_width, 4)
            d["bb_upper"] = round(self.bb_upper, 4)
            d["bb_lower"] = round(self.bb_lower, 4)
        if self.breakout_type == BreakoutType.PrevDay:
            d["prev_day_high"] = round(self.prev_day_high, 4)
            d["prev_day_low"] = round(self.prev_day_low, 4)
            d["prev_day_range"] = round(self.prev_day_range, 4)
        if self.breakout_type == BreakoutType.InitialBalance:
            d["ib_high"] = round(self.ib_high, 4)
            d["ib_low"] = round(self.ib_low, 4)
            d["ib_complete"] = bool(self.ib_complete)
        if self.extra:
            # Ensure all values in extra are JSON-serializable (convert numpy types)
            safe_extra: dict[str, Any] = {}
            for k, v in self.extra.items():
                if isinstance(v, (np.bool_, np.generic)):
                    safe_extra[k] = v.item()
                elif isinstance(v, bool):
                    safe_extra[k] = bool(v)
                elif isinstance(v, float):
                    safe_extra[k] = float(v)
                elif isinstance(v, int):
                    safe_extra[k] = int(v)
                else:
                    safe_extra[k] = v
            d["extra"] = safe_extra
        return d


# ===========================================================================
# Internal helpers
# ===========================================================================


def _compute_atr(bars: pd.DataFrame, period: int = 14) -> float:
    """Wilder ATR on a DataFrame with High / Low / Close columns.

    Returns 0.0 if there is insufficient data.
    """
    n = len(bars)
    if n < 2:
        return 0.0

    highs = bars["High"].astype(float).values
    lows = bars["Low"].astype(float).values
    closes = bars["Close"].astype(float).values

    tr = np.empty(n)
    tr[0] = highs[0] - lows[0]
    for i in range(1, n):
        tr[i] = max(
            float(highs[i] - lows[i]),
            abs(float(highs[i] - closes[i - 1])),
            abs(float(lows[i] - closes[i - 1])),
        )

    if n < period + 1:
        return float(np.mean(tr))

    atr = float(np.mean(tr[:period]))
    alpha = 1.0 / period
    for i in range(period, n):
        atr = alpha * tr[i] + (1.0 - alpha) * atr
    return atr


def _localize_bars(bars: pd.DataFrame) -> pd.DataFrame:
    """Ensure the bar index is a tz-aware DatetimeIndex in ET.

    Works whether the index is UTC, naive, or already ET.
    """
    if not isinstance(bars.index, pd.DatetimeIndex):
        return bars

    idx = bars.index
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    idx = idx.tz_convert(_ET)
    return bars.set_index(idx)


def _check_bar_quality(
    bar_open: float,
    bar_high: float,
    bar_low: float,
    bar_close: float,
    level: float,
    direction: str,
    atr: float,
    config: RangeConfig,
) -> tuple[bool, bool, float, float]:
    """Check depth and body-ratio quality gates for a candidate breakout bar.

    Returns:
        (depth_ok, body_ok, depth_value, body_ratio)
    """
    bar_range = bar_high - bar_low
    if bar_range <= 0:
        return False, False, 0.0, 0.0

    body = abs(bar_close - bar_open)
    body_ratio = body / bar_range

    depth = bar_close - level if direction == "LONG" else level - bar_close

    min_depth = config.min_depth_atr_pct * atr if atr > 0 else 0.0
    depth_ok = depth >= min_depth
    body_ok = body_ratio >= config.min_body_ratio

    return depth_ok, body_ok, max(depth, 0.0), body_ratio


# ===========================================================================
# Range builders  (one per BreakoutType)
# ===========================================================================


def _build_orb_range(
    bars: pd.DataFrame,
    config: RangeConfig,
    session_start: dt_time,
    session_end: dt_time,
) -> tuple[float, float, int, bool]:
    """Extract the opening-range high/low from bars within the OR window.

    Returns:
        (or_high, or_low, bar_count, complete)
    ``complete`` is True once the current bar timestamp is past session_end.
    """
    bars_et = _localize_bars(bars)
    now_et = pd.Timestamp(bars_et.index[-1]).to_pydatetime().time() if len(bars_et) > 0 else dt_time(0, 0)  # type: ignore[arg-type]

    idx_time = pd.DatetimeIndex(bars_et.index).time  # type: ignore[attr-defined]
    mask = (idx_time >= session_start) & (idx_time < session_end)
    or_bars = bars_et.loc[mask]

    if len(or_bars) < config.min_bars:
        return 0.0, 0.0, len(or_bars), False

    or_high = float(or_bars["High"].max())
    or_low = float(or_bars["Low"].min())
    complete = now_et >= session_end

    return or_high, or_low, len(or_bars), complete


def _build_pdr_range(
    bars: pd.DataFrame,
    config: RangeConfig,
) -> tuple[float, float, int, bool, float, float, float]:
    """Identify the previous Globex day's high/low from intraday 1m bars.

    Strategy: split bars at config.pdr_session_start (18:00 ET by default).
    Everything before today's session start is "previous day".

    Returns:
        (pdr_high, pdr_low, bar_count, complete, prev_high, prev_low, prev_range)
    ``complete`` is always True for PDR (the range is already formed).
    """
    bars_et = _localize_bars(bars)
    if len(bars_et) < 2:
        return 0.0, 0.0, 0, False, 0.0, 0.0, 0.0

    # Identify the current Globex-day start: the most recent bar whose time
    # equals or is just after pdr_session_start.
    session_start = config.pdr_session_start or dt_time(18, 0)  # 18:00 ET

    # Find the latest 18:00 ET boundary within the bar history
    idx_time_pdr = pd.DatetimeIndex(bars_et.index).time  # type: ignore[attr-defined]
    today_session_starts = bars_et.index[idx_time_pdr == session_start]

    if len(today_session_starts) == 0:
        # Fallback: use calendar midnight boundary
        today_et = pd.Timestamp(bars_et.index[-1]).to_pydatetime().date()  # type: ignore[arg-type]
        cutoff = pd.Timestamp(today_et, tz=_ET)  # type: ignore[arg-type]
        prev_bars = bars_et[bars_et.index < cutoff]
    else:
        # The most recent session-start boundary
        latest_start = today_session_starts[-1]
        prev_bars = bars_et[bars_et.index < latest_start]

    if len(prev_bars) < config.min_bars:
        # Fall back to daily bars if available: use the penultimate day's H/L
        # (caller may pass daily bars via bars_daily kwarg in the outer layer)
        return 0.0, 0.0, len(prev_bars), False, 0.0, 0.0, 0.0

    prev_high = float(prev_bars["High"].max())  # type: ignore[arg-type]
    prev_low = float(prev_bars["Low"].min())  # type: ignore[arg-type]
    prev_range = prev_high - prev_low
    bar_count = len(prev_bars)

    return prev_high, prev_low, bar_count, True, prev_high, prev_low, prev_range


def _build_ib_range(
    bars: pd.DataFrame,
    config: RangeConfig,
) -> tuple[float, float, int, bool]:
    """Build the Initial Balance range (first ``ib_duration_minutes`` of RTH).

    Returns:
        (ib_high, ib_low, bar_count, complete)
    ``complete`` is True once the current time is past the IB end time.
    """
    bars_et = _localize_bars(bars)
    if len(bars_et) < 1:
        return 0.0, 0.0, 0, False

    ib_start = config.ib_start_time or dt_time(9, 30)
    ib_end_minutes = ib_start.hour * 60 + ib_start.minute + config.ib_duration_minutes
    ib_end_h, ib_end_m = divmod(ib_end_minutes, 60)
    ib_end = dt_time(int(ib_end_h), int(ib_end_m))

    idx_time_ib = pd.DatetimeIndex(bars_et.index).time  # type: ignore[attr-defined]
    mask = (idx_time_ib >= ib_start) & (idx_time_ib < ib_end)
    ib_bars = bars_et.loc[mask]

    bar_count = len(ib_bars)
    now_et = pd.Timestamp(bars_et.index[-1]).to_pydatetime().time()  # type: ignore[arg-type]
    complete = now_et >= ib_end

    if bar_count < config.min_bars:
        return 0.0, 0.0, bar_count, complete

    ib_high = float(ib_bars["High"].max())  # type: ignore[arg-type]
    ib_low = float(ib_bars["Low"].min())  # type: ignore[arg-type]

    return ib_high, ib_low, bar_count, complete


def _build_consolidation_range(
    bars: pd.DataFrame,
    config: RangeConfig,
    atr: float,
) -> tuple[float, float, int, bool, float, float, float, int, float]:
    """Detect a Bollinger Band / ATR consolidation squeeze and extract its range.

    A "squeeze" is present when the BB bandwidth (upper − lower) is smaller
    than ``squeeze_atr_mult × ATR`` for at least ``squeeze_min_bars``
    consecutive bars.

    Returns:
        (cons_high, cons_low, bar_count, squeeze_detected,
         bb_upper, bb_lower, bb_width, squeeze_bar_count, current_bb_width)
    """
    if len(bars) < config.squeeze_bb_period + 2 or atr <= 0:
        return 0.0, 0.0, 0, False, 0.0, 0.0, 0.0, 0, 0.0

    close = bars["Close"].astype(float)
    n = config.squeeze_bb_period
    std_mult = config.squeeze_bb_std

    # Rolling Bollinger Bands
    bb_mid = pd.Series(close.rolling(n).mean())
    bb_std_s = pd.Series(close.rolling(n).std(ddof=0))
    bb_upper = pd.Series(bb_mid + std_mult * bb_std_s)
    bb_lower = pd.Series(bb_mid - std_mult * bb_std_s)
    bb_width = pd.Series((bb_upper - bb_lower).fillna(0.0))

    threshold = config.squeeze_atr_mult * atr

    # Count consecutive squeeze bars at the end of the series
    squeeze_flags = pd.Series(bb_width < threshold)
    squeeze_bar_count = 0
    for i in range(len(squeeze_flags) - 1, -1, -1):
        if squeeze_flags.iloc[i]:
            squeeze_bar_count += 1
        else:
            break

    squeeze_detected = squeeze_bar_count >= config.squeeze_min_bars
    current_bb_width = float(bb_width.iloc[-1]) if len(bb_width) > 0 else 0.0
    current_bb_upper = float(bb_upper.iloc[-1]) if len(bb_upper) > 0 else 0.0
    current_bb_lower = float(bb_lower.iloc[-1]) if len(bb_lower) > 0 else 0.0

    if not squeeze_detected:
        return 0.0, 0.0, 0, False, current_bb_upper, current_bb_lower, current_bb_width, 0, current_bb_width

    # The consolidation range is the BB upper/lower at the squeeze boundary
    squeeze_start_idx = len(bars) - squeeze_bar_count
    squeeze_slice = bars.iloc[squeeze_start_idx:]
    cons_high = float(squeeze_slice["High"].max())  # type: ignore[arg-type]
    cons_low = float(squeeze_slice["Low"].min())  # type: ignore[arg-type]
    bar_count = len(squeeze_slice)

    return (
        cons_high,
        cons_low,
        bar_count,
        True,
        current_bb_upper,
        current_bb_lower,
        current_bb_width,
        squeeze_bar_count,
        current_bb_width,
    )


def _build_weekly_range(
    bars: pd.DataFrame,
    config: RangeConfig,
) -> tuple[float, float, int, bool]:
    """Extract the prior week's high/low from intraday bars.

    Uses ``config.weekly_lookback_days`` (default 5) to identify bars
    belonging to the prior trading week.  The range is the high/low of
    those bars.

    Returns:
        (week_high, week_low, bar_count, complete)
    ``complete`` is True once at least ``min_bars`` prior-week bars exist.
    """
    bars_et = _localize_bars(bars)
    if len(bars_et) < 2:
        return 0.0, 0.0, 0, False

    today = pd.Timestamp(bars_et.index[-1]).to_pydatetime().date()  # type: ignore[arg-type]
    # Walk back to find the start of this week (Monday)
    weekday = today.weekday()  # Monday=0 .. Sunday=6
    this_week_start = today - _dt.timedelta(days=int(weekday))

    # Prior week bars: everything before this week's Monday
    cutoff = pd.Timestamp(this_week_start, tz=_ET)  # type: ignore[arg-type]
    prev_week_end = cutoff
    prev_week_start = cutoff - pd.Timedelta(days=config.weekly_lookback_days)

    mask = (bars_et.index >= prev_week_start) & (bars_et.index < prev_week_end)
    prev_bars = bars_et.loc[mask]

    if len(prev_bars) < config.min_bars:
        return 0.0, 0.0, len(prev_bars), False

    week_high = float(prev_bars["High"].max())
    week_low = float(prev_bars["Low"].min())
    return week_high, week_low, len(prev_bars), True


def _build_monthly_range(
    bars: pd.DataFrame,
    config: RangeConfig,
) -> tuple[float, float, int, bool]:
    """Extract the prior month's high/low from intraday bars.

    Uses ``config.monthly_lookback_days`` (default 20) calendar trading
    days prior to the 1st of the current month.

    Returns:
        (month_high, month_low, bar_count, complete)
    """
    bars_et = _localize_bars(bars)
    if len(bars_et) < 2:
        return 0.0, 0.0, 0, False

    today = pd.Timestamp(bars_et.index[-1]).to_pydatetime().date()  # type: ignore[arg-type]
    # First day of current month
    first_of_month = today.replace(day=1)
    cutoff = pd.Timestamp(first_of_month, tz=_ET)  # type: ignore[arg-type]
    lookback_start = cutoff - pd.Timedelta(days=config.monthly_lookback_days)

    mask = (bars_et.index >= lookback_start) & (bars_et.index < cutoff)
    prev_bars = bars_et.loc[mask]

    if len(prev_bars) < config.min_bars:
        return 0.0, 0.0, len(prev_bars), False

    month_high = float(prev_bars["High"].max())
    month_low = float(prev_bars["Low"].min())
    return month_high, month_low, len(prev_bars), True


def _build_asian_range(
    bars: pd.DataFrame,
    config: RangeConfig,
) -> tuple[float, float, int, bool]:
    """Extract the Asian session range (19:00–02:00 ET, wraps midnight).

    The range is from ``config.asian_start_time`` (19:00 ET) to
    ``config.asian_end_time`` (02:00 ET).  Because this window wraps
    midnight, bars whose time >= start OR time < end are included.

    Returns:
        (asian_high, asian_low, bar_count, complete)
    ``complete`` is True once the current time is past the end time.
    """
    bars_et = _localize_bars(bars)
    if len(bars_et) < 1:
        return 0.0, 0.0, 0, False

    start = config.asian_start_time or dt_time(19, 0)  # 19:00 ET
    end = config.asian_end_time or dt_time(2, 0)  # 02:00 ET
    now_time = pd.Timestamp(bars_et.index[-1]).to_pydatetime().time()  # type: ignore[arg-type]

    idx_time = pd.DatetimeIndex(bars_et.index).time  # type: ignore[attr-defined]

    # Wraps midnight: time >= 19:00 OR time < 02:00
    mask = (idx_time >= start) | (idx_time < end) if start > end else (idx_time >= start) & (idx_time < end)

    asian_bars = bars_et.loc[mask]
    bar_count = len(asian_bars)

    # Complete once we're past the end time and NOT in the start window
    # (i.e. we're in the 02:00–19:00 window of the next day)
    complete = end <= now_time < start if start > end else now_time >= end

    if bar_count < config.min_bars:
        return 0.0, 0.0, bar_count, complete

    asian_high = float(asian_bars["High"].max())
    asian_low = float(asian_bars["Low"].min())
    return asian_high, asian_low, bar_count, complete


def _build_bbsqueeze_range(
    bars: pd.DataFrame,
    config: RangeConfig,
    atr: float,
) -> tuple[float, float, int, bool, int, float, float, float]:
    """Detect a Bollinger Band inside Keltner Channel squeeze and extract its range.

    A "squeeze" exists when the Bollinger Bands (period/std from config) are
    fully inside the Keltner Channel (EMA ± kc_atr_mult × ATR) for at least
    ``bbsqueeze_min_squeeze_bars`` consecutive bars.

    Returns:
        (range_high, range_low, bar_count, squeeze_detected,
         squeeze_bar_count, bb_width, bb_upper, bb_lower)
    """
    n_bb = config.bbsqueeze_bb_period
    if len(bars) < n_bb + 2 or atr <= 0:
        return 0.0, 0.0, 0, False, 0, 0.0, 0.0, 0.0

    close = bars["Close"].astype(float)
    high = bars["High"].astype(float)
    low = bars["Low"].astype(float)

    # Bollinger Bands
    bb_mid = close.rolling(n_bb).mean()
    bb_std = close.rolling(n_bb).std(ddof=0)
    bb_upper = bb_mid + config.bbsqueeze_bb_std * bb_std
    bb_lower = bb_mid - config.bbsqueeze_bb_std * bb_std

    # Keltner Channel (EMA-based mid + ATR envelope)
    n_kc = config.bbsqueeze_kc_period
    kc_mid = close.ewm(span=n_kc, adjust=False).mean()

    # Per-bar ATR for KC (rolling)
    tr_series = pd.concat(
        [
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    rolling_atr = tr_series.rolling(n_kc).mean()

    kc_upper = pd.Series(kc_mid + config.bbsqueeze_kc_atr_mult * rolling_atr)
    kc_lower = pd.Series(kc_mid - config.bbsqueeze_kc_atr_mult * rolling_atr)

    # Squeeze: BB fully inside KC  (bb_upper < kc_upper AND bb_lower > kc_lower)
    # Wrap in pd.Series to guarantee .fillna/.iloc work — pyright loses the
    # Series type through the rolling/ewm arithmetic chain.
    squeeze_flags = pd.Series(  # type: ignore[call-overload]
        pd.Series(bb_upper).lt(pd.Series(kc_upper))  # type: ignore[arg-type]
        & pd.Series(bb_lower).gt(pd.Series(kc_lower))  # type: ignore[arg-type]
    ).fillna(False)

    # Count consecutive squeeze bars at the tail
    squeeze_bar_count = 0
    for i in range(len(squeeze_flags) - 1, -1, -1):
        if squeeze_flags.iloc[i]:
            squeeze_bar_count += 1
        else:
            break

    squeeze_detected = squeeze_bar_count >= config.bbsqueeze_min_squeeze_bars

    cur_bb_upper = float(pd.Series(bb_upper).iloc[-1]) if len(bb_upper) > 0 else 0.0  # type: ignore[arg-type]
    cur_bb_lower = float(pd.Series(bb_lower).iloc[-1]) if len(bb_lower) > 0 else 0.0  # type: ignore[arg-type]
    cur_bb_width = cur_bb_upper - cur_bb_lower

    if not squeeze_detected:
        return 0.0, 0.0, 0, False, 0, cur_bb_width, cur_bb_upper, cur_bb_lower

    squeeze_start_idx = len(bars) - squeeze_bar_count
    squeeze_slice = bars.iloc[squeeze_start_idx:]
    r_high = float(squeeze_slice["High"].max())
    r_low = float(squeeze_slice["Low"].min())

    return r_high, r_low, len(squeeze_slice), True, squeeze_bar_count, cur_bb_width, cur_bb_upper, cur_bb_lower


def _build_va_range(
    bars: pd.DataFrame,
    config: RangeConfig,
) -> tuple[float, float, int, bool, float, float, float]:
    """Build Value Area range (VAH/VAL) from prior session's volume profile.

    Uses the same Globex-day split as PDR (18:00 ET boundary) to isolate
    the prior session's bars and compute volume profile on them.

    Returns:
        (vah, val, bar_count, complete, poc, vah, val)
    """
    bars_et = _localize_bars(bars)
    if len(bars_et) < 10:
        return 0.0, 0.0, 0, False, 0.0, 0.0, 0.0

    # Split at 18:00 ET to get prior session bars (same logic as PDR)
    session_start = dt_time(18, 0)
    idx_time = pd.DatetimeIndex(bars_et.index).time  # type: ignore[attr-defined]
    today_session_starts = bars_et.index[idx_time == session_start]

    if len(today_session_starts) == 0:
        today_et = pd.Timestamp(bars_et.index[-1]).to_pydatetime().date()  # type: ignore[arg-type]
        cutoff = pd.Timestamp(today_et, tz=_ET)  # type: ignore[arg-type]
        prev_bars = bars_et[bars_et.index < cutoff]
    else:
        latest_start = today_session_starts[-1]
        prev_bars = bars_et[bars_et.index < latest_start]

    if len(prev_bars) < config.min_bars:
        return 0.0, 0.0, len(prev_bars), False, 0.0, 0.0, 0.0

    # Check that volume data exists
    if "Volume" not in prev_bars.columns or prev_bars["Volume"].sum() <= 0:
        # Fall back to simple H/L percentile if no volume data
        logger.debug("_build_va_range: no volume data — falling back to price-based VA")
        sorted_closes: pd.Series = prev_bars["Close"].astype(float).sort_values()  # type: ignore[assignment]
        n = len(sorted_closes)
        val = float(sorted_closes.iloc[int(n * 0.15)])
        vah = float(sorted_closes.iloc[int(n * 0.85)])
        poc = float(sorted_closes.median())
        return vah, val, len(prev_bars), True, poc, vah, val

    try:
        from lib.analysis.volume_profile import compute_volume_profile

        profile = compute_volume_profile(
            prev_bars,  # type: ignore[arg-type]
            n_bins=config.va_n_bins,
            value_area_pct=config.va_value_area_pct,
        )
        vah = float(profile["vah"])
        val = float(profile["val"])
        poc = float(profile["poc"])
    except Exception as exc:
        logger.warning("_build_va_range: volume profile computation failed: %s", exc)
        return 0.0, 0.0, len(prev_bars), False, 0.0, 0.0, 0.0

    if vah <= val or vah <= 0:
        return 0.0, 0.0, len(prev_bars), False, 0.0, 0.0, 0.0

    return vah, val, len(prev_bars), True, poc, vah, val


def _build_inside_day_range(
    bars: pd.DataFrame,
    config: RangeConfig,
) -> tuple[float, float, int, bool, float, float, float, float]:
    """Detect an inside day and return the mother bar's range.

    An "inside day" occurs when today's high/low are both contained within
    yesterday's high/low.  The breakout range is yesterday's H/L (the
    "mother bar").

    Returns:
        (mother_high, mother_low, bar_count, inside_detected,
         today_high, today_low, yesterday_high, yesterday_low)
    """
    bars_et = _localize_bars(bars)
    if len(bars_et) < 2:
        return 0.0, 0.0, 0, False, 0.0, 0.0, 0.0, 0.0

    # Split into today vs yesterday using 18:00 ET Globex boundary
    session_start = dt_time(18, 0)
    idx_time = pd.DatetimeIndex(bars_et.index).time  # type: ignore[attr-defined]
    session_starts = bars_et.index[idx_time == session_start]

    if len(session_starts) < 2:
        # Not enough sessions to compare — try calendar day fallback
        today_et = pd.Timestamp(bars_et.index[-1]).to_pydatetime().date()  # type: ignore[arg-type]
        cutoff = pd.Timestamp(today_et, tz=_ET)  # type: ignore[arg-type]
        today_bars = bars_et[bars_et.index >= cutoff]
        yesterday_bars = bars_et[bars_et.index < cutoff]
    else:
        latest_start = session_starts[-1]
        prev_start = session_starts[-2]
        today_bars = bars_et[bars_et.index >= latest_start]
        yesterday_bars = bars_et[(bars_et.index >= prev_start) & (bars_et.index < latest_start)]

    if len(today_bars) < 1 or len(yesterday_bars) < 1:
        return 0.0, 0.0, 0, False, 0.0, 0.0, 0.0, 0.0

    today_high = float(today_bars["High"].max())  # type: ignore[arg-type]
    today_low = float(today_bars["Low"].min())  # type: ignore[arg-type]
    yesterday_high = float(yesterday_bars["High"].max())  # type: ignore[arg-type]
    yesterday_low = float(yesterday_bars["Low"].min())  # type: ignore[arg-type]

    # Inside day: today's range is fully contained within yesterday's
    inside = today_high <= yesterday_high and today_low >= yesterday_low

    if not inside:
        return 0.0, 0.0, 0, False, today_high, today_low, yesterday_high, yesterday_low

    # Check compression ratio
    yesterday_range = yesterday_high - yesterday_low
    today_range = today_high - today_low
    if yesterday_range <= 0:
        return 0.0, 0.0, 0, False, today_high, today_low, yesterday_high, yesterday_low

    compression = today_range / yesterday_range
    if compression < config.inside_min_compression or compression > config.inside_max_compression:
        return 0.0, 0.0, 0, False, today_high, today_low, yesterday_high, yesterday_low

    # Breakout range is the mother bar (yesterday's H/L)
    return (
        yesterday_high,
        yesterday_low,
        len(yesterday_bars),
        True,
        today_high,
        today_low,
        yesterday_high,
        yesterday_low,
    )


def _build_gap_rejection_range(
    bars: pd.DataFrame,
    config: RangeConfig,
    atr: float,
) -> tuple[float, float, int, bool, float, float, str]:
    """Detect an overnight gap and build the gap zone as the breakout range.

    A gap exists when today's open is meaningfully separated from yesterday's
    close.  The breakout range is defined by the gap boundaries:
      - Gap up:   range = [yesterday_close, today_open]
      - Gap down: range = [today_open, yesterday_close]

    Returns:
        (range_high, range_low, bar_count, gap_detected,
         gap_size, yesterday_close, gap_direction)
    ``gap_direction`` is "UP" or "DOWN".
    """
    bars_et = _localize_bars(bars)
    if len(bars_et) < 2 or atr <= 0:
        return 0.0, 0.0, 0, False, 0.0, 0.0, ""

    # Split at 18:00 ET Globex boundary
    session_start = dt_time(18, 0)
    idx_time = pd.DatetimeIndex(bars_et.index).time  # type: ignore[attr-defined]
    session_starts = bars_et.index[idx_time == session_start]

    if len(session_starts) < 1:
        today_et = pd.Timestamp(bars_et.index[-1]).to_pydatetime().date()  # type: ignore[arg-type]
        cutoff = pd.Timestamp(today_et, tz=_ET)  # type: ignore[arg-type]
        today_bars = bars_et[bars_et.index >= cutoff]
        yesterday_bars = bars_et[bars_et.index < cutoff]
    else:
        latest_start = session_starts[-1]
        today_bars = bars_et[bars_et.index >= latest_start]
        yesterday_bars = bars_et[bars_et.index < latest_start]

    if len(today_bars) < 1 or len(yesterday_bars) < 1:
        return 0.0, 0.0, 0, False, 0.0, 0.0, ""

    yesterday_close = float(yesterday_bars["Close"].iloc[-1])  # type: ignore[arg-type]
    today_open = (
        float(today_bars["Open"].iloc[0]) if "Open" in today_bars.columns else float(today_bars["Close"].iloc[0])  # type: ignore[arg-type]
    )

    gap_size = today_open - yesterday_close
    min_gap = config.gap_min_atr_pct * atr

    if abs(gap_size) < min_gap:
        return 0.0, 0.0, 0, False, gap_size, yesterday_close, ""

    if gap_size > 0:
        # Gap up: range is [yesterday_close, today_open]
        r_high = today_open
        r_low = yesterday_close
        direction = "UP"
    else:
        # Gap down: range is [today_open, yesterday_close]
        r_high = yesterday_close
        r_low = today_open
        direction = "DOWN"

    return r_high, r_low, len(today_bars), True, gap_size, yesterday_close, direction


def _build_pivot_range(
    bars: pd.DataFrame,
    config: RangeConfig,
) -> tuple[float, float, int, bool, float, float, float, float]:
    """Compute classic floor pivots from prior session's HLC and return S1/R1 as the range.

    Supported formulas (``config.pivot_formula``):
      - ``classic``: P = (H+L+C)/3, R1 = 2P−L, S1 = 2P−H
      - ``woodie``:  P = (H+L+2C)/4, R1 = 2P−L, S1 = 2P−H
      - ``camarilla``: P = (H+L+C)/3, R1 = C+1.1*(H−L)/12, S1 = C−1.1*(H−L)/12

    Returns:
        (r1, s1, bar_count, complete, pivot, r1, s1, prev_close)
    """
    bars_et = _localize_bars(bars)
    if len(bars_et) < 2:
        return 0.0, 0.0, 0, False, 0.0, 0.0, 0.0, 0.0

    # Get prior session bars (same split as PDR)
    session_start = dt_time(18, 0)
    idx_time = pd.DatetimeIndex(bars_et.index).time  # type: ignore[attr-defined]
    session_starts = bars_et.index[idx_time == session_start]

    if len(session_starts) < 1:
        today_et = pd.Timestamp(bars_et.index[-1]).to_pydatetime().date()  # type: ignore[arg-type]
        cutoff = pd.Timestamp(today_et, tz=_ET)  # type: ignore[arg-type]
        prev_bars = bars_et[bars_et.index < cutoff]
    else:
        latest_start = session_starts[-1]
        prev_bars = bars_et[bars_et.index < latest_start]

    if len(prev_bars) < config.min_bars:
        return 0.0, 0.0, len(prev_bars), False, 0.0, 0.0, 0.0, 0.0

    h = float(prev_bars["High"].max())  # type: ignore[arg-type]
    prev_low = float(prev_bars["Low"].min())  # type: ignore[arg-type]
    c = float(prev_bars["Close"].iloc[-1])  # type: ignore[arg-type]

    formula = config.pivot_formula.lower()
    if formula == "woodie":
        pivot = (h + prev_low + 2 * c) / 4.0
        r1 = 2.0 * pivot - prev_low
        s1 = 2.0 * pivot - h
    elif formula == "camarilla":
        pivot = (h + prev_low + c) / 3.0
        r1 = c + 1.1 * (h - prev_low) / 12.0
        s1 = c - 1.1 * (h - prev_low) / 12.0
    else:  # classic
        pivot = (h + prev_low + c) / 3.0
        r1 = 2.0 * pivot - prev_low
        s1 = 2.0 * pivot - h

    if r1 <= s1 or r1 <= 0 or s1 <= 0:
        return 0.0, 0.0, len(prev_bars), False, pivot, r1, s1, c

    return r1, s1, len(prev_bars), True, pivot, r1, s1, c


def _build_fibonacci_range(
    bars: pd.DataFrame,
    config: RangeConfig,
    atr: float,
) -> tuple[float, float, int, bool, float, float, float, float]:
    """Find the prior swing high/low and compute the Fibonacci retracement zone.

    Identifies the highest high and lowest low within the last
    ``config.fib_swing_lookback`` bars.  The swing must be at least
    ``config.fib_min_swing_atr_mult × ATR`` in size.  The retracement
    zone between ``fib_lower`` (38.2%) and ``fib_upper`` (61.8%) of that
    swing becomes the breakout range.

    Returns:
        (fib_high, fib_low, bar_count, valid,
         swing_high, swing_low, fib_382, fib_618)
    """
    lookback = config.fib_swing_lookback
    if len(bars) < 10 or atr <= 0:
        return 0.0, 0.0, 0, False, 0.0, 0.0, 0.0, 0.0

    # Use the last N bars for swing detection
    window = bars.iloc[-lookback:] if len(bars) > lookback else bars

    swing_high = float(window["High"].max())  # type: ignore[arg-type]
    swing_low = float(window["Low"].min())  # type: ignore[arg-type]
    swing_size = swing_high - swing_low

    if swing_size < config.fib_min_swing_atr_mult * atr:
        return 0.0, 0.0, 0, False, swing_high, swing_low, 0.0, 0.0

    # Determine swing direction: is the high more recent than the low?
    high_idx = window["High"].astype(float).values.argmax()
    low_idx = window["Low"].astype(float).values.argmin()

    if high_idx > low_idx:
        # Upswing — retracement is measured downward from swing high
        fib_382 = swing_high - config.fib_lower * swing_size  # 38.2% retrace
        fib_618 = swing_high - config.fib_upper * swing_size  # 61.8% retrace
        # Range: fib_618 (lower) to fib_382 (upper)
        r_high = fib_382
        r_low = fib_618
    else:
        # Downswing — retracement is measured upward from swing low
        fib_382 = swing_low + config.fib_lower * swing_size
        fib_618 = swing_low + config.fib_upper * swing_size
        r_high = fib_618
        r_low = fib_382

    if r_high <= r_low or r_high <= 0:
        return 0.0, 0.0, 0, False, swing_high, swing_low, fib_382, fib_618

    return r_high, r_low, len(window), True, swing_high, swing_low, fib_382, fib_618


# ===========================================================================
# Breakout scanner (shared logic for all types)
# ===========================================================================


def _scan_for_breakout(
    bars: pd.DataFrame,
    range_high: float,
    range_low: float,
    atr: float,
    config: RangeConfig,
    scan_start_time: dt_time | None = None,
) -> tuple[bool, str, float, str, float, float]:
    """Scan bars after range formation for a breakout close beyond H/L.

    Only bars whose ET wall-clock time is >= scan_start_time (if supplied)
    are considered.

    Returns:
        (detected, direction, trigger_price, bar_time_str, depth, body_ratio)
    """
    if range_high <= 0 or range_low <= 0 or range_high <= range_low:
        return False, "", 0.0, "", 0.0, 0.0

    threshold = atr * config.atr_multiplier
    long_trigger = range_high + threshold
    short_trigger = range_low - threshold

    bars_et = _localize_bars(bars)

    for ts, row in bars_et.iterrows():
        bar_time = (
            pd.Timestamp(ts).to_pydatetime().time()  # type: ignore[arg-type]
            if hasattr(ts, "to_pydatetime")
            else (ts.time() if hasattr(ts, "time") else dt_time(0, 0))  # type: ignore[union-attr]
        )
        if scan_start_time is not None and bar_time < scan_start_time:
            continue

        bar_open = float(row["Open"] if "Open" in row.index else row["Close"] if "Close" in row.index else 0.0)  # type: ignore[arg-type]
        bar_high = float(row["High"] if "High" in row.index else 0.0)  # type: ignore[arg-type]
        bar_low = float(row["Low"] if "Low" in row.index else 0.0)  # type: ignore[arg-type]
        bar_close = float(row["Close"] if "Close" in row.index else 0.0)  # type: ignore[arg-type]

        if bar_close <= 0:
            continue

        direction = ""
        level = 0.0

        if bar_close > long_trigger:
            direction = "LONG"
            level = long_trigger
        elif bar_close < short_trigger:
            direction = "SHORT"
            level = short_trigger

        if direction:
            depth_ok, body_ok, depth, body_ratio = _check_bar_quality(
                bar_open,
                bar_high,
                bar_low,
                bar_close,
                level,
                direction,
                atr,
                config,
            )
            if depth_ok and body_ok:
                bar_time_str = str(pd.Timestamp(ts).isoformat()) if hasattr(ts, "isoformat") else str(ts)  # type: ignore[arg-type]
                return True, direction, bar_close, bar_time_str, depth, body_ratio

    return False, "", 0.0, "", 0.0, 0.0


# ===========================================================================
# Main public function
# ===========================================================================


def detect_range_breakout(
    bars: pd.DataFrame,
    symbol: str,
    config: RangeConfig | None = None,
    *,
    # ORB-specific overrides
    orb_session_start: dt_time | None = None,
    orb_session_end: dt_time | None = None,
    orb_scan_start: dt_time | None = None,
    # PDR override: supply pre-computed daily high/low
    prev_day_high: float | None = None,
    prev_day_low: float | None = None,
    # IB override: supply pre-computed IB high/low
    ib_high: float | None = None,
    ib_low: float | None = None,
    # Backward-compat: accept old short name string for breakout_type
    breakout_type_name: str | None = None,
) -> BreakoutResult:
    """Unified breakout detector for ORB, PDR, IB, and Consolidation types.

    Args:
        bars: 1-minute OHLCV DataFrame (tz-aware or UTC).  Must have columns
              ``Open``, ``High``, ``Low``, ``Close`` and a DatetimeIndex.
        symbol: Instrument symbol (for logging / result annotation).
        config: ``RangeConfig`` describing the breakout type and thresholds.
                Defaults to ``DEFAULT_CONFIGS[BreakoutType.ORB]`` if omitted.
        orb_session_start: ET time for ORB range start (ORB type only).
        orb_session_end: ET time for ORB range end (ORB type only).
        orb_scan_start: ET time after which breakout scanning begins
                        (defaults to orb_session_end).
        prev_day_high: Override PDR high (skips internal range building).
        prev_day_low: Override PDR low.
        ib_high: Override IB high (skips internal range building).
        ib_low: Override IB low.

    Returns:
        ``BreakoutResult`` populated with range, quality gate verdicts,
        and breakout detection state.  Never raises — errors surface in
        ``result.error``.
    """
    from datetime import datetime as _dt

    if config is None:
        if breakout_type_name is not None:
            bt = breakout_type_from_short_name(breakout_type_name)
            config = DEFAULT_CONFIGS.get(bt, DEFAULT_CONFIGS[BreakoutType.ORB])
        else:
            config = DEFAULT_CONFIGS[BreakoutType.ORB]

    now_str = _dt.now(tz=_ET).isoformat()
    result = BreakoutResult(
        symbol=symbol,
        breakout_type=config.breakout_type,
        label=config.label or config.breakout_type.name,
        evaluated_at=now_str,
    )

    if bars is None or bars.empty:
        result.error = "No bar data supplied"
        return result

    required_cols = {"High", "Low", "Close"}
    if not required_cols.issubset(bars.columns):
        result.error = f"Missing columns: {required_cols - set(bars.columns)}"
        return result

    # --- ATR ---
    atr = _compute_atr(bars, period=config.atr_period)
    result.atr_value = round(atr, 6)

    # --- Build range (type-specific) ---
    btype: BreakoutType = config.breakout_type  # initialised before try so except block can reference it
    try:
        btype = config.breakout_type

        if btype == BreakoutType.ORB:
            s_start = orb_session_start or dt_time(9, 30)
            s_end = orb_session_end or dt_time(10, 0)
            scan_start = orb_scan_start or s_end

            r_high, r_low, bar_count, complete = _build_orb_range(bars, config, s_start, s_end)
            result.range_complete = complete
            result.range_bar_count = bar_count

        elif btype == BreakoutType.PrevDay:
            if prev_day_high is not None and prev_day_low is not None:
                r_high, r_low = prev_day_high, prev_day_low
                bar_count = 1
                complete = True
                result.prev_day_high = r_high
                result.prev_day_low = r_low
                result.prev_day_range = r_high - r_low
            else:
                (r_high, r_low, bar_count, complete, pdr_high, pdr_low, pdr_range) = _build_pdr_range(bars, config)
                result.prev_day_high = pdr_high
                result.prev_day_low = pdr_low
                result.prev_day_range = pdr_range

            result.range_complete = complete
            result.range_bar_count = bar_count
            scan_start = None  # PDR: always scan latest bars

        elif btype == BreakoutType.InitialBalance:
            if ib_high is not None and ib_low is not None:
                r_high, r_low = ib_high, ib_low
                bar_count = 0
                complete = True
            else:
                r_high, r_low, bar_count, complete = _build_ib_range(bars, config)

            result.ib_high = r_high
            result.ib_low = r_low
            result.ib_complete = complete
            result.range_complete = complete
            result.range_bar_count = bar_count

            # IB breakout scan starts after IB window closes
            _ib_start = config.ib_start_time or dt_time(9, 30)
            ib_end_min = _ib_start.hour * 60 + _ib_start.minute + config.ib_duration_minutes
            ib_end_h, ib_end_m = divmod(int(ib_end_min), 60)
            scan_start = dt_time(ib_end_h, ib_end_m)

        elif btype == BreakoutType.Consolidation:
            (
                r_high,
                r_low,
                bar_count,
                squeeze_detected,
                bb_upper,
                bb_lower,
                bb_width,
                squeeze_bar_count,
                current_bb_width,
            ) = _build_consolidation_range(bars, config, atr)
            result.squeeze_detected = squeeze_detected
            result.squeeze_bar_count = squeeze_bar_count
            result.squeeze_bb_width = round(current_bb_width, 4)
            result.bb_upper = round(bb_upper, 4)
            result.bb_lower = round(bb_lower, 4)
            result.range_complete = squeeze_detected
            result.range_bar_count = bar_count
            scan_start = None  # scan the very latest bar for the expansion

            if not squeeze_detected:
                result.error = "No squeeze detected — cannot form consolidation range"
                return result

        elif btype == BreakoutType.Weekly:
            r_high, r_low, bar_count, complete = _build_weekly_range(bars, config)
            result.range_complete = complete
            result.range_bar_count = bar_count
            result.extra["weekly_high"] = round(r_high, 4)
            result.extra["weekly_low"] = round(r_low, 4)
            scan_start = None

        elif btype == BreakoutType.Monthly:
            r_high, r_low, bar_count, complete = _build_monthly_range(bars, config)
            result.range_complete = complete
            result.range_bar_count = bar_count
            result.extra["monthly_high"] = round(r_high, 4)
            result.extra["monthly_low"] = round(r_low, 4)
            scan_start = None

        elif btype == BreakoutType.Asian:
            r_high, r_low, bar_count, complete = _build_asian_range(bars, config)
            result.range_complete = complete
            result.range_bar_count = bar_count
            result.extra["asian_high"] = round(r_high, 4)
            result.extra["asian_low"] = round(r_low, 4)
            # Scan for breakout after the Asian window closes
            scan_start = config.asian_end_time or dt_time(2, 0)

        elif btype == BreakoutType.BollingerSqueeze:
            (
                r_high,
                r_low,
                bar_count,
                squeeze_detected,
                squeeze_bar_count,
                bb_width,
                bb_upper_val,
                bb_lower_val,
            ) = _build_bbsqueeze_range(bars, config, atr)
            result.squeeze_detected = squeeze_detected
            result.squeeze_bar_count = squeeze_bar_count
            result.squeeze_bb_width = round(bb_width, 4)
            result.bb_upper = round(bb_upper_val, 4)
            result.bb_lower = round(bb_lower_val, 4)
            result.range_complete = squeeze_detected
            result.range_bar_count = bar_count
            scan_start = None

            if not squeeze_detected:
                result.error = "No BB-inside-KC squeeze detected"
                return result

        elif btype == BreakoutType.ValueArea:
            (
                r_high,
                r_low,
                bar_count,
                complete,
                poc,
                vah,
                val,
            ) = _build_va_range(bars, config)
            result.range_complete = complete
            result.range_bar_count = bar_count
            result.extra["poc"] = round(poc, 4)
            result.extra["vah"] = round(vah, 4)
            result.extra["val"] = round(val, 4)
            scan_start = None

        elif btype == BreakoutType.InsideDay:
            (
                r_high,
                r_low,
                bar_count,
                inside_detected,
                today_high,
                today_low,
                yesterday_high,
                yesterday_low,
            ) = _build_inside_day_range(bars, config)
            result.range_complete = inside_detected
            result.range_bar_count = bar_count
            result.extra["inside_detected"] = inside_detected
            result.extra["today_high"] = round(today_high, 4)
            result.extra["today_low"] = round(today_low, 4)
            result.extra["yesterday_high"] = round(yesterday_high, 4)
            result.extra["yesterday_low"] = round(yesterday_low, 4)
            scan_start = None

            if not inside_detected:
                result.error = "No inside day detected — today's range not inside yesterday's"
                return result

        elif btype == BreakoutType.GapRejection:
            (
                r_high,
                r_low,
                bar_count,
                gap_detected,
                gap_size,
                yesterday_close,
                gap_direction,
            ) = _build_gap_rejection_range(bars, config, atr)
            result.range_complete = gap_detected
            result.range_bar_count = bar_count
            result.extra["gap_detected"] = gap_detected
            result.extra["gap_size"] = round(gap_size, 4)
            result.extra["yesterday_close"] = round(yesterday_close, 4)
            result.extra["gap_direction"] = gap_direction
            scan_start = None

            if not gap_detected:
                result.error = "No significant gap detected"
                return result

        elif btype == BreakoutType.PivotPoints:
            (
                r_high,
                r_low,
                bar_count,
                complete,
                pivot,
                r1,
                s1,
                prev_close,
            ) = _build_pivot_range(bars, config)
            result.range_complete = complete
            result.range_bar_count = bar_count
            result.extra["pivot"] = round(pivot, 4)
            result.extra["r1"] = round(r1, 4)
            result.extra["s1"] = round(s1, 4)
            result.extra["prev_close"] = round(prev_close, 4)
            scan_start = None

        elif btype == BreakoutType.Fibonacci:
            (
                r_high,
                r_low,
                bar_count,
                valid,
                swing_high,
                swing_low,
                fib_382,
                fib_618,
            ) = _build_fibonacci_range(bars, config, atr)
            result.range_complete = valid
            result.range_bar_count = bar_count
            result.extra["swing_high"] = round(swing_high, 4)
            result.extra["swing_low"] = round(swing_low, 4)
            result.extra["fib_382"] = round(fib_382, 4)
            result.extra["fib_618"] = round(fib_618, 4)
            scan_start = None

            if not valid:
                result.error = "No valid Fibonacci swing found (swing too small or insufficient data)"
                return result

        else:
            result.error = f"Unknown BreakoutType: {btype}"
            return result

    except Exception as exc:
        result.error = f"Range build error: {exc}"
        logger.warning("detect_range_breakout[%s] range build error for %s: %s", btype, symbol, exc)
        return result

    # --- Populate common range fields ---
    result.range_high = round(r_high, 4)
    result.range_low = round(r_low, 4)
    result.range_size = round(r_high - r_low, 4)

    threshold = atr * config.atr_multiplier
    result.breakout_threshold = round(threshold, 4)
    result.long_trigger = round(r_high + threshold, 4)
    result.short_trigger = round(r_low - threshold, 4)

    # --- Range size quality gate ---
    if atr > 0:
        range_atr_ratio = result.range_size / atr
        size_ok = config.min_range_atr_ratio <= range_atr_ratio <= config.max_range_atr_ratio
    else:
        size_ok = result.range_size > 0

    result.range_size_ok = size_ok

    if not size_ok:
        logger.debug(
            "detect_range_breakout[%s] %s: range size %.4f / ATR %.4f = %.2f out of [%.2f, %.2f]",
            btype,
            symbol,
            result.range_size,
            atr,
            result.range_size / atr if atr > 0 else 0,
            config.min_range_atr_ratio,
            config.max_range_atr_ratio,
        )
        return result  # breakout_detected stays False

    if r_high <= 0 or r_low <= 0 or r_high <= r_low:
        return result

    # --- Scan for breakout ---
    try:
        detected, direction, trigger, bar_time_str, depth, body_ratio = _scan_for_breakout(
            bars, r_high, r_low, atr, config, scan_start_time=scan_start
        )
    except Exception as exc:
        result.error = f"Breakout scan error: {exc}"
        logger.warning("detect_range_breakout[%s] scan error for %s: %s", btype, symbol, exc)
        return result

    result.breakout_detected = detected
    result.direction = direction
    result.trigger_price = round(trigger, 4)
    result.breakout_bar_time = bar_time_str
    result.breakout_bar_depth = round(depth, 6)
    result.breakout_bar_body_ratio = round(body_ratio, 4)

    if detected:
        # Recompute quality gate verdicts now that we have bar data
        result.depth_ok = depth >= (config.min_depth_atr_pct * atr if atr > 0 else 0)
        result.body_ratio_ok = body_ratio >= config.min_body_ratio

        logger.info(
            "🔔 %s BREAKOUT [%s]: %s %s @ %.4f (range %.4f–%.4f, ATR %.4f)",
            breakout_type_short_name(config.breakout_type),
            config.label or btype,
            direction,
            symbol,
            trigger,
            r_low,
            r_high,
            atr,
        )
    else:
        logger.debug(
            "detect_range_breakout[%s] %s: no breakout (range %.4f–%.4f)",
            btype,
            symbol,
            r_low,
            r_high,
        )

    return result


# ===========================================================================
# Convenience helpers for the engine handler layer
# ===========================================================================


def detect_pdr_breakout(
    bars: pd.DataFrame,
    symbol: str,
    prev_day_high: float | None = None,
    prev_day_low: float | None = None,
    config: RangeConfig | None = None,
) -> BreakoutResult:
    """Shortcut: detect a Previous Day Range breakout."""
    cfg = config or DEFAULT_CONFIGS[BreakoutType.PrevDay]
    return detect_range_breakout(
        bars,
        symbol,
        cfg,
        prev_day_high=prev_day_high,
        prev_day_low=prev_day_low,
    )


def detect_ib_breakout(
    bars: pd.DataFrame,
    symbol: str,
    ib_high: float | None = None,
    ib_low: float | None = None,
    config: RangeConfig | None = None,
) -> BreakoutResult:
    """Shortcut: detect an Initial Balance breakout."""
    cfg = config or DEFAULT_CONFIGS[BreakoutType.InitialBalance]
    return detect_range_breakout(
        bars,
        symbol,
        cfg,
        ib_high=ib_high,
        ib_low=ib_low,
    )


def detect_consolidation_breakout(
    bars: pd.DataFrame,
    symbol: str,
    config: RangeConfig | None = None,
) -> BreakoutResult:
    """Shortcut: detect a Consolidation/Squeeze breakout."""
    cfg = config or DEFAULT_CONFIGS[BreakoutType.Consolidation]
    return detect_range_breakout(bars, symbol, cfg)


def detect_all_breakout_types(
    bars: pd.DataFrame,
    symbol: str,
    types: list[BreakoutType] | None = None,
    configs: dict[BreakoutType, RangeConfig] | None = None,
    prev_day_high: float | None = None,
    prev_day_low: float | None = None,
    ib_high: float | None = None,
    ib_low: float | None = None,
    orb_session_start: dt_time | None = None,
    orb_session_end: dt_time | None = None,
) -> dict[BreakoutType, BreakoutResult]:
    """Run all (or a subset of) breakout type detectors for a single symbol.

    Args:
        bars: 1-minute OHLCV DataFrame.
        symbol: Instrument symbol.
        types: List of ``BreakoutType`` values to check.  Defaults to all four.
        configs: Override configs per type.  Falls back to ``DEFAULT_CONFIGS``.
        prev_day_high: Pre-computed PDR high (optional).
        prev_day_low: Pre-computed PDR low (optional).
        ib_high: Pre-computed IB high (optional).
        ib_low: Pre-computed IB low (optional).
        orb_session_start: ET time for ORB session start.
        orb_session_end: ET time for ORB session end.

    Returns:
        Dict mapping each ``BreakoutType`` to its ``BreakoutResult``.
    """
    if types is None:
        types = list(BreakoutType)
    merged_configs = {**DEFAULT_CONFIGS, **(configs or {})}

    results: dict[BreakoutType, BreakoutResult] = {}

    for btype in types:
        cfg = merged_configs.get(
            btype, DEFAULT_CONFIGS.get(btype, RangeConfig(breakout_type=btype, breakout_type_ord=btype.value / 12.0))
        )
        try:
            result = detect_range_breakout(
                bars,
                symbol,
                cfg,
                orb_session_start=orb_session_start,
                orb_session_end=orb_session_end,
                prev_day_high=prev_day_high,
                prev_day_low=prev_day_low,
                ib_high=ib_high,
                ib_low=ib_low,
            )
        except Exception as exc:
            logger.warning("detect_all_breakout_types[%s] error for %s: %s", btype, symbol, exc)
            result = BreakoutResult(
                symbol=symbol,
                breakout_type=btype,
                label=cfg.label,
                error=str(exc),
                evaluated_at=datetime.now(tz=_ET).isoformat(),
            )
        results[btype] = result

    return results
