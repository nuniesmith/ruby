"""
Range Breakout Filters — Quality-over-Quantity Gate
====================================================
Research-backed filters that sit between any range breakout detector and
signal publication.  Each filter returns a simple (pass: bool, reason: str)
tuple so callers can compose them freely.

These filters apply to ALL 13 range breakout types (ORB, PrevDay,
InitialBalance, Consolidation, Weekly, Monthly, Asian, BollingerSqueeze,
ValueArea, InsideDay, GapRejection, PivotPoints, Fibonacci) — not just ORB.

Filters implemented:
  1. NR7 (Narrow Range 7)     — Toby Crabel's #1 ORB filter
  2. Pre-Market Range Break    — Globex high/low confluence
  3. Session Window            — only allow signals inside optimal ET windows
  4. Lunch / Dead-Zone         — reject 10:30–13:30 ET chop
  5. Multi-TF EMA Bias         — higher-timeframe trend agreement (simple)
  6. MTF Analyzer              — full EMA alignment + MACD momentum + divergence

Public API:
    from lib.analysis.breakout_filters import (
        check_nr7,
        check_premarket_range,
        check_session_window,
        check_lunch_filter,
        check_multi_tf_bias,
        check_mtf_analyzer,
        apply_all_filters,
        BreakoutFilterResult,
    )

    result = apply_all_filters(
        bars_daily, bars_1m, direction="LONG",
        premarket_high=2345.6, premarket_low=2330.1,
        orb_high=2340.0, orb_low=2332.0,
        signal_time=datetime.now(tz=ZoneInfo("America/New_York")),
    )
    if result.passed:
        publish_signal(...)
    else:
        logger.info("Filtered: %s", result.summary)

Backward compatibility:
    ``ORBFilterResult`` is kept as an alias for ``BreakoutFilterResult`` so
    existing callers (``main.py``, ``mtf_analyzer.py``, tests) work unchanged
    until they are migrated to the new name.

Design:
  - Pure functions — no Redis, no side-effects, fully testable.
  - Each filter is independent so the engine can enable/disable via config.
  - ``apply_all_filters`` runs them all and returns a composite verdict.
"""

from __future__ import annotations

import contextlib
import logging
from dataclasses import dataclass, field
from datetime import datetime
from datetime import time as dt_time
from typing import TYPE_CHECKING, Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from lib.indicators.helpers import ema_numpy as _ema

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger("analysis.breakout_filters")

_EST = ZoneInfo("America/New_York")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class FilterVerdict:
    """Result of a single filter check."""

    name: str
    passed: bool
    reason: str = ""
    score_boost: float = 0.0  # optional quality-score adjustment (-1.0 … +1.0)

    def __str__(self) -> str:
        status = "✅" if self.passed else "❌"
        return f"{status} {self.name}: {self.reason}"


@dataclass
class ORBFilterResult:
    """Composite result from running all range breakout filters."""

    passed: bool = False
    verdicts: list[FilterVerdict] = field(default_factory=list)
    filters_passed: int = 0
    filters_total: int = 0
    quality_boost: float = 0.0  # aggregate score adjustment
    summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "filters_passed": self.filters_passed,
            "filters_total": self.filters_total,
            "quality_boost": round(self.quality_boost, 3),
            "summary": self.summary,
            "verdicts": [
                {"name": v.name, "passed": v.passed, "reason": v.reason, "score_boost": v.score_boost}
                for v in self.verdicts
            ],
        }


# ---------------------------------------------------------------------------
# Backward-compatible alias (Phase 1E rename: orb_filters → breakout_filters)
# ---------------------------------------------------------------------------
# ``ORBFilterResult`` keeps its original name so existing callers
# (``main.py``, ``mtf_analyzer.py``, ``analysis/__init__.py``, tests) work
# unchanged until they are migrated to the new name.

#: New canonical name for the composite filter result dataclass.
BreakoutFilterResult = ORBFilterResult


# ---------------------------------------------------------------------------
# 1. NR7 — Narrow Range 7
# ---------------------------------------------------------------------------
# Toby Crabel (1990): When today's daily range is the narrowest of the
# last 7 trading days, the subsequent ORB breakout has significantly
# higher follow-through.  A compressed range implies coiled energy.


def check_nr7(
    bars_daily: pd.DataFrame | None,
    lookback: int = 7,
    boost_pct: float = 0.20,
) -> FilterVerdict:
    """Check whether the most recent daily bar is an NR7 (Narrow Range 7).

    An NR7 day is one where today's high-low range is the *smallest*
    of the prior ``lookback`` trading days (including today).

    This filter does NOT reject signals — it *boosts* the quality score
    when present, and is neutral otherwise.

    Args:
        bars_daily: Daily OHLCV DataFrame with at least ``lookback`` rows.
                    Must have 'High' and 'Low' columns.
        lookback: Number of days to compare (default 7 per Crabel).
        boost_pct: Quality-score boost when NR7 is detected (0.0–1.0).

    Returns:
        FilterVerdict with passed=True always (NR7 is a bonus, not a gate).
        ``score_boost`` is positive when NR7 is active, 0 otherwise.
    """
    name = "NR7"

    if bars_daily is None or bars_daily.empty:
        return FilterVerdict(name=name, passed=True, reason="No daily bars — NR7 skipped")

    if len(bars_daily) < lookback:
        return FilterVerdict(
            name=name,
            passed=True,
            reason=f"Only {len(bars_daily)} daily bars (need {lookback}) — NR7 skipped",
        )

    try:
        highs = np.asarray(bars_daily["High"].astype(float).values)
        lows = np.asarray(bars_daily["Low"].astype(float).values)
    except (KeyError, ValueError) as exc:
        return FilterVerdict(name=name, passed=True, reason=f"Missing High/Low columns: {exc}")

    # Compute daily ranges for the last ``lookback`` bars
    recent_highs = highs[-lookback:]
    recent_lows = lows[-lookback:]
    daily_ranges = recent_highs - recent_lows

    # Guard against zero-range days
    if np.all(daily_ranges <= 0):
        return FilterVerdict(name=name, passed=True, reason="All zero-range days — NR7 skipped")

    today_range = daily_ranges[-1]
    is_nr7 = today_range <= np.min(daily_ranges)

    if is_nr7:
        return FilterVerdict(
            name=name,
            passed=True,
            reason=f"NR7 active — today range {today_range:.4f} is narrowest of {lookback} days",
            score_boost=boost_pct,
        )

    rank = int(np.sum(daily_ranges < today_range)) + 1  # 1-based rank
    return FilterVerdict(
        name=name,
        passed=True,
        reason=f"Not NR7 — today range ranks #{rank}/{lookback} (range={today_range:.4f})",
        score_boost=0.0,
    )


# ---------------------------------------------------------------------------
# 2. Pre-Market Range Break (Globex Confluence)
# ---------------------------------------------------------------------------
# The ORB breakout is much more reliable when the breakout direction also
# clears the Globex / pre-market extreme.  If the ORB breaks high but
# price is still *below* the pre-market high, follow-through is weak.


def check_premarket_range(
    direction: str,
    trigger_price: float,
    premarket_high: float | None = None,
    premarket_low: float | None = None,
    tolerance_pct: float = 0.001,
) -> FilterVerdict:
    """Check whether the ORB breakout also clears the pre-market range extreme.

    For a LONG signal: trigger_price should be >= premarket_high.
    For a SHORT signal: trigger_price should be <= premarket_low.

    A small tolerance (default 0.1%) accounts for tick rounding.

    Args:
        direction: "LONG" or "SHORT".
        trigger_price: The price at which the ORB breakout triggered.
        premarket_high: Highest price during the Globex / pre-market session.
        premarket_low: Lowest price during the Globex / pre-market session.
        tolerance_pct: Percentage tolerance for near-miss (0.001 = 0.1%).

    Returns:
        FilterVerdict — fails if the breakout did NOT clear the PM extreme.
    """
    name = "Pre-Market Range"
    direction_upper = direction.upper().strip()

    if direction_upper == "LONG":
        if premarket_high is None or premarket_high <= 0:
            return FilterVerdict(name=name, passed=True, reason="No pre-market high available — skipped")

        threshold = premarket_high * (1.0 - tolerance_pct)
        if trigger_price >= threshold:
            clearance = trigger_price - premarket_high
            return FilterVerdict(
                name=name,
                passed=True,
                reason=f"LONG clears PM high {premarket_high:.4f} by {clearance:+.4f}",
                score_boost=0.10 if clearance > 0 else 0.0,
            )
        else:
            gap = premarket_high - trigger_price
            return FilterVerdict(
                name=name,
                passed=False,
                reason=f"LONG trigger {trigger_price:.4f} below PM high {premarket_high:.4f} by {gap:.4f}",
            )

    elif direction_upper == "SHORT":
        if premarket_low is None or premarket_low <= 0:
            return FilterVerdict(name=name, passed=True, reason="No pre-market low available — skipped")

        threshold = premarket_low * (1.0 + tolerance_pct)
        if trigger_price <= threshold:
            clearance = premarket_low - trigger_price
            return FilterVerdict(
                name=name,
                passed=True,
                reason=f"SHORT clears PM low {premarket_low:.4f} by {clearance:+.4f}",
                score_boost=0.10 if clearance > 0 else 0.0,
            )
        else:
            gap = trigger_price - premarket_low
            return FilterVerdict(
                name=name,
                passed=False,
                reason=f"SHORT trigger {trigger_price:.4f} above PM low {premarket_low:.4f} by {gap:.4f}",
            )

    return FilterVerdict(name=name, passed=True, reason=f"Unknown direction '{direction}' — skipped")


# ---------------------------------------------------------------------------
# 3. Session Window Filter
# ---------------------------------------------------------------------------
# Empirically, ORB breakouts cluster in specific time windows:
#   - Metals (MGC/GC):  08:20–09:00 ET (London/NY overlap)
#   - Indices (MES/MNQ): 09:30–10:30 ET (cash open + first hour)
# Signals outside these windows have much lower follow-through.

# Default allowed windows (ET) — each tuple is (start_time, end_time).
DEFAULT_SESSION_WINDOWS: list[tuple[dt_time, dt_time]] = [
    (dt_time(8, 20), dt_time(10, 30)),  # Primary: London/NY overlap → first hour
]


def check_session_window(
    signal_time: datetime,
    allowed_windows: Sequence[tuple[dt_time, dt_time]] | None = None,
) -> FilterVerdict:
    """Check whether the signal time falls within an allowed trading window.

    Args:
        signal_time: Timezone-aware datetime of the signal (or naive, assumed ET).
        allowed_windows: List of (start, end) time tuples in Eastern Time.
                         Defaults to DEFAULT_SESSION_WINDOWS.

    Returns:
        FilterVerdict — fails if signal is outside all allowed windows.
    """
    name = "Session Window"
    windows = allowed_windows if allowed_windows is not None else DEFAULT_SESSION_WINDOWS

    # Ensure we have an ET time
    signal_et = signal_time.replace(tzinfo=_EST) if signal_time.tzinfo is None else signal_time.astimezone(_EST)

    t = signal_et.time()

    for start, end in windows:
        if start <= t <= end:
            return FilterVerdict(
                name=name,
                passed=True,
                reason=f"Signal at {t.strftime('%H:%M')} ET is within window {start.strftime('%H:%M')}–{end.strftime('%H:%M')}",
            )

    windows_str = ", ".join(f"{s.strftime('%H:%M')}–{e.strftime('%H:%M')}" for s, e in windows)
    return FilterVerdict(
        name=name,
        passed=False,
        reason=f"Signal at {t.strftime('%H:%M')} ET is outside allowed windows [{windows_str}]",
    )


# ---------------------------------------------------------------------------
# 4. Lunch / Dead-Zone Filter
# ---------------------------------------------------------------------------
# Breakouts between ~10:30 and ~13:30 ET fail at approximately 80% rate.
# This is the "lunch chop" zone where volume dries up and reversals dominate.

LUNCH_START = dt_time(10, 30)
LUNCH_END = dt_time(13, 30)


def check_lunch_filter(
    signal_time: datetime,
    lunch_start: dt_time = LUNCH_START,
    lunch_end: dt_time = LUNCH_END,
) -> FilterVerdict:
    """Reject signals during the lunch / dead-zone period.

    The default dead zone is 10:30–13:30 ET.  This is separate from the
    session window filter so it can be toggled independently (e.g. you
    might widen the session window but still keep the lunch block).

    Args:
        signal_time: Timezone-aware datetime of the signal.
        lunch_start: Start of dead zone (default 10:30 ET).
        lunch_end: End of dead zone (default 13:30 ET).

    Returns:
        FilterVerdict — fails if signal falls within the lunch zone.
    """
    name = "Lunch Filter"

    signal_et = signal_time.replace(tzinfo=_EST) if signal_time.tzinfo is None else signal_time.astimezone(_EST)

    t = signal_et.time()

    if lunch_start <= t <= lunch_end:
        return FilterVerdict(
            name=name,
            passed=False,
            reason=f"Signal at {t.strftime('%H:%M')} ET is in dead zone {lunch_start.strftime('%H:%M')}–{lunch_end.strftime('%H:%M')}",
        )

    return FilterVerdict(
        name=name,
        passed=True,
        reason=f"Signal at {t.strftime('%H:%M')} ET is outside lunch zone",
    )


# ---------------------------------------------------------------------------
# 5. Multi-Timeframe EMA Bias
# ---------------------------------------------------------------------------
# The ORB breakout direction must agree with the higher-timeframe trend.
# We compute a simple EMA slope on 15-minute (or configurable) bars and
# require it to point in the same direction as the breakout.


# _ema is imported from lib.indicators.helpers (ema_numpy aliased as _ema above).


# ---------------------------------------------------------------------------
# 6. MTF Analyzer — Full EMA Alignment + MACD Momentum + Divergence
# ---------------------------------------------------------------------------
# Delegates to lib.analysis.mtf_analyzer for the full computation.
# Exposed as a standalone filter so it can be toggled individually in
# apply_all_filters() and used from any caller without pulling in the
# heavier MTFAnalyzer class.


def check_mtf_analyzer(
    bars_htf: pd.DataFrame | None,
    direction: str,
    min_pass_score: float = 0.55,
    ema_fast: int = 9,
    ema_mid: int = 21,
    ema_slow: int = 50,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal_period: int = 9,
    divergence_lookback: int = 20,
) -> FilterVerdict:
    """Full MTF quality gate: EMA alignment/slope + MACD momentum + divergence.

    This is a richer alternative (and complement) to ``check_multi_tf_bias``:
      - ``check_multi_tf_bias`` checks only EMA slope direction.
      - ``check_mtf_analyzer`` additionally evaluates MACD histogram polarity,
        MACD histogram slope, and regular price/MACD divergence.

    Hard rejection criteria:
      1. Opposing MACD divergence detected (price diverges against breakout).
      2. Overall MTF score < ``min_pass_score``.

    Score composition (0.0–1.0):
      0.30  EMA fast/mid/slow fully stacked in breakout direction
      0.15  EMA slow slope trending in breakout direction
      0.25  MACD histogram polarity agrees with direction
      0.15  MACD histogram slope agrees with direction
      0.15  No opposing divergence

    Args:
        bars_htf: Higher-timeframe OHLCV bars (e.g. 15m). Needs ≥ 60 bars
                  for reliable MACD computation (macd_slow + macd_signal + slope).
        direction: Breakout direction — "LONG" or "SHORT".
        min_pass_score: Minimum score to pass (default 0.55).
        ema_fast: Fast EMA period (default 9).
        ema_mid: Mid EMA period (default 21).
        ema_slow: Slow EMA period (default 50).
        macd_fast: MACD fast EMA period (default 12).
        macd_slow: MACD slow EMA period (default 26).
        macd_signal_period: MACD signal line period (default 9).
        divergence_lookback: Bars to scan for divergence (default 20).

    Returns:
        FilterVerdict — fails if score < min_pass_score or opposing divergence.
    """
    name = "MTF Analyzer"

    try:
        from lib.analysis.mtf_analyzer import MTFAnalyzer

        analyzer = MTFAnalyzer(
            ema_fast=ema_fast,
            ema_mid=ema_mid,
            ema_slow=ema_slow,
            macd_fast=macd_fast,
            macd_slow=macd_slow,
            macd_signal=macd_signal_period,
            divergence_lookback=divergence_lookback,
            min_pass_score=min_pass_score,
        )
        _result, verdict = analyzer.evaluate(bars_htf, direction)
        # Re-stamp the name in case the inner module uses a different string
        return FilterVerdict(
            name=name,
            passed=verdict.passed,
            reason=verdict.reason,
            score_boost=verdict.score_boost,
        )
    except ImportError:
        return FilterVerdict(
            name=name,
            passed=True,
            reason="MTF analyzer module not available — skipped",
        )
    except Exception as exc:
        logger.warning("check_mtf_analyzer error (non-fatal): %s", exc)
        return FilterVerdict(
            name=name,
            passed=True,
            reason=f"MTF analyzer error — skipped: {exc}",
        )


def check_multi_tf_bias(
    bars_htf: pd.DataFrame | None,
    direction: str,
    ema_period: int = 34,
    slope_bars: int = 3,
    min_slope_pct: float = 0.0,
) -> FilterVerdict:
    """Check that the higher-timeframe EMA slope agrees with the breakout direction.

    Computes EMA(``ema_period``) on the close prices of ``bars_htf`` (typically
    15-minute bars), then measures the slope over the last ``slope_bars`` periods.

    A LONG breakout requires a positive (or flat) slope.
    A SHORT breakout requires a negative (or flat) slope.

    Args:
        bars_htf: Higher-timeframe OHLCV DataFrame (e.g. 15m bars).
        direction: "LONG" or "SHORT".
        ema_period: EMA period (default 34 — standard intermediate trend).
        slope_bars: Number of bars to measure slope over (default 3).
        min_slope_pct: Minimum absolute slope % to count as directional.
                       0.0 means any positive slope counts for LONG.

    Returns:
        FilterVerdict — fails if the HTF trend opposes the breakout.
    """
    name = "Multi-TF Bias"
    direction_upper = direction.upper().strip()

    if bars_htf is None or bars_htf.empty:
        return FilterVerdict(name=name, passed=True, reason="No HTF bars available — skipped")

    min_required = ema_period + slope_bars + 1
    if len(bars_htf) < min_required:
        return FilterVerdict(
            name=name,
            passed=True,
            reason=f"Only {len(bars_htf)} HTF bars (need {min_required}) — skipped",
        )

    try:
        closes = np.asarray(bars_htf["Close"].astype(float).values)
    except (KeyError, ValueError) as exc:
        return FilterVerdict(name=name, passed=True, reason=f"Missing Close column: {exc}")

    ema_values = _ema(np.asarray(closes), ema_period)

    # Slope: change in EMA over last `slope_bars` periods, normalised by price
    ema_now = ema_values[-1]
    ema_prev = ema_values[-(slope_bars + 1)]

    if ema_prev <= 0:
        return FilterVerdict(name=name, passed=True, reason="EMA is zero — skipped")

    slope_pct = (ema_now - ema_prev) / ema_prev

    slope_direction = "FLAT"
    if slope_pct > min_slope_pct:
        slope_direction = "UP"
    elif slope_pct < -min_slope_pct:
        slope_direction = "DOWN"

    if direction_upper == "LONG":
        if slope_direction == "DOWN":
            return FilterVerdict(
                name=name,
                passed=False,
                reason=f"LONG vs DOWN HTF EMA slope ({slope_pct:+.4%}) — counter-trend rejected",
            )
        return FilterVerdict(
            name=name,
            passed=True,
            reason=f"LONG agrees with {slope_direction} HTF EMA slope ({slope_pct:+.4%})",
            score_boost=0.05 if slope_direction == "UP" else 0.0,
        )

    elif direction_upper == "SHORT":
        if slope_direction == "UP":
            return FilterVerdict(
                name=name,
                passed=False,
                reason=f"SHORT vs UP HTF EMA slope ({slope_pct:+.4%}) — counter-trend rejected",
            )
        return FilterVerdict(
            name=name,
            passed=True,
            reason=f"SHORT agrees with {slope_direction} HTF EMA slope ({slope_pct:+.4%})",
            score_boost=0.05 if slope_direction == "DOWN" else 0.0,
        )

    return FilterVerdict(name=name, passed=True, reason=f"Unknown direction '{direction}' — skipped")


# ---------------------------------------------------------------------------
# 6. VWAP Confluence (Bonus)
# ---------------------------------------------------------------------------
# Price should be above VWAP for longs and below VWAP for shorts.
# This is a simple but effective confluence check.


def compute_session_vwap(bars_1m: pd.DataFrame | None) -> float | None:
    """Compute the session VWAP from 1-minute bars.

    VWAP = cumsum(Typical_Price × Volume) / cumsum(Volume)
    where Typical_Price = (High + Low + Close) / 3

    Returns the current (latest) VWAP value, or None if insufficient data.
    """
    if bars_1m is None or bars_1m.empty or len(bars_1m) < 2:
        return None

    try:
        high = np.asarray(bars_1m["High"].astype(float).values)
        low = np.asarray(bars_1m["Low"].astype(float).values)
        close = np.asarray(bars_1m["Close"].astype(float).values)
        volume = np.asarray(bars_1m["Volume"].astype(float).values)
    except (KeyError, ValueError):
        return None

    typical_price = (high + low + close) / 3.0
    cum_tp_vol = np.cumsum(typical_price * volume)
    cum_vol = np.cumsum(volume)

    if cum_vol[-1] <= 0:
        return None

    return float(cum_tp_vol[-1] / cum_vol[-1])


def check_vwap_confluence(
    bars_1m: pd.DataFrame | None,
    direction: str,
    trigger_price: float,
    vwap: float | None = None,
) -> FilterVerdict:
    """Check that the breakout trigger price is on the correct side of VWAP.

    LONG:  trigger_price >= VWAP  (buying above value → strength).
    SHORT: trigger_price <= VWAP  (selling below value → weakness).

    Args:
        bars_1m: 1-minute OHLCV bars for VWAP computation.
        direction: "LONG" or "SHORT".
        trigger_price: The ORB breakout trigger price.
        vwap: Pre-computed VWAP value (if None, will compute from bars).

    Returns:
        FilterVerdict.
    """
    name = "VWAP Confluence"
    direction_upper = direction.upper().strip()

    if vwap is None:
        vwap = compute_session_vwap(bars_1m)

    if vwap is None or vwap <= 0:
        return FilterVerdict(name=name, passed=True, reason="VWAP unavailable — skipped")

    if direction_upper == "LONG":
        if trigger_price >= vwap:
            return FilterVerdict(
                name=name,
                passed=True,
                reason=f"LONG trigger {trigger_price:.4f} >= VWAP {vwap:.4f}",
                score_boost=0.05,
            )
        gap = vwap - trigger_price
        return FilterVerdict(
            name=name,
            passed=False,
            reason=f"LONG trigger {trigger_price:.4f} below VWAP {vwap:.4f} by {gap:.4f}",
        )

    elif direction_upper == "SHORT":
        if trigger_price <= vwap:
            return FilterVerdict(
                name=name,
                passed=True,
                reason=f"SHORT trigger {trigger_price:.4f} <= VWAP {vwap:.4f}",
                score_boost=0.05,
            )
        gap = trigger_price - vwap
        return FilterVerdict(
            name=name,
            passed=False,
            reason=f"SHORT trigger {trigger_price:.4f} above VWAP {vwap:.4f} by {gap:.4f}",
        )

    return FilterVerdict(name=name, passed=True, reason=f"Unknown direction '{direction}' — skipped")


# ---------------------------------------------------------------------------
# Composite: apply_all_filters
# ---------------------------------------------------------------------------


def apply_all_filters(
    direction: str,
    trigger_price: float,
    signal_time: datetime,
    bars_daily: pd.DataFrame | None = None,
    bars_1m: pd.DataFrame | None = None,
    bars_htf: pd.DataFrame | None = None,
    premarket_high: float | None = None,
    premarket_low: float | None = None,
    orb_high: float | None = None,
    orb_low: float | None = None,
    vwap: float | None = None,
    # Configuration
    enable_nr7: bool = True,
    enable_premarket: bool = True,
    enable_session_window: bool = True,
    enable_lunch_filter: bool = True,
    enable_multi_tf: bool = True,
    enable_vwap: bool = True,
    enable_mtf_analyzer: bool = True,
    allowed_windows: Sequence[tuple[dt_time, dt_time]] | None = None,
    nr7_lookback: int = 7,
    ema_period: int = 34,
    mtf_min_pass_score: float = 0.55,
    # Gating mode: "all" requires every enabled hard filter to pass,
    # "majority" requires > 50% of hard filters to pass.
    gate_mode: str = "all",
) -> ORBFilterResult:
    """Run all enabled ORB quality filters and return a composite result.

    Filters are split into two categories:
      - **Hard filters** (session window, lunch, pre-market, multi-TF, VWAP,
        MTF analyzer): these can reject a signal outright.
      - **Soft filters** (NR7): these only adjust the quality score boost;
        they never reject a signal.

    The MTF Analyzer (``enable_mtf_analyzer``) is the richest hard filter —
    it evaluates EMA alignment/slope, MACD histogram polarity/slope, and
    regular divergence.  When both ``enable_multi_tf`` and
    ``enable_mtf_analyzer`` are True, the simpler EMA-slope-only filter
    (``check_multi_tf_bias``) still runs as a lightweight early gate while
    the full analyzer adds momentum and divergence checks.  If you want only
    the full analyzer, set ``enable_multi_tf=False``.

    Args:
        direction: "LONG" or "SHORT".
        trigger_price: The price that triggered the ORB breakout.
        signal_time: When the signal was generated (tz-aware preferred).
        bars_daily: Daily OHLCV bars for NR7 (at least 7 rows).
        bars_1m: 1-minute OHLCV bars for VWAP computation.
        bars_htf: Higher-timeframe bars (e.g. 15m) for multi-TF bias.
        premarket_high: Globex session high.
        premarket_low: Globex session low.
        orb_high: Opening range high (informational).
        orb_low: Opening range low (informational).
        vwap: Pre-computed VWAP (if None, computed from bars_1m).
        enable_*: Toggle individual filters on/off.
        enable_mtf_analyzer: Enable the full EMA+MACD+divergence MTF gate.
        allowed_windows: Custom session windows for session filter.
        nr7_lookback: NR7 lookback period (default 7).
        ema_period: EMA period for the simple multi-TF bias check (default 34).
        mtf_min_pass_score: Minimum MTF score for the full analyzer (default 0.55).
        gate_mode: "all" or "majority" — how to combine hard filter results.

    Returns:
        ORBFilterResult with composite pass/fail, all verdicts, and quality boost.
    """
    verdicts: list[FilterVerdict] = []

    # --- Soft filters (never reject, only boost) ---

    if enable_nr7:
        verdicts.append(check_nr7(bars_daily, lookback=nr7_lookback))

    # --- Hard filters (can reject) ---

    if enable_session_window:
        verdicts.append(check_session_window(signal_time, allowed_windows=allowed_windows))

    if enable_lunch_filter:
        verdicts.append(check_lunch_filter(signal_time))

    if enable_premarket:
        verdicts.append(
            check_premarket_range(
                direction=direction,
                trigger_price=trigger_price,
                premarket_high=premarket_high,
                premarket_low=premarket_low,
            )
        )

    if enable_multi_tf:
        verdicts.append(
            check_multi_tf_bias(
                bars_htf=bars_htf,
                direction=direction,
                ema_period=ema_period,
            )
        )

    if enable_vwap:
        verdicts.append(
            check_vwap_confluence(
                bars_1m=bars_1m,
                direction=direction,
                trigger_price=trigger_price,
                vwap=vwap,
            )
        )

    if enable_mtf_analyzer:
        verdicts.append(
            check_mtf_analyzer(
                bars_htf=bars_htf,
                direction=direction,
                min_pass_score=mtf_min_pass_score,
            )
        )

    # --- Compose result ---

    # Soft filters: NR7 is the only one — it always passes
    soft_names = {"NR7", "NR7 (Narrow Range 7)"}
    hard_verdicts = [v for v in verdicts if v.name not in soft_names]
    hard_passed = [v for v in hard_verdicts if v.passed]

    total_quality_boost = sum(v.score_boost for v in verdicts)

    if gate_mode == "majority":
        overall_passed = len(hard_passed) >= (len(hard_verdicts) / 2.0) if hard_verdicts else True
    else:
        # "all" mode: every hard filter must pass
        overall_passed = all(v.passed for v in hard_verdicts) if hard_verdicts else True

    filters_passed = sum(1 for v in verdicts if v.passed)
    filters_total = len(verdicts)

    # Build summary line
    failed_names = [v.name for v in verdicts if not v.passed]
    if overall_passed:
        summary = f"PASS ({filters_passed}/{filters_total} filters OK, boost {total_quality_boost:+.2f})"
    else:
        summary = f"REJECT — failed: {', '.join(failed_names)} ({filters_passed}/{filters_total} OK)"

    result = ORBFilterResult(
        passed=overall_passed,
        verdicts=verdicts,
        filters_passed=filters_passed,
        filters_total=filters_total,
        quality_boost=total_quality_boost,
        summary=summary,
    )

    logger.info("ORB filter result for %s: %s", direction, summary)
    return result


# ---------------------------------------------------------------------------
# Helpers for pre-market range extraction
# ---------------------------------------------------------------------------

# Globex pre-market window: 18:00 (prior evening) → 08:20 ET (before metals open)
# For simplicity we define the "morning" pre-market as 00:00–08:20 ET.
PM_START = dt_time(0, 0)
PM_END = dt_time(8, 20)


def extract_premarket_range(
    bars_1m: pd.DataFrame | None,
    pm_start: dt_time = PM_START,
    pm_end: dt_time = PM_END,
) -> tuple[float | None, float | None]:
    """Extract pre-market high and low from 1-minute bars.

    Scans bars between ``pm_start`` and ``pm_end`` (Eastern Time) and
    returns (high, low).  Returns (None, None) if no bars fall in the window.

    Args:
        bars_1m: 1-minute OHLCV DataFrame with DatetimeIndex.
        pm_start: Start of pre-market window (default 00:00 ET).
        pm_end: End of pre-market window (default 08:20 ET).

    Returns:
        (premarket_high, premarket_low) or (None, None).
    """
    if bars_1m is None or bars_1m.empty:
        return None, None

    df = bars_1m.copy()

    # Ensure Eastern timezone
    idx = df.index
    if not isinstance(idx, pd.DatetimeIndex):
        df.index = pd.to_datetime(idx)
    dti = pd.DatetimeIndex(df.index)
    if dti.tz is not None:
        if str(dti.tz) != str(_EST):
            df = df.tz_convert(_EST)
    else:
        with contextlib.suppress(Exception):
            df.index = dti.tz_localize(_EST)

    try:
        times = pd.DatetimeIndex(df.index).time  # type: ignore[attr-defined]
        pm_mask = (times >= pm_start) & (times < pm_end)
        pm_bars = df[pm_mask]
    except Exception:
        return None, None

    if pm_bars.empty:
        return None, None

    try:
        pm_high = float(pm_bars["High"].max())  # type: ignore[arg-type]
        pm_low = float(pm_bars["Low"].min())  # type: ignore[arg-type]
        return pm_high, pm_low
    except (KeyError, ValueError):
        return None, None
