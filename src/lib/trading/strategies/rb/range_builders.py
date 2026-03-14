"""
Range Builders — Per-Type Range Construction Functions
=======================================================

Each builder function takes 1-minute bars (and optionally ATR / config) and
returns a tuple describing the range for its breakout type.  Builders are
**pure functions** — no Redis, no side-effects, fully testable.

The unified ``detect_range_breakout()`` in ``detector.py`` dispatches to the
correct builder based on ``config.breakout_type``.

Public API::

    from lib.trading.strategies.rb.range_builders import (
        build_orb_range,
        build_pdr_range,
        build_ib_range,
        build_consolidation_range,
        build_weekly_range,
        build_monthly_range,
        build_asian_range,
        build_bbsqueeze_range,
        build_va_range,
        build_inside_day_range,
        build_gap_rejection_range,
        build_pivot_range,
        build_fibonacci_range,
    )

Design:
  - Each function mirrors the ``_build_*_range()`` from ``breakout.py`` but
    uses a cleaner signature with the **engine** ``RangeConfig`` (detection
    thresholds) as the config parameter.
  - Return types are kept as tuples for backward compatibility with the
    existing ``detect_range_breakout()`` dispatcher.  A future cleanup may
    replace these with typed NamedTuples.
  - ``localize_bars()`` and ``compute_atr()`` are shared helpers exposed at
    module level for reuse.
"""

from __future__ import annotations

import datetime as _dt
import logging
from datetime import time as dt_time
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

logger = logging.getLogger("strategies.rb.range_builders")

_ET = ZoneInfo("America/New_York")
_UTC = ZoneInfo("UTC")


# ===========================================================================
# Shared helpers
# ===========================================================================


def compute_atr(bars: pd.DataFrame, period: int = 14) -> float:
    """Wilder ATR on a DataFrame with High / Low / Close columns.

    Returns 0.0 if there is insufficient data.  This is the **single**
    canonical ATR implementation for the RB system — replaces the three
    copies that previously existed in ``orb.py``, ``breakout.py``, and
    ``rb_simulator.py``.
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


def localize_bars(bars: pd.DataFrame) -> pd.DataFrame:
    """Ensure the bar index is a tz-aware DatetimeIndex in US/Eastern.

    Works whether the index is UTC, naive, or already ET.  Returns the
    DataFrame unchanged if the index is not a DatetimeIndex.
    """
    if not isinstance(bars.index, pd.DatetimeIndex):
        return bars

    idx = bars.index
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    idx = idx.tz_convert(_ET)
    return bars.set_index(idx)


def check_bar_quality(
    bar_open: float,
    bar_high: float,
    bar_low: float,
    bar_close: float,
    level: float,
    direction: str,
    atr: float,
    min_depth_atr_pct: float,
    min_body_ratio: float,
) -> tuple[bool, bool, float, float]:
    """Check depth and body-ratio quality gates for a candidate breakout bar.

    Args:
        bar_open:  Open price of the breakout candidate bar.
        bar_high:  High price.
        bar_low:   Low price.
        bar_close: Close price.
        level:     The range boundary being tested (high for LONG, low for SHORT).
        direction: ``"LONG"`` or ``"SHORT"``.
        atr:       Current ATR value.
        min_depth_atr_pct: Minimum close-beyond-level depth as fraction of ATR.
        min_body_ratio:    Minimum body/range ratio for the breakout bar.

    Returns:
        ``(depth_ok, body_ok, depth_value, body_ratio)``
    """
    bar_range = bar_high - bar_low
    if bar_range <= 0:
        return False, False, 0.0, 0.0

    body = abs(bar_close - bar_open)
    body_ratio = body / bar_range

    depth = bar_close - level if direction == "LONG" else level - bar_close

    min_depth = min_depth_atr_pct * atr if atr > 0 else 0.0
    depth_ok = depth >= min_depth
    body_ok = body_ratio >= min_body_ratio

    return depth_ok, body_ok, max(depth, 0.0), body_ratio


# ===========================================================================
# Range builders — one per BreakoutType
# ===========================================================================


def build_orb_range(
    bars: pd.DataFrame,
    session_start: dt_time,
    session_end: dt_time,
    min_bars: int = 5,
) -> tuple[float, float, int, bool]:
    """Extract the opening-range high/low from bars within the OR window.

    .. note:: ``complete`` semantics vs ``compute_opening_range``

        This function sets ``complete = last_bar_time >= session_end``,
        i.e. purely based on the wall-clock time of the most recent bar.
        It returns ``True`` as soon as the dataset's newest bar sits at or
        after ``session_end``, even if no bar *after* ``session_end`` is
        explicitly present.

        :func:`~lib.trading.strategies.rb.open.detector.compute_opening_range`
        uses a stricter check: ``complete`` is only ``True`` when at least
        one bar *past* ``session_end`` is present in the data.  This is
        safer for live feeds where the last bar may land exactly on the
        boundary.

        Use ``build_orb_range`` for back-testing pipelines where the bar
        window is already fully formed.  Prefer ``compute_opening_range``
        inside the live ORB detector.

    Args:
        bars:          1-minute OHLCV bars.
        session_start: OR window start (ET wall-clock).
        session_end:   OR window end (ET wall-clock).
        min_bars:      Minimum bars required in OR window.

    Returns:
        ``(or_high, or_low, bar_count, complete)``
        ``complete`` is ``True`` once the last bar's wall-clock time is
        at or past ``session_end`` (see note above).
    """
    bars_et = localize_bars(bars)
    now_et = pd.Timestamp(bars_et.index[-1]).to_pydatetime().time() if len(bars_et) > 0 else dt_time(0, 0)  # type: ignore[arg-type]

    idx_time = pd.DatetimeIndex(bars_et.index).time  # type: ignore[attr-defined]
    mask = (idx_time >= session_start) & (idx_time < session_end)
    or_bars = bars_et.loc[mask]

    if len(or_bars) < min_bars:
        return 0.0, 0.0, len(or_bars), False

    or_high = float(or_bars["High"].max())
    or_low = float(or_bars["Low"].min())
    complete = now_et >= session_end

    return or_high, or_low, len(or_bars), complete


def build_pdr_range(
    bars: pd.DataFrame,
    *,
    pdr_session_start: dt_time = dt_time(18, 0),
    min_bars: int = 1,
    prev_day_high: float | None = None,
    prev_day_low: float | None = None,
) -> tuple[float, float, int, bool, float, float, float]:
    """Identify the previous Globex day's high/low from intraday 1m bars.

    If ``prev_day_high`` / ``prev_day_low`` are explicitly provided (e.g.
    from a pre-computed daily bar cache), they are used directly.

    Strategy: split bars at ``pdr_session_start`` (18:00 ET by default).
    Everything before today's session start is "previous day".

    Returns:
        ``(pdr_high, pdr_low, bar_count, complete, prev_high, prev_low, prev_range)``
        ``complete`` is always True for PDR (the range is already formed).
    """
    # Use explicit overrides if provided
    if prev_day_high is not None and prev_day_low is not None and prev_day_high > prev_day_low > 0:
        prev_range = prev_day_high - prev_day_low
        return prev_day_high, prev_day_low, 0, True, prev_day_high, prev_day_low, prev_range

    bars_et = localize_bars(bars)
    if len(bars_et) < 2:
        return 0.0, 0.0, 0, False, 0.0, 0.0, 0.0

    # Find the latest session-start boundary
    idx_time_pdr = pd.DatetimeIndex(bars_et.index).time  # type: ignore[attr-defined]
    today_session_starts = bars_et.index[idx_time_pdr == pdr_session_start]

    if len(today_session_starts) == 0:
        # Fallback: use calendar midnight boundary
        today_et = pd.Timestamp(bars_et.index[-1]).to_pydatetime().date()  # type: ignore[arg-type]
        cutoff = pd.Timestamp(today_et, tz=_ET)
        prev_bars = bars_et[bars_et.index < cutoff]
    else:
        latest_start = today_session_starts[-1]
        prev_bars = bars_et[bars_et.index < latest_start]

    if len(prev_bars) < min_bars:
        return 0.0, 0.0, len(prev_bars), False, 0.0, 0.0, 0.0

    prev_high = float(prev_bars["High"].max())  # type: ignore[arg-type]
    prev_low = float(prev_bars["Low"].min())  # type: ignore[arg-type]
    prev_range = prev_high - prev_low
    bar_count = len(prev_bars)

    return prev_high, prev_low, bar_count, True, prev_high, prev_low, prev_range


def build_ib_range(
    bars: pd.DataFrame,
    *,
    ib_start_time: dt_time = dt_time(9, 30),
    ib_duration_minutes: int = 60,
    min_bars: int = 10,
    ib_high_override: float | None = None,
    ib_low_override: float | None = None,
) -> tuple[float, float, int, bool]:
    """Build the Initial Balance range (first N minutes of RTH).

    If ``ib_high_override`` / ``ib_low_override`` are provided, they are
    used directly (e.g. from a pre-computed IB).

    Returns:
        ``(ib_high, ib_low, bar_count, complete)``
        ``complete`` is True once the current time is past the IB end time.
    """
    if ib_high_override is not None and ib_low_override is not None and ib_high_override > ib_low_override > 0:
        return ib_high_override, ib_low_override, 0, True

    bars_et = localize_bars(bars)
    if len(bars_et) < 1:
        return 0.0, 0.0, 0, False

    ib_end_minutes = ib_start_time.hour * 60 + ib_start_time.minute + ib_duration_minutes
    ib_end_h, ib_end_m = divmod(ib_end_minutes, 60)
    ib_end = dt_time(int(ib_end_h), int(ib_end_m))

    idx_time_ib = pd.DatetimeIndex(bars_et.index).time  # type: ignore[attr-defined]
    mask = (idx_time_ib >= ib_start_time) & (idx_time_ib < ib_end)
    ib_bars = bars_et.loc[mask]

    bar_count = len(ib_bars)
    now_et = pd.Timestamp(bars_et.index[-1]).to_pydatetime().time()  # type: ignore[arg-type]
    complete = now_et >= ib_end

    if bar_count < min_bars:
        return 0.0, 0.0, bar_count, complete

    ib_high = float(ib_bars["High"].max())
    ib_low = float(ib_bars["Low"].min())

    return ib_high, ib_low, bar_count, complete


def build_consolidation_range(
    bars: pd.DataFrame,
    atr: float,
    *,
    squeeze_bb_period: int = 20,
    squeeze_bb_std: float = 2.0,
    squeeze_atr_mult: float = 1.5,
    squeeze_min_bars: int = 5,
) -> tuple[float, float, int, bool, float, float, float, int, float]:
    """Detect a Bollinger Band / ATR consolidation squeeze and extract its range.

    A "squeeze" is present when the BB bandwidth (upper - lower) is smaller
    than ``squeeze_atr_mult * ATR`` for at least ``squeeze_min_bars``
    consecutive bars.

    Returns:
        ``(cons_high, cons_low, bar_count, squeeze_detected,
          bb_upper, bb_lower, bb_width, squeeze_bar_count, current_bb_width)``
    """
    if len(bars) < squeeze_bb_period + 2 or atr <= 0:
        return 0.0, 0.0, 0, False, 0.0, 0.0, 0.0, 0, 0.0

    close = bars["Close"].astype(float)
    n = squeeze_bb_period
    std_mult = squeeze_bb_std

    # Rolling Bollinger Bands
    bb_mid = pd.Series(close.rolling(n).mean())
    bb_std_s = pd.Series(close.rolling(n).std(ddof=0))
    bb_upper = pd.Series(bb_mid + std_mult * bb_std_s)
    bb_lower = pd.Series(bb_mid - std_mult * bb_std_s)
    bb_width = pd.Series((bb_upper - bb_lower).fillna(0.0))

    threshold = squeeze_atr_mult * atr

    # Count consecutive squeeze bars at the end of the series
    squeeze_flags = pd.Series(bb_width < threshold)
    squeeze_bar_count = 0
    for i in range(len(squeeze_flags) - 1, -1, -1):
        if squeeze_flags.iloc[i]:
            squeeze_bar_count += 1
        else:
            break

    squeeze_detected = squeeze_bar_count >= squeeze_min_bars
    current_bb_width = float(bb_width.iloc[-1]) if len(bb_width) > 0 else 0.0
    current_bb_upper = float(bb_upper.iloc[-1]) if len(bb_upper) > 0 else 0.0
    current_bb_lower = float(bb_lower.iloc[-1]) if len(bb_lower) > 0 else 0.0

    if not squeeze_detected:
        return (
            0.0,
            0.0,
            0,
            False,
            current_bb_upper,
            current_bb_lower,
            current_bb_width,
            0,
            current_bb_width,
        )

    # The consolidation range is the H/L over the squeeze window
    squeeze_start_idx = len(bars) - squeeze_bar_count
    squeeze_slice = bars.iloc[squeeze_start_idx:]
    cons_high = float(squeeze_slice["High"].max())
    cons_low = float(squeeze_slice["Low"].min())
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


def build_weekly_range(
    bars: pd.DataFrame,
    *,
    weekly_lookback_days: int = 5,
    min_bars: int = 1,
) -> tuple[float, float, int, bool]:
    """Extract the prior week's high/low from intraday bars.

    Uses ``weekly_lookback_days`` (default 5) to identify bars belonging to
    the prior trading week.

    Returns:
        ``(week_high, week_low, bar_count, complete)``
    """
    bars_et = localize_bars(bars)
    if len(bars_et) < 2:
        return 0.0, 0.0, 0, False

    today = pd.Timestamp(bars_et.index[-1]).to_pydatetime().date()  # type: ignore[arg-type]
    # Walk back to find the start of this week (Monday)
    weekday = today.weekday()  # Monday=0 .. Sunday=6
    this_week_start = today - _dt.timedelta(days=int(weekday))

    cutoff = pd.Timestamp(this_week_start, tz=_ET)  # type: ignore[arg-type]
    prev_week_end = cutoff
    prev_week_start = cutoff - pd.Timedelta(days=weekly_lookback_days)

    mask = (bars_et.index >= prev_week_start) & (bars_et.index < prev_week_end)
    prev_bars = bars_et.loc[mask]

    if len(prev_bars) < min_bars:
        return 0.0, 0.0, len(prev_bars), False

    week_high = float(prev_bars["High"].max())
    week_low = float(prev_bars["Low"].min())
    return week_high, week_low, len(prev_bars), True


def build_monthly_range(
    bars: pd.DataFrame,
    *,
    monthly_lookback_days: int = 20,
    min_bars: int = 1,
) -> tuple[float, float, int, bool]:
    """Extract the prior month's high/low from intraday bars.

    Uses ``monthly_lookback_days`` (default 20) calendar trading days prior
    to the 1st of the current month.

    Returns:
        ``(month_high, month_low, bar_count, complete)``
    """
    bars_et = localize_bars(bars)
    if len(bars_et) < 2:
        return 0.0, 0.0, 0, False

    today = pd.Timestamp(bars_et.index[-1]).to_pydatetime().date()  # type: ignore[arg-type]
    first_of_month = today.replace(day=1)
    cutoff = pd.Timestamp(first_of_month, tz=_ET)  # type: ignore[arg-type]
    lookback_start = cutoff - pd.Timedelta(days=monthly_lookback_days)

    mask = (bars_et.index >= lookback_start) & (bars_et.index < cutoff)
    prev_bars = bars_et.loc[mask]

    if len(prev_bars) < min_bars:
        return 0.0, 0.0, len(prev_bars), False

    month_high = float(prev_bars["High"].max())
    month_low = float(prev_bars["Low"].min())
    return month_high, month_low, len(prev_bars), True


def build_asian_range(
    bars: pd.DataFrame,
    *,
    asian_start_time: dt_time = dt_time(19, 0),
    asian_end_time: dt_time = dt_time(2, 0),
    min_bars: int = 10,
) -> tuple[float, float, int, bool]:
    """Extract the Asian session range (wraps midnight: 19:00-02:00 ET).

    Returns:
        ``(asian_high, asian_low, bar_count, complete)``
        ``complete`` is True once the current time is past the end time.
    """
    bars_et = localize_bars(bars)
    if len(bars_et) < 1:
        return 0.0, 0.0, 0, False

    start = asian_start_time
    end = asian_end_time
    now_time = pd.Timestamp(bars_et.index[-1]).to_pydatetime().time()  # type: ignore[arg-type]

    idx_time = pd.DatetimeIndex(bars_et.index).time  # type: ignore[attr-defined]

    # Wraps midnight: time >= 19:00 OR time < 02:00
    mask = (idx_time >= start) | (idx_time < end) if start > end else (idx_time >= start) & (idx_time < end)

    asian_bars = bars_et.loc[mask]
    bar_count = len(asian_bars)

    # Complete once we're past the end time and NOT in the start window
    complete = end <= now_time < start if start > end else now_time >= end

    if bar_count < min_bars:
        return 0.0, 0.0, bar_count, complete

    asian_high = float(asian_bars["High"].max())
    asian_low = float(asian_bars["Low"].min())
    return asian_high, asian_low, bar_count, complete


def build_bbsqueeze_range(
    bars: pd.DataFrame,
    atr: float,
    *,
    bb_period: int = 20,
    bb_std: float = 2.0,
    kc_period: int = 20,
    kc_atr_mult: float = 1.5,
    min_squeeze_bars: int = 6,
) -> tuple[float, float, int, bool, int, float, float, float]:
    """Detect a Bollinger Band inside Keltner Channel squeeze.

    A "squeeze" exists when the Bollinger Bands are fully inside the
    Keltner Channel for at least ``min_squeeze_bars`` consecutive bars.

    Returns:
        ``(range_high, range_low, bar_count, squeeze_detected,
          squeeze_bar_count, bb_width, bb_upper, bb_lower)``
    """
    if len(bars) < bb_period + 2 or atr <= 0:
        return 0.0, 0.0, 0, False, 0, 0.0, 0.0, 0.0

    close = bars["Close"].astype(float)
    high = bars["High"].astype(float)
    low = bars["Low"].astype(float)

    # Bollinger Bands
    bb_mid = close.rolling(bb_period).mean()
    bb_std_series = close.rolling(bb_period).std(ddof=0)
    bb_upper = bb_mid + bb_std * bb_std_series
    bb_lower = bb_mid - bb_std * bb_std_series

    # Keltner Channel (EMA-based mid + ATR envelope)
    kc_mid = close.ewm(span=kc_period, adjust=False).mean()

    # Per-bar ATR for KC (rolling)
    tr_series = pd.concat(
        [
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    rolling_atr = tr_series.rolling(kc_period).mean()

    kc_upper = pd.Series(kc_mid + kc_atr_mult * rolling_atr)
    kc_lower = pd.Series(kc_mid - kc_atr_mult * rolling_atr)

    # Squeeze: BB fully inside KC
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

    squeeze_detected = squeeze_bar_count >= min_squeeze_bars

    cur_bb_upper = float(pd.Series(bb_upper).iloc[-1]) if len(bb_upper) > 0 else 0.0  # type: ignore[arg-type]
    cur_bb_lower = float(pd.Series(bb_lower).iloc[-1]) if len(bb_lower) > 0 else 0.0  # type: ignore[arg-type]
    cur_bb_width = cur_bb_upper - cur_bb_lower

    if not squeeze_detected:
        return 0.0, 0.0, 0, False, 0, cur_bb_width, cur_bb_upper, cur_bb_lower

    squeeze_start_idx = len(bars) - squeeze_bar_count
    squeeze_slice = bars.iloc[squeeze_start_idx:]
    r_high = float(squeeze_slice["High"].max())
    r_low = float(squeeze_slice["Low"].min())

    return (
        r_high,
        r_low,
        len(squeeze_slice),
        True,
        squeeze_bar_count,
        cur_bb_width,
        cur_bb_upper,
        cur_bb_lower,
    )


def build_va_range(
    bars: pd.DataFrame,
    *,
    value_area_pct: float = 0.70,
    n_bins: int = 50,
    min_bars: int = 1,
) -> tuple[float, float, int, bool, float, float, float]:
    """Build Value Area range (VAH/VAL) from prior session's volume profile.

    Uses the Globex-day split (18:00 ET boundary) to isolate the prior
    session's bars and compute volume profile on them.

    Returns:
        ``(vah, val, bar_count, complete, poc, vah, val)``
    """
    bars_et = localize_bars(bars)
    if len(bars_et) < 10:
        return 0.0, 0.0, 0, False, 0.0, 0.0, 0.0

    # Split at 18:00 ET to get prior session bars
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

    if len(prev_bars) < min_bars:
        return 0.0, 0.0, len(prev_bars), False, 0.0, 0.0, 0.0

    # Check that volume data exists
    if "Volume" not in prev_bars.columns or prev_bars["Volume"].sum() <= 0:
        # Fall back to simple H/L percentile if no volume data
        logger.debug("build_va_range: no volume data — falling back to price-based VA")
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
            n_bins=n_bins,
            value_area_pct=value_area_pct,
        )
        vah = float(profile["vah"])
        val = float(profile["val"])
        poc = float(profile["poc"])
    except Exception as exc:
        logger.warning("build_va_range: volume profile computation failed: %s", exc)
        return 0.0, 0.0, len(prev_bars), False, 0.0, 0.0, 0.0

    if vah <= val or vah <= 0:
        return 0.0, 0.0, len(prev_bars), False, 0.0, 0.0, 0.0

    return vah, val, len(prev_bars), True, poc, vah, val


def build_inside_day_range(
    bars: pd.DataFrame,
    *,
    min_compression: float = 0.30,
    max_compression: float = 0.85,
    min_bars: int = 1,
) -> tuple[float, float, int, bool, float, float, float, float]:
    """Detect an inside day and return the mother bar's range.

    An "inside day" occurs when today's high/low are both contained within
    yesterday's high/low.  The breakout range is yesterday's H/L (the
    "mother bar").

    Returns:
        ``(mother_high, mother_low, bar_count, inside_detected,
          today_high, today_low, yesterday_high, yesterday_low)``
    """
    bars_et = localize_bars(bars)
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

    if len(today_bars) < 1 or len(yesterday_bars) < min_bars:
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
    if compression < min_compression or compression > max_compression:
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


def build_gap_rejection_range(
    bars: pd.DataFrame,
    atr: float,
    *,
    gap_min_atr_pct: float = 0.15,
    gap_fill_threshold_pct: float = 0.50,
    gap_rejection_bars: int = 3,
) -> tuple[float, float, int, bool, float, float, str]:
    """Detect an overnight gap and build the gap zone as the breakout range.

    A gap exists when today's open is meaningfully separated from yesterday's
    close.  The breakout range is defined by the gap boundaries:
      - Gap up:   range = [yesterday_close, today_open]
      - Gap down: range = [today_open, yesterday_close]

    Returns:
        ``(range_high, range_low, bar_count, gap_detected,
          gap_size, yesterday_close, gap_direction)``
        ``gap_direction`` is ``"UP"`` or ``"DOWN"``.
    """
    bars_et = localize_bars(bars)
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
    min_gap = gap_min_atr_pct * atr

    if abs(gap_size) < min_gap:
        return 0.0, 0.0, 0, False, gap_size, yesterday_close, ""

    if gap_size > 0:
        r_high = today_open
        r_low = yesterday_close
        direction = "UP"
    else:
        r_high = yesterday_close
        r_low = today_open
        direction = "DOWN"

    return r_high, r_low, len(today_bars), True, gap_size, yesterday_close, direction


def build_pivot_range(
    bars: pd.DataFrame,
    *,
    pivot_formula: str = "classic",
    min_bars: int = 1,
) -> tuple[float, float, int, bool, float, float, float, float]:
    """Compute floor pivots from prior session's HLC and return S1/R1 as the range.

    Supported formulas:
      - ``classic``: P = (H+L+C)/3, R1 = 2P-L, S1 = 2P-H
      - ``woodie``:  P = (H+L+2C)/4, R1 = 2P-L, S1 = 2P-H
      - ``camarilla``: P = (H+L+C)/3, R1 = C+1.1*(H-L)/12, S1 = C-1.1*(H-L)/12

    Returns:
        ``(r1, s1, bar_count, complete, pivot, r1, s1, prev_close)``
    """
    bars_et = localize_bars(bars)
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

    if len(prev_bars) < min_bars:
        return 0.0, 0.0, len(prev_bars), False, 0.0, 0.0, 0.0, 0.0

    h = float(prev_bars["High"].max())  # type: ignore[arg-type]
    prev_low = float(prev_bars["Low"].min())  # type: ignore[arg-type]
    c = float(prev_bars["Close"].iloc[-1])  # type: ignore[arg-type]

    formula = pivot_formula.lower()
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


def build_fibonacci_range(
    bars: pd.DataFrame,
    atr: float,
    *,
    fib_upper: float = 0.618,
    fib_lower: float = 0.382,
    fib_swing_lookback: int = 100,
    fib_min_swing_atr_mult: float = 1.5,
) -> tuple[float, float, int, bool, float, float, float, float]:
    """Find the prior swing high/low and compute the Fibonacci retracement zone.

    Identifies the highest high and lowest low within the last
    ``fib_swing_lookback`` bars.  The swing must be at least
    ``fib_min_swing_atr_mult * ATR`` in size.  The retracement zone
    between ``fib_lower`` (38.2%) and ``fib_upper`` (61.8%) of that swing
    becomes the breakout range.

    Returns:
        ``(fib_high, fib_low, bar_count, valid,
          swing_high, swing_low, fib_382, fib_618)``
    """
    if len(bars) < 10 or atr <= 0:
        return 0.0, 0.0, 0, False, 0.0, 0.0, 0.0, 0.0

    # Use the last N bars for swing detection
    lookback = fib_swing_lookback
    window = bars.iloc[-lookback:] if len(bars) > lookback else bars

    swing_high = float(window["High"].max())  # type: ignore[arg-type]
    swing_low = float(window["Low"].min())  # type: ignore[arg-type]
    swing_size = swing_high - swing_low

    if swing_size < fib_min_swing_atr_mult * atr:
        return 0.0, 0.0, 0, False, swing_high, swing_low, 0.0, 0.0

    # Determine swing direction: is the high more recent than the low?
    high_idx = window["High"].astype(float).values.argmax()
    low_idx = window["Low"].astype(float).values.argmin()

    if high_idx > low_idx:
        # Upswing — retracement is measured downward from swing high
        fib_382 = swing_high - fib_lower * swing_size
        fib_618 = swing_high - fib_upper * swing_size
        r_high = fib_382
        r_low = fib_618
    else:
        # Downswing — retracement is measured upward from swing low
        fib_382 = swing_low + fib_lower * swing_size
        fib_618 = swing_low + fib_upper * swing_size
        r_high = fib_618
        r_low = fib_382

    if r_high <= r_low or r_high <= 0:
        return 0.0, 0.0, 0, False, swing_high, swing_low, fib_382, fib_618

    return r_high, r_low, len(window), True, swing_high, swing_low, fib_382, fib_618


# ===========================================================================
# Dispatch helper
# ===========================================================================

# Import BreakoutType here (module level) so the dispatch dict works.
# Guarded import to avoid circular dependency issues at load time.
_BUILDER_DISPATCH: dict | None = None


def _get_builder_dispatch() -> dict:
    """Lazily build and cache the builder dispatch dict.

    This avoids circular imports at module load time while still giving
    O(1) dispatch by BreakoutType.
    """
    global _BUILDER_DISPATCH
    if _BUILDER_DISPATCH is not None:
        return _BUILDER_DISPATCH

    from lib.core.breakout_types import BreakoutType

    _BUILDER_DISPATCH = {
        BreakoutType.ORB: build_orb_range,
        BreakoutType.PrevDay: build_pdr_range,
        BreakoutType.InitialBalance: build_ib_range,
        BreakoutType.Consolidation: build_consolidation_range,
        BreakoutType.Weekly: build_weekly_range,
        BreakoutType.Monthly: build_monthly_range,
        BreakoutType.Asian: build_asian_range,
        BreakoutType.BollingerSqueeze: build_bbsqueeze_range,
        BreakoutType.ValueArea: build_va_range,
        BreakoutType.InsideDay: build_inside_day_range,
        BreakoutType.GapRejection: build_gap_rejection_range,
        BreakoutType.PivotPoints: build_pivot_range,
        BreakoutType.Fibonacci: build_fibonacci_range,
    }
    return _BUILDER_DISPATCH


def get_range_builder(breakout_type: Any) -> Any:
    """Return the builder function for a given ``BreakoutType``.

    Args:
        breakout_type: A ``BreakoutType`` enum member.

    Returns:
        The builder callable.

    Raises:
        KeyError: If no builder is registered for the type.
    """
    dispatch = _get_builder_dispatch()
    return dispatch[breakout_type]


__all__ = [
    # Shared helpers
    "compute_atr",
    "localize_bars",
    "check_bar_quality",
    # Builders
    "build_orb_range",
    "build_pdr_range",
    "build_ib_range",
    "build_consolidation_range",
    "build_weekly_range",
    "build_monthly_range",
    "build_asian_range",
    "build_bbsqueeze_range",
    "build_va_range",
    "build_inside_day_range",
    "build_gap_rejection_range",
    "build_pivot_range",
    "build_fibonacci_range",
    # Dispatch
    "get_range_builder",
]
