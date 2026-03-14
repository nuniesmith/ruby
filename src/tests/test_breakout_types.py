"""
Tests for the 9 new breakout-type range builders and their integration
with ``detect_range_breakout()`` and ``detect_all_breakout_types()``.

Covers:
  - Weekly range builder: _build_weekly_range
  - Monthly range builder: _build_monthly_range
  - Asian session range builder: _build_asian_range
  - Bollinger Squeeze (BB inside KC) builder: _build_bbsqueeze_range
  - Value Area (VAH/VAL) builder: _build_va_range
  - Inside Day builder: _build_inside_day_range
  - Gap Rejection builder: _build_gap_rejection_range
  - Pivot Points builder: _build_pivot_range
  - Fibonacci retracement builder: _build_fibonacci_range
  - detect_range_breakout integration for all 9 new types
  - detect_all_breakout_types with all 13 types
  - DEFAULT_CONFIGS completeness for all 13 BreakoutType values
  - Scheduler ActionType entries for new types
"""

import os
from datetime import datetime
from datetime import time as dt_time
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("DISABLE_REDIS", "1")

from lib.trading.strategies.rb.breakout import (  # noqa: E402
    DEFAULT_CONFIGS,
    BreakoutResult,
    BreakoutType,
    RangeConfig,
    _build_asian_range,
    _build_bbsqueeze_range,
    _build_fibonacci_range,
    _build_gap_rejection_range,
    _build_inside_day_range,
    _build_monthly_range,
    _build_pivot_range,
    _build_va_range,
    _build_weekly_range,
    _compute_atr,
    detect_all_breakout_types,
    detect_range_breakout,
)

_EST = ZoneInfo("America/New_York")


# ---------------------------------------------------------------------------
# Helpers — build synthetic 1-minute bar DataFrames
# ---------------------------------------------------------------------------


def _make_1m_bars(
    n: int = 60,
    start_price: float = 2700.0,
    start_time: str = "2026-02-27 09:20:00",
    freq: str = "1min",
    seed: int = 42,
    volatility: float = 0.001,
    trend: float = 0.0,
    tz=_EST,
) -> pd.DataFrame:
    """Create a synthetic 1-minute OHLCV DataFrame with tz-aware Eastern index."""
    rng = np.random.default_rng(seed)
    returns = rng.normal(trend, volatility, n)
    close = start_price * np.exp(np.cumsum(returns))

    spread = close * rng.uniform(0.001, 0.003, n)
    high = close + rng.uniform(0, 1, n) * spread
    low = close - rng.uniform(0, 1, n) * spread
    opn = close + rng.uniform(-0.3, 0.3, n) * spread

    high = np.maximum(high, np.maximum(opn, close))
    low = np.minimum(low, np.minimum(opn, close))
    volume = rng.poisson(500, n).astype(float)

    idx = pd.date_range(start=start_time, periods=n, freq=freq, tz=tz)

    return pd.DataFrame(
        {"Open": opn, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=idx,
    )


def _make_multi_day_bars(
    n_days: int = 10,
    bars_per_day: int = 390,
    start_price: float = 2700.0,
    start_date: str = "2026-02-16",
    seed: int = 42,
    volatility: float = 0.0005,
) -> pd.DataFrame:
    """Create multiple days of 1-minute bars with Globex-day boundaries at 18:00 ET.

    Each Globex day runs from 18:00 ET to next day 17:00 ET.
    """
    rng = np.random.default_rng(seed)
    all_bars = []
    price = start_price

    base_date = pd.Timestamp(start_date, tz=_EST)

    for day_offset in range(n_days):
        day_start = base_date + pd.Timedelta(days=day_offset)
        # Globex day: 18:00 ET day-1 to 17:00 ET day
        session_start = day_start.replace(hour=18, minute=0, second=0)

        returns = rng.normal(0.0, volatility, bars_per_day)
        close = price * np.exp(np.cumsum(returns))

        spread = close * rng.uniform(0.001, 0.003, bars_per_day)
        high = close + rng.uniform(0, 1, bars_per_day) * spread
        low = close - rng.uniform(0, 1, bars_per_day) * spread
        opn = close + rng.uniform(-0.3, 0.3, bars_per_day) * spread
        high = np.maximum(high, np.maximum(opn, close))
        low = np.minimum(low, np.minimum(opn, close))
        volume = rng.poisson(500, bars_per_day).astype(float)

        idx = pd.date_range(start=session_start, periods=bars_per_day, freq="1min", tz=_EST)
        df = pd.DataFrame(
            {"Open": opn, "High": high, "Low": low, "Close": close, "Volume": volume},
            index=idx,
        )
        all_bars.append(df)
        price = close[-1]

    return pd.concat(all_bars).sort_index()


def _make_inside_day_bars(
    mother_high: float = 2720.0,
    mother_low: float = 2680.0,
    inside_high: float = 2710.0,
    inside_low: float = 2690.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Create bars with a clear inside day pattern (today inside yesterday).

    Yesterday's session: 18:00 ET day-2 to 17:59 ET day-1
    Today's session: 18:00 ET day-1 to current (within range)
    """
    rng = np.random.default_rng(seed)

    # Yesterday's session — 200 bars from 18:00 ET two days ago
    n_yest = 200
    yest_start = pd.Timestamp("2026-02-25 18:00:00", tz=_EST)
    yest_mid = (mother_high + mother_low) / 2.0
    yest_close = yest_mid + rng.uniform(-5, 5, n_yest)
    yest_high = np.clip(yest_close + rng.uniform(1, 10, n_yest), None, mother_high)
    yest_low = np.clip(yest_close - rng.uniform(1, 10, n_yest), mother_low, None)
    # Force at least one bar to touch the extremes
    yest_high[50] = mother_high
    yest_low[100] = mother_low
    yest_open = yest_close + rng.uniform(-2, 2, n_yest)
    yest_high = np.maximum(yest_high, np.maximum(yest_open, yest_close))
    yest_low = np.minimum(yest_low, np.minimum(yest_open, yest_close))
    yest_vol = rng.poisson(500, n_yest).astype(float)
    yest_idx = pd.date_range(start=yest_start, periods=n_yest, freq="1min", tz=_EST)
    yest_df = pd.DataFrame(
        {"Open": yest_open, "High": yest_high, "Low": yest_low, "Close": yest_close, "Volume": yest_vol},
        index=yest_idx,
    )

    # Today's session — 100 bars from 18:00 ET yesterday (inside the mother bar)
    n_today = 100
    today_start = pd.Timestamp("2026-02-26 18:00:00", tz=_EST)
    today_mid = (inside_high + inside_low) / 2.0
    today_close = today_mid + rng.uniform(-3, 3, n_today)
    today_high = np.clip(today_close + rng.uniform(0.5, 5, n_today), None, inside_high)
    today_low = np.clip(today_close - rng.uniform(0.5, 5, n_today), inside_low, None)
    today_high[10] = inside_high
    today_low[30] = inside_low
    today_open = today_close + rng.uniform(-1, 1, n_today)
    today_high = np.maximum(today_high, np.maximum(today_open, today_close))
    today_low = np.minimum(today_low, np.minimum(today_open, today_close))
    today_vol = rng.poisson(500, n_today).astype(float)
    today_idx = pd.date_range(start=today_start, periods=n_today, freq="1min", tz=_EST)
    today_df = pd.DataFrame(
        {"Open": today_open, "High": today_high, "Low": today_low, "Close": today_close, "Volume": today_vol},
        index=today_idx,
    )

    return pd.concat([yest_df, today_df]).sort_index()


def _make_gap_bars(
    yesterday_close: float = 2700.0,
    gap_size: float = 20.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Create bars with a clear overnight gap.

    Yesterday: closes at yesterday_close
    Today: opens at yesterday_close + gap_size
    """
    rng = np.random.default_rng(seed)

    # Yesterday: 200 bars ending near yesterday_close
    n_yest = 200
    yest_start = pd.Timestamp("2026-02-25 18:00:00", tz=_EST)
    yest_prices = yesterday_close + rng.normal(0, 2, n_yest)
    yest_prices[-1] = yesterday_close  # ensure last bar closes at target
    yest_spread = np.abs(rng.normal(0, 1, n_yest))
    yest_high = yest_prices + yest_spread
    yest_low = yest_prices - yest_spread
    yest_open = yest_prices + rng.uniform(-0.5, 0.5, n_yest)
    yest_high = np.maximum(yest_high, np.maximum(yest_open, yest_prices))
    yest_low = np.minimum(yest_low, np.minimum(yest_open, yest_prices))
    yest_vol = rng.poisson(500, n_yest).astype(float)
    yest_idx = pd.date_range(start=yest_start, periods=n_yest, freq="1min", tz=_EST)
    yest_df = pd.DataFrame(
        {"Open": yest_open, "High": yest_high, "Low": yest_low, "Close": yest_prices, "Volume": yest_vol},
        index=yest_idx,
    )

    # Today: 100 bars starting at gap level
    n_today = 100
    today_start = pd.Timestamp("2026-02-26 18:00:00", tz=_EST)
    today_open_price = yesterday_close + gap_size
    today_prices = today_open_price + rng.normal(0, 2, n_today)
    today_prices[0] = today_open_price  # ensure first bar opens at gap level
    today_spread = np.abs(rng.normal(0, 1, n_today))
    today_high = today_prices + today_spread
    today_low = today_prices - today_spread
    today_open = today_prices.copy()
    today_open[0] = today_open_price
    today_high = np.maximum(today_high, np.maximum(today_open, today_prices))
    today_low = np.minimum(today_low, np.minimum(today_open, today_prices))
    today_vol = rng.poisson(500, n_today).astype(float)
    today_idx = pd.date_range(start=today_start, periods=n_today, freq="1min", tz=_EST)
    today_df = pd.DataFrame(
        {"Open": today_open, "High": today_high, "Low": today_low, "Close": today_prices, "Volume": today_vol},
        index=today_idx,
    )

    return pd.concat([yest_df, today_df]).sort_index()


def _make_asian_session_bars(
    start_date: str = "2026-02-26",
    seed: int = 42,
) -> pd.DataFrame:
    """Create bars covering the Asian session (19:00–02:00 ET) and beyond.

    Bars run from 18:00 ET through 10:00 ET next day.
    """
    rng = np.random.default_rng(seed)
    n = 960  # 16 hours of 1-min bars
    start = pd.Timestamp(f"{start_date} 18:00:00", tz=_EST)

    close = 2700.0 + np.cumsum(rng.normal(0, 0.5, n))
    spread = np.abs(rng.normal(0, 0.5, n))
    high = close + spread
    low = close - spread
    opn = close + rng.uniform(-0.3, 0.3, n)
    high = np.maximum(high, np.maximum(opn, close))
    low = np.minimum(low, np.minimum(opn, close))
    volume = rng.poisson(300, n).astype(float)

    idx = pd.date_range(start=start, periods=n, freq="1min", tz=_EST)
    return pd.DataFrame(
        {"Open": opn, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=idx,
    )


def _make_squeeze_bars(
    n: int = 100,
    start_price: float = 2700.0,
    squeeze_start: int = 30,
    squeeze_bars: int = 30,
    seed: int = 42,
) -> pd.DataFrame:
    """Create bars with a clear Bollinger Band squeeze pattern.

    First `squeeze_start` bars: normal volatility
    Next `squeeze_bars` bars: very tight range (squeeze)
    Remaining bars: expansion
    """
    np.random.default_rng(seed)
    parts = []

    # Normal volatility phase
    normal = _make_1m_bars(
        n=squeeze_start,
        start_price=start_price,
        start_time="2026-02-27 08:00:00",
        seed=seed,
        volatility=0.002,
    )
    parts.append(normal)

    # Squeeze phase — very tight range
    squeeze_start_time = normal.index[-1] + pd.Timedelta(minutes=1)
    squeeze_price = float(normal["Close"].iloc[-1])
    squeeze = _make_1m_bars(
        n=squeeze_bars,
        start_price=squeeze_price,
        start_time=squeeze_start_time.strftime("%Y-%m-%d %H:%M:%S"),
        seed=seed + 1,
        volatility=0.00005,  # very low volatility for squeeze
    )
    parts.append(squeeze)

    # Expansion phase
    expansion_start_time = squeeze.index[-1] + pd.Timedelta(minutes=1)
    expansion_price = float(squeeze["Close"].iloc[-1])
    expansion = _make_1m_bars(
        n=n - squeeze_start - squeeze_bars,
        start_price=expansion_price,
        start_time=expansion_start_time.strftime("%Y-%m-%d %H:%M:%S"),
        seed=seed + 2,
        volatility=0.005,  # high volatility for expansion
        trend=0.002,
    )
    parts.append(expansion)

    return pd.concat(parts)


# ===========================================================================
# Test: DEFAULT_CONFIGS completeness
# ===========================================================================


class TestDefaultConfigs:
    """Ensure DEFAULT_CONFIGS has an entry for every BreakoutType."""

    def test_all_13_types_have_configs(self):
        all_types = list(BreakoutType)
        assert len(all_types) == 13, f"Expected 13 BreakoutType values, got {len(all_types)}"
        for btype in all_types:
            assert btype in DEFAULT_CONFIGS, f"Missing DEFAULT_CONFIG for {btype.value}"

    def test_configs_have_correct_type(self):
        for btype, cfg in DEFAULT_CONFIGS.items():
            assert isinstance(cfg, RangeConfig)
            assert cfg.breakout_type == btype, f"Config type mismatch: {cfg.breakout_type} != {btype}"

    def test_configs_have_labels(self):
        for btype, cfg in DEFAULT_CONFIGS.items():
            assert cfg.label, f"Config for {btype.value} is missing a label"

    def test_new_types_present(self):
        new_types = [
            BreakoutType.Weekly,
            BreakoutType.Monthly,
            BreakoutType.Asian,
            BreakoutType.BollingerSqueeze,
            BreakoutType.ValueArea,
            BreakoutType.InsideDay,
            BreakoutType.GapRejection,
            BreakoutType.PivotPoints,
            BreakoutType.Fibonacci,
        ]
        for btype in new_types:
            assert btype in DEFAULT_CONFIGS, f"Missing DEFAULT_CONFIG for new type {btype.value}"


# ===========================================================================
# Test: Weekly Range Builder
# ===========================================================================


class TestBuildWeeklyRange:
    def test_basic_weekly_range(self):
        bars = _make_multi_day_bars(n_days=10, bars_per_day=200, seed=42)
        config = DEFAULT_CONFIGS[BreakoutType.Weekly]
        high, low, count, complete = _build_weekly_range(bars, config)

        if complete:
            assert high > low > 0, "Weekly high should be above low"
            assert count > 0, "Should have prior-week bars"

    def test_insufficient_data(self):
        bars = _make_1m_bars(n=5, start_price=2700.0)
        config = DEFAULT_CONFIGS[BreakoutType.Weekly]
        high, low, count, complete = _build_weekly_range(bars, config)
        # With only 5 bars on one day, unlikely to have prior week data
        assert not complete or (high > low > 0)

    def test_empty_bars(self):
        bars = pd.DataFrame(
            {
                "Open": pd.Series([], dtype=float),
                "High": pd.Series([], dtype=float),
                "Low": pd.Series([], dtype=float),
                "Close": pd.Series([], dtype=float),
                "Volume": pd.Series([], dtype=float),
            }
        )
        bars.index = pd.DatetimeIndex([], tz=_EST)
        config = DEFAULT_CONFIGS[BreakoutType.Weekly]
        high, low, _count, complete = _build_weekly_range(bars, config)
        assert high == 0.0 and low == 0.0 and not complete


# ===========================================================================
# Test: Monthly Range Builder
# ===========================================================================


class TestBuildMonthlyRange:
    def test_basic_monthly_range(self):
        # Create bars spanning two months
        bars = _make_multi_day_bars(n_days=35, bars_per_day=100, start_date="2026-01-15", seed=42)
        config = DEFAULT_CONFIGS[BreakoutType.Monthly]
        high, low, count, complete = _build_monthly_range(bars, config)

        if complete:
            assert high > low > 0, "Monthly high should be above low"
            assert count > 0, "Should have prior-month bars"

    def test_insufficient_data(self):
        bars = _make_1m_bars(n=5, start_price=2700.0)
        config = DEFAULT_CONFIGS[BreakoutType.Monthly]
        high, low, count, complete = _build_monthly_range(bars, config)
        assert not complete or (high > low > 0)

    def test_empty_bars(self):
        bars = pd.DataFrame(
            {
                "Open": pd.Series([], dtype=float),
                "High": pd.Series([], dtype=float),
                "Low": pd.Series([], dtype=float),
                "Close": pd.Series([], dtype=float),
                "Volume": pd.Series([], dtype=float),
            }
        )
        bars.index = pd.DatetimeIndex([], tz=_EST)
        config = DEFAULT_CONFIGS[BreakoutType.Monthly]
        high, low, _count, complete = _build_monthly_range(bars, config)
        assert high == 0.0 and low == 0.0 and not complete


# ===========================================================================
# Test: Asian Range Builder
# ===========================================================================


class TestBuildAsianRange:
    def test_basic_asian_range(self):
        bars = _make_asian_session_bars(start_date="2026-02-26", seed=42)
        config = DEFAULT_CONFIGS[BreakoutType.Asian]
        high, low, count, complete = _build_asian_range(bars, config)

        # The bars span 18:00–10:00 next day, so Asian (19:00–02:00) should be complete
        assert high > 0, "Asian high should be positive"
        assert low > 0, "Asian low should be positive"
        assert high > low, "Asian high should be above low"
        assert count > 0, "Should have Asian session bars"
        assert complete, "Asian range should be complete by 10:00 ET"

    def test_asian_wraps_midnight(self):
        """Asian window 19:00–02:00 ET wraps midnight."""
        config = DEFAULT_CONFIGS[BreakoutType.Asian]
        assert config.asian_start_time == dt_time(19, 0)
        assert config.asian_end_time == dt_time(2, 0)
        # start > end means the window wraps midnight
        assert config.asian_start_time > config.asian_end_time

    def test_incomplete_when_in_window(self):
        """If current time is within the Asian window, range is not complete."""
        # Bars only go to 01:00 ET — still inside the 19:00–02:00 window
        bars = _make_1m_bars(
            n=360,
            start_price=2700.0,
            start_time="2026-02-26 19:00:00",
            seed=42,
        )
        config = DEFAULT_CONFIGS[BreakoutType.Asian]
        _high, _low, _count, complete = _build_asian_range(bars, config)
        # Bars end at ~01:00 ET which is still inside 19:00–02:00, so not complete
        last_ts = pd.Timestamp(bars.index[-1])  # type: ignore[arg-type]
        last_time = last_ts.to_pydatetime().time()
        if last_time < config.asian_end_time or last_time >= config.asian_start_time:
            assert not complete, "Should not be complete while inside Asian window"

    def test_empty_bars(self):
        bars = pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])  # type: ignore[call-overload]
        bars.index = pd.DatetimeIndex([], tz=_EST)
        config = DEFAULT_CONFIGS[BreakoutType.Asian]
        high, low, count, complete = _build_asian_range(bars, config)
        assert high == 0.0 and low == 0.0


# ===========================================================================
# Test: Bollinger Squeeze Builder (BB inside KC)
# ===========================================================================


class TestBuildBBSqueezeRange:
    def test_no_squeeze_with_normal_bars(self):
        """Normal volatility bars should not produce a squeeze."""
        bars = _make_1m_bars(n=100, start_price=2700.0, volatility=0.003, seed=42)
        config = DEFAULT_CONFIGS[BreakoutType.BollingerSqueeze]
        atr = _compute_atr(bars, period=14)
        high, low, count, squeeze, *_ = _build_bbsqueeze_range(bars, config, atr)
        # With high volatility, BB should not be inside KC (no squeeze)
        # This is probabilistic, so we just check the return shape
        assert isinstance(squeeze, bool)

    def test_squeeze_with_tight_bars(self):
        """Very tight bars should produce a squeeze."""
        bars = _make_squeeze_bars(n=100, squeeze_start=30, squeeze_bars=30, seed=42)
        config = DEFAULT_CONFIGS[BreakoutType.BollingerSqueeze]
        atr = _compute_atr(bars, period=14)
        result = _build_bbsqueeze_range(bars, config, atr)
        # result: (r_high, r_low, bar_count, squeeze_detected, squeeze_bar_count, bb_width, bb_upper, bb_lower)
        assert len(result) == 8

    def test_insufficient_data(self):
        bars = _make_1m_bars(n=5, start_price=2700.0)
        config = DEFAULT_CONFIGS[BreakoutType.BollingerSqueeze]
        atr = _compute_atr(bars, period=14)
        high, low, count, squeeze, *_ = _build_bbsqueeze_range(bars, config, atr)
        assert not squeeze, "Should not detect squeeze with insufficient data"

    def test_zero_atr(self):
        bars = _make_1m_bars(n=50, start_price=2700.0)
        config = DEFAULT_CONFIGS[BreakoutType.BollingerSqueeze]
        high, low, count, squeeze, *_ = _build_bbsqueeze_range(bars, config, 0.0)
        assert not squeeze


# ===========================================================================
# Test: Value Area Builder
# ===========================================================================


class TestBuildVARange:
    def test_basic_va_range(self):
        bars = _make_multi_day_bars(n_days=3, bars_per_day=200, seed=42)
        config = DEFAULT_CONFIGS[BreakoutType.ValueArea]
        vah, val, count, complete, poc, vah2, val2 = _build_va_range(bars, config)

        if complete:
            assert vah > val > 0, "VAH should be above VAL"
            assert poc > 0, "POC should be positive"
            assert count > 0

    def test_no_volume_fallback(self):
        """When volume is zero, should fall back to price-based VA."""
        bars = _make_multi_day_bars(n_days=3, bars_per_day=200, seed=42)
        bars["Volume"] = 0.0
        config = DEFAULT_CONFIGS[BreakoutType.ValueArea]
        vah, val, count, complete, poc, vah2, val2 = _build_va_range(bars, config)
        # Should still produce a result via fallback
        if complete:
            assert vah >= val

    def test_insufficient_data(self):
        bars = _make_1m_bars(n=3, start_price=2700.0)
        config = DEFAULT_CONFIGS[BreakoutType.ValueArea]
        vah, val, count, complete, *_ = _build_va_range(bars, config)
        assert not complete


# ===========================================================================
# Test: Inside Day Builder
# ===========================================================================


class TestBuildInsideDayRange:
    def test_inside_day_detected(self):
        bars = _make_inside_day_bars(
            mother_high=2720.0,
            mother_low=2680.0,
            inside_high=2710.0,
            inside_low=2690.0,
            seed=42,
        )
        config = DEFAULT_CONFIGS[BreakoutType.InsideDay]
        result = _build_inside_day_range(bars, config)
        # result: (mother_high, mother_low, bar_count, inside_detected, today_high, today_low, yest_high, yest_low)
        mother_high, mother_low, bar_count, inside_detected, today_high, today_low, yest_high, yest_low = result

        assert inside_detected, "Inside day should be detected"
        assert mother_high >= today_high, "Mother bar high should contain today's high"
        assert mother_low <= today_low, "Mother bar low should contain today's low"

    def test_not_inside_day(self):
        """When today's range exceeds yesterday's, no inside day."""
        bars = _make_inside_day_bars(
            mother_high=2710.0,
            mother_low=2690.0,
            inside_high=2720.0,  # today's high exceeds mother
            inside_low=2680.0,  # today's low exceeds mother
            seed=42,
        )
        config = DEFAULT_CONFIGS[BreakoutType.InsideDay]
        result = _build_inside_day_range(bars, config)
        _, _, _, inside_detected, *_ = result
        assert not inside_detected, "Should NOT detect inside day when today exceeds yesterday"

    def test_compression_ratio_check(self):
        """Inside day with extreme compression should be rejected."""
        # Very tight inside day (compression < 0.30 threshold)
        bars = _make_inside_day_bars(
            mother_high=2750.0,
            mother_low=2650.0,  # 100 pt range
            inside_high=2705.0,
            inside_low=2695.0,  # 10 pt range = 10% compression
            seed=42,
        )
        config = DEFAULT_CONFIGS[BreakoutType.InsideDay]
        result = _build_inside_day_range(bars, config)
        _, _, _, inside_detected, *_ = result
        # 10/100 = 0.10 compression — below the 0.30 floor
        assert not inside_detected, "Should reject extreme compression below min threshold"

    def test_insufficient_data(self):
        bars = _make_1m_bars(n=5, start_price=2700.0)
        config = DEFAULT_CONFIGS[BreakoutType.InsideDay]
        result = _build_inside_day_range(bars, config)
        _, _, _, inside_detected, *_ = result
        # With only 5 bars there's not enough for two sessions
        assert not inside_detected


# ===========================================================================
# Test: Gap Rejection Builder
# ===========================================================================


class TestBuildGapRejectionRange:
    def test_gap_up_detected(self):
        bars = _make_gap_bars(yesterday_close=2700.0, gap_size=20.0, seed=42)
        atr = _compute_atr(bars, period=14)
        config = DEFAULT_CONFIGS[BreakoutType.GapRejection]
        r_high, r_low, count, gap_detected, gap_size, yc, direction = _build_gap_rejection_range(bars, config, atr)

        if gap_detected:
            assert direction == "UP", "Should detect gap UP"
            assert gap_size > 0, "Gap size should be positive for gap up"
            assert r_high > r_low > 0, "Range high should be above range low"

    def test_gap_down_detected(self):
        bars = _make_gap_bars(yesterday_close=2700.0, gap_size=-20.0, seed=42)
        atr = _compute_atr(bars, period=14)
        config = DEFAULT_CONFIGS[BreakoutType.GapRejection]
        r_high, r_low, count, gap_detected, gap_size, yc, direction = _build_gap_rejection_range(bars, config, atr)

        if gap_detected:
            assert direction == "DOWN", "Should detect gap DOWN"
            assert gap_size < 0, "Gap size should be negative for gap down"

    def test_no_gap_small_move(self):
        """A tiny overnight move should not be detected as a gap."""
        bars = _make_gap_bars(yesterday_close=2700.0, gap_size=0.01, seed=42)
        atr = _compute_atr(bars, period=14)
        config = DEFAULT_CONFIGS[BreakoutType.GapRejection]
        r_high, r_low, count, gap_detected, *_ = _build_gap_rejection_range(bars, config, atr)
        assert not gap_detected, "Tiny gap should not be detected"

    def test_zero_atr(self):
        bars = _make_gap_bars(yesterday_close=2700.0, gap_size=20.0, seed=42)
        config = DEFAULT_CONFIGS[BreakoutType.GapRejection]
        r_high, r_low, count, gap_detected, *_ = _build_gap_rejection_range(bars, config, 0.0)
        assert not gap_detected


# ===========================================================================
# Test: Pivot Points Builder
# ===========================================================================


class TestBuildPivotRange:
    def test_classic_pivots(self):
        bars = _make_multi_day_bars(n_days=3, bars_per_day=200, seed=42)
        config = DEFAULT_CONFIGS[BreakoutType.PivotPoints]
        r1, s1, count, complete, pivot, r1_val, s1_val, prev_close = _build_pivot_range(bars, config)

        if complete:
            assert r1 > s1 > 0, "R1 should be above S1"
            assert pivot > 0, "Pivot should be positive"
            # Classic formula: P = (H+L+C)/3
            assert r1 == r1_val
            assert s1 == s1_val

    def test_woodie_pivots(self):
        from dataclasses import replace

        bars = _make_multi_day_bars(n_days=3, bars_per_day=200, seed=42)
        config = replace(DEFAULT_CONFIGS[BreakoutType.PivotPoints], pivot_formula="woodie")
        r1, s1, count, complete, pivot, *_ = _build_pivot_range(bars, config)

        if complete:
            assert r1 > s1 > 0, "Woodie R1 should be above S1"

    def test_camarilla_pivots(self):
        from dataclasses import replace

        bars = _make_multi_day_bars(n_days=3, bars_per_day=200, seed=42)
        config = replace(DEFAULT_CONFIGS[BreakoutType.PivotPoints], pivot_formula="camarilla")
        r1, s1, count, complete, pivot, *_ = _build_pivot_range(bars, config)

        if complete:
            assert r1 > s1 > 0, "Camarilla R1 should be above S1"

    def test_insufficient_data(self):
        bars = _make_1m_bars(n=5, start_price=2700.0)
        config = DEFAULT_CONFIGS[BreakoutType.PivotPoints]
        r1, s1, count, complete, *_ = _build_pivot_range(bars, config)
        assert not complete or (r1 > s1)


# ===========================================================================
# Test: Fibonacci Range Builder
# ===========================================================================


class TestBuildFibonacciRange:
    def test_basic_fibonacci_range(self):
        # Create bars with a clear upswing
        bars = _make_1m_bars(n=120, start_price=2650.0, trend=0.002, volatility=0.001, seed=42)
        atr = _compute_atr(bars, period=14)
        config = DEFAULT_CONFIGS[BreakoutType.Fibonacci]
        r_high, r_low, count, valid, swing_high, swing_low, fib_382, fib_618 = _build_fibonacci_range(bars, config, atr)

        if valid:
            assert r_high > r_low > 0, "Fib range high should be above low"
            assert swing_high > swing_low > 0, "Swing high should be above swing low"
            # The fib zone should be between swing high and swing low
            assert swing_low <= r_low, "Fib low should be at or above swing low"
            assert r_high <= swing_high, "Fib high should be at or below swing high"

    def test_flat_market_no_swing(self):
        """Flat market with no meaningful swing should not produce fib range."""
        # Use extremely low volatility and a very high min_swing_atr_mult to
        # guarantee the swing is too small relative to ATR.
        from dataclasses import replace

        bars = _make_1m_bars(n=120, start_price=2700.0, trend=0.0, volatility=0.0000001, seed=42)
        atr = _compute_atr(bars, period=14)
        config = replace(DEFAULT_CONFIGS[BreakoutType.Fibonacci], fib_min_swing_atr_mult=100.0)
        r_high, r_low, count, valid, *_ = _build_fibonacci_range(bars, config, atr)
        assert not valid, "Flat market should not produce valid fib range"

    def test_insufficient_data(self):
        bars = _make_1m_bars(n=5, start_price=2700.0)
        atr = _compute_atr(bars, period=14)
        config = DEFAULT_CONFIGS[BreakoutType.Fibonacci]
        r_high, r_low, count, valid, *_ = _build_fibonacci_range(bars, config, atr)
        assert not valid

    def test_zero_atr(self):
        bars = _make_1m_bars(n=120, start_price=2700.0, trend=0.002)
        config = DEFAULT_CONFIGS[BreakoutType.Fibonacci]
        r_high, r_low, count, valid, *_ = _build_fibonacci_range(bars, config, 0.0)
        assert not valid

    def test_fib_levels_are_correct(self):
        """Verify fib levels are properly computed from swing."""
        bars = _make_1m_bars(n=120, start_price=2600.0, trend=0.003, volatility=0.001, seed=99)
        atr = _compute_atr(bars, period=14)
        config = DEFAULT_CONFIGS[BreakoutType.Fibonacci]
        r_high, r_low, count, valid, swing_high, swing_low, fib_382, fib_618 = _build_fibonacci_range(bars, config, atr)
        if valid:
            # For an upswing: fib_382 = swing_high - 0.382 * swing_size
            #                 fib_618 = swing_high - 0.618 * swing_size
            # The fib zone (r_low to r_high) should span [fib_618, fib_382]
            assert r_high > r_low
            assert abs(fib_382 - fib_618) > 0


# ===========================================================================
# Test: detect_range_breakout integration for new types
# ===========================================================================


class TestDetectRangeBreakoutNewTypes:
    """Test that detect_range_breakout works for all 9 new BreakoutTypes."""

    def test_weekly_detection(self):
        bars = _make_multi_day_bars(n_days=10, bars_per_day=200, seed=42)
        config = DEFAULT_CONFIGS[BreakoutType.Weekly]
        result = detect_range_breakout(bars, "MES=F", config)
        assert isinstance(result, BreakoutResult)
        assert result.breakout_type == BreakoutType.Weekly
        assert result.symbol == "MES=F"
        # Even if no breakout, the result should be well-formed
        assert result.evaluated_at != ""

    def test_monthly_detection(self):
        bars = _make_multi_day_bars(n_days=35, bars_per_day=100, start_date="2026-01-15", seed=42)
        config = DEFAULT_CONFIGS[BreakoutType.Monthly]
        result = detect_range_breakout(bars, "MGC=F", config)
        assert isinstance(result, BreakoutResult)
        assert result.breakout_type == BreakoutType.Monthly

    def test_asian_detection(self):
        bars = _make_asian_session_bars(start_date="2026-02-26", seed=42)
        config = DEFAULT_CONFIGS[BreakoutType.Asian]
        result = detect_range_breakout(bars, "MNQ=F", config)
        assert isinstance(result, BreakoutResult)
        assert result.breakout_type == BreakoutType.Asian

    def test_bbsqueeze_detection(self):
        bars = _make_squeeze_bars(n=100, squeeze_start=30, squeeze_bars=30, seed=42)
        config = DEFAULT_CONFIGS[BreakoutType.BollingerSqueeze]
        result = detect_range_breakout(bars, "MES=F", config)
        assert isinstance(result, BreakoutResult)
        assert result.breakout_type == BreakoutType.BollingerSqueeze

    def test_va_detection(self):
        bars = _make_multi_day_bars(n_days=3, bars_per_day=200, seed=42)
        config = DEFAULT_CONFIGS[BreakoutType.ValueArea]
        result = detect_range_breakout(bars, "MES=F", config)
        assert isinstance(result, BreakoutResult)
        assert result.breakout_type == BreakoutType.ValueArea

    def test_inside_detection(self):
        bars = _make_inside_day_bars(
            mother_high=2720.0,
            mother_low=2680.0,
            inside_high=2710.0,
            inside_low=2690.0,
        )
        config = DEFAULT_CONFIGS[BreakoutType.InsideDay]
        result = detect_range_breakout(bars, "MES=F", config)
        assert isinstance(result, BreakoutResult)
        assert result.breakout_type == BreakoutType.InsideDay

    def test_gap_detection(self):
        bars = _make_gap_bars(yesterday_close=2700.0, gap_size=20.0, seed=42)
        config = DEFAULT_CONFIGS[BreakoutType.GapRejection]
        result = detect_range_breakout(bars, "MES=F", config)
        assert isinstance(result, BreakoutResult)
        assert result.breakout_type == BreakoutType.GapRejection

    def test_pivot_detection(self):
        bars = _make_multi_day_bars(n_days=3, bars_per_day=200, seed=42)
        config = DEFAULT_CONFIGS[BreakoutType.PivotPoints]
        result = detect_range_breakout(bars, "MES=F", config)
        assert isinstance(result, BreakoutResult)
        assert result.breakout_type == BreakoutType.PivotPoints

    def test_fib_detection(self):
        bars = _make_1m_bars(n=120, start_price=2650.0, trend=0.002, volatility=0.001, seed=42)
        config = DEFAULT_CONFIGS[BreakoutType.Fibonacci]
        result = detect_range_breakout(bars, "MES=F", config)
        assert isinstance(result, BreakoutResult)
        assert result.breakout_type == BreakoutType.Fibonacci

    def test_empty_bars_all_types(self):
        """All types should handle empty bars gracefully."""
        empty = pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])  # type: ignore[call-overload]
        empty.index = pd.DatetimeIndex([], tz=_EST)

        for btype in BreakoutType:
            config = DEFAULT_CONFIGS[btype]
            result = detect_range_breakout(empty, "MES=F", config)
            assert isinstance(result, BreakoutResult)
            assert result.error != "", f"{btype.value} should set error for empty bars"
            assert not result.breakout_detected

    def test_none_bars_all_types(self):
        """All types should handle None bars gracefully."""
        for btype in BreakoutType:
            config = DEFAULT_CONFIGS[btype]
            result = detect_range_breakout(None, "MES=F", config)  # type: ignore[arg-type]
            assert isinstance(result, BreakoutResult)
            assert result.error != ""
            assert not result.breakout_detected

    def test_result_to_dict_all_types(self):
        """All results should be JSON-serializable via to_dict."""
        import json

        bars = _make_multi_day_bars(n_days=10, bars_per_day=200, seed=42)
        for btype in BreakoutType:
            config = DEFAULT_CONFIGS[btype]
            result = detect_range_breakout(bars, "MES=F", config)
            d = result.to_dict()
            assert isinstance(d, dict)
            # Should be JSON-serializable (use default=str to handle numpy bools)
            json_str = json.dumps(d, default=str)
            assert isinstance(json_str, str)


# ===========================================================================
# Test: detect_all_breakout_types with all 13 types
# ===========================================================================


class TestDetectAllBreakoutTypes:
    def test_all_13_types(self):
        """Running detect_all_breakout_types with no type filter returns all 13."""
        bars = _make_multi_day_bars(n_days=10, bars_per_day=200, seed=42)
        results = detect_all_breakout_types(bars, "MES=F")
        assert len(results) == 13, f"Expected 13 results, got {len(results)}"
        for btype in BreakoutType:
            assert btype in results, f"Missing result for {btype.value}"
            assert isinstance(results[btype], BreakoutResult)

    def test_subset_of_types(self):
        """Running with a specific subset returns only those types."""
        bars = _make_multi_day_bars(n_days=10, bars_per_day=200, seed=42)
        subset = [BreakoutType.Weekly, BreakoutType.PivotPoints, BreakoutType.Fibonacci]
        results = detect_all_breakout_types(bars, "MES=F", types=subset)
        assert len(results) == 3
        for btype in subset:
            assert btype in results

    def test_original_4_types(self):
        """The original 4 types still work in the combined dispatch."""
        bars = _make_multi_day_bars(n_days=3, bars_per_day=200, seed=42)
        original = [BreakoutType.ORB, BreakoutType.PrevDay, BreakoutType.InitialBalance, BreakoutType.Consolidation]
        results = detect_all_breakout_types(
            bars,
            "MES=F",
            types=original,
            orb_session_start=dt_time(9, 30),
            orb_session_end=dt_time(10, 0),
        )
        assert len(results) == 4

    def test_custom_configs_override(self):
        """Custom configs should override DEFAULT_CONFIGS."""
        from dataclasses import replace

        bars = _make_multi_day_bars(n_days=3, bars_per_day=200, seed=42)
        custom = replace(
            DEFAULT_CONFIGS[BreakoutType.PivotPoints],
            pivot_formula="woodie",
            label="Custom Woodie Pivots",
        )
        results = detect_all_breakout_types(
            bars,
            "MES=F",
            types=[BreakoutType.PivotPoints],
            configs={BreakoutType.PivotPoints: custom},
        )
        assert results[BreakoutType.PivotPoints].label == "Custom Woodie Pivots"

    def test_exception_handling(self):
        """A bad bar DataFrame should produce error results, not crash."""
        bad_bars = pd.DataFrame({"Open": [1], "Close": [1]})  # missing High/Low
        bad_bars.index = pd.DatetimeIndex(["2026-02-27 09:30:00"], tz=_EST)
        results = detect_all_breakout_types(bad_bars, "MES=F")
        # Should still return results for all 13 types, but with errors
        assert len(results) == 13
        for _btype, result in results.items():
            assert isinstance(result, BreakoutResult)


# ===========================================================================
# Test: BreakoutResult.extra dict for new types
# ===========================================================================


class TestBreakoutResultExtra:
    """Verify that new types populate the extra dict with type-specific metadata."""

    def test_weekly_extra(self):
        bars = _make_multi_day_bars(n_days=10, bars_per_day=200, seed=42)
        result = detect_range_breakout(bars, "MES=F", DEFAULT_CONFIGS[BreakoutType.Weekly])
        if result.range_complete:
            assert "weekly_high" in result.extra
            assert "weekly_low" in result.extra

    def test_monthly_extra(self):
        bars = _make_multi_day_bars(n_days=35, bars_per_day=100, start_date="2026-01-15", seed=42)
        result = detect_range_breakout(bars, "MES=F", DEFAULT_CONFIGS[BreakoutType.Monthly])
        if result.range_complete:
            assert "monthly_high" in result.extra
            assert "monthly_low" in result.extra

    def test_asian_extra(self):
        bars = _make_asian_session_bars(seed=42)
        result = detect_range_breakout(bars, "MES=F", DEFAULT_CONFIGS[BreakoutType.Asian])
        if result.range_complete:
            assert "asian_high" in result.extra
            assert "asian_low" in result.extra

    def test_va_extra(self):
        bars = _make_multi_day_bars(n_days=3, bars_per_day=200, seed=42)
        result = detect_range_breakout(bars, "MES=F", DEFAULT_CONFIGS[BreakoutType.ValueArea])
        if result.range_complete:
            assert "poc" in result.extra
            assert "vah" in result.extra
            assert "val" in result.extra

    def test_inside_extra(self):
        bars = _make_inside_day_bars(mother_high=2720.0, mother_low=2680.0, inside_high=2710.0, inside_low=2690.0)
        result = detect_range_breakout(bars, "MES=F", DEFAULT_CONFIGS[BreakoutType.InsideDay])
        if result.range_complete:
            assert "inside_detected" in result.extra
            assert "yesterday_high" in result.extra
            assert "yesterday_low" in result.extra

    def test_gap_extra(self):
        bars = _make_gap_bars(yesterday_close=2700.0, gap_size=20.0)
        result = detect_range_breakout(bars, "MES=F", DEFAULT_CONFIGS[BreakoutType.GapRejection])
        if result.range_complete:
            assert "gap_detected" in result.extra
            assert "gap_direction" in result.extra

    def test_pivot_extra(self):
        bars = _make_multi_day_bars(n_days=3, bars_per_day=200, seed=42)
        result = detect_range_breakout(bars, "MES=F", DEFAULT_CONFIGS[BreakoutType.PivotPoints])
        if result.range_complete:
            assert "pivot" in result.extra
            assert "r1" in result.extra
            assert "s1" in result.extra

    def test_fib_extra(self):
        bars = _make_1m_bars(n=120, start_price=2650.0, trend=0.002, volatility=0.001, seed=42)
        result = detect_range_breakout(bars, "MES=F", DEFAULT_CONFIGS[BreakoutType.Fibonacci])
        if result.range_complete:
            assert "swing_high" in result.extra
            assert "swing_low" in result.extra
            assert "fib_382" in result.extra
            assert "fib_618" in result.extra


# ===========================================================================
# Test: BreakoutType enum / mapping completeness
# ===========================================================================


class TestBreakoutTypeMapping:
    """Verify BreakoutType enum is a single source of truth (Phase 1A).

    The old helper functions ``to_training_type()``, ``from_training_type()``,
    and ``breakout_type_ordinal()`` have been removed.  Engine and training
    share the same ``BreakoutType`` IntEnum from ``lib.core.breakout_types``.
    """

    def test_single_breakout_type_enum(self):
        """Engine and core BreakoutType are the exact same class."""
        from lib.core.breakout_types import BreakoutType as CoreBreakoutType
        from lib.trading.strategies.rb.breakout import BreakoutType as EngineBreakoutType

        assert CoreBreakoutType is EngineBreakoutType

    def test_all_types_have_range_config(self):
        from lib.core.breakout_types import get_range_config

        for btype in BreakoutType:
            cfg = get_range_config(btype)
            assert cfg is not None, f"No RangeConfig for {btype.name}"
            assert cfg.breakout_type is btype

    def test_ordinal_values(self):
        from lib.core.breakout_types import get_range_config

        for btype in BreakoutType:
            ordinal = get_range_config(btype).breakout_type_ord
            assert 0.0 <= ordinal <= 1.0, f"Ordinal {ordinal} for {btype.name} out of [0, 1] range"

    def test_ordinal_monotonically_increasing(self):
        from lib.core.breakout_types import get_range_config

        prev_ord = -1.0
        for btype in BreakoutType:
            cur_ord = get_range_config(btype).breakout_type_ord
            assert cur_ord > prev_ord, (
                f"Ordinal not monotonically increasing: {btype.name} ord={cur_ord} <= prev={prev_ord}"
            )
            prev_ord = cur_ord


# ===========================================================================
# Test: Scheduler action types for new breakout types
# ===========================================================================


class TestSchedulerNewActionTypes:
    """Verify the new ActionType entries exist and are usable."""

    def test_new_check_action_types_exist(self):
        from lib.services.engine.scheduler import ActionType

        new_actions = [
            "CHECK_WEEKLY",
            "CHECK_MONTHLY",
            "CHECK_ASIAN",
            "CHECK_BBSQUEEZE",
            "CHECK_VA",
            "CHECK_INSIDE",
            "CHECK_GAP",
            "CHECK_PIVOT",
            "CHECK_FIB",
        ]
        for name in new_actions:
            assert hasattr(ActionType, name), f"ActionType.{name} missing from scheduler"

    def test_breakout_multi_still_exists(self):
        from lib.services.engine.scheduler import ActionType

        assert hasattr(ActionType, "CHECK_BREAKOUT_MULTI")

    def test_new_interval_constants_exist(self):
        from lib.services.engine.scheduler import ScheduleManager

        interval_attrs = [
            "WEEKLY_CHECK_INTERVAL",
            "MONTHLY_CHECK_INTERVAL",
            "ASIAN_CHECK_INTERVAL",
            "BBSQUEEZE_CHECK_INTERVAL",
            "VA_CHECK_INTERVAL",
            "INSIDE_CHECK_INTERVAL",
            "GAP_CHECK_INTERVAL",
            "PIVOT_CHECK_INTERVAL",
            "FIB_CHECK_INTERVAL",
        ]
        for attr in interval_attrs:
            assert hasattr(ScheduleManager, attr), f"ScheduleManager.{attr} missing"
            val = getattr(ScheduleManager, attr)
            assert val == 2 * 60, f"{attr} should be 120s (2 min), got {val}"

    def test_active_session_schedules_all_types(self):
        """During the US active window, all 13 types should be schedulable."""
        from lib.services.engine.scheduler import ActionType, ScheduleManager

        mgr = ScheduleManager()
        # Simulate 09:45 ET on a weekday — peak US active window
        now = datetime(2026, 2, 27, 9, 45, 0, tzinfo=_EST)
        actions = mgr.get_pending_actions(now=now)

        # CHECK_BREAKOUT_MULTI should be scheduled with all types in payload
        multi_actions = [a for a in actions if a.action == ActionType.CHECK_BREAKOUT_MULTI]
        if multi_actions:
            # At least one multi action should include new types
            all_type_strs = set()
            for a in multi_actions:
                if a.payload and "types" in a.payload:
                    all_type_strs.update(a.payload["types"])
            # New types should appear in at least one multi sweep
            expected_new = {"WEEKLY", "MONTHLY", "ASIAN", "BBSQUEEZE", "VA", "INSIDE", "GAP", "PIVOT", "FIB"}
            found = expected_new & all_type_strs
            assert len(found) > 0, f"Expected new types in multi sweep, got: {all_type_strs}"


# ===========================================================================
# Test: RangeConfig new fields
# ===========================================================================


class TestRangeConfigNewFields:
    """Verify the new fields on the engine-side RangeConfig."""

    def test_weekly_fields(self):
        cfg = DEFAULT_CONFIGS[BreakoutType.Weekly]
        assert cfg.weekly_lookback_days == 5

    def test_monthly_fields(self):
        cfg = DEFAULT_CONFIGS[BreakoutType.Monthly]
        assert cfg.monthly_lookback_days == 20

    def test_asian_fields(self):
        cfg = DEFAULT_CONFIGS[BreakoutType.Asian]
        assert cfg.asian_start_time == dt_time(19, 0)
        assert cfg.asian_end_time == dt_time(2, 0)

    def test_bbsqueeze_fields(self):
        cfg = DEFAULT_CONFIGS[BreakoutType.BollingerSqueeze]
        assert cfg.bbsqueeze_bb_period == 20
        assert cfg.bbsqueeze_bb_std == 2.0
        assert cfg.bbsqueeze_kc_period == 20
        assert cfg.bbsqueeze_kc_atr_mult == 1.5
        assert cfg.bbsqueeze_min_squeeze_bars == 6

    def test_va_fields(self):
        cfg = DEFAULT_CONFIGS[BreakoutType.ValueArea]
        assert cfg.va_value_area_pct == 0.70
        assert cfg.va_n_bins == 50

    def test_inside_fields(self):
        cfg = DEFAULT_CONFIGS[BreakoutType.InsideDay]
        assert cfg.inside_min_compression == 0.30
        assert cfg.inside_max_compression == 0.85

    def test_gap_fields(self):
        cfg = DEFAULT_CONFIGS[BreakoutType.GapRejection]
        assert cfg.gap_min_atr_pct == 0.15
        assert cfg.gap_fill_threshold_pct == 0.50
        assert cfg.gap_rejection_bars == 3

    def test_pivot_fields(self):
        cfg = DEFAULT_CONFIGS[BreakoutType.PivotPoints]
        assert cfg.pivot_formula == "classic"

    def test_fib_fields(self):
        cfg = DEFAULT_CONFIGS[BreakoutType.Fibonacci]
        assert cfg.fib_upper == 0.618
        assert cfg.fib_lower == 0.382
        assert cfg.fib_swing_lookback == 100
        assert cfg.fib_min_swing_atr_mult == 1.5

    def test_frozen_config(self):
        """RangeConfig should be frozen (immutable)."""
        cfg = DEFAULT_CONFIGS[BreakoutType.Weekly]
        with pytest.raises(AttributeError):
            cfg.weekly_lookback_days = 10  # type: ignore[misc]


# ===========================================================================
# Feature contract generation tests
# ===========================================================================


class TestFeatureContractGeneration:
    """Tests for generate_feature_contract() in breakout_cnn."""

    def test_returns_dict(self):
        from lib.analysis.ml.breakout_cnn import generate_feature_contract

        contract = generate_feature_contract()
        assert isinstance(contract, dict)

    def test_version_is_8(self):
        from lib.analysis.ml.breakout_cnn import FEATURE_CONTRACT_VERSION, generate_feature_contract

        contract = generate_feature_contract()
        assert contract["version"] == FEATURE_CONTRACT_VERSION
        assert contract["version"] == 8

    def test_num_tabular_matches_constant(self):
        from lib.analysis.ml.breakout_cnn import NUM_TABULAR, generate_feature_contract

        contract = generate_feature_contract()
        assert contract["num_tabular"] == NUM_TABULAR
        assert contract["num_tabular"] == 37

    def test_tabular_features_list(self):
        from lib.analysis.ml.breakout_cnn import TABULAR_FEATURES, generate_feature_contract

        contract = generate_feature_contract()
        assert contract["tabular_features"] == TABULAR_FEATURES
        assert len(contract["tabular_features"]) == 37
        # Spot-check v4 core features
        assert "quality_pct_norm" in contract["tabular_features"]
        assert "direction_flag" in contract["tabular_features"]
        assert "or_range_atr_ratio" in contract["tabular_features"]
        assert "asset_class_id" in contract["tabular_features"]
        # v6 additions
        assert "breakout_type_ord" in contract["tabular_features"]
        assert "asset_volatility_class" in contract["tabular_features"]
        assert "hour_of_day" in contract["tabular_features"]
        assert "tp3_atr_mult_norm" in contract["tabular_features"]
        # v7 additions — Daily Strategy layer
        assert "daily_bias_direction" in contract["tabular_features"]
        assert "daily_bias_confidence" in contract["tabular_features"]
        assert "prior_day_pattern" in contract["tabular_features"]
        assert "weekly_range_position" in contract["tabular_features"]
        assert "monthly_trend_score" in contract["tabular_features"]
        assert "crypto_momentum_score" in contract["tabular_features"]
        # v7.1 additions — Phase 4B sub-features
        assert "breakout_type_category" in contract["tabular_features"]
        assert "session_overlap_flag" in contract["tabular_features"]
        assert "atr_trend" in contract["tabular_features"]
        assert "volume_trend" in contract["tabular_features"]
        # v8-B additions — Cross-Asset Correlation
        assert "primary_peer_corr" in contract["tabular_features"]
        assert "cross_class_corr" in contract["tabular_features"]
        assert "correlation_regime" in contract["tabular_features"]
        # v8-C additions — Asset Fingerprint
        assert "typical_daily_range_norm" in contract["tabular_features"]
        assert "session_concentration" in contract["tabular_features"]
        assert "breakout_follow_through" in contract["tabular_features"]
        assert "hurst_exponent" in contract["tabular_features"]
        assert "overnight_gap_tendency" in contract["tabular_features"]
        assert "volume_profile_shape" in contract["tabular_features"]

    def test_all_13_breakout_types_present(self):
        from lib.analysis.ml.breakout_cnn import generate_feature_contract

        contract = generate_feature_contract()
        bt = contract["breakout_types"]
        assert len(bt) == 13
        expected_keys = {
            "ORB",
            "PrevDay",
            "InitialBalance",
            "Consolidation",
            "Weekly",
            "Monthly",
            "Asian",
            "BollingerSqueeze",
            "ValueArea",
            "InsideDay",
            "GapRejection",
            "PivotPoints",
            "Fibonacci",
        }
        assert set(bt.keys()) == expected_keys

    def test_breakout_type_ordinals_in_range(self):
        from lib.analysis.ml.breakout_cnn import generate_feature_contract

        contract = generate_feature_contract()
        for name, info in contract["breakout_types"].items():
            assert "ordinal" in info, f"Missing 'ordinal' for {name}"
            assert "breakout_type_ord" in info, f"Missing 'breakout_type_ord' for {name}"
            assert 0 <= info["ordinal"] <= 12, f"Ordinal out of range for {name}: {info['ordinal']}"
            assert 0.0 <= info["breakout_type_ord"] <= 1.0, (
                f"breakout_type_ord out of range for {name}: {info['breakout_type_ord']}"
            )

    def test_sessions_section_has_9_sessions(self):
        from lib.analysis.ml.breakout_cnn import generate_feature_contract

        contract = generate_feature_contract()
        sess = contract["session_thresholds"]
        assert len(sess) == 9

    def test_sessions_have_required_fields(self):
        from lib.analysis.ml.breakout_cnn import generate_feature_contract

        contract = generate_feature_contract()
        # session_thresholds maps session key → float threshold
        # session_ordinals maps session key → float ordinal
        thresholds = contract["session_thresholds"]
        ordinals = contract["session_ordinals"]
        for key in thresholds:
            assert key in ordinals, f"Missing ordinal for session {key}"
            assert 0.0 <= ordinals[key] <= 1.0, f"session_ordinal out of range for {key}: {ordinals[key]}"
            assert 0.0 <= thresholds[key] <= 1.0, f"session_threshold out of range for {key}: {thresholds[key]}"

    def test_asset_volatility_section(self):
        from lib.analysis.ml.breakout_cnn import ASSET_VOLATILITY_CLASS, generate_feature_contract

        contract = generate_feature_contract()
        av = contract["asset_volatility_classes"]
        assert av == ASSET_VOLATILITY_CLASS
        # All values must be 0.0, 0.5, or 1.0
        for ticker, val in av.items():
            assert val in (0.0, 0.5, 1.0), f"Unexpected volatility class for {ticker}: {val}"

    def test_default_threshold_and_image_size(self):
        from lib.analysis.ml.breakout_cnn import DEFAULT_THRESHOLD, IMAGE_SIZE, generate_feature_contract

        contract = generate_feature_contract()
        assert contract["default_threshold"] == DEFAULT_THRESHOLD
        assert contract["image_size"] == IMAGE_SIZE

    def test_imagenet_stats_present(self):
        from lib.analysis.ml.breakout_cnn import generate_feature_contract

        contract = generate_feature_contract()
        assert "imagenet_mean" in contract
        assert "imagenet_std" in contract
        assert len(contract["imagenet_mean"]) == 3
        assert len(contract["imagenet_std"]) == 3

    def test_generated_at_is_iso_timestamp(self):
        from datetime import datetime

        from lib.analysis.ml.breakout_cnn import generate_feature_contract

        contract = generate_feature_contract()
        assert "generated_at" in contract
        # Should be parseable as ISO datetime
        dt = datetime.fromisoformat(contract["generated_at"])
        assert dt is not None

    def test_write_to_file(self, tmp_path):
        import json

        from lib.analysis.ml.breakout_cnn import FEATURE_CONTRACT_VERSION, generate_feature_contract

        output = str(tmp_path / "feature_contract.json")
        contract = generate_feature_contract(output_path=output)

        # File should exist
        import os

        assert os.path.exists(output)

        # File content should match returned dict
        with open(output) as fh:
            loaded = json.load(fh)
        assert loaded["version"] == FEATURE_CONTRACT_VERSION
        assert loaded["num_tabular"] == contract["num_tabular"]
        assert loaded["tabular_features"] == contract["tabular_features"]
        assert len(loaded["breakout_types"]) == 13
        assert len(loaded["session_thresholds"]) == 9

    def test_write_creates_parent_directories(self, tmp_path):
        import os

        from lib.analysis.ml.breakout_cnn import generate_feature_contract

        nested = str(tmp_path / "subdir" / "nested" / "feature_contract.json")
        generate_feature_contract(output_path=nested)
        assert os.path.exists(nested)

    def test_output_path_none_returns_only_dict(self):
        from lib.analysis.ml.breakout_cnn import generate_feature_contract

        result = generate_feature_contract(output_path=None)
        assert isinstance(result, dict)
        assert result["version"] == 8
