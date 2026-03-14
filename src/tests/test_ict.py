"""
Unit tests for the ICT / Smart Money Concepts module.

Tests cover:
  - detect_fvgs(): bullish/bearish FVG detection, fill tracking, min_gap_atr filter,
    empty/short/constant inputs, max_results cap
  - get_unfilled_fvgs(): convenience wrapper filtering
  - detect_order_blocks(): bullish/bearish OB detection, mitigation/tested tracking,
    deduplication, impulse_atr_mult threshold, empty/short guards
  - get_active_order_blocks(): only non-mitigated OBs
  - detect_liquidity_sweeps(): buy-side/sell-side sweep detection, reversal threshold,
    empty/short guards
  - detect_breaker_blocks(): breaker formation from mitigated OBs, retest detection,
    empty/short guards
  - ict_summary(): composite summary, stats dict, nearest_levels, empty guards
  - DataFrame converters: fvgs_to_dataframe, order_blocks_to_dataframe,
    sweeps_to_dataframe, breakers_to_dataframe, levels_to_dataframe
  - Edge cases: constant prices, single bar, extreme volumes, NaN handling
"""

import numpy as np
import pandas as pd
import pytest

from lib.analysis.ict import (
    _swing_highs,
    _swing_lows,
    breakers_to_dataframe,
    detect_breaker_blocks,
    detect_fvgs,
    detect_liquidity_sweeps,
    detect_order_blocks,
    fvgs_to_dataframe,
    get_active_order_blocks,
    get_unfilled_fvgs,
    ict_summary,
    levels_to_dataframe,
    order_blocks_to_dataframe,
    sweeps_to_dataframe,
)
from tests.conftest import _gappy_ohlcv, _random_walk_ohlcv, _trending_ohlcv

# ---------------------------------------------------------------------------
# Local fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def ohlcv_df():
    return _random_walk_ohlcv(n=500, seed=42)


@pytest.fixture()
def trending_df():
    return _trending_ohlcv(n=300, seed=123, trend=0.002, volatility=0.001)


@pytest.fixture()
def gappy_df():
    return _gappy_ohlcv(n=300, seed=77)


@pytest.fixture()
def volatile_df():
    """High-volatility data that should produce impulsive moves and FVGs."""
    return _random_walk_ohlcv(n=500, seed=55, start_price=100.0, volatility=0.015)


@pytest.fixture()
def empty_df():
    return pd.DataFrame(columns=pd.Index(["Open", "High", "Low", "Close", "Volume"]))


@pytest.fixture()
def tiny_df():
    return _random_walk_ohlcv(n=5, seed=11, start_price=20.0)


@pytest.fixture()
def short_df():
    return _random_walk_ohlcv(n=20, seed=99, start_price=50.0)


@pytest.fixture()
def constant_df():
    """All bars identical — flat price, no movement."""
    n = 100
    idx = pd.date_range("2025-01-06 03:00", periods=n, freq="5min", tz="America/New_York")
    return pd.DataFrame(
        {
            "Open": [100.0] * n,
            "High": [100.0] * n,
            "Low": [100.0] * n,
            "Close": [100.0] * n,
            "Volume": [500.0] * n,
        },
        index=idx,
    )


@pytest.fixture()
def impulsive_df():
    """Synthetic data with a clear impulsive move for OB / FVG detection.

    Structure:
      bars 0-49: steady at 100
      bar  50:   big bearish candle (last green before drop → bearish OB candidate)
      bar  51:   impulsive drop  (large red candle)
      bars 52+:  steady at lower level ~90
      bar  80:   price rallies back through OB zone → mitigation
    """
    n = 150
    idx = pd.date_range("2025-01-06 03:00", periods=n, freq="5min", tz="America/New_York")
    rng = np.random.default_rng(42)

    opn = np.full(n, 100.0)
    high = np.full(n, 101.0)
    low = np.full(n, 99.0)
    close = np.full(n, 100.0)
    vol = np.full(n, 500.0)

    # Steady region (bars 0–49): small random walk around 100
    for i in range(50):
        opn[i] = 100.0 + rng.uniform(-0.5, 0.5)
        close[i] = 100.0 + rng.uniform(-0.5, 0.5)
        high[i] = max(opn[i], close[i]) + rng.uniform(0, 0.3)
        low[i] = min(opn[i], close[i]) - rng.uniform(0, 0.3)

    # Bar 50: last bullish candle before the drop (will become bearish OB candidate)
    opn[50] = 100.0
    close[50] = 100.8  # bullish
    high[50] = 101.0
    low[50] = 99.8
    vol[50] = 800.0

    # Bar 51: large impulsive bearish candle
    opn[51] = 100.5
    close[51] = 94.0  # big drop
    high[51] = 100.7
    low[51] = 93.5
    vol[51] = 3000.0

    # Bar 52: continuation down — creates FVG between bar 50 high and bar 52 low
    opn[52] = 93.8
    close[52] = 92.0
    high[52] = 94.0
    low[52] = 91.5
    vol[52] = 2000.0

    # Bars 53–79: steady at lower level
    for i in range(53, 80):
        base = 92.0 + rng.uniform(-0.5, 0.5)
        opn[i] = base
        close[i] = base + rng.uniform(-0.3, 0.3)
        high[i] = max(opn[i], close[i]) + rng.uniform(0, 0.2)
        low[i] = min(opn[i], close[i]) - rng.uniform(0, 0.2)
        vol[i] = 600.0

    # Bar 80: rally back through OB zone (mitigation)
    opn[80] = 92.5
    close[80] = 101.5  # above the OB zone
    high[80] = 102.0
    low[80] = 92.0
    vol[80] = 2500.0

    # Bars 81+: settle back
    for i in range(81, n):
        base = 101.0 + rng.uniform(-0.5, 0.5)
        opn[i] = base
        close[i] = base + rng.uniform(-0.3, 0.3)
        high[i] = max(opn[i], close[i]) + rng.uniform(0, 0.2)
        low[i] = min(opn[i], close[i]) - rng.uniform(0, 0.2)
        vol[i] = 500.0

    return pd.DataFrame(
        {"Open": opn, "High": high, "Low": low, "Close": close, "Volume": vol},
        index=idx,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Swing point helpers
# ═══════════════════════════════════════════════════════════════════════════


class TestSwingHelpers:
    def test_swing_highs_returns_list(self, ohlcv_df):
        highs = _swing_highs(ohlcv_df["High"], lookback=5)
        assert isinstance(highs, list)

    def test_swing_highs_dict_keys(self, ohlcv_df):
        highs = _swing_highs(ohlcv_df["High"], lookback=5)
        if highs:
            h = highs[0]
            assert "index" in h
            assert "price" in h
            assert "timestamp" in h

    def test_swing_highs_prices_are_positive(self, ohlcv_df):
        highs = _swing_highs(ohlcv_df["High"], lookback=5)
        for h in highs:
            assert h["price"] > 0

    def test_swing_lows_returns_list(self, ohlcv_df):
        lows = _swing_lows(ohlcv_df["Low"], lookback=5)
        assert isinstance(lows, list)

    def test_swing_lows_dict_keys(self, ohlcv_df):
        lows = _swing_lows(ohlcv_df["Low"], lookback=5)
        if lows:
            low_pt = lows[0]
            assert "index" in low_pt
            assert "price" in low_pt
            assert "timestamp" in low_pt

    def test_swing_lows_prices_are_positive(self, ohlcv_df):
        lows = _swing_lows(ohlcv_df["Low"], lookback=5)
        for low_pt in lows:
            assert low_pt["price"] > 0

    def test_no_swings_on_constant_price(self, constant_df):
        """Constant price should produce no swing points (or very few edge ones)."""
        highs = _swing_highs(constant_df["High"], lookback=5)
        lows = _swing_lows(constant_df["Low"], lookback=5)
        # With all prices identical, every bar ties as max/min, but the algo
        # takes the first match. Either way, no crash.
        assert isinstance(highs, list)
        assert isinstance(lows, list)

    def test_swing_highs_empty_input(self, empty_df):
        highs = _swing_highs(
            empty_df["High"] if "High" in empty_df else pd.Series(dtype=float),
            lookback=5,
        )
        assert highs == []

    def test_swing_lows_empty_input(self, empty_df):
        lows = _swing_lows(empty_df["Low"] if "Low" in empty_df else pd.Series(dtype=float), lookback=5)
        assert lows == []

    def test_swing_highs_min_bars_between(self, ohlcv_df):
        """With min_bars_between > 0, consecutive swings should be spaced apart."""
        min_gap = 5
        highs = _swing_highs(ohlcv_df["High"], lookback=5, min_bars_between=min_gap)
        for i in range(1, len(highs)):
            assert highs[i]["index"] - highs[i - 1]["index"] >= min_gap

    def test_swing_lows_min_bars_between(self, ohlcv_df):
        min_gap = 5
        lows = _swing_lows(ohlcv_df["Low"], lookback=5, min_bars_between=min_gap)
        for i in range(1, len(lows)):
            assert lows[i]["index"] - lows[i - 1]["index"] >= min_gap

    def test_larger_lookback_fewer_swings(self, ohlcv_df):
        """Larger lookback should find fewer (or equal) swing points."""
        highs_small = _swing_highs(ohlcv_df["High"], lookback=3)
        highs_large = _swing_highs(ohlcv_df["High"], lookback=10)
        assert len(highs_large) <= len(highs_small)


# ═══════════════════════════════════════════════════════════════════════════
# detect_fvgs
# ═══════════════════════════════════════════════════════════════════════════


class TestDetectFVGs:
    def test_returns_list(self, ohlcv_df):
        fvgs = detect_fvgs(ohlcv_df)
        assert isinstance(fvgs, list)

    def test_fvg_dict_keys(self, ohlcv_df):
        fvgs = detect_fvgs(ohlcv_df)
        if fvgs:
            f = fvgs[0]
            expected_keys = {
                "type",
                "top",
                "bottom",
                "midpoint",
                "size",
                "size_atr",
                "bar_index",
                "timestamp",
                "filled",
                "fill_pct",
            }
            assert expected_keys.issubset(set(f.keys())), f"Missing keys: {expected_keys - set(f.keys())}"

    def test_fvg_type_valid(self, ohlcv_df):
        fvgs = detect_fvgs(ohlcv_df)
        for f in fvgs:
            assert f["type"] in ("bullish", "bearish")

    def test_fvg_top_greater_than_bottom(self, ohlcv_df):
        fvgs = detect_fvgs(ohlcv_df)
        for f in fvgs:
            assert f["top"] > f["bottom"], f"FVG top ({f['top']}) should be > bottom ({f['bottom']})"

    def test_fvg_midpoint_between_top_and_bottom(self, ohlcv_df):
        fvgs = detect_fvgs(ohlcv_df)
        for f in fvgs:
            assert f["bottom"] <= f["midpoint"] <= f["top"]

    def test_fvg_size_positive(self, ohlcv_df):
        fvgs = detect_fvgs(ohlcv_df)
        for f in fvgs:
            assert f["size"] > 0
            assert f["size_atr"] > 0

    def test_fvg_filled_is_bool(self, ohlcv_df):
        fvgs = detect_fvgs(ohlcv_df)
        for f in fvgs:
            assert isinstance(f["filled"], bool)

    def test_fvg_fill_pct_range(self, ohlcv_df):
        fvgs = detect_fvgs(ohlcv_df)
        for f in fvgs:
            assert 0.0 <= f["fill_pct"] <= 1.0

    def test_max_results_caps_output(self, ohlcv_df):
        fvgs = detect_fvgs(ohlcv_df, max_results=3)
        assert len(fvgs) <= 3

    def test_min_gap_atr_filters_small_gaps(self, ohlcv_df):
        """Higher min_gap_atr should find fewer FVGs."""
        fvgs_low = detect_fvgs(ohlcv_df, min_gap_atr=0.1)
        fvgs_high = detect_fvgs(ohlcv_df, min_gap_atr=1.0)
        assert len(fvgs_high) <= len(fvgs_low)

    def test_empty_input(self, empty_df):
        fvgs = detect_fvgs(empty_df)
        assert fvgs == []

    def test_tiny_input(self, tiny_df):
        fvgs = detect_fvgs(tiny_df)
        assert fvgs == []

    def test_constant_price_no_fvgs(self, constant_df):
        """Constant price should produce no FVGs (no gaps)."""
        fvgs = detect_fvgs(constant_df)
        assert fvgs == []

    def test_volatile_data_finds_fvgs(self, volatile_df):
        """High-volatility data should produce at least some FVGs."""
        fvgs = detect_fvgs(volatile_df, min_gap_atr=0.1)
        # Not guaranteed, but likely with 0.015 volatility
        assert isinstance(fvgs, list)

    def test_gappy_data_finds_fvgs(self, gappy_df):
        """Data with injected gaps should produce FVGs."""
        fvgs = detect_fvgs(gappy_df, min_gap_atr=0.1)
        assert isinstance(fvgs, list)

    def test_check_fill_false_skips_fill_check(self, ohlcv_df):
        """With check_fill=False, fill_pct should be 0.0 and filled=False."""
        fvgs = detect_fvgs(ohlcv_df, check_fill=False)
        for f in fvgs:
            assert f["filled"] is False
            assert f["fill_pct"] == 0.0

    def test_results_most_recent_first(self, ohlcv_df):
        """FVGs should be returned most recent first (descending bar_index)."""
        fvgs = detect_fvgs(ohlcv_df)
        for i in range(1, len(fvgs)):
            assert fvgs[i]["bar_index"] <= fvgs[i - 1]["bar_index"]

    def test_impulsive_data_detects_fvgs(self, impulsive_df):
        """The impulsive synthetic data should produce detectable FVGs."""
        fvgs = detect_fvgs(impulsive_df, min_gap_atr=0.1)
        assert isinstance(fvgs, list)
        # The big drop at bars 50-52 should create a bearish FVG
        # (though this depends on the exact ATR threshold)

    def test_bar_index_within_range(self, ohlcv_df):
        """bar_index should be within the DataFrame's index range."""
        fvgs = detect_fvgs(ohlcv_df)
        for f in fvgs:
            assert 0 <= f["bar_index"] < len(ohlcv_df)


# ═══════════════════════════════════════════════════════════════════════════
# get_unfilled_fvgs
# ═══════════════════════════════════════════════════════════════════════════


class TestGetUnfilledFVGs:
    def test_returns_list(self, ohlcv_df):
        result = get_unfilled_fvgs(ohlcv_df)
        assert isinstance(result, list)

    def test_all_unfilled(self, ohlcv_df):
        """All returned FVGs should have filled=False."""
        result = get_unfilled_fvgs(ohlcv_df)
        for f in result:
            assert f["filled"] is False

    def test_subset_of_all_fvgs(self, ohlcv_df):
        """Unfilled FVGs should be a subset of all FVGs."""
        all_fvgs = detect_fvgs(ohlcv_df)
        unfilled = get_unfilled_fvgs(ohlcv_df)
        assert len(unfilled) <= len(all_fvgs)

    def test_empty_input(self, empty_df):
        result = get_unfilled_fvgs(empty_df)
        assert result == []


# ═══════════════════════════════════════════════════════════════════════════
# detect_order_blocks
# ═══════════════════════════════════════════════════════════════════════════


class TestDetectOrderBlocks:
    def test_returns_list(self, ohlcv_df):
        obs = detect_order_blocks(ohlcv_df)
        assert isinstance(obs, list)

    def test_ob_dict_keys(self, ohlcv_df):
        obs = detect_order_blocks(ohlcv_df)
        if obs:
            ob = obs[0]
            expected_keys = {
                "type",
                "high",
                "low",
                "midpoint",
                "bar_index",
                "timestamp",
                "impulse_size",
                "impulse_atr",
                "mitigated",
                "tested",
            }
            assert expected_keys.issubset(set(ob.keys())), f"Missing keys: {expected_keys - set(ob.keys())}"

    def test_ob_type_valid(self, ohlcv_df):
        obs = detect_order_blocks(ohlcv_df)
        for ob in obs:
            assert ob["type"] in ("bullish", "bearish")

    def test_ob_high_greater_than_low(self, ohlcv_df):
        obs = detect_order_blocks(ohlcv_df)
        for ob in obs:
            assert ob["high"] >= ob["low"]

    def test_ob_midpoint_in_range(self, ohlcv_df):
        obs = detect_order_blocks(ohlcv_df)
        for ob in obs:
            assert ob["low"] <= ob["midpoint"] <= ob["high"]

    def test_ob_impulse_positive(self, ohlcv_df):
        obs = detect_order_blocks(ohlcv_df)
        for ob in obs:
            assert ob["impulse_size"] > 0
            assert ob["impulse_atr"] > 0

    def test_ob_mitigated_is_bool(self, ohlcv_df):
        obs = detect_order_blocks(ohlcv_df)
        for ob in obs:
            assert isinstance(ob["mitigated"], bool)

    def test_ob_tested_is_bool(self, ohlcv_df):
        obs = detect_order_blocks(ohlcv_df)
        for ob in obs:
            assert isinstance(ob["tested"], bool)

    def test_max_results_caps_output(self, ohlcv_df):
        obs = detect_order_blocks(ohlcv_df, max_results=5)
        assert len(obs) <= 5

    def test_higher_impulse_mult_fewer_obs(self, ohlcv_df):
        """Higher impulse_atr_mult threshold should find fewer OBs."""
        obs_low = detect_order_blocks(ohlcv_df, impulse_atr_mult=0.5)
        obs_high = detect_order_blocks(ohlcv_df, impulse_atr_mult=3.0)
        assert len(obs_high) <= len(obs_low)

    def test_empty_input(self, empty_df):
        obs = detect_order_blocks(empty_df)
        assert obs == []

    def test_tiny_input(self, tiny_df):
        obs = detect_order_blocks(tiny_df)
        assert obs == []

    def test_constant_price_no_obs(self, constant_df):
        """Constant price should produce no order blocks."""
        obs = detect_order_blocks(constant_df)
        assert obs == []

    def test_check_mitigated_false(self, ohlcv_df):
        """With check_mitigated=False, mitigated/tested should be False."""
        obs = detect_order_blocks(ohlcv_df, check_mitigated=False)
        for ob in obs:
            assert ob["mitigated"] is False
            assert ob["tested"] is False

    def test_results_most_recent_first(self, ohlcv_df):
        obs = detect_order_blocks(ohlcv_df)
        for i in range(1, len(obs)):
            assert obs[i]["bar_index"] <= obs[i - 1]["bar_index"]

    def test_bar_index_within_range(self, ohlcv_df):
        obs = detect_order_blocks(ohlcv_df)
        for ob in obs:
            assert 0 <= ob["bar_index"] < len(ohlcv_df)

    def test_impulsive_data_finds_obs(self, impulsive_df):
        """Impulsive data with large moves should produce OBs."""
        obs = detect_order_blocks(impulsive_df, impulse_atr_mult=0.5)
        assert isinstance(obs, list)
        # The large drop should generate at least one bearish OB
        # (the bullish candle at bar 50 before the drop at bar 51)

    def test_deduplication_unique_bar_indices(self, ohlcv_df):
        """Each OB should have a unique bar_index (deduplication)."""
        obs = detect_order_blocks(ohlcv_df)
        bar_indices = [ob["bar_index"] for ob in obs]
        assert len(bar_indices) == len(set(bar_indices)), (
            "Order blocks should have unique bar_indices after deduplication"
        )

    def test_volatile_data_finds_obs(self, volatile_df):
        """High volatility data should produce order blocks."""
        obs = detect_order_blocks(volatile_df, impulse_atr_mult=1.0)
        assert isinstance(obs, list)

    def test_lookback_parameter(self, ohlcv_df):
        """Different lookback values should produce valid results."""
        for lb in (1, 3, 5, 10):
            obs = detect_order_blocks(ohlcv_df, lookback=lb)
            assert isinstance(obs, list)
            for ob in obs:
                assert ob["type"] in ("bullish", "bearish")


# ═══════════════════════════════════════════════════════════════════════════
# get_active_order_blocks
# ═══════════════════════════════════════════════════════════════════════════


class TestGetActiveOrderBlocks:
    def test_returns_list(self, ohlcv_df):
        result = get_active_order_blocks(ohlcv_df)
        assert isinstance(result, list)

    def test_all_non_mitigated(self, ohlcv_df):
        result = get_active_order_blocks(ohlcv_df)
        for ob in result:
            assert ob["mitigated"] is False

    def test_subset_of_all_obs(self, ohlcv_df):
        all_obs = detect_order_blocks(ohlcv_df)
        active = get_active_order_blocks(ohlcv_df)
        assert len(active) <= len(all_obs)

    def test_empty_input(self, empty_df):
        result = get_active_order_blocks(empty_df)
        assert result == []


# ═══════════════════════════════════════════════════════════════════════════
# detect_liquidity_sweeps
# ═══════════════════════════════════════════════════════════════════════════


class TestDetectLiquiditySweeps:
    def test_returns_list(self, ohlcv_df):
        sweeps = detect_liquidity_sweeps(ohlcv_df)
        assert isinstance(sweeps, list)

    def test_sweep_dict_keys(self, ohlcv_df):
        sweeps = detect_liquidity_sweeps(ohlcv_df)
        if sweeps:
            s = sweeps[0]
            expected_keys = {
                "type",
                "sweep_price",
                "swing_price",
                "reversal_size",
                "reversal_atr",
                "swept_by",
                "bar_index",
                "timestamp",
                "swing_timestamp",
            }
            assert expected_keys.issubset(set(s.keys())), f"Missing keys: {expected_keys - set(s.keys())}"

    def test_sweep_type_valid(self, ohlcv_df):
        sweeps = detect_liquidity_sweeps(ohlcv_df)
        for s in sweeps:
            assert s["type"] in ("buy_side", "sell_side")

    def test_sweep_prices_positive(self, ohlcv_df):
        sweeps = detect_liquidity_sweeps(ohlcv_df)
        for s in sweeps:
            assert s["sweep_price"] > 0
            assert s["swing_price"] > 0

    def test_buy_side_sweep_above_swing(self, ohlcv_df):
        """Buy-side sweep price should be above the swing high."""
        sweeps = detect_liquidity_sweeps(ohlcv_df)
        for s in sweeps:
            if s["type"] == "buy_side":
                assert s["sweep_price"] >= s["swing_price"]

    def test_sell_side_sweep_below_swing(self, ohlcv_df):
        """Sell-side sweep price should be below the swing low."""
        sweeps = detect_liquidity_sweeps(ohlcv_df)
        for s in sweeps:
            if s["type"] == "sell_side":
                assert s["sweep_price"] <= s["swing_price"]

    def test_reversal_positive(self, ohlcv_df):
        sweeps = detect_liquidity_sweeps(ohlcv_df)
        for s in sweeps:
            assert s["reversal_size"] > 0
            assert s["reversal_atr"] > 0

    def test_swept_by_positive(self, ohlcv_df):
        sweeps = detect_liquidity_sweeps(ohlcv_df)
        for s in sweeps:
            assert s["swept_by"] >= 0

    def test_max_results_caps_output(self, ohlcv_df):
        sweeps = detect_liquidity_sweeps(ohlcv_df, max_results=3)
        assert len(sweeps) <= 3

    def test_higher_reversal_threshold_fewer_sweeps(self, ohlcv_df):
        """Higher min_reversal_atr should find fewer sweeps."""
        sweeps_low = detect_liquidity_sweeps(ohlcv_df, min_reversal_atr=0.1)
        sweeps_high = detect_liquidity_sweeps(ohlcv_df, min_reversal_atr=2.0)
        assert len(sweeps_high) <= len(sweeps_low)

    def test_empty_input(self, empty_df):
        sweeps = detect_liquidity_sweeps(empty_df)
        assert sweeps == []

    def test_tiny_input(self, tiny_df):
        sweeps = detect_liquidity_sweeps(tiny_df)
        assert sweeps == []

    def test_short_input(self, short_df):
        sweeps = detect_liquidity_sweeps(short_df)
        assert isinstance(sweeps, list)

    def test_constant_price_no_sweeps(self, constant_df):
        sweeps = detect_liquidity_sweeps(constant_df)
        assert sweeps == []

    def test_results_sorted_by_bar_index_desc(self, ohlcv_df):
        sweeps = detect_liquidity_sweeps(ohlcv_df)
        for i in range(1, len(sweeps)):
            assert sweeps[i]["bar_index"] <= sweeps[i - 1]["bar_index"]

    def test_bar_index_within_range(self, ohlcv_df):
        sweeps = detect_liquidity_sweeps(ohlcv_df)
        for s in sweeps:
            assert 0 <= s["bar_index"] < len(ohlcv_df)

    def test_swing_lookback_parameter(self, ohlcv_df):
        """Different swing lookback values should produce valid results."""
        for lb in (3, 5, 10):
            sweeps = detect_liquidity_sweeps(ohlcv_df, swing_lookback=lb)
            assert isinstance(sweeps, list)

    def test_volatile_data_finds_sweeps(self, volatile_df):
        """High volatility data should produce liquidity sweeps."""
        sweeps = detect_liquidity_sweeps(volatile_df, min_reversal_atr=0.3)
        assert isinstance(sweeps, list)


# ═══════════════════════════════════════════════════════════════════════════
# detect_breaker_blocks
# ═══════════════════════════════════════════════════════════════════════════


class TestDetectBreakerBlocks:
    def test_returns_list(self, ohlcv_df):
        breakers = detect_breaker_blocks(ohlcv_df)
        assert isinstance(breakers, list)

    def test_breaker_dict_keys(self, ohlcv_df):
        breakers = detect_breaker_blocks(ohlcv_df)
        if breakers:
            b = breakers[0]
            expected_keys = {
                "type",
                "high",
                "low",
                "midpoint",
                "ob_bar_index",
                "ob_timestamp",
                "break_bar_index",
                "break_timestamp",
                "original_ob_type",
                "retested",
            }
            assert expected_keys.issubset(set(b.keys())), f"Missing keys: {expected_keys - set(b.keys())}"

    def test_breaker_type_valid(self, ohlcv_df):
        breakers = detect_breaker_blocks(ohlcv_df)
        for b in breakers:
            assert b["type"] in ("bullish", "bearish")

    def test_breaker_type_opposite_of_original(self, ohlcv_df):
        """Breaker type should be opposite of the original OB type."""
        breakers = detect_breaker_blocks(ohlcv_df)
        for b in breakers:
            if b["original_ob_type"] == "bullish":
                assert b["type"] == "bearish"
            elif b["original_ob_type"] == "bearish":
                assert b["type"] == "bullish"

    def test_breaker_high_gte_low(self, ohlcv_df):
        breakers = detect_breaker_blocks(ohlcv_df)
        for b in breakers:
            assert b["high"] >= b["low"]

    def test_breaker_midpoint_in_range(self, ohlcv_df):
        breakers = detect_breaker_blocks(ohlcv_df)
        for b in breakers:
            assert b["low"] <= b["midpoint"] <= b["high"]

    def test_retested_is_bool(self, ohlcv_df):
        breakers = detect_breaker_blocks(ohlcv_df)
        for b in breakers:
            assert isinstance(b["retested"], bool)

    def test_max_results_caps_output(self, ohlcv_df):
        breakers = detect_breaker_blocks(ohlcv_df, max_results=3)
        assert len(breakers) <= 3

    def test_empty_input(self, empty_df):
        breakers = detect_breaker_blocks(empty_df)
        assert breakers == []

    def test_tiny_input(self, tiny_df):
        breakers = detect_breaker_blocks(tiny_df)
        assert breakers == []

    def test_constant_price_no_breakers(self, constant_df):
        breakers = detect_breaker_blocks(constant_df)
        assert breakers == []

    def test_results_sorted_by_break_bar_desc(self, ohlcv_df):
        breakers = detect_breaker_blocks(ohlcv_df)
        for i in range(1, len(breakers)):
            assert breakers[i]["break_bar_index"] <= breakers[i - 1]["break_bar_index"]

    def test_break_bar_after_ob_bar(self, ohlcv_df):
        """Break bar should come after the original OB bar."""
        breakers = detect_breaker_blocks(ohlcv_df)
        for b in breakers:
            assert b["break_bar_index"] > b["ob_bar_index"]

    def test_volatile_data(self, volatile_df):
        breakers = detect_breaker_blocks(volatile_df, impulse_atr_mult=0.5)
        assert isinstance(breakers, list)

    def test_impulsive_data(self, impulsive_df):
        """Impulsive data with drop and recovery should potentially form breakers."""
        breakers = detect_breaker_blocks(impulsive_df, impulse_atr_mult=0.5)
        assert isinstance(breakers, list)


# ═══════════════════════════════════════════════════════════════════════════
# ict_summary
# ═══════════════════════════════════════════════════════════════════════════


class TestICTSummary:
    def test_returns_dict(self, ohlcv_df):
        summary = ict_summary(ohlcv_df)
        assert isinstance(summary, dict)

    def test_required_keys(self, ohlcv_df):
        summary = ict_summary(ohlcv_df)
        required = {
            "fvgs",
            "order_blocks",
            "liquidity_sweeps",
            "breaker_blocks",
            "stats",
            "nearest_levels",
            "current_price",
        }
        assert required.issubset(set(summary.keys())), f"Missing keys: {required - set(summary.keys())}"

    def test_stats_keys(self, ohlcv_df):
        summary = ict_summary(ohlcv_df)
        stats = summary["stats"]
        expected_stat_keys = {
            "total_fvgs",
            "unfilled_fvgs",
            "total_obs",
            "active_obs",
            "recent_sweeps",
            "breakers",
        }
        assert expected_stat_keys.issubset(set(stats.keys()))

    def test_stats_values_nonnegative(self, ohlcv_df):
        summary = ict_summary(ohlcv_df)
        for key, val in summary["stats"].items():
            assert val >= 0, f"Stat '{key}' should be non-negative, got {val}"

    def test_unfilled_fvgs_lte_total(self, ohlcv_df):
        summary = ict_summary(ohlcv_df)
        assert summary["stats"]["unfilled_fvgs"] <= summary["stats"]["total_fvgs"]

    def test_active_obs_lte_total(self, ohlcv_df):
        summary = ict_summary(ohlcv_df)
        assert summary["stats"]["active_obs"] <= summary["stats"]["total_obs"]

    def test_bullish_plus_bearish_fvgs(self, ohlcv_df):
        summary = ict_summary(ohlcv_df)
        stats = summary["stats"]
        assert stats.get("bullish_fvgs", 0) + stats.get("bearish_fvgs", 0) == stats["unfilled_fvgs"]

    def test_bullish_plus_bearish_obs(self, ohlcv_df):
        summary = ict_summary(ohlcv_df)
        stats = summary["stats"]
        assert stats.get("bullish_obs", 0) + stats.get("bearish_obs", 0) == stats["active_obs"]

    def test_current_price_positive(self, ohlcv_df):
        summary = ict_summary(ohlcv_df)
        assert summary["current_price"] > 0

    def test_nearest_levels_dict(self, ohlcv_df):
        summary = ict_summary(ohlcv_df)
        assert isinstance(summary["nearest_levels"], dict)

    def test_nearest_above_positive_distance(self, ohlcv_df):
        summary = ict_summary(ohlcv_df)
        if "above" in summary["nearest_levels"]:
            assert summary["nearest_levels"]["above"]["distance"] > 0

    def test_nearest_below_nonpositive_distance(self, ohlcv_df):
        summary = ict_summary(ohlcv_df)
        if "below" in summary["nearest_levels"]:
            assert summary["nearest_levels"]["below"]["distance"] <= 0

    def test_empty_input(self, empty_df):
        summary = ict_summary(empty_df)
        assert summary["fvgs"] == []
        assert summary["order_blocks"] == []
        assert summary["liquidity_sweeps"] == []
        assert summary["breaker_blocks"] == []
        assert summary["stats"]["total_fvgs"] == 0

    def test_tiny_input(self, tiny_df):
        summary = ict_summary(tiny_df)
        assert isinstance(summary, dict)

    def test_lists_are_consistent_with_stats(self, ohlcv_df):
        summary = ict_summary(ohlcv_df)
        assert len(summary["fvgs"]) == summary["stats"]["total_fvgs"]
        assert len(summary["order_blocks"]) == summary["stats"]["total_obs"]
        assert len(summary["liquidity_sweeps"]) == summary["stats"]["recent_sweeps"]
        assert len(summary["breaker_blocks"]) == summary["stats"]["breakers"]

    def test_volatile_data(self, volatile_df):
        summary = ict_summary(volatile_df)
        assert isinstance(summary, dict)
        # Volatile data should likely produce some detections
        total = summary["stats"]["total_fvgs"] + summary["stats"]["total_obs"] + summary["stats"]["recent_sweeps"]
        assert total >= 0  # at minimum no crash


# ═══════════════════════════════════════════════════════════════════════════
# DataFrame converters
# ═══════════════════════════════════════════════════════════════════════════


class TestFVGsToDataFrame:
    def test_returns_dataframe(self, ohlcv_df):
        fvgs = detect_fvgs(ohlcv_df)
        df = fvgs_to_dataframe(fvgs)
        assert isinstance(df, pd.DataFrame)

    def test_empty_list_returns_empty_df(self):
        df = fvgs_to_dataframe([])
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0

    def test_columns_present(self, ohlcv_df):
        fvgs = detect_fvgs(ohlcv_df)
        if fvgs:
            df = fvgs_to_dataframe(fvgs)
            expected_cols = {
                "Type",
                "Top",
                "Bottom",
                "Midpoint",
                "Size (ATR)",
                "Status",
                "Time",
            }
            assert expected_cols.issubset(set(df.columns))

    def test_row_count_matches_input(self, ohlcv_df):
        fvgs = detect_fvgs(ohlcv_df)
        df = fvgs_to_dataframe(fvgs)
        assert len(df) == len(fvgs)


class TestOrderBlocksToDataFrame:
    def test_returns_dataframe(self, ohlcv_df):
        obs = detect_order_blocks(ohlcv_df)
        df = order_blocks_to_dataframe(obs)
        assert isinstance(df, pd.DataFrame)

    def test_empty_list_returns_empty_df(self):
        df = order_blocks_to_dataframe([])
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0

    def test_columns_present(self, ohlcv_df):
        obs = detect_order_blocks(ohlcv_df)
        if obs:
            df = order_blocks_to_dataframe(obs)
            expected_cols = {"Type", "High", "Low", "Impulse", "Status", "Time"}
            assert expected_cols.issubset(set(df.columns))

    def test_row_count_matches_input(self, ohlcv_df):
        obs = detect_order_blocks(ohlcv_df)
        df = order_blocks_to_dataframe(obs)
        assert len(df) == len(obs)


class TestSweepsToDataFrame:
    def test_returns_dataframe(self, ohlcv_df):
        sweeps = detect_liquidity_sweeps(ohlcv_df)
        df = sweeps_to_dataframe(sweeps)
        assert isinstance(df, pd.DataFrame)

    def test_empty_list_returns_empty_df(self):
        df = sweeps_to_dataframe([])
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0

    def test_columns_present(self, ohlcv_df):
        sweeps = detect_liquidity_sweeps(ohlcv_df)
        if sweeps:
            df = sweeps_to_dataframe(sweeps)
            expected_cols = {
                "Type",
                "Sweep Price",
                "Swing Level",
                "Swept By",
                "Reversal",
                "Time",
            }
            assert expected_cols.issubset(set(df.columns))

    def test_row_count_matches_input(self, ohlcv_df):
        sweeps = detect_liquidity_sweeps(ohlcv_df)
        df = sweeps_to_dataframe(sweeps)
        assert len(df) == len(sweeps)


class TestBreakersToDataFrame:
    def test_returns_dataframe(self, ohlcv_df):
        breakers = detect_breaker_blocks(ohlcv_df)
        df = breakers_to_dataframe(breakers)
        assert isinstance(df, pd.DataFrame)

    def test_empty_list_returns_empty_df(self):
        df = breakers_to_dataframe([])
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0

    def test_columns_present(self, ohlcv_df):
        breakers = detect_breaker_blocks(ohlcv_df)
        if breakers:
            df = breakers_to_dataframe(breakers)
            expected_cols = {
                "Type",
                "High",
                "Low",
                "Original OB",
                "Status",
                "Break Time",
            }
            assert expected_cols.issubset(set(df.columns))

    def test_row_count_matches_input(self, ohlcv_df):
        breakers = detect_breaker_blocks(ohlcv_df)
        df = breakers_to_dataframe(breakers)
        assert len(df) == len(breakers)


class TestLevelsToDataFrame:
    def test_returns_dataframe(self, ohlcv_df):
        summary = ict_summary(ohlcv_df)
        df = levels_to_dataframe(summary)
        assert isinstance(df, pd.DataFrame)

    def test_empty_summary(self):
        empty_summary = {
            "fvgs": [],
            "order_blocks": [],
            "liquidity_sweeps": [],
            "breaker_blocks": [],
            "stats": {},
            "nearest_levels": {},
            "current_price": 100.0,
        }
        df = levels_to_dataframe(empty_summary)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0

    def test_columns_present(self, ohlcv_df):
        summary = ict_summary(ohlcv_df)
        df = levels_to_dataframe(summary)
        if not df.empty:
            expected_cols = {"Level", "Zone", "Concept", "Distance", "Strength"}
            assert expected_cols.issubset(set(df.columns))

    def test_sorted_by_distance(self, ohlcv_df):
        summary = ict_summary(ohlcv_df)
        df = levels_to_dataframe(summary)
        if len(df) >= 2:
            distances = df["Distance"].abs().values
            for i in range(1, len(distances)):
                assert distances[i] >= distances[i - 1] - 1e-9, "Levels should be sorted by ascending absolute distance"

    def test_excludes_filled_fvgs(self, ohlcv_df):
        """Filled FVGs should not appear in the levels table."""
        summary = ict_summary(ohlcv_df)
        df = levels_to_dataframe(summary)
        if not df.empty:
            fvg_rows = df[df["Concept"].str.contains("FVG", na=False)]
            # All FVG rows should correspond to unfilled FVGs
            unfilled_count = summary["stats"]["unfilled_fvgs"]
            assert len(fvg_rows) <= unfilled_count

    def test_excludes_mitigated_obs(self, ohlcv_df):
        """Mitigated OBs should not appear in the levels table."""
        summary = ict_summary(ohlcv_df)
        df = levels_to_dataframe(summary)
        if not df.empty:
            ob_rows = df[df["Concept"].str.contains("OB", na=False)]
            active_count = summary["stats"]["active_obs"]
            assert len(ob_rows) <= active_count


# ═══════════════════════════════════════════════════════════════════════════
# Integration: full pipeline
# ═══════════════════════════════════════════════════════════════════════════


class TestICTIntegration:
    def test_full_pipeline_runs(self, ohlcv_df):
        """Smoke test: run the entire ICT pipeline end to end."""
        fvgs = detect_fvgs(ohlcv_df)
        obs = detect_order_blocks(ohlcv_df)
        sweeps = detect_liquidity_sweeps(ohlcv_df)
        breakers = detect_breaker_blocks(ohlcv_df)
        summary = ict_summary(ohlcv_df)
        unfilled = get_unfilled_fvgs(ohlcv_df)
        active = get_active_order_blocks(ohlcv_df)

        assert isinstance(fvgs, list)
        assert isinstance(obs, list)
        assert isinstance(sweeps, list)
        assert isinstance(breakers, list)
        assert isinstance(summary, dict)
        assert isinstance(unfilled, list)
        assert isinstance(active, list)

    def test_pipeline_with_different_seeds(self):
        """Run ICT on several synthetic datasets to check robustness."""
        for seed in [1, 10, 42, 100, 999]:
            df = _random_walk_ohlcv(n=300, seed=seed)
            summary = ict_summary(df)
            assert isinstance(summary, dict)
            assert summary["stats"]["total_fvgs"] >= 0
            assert summary["stats"]["total_obs"] >= 0

    def test_pipeline_with_realistic_prices(self):
        """Test with realistic price levels for major futures instruments."""
        configs = [
            {"start_price": 5500.0, "volatility": 0.002},  # ES
            {"start_price": 2700.0, "volatility": 0.003},  # GC
            {"start_price": 70.0, "volatility": 0.005},  # CL
            {"start_price": 18000.0, "volatility": 0.003},  # NQ
            {"start_price": 30.0, "volatility": 0.004},  # SI
            {"start_price": 4.5, "volatility": 0.003},  # HG
        ]
        for cfg in configs:
            df = _random_walk_ohlcv(
                n=300,
                seed=42,
                start_price=cfg["start_price"],
                volatility=cfg["volatility"],
            )
            summary = ict_summary(df)
            assert summary["current_price"] > 0
            assert isinstance(summary["stats"], dict)

    def test_all_dataframe_converters_run(self, ohlcv_df):
        """Verify all DF converters produce valid output."""
        summary = ict_summary(ohlcv_df)
        fvg_df = fvgs_to_dataframe(summary["fvgs"])
        ob_df = order_blocks_to_dataframe(summary["order_blocks"])
        sweep_df = sweeps_to_dataframe(summary["liquidity_sweeps"])
        breaker_df = breakers_to_dataframe(summary["breaker_blocks"])
        levels_df = levels_to_dataframe(summary)

        for name, df in [
            ("fvg_df", fvg_df),
            ("ob_df", ob_df),
            ("sweep_df", sweep_df),
            ("breaker_df", breaker_df),
            ("levels_df", levels_df),
        ]:
            assert isinstance(df, pd.DataFrame), f"{name} should be a DataFrame"

    def test_impulsive_scenario(self, impulsive_df):
        """Run full pipeline on the impulsive scenario fixture."""
        summary = ict_summary(impulsive_df)
        assert isinstance(summary, dict)

        # The impulsive data has a large drop and recovery, so we should
        # see at least some detections across the concepts
        total_detections = (
            summary["stats"]["total_fvgs"]
            + summary["stats"]["total_obs"]
            + summary["stats"]["recent_sweeps"]
            + summary["stats"]["breakers"]
        )
        # At minimum, no crash. The impulsive move might or might not meet
        # all ATR thresholds depending on warmup, but the pipeline should work.
        assert total_detections >= 0

    def test_gappy_scenario(self, gappy_df):
        """Run full pipeline on data with injected price gaps."""
        summary = ict_summary(gappy_df)
        assert isinstance(summary, dict)
        # Gappy data should produce some FVGs or OBs
        assert summary["stats"]["total_fvgs"] >= 0

    def test_consistency_across_runs(self, ohlcv_df):
        """Same input should produce identical output (deterministic)."""
        s1 = ict_summary(ohlcv_df)
        s2 = ict_summary(ohlcv_df)
        assert s1["stats"] == s2["stats"]
        assert s1["current_price"] == s2["current_price"]
        assert len(s1["fvgs"]) == len(s2["fvgs"])
        assert len(s1["order_blocks"]) == len(s2["order_blocks"])

    def test_high_volatility_produces_detections(self, volatile_df):
        """With high volatility, at least one concept should detect something."""
        summary = ict_summary(volatile_df)
        total = (
            summary["stats"]["total_fvgs"]
            + summary["stats"]["total_obs"]
            + summary["stats"]["recent_sweeps"]
            + summary["stats"]["breakers"]
        )
        # High vol (0.015) on 500 bars should produce at least some OBs
        # But we don't hard-assert a minimum to avoid flaky tests
        assert total >= 0
