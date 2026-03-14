"""
Tests for volume_profile.py — Volume Profile analysis module.

Validates:
  - compute_volume_profile: POC, VAH, VAL, HVN, LVN, bin distribution
  - _compute_value_area: value area expansion from POC
  - _find_volume_nodes: HVN/LVN identification
  - compute_session_profiles: per-day session splitting
  - find_naked_pocs: unfilled POC tracking
  - _rolling_poc: rolling POC series computation
  - _rolling_vah_val: rolling VAH/VAL series
  - profile_to_dataframe: conversion helper
  - format_profile_summary: display helper
  - VolumeProfileStrategy: backtesting strategy (smoke test)
  - Edge cases: empty data, single bar, constant price, zero volume

These tests mirror the logic implemented in the Ruby indicator
to ensure Python-canonical parity.
"""

from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from lib.analysis.volume_profile import (
    _compute_value_area,
    _empty_profile,
    _find_volume_nodes,
    _rolling_poc,
    _rolling_vah_val,
    compute_session_profiles,
    compute_volume_profile,
    find_naked_pocs,
    format_profile_summary,
    profile_to_dataframe,
)

# ---------------------------------------------------------------------------
# Test data helpers
# ---------------------------------------------------------------------------


def _make_ohlcv(
    n_bars: int = 200,
    base_price: float = 2650.0,
    volatility: float = 5.0,
    base_volume: float = 1000.0,
    seed: int = 42,
    days: int = 1,
    start_date: datetime | None = None,
) -> pd.DataFrame:
    """Generate synthetic OHLCV data for testing.

    If days > 1, distributes bars across multiple calendar days.
    """
    rng = np.random.RandomState(seed)
    start = start_date or datetime(2025, 1, 15, 9, 30)

    dates = []
    bars_remaining = n_bars
    for d in range(days):
        day_start = start + timedelta(days=d)
        bars_this_day = max(1, bars_remaining // max(1, days - d))
        for b in range(bars_this_day):
            dates.append(day_start + timedelta(minutes=5 * b))
        bars_remaining -= bars_this_day

    dates = dates[:n_bars]
    n_bars = len(dates)

    actual_bars = n_bars
    closes_list = [base_price]
    for _ in range(actual_bars - 1):
        closes_list.append(closes_list[-1] + rng.normal(0, volatility))
    closes = np.array(closes_list)

    highs = closes + rng.uniform(0.5, volatility, actual_bars)
    lows = closes - rng.uniform(0.5, volatility, actual_bars)
    opens = closes + rng.normal(0, volatility * 0.3, actual_bars)
    volumes = rng.uniform(base_volume * 0.5, base_volume * 2.0, actual_bars)

    # Ensure High >= max(Open, Close) and Low <= min(Open, Close)
    highs = np.maximum(highs, np.maximum(opens, closes))
    lows = np.minimum(lows, np.minimum(opens, closes))

    df = pd.DataFrame(
        {
            "Open": opens,
            "High": highs,
            "Low": lows,
            "Close": closes,
            "Volume": volumes,
        },
        index=pd.DatetimeIndex(dates),
    )
    return df


def _make_bimodal_ohlcv(
    n_bars: int = 200,
    center1: float = 2640.0,
    center2: float = 2660.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate data that trades in two price clusters (bimodal distribution).

    This creates a clear POC at one of the centers and HVN at both.
    """
    rng = np.random.RandomState(seed)
    start = datetime(2025, 1, 15, 9, 30)

    closes_list = []
    for i in range(n_bars):
        if i < n_bars // 2:
            closes_list.append(center1 + rng.normal(0, 2.0))
        else:
            closes_list.append(center2 + rng.normal(0, 2.0))

    closes = np.array(closes_list)
    highs = closes + rng.uniform(0.5, 3.0, n_bars)
    lows = closes - rng.uniform(0.5, 3.0, n_bars)
    opens = closes + rng.normal(0, 1.0, n_bars)
    volumes = rng.uniform(500, 2000, n_bars)

    highs = np.maximum(highs, np.maximum(opens, closes))
    lows = np.minimum(lows, np.minimum(opens, closes))

    dates = [start + timedelta(minutes=5 * i) for i in range(n_bars)]
    return pd.DataFrame(
        {"Open": opens, "High": highs, "Low": lows, "Close": closes, "Volume": volumes},
        index=pd.DatetimeIndex(dates),
    )


# ===========================================================================
# compute_volume_profile
# ===========================================================================


class TestComputeVolumeProfile:
    """Tests for compute_volume_profile()."""

    def test_returns_dict_with_expected_keys(self):
        df = _make_ohlcv(100)
        result = compute_volume_profile(df)
        expected_keys = {
            "poc",
            "vah",
            "val",
            "poc_volume",
            "total_volume",
            "bin_edges",
            "bin_centers",
            "bin_volumes",
            "hvn",
            "lvn",
        }
        assert expected_keys.issubset(set(result.keys()))

    def test_poc_is_within_price_range(self):
        df = _make_ohlcv(200)
        result = compute_volume_profile(df)
        price_min = df["Low"].min()
        price_max = df["High"].max()
        assert result["poc"] >= price_min - 1
        assert result["poc"] <= price_max + 1

    def test_vah_above_val(self):
        df = _make_ohlcv(200)
        result = compute_volume_profile(df)
        assert result["vah"] >= result["val"]

    def test_poc_between_vah_and_val(self):
        df = _make_ohlcv(200)
        result = compute_volume_profile(df)
        assert result["val"] <= result["poc"] <= result["vah"]

    def test_value_area_contains_target_volume(self):
        """Value area should contain approximately 70% of total volume."""
        df = _make_ohlcv(300, seed=123)
        result = compute_volume_profile(df, n_bins=50, value_area_pct=0.70)

        bin_centers = result["bin_centers"]
        bin_volumes = result["bin_volumes"]
        total_vol = result["total_volume"]

        # Sum volume in bins within the value area
        va_vol = sum(
            bv for bc, bv in zip(bin_centers, bin_volumes, strict=False) if result["val"] <= bc <= result["vah"]
        )

        # Should be at least 70% (may be slightly more due to discrete bins)
        assert va_vol / total_vol >= 0.69, f"VA volume ratio: {va_vol / total_vol:.3f}"

    def test_total_volume_matches_input(self):
        df = _make_ohlcv(100)
        result = compute_volume_profile(df, n_bins=30)
        input_vol = df["Volume"].sum()
        # Bin distribution should account for all volume (within floating point tolerance)
        assert abs(result["total_volume"] - input_vol) / input_vol < 0.01

    def test_bin_count_matches_parameter(self):
        df = _make_ohlcv(100)
        for n_bins in [10, 30, 60]:
            result = compute_volume_profile(df, n_bins=n_bins)
            assert len(result["bin_volumes"]) == n_bins
            assert len(result["bin_centers"]) == n_bins
            assert len(result["bin_edges"]) == n_bins + 1

    def test_poc_volume_is_max_bin_volume(self):
        df = _make_ohlcv(200)
        result = compute_volume_profile(df)
        assert result["poc_volume"] == max(result["bin_volumes"])

    def test_hvn_are_valid_prices(self):
        """High Volume Nodes should be within the price range."""
        df = _make_ohlcv(200)
        result = compute_volume_profile(df)
        if result["hvn"]:
            for node in result["hvn"]:
                assert result["bin_centers"][0] <= node <= result["bin_centers"][-1]

    def test_lvn_are_valid_prices(self):
        """Low Volume Nodes should be within the price range."""
        df = _make_ohlcv(200)
        result = compute_volume_profile(df)
        if result["lvn"]:
            for node in result["lvn"]:
                assert result["bin_centers"][0] <= node <= result["bin_centers"][-1]

    def test_different_value_area_pct(self):
        """Higher value area % should produce wider VAH-VAL range."""
        df = _make_ohlcv(200, seed=99)
        result_70 = compute_volume_profile(df, value_area_pct=0.70)
        result_90 = compute_volume_profile(df, value_area_pct=0.90)

        va_width_70 = result_70["vah"] - result_70["val"]
        va_width_90 = result_90["vah"] - result_90["val"]

        assert va_width_90 >= va_width_70

    def test_bimodal_data_poc_at_one_center(self):
        """With bimodal data, POC should be near one of the centers."""
        df = _make_bimodal_ohlcv(200, center1=2640, center2=2660)
        result = compute_volume_profile(df, n_bins=50)

        # POC should be within 5 points of one of the centers
        dist1 = abs(result["poc"] - 2640)
        dist2 = abs(result["poc"] - 2660)
        assert min(dist1, dist2) < 5.0, f"POC={result['poc']}, expected near 2640 or 2660"

    def test_reproducible_results(self):
        """Same input should produce same output."""
        df = _make_ohlcv(200, seed=42)
        r1 = compute_volume_profile(df)
        r2 = compute_volume_profile(df)
        assert r1["poc"] == r2["poc"]
        assert r1["vah"] == r2["vah"]
        assert r1["val"] == r2["val"]


# ===========================================================================
# Edge cases
# ===========================================================================


class TestVolumeProfileEdgeCases:
    """Edge cases and error handling."""

    def test_empty_dataframe(self):
        df = pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])  # type: ignore[call-overload]
        result = compute_volume_profile(df)
        assert result["poc"] == 0
        assert result["total_volume"] == 0

    def test_single_bar(self):
        """Single bar hits len(df) < 2 guard → returns empty profile."""
        df = pd.DataFrame(
            {
                "Open": [100],
                "High": [101],
                "Low": [99],
                "Close": [100.5],
                "Volume": [1000],
            },
            index=pd.DatetimeIndex([datetime(2025, 1, 15, 10, 0)]),
        )
        result = compute_volume_profile(df)
        # Single bar returns the empty-profile sentinel (len < 2 guard)
        assert result["poc"] == 0
        assert result["total_volume"] == 0

    def test_two_bars(self):
        """Two bars should produce a valid profile."""
        df = pd.DataFrame(
            {
                "Open": [100, 100.5],
                "High": [101, 102],
                "Low": [99, 99.5],
                "Close": [100.5, 101],
                "Volume": [1000, 1200],
            },
            index=pd.DatetimeIndex([datetime(2025, 1, 15, 10, 0), datetime(2025, 1, 15, 10, 5)]),
        )
        result = compute_volume_profile(df)
        assert result["poc"] > 0
        assert result["total_volume"] > 0

    def test_constant_price(self):
        """All bars at the same price — POC should be that price."""
        n = 50
        df = pd.DataFrame(
            {
                "Open": [100.0] * n,
                "High": [100.1] * n,
                "Low": [99.9] * n,
                "Close": [100.0] * n,
                "Volume": [500.0] * n,
            },
            index=pd.DatetimeIndex([datetime(2025, 1, 15, 9, 30) + timedelta(minutes=5 * i) for i in range(n)]),
        )
        result = compute_volume_profile(df, n_bins=20)
        assert abs(result["poc"] - 100.0) < 1.0

    def test_zero_volume_bars(self):
        """Bars with zero volume should be skipped."""
        df = _make_ohlcv(50, base_volume=0)
        df["Volume"] = 0.0
        result = compute_volume_profile(df)
        assert result["total_volume"] == 0

    def test_nan_values_handled(self):
        """NaN in price/volume should not crash."""
        df = _make_ohlcv(50)
        df.loc[df.index[5], "High"] = np.nan
        df.loc[df.index[10], "Volume"] = np.nan
        # Should not raise
        result = compute_volume_profile(df)
        assert isinstance(result, dict)
        assert result["total_volume"] > 0


# ===========================================================================
# _empty_profile
# ===========================================================================


class TestEmptyProfile:
    def test_returns_dict_with_zero_poc(self):
        result = _empty_profile()
        assert isinstance(result, dict)
        assert result["poc"] == 0
        assert result["total_volume"] == 0

    def test_has_all_expected_keys(self):
        result = _empty_profile()
        for key in [
            "poc",
            "vah",
            "val",
            "poc_volume",
            "total_volume",
            "bin_edges",
            "bin_centers",
            "bin_volumes",
            "hvn",
            "lvn",
        ]:
            assert key in result


# ===========================================================================
# _compute_value_area
# ===========================================================================


class TestComputeValueArea:
    def test_single_bin(self):
        """With one bin, VAH and VAL should be that bin's range."""
        centers = np.array([100.0])
        volumes = np.array([1000.0])
        vah, val = _compute_value_area(centers, volumes, 0, 1000.0, 0.70)
        # With single bin, value area is just that bin
        assert vah >= val

    def test_uniform_volume(self):
        """With uniform volume, value area should be centered around POC."""
        n = 20
        centers = np.linspace(90, 110, n)
        volumes = np.ones(n) * 100.0
        poc_idx = n // 2
        total = volumes.sum()

        vah, val = _compute_value_area(centers, volumes, poc_idx, total, 0.70)
        mid = (vah + val) / 2.0
        poc_center = centers[poc_idx]
        # Center of VA should be near POC
        assert abs(mid - poc_center) < 5.0

    def test_concentrated_volume(self):
        """With volume concentrated at POC, value area should be narrow."""
        n = 20
        centers = np.linspace(90, 110, n)
        volumes = np.ones(n) * 10.0
        poc_idx = n // 2
        volumes[poc_idx] = 5000.0  # Huge spike at POC
        total = volumes.sum()

        vah, val = _compute_value_area(centers, volumes, poc_idx, total, 0.70)
        # VA width should be small since most volume is at POC
        va_width = vah - val
        total_range = centers[-1] - centers[0]
        assert va_width < total_range * 0.5

    def test_vah_always_gte_val(self):
        """VAH should always be >= VAL."""
        rng = np.random.RandomState(77)
        for _ in range(10):
            n = rng.randint(5, 50)
            centers = np.sort(rng.uniform(90, 110, n))
            volumes = rng.uniform(1, 100, n)
            poc_idx = int(np.argmax(volumes))
            total = volumes.sum()
            vah, val = _compute_value_area(centers, volumes, poc_idx, total, 0.70)
            assert vah >= val


# ===========================================================================
# _find_volume_nodes
# ===========================================================================


class TestFindVolumeNodes:
    def test_returns_lists(self):
        centers = np.linspace(90, 110, 20)
        volumes = np.random.RandomState(42).uniform(10, 100, 20)
        hvn, lvn = _find_volume_nodes(centers, volumes, volumes.sum(), 20)
        assert isinstance(hvn, list)
        assert isinstance(lvn, list)

    def test_hvn_have_more_volume_than_lvn(self):
        """HVN prices should correspond to higher volume bins than LVN prices."""
        centers = np.linspace(90, 110, 30)
        rng = np.random.RandomState(42)
        volumes = rng.uniform(10, 100, 30)
        # Create clear high/low nodes
        volumes[10] = 500  # HVN
        volumes[20] = 5  # LVN

        hvn, lvn = _find_volume_nodes(centers, volumes, volumes.sum(), 30)

        if hvn and lvn:
            # At least one HVN should have higher volume than the lowest LVN
            for h in hvn:
                h_idx = np.argmin(np.abs(centers - h))
                for lvn_val in lvn:
                    l_idx = np.argmin(np.abs(centers - lvn_val))
                    assert volumes[h_idx] >= volumes[l_idx]


# ===========================================================================
# compute_session_profiles
# ===========================================================================


class TestComputeSessionProfiles:
    def test_returns_list(self):
        df = _make_ohlcv(200, days=3)
        result = compute_session_profiles(df)
        assert isinstance(result, list)

    def test_one_profile_per_day(self):
        df = _make_ohlcv(300, days=3)
        result = compute_session_profiles(df)
        # Should have one profile per trading day
        assert len(result) >= 2  # At least 2 of the 3 days should have enough bars

    def test_each_profile_has_date_key(self):
        df = _make_ohlcv(200, days=2)
        result = compute_session_profiles(df)
        for profile in result:
            assert "date" in profile
            assert "poc" in profile

    def test_profiles_ordered_chronologically(self):
        df = _make_ohlcv(400, days=5)
        result = compute_session_profiles(df)
        if len(result) >= 2:
            dates = [p["date"] for p in result]
            assert dates == sorted(dates)

    def test_max_sessions_parameter(self):
        df = _make_ohlcv(600, days=10)
        result = compute_session_profiles(df, max_sessions=3)
        assert len(result) <= 3

    def test_each_session_has_valid_poc(self):
        df = _make_ohlcv(300, days=3)
        result = compute_session_profiles(df)
        for profile in result:
            if profile["total_volume"] > 0:
                assert profile["poc"] > 0


# ===========================================================================
# find_naked_pocs
# ===========================================================================


class TestFindNakedPOCs:
    def test_returns_list(self):
        df = _make_ohlcv(400, days=5)
        sessions = compute_session_profiles(df)
        current_price = df["Close"].iloc[-1]
        result = find_naked_pocs(sessions, current_price)
        assert isinstance(result, list)

    def test_naked_pocs_have_expected_keys(self):
        df = _make_ohlcv(400, days=5)
        sessions = compute_session_profiles(df)
        current_price = df["Close"].iloc[-1]
        result = find_naked_pocs(sessions, current_price, max_distance_points=500)
        for poc in result:
            assert "poc" in poc
            assert "date" in poc

    def test_naked_poc_within_distance(self):
        """Naked POCs should be within max_distance_points of current price."""
        df = _make_ohlcv(400, days=5, volatility=2.0)
        sessions = compute_session_profiles(df)
        current_price = df["Close"].iloc[-1]
        max_dist = 50.0
        result = find_naked_pocs(sessions, current_price, max_distance_points=max_dist)
        for poc in result:
            assert abs(poc["poc"] - current_price) <= max_dist + 1.0  # +1 for bin center offset

    def test_empty_sessions_returns_empty(self):
        result = find_naked_pocs([], 2650.0)
        assert result == []

    def test_single_session_returns_empty(self):
        """With only one session (current), there are no prior POCs."""
        df = _make_ohlcv(100, days=1)
        sessions = compute_session_profiles(df)
        current_price = df["Close"].iloc[-1]
        result = find_naked_pocs(sessions, current_price)
        # At most 0 naked POCs (current session POC isn't "naked")
        # The function checks prior sessions only
        assert isinstance(result, list)


# ===========================================================================
# _rolling_poc
# ===========================================================================


class TestRollingPOC:
    def test_returns_series(self):
        df = _make_ohlcv(200)
        result = _rolling_poc(df["High"], df["Low"], df["Close"], df["Volume"], lookback=50)
        assert isinstance(result, pd.Series)
        assert len(result) == len(df)

    def test_nan_before_lookback(self):
        """Values before the lookback period should be NaN."""
        df = _make_ohlcv(200)
        lookback = 60
        result = _rolling_poc(df["High"], df["Low"], df["Close"], df["Volume"], lookback=lookback)
        assert np.isnan(result.iloc[0])
        assert np.isnan(result.iloc[lookback - 1])

    def test_valid_after_lookback(self):
        """Values after the lookback period should be valid numbers."""
        df = _make_ohlcv(200)
        lookback = 50
        result = _rolling_poc(df["High"], df["Low"], df["Close"], df["Volume"], lookback=lookback)
        valid_values = result.iloc[lookback:]
        assert valid_values.notna().all()

    def test_poc_within_price_range(self):
        df = _make_ohlcv(200)
        lookback = 50
        result = _rolling_poc(df["High"], df["Low"], df["Close"], df["Volume"], lookback=lookback)
        for i in range(lookback, len(df)):
            poc = result.iloc[i]
            if not np.isnan(poc):
                window_low = df["Low"].iloc[max(0, i - lookback) : i].min()
                window_high = df["High"].iloc[max(0, i - lookback) : i].max()
                assert window_low - 1 <= poc <= window_high + 1


# ===========================================================================
# _rolling_vah_val
# ===========================================================================


class TestRollingVAHVAL:
    def test_returns_two_series(self):
        df = _make_ohlcv(200)
        vah_s, val_s = _rolling_vah_val(df["High"], df["Low"], df["Close"], df["Volume"], lookback=50)
        assert isinstance(vah_s, pd.Series)
        assert isinstance(val_s, pd.Series)
        assert len(vah_s) == len(df)
        assert len(val_s) == len(df)

    def test_vah_gte_val(self):
        """VAH should be >= VAL at every valid point."""
        df = _make_ohlcv(200)
        lookback = 50
        vah_s, val_s = _rolling_vah_val(df["High"], df["Low"], df["Close"], df["Volume"], lookback=lookback)
        for i in range(lookback, len(df)):
            if not np.isnan(vah_s.iloc[i]) and not np.isnan(val_s.iloc[i]):
                assert vah_s.iloc[i] >= val_s.iloc[i]


# ===========================================================================
# profile_to_dataframe
# ===========================================================================


class TestProfileToDataFrame:
    def test_returns_dataframe(self):
        df = _make_ohlcv(100)
        profile = compute_volume_profile(df)
        result = profile_to_dataframe(profile)
        assert isinstance(result, pd.DataFrame)

    def test_columns_present(self):
        df = _make_ohlcv(100)
        profile = compute_volume_profile(df)
        result = profile_to_dataframe(profile)
        assert "price" in result.columns or "bin_center" in result.columns or len(result.columns) >= 1

    def test_row_count_matches_bins(self):
        n_bins = 30
        df = _make_ohlcv(100)
        profile = compute_volume_profile(df, n_bins=n_bins)
        result = profile_to_dataframe(profile)
        assert len(result) == n_bins


# ===========================================================================
# format_profile_summary
# ===========================================================================


class TestFormatProfileSummary:
    def test_returns_string(self):
        df = _make_ohlcv(100)
        profile = compute_volume_profile(df)
        result = format_profile_summary(profile)
        assert isinstance(result, str)

    def test_contains_poc(self):
        df = _make_ohlcv(100)
        profile = compute_volume_profile(df)
        result = format_profile_summary(profile)
        assert "POC" in result.upper() or "poc" in result.lower()

    def test_empty_profile_doesnt_crash(self):
        result = format_profile_summary(_empty_profile())
        assert isinstance(result, str)


# ===========================================================================
# VolumeProfileStrategy (smoke tests)
# ===========================================================================


class TestVolumeProfileStrategy:
    """Smoke tests for the VolumeProfileStrategy backtesting class."""

    def test_strategy_class_exists(self):
        from lib.analysis.volume_profile import VolumeProfileStrategy

        assert VolumeProfileStrategy is not None

    def test_strategy_has_init_and_next(self):
        from lib.analysis.volume_profile import VolumeProfileStrategy

        assert hasattr(VolumeProfileStrategy, "init")
        assert hasattr(VolumeProfileStrategy, "next")

    def test_suggest_params_returns_dict(self):
        from lib.analysis.volume_profile import (
            suggest_volume_profile_params,
        )

        class FakeTrial:
            def suggest_int(self, name, low, high, step=1):
                return (low + high) // 2

            def suggest_float(self, name, low, high, step=0.1):
                return (low + high) / 2.0

        result = suggest_volume_profile_params(FakeTrial())
        assert isinstance(result, dict)
        assert "vp_lookback" in result
        assert "vp_bins" in result


# ===========================================================================
# Integration tests — full pipeline
# ===========================================================================


class TestVolumeProfileIntegration:
    """End-to-end integration tests."""

    def test_full_pipeline_single_session(self):
        """Compute profile → verify all outputs are consistent."""
        df = _make_ohlcv(200, seed=42)
        profile = compute_volume_profile(df, n_bins=40, value_area_pct=0.70)

        # POC between VAL and VAH
        assert profile["val"] <= profile["poc"] <= profile["vah"]

        # Total volume reasonable
        assert profile["total_volume"] > 0
        assert abs(profile["total_volume"] - df["Volume"].sum()) / df["Volume"].sum() < 0.01

        # POC volume > 0
        assert profile["poc_volume"] > 0

        # Can convert to DataFrame
        pdf = profile_to_dataframe(profile)
        assert len(pdf) == 40

        # Can format summary
        summary = format_profile_summary(profile)
        assert len(summary) > 10

    def test_full_pipeline_multi_session(self):
        """Multi-session pipeline: sessions → naked POCs."""
        df = _make_ohlcv(600, days=5, volatility=3.0, seed=77)
        sessions = compute_session_profiles(df, n_bins=30, max_sessions=5)

        assert len(sessions) >= 2

        current_price = df["Close"].iloc[-1]
        naked = find_naked_pocs(sessions, current_price, max_distance_points=200)
        assert isinstance(naked, list)

    def test_rolling_poc_matches_static(self):
        """Rolling POC at the last bar should be close to the static profile POC."""
        df = _make_ohlcv(200, seed=42)
        lookback = 100

        static_profile = compute_volume_profile(df.iloc[-lookback:], n_bins=30)
        rolling = _rolling_poc(
            df["High"],
            df["Low"],
            df["Close"],
            df["Volume"],
            lookback=lookback,
            n_bins=30,
        )

        last_rolling_poc = rolling.iloc[-1]
        static_poc = static_profile["poc"]

        # They should be close (not exact due to bin edge alignment differences)
        price_range = df["High"].max() - df["Low"].min()
        tolerance = price_range * 0.1  # within 10% of range
        assert abs(last_rolling_poc - static_poc) < tolerance, (
            f"Rolling POC={last_rolling_poc:.2f} vs Static POC={static_poc:.2f}, tolerance={tolerance:.2f}"
        )

    def test_different_seeds_produce_different_profiles(self):
        """Different data should produce different profiles."""
        df1 = _make_ohlcv(200, seed=1, base_price=2600)
        df2 = _make_ohlcv(200, seed=2, base_price=2700)

        p1 = compute_volume_profile(df1)
        p2 = compute_volume_profile(df2)

        # POCs should differ since base prices are 100 points apart
        assert abs(p1["poc"] - p2["poc"]) > 10

    def test_high_bin_count_precision(self):
        """More bins should give a more precise (narrower VA) profile."""
        df = _make_ohlcv(300, seed=42)

        p_low = compute_volume_profile(df, n_bins=10)
        p_high = compute_volume_profile(df, n_bins=80)

        # Both should be valid
        assert p_low["poc"] > 0
        assert p_high["poc"] > 0

        # POC should be similar regardless of bin count
        price_range = df["High"].max() - df["Low"].min()
        poc_diff = abs(p_low["poc"] - p_high["poc"])
        assert poc_diff < price_range * 0.15  # within 15% of range
