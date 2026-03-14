"""
Unit tests for the CVD (Cumulative Volume Delta) module.

Tests cover:
  - compute_cvd(): column creation, value ranges, daily anchoring
  - detect_cvd_divergences(): detection on synthetic data, empty/short guards
  - detect_absorption_candles(): detection and return shape
  - cvd_summary(): keys present, bias direction, edge cases
  - cvd_confirms_trend(): trend confirmation logic
  - Indicator helpers: _cvd_indicator, _delta_indicator, _cvd_ema_indicator
"""

import numpy as np
import pandas as pd
import pytest

from lib.analysis.cvd import (
    _estimate_buy_volume,
    compute_cvd,
    cvd_confirms_trend,
    cvd_summary,
    detect_absorption_candles,
    detect_cvd_divergences,
)

# ---------------------------------------------------------------------------
# Fixtures (local helpers reusing conftest generators)
# ---------------------------------------------------------------------------
from tests.conftest import _random_walk_ohlcv, _trending_ohlcv


@pytest.fixture()
def ohlcv_df():
    return _random_walk_ohlcv(n=500, seed=42)


@pytest.fixture()
def trending_df():
    return _trending_ohlcv(n=300, seed=123)


@pytest.fixture()
def short_df():
    return _random_walk_ohlcv(n=20, seed=99, start_price=50.0)


@pytest.fixture()
def empty_df():
    return pd.DataFrame(columns=pd.Index(["Open", "High", "Low", "Close", "Volume"]))


@pytest.fixture()
def tiny_df():
    return _random_walk_ohlcv(n=5, seed=11, start_price=20.0)


# ═══════════════════════════════════════════════════════════════════════════
# _estimate_buy_volume
# ═══════════════════════════════════════════════════════════════════════════


class TestEstimateBuyVolume:
    def test_returns_series(self, ohlcv_df):
        bv = _estimate_buy_volume(ohlcv_df["High"], ohlcv_df["Low"], ohlcv_df["Close"], ohlcv_df["Volume"])
        assert isinstance(bv, pd.Series)
        assert len(bv) == len(ohlcv_df)

    def test_buy_volume_between_zero_and_total(self, ohlcv_df):
        bv = _estimate_buy_volume(ohlcv_df["High"], ohlcv_df["Low"], ohlcv_df["Close"], ohlcv_df["Volume"])
        assert (bv >= 0).all(), "Buy volume should never be negative"
        assert (bv <= ohlcv_df["Volume"] + 1e-9).all(), "Buy volume should not exceed total volume"

    def test_zero_range_bar_splits_fifty_fifty(self):
        """When High == Low (doji), buy volume should be 50% of total."""
        high = pd.Series([100.0, 100.0])
        low = pd.Series([100.0, 100.0])
        close = pd.Series([100.0, 100.0])
        volume = pd.Series([1000.0, 2000.0])
        bv = _estimate_buy_volume(high, low, close, volume)
        np.testing.assert_allclose(np.asarray(bv), [500.0, 1000.0], rtol=1e-9)

    def test_bullish_bar_has_more_buy_volume(self):
        """Close at the high → buy_volume ≈ total volume."""
        high = pd.Series([110.0])
        low = pd.Series([100.0])
        close = pd.Series([110.0])
        volume = pd.Series([1000.0])
        bv = _estimate_buy_volume(high, low, close, volume)
        assert float(bv.iloc[0]) == pytest.approx(1000.0, rel=1e-9)

    def test_bearish_bar_has_less_buy_volume(self):
        """Close at the low → buy_volume ≈ 0."""
        high = pd.Series([110.0])
        low = pd.Series([100.0])
        close = pd.Series([100.0])
        volume = pd.Series([1000.0])
        bv = _estimate_buy_volume(high, low, close, volume)
        assert float(bv.iloc[0]) == pytest.approx(0.0, abs=1e-9)


# ═══════════════════════════════════════════════════════════════════════════
# compute_cvd
# ═══════════════════════════════════════════════════════════════════════════


class TestComputeCVD:
    def test_output_columns_present(self, ohlcv_df):
        result = compute_cvd(ohlcv_df)
        expected_cols = {
            "buy_volume",
            "sell_volume",
            "delta",
            "cvd",
            "cvd_ema",
            "cvd_slope",
        }
        assert expected_cols.issubset(set(result.columns))

    def test_output_length_matches_input(self, ohlcv_df):
        result = compute_cvd(ohlcv_df)
        assert len(result) == len(ohlcv_df)

    def test_delta_equals_buy_minus_sell(self, ohlcv_df):
        result = compute_cvd(ohlcv_df)
        expected = result["buy_volume"] - result["sell_volume"]
        np.testing.assert_allclose(np.asarray(result["delta"]), np.asarray(expected), atol=1e-9)

    def test_buy_plus_sell_equals_volume(self, ohlcv_df):
        result = compute_cvd(ohlcv_df)
        total = result["buy_volume"] + result["sell_volume"]
        np.testing.assert_allclose(np.asarray(total), np.asarray(ohlcv_df["Volume"].astype(float)), atol=1e-9)

    def test_cvd_is_cumulative_delta(self, ohlcv_df):
        """Without daily anchoring, CVD should equal cumsum of delta."""
        result = compute_cvd(ohlcv_df, anchor_daily=False)
        expected = result["delta"].cumsum()
        np.testing.assert_allclose(np.asarray(result["cvd"]), np.asarray(expected), atol=1e-6)

    def test_daily_anchoring_resets(self, ohlcv_df):
        """With daily anchoring, CVD should reset at each new date."""
        result = compute_cvd(ohlcv_df, anchor_daily=True)
        dates = result.index.to_series().dt.date
        unique_dates = dates.unique()

        if len(unique_dates) > 1:
            # The first CVD value of each day should equal the first delta of that day
            for d in unique_dates:
                mask = dates == d
                day_cvd = result.loc[mask, "cvd"]
                day_delta = result.loc[mask, "delta"]
                first_cvd = float(day_cvd.iloc[0])
                first_delta = float(day_delta.iloc[0])
                assert first_cvd == pytest.approx(first_delta, rel=1e-6), (
                    f"On date {d}, first CVD should equal first delta (daily anchor)"
                )

    def test_cvd_ema_is_smooth(self, ohlcv_df):
        """cvd_ema should be smoother than raw CVD (lower variance of diffs)."""
        result = compute_cvd(ohlcv_df)
        cvd_var = result["cvd"].diff().var()
        ema_var = result["cvd_ema"].diff().var()
        assert ema_var < cvd_var, "EMA of CVD should have lower diff variance"

    def test_empty_input(self, empty_df):
        result = compute_cvd(empty_df)
        assert result.empty or len(result) == 0

    def test_tiny_input_no_crash(self, tiny_df):
        result = compute_cvd(tiny_df)
        assert len(result) == len(tiny_df)

    def test_no_nan_in_core_columns(self, ohlcv_df):
        result = compute_cvd(ohlcv_df)
        for col in ["buy_volume", "sell_volume", "delta", "cvd"]:
            assert not bool(result[col].isna().any()), f"NaN found in {col}"


# ═══════════════════════════════════════════════════════════════════════════
# detect_cvd_divergences
# ═══════════════════════════════════════════════════════════════════════════


class TestDetectCVDDivergences:
    def test_returns_list(self, ohlcv_df):
        divs = detect_cvd_divergences(ohlcv_df)
        assert isinstance(divs, list)

    def test_divergence_dict_keys(self, ohlcv_df):
        divs = detect_cvd_divergences(ohlcv_df)
        if divs:
            d = divs[0]
            assert "type" in d
            assert d["type"] in ("bullish", "bearish")

    def test_empty_input_returns_empty(self, empty_df):
        divs = detect_cvd_divergences(empty_df)
        assert divs == []

    def test_short_input_returns_empty(self, tiny_df):
        divs = detect_cvd_divergences(tiny_df)
        assert divs == []

    def test_lookback_parameter_respected(self, ohlcv_df):
        """Smaller lookback should potentially find more divergences."""
        divs_small = detect_cvd_divergences(ohlcv_df, lookback=10)
        divs_large = detect_cvd_divergences(ohlcv_df, lookback=40)
        # Not guaranteed, but at least both should return valid lists
        assert isinstance(divs_small, list)
        assert isinstance(divs_large, list)

    def test_all_divergences_have_valid_type(self, ohlcv_df):
        divs = detect_cvd_divergences(ohlcv_df)
        for d in divs:
            assert d["type"] in ("bullish", "bearish")


# ═══════════════════════════════════════════════════════════════════════════
# detect_absorption_candles
# ═══════════════════════════════════════════════════════════════════════════


class TestDetectAbsorptionCandles:
    def test_returns_list(self, ohlcv_df):
        cvd_df = compute_cvd(ohlcv_df)
        result = detect_absorption_candles(cvd_df)
        assert isinstance(result, (list, np.ndarray, pd.Series))

    def test_length_matches_input(self, ohlcv_df):
        cvd_df = compute_cvd(ohlcv_df)
        result = detect_absorption_candles(cvd_df)
        assert len(result) == len(cvd_df)

    def test_values_are_signals(self, ohlcv_df):
        """Absorption signals should be -1, 0, or 1."""
        cvd_df = compute_cvd(ohlcv_df)
        result = detect_absorption_candles(cvd_df)
        for val in result:
            assert int(val) in (-1, 0, 1), f"Unexpected absorption value: {val}"

    def test_empty_input(self, empty_df):
        # compute_cvd on empty returns empty
        cvd_df = compute_cvd(empty_df)
        result = detect_absorption_candles(cvd_df)
        assert len(result) == 0 or all(v == 0 for v in result)


# ═══════════════════════════════════════════════════════════════════════════
# cvd_summary
# ═══════════════════════════════════════════════════════════════════════════


class TestCVDSummary:
    def test_returns_dict(self, ohlcv_df):
        cvd_df = compute_cvd(ohlcv_df)
        summary = cvd_summary(cvd_df)
        assert isinstance(summary, dict)

    def test_required_keys(self, ohlcv_df):
        cvd_df = compute_cvd(ohlcv_df)
        summary = cvd_summary(cvd_df)
        expected_keys = {
            "bias",
            "bias_emoji",
            "cvd_current",
            "delta_current",
            "cvd_slope",
        }
        assert expected_keys.issubset(set(summary.keys())), f"Missing keys: {expected_keys - set(summary.keys())}"

    def test_bias_is_valid_string(self, ohlcv_df):
        cvd_df = compute_cvd(ohlcv_df)
        summary = cvd_summary(cvd_df)
        assert summary["bias"] in ("bullish", "bearish", "neutral")

    def test_bias_emoji_present(self, ohlcv_df):
        cvd_df = compute_cvd(ohlcv_df)
        summary = cvd_summary(cvd_df)
        assert summary["bias_emoji"] in ("🟢", "🔴", "⚪")

    def test_slope_is_numeric(self, ohlcv_df):
        cvd_df = compute_cvd(ohlcv_df)
        summary = cvd_summary(cvd_df)
        assert isinstance(summary["cvd_slope"], (int, float, np.floating))

    def test_empty_input_returns_dict(self, empty_df):
        cvd_df = compute_cvd(empty_df)
        summary = cvd_summary(cvd_df)
        assert isinstance(summary, dict)

    def test_trending_data_has_nonzero_slope(self, trending_df):
        cvd_df = compute_cvd(trending_df)
        summary = cvd_summary(cvd_df)
        # Trending data should produce some non-zero CVD values
        assert summary["cvd_current"] != 0 or summary["delta_current"] != 0


# ═══════════════════════════════════════════════════════════════════════════
# cvd_confirms_trend
# ═══════════════════════════════════════════════════════════════════════════


class TestCVDConfirmsTrend:
    def test_returns_bool(self, ohlcv_df):
        cvd_df = compute_cvd(ohlcv_df)
        result = cvd_confirms_trend(cvd_df, direction="bullish")
        assert isinstance(result, bool)

    def test_accepts_bullish_direction(self, ohlcv_df):
        cvd_df = compute_cvd(ohlcv_df)
        # Should not raise
        cvd_confirms_trend(cvd_df, direction="bullish")

    def test_accepts_bearish_direction(self, ohlcv_df):
        cvd_df = compute_cvd(ohlcv_df)
        # Should not raise
        cvd_confirms_trend(cvd_df, direction="bearish")

    def test_empty_input_returns_false(self, empty_df):
        cvd_df = compute_cvd(empty_df)
        assert cvd_confirms_trend(cvd_df, direction="bullish") is False


# ═══════════════════════════════════════════════════════════════════════════
# Integration: full pipeline
# ═══════════════════════════════════════════════════════════════════════════


class TestCVDIntegration:
    def test_full_pipeline_runs_without_error(self, ohlcv_df):
        """Smoke test: run the entire CVD pipeline end to end."""
        cvd_df = compute_cvd(ohlcv_df)
        summary = cvd_summary(cvd_df)
        divs = detect_cvd_divergences(ohlcv_df)
        absorptions = detect_absorption_candles(cvd_df)
        confirms = cvd_confirms_trend(cvd_df, direction="bullish")

        assert isinstance(summary, dict)
        assert isinstance(divs, list)
        assert len(absorptions) == len(ohlcv_df)
        assert isinstance(confirms, bool)

    def test_pipeline_with_different_seeds(self):
        """Run CVD on several different synthetic datasets to check robustness."""
        for seed in [1, 10, 42, 100, 999]:
            df = _random_walk_ohlcv(n=200, seed=seed)
            cvd_df = compute_cvd(df)
            summary = cvd_summary(cvd_df)
            assert summary["bias"] in ("bullish", "bearish", "neutral")
            assert not bool(cvd_df["cvd"].isna().any())

    def test_large_volume_spikes_dont_crash(self):
        """Ensure extreme volume values don't cause overflow or NaN."""
        df = _random_walk_ohlcv(n=100, seed=7)
        df.loc[df.index[50], "Volume"] = 1e12  # huge spike
        cvd_df = compute_cvd(df)
        assert not bool(cvd_df["cvd"].isna().any())
        assert np.isfinite(cvd_df["cvd"].iloc[-1])

    def test_constant_price_doesnt_crash(self):
        """Flat price (all bars identical) should not produce NaN/error."""
        n = 100
        idx = pd.date_range("2025-01-06 03:00", periods=n, freq="5min", tz="America/New_York")
        df = pd.DataFrame(
            {
                "Open": [100.0] * n,
                "High": [100.0] * n,
                "Low": [100.0] * n,
                "Close": [100.0] * n,
                "Volume": [500.0] * n,
            },
            index=idx,
        )
        cvd_df = compute_cvd(df)
        # All deltas should be zero (50% buy = 50% sell)
        np.testing.assert_allclose(np.asarray(cvd_df["delta"]), 0.0, atol=1e-9)
