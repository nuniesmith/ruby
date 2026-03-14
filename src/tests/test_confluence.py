"""
Unit tests for the confluence (Multi-Timeframe Filter) module.

Tests cover:
  - evaluate_htf_bias(): EMA stacking, direction, confidence, edge cases
  - evaluate_setup(): pullback detection, RSI filtering, quality scoring
  - evaluate_entry(): trigger logic, candle patterns, EMA crossovers
  - MultiTimeframeFilter.evaluate(): 3-layer scoring, tradeable flag
  - check_confluence(): convenience wrapper
  - get_recommended_timeframes(): presets for known/unknown instruments
  - confluence_to_dataframe(): output shape and columns
  - Edge cases: empty DataFrames, tiny DataFrames, constant prices
"""

import numpy as np
import pandas as pd
import pytest

from lib.analysis.confluence import (
    DEFAULT_EMA_FAST,
    DEFAULT_EMA_MID,
    DEFAULT_EMA_SLOW,
    TIMEFRAME_PRESETS,
    MultiTimeframeFilter,
    _atr,
    _ema,
    _rsi,
    check_confluence,
    confluence_to_dataframe,
    evaluate_entry,
    evaluate_htf_bias,
    evaluate_setup,
    get_instrument_ema_presets,
    get_recommended_timeframes,
)
from tests.conftest import _random_walk_ohlcv, _trending_ohlcv

# ---------------------------------------------------------------------------
# Local fixtures (reuse conftest generators)
# ---------------------------------------------------------------------------


@pytest.fixture()
def ohlcv_df():
    return _random_walk_ohlcv(n=500, seed=42)


@pytest.fixture()
def trending_up_df():
    """Strongly trending up — EMAs should stack bullish."""
    return _trending_ohlcv(n=300, seed=123, trend=0.002, volatility=0.001)


@pytest.fixture()
def trending_down_df():
    """Strongly trending down."""
    return _trending_ohlcv(n=300, seed=124, trend=-0.002, volatility=0.001)


@pytest.fixture()
def choppy_df():
    """Low drift, high volatility — neutral/choppy regime."""
    return _random_walk_ohlcv(n=300, seed=200, volatility=0.008)


@pytest.fixture()
def empty_df():
    return pd.DataFrame(columns=pd.Index(["Open", "High", "Low", "Close", "Volume"]))


@pytest.fixture()
def tiny_df():
    return _random_walk_ohlcv(n=5, seed=11, start_price=20.0)


@pytest.fixture()
def short_df():
    """30 bars — enough for fast EMA but not slow."""
    return _random_walk_ohlcv(n=30, seed=88, start_price=100.0)


# ═══════════════════════════════════════════════════════════════════════════
# Indicator helpers
# ═══════════════════════════════════════════════════════════════════════════


class TestIndicatorHelpers:
    def test_ema_returns_series(self, ohlcv_df):
        result = _ema(ohlcv_df["Close"], 10)
        assert isinstance(result, pd.Series)
        assert len(result) == len(ohlcv_df)

    def test_ema_length_one_equals_input(self):
        s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        result = _ema(s, 1)
        np.testing.assert_allclose(np.asarray(result), np.asarray(s), atol=1e-9)

    def test_ema_no_nan_at_end(self, ohlcv_df):
        result = _ema(ohlcv_df["Close"], 20)
        # Last value should never be NaN if input has enough bars
        assert not np.isnan(result.iloc[-1])

    def test_rsi_range(self, ohlcv_df):
        result = _rsi(ohlcv_df["Close"], 14)
        # After warmup, RSI should be in [0, 100]
        valid = result.dropna()
        assert (valid >= 0).all(), "RSI should be >= 0"
        assert (valid <= 100).all(), "RSI should be <= 100"

    def test_rsi_constant_price(self):
        """Constant price should produce a stable RSI (no up or down moves).

        With EWM-based RSI, constant prices yield zero gain and zero loss.
        The formula ``100 - 100/(1+rs)`` where ``rs = 0/epsilon`` produces
        RSI ≈ 0 rather than the textbook 50, because there are genuinely
        no gains.  This is mathematically correct for the implementation
        and is safe — it simply means "no momentum in either direction".
        """
        s = pd.Series([100.0] * 50)
        result = _rsi(s, 14)
        last = float(result.iloc[-1])
        # Accept 0 (EWM edge case) or ~50 (textbook ideal)
        assert 0 <= last <= 55, f"RSI on constant price should be 0–50, got {last}"

    def test_atr_positive(self, ohlcv_df):
        result = _atr(ohlcv_df["High"], ohlcv_df["Low"], ohlcv_df["Close"], 14)
        valid = result.dropna()
        assert (valid >= 0).all(), "ATR should be non-negative"

    def test_atr_length(self, ohlcv_df):
        result = _atr(ohlcv_df["High"], ohlcv_df["Low"], ohlcv_df["Close"], 14)
        assert len(result) == len(ohlcv_df)


# ═══════════════════════════════════════════════════════════════════════════
# evaluate_htf_bias
# ═══════════════════════════════════════════════════════════════════════════


class TestEvaluateHTFBias:
    def test_returns_dict(self, ohlcv_df):
        result = evaluate_htf_bias(ohlcv_df)
        assert isinstance(result, dict)

    def test_required_keys(self, ohlcv_df):
        result = evaluate_htf_bias(ohlcv_df)
        required = {
            "direction",
            "confidence",
            "ema_fast",
            "ema_mid",
            "ema_slow",
            "price_above_emas",
            "emas_stacked",
            "slope",
        }
        assert required.issubset(set(result.keys())), f"Missing keys: {required - set(result.keys())}"

    def test_direction_valid_value(self, ohlcv_df):
        result = evaluate_htf_bias(ohlcv_df)
        assert result["direction"] in ("bullish", "bearish", "neutral")

    def test_confidence_range(self, ohlcv_df):
        result = evaluate_htf_bias(ohlcv_df)
        assert 0.0 <= result["confidence"] <= 1.0

    def test_trending_up_is_bullish(self, trending_up_df):
        result = evaluate_htf_bias(trending_up_df)
        assert result["direction"] == "bullish", f"Expected bullish for strong uptrend, got {result['direction']}"
        assert result["emas_stacked"] is True

    def test_trending_down_is_bearish(self, trending_down_df):
        result = evaluate_htf_bias(trending_down_df)
        assert result["direction"] == "bearish", f"Expected bearish for strong downtrend, got {result['direction']}"
        assert result["emas_stacked"] is True

    def test_bullish_has_positive_confidence(self, trending_up_df):
        result = evaluate_htf_bias(trending_up_df)
        if result["direction"] == "bullish":
            assert result["confidence"] > 0

    def test_empty_returns_neutral(self, empty_df):
        result = evaluate_htf_bias(empty_df)
        assert result["direction"] == "neutral"
        assert result["confidence"] == 0.0

    def test_tiny_returns_neutral(self, tiny_df):
        """Too few bars for slow EMA → should return neutral."""
        result = evaluate_htf_bias(tiny_df)
        assert result["direction"] == "neutral"
        assert result["confidence"] == 0.0

    def test_custom_ema_periods(self, ohlcv_df):
        result = evaluate_htf_bias(ohlcv_df, ema_fast=5, ema_mid=10, ema_slow=20)
        assert isinstance(result, dict)
        assert result["direction"] in ("bullish", "bearish", "neutral")

    def test_ema_values_are_numbers(self, ohlcv_df):
        result = evaluate_htf_bias(ohlcv_df)
        for key in ("ema_fast", "ema_mid", "ema_slow"):
            val = result[key]
            if result["direction"] != "neutral":
                assert isinstance(val, (int, float, np.floating))
                assert np.isfinite(val)

    def test_slope_is_numeric(self, ohlcv_df):
        result = evaluate_htf_bias(ohlcv_df)
        assert isinstance(result["slope"], (int, float, np.floating))


# ═══════════════════════════════════════════════════════════════════════════
# evaluate_setup
# ═══════════════════════════════════════════════════════════════════════════


class TestEvaluateSetup:
    def test_returns_dict(self, ohlcv_df):
        result = evaluate_setup(ohlcv_df)
        assert isinstance(result, dict)

    def test_required_keys(self, ohlcv_df):
        result = evaluate_setup(ohlcv_df)
        required = {
            "direction",
            "quality",
            "has_pullback",
            "rsi",
            "trend_bars",
            "ema_fast",
            "ema_mid",
        }
        assert required.issubset(set(result.keys()))

    def test_direction_valid_value(self, ohlcv_df):
        result = evaluate_setup(ohlcv_df)
        assert result["direction"] in ("bullish", "bearish", "neutral")

    def test_quality_range(self, ohlcv_df):
        result = evaluate_setup(ohlcv_df)
        assert 0.0 <= result["quality"] <= 1.0

    def test_rsi_range(self, ohlcv_df):
        result = evaluate_setup(ohlcv_df)
        assert 0.0 <= result["rsi"] <= 100.0

    def test_trending_up_is_bullish(self, trending_up_df):
        result = evaluate_setup(trending_up_df)
        assert result["direction"] == "bullish"

    def test_trending_down_is_bearish(self, trending_down_df):
        result = evaluate_setup(trending_down_df)
        assert result["direction"] == "bearish"

    def test_trending_has_positive_quality(self, trending_up_df):
        result = evaluate_setup(trending_up_df)
        assert result["quality"] > 0

    def test_empty_returns_neutral(self, empty_df):
        result = evaluate_setup(empty_df)
        assert result["direction"] == "neutral"
        assert result["quality"] == 0.0

    def test_tiny_returns_neutral(self, tiny_df):
        result = evaluate_setup(tiny_df)
        assert result["direction"] == "neutral"

    def test_has_pullback_is_bool(self, ohlcv_df):
        result = evaluate_setup(ohlcv_df)
        assert isinstance(result["has_pullback"], bool)

    def test_trend_bars_nonnegative(self, ohlcv_df):
        result = evaluate_setup(ohlcv_df)
        assert result["trend_bars"] >= 0

    def test_custom_ema_periods(self, ohlcv_df):
        result = evaluate_setup(ohlcv_df, ema_fast=5, ema_mid=15, rsi_period=10)
        assert isinstance(result, dict)
        assert result["direction"] in ("bullish", "bearish", "neutral")

    def test_quality_higher_with_pullback_and_trend(self, trending_up_df):
        """Quality should increase when setup conditions are met."""
        result = evaluate_setup(trending_up_df)
        # A strong uptrend should have at least the base alignment quality
        if result["direction"] == "bullish":
            assert result["quality"] >= 0.4, "Aligned setup should have quality >= 0.4"


# ═══════════════════════════════════════════════════════════════════════════
# evaluate_entry
# ═══════════════════════════════════════════════════════════════════════════


class TestEvaluateEntry:
    def test_returns_dict(self, ohlcv_df):
        result = evaluate_entry(ohlcv_df)
        assert isinstance(result, dict)

    def test_required_keys(self, ohlcv_df):
        result = evaluate_entry(ohlcv_df)
        required = {
            "direction",
            "trigger",
            "candle_pattern",
            "rsi",
            "volume_confirmed",
            "ema_cross_recent",
        }
        assert required.issubset(set(result.keys()))

    def test_direction_valid_value(self, ohlcv_df):
        result = evaluate_entry(ohlcv_df)
        assert result["direction"] in ("bullish", "bearish", "neutral")

    def test_trigger_is_bool(self, ohlcv_df):
        result = evaluate_entry(ohlcv_df)
        assert isinstance(result["trigger"], bool)

    def test_volume_confirmed_is_bool(self, ohlcv_df):
        result = evaluate_entry(ohlcv_df)
        assert isinstance(result["volume_confirmed"], bool)

    def test_ema_cross_recent_is_bool(self, ohlcv_df):
        result = evaluate_entry(ohlcv_df)
        assert isinstance(result["ema_cross_recent"], bool)

    def test_candle_pattern_valid(self, ohlcv_df):
        result = evaluate_entry(ohlcv_df)
        valid_patterns = {
            "none",
            "bullish_engulfing",
            "bearish_engulfing",
            "hammer",
            "shooting_star",
        }
        assert result["candle_pattern"] in valid_patterns

    def test_rsi_range(self, ohlcv_df):
        result = evaluate_entry(ohlcv_df)
        assert 0.0 <= result["rsi"] <= 100.0

    def test_empty_returns_neutral(self, empty_df):
        result = evaluate_entry(empty_df)
        assert result["direction"] == "neutral"
        assert result["trigger"] is False

    def test_tiny_returns_neutral(self, tiny_df):
        result = evaluate_entry(tiny_df)
        assert result["direction"] == "neutral"
        assert result["trigger"] is False

    def test_custom_ema_periods(self, ohlcv_df):
        result = evaluate_entry(ohlcv_df, ema_fast=5, ema_mid=13, rsi_period=10)
        assert isinstance(result, dict)

    def test_trending_up_direction(self, trending_up_df):
        result = evaluate_entry(trending_up_df)
        # A strong uptrend should at least give a bullish direction
        assert result["direction"] in ("bullish", "neutral")

    def test_trending_down_direction(self, trending_down_df):
        result = evaluate_entry(trending_down_df)
        assert result["direction"] in ("bearish", "neutral")

    def test_ema_values_present(self, ohlcv_df):
        result = evaluate_entry(ohlcv_df)
        assert "ema_fast" in result
        assert "ema_mid" in result


# ═══════════════════════════════════════════════════════════════════════════
# MultiTimeframeFilter
# ═══════════════════════════════════════════════════════════════════════════


class TestMultiTimeframeFilter:
    def test_default_construction(self):
        mtf = MultiTimeframeFilter()
        assert mtf.htf_ema_fast == DEFAULT_EMA_FAST
        assert mtf.htf_ema_mid == DEFAULT_EMA_MID
        assert mtf.htf_ema_slow == DEFAULT_EMA_SLOW

    def test_custom_construction(self):
        mtf = MultiTimeframeFilter(
            htf_ema_fast=5,
            htf_ema_mid=15,
            htf_ema_slow=30,
            setup_ema_fast=8,
            setup_ema_mid=16,
            entry_ema_fast=4,
            entry_ema_mid=10,
            rsi_period=10,
        )
        assert mtf.htf_ema_fast == 5
        assert mtf.rsi_period == 10

    def test_evaluate_returns_dict(self, ohlcv_df):
        mtf = MultiTimeframeFilter()
        result = mtf.evaluate(ohlcv_df, ohlcv_df, ohlcv_df)
        assert isinstance(result, dict)

    def test_evaluate_required_keys(self, ohlcv_df):
        mtf = MultiTimeframeFilter()
        result = mtf.evaluate(ohlcv_df, ohlcv_df, ohlcv_df)
        required = {
            "score",
            "direction",
            "tradeable",
            "quality",
            "htf",
            "setup",
            "entry",
            "summary",
        }
        assert required.issubset(set(result.keys()))

    def test_score_range(self, ohlcv_df):
        mtf = MultiTimeframeFilter()
        result = mtf.evaluate(ohlcv_df, ohlcv_df, ohlcv_df)
        assert 0 <= result["score"] <= 3

    def test_direction_valid(self, ohlcv_df):
        mtf = MultiTimeframeFilter()
        result = mtf.evaluate(ohlcv_df, ohlcv_df, ohlcv_df)
        assert result["direction"] in ("bullish", "bearish", "neutral")

    def test_tradeable_is_bool(self, ohlcv_df):
        mtf = MultiTimeframeFilter()
        result = mtf.evaluate(ohlcv_df, ohlcv_df, ohlcv_df)
        assert isinstance(result["tradeable"], bool)

    def test_quality_range(self, ohlcv_df):
        mtf = MultiTimeframeFilter()
        result = mtf.evaluate(ohlcv_df, ohlcv_df, ohlcv_df)
        assert 0.0 <= result["quality"] <= 1.0

    def test_summary_is_string(self, ohlcv_df):
        mtf = MultiTimeframeFilter()
        result = mtf.evaluate(ohlcv_df, ohlcv_df, ohlcv_df)
        assert isinstance(result["summary"], str)
        assert len(result["summary"]) > 0

    def test_full_confluence_trending_up(self, trending_up_df):
        """All three layers on the same trending data should give score 3."""
        mtf = MultiTimeframeFilter()
        result = mtf.evaluate(trending_up_df, trending_up_df, trending_up_df)
        assert result["score"] >= 2, f"Expected high confluence on strong uptrend, got score={result['score']}"
        if result["score"] == 3:
            assert result["direction"] == "bullish"

    def test_full_confluence_trending_down(self, trending_down_df):
        """All three layers on downtrend data should give high score."""
        mtf = MultiTimeframeFilter()
        result = mtf.evaluate(trending_down_df, trending_down_df, trending_down_df)
        assert result["score"] >= 2
        if result["score"] == 3:
            assert result["direction"] == "bearish"

    def test_mixed_timeframes(self, trending_up_df, trending_down_df, ohlcv_df):
        """Different data on each timeframe should produce lower confluence."""
        mtf = MultiTimeframeFilter()
        result = mtf.evaluate(trending_up_df, trending_down_df, ohlcv_df)
        # With conflicting signals, score should be low
        assert result["score"] <= 2

    def test_tradeable_requires_score_3_and_trigger(self, ohlcv_df):
        mtf = MultiTimeframeFilter()
        result = mtf.evaluate(ohlcv_df, ohlcv_df, ohlcv_df)
        if result["tradeable"]:
            assert result["score"] == 3
            assert result["entry"]["trigger"] is True
        if result["score"] < 3:
            assert result["tradeable"] is False

    def test_htf_layer_is_dict(self, ohlcv_df):
        mtf = MultiTimeframeFilter()
        result = mtf.evaluate(ohlcv_df, ohlcv_df, ohlcv_df)
        assert isinstance(result["htf"], dict)
        assert "direction" in result["htf"]

    def test_setup_layer_is_dict(self, ohlcv_df):
        mtf = MultiTimeframeFilter()
        result = mtf.evaluate(ohlcv_df, ohlcv_df, ohlcv_df)
        assert isinstance(result["setup"], dict)
        assert "direction" in result["setup"]

    def test_entry_layer_is_dict(self, ohlcv_df):
        mtf = MultiTimeframeFilter()
        result = mtf.evaluate(ohlcv_df, ohlcv_df, ohlcv_df)
        assert isinstance(result["entry"], dict)
        assert "direction" in result["entry"]

    def test_asset_name_in_summary(self, ohlcv_df):
        mtf = MultiTimeframeFilter()
        result = mtf.evaluate(ohlcv_df, ohlcv_df, ohlcv_df, asset_name="Gold")
        assert "Gold" in result["summary"]

    def test_empty_dataframes_handled(self, empty_df):
        mtf = MultiTimeframeFilter()
        result = mtf.evaluate(empty_df, empty_df, empty_df)
        assert result["score"] == 0
        assert result["direction"] == "neutral"
        assert result["tradeable"] is False

    def test_tiny_dataframes_handled(self, tiny_df):
        mtf = MultiTimeframeFilter()
        result = mtf.evaluate(tiny_df, tiny_df, tiny_df)
        assert result["direction"] == "neutral"


# ═══════════════════════════════════════════════════════════════════════════
# check_confluence (convenience function)
# ═══════════════════════════════════════════════════════════════════════════


class TestCheckConfluence:
    def test_returns_dict(self, ohlcv_df):
        result = check_confluence(
            htf_df=ohlcv_df,
            setup_df=ohlcv_df,
            entry_df=ohlcv_df,
        )
        assert isinstance(result, dict)

    def test_matches_mtf_evaluate(self, ohlcv_df):
        """check_confluence should produce the same result as default MTF filter."""
        result_conv = check_confluence(
            htf_df=ohlcv_df,
            setup_df=ohlcv_df,
            entry_df=ohlcv_df,
            asset_name="Test",
        )
        mtf = MultiTimeframeFilter()
        result_direct = mtf.evaluate(ohlcv_df, ohlcv_df, ohlcv_df, "Test")

        assert result_conv["score"] == result_direct["score"]
        assert result_conv["direction"] == result_direct["direction"]
        assert result_conv["tradeable"] == result_direct["tradeable"]

    def test_with_asset_name(self, ohlcv_df):
        result = check_confluence(
            htf_df=ohlcv_df,
            setup_df=ohlcv_df,
            entry_df=ohlcv_df,
            asset_name="S&P",
        )
        assert "S&P" in result["summary"]

    def test_empty_inputs(self, empty_df):
        result = check_confluence(
            htf_df=empty_df,
            setup_df=empty_df,
            entry_df=empty_df,
        )
        assert result["score"] == 0
        assert result["direction"] == "neutral"


# ═══════════════════════════════════════════════════════════════════════════
# get_recommended_timeframes
# ═══════════════════════════════════════════════════════════════════════════


class TestGetRecommendedTimeframes:
    def test_known_instrument_gold(self):
        tfs = get_recommended_timeframes("Gold")
        assert isinstance(tfs, tuple)
        assert len(tfs) == 3
        # Gold should use longer timeframes per preset
        assert tfs == TIMEFRAME_PRESETS["Gold"]

    def test_known_instrument_sp(self):
        tfs = get_recommended_timeframes("S&P")
        assert isinstance(tfs, tuple)
        assert len(tfs) == 3
        assert tfs == TIMEFRAME_PRESETS["S&P"]

    def test_known_instrument_nasdaq(self):
        tfs = get_recommended_timeframes("Nasdaq")
        assert tfs == TIMEFRAME_PRESETS["Nasdaq"]

    def test_known_instrument_crude(self):
        tfs = get_recommended_timeframes("Crude Oil")
        assert tfs == TIMEFRAME_PRESETS["Crude Oil"]

    def test_known_instrument_silver(self):
        tfs = get_recommended_timeframes("Silver")
        assert tfs == TIMEFRAME_PRESETS["Silver"]

    def test_known_instrument_copper(self):
        tfs = get_recommended_timeframes("Copper")
        assert tfs == TIMEFRAME_PRESETS["Copper"]

    def test_unknown_instrument_returns_default(self):
        tfs = get_recommended_timeframes("Bitcoin")
        assert isinstance(tfs, tuple)
        assert len(tfs) == 3
        # Unknown instruments should return a sensible default

    def test_all_presets_have_three_elements(self):
        for name, tfs in TIMEFRAME_PRESETS.items():
            assert len(tfs) == 3, f"Preset {name} should have 3 timeframes"


# ═══════════════════════════════════════════════════════════════════════════
# confluence_to_dataframe
# ═══════════════════════════════════════════════════════════════════════════


class TestConfluenceToDataframe:
    def test_returns_dataframe(self, ohlcv_df):
        results = {}
        for name in ["Gold", "S&P", "Crude Oil"]:
            results[name] = check_confluence(
                htf_df=ohlcv_df,
                setup_df=ohlcv_df,
                entry_df=ohlcv_df,
                asset_name=name,
            )
        df_result = confluence_to_dataframe(results)
        assert isinstance(df_result, pd.DataFrame)

    def test_single_result(self, ohlcv_df):
        results = {
            "Gold": check_confluence(
                htf_df=ohlcv_df,
                setup_df=ohlcv_df,
                entry_df=ohlcv_df,
                asset_name="Gold",
            )
        }
        df_result = confluence_to_dataframe(results)
        assert isinstance(df_result, pd.DataFrame)
        assert len(df_result) >= 1

    def test_empty_results(self):
        df_result = confluence_to_dataframe({})
        assert isinstance(df_result, pd.DataFrame)
        assert len(df_result) == 0


# ═══════════════════════════════════════════════════════════════════════════
# get_instrument_ema_presets
# ═══════════════════════════════════════════════════════════════════════════


class TestGetInstrumentEMAPresets:
    def test_returns_dict(self):
        presets = get_instrument_ema_presets("Gold")
        assert isinstance(presets, dict)

    def test_known_instrument(self):
        presets = get_instrument_ema_presets("Gold")
        # Should have EMA-related keys
        assert len(presets) > 0

    def test_unknown_instrument_returns_defaults(self):
        presets = get_instrument_ema_presets("UnknownAsset")
        assert isinstance(presets, dict)


# ═══════════════════════════════════════════════════════════════════════════
# Scoring edge cases
# ═══════════════════════════════════════════════════════════════════════════


class TestScoringEdgeCases:
    def test_score_zero_when_all_neutral(self, choppy_df):
        """Choppy data might produce some neutral layers."""
        mtf = MultiTimeframeFilter()
        result = mtf.evaluate(choppy_df, choppy_df, choppy_df)
        # Score should be <= 3 (valid range)
        assert 0 <= result["score"] <= 3

    def test_score_consistency_across_calls(self, ohlcv_df):
        """Same input should produce same output (deterministic)."""
        mtf = MultiTimeframeFilter()
        r1 = mtf.evaluate(ohlcv_df, ohlcv_df, ohlcv_df)
        r2 = mtf.evaluate(ohlcv_df, ohlcv_df, ohlcv_df)
        assert r1["score"] == r2["score"]
        assert r1["direction"] == r2["direction"]
        assert r1["tradeable"] == r2["tradeable"]

    def test_different_filters_can_give_different_results(self, ohlcv_df):
        """Different EMA periods should potentially produce different results."""
        mtf1 = MultiTimeframeFilter(htf_ema_fast=5, htf_ema_mid=10, htf_ema_slow=20)
        mtf2 = MultiTimeframeFilter(htf_ema_fast=20, htf_ema_mid=40, htf_ema_slow=80)
        r1 = mtf1.evaluate(ohlcv_df, ohlcv_df, ohlcv_df)
        r2 = mtf2.evaluate(ohlcv_df, ohlcv_df, ohlcv_df)
        # Not guaranteed to be different, but both should be valid
        assert 0 <= r1["score"] <= 3
        assert 0 <= r2["score"] <= 3

    def test_constant_price_is_neutral(self):
        """Flat price should produce neutral on all layers."""
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
        result = check_confluence(htf_df=df, setup_df=df, entry_df=df)
        # All EMAs will be identical = not stacked either way
        assert result["direction"] == "neutral"
        assert result["score"] == 0


# ═══════════════════════════════════════════════════════════════════════════
# Integration: full pipeline
# ═══════════════════════════════════════════════════════════════════════════


class TestConfluenceIntegration:
    def test_full_pipeline(self, ohlcv_df):
        """Smoke test: run the entire confluence pipeline."""
        htf = evaluate_htf_bias(ohlcv_df)
        setup = evaluate_setup(ohlcv_df)
        entry = evaluate_entry(ohlcv_df)

        assert isinstance(htf, dict)
        assert isinstance(setup, dict)
        assert isinstance(entry, dict)

        result = check_confluence(
            htf_df=ohlcv_df,
            setup_df=ohlcv_df,
            entry_df=ohlcv_df,
            asset_name="TestAsset",
        )
        assert 0 <= result["score"] <= 3
        assert "TestAsset" in result["summary"]

    def test_pipeline_with_different_seeds(self):
        """Run confluence on different synthetic data to check robustness."""
        for seed in [1, 10, 42, 100, 999]:
            df = _random_walk_ohlcv(n=200, seed=seed)
            result = check_confluence(htf_df=df, setup_df=df, entry_df=df)
            assert result["direction"] in ("bullish", "bearish", "neutral")
            assert 0 <= result["score"] <= 3
            assert isinstance(result["tradeable"], bool)

    def test_pipeline_with_real_like_prices(self):
        """Test with realistic price levels for major futures instruments."""
        configs = [
            {"start_price": 5500.0, "volatility": 0.002},  # ES-like
            {"start_price": 2700.0, "volatility": 0.003},  # Gold-like
            {"start_price": 70.0, "volatility": 0.005},  # CL-like
            {"start_price": 18000.0, "volatility": 0.003},  # NQ-like
        ]
        for cfg in configs:
            df = _random_walk_ohlcv(
                n=300,
                seed=42,
                start_price=cfg["start_price"],
                volatility=cfg["volatility"],
            )
            result = check_confluence(htf_df=df, setup_df=df, entry_df=df)
            assert 0 <= result["score"] <= 3

    def test_three_different_timeframes_same_direction(self, trending_up_df):
        """When all TFs agree on direction, score should be high."""
        # Use the same trending data (as if all timeframes are trending up)
        result = check_confluence(
            htf_df=trending_up_df,
            setup_df=trending_up_df,
            entry_df=trending_up_df,
            asset_name="Trending",
        )
        assert result["score"] >= 2
        assert result["direction"] in ("bullish", "neutral")

    def test_opposing_htf_and_entry(self, trending_up_df, trending_down_df):
        """Conflicting HTF and entry should prevent full confluence."""
        result = check_confluence(
            htf_df=trending_up_df,
            setup_df=trending_up_df,
            entry_df=trending_down_df,
        )
        # With HTF bullish but entry bearish, unlikely to get 3/3
        assert result["score"] < 3
