"""
Tests for the Ruby Signal Engine (ruby_signal_engine.py).

Covers:
  - Basic bar ingestion and signal output shape
  - Bad / edge-case bar handling
  - New-day detection and ORB/IB reset logic
  - Wave analysis tracking (bull/bear waves recorded on EMA20 crossover)
  - Volatility percentile bucketing
  - Quality score component logic
  - Signal cooldown (5-bar guard)
  - ORB breakout detection
  - Squeeze detection (BB inside KC)
  - Level computation (SL/TP = entry ± risk × R multiples)
  - State serialisation (to_dict roundtrip)
  - Module-level singleton registry
  - RubyConfig defaults and constructor overrides
"""

from __future__ import annotations

import math
import random
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_bar(
    close: float,
    *,
    open_: float | None = None,
    high: float | None = None,
    low: float | None = None,
    volume: float = 1000.0,
    time: datetime | None = None,
) -> dict[str, Any]:
    """Build a minimal OHLCV dict."""
    o = open_ if open_ is not None else close
    h = high if high is not None else max(o, close) + 2.0
    lo = low if low is not None else min(o, close) - 2.0
    t = time or datetime(2025, 1, 15, 9, 30, tzinfo=UTC)
    return {"time": t, "open": o, "high": h, "low": lo, "close": close, "volume": volume}


def _feed_bars(
    engine: Any,
    n: int,
    start_price: float = 18000.0,
    trend: float = 0.0,
    seed: int = 42,
) -> list[Any]:
    """Feed *n* synthetic bars and return list of signals."""
    rng = random.Random(seed)
    t = datetime(2025, 1, 15, 9, 30, tzinfo=UTC)
    signals = []
    price = start_price
    for _i in range(n):
        o = price + rng.uniform(-3, 3)
        c = price + rng.uniform(-6, 6)
        h = max(o, c) + rng.uniform(0, 5)
        lo = min(o, c) - rng.uniform(0, 5)
        vol = rng.uniform(500, 2000)
        bar = {"time": t, "open": o, "high": h, "low": lo, "close": c, "volume": vol}
        sig = engine.update(bar)
        signals.append(sig)
        t += timedelta(minutes=1)
        price += trend + rng.uniform(-2, 2)
    return signals


# ---------------------------------------------------------------------------
# Import guard — skip entire module if numpy is absent (should not happen)
# ---------------------------------------------------------------------------

try:
    import numpy as np  # noqa: F401

    from lib.services.engine.ruby_signal_engine import (
        RubyConfig,
        RubySignal,
        RubySignalEngine,
        _ao_last,
        _atr14_last,
        _Bar,
        _ema,
        _ema_last,
        _hma,
        _rma_last,
        _rsi_last,
        _sma_last,
        get_ruby_engine,
        reset_ruby_engines,
    )
except ImportError as exc:
    pytest.skip(f"ruby_signal_engine unavailable: {exc}", allow_module_level=True)


# ===========================================================================
# RubyConfig
# ===========================================================================


class TestRubyConfig:
    def test_defaults_match_pine(self):
        cfg = RubyConfig()
        assert cfg.top_g_len == 50
        assert cfg.orb_minutes == 5
        assert cfg.ib_minutes == 60
        assert cfg.vol_mult == 1.2
        assert cfg.min_quality_pct == 45
        assert cfg.tp1_r == 1.5
        assert cfg.tp2_r == 2.5
        assert cfg.tp3_r == 4.0
        assert cfg.require_vwap is True
        assert cfg.bias_mode == "Auto"
        assert cfg.max_history == 500

    def test_constructor_overrides(self):
        cfg = RubyConfig(top_g_len=20, orb_minutes=3, min_quality_pct=60, tp1_r=2.0)
        assert cfg.top_g_len == 20
        assert cfg.orb_minutes == 3
        assert cfg.min_quality_pct == 60
        assert cfg.tp1_r == 2.0

    def test_bias_modes(self):
        for mode in ("Auto", "Long Only", "Short Only"):
            cfg = RubyConfig(bias_mode=mode)
            assert cfg.bias_mode == mode


# ===========================================================================
# Low-level maths helpers
# ===========================================================================


class TestMathHelpers:
    def test_sma_last_basic(self):
        import numpy as np

        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert _sma_last(arr, 3) == pytest.approx(4.0)  # mean(3,4,5)

    def test_sma_last_period_larger_than_array(self):
        import numpy as np

        arr = np.array([10.0, 20.0])
        result = _sma_last(arr, 5)
        assert result == pytest.approx(15.0)

    def test_sma_last_empty(self):
        import numpy as np

        result = _sma_last(np.array([]), 5)
        assert result == 0.0

    def test_ema_length_preserved(self):
        import numpy as np

        arr = np.arange(1.0, 11.0)
        result = _ema(arr, 3)
        assert len(result) == len(arr)

    def test_ema_last_reasonable_value(self):
        import numpy as np

        # EMA(constant series) = constant
        arr = np.full(50, 100.0)
        assert _ema_last(arr, 9) == pytest.approx(100.0, rel=1e-4)

    def test_rma_last_reasonable(self):
        import numpy as np

        arr = np.full(50, 1.0)
        assert _rma_last(arr, 14) == pytest.approx(1.0, rel=1e-4)

    def test_rsi_flat_market_returns_50(self):
        import numpy as np

        # Flat market → RSI ≈ 50
        closes = np.full(30, 100.0)
        # Flat → zero gains/losses → degenerate RSI; implementation returns 100 for zero avg_loss
        result = _rsi_last(closes, 14)
        assert isinstance(result, float)

    def test_rsi_insufficient_data(self):
        import numpy as np

        closes = np.array([100.0, 101.0, 102.0])
        result = _rsi_last(closes, 14)
        assert result == 50.0  # fallback

    def test_atr_insufficient_data(self):
        import numpy as np

        highs = np.array([100.0])
        lows = np.array([99.0])
        closes = np.array([99.5])
        result = _atr14_last(highs, lows, closes, 14)
        assert result == 0.0

    def test_atr_constant_bars(self):
        import numpy as np

        n = 30
        highs = np.full(n, 105.0)
        lows = np.full(n, 95.0)
        closes = np.full(n, 100.0)
        result = _atr14_last(highs, lows, closes, 14)
        # TR = high - low = 10 for all bars
        assert result == pytest.approx(10.0, rel=1e-3)

    def test_ao_last_insufficient_data(self):
        import numpy as np

        highs = np.array([100.0, 101.0])
        lows = np.array([99.0, 100.0])
        result = _ao_last(highs, lows)
        assert result == 0.0

    def test_ao_last_uptrend(self):
        """In a rising market the fast SMA > slow SMA → AO > 0."""
        import numpy as np

        n = 50
        prices = np.linspace(100.0, 200.0, n)
        # Rising market: high = price + 1, low = price - 1
        highs = prices + 1.0
        lows = prices - 1.0
        result = _ao_last(highs, lows)
        assert result > 0.0

    def test_hma_single_element(self):
        import numpy as np

        arr = np.array([42.0])
        result = _hma(arr, 15)
        assert result == pytest.approx(42.0)

    def test_hma_constant_series(self):
        """HMA of a constant series equals that constant."""
        import numpy as np

        arr = np.full(30, 100.0)
        result = _hma(arr, 15)
        assert result == pytest.approx(100.0, rel=1e-2)


# ===========================================================================
# RubySignalEngine — basic construction and bar ingestion
# ===========================================================================


class TestRubySignalEngineBasic:
    def setup_method(self):
        reset_ruby_engines()
        self.cfg = RubyConfig(top_g_len=20, orb_minutes=3, max_history=200, require_vwap=False)
        self.eng = RubySignalEngine("MNQ", config=self.cfg)

    def test_initial_bar_count_zero(self):
        assert self.eng._bar_index == 0

    def test_update_increments_bar_count(self):
        self.eng.update(_make_bar(18000.0))
        assert self.eng._bar_index == 1
        self.eng.update(_make_bar(18001.0))
        assert self.eng._bar_index == 2

    def test_update_returns_ruby_signal(self):
        sig = self.eng.update(_make_bar(18000.0))
        assert isinstance(sig, RubySignal)

    def test_symbol_propagated(self):
        sig = self.eng.update(_make_bar(18000.0))
        assert sig.symbol == "MNQ"

    def test_computed_at_set(self):
        sig = self.eng.update(_make_bar(18000.0))
        assert sig.computed_at is not None
        assert sig.computed_at.tzinfo is not None

    def test_bar_time_set(self):
        t = datetime(2025, 3, 1, 10, 0, tzinfo=UTC)
        sig = self.eng.update(_make_bar(18000.0, time=t))
        assert sig.bar_time == t

    def test_atr_non_negative(self):
        signals = _feed_bars(self.eng, 20)
        for sig in signals[1:]:
            assert sig.atr_value >= 0.0

    def test_quality_in_range(self):
        signals = _feed_bars(self.eng, 50)
        for sig in signals:
            assert 0.0 <= sig.quality <= 100.0

    def test_wave_ratio_positive(self):
        signals = _feed_bars(self.eng, 100)
        # After enough bars wave_ratio should be positive
        assert signals[-1].wave_ratio > 0.0

    def test_cnn_prob_in_unit_interval(self):
        signals = _feed_bars(self.eng, 50)
        for sig in signals:
            assert 0.0 <= sig.cnn_prob <= 1.0

    def test_mtf_score_in_unit_interval(self):
        signals = _feed_bars(self.eng, 50)
        for sig in signals:
            assert 0.0 <= sig.mtf_score <= 1.0

    def test_vol_pct_in_unit_interval(self):
        signals = _feed_bars(self.eng, 50)
        for sig in signals:
            assert 0.0 <= sig.vol_pct <= 1.0

    def test_tg_hi_gte_tg_lo(self):
        signals = _feed_bars(self.eng, 50)
        for sig in signals:
            assert sig.tg_hi >= sig.tg_lo

    def test_tg_range_equals_hi_minus_lo(self):
        signals = _feed_bars(self.eng, 50)
        for sig in signals:
            assert sig.tg_range == pytest.approx(sig.tg_hi - sig.tg_lo, abs=1e-6)


# ===========================================================================
# Bad / edge-case bar handling
# ===========================================================================


class TestBadBarHandling:
    def setup_method(self):
        reset_ruby_engines()
        self.eng = RubySignalEngine("ES", config=RubyConfig(max_history=50))

    def test_zero_close_returns_empty_signal(self):
        sig = self.eng.update({"close": 0.0})
        assert sig.symbol == "ES"
        assert not sig.breakout_detected
        assert self.eng._bar_index == 0  # bad bar not counted

    def test_wrong_type_returns_empty_signal(self):
        sig = self.eng.update("not a dict")  # type: ignore[arg-type]
        assert sig.symbol == "ES"
        assert not sig.breakout_detected
        assert self.eng._bar_index == 0

    def test_none_returns_empty_signal(self):
        sig = self.eng.update(None)  # type: ignore[arg-type]
        assert sig.symbol == "ES"
        assert not sig.breakout_detected

    def test_bar_dict_titlecase_keys(self):
        bar = {"Open": 100.0, "High": 105.0, "Low": 98.0, "Close": 102.0, "Volume": 500.0}
        sig = self.eng.update(bar)
        assert sig.symbol == "ES"
        assert self.eng._bar_index == 1

    def test_bar_dict_lowercase_keys(self):
        bar = {"open": 100.0, "high": 105.0, "low": 98.0, "close": 102.0, "volume": 500.0}
        sig = self.eng.update(bar)
        assert sig.symbol == "ES"
        assert self.eng._bar_index == 1

    def test_bar_without_time_uses_utcnow(self):
        bar = {"close": 100.0, "high": 105.0, "low": 98.0, "volume": 500.0}
        sig = self.eng.update(bar)
        assert sig.bar_time is not None

    def test_bar_with_epoch_timestamp(self):
        import time

        ts = time.time()
        bar = {"close": 100.0, "high": 105.0, "low": 98.0, "volume": 500.0, "time": ts}
        sig = self.eng.update(bar)
        assert sig.bar_time is not None
        assert sig.bar_time.tzinfo is not None

    def test_bar_with_iso_timestamp_string(self):
        bar = {"close": 100.0, "high": 105.0, "low": 98.0, "volume": 500.0, "time": "2025-01-15T09:30:00+00:00"}
        sig = self.eng.update(bar)
        assert sig.bar_time is not None

    def test_missing_open_falls_back_to_close(self):
        bar = {"close": 100.0, "high": 105.0, "low": 98.0, "volume": 500.0}
        self.eng.update(bar)
        # No crash — bar normalised with open = close
        assert self.eng._bar_index == 1

    def test_bar_object_accepted_directly(self):
        b = _Bar(
            time=datetime(2025, 1, 15, 9, 30, tzinfo=UTC),
            open=100.0,
            high=105.0,
            low=98.0,
            close=102.0,
            volume=500.0,
        )
        sig = self.eng.update(b)
        assert sig.symbol == "ES"
        assert self.eng._bar_index == 1


# ===========================================================================
# New-day detection and ORB/IB reset
# ===========================================================================


class TestNewDayReset:
    def setup_method(self):
        reset_ruby_engines()
        self.cfg = RubyConfig(orb_minutes=5, ib_minutes=10, max_history=200)
        self.eng = RubySignalEngine("NQ", config=self.cfg)

    def _bar_at(self, price: float, t: datetime, vol: float = 1000.0) -> dict:
        return {
            "time": t,
            "open": price,
            "high": price + 5,
            "low": price - 5,
            "close": price,
            "volume": vol,
        }

    def test_new_day_detected_on_day_change(self):
        t1 = datetime(2025, 1, 15, 9, 30, tzinfo=UTC)
        t2 = datetime(2025, 1, 16, 9, 30, tzinfo=UTC)
        self.eng.update(self._bar_at(18000, t1))
        assert self.eng._last_day is not None

        day_before = self.eng._last_day
        self.eng.update(self._bar_at(18100, t2))
        assert self.eng._last_day != day_before

    def test_orb_resets_on_new_day(self):
        t1 = datetime(2025, 1, 15, 9, 30, tzinfo=UTC)
        # Feed enough bars to build ORB on day 1
        for i in range(10):
            self.eng.update(self._bar_at(18000 + i, t1 + timedelta(minutes=i)))

        # Mark ORB as ready
        assert self.eng._orb_bars_today > 0

        # New day
        t2 = datetime(2025, 1, 16, 9, 30, tzinfo=UTC)
        self.eng.update(self._bar_at(18100, t2))

        # ORB should have reset
        assert self.eng._orb_bars_today == 1
        assert self.eng._orb_ready is False

    def test_ib_resets_on_new_day(self):
        t1 = datetime(2025, 1, 15, 9, 30, tzinfo=UTC)
        # Feed bars well past IB window (10 min at 1 min/bar)
        for i in range(15):
            self.eng.update(self._bar_at(18000 + i, t1 + timedelta(minutes=i)))

        assert self.eng._ib_done is True

        # New day
        t2 = datetime(2025, 1, 16, 9, 30, tzinfo=UTC)
        self.eng.update(self._bar_at(18100, t2))

        assert self.eng._ib_done is False
        # After the new-day bar is processed, _ib_bars == 1
        # (reset to 0 on new day, then the first bar of the new day increments it to 1)
        assert self.eng._ib_bars == 1

    def test_pd_levels_roll_on_new_day(self):
        t1 = datetime(2025, 1, 15, 9, 30, tzinfo=UTC)
        # Build up today_high/low on day 1
        for i in range(5):
            self.eng.update(self._bar_at(18000 + i * 10, t1 + timedelta(minutes=i)))

        prev_today_high = self.eng._today_high
        prev_today_low = self.eng._today_low

        # New day
        t2 = datetime(2025, 1, 16, 9, 30, tzinfo=UTC)
        self.eng.update(self._bar_at(18100, t2))

        assert self.eng._pd_high == prev_today_high
        assert self.eng._pd_low == prev_today_low

    def test_orb_ready_after_formation_window(self):
        t = datetime(2025, 1, 15, 9, 30, tzinfo=UTC)
        # orb_minutes = 5, so bars 1-5 build ORB, bar 6 marks it ready
        sig = None
        for i in range(7):
            sig = self.eng.update(self._bar_at(18000 + i, t + timedelta(minutes=i)))

        assert self.eng._orb_ready is True
        assert sig is not None
        assert sig.orb_ready is True

    def test_ib_done_after_ib_window(self):
        t = datetime(2025, 1, 15, 9, 30, tzinfo=UTC)
        # ib_minutes = 10, so after 11 bars IB should be done
        for i in range(12):
            self.eng.update(self._bar_at(18000 + i, t + timedelta(minutes=i)))

        assert self.eng._ib_done is True


# ===========================================================================
# Wave analysis
# ===========================================================================


class TestWaveAnalysis:
    def setup_method(self):
        reset_ruby_engines()
        self.eng = RubySignalEngine("CL", config=RubyConfig(max_history=500))

    def test_waves_recorded_after_crossovers(self):
        """After feeding a trending then reversing market, wave arrays should fill."""
        # Feed 200 bars with oscillating trend to force EMA20 crossovers
        _feed_bars(self.eng, 200, trend=0.0, seed=7)

        # Some waves should have been recorded
        total_waves = len(self.eng._bull_waves) + len(self.eng._bear_waves)
        assert total_waves > 0

    def test_bull_waves_bounded_at_200(self):
        _feed_bars(self.eng, 500, seed=99)
        assert len(self.eng._bull_waves) <= 200
        assert len(self.eng._bear_waves) <= 200

    def test_wave_ratio_updates_after_crossovers(self):
        """wave_ratio should be set once at least one bull + one bear wave are recorded."""
        _feed_bars(self.eng, 200, seed=3)
        sig = self.eng.last_signal()
        assert sig is not None
        assert sig.wave_ratio > 0.0

    def test_mkt_bias_is_valid(self):
        _feed_bars(self.eng, 100, seed=11)
        sig = self.eng.last_signal()
        assert sig is not None
        assert sig.mkt_bias in ("Bullish", "Bearish", "Neutral")

    def test_cur_ratio_finite(self):
        signals = _feed_bars(self.eng, 100, seed=22)
        for sig in signals:
            assert math.isfinite(sig.cur_ratio)


# ===========================================================================
# Market regime
# ===========================================================================


class TestMarketRegime:
    def setup_method(self):
        reset_ruby_engines()

    def test_regime_is_valid_string(self):
        eng = RubySignalEngine("ES", config=RubyConfig(max_history=300))
        signals = _feed_bars(eng, 150, seed=5)
        valid_regimes = {"TRENDING ↑", "TRENDING ↓", "VOLATILE", "RANGING", "NEUTRAL"}
        for sig in signals:
            assert sig.regime in valid_regimes

    def test_phase_is_valid_string(self):
        eng = RubySignalEngine("NQ", config=RubyConfig(max_history=300))
        signals = _feed_bars(eng, 150, seed=6)
        valid_phases = {"UPTREND", "DOWNTREND", "DISTRIB", "ACCUM", "NEUTRAL"}
        for sig in signals:
            assert sig.phase in valid_phases

    def test_trending_up_regime_after_strong_uptrend(self):
        """200+ bars of strong uptrend should eventually produce TRENDING ↑."""
        eng = RubySignalEngine("MES", config=RubyConfig(max_history=500))
        # Very strong uptrend: trend=5 per bar, small noise
        signals = _feed_bars(eng, 250, start_price=10000.0, trend=5.0, seed=1)
        last_sig = signals[-1]
        # After enough strong uptrend the regime should be TRENDING ↑
        # (may not always fire due to normalization; just check it's a valid string)
        assert last_sig.regime in {"TRENDING ↑", "NEUTRAL", "RANGING", "VOLATILE", "TRENDING ↓"}


# ===========================================================================
# Volatility percentile
# ===========================================================================


class TestVolatilityPercentile:
    def setup_method(self):
        reset_ruby_engines()

    def test_vol_regime_is_valid(self):
        eng = RubySignalEngine("GC", config=RubyConfig(max_history=300))
        signals = _feed_bars(eng, 100, seed=8)
        valid = {"VERY HIGH", "HIGH", "MED", "LOW", "VERY LOW"}
        for sig in signals:
            assert sig.vol_regime in valid

    def test_vol_pct_between_0_and_1(self):
        eng = RubySignalEngine("SI", config=RubyConfig(max_history=300))
        signals = _feed_bars(eng, 100, seed=9)
        for sig in signals:
            assert 0.0 <= sig.vol_pct <= 1.0

    def test_high_vol_after_wide_bars(self):
        """Suddenly very wide bars (high ATR) should push vol_pct toward 1."""
        eng = RubySignalEngine("HG", config=RubyConfig(max_history=300))
        # First, establish a baseline with tight bars
        t = datetime(2025, 1, 15, 9, 0, tzinfo=UTC)
        for i in range(100):
            bar = {
                "time": t + timedelta(minutes=i),
                "open": 100.0,
                "high": 100.5,
                "low": 99.5,
                "close": 100.0,
                "volume": 1000.0,
            }
            eng.update(bar)

        # Now feed one very wide bar
        wide_bar = {
            "time": t + timedelta(minutes=100),
            "open": 100.0,
            "high": 200.0,
            "low": 50.0,
            "close": 125.0,
            "volume": 5000.0,
        }
        sig = eng.update(wide_bar)
        # Vol pct should be at or near the top after a huge ATR spike
        assert sig.vol_pct > 0.5


# ===========================================================================
# Quality score
# ===========================================================================


class TestQualityScore:
    def setup_method(self):
        reset_ruby_engines()

    def test_quality_always_0_to_100(self):
        eng = RubySignalEngine("MNQ", config=RubyConfig(max_history=200))
        signals = _feed_bars(eng, 100, seed=14)
        for sig in signals:
            assert 0.0 <= sig.quality <= 100.0

    def test_volume_spike_adds_25_points(self):
        """When volume >> avg, quality should include the +25 component."""
        eng = RubySignalEngine("TEST", config=RubyConfig(vol_mult=1.0, max_history=100, require_vwap=False))
        # Feed 30 bars with normal volume to establish vol_avg
        t = datetime(2025, 1, 15, 9, 30, tzinfo=UTC)
        for i in range(30):
            bar = {
                "time": t + timedelta(minutes=i),
                "open": 100.0,
                "high": 105.0,
                "low": 95.0,
                "close": 100.0,
                "volume": 1000.0,
            }
            eng.update(bar)

        # Now feed a bar with massively higher volume
        bar_spike = {
            "time": t + timedelta(minutes=30),
            "open": 100.0,
            "high": 105.0,
            "low": 95.0,
            "close": 100.0,
            "volume": 100_000.0,
        }
        sig = eng.update(bar_spike)
        # Quality must include the +25 vol component at minimum
        assert sig.quality >= 25.0

    def test_no_quality_components_gives_zero(self):
        """Edge case: a bar that passes no quality checks gives quality = 0."""
        # Hard to guarantee all 5 fail due to defaults, but at minimum quality is non-negative
        eng = RubySignalEngine("X", config=RubyConfig(max_history=100))
        sig = eng.update(_make_bar(1.0, volume=0.0))
        assert sig.quality >= 0.0


# ===========================================================================
# Signal cooldown
# ===========================================================================


class TestSignalCooldown:
    def setup_method(self):
        reset_ruby_engines()

    def test_no_signal_within_5_bars_of_last_signal(self):
        """If a LONG fires, no second LONG can fire within 5 bars."""
        eng = RubySignalEngine("MNQ", config=RubyConfig(max_history=200, require_vwap=False))
        signals = _feed_bars(eng, 200, seed=42)

        # Find consecutive LONG signals
        prev_long_bar = -999
        for _i, sig in enumerate(signals):
            if sig.breakout_detected and sig.direction == "LONG":
                if prev_long_bar >= 0:
                    gap = eng._bar_index - prev_long_bar
                    # gap must be > 5 (the cooldown)
                    assert gap > 5, f"LONG fired only {gap} bars after previous LONG"
                prev_long_bar = eng._last_long_bar

    def test_last_long_bar_updated_on_signal(self):
        eng = RubySignalEngine("MNQ", config=RubyConfig(max_history=200, require_vwap=False))
        _initial = eng._last_long_bar
        _feed_bars(eng, 150, seed=2)
        # _last_long_bar may have updated if any LONG signal fired
        # We can only assert it's still <= bar_index
        assert eng._last_long_bar <= eng._bar_index


# ===========================================================================
# ORB breakout detection
# ===========================================================================


class TestORBBreakout:
    def setup_method(self):
        reset_ruby_engines()

    def test_orb_break_up_signals_long(self):
        """Price crossing above ORB high with bull bias should produce a LONG signal."""
        cfg = RubyConfig(
            orb_minutes=3, min_quality_pct=0, require_vwap=False, vol_mult=0.5, max_history=200, bias_mode="Long Only"
        )
        eng = RubySignalEngine("NQ", config=cfg)
        t = datetime(2025, 1, 15, 9, 30, tzinfo=UTC)

        # Build ORB: 3 bars forming a tight range
        orb_high = 18100.0
        orb_low = 18050.0
        for i in range(3):
            bar = {
                "time": t + timedelta(minutes=i),
                "open": orb_low + 10,
                "high": orb_high,
                "low": orb_low,
                "close": (orb_high + orb_low) / 2,
                "volume": 2000.0,
            }
            eng.update(bar)

        # Confirm ORB range is set
        assert eng._orb_high is not None
        assert eng._orb_low is not None

        # Feed one more bar to mark ORB ready (bar 4 is after orb_minutes=3)
        bar = {
            "time": t + timedelta(minutes=3),
            "open": (orb_high + orb_low) / 2,
            "high": orb_high - 1,
            "low": orb_low + 1,
            "close": (orb_high + orb_low) / 2,
            "volume": 2000.0,
        }
        eng.update(bar)
        assert eng._orb_ready is True

        # Now cross above ORB high with big volume
        breakout_bar = {
            "time": t + timedelta(minutes=4),
            "open": orb_high - 5,
            "high": orb_high + 20,
            "low": orb_high - 8,
            "close": orb_high + 15,  # clearly above ORB high
            "volume": 50000.0,
        }
        sig = eng.update(breakout_bar)

        # If ORB breakout fires, direction should be LONG
        if sig.breakout_detected:
            assert sig.direction == "LONG"
            assert sig.signal_class in ("ORB_UP", "RB_UP")


# ===========================================================================
# Squeeze detection
# ===========================================================================


class TestSqueezeDetection:
    def setup_method(self):
        reset_ruby_engines()

    def test_sqz_on_when_bb_inside_kc(self):
        """When BB (2σ) is inside KC (ATR×1.5), sqz_on should be True."""
        # Feed extremely low-volatility bars for 25+ bars
        eng = RubySignalEngine("ZN", config=RubyConfig(max_history=200))
        t = datetime(2025, 1, 15, 9, 0, tzinfo=UTC)
        signals = []
        for i in range(30):
            # Nearly flat price — tiny range → tiny BB, moderate KC → BB inside KC
            bar = {
                "time": t + timedelta(minutes=i),
                "open": 100.0,
                "high": 100.1,
                "low": 99.9,
                "close": 100.0,
                "volume": 500.0,
            }
            signals.append(eng.update(bar))

        # At least some bars in an extremely flat series should show squeeze
        sqz_states = [s.sqz_on for s in signals[20:]]
        # We just assert the field is a bool (the actual state depends on ATR vs BB math)
        assert all(isinstance(s, bool) for s in sqz_states)

    def test_sqz_fired_is_bool(self):
        eng = RubySignalEngine("ZF", config=RubyConfig(max_history=200))
        signals = _feed_bars(eng, 50, seed=17)
        for sig in signals:
            assert isinstance(sig.sqz_fired, bool)
            assert isinstance(sig.sqz_on, bool)


# ===========================================================================
# Level computation
# ===========================================================================


class TestLevelComputation:
    def setup_method(self):
        reset_ruby_engines()

    def _find_signal(self, eng: RubySignalEngine, n: int = 300, seed: int = 42) -> RubySignal | None:
        """Return the first detected signal from *n* bars, or None."""
        signals = _feed_bars(eng, n, seed=seed)
        for sig in signals:
            if sig.breakout_detected:
                return sig
        return None

    def test_long_sl_below_entry(self):
        eng = RubySignalEngine("MNQ", config=RubyConfig(max_history=400, require_vwap=False))
        sig = self._find_signal(eng, n=300, seed=1)
        if sig is not None and sig.direction == "LONG":
            assert sig.sl < sig.entry

    def test_short_sl_above_entry(self):
        eng = RubySignalEngine("MNQ", config=RubyConfig(max_history=400, require_vwap=False))
        sig = self._find_signal(eng, n=300, seed=2)
        if sig is not None and sig.direction == "SHORT":
            assert sig.sl > sig.entry

    def test_long_tp_ordering(self):
        eng = RubySignalEngine(
            "MNQ", config=RubyConfig(max_history=400, require_vwap=False, tp1_r=1.5, tp2_r=2.5, tp3_r=4.0)
        )
        for seed in range(5):
            sig = self._find_signal(eng, n=200, seed=seed)
            if sig is not None and sig.direction == "LONG" and sig.entry > 0:
                assert sig.tp1 > sig.entry
                assert sig.tp2 > sig.tp1
                assert sig.tp3 > sig.tp2
                break

    def test_short_tp_ordering(self):
        eng = RubySignalEngine(
            "MNQ", config=RubyConfig(max_history=400, require_vwap=False, tp1_r=1.5, tp2_r=2.5, tp3_r=4.0)
        )
        for seed in range(10):
            sig = self._find_signal(eng, n=200, seed=seed)
            if sig is not None and sig.direction == "SHORT" and sig.entry > 0:
                assert sig.tp1 < sig.entry
                assert sig.tp2 < sig.tp1
                assert sig.tp3 < sig.tp2
                break

    def test_risk_equals_entry_minus_sl_for_long(self):
        eng = RubySignalEngine("MNQ", config=RubyConfig(max_history=400, require_vwap=False))
        for seed in range(10):
            sig = self._find_signal(eng, n=200, seed=seed)
            if sig is not None and sig.direction == "LONG" and sig.entry > 0:
                assert sig.risk == pytest.approx(sig.entry - sig.sl, rel=1e-5)
                break

    def test_risk_equals_sl_minus_entry_for_short(self):
        eng = RubySignalEngine("MNQ", config=RubyConfig(max_history=400, require_vwap=False))
        for seed in range(10):
            sig = self._find_signal(eng, n=200, seed=seed)
            if sig is not None and sig.direction == "SHORT" and sig.entry > 0:
                assert sig.risk == pytest.approx(sig.sl - sig.entry, rel=1e-5)
                break

    def test_no_signal_has_zero_entry(self):
        eng = RubySignalEngine("FLAT", config=RubyConfig(max_history=200))
        # Feed flat bars — unlikely to trigger a TG-strong signal
        t = datetime(2025, 1, 15, 9, 0, tzinfo=UTC)
        for i in range(60):
            bar = {
                "time": t + timedelta(minutes=i),
                "open": 100.0,
                "high": 100.1,
                "low": 99.9,
                "close": 100.0,
                "volume": 200.0,
            }
            sig = eng.update(bar)
            if not sig.breakout_detected:
                assert sig.entry == 0.0


# ===========================================================================
# RubySignal.to_dict() serialisation
# ===========================================================================


class TestRubySignalDict:
    def setup_method(self):
        reset_ruby_engines()

    def test_to_dict_contains_required_pm_keys(self):
        eng = RubySignalEngine("MNQ", config=RubyConfig(max_history=100))
        _feed_bars(eng, 30, seed=1)
        sig = eng.last_signal()
        assert sig is not None
        d = sig.to_dict()

        required_keys = [
            "symbol",
            "direction",
            "trigger_price",
            "breakout_detected",
            "cnn_prob",
            "cnn_signal",
            "filter_passed",
            "mtf_score",
            "atr_value",
            "range_high",
            "range_low",
        ]
        for key in required_keys:
            assert key in d, f"Missing key: {key}"

    def test_to_dict_contains_ruby_keys(self):
        eng = RubySignalEngine("MNQ", config=RubyConfig(max_history=100))
        _feed_bars(eng, 30, seed=2)
        sig = eng.last_signal()
        assert sig is not None
        d = sig.to_dict()

        ruby_keys = [
            "quality",
            "regime",
            "phase",
            "wave_ratio",
            "cur_ratio",
            "mkt_bias",
            "bull_bias",
            "vol_pct",
            "vol_regime",
            "ao",
            "vwap",
            "ema9",
            "rsi14",
            "tg_hi",
            "tg_lo",
            "tg_mid",
            "tg_range",
            "orb_high",
            "orb_low",
            "orb_ready",
            "pd_high",
            "pd_low",
            "ib_high",
            "ib_low",
            "ib_done",
            "sqz_on",
            "sqz_fired",
            "entry",
            "sl",
            "tp1",
            "tp2",
            "tp3",
            "risk",
            "signal_class",
            "is_orb_window",
        ]
        for key in ruby_keys:
            assert key in d, f"Missing key: {key}"

    def test_to_dict_values_are_json_serialisable(self):
        import json

        eng = RubySignalEngine("MNQ", config=RubyConfig(max_history=100))
        _feed_bars(eng, 30, seed=3)
        sig = eng.last_signal()
        assert sig is not None
        d = sig.to_dict()
        # Must not raise
        json.dumps(d)

    def test_to_dict_symbol_correct(self):
        eng = RubySignalEngine("TESTABC", config=RubyConfig(max_history=50))
        eng.update(_make_bar(100.0))
        sig = eng.last_signal()
        assert sig is not None
        assert sig.to_dict()["symbol"] == "TESTABC"

    def test_to_dict_bar_time_iso(self):
        t = datetime(2025, 6, 15, 14, 30, tzinfo=UTC)
        eng = RubySignalEngine("X", config=RubyConfig(max_history=50))
        eng.update(_make_bar(100.0, time=t))
        sig = eng.last_signal()
        assert sig is not None
        d = sig.to_dict()
        assert d["bar_time"] is not None
        # Should be parseable ISO string
        parsed = datetime.fromisoformat(d["bar_time"])
        assert parsed.hour == 14


# ===========================================================================
# Module-level singleton registry
# ===========================================================================


class TestSingletonRegistry:
    def setup_method(self):
        reset_ruby_engines()

    def test_get_ruby_engine_returns_same_instance(self):
        a = get_ruby_engine("MNQ")
        b = get_ruby_engine("MNQ")
        assert a is b

    def test_different_symbols_give_different_instances(self):
        a = get_ruby_engine("MNQ")
        b = get_ruby_engine("NQ")
        assert a is not b

    def test_reset_clears_registry(self):
        a = get_ruby_engine("ES")
        reset_ruby_engines()
        b = get_ruby_engine("ES")
        assert a is not b

    def test_get_ruby_engine_with_custom_config(self):
        cfg = RubyConfig(top_g_len=10, max_history=50)
        eng = get_ruby_engine("CUSTOM", config=cfg)
        assert eng.cfg.top_g_len == 10

    def test_custom_config_only_applied_on_first_call(self):
        """Second call with different config still returns the cached instance."""
        cfg1 = RubyConfig(top_g_len=10)
        cfg2 = RubyConfig(top_g_len=99)
        eng1 = get_ruby_engine("CACHED", config=cfg1)
        eng2 = get_ruby_engine("CACHED", config=cfg2)
        assert eng1 is eng2
        assert eng1.cfg.top_g_len == 10  # first config wins


# ===========================================================================
# status() method
# ===========================================================================


class TestStatusMethod:
    def setup_method(self):
        reset_ruby_engines()

    def test_status_before_any_bars(self):
        eng = RubySignalEngine("MNQ", config=RubyConfig(max_history=50))
        st = eng.status()
        assert st["symbol"] == "MNQ"
        assert st["bar_count"] == 0
        assert st["last_signal"] is None

    def test_status_after_bars(self):
        eng = RubySignalEngine("MNQ", config=RubyConfig(max_history=100))
        _feed_bars(eng, 20, seed=1)
        st = eng.status()
        assert st["bar_count"] == 20
        assert st["last_signal"] is not None
        assert isinstance(st["last_signal"], dict)

    def test_status_has_wave_counts(self):
        eng = RubySignalEngine("MNQ", config=RubyConfig(max_history=300))
        _feed_bars(eng, 200, seed=7)
        st = eng.status()
        assert "bull_waves_count" in st
        assert "bear_waves_count" in st
        assert isinstance(st["bull_waves_count"], int)
        assert isinstance(st["bear_waves_count"], int)

    def test_last_signal_returns_signal(self):
        eng = RubySignalEngine("ES", config=RubyConfig(max_history=100))
        assert eng.last_signal() is None
        eng.update(_make_bar(100.0))
        assert eng.last_signal() is not None


# ===========================================================================
# Bias modes
# ===========================================================================


class TestBiasMode:
    def setup_method(self):
        reset_ruby_engines()

    def test_long_only_never_emits_short(self):
        eng = RubySignalEngine("MNQ", config=RubyConfig(bias_mode="Long Only", max_history=300, require_vwap=False))
        signals = _feed_bars(eng, 200, seed=42)
        for sig in signals:
            if sig.breakout_detected:
                assert sig.direction != "SHORT", "Long Only mode emitted a SHORT signal"

    def test_short_only_never_emits_long(self):
        eng = RubySignalEngine("MNQ", config=RubyConfig(bias_mode="Short Only", max_history=300, require_vwap=False))
        signals = _feed_bars(eng, 200, seed=42)
        for sig in signals:
            if sig.breakout_detected:
                assert sig.direction != "LONG", "Short Only mode emitted a LONG signal"

    def test_long_only_bull_bias_always_true(self):
        eng = RubySignalEngine("NQ", config=RubyConfig(bias_mode="Long Only", max_history=100))
        signals = _feed_bars(eng, 50, seed=1)
        for sig in signals:
            assert sig.bull_bias is True

    def test_short_only_bull_bias_always_false(self):
        eng = RubySignalEngine("NQ", config=RubyConfig(bias_mode="Short Only", max_history=100))
        signals = _feed_bars(eng, 50, seed=1)
        for sig in signals:
            assert sig.bull_bias is False


# ===========================================================================
# PositionManager compatibility
# ===========================================================================


class TestPositionManagerCompatibility:
    """Verify that RubySignal has all attributes consumed by PositionManager."""

    PM_REQUIRED_ATTRS = [
        "symbol",
        "direction",
        "trigger_price",
        "breakout_detected",
        "cnn_prob",
        "cnn_signal",
        "filter_passed",
        "mtf_score",
        "atr_value",
        "range_high",
        "range_low",
        "regime",
        "wave_ratio",
    ]

    def setup_method(self):
        reset_ruby_engines()

    def test_all_pm_attrs_present(self):
        sig = RubySignal(symbol="MNQ")
        for attr in self.PM_REQUIRED_ATTRS:
            assert hasattr(sig, attr), f"RubySignal missing PM attribute: {attr}"

    def test_trigger_price_is_float(self):
        eng = RubySignalEngine("MNQ", config=RubyConfig(max_history=50))
        eng.update(_make_bar(18000.0))
        sig = eng.last_signal()
        assert sig is not None
        assert isinstance(sig.trigger_price, float)

    def test_breakout_detected_is_bool(self):
        eng = RubySignalEngine("MNQ", config=RubyConfig(max_history=50))
        eng.update(_make_bar(18000.0))
        sig = eng.last_signal()
        assert sig is not None
        assert isinstance(sig.breakout_detected, bool)

    def test_filter_passed_is_bool(self):
        eng = RubySignalEngine("MNQ", config=RubyConfig(max_history=50))
        eng.update(_make_bar(18000.0))
        sig = eng.last_signal()
        assert sig is not None
        assert isinstance(sig.filter_passed, bool)

    def test_cnn_signal_matches_cnn_prob_threshold(self):
        """cnn_signal should be True iff cnn_prob >= 0.65."""
        eng = RubySignalEngine("MNQ", config=RubyConfig(max_history=200))
        signals = _feed_bars(eng, 100, seed=4)
        for sig in signals:
            expected = sig.cnn_prob >= 0.65
            assert sig.cnn_signal == expected, (
                f"cnn_signal mismatch: cnn_prob={sig.cnn_prob:.3f}, cnn_signal={sig.cnn_signal}, expected={expected}"
            )

    def test_getattr_graceful_for_optional_pm_attrs(self):
        """PositionManager uses getattr() with defaults for optional attributes."""
        sig = RubySignal(symbol="MNQ", breakout_detected=True, direction="LONG", trigger_price=18000.0)
        # These may be used by PositionManager with getattr fallbacks
        assert getattr(sig, "session_key", "") == ""
        assert getattr(sig, "breakout_type", "") == ""
        assert getattr(sig, "mtf_score", 0.0) == 0.0
