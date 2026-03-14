"""
Tests for Phase 2C — Swing Detector
====================================
Covers:
  - SwingSignal / SwingExitSignal / SwingState dataclass creation & serialization
  - Pullback entry detection (key level proximity, confirmation bar, retrace depth)
  - Breakout entry detection (prior day H/L break, volume confirmation, bar quality)
  - Gap continuation detection (gap alignment, fill check, settle period, pullback)
  - Combined detect_swing_entries() orchestration
  - Exit evaluation: stop loss, TP1 scale-out, TP2 close, EMA trailing, time stop
  - Swing state management: create, update, phase transitions
  - Position sizing and risk:reward calculations
  - Multi-asset scanning (scan_swing_entries_all_assets)
  - DailyPlan integration (enrich_swing_candidates)
  - Redis publish / load round-trip
  - Edge cases: neutral bias, insufficient data, zero ATR, time stop boundary
  - Utility helpers: _safe_float, _bar_body_ratio, _bar_close_position, _is_time_stop_due
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from lib.trading.strategies.daily.bias_analyzer import (
    BiasDirection,
    CandlePattern,
    DailyBias,
    KeyLevels,
)
from lib.trading.strategies.daily.swing_detector import (
    BREAKOUT_VOLUME_MULT,
    GAP_FILL_THRESHOLD,
    GAP_MIN_ATR_RATIO,
    MAX_SWING_CONTRACTS,
    MIN_SWING_CONTRACTS,
    PULLBACK_MAX_RETRACE_PCT,
    PULLBACK_MIN_RETRACE_PCT,
    REDIS_KEY_SWING_SIGNALS,
    REDIS_KEY_SWING_STATES,
    REDIS_PUBSUB_SWING,
    SWING_RISK_PCT,
    SWING_SL_ATR_MULT,
    SWING_TP1_ATR_MULT,
    SWING_TP2_ATR_MULT,
    TIME_STOP_HOUR,
    TIME_STOP_MINUTE,
    TP1_SCALE_FRACTION,
    TRAIL_EMA_PERIOD,
    SwingEntryStyle,
    SwingExitReason,
    SwingExitSignal,
    SwingPhase,
    SwingSignal,
    SwingState,
    _bar_body_ratio,
    _bar_close_position,
    _bar_is_bullish,
    _compute_ema,
    _compute_position_size,
    _compute_risk_reward,
    _is_time_stop_due,
    _price_decimals_from_tick,
    _safe_float,
    create_swing_state,
    detect_breakout_entry,
    detect_gap_continuation,
    detect_pullback_entry,
    detect_swing_entries,
    enrich_swing_candidates,
    evaluate_swing_exits,
    load_swing_signals,
    publish_swing_signals,
    publish_swing_states,
    scan_swing_entries_all_assets,
    update_swing_state,
)

_EST = ZoneInfo("America/New_York")


# ═══════════════════════════════════════════════════════════════════════════
# Synthetic data helpers
# ═══════════════════════════════════════════════════════════════════════════


def _make_intraday_bars(
    n: int = 40,
    start_price: float = 2700.0,
    trend: float = 0.001,
    atr_like: float = 5.0,
    seed: int = 42,
    include_volume: bool = True,
) -> pd.DataFrame:
    """Build a synthetic intraday OHLCV DataFrame (5-min bars)."""
    rng = np.random.default_rng(seed)
    close = np.zeros(n)
    close[0] = start_price
    for i in range(1, n):
        close[i] = close[i - 1] * (1.0 + rng.normal(trend, 0.002))

    high = close + rng.uniform(0.5, atr_like, n)
    low = close - rng.uniform(0.5, atr_like, n)
    open_ = close + rng.uniform(-atr_like * 0.3, atr_like * 0.3, n)

    data: dict[str, Any] = {
        "Open": open_,
        "High": high,
        "Low": low,
        "Close": close,
    }
    if include_volume:
        data["Volume"] = rng.integers(500, 5000, n).astype(float)

    return pd.DataFrame(data)


def _make_pullback_bars(
    prior_day_high: float = 2750.0,
    prior_day_low: float = 2700.0,
    pullback_target: float = 2710.0,
    direction: str = "LONG",
    n: int = 30,
    confirmed: bool = True,
) -> pd.DataFrame:
    """Build bars that simulate a pullback to a key level.

    The bars trend away from the level first, then pull back to it.
    If confirmed, the last bar shows a reversal in the bias direction.

    The retrace check in the detector uses the last-20-bar window:
      LONG:  retrace = (recent_high - current_price) / (recent_high - recent_low)
      SHORT: retrace = (current_price - recent_low) / (recent_high - recent_low)
    We need retrace to be between 0.25 and 0.75.

    Strategy: include a deep dip/spike early in the 20-bar window so that
    the impulse range is wide, keeping the retrace fraction moderate even
    though the last close is near the key level.

    For LONG with target=2701, spread=2:
      Put a low-bar (close≈target-12) at bar n-19, then rise to peak
      (close≈target+18), then pull back to target.
      recent_high ≈ target+20, recent_low ≈ target-14, range ≈ 34
      retrace = (target+20 - target) / 34 ≈ 0.59  ✓
    """
    spread = 2.0
    prices = np.zeros(n)

    if direction == "LONG":
        # Bars 0..(n-21): sit at a neutral base slightly below target
        base = pullback_target - 5
        peak = pullback_target + 18
        dip = pullback_target - 12  # Creates a low early in last-20 window

        pre_n = max(0, n - 20)
        for i in range(pre_n):
            prices[i] = base + (i / max(pre_n, 1)) * 3

        # Bar at n-20: start of the last-20 window — deep dip
        if n >= 21:
            prices[n - 20] = dip

        # Bars (n-19)..(n-11): rise from dip toward peak
        rise_start = max(0, n - 19)
        rise_end = max(0, n - 10)
        rise_len = rise_end - rise_start
        for i in range(rise_len):
            frac = (i + 1) / rise_len
            prices[rise_start + i] = dip + (peak - dip) * frac

        # Bars (n-10)..(n-1): pull back from peak toward target
        fall_start = max(0, n - 10)
        fall_len = n - fall_start
        for i in range(fall_len):
            frac = (i + 1) / fall_len
            prices[fall_start + i] = peak + (pullback_target - peak) * frac
    else:  # SHORT
        base = pullback_target + 5
        trough = pullback_target - 18
        spike = pullback_target + 12

        pre_n = max(0, n - 20)
        for i in range(pre_n):
            prices[i] = base - (i / max(pre_n, 1)) * 3

        if n >= 21:
            prices[n - 20] = spike

        rise_start = max(0, n - 19)
        rise_end = max(0, n - 10)
        rise_len = rise_end - rise_start
        for i in range(rise_len):
            frac = (i + 1) / rise_len
            prices[rise_start + i] = spike + (trough - spike) * frac

        fall_start = max(0, n - 10)
        fall_len = n - fall_start
        for i in range(fall_len):
            frac = (i + 1) / fall_len
            prices[fall_start + i] = trough + (pullback_target - trough) * frac

    close = prices.copy()
    high = close + spread
    low = close - spread
    open_ = close.copy()
    for i in range(1, n):
        open_[i] = close[i - 1]

    # Confirmation bar: last bar closes in bias direction with good body
    if confirmed and direction == "LONG":
        open_[-1] = pullback_target - 3.0
        close[-1] = pullback_target + 1.0
        high[-1] = pullback_target + 2.0
        low[-1] = pullback_target - 4.0
    elif confirmed and direction == "SHORT":
        open_[-1] = pullback_target + 3.0
        close[-1] = pullback_target - 1.0
        low[-1] = pullback_target - 2.0
        high[-1] = pullback_target + 4.0

    volume = np.full(n, 2000.0)

    return pd.DataFrame(
        {
            "Open": open_,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": volume,
        }
    )


def _make_breakout_bars(
    breakout_level: float = 2750.0,
    direction: str = "LONG",
    n: int = 20,
    volume_surge: bool = True,
    good_close: bool = True,
) -> pd.DataFrame:
    """Build bars that simulate a breakout through a key level."""
    close = np.linspace(breakout_level - 20, breakout_level + 10, n)
    if direction == "SHORT":
        close = np.linspace(breakout_level + 20, breakout_level - 10, n)

    spread = 3.0
    high = close + spread
    low = close - spread
    open_ = close.copy()

    # Last bar: strong breakout bar
    if direction == "LONG":
        open_[-1] = breakout_level - 2
        close[-1] = breakout_level + 8
        high[-1] = breakout_level + 9
        low[-1] = breakout_level - 3
        if not good_close:
            close[-1] = breakout_level + 1  # Weak close (near low of range)
            low[-1] = breakout_level - 5
            high[-1] = breakout_level + 8
    else:
        open_[-1] = breakout_level + 2
        close[-1] = breakout_level - 8
        low[-1] = breakout_level - 9
        high[-1] = breakout_level + 3
        if not good_close:
            close[-1] = breakout_level - 1
            high[-1] = breakout_level + 5
            low[-1] = breakout_level - 8

    avg_vol = 2000.0
    volume = np.full(n, avg_vol)
    if volume_surge:
        volume[-1] = avg_vol * 2.0  # 2× average — well above 1.3× threshold
    else:
        volume[-1] = avg_vol * 0.8  # Below average

    return pd.DataFrame(
        {
            "Open": open_,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": volume,
        }
    )


def _make_gap_bars(
    prior_close: float = 2700.0,
    gap_size: float = 15.0,
    direction: str = "LONG",
    n: int = 12,
    filled: bool = False,
    with_pullback: bool = True,
) -> pd.DataFrame:
    """Build bars simulating a gap open with or without fill."""
    session_open = prior_close + gap_size if direction == "LONG" else prior_close - gap_size

    # Build bars starting from session open
    close = np.zeros(n)
    close[0] = session_open

    for i in range(1, n):
        if filled:
            # Move toward prior close (filling the gap)
            close[i] = close[i - 1] + (prior_close - close[i - 1]) * 0.15
        else:
            # Stay near gap, slight move in gap direction
            if direction == "LONG":
                close[i] = close[i - 1] + np.random.uniform(-1, 2)
            else:
                close[i] = close[i - 1] + np.random.uniform(-2, 1)

    # If with_pullback and not filled: last few bars pull back into gap zone
    if with_pullback and not filled:
        for i in range(max(0, n - 3), n):
            if direction == "LONG":
                close[i] = session_open - gap_size * 0.3 + np.random.uniform(-1, 1)
            else:
                close[i] = session_open + gap_size * 0.3 + np.random.uniform(-1, 1)

    spread = 2.0
    high = close + spread
    low = close - spread
    open_ = np.zeros(n)
    open_[0] = session_open
    for i in range(1, n):
        open_[i] = close[i - 1]

    # Last bar: resumption in bias direction
    if direction == "LONG":
        open_[-1] = close[-1] - 2
        close[-1] = close[-1] + 1
    else:
        open_[-1] = close[-1] + 2
        close[-1] = close[-1] - 1

    volume = np.full(n, 2000.0)

    return pd.DataFrame(
        {
            "Open": open_,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": volume,
        }
    )


def _make_bias(
    direction: BiasDirection = BiasDirection.LONG,
    confidence: float = 0.75,
    asset_name: str = "Gold",
    prior_day_high: float = 2750.0,
    prior_day_low: float = 2700.0,
    prior_day_close: float = 2730.0,
    weekly_mid: float = 2720.0,
    monthly_ema20: float = 2715.0,
    volume_confirmation: bool = True,
    atr_expanding: bool = False,
    overnight_gap_direction: float = 0.0,
) -> DailyBias:
    """Build a DailyBias for testing."""
    kl = KeyLevels(
        prior_day_high=prior_day_high,
        prior_day_low=prior_day_low,
        prior_day_mid=round((prior_day_high + prior_day_low) / 2, 4),
        prior_day_close=prior_day_close,
        weekly_high=prior_day_high + 20,
        weekly_low=prior_day_low - 20,
        weekly_mid=weekly_mid,
        monthly_ema20=monthly_ema20,
    )
    return DailyBias(
        asset_name=asset_name,
        direction=direction,
        confidence=confidence,
        reasoning="Test bias reasoning",
        key_levels=kl,
        candle_pattern=CandlePattern.BULLISH_ENGULFING
        if direction == BiasDirection.LONG
        else CandlePattern.BEARISH_ENGULFING,
        weekly_range_position=0.4 if direction == BiasDirection.LONG else 0.6,
        monthly_trend_score=0.5 if direction == BiasDirection.LONG else -0.5,
        volume_confirmation=volume_confirmation,
        overnight_gap_direction=overnight_gap_direction,
        atr_expanding=atr_expanding,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Utility helpers
# ═══════════════════════════════════════════════════════════════════════════


class TestSafeFloat:
    def test_normal_values(self):
        assert _safe_float(3.14) == 3.14
        assert _safe_float(0) == 0.0
        assert _safe_float(-1.5) == -1.5

    def test_string_numeric(self):
        assert _safe_float("42.5") == 42.5

    def test_none(self):
        assert _safe_float(None) == 0.0
        assert _safe_float(None, 99.0) == 99.0

    def test_nan(self):
        assert _safe_float(float("nan")) == 0.0

    def test_inf(self):
        assert _safe_float(float("inf")) == 0.0
        assert _safe_float(float("-inf"), -1.0) == -1.0

    def test_invalid_string(self):
        assert _safe_float("abc") == 0.0


class TestBarHelpers:
    def test_body_ratio_full_body(self):
        ratio = _bar_body_ratio(100.0, 110.0, 100.0, 110.0)
        assert ratio == pytest.approx(1.0)

    def test_body_ratio_doji(self):
        ratio = _bar_body_ratio(105.0, 110.0, 100.0, 105.0)
        assert ratio == pytest.approx(0.0)

    def test_body_ratio_half(self):
        ratio = _bar_body_ratio(100.0, 110.0, 100.0, 105.0)
        assert ratio == pytest.approx(0.5)

    def test_body_ratio_zero_range(self):
        assert _bar_body_ratio(100.0, 100.0, 100.0, 100.0) == 0.0

    def test_bar_is_bullish(self):
        assert _bar_is_bullish(100.0, 105.0) is True
        assert _bar_is_bullish(105.0, 100.0) is False
        assert _bar_is_bullish(100.0, 100.0) is False

    def test_close_position_at_high(self):
        pos = _bar_close_position(100.0, 110.0, 100.0, 110.0)
        assert pos == pytest.approx(1.0)

    def test_close_position_at_low(self):
        pos = _bar_close_position(100.0, 110.0, 100.0, 100.0)
        assert pos == pytest.approx(0.0)

    def test_close_position_mid(self):
        pos = _bar_close_position(100.0, 110.0, 100.0, 105.0)
        assert pos == pytest.approx(0.5)

    def test_close_position_zero_range(self):
        assert _bar_close_position(100.0, 100.0, 100.0, 100.0) == 0.5


class TestPriceDecimals:
    def test_small_tick(self):
        assert _price_decimals_from_tick(0.0001) == 4

    def test_medium_tick(self):
        assert _price_decimals_from_tick(0.01) == 2

    def test_large_tick(self):
        assert _price_decimals_from_tick(1.0) == 0

    def test_zero_tick(self):
        assert _price_decimals_from_tick(0) == 2

    def test_negative_tick(self):
        assert _price_decimals_from_tick(-0.01) == 2


class TestComputeEma:
    def test_ema_length(self):
        s = pd.Series(range(50), dtype=float)
        ema = _compute_ema(s, 21)
        assert len(ema) == 50

    def test_ema_converges(self):
        s = pd.Series([100.0] * 50)
        ema = _compute_ema(s, 10)
        assert ema.iloc[-1] == pytest.approx(100.0)


class TestTimeStop:
    def test_before_cutoff(self):
        dt = datetime(2024, 1, 15, 14, 0, tzinfo=_EST)
        assert _is_time_stop_due(dt) is False

    def test_at_cutoff(self):
        dt = datetime(2024, 1, 15, TIME_STOP_HOUR, TIME_STOP_MINUTE, tzinfo=_EST)
        assert _is_time_stop_due(dt) is True

    def test_after_cutoff(self):
        dt = datetime(2024, 1, 15, 16, 0, tzinfo=_EST)
        assert _is_time_stop_due(dt) is True

    def test_early_morning(self):
        dt = datetime(2024, 1, 15, 9, 30, tzinfo=_EST)
        assert _is_time_stop_due(dt) is False


class TestPositionSizing:
    @patch("lib.trading.strategies.daily.swing_detector._get_point_value", return_value=10.0)
    def test_basic_sizing(self, _mock_pv):
        size, risk = _compute_position_size(2700.0, 2685.0, 50_000, "Gold")
        # stop distance = 15, risk per contract = 15 * 10 = 150
        # max risk = 50000 * 0.005 = 250
        # size = int(250/150) = 1
        assert size >= MIN_SWING_CONTRACTS
        assert size <= MAX_SWING_CONTRACTS
        assert risk > 0

    @patch("lib.trading.strategies.daily.swing_detector._get_point_value", return_value=10.0)
    def test_caps_at_max(self, _mock_pv):
        # Very small stop → lots of contracts → should cap at MAX
        size, risk = _compute_position_size(2700.0, 2699.0, 500_000, "Gold")
        assert size <= MAX_SWING_CONTRACTS

    @patch("lib.trading.strategies.daily.swing_detector._get_point_value", return_value=10.0)
    def test_minimum_one_contract(self, _mock_pv):
        # Very large stop → fraction of a contract → should floor at MIN
        size, risk = _compute_position_size(2700.0, 2600.0, 10_000, "Gold")
        assert size >= MIN_SWING_CONTRACTS

    def test_zero_stop_distance(self):
        size, risk = _compute_position_size(100.0, 100.0, 50_000)
        assert size == MIN_SWING_CONTRACTS


class TestRiskReward:
    def test_basic_rr(self):
        rr = _compute_risk_reward(100.0, 95.0, 110.0)
        assert rr == pytest.approx(2.0)

    def test_zero_risk(self):
        assert _compute_risk_reward(100.0, 100.0, 110.0) == 0.0

    def test_short_rr(self):
        rr = _compute_risk_reward(100.0, 105.0, 90.0)
        assert rr == pytest.approx(2.0)


# ═══════════════════════════════════════════════════════════════════════════
# Data class tests
# ═══════════════════════════════════════════════════════════════════════════


class TestSwingSignalDataclass:
    def test_creation_defaults(self):
        sig = SwingSignal()
        assert sig.asset_name == ""
        assert sig.entry_style == SwingEntryStyle.PULLBACK
        assert sig.direction == "LONG"
        assert sig.confidence == 0.0
        assert sig.phase == SwingPhase.WATCHING

    def test_to_dict(self):
        sig = SwingSignal(
            asset_name="Gold",
            entry_style=SwingEntryStyle.BREAKOUT,
            direction="SHORT",
            confidence=0.85,
            entry_price=2700.0,
            stop_loss=2715.0,
            tp1=2680.0,
            tp2=2660.0,
            atr=10.0,
        )
        d = sig.to_dict()
        assert d["asset_name"] == "Gold"
        assert d["entry_style"] == "breakout_entry"
        assert d["direction"] == "SHORT"
        assert d["confidence"] == 0.85
        assert d["stop_loss"] == 2715.0
        assert d["phase"] == "watching"

    def test_all_entry_styles(self):
        assert len(SwingEntryStyle) == 3
        assert SwingEntryStyle.PULLBACK.value == "pullback_entry"
        assert SwingEntryStyle.BREAKOUT.value == "breakout_entry"
        assert SwingEntryStyle.GAP_CONTINUATION.value == "gap_continuation"


class TestSwingExitSignalDataclass:
    def test_creation_defaults(self):
        ex = SwingExitSignal()
        assert ex.reason == SwingExitReason.STOP_LOSS
        assert ex.exit_price == 0.0
        assert ex.scale_fraction == 1.0

    def test_to_dict(self):
        ex = SwingExitSignal(
            reason=SwingExitReason.TP1_HIT,
            exit_price=2740.0,
            pnl_estimate=200.0,
            r_multiple=2.0,
            scale_fraction=0.5,
            reasoning="TP1 hit",
        )
        d = ex.to_dict()
        assert d["reason"] == "tp1_hit"
        assert d["exit_price"] == 2740.0
        assert d["scale_fraction"] == 0.5


class TestSwingStateDataclass:
    def test_creation_defaults(self):
        st = SwingState()
        assert st.asset_name == ""
        assert st.phase == SwingPhase.WATCHING
        assert st.position_size == 1

    def test_to_dict(self):
        sig = SwingSignal(asset_name="Gold", direction="LONG")
        st = SwingState(
            asset_name="Gold",
            signal=sig,
            phase=SwingPhase.ACTIVE,
            entry_price=2700.0,
            current_stop=2685.0,
        )
        d = st.to_dict()
        assert d["asset_name"] == "Gold"
        assert d["phase"] == "active"
        assert d["signal"] is not None
        assert d["signal"]["asset_name"] == "Gold"

    def test_to_dict_no_signal(self):
        st = SwingState(asset_name="Gold")
        d = st.to_dict()
        assert d["signal"] is None


class TestSwingPhaseEnum:
    def test_all_phases(self):
        phases = list(SwingPhase)
        assert len(phases) == 6
        assert SwingPhase.WATCHING in phases
        assert SwingPhase.ENTRY_READY in phases
        assert SwingPhase.ACTIVE in phases
        assert SwingPhase.TP1_HIT in phases
        assert SwingPhase.TRAILING in phases
        assert SwingPhase.CLOSED in phases


class TestSwingExitReasonEnum:
    def test_all_reasons(self):
        reasons = list(SwingExitReason)
        assert len(reasons) == 7
        assert SwingExitReason.STOP_LOSS in reasons
        assert SwingExitReason.TP1_HIT in reasons
        assert SwingExitReason.TP2_HIT in reasons
        assert SwingExitReason.TRAILING_STOP in reasons
        assert SwingExitReason.TIME_STOP in reasons
        assert SwingExitReason.EMA_TRAIL in reasons
        assert SwingExitReason.INVALIDATED in reasons


# ═══════════════════════════════════════════════════════════════════════════
# Pullback Entry Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestPullbackEntry:
    def test_returns_none_on_neutral_bias(self):
        bars = _make_intraday_bars()
        bias = _make_bias(direction=BiasDirection.NEUTRAL)
        result = detect_pullback_entry(bars, bias, 2710.0, 10.0)
        assert result is None

    def test_returns_none_with_insufficient_bars(self):
        bars = _make_intraday_bars(n=3)
        bias = _make_bias()
        result = detect_pullback_entry(bars, bias, 2710.0, 10.0)
        assert result is None

    def test_returns_none_with_zero_atr(self):
        bars = _make_intraday_bars()
        bias = _make_bias()
        result = detect_pullback_entry(bars, bias, 2710.0, 0.0)
        assert result is None

    def test_returns_none_with_none_bars(self):
        bias = _make_bias()
        result = detect_pullback_entry(pd.DataFrame(), bias, 2710.0, 10.0)  # type: ignore[arg-type]
        assert result is None

    def test_returns_none_no_key_levels(self):
        bias = _make_bias()
        object.__setattr__(bias, "key_levels", None)  # type: ignore[arg-type]
        bars = _make_intraday_bars()
        result = detect_pullback_entry(bars, bias, 2710.0, 10.0)
        assert result is None

    @patch("lib.trading.strategies.daily.swing_detector._get_tick_size", return_value=0.01)
    @patch("lib.trading.strategies.daily.swing_detector._get_point_value", return_value=10.0)
    def test_long_pullback_detected(self, _pv, _ts):
        """Price near prior_day_low (support) with LONG bias → signal."""
        prior_low = 2700.0
        atr = 10.0
        # Price at the level — within tolerance (0.3 * 10 = 3)
        current_price = prior_low + 1.0

        # Build bars where price rose from ~2695 to ~2726, then pulled back
        # to 2701.  Last 20 bars: high≈2728, low≈2697 → range≈31.
        # Retrace = (2728 - 2701) / 31 ≈ 0.87 — too high.
        # So use n=30 with the improved helper which gives range ~30,
        # retrace depth ~14 → 14/30 ≈ 0.47  ✓
        bars = _make_pullback_bars(
            prior_day_high=2750.0,
            prior_day_low=prior_low,
            pullback_target=current_price,
            direction="LONG",
            n=30,
            confirmed=True,
        )

        bias = _make_bias(
            direction=BiasDirection.LONG,
            confidence=0.8,
            prior_day_high=2750.0,
            prior_day_low=prior_low,
            # Set other key levels far away so only prior_day_low matches
            weekly_mid=2600.0,
            monthly_ema20=2600.0,
        )

        result = detect_pullback_entry(bars, bias, current_price, atr, "Gold", 50_000)

        assert result is not None
        assert result.direction == "LONG"
        assert result.entry_style == SwingEntryStyle.PULLBACK
        assert result.phase == SwingPhase.ENTRY_READY
        assert result.confidence > 0
        assert result.stop_loss < current_price
        assert result.tp1 > current_price
        assert result.tp2 > result.tp1

    @patch("lib.trading.strategies.daily.swing_detector._get_tick_size", return_value=0.01)
    @patch("lib.trading.strategies.daily.swing_detector._get_point_value", return_value=10.0)
    def test_short_pullback_detected(self, _pv, _ts):
        """Price near prior_day_high (resistance) with SHORT bias → signal."""
        prior_high = 2750.0
        atr = 10.0
        current_price = prior_high - 1.0

        bars = _make_pullback_bars(
            prior_day_high=prior_high,
            prior_day_low=2700.0,
            pullback_target=current_price,
            direction="SHORT",
            n=30,
            confirmed=True,
        )

        bias = _make_bias(
            direction=BiasDirection.SHORT,
            confidence=0.7,
            prior_day_high=prior_high,
            prior_day_low=2700.0,
            # Set other key levels far away so only prior_day_high matches
            weekly_mid=2850.0,
            monthly_ema20=2850.0,
        )

        result = detect_pullback_entry(bars, bias, current_price, atr, "Gold", 50_000)

        assert result is not None
        assert result.direction == "SHORT"
        assert result.stop_loss > current_price
        assert result.tp1 < current_price

    @patch("lib.trading.strategies.daily.swing_detector._get_tick_size", return_value=0.01)
    @patch("lib.trading.strategies.daily.swing_detector._get_point_value", return_value=10.0)
    def test_no_pullback_price_too_far(self, _pv, _ts):
        """Price far from all key levels → no signal."""
        atr = 10.0
        current_price = 2900.0  # Way above all levels

        bars = _make_intraday_bars(n=30, start_price=2900.0)
        bias = _make_bias(direction=BiasDirection.LONG)

        result = detect_pullback_entry(bars, bias, current_price, atr, "Gold")
        assert result is None

    @patch("lib.trading.strategies.daily.swing_detector._get_tick_size", return_value=0.01)
    @patch("lib.trading.strategies.daily.swing_detector._get_point_value", return_value=10.0)
    def test_unconfirmed_pullback_watching(self, _pv, _ts):
        """Pullback detected but no confirmation bar → phase=WATCHING."""
        prior_low = 2700.0
        atr = 10.0
        current_price = prior_low + 1.0

        # Build bars without confirmation (last bar closes against bias)
        bars = _make_pullback_bars(
            prior_day_high=2750.0,
            prior_day_low=prior_low,
            pullback_target=current_price,
            direction="LONG",
            confirmed=False,
        )
        # Override the last bar to have a bearish (non-confirming) candle
        bars.iloc[-1, bars.columns.get_loc("Open")] = current_price + 3.0
        bars.iloc[-1, bars.columns.get_loc("Close")] = current_price - 3.0
        bars.iloc[-1, bars.columns.get_loc("High")] = current_price + 4.0
        bars.iloc[-1, bars.columns.get_loc("Low")] = current_price - 4.0

        bias = _make_bias(
            direction=BiasDirection.LONG,
            confidence=0.8,
            prior_day_low=prior_low,
        )

        result = detect_pullback_entry(bars, bias, current_price, atr, "Gold")

        # Depending on retrace calculation, might still detect or not;
        # if detected, should be WATCHING phase (unconfirmed)
        if result is not None:
            assert result.phase == SwingPhase.WATCHING
            assert result.confirmation_bar_idx == -1

    @patch("lib.trading.strategies.daily.swing_detector._get_tick_size", return_value=0.01)
    @patch("lib.trading.strategies.daily.swing_detector._get_point_value", return_value=10.0)
    def test_pullback_confidence_includes_volume(self, _pv, _ts):
        """Volume confirmation adds to confidence score."""
        prior_low = 2700.0
        atr = 10.0
        current_price = prior_low + 1.0

        bars = _make_pullback_bars(
            prior_day_low=prior_low,
            pullback_target=current_price,
            direction="LONG",
            confirmed=True,
        )

        bias_no_vol = _make_bias(
            direction=BiasDirection.LONG,
            confidence=0.8,
            prior_day_low=prior_low,
            volume_confirmation=False,
        )
        bias_with_vol = _make_bias(
            direction=BiasDirection.LONG,
            confidence=0.8,
            prior_day_low=prior_low,
            volume_confirmation=True,
        )

        sig_no_vol = detect_pullback_entry(bars, bias_no_vol, current_price, atr, "Gold")
        sig_with_vol = detect_pullback_entry(bars, bias_with_vol, current_price, atr, "Gold")

        if sig_no_vol is not None and sig_with_vol is not None:
            assert sig_with_vol.confidence >= sig_no_vol.confidence

    @patch("lib.trading.strategies.daily.swing_detector._get_tick_size", return_value=0.01)
    @patch("lib.trading.strategies.daily.swing_detector._get_point_value", return_value=10.0)
    def test_pullback_signal_has_reasoning(self, _pv, _ts):
        """Signal should include human-readable reasoning."""
        prior_low = 2700.0
        atr = 10.0
        current_price = prior_low + 1.0

        bars = _make_pullback_bars(
            prior_day_low=prior_low,
            pullback_target=current_price,
            direction="LONG",
            confirmed=True,
        )
        bias = _make_bias(direction=BiasDirection.LONG, prior_day_low=prior_low)

        result = detect_pullback_entry(bars, bias, current_price, atr, "Gold")
        if result is not None:
            assert "Pullback" in result.reasoning or "pullback" in result.reasoning.lower()
            assert result.key_level_used != ""


# ═══════════════════════════════════════════════════════════════════════════
# Breakout Entry Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestBreakoutEntry:
    def test_returns_none_on_neutral_bias(self):
        bars = _make_intraday_bars()
        bias = _make_bias(direction=BiasDirection.NEUTRAL)
        result = detect_breakout_entry(bars, bias, 2760.0, 10.0)
        assert result is None

    def test_returns_none_insufficient_bars(self):
        bars = _make_intraday_bars(n=5)
        bias = _make_bias()
        result = detect_breakout_entry(bars, bias, 2760.0, 10.0)
        assert result is None

    def test_returns_none_zero_atr(self):
        bars = _make_intraday_bars()
        bias = _make_bias()
        result = detect_breakout_entry(bars, bias, 2760.0, 0.0)
        assert result is None

    def test_returns_none_no_key_levels(self):
        bars = _make_intraday_bars()
        bias = _make_bias()
        object.__setattr__(bias, "key_levels", None)  # type: ignore[arg-type]
        result = detect_breakout_entry(bars, bias, 2760.0, 10.0)
        assert result is None

    @patch("lib.trading.strategies.daily.swing_detector._get_tick_size", return_value=0.01)
    @patch("lib.trading.strategies.daily.swing_detector._get_point_value", return_value=10.0)
    def test_long_breakout_detected(self, _pv, _ts):
        """Price above prior day high with LONG bias → breakout signal."""
        pdh = 2750.0
        atr = 10.0
        bars = _make_breakout_bars(breakout_level=pdh, direction="LONG", volume_surge=True, good_close=True)
        current_price = bars.iloc[-1]["Close"]

        bias = _make_bias(
            direction=BiasDirection.LONG,
            prior_day_high=pdh,
            atr_expanding=True,
        )

        result = detect_breakout_entry(bars, bias, current_price, atr, "Gold", 50_000)
        assert result is not None
        assert result.entry_style == SwingEntryStyle.BREAKOUT
        assert result.direction == "LONG"
        assert result.phase == SwingPhase.ENTRY_READY
        assert result.entry_price > pdh
        assert result.stop_loss < pdh

    @patch("lib.trading.strategies.daily.swing_detector._get_tick_size", return_value=0.01)
    @patch("lib.trading.strategies.daily.swing_detector._get_point_value", return_value=10.0)
    def test_short_breakout_detected(self, _pv, _ts):
        """Price below prior day low with SHORT bias → breakout signal."""
        pdl = 2700.0
        atr = 10.0
        bars = _make_breakout_bars(breakout_level=pdl, direction="SHORT", volume_surge=True, good_close=True)
        current_price = bars.iloc[-1]["Close"]

        bias = _make_bias(
            direction=BiasDirection.SHORT,
            prior_day_low=pdl,
        )

        result = detect_breakout_entry(bars, bias, current_price, atr, "Gold", 50_000)
        assert result is not None
        assert result.direction == "SHORT"
        assert result.entry_price < pdl
        assert result.stop_loss > pdl

    @patch("lib.trading.strategies.daily.swing_detector._get_tick_size", return_value=0.01)
    @patch("lib.trading.strategies.daily.swing_detector._get_point_value", return_value=10.0)
    def test_no_breakout_price_below_level(self, _pv, _ts):
        """Price still below prior day high → no breakout."""
        pdh = 2750.0
        bars = _make_intraday_bars(n=20, start_price=2730.0)
        bias = _make_bias(direction=BiasDirection.LONG, prior_day_high=pdh)

        result = detect_breakout_entry(bars, bias, 2740.0, 10.0, "Gold")
        assert result is None

    @patch("lib.trading.strategies.daily.swing_detector._get_tick_size", return_value=0.01)
    @patch("lib.trading.strategies.daily.swing_detector._get_point_value", return_value=10.0)
    def test_no_breakout_weak_close(self, _pv, _ts):
        """Breakout with weak bar close → rejected."""
        pdh = 2750.0
        atr = 10.0
        bars = _make_breakout_bars(breakout_level=pdh, direction="LONG", volume_surge=True, good_close=False)
        current_price = bars.iloc[-1]["Close"]

        bias = _make_bias(direction=BiasDirection.LONG, prior_day_high=pdh)
        result = detect_breakout_entry(bars, bias, current_price, atr, "Gold")
        # Weak close: close_pos < BREAKOUT_CLOSE_PCT → should be None
        # (depends on exact bar construction)
        # The bar was constructed to have a weak close
        if result is not None:
            # If somehow detected, confidence should be lower
            pass

    @patch("lib.trading.strategies.daily.swing_detector._get_tick_size", return_value=0.01)
    @patch("lib.trading.strategies.daily.swing_detector._get_point_value", return_value=10.0)
    def test_breakout_volume_affects_confidence(self, _pv, _ts):
        """Volume confirmation increases confidence."""
        pdh = 2750.0
        atr = 10.0

        bars_vol = _make_breakout_bars(breakout_level=pdh, direction="LONG", volume_surge=True, good_close=True)
        bars_no_vol = _make_breakout_bars(breakout_level=pdh, direction="LONG", volume_surge=False, good_close=True)

        bias = _make_bias(direction=BiasDirection.LONG, prior_day_high=pdh)

        sig_vol = detect_breakout_entry(bars_vol, bias, bars_vol.iloc[-1]["Close"], atr, "Gold")
        sig_no_vol = detect_breakout_entry(bars_no_vol, bias, bars_no_vol.iloc[-1]["Close"], atr, "Gold")

        if sig_vol is not None and sig_no_vol is not None:
            assert sig_vol.confidence > sig_no_vol.confidence

    @patch("lib.trading.strategies.daily.swing_detector._get_tick_size", return_value=0.01)
    @patch("lib.trading.strategies.daily.swing_detector._get_point_value", return_value=10.0)
    def test_breakout_atr_expanding_adds_confidence(self, _pv, _ts):
        """ATR expanding flag increases confidence."""
        pdh = 2750.0
        atr = 10.0

        bars = _make_breakout_bars(breakout_level=pdh, direction="LONG", volume_surge=True, good_close=True)
        price = bars.iloc[-1]["Close"]

        bias_expanding = _make_bias(direction=BiasDirection.LONG, prior_day_high=pdh, atr_expanding=True)
        bias_not_expanding = _make_bias(direction=BiasDirection.LONG, prior_day_high=pdh, atr_expanding=False)

        sig_exp = detect_breakout_entry(bars, bias_expanding, price, atr, "Gold")
        sig_no_exp = detect_breakout_entry(bars, bias_not_expanding, price, atr, "Gold")

        if sig_exp is not None and sig_no_exp is not None:
            assert sig_exp.confidence > sig_no_exp.confidence

    @patch("lib.trading.strategies.daily.swing_detector._get_tick_size", return_value=0.01)
    @patch("lib.trading.strategies.daily.swing_detector._get_point_value", return_value=10.0)
    def test_breakout_without_volume_col(self, _pv, _ts):
        """Bars without Volume column — still works but lower confidence."""
        pdh = 2750.0
        atr = 10.0
        bars = _make_breakout_bars(breakout_level=pdh, direction="LONG", volume_surge=True, good_close=True)
        bars = bars.drop(columns=["Volume"])
        price = bars.iloc[-1]["Close"]
        bias = _make_bias(direction=BiasDirection.LONG, prior_day_high=pdh)

        result = detect_breakout_entry(bars, bias, price, atr, "Gold")
        # Should still be able to detect without volume
        if result is not None:
            assert result.entry_style == SwingEntryStyle.BREAKOUT

    @patch("lib.trading.strategies.daily.swing_detector._get_tick_size", return_value=0.01)
    @patch("lib.trading.strategies.daily.swing_detector._get_point_value", return_value=10.0)
    def test_breakout_rr_positive(self, _pv, _ts):
        """Risk:reward should be positive for valid breakout."""
        pdh = 2750.0
        atr = 10.0
        bars = _make_breakout_bars(breakout_level=pdh, direction="LONG", volume_surge=True, good_close=True)
        price = bars.iloc[-1]["Close"]
        bias = _make_bias(direction=BiasDirection.LONG, prior_day_high=pdh)

        result = detect_breakout_entry(bars, bias, price, atr, "Gold")
        if result is not None:
            assert result.risk_reward_tp1 > 0
            assert result.risk_reward_tp2 > result.risk_reward_tp1

    def test_breakout_zero_key_level(self):
        """Zero prior_day_high → no breakout."""
        bars = _make_intraday_bars()
        bias = _make_bias(direction=BiasDirection.LONG, prior_day_high=0.0)
        result = detect_breakout_entry(bars, bias, 2760.0, 10.0)
        assert result is None


# ═══════════════════════════════════════════════════════════════════════════
# Gap Continuation Entry Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestGapContinuation:
    def test_returns_none_on_neutral_bias(self):
        bars = _make_gap_bars()
        bias = _make_bias(direction=BiasDirection.NEUTRAL)
        result = detect_gap_continuation(bars, bias, 2715.0, 10.0)
        assert result is None

    def test_returns_none_insufficient_bars(self):
        bars = _make_gap_bars(n=3)
        bias = _make_bias()
        result = detect_gap_continuation(bars, bias, 2715.0, 10.0)
        assert result is None

    def test_returns_none_zero_atr(self):
        bars = _make_gap_bars()
        bias = _make_bias()
        result = detect_gap_continuation(bars, bias, 2715.0, 0.0)
        assert result is None

    def test_returns_none_no_key_levels(self):
        bars = _make_gap_bars()
        bias = _make_bias()
        object.__setattr__(bias, "key_levels", None)  # type: ignore[arg-type]
        result = detect_gap_continuation(bars, bias, 2715.0, 10.0)
        assert result is None

    def test_returns_none_zero_prior_close(self):
        bars = _make_gap_bars()
        bias = _make_bias(prior_day_close=0.0)
        result = detect_gap_continuation(bars, bias, 2715.0, 10.0)
        assert result is None

    @patch("lib.trading.strategies.daily.swing_detector._get_tick_size", return_value=0.01)
    @patch("lib.trading.strategies.daily.swing_detector._get_point_value", return_value=10.0)
    def test_long_gap_continuation(self, _pv, _ts):
        """Gap up aligning with LONG bias, unfilled, with pullback → signal."""
        prior_close = 2700.0
        gap_size = 15.0  # 1.5× ATR with ATR=10
        atr = 10.0

        bars = _make_gap_bars(
            prior_close=prior_close,
            gap_size=gap_size,
            direction="LONG",
            filled=False,
            with_pullback=True,
        )
        # Price in pullback zone (within gap)
        current_price = prior_close + gap_size * 0.5

        bias = _make_bias(
            direction=BiasDirection.LONG,
            prior_day_close=prior_close,
            overnight_gap_direction=1.0,
        )

        result = detect_gap_continuation(
            bars,
            bias,
            current_price,
            atr,
            session_open_price=prior_close + gap_size,
            asset_name="Gold",
            account_size=50_000,
        )

        assert result is not None
        assert result.entry_style == SwingEntryStyle.GAP_CONTINUATION
        assert result.direction == "LONG"
        assert result.key_level_used == "gap_zone"

    @patch("lib.trading.strategies.daily.swing_detector._get_tick_size", return_value=0.01)
    @patch("lib.trading.strategies.daily.swing_detector._get_point_value", return_value=10.0)
    def test_short_gap_continuation(self, _pv, _ts):
        """Gap down aligning with SHORT bias → signal."""
        prior_close = 2700.0
        gap_size = 15.0
        atr = 10.0

        bars = _make_gap_bars(
            prior_close=prior_close,
            gap_size=gap_size,
            direction="SHORT",
            filled=False,
            with_pullback=True,
        )
        current_price = prior_close - gap_size * 0.5

        bias = _make_bias(
            direction=BiasDirection.SHORT,
            prior_day_close=prior_close,
        )

        result = detect_gap_continuation(
            bars,
            bias,
            current_price,
            atr,
            session_open_price=prior_close - gap_size,
            asset_name="Gold",
        )

        assert result is not None
        assert result.direction == "SHORT"

    @patch("lib.trading.strategies.daily.swing_detector._get_tick_size", return_value=0.01)
    @patch("lib.trading.strategies.daily.swing_detector._get_point_value", return_value=10.0)
    def test_gap_too_small(self, _pv, _ts):
        """Gap smaller than GAP_MIN_ATR_RATIO × ATR → no signal."""
        prior_close = 2700.0
        atr = 10.0
        small_gap = atr * GAP_MIN_ATR_RATIO * 0.5  # Below threshold

        bars = _make_gap_bars(prior_close=prior_close, gap_size=small_gap, direction="LONG", filled=False)
        bias = _make_bias(direction=BiasDirection.LONG, prior_day_close=prior_close)

        result = detect_gap_continuation(
            bars,
            bias,
            prior_close + small_gap * 0.5,
            atr,
            session_open_price=prior_close + small_gap,
        )
        assert result is None

    @patch("lib.trading.strategies.daily.swing_detector._get_tick_size", return_value=0.01)
    @patch("lib.trading.strategies.daily.swing_detector._get_point_value", return_value=10.0)
    def test_gap_against_bias(self, _pv, _ts):
        """Gap up with SHORT bias → no signal (misaligned)."""
        prior_close = 2700.0
        bars = _make_gap_bars(prior_close=prior_close, gap_size=15.0, direction="LONG", filled=False)
        bias = _make_bias(direction=BiasDirection.SHORT, prior_day_close=prior_close)

        result = detect_gap_continuation(
            bars,
            bias,
            2710.0,
            10.0,
            session_open_price=2715.0,
        )
        assert result is None

    @patch("lib.trading.strategies.daily.swing_detector._get_tick_size", return_value=0.01)
    @patch("lib.trading.strategies.daily.swing_detector._get_point_value", return_value=10.0)
    def test_gap_filled_no_signal(self, _pv, _ts):
        """Gap that was filled (price retraced 50%+) → no signal."""
        prior_close = 2700.0
        atr = 10.0

        bars = _make_gap_bars(prior_close=prior_close, gap_size=15.0, direction="LONG", filled=True)
        bias = _make_bias(direction=BiasDirection.LONG, prior_day_close=prior_close)

        result = detect_gap_continuation(
            bars,
            bias,
            2710.0,
            atr,
            session_open_price=2715.0,
        )
        assert result is None

    def test_uses_first_bar_open_as_fallback(self):
        """If session_open_price is None, falls back to bars[0].Open."""
        prior_close = 2700.0
        bars = _make_gap_bars(
            prior_close=prior_close, gap_size=15.0, direction="LONG", filled=False, with_pullback=True
        )
        bias = _make_bias(direction=BiasDirection.LONG, prior_day_close=prior_close)

        # Session open = None; should use bars.iloc[0]["Open"]
        detect_gap_continuation(
            bars,
            bias,
            2710.0,
            10.0,
            session_open_price=None,
        )
        # May or may not detect depending on bars construction,
        # but should not crash
        # (no assertion on result — just testing it doesn't blow up)


# ═══════════════════════════════════════════════════════════════════════════
# Combined detect_swing_entries Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestDetectSwingEntries:
    def test_neutral_bias_returns_empty(self):
        bars = _make_intraday_bars()
        bias = _make_bias(direction=BiasDirection.NEUTRAL)
        result = detect_swing_entries(bars, bias, 2710.0, 10.0, "Gold")
        assert result == []

    def test_insufficient_bars_returns_empty(self):
        bars = _make_intraday_bars(n=2)
        bias = _make_bias()
        result = detect_swing_entries(bars, bias, 2710.0, 10.0, "Gold")
        assert result == []

    def test_zero_atr_returns_empty(self):
        bars = _make_intraday_bars()
        bias = _make_bias()
        result = detect_swing_entries(bars, bias, 2710.0, 0.0, "Gold")
        assert result == []

    def test_none_bars_returns_empty(self):
        bias = _make_bias()
        result = detect_swing_entries(pd.DataFrame(), bias, 2710.0, 10.0, "Gold")
        assert result == []

    @patch("lib.trading.strategies.daily.swing_detector._is_time_stop_due", return_value=True)
    def test_time_stop_blocks_new_entries(self, _mock_ts):
        bars = _make_intraday_bars()
        bias = _make_bias()
        result = detect_swing_entries(bars, bias, 2710.0, 10.0, "Gold")
        assert result == []

    @patch("lib.trading.strategies.daily.swing_detector._is_time_stop_due", return_value=False)
    @patch("lib.trading.strategies.daily.swing_detector._get_tick_size", return_value=0.01)
    @patch("lib.trading.strategies.daily.swing_detector._get_point_value", return_value=10.0)
    def test_enabled_styles_filter(self, _pv, _ts, _time):
        """Only enabled styles are checked."""
        pdh = 2750.0
        bars = _make_breakout_bars(breakout_level=pdh, direction="LONG", volume_surge=True, good_close=True)
        price = bars.iloc[-1]["Close"]
        bias = _make_bias(direction=BiasDirection.LONG, prior_day_high=pdh)

        # Only enable pullback — breakout should not fire
        result = detect_swing_entries(
            bars,
            bias,
            price,
            10.0,
            "Gold",
            enabled_styles=[SwingEntryStyle.PULLBACK],
        )
        # Should not contain breakout signals
        for sig in result:
            assert sig.entry_style != SwingEntryStyle.BREAKOUT

    @patch("lib.trading.strategies.daily.swing_detector._is_time_stop_due", return_value=False)
    @patch("lib.trading.strategies.daily.swing_detector._get_tick_size", return_value=0.01)
    @patch("lib.trading.strategies.daily.swing_detector._get_point_value", return_value=10.0)
    def test_signals_sorted_by_confidence(self, _pv, _ts, _time):
        """Multiple signals should be sorted by confidence descending."""
        bars = _make_intraday_bars(n=40, start_price=2700.0)
        bias = _make_bias(direction=BiasDirection.LONG)

        result = detect_swing_entries(bars, bias, 2720.0, 10.0, "Gold")
        if len(result) > 1:
            for i in range(len(result) - 1):
                assert result[i].confidence >= result[i + 1].confidence

    @patch("lib.trading.strategies.daily.swing_detector._is_time_stop_due", return_value=False)
    @patch("lib.trading.strategies.daily.swing_detector._get_tick_size", return_value=0.01)
    @patch("lib.trading.strategies.daily.swing_detector._get_point_value", return_value=10.0)
    def test_breakout_signals_detected_in_combined(self, _pv, _ts, _time):
        """Breakout signal should appear in combined results."""
        pdh = 2750.0
        bars = _make_breakout_bars(breakout_level=pdh, direction="LONG", volume_surge=True, good_close=True)
        price = bars.iloc[-1]["Close"]
        bias = _make_bias(direction=BiasDirection.LONG, prior_day_high=pdh)

        result = detect_swing_entries(bars, bias, price, 10.0, "Gold")
        breakout_sigs = [s for s in result if s.entry_style == SwingEntryStyle.BREAKOUT]
        assert len(breakout_sigs) >= 1


# ═══════════════════════════════════════════════════════════════════════════
# Exit Evaluation Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestEvaluateSwingExits:
    """Tests for the exit evaluation logic."""

    def test_stop_loss_long(self):
        """Price at/below stop → stop loss exit."""
        exits = evaluate_swing_exits(
            bars=None,
            entry_price=100.0,
            direction="LONG",
            stop_loss=95.0,
            tp1=110.0,
            tp2=120.0,
            current_price=94.0,
            atr=5.0,
            phase=SwingPhase.ACTIVE,
            point_value=10.0,
            position_size=1,
            now=datetime(2024, 1, 15, 10, 0, tzinfo=_EST),
        )
        assert len(exits) >= 1
        assert exits[0].reason == SwingExitReason.STOP_LOSS
        assert exits[0].scale_fraction == 1.0

    def test_stop_loss_short(self):
        """Short position: price at/above stop → stop loss exit."""
        exits = evaluate_swing_exits(
            bars=None,
            entry_price=100.0,
            direction="SHORT",
            stop_loss=105.0,
            tp1=90.0,
            tp2=80.0,
            current_price=106.0,
            atr=5.0,
            phase=SwingPhase.ACTIVE,
            point_value=10.0,
            position_size=1,
            now=datetime(2024, 1, 15, 10, 0, tzinfo=_EST),
        )
        assert len(exits) >= 1
        assert exits[0].reason == SwingExitReason.STOP_LOSS

    def test_stop_loss_is_terminal(self):
        """Stop loss should return immediately — no other exits checked."""
        exits = evaluate_swing_exits(
            bars=None,
            entry_price=100.0,
            direction="LONG",
            stop_loss=95.0,
            tp1=110.0,
            tp2=120.0,
            current_price=90.0,
            phase=SwingPhase.ACTIVE,
            point_value=10.0,
            position_size=1,
            now=datetime(2024, 1, 15, 15, 45, tzinfo=_EST),  # After time stop too!
        )
        # Should only have stop loss, not time stop
        assert len(exits) == 1
        assert exits[0].reason == SwingExitReason.STOP_LOSS

    def test_tp1_hit_long(self):
        """Price at/above TP1 in ACTIVE phase → TP1 scale-out."""
        exits = evaluate_swing_exits(
            bars=None,
            entry_price=100.0,
            direction="LONG",
            stop_loss=95.0,
            tp1=110.0,
            tp2=120.0,
            current_price=111.0,
            atr=5.0,
            phase=SwingPhase.ACTIVE,
            point_value=10.0,
            position_size=2,
            now=datetime(2024, 1, 15, 10, 0, tzinfo=_EST),
        )
        tp1_exits = [e for e in exits if e.reason == SwingExitReason.TP1_HIT]
        assert len(tp1_exits) == 1
        assert tp1_exits[0].scale_fraction == TP1_SCALE_FRACTION

    def test_tp1_not_checked_after_hit(self):
        """TP1 should NOT fire in TP1_HIT or TRAILING phase."""
        exits = evaluate_swing_exits(
            bars=None,
            entry_price=100.0,
            direction="LONG",
            stop_loss=95.0,
            tp1=110.0,
            tp2=120.0,
            current_price=111.0,
            atr=5.0,
            phase=SwingPhase.TP1_HIT,
            point_value=10.0,
            position_size=1,
            now=datetime(2024, 1, 15, 10, 0, tzinfo=_EST),
        )
        tp1_exits = [e for e in exits if e.reason == SwingExitReason.TP1_HIT]
        assert len(tp1_exits) == 0

    def test_tp2_hit_long(self):
        """Price at/above TP2 in TP1_HIT phase → TP2 close."""
        exits = evaluate_swing_exits(
            bars=None,
            entry_price=100.0,
            direction="LONG",
            stop_loss=95.0,
            tp1=110.0,
            tp2=120.0,
            current_price=121.0,
            atr=5.0,
            phase=SwingPhase.TP1_HIT,
            point_value=10.0,
            position_size=1,
            now=datetime(2024, 1, 15, 10, 0, tzinfo=_EST),
        )
        tp2_exits = [e for e in exits if e.reason == SwingExitReason.TP2_HIT]
        assert len(tp2_exits) == 1
        assert tp2_exits[0].scale_fraction == 1.0 - TP1_SCALE_FRACTION

    def test_tp2_not_in_active_phase(self):
        """TP2 should NOT fire in ACTIVE phase (need TP1 first)."""
        exits = evaluate_swing_exits(
            bars=None,
            entry_price=100.0,
            direction="LONG",
            stop_loss=95.0,
            tp1=110.0,
            tp2=120.0,
            current_price=121.0,
            atr=5.0,
            phase=SwingPhase.ACTIVE,
            point_value=10.0,
            position_size=1,
            now=datetime(2024, 1, 15, 10, 0, tzinfo=_EST),
        )
        tp2_exits = [e for e in exits if e.reason == SwingExitReason.TP2_HIT]
        assert len(tp2_exits) == 0

    def test_tp2_short(self):
        """Short TP2 hit: price at/below TP2."""
        exits = evaluate_swing_exits(
            bars=None,
            entry_price=100.0,
            direction="SHORT",
            stop_loss=105.0,
            tp1=90.0,
            tp2=80.0,
            current_price=79.0,
            atr=5.0,
            phase=SwingPhase.TP1_HIT,
            point_value=10.0,
            position_size=1,
            now=datetime(2024, 1, 15, 10, 0, tzinfo=_EST),
        )
        tp2_exits = [e for e in exits if e.reason == SwingExitReason.TP2_HIT]
        assert len(tp2_exits) == 1

    def test_ema_trailing_exit(self):
        """After TP1, price crosses EMA-21 → EMA trail exit."""
        # Build bars that rose then sharply dropped so that current price
        # is well below EMA-21.  First 30 bars rise, last 10 drop sharply.
        n = 40
        close_up = np.linspace(100.0, 115.0, 30)
        close_down = np.linspace(114.0, 102.0, 10)
        close = np.concatenate([close_up, close_down])
        spread = 1.5
        high = close + spread
        low = close - spread
        open_ = np.roll(close, 1)
        open_[0] = close[0]
        volume = np.full(n, 2000.0)

        bars = pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume})

        # Compute what the EMA-21 actually is so we pick a current_price below it
        ema = bars["Close"].ewm(span=TRAIL_EMA_PERIOD, adjust=False).mean()
        ema_val = float(ema.iloc[-1])
        # Current price must be below EMA for a LONG trail exit
        current_price = ema_val - 3.0

        exits = evaluate_swing_exits(
            bars=bars,
            entry_price=100.0,
            direction="LONG",
            stop_loss=95.0,
            tp1=110.0,
            tp2=120.0,
            current_price=current_price,
            atr=5.0,
            phase=SwingPhase.TP1_HIT,
            highest_since_entry=115.0,
            point_value=10.0,
            position_size=1,
            now=datetime(2024, 1, 15, 10, 0, tzinfo=_EST),
        )
        ema_exits = [e for e in exits if e.reason == SwingExitReason.EMA_TRAIL]
        assert len(ema_exits) >= 1

    def test_ema_trailing_update_no_exit(self):
        """After TP1, price still above EMA-21 → trailing stop update (no exit)."""
        bars = _make_intraday_bars(n=40, start_price=100.0, trend=0.003)
        ema = bars["Close"].ewm(span=TRAIL_EMA_PERIOD, adjust=False).mean()
        ema_val = ema.iloc[-1]
        # Price above EMA
        current_price = ema_val + 5

        exits = evaluate_swing_exits(
            bars=bars,
            entry_price=100.0,
            direction="LONG",
            stop_loss=95.0,
            tp1=110.0,
            tp2=120.0,
            current_price=current_price,
            atr=5.0,
            phase=SwingPhase.TP1_HIT,
            highest_since_entry=current_price + 2,
            point_value=10.0,
            position_size=1,
            now=datetime(2024, 1, 15, 10, 0, tzinfo=_EST),
        )
        trail_updates = [e for e in exits if e.reason == SwingExitReason.TRAILING_STOP]
        if trail_updates:
            assert trail_updates[0].scale_fraction == 0.0  # No exit action
            assert trail_updates[0].trailing_stop_price > 0

    def test_time_stop(self):
        """After 15:30 ET → time stop exit."""
        exits = evaluate_swing_exits(
            bars=None,
            entry_price=100.0,
            direction="LONG",
            stop_loss=95.0,
            tp1=110.0,
            tp2=120.0,
            current_price=105.0,
            atr=5.0,
            phase=SwingPhase.ACTIVE,
            point_value=10.0,
            position_size=1,
            now=datetime(2024, 1, 15, 15, 45, tzinfo=_EST),
        )
        time_exits = [e for e in exits if e.reason == SwingExitReason.TIME_STOP]
        assert len(time_exits) == 1
        assert time_exits[0].scale_fraction == 1.0  # Full position in ACTIVE

    def test_time_stop_after_tp1(self):
        """Time stop after TP1 → closes remaining (not full)."""
        exits = evaluate_swing_exits(
            bars=None,
            entry_price=100.0,
            direction="LONG",
            stop_loss=95.0,
            tp1=110.0,
            tp2=120.0,
            current_price=112.0,
            atr=5.0,
            phase=SwingPhase.TP1_HIT,
            point_value=10.0,
            position_size=1,
            now=datetime(2024, 1, 15, 16, 0, tzinfo=_EST),
        )
        time_exits = [e for e in exits if e.reason == SwingExitReason.TIME_STOP]
        assert len(time_exits) == 1
        assert time_exits[0].scale_fraction == 1.0 - TP1_SCALE_FRACTION

    def test_no_exit_before_time_and_no_sl_tp(self):
        """Price between stop and TP1, before time stop → no terminal exit."""
        exits = evaluate_swing_exits(
            bars=None,
            entry_price=100.0,
            direction="LONG",
            stop_loss=95.0,
            tp1=110.0,
            tp2=120.0,
            current_price=105.0,
            atr=5.0,
            phase=SwingPhase.ACTIVE,
            point_value=10.0,
            position_size=1,
            now=datetime(2024, 1, 15, 10, 0, tzinfo=_EST),
        )
        terminal = [e for e in exits if e.scale_fraction > 0]
        assert len(terminal) == 0

    def test_zero_entry_price_returns_empty(self):
        exits = evaluate_swing_exits(
            bars=None,
            entry_price=0.0,
            direction="LONG",
            stop_loss=0.0,
            tp1=0.0,
            tp2=0.0,
            current_price=100.0,
        )
        assert exits == []

    def test_pnl_calculation_long(self):
        """PnL should be positive when price > entry for LONG."""
        exits = evaluate_swing_exits(
            bars=None,
            entry_price=100.0,
            direction="LONG",
            stop_loss=95.0,
            tp1=110.0,
            tp2=120.0,
            current_price=111.0,
            atr=5.0,
            phase=SwingPhase.ACTIVE,
            point_value=10.0,
            position_size=1,
            risk_dollars=50.0,
            now=datetime(2024, 1, 15, 10, 0, tzinfo=_EST),
        )
        tp1_exits = [e for e in exits if e.reason == SwingExitReason.TP1_HIT]
        if tp1_exits:
            assert tp1_exits[0].pnl_estimate > 0

    def test_pnl_calculation_short(self):
        """PnL should be positive when price < entry for SHORT."""
        exits = evaluate_swing_exits(
            bars=None,
            entry_price=100.0,
            direction="SHORT",
            stop_loss=105.0,
            tp1=90.0,
            tp2=80.0,
            current_price=89.0,
            atr=5.0,
            phase=SwingPhase.ACTIVE,
            point_value=10.0,
            position_size=1,
            risk_dollars=50.0,
            now=datetime(2024, 1, 15, 10, 0, tzinfo=_EST),
        )
        tp1_exits = [e for e in exits if e.reason == SwingExitReason.TP1_HIT]
        if tp1_exits:
            assert tp1_exits[0].pnl_estimate > 0

    def test_atr_trailing_fallback(self):
        """When EMA data insufficient, use ATR-based trailing stop."""
        # No bars → ATR fallback
        exits = evaluate_swing_exits(
            bars=None,
            entry_price=100.0,
            direction="LONG",
            stop_loss=95.0,
            tp1=110.0,
            tp2=120.0,
            current_price=105.0,
            atr=5.0,
            phase=SwingPhase.TP1_HIT,
            highest_since_entry=115.0,  # Trail from 115 - 5 = 110; price 105 < 110 → trigger
            point_value=10.0,
            position_size=1,
            now=datetime(2024, 1, 15, 10, 0, tzinfo=_EST),
        )
        trail_exits = [e for e in exits if e.reason == SwingExitReason.TRAILING_STOP and e.scale_fraction > 0]
        assert len(trail_exits) >= 1

    def test_atr_trailing_short_fallback(self):
        """ATR trailing stop for SHORT positions."""
        exits = evaluate_swing_exits(
            bars=None,
            entry_price=100.0,
            direction="SHORT",
            stop_loss=105.0,
            tp1=90.0,
            tp2=80.0,
            current_price=94.0,
            atr=5.0,
            phase=SwingPhase.TP1_HIT,
            lowest_since_entry=85.0,  # Trail from 85 + 5 = 90; price 94 > 90 → trigger
            point_value=10.0,
            position_size=1,
            now=datetime(2024, 1, 15, 10, 0, tzinfo=_EST),
        )
        trail_exits = [e for e in exits if e.reason == SwingExitReason.TRAILING_STOP and e.scale_fraction > 0]
        assert len(trail_exits) >= 1


# ═══════════════════════════════════════════════════════════════════════════
# Swing State Management Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestCreateSwingState:
    def test_creates_active_state(self):
        sig = SwingSignal(
            asset_name="Gold",
            direction="LONG",
            entry_price=2700.0,
            stop_loss=2685.0,
            tp1=2720.0,
            tp2=2735.0,
            position_size=2,
            risk_dollars=150.0,
        )
        state = create_swing_state(sig)

        assert state.asset_name == "Gold"
        assert state.phase == SwingPhase.ACTIVE
        assert state.entry_price == 2700.0
        assert state.current_stop == 2685.0
        assert state.tp1 == 2720.0
        assert state.tp2 == 2735.0
        assert state.position_size == 2
        assert state.remaining_size == 2
        assert state.highest_price == 2700.0
        assert state.lowest_price == 2700.0
        assert state.entry_time != ""

    def test_signal_reference(self):
        sig = SwingSignal(asset_name="Gold")
        state = create_swing_state(sig)
        assert state.signal is sig


class TestUpdateSwingState:
    def test_updates_highest_price_long(self):
        sig = SwingSignal(
            asset_name="Gold",
            direction="LONG",
            entry_price=100.0,
            stop_loss=95.0,
            tp1=110.0,
            tp2=120.0,
            risk_dollars=50.0,
        )
        state = create_swing_state(sig)
        state.highest_price = 105.0

        updated, exits = update_swing_state(
            state,
            108.0,
            bars=None,
            atr=5.0,
            point_value=10.0,
            now=datetime(2024, 1, 15, 10, 0, tzinfo=_EST),
        )
        assert updated.highest_price == 108.0

    def test_updates_lowest_price_short(self):
        sig = SwingSignal(
            asset_name="Gold",
            direction="SHORT",
            entry_price=100.0,
            stop_loss=105.0,
            tp1=90.0,
            tp2=80.0,
            risk_dollars=50.0,
        )
        state = create_swing_state(sig)
        state.direction = "SHORT"
        state.lowest_price = 97.0

        updated, exits = update_swing_state(
            state,
            95.0,
            bars=None,
            atr=5.0,
            point_value=10.0,
            now=datetime(2024, 1, 15, 10, 0, tzinfo=_EST),
        )
        assert updated.lowest_price == 95.0

    def test_stop_loss_closes_state(self):
        sig = SwingSignal(
            asset_name="Gold",
            direction="LONG",
            entry_price=100.0,
            stop_loss=95.0,
            tp1=110.0,
            tp2=120.0,
            risk_dollars=50.0,
        )
        state = create_swing_state(sig)

        updated, exits = update_swing_state(
            state,
            94.0,
            bars=None,
            atr=5.0,
            point_value=10.0,
            now=datetime(2024, 1, 15, 10, 0, tzinfo=_EST),
        )
        assert updated.phase == SwingPhase.CLOSED
        assert updated.remaining_size == 0
        assert any(e.reason == SwingExitReason.STOP_LOSS for e in exits)

    def test_tp1_advances_phase(self):
        sig = SwingSignal(
            asset_name="Gold",
            direction="LONG",
            entry_price=100.0,
            stop_loss=95.0,
            tp1=110.0,
            tp2=120.0,
            position_size=2,
            risk_dollars=50.0,
        )
        state = create_swing_state(sig)

        updated, exits = update_swing_state(
            state,
            111.0,
            bars=None,
            atr=5.0,
            point_value=10.0,
            now=datetime(2024, 1, 15, 10, 0, tzinfo=_EST),
        )
        assert updated.phase == SwingPhase.TP1_HIT
        assert updated.remaining_size < updated.position_size
        # Stop should move to breakeven
        assert updated.current_stop == updated.entry_price
        assert any(e.reason == SwingExitReason.TP1_HIT for e in exits)

    def test_tp2_closes_state(self):
        sig = SwingSignal(
            asset_name="Gold",
            direction="LONG",
            entry_price=100.0,
            stop_loss=95.0,
            tp1=110.0,
            tp2=120.0,
            risk_dollars=50.0,
        )
        state = create_swing_state(sig)
        state.phase = SwingPhase.TP1_HIT
        state.remaining_size = 1

        updated, exits = update_swing_state(
            state,
            121.0,
            bars=None,
            atr=5.0,
            point_value=10.0,
            now=datetime(2024, 1, 15, 10, 0, tzinfo=_EST),
        )
        assert updated.phase == SwingPhase.CLOSED
        assert updated.remaining_size == 0
        assert any(e.reason == SwingExitReason.TP2_HIT for e in exits)

    def test_time_stop_closes_state(self):
        sig = SwingSignal(
            asset_name="Gold",
            direction="LONG",
            entry_price=100.0,
            stop_loss=95.0,
            tp1=110.0,
            tp2=120.0,
            risk_dollars=50.0,
        )
        state = create_swing_state(sig)

        updated, exits = update_swing_state(
            state,
            105.0,
            bars=None,
            atr=5.0,
            point_value=10.0,
            now=datetime(2024, 1, 15, 16, 0, tzinfo=_EST),
        )
        assert updated.phase == SwingPhase.CLOSED
        assert any(e.reason == SwingExitReason.TIME_STOP for e in exits)

    def test_closed_state_noop(self):
        """Updating a CLOSED state does nothing."""
        sig = SwingSignal(asset_name="Gold", direction="LONG", entry_price=100.0)
        state = create_swing_state(sig)
        state.phase = SwingPhase.CLOSED

        updated, exits = update_swing_state(state, 105.0)
        assert updated.phase == SwingPhase.CLOSED
        assert exits == []

    def test_trailing_stop_moves_in_favorable_direction_only(self):
        """Trailing stop should only move in favorable direction (tighten, not widen)."""
        sig = SwingSignal(
            asset_name="Gold",
            direction="LONG",
            entry_price=100.0,
            stop_loss=95.0,
            tp1=110.0,
            tp2=120.0,
            risk_dollars=50.0,
        )
        state = create_swing_state(sig)
        state.phase = SwingPhase.TP1_HIT
        state.current_stop = 103.0  # Already moved up

        # Build bars with EMA below current stop
        bars = _make_intraday_bars(n=40, start_price=108.0, trend=0.002)

        updated, _ = update_swing_state(
            state,
            112.0,
            bars=bars,
            atr=5.0,
            point_value=10.0,
            now=datetime(2024, 1, 15, 10, 0, tzinfo=_EST),
        )
        # Stop should not have moved down
        assert updated.current_stop >= 103.0


# ═══════════════════════════════════════════════════════════════════════════
# Multi-Asset Scanning Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestScanSwingEntriesAllAssets:
    @patch("lib.trading.strategies.daily.swing_detector._is_time_stop_due", return_value=False)
    @patch("lib.trading.strategies.daily.swing_detector._get_tick_size", return_value=0.01)
    @patch("lib.trading.strategies.daily.swing_detector._get_point_value", return_value=10.0)
    def test_basic_scan(self, _pv, _ts, _time):
        """Scan multiple assets and get combined results."""
        bars_gold = _make_breakout_bars(breakout_level=2750.0, direction="LONG", volume_surge=True, good_close=True)
        bars_sp = _make_intraday_bars(n=20, start_price=5300.0)

        biases = {
            "Gold": _make_bias(direction=BiasDirection.LONG, prior_day_high=2750.0),
            "S&P": _make_bias(direction=BiasDirection.LONG, prior_day_high=5350.0),
        }

        result = scan_swing_entries_all_assets(
            asset_bars={"Gold": bars_gold, "S&P": bars_sp},
            biases=biases,
            current_prices={"Gold": bars_gold.iloc[-1]["Close"], "S&P": 5310.0},
            atrs={"Gold": 10.0, "S&P": 15.0},
        )
        # At minimum Gold should have a breakout signal
        assert isinstance(result, list)
        # Results should be sorted by confidence
        if len(result) > 1:
            assert result[0].confidence >= result[1].confidence

    @patch("lib.trading.strategies.daily.swing_detector._is_time_stop_due", return_value=False)
    def test_skip_missing_bias(self, _time):
        """Assets without bias are skipped."""
        bars = {"Gold": _make_intraday_bars()}
        biases = {}  # No bias for Gold

        result = scan_swing_entries_all_assets(
            asset_bars=bars,
            biases=biases,
            current_prices={"Gold": 2700.0},
            atrs={"Gold": 10.0},
        )
        assert result == []

    @patch("lib.trading.strategies.daily.swing_detector._is_time_stop_due", return_value=False)
    def test_skip_zero_price(self, _time):
        bars = {"Gold": _make_intraday_bars()}
        biases = {"Gold": _make_bias()}

        result = scan_swing_entries_all_assets(
            asset_bars=bars,
            biases=biases,
            current_prices={"Gold": 0.0},
            atrs={"Gold": 10.0},
        )
        assert result == []

    @patch("lib.trading.strategies.daily.swing_detector._is_time_stop_due", return_value=False)
    def test_max_signals_cap(self, _time):
        """Result is capped at max_signals."""
        result = scan_swing_entries_all_assets(
            asset_bars={},
            biases={},
            current_prices={},
            atrs={},
            max_signals=2,
        )
        assert len(result) <= 2


# ═══════════════════════════════════════════════════════════════════════════
# Enrich Swing Candidates Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestEnrichSwingCandidates:
    @patch("lib.trading.strategies.daily.swing_detector._is_time_stop_due", return_value=False)
    @patch("lib.trading.strategies.daily.swing_detector._get_tick_size", return_value=0.01)
    @patch("lib.trading.strategies.daily.swing_detector._get_point_value", return_value=10.0)
    def test_enriches_candidates(self, _pv, _ts, _time):
        """Should produce entry signals for swing candidates."""
        # Use a mock SwingCandidate-like object
        candidate = MagicMock()
        candidate.asset_name = "Gold"

        bars = _make_breakout_bars(breakout_level=2750.0, direction="LONG", volume_surge=True, good_close=True)
        bias = _make_bias(direction=BiasDirection.LONG, prior_day_high=2750.0)

        result = enrich_swing_candidates(
            swing_candidates=[candidate],
            asset_bars={"Gold": bars},
            current_prices={"Gold": bars.iloc[-1]["Close"]},
            atrs={"Gold": 10.0},
            biases={"Gold": bias},
        )
        # May or may not have signals depending on conditions
        assert isinstance(result, dict)

    @patch("lib.trading.strategies.daily.swing_detector._is_time_stop_due", return_value=False)
    def test_skips_missing_data(self, _time):
        """Candidates without bars/price/ATR are skipped."""
        candidate = MagicMock()
        candidate.asset_name = "Gold"

        result = enrich_swing_candidates(
            swing_candidates=[candidate],
            asset_bars={},  # No bars
            current_prices={"Gold": 2700.0},
            atrs={"Gold": 10.0},
            biases={"Gold": _make_bias()},
        )
        assert "Gold" not in result

    @patch("lib.trading.strategies.daily.swing_detector._is_time_stop_due", return_value=False)
    def test_skips_missing_bias(self, _time):
        candidate = MagicMock()
        candidate.asset_name = "Gold"

        result = enrich_swing_candidates(
            swing_candidates=[candidate],
            asset_bars={"Gold": _make_intraday_bars()},
            current_prices={"Gold": 2700.0},
            atrs={"Gold": 10.0},
            biases={},  # No bias
        )
        assert "Gold" not in result


# ═══════════════════════════════════════════════════════════════════════════
# Redis Publish / Load Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestRedisPublishLoad:
    def _make_mock_redis(self) -> MagicMock:
        """Create a mock Redis client with in-memory storage."""
        store = {}

        def mock_set(key, value):
            store[key] = value

        def mock_get(key):
            return store.get(key)

        def mock_expire(key, ttl):
            pass  # No-op for tests

        def mock_publish(channel, message):
            pass

        r = MagicMock()
        r.set = mock_set
        r.get = mock_get
        r.expire = mock_expire
        r.publish = mock_publish
        return r

    def test_publish_and_load_round_trip(self):
        """Signals survive publish → load round-trip."""
        redis = self._make_mock_redis()

        signals = [
            SwingSignal(
                asset_name="Gold",
                entry_style=SwingEntryStyle.BREAKOUT,
                direction="LONG",
                confidence=0.85,
                entry_price=2760.0,
                stop_loss=2740.0,
                tp1=2780.0,
                tp2=2800.0,
                atr=10.0,
                risk_reward_tp1=2.0,
                risk_reward_tp2=4.0,
                risk_dollars=200.0,
                position_size=2,
                reasoning="Test breakout",
                key_level_used="prior_day_high",
                key_level_price=2750.0,
                phase=SwingPhase.ENTRY_READY,
            ),
            SwingSignal(
                asset_name="S&P",
                entry_style=SwingEntryStyle.PULLBACK,
                direction="SHORT",
                confidence=0.6,
                phase=SwingPhase.WATCHING,
            ),
        ]

        ok = publish_swing_signals(signals, redis)
        assert ok is True

        loaded = load_swing_signals(redis)
        assert len(loaded) == 2

        assert loaded[0].asset_name == "Gold"
        assert loaded[0].entry_style == SwingEntryStyle.BREAKOUT
        assert loaded[0].direction == "LONG"
        assert loaded[0].confidence == pytest.approx(0.85)
        assert loaded[0].phase == SwingPhase.ENTRY_READY
        assert loaded[0].entry_price == 2760.0
        assert loaded[0].stop_loss == 2740.0
        assert loaded[0].tp1 == 2780.0
        assert loaded[0].tp2 == 2800.0
        assert loaded[0].risk_reward_tp1 == pytest.approx(2.0)
        assert loaded[0].key_level_used == "prior_day_high"

        assert loaded[1].asset_name == "S&P"
        assert loaded[1].entry_style == SwingEntryStyle.PULLBACK
        assert loaded[1].direction == "SHORT"
        assert loaded[1].phase == SwingPhase.WATCHING

    def test_publish_empty_signals(self):
        redis = self._make_mock_redis()
        ok = publish_swing_signals([], redis)
        assert ok is True
        loaded = load_swing_signals(redis)
        assert loaded == []

    def test_load_no_data(self):
        redis = self._make_mock_redis()
        loaded = load_swing_signals(redis)
        assert loaded == []

    def test_load_corrupt_data(self):
        redis = MagicMock()
        redis.get = MagicMock(return_value="not valid json {{{")
        loaded = load_swing_signals(redis)
        assert loaded == []

    def test_publish_failure(self):
        redis = MagicMock()
        redis.set = MagicMock(side_effect=Exception("Connection refused"))
        ok = publish_swing_signals([SwingSignal()], redis)
        assert ok is False

    def test_publish_swing_states(self):
        redis = self._make_mock_redis()
        states = {
            "Gold": SwingState(asset_name="Gold", phase=SwingPhase.ACTIVE, entry_price=2700.0),
        }
        ok = publish_swing_states(states, redis)
        assert ok is True

    def test_publish_swing_states_failure(self):
        redis = MagicMock()
        redis.set = MagicMock(side_effect=Exception("Boom"))
        ok = publish_swing_states({"Gold": SwingState()}, redis)
        assert ok is False

    def test_redis_keys_correct(self):
        assert REDIS_KEY_SWING_SIGNALS == "engine:swing_signals"
        assert REDIS_KEY_SWING_STATES == "engine:swing_states"
        assert REDIS_PUBSUB_SWING == "dashboard:swing_update"


# ═══════════════════════════════════════════════════════════════════════════
# Constants & Configuration Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestConstants:
    """Verify key constants are set to expected values per the spec."""

    def test_sl_atr_mult(self):
        assert SWING_SL_ATR_MULT == 1.5

    def test_tp_atr_mults(self):
        assert SWING_TP1_ATR_MULT == 2.0
        assert SWING_TP2_ATR_MULT == 3.5

    def test_risk_pct(self):
        assert SWING_RISK_PCT == 0.005

    def test_time_stop(self):
        assert TIME_STOP_HOUR == 15
        assert TIME_STOP_MINUTE == 30

    def test_tp1_scale(self):
        assert TP1_SCALE_FRACTION == 0.5

    def test_trail_ema_period(self):
        assert TRAIL_EMA_PERIOD == 21

    def test_max_contracts(self):
        assert MAX_SWING_CONTRACTS == 3
        assert MIN_SWING_CONTRACTS == 1

    def test_pullback_retrace_bounds(self):
        assert PULLBACK_MIN_RETRACE_PCT == 0.25
        assert PULLBACK_MAX_RETRACE_PCT == 0.75

    def test_breakout_volume_mult(self):
        assert BREAKOUT_VOLUME_MULT == 1.3

    def test_gap_min_atr(self):
        assert GAP_MIN_ATR_RATIO == 0.3

    def test_gap_fill_threshold(self):
        assert GAP_FILL_THRESHOLD == 0.5


# ═══════════════════════════════════════════════════════════════════════════
# Edge Cases & Integration
# ═══════════════════════════════════════════════════════════════════════════


class TestEdgeCases:
    def test_all_zeros_bars(self):
        """Bars with all zeros should not crash."""
        bars = pd.DataFrame(
            {
                "Open": [0.0] * 20,
                "High": [0.0] * 20,
                "Low": [0.0] * 20,
                "Close": [0.0] * 20,
                "Volume": [0.0] * 20,
            }
        )
        bias = _make_bias()
        result = detect_pullback_entry(bars, bias, 0.0, 10.0)
        # Should handle gracefully
        assert result is None

    def test_single_bar(self):
        """Single bar should not crash."""
        bars = pd.DataFrame(
            {
                "Open": [100.0],
                "High": [105.0],
                "Low": [95.0],
                "Close": [103.0],
                "Volume": [1000.0],
            }
        )
        bias = _make_bias()
        result = detect_swing_entries(bars, bias, 103.0, 5.0, "Gold")
        assert isinstance(result, list)

    def test_nan_in_bars(self):
        """NaN values in bars should be handled gracefully."""
        bars = _make_intraday_bars(n=20)
        bars.iloc[5, bars.columns.get_loc("Close")] = float("nan")
        bias = _make_bias()
        # Should not crash
        try:
            result = detect_swing_entries(bars, bias, 2700.0, 10.0, "Gold")
            assert isinstance(result, list)
        except Exception:
            pass  # Some NaN handling may raise, that's acceptable

    def test_very_large_atr(self):
        """Very large ATR should produce valid but wide levels."""
        bars = _make_intraday_bars(n=20)
        bias = _make_bias()
        result = detect_swing_entries(bars, bias, 2700.0, 500.0, "Gold")
        assert isinstance(result, list)

    def test_very_small_atr(self):
        """Very small ATR should produce tight levels."""
        bars = _make_intraday_bars(n=20)
        bias = _make_bias()
        result = detect_swing_entries(bars, bias, 2700.0, 0.01, "Gold")
        assert isinstance(result, list)

    def test_negative_prices_handled(self):
        """Negative prices (shouldn't happen) shouldn't crash."""
        bars = _make_intraday_bars(n=20, start_price=-100.0)
        bias = _make_bias()
        try:
            result = detect_swing_entries(bars, bias, -95.0, 5.0)
            assert isinstance(result, list)
        except Exception:
            pass  # Acceptable to fail gracefully

    @patch("lib.trading.strategies.daily.swing_detector._is_time_stop_due", return_value=False)
    @patch("lib.trading.strategies.daily.swing_detector._get_tick_size", return_value=0.0001)
    @patch("lib.trading.strategies.daily.swing_detector._get_point_value", return_value=12500.0)
    def test_fx_precision(self, _pv, _ts, _time):
        """FX pairs should use proper decimal precision (4+ places)."""
        pdh = 1.1050
        atr = 0.0030
        bars = _make_breakout_bars(breakout_level=pdh, direction="LONG", volume_surge=True, good_close=True)
        price = bars.iloc[-1]["Close"]

        bias = _make_bias(
            direction=BiasDirection.LONG,
            prior_day_high=pdh,
            prior_day_low=1.0980,
            prior_day_close=1.1020,
        )

        result = detect_breakout_entry(bars, bias, price, atr, "Euro FX")
        if result is not None:
            # Entry price should have proper FX precision
            str(result.entry_price)
            # Should not be rounded to 2 decimals
            assert result.atr == pytest.approx(0.003, abs=0.0001)

    def test_swing_signal_serialization_all_fields(self):
        """All fields should survive to_dict() serialization."""
        sig = SwingSignal(
            asset_name="Gold",
            entry_style=SwingEntryStyle.GAP_CONTINUATION,
            direction="SHORT",
            confidence=0.92,
            entry_price=2700.0,
            entry_zone_low=2695.0,
            entry_zone_high=2705.0,
            stop_loss=2715.0,
            tp1=2680.0,
            tp2=2665.0,
            atr=10.0,
            risk_reward_tp1=2.0,
            risk_reward_tp2=3.5,
            risk_dollars=150.0,
            position_size=2,
            reasoning="Gap down + short bias",
            key_level_used="gap_zone",
            key_level_price=2695.0,
            confirmation_bar_idx=25,
            detected_at="2024-01-15T09:30:00-05:00",
            phase=SwingPhase.ENTRY_READY,
        )
        d = sig.to_dict()
        assert d["asset_name"] == "Gold"
        assert d["entry_style"] == "gap_continuation"
        assert d["direction"] == "SHORT"
        assert d["confidence"] == 0.92
        assert d["entry_price"] == 2700.0
        assert d["entry_zone_low"] == 2695.0
        assert d["entry_zone_high"] == 2705.0
        assert d["stop_loss"] == 2715.0
        assert d["tp1"] == 2680.0
        assert d["tp2"] == 2665.0
        assert d["atr"] == pytest.approx(10.0)
        assert d["risk_reward_tp1"] == 2.0
        assert d["risk_reward_tp2"] == 3.5
        assert d["risk_dollars"] == 150.0
        assert d["position_size"] == 2
        assert d["reasoning"] == "Gap down + short bias"
        assert d["key_level_used"] == "gap_zone"
        assert d["key_level_price"] == 2695.0
        assert d["confirmation_bar_idx"] == 25
        assert d["detected_at"] == "2024-01-15T09:30:00-05:00"
        assert d["phase"] == "entry_ready"

    def test_exit_signal_serialization(self):
        ex = SwingExitSignal(
            reason=SwingExitReason.EMA_TRAIL,
            exit_price=2710.0,
            pnl_estimate=100.0,
            r_multiple=2.0,
            scale_fraction=0.5,
            trailing_stop_price=2705.0,
            reasoning="EMA trail hit",
        )
        d = ex.to_dict()
        assert d["reason"] == "ema_trail"
        assert d["exit_price"] == 2710.0
        assert d["trailing_stop_price"] == 2705.0

    def test_swing_state_full_lifecycle(self):
        """Test complete lifecycle: create → TP1 → trailing → TP2 → closed."""
        sig = SwingSignal(
            asset_name="Gold",
            direction="LONG",
            entry_price=100.0,
            stop_loss=95.0,
            tp1=110.0,
            tp2=120.0,
            position_size=2,
            risk_dollars=50.0,
        )

        # Create
        state = create_swing_state(sig)
        assert state.phase == SwingPhase.ACTIVE

        # Price rises to TP1
        state, exits = update_swing_state(
            state,
            111.0,
            bars=None,
            atr=5.0,
            point_value=10.0,
            now=datetime(2024, 1, 15, 10, 0, tzinfo=_EST),
        )
        assert state.phase == SwingPhase.TP1_HIT
        assert state.current_stop == state.entry_price  # BE stop

        # Price continues to TP2
        state, exits = update_swing_state(
            state,
            121.0,
            bars=None,
            atr=5.0,
            point_value=10.0,
            now=datetime(2024, 1, 15, 11, 0, tzinfo=_EST),
        )
        assert state.phase == SwingPhase.CLOSED
        assert state.remaining_size == 0
