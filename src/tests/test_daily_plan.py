"""
Tests for Phase 2B (Daily Plan Generator) and Phase 3A (Focus Asset Selection).

Covers:
  - DailyPlan / SwingCandidate / ScalpFocusAsset dataclass creation & serialization
  - select_daily_focus_assets() composite ranking logic
  - generate_daily_plan() orchestration
  - Swing candidate level computation (wider SL/TP than scalp)
  - Session fit scoring
  - ATR opportunity scoring
  - Redis publish / load round-trip
  - Edge cases: empty data, no directional bias, all neutral
  - Integration with bias_analyzer (Phase 2A)
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Imports under test
# ---------------------------------------------------------------------------
from lib.trading.strategies.daily.bias_analyzer import (
    BiasDirection,
    CandlePattern,
    DailyBias,
    KeyLevels,
    compute_all_daily_biases,
    compute_daily_bias,
)
from lib.trading.strategies.daily.daily_plan import (
    ASSET_SESSION_MAP,
    MAX_SCALP_FOCUS,
    MAX_SWING_CANDIDATES,
    MIN_SCALP_SCORE,
    MIN_SWING_SCORE,
    SWING_SL_ATR_MULT,
    SWING_TP1_ATR_MULT,
    SWING_TP2_ATR_MULT,
    SWING_TP3_ATR_MULT,
    DailyPlan,
    ScalpFocusAsset,
    SwingCandidate,
    _build_swing_candidate,
    _compute_atr_opportunity_score,
    _compute_session_fit_score,
    _get_current_session,
    _safe_float,
    generate_daily_plan,
    select_daily_focus_assets,
)

_EST = ZoneInfo("America/New_York")


# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------


def _make_daily_bars(
    n: int = 30,
    start_price: float = 2700.0,
    trend: float = 0.002,
    seed: int = 42,
) -> pd.DataFrame:
    """Build a synthetic daily OHLCV DataFrame."""
    rng = np.random.default_rng(seed)
    returns = rng.normal(trend, 0.01, n)
    close = start_price * np.exp(np.cumsum(returns))

    spread = close * rng.uniform(0.003, 0.012, n)
    high = close + rng.uniform(0, 1, n) * spread
    low = close - rng.uniform(0, 1, n) * spread
    opn = close + rng.uniform(-0.3, 0.3, n) * spread

    high = np.maximum(high, np.maximum(opn, close))
    low = np.minimum(low, np.minimum(opn, close))

    volume = rng.poisson(50_000, n).astype(float)
    volume = np.maximum(volume, 1)

    idx = pd.date_range(
        start="2025-01-06",
        periods=n,
        freq="B",
        tz="America/New_York",
    )

    return pd.DataFrame(
        {"Open": opn, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=idx,
    )


def _make_weekly_bars(
    n: int = 12,
    start_price: float = 2650.0,
    seed: int = 99,
) -> pd.DataFrame:
    """Build a synthetic weekly OHLCV DataFrame."""
    rng = np.random.default_rng(seed)
    close = start_price + np.cumsum(rng.normal(10, 30, n))
    spread = np.abs(close) * rng.uniform(0.01, 0.03, n)
    high = close + rng.uniform(0.3, 1, n) * spread
    low = close - rng.uniform(0.3, 1, n) * spread
    opn = close + rng.uniform(-0.5, 0.5, n) * spread

    high = np.maximum(high, np.maximum(opn, close))
    low = np.minimum(low, np.minimum(opn, close))

    idx = pd.date_range(start="2024-11-01", periods=n, freq="W-FRI", tz="America/New_York")

    return pd.DataFrame(
        {"Open": opn, "High": high, "Low": low, "Close": close},
        index=idx,
    )


def _make_bias(
    name: str = "Gold",
    direction: BiasDirection = BiasDirection.LONG,
    confidence: float = 0.72,
    atr_expanding: bool = True,
    weekly_pos: float = 0.35,
    vol_confirm: bool = True,
    overnight_gap_dir: float = 0.5,
) -> DailyBias:
    """Build a DailyBias for testing without running full computation."""
    bias = DailyBias(asset_name=name)
    bias.direction = direction
    bias.confidence = confidence
    bias.atr_expanding = atr_expanding
    bias.weekly_range_position = weekly_pos
    bias.volume_confirmation = vol_confirm
    bias.overnight_gap_direction = overnight_gap_dir
    bias.reasoning = f"Test bias for {name}: {direction.value} ({confidence:.0%})"
    bias.candle_pattern = (
        CandlePattern.BULLISH_ENGULFING if direction == BiasDirection.LONG else CandlePattern.BEARISH_ENGULFING
    )
    bias.key_levels = KeyLevels(
        prior_day_high=2720.0,
        prior_day_low=2690.0,
        prior_day_mid=2705.0,
        prior_day_close=2710.0,
        weekly_high=2740.0,
        weekly_low=2670.0,
    )
    return bias


# ---------------------------------------------------------------------------
# Tests: _safe_float
# ---------------------------------------------------------------------------


class TestSafeFloat:
    def test_normal_float(self):
        assert _safe_float(3.14) == 3.14

    def test_string_number(self):
        assert _safe_float("42.5") == 42.5

    def test_none_returns_default(self):
        assert _safe_float(None) == 0.0
        assert _safe_float(None, 99.0) == 99.0

    def test_nan_returns_default(self):
        assert _safe_float(float("nan")) == 0.0

    def test_inf_returns_default(self):
        assert _safe_float(float("inf")) == 0.0

    def test_invalid_string(self):
        assert _safe_float("not_a_number") == 0.0


# ---------------------------------------------------------------------------
# Tests: Session fit scoring
# ---------------------------------------------------------------------------


class TestSessionFitScore:
    @patch("lib.trading.strategies.daily.daily_plan._get_current_session", return_value="us")
    def test_primary_session_match(self, _mock):
        # S&P's primary session is "us"
        score = _compute_session_fit_score("S&P")
        assert score == 100.0

    @patch("lib.trading.strategies.daily.daily_plan._get_current_session", return_value="us")
    def test_secondary_session_match(self, _mock):
        # Gold has sessions ("london", "us") — "us" is secondary
        score = _compute_session_fit_score("Gold")
        assert score == 70.0

    @patch("lib.trading.strategies.daily.daily_plan._get_current_session", return_value="london")
    def test_primary_london(self, _mock):
        # Gold's primary is "london"
        score = _compute_session_fit_score("Gold")
        assert score == 100.0

    @patch("lib.trading.strategies.daily.daily_plan._get_current_session", return_value="asian")
    def test_no_session_match(self, _mock):
        # S&P only has ("us",) — Asian session is a mismatch
        score = _compute_session_fit_score("S&P")
        assert score == 20.0

    @patch("lib.trading.strategies.daily.daily_plan._get_current_session", return_value="off-hours")
    def test_off_hours(self, _mock):
        score = _compute_session_fit_score("Gold")
        assert score == 10.0

    @patch("lib.trading.strategies.daily.daily_plan._get_current_session", return_value="us")
    def test_unknown_asset(self, _mock):
        score = _compute_session_fit_score("UnknownAssetXYZ")
        assert score == 30.0


# ---------------------------------------------------------------------------
# Tests: ATR opportunity scoring
# ---------------------------------------------------------------------------


class TestAtrOpportunityScore:
    def test_zero_price_returns_low(self):
        assert _compute_atr_opportunity_score(1.0, 0.0) == 10.0

    def test_zero_atr_returns_low(self):
        assert _compute_atr_opportunity_score(0.0, 100.0) == 10.0

    def test_very_tight_asset(self):
        # NATR = 0.01% — very tight
        score = _compute_atr_opportunity_score(0.01, 100.0)
        assert 10.0 <= score <= 25.0

    def test_moderate_asset(self):
        # NATR = 0.2% — typical equity index
        score = _compute_atr_opportunity_score(10.0, 5000.0)
        assert 25.0 <= score <= 55.0

    def test_good_opportunity(self):
        # NATR = 0.5% — metals/oil territory
        score = _compute_atr_opportunity_score(13.5, 2700.0)
        assert 55.0 <= score <= 80.0

    def test_excellent_opportunity(self):
        # NATR = 1.0% — volatile day / crypto
        score = _compute_atr_opportunity_score(10.0, 1000.0)
        assert 80.0 <= score <= 100.0

    def test_extreme_atr(self):
        # NATR = 3% — extremely volatile
        score = _compute_atr_opportunity_score(30.0, 1000.0)
        assert score >= 95.0
        assert score <= 100.0

    def test_monotonically_increasing(self):
        """Higher NATR should produce higher or equal scores."""
        prev = 0.0
        for atr_pct in [0.01, 0.05, 0.1, 0.2, 0.4, 0.6, 1.0, 2.0]:
            atr = 1000.0 * (atr_pct / 100.0)
            score = _compute_atr_opportunity_score(atr, 1000.0)
            assert score >= prev, f"Score should increase: {atr_pct}% → {score} (was {prev})"
            prev = score


# ---------------------------------------------------------------------------
# Tests: SwingCandidate dataclass
# ---------------------------------------------------------------------------


class TestSwingCandidate:
    def test_creation_and_to_dict(self):
        sc = SwingCandidate(
            asset_name="Gold",
            direction="LONG",
            confidence=0.82,
            swing_score=75.5,
            entry_zone_low=2690.0,
            entry_zone_high=2710.0,
            stop_loss=2660.0,
            tp1=2740.0,
            tp2=2780.0,
            tp3=2820.0,
            atr=18.5,
            last_price=2700.0,
            risk_dollars=250.0,
            position_size=2,
            reasoning="Bullish engulfing + monthly uptrend",
            entry_styles=["pullback_to_key_level", "breakout_entry"],
        )

        d = sc.to_dict()
        assert d["asset_name"] == "Gold"
        assert d["direction"] == "LONG"
        assert d["confidence"] == 0.82
        assert d["swing_score"] == 75.5
        assert d["entry_zone_low"] == 2690.0
        assert d["stop_loss"] == 2660.0
        assert d["tp1"] == 2740.0
        assert d["tp3"] == 2820.0
        assert d["position_size"] == 2
        assert "pullback_to_key_level" in d["entry_styles"]

    def test_default_values(self):
        sc = SwingCandidate(
            asset_name="Test",
            direction="SHORT",
            confidence=0.5,
            swing_score=40.0,
        )
        assert sc.tp1 == 0.0
        assert sc.position_size == 1
        assert sc.entry_styles == []
        assert sc.key_levels == {}


# ---------------------------------------------------------------------------
# Tests: ScalpFocusAsset dataclass
# ---------------------------------------------------------------------------


class TestScalpFocusAsset:
    def test_creation_and_to_dict(self):
        sf = ScalpFocusAsset(
            asset_name="Nasdaq",
            composite_score=78.3,
            signal_quality_score=85.0,
            atr_opportunity_score=60.0,
            rb_setup_density_score=70.0,
            session_fit_score=100.0,
            catalyst_score=50.0,
            bias_direction="LONG",
            bias_confidence=0.65,
            last_price=21500.0,
            atr=45.0,
        )

        d = sf.to_dict()
        assert d["asset_name"] == "Nasdaq"
        assert d["composite_score"] == 78.3
        assert d["signal_quality_score"] == 85.0
        assert d["bias_direction"] == "LONG"
        assert d["atr"] == 45.0


# ---------------------------------------------------------------------------
# Tests: DailyPlan dataclass
# ---------------------------------------------------------------------------


class TestDailyPlan:
    def test_empty_plan(self):
        plan = DailyPlan()
        d = plan.to_dict()
        assert d["total_scalp_focus"] == 0
        assert d["total_swing_candidates"] == 0
        assert d["scalp_focus_names"] == []
        assert d["swing_candidate_names"] == []
        assert d["no_trade"] is False

    def test_plan_with_selections(self):
        plan = DailyPlan(
            scalp_focus=[
                ScalpFocusAsset(asset_name="Gold", composite_score=80.0),
                ScalpFocusAsset(asset_name="Nasdaq", composite_score=75.0),
            ],
            swing_candidates=[
                SwingCandidate(
                    asset_name="S&P",
                    direction="LONG",
                    confidence=0.7,
                    swing_score=65.0,
                ),
            ],
            market_context="Risk-on environment, bullish equities",
            grok_available=True,
            computed_at="2025-01-15T05:30:00-05:00",
            account_size=50_000,
            session="pre-market",
        )

        d = plan.to_dict()
        assert d["total_scalp_focus"] == 2
        assert d["total_swing_candidates"] == 1
        assert d["scalp_focus_names"] == ["Gold", "Nasdaq"]
        assert d["swing_candidate_names"] == ["S&P"]
        assert d["grok_available"] is True
        assert d["market_context"].startswith("Risk-on")

    def test_no_trade_plan(self):
        plan = DailyPlan(
            no_trade=True,
            no_trade_reason="No assets meet minimum quality thresholds",
        )
        d = plan.to_dict()
        assert d["no_trade"] is True
        assert "quality" in d["no_trade_reason"]

    def test_to_dict_serializable(self):
        """Ensure to_dict() output is JSON-serializable."""
        plan = DailyPlan(
            scalp_focus=[ScalpFocusAsset(asset_name="Gold", composite_score=60.0)],
            swing_candidates=[
                SwingCandidate(
                    asset_name="Gold",
                    direction="LONG",
                    confidence=0.8,
                    swing_score=70.0,
                    key_levels={"prior_day_high": 2720.0},
                ),
            ],
            all_biases={"Gold": {"direction": "LONG", "confidence": 0.8}},
        )
        # Should not raise
        serialized = json.dumps(plan.to_dict(), default=str)
        assert isinstance(serialized, str)
        roundtrip = json.loads(serialized)
        assert roundtrip["scalp_focus_names"] == ["Gold"]


# ---------------------------------------------------------------------------
# Tests: DailyPlan Redis round-trip
# ---------------------------------------------------------------------------


class TestDailyPlanRedis:
    def _mock_redis(self) -> MagicMock:
        """Create a mock Redis client that stores data in a dict."""
        store: dict[str, Any] = {}
        r = MagicMock()
        r.set = lambda k, v: store.__setitem__(k, v)
        r.get = lambda k: store.get(k)
        r.expire = MagicMock()
        r.publish = MagicMock()
        return r

    def test_publish_and_load_round_trip(self):
        r = self._mock_redis()

        plan = DailyPlan(
            scalp_focus=[
                ScalpFocusAsset(asset_name="Gold", composite_score=80.0, atr=18.5),
                ScalpFocusAsset(asset_name="Nasdaq", composite_score=75.0, atr=45.0),
            ],
            swing_candidates=[
                SwingCandidate(
                    asset_name="S&P",
                    direction="LONG",
                    confidence=0.72,
                    swing_score=65.0,
                    tp1=5250.0,
                    tp3=5400.0,
                ),
            ],
            market_context="Test Grok context",
            grok_available=True,
            computed_at="2025-01-15T05:30:00",
            account_size=50_000,
            session="pre-market",
            all_biases={"Gold": {"direction": "LONG", "confidence": 0.8}},
        )

        assert plan.publish_to_redis(r) is True

        # Load it back
        loaded = DailyPlan.load_from_redis(r)
        assert loaded is not None
        assert len(loaded.scalp_focus) == 2
        assert loaded.scalp_focus[0].asset_name == "Gold"
        assert loaded.scalp_focus[0].composite_score == 80.0
        assert loaded.scalp_focus[1].asset_name == "Nasdaq"
        assert len(loaded.swing_candidates) == 1
        assert loaded.swing_candidates[0].asset_name == "S&P"
        assert loaded.swing_candidates[0].direction == "LONG"
        assert loaded.swing_candidates[0].tp1 == 5250.0
        assert loaded.market_context == "Test Grok context"
        assert loaded.grok_available is True
        assert loaded.all_biases["Gold"]["direction"] == "LONG"

    def test_load_from_empty_redis(self):
        r = MagicMock()
        r.get = MagicMock(return_value=None)
        assert DailyPlan.load_from_redis(r) is None

    def test_load_from_corrupted_redis(self):
        r = MagicMock()
        r.get = MagicMock(return_value="not valid json{{{")
        assert DailyPlan.load_from_redis(r) is None

    def test_publish_failure_returns_false(self):
        r = MagicMock()
        r.set = MagicMock(side_effect=ConnectionError("Redis down"))
        plan = DailyPlan()
        assert plan.publish_to_redis(r) is False


# ---------------------------------------------------------------------------
# Tests: _build_swing_candidate
# ---------------------------------------------------------------------------


class TestBuildSwingCandidate:
    def test_long_swing_levels(self):
        bias = _make_bias("Gold", BiasDirection.LONG, confidence=0.8)
        candidate = _build_swing_candidate(
            asset_name="Gold",
            bias=bias,
            last_price=2700.0,
            atr=18.0,
            swing_score=70.0,
            account_size=50_000,
        )

        assert candidate.direction == "LONG"
        assert candidate.asset_name == "Gold"
        assert candidate.confidence == 0.8
        assert candidate.swing_score == 70.0

        # For LONG: SL below entry, TPs above entry
        assert candidate.stop_loss < candidate.entry_zone_low
        assert candidate.tp1 > candidate.entry_zone_high
        assert candidate.tp2 > candidate.tp1
        assert candidate.tp3 > candidate.tp2
        assert candidate.position_size >= 1
        assert candidate.risk_dollars > 0

    def test_short_swing_levels(self):
        bias = _make_bias("Nasdaq", BiasDirection.SHORT, confidence=0.65)
        candidate = _build_swing_candidate(
            asset_name="Nasdaq",
            bias=bias,
            last_price=21000.0,
            atr=90.0,
            swing_score=55.0,
            account_size=50_000,
        )

        assert candidate.direction == "SHORT"
        # For SHORT: SL above entry, TPs below entry
        assert candidate.stop_loss > candidate.entry_zone_high
        assert candidate.tp1 < candidate.entry_zone_low
        assert candidate.tp2 < candidate.tp1
        assert candidate.tp3 < candidate.tp2

    def test_swing_levels_wider_than_typical_scalp(self):
        """Swing SL should be at SWING_SL_ATR_MULT × ATR — wider than scalp's 1.5×."""
        bias = _make_bias("Gold", BiasDirection.LONG, confidence=0.75)
        atr = 18.0
        candidate = _build_swing_candidate(
            asset_name="Gold",
            bias=bias,
            last_price=2700.0,
            atr=atr,
            swing_score=60.0,
            account_size=50_000,
        )

        # Check SL distance is approximately SWING_SL_ATR_MULT × ATR from midpoint
        midpoint = (candidate.entry_zone_low + candidate.entry_zone_high) / 2
        sl_distance = abs(midpoint - candidate.stop_loss)
        expected_sl = atr * SWING_SL_ATR_MULT
        assert abs(sl_distance - expected_sl) < atr * 0.15  # Within 15% tolerance

        # Check TP distances
        tp1_distance = abs(candidate.tp1 - midpoint)
        expected_tp1 = atr * SWING_TP1_ATR_MULT
        assert abs(tp1_distance - expected_tp1) < atr * 0.15

    def test_entry_styles_populated(self):
        bias = _make_bias("Gold", BiasDirection.LONG, confidence=0.7)
        bias.atr_expanding = True
        bias.overnight_gap_direction = 0.5  # Positive gap aligns with LONG
        candidate = _build_swing_candidate(
            asset_name="Gold",
            bias=bias,
            last_price=2700.0,
            atr=18.0,
            swing_score=60.0,
            account_size=50_000,
        )
        assert len(candidate.entry_styles) > 0
        # Should include breakout since ATR is expanding
        assert "breakout_entry" in candidate.entry_styles

    def test_position_size_capped_at_3(self):
        """Swing position size should be capped at 3 micros — patience trade."""
        bias = _make_bias("Gold", BiasDirection.LONG, confidence=0.9)
        # Very large account should still cap at 3
        candidate = _build_swing_candidate(
            asset_name="Gold",
            bias=bias,
            last_price=2700.0,
            atr=2.0,  # Tiny ATR → many contracts fit in risk budget
            swing_score=80.0,
            account_size=500_000,  # Very large account
        )
        assert candidate.position_size <= 3


# ---------------------------------------------------------------------------
# Tests: select_daily_focus_assets
# ---------------------------------------------------------------------------


class TestSelectDailyFocusAssets:
    """Test the Phase 3A composite ranking and selection."""

    def _make_biases(self) -> dict[str, DailyBias]:
        return {
            "Gold": _make_bias("Gold", BiasDirection.LONG, 0.82, atr_expanding=True),
            "S&P": _make_bias("S&P", BiasDirection.LONG, 0.55, atr_expanding=False),
            "Nasdaq": _make_bias("Nasdaq", BiasDirection.SHORT, 0.70),
            "Euro FX": _make_bias("Euro FX", BiasDirection.NEUTRAL, 0.10),
            "Crude Oil": _make_bias("Crude Oil", BiasDirection.LONG, 0.45),
            "Russell 2000": _make_bias("Russell 2000", BiasDirection.SHORT, 0.30),
        }

    @patch("lib.trading.strategies.daily.daily_plan._fetch_asset_data")
    @patch("lib.trading.strategies.daily.daily_plan._compute_signal_quality_for_asset")
    def test_returns_correct_types(self, mock_sq, mock_data):
        mock_data.return_value = (2700.0, 18.0)
        mock_sq.return_value = 65.0
        biases = self._make_biases()

        scalp, swing = select_daily_focus_assets(
            biases=biases,
            asset_names=list(biases.keys()),
        )

        assert isinstance(scalp, list)
        assert isinstance(swing, list)
        assert all(isinstance(s, ScalpFocusAsset) for s in scalp)
        assert all(isinstance(s, SwingCandidate) for s in swing)

    @patch("lib.trading.strategies.daily.daily_plan._fetch_asset_data")
    @patch("lib.trading.strategies.daily.daily_plan._compute_signal_quality_for_asset")
    def test_max_scalp_count(self, mock_sq, mock_data):
        mock_data.return_value = (2700.0, 18.0)
        mock_sq.return_value = 70.0
        biases = self._make_biases()

        scalp, _ = select_daily_focus_assets(
            biases=biases,
            asset_names=list(biases.keys()),
            max_scalp=3,
        )

        assert len(scalp) <= 3

    @patch("lib.trading.strategies.daily.daily_plan._fetch_asset_data")
    @patch("lib.trading.strategies.daily.daily_plan._compute_signal_quality_for_asset")
    def test_max_swing_count(self, mock_sq, mock_data):
        mock_data.return_value = (2700.0, 18.0)
        mock_sq.return_value = 70.0
        biases = self._make_biases()

        _, swing = select_daily_focus_assets(
            biases=biases,
            asset_names=list(biases.keys()),
            max_swing=1,
        )

        assert len(swing) <= 1

    @patch("lib.trading.strategies.daily.daily_plan._fetch_asset_data")
    @patch("lib.trading.strategies.daily.daily_plan._compute_signal_quality_for_asset")
    def test_scalp_sorted_by_composite_score(self, mock_sq, mock_data):
        mock_data.return_value = (2700.0, 18.0)
        mock_sq.return_value = 70.0
        biases = self._make_biases()

        scalp, _ = select_daily_focus_assets(
            biases=biases,
            asset_names=list(biases.keys()),
        )

        if len(scalp) >= 2:
            scores = [s.composite_score for s in scalp]
            assert scores == sorted(scores, reverse=True)

    @patch("lib.trading.strategies.daily.daily_plan._fetch_asset_data")
    @patch("lib.trading.strategies.daily.daily_plan._compute_signal_quality_for_asset")
    def test_neutral_bias_excluded_from_swing(self, mock_sq, mock_data):
        mock_data.return_value = (2700.0, 18.0)
        mock_sq.return_value = 70.0

        biases = {
            "Gold": _make_bias("Gold", BiasDirection.NEUTRAL, 0.05),
            "S&P": _make_bias("S&P", BiasDirection.NEUTRAL, 0.08),
        }

        _, swing = select_daily_focus_assets(
            biases=biases,
            asset_names=list(biases.keys()),
        )

        assert len(swing) == 0

    @patch("lib.trading.strategies.daily.daily_plan._fetch_asset_data")
    @patch("lib.trading.strategies.daily.daily_plan._compute_signal_quality_for_asset")
    def test_no_data_assets_skipped(self, mock_sq, mock_data):
        mock_data.return_value = (0.0, 0.0)  # No data
        mock_sq.return_value = 50.0
        biases = self._make_biases()

        scalp, swing = select_daily_focus_assets(
            biases=biases,
            asset_names=list(biases.keys()),
        )

        assert len(scalp) == 0
        assert len(swing) == 0

    @patch("lib.trading.strategies.daily.daily_plan._fetch_asset_data")
    @patch("lib.trading.strategies.daily.daily_plan._compute_signal_quality_for_asset")
    def test_swing_has_valid_levels(self, mock_sq, mock_data):
        mock_data.return_value = (2700.0, 18.0)
        mock_sq.return_value = 70.0
        biases = {"Gold": _make_bias("Gold", BiasDirection.LONG, 0.85)}

        _, swing = select_daily_focus_assets(
            biases=biases,
            asset_names=["Gold"],
            max_swing=1,
        )

        if swing:
            sc = swing[0]
            assert sc.entry_zone_low > 0
            assert sc.entry_zone_high > sc.entry_zone_low
            assert sc.stop_loss > 0
            assert sc.tp1 > 0
            assert sc.atr > 0

    @patch("lib.trading.strategies.daily.daily_plan._fetch_asset_data")
    @patch("lib.trading.strategies.daily.daily_plan._compute_signal_quality_for_asset")
    def test_composite_score_components(self, mock_sq, mock_data):
        """Verify the composite score reflects the weighted components."""
        mock_data.return_value = (5000.0, 25.0)  # NATR ~0.5% → moderate opportunity
        mock_sq.return_value = 90.0

        biases = {"S&P": _make_bias("S&P", BiasDirection.LONG, 0.7)}

        scalp, _ = select_daily_focus_assets(
            biases=biases,
            asset_names=["S&P"],
        )

        assert len(scalp) == 1
        sf = scalp[0]
        assert sf.signal_quality_score == 90.0
        assert sf.atr_opportunity_score > 0
        assert sf.session_fit_score > 0
        # Composite should be non-zero
        assert sf.composite_score > 0


# ---------------------------------------------------------------------------
# Tests: generate_daily_plan
# ---------------------------------------------------------------------------


class TestGenerateDailyPlan:
    """Test the Phase 2B orchestrator."""

    @patch("lib.trading.strategies.daily.daily_plan._fetch_asset_data")
    @patch("lib.trading.strategies.daily.daily_plan._compute_signal_quality_for_asset")
    @patch("lib.trading.strategies.daily.daily_plan.compute_all_daily_biases")
    def test_basic_generation(self, mock_biases, mock_sq, mock_data):
        mock_data.return_value = (2700.0, 18.0)
        mock_sq.return_value = 65.0
        mock_biases.return_value = {
            "Gold": _make_bias("Gold", BiasDirection.LONG, 0.75),
            "Nasdaq": _make_bias("Nasdaq", BiasDirection.SHORT, 0.60),
        }

        plan = generate_daily_plan(
            account_size=50_000,
            asset_names=["Gold", "Nasdaq"],
            daily_data={
                "Gold": _make_daily_bars(start_price=2700.0, seed=42),
                "Nasdaq": _make_daily_bars(start_price=21000.0, seed=43),
            },
            include_grok=False,
        )

        assert isinstance(plan, DailyPlan)
        assert plan.computed_at != ""
        assert plan.account_size == 50_000
        assert plan.session in ("pre-market", "active", "off-hours")
        assert "Gold" in plan.all_biases
        assert "Nasdaq" in plan.all_biases

    @patch("lib.trading.strategies.daily.daily_plan._fetch_asset_data")
    @patch("lib.trading.strategies.daily.daily_plan._compute_signal_quality_for_asset")
    @patch("lib.trading.strategies.daily.daily_plan.compute_all_daily_biases")
    def test_no_data_produces_no_trade(self, mock_biases, mock_sq, mock_data):
        mock_biases.return_value = {}
        mock_data.return_value = (0.0, 0.0)
        mock_sq.return_value = 0.0

        plan = generate_daily_plan(
            account_size=50_000,
            asset_names=["Gold"],
            daily_data={},  # Empty data
            include_grok=False,
        )

        assert plan.no_trade is True

    @patch("lib.trading.strategies.daily.daily_plan._fetch_asset_data")
    @patch("lib.trading.strategies.daily.daily_plan._compute_signal_quality_for_asset")
    @patch("lib.trading.strategies.daily.daily_plan.compute_all_daily_biases")
    def test_all_neutral_note(self, mock_biases, mock_sq, mock_data):
        mock_data.return_value = (2700.0, 18.0)
        mock_sq.return_value = 65.0
        mock_biases.return_value = {
            "Gold": _make_bias("Gold", BiasDirection.NEUTRAL, 0.05),
            "S&P": _make_bias("S&P", BiasDirection.NEUTRAL, 0.08),
        }

        plan = generate_daily_plan(
            account_size=50_000,
            asset_names=["Gold", "S&P"],
            daily_data={
                "Gold": _make_daily_bars(start_price=2700.0),
                "S&P": _make_daily_bars(start_price=5500.0),
            },
            include_grok=False,
        )

        # Should note low conviction even if scalp focus is selected
        assert "neutral" in plan.no_trade_reason.lower() or "conviction" in plan.no_trade_reason.lower()

    @patch("lib.trading.strategies.daily.daily_plan._fetch_asset_data")
    @patch("lib.trading.strategies.daily.daily_plan._compute_signal_quality_for_asset")
    @patch("lib.trading.strategies.daily.daily_plan.compute_all_daily_biases")
    def test_grok_not_called_when_disabled(self, mock_biases, mock_sq, mock_data):
        mock_data.return_value = (2700.0, 18.0)
        mock_sq.return_value = 65.0
        mock_biases.return_value = {
            "Gold": _make_bias("Gold", BiasDirection.LONG, 0.7),
        }

        with patch.dict("os.environ", {"XAI_API_KEY": ""}, clear=False):
            plan = generate_daily_plan(
                account_size=50_000,
                asset_names=["Gold"],
                daily_data={"Gold": _make_daily_bars()},
                include_grok=True,  # Enabled, but no API key
            )

        assert plan.grok_available is False
        assert plan.market_context == ""

    @patch("lib.trading.strategies.daily.daily_plan._fetch_asset_data")
    @patch("lib.trading.strategies.daily.daily_plan._compute_signal_quality_for_asset")
    @patch("lib.trading.strategies.daily.daily_plan.compute_all_daily_biases")
    def test_plan_to_dict_is_complete(self, mock_biases, mock_sq, mock_data):
        mock_data.return_value = (2700.0, 18.0)
        mock_sq.return_value = 70.0
        mock_biases.return_value = {
            "Gold": _make_bias("Gold", BiasDirection.LONG, 0.80),
        }

        plan = generate_daily_plan(
            account_size=50_000,
            asset_names=["Gold"],
            daily_data={"Gold": _make_daily_bars()},
            include_grok=False,
        )

        d = plan.to_dict()
        required_keys = {
            "scalp_focus",
            "swing_candidates",
            "scalp_focus_names",
            "swing_candidate_names",
            "market_context",
            "grok_available",
            "no_trade",
            "no_trade_reason",
            "all_biases",
            "computed_at",
            "account_size",
            "session",
            "total_scalp_focus",
            "total_swing_candidates",
        }
        assert required_keys.issubset(set(d.keys()))


# ---------------------------------------------------------------------------
# Tests: Integration with bias_analyzer (Phase 2A → 2B pipeline)
# ---------------------------------------------------------------------------


class TestBiasAnalyzerIntegration:
    """End-to-end: daily bars → bias_analyzer → daily_plan selection."""

    def test_bias_to_plan_pipeline(self):
        """Smoke test: run real bias computation then feed into selection."""
        daily_data = {
            "Gold": _make_daily_bars(n=30, start_price=2700.0, trend=0.003, seed=10),
            "Nasdaq": _make_daily_bars(n=30, start_price=21000.0, trend=-0.002, seed=20),
            "S&P": _make_daily_bars(n=30, start_price=5500.0, trend=0.001, seed=30),
        }

        weekly_data = {
            "Gold": _make_weekly_bars(n=10, start_price=2650.0, seed=11),
            "Nasdaq": _make_weekly_bars(n=10, start_price=20500.0, seed=21),
            "S&P": _make_weekly_bars(n=10, start_price=5400.0, seed=31),
        }

        biases = compute_all_daily_biases(
            daily_data=daily_data,
            weekly_data=weekly_data,
        )

        assert len(biases) == 3
        assert all(isinstance(b, DailyBias) for b in biases.values())

        # All biases should have direction and confidence set
        for _name, bias in biases.items():
            assert bias.direction in (BiasDirection.LONG, BiasDirection.SHORT, BiasDirection.NEUTRAL)
            assert 0.0 <= bias.confidence <= 1.0
            assert bias.reasoning  # Non-empty reasoning

    def test_single_asset_bias_fields(self):
        """Verify a single bias has all expected fields populated."""
        daily = _make_daily_bars(n=30, start_price=2700.0, trend=0.005, seed=55)
        weekly = _make_weekly_bars(n=10, start_price=2650.0, seed=56)

        bias = compute_daily_bias(
            daily_bars=daily,
            weekly_bars=weekly,
            asset_name="Gold",
            current_open=2715.0,
        )

        assert bias.asset_name == "Gold"
        assert bias.key_levels is not None
        assert bias.key_levels.prior_day_high > 0
        assert bias.key_levels.prior_day_low > 0
        assert bias.candle_pattern is not None
        # Monthly trend should be computed (we have 30 bars)
        assert -1.0 <= bias.monthly_trend_score <= 1.0
        # Weekly range position
        assert 0.0 <= bias.weekly_range_position <= 1.0


# ---------------------------------------------------------------------------
# Tests: _get_current_session
# ---------------------------------------------------------------------------


class TestGetCurrentSession:
    """Test session detection logic with mocked time."""

    @patch("lib.trading.strategies.daily.daily_plan.datetime")
    def test_asian_session_evening(self, mock_dt):
        mock_dt.now.return_value = datetime(2025, 1, 15, 20, 30, tzinfo=_EST)
        assert _get_current_session() == "asian"

    @patch("lib.trading.strategies.daily.daily_plan.datetime")
    def test_asian_session_early_morning(self, mock_dt):
        mock_dt.now.return_value = datetime(2025, 1, 15, 1, 0, tzinfo=_EST)
        assert _get_current_session() == "asian"

    @patch("lib.trading.strategies.daily.daily_plan.datetime")
    def test_london_session(self, mock_dt):
        mock_dt.now.return_value = datetime(2025, 1, 15, 4, 30, tzinfo=_EST)
        assert _get_current_session() == "london"

    @patch("lib.trading.strategies.daily.daily_plan.datetime")
    def test_us_session(self, mock_dt):
        mock_dt.now.return_value = datetime(2025, 1, 15, 10, 0, tzinfo=_EST)
        assert _get_current_session() == "us"

    @patch("lib.trading.strategies.daily.daily_plan.datetime")
    def test_off_hours(self, mock_dt):
        mock_dt.now.return_value = datetime(2025, 1, 15, 17, 30, tzinfo=_EST)
        assert _get_current_session() == "off-hours"


# ---------------------------------------------------------------------------
# Tests: ASSET_SESSION_MAP completeness
# ---------------------------------------------------------------------------


class TestAssetSessionMap:
    """Verify session map covers known assets."""

    def test_core_futures_covered(self):
        core = ["Gold", "Silver", "S&P", "Nasdaq", "Crude Oil", "Euro FX"]
        for name in core:
            assert name in ASSET_SESSION_MAP, f"{name} missing from ASSET_SESSION_MAP"
            assert len(ASSET_SESSION_MAP[name]) >= 1

    def test_all_sessions_valid(self):
        valid_sessions = {"asian", "london", "us"}
        for name, sessions in ASSET_SESSION_MAP.items():
            for s in sessions:
                assert s in valid_sessions, f"{name} has invalid session '{s}'"


# ---------------------------------------------------------------------------
# Tests: Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    @patch("lib.trading.strategies.daily.daily_plan._fetch_asset_data")
    @patch("lib.trading.strategies.daily.daily_plan._compute_signal_quality_for_asset")
    def test_single_asset(self, mock_sq, mock_data):
        mock_data.return_value = (2700.0, 18.0)
        mock_sq.return_value = 75.0
        biases = {"Gold": _make_bias("Gold", BiasDirection.LONG, 0.8)}

        scalp, swing = select_daily_focus_assets(
            biases=biases,
            asset_names=["Gold"],
            max_scalp=4,
            max_swing=2,
        )

        assert len(scalp) == 1
        assert scalp[0].asset_name == "Gold"
        # One directional asset should produce one swing candidate
        assert len(swing) <= 1

    @patch("lib.trading.strategies.daily.daily_plan._fetch_asset_data")
    @patch("lib.trading.strategies.daily.daily_plan._compute_signal_quality_for_asset")
    def test_empty_biases(self, mock_sq, mock_data):
        mock_data.return_value = (2700.0, 18.0)
        mock_sq.return_value = 50.0

        scalp, swing = select_daily_focus_assets(
            biases={},
            asset_names=[],
        )

        assert len(scalp) == 0
        assert len(swing) == 0

    @patch("lib.trading.strategies.daily.daily_plan._fetch_asset_data")
    @patch("lib.trading.strategies.daily.daily_plan._compute_signal_quality_for_asset")
    def test_low_confidence_excluded_from_swing(self, mock_sq, mock_data):
        """Assets with confidence <= 0.15 should not be swing candidates."""
        mock_data.return_value = (2700.0, 18.0)
        mock_sq.return_value = 60.0

        biases = {
            "Gold": _make_bias("Gold", BiasDirection.LONG, 0.12),  # Below 0.15 threshold
        }

        _, swing = select_daily_focus_assets(
            biases=biases,
            asset_names=["Gold"],
        )

        assert len(swing) == 0

    @patch("lib.trading.strategies.daily.daily_plan._fetch_asset_data")
    @patch("lib.trading.strategies.daily.daily_plan._compute_signal_quality_for_asset")
    def test_rb_density_fallback_without_redis(self, mock_sq, mock_data):
        """Without Redis, RB density should return neutral score (40)."""
        mock_data.return_value = (2700.0, 18.0)
        mock_sq.return_value = 70.0
        biases = {"Gold": _make_bias("Gold", BiasDirection.LONG, 0.7)}

        scalp, _ = select_daily_focus_assets(
            biases=biases,
            asset_names=["Gold"],
            redis_client=None,  # No Redis
        )

        assert len(scalp) == 1
        # RB density should be the neutral fallback of 40
        assert scalp[0].rb_setup_density_score == 40.0

    @patch("lib.trading.strategies.daily.daily_plan._fetch_asset_data")
    @patch("lib.trading.strategies.daily.daily_plan._compute_signal_quality_for_asset")
    def test_catalyst_score_fallback_without_redis(self, mock_sq, mock_data):
        mock_data.return_value = (2700.0, 18.0)
        mock_sq.return_value = 70.0
        biases = {"Gold": _make_bias("Gold", BiasDirection.LONG, 0.7)}

        scalp, _ = select_daily_focus_assets(
            biases=biases,
            asset_names=["Gold"],
            redis_client=None,
        )

        assert len(scalp) == 1
        assert scalp[0].catalyst_score == 30.0  # Neutral fallback


# ---------------------------------------------------------------------------
# Tests: Constants sanity
# ---------------------------------------------------------------------------


class TestConstants:
    def test_swing_atr_multipliers_ordered(self):
        """TP multipliers should increase: SL < TP1 < TP2 < TP3."""
        assert SWING_SL_ATR_MULT < SWING_TP1_ATR_MULT
        assert SWING_TP1_ATR_MULT < SWING_TP2_ATR_MULT
        assert SWING_TP2_ATR_MULT < SWING_TP3_ATR_MULT

    def test_selection_counts_positive(self):
        assert MAX_SCALP_FOCUS >= 1
        assert MAX_SWING_CANDIDATES >= 1

    def test_minimum_thresholds_reasonable(self):
        assert 0 < MIN_SCALP_SCORE < 100
        assert 0 < MIN_SWING_SCORE < 100
