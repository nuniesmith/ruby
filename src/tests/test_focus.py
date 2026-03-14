"""
Tests for the Daily Focus Computation module.

Covers:
  - compute_asset_focus() returns correct structure
  - _derive_bias() logic for LONG / SHORT / NEUTRAL
  - _compute_entry_zone() level calculations
  - _compute_position_size() micro contract sizing
  - should_not_trade() conditions (all low quality, extreme vol, time-based)
  - compute_daily_focus() full payload structure
  - publish_focus_to_redis() serialization
"""

import math
import sys
from datetime import datetime
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

import pytest

from lib.services.engine.focus import (
    MIN_QUALITY_THRESHOLD,
    _compute_entry_zone,
    _compute_position_size,
    _derive_bias,
    _safe_float,
    should_not_trade,
)

_EST = ZoneInfo("America/New_York")


# ---------------------------------------------------------------------------
# _safe_float
# ---------------------------------------------------------------------------


class TestSafeFloat:
    def test_normal_float(self):
        assert _safe_float(3.14) == 3.14

    def test_int_to_float(self):
        assert _safe_float(42) == 42.0

    def test_string_number(self):
        assert _safe_float("2.5") == 2.5

    def test_none_returns_default(self):
        assert _safe_float(None) == 0.0
        assert _safe_float(None, 99.0) == 99.0

    def test_nan_returns_default(self):
        assert _safe_float(float("nan"), 1.0) == 1.0

    def test_inf_returns_default(self):
        assert _safe_float(float("inf"), 2.0) == 2.0
        assert _safe_float(float("-inf"), 3.0) == 3.0

    def test_bad_string_returns_default(self):
        assert _safe_float("not_a_number", 5.0) == 5.0


# ---------------------------------------------------------------------------
# _derive_bias
# ---------------------------------------------------------------------------


class TestDeriveBias:
    """Test bias derivation from wave + signal quality results."""

    def _make_wave(self, bias="BULLISH", dominance=0.2):
        return {"bias": bias, "dominance": dominance}

    def _make_sq(self, ao=1.0, context="UPTREND"):
        return {"ao": ao, "market_context": context}

    def test_long_bias_bullish_wave_positive_ao(self):
        result = _derive_bias(
            self._make_wave("BULLISH", 0.2),
            self._make_sq(1.0, "UPTREND"),
            quality=0.7,
        )
        assert result == "LONG"

    def test_short_bias_bearish_wave_negative_ao(self):
        result = _derive_bias(
            self._make_wave("BEARISH", -0.2),
            self._make_sq(-1.0, "DOWNTREND"),
            quality=0.7,
        )
        assert result == "SHORT"

    def test_neutral_when_quality_below_threshold(self):
        result = _derive_bias(
            self._make_wave("BULLISH", 0.3),
            self._make_sq(1.0, "UPTREND"),
            quality=0.4,  # below 0.55
        )
        assert result == "NEUTRAL"

    def test_neutral_when_wave_is_neutral(self):
        result = _derive_bias(
            self._make_wave("NEUTRAL", 0.01),
            self._make_sq(0.0, "RANGING"),
            quality=0.7,
        )
        assert result == "NEUTRAL"

    def test_long_with_strong_dominance_even_without_ao(self):
        """Strong dominance (>0.15) should give LONG even if AO is negative."""
        result = _derive_bias(
            self._make_wave("BULLISH", 0.2),
            self._make_sq(-0.5, "RANGING"),  # AO negative, context not UPTREND
            quality=0.7,
        )
        assert result == "LONG"

    def test_short_with_strong_dominance_even_without_ao(self):
        result = _derive_bias(
            self._make_wave("BEARISH", -0.2),
            self._make_sq(0.5, "RANGING"),
            quality=0.7,
        )
        assert result == "SHORT"

    def test_neutral_when_dominance_too_weak(self):
        """Weak dominance (< 0.05) should not trigger a directional bias."""
        result = _derive_bias(
            self._make_wave("BULLISH", 0.03),
            self._make_sq(-0.5, "RANGING"),
            quality=0.7,
        )
        assert result == "NEUTRAL"


# ---------------------------------------------------------------------------
# _compute_entry_zone
# ---------------------------------------------------------------------------


class TestComputeEntryZone:
    def test_long_entry_zone_structure(self):
        zone = _compute_entry_zone(last_price=5200.0, bias="LONG", atr=10.0, wave_ratio=2.0)
        assert "entry_low" in zone
        assert "entry_high" in zone
        assert "stop" in zone
        assert "tp1" in zone
        assert "tp2" in zone

    def test_long_entry_below_price(self):
        zone = _compute_entry_zone(last_price=5200.0, bias="LONG", atr=10.0, wave_ratio=2.0)
        # Entry low should be below the current price
        assert zone["entry_low"] < 5200.0
        # Stop should be below entry
        assert zone["stop"] < zone["entry_low"]
        # TP1 and TP2 above entry
        assert zone["tp1"] > zone["entry_high"]
        assert zone["tp2"] > zone["tp1"]

    def test_short_entry_above_price(self):
        zone = _compute_entry_zone(last_price=5200.0, bias="SHORT", atr=10.0, wave_ratio=2.0)
        # Entry high should be above current price
        assert zone["entry_high"] > 5200.0
        # Stop should be above entry
        assert zone["stop"] > zone["entry_high"]
        # TP1 and TP2 below entry
        assert zone["tp1"] < zone["entry_low"]
        assert zone["tp2"] < zone["tp1"]

    def test_neutral_has_symmetric_zone(self):
        zone = _compute_entry_zone(last_price=5200.0, bias="NEUTRAL", atr=10.0, wave_ratio=1.0)
        # Both entry boundaries should bracket the price
        assert zone["entry_low"] < 5200.0
        assert zone["entry_high"] > 5200.0

    def test_zero_atr_uses_fallback(self):
        """Should not crash on zero ATR."""
        zone = _compute_entry_zone(last_price=5200.0, bias="LONG", atr=0.0, wave_ratio=1.0)
        # Fallback ATR is 0.5% of price = 26.0
        assert zone["entry_low"] < 5200.0
        assert zone["stop"] < zone["entry_low"]


# ---------------------------------------------------------------------------
# _compute_position_size
# ---------------------------------------------------------------------------


class TestComputePositionSize:
    def test_basic_gold_sizing(self):
        """MGC: tick=0.10, point=10. A $5 stop = 50 ticks × $1/tick = $50/contract."""
        contracts, risk = _compute_position_size(
            last_price=5200.0,
            stop_price=5195.0,  # $5 stop distance
            tick_size=0.10,
            point_value=10,
            max_risk_dollars=375.0,
        )
        # Risk per contract = (5.0 / 0.10) × (0.10 × 10) = 50 × 1.0 = $50
        # Max contracts = 375 / 50 = 7
        assert contracts == 7
        assert risk == pytest.approx(350.0, rel=0.01)

    def test_basic_nq_sizing(self):
        """MNQ: tick=0.25, point=2. A 50pt stop = 200 ticks × $0.50/tick = $100/contract."""
        contracts, risk = _compute_position_size(
            last_price=20000.0,
            stop_price=19950.0,  # 50 point stop
            tick_size=0.25,
            point_value=2,
            max_risk_dollars=375.0,
        )
        # Risk per contract = (50 / 0.25) × (0.25 × 2) = 200 × 0.50 = $100
        # Max contracts = 375 / 100 = 3
        assert contracts == 3
        assert risk == pytest.approx(300.0, rel=0.01)

    def test_minimum_one_contract(self):
        """Should always return at least 1 contract."""
        contracts, risk = _compute_position_size(
            last_price=5200.0,
            stop_price=5100.0,  # huge stop
            tick_size=0.10,
            point_value=10,
            max_risk_dollars=10.0,  # tiny budget
        )
        assert contracts >= 1

    def test_zero_stop_distance(self):
        """Should handle zero stop distance gracefully."""
        contracts, risk = _compute_position_size(
            last_price=5200.0,
            stop_price=5200.0,  # zero distance
            tick_size=0.10,
            point_value=10,
            max_risk_dollars=375.0,
        )
        assert contracts >= 1

    def test_zero_tick_size(self):
        contracts, risk = _compute_position_size(
            last_price=100.0,
            stop_price=99.0,
            tick_size=0.0,
            point_value=10,
            max_risk_dollars=375.0,
        )
        assert contracts == 1


# ---------------------------------------------------------------------------
# should_not_trade
# ---------------------------------------------------------------------------


class TestShouldNotTrade:
    def _make_asset(self, quality=0.7, vol_pct=0.5, symbol="Gold"):
        return {
            "symbol": symbol,
            "quality": quality,
            "vol_percentile": vol_pct,
            "skip": quality < MIN_QUALITY_THRESHOLD,
        }

    def test_no_assets_means_no_trade(self):
        result, reason = should_not_trade([])
        assert result is True
        assert "No market data" in reason

    def test_all_low_quality_triggers_no_trade(self):
        assets = [
            self._make_asset(quality=0.3, symbol="Gold"),
            self._make_asset(quality=0.4, symbol="Nasdaq"),
            self._make_asset(quality=0.2, symbol="S&P"),
        ]
        result, reason = should_not_trade(assets)
        assert result is True
        assert "quality" in reason.lower()

    def test_one_high_quality_allows_trading(self):
        assets = [
            self._make_asset(quality=0.3, symbol="Gold"),
            self._make_asset(quality=0.7, symbol="Nasdaq"),  # above threshold
            self._make_asset(quality=0.2, symbol="S&P"),
        ]
        result, reason = should_not_trade(assets)
        assert result is False

    def test_extreme_volatility_triggers_no_trade(self):
        assets = [
            self._make_asset(quality=0.8, vol_pct=0.92, symbol="Gold"),
        ]
        result, reason = should_not_trade(assets)
        assert result is True
        assert "volatility" in reason.lower()

    def test_normal_volatility_allows_trading(self):
        assets = [
            self._make_asset(quality=0.8, vol_pct=0.6, symbol="Gold"),
        ]
        result, reason = should_not_trade(assets)
        assert result is False

    def test_after_10am_with_no_quality_setups(self):
        """After 10 AM ET with no tradeable assets should trigger no-trade."""
        assets = [
            self._make_asset(quality=0.3, vol_pct=0.4, symbol="Gold"),
            self._make_asset(quality=0.4, vol_pct=0.4, symbol="Nasdaq"),
        ]
        # Mock the time to be 10:30 AM ET
        mock_now = datetime(2026, 2, 27, 10, 30, 0, tzinfo=_EST)
        with patch("lib.services.engine.focus.datetime") as mock_dt:
            mock_dt.now.return_value = mock_now
            mock_dt.fromisoformat = datetime.fromisoformat
            result, reason = should_not_trade(assets)
        # All assets have quality < 0.55, so condition 1 should trigger first
        assert result is True

    def test_mixed_quality_and_normal_vol(self):
        """Standard good conditions — should allow trading."""
        assets = [
            self._make_asset(quality=0.75, vol_pct=0.45, symbol="Gold"),
            self._make_asset(quality=0.60, vol_pct=0.50, symbol="Nasdaq"),
            self._make_asset(quality=0.40, vol_pct=0.55, symbol="S&P"),
        ]
        result, reason = should_not_trade(assets)
        assert result is False
        assert reason == ""


# ---------------------------------------------------------------------------
# _compute_entry_zone edge cases
# ---------------------------------------------------------------------------


class TestEntryZoneEdgeCases:
    def test_strong_wave_tightens_entry(self):
        """Strong wave ratio (>1.5) should produce a tighter entry zone."""
        zone_strong = _compute_entry_zone(last_price=5200.0, bias="LONG", atr=10.0, wave_ratio=2.0)
        zone_weak = _compute_entry_zone(last_price=5200.0, bias="LONG", atr=10.0, wave_ratio=1.0)
        # Strong wave = tighter zone = smaller difference between entry_high and entry_low
        strong_width = zone_strong["entry_high"] - zone_strong["entry_low"]
        weak_width = zone_weak["entry_high"] - zone_weak["entry_low"]
        assert strong_width < weak_width

    def test_all_values_are_finite(self):
        """No NaN or Inf in output."""
        for bias in ("LONG", "SHORT", "NEUTRAL"):
            zone = _compute_entry_zone(last_price=100.0, bias=bias, atr=1.0, wave_ratio=1.5)
            for key, val in zone.items():
                assert math.isfinite(val), f"{key} is not finite for bias={bias}"


# ---------------------------------------------------------------------------
# Focus payload structure (integration-style, with mocked data)
# ---------------------------------------------------------------------------


class TestComputeDailyFocusPayload:
    """Test that compute_daily_focus returns the expected payload shape."""

    @patch("lib.services.engine.focus.compute_asset_focus")
    def test_payload_structure(self, mock_asset_focus):
        """Verify the top-level focus payload has all required fields."""
        from lib.services.engine.focus import compute_daily_focus

        mock_asset_focus.return_value = {
            "symbol": "Gold",
            "ticker": "MGC=F",
            "bias": "LONG",
            "bias_emoji": "🟢",
            "last_price": 5200.0,
            "entry_low": 5195.0,
            "entry_high": 5201.5,
            "stop": 5185.0,
            "tp1": 5220.0,
            "tp2": 5235.0,
            "wave_ratio": 1.85,
            "wave_ratio_text": "1.85x",
            "quality": 0.72,
            "quality_pct": 72.0,
            "high_quality": True,
            "vol_percentile": 0.45,
            "vol_cluster": "MEDIUM",
            "market_phase": "UPTREND",
            "trend_direction": "BULLISH ↗️",
            "momentum_state": "ACCELERATING",
            "dominance_text": "Bullish +1.85x",
            "position_size": 7,
            "risk_dollars": 350.0,
            "max_risk_allowed": 375.0,
            "atr": 5.2,
            "notes": "",
            "skip": False,
        }

        result = compute_daily_focus(account_size=50_000, symbols=["Gold"])

        assert "assets" in result
        assert "no_trade" in result
        assert "no_trade_reason" in result
        assert "computed_at" in result
        assert "account_size" in result
        assert "session_mode" in result
        assert "total_assets" in result
        assert "tradeable_assets" in result
        assert result["account_size"] == 50_000
        assert result["total_assets"] == 1
        assert result["tradeable_assets"] == 1

    @patch("lib.services.engine.focus.compute_asset_focus")
    def test_sorts_by_quality_desc(self, mock_asset_focus):
        """Assets should be sorted by quality (best first)."""
        from lib.services.engine.focus import compute_daily_focus

        def side_effect(name, account_size=50_000, **kwargs):
            qualities = {"Gold": 0.8, "Nasdaq": 0.6, "S&P": 0.9}
            q = qualities.get(name, 0.5)
            return {
                "symbol": name,
                "ticker": f"{name}=F",
                "bias": "LONG",
                "bias_emoji": "🟢",
                "last_price": 100.0,
                "entry_low": 99.0,
                "entry_high": 101.0,
                "stop": 97.0,
                "tp1": 104.0,
                "tp2": 107.0,
                "wave_ratio": 1.5,
                "wave_ratio_text": "1.50x",
                "quality": q,
                "quality_pct": q * 100,
                "high_quality": q >= 0.55,
                "vol_percentile": 0.5,
                "vol_cluster": "MEDIUM",
                "market_phase": "UPTREND",
                "trend_direction": "BULLISH",
                "momentum_state": "NEUTRAL",
                "dominance_text": "Neutral",
                "position_size": 1,
                "risk_dollars": 50.0,
                "max_risk_allowed": 375.0,
                "atr": 2.0,
                "notes": "",
                "skip": False,
            }

        mock_asset_focus.side_effect = side_effect

        result = compute_daily_focus(account_size=50_000, symbols=["Gold", "Nasdaq", "S&P"])

        symbols = [a["symbol"] for a in result["assets"]]
        # S&P (0.9) should come first, then Gold (0.8), then Nasdaq (0.6)
        assert symbols[0] == "S&P"
        assert symbols[1] == "Gold"
        assert symbols[2] == "Nasdaq"

    @patch("lib.services.engine.focus.compute_asset_focus")
    def test_no_trade_when_all_low_quality(self, mock_asset_focus):
        """Should flag no_trade when all assets are below quality threshold."""
        from lib.services.engine.focus import compute_daily_focus

        mock_asset_focus.return_value = {
            "symbol": "Gold",
            "ticker": "MGC=F",
            "bias": "NEUTRAL",
            "bias_emoji": "⚪",
            "last_price": 5200.0,
            "entry_low": 5190.0,
            "entry_high": 5210.0,
            "stop": 5180.0,
            "tp1": 5220.0,
            "tp2": 5240.0,
            "wave_ratio": 0.8,
            "wave_ratio_text": "0.80x",
            "quality": 0.3,
            "quality_pct": 30.0,
            "high_quality": False,
            "vol_percentile": 0.5,
            "vol_cluster": "MEDIUM",
            "market_phase": "RANGING",
            "trend_direction": "NEUTRAL ↔️",
            "momentum_state": "NEUTRAL",
            "dominance_text": "Neutral",
            "position_size": 1,
            "risk_dollars": 50.0,
            "max_risk_allowed": 375.0,
            "atr": 5.0,
            "notes": "Quality too low (30%) — skip today",
            "skip": True,
        }

        result = compute_daily_focus(account_size=50_000, symbols=["Gold"])

        assert result["no_trade"] is True
        assert "quality" in result["no_trade_reason"].lower()


# ---------------------------------------------------------------------------
# publish_focus_to_redis
# ---------------------------------------------------------------------------


class TestPublishFocusToRedis:
    """Test publish_focus_to_redis.

    publish_focus_to_redis does ``from cache import REDIS_AVAILABLE, _r, cache_set``
    inside the function body.  The real ``cache`` module imports ``yfinance`` at the
    top level, which may not be installed in the test venv.  To avoid that we
    inject a lightweight mock into ``sys.modules["lib.core.cache"]`` *before* the function
    runs its lazy import.
    """

    _original_cache_module = None

    def setup_method(self):
        """Save the real cache module before each test."""
        self._original_cache_module = sys.modules.get("lib.core.cache", None)

    def teardown_method(self):
        """Restore the real cache module after each test."""
        if self._original_cache_module is not None:
            sys.modules["lib.core.cache"] = self._original_cache_module
        else:
            sys.modules.pop("lib.core.cache", None)

    def _ensure_mock_cache(self):
        """Install (or refresh) a mock cache module in sys.modules.

        Returns the mock so tests can inspect call args.
        """
        mock_mod = MagicMock()
        mock_mod.REDIS_AVAILABLE = False
        mock_mod._r = None
        mock_mod.cache_set = MagicMock()
        mock_mod.cache_get = MagicMock(return_value=None)
        sys.modules["lib.core.cache"] = mock_mod
        return mock_mod

    def test_publishes_to_cache_key(self):
        """Should write focus payload to engine:daily_focus key."""
        mock_cache = self._ensure_mock_cache()

        from lib.services.engine.focus import publish_focus_to_redis

        data = {
            "assets": [],
            "no_trade": False,
            "no_trade_reason": "",
            "computed_at": "2026-02-27T08:00:00-03:00",
        }

        result = publish_focus_to_redis(data)
        assert result is True

        # Verify cache_set was called with the right key
        calls = mock_cache.cache_set.call_args_list
        keys_written = [c[0][0] for c in calls]
        assert "engine:daily_focus" in keys_written
        assert "engine:daily_focus:ts" in keys_written

    def test_handles_non_serializable_gracefully(self):
        """Should handle data that can't be JSON-serialized."""
        self._ensure_mock_cache()

        from lib.services.engine.focus import publish_focus_to_redis

        # datetime objects are handled via default=str
        data = {
            "assets": [],
            "computed_at": datetime.now(tz=_EST),
        }

        result = publish_focus_to_redis(data)
        assert result is True
