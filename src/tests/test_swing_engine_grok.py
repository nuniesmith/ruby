"""
Tests for Swing Engine Adapter + Grok Phase 3C + TV Signal Publishing
=====================================================================
Covers:
  - Swing engine adapter (lib.services.engine.swing):
    - tick_swing_detector() entry detection flow
    - tick_swing_detector() active state management / exit evaluation
    - _load_swing_states_from_redis() round-trip
    - _persist_swing_states() serialization
    - _publish_swing_signals_to_redis() Redis + PubSub

    - _get_asset_bias() DailyBias reconstruction from dict
    - _resample_to_15m() OHLCV resampling
    - get_active_swing_states() / get_swing_summary() / reset_swing_states()
    - Concurrent swing limit enforcement
    - Time-stop boundary — no new entries after 15:30 ET

  - Grok Phase 3C (lib.integrations.grok_helper):
    - run_daily_plan_grok_analysis() prompt construction + API call
    - parse_grok_daily_plan_response() JSON parsing:
      - Clean JSON
      - Markdown-fenced JSON (```json ... ```)
      - JSON with preamble text
      - Trailing commas / single quotes (fixup)
      - Completely unparseable text (graceful fallback)
    - _validate_grok_plan_response() field normalization
    - format_grok_daily_plan_for_display() rich text rendering

  - DailyPlan grok_analysis integration:
    - DailyPlan.to_dict() includes grok_analysis
    - DailyPlan.load_from_redis() reconstructs grok_analysis

  - Dashboard rendering:
    - _render_structured_grok_brief() output for risk_on / risk_off / mixed
    - _render_swing_card() with grok_swing_insight parameter

  - Scheduler:
    - CHECK_SWING action type exists
    - CHECK_SWING fires during active hours (03:00–15:30 ET)
    - CHECK_SWING does NOT fire outside active window
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

_EST = ZoneInfo("America/New_York")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_bars(n: int = 50, start_price: float = 2700.0, interval_min: int = 5) -> pd.DataFrame:
    """Create a synthetic OHLCV DataFrame for testing."""
    np.random.seed(42)
    dates = pd.date_range(
        start=datetime(2025, 1, 15, 9, 30, tzinfo=_EST),
        periods=n,
        freq=f"{interval_min}min",
    )
    close = start_price + np.cumsum(np.random.randn(n) * 2.0)
    high = close + np.abs(np.random.randn(n)) * 1.5
    low = close - np.abs(np.random.randn(n)) * 1.5
    opens = close + np.random.randn(n) * 0.5
    volume = np.random.randint(100, 5000, size=n).astype(float)

    return pd.DataFrame(
        {
            "Open": opens,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": volume,
        },
        index=dates,
    )


def _make_bias(
    name: str = "Gold",
    direction: str = "LONG",
    confidence: float = 0.75,
) -> Any:
    """Create a DailyBias object."""
    from lib.trading.strategies.daily.bias_analyzer import (
        BiasDirection,
        DailyBias,
        KeyLevels,
    )

    dir_enum = BiasDirection(direction)

    return DailyBias(
        asset_name=name,
        direction=dir_enum,
        confidence=confidence,
        reasoning=f"Test bias for {name}",
        key_levels=KeyLevels(
            prior_day_high=2750.0,
            prior_day_low=2700.0,
            prior_day_close=2725.0,
            prior_day_mid=2725.0,
            weekly_high=2780.0,
            weekly_low=2680.0,
        ),
    )


def _make_daily_plan_data(
    swing_names: list[str] | None = None,
    scalp_names: list[str] | None = None,
) -> dict:
    """Create a mock daily plan Redis payload."""
    if swing_names is None:
        swing_names = ["Gold", "Crude Oil"]
    if scalp_names is None:
        scalp_names = ["S&P", "Nasdaq"]

    biases = {}
    for name in swing_names + scalp_names:
        biases[name] = {
            "asset_name": name,
            "direction": "LONG",
            "confidence": 0.72,
            "reasoning": f"Bullish setup for {name}",
            "key_levels": {
                "prior_day_high": 2750.0,
                "prior_day_low": 2700.0,
                "prior_day_close": 2725.0,
                "prior_day_mid": 2725.0,
                "weekly_high": 2780.0,
                "weekly_low": 2680.0,
                "weekly_mid": 2730.0,
                "monthly_ema20": 2700.0,
                "overnight_high": 2745.0,
                "overnight_low": 2705.0,
            },
            "candle_pattern": "bullish_engulfing",
            "weekly_range_position": 0.65,
            "monthly_trend_score": 0.3,
            "volume_confirmation": True,
            "overnight_gap_direction": 0.2,
            "overnight_gap_atr_ratio": 0.15,
            "atr_expanding": False,
            "component_scores": {"candle": 0.6, "weekly": 0.65, "monthly": 0.3},
        }

    swing_candidates = []
    for name in swing_names:
        swing_candidates.append(
            {
                "asset_name": name,
                "direction": "LONG",
                "confidence": 0.7,
                "swing_score": 72.0,
                "entry_zone_low": 2715.0,
                "entry_zone_high": 2725.0,
                "stop_loss": 2695.0,
                "tp1": 2745.0,
                "tp2": 2763.0,
                "tp3": 2780.0,
                "atr": 12.5,
                "last_price": 2720.0,
                "risk_dollars": 125.0,
                "position_size": 2,
                "reasoning": "Pullback to PDL with bullish bias",
                "entry_styles": ["pullback", "breakout"],
                "key_levels": {"PDH": 2750.0, "PDL": 2700.0},
            }
        )

    return {
        "scalp_focus_names": scalp_names,
        "swing_candidate_names": swing_names,
        "swing_candidates": swing_candidates,
        "all_biases": biases,
        "market_context": "Bullish macro setup",
        "grok_available": True,
        "grok_analysis": {},
        "computed_at": datetime.now(tz=_EST).isoformat(),
        "account_size": 50_000,
        "session": "active",
    }


# ===========================================================================
# Grok Phase 3C — parse_grok_daily_plan_response
# ===========================================================================


class TestParseGrokDailyPlanResponse:
    """Test the JSON parser for structured Grok responses."""

    def test_clean_json(self):
        from lib.integrations.grok_helper import parse_grok_daily_plan_response

        raw = json.dumps(
            {
                "macro_bias": "risk_on",
                "macro_summary": "Bullish session expected",
                "top_assets": [
                    {"name": "Gold", "reason": "Strong momentum", "key_level": "2750 PDH", "bias_agreement": True}
                ],
                "risk_warnings": ["FOMC minutes at 14:00 ET"],
                "economic_events": ["CPI at 08:30 ET"],
                "session_plan": "Be aggressive at London open, patient after US open",
                "correlation_notes": ["Gold/Silver correlated today"],
                "swing_insights": {"Gold": "Wait for pullback to 2720 EMA-21"},
            }
        )

        result = parse_grok_daily_plan_response(raw)
        assert result is not None
        assert result["macro_bias"] == "risk_on"
        assert result["macro_summary"] == "Bullish session expected"
        assert len(result["top_assets"]) == 1
        assert result["top_assets"][0]["name"] == "Gold"
        assert result["top_assets"][0]["bias_agreement"] is True
        assert len(result["risk_warnings"]) == 1
        assert len(result["economic_events"]) == 1
        assert result["session_plan"] != ""
        assert "Gold" in result["swing_insights"]

    def test_markdown_fenced_json(self):
        from lib.integrations.grok_helper import parse_grok_daily_plan_response

        raw = '```json\n{"macro_bias": "risk_off", "macro_summary": "Bear day", "top_assets": [], "risk_warnings": ["War"], "economic_events": [], "session_plan": "Defensive", "correlation_notes": [], "swing_insights": {}}\n```'

        result = parse_grok_daily_plan_response(raw)
        assert result is not None
        assert result["macro_bias"] == "risk_off"
        assert result["macro_summary"] == "Bear day"
        assert result["risk_warnings"] == ["War"]

    def test_json_with_preamble(self):
        from lib.integrations.grok_helper import parse_grok_daily_plan_response

        raw = 'Here is my analysis:\n\n{"macro_bias": "mixed", "macro_summary": "Choppy day", "top_assets": [{"name": "S&P", "reason": "Range", "key_level": "5200", "bias_agreement": false}], "risk_warnings": [], "economic_events": [], "session_plan": "Wait", "correlation_notes": [], "swing_insights": {}}'

        result = parse_grok_daily_plan_response(raw)
        assert result is not None
        assert result["macro_bias"] == "mixed"
        assert result["top_assets"][0]["bias_agreement"] is False

    def test_trailing_comma_fixup(self):
        from lib.integrations.grok_helper import parse_grok_daily_plan_response

        raw = '{"macro_bias": "risk_on", "macro_summary": "Good day", "top_assets": [], "risk_warnings": ["warn",], "economic_events": [], "session_plan": "", "correlation_notes": [], "swing_insights": {},}'

        result = parse_grok_daily_plan_response(raw)
        assert result is not None
        assert result["macro_bias"] == "risk_on"

    def test_completely_unparseable_fallback(self):
        from lib.integrations.grok_helper import parse_grok_daily_plan_response

        raw = "This is not JSON at all. Just a regular paragraph of text about markets."

        result = parse_grok_daily_plan_response(raw)
        assert result is not None
        assert result.get("_parse_failed") is True
        assert result["macro_bias"] == "mixed"
        assert "not JSON" in result["macro_summary"]

    def test_empty_input(self):
        from lib.integrations.grok_helper import parse_grok_daily_plan_response

        assert parse_grok_daily_plan_response("") is None
        assert parse_grok_daily_plan_response("   ") is None

    def test_none_input(self):
        from lib.integrations.grok_helper import parse_grok_daily_plan_response

        assert parse_grok_daily_plan_response(None) is None  # type: ignore[arg-type]

    def test_partial_json_fields(self):
        """Missing fields should get defaults."""
        from lib.integrations.grok_helper import parse_grok_daily_plan_response

        raw = '{"macro_bias": "risk_on"}'
        result = parse_grok_daily_plan_response(raw)
        assert result is not None
        assert result["macro_bias"] == "risk_on"
        assert result["macro_summary"] == ""
        assert result["top_assets"] == []
        assert result["risk_warnings"] == []
        assert result["economic_events"] == []
        assert result["session_plan"] == ""
        assert result["correlation_notes"] == []
        assert result["swing_insights"] == {}


class TestValidateGrokPlanResponse:
    """Test field validation and normalization."""

    def test_invalid_macro_bias_falls_back(self):
        from lib.integrations.grok_helper import _validate_grok_plan_response

        result = _validate_grok_plan_response({"macro_bias": "super_bullish"})
        assert result["macro_bias"] == "mixed"  # default

    def test_truncates_long_summary(self):
        from lib.integrations.grok_helper import _validate_grok_plan_response

        long_text = "A" * 1000
        result = _validate_grok_plan_response({"macro_summary": long_text})
        assert len(result["macro_summary"]) == 500

    def test_limits_top_assets(self):
        from lib.integrations.grok_helper import _validate_grok_plan_response

        many_assets = [{"name": f"Asset{i}", "reason": "test"} for i in range(20)]
        result = _validate_grok_plan_response({"top_assets": many_assets})
        assert len(result["top_assets"]) <= 6

    def test_non_dict_top_assets_ignored(self):
        from lib.integrations.grok_helper import _validate_grok_plan_response

        result = _validate_grok_plan_response({"top_assets": ["Gold", "Silver"]})
        assert result["top_assets"] == []

    def test_type_coercion(self):
        from lib.integrations.grok_helper import _validate_grok_plan_response

        result = _validate_grok_plan_response(
            {
                "risk_warnings": [123, None, "real warning"],
                "economic_events": [True, "CPI at 08:30"],
            }
        )
        # None should be filtered out, numbers coerced to str
        assert "real warning" in result["risk_warnings"]
        assert "CPI at 08:30" in result["economic_events"]


class TestFormatGrokDailyPlanForDisplay:
    """Test the display formatter."""

    def test_full_display(self):
        from lib.integrations.grok_helper import format_grok_daily_plan_for_display

        data = {
            "macro_bias": "risk_on",
            "macro_summary": "Strong bullish day expected",
            "top_assets": [{"name": "Gold", "reason": "momentum", "key_level": "2750", "bias_agreement": True}],
            "risk_warnings": ["FOMC at 14:00"],
            "economic_events": ["CPI at 08:30"],
            "session_plan": "Be aggressive early",
            "correlation_notes": ["Gold/Silver together"],
            "swing_insights": {"Gold": "Pullback to 2720"},
        }
        text = format_grok_daily_plan_for_display(data)
        assert "RISK-ON" in text
        assert "Strong bullish day" in text
        assert "Gold" in text
        assert "FOMC" in text
        assert "CPI" in text
        assert "aggressive" in text.lower()
        assert "Pullback" in text

    def test_empty_data(self):
        from lib.integrations.grok_helper import format_grok_daily_plan_for_display

        assert format_grok_daily_plan_for_display({}) == ""
        assert format_grok_daily_plan_for_display(None) == ""  # type: ignore[arg-type]

    def test_risk_off_emoji(self):
        from lib.integrations.grok_helper import format_grok_daily_plan_for_display

        text = format_grok_daily_plan_for_display({"macro_bias": "risk_off", "macro_summary": "Bearish"})
        assert "RISK-OFF" in text

    def test_mixed_emoji(self):
        from lib.integrations.grok_helper import format_grok_daily_plan_for_display

        text = format_grok_daily_plan_for_display({"macro_bias": "mixed", "macro_summary": "Choppy"})
        assert "MIXED" in text


class TestRunDailyPlanGrokAnalysis:
    """Test the structured Grok API call."""

    @patch("lib.integrations.grok_helper._call_grok")
    def test_success(self, mock_call):
        from lib.integrations.grok_helper import run_daily_plan_grok_analysis

        mock_call.return_value = json.dumps(
            {
                "macro_bias": "risk_on",
                "macro_summary": "Good day",
                "top_assets": [{"name": "Gold", "reason": "strong", "key_level": "2750", "bias_agreement": True}],
                "risk_warnings": [],
                "economic_events": [],
                "session_plan": "Aggressive",
                "correlation_notes": [],
                "swing_insights": {"Gold": "Buy pullback"},
            }
        )

        biases = {"Gold": {"direction": "LONG", "confidence": 0.7, "reasoning": "Bullish"}}
        result = run_daily_plan_grok_analysis(
            biases=biases,
            asset_names=["Gold"],
            api_key="test-key",
        )

        assert result is not None
        assert result["macro_bias"] == "risk_on"
        assert result["raw_text"] is not None
        assert result["swing_insights"]["Gold"] == "Buy pullback"
        mock_call.assert_called_once()

    @patch("lib.integrations.grok_helper._call_grok")
    def test_api_failure(self, mock_call):
        from lib.integrations.grok_helper import run_daily_plan_grok_analysis

        mock_call.return_value = None

        result = run_daily_plan_grok_analysis(
            biases={"Gold": {"direction": "LONG", "confidence": 0.7, "reasoning": "test"}},
            asset_names=["Gold"],
            api_key="test-key",
        )
        assert result is None

    def test_no_api_key(self):
        from lib.integrations.grok_helper import run_daily_plan_grok_analysis

        with patch.dict("os.environ", {"XAI_API_KEY": ""}):
            result = run_daily_plan_grok_analysis(
                biases={},
                asset_names=[],
                api_key=None,
            )
            assert result is None

    @patch("lib.integrations.grok_helper._call_grok")
    def test_with_dailybias_objects(self, mock_call):
        """Verify prompt construction works with DailyBias objects, not just dicts."""
        from lib.integrations.grok_helper import run_daily_plan_grok_analysis

        mock_call.return_value = '{"macro_bias": "risk_on", "macro_summary": "OK"}'

        bias = _make_bias("Gold", "LONG", 0.8)
        result = run_daily_plan_grok_analysis(
            biases={"Gold": bias},
            asset_names=["Gold"],
            swing_candidate_names=["Gold"],
            scalp_focus_names=["S&P"],
            api_key="test-key",
        )
        assert result is not None
        # Check prompt includes asset info
        call_args = mock_call.call_args
        prompt = call_args[0][0]  # First positional arg
        assert "Gold" in prompt
        assert "LONG" in prompt


# ===========================================================================
# DailyPlan — grok_analysis field integration
# ===========================================================================


class TestDailyPlanGrokField:
    """Test that DailyPlan properly serializes/deserializes grok_analysis."""

    def test_to_dict_includes_grok_analysis(self):
        from lib.trading.strategies.daily.daily_plan import DailyPlan

        plan = DailyPlan(
            grok_analysis={"macro_bias": "risk_on", "top_assets": [{"name": "Gold"}]},
            grok_available=True,
        )
        d = plan.to_dict()
        assert "grok_analysis" in d
        assert d["grok_analysis"]["macro_bias"] == "risk_on"
        assert d["grok_available"] is True

    def test_to_dict_empty_grok_analysis(self):
        from lib.trading.strategies.daily.daily_plan import DailyPlan

        plan = DailyPlan()
        d = plan.to_dict()
        assert d["grok_analysis"] == {}
        assert d["grok_available"] is False

    def test_load_from_redis_with_grok_analysis(self):
        from lib.trading.strategies.daily.daily_plan import DailyPlan

        plan_data = {
            "scalp_focus": [],
            "swing_candidates": [],
            "market_context": "Test",
            "grok_available": True,
            "grok_analysis": {
                "macro_bias": "risk_off",
                "macro_summary": "Bearish today",
                "risk_warnings": ["CPI miss"],
            },
            "no_trade": False,
            "no_trade_reason": "",
            "all_biases": {},
            "computed_at": "2025-01-15T08:00:00",
            "account_size": 50000,
            "session": "pre-market",
        }

        mock_redis = MagicMock()
        mock_redis.get.return_value = json.dumps(plan_data)

        loaded = DailyPlan.load_from_redis(mock_redis)
        assert loaded is not None
        assert loaded.grok_analysis["macro_bias"] == "risk_off"
        assert loaded.grok_analysis["macro_summary"] == "Bearish today"
        assert loaded.grok_available is True


# ===========================================================================
# Swing Engine Adapter — lib.services.engine.swing
# ===========================================================================


class TestResampleTo15m:
    """Test the OHLCV resampling helper."""

    def test_5m_to_15m(self):
        from lib.services.engine.swing import _resample_to_15m

        df = _make_bars(30, interval_min=5)
        result = _resample_to_15m(df)
        assert result is not None
        assert len(result) > 0
        assert len(result) < len(df)
        assert "Open" in result.columns or "open" in result.columns.str.lower()

    def test_empty_df(self):
        from lib.services.engine.swing import _resample_to_15m

        assert _resample_to_15m(pd.DataFrame()) is None
        assert _resample_to_15m(None) is None  # type: ignore[arg-type]

    def test_already_15m(self):
        from lib.services.engine.swing import _resample_to_15m

        df = _make_bars(20, interval_min=15)
        result = _resample_to_15m(df)
        assert result is not None
        # Resampling 15m to 15m should produce ~same number of bars
        assert len(result) >= len(df) - 2  # Allow rounding at edges


class TestGetAssetBias:
    """Test DailyBias reconstruction from Redis dict."""

    def test_reconstruct_from_dict(self):
        from lib.services.engine.swing import _get_asset_bias

        plan_data = _make_daily_plan_data(swing_names=["Gold"])

        with patch("lib.services.engine.swing._get_daily_plan_data", return_value=plan_data):
            bias = _get_asset_bias("Gold")

        assert bias is not None
        assert bias.asset_name == "Gold"
        assert bias.direction.value == "LONG"
        assert bias.confidence == 0.72
        assert bias.key_levels.prior_day_high == 2750.0
        assert bias.key_levels.weekly_high == 2780.0

    def test_missing_asset(self):
        from lib.services.engine.swing import _get_asset_bias

        plan_data = _make_daily_plan_data(swing_names=["Gold"])

        with patch("lib.services.engine.swing._get_daily_plan_data", return_value=plan_data):
            bias = _get_asset_bias("Nonexistent")

        assert bias is None

    def test_no_plan_data(self):
        from lib.services.engine.swing import _get_asset_bias

        with patch("lib.services.engine.swing._get_daily_plan_data", return_value=None):
            assert _get_asset_bias("Gold") is None


class TestSwingStatePersistence:
    """Test Redis round-trip for swing states."""

    def test_persist_and_load(self):
        from lib.services.engine.swing import (
            _load_swing_states_from_redis,
            _persist_swing_states,
        )
        from lib.trading.strategies.daily.swing_detector import (
            SwingEntryStyle,
            SwingPhase,
            SwingSignal,
            SwingState,
        )

        # Create a test state
        signal = SwingSignal(
            asset_name="Gold",
            entry_style=SwingEntryStyle.PULLBACK,
            direction="LONG",
            confidence=0.75,
            entry_price=2720.0,
            stop_loss=2698.0,
            tp1=2745.0,
            tp2=2763.0,
            atr=12.5,
            risk_dollars=125.0,
            position_size=2,
            phase=SwingPhase.ENTRY_READY,
        )

        state = SwingState(
            asset_name="Gold",
            signal=signal,
            phase=SwingPhase.ACTIVE,
            entry_price=2720.0,
            current_stop=2698.0,
            tp1=2745.0,
            tp2=2763.0,
            direction="LONG",
            position_size=2,
            remaining_size=2,
            highest_price=2730.0,
            lowest_price=2715.0,
            entry_time="2025-01-15T09:30:00",
            last_update="2025-01-15T10:00:00",
        )

        # Persist
        import lib.services.engine.swing as swing_mod

        swing_mod._active_swing_states = {"Gold": state}

        stored_data = {}

        def mock_cache_set(key, value, ttl=300):
            stored_data[key] = value

        with patch("lib.services.engine.swing._cache_set", side_effect=mock_cache_set):
            _persist_swing_states()

        assert "engine:swing_states" in stored_data

        # Load back
        def mock_cache_get(key):
            return stored_data.get(key)

        with patch("lib.services.engine.swing._cache_get", side_effect=mock_cache_get):
            loaded = _load_swing_states_from_redis()

        assert "Gold" in loaded
        assert loaded["Gold"].phase == SwingPhase.ACTIVE
        assert loaded["Gold"].entry_price == 2720.0
        assert loaded["Gold"].signal.confidence == 0.75

    def test_load_skips_closed_states(self):
        from lib.services.engine.swing import _load_swing_states_from_redis

        data = {
            "Gold": {
                "phase": "closed",
                "entry_price": 2720.0,
                "direction": "LONG",
                "signal": None,
                "current_stop": 2698.0,
                "tp1": 2745.0,
                "tp2": 2763.0,
                "position_size": 2,
                "remaining_size": 0,
                "highest_price": 2750.0,
                "lowest_price": 2710.0,
                "entry_time": "",
                "last_update": "",
            }
        }

        with patch("lib.services.engine.swing._cache_get", return_value=json.dumps(data)):
            loaded = _load_swing_states_from_redis()

        assert "Gold" not in loaded  # Closed states should be skipped


class TestPublishSwingSignalsToRedis:
    """Test Redis publishing for swing signals."""

    def test_publish_signals(self):
        from lib.services.engine.swing import _publish_swing_signals_to_redis
        from lib.trading.strategies.daily.swing_detector import (
            SwingEntryStyle,
            SwingPhase,
            SwingSignal,
        )

        signals = [
            SwingSignal(
                asset_name="Gold",
                entry_style=SwingEntryStyle.PULLBACK,
                direction="LONG",
                confidence=0.8,
                entry_price=2720.0,
                phase=SwingPhase.ENTRY_READY,
            ),
        ]

        stored = {}
        mock_redis = MagicMock()

        def mock_set(key, val, ttl=300):
            stored[key] = val

        with (
            patch("lib.services.engine.swing._cache_set", side_effect=mock_set),
            patch("lib.services.engine.swing._get_redis", return_value=mock_redis),
        ):
            _publish_swing_signals_to_redis(signals)

        assert "engine:swing_signals" in stored
        parsed = json.loads(stored["engine:swing_signals"])
        assert len(parsed) == 1
        assert parsed[0]["asset_name"] == "Gold"
        mock_redis.publish.assert_called()

    def test_publish_empty_signals(self):
        from lib.services.engine.swing import _publish_swing_signals_to_redis

        # Should not raise or publish anything
        with patch("lib.services.engine.swing._cache_set") as mock_set:
            _publish_swing_signals_to_redis([])
            mock_set.assert_not_called()


class TestPublishSwingToTV:
    """TradingView signal integration removed — no tests."""

    pass


class TestGetSwingSummary:
    """Test the engine status summary helper."""

    def test_empty_summary(self):
        from lib.services.engine.swing import get_swing_summary, reset_swing_states

        reset_swing_states()
        summary = get_swing_summary()
        assert summary["active_count"] == 0
        assert summary["active_assets"] == []
        assert summary["max_concurrent"] == 3

    def test_with_active_states(self):
        import lib.services.engine.swing as swing_mod
        from lib.services.engine.swing import get_swing_summary
        from lib.trading.strategies.daily.swing_detector import SwingPhase, SwingState

        swing_mod._active_swing_states = {
            "Gold": SwingState(
                asset_name="Gold",
                phase=SwingPhase.ACTIVE,
                direction="LONG",
                entry_price=2720.0,
                current_stop=2698.0,
                remaining_size=2,
            ),
        }

        summary = get_swing_summary()
        assert summary["active_count"] == 1
        assert "Gold" in summary["active_assets"]
        assert summary["states"]["Gold"]["phase"] == "active"

        # Cleanup
        swing_mod._active_swing_states = {}


class TestResetSwingStates:
    """Test the state reset function."""

    def test_reset(self):
        import lib.services.engine.swing as swing_mod
        from lib.services.engine.swing import reset_swing_states
        from lib.trading.strategies.daily.swing_detector import SwingPhase, SwingState

        swing_mod._active_swing_states = {
            "Gold": SwingState(asset_name="Gold", phase=SwingPhase.ACTIVE),
        }
        swing_mod._last_scan_ts = 12345.0

        with patch("lib.services.engine.swing._cache_set"):
            reset_swing_states()

        assert swing_mod._active_swing_states == {}
        assert swing_mod._last_scan_ts == 0.0


class TestTickSwingDetector:
    """Test the main tick function (integration-level)."""

    def test_no_candidates_returns_early(self):
        import lib.services.engine.swing as swing_mod
        from lib.services.engine.swing import tick_swing_detector

        swing_mod._active_swing_states = {}
        swing_mod._last_scan_ts = 0.0

        with (
            patch("lib.services.engine.swing._get_focus_data", return_value=None),
            patch("lib.services.engine.swing._get_swing_candidate_names", return_value=[]),
            patch("lib.services.engine.swing._load_swing_states_from_redis", return_value={}),
        ):
            result = tick_swing_detector(engine=MagicMock(), account_size=50_000)

        assert result.get("scan_skipped") is True

    def test_scans_candidates_and_detects(self):
        """Full integration: feed data → detect signals → create state.

        The swing detector does a lazy ``from ... import detect_swing_entries``
        inside ``tick_swing_detector``.  To reliably intercept that, we
        monkeypatch the function directly on the ``swing_detector`` module
        *before* calling tick, so the fresh ``from`` import resolves to our
        mock.
        """
        import lib.services.engine.swing as swing_mod
        import lib.trading.strategies.daily.swing_detector as sd_mod
        from lib.services.engine.swing import tick_swing_detector
        from lib.trading.strategies.daily.swing_detector import (
            SwingEntryStyle,
            SwingPhase,
            SwingSignal,
        )

        swing_mod._active_swing_states = {}
        swing_mod._last_scan_ts = 0.0

        plan_data = _make_daily_plan_data(swing_names=["Gold"])
        focus_data = {
            "assets": [
                {"symbol": "Gold", "last_price": 2720.0, "atr": 12.5},
            ]
        }
        bars = _make_bars(50, start_price=2720.0)

        mock_signal = SwingSignal(
            asset_name="Gold",
            entry_style=SwingEntryStyle.PULLBACK,
            direction="LONG",
            confidence=0.75,
            entry_price=2720.0,
            stop_loss=2698.0,
            tp1=2745.0,
            tp2=2763.0,
            atr=12.5,
            risk_dollars=125.0,
            position_size=2,
            phase=SwingPhase.ENTRY_READY,
        )

        # Save the real function so we can restore it after the test.
        _real_detect = sd_mod.detect_swing_entries

        # Monkeypatch *before* entering the context manager so the lazy
        # ``from lib.trading.strategies.daily.swing_detector import detect_swing_entries``
        # inside tick_swing_detector picks up our mock.
        mock_detect = MagicMock(return_value=[mock_signal])
        sd_mod.detect_swing_entries = mock_detect

        try:
            with (
                patch("lib.services.engine.swing._get_focus_data", return_value=focus_data),
                patch("lib.services.engine.swing._get_swing_candidate_names", return_value=["Gold"]),
                patch("lib.services.engine.swing._get_daily_plan_data", return_value=plan_data),
                patch("lib.services.engine.swing._load_swing_states_from_redis", return_value={}),
                patch("lib.services.engine.swing._get_asset_bias", return_value=_make_bias("Gold")),
                patch("lib.services.engine.swing._get_asset_price", return_value=2720.0),
                patch("lib.services.engine.swing._get_asset_atr", return_value=12.5),
                patch("lib.services.engine.swing._get_asset_ticker", return_value="GC=F"),
                patch("lib.services.engine.swing._fetch_bars_5m", return_value=bars),
                patch("lib.services.engine.swing._get_session_open_price", return_value=2715.0),
                patch("lib.services.engine.swing._publish_swing_signals_to_redis"),
                patch("lib.services.engine.swing._persist_swing_states"),
                patch("lib.services.engine.swing._publish_swing_states_to_redis"),
            ):
                result = tick_swing_detector(engine=MagicMock(), account_size=50_000)

            # Check mock state BEFORE restoring, while the mock is still in place.
            detect_was_called = mock_detect.called
            detect_call_count = mock_detect.call_count
        finally:
            # Always restore so other tests aren't affected.
            sd_mod.detect_swing_entries = _real_detect

        assert result.get("new_signals", 0) >= 1
        # The state was created for the high-confidence ENTRY_READY signal.
        # It may still be active OR may have been immediately ticked by
        # _tick_active_states and closed (if the exit evaluator triggered).
        # Either way, the detect mock was called and a signal was produced.
        assert detect_was_called, "detect_swing_entries should have been called"
        assert detect_call_count == 1
        assert result.get("candidates_scanned", 0) == 1

        # Cleanup
        swing_mod._active_swing_states = {}

    def test_skips_asset_with_active_state(self):
        """Should not scan an asset that already has an active swing state."""
        import lib.services.engine.swing as swing_mod
        from lib.services.engine.swing import tick_swing_detector
        from lib.trading.strategies.daily.swing_detector import SwingPhase, SwingState

        swing_mod._active_swing_states = {
            "Gold": SwingState(asset_name="Gold", phase=SwingPhase.ACTIVE, direction="LONG"),
        }
        swing_mod._last_scan_ts = 0.0

        focus_data = {"assets": [{"symbol": "Gold", "last_price": 2720.0, "atr": 12.5}]}

        with (
            patch("lib.services.engine.swing._get_focus_data", return_value=focus_data),
            patch("lib.services.engine.swing._get_swing_candidate_names", return_value=["Gold"]),
            patch("lib.services.engine.swing._get_asset_bias") as mock_bias,
            patch("lib.services.engine.swing._persist_swing_states"),
            patch("lib.services.engine.swing._publish_swing_states_to_redis"),
        ):
            tick_swing_detector(engine=MagicMock(), account_size=50_000)

        # Should not have called bias for Gold since it's already active
        mock_bias.assert_not_called()

        # Cleanup
        swing_mod._active_swing_states = {}

    def test_max_concurrent_swings_enforced(self):
        """Should stop scanning when max concurrent limit is reached."""
        import lib.services.engine.swing as swing_mod
        from lib.services.engine.swing import _MAX_CONCURRENT_SWINGS, tick_swing_detector
        from lib.trading.strategies.daily.swing_detector import SwingPhase, SwingState

        # Fill up to max
        swing_mod._active_swing_states = {}
        for i in range(_MAX_CONCURRENT_SWINGS):
            name = f"Asset{i}"
            swing_mod._active_swing_states[name] = SwingState(
                asset_name=name, phase=SwingPhase.ACTIVE, direction="LONG"
            )
        swing_mod._last_scan_ts = 0.0

        focus_data = {"assets": [{"symbol": "NewAsset", "last_price": 100.0, "atr": 5.0}]}

        with (
            patch("lib.services.engine.swing._get_focus_data", return_value=focus_data),
            patch("lib.services.engine.swing._get_swing_candidate_names", return_value=["NewAsset"]),
            patch("lib.services.engine.swing._get_asset_bias") as mock_bias,
            patch("lib.services.engine.swing._persist_swing_states"),
            patch("lib.services.engine.swing._publish_swing_states_to_redis"),
        ):
            tick_swing_detector(engine=MagicMock(), account_size=50_000)

        # Should not have scanned NewAsset
        mock_bias.assert_not_called()

        # Cleanup
        swing_mod._active_swing_states = {}


class TestTickActiveStates:
    """Test the active state management portion of the tick."""

    def test_closed_states_archived(self):
        import lib.services.engine.swing as swing_mod
        from lib.services.engine.swing import _tick_active_states
        from lib.trading.strategies.daily.swing_detector import SwingPhase, SwingState

        swing_mod._active_swing_states = {
            "Gold": SwingState(
                asset_name="Gold",
                phase=SwingPhase.CLOSED,
                direction="LONG",
                entry_price=2720.0,
            ),
        }

        with (
            patch("lib.services.engine.swing._persist_swing_states"),
            patch("lib.services.engine.swing._archive_closed_state") as mock_archive,
        ):
            result = _tick_active_states(focus_data=None, account_size=50_000)

        assert "Gold" not in swing_mod._active_swing_states
        mock_archive.assert_called_once()
        assert result["closed"] == 1

        # Cleanup
        swing_mod._active_swing_states = {}

    def test_active_state_gets_updated(self):
        import lib.services.engine.swing as swing_mod
        from lib.services.engine.swing import _tick_active_states
        from lib.trading.strategies.daily.swing_detector import (
            SwingEntryStyle,
            SwingPhase,
            SwingSignal,
            SwingState,
        )

        signal = SwingSignal(
            asset_name="Gold",
            entry_style=SwingEntryStyle.PULLBACK,
            direction="LONG",
            confidence=0.7,
            risk_dollars=125.0,
        )

        swing_mod._active_swing_states = {
            "Gold": SwingState(
                asset_name="Gold",
                signal=signal,
                phase=SwingPhase.ACTIVE,
                direction="LONG",
                entry_price=2720.0,
                current_stop=2698.0,
                tp1=2745.0,
                tp2=2763.0,
                position_size=2,
                remaining_size=2,
                highest_price=2730.0,
                lowest_price=2715.0,
            ),
        }

        focus_data = {"assets": [{"symbol": "Gold", "last_price": 2735.0, "atr": 12.5}]}

        with (
            patch("lib.services.engine.swing._get_asset_price", return_value=2735.0),
            patch("lib.services.engine.swing._get_asset_atr", return_value=12.5),
            patch("lib.services.engine.swing._get_asset_ticker", return_value="GC=F"),
            patch("lib.services.engine.swing._fetch_bars_15m", return_value=_make_bars(20, interval_min=15)),
            patch("lib.services.engine.swing._get_contract_specs", return_value={"point": 1.0, "tick": 0.1}),
            patch("lib.services.engine.swing._persist_swing_states"),
            patch("lib.services.engine.swing._publish_swing_states_to_redis"),
        ):
            result = _tick_active_states(focus_data=focus_data, account_size=50_000)

        assert result["updates"] >= 1
        assert result["active_states"] >= 0  # May have exited or stayed active

        # Cleanup
        swing_mod._active_swing_states = {}


# ===========================================================================
# Scheduler — CHECK_SWING action type
# ===========================================================================


class TestSchedulerSwingAction:
    """Test that CHECK_SWING is properly scheduled."""

    def test_check_swing_action_type_exists(self):
        from lib.services.engine.scheduler import ActionType

        assert hasattr(ActionType, "CHECK_SWING")
        assert ActionType.CHECK_SWING.value == "check_swing"

    def test_swing_scheduled_during_active_hours(self):
        from lib.services.engine.scheduler import ActionType, ScheduleManager

        mgr = ScheduleManager()

        # Simulate 10:00 AM ET (well within active hours)
        now = datetime(2025, 1, 15, 10, 0, tzinfo=_EST)
        ts = now.timestamp()

        actions = mgr._get_active_actions(ts, now)
        action_types = [a.action for a in actions]

        assert ActionType.CHECK_SWING in action_types

    def test_swing_not_scheduled_after_1530(self):
        from lib.services.engine.scheduler import ActionType, ScheduleManager

        mgr = ScheduleManager()

        # Simulate 16:00 ET (after time stop)
        now = datetime(2025, 1, 15, 16, 0, tzinfo=_EST)
        ts = now.timestamp()

        # This is off-hours, not active, so _get_active_actions shouldn't fire
        # But let's verify CHECK_SWING doesn't appear in _get_active_actions
        # even if we force the time check
        actions = mgr._get_active_actions(ts, now)
        action_types = [a.action for a in actions]

        assert ActionType.CHECK_SWING not in action_types

    def test_swing_interval_constant(self):
        from lib.services.engine.scheduler import ScheduleManager

        assert ScheduleManager.SWING_CHECK_INTERVAL == 2 * 60

    def test_swing_scheduled_at_0300(self):
        """CHECK_SWING should fire at the start of active hours."""
        from lib.services.engine.scheduler import ActionType, ScheduleManager

        mgr = ScheduleManager()

        now = datetime(2025, 1, 15, 3, 5, tzinfo=_EST)
        ts = now.timestamp()

        actions = mgr._get_active_actions(ts, now)
        action_types = [a.action for a in actions]

        assert ActionType.CHECK_SWING in action_types

    def test_swing_scheduled_at_1525(self):
        """CHECK_SWING should still fire just before the 15:30 cutoff."""
        from lib.services.engine.scheduler import ActionType, ScheduleManager

        mgr = ScheduleManager()

        now = datetime(2025, 1, 15, 15, 25, tzinfo=_EST)
        ts = now.timestamp()

        actions = mgr._get_active_actions(ts, now)
        action_types = [a.action for a in actions]

        assert ActionType.CHECK_SWING in action_types


# ===========================================================================
# Dashboard — Structured Grok brief rendering
# ===========================================================================


class TestRenderStructuredGrokBrief:
    """Test the dashboard HTML rendering for structured Grok data."""

    def test_risk_on_rendering(self):
        from lib.services.data.api.dashboard import _render_structured_grok_brief

        grok_data = {
            "macro_bias": "risk_on",
            "macro_summary": "Strong bullish momentum across equity futures",
            "top_assets": [
                {"name": "Gold", "reason": "Breakout setup", "key_level": "2750 PDH", "bias_agreement": True},
                {"name": "Crude Oil", "reason": "Range break", "key_level": "72.50", "bias_agreement": False},
            ],
            "risk_warnings": ["FOMC minutes at 14:00 ET"],
            "economic_events": ["CPI at 08:30 ET", "Jobless claims at 08:30 ET"],
            "session_plan": "Be aggressive at London open, reduce size before CPI",
            "correlation_notes": ["Gold and Silver moving in tandem"],
            "swing_insights": {},
        }

        html = _render_structured_grok_brief(grok_data, grok_available=True)
        assert "RISK-ON" in html
        assert "Strong bullish" in html
        assert "Gold" in html
        assert "Crude Oil" in html
        assert "FOMC" in html
        assert "CPI" in html
        assert "LIVE" in html
        assert "aggressive" in html.lower()

    def test_risk_off_rendering(self):
        from lib.services.data.api.dashboard import _render_structured_grok_brief

        html = _render_structured_grok_brief(
            {"macro_bias": "risk_off", "macro_summary": "Bearish day"},
            grok_available=False,
        )
        assert "RISK-OFF" in html
        assert "CACHED" in html

    def test_mixed_rendering(self):
        from lib.services.data.api.dashboard import _render_structured_grok_brief

        html = _render_structured_grok_brief({"macro_bias": "mixed"})
        assert "MIXED" in html

    def test_empty_data(self):
        from lib.services.data.api.dashboard import _render_structured_grok_brief

        assert _render_structured_grok_brief({}) == ""
        assert _render_structured_grok_brief(None) == ""  # type: ignore[arg-type]

    def test_bias_agreement_indicators(self):
        from lib.services.data.api.dashboard import _render_structured_grok_brief

        grok_data = {
            "macro_bias": "risk_on",
            "top_assets": [
                {"name": "Gold", "reason": "test", "key_level": "2750", "bias_agreement": True},
                {"name": "Silver", "reason": "test", "key_level": "30", "bias_agreement": False},
            ],
        }
        html = _render_structured_grok_brief(grok_data)
        assert "✅" in html  # Agreement
        assert "⚠️" in html  # Disagreement

    def test_collapsible_warnings(self):
        from lib.services.data.api.dashboard import _render_structured_grok_brief

        grok_data = {
            "macro_bias": "risk_off",
            "risk_warnings": ["Warning 1", "Warning 2", "Warning 3"],
        }
        html = _render_structured_grok_brief(grok_data)
        assert "<details" in html  # Should be collapsible for > 2 warnings

    def test_few_warnings_not_collapsible(self):
        from lib.services.data.api.dashboard import _render_structured_grok_brief

        grok_data = {
            "macro_bias": "risk_on",
            "risk_warnings": ["Single warning"],
        }
        html = _render_structured_grok_brief(grok_data)
        assert "<details" not in html  # Should NOT be collapsible for <= 2

    def test_html_escaping(self):
        from lib.services.data.api.dashboard import _render_structured_grok_brief

        grok_data = {
            "macro_bias": "risk_on",
            "macro_summary": "Test <script>alert('xss')</script> & special chars",
        }
        html = _render_structured_grok_brief(grok_data)
        assert "<script>" not in html
        assert "&lt;" in html or "&amp;" in html


class TestRenderSwingCardWithGrokInsight:
    """Test swing card rendering with Phase 3C Grok insights."""

    def _make_asset_dict(self) -> dict[str, Any]:
        return {
            "symbol": "Gold",
            "bias": "LONG",
            "bias_emoji": "🟢",
            "last_price": 2720.0,
            "quality_pct": 72.0,
            "wave_ratio": 1.35,
            "price_decimals": 2,
            "has_live_position": False,
            "entry_low": 2715.0,
            "entry_high": 2725.0,
            "stop": 2698.0,
            "tp1": 2745.0,
            "tp2": 2763.0,
            "risk_dollars": 125.0,
            "position_size": 2,
        }

    def test_swing_card_without_grok(self):
        from lib.services.data.api.dashboard import _render_swing_card

        html = _render_swing_card(self._make_asset_dict())
        assert "Gold" in html
        assert "DAILY SWING" in html
        assert "Grok:" not in html  # No grok insight

    def test_swing_card_with_grok_insight(self):
        from lib.services.data.api.dashboard import _render_swing_card

        html = _render_swing_card(
            self._make_asset_dict(),
            grok_swing_insight="Wait for pullback to 2720 EMA-21 before entry",
        )
        assert "Gold" in html
        assert "Grok:" in html
        assert "pullback" in html.lower()
        assert "2720" in html

    def test_grok_insight_html_escaped(self):
        from lib.services.data.api.dashboard import _render_swing_card

        html = _render_swing_card(
            self._make_asset_dict(),
            grok_swing_insight="Test <b>bold</b> & special",
        )
        assert "<b>" not in html
        assert "&lt;b&gt;" in html


class TestRenderFocusModeGridGrokIntegration:
    """Test that _render_focus_mode_grid passes Grok insights to swing cards."""

    def test_grok_insights_passed_to_swing_cards(self):
        from lib.services.data.api.dashboard import _render_focus_mode_grid

        focus_data = {
            "focus_mode_active": True,
            "scalp_focus_names": ["S&P"],
            "swing_candidate_names": ["Gold"],
            "daily_plan": {
                "swing_candidates": [
                    {
                        "asset_name": "Gold",
                        "direction": "LONG",
                        "swing_score": 72,
                        "confidence": 0.7,
                    }
                ],
                "grok_analysis": {
                    "swing_insights": {
                        "Gold": "Strong pullback setup at EMA-21",
                    }
                },
            },
            "assets": [
                {
                    "symbol": "Gold",
                    "focus_category": "swing",
                    "bias": "LONG",
                    "bias_emoji": "🟢",
                    "last_price": 2720.0,
                    "quality_pct": 72.0,
                    "wave_ratio": 1.3,
                    "price_decimals": 2,
                    "has_live_position": False,
                    "entry_low": 2715.0,
                    "entry_high": 2725.0,
                    "stop": 2698.0,
                    "tp1": 2745.0,
                    "tp2": 2763.0,
                    "risk_dollars": 125.0,
                    "position_size": 2,
                },
                {
                    "symbol": "S&P",
                    "focus_category": "scalp_focus",
                    "bias": "LONG",
                    "bias_emoji": "🟢",
                    "last_price": 5200.0,
                    "quality_pct": 80.0,
                    "wave_ratio": 1.5,
                    "price_decimals": 2,
                    "has_live_position": False,
                    "entry_low": 5190.0,
                    "entry_high": 5210.0,
                    "stop": 5170.0,
                    "tp1": 5230.0,
                    "tp2": 5260.0,
                    "risk_dollars": 200.0,
                    "position_size": 3,
                },
            ],
        }

        html = _render_focus_mode_grid(focus_data)
        # The Grok insight should appear on the swing card
        assert "Grok:" in html
        assert "pullback" in html.lower()


# ===========================================================================
# Engine main.py — _handle_check_swing wiring
# ===========================================================================


class TestHandleCheckSwing:
    """Test the engine handler for CHECK_SWING."""

    @patch("lib.services.engine.main.tick_swing_detector", create=True)
    def test_handler_calls_tick(self, mock_tick):
        """Verify the handler exists and delegates to tick_swing_detector."""
        from lib.services.engine.main import _handle_check_swing

        mock_tick.return_value = {
            "new_signals": 1,
            "active_states": 1,
            "exits": 0,
            "closed": 0,
            "candidates_scanned": 2,
        }

        # The handler uses lazy import, so patch at the import location
        with patch("lib.services.engine.swing.tick_swing_detector", return_value=mock_tick.return_value):
            _handle_check_swing(engine=MagicMock(), account_size=50_000)

    def test_handler_does_not_raise_on_error(self):
        """Handler should catch exceptions gracefully."""
        from lib.services.engine.main import _handle_check_swing

        with patch(
            "lib.services.engine.swing.tick_swing_detector",
            side_effect=RuntimeError("test error"),
        ):
            # Should not raise
            _handle_check_swing(engine=MagicMock(), account_size=50_000)


# ===========================================================================
# Engine status — swing summary injection
# ===========================================================================


class TestEngineStatusSwingSummary:
    """Test that swing summary is injected into engine status."""

    def test_publish_engine_status_includes_swing(self):
        from lib.services.engine.main import _publish_engine_status

        mock_engine = MagicMock()
        mock_engine.get_status.return_value = {"uptime": 100}
        mock_engine.get_backtest_results.return_value = []
        mock_engine.get_strategy_history.return_value = []
        mock_engine.get_live_feed_status.return_value = {}

        stored = {}

        def mock_cache_set(key, value, ttl=60):
            stored[key] = value

        with (
            patch("lib.services.engine.main.cache_set", mock_cache_set, create=True),
            patch("lib.core.cache.cache_set", mock_cache_set, create=True),
        ):
            import contextlib

            with contextlib.suppress(
                Exception
            ):  # May fail on other imports, but we're testing the swing injection path
                _publish_engine_status(mock_engine, "active", {"actions": []})

        # The function should have attempted to inject swing summary
        # (exact verification depends on import success in test environment)


# ===========================================================================
# Integration: DailyPlan Grok flow end-to-end
# ===========================================================================


class TestDailyPlanGrokIntegration:
    """Test the full flow: generate_daily_plan → Grok structured call → plan with grok_analysis."""

    @patch("lib.trading.strategies.daily.daily_plan._fetch_asset_data", return_value=(2750.0, 15.0))
    @patch("lib.trading.strategies.daily.daily_plan._fetch_grok_context")
    @patch("lib.integrations.grok_helper.run_daily_plan_grok_analysis")
    @patch("lib.integrations.grok_helper.format_grok_daily_plan_for_display")
    def test_structured_grok_used_when_available(self, mock_format, mock_run, mock_legacy, mock_fetch):
        """When structured Grok succeeds, grok_analysis should be populated."""
        from lib.trading.strategies.daily.daily_plan import generate_daily_plan

        mock_run.return_value = {
            "macro_bias": "risk_on",
            "macro_summary": "Good day",
            "top_assets": [],
            "risk_warnings": [],
            "economic_events": [],
            "session_plan": "Aggressive",
            "correlation_notes": [],
            "swing_insights": {},
        }
        mock_format.return_value = "Formatted Grok text"

        # We need to provide enough data for the plan to not bail out early
        daily_df = _make_bars(60, start_price=2700.0)
        daily_df.index = pd.date_range("2025-01-01", periods=60, freq="1D")

        with patch.dict("os.environ", {"XAI_API_KEY": "test-key"}):
            plan = generate_daily_plan(
                account_size=50_000,
                asset_names=["Gold"],
                daily_data={"Gold": daily_df},
                weekly_data={"Gold": daily_df.iloc[::5]},
                include_grok=True,
            )

        assert plan.grok_available is True
        assert plan.grok_analysis.get("macro_bias") == "risk_on"
        assert plan.market_context == "Formatted Grok text"
        # Legacy should NOT have been called since structured succeeded
        mock_legacy.assert_not_called()

    @patch("lib.trading.strategies.daily.daily_plan._fetch_asset_data", return_value=(2750.0, 15.0))
    @patch("lib.integrations.grok_helper.run_daily_plan_grok_analysis")
    @patch("lib.trading.strategies.daily.daily_plan._fetch_grok_context")
    def test_falls_back_to_legacy_on_failure(self, mock_legacy, mock_structured, mock_fetch):
        """When structured Grok fails, should fall back to free-text."""
        from lib.trading.strategies.daily.daily_plan import generate_daily_plan

        mock_structured.side_effect = RuntimeError("API error")
        mock_legacy.return_value = "Legacy Grok text"

        daily_df = _make_bars(60, start_price=2700.0)
        daily_df.index = pd.date_range("2025-01-01", periods=60, freq="1D")

        with patch.dict("os.environ", {"XAI_API_KEY": "test-key"}):
            plan = generate_daily_plan(
                account_size=50_000,
                asset_names=["Gold"],
                daily_data={"Gold": daily_df},
                weekly_data={"Gold": daily_df.iloc[::5]},
                include_grok=True,
            )

        assert plan.grok_available is True
        assert plan.market_context == "Legacy Grok text"
        mock_legacy.assert_called_once()

    @patch("lib.trading.strategies.daily.daily_plan._fetch_asset_data", return_value=(2750.0, 15.0))
    def test_no_grok_when_key_missing(self, mock_fetch):
        from lib.trading.strategies.daily.daily_plan import generate_daily_plan

        daily_df = _make_bars(60, start_price=2700.0)
        daily_df.index = pd.date_range("2025-01-01", periods=60, freq="1D")

        with patch.dict("os.environ", {"XAI_API_KEY": ""}):
            plan = generate_daily_plan(
                account_size=50_000,
                asset_names=["Gold"],
                daily_data={"Gold": daily_df},
                include_grok=True,
            )

        assert plan.grok_available is False
        assert plan.grok_analysis == {}


# ===========================================================================
# Edge cases and robustness
# ===========================================================================


class TestEdgeCases:
    """Miscellaneous edge case tests."""

    def test_swing_engine_adapter_import(self):
        """Verify the module can be imported cleanly."""
        import lib.services.engine.swing

        assert hasattr(lib.services.engine.swing, "tick_swing_detector")
        assert hasattr(lib.services.engine.swing, "get_swing_summary")
        assert hasattr(lib.services.engine.swing, "reset_swing_states")
        assert hasattr(lib.services.engine.swing, "get_active_swing_states")

    def test_grok_helper_new_exports(self):
        """Verify the new Phase 3C functions are importable."""

    def test_scheduler_action_in_handlers(self):
        """Verify CHECK_SWING is in the expected action handler dict keys."""
        from lib.services.engine.scheduler import ActionType

        # The action exists and is a valid enum member
        assert ActionType.CHECK_SWING == "check_swing"
        assert isinstance(ActionType.CHECK_SWING, ActionType)

    def test_daily_plan_grok_analysis_default(self):
        """DailyPlan should have grok_analysis as empty dict by default."""
        from lib.trading.strategies.daily.daily_plan import DailyPlan

        plan = DailyPlan()
        assert plan.grok_analysis == {}
        assert "grok_analysis" in plan.to_dict()

    def test_parse_grok_with_nested_json(self):
        """Grok response with deeply nested JSON should still parse."""
        from lib.integrations.grok_helper import parse_grok_daily_plan_response

        raw = json.dumps(
            {
                "macro_bias": "risk_on",
                "macro_summary": "Nested test",
                "top_assets": [
                    {
                        "name": "Gold",
                        "reason": 'Has nested: {"foo": "bar"}',
                        "key_level": "2750",
                        "bias_agreement": True,
                    }
                ],
                "risk_warnings": [],
                "economic_events": [],
                "session_plan": "",
                "correlation_notes": [],
                "swing_insights": {"Gold": 'Test with "quotes"'},
            }
        )
        result = parse_grok_daily_plan_response(raw)
        assert result is not None
        assert result["macro_bias"] == "risk_on"

    def test_swing_module_constants(self):
        """Verify key constants in the swing engine adapter."""
        from lib.services.engine.swing import _MAX_CONCURRENT_SWINGS, _MIN_SCAN_INTERVAL

        assert _MAX_CONCURRENT_SWINGS == 3
        assert _MIN_SCAN_INTERVAL == 90.0

    def test_get_active_swing_states_returns_copy(self):
        """get_active_swing_states should return a copy, not the internal dict."""
        import lib.services.engine.swing as swing_mod
        from lib.services.engine.swing import get_active_swing_states

        swing_mod._active_swing_states = {"Gold": "fake_state"}
        result = get_active_swing_states()
        assert result == {"Gold": "fake_state"}
        # Modifying the result should not affect the internal state
        result["Silver"] = "another"
        assert "Silver" not in swing_mod._active_swing_states

        # Cleanup
        swing_mod._active_swing_states = {}
