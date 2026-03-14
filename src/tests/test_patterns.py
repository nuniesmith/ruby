"""
Tests for the "Should Not Trade" Detector.

Covers:
  - evaluate_no_trade() — all 7 condition checks
  - Individual checker functions (_check_* helpers)
  - NoTradeResult / NoTradeCheck data structures
  - Severity escalation logic
  - publish_no_trade_alert() / clear_no_trade_alert() Redis publishing
  - Edge cases: empty data, missing risk status, time boundaries
  - Integration with focus data and risk status shapes
"""

import json
from datetime import datetime
from typing import Any
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

import pytest

from lib.services.engine.patterns import (
    DEFAULT_EXTREME_VOL,
    DEFAULT_LATE_SESSION_HOUR,
    DEFAULT_MAX_CONSECUTIVE_LOSSES,
    DEFAULT_MAX_DAILY_LOSS,
    DEFAULT_MIN_QUALITY,
    DEFAULT_SESSION_END_HOUR,
    NoTradeCheck,
    NoTradeCondition,
    NoTradeResult,
    _check_all_low_quality,
    _check_consecutive_losses,
    _check_daily_loss,
    _check_extreme_volatility,
    _check_late_session_no_setups,
    _check_no_market_data,
    _check_session_ended,
    _safe_float,
    clear_no_trade_alert,
    evaluate_no_trade,
    publish_no_trade_alert,
)

_EST = ZoneInfo("America/New_York")


# ---------------------------------------------------------------------------
# Helpers: build focus asset dicts and risk status dicts for tests
# ---------------------------------------------------------------------------


def _asset(
    symbol: str = "MGC",
    quality: float = 0.70,
    vol_percentile: float = 0.50,
    bias: str = "LONG",
    skip: bool = False,
    last_price: float = 2700.0,
    tp1: float = 2720.0,
    stop: float = 2680.0,
    **kwargs,
) -> dict[str, Any]:
    """Build a minimal focus asset dict for testing."""
    d = {
        "symbol": symbol,
        "quality": quality,
        "quality_pct": quality * 100,
        "vol_percentile": vol_percentile,
        "bias": bias,
        "skip": skip,
        "last_price": last_price,
        "tp1": tp1,
        "stop": stop,
    }
    d.update(kwargs)
    return d


def _risk_status(
    daily_pnl: float = 0.0,
    consecutive_losses: int = 0,
    daily_trade_count: int = 0,
    can_trade: bool = True,
    **kwargs,
) -> dict[str, Any]:
    """Build a minimal risk status dict for testing."""
    d = {
        "daily_pnl": daily_pnl,
        "consecutive_losses": consecutive_losses,
        "daily_trade_count": daily_trade_count,
        "can_trade": can_trade,
    }
    d.update(kwargs)
    return d


def _now(hour: int = 8, minute: int = 0, day: int = 15, month: int = 1, year: int = 2025) -> datetime:
    """Create a fixed Eastern Time datetime."""
    return datetime(year, month, day, hour, minute, 0, tzinfo=_EST)


# ===========================================================================
# Test: _safe_float helper
# ===========================================================================


class TestSafeFloat:
    def test_normal_float(self):
        assert _safe_float(3.14) == 3.14

    def test_int_to_float(self):
        assert _safe_float(42) == 42.0

    def test_string_float(self):
        assert _safe_float("1.5") == 1.5

    def test_none_returns_default(self):
        assert _safe_float(None) == 0.0

    def test_none_custom_default(self):
        assert _safe_float(None, default=99.9) == 99.9

    def test_nan_returns_default(self):
        assert _safe_float(float("nan")) == 0.0

    def test_inf_returns_default(self):
        assert _safe_float(float("inf")) == 0.0

    def test_neg_inf_returns_default(self):
        assert _safe_float(float("-inf")) == 0.0

    def test_non_numeric_string(self):
        assert _safe_float("abc") == 0.0

    def test_empty_string(self):
        assert _safe_float("") == 0.0

    def test_bool_true(self):
        assert _safe_float(True) == 1.0

    def test_bool_false(self):
        assert _safe_float(False) == 0.0


# ===========================================================================
# Test: NoTradeResult and NoTradeCheck data structures
# ===========================================================================


class TestDataStructures:
    def test_no_trade_check_defaults(self):
        check = NoTradeCheck(
            condition=NoTradeCondition.ALL_LOW_QUALITY,
            triggered=False,
            reason="",
        )
        assert check.severity == "warning"
        assert check.details == {}

    def test_no_trade_check_with_details(self):
        check = NoTradeCheck(
            condition=NoTradeCondition.EXTREME_VOLATILITY,
            triggered=True,
            reason="Too volatile",
            severity="critical",
            details={"max_vol": 0.92},
        )
        assert check.triggered is True
        assert check.details["max_vol"] == 0.92

    def test_no_trade_result_defaults(self):
        result = NoTradeResult()
        assert result.should_skip is False
        assert result.reasons == []
        assert result.checks == []
        assert result.primary_reason == ""
        assert result.severity == "info"

    def test_no_trade_result_to_dict(self):
        check = NoTradeCheck(
            condition=NoTradeCondition.ALL_LOW_QUALITY,
            triggered=True,
            reason="All assets low quality",
            severity="critical",
            details={"best_quality": 0.40},
        )
        result = NoTradeResult(
            should_skip=True,
            reasons=["All assets low quality"],
            checks=[check],
            checked_at="2025-01-15T08:00:00-03:00",
            primary_reason="All assets low quality",
            severity="critical",
        )
        d = result.to_dict()
        assert d["should_skip"] is True
        assert len(d["checks"]) == 1
        assert d["checks"][0]["condition"] == "all_low_quality"
        assert d["checks"][0]["triggered"] is True
        assert d["severity"] == "critical"
        assert d["primary_reason"] == "All assets low quality"

    def test_no_trade_result_to_dict_empty(self):
        result = NoTradeResult()
        d = result.to_dict()
        assert d["should_skip"] is False
        assert d["checks"] == []
        assert d["reasons"] == []

    def test_no_trade_condition_enum_values(self):
        assert NoTradeCondition.ALL_LOW_QUALITY.value == "all_low_quality"
        assert NoTradeCondition.EXTREME_VOLATILITY.value == "extreme_volatility"
        assert NoTradeCondition.DAILY_LOSS_EXCEEDED.value == "daily_loss_exceeded"
        assert NoTradeCondition.CONSECUTIVE_LOSSES.value == "consecutive_losses"
        assert NoTradeCondition.LATE_SESSION_NO_SETUPS.value == "late_session_no_setups"
        assert NoTradeCondition.NO_MARKET_DATA.value == "no_market_data"
        assert NoTradeCondition.SESSION_ENDED.value == "session_ended"


# ===========================================================================
# Test: _check_no_market_data
# ===========================================================================


class TestCheckNoMarketData:
    def test_empty_list_triggers(self):
        check = _check_no_market_data([])
        assert check.triggered is True
        assert check.condition == NoTradeCondition.NO_MARKET_DATA
        assert check.severity == "critical"
        assert "No market data" in check.reason

    def test_none_coerced_as_empty(self):
        # Passing an empty list (callers should convert None before)
        check = _check_no_market_data([])
        assert check.triggered is True

    def test_with_assets_does_not_trigger(self):
        check = _check_no_market_data([_asset()])
        assert check.triggered is False
        assert check.reason == ""
        assert check.details["asset_count"] == 1

    def test_multiple_assets(self):
        assets = [_asset("MGC"), _asset("MNQ"), _asset("MES")]
        check = _check_no_market_data(assets)
        assert check.triggered is False
        assert check.details["asset_count"] == 3


# ===========================================================================
# Test: _check_all_low_quality
# ===========================================================================


class TestCheckAllLowQuality:
    def test_all_below_threshold_triggers(self):
        assets = [
            _asset("MGC", quality=0.30),
            _asset("MNQ", quality=0.40),
            _asset("MES", quality=0.20),
        ]
        check = _check_all_low_quality(assets)
        assert check.triggered is True
        assert check.condition == NoTradeCondition.ALL_LOW_QUALITY
        assert check.severity == "critical"
        assert "40%" in check.reason  # best quality: MNQ at 40%
        assert "MNQ" in check.reason  # best asset

    def test_one_above_threshold_does_not_trigger(self):
        assets = [
            _asset("MGC", quality=0.30),
            _asset("MNQ", quality=0.60),  # above 0.55
            _asset("MES", quality=0.20),
        ]
        check = _check_all_low_quality(assets)
        assert check.triggered is False
        assert check.details["above_threshold"] == 1

    def test_all_above_threshold(self):
        assets = [
            _asset("MGC", quality=0.70),
            _asset("MNQ", quality=0.80),
        ]
        check = _check_all_low_quality(assets)
        assert check.triggered is False
        assert check.details["above_threshold"] == 2

    def test_exact_threshold_not_triggered(self):
        """Quality exactly at threshold (0.55) is NOT below threshold."""
        assets = [_asset("MGC", quality=0.55)]
        check = _check_all_low_quality(assets)
        assert check.triggered is False

    def test_just_below_threshold_triggers(self):
        assets = [_asset("MGC", quality=0.549)]
        check = _check_all_low_quality(assets)
        assert check.triggered is True

    def test_empty_list_does_not_trigger(self):
        check = _check_all_low_quality([])
        assert check.triggered is False

    def test_custom_threshold(self):
        assets = [_asset("MGC", quality=0.65)]
        check = _check_all_low_quality(assets, min_quality=0.70)
        assert check.triggered is True  # 0.65 < 0.70

    def test_details_include_qualities(self):
        assets = [_asset("MGC", quality=0.30), _asset("MNQ", quality=0.40)]
        check = _check_all_low_quality(assets)
        assert check.triggered is True
        assert "qualities" in check.details
        assert "MGC" in check.details["qualities"]
        assert "MNQ" in check.details["qualities"]

    def test_nan_quality_treated_as_zero(self):
        assets = [_asset("MGC", quality=float("nan"))]
        check = _check_all_low_quality(assets)
        assert check.triggered is True  # NaN → 0.0 < 0.55


# ===========================================================================
# Test: _check_extreme_volatility
# ===========================================================================


class TestCheckExtremeVolatility:
    def test_extreme_vol_triggers(self):
        assets = [_asset("MGC", vol_percentile=0.92)]
        check = _check_extreme_volatility(assets)
        assert check.triggered is True
        assert check.condition == NoTradeCondition.EXTREME_VOLATILITY
        assert check.severity == "critical"
        assert "MGC" in check.reason
        assert "92%" in check.reason

    def test_multiple_extreme_assets(self):
        assets = [
            _asset("MGC", vol_percentile=0.90),
            _asset("MNQ", vol_percentile=0.95),
            _asset("MES", vol_percentile=0.50),
        ]
        check = _check_extreme_volatility(assets)
        assert check.triggered is True
        assert "MGC" in check.reason
        assert "MNQ" in check.reason
        # MES should NOT be in the reason (it's below threshold)
        assert "MES" not in check.reason
        assert len(check.details["extreme_assets"]) == 2

    def test_no_extreme_vol(self):
        assets = [
            _asset("MGC", vol_percentile=0.50),
            _asset("MNQ", vol_percentile=0.70),
        ]
        check = _check_extreme_volatility(assets)
        assert check.triggered is False

    def test_exactly_at_threshold_does_not_trigger(self):
        """Vol percentile exactly at 0.88 is NOT > 0.88."""
        assets = [_asset("MGC", vol_percentile=0.88)]
        check = _check_extreme_volatility(assets)
        assert check.triggered is False

    def test_just_above_threshold_triggers(self):
        assets = [_asset("MGC", vol_percentile=0.881)]
        check = _check_extreme_volatility(assets)
        assert check.triggered is True

    def test_empty_list(self):
        check = _check_extreme_volatility([])
        assert check.triggered is False

    def test_custom_threshold(self):
        assets = [_asset("MGC", vol_percentile=0.75)]
        check = _check_extreme_volatility(assets, extreme_vol=0.70)
        assert check.triggered is True

    def test_details_include_all_percentiles(self):
        assets = [
            _asset("MGC", vol_percentile=0.92),
            _asset("MNQ", vol_percentile=0.50),
        ]
        check = _check_extreme_volatility(assets)
        assert "vol_percentiles" in check.details
        assert "MGC" in check.details["vol_percentiles"]
        assert "MNQ" in check.details["vol_percentiles"]


# ===========================================================================
# Test: _check_daily_loss
# ===========================================================================


class TestCheckDailyLoss:
    def test_loss_exceeds_threshold_triggers(self):
        rs = _risk_status(daily_pnl=-300.0)
        check = _check_daily_loss(rs)
        assert check.triggered is True
        assert check.condition == NoTradeCondition.DAILY_LOSS_EXCEEDED
        assert check.severity == "critical"
        assert "-300" in check.reason or "300" in check.reason

    def test_loss_at_threshold_triggers(self):
        """Exactly at -$250 should trigger (<=)."""
        rs = _risk_status(daily_pnl=-250.0)
        check = _check_daily_loss(rs)
        assert check.triggered is True

    def test_loss_below_threshold_does_not_trigger(self):
        rs = _risk_status(daily_pnl=-100.0)
        check = _check_daily_loss(rs)
        assert check.triggered is False

    def test_positive_pnl_does_not_trigger(self):
        rs = _risk_status(daily_pnl=500.0)
        check = _check_daily_loss(rs)
        assert check.triggered is False
        assert check.details["headroom"] > 0

    def test_zero_pnl_does_not_trigger(self):
        rs = _risk_status(daily_pnl=0.0)
        check = _check_daily_loss(rs)
        assert check.triggered is False

    def test_none_risk_status_does_not_trigger(self):
        check = _check_daily_loss(None)
        assert check.triggered is False
        assert "skipping" in check.details.get("note", "").lower()

    def test_custom_threshold(self):
        rs = _risk_status(daily_pnl=-400.0)
        check = _check_daily_loss(rs, max_daily_loss=-500.0)
        assert check.triggered is False  # -400 > -500 threshold
        check2 = _check_daily_loss(rs, max_daily_loss=-300.0)
        assert check2.triggered is True  # -400 <= -300

    def test_details_include_headroom(self):
        rs = _risk_status(daily_pnl=-100.0)
        check = _check_daily_loss(rs)
        assert "headroom" in check.details
        assert check.details["headroom"] == pytest.approx(150.0, abs=0.01)  # -100 - (-250) = 150

    def test_details_include_trade_count(self):
        rs = _risk_status(daily_pnl=-300.0, daily_trade_count=5)
        check = _check_daily_loss(rs)
        assert check.details["trade_count"] == 5


# ===========================================================================
# Test: _check_consecutive_losses
# ===========================================================================


class TestCheckConsecutiveLosses:
    def test_above_max_triggers(self):
        rs = _risk_status(consecutive_losses=3)
        check = _check_consecutive_losses(rs)
        assert check.triggered is True
        assert check.condition == NoTradeCondition.CONSECUTIVE_LOSSES
        assert check.severity == "warning"
        assert "3" in check.reason
        assert "consecutive" in check.reason.lower()

    def test_at_max_does_not_trigger(self):
        """Exactly at max (2) should NOT trigger (> not >=)."""
        rs = _risk_status(consecutive_losses=2)
        check = _check_consecutive_losses(rs)
        assert check.triggered is False

    def test_below_max_does_not_trigger(self):
        rs = _risk_status(consecutive_losses=1)
        check = _check_consecutive_losses(rs)
        assert check.triggered is False

    def test_zero_losses(self):
        rs = _risk_status(consecutive_losses=0)
        check = _check_consecutive_losses(rs)
        assert check.triggered is False

    def test_none_risk_status(self):
        check = _check_consecutive_losses(None)
        assert check.triggered is False

    def test_custom_max(self):
        rs = _risk_status(consecutive_losses=2)
        check = _check_consecutive_losses(rs, max_consecutive=1)
        assert check.triggered is True  # 2 > 1

    def test_high_streak(self):
        rs = _risk_status(consecutive_losses=10)
        check = _check_consecutive_losses(rs)
        assert check.triggered is True
        assert check.details["consecutive_losses"] == 10


# ===========================================================================
# Test: _check_late_session_no_setups
# ===========================================================================


class TestCheckLateSessionNoSetups:
    def test_after_10am_no_tradeable_triggers(self):
        now = _now(hour=10, minute=30)
        assets = [
            _asset("MGC", skip=True),
            _asset("MNQ", skip=True),
        ]
        check = _check_late_session_no_setups(assets, now=now)
        assert check.triggered is True
        assert check.condition == NoTradeCondition.LATE_SESSION_NO_SETUPS
        assert check.severity == "warning"
        assert "10" in check.reason

    def test_after_10am_with_tradeable_does_not_trigger(self):
        now = _now(hour=10, minute=30)
        assets = [
            _asset("MGC", skip=True),
            _asset("MNQ", skip=False),  # one tradeable
        ]
        check = _check_late_session_no_setups(assets, now=now)
        assert check.triggered is False
        assert check.details["tradeable_assets"] == 1

    def test_before_10am_does_not_trigger(self):
        now = _now(hour=9, minute=59)
        assets = [_asset("MGC", skip=True)]
        check = _check_late_session_no_setups(assets, now=now)
        assert check.triggered is False
        assert "Outside late-session" in check.details.get("note", "")

    def test_exactly_at_10am_triggers_if_no_setups(self):
        now = _now(hour=10, minute=0)
        assets = [_asset("MGC", skip=True)]
        check = _check_late_session_no_setups(assets, now=now)
        assert check.triggered is True

    def test_at_session_end_does_not_trigger(self):
        """At 12:00 (session end), this check yields to session_ended check."""
        now = _now(hour=12, minute=0)
        assets = [_asset("MGC", skip=True)]
        check = _check_late_session_no_setups(assets, now=now)
        assert check.triggered is False

    def test_after_session_end_does_not_trigger(self):
        now = _now(hour=14, minute=0)
        assets = [_asset("MGC", skip=True)]
        check = _check_late_session_no_setups(assets, now=now)
        assert check.triggered is False

    def test_early_morning_does_not_trigger(self):
        now = _now(hour=3, minute=0)
        assets = [_asset("MGC", skip=True)]
        check = _check_late_session_no_setups(assets, now=now)
        assert check.triggered is False

    def test_custom_late_hour(self):
        now = _now(hour=9, minute=0)
        assets = [_asset("MGC", skip=True)]
        check = _check_late_session_no_setups(assets, now=now, late_hour=9)
        assert check.triggered is True  # 9 >= 9 with no tradeable

    def test_empty_assets_in_late_session(self):
        now = _now(hour=10, minute=30)
        check = _check_late_session_no_setups([], now=now)
        assert check.triggered is True  # empty = no tradeable
        assert check.details["tradeable_assets"] == 0

    def test_details_include_tradeable_symbols(self):
        now = _now(hour=10, minute=30)
        assets = [_asset("MGC", skip=False), _asset("MNQ", skip=True)]
        check = _check_late_session_no_setups(assets, now=now)
        assert check.triggered is False
        assert "MGC" in check.details["tradeable_symbols"]
        assert "MNQ" not in check.details["tradeable_symbols"]


# ===========================================================================
# Test: _check_session_ended
# ===========================================================================


class TestCheckSessionEnded:
    def test_after_session_end_triggers(self):
        now = _now(hour=13, minute=0)
        check = _check_session_ended(now=now)
        assert check.triggered is True
        assert check.condition == NoTradeCondition.SESSION_ENDED
        assert check.severity == "info"

    def test_at_session_end_triggers(self):
        now = _now(hour=12, minute=0)
        check = _check_session_ended(now=now)
        assert check.triggered is True

    def test_before_session_end_does_not_trigger(self):
        now = _now(hour=11, minute=59)
        check = _check_session_ended(now=now)
        assert check.triggered is False

    def test_early_morning(self):
        now = _now(hour=3, minute=0)
        check = _check_session_ended(now=now)
        assert check.triggered is False

    def test_custom_session_end(self):
        now = _now(hour=14, minute=0)
        check = _check_session_ended(now=now, session_end_hour=15)
        assert check.triggered is False  # 14 < 15

    def test_late_evening(self):
        now = _now(hour=23, minute=0)
        check = _check_session_ended(now=now)
        assert check.triggered is True

    def test_details_include_hours(self):
        now = _now(hour=13)
        check = _check_session_ended(now=now)
        assert check.details["current_hour"] == 13
        assert check.details["session_end"] == 12


# ===========================================================================
# Test: evaluate_no_trade() — main evaluator
# ===========================================================================


class TestEvaluateNoTrade:
    def test_all_clear_returns_false(self):
        """Good data, good risk, during active session → no skip."""
        assets = [_asset("MGC", quality=0.70, vol_percentile=0.50)]
        rs = _risk_status(daily_pnl=0.0, consecutive_losses=0)
        now = _now(hour=8)
        result = evaluate_no_trade(assets, risk_status=rs, now=now)
        assert result.should_skip is False
        assert result.reasons == []
        assert result.severity == "info"

    def test_empty_assets_triggers(self):
        result = evaluate_no_trade([], now=_now(hour=8))
        assert result.should_skip is True
        assert any("No market data" in r for r in result.reasons)

    def test_all_low_quality_triggers(self):
        assets = [
            _asset("MGC", quality=0.30),
            _asset("MNQ", quality=0.40),
        ]
        result = evaluate_no_trade(assets, now=_now(hour=8))
        assert result.should_skip is True
        assert any("quality" in r.lower() for r in result.reasons)
        assert result.severity == "critical"

    def test_extreme_vol_triggers(self):
        assets = [_asset("MGC", quality=0.70, vol_percentile=0.92)]
        result = evaluate_no_trade(assets, now=_now(hour=8))
        assert result.should_skip is True
        assert any("volatility" in r.lower() for r in result.reasons)

    def test_daily_loss_triggers(self):
        assets = [_asset("MGC", quality=0.70)]
        rs = _risk_status(daily_pnl=-300.0)
        result = evaluate_no_trade(assets, risk_status=rs, now=_now(hour=8))
        assert result.should_skip is True
        assert any("loss" in r.lower() for r in result.reasons)

    def test_consecutive_losses_triggers(self):
        assets = [_asset("MGC", quality=0.70)]
        rs = _risk_status(consecutive_losses=3)
        result = evaluate_no_trade(assets, risk_status=rs, now=_now(hour=8))
        assert result.should_skip is True
        assert any("consecutive" in r.lower() for r in result.reasons)

    def test_late_session_no_setups_triggers(self):
        assets = [_asset("MGC", quality=0.30, skip=True)]
        result = evaluate_no_trade(assets, now=_now(hour=10, minute=30))
        assert result.should_skip is True
        # Should have both all_low_quality and late_session_no_setups
        condition_types = [c.condition for c in result.checks if c.triggered]
        assert NoTradeCondition.ALL_LOW_QUALITY in condition_types
        assert NoTradeCondition.LATE_SESSION_NO_SETUPS in condition_types

    def test_session_ended_triggers(self):
        assets = [_asset("MGC", quality=0.70)]
        result = evaluate_no_trade(assets, now=_now(hour=13))
        assert result.should_skip is True
        assert any("session" in r.lower() for r in result.reasons)

    def test_multiple_conditions_combined(self):
        """Multiple conditions can be triggered simultaneously."""
        assets = [_asset("MGC", quality=0.30, vol_percentile=0.92)]
        rs = _risk_status(daily_pnl=-300.0, consecutive_losses=4)
        result = evaluate_no_trade(assets, risk_status=rs, now=_now(hour=10, minute=30))
        assert result.should_skip is True
        assert len(result.reasons) >= 3  # low quality + extreme vol + daily loss + consecutive + late
        triggered = [c for c in result.checks if c.triggered]
        assert len(triggered) >= 3

    def test_primary_reason_is_first(self):
        """primary_reason should be the first triggered reason."""
        assets = []  # no market data triggers first
        result = evaluate_no_trade(assets, now=_now(hour=8))
        assert result.primary_reason == result.reasons[0]

    def test_severity_escalation_to_critical(self):
        """If any check is critical, overall severity should be critical."""
        assets = [_asset("MGC", quality=0.30)]  # triggers critical
        result = evaluate_no_trade(assets, now=_now(hour=8))
        assert result.severity == "critical"

    def test_severity_warning_only(self):
        """Only warning-level checks → overall severity is warning."""
        assets = [_asset("MGC", quality=0.70)]
        rs = _risk_status(consecutive_losses=3)  # warning severity
        result = evaluate_no_trade(assets, risk_status=rs, now=_now(hour=8))
        assert result.should_skip is True
        assert result.severity == "warning"

    def test_session_ended_info_severity(self):
        """Session ended alone is info severity."""
        assets = [_asset("MGC", quality=0.70, vol_percentile=0.50)]
        rs = _risk_status(daily_pnl=0.0, consecutive_losses=0)
        result = evaluate_no_trade(assets, risk_status=rs, now=_now(hour=13))
        # Only session_ended triggered (info severity)
        triggered = [c for c in result.checks if c.triggered]
        assert len(triggered) == 1
        assert triggered[0].condition == NoTradeCondition.SESSION_ENDED
        assert result.severity == "info"

    def test_checked_at_timestamp(self):
        now = _now(hour=8, minute=30)
        result = evaluate_no_trade([_asset()], now=now)
        assert "2025-01-15" in result.checked_at
        assert "08:30" in result.checked_at

    def test_all_checks_present_in_result(self):
        """All 7 checks should always be present, even if not triggered."""
        assets = [_asset("MGC", quality=0.70)]
        result = evaluate_no_trade(assets, now=_now(hour=8))
        assert len(result.checks) == 7
        conditions = {c.condition for c in result.checks}
        assert NoTradeCondition.NO_MARKET_DATA in conditions
        assert NoTradeCondition.ALL_LOW_QUALITY in conditions
        assert NoTradeCondition.EXTREME_VOLATILITY in conditions
        assert NoTradeCondition.DAILY_LOSS_EXCEEDED in conditions
        assert NoTradeCondition.CONSECUTIVE_LOSSES in conditions
        assert NoTradeCondition.LATE_SESSION_NO_SETUPS in conditions
        assert NoTradeCondition.SESSION_ENDED in conditions

    def test_no_risk_status_skips_loss_checks(self):
        """Without risk_status, daily loss and consecutive checks should not trigger."""
        assets = [_asset("MGC", quality=0.70)]
        result = evaluate_no_trade(assets, risk_status=None, now=_now(hour=8))
        assert result.should_skip is False
        loss_check = next(c for c in result.checks if c.condition == NoTradeCondition.DAILY_LOSS_EXCEEDED)
        assert loss_check.triggered is False
        streak_check = next(c for c in result.checks if c.condition == NoTradeCondition.CONSECUTIVE_LOSSES)
        assert streak_check.triggered is False

    def test_custom_thresholds(self):
        """All thresholds should be overridable."""
        assets = [_asset("MGC", quality=0.60, vol_percentile=0.75)]
        rs = _risk_status(daily_pnl=-100.0, consecutive_losses=1)
        result = evaluate_no_trade(
            assets,
            risk_status=rs,
            now=_now(hour=9),
            min_quality=0.70,  # 0.60 < 0.70 → triggers
            extreme_vol=0.70,  # 0.75 > 0.70 → triggers
            max_daily_loss=-50.0,  # -100 <= -50 → triggers
            max_consecutive_losses=0,  # 1 > 0 → triggers
        )
        assert result.should_skip is True
        triggered = [c for c in result.checks if c.triggered]
        assert len(triggered) >= 4


# ===========================================================================
# Test: evaluate_no_trade — realistic scenarios
# ===========================================================================


class TestEvaluateRealisticScenarios:
    def test_normal_morning_trading(self):
        """8 AM, good quality assets, no losses → trade OK."""
        assets = [
            _asset("MGC", quality=0.72, vol_percentile=0.45),
            _asset("MNQ", quality=0.65, vol_percentile=0.55),
            _asset("MES", quality=0.58, vol_percentile=0.40),
        ]
        rs = _risk_status(daily_pnl=0.0, consecutive_losses=0)
        result = evaluate_no_trade(assets, risk_status=rs, now=_now(hour=8))
        assert result.should_skip is False

    def test_bad_day_low_quality_everywhere(self):
        """Every asset is low quality → NO TRADE."""
        assets = [
            _asset("MGC", quality=0.35, vol_percentile=0.50),
            _asset("MNQ", quality=0.28, vol_percentile=0.60),
            _asset("MES", quality=0.42, vol_percentile=0.45),
        ]
        result = evaluate_no_trade(assets, now=_now(hour=7))
        assert result.should_skip is True
        assert result.severity == "critical"

    def test_vix_spike_day(self):
        """One asset has extreme volatility → NO TRADE."""
        assets = [
            _asset("MGC", quality=0.70, vol_percentile=0.60),
            _asset("MNQ", quality=0.65, vol_percentile=0.93),  # extreme
        ]
        result = evaluate_no_trade(assets, now=_now(hour=8))
        assert result.should_skip is True
        assert "MNQ" in result.primary_reason

    def test_losing_streak_circuit_breaker(self):
        """3 consecutive losses → take a break."""
        assets = [_asset("MGC", quality=0.70)]
        rs = _risk_status(daily_pnl=-180.0, consecutive_losses=3)
        result = evaluate_no_trade(assets, risk_status=rs, now=_now(hour=9))
        assert result.should_skip is True
        assert any("consecutive" in r.lower() for r in result.reasons)

    def test_daily_loss_max_hit(self):
        """Daily loss exceeds $250 → stop trading."""
        assets = [_asset("MGC", quality=0.70)]
        rs = _risk_status(daily_pnl=-275.0)
        result = evaluate_no_trade(assets, risk_status=rs, now=_now(hour=9))
        assert result.should_skip is True

    def test_late_session_still_has_setups(self):
        """10:30 AM but still has quality setups → trade OK."""
        assets = [
            _asset("MGC", quality=0.72, skip=False),
            _asset("MNQ", quality=0.40, skip=True),
        ]
        rs = _risk_status(daily_pnl=100.0, consecutive_losses=0)
        result = evaluate_no_trade(assets, risk_status=rs, now=_now(hour=10, minute=30))
        assert result.should_skip is False

    def test_late_session_no_setups_left(self):
        """10:30 AM with all setups invalidated → NO TRADE."""
        assets = [
            _asset("MGC", quality=0.40, skip=True),
            _asset("MNQ", quality=0.35, skip=True),
        ]
        result = evaluate_no_trade(assets, now=_now(hour=10, minute=30))
        assert result.should_skip is True

    def test_mixed_quality_keeps_trading(self):
        """Some assets below quality, but at least one good → trade OK."""
        assets = [
            _asset("MGC", quality=0.30, vol_percentile=0.50, skip=True),
            _asset("MNQ", quality=0.72, vol_percentile=0.40, skip=False),
        ]
        rs = _risk_status(daily_pnl=-50.0, consecutive_losses=0)
        result = evaluate_no_trade(assets, risk_status=rs, now=_now(hour=8))
        assert result.should_skip is False


# ===========================================================================
# Test: publish_no_trade_alert
# ===========================================================================


class TestPublishNoTradeAlert:
    def test_publish_serializes_result(self):
        check = NoTradeCheck(
            condition=NoTradeCondition.ALL_LOW_QUALITY,
            triggered=True,
            reason="All low quality",
            severity="critical",
        )
        result = NoTradeResult(
            should_skip=True,
            reasons=["All low quality"],
            checks=[check],
            checked_at="2025-01-15T08:00:00-03:00",
            primary_reason="All low quality",
            severity="critical",
        )

        captured = {}

        def fake_cache_set(key, data, ttl=None):
            captured["key"] = key
            captured["data"] = json.loads(data)

        cache_mod = MagicMock()
        cache_mod.cache_set = fake_cache_set
        cache_mod.REDIS_AVAILABLE = False
        cache_mod._r = None

        with patch.dict("sys.modules", {"lib.core.cache": cache_mod}):
            ok = publish_no_trade_alert(result)

        assert ok is True
        assert captured["key"] == "engine:no_trade"
        assert captured["data"]["should_skip"] is True
        assert len(captured["data"]["checks"]) == 1
        assert captured["data"]["severity"] == "critical"

    def test_publish_with_redis_pubsub(self):
        result = NoTradeResult(
            should_skip=True,
            reasons=["Test reason"],
            checks=[],
            checked_at="2025-01-15T08:00:00-03:00",
            primary_reason="Test reason",
            severity="warning",
        )

        mock_redis = MagicMock()
        cache_mod = MagicMock()
        cache_mod.cache_set = MagicMock()
        cache_mod.REDIS_AVAILABLE = True
        cache_mod._r = mock_redis

        with patch.dict("sys.modules", {"lib.core.cache": cache_mod}):
            ok = publish_no_trade_alert(result)

        assert ok is True
        mock_redis.publish.assert_called_once()
        call_args = mock_redis.publish.call_args
        assert call_args[0][0] == "dashboard:no_trade"

    def test_publish_empty_result(self):
        result = NoTradeResult()

        cache_mod = MagicMock()
        cache_mod.cache_set = MagicMock()
        cache_mod.REDIS_AVAILABLE = False
        cache_mod._r = None

        with patch.dict("sys.modules", {"lib.core.cache": cache_mod}):
            ok = publish_no_trade_alert(result)

        assert ok is True


# ===========================================================================
# Test: clear_no_trade_alert
# ===========================================================================


class TestClearNoTradeAlert:
    def test_clear_writes_cleared_payload(self):
        captured = {}

        def fake_cache_set(key, data, ttl=None):
            captured["key"] = key
            captured["data"] = json.loads(data)

        cache_mod = MagicMock()
        cache_mod.cache_set = fake_cache_set
        cache_mod.REDIS_AVAILABLE = False
        cache_mod._r = None

        with patch.dict("sys.modules", {"lib.core.cache": cache_mod}):
            ok = clear_no_trade_alert()

        assert ok is True
        assert captured["key"] == "engine:no_trade"
        assert captured["data"]["should_skip"] is False
        assert captured["data"]["reasons"] == []

    def test_clear_publishes_to_redis(self):
        mock_redis = MagicMock()
        cache_mod = MagicMock()
        cache_mod.cache_set = MagicMock()
        cache_mod.REDIS_AVAILABLE = True
        cache_mod._r = mock_redis

        with patch.dict("sys.modules", {"lib.core.cache": cache_mod}):
            ok = clear_no_trade_alert()

        assert ok is True
        mock_redis.publish.assert_called_once()
        call_args = mock_redis.publish.call_args
        assert call_args[0][0] == "dashboard:no_trade"
        payload = json.loads(call_args[0][1])
        assert payload["should_skip"] is False
        assert payload["cleared"] is True


# ===========================================================================
# Test: Serialization round-trip
# ===========================================================================


class TestSerialization:
    def test_result_to_dict_round_trip(self):
        """to_dict() output should be JSON-serializable and contain all fields."""
        checks = [
            NoTradeCheck(
                condition=NoTradeCondition.ALL_LOW_QUALITY,
                triggered=True,
                reason="Low quality",
                severity="critical",
                details={"best_quality": 0.3, "qualities": {"MGC": 0.3}},
            ),
            NoTradeCheck(
                condition=NoTradeCondition.EXTREME_VOLATILITY,
                triggered=False,
                reason="",
                details={"vol_percentiles": {"MGC": 0.5}},
            ),
        ]
        result = NoTradeResult(
            should_skip=True,
            reasons=["Low quality"],
            checks=checks,
            checked_at="2025-01-15T08:00:00-03:00",
            primary_reason="Low quality",
            severity="critical",
        )
        d = result.to_dict()

        # Should be JSON-serializable
        json_str = json.dumps(d, default=str)
        restored = json.loads(json_str)

        assert restored["should_skip"] is True
        assert len(restored["checks"]) == 2
        assert restored["checks"][0]["condition"] == "all_low_quality"
        assert restored["checks"][0]["triggered"] is True
        assert restored["checks"][1]["condition"] == "extreme_volatility"
        assert restored["checks"][1]["triggered"] is False

    def test_all_conditions_serializable(self):
        """All enum values should be serializable."""
        for cond in NoTradeCondition:
            check = NoTradeCheck(
                condition=cond,
                triggered=False,
                reason="",
            )
            d = {
                "condition": check.condition.value,
                "triggered": check.triggered,
            }
            json_str = json.dumps(d)
            assert cond.value in json_str


# ===========================================================================
# Test: Default thresholds
# ===========================================================================


class TestDefaultThresholds:
    def test_min_quality_is_55_pct(self):
        assert DEFAULT_MIN_QUALITY == 0.55

    def test_extreme_vol_is_88_pct(self):
        assert DEFAULT_EXTREME_VOL == 0.88

    def test_max_daily_loss_is_neg_250(self):
        assert DEFAULT_MAX_DAILY_LOSS == -250.0

    def test_max_consecutive_losses_is_2(self):
        assert DEFAULT_MAX_CONSECUTIVE_LOSSES == 2

    def test_late_session_hour_is_10(self):
        assert DEFAULT_LATE_SESSION_HOUR == 10

    def test_session_end_hour_is_12(self):
        assert DEFAULT_SESSION_END_HOUR == 12
