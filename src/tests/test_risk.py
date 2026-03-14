"""
Tests for the Risk Rules Engine.

Covers:
  - RiskManager initialization and day-reset logic
  - can_enter_trade() — all 7 rule checks
  - register_open() / register_close() position tracking
  - sync_positions() from Rithmic trading connection
  - check_overnight_risk() time-based warning
  - get_status() comprehensive status dict
  - publish_to_redis() serialization
  - Daily P&L tracking and consecutive loss streaks
  - Edge cases: day rollover, stacking rules, circuit breaker
"""

import json
from datetime import datetime
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

from lib.services.engine.risk import (
    RiskManager,
    RiskState,
    TradeRecord,
)

_EST = ZoneInfo("America/New_York")


# ---------------------------------------------------------------------------
# Helper: create a RiskManager with a fixed clock
# ---------------------------------------------------------------------------


def _make_rm(
    hour: int = 8,
    minute: int = 0,
    account_size: int = 50_000,
    day: int = 15,
    month: int = 1,
    year: int = 2025,
    **kwargs,
) -> RiskManager:
    """Create a RiskManager with a deterministic clock for testing."""
    fixed_now = datetime(year, month, day, hour, minute, 0, tzinfo=_EST)
    return RiskManager(
        account_size=account_size,
        now_fn=lambda: fixed_now,
        **kwargs,
    )


def _make_rm_fn(
    account_size: int = 50_000,
    **kwargs,
):
    """Create a RiskManager with a mutable clock for multi-step tests."""
    current_time = [datetime(2025, 1, 15, 8, 0, 0, tzinfo=_EST)]

    def now_fn():
        return current_time[0]

    rm = RiskManager(account_size=account_size, now_fn=now_fn, **kwargs)
    return rm, current_time


# ===========================================================================
# Test: Initialization
# ===========================================================================


class TestRiskManagerInit:
    def test_default_params(self):
        rm = _make_rm()
        assert rm.account_size == 50_000
        assert rm.max_risk_per_trade == 50_000 * 0.0075  # $375
        assert rm.max_daily_loss == -1500.0  # updated default for 150k
        assert rm.max_open_trades == 2

    def test_custom_params(self):
        rm = _make_rm(
            account_size=150_000,
            risk_pct_per_trade=0.005,
            max_daily_loss=-1000.0,
            max_open_trades=3,
        )
        assert rm.account_size == 150_000
        assert rm.max_risk_per_trade == 150_000 * 0.005  # $750
        assert rm.max_daily_loss == -1000.0
        assert rm.max_open_trades == 3

    def test_initial_state_clean(self):
        rm = _make_rm()
        assert rm.open_trade_count == 0
        assert rm.daily_pnl == 0.0
        assert rm.consecutive_losses == 0
        assert rm.closed_trades == []

    def test_trading_date_set_on_init(self):
        rm = _make_rm(day=20, month=3, year=2025)
        status = rm.get_status()
        assert status["trading_date"] == "2025-03-20"


# ===========================================================================
# Test: can_enter_trade() — Rule checks
# ===========================================================================


class TestCanEnterTrade:
    def test_basic_trade_allowed(self):
        """A simple trade during active hours should be allowed."""
        rm = _make_rm(hour=8)  # 8 AM ET — well within active hours
        ok, reason = rm.can_enter_trade("MGC", "LONG", 1)
        assert ok is True
        assert reason == ""

    def test_basic_trade_with_risk(self):
        """Trade within risk budget should be allowed."""
        rm = _make_rm(hour=8)
        ok, reason = rm.can_enter_trade("MGC", "LONG", 2, risk_per_contract=100.0)
        assert ok is True  # 2 × $100 = $200 < $375 max

    # --- Rule 1: Daily loss limit ---

    def test_daily_loss_exceeded_blocks_trade(self):
        rm, clock = _make_rm_fn()
        # Simulate losses
        rm.register_open("MGC", "LONG", 1, 2700.0)
        rm.register_close("MGC", 2650.0, -1600.0)  # -$1600 > -$1500 limit
        ok, reason = rm.can_enter_trade("MNQ", "LONG", 1)
        assert ok is False
        assert "Daily loss limit" in reason

    def test_daily_loss_not_exceeded_allows_trade(self):
        rm, clock = _make_rm_fn()
        rm.register_open("MGC", "LONG", 1, 2700.0)
        rm.register_close("MGC", 2690.0, -100.0)  # -$100 < -$500 limit
        ok, reason = rm.can_enter_trade("MNQ", "LONG", 1)
        assert ok is True

    # --- Rule 2: Max open trades ---

    def test_max_open_trades_blocks(self):
        rm = _make_rm(hour=8)
        rm.register_open("MGC", "LONG", 1, 2700.0)
        rm.register_open("MNQ", "LONG", 1, 21000.0)
        ok, reason = rm.can_enter_trade("MES", "LONG", 1)
        assert ok is False
        assert "Max open trades" in reason

    def test_max_open_trades_allows_when_below(self):
        rm = _make_rm(hour=8)
        rm.register_open("MGC", "LONG", 1, 2700.0)
        ok, reason = rm.can_enter_trade("MNQ", "LONG", 1)
        assert ok is True  # 1/2 trades — room for one more

    def test_stack_bypasses_open_trade_limit(self):
        """Stacking adds to existing, so open trade count check is skipped."""
        rm = _make_rm(hour=8)
        rm.register_open("MGC", "LONG", 1, 2700.0)
        rm.register_open("MNQ", "LONG", 1, 21000.0)
        # Stack on MGC — is_stack=True means we don't count it as a new trade
        ok, reason = rm.can_enter_trade("MGC", "LONG", 1, is_stack=True, unrealized_r=1.0, wave_ratio=2.0)
        assert ok is True  # stacking bypasses open trade limit

    # --- Rule 3: Per-trade risk limit ---

    def test_per_trade_risk_exceeds_limit(self):
        rm = _make_rm(hour=8)
        ok, reason = rm.can_enter_trade("MGC", "LONG", 2, risk_per_contract=250.0)
        assert ok is False  # 2 × $250 = $500 > $375 max
        assert "Risk" in reason and "exceeds" in reason

    def test_per_trade_risk_within_limit(self):
        rm = _make_rm(hour=8)
        ok, reason = rm.can_enter_trade("MGC", "LONG", 1, risk_per_contract=200.0)
        assert ok is True  # 1 × $200 = $200 < $375 max

    def test_zero_risk_per_contract_allowed(self):
        """When risk_per_contract is 0 (unknown), skip the risk check."""
        rm = _make_rm(hour=8)
        ok, reason = rm.can_enter_trade("MGC", "LONG", 5, risk_per_contract=0)
        assert ok is True

    # --- Rule 4: No new entries after cutoff ---

    def test_after_entry_cutoff_blocks(self):
        rm = _make_rm(hour=10, minute=30)  # 10:30 AM — past 10:00 cutoff
        ok, reason = rm.can_enter_trade("MGC", "LONG", 1)
        assert ok is False
        assert "cutoff" in reason.lower()

    def test_before_entry_cutoff_allows(self):
        rm = _make_rm(hour=9, minute=59)  # 9:59 AM — before 10:00 cutoff
        ok, reason = rm.can_enter_trade("MGC", "LONG", 1)
        assert ok is True

    def test_exactly_at_cutoff_blocks(self):
        rm = _make_rm(hour=10, minute=0)  # exactly 10:00 AM
        ok, reason = rm.can_enter_trade("MGC", "LONG", 1)
        assert ok is False

    # --- Rule 5: Session ended ---

    def test_after_session_end_blocks(self):
        rm = _make_rm(hour=13)  # 1 PM — past 12:00 session end
        ok, reason = rm.can_enter_trade("MGC", "LONG", 1)
        assert ok is False
        assert "session" in reason.lower() and "ended" in reason.lower()

    def test_exactly_at_session_end_blocks(self):
        rm = _make_rm(hour=12, minute=0)
        ok, reason = rm.can_enter_trade("MGC", "LONG", 1)
        assert ok is False

    # --- Rule 6: Stacking rules ---

    def test_stack_requires_min_r(self):
        rm = _make_rm(hour=8)
        rm.register_open("MGC", "LONG", 1, 2700.0)
        ok, reason = rm.can_enter_trade("MGC", "LONG", 1, is_stack=True, unrealized_r=0.3, wave_ratio=2.0)
        assert ok is False
        assert "unrealized" in reason.lower()

    def test_stack_requires_min_wave(self):
        rm = _make_rm(hour=8)
        rm.register_open("MGC", "LONG", 1, 2700.0)
        ok, reason = rm.can_enter_trade("MGC", "LONG", 1, is_stack=True, unrealized_r=1.0, wave_ratio=1.2)
        assert ok is False
        assert "wave ratio" in reason.lower()

    def test_stack_both_conditions_met(self):
        rm = _make_rm(hour=8)
        rm.register_open("MGC", "LONG", 1, 2700.0)
        ok, reason = rm.can_enter_trade("MGC", "LONG", 1, is_stack=True, unrealized_r=1.0, wave_ratio=2.0)
        assert ok is True

    def test_stack_both_conditions_failed(self):
        rm = _make_rm(hour=8)
        rm.register_open("MGC", "LONG", 1, 2700.0)
        ok, reason = rm.can_enter_trade("MGC", "LONG", 1, is_stack=True, unrealized_r=0.1, wave_ratio=1.0)
        assert ok is False
        # Should have both reasons
        assert "unrealized" in reason.lower()
        assert "wave" in reason.lower()

    # --- Rule 7: Consecutive losses circuit breaker ---

    def test_consecutive_losses_circuit_breaker(self):
        rm, clock = _make_rm_fn()
        # Simulate 3 consecutive losses
        for i in range(3):
            rm.register_open(f"MGC{i}", "LONG", 1, 2700.0)
            rm.register_close(f"MGC{i}", 2690.0, -50.0)
        assert rm.consecutive_losses == 3
        ok, reason = rm.can_enter_trade("MNQ", "LONG", 1)
        assert ok is False
        assert "consecutive" in reason.lower()

    def test_win_resets_consecutive_losses(self):
        rm, clock = _make_rm_fn()
        rm.register_open("MGC0", "LONG", 1, 2700.0)
        rm.register_close("MGC0", 2690.0, -50.0)
        rm.register_open("MGC1", "LONG", 1, 2700.0)
        rm.register_close("MGC1", 2690.0, -50.0)
        assert rm.consecutive_losses == 2
        # Now a win
        rm.register_open("MGC2", "LONG", 1, 2700.0)
        rm.register_close("MGC2", 2720.0, 100.0)
        assert rm.consecutive_losses == 0
        ok, reason = rm.can_enter_trade("MNQ", "LONG", 1)
        assert ok is True

    # --- Multiple rules triggered ---

    def test_multiple_rules_combined(self):
        """After cutoff + max trades + daily loss should all show in reason."""
        rm = _make_rm(hour=10, minute=30)
        rm.register_open("MGC", "LONG", 1, 2700.0)
        rm.register_open("MNQ", "LONG", 1, 21000.0)
        rm._daily_pnl = -1600.0  # exceeds -$1500 limit
        ok, reason = rm.can_enter_trade("MES", "LONG", 1)
        assert ok is False
        assert "cutoff" in reason.lower()
        assert "Max open trades" in reason
        assert "Daily loss" in reason


# ===========================================================================
# Test: Position tracking
# ===========================================================================


class TestPositionTracking:
    def test_register_open_increases_count(self):
        rm = _make_rm()
        assert rm.open_trade_count == 0
        rm.register_open("MGC", "LONG", 2, 2700.0, risk_dollars=200.0)
        assert rm.open_trade_count == 1
        assert "MGC" in rm.open_positions

    def test_register_open_stores_details(self):
        rm = _make_rm()
        rm.register_open("MGC", "LONG", 3, 2705.50, risk_dollars=150.0)
        pos = rm.open_positions["MGC"]
        assert pos["side"] == "LONG"
        assert pos["quantity"] == 3
        assert pos["entry_price"] == 2705.50
        assert pos["risk_dollars"] == 150.0

    def test_register_close_removes_position(self):
        rm = _make_rm()
        rm.register_open("MGC", "LONG", 1, 2700.0)
        rm.register_close("MGC", 2720.0, 200.0)
        assert rm.open_trade_count == 0
        assert "MGC" not in rm.open_positions

    def test_register_close_updates_pnl(self):
        rm = _make_rm()
        rm.register_open("MGC", "LONG", 1, 2700.0)
        rm.register_close("MGC", 2720.0, 200.0)
        assert rm.daily_pnl == 200.0

    def test_register_close_loss_updates_streak(self):
        rm = _make_rm()
        rm.register_open("MGC", "LONG", 1, 2700.0)
        rm.register_close("MGC", 2680.0, -100.0)
        assert rm.consecutive_losses == 1
        assert rm.daily_pnl == -100.0

    def test_closed_trades_recorded(self):
        rm = _make_rm()
        rm.register_open("MGC", "LONG", 2, 2700.0)
        rm.register_close("MGC", 2720.0, 200.0)
        trades = rm.closed_trades
        assert len(trades) == 1
        assert trades[0].symbol == "MGC"
        assert trades[0].pnl == 200.0
        assert trades[0].is_win is True
        assert trades[0].closed is True

    def test_update_unrealized(self):
        rm = _make_rm()
        rm.register_open("MGC", "LONG", 1, 2700.0)
        rm.update_unrealized("MGC", 150.0)
        assert rm.open_positions["MGC"]["unrealized_pnl"] == 150.0

    def test_update_unrealized_unknown_symbol(self):
        """Updating unrealized for unknown symbol is a no-op."""
        rm = _make_rm()
        rm.update_unrealized("UNKNOWN", 100.0)  # should not raise

    def test_close_unknown_symbol(self):
        """Closing an unregistered symbol still updates P&L."""
        rm = _make_rm()
        rm.register_close("UNKNOWN", 100.0, -50.0)
        assert rm.daily_pnl == -50.0


# ===========================================================================
# Test: sync_positions() — Rithmic integration
# ===========================================================================


class TestSyncPositions:
    def test_sync_adds_new_positions(self):
        rm = _make_rm()
        rm.sync_positions(
            [
                {
                    "symbol": "MGC",
                    "side": "LONG",
                    "quantity": 2,
                    "avgPrice": 2700.0,
                    "unrealizedPnL": 50.0,
                },
                {
                    "symbol": "MNQ",
                    "side": "SHORT",
                    "quantity": 1,
                    "avgPrice": 21000.0,
                    "unrealizedPnL": -30.0,
                },
            ]
        )
        assert rm.open_trade_count == 2
        assert rm.open_positions["MGC"]["side"] == "LONG"
        assert rm.open_positions["MNQ"]["side"] == "SHORT"

    def test_sync_removes_closed_positions(self):
        rm = _make_rm()
        rm.register_open("MGC", "LONG", 1, 2700.0)
        rm.register_open("MNQ", "SHORT", 1, 21000.0)
        assert rm.open_trade_count == 2
        # Sync with only MGC remaining
        rm.sync_positions(
            [
                {
                    "symbol": "MGC",
                    "side": "LONG",
                    "quantity": 1,
                    "avgPrice": 2700.0,
                    "unrealizedPnL": 80.0,
                },
            ]
        )
        assert rm.open_trade_count == 1
        assert "MNQ" not in rm.open_positions

    def test_sync_updates_unrealized(self):
        rm = _make_rm()
        rm.register_open("MGC", "LONG", 1, 2700.0)
        rm.sync_positions(
            [
                {
                    "symbol": "MGC",
                    "side": "LONG",
                    "quantity": 1,
                    "avgPrice": 2700.0,
                    "unrealizedPnL": 200.0,
                },
            ]
        )
        assert rm.open_positions["MGC"]["unrealized_pnl"] == 200.0

    def test_sync_empty_clears_all(self):
        rm = _make_rm()
        rm.register_open("MGC", "LONG", 1, 2700.0)
        rm.sync_positions([])
        assert rm.open_trade_count == 0

    def test_sync_skips_empty_symbol(self):
        rm = _make_rm()
        rm.sync_positions(
            [
                {"symbol": "", "side": "LONG", "quantity": 1},
                {"symbol": "MGC", "side": "LONG", "quantity": 1, "avgPrice": 2700.0},
            ]
        )
        assert rm.open_trade_count == 1


# ===========================================================================
# Test: check_overnight_risk()
# ===========================================================================


class TestOvernightRisk:
    def test_no_positions_no_warning(self):
        rm = _make_rm(hour=11, minute=45)  # past overnight warning time
        has_risk, msg = rm.check_overnight_risk()
        assert has_risk is False
        assert msg == ""

    def test_positions_before_warning_time(self):
        rm = _make_rm(hour=10, minute=0)  # before 11:30 warning
        rm.register_open("MGC", "LONG", 1, 2700.0)
        has_risk, msg = rm.check_overnight_risk()
        assert has_risk is False

    def test_positions_at_warning_time(self):
        rm = _make_rm(hour=11, minute=30)  # exactly at warning time
        rm.register_open("MGC", "LONG", 1, 2700.0)
        has_risk, msg = rm.check_overnight_risk()
        assert has_risk is True
        assert "OVERNIGHT" in msg
        assert "MGC" in msg

    def test_positions_past_warning_time(self):
        rm = _make_rm(hour=11, minute=45)
        rm.register_open("MGC", "LONG", 1, 2700.0)
        rm.register_open("MNQ", "SHORT", 1, 21000.0)
        has_risk, msg = rm.check_overnight_risk()
        assert has_risk is True
        assert "2 open position" in msg
        assert "MGC" in msg
        assert "MNQ" in msg


# ===========================================================================
# Test: get_status()
# ===========================================================================


class TestGetStatus:
    def test_status_structure(self):
        rm = _make_rm(hour=8)
        status = rm.get_status()
        # Required keys
        assert "account_size" in status
        assert "max_risk_per_trade" in status
        assert "max_daily_loss" in status
        assert "max_open_trades" in status
        assert "open_trade_count" in status
        assert "open_positions" in status
        assert "daily_pnl" in status
        assert "can_trade" in status
        assert "block_reason" in status
        assert "risk_pct_of_account" in status
        assert "rules" in status
        assert "last_check" in status
        assert "trading_date" in status

    def test_status_can_trade_during_active(self):
        rm = _make_rm(hour=8)
        status = rm.get_status()
        assert status["can_trade"] is True
        assert status["block_reason"] == ""

    def test_status_blocked_after_cutoff(self):
        rm = _make_rm(hour=10, minute=30)
        status = rm.get_status()
        assert status["can_trade"] is False
        assert "cutoff" in status["block_reason"]

    def test_status_blocked_daily_loss(self):
        rm = _make_rm(hour=8)
        rm._daily_pnl = -1600.0  # exceeds -$1500 limit
        status = rm.get_status()
        assert status["can_trade"] is False
        assert "daily loss" in status["block_reason"]

    def test_status_blocked_max_trades(self):
        rm = _make_rm(hour=8)
        rm.register_open("MGC", "LONG", 1, 2700.0)
        rm.register_open("MNQ", "SHORT", 1, 21000.0)
        status = rm.get_status()
        assert status["can_trade"] is False
        assert "max trades" in status["block_reason"]

    def test_status_risk_percentage(self):
        rm = _make_rm(hour=8, account_size=100_000)
        rm.register_open("MGC", "LONG", 1, 2700.0, risk_dollars=2000.0)
        status = rm.get_status()
        assert status["risk_pct_of_account"] == 2.0  # $2000 / $100000 = 2%

    def test_status_open_positions_detail(self):
        rm = _make_rm(hour=8)
        rm.register_open("MGC", "LONG", 2, 2700.0, risk_dollars=200.0)
        status = rm.get_status()
        pos = status["open_positions"]
        assert "MGC" in pos
        assert pos["MGC"]["side"] == "LONG"
        assert pos["MGC"]["quantity"] == 2
        assert pos["MGC"]["risk_dollars"] == 200.0

    def test_status_overnight_warning(self):
        rm = _make_rm(hour=11, minute=45)
        rm.register_open("MGC", "LONG", 1, 2700.0)
        status = rm.get_status()
        assert status["is_overnight_warning"] is True

    def test_status_no_overnight_without_positions(self):
        rm = _make_rm(hour=11, minute=45)
        status = rm.get_status()
        assert status["is_overnight_warning"] is False

    def test_status_rules_section(self):
        rm = _make_rm(hour=8)
        rules = rm.get_status()["rules"]
        assert "risk_pct_per_trade" in rules
        assert "no_entry_after" in rules
        assert "session_end" in rules
        assert "stack_min_r" in rules
        assert "stack_min_wave" in rules

    def test_status_consecutive_losses_blocks(self):
        rm, clock = _make_rm_fn()
        for i in range(3):
            rm.register_open(f"X{i}", "LONG", 1, 100.0)
            rm.register_close(f"X{i}", 90.0, -10.0)
        status = rm.get_status()
        assert status["can_trade"] is False
        assert "consecutive" in status["block_reason"]


# ===========================================================================
# Test: Day reset logic
# ===========================================================================


class TestDayReset:
    def test_day_change_resets_counters(self):
        current = [datetime(2025, 1, 15, 8, 0, 0, tzinfo=_EST)]
        rm = RiskManager(now_fn=lambda: current[0])
        rm.register_open("MGC", "LONG", 1, 2700.0)
        rm.register_close("MGC", 2680.0, -100.0)
        assert rm.daily_pnl == -100.0
        assert rm.consecutive_losses == 1

        # Advance to next day
        current[0] = datetime(2025, 1, 16, 8, 0, 0, tzinfo=_EST)
        assert rm.daily_pnl == 0.0
        assert rm.consecutive_losses == 0
        assert len(rm.closed_trades) == 0

    def test_day_change_preserves_open_positions(self):
        current = [datetime(2025, 1, 15, 8, 0, 0, tzinfo=_EST)]
        rm = RiskManager(now_fn=lambda: current[0])
        rm.register_open("MGC", "LONG", 1, 2700.0)

        # Advance to next day
        current[0] = datetime(2025, 1, 16, 8, 0, 0, tzinfo=_EST)
        # Open positions survive day rollover (overnight positions)
        assert rm.open_trade_count == 1
        assert "MGC" in rm.open_positions


# ===========================================================================
# Test: publish_to_redis()
# ===========================================================================


class TestPublishToRedis:
    def test_publish_calls_cache_set(self):
        rm = _make_rm(hour=8)

        mock_cache_set = MagicMock()
        _mock_r = MagicMock()

        cache_mod = MagicMock()
        cache_mod.cache_set = mock_cache_set
        cache_mod.REDIS_AVAILABLE = False
        cache_mod._r = None

        with (
            patch.dict("sys.modules", {"lib.core.cache": cache_mod}),
            patch(
                "lib.services.engine.risk.RiskManager.publish_to_redis",
                wraps=rm.publish_to_redis,
            ),
        ):
            result = rm.publish_to_redis()

            assert result is True
            mock_cache_set.assert_called_once()
            call_args = mock_cache_set.call_args
            assert call_args[0][0] == "engine:risk_status"
            # Verify the payload is valid JSON
            payload = json.loads(call_args[0][1])
            assert "account_size" in payload
            assert "can_trade" in payload

    def test_publish_serializes_status(self):
        rm = _make_rm(hour=8)
        rm.register_open("MGC", "LONG", 2, 2700.0, risk_dollars=300.0)

        captured = {}

        def fake_cache_set(key, data, ttl=None):
            captured["key"] = key
            captured["data"] = json.loads(data)

        cache_mod = MagicMock()
        cache_mod.cache_set = fake_cache_set
        cache_mod.REDIS_AVAILABLE = False
        cache_mod._r = None

        with patch.dict("sys.modules", {"lib.core.cache": cache_mod}):
            rm.publish_to_redis()

        assert captured["key"] == "engine:risk_status"
        assert captured["data"]["open_trade_count"] == 1
        assert "MGC" in captured["data"]["open_positions"]


# ===========================================================================
# Test: Edge cases
# ===========================================================================


class TestEdgeCases:
    def test_pre_market_allows_trade(self):
        """Pre-market (before 5 AM) should allow trades — no cutoff."""
        rm = _make_rm(hour=3)  # 3 AM ET
        ok, reason = rm.can_enter_trade("MGC", "LONG", 1)
        assert ok is True

    def test_zero_account_size(self):
        """Edge: zero account size shouldn't crash."""
        rm = _make_rm(account_size=0, hour=8)
        status = rm.get_status()
        assert status["risk_pct_of_account"] == 0

    def test_multiple_opens_and_closes(self):
        rm = _make_rm(hour=8)
        rm.register_open("MGC", "LONG", 1, 2700.0)
        rm.register_open("MNQ", "SHORT", 2, 21000.0)
        rm.register_close("MGC", 2720.0, 200.0)
        assert rm.open_trade_count == 1
        assert rm.daily_pnl == 200.0
        rm.register_close("MNQ", 20950.0, 100.0)
        assert rm.open_trade_count == 0
        assert rm.daily_pnl == 300.0

    def test_daily_trade_count_tracks_opens(self):
        rm = _make_rm(hour=8)
        rm.register_open("MGC", "LONG", 1, 2700.0)
        rm.register_open("MNQ", "SHORT", 1, 21000.0)
        status = rm.get_status()
        assert status["daily_trade_count"] == 2

    def test_trade_record_dataclass(self):
        tr = TradeRecord(
            symbol="MGC",
            side="LONG",
            quantity=2,
            entry_price=2700.0,
            exit_price=2720.0,
            pnl=200.0,
            is_win=True,
            closed=True,
        )
        assert tr.symbol == "MGC"
        assert tr.pnl == 200.0

    def test_risk_state_dataclass(self):
        rs = RiskState()
        assert rs.can_trade is True
        assert rs.daily_pnl == 0.0
        assert rs.consecutive_losses == 0
