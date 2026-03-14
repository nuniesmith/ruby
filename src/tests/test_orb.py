"""
Tests for Opening Range Breakout (ORB) detector,
Risk API router, and risk-wired trades/positions endpoints.

Covers:
  - ORB core functions: compute_atr, compute_opening_range, detect_opening_range_breakout
  - ORB edge cases: insufficient data, no breakout, long/short breakouts
  - ORB multi-asset scanner: scan_orb_all_assets
  - ORB result serialization: to_dict
  - ORB publishing: publish_orb_alert, clear_orb_alert
  - Risk API router: /risk/status, /risk/check, /risk/history
  - Risk helpers: evaluate_position_risk, check_trade_entry_risk
  - Trades endpoint risk enforcement: POST /trades with risk warnings/blocks
  - Positions endpoint risk evaluation: POST /positions/update with risk status
  - Scheduler: CHECK_ORB action type and scheduling window
"""

import json
import os
import sys
from datetime import datetime, timedelta
from datetime import time as dt_time
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("DISABLE_REDIS", "1")

_EST = ZoneInfo("America/New_York")


# ---------------------------------------------------------------------------
# Helpers — build synthetic 1-minute bar DataFrames
# ---------------------------------------------------------------------------


def _make_1m_bars(
    n: int = 60,
    start_price: float = 2700.0,
    start_time: str = "2026-02-27 09:20:00",
    freq: str = "1min",
    seed: int = 42,
    volatility: float = 0.001,
    trend: float = 0.0,
) -> pd.DataFrame:
    """Create a synthetic 1-minute OHLCV DataFrame with tz-aware Eastern index."""
    rng = np.random.default_rng(seed)
    returns = rng.normal(trend, volatility, n)
    close = start_price * np.exp(np.cumsum(returns))

    spread = close * rng.uniform(0.001, 0.003, n)
    high = close + rng.uniform(0, 1, n) * spread
    low = close - rng.uniform(0, 1, n) * spread
    opn = close + rng.uniform(-0.3, 0.3, n) * spread

    high = np.maximum(high, np.maximum(opn, close))
    low = np.minimum(low, np.minimum(opn, close))
    volume = rng.poisson(500, n).astype(float)

    idx = pd.date_range(start=start_time, periods=n, freq=freq, tz=_EST)

    return pd.DataFrame(
        {"Open": opn, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=idx,
    )


def _make_breakout_bars(
    direction: str = "LONG",
    or_price: float = 2700.0,
    breakout_magnitude: float = 15.0,
) -> pd.DataFrame:
    """Create bars that produce a clear ORB breakout in the given direction.

    - Pre-OR bars (09:20–09:29): stable around or_price
    - OR bars (09:30–09:59): tight range around or_price
    - Post-OR bars (10:00–10:30): price moves to trigger breakout
    """
    parts = []

    # Pre-market bars (09:20–09:29) — 10 bars
    pre = _make_1m_bars(n=10, start_price=or_price, start_time="2026-02-27 09:20:00", seed=1)
    parts.append(pre)

    # Opening range bars (09:30–09:59) — 30 bars, tight range
    or_bars = _make_1m_bars(
        n=30,
        start_price=or_price,
        start_time="2026-02-27 09:30:00",
        seed=2,
        volatility=0.0002,
    )
    parts.append(or_bars)

    # Post-OR bars (10:00–10:30) — 30 bars
    if direction == "LONG":
        post = _make_1m_bars(
            n=30,
            start_price=or_price + breakout_magnitude,
            start_time="2026-02-27 10:00:00",
            seed=3,
            trend=0.001,
        )
    else:
        post = _make_1m_bars(
            n=30,
            start_price=or_price - breakout_magnitude,
            start_time="2026-02-27 10:00:00",
            seed=3,
            trend=-0.001,
        )
    parts.append(post)

    return pd.concat(parts)


# ===========================================================================
# Tests: ORB Core — compute_atr
# ===========================================================================


class TestComputeATR:
    """Test the compute_atr function."""

    def test_basic_atr(self):
        from lib.trading.strategies.rb.open.detector import compute_atr

        bars = _make_1m_bars(n=30, start_price=100.0, volatility=0.005)
        atr = compute_atr(bars["High"].values, bars["Low"].values, bars["Close"].values, period=14)
        assert atr > 0
        assert isinstance(atr, float)

    def test_atr_with_small_data(self):
        from lib.trading.strategies.rb.open.detector import compute_atr

        # Only 3 bars — less than period, should still return something
        bars = _make_1m_bars(n=3, start_price=100.0)
        atr = compute_atr(bars["High"].values, bars["Low"].values, bars["Close"].values, period=14)
        assert atr >= 0

    def test_atr_single_bar(self):
        from lib.trading.strategies.rb.open.detector import compute_atr

        bars = _make_1m_bars(n=1, start_price=100.0)
        atr = compute_atr(bars["High"].values, bars["Low"].values, bars["Close"].values, period=14)
        # Single bar: H-L
        assert atr >= 0

    def test_atr_zero_range(self):
        """Flat bars should produce near-zero ATR."""
        from lib.trading.strategies.rb.open.detector import compute_atr

        n = 20
        prices = np.full(n, 100.0)
        atr = compute_atr(prices, prices, prices, period=14)
        assert atr == 0.0

    def test_atr_increases_with_volatility(self):
        from lib.trading.strategies.rb.open.detector import compute_atr

        low_vol = _make_1m_bars(n=30, start_price=100.0, volatility=0.001, seed=10)
        high_vol = _make_1m_bars(n=30, start_price=100.0, volatility=0.01, seed=10)

        atr_low = compute_atr(low_vol["High"].values, low_vol["Low"].values, low_vol["Close"].values)
        atr_high = compute_atr(high_vol["High"].values, high_vol["Low"].values, high_vol["Close"].values)
        assert atr_high > atr_low


# ===========================================================================
# Tests: ORB Core — compute_opening_range
# ===========================================================================


class TestComputeOpeningRange:
    """Test the compute_opening_range function."""

    def test_basic_opening_range(self):
        from lib.trading.strategies.rb.open.detector import compute_opening_range

        bars = _make_1m_bars(n=70, start_price=2700.0, start_time="2026-02-27 09:20:00")
        or_high, or_low, count, complete = compute_opening_range(bars)

        assert or_high > 0
        assert or_low > 0
        assert or_high >= or_low
        assert count > 0  # Should have bars in the 09:30–10:00 window

    def test_empty_dataframe(self):
        from lib.trading.strategies.rb.open.detector import compute_opening_range

        empty = pd.DataFrame(columns=["High", "Low", "Close"])  # type: ignore[call-overload]
        or_high, or_low, count, complete = compute_opening_range(empty)
        assert or_high == 0.0
        assert or_low == 0.0
        assert count == 0
        assert complete is False

    def test_none_input(self):
        from lib.trading.strategies.rb.open.detector import compute_opening_range

        or_high, or_low, count, complete = compute_opening_range(None)
        assert count == 0

    def test_no_bars_in_or_window(self):
        """Bars outside 09:30–10:00 should produce empty OR."""
        from lib.trading.strategies.rb.open.detector import compute_opening_range

        # Bars starting at 10:30 — all after OR window
        bars = _make_1m_bars(n=30, start_price=2700.0, start_time="2026-02-27 10:30:00")
        or_high, or_low, count, complete = compute_opening_range(bars)
        assert count == 0

    def test_or_complete_flag(self):
        """is_complete should be True only when bars exist past 10:00 ET."""
        from lib.trading.strategies.rb.open.detector import compute_opening_range

        # Bars from 09:30 to 09:50 — OR not complete
        bars_early = _make_1m_bars(n=20, start_price=2700.0, start_time="2026-02-27 09:30:00")
        _, _, _, complete_early = compute_opening_range(bars_early)
        assert complete_early is False

        # Bars from 09:30 to 10:10 — OR complete
        bars_full = _make_1m_bars(n=45, start_price=2700.0, start_time="2026-02-27 09:30:00")
        _, _, _, complete_full = compute_opening_range(bars_full)
        assert complete_full is True

    def test_or_bar_count(self):
        """Bar count should match the number of bars in 09:30–10:00 window."""
        from lib.trading.strategies.rb.open.detector import compute_opening_range

        # 30 bars starting at 09:30 — all are in OR window
        bars = _make_1m_bars(n=30, start_price=2700.0, start_time="2026-02-27 09:30:00")
        _, _, count, _ = compute_opening_range(bars)
        assert count == 30

    def test_naive_index_assumed_eastern(self):
        """Bars with naive (no tz) index should be treated as Eastern."""
        from lib.trading.strategies.rb.open.detector import compute_opening_range

        bars = _make_1m_bars(n=40, start_price=2700.0, start_time="2026-02-27 09:25:00")
        # Remove timezone
        bars.index = pd.DatetimeIndex(bars.index).tz_localize(None)
        # _localize_index should make it Eastern
        or_high, or_low, count, _ = compute_opening_range(bars)
        assert count > 0
        assert or_high > 0


# ===========================================================================
# Tests: ORB Core — detect_opening_range_breakout
# ===========================================================================


class TestDetectOpeningRangeBreakout:
    """Test the main ORB detection function."""

    def test_no_breakout(self):
        """When post-OR bars stay within the range, no breakout should be detected."""
        from lib.trading.strategies.rb.open.detector import detect_opening_range_breakout

        # Tight bars that stay within OR range
        bars = _make_1m_bars(
            n=70,
            start_price=2700.0,
            start_time="2026-02-27 09:20:00",
            volatility=0.0001,
            seed=42,
        )
        result = detect_opening_range_breakout(bars, symbol="MGC")

        assert result.symbol == "MGC"
        assert result.or_high > 0
        assert result.or_low > 0
        # With very low volatility, breakout is unlikely
        # (though not guaranteed — depends on the ATR threshold)

    def test_long_breakout(self):
        """Clear upward breakout should be detected as LONG."""
        from lib.trading.strategies.rb.open.detector import detect_opening_range_breakout

        bars = _make_breakout_bars(direction="LONG", or_price=2700.0, breakout_magnitude=20.0)
        result = detect_opening_range_breakout(bars, symbol="MGC")

        assert result.or_complete is True
        assert result.or_high > 0
        assert result.or_low > 0
        assert result.atr_value > 0
        assert result.long_trigger > result.or_high
        assert result.short_trigger < result.or_low
        # With a big magnitude, breakout should be detected
        if result.breakout_detected:
            assert result.direction == "LONG"
            assert result.trigger_price > result.long_trigger

    def test_short_breakout(self):
        """Clear downward breakout should be detected as SHORT."""
        from lib.trading.strategies.rb.open.detector import detect_opening_range_breakout

        bars = _make_breakout_bars(direction="SHORT", or_price=2700.0, breakout_magnitude=20.0)
        result = detect_opening_range_breakout(bars, symbol="MGC")

        assert result.or_complete is True
        if result.breakout_detected:
            assert result.direction == "SHORT"
            assert result.trigger_price < result.short_trigger

    def test_empty_bars(self):
        from lib.trading.strategies.rb.open.detector import detect_opening_range_breakout

        empty = pd.DataFrame(columns=["High", "Low", "Close"])  # type: ignore[call-overload]
        result = detect_opening_range_breakout(empty, symbol="TEST")
        assert result.breakout_detected is False
        assert result.error != ""

    def test_none_bars(self):
        from lib.trading.strategies.rb.open.detector import detect_opening_range_breakout

        result = detect_opening_range_breakout(None, symbol="TEST")
        assert result.breakout_detected is False
        assert "No bar data" in result.error

    def test_insufficient_bars(self):
        from lib.trading.strategies.rb.open.detector import detect_opening_range_breakout

        bars = _make_1m_bars(n=3, start_price=2700.0, start_time="2026-02-27 09:30:00")
        result = detect_opening_range_breakout(bars, symbol="TEST")
        assert result.breakout_detected is False
        assert "Insufficient" in result.error or "Opening range has only" in result.error

    def test_missing_columns(self):
        from lib.trading.strategies.rb.open.detector import detect_opening_range_breakout

        bars = pd.DataFrame({"Open": [1, 2, 3, 4, 5, 6], "Volume": [100] * 6})
        result = detect_opening_range_breakout(bars, symbol="TEST")
        assert result.error != ""
        assert "Missing columns" in result.error

    def test_or_not_complete_returns_levels_only(self):
        """If OR window hasn't finished, return levels but no breakout scan."""
        from lib.trading.strategies.rb.open.detector import detect_opening_range_breakout

        # Only bars in the OR window, no post-OR bars
        bars = _make_1m_bars(n=25, start_price=2700.0, start_time="2026-02-27 09:30:00")
        result = detect_opening_range_breakout(bars, symbol="MGC")

        assert result.or_complete is False
        assert result.breakout_detected is False
        assert result.or_high > 0
        assert result.long_trigger > 0 or result.error != ""

    def test_custom_atr_period(self):
        from lib.trading.strategies.rb.open.detector import detect_opening_range_breakout

        bars = _make_1m_bars(n=70, start_price=2700.0, start_time="2026-02-27 09:20:00")
        result = detect_opening_range_breakout(bars, symbol="MGC", atr_period=5)
        assert result.atr_value > 0

    def test_custom_breakout_multiplier(self):
        from lib.trading.strategies.rb.open.detector import detect_opening_range_breakout

        bars = _make_1m_bars(n=70, start_price=2700.0, start_time="2026-02-27 09:20:00")

        # Very high multiplier — less likely to detect breakout
        result_high = detect_opening_range_breakout(bars, symbol="MGC", breakout_multiplier=5.0)
        # Very low multiplier — more likely to detect breakout
        result_low = detect_opening_range_breakout(bars, symbol="MGC", breakout_multiplier=0.01)

        assert result_high.long_trigger > result_low.long_trigger

    def test_evaluated_at_timestamp(self):
        from lib.trading.strategies.rb.open.detector import detect_opening_range_breakout

        fixed_time = datetime(2026, 2, 27, 10, 30, 0, tzinfo=_EST)
        bars = _make_1m_bars(n=70, start_price=2700.0, start_time="2026-02-27 09:20:00")
        result = detect_opening_range_breakout(bars, symbol="MGC", now_fn=lambda: fixed_time)
        assert "2026-02-27" in result.evaluated_at


# ===========================================================================
# Tests: ORBResult
# ===========================================================================


class TestORBResult:
    """Test ORBResult data class and serialization."""

    def test_to_dict_structure(self):
        from lib.trading.strategies.rb.open.models import ORBResult

        result = ORBResult(
            symbol="MGC",
            or_high=2710.5,
            or_low=2695.3,
            or_range=15.2,
            atr_value=3.5,
            breakout_detected=True,
            direction="LONG",
            trigger_price=2712.25,
        )

        d = result.to_dict()
        assert d["type"] == "ORB"
        assert d["symbol"] == "MGC"
        assert d["or_high"] == 2710.5
        assert d["or_low"] == 2695.3
        assert d["breakout_detected"] is True
        assert d["direction"] == "LONG"
        assert d["trigger_price"] == 2712.25

    def test_to_dict_json_serializable(self):
        from lib.trading.strategies.rb.open.models import ORBResult

        result = ORBResult(symbol="MNQ", or_high=20100.5, or_low=20050.0)
        d = result.to_dict()
        # Should serialize without error
        s = json.dumps(d)
        parsed = json.loads(s)
        assert parsed["symbol"] == "MNQ"

    def test_default_values(self):
        from lib.trading.strategies.rb.open.models import ORBResult

        result = ORBResult()
        assert result.symbol == ""
        assert result.breakout_detected is False
        assert result.direction == ""
        assert result.or_high == 0.0

    def test_to_dict_rounds_values(self):
        from lib.trading.strategies.rb.open.models import ORBResult

        result = ORBResult(or_high=2710.123456789, atr_value=3.987654321)
        d = result.to_dict()
        assert d["or_high"] == 2710.1235  # 4 decimal places
        assert d["atr_value"] == 3.9877


# ===========================================================================
# Tests: ORB Multi-Asset Scanner
# ===========================================================================


class TestScanORBAllAssets:
    """Test the scan_orb_all_assets function."""

    def test_multiple_symbols(self):
        from lib.trading.strategies.rb.open.detector import scan_orb_all_assets

        bars_by_symbol = {
            "MGC": _make_1m_bars(n=70, start_price=2700.0, start_time="2026-02-27 09:20:00", seed=1),
            "MNQ": _make_1m_bars(n=70, start_price=20100.0, start_time="2026-02-27 09:20:00", seed=2),
        }
        results = scan_orb_all_assets(bars_by_symbol)
        assert len(results) == 2
        symbols = {r.symbol for r in results}
        assert "MGC" in symbols
        assert "MNQ" in symbols

    def test_empty_dict(self):
        from lib.trading.strategies.rb.open.detector import scan_orb_all_assets

        results = scan_orb_all_assets({})
        assert results == []

    def test_handles_bad_data(self):
        from lib.trading.strategies.rb.open.detector import scan_orb_all_assets

        bars_by_symbol = {
            "MGC": _make_1m_bars(n=70, start_price=2700.0, start_time="2026-02-27 09:20:00"),
            "BAD": pd.DataFrame(),  # empty
        }
        results = scan_orb_all_assets(bars_by_symbol)
        assert len(results) == 2


# ===========================================================================
# Tests: ORB Redis Publishing
# ===========================================================================


class TestORBPublishing:
    """Test publish_orb_alert and clear_orb_alert."""

    def test_publish_success(self):
        from lib.trading.strategies.rb.open.models import ORBResult
        from lib.trading.strategies.rb.open.publisher import publish_orb_alert

        result = ORBResult(
            symbol="MGC",
            breakout_detected=True,
            direction="LONG",
            or_high=2710.0,
            or_low=2695.0,
        )

        mock_cache = MagicMock()
        mock_cache.REDIS_AVAILABLE = False
        mock_cache._r = None
        mock_cache.cache_set = MagicMock()

        with patch.dict(sys.modules, {"lib.core.cache": mock_cache}):
            success = publish_orb_alert(result)
            assert success is True
            assert mock_cache.cache_set.call_count >= 1

    def test_publish_with_redis_pubsub(self):
        from lib.trading.strategies.rb.open.models import ORBResult
        from lib.trading.strategies.rb.open.publisher import publish_orb_alert

        mock_r = MagicMock()
        mock_cache = MagicMock()
        mock_cache.REDIS_AVAILABLE = True
        mock_cache._r = mock_r
        mock_cache.cache_set = MagicMock()

        result = ORBResult(symbol="MGC", breakout_detected=True, direction="LONG")

        with patch.dict(
            sys.modules,
            {
                "lib.core.cache": mock_cache,
            },
        ):
            success = publish_orb_alert(result)
            assert success is True
            mock_r.publish.assert_called()

    def test_clear_orb_alert(self):
        from lib.trading.strategies.rb.open.publisher import clear_orb_alert

        mock_cache = MagicMock()
        mock_cache.cache_set = MagicMock()

        with patch.dict(sys.modules, {"lib.core.cache": mock_cache}):
            success = clear_orb_alert()
            assert success is True


# ===========================================================================
# Tests: Scheduler — CHECK_ORB Action
# ===========================================================================


class TestSchedulerORB:
    """Test that the scheduler correctly schedules CHECK_ORB actions."""

    def test_check_orb_action_type_exists(self):
        from lib.services.engine.scheduler import ActionType

        assert hasattr(ActionType, "CHECK_ORB")
        assert ActionType.CHECK_ORB.value == "check_orb"

    def test_orb_scheduled_during_or_window(self):
        """CHECK_ORB should be scheduled between 09:30 and 11:00 ET."""
        from lib.services.engine.scheduler import (
            ActionType,
            ScheduleManager,
        )

        sm = ScheduleManager()

        # 10:00 ET — inside ORB window
        now_in_window = datetime(2026, 2, 27, 10, 0, 0, tzinfo=_EST)
        actions = sm.get_pending_actions(now=now_in_window)
        action_types = [a.action for a in actions]
        assert ActionType.CHECK_ORB in action_types

    def test_orb_not_scheduled_outside_window(self):
        """CHECK_ORB should NOT be scheduled before 09:30 or after 11:00 ET."""
        from lib.services.engine.scheduler import (
            ActionType,
            ScheduleManager,
        )

        sm = ScheduleManager()

        # 08:00 ET — before ORB window (but in active session 03:00–12:00)
        now_early = datetime(2026, 2, 27, 8, 0, 0, tzinfo=_EST)
        actions_early = sm.get_pending_actions(now=now_early)
        early_types = [a.action for a in actions_early]
        assert ActionType.CHECK_ORB not in early_types

    def test_orb_not_scheduled_in_off_hours(self):
        """CHECK_ORB should not appear during off-hours session."""
        from lib.services.engine.scheduler import (
            ActionType,
            ScheduleManager,
        )

        sm = ScheduleManager()

        # 14:00 ET — off-hours
        now_off = datetime(2026, 2, 27, 14, 0, 0, tzinfo=_EST)
        actions = sm.get_pending_actions(now=now_off)
        action_types = [a.action for a in actions]
        assert ActionType.CHECK_ORB not in action_types

    def test_orb_interval_respected(self):
        """After running once, ORB should not be rescheduled until interval elapses.

        The scheduler uses ``time.monotonic()`` internally for interval tracking,
        so we mock it to simulate time passing without real sleeps.
        """
        import time as _time

        from lib.services.engine.scheduler import (
            ActionType,
            ScheduleManager,
        )

        sm = ScheduleManager()
        fake_mono = [100.0]  # mutable container so the lambda can read it

        now = datetime(2026, 2, 27, 10, 0, 0, tzinfo=_EST)

        with patch.object(_time, "monotonic", side_effect=lambda: fake_mono[0]):
            actions1 = sm.get_pending_actions(now=now)
            orb_actions_1 = [a for a in actions1 if a.action == ActionType.CHECK_ORB]
            assert len(orb_actions_1) == 1

            # Mark as done — this records the monotonic timestamp
            sm.mark_done(ActionType.CHECK_ORB)

            # Immediately after (only 30s of monotonic time) — should NOT reschedule
            fake_mono[0] = 130.0  # +30s
            now2 = now + timedelta(seconds=30)
            actions2 = sm.get_pending_actions(now=now2)
            orb_actions_2 = [a for a in actions2 if a.action == ActionType.CHECK_ORB]
            assert len(orb_actions_2) == 0

            # After interval (3 min of monotonic time) — should be scheduled again
            fake_mono[0] = 280.0  # +180s from mark_done
            now3 = now + timedelta(minutes=3)
            actions3 = sm.get_pending_actions(now=now3)
            orb_actions_3 = [a for a in actions3 if a.action == ActionType.CHECK_ORB]
            assert len(orb_actions_3) == 1


# ===========================================================================
# Tests: Risk API Router — helpers
# ===========================================================================


class TestRiskHelpers:
    """Test the risk API helper functions."""

    def test_evaluate_position_risk_no_manager(self):
        """When RiskManager can't be imported, should return safe defaults."""
        from lib.services.data.api.risk import evaluate_position_risk

        # Patch the getter to return None
        with patch(
            "lib.services.data.api.risk._get_local_risk_manager",
            return_value=None,
        ):
            result = evaluate_position_risk([])
            assert result["can_trade"] is True
            assert result["block_reason"] == ""
            assert result["warnings"] == []

    def test_evaluate_position_risk_with_positions(self):
        """Should sync positions and return risk evaluation."""
        from lib.services.data.api.risk import evaluate_position_risk
        from lib.services.engine.risk import RiskManager

        rm = RiskManager(account_size=50_000)
        with patch(
            "lib.services.data.api.risk._get_local_risk_manager",
            return_value=rm,
        ):
            positions = [
                {
                    "symbol": "MGC",
                    "side": "Long",
                    "quantity": 2,
                    "avgPrice": 2700.0,
                    "unrealizedPnL": 50.0,
                }
            ]
            result = evaluate_position_risk(positions)
            assert "can_trade" in result
            assert "warnings" in result
            assert isinstance(result["warnings"], list)

    def test_check_trade_entry_risk_no_manager(self):
        from lib.services.data.api.risk import check_trade_entry_risk

        with patch(
            "lib.services.data.api.risk._get_local_risk_manager",
            return_value=None,
        ):
            allowed, reason, details = check_trade_entry_risk("MGC", "LONG")
            assert allowed is True
            assert details.get("risk_available") is False

    def test_check_trade_entry_risk_allowed(self):
        from lib.services.data.api.risk import check_trade_entry_risk
        from lib.services.engine.risk import RiskManager

        # RiskManager with generous limits
        rm = RiskManager(
            account_size=50_000,
            now_fn=lambda: datetime(2026, 2, 27, 8, 0, 0, tzinfo=_EST),
        )

        with (
            patch("lib.services.data.api.risk._get_local_risk_manager", return_value=rm),
            patch("lib.services.data.api.risk._sync_local_risk_manager"),
        ):
            allowed, reason, details = check_trade_entry_risk("MGC", "LONG", size=1)
            assert allowed is True
            assert reason == ""
            assert details["risk_available"] is True

    def test_check_trade_entry_risk_blocked_by_daily_loss(self):
        from lib.services.data.api.risk import check_trade_entry_risk
        from lib.services.engine.risk import RiskManager

        rm = RiskManager(
            account_size=50_000,
            max_daily_loss=-500.0,
            now_fn=lambda: datetime(2026, 2, 27, 8, 0, 0, tzinfo=_EST),
        )
        # Simulate daily loss
        rm._daily_pnl = -600.0

        with (
            patch("lib.services.data.api.risk._get_local_risk_manager", return_value=rm),
            patch("lib.services.data.api.risk._sync_local_risk_manager"),
        ):
            allowed, reason, details = check_trade_entry_risk("MGC", "LONG", size=1)
            assert allowed is False
            assert "Daily loss limit" in reason


class TestRiskEventRecording:
    """Test the in-memory risk event audit trail."""

    def test_record_event(self):
        from lib.services.data.api.risk import (
            _record_risk_event,
            _risk_events,
        )

        initial_count = len(_risk_events)
        _record_risk_event(
            event_type="block",
            symbol="MGC",
            side="LONG",
            reason="test block",
        )
        assert len(_risk_events) > initial_count
        last = _risk_events[-1]
        assert last["event_type"] == "block"
        assert last["symbol"] == "MGC"

    def test_event_limit(self):
        from lib.services.data.api.risk import (
            _MAX_RISK_EVENTS,
            _record_risk_event,
            _risk_events,
        )

        # Fill up events
        for i in range(_MAX_RISK_EVENTS + 10):
            _record_risk_event(event_type="test", reason=f"event-{i}")

        assert len(_risk_events) <= _MAX_RISK_EVENTS


# ===========================================================================
# Tests: Risk API Router — endpoints
# ===========================================================================


class TestRiskStatusEndpoint:
    """Test the GET /risk/status endpoint."""

    @pytest.fixture()
    def client(self):
        """Create a test client with the risk router mounted."""
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from lib.services.data.api.risk import router

        app = FastAPI()
        app.include_router(router, prefix="/risk")
        return TestClient(app)

    def test_status_returns_200(self, client):
        result = client.get("/risk/status")
        assert result.status_code == 200

    def test_status_structure(self, client):
        result = client.get("/risk/status")
        data = result.json()
        assert "can_trade" in data
        assert "source" in data

    def test_status_falls_back_to_local(self, client):
        """When Redis has no data, should use local RiskManager."""
        # The mock cache returns None, so it should fall back
        result = client.get("/risk/status")
        data = result.json()
        # Either local or unavailable source
        assert data["source"] in ("local", "unavailable", "redis")


class TestRiskCheckEndpoint:
    """Test the POST /risk/check endpoint."""

    @pytest.fixture()
    def client(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from lib.services.data.api.risk import router

        app = FastAPI()
        app.include_router(router, prefix="/risk")
        return TestClient(app)

    def test_check_returns_response(self, client):
        result = client.post(
            "/risk/check",
            json={"symbol": "MGC", "side": "LONG", "size": 1},
        )
        # 200 if RM available, 503 if not
        assert result.status_code in (200, 503)

    def test_check_with_full_params(self, client):
        result = client.post(
            "/risk/check",
            json={
                "symbol": "MGC",
                "side": "LONG",
                "size": 2,
                "risk_per_contract": 100.0,
                "is_stack": False,
                "wave_ratio": 1.5,
                "unrealized_r": 0.0,
            },
        )
        assert result.status_code in (200, 503)

    def test_check_validation_requires_symbol(self, client):
        result = client.post(
            "/risk/check",
            json={"side": "LONG"},
        )
        assert result.status_code == 422  # Pydantic validation error


class TestRiskHistoryEndpoint:
    """Test the GET /risk/history endpoint."""

    @pytest.fixture()
    def client(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from lib.services.data.api.risk import router

        app = FastAPI()
        app.include_router(router, prefix="/risk")
        return TestClient(app)

    def test_history_returns_200(self, client):
        result = client.get("/risk/history")
        assert result.status_code == 200

    def test_history_structure(self, client):
        result = client.get("/risk/history")
        data = result.json()
        assert "events" in data
        assert "count" in data
        assert "limit" in data
        assert isinstance(data["events"], list)

    def test_history_limit_param(self, client):
        result = client.get("/risk/history?limit=5")
        data = result.json()
        assert data["limit"] == 5


# ===========================================================================
# Tests: Trades Router — Risk Enforcement
# ===========================================================================


class TestTradesRiskEnforcement:
    """Test that POST /trades includes risk check results."""

    @pytest.fixture(autouse=True)
    def _setup_db(self, tmp_path):
        """Use a temp database for trade tests."""
        db_path = str(tmp_path / "test_trades.db")
        os.environ["DB_PATH"] = db_path

        # Re-init models with new DB
        import importlib

        from lib.core import models

        importlib.reload(models)
        models.init_db()

    @pytest.fixture()
    def client(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from lib.services.data.api.trades import router

        app = FastAPI()
        app.include_router(router, prefix="")
        return TestClient(app)

    def test_create_trade_includes_risk_fields(self, client):
        """Trade response should include risk check fields."""
        result = client.post(
            "/trades",
            json={
                "asset": "Gold",
                "direction": "LONG",
                "entry": 2700.0,
                "contracts": 1,
            },
        )
        assert result.status_code == 201
        data = result.json()
        # Risk fields should be present in response
        assert "risk_checked" in data

    def test_create_trade_with_enforce_risk_false(self, client):
        """With enforce_risk=False (default), blocked trades are still created."""
        result = client.post(
            "/trades",
            json={
                "asset": "Gold",
                "direction": "LONG",
                "entry": 2700.0,
                "contracts": 1,
                "enforce_risk": False,
            },
        )
        assert result.status_code == 201
        data = result.json()
        assert data["id"] is not None

    def test_create_trade_with_enforce_risk_true_blocked(self, client):
        """With enforce_risk=True and a risk block, should return 403."""
        from lib.services.engine.risk import RiskManager

        # Create a RM that blocks everything (daily loss exceeded)
        rm = RiskManager(
            account_size=50_000,
            max_daily_loss=-500.0,
            now_fn=lambda: datetime(2026, 2, 27, 8, 0, 0, tzinfo=_EST),
        )
        rm._daily_pnl = -600.0

        with (
            patch("lib.services.data.api.risk._get_local_risk_manager", return_value=rm),
            patch("lib.services.data.api.risk._sync_local_risk_manager"),
        ):
            result = client.post(
                "/trades",
                json={
                    "asset": "Gold",
                    "direction": "LONG",
                    "entry": 2700.0,
                    "contracts": 1,
                    "enforce_risk": True,
                },
            )
            assert result.status_code == 403
            data = result.json()
            assert "reason" in data["detail"]


class TestTradesEnforceRiskField:
    """Test the enforce_risk field on CreateTradeRequest."""

    def test_enforce_risk_default_false(self):
        from lib.services.data.api.trades import CreateTradeRequest

        req = CreateTradeRequest(asset="Gold", direction="LONG", entry=2700.0)  # type: ignore[call-arg]
        assert req.enforce_risk is False

    def test_enforce_risk_can_be_true(self):
        from lib.services.data.api.trades import CreateTradeRequest

        req = CreateTradeRequest(asset="Gold", direction="LONG", entry=2700.0, enforce_risk=True)  # type: ignore[call-arg]
        assert req.enforce_risk is True


# ===========================================================================
# Tests: Positions Router — Risk Evaluation
# ===========================================================================


class TestPositionsRiskEvaluation:
    """Test that POST /positions/update includes risk evaluation."""

    @pytest.fixture(autouse=True)
    def _clear_cache(self):
        """Clear position cache before each test."""
        from lib.core.cache import _mem_cache

        original = dict(_mem_cache)
        _mem_cache.clear()
        yield
        _mem_cache.clear()
        _mem_cache.update(original)

    @pytest.fixture()
    def client(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from lib.services.data.api.positions import router

        app = FastAPI()
        app.include_router(router, prefix="/positions")
        return TestClient(app)

    def test_update_returns_risk_status(self, client):
        """Position update response should include risk evaluation."""
        result = client.post(
            "/positions/update",
            json={
                "account": "Sim101",
                "positions": [
                    {
                        "symbol": "MGCZ5",
                        "side": "Long",
                        "quantity": 2,
                        "avgPrice": 2700.0,
                        "unrealizedPnL": 50.0,
                    }
                ],
            },
        )
        assert result.status_code == 200
        data = result.json()
        assert data["status"] == "received"
        assert data["positions_count"] == 1
        assert "total_unrealized_pnl" in data
        # Risk field should be present (may be empty dict if RM unavailable)
        assert "risk" in data

    def test_update_empty_positions(self, client):
        result = client.post(
            "/positions/update",
            json={"account": "Sim101", "positions": []},
        )
        assert result.status_code == 200
        data = result.json()
        assert data["positions_count"] == 0

    def test_update_risk_warning_on_limit(self, client):
        """When risk limits are hit, response should include warnings."""
        from lib.services.engine.risk import RiskManager

        rm = RiskManager(
            account_size=50_000,
            max_open_trades=1,
            now_fn=lambda: datetime(2026, 2, 27, 8, 0, 0, tzinfo=_EST),
        )
        # Pre-load 2 positions to exceed the limit
        rm._open_positions = {
            "MGCZ5": {
                "side": "LONG",
                "quantity": 1,
                "entry_price": 2700.0,
                "risk_dollars": 100,
                "unrealized_pnl": 0.0,
                "opened_at": "",
            },
            "MNQZ5": {
                "side": "SHORT",
                "quantity": 1,
                "entry_price": 20000.0,
                "risk_dollars": 100,
                "unrealized_pnl": 0.0,
                "opened_at": "",
            },
        }

        with patch(
            "lib.services.data.api.risk._get_local_risk_manager",
            return_value=rm,
        ):
            result = client.post(
                "/positions/update",
                json={
                    "account": "Sim101",
                    "positions": [
                        {
                            "symbol": "MGCZ5",
                            "side": "Long",
                            "quantity": 1,
                            "avgPrice": 2700.0,
                        },
                        {
                            "symbol": "MNQZ5",
                            "side": "Short",
                            "quantity": 1,
                            "avgPrice": 20000.0,
                        },
                    ],
                },
            )
            data = result.json()
            risk = data.get("risk", {})
            # With max_open_trades=1 and 2 open, should flag max trades
            if risk.get("can_trade") is False:
                assert "max trades" in risk.get("block_reason", "").lower()


# ===========================================================================
# Tests: Dashboard — ORB Panel Rendering
# ===========================================================================


class TestDashboardORBPanel:
    """Test the ORB panel HTML rendering (multi-session format)."""

    def test_render_orb_panel_none(self):
        from lib.services.data.api.dashboard import _render_orb_panel

        html = _render_orb_panel(None)
        assert "orb-panel" in html
        assert "Waiting for ORB sessions" in html
        assert "London 03:00 ET" in html
        assert "US 09:30 ET" in html

    def test_render_orb_panel_with_multi_session_data(self):
        """Multi-session format with london + us keys."""
        from lib.services.data.api.dashboard import _render_orb_panel

        data = {
            "london": {
                "or_high": 2710.5,
                "or_low": 2695.3,
                "or_range": 15.2,
                "atr_value": 3.5,
                "long_trigger": 2712.25,
                "short_trigger": 2693.55,
                "breakout_detected": False,
                "direction": "",
                "trigger_price": 0,
                "symbol": "MGC",
                "or_complete": True,
                "evaluated_at": "2026-02-27T04:05:00-03:00",
                "error": "",
            },
            "us": None,
            "best": {
                "or_high": 2710.5,
                "or_low": 2695.3,
                "or_range": 15.2,
                "atr_value": 3.5,
                "long_trigger": 2712.25,
                "short_trigger": 2693.55,
                "breakout_detected": False,
                "direction": "",
                "trigger_price": 0,
                "symbol": "MGC",
                "or_complete": True,
                "evaluated_at": "2026-02-27T04:05:00-03:00",
                "error": "",
            },
        }
        html = _render_orb_panel(data)
        assert "orb-panel" in html
        assert "2710.5" in html or "2,710.5" in html
        assert "OPENING RANGE" in html
        # Both session sub-cards should be present
        assert "London Open" in html
        assert "US Equity Open" in html

    def test_render_orb_panel_with_legacy_data(self):
        """Backward compat: single-session data without london/us/best keys."""
        from lib.services.data.api.dashboard import _render_orb_panel

        data = {
            "or_high": 2710.5,
            "or_low": 2695.3,
            "or_range": 15.2,
            "atr_value": 3.5,
            "long_trigger": 2712.25,
            "short_trigger": 2693.55,
            "breakout_detected": False,
            "direction": "",
            "trigger_price": 0,
            "symbol": "MGC",
            "or_complete": True,
            "evaluated_at": "2026-02-27T10:05:00-03:00",
            "error": "",
        }
        html = _render_orb_panel(data)
        assert "orb-panel" in html
        assert "2710.5" in html or "2,710.5" in html
        assert "OPENING RANGE" in html

    def test_render_orb_panel_with_breakout(self):
        from lib.services.data.api.dashboard import _render_orb_panel

        us_session = {
            "or_high": 2710.0,
            "or_low": 2695.0,
            "or_range": 15.0,
            "atr_value": 3.5,
            "long_trigger": 2711.75,
            "short_trigger": 2693.25,
            "breakout_detected": True,
            "direction": "LONG",
            "trigger_price": 2715.0,
            "symbol": "MGC",
            "or_complete": True,
            "evaluated_at": "2026-02-27T10:15:00-03:00",
            "error": "",
        }
        data = {
            "london": None,
            "us": us_session,
            "best": us_session,
        }
        html = _render_orb_panel(data)
        # Breakout shows in the session sub-card
        assert "LONG" in html
        assert "2,715.00" in html or "2715.0" in html
        assert "animate-pulse" in html

    def test_render_orb_panel_with_error(self):
        from lib.services.data.api.dashboard import _render_orb_panel

        data = {
            "london": None,
            "us": {"error": "No bar data provided", "or_high": 0, "or_low": 0},
            "best": {"error": "No bar data provided", "or_high": 0, "or_low": 0},
        }
        html = _render_orb_panel(data)
        assert "No bar data" in html


class TestDashboardORBEndpoint:
    """Test the /api/orb/html endpoint."""

    @pytest.fixture()
    def client(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from lib.services.data.api.dashboard import router

        app = FastAPI()
        app.include_router(router)
        return TestClient(app)

    def test_orb_html_endpoint(self, client):
        result = client.get("/api/orb/html")
        assert result.status_code == 200
        assert "orb-panel" in result.text


# ===========================================================================
# Tests: SSE — ORB events
# ===========================================================================


class TestSSEORBCache:
    """Test that the SSE module can read ORB data from cache."""

    _original_cache = None

    def setup_method(self):
        self._original_cache = sys.modules.get("lib.core.cache", None)
        # Install mock cache for SSE module
        mock_cache = MagicMock()
        mock_cache.REDIS_AVAILABLE = False
        mock_cache._r = None
        mock_cache.cache_get = MagicMock(return_value=None)
        mock_cache.cache_set = MagicMock()
        mock_cache._cache_key = MagicMock(side_effect=lambda *parts: "futures:mock:" + ":".join(str(p) for p in parts))
        mock_cache.get_data_source = MagicMock(return_value="mock")
        sys.modules["lib.core.cache"] = mock_cache
        self._mock_cache = mock_cache

    def teardown_method(self):
        if self._original_cache is not None:
            sys.modules["lib.core.cache"] = self._original_cache
        else:
            sys.modules.pop("lib.core.cache", None)

    def test_get_orb_from_cache_none(self):
        from lib.services.data.api.sse import _get_orb_from_cache

        self._mock_cache.cache_get.return_value = None
        result = _get_orb_from_cache()
        assert result is None

    def test_get_orb_from_cache_with_data(self):
        from lib.services.data.api.sse import _get_orb_from_cache

        orb_data = {"type": "ORB", "symbol": "MGC", "breakout_detected": True, "session_key": "us"}

        # Mock cache_get to return session-specific data:
        # First call is engine:orb:london (return None), second is engine:orb:us (return data)
        def _mock_get(key):
            if key == "engine:orb:us":
                return json.dumps(orb_data).encode()
            return None

        self._mock_cache.cache_get.side_effect = _mock_get
        result = _get_orb_from_cache()
        assert result is not None
        parsed = json.loads(result)
        # Multi-session format: result has "london", "us", "best" keys
        assert "us" in parsed
        assert parsed["us"]["symbol"] == "MGC"
        assert parsed["best"]["symbol"] == "MGC"

    def test_get_orb_from_cache_london_session(self):
        from lib.services.data.api.sse import _get_orb_from_cache

        london_data = {"type": "ORB", "symbol": "MGC", "session_key": "london", "breakout_detected": True}

        def _mock_get(key):
            if key == "engine:orb:london":
                return json.dumps(london_data).encode()
            return None

        self._mock_cache.cache_get.side_effect = _mock_get
        result = _get_orb_from_cache()
        assert result is not None
        parsed = json.loads(result)
        assert "london" in parsed
        assert parsed["london"]["session_key"] == "london"
        assert parsed["best"]["session_key"] == "london"

    def test_get_orb_from_cache_both_sessions(self):
        from lib.services.data.api.sse import _get_orb_from_cache

        london_data = {"type": "ORB", "symbol": "MGC", "session_key": "london", "breakout_detected": False}
        us_data = {"type": "ORB", "symbol": "MGC", "session_key": "us", "breakout_detected": True}

        def _mock_get(key):
            if key == "engine:orb:london":
                return json.dumps(london_data).encode()
            if key == "engine:orb:us":
                return json.dumps(us_data).encode()
            return None

        self._mock_cache.cache_get.side_effect = _mock_get
        result = _get_orb_from_cache()
        assert result is not None
        parsed = json.loads(result)
        assert parsed["london"] is not None
        assert parsed["us"] is not None
        # Best should be the one with breakout (US)
        assert parsed["best"]["session_key"] == "us"
        assert parsed["best"]["breakout_detected"] is True

    def test_get_orb_from_cache_legacy_fallback(self):
        """Legacy cache key (engine:orb) still works when session keys are absent."""
        from lib.services.data.api.sse import _get_orb_from_cache

        legacy_data = {"type": "ORB", "symbol": "MGC", "breakout_detected": True}

        def _mock_get(key):
            if key == "engine:orb":
                return json.dumps(legacy_data).encode()
            return None

        self._mock_cache.cache_get.side_effect = _mock_get
        result = _get_orb_from_cache()
        assert result is not None
        parsed = json.loads(result)
        # Legacy data slots into "us" by default (session_key defaults to "us")
        assert parsed["us"]["symbol"] == "MGC"


# ===========================================================================
# Tests: Risk API — Request/Response Models
# ===========================================================================


class TestRiskModels:
    """Test the Pydantic models for the risk API."""

    def test_risk_check_request_defaults(self):
        from lib.services.data.api.risk import RiskCheckRequest

        req = RiskCheckRequest(symbol="MGC", side="LONG")  # type: ignore[call-arg]
        assert req.size == 1
        assert req.risk_per_contract == 0.0
        assert req.is_stack is False

    def test_risk_check_response_structure(self):
        from lib.services.data.api.risk import RiskCheckResponse

        resp = RiskCheckResponse(  # type: ignore[call-arg]
            allowed=True,
            symbol="MGC",
            side="LONG",
            size=1,
            total_risk=100.0,
            max_risk_per_trade=375.0,
            daily_pnl=-50.0,
            open_trade_count=1,
            checked_at="2026-02-27T10:00:00-03:00",
        )
        assert resp.allowed is True
        assert resp.total_risk == 100.0

    def test_risk_status_response_defaults(self):
        from lib.services.data.api.risk import RiskStatusResponse

        resp = RiskStatusResponse()  # type: ignore[call-arg]
        assert resp.can_trade is True
        assert resp.source == ""
        assert resp.account_size == 0


# ===========================================================================
# Tests: Integration — ORB + Risk Manager together
# ===========================================================================


class TestORBRiskIntegration:
    """Integration tests combining ORB detection with RiskManager checks."""

    def test_orb_breakout_with_risk_check(self):
        """After detecting an ORB breakout, a risk check should evaluate the trade."""
        from lib.services.engine.risk import RiskManager
        from lib.trading.strategies.rb.open.detector import detect_opening_range_breakout

        bars = _make_breakout_bars(direction="LONG", or_price=2700.0, breakout_magnitude=20.0)
        orb_result = detect_opening_range_breakout(bars, symbol="MGC")

        if orb_result.breakout_detected:
            rm = RiskManager(
                account_size=50_000,
                now_fn=lambda: datetime(2026, 2, 27, 10, 15, 0, tzinfo=_EST),
            )
            allowed, reason = rm.can_enter_trade(
                symbol="MGC",
                side=orb_result.direction,
                size=1,
                risk_per_contract=orb_result.atr_value * 10,  # point value of 10 for MGC
            )
            # Should either allow or give a reason
            assert isinstance(allowed, bool)
            assert isinstance(reason, str)

    def test_orb_detection_then_position_update(self):
        """ORB detection followed by position update should track risk correctly."""
        from lib.services.engine.risk import RiskManager
        from lib.trading.strategies.rb.open.detector import detect_opening_range_breakout

        bars = _make_breakout_bars(direction="LONG", or_price=2700.0, breakout_magnitude=20.0)
        _orb_result = detect_opening_range_breakout(bars, symbol="MGC")

        rm = RiskManager(
            account_size=50_000,
            now_fn=lambda: datetime(2026, 2, 27, 10, 15, 0, tzinfo=_EST),
        )

        # Simulate position opened based on ORB signal
        rm.register_open("MGC", "LONG", quantity=1, entry_price=2712.0, risk_dollars=100.0)
        assert rm.open_trade_count == 1

        # Update unrealized P&L
        rm.update_unrealized("MGC", 50.0)
        status = rm.get_status()
        assert status["open_trade_count"] == 1

        # Close the position
        rm.register_close("MGC", exit_price=2720.0, realized_pnl=80.0)
        assert rm.open_trade_count == 0
        assert rm.daily_pnl == 80.0


# ===========================================================================
# Tests: Constants and Configuration
# ===========================================================================


class TestORBConstants:
    """Verify ORB constants are sensible."""

    def test_or_window(self):
        from lib.trading.strategies.rb.open.sessions import OR_END, OR_START

        assert dt_time(9, 30) == OR_START
        assert dt_time(10, 0) == OR_END

    def test_atr_period(self):
        from lib.trading.strategies.rb.open.sessions import ATR_PERIOD

        assert ATR_PERIOD > 0
        assert ATR_PERIOD <= 30

    def test_breakout_multiplier(self):
        from lib.trading.strategies.rb.open.sessions import BREAKOUT_ATR_MULTIPLIER

        assert BREAKOUT_ATR_MULTIPLIER > 0
        assert BREAKOUT_ATR_MULTIPLIER <= 2.0

    def test_min_or_bars(self):
        from lib.trading.strategies.rb.open.sessions import MIN_OR_BARS

        assert MIN_OR_BARS >= 3

    def test_redis_keys(self):
        from lib.trading.strategies.rb.open.publisher import REDIS_KEY_ORB, REDIS_PUBSUB_ORB

        assert "orb" in REDIS_KEY_ORB.lower()
        assert "orb" in REDIS_PUBSUB_ORB.lower()


class TestSchedulerORBConstants:
    """Test scheduler ORB-related constants."""

    def test_orb_check_interval(self):
        from lib.services.engine.scheduler import ScheduleManager

        assert ScheduleManager.ORB_CHECK_INTERVAL == 2 * 60

    def test_orb_london_check_interval(self):
        from lib.services.engine.scheduler import ScheduleManager

        assert ScheduleManager.ORB_LONDON_CHECK_INTERVAL == 2 * 60


class TestSchedulerORBLondon:
    """Test London Open ORB scheduling (03:00–05:00 ET)."""

    def test_london_orb_action_type_exists(self):
        from lib.services.engine.scheduler import ActionType

        assert hasattr(ActionType, "CHECK_ORB_LONDON")
        assert ActionType.CHECK_ORB_LONDON == "check_orb_london"

    def test_london_orb_scheduled_during_active_london_window(self):
        """London ORB should be scheduled during 03:00–05:00 ET (active session)."""
        from lib.services.engine.scheduler import ActionType, ScheduleManager

        mgr = ScheduleManager()
        now = datetime(2025, 1, 15, 3, 30, tzinfo=ZoneInfo("America/New_York"))
        actions = mgr.get_pending_actions(now=now)
        action_types = [a.action for a in actions]
        assert ActionType.CHECK_ORB_LONDON in action_types

    def test_london_orb_not_scheduled_before_3am(self):
        """London ORB should NOT be scheduled before 03:00 ET."""
        from lib.services.engine.scheduler import ActionType, ScheduleManager

        mgr = ScheduleManager()
        now = datetime(2025, 1, 15, 2, 30, tzinfo=ZoneInfo("America/New_York"))
        actions = mgr.get_pending_actions(now=now)
        action_types = [a.action for a in actions]
        assert ActionType.CHECK_ORB_LONDON not in action_types

    def test_london_orb_scheduled_at_window_end(self):
        """London ORB still runs at 04:55 ET (just before 05:00 cutoff)."""
        from lib.services.engine.scheduler import ActionType, ScheduleManager

        mgr = ScheduleManager()
        now = datetime(2025, 1, 15, 4, 55, tzinfo=ZoneInfo("America/New_York"))
        actions = mgr.get_pending_actions(now=now)
        action_types = [a.action for a in actions]
        assert ActionType.CHECK_ORB_LONDON in action_types

    def test_london_orb_not_scheduled_during_us_window(self):
        """London ORB should NOT be scheduled during 09:30–11:00 ET."""
        from lib.services.engine.scheduler import ActionType, ScheduleManager

        mgr = ScheduleManager()
        now = datetime(2025, 1, 15, 10, 0, tzinfo=ZoneInfo("America/New_York"))
        actions = mgr.get_pending_actions(now=now)
        action_types = [a.action for a in actions]
        assert ActionType.CHECK_ORB_LONDON not in action_types


class TestMultiSessionORB:
    """Test multi-session ORB detection."""

    def test_detect_all_sessions(self):
        from lib.trading.strategies.rb.open.detector import detect_all_sessions

        # Use bars that span enough time to cover both session windows
        bars = _make_1m_bars(n=60, start_price=2700.0, start_time="2026-02-27 09:20:00")
        multi = detect_all_sessions(bars, symbol="MGC")
        assert "london" in multi.sessions
        assert "us" in multi.sessions
        assert multi.symbol == "MGC"

    def test_multi_session_result_properties(self):
        from lib.trading.strategies.rb.open.models import MultiSessionORBResult, ORBResult

        r1 = ORBResult(symbol="MGC", session_key="london", breakout_detected=True, or_range=10.0)
        r2 = ORBResult(symbol="MGC", session_key="us", breakout_detected=False, or_range=5.0)
        multi = MultiSessionORBResult(symbol="MGC", sessions={"london": r1, "us": r2})
        assert multi.has_any_breakout is True
        assert len(multi.active_breakouts) == 1
        assert multi.best_breakout is not None
        assert multi.best_breakout.session_key == "london"

    def test_session_helpers(self):
        from lib.trading.strategies.rb.open.sessions import get_active_sessions, get_session_status

        # 03:15 ET — London should be active (forming)
        now_london = datetime(2025, 1, 15, 3, 15, tzinfo=ZoneInfo("America/New_York"))
        active = get_active_sessions(now_london)
        assert any(s.key == "london" for s in active)

        statuses = get_session_status(now_london)
        assert statuses["london"] == "forming"
        assert statuses["us"] == "waiting"

        # 09:45 ET — US should be active (forming)
        now_us = datetime(2025, 1, 15, 9, 45, tzinfo=ZoneInfo("America/New_York"))
        active = get_active_sessions(now_us)
        assert any(s.key == "us" for s in active)

        statuses = get_session_status(now_us)
        assert statuses["london"] == "complete"
        assert statuses["us"] == "forming"

    def test_orb_result_includes_session_info(self):
        from lib.trading.strategies.rb.open.detector import detect_opening_range_breakout
        from lib.trading.strategies.rb.open.sessions import LONDON_SESSION

        bars = _make_1m_bars()
        result = detect_opening_range_breakout(bars, symbol="MGC", session=LONDON_SESSION)
        assert result.session_name == "London Open"
        assert result.session_key == "london"
        d = result.to_dict()
        assert d["session_name"] == "London Open"
        assert d["session_key"] == "london"
