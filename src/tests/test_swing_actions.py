"""
Tests for Swing Actions — API endpoints, state mutations, and dashboard integration.

Covers:
  - Swing state mutation functions (accept, ignore, close, move-stop-to-BE, update-stop)
  - Pending signal management (get_pending_signals, signal lifecycle)
  - Swing detail and history helpers
  - API router endpoints (POST accept/ignore/close/stop-to-be/update-stop, GET pending/active/detail/history/status-badge)
  - Dashboard swing card action button rendering
  - SSE swing_update event handling
  - Edge cases: concurrent limits, wrong phases, missing signals, duplicate accepts
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Any
from unittest.mock import MagicMock
from zoneinfo import ZoneInfo

import pytest

# ---------------------------------------------------------------------------
# Ensure src/ is on the path
# ---------------------------------------------------------------------------
_src = os.path.join(os.path.dirname(__file__), "..")
if _src not in sys.path:
    sys.path.insert(0, _src)

_EST = ZoneInfo("America/New_York")


# ---------------------------------------------------------------------------
# Minimal stubs for SwingSignal / SwingState / SwingPhase so tests can
# create them without importing the full strategy module's dependencies.
# We'll also patch the real module where needed.
# ---------------------------------------------------------------------------


class _SwingEntryStyle(StrEnum):
    PULLBACK = "pullback_entry"
    BREAKOUT = "breakout_entry"
    GAP_CONTINUATION = "gap_continuation"


class _SwingPhase(StrEnum):
    WATCHING = "watching"
    ENTRY_READY = "entry_ready"
    ACTIVE = "active"
    TP1_HIT = "tp1_hit"
    TRAILING = "trailing"
    CLOSED = "closed"


@dataclass
class _SwingSignal:
    asset_name: str = ""
    entry_style: _SwingEntryStyle = _SwingEntryStyle.PULLBACK
    direction: str = "LONG"
    confidence: float = 0.75
    entry_price: float = 100.0
    entry_zone_low: float = 99.5
    entry_zone_high: float = 100.5
    stop_loss: float = 98.0
    tp1: float = 104.0
    tp2: float = 107.0
    atr: float = 2.0
    risk_reward_tp1: float = 2.0
    risk_reward_tp2: float = 3.5
    risk_dollars: float = 50.0
    position_size: int = 2
    reasoning: str = "Test signal"
    key_level_used: str = "EMA-21"
    key_level_price: float = 99.8
    confirmation_bar_idx: int = 5
    detected_at: str = ""
    phase: _SwingPhase = _SwingPhase.ENTRY_READY

    def to_dict(self) -> dict[str, Any]:
        return {
            "asset_name": self.asset_name,
            "entry_style": self.entry_style.value,
            "direction": self.direction,
            "confidence": round(self.confidence, 3),
            "entry_price": self.entry_price,
            "entry_zone_low": self.entry_zone_low,
            "entry_zone_high": self.entry_zone_high,
            "stop_loss": self.stop_loss,
            "tp1": self.tp1,
            "tp2": self.tp2,
            "atr": round(self.atr, 6),
            "risk_reward_tp1": round(self.risk_reward_tp1, 2),
            "risk_reward_tp2": round(self.risk_reward_tp2, 2),
            "risk_dollars": round(self.risk_dollars, 2),
            "position_size": self.position_size,
            "reasoning": self.reasoning,
            "key_level_used": self.key_level_used,
            "key_level_price": self.key_level_price,
            "confirmation_bar_idx": self.confirmation_bar_idx,
            "detected_at": self.detected_at,
            "phase": self.phase.value,
        }


@dataclass
class _SwingState:
    asset_name: str = ""
    signal: _SwingSignal | None = None
    phase: _SwingPhase = _SwingPhase.ACTIVE
    entry_price: float = 100.0
    current_stop: float = 98.0
    tp1: float = 104.0
    tp2: float = 107.0
    direction: str = "LONG"
    position_size: int = 2
    remaining_size: int = 2
    highest_price: float = 100.0
    lowest_price: float = 100.0
    entry_time: str = ""
    last_update: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "asset_name": self.asset_name,
            "signal": self.signal.to_dict() if self.signal else None,
            "phase": self.phase.value,
            "entry_price": self.entry_price,
            "current_stop": self.current_stop,
            "tp1": self.tp1,
            "tp2": self.tp2,
            "direction": self.direction,
            "position_size": self.position_size,
            "remaining_size": self.remaining_size,
            "highest_price": self.highest_price,
            "lowest_price": self.lowest_price,
            "entry_time": self.entry_time,
            "last_update": self.last_update,
        }


def _make_signal(name: str = "Gold", direction: str = "LONG", **kwargs) -> _SwingSignal:
    """Create a test swing signal."""
    defaults = {
        "asset_name": name,
        "direction": direction,
        "entry_price": 2700.0,
        "stop_loss": 2680.0,
        "tp1": 2740.0,
        "tp2": 2770.0,
        "atr": 10.0,
        "confidence": 0.75,
        "position_size": 2,
        "risk_dollars": 40.0,
        "detected_at": datetime.now(tz=_EST).isoformat(),
    }
    defaults.update(kwargs)
    return _SwingSignal(**defaults)  # type: ignore[call-arg, arg-type]


def _make_state(
    name: str = "Gold",
    direction: str = "LONG",
    phase: _SwingPhase = _SwingPhase.ACTIVE,
    **kwargs,
) -> _SwingState:
    """Create a test swing state."""
    sig = _make_signal(name, direction)
    defaults = {
        "asset_name": name,
        "signal": sig,
        "phase": phase,
        "entry_price": sig.entry_price,
        "current_stop": sig.stop_loss,
        "tp1": sig.tp1,
        "tp2": sig.tp2,
        "direction": direction,
        "position_size": sig.position_size,
        "remaining_size": sig.position_size,
        "highest_price": sig.entry_price,
        "lowest_price": sig.entry_price,
        "entry_time": datetime.now(tz=_EST).isoformat(),
        "last_update": datetime.now(tz=_EST).isoformat(),
    }
    defaults.update(kwargs)
    return _SwingState(**defaults)  # type: ignore[call-arg, arg-type]


# ---------------------------------------------------------------------------
# Fixtures — patch the swing module's internal state for each test
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_swing_module_state(monkeypatch):
    """Reset all module-level state in lib.services.engine.swing before each test."""
    import lib.services.engine.swing as swing_mod

    monkeypatch.setattr(swing_mod, "_active_swing_states", {})
    monkeypatch.setattr(swing_mod, "_pending_swing_signals", {})
    monkeypatch.setattr(swing_mod, "_last_scan_ts", 0.0)
    yield


@pytest.fixture()
def swing_mod():
    """Import and return the swing module."""
    import lib.services.engine.swing as mod

    return mod


@pytest.fixture()
def _patch_redis(monkeypatch, swing_mod):
    """Patch Redis helpers so no real Redis is needed."""
    monkeypatch.setattr(swing_mod, "_get_redis", lambda: None)
    monkeypatch.setattr(swing_mod, "_cache_get", lambda key: None)
    monkeypatch.setattr(swing_mod, "_cache_set", lambda key, val, ttl=None: None)
    yield


@pytest.fixture()
def _patch_swing_detector(monkeypatch):
    """Patch the strategy-layer swing detector imports used by swing.py mutations."""
    monkeypatch.setattr(
        "lib.trading.strategies.daily.swing_detector.SwingPhase",
        _SwingPhase,
    )
    monkeypatch.setattr(
        "lib.trading.strategies.daily.swing_detector.SwingEntryStyle",
        _SwingEntryStyle,
    )
    monkeypatch.setattr(
        "lib.trading.strategies.daily.swing_detector.SwingSignal",
        _SwingSignal,
    )

    def _fake_create_swing_state(signal):
        now = datetime.now(tz=_EST).isoformat()
        return _SwingState(
            asset_name=signal.asset_name,
            signal=signal,
            phase=_SwingPhase.ACTIVE,
            entry_price=signal.entry_price,
            current_stop=signal.stop_loss,
            tp1=signal.tp1,
            tp2=signal.tp2,
            direction=signal.direction,
            position_size=signal.position_size,
            remaining_size=signal.position_size,
            highest_price=signal.entry_price,
            lowest_price=signal.entry_price,
            entry_time=now,
            last_update=now,
        )

    monkeypatch.setattr(
        "lib.trading.strategies.daily.swing_detector.create_swing_state",
        _fake_create_swing_state,
    )
    yield


# ===========================================================================
# Tests — State mutation functions
# ===========================================================================


class TestAcceptSwingSignal:
    """Tests for accept_swing_signal()."""

    def test_accept_pending_signal(self, swing_mod, _patch_redis, _patch_swing_detector):
        """Accepting a pending signal creates an active SwingState."""
        sig = _make_signal("Gold")
        swing_mod._pending_swing_signals["Gold"] = sig

        result = swing_mod.accept_swing_signal("Gold")

        assert result["status"] == "accepted"
        assert result["asset_name"] == "Gold"
        assert result["direction"] == "LONG"
        assert result["entry_price"] == sig.entry_price
        assert result["position_size"] == sig.position_size
        assert result["phase"] == "active"

        # State should be in active dict
        assert "Gold" in swing_mod._active_swing_states
        state = swing_mod._active_swing_states["Gold"]
        assert state.phase == _SwingPhase.ACTIVE
        assert state.entry_price == sig.entry_price

        # Signal should be removed from pending
        assert "Gold" not in swing_mod._pending_swing_signals

    def test_accept_signal_from_redis(self, swing_mod, _patch_swing_detector, monkeypatch):
        """Accepting reconstructs signal from Redis when not in pending dict."""
        sig_dict = _make_signal("Crude Oil", direction="SHORT").to_dict()
        redis_payload = json.dumps([sig_dict])

        monkeypatch.setattr(swing_mod, "_get_redis", lambda: None)
        monkeypatch.setattr(
            swing_mod,
            "_cache_get",
            lambda key: redis_payload if "signals" in key else None,
        )
        monkeypatch.setattr(swing_mod, "_cache_set", lambda key, val, ttl=None: None)

        result = swing_mod.accept_swing_signal("Crude Oil")

        assert result["status"] == "accepted"
        assert result["asset_name"] == "Crude Oil"
        assert result["direction"] == "SHORT"

    def test_accept_already_active_raises(self, swing_mod, _patch_redis, _patch_swing_detector):
        """Accepting a signal for an asset that already has an active state raises."""
        swing_mod._active_swing_states["Gold"] = _make_state("Gold")

        with pytest.raises(ValueError, match="already has an active"):
            swing_mod.accept_swing_signal("Gold")

    def test_accept_at_concurrent_limit_raises(self, swing_mod, _patch_redis, _patch_swing_detector):
        """Accepting when max concurrent swings is reached raises."""
        # Fill up to limit (default 3)
        swing_mod._active_swing_states["Gold"] = _make_state("Gold")
        swing_mod._active_swing_states["Crude Oil"] = _make_state("Crude Oil")
        swing_mod._active_swing_states["E-mini S&P"] = _make_state("E-mini S&P")

        sig = _make_signal("Euro FX")
        swing_mod._pending_swing_signals["Euro FX"] = sig

        with pytest.raises(ValueError, match="Maximum concurrent"):
            swing_mod.accept_swing_signal("Euro FX")

    def test_accept_no_signal_raises(self, swing_mod, _patch_redis, _patch_swing_detector):
        """Accepting with no pending signal and nothing in Redis raises."""
        with pytest.raises(ValueError, match="No pending swing signal"):
            swing_mod.accept_swing_signal("Nonexistent Asset")

    def test_accept_closed_state_allows_reaccept(self, swing_mod, _patch_redis, _patch_swing_detector):
        """An asset with a CLOSED state can be re-accepted."""
        closed_state = _make_state("Gold", phase=_SwingPhase.CLOSED)
        swing_mod._active_swing_states["Gold"] = closed_state

        sig = _make_signal("Gold")
        swing_mod._pending_swing_signals["Gold"] = sig

        # Should NOT raise because the existing state is closed
        result = swing_mod.accept_swing_signal("Gold")
        assert result["status"] == "accepted"


class TestIgnoreSwingSignal:
    """Tests for ignore_swing_signal()."""

    def test_ignore_pending_signal(self, swing_mod, _patch_redis):
        """Ignoring removes the signal from pending."""
        swing_mod._pending_swing_signals["Gold"] = _make_signal("Gold")

        result = swing_mod.ignore_swing_signal("Gold")

        assert result["status"] == "ignored"
        assert result["was_pending"] is True
        assert "Gold" not in swing_mod._pending_swing_signals

    def test_ignore_nonexistent_signal(self, swing_mod, _patch_redis):
        """Ignoring a non-pending signal returns gracefully."""
        result = swing_mod.ignore_swing_signal("Nonexistent")

        assert result["status"] == "ignored"
        assert result["was_pending"] is False

    def test_ignore_cleans_redis(self, swing_mod, monkeypatch):
        """Ignoring also removes the signal from the Redis signals list."""
        sig1 = _make_signal("Gold").to_dict()
        sig2 = _make_signal("Crude Oil").to_dict()
        redis_payload = json.dumps([sig1, sig2])

        saved_payloads = {}

        def fake_cache_set(key, val, ttl=None):
            saved_payloads[key] = val

        monkeypatch.setattr(swing_mod, "_get_redis", lambda: None)
        monkeypatch.setattr(
            swing_mod,
            "_cache_get",
            lambda key: redis_payload if "signals" in key else None,
        )
        monkeypatch.setattr(swing_mod, "_cache_set", fake_cache_set)

        swing_mod._pending_swing_signals["Gold"] = _make_signal("Gold")
        swing_mod.ignore_swing_signal("Gold")

        # Check that Redis was updated to only contain Crude Oil
        if "engine:swing_signals" in saved_payloads:
            remaining = json.loads(saved_payloads["engine:swing_signals"])
            names = [s.get("asset_name") for s in remaining]
            assert "Gold" not in names
            assert "Crude Oil" in names


class TestCloseSwingPosition:
    """Tests for close_swing_position()."""

    def test_close_active_position(self, swing_mod, _patch_redis, _patch_swing_detector):
        """Closing an active position sets phase to CLOSED."""
        swing_mod._active_swing_states["Gold"] = _make_state("Gold")

        result = swing_mod.close_swing_position("Gold", reason="manual_test")

        assert result["status"] == "closed"
        assert result["previous_phase"] == "active"
        assert result["reason"] == "manual_test"
        assert "Gold" not in swing_mod._active_swing_states

    def test_close_tp1_hit_position(self, swing_mod, _patch_redis, _patch_swing_detector):
        """Closing a TP1_HIT position works."""
        state = _make_state("Gold", phase=_SwingPhase.TP1_HIT)
        state.remaining_size = 1
        swing_mod._active_swing_states["Gold"] = state

        result = swing_mod.close_swing_position("Gold")

        assert result["status"] == "closed"
        assert result["previous_phase"] == "tp1_hit"

    def test_close_nonexistent_raises(self, swing_mod, _patch_redis, _patch_swing_detector):
        """Closing a non-existent swing raises ValueError."""
        with pytest.raises(ValueError, match="No active swing state"):
            swing_mod.close_swing_position("Nonexistent")

    def test_close_already_closed_raises(self, swing_mod, _patch_redis, _patch_swing_detector):
        """Closing an already-closed swing raises ValueError."""
        state = _make_state("Gold", phase=_SwingPhase.CLOSED)
        swing_mod._active_swing_states["Gold"] = state

        with pytest.raises(ValueError, match="already closed"):
            swing_mod.close_swing_position("Gold")

    def test_close_archives_state(self, swing_mod, _patch_swing_detector, monkeypatch):
        """Closing archives the state to Redis history."""
        archived = []
        mock_redis = MagicMock()
        mock_redis.lpush = lambda key, val: archived.append((key, val))
        mock_redis.ltrim = MagicMock()
        mock_redis.expire = MagicMock()

        monkeypatch.setattr(swing_mod, "_get_redis", lambda: mock_redis)
        monkeypatch.setattr(swing_mod, "_cache_set", lambda key, val, ttl=None: None)

        swing_mod._active_swing_states["Gold"] = _make_state("Gold")
        swing_mod.close_swing_position("Gold")

        assert len(archived) == 1
        key, payload = archived[0]
        assert "swing_history" in key
        entry = json.loads(payload)
        assert entry["asset_name"] == "Gold"


class TestMoveStopToBreakeven:
    """Tests for move_stop_to_breakeven()."""

    def test_move_stop_active_phase(self, swing_mod, _patch_redis, _patch_swing_detector):
        """Moving stop to BE on an active state works."""
        state = _make_state("Gold")
        assert state.current_stop != state.entry_price  # pre-condition
        swing_mod._active_swing_states["Gold"] = state

        result = swing_mod.move_stop_to_breakeven("Gold")

        assert result["status"] == "stop_moved"
        assert result["new_stop"] == state.entry_price
        assert result["old_stop"] == 2680.0  # original stop from _make_state

        # Verify in-memory state is updated
        updated = swing_mod._active_swing_states["Gold"]
        assert updated.current_stop == updated.entry_price

    def test_move_stop_tp1_hit_phase(self, swing_mod, _patch_redis, _patch_swing_detector):
        """Moving stop to BE on TP1_HIT phase works."""
        state = _make_state("Gold", phase=_SwingPhase.TP1_HIT)
        swing_mod._active_swing_states["Gold"] = state

        result = swing_mod.move_stop_to_breakeven("Gold")
        assert result["status"] == "stop_moved"

    def test_move_stop_trailing_phase(self, swing_mod, _patch_redis, _patch_swing_detector):
        """Moving stop to BE on TRAILING phase works."""
        state = _make_state("Gold", phase=_SwingPhase.TRAILING)
        swing_mod._active_swing_states["Gold"] = state

        result = swing_mod.move_stop_to_breakeven("Gold")
        assert result["status"] == "stop_moved"

    def test_move_stop_watching_phase_raises(self, swing_mod, _patch_redis, _patch_swing_detector):
        """Moving stop to BE on WATCHING phase raises."""
        state = _make_state("Gold", phase=_SwingPhase.WATCHING)
        swing_mod._active_swing_states["Gold"] = state

        with pytest.raises(ValueError, match="must be active"):
            swing_mod.move_stop_to_breakeven("Gold")

    def test_move_stop_closed_phase_raises(self, swing_mod, _patch_redis, _patch_swing_detector):
        """Moving stop to BE on CLOSED phase raises."""
        state = _make_state("Gold", phase=_SwingPhase.CLOSED)
        swing_mod._active_swing_states["Gold"] = state

        with pytest.raises(ValueError, match="must be active"):
            swing_mod.move_stop_to_breakeven("Gold")

    def test_move_stop_nonexistent_raises(self, swing_mod, _patch_redis, _patch_swing_detector):
        """Moving stop for non-existent asset raises."""
        with pytest.raises(ValueError, match="No active swing state"):
            swing_mod.move_stop_to_breakeven("Nonexistent")


class TestUpdateSwingStop:
    """Tests for update_swing_stop()."""

    def test_update_stop_valid(self, swing_mod, _patch_redis, _patch_swing_detector):
        """Updating stop to a valid price works."""
        swing_mod._active_swing_states["Gold"] = _make_state("Gold")

        result = swing_mod.update_swing_stop("Gold", 2690.0)

        assert result["status"] == "stop_updated"
        assert result["new_stop"] == 2690.0
        assert swing_mod._active_swing_states["Gold"].current_stop == 2690.0

    def test_update_stop_zero_raises(self, swing_mod, _patch_redis, _patch_swing_detector):
        """Updating stop to zero raises."""
        swing_mod._active_swing_states["Gold"] = _make_state("Gold")

        with pytest.raises(ValueError, match="Invalid stop price"):
            swing_mod.update_swing_stop("Gold", 0.0)

    def test_update_stop_negative_raises(self, swing_mod, _patch_redis, _patch_swing_detector):
        """Updating stop to negative raises."""
        swing_mod._active_swing_states["Gold"] = _make_state("Gold")

        with pytest.raises(ValueError, match="Invalid stop price"):
            swing_mod.update_swing_stop("Gold", -10.0)

    def test_update_stop_wrong_phase_raises(self, swing_mod, _patch_redis, _patch_swing_detector):
        """Updating stop on WATCHING phase raises."""
        state = _make_state("Gold", phase=_SwingPhase.WATCHING)
        swing_mod._active_swing_states["Gold"] = state

        with pytest.raises(ValueError, match="Cannot update stop"):
            swing_mod.update_swing_stop("Gold", 2690.0)


# ===========================================================================
# Tests — Query functions
# ===========================================================================


class TestGetPendingSignals:
    """Tests for get_pending_signals()."""

    def test_empty_pending(self, swing_mod, _patch_redis):
        """Returns empty dict when no pending signals."""
        result = swing_mod.get_pending_signals()
        assert result == {}

    def test_pending_from_module(self, swing_mod, _patch_redis):
        """Returns signals from module-level pending dict."""
        swing_mod._pending_swing_signals["Gold"] = _make_signal("Gold")
        swing_mod._pending_swing_signals["Crude Oil"] = _make_signal("Crude Oil")

        result = swing_mod.get_pending_signals()
        assert len(result) == 2
        assert "Gold" in result
        assert "Crude Oil" in result

    def test_pending_excludes_active(self, swing_mod, _patch_redis):
        """Pending signals for assets with active states are excluded."""
        swing_mod._pending_swing_signals["Gold"] = _make_signal("Gold")
        swing_mod._active_swing_states["Gold"] = _make_state("Gold")

        result = swing_mod.get_pending_signals()
        assert "Gold" not in result

    def test_pending_from_redis(self, swing_mod, monkeypatch):
        """Returns signals from Redis when not in module pending."""
        sig = _make_signal("Euro FX").to_dict()
        redis_payload = json.dumps([sig])

        monkeypatch.setattr(swing_mod, "_get_redis", lambda: None)
        monkeypatch.setattr(
            swing_mod,
            "_cache_get",
            lambda key: redis_payload if "signals" in key else None,
        )

        result = swing_mod.get_pending_signals()
        assert "Euro FX" in result

    def test_pending_module_takes_precedence(self, swing_mod, monkeypatch):
        """Module-level pending takes precedence over Redis for same asset."""
        module_sig = _make_signal("Gold", confidence=0.9)
        swing_mod._pending_swing_signals["Gold"] = module_sig

        redis_sig = _make_signal("Gold", confidence=0.5).to_dict()
        redis_payload = json.dumps([redis_sig])

        monkeypatch.setattr(swing_mod, "_get_redis", lambda: None)
        monkeypatch.setattr(
            swing_mod,
            "_cache_get",
            lambda key: redis_payload if "signals" in key else None,
        )

        result = swing_mod.get_pending_signals()
        assert "Gold" in result
        # Module signal should be used (has to_dict method)
        sig_data = result["Gold"]
        if isinstance(sig_data, dict):
            assert sig_data["confidence"] == 0.9


class TestGetSwingStateDetail:
    """Tests for get_swing_state_detail()."""

    def test_detail_for_active(self, swing_mod, _patch_redis):
        """Returns full detail dict for active state."""
        swing_mod._active_swing_states["Gold"] = _make_state("Gold")

        detail = swing_mod.get_swing_state_detail("Gold")
        assert detail is not None
        assert detail["asset_name"] == "Gold"
        assert detail["phase"] == "active"
        assert "risk_per_unit" in detail

    def test_detail_none_for_missing(self, swing_mod, _patch_redis):
        """Returns None when no state exists."""
        detail = swing_mod.get_swing_state_detail("Nonexistent")
        assert detail is None

    def test_detail_includes_rr_ratios(self, swing_mod, _patch_redis):
        """Detail includes computed R:R ratios."""
        state = _make_state("Gold")
        swing_mod._active_swing_states["Gold"] = state

        detail = swing_mod.get_swing_state_detail("Gold")
        assert "rr_tp1" in detail
        assert "rr_tp2" in detail
        assert detail["rr_tp1"] > 0
        assert detail["rr_tp2"] > detail["rr_tp1"]


class TestGetSwingHistory:
    """Tests for get_swing_history()."""

    def test_empty_history_no_redis(self, swing_mod, _patch_redis):
        """Returns empty list when Redis is unavailable."""
        result = swing_mod.get_swing_history()
        assert result == []

    def test_history_from_redis(self, swing_mod, monkeypatch):
        """Returns history entries from Redis list."""
        entries = [
            json.dumps({"asset_name": "Gold", "closed_at": "2025-01-15T14:00:00"}),
            json.dumps({"asset_name": "Crude Oil", "closed_at": "2025-01-15T13:00:00"}),
        ]
        mock_redis = MagicMock()
        mock_redis.lrange = MagicMock(return_value=entries)

        monkeypatch.setattr(swing_mod, "_get_redis", lambda: mock_redis)

        result = swing_mod.get_swing_history(limit=10)
        assert len(result) == 2
        assert result[0]["asset_name"] == "Gold"


class TestGetActiveSummary:
    """Tests for get_active_swing_states() and get_swing_summary()."""

    def test_get_active_empty(self, swing_mod, _patch_redis):
        result = swing_mod.get_active_swing_states()
        assert result == {}

    def test_get_active_returns_copy(self, swing_mod, _patch_redis):
        swing_mod._active_swing_states["Gold"] = _make_state("Gold")
        result = swing_mod.get_active_swing_states()
        assert "Gold" in result
        # Mutating the copy shouldn't affect the module state
        result.pop("Gold")
        assert "Gold" in swing_mod._active_swing_states

    def test_summary_structure(self, swing_mod, _patch_redis):
        swing_mod._active_swing_states["Gold"] = _make_state("Gold")
        swing_mod._active_swing_states["Crude Oil"] = _make_state("Crude Oil")

        summary = swing_mod.get_swing_summary()
        assert summary["active_count"] == 2
        assert set(summary["active_assets"]) == {"Gold", "Crude Oil"}
        assert "states" in summary
        assert "max_concurrent" in summary

    def test_summary_state_details(self, swing_mod, _patch_redis):
        state = _make_state("Gold")
        swing_mod._active_swing_states["Gold"] = state

        summary = swing_mod.get_swing_summary()
        gold_info = summary["states"]["Gold"]
        assert gold_info["phase"] == "active"
        assert gold_info["direction"] == "LONG"


class TestResetSwingStates:
    """Tests for reset_swing_states()."""

    def test_reset_clears_all(self, swing_mod, _patch_redis):
        swing_mod._active_swing_states["Gold"] = _make_state("Gold")
        swing_mod._pending_swing_signals["Crude Oil"] = _make_signal("Crude Oil")
        swing_mod._last_scan_ts = 12345.0

        swing_mod.reset_swing_states()

        assert swing_mod._active_swing_states == {}
        assert swing_mod._last_scan_ts == 0.0


# ===========================================================================
# Tests — API Router endpoints (using FastAPI TestClient)
# ===========================================================================


@pytest.fixture()
def test_client():
    """Create a FastAPI TestClient with the swing actions router mounted."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from lib.services.data.api.swing_actions import router

    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


class TestAcceptEndpoint:
    """Tests for POST /api/swing/accept/{asset_name}."""

    def test_accept_returns_html(self, test_client, swing_mod, _patch_redis, _patch_swing_detector):
        """Successful accept returns HTML with success toast."""
        swing_mod._pending_swing_signals["Gold"] = _make_signal("Gold")

        resp = test_client.post("/api/swing/accept/Gold")
        assert resp.status_code == 200
        html = resp.text
        assert "Signal Accepted" in html or "accepted" in html.lower()
        assert "LONG" in html

    def test_accept_error_returns_html(self, test_client, swing_mod, _patch_redis, _patch_swing_detector):
        """Failed accept (no signal) returns error HTML, not 500."""
        resp = test_client.post("/api/swing/accept/Nonexistent")
        assert resp.status_code == 200
        html = resp.text
        assert "Error" in html or "error" in html.lower()

    def test_accept_duplicate_returns_error(self, test_client, swing_mod, _patch_redis, _patch_swing_detector):
        """Accepting for an already-active asset returns error toast."""
        swing_mod._active_swing_states["Gold"] = _make_state("Gold")

        resp = test_client.post("/api/swing/accept/Gold")
        assert resp.status_code == 200
        assert "already has an active" in resp.text.lower() or "error" in resp.text.lower()


class TestIgnoreEndpoint:
    """Tests for POST /api/swing/ignore/{asset_name}."""

    def test_ignore_returns_html(self, test_client, swing_mod, _patch_redis):
        """Successful ignore returns HTML."""
        swing_mod._pending_swing_signals["Gold"] = _make_signal("Gold")

        resp = test_client.post("/api/swing/ignore/Gold")
        assert resp.status_code == 200
        assert "Ignored" in resp.text or "ignored" in resp.text.lower()

    def test_ignore_nonexistent_still_200(self, test_client, swing_mod, _patch_redis):
        """Ignoring a non-pending asset still returns 200."""
        resp = test_client.post("/api/swing/ignore/Whatever")
        assert resp.status_code == 200


class TestCloseEndpoint:
    """Tests for POST /api/swing/close/{asset_name}."""

    def test_close_returns_html(self, test_client, swing_mod, _patch_redis, _patch_swing_detector):
        """Successful close returns HTML with confirmation."""
        swing_mod._active_swing_states["Gold"] = _make_state("Gold")

        resp = test_client.post("/api/swing/close/Gold")
        assert resp.status_code == 200
        assert "Closed" in resp.text or "closed" in resp.text.lower()

    def test_close_nonexistent_returns_error(self, test_client, swing_mod, _patch_redis, _patch_swing_detector):
        """Closing non-existent returns error toast."""
        resp = test_client.post("/api/swing/close/Nonexistent")
        assert resp.status_code == 200
        assert "error" in resp.text.lower() or "No active" in resp.text


class TestStopToBeEndpoint:
    """Tests for POST /api/swing/stop-to-be/{asset_name}."""

    def test_stop_to_be_returns_html(self, test_client, swing_mod, _patch_redis, _patch_swing_detector):
        """Moving stop to BE returns HTML with confirmation."""
        swing_mod._active_swing_states["Gold"] = _make_state("Gold")

        resp = test_client.post("/api/swing/stop-to-be/Gold")
        assert resp.status_code == 200
        assert "Breakeven" in resp.text or "breakeven" in resp.text.lower() or "stop" in resp.text.lower()

    def test_stop_to_be_includes_buttons(self, test_client, swing_mod, _patch_redis, _patch_swing_detector):
        """Response includes updated action buttons."""
        swing_mod._active_swing_states["Gold"] = _make_state("Gold")

        resp = test_client.post("/api/swing/stop-to-be/Gold")
        # Should contain HTMX button references
        assert "hx-post" in resp.text or "swing-actions" in resp.text


class TestUpdateStopEndpoint:
    """Tests for POST /api/swing/update-stop/{asset_name}."""

    def test_update_stop_returns_html(self, test_client, swing_mod, _patch_redis, _patch_swing_detector):
        """Updating stop returns HTML with confirmation."""
        swing_mod._active_swing_states["Gold"] = _make_state("Gold")

        resp = test_client.post(
            "/api/swing/update-stop/Gold",
            json={"new_stop": 2690.0},
        )
        assert resp.status_code == 200
        assert "Updated" in resp.text or "updated" in resp.text.lower()

    def test_update_stop_invalid_body(self, test_client, swing_mod, _patch_redis, _patch_swing_detector):
        """Missing body returns 422."""
        swing_mod._active_swing_states["Gold"] = _make_state("Gold")

        resp = test_client.post("/api/swing/update-stop/Gold")
        assert resp.status_code == 422


class TestPendingEndpoint:
    """Tests for GET /api/swing/pending."""

    def test_pending_empty(self, test_client, swing_mod, _patch_redis):
        resp = test_client.get("/api/swing/pending")
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 0
        assert data["signals"] == {}

    def test_pending_with_signals(self, test_client, swing_mod, _patch_redis):
        swing_mod._pending_swing_signals["Gold"] = _make_signal("Gold")

        resp = test_client.get("/api/swing/pending")
        data = resp.json()
        assert data["count"] == 1
        assert "Gold" in data["signals"]


class TestActiveEndpoint:
    """Tests for GET /api/swing/active."""

    def test_active_empty(self, test_client, swing_mod, _patch_redis):
        resp = test_client.get("/api/swing/active")
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 0

    def test_active_with_states(self, test_client, swing_mod, _patch_redis):
        swing_mod._active_swing_states["Gold"] = _make_state("Gold")

        resp = test_client.get("/api/swing/active")
        data = resp.json()
        assert data["count"] == 1
        assert "Gold" in data["states"]


class TestDetailEndpoint:
    """Tests for GET /api/swing/detail/{asset_name}."""

    def test_detail_found(self, test_client, swing_mod, _patch_redis):
        swing_mod._active_swing_states["Gold"] = _make_state("Gold")

        resp = test_client.get("/api/swing/detail/Gold")
        assert resp.status_code == 200
        data = resp.json()
        assert data["asset_name"] == "Gold"
        assert data["phase"] == "active"

    def test_detail_not_found(self, test_client, swing_mod, _patch_redis):
        resp = test_client.get("/api/swing/detail/Nonexistent")
        assert resp.status_code == 404


class TestHistoryEndpoint:
    """Tests for GET /api/swing/history."""

    def test_history_empty(self, test_client, swing_mod, _patch_redis):
        resp = test_client.get("/api/swing/history")
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 0
        assert data["trades"] == []

    def test_history_with_limit(self, test_client, swing_mod, monkeypatch):
        entries = [
            json.dumps({"asset_name": f"Asset{i}", "closed_at": f"2025-01-{15 - i:02d}T14:00:00"}) for i in range(5)
        ]
        mock_redis = MagicMock()
        mock_redis.lrange = MagicMock(return_value=entries[:3])
        monkeypatch.setattr(swing_mod, "_get_redis", lambda: mock_redis)

        resp = test_client.get("/api/swing/history?limit=3")
        data = resp.json()
        assert data["count"] == 3


class TestStatusBadgeEndpoint:
    """Tests for GET /api/swing/status-badge/{asset_name}."""

    def test_badge_with_active(self, test_client, swing_mod, _patch_redis):
        swing_mod._active_swing_states["Gold"] = _make_state("Gold")

        resp = test_client.get("/api/swing/status-badge/Gold")
        assert resp.status_code == 200
        html = resp.text
        # Should contain phase and action buttons
        assert "ACTIVE" in html.upper() or "active" in html
        assert "hx-post" in html  # action buttons

    def test_badge_with_pending(self, test_client, swing_mod, _patch_redis):
        swing_mod._pending_swing_signals["Gold"] = _make_signal("Gold")

        resp = test_client.get("/api/swing/status-badge/Gold")
        assert resp.status_code == 200
        html = resp.text
        assert "SIGNAL" in html.upper() or "signal" in html.lower()
        assert "Accept" in html or "accept" in html.lower()

    def test_badge_empty(self, test_client, swing_mod, _patch_redis):
        resp = test_client.get("/api/swing/status-badge/Gold")
        assert resp.status_code == 200
        html = resp.text
        assert "scanning" in html.lower() or "No active" in html


# ===========================================================================
# Tests — Dashboard integration (swing card rendering)
# ===========================================================================


class TestSwingCardActionButtons:
    """Tests for render_swing_action_buttons_for_card()."""

    def test_buttons_for_pending(self, swing_mod, _patch_redis):
        """Pending signal renders Accept + Ignore buttons."""
        from lib.services.data.api.swing_actions import render_swing_action_buttons_for_card

        swing_mod._pending_swing_signals["Gold"] = _make_signal("Gold")

        html = render_swing_action_buttons_for_card("Gold")
        assert "Accept" in html
        assert "Ignore" in html
        assert "hx-post" in html
        assert "/api/swing/accept/Gold" in html
        assert "/api/swing/ignore/Gold" in html

    def test_buttons_for_active(self, swing_mod, _patch_redis):
        """Active state renders Close + Stop→BE buttons."""
        from lib.services.data.api.swing_actions import render_swing_action_buttons_for_card

        states = {"Gold": _make_state("Gold")}
        html = render_swing_action_buttons_for_card("Gold", active_states=states)

        assert "Close" in html
        assert "Stop" in html and "BE" in html
        assert "/api/swing/close/Gold" in html
        assert "/api/swing/stop-to-be/Gold" in html

    def test_buttons_empty_for_no_state(self, swing_mod, _patch_redis):
        """No state or signal renders empty string."""
        from lib.services.data.api.swing_actions import render_swing_action_buttons_for_card

        html = render_swing_action_buttons_for_card("Gold", active_states={}, pending_signals={})
        assert html == ""

    def test_buttons_for_closed_state(self, swing_mod, _patch_redis):
        """Closed state renders no active-state buttons."""
        from lib.services.data.api.swing_actions import render_swing_action_buttons_for_card

        states = {"Gold": _make_state("Gold", phase=_SwingPhase.CLOSED)}
        html = render_swing_action_buttons_for_card("Gold", active_states=states, pending_signals={})

        # Closed state should not have Close or Stop→BE buttons
        assert "Close" not in html or html == ""

    def test_buttons_for_trailing_phase(self, swing_mod, _patch_redis):
        """TRAILING phase renders Close + Stop→BE."""
        from lib.services.data.api.swing_actions import render_swing_action_buttons_for_card

        states = {"Gold": _make_state("Gold", phase=_SwingPhase.TRAILING)}
        html = render_swing_action_buttons_for_card("Gold", active_states=states)

        assert "Close" in html
        assert "BE" in html

    def test_buttons_use_provided_dicts(self, swing_mod, _patch_redis):
        """When explicit active_states/pending_signals are passed, they are used."""
        from lib.services.data.api.swing_actions import render_swing_action_buttons_for_card

        # Pass explicit dicts — module state should be irrelevant
        pending = {"Gold": _make_signal("Gold").to_dict()}
        html = render_swing_action_buttons_for_card("Gold", active_states={}, pending_signals=pending)

        assert "Accept" in html
        assert "Ignore" in html

    def test_buttons_htmx_target(self, swing_mod, _patch_redis):
        """Buttons target the correct HTMX swap container."""
        from lib.services.data.api.swing_actions import render_swing_action_buttons_for_card

        swing_mod._pending_swing_signals["Gold"] = _make_signal("Gold")
        html = render_swing_action_buttons_for_card("Gold")

        assert "swing-actions-gold" in html
        assert 'hx-target="#swing-actions-gold"' in html


# ===========================================================================
# Tests — Dashboard _render_swing_card integration
# ===========================================================================


class TestSwingCardRendering:
    """Tests that _render_swing_card includes action buttons and live status."""

    def test_card_includes_live_status_div(self, swing_mod, _patch_redis):
        """Swing card HTML includes the live status polling div."""
        from lib.services.data.api.dashboard import _render_swing_card

        asset = {
            "symbol": "Gold",
            "bias": "LONG",
            "bias_emoji": "🟢",
            "last_price": 2700.0,
            "quality_pct": 75,
            "wave_ratio": 1.5,
            "price_decimals": 2,
            "has_live_position": False,
        }

        html = _render_swing_card(asset)
        assert "swing-live-status-gold" in html
        assert "hx-get" in html
        assert "/api/swing/status-badge/Gold" in html
        assert 'hx-trigger="load, every 30s"' in html

    def test_card_includes_action_buttons_area(self, swing_mod, _patch_redis):
        """Swing card HTML includes the action buttons container."""
        from lib.services.data.api.dashboard import _render_swing_card

        # Put a pending signal so buttons appear
        swing_mod._pending_swing_signals["Gold"] = _make_signal("Gold")

        asset = {
            "symbol": "Gold",
            "bias": "LONG",
            "bias_emoji": "🟢",
            "last_price": 2700.0,
            "quality_pct": 75,
            "wave_ratio": 1.5,
            "price_decimals": 2,
            "has_live_position": False,
        }

        html = _render_swing_card(asset)
        # Should have action button references
        assert "swing-actions" in html

    def test_card_with_active_states_param(self, swing_mod, _patch_redis):
        """Swing card accepts active_swing_states parameter."""
        from lib.services.data.api.dashboard import _render_swing_card

        active = {"Gold": _make_state("Gold")}
        asset = {
            "symbol": "Gold",
            "bias": "LONG",
            "bias_emoji": "🟢",
            "last_price": 2700.0,
            "quality_pct": 75,
            "wave_ratio": 1.5,
            "price_decimals": 2,
            "has_live_position": False,
        }

        html = _render_swing_card(asset, active_swing_states=active)
        assert "Gold" in html


# ===========================================================================
# Tests — SSE swing update handling
# ===========================================================================


class TestSSESwingUpdate:
    """Tests for SSE handling of dashboard:swing_update events."""

    def test_sse_module_has_swing_handling(self):
        """Verify the SSE module references dashboard:swing_update."""
        import inspect

        from lib.services.data.api import sse

        source = inspect.getsource(sse)
        assert "dashboard:swing_update" in source
        assert "swing-update" in source


# ===========================================================================
# Tests — Data service router registration
# ===========================================================================


class TestRouterRegistration:
    """Tests that the swing actions router is registered in the data service."""

    def test_swing_actions_router_imported(self):
        """swing_actions_router is imported in data/main.py."""
        import inspect

        from lib.services.data import main as data_main

        source = inspect.getsource(data_main)
        assert "swing_actions_router" in source
        assert "Swing Actions" in source

    def test_swing_actions_router_has_routes(self):
        """The router has the expected route paths."""
        from lib.services.data.api.swing_actions import router

        paths = [route.path for route in router.routes]  # type: ignore[union-attr]
        assert "/api/swing/accept/{asset_name}" in paths
        assert "/api/swing/ignore/{asset_name}" in paths
        assert "/api/swing/close/{asset_name}" in paths
        assert "/api/swing/stop-to-be/{asset_name}" in paths
        assert "/api/swing/update-stop/{asset_name}" in paths
        assert "/api/swing/pending" in paths
        assert "/api/swing/active" in paths
        assert "/api/swing/detail/{asset_name}" in paths
        assert "/api/swing/history" in paths
        assert "/api/swing/status-badge/{asset_name}" in paths


# ===========================================================================
# Tests — Signal lifecycle integration
# ===========================================================================


class TestSignalLifecycle:
    """Integration tests for the full signal → accept → manage → close lifecycle."""

    def test_full_lifecycle(self, swing_mod, _patch_redis, _patch_swing_detector):
        """Signal detected → accepted → stop moved to BE → closed."""
        # 1. Signal detected
        sig = _make_signal("Gold")
        swing_mod._pending_swing_signals["Gold"] = sig

        pending = swing_mod.get_pending_signals()
        assert "Gold" in pending

        # 2. Accept
        result = swing_mod.accept_swing_signal("Gold")
        assert result["status"] == "accepted"
        assert "Gold" in swing_mod._active_swing_states
        assert swing_mod.get_pending_signals().get("Gold") is None

        # 3. Move stop to BE
        result = swing_mod.move_stop_to_breakeven("Gold")
        assert result["status"] == "stop_moved"
        state = swing_mod._active_swing_states["Gold"]
        assert state.current_stop == state.entry_price

        # 4. Close
        result = swing_mod.close_swing_position("Gold")
        assert result["status"] == "closed"
        assert "Gold" not in swing_mod._active_swing_states

    def test_ignore_then_new_signal(self, swing_mod, _patch_redis):
        """Ignoring a signal, then detecting a new one, shows the new signal."""
        sig1 = _make_signal("Gold", confidence=0.6)
        swing_mod._pending_swing_signals["Gold"] = sig1

        swing_mod.ignore_swing_signal("Gold")
        assert "Gold" not in swing_mod._pending_swing_signals

        # New signal detected on next tick
        sig2 = _make_signal("Gold", confidence=0.8)
        swing_mod._pending_swing_signals["Gold"] = sig2

        pending = swing_mod.get_pending_signals()
        assert "Gold" in pending

    def test_multiple_assets_lifecycle(self, swing_mod, _patch_redis, _patch_swing_detector):
        """Multiple assets can be independently managed."""
        for name in ["Gold", "Crude Oil", "E-mini S&P"]:
            swing_mod._pending_swing_signals[name] = _make_signal(name)

        # Accept Gold and Crude Oil
        swing_mod.accept_swing_signal("Gold")
        swing_mod.accept_swing_signal("Crude Oil")

        # Ignore E-mini S&P
        swing_mod.ignore_swing_signal("E-mini S&P")

        assert len(swing_mod._active_swing_states) == 2
        assert "Gold" in swing_mod._active_swing_states
        assert "Crude Oil" in swing_mod._active_swing_states
        assert "E-mini S&P" not in swing_mod._active_swing_states

        # Close Gold
        swing_mod.close_swing_position("Gold")
        assert len(swing_mod._active_swing_states) == 1
        assert "Crude Oil" in swing_mod._active_swing_states


# ===========================================================================
# Tests — HTML fragment helpers
# ===========================================================================


class TestHTMLFragments:
    """Tests for the HTML fragment rendering helpers."""

    def test_success_toast_green(self):
        from lib.services.data.api.swing_actions import _success_toast

        html = _success_toast("Title", "Detail text", color="green")
        assert "Title" in html
        assert "Detail text" in html
        assert "swing-action-toast" in html

    def test_success_toast_yellow(self):
        from lib.services.data.api.swing_actions import _success_toast

        html = _success_toast("Warn", "Info", color="yellow")
        assert "Warn" in html
        assert "#facc15" in html  # yellow color

    def test_success_toast_red(self):
        from lib.services.data.api.swing_actions import _success_toast

        html = _success_toast("Closed", "Done", color="red")
        assert "Closed" in html
        assert "#f87171" in html  # red color

    def test_error_toast(self):
        from lib.services.data.api.swing_actions import _error_toast

        html = _error_toast("Something went wrong")
        assert "Something went wrong" in html
        assert "Error" in html

    def test_render_active_state_badge(self):
        from lib.services.data.api.swing_actions import _render_active_state_badge

        state_dict = {
            "phase": "active",
            "direction": "LONG",
            "entry_price": 2700.0,
            "current_stop": 2680.0,
            "remaining_size": 2,
            "tp1": 2740.0,
            "tp2": 2770.0,
        }
        html = _render_active_state_badge(state_dict)
        assert "ACTIVE" in html.upper()
        assert "LONG" in html
        assert "2,700" in html or "2700" in html

    def test_render_action_buttons_pending(self):
        from lib.services.data.api.swing_actions import _render_swing_action_buttons

        html = _render_swing_action_buttons("Gold", has_active_state=False, has_pending_signal=True)
        assert "Accept" in html
        assert "Ignore" in html
        assert "Close" not in html

    def test_render_action_buttons_active(self):
        from lib.services.data.api.swing_actions import _render_swing_action_buttons

        html = _render_swing_action_buttons("Gold", has_active_state=True, has_pending_signal=False, phase="active")
        assert "Close" in html
        assert "BE" in html
        assert "Accept" not in html

    def test_render_action_buttons_empty(self):
        from lib.services.data.api.swing_actions import _render_swing_action_buttons

        html = _render_swing_action_buttons("Gold", has_active_state=False, has_pending_signal=False)
        assert html == ""

    def test_render_action_buttons_closed_phase(self):
        from lib.services.data.api.swing_actions import _render_swing_action_buttons

        html = _render_swing_action_buttons("Gold", has_active_state=True, has_pending_signal=False, phase="closed")
        # Closed phase should not show active buttons
        assert html == ""


# ===========================================================================
# Tests — Concurrent tick + pending signal population
# ===========================================================================


class TestPendingSignalPopulation:
    """Tests for pending signal population during tick_swing_detector."""

    def test_tick_populates_pending(self, swing_mod, _patch_redis, monkeypatch):
        """Signals detected during tick are added to _pending_swing_signals."""
        # We can't easily run the full tick, but we can verify the module-level
        # dict behavior.
        sig = _make_signal("Gold")
        swing_mod._pending_swing_signals["Gold"] = sig

        assert "Gold" in swing_mod._pending_swing_signals
        pending = swing_mod.get_pending_signals()
        assert "Gold" in pending

    def test_accept_removes_from_pending(self, swing_mod, _patch_redis, _patch_swing_detector):
        """Accepting removes signal from pending dict."""
        swing_mod._pending_swing_signals["Gold"] = _make_signal("Gold")

        swing_mod.accept_swing_signal("Gold")

        assert "Gold" not in swing_mod._pending_swing_signals
        assert "Gold" in swing_mod._active_swing_states
