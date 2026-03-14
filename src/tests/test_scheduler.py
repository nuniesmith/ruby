"""
Tests for the Session-Aware Engine Scheduler.

Covers:
  - SessionMode detection by hour (pre-market, active, off-hours)
  - ScheduleManager.get_pending_actions() returns correct actions per session
  - Once-per-day actions don't repeat after mark_done()
  - Interval-based actions respect cooldown periods
  - Session transitions reset appropriate trackers
  - Day transitions reset daily counters
  - sleep_interval varies by session
  - time_until_next_session calculations
"""

from datetime import datetime
from zoneinfo import ZoneInfo

from lib.services.engine.scheduler import (
    ActionType,
    ScheduleManager,
    SessionMode,
)

_EST = ZoneInfo("America/New_York")


# ---------------------------------------------------------------------------
# SessionMode detection
# ---------------------------------------------------------------------------


class TestSessionMode:
    """Test that get_session_mode returns the right mode for each hour."""

    def test_midnight_is_pre_market(self):
        now = datetime(2026, 2, 27, 0, 0, 0, tzinfo=_EST)
        assert ScheduleManager.get_session_mode(now) == SessionMode.PRE_MARKET

    def test_1am_is_pre_market(self):
        now = datetime(2026, 2, 27, 1, 0, 0, tzinfo=_EST)
        assert ScheduleManager.get_session_mode(now) == SessionMode.PRE_MARKET

    def test_2_59am_is_pre_market(self):
        now = datetime(2026, 2, 27, 2, 59, 59, tzinfo=_EST)
        assert ScheduleManager.get_session_mode(now) == SessionMode.PRE_MARKET

    def test_3am_is_active(self):
        now = datetime(2026, 2, 27, 3, 0, 0, tzinfo=_EST)
        assert ScheduleManager.get_session_mode(now) == SessionMode.ACTIVE

    def test_9_30am_is_active(self):
        now = datetime(2026, 2, 27, 9, 30, 0, tzinfo=_EST)
        assert ScheduleManager.get_session_mode(now) == SessionMode.ACTIVE

    def test_11_59am_is_active(self):
        now = datetime(2026, 2, 27, 11, 59, 59, tzinfo=_EST)
        assert ScheduleManager.get_session_mode(now) == SessionMode.ACTIVE

    def test_noon_is_off_hours(self):
        now = datetime(2026, 2, 27, 12, 0, 0, tzinfo=_EST)
        assert ScheduleManager.get_session_mode(now) == SessionMode.OFF_HOURS

    def test_6pm_is_evening(self):
        now = datetime(2026, 2, 27, 18, 0, 0, tzinfo=_EST)
        assert ScheduleManager.get_session_mode(now) == SessionMode.EVENING

    def test_11pm_is_evening(self):
        now = datetime(2026, 2, 27, 23, 0, 0, tzinfo=_EST)
        assert ScheduleManager.get_session_mode(now) == SessionMode.EVENING


# ---------------------------------------------------------------------------
# Pre-market actions
# ---------------------------------------------------------------------------


class TestPreMarketActions:
    """Test that pre-market session returns the right once-per-day actions."""

    def test_first_call_returns_all_premarket_actions(self):
        mgr = ScheduleManager()
        now = datetime(2026, 2, 27, 1, 0, 0, tzinfo=_EST)
        pending = mgr.get_pending_actions(now=now)

        action_types = {a.action for a in pending}
        assert ActionType.COMPUTE_DAILY_FOCUS in action_types
        assert ActionType.GROK_MORNING_BRIEF in action_types
        assert ActionType.PREP_ALERTS in action_types

    def test_no_active_actions_during_premarket(self):
        mgr = ScheduleManager()
        now = datetime(2026, 2, 27, 1, 0, 0, tzinfo=_EST)
        pending = mgr.get_pending_actions(now=now)

        action_types = {a.action for a in pending}
        assert ActionType.RUBY_RECOMPUTE not in action_types
        assert ActionType.GROK_LIVE_UPDATE not in action_types

    def test_no_off_hours_actions_during_premarket(self):
        mgr = ScheduleManager()
        now = datetime(2026, 2, 27, 1, 0, 0, tzinfo=_EST)
        pending = mgr.get_pending_actions(now=now)

        action_types = {a.action for a in pending}
        assert ActionType.HISTORICAL_BACKFILL not in action_types
        assert ActionType.RUN_OPTIMIZATION not in action_types
        assert ActionType.RUN_BACKTEST not in action_types

    def test_premarket_actions_dont_repeat_after_done(self):
        mgr = ScheduleManager()
        now = datetime(2026, 2, 27, 1, 0, 0, tzinfo=_EST)

        # First call should include daily focus
        pending1 = mgr.get_pending_actions(now=now)
        assert any(a.action == ActionType.COMPUTE_DAILY_FOCUS for a in pending1)

        # Mark it done (pass now= so the recorded date matches the test date)
        mgr.mark_done(ActionType.COMPUTE_DAILY_FOCUS, now=now)

        # Second call should NOT include daily focus (already ran today)
        now2 = datetime(2026, 2, 27, 1, 5, 0, tzinfo=_EST)
        pending2 = mgr.get_pending_actions(now=now2)
        assert not any(a.action == ActionType.COMPUTE_DAILY_FOCUS for a in pending2)


# ---------------------------------------------------------------------------
# Active session actions
# ---------------------------------------------------------------------------


class TestActiveActions:
    def test_active_session_includes_recurring_actions(self):
        mgr = ScheduleManager()
        now = datetime(2026, 2, 27, 8, 0, 0, tzinfo=_EST)
        pending = mgr.get_pending_actions(now=now)

        action_types = {a.action for a in pending}
        # All interval-based actions should be pending on first call
        assert ActionType.RUBY_RECOMPUTE in action_types
        assert ActionType.PUBLISH_FOCUS_UPDATE in action_types
        assert ActionType.CHECK_RISK_RULES in action_types
        assert ActionType.CHECK_NO_TRADE in action_types
        assert ActionType.GROK_LIVE_UPDATE in action_types

    def test_active_catches_up_daily_focus_if_missed(self):
        """If daily focus wasn't computed during pre-market, active should compute it."""
        mgr = ScheduleManager()
        now = datetime(2026, 2, 27, 6, 0, 0, tzinfo=_EST)
        pending = mgr.get_pending_actions(now=now)

        action_types = {a.action for a in pending}
        assert ActionType.COMPUTE_DAILY_FOCUS in action_types

    def test_interval_actions_respect_cooldown(self):
        mgr = ScheduleManager()
        now = datetime(2026, 2, 27, 8, 0, 0, tzinfo=_EST)

        # First call — everything is pending
        pending1 = mgr.get_pending_actions(now=now)
        assert any(a.action == ActionType.RUBY_RECOMPUTE for a in pending1)

        # Mark done
        mgr.mark_done(ActionType.RUBY_RECOMPUTE)

        # Immediately call again — should NOT be pending (5 min cooldown)
        now2 = datetime(2026, 2, 27, 8, 0, 5, tzinfo=_EST)
        pending2 = mgr.get_pending_actions(now=now2)
        assert not any(a.action == ActionType.RUBY_RECOMPUTE for a in pending2)

    def test_no_off_hours_during_active(self):
        mgr = ScheduleManager()
        now = datetime(2026, 2, 27, 9, 0, 0, tzinfo=_EST)
        pending = mgr.get_pending_actions(now=now)

        action_types = {a.action for a in pending}
        assert ActionType.HISTORICAL_BACKFILL not in action_types
        assert ActionType.RUN_OPTIMIZATION not in action_types


# ---------------------------------------------------------------------------
# Off-hours actions
# ---------------------------------------------------------------------------


class TestOffHoursActions:
    def test_off_hours_includes_background_tasks(self):
        mgr = ScheduleManager()
        now = datetime(2026, 2, 27, 14, 0, 0, tzinfo=_EST)
        pending = mgr.get_pending_actions(now=now)

        action_types = {a.action for a in pending}
        assert ActionType.HISTORICAL_BACKFILL in action_types
        assert ActionType.RUN_OPTIMIZATION in action_types
        assert ActionType.RUN_BACKTEST in action_types
        assert ActionType.NEXT_DAY_PREP in action_types

    def test_off_hours_excludes_active_actions(self):
        mgr = ScheduleManager()
        now = datetime(2026, 2, 27, 14, 0, 0, tzinfo=_EST)
        pending = mgr.get_pending_actions(now=now)

        action_types = {a.action for a in pending}
        assert ActionType.RUBY_RECOMPUTE not in action_types
        assert ActionType.GROK_LIVE_UPDATE not in action_types

    def test_off_hours_once_per_session(self):
        mgr = ScheduleManager()
        now = datetime(2026, 2, 27, 14, 0, 0, tzinfo=_EST)

        pending1 = mgr.get_pending_actions(now=now)
        assert any(a.action == ActionType.HISTORICAL_BACKFILL for a in pending1)

        mgr.mark_done(ActionType.HISTORICAL_BACKFILL)

        now2 = datetime(2026, 2, 27, 15, 0, 0, tzinfo=_EST)
        pending2 = mgr.get_pending_actions(now=now2)
        assert not any(a.action == ActionType.HISTORICAL_BACKFILL for a in pending2)


# ---------------------------------------------------------------------------
# Session transitions
# ---------------------------------------------------------------------------


class TestSessionTransitions:
    def test_premarket_to_active_transition(self):
        mgr = ScheduleManager()
        # Start in pre-market
        now1 = datetime(2026, 2, 27, 2, 0, 0, tzinfo=_EST)
        mgr.get_pending_actions(now=now1)
        assert mgr.current_session == SessionMode.PRE_MARKET

        # Transition to active (03:00 ET — London open)
        now2 = datetime(2026, 2, 27, 3, 0, 0, tzinfo=_EST)
        pending = mgr.get_pending_actions(now=now2)
        assert mgr.current_session == SessionMode.ACTIVE

        # Should now see active-session actions
        action_types = {a.action for a in pending}
        assert ActionType.RUBY_RECOMPUTE in action_types

    def test_active_to_off_hours_transition(self):
        mgr = ScheduleManager()
        # Start in active
        now1 = datetime(2026, 2, 27, 11, 0, 0, tzinfo=_EST)
        mgr.get_pending_actions(now=now1)
        assert mgr.current_session == SessionMode.ACTIVE

        # Transition to off-hours
        now2 = datetime(2026, 2, 27, 12, 0, 0, tzinfo=_EST)
        pending = mgr.get_pending_actions(now=now2)
        assert mgr.current_session == SessionMode.OFF_HOURS

        action_types = {a.action for a in pending}
        assert ActionType.HISTORICAL_BACKFILL in action_types


# ---------------------------------------------------------------------------
# Day transitions
# ---------------------------------------------------------------------------


class TestDayTransitions:
    def test_day_change_resets_daily_counters(self):
        mgr = ScheduleManager()

        # Day 1 pre-market — compute daily focus
        now1 = datetime(2026, 2, 27, 1, 0, 0, tzinfo=_EST)
        pending1 = mgr.get_pending_actions(now=now1)
        assert any(a.action == ActionType.COMPUTE_DAILY_FOCUS for a in pending1)
        mgr.mark_done(ActionType.COMPUTE_DAILY_FOCUS, now=now1)

        # Still Day 1 — should NOT be pending
        now2 = datetime(2026, 2, 27, 2, 0, 0, tzinfo=_EST)
        pending2 = mgr.get_pending_actions(now=now2)
        assert not any(a.action == ActionType.COMPUTE_DAILY_FOCUS for a in pending2)

        # Day 2 pre-market — should be pending again
        now3 = datetime(2026, 2, 28, 1, 0, 0, tzinfo=_EST)
        pending3 = mgr.get_pending_actions(now=now3)
        assert any(a.action == ActionType.COMPUTE_DAILY_FOCUS for a in pending3)


# ---------------------------------------------------------------------------
# Sleep intervals
# ---------------------------------------------------------------------------


class TestSleepInterval:
    def test_premarket_sleep(self):
        mgr = ScheduleManager()
        now = datetime(2026, 2, 27, 1, 0, 0, tzinfo=_EST)
        mgr.get_pending_actions(now=now)
        assert mgr.sleep_interval == ScheduleManager.SLEEP_PRE_MARKET

    def test_active_sleep(self):
        mgr = ScheduleManager()
        now = datetime(2026, 2, 27, 8, 0, 0, tzinfo=_EST)
        mgr.get_pending_actions(now=now)
        assert mgr.sleep_interval == ScheduleManager.SLEEP_ACTIVE

    def test_off_hours_sleep(self):
        mgr = ScheduleManager()
        now = datetime(2026, 2, 27, 14, 0, 0, tzinfo=_EST)
        mgr.get_pending_actions(now=now)
        assert mgr.sleep_interval == ScheduleManager.SLEEP_OFF_HOURS


# ---------------------------------------------------------------------------
# Priority ordering
# ---------------------------------------------------------------------------


class TestPriorityOrdering:
    def test_actions_sorted_by_priority(self):
        mgr = ScheduleManager()
        now = datetime(2026, 2, 27, 3, 0, 0, tzinfo=_EST)
        pending = mgr.get_pending_actions(now=now)

        # Verify monotonically non-decreasing priority
        priorities = [a.priority for a in pending]
        for i in range(1, len(priorities)):
            assert priorities[i] >= priorities[i - 1], (
                f"Action at index {i} (priority {priorities[i]}) is out of order "
                f"after index {i - 1} (priority {priorities[i - 1]})"
            )

    def test_daily_focus_is_highest_priority(self):
        """COMPUTE_DAILY_FOCUS should always be priority 0 (highest)."""
        mgr = ScheduleManager()
        now = datetime(2026, 2, 27, 3, 0, 0, tzinfo=_EST)
        pending = mgr.get_pending_actions(now=now)

        focus_actions = [a for a in pending if a.action == ActionType.COMPUTE_DAILY_FOCUS]
        assert len(focus_actions) == 1
        assert focus_actions[0].priority == 0


# ---------------------------------------------------------------------------
# get_status
# ---------------------------------------------------------------------------


class TestGetStatus:
    def test_status_structure(self):
        mgr = ScheduleManager()
        now = datetime(2026, 2, 27, 8, 0, 0, tzinfo=_EST)
        mgr.get_pending_actions(now=now)
        status = mgr.get_status(now=now)

        assert "session_mode" in status
        assert "session_emoji" in status
        assert "current_time_et" in status
        assert "sleep_interval" in status
        assert "actions" in status
        assert status["session_mode"] == "active"

    def test_status_reflects_completed_actions(self):  # noqa: E501
        mgr = ScheduleManager()
        now = datetime(2026, 2, 27, 3, 0, 0, tzinfo=_EST)
        mgr.get_pending_actions(now=now)
        mgr.mark_done(ActionType.COMPUTE_DAILY_FOCUS)

        status = mgr.get_status()
        focus_status = status["actions"]["compute_daily_focus"]
        assert focus_status["run_count_today"] == 1


# ---------------------------------------------------------------------------
# time_until_next_session
# ---------------------------------------------------------------------------


class TestTimeUntilNextSession:
    def test_premarket_to_active(self):
        mgr = ScheduleManager()
        # 01:00 AM is in PRE_MARKET (00:00–03:00), next session is ACTIVE at 03:00
        now = datetime(2026, 2, 27, 1, 0, 0, tzinfo=_EST)
        next_session, seconds = mgr.time_until_next_session(now)
        assert next_session == SessionMode.ACTIVE
        # 1:00 AM → 3:00 AM = 7200 seconds
        assert 7100 < seconds <= 7200

    def test_active_to_off_hours(self):
        mgr = ScheduleManager()
        now = datetime(2026, 2, 27, 11, 0, 0, tzinfo=_EST)
        next_session, seconds = mgr.time_until_next_session(now)
        assert next_session == SessionMode.OFF_HOURS
        # 11:00 AM → 12:00 PM = 3600 seconds
        assert 3500 < seconds <= 3600

    def test_off_hours_to_premarket(self):
        mgr = ScheduleManager()
        now = datetime(2026, 2, 27, 23, 0, 0, tzinfo=_EST)
        next_session, seconds = mgr.time_until_next_session(now)
        assert next_session == SessionMode.PRE_MARKET
        # 23:00 → 00:00 = 3600 seconds
        assert 3500 < seconds <= 3600


# ---------------------------------------------------------------------------
# mark_failed does not prevent retry
# ---------------------------------------------------------------------------


class TestMarkFailed:
    def test_failed_action_can_retry(self):
        mgr = ScheduleManager()
        now = datetime(2026, 2, 27, 3, 0, 0, tzinfo=_EST)
        pending1 = mgr.get_pending_actions(now=now)
        assert any(a.action == ActionType.COMPUTE_DAILY_FOCUS for a in pending1)

        # Mark as failed (not done)
        mgr.mark_failed(ActionType.COMPUTE_DAILY_FOCUS, "test error")

        # Should still be pending on next call
        now2 = datetime(2026, 2, 27, 3, 1, 0, tzinfo=_EST)
        pending2 = mgr.get_pending_actions(now=now2)
        assert any(a.action == ActionType.COMPUTE_DAILY_FOCUS for a in pending2)


# ---------------------------------------------------------------------------
# DAILY_REPORT action
# ---------------------------------------------------------------------------


class TestDailyReportAction:
    def test_daily_report_present_in_off_hours(self):
        """DAILY_REPORT should appear when off-hours begins."""
        mgr = ScheduleManager()
        now = datetime(2026, 2, 27, 12, 0, 0, tzinfo=_EST)
        pending = mgr.get_pending_actions(now=now)

        action_types = {a.action for a in pending}
        assert ActionType.DAILY_REPORT in action_types

    def test_daily_report_not_present_in_active_session(self):
        """DAILY_REPORT must never fire during active trading hours."""
        mgr = ScheduleManager()
        now = datetime(2026, 2, 27, 9, 30, 0, tzinfo=_EST)
        pending = mgr.get_pending_actions(now=now)

        action_types = {a.action for a in pending}
        assert ActionType.DAILY_REPORT not in action_types

    def test_daily_report_not_present_in_pre_market(self):
        """DAILY_REPORT must not fire during pre-market."""
        mgr = ScheduleManager()
        now = datetime(2026, 2, 27, 1, 0, 0, tzinfo=_EST)
        pending = mgr.get_pending_actions(now=now)

        action_types = {a.action for a in pending}
        assert ActionType.DAILY_REPORT not in action_types

    def test_daily_report_is_highest_priority_off_hours(self):
        """DAILY_REPORT should be priority 0 in off-hours (fires before backfill/retrain)."""
        mgr = ScheduleManager()
        now = datetime(2026, 2, 27, 12, 0, 0, tzinfo=_EST)
        pending = mgr.get_pending_actions(now=now)

        report_actions = [a for a in pending if a.action == ActionType.DAILY_REPORT]
        assert len(report_actions) == 1
        assert report_actions[0].priority == 0

    def test_daily_report_runs_once_per_day(self):
        """DAILY_REPORT must not re-queue after mark_done()."""
        mgr = ScheduleManager()
        now = datetime(2026, 2, 27, 12, 0, 0, tzinfo=_EST)

        pending1 = mgr.get_pending_actions(now=now)
        assert any(a.action == ActionType.DAILY_REPORT for a in pending1)

        mgr.mark_done(ActionType.DAILY_REPORT, now=now)

        now2 = datetime(2026, 2, 27, 18, 0, 0, tzinfo=_EST)
        pending2 = mgr.get_pending_actions(now=now2)
        assert not any(a.action == ActionType.DAILY_REPORT for a in pending2)

    def test_daily_report_requeues_next_day(self):
        """DAILY_REPORT should reappear at the start of the next day's off-hours."""
        mgr = ScheduleManager()
        now = datetime(2026, 2, 27, 12, 0, 0, tzinfo=_EST)
        mgr.get_pending_actions(now=now)
        mgr.mark_done(ActionType.DAILY_REPORT, now=now)

        # Next day's off-hours
        tomorrow = datetime(2026, 2, 28, 12, 0, 0, tzinfo=_EST)
        pending = mgr.get_pending_actions(now=tomorrow)
        assert any(a.action == ActionType.DAILY_REPORT for a in pending)

    def test_off_hours_priority_ordering_with_daily_report(self):
        """All off-hours actions must be sorted monotonically by priority."""
        mgr = ScheduleManager()
        now = datetime(2026, 2, 27, 12, 0, 0, tzinfo=_EST)
        pending = mgr.get_pending_actions(now=now)

        priorities = [a.priority for a in pending]
        for i in range(1, len(priorities)):
            assert priorities[i] >= priorities[i - 1], (
                f"Off-hours action at index {i} (priority {priorities[i]}) "
                f"is out of order after index {i - 1} (priority {priorities[i - 1]})"
            )


# ---------------------------------------------------------------------------
# SessionMode enum values
# ---------------------------------------------------------------------------


class TestSessionModeEnum:
    def test_enum_string_values(self):
        assert SessionMode.PRE_MARKET.value == "pre-market"
        assert SessionMode.ACTIVE.value == "active"
        assert SessionMode.OFF_HOURS.value == "off-hours"

    def test_enum_is_str_subclass(self):
        """SessionMode should be usable as a string."""
        assert isinstance(SessionMode.ACTIVE, str)
        assert SessionMode.ACTIVE == "active"
