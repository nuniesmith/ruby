"""
Session-Aware Engine Scheduler — Full 24-Hour Globex Coverage
=============================================================
Manages the engine's behavior based on Eastern Time trading sessions.
The Globex futures day begins at 18:00 ET (after the 17:00–18:00
settlement break), so the scheduler is organised around that 24-hour
cycle rather than the calendar midnight.

DST handling
------------
All session boundary times are **ET wall-clock** (America/New_York).
Python's ZoneInfo handles EST↔EDT transitions automatically — no manual
UTC offsets needed.  In summer (EDT = UTC-4) all UTC equivalents shift
one hour earlier vs winter (EST = UTC-5).

Session windows (ET wall-clock, Globex-day order):
  - **Evening / Overnight (18:00–03:00 ET):**
      CME Globex open (18:00), Sydney/ASX (18:30), Tokyo/TSE (19:00),
      Shanghai/HK (21:00).  ORB checks fire every 2 min within each
      scan window.  All other actions sleep at the off-hours interval.
  - **Pre-market (00:00–03:00 ET):**
      Compute daily focus once, run Grok morning briefing, prepare alerts.
  - **Active (03:00–12:00 ET):**
      Frankfurt/Xetra (03:00), London Open (03:00–08:00),
      London-NY Crossover (08:00–10:00), US Equity Open (09:30–11:00).
      Live Ruby recomputation every 5 min, Grok updates every 15 min.
  - **Off-hours / Afternoon (12:00–18:00 ET):**
      CME Settlement ORB (14:00–15:30).  Historical backfill, optimisation,
      backtesting, CNN dataset generation + retraining.

The ScheduleManager is consumed by the engine main loop.  It tracks what
has already run within each session to avoid redundant work.

Usage:
    from lib.services.engine.scheduler import ScheduleManager

    mgr = ScheduleManager()
    while running:
        actions = mgr.get_pending_actions()
        for action in actions:
            run(action)
        mgr.mark_done(action)
        sleep(mgr.sleep_interval)
"""

import logging
import time
from dataclasses import dataclass
from datetime import date, datetime
from enum import StrEnum
from typing import Any
from zoneinfo import ZoneInfo

logger = logging.getLogger("engine.scheduler")

_EST = ZoneInfo("America/New_York")


class SessionMode(StrEnum):
    EVENING = "evening"  # 18:00–00:00 ET — overnight ORB sessions
    PRE_MARKET = "pre-market"  # 00:00–03:00 ET — daily focus + briefing
    ACTIVE = "active"  # 03:00–12:00 ET — London + US live trading
    OFF_HOURS = "off-hours"  # 12:00–18:00 ET — backfill, CNN, backtest


class ActionType(StrEnum):
    """All schedulable engine actions."""

    # Pre-market actions (run once per day, 00:00–03:00 ET)
    COMPUTE_DAILY_FOCUS = "compute_daily_focus"
    GROK_MORNING_BRIEF = "grok_morning_brief"
    PREP_ALERTS = "prep_alerts"

    # Active session actions (recurring, 03:00–12:00 ET)
    RUBY_RECOMPUTE = "fks_recompute"
    PUBLISH_FOCUS_UPDATE = "publish_focus_update"
    GROK_LIVE_UPDATE = "grok_live_update"
    CHECK_RISK_RULES = "check_risk_rules"
    CHECK_NO_TRADE = "check_no_trade"

    # ── ORB session checks — fired every 2 min within scan windows ──────────
    # US Equity Open  09:30–11:00 ET
    CHECK_ORB = "check_orb"
    # Frankfurt / Xetra Open  03:00–04:30 ET  (08:00 CET / 09:00 CEST)
    CHECK_ORB_FRANKFURT = "check_orb_frankfurt"
    # London Open  03:00–05:00 ET  (primary session)
    CHECK_ORB_LONDON = "check_orb_london"
    # London–NY Crossover  08:00–10:00 ET
    CHECK_ORB_LONDON_NY = "check_orb_london_ny"

    # Evening / overnight sessions (18:00–03:00 ET, wraps_midnight)
    # CME Globex re-open  18:00–20:00 ET
    CHECK_ORB_CME = "check_orb_cme"
    # Sydney / ASX Open  18:30–20:30 ET
    CHECK_ORB_SYDNEY = "check_orb_sydney"
    # Tokyo / TSE Open  19:00–21:00 ET
    CHECK_ORB_TOKYO = "check_orb_tokyo"
    # Shanghai / HK Open  21:00–23:00 ET
    CHECK_ORB_SHANGHAI = "check_orb_shanghai"

    # CME Settlement  14:00–15:30 ET  (metals/energy settlement window)
    CHECK_ORB_CME_SETTLE = "check_orb_cme_settle"

    # ── Crypto-specific ORB sessions (only active when ENABLE_KRAKEN_CRYPTO=1) ──
    # UTC 00:00 crypto session  19:00–21:00 ET EST / 20:00–22:00 ET EDT
    # High-volume Asia open window for BTC/ETH/SOL/etc.
    CHECK_ORB_CRYPTO_UTC0 = "check_orb_crypto_utc0"
    # UTC 12:00 crypto session  07:00–09:00 ET EST / 08:00–10:00 ET EDT
    # London morning crypto window; pre-US-open positioning.
    CHECK_ORB_CRYPTO_UTC12 = "check_orb_crypto_utc12"

    # ── Multi-breakout-type checks — PDR, IB, Consolidation + 9 researched ──
    # PDR (Previous Day Range) — active whenever markets are open
    # Runs every 2 min during London open, LN-NY cross, and US open.
    CHECK_PDR = "check_pdr"
    # IB (Initial Balance) — only relevant during / after 09:30–10:30 ET RTH
    CHECK_IB = "check_ib"
    # Consolidation / Squeeze — scans for BB contraction + expansion bar
    # relevant throughout the full active window.
    CHECK_CONSOLIDATION = "check_consolidation"
    # Parallel multi-type sweep: runs multiple breakout types in one shot
    # for a given session's asset list.  Fired by session-specific handlers.
    # Payload carries {"session_key": "<key>", "types": ["PDR", "CONS", ...]}.
    CHECK_BREAKOUT_MULTI = "check_breakout_multi"

    # ── New researched breakout-type checks (9 additional) ──────────────────
    # These run via CHECK_BREAKOUT_MULTI payloads in most cases, but
    # dedicated action types exist for types that need specific scheduling
    # windows or independent cadences.
    CHECK_WEEKLY = "check_weekly"  # Prior week H/L — once at session open
    CHECK_MONTHLY = "check_monthly"  # Prior month H/L — once at session open
    CHECK_ASIAN = "check_asian"  # Asian range — after 02:00 ET
    CHECK_BBSQUEEZE = "check_bbsqueeze"  # BB inside KC squeeze — all active
    CHECK_VA = "check_va"  # Value Area VAH/VAL — all active
    CHECK_INSIDE = "check_inside"  # Inside Day — after session open
    CHECK_GAP = "check_gap"  # Gap Rejection — at session open
    CHECK_PIVOT = "check_pivot"  # Pivot Points S1/R1 — all active
    CHECK_FIB = "check_fib"  # Fibonacci retracement — all active

    # Off-hours actions (12:00–18:00 ET, run once per session)
    HISTORICAL_BACKFILL = "historical_backfill"
    RUN_OPTIMIZATION = "run_optimization"
    RUN_BACKTEST = "run_backtest"
    NEXT_DAY_PREP = "next_day_prep"

    # CNN dataset & training actions (off-hours, run once per session)
    GENERATE_CHART_DATASET = "generate_chart_dataset"
    TRAIN_BREAKOUT_CNN = "train_breakout_cnn"

    # Daily report — runs once per day at end of active session (~12:00 ET)
    DAILY_REPORT = "daily_report"

    # EOD position management — hard 4:00 PM ET stop
    # POSITION_CLOSE_WARNING fires at 15:45 ET — 15-minute alert that EOD
    # flat requirement is approaching.  Publishes a dashboard alert and
    # sends a Grok notification.  Runs once per day.
    POSITION_CLOSE_WARNING = "position_close_warning"
    # EOD_POSITION_CLOSE fires at exactly 16:00 ET.  Calls
    # cancel_all_orders() then exit_position() on every connected Rithmic
    # account that has open positions.  Runs once per day.
    # This is a hard safety net — the trader is expected to be flat before
    # this fires.  Rithmic OrderPlacement.MANUAL is used so the audit trail
    # shows a human-initiated close, not an algo.
    EOD_POSITION_CLOSE = "eod_position_close"

    # Swing detector — runs every 2 min during active session (03:00–15:30 ET)
    # Scans daily-plan swing candidates for pullback/breakout/gap entries,
    # manages SwingState per asset, publishes signals + states to Redis.
    CHECK_SWING = "check_swing"

    # News sentiment pipeline — Finnhub + Alpha Vantage + VADER + Grok hybrid.
    # Runs twice per day:
    #   07:00 ET (pre-market, inside PRE_MARKET window near its end) — morning
    #   12:00 ET (midday, first tick of OFF_HOURS) — midday refresh
    # Results cached in Redis: engine:news_sentiment:<SYMBOL> (2h TTL).
    CHECK_NEWS_SENTIMENT = "check_news_sentiment"

    # News sentiment midday run sentinel — used so the off-hours scheduler
    # can fire a second CHECK_NEWS_SENTIMENT without the "ran today" guard
    # blocking it (we track pre-market and midday separately).
    CHECK_NEWS_SENTIMENT_MIDDAY = "check_news_sentiment_midday"


@dataclass
class ScheduledAction:
    """A single action the engine should execute."""

    action: ActionType
    priority: int = 0  # lower = higher priority
    description: str = ""
    # Optional payload passed to the handler.  Used by CHECK_BREAKOUT_MULTI
    # and CHECK_PDR / CHECK_IB / CHECK_CONSOLIDATION to convey which session
    # key and which BreakoutType subset to run without needing separate
    # ActionType variants per session.
    payload: dict[str, Any] | None = None


@dataclass
class _ActionTracker:
    """Tracks when an action was last executed and whether it's been
    completed for the current session/day."""

    last_run: float | None = None  # timestamp
    last_run_date: date | None = None  # for once-per-day actions
    last_run_session: str | None = None  # for once-per-session actions
    run_count_today: int = 0


class ScheduleManager:
    """Session-aware scheduler for engine actions.

    Determines which actions need to run based on the current ET time,
    what has already been completed, and configured intervals.

    The scheduler covers the full 24-hour Globex day starting at 18:00 ET:
      18:00–00:00 ET  EVENING   — overnight ORB checks (CME/Sydney/Tokyo/Shanghai)
      00:00–03:00 ET  PRE_MARKET — daily focus, Grok briefing, alert prep
      03:00–12:00 ET  ACTIVE    — Frankfurt/London/LN-NY/US ORB + live recompute
      12:00–18:00 ET  OFF_HOURS — CME settlement ORB, backfill, CNN, backtest

    Thread-safe: all state is read/written from a single engine thread.
    """

    # Recurring interval configuration (seconds) — all ORB/breakout checks = 2 min
    RUBY_INTERVAL = 5 * 60  # 5 min during active
    GROK_INTERVAL = 15 * 60  # 15 min during active
    RISK_CHECK_INTERVAL = 60  # 1 min during active
    NO_TRADE_INTERVAL = 2 * 60  # 2 min during active
    ORB_CHECK_INTERVAL = 2 * 60  # US open          09:30–11:00 ET
    ORB_FRANKFURT_CHECK_INTERVAL = 2 * 60  # Frankfurt/Xetra  03:00–04:30 ET
    ORB_LONDON_CHECK_INTERVAL = 2 * 60  # London open      03:00–05:00 ET
    ORB_LONDON_NY_CHECK_INTERVAL = 2 * 60  # London-NY cross  08:00–10:00 ET
    ORB_CME_CHECK_INTERVAL = 2 * 60  # CME Globex open  18:00–20:00 ET
    ORB_SYDNEY_CHECK_INTERVAL = 2 * 60  # Sydney / ASX     18:30–20:30 ET
    ORB_TOKYO_CHECK_INTERVAL = 2 * 60  # Tokyo / TSE      19:00–21:00 ET
    ORB_SHANGHAI_CHECK_INTERVAL = 2 * 60  # Shanghai / HK    21:00–23:00 ET
    ORB_CME_SETTLE_CHECK_INTERVAL = 2 * 60  # CME settlement   14:00–15:30 ET
    ORB_CRYPTO_UTC0_CHECK_INTERVAL = 2 * 60  # Crypto UTC 00:00  19:00–21:00 ET
    ORB_CRYPTO_UTC12_CHECK_INTERVAL = 2 * 60  # Crypto UTC 12:00  07:00–09:00 ET
    # Multi-breakout-type checks fire on the same 2-min cadence.
    PDR_CHECK_INTERVAL = 2 * 60  # PDR scan  — all active windows
    IB_CHECK_INTERVAL = 2 * 60  # IB scan   — 10:30–12:00 ET only
    CONS_CHECK_INTERVAL = 2 * 60  # Squeeze   — all active windows
    BREAKOUT_MULTI_CHECK_INTERVAL = 2 * 60  # Parallel multi-type sweep
    # New researched breakout type intervals (all 2 min)
    WEEKLY_CHECK_INTERVAL = 2 * 60  # Weekly H/L
    MONTHLY_CHECK_INTERVAL = 2 * 60  # Monthly H/L
    ASIAN_CHECK_INTERVAL = 2 * 60  # Asian range breakout
    BBSQUEEZE_CHECK_INTERVAL = 2 * 60  # Bollinger squeeze
    VA_CHECK_INTERVAL = 2 * 60  # Value Area
    INSIDE_CHECK_INTERVAL = 2 * 60  # Inside Day
    GAP_CHECK_INTERVAL = 2 * 60  # Gap Rejection
    PIVOT_CHECK_INTERVAL = 2 * 60  # Pivot Points
    FIB_CHECK_INTERVAL = 2 * 60  # Fibonacci retracement
    FOCUS_PUBLISH_INTERVAL = 30  # 30 s during active (throttled downstream)
    STATUS_PUBLISH_INTERVAL = 10  # 10 s always
    SWING_CHECK_INTERVAL = 2 * 60  # 2 min — swing detector scan cadence

    # Sleep intervals per session
    SLEEP_EVENING = 30.0  # check every 30 s during evening overnight ORBs
    SLEEP_PRE_MARKET = 30.0  # check every 30 s during pre-market
    SLEEP_ACTIVE = 10.0  # check every 10 s during active hours
    SLEEP_OFF_HOURS = 60.0  # check every 60 s during off-hours

    def __init__(self) -> None:
        self._trackers: dict[ActionType, _ActionTracker] = {action: _ActionTracker() for action in ActionType}
        self._current_session: SessionMode | None = None
        self._session_started_at: float | None = None
        self._today: date | None = None
        self._session_transition_logged: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @staticmethod
    def get_session_mode(now: datetime | None = None) -> SessionMode:
        """Determine current trading session based on ET wall-clock time.

        Globex-day boundaries (all ET, DST-aware via ZoneInfo):
          - Evening    18:00–00:00 ET  overnight ORB windows (CME/Sydney/Tokyo/Shanghai)
          - Pre-market 00:00–03:00 ET  daily focus + Grok briefing
          - Active     03:00–12:00 ET  Frankfurt/London/LN-NY/US live trading
          - Off-hours  12:00–18:00 ET  CNN training, backfill, backtest, CME settle ORB
        """
        if now is None:
            now = datetime.now(tz=_EST)
        hour = now.hour
        if 18 <= hour <= 23:
            return SessionMode.EVENING
        elif 0 <= hour < 3:
            return SessionMode.PRE_MARKET
        elif 3 <= hour < 12:
            return SessionMode.ACTIVE
        else:
            return SessionMode.OFF_HOURS

    @property
    def current_session(self) -> SessionMode:
        """Return the current session mode (cached, updated each cycle)."""
        return self._current_session or self.get_session_mode()

    @property
    def sleep_interval(self) -> float:
        """How long the main loop should sleep between scheduler cycles."""
        session = self.current_session
        if session == SessionMode.EVENING:
            return self.SLEEP_EVENING
        elif session == SessionMode.PRE_MARKET:
            return self.SLEEP_PRE_MARKET
        elif session == SessionMode.ACTIVE:
            return self.SLEEP_ACTIVE
        else:
            return self.SLEEP_OFF_HOURS

    def get_pending_actions(
        self,
        now: datetime | None = None,
    ) -> list[ScheduledAction]:
        """Return ordered list of actions that should run right now.

        Call this each engine loop iteration.  It handles:
          - Session transitions (resets per-session trackers)
          - Day transitions (resets per-day trackers)
          - Interval-based recurring actions
          - Once-per-session and once-per-day actions

        Returns actions sorted by priority (lowest number = highest priority).
        """
        if now is None:
            now = datetime.now(tz=_EST)
        ts = time.monotonic()

        session = self.get_session_mode(now)
        today = now.date()

        # Handle day transition
        if self._today != today:
            self._on_day_change(today)

        # Handle session transition
        if self._current_session != session:
            self._on_session_change(session, now)

        self._current_session = session

        # Gather pending actions based on session
        pending: list[ScheduledAction] = []

        if session == SessionMode.EVENING:
            pending.extend(self._get_evening_actions(ts, now))
        elif session == SessionMode.PRE_MARKET:
            pending.extend(self._get_pre_market_actions(ts, today, now=now))
        elif session == SessionMode.ACTIVE:
            pending.extend(self._get_active_actions(ts, now))
        else:
            pending.extend(self._get_off_hours_actions(ts, today=today))

        # Sort by priority
        pending.sort(key=lambda a: a.priority)
        return pending

    def mark_done(self, action: ActionType, now: datetime | None = None) -> None:
        """Mark an action as completed. Called after successful execution.

        Parameters
        ----------
        action : ActionType
            The action that completed.
        now : datetime, optional
            Override the current time (used by tests).  When *None* the
            real wall-clock time is used.
        """
        if now is None:
            now = datetime.now(tz=_EST)
        tracker = self._trackers[action]
        tracker.last_run = time.monotonic()
        tracker.last_run_date = now.date()
        tracker.last_run_session = self._current_session.value if self._current_session else None
        tracker.run_count_today += 1
        logger.debug(
            "Action completed: %s (run #%d today)",
            action.value,
            tracker.run_count_today,
        )

    def mark_failed(self, action: ActionType, error: str) -> None:
        """Mark an action as failed. It will be retried on the next cycle."""
        logger.warning("Action failed: %s — %s", action.value, error)
        # Don't update last_run so it gets retried

    def get_status(self, now: datetime | None = None) -> dict:
        """Return scheduler status for health/monitoring.

        Parameters
        ----------
        now : datetime, optional
            Override the current time (used by tests).
        """
        if now is None:
            now = datetime.now(tz=_EST)
        session = self.get_session_mode(now)

        action_statuses = {}
        for action, tracker in self._trackers.items():
            action_statuses[action.value] = {
                "last_run": tracker.last_run_date.isoformat() if tracker.last_run_date else None,
                "run_count_today": tracker.run_count_today,
                "last_session": tracker.last_run_session,
            }

        return {
            "session_mode": session.value,
            "session_emoji": self._session_emoji(session),
            "current_time_et": now.strftime("%H:%M:%S"),
            "sleep_interval": self.sleep_interval,
            "actions": action_statuses,
        }

    def time_until_next_session(self, now: datetime | None = None) -> tuple[SessionMode, float]:
        """Return the next session and seconds until it starts.

        Globex-day order: EVENING (18:00) → PRE_MARKET (00:00) → ACTIVE (03:00)
                          → OFF_HOURS (12:00) → EVENING (18:00)
        """
        if now is None:
            now = datetime.now(tz=_EST)
        hour = now.hour

        if 18 <= hour <= 23:
            # In evening, next is pre-market at 00:00 (midnight)
            next_session = SessionMode.PRE_MARKET
            target_hour = 24
        elif 0 <= hour < 3:
            # In pre-market, next is active at 03:00
            next_session = SessionMode.ACTIVE
            target_hour = 3
        elif 3 <= hour < 12:
            # In active, next is off-hours at 12:00
            next_session = SessionMode.OFF_HOURS
            target_hour = 12
        else:
            # In off-hours (12:00–18:00), next is evening at 18:00
            next_session = SessionMode.EVENING
            target_hour = 18

        # Calculate seconds until target hour
        effective_hour = hour if hour < 24 else 0
        seconds_remaining = (target_hour - effective_hour - 1) * 3600
        seconds_remaining += (60 - now.minute - 1) * 60
        seconds_remaining += 60 - now.second
        # Clamp to non-negative
        seconds_remaining = max(0, seconds_remaining)

        return next_session, seconds_remaining

    # ------------------------------------------------------------------
    # Internal: per-session action generators
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Internal: per-session action generators
    # ------------------------------------------------------------------

    def _get_evening_actions(
        self,
        ts: float,
        now: datetime,
    ) -> list[ScheduledAction]:
        """Evening / overnight (18:00–00:00 ET): overnight ORB checks only.

        Fires CME Globex open, Sydney/ASX, Tokyo/TSE, and Shanghai/HK ORB
        checks every 2 minutes within their respective scan windows.
        Also fires the crypto UTC-midnight session check (19:00–21:00 ET)
        when ENABLE_KRAKEN_CRYPTO is active.
        All other costly actions (backfill, CNN, etc.) are deferred to
        off-hours (12:00–18:00 ET) so they don't compete with live data.
        """
        actions: list[ScheduledAction] = []
        now_time = now.time()

        from datetime import time as _dt_time

        # Detect whether crypto ORB sessions are enabled (lazy import to avoid
        # startup failures when the Kraken integration is not installed).
        _crypto_enabled = False
        try:
            from lib.core.models import ENABLE_KRAKEN_CRYPTO as _ekc

            _crypto_enabled = bool(_ekc)
        except Exception:
            pass

        # --- CME Globex Re-Open ORB — 18:00–20:00 ET ---
        if _dt_time(18, 0) <= now_time <= _dt_time(20, 0) and self._interval_elapsed(
            ActionType.CHECK_ORB_CME, ts, self.ORB_CME_CHECK_INTERVAL
        ):
            actions.append(
                ScheduledAction(
                    action=ActionType.CHECK_ORB_CME,
                    priority=1,
                    description="Check CME Globex Open ORB (18:00–18:30 ET opening range)",
                )
            )

        # --- Sydney / ASX Open ORB — 18:30–20:30 ET ---
        if _dt_time(18, 30) <= now_time <= _dt_time(20, 30) and self._interval_elapsed(
            ActionType.CHECK_ORB_SYDNEY, ts, self.ORB_SYDNEY_CHECK_INTERVAL
        ):
            actions.append(
                ScheduledAction(
                    action=ActionType.CHECK_ORB_SYDNEY,
                    priority=2,
                    description="Check Sydney/ASX Open ORB (18:30–19:00 ET opening range)",
                )
            )

        # --- Tokyo / TSE Open ORB — 19:00–21:00 ET ---
        if _dt_time(19, 0) <= now_time <= _dt_time(21, 0) and self._interval_elapsed(
            ActionType.CHECK_ORB_TOKYO, ts, self.ORB_TOKYO_CHECK_INTERVAL
        ):
            actions.append(
                ScheduledAction(
                    action=ActionType.CHECK_ORB_TOKYO,
                    priority=2,
                    description="Check Tokyo/TSE Open ORB (19:00–19:30 ET opening range)",
                )
            )

        # --- Shanghai / HK Open ORB — 21:00–23:00 ET ---
        if _dt_time(21, 0) <= now_time <= _dt_time(23, 0) and self._interval_elapsed(
            ActionType.CHECK_ORB_SHANGHAI, ts, self.ORB_SHANGHAI_CHECK_INTERVAL
        ):
            actions.append(
                ScheduledAction(
                    action=ActionType.CHECK_ORB_SHANGHAI,
                    priority=2,
                    description="Check Shanghai/HK Open ORB (21:00–21:30 ET opening range)",
                )
            )

        # --- Crypto UTC 00:00 ORB — 19:00–21:00 ET (EST) / 20:00–22:00 ET (EDT) ---
        # Uses crypto_utc0 session (wraps_midnight=True); only scans KRAKEN:* tickers.
        # We check for 19:00 ET here; ZoneInfo handles the EDT offset automatically
        # so the check fires at the correct wall-clock ET time year-round.
        if (
            _crypto_enabled
            and _dt_time(19, 0) <= now_time <= _dt_time(21, 0)
            and self._interval_elapsed(ActionType.CHECK_ORB_CRYPTO_UTC0, ts, self.ORB_CRYPTO_UTC0_CHECK_INTERVAL)
        ):
            actions.append(
                ScheduledAction(
                    action=ActionType.CHECK_ORB_CRYPTO_UTC0,
                    priority=3,
                    description="Check Crypto UTC-midnight ORB (19:00–19:30 ET / 00:00 UTC window)",
                    payload={"session_key": "crypto_utc0"},
                )
            )

        return actions

    def _get_pre_market_actions(
        self,
        ts: float,
        today: date,
        now: datetime | None = None,
    ) -> list[ScheduledAction]:
        """Pre-market (00:00–03:00 ET): focus computation + morning prep."""
        actions: list[ScheduledAction] = []

        # Daily focus — run once per day
        if not self._ran_today(ActionType.COMPUTE_DAILY_FOCUS, today):
            actions.append(
                ScheduledAction(
                    action=ActionType.COMPUTE_DAILY_FOCUS,
                    priority=0,
                    description="Compute daily focus for today's trading plan",
                )
            )

        # Grok morning briefing — run once per day
        if not self._ran_today(ActionType.GROK_MORNING_BRIEF, today):
            actions.append(
                ScheduledAction(
                    action=ActionType.GROK_MORNING_BRIEF,
                    priority=1,
                    description="Generate Grok AI morning market briefing",
                )
            )

        # News sentiment — morning run at 07:00 ET (once per day, inside PRE_MARKET)
        # Fires when wall-clock time is ≥07:00 ET and hasn't run today.
        from datetime import time as _pm_time2

        now_time_news = now.time() if now is not None else datetime.now(tz=_EST).time()
        if now_time_news >= _pm_time2(7, 0) and not self._ran_today(ActionType.CHECK_NEWS_SENTIMENT, today):
            actions.append(
                ScheduledAction(
                    action=ActionType.CHECK_NEWS_SENTIMENT,
                    priority=3,
                    description="Run news sentiment pipeline (morning — Finnhub + AV + Grok)",
                )
            )

        # Prep alerts — run once per day
        if not self._ran_today(ActionType.PREP_ALERTS, today):
            actions.append(
                ScheduledAction(
                    action=ActionType.PREP_ALERTS,
                    priority=2,
                    description="Prepare alert thresholds for active session",
                )
            )

        # --- Crypto UTC 12:00 ORB — 07:00–09:00 ET (EST) / 08:00–10:00 ET (EDT) ---
        # London morning crypto window; high-volume pre-US-open positioning.
        # Detect crypto enabled flag here too (shared logic with evening actions).
        _crypto_enabled_pm = False
        try:
            from lib.core.models import ENABLE_KRAKEN_CRYPTO as _ekc_pm

            _crypto_enabled_pm = bool(_ekc_pm)
        except Exception:
            pass

        from datetime import time as _pm_time

        now_time_pm = now.time() if now is not None else datetime.now(tz=_EST).time()
        if (
            _crypto_enabled_pm
            and _pm_time(7, 0) <= now_time_pm <= _pm_time(9, 0)
            and self._interval_elapsed(ActionType.CHECK_ORB_CRYPTO_UTC12, ts, self.ORB_CRYPTO_UTC12_CHECK_INTERVAL)
        ):
            actions.append(
                ScheduledAction(
                    action=ActionType.CHECK_ORB_CRYPTO_UTC12,
                    priority=5,
                    description="Check Crypto UTC-noon ORB (07:00–07:30 ET / 12:00 UTC window)",
                    payload={"session_key": "crypto_utc12"},
                )
            )

        return actions

    def _get_active_actions(
        self,
        ts: float,
        now: datetime,
    ) -> list[ScheduledAction]:
        """Active (03:00–12:00 ET): live recomputation + ORB + multi-type breakout checks.

        Sub-sessions within the active window (ET wall-clock):
          - Frankfurt/Xetra  03:00–04:30 ET  (08:00 CET / 09:00 CEST)
          - London Open      03:00–05:00 ET  (primary ORB session)
          - London-NY Cross  08:00–10:00 ET
          - US Equity Open   09:30–11:00 ET

        In addition to per-session ORB checks, each window fires:
          - CHECK_PDR        — Previous Day Range breakout scan
          - CHECK_CONSOLIDATION — Bollinger squeeze expansion scan
          - CHECK_IB         — Initial Balance breakout (10:30 ET onwards)
          - CHECK_BREAKOUT_MULTI — parallel PDR+IB+CONS sweep for the session

        All multi-type checks use the same 2-minute cadence as ORB and carry
        a ``payload={"session_key": "<key>"}`` so the handler knows which
        session-asset list to use.
        """
        actions: list[ScheduledAction] = []
        today = now.date()
        now_time = now.time()

        from datetime import time as _dt_time

        # Daily focus — also compute at start of active if missed in pre-market
        if not self._ran_today(ActionType.COMPUTE_DAILY_FOCUS, today):
            actions.append(
                ScheduledAction(
                    action=ActionType.COMPUTE_DAILY_FOCUS,
                    priority=0,
                    description="Compute daily focus (catch-up — missed pre-market)",
                )
            )

        # Ruby recomputation — every 5 minutes
        if self._interval_elapsed(ActionType.RUBY_RECOMPUTE, ts, self.RUBY_INTERVAL):
            actions.append(
                ScheduledAction(
                    action=ActionType.RUBY_RECOMPUTE,
                    priority=1,
                    description="Recompute Ruby wave/vol/quality for all assets",
                )
            )

        # Publish focus update to Redis — every 30 seconds
        if self._interval_elapsed(ActionType.PUBLISH_FOCUS_UPDATE, ts, self.FOCUS_PUBLISH_INTERVAL):
            actions.append(
                ScheduledAction(
                    action=ActionType.PUBLISH_FOCUS_UPDATE,
                    priority=2,
                    description="Publish focus update to Redis for SSE",
                )
            )

        # Risk rules check — every 1 minute
        if self._interval_elapsed(ActionType.CHECK_RISK_RULES, ts, self.RISK_CHECK_INTERVAL):
            actions.append(
                ScheduledAction(
                    action=ActionType.CHECK_RISK_RULES,
                    priority=3,
                    description="Check risk rules (position limits, daily loss, time)",
                )
            )

        # No-trade detector — every 2 minutes
        if self._interval_elapsed(ActionType.CHECK_NO_TRADE, ts, self.NO_TRADE_INTERVAL):
            actions.append(
                ScheduledAction(
                    action=ActionType.CHECK_NO_TRADE,
                    priority=4,
                    description="Check should-not-trade conditions",
                )
            )

        # Grok live update — every 15 minutes
        if self._interval_elapsed(ActionType.GROK_LIVE_UPDATE, ts, self.GROK_INTERVAL):
            actions.append(
                ScheduledAction(
                    action=ActionType.GROK_LIVE_UPDATE,
                    priority=5,
                    description="Run Grok 15-minute live market update",
                )
            )

        # ── Swing detector — every 2 min during active hours (03:00–15:30 ET) ──
        # Scans daily-plan swing candidates for pullback, breakout, and gap-
        # continuation entries.  Manages per-asset SwingState (TP/SL/trail)
        # and publishes signals + states to Redis for dashboard display.
        # Stops scanning after 15:30 ET (swing time-stop — no new entries).
        if _dt_time(3, 0) <= now_time <= _dt_time(15, 30) and self._interval_elapsed(
            ActionType.CHECK_SWING, ts, self.SWING_CHECK_INTERVAL
        ):
            actions.append(
                ScheduledAction(
                    action=ActionType.CHECK_SWING,
                    priority=5,
                    description="Swing detector: scan candidates for entry/exit signals",
                )
            )

        # --- Frankfurt / Xetra Open ORB — every 2 min during 03:00–04:30 ET ---
        # Xetra opens 08:00 CET (= 03:00 EST / 02:00 EDT) — fires at same ET
        # time as London open, but uses frankfurt session asset list.
        if _dt_time(3, 0) <= now_time <= _dt_time(4, 30) and self._interval_elapsed(
            ActionType.CHECK_ORB_FRANKFURT, ts, self.ORB_FRANKFURT_CHECK_INTERVAL
        ):
            actions.append(
                ScheduledAction(
                    action=ActionType.CHECK_ORB_FRANKFURT,
                    priority=6,
                    description="Check Frankfurt/Xetra Open ORB (03:00–03:30 ET opening range)",
                )
            )

        # --- London Open ORB — every 2 min during 03:00–05:00 ET ---
        if _dt_time(3, 0) <= now_time <= _dt_time(5, 0) and self._interval_elapsed(
            ActionType.CHECK_ORB_LONDON, ts, self.ORB_LONDON_CHECK_INTERVAL
        ):
            actions.append(
                ScheduledAction(
                    action=ActionType.CHECK_ORB_LONDON,
                    priority=6,
                    description="Check London Open ORB (03:00–03:30 ET opening range)",
                )
            )

        # --- London-NY Crossover ORB — every 2 min during 08:00–10:00 ET ---
        if _dt_time(8, 0) <= now_time <= _dt_time(10, 0) and self._interval_elapsed(
            ActionType.CHECK_ORB_LONDON_NY, ts, self.ORB_LONDON_NY_CHECK_INTERVAL
        ):
            actions.append(
                ScheduledAction(
                    action=ActionType.CHECK_ORB_LONDON_NY,
                    priority=6,
                    description="Check London-NY Crossover ORB (08:00–08:30 ET opening range)",
                )
            )

        # --- US Equity Open ORB — every 2 min during 09:30–11:00 ET ---
        if _dt_time(9, 30) <= now_time <= _dt_time(11, 0) and self._interval_elapsed(
            ActionType.CHECK_ORB, ts, self.ORB_CHECK_INTERVAL
        ):
            actions.append(
                ScheduledAction(
                    action=ActionType.CHECK_ORB,
                    priority=6,
                    description="Check US Equity Open ORB (09:30–10:00 ET OR)",
                )
            )

        # ── Multi-BreakoutType parallel checks ────────────────────────────────
        # All 13 breakout types are checked during their relevant windows.
        # PDR, WEEKLY, MONTHLY, VA, PIVOT, FIB, INSIDE, GAP use pre-computed
        # ranges (prior session/day/week data).  CONS, BBSQUEEZE detect live
        # compression.  ASIAN fires after the Asian window closes (02:00 ET).
        #
        # Strategy:
        #   - HTF types (WEEKLY, MONTHLY, PIVOT, FIB) scan all active windows
        #   - Session-derived types (PDR, VA, INSIDE, GAP) scan all active
        #   - Squeeze types (CONS, BBSQUEEZE) scan all active (detect live)
        #   - ASIAN fires only after 02:00 ET when range is complete
        #   - IB fires only after 10:30 ET

        # -- Frankfurt/London window (03:00–05:00 ET) -- all types except IB --
        if _dt_time(3, 0) <= now_time <= _dt_time(5, 0) and self._interval_elapsed(
            ActionType.CHECK_BREAKOUT_MULTI, ts, self.BREAKOUT_MULTI_CHECK_INTERVAL
        ):
            types_london = [
                "PDR",
                "CONS",
                "WEEKLY",
                "MONTHLY",
                "BBSQUEEZE",
                "VA",
                "INSIDE",
                "GAP",
                "PIVOT",
                "FIB",
            ]
            # Asian range is complete by 02:00 ET, so include it in London window
            if now_time >= _dt_time(3, 0):
                types_london.append("ASIAN")
            actions.append(
                ScheduledAction(
                    action=ActionType.CHECK_BREAKOUT_MULTI,
                    priority=7,
                    description="Multi-type breakout sweep: all 13 types (London session assets)",
                    payload={"session_key": "london", "types": types_london},
                )
            )

        # -- PDR during London-NY crossover (08:00–10:00 ET) --
        if _dt_time(8, 0) <= now_time <= _dt_time(10, 0) and self._interval_elapsed(
            ActionType.CHECK_PDR, ts, self.PDR_CHECK_INTERVAL
        ):
            actions.append(
                ScheduledAction(
                    action=ActionType.CHECK_PDR,
                    priority=7,
                    description="PDR breakout scan (London-NY crossover assets)",
                    payload={"session_key": "london_ny"},
                )
            )

        # -- CONS (squeeze) during London-NY crossover (08:00–10:00 ET) --
        if _dt_time(8, 0) <= now_time <= _dt_time(10, 0) and self._interval_elapsed(
            ActionType.CHECK_CONSOLIDATION, ts, self.CONS_CHECK_INTERVAL
        ):
            actions.append(
                ScheduledAction(
                    action=ActionType.CHECK_CONSOLIDATION,
                    priority=7,
                    description="Consolidation/squeeze breakout scan (London-NY assets)",
                    payload={"session_key": "london_ny"},
                )
            )

        # -- HTF + researched types during London-NY crossover (08:00–10:00 ET) --
        if _dt_time(8, 0) <= now_time <= _dt_time(10, 0) and self._interval_elapsed(
            ActionType.CHECK_WEEKLY, ts, self.WEEKLY_CHECK_INTERVAL
        ):
            actions.append(
                ScheduledAction(
                    action=ActionType.CHECK_BREAKOUT_MULTI,
                    priority=8,
                    description="HTF + researched breakout sweep (London-NY assets)",
                    payload={
                        "session_key": "london_ny",
                        "types": [
                            "WEEKLY",
                            "MONTHLY",
                            "ASIAN",
                            "BBSQUEEZE",
                            "VA",
                            "INSIDE",
                            "GAP",
                            "PIVOT",
                            "FIB",
                        ],
                    },
                )
            )

        # -- IB breakout — only valid after 60-min IB window closes (10:30 ET) --
        if _dt_time(10, 30) <= now_time <= _dt_time(12, 0) and self._interval_elapsed(
            ActionType.CHECK_IB, ts, self.IB_CHECK_INTERVAL
        ):
            actions.append(
                ScheduledAction(
                    action=ActionType.CHECK_IB,
                    priority=7,
                    description="Initial Balance breakout scan (US session assets)",
                    payload={"session_key": "us"},
                )
            )

        # -- Full multi-type sweep during US Equity Open (09:30–11:00 ET) --
        # ORB is handled separately above; run all other types in parallel.
        if _dt_time(9, 30) <= now_time <= _dt_time(11, 0) and self._interval_elapsed(
            ActionType.CHECK_BREAKOUT_MULTI, ts, self.BREAKOUT_MULTI_CHECK_INTERVAL
        ):
            actions.append(
                ScheduledAction(
                    action=ActionType.CHECK_BREAKOUT_MULTI,
                    priority=7,
                    description="Multi-type breakout sweep: all types (US session assets)",
                    payload={
                        "session_key": "us",
                        "types": [
                            "PDR",
                            "IB",
                            "CONS",
                            "WEEKLY",
                            "MONTHLY",
                            "ASIAN",
                            "BBSQUEEZE",
                            "VA",
                            "INSIDE",
                            "GAP",
                            "PIVOT",
                            "FIB",
                        ],
                    },
                )
            )

        # ── EOD 15:45 ET warning — once per day ──────────────────────────────
        # Fires a dashboard alert and Grok notification 15 minutes before the
        # hard 4 PM close so the trader can flatten manually if they choose.
        if now_time >= _dt_time(15, 45) and not self._ran_today(ActionType.POSITION_CLOSE_WARNING, today):
            actions.append(
                ScheduledAction(
                    action=ActionType.POSITION_CLOSE_WARNING,
                    priority=0,  # highest priority — safety critical
                    description="EOD 15-min warning: 4:00 PM hard close approaching (15:45 ET)",
                )
            )

        # ── EOD 16:00 ET hard position close — once per day ──────────────────
        # cancel_all_orders() + exit_position() on every Rithmic account with
        # an open position.  This is the last-resort safety net.  Fires as soon
        # as the scheduler detects 16:00 ET has been reached.
        if now_time >= _dt_time(16, 0) and not self._ran_today(ActionType.EOD_POSITION_CLOSE, today):
            actions.append(
                ScheduledAction(
                    action=ActionType.EOD_POSITION_CLOSE,
                    priority=0,  # highest priority — safety critical
                    description="EOD hard close: cancel all orders + exit all positions (16:00 ET)",
                )
            )

        return actions

    def _get_off_hours_actions(self, ts: float, today: date | None = None) -> list[ScheduledAction]:
        """Off-hours (12:00–18:00 ET): CME settlement ORB, daily report, backfill, CNN training.

        The overnight ORB windows (CME Globex open, Sydney, Tokyo, Shanghai)
        are now handled by _get_evening_actions() (18:00–00:00 ET).
        This window handles the daytime settlement ORB, PDR/CONS scans during
        the settlement window, and all off-hours batch tasks.
        """
        actions: list[ScheduledAction] = []
        session = SessionMode.OFF_HOURS.value
        if today is None:
            today = datetime.now(tz=_EST).date()

        now = datetime.now(tz=_EST)
        now_time = now.time()

        from datetime import time as _dt_time

        # ── EOD 16:00 ET hard position close — catch-up in off-hours ─────────
        # If the scheduler was not running at exactly 16:00 ET (e.g. service
        # restarted at 16:05), fire the hard close as soon as off-hours begins.
        # The _ran_today guard prevents double-firing.
        if not self._ran_today(ActionType.EOD_POSITION_CLOSE, today):
            actions.append(
                ScheduledAction(
                    action=ActionType.EOD_POSITION_CLOSE,
                    priority=0,
                    description="EOD hard close (catch-up): cancel all orders + exit all positions",
                )
            )

        # --- CME Settlement ORB — every 2 min during 14:00–15:30 ET ---
        # Metals and energy settlement window; directional resolution before close.
        if _dt_time(14, 0) <= now_time <= _dt_time(15, 30) and self._interval_elapsed(
            ActionType.CHECK_ORB_CME_SETTLE, ts, self.ORB_CME_SETTLE_CHECK_INTERVAL
        ):
            actions.append(
                ScheduledAction(
                    action=ActionType.CHECK_ORB_CME_SETTLE,
                    priority=2,
                    description="Check CME Settlement ORB (14:00–14:30 ET metals/energy window)",
                )
            )

        # Daily report — once per day, first thing when off-hours begins.
        # Summarises the just-completed trading session: ORB signals, CNN stats,
        # filter rates, risk events. Publishes to Redis + optional email alert.
        if not self._ran_today(ActionType.DAILY_REPORT, today):
            actions.append(
                ScheduledAction(
                    action=ActionType.DAILY_REPORT,
                    priority=0,
                    description="Generate and publish daily trading session report",
                )
            )

        # News sentiment midday refresh — once per off-hours session (12:00 ET)
        if not self._ran_this_session(ActionType.CHECK_NEWS_SENTIMENT_MIDDAY, session):
            actions.append(
                ScheduledAction(
                    action=ActionType.CHECK_NEWS_SENTIMENT_MIDDAY,
                    priority=1,
                    description="Run news sentiment pipeline (midday refresh — Finnhub + AV + Grok)",
                )
            )

        # Historical backfill — once per off-hours session
        if not self._ran_this_session(ActionType.HISTORICAL_BACKFILL, session):
            actions.append(
                ScheduledAction(
                    action=ActionType.HISTORICAL_BACKFILL,
                    priority=2,
                    description="Backfill historical 1-min bars to Postgres",
                )
            )

        # Optimization — once per off-hours session
        if not self._ran_this_session(ActionType.RUN_OPTIMIZATION, session):
            actions.append(
                ScheduledAction(
                    action=ActionType.RUN_OPTIMIZATION,
                    priority=2,
                    description="Run Optuna strategy optimization",
                )
            )

        # Backtesting — once per off-hours session
        if not self._ran_this_session(ActionType.RUN_BACKTEST, session):
            actions.append(
                ScheduledAction(
                    action=ActionType.RUN_BACKTEST,
                    priority=3,
                    description="Run walk-forward backtesting",
                )
            )

        # Next-day prep — once per off-hours session
        if not self._ran_this_session(ActionType.NEXT_DAY_PREP, session):
            actions.append(
                ScheduledAction(
                    action=ActionType.NEXT_DAY_PREP,
                    priority=4,
                    description="Prepare next trading day parameters",
                )
            )

        # CNN chart dataset generation — once per off-hours session
        # Generates Ruby-style chart images + auto-labels from historical bars
        # for training the breakout pattern recognition CNN.
        if not self._ran_this_session(ActionType.GENERATE_CHART_DATASET, session):
            actions.append(
                ScheduledAction(
                    action=ActionType.GENERATE_CHART_DATASET,
                    priority=5,
                    description="Generate labeled chart dataset for CNN training",
                )
            )

        # CNN model training — once per off-hours session, after dataset generation
        # Only schedule if dataset generation already ran this session.
        if self._ran_this_session(ActionType.GENERATE_CHART_DATASET, session) and not self._ran_this_session(
            ActionType.TRAIN_BREAKOUT_CNN, session
        ):
            actions.append(
                ScheduledAction(
                    action=ActionType.TRAIN_BREAKOUT_CNN,
                    priority=6,
                    description="Train/retrain EfficientNetV2 breakout CNN on latest dataset",
                )
            )

        return actions

    # ------------------------------------------------------------------
    # Internal: helpers
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Internal: helpers
    # ------------------------------------------------------------------

    def _ran_today(self, action: ActionType, today: date) -> bool:
        """Check if action has already run today."""
        return self._trackers[action].last_run_date == today

    def _ran_this_session(self, action: ActionType, session_value: str, today: date | None = None) -> bool:
        """Check if action has already run during this session instance."""
        tracker = self._trackers[action]
        if tracker.last_run_session != session_value:
            return False
        # Also ensure it ran today (not a stale session marker from yesterday)
        if today is None:
            today = datetime.now(tz=_EST).date()
        return tracker.last_run_date == today

    def _interval_elapsed(
        self,
        action: ActionType,
        now_ts: float,
        interval_seconds: float,
    ) -> bool:
        """Check if enough time has passed since the last run."""
        tracker = self._trackers[action]
        if tracker.last_run is None:
            return True  # never run → due immediately
        return (now_ts - tracker.last_run) >= interval_seconds

    def _on_day_change(self, today: date) -> None:
        """Reset daily counters on day transition."""
        logger.info("=" * 50)
        logger.info("  Day change detected: %s", today.isoformat())
        logger.info("=" * 50)
        self._today = today
        for tracker in self._trackers.values():
            tracker.run_count_today = 0

    def _on_session_change(self, new_session: SessionMode, now: datetime) -> None:
        """Handle session transition."""
        old = self._current_session
        logger.info("=" * 50)
        logger.info(
            "  Session transition: %s → %s %s at %s ET",
            old.value if old else "INIT",
            new_session.value,
            self._session_emoji(new_session),
            now.strftime("%H:%M:%S"),
        )
        logger.info("=" * 50)
        self._session_started_at = time.monotonic()
        self._session_transition_logged = True

    @staticmethod
    def _session_emoji(session: SessionMode) -> str:
        return {
            SessionMode.EVENING: "🌃",
            SessionMode.PRE_MARKET: "🌙",
            SessionMode.ACTIVE: "🟢",
            SessionMode.OFF_HOURS: "⚙️",
        }.get(session, "❓")
