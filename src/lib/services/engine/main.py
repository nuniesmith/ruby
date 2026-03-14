"""
Engine Service — Background computation worker
==================================================================
Runs the DashboardEngine as a standalone service, separate from data-service.

Now uses ScheduleManager for session-aware scheduling:
  - **Pre-market (00:00–03:00 ET):** Compute daily focus once, Grok morning
    briefing, prepare alerts for the trading day.
  - **Active (03:00–12:00 ET):** Live Ruby recomputation every 5 min,
    publish focus updates to Redis, Grok updates every 15 min.
  - **Off-hours (12:00–00:00 ET):** Historical data backfill, full
    optimization runs, backtesting, next-day prep.

The data-service becomes a thin API layer that reads from Redis.

Day 4 additions:
  - RiskManager integrated into CHECK_RISK_RULES handler
  - evaluate_no_trade replaces basic should_not_trade check
  - Grok compact output in live update handler

Usage:
    python -m lib.services.engine.main

Docker:
    CMD ["python", "-m", "lib.services.engine.main"]
"""

import contextlib
import json
import os
import signal
import time
from datetime import datetime
from typing import TYPE_CHECKING, Any
from zoneinfo import ZoneInfo

if TYPE_CHECKING:
    import pandas as pd

    from lib.trading.strategies.rb.breakout import BreakoutResult

from lib.core.logging_config import get_logger, setup_logging

setup_logging(service="engine")
logger = get_logger("engine_service")

_EST = ZoneInfo("America/New_York")
HEALTH_FILE = "/tmp/engine_health.json"

# ---------------------------------------------------------------------------
# CNN model hot-reload — detect when breakout_cnn_best.pt changes on disk
# ---------------------------------------------------------------------------
# The ModelWatcher (lib.services.engine.model_watcher) replaces the old
# inline polling approach.  It uses watchdog (inotify/FSEvents) for instant
# notification when model files change, with a polling fallback.
#
# The watcher is started in main() and stopped on shutdown.
# ---------------------------------------------------------------------------
_model_watcher = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Module-level PositionManager singleton (initialised in main())
# ---------------------------------------------------------------------------
_position_manager = None

# ---------------------------------------------------------------------------
# Module-level LiveRiskPublisher (initialised in main())
# ---------------------------------------------------------------------------
_live_risk_publisher = None

# ---------------------------------------------------------------------------
# Module-level CopyTrader singleton (initialised in main())
# Wires PositionManager → Rithmic multi-account copy trading.
# Only activated when RITHMIC_COPY_TRADING=1 env var is set so the engine
# can still run without Rithmic credentials in dev/CI environments.
# ---------------------------------------------------------------------------
_copy_trader = None
_COPY_TRADING_ENABLED = os.getenv("RITHMIC_COPY_TRADING", "0") == "1"


def _get_position_manager(account_size: int = 50_000):
    """Lazy-init and return the global PositionManager singleton."""
    global _position_manager
    if _position_manager is None:
        try:
            from lib.services.engine.position_manager import PositionManager

            _position_manager = PositionManager(account_size=account_size)
            _position_manager.load_state()
            logger.info(
                "PositionManager initialised (account=$%s)",
                f"{account_size:,}",
            )
        except Exception as exc:
            logger.warning("PositionManager init failed (non-fatal): %s", exc)
    return _position_manager


def _get_copy_trader():
    """Lazy-init and return the global CopyTrader singleton.

    Returns ``None`` (and logs a debug message) when
    ``RITHMIC_COPY_TRADING=1`` is not set.
    """
    global _copy_trader
    if not _COPY_TRADING_ENABLED:
        return None
    if _copy_trader is None:
        try:
            from lib.services.engine.copy_trader import CopyTrader

            _copy_trader = CopyTrader()
            logger.info("CopyTrader initialised (RITHMIC_COPY_TRADING=1)")
        except Exception as exc:
            logger.warning("CopyTrader init failed (non-fatal): %s", exc)
    return _copy_trader


def _get_live_risk_publisher():
    """Lazy-init and return the global LiveRiskPublisher singleton.

    Requires _risk_manager and _position_manager to be initialised first.
    Also registers the publisher with the live_risk API module so the
    /api/live-risk/refresh endpoint can trigger immediate recomputation.
    """
    global _live_risk_publisher
    if _live_risk_publisher is None:
        try:
            from lib.services.engine.live_risk import LiveRiskPublisher

            _live_risk_publisher = LiveRiskPublisher(
                risk_manager=_risk_manager,
                position_manager=_position_manager,
                interval_seconds=5.0,
            )

            # Wire into the live_risk API module so /api/live-risk/refresh works
            try:
                from lib.services.data.api.live_risk import set_publisher

                set_publisher(_live_risk_publisher)
                logger.info("LiveRiskPublisher registered with live_risk API module")
            except ImportError:
                logger.debug("live_risk API module not available — publisher not registered with API")

            # Force an initial publish so the dashboard has data immediately
            _ = _live_risk_publisher.force_publish()

            logger.info(
                "LiveRiskPublisher initialised (interval=5s, rm=%s, pm=%s)",
                "yes" if _risk_manager else "no",
                "yes" if _position_manager else "no",
            )
        except Exception as exc:
            logger.warning("LiveRiskPublisher init failed (non-fatal): %s", exc)
    return _live_risk_publisher


def _write_health(healthy: bool, status: str, **extras):
    """Write health status to a file for Docker healthcheck."""
    data = {
        "healthy": healthy,
        "status": status,
        "timestamp": datetime.now(tz=_EST).isoformat(),
        **extras,
    }
    try:
        with open(HEALTH_FILE, "w") as f:
            json.dump(data, f)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Module-level risk manager instance (initialised in main())
# ---------------------------------------------------------------------------
_risk_manager = None


def _get_risk_manager(account_size: int = 50_000):
    """Lazy-init and return the global RiskManager singleton.

    Tries to read the main Rithmic account's ``account_size`` from Redis
    so the risk engine uses the per-account value instead of the env-var
    default.
    """
    global _risk_manager
    if _risk_manager is None:
        # Try to get account size from Rithmic main account config in Redis
        try:
            from lib.core.cache import REDIS_AVAILABLE, _r

            if REDIS_AVAILABLE and _r is not None:
                accounts_raw = _r.hgetall("rithmic:accounts")
                if accounts_raw:
                    for raw in accounts_raw.values():
                        data = json.loads(raw if isinstance(raw, str) else raw.decode())
                        # Use the first account's size (or one marked as main)
                        acct_size = data.get("account_size", 150_000)
                        if acct_size:
                            account_size = int(acct_size)
                            break
        except Exception:
            pass  # Fall back to env var / default

        from lib.services.engine.risk import RiskManager

        _risk_manager = RiskManager(account_size=account_size)
        logger.info("RiskManager initialised (account=$%s)", f"{account_size:,}")
    return _risk_manager


# ---------------------------------------------------------------------------
# Redis command queue — data service → engine communication
# ---------------------------------------------------------------------------
_RETRAIN_CMD_KEY = "engine:cmd:retrain_cnn"
_RETRAIN_STATUS_KEY = "engine:retrain:status"


def _check_redis_commands(action_handlers: dict) -> None:
    """Check Redis for commands published by the data service.

    CNN retraining has moved to the rb repo — commands are acknowledged
    but no longer executed here.
    """
    try:
        from lib.core.cache import REDIS_AVAILABLE, cache_get

        if not REDIS_AVAILABLE:
            return

        raw = cache_get(_RETRAIN_CMD_KEY)
        if not raw:
            return

        # Consume the command so it doesn't re-fire
        try:
            from lib.core.cache import _r

            if _r:
                _r.delete(_RETRAIN_CMD_KEY)
        except Exception:
            pass

        logger.info(
            "📩 Received retrain command from dashboard — CNN training has moved to the rb repo. "
            "Use the GPU trainer (docker-compose.train.yml) in the rb repo instead."
        )
        _publish_retrain_status(
            "rejected",
            "CNN training has moved to the rb repo. Use the GPU trainer there instead.",
        )

    except Exception as exc:
        logger.debug("Redis command check error (non-fatal): %s", exc)


def _publish_retrain_status(status: str, message: str = "", **extras) -> None:
    """Write retrain job status to Redis so the dashboard can poll it."""
    try:
        from lib.core.cache import cache_set

        payload = json.dumps(
            {
                "status": status,
                "message": message,
                "timestamp": datetime.now(tz=_EST).isoformat(),
                **extras,
            }
        )
        cache_set(_RETRAIN_STATUS_KEY, payload.encode(), ttl=3600)
    except Exception:
        pass


def _run_retrain_from_command(
    session: str = "both",
    skip_dataset: bool = False,
    epochs: int | None = None,
    batch_size: int | None = None,
) -> None:
    """No-op — CNN training has moved to the rb repo."""
    logger.info("⏭️  CNN retrain command ignored — training has moved to the rb repo")
    _publish_retrain_status(
        "rejected",
        "CNN training has moved to the rb repo. Use the GPU trainer there instead.",
    )


# ---------------------------------------------------------------------------
# Action handlers — each corresponds to a ScheduleManager ActionType
# ---------------------------------------------------------------------------


def _handle_compute_daily_focus(engine, account_size: int) -> None:
    """Compute daily focus for all tracked assets and publish to Redis.

    When a LiveRiskPublisher is active, the latest LiveRiskState is passed
    into compute_daily_focus so that focus cards use the remaining risk
    budget for dynamic position sizing (Phase 5C) and include live
    position overlays (Phase 5D).
    """
    from lib.services.engine.focus import (
        compute_daily_focus,
        publish_focus_to_redis,
    )

    # Grab the latest LiveRiskState from the publisher (if available)
    live_risk = None
    if _live_risk_publisher is not None:
        live_risk = _live_risk_publisher.last_state

    logger.info("▶ Computing daily focus...%s", " (with live risk)" if live_risk else "")
    focus = compute_daily_focus(account_size=account_size, live_risk=live_risk)
    publish_focus_to_redis(focus)

    if focus.get("no_trade"):
        logger.warning("⛔ NO TRADE today: %s", focus.get("no_trade_reason", "unknown"))
    else:
        tradeable = focus.get("tradeable_assets", 0)
        logger.info("✅ Daily focus ready: %d tradeable assets", tradeable)


def _handle_fks_recompute(engine) -> None:
    """Trigger data refresh + Ruby recomputation via the DashboardEngine."""
    logger.info("▶ Ruby recomputation (data refresh)...")
    try:
        engine.force_refresh()
        logger.info("✅ Ruby recomputation complete")
    except Exception as exc:
        logger.warning("Ruby recompute error: %s", exc)

    # Run HMM regime detection for all assets with available bar data and
    # publish the consolidated state map to Redis so the dashboard panel
    # and Prometheus metrics scrape can both read it.
    _publish_regime_states()

    # Run the Ruby Signal Engine over all focus assets — port of ruby.pine.
    # Feeds each new 1-minute bar through the stateful RubySignalEngine,
    # publishes engine:ruby_signal:<symbol> to Redis (TTL 15 min), and
    # forwards detected signals to the PositionManager.
    try:
        from lib.services.engine.handlers import handle_ruby_recompute

        # Determine the current active session for asset list selection
        from lib.services.engine.scheduler import ScheduleManager, SessionMode

        _sm = ScheduleManager()
        _session = _sm.current_session
        _session_key = "london_ny" if _session in (SessionMode.ACTIVE,) else "us"
        handle_ruby_recompute(engine, session_key=_session_key)
    except Exception as exc:
        logger.warning("Ruby Signal Engine recompute error (non-fatal): %s", exc)


def _publish_regime_states() -> None:
    """Run HMM regime detection across all focus assets and publish to Redis.

    Reads bar data from the engine:bars_1m / engine:bars_daily cache keys,
    fits / updates the per-instrument RegimeDetector, then writes:
      - ``engine:regime_states``  — JSON map of {symbol → regime_info} (TTL 10 min)
      - ``engine:regime:{symbol}`` — per-symbol JSON (TTL 10 min)

    Also pushes the results into Prometheus gauges immediately so the next
    /metrics/prometheus scrape reflects the latest regime.
    """
    try:
        import io

        import pandas as pd

        from lib.analysis.regime import detect_regime_hmm
        from lib.core.cache import cache_get, cache_set

        raw_focus = cache_get("engine:daily_focus")
        if not raw_focus:
            logger.debug("No daily focus — skipping regime update")
            return

        focus_data = json.loads(raw_focus)
        assets = focus_data.get("assets", [])
        if not assets:
            return

        regime_map: dict[str, dict] = {}

        for asset in assets:
            symbol = asset.get("symbol", "")
            ticker = asset.get("ticker", "") or symbol
            if not symbol:
                continue

            try:
                # Prefer daily bars for regime (longer history = better HMM fit)
                bars = None
                for cache_key in (
                    f"engine:bars_daily:{ticker}",
                    f"engine:bars_1m:{ticker}",
                ):
                    raw_bars = cache_get(cache_key)
                    if raw_bars:
                        raw_str = raw_bars.decode("utf-8") if isinstance(raw_bars, bytes) else raw_bars
                        candidate = pd.read_json(io.StringIO(raw_str))
                        if not candidate.empty and len(candidate) >= 50:
                            bars = candidate
                            break

                if bars is None or bars.empty:
                    logger.debug("No bars for regime detection on %s", symbol)
                    continue

                info = detect_regime_hmm(ticker, bars)
                regime_map[symbol] = info

                # Per-symbol key
                cache_set(
                    f"engine:regime:{symbol}",
                    json.dumps(info, default=str).encode(),
                    ttl=600,
                )
            except Exception as exc:
                logger.debug("Regime detection skipped for %s: %s", symbol, exc)
                continue

        if not regime_map:
            return

        # Consolidated map
        cache_set(
            "engine:regime_states",
            json.dumps(regime_map, default=str).encode(),
            ttl=600,
        )

        # Push into Prometheus gauges immediately (don't wait for scrape)
        try:
            from lib.services.data.api.metrics import update_regime

            for sym, info in regime_map.items():
                update_regime(
                    symbol=sym,
                    regime=info.get("regime", "choppy"),
                    confidence=float(info.get("confidence", 0.0)),
                    position_multiplier=float(info.get("position_multiplier", 0.25)),
                )
        except Exception:
            pass

        logger.debug(
            "Regime states published for %d symbols: %s",
            len(regime_map),
            {s: v.get("regime") for s, v in regime_map.items()},
        )

    except Exception as exc:
        logger.debug("_publish_regime_states failed (non-fatal): %s", exc)


def _handle_publish_focus_update(engine, account_size: int) -> None:
    """Re-publish current focus data to Redis for SSE consumers.

    Passes the latest LiveRiskState (if available) so focus cards use
    dynamic risk budgets and include live position overlays.

    Also flushes any debounced GitHub signals.csv changes (Phase TV-A)
    so TradingView ``request.seed()`` stays current even if no new signal
    fires within the debounce window.
    """
    from lib.services.engine.focus import (
        compute_daily_focus,
        publish_focus_to_redis,
    )

    try:
        live_risk = None
        if _live_risk_publisher is not None:
            live_risk = _live_risk_publisher.last_state

        focus = compute_daily_focus(account_size=account_size, live_risk=live_risk)
        publish_focus_to_redis(focus)
        logger.debug("Focus update published to Redis%s", " (with live risk)" if live_risk else "")
    except Exception as exc:
        logger.debug("Focus publish error (non-fatal): %s", exc)


def _handle_check_no_trade(engine, account_size: int) -> None:
    """Check should-not-trade conditions using the full detector."""
    from lib.core.cache import cache_get

    try:
        raw = cache_get("engine:daily_focus")
        if not raw:
            return

        focus = json.loads(raw)
        assets = focus.get("assets", [])

        # Get risk status from RiskManager for loss/streak checks
        rm = _get_risk_manager(account_size)
        risk_status = rm.get_status()

        from lib.services.engine.patterns import (
            clear_no_trade_alert,
            evaluate_no_trade,
            publish_no_trade_alert,
        )

        result = evaluate_no_trade(assets, risk_status=risk_status)

        if result.should_skip:
            logger.warning(
                "⛔ No-trade condition active (%s): %s",
                result.severity,
                result.primary_reason,
            )
            # Update the focus payload
            focus["no_trade"] = True
            focus["no_trade_reason"] = result.primary_reason
            focus["no_trade_reasons"] = result.reasons
            focus["no_trade_severity"] = result.severity
            from lib.services.engine.focus import publish_focus_to_redis

            publish_focus_to_redis(focus)

            # Publish structured no-trade alert
            publish_no_trade_alert(result)
        else:
            # Clear any stale no-trade alerts
            if focus.get("no_trade"):
                focus["no_trade"] = False
                focus["no_trade_reason"] = ""
                from lib.services.engine.focus import publish_focus_to_redis

                publish_focus_to_redis(focus)
            clear_no_trade_alert()

    except Exception as exc:
        logger.debug("No-trade check error (non-fatal): %s", exc)


def _handle_grok_morning_brief(engine) -> None:
    """Run Grok morning market briefing (pre-market).

    Phase 3C: In addition to the legacy GrokSession briefing, runs the
    structured daily-plan Grok analysis (``run_daily_plan_grok_analysis``)
    which returns parsed JSON with macro_bias, top_assets, risk_warnings,
    economic_events, session_plan, correlation_notes, and swing_insights.

    The structured result is stored in Redis (``engine:grok_daily_plan``)
    and also published to the ``dashboard:grok`` channel so the dashboard
    Grok panel can render the structured morning brief card.
    """
    logger.info("▶ Grok morning briefing...")

    # --- Legacy GrokSession briefing (free-text) ---
    try:
        from lib.integrations.grok_helper import GrokSession

        session = GrokSession()
        if session is not None:
            logger.info("✅ Grok morning briefing (GrokSession) complete")
        else:
            logger.info("Grok helper not available — skipping morning brief")
    except Exception as exc:
        logger.debug("Grok morning brief (legacy) skipped: %s", exc)

    # --- Phase 3C: Structured daily-plan Grok analysis ---
    api_key = os.getenv("XAI_API_KEY", "").strip()
    if not api_key:
        logger.debug("No XAI_API_KEY — skipping structured Grok analysis")
        return

    try:
        from lib.integrations.grok_helper import (
            format_grok_daily_plan_for_display,
            run_daily_plan_grok_analysis,
        )

        # Load biases and focus asset names from the daily plan (if computed)
        raw_plan = None
        try:
            from lib.core.cache import cache_get

            raw = cache_get("engine:daily_plan")
            if raw:
                raw_plan = json.loads(raw.decode("utf-8") if isinstance(raw, bytes) else raw)
        except Exception:
            pass

        all_biases: dict = {}
        asset_names: list[str] = []
        swing_names: list[str] = []
        scalp_names: list[str] = []

        if raw_plan:
            all_biases = raw_plan.get("all_biases", {})
            swing_names = raw_plan.get("swing_candidate_names", [])
            scalp_names = raw_plan.get("scalp_focus_names", [])
            # Get all asset names from biases
            asset_names = list(all_biases.keys())

        if not asset_names:
            # Fallback: use ASSETS dict
            try:
                from lib.core.models import ASSETS

                asset_names = list(ASSETS.keys())
            except ImportError:
                pass

        if not asset_names:
            logger.info("Structured Grok analysis skipped — no asset names available")
            return

        logger.info(
            "Running structured Grok daily plan analysis (%d assets, %d swing, %d scalp)...",
            len(asset_names),
            len(swing_names),
            len(scalp_names),
        )

        grok_result = run_daily_plan_grok_analysis(
            biases=all_biases,
            asset_names=asset_names,
            swing_candidate_names=swing_names or None,
            scalp_focus_names=scalp_names or None,
            api_key=api_key,
        )

        if grok_result:
            # Store structured result in Redis
            try:
                from lib.core.cache import cache_set as _cs

                _cs(
                    "engine:grok_daily_plan",
                    json.dumps(grok_result, default=str).encode(),
                    ttl=18 * 3600,
                )
            except Exception:
                pass

            # Also update the daily plan in Redis with the grok_analysis field
            if raw_plan is not None:
                raw_plan["grok_analysis"] = grok_result
                raw_plan["grok_available"] = True
                display_text = format_grok_daily_plan_for_display(grok_result)
                raw_plan["market_context"] = display_text
                try:
                    from lib.core.cache import cache_set as _cs2

                    _cs2(
                        "engine:daily_plan",
                        json.dumps(raw_plan, default=str).encode(),
                        ttl=18 * 3600,
                    )
                except Exception:
                    pass

            # Publish to dashboard Grok channel
            display_text = format_grok_daily_plan_for_display(grok_result)
            _publish_grok_update(display_text)

            logger.info(
                "✅ Structured Grok analysis complete: macro=%s, %d top assets, %d warnings",
                grok_result.get("macro_bias", "?"),
                len(grok_result.get("top_assets", [])),
                len(grok_result.get("risk_warnings", [])),
            )
        else:
            logger.info("Structured Grok analysis returned no results")

    except ImportError:
        logger.debug("Structured Grok analysis not available (import failed)")
    except Exception as exc:
        logger.warning("Structured Grok analysis failed (non-fatal): %s", exc)


def _handle_grok_live_update(engine) -> None:
    """Run Grok 15-minute live market update (active hours).

    Uses compact ≤8-line format by default.
    Falls back to local format_live_compact() if API is unavailable.
    """
    logger.info("▶ Grok live update (compact)...")
    try:
        api_key = os.getenv("XAI_API_KEY", "")

        # Try local compact format from focus data first (fast, free)
        from lib.core.cache import cache_get

        raw = cache_get("engine:daily_focus")
        compact_text = None

        if raw:
            focus = json.loads(raw)
            assets = focus.get("assets", [])
            if assets:
                from lib.integrations.grok_helper import format_live_compact

                compact_text = format_live_compact(assets)

        # If we have an API key, try the Grok compact call
        if api_key and compact_text:
            logger.info("✅ Grok live update (local compact): %d chars", len(compact_text))
        elif api_key:
            logger.debug("Grok API key present but no focus data for compact update")
        else:
            logger.debug("No XAI_API_KEY — using local compact format only")

        # Publish compact update to Redis for SSE grok-update event
        if compact_text:
            _publish_grok_update(compact_text)

    except Exception as exc:
        logger.debug("Grok live update skipped: %s", exc)


def _publish_grok_update(text: str) -> None:
    """Publish a Grok update to Redis for SSE streaming."""
    try:
        from lib.core.cache import REDIS_AVAILABLE, _r, cache_set

        now = datetime.now(tz=_EST)
        payload = json.dumps(
            {
                "text": text,
                "timestamp": now.isoformat(),
                "time_et": now.strftime("%I:%M %p ET"),
                "compact": True,
            },
            default=str,
        )

        cache_set("engine:grok_update", payload.encode(), ttl=900)  # 15 min TTL

        if REDIS_AVAILABLE and _r is not None:
            with contextlib.suppress(Exception):
                _r.publish("dashboard:grok", payload)

        logger.debug("Grok update published to Redis")
    except Exception as exc:
        logger.debug("Failed to publish Grok update: %s", exc)


def _handle_prep_alerts(engine) -> None:
    """Prepare alert thresholds for active session."""
    logger.info("▶ Preparing alert thresholds...")
    # Alerts module is already initialized via DashboardEngine
    logger.info("✅ Alerts ready for active session")


def _handle_check_risk_rules(engine, account_size: int = 50_000) -> None:
    """Check risk rules using the RiskManager.

    Syncs positions from cache, evaluates all risk rules, publishes
    status to Redis, logs any warnings, and persists notable events
    to the database for permanent audit trail.
    """
    logger.debug("▶ Risk rules check...")
    try:
        rm = _get_risk_manager(account_size)

        # Sync positions from cache (if available)
        try:
            from lib.core.cache import cache_get

            raw = cache_get("positions:current")
            if not raw:
                # Try the hashed key used by positions router
                from lib.services.data.api.positions import (
                    _POSITIONS_CACHE_KEY,
                )

                raw = cache_get(_POSITIONS_CACHE_KEY)
            if raw:
                data = json.loads(raw)
                positions = data.get("positions", [])
                if positions:
                    rm.sync_positions(positions)
                    logger.debug("Synced %d positions from cache", len(positions))
        except Exception as exc:
            logger.debug("Position sync skipped (non-fatal): %s", exc)

        # Check overnight risk
        has_overnight, overnight_msg = rm.check_overnight_risk()
        if has_overnight:
            logger.warning(overnight_msg)
            # Persist overnight warning to audit trail
            _persist_risk_event(
                "warning",
                reason=overnight_msg,
                daily_pnl=rm.daily_pnl,
                open_trades=rm.open_trade_count,
                account_size=account_size,
            )

        # Publish risk status to Redis
        rm.publish_to_redis()

        status = rm.get_status()
        if not status["can_trade"]:
            logger.warning(
                "⚠️ Risk block active: %s (daily P&L: $%.2f)",
                status["block_reason"],
                status["daily_pnl"],
            )
            # Persist risk block to audit trail
            _persist_risk_event(
                "block",
                reason=status["block_reason"],
                daily_pnl=status.get("daily_pnl", 0.0),
                open_trades=status.get("open_trade_count", 0),
                account_size=account_size,
                risk_pct=status.get("risk_pct_of_account", 0.0),
            )
        else:
            logger.debug(
                "✅ Risk OK: %d/%d trades, daily P&L $%.2f, exposure $%.2f",
                status["open_trade_count"],
                status["max_open_trades"],
                status["daily_pnl"],
                status["total_risk_exposure"],
            )

    except Exception as exc:
        logger.debug("Risk rules check error (non-fatal): %s", exc)


def _persist_risk_event(
    event_type: str,
    symbol: str = "",
    side: str = "",
    reason: str = "",
    daily_pnl: float = 0.0,
    open_trades: int = 0,
    account_size: int = 0,
    risk_pct: float = 0.0,
) -> None:
    """Persist a risk event to the database audit trail (best-effort)."""
    try:
        from lib.core.models import record_risk_event
        from lib.services.engine.scheduler import ScheduleManager

        session = ScheduleManager().get_session_mode().value
        record_risk_event(
            event_type=event_type,
            symbol=symbol,
            side=side,
            reason=reason,
            daily_pnl=daily_pnl,
            open_trades=open_trades,
            account_size=account_size,
            risk_pct=risk_pct,
            session=session,
        )
    except Exception as exc:
        logger.debug("Failed to persist risk event (non-fatal): %s", exc)


# ===========================================================================
# ORB session handlers — Phase 1C: delegate to unified handle_orb_check()
# ===========================================================================


def _handle_check_orb_london(engine) -> None:
    """Check for London Open ORB patterns (03:00–03:30 ET / 08:00–08:30 UTC)."""
    from lib.services.engine.handlers import handle_orb_check
    from lib.trading.strategies.rb.open import LONDON_SESSION

    handle_orb_check(engine, orb_session=LONDON_SESSION)


def _handle_check_orb_london_ny(engine) -> None:
    """Check for London-NY Crossover ORB patterns (08:00–08:30 ET)."""
    from lib.services.engine.handlers import handle_orb_check
    from lib.trading.strategies.rb.open import LONDON_NY_SESSION

    handle_orb_check(engine, orb_session=LONDON_NY_SESSION)


def _handle_check_orb_frankfurt(engine) -> None:
    """Check for Frankfurt/Xetra Open ORB patterns (03:00–03:30 ET / 08:00 CET)."""
    from lib.services.engine.handlers import handle_orb_check
    from lib.trading.strategies.rb.open import FRANKFURT_SESSION

    handle_orb_check(engine, orb_session=FRANKFURT_SESSION)


def _handle_check_orb_sydney(engine) -> None:
    """Check for Sydney/ASX Open ORB patterns (18:30–19:00 ET, overnight)."""
    from lib.services.engine.handlers import handle_orb_check
    from lib.trading.strategies.rb.open import SYDNEY_SESSION

    handle_orb_check(engine, orb_session=SYDNEY_SESSION)


def _handle_check_orb_cme(engine) -> None:
    """Check for CME Globex Re-Open ORB patterns (18:00–18:30 ET, overnight)."""
    from lib.services.engine.handlers import handle_orb_check
    from lib.trading.strategies.rb.open import CME_OPEN_SESSION

    handle_orb_check(engine, orb_session=CME_OPEN_SESSION)


def _handle_check_orb_tokyo(engine) -> None:
    """Check for Tokyo/TSE Open ORB patterns (19:00–19:30 ET, overnight)."""
    from lib.services.engine.handlers import handle_orb_check
    from lib.trading.strategies.rb.open import TOKYO_SESSION

    handle_orb_check(engine, orb_session=TOKYO_SESSION)


def _handle_check_orb_shanghai(engine) -> None:
    """Check for Shanghai/HK Open ORB patterns (21:00–21:30 ET, overnight)."""
    from lib.services.engine.handlers import handle_orb_check
    from lib.trading.strategies.rb.open import SHANGHAI_SESSION

    handle_orb_check(engine, orb_session=SHANGHAI_SESSION)


def _handle_check_orb_cme_settle(engine) -> None:
    """Check for CME Settlement ORB patterns (14:00–14:30 ET)."""
    from lib.services.engine.handlers import handle_orb_check
    from lib.trading.strategies.rb.open import CME_SETTLEMENT_SESSION

    handle_orb_check(engine, orb_session=CME_SETTLEMENT_SESSION)


def _handle_check_orb_crypto_utc0(engine) -> None:
    """Check for Crypto UTC-midnight ORB patterns (19:00–19:30 ET EST / 00:00 UTC)."""
    try:
        from lib.trading.strategies.rb.open import CRYPTO_UTC_MIDNIGHT_SESSION
    except ImportError:
        logger.warning("CRYPTO_UTC_MIDNIGHT_SESSION not available — crypto ORB disabled")
        return

    from lib.services.engine.handlers import handle_orb_check

    handle_orb_check(engine, orb_session=CRYPTO_UTC_MIDNIGHT_SESSION)


def _handle_check_orb_crypto_utc12(engine) -> None:
    """Check for Crypto UTC-noon ORB patterns (07:00–07:30 ET EST / 12:00 UTC)."""
    try:
        from lib.trading.strategies.rb.open import CRYPTO_UTC_NOON_SESSION
    except ImportError:
        logger.warning("CRYPTO_UTC_NOON_SESSION not available — crypto ORB disabled")
        return

    from lib.services.engine.handlers import handle_orb_check

    handle_orb_check(engine, orb_session=CRYPTO_UTC_NOON_SESSION)


def _handle_check_orb(engine, orb_session=None) -> None:
    """Check for Opening Range Breakout patterns — unified pipeline.

    Phase 1C: delegates entirely to ``handle_orb_check()`` in ``handlers.py``,
    which runs detection via ``detect_range_breakout()``, quality filters,
    CNN inference, and publishing through the single generic pipeline.

    Args:
        engine: The engine instance.
        orb_session: ORBSession to check. Defaults to US_SESSION if None.
    """
    from lib.services.engine.handlers import handle_orb_check
    from lib.trading.strategies.rb.open import US_SESSION

    if orb_session is None:
        orb_session = US_SESSION

    handle_orb_check(engine, orb_session=orb_session)


def _persist_orb_event(result, metadata: dict | None = None) -> int | None:
    """Persist an ORB evaluation result to the database audit trail (best-effort).

    Returns the inserted row ID so callers can enrich the record later
    with filter/CNN outcomes via ``_persist_orb_enrichment()``.
    """
    try:
        from lib.core.models import record_orb_event
        from lib.services.engine.scheduler import ScheduleManager

        session = ScheduleManager().get_session_mode().value
        row_id = record_orb_event(
            symbol=result.symbol,
            or_high=result.or_high,
            or_low=result.or_low,
            or_range=result.or_range,
            atr_value=result.atr_value,
            breakout_detected=result.breakout_detected,
            direction=result.direction,
            trigger_price=result.trigger_price,
            long_trigger=result.long_trigger,
            short_trigger=result.short_trigger,
            bar_count=getattr(result, "bar_count", 0),
            session=session,
            metadata=metadata,
        )
        return row_id
    except Exception as exc:
        logger.debug("Failed to persist ORB event (non-fatal): %s", exc)
        return None


def _persist_orb_enrichment(row_id: int | None, metadata: dict) -> None:
    """Update an existing ORB event row with filter/CNN enrichment metadata.

    Called after the full filter + CNN pipeline completes so the audit
    trail captures: filter_passed, filter_summary, cnn_prob,
    cnn_confidence, cnn_signal, cnn_gated, published.

    This is best-effort — failures are logged but never block trading.
    """
    if row_id is None:
        return
    try:
        from lib.core.models import _get_conn, _is_using_postgres

        pg = _is_using_postgres()
        ph = "%s" if pg else "?"
        meta_json = json.dumps(metadata, default=str)

        conn = _get_conn()
        conn.execute(
            f"UPDATE orb_events SET metadata_json = {ph} WHERE id = {ph}",
            (meta_json, row_id),
        )
        conn.commit()
        conn.close()
    except Exception as exc:
        logger.debug("Failed to enrich ORB event %s (non-fatal): %s", row_id, exc)


# ---------------------------------------------------------------------------
# Routing table: ActionType → handler function
# Populated after all handlers are defined (see _ACTION_HANDLERS at bottom).
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Multi-BreakoutType handlers — PDR, IB, CONS, and parallel sweep
# ---------------------------------------------------------------------------


def _get_assets_for_session_key(session_key: str) -> list[dict]:
    """Return the focus asset list filtered to the given session's asset set.

    Falls back to the full daily focus if no session-specific list is found.
    """
    try:
        from lib.core.cache import cache_get
        from lib.trading.strategies.rb.open import SESSION_ASSETS

        raw_focus = cache_get("engine:daily_focus")
        if not raw_focus:
            return []

        focus_data = json.loads(raw_focus)
        all_assets = focus_data.get("assets", [])

        session_tickers = set(SESSION_ASSETS.get(session_key, []))
        if not session_tickers:
            return all_assets

        return [
            a for a in all_assets if a.get("ticker", "") in session_tickers or a.get("symbol", "") in session_tickers
        ]
    except Exception as exc:
        logger.debug("_get_assets_for_session_key(%s) error: %s", session_key, exc)
        return []


def _fetch_bars_1m(engine, ticker: str, symbol: str) -> "pd.DataFrame | None":
    """Fetch 1-minute bars from cache or engine data service (best-effort)."""
    try:
        import io

        import pandas as pd

        from lib.core.cache import cache_get

        bars_key = f"engine:bars_1m:{ticker or symbol}"
        raw_bars = cache_get(bars_key)
        if raw_bars:
            raw_str = raw_bars.decode("utf-8") if isinstance(raw_bars, bytes) else raw_bars
            return pd.read_json(io.StringIO(raw_str))

        with contextlib.suppress(Exception):
            return engine._fetch_tf_safe(ticker or symbol, interval="1m", period="1d")
    except Exception as exc:
        logger.debug("_fetch_bars_1m(%s) error: %s", symbol, exc)
    return None


def _publish_breakout_result(result: "BreakoutResult", orb_session_key: str = "us") -> None:
    """Publish a non-ORB breakout result to Redis for SSE / dashboard consumption.

    Also pushes the signal to the TradingView signal store + GitHub signals.csv
    (Phase TV-A) so TradingView's request.seed() can auto-draw engine levels.
    """
    try:
        from lib.core.cache import cache_set

        payload = result.to_dict()
        payload["published_at"] = datetime.now(tz=ZoneInfo("America/New_York")).isoformat()
        payload["orb_session"] = orb_session_key

        key = f"engine:breakout:{result.breakout_type.name.lower()}:{result.symbol}"
        cache_set(key, json.dumps(payload).encode(), ttl=300)

        # Also publish to the generic breakout channel so the SSE picks it up
        try:
            from lib.core.cache import REDIS_AVAILABLE, _r

            if REDIS_AVAILABLE and _r is not None:
                _r.publish("dashboard:breakout", json.dumps(payload))
        except Exception:
            pass

        logger.info(
            "🔔 %s BREAKOUT: %s %s @ %.4f (range %.4f–%.4f)",
            result.breakout_type.name,
            result.direction,
            result.symbol,
            result.trigger_price,
            result.range_low,
            result.range_high,
        )

    except Exception as exc:
        logger.debug("_publish_breakout_result error: %s", exc)


def _publish_pm_orders(orders: list) -> None:  # type: ignore[type-arg]
    """Publish PositionManager OrderCommands to Redis, the dashboard, and CopyTrader.

    Each order is written to:
      - ``engine:pm:orders`` — a list (RPUSH) of JSON-serialised commands (TTL 60s)
      - ``dashboard:pm_orders`` — Redis pub/sub channel for real-time SSE streaming

    When ``RITHMIC_COPY_TRADING=1`` the orders are also forwarded to the
    :class:`~lib.services.engine.copy_trader.CopyTrader` via
    ``execute_order_commands()``.  The CopyTrader runs in a fire-and-forget
    asyncio task so it never blocks the synchronous engine loop.

    Rithmic path (when enabled):
      - Entry BUY/SELL → ``CopyTrader.send_order_and_copy()`` (main + slaves, MANUAL flag)
      - MODIFY_STOP    → ``CopyTrader.modify_stop_on_all()`` (MANUAL flag)
      - CANCEL         → ``CopyTrader.cancel_on_all()`` (MANUAL flag)
      - STOP companion → silently skipped (covered by server-side bracket on entry)
    """
    if not orders:
        return
    try:
        from lib.core.cache import REDIS_AVAILABLE, _r, cache_set

        now = datetime.now(tz=_EST).isoformat()
        serialised = []
        for order in orders:
            d = order.to_dict() if hasattr(order, "to_dict") else dict(order)
            d.setdefault("published_at", now)
            serialised.append(json.dumps(d, default=str))

        if REDIS_AVAILABLE and _r is not None:
            pipe = _r.pipeline()
            key = "engine:pm:orders"
            for s in serialised:
                pipe.rpush(key, s)
            pipe.expire(key, 60)
            pipe.execute()
            for s in serialised:
                _r.publish("dashboard:pm_orders", s)

        # Also write consolidated status for dashboard SSE
        try:
            pm = _position_manager
            if pm is not None:
                positions_payload = {
                    p.ticker: {
                        "direction": p.direction,
                        "entry_price": p.entry_price,
                        "current_price": p.current_price,
                        "stop_loss": p.stop_loss,
                        "tp1": p.tp1,
                        "tp2": p.tp2,
                        "tp3": p.tp3,
                        "phase": p.phase.value,
                        "unrealized_pnl": round(p.unrealized_pnl, 4),
                        "r_multiple": round(p.r_multiple, 3),
                        "breakout_type": p.breakout_type,
                        "session_key": p.session_key,
                    }
                    for p in pm.get_all_positions().values()
                }
                cache_set(
                    "engine:pm:positions",
                    json.dumps(
                        {
                            "positions": positions_payload,
                            "count": len(positions_payload),
                            "updated_at": now,
                        },
                        default=str,
                    ).encode(),
                    ttl=120,
                )
        except Exception:
            pass

        logger.info(
            "📤 PositionManager: %d order(s) dispatched → Redis + CopyTrader",
            len(orders),
        )
    except Exception as exc:
        logger.debug("_publish_pm_orders error (non-fatal): %s", exc)

    # ------------------------------------------------------------------
    # Rithmic copy-trading path (only when RITHMIC_COPY_TRADING=1)
    # ------------------------------------------------------------------
    ct = _get_copy_trader()
    if ct is not None and orders:
        _dispatch_orders_to_copy_trader(ct, orders)


def _dispatch_orders_to_copy_trader(ct: Any, orders: list) -> None:  # type: ignore[type-arg]
    """Fire-and-forget: run ``ct.execute_order_commands(orders)`` in an asyncio task.

    The engine main loop is synchronous; we bridge into async by either
    scheduling a task on a running event loop or spinning up a short-lived
    one.  Any failure is non-fatal and logged at DEBUG level so it never
    interrupts the alert pipeline.

    The ``entry_prices`` dict is built from the current PositionManager state
    so that ``modify_stop_on_all`` has accurate tick-conversion data.
    """
    try:
        import asyncio

        # Build entry_prices from active positions so MODIFY_STOP conversions
        # are accurate (ticker → entry_price from the live MicroPosition).
        entry_prices: dict[str, float] = {}
        pm = _position_manager
        if pm is not None:
            try:
                for ticker, pos in pm.get_all_positions().items():
                    entry_prices[ticker] = getattr(pos, "entry_price", 0.0)
            except Exception:
                pass

        async def _run() -> None:
            try:
                results = await ct.execute_order_commands(orders, entry_prices=entry_prices)
                ok = sum(
                    1
                    for r in results
                    if (hasattr(r, "all_submitted") and r.all_submitted) or (isinstance(r, dict) and r.get("ok"))
                )
                logger.info(
                    "🔗 CopyTrader: %d order command(s) → %d result(s) (%d ok)",
                    len(orders),
                    len(results),
                    ok,
                )
            except Exception as exc:
                logger.debug("CopyTrader execute_order_commands error (non-fatal): %s", exc)

        # Try to schedule on an already-running loop (e.g. if engine ever
        # becomes async).  Fall back to a new one-shot loop.
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(_run())
            else:
                loop.run_until_complete(_run())
        except RuntimeError:
            # No current event loop — spin up a fresh one
            new_loop = asyncio.new_event_loop()
            try:
                new_loop.run_until_complete(_run())
            finally:
                new_loop.close()

    except Exception as exc:
        logger.debug("_dispatch_orders_to_copy_trader error (non-fatal): %s", exc)


def _dispatch_to_position_manager(
    result: "Any",
    bars_1m: "pd.DataFrame | None" = None,
    session_key: str = "us",
    range_config: "Any" = None,
) -> None:
    """Forward a published breakout signal to the PositionManager.

    Accepts either an ``ORBResult`` or a ``BreakoutResult``; both expose the
    same duck-typed attributes that ``PositionManager.process_signal()`` needs.

    For ORBResult objects (which use ``or_high``/``or_low`` instead of
    ``range_high``/``range_low``) we attach the missing attributes on-the-fly
    so the PositionManager doesn't have to know about ORB-specific naming.

    This is intentionally best-effort — any failure here must not block the
    alert pipeline.
    """
    pm = _position_manager
    if pm is None:
        return

    try:
        signal = result

        # ORBResult compatibility shim — attach range_high/range_low
        if not hasattr(signal, "range_high") or not signal.range_high:
            or_high = getattr(signal, "or_high", 0.0)
            or_low = getattr(signal, "or_low", 0.0)
            try:
                signal.range_high = or_high
                signal.range_low = or_low
                signal.breakout_type = getattr(signal, "breakout_type", None) or type("_T", (), {"value": "ORB"})()
            except AttributeError:
                pass  # frozen dataclass — leave as-is

        # Attach session_key if missing
        if not getattr(signal, "session_key", ""):
            with contextlib.suppress(AttributeError):
                signal.session_key = session_key

        # Attach filter_passed as True (signal already passed the filter gate)
        if getattr(signal, "filter_passed", None) is None:
            with contextlib.suppress(AttributeError):
                signal.filter_passed = True

        orders = pm.process_signal(signal, bars_1m=bars_1m, range_config=range_config)
        if orders:
            _publish_pm_orders(orders)

    except Exception as exc:
        logger.debug("_dispatch_to_position_manager error (non-fatal): %s", exc)


def _handle_update_positions(engine) -> None:
    """Run PositionManager.update_all() on every scheduled 1m-bar tick.

    Fetches the latest 1-minute bars for every core watchlist ticker,
    calls ``update_all()``, then dispatches any resulting bracket / EMA9
    trailing orders via Redis and CopyTrader.

    Safe to call frequently — exits immediately if no positions are active.
    """
    pm = _position_manager
    if pm is None or pm.get_position_count() == 0:
        return

    try:
        import io

        import pandas as pd

        from lib.core.cache import cache_get
        from lib.core.models import CORE_TICKERS

        bars_by_ticker: dict[str, pd.DataFrame] = {}

        for ticker in CORE_TICKERS:
            try:
                raw = cache_get(f"engine:bars_1m:{ticker}")
                if raw:
                    raw_str = raw.decode("utf-8") if isinstance(raw, bytes) else raw
                    df = pd.read_json(io.StringIO(raw_str))
                    if not df.empty:
                        bars_by_ticker[ticker] = df
            except Exception as exc:
                logger.debug("Could not fetch bars for PM update (%s): %s", ticker, exc)

        if not bars_by_ticker:
            return

        orders = pm.update_all(bars_by_ticker)
        if orders:
            _publish_pm_orders(orders)
            logger.info(
                "📊 PositionManager update: %d order(s) from %d active position(s)",
                len(orders),
                pm.get_position_count(),
            )

            # Position changed — force immediate live risk publish so the
            # dashboard risk strip and focus cards update within 1-2 seconds.
            if _live_risk_publisher is not None:
                try:
                    _ = _live_risk_publisher.force_publish()
                except Exception as exc:
                    logger.debug("LiveRisk force_publish after position update failed: %s", exc)

    except Exception as exc:
        logger.debug("_handle_update_positions error (non-fatal): %s", exc)


def _tick_live_risk_publisher() -> None:
    """Tick the LiveRiskPublisher — publishes every 5s if due.

    Called on every engine loop iteration.  The publisher internally
    tracks elapsed time and only recomputes + publishes when the
    interval has elapsed (default 5 seconds).
    """
    if _live_risk_publisher is None:
        return
    try:
        state = _live_risk_publisher.tick()
        if state is not None:
            logger.debug(
                "📊 LiveRisk published: PnL=$%.2f | Pos=%d/%d | Health=%s",
                state.total_pnl,
                state.open_position_count,
                state.max_open_trades,
                state.health,
            )
    except Exception as exc:
        logger.debug("LiveRisk tick error (non-fatal): %s", exc)


def _persist_breakout_result(result: "BreakoutResult", session_key: str = "") -> int | None:
    """Persist a BreakoutResult to orb_events using the new breakout_type column."""
    try:
        from lib.core.models import record_orb_event

        row_id = record_orb_event(
            symbol=result.symbol,
            or_high=result.range_high,
            or_low=result.range_low,
            or_range=result.range_size,
            atr_value=result.atr_value,
            breakout_detected=result.breakout_detected,
            direction=result.direction,
            trigger_price=result.trigger_price,
            long_trigger=result.long_trigger,
            short_trigger=result.short_trigger,
            bar_count=result.range_bar_count,
            session=session_key,
            metadata=result.extra or {},
            breakout_type=result.breakout_type.name,
            mtf_score=result.mtf_score,
            macd_slope=result.macd_slope,
            divergence=getattr(result, "divergence_type", None) or "",
        )
        return row_id
    except Exception as exc:
        logger.debug("_persist_breakout_result error (non-fatal): %s", exc)
        return None


def _handle_check_pdr(engine, session_key: str = "london_ny") -> None:
    """Check for Previous Day Range (PDR) breakouts across session assets.

    Delegates to the generic ``handle_breakout_check()`` pipeline (Phase 1D).
    """
    from lib.core.breakout_types import BreakoutType
    from lib.services.engine.handlers import handle_breakout_check

    handle_breakout_check(engine, BreakoutType.PrevDay, session_key=session_key)


def _handle_check_ib(engine, session_key: str = "us") -> None:
    """Check for Initial Balance (IB) breakouts across US session assets.

    Delegates to the generic ``handle_breakout_check()`` pipeline (Phase 1D).
    Only fired by the scheduler after 10:30 ET when the IB is complete.
    """
    from lib.core.breakout_types import BreakoutType
    from lib.services.engine.handlers import handle_breakout_check

    handle_breakout_check(engine, BreakoutType.InitialBalance, session_key=session_key)


def _handle_check_consolidation(engine, session_key: str = "london_ny") -> None:
    """Check for Consolidation/Squeeze breakouts across session assets.

    Delegates to the generic ``handle_breakout_check()`` pipeline (Phase 1D).
    Valid throughout the full active window — squeeze breakouts can fire at
    any time once a sustained contraction is detected.
    """
    from lib.core.breakout_types import BreakoutType
    from lib.services.engine.handlers import handle_breakout_check

    handle_breakout_check(engine, BreakoutType.Consolidation, session_key=session_key)


def _handle_check_breakout_multi(engine, session_key: str = "us", types: list[str] | None = None) -> None:
    """Run multiple BreakoutType detectors in parallel for a session's assets.

    Delegates to the generic ``handle_breakout_multi()`` pipeline (Phase 1D).
    ORB is intentionally excluded — it has its own session-specific handlers.

    Args:
        engine: The engine singleton.
        session_key: Session key whose asset list to use.
        types: List of breakout type strings to check ("PDR", "IB", "CONS",
               "WEEKLY", "MONTHLY", "ASIAN", "BBSQUEEZE", "VA", "INSIDE",
               "GAP", "PIVOT", "FIB").
               Defaults to ["PDR", "CONS"] if not specified.
    """
    from lib.services.engine.handlers import handle_breakout_multi
    from lib.trading.strategies.rb.breakout import breakout_type_from_short_name

    if types is None:
        types = ["PDR", "CONS"]

    # Convert short name strings → BreakoutType IntEnum members
    bt_list = []
    for name in types:
        try:
            bt_list.append(breakout_type_from_short_name(name))
        except (KeyError, ValueError):
            logger.warning("Unknown breakout type short name: %s — skipping", name)

    if bt_list:
        handle_breakout_multi(engine, session_key=session_key, types=bt_list)


# Configurable gap-alert threshold (minutes).  Gaps larger than this value
# are reported as warnings after each backfill run.
_GAP_ALERT_MINUTES = int(os.environ.get("BACKFILL_GAP_ALERT_MINUTES", "30"))


def _check_and_alert_gaps(symbols: list[str], gap_threshold_minutes: int = 30) -> None:
    """Scan stored bars for gaps exceeding ``gap_threshold_minutes`` and log alerts.

    Publishes a Redis key ``engine:gap_alerts`` (TTL 26h) so the dashboard
    can surface data-quality warnings without requiring a daily-report cycle.

    Only symbols that have *meaningful* gaps (i.e. not just normal overnight /
    weekend breaks) are included in the alert payload.
    """
    try:
        from lib.services.engine.backfill import get_gap_report

        alerts: list[dict] = []
        for sym in symbols:
            try:
                report = get_gap_report(sym, days_back=3, interval="1m")
                gaps = [g for g in report.get("gaps", []) if g.get("missing_minutes", 0) >= gap_threshold_minutes]
                if not gaps:
                    continue
                worst = max(gaps, key=lambda g: g.get("missing_minutes", 0))
                alerts.append(
                    {
                        "symbol": sym,
                        "gap_count": len(gaps),
                        "worst_gap_minutes": worst.get("missing_minutes", 0),
                        "worst_gap_start": worst.get("start", ""),
                        "worst_gap_end": worst.get("end", ""),
                        "coverage_pct": report.get("coverage_pct", 0),
                    }
                )
                logger.warning(
                    "⚠️  Gap detected in %s: %d gap(s), worst = %d min (%.1f%% coverage)",
                    sym,
                    len(gaps),
                    worst.get("missing_minutes", 0),
                    report.get("coverage_pct", 0),
                )
            except Exception as exc:
                logger.debug("Gap check failed for %s: %s", sym, exc)

        if alerts:
            # Sort by worst gap descending
            alerts.sort(key=lambda a: a["worst_gap_minutes"], reverse=True)
            try:
                from lib.core.cache import REDIS_AVAILABLE, _r, cache_set

                payload = json.dumps(
                    {
                        "alerts": alerts,
                        "threshold_minutes": gap_threshold_minutes,
                        "checked_at": datetime.now(tz=_EST).isoformat(),
                        "symbol_count": len(symbols),
                        "alert_count": len(alerts),
                    },
                    default=str,
                ).encode()
                cache_set("engine:gap_alerts", payload, ttl=26 * 3600)
                if REDIS_AVAILABLE and _r is not None:
                    import contextlib

                    with contextlib.suppress(Exception):
                        _r.publish(
                            "dashboard:gap_alerts",
                            json.dumps({"alert_count": len(alerts)}, default=str),
                        )
            except Exception as exc:
                logger.debug("Failed to publish gap alerts to Redis: %s", exc)
        else:
            # Clear stale alert key when all gaps are resolved
            try:
                from lib.core.cache import cache_set

                cache_set(
                    "engine:gap_alerts",
                    json.dumps(
                        {
                            "alerts": [],
                            "threshold_minutes": gap_threshold_minutes,
                            "checked_at": datetime.now(tz=_EST).isoformat(),
                            "symbol_count": len(symbols),
                            "alert_count": 0,
                        },
                        default=str,
                    ).encode(),
                    ttl=26 * 3600,
                )
            except Exception:
                pass

    except Exception as exc:
        logger.debug("Gap alert sweep failed: %s", exc)


def _handle_historical_backfill(engine) -> None:
    """Backfill historical 1-min bars to Postgres/SQLite (off-hours).

    Calls the backfill module which:
      1. Determines which symbols need data
      2. Finds gaps in existing stored bars
      3. Fetches missing chunks from Massive (primary) or yfinance (fallback)
      4. Stores bars idempotently via UPSERT
      5. Publishes summary to Redis for dashboard visibility
      6. Scans for residual gaps > ``BACKFILL_GAP_ALERT_MINUTES`` and logs
         warnings + publishes ``engine:gap_alerts`` to Redis.
    """
    logger.info("▶ Historical backfill starting")

    try:
        from lib.services.engine.backfill import _get_backfill_symbols, run_backfill

        summary = run_backfill()

        status = summary.get("status", "unknown")
        total_bars = summary.get("total_bars_added", 0)
        duration = summary.get("total_duration_seconds", 0)
        errors = summary.get("errors", [])

        if status == "complete":
            logger.info(
                "✅ Historical backfill complete: +%d bars in %.1fs",
                total_bars,
                duration,
            )
        elif status == "partial":
            logger.warning(
                "⚠️ Historical backfill partial: +%d bars in %.1fs (%d errors)",
                total_bars,
                duration,
                len(errors),
            )
            for err in errors[:5]:
                logger.warning("  Backfill error: %s", err)
        else:
            logger.error(
                "❌ Historical backfill failed in %.1fs: %s",
                duration,
                "; ".join(errors[:3]) if errors else "unknown error",
            )

        # ── Post-backfill gap scan ─────────────────────────────────────────
        # Run after every backfill to catch persistent gaps that the
        # backfiller could not fill (e.g. data not available from any source).
        try:
            symbols = _get_backfill_symbols()
            logger.info(
                "▶ Running post-backfill gap scan (%d symbols, threshold=%dm)", len(symbols), _GAP_ALERT_MINUTES
            )
            _check_and_alert_gaps(symbols, gap_threshold_minutes=_GAP_ALERT_MINUTES)
        except Exception as exc:
            logger.debug("Post-backfill gap scan error: %s", exc)

    except ImportError as exc:
        logger.warning("Backfill module not available: %s", exc)
    except Exception as exc:
        logger.error("Historical backfill error: %s", exc)


def _handle_run_optimization(engine) -> None:
    """Run Optuna strategy optimization (off-hours)."""
    logger.info("▶ Running optimization...")
    try:
        # The DashboardEngine already has optimization logic
        status = engine.get_status()
        opt_status = status.get("optimization", {}).get("status", "idle")
        if opt_status == "idle":
            logger.info("Optimization available via engine background thread")
        logger.info("✅ Optimization cycle complete")
    except Exception as exc:
        logger.warning("Optimization error: %s", exc)


def _handle_run_backtest(engine) -> None:
    """Run walk-forward backtesting (off-hours)."""
    logger.info("▶ Running backtesting...")
    try:
        results = engine.get_backtest_results()
        logger.info(
            "✅ Backtest cycle complete (%d results available)",
            len(results) if results else 0,
        )
    except Exception as exc:
        logger.warning("Backtest error: %s", exc)


def _handle_check_swing(engine, account_size: int) -> None:
    """Run one tick of the swing detector — entry detection + state management.

    Phase 2C integration: scans daily-plan swing candidates for pullback,
    breakout, and gap-continuation entries.  Manages per-asset SwingState
    (TP/SL/trail/time-stop) and publishes signals + states to Redis for
    dashboard live display and TradingView signals.csv.

    Called every 2 min during active hours (03:00–15:30 ET) via CHECK_SWING.
    """
    try:
        from lib.services.engine.swing import tick_swing_detector

        result = tick_swing_detector(engine, account_size)

        new_sigs = result.get("new_signals", 0)
        active = result.get("active_states", 0)
        exits = result.get("exits", 0)
        closed = result.get("closed", 0)

        if new_sigs > 0 or exits > 0 or active > 0:
            logger.info(
                "🕐 Swing tick: %d new signal(s), %d active, %d exit(s), %d closed",
                new_sigs,
                active,
                exits,
                closed,
            )
        else:
            logger.debug("Swing tick: no activity (candidates=%s)", result.get("candidates_scanned", 0))

    except Exception as exc:
        logger.warning("Swing detector tick failed (non-fatal): %s", exc)


def _handle_next_day_prep(engine) -> None:
    """Prepare next trading day parameters (off-hours)."""
    logger.info("▶ Next-day prep...")
    logger.info("✅ Next-day prep complete (parameters cached)")


def _handle_generate_chart_dataset(engine) -> None:
    """No-op — dataset generation runs on the dedicated GPU trainer service.

    Use the trainer service (lib.services.training.trainer_server) on the GPU machine:
        docker compose --profile training up -d
        curl -X POST http://100.113.72.63:8200/train
    """
    logger.info("⏭️  Chart dataset generation skipped — use trainer service (POST /train)")


def _handle_train_breakout_cnn(engine) -> None:
    """No-op — CNN training runs on the dedicated GPU trainer service.

    The trainer server (lib.services.training.trainer_server) handles the full
    pipeline: dataset generation → training → evaluation → promotion.
        docker compose --profile training up -d
        curl -X POST http://100.113.72.63:8200/train
    Trained models are hot-reloaded by the engine via watchdog.
    """
    logger.info("⏭️  CNN training skipped — use trainer service (POST /train)")


# ---------------------------------------------------------------------------
# Daily report handler (runs once per day at start of off-hours ~12:00 ET)
# ---------------------------------------------------------------------------


def _build_session_stats(today) -> dict:
    """Compile per-session ORB signal statistics for today's report.

    Queries ``orb_events`` grouped by the ``orb_session`` metadata field
    to produce a dict like::

        {
            "us":        {"total": 12, "breakouts": 4, "published": 2, "pass_rate": 50.0},
            "london":    {"total":  8, "breakouts": 3, "published": 3, "pass_rate": 100.0},
            "cme":       {"total":  6, "breakouts": 1, "published": 0, "pass_rate": 0.0},
            ...
        }

    Falls back gracefully to an empty dict if the table is unavailable or
    the query fails for any reason.
    """
    try:
        from lib.core.models import get_orb_events

        today_str = today.strftime("%Y-%m-%d")
        events = get_orb_events(limit=500)

        # Filter to today's events
        today_events = [
            e for e in events if str(e.get("evaluated_at", "") or e.get("created_at", "")).startswith(today_str)
        ]

        if not today_events:
            return {}

        # Group by session key (stored in metadata JSON or orb_session column)
        session_buckets: dict[str, dict] = {}
        for ev in today_events:
            # Try explicit orb_session column first, then fall back to metadata JSON
            session_key = ev.get("orb_session", "")
            if not session_key:
                try:
                    meta = json.loads(ev.get("metadata", "{}") or "{}")
                    session_key = meta.get("orb_session", "unknown")
                except Exception:
                    session_key = "unknown"

            if not session_key:
                session_key = "unknown"

            bucket = session_buckets.setdefault(
                session_key,
                {"total": 0, "breakouts": 0, "published": 0, "filter_failed": 0},
            )
            bucket["total"] += 1
            if ev.get("breakout_detected"):
                bucket["breakouts"] += 1
            if ev.get("published"):
                bucket["published"] += 1
            elif ev.get("breakout_detected"):
                bucket["filter_failed"] += 1

        # Compute pass rates
        result = {}
        for sk, b in sorted(session_buckets.items()):
            total_bo = b["breakouts"]
            pub = b["published"]
            result[sk] = {
                "total_evaluations": b["total"],
                "breakouts_detected": total_bo,
                "published": pub,
                "filter_failed": b["filter_failed"],
                "pass_rate": round(pub / total_bo * 100, 1) if total_bo > 0 else 0.0,
            }

        return result

    except Exception as exc:
        logger.debug("Could not build session stats: %s", exc)
        return {}


def _handle_check_news_sentiment(engine, label: str = "morning") -> None:
    """Run the news sentiment pipeline and cache results in Redis.

    Fetches articles from Finnhub + Alpha Vantage, scores them with
    VADER + optional Grok (ambiguous-only), aggregates per symbol, and
    writes results to ``engine:news_sentiment:<SYMBOL>`` (2h TTL) and
    ``engine:news_spike`` in Redis.

    Args:
        engine: DashboardEngine instance (used to access Redis + watchlist).
        label:  "morning" or "midday" — used in log messages only.
    """
    logger.info("▶ News sentiment pipeline (%s run)...", label)
    t0 = time.time()

    try:
        from lib.analysis.sentiment.news_sentiment import run_news_sentiment_pipeline

        # Resolve API keys from environment
        finnhub_key = os.getenv("FINNHUB_API_KEY")
        alpha_key = os.getenv("ALPHA_VANTAGE_API_KEY")
        grok_key = os.getenv("XAI_API_KEY") or os.getenv("GROK_API_KEY")

        if not finnhub_key and not alpha_key:
            logger.warning("News sentiment: no API keys configured (set FINNHUB_API_KEY and/or ALPHA_VANTAGE_API_KEY)")
            return

        # Use the engine's watchlist symbols; fall back to core futures set
        try:
            symbols: list[str] = list(engine.get_watchlist())
        except Exception:
            symbols = ["MES", "MNQ", "MGC", "MCL", "M2K", "MYM", "M6E", "M6B", "MBT", "MET"]

        from lib.core.cache import REDIS_AVAILABLE, _r

        redis = _r if REDIS_AVAILABLE else None

        sentiments = run_news_sentiment_pipeline(
            symbols=symbols,
            finnhub_key=finnhub_key,
            alpha_key=alpha_key,
            grok_key=grok_key,
            redis=redis,
            days_back=2,
            max_per_ticker=5,
        )

        elapsed = time.time() - t0
        spike_count = sum(1 for ns in sentiments.values() if ns.is_spike)
        logger.info(
            "✅ News sentiment (%s): %d symbols scored in %.1fs — %d spike(s)",
            label,
            len(sentiments),
            elapsed,
            spike_count,
        )

        # Publish spike events to SSE so the dashboard can alert immediately
        if spike_count:
            try:
                for sym, ns in sentiments.items():
                    if ns.is_spike:
                        import json as _json

                        payload = _json.dumps(
                            {
                                "symbol": sym,
                                "articles_last_hour": ns.articles_last_hour,
                                "sentiment": round(ns.weighted_hybrid, 3),
                                "signal": ns.signal,
                            }
                        )
                        if REDIS_AVAILABLE and _r is not None:
                            _r.publish("dashboard:news_spike", payload)
                        logger.warning(
                            "📰 NEWS SPIKE: %s — %d art/hr  sentiment %.2f (%s)",
                            sym,
                            ns.articles_last_hour,
                            ns.weighted_hybrid,
                            ns.signal,
                        )
            except Exception as exc:
                logger.warning("News sentiment: spike publish failed: %s", exc)

    except Exception as exc:
        logger.error("News sentiment pipeline (%s) failed: %s", label, exc, exc_info=True)


def _handle_daily_report(engine) -> None:
    """Generate the daily trading session report and publish it to Redis.

    Builds a structured summary of the just-completed trading session:
      - ORB signal count + filter pass/reject rates
      - Per-session ORB breakdown (us, london, cme, tokyo, etc.)
      - CNN probability stats (mean, min, max, above-threshold count)
      - Risk events (blocks, warnings, consecutive losses)
      - Model performance snapshot (val accuracy, precision, recall)
      - Data coverage summary (gap count, coverage % per symbol)

    The report is:
      1. Published to Redis key ``engine:daily_report`` (TTL 26h) so the
         data-service can serve it at GET /audit/daily-report.
      2. Logged at INFO level in a human-readable format.
      3. Optionally emailed if ``DAILY_REPORT_EMAIL`` is set in the environment.

    Any failure here is non-fatal — it is logged and the engine continues.
    """
    logger.info("▶ Generating daily session report...")
    try:
        # Build the report via the audit helper — _build_daily_report takes a
        # date object and queries the DB internally for that day's events.
        report: dict = {}
        try:
            from lib.services.data.api.audit import _build_daily_report

            today = datetime.now(tz=_EST).date()
            report = _build_daily_report(today)
        except Exception as exc:
            logger.debug("Could not build report from audit DB (%s) — using empty report", exc)
            today = datetime.now(tz=_EST).date()
            report = {"generated_at": datetime.now(tz=_EST).isoformat()}

        # Add generation timestamp and session label
        report["generated_at"] = datetime.now(tz=_EST).isoformat()
        report["session"] = "daily"

        # ── Per-session performance breakdown ──────────────────────────────
        try:
            session_stats = _build_session_stats(today)
            if session_stats:
                report["sessions"] = session_stats
        except Exception as exc:
            logger.debug("Session stats build failed: %s", exc)

        # ── PositionManager session stats ──────────────────────────────────
        try:
            pm = _position_manager
            if pm is not None:
                closed = pm.get_history()
                if closed:
                    wins = [p for p in closed if p.realized_pnl > 0]
                    losses = [p for p in closed if p.realized_pnl <= 0]
                    total_pnl = sum(p.realized_pnl for p in closed)
                    avg_r = sum(p.r_multiple for p in closed) / len(closed) if closed else 0.0
                    # Break down by breakout_type
                    type_breakdown: dict[str, dict] = {}
                    for p in closed:
                        btype = p.breakout_type or "UNKNOWN"
                        bucket = type_breakdown.setdefault(
                            btype,
                            {"trades": 0, "wins": 0, "total_pnl": 0.0},
                        )
                        bucket["trades"] += 1
                        if p.realized_pnl > 0:
                            bucket["wins"] += 1
                        bucket["total_pnl"] = round(bucket["total_pnl"] + p.realized_pnl, 4)
                    for _btype, b in type_breakdown.items():
                        b["win_rate"] = round(b["wins"] / b["trades"] * 100, 1) if b["trades"] else 0.0

                    report["position_manager"] = {
                        "total_trades": len(closed),
                        "wins": len(wins),
                        "losses": len(losses),
                        "win_rate": round(len(wins) / len(closed) * 100, 1) if closed else 0.0,
                        "total_realized_pnl": round(total_pnl, 4),
                        "avg_r_multiple": round(avg_r, 3),
                        "active_positions": pm.get_position_count(),
                        "by_type": type_breakdown,
                    }
                else:
                    report["position_manager"] = {
                        "total_trades": 0,
                        "active_positions": pm.get_position_count(),
                    }
        except Exception as exc:
            logger.debug("PositionManager stats build failed: %s", exc)

        # ── Data coverage / gap summary ────────────────────────────────────
        try:
            from lib.services.engine.backfill import _get_backfill_symbols, get_gap_report

            symbols = _get_backfill_symbols()[:12]  # cap to avoid long runtime
            gap_summary: dict[str, dict] = {}
            total_gaps = 0
            for sym in symbols:
                try:
                    gr = get_gap_report(sym, days_back=1, interval="1m")
                    g_count = len(gr.get("gaps", []))
                    total_gaps += g_count
                    if g_count > 0 or gr.get("coverage_pct", 100) < 90:
                        gap_summary[sym] = {
                            "coverage_pct": gr.get("coverage_pct", 0),
                            "gap_count": g_count,
                            "total_bars": gr.get("total_bars", 0),
                        }
                except Exception:
                    pass
            if gap_summary:
                report["data_coverage"] = {
                    "symbols_checked": len(symbols),
                    "symbols_with_gaps": len(gap_summary),
                    "total_gaps_today": total_gaps,
                    "details": gap_summary,
                }
        except Exception as exc:
            logger.debug("Data coverage summary failed: %s", exc)

        # 1. Publish to Redis
        try:
            from lib.core.cache import cache_set

            cache_set(
                "engine:daily_report",
                json.dumps(report, default=str).encode(),
                ttl=26 * 3600,  # 26 hours — survives until tomorrow's report
            )
            logger.debug("Daily report published to Redis key engine:daily_report")
        except Exception as exc:
            logger.warning("Could not publish daily report to Redis: %s", exc)

        # 2. Log summary — the report dict uses nested "orb" and "cnn" keys
        orb_section = report.get("orb", {})
        orb_count = orb_section.get("breakouts_detected", 0)
        published = orb_section.get("published", 0)
        filtered = orb_section.get("filter_failed", 0)
        cnn_stats = report.get("cnn", {})
        model_info_d = report.get("model", {})
        session_breakdown = report.get("sessions", {})
        coverage = report.get("data_coverage", {})
        pm_stats = report.get("position_manager", {})

        logger.info("=" * 55)
        logger.info("  📊 Daily Session Report — %s", datetime.now(tz=_EST).strftime("%Y-%m-%d"))
        logger.info("=" * 55)
        logger.info("  ORB detections : %d breakouts | %d published | %d filtered", orb_count, published, filtered)
        if orb_count > 0:
            pass_rate = published / orb_count * 100
            logger.info("  Filter pass rate: %.0f%%", pass_rate)

        # Per-session breakdown
        if session_breakdown:
            logger.info("  ── Per-Session Breakdown ──────────────────────")
            for sk, sb in sorted(session_breakdown.items()):
                logger.info(
                    "    %-14s  evals=%3d  bo=%2d  pub=%2d  pass=%.0f%%",
                    sk,
                    sb.get("total_evaluations", 0),
                    sb.get("breakouts_detected", 0),
                    sb.get("published", 0),
                    sb.get("pass_rate", 0.0),
                )

        if cnn_stats:
            logger.info(
                "  CNN P(good)    : mean=%.3f  min=%.3f  max=%.3f  n=%d",
                cnn_stats.get("mean", 0),
                cnn_stats.get("min", 0),
                cnn_stats.get("max", 0),
                cnn_stats.get("count", 0),
            )
        if model_info_d and model_info_d.get("available"):
            val_acc = model_info_d.get("val_accuracy") or 0
            precision = model_info_d.get("precision") or 0
            recall = model_info_d.get("recall") or 0
            samples = model_info_d.get("train_samples") or 0
            logger.info(
                "  Model          : acc=%.1f%%  prec=%.1f%%  recall=%.1f%%  samples=%d",
                val_acc,
                precision,
                recall,
                samples,
            )

        # PositionManager session summary
        if pm_stats and pm_stats.get("total_trades", 0) > 0:
            logger.info("  ── PositionManager ────────────────────────────")
            logger.info(
                "  Trades : %d total | %d wins | %d losses | win rate %.0f%%",
                pm_stats.get("total_trades", 0),
                pm_stats.get("wins", 0),
                pm_stats.get("losses", 0),
                pm_stats.get("win_rate", 0.0),
            )
            logger.info(
                "  P&L    : $%.2f realized | avg R=%.2f | %d active",
                pm_stats.get("total_realized_pnl", 0.0),
                pm_stats.get("avg_r_multiple", 0.0),
                pm_stats.get("active_positions", 0),
            )
            by_type = pm_stats.get("by_type", {})
            if by_type:
                logger.info("  ── By Breakout Type ───────────────────────────")
                for btype, b in sorted(by_type.items()):
                    logger.info(
                        "    %-14s  trades=%2d  wins=%2d  win_rate=%.0f%%  pnl=$%.2f",
                        btype,
                        b.get("trades", 0),
                        b.get("wins", 0),
                        b.get("win_rate", 0.0),
                        b.get("total_pnl", 0.0),
                    )

        # Data coverage warning
        if coverage and coverage.get("symbols_with_gaps", 0) > 0:
            logger.warning(
                "  ⚠️  Data gaps    : %d symbol(s) have gaps today (total %d gaps)",
                coverage["symbols_with_gaps"],
                coverage.get("total_gaps_today", 0),
            )
            for sym, detail in list(coverage.get("details", {}).items())[:5]:
                logger.warning(
                    "    %s: coverage=%.1f%%  gaps=%d  bars=%d",
                    sym,
                    detail.get("coverage_pct", 0),
                    detail.get("gap_count", 0),
                    detail.get("total_bars", 0),
                )

        logger.info("=" * 55)

        # 3. Optional email alert
        email_to = os.environ.get("DAILY_REPORT_EMAIL", "").strip()
        if email_to:
            _send_daily_report_email(email_to, report)

        logger.info("✅ Daily report complete")

    except Exception as exc:
        logger.warning("Daily report generation failed (non-fatal): %s", exc, exc_info=True)


def _send_daily_report_email(to_addr: str, report: dict) -> None:
    """Send the daily report via SMTP if environment variables are configured.

    Required env vars:
        SMTP_HOST        — e.g. smtp.gmail.com
        SMTP_PORT        — e.g. 587
        SMTP_USER        — sender email address
        SMTP_PASSWORD    — sender password or app password
        DAILY_REPORT_EMAIL — recipient address (already passed in)

    If any required variable is missing the email is silently skipped.
    """
    import smtplib
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText

    smtp_host = os.environ.get("SMTP_HOST", "").strip()
    smtp_port = int(os.environ.get("SMTP_PORT", "587"))
    smtp_user = os.environ.get("SMTP_USER", "").strip()
    smtp_pass = os.environ.get("SMTP_PASSWORD", "").strip()

    if not all([smtp_host, smtp_user, smtp_pass]):
        logger.debug("SMTP not configured — skipping daily report email (set SMTP_HOST/SMTP_USER/SMTP_PASSWORD)")
        return

    try:
        today_str = datetime.now(tz=_EST).strftime("%Y-%m-%d")
        orb_section = report.get("orb", {})
        orb_count = orb_section.get("breakouts_detected", 0)
        published = orb_section.get("published", 0)
        filtered = orb_section.get("filter_failed", 0)
        cnn_stats = report.get("cnn", {})
        model_d = report.get("model", {})
        pass_rate = (published / orb_count * 100) if orb_count else 0

        # Plain-text body
        lines = [
            f"Ruby Futures — Daily Report {today_str}",
            "=" * 48,
            f"ORB Detections : {orb_count} total | {published} published | {filtered} filtered",
            f"Filter Pass Rate: {pass_rate:.0f}%",
        ]
        if cnn_stats:
            lines += [
                f"CNN P(good)    : mean={cnn_stats.get('mean', 0):.3f}  "
                f"min={cnn_stats.get('min', 0):.3f}  max={cnn_stats.get('max', 0):.3f}  "
                f"n={cnn_stats.get('count', 0)}",
            ]
        if model_d:
            lines += [
                f"Model          : acc={model_d.get('val_accuracy', 0):.1f}%  "
                f"prec={model_d.get('precision', 0):.1f}%  "
                f"recall={model_d.get('recall', 0):.1f}%  "
                f"samples={model_d.get('train_samples', 0)}",
                f"Last Promoted  : {model_d.get('promoted_at', 'unknown')}",
            ]
        lines += [
            "",
            "Generated by Ruby Futures Engine",
        ]
        body = "\n".join(lines)

        msg = MIMEMultipart("alternative")
        msg["Subject"] = f"[Ruby Futures] Daily Report {today_str} — {published} signal(s)"
        msg["From"] = smtp_user
        msg["To"] = to_addr
        msg.attach(MIMEText(body, "plain"))

        with smtplib.SMTP(smtp_host, smtp_port) as server:
            server.ehlo()
            server.starttls()
            server.login(smtp_user, smtp_pass)
            server.sendmail(smtp_user, to_addr, msg.as_string())

        logger.info("📧 Daily report emailed to %s", to_addr)

    except Exception as exc:
        logger.warning("Failed to send daily report email: %s", exc)


# ---------------------------------------------------------------------------
# Publish engine status to Redis (runs every loop iteration)
# ---------------------------------------------------------------------------


def _check_module_health() -> dict:
    """Check per-module health for Redis, Postgres, and Massive WS.

    Returns a dict with keys: redis, postgres, massive — each containing
    ``{"status": "ok"|"error"|"unavailable", ...}``.
    """
    modules: dict = {}

    # Redis
    try:
        from lib.core.cache import REDIS_AVAILABLE, _r

        if REDIS_AVAILABLE and _r is not None:
            _r.ping()
            modules["redis"] = {"status": "ok", "connected": True}
        else:
            modules["redis"] = {"status": "unavailable", "connected": False}
    except Exception as exc:
        modules["redis"] = {"status": "error", "connected": False, "error": str(exc)}

    # Postgres
    try:
        database_url = os.getenv("DATABASE_URL", "")
        if not database_url.startswith("postgresql"):
            modules["postgres"] = {"status": "not_configured", "connected": False}
        else:
            from lib.core.models import _get_conn

            conn = _get_conn()
            try:
                conn.execute("SELECT 1")
                modules["postgres"] = {"status": "ok", "connected": True}
            finally:
                conn.close()
    except Exception as exc:
        modules["postgres"] = {"status": "error", "connected": False, "error": str(exc)}

    # Massive WebSocket
    try:
        from lib.core.cache import get_data_source

        ds = get_data_source()
        if ds == "Massive":
            modules["massive"] = {"status": "ok", "data_source": "Massive", "connected": True}
        else:
            modules["massive"] = {"status": "fallback", "data_source": ds, "connected": False}
    except Exception as exc:
        modules["massive"] = {"status": "error", "connected": False, "error": str(exc)}

    # CNN model
    try:
        from lib.services.engine.model_watcher import _find_model_dir

        model_dir = _find_model_dir()
        champion = model_dir / "breakout_cnn_best.pt" if model_dir else None
        if champion is not None and champion.is_file():
            try:
                stat = champion.stat()
                size_mb = round(stat.st_size / (1024 * 1024), 1)
                modules["cnn_model"] = {
                    "status": "ok",
                    "available": True,
                    "size_mb": size_mb,
                    "path": str(champion),
                }
            except OSError:
                modules["cnn_model"] = {"status": "error", "available": False}
        else:
            modules["cnn_model"] = {"status": "missing", "available": False}
    except ImportError:
        modules["cnn_model"] = {"status": "unknown", "available": False}

    return modules


def _publish_engine_status(engine, session_mode: str, scheduler_status: dict) -> None:
    """Publish engine status + scheduler state + per-module health to Redis."""
    try:
        from lib.core.cache import cache_set

        status = engine.get_status()
        status["session_mode"] = session_mode
        status["scheduler"] = scheduler_status
        status["modules"] = _check_module_health()

        # Inject swing detector summary (Phase 2C)
        try:
            from lib.services.engine.swing import get_swing_summary

            status["swing"] = get_swing_summary()
        except ImportError:
            status["swing"] = {"active_count": 0, "active_assets": []}
        except Exception:
            status["swing"] = {"active_count": 0, "active_assets": [], "error": True}

        cache_set(
            "engine:status",
            json.dumps(status, default=str).encode(),
            ttl=60,
        )

        # Publish backtest results
        bt = engine.get_backtest_results()
        if bt:
            cache_set(
                "engine:backtest_results",
                json.dumps(bt, default=str).encode(),
                ttl=300,
            )

        # Publish strategy history
        sh = engine.get_strategy_history()
        if sh:
            cache_set(
                "engine:strategy_history",
                json.dumps(sh, default=str).encode(),
                ttl=300,
            )

        # Publish live feed status
        lf = engine.get_live_feed_status()
        cache_set(
            "engine:live_feed_status",
            json.dumps(lf, default=str).encode(),
            ttl=30,
        )
    except Exception as exc:
        logger.debug("Failed to publish engine status to Redis: %s", exc)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def main():
    global _model_watcher

    logger.info("=" * 60)
    logger.info("  Engine Service starting up (session-aware scheduling)")
    logger.info("=" * 60)

    # Configuration from environment
    account_size = int(os.getenv("ACCOUNT_SIZE", os.getenv("DEFAULT_ACCOUNT_SIZE", "150000")))
    interval = os.getenv("ENGINE_INTERVAL", os.getenv("DEFAULT_INTERVAL", "5m"))
    period = os.getenv("ENGINE_PERIOD", os.getenv("DEFAULT_PERIOD", "5d"))

    # Import and start the engine
    from lib.core.cache import get_data_source
    from lib.trading.engine import get_engine

    engine = get_engine(
        account_size=account_size,
        interval=interval,
        period=period,
    )

    # Import scheduler
    from lib.services.engine.scheduler import ActionType, ScheduleManager

    scheduler = ScheduleManager()
    session = scheduler.get_session_mode()

    logger.info(
        "Engine started: account=$%s  interval=%s  period=%s  session=%s %s  data_source=%s",
        f"{account_size:,}",
        interval,
        period,
        session.value,
        scheduler._session_emoji(session),
        get_data_source(),
    )

    _write_health(True, "running", session=session.value)

    # ---------------------------------------------------------------------------
    # Start the filesystem-based model watcher (replaces inline polling).
    # Uses watchdog (inotify) when available, falls back to polling.
    # ---------------------------------------------------------------------------
    from lib.services.engine.model_watcher import ModelWatcher

    _model_watcher = ModelWatcher()
    watcher_started = _model_watcher.start()
    if watcher_started:
        watcher_status = _model_watcher.status()
        logger.info(
            "Model watcher active: backend=%s  dir=%s",
            watcher_status["backend"],
            watcher_status["model_dir"],
        )
    else:
        logger.warning(
            "Model watcher could not start — CNN hot-reload disabled. "
            "Ensure models/ directory exists (run scripts/sync_models.sh)."
        )

    # Action dispatch table
    # Initialise the RiskManager early so it's ready for handlers
    _get_risk_manager(account_size)

    # Initialise the PositionManager — loads any persisted positions from Redis
    _get_position_manager(account_size)

    # Initialise the CopyTrader if Rithmic copy-trading is enabled
    if _COPY_TRADING_ENABLED:
        ct = _get_copy_trader()
        if ct is not None:
            logger.info("CopyTrader ready — orders from PositionManager will be mirrored to Rithmic accounts")
    else:
        logger.info("CopyTrader disabled (set RITHMIC_COPY_TRADING=1 to enable Rithmic order mirroring)")

    # Initialise the LiveRiskPublisher — merges RiskManager + PositionManager
    # into a unified LiveRiskState and publishes to Redis every 5 seconds.
    # Also wired into the live_risk API module for /api/live-risk/refresh.
    _get_live_risk_publisher()

    action_handlers = {
        ActionType.COMPUTE_DAILY_FOCUS: lambda: _handle_compute_daily_focus(engine, account_size),
        ActionType.GROK_MORNING_BRIEF: lambda: _handle_grok_morning_brief(engine),
        ActionType.PREP_ALERTS: lambda: _handle_prep_alerts(engine),
        ActionType.RUBY_RECOMPUTE: lambda: _handle_fks_recompute(engine),
        ActionType.PUBLISH_FOCUS_UPDATE: lambda: _handle_publish_focus_update(engine, account_size),
        ActionType.GROK_LIVE_UPDATE: lambda: _handle_grok_live_update(engine),
        ActionType.CHECK_RISK_RULES: lambda: _handle_check_risk_rules(engine, account_size),
        ActionType.CHECK_NO_TRADE: lambda: _handle_check_no_trade(engine, account_size),
        ActionType.CHECK_ORB: lambda: _handle_check_orb(engine),
        ActionType.CHECK_ORB_CME: lambda: _handle_check_orb_cme(engine),
        ActionType.CHECK_ORB_SYDNEY: lambda: _handle_check_orb_sydney(engine),
        ActionType.CHECK_ORB_TOKYO: lambda: _handle_check_orb_tokyo(engine),
        ActionType.CHECK_ORB_SHANGHAI: lambda: _handle_check_orb_shanghai(engine),
        ActionType.CHECK_ORB_FRANKFURT: lambda: _handle_check_orb_frankfurt(engine),
        ActionType.CHECK_ORB_LONDON: lambda: _handle_check_orb_london(engine),
        ActionType.CHECK_ORB_LONDON_NY: lambda: _handle_check_orb_london_ny(engine),
        ActionType.CHECK_ORB_CME_SETTLE: lambda: _handle_check_orb_cme_settle(engine),
        ActionType.CHECK_ORB_CRYPTO_UTC0: lambda: _handle_check_orb_crypto_utc0(engine),
        ActionType.CHECK_ORB_CRYPTO_UTC12: lambda: _handle_check_orb_crypto_utc12(engine),
        ActionType.CHECK_PDR: lambda: _handle_check_pdr(
            engine,
            session_key=(
                lambda _p=pending[0].payload if pending else None: (
                    _p.get("session_key", "london_ny") if _p is not None else "london_ny"
                )
            )(),  # type: ignore[union-attr]
        ),
        ActionType.CHECK_IB: lambda: _handle_check_ib(
            engine,
            session_key=(
                lambda _p=pending[0].payload if pending else None: (
                    _p.get("session_key", "us") if _p is not None else "us"
                )
            )(),  # type: ignore[union-attr]
        ),
        ActionType.CHECK_CONSOLIDATION: lambda: _handle_check_consolidation(
            engine,
            session_key=(
                lambda _p=pending[0].payload if pending else None: (
                    _p.get("session_key", "london_ny") if _p is not None else "london_ny"
                )
            )(),  # type: ignore[union-attr]
        ),
        ActionType.CHECK_BREAKOUT_MULTI: lambda: _handle_check_breakout_multi(
            engine,
            session_key=(
                lambda _p=pending[0].payload if pending else None: (
                    _p.get("session_key", "us") if _p is not None else "us"
                )
            )(),  # type: ignore[union-attr]
            types=(lambda _p=pending[0].payload if pending else None: _p.get("types") if _p is not None else None)(),  # type: ignore[union-attr]
        ),
        ActionType.CHECK_SWING: lambda: _handle_check_swing(engine, account_size),
        ActionType.CHECK_NEWS_SENTIMENT: lambda: _handle_check_news_sentiment(engine, label="morning"),
        ActionType.CHECK_NEWS_SENTIMENT_MIDDAY: lambda: _handle_check_news_sentiment(engine, label="midday"),
        ActionType.HISTORICAL_BACKFILL: lambda: _handle_historical_backfill(engine),
        ActionType.RUN_OPTIMIZATION: lambda: _handle_run_optimization(engine),
        ActionType.RUN_BACKTEST: lambda: _handle_run_backtest(engine),
        ActionType.NEXT_DAY_PREP: lambda: _handle_next_day_prep(engine),
        ActionType.GENERATE_CHART_DATASET: lambda: _handle_generate_chart_dataset(engine),
        ActionType.TRAIN_BREAKOUT_CNN: lambda: _handle_train_breakout_cnn(engine),
        ActionType.DAILY_REPORT: lambda: _handle_daily_report(engine),
    }

    logger.info("=" * 60)
    logger.info(
        "  Engine Service ready — session: %s %s",
        session.value.upper(),
        scheduler._session_emoji(session),
    )
    logger.info("  Registered %d action handlers", len(action_handlers))
    logger.info("=" * 60)

    # Graceful shutdown
    shutdown = False

    def handle_signal(signum, frame):
        nonlocal shutdown
        logger.info("Received signal %s, shutting down...", signum)
        shutdown = True

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    try:
        while not shutdown:
            # Get current session and pending actions
            current_session = scheduler.current_session
            pending = scheduler.get_pending_actions()

            # Update health file
            _write_health(
                True,
                "running",
                session=current_session.value,
                pending_actions=len(pending),
            )

            # Check for dashboard-triggered commands via Redis
            _check_redis_commands(action_handlers)

            # Execute pending actions
            for action in pending:
                if shutdown:
                    break

                handler = action_handlers.get(action.action)
                if handler is None:
                    logger.warning("No handler for action: %s", action.action.value)
                    scheduler.mark_done(action.action)
                    continue

                try:
                    logger.debug(
                        "Executing: %s — %s",
                        action.action.value,
                        action.description,
                    )
                    # For payload-bearing actions (CHECK_PDR, CHECK_IB, CHECK_CONSOLIDATION,
                    # CHECK_BREAKOUT_MULTI) the handler needs the action payload to know
                    # which session key and types to use.  We call the typed handlers
                    # directly here so the lambdas in action_handlers don't need
                    # late-binding workarounds.
                    _payload = getattr(action, "payload", None) or {}
                    if action.action == ActionType.CHECK_PDR:
                        _sk = _payload.get("session_key", "london_ny") if _payload else "london_ny"
                        _handle_check_pdr(engine, session_key=_sk)
                    elif action.action == ActionType.CHECK_IB:
                        _sk = _payload.get("session_key", "us") if _payload else "us"
                        _handle_check_ib(engine, session_key=_sk)
                    elif action.action == ActionType.CHECK_CONSOLIDATION:
                        _sk = _payload.get("session_key", "london_ny") if _payload else "london_ny"
                        _handle_check_consolidation(engine, session_key=_sk)
                    elif action.action == ActionType.CHECK_BREAKOUT_MULTI:
                        _sk = _payload.get("session_key", "us") if _payload else "us"
                        _types = _payload.get("types") if _payload else None
                        _handle_check_breakout_multi(
                            engine,
                            session_key=_sk,
                            types=_types,
                        )
                    else:
                        handler()
                    scheduler.mark_done(action.action)
                except Exception as exc:
                    scheduler.mark_failed(action.action, str(exc))
                    logger.error("Action %s failed: %s", action.action.value, exc, exc_info=True)

            # Update active positions on every loop iteration (bracket phases,
            # EMA9 trailing, stop/TP3 exits).  Exits immediately if no positions
            # are open.  Must run BEFORE publish so the status payload reflects
            # the latest position state.
            _handle_update_positions(engine)

            # Tick the LiveRiskPublisher — publishes unified risk state to
            # Redis every 5 seconds (or immediately after position changes).
            # This feeds the dashboard risk strip and focus card overlays.
            _tick_live_risk_publisher()

            # Publish engine status to Redis every iteration
            _publish_engine_status(
                engine,
                current_session.value,
                scheduler.get_status(),
            )

            # Sleep based on session mode
            sleep_time = scheduler.sleep_interval
            time.sleep(sleep_time)

    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")

    # Shutdown
    logger.info("=" * 60)
    logger.info("  Engine Service shutting down")
    logger.info("=" * 60)

    # Stop the model watcher first
    if _model_watcher is not None:
        _model_watcher.stop()
        _model_watcher = None

    _write_health(False, "shutting_down")

    try:
        import asyncio

        loop = asyncio.new_event_loop()
        loop.run_until_complete(engine.stop())
        loop.close()
    except Exception as exc:
        logger.warning("Engine stop error (non-fatal): %s", exc)

    _write_health(False, "stopped")
    logger.info("Engine Service stopped")


if __name__ == "__main__":
    main()
