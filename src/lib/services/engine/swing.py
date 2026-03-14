"""
Swing Engine Adapter — Phase 2C Integration
============================================
Bridges the swing detector (pure computation) with the engine's live data
layer, Redis state persistence, and dashboard publishing.

Responsibilities:
  1. Fetch intraday bars, biases, ATRs, prices for swing candidates
  2. Call detect_swing_entries() / evaluate_swing_exits() per asset
  3. Manage per-asset SwingState across engine ticks
  4. Publish swing signals + states to Redis for dashboard SSE
  5. Publish swing signals to TradingView signal store for signals.csv

The main entry point is ``tick_swing_detector(engine, account_size)`` which
the engine calls on every CHECK_SWING scheduled action (every 2 min during
active hours, 03:00–15:30 ET).

State lifecycle:
  - Signals detected → published to Redis (dashboard:swing_update)
  - ENTRY_READY signals → SwingState created, persisted
  - Active states ticked on each call → exits evaluated, states updated
  - Closed states archived, removed from active dict

Usage:
    from lib.services.engine.swing import tick_swing_detector

    # In the engine main loop handler for CHECK_SWING:
    _handle_check_swing = lambda: tick_swing_detector(engine, account_size)
"""

from __future__ import annotations

import contextlib
import io
import json
import logging
import time
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

logger = logging.getLogger("engine.swing")

_EST = ZoneInfo("America/New_York")

# ---------------------------------------------------------------------------
# Module-level state — persists across engine ticks
# ---------------------------------------------------------------------------

# {asset_name: SwingState} — active swing trades being managed
_active_swing_states: dict[str, Any] = {}

# Timestamp of last full scan (to avoid redundant work)
_last_scan_ts: float = 0.0

# Minimum seconds between full scans (safety net — scheduler already gates)
_MIN_SCAN_INTERVAL = 90.0

# Maximum concurrent swing trades across all assets
_MAX_CONCURRENT_SWINGS = 3

# ---------------------------------------------------------------------------
# Helpers — data fetching
# ---------------------------------------------------------------------------


def _get_redis():
    """Get the Redis client (lazy import, returns None if unavailable)."""
    try:
        from lib.core.cache import REDIS_AVAILABLE, _r

        if REDIS_AVAILABLE and _r is not None:
            return _r
    except ImportError:
        pass
    return None


def _cache_get(key: str) -> str | None:
    """Read a key from the cache layer."""
    try:
        from lib.core.cache import cache_get

        raw = cache_get(key)
        if raw is None:
            return None
        return raw.decode("utf-8") if isinstance(raw, bytes) else raw
    except Exception:
        return None


def _cache_set(key: str, value: str, ttl: int = 300) -> None:
    """Write a key to the cache layer."""
    try:
        from lib.core.cache import cache_set

        cache_set(key, value.encode() if isinstance(value, str) else value, ttl=ttl)
    except Exception as exc:
        logger.debug("_cache_set(%s) error: %s", key, exc)


def _get_daily_plan_data() -> dict[str, Any] | None:
    """Load the current daily plan from Redis."""
    raw = _cache_get("engine:daily_plan")
    if not raw:
        return None
    try:
        return json.loads(raw)
    except Exception:
        return None


def _get_focus_data() -> dict[str, Any] | None:
    """Load the current daily focus from Redis."""
    raw = _cache_get("engine:daily_focus")
    if not raw:
        return None
    try:
        return json.loads(raw)
    except Exception:
        return None


def _get_swing_candidate_names() -> list[str]:
    """Get swing candidate asset names from the daily plan."""
    plan = _get_daily_plan_data()
    if plan:
        return plan.get("swing_candidate_names", [])

    # Fallback: check the dedicated key
    raw = _cache_get("engine:swing_assets")
    if raw:
        try:
            return json.loads(raw)
        except Exception:
            pass
    return []


def _get_swing_candidates_data() -> list[dict[str, Any]]:
    """Get full swing candidate dicts from the daily plan."""
    plan = _get_daily_plan_data()
    if plan:
        return plan.get("swing_candidates", [])
    return []


def _fetch_bars_15m(ticker_or_symbol: str) -> pd.DataFrame | None:
    """Fetch 15-minute bars for swing analysis.

    Tries cache first (engine:bars_15m:<ticker>), then falls back to
    resampling 5-minute bars if available, then 1-minute bars.
    """
    # Try 15m bars from cache
    raw = _cache_get(f"engine:bars_15m:{ticker_or_symbol}")
    if raw:
        try:
            return pd.read_json(io.StringIO(raw))
        except Exception:
            pass

    # Try 5m bars and resample to 15m
    raw = _cache_get(f"engine:bars_5m:{ticker_or_symbol}")
    if not raw:
        # Also try the standard data format
        try:
            from lib.core.cache import get_data
            from lib.core.models import ASSETS

            # Resolve ticker
            ticker = ticker_or_symbol
            for name, t in ASSETS.items():
                if name == ticker_or_symbol or t == ticker_or_symbol:
                    ticker = t
                    break

            df = get_data(ticker, "5m", "5d")
            if df is not None and not df.empty:
                return _resample_to_15m(df)
        except Exception:
            pass
        return None

    try:
        df_5m = pd.read_json(io.StringIO(raw))
        return _resample_to_15m(df_5m)
    except Exception:
        return None


def _fetch_bars_5m(ticker_or_symbol: str) -> pd.DataFrame | None:
    """Fetch 5-minute bars (used for swing entries with finer granularity)."""
    raw = _cache_get(f"engine:bars_5m:{ticker_or_symbol}")
    if raw:
        try:
            return pd.read_json(io.StringIO(raw))
        except Exception:
            pass

    try:
        from lib.core.cache import get_data
        from lib.core.models import ASSETS

        ticker = ticker_or_symbol
        for name, t in ASSETS.items():
            if name == ticker_or_symbol or t == ticker_or_symbol:
                ticker = t
                break

        df = get_data(ticker, "5m", "5d")
        if df is not None and not df.empty:
            return df
    except Exception:
        pass
    return None


def _resample_to_15m(df: pd.DataFrame) -> pd.DataFrame | None:
    """Resample 5m (or 1m) bars to 15m for the swing detector."""
    try:
        if df is None or df.empty:
            return None

        # Ensure DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            if "Datetime" in df.columns:
                df = df.set_index("Datetime")
            elif "datetime" in df.columns:
                df = df.set_index("datetime")
            elif "Date" in df.columns:
                df = df.set_index("Date")

        if not isinstance(df.index, pd.DatetimeIndex):
            # Last resort: try to parse the index
            df.index = pd.to_datetime(df.index)

        # Standard OHLCV column names
        ohlcv_map: dict[str, str] = {}
        for col in df.columns:
            cl = col.lower()
            if cl == "open":
                ohlcv_map["Open"] = col
            elif cl == "high":
                ohlcv_map["High"] = col
            elif cl == "low":
                ohlcv_map["Low"] = col
            elif cl == "close":
                ohlcv_map["Close"] = col
            elif cl in ("volume", "vol"):
                ohlcv_map["Volume"] = col

        if not all(k in ohlcv_map for k in ("Open", "High", "Low", "Close")):
            return df  # Can't resample, return as-is

        agg_dict: dict[str, str] = {
            ohlcv_map["Open"]: "first",
            ohlcv_map["High"]: "max",
            ohlcv_map["Low"]: "min",
            ohlcv_map["Close"]: "last",
        }
        if "Volume" in ohlcv_map:
            agg_dict[ohlcv_map["Volume"]] = "sum"

        result = df.resample("15min").agg(agg_dict).dropna()
        resampled = pd.DataFrame(result)

        return resampled if not resampled.empty else None
    except Exception as exc:
        logger.debug("_resample_to_15m error: %s", exc)
        return None


def _get_asset_ticker(name: str) -> str | None:
    """Resolve asset name to ticker symbol."""
    try:
        from lib.core.models import ASSETS

        return ASSETS.get(name)
    except ImportError:
        return None


def _get_contract_specs(name: str) -> dict[str, Any]:
    """Get micro contract specs for an asset."""
    try:
        from lib.core.models import MICRO_CONTRACT_SPECS

        return MICRO_CONTRACT_SPECS.get(name, {})
    except ImportError:
        return {}


def _get_asset_price(name: str, focus_data: dict[str, Any] | None = None) -> float:
    """Get the latest price for an asset from focus data or cache."""
    # Try focus data first (fastest, most recent)
    if focus_data:
        for asset in focus_data.get("assets", []):
            if asset.get("symbol") == name:
                price = asset.get("last_price", 0.0)
                if price > 0:
                    return float(price)

    # Try per-asset cache key
    raw = _cache_get(f"engine:price:{name}")
    if raw:
        try:
            return float(raw)
        except (ValueError, TypeError):
            pass

    # Try fetching from data layer
    ticker = _get_asset_ticker(name)
    if ticker:
        try:
            from lib.core.cache import get_data

            df = get_data(ticker, "5m", "1d")
            if df is not None and not df.empty:
                return float(df["Close"].iloc[-1])
        except Exception:
            pass

    return 0.0


def _get_asset_atr(name: str, focus_data: dict[str, Any] | None = None) -> float:
    """Get ATR for an asset from focus data or swing candidate data."""
    # Try focus data
    if focus_data:
        for asset in focus_data.get("assets", []):
            if asset.get("symbol") == name:
                atr = asset.get("atr", 0.0)
                if atr > 0:
                    return float(atr)

    # Try swing candidate data from daily plan
    for cand in _get_swing_candidates_data():
        if cand.get("asset_name") == name:
            atr = cand.get("atr", 0.0)
            if atr > 0:
                return float(atr)

    return 0.0


def _get_asset_bias(name: str) -> Any:
    """Get DailyBias for an asset from the daily plan."""
    plan = _get_daily_plan_data()
    if not plan:
        return None

    all_biases = plan.get("all_biases", {})
    bias_data = all_biases.get(name)
    if not bias_data:
        return None

    try:
        from lib.trading.strategies.daily.bias_analyzer import (
            BiasDirection,
            CandlePattern,
            DailyBias,
            KeyLevels,
        )

        # Reconstruct DailyBias from dict
        key_levels_data = bias_data.get("key_levels", {})
        key_levels = KeyLevels(
            prior_day_high=key_levels_data.get("prior_day_high", 0.0),
            prior_day_low=key_levels_data.get("prior_day_low", 0.0),
            prior_day_mid=key_levels_data.get("prior_day_mid", 0.0),
            prior_day_close=key_levels_data.get("prior_day_close", 0.0),
            weekly_high=key_levels_data.get("weekly_high", 0.0),
            weekly_low=key_levels_data.get("weekly_low", 0.0),
            weekly_mid=key_levels_data.get("weekly_mid", 0.0),
            monthly_ema20=key_levels_data.get("monthly_ema20", 0.0),
            overnight_high=key_levels_data.get("overnight_high", 0.0),
            overnight_low=key_levels_data.get("overnight_low", 0.0),
        )

        direction_str = bias_data.get("direction", "NEUTRAL")
        try:
            direction = BiasDirection(direction_str)
        except ValueError:
            direction = BiasDirection.NEUTRAL

        candle_str = bias_data.get("candle_pattern", "neutral_candle")
        try:
            candle_pattern = CandlePattern(candle_str)
        except ValueError:
            candle_pattern = CandlePattern.NEUTRAL_CANDLE

        return DailyBias(
            asset_name=name,
            direction=direction,
            confidence=bias_data.get("confidence", 0.0),
            reasoning=bias_data.get("reasoning", ""),
            key_levels=key_levels,
            candle_pattern=candle_pattern,
            weekly_range_position=bias_data.get("weekly_range_position", 0.5),
            monthly_trend_score=bias_data.get("monthly_trend_score", 0.0),
            volume_confirmation=bias_data.get("volume_confirmation", False),
            overnight_gap_direction=bias_data.get("overnight_gap_direction", 0.0),
            overnight_gap_atr_ratio=bias_data.get("overnight_gap_atr_ratio", 0.0),
            atr_expanding=bias_data.get("atr_expanding", False),
            component_scores=bias_data.get("component_scores", {}),
        )
    except Exception as exc:
        logger.debug("Failed to reconstruct DailyBias for %s: %s", name, exc)
        return None


def _get_session_open_price(name: str) -> float | None:
    """Get today's session open price for gap detection."""
    ticker = _get_asset_ticker(name)
    if not ticker:
        return None

    try:
        from lib.core.cache import get_data

        df = get_data(ticker, "1d", "2d")
        if df is not None and len(df) >= 1:
            return float(df["Open"].iloc[-1])
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# State management — load/save from Redis
# ---------------------------------------------------------------------------

_REDIS_KEY_SWING_STATES = "engine:swing_states"
_REDIS_KEY_SWING_SIGNALS = "engine:swing_signals"
_REDIS_KEY_SWING_HISTORY = "engine:swing_history"
_REDIS_PUBSUB_SWING = "dashboard:swing_update"


def _publish_swing_action(action: str, asset_name: str, **kwargs: Any) -> None:
    """Publish a structured swing action event to Redis PubSub.

    Sends a JSON payload with action type, asset name, and extra metadata
    so the SSE JS listener can show rich messages in the market events feed.

    Args:
        action: Action type — 'accepted', 'ignored', 'closed', 'stop_moved',
                'stop_updated', 'signal_detected', 'exit', 'state_created'.
        asset_name: The asset name (e.g. 'Gold', 'Crude Oil').
        **kwargs: Additional metadata (direction, phase, confidence, reason, etc.)
    """
    try:
        payload = json.dumps(
            {"action": action, "asset_name": asset_name, **kwargs},
            default=str,
        )
        r = _get_redis()
        if r is not None:
            r.publish(_REDIS_PUBSUB_SWING, payload)
    except Exception as exc:
        logger.debug("Failed to publish swing action event: %s", exc)


def _load_swing_states_from_redis() -> dict[str, Any]:
    """Load persisted swing states from Redis on startup."""
    global _active_swing_states

    raw = _cache_get(_REDIS_KEY_SWING_STATES)
    if not raw:
        return {}

    try:
        from lib.trading.strategies.daily.swing_detector import (
            SwingEntryStyle,
            SwingPhase,
            SwingSignal,
            SwingState,
        )

        data = json.loads(raw)
        states = {}
        for name, sd in data.items():
            if sd.get("phase") == SwingPhase.CLOSED.value:
                continue  # Don't reload closed states

            sig_data = sd.get("signal")
            signal = None
            if sig_data:
                signal = SwingSignal(
                    asset_name=sig_data.get("asset_name", ""),
                    entry_style=SwingEntryStyle(sig_data.get("entry_style", "pullback_entry")),
                    direction=sig_data.get("direction", "LONG"),
                    confidence=sig_data.get("confidence", 0.0),
                    entry_price=sig_data.get("entry_price", 0.0),
                    entry_zone_low=sig_data.get("entry_zone_low", 0.0),
                    entry_zone_high=sig_data.get("entry_zone_high", 0.0),
                    stop_loss=sig_data.get("stop_loss", 0.0),
                    tp1=sig_data.get("tp1", 0.0),
                    tp2=sig_data.get("tp2", 0.0),
                    atr=sig_data.get("atr", 0.0),
                    risk_reward_tp1=sig_data.get("risk_reward_tp1", 0.0),
                    risk_reward_tp2=sig_data.get("risk_reward_tp2", 0.0),
                    risk_dollars=sig_data.get("risk_dollars", 0.0),
                    position_size=sig_data.get("position_size", 1),
                    reasoning=sig_data.get("reasoning", ""),
                    key_level_used=sig_data.get("key_level_used", ""),
                    key_level_price=sig_data.get("key_level_price", 0.0),
                    detected_at=sig_data.get("detected_at", ""),
                    phase=SwingPhase(sig_data.get("phase", "watching")),
                )

            state = SwingState(
                asset_name=name,
                signal=signal,
                phase=SwingPhase(sd.get("phase", "watching")),
                entry_price=sd.get("entry_price", 0.0),
                current_stop=sd.get("current_stop", 0.0),
                tp1=sd.get("tp1", 0.0),
                tp2=sd.get("tp2", 0.0),
                direction=sd.get("direction", "LONG"),
                position_size=sd.get("position_size", 1),
                remaining_size=sd.get("remaining_size", 1),
                highest_price=sd.get("highest_price", 0.0),
                lowest_price=sd.get("lowest_price", float("inf")),
                entry_time=sd.get("entry_time", ""),
                last_update=sd.get("last_update", ""),
            )
            states[name] = state

        return states

    except Exception as exc:
        logger.warning("Failed to load swing states from Redis: %s", exc)
        return {}


def _persist_swing_states() -> None:
    """Persist current swing states to Redis."""
    global _active_swing_states

    try:
        payload = json.dumps(
            {name: st.to_dict() for name, st in _active_swing_states.items()},
            default=str,
        )
        _cache_set(_REDIS_KEY_SWING_STATES, payload, ttl=18 * 3600)
    except Exception as exc:
        logger.debug("Failed to persist swing states: %s", exc)


def _publish_swing_signals_to_redis(signals: list[Any]) -> None:
    """Publish detected swing signals to Redis for dashboard SSE."""
    if not signals:
        return

    try:
        payload = json.dumps(
            [s.to_dict() for s in signals],
            default=str,
        )
        _cache_set(_REDIS_KEY_SWING_SIGNALS, payload, ttl=18 * 3600)

        r = _get_redis()
        if r is not None:
            r.publish(_REDIS_PUBSUB_SWING, payload)

        logger.info(
            "📡 Published %d swing signal(s) to Redis",
            len(signals),
        )
    except Exception as exc:
        logger.debug("Failed to publish swing signals: %s", exc)


def _publish_swing_states_to_redis() -> None:
    """Publish active swing states to Redis."""
    global _active_swing_states

    try:
        payload = json.dumps(
            {name: st.to_dict() for name, st in _active_swing_states.items()},
            default=str,
        )
        _cache_set(_REDIS_KEY_SWING_STATES, payload, ttl=18 * 3600)

        r = _get_redis()
        if r is not None:
            r.publish(_REDIS_PUBSUB_SWING, payload)
    except Exception as exc:
        logger.debug("Failed to publish swing states: %s", exc)


def _archive_closed_state(name: str, state: Any) -> None:
    """Archive a closed swing state to Redis history list."""
    try:
        r = _get_redis()
        if r is None:
            return

        entry = json.dumps(
            {
                "asset_name": name,
                "state": state.to_dict(),
                "closed_at": datetime.now(tz=_EST).isoformat(),
            },
            default=str,
        )
        r.lpush(_REDIS_KEY_SWING_HISTORY, entry)
        r.ltrim(_REDIS_KEY_SWING_HISTORY, 0, 99)  # Keep last 100
        r.expire(_REDIS_KEY_SWING_HISTORY, 7 * 24 * 3600)  # 7 days
    except Exception as exc:
        logger.debug("Failed to archive swing state: %s", exc)


# ---------------------------------------------------------------------------
# Main tick function — called by the engine on CHECK_SWING
# ---------------------------------------------------------------------------


def tick_swing_detector(engine: Any, account_size: int) -> dict[str, Any]:
    """Run one tick of the swing detector.

    This is the main entry point called by the engine's CHECK_SWING handler.
    It performs two phases:

    Phase A — Entry Detection:
      For each swing candidate from the daily plan that doesn't already have
      an active SwingState, fetch bars/bias/ATR/price and run
      detect_swing_entries(). Signals are published to Redis.

    Phase B — Active State Management:
      For each active SwingState, fetch latest price and bars, call
      update_swing_state() to evaluate exits (TP1/TP2/SL/trail/time-stop).
      Updated states are published to Redis.

    Args:
        engine: The DashboardEngine instance (for data access if needed).
        account_size: Account size for position sizing.

    Returns:
        Summary dict with counts and status.
    """
    global _active_swing_states, _last_scan_ts

    now_ts = time.time()
    datetime.now(tz=_EST)

    # Rate-limit scans (defensive — scheduler already gates at 2 min)
    if now_ts - _last_scan_ts < _MIN_SCAN_INTERVAL:
        # Still update active states even if we skip the entry scan
        pass

    # Load swing states from Redis on first call
    if not _active_swing_states:
        loaded = _load_swing_states_from_redis()
        if loaded:
            _active_swing_states = loaded
            logger.info(
                "🔄 Loaded %d active swing state(s) from Redis",
                len(_active_swing_states),
            )

    # Get focus data (prices, ATRs live from latest computation)
    focus_data = _get_focus_data()

    # Get swing candidate names from daily plan
    candidate_names = _get_swing_candidate_names()
    if not candidate_names:
        # No swing candidates today — just manage existing states
        result = _tick_active_states(focus_data, account_size)
        result["scan_skipped"] = True
        result["reason"] = "No swing candidates in daily plan"
        return result

    logger.info(
        "🔍 Swing tick: %d candidate(s), %d active state(s)",
        len(candidate_names),
        len(_active_swing_states),
    )

    # ── Phase A: Entry Detection ────────────────────────────────────────
    all_new_signals: list[Any] = []

    # Only scan for new entries if we haven't hit the concurrent limit
    active_non_closed = {
        k: v for k, v in _active_swing_states.items() if hasattr(v, "phase") and v.phase.value not in ("closed",)
    }

    for name in candidate_names:
        # Skip if we already have an active state for this asset
        if name in active_non_closed:
            logger.debug("Skipping %s — already has active swing state (%s)", name, active_non_closed[name].phase.value)
            continue

        # Skip if we've hit the concurrent swing limit
        if len(active_non_closed) >= _MAX_CONCURRENT_SWINGS:
            logger.debug("Skipping %s — max concurrent swings (%d) reached", name, _MAX_CONCURRENT_SWINGS)
            break

        # Fetch data for this asset
        bias = _get_asset_bias(name)
        if bias is None:
            logger.debug("Skipping %s — no bias data", name)
            continue

        price = _get_asset_price(name, focus_data)
        atr = _get_asset_atr(name, focus_data)
        if price <= 0 or atr <= 0:
            logger.debug("Skipping %s — price=%.2f atr=%.4f", name, price, atr)
            continue

        ticker = _get_asset_ticker(name)
        if not ticker:
            continue

        # Fetch intraday bars (prefer 5m for swing entries, resample to 15m)
        bars = _fetch_bars_5m(ticker)
        if bars is None or len(bars) < 10:
            logger.debug("Skipping %s — insufficient bar data (%d)", name, len(bars) if bars is not None else 0)
            continue

        session_open = _get_session_open_price(name)

        # Run swing entry detection
        try:
            from lib.trading.strategies.daily.swing_detector import detect_swing_entries

            signals = detect_swing_entries(
                bars=bars,
                bias=bias,
                current_price=price,
                atr=atr,
                asset_name=name,
                account_size=account_size,
                session_open_price=session_open,
            )

            if signals:
                all_new_signals.extend(signals)
                logger.info(
                    "⚡ Swing signal(s) for %s: %s",
                    name,
                    ", ".join(f"{s.entry_style.value}({s.direction}, {s.confidence:.0%})" for s in signals),
                )

                # Publish signal detection event for dashboard market events
                top = signals[0]
                _publish_swing_action(
                    "signal_detected",
                    name,
                    direction=top.direction,
                    confidence=top.confidence,
                    entry_style=top.entry_style.value,
                    entry_price=top.entry_price,
                    signal_count=len(signals),
                )

                # Auto-create SwingState for the top signal if it's ENTRY_READY
                from lib.trading.strategies.daily.swing_detector import SwingPhase

                top_signal = signals[0]
                if (
                    top_signal.phase == SwingPhase.ENTRY_READY
                    and top_signal.confidence >= 0.6
                    and len(active_non_closed) < _MAX_CONCURRENT_SWINGS
                ):
                    from lib.trading.strategies.daily.swing_detector import create_swing_state

                    state = create_swing_state(top_signal)
                    _active_swing_states[name] = state
                    active_non_closed[name] = state
                    logger.info(
                        "✅ Swing state CREATED for %s: %s %s @ %.4f | SL=%.4f | TP1=%.4f | TP2=%.4f | %d micros",
                        name,
                        state.direction,
                        top_signal.entry_style.value,
                        state.entry_price,
                        state.current_stop,
                        state.tp1,
                        state.tp2,
                        state.position_size,
                    )

        except Exception as exc:
            logger.warning("Swing detection failed for %s: %s", name, exc, exc_info=True)

    # Publish new signals and cache them as pending
    if all_new_signals:
        for sig in all_new_signals:
            sig_name = sig.asset_name if hasattr(sig, "asset_name") else ""
            if sig_name and sig_name not in _active_swing_states:
                _pending_swing_signals[sig_name] = sig

        _publish_swing_signals_to_redis(all_new_signals)

    _last_scan_ts = now_ts

    # ── Phase B: Active State Management ────────────────────────────────
    result = _tick_active_states(focus_data, account_size)
    result["new_signals"] = len(all_new_signals)
    result["candidates_scanned"] = len(candidate_names)

    return result


def _tick_active_states(
    focus_data: dict[str, Any] | None,
    account_size: int,
) -> dict[str, Any]:
    """Tick all active swing states — evaluate exits, update stops.

    Returns:
        Summary dict.
    """
    global _active_swing_states

    if not _active_swing_states:
        return {
            "active_states": 0,
            "exits": 0,
            "updates": 0,
        }

    from lib.trading.strategies.daily.swing_detector import (
        SwingPhase,
        update_swing_state,
    )

    exit_count = 0
    update_count = 0
    closed_names: list[str] = []

    for name, state in list(_active_swing_states.items()):
        if state.phase == SwingPhase.CLOSED:
            closed_names.append(name)
            continue

        # Fetch latest price
        price = _get_asset_price(name, focus_data)
        if price <= 0:
            logger.debug("Cannot update swing state for %s — no price", name)
            continue

        # Fetch ATR
        atr = _get_asset_atr(name, focus_data)

        # Fetch bars for EMA trailing
        ticker = _get_asset_ticker(name)
        bars = None
        if ticker:
            bars = _fetch_bars_15m(ticker)

        # Get point value for P&L calculation
        specs = _get_contract_specs(name)
        point_value = float(specs.get("point", 1.0)) if specs else 1.0

        # Update state
        try:
            updated_state, exit_signals = update_swing_state(
                state=state,
                current_price=price,
                bars=bars,
                atr=atr,
                point_value=point_value,
            )
            _active_swing_states[name] = updated_state
            update_count += 1

            if exit_signals:
                for ex in exit_signals:
                    logger.info(
                        "📊 Swing exit for %s: %s @ %.4f | PnL=$%.2f | R=%.2fx | %s",
                        name,
                        ex.reason.value,
                        ex.exit_price,
                        ex.pnl_estimate,
                        ex.r_multiple,
                        ex.reasoning,
                    )
                    _publish_swing_action(
                        "exit",
                        name,
                        reason=ex.reason.value,
                        exit_price=ex.exit_price,
                        pnl_estimate=ex.pnl_estimate,
                        r_multiple=ex.r_multiple,
                        reasoning=ex.reasoning,
                        direction=updated_state.direction,
                        entry_price=updated_state.entry_price,
                    )
                exit_count += len(exit_signals)

            if updated_state.phase == SwingPhase.CLOSED:
                closed_names.append(name)
                logger.info(
                    "🏁 Swing CLOSED for %s: phase=%s, remaining=%d",
                    name,
                    updated_state.phase.value,
                    updated_state.remaining_size,
                )

        except Exception as exc:
            logger.warning("Swing state update failed for %s: %s", name, exc)

    # Archive and remove closed states
    for name in closed_names:
        state = _active_swing_states.pop(name, None)
        if state is not None:
            _archive_closed_state(name, state)

    # Persist and publish
    _persist_swing_states()
    if _active_swing_states:
        _publish_swing_states_to_redis()

    active_count = len(_active_swing_states)
    if active_count > 0 or exit_count > 0:
        logger.info(
            "📈 Swing states: %d active, %d updated, %d exits, %d closed",
            active_count,
            update_count,
            exit_count,
            len(closed_names),
        )

    return {
        "active_states": active_count,
        "exits": exit_count,
        "updates": update_count,
        "closed": len(closed_names),
    }


# ---------------------------------------------------------------------------
# Utility functions for external access
# ---------------------------------------------------------------------------


def get_active_swing_states() -> dict[str, Any]:
    """Get the current active swing states dict (for dashboard / API)."""
    return dict(_active_swing_states)


def get_swing_summary() -> dict[str, Any]:
    """Get a summary of swing detector status for engine status publishing."""
    global _active_swing_states, _last_scan_ts

    active = {}
    for name, state in _active_swing_states.items():
        active[name] = {
            "phase": state.phase.value if hasattr(state, "phase") else "unknown",
            "direction": state.direction if hasattr(state, "direction") else "",
            "entry_price": state.entry_price if hasattr(state, "entry_price") else 0,
            "current_stop": state.current_stop if hasattr(state, "current_stop") else 0,
            "remaining_size": state.remaining_size if hasattr(state, "remaining_size") else 0,
        }

    return {
        "active_count": len(_active_swing_states),
        "active_assets": list(_active_swing_states.keys()),
        "states": active,
        "last_scan_ts": _last_scan_ts,
        "last_scan_ago_s": round(time.time() - _last_scan_ts, 1) if _last_scan_ts > 0 else None,
        "max_concurrent": _MAX_CONCURRENT_SWINGS,
    }


def reset_swing_states() -> None:
    """Reset all swing states (for testing or manual intervention)."""
    global _active_swing_states, _last_scan_ts
    _active_swing_states = {}
    _last_scan_ts = 0.0
    _cache_set(_REDIS_KEY_SWING_STATES, "{}", ttl=18 * 3600)
    logger.info("🔄 Swing states reset")


# ---------------------------------------------------------------------------
# Swing state mutation functions — called by dashboard action endpoints
# ---------------------------------------------------------------------------

# Pending signals cache: signals detected but not yet accepted/ignored.
# Keyed by asset_name → SwingSignal.  Populated during tick_swing_detector
# Phase A when auto-create is disabled or confidence is below threshold.
# Also populated for ALL detected signals so the dashboard can show them.
_pending_swing_signals: dict[str, Any] = {}


def get_pending_signals() -> dict[str, Any]:
    """Return pending (unacted) swing signals for the dashboard.

    Returns a dict of {asset_name: signal.to_dict()} for signals that
    have been detected but not yet accepted into an active SwingState.
    """
    # Also load from Redis in case signals were published by a previous tick
    result: dict[str, Any] = {}

    # First, check module-level pending signals
    for name, sig in _pending_swing_signals.items():
        if name not in _active_swing_states:
            try:
                result[name] = sig.to_dict() if hasattr(sig, "to_dict") else sig
            except Exception:
                result[name] = sig

    # Also check Redis for recently published signals
    try:
        raw = _cache_get(_REDIS_KEY_SWING_SIGNALS)
        if raw:
            data = json.loads(raw)
            if isinstance(data, list):
                for sig_dict in data:
                    name = sig_dict.get("asset_name", "")
                    if name and name not in _active_swing_states and name not in result:
                        result[name] = sig_dict
    except Exception as exc:
        logger.debug("Failed to load pending signals from Redis: %s", exc)

    return result


def accept_swing_signal(asset_name: str) -> dict[str, Any]:
    """Accept a pending swing signal — create an active SwingState.

    Called from the dashboard when the user clicks 'Accept Signal' on a
    swing card.  Looks up the pending signal for the asset, creates a
    SwingState, and publishes it.

    Args:
        asset_name: The asset name (e.g. 'Gold', 'Crude Oil').

    Returns:
        Dict with status and details of the created state.

    Raises:
        ValueError: If no pending signal exists or asset already has an active state.
    """
    global _active_swing_states, _pending_swing_signals

    # Check if already active
    if asset_name in _active_swing_states:
        phase = "unknown"
        with contextlib.suppress(Exception):
            phase = _active_swing_states[asset_name].phase.value
        if phase != "closed":
            raise ValueError(f"Asset '{asset_name}' already has an active swing state (phase={phase})")

    # Check concurrent limit
    active_non_closed = {
        k: v for k, v in _active_swing_states.items() if hasattr(v, "phase") and v.phase.value != "closed"
    }
    if len(active_non_closed) >= _MAX_CONCURRENT_SWINGS:
        raise ValueError(
            f"Maximum concurrent swings ({_MAX_CONCURRENT_SWINGS}) reached. "
            f"Close an existing swing before accepting a new one."
        )

    # Find the pending signal
    signal = _pending_swing_signals.get(asset_name)

    if signal is None:
        # Try to reconstruct from Redis
        try:
            raw = _cache_get(_REDIS_KEY_SWING_SIGNALS)
            if raw:
                data = json.loads(raw)
                if isinstance(data, list):
                    for sig_dict in data:
                        if sig_dict.get("asset_name") == asset_name:
                            # Reconstruct SwingSignal from dict
                            from lib.trading.strategies.daily.swing_detector import (
                                SwingEntryStyle,
                                SwingPhase,
                                SwingSignal,
                            )

                            signal = SwingSignal(
                                asset_name=sig_dict.get("asset_name", ""),
                                entry_style=SwingEntryStyle(sig_dict.get("entry_style", "pullback_entry")),
                                direction=sig_dict.get("direction", "LONG"),
                                confidence=sig_dict.get("confidence", 0.0),
                                entry_price=sig_dict.get("entry_price", 0.0),
                                entry_zone_low=sig_dict.get("entry_zone_low", 0.0),
                                entry_zone_high=sig_dict.get("entry_zone_high", 0.0),
                                stop_loss=sig_dict.get("stop_loss", 0.0),
                                tp1=sig_dict.get("tp1", 0.0),
                                tp2=sig_dict.get("tp2", 0.0),
                                atr=sig_dict.get("atr", 0.0),
                                risk_reward_tp1=sig_dict.get("risk_reward_tp1", 0.0),
                                risk_reward_tp2=sig_dict.get("risk_reward_tp2", 0.0),
                                risk_dollars=sig_dict.get("risk_dollars", 0.0),
                                position_size=sig_dict.get("position_size", 1),
                                reasoning=sig_dict.get("reasoning", ""),
                                key_level_used=sig_dict.get("key_level_used", ""),
                                key_level_price=sig_dict.get("key_level_price", 0.0),
                                confirmation_bar_idx=sig_dict.get("confirmation_bar_idx", -1),
                                detected_at=sig_dict.get("detected_at", ""),
                                phase=SwingPhase(sig_dict.get("phase", "entry_ready")),
                            )
                            break
        except Exception as exc:
            logger.debug("Failed to reconstruct signal from Redis: %s", exc)

    if signal is None:
        raise ValueError(
            f"No pending swing signal found for '{asset_name}'. "
            f"Wait for the next swing scan or check that the asset is a swing candidate."
        )

    # Create the swing state
    from lib.trading.strategies.daily.swing_detector import SwingPhase, create_swing_state

    # Ensure signal phase is ENTRY_READY for acceptance
    if hasattr(signal, "phase"):
        signal.phase = SwingPhase.ENTRY_READY

    state = create_swing_state(signal)
    _active_swing_states[asset_name] = state

    # Remove from pending
    _pending_swing_signals.pop(asset_name, None)

    # Persist and publish
    _persist_swing_states()
    _publish_swing_states_to_redis()

    logger.info(
        "✅ Swing ACCEPTED for %s: %s %s @ %.4f | SL=%.4f | TP1=%.4f | TP2=%.4f | %d micros",
        asset_name,
        state.direction,
        signal.entry_style.value if hasattr(signal, "entry_style") else "unknown",
        state.entry_price,
        state.current_stop,
        state.tp1,
        state.tp2,
        state.position_size,
    )

    _publish_swing_action(
        "accepted",
        asset_name,
        direction=state.direction,
        entry_price=state.entry_price,
        stop_loss=state.current_stop,
        tp1=state.tp1,
        tp2=state.tp2,
        position_size=state.position_size,
        phase=state.phase.value,
    )

    return {
        "status": "accepted",
        "asset_name": asset_name,
        "direction": state.direction,
        "entry_price": state.entry_price,
        "stop_loss": state.current_stop,
        "tp1": state.tp1,
        "tp2": state.tp2,
        "position_size": state.position_size,
        "phase": state.phase.value,
    }


def ignore_swing_signal(asset_name: str) -> dict[str, Any]:
    """Ignore/dismiss a pending swing signal for an asset.

    Called from the dashboard when the user clicks 'Ignore' on a swing card.
    Removes the signal from the pending set so it won't be shown again
    (until the next scan detects a new signal).

    Args:
        asset_name: The asset name to ignore.

    Returns:
        Dict with status.
    """
    global _pending_swing_signals

    had_pending = asset_name in _pending_swing_signals
    _pending_swing_signals.pop(asset_name, None)

    # Also clean from Redis signals list
    try:
        raw = _cache_get(_REDIS_KEY_SWING_SIGNALS)
        if raw:
            data = json.loads(raw)
            if isinstance(data, list):
                filtered = [s for s in data if s.get("asset_name") != asset_name]
                if len(filtered) != len(data):
                    _cache_set(
                        _REDIS_KEY_SWING_SIGNALS,
                        json.dumps(filtered, default=str),
                        ttl=18 * 3600,
                    )
    except Exception as exc:
        logger.debug("Failed to clean ignored signal from Redis: %s", exc)

    logger.info(
        "🚫 Swing signal IGNORED for %s%s",
        asset_name,
        " (was pending)" if had_pending else " (not in pending set)",
    )

    _publish_swing_action("ignored", asset_name, was_pending=had_pending)

    return {
        "status": "ignored",
        "asset_name": asset_name,
        "was_pending": had_pending,
    }


def close_swing_position(asset_name: str, reason: str = "manual") -> dict[str, Any]:
    """Close an active swing position manually.

    Called from the dashboard when the user clicks 'Close Position' on an
    active swing card.

    Args:
        asset_name: The asset name to close.
        reason: Reason for closing (e.g. 'manual', 'invalidated').

    Returns:
        Dict with status and final state details.

    Raises:
        ValueError: If no active swing state exists for the asset.
    """
    global _active_swing_states

    state = _active_swing_states.get(asset_name)
    if state is None:
        raise ValueError(f"No active swing state for '{asset_name}'")

    from lib.trading.strategies.daily.swing_detector import SwingPhase

    if state.phase == SwingPhase.CLOSED:
        raise ValueError(f"Swing for '{asset_name}' is already closed")

    # Record the close
    prev_phase = state.phase.value
    state.phase = SwingPhase.CLOSED
    state.remaining_size = 0
    state.last_update = datetime.now(tz=_EST).isoformat()

    # Archive and remove
    _archive_closed_state(asset_name, state)
    _active_swing_states.pop(asset_name, None)

    # Persist and publish
    _persist_swing_states()
    _publish_swing_states_to_redis()

    logger.info(
        "🏁 Swing CLOSED (manual) for %s: was %s, reason=%s, entry=%.4f, stop=%.4f",
        asset_name,
        prev_phase,
        reason,
        state.entry_price,
        state.current_stop,
    )

    _publish_swing_action(
        "closed",
        asset_name,
        previous_phase=prev_phase,
        reason=reason,
        entry_price=state.entry_price,
        direction=state.direction,
    )

    return {
        "status": "closed",
        "asset_name": asset_name,
        "previous_phase": prev_phase,
        "reason": reason,
        "entry_price": state.entry_price,
        "direction": state.direction,
    }


def move_stop_to_breakeven(asset_name: str) -> dict[str, Any]:
    """Move the stop-loss to breakeven (entry price) for an active swing.

    Called from the dashboard when the user clicks 'Move Stop to BE' on
    an active swing card.  Only works for ACTIVE, TP1_HIT, or TRAILING phases.

    Args:
        asset_name: The asset name.

    Returns:
        Dict with status and updated stop details.

    Raises:
        ValueError: If no active swing state or state is in wrong phase.
    """
    global _active_swing_states

    state = _active_swing_states.get(asset_name)
    if state is None:
        raise ValueError(f"No active swing state for '{asset_name}'")

    from lib.trading.strategies.daily.swing_detector import SwingPhase

    allowed_phases = {SwingPhase.ACTIVE, SwingPhase.TP1_HIT, SwingPhase.TRAILING}
    if state.phase not in allowed_phases:
        raise ValueError(
            f"Cannot move stop to BE — swing for '{asset_name}' is in "
            f"phase '{state.phase.value}' (must be active/tp1_hit/trailing)"
        )

    old_stop = state.current_stop
    state.current_stop = state.entry_price
    state.last_update = datetime.now(tz=_EST).isoformat()

    # Update in the active states dict
    _active_swing_states[asset_name] = state

    # Persist and publish
    _persist_swing_states()
    _publish_swing_states_to_redis()

    logger.info(
        "🛡️ Swing stop moved to BREAKEVEN for %s: %.4f → %.4f (entry)",
        asset_name,
        old_stop,
        state.entry_price,
    )

    _publish_swing_action(
        "stop_moved",
        asset_name,
        old_stop=old_stop,
        new_stop=state.entry_price,
        entry_price=state.entry_price,
        direction=state.direction,
        phase=state.phase.value,
    )

    return {
        "status": "stop_moved",
        "asset_name": asset_name,
        "old_stop": old_stop,
        "new_stop": state.entry_price,
        "entry_price": state.entry_price,
        "direction": state.direction,
        "phase": state.phase.value,
    }


def update_swing_stop(asset_name: str, new_stop: float) -> dict[str, Any]:
    """Manually update the stop-loss price for an active swing.

    Called from the dashboard for fine-grained stop adjustment.
    Validates that the new stop is on the correct side of the entry.

    Args:
        asset_name: The asset name.
        new_stop: New stop-loss price.

    Returns:
        Dict with status and updated stop details.

    Raises:
        ValueError: If no active state, wrong phase, or invalid stop price.
    """
    global _active_swing_states

    state = _active_swing_states.get(asset_name)
    if state is None:
        raise ValueError(f"No active swing state for '{asset_name}'")

    from lib.trading.strategies.daily.swing_detector import SwingPhase

    allowed_phases = {SwingPhase.ACTIVE, SwingPhase.TP1_HIT, SwingPhase.TRAILING}
    if state.phase not in allowed_phases:
        raise ValueError(f"Cannot update stop — swing for '{asset_name}' is in phase '{state.phase.value}'")

    if new_stop <= 0:
        raise ValueError(f"Invalid stop price: {new_stop}")

    # Validate stop is on the correct side
    if state.direction == "LONG" and new_stop >= state.entry_price * 1.05:
        raise ValueError(
            f"Stop {new_stop:.4f} is too far above entry {state.entry_price:.4f} for a LONG. "
            f"Use close_swing_position() to exit."
        )
    elif state.direction == "SHORT" and new_stop <= state.entry_price * 0.95:
        raise ValueError(
            f"Stop {new_stop:.4f} is too far below entry {state.entry_price:.4f} for a SHORT. "
            f"Use close_swing_position() to exit."
        )

    old_stop = state.current_stop
    state.current_stop = new_stop
    state.last_update = datetime.now(tz=_EST).isoformat()

    _active_swing_states[asset_name] = state

    # Persist and publish
    _persist_swing_states()
    _publish_swing_states_to_redis()

    logger.info(
        "🎯 Swing stop UPDATED for %s: %.4f → %.4f",
        asset_name,
        old_stop,
        new_stop,
    )

    _publish_swing_action(
        "stop_updated",
        asset_name,
        old_stop=old_stop,
        new_stop=new_stop,
        entry_price=state.entry_price,
        direction=state.direction,
        phase=state.phase.value,
    )

    return {
        "status": "stop_updated",
        "asset_name": asset_name,
        "old_stop": old_stop,
        "new_stop": new_stop,
        "entry_price": state.entry_price,
        "direction": state.direction,
        "phase": state.phase.value,
    }


def get_swing_state_detail(asset_name: str) -> dict[str, Any] | None:
    """Get detailed state information for a specific swing asset.

    Returns the full state dict if active, or None if not found.
    """
    state = _active_swing_states.get(asset_name)
    if state is None:
        return None

    result = state.to_dict() if hasattr(state, "to_dict") else {"phase": "unknown"}
    result["asset_name"] = asset_name

    # Add computed fields
    if hasattr(state, "entry_price") and hasattr(state, "current_stop"):
        risk_per_unit = abs(state.entry_price - state.current_stop)
        result["risk_per_unit"] = round(risk_per_unit, 6)

        if hasattr(state, "tp1") and risk_per_unit > 0:
            reward_tp1 = abs(state.tp1 - state.entry_price)
            result["rr_tp1"] = round(reward_tp1 / risk_per_unit, 2)

        if hasattr(state, "tp2") and risk_per_unit > 0:
            reward_tp2 = abs(state.tp2 - state.entry_price)
            result["rr_tp2"] = round(reward_tp2 / risk_per_unit, 2)

    return result


def get_swing_history(limit: int = 20) -> list[dict[str, Any]]:
    """Get recent closed swing trade history from Redis.

    Args:
        limit: Maximum number of entries to return.

    Returns:
        List of archived swing state dicts, most recent first.
    """
    try:
        r = _get_redis()
        if r is None:
            return []

        raw_list = r.lrange(_REDIS_KEY_SWING_HISTORY, 0, limit - 1)
        if not raw_list:
            return []

        result = []
        for raw in raw_list:
            try:
                entry = json.loads(raw)
                result.append(entry)
            except (json.JSONDecodeError, TypeError):
                continue
        return result

    except Exception as exc:
        logger.debug("Failed to load swing history: %s", exc)
        return []
