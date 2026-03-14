"""
SSE (Server-Sent Events) Router — TASK-302
============================================
Streams live dashboard updates to the browser via SSE.

Architecture:
    Engine → XADD dashboard:stream:focus (durable) + PUBLISH dashboard:live (trigger)
    Data-service SSE → on connect: XREVRANGE last 8 messages (catch-up),
                       then subscribe to pub/sub for live updates
    Browser → hx-ext="sse" sse-connect="/sse/dashboard" with per-asset event names

Endpoints:
    GET /sse/dashboard  — Main SSE stream (focus updates, alerts, heartbeat)

Event types sent to browser:
    - focus-update       — Full focus payload (all assets)
    - {symbol}-update    — Per-asset update (e.g. mgc-update, mnq-update)
    - no-trade-alert     — No-trade condition triggered
    - session-change     — Session mode changed (pre-market/active/off-hours)
    - positions-update   — Live positions changed
    - grok-update        — Grok compact AI summary (TASK-602)
    - risk-update        — Risk status changed (TASK-502)
    - heartbeat          — Keep-alive every 30 seconds
"""

import asyncio
import contextlib
import json
import logging
import os
import time
from collections.abc import AsyncGenerator
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

logger = logging.getLogger("api.sse")

_EST = ZoneInfo("America/New_York")

router = APIRouter(tags=["SSE"])

# ---------------------------------------------------------------------------
# Throttle settings — max 1 update per asset per N seconds
# ---------------------------------------------------------------------------
_THROTTLE_SECONDS = 7.0
_HEARTBEAT_INTERVAL = 30.0
_CATCHUP_COUNT = 1

# ---------------------------------------------------------------------------
# Reconnect / backoff settings for the pub/sub Redis connection
# ---------------------------------------------------------------------------
# How many consecutive pubsub errors before we give up and close the stream
# (closing lets the browser's EventSource retry directive reconnect cleanly).
_PUBSUB_MAX_ERRORS = 8
# Backoff sequence (seconds) used when a reconnect attempt fails.
_PUBSUB_BACKOFF = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 30.0]

# Track last send time per event type for throttling
_last_sent: dict[str, float] = {}


def _should_throttle(event_key: str) -> bool:
    """Return True if we should skip this event due to throttling."""
    now = time.monotonic()
    last = _last_sent.get(event_key, 0)
    if now - last < _THROTTLE_SECONDS:
        return True
    _last_sent[event_key] = now
    return False


def _format_sse(
    data: str,
    event: str | None = None,
    id: str | None = None,
    retry: int | None = None,
) -> str:
    """Format a single SSE message according to the spec.

    See: https://html.spec.whatwg.org/multipage/server-sent-events.html
    """
    lines = []
    if id is not None:
        lines.append(f"id: {id}")
    if event is not None:
        lines.append(f"event: {event}")
    if retry is not None:
        lines.append(f"retry: {retry}")
    # Data can be multi-line; each line needs its own "data:" prefix
    for line in data.split("\n"):
        lines.append(f"data: {line}")
    lines.append("")  # blank line terminates the event
    lines.append("")
    return "\n".join(lines)


def _make_heartbeat_event() -> str:
    """Create a heartbeat SSE event with current server time."""
    now = datetime.now(tz=_EST)
    payload = json.dumps(
        {
            "type": "heartbeat",
            "time_et": now.strftime("%H:%M:%S ET"),
            "timestamp": now.isoformat(),
        }
    )
    return _format_sse(data=payload, event="heartbeat")


def _make_session_event(session_mode: str) -> str:
    """Create a session-change SSE event."""
    now = datetime.now(tz=_EST)
    emoji = {
        "pre_market": "\U0001f319",
        "active": "\U0001f7e2",
        "off_hours": "\u2699\ufe0f",
    }.get(session_mode, "")
    payload = json.dumps(
        {
            "type": "session-change",
            "session": session_mode,
            "emoji": emoji,
            "timestamp": now.isoformat(),
        }
    )
    return _format_sse(data=payload, event="session-change")


# ---------------------------------------------------------------------------
# Redis helpers — async wrappers around the sync redis client
# ---------------------------------------------------------------------------


def _get_redis():
    """Get the shared Redis client from the cache module, or None."""
    try:
        from lib.core.cache import REDIS_AVAILABLE, _r

        if REDIS_AVAILABLE and _r is not None:
            return _r
    except ImportError:
        pass
    return None


def _make_redis_connection():
    """Create a *fresh* dedicated Redis connection for SSE pub/sub.

    SSE pub/sub must never share the application-wide ``_r`` connection
    because ``pubsub.get_message()`` alters connection state in a way that
    is not safe to share with cache reads/writes.  A dedicated connection
    also means we can close and re-open it on error without affecting the
    rest of the app.

    Returns a ``redis.Redis`` instance on success, or ``None`` if Redis is
    not configured / the connection attempt fails.
    """
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    try:
        import redis as redis_lib  # type: ignore[import-unresolved]

        client = redis_lib.from_url(redis_url, decode_responses=False)
        client.ping()
        return client
    except Exception as exc:
        logger.debug("SSE: could not create dedicated Redis connection: %s", exc)
        return None


def _teardown_pubsub(pubsub, client) -> None:
    """Cleanly unsubscribe and close a pubsub handle and its connection."""
    if pubsub is not None:
        with contextlib.suppress(Exception):
            pubsub.punsubscribe("dashboard:*")
        with contextlib.suppress(Exception):
            pubsub.close()
    if client is not None:
        with contextlib.suppress(Exception):
            client.close()


def _get_catchup_messages(count: int = _CATCHUP_COUNT) -> list[dict[str, str]]:
    """Read the last N messages from the Redis Stream for catch-up.

    Returns list of dicts with keys: id, data, ts.
    """
    r = _get_redis()
    if r is None:
        return []

    try:
        # XREVRANGE returns newest first; we want oldest first for the client
        raw = r.xrevrange("dashboard:stream:focus", count=count)  # type: ignore[union-attr]
        if not raw:
            return []

        messages: list[dict[str, str]] = []
        for msg_id, fields in reversed(list(raw)):  # type: ignore[arg-type]
            # msg_id is bytes, fields is dict of bytes
            entry = {
                "id": msg_id.decode() if isinstance(msg_id, bytes) else str(msg_id),
                "data": fields.get(b"data", b"{}").decode()
                if isinstance(fields.get(b"data", b"{}"), bytes)
                else str(fields.get(b"data", "{}")),
                "ts": fields.get(b"ts", b"").decode()
                if isinstance(fields.get(b"ts", b""), bytes)
                else str(fields.get(b"ts", "")),
            }
            messages.append(entry)
        return messages
    except Exception as exc:
        logger.debug("Failed to read catchup from Redis Stream: %s", exc)
        return []


def _get_focus_from_cache() -> str | None:
    """Read the current focus JSON from Redis cache."""
    try:
        from lib.core.cache import cache_get

        raw = cache_get("engine:daily_focus")
        if raw:
            return raw.decode() if isinstance(raw, bytes) else str(raw)
    except Exception:
        pass
    return None


def _get_positions_from_cache() -> str | None:
    """Read current positions from Redis cache."""
    try:
        from lib.core.cache import _cache_key, cache_get

        key = _cache_key("live_positions", "current")
        raw = cache_get(key)
        if raw:
            return raw.decode() if isinstance(raw, bytes) else str(raw)
    except Exception:
        pass
    return None


def _get_engine_status() -> dict[str, Any] | None:
    """Read engine status from Redis."""
    try:
        from lib.core.cache import cache_get

        raw = cache_get("engine:status")
        if raw:
            return json.loads(raw)
    except Exception:
        pass
    return None


def _get_grok_from_cache() -> str | None:
    """Read the latest Grok compact update from Redis cache (TASK-602)."""
    try:
        from lib.core.cache import cache_get

        raw = cache_get("engine:grok_update")
        if raw:
            return raw.decode() if isinstance(raw, bytes) else str(raw)
    except Exception:
        pass
    return None


def _get_reddit_signals() -> str | None:
    """Read cached Reddit 1h signals for all assets from Redis.

    Returns a JSON string with shape:
        {"GC": {"signal": "BULL", "weighted_sentiment": 0.31, ...}, ...}
    or None if no data is available.
    """
    try:
        r = _get_redis()
        if r is None:
            return None
        from lib.integrations.reddit_watcher import ASSET_KEYWORDS

        out: dict[str, dict] = {}
        for asset in ASSET_KEYWORDS:
            key = f"reddit:signal:{asset}"
            raw = r.hgetall(key)
            if raw:
                out[asset] = {
                    (k.decode() if isinstance(k, bytes) else k): (
                        float(v)
                        if (k.decode() if isinstance(k, bytes) else k) != "signal"
                        else (v.decode() if isinstance(v, bytes) else v)
                    )
                    for k, v in raw.items()
                }
        return json.dumps(out) if out else None
    except Exception:
        return None


def _get_risk_from_cache() -> str | None:
    """Read the latest risk status from Redis cache."""
    try:
        from lib.core.cache import cache_get

        raw = cache_get("engine:risk_status")
        if raw:
            return raw.decode() if isinstance(raw, bytes) else str(raw)
    except Exception:
        pass
    return None


def _get_orb_from_cache() -> str | None:
    """Read multi-session ORB results from Redis cache (TASK-801).

    Fetches both London Open (engine:orb:london) and US Equity Open
    (engine:orb:us) session results and combines them into a single
    JSON payload with keys: london, us, best.

    Falls back to the legacy engine:orb key if session-specific keys
    are not present.
    """
    try:
        from lib.core.cache import cache_get

        sessions: dict[str, Any] = {}

        # Fetch London Open ORB
        raw_london = cache_get("engine:orb:london")
        if raw_london:
            data = raw_london.decode() if isinstance(raw_london, bytes) else str(raw_london)
            if data:
                with contextlib.suppress(json.JSONDecodeError, TypeError):
                    sessions["london"] = json.loads(data)

        # Fetch US Equity Open ORB
        raw_us = cache_get("engine:orb:us")
        if raw_us:
            data = raw_us.decode() if isinstance(raw_us, bytes) else str(raw_us)
            if data:
                with contextlib.suppress(json.JSONDecodeError, TypeError):
                    sessions["us"] = json.loads(data)

        # Fallback: legacy combined key
        if not sessions:
            raw = cache_get("engine:orb")
            if raw:
                data = raw.decode() if isinstance(raw, bytes) else str(raw)
                if data:
                    try:
                        legacy = json.loads(data)
                        sk = legacy.get("session_key", "us")
                        sessions[sk] = legacy
                    except (json.JSONDecodeError, TypeError):
                        pass

        if not sessions:
            return None

        # Determine "best" result: prefer one with a breakout
        best = None
        for key in ("london", "us"):
            s = sessions.get(key)
            if s and s.get("breakout_detected"):
                best = s
                break
        if best is None:
            for key in ("london", "us"):
                s = sessions.get(key)
                if s and not s.get("error"):
                    best = s
                    break
        if best is None:
            for s in sessions.values():
                if s:
                    best = s
                    break

        combined = {
            "london": sessions.get("london"),
            "us": sessions.get("us"),
            "best": best,
        }
        return json.dumps(combined, default=str)
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Main SSE generator
# ---------------------------------------------------------------------------


async def _dashboard_event_generator(request: Request) -> AsyncGenerator[str]:
    """Async generator that yields SSE events for the dashboard.

    Flow:
    1. Send retry directive (auto-reconnect after 3s)
    2. Send catch-up messages from Redis Stream
    3. If Redis pub/sub available, subscribe and forward live events
       - Maintains a *dedicated* Redis connection (not the shared cache _r)
       - On connection error, attempts reconnect with exponential backoff
       - After _PUBSUB_MAX_ERRORS consecutive failures, closes the stream so
         the browser's EventSource retry fires a clean reconnect
    4. Otherwise, fall back to polling Redis every 5 seconds
    5. Send heartbeat every 30 seconds
    """

    # 1. Retry directive — tells browser to reconnect after 3 seconds on disconnect
    yield _format_sse(data="connected", event="connected", retry=3000)

    # 2. Catch-up: send last N focus updates from Redis Stream
    catchup = _get_catchup_messages()
    if catchup:
        for msg in catchup:
            try:
                data_str = msg["data"]
                yield _format_sse(data=data_str, event="focus-update", id=msg["id"])

                # Also emit per-asset events from the catchup data
                try:
                    focus = json.loads(data_str)
                    for asset in focus.get("assets", []):
                        symbol = asset.get("symbol", "").lower().replace(" ", "_")
                        if symbol:
                            asset_json = json.dumps(asset, default=str)
                            yield _format_sse(
                                data=asset_json,
                                event=f"{symbol}-update",
                                id=msg["id"],
                            )

                    # No-trade catchup
                    if focus.get("no_trade"):
                        yield _format_sse(
                            data=json.dumps(
                                {
                                    "no_trade": True,
                                    "reason": focus.get("no_trade_reason", ""),
                                }
                            ),
                            event="no-trade-alert",
                        )
                except (json.JSONDecodeError, TypeError):
                    pass

            except Exception as exc:
                logger.debug("Catchup message error: %s", exc)
    else:
        # No stream data — try the cache key directly
        cached = _get_focus_from_cache()
        if cached:
            yield _format_sse(data=cached, event="focus-update")

    # Send initial positions
    pos = _get_positions_from_cache()
    if pos:
        yield _format_sse(data=pos, event="positions-update")

    # Send initial Grok compact update (TASK-602)
    grok = _get_grok_from_cache()
    if grok:
        yield _format_sse(data=grok, event="grok-update")

    # Send initial risk status
    risk = _get_risk_from_cache()
    if risk:
        yield _format_sse(data=risk, event="risk-update")

    # Send initial ORB status (TASK-801)
    orb = _get_orb_from_cache()
    if orb:
        yield _format_sse(data=orb, event="orb-update")

    # Send initial Reddit signals
    reddit = _get_reddit_signals()
    if reddit:
        yield _format_sse(data=reddit, event="reddit-signal")

    # Send initial session info
    status = _get_engine_status()
    if status:
        session_mode = status.get("session_mode", "unknown")
        yield _make_session_event(session_mode)

    # Flush hint: yield an SSE comment to force the HTTP chunked-encoding
    # layer to push all buffered catch-up data to the client before we
    # enter the long-lived pub/sub / polling loop.  SSE comments (lines
    # starting with ':') are silently ignored by EventSource but cause
    # the server to emit a chunk boundary.  Unlike asyncio.sleep(0) this
    # doesn't interact with test patches that raise CancelledError.
    yield ": flush\n\n"

    # 3. Set up pub/sub on a *dedicated* Redis connection.
    #    Using a dedicated connection (not the shared cache._r) means:
    #      - pubsub state never contaminates cache reads/writes
    #      - we can close + reopen the connection on error without side-effects
    sse_client = _make_redis_connection()
    pubsub = None
    use_pubsub = False

    if sse_client is not None:
        try:
            pubsub = sse_client.pubsub()
            pubsub.psubscribe("dashboard:*")
            use_pubsub = True
            logger.info("SSE client connected (pub/sub mode)")
        except Exception as exc:
            logger.warning("SSE: initial pub/sub subscribe failed, falling back to polling: %s", exc)
            _teardown_pubsub(pubsub, sse_client)
            pubsub = None
            sse_client = None
            use_pubsub = False

    if not use_pubsub:
        logger.info("SSE client connected (polling mode)")

    last_heartbeat = time.monotonic()
    last_focus_hash = ""
    last_positions_hash = ""
    last_grok_hash = ""
    last_risk_hash = ""
    last_orb_hash = ""
    last_reddit_hash = ""
    last_session = ""

    # Consecutive pubsub error counter — reset to 0 on any successful read.
    pubsub_errors = 0

    try:
        while True:
            # Check if client disconnected
            if await request.is_disconnected():
                logger.debug("SSE client disconnected")
                break

            now = time.monotonic()

            # --- Heartbeat ---
            if now - last_heartbeat >= _HEARTBEAT_INTERVAL:
                yield _make_heartbeat_event()
                last_heartbeat = now

            if use_pubsub and pubsub is not None:
                # ---- Pub/sub mode: check for messages ----
                try:
                    message = pubsub.get_message(ignore_subscribe_messages=True, timeout=0.1)
                    # Successful read (even if no message) — reset error counter
                    pubsub_errors = 0

                    if message and message["type"] in ("message", "pmessage"):
                        raw_channel = message.get("channel", b"")
                        channel: str = (
                            raw_channel.decode() if isinstance(raw_channel, bytes) else str(raw_channel or "")
                        )
                        raw_data = message.get("data", b"")
                        data: str = raw_data.decode() if isinstance(raw_data, bytes) else str(raw_data or "")

                        if channel == "dashboard:live":
                            # Full focus update
                            if not _should_throttle("focus-update"):
                                yield _format_sse(data=data, event="focus-update")

                                # Emit per-asset events
                                try:
                                    focus = json.loads(data)
                                    for asset in focus.get("assets", []):
                                        symbol = asset.get("symbol", "").lower().replace(" ", "_")
                                        if symbol and not _should_throttle(f"{symbol}-update"):
                                            asset_json = json.dumps(asset, default=str)
                                            yield _format_sse(
                                                data=asset_json,
                                                event=f"{symbol}-update",
                                            )
                                except (json.JSONDecodeError, TypeError):
                                    pass

                        elif channel.startswith("dashboard:asset:"):
                            # Per-asset update
                            symbol = channel.split(":")[-1]
                            event_name = f"{symbol}-update"
                            if not _should_throttle(event_name):
                                yield _format_sse(data=data, event=event_name)

                        elif channel == "dashboard:no_trade":
                            yield _format_sse(data=data, event="no-trade-alert")

                        elif channel == "dashboard:session":
                            yield _format_sse(data=data, event="session-change")

                        elif channel == "dashboard:positions":
                            if not _should_throttle("positions-update"):
                                yield _format_sse(data=data, event="positions-update")

                        elif channel == "dashboard:grok":
                            # Grok compact AI summary (TASK-602)
                            if not _should_throttle("grok-update"):
                                yield _format_sse(data=data, event="grok-update")

                        elif channel == "dashboard:risk":
                            # Risk status update
                            if not _should_throttle("risk-update"):
                                yield _format_sse(data=data, event="risk-update")

                        elif channel.startswith("dashboard:orb") and not _should_throttle("orb-update"):
                            # Opening Range Breakout alert (TASK-801)
                            # Channels: dashboard:orb, dashboard:orb:london, dashboard:orb:us
                            # Re-fetch the combined multi-session data from cache
                            # so the dashboard always gets a complete picture.
                            combined = _get_orb_from_cache()
                            if combined:
                                yield _format_sse(data=combined, event="orb-update")
                            else:
                                yield _format_sse(data=data, event="orb-update")

                        elif channel == "dashboard:swing_update" and not _should_throttle("swing-update"):
                            # Swing detector state/signal update (Phase 3D)
                            # Pushed when swing signals are detected, states are
                            # created/updated/closed, or manual actions (accept/
                            # ignore/close/stop-to-BE) are performed.
                            yield _format_sse(data=data, event="swing-update")

                        elif channel == "dashboard:copy_trade" and not _should_throttle("copy-trade-update"):
                            # Copy-trade batch result (RITHMIC-F)
                            # Pushed by CopyTrader after every SEND ALL / execute_order_commands.
                            # Payload: CopyBatchResult.to_dict() — includes compliance_log,
                            # per-account results, rate counter, batch_id.
                            yield _format_sse(data=data, event="copy-trade-update")

                except Exception as exc:
                    pubsub_errors += 1
                    backoff = _PUBSUB_BACKOFF[min(pubsub_errors - 1, len(_PUBSUB_BACKOFF) - 1)]
                    logger.warning(
                        "SSE: pub/sub read error #%d/%d (backoff %.1fs): %s",
                        pubsub_errors,
                        _PUBSUB_MAX_ERRORS,
                        backoff,
                        exc,
                    )

                    # Tear down the broken connection before sleeping
                    _teardown_pubsub(pubsub, sse_client)
                    pubsub = None
                    sse_client = None

                    if pubsub_errors >= _PUBSUB_MAX_ERRORS:
                        # Too many consecutive failures — close the stream so the
                        # browser's EventSource retry fires a fresh HTTP request.
                        logger.error(
                            "SSE: exceeded %d consecutive pub/sub errors, closing stream for client reconnect",
                            _PUBSUB_MAX_ERRORS,
                        )
                        return

                    # Wait before attempting reconnect (runs in the event loop
                    # so other coroutines / disconnect checks are not blocked)
                    await asyncio.sleep(backoff)

                    # Attempt to re-establish the dedicated connection + subscription
                    sse_client = _make_redis_connection()
                    if sse_client is not None:
                        try:
                            pubsub = sse_client.pubsub()
                            pubsub.psubscribe("dashboard:*")
                            pubsub_errors = 0  # reconnect succeeded — reset counter
                            logger.info("SSE: pub/sub reconnected successfully")
                        except Exception as reconnect_exc:
                            logger.warning("SSE: pub/sub resubscribe failed: %s", reconnect_exc)
                            _teardown_pubsub(pubsub, sse_client)
                            pubsub = None
                            sse_client = None
                    else:
                        logger.warning("SSE: Redis unavailable after reconnect attempt, staying in backoff loop")

                # Also check for session changes via engine status (pubsub might miss it)
                try:
                    status = _get_engine_status()
                    if status:
                        current_session = status.get("session_mode", "")
                        if current_session and current_session != last_session:
                            last_session = current_session
                            yield _make_session_event(current_session)
                except Exception:
                    pass

            else:
                # ---- Polling mode: check Redis cache periodically ----
                try:
                    # Check focus data
                    cached = _get_focus_from_cache()
                    if cached:
                        focus_hash = str(hash(cached))
                        if focus_hash != last_focus_hash:
                            last_focus_hash = focus_hash
                            if not _should_throttle("focus-update"):
                                yield _format_sse(data=cached, event="focus-update")

                                # Per-asset events
                                try:
                                    focus = json.loads(cached)
                                    for asset in focus.get("assets", []):
                                        symbol = asset.get("symbol", "").lower().replace(" ", "_")
                                        if symbol and not _should_throttle(f"{symbol}-update"):
                                            asset_json = json.dumps(asset, default=str)
                                            yield _format_sse(
                                                data=asset_json,
                                                event=f"{symbol}-update",
                                            )

                                    if focus.get("no_trade"):
                                        yield _format_sse(
                                            data=json.dumps(
                                                {
                                                    "no_trade": True,
                                                    "reason": focus.get("no_trade_reason", ""),
                                                }
                                            ),
                                            event="no-trade-alert",
                                        )
                                except (json.JSONDecodeError, TypeError):
                                    pass

                    # Check positions
                    pos = _get_positions_from_cache()
                    if pos:
                        pos_hash = str(hash(pos))
                        if pos_hash != last_positions_hash:
                            last_positions_hash = pos_hash
                            if not _should_throttle("positions-update"):
                                yield _format_sse(data=pos, event="positions-update")

                    # Check Grok update (TASK-602)
                    grok = _get_grok_from_cache()
                    if grok:
                        grok_hash = str(hash(grok))
                        if grok_hash != last_grok_hash:
                            last_grok_hash = grok_hash
                            if not _should_throttle("grok-update"):
                                yield _format_sse(data=grok, event="grok-update")

                    # Check risk status
                    risk = _get_risk_from_cache()
                    if risk:
                        risk_hash = str(hash(risk))
                        if risk_hash != last_risk_hash:
                            last_risk_hash = risk_hash
                            if not _should_throttle("risk-update"):
                                yield _format_sse(data=risk, event="risk-update")

                    # Check Reddit signals
                    reddit = _get_reddit_signals()
                    if reddit:
                        reddit_hash = str(hash(reddit))
                        if reddit_hash != last_reddit_hash:
                            last_reddit_hash = reddit_hash
                            if not _should_throttle("reddit-signal"):
                                yield _format_sse(data=reddit, event="reddit-signal")

                    # Check ORB status (TASK-801)
                    orb = _get_orb_from_cache()
                    if orb:
                        orb_hash = str(hash(orb))
                        if orb_hash != last_orb_hash:
                            last_orb_hash = orb_hash
                            if not _should_throttle("orb-update"):
                                yield _format_sse(data=orb, event="orb-update")

                    # Check session
                    status = _get_engine_status()
                    if status:
                        current_session = status.get("session_mode", "")
                        if current_session and current_session != last_session:
                            last_session = current_session
                            yield _make_session_event(current_session)

                except Exception as exc:
                    logger.debug("Polling read error: %s", exc)

            # Small sleep to avoid busy-looping
            # Pub/sub mode: 0.25s (messages arrive via subscription)
            # Polling mode: 5s (we poll cache keys)
            sleep_time = 0.25 if use_pubsub else 5.0
            await asyncio.sleep(sleep_time)

    except asyncio.CancelledError:
        logger.debug("SSE generator cancelled")
    except GeneratorExit:
        logger.debug("SSE generator exited")
    except Exception as exc:
        logger.error("SSE generator error: %s", exc, exc_info=True)
    finally:
        # Clean up the dedicated pub/sub connection (not the shared cache._r)
        _teardown_pubsub(pubsub, sse_client)
        logger.debug("SSE connection closed")


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------


@router.get("/sse/dashboard")
async def sse_dashboard(request: Request):
    """Server-Sent Events endpoint for live dashboard updates.

    Connect from HTMX:
        <div hx-ext="sse" sse-connect="/sse/dashboard">
            <div sse-swap="focus-update" hx-swap="innerHTML">...</div>
            <div sse-swap="mgc-update" hx-swap="innerHTML">...</div>
            <div sse-swap="no-trade-alert" hx-swap="innerHTML">...</div>
            <div sse-swap="grok-update" hx-swap="innerHTML">...</div>
            <div sse-swap="risk-update" hx-swap="innerHTML">...</div>
            <div sse-swap="heartbeat">...</div>
        </div>

    Or from JavaScript:
        const es = new EventSource('/sse/dashboard');
        es.addEventListener('focus-update', (e) => { ... });
        es.addEventListener('mgc-update', (e) => { ... });
        es.addEventListener('no-trade-alert', (e) => { ... });
        es.addEventListener('grok-update', (e) => { ... });
        es.addEventListener('risk-update', (e) => { ... });
        es.addEventListener('heartbeat', (e) => { ... });

    Events:
        - connected         — Initial connection confirmation
        - focus-update      — Full focus payload (JSON)
        - {symbol}-update   — Per-asset update (JSON), e.g. mgc-update
        - no-trade-alert    — No-trade condition (JSON)
        - session-change    — Session mode changed (JSON)
        - positions-update  — Live positions changed (JSON)
        - grok-update       — Grok compact AI summary (TASK-602)
        - risk-update       — Risk status changed (TASK-502)
        - heartbeat         — Keep-alive with server time (JSON)

    Catch-up: On connect, the last 8 focus updates from the Redis Stream
    are sent immediately so the client doesn't miss anything.

    Throttling: Max 1 update per asset per 7 seconds to avoid overwhelming
    the browser during high-frequency engine recomputation.

    Auto-reconnect: The retry directive tells the browser to reconnect
    after 3 seconds if the connection drops. HTMX handles this natively.
    """
    return StreamingResponse(
        _dashboard_event_generator(request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate, max-age=0",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        },
    )


# ---------------------------------------------------------------------------
# CHARTS-A: Per-symbol bar SSE stream
# ---------------------------------------------------------------------------


async def _bars_event_generator(request: Request, symbol: str) -> AsyncGenerator[str]:
    """Yield SSE events for 1-minute bar closes for *symbol*.

    Flow:
    1. On connect, send the most-recent stored bar as a ``bar`` event so the
       client can prime the chart without waiting for the next bar close.
    2. Subscribe to Redis pub/sub channel ``engine:bars_1m:{symbol}``.
       The engine publishes a JSON bar payload to this channel on every 1m
       bar close: ``{"time": <unix>, "open": …, "high": …, "low": …,
       "close": …, "volume": …}``.
    3. If no Redis event arrives within 60 seconds, re-fetch the latest bar
       from the ``/bars/{symbol}`` store and re-stream it as a keepalive so
       the chart stays fresh even when the market is slow.
    4. Emit ``heartbeat`` events every 30 seconds so the browser's
       ``EventSource`` doesn't time out on idle symbols.

    Event types
    -----------
    ``bar``       — ``{"time": <unix_s>, "open":…, "high":…, "low":…,
                       "close":…, "volume":…}``
    ``heartbeat`` — ``{"symbol": "<sym>", "ts": "<iso>"}``
    ``error``     — ``{"detail": "<msg>"}``
    """
    yield _format_sse(
        data=json.dumps({"symbol": symbol, "ts": datetime.now(tz=_EST).isoformat()}),
        event="connected",
        retry=3000,
    )

    # ── Prime: send the latest stored bar immediately ────────────────────────
    try:
        import pandas as pd

        from lib.services.data.api.bars import _fetch_stored_bars

        df = _fetch_stored_bars(symbol, interval="1m", days_back=1)
        if df is not None and not df.empty:
            last = df.iloc[-1]
            ts_val = str(df.index[-1])
            try:
                _ts = pd.Timestamp(ts_val)
                ts_unix = int(_ts.timestamp()) if _ts is not pd.NaT else int(datetime.now(tz=_EST).timestamp())  # type: ignore[union-attr]
            except Exception:
                ts_unix = int(datetime.now(tz=_EST).timestamp())

            # TODO: replace with lib.core.utils.safe_float (nested closure — can't be a top-level import without restructuring)
            def _safe_float(v: Any) -> float:
                try:
                    return round(float(v), 6)
                except (TypeError, ValueError):
                    return 0.0

            bar_payload = {
                "time": ts_unix,
                "open": _safe_float(last.get("Open", last.get("open", 0))),
                "high": _safe_float(last.get("High", last.get("high", 0))),
                "low": _safe_float(last.get("Low", last.get("low", 0))),
                "close": _safe_float(last.get("Close", last.get("close", 0))),
                "volume": int(last.get("Volume", last.get("volume", 0)) or 0),
            }
            yield _format_sse(data=json.dumps(bar_payload), event="bar")
    except Exception as exc:
        logger.debug("bars_sse: prime failed for %s: %s", symbol, exc)

    # ── Live: subscribe to Redis pub/sub ─────────────────────────────────────
    pubsub = None
    redis_conn = None
    channel = f"engine:bars_1m:{symbol}"
    last_redis_event = time.monotonic()
    heartbeat_deadline = time.monotonic() + _HEARTBEAT_INTERVAL
    REDIS_TIMEOUT = 60.0  # seconds without Redis event before fallback

    try:
        redis_conn = _make_redis_connection()
        if redis_conn is not None:
            pubsub = redis_conn.pubsub()
            pubsub.subscribe(channel)
    except Exception as exc:
        logger.debug("bars_sse: Redis subscribe failed for %s: %s", symbol, exc)

    try:
        while True:
            if await request.is_disconnected():
                break

            now = time.monotonic()

            # ── Heartbeat ────────────────────────────────────────────────────
            if now >= heartbeat_deadline:
                yield _format_sse(
                    data=json.dumps({"symbol": symbol, "ts": datetime.now(tz=_EST).isoformat()}),
                    event="heartbeat",
                )
                heartbeat_deadline = now + _HEARTBEAT_INTERVAL

            # ── Redis pub/sub message ─────────────────────────────────────────
            if pubsub is not None:
                try:
                    msg = pubsub.get_message(ignore_subscribe_messages=True, timeout=0.1)
                    if msg and msg.get("type") == "message":
                        raw = msg.get("data", b"")
                        if isinstance(raw, bytes):
                            raw = raw.decode("utf-8")
                        try:
                            bar_data = json.loads(raw)
                            # Ensure time is unix seconds (engine may publish ms)
                            if "time" in bar_data:
                                t = int(bar_data["time"])
                                bar_data["time"] = t // 1000 if t > 10_000_000_000 else t
                            yield _format_sse(data=json.dumps(bar_data), event="bar")
                            last_redis_event = now
                        except Exception as exc:
                            logger.debug("bars_sse: bad bar payload for %s: %s", symbol, exc)
                except Exception as exc:
                    logger.debug("bars_sse: pubsub error for %s: %s", symbol, exc)
                    pubsub = None  # fall through to fallback on next iteration

            # ── Fallback: re-fetch latest bar if Redis has been silent ────────
            if now - last_redis_event > REDIS_TIMEOUT:
                try:
                    import pandas as pd

                    from lib.services.data.api.bars import _fetch_stored_bars

                    df = _fetch_stored_bars(symbol, interval="1m", days_back=1)
                    if df is not None and not df.empty:
                        last_row = df.iloc[-1]
                        ts_val = str(df.index[-1])
                        try:
                            _ts2 = pd.Timestamp(ts_val)
                            ts_unix = (
                                int(_ts2.timestamp()) if _ts2 is not pd.NaT else int(datetime.now(tz=_EST).timestamp())  # type: ignore[union-attr]
                            )
                        except Exception:
                            ts_unix = int(datetime.now(tz=_EST).timestamp())

                        def _sf(v: Any) -> float:
                            try:
                                return round(float(v), 6)
                            except (TypeError, ValueError):
                                return 0.0

                        yield _format_sse(
                            data=json.dumps(
                                {
                                    "time": ts_unix,
                                    "open": _sf(last_row.get("Open", last_row.get("open", 0))),
                                    "high": _sf(last_row.get("High", last_row.get("high", 0))),
                                    "low": _sf(last_row.get("Low", last_row.get("low", 0))),
                                    "close": _sf(last_row.get("Close", last_row.get("close", 0))),
                                    "volume": int(last_row.get("Volume", last_row.get("volume", 0)) or 0),
                                }
                            ),
                            event="bar",
                        )
                except Exception as exc:
                    logger.debug("bars_sse: fallback fetch failed for %s: %s", symbol, exc)
                last_redis_event = now  # reset regardless so we don't spam

            await asyncio.sleep(0.2)

    except asyncio.CancelledError:
        pass
    except Exception as exc:
        logger.debug("bars_sse: generator error for %s: %s", symbol, exc)
        yield _format_sse(data=json.dumps({"detail": str(exc)}), event="error")
    finally:
        _teardown_pubsub(pubsub, redis_conn)


@router.get("/sse/bars/{symbol:path}")
async def sse_bars(symbol: str, request: Request):
    """Server-Sent Events endpoint for live 1-minute bar closes for a symbol.

    Streams ``bar`` events as each 1-minute candle closes on the engine.
    The browser chart calls ``candleSeries.update(bar)`` on each event.

    Connect from JavaScript::

        const es = new EventSource('/sse/bars/MGC=F');
        es.addEventListener('bar', (e) => {
            const bar = JSON.parse(e.data);
            candleSeries.update(bar);  // Lightweight Charts
        });
        es.addEventListener('heartbeat', (e) => console.log('alive', e.data));

    Bar payload::

        {"time": 1712345660, "open": 2300.1, "high": 2301.5,
         "low": 2299.8, "close": 2301.0, "volume": 342}

    Fallback: if Redis is not available or the engine is not publishing,
    the endpoint falls back to polling ``/bars/{symbol}`` every 60 seconds
    and re-streaming the latest bar so the chart stays reasonably fresh.
    """
    return StreamingResponse(
        _bars_event_generator(request, symbol),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate, max-age=0",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/sse/health")
async def sse_health():
    """Health check for SSE subsystem.

    Returns the status of Redis connectivity and stream stats.
    """
    r = _get_redis()
    redis_ok = r is not None

    stream_length: int = 0
    if redis_ok and r is not None:
        try:
            info = r.xinfo_stream("dashboard:stream:focus")  # type: ignore[union-attr]
            if isinstance(info, dict):
                stream_length = int(info.get(b"length", info.get("length", 0)))
        except Exception:
            pass

    return {
        "status": "ok" if redis_ok else "degraded",
        "redis_connected": redis_ok,
        "stream_length": stream_length,
        "mode": "pubsub" if redis_ok else "polling",
        "throttle_seconds": _THROTTLE_SECONDS,
        "heartbeat_interval": _HEARTBEAT_INTERVAL,
        "catchup_count": _CATCHUP_COUNT,
    }
