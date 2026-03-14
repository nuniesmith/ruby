"""
Opening Range Breakout — Redis Publisher
=========================================
Handles all Redis I/O for ORB results: key helpers, pub/sub publish,
multi-session combined payloads, and end-of-day clearing.

Redis key scheme
----------------
Per-session (TTL 300 s):
    engine:orb:{session_key}        — latest ORBResult JSON for that session
    engine:orb:{session_key}:ts     — ISO timestamp of last publish

Legacy combined (backward compat with dashboard):
    engine:orb                      — best / most recent ORBResult JSON
    engine:orb:ts                   — ISO timestamp

Multi-session combined:
    engine:orb:multi:{symbol}       — MultiSessionORBResult JSON (TTL 300 s)

Pub/Sub channels:
    dashboard:orb:{session_key}     — per-session trigger for SSE stream
    dashboard:orb                   — legacy trigger (always fired alongside)

Public API
----------
publish_orb_alert(result, session)            → bool
publish_multi_session_orb(multi)              → bool
clear_orb_alert(session)                      → bool
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import TYPE_CHECKING
from zoneinfo import ZoneInfo

from lib.trading.strategies.rb.open.sessions import ORB_SESSIONS, ORBSession

if TYPE_CHECKING:
    from lib.trading.strategies.rb.open.models import MultiSessionORBResult, ORBResult

logger = logging.getLogger("engine.orb")

_EST = ZoneInfo("America/New_York")

# ---------------------------------------------------------------------------
# TTL for all ORB Redis keys (5 minutes)
# ---------------------------------------------------------------------------
REDIS_TTL = 300

# ---------------------------------------------------------------------------
# Legacy combined keys (kept for backward compatibility with older dashboard
# code that reads a single engine:orb key rather than the per-session keys)
# ---------------------------------------------------------------------------
REDIS_KEY_ORB = "engine:orb"
REDIS_KEY_ORB_TS = "engine:orb:ts"
REDIS_PUBSUB_ORB = "dashboard:orb"


# ---------------------------------------------------------------------------
# Key helpers
# ---------------------------------------------------------------------------


def _redis_key_orb(session: ORBSession) -> str:
    """Return the Redis storage key for *session*'s latest ORB result."""
    return f"engine:orb:{session.key}"


def _redis_key_orb_ts(session: ORBSession) -> str:
    """Return the Redis storage key for *session*'s last-publish timestamp."""
    return f"engine:orb:{session.key}:ts"


def _redis_pubsub_orb(session: ORBSession) -> str:
    """Return the Redis pub/sub channel for *session*'s ORB alerts."""
    return f"dashboard:orb:{session.key}"


# ---------------------------------------------------------------------------
# Single-result publisher
# ---------------------------------------------------------------------------


def publish_orb_alert(result: ORBResult, session: ORBSession | None = None) -> bool:
    """Publish an ORB alert to Redis for SSE / dashboard consumption.

    Writes to:

    * ``engine:orb:{session_key}``    — session-specific result (TTL 300 s)
    * ``engine:orb:{session_key}:ts`` — timestamp of last publish
    * ``engine:orb``                  — legacy combined key (always updated)
    * ``engine:orb:ts``               — legacy timestamp
    * Pub/sub ``dashboard:orb:{session_key}`` — per-session SSE trigger
    * Pub/sub ``dashboard:orb``       — legacy SSE trigger

    Args:
        result:  The :class:`~models.ORBResult` to publish.
        session: The :class:`~sessions.ORBSession` the result belongs to.
                 When ``None`` the session is resolved from
                 ``result.session_key`` via :data:`~sessions.ORB_SESSIONS`.

    Returns:
        ``True`` on success, ``False`` on any error.
    """
    try:
        from lib.core.cache import REDIS_AVAILABLE, _r, cache_set
    except ImportError:
        logger.error("Cannot import cache module for ORB publish")
        return False

    # Resolve session object from result when not provided
    if session is None:
        for s in ORB_SESSIONS:
            if s.key == result.session_key:
                session = s
                break

    try:
        payload_json = json.dumps(result.to_dict(), default=str)
    except (TypeError, ValueError) as exc:
        logger.error("Failed to serialise ORB result: %s", exc)
        return False

    now_iso = datetime.now(tz=_EST).isoformat().encode()

    try:
        # --- Per-session keys ---
        if session is not None:
            cache_set(_redis_key_orb(session), payload_json.encode(), ttl=REDIS_TTL)
            cache_set(_redis_key_orb_ts(session), now_iso, ttl=REDIS_TTL)

        # --- Legacy combined keys ---
        cache_set(REDIS_KEY_ORB, payload_json.encode(), ttl=REDIS_TTL)
        cache_set(REDIS_KEY_ORB_TS, now_iso, ttl=REDIS_TTL)

        # --- Pub/sub ---
        if REDIS_AVAILABLE and _r is not None:
            try:
                if session is not None:
                    _r.publish(_redis_pubsub_orb(session), payload_json)
                _r.publish(REDIS_PUBSUB_ORB, payload_json)
            except Exception as exc:
                logger.debug("ORB pub/sub publish failed (non-fatal): %s", exc)

        logger.info(
            "ORB [%s] alert published: %s %s (OR %.4f–%.4f)",
            result.session_name or "unknown",
            result.direction,
            result.symbol,
            result.or_low,
            result.or_high,
        )
        return True

    except Exception as exc:
        logger.error("Failed to publish ORB to Redis: %s", exc)
        return False


# ---------------------------------------------------------------------------
# Multi-session publisher
# ---------------------------------------------------------------------------


def publish_multi_session_orb(multi: MultiSessionORBResult) -> bool:
    """Publish all session results from a :class:`~models.MultiSessionORBResult`.

    Publishes each individual session result via :func:`publish_orb_alert`,
    then writes a combined payload to ``engine:orb:multi:{symbol}`` for the
    dashboard's multi-session panel.

    Args:
        multi: Aggregated results for one symbol across all sessions.

    Returns:
        ``True`` if every publish succeeded, ``False`` if any failed.
    """
    try:
        from lib.core.cache import cache_set
    except ImportError:
        logger.error("Cannot import cache module for multi-session ORB publish")
        return False

    success = True

    # Publish each individual session result
    for session_key, result in multi.sessions.items():
        session_obj: ORBSession | None = None
        for s in ORB_SESSIONS:
            if s.key == session_key:
                session_obj = s
                break
        if not publish_orb_alert(result, session=session_obj):
            success = False

    # Write the combined multi-session payload
    try:
        combined_payload = json.dumps(multi.to_dict(), default=str)
        cache_set(
            f"engine:orb:multi:{multi.symbol}",
            combined_payload.encode(),
            ttl=REDIS_TTL,
        )
    except Exception as exc:
        logger.error("Failed to publish multi-session ORB combined payload: %s", exc)
        success = False

    return success


# ---------------------------------------------------------------------------
# Alert clearing (end-of-day / manual reset)
# ---------------------------------------------------------------------------


def clear_orb_alert(session: ORBSession | None = None) -> bool:
    """Clear ORB alert(s) from Redis.

    When *session* is ``None``, clears all session-specific keys and the
    legacy combined keys.  When a specific session is provided, only that
    session's keys are cleared.

    Uses ``DEL`` when a live Redis connection is available so keys vanish
    immediately — dashboards will never briefly read a stale empty payload.
    Falls back to ``cache_set(..., ttl=1)`` for the in-memory cache used in
    tests / offline mode (the in-memory store has no native DEL).

    Args:
        session: Session to clear.  ``None`` clears everything.

    Returns:
        ``True`` on success, ``False`` if the cache module is unavailable.
    """
    try:
        from lib.core.cache import REDIS_AVAILABLE, _r, cache_set
    except ImportError:
        return False

    def _delete(key: str) -> None:
        """Delete *key* immediately via DEL, or expire it in 1 s as fallback."""
        if REDIS_AVAILABLE and _r is not None:
            try:
                _r.delete(key)
                return
            except Exception as exc:
                logger.debug("Redis DEL failed for %s (falling back to ttl=1): %s", key, exc)
        # In-memory fallback: set empty value with a 1-second TTL
        cache_set(key, b"", ttl=1)

    try:
        if session is not None:
            _delete(_redis_key_orb(session))
            _delete(_redis_key_orb_ts(session))
        else:
            for s in ORB_SESSIONS:
                _delete(_redis_key_orb(s))
                _delete(_redis_key_orb_ts(s))
            _delete(REDIS_KEY_ORB)
            _delete(REDIS_KEY_ORB_TS)
        return True
    except Exception:
        return False
