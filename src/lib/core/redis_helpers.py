"""
Centralized Redis helpers for the futures trading engine.

Provides thin, consistent wrappers around raw Redis pub/sub and key-value
operations so individual modules don't need to repeat the same boilerplate:

    from lib.core.redis_helpers import publish, cache_set_json, cache_get_json

Design principles:
  - All functions are safe to call even when Redis is unavailable (they log
    a debug message and return a sensible default).
  - JSON serialisation uses ``default=str`` so datetime / Decimal values
    don't cause crashes.
  - TTLs are expressed as named constants so every caller uses the same
    values.
  - pub/sub channel names are defined as module-level constants so a typo
    in one place doesn't create a ghost channel.

Public API
----------
publish(channel, payload)                    — fire-and-forget pub/sub
publish_json(channel, data)                  — serialise dict then publish
cache_set_json(key, data, ttl)               — set key with JSON payload
cache_get_json(key)                          -> dict | None
cache_set_raw(key, data, ttl)                — set key with raw bytes
cache_get_raw(key)                           -> bytes | None
delete(key)                                  — delete a single key
stream_add(stream, data, maxlen)             — XADD to a Redis stream
get_redis()                                  -> redis.Redis | None
is_available()                               -> bool

Per-session CNN gate API
------------------------
get_cnn_gate(session_key)                    -> bool
set_cnn_gate(session_key, enabled)           -> bool
get_all_cnn_gates()                          -> dict[str, bool]
reset_cnn_gate(session_key)                  -> bool
reset_all_cnn_gates()                        -> int

The per-session CNN gate lets you enable the hard CNN filter on individual
ORB sessions without touching the global ORB_CNN_GATE env var.  Redis keys
are of the form ``engine:config:cnn_gate:{session_key}`` and contain either
``"1"`` (enabled) or ``"0"`` (disabled).  When a key is absent the global
env-var fallback (``ORB_CNN_GATE``) is used, which preserves backward
compatibility.

Channel constants
-----------------
CH_LIVE, CH_RISK, CH_ORB, CH_ORB_LONDON, CH_ORB_US,
CH_NO_TRADE, CH_RISK_WARNING, CH_OVERNIGHT_WARNING,
CH_GROK, CH_BACKFILL, CH_ASSET_PREFIX

Key constants
-------------
KEY_DAILY_FOCUS, KEY_DAILY_FOCUS_TS, KEY_ENGINE_STATUS,
KEY_RISK_STATUS, KEY_GROK_UPDATE, KEY_DAILY_REPORT,
KEY_RETRAIN_CMD, KEY_RETRAIN_STATUS, KEY_BACKFILL_STATUS,
KEY_BARS_1M, KEY_BARS_15M, KEY_BARS_DAILY,
KEY_CNN_GATE_PREFIX

TTL constants
-------------
TTL_TICK (5 s), TTL_SHORT (60 s), TTL_MEDIUM (300 s),
TTL_LONG (3600 s), TTL_DAY (86400 s), TTL_REPORT (93600 s)
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, cast

logger = logging.getLogger("redis_helpers")

# ---------------------------------------------------------------------------
# Pub/Sub channel names
# ---------------------------------------------------------------------------

CH_LIVE = "dashboard:live"
CH_RISK = "dashboard:risk"
CH_ORB = "dashboard:orb"
CH_ORB_LONDON = "dashboard:orb:london"
CH_ORB_US = "dashboard:orb:us"
CH_NO_TRADE = "dashboard:no_trade"
CH_RISK_WARNING = "dashboard:risk_warning"
CH_OVERNIGHT_WARNING = "dashboard:overnight_warning"
CH_GROK = "dashboard:grok"
CH_BACKFILL = "dashboard:backfill"

# Prefix — append symbol to get per-asset channel, e.g. "dashboard:asset:MGC=F"
CH_ASSET_PREFIX = "dashboard:asset:"


def orb_channel(session: str) -> str:
    """Return the pub/sub channel for an ORB session.

    Args:
        session: ``"london"`` or ``"us"`` (case-insensitive).

    Returns:
        The channel string, e.g. ``"dashboard:orb:london"``.
    """
    return f"dashboard:orb:{session.lower()}"


def asset_channel(symbol: str) -> str:
    """Return the per-asset pub/sub channel for a symbol.

    Args:
        symbol: Ticker symbol, e.g. ``"MGC=F"``.

    Returns:
        The channel string, e.g. ``"dashboard:asset:MGC=F"``.
    """
    return f"{CH_ASSET_PREFIX}{symbol}"


# ---------------------------------------------------------------------------
# Redis key names
# ---------------------------------------------------------------------------

KEY_DAILY_FOCUS = "engine:daily_focus"
KEY_DAILY_FOCUS_TS = "engine:daily_focus:ts"
KEY_ENGINE_STATUS = "engine:status"
KEY_RISK_STATUS = "engine:risk_status"
KEY_GROK_UPDATE = "engine:grok_update"
KEY_DAILY_REPORT = "engine:daily_report"
KEY_RETRAIN_CMD = "engine:retrain_cmd"
KEY_RETRAIN_STATUS = "engine:retrain_status"
KEY_BACKFILL_STATUS = "engine:backfill_status"
KEY_MODEL_HEALTH = "engine:model_health"

# Per-session CNN gate config keys: ``engine:config:cnn_gate:{session_key}``
# Value is ``"1"`` (gate enabled) or ``"0"`` (gate disabled).
# Absence of the key means "fall back to ORB_CNN_GATE env var".
KEY_CNN_GATE_PREFIX = "engine:config:cnn_gate:"

# All known session keys — used for bulk operations.
_ALL_SESSION_KEYS: tuple[str, ...] = (
    "cme",
    "sydney",
    "tokyo",
    "shanghai",
    "frankfurt",
    "london",
    "london_ny",
    "us",
    "cme_settle",
)


def bars_1m_key(symbol: str) -> str:
    """Redis key for 1-minute bar cache for *symbol*."""
    return f"engine:bars_1m:{symbol}"


def bars_15m_key(symbol: str) -> str:
    """Redis key for 15-minute bar cache for *symbol*."""
    return f"engine:bars_15m:{symbol}"


def bars_daily_key(symbol: str) -> str:
    """Redis key for daily bar cache for *symbol*."""
    return f"engine:bars_daily:{symbol}"


# ---------------------------------------------------------------------------
# TTL constants (seconds)
# ---------------------------------------------------------------------------

TTL_TICK = 5  # live-tick data (almost no caching)
TTL_SHORT = 60  # 1-minute bar data, retrain status
TTL_MEDIUM = 300  # 5-minute focus data, engine status
TTL_LONG = 3_600  # optimisation results, grok updates (15 min used inline)
TTL_DAY = 86_400  # backfill status, audit data
TTL_REPORT = 93_600  # daily report (26 hours — survives until next report)


# ---------------------------------------------------------------------------
# Internal: lazy Redis accessor
# ---------------------------------------------------------------------------


def get_redis():
    """Return the shared Redis client, or ``None`` if unavailable.

    Imports from ``lib.core.cache`` (the canonical client) rather than
    creating a second connection.  Returns ``None`` if Redis is not
    configured or unreachable.
    """
    try:
        from lib.core.cache import REDIS_AVAILABLE, _r  # type: ignore[import-unresolved]

        if REDIS_AVAILABLE and _r is not None:
            return _r
    except Exception as exc:
        logger.debug("get_redis: cannot import cache module: %s", exc)
    return None


def is_available() -> bool:
    """Return ``True`` if a Redis connection is currently active."""
    return get_redis() is not None


# ---------------------------------------------------------------------------
# Pub/Sub helpers
# ---------------------------------------------------------------------------


def publish(channel: str, payload: str | bytes) -> bool:
    """Publish *payload* to *channel*.

    Args:
        channel: Redis pub/sub channel name.
        payload: Raw string or bytes to publish.

    Returns:
        ``True`` if published successfully, ``False`` otherwise.
    """
    r = get_redis()
    if r is None:
        logger.debug("publish(%s): Redis unavailable — skipped", channel)
        return False
    try:
        r.publish(channel, payload)
        return True
    except Exception as exc:
        logger.debug("publish(%s) failed: %s", channel, exc)
        return False


def publish_json(channel: str, data: dict[str, Any]) -> bool:
    """Serialise *data* to JSON and publish to *channel*.

    Uses ``default=str`` so datetimes, Decimals etc. don't crash.

    Args:
        channel: Redis pub/sub channel name.
        data: Dictionary to serialise and publish.

    Returns:
        ``True`` if published successfully, ``False`` otherwise.
    """
    try:
        payload = json.dumps(data, default=str)
    except Exception as exc:
        logger.warning("publish_json(%s): serialisation failed: %s", channel, exc)
        return False
    return publish(channel, payload)


# ---------------------------------------------------------------------------
# Key-value helpers
# ---------------------------------------------------------------------------


def cache_set_json(key: str, data: dict[str, Any], ttl: int) -> bool:
    """Serialise *data* to JSON and store under *key* with the given TTL.

    Args:
        key: Redis key name.
        data: Dictionary to store.
        ttl: Expiry in seconds.

    Returns:
        ``True`` on success, ``False`` on any error.
    """
    try:
        from lib.core.cache import cache_set  # type: ignore[import-unresolved]

        payload = json.dumps(data, default=str).encode()
        cache_set(key, payload, ttl)
        return True
    except Exception as exc:
        logger.debug("cache_set_json(%s) failed: %s", key, exc)
        return False


def cache_get_json(key: str) -> dict[str, Any] | None:
    """Fetch and deserialise a JSON value stored under *key*.

    Args:
        key: Redis key name.

    Returns:
        Parsed dict, or ``None`` if the key is missing or parse fails.
    """
    try:
        from lib.core.cache import cache_get  # type: ignore[import-unresolved]

        raw = cache_get(key)
        if raw is None:
            return None
        return json.loads(raw)  # type: ignore[arg-type]
    except Exception as exc:
        logger.debug("cache_get_json(%s) failed: %s", key, exc)
        return None


def cache_set_raw(key: str, data: bytes, ttl: int) -> bool:
    """Store raw bytes under *key* with the given TTL.

    Args:
        key: Redis key name.
        data: Raw bytes payload.
        ttl: Expiry in seconds.

    Returns:
        ``True`` on success, ``False`` on any error.
    """
    try:
        from lib.core.cache import cache_set  # type: ignore[import-unresolved]

        cache_set(key, data, ttl)
        return True
    except Exception as exc:
        logger.debug("cache_set_raw(%s) failed: %s", key, exc)
        return False


def cache_get_raw(key: str) -> bytes | None:
    """Fetch raw bytes stored under *key*.

    Args:
        key: Redis key name.

    Returns:
        Raw bytes, or ``None`` if the key is missing or an error occurs.
    """
    try:
        from lib.core.cache import cache_get  # type: ignore[import-unresolved]

        return cache_get(key)
    except Exception as exc:
        logger.debug("cache_get_raw(%s) failed: %s", key, exc)
        return None


def delete(key: str) -> bool:
    """Delete *key* from Redis (or in-memory fallback).

    Args:
        key: Redis key name.

    Returns:
        ``True`` if the key was deleted (or didn't exist), ``False`` on error.
    """
    r = get_redis()
    if r is not None:
        try:
            r.delete(key)
            return True
        except Exception as exc:
            logger.debug("delete(%s) via Redis failed: %s — trying cache fallback", key, exc)

    # Fallback: remove from in-memory cache dict directly
    try:
        from lib.core.cache import _mem_cache  # type: ignore[import-unresolved]

        _mem_cache.pop(key, None)
        return True
    except Exception as exc:
        logger.debug("delete(%s) fallback failed: %s", key, exc)
        return False


# ---------------------------------------------------------------------------
# Stream helpers
# ---------------------------------------------------------------------------


def stream_add(stream: str, data: dict[str, str], maxlen: int = 100) -> bool:
    """XADD *data* to a Redis stream, trimming to *maxlen* entries.

    Args:
        stream: Redis stream key name.
        data: Field-value dict (values must be strings).
        maxlen: Approximate maximum entries to retain.

    Returns:
        ``True`` if the entry was added, ``False`` otherwise.
    """
    r = get_redis()
    if r is None:
        logger.debug("stream_add(%s): Redis unavailable — skipped", stream)
        return False
    try:
        r.xadd(stream, cast("Any", data), maxlen=maxlen, approximate=True)
        return True
    except Exception as exc:
        logger.debug("stream_add(%s) failed: %s", stream, exc)
        return False


# ---------------------------------------------------------------------------
# Per-session CNN gate helpers
# ---------------------------------------------------------------------------


def _cnn_gate_key(session_key: str) -> str:
    """Return the Redis key for the CNN gate flag of *session_key*."""
    return f"{KEY_CNN_GATE_PREFIX}{session_key.lower().strip()}"


def get_cnn_gate(session_key: str | None) -> bool:
    """Return whether the CNN hard gate is enabled for *session_key*.

    Resolution order:
      1. Redis key ``engine:config:cnn_gate:{session_key}`` — per-session
         override set via :func:`set_cnn_gate` or the dashboard.
      2. ``ORB_CNN_GATE`` environment variable (``"1"`` = enabled globally).
      3. Default: **disabled** (``False``).

    This means you can enable the gate for overnight sessions selectively
    without flipping the global env var::

        # Enable CNN gate for overnight sessions only:
        set_cnn_gate("cme", True)
        set_cnn_gate("sydney", True)
        set_cnn_gate("tokyo", True)
        set_cnn_gate("shanghai", True)

        # Daytime sessions still use the env-var default (off by default):
        get_cnn_gate("london")   # → False  (unless ORB_CNN_GATE=1)
        get_cnn_gate("cme")      # → True   (Redis override)

    Args:
        session_key: ORBSession.key string (e.g. ``"london"``, ``"tokyo"``).
                     ``None`` or empty string falls through to the env var.

    Returns:
        ``True`` if the CNN hard gate should block low-probability signals
        for this session; ``False`` otherwise.
    """
    if session_key:
        raw = cache_get_raw(_cnn_gate_key(session_key))
        if raw is not None:
            val = raw.decode("utf-8").strip()
            return val == "1"

    # Fall back to global env var
    return os.environ.get("ORB_CNN_GATE", "0").strip() == "1"


def set_cnn_gate(session_key: str, enabled: bool, ttl: int = TTL_DAY * 30) -> bool:
    """Enable or disable the CNN hard gate for *session_key* in Redis.

    The value persists for *ttl* seconds (default 30 days) so it survives
    engine restarts.  Use :func:`reset_cnn_gate` to remove the override and
    revert to the env-var default.

    Args:
        session_key: ORBSession.key string (e.g. ``"cme"``, ``"us"``).
        enabled: ``True`` to enable the hard gate; ``False`` to disable.
        ttl: Key lifetime in seconds (default: 30 days).

    Returns:
        ``True`` if the value was stored successfully; ``False`` on error.

    Example::

        # Enable gate for overnight sessions after signal quality validates:
        from lib.core.redis_helpers import set_cnn_gate
        set_cnn_gate("cme", True)
        set_cnn_gate("sydney", True)
        set_cnn_gate("tokyo", True)
        set_cnn_gate("shanghai", True)
    """
    key = _cnn_gate_key(session_key)
    value = b"1" if enabled else b"0"
    ok = cache_set_raw(key, value, ttl)
    if ok:
        logger.info(
            "CNN gate %s for session '%s' (key=%s, ttl=%ds)",
            "ENABLED" if enabled else "DISABLED",
            session_key,
            key,
            ttl,
        )
    else:
        logger.warning("Failed to set CNN gate for session '%s' in Redis", session_key)
    return ok


def get_all_cnn_gates() -> dict[str, bool | None]:
    """Return the CNN gate state for every known session.

    Returns a dict mapping session key → gate state:
      - ``True``  — gate explicitly enabled in Redis.
      - ``False`` — gate explicitly disabled in Redis.
      - ``None``  — no Redis override; effective value comes from
        ``ORB_CNN_GATE`` env var (see :func:`get_cnn_gate`).

    The global env-var fallback is shown separately under the special key
    ``"_global_env"`` so the caller can display the full picture.

    Returns:
        Dict of session key → ``True`` / ``False`` / ``None``, plus
        ``"_global_env"`` → ``True`` / ``False``.

    Example::

        gates = get_all_cnn_gates()
        # {'cme': True, 'sydney': True, 'tokyo': None, ..., '_global_env': False}
    """
    result: dict[str, bool | None] = {}
    for sk in _ALL_SESSION_KEYS:
        raw = cache_get_raw(_cnn_gate_key(sk))
        if raw is None:
            result[sk] = None  # no override — env-var fallback applies
        else:
            val = raw.decode("utf-8").strip()
            result[sk] = val == "1"

    # Show the global env-var default so callers can compute effective values
    result["_global_env"] = os.environ.get("ORB_CNN_GATE", "0").strip() == "1"
    return result


def reset_cnn_gate(session_key: str) -> bool:
    """Remove the per-session CNN gate override for *session_key*.

    After this call, :func:`get_cnn_gate` for the session will fall back
    to the ``ORB_CNN_GATE`` env var.

    Args:
        session_key: ORBSession.key string.

    Returns:
        ``True`` on success (including if the key didn't exist).
    """
    ok = delete(_cnn_gate_key(session_key))
    if ok:
        logger.info("CNN gate override removed for session '%s' — now using env-var default", session_key)
    return ok


def reset_all_cnn_gates() -> int:
    """Remove all per-session CNN gate overrides.

    After this call every session falls back to the ``ORB_CNN_GATE`` env var.

    Returns:
        Number of keys that were deleted (0 if none were set).
    """
    count = 0
    for sk in _ALL_SESSION_KEYS:
        if delete(_cnn_gate_key(sk)):
            count += 1
    if count:
        logger.info("Removed %d CNN gate override(s) — all sessions now use env-var default", count)
    return count


# ---------------------------------------------------------------------------
# Convenience: publish + persist in one call
# ---------------------------------------------------------------------------


def publish_and_cache(
    channel: str,
    key: str,
    data: dict[str, Any],
    ttl: int,
) -> bool:
    """Serialise *data*, store it under *key*, and publish to *channel*.

    This is the most common pattern in the engine:

        cache_set("engine:foo", payload, ttl=300)
        r.publish("dashboard:foo", payload)

    Combined into a single call::

        publish_and_cache(CH_RISK, KEY_RISK_STATUS, status_dict, TTL_MEDIUM)

    Args:
        channel: Pub/sub channel to publish to.
        key: Redis key to persist the payload under.
        data: Dictionary to serialise.
        ttl: Key expiry in seconds.

    Returns:
        ``True`` if both the cache write and publish succeeded.
    """
    try:
        payload = json.dumps(data, default=str)
    except Exception as exc:
        logger.warning("publish_and_cache: serialisation failed: %s", exc)
        return False

    cached = cache_set_raw(key, payload.encode(), ttl)
    published = publish(channel, payload)
    return cached and published
