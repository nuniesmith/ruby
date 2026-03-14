"""
Opening Range Breakout (ORB) — ``rb.open`` Package
====================================================
Unified public façade for the ORB sub-system.

Sub-modules
-----------
sessions    — ORBSession dataclass, all session instances, lookup tables,
              session-status helpers.
assets      — SESSION_ASSETS, CRYPTO_SYMBOL_OVERRIDES, asset helpers,
              runtime Kraken crypto-ticker injection.
models      — ORBResult, MultiSessionORBResult dataclasses.
detector    — compute_atr, compute_opening_range, detect_opening_range_breakout,
              detect_all_sessions, scan_orb_all_assets,
              scan_orb_all_sessions_all_assets.
publisher   — publish_orb_alert, publish_multi_session_orb, clear_orb_alert,
              Redis key helpers and constants.

Backward compatibility
----------------------
All symbols that were previously imported from
``lib.services.engine.rb.orb`` are re-exported here unchanged, so
existing callers only need to update their import path::

    # old
    from lib.trading.strategies.rb.orb import LONDON_SESSION, detect_opening_range_breakout

    # new
    from lib.trading.strategies.rb.open import LONDON_SESSION, detect_opening_range_breakout
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Assets — per-session ticker lists, crypto overrides, helpers
# ---------------------------------------------------------------------------
from lib.trading.strategies.rb.open.assets import (
    CRYPTO_SYMBOL_OVERRIDES,
    SESSION_ASSETS,
    get_session_assets,
    get_symbol_session_overrides,
)

# ---------------------------------------------------------------------------
# Detector — pure detection logic
# ---------------------------------------------------------------------------
from lib.trading.strategies.rb.open.detector import (
    compute_atr,
    compute_opening_range,
    detect_all_sessions,
    detect_opening_range_breakout,
    scan_orb_all_assets,
    scan_orb_all_sessions_all_assets,
)

# ---------------------------------------------------------------------------
# Models — result dataclasses
# ---------------------------------------------------------------------------
from lib.trading.strategies.rb.open.models import (
    MultiSessionORBResult,
    ORBResult,
)

# ---------------------------------------------------------------------------
# Publisher — Redis I/O
# ---------------------------------------------------------------------------
from lib.trading.strategies.rb.open.publisher import (
    REDIS_KEY_ORB,
    REDIS_KEY_ORB_TS,
    REDIS_PUBSUB_ORB,
    REDIS_TTL,
    clear_orb_alert,
    publish_multi_session_orb,
    publish_orb_alert,
)

# ---------------------------------------------------------------------------
# Sessions — dataclass, instances, lookups, helpers
# ---------------------------------------------------------------------------
from lib.trading.strategies.rb.open.sessions import (
    ATR_PERIOD,
    BREAKOUT_ATR_MULTIPLIER,
    CME_OPEN_SESSION,
    CME_SETTLEMENT_SESSION,
    CRYPTO_UTC_MIDNIGHT_SESSION,
    CRYPTO_UTC_NOON_SESSION,
    DATASET_SESSIONS,
    DAYTIME_SESSIONS,
    FRANKFURT_SESSION,
    LONDON_NY_SESSION,
    LONDON_SESSION,
    MAX_OR_BARS,
    MIN_OR_BARS,
    OR_END,
    OR_START,
    ORB_SESSIONS,
    OVERNIGHT_SESSIONS,
    SESSION_BY_KEY,
    SHANGHAI_SESSION,
    SYDNEY_SESSION,
    TOKYO_SESSION,
    US_SESSION,
    ORBSession,
    get_active_session_keys,
    get_active_sessions,
    get_session_by_key,
    get_session_for_utc,
    get_session_status,
    is_any_session_active,
)

__all__ = [
    # --- Sessions ---
    "ORBSession",
    "CME_OPEN_SESSION",
    "SYDNEY_SESSION",
    "TOKYO_SESSION",
    "SHANGHAI_SESSION",
    "FRANKFURT_SESSION",
    "LONDON_SESSION",
    "LONDON_NY_SESSION",
    "US_SESSION",
    "CME_SETTLEMENT_SESSION",
    "CRYPTO_UTC_MIDNIGHT_SESSION",
    "CRYPTO_UTC_NOON_SESSION",
    "ORB_SESSIONS",
    "SESSION_BY_KEY",
    "OVERNIGHT_SESSIONS",
    "DAYTIME_SESSIONS",
    "DATASET_SESSIONS",
    # Legacy scalar aliases
    "OR_START",
    "OR_END",
    "ATR_PERIOD",
    "BREAKOUT_ATR_MULTIPLIER",
    "MIN_OR_BARS",
    "MAX_OR_BARS",
    # Session helpers
    "get_session_for_utc",
    "get_active_session_keys",
    "get_active_sessions",
    "is_any_session_active",
    "get_session_status",
    "get_session_by_key",
    # --- Assets ---
    "SESSION_ASSETS",
    "CRYPTO_SYMBOL_OVERRIDES",
    "get_session_assets",
    "get_symbol_session_overrides",
    # --- Models ---
    "ORBResult",
    "MultiSessionORBResult",
    # --- Detector ---
    "compute_atr",
    "compute_opening_range",
    "detect_opening_range_breakout",
    "detect_all_sessions",
    "scan_orb_all_assets",
    "scan_orb_all_sessions_all_assets",
    # --- Publisher ---
    "REDIS_KEY_ORB",
    "REDIS_KEY_ORB_TS",
    "REDIS_PUBSUB_ORB",
    "REDIS_TTL",
    "publish_orb_alert",
    "publish_multi_session_orb",
    "clear_orb_alert",
]
