"""
Opening Range Breakout — Legacy Entry Point (Deprecated)
=========================================================
This file previously contained the full ORB implementation (~1 800 lines).
It has been refactored into focused sub-modules inside this package:

    sessions.py   — ORBSession dataclass, all session instances, lookup
                    tables, convenience groupings, and session-status helpers.
    assets.py     — SESSION_ASSETS, CRYPTO_SYMBOL_OVERRIDES, per-symbol
                    override helpers, and runtime Kraken crypto injection.
    models.py     — ORBResult and MultiSessionORBResult dataclasses.
    detector.py   — compute_atr, compute_opening_range,
                    detect_opening_range_breakout, detect_all_sessions,
                    scan_orb_all_assets, scan_orb_all_sessions_all_assets.
    publisher.py  — Redis key helpers, publish_orb_alert,
                    publish_multi_session_orb, clear_orb_alert.
    __init__.py   — Unified re-export façade (the preferred import target).

Migration guide
---------------
Replace any import of the form::

    from lib.trading.strategies.rb.open.main import <symbol>

with::

    from lib.trading.strategies.rb.open import <symbol>

or, if you want to keep imports short::

    import lib.services.engine.rb.open as orb
    orb.detect_opening_range_breakout(...)

All symbols that existed in the old monolithic ``main.py`` are re-exported
unchanged from the package ``__init__.py`` — no call-site logic needs to
change, only the import path.
"""

from __future__ import annotations

import warnings

warnings.warn(
    (
        "Importing from 'lib.trading.strategies.rb.open.main' is deprecated and will be "
        "removed in a future release. Update your import to:\n\n"
        "    from lib.trading.strategies.rb.open import <symbol>\n\n"
        "All symbols are re-exported unchanged from the package __init__.py."
    ),
    DeprecationWarning,
    stacklevel=2,
)

# Re-export the entire public API from the package so that any code still
# importing directly from this module continues to work without modification.
from lib.trading.strategies.rb.open import (  # noqa: F401, E402
    ATR_PERIOD,
    BREAKOUT_ATR_MULTIPLIER,
    CME_OPEN_SESSION,
    CME_SETTLEMENT_SESSION,
    CRYPTO_SYMBOL_OVERRIDES,
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
    REDIS_KEY_ORB,
    REDIS_KEY_ORB_TS,
    REDIS_PUBSUB_ORB,
    REDIS_TTL,
    SESSION_ASSETS,
    SESSION_BY_KEY,
    SHANGHAI_SESSION,
    SYDNEY_SESSION,
    TOKYO_SESSION,
    US_SESSION,
    MultiSessionORBResult,
    ORBResult,
    ORBSession,
    clear_orb_alert,
    compute_atr,
    compute_opening_range,
    detect_all_sessions,
    detect_opening_range_breakout,
    get_active_session_keys,
    get_active_sessions,
    get_session_assets,
    get_session_by_key,
    get_session_for_utc,
    get_session_status,
    get_symbol_session_overrides,
    is_any_session_active,
    publish_multi_session_orb,
    publish_orb_alert,
    scan_orb_all_assets,
    scan_orb_all_sessions_all_assets,
)
