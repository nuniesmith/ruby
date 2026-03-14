"""
Opening Range Breakout — Asset Lists & Symbol Overrides
========================================================
Canonical home for per-session asset focus lists, per-symbol parameter
overrides (crypto), and the runtime injection of Kraken crypto tickers
into session asset lists when ``ENABLE_KRAKEN_CRYPTO`` is set.

Public API
----------
SESSION_ASSETS          dict[str, list[str]]   — session key → tickers
CRYPTO_SYMBOL_OVERRIDES dict[str, dict]        — ticker → threshold overrides
get_session_assets(session)                    — safe accessor with fallback
get_symbol_session_overrides(symbol, session)  — per-symbol threshold overrides
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lib.trading.strategies.rb.open.sessions import ORBSession

# ---------------------------------------------------------------------------
# Per-session asset focus lists
# ---------------------------------------------------------------------------
# Maps session key → list of Yahoo / internal tickers relevant for that
# session window.  The ORB check loop filters assets to this list, avoiding
# e.g. scanning MES during Tokyo (near-zero volume) or 6E during US Equity
# Open (FX already moved 5 hours earlier).
#
# Ticker values must match the ASSETS dict in lib/core/models.py.
#
# Extended symbol set (all micro CME contracts + FX futures):
#   MGC=F  Micro Gold          MES=F  Micro S&P 500
#   MCL=F  Micro Crude Oil     MNQ=F  Micro Nasdaq-100
#   MHG=F  Micro Copper        M2K=F  Micro Russell 2000
#   SIL=F  Micro Silver        MYM=F  Micro Dow Jones
#   MBT=F  Micro Bitcoin       6E=F   Euro FX
#   6B=F   British Pound       6J=F   Japanese Yen
#   6A=F   Australian Dollar   6C=F   Canadian Dollar
# ---------------------------------------------------------------------------
SESSION_ASSETS: dict[str, list[str]] = {
    # ------------------------------------------------------------------
    # 1. CME Globex re-open (18:00 ET) — all CME micros.
    #    FX included; overnight gaps in 6E/6J drive early direction.
    # ------------------------------------------------------------------
    "cme": [
        "MGC=F",
        "MCL=F",
        "MHG=F",
        "SIL=F",
        "MES=F",
        "MNQ=F",
        "M2K=F",
        "MYM=F",
        "6E=F",
        "6B=F",
        "6J=F",
        "MBT=F",
    ],
    # ------------------------------------------------------------------
    # 2. Sydney / ASX (18:30 ET) — thin overnight; metals, energy,
    #    AUD-correlated FX, MBT.
    # ------------------------------------------------------------------
    "sydney": [
        "MGC=F",
        "MCL=F",
        "SIL=F",
        "6A=F",
        "6J=F",
        "MBT=F",
    ],
    # ------------------------------------------------------------------
    # 3. Tokyo / TSE (19:00 ET) — metals, JPY/AUD FX, thin index futures.
    # ------------------------------------------------------------------
    "tokyo": [
        "MGC=F",
        "MCL=F",
        "MHG=F",
        "SIL=F",
        "6J=F",
        "6A=F",
    ],
    # ------------------------------------------------------------------
    # 4. Shanghai / HK (21:00 ET) — copper and gold dominant;
    #    CNH-proxy via 6J.
    # ------------------------------------------------------------------
    "shanghai": [
        "MGC=F",
        "MHG=F",
        "MCL=F",
        "SIL=F",
        "6J=F",
    ],
    # ------------------------------------------------------------------
    # 5. Frankfurt / Xetra (03:00 ET) — EUR FX, DAX-correlated index
    #    futures, metals.
    # ------------------------------------------------------------------
    "frankfurt": [
        "MGC=F",
        "MCL=F",
        "MES=F",
        "MNQ=F",
        "MYM=F",
        "6E=F",
        "6B=F",
    ],
    # ------------------------------------------------------------------
    # 6. London Open (03:00 ET) — primary session; all major CME
    #    contracts + FX.
    # ------------------------------------------------------------------
    "london": [
        "MGC=F",
        "MCL=F",
        "MHG=F",
        "SIL=F",
        "MES=F",
        "MNQ=F",
        "M2K=F",
        "MYM=F",
        "6E=F",
        "6B=F",
        "6J=F",
    ],
    # ------------------------------------------------------------------
    # 7. London-NY Crossover (08:00 ET) — highest conviction; full
    #    universe including MBT.
    # ------------------------------------------------------------------
    "london_ny": [
        "MGC=F",
        "MCL=F",
        "MHG=F",
        "SIL=F",
        "MES=F",
        "MNQ=F",
        "M2K=F",
        "MYM=F",
        "6E=F",
        "6B=F",
        "6J=F",
        "MBT=F",
    ],
    # ------------------------------------------------------------------
    # 8. US Equity Open (09:30 ET) — index futures primary; gold
    #    correlation window.
    # ------------------------------------------------------------------
    "us": [
        "MGC=F",
        "MCL=F",
        "MES=F",
        "MNQ=F",
        "M2K=F",
        "MYM=F",
    ],
    # ------------------------------------------------------------------
    # 9. CME Settlement (14:00 ET) — metals and energy resolution
    #    before close.
    # ------------------------------------------------------------------
    "cme_settle": [
        "MGC=F",
        "MCL=F",
        "MHG=F",
        "SIL=F",
    ],
}


# ---------------------------------------------------------------------------
# Per-symbol parameter overrides for crypto ORB detection
# ---------------------------------------------------------------------------
# Crypto assets have much larger ATR values (BTC ~$1 000+/day) and wider
# intraday ranges than CME micro futures.  These overrides are applied
# inside detect_opening_range_breakout() when the caller passes a crypto
# symbol.
#
# Structure:
#   {ticker: {
#       "breakout_multiplier": float,
#       "max_or_atr_ratio":    float,
#       "min_or_atr_ratio":    float,
#       "min_depth_atr_pct":   float,
#       "min_body_ratio":      float,
#   }}
# ---------------------------------------------------------------------------
CRYPTO_SYMBOL_OVERRIDES: dict[str, dict[str, float]] = {
    # Bitcoin — highest volatility; BTC ATR is typically 1–3 % of price.
    "KRAKEN:XBT/USD": {
        "breakout_multiplier": 0.70,
        "max_or_atr_ratio": 3.0,
        "min_or_atr_ratio": 0.03,
        "min_depth_atr_pct": 0.08,
        "min_body_ratio": 0.40,
    },
    # Ethereum — slightly tighter than BTC
    "KRAKEN:ETH/USD": {
        "breakout_multiplier": 0.65,
        "max_or_atr_ratio": 2.8,
        "min_or_atr_ratio": 0.03,
        "min_depth_atr_pct": 0.09,
        "min_body_ratio": 0.42,
    },
    # Solana — highly volatile; wider range acceptable
    "KRAKEN:SOL/USD": {
        "breakout_multiplier": 0.68,
        "max_or_atr_ratio": 3.0,
        "min_or_atr_ratio": 0.03,
        "min_depth_atr_pct": 0.08,
        "min_body_ratio": 0.40,
    },
    # Mid-cap alts — moderate volatility
    "KRAKEN:LINK/USD": {
        "breakout_multiplier": 0.65,
        "max_or_atr_ratio": 2.8,
        "min_or_atr_ratio": 0.03,
        "min_depth_atr_pct": 0.09,
        "min_body_ratio": 0.42,
    },
    "KRAKEN:AVAX/USD": {
        "breakout_multiplier": 0.65,
        "max_or_atr_ratio": 2.8,
        "min_or_atr_ratio": 0.03,
        "min_depth_atr_pct": 0.09,
        "min_body_ratio": 0.42,
    },
    "KRAKEN:DOT/USD": {
        "breakout_multiplier": 0.65,
        "max_or_atr_ratio": 2.8,
        "min_or_atr_ratio": 0.03,
        "min_depth_atr_pct": 0.09,
        "min_body_ratio": 0.42,
    },
    "KRAKEN:ADA/USD": {
        "breakout_multiplier": 0.60,
        "max_or_atr_ratio": 2.5,
        "min_or_atr_ratio": 0.03,
        "min_depth_atr_pct": 0.10,
        "min_body_ratio": 0.43,
    },
    "KRAKEN:POL/USD": {
        "breakout_multiplier": 0.62,
        "max_or_atr_ratio": 2.6,
        "min_or_atr_ratio": 0.03,
        "min_depth_atr_pct": 0.10,
        "min_body_ratio": 0.43,
    },
    "KRAKEN:XRP/USD": {
        "breakout_multiplier": 0.60,
        "max_or_atr_ratio": 2.5,
        "min_or_atr_ratio": 0.03,
        "min_depth_atr_pct": 0.10,
        "min_body_ratio": 0.43,
    },
}

# Conservative fallback overrides for any KRAKEN: ticker not listed above
_CRYPTO_FALLBACK_OVERRIDES: dict[str, float] = {
    "breakout_multiplier": 0.65,
    "max_or_atr_ratio": 2.8,
    "min_or_atr_ratio": 0.03,
    "min_depth_atr_pct": 0.09,
    "min_body_ratio": 0.42,
}


# ---------------------------------------------------------------------------
# Runtime crypto-ticker injection
# ---------------------------------------------------------------------------
# Crypto markets are 24/7, so Kraken tickers are relevant in every session.
# We append them to each existing session list and also populate the
# dedicated crypto session lists (crypto_utc0, crypto_utc12).
#
# Gated by ENABLE_KRAKEN_CRYPTO (same env flag as lib/core/models.py).
# Populated once at import time so subsequent callers see the merged lists.
#
# A module-level guard (_CRYPTO_INJECTED) ensures the injection is
# idempotent — reloading this module (e.g. during hot-reload or testing)
# will not append duplicate tickers or register sessions twice.
# ---------------------------------------------------------------------------

_KRAKEN_CRYPTO_TICKERS: list[str] = []
_CRYPTO_INJECTED: bool = False

try:
    from lib.core.models import ENABLE_KRAKEN_CRYPTO as _ENABLE_KRAKEN_CRYPTO

    if _ENABLE_KRAKEN_CRYPTO:
        from lib.integrations.kraken_client import KRAKEN_PAIRS as _KRAKEN_PAIRS

        _KRAKEN_CRYPTO_TICKERS.extend(p["internal_ticker"] for p in _KRAKEN_PAIRS.values())
except ImportError:
    pass

if _KRAKEN_CRYPTO_TICKERS and not _CRYPTO_INJECTED:
    # Append crypto tickers to every existing CME session list
    for _sk in list(SESSION_ASSETS.keys()):
        SESSION_ASSETS[_sk] = SESSION_ASSETS[_sk] + _KRAKEN_CRYPTO_TICKERS

    # Populate dedicated crypto-session asset lists
    SESSION_ASSETS["crypto_utc0"] = list(_KRAKEN_CRYPTO_TICKERS)
    SESSION_ASSETS["crypto_utc12"] = list(_KRAKEN_CRYPTO_TICKERS)

    # Register crypto sessions in the master ORB_SESSIONS list and
    # SESSION_BY_KEY lookup so the scheduler picks them up automatically.
    # Import here (after sessions module is fully initialised) to avoid
    # circular imports at module load time.
    from lib.trading.strategies.rb.open.sessions import (
        CRYPTO_UTC_MIDNIGHT_SESSION,
        CRYPTO_UTC_NOON_SESSION,
        ORB_SESSIONS,
        SESSION_BY_KEY,
        _rebuild_session_groups,
    )

    if CRYPTO_UTC_MIDNIGHT_SESSION not in ORB_SESSIONS:
        ORB_SESSIONS.append(CRYPTO_UTC_MIDNIGHT_SESSION)
        SESSION_BY_KEY[CRYPTO_UTC_MIDNIGHT_SESSION.key] = CRYPTO_UTC_MIDNIGHT_SESSION

    if CRYPTO_UTC_NOON_SESSION not in ORB_SESSIONS:
        ORB_SESSIONS.append(CRYPTO_UTC_NOON_SESSION)
        SESSION_BY_KEY[CRYPTO_UTC_NOON_SESSION.key] = CRYPTO_UTC_NOON_SESSION

    # Rebuild OVERNIGHT_SESSIONS / DAYTIME_SESSIONS / DATASET_SESSIONS
    # in-place so they include the newly registered crypto sessions.
    _rebuild_session_groups()

    _CRYPTO_INJECTED = True


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def get_symbol_session_overrides(symbol: str, session: ORBSession) -> dict[str, float]:
    """Return per-symbol parameter overrides for *symbol* in *session*.

    For known Kraken crypto tickers the returned dict may contain any of
    ``breakout_multiplier``, ``max_or_atr_ratio``, ``min_or_atr_ratio``,
    ``min_depth_atr_pct``, and ``min_body_ratio`` keys which should
    override the session defaults.

    For unknown ``KRAKEN:`` prefixed tickers a conservative fallback dict
    is returned.  For non-crypto symbols an empty dict is returned so the
    caller uses session defaults unchanged.

    Args:
        symbol:  Instrument ticker (e.g. ``"KRAKEN:XBT/USD"``, ``"MGC=F"``).
        session: The :class:`~sessions.ORBSession` being evaluated (currently
                 unused but kept for future per-session crypto adjustments).

    Returns:
        Dict of threshold overrides (may be empty).
    """
    # Exact match first
    if symbol in CRYPTO_SYMBOL_OVERRIDES:
        return dict(CRYPTO_SYMBOL_OVERRIDES[symbol])

    # Prefix match — unknown crypto asset; use conservative fallback
    if symbol.startswith("KRAKEN:"):
        return dict(_CRYPTO_FALLBACK_OVERRIDES)

    return {}


def get_session_assets(session: ORBSession) -> list[str]:
    """Return the list of tickers relevant for *session*.

    Falls back to all configured ASSETS from ``lib.core.models`` if the
    session key has no specific list registered in :data:`SESSION_ASSETS`.

    Args:
        session: The :class:`~sessions.ORBSession` to look up.

    Returns:
        List of ticker strings (e.g. ``["MGC=F", "MES=F"]``).
    """
    if session.key in SESSION_ASSETS:
        return SESSION_ASSETS[session.key]

    # Fallback: return all asset tickers from the models config
    try:
        from lib.core.models import ASSETS

        return list(ASSETS.values())
    except Exception:
        return []
