"""
Data Source Router — Routes market data requests to the active data source
===========================================================================
Centralizes the logic for determining which data source (Kraken, Rithmic,
or both) should serve data for a given request. Used by DOM, charts,
signals, and the simulation engine.

The active source is stored in Redis at ``settings:data_source`` and can
be changed via the settings API.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

from fastapi import APIRouter

logger = logging.getLogger("source_router")

__all__ = [
    "get_active_source",
    "is_crypto_symbol",
    "is_futures_symbol",
    "should_use_source",
    "get_available_symbols",
    "source_api_router",
]

# ---------------------------------------------------------------------------
# Redis key for the active data source setting
# ---------------------------------------------------------------------------
_REDIS_DATA_SOURCE_KEY = "settings:data_source"

_VALID_SOURCES = ("kraken", "rithmic", "both")

# ---------------------------------------------------------------------------
# Futures symbol map — the 9 focus futures contracts
# ---------------------------------------------------------------------------
FUTURES_SYMBOLS: dict[str, dict[str, str]] = {
    "MGC": {"name": "Micro Gold", "data_ticker": "MGC=F"},
    "SIL": {"name": "Micro Silver", "data_ticker": "SIL=F"},
    "MES": {"name": "Micro E-mini S&P 500", "data_ticker": "MES=F"},
    "MNQ": {"name": "Micro E-mini Nasdaq", "data_ticker": "MNQ=F"},
    "M2K": {"name": "Micro E-mini Russell", "data_ticker": "M2K=F"},
    "MYM": {"name": "Micro E-mini Dow", "data_ticker": "MYM=F"},
    "ZN": {"name": "10-Year T-Note", "data_ticker": "ZN=F"},
    "ZB": {"name": "30-Year T-Bond", "data_ticker": "ZB=F"},
    "ZW": {"name": "Wheat", "data_ticker": "ZW=F"},
}

# Shorthand crypto names that should route to Kraken
_CRYPTO_SHORTHANDS = frozenset(
    {
        "BTC",
        "ETH",
        "SOL",
        "LINK",
        "AVAX",
        "DOT",
        "ADA",
        "POL",
        "XRP",
    }
)


# ---------------------------------------------------------------------------
# Redis helpers (same pattern as the rest of the project)
# ---------------------------------------------------------------------------


def _redis_client():
    """Return ``(client, available)`` — ``(None, False)`` if unavailable."""
    try:
        from lib.core.cache import REDIS_AVAILABLE, _r  # type: ignore[import-unresolved]

        return _r, REDIS_AVAILABLE
    except ImportError:
        return None, False


# ---------------------------------------------------------------------------
# Core routing functions
# ---------------------------------------------------------------------------


def get_active_source() -> str:
    """Read the active data source from Redis.

    Returns one of ``"kraken"``, ``"rithmic"``, or ``"both"``.
    Defaults to ``"kraken"`` if the key is not set or Redis is unavailable.
    """
    r, available = _redis_client()
    if not available or r is None:
        return "kraken"
    try:
        raw = r.get(_REDIS_DATA_SOURCE_KEY)
        if raw is None:
            return "kraken"
        value = raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)
        return value if value in _VALID_SOURCES else "kraken"
    except Exception:
        return "kraken"


def is_crypto_symbol(symbol: str) -> bool:
    """Return ``True`` if *symbol* looks like a crypto / Kraken symbol.

    Matches:
    - Symbols starting with ``"KRAKEN:"``
    - Shorthand names like ``"BTC"``, ``"ETH"``, ``"SOL"``, etc.
    """
    if symbol.upper().startswith("KRAKEN:"):
        return True
    if symbol.upper() in _CRYPTO_SHORTHANDS:
        return True
    return False


def is_futures_symbol(symbol: str) -> bool:
    """Return ``True`` if *symbol* is a recognised CME futures symbol.

    Checks the symbol (case-insensitive) against the ``FUTURES_SYMBOLS`` map
    and common ticker suffixes like ``"MES=F"``.
    """
    sym = symbol.upper()
    if sym in FUTURES_SYMBOLS:
        return True
    # Also check if the symbol matches a data_ticker (e.g. "MES=F")
    for info in FUTURES_SYMBOLS.values():
        if sym == info["data_ticker"].upper():
            return True
    return False


def should_use_source(symbol: str) -> str:
    """Determine which data source to use for *symbol*.

    Logic:
    - If active source is ``"both"``, route to the appropriate source based
      on symbol type (crypto → ``"kraken"``, futures → ``"rithmic"``).
    - If active source is ``"kraken"`` but the symbol is a futures symbol,
      return ``"mock"`` (simulated data).
    - If active source is ``"rithmic"`` but the symbol is a crypto symbol,
      return ``"mock"``.
    - Otherwise, return the active source.

    Returns:
        One of ``"kraken"``, ``"rithmic"``, or ``"mock"``.
    """
    active = get_active_source()
    crypto = is_crypto_symbol(symbol)
    futures = is_futures_symbol(symbol)

    if active == "both":
        if crypto:
            return "kraken"
        if futures:
            return "rithmic"
        # Unknown symbol — default to mock
        return "mock"

    if active == "kraken":
        if futures:
            return "mock"
        return "kraken"

    if active == "rithmic":
        if crypto:
            return "mock"
        return "rithmic"

    return "mock"


def _get_kraken_status() -> str:
    """Check Kraken feed connectivity status."""
    try:
        from lib.integrations.kraken_client import get_kraken_feed  # type: ignore[import-unresolved]

        feed = get_kraken_feed()
        if feed is not None:
            if getattr(feed, "is_connected", False):
                return "connected"
            if getattr(feed, "is_running", False):
                return "connecting"
            return "disconnected"
        return "disconnected"
    except Exception:
        return "disconnected"


def _get_rithmic_status() -> str:
    """Check Rithmic connectivity status."""
    try:
        from lib.integrations.rithmic_client import get_manager  # type: ignore[import-unresolved]

        mgr = get_manager()
        all_status = mgr.get_all_status()
        if not all_status:
            return "not_configured"
        # If any account is connected, report connected
        for _key, st in all_status.items():
            if isinstance(st, dict) and st.get("connected"):
                return "connected"
        return "disconnected"
    except Exception:
        return "not_configured"


def get_available_symbols() -> dict[str, Any]:
    """Return categorised symbol lists with source and liveness info.

    Returns a dict with ``"crypto"``, ``"futures"``, and ``"active_source"`` keys.
    """
    active = get_active_source()
    kraken_status = _get_kraken_status()
    rithmic_status = _get_rithmic_status()

    # --- Crypto symbols from Kraken ---
    crypto_symbols: list[dict[str, Any]] = []
    try:
        from lib.integrations.kraken_client import KRAKEN_PAIRS  # type: ignore[import-unresolved]

        kraken_live = kraken_status == "connected" and active in ("kraken", "both")
        for name, info in KRAKEN_PAIRS.items():
            crypto_symbols.append(
                {
                    "symbol": info["internal_ticker"],
                    "name": name,
                    "source": "kraken",
                    "base": info.get("base", ""),
                    "live": kraken_live,
                }
            )
    except Exception:
        # Kraken client not available — provide minimal list
        for shorthand in sorted(_CRYPTO_SHORTHANDS):
            crypto_symbols.append(
                {
                    "symbol": f"KRAKEN:{shorthand}USD",
                    "name": shorthand,
                    "source": "kraken",
                    "base": shorthand,
                    "live": False,
                }
            )

    # --- Futures symbols ---
    rithmic_live = rithmic_status == "connected" and active in ("rithmic", "both")
    futures_symbols: list[dict[str, Any]] = []
    for sym, info in FUTURES_SYMBOLS.items():
        futures_symbols.append(
            {
                "symbol": sym,
                "name": info["name"],
                "data_ticker": info["data_ticker"],
                "source": "rithmic",
                "live": rithmic_live,
            }
        )

    return {
        "crypto": crypto_symbols,
        "futures": futures_symbols,
        "active_source": active,
    }


# ---------------------------------------------------------------------------
# FastAPI router — /api/sources/*
# ---------------------------------------------------------------------------

source_api_router = APIRouter(prefix="/api/sources", tags=["Data Sources"])


@source_api_router.get("/symbols")
def api_available_symbols() -> dict:
    """Return all available symbols grouped by asset class.

    Includes crypto (Kraken) and futures (Rithmic) symbols with their
    current liveness status based on the active data source configuration.
    """
    return get_available_symbols()


@source_api_router.get("/status")
def api_source_status() -> dict:
    """Return status of all data sources.

    Includes connectivity state for Kraken and Rithmic, the currently
    active source, and whether simulation mode is enabled.
    """
    active = get_active_source()
    kraken_status = _get_kraken_status()
    rithmic_status = _get_rithmic_status()
    sim_enabled = os.getenv("SIM_ENABLED", "0") == "1"

    return {
        "active_source": active,
        "kraken_status": kraken_status,
        "rithmic_status": rithmic_status,
        "available_sources": list(_VALID_SOURCES),
        "sim_enabled": sim_enabled,
        "sim_data_source": active if active != "both" else "kraken",
        "timestamp": time.time(),
    }
