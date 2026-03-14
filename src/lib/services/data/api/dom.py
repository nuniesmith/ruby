"""
DOM (Depth of Market) API Routes
=================================
Provides REST and SSE endpoints for real-time depth-of-market data,
powering the DOM ladder widget on the dashboard.

Endpoints:
    GET /api/dom/snapshot?symbol=MES  — Current DOM state (live or mock)
    GET /api/dom/config               — DOM display configuration
    GET /sse/dom?symbol=MES           — SSE stream of DOM updates

The snapshot and SSE endpoints return Level-2 order-book data structured
as bid/ask price ladders with size at each level.  When live Kraken data
is available via Redis, the DOM uses real prices from the WebSocket feed.
For futures symbols (or when no live data is cached), it falls back to
mock data.

Usage:
    from lib.services.data.api.dom import router, sse_router

    app.include_router(router)
    app.include_router(sse_router)
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, Query, Request
from fastapi.responses import StreamingResponse

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

from lib.core.logging_config import get_logger
from lib.services.data.source_router import get_active_source, should_use_source

logger = get_logger(__name__)

__all__ = [
    "router",
    "sse_router",
]

# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------

router = APIRouter(prefix="/api/dom", tags=["dom"])
sse_router = APIRouter(tags=["dom-sse"])


# ---------------------------------------------------------------------------
# Constants / defaults
# ---------------------------------------------------------------------------

_DEFAULT_SYMBOL = "MES"
_DEFAULT_LEVELS = 10
_TICK_SIZES: dict[str, float] = {
    "MES": 0.25,
    "MNQ": 0.25,
    "MGC": 0.10,
    "MCL": 0.01,
    "ES": 0.25,
    "NQ": 0.25,
    "GC": 0.10,
    "CL": 0.01,
}
_MOCK_BASE_PRICES: dict[str, float] = {
    "MES": 5420.00,
    "MNQ": 18950.00,
    "MGC": 2345.00,
    "MCL": 78.50,
    "ES": 5420.00,
    "NQ": 18950.00,
    "GC": 2345.00,
    "CL": 78.50,
}

_SSE_UPDATE_INTERVAL_S = 1.0
_SSE_HEARTBEAT_INTERVAL_S = 15.0

# ---------------------------------------------------------------------------
# Crypto DOM symbol mapping
# ---------------------------------------------------------------------------
# Maps shorthand crypto symbols and full internal tickers to the canonical
# Kraken internal ticker format used in Redis cache keys and sim positions.

_CRYPTO_DOM_SYMBOLS: dict[str, str] = {
    "BTC": "KRAKEN:XBTUSD",
    "ETH": "KRAKEN:ETHUSD",
    "SOL": "KRAKEN:SOLUSD",
    "KRAKEN:XBTUSD": "KRAKEN:XBTUSD",
    "KRAKEN:ETHUSD": "KRAKEN:ETHUSD",
    "KRAKEN:SOLUSD": "KRAKEN:SOLUSD",
}

# Tick sizes for crypto symbols (keyed by internal ticker)
_CRYPTO_TICK_SIZES: dict[str, float] = {
    "KRAKEN:XBTUSD": 0.1,
    "KRAKEN:ETHUSD": 0.01,
    "KRAKEN:SOLUSD": 0.001,
}


# ---------------------------------------------------------------------------
# Redis helpers
# ---------------------------------------------------------------------------


def _redis_client():
    """Return ``(client, available)`` — ``(None, False)`` if unavailable."""
    try:
        from lib.core.cache import REDIS_AVAILABLE, _r

        return _r, REDIS_AVAILABLE
    except ImportError:
        return None, False


def _get_live_bar(internal_ticker: str) -> dict[str, Any] | None:
    """Fetch the latest cached bar from Redis for *internal_ticker*.

    The KrakenFeedManager publishes bars to ``kraken:live:{internal_ticker}``
    with a 120-second TTL.  Returns the parsed dict or ``None``.
    """
    r, available = _redis_client()
    if not available or r is None:
        return None
    try:
        cache_key = f"kraken:live:{internal_ticker}"
        raw = r.get(cache_key)
        if raw is None:
            return None
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", errors="replace")
        return json.loads(raw)
    except Exception:
        return None


def _get_sim_positions() -> list[dict[str, Any]]:
    """Fetch open simulation positions from Redis.

    Returns a list of position dicts (may be empty).
    """
    r, available = _redis_client()
    if not available or r is None:
        return []
    try:
        raw = r.get("sim:positions")
        if raw is None:
            return []
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", errors="replace")
        data = json.loads(raw)
        return data if isinstance(data, list) else []
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Snapshot builders
# ---------------------------------------------------------------------------


def _build_mock_snapshot(
    symbol: str,
    levels: int = _DEFAULT_LEVELS,
) -> dict[str, Any]:
    """Build a mock DOM snapshot for *symbol*.

    Returns a dict with ``bids``, ``asks``, ``last``, ``spread``, and
    metadata fields ready for JSON serialisation.
    """
    tick = _TICK_SIZES.get(symbol, 0.25)
    base = _MOCK_BASE_PRICES.get(symbol, 5000.00)

    best_bid = base - tick
    best_ask = base

    bids: list[dict[str, Any]] = []
    asks: list[dict[str, Any]] = []

    for i in range(levels):
        bids.append(
            {
                "price": round(best_bid - i * tick, 4),
                "size": max(5, 60 - i * 4),
                "cumulative_size": 0,
            }
        )
        asks.append(
            {
                "price": round(best_ask + i * tick, 4),
                "size": max(5, 55 - i * 4),
                "cumulative_size": 0,
            }
        )

    # Compute cumulative sizes
    running = 0
    for lvl in bids:
        running += lvl["size"]
        lvl["cumulative_size"] = running
    running = 0
    for lvl in asks:
        running += lvl["size"]
        lvl["cumulative_size"] = running

    bid_total = sum(b["size"] for b in bids)
    ask_total = sum(a["size"] for a in asks)

    return {
        "symbol": symbol,
        "bids": bids,
        "asks": asks,
        "last": round(best_bid + tick / 2, 4),
        "spread": round(best_ask - best_bid, 4),
        "bid_total": bid_total,
        "ask_total": ask_total,
        "imbalance_ratio": round(bid_total / max(ask_total, 1), 3),
        "levels": levels,
        "timestamp": time.time(),
        "source": "mock",
    }


def _build_live_snapshot(
    symbol: str,
    internal_ticker: str,
    levels: int = _DEFAULT_LEVELS,
) -> dict[str, Any] | None:
    """Build a DOM snapshot from live Kraken data cached in Redis.

    Reads the latest bar from ``kraken:live:{internal_ticker}`` and
    constructs a synthetic order book around the close price.  Also
    checks ``sim:positions`` to annotate price levels where the user
    has an open simulation position.

    Returns ``None`` if no live data is available (caller should fall
    back to mock).

    Args:
        symbol: The original symbol the client requested.
        internal_ticker: The canonical Kraken internal ticker
            (e.g. ``"KRAKEN:XBTUSD"``).
        levels: Number of price levels per side.
    """
    bar = _get_live_bar(internal_ticker)
    if bar is None:
        return None

    close_price = float(bar.get("close", 0))
    if close_price <= 0:
        return None

    tick = _CRYPTO_TICK_SIZES.get(internal_ticker, 0.01)

    # Use the close price as the mid-point for the synthetic book
    best_bid = round(close_price - tick, 10)
    best_ask = round(close_price, 10)

    # Fetch sim positions to annotate levels
    sim_positions = _get_sim_positions()
    position_prices: dict[float, dict[str, Any]] = {}
    for pos in sim_positions:
        if pos.get("symbol") == internal_ticker:
            entry_price = float(pos.get("entry_price", 0))
            if entry_price > 0:
                position_prices[round(entry_price, 10)] = {
                    "side": pos.get("side", ""),
                    "qty": pos.get("qty", 0),
                    "unrealized_pnl": pos.get("unrealized_pnl", 0),
                }
            sl = pos.get("stop_loss")
            if sl is not None and float(sl) > 0:
                position_prices[round(float(sl), 10)] = {"marker": "stop_loss"}
            tp = pos.get("take_profit")
            if tp is not None and float(tp) > 0:
                position_prices[round(float(tp), 10)] = {"marker": "take_profit"}

    # Synthetic sizes using bar volume as a rough guide
    bar_volume = float(bar.get("volume", 100))
    base_size = max(5, int(bar_volume / max(levels * 2, 1)))

    bids: list[dict[str, Any]] = []
    asks: list[dict[str, Any]] = []

    for i in range(levels):
        bid_price = round(best_bid - i * tick, 10)
        ask_price = round(best_ask + i * tick, 10)

        # Size tapers away from the spread
        bid_size = max(1, base_size - i * max(1, base_size // (levels + 1)))
        ask_size = max(1, base_size - i * max(1, base_size // (levels + 1)))

        bid_entry: dict[str, Any] = {
            "price": bid_price,
            "size": bid_size,
            "cumulative_size": 0,
        }
        ask_entry: dict[str, Any] = {
            "price": ask_price,
            "size": ask_size,
            "cumulative_size": 0,
        }

        # Annotate position markers
        if bid_price in position_prices:
            bid_entry["position"] = position_prices[bid_price]
        if ask_price in position_prices:
            ask_entry["position"] = position_prices[ask_price]

        bids.append(bid_entry)
        asks.append(ask_entry)

    # Cumulative sizes
    running = 0
    for lvl in bids:
        running += lvl["size"]
        lvl["cumulative_size"] = running
    running = 0
    for lvl in asks:
        running += lvl["size"]
        lvl["cumulative_size"] = running

    bid_total = sum(b["size"] for b in bids)
    ask_total = sum(a["size"] for a in asks)

    return {
        "symbol": symbol,
        "internal_ticker": internal_ticker,
        "bids": bids,
        "asks": asks,
        "last": close_price,
        "spread": round(best_ask - best_bid, 10),
        "bid_total": bid_total,
        "ask_total": ask_total,
        "imbalance_ratio": round(bid_total / max(ask_total, 1), 3),
        "levels": levels,
        "bar": {
            "open": bar.get("open"),
            "high": bar.get("high"),
            "low": bar.get("low"),
            "close": bar.get("close"),
            "volume": bar.get("volume"),
            "vwap": bar.get("vwap"),
        },
        "timestamp": time.time(),
        "source": "kraken_live",
    }


def _build_snapshot(
    symbol: str,
    levels: int = _DEFAULT_LEVELS,
) -> dict[str, Any]:
    """Build a DOM snapshot, preferring live data from the active source.

    Uses the source router to decide which data source should serve data
    for *symbol*.  If the router says ``"mock"``, skips live lookups
    entirely.  When the active source is ``"both"``, tries live data
    for the appropriate source and falls back to mock.

    Args:
        symbol: The requested symbol (e.g. ``"BTC"``, ``"KRAKEN:XBTUSD"``,
            ``"MES"``).
        levels: Number of price levels per side.
    """
    source = should_use_source(symbol)

    # Source router says mock — go straight to mock data
    if source == "mock":
        logger.debug("source router says mock for symbol", symbol=symbol, active_source=get_active_source())
        return _build_mock_snapshot(symbol, levels=levels)

    # Try live Kraken data for crypto symbols
    internal_ticker = _CRYPTO_DOM_SYMBOLS.get(symbol)

    if internal_ticker is not None and source in ("kraken", "rithmic"):
        live = _build_live_snapshot(symbol, internal_ticker, levels=levels)
        if live is not None:
            return live
        # No live data cached — fall back to mock, but log it
        logger.debug("no live data for crypto symbol, falling back to mock", symbol=symbol)

    return _build_mock_snapshot(symbol, levels=levels)


def _build_mock_config() -> dict[str, Any]:
    """Return DOM display configuration for the dashboard widget."""
    # Merge futures and crypto symbols for the available_symbols list
    all_symbols = sorted(set(_TICK_SIZES.keys()) | set(_CRYPTO_DOM_SYMBOLS.keys()))
    all_tick_sizes = dict(_TICK_SIZES)
    for shorthand, internal in _CRYPTO_DOM_SYMBOLS.items():
        all_tick_sizes[shorthand] = _CRYPTO_TICK_SIZES.get(internal, 0.01)

    return {
        "default_symbol": _DEFAULT_SYMBOL,
        "default_levels": _DEFAULT_LEVELS,
        "available_symbols": all_symbols,
        "tick_sizes": all_tick_sizes,
        "crypto_symbols": _CRYPTO_DOM_SYMBOLS,
        "color_scheme": {
            "bid_fill": "#0d6efd33",
            "ask_fill": "#dc354533",
            "bid_text": "#0d6efd",
            "ask_text": "#dc3545",
            "last_highlight": "#ffc107",
            "spread_bar": "#6c757d",
        },
        "update_interval_ms": int(_SSE_UPDATE_INTERVAL_S * 1000),
        "max_levels": 20,
        "show_cumulative": True,
        "show_imbalance_bar": True,
    }


# ---------------------------------------------------------------------------
# REST routes
# ---------------------------------------------------------------------------


@router.get("/snapshot")
async def dom_snapshot(
    symbol: str = Query(default=_DEFAULT_SYMBOL, description="Instrument symbol"),
    levels: int = Query(default=_DEFAULT_LEVELS, ge=1, le=50, description="Price levels per side"),
) -> dict[str, Any]:
    """Return a point-in-time DOM snapshot for *symbol*.

    Returns Level-2 bid/ask ladders with size, cumulative size,
    spread, and imbalance ratio.  Uses live Kraken data for crypto
    symbols when available; falls back to mock data otherwise.
    """
    logger.debug("dom_snapshot", symbol=symbol, levels=levels)
    return _build_snapshot(symbol, levels=levels)


@router.get("/config")
async def dom_config() -> dict[str, Any]:
    """Return DOM display configuration for the dashboard widget.

    Includes colour scheme, available symbols, tick sizes, and
    default rendering options.
    """
    logger.debug("dom_config")
    return _build_mock_config()


# ---------------------------------------------------------------------------
# SSE stream
# ---------------------------------------------------------------------------


async def _dom_event_generator(
    symbol: str,
    levels: int,
    request: Request,
) -> AsyncGenerator[str]:
    """Yield SSE-formatted DOM updates at a regular interval.

    For crypto symbols, reads live bar data from Redis
    (``kraken:live:{internal_ticker}``) and builds snapshots from real
    prices.  For futures symbols (or when no live data is available),
    falls back to mock snapshots.

    A heartbeat comment (``:``) is sent periodically to keep the
    connection alive even if no data changes.
    """
    internal_ticker = _CRYPTO_DOM_SYMBOLS.get(symbol)
    source_label = "live" if internal_ticker else "mock"
    logger.info("dom SSE stream started", symbol=symbol, levels=levels, source=source_label)
    last_heartbeat = time.monotonic()

    try:
        while True:
            # Check if client disconnected
            if await request.is_disconnected():
                logger.info("dom SSE client disconnected", symbol=symbol)
                break

            # Try live Kraken data for crypto, fall back to mock
            snapshot = _build_snapshot(symbol, levels=levels)

            payload = json.dumps(snapshot, default=str)
            yield f"event: dom-update\ndata: {payload}\n\n"

            # Heartbeat (SSE comment) to keep proxies / load-balancers happy
            now = time.monotonic()
            if now - last_heartbeat >= _SSE_HEARTBEAT_INTERVAL_S:
                yield ": heartbeat\n\n"
                last_heartbeat = now

            await asyncio.sleep(_SSE_UPDATE_INTERVAL_S)

    except asyncio.CancelledError:
        logger.info("dom SSE stream cancelled", symbol=symbol)
    except Exception:
        logger.exception("dom SSE stream error", symbol=symbol)
    finally:
        logger.info("dom SSE stream closed", symbol=symbol)


@sse_router.get("/sse/dom")
async def sse_dom(
    request: Request,
    symbol: str = Query(default=_DEFAULT_SYMBOL, description="Instrument symbol"),
    levels: int = Query(default=_DEFAULT_LEVELS, ge=1, le=50, description="Price levels per side"),
) -> StreamingResponse:
    """SSE endpoint streaming real-time DOM updates.

    Connect via ``EventSource("/sse/dom?symbol=MES")`` in the browser.
    Events are named ``dom-update`` and contain a JSON DOM snapshot.

    For crypto symbols (``BTC``, ``ETH``, ``SOL``, or full
    ``KRAKEN:*`` tickers), the stream uses live prices from the Kraken
    WebSocket feed cached in Redis.  For futures symbols the stream
    uses mock data until Rithmic credentials are wired in.

    The stream sends an update roughly every second (configurable via
    ``dom_config.update_interval_ms``).
    """
    logger.info("sse_dom connection opened", symbol=symbol, levels=levels)

    return StreamingResponse(
        _dom_event_generator(symbol, levels, request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # disable nginx buffering
        },
    )
