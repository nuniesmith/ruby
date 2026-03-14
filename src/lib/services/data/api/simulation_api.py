"""
Simulation API Router — Paper Trading Control
===============================================
Provides REST and SSE endpoints for controlling the KRAKEN-SIM paper
trading simulation engine from the dashboard.

REST endpoints (prefix ``/api/sim``):
    GET  /api/sim/status      — Current sim state (positions, P&L, orders)
    POST /api/sim/order       — Submit a market or limit order
    POST /api/sim/close/{symbol} — Close position for a specific symbol
    POST /api/sim/close-all   — Close all open positions
    POST /api/sim/reset       — Reset the simulation (balance, history)
    GET  /api/sim/trades      — List completed sim trades
    GET  /api/sim/pnl         — P&L summary

SSE endpoint:
    GET  /sse/sim             — Stream of sim events (fills, P&L updates)

The router expects a ``SimulationEngine`` instance at ``app.state.sim_engine``.
This is wired up during the FastAPI lifespan in ``main.py``.

Usage::

    from lib.services.data.api.simulation_api import router, sse_router

    app.include_router(router)
    app.include_router(sse_router)
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from collections.abc import AsyncGenerator

logger = logging.getLogger("api.simulation")

__all__ = [
    "router",
    "sse_router",
]

# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------

router = APIRouter(prefix="/api/sim", tags=["simulation"])
sse_router = APIRouter(tags=["simulation-sse"])


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class OrderRequest(BaseModel):
    """Request body for submitting a simulated order."""

    symbol: str = Field(..., description="Internal ticker, e.g. 'KRAKEN:XBTUSD'")
    side: str = Field(..., description="'long' or 'short'")
    qty: float = Field(default=1.0, gt=0, description="Number of contracts / units")
    order_type: str = Field(default="market", description="'market' or 'limit'")
    limit_price: float | None = Field(default=None, gt=0, description="Limit price (required for limit orders)")
    stop_loss: float | None = Field(default=None, description="Stop-loss price")
    take_profit: float | None = Field(default=None, description="Take-profit price")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_engine(request: Request):
    """Retrieve the SimulationEngine from app state.

    Raises HTTPException 503 if the engine is not available.
    """
    engine = getattr(request.app.state, "sim_engine", None)
    if engine is None:
        # Fallback: try the module-level singleton
        try:
            from lib.services.engine.simulation import get_sim_engine

            engine = get_sim_engine()
        except ImportError:
            pass

    if engine is None:
        raise HTTPException(
            status_code=503,
            detail="Simulation engine is not initialised. Set SIM_ENABLED=1 to enable.",
        )
    return engine


def _redis_pubsub():
    """Return a Redis pub/sub object subscribed to ``futures:events``, or None."""
    try:
        from lib.core.cache import REDIS_AVAILABLE, _r

        if not REDIS_AVAILABLE or _r is None:
            return None
        ps = _r.pubsub()
        ps.subscribe("futures:events")
        return ps
    except Exception:
        return None


# ---------------------------------------------------------------------------
# REST routes
# ---------------------------------------------------------------------------


@router.get("/status")
async def sim_status(request: Request) -> dict[str, Any]:
    """Return the full simulation state.

    Includes open positions, pending orders, P&L summary, today's
    trades, latest prices, and engine metadata.
    """
    engine = _get_engine(request)
    logger.debug("sim_status requested")
    return engine.get_status()


@router.post("/order")
async def sim_order(request: Request, body: OrderRequest) -> dict[str, Any]:
    """Submit a simulated order (market or limit).

    For market orders the fill is immediate at the current tick price.
    For limit orders the order is held pending until the price crosses
    the specified limit price.

    Returns the fill confirmation or pending order details.
    """
    engine = _get_engine(request)

    order_type = body.order_type.lower()
    logger.info(
        "sim_order",
        extra={
            "symbol": body.symbol,
            "side": body.side,
            "qty": body.qty,
            "order_type": order_type,
            "limit_price": body.limit_price,
        },
    )

    if order_type == "market":
        result = engine.submit_market_order(
            symbol=body.symbol,
            side=body.side,
            qty=body.qty,
            stop_loss=body.stop_loss,
            take_profit=body.take_profit,
        )
    elif order_type == "limit":
        if body.limit_price is None or body.limit_price <= 0:
            raise HTTPException(status_code=400, detail="limit_price is required for limit orders and must be > 0.")
        result = engine.submit_limit_order(
            symbol=body.symbol,
            side=body.side,
            qty=body.qty,
            limit_price=body.limit_price,
            stop_loss=body.stop_loss,
            take_profit=body.take_profit,
        )
    else:
        raise HTTPException(status_code=400, detail=f"Invalid order_type: {order_type!r}. Use 'market' or 'limit'.")

    # If the engine returned an error dict, surface it as 400
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])

    return result


@router.post("/close/{symbol}")
async def sim_close_position(request: Request, symbol: str) -> dict[str, Any]:
    """Close the open position for *symbol* at the current market price.

    The symbol should be the internal ticker (e.g. ``KRAKEN:XBTUSD``).
    URL-encoded colons (``%3A``) are accepted.
    """
    engine = _get_engine(request)
    logger.info("sim_close_position", extra={"symbol": symbol})

    result = engine.close_position(symbol)
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result


@router.post("/close-all")
async def sim_close_all(request: Request) -> dict[str, Any]:
    """Close all open positions and cancel all pending orders."""
    engine = _get_engine(request)
    logger.info("sim_close_all")
    return engine.close_all()


@router.post("/reset")
async def sim_reset(request: Request) -> dict[str, Any]:
    """Reset the simulation — close all positions, clear history, restore balance."""
    engine = _get_engine(request)
    logger.info("sim_reset")
    return engine.reset()


@router.get("/trades")
async def sim_trades(
    request: Request,
    today_only: bool = Query(default=False, description="If true, only return today's trades"),
) -> list[dict[str, Any]]:
    """Return completed simulation trades.

    By default returns all trades.  Pass ``?today_only=true`` to filter
    to today's UTC date only.
    """
    engine = _get_engine(request)
    logger.debug("sim_trades", extra={"today_only": today_only})
    return engine.get_recorded_trades(today_only=today_only)


@router.get("/pnl")
async def sim_pnl(request: Request) -> dict[str, Any]:
    """Return a concise P&L summary.

    Includes balance, daily P&L, total P&L, unrealised P&L, and counts
    of open positions, pending orders, and closed trades.
    """
    engine = _get_engine(request)
    logger.debug("sim_pnl")
    return engine.get_pnl_summary()


# ---------------------------------------------------------------------------
# SSE stream
# ---------------------------------------------------------------------------


async def _sim_event_generator(
    request: Request,
) -> AsyncGenerator[str]:
    """Yield SSE-formatted simulation events.

    Listens to the ``futures:events`` Redis pub/sub channel and forwards
    events that start with ``sim_`` to the connected client.  Also
    periodically sends a full status snapshot and heartbeat comments to
    keep the connection alive through proxies / load-balancers.

    Falls back to polling the engine status if Redis is unavailable.
    """
    engine = getattr(request.app.state, "sim_engine", None)
    if engine is None:
        try:
            from lib.services.engine.simulation import get_sim_engine

            engine = get_sim_engine()
        except ImportError:
            pass

    pubsub = _redis_pubsub()
    use_redis = pubsub is not None

    logger.info("sim SSE stream started", extra={"redis": use_redis})
    last_heartbeat = time.monotonic()
    last_status = time.monotonic()

    _STATUS_INTERVAL_S = 2.0
    _HEARTBEAT_INTERVAL_S = 15.0
    _POLL_INTERVAL_S = 0.1

    try:
        while True:
            # Check if client disconnected
            if await request.is_disconnected():
                logger.info("sim SSE client disconnected")
                break

            now = time.monotonic()

            # Try to read events from Redis pub/sub
            if use_redis and pubsub is not None:
                try:
                    msg = pubsub.get_message(ignore_subscribe_messages=True, timeout=0.0)
                    if msg and msg.get("type") == "message":
                        raw = msg.get("data", b"")
                        if isinstance(raw, bytes):
                            raw = raw.decode("utf-8", errors="replace")

                        # Only forward sim-related events
                        try:
                            data = json.loads(raw)
                            event_type = data.get("event", "")
                            if event_type.startswith("sim_"):
                                yield f"event: {event_type}\ndata: {raw}\n\n"
                        except (json.JSONDecodeError, TypeError):
                            pass
                except Exception:
                    # Redis connection issue — degrade to polling
                    use_redis = False
                    pubsub = None

            # Periodic full status snapshot
            if engine is not None and now - last_status >= _STATUS_INTERVAL_S:
                try:
                    status = engine.get_status()
                    payload = json.dumps(status, default=str)
                    yield f"event: sim-status\ndata: {payload}\n\n"
                except Exception:
                    pass
                last_status = now

            # Heartbeat to keep connection alive
            if now - last_heartbeat >= _HEARTBEAT_INTERVAL_S:
                yield ": heartbeat\n\n"
                last_heartbeat = now

            await asyncio.sleep(_POLL_INTERVAL_S)

    except asyncio.CancelledError:
        logger.info("sim SSE stream cancelled")
    except Exception:
        logger.exception("sim SSE stream error")
    finally:
        if pubsub is not None:
            try:
                pubsub.unsubscribe()
                pubsub.close()
            except Exception:
                pass
        logger.info("sim SSE stream closed")


@sse_router.get("/sse/sim")
async def sse_sim(request: Request) -> StreamingResponse:
    """SSE endpoint streaming real-time simulation events.

    Connect via ``EventSource("/sse/sim")`` in the browser.  Events
    include:

    - ``sim-status`` — periodic full state snapshot (every ~2 s)
    - ``sim_fill`` — order fill notifications
    - ``sim_pnl`` — P&L update notifications

    A heartbeat comment is sent every 15 s to keep proxies happy.
    """
    logger.info("sse_sim connection opened")

    return StreamingResponse(
        _sim_event_generator(request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
