"""
Copy Trade API Router — Prop-Firm Compliant "SEND ALL" Order Execution
=======================================================================
Exposes the WebUI order-execution endpoints that sit in front of the
:class:`~lib.services.engine.copy_trader.CopyTrader` singleton.

Endpoints
---------
POST /api/copy-trade/send
    "SEND ALL" button handler — place on main + copy to all enabled slaves.
    Every order: ``OrderPlacement.MANUAL`` + humanised delay.
    Returns :class:`~lib.services.engine.copy_trader.CopyBatchResult` JSON
    and publishes ``copy-trade-update`` SSE event.

POST /api/copy-trade/send-from-ticker
    Convenience: resolve Yahoo ticker → Rithmic contract → SEND ALL.
    Useful for the auto-entry path when PositionManager fires a signal.

GET  /api/copy-trade/status
    CopyTrader singleton status: main/slave connection state, rate counter,
    contract cache size, recent batches.

GET  /api/copy-trade/history?limit=50
    Recent order history (newest first), up to ``limit`` entries.

GET  /api/copy-trade/compliance-log?limit=20
    Last N compliance checklists from Redis (audit trail).

GET  /api/copy-trade/rate
    Current rolling rate-limit counter status.

POST /api/copy-trade/high-impact
    Toggle high-impact mode (NFP/FOMC days) — increases slave delay to
    1–2 s.  Body: ``{"enabled": true}``.

POST /api/copy-trade/invalidate-cache
    Force re-resolve all front-month contracts (call when roll date hits).

Architecture
------------
The CopyTrader singleton is **engine-side** (see ``engine/main.py``).
The data service communicates with it either directly (embedded mode) or
via Redis when ``ENGINE_MODE=remote``.

In **embedded mode** (default for dev): the singleton is imported directly
and its async methods are awaited from the FastAPI async handlers.

In **remote mode** (production, separate engine container): orders are
written to a Redis command queue (``engine:cmd:copy_trade``), the engine
picks them up and executes them, then publishes the result to
``dashboard:copy_trade``.  The API waits up to 5 s for the result key
before returning a ``202 Accepted`` response with a ``batch_id`` the
browser can use to poll.

Environment Variables
---------------------
``RITHMIC_COPY_TRADING``   — "1" to enable Rithmic path (else 403 on send)
``ENGINE_MODE``            — "remote" to use Redis queue, else direct call
``CT_REMOTE_TIMEOUT``      — seconds to wait for remote result (default 5)
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import time
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Redis — imported at module level so tests can patch these names directly.
# ---------------------------------------------------------------------------
REDIS_AVAILABLE: bool = False
_r: Any = None
with contextlib.suppress(Exception):
    from lib.core.cache import REDIS_AVAILABLE, _r  # type: ignore[assignment]

logger = logging.getLogger("api.copy_trade")

router = APIRouter(tags=["Copy Trade"])

_COPY_TRADING_ENABLED = os.getenv("RITHMIC_COPY_TRADING", "0") == "1"
_ENGINE_MODE = os.getenv("ENGINE_MODE", "embedded")
_REMOTE_TIMEOUT = float(os.getenv("CT_REMOTE_TIMEOUT", "5"))

# Redis keys
_CMD_KEY = "engine:cmd:copy_trade"
_RESULT_KEY_PREFIX = "engine:copy_trade:result:"
_COMPLIANCE_KEY = "engine:copy_trader:compliance_log"


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class SendAllRequest(BaseModel):
    """Body for POST /api/copy-trade/send."""

    security_code: str = Field(..., description="Rithmic security code, e.g. 'MGCQ6'")
    exchange: str = Field(..., description="Exchange code, e.g. 'NYMEX'")
    side: str = Field(..., description="'BUY' or 'SELL'")
    qty: int = Field(1, ge=1, le=20, description="Number of contracts")
    order_type: str = Field("MARKET", description="'MARKET' or 'LIMIT'")
    price: float = Field(0.0, ge=0.0, description="Limit price (0 for MARKET)")
    stop_ticks: int = Field(20, ge=0, le=500, description="Hard stop in ticks (server-side bracket)")
    target_ticks: int | None = Field(None, ge=1, le=2000, description="Take-profit in ticks (optional)")
    tag_prefix: str = Field("WEBUI", description="Order tag prefix for audit trail")
    reason: str = Field("", max_length=200, description="Human-readable reason (logged + tagged)")


class SendFromTickerRequest(BaseModel):
    """Body for POST /api/copy-trade/send-from-ticker."""

    ticker: str = Field(..., description="Yahoo Finance ticker, e.g. 'MGC=F'")
    side: str = Field(..., description="'BUY' or 'SELL'")
    qty: int = Field(1, ge=1, le=20)
    order_type: str = Field("MARKET")
    price: float = Field(0.0, ge=0.0)
    stop_ticks: int = Field(20, ge=0, le=500)
    target_ticks: int | None = Field(None, ge=1, le=2000)
    tag_prefix: str = Field("WEBUI")
    reason: str = Field("", max_length=200)


class HighImpactRequest(BaseModel):
    """Body for POST /api/copy-trade/high-impact."""

    enabled: bool


class PyramidRequest(BaseModel):
    """Request body for the ADD PYRAMID endpoint."""

    ticker: str = Field(..., description="Yahoo ticker, e.g. 'MGC=F'")
    cnn_prob: float = Field(0.70, ge=0.0, le=1.0, description="CNN confidence for this add")
    regime: str = Field("", description="Current regime string, e.g. 'TRENDING_UP'")
    wave_ratio: float = Field(0.0, ge=0.0, description="Wave dominance ratio from wave_analysis")
    reason: str = Field("", max_length=120, description="Optional reason tag")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _check_enabled() -> None:
    """Raise 403 if copy trading is not enabled."""
    if not _COPY_TRADING_ENABLED:
        raise HTTPException(
            status_code=403,
            detail=(
                "Rithmic copy trading is disabled. Set RITHMIC_COPY_TRADING=1 in the engine environment to enable."
            ),
        )


def _get_copy_trader_direct():
    """Return the CopyTrader singleton for direct (embedded) calls.

    Returns ``None`` if the module or singleton is unavailable.
    """
    try:
        from lib.services.engine.copy_trader import get_copy_trader

        return get_copy_trader()
    except Exception as exc:
        logger.warning("copy_trader singleton unavailable: %s", exc)
        return None


def _get_position_manager():
    """Return the PositionManager singleton if available."""
    try:
        from lib.services.engine.main import _position_manager

        return _position_manager
    except (ImportError, AttributeError):
        return None


def _publish_sse(payload: dict[str, Any]) -> None:
    """Publish a ``copy-trade-update`` event to the SSE dashboard channel."""
    try:
        if REDIS_AVAILABLE and _r is not None:
            _r.publish("dashboard:copy_trade", json.dumps(payload, default=str))
    except Exception as exc:
        logger.debug("copy_trade SSE publish error (non-fatal): %s", exc)


def _enqueue_remote_command(batch_id: str, command: dict[str, Any]) -> None:
    """Write an order command to the Redis queue for the remote engine."""
    try:
        if REDIS_AVAILABLE and _r is not None:
            entry = json.dumps({"batch_id": batch_id, **command}, default=str)
            _r.rpush(_CMD_KEY, entry)
            _r.expire(_CMD_KEY, 60)
    except Exception as exc:
        logger.warning("copy_trade remote enqueue error: %s", exc)


async def _wait_for_remote_result(batch_id: str, timeout: float = _REMOTE_TIMEOUT) -> dict[str, Any] | None:
    """Poll Redis for a result written by the remote engine."""
    key = f"{_RESULT_KEY_PREFIX}{batch_id}"
    deadline = time.monotonic() + timeout
    try:
        if not REDIS_AVAILABLE or _r is None:
            return None

        while time.monotonic() < deadline:
            raw = _r.get(key)
            if raw:
                return json.loads(raw.decode() if isinstance(raw, bytes) else raw)
            await asyncio.sleep(0.25)
    except Exception as exc:
        logger.debug("copy_trade remote result poll error: %s", exc)
    return None


def _batch_result_to_response(result: Any) -> dict[str, Any]:
    """Serialise a CopyBatchResult (or dict from remote) to a plain dict."""
    if result is None:
        return {"ok": False, "error": "no result"}
    if isinstance(result, dict):
        return result
    if hasattr(result, "to_dict"):
        return result.to_dict()
    return {"ok": False, "error": "unrecognised result type"}


def _get_compliance_log(limit: int = 20) -> list[dict[str, Any]]:
    """Read the last ``limit`` compliance log entries from Redis."""
    entries: list[dict[str, Any]] = []
    try:
        if REDIS_AVAILABLE and _r is not None:
            raw_list = _r.lrange(_COMPLIANCE_KEY, 0, limit - 1)
            for raw in raw_list:
                with contextlib.suppress(Exception):
                    entries.append(json.loads(raw.decode() if isinstance(raw, bytes) else raw))
    except Exception as exc:
        logger.debug("compliance log read error: %s", exc)
    return entries


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/api/copy-trade/send")
async def send_all(body: SendAllRequest) -> JSONResponse:
    """**SEND ALL** — place on main + copy to all enabled slave accounts.

    This is the primary "one click" endpoint wired to the WebUI button.

    Every order enforces:
    - ``OrderPlacement.MANUAL``
    - Randomised humanised delay between slave copies
    - Server-side hard stop (``stop_ticks``)
    - Compliance checklist logged to Redis + SSE

    When ``RITHMIC_COPY_TRADING=1`` is **not** set, returns **403**.
    When ``ENGINE_MODE=remote``, the command is enqueued and a ``202``
    response is returned with the ``batch_id`` for polling.
    """
    _check_enabled()

    side = body.side.upper()
    if side not in ("BUY", "SELL"):
        raise HTTPException(status_code=422, detail="side must be 'BUY' or 'SELL'")

    order_type = body.order_type.upper()
    if order_type not in ("MARKET", "LIMIT"):
        raise HTTPException(status_code=422, detail="order_type must be 'MARKET' or 'LIMIT'")

    batch_id = f"WEBUI_{body.security_code}_{int(time.time())}"
    if body.reason:
        batch_id += f"_{body.reason[:20].replace(' ', '_')}"

    if _ENGINE_MODE == "remote":
        # ---- Remote engine mode: enqueue + poll ----
        cmd: dict[str, Any] = {
            "action": "send_order_and_copy",
            "security_code": body.security_code,
            "exchange": body.exchange,
            "side": side,
            "qty": body.qty,
            "order_type": order_type,
            "price": body.price,
            "stop_ticks": body.stop_ticks,
            "target_ticks": body.target_ticks,
            "tag_prefix": body.tag_prefix,
            "reason": body.reason,
            "submitted_at": datetime.now(UTC).isoformat(),
        }
        _enqueue_remote_command(batch_id, cmd)

        result = await _wait_for_remote_result(batch_id)
        if result is None:
            # Engine didn't respond in time — return accepted with batch_id
            return JSONResponse(
                status_code=202,
                content={
                    "ok": True,
                    "accepted": True,
                    "batch_id": batch_id,
                    "message": (
                        "Order enqueued — engine will execute shortly. "
                        f"Poll /api/copy-trade/result/{batch_id} for status."
                    ),
                },
            )
        _publish_sse({"event": "send_all", "batch_id": batch_id, "result": result})
        return JSONResponse(content={"ok": True, "batch_id": batch_id, "result": result})

    # ---- Embedded mode: direct async call ----
    ct = _get_copy_trader_direct()
    if ct is None:
        raise HTTPException(
            status_code=503,
            detail="CopyTrader not initialised — ensure the engine has started.",
        )

    try:
        result = await ct.send_order_and_copy(
            security_code=body.security_code,
            exchange=body.exchange,
            side=side,
            qty=body.qty,
            order_type=order_type,
            price=body.price,
            stop_ticks=body.stop_ticks,
            target_ticks=body.target_ticks,
            tag_prefix=body.tag_prefix,
            reason=body.reason,
        )
    except Exception as exc:
        logger.error("send_order_and_copy failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Order execution error: {exc}") from exc

    response_data = _batch_result_to_response(result)
    _publish_sse({"event": "send_all", "batch_id": batch_id, "result": response_data})

    return JSONResponse(
        content={
            "ok": True,
            "batch_id": batch_id,
            "result": response_data,
        }
    )


@router.post("/api/copy-trade/send-from-ticker")
async def send_from_ticker(body: SendFromTickerRequest) -> JSONResponse:
    """Resolve Yahoo ticker → Rithmic front-month contract → SEND ALL.

    Useful for the HTMX signal cards that already know the ticker but not
    the current contract month code (e.g. "MGC=F" → "MGCQ6").
    """
    _check_enabled()

    side = body.side.upper()
    if side not in ("BUY", "SELL"):
        raise HTTPException(status_code=422, detail="side must be 'BUY' or 'SELL'")

    batch_id = f"WEBUI_TICKER_{body.ticker}_{int(time.time())}"

    ct = _get_copy_trader_direct()
    if ct is None:
        raise HTTPException(status_code=503, detail="CopyTrader not initialised.")

    try:
        result = await ct.send_order_from_ticker(
            ticker=body.ticker,
            side=side,
            qty=body.qty,
            order_type=body.order_type.upper(),
            price=body.price,
            stop_ticks=body.stop_ticks,
            target_ticks=body.target_ticks,
            tag_prefix=body.tag_prefix,
            reason=body.reason,
        )
    except Exception as exc:
        logger.error("send_order_from_ticker failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Order execution error: {exc}") from exc

    response_data = _batch_result_to_response(result)
    _publish_sse({"event": "send_from_ticker", "batch_id": batch_id, "result": response_data})

    return JSONResponse(content={"ok": True, "batch_id": batch_id, "result": response_data})


@router.get("/api/copy-trade/status")
async def get_status() -> JSONResponse:
    """CopyTrader connection status, rate counter, and recent activity."""
    ct = _get_copy_trader_direct()
    if ct is None:
        return JSONResponse(
            content={
                "ok": False,
                "enabled": _COPY_TRADING_ENABLED,
                "engine_mode": _ENGINE_MODE,
                "error": "CopyTrader not initialised",
                "status": None,
            }
        )

    return JSONResponse(
        content={
            "ok": True,
            "enabled": _COPY_TRADING_ENABLED,
            "engine_mode": _ENGINE_MODE,
            "status": ct.status_summary(),
        }
    )


@router.get("/api/copy-trade/history")
async def get_history(limit: int = 50) -> JSONResponse:
    """Recent order batch history, newest first."""
    if limit < 1 or limit > 500:
        raise HTTPException(status_code=422, detail="limit must be between 1 and 500")

    ct = _get_copy_trader_direct()
    if ct is None:
        # Fallback: read from Redis log
        history: list[dict[str, Any]] = []
        try:
            if REDIS_AVAILABLE and _r is not None:
                raw_list = _r.lrange("engine:copy_trader:order_log", 0, limit - 1)
                for raw in raw_list:
                    with contextlib.suppress(Exception):
                        history.append(json.loads(raw.decode() if isinstance(raw, bytes) else raw))
        except Exception:
            pass
        return JSONResponse(content={"ok": True, "history": history, "source": "redis"})

    return JSONResponse(
        content={
            "ok": True,
            "history": ct.get_order_history(limit=limit),
            "source": "singleton",
        }
    )


@router.get("/api/copy-trade/compliance-log")
async def get_compliance_log(limit: int = 20) -> JSONResponse:
    """Last N compliance checklists (audit trail for prop-firm review)."""
    if limit < 1 or limit > 200:
        raise HTTPException(status_code=422, detail="limit must be between 1 and 200")
    return JSONResponse(
        content={
            "ok": True,
            "entries": _get_compliance_log(limit=limit),
            "count": limit,
        }
    )


@router.get("/api/copy-trade/rate")
async def get_rate_status() -> JSONResponse:
    """Current rolling 60-minute action counter status."""
    ct = _get_copy_trader_direct()
    if ct is None:
        return JSONResponse(content={"ok": False, "error": "CopyTrader not initialised"})
    return JSONResponse(content={"ok": True, "rate": ct.get_rate_status()})


@router.get("/api/copy-trade/rate-alert")
async def get_rate_alert() -> JSONResponse:
    """WebUI-friendly rate-limit alert: level, message, counters."""
    ct = _get_copy_trader_direct()
    if ct is None:
        return JSONResponse(
            content={
                "ok": False,
                "level": "ok",
                "message": "CopyTrader not initialised",
                "actions_60min": 0,
                "daily_actions": 0,
            }
        )
    return JSONResponse(content={"ok": True, **ct.get_rate_alert()})


@router.get("/api/copy-trade/focus")
async def get_focus_status() -> JSONResponse:
    """Return the PositionManager focus-lock state: focused asset + pyramid level."""
    pm = _get_position_manager()
    if pm is None:
        return JSONResponse(content={"ok": False, "focus_asset": None, "pyramid_enabled": False})
    summary = pm.status_summary()
    return JSONResponse(
        content={
            "ok": True,
            "focus_asset": summary.get("focus_asset"),
            "focus_lock_enabled": summary.get("focus_lock_enabled", True),
            "pyramid_enabled": summary.get("pyramid_enabled", True),
            "active_positions": summary.get("active_positions", 0),
            "positions": [
                {
                    "ticker": p.get("ticker"),
                    "symbol": p.get("symbol"),
                    "direction": p.get("direction"),
                    "phase": p.get("phase"),
                    "r_multiple": p.get("r_multiple"),
                    "pyramid_level": p.get("pyramid_level", 0),
                    "pyramid_contracts": p.get("pyramid_contracts", 0),
                    "total_contracts": p.get("total_contracts", 1),
                    "cnn_prob": p.get("cnn_prob"),
                }
                for p in summary.get("positions", [])
            ],
        }
    )


@router.post("/api/copy-trade/pyramid")
async def add_pyramid(body: PyramidRequest) -> JSONResponse:
    """ADD PYRAMID — add 1 contract to an existing winning position.

    Calls PositionManager.get_next_pyramid_level() to validate all gates,
    then apply_pyramid() to get the order commands, then routes them
    through CopyTrader.execute_order_commands() for multi-account execution.

    Returns 422 if no position exists for the ticker or if pyramid gates
    are not met (e.g. insufficient R-multiple, CNN too low, cooldown).
    """
    _check_enabled()

    pm = _get_position_manager()
    if pm is None:
        raise HTTPException(status_code=503, detail="PositionManager not initialised")

    # Get the existing position for this ticker
    pos = pm.get_position(body.ticker)
    if pos is None:
        raise HTTPException(
            status_code=422,
            detail=f"No active position for {body.ticker} — can only pyramid into existing positions",
        )

    # Get current price from the position's last known price
    current_price = pos.current_price

    # Evaluate pyramid gates
    pyramid_action = pm.get_next_pyramid_level(
        pos,
        current_price=current_price,
        cnn_prob=body.cnn_prob,
        regime=body.regime,
        wave_ratio=body.wave_ratio,
    )

    if pyramid_action is None:
        return JSONResponse(
            status_code=422,
            content={
                "ok": False,
                "error": "Pyramid gates not met",
                "detail": (
                    f"Position {body.ticker} {pos.direction}: R={pos.r_multiple:.2f}, "
                    f"level={pos.pyramid_level}, CNN={body.cnn_prob:.3f} — "
                    "check R-multiple threshold, CNN probability, cooldown, and regime requirements"
                ),
                "current_r": pos.r_multiple,
                "current_pyramid_level": pos.pyramid_level,
            },
        )

    # Apply the pyramid: get order commands
    orders = pm.apply_pyramid(pos, pyramid_action, current_price)

    # Route through CopyTrader
    ct = _get_copy_trader_direct()
    batch_id = f"PYRAMID_{body.ticker}_{int(time.time())}"

    if ct is not None:
        try:
            result = await ct.execute_order_commands(orders)
            response_data = _batch_result_to_response(result)
            _publish_sse(
                {"event": "pyramid", "batch_id": batch_id, "ticker": body.ticker, "level": pyramid_action["level"]}
            )
        except Exception as exc:
            logger.error("pyramid execute_order_commands failed: %s", exc, exc_info=True)
            response_data = {"ok": False, "error": str(exc)}
    else:
        # No CopyTrader — just log the order commands (dev/test mode)
        logger.info("Pyramid orders (no CopyTrader): %s", [o.to_dict() for o in orders])
        response_data = {"ok": True, "note": "CopyTrader not active — orders logged only"}

    return JSONResponse(
        content={
            "ok": True,
            "batch_id": batch_id,
            "ticker": body.ticker,
            "pyramid_action": pyramid_action,
            "orders_generated": len(orders),
            "result": response_data,
        }
    )


@router.post("/api/copy-trade/high-impact")
async def set_high_impact(body: HighImpactRequest) -> JSONResponse:
    """Toggle high-impact mode (NFP/FOMC days).

    When **enabled**, slave copy delays increase from 200–800 ms to 1–2 s
    to make copied orders look maximally human on high-volatility days.
    """
    ct = _get_copy_trader_direct()
    if ct is None:
        raise HTTPException(status_code=503, detail="CopyTrader not initialised.")

    ct.set_high_impact_mode(body.enabled)
    logger.info(
        "CopyTrader high-impact mode %s via API",
        "ENABLED" if body.enabled else "DISABLED",
    )
    return JSONResponse(
        content={
            "ok": True,
            "high_impact_mode": body.enabled,
            "message": (
                f"High-impact mode {'enabled' if body.enabled else 'disabled'}. "
                + ("Slave delays: 1–2 s." if body.enabled else "Slave delays: 200–800 ms.")
            ),
        }
    )


@router.post("/api/copy-trade/invalidate-cache")
async def invalidate_contract_cache(ticker: str | None = None) -> JSONResponse:
    """Force re-resolve front-month contracts on roll date.

    Pass ``ticker`` query param to invalidate a single symbol, or omit to
    clear all cached contracts.
    """
    ct = _get_copy_trader_direct()
    if ct is None:
        raise HTTPException(status_code=503, detail="CopyTrader not initialised.")

    ct.invalidate_contract_cache(ticker=ticker)
    msg = f"Contract cache cleared for {ticker}" if ticker else "All contract caches cleared"
    logger.info("copy_trade: %s", msg)
    return JSONResponse(content={"ok": True, "message": msg})


@router.get("/api/copy-trade/result/{batch_id}")
async def poll_result(batch_id: str) -> JSONResponse:
    """Poll for an async result written by the remote engine.

    Used when ``ENGINE_MODE=remote`` and ``/api/copy-trade/send`` returned 202.
    Returns 202 again if still pending, 200 with the result once ready.
    """
    key = f"{_RESULT_KEY_PREFIX}{batch_id}"
    raw = None
    try:
        if REDIS_AVAILABLE and _r is not None:
            raw = _r.get(key)
    except Exception:
        pass

    if raw is None:
        return JSONResponse(
            status_code=202,
            content={
                "ok": True,
                "pending": True,
                "batch_id": batch_id,
                "message": "Result not yet available — try again shortly.",
            },
        )

    result = json.loads(raw.decode() if isinstance(raw, bytes) else raw)
    return JSONResponse(content={"ok": True, "pending": False, "batch_id": batch_id, "result": result})


# ---------------------------------------------------------------------------
# HTMX fragment helpers
# ---------------------------------------------------------------------------


@router.get("/api/copy-trade/status/html", response_class=None)  # type: ignore[arg-type]
async def get_status_html() -> Any:
    """Return an HTMX-ready HTML fragment for the copy-trade status strip."""
    from fastapi.responses import HTMLResponse

    ct = _get_copy_trader_direct()
    if not _COPY_TRADING_ENABLED:
        return HTMLResponse(
            content=(
                '<div class="conn-dot conn-off" style="display:inline-block;margin-right:6px"></div>'
                '<span style="font-size:10px;color:var(--muted)">COPY TRADING DISABLED '
                "— set RITHMIC_COPY_TRADING=1</span>"
            )
        )

    if ct is None:
        return HTMLResponse(
            content=(
                '<div class="conn-dot conn-warn pulse" style="display:inline-block;margin-right:6px"></div>'
                '<span style="font-size:10px;color:var(--amber)">INITIALISING…</span>'
            )
        )

    summary = ct.status_summary()
    main = summary.get("main")
    rate = summary.get("rate_limit", {})
    slave_count = summary.get("enabled_slave_count", 0)
    high_impact = summary.get("high_impact_mode", False)

    if main and main.get("connected"):
        dot_cls = "conn-ok"
        color = "var(--green)"
        label = f"LIVE — main + {slave_count} slave(s)"
    elif main:
        dot_cls = "conn-warn pulse"
        color = "var(--amber)"
        label = "MAIN DISCONNECTED"
    else:
        dot_cls = "conn-off"
        color = "var(--muted)"
        label = "NO ACCOUNTS"

    warn = rate.get("is_warn", False)
    rate_txt = f"{rate.get('count', 0)}/{rate.get('hard_limit', 4500)} actions/60min"
    if rate.get("is_hard_limit"):
        rate_txt = f"⛔ RATE LIMIT — {rate_txt}"
    elif warn:
        rate_txt = f"⚠ {rate_txt}"

    hi_badge = (
        '<span style="font-size:9px;background:var(--amber);color:#000;'
        'border-radius:3px;padding:1px 5px;margin-left:6px">HIGH IMPACT</span>'
        if high_impact
        else ""
    )

    return HTMLResponse(
        content=(
            f'<div class="conn-dot {dot_cls}" style="display:inline-block;margin-right:6px"></div>'
            f'<span style="font-size:10px;color:{color}">{label}</span>'
            f"{hi_badge}"
            f'<span style="font-size:9px;color:var(--muted);margin-left:10px">{rate_txt}</span>'
        )
    )


@router.get("/api/copy-trade/history/html", response_class=None)  # type: ignore[arg-type]
async def get_history_html(limit: int = 10) -> Any:
    """Return an HTMX-ready HTML table of the last N order batches."""
    from fastapi.responses import HTMLResponse

    ct = _get_copy_trader_direct()
    history = ct.get_order_history(limit=limit) if ct else []

    if not history:
        return HTMLResponse(
            content='<div style="font-size:11px;color:var(--muted);padding:8px">No copy-trade batches yet.</div>'
        )

    rows = []
    for batch in history:
        ts = batch.get("submitted_at", "")[:19].replace("T", " ")
        side = batch.get("side", "")
        qty = batch.get("qty", "")
        sec = batch.get("security_code", "")
        ok = batch.get("ok", False)
        failed = batch.get("failed_count", 0)
        total = batch.get("total_orders", 0)
        color = "var(--green)" if ok else "var(--red)"
        status_txt = f"{total - failed}/{total} ok" if total else ("✓" if ok else "✗")
        rows.append(
            f"<tr>"
            f'<td style="color:var(--muted);font-family:var(--mono)">{ts}</td>'
            f'<td style="color:{"var(--green)" if side == "BUY" else "var(--red)"}">{side}</td>'
            f'<td style="color:#fff">{qty}× {sec}</td>'
            f'<td style="color:{color}">{status_txt}</td>'
            f"</tr>"
        )

    return HTMLResponse(
        content=(
            '<table class="tbl" style="width:100%;font-size:11px">'
            "<thead><tr>"
            "<th>Time</th><th>Side</th><th>Order</th><th>Status</th>"
            "</tr></thead>"
            "<tbody>" + "".join(rows) + "</tbody>"
            "</table>"
        )
    )


@router.get("/api/copy-trade/accounts/html", response_class=None)  # type: ignore[arg-type]
async def get_accounts_html() -> Any:
    """Return an HTMX-ready HTML fragment showing per-account status cards."""
    from fastapi.responses import HTMLResponse

    ct = _get_copy_trader_direct()
    if not _COPY_TRADING_ENABLED or ct is None:
        return HTMLResponse(
            content='<div style="font-size:11px;color:var(--muted);padding:6px">Copy trading not active.</div>'
        )

    summary = ct.status_summary()
    main = summary.get("main")
    slaves = summary.get("slaves", [])
    rate = summary.get("rate_limit", {})

    def _account_card(acct: dict, role: str) -> str:
        connected = acct.get("connected", False)
        label = acct.get("label", acct.get("key", "?"))
        orders = acct.get("order_count", 0)
        last_ts = acct.get("last_order_at", "") or ""
        last_disp = last_ts[11:19] if len(last_ts) >= 19 else "—"
        dot_cls = "conn-ok" if connected else "conn-warn pulse"
        dot_color = "var(--green)" if connected else "var(--amber)"
        enabled = acct.get("enabled", True)
        enabled_badge = (
            "" if enabled else '<span style="font-size:9px;color:var(--muted);margin-left:4px">[disabled]</span>'
        )
        return (
            f'<div style="background:var(--s3);border:1px solid var(--b1);border-radius:4px;padding:8px 10px;margin-bottom:6px">'
            f'<div style="display:flex;align-items:center;gap:6px;margin-bottom:4px">'
            f'<div class="conn-dot {dot_cls}" style="width:7px;height:7px"></div>'
            f'<span style="font-size:11px;color:{dot_color};font-weight:600">{label}</span>'
            f'<span style="font-size:9px;color:var(--muted);margin-left:4px">{role}</span>'
            f"{enabled_badge}"
            f"</div>"
            f'<div style="display:grid;grid-template-columns:1fr 1fr;gap:4px;font-size:10px;color:var(--muted)">'
            f'<span>Orders: <span style="color:var(--text)">{orders}</span></span>'
            f'<span>Last: <span style="color:var(--text)">{last_disp}</span></span>'
            f"</div>"
            f"</div>"
        )

    html_parts = []

    if main:
        html_parts.append(_account_card(main, "MAIN"))

    for slave in slaves:
        html_parts.append(_account_card(slave, "SLAVE"))

    # Rate limit strip
    count = rate.get("actions_60min", 0)
    daily = rate.get("daily_actions", 0)
    rate_color = "var(--red)" if rate.get("blocked") else ("var(--amber)" if rate.get("warn") else "var(--green)")
    html_parts.append(
        f'<div style="font-size:10px;color:var(--muted);border-top:1px solid var(--b1);padding-top:6px;margin-top:2px">'
        f'Rate: <span style="color:{rate_color}">{count}/hr</span> · '
        f'Daily: <span style="color:var(--text)">{daily}</span>'
        f"</div>"
    )

    if not html_parts:
        return HTMLResponse(content='<div style="font-size:11px;color:var(--muted)">No accounts connected.</div>')

    return HTMLResponse(content="".join(html_parts))
