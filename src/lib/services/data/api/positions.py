"""
Positions API router — live position management (v3, broker-agnostic).

Handles receiving live position snapshots from any connected broker
(TradingView/Tradovate, or future integrations) and serving them to
the dashboard.  Also provides proxy endpoints that let the web
dashboard send commands *back* to the connected broker (execute signal,
flatten, cancel orders).

Endpoints:
  - POST /positions/update        — Push a position snapshot (from broker)
  - POST /positions/heartbeat     — Broker keep-alive with account summary
  - POST /positions/execute       — Proxy: forward a trade signal to broker
  - POST /positions/flatten       — Proxy: flatten all positions via broker
  - POST /positions/cancel_orders — Proxy: cancel working orders via broker
  - GET  /positions/broker_status — Read broker connection status
  - GET  /positions/broker_orders — Read recent order events from broker
  - GET  /positions/              — Get current cached positions
  - DELETE /positions/            — Clear cached positions

Risk enforcement:
  When a broker pushes a position snapshot via POST /positions/update,
  the router syncs the positions into the local RiskManager and evaluates
  all risk rules.  The response includes risk status fields so the broker
  connector knows immediately if trading limits have been hit.
"""

import json
import logging
import os
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

import httpx
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from lib.core.cache import (  # noqa: PLC2701
    _cache_key,
    cache_get,
    cache_set,
)

_EST = ZoneInfo("America/New_York")
logger = logging.getLogger("api.positions")

router = APIRouter(tags=["Positions"])

# ---------------------------------------------------------------------------
# Cache key & TTL for live positions
# ---------------------------------------------------------------------------
_POSITIONS_CACHE_KEY = _cache_key("live_positions", "current")
_POSITIONS_TTL = 7200  # 2 hours — positions persist across brief disconnects

# Broker heartbeat cache (separate from positions — shorter TTL)
_HEARTBEAT_CACHE_KEY = _cache_key("broker_heartbeat", "current")
_HEARTBEAT_TTL = 60  # 1 minute — if no heartbeat within 60s, broker is "stale"

# Broker status probe cache
_BROKER_STATUS_CACHE_KEY = _cache_key("broker_status", "latest")
_BROKER_STATUS_TTL = 30  # 30 seconds — refreshed on each probe or heartbeat

# Broker connector URL — the TradingView/Tradovate webhook relay or
# any future broker connector that accepts commands.
# Module-level defaults — overridden at call time by persisted Redis settings
# (edited via the Settings → Services UI).  Env vars are the last fallback.
_BROKER_HOST_DEFAULT = os.getenv("TV_BROKER_HOST", "")
_BROKER_PORT_DEFAULT = int(os.getenv("TV_BROKER_PORT", "") or "0")
_BROKER_TIMEOUT = 5.0  # seconds


def _get_broker_host_port() -> tuple[str, int]:
    """Return (host, port) for the broker connector.

    Priority:
      1. Persisted Redis settings (saved via Settings → Services UI)
      2. ``TV_BROKER_HOST`` / ``TV_BROKER_PORT`` environment variables
      3. Hard-coded defaults
    """
    try:
        from lib.core.cache import cache_get

        raw = cache_get("settings:overrides")
        if raw:
            import json as _json

            data = _json.loads(raw)
            svc = data.get("services", {})
            host = svc.get("tv_broker_host") or _BROKER_HOST_DEFAULT
            port = int(svc.get("tv_broker_port") or _BROKER_PORT_DEFAULT)
            return host, port
    except Exception:
        pass
    return _BROKER_HOST_DEFAULT, _BROKER_PORT_DEFAULT


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class Position(BaseModel):
    """A single live position from the broker."""

    symbol: str = Field(..., description="Instrument full name, e.g. 'MESZ5' or 'MGC'")
    side: str = Field(..., description="Long or Short")
    quantity: float = Field(..., description="Number of contracts")
    avgPrice: float = Field(..., description="Average fill price")
    unrealizedPnL: float = Field(0.0, description="Current unrealized P&L in USD")
    instrument: str | None = Field(None, description="Master instrument name, e.g. 'MES'")
    tickSize: float | None = Field(None, description="Tick size for this instrument")
    pointValue: float | None = Field(None, description="Point value for this instrument")
    lastUpdate: str | None = Field(None, description="ISO timestamp of last update")


class PendingOrder(BaseModel):
    """A working/accepted order from the broker."""

    orderId: str = Field("", description="Broker order ID")
    name: str = Field("", description="Order name/label")
    instrument: str = Field("", description="Instrument full name")
    action: str = Field("", description="Buy, Sell, SellShort, BuyToCover")
    type: str = Field("", description="Market, Limit, StopMarket, etc.")
    quantity: int = Field(0, description="Order quantity")
    limitPrice: float = Field(0.0, description="Limit price (0 if not applicable)")
    stopPrice: float = Field(0.0, description="Stop price (0 if not applicable)")
    state: str = Field("", description="Order state: Working, Accepted, etc.")


class PositionsPayload(BaseModel):
    """Payload pushed by a broker connector (TradingView/Tradovate, etc.).

    Fields are broker-agnostic; any connector that pushes positions should
    conform to this schema.
    """

    account: str = Field(..., description="Broker account name, e.g. 'Tradovate-Sim101'")
    positions: list[Position] = Field(default_factory=list, description="List of open positions")
    pendingOrders: list[PendingOrder] = Field(default_factory=list, description="Working/accepted orders")
    timestamp: str | None = Field(None, description="UTC timestamp from broker")
    cashBalance: float = Field(0.0, description="Account cash balance")
    realizedPnL: float = Field(0.0, description="Today's realized P&L")
    totalUnrealizedPnL: float = Field(0.0, description="Sum of all unrealized P&L")
    riskBlocked: bool = Field(False, description="True if broker-side risk enforcement is blocking new trades")
    riskBlockReason: str = Field("", description="Reason for risk block (if any)")
    broker_version: str = Field("1.0", description="Broker connector version string")
    source: str = Field("unknown", description="Source identifier: 'tradingview', 'tradovate', etc.")


class HeartbeatPayload(BaseModel):
    """Lightweight heartbeat from the broker connector."""

    account: str = Field(..., description="Broker account name")
    state: str = Field("", description="Connection state (e.g. Realtime, Connected)")
    connected: bool = Field(True, description="Whether the account is connected")
    positions: int = Field(0, description="Number of open positions")
    cashBalance: float = Field(0.0, description="Account cash balance")
    riskBlocked: bool = Field(False, description="Whether risk enforcement is blocking")
    broker_version: str = Field("1.0", description="Broker connector version")
    listenerPort: int = Field(0, description="Broker connector listener port (if applicable)")
    timestamp: str | None = Field(None, description="UTC timestamp")
    source: str = Field("unknown", description="Source identifier: 'tradingview', 'tradovate', etc.")


class PositionsResponse(BaseModel):
    """Response returned by GET /."""

    account: str = ""
    positions: list[Position] = []
    timestamp: str = ""
    received_at: str = ""
    has_positions: bool = False
    total_unrealized_pnl: float = 0.0
    cash_balance: float = 0.0
    realized_pnl: float = 0.0
    pending_orders: list[PendingOrder] = []
    broker_connected: bool = False
    broker_version: str = ""
    source: str = ""


class ExecuteSignalRequest(BaseModel):
    """Request body for sending a trade signal to the broker."""

    direction: str = Field(..., description="'long' or 'short'")
    quantity: int = Field(1, ge=1, description="Number of contracts (will be risk-sized by broker)")
    order_type: str = Field("market", description="'market', 'limit', or 'stop'")
    limit_price: float = Field(0.0, description="Limit price (for limit/stop orders)")
    stop_loss: float = Field(0.0, description="Exact stop loss price (0 = use broker default)")
    take_profit: float = Field(0.0, description="Exact take profit price (0 = use broker default)")
    tp2: float = Field(0.0, description="Second take profit target (0 = none)")
    strategy: str = Field("", description="Strategy name for logging")
    asset: str = Field("", description="Asset name for logging")
    signal_id: str = Field("", description="Unique signal ID for tracking (auto-generated if empty)")
    enforce_risk: bool = Field(
        True,
        description="If True, run a pre-flight risk check before forwarding to broker",
    )


class FlattenRequest(BaseModel):
    """Request body for flattening all positions."""

    reason: str = Field("dashboard", description="Reason for flattening (for audit trail)")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Helper: forward HTTP requests to the broker connector
# ---------------------------------------------------------------------------


def _get_broker_url() -> str:
    """Return the base URL for the broker connector.

    Host is always resolved from persisted Redis settings (Settings UI) or
    env-var fallback via ``_get_broker_host_port()``.  Port is then
    cross-checked against the ``listenerPort`` field in the latest heartbeat
    so that a connector restart on a different port is picked up automatically.

    If no explicit host/port is configured but a heartbeat is present with
    a ``listenerPort``, we fall back to ``localhost`` — the broker connector
    is assumed to be local when it sends heartbeats without an explicit host.
    """
    host, port = _get_broker_host_port()
    try:
        raw = cache_get(_HEARTBEAT_CACHE_KEY)
        if raw:
            hb = json.loads(raw)
            hb_port = hb.get("listenerPort", 0)
            if hb_port:
                port = hb_port
            # If no host is explicitly configured but a heartbeat exists
            # with a listenerPort, assume the broker is on localhost.
            if not host and port:
                host = "localhost"
    except Exception:
        pass
    if not host or not port:
        return ""
    return f"http://{host}:{port}"


def _is_broker_alive() -> bool:
    """Check whether we've received a heartbeat recently."""
    try:
        raw = cache_get(_HEARTBEAT_CACHE_KEY)
        if raw is None:
            return False
        hb = json.loads(raw)
        received = hb.get("received_at", "")
        if not received:
            return False
        dt = datetime.fromisoformat(received)
        age = (datetime.now(tz=_EST) - dt).total_seconds()
        return age < _HEARTBEAT_TTL
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Endpoints — Broker → Python (push)
# ---------------------------------------------------------------------------


@router.post("/update")
def update_positions(payload: PositionsPayload):
    """Receive live position snapshot from a broker connector.

    The broker connector POSTs here on every position change (open, close,
    partial fill, P&L tick).  Also sends account balance, realized P&L,
    pending orders, and risk-block state.

    The response includes risk evaluation fields so the broker connector
    knows immediately if any risk limits have been reached.
    """
    received_at = datetime.now(tz=_EST).isoformat()

    position_dicts = [p.model_dump() for p in payload.positions]
    pending_order_dicts = [o.model_dump() for o in payload.pendingOrders]

    data = {
        "account": payload.account,
        "positions": position_dicts,
        "pendingOrders": pending_order_dicts,
        "timestamp": payload.timestamp or received_at,
        "received_at": received_at,
        "cashBalance": payload.cashBalance,
        "realizedPnL": payload.realizedPnL,
        "totalUnrealizedPnL": payload.totalUnrealizedPnL,
        "riskBlocked": payload.riskBlocked,
        "riskBlockReason": payload.riskBlockReason,
        "broker_version": payload.broker_version,
        "source": payload.source,
    }

    cache_set(
        _POSITIONS_CACHE_KEY,
        json.dumps(data).encode(),
        _POSITIONS_TTL,
    )

    total_pnl = sum(p.unrealizedPnL for p in payload.positions)
    open_count = len([p for p in payload.positions if p.quantity > 0])

    logger.info(
        "Position update: account=%s positions=%d total_pnl=%.2f balance=%.2f source=%s",
        payload.account,
        open_count,
        total_pnl,
        payload.cashBalance,
        payload.source,
    )

    # --- Risk evaluation ---
    risk_status: dict[str, Any] = {}
    try:
        from lib.services.data.api.risk import evaluate_position_risk

        risk_status = evaluate_position_risk(position_dicts)

        if not risk_status.get("can_trade", True):
            logger.warning(
                "⚠️ Risk block after position sync: %s (daily P&L $%.2f)",
                risk_status.get("block_reason", ""),
                risk_status.get("daily_pnl", 0.0),
            )
        for warning in risk_status.get("warnings", []):
            logger.warning("⚠️ Risk warning: %s", warning)
    except Exception as exc:
        logger.debug("Risk evaluation skipped (non-fatal): %s", exc)

    return {
        "status": "received",
        "account": payload.account,
        "positions_count": len(payload.positions),
        "open_positions": open_count,
        "total_unrealized_pnl": round(total_pnl, 2),
        "received_at": received_at,
        "risk": risk_status,
    }


@router.post("/heartbeat")
def receive_heartbeat(payload: HeartbeatPayload):
    """Receive a keep-alive heartbeat from the broker connector.

    The connector sends this periodically so the dashboard knows the
    connection is alive even when there are no position changes.  The
    heartbeat also carries the connector's listener port so the
    execute/flatten proxy endpoints know where to forward requests.

    Also triggers a background probe of broker /status to keep the
    cached rich status fresh for the health indicators.
    """
    received_at = datetime.now(tz=_EST).isoformat()

    hb_data = {
        "account": payload.account,
        "state": payload.state,
        "connected": payload.connected,
        "positions": payload.positions,
        "cashBalance": payload.cashBalance,
        "riskBlocked": payload.riskBlocked,
        "broker_version": payload.broker_version,
        "listenerPort": payload.listenerPort,
        "received_at": received_at,
        "timestamp": payload.timestamp or received_at,
        "source": payload.source,
    }

    cache_set(
        _HEARTBEAT_CACHE_KEY,
        json.dumps(hb_data).encode(),
        _HEARTBEAT_TTL,
    )

    logger.debug(
        "Broker heartbeat: account=%s state=%s positions=%d source=%s",
        payload.account,
        payload.state,
        payload.positions,
        payload.source,
    )

    # Probe broker /status in-band to keep cached rich status fresh
    _probe_broker_status()

    return {
        "status": "ok",
        "received_at": received_at,
    }


# ---------------------------------------------------------------------------
# Endpoints — Python → Broker (proxy to broker connector)
# ---------------------------------------------------------------------------


@router.post("/execute")
def execute_signal(req: ExecuteSignalRequest):
    """Send a trade signal to the broker via its HTTP connector.

    This is the primary way the web dashboard triggers order execution.
    The signal is forwarded to the broker's ``/execute_signal`` endpoint,
    which queues it for execution.

    Optionally runs a pre-flight risk check before forwarding.
    """
    if not _is_broker_alive():
        raise HTTPException(
            status_code=503,
            detail="Broker connector is not connected (no recent heartbeat)",
        )

    # --- Optional pre-flight risk check ---
    if req.enforce_risk:
        try:
            from lib.services.data.api.risk import check_trade_entry_risk

            allowed, reason, details = check_trade_entry_risk(
                symbol=req.asset or "UNKNOWN",
                side=req.direction.upper(),
                size=req.quantity,
            )
            if not allowed:
                return {
                    "status": "rejected",
                    "reason": f"Risk check failed: {reason}",
                    "risk_details": details,
                }
        except Exception as exc:
            logger.debug("Pre-flight risk check unavailable (non-fatal): %s", exc)

    # --- Forward to broker connector ---
    broker_url = _get_broker_url()
    if not broker_url:
        raise HTTPException(
            status_code=503,
            detail="Broker connector URL is not configured (set TV_BROKER_HOST/TV_BROKER_PORT)",
        )

    signal_payload = req.model_dump()

    try:
        with httpx.Client(timeout=_BROKER_TIMEOUT) as client:
            resp = client.post(f"{broker_url}/execute_signal", json=signal_payload)
            resp.raise_for_status()
            result = resp.json()
            result["forwarded_to"] = broker_url
            return result
    except httpx.ConnectError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Cannot connect to broker at {broker_url}",
        ) from exc
    except httpx.TimeoutException as exc:
        raise HTTPException(
            status_code=504,
            detail=f"Broker at {broker_url} timed out",
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Broker communication error: {exc}",
        ) from exc


@router.post("/flatten")
def flatten_all(req: FlattenRequest):
    """Flatten all positions by forwarding to the broker's /flatten endpoint.

    Closes all open positions at market and cancels all working orders.
    """
    if not _is_broker_alive():
        raise HTTPException(
            status_code=503,
            detail="Broker connector is not connected (no recent heartbeat)",
        )

    broker_url = _get_broker_url()
    if not broker_url:
        raise HTTPException(
            status_code=503,
            detail="Broker connector URL is not configured",
        )

    try:
        with httpx.Client(timeout=_BROKER_TIMEOUT) as client:
            resp = client.post(
                f"{broker_url}/flatten",
                json={"reason": req.reason},
            )
            resp.raise_for_status()
            result = resp.json()
            result["forwarded_to"] = broker_url
            logger.warning("🔴 FLATTEN ALL sent to broker — reason: %s", req.reason)
            return result
    except httpx.ConnectError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Cannot connect to broker at {broker_url}",
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Broker communication error: {exc}",
        ) from exc


@router.post("/cancel_orders")
def cancel_orders():
    """Cancel all working orders by forwarding to the broker's /cancel_orders endpoint."""
    if not _is_broker_alive():
        raise HTTPException(
            status_code=503,
            detail="Broker connector is not connected (no recent heartbeat)",
        )

    broker_url = _get_broker_url()
    if not broker_url:
        raise HTTPException(
            status_code=503,
            detail="Broker connector URL is not configured",
        )

    try:
        with httpx.Client(timeout=_BROKER_TIMEOUT) as client:
            resp = client.post(f"{broker_url}/cancel_orders")
            resp.raise_for_status()
            result = resp.json()
            result["forwarded_to"] = broker_url
            logger.info("Cancel all orders sent to broker")
            return result
    except httpx.ConnectError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Cannot connect to broker at {broker_url}",
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Broker communication error: {exc}",
        ) from exc


def _probe_broker_status() -> dict[str, Any]:
    """Fetch broker /status and cache the result for health indicators.

    Called by get_broker_status() and can also be called by the
    health endpoint to ensure fresh data.  Returns the parsed
    JSON dict or {} on failure.
    """
    if not _is_broker_alive():
        return {}

    broker_url = _get_broker_url()
    if not broker_url:
        return {}

    broker_status: dict[str, Any] = {}
    try:
        with httpx.Client(timeout=_BROKER_TIMEOUT) as client:
            resp = client.get(f"{broker_url}/status")
            resp.raise_for_status()
            broker_status = resp.json()

        # Cache the rich status so health indicators can read it
        cache_set(
            _BROKER_STATUS_CACHE_KEY,
            json.dumps(broker_status).encode(),
            _BROKER_STATUS_TTL,
        )
    except Exception as exc:
        logger.debug("Could not fetch broker /status: %s", exc)

    return broker_status


@router.get("/broker_status")
def get_broker_status():
    """Read the broker connector's /status endpoint.

    Returns account info, position count, balance, risk state,
    and connector configuration.  Also includes heartbeat age to
    indicate connection freshness.
    """
    # Return cached heartbeat data + live status if broker is reachable
    heartbeat_data = {}
    try:
        raw = cache_get(_HEARTBEAT_CACHE_KEY)
        if raw:
            heartbeat_data = json.loads(raw)
    except Exception:
        pass

    broker_alive = _is_broker_alive()
    broker_status = _probe_broker_status() if broker_alive else {}

    return {
        "broker_alive": broker_alive,
        "heartbeat": heartbeat_data,
        "live_status": broker_status,
        "broker_url": _get_broker_url(),
        "timestamp": datetime.now(tz=_EST).isoformat(),
    }


@router.get("/broker_orders")
def get_broker_orders():
    """Read the broker connector's /orders endpoint.

    Returns recent order events (fills, rejects, cancellations)
    tracked by the connector for audit and display on the dashboard.
    """
    if not _is_broker_alive():
        return {
            "broker_alive": False,
            "events": [],
            "error": "Broker connector is not connected",
        }

    broker_url = _get_broker_url()
    if not broker_url:
        return {
            "broker_alive": False,
            "events": [],
            "error": "Broker connector URL is not configured",
        }

    try:
        with httpx.Client(timeout=_BROKER_TIMEOUT) as client:
            resp = client.get(f"{broker_url}/orders")
            resp.raise_for_status()
            return resp.json()
    except Exception as exc:
        return {
            "broker_alive": True,
            "events": [],
            "error": f"Could not fetch orders: {exc}",
        }


# ---------------------------------------------------------------------------
# Endpoints — Dashboard reads (GET)
# ---------------------------------------------------------------------------


@router.get("/", response_model=PositionsResponse)
def get_positions():
    """Get current live positions from the broker.

    Returns the most recent position snapshot pushed by the broker
    connector.  If no data has been received (or the cache has
    expired), returns an empty response.
    """
    raw = cache_get(_POSITIONS_CACHE_KEY)
    if raw is None:
        return PositionsResponse()

    try:
        data = json.loads(raw.decode())
        positions = [Position(**p) for p in data.get("positions", [])]
        pending = [PendingOrder(**o) for o in data.get("pendingOrders", [])]
        total_pnl = sum(p.unrealizedPnL for p in positions)

        return PositionsResponse(
            account=data.get("account", ""),
            positions=positions,
            timestamp=data.get("timestamp", ""),
            received_at=data.get("received_at", ""),
            has_positions=len(positions) > 0,
            total_unrealized_pnl=round(total_pnl, 2),
            cash_balance=data.get("cashBalance", 0.0),
            realized_pnl=data.get("realizedPnL", 0.0),
            pending_orders=pending,
            broker_connected=_is_broker_alive(),
            broker_version=data.get("broker_version", ""),
            source=data.get("source", ""),
        )
    except Exception as exc:
        logger.error("Failed to parse cached positions: %s", exc)
        return PositionsResponse()


@router.delete("/")
def clear_positions():
    """Clear cached live positions and heartbeat (e.g. end of day reset).

    Useful when the broker is closed but stale position data
    remains in cache.
    """
    keys_to_clear = [_POSITIONS_CACHE_KEY, _HEARTBEAT_CACHE_KEY]

    # Read REDIS_AVAILABLE at call time from the cache module (not from
    # the stale name binding captured at import time) so that test
    # fixtures that toggle the flag are respected.
    import lib.core.cache as _cache_mod

    if _cache_mod.REDIS_AVAILABLE:
        if _cache_mod._r is not None:
            for key in keys_to_clear:
                _cache_mod._r.delete(key)
    else:
        for key in keys_to_clear:
            _cache_mod._mem_cache.pop(key, None)

    return {"status": "cleared", "timestamp": datetime.now(tz=_EST).isoformat()}


# ---------------------------------------------------------------------------
# Helper: read live positions from cache (importable by other modules)
# ---------------------------------------------------------------------------


def get_live_positions() -> dict[str, Any]:
    """Read the latest positions from cache.

    Returns a dict with keys: account, positions (list of dicts),
    timestamp, received_at, has_positions, total_unrealized_pnl,
    cash_balance, realized_pnl, pending_orders, broker_connected,
    broker_version, source.

    Importable by other modules without going through HTTP.
    """
    raw = cache_get(_POSITIONS_CACHE_KEY)
    if raw is None:
        return {
            "account": "",
            "positions": [],
            "timestamp": "",
            "received_at": "",
            "has_positions": False,
            "total_unrealized_pnl": 0.0,
            "cash_balance": 0.0,
            "realized_pnl": 0.0,
            "pending_orders": [],
            "broker_connected": False,
            "broker_version": "",
            "source": "",
        }

    try:
        data = json.loads(raw.decode())
        positions = data.get("positions", [])
        total_pnl = sum(p.get("unrealizedPnL", 0) for p in positions)
        broker_conn = _is_broker_alive()
        broker_ver = data.get("broker_version", "")
        return {
            "account": data.get("account", ""),
            "positions": positions,
            "timestamp": data.get("timestamp", ""),
            "received_at": data.get("received_at", ""),
            "has_positions": len(positions) > 0,
            "total_unrealized_pnl": round(total_pnl, 2),
            "cash_balance": data.get("cashBalance", 0.0),
            "realized_pnl": data.get("realizedPnL", 0.0),
            "pending_orders": data.get("pendingOrders", []),
            "broker_connected": broker_conn,
            "broker_version": broker_ver,
            "source": data.get("source", ""),
        }
    except Exception:
        return {
            "account": "",
            "positions": [],
            "timestamp": "",
            "received_at": "",
            "has_positions": False,
            "total_unrealized_pnl": 0.0,
            "cash_balance": 0.0,
            "realized_pnl": 0.0,
            "pending_orders": [],
            "broker_connected": False,
            "broker_version": "",
            "source": "",
        }


# ---------------------------------------------------------------------------
# Backward compatibility — old class names used by test_bridge_trading.py
# and any other code that imported from the old NT8-specific module.
# These are thin aliases so existing imports don't break.
# ---------------------------------------------------------------------------
NTPosition = Position
NTPendingOrder = PendingOrder
NTPositionsPayload = PositionsPayload
NTHeartbeatPayload = HeartbeatPayload
NTPositionsResponse = PositionsResponse
