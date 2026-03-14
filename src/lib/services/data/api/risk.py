"""
Risk API router — real-time risk engine endpoints.

Provides:
  - GET  /risk/status  — current risk engine state (from Redis or RiskManager)
  - POST /risk/check   — pre-flight check for a proposed trade entry
  - GET  /risk/history — recent risk events / blocks for audit trail

The RiskManager singleton lives in the engine service and publishes its
state to Redis.  This router reads from Redis (remote mode) or directly
from a local RiskManager instance (embedded mode).

When the data-service starts, it can optionally hold a local RiskManager
for request-scoped risk checks (pre-flight).  The engine's periodic
CHECK_RISK_RULES action keeps Redis up to date for the status endpoint.
"""

import json
import logging
import os
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger("api.risk")

_EST = ZoneInfo("America/New_York")

router = APIRouter(tags=["Risk"])


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class RiskCheckRequest(BaseModel):
    """Pre-flight risk check for a proposed trade."""

    symbol: str = Field(..., description="Instrument symbol, e.g. 'MGC', 'MNQ'")
    side: str = Field(..., description="LONG or SHORT")
    size: int = Field(1, ge=1, description="Number of contracts")
    risk_per_contract: float = Field(0.0, ge=0, description="Dollar risk per contract (stop distance × point value)")
    is_stack: bool = Field(False, description="True if adding to an existing position")
    wave_ratio: float = Field(1.0, description="Current wave ratio for stacking check")
    unrealized_r: float = Field(0.0, description="Current unrealized P&L in R-multiples for stacking check")


class RiskCheckResponse(BaseModel):
    """Result of a pre-flight risk check."""

    allowed: bool = Field(..., description="True if the trade is permitted")
    reason: str = Field("", description="Reason if blocked (empty when allowed)")
    symbol: str = ""
    side: str = ""
    size: int = 1
    total_risk: float = Field(0.0, description="Total dollar risk for this trade")
    max_risk_per_trade: float = Field(0.0, description="Account max risk per trade")
    daily_pnl: float = Field(0.0, description="Current daily P&L")
    open_trade_count: int = Field(0, description="Current open trade count")
    checked_at: str = Field("", description="ISO timestamp of check")


class RiskStatusResponse(BaseModel):
    """Current risk engine status."""

    account_size: int = 0
    max_risk_per_trade: float = 0.0
    max_daily_loss: float = 0.0
    max_open_trades: int = 0
    open_trade_count: int = 0
    daily_pnl: float = 0.0
    daily_trade_count: int = 0
    consecutive_losses: int = 0
    total_risk_exposure: float = 0.0
    risk_pct_of_account: float = 0.0
    can_trade: bool = True
    block_reason: str = ""
    is_past_entry_cutoff: bool = False
    is_overnight_warning: bool = False
    is_daily_loss_exceeded: bool = False
    is_max_trades_reached: bool = False
    last_check: str = ""
    trading_date: str = ""
    source: str = Field("", description="'redis' or 'local' — where data came from")


class RiskEvent(BaseModel):
    """A single risk event for the audit trail."""

    timestamp: str
    event_type: str  # "block", "warning", "clear"
    symbol: str = ""
    side: str = ""
    reason: str = ""
    daily_pnl: float = 0.0
    open_trades: int = 0


# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------

# Local RiskManager for request-scoped checks (lazy-initialised)
_local_risk_manager = None

# In-memory audit trail of recent risk events
_risk_events: list[dict[str, Any]] = []
_MAX_RISK_EVENTS = 100


def _get_local_risk_manager():
    """Lazy-init a local RiskManager for pre-flight checks.

    This is separate from the engine's RiskManager — it reads positions
    from cache and provides synchronous risk checks for API requests.
    """
    global _local_risk_manager
    if _local_risk_manager is None:
        try:
            from lib.services.engine.risk import RiskManager

            account_size = int(os.getenv("ACCOUNT_SIZE", "50000"))
            _local_risk_manager = RiskManager(account_size=account_size)
            logger.info("Local RiskManager initialised (account=$%s)", f"{account_size:,}")
        except ImportError:
            logger.warning("Could not import RiskManager — risk checks unavailable")
            return None
    return _local_risk_manager


def _sync_local_risk_manager():
    """Sync positions from cache into the local RiskManager."""
    rm = _get_local_risk_manager()
    if rm is None:
        return

    try:
        from lib.core.cache import cache_get

        raw = cache_get("positions:current")
        if not raw:
            try:
                from lib.core.cache import _cache_key

                key = _cache_key("live_positions", "current")
                raw = cache_get(key)
            except Exception:
                pass

        if raw:
            data = json.loads(raw)
            positions = data.get("positions", [])
            if positions:
                rm.sync_positions(positions)
    except Exception as exc:
        logger.debug("Position sync for local RM failed (non-fatal): %s", exc)


def _record_risk_event(
    event_type: str,
    symbol: str = "",
    side: str = "",
    reason: str = "",
    daily_pnl: float = 0.0,
    open_trades: int = 0,
):
    """Record a risk event in the in-memory audit trail."""
    event = {
        "timestamp": datetime.now(tz=_EST).isoformat(),
        "event_type": event_type,
        "symbol": symbol,
        "side": side,
        "reason": reason,
        "daily_pnl": daily_pnl,
        "open_trades": open_trades,
    }
    _risk_events.append(event)
    # Trim to max size
    while len(_risk_events) > _MAX_RISK_EVENTS:
        _risk_events.pop(0)

    # Also publish to Redis for persistence
    try:
        from lib.core.cache import REDIS_AVAILABLE, _r

        if REDIS_AVAILABLE and _r is not None:
            _r.lpush("engine:risk_events", json.dumps(event))
            _r.ltrim("engine:risk_events", 0, _MAX_RISK_EVENTS - 1)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("/status", response_model=RiskStatusResponse)
def get_risk_status():
    """Get the current risk engine status.

    Reads from Redis first (engine's published state), falls back to
    a local RiskManager if Redis is unavailable.
    """
    # Try Redis first (engine publishes here)
    try:
        from lib.core.cache import cache_get

        raw = cache_get("engine:risk_status")
        if raw:
            data = json.loads(raw)
            data["source"] = "redis"
            return RiskStatusResponse(**data)
    except Exception as exc:
        logger.debug("Redis risk status unavailable: %s", exc)

    # Fall back to local RiskManager
    rm = _get_local_risk_manager()
    if rm is not None:
        _sync_local_risk_manager()
        status = rm.get_status()
        status["source"] = "local"
        return RiskStatusResponse(**status)

    # No risk data available
    return RiskStatusResponse(
        source="unavailable",
        last_check=datetime.now(tz=_EST).isoformat(),
    )


@router.post("/check", response_model=RiskCheckResponse)
def check_trade_risk(req: RiskCheckRequest):
    """Pre-flight risk check for a proposed trade.

    Evaluates all risk rules against the proposed trade without
    actually creating it.  Use this before sending an order to
    the broker to verify the trade won't violate risk limits.

    Returns allowed=True if the trade passes all checks, or
    allowed=False with a reason string explaining which rule(s) blocked it.
    """
    rm = _get_local_risk_manager()
    if rm is None:
        raise HTTPException(
            status_code=503,
            detail="Risk engine not available — cannot perform pre-flight check",
        )

    # Sync latest positions before checking
    _sync_local_risk_manager()

    now = datetime.now(tz=_EST)

    allowed, reason = rm.can_enter_trade(
        symbol=req.symbol,
        side=req.side.upper(),
        size=req.size,
        risk_per_contract=req.risk_per_contract,
        is_stack=req.is_stack,
        wave_ratio=req.wave_ratio,
        unrealized_r=req.unrealized_r,
    )

    total_risk = req.risk_per_contract * req.size if req.risk_per_contract > 0 else 0.0

    # Record the check as a risk event
    if not allowed:
        _record_risk_event(
            event_type="block",
            symbol=req.symbol,
            side=req.side.upper(),
            reason=reason,
            daily_pnl=rm.daily_pnl,
            open_trades=rm.open_trade_count,
        )
        logger.warning(
            "Risk check BLOCKED: %s %s %dx — %s",
            req.side,
            req.symbol,
            req.size,
            reason,
        )

        # Send alert for blocked trade
        try:
            from lib.core.alerts import send_risk_alert

            send_risk_alert(
                title=f"🚫 Trade Blocked: {req.side} {req.symbol}",
                message=reason,
            )
        except Exception:
            pass
    else:
        logger.info(
            "Risk check PASSED: %s %s %dx (risk $%.2f)",
            req.side,
            req.symbol,
            req.size,
            total_risk,
        )

    return RiskCheckResponse(
        allowed=allowed,
        reason=reason,
        symbol=req.symbol,
        side=req.side.upper(),
        size=req.size,
        total_risk=round(total_risk, 2),
        max_risk_per_trade=round(rm.max_risk_per_trade, 2),
        daily_pnl=round(rm.daily_pnl, 2),
        open_trade_count=rm.open_trade_count,
        checked_at=now.isoformat(),
    )


@router.get("/history")
def get_risk_history(limit: int = 50):
    """Get recent risk events (blocks, warnings, clears).

    Returns the most recent risk events from the in-memory audit trail,
    falling back to Redis if the in-memory list is empty.
    """
    events = list(reversed(_risk_events[-limit:]))

    # If in-memory is empty, try Redis
    if not events:
        try:
            from lib.core.cache import REDIS_AVAILABLE, _r

            if REDIS_AVAILABLE and _r is not None:
                raw_list = _r.lrange("engine:risk_events", 0, limit - 1)  # type: ignore[union-attr]
                if raw_list and isinstance(raw_list, list):
                    events = [json.loads(item) for item in raw_list]
        except Exception:
            pass

    return {
        "events": events,
        "count": len(events),
        "limit": limit,
        "timestamp": datetime.now(tz=_EST).isoformat(),
    }


# ---------------------------------------------------------------------------
# Helpers for other routers (positions, trades) to call
# ---------------------------------------------------------------------------


def evaluate_position_risk(positions: list[dict[str, Any]]) -> dict[str, Any]:
    """Evaluate risk for a set of positions (called from positions router).

    Returns a dict with:
      - can_trade: bool
      - block_reason: str
      - warnings: list[str]
      - risk_pct: float
    """
    rm = _get_local_risk_manager()
    if rm is None:
        return {
            "can_trade": True,
            "block_reason": "",
            "warnings": [],
            "risk_pct": 0.0,
        }

    # Sync the positions
    rm.sync_positions(positions)

    status = rm.get_status()
    warnings = []

    if status.get("is_overnight_warning"):
        warnings.append("Session ending — close or protect open positions")
    if status.get("consecutive_losses", 0) >= 2:
        warnings.append(f"{status['consecutive_losses']} consecutive losses — consider pausing")
    if status.get("risk_pct_of_account", 0) > 3.0:
        warnings.append(f"Total exposure {status['risk_pct_of_account']:.1f}% of account")

    return {
        "can_trade": status["can_trade"],
        "block_reason": status["block_reason"],
        "warnings": warnings,
        "risk_pct": status.get("risk_pct_of_account", 0.0),
        "daily_pnl": status.get("daily_pnl", 0.0),
        "open_trade_count": status.get("open_trade_count", 0),
    }


def check_trade_entry_risk(
    symbol: str,
    side: str,
    size: int = 1,
    risk_per_contract: float = 0.0,
) -> tuple[bool, str, dict[str, Any]]:
    """Quick risk check for the trades router.

    Returns:
        (allowed, reason, details_dict)
    """
    rm = _get_local_risk_manager()
    if rm is None:
        # No risk manager — allow but flag it
        return True, "", {"risk_available": False}

    _sync_local_risk_manager()

    allowed, reason = rm.can_enter_trade(
        symbol=symbol,
        side=side.upper(),
        size=size,
        risk_per_contract=risk_per_contract,
    )

    details = {
        "risk_available": True,
        "allowed": allowed,
        "reason": reason,
        "daily_pnl": round(rm.daily_pnl, 2),
        "open_trade_count": rm.open_trade_count,
        "max_risk_per_trade": round(rm.max_risk_per_trade, 2),
    }

    if not allowed:
        _record_risk_event(
            event_type="block",
            symbol=symbol,
            side=side.upper(),
            reason=reason,
            daily_pnl=rm.daily_pnl,
            open_trades=rm.open_trade_count,
        )

    return allowed, reason, details
