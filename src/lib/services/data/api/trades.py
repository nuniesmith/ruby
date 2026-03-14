"""
Trades API router — trade CRUD endpoints.

Provides endpoints for creating, closing, cancelling, and listing trades.
Includes a legacy /log_trade endpoint for backwards compatibility with
older trading scripts.

Risk enforcement:
  - POST /trades runs a pre-flight risk check via the local RiskManager
    before creating the trade.  If the risk check fails, the response
    includes ``risk_blocked=True`` and the ``risk_reason`` string.
    By default the trade is still created (the system is advisory), but
    callers can set ``enforce_risk=True`` in the request body to get a
    403 rejection instead.

Position management (live trading connection) is handled by positions.py.
Asset/account info endpoints are handled by analysis.py.
"""

import logging
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from lib.core.models import (
    _get_conn,
    cancel_trade,
    close_trade,
    create_trade,
    get_all_trades,
    get_closed_trades,
    get_open_trades,
    get_today_pnl,
)

_EST = ZoneInfo("America/New_York")
logger = logging.getLogger("api.trades")

router = APIRouter(tags=["trades"])


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class CreateTradeRequest(BaseModel):
    account_size: int = Field(
        150000,
        description="Account size: 50000, 100000, or 150000",
    )
    asset: str = Field(..., description="Asset name, e.g. 'Gold', 'E-mini S&P'")
    direction: str = Field(..., description="LONG or SHORT")
    entry: float = Field(..., description="Entry price")
    sl: float | None = Field(None, description="Stop loss price")
    tp: float | None = Field(None, description="Take profit price")
    contracts: int = Field(1, ge=1, description="Number of contracts")
    strategy: str = Field("", description="Strategy name")
    notes: str = Field("", description="Trade notes")
    enforce_risk: bool = Field(
        False,
        description=(
            "If True, the trade is rejected (HTTP 403) when risk rules block it. "
            "If False (default), the trade is created with a risk warning in the response."
        ),
    )


class CloseTradeRequest(BaseModel):
    close_price: float = Field(..., description="Exit price")


class GradeTradeRequest(BaseModel):
    grade: str = Field(..., description="Trade quality grade: A, B, C, D, or F")


class LegacyTradeRequest(BaseModel):
    """Backwards-compatible with the original /log_trade endpoint."""

    asset: str
    direction: str
    entry: float
    exit_price: float
    contracts: int = 1
    pnl: float = 0.0
    strategy: str = ""
    notes: str = ""


class TradeResponse(BaseModel):
    id: int
    created_at: str
    account_size: int
    asset: str
    direction: str
    entry: float
    sl: float | None = None
    tp: float | None = None
    contracts: int
    status: str
    close_price: float | None = None
    close_time: str | None = None
    pnl: float | None = None
    rr: float | None = None
    notes: str = ""
    strategy: str = ""
    grade: str = ""
    source: str = "manual"
    # Risk fields (populated on create only)
    risk_checked: bool = Field(False, description="Whether a risk check was performed")
    risk_blocked: bool = Field(False, description="True if risk rules would block this trade")
    risk_reason: str = Field("", description="Risk block reason (empty if allowed)")
    risk_details: dict[str, Any] | None = Field(None, description="Full risk check details")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_trade_by_id(trade_id: int) -> dict[str, Any]:
    """Look up a trade by ID from all trades, raise 404 if not found."""
    all_trades = get_all_trades()
    for t in all_trades:
        if t["id"] == trade_id:
            return t
    raise HTTPException(status_code=404, detail=f"Trade {trade_id} not found")


# ---------------------------------------------------------------------------
# Trade CRUD endpoints
# ---------------------------------------------------------------------------


@router.post("/trades", response_model=TradeResponse, status_code=201)
def api_create_trade(req: CreateTradeRequest):
    """Open a new trade.

    Performs a pre-flight risk check before creating the trade.
    If ``enforce_risk`` is True and the check fails, returns HTTP 403.
    Otherwise the trade is created with risk warning fields in the response.
    """
    # --- Pre-flight risk check ---
    risk_checked = False
    risk_blocked = False
    risk_reason = ""
    risk_details: dict[str, Any] | None = None

    try:
        from lib.services.data.api.risk import check_trade_entry_risk

        # Compute per-contract risk if stop loss is provided
        risk_per_contract = 0.0
        if req.sl and req.sl > 0 and req.entry > 0:
            try:
                from lib.core.models import CONTRACT_SPECS

                spec = CONTRACT_SPECS.get(req.asset)
                point_value: float = float(spec["point"]) if spec else 1.0
                risk_per_contract = abs(req.entry - req.sl) * point_value
            except Exception:
                pass

        allowed, reason, details = check_trade_entry_risk(
            symbol=req.asset,
            side=req.direction.upper(),
            size=req.contracts,
            risk_per_contract=risk_per_contract,
        )

        risk_checked = True
        risk_blocked = not allowed
        risk_reason = reason
        risk_details = details

        if not allowed:
            logger.warning(
                "Risk check BLOCKED trade: %s %s %dx %s — %s",
                req.direction,
                req.asset,
                req.contracts,
                req.strategy,
                reason,
            )

            if req.enforce_risk:
                raise HTTPException(
                    status_code=403,
                    detail={
                        "error": "Trade blocked by risk rules",
                        "reason": reason,
                        "risk_details": details,
                    },
                )
    except HTTPException:
        raise
    except Exception as exc:
        logger.debug("Risk check unavailable (non-fatal): %s", exc)

    # --- Create the trade ---
    trade_id = create_trade(
        account_size=req.account_size,
        asset=req.asset,
        direction=req.direction.upper(),
        entry=req.entry,
        sl=req.sl or 0.0,
        tp=req.tp or 0.0,
        contracts=req.contracts,
        strategy=req.strategy,
        notes=req.notes,
    )
    trade = _get_trade_by_id(trade_id)

    return TradeResponse(
        **trade,
        risk_checked=risk_checked,
        risk_blocked=risk_blocked,
        risk_reason=risk_reason,
        risk_details=risk_details,
    )


@router.post("/trades/{trade_id}/close", response_model=TradeResponse)
def api_close_trade(trade_id: int, req: CloseTradeRequest):
    """Close an open trade with exit price."""
    close_trade(trade_id, req.close_price)
    trade = _get_trade_by_id(trade_id)
    return TradeResponse(**trade)


@router.post("/trades/{trade_id}/cancel")
def api_cancel_trade(trade_id: int):
    """Cancel an open trade (no fill)."""
    cancel_trade(trade_id)
    trade = _get_trade_by_id(trade_id)
    return {
        "status": "cancelled",
        "trade_id": trade_id,
        "trade": trade,
    }


@router.get("/trades", response_model=list[TradeResponse])
def api_list_trades(
    status: str | None = Query(None, description="Filter: open, closed"),
    account_size: int | None = Query(None, description="Filter by account size"),
):
    """List trades, optionally filtered by status and account size."""
    if status == "open":
        trades = get_open_trades(account_size=account_size or 150_000)
    elif status == "closed":
        trades = get_closed_trades(account_size=account_size or 150_000)
    else:
        trades = get_all_trades()
        if account_size:
            trades = [t for t in trades if t.get("account_size") == account_size]

    return [TradeResponse(**t) for t in trades]


@router.get("/trades/open", response_model=list[TradeResponse])
def api_open_trades(
    account_size: int = Query(150_000, description="Account size"),
):
    """List currently open trades."""
    trades = get_open_trades(account_size=account_size)
    return [TradeResponse(**t) for t in trades]


@router.get("/trades/{trade_id}", response_model=TradeResponse)
def api_get_trade(trade_id: int):
    """Get a single trade by ID."""
    return TradeResponse(**_get_trade_by_id(trade_id))


@router.patch("/trades/{trade_id}/grade")
def api_grade_trade(trade_id: int, req: GradeTradeRequest):
    """Set or update the quality grade on a trade.

    Accepted grades: A, B, C, D, F (stored as-is; validation is advisory).
    """
    valid_grades = {"A", "B", "C", "D", "F", ""}
    grade = req.grade.strip().upper()
    if grade and grade not in valid_grades:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid grade '{grade}'. Use one of: A, B, C, D, F (or empty to clear).",
        )

    conn = _get_conn()
    try:
        result = conn.execute(
            "UPDATE trades_v2 SET grade = ? WHERE id = ?",
            (grade, trade_id),
        )
        # Check rowcount — works on both SQLite and Postgres wrappers
        rowcount = getattr(result, "rowcount", None)
        if rowcount is not None and rowcount == 0:
            conn.close()
            raise HTTPException(status_code=404, detail=f"Trade {trade_id} not found")
        conn.commit()
    except HTTPException:
        raise
    except Exception as exc:
        conn.close()
        raise HTTPException(status_code=500, detail=f"Failed to update grade: {exc}") from exc
    conn.close()

    trade = _get_trade_by_id(trade_id)
    return {
        "status": "graded",
        "trade_id": trade_id,
        "grade": grade,
        "trade": TradeResponse(**trade),
    }


@router.get("/trades/today/pnl")
def api_today_pnl(account_size: int = Query(150_000)):
    """Get today's net P&L."""
    pnl = get_today_pnl(account_size)
    return {"date": datetime.now(tz=_EST).strftime("%Y-%m-%d"), "pnl": pnl}


# ---------------------------------------------------------------------------
# Legacy endpoint (backwards compatibility)
# ---------------------------------------------------------------------------


@router.post("/log_trade")
def log_trade(req: LegacyTradeRequest):
    """Legacy: create and immediately close a trade in one call.

    Kept for backwards compatibility with older trading scripts.
    """
    trade_id = create_trade(
        account_size=150_000,
        asset=req.asset,
        direction=req.direction.upper(),
        entry=req.entry,
        sl=0.0,
        tp=0.0,
        contracts=req.contracts,
        strategy=req.strategy,
        notes=req.notes,
    )
    close_trade(trade_id, req.exit_price)
    trade = _get_trade_by_id(trade_id)
    return {
        "status": "logged",
        "trade_id": trade_id,
        "pnl": trade.get("pnl", 0),
        "trade": trade,
    }
