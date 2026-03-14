"""
Trade Executor API Routes
==========================
Endpoints for the staged trade execution system.

    POST /api/trade/engage           — Engage a trade from the plan (SCOUT entry)
    GET  /api/trade/active           — Get all active trades
    GET  /api/trade/active/{symbol}  — Get active trade for a symbol
    POST /api/trade/partial          — Take partial profit
    POST /api/trade/close/{symbol}   — Close a trade
    GET  /api/trade/status           — Overall executor status
    POST /api/trade/set-stop         — Manually set/move stop on active trade
    GET  /api/trade/history          — Recent completed trades
    POST /api/trade/tick             — Process a price tick for active trades
    GET  /api/trade/rithmic-ready    — Check if Rithmic execution is ready
"""

from __future__ import annotations

import contextlib
import logging
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

logger = logging.getLogger("api.trade_executor")

router = APIRouter(tags=["TradeExecutor"])

# ---------------------------------------------------------------------------
# Module-level executor instance (initialised lazily)
# ---------------------------------------------------------------------------

_executor = None
_copy_trader = None


def _get_executor():
    """Lazily initialise the TradeExecutor singleton."""
    global _executor
    if _executor is None:
        from lib.services.engine.trade_executor import TradeExecutor

        _executor = TradeExecutor()
    return _executor


def _get_copy_trader():
    """Get the global CopyTrader instance.

    Uses the same ``_get_copy_trader_direct()`` helper that the copy-trade
    API routes use — returns ``None`` when the CopyTrader singleton is not
    yet initialised (e.g. no Rithmic creds configured).
    """
    global _copy_trader
    if _copy_trader is not None:
        return _copy_trader
    try:
        from lib.services.data.api.copy_trade import _get_copy_trader_direct

        _copy_trader = _get_copy_trader_direct()
    except ImportError:
        pass
    return _copy_trader


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@router.post("/api/trade/engage")
async def engage_trade(request: Request) -> JSONResponse:
    """Engage a trade — place the initial scout entry.

    Body:
        symbol: str — e.g. "MGC"
        direction: str — "LONG" or "SHORT"
        entry_zone_low: float
        entry_zone_high: float
        ideal_entry: float
        invalidation_level: float
        target_1: float
        target_2: float
        target_3: float
        max_contracts: int (default 4)
        initial_size: int (default 1)
        build_levels: list[float] (optional — limit order prices)
        build_sizes: list[int] (optional — qty at each build level)
        max_risk_dollars: float (default 825)
        reason: str
        plan_confidence: float (0-100)
    """
    executor = _get_executor()
    ct = _get_copy_trader()

    body = await request.json()
    symbol = body.get("symbol", "")
    direction = body.get("direction", "")

    if not symbol or direction not in ("LONG", "SHORT"):
        return JSONResponse({"ok": False, "error": "symbol and direction (LONG/SHORT) required"}, status_code=400)

    # Check if already in a trade for this symbol
    existing = executor.get_active_trade(symbol)
    if existing:
        return JSONResponse(
            {"ok": False, "error": f"Already in an active trade for {symbol} ({existing.phase.value})"}, status_code=409
        )

    # Build Rithmic mapping
    from lib.services.engine.copy_trader import TICKER_TO_RITHMIC

    product_code = symbol  # default: assume it's already a product code
    exchange = body.get("exchange", "CME")

    # Try to resolve from ticker mapping
    ticker_key = f"{symbol}=F"
    if ticker_key in TICKER_TO_RITHMIC:
        mapping = TICKER_TO_RITHMIC[ticker_key]
        product_code = mapping["product_code"]
        exchange = mapping["exchange"]

    from lib.services.engine.trade_executor import StagedTradePlan

    plan = StagedTradePlan(
        symbol=symbol,
        product_code=product_code,
        exchange=exchange,
        direction=direction,
        entry_zone_low=float(body.get("entry_zone_low", 0)),
        entry_zone_high=float(body.get("entry_zone_high", 0)),
        ideal_entry=float(body.get("ideal_entry", 0)),
        invalidation_level=float(body.get("invalidation_level", 0)),
        target_1=float(body.get("target_1", 0)),
        target_2=float(body.get("target_2", 0)),
        target_3=float(body.get("target_3", 0)),
        max_contracts=int(body.get("max_contracts", 4)),
        initial_size=int(body.get("initial_size", 1)),
        build_levels=body.get("build_levels", []),
        build_sizes=body.get("build_sizes", [1, 1, 1]),
        max_risk_dollars=float(body.get("max_risk_dollars", 825)),
        reason=body.get("reason", ""),
        plan_confidence=float(body.get("plan_confidence", 0)),
    )

    if ct is None:
        # No copy trader available — return a simulated response
        logger.warning("CopyTrader not available — engage_trade will simulate")
        from lib.services.engine.trade_executor import ActiveTrade, TradePhase

        trade = ActiveTrade(plan=plan)
        trade.phase = TradePhase.SCOUT
        trade.log_event("⚠️ SIMULATED — CopyTrader not connected")
        trade.log_event(f"Would enter: {direction} {plan.initial_size}x {symbol} MARKET")
        executor._active_trades[plan.trade_id] = trade
        return JSONResponse(
            {
                "ok": True,
                "simulated": True,
                "trade": trade.to_dict(),
            }
        )

    trade = await executor.engage_trade(plan, ct)

    return JSONResponse(
        {
            "ok": trade.is_active,
            "trade": trade.to_dict(),
            "error": "" if trade.is_active else "Trade failed to engage — check events",
        }
    )


@router.get("/api/trade/active")
async def get_active_trades() -> JSONResponse:
    """Get all active trades."""
    executor = _get_executor()
    return JSONResponse(
        {
            "ok": True,
            "trades": [t.to_dict() for t in executor.active_trades],
            "count": executor.active_trade_count,
        }
    )


@router.get("/api/trade/active/{symbol}")
async def get_active_trade(symbol: str) -> JSONResponse:
    """Get the active trade for a symbol."""
    executor = _get_executor()
    trade = executor.get_active_trade(symbol.upper())

    if trade is None:
        return JSONResponse({"ok": False, "error": f"No active trade for {symbol}"}, status_code=404)

    return JSONResponse({"ok": True, "trade": trade.to_dict()})


@router.post("/api/trade/partial")
async def take_partial_profit(request: Request) -> JSONResponse:
    """Take partial profit on an active trade.

    Body:
        symbol: str
        qty: int (default 1)
        reason: str
    """
    executor = _get_executor()
    ct = _get_copy_trader()
    body = await request.json()

    symbol = body.get("symbol", "").upper()
    qty = int(body.get("qty", 1))
    reason = body.get("reason", "Manual partial")

    trade = executor.get_active_trade(symbol)
    if not trade:
        return JSONResponse({"ok": False, "error": f"No active trade for {symbol}"}, status_code=404)

    if ct is None:
        trade.log_event(f"⚠️ SIMULATED partial: -{qty}x @ market ({reason})")
        trade.total_contracts = max(0, trade.total_contracts - qty)
        return JSONResponse({"ok": True, "simulated": True, "trade": trade.to_dict()})

    await executor.take_partial_profit(trade, ct, qty=qty, reason=reason)
    return JSONResponse({"ok": True, "trade": trade.to_dict()})


@router.post("/api/trade/close/{symbol}")
async def close_trade(symbol: str, request: Request) -> JSONResponse:
    """Close an active trade.

    Body (optional):
        reason: str
    """
    executor = _get_executor()
    ct = _get_copy_trader()

    body: dict[str, Any] = {}
    with contextlib.suppress(Exception):
        body = await request.json()

    reason = body.get("reason", "Manual close from WebUI")

    trade = executor.get_active_trade(symbol.upper())
    if not trade:
        return JSONResponse({"ok": False, "error": f"No active trade for {symbol}"}, status_code=404)

    if ct is None:
        from lib.services.engine.trade_executor import TradePhase

        trade.phase = TradePhase.CLOSED
        trade.log_event(f"⚠️ SIMULATED close: {reason}")
        return JSONResponse({"ok": True, "simulated": True, "trade": trade.to_dict()})

    await executor.close_trade(trade, ct, reason=reason)
    return JSONResponse({"ok": True, "trade": trade.to_dict()})


@router.post("/api/trade/set-stop")
async def set_stop(request: Request) -> JSONResponse:
    """Manually set or move the stop on an active trade.

    Body:
        symbol: str
        stop_price: float
        reason: str
    """
    executor = _get_executor()
    body = await request.json()

    symbol = body.get("symbol", "").upper()
    stop_price = float(body.get("stop_price", 0))
    reason = body.get("reason", "Manual stop adjustment")

    trade = executor.get_active_trade(symbol)
    if not trade:
        return JSONResponse({"ok": False, "error": f"No active trade for {symbol}"}, status_code=404)

    old_stop = trade.current_stop_price
    trade.current_stop_price = stop_price

    from lib.services.engine.trade_executor import StopStrategy

    if stop_price > 0:
        trade.stop_strategy = StopStrategy.SAFE

    trade.log_event(f"🛑 Stop {'set' if old_stop == 0 else 'moved'}: {old_stop:.2f} → {stop_price:.2f} — {reason}")

    # Place/modify the actual stop order via CopyTrader when available
    ct = _get_copy_trader()
    rithmic_result: dict[str, Any] | None = None

    if ct is not None and stop_price > 0:
        try:
            from lib.services.engine.copy_trader import TICKER_TO_RITHMIC

            # Resolve product_code and exchange from symbol
            ticker_key = f"{symbol}=F"
            product_code = (
                trade.plan.product_code if hasattr(trade, "plan") and hasattr(trade.plan, "product_code") else symbol
            )
            exchange = trade.plan.exchange if hasattr(trade, "plan") and hasattr(trade.plan, "exchange") else "CME"

            if ticker_key in TICKER_TO_RITHMIC:
                mapping = TICKER_TO_RITHMIC[ticker_key]
                product_code = mapping["product_code"]
                exchange = mapping["exchange"]

            # Resolve the front-month contract
            resolved = await ct.resolve_front_month(ticker_key)
            if resolved is not None:
                security_code, resolved_exchange = resolved
                exchange = resolved_exchange

                entry_price = trade.avg_entry_price if hasattr(trade, "avg_entry_price") else 0.0

                rithmic_result = await ct.modify_stop_on_all(
                    security_code=security_code,
                    exchange=exchange,
                    new_stop_price=stop_price,
                    product_code=product_code,
                    entry_price=entry_price,
                    reason=reason,
                )

                if rithmic_result.get("ok"):
                    trade.log_event(
                        f"✅ Rithmic stop placed: {rithmic_result.get('stop_ticks', '?')} ticks on "
                        f"{len(rithmic_result.get('accounts_modified', []))} account(s)"
                    )
                else:
                    trade.log_event(f"⚠️ Rithmic stop failed: {rithmic_result.get('reason', 'unknown')}")
            else:
                logger.warning("set-stop: could not resolve front-month contract for %s", ticker_key)
                trade.log_event(f"⚠️ Could not resolve contract for {ticker_key} — stop updated in-memory only")

        except Exception as exc:
            logger.error("set-stop: Rithmic stop placement error: %s", exc, exc_info=True)
            trade.log_event(f"⚠️ Rithmic stop error: {exc} — stop updated in-memory only")

    note = "Stop updated in executor."
    if rithmic_result and rithmic_result.get("ok"):
        note = "Stop updated in executor and placed on Rithmic."
    elif ct is None:
        note = "Stop updated in executor. Server-side stop order requires Rithmic connection."
    elif rithmic_result:
        note = f"Stop updated in executor. Rithmic placement failed: {rithmic_result.get('reason', 'unknown')}"

    return JSONResponse(
        {
            "ok": True,
            "trade": trade.to_dict(),
            "note": note,
            "rithmic": rithmic_result,
        }
    )


@router.get("/api/trade/status")
async def executor_status() -> JSONResponse:
    """Get overall trade executor status."""
    executor = _get_executor()

    # Also get risk status
    try:
        from lib.services.engine.risk import TPT_RULES

        tpt_info = {
            "account_size": TPT_RULES.account_size,
            "profit_target": TPT_RULES.profit_target,
            "max_position_size": TPT_RULES.max_position_size,
            "daily_loss_limit": TPT_RULES.daily_loss_limit,
            "eod_trailing_drawdown": TPT_RULES.eod_trailing_drawdown,
            "daily_profit_goal_min": TPT_RULES.daily_profit_goal_min,
            "daily_profit_goal_max": TPT_RULES.daily_profit_goal_max,
        }
    except ImportError:
        tpt_info = {}

    return JSONResponse(
        {
            "ok": True,
            "executor": executor.status_summary(),
            "tpt_rules": tpt_info,
        }
    )


@router.get("/api/trade/history")
async def trade_history() -> JSONResponse:
    """Get completed trade history."""
    executor = _get_executor()
    return JSONResponse(
        {
            "ok": True,
            "completed": executor._completed_trades[-20:] if executor._completed_trades else [],
        }
    )


# ---------------------------------------------------------------------------
# Tick processing
# ---------------------------------------------------------------------------


@router.post("/api/trade/tick")
async def process_tick(request: Request) -> JSONResponse:
    """Process a price tick for all active trades.

    Body: { "symbol": str, "price": float }

    Calls ``executor.process_live_tick(symbol, price, copy_trader)`` if available,
    otherwise falls back to ``executor.on_tick(price, trade)`` for matching trades.

    Returns the list of actions taken.
    """
    executor = _get_executor()
    ct = _get_copy_trader()
    body = await request.json()

    symbol = body.get("symbol", "").upper()
    price = float(body.get("price", 0))

    if not symbol or price <= 0:
        return JSONResponse(
            {"ok": False, "error": "symbol (str) and price (positive float) required"},
            status_code=400,
        )

    # Try the preferred process_live_tick method first
    if hasattr(executor, "process_live_tick"):
        try:
            actions = await executor.process_live_tick(symbol, price, ct)
            return JSONResponse({"ok": True, "symbol": symbol, "price": price, "actions": actions})
        except Exception as exc:
            logger.error("process_live_tick error for %s @ %.4f: %s", symbol, price, exc, exc_info=True)
            return JSONResponse(
                {"ok": False, "error": f"process_live_tick failed: {exc}"},
                status_code=500,
            )

    # Fallback: call on_tick for matching active trade
    trade = executor.get_active_trade(symbol)
    if trade is None:
        return JSONResponse(
            {"ok": True, "symbol": symbol, "price": price, "actions": [], "note": "No active trade for symbol"},
        )

    try:
        actions = executor.on_tick(price, trade)
        return JSONResponse({"ok": True, "symbol": symbol, "price": price, "actions": actions})
    except Exception as exc:
        logger.error("on_tick error for %s @ %.4f: %s", symbol, price, exc, exc_info=True)
        return JSONResponse(
            {"ok": False, "error": f"on_tick failed: {exc}"},
            status_code=500,
        )


# ---------------------------------------------------------------------------
# Rithmic readiness check
# ---------------------------------------------------------------------------


@router.get("/api/trade/rithmic-ready")
async def rithmic_ready() -> JSONResponse:
    """Check if the system is ready for live Rithmic execution.

    Returns whether CopyTrader is initialised and has a connected main account.
    """
    ct = _get_copy_trader()

    if ct is None:
        return JSONResponse(
            {
                "ok": True,
                "ready": False,
                "reason": "CopyTrader not initialised",
                "main_connected": False,
            }
        )

    main_connected = ct._main is not None and ct._main.connected

    if not main_connected:
        return JSONResponse(
            {
                "ok": True,
                "ready": False,
                "reason": "Main account not connected",
                "main_connected": False,
            }
        )

    return JSONResponse(
        {
            "ok": True,
            "ready": True,
            "reason": "",
            "main_connected": True,
            "main_label": ct._main.label if ct._main else "",
        }
    )
