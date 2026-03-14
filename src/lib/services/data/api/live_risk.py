"""
Live Risk API — Real-Time Risk State Endpoint
==============================================
Phase 5B: Serves the unified LiveRiskState to the dashboard.
Phase 5E: Provides the HTML for the persistent risk dashboard strip.

Endpoints:
    GET  /api/live-risk          — JSON LiveRiskState payload
    GET  /api/live-risk/html     — HTML risk strip (HTMX partial)
    GET  /api/live-risk/summary  — Compact summary for TradingView
    POST /api/live-risk/refresh  — Force immediate recomputation

The risk strip is a persistent horizontal bar at the top of the trading
dashboard that shows: Daily P&L, Open Positions, Risk Exposure %,
Margin Used/Available, Consecutive Losses, Session Time Remaining.

Color-coded: green (healthy) → yellow (approaching limits) → red (blocked).
Updates via SSE channel `dashboard:live_risk` — 1-2 second latency.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from fastapi import APIRouter
from fastapi.responses import HTMLResponse, JSONResponse

logger = logging.getLogger("api.live_risk")

router = APIRouter(tags=["live-risk"])

_EST = ZoneInfo("America/New_York")

# Module-level reference to the LiveRiskPublisher (set by engine startup)
_publisher = None


def set_publisher(publisher) -> None:
    """Set the LiveRiskPublisher reference (called during engine boot)."""
    global _publisher
    _publisher = publisher


def _get_live_risk_state() -> dict[str, Any] | None:
    """Get the latest LiveRiskState, trying multiple sources."""
    # 1. Try the publisher's cached state
    if _publisher is not None:
        last = _publisher.last_state
        if last is not None:
            return last.to_dict()

    # 2. Try loading from Redis
    try:
        from lib.services.engine.live_risk import load_from_redis

        state = load_from_redis()
        if state is not None:
            return state.to_dict()
    except ImportError:
        pass

    # 3. Try building from Redis risk_status key (legacy fallback)
    try:
        from lib.core.cache import cache_get

        raw = cache_get("engine:risk_status")
        if raw:
            data = json.loads(raw)
            return _legacy_risk_to_live_risk(data)
    except Exception:
        pass

    return None


def _legacy_risk_to_live_risk(data: dict[str, Any]) -> dict[str, Any]:
    """Convert legacy RiskManager.get_status() dict to LiveRiskState shape."""
    now = datetime.now(tz=_EST)

    # Determine health
    can_trade = data.get("can_trade", True)
    consecutive = data.get("consecutive_losses", 0)
    daily_pnl = data.get("daily_pnl", 0.0)
    max_daily = data.get("max_daily_loss", -1500.0)

    health = "green"
    health_reason = "Ready to trade"

    if not can_trade:
        health = "red"
        health_reason = data.get("block_reason", "Trading blocked")
    elif consecutive >= 3:
        health = "red"
        health_reason = f"{consecutive} consecutive losses"
    elif data.get("is_max_trades_reached", False):
        health = "yellow"
        health_reason = "Max positions reached"
    elif data.get("is_past_entry_cutoff", False):
        health = "yellow"
        health_reason = "Past entry cutoff"
    elif consecutive >= 2:
        health = "yellow"
        health_reason = f"{consecutive} consecutive losses"
    elif max_daily < 0 and daily_pnl < 0 and abs(daily_pnl) / abs(max_daily) >= 0.7:
        health = "yellow"
        health_reason = "Approaching daily loss limit"
    elif daily_pnl > 0:
        health_reason = f"Healthy — +${daily_pnl:,.0f} today"

    account_size = data.get("account_size", 150_000)
    max_risk = data.get("max_risk_per_trade", account_size * 0.0075)
    max_open = data.get("max_open_trades", 2)
    open_count = data.get("open_trade_count", 0)
    remaining_slots = max(0, max_open - open_count)

    # Build positions from legacy open_positions dict
    positions = []
    open_positions = data.get("open_positions", {})
    if isinstance(open_positions, dict):
        for sym, pos_info in open_positions.items():
            positions.append(
                {
                    "symbol": sym,
                    "asset_name": sym,
                    "side": pos_info.get("side", "?"),
                    "quantity": pos_info.get("quantity", 0),
                    "entry_price": pos_info.get("entry_price", 0),
                    "current_price": 0,
                    "stop_price": 0,
                    "unrealized_pnl": pos_info.get("unrealized_pnl", 0),
                    "r_multiple": 0,
                    "bracket_phase": "INITIAL",
                    "hold_duration_seconds": 0,
                    "risk_dollars": pos_info.get("risk_dollars", 0),
                    "margin_used": 0,
                    "source": "engine",
                }
            )

    total_unrealized = sum(p.get("unrealized_pnl", 0) for p in positions)

    return {
        "account_size": account_size,
        "max_risk_per_trade": round(max_risk, 2),
        "max_daily_loss": max_daily,
        "max_open_trades": max_open,
        "risk_pct_per_trade": data.get("rules", {}).get("risk_pct_per_trade", 0.0075),
        "daily_pnl": round(daily_pnl, 2),
        "daily_trade_count": data.get("daily_trade_count", 0),
        "consecutive_losses": consecutive,
        "can_trade": can_trade,
        "block_reason": data.get("block_reason", ""),
        "is_past_entry_cutoff": data.get("is_past_entry_cutoff", False),
        "is_daily_loss_exceeded": data.get("is_daily_loss_exceeded", False),
        "is_max_trades_reached": data.get("is_max_trades_reached", False),
        "is_overnight_warning": data.get("is_overnight_warning", False),
        "trading_date": data.get("trading_date", ""),
        "last_trade_time": data.get("last_trade", ""),
        "positions": positions,
        "open_position_count": open_count,
        "total_unrealized_pnl": round(total_unrealized, 2),
        "total_realized_pnl": round(daily_pnl, 2),
        "total_pnl": round(daily_pnl + total_unrealized, 2),
        "total_margin_used": 0,
        "margin_remaining": account_size,
        "total_risk_exposure": data.get("total_risk_exposure", 0),
        "risk_pct_of_account": data.get("risk_pct_of_account", 0),
        "remaining_risk_budget": round(max_risk * remaining_slots, 2),
        "remaining_trade_slots": remaining_slots,
        "session_time_remaining": "",
        "session_active": True,
        "computed_at": now.isoformat(),
        "computed_ts": time.time(),
        "health": health,
        "health_reason": health_reason,
    }


# ═══════════════════════════════════════════════════════════════════════════
# JSON endpoints
# ═══════════════════════════════════════════════════════════════════════════


@router.get("/api/live-risk")
async def get_live_risk():
    """Get the current LiveRiskState as JSON.

    This is the primary endpoint for dashboard HTMX polling and
    SSE catch-up reads.
    """
    state = _get_live_risk_state()
    if state is None:
        return JSONResponse(
            {
                "status": "unavailable",
                "message": "Live risk state not yet computed",
                "health": "yellow",
                "health_reason": "Awaiting first computation",
                "can_trade": True,
                "daily_pnl": 0.0,
                "open_position_count": 0,
                "remaining_trade_slots": 2,
            }
        )
    return JSONResponse(state)


@router.get("/api/live-risk/summary")
async def get_live_risk_summary():
    """Compact summary for TradingView or lightweight consumers."""
    state = _get_live_risk_state()
    if state is None:
        return JSONResponse({"status": "unavailable"})

    return JSONResponse(
        {
            "daily_pnl": state.get("daily_pnl", 0),
            "total_pnl": state.get("total_pnl", 0),
            "open_positions": state.get("open_position_count", 0),
            "max_positions": state.get("max_open_trades", 2),
            "consecutive_losses": state.get("consecutive_losses", 0),
            "can_trade": state.get("can_trade", True),
            "health": state.get("health", "green"),
            "remaining_budget": state.get("remaining_risk_budget", 0),
            "session_remaining": state.get("session_time_remaining", ""),
        }
    )


@router.post("/api/live-risk/refresh")
async def refresh_live_risk():
    """Force an immediate recomputation of LiveRiskState."""
    if _publisher is not None:
        try:
            state = _publisher.force_publish()
            return JSONResponse(
                {
                    "status": "ok",
                    "health": state.health,
                    "computed_at": state.computed_at,
                }
            )
        except Exception as exc:
            logger.error("Force refresh failed: %s", exc)
            return JSONResponse(
                {"status": "error", "message": str(exc)},
                status_code=500,
            )

    return JSONResponse(
        {"status": "error", "message": "LiveRiskPublisher not initialized"},
        status_code=503,
    )


# ═══════════════════════════════════════════════════════════════════════════
# HTML endpoint — Risk Dashboard Strip (Phase 5E)
# ═══════════════════════════════════════════════════════════════════════════


def _health_colors(health: str) -> tuple[str, str, str]:
    """Return (bg_color, text_color, border_color) for health status."""
    if health == "red":
        return "#2d0a0a", "#ff4444", "#ff4444"
    if health == "yellow":
        return "#2d2a0a", "#ffaa00", "#ffaa00"
    return "#0a2d0a", "#44ff88", "#44ff88"


def _format_pnl(pnl: float) -> str:
    """Format P&L with sign and color hint."""
    if pnl >= 0:
        return f"+${pnl:,.0f}"
    return f"-${abs(pnl):,.0f}"


def _format_pnl_pct(pnl: float, account_size: float) -> str:
    """Format P&L as percentage of account."""
    if account_size <= 0:
        return "0.0%"
    pct = (pnl / account_size) * 100
    sign = "+" if pct >= 0 else ""
    return f"{sign}{pct:.1f}%"


@router.get("/api/live-risk/html")
async def get_live_risk_html():
    """HTML risk strip for HTMX dashboard injection.

    This is the persistent horizontal bar at the top of the trading
    dashboard.  It shows key risk metrics at a glance and is color-coded
    by health status.

    Refreshed via:
      - HTMX hx-get="/api/live-risk/html" hx-trigger="every 5s"
      - Or SSE swap on dashboard:live_risk channel
    """
    state = _get_live_risk_state()

    if state is None:
        return HTMLResponse(
            '<div id="risk-strip" class="risk-strip risk-yellow" '
            'style="padding:8px 16px;text-align:center;font-size:13px;'
            'background:#2d2a0a;color:#ffaa00;border-bottom:2px solid #ffaa00;">'
            "⏳ Live risk data loading..."
            "</div>"
        )

    health = state.get("health", "green")
    bg, text_col, border = _health_colors(health)
    can_trade = state.get("can_trade", True)
    block_reason = state.get("block_reason", "")
    daily_pnl = state.get("daily_pnl", 0.0)
    total_pnl = state.get("total_pnl", 0.0)
    account_size = state.get("account_size", 150_000)
    open_count = state.get("open_position_count", 0)
    max_open = state.get("max_open_trades", 2)
    consecutive = state.get("consecutive_losses", 0)
    risk_pct = state.get("risk_pct_of_account", 0.0)
    remaining_budget = state.get("remaining_risk_budget", 0.0)
    remaining_slots = state.get("remaining_trade_slots", 0)
    session_remaining = state.get("session_time_remaining", "")
    state.get("total_margin_used", 0.0)
    state.get("margin_remaining", account_size)
    health_reason = state.get("health_reason", "")
    unrealized = state.get("total_unrealized_pnl", 0.0)

    pnl_color = "#44ff88" if daily_pnl >= 0 else "#ff4444"
    total_pnl_color = "#44ff88" if total_pnl >= 0 else "#ff4444"
    unreal_color = "#44ff88" if unrealized >= 0 else "#ff4444"
    consec_color = "#ff4444" if consecutive >= 3 else ("#ffaa00" if consecutive >= 2 else "#44ff88")
    pos_color = "#ff4444" if open_count >= max_open else ("#ffaa00" if open_count >= max_open - 1 else "#44ff88")

    # Build the risk blocked banner if needed
    blocked_banner = ""
    if not can_trade:
        blocked_banner = (
            f'<div style="background:#4a0000;color:#ff4444;padding:6px 16px;'
            f"text-align:center;font-weight:bold;font-size:14px;"
            f'border-bottom:2px solid #ff4444;animation:pulse_red 1.5s infinite;">'
            f"🚫 RISK BLOCKED — {_esc(block_reason)}"
            f"</div>"
        )

    # Build position pills (compact per-position indicators)
    position_pills = ""
    positions = state.get("positions", [])
    if positions:
        pills = []
        for pos in positions:
            pos_side = pos.get("side", "?")
            pos_sym = pos.get("asset_name", pos.get("symbol", "?"))
            pos_pnl = pos.get("unrealized_pnl", 0)
            pos_phase = pos.get("bracket_phase", "INIT")
            pos_pnl_color = "#44ff88" if pos_pnl >= 0 else "#ff4444"
            side_emoji = "🟢" if pos_side == "LONG" else "🔴"
            phase_short = {
                "INITIAL": "①",
                "TP1_HIT": "②",
                "TP2_HIT": "③",
                "TRAILING": "🏃",
                "CLOSED": "✓",
            }.get(pos_phase, "①")

            pill = (
                f'<span style="display:inline-block;background:#1a1a2e;border:1px solid #333;'
                f'border-radius:4px;padding:2px 8px;margin:0 3px;font-size:11px;white-space:nowrap;">'
                f"{side_emoji} <b>{_esc(pos_sym)}</b> "
                f'<span style="color:{pos_pnl_color}">{_format_pnl(pos_pnl)}</span> '
                f"{phase_short}"
                f"</span>"
            )
            pills.append(pill)
        position_pills = " ".join(pills)

    html = f"""
    <style>
    @keyframes pulse_red {{
        0%, 100% {{ opacity: 1.0; }}
        50% {{ opacity: 0.7; }}
    }}
    @keyframes pulse_green {{
        0%, 100% {{ box-shadow: 0 0 0 0 rgba(68,255,136,0.4); }}
        50% {{ box-shadow: 0 0 8px 2px rgba(68,255,136,0.2); }}
    }}
    .risk-strip {{
        font-family: 'SF Mono', 'Fira Code', monospace;
        font-size: 12px;
        line-height: 1.4;
    }}
    .risk-metric {{
        display: inline-block;
        padding: 0 12px;
        border-right: 1px solid #333;
        white-space: nowrap;
    }}
    .risk-metric:last-child {{
        border-right: none;
    }}
    .risk-label {{
        color: #888;
        font-size: 10px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }}
    .risk-value {{
        font-weight: bold;
        font-size: 13px;
    }}
    </style>
    {blocked_banner}
    <div id="risk-strip" class="risk-strip"
         style="background:{bg};color:{text_col};border-bottom:2px solid {border};
                padding:6px 12px;display:flex;align-items:center;justify-content:space-between;
                flex-wrap:wrap;gap:4px;"
         hx-get="/api/live-risk/html" hx-trigger="every 5s" hx-swap="outerHTML">

        <!-- Left: P&L + positions -->
        <div style="display:flex;align-items:center;gap:4px;flex-wrap:wrap;">
            <span class="risk-metric">
                <span class="risk-label">Daily P&L</span><br>
                <span class="risk-value" style="color:{pnl_color}">{_format_pnl(daily_pnl)}</span>
                <span style="color:#666;font-size:10px;">{_format_pnl_pct(daily_pnl, account_size)}</span>
            </span>
            <span class="risk-metric">
                <span class="risk-label">Unrealized</span><br>
                <span class="risk-value" style="color:{unreal_color}">{_format_pnl(unrealized)}</span>
            </span>
            <span class="risk-metric">
                <span class="risk-label">Total</span><br>
                <span class="risk-value" style="color:{total_pnl_color}">{_format_pnl(total_pnl)}</span>
            </span>
            <span class="risk-metric">
                <span class="risk-label">Positions</span><br>
                <span class="risk-value" style="color:{pos_color}">{open_count}/{max_open}</span>
            </span>
        </div>

        <!-- Center: Live position pills -->
        <div style="display:flex;align-items:center;flex-wrap:wrap;gap:2px;">
            {position_pills}
        </div>

        <!-- Right: Risk metrics -->
        <div style="display:flex;align-items:center;gap:4px;flex-wrap:wrap;">
            <span class="risk-metric">
                <span class="risk-label">Risk %</span><br>
                <span class="risk-value">{risk_pct:.1f}%</span>
            </span>
            <span class="risk-metric">
                <span class="risk-label">Budget</span><br>
                <span class="risk-value">${remaining_budget:,.0f}</span>
                <span style="color:#666;font-size:10px;">({remaining_slots} slots)</span>
            </span>
            <span class="risk-metric">
                <span class="risk-label">C.Loss</span><br>
                <span class="risk-value" style="color:{consec_color}">{consecutive}</span>
            </span>
            <span class="risk-metric">
                <span class="risk-label">Session</span><br>
                <span class="risk-value">{_esc(session_remaining) or "—"}</span>
            </span>
            <span class="risk-metric" style="border-right:none;">
                <span class="risk-label">Status</span><br>
                <span class="risk-value" style="color:{text_col};">
                    {"🟢" if health == "green" else "🟡" if health == "yellow" else "🔴"}
                    {_esc(health_reason)[:40]}
                </span>
            </span>
        </div>
    </div>
    """

    return HTMLResponse(html.strip())


# ═══════════════════════════════════════════════════════════════════════════
# Position detail HTML (for focus card overlay — Phase 5D)
# ═══════════════════════════════════════════════════════════════════════════


@router.get("/api/live-risk/position/{asset_name}/html")
async def get_position_html(asset_name: str):
    """HTML partial for a live position overlay on a focus card.

    When you're in a trade, the focus card flips from "setup" mode to
    "live position" mode.  This endpoint provides the position-specific
    HTML fragment.
    """
    state = _get_live_risk_state()
    if state is None:
        return HTMLResponse("")

    positions = state.get("positions", [])
    pos = None
    for p in positions:
        if p.get("asset_name") == asset_name or p.get("symbol") == asset_name:
            pos = p
            break

    if pos is None:
        # No position — return empty (card stays in setup mode)
        return HTMLResponse("")

    side = pos.get("side", "?")
    entry = pos.get("entry_price", 0)
    current = pos.get("current_price", 0)
    pnl = pos.get("unrealized_pnl", 0)
    r_mult = pos.get("r_multiple", 0)
    phase = pos.get("bracket_phase", "INITIAL")
    hold_secs = pos.get("hold_duration_seconds", 0)
    quantity = pos.get("quantity", 0)
    symbol = pos.get("symbol", "")

    pnl_color = "#44ff88" if pnl >= 0 else "#ff4444"
    r_color = "#44ff88" if r_mult >= 0 else "#ff4444"
    side_color = "#44ff88" if side == "LONG" else "#ff4444"
    phase_label = {
        "INITIAL": "Phase 1 — Entry",
        "TP1_HIT": "Phase 2 — Breakeven",
        "TP2_HIT": "Phase 3 — Trailing",
        "TRAILING": "Phase 3 — EMA9 Trail",
        "CLOSED": "Closed",
    }.get(phase, phase)

    # Hold duration formatting
    if hold_secs >= 3600:
        hold_str = f"{hold_secs // 3600}h {(hold_secs % 3600) // 60}m"
    elif hold_secs >= 60:
        hold_str = f"{hold_secs // 60}m {hold_secs % 60}s"
    else:
        hold_str = f"{hold_secs}s"

    # Bracket progress bar
    phases = ["ENTRY", "TP1", "TP2", "TP3"]
    phase_checks = {
        "INITIAL": 0,
        "TP1_HIT": 1,
        "TP2_HIT": 2,
        "TRAILING": 2,
        "CLOSED": 3,
    }
    completed = phase_checks.get(phase, 0)
    progress_items = []
    for i, ph in enumerate(phases):
        if i < completed:
            progress_items.append(f'<span style="color:#44ff88;">✓ {ph}</span>')
        elif i == completed:
            progress_items.append(f'<span style="color:#ffaa00;font-weight:bold;">► {ph}</span>')
        else:
            progress_items.append(f'<span style="color:#555;">{ph}</span>')
    progress_bar = " — ".join(progress_items)

    html = f"""
    <div style="background:#0a1a0a;border:1px solid {side_color};border-radius:6px;padding:10px;
                margin-top:6px;">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">
            <span style="color:{side_color};font-weight:bold;font-size:14px;">
                {"🟢" if side == "LONG" else "🔴"} {side} LIVE — {_esc(phase_label or "")}
            </span>
            <span style="color:#888;font-size:11px;">⏱ {hold_str}</span>
        </div>

        <div style="font-size:11px;color:#aaa;margin-bottom:6px;">
            {progress_bar}
        </div>

        <div style="display:grid;grid-template-columns:1fr 1fr;gap:4px;font-size:12px;">
            <div>
                <span style="color:#666;">Entry:</span>
                <span style="color:#ddd;">{entry:,.2f}</span>
            </div>
            <div>
                <span style="color:#666;">Current:</span>
                <span style="color:#ddd;">{current:,.2f}</span>
            </div>
            <div>
                <span style="color:#666;">P&L:</span>
                <span style="color:{pnl_color};font-weight:bold;">{_format_pnl(pnl)}</span>
            </div>
            <div>
                <span style="color:#666;">R-Multiple:</span>
                <span style="color:{r_color};">{r_mult:+.2f}R</span>
            </div>
            <div>
                <span style="color:#666;">Qty:</span>
                <span style="color:#ddd;">{quantity}×</span>
            </div>
            <div>
                <span style="color:#666;">Symbol:</span>
                <span style="color:#ddd;">{_esc(symbol)}</span>
            </div>
        </div>

        <div style="margin-top:8px;display:flex;gap:6px;">
            <button hx-post="/api/positions/flatten"
                    hx-vals='{{"symbol": "{_esc(symbol)}"}}'
                    hx-confirm="Close {_esc(asset_name)} position?"
                    style="background:#4a0000;color:#ff4444;border:1px solid #ff4444;
                           border-radius:4px;padding:4px 12px;font-size:11px;cursor:pointer;">
                ✕ Close Position
            </button>
        </div>
    </div>
    """

    return HTMLResponse(html.strip())


def _esc(text: str) -> str:
    """Simple HTML escaping."""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#x27;")
    )
