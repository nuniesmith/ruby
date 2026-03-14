"""
Swing Actions API Router — HTMX endpoints for swing trade management.

Provides mutation endpoints that the dashboard swing cards call via HTMX
to accept, ignore, close, or adjust active swing trades.  Each endpoint
returns an HTML fragment that HTMX swaps into the swing card in-place,
giving instant visual feedback without a full page reload.

Endpoints:
    POST /api/swing/accept/{asset_name}   — Accept a pending swing signal → create SwingState
    POST /api/swing/ignore/{asset_name}   — Dismiss a pending swing signal
    POST /api/swing/close/{asset_name}    — Close an active swing position
    POST /api/swing/stop-to-be/{asset_name} — Move stop-loss to breakeven
    POST /api/swing/update-stop/{asset_name} — Manually set a new stop price
    GET  /api/swing/pending               — List pending (unacted) swing signals
    GET  /api/swing/active                — List active swing states
    GET  /api/swing/detail/{asset_name}   — Full detail for one swing state
    GET  /api/swing/history               — Recent closed swing trade history
    GET  /api/swing/status-badge/{asset_name} — HTML badge fragment for a swing card
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

logger = logging.getLogger("api.swing_actions")

_EST = ZoneInfo("America/New_York")

router = APIRouter(tags=["Swing Actions"])


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------


class UpdateStopRequest(BaseModel):
    """Request body for manually updating a swing stop price."""

    new_stop: float = Field(..., gt=0, description="New stop-loss price")


class CloseSwingRequest(BaseModel):
    """Optional request body for closing a swing with a reason."""

    reason: str = Field("manual", description="Reason for closing")


# ---------------------------------------------------------------------------
# HTML fragment helpers
# ---------------------------------------------------------------------------


def _success_toast(title: str, detail: str, color: str = "green") -> str:
    """Render a small success toast that auto-fades."""
    return (
        f'<div class="swing-action-toast" style="'
        f"padding:6px 10px;border-radius:6px;font-size:11px;font-weight:600;"
        f"background:rgba({'34,197,94' if color == 'green' else '250,204,21' if color == 'yellow' else '239,68,68'},0.15);"
        f"color:{'#4ade80' if color == 'green' else '#facc15' if color == 'yellow' else '#f87171'};"
        f"border:1px solid rgba({'34,197,94' if color == 'green' else '250,204,21' if color == 'yellow' else '239,68,68'},0.3);"
        f'animation:fadeIn .2s ease">'
        f"<div>{title}</div>"
        f'<div style="font-size:9px;font-weight:400;color:var(--text-muted);margin-top:2px">{detail}</div>'
        f"</div>"
    )


def _error_toast(message: str) -> str:
    """Render an error toast."""
    return (
        '<div class="swing-action-toast" style="'
        "padding:6px 10px;border-radius:6px;font-size:11px;font-weight:600;"
        "background:rgba(239,68,68,0.15);color:#f87171;"
        'border:1px solid rgba(239,68,68,0.3)">'
        f"<div>⚠️ Error</div>"
        f'<div style="font-size:9px;font-weight:400;color:var(--text-muted);margin-top:2px">{message}</div>'
        "</div>"
    )


def _render_active_state_badge(state: dict[str, Any]) -> str:
    """Render a compact status badge for an active swing state."""
    phase = state.get("phase", "unknown")
    direction = state.get("direction", "")
    entry = state.get("entry_price", 0)
    stop = state.get("current_stop", 0)
    remaining = state.get("remaining_size", 0)
    tp1 = state.get("tp1", 0)
    tp2 = state.get("tp2", 0)

    # Phase colors
    phase_colors = {
        "active": ("#4ade80", "rgba(34,197,94,0.12)"),
        "tp1_hit": ("#facc15", "rgba(250,204,21,0.12)"),
        "trailing": ("#60a5fa", "rgba(96,165,250,0.12)"),
        "closed": ("#71717a", "rgba(113,113,122,0.12)"),
        "watching": ("#c084fc", "rgba(192,132,252,0.12)"),
        "entry_ready": ("#fb923c", "rgba(251,146,60,0.12)"),
    }
    fg, bg = phase_colors.get(phase, ("#71717a", "rgba(113,113,122,0.12)"))

    dir_emoji = "🟢" if direction == "LONG" else "🔴" if direction == "SHORT" else "⚪"

    return (
        f'<div style="padding:4px 8px;border-radius:6px;background:{bg};'
        f'border:1px solid {fg}33;font-size:10px">'
        f'<div style="display:flex;align-items:center;gap:6px;margin-bottom:2px">'
        f'<span style="font-weight:700;color:{fg};text-transform:uppercase">{phase}</span>'
        f"<span>{dir_emoji} {direction}</span>"
        f'<span style="color:var(--text-muted)">{remaining} micros</span>'
        f"</div>"
        f'<div style="display:flex;gap:8px;font-family:monospace;font-size:9px;color:var(--text-secondary)">'
        f"<span>Entry: {entry:,.4f}</span>"
        f'<span style="color:#f87171">SL: {stop:,.4f}</span>'
        f'<span style="color:#4ade80">TP1: {tp1:,.4f}</span>'
        f'<span style="color:#22c55e">TP2: {tp2:,.4f}</span>'
        f"</div>"
        f"</div>"
    )


def _render_swing_action_buttons(
    asset_name: str,
    has_active_state: bool,
    has_pending_signal: bool,
    phase: str = "",
) -> str:
    """Render the HTMX action buttons for a swing card.

    Button set depends on current state:
      - Pending signal: Accept / Ignore
      - Active state: Close / Move Stop to BE
      - No state: (empty)
    """
    asset_slug = asset_name.replace(" ", "_").replace("&", "").lower()
    target_id = f"swing-actions-{asset_slug}"

    btn_style = (
        "padding:4px 10px;border-radius:5px;font-size:10px;font-weight:600;"
        "cursor:pointer;border:1px solid;transition:all .12s;"
        "font-family:inherit;display:inline-flex;align-items:center;gap:3px"
    )

    buttons = []

    if has_pending_signal and not has_active_state:
        # Accept button
        buttons.append(
            f'<button hx-post="/api/swing/accept/{asset_name}" '
            f'hx-target="#{target_id}" hx-swap="innerHTML" '
            f'hx-confirm="Accept swing signal for {asset_name}?" '
            f'style="{btn_style};background:rgba(34,197,94,0.15);color:#4ade80;'
            f'border-color:rgba(34,197,94,0.3)" '
            f"onmouseover=\"this.style.background='rgba(34,197,94,0.25)'\" "
            f"onmouseout=\"this.style.background='rgba(34,197,94,0.15)'\">"
            f"✅ Accept</button>"
        )
        # Ignore button
        buttons.append(
            f'<button hx-post="/api/swing/ignore/{asset_name}" '
            f'hx-target="#{target_id}" hx-swap="innerHTML" '
            f'style="{btn_style};background:rgba(113,113,122,0.15);color:var(--text-muted);'
            f'border-color:rgba(113,113,122,0.3)" '
            f"onmouseover=\"this.style.background='rgba(113,113,122,0.25)'\" "
            f"onmouseout=\"this.style.background='rgba(113,113,122,0.15)'\">"
            f"🚫 Ignore</button>"
        )

    if has_active_state and phase not in ("closed", ""):
        # Close Position button
        buttons.append(
            f'<button hx-post="/api/swing/close/{asset_name}" '
            f'hx-target="#{target_id}" hx-swap="innerHTML" '
            f'hx-confirm="Close swing position for {asset_name}?" '
            f'style="{btn_style};background:rgba(239,68,68,0.15);color:#f87171;'
            f'border-color:rgba(239,68,68,0.3)" '
            f"onmouseover=\"this.style.background='rgba(239,68,68,0.25)'\" "
            f"onmouseout=\"this.style.background='rgba(239,68,68,0.15)'\">"
            f"✖ Close</button>"
        )
        # Move Stop to BE button (only for active/tp1_hit/trailing)
        if phase in ("active", "tp1_hit", "trailing"):
            buttons.append(
                f'<button hx-post="/api/swing/stop-to-be/{asset_name}" '
                f'hx-target="#{target_id}" hx-swap="innerHTML" '
                f'hx-confirm="Move stop to breakeven for {asset_name}?" '
                f'style="{btn_style};background:rgba(96,165,250,0.15);color:#60a5fa;'
                f'border-color:rgba(96,165,250,0.3)" '
                f"onmouseover=\"this.style.background='rgba(96,165,250,0.25)'\" "
                f"onmouseout=\"this.style.background='rgba(96,165,250,0.15)'\">"
                f"🛡️ Stop→BE</button>"
            )

    if not buttons:
        return ""

    return (
        f'<div id="{target_id}" '
        f'style="display:flex;flex-wrap:wrap;gap:4px;margin-top:6px;padding-top:6px;'
        f'border-top:1px solid var(--border-subtle)">' + "".join(buttons) + "</div>"
    )


# ---------------------------------------------------------------------------
# POST endpoints — mutations
# ---------------------------------------------------------------------------


@router.post("/api/swing/accept/{asset_name}")
def accept_signal(asset_name: str) -> HTMLResponse:
    """Accept a pending swing signal — create an active SwingState.

    Returns an HTML fragment with a success toast + updated action buttons
    that HTMX swaps into the swing card's action area.
    """
    try:
        from lib.services.engine.swing import accept_swing_signal

        result = accept_swing_signal(asset_name)

        direction = result.get("direction", "")
        entry = result.get("entry_price", 0)
        size = result.get("position_size", 0)

        html = _success_toast(
            f"✅ Signal Accepted — {direction}",
            f"Entry: {entry:,.4f} | {size} micros | TP1→TP2 bracket active",
            color="green",
        )
        # Append updated buttons (now showing Close / Stop→BE)
        html += _render_swing_action_buttons(
            asset_name,
            has_active_state=True,
            has_pending_signal=False,
            phase="active",
        )
        return HTMLResponse(content=html)

    except ValueError as exc:
        return HTMLResponse(content=_error_toast(str(exc)), status_code=200)
    except Exception as exc:
        logger.error("Failed to accept swing signal for %s: %s", asset_name, exc, exc_info=True)
        return HTMLResponse(content=_error_toast(f"Internal error: {exc}"), status_code=200)


@router.post("/api/swing/ignore/{asset_name}")
def ignore_signal(asset_name: str) -> HTMLResponse:
    """Dismiss a pending swing signal.

    Returns an HTML fragment confirming the signal was ignored.
    """
    try:
        from lib.services.engine.swing import ignore_swing_signal

        ignore_swing_signal(asset_name)

        html = _success_toast(
            "🚫 Signal Ignored",
            f"{asset_name} — will re-evaluate on next scan",
            color="yellow",
        )
        return HTMLResponse(content=html)

    except Exception as exc:
        logger.error("Failed to ignore swing signal for %s: %s", asset_name, exc, exc_info=True)
        return HTMLResponse(content=_error_toast(f"Internal error: {exc}"), status_code=200)


@router.post("/api/swing/close/{asset_name}")
def close_position(asset_name: str) -> HTMLResponse:
    """Close an active swing position manually.

    Returns an HTML fragment confirming the closure.
    """
    try:
        from lib.services.engine.swing import close_swing_position

        result = close_swing_position(asset_name, reason="manual_dashboard")

        prev_phase = result.get("previous_phase", "unknown")
        direction = result.get("direction", "")
        entry = result.get("entry_price", 0)

        html = _success_toast(
            f"🏁 Position Closed — {direction}",
            f"Entry was {entry:,.4f} | Was in {prev_phase} phase",
            color="red",
        )
        return HTMLResponse(content=html)

    except ValueError as exc:
        return HTMLResponse(content=_error_toast(str(exc)), status_code=200)
    except Exception as exc:
        logger.error("Failed to close swing for %s: %s", asset_name, exc, exc_info=True)
        return HTMLResponse(content=_error_toast(f"Internal error: {exc}"), status_code=200)


@router.post("/api/swing/stop-to-be/{asset_name}")
def stop_to_breakeven(asset_name: str) -> HTMLResponse:
    """Move the stop-loss to breakeven (entry price).

    Returns an HTML fragment confirming the stop move.
    """
    try:
        from lib.services.engine.swing import move_stop_to_breakeven

        result = move_stop_to_breakeven(asset_name)

        old_stop = result.get("old_stop", 0)
        new_stop = result.get("new_stop", 0)

        html = _success_toast(
            "🛡️ Stop → Breakeven",
            f"{old_stop:,.4f} → {new_stop:,.4f} (entry price)",
            color="green",
        )
        # Keep the action buttons visible
        html += _render_swing_action_buttons(
            asset_name,
            has_active_state=True,
            has_pending_signal=False,
            phase=result.get("phase", "active"),
        )
        return HTMLResponse(content=html)

    except ValueError as exc:
        return HTMLResponse(content=_error_toast(str(exc)), status_code=200)
    except Exception as exc:
        logger.error("Failed to move stop to BE for %s: %s", asset_name, exc, exc_info=True)
        return HTMLResponse(content=_error_toast(f"Internal error: {exc}"), status_code=200)


@router.post("/api/swing/update-stop/{asset_name}")
def update_stop(asset_name: str, req: UpdateStopRequest) -> HTMLResponse:
    """Manually set a new stop-loss price.

    Returns an HTML fragment confirming the stop update.
    """
    try:
        from lib.services.engine.swing import update_swing_stop

        result = update_swing_stop(asset_name, req.new_stop)

        old_stop = result.get("old_stop", 0)
        new_stop = result.get("new_stop", 0)

        html = _success_toast(
            "🎯 Stop Updated",
            f"{old_stop:,.4f} → {new_stop:,.4f}",
            color="green",
        )
        html += _render_swing_action_buttons(
            asset_name,
            has_active_state=True,
            has_pending_signal=False,
            phase=result.get("phase", "active"),
        )
        return HTMLResponse(content=html)

    except ValueError as exc:
        return HTMLResponse(content=_error_toast(str(exc)), status_code=200)
    except Exception as exc:
        logger.error("Failed to update stop for %s: %s", asset_name, exc, exc_info=True)
        return HTMLResponse(content=_error_toast(f"Internal error: {exc}"), status_code=200)


# ---------------------------------------------------------------------------
# GET endpoints — read-only queries
# ---------------------------------------------------------------------------


@router.get("/api/swing/pending")
def list_pending_signals() -> JSONResponse:
    """List pending (unacted) swing signals.

    Returns JSON with signal details for each asset that has a detected
    but not yet accepted/ignored signal.
    """
    try:
        from lib.services.engine.swing import get_pending_signals

        pending = get_pending_signals()
        return JSONResponse(
            content={
                "count": len(pending),
                "signals": pending,
                "timestamp": datetime.now(tz=_EST).isoformat(),
            }
        )
    except Exception as exc:
        logger.error("Failed to get pending signals: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/api/swing/active")
def list_active_states() -> JSONResponse:
    """List all active swing states.

    Returns JSON with state details for each asset that has an active
    swing trade being managed.
    """
    try:
        from lib.services.engine.swing import get_active_swing_states

        states = get_active_swing_states()
        serialized = {}
        for name, state in states.items():
            try:
                serialized[name] = state.to_dict() if hasattr(state, "to_dict") else state
            except Exception:
                serialized[name] = {"phase": "unknown", "error": True}

        return JSONResponse(
            content={
                "count": len(serialized),
                "states": serialized,
                "timestamp": datetime.now(tz=_EST).isoformat(),
            }
        )
    except Exception as exc:
        logger.error("Failed to get active states: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/api/swing/detail/{asset_name}")
def get_detail(asset_name: str) -> JSONResponse:
    """Get full detail for a specific swing state.

    Returns JSON with the full state dict including computed fields
    (risk per unit, R:R ratios).
    """
    try:
        from lib.services.engine.swing import get_swing_state_detail

        detail = get_swing_state_detail(asset_name)
        if detail is None:
            raise HTTPException(status_code=404, detail=f"No active swing state for '{asset_name}'")

        return JSONResponse(content=detail)
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to get swing detail for %s: %s", asset_name, exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/api/swing/history")
def get_history(limit: int = Query(20, ge=1, le=100)) -> JSONResponse:
    """Get recent closed swing trade history.

    Returns JSON list of archived swing states, most recent first.
    """
    try:
        from lib.services.engine.swing import get_swing_history

        history = get_swing_history(limit=limit)
        return JSONResponse(
            content={
                "count": len(history),
                "trades": history,
                "timestamp": datetime.now(tz=_EST).isoformat(),
            }
        )
    except Exception as exc:
        logger.error("Failed to get swing history: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/api/swing/status-badge/{asset_name}")
def get_status_badge(asset_name: str) -> HTMLResponse:
    """Get an HTML status badge fragment for a specific swing asset.

    Used by HTMX polling to keep the swing card status up-to-date.
    Returns the badge + action buttons HTML.
    """
    try:
        from lib.services.engine.swing import (
            get_pending_signals,
            get_swing_state_detail,
        )

        # Check if there's an active state
        detail = get_swing_state_detail(asset_name)
        if detail is not None:
            phase = detail.get("phase", "unknown")
            badge_html = _render_active_state_badge(detail)
            buttons_html = _render_swing_action_buttons(
                asset_name,
                has_active_state=True,
                has_pending_signal=False,
                phase=phase,
            )
            return HTMLResponse(content=badge_html + buttons_html)

        # Check if there's a pending signal
        pending = get_pending_signals()
        if asset_name in pending:
            sig = pending[asset_name]
            direction = sig.get("direction", "")
            confidence = sig.get("confidence", 0)
            entry_style = sig.get("entry_style", "")

            signal_html = (
                '<div style="padding:4px 8px;border-radius:6px;'
                "background:rgba(251,146,60,0.1);border:1px solid rgba(251,146,60,0.2);"
                'font-size:10px">'
                '<div style="display:flex;align-items:center;gap:6px">'
                '<span style="font-weight:700;color:#fb923c">⚡ SIGNAL DETECTED</span>'
                f'<span style="color:var(--text-secondary)">{direction} · {entry_style}</span>'
                f'<span style="color:var(--text-muted)">{confidence:.0%} conf</span>'
                "</div>"
                "</div>"
            )
            buttons_html = _render_swing_action_buttons(
                asset_name,
                has_active_state=False,
                has_pending_signal=True,
            )
            return HTMLResponse(content=signal_html + buttons_html)

        # No state, no signal
        return HTMLResponse(
            content=(
                '<div style="font-size:10px;color:var(--text-faint);padding:4px 0">No active signal — scanning...</div>'
            )
        )

    except Exception as exc:
        logger.error("Failed to get status badge for %s: %s", asset_name, exc)
        return HTMLResponse(content=_error_toast(f"Error: {exc}"), status_code=200)


# ---------------------------------------------------------------------------
# Exported helper for dashboard rendering
# ---------------------------------------------------------------------------


def render_swing_action_buttons_for_card(
    asset_name: str,
    active_states: dict[str, Any] | None = None,
    pending_signals: dict[str, Any] | None = None,
) -> str:
    """Render swing action buttons HTML for embedding in a swing card.

    Called by the dashboard renderer (_render_swing_card) to inject
    action buttons into the card HTML.

    Args:
        asset_name: The asset name.
        active_states: Current active swing states dict (optional, will fetch if None).
        pending_signals: Current pending signals dict (optional, will fetch if None).

    Returns:
        HTML string with the action buttons, or empty string if no actions available.
    """
    has_active = False
    phase = ""
    has_pending = False

    if active_states is not None:
        state = active_states.get(asset_name)
        if state is not None:
            has_active = True
            if hasattr(state, "phase"):
                phase = state.phase.value
            elif isinstance(state, dict):
                phase = state.get("phase", "")
    else:
        try:
            from lib.services.engine.swing import get_active_swing_states

            states = get_active_swing_states()
            state = states.get(asset_name)
            if state is not None:
                has_active = True
                phase = state.phase.value if hasattr(state, "phase") else ""
        except Exception:
            pass

    if pending_signals is not None:
        has_pending = asset_name in pending_signals
    else:
        try:
            from lib.services.engine.swing import get_pending_signals

            pending = get_pending_signals()
            has_pending = asset_name in pending
        except Exception:
            pass

    if phase == "closed":
        has_active = False

    return _render_swing_action_buttons(
        asset_name,
        has_active_state=has_active,
        has_pending_signal=has_pending,
        phase=phase,
    )
