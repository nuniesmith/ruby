"""
Static Page Routes — Serve standalone HTML pages from ``static/``
==================================================================
Each page is a self-contained HTML file (dark theme, HTMX-powered) that
communicates with the data service API endpoints.

Pages served:
    GET /chat     → static/chat.html     (RustAssistant chat interface)
    GET /dom      → static/dom.html      (Depth of Market ladder)
    GET /journal  → static/journal.html  (Trade journal)
"""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

logger = logging.getLogger("api.static_pages")

router = APIRouter(tags=["Static Pages"])

# ---------------------------------------------------------------------------
# Path resolution — same strategy as pipeline.py / pine.py:
#   1. /app/static/ (Docker container)
#   2. Relative from this file (dev layout: src/lib/services/data/api/ → ../../../../static/)
#   3. cwd()/static/ (running from repo root)
# ---------------------------------------------------------------------------

_STATIC_CANDIDATES: list[Path] = [
    Path("/app/static"),
    Path(__file__).resolve().parent.parent.parent.parent.parent.parent / "static",
    Path(__file__).resolve().parent.parent.parent.parent.parent / "static",
    Path.cwd() / "static",
]


def _resolve_static(filename: str) -> Path | None:
    """Locate a static HTML file across Docker + local dev paths."""
    for base in _STATIC_CANDIDATES:
        candidate = base / filename
        if candidate.exists():
            return candidate
    return None


def _not_found_html(page_name: str, filename: str, emoji: str = "📄") -> str:
    """Return a styled 'not found' fallback page."""
    return (
        f"<html><body style='background:#0a0a0f;color:#94a3b8;font-family:monospace;"
        f"display:flex;align-items:center;justify-content:center;height:100vh'>"
        f"<div style='text-align:center'>"
        f"<div style='font-size:2rem;margin-bottom:1rem'>{emoji}</div>"
        f"<div>{page_name} not found — place it at <code>static/{filename}</code></div>"
        f"</div></body></html>"
    )


# ---------------------------------------------------------------------------
# GET /chat — RustAssistant Chat Interface
# ---------------------------------------------------------------------------


@router.get("/chat", response_class=HTMLResponse)
async def chat_page() -> HTMLResponse:
    """Serve the RustAssistant chat page.

    Connects to ``POST /api/chat``, ``GET /sse/chat``, and
    ``GET /api/chat/history`` for real-time AI chat with market context.
    """
    path = _resolve_static("chat.html")
    if path:
        logger.info("Serving chat.html from %s", path)
        return HTMLResponse(content=path.read_text(), headers={"Cache-Control": "no-cache"})

    logger.warning("chat.html not found in any candidate path")
    return HTMLResponse(content=_not_found_html("Chat", "chat.html", "💬"), status_code=200)


# ---------------------------------------------------------------------------
# GET /dom — Depth of Market Ladder
# ---------------------------------------------------------------------------


@router.get("/dom", response_class=HTMLResponse)
async def dom_page() -> HTMLResponse:
    """Serve the Depth of Market (DOM) ladder page.

    Reads live data from ``GET /api/dom/snapshot`` and ``GET /sse/dom``.
    Click-to-trade features are gated behind funded accounts.
    """
    path = _resolve_static("dom.html")
    if path:
        logger.info("Serving dom.html from %s", path)
        return HTMLResponse(content=path.read_text(), headers={"Cache-Control": "no-cache"})

    logger.warning("dom.html not found in any candidate path")
    return HTMLResponse(content=_not_found_html("DOM", "dom.html", "📊"), status_code=200)


# ---------------------------------------------------------------------------
# GET /journal — Trade Journal
# ---------------------------------------------------------------------------


@router.get("/journal", response_class=HTMLResponse)
async def journal_page() -> HTMLResponse:
    """Serve the standalone trade journal page.

    Reads from ``GET /api/journal/trades`` and supports grading via
    ``POST /api/journal/trades/{id}/grade``.  Auto-sync from Rithmic
    fills is available when JOURNAL-SYNC phase is complete.
    """
    path = _resolve_static("journal.html")
    if path:
        logger.info("Serving journal.html from %s", path)
        return HTMLResponse(content=path.read_text(), headers={"Cache-Control": "no-cache"})

    logger.warning("journal.html not found in any candidate path")
    return HTMLResponse(content=_not_found_html("Journal", "journal.html", "📓"), status_code=200)


# ---------------------------------------------------------------------------
# GET /pretrade — Pre-Trade Analysis
# ---------------------------------------------------------------------------


@router.get("/pretrade", response_class=HTMLResponse)
async def pretrade_page() -> HTMLResponse:
    """Serve the Pre-Trade Analysis page.

    Connects to ``GET /api/pretrade/assets``, ``POST /api/pretrade/analyze``,
    ``POST /api/pretrade/select``, and ``GET /api/pretrade/watchlist`` for
    asset selection, opportunity scoring, and live monitoring.
    """
    path = _resolve_static("pretrade.html")
    if path:
        logger.info("Serving pretrade.html from %s", path)
        return HTMLResponse(content=path.read_text(), headers={"Cache-Control": "no-cache"})

    logger.warning("pretrade.html not found in any candidate path")
    return HTMLResponse(content=_not_found_html("Pre-Trade Analysis", "pretrade.html", "🔍"), status_code=200)
