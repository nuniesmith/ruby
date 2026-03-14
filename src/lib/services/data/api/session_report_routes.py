"""
Session Report API Routes
===========================
Endpoints for daily pre/post session reports and performance tracking.

    GET  /api/reports/pre-session          — Get today's pre-session report
    POST /api/reports/pre-session          — Generate & save pre-session report
    GET  /api/reports/post-session         — Get today's post-session report
    POST /api/reports/post-session         — Generate & save post-session report
    GET  /api/reports/history              — Get report history (last N days)
    POST /api/reports/pre-session/notes    — Update trader notes on pre-session
    POST /api/reports/post-session/notes   — Update trader notes on post-session
"""

from __future__ import annotations

import contextlib
import logging
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

logger = logging.getLogger("api.session_reports")

router = APIRouter(tags=["SessionReports"])


@router.post("/api/reports/pre-session")
async def create_pre_session_report(request: Request) -> JSONResponse:
    """Generate and save a pre-session report from the morning pipeline.

    Body (optional):
        plan_data: dict — pipeline plan output
        focus_assets: list — from compute_daily_focus()
        account_size: int (default 150000)
        trader_notes: str
    """
    from lib.services.engine.session_report import (
        generate_pre_session_report,
        save_pre_session_report,
    )

    body: dict[str, Any] = {}
    with contextlib.suppress(Exception):
        body = await request.json()

    report = generate_pre_session_report(
        plan_data=body.get("plan_data"),
        focus_assets=body.get("focus_assets"),
        account_size=int(body.get("account_size", 150_000)),
        trader_notes=body.get("trader_notes", ""),
    )

    saved = save_pre_session_report(report)

    from dataclasses import asdict

    return JSONResponse(
        {
            "ok": saved,
            "report": asdict(report),
        }
    )


@router.get("/api/reports/pre-session")
async def get_pre_session_report() -> JSONResponse:
    """Get today's pre-session report."""
    import json
    from datetime import date

    from lib.services.engine.session_report import _get_db, init_report_tables

    init_report_tables()
    today = date.today().isoformat()

    conn = _get_db()
    try:
        row = conn.execute(
            "SELECT data, trader_notes FROM pre_session_reports WHERE report_date = ?",
            (today,),
        ).fetchone()

        if row:
            return JSONResponse(
                {
                    "ok": True,
                    "report": json.loads(row["data"]),
                    "trader_notes": row["trader_notes"],
                }
            )
        else:
            return JSONResponse(
                {
                    "ok": False,
                    "error": "No pre-session report for today. Run the morning pipeline first.",
                }
            )
    finally:
        conn.close()


@router.post("/api/reports/post-session")
async def create_post_session_report(request: Request) -> JSONResponse:
    """Generate and save a post-session report.

    Body (optional):
        trades: list[dict] — trade records for the day
        daily_pnl: float
        max_drawdown: float
        max_contracts_used: int
        trader_notes: str
    """
    from lib.services.engine.session_report import (
        generate_post_session_report,
        save_post_session_report,
    )

    body: dict[str, Any] = {}
    with contextlib.suppress(Exception):
        body = await request.json()

    # Try to load today's plan data for adherence checking
    plan_data = None
    try:
        import json
        from datetime import date

        from lib.services.engine.session_report import _get_db, init_report_tables

        init_report_tables()
        conn = _get_db()
        row = conn.execute(
            "SELECT data FROM pre_session_reports WHERE report_date = ?",
            (date.today().isoformat(),),
        ).fetchone()
        if row:
            plan_data = json.loads(row["data"])
        conn.close()
    except Exception:
        pass

    report = generate_post_session_report(
        trades=body.get("trades", []),
        plan_data=plan_data,
        daily_pnl=float(body.get("daily_pnl", 0)),
        max_drawdown=float(body.get("max_drawdown", 0)),
        max_contracts_used=int(body.get("max_contracts_used", 0)),
        trader_notes=body.get("trader_notes", ""),
    )

    saved = save_post_session_report(report)

    from dataclasses import asdict

    return JSONResponse(
        {
            "ok": saved,
            "report": asdict(report),
        }
    )


@router.get("/api/reports/post-session")
async def get_post_session_report() -> JSONResponse:
    """Get today's post-session report."""
    import json
    from datetime import date

    from lib.services.engine.session_report import _get_db, init_report_tables

    init_report_tables()
    today = date.today().isoformat()

    conn = _get_db()
    try:
        row = conn.execute(
            "SELECT data, trader_notes, session_grade, net_pnl FROM post_session_reports WHERE report_date = ?",
            (today,),
        ).fetchone()

        if row:
            return JSONResponse(
                {
                    "ok": True,
                    "report": json.loads(row["data"]),
                    "grade": row["session_grade"],
                    "pnl": row["net_pnl"],
                }
            )
        else:
            return JSONResponse(
                {
                    "ok": False,
                    "error": "No post-session report for today yet.",
                }
            )
    finally:
        conn.close()


@router.get("/api/reports/history")
async def get_report_history(days: int = 30) -> JSONResponse:
    """Get report history for the last N trading days."""
    from lib.services.engine.session_report import get_report_history

    history = get_report_history(days=min(days, 90))
    return JSONResponse({"ok": True, **history})


@router.post("/api/reports/pre-session/notes")
async def update_pre_session_notes(request: Request) -> JSONResponse:
    """Update trader notes on today's pre-session report."""
    from datetime import date

    from lib.services.engine.session_report import _get_db, init_report_tables

    body = await request.json()
    notes = body.get("notes", "")

    init_report_tables()
    conn = _get_db()
    try:
        conn.execute(
            "UPDATE pre_session_reports SET trader_notes = ? WHERE report_date = ?",
            (notes, date.today().isoformat()),
        )
        conn.commit()
        return JSONResponse({"ok": True})
    except Exception as exc:
        logger.error("Failed to update pre-session notes: %s", exc)
        return JSONResponse({"ok": False, "error": str(exc)})
    finally:
        conn.close()


@router.post("/api/reports/post-session/notes")
async def update_post_session_notes(request: Request) -> JSONResponse:
    """Update trader notes on today's post-session report."""
    from datetime import date

    from lib.services.engine.session_report import _get_db, init_report_tables

    body = await request.json()
    notes = body.get("notes", "")

    init_report_tables()
    conn = _get_db()
    try:
        conn.execute(
            "UPDATE post_session_reports SET trader_notes = ? WHERE report_date = ?",
            (notes, date.today().isoformat()),
        )
        conn.commit()
        return JSONResponse({"ok": True})
    except Exception as exc:
        logger.error("Failed to update post-session notes: %s", exc)
        return JSONResponse({"ok": False, "error": str(exc)})
    finally:
        conn.close()
