"""
Audit API router — persistent risk/ORB event history endpoints.

Provides durable audit trail endpoints backed by the database (Postgres
or SQLite), complementing the in-memory / Redis-based risk history in
``risk.py``.

Endpoints:
  - GET  /audit/risk          — query persisted risk events
  - GET  /audit/orb           — query persisted ORB events
  - GET  /audit/summary       — aggregated summary for last N days
  - GET  /audit/daily-report  — structured daily ORB report (JSON)
  - POST /audit/risk          — manually record a risk event (internal use)
  - POST /audit/orb           — manually record an ORB event (internal use)

These tables are created automatically by ``models.init_db()`` on startup.
The engine service writes events via ``models.record_risk_event()`` and
``models.record_orb_event()`` during its CHECK_RISK_RULES and CHECK_ORB
handlers.  This router provides read access for dashboards and external
consumers, plus write access for manual / programmatic event injection.

Usage:
    # In main.py:
    from lib.services.data.api.audit import router as audit_router
    app.include_router(audit_router, prefix="/audit", tags=["Audit"])
"""

import contextlib
import json
import logging
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

# Ensure bare imports resolve

logger = logging.getLogger("api.audit")

_EST = ZoneInfo("America/New_York")

router = APIRouter(tags=["Audit"])


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class RiskEventCreate(BaseModel):
    """Request body for manually recording a risk event."""

    event_type: str = Field(..., description="Event type: 'block', 'warning', 'clear', 'circuit_breaker'")
    symbol: str = Field("", description="Instrument symbol, e.g. 'MGC'")
    side: str = Field("", description="LONG or SHORT")
    reason: str = Field("", description="Human-readable reason")
    daily_pnl: float = Field(0.0, description="Daily P&L at time of event")
    open_trades: int = Field(0, ge=0, description="Open trade count")
    account_size: int = Field(0, ge=0, description="Account size")
    risk_pct: float = Field(0.0, ge=0, description="Risk as % of account")
    session: str = Field("", description="Session mode (pre_market, active, off_hours)")
    metadata: dict | None = Field(None, description="Optional extra data")


class ORBEventCreate(BaseModel):
    """Request body for manually recording an ORB event."""

    symbol: str = Field(..., description="Instrument symbol")
    or_high: float = Field(0.0, description="Opening range high")
    or_low: float = Field(0.0, description="Opening range low")
    or_range: float = Field(0.0, description="OR high - OR low")
    atr_value: float = Field(0.0, description="ATR value")
    breakout_detected: bool = Field(False, description="Was a breakout detected?")
    direction: str = Field("", description="LONG, SHORT, or empty")
    trigger_price: float = Field(0.0, description="Breakout trigger price")
    long_trigger: float = Field(0.0, description="Upper breakout level")
    short_trigger: float = Field(0.0, description="Lower breakout level")
    bar_count: int = Field(0, ge=0, description="Bars in the opening range")
    session: str = Field("", description="Session mode")
    metadata: dict | None = Field(None, description="Optional extra data")


class RiskEventResponse(BaseModel):
    """A single risk event from the audit trail."""

    id: int | None = None
    timestamp: str = ""
    event_type: str = ""
    symbol: str = ""
    side: str = ""
    reason: str = ""
    daily_pnl: float = 0.0
    open_trades: int = 0
    account_size: int = 0
    risk_pct: float = 0.0
    session: str = ""
    metadata_json: str = "{}"


class ORBEventResponse(BaseModel):
    """A single ORB event from the audit trail."""

    id: int | None = None
    timestamp: str = ""
    symbol: str = ""
    or_high: float = 0.0
    or_low: float = 0.0
    or_range: float = 0.0
    atr_value: float = 0.0
    breakout_detected: int = 0
    direction: str = ""
    trigger_price: float = 0.0
    long_trigger: float = 0.0
    short_trigger: float = 0.0
    bar_count: int = 0
    session: str = ""
    metadata_json: str = "{}"


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("/risk")
def get_risk_events(
    limit: int = Query(50, ge=1, le=1000, description="Max events to return"),
    event_type: str | None = Query(None, description="Filter by event type (block, warning, clear)"),
    symbol: str | None = Query(None, description="Filter by symbol"),
    since: str | None = Query(None, description="ISO timestamp — only return events after this time"),
):
    """Query persisted risk events from the database.

    Returns the most recent events matching the filters, ordered by
    timestamp descending (most recent first).

    These events are written by the engine's CHECK_RISK_RULES handler
    and by the risk pre-flight check endpoint.
    """
    try:
        from lib.core.models import get_risk_events as _get_risk_events

        events = _get_risk_events(
            limit=limit,
            event_type=event_type,
            symbol=symbol,
            since=since,
        )

        # Parse metadata_json for each event
        for ev in events:
            if "metadata_json" in ev:
                try:
                    ev["metadata"] = json.loads(ev["metadata_json"])
                except (json.JSONDecodeError, TypeError):
                    ev["metadata"] = {}

        return {
            "events": events,
            "count": len(events),
            "limit": limit,
            "filters": {
                "event_type": event_type,
                "symbol": symbol,
                "since": since,
            },
            "timestamp": datetime.now(tz=_EST).isoformat(),
        }
    except Exception as exc:
        logger.error("Failed to query risk events: %s", exc)
        raise HTTPException(status_code=500, detail=f"Failed to query risk events: {exc}") from exc


@router.get("/orb")
def get_orb_events(
    limit: int = Query(50, ge=1, le=1000, description="Max events to return"),
    symbol: str | None = Query(None, description="Filter by symbol"),
    breakout_only: bool = Query(False, description="Only return events where breakout was detected"),
    since: str | None = Query(None, description="ISO timestamp — only return events after this time"),
    breakout_type: str | None = Query(
        None,
        description=(
            "Filter by breakout type: ORB | PDR | IB | CONS | WEEKLY | MONTHLY | "
            "ASIAN | BBSQUEEZE | VA | INSIDE | GAP | PIVOT | FIB.  "
            "Case-insensitive.  Omit or pass 'ALL' for all types."
        ),
    ),
):
    """Query persisted ORB events from the database.

    Returns the most recent ORB evaluations matching the filters,
    ordered by timestamp descending (most recent first).

    These events are written by the engine's CHECK_ORB handler.
    """
    try:
        from lib.core.models import get_orb_events as _get_orb_events

        events = _get_orb_events(
            limit=limit,
            symbol=symbol,
            breakout_only=breakout_only,
            since=since,
            breakout_type=breakout_type,
        )

        # Parse metadata_json and convert breakout_detected to bool
        for ev in events:
            if "metadata_json" in ev:
                try:
                    ev["metadata"] = json.loads(ev["metadata_json"])
                except (json.JSONDecodeError, TypeError):
                    ev["metadata"] = {}
            # Convert integer breakout_detected to boolean for API consumers
            if "breakout_detected" in ev:
                ev["breakout_detected_bool"] = bool(ev["breakout_detected"])

        return {
            "events": events,
            "count": len(events),
            "limit": limit,
            "filters": {
                "symbol": symbol,
                "breakout_only": breakout_only,
                "since": since,
                "breakout_type": breakout_type,
            },
            "timestamp": datetime.now(tz=_EST).isoformat(),
        }
    except Exception as exc:
        logger.error("Failed to query ORB events: %s", exc)
        raise HTTPException(status_code=500, detail=f"Failed to query ORB events: {exc}") from exc


@router.get("/summary")
def get_audit_summary(
    days: int = Query(7, ge=1, le=365, description="Number of days to summarise"),
):
    """Get an aggregated summary of audit events for the last N days.

    Returns counts and breakdowns for both risk and ORB events,
    useful for dashboard widgets and reporting.
    """
    try:
        from lib.core.models import get_audit_summary as _get_audit_summary

        summary = _get_audit_summary(days_back=days)
        summary["timestamp"] = datetime.now(tz=_EST).isoformat()
        return summary
    except Exception as exc:
        logger.error("Failed to build audit summary: %s", exc)
        raise HTTPException(status_code=500, detail=f"Failed to build audit summary: {exc}") from exc


@router.post("/risk", status_code=201)
def create_risk_event(req: RiskEventCreate):
    """Manually record a risk event for the audit trail.

    This endpoint is primarily for internal use — the engine records
    events automatically.  Use this for manual annotations, testing,
    or external system integrations.
    """
    try:
        from lib.core.models import record_risk_event

        row_id = record_risk_event(
            event_type=req.event_type,
            symbol=req.symbol,
            side=req.side,
            reason=req.reason,
            daily_pnl=req.daily_pnl,
            open_trades=req.open_trades,
            account_size=req.account_size,
            risk_pct=req.risk_pct,
            session=req.session,
            metadata=req.metadata,
        )

        if row_id is None:
            raise HTTPException(status_code=500, detail="Failed to insert risk event")

        logger.info(
            "Risk event recorded: id=%s type=%s symbol=%s",
            row_id,
            req.event_type,
            req.symbol,
        )

        return {
            "id": row_id,
            "event_type": req.event_type,
            "symbol": req.symbol,
            "timestamp": datetime.now(tz=_EST).isoformat(),
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to record risk event: %s", exc)
        raise HTTPException(status_code=500, detail=f"Failed to record risk event: {exc}") from exc


@router.post("/orb", status_code=201)
def create_orb_event(req: ORBEventCreate):
    """Manually record an ORB event for the audit trail.

    This endpoint is primarily for internal use — the engine records
    events automatically.  Use this for manual annotations, testing,
    or external system integrations.
    """
    try:
        from lib.core.models import record_orb_event

        row_id = record_orb_event(
            symbol=req.symbol,
            or_high=req.or_high,
            or_low=req.or_low,
            or_range=req.or_range,
            atr_value=req.atr_value,
            breakout_detected=req.breakout_detected,
            direction=req.direction,
            trigger_price=req.trigger_price,
            long_trigger=req.long_trigger,
            short_trigger=req.short_trigger,
            bar_count=req.bar_count,
            session=req.session,
            metadata=req.metadata,
        )

        if row_id is None:
            raise HTTPException(status_code=500, detail="Failed to insert ORB event")

        logger.info(
            "ORB event recorded: id=%s symbol=%s breakout=%s direction=%s",
            row_id,
            req.symbol,
            req.breakout_detected,
            req.direction,
        )

        return {
            "id": row_id,
            "symbol": req.symbol,
            "breakout_detected": req.breakout_detected,
            "direction": req.direction,
            "timestamp": datetime.now(tz=_EST).isoformat(),
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to record ORB event: %s", exc)
        raise HTTPException(status_code=500, detail=f"Failed to record ORB event: {exc}") from exc


# ---------------------------------------------------------------------------
# Daily report helpers (inline — mirrors scripts/daily_report.py logic)
# ---------------------------------------------------------------------------


def _parse_metadata_json(ev: dict) -> dict:
    """Safely parse metadata_json from an event row."""
    raw = ev.get("metadata_json", "{}")
    if isinstance(raw, dict):
        return raw
    try:
        return json.loads(raw) if raw else {}
    except (json.JSONDecodeError, TypeError):
        return {}


def _format_time_et(ts: str) -> str:
    """Extract HH:MM ET from an ISO timestamp string."""
    try:
        dt = datetime.fromisoformat(ts)
        dt = dt.replace(tzinfo=_EST) if dt.tzinfo is None else dt.astimezone(_EST)
        return dt.strftime("%H:%M ET")
    except Exception:
        return ts[:16] if len(ts) >= 16 else ts


def _query_orb_events_range(since: str, until: str, limit: int = 500) -> list[dict]:
    """Query ORB events between two ISO timestamps."""
    try:
        from lib.core.models import _get_conn, _is_using_postgres, _row_to_dict

        pg = _is_using_postgres()
        ph = "%s" if pg else "?"
        conn = _get_conn()
        sql = f"SELECT * FROM orb_events WHERE timestamp >= {ph} AND timestamp < {ph} ORDER BY timestamp ASC LIMIT {ph}"
        cur = conn.execute(sql, (since, until, limit))
        rows = cur.fetchall()
        conn.close()
        return [_row_to_dict(r) for r in rows]
    except Exception as exc:
        logger.error("Failed to query ORB events for report: %s", exc)
        return []


def _query_risk_events_range(since: str, until: str, limit: int = 500) -> list[dict]:
    """Query risk events between two ISO timestamps."""
    try:
        from lib.core.models import _get_conn, _is_using_postgres, _row_to_dict

        pg = _is_using_postgres()
        ph = "%s" if pg else "?"
        conn = _get_conn()
        sql = (
            f"SELECT * FROM risk_events WHERE timestamp >= {ph} AND timestamp < {ph} ORDER BY timestamp ASC LIMIT {ph}"
        )
        cur = conn.execute(sql, (since, until, limit))
        rows = cur.fetchall()
        conn.close()
        return [_row_to_dict(r) for r in rows]
    except Exception as exc:
        logger.error("Failed to query risk events for report: %s", exc)
        return []


def _get_model_info_for_report() -> dict[str, Any]:
    """Check the state of the CNN model files."""
    info: dict[str, Any] = {"available": False}

    model_dirs = [
        Path("/app/models"),
        Path("models"),
    ]

    for mdir in model_dirs:
        best = mdir / "breakout_cnn_best.pt"
        if best.is_file():
            stat = best.stat()
            info["available"] = True
            info["path"] = str(best)
            info["size_mb"] = round(stat.st_size / (1024 * 1024), 1)
            info["last_modified"] = datetime.fromtimestamp(stat.st_mtime, tz=_EST).isoformat()

            history = mdir / "training_history.csv"
            if history.is_file():
                try:
                    lines = history.read_text().strip().split("\n")
                    if len(lines) > 1:
                        header = lines[0].split(",")
                        last = lines[-1].split(",")
                        if len(header) == len(last):
                            last_epoch = dict(zip(header, last, strict=False))
                            info["last_training"] = {
                                k: v
                                for k, v in last_epoch.items()
                                if k
                                in (
                                    "epoch",
                                    "val_acc",
                                    "val_loss",
                                    "val_precision",
                                    "val_recall",
                                )
                            }
                except Exception:
                    pass

            try:
                checkpoints = list(mdir.glob("breakout_cnn_*.pt"))
                info["checkpoint_count"] = len(checkpoints)
            except Exception:
                pass

            break

    for ddir in [Path("/app/dataset"), Path("dataset")]:
        stats_file = ddir / "dataset_stats.json"
        if stats_file.is_file():
            with contextlib.suppress(Exception):
                info["dataset_stats"] = json.loads(stats_file.read_text())
            break
        labels_file = ddir / "labels.csv"
        if labels_file.is_file():
            try:
                line_count = sum(1 for _ in labels_file.open()) - 1
                info["dataset_size"] = line_count
            except Exception:
                pass
            break

    return info


def _build_daily_report(target_date: date) -> dict[str, Any]:
    """Build a structured report dict for a single trading day."""
    day_start = datetime(target_date.year, target_date.month, target_date.day, 0, 0, 0, tzinfo=_EST)
    day_end = day_start + timedelta(days=1)
    since = day_start.isoformat()
    until = day_end.isoformat()

    orb_events = _query_orb_events_range(since, until)
    risk_events = _query_risk_events_range(since, until)

    breakouts = [e for e in orb_events if e.get("breakout_detected", 0) == 1]
    non_breakouts = [e for e in orb_events if e.get("breakout_detected", 0) != 1]

    symbols_seen: dict[str, list[dict]] = {}
    for ev in orb_events:
        sym = ev.get("symbol", "UNKNOWN")
        symbols_seen.setdefault(sym, []).append(ev)

    cnn_probs: list[float] = []
    filter_passed_count = 0
    filter_failed_count = 0
    cnn_gated_count = 0
    published_count = 0

    for ev in breakouts:
        meta = _parse_metadata_json(ev)
        prob = meta.get("cnn_prob")
        if prob is not None:
            with contextlib.suppress(TypeError, ValueError):
                cnn_probs.append(float(prob))
        fp = meta.get("filter_passed")
        if fp is True:
            filter_passed_count += 1
        elif fp is False:
            filter_failed_count += 1
        if meta.get("cnn_gated"):
            cnn_gated_count += 1
        if meta.get("published"):
            published_count += 1

    if filter_passed_count == 0 and filter_failed_count == 0:
        published_count = len(breakouts)

    risk_blocks = [r for r in risk_events if r.get("event_type") == "block"]
    risk_warnings = [r for r in risk_events if r.get("event_type") == "warning"]

    cnn_stats: dict[str, Any] = {}
    if cnn_probs:
        cnn_stats = {
            "count": len(cnn_probs),
            "min": round(min(cnn_probs), 4),
            "max": round(max(cnn_probs), 4),
            "mean": round(sum(cnn_probs) / len(cnn_probs), 4),
            "above_50": sum(1 for p in cnn_probs if p >= 0.5),
            "above_70": sum(1 for p in cnn_probs if p >= 0.7),
        }

    model_info = _get_model_info_for_report()

    report: dict[str, Any] = {
        "date": target_date.isoformat(),
        "day_of_week": target_date.strftime("%A"),
        "generated_at": datetime.now(tz=_EST).isoformat(),
        "orb": {
            "total_evaluations": len(orb_events),
            "breakouts_detected": len(breakouts),
            "non_breakouts": len(non_breakouts),
            "filter_passed": filter_passed_count,
            "filter_failed": filter_failed_count,
            "cnn_gated": cnn_gated_count,
            "published": published_count,
            "symbols": {},
        },
        "cnn": cnn_stats,
        "risk": {
            "total_events": len(risk_events),
            "blocks": len(risk_blocks),
            "warnings": len(risk_warnings),
            "events": [
                {
                    "time": _format_time_et(r.get("timestamp", "")),
                    "type": r.get("event_type", ""),
                    "symbol": r.get("symbol", ""),
                    "reason": r.get("reason", ""),
                }
                for r in risk_events[:20]
            ],
        },
        "model": model_info,
        "breakouts": [],
    }

    for sym, events in sorted(symbols_seen.items()):
        sym_breakouts = [e for e in events if e.get("breakout_detected", 0) == 1]
        report["orb"]["symbols"][sym] = {
            "evaluations": len(events),
            "breakouts": len(sym_breakouts),
            "details": [],
        }
        for b in sym_breakouts:
            meta = _parse_metadata_json(b)
            detail = {
                "time": _format_time_et(b.get("timestamp", "")),
                "direction": b.get("direction", ""),
                "trigger_price": b.get("trigger_price", 0),
                "or_high": b.get("or_high", 0),
                "or_low": b.get("or_low", 0),
                "or_range": b.get("or_range", 0),
                "atr": b.get("atr_value", 0),
                "cnn_prob": meta.get("cnn_prob"),
                "cnn_confidence": meta.get("cnn_confidence", ""),
                "filter_passed": meta.get("filter_passed"),
                "filter_summary": meta.get("filter_summary", ""),
                "published": meta.get("published", False),
            }
            report["orb"]["symbols"][sym]["details"].append(detail)
            report["breakouts"].append({"symbol": sym, **detail})

    return report


# ---------------------------------------------------------------------------
# Daily report endpoint
# ---------------------------------------------------------------------------


@router.get("/daily-report")
def get_daily_report(
    target_date: str | None = Query(
        None,
        alias="date",
        description="Target date (YYYY-MM-DD). Defaults to today (ET).",
    ),
    days: int | None = Query(
        None,
        ge=1,
        le=30,
        description="Return summary for last N days instead of single-day detail.",
    ),
):
    """Structured daily ORB report — what happened today.

    Returns a JSON report covering:
      - ORB evaluations: total scans, breakouts detected, filter pass/fail
      - Per-symbol breakdown with direction, trigger price, OR range, CNN prob
      - CNN probability distribution statistics
      - Risk events during the session
      - Model status (available, last training metrics, dataset size)

    Use ``?date=2026-06-15`` for a specific date, or ``?days=5`` for a
    multi-day summary array.  Defaults to today (Eastern Time).
    """
    try:
        if target_date:
            try:
                td = date.fromisoformat(target_date)
            except ValueError as exc:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid date format: {target_date!r} (expected YYYY-MM-DD)",
                ) from exc
        else:
            td = datetime.now(tz=_EST).date()

        if days:
            reports = []
            for i in range(days):
                d = td - timedelta(days=i)
                reports.append(_build_daily_report(d))

            # Compute aggregate stats
            total_evals = sum(r["orb"]["total_evaluations"] for r in reports)
            total_bos = sum(r["orb"]["breakouts_detected"] for r in reports)
            total_pub = sum(r["orb"]["published"] for r in reports)
            total_risk = sum(r["risk"]["total_events"] for r in reports)
            days_with_bos = sum(1 for r in reports if r["orb"]["breakouts_detected"] > 0)

            return {
                "mode": "multi_day",
                "days": days,
                "start_date": (td - timedelta(days=days - 1)).isoformat(),
                "end_date": td.isoformat(),
                "generated_at": datetime.now(tz=_EST).isoformat(),
                "aggregate": {
                    "total_evaluations": total_evals,
                    "total_breakouts": total_bos,
                    "total_published": total_pub,
                    "total_risk_events": total_risk,
                    "days_with_breakouts": days_with_bos,
                    "breakout_rate": (round(total_bos / total_evals, 4) if total_evals > 0 else 0),
                    "publication_rate": (round(total_pub / total_bos, 4) if total_bos > 0 else 0),
                },
                "reports": reports,
            }

        report = _build_daily_report(td)
        report["mode"] = "single_day"
        return report

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to build daily report: %s", exc)
        raise HTTPException(status_code=500, detail=f"Failed to build daily report: {exc}") from exc
