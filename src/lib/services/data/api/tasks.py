"""
Tasks & Issue Capture API Router
==================================
Lightweight task/bug/note capture that works in three modes:

  1. Quick-capture from the dashboard  — one click while trading
  2. Chat-driven capture               — assistant detects intent and calls POST /api/tasks
  3. RustAssistant GitHub integration  — forwards tasks to RA which can open
                                         GitHub issues / PRs on any connected repo

Storage
-------
Tasks are stored in a dedicated ``tasks`` table in the existing SQLite / Postgres
database (same connection as trades_v2 / daily_journal).  The table is created
idempotently inside this module so no changes to models.py are required.

RustAssistant integration
--------------------------
When RA_BASE_URL + RA_API_KEY are set and the task has a ``repo`` field,
a background task POSTs to:

    POST {RA_BASE_URL}/api/github/issue

with a JSON body that RA understands:

    {
      "repo":    "jordan/futures",   # or the RA repo slug
      "title":   "<task title>",
      "body":    "<markdown body>",
      "labels":  ["bug"|"enhancement"|"note"],
      "draft":   false
    }

RA returns { "html_url": "https://github.com/..." } on success.
The task row is updated with the resulting GitHub URL.

Endpoints
---------
  POST   /api/tasks                 — Create a task
  GET    /api/tasks                 — List tasks (filtered by status/type)
  GET    /api/tasks/{id}            — Single task JSON
  PUT    /api/tasks/{id}            — Update a task
  DELETE /api/tasks/{id}            — Delete a task
  POST   /api/tasks/{id}/github     — (Re)push task to GitHub via RA
  GET    /api/tasks/html            — HTMX fragment — task feed panel
  GET    /api/tasks/{id}/html       — HTMX fragment — single task card

Task types
----------
  bug       — Something broken / wrong in the UI or engine
  task      — A future piece of work / improvement
  note      — Quick observation recorded while trading

Task statuses
-------------
  open      — Newly captured, not yet acted on
  planned   — Acknowledged / added to backlog
  in_progress — Being worked on (RA has a branch / PR open)
  done      — Completed / merged
  dismissed — Won't fix / not relevant
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import sqlite3
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

logger = logging.getLogger("api.tasks")

_ET = ZoneInfo("America/New_York")

router = APIRouter(tags=["Tasks"])

# ---------------------------------------------------------------------------
# Database — reuse the existing DB_PATH / DATABASE_URL from models
# ---------------------------------------------------------------------------

_DB_PATH = os.getenv("DB_PATH", "futures_journal.db")
_DATABASE_URL = os.getenv("DATABASE_URL", "")
_USE_POSTGRES = _DATABASE_URL.startswith("postgresql")

_SCHEMA_SQLITE = """
CREATE TABLE IF NOT EXISTS tasks (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at      TEXT    NOT NULL,
    updated_at      TEXT    NOT NULL,
    task_type       TEXT    NOT NULL DEFAULT 'task',
    status          TEXT    NOT NULL DEFAULT 'open',
    priority        TEXT    NOT NULL DEFAULT 'medium',
    title           TEXT    NOT NULL,
    description     TEXT    NOT NULL DEFAULT '',
    page_context    TEXT    NOT NULL DEFAULT '',
    market_snapshot TEXT    NOT NULL DEFAULT '',
    repo            TEXT    NOT NULL DEFAULT '',
    github_url      TEXT    NOT NULL DEFAULT '',
    github_number   INTEGER NOT NULL DEFAULT 0,
    tags            TEXT    NOT NULL DEFAULT '',
    source          TEXT    NOT NULL DEFAULT 'manual',
    session_id      TEXT    NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_tasks_status    ON tasks (status, created_at);
CREATE INDEX IF NOT EXISTS idx_tasks_type      ON tasks (task_type, created_at);
CREATE INDEX IF NOT EXISTS idx_tasks_session   ON tasks (session_id);
"""

_SCHEMA_PG = """
CREATE TABLE IF NOT EXISTS tasks (
    id              SERIAL PRIMARY KEY,
    created_at      TEXT    NOT NULL,
    updated_at      TEXT    NOT NULL,
    task_type       TEXT    NOT NULL DEFAULT 'task',
    status          TEXT    NOT NULL DEFAULT 'open',
    priority        TEXT    NOT NULL DEFAULT 'medium',
    title           TEXT    NOT NULL,
    description     TEXT    NOT NULL DEFAULT '',
    page_context    TEXT    NOT NULL DEFAULT '',
    market_snapshot TEXT    NOT NULL DEFAULT '',
    repo            TEXT    NOT NULL DEFAULT '',
    github_url      TEXT    NOT NULL DEFAULT '',
    github_number   INTEGER NOT NULL DEFAULT 0,
    tags            TEXT    NOT NULL DEFAULT '',
    source          TEXT    NOT NULL DEFAULT 'manual',
    session_id      TEXT    NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_tasks_status    ON tasks (status, created_at);
CREATE INDEX IF NOT EXISTS idx_tasks_type      ON tasks (task_type, created_at);
CREATE INDEX IF NOT EXISTS idx_tasks_session   ON tasks (session_id);
"""

# ---------------------------------------------------------------------------
# Valid enum values
# ---------------------------------------------------------------------------

TASK_TYPES = {"bug", "task", "note"}
TASK_STATUSES = {"open", "planned", "in_progress", "done", "dismissed"}
TASK_PRIORITIES = {"low", "medium", "high", "critical"}

# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------


def _get_conn():
    """Return a DB connection using the same backend as models.py."""
    if _USE_POSTGRES:
        try:
            from lib.core.models import _get_conn as _models_conn

            return _models_conn()
        except Exception as exc:
            logger.warning("Postgres conn failed, falling back to SQLite: %s", exc)

    conn = sqlite3.connect(_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _init_tasks_table() -> None:
    """Create the tasks table idempotently (called at router import time)."""
    try:
        conn = _get_conn()
        if _USE_POSTGRES:
            conn.executescript(_SCHEMA_PG)
        else:
            conn.executescript(_SCHEMA_SQLITE)
        conn.commit()
        conn.close()
        logger.debug("tasks table ready")
    except Exception as exc:
        logger.error("Failed to initialise tasks table: %s", exc)


# Initialise on import so the table is always ready
_init_tasks_table()


def _row_to_dict(row) -> dict[str, Any]:
    """Convert a sqlite3.Row or dict-like row to a plain dict."""
    if row is None:
        return {}
    if hasattr(row, "keys"):
        return dict(row)
    return dict(zip(row.keys(), tuple(row), strict=False))


def _now_iso() -> str:
    return datetime.now(tz=_ET).isoformat()


# ---------------------------------------------------------------------------
# RustAssistant GitHub integration
# ---------------------------------------------------------------------------

_RA_BASE_URL: str = os.environ.get("RA_BASE_URL", "").rstrip("/")
_RA_API_KEY: str = os.environ.get("RA_API_KEY", "")
_RA_REPO_ID: str = os.environ.get("RA_REPO_ID", "")  # e.g. "futures-bot"
_GITHUB_REPO: str = os.environ.get("GITHUB_REPO", "")  # e.g. "jordan/futures"


def _ra_available() -> bool:
    return bool(_RA_BASE_URL and _RA_API_KEY)


async def _push_to_github(task: dict[str, Any]) -> dict[str, Any]:
    """Forward a task to RustAssistant's GitHub issue endpoint.

    Returns a dict with at minimum:
        { "ok": bool, "html_url": str | None, "number": int | None, "error": str | None }

    Never raises — errors are returned in the dict so the caller can
    decide whether to surface them.
    """
    if not _ra_available():
        return {"ok": False, "html_url": None, "number": None, "error": "RA not configured"}

    repo = task.get("repo") or _GITHUB_REPO or _RA_REPO_ID
    if not repo:
        return {"ok": False, "html_url": None, "number": None, "error": "No repo configured"}

    # Build a rich markdown body for the GitHub issue
    task_type = task.get("task_type", "task")
    priority = task.get("priority", "medium")
    page_ctx = task.get("page_context", "")
    mkt_snap = task.get("market_snapshot", "")
    description = task.get("description", "")
    tags_raw = task.get("tags", "")
    created_at = task.get("created_at", _now_iso())
    session_id = task.get("session_id", "")

    body_parts = []
    if description:
        body_parts.append(description)
    if page_ctx:
        body_parts.append(f"\n**Dashboard context:** `{page_ctx}`")
    if mkt_snap:
        body_parts.append(f"\n**Market snapshot at capture:**\n```\n{mkt_snap}\n```")
    body_parts.append(f"\n---\n*Captured: {created_at} | Priority: {priority} | Session: {session_id or 'n/a'}*")
    if tags_raw:
        body_parts.append(f"*Tags: {tags_raw}*")

    body_parts.append("\n*Auto-captured via Ruby Futures dashboard → RustAssistant*")

    label_map = {"bug": "bug", "task": "enhancement", "note": "documentation"}
    labels = [label_map.get(task_type, "enhancement")]
    if priority in ("high", "critical"):
        labels.append("priority:high")

    payload = {
        "repo": repo,
        "title": task.get("title", "(no title)"),
        "body": "\n".join(body_parts),
        "labels": labels,
        "draft": False,
    }

    import httpx

    headers = {
        "Authorization": f"Bearer {_RA_API_KEY}",
        "Content-Type": "application/json",
    }
    if _RA_REPO_ID:
        headers["x-repo-id"] = _RA_REPO_ID

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(
                f"{_RA_BASE_URL}/api/github/issue",
                headers=headers,
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()
            return {
                "ok": True,
                "html_url": data.get("html_url"),
                "number": data.get("number"),
                "error": None,
            }
    except httpx.HTTPStatusError as exc:
        msg = f"RA HTTP {exc.response.status_code}: {exc.response.text[:200]}"
        logger.warning("GitHub push failed: %s", msg)
        return {"ok": False, "html_url": None, "number": None, "error": msg}
    except Exception as exc:
        logger.warning("GitHub push failed: %s", exc)
        return {"ok": False, "html_url": None, "number": None, "error": str(exc)}


# ---------------------------------------------------------------------------
# Market snapshot helper
# ---------------------------------------------------------------------------


def _capture_market_snapshot() -> str:
    """Build a compact market snapshot string from Redis for task context."""
    parts: list[str] = []
    try:
        from lib.core.cache import cache_get

        raw = cache_get("engine:grok_update")
        if raw:
            upd = json.loads(raw.decode() if isinstance(raw, bytes) else raw)
            text = upd.get("text", "")
            ts = upd.get("time_et", "")
            if text:
                parts.append(f"AI ({ts}): {text[:300]}")

        raw_focus = cache_get("engine:daily_focus")
        if raw_focus:
            focus = json.loads(raw_focus.decode() if isinstance(raw_focus, bytes) else raw_focus)
            assets = focus.get("assets", [])
            if assets:
                from lib.integrations.grok_helper import format_live_compact

                parts.append(format_live_compact(assets)[:400])
    except Exception:
        pass

    return "\n".join(parts) if parts else ""


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class TaskCreateRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=300)
    description: str = Field(default="", max_length=4000)
    task_type: str = Field(default="task")
    priority: str = Field(default="medium")
    page_context: str = Field(
        default="", max_length=200, description="Dashboard page/panel where the issue was observed"
    )
    tags: str = Field(default="", max_length=300, description="Comma-separated tags")
    repo: str = Field(default="", max_length=200, description="GitHub repo slug override, e.g. jordan/futures")
    push_to_github: bool = Field(default=False, description="Immediately open a GitHub issue via RustAssistant")
    capture_market: bool = Field(default=True, description="Snapshot current market context into the task")
    session_id: str = Field(default="", description="Chat session ID that triggered this task, if any")
    source: str = Field(default="manual", description="manual | chat | quick_capture")


class TaskUpdateRequest(BaseModel):
    title: str | None = None
    description: str | None = None
    status: str | None = None
    priority: str | None = None
    tags: str | None = None
    repo: str | None = None


class TaskResponse(BaseModel):
    id: int
    created_at: str
    updated_at: str
    task_type: str
    status: str
    priority: str
    title: str
    description: str
    page_context: str
    market_snapshot: str
    repo: str
    github_url: str
    github_number: int
    tags: str
    source: str
    session_id: str


# ---------------------------------------------------------------------------
# CRUD helpers
# ---------------------------------------------------------------------------


def _insert_task(data: dict[str, Any]) -> int:
    conn = _get_conn()
    try:
        cur = conn.execute(
            """
            INSERT INTO tasks
              (created_at, updated_at, task_type, status, priority,
               title, description, page_context, market_snapshot,
               repo, github_url, github_number, tags, source, session_id)
            VALUES
              (:created_at, :updated_at, :task_type, :status, :priority,
               :title, :description, :page_context, :market_snapshot,
               :repo, :github_url, :github_number, :tags, :source, :session_id)
            """,
            data,
        )
        conn.commit()
        return cur.lastrowid  # type: ignore[return-value]
    finally:
        conn.close()


def _fetch_task(task_id: int) -> dict[str, Any] | None:
    conn = _get_conn()
    try:
        cur = conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,))
        row = cur.fetchone()
        return _row_to_dict(row) if row else None
    finally:
        conn.close()


def _fetch_tasks(
    status: str | None = None,
    task_type: str | None = None,
    limit: int = 100,
    offset: int = 0,
) -> list[dict[str, Any]]:
    conn = _get_conn()
    try:
        where_clauses: list[str] = []
        params: list[Any] = []

        if status:
            where_clauses.append("status = ?")
            params.append(status)
        if task_type:
            where_clauses.append("task_type = ?")
            params.append(task_type)

        where = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""
        params.extend([limit, offset])

        cur = conn.execute(
            f"SELECT * FROM tasks {where} ORDER BY created_at DESC LIMIT ? OFFSET ?",
            params,
        )
        return [_row_to_dict(r) for r in cur.fetchall()]
    finally:
        conn.close()


def _update_task_row(task_id: int, updates: dict[str, Any]) -> bool:
    if not updates:
        return False
    updates["updated_at"] = _now_iso()
    set_clause = ", ".join(f"{k} = :{k}" for k in updates)
    updates["id"] = task_id
    conn = _get_conn()
    try:
        cur = conn.execute(f"UPDATE tasks SET {set_clause} WHERE id = :id", updates)
        conn.commit()
        # rowcount may not be available on all backends (e.g. _PgCursorWrapper)
        rowcount = getattr(cur, "rowcount", None)
        return bool(rowcount > 0) if rowcount is not None else True
    finally:
        conn.close()


def _delete_task_row(task_id: int) -> bool:
    conn = _get_conn()
    try:
        cur = conn.execute("DELETE FROM tasks WHERE id = ?", (task_id,))
        conn.commit()
        rowcount = getattr(cur, "rowcount", None)
        return bool(rowcount > 0) if rowcount is not None else True
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# HTML rendering helpers
# ---------------------------------------------------------------------------

_TYPE_COLORS = {
    "bug": ("#FF4560", "🐛"),
    "task": ("#3B9EFF", "✅"),
    "note": ("#A78BFA", "📝"),
}

_STATUS_COLORS = {
    "open": "#FFB020",
    "planned": "#3B9EFF",
    "in_progress": "#22D3EE",
    "done": "#00E87A",
    "dismissed": "#71717a",
}

_PRIORITY_COLORS = {
    "critical": "#FF4560",
    "high": "#FFB020",
    "medium": "#3B9EFF",
    "low": "#71717a",
}


def _render_task_card(t: dict[str, Any], compact: bool = False) -> str:
    """Render a single task as an HTML card fragment (HTMX-swappable)."""
    tid = t.get("id", 0)
    title = t.get("title", "(no title)")
    desc = t.get("description", "")
    ttype = t.get("task_type", "task")
    status = t.get("status", "open")
    priority = t.get("priority", "medium")
    created = t.get("created_at", "")[:16].replace("T", " ")
    page_ctx = t.get("page_context", "")
    gh_url = t.get("github_url", "")
    tags_raw = t.get("tags", "")
    source = t.get("source", "manual")

    type_color, type_icon = _TYPE_COLORS.get(ttype, ("#71717a", "•"))
    status_color = _STATUS_COLORS.get(status, "#71717a")
    priority_color = _PRIORITY_COLORS.get(priority, "#71717a")

    # Tags
    tags_html = ""
    if tags_raw:
        pills = "".join(
            f'<span style="font-size:9px;padding:1px 6px;border-radius:3px;'
            f'background:rgba(167,139,250,.15);color:#A78BFA;border:1px solid rgba(167,139,250,.3)">'
            f"{tag.strip()}</span>"
            for tag in tags_raw.split(",")
            if tag.strip()
        )
        tags_html = f'<div style="display:flex;gap:4px;flex-wrap:wrap;margin-top:5px">{pills}</div>'

    # GitHub link
    gh_html = ""
    if gh_url:
        gh_html = (
            f'<a href="{gh_url}" target="_blank" rel="noopener" '
            f'style="font-size:10px;color:#3B9EFF;text-decoration:none;margin-left:auto">'
            f"↗ GitHub</a>"
        )

    # Page context badge
    ctx_html = ""
    if page_ctx:
        ctx_html = (
            f'<span style="font-size:9px;padding:1px 6px;border-radius:3px;'
            f'background:rgba(34,211,238,.1);color:#22D3EE;border:1px solid rgba(34,211,238,.2)">'
            f"📍 {page_ctx}</span>"
        )

    # Source badge
    src_icon = {"chat": "💬", "quick_capture": "⚡", "manual": "✍️"}.get(source, "•")

    # Status selector (inline HTMX)
    status_options = "".join(
        f'<option value="{s}" {"selected" if s == status else ""}>{s.replace("_", " ").title()}</option>'
        for s in TASK_STATUSES
    )

    desc_html = ""
    if desc and not compact:
        safe_desc = desc.replace("<", "&lt;").replace(">", "&gt;")
        desc_html = (
            f'<div style="font-size:11px;color:#a1a1aa;line-height:1.55;margin:6px 0 4px;'
            f'white-space:pre-wrap">{safe_desc}</div>'
        )

    push_btn = ""
    if _ra_available() and not gh_url:
        push_btn = (
            f'<button onclick="pushTaskToGitHub({tid})" '
            f'style="font-size:9px;padding:2px 8px;border-radius:3px;cursor:pointer;'
            f"background:rgba(0,232,122,.1);border:1px solid rgba(0,232,122,.3);"
            f'color:#00E87A;font-family:inherit" title="Open GitHub issue via RustAssistant">'
            f"↑ GitHub</button>"
        )

    return f"""
<div id="task-card-{tid}"
     style="background:rgba(24,24,27,.9);border:1px solid #27272a;border-left:3px solid {type_color};
            border-radius:5px;padding:10px 12px;margin-bottom:7px;transition:border-color .15s"
     onmouseenter="this.style.borderColor='{type_color}'"
     onmouseleave="this.style.borderLeftColor='{type_color}';this.style.borderColor='#27272a';this.style.borderLeftColor='{type_color}'">
  <div style="display:flex;align-items:flex-start;gap:8px">
    <span style="font-size:14px;flex-shrink:0;margin-top:1px">{type_icon}</span>
    <div style="flex:1;min-width:0">
      <div style="display:flex;align-items:center;gap:6px;flex-wrap:wrap;margin-bottom:3px">
        <span style="font-size:12px;font-weight:600;color:#f4f4f5;line-height:1.3">{title}</span>
        <span style="font-size:9px;padding:1px 5px;border-radius:2px;
                     background:{priority_color}22;color:{priority_color};
                     border:1px solid {priority_color}44;text-transform:uppercase;
                     letter-spacing:.06em">{priority}</span>
        {ctx_html}
        {gh_html}
      </div>
      {desc_html}
      {tags_html}
      <div style="display:flex;align-items:center;gap:8px;margin-top:6px;flex-wrap:wrap">
        <span style="font-size:9px;color:#52525b">{src_icon} {created}</span>
        <select onchange="updateTaskStatus({tid}, this.value)"
                style="font-size:9px;padding:1px 5px;border-radius:3px;cursor:pointer;
                       background:rgba(39,39,42,.8);border:1px solid #3f3f46;
                       color:{status_color};font-family:inherit">
          {status_options}
        </select>
        {push_btn}
        <button onclick="deleteTask({tid})"
                style="font-size:9px;padding:2px 6px;border-radius:3px;cursor:pointer;
                       background:rgba(255,69,96,.08);border:1px solid rgba(255,69,96,.2);
                       color:#FF4560;font-family:inherit;margin-left:auto">✕</button>
      </div>
    </div>
  </div>
</div>
"""


def _render_tasks_panel(tasks: list[dict[str, Any]]) -> str:
    """Render the full tasks feed panel as an HTML fragment."""
    counts = {
        "open": sum(1 for t in tasks if t.get("status") == "open"),
        "in_progress": sum(1 for t in tasks if t.get("status") == "in_progress"),
        "done": sum(1 for t in tasks if t.get("status") == "done"),
    }
    bugs = sum(1 for t in tasks if t.get("task_type") == "bug")
    _notes = sum(1 for t in tasks if t.get("task_type") == "note")

    if not tasks:
        empty = (
            '<div style="text-align:center;padding:32px 16px;color:#52525b;font-size:12px">'
            "No tasks yet — use the quick-capture buttons or ask the assistant to log something."
            "</div>"
        )
        cards_html = empty
    else:
        cards_html = "\n".join(_render_task_card(t) for t in tasks)

    ra_badge = ""
    if _ra_available():
        ra_badge = (
            '<span style="font-size:9px;padding:1px 7px;border-radius:3px;'
            'background:rgba(0,232,122,.1);color:#00E87A;border:1px solid rgba(0,232,122,.25)">'
            "⚡ RA connected</span>"
        )

    return f"""
<div id="tasks-panel" style="font-family:ui-monospace,'Cascadia Code',monospace">
  <div style="display:flex;align-items:center;gap:8px;margin-bottom:10px;flex-wrap:wrap">
    <div style="display:flex;gap:6px">
      <span style="font-size:10px;padding:2px 8px;border-radius:3px;
                   background:rgba(255,176,32,.12);color:#FFB020;border:1px solid rgba(255,176,32,.25)">
        {counts["open"]} open
      </span>
      <span style="font-size:10px;padding:2px 8px;border-radius:3px;
                   background:rgba(34,211,238,.1);color:#22D3EE;border:1px solid rgba(34,211,238,.2)">
        {counts["in_progress"]} active
      </span>
      <span style="font-size:10px;padding:2px 8px;border-radius:3px;
                   background:rgba(0,232,122,.1);color:#00E87A;border:1px solid rgba(0,232,122,.2)">
        {counts["done"]} done
      </span>
      <span style="font-size:10px;padding:2px 8px;border-radius:3px;
                   background:rgba(255,69,96,.1);color:#FF4560;border:1px solid rgba(255,69,96,.2)">
        🐛 {bugs}
      </span>
    </div>
    {ra_badge}
  </div>
  <div id="tasks-list">
    {cards_html}
  </div>
</div>
"""


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/api/tasks", response_model=TaskResponse)
async def create_task(req: TaskCreateRequest):
    """Create a new task/bug/note.

    If ``push_to_github=true`` and RustAssistant is available, also opens
    a GitHub issue in the background and updates the task row with the URL.
    """
    if req.task_type not in TASK_TYPES:
        raise HTTPException(status_code=422, detail=f"task_type must be one of {sorted(TASK_TYPES)}")
    if req.priority not in TASK_PRIORITIES:
        raise HTTPException(status_code=422, detail=f"priority must be one of {sorted(TASK_PRIORITIES)}")

    now = _now_iso()

    market_snapshot = ""
    if req.capture_market:
        market_snapshot = await asyncio.to_thread(_capture_market_snapshot)

    row: dict[str, Any] = {
        "created_at": now,
        "updated_at": now,
        "task_type": req.task_type,
        "status": "open",
        "priority": req.priority,
        "title": req.title.strip(),
        "description": req.description.strip(),
        "page_context": req.page_context.strip(),
        "market_snapshot": market_snapshot,
        "repo": req.repo.strip(),
        "github_url": "",
        "github_number": 0,
        "tags": req.tags.strip(),
        "source": req.source,
        "session_id": req.session_id.strip(),
    }

    task_id: int = await asyncio.to_thread(_insert_task, row)
    row["id"] = task_id

    # Push to GitHub in background if requested
    if req.push_to_github and _ra_available():
        asyncio.create_task(_push_github_and_update(task_id, row))

    # Publish to Redis so the chat / dashboard can subscribe
    _publish_task_event("created", row)

    return TaskResponse(**row)


async def _push_github_and_update(task_id: int, task: dict[str, Any]) -> None:
    """Background coroutine: push to GitHub via RA and update the DB row."""
    result = await _push_to_github(task)
    if result["ok"]:
        updates = {
            "github_url": result.get("html_url") or "",
            "github_number": result.get("number") or 0,
            "status": "planned",
        }
        await asyncio.to_thread(_update_task_row, task_id, updates)
        logger.info(
            "Task %d pushed to GitHub: %s (#%s)",
            task_id,
            updates["github_url"],
            updates["github_number"],
        )
        _publish_task_event("github_linked", {**task, **updates, "id": task_id})
    else:
        logger.warning("Task %d GitHub push failed: %s", task_id, result.get("error"))


def _publish_task_event(event_type: str, task: dict[str, Any]) -> None:
    """Publish a task event to Redis pub/sub for dashboard subscribers."""
    with contextlib.suppress(Exception):
        from lib.core.cache import REDIS_AVAILABLE, _r

        if REDIS_AVAILABLE and _r is not None:
            payload = json.dumps({"event": event_type, "task": task})
            _r.publish("dashboard:tasks", payload)


@router.get("/api/tasks")
async def list_tasks(
    status: str | None = Query(default=None, description="Filter by status"),
    task_type: str | None = Query(default=None, description="Filter by type"),
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
):
    """List tasks with optional filtering."""
    if status and status not in TASK_STATUSES:
        raise HTTPException(status_code=422, detail=f"status must be one of {sorted(TASK_STATUSES)}")
    if task_type and task_type not in TASK_TYPES:
        raise HTTPException(status_code=422, detail=f"task_type must be one of {sorted(TASK_TYPES)}")

    tasks = await asyncio.to_thread(_fetch_tasks, status, task_type, limit, offset)
    return {"status": "ok", "count": len(tasks), "tasks": tasks}


@router.get("/api/tasks/html", response_class=HTMLResponse)
async def tasks_html(
    status: str | None = Query(default=None),
    task_type: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
):
    """Return the tasks feed as an HTMX-swappable HTML fragment."""
    tasks = await asyncio.to_thread(_fetch_tasks, status, task_type, limit, 0)
    return HTMLResponse(_render_tasks_panel(tasks))


@router.get("/api/tasks/{task_id}", response_model=TaskResponse)
async def get_task(task_id: int):
    """Fetch a single task by ID."""
    task = await asyncio.to_thread(_fetch_task, task_id)
    if not task:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
    return TaskResponse(**task)


@router.get("/api/tasks/{task_id}/html", response_class=HTMLResponse)
async def get_task_html(task_id: int):
    """Return a single task card as an HTMX-swappable fragment."""
    task = await asyncio.to_thread(_fetch_task, task_id)
    if not task:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
    return HTMLResponse(_render_task_card(task))


@router.put("/api/tasks/{task_id}", response_model=TaskResponse)
async def update_task(task_id: int, req: TaskUpdateRequest):
    """Update fields on an existing task."""
    existing = await asyncio.to_thread(_fetch_task, task_id)
    if not existing:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

    updates: dict[str, Any] = {}
    if req.title is not None:
        updates["title"] = req.title.strip()
    if req.description is not None:
        updates["description"] = req.description.strip()
    if req.tags is not None:
        updates["tags"] = req.tags.strip()
    if req.repo is not None:
        updates["repo"] = req.repo.strip()
    if req.status is not None:
        if req.status not in TASK_STATUSES:
            raise HTTPException(status_code=422, detail=f"status must be one of {sorted(TASK_STATUSES)}")
        updates["status"] = req.status
    if req.priority is not None:
        if req.priority not in TASK_PRIORITIES:
            raise HTTPException(status_code=422, detail=f"priority must be one of {sorted(TASK_PRIORITIES)}")
        updates["priority"] = req.priority

    await asyncio.to_thread(_update_task_row, task_id, updates)
    updated = await asyncio.to_thread(_fetch_task, task_id)
    if updated:
        _publish_task_event("updated", updated)
    return TaskResponse(**(updated or {}))


@router.delete("/api/tasks/{task_id}")
async def delete_task(task_id: int):
    """Delete a task permanently."""
    deleted = await asyncio.to_thread(_delete_task_row, task_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
    _publish_task_event("deleted", {"id": task_id})
    return {"status": "deleted", "id": task_id}


@router.post("/api/tasks/{task_id}/github")
async def push_task_to_github(task_id: int):
    """(Re)push a task to GitHub via RustAssistant.

    Can be called again to retry a previously failed push, or to open a
    new issue if the old one was closed.
    """
    if not _ra_available():
        raise HTTPException(status_code=503, detail="RustAssistant not configured (RA_BASE_URL / RA_API_KEY)")

    task = await asyncio.to_thread(_fetch_task, task_id)
    if not task:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")

    result = await _push_to_github(task)

    if result["ok"]:
        updates = {
            "github_url": result.get("html_url") or "",
            "github_number": result.get("number") or 0,
            "status": "planned",
        }
        await asyncio.to_thread(_update_task_row, task_id, updates)
        updated = await asyncio.to_thread(_fetch_task, task_id)
        if updated:
            _publish_task_event("github_linked", updated)
        return {
            "status": "ok",
            "github_url": result["html_url"],
            "github_number": result["number"],
            "task": updated,
        }

    raise HTTPException(status_code=502, detail=f"GitHub push failed: {result.get('error')}")


# ---------------------------------------------------------------------------
# Status endpoint
# ---------------------------------------------------------------------------


@router.get("/api/tasks/status")
async def tasks_status():
    """Return configuration status for the tasks subsystem."""
    tasks = await asyncio.to_thread(_fetch_tasks, None, None, 1000, 0)
    counts: dict[str, int] = {}
    for t in tasks:
        counts[t.get("status", "open")] = counts.get(t.get("status", "open"), 0) + 1

    return {
        "status": "ok",
        "ra_available": _ra_available(),
        "ra_base_url": _RA_BASE_URL or None,
        "github_repo": _GITHUB_REPO or _RA_REPO_ID or None,
        "total_tasks": len(tasks),
        "counts_by_status": counts,
        "timestamp": datetime.now(tz=_ET).isoformat(),
    }
