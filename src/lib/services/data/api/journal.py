"""
Journal API router — daily P&L journal endpoints.

Endpoints:
    GET  /journal/html               — HTMX-swappable fragment (dashboard panel)
    GET  /journal/page               — Standalone full-page view with dark theme + nav bar
    GET  /journal/entries            — JSON list of recent journal entries
    GET  /journal/stats              — JSON aggregated statistics
    GET  /journal/today              — JSON today's entry (if it exists)
    GET  /journal/tags               — JSON all unique tags with usage counts
    GET  /journal/trades             — JSON trades from trades_v2 (source/account filter)
    GET  /journal/trades/html        — HTMX trade-review panel (account filter + grade UI)
    POST /journal/save               — Save / upsert a daily journal entry
    PUT  /journal/entry/{date}       — Update fields on an existing entry
    POST /journal/trades/{id}/grade  — Set / update the quality grade on a trade
    POST /journal/sync               — Manually trigger Rithmic fill → trades_v2 sync
    GET  /journal/sync/status        — Last sync result (from Redis cache)

Provides endpoints for saving end-of-day journal entries,
retrieving journal history, computing journal statistics,
auto-syncing trades from Rithmic fills, and an improved HTMX-powered UI
with inline editing, tag filtering, and account-level trade review.
"""

import logging
from datetime import date, datetime
from zoneinfo import ZoneInfo

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from lib.core.models import (
    _get_conn,
    get_daily_journal,
    get_journal_stats,
    save_daily_journal,
)

logger = logging.getLogger("api.journal")

_EST = ZoneInfo("America/New_York")

router = APIRouter(tags=["journal"])

# ---------------------------------------------------------------------------
# Sync request model
# ---------------------------------------------------------------------------


class SyncRequest(BaseModel):
    """Optional body for POST /journal/sync."""

    account_key: str | None = Field(
        None,
        description="Sync only this Rithmic account key.  Omit to sync all enabled accounts.",
    )


class GradeRequest(BaseModel):
    """Request body for grading a trade from the journal route."""

    grade: str = Field(..., description="Grade: A, B, C, D, or F (empty string to clear)")
    notes: str = Field("", description="Optional free-form notes to append")


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class JournalEntryRequest(BaseModel):
    """Request body for saving a daily journal entry."""

    trade_date: str = Field(
        ...,
        description="Date string in YYYY-MM-DD format",
    )
    account_size: int = Field(
        150000,
        description="Account size: 50000, 100000, or 150000",
    )
    gross_pnl: float = Field(0.0, description="Gross P&L for the day")
    net_pnl: float = Field(0.0, description="Net P&L after commissions")
    commissions: float = Field(0.0, description="Total commissions paid")
    num_contracts: int = Field(0, description="Total contracts traded")
    instruments: str = Field("", description="Comma-separated instrument names")
    notes: str = Field("", description="Free-form notes about the trading day")
    tags: str = Field("", description="Comma-separated tags (e.g. 'orb,london,high-vol')")


class JournalEntryResponse(BaseModel):
    """Response after saving a journal entry."""

    status: str
    trade_date: str
    net_pnl: float
    timestamp: str


class JournalStatsResponse(BaseModel):
    """Aggregated journal statistics.

    Field names match the dict returned by models.get_journal_stats().
    """

    total_days: int = 0
    win_days: int = 0
    loss_days: int = 0
    break_even_days: int = 0
    win_rate: float = 0.0
    total_net: float = 0.0
    total_gross: float = 0.0
    total_commissions: float = 0.0
    avg_daily_net: float = 0.0
    best_day: float = 0.0
    worst_day: float = 0.0
    current_streak: int = 0


# ---------------------------------------------------------------------------
# Known tag categories (used for colour coding in the UI)
# ---------------------------------------------------------------------------

_TAG_COLORS: dict[str, str] = {
    # session tags
    "orb": "#60a5fa",
    "london": "#34d399",
    "us": "#4ade80",
    "pre-market": "#c084fc",
    "overnight": "#818cf8",
    "cme": "#2dd4bf",
    # quality tags
    "high-vol": "#fbbf24",
    "low-vol": "#a1a1aa",
    "trend": "#4ade80",
    "choppy": "#f87171",
    "news": "#fb923c",
    # outcome tags
    "runner": "#22c55e",
    "stopped-out": "#ef4444",
    "scratched": "#a1a1aa",
    "max-loss": "#dc2626",
    "max-profit": "#16a34a",
    # strategy tags
    "pdr": "#60a5fa",
    "ib": "#a78bfa",
    "consolidation": "#fb923c",
    "cnn-signal": "#e879f9",
}

_DEFAULT_TAG_COLOR = "#71717a"


def _tag_color(tag: str) -> str:
    """Return the display colour for a tag string."""
    return _TAG_COLORS.get(tag.lower().strip(), _DEFAULT_TAG_COLOR)


def _render_tag_pill(tag: str, active: bool = True, clickable: bool = False, target: str = "") -> str:
    """Render a single tag as a coloured pill span."""
    t = tag.strip()
    if not t:
        return ""
    color = _tag_color(t)
    opacity = "1" if active else "0.45"
    extra = ""
    if clickable and target:
        extra = (
            f' style="cursor:pointer;opacity:{opacity};font-size:9px;padding:1px 6px;border-radius:9999px;'
            f"background:{color}22;color:{color};border:1px solid {color}55;display:inline-block;"
            f'margin:1px;white-space:nowrap"'
            f' hx-get="/journal/html?tag={t}"'
            f' hx-target="{target}"'
            f' hx-swap="innerHTML"'
        )
    else:
        extra = (
            f' style="opacity:{opacity};font-size:9px;padding:1px 6px;border-radius:9999px;'
            f"background:{color}22;color:{color};border:1px solid {color}55;display:inline-block;"
            f'margin:1px;white-space:nowrap"'
        )
    return f"<span{extra}>{t}</span>"


def _parse_tags(raw: str) -> list[str]:
    """Split a comma-separated tag string into a cleaned list."""
    if not raw:
        return []
    return [t.strip().lower() for t in raw.split(",") if t.strip()]


# ---------------------------------------------------------------------------
# HTML panel renderer
# ---------------------------------------------------------------------------


def _render_journal_panel(
    entries: list[dict],
    stats: dict,
    active_tag: str = "",
    edit_date: str = "",
    limit: int = 30,
) -> str:
    """Render the full journal panel HTML (stats + filter bar + table)."""

    now_str = datetime.now(tz=_EST).strftime("%I:%M %p ET")

    # ── Collect all tags across entries for the filter bar ──────────────────
    all_tags: dict[str, int] = {}
    for e in entries:
        for t in _parse_tags(str(e.get("tags", ""))):
            all_tags[t] = all_tags.get(t, 0) + 1

    # ── Stats row ───────────────────────────────────────────────────────────
    total_net = stats.get("total_net", 0.0)
    win_rate = stats.get("win_rate", 0.0)
    streak = stats.get("current_streak", 0)
    avg_daily = stats.get("avg_daily_net", 0.0)
    total_days = stats.get("total_days", 0)
    best_day = stats.get("best_day", 0.0)
    worst_day = stats.get("worst_day", 0.0)

    net_color = "#22c55e" if total_net >= 0 else "#ef4444"
    wr_color = "#22c55e" if win_rate >= 50 else "#f87171"
    streak_color = "#22c55e" if streak > 0 else ("#ef4444" if streak < 0 else "#a1a1aa")
    streak_str = f"+{streak}W" if streak > 0 else (f"{streak}L" if streak < 0 else "—")
    avg_color = "#22c55e" if avg_daily >= 0 else "#ef4444"

    stats_html = f"""
    <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:4px;margin-bottom:8px">
        <div style="background:var(--bg-panel-inner,rgba(39,39,42,0.4));border-radius:5px;padding:5px 6px;text-align:center">
            <div style="font-size:8px;color:var(--text-faint)">Total Net</div>
            <div style="font-size:11px;font-family:monospace;color:{net_color};font-weight:700">
                {"+" if total_net >= 0 else ""}${total_net:,.0f}
            </div>
        </div>
        <div style="background:var(--bg-panel-inner,rgba(39,39,42,0.4));border-radius:5px;padding:5px 6px;text-align:center">
            <div style="font-size:8px;color:var(--text-faint)">Win Rate</div>
            <div style="font-size:11px;font-family:monospace;color:{wr_color};font-weight:700">{win_rate:.1f}%</div>
        </div>
        <div style="background:var(--bg-panel-inner,rgba(39,39,42,0.4));border-radius:5px;padding:5px 6px;text-align:center">
            <div style="font-size:8px;color:var(--text-faint)">Streak</div>
            <div style="font-size:11px;font-family:monospace;color:{streak_color};font-weight:700">{streak_str}</div>
        </div>
        <div style="background:var(--bg-panel-inner,rgba(39,39,42,0.4));border-radius:5px;padding:5px 6px;text-align:center">
            <div style="font-size:8px;color:var(--text-faint)">Avg/Day</div>
            <div style="font-size:11px;font-family:monospace;color:{avg_color}">
                {"+" if avg_daily >= 0 else ""}${avg_daily:,.0f}
            </div>
        </div>
        <div style="background:var(--bg-panel-inner,rgba(39,39,42,0.4));border-radius:5px;padding:5px 6px;text-align:center">
            <div style="font-size:8px;color:var(--text-faint)">Best Day</div>
            <div style="font-size:11px;font-family:monospace;color:#22c55e">+${best_day:,.0f}</div>
        </div>
        <div style="background:var(--bg-panel-inner,rgba(39,39,42,0.4));border-radius:5px;padding:5px 6px;text-align:center">
            <div style="font-size:8px;color:var(--text-faint)">Worst Day</div>
            <div style="font-size:11px;font-family:monospace;color:#ef4444">${worst_day:,.0f}</div>
        </div>
    </div>"""

    # ── Tag filter bar ───────────────────────────────────────────────────────
    filter_pills = ""
    # "All" pill
    all_active = not active_tag
    all_bg = "rgba(167,139,250,0.2)" if all_active else "transparent"
    all_border = "rgba(167,139,250,0.6)" if all_active else "rgba(113,113,122,0.4)"
    filter_pills += (
        f'<span style="cursor:pointer;font-size:9px;padding:1px 7px;border-radius:9999px;'
        f'background:{all_bg};color:#a78bfa;border:1px solid {all_border};display:inline-block;margin:1px;white-space:nowrap"'
        f' hx-get="/journal/html?limit={limit}"'
        f' hx-target="#journal-panel-inner"'
        f' hx-swap="innerHTML">All ({total_days})</span>'
    )
    for tag, cnt in sorted(all_tags.items(), key=lambda x: -x[1]):
        is_active = tag == active_tag
        color = _tag_color(tag)
        bg = f"{color}33" if is_active else "transparent"
        border = f"{color}88" if is_active else f"{color}44"
        filter_pills += (
            f'<span style="cursor:pointer;font-size:9px;padding:1px 6px;border-radius:9999px;'
            f'background:{bg};color:{color};border:1px solid {border};display:inline-block;margin:1px;white-space:nowrap"'
            f' hx-get="/journal/html?tag={tag}&limit={limit}"'
            f' hx-target="#journal-panel-inner"'
            f' hx-swap="innerHTML">{tag} ({cnt})</span>'
        )

    # ── Entry table ──────────────────────────────────────────────────────────
    rows_html = ""
    for entry in entries:
        d = str(entry.get("trade_date", ""))
        net = float(entry.get("net_pnl", 0.0))
        gross = float(entry.get("gross_pnl", 0.0))
        comm = float(entry.get("commissions", 0.0))
        instr = str(entry.get("instruments", ""))
        notes = str(entry.get("notes", ""))
        tags_raw = str(entry.get("tags", ""))
        num_contracts = int(entry.get("num_contracts", 0))
        entry.get("id", "")

        net_c = "#22c55e" if net > 0 else ("#ef4444" if net < 0 else "#a1a1aa")
        net_sign = "+" if net > 0 else ""
        net_str = f"{net_sign}${net:,.2f}"

        # Format date nicely
        try:
            dt = datetime.strptime(d, "%Y-%m-%d")
            date_display = dt.strftime("%b %d")
            weekday = dt.strftime("%a")
        except Exception:
            date_display = d
            weekday = ""

        # Tag pills
        tag_pills_html = "".join(_render_tag_pill(t) for t in _parse_tags(tags_raw))

        # Inline edit row (shown when edit_date matches)
        is_editing = d == edit_date

        if is_editing:
            # Render an inline edit form
            rows_html += f"""
            <tr id="journal-row-{d}" style="background:rgba(167,139,250,0.08);border-bottom:1px solid var(--border-subtle,#27272a)">
                <td colspan="7" style="padding:6px 4px">
                    <form hx-post="/journal/save"
                          hx-target="#journal-panel-inner"
                          hx-swap="innerHTML"
                          style="display:grid;grid-template-columns:repeat(2,1fr);gap:4px">
                        <input type="hidden" name="trade_date" value="{d}">
                        <div>
                            <label style="font-size:8px;color:var(--text-faint)">Date</label>
                            <input type="date" name="trade_date_display" value="{d}" readonly
                                   style="width:100%;font-size:10px;background:var(--bg-input,#27272a);
                                          color:var(--text-primary);border:1px solid var(--border-panel,#3f3f46);
                                          border-radius:3px;padding:2px 4px">
                        </div>
                        <div>
                            <label style="font-size:8px;color:var(--text-faint)">Net P&amp;L ($)</label>
                            <input type="number" name="net_pnl" value="{net:.2f}" step="0.01"
                                   style="width:100%;font-size:10px;background:var(--bg-input,#27272a);
                                          color:var(--text-primary);border:1px solid var(--border-panel,#3f3f46);
                                          border-radius:3px;padding:2px 4px">
                        </div>
                        <div>
                            <label style="font-size:8px;color:var(--text-faint)">Gross P&amp;L ($)</label>
                            <input type="number" name="gross_pnl" value="{gross:.2f}" step="0.01"
                                   style="width:100%;font-size:10px;background:var(--bg-input,#27272a);
                                          color:var(--text-primary);border:1px solid var(--border-panel,#3f3f46);
                                          border-radius:3px;padding:2px 4px">
                        </div>
                        <div>
                            <label style="font-size:8px;color:var(--text-faint)">Contracts</label>
                            <input type="number" name="num_contracts" value="{num_contracts}" min="0"
                                   style="width:100%;font-size:10px;background:var(--bg-input,#27272a);
                                          color:var(--text-primary);border:1px solid var(--border-panel,#3f3f46);
                                          border-radius:3px;padding:2px 4px">
                        </div>
                        <div>
                            <label style="font-size:8px;color:var(--text-faint)">Instruments</label>
                            <input type="text" name="instruments" value="{instr}"
                                   placeholder="MES, MGC, ..."
                                   style="width:100%;font-size:10px;background:var(--bg-input,#27272a);
                                          color:var(--text-primary);border:1px solid var(--border-panel,#3f3f46);
                                          border-radius:3px;padding:2px 4px">
                        </div>
                        <div>
                            <label style="font-size:8px;color:var(--text-faint)">Tags (comma-separated)</label>
                            <input type="text" name="tags" value="{tags_raw}"
                                   placeholder="orb, london, trend, ..."
                                   style="width:100%;font-size:10px;background:var(--bg-input,#27272a);
                                          color:var(--text-primary);border:1px solid var(--border-panel,#3f3f46);
                                          border-radius:3px;padding:2px 4px">
                        </div>
                        <div style="grid-column:span 2">
                            <label style="font-size:8px;color:var(--text-faint)">Notes</label>
                            <textarea name="notes" rows="2"
                                      style="width:100%;font-size:10px;background:var(--bg-input,#27272a);
                                             color:var(--text-primary);border:1px solid var(--border-panel,#3f3f46);
                                             border-radius:3px;padding:2px 4px;resize:vertical">{notes}</textarea>
                        </div>
                        <div style="grid-column:span 2;display:flex;gap:4px;justify-content:flex-end">
                            <button type="submit"
                                    style="font-size:9px;padding:2px 8px;background:#22c55e;color:#fff;
                                           border:none;border-radius:3px;cursor:pointer">
                                💾 Save
                            </button>
                            <button type="button"
                                    hx-get="/journal/html?limit={limit}"
                                    hx-target="#journal-panel-inner"
                                    hx-swap="innerHTML"
                                    style="font-size:9px;padding:2px 8px;background:var(--bg-input,#27272a);
                                           color:var(--text-muted);border:1px solid var(--border-panel,#3f3f46);
                                           border-radius:3px;cursor:pointer">
                                ✕ Cancel
                            </button>
                        </div>
                    </form>
                </td>
            </tr>"""
        else:
            # Normal display row
            short_notes = notes[:60] + "…" if len(notes) > 60 else notes
            instr_display = instr[:20] + "…" if len(instr) > 20 else instr

            rows_html += f"""
            <tr id="journal-row-{d}"
                style="border-bottom:1px solid var(--border-subtle,#27272a);
                       transition:background 0.1s"
                onmouseover="this.style.background='rgba(255,255,255,0.03)'"
                onmouseout="this.style.background=''">
                <td style="padding:4px 3px;white-space:nowrap">
                    <div style="font-size:10px;color:var(--text-primary)">{date_display}</div>
                    <div style="font-size:8px;color:var(--text-faint)">{weekday}</div>
                </td>
                <td style="padding:4px 3px;text-align:right;font-family:monospace">
                    <span style="font-size:11px;font-weight:700;color:{net_c}">{net_str}</span>
                </td>
                <td style="padding:4px 3px;text-align:right;font-family:monospace">
                    <span style="font-size:9px;color:var(--text-muted)">${gross:,.0f}</span>
                    <div style="font-size:8px;color:var(--text-faint)">-${comm:,.0f} comm</div>
                </td>
                <td style="padding:4px 3px">
                    <span style="font-size:9px;color:var(--text-muted)">{instr_display or "—"}</span>
                </td>
                <td style="padding:4px 3px">
                    <div style="max-width:120px;overflow:hidden">{tag_pills_html}</div>
                </td>
                <td style="padding:4px 3px;max-width:140px">
                    <span style="font-size:8px;color:var(--text-faint);display:block;
                                 overflow:hidden;text-overflow:ellipsis;white-space:nowrap"
                          title="{notes}">{short_notes or "—"}</span>
                </td>
                <td style="padding:4px 3px;text-align:right">
                    <button
                        hx-get="/journal/html?edit={d}&limit={limit}"
                        hx-target="#journal-panel-inner"
                        hx-swap="innerHTML"
                        style="font-size:8px;padding:1px 6px;background:var(--bg-input,#27272a);
                               color:var(--text-muted);border:1px solid var(--border-panel,#3f3f46);
                               border-radius:3px;cursor:pointer"
                        title="Edit this entry">
                        ✎
                    </button>
                </td>
            </tr>"""

    if not rows_html:
        rows_html = """
        <tr>
            <td colspan="7" style="padding:16px;text-align:center;color:var(--text-faint);font-size:11px">
                No journal entries found
                {filter_note}
            </td>
        </tr>""".replace(
            "{filter_note}",
            f" for tag <em>{active_tag}</em>" if active_tag else "",
        )

    # ── Limit selector ───────────────────────────────────────────────────────
    limit_opts = "".join(
        f'<option value="{n}" {"selected" if n == limit else ""}>{n} days</option>' for n in [14, 30, 60, 90, 180]
    )

    # ── Add new entry button / quick form ────────────────────────────────────
    today_str = date.today().isoformat()
    quick_add_form = f"""
    <details style="margin-top:8px">
        <summary style="cursor:pointer;font-size:9px;color:var(--text-muted);
                        padding:3px 0;list-style:none;display:flex;align-items:center;gap:4px">
            <span>▶</span>
            <span>+ Add Today's Entry</span>
        </summary>
        <form hx-post="/journal/save"
              hx-target="#journal-panel-inner"
              hx-swap="innerHTML"
              style="display:grid;grid-template-columns:repeat(2,1fr);gap:4px;
                     margin-top:6px;padding:8px;
                     background:var(--bg-panel-inner,rgba(39,39,42,0.4));
                     border-radius:5px;border:1px solid var(--border-panel,#3f3f46)">
            <input type="hidden" name="trade_date" value="{today_str}">
            <div>
                <label style="font-size:8px;color:var(--text-faint)">Net P&amp;L ($)</label>
                <input type="number" name="net_pnl" value="0" step="0.01"
                       style="width:100%;font-size:10px;background:var(--bg-input,#27272a);
                              color:var(--text-primary);border:1px solid var(--border-panel,#3f3f46);
                              border-radius:3px;padding:2px 4px">
            </div>
            <div>
                <label style="font-size:8px;color:var(--text-faint)">Gross P&amp;L ($)</label>
                <input type="number" name="gross_pnl" value="0" step="0.01"
                       style="width:100%;font-size:10px;background:var(--bg-input,#27272a);
                              color:var(--text-primary);border:1px solid var(--border-panel,#3f3f46);
                              border-radius:3px;padding:2px 4px">
            </div>
            <div>
                <label style="font-size:8px;color:var(--text-faint)">Contracts</label>
                <input type="number" name="num_contracts" value="0" min="0"
                       style="width:100%;font-size:10px;background:var(--bg-input,#27272a);
                              color:var(--text-primary);border:1px solid var(--border-panel,#3f3f46);
                              border-radius:3px;padding:2px 4px">
            </div>
            <div>
                <label style="font-size:8px;color:var(--text-faint)">Instruments</label>
                <input type="text" name="instruments" placeholder="MES, MGC, ..."
                       style="width:100%;font-size:10px;background:var(--bg-input,#27272a);
                              color:var(--text-primary);border:1px solid var(--border-panel,#3f3f46);
                              border-radius:3px;padding:2px 4px">
            </div>
            <div style="grid-column:span 2">
                <label style="font-size:8px;color:var(--text-faint)">Tags (comma-separated)</label>
                <input type="text" name="tags" placeholder="orb, london, trend, ..."
                       style="width:100%;font-size:10px;background:var(--bg-input,#27272a);
                              color:var(--text-primary);border:1px solid var(--border-panel,#3f3f46);
                              border-radius:3px;padding:2px 4px">
            </div>
            <div style="grid-column:span 2">
                <label style="font-size:8px;color:var(--text-faint)">Notes</label>
                <textarea name="notes" rows="2"
                          style="width:100%;font-size:10px;background:var(--bg-input,#27272a);
                                 color:var(--text-primary);border:1px solid var(--border-panel,#3f3f46);
                                 border-radius:3px;padding:2px 4px;resize:vertical"></textarea>
            </div>
            <div style="grid-column:span 2;text-align:right">
                <button type="submit"
                        style="font-size:9px;padding:3px 10px;background:#22c55e;color:#fff;
                               border:none;border-radius:3px;cursor:pointer">
                    💾 Save Entry
                </button>
            </div>
        </form>
    </details>"""

    # ── Tag legend ────────────────────────────────────────────────────────────
    known_tags_html = "".join(
        f'<span style="font-size:8px;padding:1px 5px;border-radius:9999px;'
        f'background:{c}22;color:{c};border:1px solid {c}44;display:inline-block;margin:1px">{t}</span>'
        for t, c in list(_TAG_COLORS.items())[:12]
    )

    return f"""
<div id="journal-panel-inner">
    <!-- Header row -->
    <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:6px">
        <div style="font-size:8px;color:var(--text-faint)">{total_days} days recorded · {now_str}</div>
        <select style="font-size:9px;background:var(--bg-input,#27272a);color:var(--text-secondary);
                       border:1px solid var(--border-panel,#3f3f46);border-radius:3px;padding:1px 3px"
                hx-get="/journal/html{("?tag=" + active_tag) if active_tag else ""}"
                hx-trigger="change"
                hx-target="#journal-panel-inner"
                hx-swap="innerHTML"
                name="limit">
            {limit_opts}
        </select>
    </div>

    <!-- Stats grid -->
    {stats_html}

    <!-- Tag filter pills -->
    <div style="margin-bottom:6px;line-height:1.8">
        <div style="font-size:8px;color:var(--text-faint);margin-bottom:2px">Filter by tag:</div>
        {filter_pills}
    </div>

    <!-- Entry table -->
    <div style="overflow-x:auto;max-height:380px;overflow-y:auto">
        <table style="width:100%;border-collapse:collapse;font-size:10px">
            <thead>
                <tr style="border-bottom:2px solid var(--border-panel,#3f3f46)">
                    <th style="padding:3px;text-align:left;font-size:8px;color:var(--text-faint);
                               font-weight:600;white-space:nowrap">Date</th>
                    <th style="padding:3px;text-align:right;font-size:8px;color:var(--text-faint);
                               font-weight:600">Net P&amp;L</th>
                    <th style="padding:3px;text-align:right;font-size:8px;color:var(--text-faint);
                               font-weight:600">Gross / Comm</th>
                    <th style="padding:3px;text-align:left;font-size:8px;color:var(--text-faint);
                               font-weight:600">Instruments</th>
                    <th style="padding:3px;text-align:left;font-size:8px;color:var(--text-faint);
                               font-weight:600">Tags</th>
                    <th style="padding:3px;text-align:left;font-size:8px;color:var(--text-faint);
                               font-weight:600">Notes</th>
                    <th style="padding:3px;width:28px"></th>
                </tr>
            </thead>
            <tbody>
                {rows_html}
            </tbody>
        </table>
    </div>

    <!-- Quick add form -->
    {quick_add_form}

    <!-- Tag legend (collapsed by default) -->
    <details style="margin-top:6px">
        <summary style="cursor:pointer;font-size:8px;color:var(--text-faint);list-style:none;
                        display:flex;align-items:center;gap:3px">
            <span>▶</span> Tag legend
        </summary>
        <div style="margin-top:4px;line-height:2">{known_tags_html}</div>
    </details>
</div>"""


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/save", response_model=JournalEntryResponse)
def save_journal_entry(entry: JournalEntryRequest):
    """Save or update a daily journal entry.

    If an entry for the given date already exists, it will be updated
    (SQLite UPSERT via INSERT OR REPLACE on the unique trade_date column).
    Tags are stored as a comma-separated string in the notes-adjacent
    ``tags`` column (added via migration if absent).
    """
    try:
        _ensure_tags_column()
        save_daily_journal(
            trade_date=entry.trade_date,
            account_size=entry.account_size,
            gross_pnl=entry.gross_pnl,
            net_pnl=entry.net_pnl,
            num_contracts=entry.num_contracts,
            instruments=entry.instruments,
            notes=entry.notes,
        )
        # Persist tags separately (tags column may not exist in models.save_daily_journal)
        _update_tags(entry.trade_date, entry.tags)
        return JournalEntryResponse(
            status="saved",
            trade_date=entry.trade_date,
            net_pnl=entry.net_pnl,
            timestamp=datetime.now(tz=_EST).isoformat(),
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to save journal: {exc}") from exc


@router.post("/save", response_class=HTMLResponse, include_in_schema=False)
def save_journal_entry_htmx(
    trade_date: str = "",
    net_pnl: float = 0.0,
    gross_pnl: float = 0.0,
    num_contracts: int = 0,
    instruments: str = "",
    notes: str = "",
    tags: str = "",
    limit: int = 30,
):
    """HTMX form handler — saves an entry and returns the refreshed panel HTML.

    This endpoint accepts form-encoded data from the inline edit form
    and the quick-add form, then returns the full refreshed panel.
    """
    try:
        _ensure_tags_column()
        if not trade_date:
            trade_date = date.today().isoformat()
        save_daily_journal(
            trade_date=trade_date,
            account_size=150000,
            gross_pnl=gross_pnl,
            net_pnl=net_pnl,
            num_contracts=num_contracts,
            instruments=instruments,
            notes=notes,
        )
        _update_tags(trade_date, tags)
    except Exception as exc:
        return HTMLResponse(content=f'<div style="color:#ef4444;font-size:10px;padding:8px">Save failed: {exc}</div>')

    # Re-render the panel
    try:
        df = get_daily_journal(limit=limit)
        stats = get_journal_stats()
        entries = df.to_dict(orient="records") if not df.empty else []
        entries = _inject_tags(entries)
        html = _render_journal_panel(entries, stats, limit=limit)
        return HTMLResponse(content=html)
    except Exception as exc:
        return HTMLResponse(content=f'<div style="color:#ef4444;font-size:10px;padding:8px">Render failed: {exc}</div>')


@router.get("/html", response_class=HTMLResponse)
def get_journal_html(
    limit: int = Query(default=30, ge=7, le=365, description="Number of recent entries to show"),
    tag: str = Query(default="", description="Filter entries by tag"),
    edit: str = Query(default="", description="Date to open in inline edit mode (YYYY-MM-DD)"),
):
    """Return the journal panel as an HTMX-swappable HTML fragment.

    Supports:
      - ``limit``: number of recent entries to show
      - ``tag``: filter to entries containing this tag
      - ``edit``: open a specific date row in inline edit mode
    """
    try:
        _ensure_tags_column()
        df = get_daily_journal(limit=limit)
        stats = get_journal_stats()

        entries: list[dict] = df.to_dict(orient="records") if not df.empty else []
        entries = _inject_tags(entries)

        # Apply tag filter
        if tag:
            entries = [e for e in entries if tag in _parse_tags(str(e.get("tags", "")))]

        html = _render_journal_panel(entries, stats, active_tag=tag, edit_date=edit, limit=limit)
        return HTMLResponse(content=html)
    except Exception as exc:
        return HTMLResponse(content=f'<div style="color:#ef4444;font-size:10px;padding:8px">Journal error: {exc}</div>')


# ---------------------------------------------------------------------------
# Standalone full-page journal view
# ---------------------------------------------------------------------------

_JOURNAL_BODY = """\
<!-- Page header -->
<div style="display:flex;align-items:center;justify-content:space-between;
            margin-bottom:1.25rem;padding-bottom:.75rem;
            border-bottom:1px solid var(--border-subtle)">
  <div>
    <h1 style="font-size:1.25rem;font-weight:700;color:var(--text-primary);
               letter-spacing:-.02em;margin-bottom:.15rem">
      📓 Trade Journal
    </h1>
    <p style="font-size:.75rem;color:var(--text-muted)">
      Daily P&amp;L log — track performance, tag setups, and review notes
    </p>
  </div>
  <a href="/"
     style="font-size:.75rem;color:var(--text-muted);text-decoration:none;
            padding:4px 10px;border:1px solid var(--border-panel);border-radius:6px;
            transition:color .12s,border-color .12s"
     onmouseover="this.style.color='var(--text-primary)';this.style.borderColor='var(--text-muted)'"
     onmouseout="this.style.color='var(--text-muted)';this.style.borderColor='var(--border-panel)'">
    ← Dashboard
  </a>
</div>

<!-- Scale up the compact widget fonts for standalone page use.
     All sizing inside #journal-page-wrapper is bumped up ~2px relative to
     the dashboard panel which was designed for a narrow sidebar column. -->
<style>
  /* Base font bump — the panel uses unitless pixel values in inline styles;
     we override key structural elements here rather than fighting specificity
     on every individual inline style. */
  #journal-page-wrapper { font-size: 13px; }
  #journal-page-wrapper table { font-size: 12px; }
  #journal-page-wrapper td   { padding: 6px 8px; font-size: 12px; }
  #journal-page-wrapper th   { padding: 5px 8px; font-size: 10px; }

  /* Stats grid tiles */
  #journal-page-wrapper [style*="font-size:11px;font-family:monospace"] { font-size: 14px !important; }
  #journal-page-wrapper [style*="font-size:11px;font-weight:700"]       { font-size: 14px !important; }

  /* Tag pills and meta labels */
  #journal-page-wrapper [style*="font-size:8px;color:var(--text-faint)"] { font-size: 11px !important; }
  #journal-page-wrapper [style*="font-size:9px;padding:1px"]             { font-size: 10px !important; }

  /* Expand the entry table scroll area to use the full viewport */
  #journal-page-wrapper [style*="max-height:380px"] { max-height: 55vh !important; }

  /* Widen the notes / tag columns that were capped for the sidebar */
  #journal-page-wrapper [style*="max-width:120px"] { max-width: 220px !important; }
  #journal-page-wrapper [style*="max-width:140px"] { max-width: 300px !important; }

  /* Quick-add / legend summary toggles */
  #journal-page-wrapper details > summary { font-size: 12px !important; }

  /* Input fields in the quick-add and inline edit forms */
  #journal-page-wrapper input,
  #journal-page-wrapper textarea,
  #journal-page-wrapper select { font-size: 12px !important; }
</style>

<!-- Page-level tab bar -->
<div style="display:flex;gap:4px;margin-bottom:1rem;border-bottom:1px solid var(--border-subtle);
            padding-bottom:.5rem">
  <button id="jp-tab-daily"
          onclick="jpShowTab('daily')"
          style="padding:5px 14px;border-radius:7px 7px 0 0;border:1px solid transparent;
                 background:var(--bg-panel);color:var(--text-primary);font-weight:700;
                 font-size:.78rem;cursor:pointer;font-family:inherit;
                 border-color:var(--border-panel);border-bottom-color:var(--bg-panel)">
    📓 Daily Log
  </button>
  <button id="jp-tab-trades"
          onclick="jpShowTab('trades')"
          style="padding:5px 14px;border-radius:7px 7px 0 0;border:1px solid transparent;
                 background:transparent;color:var(--text-muted);font-weight:500;
                 font-size:.78rem;cursor:pointer;font-family:inherit">
    📋 Trade Review
  </button>
</div>

<!-- Journal content wrapper — single full-width panel -->
<div id="journal-page-wrapper"
     style="background:var(--bg-panel);border:1px solid var(--border-panel);
            border-radius:10px;padding:1.25rem;max-width:1200px;
            min-height:calc(100vh - 220px)">

  <!-- Daily log tab -->
  <div id="jp-section-daily">
    <!-- #journal-panel-inner is the HTMX swap target used by all inner
         interactions (tag filters, edit rows, limit selector, save form).
         The outer wrapper uses a different ID so there is no ambiguity when
         _render_journal_panel returns its own <div id="journal-panel-inner">
         and HTMX replaces the innerHTML of THIS div with that response. -->
    <div id="journal-panel-inner"
         hx-get="/journal/html?limit=60"
         hx-trigger="load"
         hx-swap="outerHTML">
      <div style="padding:3rem;text-align:center;color:var(--text-faint);font-size:.85rem">
        Loading journal&hellip;
      </div>
    </div>
  </div>

  <!-- Trade Review tab (hidden until selected) -->
  <div id="jp-section-trades" style="display:none">
    <div id="journal-trades-panel"
         hx-get="/journal/trades/html?limit=50&source=rithmic_sync"
         hx-trigger="revealed"
         hx-swap="outerHTML">
      <div style="padding:3rem;text-align:center;color:var(--text-faint);font-size:.85rem">
        Loading trade review&hellip;
      </div>
    </div>
  </div>

</div>

<script>
(function() {
  'use strict';
  function jpShowTab(name) {
    var tabs = ['daily', 'trades'];
    tabs.forEach(function(t) {
      var sec = document.getElementById('jp-section-' + t);
      var btn = document.getElementById('jp-tab-' + t);
      if (!sec || !btn) return;
      var isActive = (t === name);
      sec.style.display = isActive ? 'block' : 'none';
      if (isActive) {
        btn.style.background = 'var(--bg-panel)';
        btn.style.color = 'var(--text-primary)';
        btn.style.fontWeight = '700';
        btn.style.borderColor = 'var(--border-panel)';
        btn.style.borderBottomColor = 'var(--bg-panel)';
        // Trigger htmx load on the trades panel when first revealed
        if (t === 'trades') {
          var panel = document.getElementById('journal-trades-panel');
          if (panel && panel.getAttribute('hx-trigger') === 'revealed') {
            htmx.trigger(panel, 'revealed');
            panel.removeAttribute('hx-trigger');
          }
        }
      } else {
        btn.style.background = 'transparent';
        btn.style.color = 'var(--muted, #71717a)';
        btn.style.fontWeight = '500';
        btn.style.borderColor = 'transparent';
        btn.style.borderBottomColor = 'transparent';
      }
    });
    try { localStorage.setItem('journalTab', name); } catch(e) {}
  }
  // Restore last tab
  try {
    var last = localStorage.getItem('journalTab');
    if (last && last !== 'daily') { jpShowTab(last); }
  } catch(e) {}
  window.jpShowTab = jpShowTab;
})();
</script>
"""


@router.get("/page", response_class=HTMLResponse)
def get_journal_page():
    """Serve the standalone full-page Journal view with dark theme and nav bar.

    This is the page linked from the top nav across the dashboard, trainer,
    settings, and ORB history pages.  It renders a full HTML shell and then
    loads the HTMX journal fragment (GET /journal/html) on page load so the
    inner panel can still be refreshed/swapped by HTMX interactions.
    """
    from lib.services.data.api.dashboard import _build_page_shell

    # Extra head: widen the co-page container slightly for the journal layout
    extra_head = """<style>
.co-page { max-width: 1300px; }
</style>"""

    return HTMLResponse(
        content=_build_page_shell(
            title="Trade Journal — Ruby Futures",
            favicon_emoji="📓",
            active_path="/journal/page",
            body_content=_JOURNAL_BODY,
            extra_head=extra_head,
        )
    )


@router.get("/entries")
def get_journal_entries(
    limit: int = Query(30, ge=1, le=365, description="Number of recent entries"),
    account_size: int | None = Query(None, description="Filter by account size (50000, 100000, 150000)"),
    tag: str | None = Query(None, description="Filter by tag string"),
):
    """Retrieve recent daily journal entries.

    Returns a list of journal entry dicts, most recent first.
    """
    try:
        _ensure_tags_column()
        df = get_daily_journal(limit=limit)

        # Convert DataFrame to list of dicts for JSON serialization
        if hasattr(df, "to_dict"):
            entries = df.to_dict(orient="records") if not df.empty else []
        else:
            entries = list(df) if len(df) > 0 else []

        entries = _inject_tags(entries)

        # Optional filter by account size
        if account_size is not None:
            entries = [e for e in entries if e.get("account_size") == account_size]

        # Optional filter by tag
        if tag:
            entries = [e for e in entries if tag in _parse_tags(str(e.get("tags", "")))]

        return {
            "entries": entries,
            "count": len(entries),
            "timestamp": datetime.now(tz=_EST).isoformat(),
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve journal: {exc}") from exc


@router.get("/stats", response_model=JournalStatsResponse)
def get_stats(
    account_size: int | None = Query(None, description="Filter stats by account size"),
):
    """Get aggregated journal statistics.

    Computes win rate, streaks, averages, and totals across
    all recorded journal entries.
    """
    try:
        stats = get_journal_stats()
        return JournalStatsResponse(**stats)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to compute journal stats: {exc}") from exc


@router.get("/today")
def get_today_entry():
    """Get today's journal entry if it exists.

    Convenience endpoint that checks if a journal entry has already
    been saved for the current trading day.
    """
    today_str = date.today().strftime("%Y-%m-%d")
    try:
        _ensure_tags_column()
        df = get_daily_journal(limit=1)

        # Convert DataFrame to list of dicts for iteration
        if hasattr(df, "to_dict"):
            entries = df.to_dict(orient="records") if not df.empty else []
        else:
            entries = list(df) if len(df) > 0 else []

        entries = _inject_tags(entries)

        for entry in entries:
            if entry.get("trade_date") == today_str:
                return {
                    "exists": True,
                    "entry": entry,
                    "timestamp": datetime.now(tz=_EST).isoformat(),
                }
        return {
            "exists": False,
            "entry": None,
            "trade_date": today_str,
            "timestamp": datetime.now(tz=_EST).isoformat(),
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to check today's journal: {exc}") from exc


@router.get("/trades")
def get_journal_trades(
    source: str | None = Query(None, description="Filter by source: 'manual' or 'rithmic_sync'"),
    account: str | None = Query(None, description="Filter by account key (matched against notes field)"),
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of trades to return"),
    status: str | None = Query(None, description="Filter by status: 'OPEN', 'CLOSED', 'CANCELLED'"),
    date_from: str | None = Query(None, description="Filter trades on or after this date (YYYY-MM-DD)"),
    date_to: str | None = Query(None, description="Filter trades on or before this date (YYYY-MM-DD)"),
):
    """Return recent trades from trades_v2, spanning both manual and rithmic_sync sources.

    Supports optional filtering by:
      - ``source``: 'manual' | 'rithmic_sync'
      - ``account``: account key substring matched against the notes field
      - ``status``: 'OPEN' | 'CLOSED' | 'CANCELLED'
      - ``limit``: maximum rows to return (default 100)

    This endpoint feeds the journal UI's account filter and trade review panel.
    """
    try:
        conn = _get_conn()

        conditions: list[str] = []
        params: list = []

        if source:
            conditions.append("source = ?")
            params.append(source)

        if account:
            conditions.append("notes LIKE ?")
            params.append(f"%{account}%")

        if status:
            conditions.append("status = ?")
            params.append(status.upper())

        if date_from:
            conditions.append("created_at >= ?")
            params.append(date_from)

        if date_to:
            conditions.append("created_at <= ?")
            params.append(f"{date_to} 23:59:59")

        where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        sql = f"""
            SELECT id, created_at, account_size, asset, direction,
                   entry, sl, tp, contracts, status,
                   close_price, close_time, pnl, rr,
                   notes, strategy,
                   COALESCE(grade, '')  AS grade,
                   COALESCE(source, 'manual') AS source
            FROM trades_v2
            {where_clause}
            ORDER BY created_at DESC
            LIMIT ?
        """
        params.append(limit)

        rows = conn.execute(sql, tuple(params)).fetchall()
        conn.close()

        trades = []
        for row in rows:
            if hasattr(row, "keys"):
                d = {k: row[k] for k in row}
            elif hasattr(row, "_data"):
                d = dict(row._data)
            else:
                d = dict(row)
            trades.append(d)

        return {
            "trades": trades,
            "count": len(trades),
            "filters": {
                "source": source,
                "account": account,
                "status": status,
                "limit": limit,
            },
            "timestamp": datetime.now(tz=_EST).isoformat(),
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve trades: {exc}") from exc


# ---------------------------------------------------------------------------
# Trade review HTML panel (HTMX fragment — account filter + grade UI)
# ---------------------------------------------------------------------------


@router.get("/trades/html", response_class=HTMLResponse)
def get_journal_trades_html(
    account: str | None = Query(None, description="Filter by account key"),
    source: str | None = Query(None, description="Filter: manual | rithmic_sync"),
    status: str | None = Query(None, description="Filter: OPEN | CLOSED"),
    limit: int = Query(50, ge=1, le=500),
    date_from: str | None = Query(None),
    date_to: str | None = Query(None),
):
    """Return an HTMX-swappable HTML fragment for the trade review panel.

    Includes an account filter dropdown (populated from distinct account keys
    found in the notes column of trades_v2) and a per-trade grade selector.
    """
    try:
        conn = _get_conn()

        # --- Collect known account keys from rithmic_sync trades ---
        acct_rows = conn.execute(
            "SELECT DISTINCT notes FROM trades_v2 WHERE source = 'rithmic_sync' LIMIT 200"
        ).fetchall()

        known_accounts: list[str] = []
        import re as _re

        for row in acct_rows:
            notes_val = row[0] if isinstance(row, (list, tuple)) else row.get("notes", "")
            # Extract account key pattern: "rithmic_sync:KEY" or "[KEY]"
            m = _re.search(r"rithmic_sync:([^\s\[]+)", str(notes_val))
            if m:
                ak = m.group(1).strip()
                if ak and ak not in known_accounts:
                    known_accounts.append(ak)
            m2 = _re.search(r"\[([^\]]+)\]", str(notes_val))
            if m2:
                ak2 = m2.group(1).strip()
                if ak2 and ak2 not in known_accounts:
                    known_accounts.append(ak2)

        conn.close()

        # --- Build query params to fetch trades ---
        conditions: list[str] = []
        params: list = []
        if source:
            conditions.append("source = ?")
            params.append(source)
        if account:
            conditions.append("notes LIKE ?")
            params.append(f"%{account}%")
        if status:
            conditions.append("status = ?")
            params.append(status.upper())
        if date_from:
            conditions.append("created_at >= ?")
            params.append(date_from)
        if date_to:
            conditions.append("created_at <= ?")
            params.append(f"{date_to} 23:59:59")

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        sql = f"""
            SELECT id, created_at, asset, direction, entry, close_price,
                   contracts, status, pnl, rr, strategy, notes,
                   COALESCE(grade, '') AS grade,
                   COALESCE(source, 'manual') AS source
            FROM trades_v2
            {where}
            ORDER BY created_at DESC
            LIMIT ?
        """
        params.append(limit)

        conn2 = _get_conn()
        rows = conn2.execute(sql, tuple(params)).fetchall()
        conn2.close()

        trades = []
        for row in rows:
            if hasattr(row, "keys"):
                trades.append({k: row[k] for k in row})
            elif hasattr(row, "_data"):
                trades.append(dict(row._data))
            else:
                trades.append(dict(row))

        # --- Render HTML ---
        now_str = datetime.now(tz=_EST).strftime("%I:%M %p ET")

        # Account filter dropdown
        acct_options = '<option value="">All accounts</option>'
        for ak in sorted(known_accounts):
            sel = "selected" if ak == account else ""
            acct_options += f'<option value="{ak}" {sel}>{ak}</option>'

        source_options = ""
        for sv, sl in [("", "All sources"), ("rithmic_sync", "Rithmic sync"), ("manual", "Manual")]:
            sel = "selected" if sv == (source or "") else ""
            source_options += f'<option value="{sv}" {sel}>{sl}</option>'

        status_options = ""
        for sv, sl in [("", "All statuses"), ("OPEN", "Open"), ("CLOSED", "Closed"), ("CANCELLED", "Cancelled")]:
            sel = "selected" if sv == (status or "") else ""
            status_options += f'<option value="{sv}" {sel}>{sl}</option>'

        # Build filter bar query string helper (preserves all current filters)
        def _qs(**overrides) -> str:
            base = {
                "account": account or "",
                "source": source or "",
                "status": status or "",
                "limit": str(limit),
                "date_from": date_from or "",
                "date_to": date_to or "",
            }
            base.update({k: v for k, v in overrides.items() if v is not None})
            parts = [f"{k}={v}" for k, v in base.items() if v]
            return ("?" + "&".join(parts)) if parts else ""

        # Sync status badge
        sync_status_html = ""
        try:
            from lib.services.engine.journal_sync import get_last_sync_status

            last_sync = get_last_sync_status()
            if last_sync:
                ts = last_sync.get("timestamp", "")[:16].replace("T", " ")
                written = last_sync.get("trades_written", 0)
                fills = last_sync.get("fills_retrieved", 0)
                st = last_sync.get("status", "")
                color = "#22c55e" if st == "ok" else "#fbbf24" if st == "partial" else "#ef4444"
                sync_status_html = (
                    f'<span style="font-size:8px;color:{color};margin-left:8px" title="Last sync: {ts}">'
                    f"⟳ synced {ts} — {fills} fills → {written} trades</span>"
                )
        except Exception:
            pass

        # Trade rows
        grade_colors = {"A": "#22c55e", "B": "#4ade80", "C": "#fbbf24", "D": "#fb923c", "F": "#ef4444"}

        rows_html = ""
        for t in trades:
            tid = t.get("id", "")
            asset = str(t.get("asset", ""))
            direction = str(t.get("direction", ""))
            entry = float(t.get("entry") or 0.0)
            close = t.get("close_price")
            pnl = t.get("pnl")
            contracts = int(t.get("contracts") or 1)
            tstatus = str(t.get("status", ""))
            grade = str(t.get("grade") or "")
            src = str(t.get("source") or "manual")
            created = str(t.get("created_at", ""))[:16]
            notes_val = str(t.get("notes") or "")

            dir_color = "#22c55e" if direction == "LONG" else "#ef4444"
            pnl_color = "#22c55e" if (pnl or 0) > 0 else ("#ef4444" if (pnl or 0) < 0 else "#a1a1aa")
            pnl_str = f"+${pnl:,.2f}" if (pnl or 0) > 0 else (f"${pnl:,.2f}" if pnl is not None else "—")
            close_str = f"{close:.4f}" if close is not None else "open"
            src_badge_bg = "rgba(96,165,250,0.15)" if src == "rithmic_sync" else "rgba(113,113,122,0.15)"
            src_badge_color = "#60a5fa" if src == "rithmic_sync" else "#a1a1aa"
            src_label = "sync" if src == "rithmic_sync" else "manual"

            grade_color = grade_colors.get(grade, "#71717a")
            grade_opts = "".join(
                f'<option value="{g}" {"selected" if g == grade else ""}>{g or "—"}</option>'
                for g in ["", "A", "B", "C", "D", "F"]
            )

            # Extract account key from notes for display
            import re as _re2

            acct_display = ""
            m = _re2.search(r"rithmic_sync:([^\s\[]+)", notes_val)
            if m:
                acct_display = m.group(1)[:12]
            elif "[" in notes_val:
                m2 = _re2.search(r"\[([^\]]+)\]", notes_val)
                if m2:
                    acct_display = m2.group(1)[:12]

            rows_html += f"""
            <tr style="border-bottom:1px solid var(--border-subtle,#27272a)">
                <td style="padding:4px 3px;white-space:nowrap;font-size:9px;color:var(--text-faint)">{created}</td>
                <td style="padding:4px 3px;font-size:10px;color:var(--text-primary);font-weight:600">{asset}</td>
                <td style="padding:4px 3px;font-size:9px;color:{dir_color};font-weight:700">{direction}</td>
                <td style="padding:4px 3px;text-align:right;font-size:9px;font-family:monospace">{entry:.4f}</td>
                <td style="padding:4px 3px;text-align:right;font-size:9px;font-family:monospace;color:var(--text-muted)">{close_str}</td>
                <td style="padding:4px 3px;text-align:right;font-size:10px;font-weight:700;color:{pnl_color};font-family:monospace">{pnl_str}</td>
                <td style="padding:4px 3px;text-align:center;font-size:9px;color:var(--text-muted)">{contracts}</td>
                <td style="padding:4px 3px">
                    <span style="font-size:8px;background:{src_badge_bg};color:{src_badge_color};
                                 padding:1px 5px;border-radius:9999px">{src_label}</span>
                </td>
                <td style="padding:4px 3px;font-size:8px;color:var(--text-faint)">{acct_display}</td>
                <td style="padding:4px 3px">
                    <select
                        style="font-size:9px;background:var(--bg-input,#27272a);color:{grade_color};
                               border:1px solid var(--border-panel,#3f3f46);border-radius:3px;
                               padding:1px 3px;cursor:pointer"
                        hx-post="/journal/trades/{tid}/grade"
                        hx-trigger="change"
                        hx-target="closest tr"
                        hx-swap="outerHTML"
                        name="grade">
                        {grade_opts}
                    </select>
                </td>
            </tr>"""

        if not rows_html:
            rows_html = """
            <tr>
                <td colspan="10" style="padding:16px;text-align:center;
                    color:var(--text-faint);font-size:11px">
                    No trades found — try adjusting the filters above
                </td>
            </tr>"""

        # Summary stats from visible trades
        closed_trades = [t for t in trades if t.get("status") == "CLOSED"]
        total_pnl = sum(float(t.get("pnl") or 0) for t in closed_trades)
        win_trades = sum(1 for t in closed_trades if (t.get("pnl") or 0) > 0)
        loss_trades = sum(1 for t in closed_trades if (t.get("pnl") or 0) < 0)
        win_rate = (win_trades / len(closed_trades) * 100) if closed_trades else 0.0
        pnl_color2 = "#22c55e" if total_pnl >= 0 else "#ef4444"
        wr_color = "#22c55e" if win_rate >= 50 else "#f87171"

        return HTMLResponse(
            content=f"""
<div id="journal-trades-panel">
    <!-- Header row -->
    <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:6px;flex-wrap:wrap;gap:4px">
        <div style="font-size:8px;color:var(--text-faint)">{len(trades)} trade(s) · {now_str}{sync_status_html}</div>
        <button
            hx-post="/journal/sync"
            hx-target="#journal-trades-panel"
            hx-swap="outerHTML"
            hx-include="[name='account_key']"
            style="font-size:8px;padding:2px 8px;background:rgba(96,165,250,0.15);
                   color:#60a5fa;border:1px solid rgba(96,165,250,0.3);border-radius:4px;cursor:pointer">
            ⟳ Sync Now
        </button>
    </div>

    <!-- Summary stats -->
    <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:4px;margin-bottom:8px">
        <div style="background:var(--bg-panel-inner,rgba(39,39,42,0.4));border-radius:5px;padding:4px 6px;text-align:center">
            <div style="font-size:8px;color:var(--text-faint)">Trades</div>
            <div style="font-size:12px;font-family:monospace;font-weight:700">{len(closed_trades)}</div>
        </div>
        <div style="background:var(--bg-panel-inner,rgba(39,39,42,0.4));border-radius:5px;padding:4px 6px;text-align:center">
            <div style="font-size:8px;color:var(--text-faint)">Net P&L</div>
            <div style="font-size:12px;font-family:monospace;font-weight:700;color:{pnl_color2}">
                {"+" if total_pnl >= 0 else ""}${total_pnl:,.0f}
            </div>
        </div>
        <div style="background:var(--bg-panel-inner,rgba(39,39,42,0.4));border-radius:5px;padding:4px 6px;text-align:center">
            <div style="font-size:8px;color:var(--text-faint)">Win Rate</div>
            <div style="font-size:12px;font-family:monospace;font-weight:700;color:{wr_color}">{win_rate:.1f}%</div>
        </div>
        <div style="background:var(--bg-panel-inner,rgba(39,39,42,0.4));border-radius:5px;padding:4px 6px;text-align:center">
            <div style="font-size:8px;color:var(--text-faint)">W / L</div>
            <div style="font-size:12px;font-family:monospace;font-weight:700">
                <span style="color:#22c55e">{win_trades}</span>
                <span style="color:var(--text-faint)"> / </span>
                <span style="color:#ef4444">{loss_trades}</span>
            </div>
        </div>
    </div>

    <!-- Filters -->
    <div style="display:flex;gap:4px;flex-wrap:wrap;margin-bottom:6px;align-items:center">
        <select name="account_key"
                style="font-size:9px;background:var(--bg-input,#27272a);color:var(--text-secondary);
                       border:1px solid var(--border-panel,#3f3f46);border-radius:3px;padding:1px 4px"
                hx-get="/journal/trades/html"
                hx-trigger="change"
                hx-target="#journal-trades-panel"
                hx-swap="outerHTML"
                hx-include="[name='src_filter'],[name='status_filter'],[name='limit_filter']">
            {acct_options}
        </select>
        <select name="src_filter"
                style="font-size:9px;background:var(--bg-input,#27272a);color:var(--text-secondary);
                       border:1px solid var(--border-panel,#3f3f46);border-radius:3px;padding:1px 4px"
                hx-get="/journal/trades/html"
                hx-trigger="change"
                hx-target="#journal-trades-panel"
                hx-swap="outerHTML"
                hx-include="[name='account_key'],[name='status_filter'],[name='limit_filter']">
            {source_options}
        </select>
        <select name="status_filter"
                style="font-size:9px;background:var(--bg-input,#27272a);color:var(--text-secondary);
                       border:1px solid var(--border-panel,#3f3f46);border-radius:3px;padding:1px 4px"
                hx-get="/journal/trades/html"
                hx-trigger="change"
                hx-target="#journal-trades-panel"
                hx-swap="outerHTML"
                hx-include="[name='account_key'],[name='src_filter'],[name='limit_filter']">
            {status_options}
        </select>
        <select name="limit_filter"
                style="font-size:9px;background:var(--bg-input,#27272a);color:var(--text-secondary);
                       border:1px solid var(--border-panel,#3f3f46);border-radius:3px;padding:1px 4px"
                hx-get="/journal/trades/html"
                hx-trigger="change"
                hx-target="#journal-trades-panel"
                hx-swap="outerHTML"
                hx-include="[name='account_key'],[name='src_filter'],[name='status_filter']">
            {"".join(f'<option value="{n}" {"selected" if n == limit else ""}>{n} trades</option>' for n in [25, 50, 100, 200])}
        </select>
    </div>

    <!-- Trade table -->
    <div style="overflow-x:auto;max-height:420px;overflow-y:auto">
        <table style="width:100%;border-collapse:collapse;font-size:10px">
            <thead>
                <tr style="border-bottom:2px solid var(--border-panel,#3f3f46)">
                    <th style="padding:3px;text-align:left;font-size:8px;color:var(--text-faint);font-weight:600">Time</th>
                    <th style="padding:3px;text-align:left;font-size:8px;color:var(--text-faint);font-weight:600">Symbol</th>
                    <th style="padding:3px;text-align:left;font-size:8px;color:var(--text-faint);font-weight:600">Side</th>
                    <th style="padding:3px;text-align:right;font-size:8px;color:var(--text-faint);font-weight:600">Entry</th>
                    <th style="padding:3px;text-align:right;font-size:8px;color:var(--text-faint);font-weight:600">Exit</th>
                    <th style="padding:3px;text-align:right;font-size:8px;color:var(--text-faint);font-weight:600">P&amp;L</th>
                    <th style="padding:3px;text-align:center;font-size:8px;color:var(--text-faint);font-weight:600">Qty</th>
                    <th style="padding:3px;text-align:left;font-size:8px;color:var(--text-faint);font-weight:600">Src</th>
                    <th style="padding:3px;text-align:left;font-size:8px;color:var(--text-faint);font-weight:600">Acct</th>
                    <th style="padding:3px;text-align:left;font-size:8px;color:var(--text-faint);font-weight:600">Grade</th>
                </tr>
            </thead>
            <tbody>
                {rows_html}
            </tbody>
        </table>
    </div>
</div>"""
        )

    except Exception as exc:
        return HTMLResponse(
            content=f'<div style="color:#ef4444;font-size:10px;padding:8px">Trade panel error: {exc}</div>'
        )


# ---------------------------------------------------------------------------
# Grade endpoint (journal-scoped — mirrors /trades/{id}/grade)
# ---------------------------------------------------------------------------


@router.post("/trades/{trade_id}/grade")
def journal_grade_trade(trade_id: int, grade: str = Query(..., description="Grade: A, B, C, D, F or empty")):
    """Set the quality grade on a trade from the journal UI.

    This is the HTMX target for the grade <select> in the trade review panel.
    Returns a replacement <tr> row fragment on success.

    Accepted grades: A, B, C, D, F (or empty string to clear).
    """
    valid_grades = {"A", "B", "C", "D", "F", ""}
    g = grade.strip().upper()
    if g and g not in valid_grades:
        raise HTTPException(status_code=422, detail=f"Invalid grade '{g}'. Use A, B, C, D, F or empty.")

    conn = _get_conn()
    try:
        result = conn.execute(
            "UPDATE trades_v2 SET grade = ? WHERE id = ?",
            (g, trade_id),
        )
        rowcount = getattr(result, "rowcount", None)
        if rowcount is not None and rowcount == 0:
            conn.close()
            raise HTTPException(status_code=404, detail=f"Trade {trade_id} not found")
        conn.commit()

        # Re-fetch the row to return updated <tr>
        row = conn.execute(
            """SELECT id, created_at, asset, direction, entry, close_price,
                      contracts, status, pnl, rr, strategy, notes,
                      COALESCE(grade, '') AS grade,
                      COALESCE(source, 'manual') AS source
               FROM trades_v2 WHERE id = ?""",
            (trade_id,),
        ).fetchone()
        conn.close()
    except HTTPException:
        raise
    except Exception as exc:
        conn.close()
        raise HTTPException(status_code=500, detail=f"Failed to update grade: {exc}") from exc

    if row is None:
        raise HTTPException(status_code=404, detail=f"Trade {trade_id} not found after update")

    if hasattr(row, "keys"):
        t = {k: row[k] for k in row}
    elif hasattr(row, "_data"):
        t = dict(row._data)
    else:
        t = dict(row)

    grade_colors = {"A": "#22c55e", "B": "#4ade80", "C": "#fbbf24", "D": "#fb923c", "F": "#ef4444"}
    asset = str(t.get("asset", ""))
    direction = str(t.get("direction", ""))
    entry = float(t.get("entry") or 0.0)
    close = t.get("close_price")
    pnl = t.get("pnl")
    contracts = int(t.get("contracts") or 1)
    grade_val = str(t.get("grade") or "")
    src = str(t.get("source") or "manual")
    created = str(t.get("created_at", ""))[:16]
    notes_val = str(t.get("notes") or "")

    dir_color = "#22c55e" if direction == "LONG" else "#ef4444"
    pnl_color = "#22c55e" if (pnl or 0) > 0 else ("#ef4444" if (pnl or 0) < 0 else "#a1a1aa")
    pnl_str = f"+${pnl:,.2f}" if (pnl or 0) > 0 else (f"${pnl:,.2f}" if pnl is not None else "—")
    close_str = f"{close:.4f}" if close is not None else "open"
    src_badge_bg = "rgba(96,165,250,0.15)" if src == "rithmic_sync" else "rgba(113,113,122,0.15)"
    src_badge_color = "#60a5fa" if src == "rithmic_sync" else "#a1a1aa"
    src_label = "sync" if src == "rithmic_sync" else "manual"
    grade_color = grade_colors.get(grade_val, "#71717a")

    import re as _re3

    acct_display = ""
    m = _re3.search(r"rithmic_sync:([^\s\[]+)", notes_val)
    if m:
        acct_display = m.group(1)[:12]
    elif "[" in notes_val:
        m2 = _re3.search(r"\[([^\]]+)\]", notes_val)
        if m2:
            acct_display = m2.group(1)[:12]

    grade_opts = "".join(
        f'<option value="{gv}" {"selected" if gv == grade_val else ""}>{gv or "—"}</option>'
        for gv in ["", "A", "B", "C", "D", "F"]
    )

    from fastapi.responses import HTMLResponse as _HTML

    return _HTML(
        content=f"""
    <tr style="border-bottom:1px solid var(--border-subtle,#27272a)">
        <td style="padding:4px 3px;white-space:nowrap;font-size:9px;color:var(--text-faint)">{created}</td>
        <td style="padding:4px 3px;font-size:10px;color:var(--text-primary);font-weight:600">{asset}</td>
        <td style="padding:4px 3px;font-size:9px;color:{dir_color};font-weight:700">{direction}</td>
        <td style="padding:4px 3px;text-align:right;font-size:9px;font-family:monospace">{entry:.4f}</td>
        <td style="padding:4px 3px;text-align:right;font-size:9px;font-family:monospace;color:var(--text-muted)">{close_str}</td>
        <td style="padding:4px 3px;text-align:right;font-size:10px;font-weight:700;color:{pnl_color};font-family:monospace">{pnl_str}</td>
        <td style="padding:4px 3px;text-align:center;font-size:9px;color:var(--text-muted)">{contracts}</td>
        <td style="padding:4px 3px">
            <span style="font-size:8px;background:{src_badge_bg};color:{src_badge_color};
                         padding:1px 5px;border-radius:9999px">{src_label}</span>
        </td>
        <td style="padding:4px 3px;font-size:8px;color:var(--text-faint)">{acct_display}</td>
        <td style="padding:4px 3px">
            <select
                style="font-size:9px;background:var(--bg-input,#27272a);color:{grade_color};
                       border:1px solid var(--border-panel,#3f3f46);border-radius:3px;
                       padding:1px 3px;cursor:pointer"
                hx-post="/journal/trades/{trade_id}/grade"
                hx-trigger="change"
                hx-target="closest tr"
                hx-swap="outerHTML"
                name="grade">
                {grade_opts}
            </select>
        </td>
    </tr>"""
    )


# ---------------------------------------------------------------------------
# Manual sync trigger endpoint
# ---------------------------------------------------------------------------


@router.post("/sync")
async def trigger_journal_sync(
    background_tasks: BackgroundTasks,
    req: SyncRequest | None = None,
):
    """Manually trigger a Rithmic fill → trades_v2 sync.

    Runs the sync in the background and returns immediately with a 202.
    The sync result is stored in Redis under ``journal:last_sync`` and
    can be polled via ``GET /journal/sync/status``.

    Requires ``async-rithmic`` to be installed and at least one enabled
    Rithmic account with credentials configured.
    """
    account_key = req.account_key if req else None

    async def _run_sync():
        try:
            from lib.services.engine.journal_sync import run_journal_sync

            result = await run_journal_sync(account_key=account_key)
            logger.info(
                "manual journal sync complete: fills=%d matched=%d written=%d",
                result.get("fills_retrieved", 0),
                result.get("trades_matched", 0),
                result.get("trades_written", 0),
            )
        except Exception as exc:
            logger.error("manual journal sync error: %s", exc, exc_info=True)

    background_tasks.add_task(_run_sync)

    return {
        "status": "accepted",
        "message": "Journal sync started in background",
        "account_key": account_key,
        "poll": "/journal/sync/status",
    }


@router.get("/sync/status")
def get_sync_status():
    """Return the result of the most recent journal sync from Redis.

    Returns ``null`` when no sync has been run yet this session.
    """
    try:
        from lib.services.engine.journal_sync import get_last_sync_status

        status = get_last_sync_status()
        if status is None:
            return {"status": "never_run", "message": "No sync has been run yet"}
        return status
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve sync status: {exc}") from exc


@router.get("/tags")
def get_all_tags():
    """Return all unique tags used across journal entries with usage counts."""
    try:
        _ensure_tags_column()
        df = get_daily_journal(limit=9999)
        entries = df.to_dict(orient="records") if not df.empty else []
        entries = _inject_tags(entries)

        tag_counts: dict[str, int] = {}
        for e in entries:
            for t in _parse_tags(str(e.get("tags", ""))):
                tag_counts[t] = tag_counts.get(t, 0) + 1

        return {
            "tags": dict(sorted(tag_counts.items(), key=lambda x: -x[1])),
            "count": len(tag_counts),
            "timestamp": datetime.now(tz=_EST).isoformat(),
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve tags: {exc}") from exc


# ---------------------------------------------------------------------------
# Internal helpers — tags column management
# ---------------------------------------------------------------------------

_tags_column_checked = False


def _ensure_tags_column() -> None:
    """Add a ``tags`` column to ``daily_journal`` if it does not yet exist.

    This is a safe, idempotent migration — no-op if the column is already
    present.  Runs once per process lifetime.
    """
    global _tags_column_checked
    if _tags_column_checked:
        return
    try:
        from lib.core.models import _get_conn, _is_using_postgres

        conn = _get_conn()
        if _is_using_postgres():
            # Postgres: use a DO block to avoid errors on repeated runs
            conn.execute("""
                DO $$
                BEGIN
                    IF NOT EXISTS (
                        SELECT 1 FROM information_schema.columns
                        WHERE table_name='daily_journal' AND column_name='tags'
                    ) THEN
                        ALTER TABLE daily_journal ADD COLUMN tags TEXT DEFAULT '';
                    END IF;
                END $$;
            """)
        else:
            # SQLite: check pragma first
            cols = conn.execute("PRAGMA table_info(daily_journal)").fetchall()
            col_names = [c[1] if isinstance(c, (list, tuple)) else c.get("name", "") for c in cols]
            if "tags" not in col_names:
                conn.execute("ALTER TABLE daily_journal ADD COLUMN tags TEXT DEFAULT ''")
        conn.commit()
        conn.close()
        _tags_column_checked = True
    except Exception:
        # Don't crash if migration fails — tags will just be empty
        _tags_column_checked = True


def _update_tags(trade_date: str, tags: str) -> None:
    """Write the tags string for a given trade_date row.

    Called after ``save_daily_journal`` to persist the tags field which
    is not part of the core models API.
    """
    if not trade_date:
        return
    try:
        from lib.core.models import _get_conn

        cleaned = ",".join(_parse_tags(tags))
        conn = _get_conn()
        conn.execute(
            "UPDATE daily_journal SET tags = ? WHERE trade_date = ?",
            (cleaned, trade_date),
        )
        conn.commit()
        conn.close()
    except Exception:
        pass  # Non-fatal — tags are optional


def _inject_tags(entries: list[dict]) -> list[dict]:
    """Fetch and inject the ``tags`` column values into an entry list.

    The core ``get_daily_journal()`` may not return the ``tags`` column
    if it was added after the initial schema creation.  This function
    fetches the tags for all entries in one query and merges them in.

    Returns the same list with ``tags`` populated.
    """
    if not entries:
        return entries
    try:
        from lib.core.models import _get_conn

        dates = [str(e.get("trade_date", "")) for e in entries]
        placeholders = ",".join("?" * len(dates))
        conn = _get_conn()
        rows = conn.execute(
            f"SELECT trade_date, tags FROM daily_journal WHERE trade_date IN ({placeholders})",
            dates,
        ).fetchall()
        conn.close()

        tag_map: dict[str, str] = {}
        for row in rows:
            if isinstance(row, (list, tuple)):
                tag_map[str(row[0])] = str(row[1] or "")
            else:
                tag_map[str(row.get("trade_date", ""))] = str(row.get("tags", "") or "")

        for e in entries:
            d = str(e.get("trade_date", ""))
            if "tags" not in e or not e["tags"]:
                e["tags"] = tag_map.get(d, "")
    except Exception:
        pass  # Return as-is if the query fails

    return entries
