"""
Session Report Generator — Pre & Post Trading Day Reports
============================================================
Generates daily reports to track performance and improve the system.

Pre-Session Report (generated after morning pipeline):
    - Market conditions summary
    - Focus assets and why
    - Key levels for the day
    - Risk parameters
    - Macro context (DXY, VIX, events)

Post-Session Report (generated at EOD or manual trigger):
    - Trades taken vs planned
    - P&L breakdown by trade
    - Plan adherence score
    - What worked / what didn't
    - Risk utilization
    - System improvement notes

Reports are stored in SQLite (futures_journal.db) for historical analysis.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
from dataclasses import asdict, dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger("engine.session_report")

# Database path — same DB as the journal
DB_PATH = Path(os.getenv("JOURNAL_DB_PATH", "futures_journal.db"))


# ---------------------------------------------------------------------------
# Report dataclasses
# ---------------------------------------------------------------------------


@dataclass
class PreSessionReport:
    """Pre-session analysis report."""

    report_date: str = ""
    generated_at: str = ""

    # Market conditions
    market_regime: str = ""  # "BULLISH", "BEARISH", "RANGING"
    regime_confidence: float = 0.0
    volatility_level: str = ""  # "LOW", "MEDIUM", "HIGH", "EXTREME"

    # Focus assets
    focus_assets: list[dict[str, Any]] = field(default_factory=list)
    # Each: {symbol, direction, entry_zone, targets, confidence, reason}

    # Key levels
    key_levels: list[dict[str, Any]] = field(default_factory=list)

    # Macro context
    dxy_level: float = 0.0
    vix_level: float = 0.0
    economic_events: list[dict[str, Any]] = field(default_factory=list)
    sentiment_score: float = 0.0
    sentiment_label: str = ""

    # Risk parameters
    account_size: int = 150_000
    daily_loss_limit: float = 3_300.0
    max_contracts: int = 15
    daily_profit_goal: float = 1_800.0

    # AI summary
    grok_summary: str = ""

    # Trader notes (added by the trader)
    trader_notes: str = ""

    def __post_init__(self):
        if not self.report_date:
            self.report_date = date.today().isoformat()
        if not self.generated_at:
            self.generated_at = datetime.now(UTC).isoformat()


@dataclass
class PostSessionReport:
    """Post-session trading review report."""

    report_date: str = ""
    generated_at: str = ""

    # P&L
    gross_pnl: float = 0.0
    net_pnl: float = 0.0  # After fees
    fees: float = 0.0

    # Trade stats
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    avg_winner: float = 0.0
    avg_loser: float = 0.0
    largest_winner: float = 0.0
    largest_loser: float = 0.0
    profit_factor: float = 0.0  # gross wins / gross losses

    # Plan adherence
    planned_trades: int = 0  # Trades that matched the plan
    unplanned_trades: int = 0  # Trades taken outside the plan
    plan_adherence_pct: float = 0.0  # planned / total * 100

    # Risk utilization
    max_drawdown_intraday: float = 0.0
    max_contracts_used: int = 0
    risk_budget_used_pct: float = 0.0  # % of daily loss limit

    # Individual trades
    trades: list[dict[str, Any]] = field(default_factory=list)
    # Each: {symbol, direction, entry, exit, pnl, planned, grade, reason}

    # Session quality
    session_grade: str = ""  # A, B, C, D, F
    what_worked: list[str] = field(default_factory=list)
    what_didnt_work: list[str] = field(default_factory=list)
    improvement_notes: list[str] = field(default_factory=list)

    # Goals check
    hit_daily_goal: bool = False
    hit_stretch_goal: bool = False
    profit_target_progress_pct: float = 0.0  # % of $9000 TPT target

    # Trader notes
    trader_notes: str = ""

    def __post_init__(self):
        if not self.report_date:
            self.report_date = date.today().isoformat()
        if not self.generated_at:
            self.generated_at = datetime.now(UTC).isoformat()


# ---------------------------------------------------------------------------
# Database operations
# ---------------------------------------------------------------------------


def _get_db() -> sqlite3.Connection:
    """Get a connection to the journal database."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_report_tables() -> None:
    """Create report tables if they don't exist."""
    conn = _get_db()
    try:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS pre_session_reports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                report_date TEXT NOT NULL,
                generated_at TEXT NOT NULL,
                data JSON NOT NULL,
                trader_notes TEXT DEFAULT '',
                created_at TEXT DEFAULT (datetime('now')),
                UNIQUE(report_date)
            );

            CREATE TABLE IF NOT EXISTS post_session_reports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                report_date TEXT NOT NULL,
                generated_at TEXT NOT NULL,
                data JSON NOT NULL,
                session_grade TEXT DEFAULT '',
                net_pnl REAL DEFAULT 0,
                win_rate REAL DEFAULT 0,
                plan_adherence REAL DEFAULT 0,
                trader_notes TEXT DEFAULT '',
                created_at TEXT DEFAULT (datetime('now')),
                UNIQUE(report_date)
            );

            CREATE TABLE IF NOT EXISTS daily_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                report_date TEXT NOT NULL,
                metric_name TEXT NOT NULL,
                metric_value REAL NOT NULL,
                metadata JSON DEFAULT '{}',
                created_at TEXT DEFAULT (datetime('now'))
            );

            CREATE INDEX IF NOT EXISTS idx_pre_session_date ON pre_session_reports(report_date);
            CREATE INDEX IF NOT EXISTS idx_post_session_date ON post_session_reports(report_date);
            CREATE INDEX IF NOT EXISTS idx_daily_metrics_date ON daily_metrics(report_date);
        """)
        conn.commit()
        logger.info("Session report tables initialized")
    except Exception as exc:
        logger.error("Failed to initialize report tables: %s", exc)
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def generate_pre_session_report(
    plan_data: dict[str, Any] | None = None,
    focus_assets: list[dict[str, Any]] | None = None,
    account_size: int = 150_000,
    trader_notes: str = "",
) -> PreSessionReport:
    """Generate a pre-session report from the morning pipeline output.

    Args:
        plan_data: Output from the morning pipeline (plan step)
        focus_assets: List of focus asset dicts from compute_daily_focus()
        account_size: Account size
        trader_notes: Any notes from the trader

    Returns:
        PreSessionReport instance
    """
    report = PreSessionReport(
        account_size=account_size,
        trader_notes=trader_notes,
    )

    if plan_data:
        report.market_regime = plan_data.get("regime", {}).get("label", "UNKNOWN")
        report.regime_confidence = plan_data.get("regime", {}).get("conf", 0)
        report.grok_summary = plan_data.get("grok_summary", "")
        report.key_levels = plan_data.get("levels", [])
        report.sentiment_score = plan_data.get("sentiment", {}).get("score", 50)
        report.sentiment_label = plan_data.get("sentiment", {}).get("label", "NEUTRAL")

        # Cross-asset context
        cross = plan_data.get("cross_asset", [])
        for item in cross:
            if item.get("sym") == "DXY":
                report.dxy_level = item.get("price", 0)
            elif item.get("sym") == "VIX":
                report.vix_level = item.get("price", 0)

        report.economic_events = plan_data.get("events", [])

    if focus_assets:
        for asset in focus_assets:
            report.focus_assets.append(
                {
                    "symbol": asset.get("symbol", ""),
                    "direction": asset.get("bias", "NEUTRAL"),
                    "entry_zone": f"{asset.get('entry_low', 0):.2f} - {asset.get('entry_high', 0):.2f}",
                    "confidence": asset.get("quality_pct", 0),
                    "reason": asset.get("notes", ""),
                }
            )

    return report


def generate_post_session_report(
    trades: list[dict[str, Any]],
    plan_data: dict[str, Any] | None = None,
    daily_pnl: float = 0.0,
    max_drawdown: float = 0.0,
    max_contracts_used: int = 0,
    trader_notes: str = "",
) -> PostSessionReport:
    """Generate a post-session report from the day's trading activity.

    Args:
        trades: List of trade dicts from the journal
        plan_data: The morning plan (to check adherence)
        daily_pnl: Net P&L for the day
        max_drawdown: Maximum intraday drawdown
        max_contracts_used: Peak concurrent contracts
        trader_notes: Trader's end-of-day notes

    Returns:
        PostSessionReport instance
    """
    report = PostSessionReport(
        net_pnl=daily_pnl,
        max_drawdown_intraday=max_drawdown,
        max_contracts_used=max_contracts_used,
        trader_notes=trader_notes,
    )

    report.total_trades = len(trades)

    if trades:
        winners = [t for t in trades if t.get("pnl", 0) > 0]
        losers = [t for t in trades if t.get("pnl", 0) < 0]

        report.winning_trades = len(winners)
        report.losing_trades = len(losers)
        report.win_rate = (len(winners) / len(trades) * 100) if trades else 0

        if winners:
            report.avg_winner = sum(t["pnl"] for t in winners) / len(winners)
            report.largest_winner = max(t["pnl"] for t in winners)

        if losers:
            report.avg_loser = sum(t["pnl"] for t in losers) / len(losers)
            report.largest_loser = min(t["pnl"] for t in losers)

        gross_wins = sum(t["pnl"] for t in winners) if winners else 0
        gross_losses = abs(sum(t["pnl"] for t in losers)) if losers else 0
        report.profit_factor = (
            (gross_wins / gross_losses) if gross_losses > 0 else float("inf") if gross_wins > 0 else 0
        )

        report.gross_pnl = sum(t.get("pnl", 0) for t in trades)

        # Plan adherence
        planned = [t for t in trades if t.get("planned", False)]
        report.planned_trades = len(planned)
        report.unplanned_trades = len(trades) - len(planned)
        report.plan_adherence_pct = (len(planned) / len(trades) * 100) if trades else 0

        report.trades = trades

    # Goals
    report.hit_daily_goal = report.net_pnl >= 500
    report.hit_stretch_goal = report.net_pnl >= 1800
    report.profit_target_progress_pct = (report.net_pnl / 9000 * 100) if report.net_pnl > 0 else 0

    # Risk utilization
    report.risk_budget_used_pct = (
        (abs(min(0, report.max_drawdown_intraday)) / 3300 * 100) if report.max_drawdown_intraday < 0 else 0
    )

    # Session grade (automated)
    report.session_grade = _compute_session_grade(report)

    # Auto-generated insights
    report.what_worked = _analyze_what_worked(trades)
    report.what_didnt_work = _analyze_what_didnt_work(trades)
    report.improvement_notes = _generate_improvement_notes(report)

    return report


def _compute_session_grade(report: PostSessionReport) -> str:
    """Compute an automatic session grade."""
    score = 50  # Start at C

    # P&L contribution
    if report.net_pnl >= 1800:
        score += 25
    elif report.net_pnl >= 500:
        score += 15
    elif report.net_pnl > 0:
        score += 5
    elif report.net_pnl > -500:
        score -= 5
    else:
        score -= 15

    # Plan adherence
    if report.plan_adherence_pct >= 90:
        score += 15
    elif report.plan_adherence_pct >= 70:
        score += 10
    elif report.plan_adherence_pct >= 50:
        score += 0
    else:
        score -= 10

    # Win rate
    if report.win_rate >= 60:
        score += 10
    elif report.win_rate >= 40:
        score += 0
    else:
        score -= 5

    # Risk discipline
    if report.risk_budget_used_pct < 30:
        score += 5
    elif report.risk_budget_used_pct > 70:
        score -= 10

    if score >= 85:
        return "A"
    elif score >= 70:
        return "B"
    elif score >= 55:
        return "C"
    elif score >= 40:
        return "D"
    else:
        return "F"


def _analyze_what_worked(trades: list[dict[str, Any]]) -> list[str]:
    """Auto-analyze what worked in the session."""
    notes = []
    winners = [t for t in trades if t.get("pnl", 0) > 0]

    if not winners:
        return ["No winning trades to analyze"]

    # Check for planned winners
    planned_winners = [t for t in winners if t.get("planned", False)]
    if planned_winners:
        notes.append(f"{len(planned_winners)} planned trades hit targets")

    # Check for good R:R
    for t in winners:
        if t.get("pnl", 0) > 500:
            notes.append(f"{t.get('symbol', '?')} {t.get('direction', '?')}: +${t['pnl']:.0f}")

    return notes or ["Session had winning trades"]


def _analyze_what_didnt_work(trades: list[dict[str, Any]]) -> list[str]:
    """Auto-analyze what didn't work."""
    notes = []
    losers = [t for t in trades if t.get("pnl", 0) < 0]

    if not losers:
        return ["No losing trades — clean session"]

    # Unplanned losers
    unplanned_losers = [t for t in losers if not t.get("planned", False)]
    if unplanned_losers:
        notes.append(f"{len(unplanned_losers)} unplanned trades lost money — stick to the plan")

    for t in losers:
        if t.get("pnl", 0) < -300:
            notes.append(f"{t.get('symbol', '?')}: -${abs(t['pnl']):.0f} — review stop placement")

    return notes or ["Minor losses within tolerance"]


def _generate_improvement_notes(report: PostSessionReport) -> list[str]:
    """Generate system improvement suggestions."""
    notes = []

    if report.plan_adherence_pct < 70:
        notes.append("Low plan adherence — review if plan was clear enough or if discipline slipped")

    if report.win_rate < 40 and report.total_trades > 3:
        notes.append("Low win rate with multiple trades — consider fewer, higher-conviction entries")

    if report.risk_budget_used_pct > 60:
        notes.append("Used >60% of risk budget — consider smaller position sizes")

    if report.max_contracts_used > 10:
        notes.append(f"Peak {report.max_contracts_used} contracts — close to TPT 15-contract limit")

    if report.unplanned_trades > report.planned_trades:
        notes.append("More unplanned than planned trades — the morning plan needs work")

    if not notes:
        notes.append("Solid session — keep executing the process")

    return notes


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def save_pre_session_report(report: PreSessionReport) -> bool:
    """Save a pre-session report to the database."""
    init_report_tables()
    conn = _get_db()
    try:
        conn.execute(
            """INSERT OR REPLACE INTO pre_session_reports
               (report_date, generated_at, data, trader_notes)
               VALUES (?, ?, ?, ?)""",
            (report.report_date, report.generated_at, json.dumps(asdict(report), default=str), report.trader_notes),
        )
        conn.commit()
        logger.info("Pre-session report saved for %s", report.report_date)
        return True
    except Exception as exc:
        logger.error("Failed to save pre-session report: %s", exc)
        return False
    finally:
        conn.close()


def save_post_session_report(report: PostSessionReport) -> bool:
    """Save a post-session report to the database."""
    init_report_tables()
    conn = _get_db()
    try:
        conn.execute(
            """INSERT OR REPLACE INTO post_session_reports
               (report_date, generated_at, data, session_grade, net_pnl, win_rate, plan_adherence, trader_notes)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                report.report_date,
                report.generated_at,
                json.dumps(asdict(report), default=str),
                report.session_grade,
                report.net_pnl,
                report.win_rate,
                report.plan_adherence_pct,
                report.trader_notes,
            ),
        )
        conn.commit()
        logger.info(
            "Post-session report saved for %s (grade=%s, pnl=$%.2f)",
            report.report_date,
            report.session_grade,
            report.net_pnl,
        )
        return True
    except Exception as exc:
        logger.error("Failed to save post-session report: %s", exc)
        return False
    finally:
        conn.close()


def get_report_history(days: int = 30) -> dict[str, Any]:
    """Get report history for the last N days."""
    init_report_tables()
    conn = _get_db()
    try:
        pre_reports = conn.execute(
            "SELECT report_date, data FROM pre_session_reports ORDER BY report_date DESC LIMIT ?",
            (days,),
        ).fetchall()

        post_reports = conn.execute(
            "SELECT report_date, session_grade, net_pnl, win_rate, plan_adherence, data FROM post_session_reports ORDER BY report_date DESC LIMIT ?",
            (days,),
        ).fetchall()

        return {
            "pre_sessions": [{"date": r["report_date"], "data": json.loads(r["data"])} for r in pre_reports],
            "post_sessions": [
                {
                    "date": r["report_date"],
                    "grade": r["session_grade"],
                    "pnl": r["net_pnl"],
                    "win_rate": r["win_rate"],
                    "adherence": r["plan_adherence"],
                    "data": json.loads(r["data"]),
                }
                for r in post_reports
            ],
            "summary": _compute_history_summary(post_reports),
        }
    except Exception as exc:
        logger.error("Failed to load report history: %s", exc)
        return {"pre_sessions": [], "post_sessions": [], "summary": {}}
    finally:
        conn.close()


def _compute_history_summary(post_reports: list) -> dict[str, Any]:
    """Compute aggregate stats from report history."""
    if not post_reports:
        return {}

    pnls = [r["net_pnl"] for r in post_reports if r["net_pnl"] is not None]
    grades = [r["session_grade"] for r in post_reports if r["session_grade"]]

    return {
        "total_days": len(post_reports),
        "total_pnl": sum(pnls),
        "avg_daily_pnl": sum(pnls) / len(pnls) if pnls else 0,
        "best_day": max(pnls) if pnls else 0,
        "worst_day": min(pnls) if pnls else 0,
        "winning_days": sum(1 for p in pnls if p > 0),
        "losing_days": sum(1 for p in pnls if p < 0),
        "avg_grade": max(set(grades), key=grades.count) if grades else "N/A",
        "profit_target_progress": sum(pnls) / 9000 * 100 if pnls else 0,
    }
