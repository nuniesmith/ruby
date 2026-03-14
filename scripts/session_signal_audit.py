#!/usr/bin/env python
"""
Session Signal Quality Audit
==============================
Analyses ORB signal quality for each session by cross-referencing audit-trail
events with paper-trade outcomes stored in the database.

Purpose:
  - Validate overnight session signal quality (Priority 2 todo item):
    "Validate overnight session signals (CME open, Tokyo, Shanghai) against
    paper trades — confirm signal quality before enabling hard CNN gate
    for overnight."
  - Provide per-session win-rate, profit-factor, CNN probability distributions,
    and filter rejection stats so you can make data-driven decisions about
    which sessions to enable the hard CNN gate on.

Output:
  - Console table (always)
  - Optional JSON export  (--export-json PATH)
  - Optional CSV export   (--export-csv PATH)
  - Optional Redis push   (--push-redis)  → engine:audit:session_quality

Usage::

    cd futures

    # Full audit — all sessions, last 30 days
    PYTHONPATH=src .venv/bin/python scripts/session_signal_audit.py

    # Overnight sessions only, 14 days, verbose
    PYTHONPATH=src .venv/bin/python scripts/session_signal_audit.py \\
        --sessions cme sydney tokyo shanghai --days 14 -v

    # Export and push to Redis for Grafana
    PYTHONPATH=src .venv/bin/python scripts/session_signal_audit.py \\
        --days 30 --export-json audit_session_quality.json --push-redis

    # Recommend CNN gate enablement based on win-rate threshold
    PYTHONPATH=src .venv/bin/python scripts/session_signal_audit.py \\
        --days 30 --recommend --min-win-rate 0.58 --min-signals 5

Environment variables:
    DATABASE_URL          Postgres DSN (falls back to SQLite at data/futures.db)
    REDIS_URL             Redis URL (for --push-redis)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("session_signal_audit")

_EST = ZoneInfo("America/New_York")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_ALL_SESSION_KEYS = [
    "cme",
    "sydney",
    "tokyo",
    "shanghai",
    "frankfurt",
    "london",
    "london_ny",
    "us",
    "cme_settle",
]

_OVERNIGHT_SESSIONS = {"cme", "sydney", "tokyo", "shanghai"}

_SESSION_LABELS: dict[str, str] = {
    "cme": "CME Open  18:00 ET",
    "sydney": "Sydney    18:30 ET",
    "tokyo": "Tokyo     19:00 ET",
    "shanghai": "Shanghai  21:00 ET",
    "frankfurt": "Frankfurt 03:00 ET",
    "london": "London    03:00 ET",
    "london_ny": "London-NY 08:00 ET",
    "us": "US Equity 09:30 ET",
    "cme_settle": "CME Settl 14:00 ET",
}

# CNN gate thresholds (must match breakout_cnn.py SESSION_THRESHOLDS)
_SESSION_THRESHOLDS: dict[str, float] = {
    "cme": 0.75,
    "sydney": 0.72,
    "tokyo": 0.74,
    "shanghai": 0.74,
    "frankfurt": 0.80,
    "london": 0.82,
    "london_ny": 0.82,
    "us": 0.82,
    "cme_settle": 0.78,
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class SignalRecord:
    """A single ORB signal event from the audit trail."""

    row_id: int
    symbol: str
    session: str
    direction: str
    trigger_price: float
    or_high: float
    or_low: float
    atr_value: float
    breakout_detected: bool
    filter_passed: bool | None
    cnn_prob: float | None
    cnn_signal: bool | None
    cnn_gated: bool | None
    published: bool | None
    created_at: datetime | None
    # trade outcome (joined from trades table, may be None)
    trade_pnl: float | None = None
    trade_outcome: str | None = None  # "win" | "loss" | "scratch" | None
    trade_exit_price: float | None = None


@dataclass
class SessionStats:
    """Aggregated statistics for one ORB session."""

    session_key: str
    label: str
    is_overnight: bool
    days_analysed: int

    # Signal counts
    total_orb_detections: int = 0
    filter_passed_count: int = 0
    filter_rejected_count: int = 0
    cnn_above_threshold_count: int = 0
    cnn_below_threshold_count: int = 0
    cnn_gated_count: int = 0
    published_count: int = 0

    # CNN probability distribution (over published signals)
    cnn_probs: list[float] = field(default_factory=list)
    cnn_threshold: float = 0.82

    # Trade outcomes (where paper-trade records exist)
    trade_count: int = 0
    win_count: int = 0
    loss_count: int = 0
    scratch_count: int = 0
    total_pnl: float = 0.0
    avg_win_pnl: float = 0.0
    avg_loss_pnl: float = 0.0

    # Computed fields (populated by finalise())
    filter_pass_rate: float = 0.0
    cnn_pass_rate: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_cnn_prob: float = 0.0
    cnn_prob_p25: float = 0.0
    cnn_prob_p75: float = 0.0
    gate_recommendation: str = "unknown"  # "enable" | "hold" | "insufficient_data"

    def finalise(self, min_signals: int = 5, min_win_rate: float = 0.58) -> None:
        """Compute derived metrics after all records have been accumulated."""
        # Filter pass rate
        total_filtered = self.filter_passed_count + self.filter_rejected_count
        self.filter_pass_rate = self.filter_passed_count / total_filtered if total_filtered > 0 else 0.0

        # CNN pass rate (of signals that reached CNN)
        total_cnn = self.cnn_above_threshold_count + self.cnn_below_threshold_count
        self.cnn_pass_rate = self.cnn_above_threshold_count / total_cnn if total_cnn > 0 else 0.0

        # Win rate
        self.win_rate = self.win_count / self.trade_count if self.trade_count > 0 else 0.0

        # Profit factor
        gross_profit = sum(p for p in [self.avg_win_pnl * self.win_count] if p > 0)
        gross_loss = abs(self.avg_loss_pnl * self.loss_count)
        self.profit_factor = (
            gross_profit / gross_loss if gross_loss > 0 else (float("inf") if gross_profit > 0 else 0.0)
        )

        # CNN probability stats
        if self.cnn_probs:
            import statistics

            self.avg_cnn_prob = statistics.mean(self.cnn_probs)
            sorted_p = sorted(self.cnn_probs)
            n = len(sorted_p)
            self.cnn_prob_p25 = sorted_p[max(0, int(n * 0.25))]
            self.cnn_prob_p75 = sorted_p[min(n - 1, int(n * 0.75))]

        # Gate recommendation
        if self.trade_count < min_signals:
            self.gate_recommendation = "insufficient_data"
        elif self.win_rate >= min_win_rate and self.profit_factor >= 1.2:
            self.gate_recommendation = "enable"
        else:
            self.gate_recommendation = "hold"

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("cnn_probs", None)  # exclude raw list from JSON
        return d


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------


def _get_conn():
    """Return a database connection (Postgres preferred, SQLite fallback)."""
    db_url = os.environ.get("DATABASE_URL", "")
    if db_url and db_url.startswith("postgresql"):
        try:
            import psycopg2  # type: ignore[import-unresolved]

            conn = psycopg2.connect(db_url)
            conn.autocommit = False
            return conn, True  # (conn, is_postgres)
        except ImportError:
            logger.warning("psycopg2 not installed — falling back to SQLite")
        except Exception as exc:
            logger.warning("Postgres connect failed (%s) — falling back to SQLite", exc)

    # SQLite fallback
    import sqlite3

    db_path = PROJECT_ROOT / "data" / "futures.db"
    if not db_path.exists():
        # Try common alternate locations
        for alt in [PROJECT_ROOT / "futures.db", Path("/app/data/futures.db")]:
            if alt.exists():
                db_path = alt
                break

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn, False  # (conn, is_postgres)


def _fetch_orb_events(
    conn,
    is_postgres: bool,
    days_back: int,
    sessions: list[str],
) -> list[dict[str, Any]]:
    """Fetch ORB audit events from the database."""
    ph = "%s" if is_postgres else "?"
    since = datetime.now(tz=_EST) - timedelta(days=days_back)
    since_str = since.isoformat()

    # Build session filter
    sess_placeholders = ", ".join([ph] * len(sessions))
    params: list[Any] = [since_str] + sessions

    # The orb_events table schema (from models.py / record_orb_event):
    #   id, symbol, or_high, or_low, or_range, atr_value,
    #   breakout_detected, direction, trigger_price, long_trigger, short_trigger,
    #   bar_count, session, metadata_json, created_at
    query = f"""
        SELECT
            id,
            symbol,
            or_high,
            or_low,
            atr_value,
            breakout_detected,
            direction,
            trigger_price,
            session,
            metadata_json,
            created_at
        FROM orb_events
        WHERE created_at >= {ph}
          AND (
              -- session stored directly in the session column
              session IN ({sess_placeholders})
              -- OR session stored in metadata_json (older rows)
              OR metadata_json LIKE '%"orb_session"%'
          )
        ORDER BY created_at DESC
    """
    try:
        cur = conn.cursor()
        cur.execute(query, params)
        rows = cur.fetchall()
        return [dict(r) for r in rows]
    except Exception as exc:
        logger.warning("orb_events query failed: %s", exc)
        return []


def _fetch_trades(
    conn,
    is_postgres: bool,
    days_back: int,
) -> list[dict[str, Any]]:
    """Fetch paper trade records to join with ORB signals."""
    ph = "%s" if is_postgres else "?"
    since = datetime.now(tz=_EST) - timedelta(days=days_back)
    since_str = since.isoformat()

    # Common trade table schemas vary — try a few column name patterns
    queries = [
        # Trade log / risk manager style
        f"""
        SELECT symbol, direction, entry_price, exit_price, pnl, created_at,
               metadata_json
        FROM trades
        WHERE created_at >= {ph}
        ORDER BY created_at DESC
        """,
        # Alternate: sim_trades table
        f"""
        SELECT symbol, direction, entry_price, exit_price, pnl, created_at
        FROM sim_trades
        WHERE created_at >= {ph}
        ORDER BY created_at DESC
        """,
    ]

    for query in queries:
        try:
            cur = conn.cursor()
            cur.execute(query, [since_str])
            rows = cur.fetchall()
            return [dict(r) for r in rows]
        except Exception:
            continue

    logger.debug("No trades table found — trade outcome analysis will be skipped")
    return []


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def _parse_metadata(raw: str | bytes | None) -> dict[str, Any]:
    """Parse metadata_json field, returning empty dict on failure."""
    if not raw:
        return {}
    try:
        s = raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)
        return json.loads(s)
    except Exception:
        return {}


def _parse_signal_records(
    rows: list[dict[str, Any]],
    sessions: list[str],
) -> list[SignalRecord]:
    """Convert raw DB rows into SignalRecord objects, filtering by session."""
    records: list[SignalRecord] = []
    for row in rows:
        meta = _parse_metadata(row.get("metadata_json"))

        # Resolve session: prefer metadata orb_session, fall back to session column
        session = meta.get("orb_session") or row.get("session") or "unknown"
        session = session.lower().strip()

        if session not in sessions:
            continue

        # Parse created_at
        created_at: datetime | None = None
        raw_ts = row.get("created_at")
        if raw_ts:
            try:
                if isinstance(raw_ts, str):
                    created_at = datetime.fromisoformat(raw_ts.replace("Z", "+00:00"))
                elif isinstance(raw_ts, datetime):
                    created_at = raw_ts
            except Exception:
                pass

        records.append(
            SignalRecord(
                row_id=row.get("id", 0),
                symbol=row.get("symbol", ""),
                session=session,
                direction=row.get("direction", ""),
                trigger_price=float(row.get("trigger_price") or 0),
                or_high=float(row.get("or_high") or 0),
                or_low=float(row.get("or_low") or 0),
                atr_value=float(row.get("atr_value") or 0),
                breakout_detected=bool(row.get("breakout_detected", False)),
                filter_passed=meta.get("filter_passed"),
                cnn_prob=meta.get("cnn_prob"),
                cnn_signal=meta.get("cnn_signal"),
                cnn_gated=meta.get("cnn_gated"),
                published=meta.get("published"),
                created_at=created_at,
            )
        )
    return records


def _match_trades(
    signals: list[SignalRecord],
    trades: list[dict[str, Any]],
    price_tolerance: float = 0.005,
    time_window_minutes: int = 120,
) -> None:
    """Mutate signals in-place to attach trade outcome data.

    Matching logic:
      - Same symbol (case-insensitive; strips =F suffix)
      - Same direction (LONG / SHORT matches BUY / SELL)
      - Entry price within ±0.5% of signal trigger price
      - Trade created within *time_window_minutes* after the signal
    """
    if not trades:
        return

    def _norm_symbol(s: str) -> str:
        return s.upper().replace("=F", "").strip()

    def _norm_direction(d: str) -> str:
        d = d.upper()
        if d in ("LONG", "BUY"):
            return "LONG"
        if d in ("SHORT", "SELL"):
            return "SHORT"
        return d

    for sig in signals:
        if not sig.published:
            continue
        if sig.created_at is None:
            continue

        sig_sym = _norm_symbol(sig.symbol)
        sig_dir = _norm_direction(sig.direction)
        sig_price = sig.trigger_price
        sig_ts = sig.created_at

        for trade in trades:
            t_sym = _norm_symbol(str(trade.get("symbol", "")))
            t_dir = _norm_direction(str(trade.get("direction", "")))
            t_price = float(trade.get("entry_price") or 0)
            t_pnl = float(trade.get("pnl") or 0)

            # Parse trade timestamp
            raw_ts = trade.get("created_at")
            if not raw_ts:
                continue
            try:
                if isinstance(raw_ts, str):
                    t_ts = datetime.fromisoformat(raw_ts.replace("Z", "+00:00"))
                elif isinstance(raw_ts, datetime):
                    t_ts = raw_ts
                else:
                    continue
            except Exception:
                continue

            # Check criteria
            if t_sym != sig_sym:
                continue
            if t_dir != sig_dir:
                continue
            if sig_price > 0:
                price_diff = abs(t_price - sig_price) / sig_price
                if price_diff > price_tolerance:
                    continue
            delta = (t_ts - sig_ts).total_seconds()
            if delta < -60 or delta > time_window_minutes * 60:
                continue

            # Match found
            sig.trade_pnl = t_pnl
            sig.trade_exit_price = float(trade.get("exit_price") or 0)

            SCRATCH_THRESHOLD = 0.50  # $0.50 abs PnL → scratch
            if abs(t_pnl) <= SCRATCH_THRESHOLD:
                sig.trade_outcome = "scratch"
            elif t_pnl > 0:
                sig.trade_outcome = "win"
            else:
                sig.trade_outcome = "loss"
            break  # stop on first match


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def _aggregate_stats(
    signals: list[SignalRecord],
    days_back: int,
    min_signals: int,
    min_win_rate: float,
) -> list[SessionStats]:
    """Group signals by session and compute stats."""
    from collections import defaultdict

    buckets: dict[str, list[SignalRecord]] = defaultdict(list)
    for sig in signals:
        buckets[sig.session].append(sig)

    stats_list: list[SessionStats] = []
    for sk in _ALL_SESSION_KEYS:
        recs = buckets.get(sk, [])
        threshold = _SESSION_THRESHOLDS.get(sk, 0.82)

        st = SessionStats(
            session_key=sk,
            label=_SESSION_LABELS.get(sk, sk),
            is_overnight=sk in _OVERNIGHT_SESSIONS,
            days_analysed=days_back,
            cnn_threshold=threshold,
        )

        for sig in recs:
            if not sig.breakout_detected:
                continue

            st.total_orb_detections += 1

            fp = sig.filter_passed
            if fp is True:
                st.filter_passed_count += 1
            elif fp is False:
                st.filter_rejected_count += 1

            prob = sig.cnn_prob
            if prob is not None:
                if prob >= threshold:
                    st.cnn_above_threshold_count += 1
                else:
                    st.cnn_below_threshold_count += 1

            if sig.cnn_gated:
                st.cnn_gated_count += 1

            if sig.published:
                st.published_count += 1
                if prob is not None:
                    st.cnn_probs.append(prob)

            # Trade outcomes
            if sig.trade_outcome is not None:
                st.trade_count += 1
                pnl = sig.trade_pnl or 0.0
                if sig.trade_outcome == "win":
                    st.win_count += 1
                    st.avg_win_pnl = (st.avg_win_pnl * (st.win_count - 1) + pnl) / st.win_count
                elif sig.trade_outcome == "loss":
                    st.loss_count += 1
                    st.avg_loss_pnl = (st.avg_loss_pnl * (st.loss_count - 1) + pnl) / st.loss_count
                else:
                    st.scratch_count += 1
                st.total_pnl += pnl

        st.finalise(min_signals=min_signals, min_win_rate=min_win_rate)
        stats_list.append(st)

    return stats_list


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------

_COL_WIDTHS = {
    "session": 22,
    "detections": 6,
    "published": 6,
    "filt%": 6,
    "cnn%": 6,
    "avg_p": 6,
    "trades": 6,
    "wins": 5,
    "win%": 6,
    "pf": 6,
    "pnl": 8,
    "gate_rec": 14,
}

_ANSI = {
    "reset": "\033[0m",
    "bold": "\033[1m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "red": "\033[31m",
    "cyan": "\033[36m",
    "dim": "\033[2m",
    "blue": "\033[34m",
}


def _c(text: str, color: str, use_color: bool = True) -> str:
    if not use_color:
        return text
    return f"{_ANSI.get(color, '')}{text}{_ANSI['reset']}"


def _fmt_pct(val: float, use_color: bool = True) -> str:
    s = f"{val * 100:.0f}%"
    if val >= 0.60:
        return _c(s, "green", use_color)
    if val >= 0.45:
        return _c(s, "yellow", use_color)
    return _c(s, "red", use_color)


def _fmt_pf(val: float, use_color: bool = True) -> str:
    if val == float("inf"):
        return _c("∞", "green", use_color)
    s = f"{val:.2f}"
    if val >= 1.5:
        return _c(s, "green", use_color)
    if val >= 1.0:
        return _c(s, "yellow", use_color)
    return _c(s, "red", use_color)


def _fmt_gate(rec: str, use_color: bool = True) -> str:
    if rec == "enable":
        return _c("✅ ENABLE", "green", use_color)
    if rec == "hold":
        return _c("⏸  HOLD", "yellow", use_color)
    return _c("❓ n/a", "dim", use_color)


def print_table(stats: list[SessionStats], use_color: bool = True, verbose: bool = False) -> None:
    """Print a formatted per-session summary table to stdout."""
    W = _COL_WIDTHS
    sep = "─"

    # Header
    hdr = (
        f"{'Session':<{W['session']}} "
        f"{'Det':>{W['detections']}} "
        f"{'Pub':>{W['published']}} "
        f"{'Filt%':>{W['filt%']}} "
        f"{'CNN%':>{W['cnn%']}} "
        f"{'AvgP':>{W['avg_p']}} "
        f"{'Trd':>{W['trades']}} "
        f"{'Win':>{W['wins']}} "
        f"{'WR%':>{W['win%']}} "
        f"{'PF':>{W['pf']}} "
        f"{'PnL$':>{W['pnl']}} "
        f"{'Gate Rec':<{W['gate_rec']}}"
    )
    total_w = len(hdr)
    print()
    print(_c("  ORB Session Signal Quality Audit", "bold", use_color))
    print(_c("  " + sep * total_w, "dim", use_color))
    print("  " + _c(hdr, "dim", use_color))
    print("  " + _c(sep * total_w, "dim", use_color))

    overnight_printed = False
    daytime_printed = False

    for st in stats:
        # Section dividers
        if st.is_overnight and not overnight_printed:
            print("  " + _c("  🌙 OVERNIGHT SESSIONS", "cyan", use_color))
            overnight_printed = True
        if not st.is_overnight and not daytime_printed:
            print("  " + _c("  ☀  DAYTIME SESSIONS", "cyan", use_color))
            daytime_printed = True

        det_s = str(st.total_orb_detections) if st.total_orb_detections else _c("0", "dim", use_color)
        pub_s = str(st.published_count) if st.published_count else _c("0", "dim", use_color)
        filt_s = (
            _fmt_pct(st.filter_pass_rate, use_color)
            if st.filter_passed_count + st.filter_rejected_count > 0
            else _c("n/a", "dim", use_color)
        )
        cnn_s = (
            _fmt_pct(st.cnn_pass_rate, use_color)
            if (st.cnn_above_threshold_count + st.cnn_below_threshold_count) > 0
            else _c("n/a", "dim", use_color)
        )
        avgp_s = f"{st.avg_cnn_prob:.2f}" if st.avg_cnn_prob > 0 else _c("n/a", "dim", use_color)
        trd_s = str(st.trade_count) if st.trade_count else _c("0", "dim", use_color)
        win_s = str(st.win_count) if st.win_count else _c("0", "dim", use_color)
        wr_s = _fmt_pct(st.win_rate, use_color) if st.trade_count > 0 else _c("n/a", "dim", use_color)
        pf_s = _fmt_pf(st.profit_factor, use_color) if st.trade_count > 0 else _c("n/a", "dim", use_color)
        pnl_s = f"${st.total_pnl:+.0f}" if st.trade_count > 0 else _c("n/a", "dim", use_color)
        gate_s = _fmt_gate(st.gate_recommendation, use_color)

        row = (
            f"{st.label:<{W['session']}} "
            f"{det_s:>{W['detections']}} "
            f"{pub_s:>{W['published']}} "
            f"{filt_s:>{W['filt%']}} "
            f"{cnn_s:>{W['cnn%']}} "
            f"{avgp_s:>{W['avg_p']}} "
            f"{trd_s:>{W['trades']}} "
            f"{win_s:>{W['wins']}} "
            f"{wr_s:>{W['win%']}} "
            f"{pf_s:>{W['pf']}} "
            f"{pnl_s:>{W['pnl']}} "
            f"{gate_s}"
        )
        print("  " + row)

        if verbose and st.total_orb_detections > 0:
            # Verbose: show extra distribution info
            p_line = ""
            if st.cnn_probs:
                p_line = (
                    f"    CNN P distribution — "
                    f"p25={st.cnn_prob_p25:.2f}  "
                    f"avg={st.avg_cnn_prob:.2f}  "
                    f"p75={st.cnn_prob_p75:.2f}  "
                    f"threshold={st.cnn_threshold:.2f}  "
                    f"n={len(st.cnn_probs)}"
                )
            filter_line = (
                f"    Filters — "
                f"passed={st.filter_passed_count}  "
                f"rejected={st.filter_rejected_count}  "
                f"cnn_gated={st.cnn_gated_count}"
            )
            if p_line:
                print(_c(p_line, "dim", use_color))
            print(_c(filter_line, "dim", use_color))

    print("  " + _c(sep * total_w, "dim", use_color))
    print()


def print_recommendations(
    stats: list[SessionStats],
    use_color: bool = True,
    min_signals: int = 5,
) -> None:
    """Print a CNN gate recommendation summary."""
    enable = [s for s in stats if s.gate_recommendation == "enable"]
    hold = [s for s in stats if s.gate_recommendation == "hold"]
    no_data = [s for s in stats if s.gate_recommendation == "insufficient_data"]

    print(_c("  ── CNN Gate Recommendations ──────────────────────────────────", "bold", use_color))
    print()

    if enable:
        print(_c(f"  ✅ Ready to enable CNN gate ({len(enable)} session(s)):", "green", use_color))
        for s in enable:
            cmd = f"    from lib.core.redis_helpers import set_cnn_gate; set_cnn_gate('{s.session_key}', True)"
            print(f"     • {s.label}  (WR={s.win_rate:.0%}, PF={s.profit_factor:.2f}, n={s.trade_count})")
            print(_c(f"       {cmd}", "dim", use_color))
        print()
    else:
        print(_c("  ⏸  No sessions ready to enable CNN gate yet.", "yellow", use_color))
        print()

    if hold:
        print(
            _c(f"  ⏸  Hold — more data needed or quality below target ({len(hold)} session(s)):", "yellow", use_color)
        )
        for s in hold:
            if s.trade_count > 0:
                print(f"     • {s.label}  (WR={s.win_rate:.0%}, PF={s.profit_factor:.2f}, n={s.trade_count})")
            else:
                print(f"     • {s.label}  (published={s.published_count}, no matched trades)")
        print()

    if no_data:
        print(
            _c(
                f"  ❓ Insufficient data (< {min_signals} matched trades) ({len(no_data)} session(s)):",
                "dim",
                use_color,
            )
        )
        for s in no_data:
            print(f"     • {s.label}  (published={s.published_count}, matched trades={s.trade_count})")
        print()

    print(_c("  To enable overnight CNN gate via API:", "dim", use_color))
    print(_c("    curl -X PUT 'http://localhost:8000/cnn/gate/cme?enabled=true'", "dim", use_color))
    print(_c("    curl -X PUT 'http://localhost:8000/cnn/gate/sydney?enabled=true'", "dim", use_color))
    print()


# ---------------------------------------------------------------------------
# Export helpers
# ---------------------------------------------------------------------------


def export_json(stats: list[SessionStats], path: str) -> None:
    data = {
        "generated_at": datetime.now(tz=_EST).isoformat(),
        "sessions": [s.to_dict() for s in stats],
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    logger.info("JSON exported → %s", path)


def export_csv(stats: list[SessionStats], path: str) -> None:
    import csv

    if not stats:
        return

    rows = [s.to_dict() for s in stats]
    fieldnames = list(rows[0].keys())

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    logger.info("CSV exported → %s", path)


def push_to_redis(stats: list[SessionStats]) -> bool:
    """Push session quality data to Redis for Grafana / dashboard consumption."""
    try:
        from lib.core.redis_helpers import TTL_DAY, cache_set_json

        data = {
            "generated_at": datetime.now(tz=_EST).isoformat(),
            "sessions": {s.session_key: s.to_dict() for s in stats},
        }
        ok = cache_set_json("engine:audit:session_quality", data, ttl=TTL_DAY)
        if ok:
            logger.info("Session quality data pushed to Redis (engine:audit:session_quality)")
        else:
            logger.warning("Redis push failed — Redis may be unavailable")
        return ok
    except Exception as exc:
        logger.warning("Redis push error: %s", exc)
        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run_audit(
    sessions: list[str],
    days_back: int,
    min_signals: int,
    min_win_rate: float,
    verbose: bool,
    export_json_path: str | None,
    export_csv_path: str | None,
    push_redis: bool,
    recommend: bool,
    use_color: bool,
) -> list[SessionStats]:
    logger.info(
        "Starting session signal audit: sessions=%s days=%d",
        ", ".join(sessions),
        days_back,
    )

    # Connect to database
    try:
        conn, is_postgres = _get_conn()
        db_type = "Postgres" if is_postgres else "SQLite"
        logger.info("Database: %s", db_type)
    except Exception as exc:
        logger.error("Cannot connect to database: %s", exc)
        print(f"\n  ❌ Database unavailable: {exc}")
        print("  Set DATABASE_URL env var or ensure data/futures.db exists.\n")
        return []

    # Fetch raw events
    rows = _fetch_orb_events(conn, is_postgres, days_back, sessions)
    logger.info("Fetched %d ORB event row(s) from %s", len(rows), db_type)

    # Fetch trades for outcome matching
    trades = _fetch_trades(conn, is_postgres, days_back)
    logger.info("Fetched %d trade record(s)", len(trades))

    conn.close()

    # Parse into signal records
    signals = _parse_signal_records(rows, sessions)
    logger.info("Parsed %d breakout signal record(s)", len(signals))

    # Match trades to signals
    _match_trades(signals, trades)
    matched = sum(1 for s in signals if s.trade_outcome is not None)
    logger.info("Matched %d signal(s) to paper trades", matched)

    # Aggregate per-session stats
    stats = _aggregate_stats(signals, days_back, min_signals, min_win_rate)

    # Print table
    print_table(stats, use_color=use_color, verbose=verbose)

    if recommend:
        print_recommendations(stats, use_color=use_color, min_signals=min_signals)

    # Exports
    if export_json_path:
        export_json(stats, export_json_path)

    if export_csv_path:
        export_csv(stats, export_csv_path)

    if push_redis:
        push_to_redis(stats)

    return stats


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ORB Session Signal Quality Audit",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full audit, last 30 days:
  PYTHONPATH=src .venv/bin/python scripts/session_signal_audit.py

  # Overnight sessions only, 14 days:
  PYTHONPATH=src .venv/bin/python scripts/session_signal_audit.py \\
      --sessions cme sydney tokyo shanghai --days 14 -v

  # Export + push to Redis:
  PYTHONPATH=src .venv/bin/python scripts/session_signal_audit.py \\
      --days 30 --export-json audit.json --push-redis

  # Print gate recommendations:
  PYTHONPATH=src .venv/bin/python scripts/session_signal_audit.py \\
      --days 30 --recommend --min-win-rate 0.58 --min-signals 5
        """,
    )

    parser.add_argument(
        "--sessions",
        nargs="+",
        default=_ALL_SESSION_KEYS,
        metavar="SESSION",
        choices=_ALL_SESSION_KEYS,
        help=("Sessions to audit (default: all). Choices: " + ", ".join(_ALL_SESSION_KEYS)),
    )
    parser.add_argument(
        "--days",
        type=int,
        default=30,
        metavar="N",
        help="Number of days of history to analyse (default: 30)",
    )
    parser.add_argument(
        "--min-signals",
        type=int,
        default=5,
        metavar="N",
        help="Minimum matched trades before making a gate recommendation (default: 5)",
    )
    parser.add_argument(
        "--min-win-rate",
        type=float,
        default=0.58,
        metavar="RATE",
        help="Minimum win rate (0–1) for 'enable' recommendation (default: 0.58)",
    )
    parser.add_argument(
        "--recommend",
        action="store_true",
        help="Print CNN gate enable/hold recommendations after the table",
    )
    parser.add_argument(
        "--export-json",
        metavar="PATH",
        default=None,
        help="Export results to a JSON file",
    )
    parser.add_argument(
        "--export-csv",
        metavar="PATH",
        default=None,
        help="Export results to a CSV file",
    )
    parser.add_argument(
        "--push-redis",
        action="store_true",
        help="Push results to Redis at engine:audit:session_quality",
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable ANSI colour output",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Show per-session CNN probability distribution and filter breakdown",
    )

    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    use_color = not args.no_color and sys.stdout.isatty()

    run_audit(
        sessions=args.sessions,
        days_back=args.days,
        min_signals=args.min_signals,
        min_win_rate=args.min_win_rate,
        verbose=args.verbose,
        export_json_path=args.export_json,
        export_csv_path=args.export_csv,
        push_redis=args.push_redis,
        recommend=args.recommend,
        use_color=use_color,
    )


if __name__ == "__main__":
    main()
