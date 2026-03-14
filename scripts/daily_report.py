#!/usr/bin/env python3
"""
Daily ORB Report — structured end-of-day trading summary
==========================================================

Queries the ``orb_events`` and ``risk_events`` tables from Postgres (or
SQLite fallback) and prints a concise, structured summary of the day's
ORB activity.

What it shows:
  - Total ORB evaluations vs breakouts detected
  - Per-symbol breakdown: direction, trigger price, OR range, ATR, CNN prob
  - Filter pass/fail counts (from metadata_json)
  - Risk events that occurred during the session
  - CNN probability distribution summary (if available)
  - Overnight retrain status (if model files exist)

Usage:
    # Today's report (default)
    python scripts/daily_report.py

    # Specific date
    python scripts/daily_report.py --date 2026-06-15

    # Last N days summary
    python scripts/daily_report.py --days 5

    # JSON output (for piping to other tools)
    python scripts/daily_report.py --json

    # From inside Docker
    docker compose exec engine python scripts/daily_report.py

Environment:
    DATABASE_URL   — Postgres connection string (preferred)
    DB_PATH        — SQLite fallback path
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

# ---------------------------------------------------------------------------
# Ensure the project src/ is on sys.path so `lib.*` imports work whether
# this is run from the repo root or from inside the Docker container.
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent
_SRC = _PROJECT_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# Also check Docker layout (/app/src)
_DOCKER_SRC = Path("/app/src")
if _DOCKER_SRC.is_dir() and str(_DOCKER_SRC) not in sys.path:
    sys.path.insert(0, str(_DOCKER_SRC))

_EST = ZoneInfo("America/New_York")

# ANSI colour helpers (disabled when piping to file)
_USE_COLOR = sys.stdout.isatty()


def _c(code: str, text: str) -> str:
    if not _USE_COLOR:
        return text
    return f"\033[{code}m{text}\033[0m"


def _bold(t: str) -> str:
    return _c("1", t)


def _green(t: str) -> str:
    return _c("32", t)


def _red(t: str) -> str:
    return _c("31", t)


def _yellow(t: str) -> str:
    return _c("33", t)


def _cyan(t: str) -> str:
    return _c("36", t)


def _dim(t: str) -> str:
    return _c("2", t)


# ---------------------------------------------------------------------------
# Database helpers — reuse the project's own models layer when possible,
# fall back to direct connections if imports fail.
# ---------------------------------------------------------------------------


def _get_orb_events(since: str, until: str, limit: int = 500) -> list[dict]:
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
        print(f"  {_red('ERROR')} querying ORB events: {exc}", file=sys.stderr)
        return []


def _get_risk_events(since: str, until: str, limit: int = 500) -> list[dict]:
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
        print(f"  {_red('ERROR')} querying risk events: {exc}", file=sys.stderr)
        return []


# ---------------------------------------------------------------------------
# Report building
# ---------------------------------------------------------------------------


def _parse_metadata(ev: dict) -> dict:
    """Safely parse metadata_json from an event row."""
    raw = ev.get("metadata_json", "{}")
    if isinstance(raw, dict):
        return raw
    try:
        return json.loads(raw) if raw else {}
    except (json.JSONDecodeError, TypeError):
        return {}


def _format_price(val: Any) -> str:
    """Format a price value for display."""
    try:
        f = float(val)
        if f == 0.0:
            return "—"
        # Use appropriate decimal places based on magnitude
        if f > 1000:
            return f"{f:,.2f}"
        elif f > 10:
            return f"{f:,.4f}"
        else:
            return f"{f:,.6f}"
    except (TypeError, ValueError):
        return "—"


def _format_time(ts: str) -> str:
    """Extract HH:MM ET from an ISO timestamp."""
    try:
        dt = datetime.fromisoformat(ts)
        dt = dt.replace(tzinfo=_EST) if dt.tzinfo is None else dt.astimezone(_EST)
        return dt.strftime("%H:%M ET")
    except Exception:
        return ts[:16] if len(ts) >= 16 else ts


def build_daily_report(target_date: date) -> dict:
    """Build a structured report dict for a single trading day.

    Returns a dict with all the data needed for both terminal output
    and JSON serialisation.
    """
    # Build time window: target_date 00:00 ET → target_date+1 00:00 ET
    day_start = datetime(target_date.year, target_date.month, target_date.day, 0, 0, 0, tzinfo=_EST)
    day_end = day_start + timedelta(days=1)
    since = day_start.isoformat()
    until = day_end.isoformat()

    orb_events = _get_orb_events(since, until)
    risk_events = _get_risk_events(since, until)

    # Separate breakout vs non-breakout evaluations
    breakouts = [e for e in orb_events if e.get("breakout_detected", 0) == 1]
    non_breakouts = [e for e in orb_events if e.get("breakout_detected", 0) != 1]

    # Per-symbol breakdown
    symbols_seen: dict[str, list[dict]] = {}
    for ev in orb_events:
        sym = ev.get("symbol", "UNKNOWN")
        symbols_seen.setdefault(sym, []).append(ev)

    # Extract CNN probabilities from metadata
    cnn_probs: list[float] = []
    filter_passed_count = 0
    filter_failed_count = 0
    cnn_gated_count = 0
    published_count = 0

    for ev in breakouts:
        meta = _parse_metadata(ev)
        # CNN probability
        prob = meta.get("cnn_prob")
        if prob is not None:
            with contextlib.suppress(TypeError, ValueError):
                cnn_probs.append(float(prob))
        # Filter outcome
        fp = meta.get("filter_passed")
        if fp is True:
            filter_passed_count += 1
        elif fp is False:
            filter_failed_count += 1
        # CNN gate
        if meta.get("cnn_gated"):
            cnn_gated_count += 1
        # Published
        if meta.get("published"):
            published_count += 1

    # If metadata doesn't have filter info, treat all breakouts as potentially published
    if filter_passed_count == 0 and filter_failed_count == 0:
        # No filter metadata — estimate from breakout count
        published_count = len(breakouts)

    # Risk events breakdown
    risk_blocks = [r for r in risk_events if r.get("event_type") == "block"]
    risk_warnings = [r for r in risk_events if r.get("event_type") == "warning"]

    # CNN stats
    cnn_stats: dict[str, Any] = {}
    if cnn_probs:
        cnn_stats = {
            "count": len(cnn_probs),
            "min": min(cnn_probs),
            "max": max(cnn_probs),
            "mean": sum(cnn_probs) / len(cnn_probs),
            "above_50": sum(1 for p in cnn_probs if p >= 0.5),
            "above_70": sum(1 for p in cnn_probs if p >= 0.7),
        }

    # Model file info
    model_info = _get_model_info()

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
                    "time": _format_time(r.get("timestamp", "")),
                    "type": r.get("event_type", ""),
                    "symbol": r.get("symbol", ""),
                    "reason": r.get("reason", ""),
                }
                for r in risk_events[:20]  # cap display
            ],
        },
        "model": model_info,
        "raw_breakouts": [],
    }

    # Per-symbol data
    for sym, events in sorted(symbols_seen.items()):
        sym_breakouts = [e for e in events if e.get("breakout_detected", 0) == 1]
        sym_evals = len(events)
        report["orb"]["symbols"][sym] = {
            "evaluations": sym_evals,
            "breakouts": len(sym_breakouts),
            "details": [],
        }
        for b in sym_breakouts:
            meta = _parse_metadata(b)
            detail = {
                "time": _format_time(b.get("timestamp", "")),
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
            report["raw_breakouts"].append({"symbol": sym, **detail})

    return report


def _get_model_info() -> dict:
    """Check the state of the CNN model files."""
    info: dict[str, Any] = {"available": False}

    # Check multiple possible locations
    model_dirs = [
        _PROJECT_ROOT / "models",
        Path("/app/models"),
    ]

    for mdir in model_dirs:
        best = mdir / "breakout_cnn_best.pt"
        if best.is_file():
            stat = best.stat()
            info["available"] = True
            info["path"] = str(best)
            info["size_mb"] = round(stat.st_size / (1024 * 1024), 1)
            info["last_modified"] = datetime.fromtimestamp(stat.st_mtime, tz=_EST).isoformat()

            # Check for training history
            history = mdir / "training_history.csv"
            if history.is_file():
                info["training_history"] = str(history)
                # Read last line for most recent metrics
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
                                if k in ("epoch", "val_acc", "val_loss", "val_precision", "val_recall")
                            }
                except Exception:
                    pass

            # Count other checkpoint files
            try:
                checkpoints = list(mdir.glob("breakout_cnn_*.pt"))
                info["checkpoint_count"] = len(checkpoints)
            except Exception:
                pass

            break

    # Check dataset stats
    for ddir in [_PROJECT_ROOT / "dataset", Path("/app/dataset")]:
        stats_file = ddir / "dataset_stats.json"
        if stats_file.is_file():
            with contextlib.suppress(Exception):
                info["dataset_stats"] = json.loads(stats_file.read_text())
            break
        labels_file = ddir / "labels.csv"
        if labels_file.is_file():
            try:
                line_count = sum(1 for _ in labels_file.open()) - 1  # minus header
                info["dataset_size"] = line_count
            except Exception:
                pass
            break

    return info


# ---------------------------------------------------------------------------
# Terminal output formatting
# ---------------------------------------------------------------------------


def print_report(report: dict) -> None:
    """Pretty-print a daily report to the terminal."""
    orb = report["orb"]
    cnn = report.get("cnn", {})
    risk = report["risk"]
    model = report.get("model", {})

    # Header
    print()
    print(_bold("=" * 72))
    print(_bold(f"  📊  DAILY ORB REPORT — {report['date']} ({report['day_of_week']})"))
    print(_bold("=" * 72))
    print(f"  Generated: {_dim(_format_time(report['generated_at']))}")
    print()

    # ── ORB Overview ──
    print(_bold("  ── ORB Overview " + "─" * 53))
    total = orb["total_evaluations"]
    bk = orb["breakouts_detected"]
    pub = orb["published"]

    if total == 0:
        print(f"  {_dim('No ORB evaluations recorded for this date.')}")
        print(f"  {_dim('This could mean the engine was not running during 09:30–11:00 ET,')}")
        print(f"  {_dim('or no 1-minute bar data was available.')}")
    else:
        print(f"  Total evaluations:   {_bold(str(total))}")
        print(f"  Breakouts detected:  {_bold(_green(str(bk)) if bk > 0 else str(bk))}")
        if orb["filter_passed"] > 0 or orb["filter_failed"] > 0:
            print(f"  Filters passed:      {_green(str(orb['filter_passed']))}")
            print(f"  Filters rejected:    {_red(str(orb['filter_failed']))}")
        if orb["cnn_gated"] > 0:
            print(f"  CNN gated:           {_yellow(str(orb['cnn_gated']))}")
        print(f"  Published/alerted:   {_bold(_cyan(str(pub)))}")
    print()

    # ── Per-Symbol Breakdown ──
    if orb["symbols"]:
        print(_bold("  ── Per-Symbol Breakdown " + "─" * 46))
        for sym, data in sorted(orb["symbols"].items()):
            eval_count = data["evaluations"]
            bk_count = data["breakouts"]
            status = _green("● BREAKOUT") if bk_count > 0 else _dim("○ quiet")
            print(f"  {_bold(sym):>8s}  evals={eval_count:<4d}  breakouts={bk_count:<3d}  {status}")

            for d in data["details"]:
                direction = d["direction"]
                dir_color = _green(direction) if direction == "LONG" else _red(direction)
                trigger = _format_price(d["trigger_price"])
                or_range = _format_price(d["or_range"])
                atr = _format_price(d["atr"])

                parts = [
                    f"    {d['time']}",
                    f"{dir_color:>10s}",
                    f"@ {trigger}",
                    f"OR={or_range}",
                    f"ATR={atr}",
                ]

                # CNN probability
                if d.get("cnn_prob") is not None:
                    prob = float(d["cnn_prob"])
                    prob_str = f"CNN={prob:.1%}"
                    if prob >= 0.7:
                        prob_str = _green(prob_str)
                    elif prob >= 0.5:
                        prob_str = _yellow(prob_str)
                    else:
                        prob_str = _red(prob_str)
                    conf = d.get("cnn_confidence", "")
                    parts.append(f"{prob_str} ({conf})")

                # Filter result
                fp = d.get("filter_passed")
                if fp is True:
                    parts.append(_green("✓ filters"))
                elif fp is False:
                    parts.append(_red("✗ filtered"))
                    fs = d.get("filter_summary", "")
                    if fs:
                        parts.append(_dim(f"({fs[:40]})"))

                # Published
                if d.get("published"):
                    parts.append(_cyan("📢 published"))

                print("  ".join(parts))
        print()

    # ── CNN Probability Distribution ──
    if cnn:
        print(_bold("  ── CNN Probability Distribution " + "─" * 38))
        print(f"  Signals scored:  {cnn['count']}")
        print(f"  P(good) range:   {cnn['min']:.3f} – {cnn['max']:.3f}")
        print(f"  P(good) mean:    {cnn['mean']:.3f}")
        print(f"  Above 50%:       {cnn['above_50']}/{cnn['count']}")
        print(f"  Above 70%:       {cnn['above_70']}/{cnn['count']}")

        # Simple ASCII histogram
        if cnn["count"] >= 2:
            print()
            _print_prob_histogram(
                [b.get("cnn_prob") for b in report.get("raw_breakouts", []) if b.get("cnn_prob") is not None]
            )
        print()

    # ── Risk Events ──
    if risk["total_events"] > 0:
        print(_bold("  ── Risk Events " + "─" * 54))
        print(f"  Total:    {risk['total_events']}")
        print(f"  Blocks:   {_red(str(risk['blocks']))}")
        print(f"  Warnings: {_yellow(str(risk['warnings']))}")
        if risk["events"]:
            print()
            for r in risk["events"][:10]:
                etype = r["type"]
                if etype == "block":
                    etype = _red("BLOCK  ")
                elif etype == "warning":
                    etype = _yellow("WARNING")
                else:
                    etype = _dim(f"{etype:7s}")
                sym = r.get("symbol", "")
                reason = r.get("reason", "")
                print(f"    {r['time']}  {etype}  {sym:>6s}  {reason[:50]}")
        print()
    else:
        print(_bold("  ── Risk Events " + "─" * 54))
        print(f"  {_green('No risk events — clean session')}")
        print()

    # ── Model Status ──
    print(_bold("  ── CNN Model Status " + "─" * 50))
    if model.get("available"):
        print(f"  Model:      {_green('✓ available')}  ({model.get('size_mb', '?')} MB)")
        print(f"  Modified:   {_format_time(model.get('last_modified', ''))}")
        if model.get("checkpoint_count"):
            print(f"  Checkpoints: {model['checkpoint_count']}")
        lt = model.get("last_training", {})
        if lt:
            print(
                f"  Last train:  epoch={lt.get('epoch', '?')}"
                f"  val_acc={lt.get('val_acc', '?')}"
                f"  precision={lt.get('val_precision', '?')}"
                f"  recall={lt.get('val_recall', '?')}"
            )
        ds = model.get("dataset_size") or model.get("dataset_stats", {}).get("total_images")
        if ds:
            print(f"  Dataset:     {ds} images")
    else:
        print(f"  Model:      {_yellow('✗ not found')} — CNN inference will be skipped")
        print(f"  {_dim('Run overnight retraining or place breakout_cnn_best.pt in models/')}")
    print()

    # ── Action Items ──
    _print_action_items(report)

    print(_bold("=" * 72))
    print()


def _print_prob_histogram(probs: list[float]) -> None:
    """Print a simple ASCII histogram of CNN probabilities."""
    if not probs:
        return

    buckets = [0] * 10  # 0.0-0.1, 0.1-0.2, ... 0.9-1.0
    for p in probs:
        idx = min(int(p * 10), 9)
        buckets[idx] += 1

    max_count = max(buckets) if buckets else 1
    bar_width = 30

    print("    P(good)    Count  Distribution")
    print("    " + "─" * 50)
    for i, count in enumerate(buckets):
        lo = i / 10
        hi = (i + 1) / 10
        bar_len = int((count / max_count) * bar_width) if max_count > 0 else 0
        bar = "█" * bar_len
        label = f"{lo:.1f}–{hi:.1f}"

        # Colour the bar by probability region
        if i >= 7:
            bar = _green(bar)
        elif i >= 5:
            bar = _yellow(bar)
        else:
            bar = _red(bar)

        print(f"    {label}    {count:>3d}    {bar}")


def _print_action_items(report: dict) -> None:
    """Print contextual action items based on the day's results."""
    items: list[str] = []
    orb = report["orb"]
    cnn = report.get("cnn", {})
    model = report.get("model", {})

    if orb["total_evaluations"] == 0:
        items.append("No ORB evaluations today — check engine logs and 1m bar data availability")

    if orb["breakouts_detected"] > 0 and orb["published"] == 0:
        items.append("Breakouts detected but none published — review filter strictness (ORB_FILTER_GATE env var)")

    if cnn and cnn.get("count", 0) > 0:
        mean_p = cnn.get("mean", 0)
        if mean_p < 0.4:
            items.append(
                f"CNN mean P(good) is low ({mean_p:.2f}) — dataset may need more "
                "positive examples or model may be undertrained"
            )
        elif mean_p > 0.85:
            items.append(
                f"CNN mean P(good) is very high ({mean_p:.2f}) — potential overfitting, check validation metrics"
            )

    if not model.get("available"):
        items.append("No CNN model found — overnight retraining has not run yet or model path is wrong")

    if items:
        print(_bold("  ── Action Items " + "─" * 53))
        for i, item in enumerate(items, 1):
            print(f"  {_yellow(f'{i}.')} {item}")
        print()


# ---------------------------------------------------------------------------
# Multi-day summary
# ---------------------------------------------------------------------------


def build_multi_day_summary(days: int) -> list[dict]:
    """Build reports for the last N days."""
    reports = []
    today = datetime.now(tz=_EST).date()
    for i in range(days):
        d = today - timedelta(days=i)
        report = build_daily_report(d)
        reports.append(report)
    return reports


def print_multi_day_summary(reports: list[dict]) -> None:
    """Print a compact multi-day summary table."""
    print()
    print(_bold("=" * 72))
    print(_bold(f"  📊  ORB SUMMARY — Last {len(reports)} day(s)"))
    print(_bold("=" * 72))
    print()

    # Table header
    header = (
        f"  {'Date':>12s}  {'Day':>9s}  {'Evals':>5s}  {'BOs':>4s}  "
        f"{'Filt✓':>5s}  {'Pub':>4s}  {'CNN μ':>6s}  {'Risk':>5s}"
    )
    print(_bold(header))
    print("  " + "─" * 66)

    total_evals = 0
    total_bos = 0
    total_pub = 0
    total_risk = 0
    days_with_breakouts = 0

    for r in reversed(reports):  # oldest first
        orb = r["orb"]
        cnn = r.get("cnn", {})
        risk = r["risk"]

        evals = orb["total_evaluations"]
        bos = orb["breakouts_detected"]
        filt = orb["filter_passed"]
        pub = orb["published"]
        cnn_mean = f"{cnn['mean']:.2f}" if cnn.get("mean") is not None else "—"
        risk_total = risk["total_events"]

        total_evals += evals
        total_bos += bos
        total_pub += pub
        total_risk += risk_total
        if bos > 0:
            days_with_breakouts += 1

        # Highlight rows with breakouts
        date_str = r["date"]
        day_str = r["day_of_week"][:3]
        bo_str = str(bos)
        if bos > 0:
            bo_str = _green(bo_str)

        risk_str = str(risk_total)
        if risk_total > 0:
            risk_str = _yellow(risk_str)

        print(
            f"  {date_str:>12s}  {day_str:>9s}  {evals:>5d}  {bo_str:>4s}  "
            f"{filt:>5d}  {pub:>4d}  {cnn_mean:>6s}  {risk_str:>5s}"
        )

    # Totals row
    print("  " + "─" * 66)
    print(
        f"  {'TOTAL':>12s}  {len(reports):>7d}d  {total_evals:>5d}  "
        f"{total_bos:>4d}  {'':>5s}  {total_pub:>4d}  {'':>6s}  {total_risk:>5d}"
    )
    print()

    # Summary stats
    print(f"  Days with breakouts: {days_with_breakouts}/{len(reports)}")
    if total_bos > 0:
        pub_rate = total_pub / total_bos * 100
        print(f"  Publication rate:    {pub_rate:.0f}% ({total_pub}/{total_bos} breakouts → published)")
    if total_evals > 0:
        bo_rate = total_bos / total_evals * 100
        print(f"  Breakout rate:       {bo_rate:.1f}% ({total_bos}/{total_evals} evaluations)")
    print()
    print(_bold("=" * 72))
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Daily ORB Report — structured trading-day summary",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python scripts/daily_report.py                # today's report\n"
            "  python scripts/daily_report.py --date 2026-06-15\n"
            "  python scripts/daily_report.py --days 5       # last 5 days summary\n"
            "  python scripts/daily_report.py --json         # JSON output\n"
            "  python scripts/daily_report.py --days 5 --json\n"
        ),
    )
    parser.add_argument(
        "--date",
        "-d",
        type=str,
        default=None,
        help="Target date (YYYY-MM-DD). Defaults to today (ET).",
    )
    parser.add_argument(
        "--days",
        "-n",
        type=int,
        default=None,
        help="Show summary for the last N days (instead of single-day detail).",
    )
    parser.add_argument(
        "--json",
        "-j",
        action="store_true",
        default=False,
        help="Output as JSON instead of formatted text.",
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        default=False,
        help="Disable ANSI colour codes.",
    )

    args = parser.parse_args()

    if args.no_color:
        global _USE_COLOR
        _USE_COLOR = False

    # Determine target date
    if args.date:
        try:
            target = date.fromisoformat(args.date)
        except ValueError:
            print(f"Invalid date format: {args.date} (expected YYYY-MM-DD)", file=sys.stderr)
            sys.exit(1)
    else:
        target = datetime.now(tz=_EST).date()

    # Multi-day mode
    if args.days:
        reports = build_multi_day_summary(args.days)
        if args.json:
            print(json.dumps(reports, indent=2, default=str))
        else:
            print_multi_day_summary(reports)
            # Also show today's detail report
            if reports:
                print_report(reports[0])
        return

    # Single-day mode
    report = build_daily_report(target)

    if args.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        print_report(report)


if __name__ == "__main__":
    main()
