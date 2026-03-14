#!/usr/bin/env python
"""
Live Signal Monitor — Watch ORB + CNN signals in real time
===========================================================

Polls Redis for published ORB signals and displays them in the terminal
with colour-coded output.  Also shows signal ACK status and basic
engine health.

Usage:
    PYTHONPATH=src python scripts/monitor_signals.py
    PYTHONPATH=src python scripts/monitor_signals.py --interval 2
    PYTHONPATH=src python scripts/monitor_signals.py --json

Requirements:
    - Redis must be reachable (uses the same REDIS_URL as the engine)
    - PYTHONPATH must include src/ so lib.core.cache is importable
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from zoneinfo import ZoneInfo

# ---------------------------------------------------------------------------
# Ensure PYTHONPATH is set so we can import from the project
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(_PROJECT_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

_EST = ZoneInfo("America/New_York")

# ---------------------------------------------------------------------------
# Colour helpers (ANSI — works in most terminals, ignored in redirected output)
# ---------------------------------------------------------------------------
_NO_COLOUR = not sys.stdout.isatty() or os.getenv("NO_COLOR")


def _c(code: str, text: str) -> str:
    if _NO_COLOUR:
        return text
    return f"\033[{code}m{text}\033[0m"


def _green(t: str) -> str:
    return _c("32", t)


def _red(t: str) -> str:
    return _c("31", t)


def _yellow(t: str) -> str:
    return _c("33", t)


def _cyan(t: str) -> str:
    return _c("36", t)


def _bold(t: str) -> str:
    return _c("1", t)


def _dim(t: str) -> str:
    return _c("2", t)


# ---------------------------------------------------------------------------
# Redis helpers
# ---------------------------------------------------------------------------


def _get_redis():
    """Return a (cache_get, _r) tuple from the project's cache module."""
    try:
        from lib.core.cache import _r, cache_get

        return cache_get, _r
    except Exception as exc:
        print(_red(f"ERROR: Cannot import lib.core.cache: {exc}"))
        print(_dim("Make sure PYTHONPATH includes the src/ directory and Redis is configured."))
        sys.exit(1)


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------


def _format_signal(signal: dict) -> str:
    """Pretty-print a single ORB signal dict."""
    ts_raw = signal.get("ts") or signal.get("timestamp", "")
    try:
        dt = datetime.fromisoformat(ts_raw.replace("Z", "+00:00")).astimezone(_EST)
        ts_str = dt.strftime("%H:%M:%S ET")
    except Exception:
        ts_str = ts_raw[:19] if ts_raw else "??:??:??"

    direction = signal.get("direction", "???")
    symbol = signal.get("symbol", "???")
    trigger = signal.get("trigger_price", 0)
    cnn_prob = signal.get("cnn_prob")
    session = signal.get("session", "")
    filters = signal.get("filter_summary", "")

    # Direction colouring
    dir_str = _green(f"LONG  {symbol}") if direction == "LONG" else _red(f"SHORT {symbol}")

    # CNN probability colouring
    if cnn_prob is not None:
        prob_val = float(cnn_prob)
        if prob_val >= 0.85:
            cnn_str = _green(f"CNN {prob_val:.3f}")
        elif prob_val >= 0.70:
            cnn_str = _yellow(f"CNN {prob_val:.3f}")
        else:
            cnn_str = _red(f"CNN {prob_val:.3f}")
    else:
        cnn_str = _dim("CNN n/a")

    trigger_str = f"@ {trigger:,.4f}" if isinstance(trigger, (int, float)) else f"@ {trigger}"

    parts = [
        f"[{_cyan(ts_str)}]",
        dir_str,
        trigger_str,
        f"({cnn_str})",
    ]

    if session:
        parts.append(_dim(f"[{session}]"))
    if filters:
        parts.append(_dim(f"F: {filters[:50]}"))

    return "  ".join(parts)


def _format_health(health: dict) -> str:
    """Format engine health data."""
    status = health.get("status", "unknown")
    session = health.get("session", "?")
    pending = health.get("pending_actions", health.get("pending", "?"))
    ts = health.get("timestamp", "")

    healthy = health.get("healthy", False)
    status_str = _green(status) if healthy else _red(status)

    return f"Engine: {status_str} | Session: {_bold(session)} | Pending: {pending} | Last: {ts[:19]}"


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def run_monitor(interval: float = 1.0, json_mode: bool = False) -> None:
    """Main monitoring loop."""
    cache_get, _r = _get_redis()

    print(_bold("=" * 65))
    print(_bold("  ORB + CNN Live Signal Monitor"))
    print(_bold("=" * 65))
    print(_dim(f"  Polling every {interval}s — press Ctrl+C to stop"))
    print()

    last_signal_ts: str | None = None
    signals_seen = 0
    start_time = datetime.now(tz=_EST)

    try:
        while True:
            now = datetime.now(tz=_EST)

            # --- Read latest signal ---
            try:
                raw = cache_get("signals:orb")
                if raw:
                    raw_str = raw.decode() if isinstance(raw, bytes) else raw
                    signal = json.loads(raw_str)
                    ts = signal.get("ts") or signal.get("timestamp")

                    if ts and ts != last_signal_ts:
                        signals_seen += 1
                        last_signal_ts = ts

                        if json_mode:
                            print(json.dumps(signal, indent=2))
                        else:
                            print(_bold(f"  Signal #{signals_seen}:  ") + _format_signal(signal))

                        # Check for ACK
                        if _r:
                            try:
                                ack = _r.get(f"signals:ack:{ts}")
                                if ack:
                                    ack_str = ack.decode() if isinstance(ack, bytes) else ack
                                    print(_dim(f"    -> ACK: {ack_str}"))
                            except Exception:
                                pass
                        print()
            except Exception as exc:
                if not json_mode:
                    print(_dim(f"  (signal read error: {exc})"))

            # --- Engine health (every 10 iterations) ---
            if not json_mode and signals_seen == 0 and int(time.time()) % 10 == 0:
                try:
                    raw_health = cache_get("engine:health")
                    if raw_health:
                        raw_h_str = raw_health.decode() if isinstance(raw_health, bytes) else raw_health
                        health = json.loads(raw_h_str)
                        uptime = (now - start_time).total_seconds()
                        print(
                            f"\r  {_format_health(health)}  | Signals: {signals_seen} | Monitor uptime: {uptime:.0f}s",
                            end="",
                            flush=True,
                        )
                except Exception:
                    pass

            # --- ACK count ---
            if _r and not json_mode and signals_seen > 0 and int(time.time()) % 30 == 0:
                try:
                    ack_keys = list(_r.keys("signals:ack:*"))
                    if ack_keys:
                        print(_dim(f"  ({len(ack_keys)} total ACKs)"))
                except Exception:
                    pass

            time.sleep(interval)

    except KeyboardInterrupt:
        elapsed = (datetime.now(tz=_EST) - start_time).total_seconds()
        print()
        print(_bold("=" * 65))
        print(f"  Monitor stopped — {signals_seen} signal(s) in {elapsed:.0f}s")
        print(_bold("=" * 65))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Live monitor for ORB + CNN trading signals",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=1.0,
        help="Polling interval in seconds (default: 1.0)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="json_mode",
        help="Output raw JSON instead of formatted text",
    )
    args = parser.parse_args()

    run_monitor(interval=args.interval, json_mode=args.json_mode)


if __name__ == "__main__":
    main()
