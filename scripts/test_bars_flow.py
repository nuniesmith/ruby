#!/usr/bin/env python3
"""
test_bars_flow.py — End-to-end bars data flow test

Tests the full chain the trainer uses:
  browser/trainer → engine /bars/{symbol} → Massive/Kraken/DB → dataset_generator

Usage:
    python3 scripts/test_bars_flow.py [--base-url http://localhost:8050] [--days 3]
"""

import argparse
import json
import sys
import time
import urllib.request
from typing import Any

# ── Symbols exactly as the trainer sends them ────────────────────────────────
TRAINER_SYMBOLS = [
    "MGC",
    "SIL",
    "MHG",
    "MCL",
    "MNG",  # metals / energy
    "MES",
    "MNQ",
    "M2K",
    "MYM",  # equity index
    "6E",
    "6B",
    "6J",
    "6A",
    "6C",
    "6S",  # FX
    "ZN",
    "ZB",  # rates
    "ZC",
    "ZS",
    "ZW",  # ag
    "MBT",
    "MET",  # CME crypto futures
    "BTC",
    "ETH",
    "SOL",  # Kraken spot crypto
]

# Expected data source per symbol family
EXPECTED_SOURCES = {
    "BTC": "kraken",
    "ETH": "kraken",
    "SOL": "kraken",
    "MBT": "massive",
    "MET": "massive",
}

# ── ANSI colours ─────────────────────────────────────────────────────────────
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"


def _get(url: str, timeout: int = 20) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read())


def check_health(base: str) -> bool:
    print(f"\n{BOLD}=== [1/4] Engine /health ==={RESET}")
    try:
        d = _get(f"{base}/health", timeout=10)
        status = d.get("status", "?")
        colour = GREEN if status == "ok" else YELLOW
        print(f"  Status : {colour}{status}{RESET}")

        components = d.get("components", {})
        for name, info in components.items():
            s = info.get("status", "?") if isinstance(info, dict) else str(info)
            c = GREEN if s in ("ok", "running") else (YELLOW if s == "degraded" else RED)
            extra = ""
            if isinstance(info, dict):
                if info.get("error"):
                    extra = f"  {YELLOW}({info['error'][:80]}){RESET}"
                elif info.get("connected") is False:
                    extra = f"  {YELLOW}(disconnected){RESET}"
            print(f"  {name:<16} {c}{s}{RESET}{extra}")
        return status in ("ok", "degraded")
    except Exception as e:
        print(f"  {RED}FAILED: {e}{RESET}")
        return False


def check_assets(base: str) -> list[str]:
    print(f"\n{BOLD}=== [2/4] /bars/assets — known symbols ==={RESET}")
    try:
        d = _get(f"{base}/bars/assets", timeout=10)
        assets = d.get("assets", [])
        tickers_with_data = [a["ticker"].split("=")[0].replace("KRAKEN:", "") for a in assets if a.get("has_data")]
        print(f"  Total assets registered : {len(assets)}")
        print(f"  Assets with cached data : {len(tickers_with_data)}")
        return tickers_with_data
    except Exception as e:
        print(f"  {RED}FAILED: {e}{RESET}")
        return []


def check_bars(base: str, days: int) -> dict[str, Any]:
    print(f"\n{BOLD}=== [3/4] /bars/{{symbol}} — {len(TRAINER_SYMBOLS)} trainer symbols x {days}d ==={RESET}")

    results: dict[str, Any] = {}
    ok_count = 0
    gap_count = 0  # has bars but sparse / gaps

    for sym in TRAINER_SYMBOLS:
        url = f"{base}/bars/{sym}?interval=1m&days_back={days}"
        t0 = time.monotonic()
        try:
            d = _get(url, timeout=25)
            elapsed = time.monotonic() - t0
            bc = d.get("bar_count", 0)
            src = d.get("source", "?")
            err = d.get("error", "")

            # Rough bar-count sanity: 1m bars, ~390 RTH bars/day for US futures
            # crypto runs ~1440/day; use 200/day as a loose floor
            expected_min = max(50, days * 200)
            sparse = bc > 0 and bc < expected_min

            results[sym] = {
                "bar_count": bc,
                "source": src,
                "error": err,
                "elapsed_ms": round(elapsed * 1000),
                "sparse": sparse,
            }

            if bc > 0:
                ok_count += 1
                colour = YELLOW if sparse else GREEN
                tag = " (sparse)" if sparse else ""
                if sparse:
                    gap_count += 1
                exp_src = EXPECTED_SOURCES.get(sym, "")
                src_warn = ""
                if exp_src and exp_src not in src.lower():
                    src_warn = f" {YELLOW}[expected {exp_src}]{RESET}"
                print(
                    f"  {colour}✓{RESET} {sym:<6} {bc:>5} bars  src={src:<12} {elapsed * 1000:>5.0f}ms{tag}{src_warn}"
                )
            else:
                colour = RED
                print(
                    f"  {colour}✗{RESET} {sym:<6}   0 bars  "
                    f"{'err=' + err[:40] if err else 'no data':<44} "
                    f"{elapsed * 1000:>5.0f}ms"
                )

        except Exception as e:
            elapsed = time.monotonic() - t0
            results[sym] = {"bar_count": 0, "source": "?", "error": str(e), "elapsed_ms": round(elapsed * 1000)}
            print(f"  {RED}✗{RESET} {sym:<6} EXCEPTION: {e}")

    total = len(TRAINER_SYMBOLS)
    missing = total - ok_count
    colour = GREEN if missing == 0 else (YELLOW if missing <= 3 else RED)
    print(
        f"\n  Summary: {colour}{ok_count}/{total} symbols have data{RESET}"
        + (f"  {YELLOW}({gap_count} sparse){RESET}" if gap_count else "")
    )
    return results


def check_dataset_generator(base: str, days: int) -> None:
    print(f"\n{BOLD}=== [4/4] Dataset generator bars resolution (via engine source) ==={RESET}")

    # Simulate what dataset_generator.load_bars() does when source=engine:
    # it calls GET /bars/{symbol}?interval=1m&days_back=N&auto_fill=true
    test_symbols = ["MGC", "MES", "BTC", "6E", "MBT"]
    print(f"  Testing auto_fill=true for {test_symbols} ...")

    for sym in test_symbols:
        url = f"{base}/bars/{sym}?interval=1m&days_back={days}&auto_fill=true"
        t0 = time.monotonic()
        try:
            d = _get(url, timeout=30)
            elapsed = time.monotonic() - t0
            bc = d.get("bar_count", 0)
            src = d.get("source", "?")
            filled = d.get("filled", False)
            added = d.get("bars_added", 0)
            fill_err = d.get("fill_error") or ""

            colour = GREEN if bc > 0 else RED
            fill_info = f"  filled={filled} added={added}" if filled else ""
            fill_warn = f"  {YELLOW}fill_error: {fill_err[:60]}{RESET}" if fill_err else ""
            print(
                f"  {colour}{'✓' if bc > 0 else '✗'}{RESET} {sym:<6} "
                f"{bc:>5} bars  src={src:<12} {elapsed * 1000:>5.0f}ms"
                f"{fill_info}{fill_warn}"
            )
        except Exception as e:
            print(f"  {RED}✗{RESET} {sym:<6} EXCEPTION: {e}")

    # Also confirm the engine-sourced DataResolver path that _load_bars_from_engine uses
    print("\n  Confirming engine URL used by trainer DataResolver ...")
    try:
        # The trainer reads ENGINE_DATA_URL; locally that's the data service
        import os

        engine_url = os.environ.get("ENGINE_DATA_URL", base)
        d = _get(f"{engine_url}/bars/MGC?interval=1m&days_back=1", timeout=15)
        bc = d.get("bar_count", 0)
        print(f"  ENGINE_DATA_URL={engine_url}")
        print(f"  MGC 1-day check: {GREEN if bc > 0 else RED}{bc} bars{RESET}")
    except Exception as e:
        print(f"  {RED}Could not confirm ENGINE_DATA_URL: {e}{RESET}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Test bars data flow end-to-end")
    parser.add_argument(
        "--base-url", default="http://localhost:8050", help="Data service base URL (default: http://localhost:8050)"
    )
    parser.add_argument("--days", type=int, default=3, help="days_back to request (default: 3)")
    args = parser.parse_args()

    base = args.base_url.rstrip("/")
    days = args.days

    print(f"{BOLD}{CYAN}Ruby Futures — Bars Data Flow Test{RESET}")
    print(f"Data service : {base}")
    print(f"Days back    : {days}")

    healthy = check_health(base)
    if not healthy:
        print(f"\n{RED}Engine health check failed — aborting further tests.{RESET}")
        return 1

    check_assets(base)
    bar_results = check_bars(base, days)
    check_dataset_generator(base, days)

    # Final verdict
    missing = [s for s, r in bar_results.items() if r["bar_count"] == 0]
    print(f"\n{BOLD}=== Result ==={RESET}")
    if not missing:
        print(
            f"{GREEN}{BOLD}All {len(TRAINER_SYMBOLS)} symbols have bar data.  "
            f"Trainer dataset generation should succeed.{RESET}"
        )
        return 0
    elif len(missing) <= 3:
        print(f"{YELLOW}{BOLD}{len(missing)} symbol(s) missing data: {', '.join(missing)}{RESET}")
        print(f"{YELLOW}Training will skip these symbols but can proceed.{RESET}")
        return 0
    else:
        print(f"{RED}{BOLD}{len(missing)} symbol(s) missing data: {', '.join(missing)}{RESET}")
        print(f"{RED}Too many symbols missing — trainer dataset generation will be severely impacted.{RESET}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
