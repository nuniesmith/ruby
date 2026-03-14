#!/usr/bin/env python3
"""
Offline Training Pipeline
==========================
Generates a CNN training dataset entirely from Postgres-backed bars
(via the data service), then runs training. No external API keys needed
once the DataSyncService has populated the rolling window.

Usage:
    python scripts/train_offline.py [--data-url http://localhost:8000] [--symbols MGC,MES,MNQ]
    python scripts/train_offline.py --check-only  # just verify data availability
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup — ensure project root is on sys.path so we can import lib.*
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"

# Insert src/ at the front so ``import lib.…`` resolves correctly regardless
# of working directory or PYTHONPATH.
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Terminal colours
# ---------------------------------------------------------------------------
BOLD = "\033[1m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
CYAN = "\033[96m"
DIM = "\033[2m"
RESET = "\033[0m"

# ---------------------------------------------------------------------------
# Default symbols — 9 CME micro futures + 3 Kraken crypto
# ---------------------------------------------------------------------------
DEFAULT_FUTURES = [
    "MGC=F",
    "SI=F",
    "HG=F",
    "CL=F",
    "NG=F",
    "ES=F",
    "NQ=F",
    "RTY=F",
    "YM=F",
]

DEFAULT_CRYPTO = [
    "KRAKEN:XBTUSD",
    "KRAKEN:XETHZUSD",
    "KRAKEN:SOLUSD",
]

ALL_DEFAULT_SYMBOLS = DEFAULT_FUTURES + DEFAULT_CRYPTO

# Short aliases that resolve to Yahoo-style or Kraken tickers
SHORT_TO_TICKER = {
    "MGC": "MGC=F",
    "MES": "ES=F",
    "MNQ": "NQ=F",
    "MYM": "YM=F",
    "M2K": "RTY=F",
    "SIL": "SI=F",
    "ZB": "ZB=F",
    "ZN": "ZN=F",
    "ZW": "ZW=F",
    "BTC": "KRAKEN:XBTUSD",
    "ETH": "KRAKEN:XETHZUSD",
    "SOL": "KRAKEN:SOLUSD",
}

# Minimum bars thresholds
MIN_BARS_WARN = 100_000
MIN_BARS_SKIP = 10_000


# ---------------------------------------------------------------------------
# Logging / display helpers
# ---------------------------------------------------------------------------


def _log(msg: str, color: str = "") -> None:
    print(f"{color}{msg}{RESET}")


def _ok(msg: str) -> None:
    _log(f"  ✓ {msg}", GREEN)


def _warn(msg: str) -> None:
    _log(f"  ⚠ {msg}", YELLOW)


def _fail(msg: str) -> None:
    _log(f"  ✗ {msg}", RED)


def _info(msg: str) -> None:
    _log(f"  ℹ {msg}", CYAN)


def _section(title: str) -> None:
    print()
    _log(f"{'─' * 60}", DIM)
    _log(f"  {title}", BOLD)
    _log(f"{'─' * 60}", DIM)
    print()


def _format_duration(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h {m}m {s}s"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------


def _http_get(url: str, timeout: int = 15) -> tuple[int, dict | None]:
    """GET request returning (status_code, json_body). Returns (-1, None) on error."""
    try:
        import requests

        resp = requests.get(url, timeout=timeout)
        try:
            body = resp.json()
        except Exception:
            body = None
        return resp.status_code, body
    except Exception as exc:
        print(f"{DIM}  HTTP GET {url} failed: {exc}{RESET}")
        return -1, None


# ---------------------------------------------------------------------------
# Step 1: Health check / data availability
# ---------------------------------------------------------------------------


def check_data_availability(
    data_url: str,
    symbols: list[str],
    days: int,
) -> dict[str, dict]:
    """Check sync status and bar counts for each symbol.

    Returns a dict of symbol → {bar_count, status, last_synced, skip, warn}.
    """
    _section("Data Availability Check")
    _info(f"Data service URL: {data_url}")
    _info(f"Symbols: {len(symbols)}")
    _info(f"Days requested: {days}")
    print()

    # ── Try sync status endpoint first ──────────────────────────────
    status_url = f"{data_url}/api/data/sync/status"
    _info(f"GET {status_url}")
    code, body = _http_get(status_url)

    sync_meta: dict[str, dict] = {}  # symbol → sync metadata
    if code == 200 and body:
        for entry in body.get("symbols", []):
            sym = entry.get("symbol", "")
            sync_meta[sym] = entry

        svc = body.get("service", {})
        if svc.get("running"):
            _ok(f"DataSyncService is running (cycles completed: {svc.get('cycles_completed', '?')})")
        else:
            _warn("DataSyncService is NOT running — data may be stale")
    else:
        _warn(f"Could not reach sync status endpoint (HTTP {code})")
        _info("Will probe individual symbols via /api/data/bars")

    print()

    # ── Table header ────────────────────────────────────────────────
    hdr = f"  {'Symbol':<22} {'Bars':>10} {'Status':<12} {'Last Synced':<22} {'Action':<10}"
    _log(hdr, BOLD)
    _log(f"  {'─' * 76}", DIM)

    result: dict[str, dict] = {}

    for sym in symbols:
        entry: dict = {
            "bar_count": 0,
            "status": "unknown",
            "last_synced": None,
            "skip": False,
            "warn": False,
        }

        # Check sync metadata first
        meta = sync_meta.get(sym, {})
        if meta:
            entry["bar_count"] = meta.get("bar_count", 0)
            entry["status"] = meta.get("status", "unknown")
            entry["last_synced"] = meta.get("last_synced")

        # If we didn't get bar counts from sync status, probe the bars endpoint
        if entry["bar_count"] == 0:
            probe_url = f"{data_url}/api/data/bars?symbol={sym}&interval=1m&days={days}"
            probe_code, probe_body = _http_get(probe_url, timeout=30)
            if probe_code == 200 and probe_body:
                entry["bar_count"] = probe_body.get("count", 0)
                entry["status"] = "available" if entry["bar_count"] > 0 else "empty"

        # Determine action
        bc = entry["bar_count"]
        if bc < MIN_BARS_SKIP:
            entry["skip"] = True
            action = f"{RED}SKIP{RESET}"
        elif bc < MIN_BARS_WARN:
            entry["warn"] = True
            action = f"{YELLOW}WARN{RESET}"
        else:
            action = f"{GREEN}OK{RESET}"

        last_sync_str = entry["last_synced"] or "—"
        if len(last_sync_str) > 20:
            last_sync_str = last_sync_str[:19]

        print(f"  {sym:<22} {bc:>10,} {entry['status']:<12} {last_sync_str:<22} {action}")
        result[sym] = entry

    print()

    # Summary
    total_bars = sum(e["bar_count"] for e in result.values())
    skip_count = sum(1 for e in result.values() if e["skip"])
    warn_count = sum(1 for e in result.values() if e["warn"])
    ok_count = len(result) - skip_count - warn_count

    _info(f"Total bars across all symbols: {total_bars:,}")
    if ok_count:
        _ok(f"{ok_count} symbols ready")
    if warn_count:
        _warn(f"{warn_count} symbols have low bar count (< {MIN_BARS_WARN:,})")
    if skip_count:
        _fail(f"{skip_count} symbols will be skipped (< {MIN_BARS_SKIP:,} bars)")

    return result


# ---------------------------------------------------------------------------
# Step 2: Generate dataset
# ---------------------------------------------------------------------------


def generate_dataset_offline(
    symbols: list[str],
    days: int,
    output_dir: str,
    data_url: str,
) -> dict | None:
    """Generate the CNN training dataset using the data service for bars.

    Sets ENGINE_DATA_URL so the EngineDataClient routes bar requests through
    the data service (Postgres-backed store).
    """
    _section("Dataset Generation")

    # Point the EngineDataClient at the data service
    os.environ["ENGINE_DATA_URL"] = data_url
    _info(f"ENGINE_DATA_URL set to {data_url}")
    _info(f"Symbols: {', '.join(symbols)}")
    _info(f"Days back: {days}")
    _info(f"Output dir: {output_dir}")
    print()

    try:
        from lib.services.training.dataset_generator import (
            DatasetConfig,
            generate_dataset,
        )
    except ImportError as exc:
        _fail(f"Cannot import dataset_generator: {exc}")
        _info("Make sure you're running from the project root or src/ is on PYTHONPATH")
        return None

    # Reset the EngineDataClient singleton so it picks up the new URL
    try:
        from lib.services.data.engine_data_client import reset_client

        reset_client()
    except ImportError:
        pass

    cfg = DatasetConfig()
    cfg.output_dir = output_dir
    cfg.image_dir = os.path.join(output_dir, "images")
    cfg.bars_source = "engine"
    cfg.skip_existing = True

    _info("Starting dataset generation...")
    start = time.monotonic()

    try:
        stats = generate_dataset(
            symbols=symbols,
            days_back=days,
            config=cfg,
        )
    except Exception as exc:
        _fail(f"Dataset generation failed: {exc}")
        import traceback

        traceback.print_exc()
        return None

    elapsed = time.monotonic() - start

    _section("Dataset Generation Results")
    _info(f"Time elapsed: {_format_duration(elapsed)}")

    stats_dict = stats.to_dict() if hasattr(stats, "to_dict") else {}

    total_images = stats_dict.get("total_images", getattr(stats, "total_images", 0))
    csv_path = stats_dict.get("csv_path", getattr(stats, "csv_path", ""))

    _ok(f"Total images generated: {total_images:,}")
    if csv_path:
        _ok(f"CSV manifest: {csv_path}")

    # Per-symbol counts
    per_symbol = stats_dict.get("per_symbol", getattr(stats, "per_symbol", {}))
    if per_symbol:
        print()
        _log(f"  {'Symbol':<20} {'Images':>10}", BOLD)
        _log(f"  {'─' * 30}", DIM)
        for sym, count in sorted(per_symbol.items()):
            count_val = count if isinstance(count, int) else count.get("images", count.get("total", 0))
            print(f"  {sym:<20} {count_val:>10,}")

    # Dataset split summary
    _print_split_summary(output_dir)

    return stats_dict


def _print_split_summary(output_dir: str) -> None:
    """Print train/val split stats if split CSVs exist."""
    import csv as csv_mod

    for split_name in ("train", "val"):
        csv_path = os.path.join(output_dir, f"{split_name}.csv")
        if os.path.isfile(csv_path):
            try:
                with open(csv_path, "r") as f:
                    reader = csv_mod.reader(f)
                    header = next(reader, None)
                    row_count = sum(1 for _ in reader)
                _info(f"{split_name}.csv: {row_count:,} rows")
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Step 3: Trigger training (optional)
# ---------------------------------------------------------------------------


def trigger_training(data_url: str, trainer_url: str | None = None) -> bool:
    """Trigger training via the trainer service.

    If --train is passed, this calls run_full_retrain.py or the trainer HTTP API.
    """
    _section("Training")

    # Try the trainer URL (typically port 8200)
    url = trainer_url or os.getenv("TRAINER_URL", "http://localhost:8200")

    _info(f"Trainer URL: {url}")

    # Try triggering via HTTP POST
    try:
        import requests

        body = {
            "train_mode": "combined",
            "step": "train",
        }
        _info(f"POST {url}/train")
        resp = requests.post(f"{url}/train", json=body, timeout=30)
        if resp.status_code in (200, 202):
            _ok(f"Training triggered (HTTP {resp.status_code})")
            return True
        else:
            _warn(f"Trainer returned HTTP {resp.status_code}: {resp.text[:200]}")
    except Exception as exc:
        _warn(f"Could not reach trainer at {url}: {exc}")

    # Fall back to running the retrain script directly
    _info("Attempting to run training in-process...")
    try:
        retrain_script = SCRIPT_DIR / "run_full_retrain.py"
        if retrain_script.is_file():
            _info(f"Delegating to {retrain_script}")
            import subprocess

            result = subprocess.run(
                [sys.executable, str(retrain_script), "--start-step", "7"],
                cwd=str(PROJECT_ROOT),
                timeout=7200,
            )
            if result.returncode == 0:
                _ok("Training completed successfully")
                return True
            else:
                _fail(f"Training script exited with code {result.returncode}")
                return False
        else:
            _warn("run_full_retrain.py not found — skipping training")
            return False
    except subprocess.TimeoutExpired:
        _fail("Training timed out (2 hours)")
        return False
    except Exception as exc:
        _fail(f"Training failed: {exc}")
        return False


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def _resolve_symbols(symbols_str: str) -> list[str]:
    """Resolve comma-separated symbol names to tickers.

    Accepts short names (MGC, MES, BTC) and full tickers (MGC=F, KRAKEN:XBTUSD).
    """
    raw = [s.strip() for s in symbols_str.split(",") if s.strip()]
    resolved = []
    for s in raw:
        if s.upper() == "ALL":
            return list(ALL_DEFAULT_SYMBOLS)
        resolved.append(SHORT_TO_TICKER.get(s, s))
    return resolved


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline Training Pipeline — generate CNN dataset from Postgres-backed bars",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/train_offline.py --check-only
  python scripts/train_offline.py --symbols MGC,MES,MNQ --days 180
  python scripts/train_offline.py --data-url http://myserver:8000 --train
  python scripts/train_offline.py --symbols ALL --output-dir dataset_full
        """,
    )
    parser.add_argument(
        "--data-url",
        default=os.getenv("ENGINE_DATA_URL", os.getenv("DATA_SERVICE_URL", "http://localhost:8000")),
        help="Base URL of the data service (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--symbols",
        default="ALL",
        help="Comma-separated symbols (short: MGC,MES,BTC or full: MGC=F,KRAKEN:XBTUSD). "
        "Use ALL for all 9 futures + 3 crypto (default: ALL)",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=365,
        help="Days of historical bar data to use (default: 365)",
    )
    parser.add_argument(
        "--output-dir",
        default="dataset",
        help="Output directory for generated dataset (default: dataset)",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only check data availability — do not generate dataset or train",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Trigger training after dataset generation",
    )
    parser.add_argument(
        "--trainer-url",
        default=None,
        help="Trainer service URL (default: http://localhost:8200). Only used with --train.",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    data_url = args.data_url.rstrip("/")
    symbols = _resolve_symbols(args.symbols)
    days = args.days
    output_dir = args.output_dir

    print()
    _log(f"{'═' * 60}", CYAN)
    _log(f"  Offline Training Pipeline", BOLD)
    _log(f"{'═' * 60}", CYAN)
    print()
    _info(f"Data service: {data_url}")
    _info(f"Symbols:      {len(symbols)}")
    _info(f"Days back:    {days}")
    _info(f"Output dir:   {output_dir}")
    if args.check_only:
        _info("Mode:         CHECK ONLY")
    elif args.train:
        _info("Mode:         GENERATE + TRAIN")
    else:
        _info("Mode:         GENERATE ONLY")

    overall_start = time.monotonic()

    # ── Step 1: Check data availability ─────────────────────────────
    availability = check_data_availability(data_url, symbols, days)

    # Filter out symbols that should be skipped
    viable_symbols = [s for s in symbols if not availability.get(s, {}).get("skip", False)]
    skipped = len(symbols) - len(viable_symbols)

    if not viable_symbols:
        print()
        _fail("No symbols have sufficient data for dataset generation.")
        _info("Ensure the DataSyncService has run at least one full cycle.")
        _info(f"  Check: curl {data_url}/api/data/sync/status")
        return 1

    if skipped:
        print()
        _warn(f"{skipped} symbol(s) skipped due to insufficient data")
        _info(f"Proceeding with {len(viable_symbols)} symbols: {', '.join(viable_symbols)}")

    if args.check_only:
        print()
        _ok("Data availability check complete. Use without --check-only to generate dataset.")
        return 0

    # ── Step 2: Generate dataset ────────────────────────────────────
    stats = generate_dataset_offline(
        symbols=viable_symbols,
        days=days,
        output_dir=output_dir,
        data_url=data_url,
    )

    if stats is None:
        _fail("Dataset generation failed")
        return 1

    total_images = stats.get("total_images", 0)
    if total_images == 0:
        _warn("Dataset generation produced 0 images")
        _info("This may indicate the data doesn't have enough trading sessions")
        # Don't fail — could be a partial run
    else:
        _ok(f"Dataset ready: {total_images:,} images")

    # ── Step 3: Training (optional) ─────────────────────────────────
    if args.train:
        success = trigger_training(data_url, trainer_url=args.trainer_url)
        if not success:
            _warn("Training step failed — dataset is still available for manual training")
    else:
        print()
        _info("Skipping training (use --train to trigger automatically)")

    # ── Summary ─────────────────────────────────────────────────────
    overall_elapsed = time.monotonic() - overall_start

    _section("Pipeline Summary")
    _info(f"Total time:     {_format_duration(overall_elapsed)}")
    _info(f"Symbols:        {len(viable_symbols)}")
    _info(f"Total images:   {total_images:,}")
    _info(f"Output dir:     {os.path.abspath(output_dir)}")
    print()
    _ok("Offline training pipeline completed ✓")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
