#!/usr/bin/env python3
"""
run_full_retrain.py — Full v9 Retrain Pipeline Orchestrator
=============================================================
Standalone script that orchestrates the complete v9 CNN retrain pipeline
by making HTTP calls to the trainer server.  No server imports needed —
everything is done via ``requests``.

Pipeline steps:
  1. Fix dataset paths   — strip Docker ``/app/`` prefix from CSV image paths
  2. Validate dataset     — check CSV integrity and image coverage
  3. Load data            — fetch 365 days of bars via ``POST /train`` (step=load_data)
  4. Generate dataset     — render chart images + labels.csv (step=generate_dataset)
  5. Validate again       — re-check after generation (look for missing images)
  6. Repair if needed     — re-generate any missing images via ``POST /train/repair``
  7. Train                — combined-mode training (step=train)
  8. Compare with champion— compare new model metrics vs existing champion

Usage:
    # Full pipeline:
    python scripts/run_full_retrain.py

    # Start from a specific step (skip earlier steps):
    python scripts/run_full_retrain.py --start-step 3

    # Dry run — show what would happen:
    python scripts/run_full_retrain.py --dry-run

    # Custom trainer URL:
    python scripts/run_full_retrain.py --trainer-url http://oryx:8200

    # Skip repair step even if missing images are found:
    python scripts/run_full_retrain.py --skip-repair

    # Custom training hyperparameters:
    python scripts/run_full_retrain.py --epochs 80 --batch-size 32 --patience 12

Exit codes:
    0 — pipeline completed successfully
    1 — pipeline failed at some step
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
import textwrap
import time
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

# ---------------------------------------------------------------------------
# Terminal colours (match existing scripts/ style)
# ---------------------------------------------------------------------------
BOLD = "\033[1m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
CYAN = "\033[96m"
DIM = "\033[2m"
RESET = "\033[0m"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_TRAINER_URL = "http://localhost:8200"
DEFAULT_DAYS_BACK = 365
DEFAULT_EPOCHS = 80
DEFAULT_BATCH_SIZE = 32
DEFAULT_PATIENCE = 12
POLL_INTERVAL = 30  # seconds between status polls
MAX_WAIT_TRAIN = 7200  # 2 hours max per training step

DOCKER_PREFIX = "/app/"

ALL_SYMBOLS = ["M2K", "MES", "MGC", "MNQ", "MYM", "SIL", "ZB", "ZN", "ZW"]

CSV_FILES = [
    PROJECT_ROOT / "dataset" / "labels.csv",
    PROJECT_ROOT / "dataset" / "train.csv",
    PROJECT_ROOT / "dataset" / "val.csv",
]

TERMINAL_STATES = {"done", "failed", "cancelled", "idle"}

TOTAL_STEPS = 8


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------
def _log(msg: str, color: str = "") -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"{DIM}[{ts}]{RESET} {color}{msg}{RESET}")


def _ok(msg: str) -> None:
    _log(f"✓ {msg}", GREEN)


def _fail(msg: str) -> None:
    _log(f"✗ {msg}", RED)


def _info(msg: str) -> None:
    _log(f"  {msg}", CYAN)


def _warn(msg: str) -> None:
    _log(f"⚠ {msg}", YELLOW)


def _dim(msg: str) -> None:
    _log(f"  {msg}", DIM)


def _section(title: str) -> None:
    print(f"\n{BOLD}{'─' * 70}{RESET}")
    _log(title, BOLD)
    print(f"{BOLD}{'─' * 70}{RESET}")


def _banner(title: str) -> None:
    print()
    print(f"{BOLD}{'=' * 70}{RESET}")
    print(f"{BOLD}  {title}{RESET}")
    print(f"{BOLD}{'=' * 70}{RESET}")
    print()


def _step_header(step_num: int, title: str) -> None:
    print()
    print(f"{BOLD}{'━' * 70}{RESET}")
    _log(f"Step {step_num}/{TOTAL_STEPS}: {title}", BOLD)
    print(f"{BOLD}{'━' * 70}{RESET}")
    print()


# ---------------------------------------------------------------------------
# HTTP helpers (use requests library)
# ---------------------------------------------------------------------------
def _http_get(url: str, timeout: float = 30) -> dict | None:
    """Simple HTTP GET returning parsed JSON or None."""
    import requests

    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        return resp.json()
    except (requests.RequestException, ValueError):
        return None


def _http_post(url: str, body: dict | None = None, timeout: float = 30) -> tuple[int, dict | None]:
    """Simple HTTP POST returning (status_code, parsed_json)."""
    import requests

    try:
        resp = requests.post(url, json=body or {}, timeout=timeout)
        try:
            data = resp.json()
        except ValueError:
            data = None
        return resp.status_code, data
    except requests.RequestException:
        return 0, None


# ---------------------------------------------------------------------------
# Duration formatting
# ---------------------------------------------------------------------------
def _format_duration(seconds: float) -> str:
    """Format seconds into a human-readable duration string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    secs = seconds % 60
    if minutes < 60:
        return f"{minutes}m {secs:.0f}s"
    hours = minutes // 60
    mins = minutes % 60
    return f"{hours}h {mins}m {secs:.0f}s"


# ---------------------------------------------------------------------------
# Poll training status until terminal state
# ---------------------------------------------------------------------------
def _poll_until_done(
    base_url: str,
    label: str,
    max_wait: float = MAX_WAIT_TRAIN,
) -> dict | None:
    """Poll GET /status every POLL_INTERVAL seconds until a terminal state.

    Returns the final status dict, or None if polling failed entirely.
    """
    start_time = time.monotonic()
    last_status = ""
    last_progress = ""

    while True:
        elapsed = time.monotonic() - start_time
        if elapsed > max_wait:
            _fail(f"[{label}] Timed out after {_format_duration(elapsed)}")
            return None

        status = _http_get(f"{base_url}/status")
        if not status:
            _warn(f"[{label}] Failed to poll /status — retrying in {POLL_INTERVAL}s...")
            time.sleep(POLL_INTERVAL)
            continue

        current_status = status.get("status", "unknown")
        current_progress = status.get("progress", "")

        # Log transitions and periodic updates
        if current_status != last_status or current_progress != last_progress:
            elapsed_str = _format_duration(elapsed)

            if current_status == "generating_dataset":
                color = YELLOW
            elif current_status in ("training", "evaluating"):
                color = CYAN
            elif current_status in ("promoting", "done"):
                color = GREEN
            elif current_status == "failed":
                color = RED
            elif current_status == "cancelled":
                color = YELLOW
            else:
                color = DIM

            status_line = f"[{label}] [{elapsed_str}] {current_status}"
            if current_progress:
                status_line += f" — {current_progress}"
            _log(f"  {status_line}", color)

            last_status = current_status
            last_progress = current_progress

        # Terminal states
        if current_status in TERMINAL_STATES:
            return status

        time.sleep(POLL_INTERVAL)


# ---------------------------------------------------------------------------
# Step 1: Fix dataset paths (inline logic)
# ---------------------------------------------------------------------------
def step_fix_dataset_paths(dry_run: bool = False) -> tuple[bool, int]:
    """Strip Docker /app/ prefix from dataset CSV image paths.

    Returns (success, total_fixed_count).
    """
    _step_header(1, "Fix Dataset Paths")
    _info(f"Stripping Docker '{DOCKER_PREFIX}' prefix from CSV image paths")
    _info(f"Project root: {PROJECT_ROOT}")
    print()

    # Change to project root so relative paths work
    os.chdir(str(PROJECT_ROOT))

    total_fixed = 0
    total_rows = 0
    files_processed = 0

    for csv_path in CSV_FILES:
        if not csv_path.is_file():
            _dim(f"Skipping (not found): {csv_path.name}")
            continue

        files_processed += 1

        # Read CSV
        rows: list[dict] = []
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None or "image_path" not in reader.fieldnames:
                _warn(f"No 'image_path' column in {csv_path.name} — skipping")
                continue
            fieldnames = list(reader.fieldnames)
            for row in reader:
                rows.append(row)

        total_rows += len(rows)

        # Count and fix docker-prefixed paths
        fixed_count = 0
        for row in rows:
            p = str(row.get("image_path", "")).strip()
            if p.startswith(DOCKER_PREFIX):
                row["image_path"] = p[len(DOCKER_PREFIX) :]
                fixed_count += 1

        total_fixed += fixed_count

        if fixed_count == 0:
            _ok(f"{csv_path.name}: all {len(rows):,d} paths already local")
            continue

        if dry_run:
            _warn(f"{csv_path.name}: would fix {fixed_count:,d} / {len(rows):,d} paths (dry run)")
            continue

        # Create backup
        bak_path = csv_path.with_suffix(".csv.bak")
        if not bak_path.exists():
            shutil.copy2(csv_path, bak_path)
            _dim(f"Backup created: {bak_path.name}")

        # Write fixed CSV
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        _ok(f"{csv_path.name}: fixed {fixed_count:,d} / {len(rows):,d} paths")

    print()
    if total_fixed == 0:
        _ok(f"All paths already correct across {files_processed} file(s) ({total_rows:,d} rows)")
    elif dry_run:
        _warn(f"DRY RUN: would fix {total_fixed:,d} paths across {files_processed} file(s)")
    else:
        _ok(f"Fixed {total_fixed:,d} paths across {files_processed} file(s)")

    return True, total_fixed


# ---------------------------------------------------------------------------
# Step 2 / 5: Validate dataset (inline logic)
# ---------------------------------------------------------------------------
def step_validate_dataset(step_num: int, label: str = "Validate Dataset") -> tuple[bool, int]:
    """Validate dataset CSVs — check image existence.

    Returns (all_images_found, missing_count).
    """
    _step_header(step_num, label)

    os.chdir(str(PROJECT_ROOT))

    labels_csv = PROJECT_ROOT / "dataset" / "labels.csv"
    if not labels_csv.is_file():
        _warn("dataset/labels.csv not found — nothing to validate")
        return True, 0

    # Read CSV
    rows: list[dict] = []
    with open(labels_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            _fail("labels.csv has no header row")
            return False, 0
        for row in reader:
            rows.append(row)

    total = len(rows)
    _info(f"Total rows in labels.csv: {total:,d}")

    if "image_path" not in rows[0] if rows else True:
        _warn("No 'image_path' column — cannot verify images")
        return True, 0

    # Check images
    missing = 0
    empty = 0
    found = 0
    missing_samples: list[str] = []
    label_counts: dict[str, int] = {}
    symbol_counts: dict[str, int] = {}

    for row in rows:
        p = str(row.get("image_path", "")).strip()

        # Track label distribution
        lbl = row.get("label", "unknown")
        label_counts[lbl] = label_counts.get(lbl, 0) + 1

        # Track symbol distribution
        sym = row.get("symbol", "unknown")
        symbol_counts[sym] = symbol_counts.get(sym, 0) + 1

        if not p or p.lower() == "nan":
            empty += 1
            continue

        if os.path.isfile(p):
            found += 1
        else:
            missing += 1
            if len(missing_samples) < 5:
                missing_samples.append(p)

    # Report
    _info(f"Images found:    {found:,d}")
    if empty:
        _warn(f"Empty paths:     {empty:,d}")
    if missing:
        _fail(f"Missing images:  {missing:,d}")
        for sample in missing_samples:
            _dim(f"  Missing: {sample}")
        if missing > len(missing_samples):
            _dim(f"  ... and {missing - len(missing_samples):,d} more")
    else:
        _ok(f"All {found:,d} images verified on disk")

    # Label distribution
    print()
    _info("Label distribution:")
    for lbl_name in sorted(label_counts.keys()):
        _info(f"  {lbl_name:<25} {label_counts[lbl_name]:>6,d}")

    # Symbol distribution
    print()
    _info("Symbol distribution:")
    for sym_name in sorted(symbol_counts.keys()):
        _info(f"  {sym_name:<10} {symbol_counts[sym_name]:>6,d}")

    print()
    coverage_pct = (found / total * 100) if total > 0 else 0.0
    _info(f"Coverage: {coverage_pct:.1f}% ({found:,d} / {total:,d})")

    return missing == 0, missing


# ---------------------------------------------------------------------------
# Step 3: Load data (fetch bars)
# ---------------------------------------------------------------------------
def step_load_data(base_url: str, args: argparse.Namespace) -> bool:
    """POST /train with step=load_data to fetch 365 days of bars."""
    _step_header(3, "Load Data (Fetch Bars)")
    _info(f"Fetching {args.days_back} days of bar data for {len(ALL_SYMBOLS)} symbols")
    _info(f"Symbols: {', '.join(ALL_SYMBOLS)}")
    print()

    if args.dry_run:
        body = {
            "symbols": ALL_SYMBOLS,
            "train_mode": "combined",
            "days_back": args.days_back,
            "step": "load_data",
        }
        _warn("DRY RUN — would POST to /train with:")
        print(f"{DIM}{json.dumps(body, indent=2)}{RESET}")
        return True

    body = {
        "symbols": ALL_SYMBOLS,
        "train_mode": "combined",
        "days_back": args.days_back,
        "step": "load_data",
    }

    _info(f"POST {base_url}/train (step=load_data)")
    status_code, resp = _http_post(f"{base_url}/train", body)

    if status_code == 202:
        _ok("Load data started (202 Accepted)")
    elif status_code == 409:
        _warn("Server busy (409 Conflict) — waiting for current operation...")
    else:
        _fail(f"POST /train failed with status {status_code}")
        if resp:
            _fail(f"  Response: {json.dumps(resp)}")
        return False

    # Poll
    _info(f"Polling status every {POLL_INTERVAL}s...")
    final = _poll_until_done(base_url, "load_data")

    if final is None:
        _fail("Load data timed out")
        return False

    if final.get("status") == "done":
        _ok("Bar data loaded successfully")
        return True
    elif final.get("status") == "failed":
        _fail(f"Load data failed: {final.get('error', 'unknown')}")
        return False
    else:
        _warn(f"Load data ended with status: {final.get('status')}")
        return final.get("status") == "idle"  # idle after load_data is OK


# ---------------------------------------------------------------------------
# Step 4: Generate dataset
# ---------------------------------------------------------------------------
def step_generate_dataset(base_url: str, args: argparse.Namespace) -> bool:
    """POST /train with step=generate_dataset to render images + labels.csv."""
    _step_header(4, "Generate Dataset")
    _info(f"Rendering chart images + labels.csv for {len(ALL_SYMBOLS)} symbols")
    print()

    if args.dry_run:
        body = {
            "symbols": ALL_SYMBOLS,
            "train_mode": "combined",
            "days_back": args.days_back,
            "step": "generate_dataset",
        }
        _warn("DRY RUN — would POST to /train with:")
        print(f"{DIM}{json.dumps(body, indent=2)}{RESET}")
        return True

    body = {
        "symbols": ALL_SYMBOLS,
        "train_mode": "combined",
        "days_back": args.days_back,
        "step": "generate_dataset",
    }

    _info(f"POST {base_url}/train (step=generate_dataset)")
    status_code, resp = _http_post(f"{base_url}/train", body)

    if status_code == 202:
        _ok("Dataset generation started (202 Accepted)")
    elif status_code == 409:
        _warn("Server busy (409 Conflict) — waiting for current operation...")
    else:
        _fail(f"POST /train failed with status {status_code}")
        if resp:
            _fail(f"  Response: {json.dumps(resp)}")
        return False

    # Poll
    _info(f"Polling status every {POLL_INTERVAL}s...")
    final = _poll_until_done(base_url, "generate_dataset")

    if final is None:
        _fail("Dataset generation timed out")
        return False

    if final.get("status") == "done":
        _ok("Dataset generated successfully")
        return True
    elif final.get("status") == "failed":
        _fail(f"Dataset generation failed: {final.get('error', 'unknown')}")
        return False
    else:
        _warn(f"Dataset generation ended with status: {final.get('status')}")
        return final.get("status") == "idle"


# ---------------------------------------------------------------------------
# Step 6: Repair dataset (re-generate missing images)
# ---------------------------------------------------------------------------
def step_repair_dataset(base_url: str, args: argparse.Namespace) -> bool:
    """POST /train/repair to re-generate missing images."""
    _step_header(6, "Repair Dataset (Re-generate Missing Images)")

    if args.dry_run:
        body = {
            "symbols": ALL_SYMBOLS,
            "days_back": args.days_back,
        }
        _warn("DRY RUN — would POST to /train/repair with:")
        print(f"{DIM}{json.dumps(body, indent=2)}{RESET}")
        return True

    body = {
        "symbols": ALL_SYMBOLS,
        "days_back": args.days_back,
    }

    _info(f"POST {base_url}/train/repair")
    status_code, resp = _http_post(f"{base_url}/train/repair", body)

    if status_code == 202:
        _ok("Repair started (202 Accepted)")
    elif status_code == 409:
        _warn("Server busy (409 Conflict) — waiting for current operation...")
    else:
        _fail(f"POST /train/repair failed with status {status_code}")
        if resp:
            _fail(f"  Response: {json.dumps(resp)}")
        return False

    # Poll
    _info(f"Polling status every {POLL_INTERVAL}s...")
    final = _poll_until_done(base_url, "repair")

    if final is None:
        _fail("Repair timed out")
        return False

    if final.get("status") == "done":
        _ok("Repair completed successfully")
        return True
    elif final.get("status") == "failed":
        _fail(f"Repair failed: {final.get('error', 'unknown')}")
        return False
    else:
        _warn(f"Repair ended with status: {final.get('status')}")
        return final.get("status") == "idle"


# ---------------------------------------------------------------------------
# Step 7: Train (combined mode)
# ---------------------------------------------------------------------------
def step_train(base_url: str, args: argparse.Namespace) -> tuple[bool, dict | None]:
    """POST /train with step=train for combined-mode training.

    Returns (success, final_status_dict).
    """
    _step_header(7, "Train (Combined Mode)")
    _info(f"Symbols:    {', '.join(ALL_SYMBOLS)}")
    _info("Mode:       combined")
    _info(f"Epochs:     {args.epochs}")
    _info(f"Batch size: {args.batch_size}")
    _info(f"Patience:   {args.patience}")
    print()

    if args.dry_run:
        body = {
            "symbols": ALL_SYMBOLS,
            "train_mode": "combined",
            "days_back": args.days_back,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "patience": args.patience,
            "step": "train",
        }
        _warn("DRY RUN — would POST to /train with:")
        print(f"{DIM}{json.dumps(body, indent=2)}{RESET}")
        return True, None

    body = {
        "symbols": ALL_SYMBOLS,
        "train_mode": "combined",
        "days_back": args.days_back,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "patience": args.patience,
        "step": "train",
    }

    _info(f"POST {base_url}/train (step=train)")
    status_code, resp = _http_post(f"{base_url}/train", body)

    if status_code == 202:
        _ok("Training started (202 Accepted)")
        if resp:
            _dim(f"Params: {json.dumps(resp.get('params', {}))}")
    elif status_code == 409:
        _warn("Server busy (409 Conflict) — waiting for current operation...")
    else:
        _fail(f"POST /train failed with status {status_code}")
        if resp:
            _fail(f"  Response: {json.dumps(resp)}")
        return False, None

    # Poll
    print()
    _info(f"Polling status every {POLL_INTERVAL}s...")
    final = _poll_until_done(base_url, "train")

    if final is None:
        _fail("Training timed out")
        return False, None

    if final.get("status") == "done":
        _ok("Training completed successfully")
        # Display metrics
        last_result = final.get("last_result", {})
        metrics = last_result.get("metrics", {})
        if metrics:
            print()
            _info(f"Accuracy:   {metrics.get('val_accuracy', 0):.1f}%")
            _info(f"Precision:  {metrics.get('val_precision', 0):.1f}%")
            _info(f"Recall:     {metrics.get('val_recall', 0):.1f}%")
            _info(f"Epochs:     {metrics.get('epochs_trained', '?')} (best: {metrics.get('best_epoch', '?')})")

        promoted = last_result.get("promoted", False)
        if promoted:
            _ok("Model promoted to champion")
        else:
            reason = last_result.get("reason", "unknown")
            _warn(f"Model NOT promoted: {reason}")

        return True, final
    elif final.get("status") == "failed":
        _fail(f"Training failed: {final.get('error', 'unknown')}")
        return False, final
    else:
        _warn(f"Training ended with status: {final.get('status')}")
        return False, final


# ---------------------------------------------------------------------------
# Step 8: Compare with champion
# ---------------------------------------------------------------------------
def step_compare_champion(base_url: str, train_result: dict | None) -> bool:
    """Compare the new model with the existing champion."""
    _step_header(8, "Compare with Champion Model")

    # Get current status which includes champion info
    status = _http_get(f"{base_url}/status")
    if not status:
        _warn("Cannot reach trainer server for comparison")
        return True  # non-fatal

    champion = status.get("champion", {})
    if not champion.get("exists"):
        _info("No previous champion model to compare against")
        _ok("New model is the first champion")
        return True

    champion_metrics = champion.get("metrics", {})
    champion_acc = champion_metrics.get("val_accuracy", 0)
    champion_prec = champion_metrics.get("val_precision", 0)
    champion_rec = champion_metrics.get("val_recall", 0)
    champion_version = champion.get("version", "unknown")
    champion_trained = champion.get("trained_at", "unknown")

    _info(f"Champion model:   v{champion_version} (trained: {champion_trained})")
    _info(f"  Accuracy:       {champion_acc:.1f}%")
    _info(f"  Precision:      {champion_prec:.1f}%")
    _info(f"  Recall:         {champion_rec:.1f}%")

    # New model metrics
    new_metrics: dict = {}
    if train_result:
        last_result = train_result.get("last_result", {})
        new_metrics = last_result.get("metrics", {})

    if new_metrics:
        new_acc = new_metrics.get("val_accuracy", 0)
        new_prec = new_metrics.get("val_precision", 0)
        new_rec = new_metrics.get("val_recall", 0)

        print()
        _info("New model:")
        _info(f"  Accuracy:       {new_acc:.1f}%")
        _info(f"  Precision:      {new_prec:.1f}%")
        _info(f"  Recall:         {new_rec:.1f}%")

        # Comparison table
        print()
        header = f"  {'Metric':<15} {'Champion':>12} {'New':>12} {'Delta':>12}"
        sep = f"  {'─' * 15} {'─' * 12} {'─' * 12} {'─' * 12}"
        print(f"{BOLD}{header}{RESET}")
        print(f"{DIM}{sep}{RESET}")

        for metric_name, old_val, new_val in [
            ("Accuracy", champion_acc, new_acc),
            ("Precision", champion_prec, new_prec),
            ("Recall", champion_rec, new_rec),
        ]:
            delta = new_val - old_val
            if delta > 0:
                delta_str = f"+{delta:.1f}%"
                color = GREEN
            elif delta < 0:
                delta_str = f"{delta:.1f}%"
                color = RED
            else:
                delta_str = "0.0%"
                color = DIM

            row = f"  {metric_name:<15} {old_val:>11.1f}% {new_val:>11.1f}% {color}{delta_str:>12}{RESET}"
            print(row)

        print()
        acc_diff = new_acc - champion_acc
        if acc_diff > 0:
            _ok(f"New model improves accuracy by {acc_diff:+.1f}%")
        elif acc_diff < 0:
            _warn(f"New model accuracy is lower by {acc_diff:.1f}%")
        else:
            _info("Accuracy is unchanged")
    else:
        _warn("No new model metrics available for comparison")

    return True


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------
def _check_health(base_url: str) -> bool:
    """Verify the trainer server is reachable and healthy."""
    _info(f"Checking trainer server at {base_url}...")

    health = _http_get(f"{base_url}/health")
    if health and health.get("status") == "healthy":
        _ok("Trainer server is healthy")
        return True

    status = _http_get(f"{base_url}/status")
    if status:
        _ok("Trainer server is reachable (via /status)")
        gpu = status.get("gpu", {})
        if gpu.get("available"):
            _ok(f"GPU: {gpu.get('device_name')} ({gpu.get('memory_total_gb')} GB)")
        else:
            _warn("GPU not available — training will be slow")
        return True

    _fail(f"Cannot reach trainer server at {base_url}")
    _info("Is the trainer server running? Try: docker compose -f docker-compose.trainer.yml up")
    return False


# ---------------------------------------------------------------------------
# Main pipeline orchestration
# ---------------------------------------------------------------------------
def run_full_retrain(args: argparse.Namespace) -> int:
    """Run the full v9 retrain pipeline. Returns exit code."""
    base_url = args.trainer_url.rstrip("/")

    _banner("Full v9 Retrain Pipeline")

    # Show plan
    _section("Pipeline Plan")
    _info(f"Trainer URL:  {base_url}")
    _info(f"Days back:    {args.days_back}")
    _info(f"Epochs:       {args.epochs}")
    _info(f"Batch size:   {args.batch_size}")
    _info(f"Patience:     {args.patience}")
    _info(f"Start step:   {args.start_step}")
    _info(f"Symbols:      {', '.join(ALL_SYMBOLS)}")
    if args.dry_run:
        _warn("DRY RUN — no API calls will be made")
    if args.skip_repair:
        _info("Repair step:  SKIPPED (--skip-repair)")
    print()

    steps = [
        "1. Fix dataset paths",
        "2. Validate dataset",
        "3. Load data (fetch bars)",
        "4. Generate dataset",
        "5. Validate dataset (post-generation)",
        "6. Repair if needed",
        "7. Train (combined mode)",
        "8. Compare with champion",
    ]
    for i, step_name in enumerate(steps, 1):
        marker = "→" if i >= args.start_step else "○"
        color = CYAN if i >= args.start_step else DIM
        _log(f"  {marker} {step_name}", color)
    print()

    # Health check (skip in dry-run mode, needed for steps 3+)
    if not args.dry_run and args.start_step >= 3:
        if not _check_health(base_url):
            return 1
        print()

    overall_start = time.monotonic()
    train_final_status: dict | None = None
    needs_repair = False
    missing_count = 0

    # ── Step 1: Fix dataset paths ──────────────────────────────────────
    if args.start_step <= 1:
        success, fixed_count = step_fix_dataset_paths(dry_run=args.dry_run)
        if not success:
            _fail("Step 1 failed — aborting pipeline")
            return 1

    # ── Step 2: Validate dataset ───────────────────────────────────────
    if args.start_step <= 2:
        all_found, step2_missing = step_validate_dataset(2, "Validate Dataset (Pre-check)")
        if not all_found:
            missing_count = step2_missing
            _warn(f"{missing_count:,d} images missing — will attempt repair after generation")
            needs_repair = True

    # ── Step 3: Load data ──────────────────────────────────────────────
    if args.start_step <= 3:
        # Health check if not done already
        if not args.dry_run and args.start_step <= 2 and not _check_health(base_url):
            return 1

        success = step_load_data(base_url, args)
        if not success:
            _fail("Step 3 (load data) failed — aborting pipeline")
            return 1

    # ── Step 4: Generate dataset ───────────────────────────────────────
    if args.start_step <= 4:
        success = step_generate_dataset(base_url, args)
        if not success:
            _fail("Step 4 (generate dataset) failed — aborting pipeline")
            return 1

    # ── Step 5: Validate again ─────────────────────────────────────────
    if args.start_step <= 5:
        all_found, step5_missing = step_validate_dataset(5, "Validate Dataset (Post-generation)")
        if not all_found:
            missing_count = step5_missing
            _warn(f"{missing_count:,d} images still missing after generation")
            needs_repair = True
        else:
            needs_repair = False
            _ok("Dataset is complete — no repair needed")

    # ── Step 6: Repair if needed ───────────────────────────────────────
    if args.start_step <= 6:
        if needs_repair and not args.skip_repair:
            success = step_repair_dataset(base_url, args)
            if not success:
                _warn("Step 6 (repair) failed — continuing with training anyway")
                # Non-fatal: the model may still train with partial data
        elif needs_repair and args.skip_repair:
            _step_header(6, "Repair Dataset (SKIPPED)")
            _warn("Skipping repair (--skip-repair flag set)")
            _warn(f"Training will proceed with {missing_count:,d} missing images")
        else:
            _step_header(6, "Repair Dataset (Not Needed)")
            _ok("No missing images — repair step skipped")

    # ── Step 7: Train ──────────────────────────────────────────────────
    if args.start_step <= 7:
        success, train_final_status = step_train(base_url, args)
        if not success:
            _fail("Step 7 (training) failed — aborting pipeline")
            return 1

    # ── Step 8: Compare with champion ──────────────────────────────────
    if args.start_step <= 8 and not args.dry_run:
        step_compare_champion(base_url, train_final_status)
    elif args.dry_run:
        _step_header(8, "Compare with Champion Model")
        _warn("DRY RUN — comparison skipped")

    # ── Final summary ──────────────────────────────────────────────────
    overall_elapsed = time.monotonic() - overall_start

    _section("Pipeline Complete")
    _info(f"Total time: {_format_duration(overall_elapsed)}")

    if args.dry_run:
        _warn("DRY RUN — no actual changes were made")
    else:
        _ok("Full v9 retrain pipeline completed successfully ✓")

    print()
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Orchestrate the full v9 CNN retrain pipeline via the trainer server API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Pipeline Steps:
              1. Fix dataset paths      Strip Docker /app/ prefix from CSV image paths
              2. Validate dataset        Check CSV integrity and image coverage
              3. Load data               Fetch 365 days of bars (POST /train step=load_data)
              4. Generate dataset        Render chart images + labels.csv (step=generate_dataset)
              5. Validate again          Re-check for missing images after generation
              6. Repair if needed        Re-generate missing images (POST /train/repair)
              7. Train                   Combined-mode training (step=train)
              8. Compare with champion   Compare new model metrics vs existing champion

            Examples:
              python scripts/run_full_retrain.py                        # full pipeline
              python scripts/run_full_retrain.py --start-step 7         # just train + compare
              python scripts/run_full_retrain.py --dry-run              # show what would happen
              python scripts/run_full_retrain.py --skip-repair          # skip repair even if needed
              python scripts/run_full_retrain.py --epochs 40 --patience 8
        """),
    )
    parser.add_argument(
        "--trainer-url",
        default=DEFAULT_TRAINER_URL,
        help=f"Trainer server base URL (default: {DEFAULT_TRAINER_URL})",
    )
    parser.add_argument(
        "--start-step",
        type=int,
        default=1,
        choices=range(1, TOTAL_STEPS + 1),
        metavar="N",
        help=f"Start from step N (1-{TOTAL_STEPS}, default: 1)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would happen without making API calls or modifying files",
    )
    parser.add_argument(
        "--skip-repair",
        action="store_true",
        help="Skip the repair step even if missing images are found",
    )
    parser.add_argument(
        "--days-back",
        type=int,
        default=DEFAULT_DAYS_BACK,
        help=f"Days of history for dataset generation (default: {DEFAULT_DAYS_BACK})",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help=f"Training epochs (default: {DEFAULT_EPOCHS})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Batch size (default: {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=DEFAULT_PATIENCE,
        help=f"Early stopping patience (default: {DEFAULT_PATIENCE})",
    )

    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    sys.exit(run_full_retrain(args))


if __name__ == "__main__":
    main()
