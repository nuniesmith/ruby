#!/usr/bin/env python3
"""
run_per_group_training.py — 3-Tier CNN Training Orchestrator
==============================================================
Standalone script that orchestrates hierarchical CNN training runs by
making HTTP calls to the trainer server's ``POST /train`` endpoint.

3-Tier Training Architecture:
  Tier 1 — Per-Asset: Train individual models for each symbol.
    - MGC alone, SIL alone
    - MES alone, MNQ alone, M2K alone, MYM alone
    - ZN alone

  Tier 2 — Per-Group: Train group models on grouped assets.
    - metals:        MGC + SIL  (trained together)
    - equity_micros: MES + MNQ + M2K + MYM  (trained together)
    - treasuries:    ZN  (same as per-asset, but named as group)

  Tier 3 — Master: Train a master ensemble model on ALL symbols,
    using the per-group models as a comparison baseline.
    - master: (MGC, SIL) + (MES, MNQ, M2K, MYM) + (ZN)

Inference Resolution (breakout_cnn._resolve_model_name):
  1. Per-asset model:  breakout_cnn_best_{SYMBOL}.pt
  2. Per-group model:  breakout_cnn_best_{group}.pt
  3. Combined/master:  breakout_cnn_best.pt

Supports:
  - ``--tier 1|2|3|all`` to run specific tiers
  - ``--group metals`` to train only one group
  - ``--dry-run`` to show what would be sent without calling the API
  - ``--trainer-url`` to point at a non-default trainer server
  - ``--step`` to run a specific pipeline step instead of full
  - ``--skip-combined`` to skip the tier-3 master training

Usage:
    # Full 3-tier training (per-asset → per-group → master):
    python scripts/run_per_group_training.py --tier all

    # Tier 1 only — train each asset individually:
    python scripts/run_per_group_training.py --tier 1

    # Tier 2 only — train group models:
    python scripts/run_per_group_training.py --tier 2

    # Tier 3 only — train master model:
    python scripts/run_per_group_training.py --tier 3

    # Legacy mode — per-group + combined (same as before):
    python scripts/run_per_group_training.py

    # Train only metals group (tier 1 assets + tier 2 group):
    python scripts/run_per_group_training.py --group metals --tier all

    # Dry run — show what would be sent:
    python scripts/run_per_group_training.py --dry-run --tier all

    # Point at a different trainer server:
    python scripts/run_per_group_training.py --trainer-url http://oryx:8200

    # Run only the train step (assumes dataset already exists):
    python scripts/run_per_group_training.py --step train --tier all

    # Custom training hyperparameters:
    python scripts/run_per_group_training.py --epochs 80 --batch-size 32 --patience 12

Exit codes:
    0 — all training runs completed successfully
    1 — one or more training runs failed
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
import time
from datetime import datetime

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
# Asset groups — mirrors ASSET_GROUPS in trainer_server.py
# ---------------------------------------------------------------------------
ASSET_GROUPS: dict[str, list[str]] = {
    "metals": ["MGC", "SIL"],
    "equity_micros": ["MES", "MNQ", "M2K", "MYM"],
    "treasuries": ["ZN"],
}

ALL_SYMBOLS: list[str] = sorted({s for symbols in ASSET_GROUPS.values() for s in symbols})

# Tier descriptions for display
TIER_NAMES: dict[str, str] = {
    "1": "Per-Asset (individual symbol models)",
    "2": "Per-Group (group models: metals, equity_micros, treasuries)",
    "3": "Master (combined model trained on all symbols)",
}

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_TRAINER_URL = "http://trainer:8200"
DEFAULT_DAYS_BACK = 365
DEFAULT_EPOCHS = 80
DEFAULT_BATCH_SIZE = 32
DEFAULT_PATIENCE = 12
POLL_INTERVAL = 30  # seconds between status polls
MAX_WAIT_TRAIN = 7200  # 2 hours max per training run

# Terminal states from TrainStatus enum
TERMINAL_STATES = {"done", "failed", "cancelled", "idle"}


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
# Build the training request body
# ---------------------------------------------------------------------------
def _build_train_request(
    symbols: list[str],
    train_mode: str,
    args: argparse.Namespace,
) -> dict:
    """Build the JSON body for POST /train."""
    body: dict = {
        "symbols": symbols,
        "train_mode": train_mode,
        "days_back": args.days_back,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "patience": args.patience,
    }
    if args.step:
        body["step"] = args.step
    return body


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
# Extract and display results from a completed training run
# ---------------------------------------------------------------------------
def _extract_results(status: dict, label: str) -> dict:
    """Extract metrics from a completed training status response.

    Returns a results dict with keys: status, accuracy, precision, recall,
    epochs_trained, best_epoch, promoted, error.
    """
    results: dict = {
        "label": label,
        "status": status.get("status", "unknown"),
        "accuracy": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "epochs_trained": 0,
        "best_epoch": 0,
        "promoted": False,
        "error": status.get("error"),
    }

    last_result = status.get("last_result")
    if not last_result:
        return results

    metrics = last_result.get("metrics", {})
    results["accuracy"] = metrics.get("val_accuracy", 0.0)
    results["precision"] = metrics.get("val_precision", 0.0)
    results["recall"] = metrics.get("val_recall", 0.0)
    results["epochs_trained"] = metrics.get("epochs_trained", 0)
    results["best_epoch"] = metrics.get("best_epoch", 0)
    results["promoted"] = last_result.get("promoted", False)

    return results


def _display_results(results: dict) -> None:
    """Display results for a single training run."""
    label = results["label"]

    if results["status"] == "done":
        _ok(f"[{label}] Training completed")
        _info(f"[{label}] Accuracy:   {results['accuracy']:.1f}%")
        _info(f"[{label}] Precision:  {results['precision']:.1f}%")
        _info(f"[{label}] Recall:     {results['recall']:.1f}%")
        _info(f"[{label}] Epochs:     {results['epochs_trained']} (best: {results['best_epoch']})")
        if results["promoted"]:
            _ok(f"[{label}] Model promoted to champion")
        else:
            _warn(f"[{label}] Model NOT promoted")
    elif results["status"] == "failed":
        _fail(f"[{label}] Training FAILED")
        if results["error"]:
            _fail(f"[{label}] Error: {results['error']}")
    elif results["status"] == "cancelled":
        _warn(f"[{label}] Training was cancelled")
    else:
        _warn(f"[{label}] Ended with status: {results['status']}")


# ---------------------------------------------------------------------------
# Comparison summary table
# ---------------------------------------------------------------------------
def _print_comparison_table(all_results: list[dict]) -> None:
    """Print a formatted comparison table of all training results."""
    _section("Comparison Summary")
    print()

    # Table header
    header = f"  {'Group':<20} {'Status':<12} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'Epochs':>8} {'Promoted':>10}"
    separator = f"  {'─' * 20} {'─' * 12} {'─' * 10} {'─' * 10} {'─' * 10} {'─' * 8} {'─' * 10}"

    print(f"{BOLD}{header}{RESET}")
    print(f"{DIM}{separator}{RESET}")

    for r in all_results:
        label = r["label"]
        status = r["status"]

        if status == "done":
            acc = f"{r['accuracy']:.1f}%"
            prec = f"{r['precision']:.1f}%"
            rec = f"{r['recall']:.1f}%"
            epochs = str(r["epochs_trained"])
            promoted = "✓ yes" if r["promoted"] else "✗ no"
            color = GREEN if r["promoted"] else YELLOW
        elif status == "failed":
            acc = prec = rec = epochs = "—"
            promoted = "—"
            color = RED
        elif status == "dry_run":
            acc = prec = rec = epochs = "(dry run)"
            promoted = "—"
            color = DIM
        else:
            acc = prec = rec = epochs = "—"
            promoted = "—"
            color = YELLOW

        row = f"  {label:<20} {status:<12} {acc:>10} {prec:>10} {rec:>10} {epochs:>8} {promoted:>10}"
        print(f"{color}{row}{RESET}")

    print()

    # Find best per-group accuracy
    group_results = [r for r in all_results if r["label"] != "combined" and r["status"] == "done"]
    combined_results = [r for r in all_results if r["label"] == "combined" and r["status"] == "done"]

    if group_results and combined_results:
        avg_group_acc = sum(r["accuracy"] for r in group_results) / len(group_results)
        combined_acc = combined_results[0]["accuracy"]
        diff = avg_group_acc - combined_acc

        _info(f"Average per-group accuracy: {avg_group_acc:.1f}%")
        _info(f"Combined model accuracy:    {combined_acc:.1f}%")

        if diff > 0:
            _ok(f"Per-group models outperform combined by {diff:+.1f}%")
        elif diff < 0:
            _warn(f"Combined model outperforms per-group average by {abs(diff):.1f}%")
        else:
            _info("Per-group and combined models have identical average accuracy")
    elif group_results:
        avg_group_acc = sum(r["accuracy"] for r in group_results) / len(group_results)
        _info(f"Average per-group accuracy: {avg_group_acc:.1f}%")
        _info("(No combined run to compare against)")


# ---------------------------------------------------------------------------
# Run a single training job
# ---------------------------------------------------------------------------
def _run_training_job(
    base_url: str,
    label: str,
    symbols: list[str],
    train_mode: str,
    args: argparse.Namespace,
) -> dict:
    """Fire a training request and poll until complete. Returns results dict."""
    _section(f"Training: {label}")
    _info(f"Symbols:    {', '.join(symbols)}")
    _info(f"Train mode: {train_mode}")
    _info(f"Step:       {args.step or 'full'}")
    _info(f"Days back:  {args.days_back}")
    _info(f"Epochs:     {args.epochs}")
    _info(f"Batch size: {args.batch_size}")
    _info(f"Patience:   {args.patience}")
    print()

    body = _build_train_request(symbols, train_mode, args)

    # Dry-run mode — just show the request body
    if args.dry_run:
        _warn("DRY RUN — would POST to /train with:")
        print(f"{DIM}{json.dumps(body, indent=2)}{RESET}")
        return {
            "label": label,
            "status": "dry_run",
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "epochs_trained": 0,
            "best_epoch": 0,
            "promoted": False,
            "error": None,
        }

    # Fire the training request
    _info(f"POST {base_url}/train")
    _dim(f"Body: {json.dumps(body)}")

    status_code, resp = _http_post(f"{base_url}/train", body)

    if status_code == 202:
        _ok("Training started (202 Accepted)")
        if resp:
            _dim(f"Params: {json.dumps(resp.get('params', {}))}")
    elif status_code == 409:
        _warn("Training already in progress (409 Conflict)")
        _info("Waiting for current run to finish before polling...")
    else:
        _fail(f"POST /train failed with status {status_code}")
        if resp:
            _fail(f"  Response: {json.dumps(resp)}")
        return {
            "label": label,
            "status": "failed",
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "epochs_trained": 0,
            "best_epoch": 0,
            "promoted": False,
            "error": f"POST /train returned {status_code}",
        }

    # Poll until complete
    print()
    _info(f"Polling training status every {POLL_INTERVAL}s...")

    final_status = _poll_until_done(base_url, label)

    if final_status is None:
        return {
            "label": label,
            "status": "failed",
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "epochs_trained": 0,
            "best_epoch": 0,
            "promoted": False,
            "error": "Polling timed out",
        }

    # Extract and display results
    results = _extract_results(final_status, label)
    print()
    _display_results(results)
    return results


# ---------------------------------------------------------------------------
# Check server health
# ---------------------------------------------------------------------------
def _check_health(base_url: str) -> bool:
    """Verify the trainer server is reachable and healthy."""
    _info(f"Checking trainer server at {base_url}...")

    health = _http_get(f"{base_url}/health")
    if health and health.get("status") == "healthy":
        _ok("Trainer server is healthy")
        return True

    # Fall back to /status
    status = _http_get(f"{base_url}/status")
    if status:
        _ok("Trainer server is reachable (via /status)")
        gpu = status.get("gpu", {})
        if gpu.get("available"):
            _ok(f"GPU: {gpu.get('device_name')} ({gpu.get('memory_total_gb')} GB)")
        else:
            _warn("GPU not available — training will be slow")

        champion = status.get("champion", {})
        if champion.get("exists"):
            _info(f"Champion model exists (trained: {champion.get('trained_at', 'unknown')})")
        else:
            _dim("No champion model yet")

        current_status = status.get("status", "unknown")
        if current_status not in ("idle", "done", "failed", "cancelled"):
            _warn(f"Server is currently busy: {current_status}")
            _warn("Training requests may be rejected (409 Conflict)")
        return True

    _fail(f"Cannot reach trainer server at {base_url}")
    _info("Is the trainer server running? Try: docker compose -f docker-compose.trainer.yml up")
    return False


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------
def _resolve_tiers(args: argparse.Namespace) -> list[str]:
    """Determine which tiers to run based on --tier argument."""
    tier_arg = getattr(args, "tier", None) or "legacy"
    if tier_arg == "all":
        return ["1", "2", "3"]
    if tier_arg == "legacy":
        # Original behaviour: per-group + combined
        return ["2", "3"]
    # Single tier
    return [tier_arg]


def _run_tier1_per_asset(
    base_url: str,
    groups_to_train: dict[str, list[str]],
    args: argparse.Namespace,
    all_results: list[dict],
) -> int:
    """Tier 1: Train each symbol individually. Returns failure count."""
    _banner("Tier 1 — Per-Asset Training")
    failures = 0

    for group_name, symbols in groups_to_train.items():
        _section(f"Tier 1 — {group_name} assets ({', '.join(symbols)})")

        for symbol in symbols:
            label = f"asset:{symbol}"
            results = _run_training_job(
                base_url=base_url,
                label=label,
                symbols=[symbol],
                train_mode="per_asset",
                args=args,
            )
            results["tier"] = "1"
            results["group"] = group_name
            all_results.append(results)
            if results["status"] == "failed":
                failures += 1
                if not args.continue_on_failure:
                    _fail(f"Stopping tier 1 — {symbol} failed (use --continue-on-failure to keep going)")
                    return failures

    return failures


def _run_tier2_per_group(
    base_url: str,
    groups_to_train: dict[str, list[str]],
    args: argparse.Namespace,
    all_results: list[dict],
) -> int:
    """Tier 2: Train group models. Returns failure count."""
    _banner("Tier 2 — Per-Group Training")
    failures = 0

    for group_name, symbols in groups_to_train.items():
        label = f"group:{group_name}"
        results = _run_training_job(
            base_url=base_url,
            label=label,
            symbols=symbols,
            train_mode="per_group",
            args=args,
        )
        results["tier"] = "2"
        results["group"] = group_name
        all_results.append(results)
        if results["status"] == "failed":
            failures += 1
            if not args.continue_on_failure:
                _fail(f"Stopping tier 2 — {group_name} failed (use --continue-on-failure to keep going)")
                return failures

    return failures


def _run_tier3_master(
    base_url: str,
    args: argparse.Namespace,
    all_results: list[dict],
) -> int:
    """Tier 3: Train master model on all symbols. Returns failure count."""
    _banner("Tier 3 — Master Model Training")

    label = "master:combined"
    results = _run_training_job(
        base_url=base_url,
        label=label,
        symbols=ALL_SYMBOLS,
        train_mode="combined",
        args=args,
    )
    results["tier"] = "3"
    results["group"] = "all"
    all_results.append(results)
    return 1 if results["status"] == "failed" else 0


def _print_tiered_comparison(all_results: list[dict]) -> None:
    """Print a tiered comparison table grouping results by tier."""
    _section("Tiered Comparison Summary")
    print()

    # Table header
    header = f"  {'Tier':<6} {'Model':<24} {'Status':<12} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'Epochs':>8} {'Promoted':>10}"
    separator = f"  {'─' * 6} {'─' * 24} {'─' * 12} {'─' * 10} {'─' * 10} {'─' * 10} {'─' * 8} {'─' * 10}"

    print(f"{BOLD}{header}{RESET}")
    print(f"{DIM}{separator}{RESET}")

    current_tier = ""
    for r in all_results:
        tier = r.get("tier", "?")
        label = r["label"]
        status = r["status"]

        # Tier separator
        if tier != current_tier:
            if current_tier:
                print(f"{DIM}{separator}{RESET}")
            current_tier = tier

        if status == "done":
            acc = f"{r['accuracy']:.1f}%"
            prec = f"{r['precision']:.1f}%"
            rec = f"{r['recall']:.1f}%"
            epochs = str(r["epochs_trained"])
            promoted = "✓ yes" if r["promoted"] else "✗ no"
            color = GREEN if r["promoted"] else YELLOW
        elif status == "failed":
            acc = prec = rec = epochs = "—"
            promoted = "—"
            color = RED
        elif status == "dry_run":
            acc = prec = rec = epochs = "(dry run)"
            promoted = "—"
            color = DIM
        else:
            acc = prec = rec = epochs = "—"
            promoted = "—"
            color = YELLOW

        row = f"  {tier:<6} {label:<24} {status:<12} {acc:>10} {prec:>10} {rec:>10} {epochs:>8} {promoted:>10}"
        print(f"{color}{row}{RESET}")

    print()

    # Per-tier averages
    for tier_num in ("1", "2", "3"):
        tier_results = [r for r in all_results if r.get("tier") == tier_num and r["status"] == "done"]
        if tier_results:
            avg_acc = sum(r["accuracy"] for r in tier_results) / len(tier_results)
            tier_name = TIER_NAMES.get(tier_num, f"Tier {tier_num}")
            _info(f"Tier {tier_num} avg accuracy ({tier_name}): {avg_acc:.1f}%  ({len(tier_results)} models)")

    # Compare tier 2 vs tier 3
    tier2 = [r for r in all_results if r.get("tier") == "2" and r["status"] == "done"]
    tier3 = [r for r in all_results if r.get("tier") == "3" and r["status"] == "done"]
    if tier2 and tier3:
        avg_group = sum(r["accuracy"] for r in tier2) / len(tier2)
        master_acc = tier3[0]["accuracy"]
        diff = avg_group - master_acc
        print()
        if diff > 0:
            _ok(f"Per-group models outperform master by {diff:+.1f}%  →  recommend per-group inference")
        elif diff < 0:
            _warn(f"Master model outperforms per-group avg by {abs(diff):.1f}%  →  recommend master inference")
        else:
            _info("Per-group and master models have identical average accuracy")

    # Compare tier 1 vs tier 2 within groups
    for group_name in ASSET_GROUPS:
        t1 = [r for r in all_results if r.get("tier") == "1" and r.get("group") == group_name and r["status"] == "done"]
        t2 = [r for r in all_results if r.get("tier") == "2" and r.get("group") == group_name and r["status"] == "done"]
        if t1 and t2:
            avg_asset = sum(r["accuracy"] for r in t1) / len(t1)
            group_acc = t2[0]["accuracy"]
            diff = avg_asset - group_acc
            if diff > 0:
                _ok(f"  {group_name}: per-asset avg beats group by {diff:+.1f}%")
            elif diff < 0:
                _warn(f"  {group_name}: group model beats per-asset avg by {abs(diff):.1f}%")
            else:
                _info(f"  {group_name}: tied")


def run_per_group_training(args: argparse.Namespace) -> int:
    """Run tiered training orchestration. Returns exit code."""
    base_url = args.trainer_url.rstrip("/")
    tiers = _resolve_tiers(args)

    _banner("3-Tier CNN Training Orchestrator")

    # Determine which groups to train
    if args.group:
        group_name = args.group.lower().strip()
        if group_name not in ASSET_GROUPS:
            _fail(f"Unknown group: {group_name}")
            _info(f"Available groups: {', '.join(ASSET_GROUPS.keys())}")
            return 1
        groups_to_train = {group_name: ASSET_GROUPS[group_name]}
    else:
        groups_to_train = dict(ASSET_GROUPS)

    # Show plan
    _section("Training Plan")
    _info(f"Trainer URL:  {base_url}")
    _info(f"Step:         {args.step or 'full'}")
    _info(f"Days back:    {args.days_back}")
    _info(f"Epochs:       {args.epochs}")
    _info(f"Batch size:   {args.batch_size}")
    _info(f"Patience:     {args.patience}")
    _info(f"Tiers:        {', '.join(tiers)}")
    for tier in tiers:
        _info(f"  Tier {tier}: {TIER_NAMES.get(tier, '?')}")

    if "1" in tiers or "2" in tiers:
        _info(f"Groups:       {len(groups_to_train)}")
        for name, symbols in groups_to_train.items():
            _info(f"  {name:<20} → {', '.join(symbols)}")

    if "1" in tiers:
        total_assets = sum(len(s) for s in groups_to_train.values())
        _info(f"Tier 1 models: {total_assets} (one per asset)")

    if "3" in tiers and not args.skip_combined:
        _info(f"  {'master':<20} → {', '.join(ALL_SYMBOLS)}")
    elif "3" in tiers and args.skip_combined:
        _warn("Tier 3 (master) skipped via --skip-combined")

    if args.dry_run:
        _warn("DRY RUN — no API calls will be made")
    print()

    # Health check (skip in dry-run mode)
    if not args.dry_run:
        if not _check_health(base_url):
            return 1
        print()

    # Track all results
    all_results: list[dict] = []
    failures = 0
    overall_start = time.monotonic()

    # ── Tier 1: Per-asset training ─────────────────────────────────────
    if "1" in tiers:
        tier1_failures = _run_tier1_per_asset(
            base_url,
            groups_to_train,
            args,
            all_results,
        )
        failures += tier1_failures
        if tier1_failures > 0 and not args.continue_on_failure:
            _fail("Stopping — tier 1 had failures")
        else:
            _ok(f"Tier 1 complete — {tier1_failures} failures")
        print()

    # ── Tier 2: Per-group training ─────────────────────────────────────
    if "2" in tiers and (failures == 0 or args.continue_on_failure):
        tier2_failures = _run_tier2_per_group(
            base_url,
            groups_to_train,
            args,
            all_results,
        )
        failures += tier2_failures
        if tier2_failures > 0 and not args.continue_on_failure:
            _fail("Stopping — tier 2 had failures")
        else:
            _ok(f"Tier 2 complete — {tier2_failures} failures")
        print()

    # ── Tier 3: Master model training ──────────────────────────────────
    if "3" in tiers and not args.skip_combined and (failures == 0 or args.continue_on_failure):
        tier3_failures = _run_tier3_master(base_url, args, all_results)
        failures += tier3_failures
        if tier3_failures == 0:
            _ok("Tier 3 complete — master model trained")
        else:
            _fail("Tier 3 failed — master model training failed")
        print()

    # ── Summary ────────────────────────────────────────────────────────
    overall_elapsed = time.monotonic() - overall_start

    # Use tiered table if any tier metadata exists, else fall back to legacy
    has_tiers = any(r.get("tier") for r in all_results)
    if has_tiers:
        _print_tiered_comparison(all_results)
    else:
        _print_comparison_table(all_results)

    _section("Final Summary")
    total_runs = len(all_results)
    successful = sum(1 for r in all_results if r["status"] in ("done", "dry_run"))
    failed = sum(1 for r in all_results if r["status"] == "failed")

    _info(f"Total runs:     {total_runs}")
    _info(f"Successful:     {successful}")
    if failed:
        _fail(f"Failed:         {failed}")
    else:
        _info(f"Failed:         {failed}")
    _info(f"Total time:     {_format_duration(overall_elapsed)}")
    print()

    if failed == 0:
        _ok("All training runs completed successfully ✓")
        if has_tiers:
            print()
            _info("Model files produced:")
            for r in all_results:
                if r["status"] in ("done", "dry_run"):
                    tier = r.get("tier", "?")
                    label = r["label"]
                    if tier == "1":
                        symbol = label.replace("asset:", "")
                        _dim(f"  breakout_cnn_best_{symbol}.pt")
                    elif tier == "2":
                        group = label.replace("group:", "")
                        _dim(f"  breakout_cnn_best_{group}.pt")
                    elif tier == "3":
                        _dim("  breakout_cnn_best.pt  (master/champion)")
            print()
            _info("Inference resolution order:")
            _dim("  1. Per-asset:  breakout_cnn_best_{SYMBOL}.pt")
            _dim("  2. Per-group:  breakout_cnn_best_{group}.pt")
            _dim("  3. Master:     breakout_cnn_best.pt")
    else:
        _fail(f"{failed} training run(s) failed ✗")

    print()
    return 1 if failed > 0 else 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Orchestrate 3-tier CNN training runs via the trainer server API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            3-Tier Training Architecture:
              Tier 1 — Per-Asset:   Individual model per symbol
              Tier 2 — Per-Group:   Group model (metals, equity_micros, treasuries)
              Tier 3 — Master:      Combined model on all symbols

            Asset Groups:
              metals          MGC, SIL
              equity_micros   MES, MNQ, M2K, MYM
              treasuries      ZN

            Inference Resolution Order:
              1. breakout_cnn_best_{SYMBOL}.pt   (per-asset)
              2. breakout_cnn_best_{group}.pt    (per-group)
              3. breakout_cnn_best.pt            (master/combined)

            Examples:
              # Full 3-tier training (per-asset → per-group → master):
              python scripts/run_per_group_training.py --tier all

              # Tier 1 only — individual asset models:
              python scripts/run_per_group_training.py --tier 1

              # Tier 2 only — group models (legacy default):
              python scripts/run_per_group_training.py --tier 2

              # Tier 3 only — master model:
              python scripts/run_per_group_training.py --tier 3

              # Legacy mode (tier 2 + tier 3, same as before):
              python scripts/run_per_group_training.py

              # Metals group only, all tiers:
              python scripts/run_per_group_training.py --group metals --tier all

              # Dry run:
              python scripts/run_per_group_training.py --dry-run --tier all

              # Train step only (dataset must exist):
              python scripts/run_per_group_training.py --step train --tier all

              # Custom hyperparams:
              python scripts/run_per_group_training.py --epochs 40 --patience 8 --tier all
        """),
    )
    parser.add_argument(
        "--trainer-url",
        default=DEFAULT_TRAINER_URL,
        help=f"Trainer server base URL (default: {DEFAULT_TRAINER_URL})",
    )
    parser.add_argument(
        "--tier",
        default=None,
        choices=["1", "2", "3", "all"],
        help=(
            "Training tier: 1=per-asset, 2=per-group, 3=master, all=1+2+3. "
            "Default (no flag): legacy mode (tier 2 + tier 3)"
        ),
    )
    parser.add_argument(
        "--step",
        default=None,
        choices=["full", "load_data", "generate_dataset", "train"],
        help="Pipeline step to run (default: full)",
    )
    parser.add_argument(
        "--group",
        default=None,
        choices=list(ASSET_GROUPS.keys()),
        help="Train only this group (default: all groups)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be sent without actually calling the API",
    )
    parser.add_argument(
        "--skip-combined",
        action="store_true",
        help="Skip the tier-3 master/combined training run",
    )
    parser.add_argument(
        "--continue-on-failure",
        action="store_true",
        help="Continue training remaining models even if one fails",
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

    args = parser.parse_args()

    # Normalize --step "full" to None (server expects None for full pipeline)
    if args.step == "full":
        args.step = None

    return args


def main() -> None:
    args = _parse_args()
    sys.exit(run_per_group_training(args))


if __name__ == "__main__":
    main()
