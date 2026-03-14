#!/usr/bin/env python3
"""
Smoke Test — Trainer Server + All 13 Breakout Types
=====================================================
Launches the trainer server locally (no Docker), fires a small training
run covering all 13 breakout types × all 9 sessions, polls until complete,
and reports results.

This validates the full pipeline end-to-end on a CUDA-capable machine:
  1. Trainer server boots and responds to /health
  2. Dataset generation works for all 13 breakout types
  3. CNN training runs on GPU with the generated dataset
  4. Evaluation gates are checked
  5. Champion promotion (or rejection) completes without errors

Usage:
    # From the futures repo root:
    python scripts/smoke_test_trainer.py

    # With custom parameters:
    python scripts/smoke_test_trainer.py --symbols MGC MES --days 7 --epochs 3

    # Quick mode — minimal dataset, 1 symbol, 2 epochs:
    python scripts/smoke_test_trainer.py --quick

    # Just check that the server boots and GPU is visible:
    python scripts/smoke_test_trainer.py --health-only

Environment:
    MASSIVE_API_KEY   — Required for dataset generation (reads from rb/.env if not set)
    TRAINER_PORT      — Port for the smoke test server (default: 8201, avoids conflict with production)
    MODELS_DIR        — Where to write model checkpoints (default: ./models)

Prerequisites:
    - CUDA-capable GPU with nvidia-smi visible
    - PyTorch with CUDA support installed in the venv
    - MASSIVE_API_KEY set (or present in ~/github/rb/.env)
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import textwrap
import time
from datetime import datetime
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"

# Use a non-default port so we don't collide with a running production server
DEFAULT_PORT = 8201
POLL_INTERVAL = 5  # seconds between status polls
MAX_WAIT_HEALTH = 30  # seconds to wait for server to become healthy
MAX_WAIT_TRAIN = 1800  # 30 minutes max for the full pipeline

# Minimal training params for smoke testing
QUICK_SYMBOLS = ["MGC"]
QUICK_DAYS = 5
QUICK_EPOCHS = 2
QUICK_BATCH_SIZE = 16
QUICK_PATIENCE = 2

DEFAULT_SYMBOLS = ["MGC", "MES"]
DEFAULT_DAYS = 7
DEFAULT_EPOCHS = 3
DEFAULT_BATCH_SIZE = 32
DEFAULT_PATIENCE = 3

# Relaxed gates for smoke test — we're testing the pipeline, not model quality
SMOKE_MIN_ACC = 40.0
SMOKE_MIN_PRECISION = 30.0
SMOKE_MIN_RECALL = 30.0

# ---------------------------------------------------------------------------
# Colors
# ---------------------------------------------------------------------------

BOLD = "\033[1m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
CYAN = "\033[96m"
DIM = "\033[2m"
RESET = "\033[0m"


def _log(msg: str, color: str = "") -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    prefix = f"{DIM}[{ts}]{RESET}"
    print(f"{prefix} {color}{msg}{RESET}")


def _ok(msg: str) -> None:
    _log(f"✓ {msg}", GREEN)


def _warn(msg: str) -> None:
    _log(f"⚠ {msg}", YELLOW)


def _fail(msg: str) -> None:
    _log(f"✗ {msg}", RED)


def _info(msg: str) -> None:
    _log(f"  {msg}", CYAN)


def _dim(msg: str) -> None:
    _log(f"  {msg}", DIM)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_massive_api_key() -> str | None:
    """Try to find MASSIVE_API_KEY from env or rb/.env file."""
    key = os.environ.get("MASSIVE_API_KEY", "").strip()
    if key:
        return key

    # Try to read from ~/github/rb/.env
    rb_env = Path.home() / "github" / "rb" / ".env"
    if rb_env.exists():
        for line in rb_env.read_text().splitlines():
            line = line.strip()
            if line.startswith("MASSIVE_API_KEY=") and not line.startswith("#"):
                val = line.split("=", 1)[1].strip()
                if val:
                    return val
    return None


def _check_cuda() -> dict:
    """Check CUDA availability via PyTorch."""
    try:
        import torch

        if torch.cuda.is_available():
            return {
                "available": True,
                "device_count": torch.cuda.device_count(),
                "device_name": torch.cuda.get_device_name(0),
                "memory_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2),
                "torch_version": torch.__version__,
                "cuda_version": torch.version.cuda or "N/A",
            }
        return {"available": False, "torch_version": torch.__version__}
    except ImportError:
        return {"available": False, "error": "PyTorch not installed"}


def _http_get(url: str, timeout: float = 30) -> dict | None:
    """Simple HTTP GET returning parsed JSON or None."""
    try:
        req = Request(url)
        with urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except (URLError, OSError, json.JSONDecodeError):
        return None


def _http_post(url: str, body: dict | None = None, timeout: float = 30) -> tuple[int, dict | None]:
    """Simple HTTP POST returning (status_code, parsed_json)."""
    try:
        data = json.dumps(body or {}).encode() if body else b"{}"
        req = Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
        with urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read().decode())
    except URLError as exc:
        if hasattr(exc, "code"):
            try:
                body_text = exc.read().decode() if hasattr(exc, "read") else "{}"
                return exc.code, json.loads(body_text)
            except Exception:
                return exc.code, None
        return 0, None
    except (OSError, json.JSONDecodeError):
        return 0, None


def _wait_for_health(base_url: str, max_wait: float = MAX_WAIT_HEALTH) -> bool:
    """Poll /health until the server responds 200."""
    deadline = time.monotonic() + max_wait
    while time.monotonic() < deadline:
        resp = _http_get(f"{base_url}/health")
        if resp and resp.get("status") == "healthy":
            return True
        time.sleep(0.5)
    return False


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
# Main smoke test
# ---------------------------------------------------------------------------


def run_smoke_test(args: argparse.Namespace) -> int:
    """Run the full smoke test. Returns 0 on success, 1 on failure."""

    port = int(os.environ.get("TRAINER_PORT", str(DEFAULT_PORT)))
    base_url = f"http://127.0.0.1:{port}"
    models_dir = os.environ.get("MODELS_DIR", str(PROJECT_ROOT / "models"))
    dataset_dir = str(PROJECT_ROOT / "smoke_test_dataset")

    print()
    print(f"{BOLD}{'=' * 70}{RESET}")
    print(f"{BOLD}  Trainer Smoke Test — All 13 Breakout Types{RESET}")
    print(f"{BOLD}{'=' * 70}{RESET}")
    print()

    # ── Step 0: Pre-flight checks ──────────────────────────────────────────

    _info("Pre-flight checks...")

    # CUDA
    cuda_info = _check_cuda()
    if cuda_info.get("available"):
        _ok(f"CUDA available: {cuda_info['device_name']} ({cuda_info['memory_gb']} GB)")
        _dim(f"PyTorch {cuda_info['torch_version']}, CUDA {cuda_info['cuda_version']}")
    else:
        _warn("CUDA not available — training will use CPU (slow)")
        if "error" in cuda_info:
            _fail(cuda_info["error"])

    # Massive API key
    api_key = _load_massive_api_key()
    if api_key:
        _ok(f"MASSIVE_API_KEY found ({api_key[:8]}...)")
    else:
        _fail("MASSIVE_API_KEY not found — dataset generation will fail")
        _info("Set MASSIVE_API_KEY in env or ensure ~/github/rb/.env exists")
        return 1

    # src directory
    if not SRC_DIR.exists():
        _fail(f"Source directory not found: {SRC_DIR}")
        return 1
    _ok(f"Source directory: {SRC_DIR}")

    # Resolve symbols and training params
    if args.quick:
        symbols = QUICK_SYMBOLS
        days = QUICK_DAYS
        epochs = QUICK_EPOCHS
        batch_size = QUICK_BATCH_SIZE
        patience = QUICK_PATIENCE
    else:
        symbols = args.symbols
        days = args.days
        epochs = args.epochs
        batch_size = args.batch_size
        patience = args.patience

    print()
    _info(f"Symbols:        {', '.join(symbols)}")
    _info(f"Days back:      {days}")
    _info("Breakout type:  all (13 types)")
    _info("Session:        all (9 sessions)")
    _info(f"Epochs:         {epochs}")
    _info(f"Batch size:     {batch_size}")
    _info(f"Patience:       {patience}")
    _info(f"Trainer port:   {port}")
    _info(f"Models dir:     {models_dir}")
    _info(f"Dataset dir:    {dataset_dir}")
    print()

    if args.health_only:
        _info("--health-only mode: will start server, check health, and exit")
        print()

    # ── Step 1: Start the trainer server ───────────────────────────────────

    _info("Starting trainer server...")

    env = os.environ.copy()
    env.update(
        {
            "PYTHONPATH": str(SRC_DIR),
            "PYTHONUNBUFFERED": "1",
            "MASSIVE_API_KEY": api_key,
            "TRAINER_HOST": "127.0.0.1",
            "TRAINER_PORT": str(port),
            "MODELS_DIR": models_dir,
            "DATASET_DIR": dataset_dir,
            # Relaxed gates for smoke test
            "CNN_RETRAIN_MIN_ACC": str(SMOKE_MIN_ACC),
            "CNN_RETRAIN_MIN_PRECISION": str(SMOKE_MIN_PRECISION),
            "CNN_RETRAIN_MIN_RECALL": str(SMOKE_MIN_RECALL),
        }
    )

    server_proc = subprocess.Popen(
        [sys.executable, "-m", "lib.services.training.trainer_server"],
        cwd=str(SRC_DIR),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    # Ensure cleanup on exit
    def _cleanup(signum=None, frame=None):
        if server_proc.poll() is None:
            _dim("Shutting down trainer server...")
            server_proc.terminate()
            try:
                server_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                server_proc.kill()

    signal.signal(signal.SIGINT, _cleanup)
    signal.signal(signal.SIGTERM, _cleanup)

    try:
        # Wait for health
        if _wait_for_health(base_url, max_wait=MAX_WAIT_HEALTH):
            _ok(f"Trainer server healthy at {base_url}")
        else:
            _fail(f"Trainer server failed to start within {MAX_WAIT_HEALTH}s")
            # Dump any output
            server_proc.terminate()
            stdout, _ = server_proc.communicate(timeout=5)
            if stdout:
                print(stdout[:2000])
            return 1

        # ── Step 2: Check /status ──────────────────────────────────────────

        status = _http_get(f"{base_url}/status")
        if status:
            _ok("GET /status responded")
            gpu = status.get("gpu", {})
            if gpu.get("available"):
                _ok(f"GPU visible to server: {gpu.get('device_name')} ({gpu.get('memory_total_gb')} GB)")
            else:
                _warn("GPU not visible to server — will train on CPU")

            champion = status.get("champion", {})
            if champion.get("exists"):
                _ok(f"Champion model exists (trained: {champion.get('trained_at', 'unknown')})")
            else:
                _dim("No champion model yet (expected for first run)")
        else:
            _fail("GET /status failed")
            return 1

        if args.health_only:
            print()
            _ok("Health check passed — server is operational")
            print()
            return 0

        # ── Step 3: Fire training run ──────────────────────────────────────

        print()
        _info("Firing training run: POST /train")
        _info(f"  breakout_type=all, session=all, symbols={symbols}")
        print()

        train_body = {
            "symbols": symbols,
            "days_back": days,
            "breakout_type": "all",
            "orb_session": "all",
            "bars_source": "massive",
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": 0.0002,
            "patience": patience,
            "min_accuracy": SMOKE_MIN_ACC,
            "min_precision": SMOKE_MIN_PRECISION,
            "min_recall": SMOKE_MIN_RECALL,
            "force_promote": True,  # Always promote in smoke test
        }

        status_code, resp = _http_post(f"{base_url}/train", train_body)
        if status_code == 202:
            _ok("Training started (202 Accepted)")
            if resp:
                _dim(f"Params: {json.dumps(resp.get('params', {}), indent=None)}")
        elif status_code == 409:
            _warn("Training already in progress (409 Conflict)")
        else:
            _fail(f"POST /train failed with status {status_code}")
            if resp:
                _fail(f"  Response: {json.dumps(resp)}")
            return 1

        # ── Step 4: Poll until complete ────────────────────────────────────

        print()
        _info("Polling training status...")

        start_time = time.monotonic()
        last_status = ""
        last_progress = ""
        dataset_logged = False
        training_logged = False

        while True:
            elapsed = time.monotonic() - start_time
            if elapsed > MAX_WAIT_TRAIN:
                _fail(f"Training timed out after {_format_duration(elapsed)}")
                return 1

            status = _http_get(f"{base_url}/status")
            if not status:
                _warn("Failed to poll /status — retrying...")
                time.sleep(POLL_INTERVAL)
                continue

            current_status = status.get("status", "unknown")
            current_progress = status.get("progress", "")

            # Log transitions
            if current_status != last_status or current_progress != last_progress:
                elapsed_str = _format_duration(elapsed)
                if current_status == "generating_dataset":
                    color = YELLOW
                    if not dataset_logged:
                        _info(f"[{elapsed_str}] Dataset generation started...")
                        dataset_logged = True
                elif current_status == "training":
                    color = CYAN
                    if not training_logged:
                        _info(
                            f"[{elapsed_str}] CNN training started on {'GPU' if cuda_info.get('available') else 'CPU'}..."
                        )
                        training_logged = True
                elif current_status == "evaluating":
                    color = CYAN
                elif current_status in ("promoting", "done"):
                    color = GREEN
                elif current_status == "failed":
                    color = RED
                elif current_status == "cancelled":
                    color = YELLOW
                else:
                    color = DIM

                status_line = f"[{elapsed_str}] {current_status}"
                if current_progress:
                    status_line += f" — {current_progress}"
                _log(f"  {status_line}", color)

                last_status = current_status
                last_progress = current_progress

            # Terminal states
            if current_status in ("done", "failed", "cancelled"):
                break

            time.sleep(POLL_INTERVAL)

        # ── Step 5: Report results ─────────────────────────────────────────

        total_elapsed = time.monotonic() - start_time
        print()
        print(f"{BOLD}{'─' * 70}{RESET}")
        print(f"{BOLD}  Results{RESET}")
        print(f"{BOLD}{'─' * 70}{RESET}")
        print()

        if current_status == "done":
            result = status.get("last_result", {})
            metrics = result.get("metrics", {})
            gates = result.get("gates", {})
            dataset_info = result.get("dataset", {})
            promoted = result.get("promoted", False)

            _ok(f"Training completed in {_format_duration(total_elapsed)}")
            print()

            # Dataset stats
            total_images = dataset_info.get("total_images", "?")
            _info(f"Dataset:    {total_images} images generated")

            # Metrics
            val_acc = metrics.get("val_accuracy", 0)
            val_prec = metrics.get("val_precision", 0)
            val_rec = metrics.get("val_recall", 0)
            epochs_trained = metrics.get("epochs_trained", "?")
            best_epoch = metrics.get("best_epoch", "?")

            _info(f"Accuracy:   {val_acc:.1f}%")
            _info(f"Precision:  {val_prec:.1f}%")
            _info(f"Recall:     {val_rec:.1f}%")
            _info(f"Epochs:     {epochs_trained} (best: {best_epoch})")
            print()

            # Gates
            if gates.get("passed"):
                _ok("All validation gates passed")
            else:
                failures = gates.get("failures", [])
                _warn(f"Validation gates failed: {'; '.join(failures)}")

            # Promotion
            if promoted:
                _ok("Model promoted to champion")
            else:
                reason = result.get("reason", "unknown")
                _warn(f"Not promoted: {reason}")

            print()

            # Verify model file exists
            champion_pt = Path(models_dir) / "breakout_cnn_best.pt"
            champion_meta = Path(models_dir) / "breakout_cnn_best_meta.json"
            if champion_pt.exists():
                size_mb = champion_pt.stat().st_size / 1e6
                _ok(f"Champion .pt exists ({size_mb:.1f} MB)")
            else:
                _warn("Champion .pt not found on disk")

            if champion_meta.exists():
                _ok("Champion metadata JSON exists")
            else:
                _dim("Champion metadata JSON not found")

            print()
            print(f"{BOLD}{'─' * 70}{RESET}")
            _ok(f"SMOKE TEST PASSED — pipeline ran end-to-end in {_format_duration(total_elapsed)}")
            print(f"{BOLD}{'─' * 70}{RESET}")
            print()
            return 0

        elif current_status == "failed":
            error = status.get("error", "unknown error")
            _fail(f"Training FAILED after {_format_duration(total_elapsed)}")
            _fail(f"  Error: {error}")

            # Check if it's a data issue vs code issue
            if "Insufficient training data" in str(error):
                _warn("This may indicate the Massive API returned no data for the requested symbols/days.")
                _info("Try: --symbols MGC --days 30  (longer window, single symbol)")
            elif "torch" in str(error).lower() or "cuda" in str(error).lower():
                _warn("This may be a GPU/CUDA issue.")
                _info("Check: nvidia-smi, torch.cuda.is_available()")

            print()
            print(f"{BOLD}{'─' * 70}{RESET}")
            _fail("SMOKE TEST FAILED")
            print(f"{BOLD}{'─' * 70}{RESET}")
            print()
            return 1

        else:
            _warn(f"Training ended with unexpected status: {current_status}")
            print()
            return 1

    finally:
        _cleanup()
        # Clean up the smoke test dataset directory if it was created
        # (leave models/ untouched — those are valuable)
        if args.cleanup:
            import shutil

            ds_path = Path(dataset_dir)
            if ds_path.exists() and "smoke_test" in str(ds_path):
                _dim(f"Cleaning up smoke test dataset: {ds_path}")
                shutil.rmtree(ds_path, ignore_errors=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Smoke test the trainer server with all 13 breakout types",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              python scripts/smoke_test_trainer.py                     # default: MGC+MES, 7 days, 3 epochs
              python scripts/smoke_test_trainer.py --quick             # fast: MGC only, 5 days, 2 epochs
              python scripts/smoke_test_trainer.py --health-only       # just check server boots + GPU
              python scripts/smoke_test_trainer.py --symbols MGC MES MNQ --days 14 --epochs 5
        """),
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=DEFAULT_SYMBOLS,
        help=f"Symbols to train on (default: {' '.join(DEFAULT_SYMBOLS)})",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=DEFAULT_DAYS,
        help=f"Days of history for dataset generation (default: {DEFAULT_DAYS})",
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
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: 1 symbol, 5 days, 2 epochs",
    )
    parser.add_argument(
        "--health-only",
        action="store_true",
        help="Only check server health + GPU, then exit",
    )
    parser.add_argument(
        "--no-cleanup",
        action="store_true",
        dest="no_cleanup",
        help="Don't clean up the smoke test dataset directory",
    )

    args = parser.parse_args()
    args.cleanup = not args.no_cleanup

    sys.exit(run_smoke_test(args))


if __name__ == "__main__":
    main()
