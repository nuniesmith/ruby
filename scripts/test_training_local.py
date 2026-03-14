#!/usr/bin/env python3
"""
test_training_local.py — Local GPU Training Test (Real Dataset)
================================================================
Tests the full CNN training pipeline locally on the oryx server using
the real dataset (no Docker, no trainer server, no synthetic data).

What it does:
  1. Verifies CUDA / GPU availability and reports device info
  2. Loads the real dataset from ``dataset/`` (labels.csv or train/val split)
  3. Fixes Docker ``/app/`` path prefixes in-memory if needed (non-destructive)
  4. Runs 3 epochs of training with batch_size=16 to verify the pipeline works
  5. Reports GPU memory usage, throughput (samples/sec), and loss curves
  6. Saves a test model to ``models/test_run_*.pt``
  7. Cleans up test models on success (keeps them on failure for debugging)

This is a quick integration test (~2-5 min on RTX 3080) to verify
everything works before committing to a full 80-epoch training run.

Usage:
    .venv/bin/python scripts/test_training_local.py

    # Keep test model after success (for inspection):
    .venv/bin/python scripts/test_training_local.py --keep-model

    # Use a specific CSV:
    .venv/bin/python scripts/test_training_local.py --csv dataset/labels.csv

    # Override epochs or batch size:
    .venv/bin/python scripts/test_training_local.py --epochs 5 --batch-size 8

Exit codes:
    0 — training completed successfully
    1 — training failed
"""

from __future__ import annotations

import argparse
import gc
import glob
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup — ensure src/ is importable (same pattern as other scripts/)
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Also set PYTHONPATH for any subprocess that might be spawned by DataLoader
os.environ.setdefault("PYTHONPATH", str(SRC_DIR))

# ---------------------------------------------------------------------------
# Terminal colours (match smoke_test_v8_training.py style)
# ---------------------------------------------------------------------------
BOLD = "\033[1m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
CYAN = "\033[96m"
DIM = "\033[2m"
RESET = "\033[0m"

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_CSV = str(PROJECT_ROOT / "dataset" / "labels.csv")
TRAIN_CSV = str(PROJECT_ROOT / "dataset" / "train.csv")
VAL_CSV = str(PROJECT_ROOT / "dataset" / "val.csv")
MODEL_DIR = str(PROJECT_ROOT / "models")
DOCKER_PREFIX = "/app/"

TEST_EPOCHS = 3
TEST_BATCH_SIZE = 16
TEST_LR = 2e-4
TEST_PATIENCE = 10  # won't trigger in 3 epochs
TEST_FREEZE_EPOCHS = 1
TEST_WARMUP_EPOCHS = 1
TEST_GRAD_ACCUM = 2
TEST_MIXUP_ALPHA = 0.2


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


def _section(title: str) -> None:
    print(f"\n{BOLD}{'─' * 60}{RESET}")
    _log(title, BOLD)
    print(f"{BOLD}{'─' * 60}{RESET}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _fix_docker_paths_in_csv(csv_path: str) -> str:
    """Create a temp copy of a CSV with /app/ prefixes stripped.

    Returns the path to the fixed CSV (may be the original if no fix needed).
    This is non-destructive — the original CSV is never modified.
    """
    import pandas as pd

    df = pd.read_csv(csv_path)
    if "image_path" not in df.columns:
        return csv_path

    docker_count = sum(1 for p in df["image_path"] if str(p).startswith(DOCKER_PREFIX))
    if docker_count == 0:
        return csv_path

    _warn(f"Found {docker_count:,d} Docker-prefixed paths — fixing in temp copy")

    df["image_path"] = df["image_path"].apply(
        lambda p: str(p)[len(DOCKER_PREFIX) :] if str(p).startswith(DOCKER_PREFIX) else str(p)
    )

    # Write to a temp file next to the original
    import tempfile

    tmp_dir = os.path.dirname(csv_path)
    fd, tmp_path = tempfile.mkstemp(suffix=".csv", prefix="test_train_fixed_", dir=tmp_dir)
    os.close(fd)
    df.to_csv(tmp_path, index=False)
    _info(f"  Temp CSV: {tmp_path}")

    return tmp_path


def _gpu_memory_stats() -> dict:
    """Return current GPU memory usage stats (in MB)."""
    try:
        import torch

        if not torch.cuda.is_available():
            return {}
        return {
            "allocated_mb": round(torch.cuda.memory_allocated() / 1024**2, 1),
            "reserved_mb": round(torch.cuda.memory_reserved() / 1024**2, 1),
            "max_allocated_mb": round(torch.cuda.max_memory_allocated() / 1024**2, 1),
            "max_reserved_mb": round(torch.cuda.max_memory_reserved() / 1024**2, 1),
            "free_mb": round(torch.cuda.mem_get_info()[0] / 1024**2, 1),
            "total_mb": round(torch.cuda.mem_get_info()[1] / 1024**2, 1),
        }
    except Exception:
        return {}


def _find_test_models() -> list[str]:
    """Find all test_run_*.pt model files."""
    return sorted(glob.glob(os.path.join(MODEL_DIR, "test_run_*.pt")))


# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------
def check_environment() -> bool:
    """Verify CUDA, torch, and dataset are available."""
    _section("Environment Check")
    ok = True

    # Python version
    _info(f"Python: {sys.version.split()[0]}")
    _info(f"Project root: {PROJECT_ROOT}")

    # Torch
    try:
        import torch

        _ok(f"PyTorch: {torch.__version__}")
    except ImportError:
        _fail("PyTorch not installed — cannot train")
        return False

    # CUDA
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        vram_total = torch.cuda.get_device_properties(0).total_mem / 1024**3
        _ok(f"CUDA available: {device_name} ({vram_total:.1f} GiB VRAM)")
        _info(f"  CUDA version: {torch.version.cuda}")
        _info(f"  cuDNN: {torch.backends.cudnn.version()}")  # type: ignore[attr-defined]

        # Initial memory
        mem = _gpu_memory_stats()
        if mem:
            _info(f"  GPU memory: {mem['free_mb']:.0f} MB free / {mem['total_mb']:.0f} MB total")
    else:
        _warn("CUDA not available — training will run on CPU (very slow)")

    # torchvision
    try:
        import torchvision  # noqa: F811

        _ok(f"torchvision: {torchvision.__version__}")
    except ImportError:
        _fail("torchvision not installed")
        ok = False

    # PIL
    try:
        import importlib.util

        if importlib.util.find_spec("PIL") is not None:
            _ok("Pillow available")
        else:
            raise ImportError("PIL not found")
    except ImportError:
        _fail("Pillow not installed")
        ok = False

    # Breakout CNN module
    try:
        from lib.analysis.ml.breakout_cnn import (
            get_device,
        )
        from lib.analysis.ml.breakout_cnn import (
            train_model as _train_model_check,
        )

        _ = _train_model_check  # verify import succeeded
        _ok("breakout_cnn module importable")
        _info(f"  Training device: {get_device()}")
    except ImportError as e:
        _fail(f"Cannot import breakout_cnn: {e}")
        ok = False

    return ok


def check_dataset(csv_path: str) -> tuple[bool, str, str | None]:
    """Check dataset CSV and return (ok, train_csv, val_csv_or_none).

    If train.csv + val.csv exist alongside labels.csv, prefer those.
    """
    _section("Dataset Check")
    import pandas as pd

    # Check if pre-split CSVs exist
    train_csv: str | None = None
    val_csv: str | None = None

    if os.path.isfile(TRAIN_CSV) and os.path.isfile(VAL_CSV):
        train_df = pd.read_csv(TRAIN_CSV)
        val_df = pd.read_csv(VAL_CSV)
        _ok("Pre-split datasets found:")
        _info(f"  train.csv: {len(train_df):,d} rows")
        _info(f"  val.csv:   {len(val_df):,d} rows")
        train_csv = TRAIN_CSV
        val_csv = VAL_CSV
    elif os.path.isfile(csv_path):
        df = pd.read_csv(csv_path)
        _ok(f"Dataset CSV: {csv_path} ({len(df):,d} rows)")
        _info("  No pre-split train/val — will use auto-split (85/15)")
        train_csv = csv_path
        val_csv = None
    else:
        _fail(f"Dataset CSV not found: {csv_path}")
        return False, "", None

    # Quick image check (sample 50 random rows)
    check_csv = train_csv
    df = pd.read_csv(check_csv)
    if "image_path" in df.columns:
        sample_size = min(50, len(df))
        sample = df.sample(n=sample_size, random_state=42)
        found = 0
        docker_paths = 0
        for _, row in sample.iterrows():
            raw = str(row["image_path"]).strip()
            if raw.startswith(DOCKER_PREFIX):
                docker_paths += 1
                local = raw[len(DOCKER_PREFIX) :]
            else:
                local = raw
            if os.path.isfile(local) or os.path.isfile(raw):
                found += 1

        _info(f"  Image spot-check: {found}/{sample_size} found")
        if docker_paths > 0:
            _warn(f"  {docker_paths}/{sample_size} paths have Docker /app/ prefix")
            _info("  Will fix in temp copy (non-destructive)")

        if found == 0:
            _fail("No images found in spot-check — verify dataset/images/ exists")
            return False, "", None

    # Label check
    if "label" in df.columns:
        labels = df["label"].value_counts()
        _info("  Labels: %s" % dict(labels))

    return True, train_csv, val_csv


# ---------------------------------------------------------------------------
# Training run
# ---------------------------------------------------------------------------
def run_training(
    train_csv: str,
    val_csv: str | None,
    epochs: int = TEST_EPOCHS,
    batch_size: int = TEST_BATCH_SIZE,
) -> tuple[bool, str | None]:
    """Run a short training test and return (success, model_path).

    Returns:
        (success, model_path) — True and the saved model path on success,
        (False, None) on failure.
    """
    _section("Training Run")

    import torch

    from lib.analysis.ml.breakout_cnn import train_model

    # Fix Docker paths in temp copies
    temp_files: list[str] = []
    try:
        fixed_train = _fix_docker_paths_in_csv(train_csv)
        if fixed_train != train_csv:
            temp_files.append(fixed_train)
            train_csv = fixed_train

        if val_csv:
            fixed_val = _fix_docker_paths_in_csv(val_csv)
            if fixed_val != val_csv:
                temp_files.append(fixed_val)
                val_csv = fixed_val

        # Pre-training GPU state
        pre_mem = _gpu_memory_stats()
        if pre_mem:
            _info(f"GPU before training: {pre_mem['allocated_mb']:.0f} MB allocated, {pre_mem['free_mb']:.0f} MB free")

        # Use a test-specific model dir prefix
        test_model_dir = MODEL_DIR
        os.makedirs(test_model_dir, exist_ok=True)

        _info("Training config:")
        _info(f"  Epochs:          {epochs}")
        _info(f"  Batch size:      {batch_size}")
        _info(f"  Grad accum:      {TEST_GRAD_ACCUM}  (effective batch: {batch_size * TEST_GRAD_ACCUM})")
        _info(f"  Learning rate:   {TEST_LR}")
        _info(f"  Freeze epochs:   {TEST_FREEZE_EPOCHS}")
        _info(f"  Warmup epochs:   {TEST_WARMUP_EPOCHS}")
        _info(f"  Mixup alpha:     {TEST_MIXUP_ALPHA}")
        _info(f"  Train CSV:       {train_csv}")
        _info(f"  Val CSV:         {val_csv or '(auto-split 85/15)'}")
        _info(f"  Device:          {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")

        # Capture training time
        start_time = time.time()

        # Hook into the training loop via logging to capture per-epoch stats
        # We use the standard train_model() function — same as production
        import logging

        epoch_logs: list[str] = []
        _original_handler_count = len(logging.getLogger("analysis.breakout_cnn").handlers)

        class _EpochCapture(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                msg = record.getMessage()
                if "Epoch " in msg and "Train Loss" in msg:
                    epoch_logs.append(msg)

        capture_handler = _EpochCapture()
        cnn_logger = logging.getLogger("analysis.breakout_cnn")
        cnn_logger.addHandler(capture_handler)
        cnn_logger.setLevel(logging.INFO)

        # Also add a stream handler so training logs are visible
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(logging.Formatter(f"{DIM}[%(name)s]{RESET} %(message)s"))
        cnn_logger.addHandler(stream_handler)

        print()  # blank line before training output
        result = train_model(
            data_csv=train_csv,
            val_csv=val_csv,
            epochs=epochs,
            batch_size=batch_size,
            lr=TEST_LR,
            weight_decay=2e-4,
            freeze_epochs=TEST_FREEZE_EPOCHS,
            model_dir=test_model_dir,
            image_root=None,  # paths in CSV are relative to cwd (project root)
            num_workers=4,
            save_best=True,
            patience=TEST_PATIENCE,
            grad_accum_steps=TEST_GRAD_ACCUM,
            mixup_alpha=TEST_MIXUP_ALPHA,
            warmup_epochs=TEST_WARMUP_EPOCHS,
        )
        print()  # blank line after training output

        elapsed = time.time() - start_time

        # Clean up logging handlers
        cnn_logger.removeHandler(capture_handler)
        cnn_logger.removeHandler(stream_handler)

        if result is None:
            _fail("train_model() returned None — training failed")
            return False, None

        # ------------------------------------------------------------------
        # Report results
        # ------------------------------------------------------------------
        _section("Training Results")

        _ok(f"Training completed in {elapsed:.1f}s ({elapsed / 60:.1f} min)")
        _info(f"  Epochs trained: {result.epochs_trained}")
        _info(f"  Best epoch:     {result.best_epoch}")
        _info(f"  Model path:     {result.model_path}")

        if os.path.isfile(result.model_path):
            model_size_mb = os.path.getsize(result.model_path) / 1024**2
            _info(f"  Model size:     {model_size_mb:.1f} MB")

        # Throughput estimate
        import pandas as pd

        total_samples = len(pd.read_csv(train_csv))
        total_samples_processed = total_samples * result.epochs_trained
        throughput = total_samples_processed / elapsed
        _info(f"  Throughput:     {throughput:.0f} samples/sec")
        _info(f"  Total samples:  {total_samples_processed:,d} ({total_samples:,d} x {result.epochs_trained} epochs)")

        # GPU memory post-training
        post_mem = _gpu_memory_stats()
        if post_mem:
            _section("GPU Memory Usage")
            _info(f"  Current allocated: {post_mem['allocated_mb']:.0f} MB")
            _info(f"  Current reserved:  {post_mem['reserved_mb']:.0f} MB")
            _info(f"  Peak allocated:    {post_mem['max_allocated_mb']:.0f} MB")
            _info(f"  Peak reserved:     {post_mem['max_reserved_mb']:.0f} MB")
            _info(f"  Free:              {post_mem['free_mb']:.0f} MB / {post_mem['total_mb']:.0f} MB total")

        # Loss curve from captured logs
        if epoch_logs:
            _section("Loss Curve (per epoch)")
            for log_line in epoch_logs:
                _info(f"  {log_line}")

        # Rename the model to test_run_* pattern for easy identification/cleanup
        test_model_name = f"test_run_{datetime.now():%Y%m%d_%H%M%S}.pt"
        test_model_path = os.path.join(test_model_dir, test_model_name)
        try:
            os.rename(result.model_path, test_model_path)
            _ok(f"Model renamed: {test_model_path}")
        except OSError:
            # If rename fails (e.g. cross-device), just copy
            import shutil

            shutil.copy2(result.model_path, test_model_path)
            os.remove(result.model_path)
            _ok(f"Model moved: {test_model_path}")

        # Also clean up the _final.pt model that train_model always saves
        final_models = glob.glob(os.path.join(test_model_dir, "breakout_cnn_*_final.pt"))
        for fm in final_models:
            # Only remove finals created during this run (within last few minutes)
            try:
                age_s = time.time() - os.path.getmtime(fm)
                if age_s < elapsed + 60:
                    os.remove(fm)
                    _info(f"  Cleaned up final model: {os.path.basename(fm)}")
            except OSError:
                pass

        return True, test_model_path

    finally:
        # Clean up temp CSV files
        for tmp in temp_files:
            try:
                os.remove(tmp)
                _info(f"  Cleaned up temp CSV: {os.path.basename(tmp)}")
            except OSError:
                pass

        # Release GPU memory
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test the full CNN training pipeline locally with real data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--csv",
        default=DEFAULT_CSV,
        help=f"Path to dataset CSV (default: {DEFAULT_CSV})",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=TEST_EPOCHS,
        help=f"Number of test epochs (default: {TEST_EPOCHS})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=TEST_BATCH_SIZE,
        help=f"Batch size (default: {TEST_BATCH_SIZE})",
    )
    parser.add_argument(
        "--keep-model",
        action="store_true",
        help="Keep the test model after successful run (default: clean up)",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    print(f"\n{BOLD}Local Training Test — Breakout CNN{RESET}")
    print(f"{DIM}Server: oryx | {datetime.now():%Y-%m-%d %H:%M:%S}{RESET}")
    print(f"{DIM}{'─' * 60}{RESET}")

    # Change to project root so relative paths work
    os.chdir(str(PROJECT_ROOT))

    # 1. Environment check
    if not check_environment():
        _fail("Environment check failed — cannot proceed")
        return 1

    # 2. Dataset check
    ok, train_csv, val_csv = check_dataset(args.csv)
    if not ok:
        _fail("Dataset check failed — cannot proceed")
        return 1

    # 3. Run training
    try:
        success, model_path = run_training(
            train_csv=train_csv,
            val_csv=val_csv,
            epochs=args.epochs,
            batch_size=args.batch_size,
        )
    except Exception as e:
        _section("Training FAILED")
        _fail(f"Exception during training: {e}")
        traceback.print_exc()
        return 1

    if not success:
        _fail("Training did not complete successfully")
        return 1

    # 4. Cleanup or keep
    _section("Cleanup")
    if model_path and os.path.isfile(model_path):
        if args.keep_model:
            _ok(f"Keeping test model: {model_path}")
        else:
            try:
                os.remove(model_path)
                _ok(f"Cleaned up test model: {os.path.basename(model_path)}")
            except OSError as e:
                _warn(f"Could not remove test model: {e}")

    # Also clean up any stale test_run_* models older than 1 hour
    stale_models = _find_test_models()
    stale_cleaned = 0
    for m in stale_models:
        try:
            age_h = (time.time() - os.path.getmtime(m)) / 3600
            if age_h > 1.0:
                os.remove(m)
                stale_cleaned += 1
        except OSError:
            pass
    if stale_cleaned > 0:
        _info(f"  Cleaned up {stale_cleaned} stale test model(s) (>1h old)")

    _section("Done")
    _ok("Local training test PASSED — pipeline is healthy ✓")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
