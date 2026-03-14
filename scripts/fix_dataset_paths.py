#!/usr/bin/env python3
"""
fix_dataset_paths.py — Strip Docker ``/app/`` Prefix from Dataset CSVs
========================================================================
Quick, idempotent script that fixes image paths in the breakout CNN
training dataset CSVs.  The dataset is generated inside a Docker
container where paths look like ``/app/dataset/images/...``, but on the
local host (oryx server) the correct path is ``dataset/images/...``.

What it does:
  1. Reads ``dataset/labels.csv``
  2. Strips the ``/app/`` prefix from every ``image_path`` entry
  3. Verifies that all images now resolve on disk
  4. Writes back to ``dataset/labels.csv``
  5. Also fixes ``dataset/train.csv`` and ``dataset/val.csv`` if they exist

Safety:
  - **Idempotent** — running twice is perfectly safe.  Paths that don't
    start with ``/app/`` are left untouched.
  - Creates a ``.bak`` backup of each CSV before the first modification.
  - Reports exactly what changed (or "nothing to do").

Usage:
    # From the project root:
    .venv/bin/python scripts/fix_dataset_paths.py

    # Dry-run (show what would change without writing):
    .venv/bin/python scripts/fix_dataset_paths.py --dry-run

    # Skip image verification (faster):
    .venv/bin/python scripts/fix_dataset_paths.py --skip-verify

Exit codes:
    0 — all paths fixed (or already correct) and all images verified
    1 — some images still missing after fix, or other error
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Path setup — same pattern as other scripts/ in this project
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

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
# Constants
# ---------------------------------------------------------------------------
DOCKER_PREFIX = "/app/"

CSV_FILES = [
    PROJECT_ROOT / "dataset" / "labels.csv",
    PROJECT_ROOT / "dataset" / "train.csv",
    PROJECT_ROOT / "dataset" / "val.csv",
]


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
# Core logic
# ---------------------------------------------------------------------------
def fix_csv(
    csv_path: Path,
    dry_run: bool = False,
    verify_images: bool = True,
) -> tuple[int, int, int]:
    """Fix Docker path prefixes in a single CSV file.

    Args:
        csv_path: Path to the CSV file.
        dry_run: If True, don't write changes — just report.
        verify_images: If True, check that fixed paths resolve on disk.

    Returns:
        (fixed_count, already_ok_count, missing_count)
    """
    import pandas as pd

    _section(f"Processing: {csv_path.name}")

    if not csv_path.is_file():
        _warn(f"File not found, skipping: {csv_path}")
        return 0, 0, 0

    df = pd.read_csv(csv_path)
    total = len(df)
    _info(f"Loaded {total:,d} rows")

    if "image_path" not in df.columns:
        _warn("No 'image_path' column — skipping")
        return 0, 0, 0

    # Count current state
    docker_count = 0
    local_count = 0
    empty_count = 0

    for path in df["image_path"]:
        p = str(path).strip()
        if not p or p.lower() == "nan":
            empty_count += 1
        elif p.startswith(DOCKER_PREFIX):
            docker_count += 1
        else:
            local_count += 1

    _info(f"Docker-prefixed (/app/...): {docker_count:,d}")
    _info(f"Already local:             {local_count:,d}")
    if empty_count > 0:
        _warn(f"Empty / NaN paths:         {empty_count:,d}")

    if docker_count == 0:
        _ok(f"Nothing to fix — all {local_count:,d} paths are already local")
        # Still verify images if requested
        missing = 0
        if verify_images:
            missing = _verify_images(df)
        return 0, local_count, missing

    # Strip /app/ prefix
    _info(f"Stripping '{DOCKER_PREFIX}' from {docker_count:,d} paths...")

    def _strip(p: str) -> str:
        s = str(p).strip()
        if s.startswith(DOCKER_PREFIX):
            return s[len(DOCKER_PREFIX) :]
        return s

    df["image_path"] = df["image_path"].apply(_strip)

    # Verify images resolve
    missing = 0
    if verify_images:
        missing = _verify_images(df)

    if dry_run:
        _warn(f"DRY RUN — would fix {docker_count:,d} paths in {csv_path.name}")
        return docker_count, local_count, missing

    # Create backup before first write (only if .bak doesn't already exist)
    # Non-fatal: if the directory is owned by root (e.g. Docker bind-mount),
    # we skip the backup rather than aborting the entire fix.
    bak_path = csv_path.with_suffix(".csv.bak")
    if not bak_path.exists():
        try:
            shutil.copy2(csv_path, bak_path)
            _info(f"Backup created: {bak_path.name}")
        except PermissionError:
            _warn(f"Could not create backup (permission denied) — proceeding without .bak")
    else:
        _info(f"Backup already exists: {bak_path.name}")

    # Write fixed CSV
    df.to_csv(csv_path, index=False)
    _ok(f"Fixed {docker_count:,d} paths → wrote {csv_path.name}")

    return docker_count, local_count, missing


def _verify_images(df) -> int:
    """Verify that all image_path entries in a DataFrame resolve on disk.

    Returns the count of missing images.
    """
    _info("Verifying images exist on disk...")

    missing = 0
    missing_samples: list[str] = []

    for path in df["image_path"]:
        p = str(path).strip()
        if not p or p.lower() == "nan":
            continue
        if not os.path.isfile(p):
            missing += 1
            if len(missing_samples) < 5:
                missing_samples.append(p)

    total = len(df)
    found = total - missing

    if missing == 0:
        _ok(f"All {found:,d} images verified on disk")
    else:
        _fail(f"{missing:,d} / {total:,d} images NOT found")
        for sample in missing_samples:
            _info(f"  Missing: {sample}")
        if missing > len(missing_samples):
            _info(f"  ... and {missing - len(missing_samples):,d} more")

    return missing


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Strip Docker /app/ prefix from dataset CSV image paths",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would change without writing files",
    )
    parser.add_argument(
        "--skip-verify",
        action="store_true",
        help="Skip image file existence verification (faster)",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    print(f"\n{BOLD}Fix Dataset Paths — Strip Docker /app/ Prefix{RESET}")
    print(f"{DIM}Project root: {PROJECT_ROOT}{RESET}")
    if args.dry_run:
        print(f"{YELLOW}⚠ DRY RUN — no files will be modified{RESET}")
    print()

    # Change to project root so relative paths work
    os.chdir(str(PROJECT_ROOT))

    total_fixed = 0
    total_already_ok = 0
    total_missing = 0
    files_processed = 0

    for csv_path in CSV_FILES:
        fixed, already_ok, missing = fix_csv(
            csv_path=csv_path,
            dry_run=args.dry_run,
            verify_images=not args.skip_verify,
        )
        total_fixed += fixed
        total_already_ok += already_ok
        total_missing += missing
        if csv_path.is_file():
            files_processed += 1

    # Summary
    _section("Summary")
    _info(f"Files processed:  {files_processed}")
    _info(f"Paths fixed:      {total_fixed:,d}")
    _info(f"Already correct:  {total_already_ok:,d}")

    if not args.skip_verify:
        if total_missing == 0:
            _ok("All images verified — dataset paths are correct ✓")
        else:
            _fail(f"{total_missing:,d} images still missing after fix")

    if total_fixed == 0 and total_missing == 0:
        _ok("Nothing to do — all paths already correct and all images present ✓")
    elif total_fixed > 0 and total_missing == 0:
        if args.dry_run:
            _warn(f"DRY RUN — {total_fixed:,d} paths would be fixed")
        else:
            _ok(f"Fixed {total_fixed:,d} paths across {files_processed} file(s) ✓")

    print()
    return 1 if total_missing > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
