#!/usr/bin/env python3
"""
validate_dataset.py — Dataset Validation & Reporting
======================================================
Standalone script that validates the breakout CNN training dataset,
reports statistics, and optionally fixes Docker-prefixed image paths.

What it checks:
  1. Reads ``dataset/labels.csv`` and counts total rows
  2. Checks how many images exist on disk (trying both ``/app/`` Docker
     prefix and the local relative path)
  3. Reports label distribution (good_long, good_short, bad_long, bad_short)
  4. Reports symbol distribution (per-ticker row counts)
  5. Reports breakout_type distribution (ORB, PrevDay, etc.)
  6. Reports session distribution (us, london, cme, etc.)
  7. Reports per-symbol missing image counts
  8. Optionally fixes CSV paths by stripping the ``/app/`` prefix (``--fix``)
  9. Generates a summary JSON at ``dataset/validation_report.json``

Usage:
    # Validate only (read-only):
    .venv/bin/python scripts/validate_dataset.py

    # Validate and fix Docker path prefixes in labels.csv:
    .venv/bin/python scripts/validate_dataset.py --fix

    # Custom CSV path:
    .venv/bin/python scripts/validate_dataset.py --csv dataset/train.csv

Exit codes:
    0 — dataset is valid (all images found)
    1 — some images are missing or other issues detected
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
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
# Defaults
# ---------------------------------------------------------------------------
DEFAULT_CSV = str(PROJECT_ROOT / "dataset" / "labels.csv")
DEFAULT_REPORT = str(PROJECT_ROOT / "dataset" / "validation_report.json")
DOCKER_PREFIX = "/app/"


# ---------------------------------------------------------------------------
# Logging helpers (match smoke_test_v8_training.py style)
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
# Path resolution helpers
# ---------------------------------------------------------------------------
def _strip_docker_prefix(path: str) -> str:
    """Strip ``/app/`` prefix from an image path if present."""
    p = str(path).strip()
    if p.startswith(DOCKER_PREFIX):
        return p[len(DOCKER_PREFIX) :]
    return p


def _resolve_image_path(raw_path: str) -> tuple[str, bool]:
    """Try to resolve an image path, checking both Docker and local forms.

    Returns:
        (resolved_path, exists) — the resolved path string and whether the
        file was found on disk.
    """
    raw = str(raw_path).strip()

    # 1. Try raw path as-is (works inside Docker or if already local)
    if os.path.isfile(raw):
        return raw, True

    # 2. Try stripping /app/ prefix (Docker → local)
    local = _strip_docker_prefix(raw)
    if local != raw and os.path.isfile(local):
        return local, True

    # 3. Try relative to project root
    project_rel = os.path.join(str(PROJECT_ROOT), local)
    if os.path.isfile(project_rel):
        return project_rel, True

    return raw, False


# ---------------------------------------------------------------------------
# Main validation logic
# ---------------------------------------------------------------------------
def validate_dataset(
    csv_path: str,
    fix_paths: bool = False,
    report_path: str = DEFAULT_REPORT,
) -> dict:
    """Validate a dataset CSV and generate a comprehensive report.

    Args:
        csv_path: Path to the CSV file to validate.
        fix_paths: If True, strip ``/app/`` prefix from image_path entries
                   and write back to the same CSV.
        report_path: Where to write the JSON summary report.

    Returns:
        Report dict with validation results.
    """
    import pandas as pd

    _section(f"Validating dataset: {csv_path}")

    if not os.path.isfile(csv_path):
        _fail(f"CSV not found: {csv_path}")
        return {"valid": False, "error": f"CSV not found: {csv_path}"}

    df = pd.read_csv(csv_path)
    total_rows = len(df)
    _ok(f"Loaded CSV: {total_rows:,} rows, {len(df.columns)} columns")
    _info(f"Columns: {', '.join(df.columns[:10])}{'...' if len(df.columns) > 10 else ''}")

    report: dict = {
        "csv_path": csv_path,
        "timestamp": datetime.now().isoformat(),
        "total_rows": total_rows,
        "columns": list(df.columns),
        "valid": True,
        "errors": [],
        "warnings": [],
    }

    # ------------------------------------------------------------------
    # 1. Check required columns
    # ------------------------------------------------------------------
    _section("Column Validation")
    required = ["image_path", "label", "symbol"]
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        _fail(f"Missing required columns: {missing_cols}")
        report["errors"].append(f"Missing columns: {missing_cols}")
        report["valid"] = False
    else:
        _ok(f"All required columns present: {required}")

    # ------------------------------------------------------------------
    # 2. Label distribution
    # ------------------------------------------------------------------
    _section("Label Distribution")
    if "label" in df.columns:
        label_counts = df["label"].value_counts().to_dict()
        report["label_distribution"] = {str(k): int(v) for k, v in label_counts.items()}
        for label, count in sorted(label_counts.items()):
            pct = count / total_rows * 100
            bar = "█" * int(pct / 2)
            _info(f"  {label:<15s} {count:>6,d}  ({pct:5.1f}%)  {bar}")

        # Good vs bad ratio
        good_count = sum(int(v) for k, v in label_counts.items() if str(k).startswith("good"))
        bad_count = sum(int(v) for k, v in label_counts.items() if str(k).startswith("bad"))
        _info("  " + "─" * 40)
        _info(f"  Good total:    {good_count:>6,d}  ({good_count / total_rows * 100:.1f}%)")
        _info(f"  Bad total:     {bad_count:>6,d}  ({bad_count / total_rows * 100:.1f}%)")
        report["good_count"] = good_count
        report["bad_count"] = bad_count
    else:
        _warn("No 'label' column found")

    # ------------------------------------------------------------------
    # 3. Symbol distribution
    # ------------------------------------------------------------------
    _section("Symbol Distribution")
    if "symbol" in df.columns:
        symbol_counts = df["symbol"].value_counts().to_dict()
        report["symbol_distribution"] = {str(k): int(v) for k, v in symbol_counts.items()}
        report["unique_symbols"] = len(symbol_counts)
        _info(f"  {len(symbol_counts)} unique symbols")
        for sym, count in sorted(symbol_counts.items(), key=lambda x: -int(x[1])):
            pct = count / total_rows * 100
            bar = "█" * max(1, int(pct / 2))
            _info(f"  {sym:<10s} {count:>6,d}  ({pct:5.1f}%)  {bar}")
    else:
        _warn("No 'symbol' column found")

    # ------------------------------------------------------------------
    # 4. Breakout type distribution
    # ------------------------------------------------------------------
    _section("Breakout Type Distribution")
    if "breakout_type" in df.columns:
        bt_counts = df["breakout_type"].value_counts().to_dict()
        report["breakout_type_distribution"] = {str(k): int(v) for k, v in bt_counts.items()}
        for bt, count in sorted(bt_counts.items(), key=lambda x: -int(x[1])):
            pct = count / total_rows * 100
            bar = "█" * max(1, int(pct / 2))
            _info(f"  {str(bt):<25s} {count:>6,d}  ({pct:5.1f}%)  {bar}")
    else:
        _warn("No 'breakout_type' column found")
        report["warnings"].append("No breakout_type column")

    # ------------------------------------------------------------------
    # 5. Session distribution
    # ------------------------------------------------------------------
    _section("Session Distribution")
    if "session_key" in df.columns:
        session_counts = df["session_key"].value_counts().to_dict()
        report["session_distribution"] = {str(k): int(v) for k, v in session_counts.items()}
        for sess, count in sorted(session_counts.items(), key=lambda x: -int(x[1])):
            pct = count / total_rows * 100
            bar = "█" * max(1, int(pct / 2))
            _info(f"  {str(sess):<20s} {count:>6,d}  ({pct:5.1f}%)  {bar}")
    else:
        _warn("No 'session_key' column found — inferring from breakout_time")
        # Try to infer session from breakout_time
        if "breakout_time" in df.columns:
            sessions: list[str] = []
            for bt in df["breakout_time"]:
                try:
                    bt_str = str(bt).strip()
                    if " " in bt_str:
                        hour = int(bt_str.split(" ")[1].split(":")[0])
                        if hour < 3:
                            sessions.append("asia")
                        elif hour < 8:
                            sessions.append("london")
                        else:
                            sessions.append("us")
                    else:
                        sessions.append("unknown")
                except Exception:
                    sessions.append("unknown")
            session_counts_inferred = dict(Counter(sessions))
            report["session_distribution_inferred"] = {str(k): int(v) for k, v in session_counts_inferred.items()}
            for sess, count in sorted(session_counts_inferred.items(), key=lambda x: -int(x[1])):
                pct = count / total_rows * 100
                _info(f"  {sess:<20s} {count:>6,d}  ({pct:5.1f}%)  (inferred)")

    # ------------------------------------------------------------------
    # 6. Image path validation
    # ------------------------------------------------------------------
    _section("Image Path Validation")
    if "image_path" not in df.columns:
        _fail("No 'image_path' column — cannot check images")
        report["errors"].append("No image_path column")
        report["valid"] = False
    else:
        # Count path prefix styles
        docker_prefixed = sum(1 for p in df["image_path"] if str(p).startswith(DOCKER_PREFIX))
        local_paths = total_rows - docker_prefixed
        _info(f"  Docker-prefixed (/app/...): {docker_prefixed:,d}")
        _info(f"  Local paths:               {local_paths:,d}")
        report["docker_prefixed_count"] = int(docker_prefixed)
        report["local_path_count"] = int(local_paths)

        # Check existence
        found = 0
        missing = 0
        empty_paths = 0
        missing_per_symbol: dict[str, int] = defaultdict(int)
        missing_samples: list[str] = []

        for _, row in df.iterrows():
            raw_path = row.get("image_path", "")
            if not raw_path or (isinstance(raw_path, float) and str(raw_path) == "nan"):
                empty_paths += 1
                continue

            _, exists = _resolve_image_path(str(raw_path))
            if exists:
                found += 1
            else:
                missing += 1
                sym = str(row.get("symbol", "UNKNOWN"))
                missing_per_symbol[sym] += 1
                if len(missing_samples) < 5:
                    missing_samples.append(str(raw_path))

        report["images_found"] = int(found)
        report["images_missing"] = int(missing)
        report["empty_image_paths"] = int(empty_paths)
        report["missing_per_symbol"] = dict(missing_per_symbol)

        if found == total_rows:
            _ok(f"All {found:,d} images found on disk")
        else:
            _ok(f"Images found:   {found:>6,d} / {total_rows:,d}")
            if missing > 0:
                _fail(f"Images missing: {missing:>6,d}")
                report["valid"] = False
                report["errors"].append(f"{missing} images not found on disk")
            if empty_paths > 0:
                _warn(f"Empty paths:    {empty_paths:>6,d}")
                report["warnings"].append(f"{empty_paths} empty image paths")

        # Per-symbol missing
        if missing_per_symbol:
            _section("Missing Images by Symbol")
            for sym, cnt in sorted(missing_per_symbol.items(), key=lambda x: -x[1]):
                _info(f"  {sym:<10s} {cnt:>6,d} missing")
            if missing_samples:
                _info("  Sample missing paths:")
                for p in missing_samples:
                    _info(f"    {p}")

    # ------------------------------------------------------------------
    # 7. Data quality checks
    # ------------------------------------------------------------------
    _section("Data Quality Checks")
    quality_issues = 0

    # Check for NaN in critical columns
    for col in ["label", "symbol", "image_path"]:
        if col in df.columns:
            nan_count = int(df[col].isna().sum())  # type: ignore[union-attr]
            if nan_count > 0:
                _warn(f"  {col}: {nan_count:,d} NaN values")
                report["warnings"].append(f"{col} has {nan_count} NaN values")
                quality_issues += 1

    # Check for duplicate image paths
    if "image_path" in df.columns:
        dup_count = int(df["image_path"].duplicated().sum())  # type: ignore[union-attr]
        if dup_count > 0:
            _warn(f"  Duplicate image_path entries: {dup_count:,d}")
            report["warnings"].append(f"{dup_count} duplicate image paths")
            quality_issues += 1
        else:
            _ok("No duplicate image paths")

    # Check label values are expected
    if "label" in df.columns:
        expected_labels = {"good_long", "good_short", "bad_long", "bad_short"}
        actual_labels = set(df["label"].dropna().unique())
        unexpected = actual_labels - expected_labels
        if unexpected:
            _warn(f"  Unexpected label values: {unexpected}")
            report["warnings"].append(f"Unexpected labels: {unexpected}")
            quality_issues += 1
        else:
            _ok(f"All labels are valid: {sorted(actual_labels)}")

    if quality_issues == 0:
        _ok("No data quality issues detected")

    report["quality_issues"] = quality_issues

    # ------------------------------------------------------------------
    # 8. Fix paths if requested
    # ------------------------------------------------------------------
    if fix_paths and "image_path" in df.columns:
        _section("Fixing Docker Path Prefixes")
        fixed_count = 0
        for i in range(len(df)):
            raw = str(df.iloc[i]["image_path"]).strip()
            if raw.startswith(DOCKER_PREFIX):
                df.at[df.index[i], "image_path"] = _strip_docker_prefix(raw)
                fixed_count += 1

        if fixed_count > 0:
            df.to_csv(csv_path, index=False)
            _ok(f"Fixed {fixed_count:,d} paths in {csv_path}")
            report["paths_fixed"] = int(fixed_count)
        else:
            _ok("No paths needed fixing (already local)")
            report["paths_fixed"] = 0

    # ------------------------------------------------------------------
    # 9. Write report JSON
    # ------------------------------------------------------------------
    _section("Summary")

    if report["valid"]:
        _ok(f"Dataset is VALID — {total_rows:,d} rows, all checks passed")
    else:
        _fail("Dataset has ISSUES — see errors above")
        for err in report["errors"]:
            _fail(f"  → {err}")

    if report.get("warnings"):
        for w in report["warnings"]:
            _warn(f"  → {w}")

    # Write report
    os.makedirs(os.path.dirname(report_path) or ".", exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    _ok(f"Report written to {report_path}")

    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate breakout CNN training dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--csv",
        default=DEFAULT_CSV,
        help=f"Path to dataset CSV (default: {DEFAULT_CSV})",
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Fix Docker /app/ prefixes in image_path entries",
    )
    parser.add_argument(
        "--report",
        default=DEFAULT_REPORT,
        help=f"Output path for JSON report (default: {DEFAULT_REPORT})",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    print(f"\n{BOLD}Dataset Validation Tool{RESET}")
    print(f"{DIM}Project root: {PROJECT_ROOT}{RESET}")
    print(f"{DIM}CSV path:     {args.csv}{RESET}")
    if args.fix:
        print(f"{YELLOW}⚠ Fix mode enabled — will modify CSV paths in-place{RESET}")

    # Change to project root so relative paths work
    os.chdir(str(PROJECT_ROOT))

    report = validate_dataset(
        csv_path=args.csv,
        fix_paths=args.fix,
        report_path=args.report,
    )

    print()
    return 0 if report.get("valid", False) else 1


if __name__ == "__main__":
    sys.exit(main())
