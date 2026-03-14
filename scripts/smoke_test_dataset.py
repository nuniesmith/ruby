#!/usr/bin/env python3
"""
smoke_test_dataset.py — Validate real data loading & dataset generation before full v8 run.

This script does a quick end-to-end check of the dataset generation pipeline
using real data (not synthetic). It tests:

  1. ENGINE_DATA_URL connectivity (can the trainer reach the data service?)
  2. Bar loading for a small subset of symbols (1 futures + 1 crypto)
  3. Chart rendering (parity renderer produces valid PNGs)
  4. Label CSV generation (correct columns, valid values)
  5. Cross-asset peer bar loading (v8-B features)
  6. DatasetConfig defaults match v8 expectations

Run this BEFORE committing to the full 6-10 hour dataset generation.
Typical runtime: 30-90 seconds depending on network.

Usage:
    # From the project root (or inside the trainer container):
    python scripts/smoke_test_dataset.py

    # With explicit engine URL:
    ENGINE_DATA_URL=http://100.122.184.58:8050 python scripts/smoke_test_dataset.py

    # Quick mode — skip chart rendering (just test bar loading):
    python scripts/smoke_test_dataset.py --quick

    # Verbose logging:
    python scripts/smoke_test_dataset.py -v

Exit codes:
    0 — all checks passed, safe to run full dataset generation
    1 — one or more critical checks failed
    2 — warnings only (dataset generation will likely work but may be degraded)
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure the project src/ is importable
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
_SRC_DIR = _PROJECT_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

# Try to load .env for ENGINE_DATA_URL, API_KEY, etc.
try:
    from dotenv import load_dotenv

    _env_path = _PROJECT_ROOT / ".env"
    if _env_path.is_file():
        load_dotenv(_env_path)
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Colours & helpers
# ---------------------------------------------------------------------------

_RED = "\033[0;31m"
_GREEN = "\033[0;32m"
_YELLOW = "\033[1;33m"
_CYAN = "\033[0;36m"
_BOLD = "\033[1m"
_DIM = "\033[2m"
_NC = "\033[0m"

_PASS = 0
_FAIL = 0
_WARN = 0
_SKIP = 0


def ok(msg: str) -> None:
    global _PASS
    _PASS += 1
    print(f"  {_GREEN}✓{_NC} {msg}")


def fail(msg: str) -> None:
    global _FAIL
    _FAIL += 1
    print(f"  {_RED}✗{_NC} {msg}")


def warn(msg: str) -> None:
    global _WARN
    _WARN += 1
    print(f"  {_YELLOW}⚠{_NC} {msg}")


def skip(msg: str) -> None:
    global _SKIP
    _SKIP += 1
    print(f"  {_DIM}⊘ {msg}{_NC}")


def info(msg: str) -> None:
    print(f"  {_CYAN}ℹ{_NC} {msg}")


def header(msg: str) -> None:
    print(f"\n{_BOLD}{_CYAN}━━━ {msg} ━━━{_NC}")


# ---------------------------------------------------------------------------
# v8 expected values (must match DatasetConfig defaults & trainer_server.py)
# ---------------------------------------------------------------------------

ALL_25_SYMBOLS = [
    "MGC",
    "SIL",
    "MHG",
    "MCL",
    "MNG",  # metals + energy
    "MES",
    "MNQ",
    "M2K",
    "MYM",  # equity indices
    "6E",
    "6B",
    "6J",
    "6A",
    "6C",
    "6S",  # FX
    "ZN",
    "ZB",  # bonds
    "ZC",
    "ZS",
    "ZW",  # grains
    "MBT",
    "MET",  # micro BTC/ETH
    "BTC",
    "ETH",
    "SOL",  # Kraken spot crypto
]

# Smoke-test subset: 1 liquid futures + 1 crypto (covers both data paths)
SMOKE_SYMBOLS = ["MES", "BTC"]

EXPECTED_CSV_COLUMNS = {
    "image_path",
    "label",
    "symbol",
    "breakout_type",
    "breakout_type_ord",
    "session",
    "session_ord",
}

# v8 feature columns that should appear in the CSV
V8_FEATURE_COLUMNS = {
    "asset_class_ord",
    "asset_id_ord",
    "atr_norm",
    "volume_ratio",
    "rsi_14",
    "vwap_distance",
    "spread_norm",
}


# ---------------------------------------------------------------------------
# Test functions
# ---------------------------------------------------------------------------


def test_engine_url() -> str | None:
    """Check ENGINE_DATA_URL is set and reachable."""
    header("Engine Connectivity")

    engine_url = (os.getenv("ENGINE_DATA_URL") or os.getenv("DATA_SERVICE_URL") or "http://data:8000").rstrip("/")

    info(f"ENGINE_DATA_URL = {engine_url}")

    # Warn about the known port-8100 bug
    if ":8100" in engine_url:
        fail(
            f"ENGINE_DATA_URL uses port 8100 — the data service is on port 8050! "
            f"Fix: export ENGINE_DATA_URL={engine_url.replace(':8100', ':8050')}"
        )
        return None

    if ":8050" in engine_url:
        ok("Port 8050 (correct — matches data service mapping)")
    elif "data:8000" in engine_url:
        info("Using Docker internal URL (data:8000) — OK if running inside compose")

    # Try to reach /health
    try:
        import requests

        resp = requests.get(f"{engine_url}/health", timeout=10)
        if resp.status_code == 200:
            ok(f"Engine /health responded OK (HTTP {resp.status_code})")
            try:
                health = resp.json()
                info(f"  status={health.get('status', '?')}, version={health.get('version', '?')}")
            except Exception:
                pass
            return engine_url
        else:
            fail(f"Engine /health returned HTTP {resp.status_code}")
            return None
    except ImportError:
        fail("'requests' package not installed — pip install requests")
        return None
    except Exception as exc:
        fail(f"Cannot reach engine at {engine_url}: {exc}")
        info("Is the data service running? Try: docker compose up -d data")
        info("Or set ENGINE_DATA_URL to the correct Tailscale IP:port")
        return None


def test_bar_loading(engine_url: str | None, days: int = 5) -> dict[str, bool]:
    """Try loading bars for smoke-test symbols."""
    header("Bar Loading")

    results: dict[str, bool] = {}

    if engine_url is None:
        # Try to import load_bars and use fallback sources
        info("Engine unreachable — testing fallback bar sources (db, csv)")

    try:
        from lib.services.training.dataset_generator import load_bars
    except ImportError as exc:
        fail(f"Cannot import dataset_generator: {exc}")
        return results

    for symbol in SMOKE_SYMBOLS:
        t0 = time.monotonic()
        try:
            df = load_bars(symbol, source="engine", days=days)
            elapsed = time.monotonic() - t0

            if df is not None and not df.empty:
                ok(f"{symbol}: {len(df)} bars loaded in {elapsed:.1f}s (range: {df.index.min()} → {df.index.max()})")
                # Validate columns
                expected_cols = {"Open", "High", "Low", "Close", "Volume"}
                actual_cols = set(df.columns)
                # Also accept lowercase
                actual_lower = {c.lower() for c in actual_cols}
                if expected_cols.issubset(actual_cols) or {c.lower() for c in expected_cols}.issubset(actual_lower):
                    ok(f"{symbol}: OHLCV columns present")
                else:
                    warn(f"{symbol}: unexpected columns: {actual_cols}")

                # Check for NaN-heavy data
                nan_pct = df.isnull().mean().mean() * 100
                if nan_pct > 5:
                    warn(f"{symbol}: {nan_pct:.1f}% NaN values in bars")
                else:
                    ok(f"{symbol}: data quality OK ({nan_pct:.2f}% NaN)")

                results[symbol] = True
            else:
                fail(f"{symbol}: load_bars returned empty/None ({elapsed:.1f}s)")
                results[symbol] = False
        except Exception as exc:
            elapsed = time.monotonic() - t0
            fail(f"{symbol}: load_bars raised {type(exc).__name__}: {exc} ({elapsed:.1f}s)")
            results[symbol] = False

    if not any(results.values()):
        info("")
        info("No bars loaded from any source. Common fixes:")
        info("  1. Start the data service: docker compose up -d postgres redis data")
        info("  2. Set ENGINE_DATA_URL: export ENGINE_DATA_URL=http://<tailscale-ip>:8050")
        info("  3. Place CSV bar files in data/bars/<SYMBOL>_1m.csv for offline mode")

    return results


def test_peer_bars(days: int = 5) -> None:
    """Test cross-asset peer bar resolution (v8-B)."""
    header("Cross-Asset Peer Bars (v8-B)")

    try:
        from lib.services.training.dataset_generator import _resolve_peer_tickers
    except ImportError:
        skip("Cannot import _resolve_peer_tickers — skipping")
        return

    for symbol in SMOKE_SYMBOLS:
        try:
            peers = _resolve_peer_tickers(symbol)
            if peers:
                ok(f"{symbol}: {len(peers)} peer tickers resolved: {', '.join(sorted(peers)[:5])}")
                if len(peers) > 5:
                    info(f"  ... and {len(peers) - 5} more")
            else:
                warn(f"{symbol}: no peer tickers found (cross-asset features will be zeros)")
        except Exception as exc:
            warn(f"{symbol}: peer resolution failed: {exc}")


def test_dataset_config() -> None:
    """Validate DatasetConfig defaults match v8 training recipe."""
    header("DatasetConfig Defaults")

    try:
        from lib.services.training.dataset_generator import DatasetConfig
    except ImportError as exc:
        fail(f"Cannot import DatasetConfig: {exc}")
        return

    cfg = DatasetConfig()

    checks = [
        ("breakout_type", cfg.breakout_type, "all"),
        ("orb_session", cfg.orb_session, "all"),
        ("max_samples_per_type_label", cfg.max_samples_per_type_label, 800),
        ("max_samples_per_session_label", cfg.max_samples_per_session_label, 400),
        ("use_parity_renderer", cfg.use_parity_renderer, True),
        ("bars_source", cfg.bars_source, "engine"),
    ]

    for name, actual, expected in checks:
        if actual == expected:
            ok(f"{name} = {actual!r}")
        else:
            warn(f"{name} = {actual!r} (expected {expected!r} for v8)")

    info(f"window_size={cfg.window_size}, step_size={cfg.step_size}, chart_dpi={cfg.chart_dpi}")
    info(f"sl_atr_mult={cfg.sl_atr_mult}, tp1_atr_mult={cfg.tp1_atr_mult}, tp2_atr_mult={cfg.tp2_atr_mult}")


def test_chart_rendering(bars_loaded: dict[str, bool]) -> bool:
    """Generate a single chart image to test the parity renderer."""
    header("Chart Rendering (Parity Renderer)")

    # Pick a symbol that had bars loaded
    symbol = next((s for s, ok_ in bars_loaded.items() if ok_), None)
    if symbol is None:
        skip("No bars available — cannot test chart rendering")
        return False

    try:
        from lib.services.training.dataset_generator import load_bars
    except ImportError:
        fail("Cannot import load_bars")
        return False

    df = load_bars(symbol, source="engine", days=5)
    if df is None or df.empty:
        skip(f"No bars for {symbol} on re-load — cannot test rendering")
        return False

    # Take a 240-bar window
    window_size = min(240, len(df))
    window = df.iloc[:window_size]

    tmpdir = tempfile.mkdtemp(prefix="futures_smoke_chart_")
    out_path = os.path.join(tmpdir, f"{symbol}_smoke_test.png")

    try:
        from lib.services.training.chart_renderer_parity import render_chart
    except ImportError:
        try:
            from lib.services.training.chart_renderer import render_chart
        except ImportError:
            fail("Cannot import any chart renderer")
            shutil.rmtree(tmpdir, ignore_errors=True)
            return False

    t0 = time.monotonic()
    try:
        render_chart(
            bars=window,
            output_path=out_path,
            symbol=symbol,
        )
        elapsed = time.monotonic() - t0

        if os.path.isfile(out_path):
            size_kb = os.path.getsize(out_path) / 1024
            ok(f"Chart rendered: {out_path} ({size_kb:.1f} KB, {elapsed:.2f}s)")
            if size_kb < 1:
                warn(f"Chart file is suspiciously small ({size_kb:.1f} KB)")
            else:
                ok(f"Chart size looks reasonable ({size_kb:.1f} KB)")

            # Verify it's a valid PNG
            with open(out_path, "rb") as f:
                magic = f.read(8)
            if magic[:4] == b"\x89PNG":
                ok("Valid PNG header")
            else:
                warn(f"File does not start with PNG magic bytes: {magic[:8]!r}")

            shutil.rmtree(tmpdir, ignore_errors=True)
            return True
        else:
            fail("Chart file was not created")
            shutil.rmtree(tmpdir, ignore_errors=True)
            return False

    except Exception as exc:
        elapsed = time.monotonic() - t0
        fail(f"Chart rendering failed: {type(exc).__name__}: {exc} ({elapsed:.2f}s)")
        shutil.rmtree(tmpdir, ignore_errors=True)
        return False


def test_mini_dataset(bars_loaded: dict[str, bool]) -> None:
    """Generate a tiny dataset (1 symbol, 3 days) and validate the CSV output."""
    header("Mini Dataset Generation")

    symbol = next((s for s, ok_ in bars_loaded.items() if ok_), None)
    if symbol is None:
        skip("No bars available — cannot test dataset generation")
        return

    tmpdir = tempfile.mkdtemp(prefix="futures_smoke_ds_")
    image_dir = os.path.join(tmpdir, "images")

    try:
        from lib.services.training.dataset_generator import (
            DatasetConfig,
            generate_dataset_for_symbol,
            load_bars,
            load_daily_bars,
        )
    except ImportError as exc:
        fail(f"Cannot import dataset_generator components: {exc}")
        shutil.rmtree(tmpdir, ignore_errors=True)
        return

    cfg = DatasetConfig(
        output_dir=tmpdir,
        image_dir=image_dir,
        bars_source="engine",
        orb_session="us",  # Just US session for speed
        breakout_type="ORB",  # Just ORB for speed
        use_parity_renderer=True,
        skip_existing=False,
    )

    # Load bars
    bars_1m = load_bars(symbol, source="engine", days=5)
    if bars_1m is None or bars_1m.empty:
        skip(f"No bars for {symbol} — cannot test mini dataset")
        shutil.rmtree(tmpdir, ignore_errors=True)
        return

    bars_daily = load_daily_bars(symbol, source="engine")

    info(f"Generating mini dataset for {symbol} ({len(bars_1m)} bars, ORB/US only)...")
    t0 = time.monotonic()

    try:
        rows, stats = generate_dataset_for_symbol(
            symbol=symbol,
            bars_1m=bars_1m,
            bars_daily=bars_daily,
            config=cfg,
        )
        elapsed = time.monotonic() - t0

        ok(f"Generated {len(rows)} rows in {elapsed:.1f}s")
        info(f"  windows={stats.total_windows}, trades={stats.total_trades}, images={stats.total_images}")
        info(f"  labels: {dict(stats.label_distribution)}")
        info(f"  render_failures={stats.render_failures}, skipped={stats.skipped_existing}")

        if len(rows) == 0:
            warn("Zero rows generated — this may be normal for very short bar history")
            warn("Full run with 120 days should produce ample data")
        else:
            # Check CSV row structure
            import pandas as pd

            df = pd.DataFrame(rows)
            actual_cols = set(df.columns)

            missing_required = EXPECTED_CSV_COLUMNS - actual_cols
            if missing_required:
                fail(f"Missing required CSV columns: {missing_required}")
            else:
                ok(f"All required CSV columns present ({len(EXPECTED_CSV_COLUMNS)})")

            # Check v8 feature columns
            v8_present = V8_FEATURE_COLUMNS & actual_cols
            v8_missing = V8_FEATURE_COLUMNS - actual_cols
            if v8_present:
                ok(f"v8 feature columns found: {len(v8_present)}/{len(V8_FEATURE_COLUMNS)}")
            if v8_missing:
                warn(f"v8 feature columns missing: {v8_missing}")
                info("These may be added by _build_row() — check if the full pipeline populates them")

            # Check label values
            if "label" in df.columns:
                labels = df["label"].unique()
                valid_labels = {"long", "short", "no_trade"}
                unexpected = set(labels) - valid_labels
                if unexpected:
                    warn(f"Unexpected label values: {unexpected}")
                else:
                    ok(f"Label values valid: {sorted(labels)}")

            # Check images exist
            if "image_path" in df.columns:
                sample_paths = df["image_path"].head(3).tolist()
                existing = sum(1 for p in sample_paths if os.path.isfile(p))
                ok(f"Sample images: {existing}/{len(sample_paths)} exist on disk")

            # Check breakout_type
            if "breakout_type" in df.columns:
                btypes = df["breakout_type"].unique()
                ok(f"Breakout types: {sorted(btypes)}")

            info(f"Total CSV columns: {len(actual_cols)}")

        if stats.errors:
            for err in stats.errors[:3]:
                warn(f"Error: {err}")

    except Exception as exc:
        elapsed = time.monotonic() - t0
        fail(f"Mini dataset generation failed: {type(exc).__name__}: {exc} ({elapsed:.1f}s)")
        import traceback

        traceback.print_exc()
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_full_symbols_list() -> None:
    """Verify the full 25-symbol list is importable and consistent."""
    header("Symbol List Validation")

    try:
        from lib.services.training.trainer_server import DEFAULT_SYMBOLS
    except ImportError:
        # Fall back to our hardcoded list
        info("Cannot import DEFAULT_SYMBOLS from trainer_server — using local list")
        DEFAULT_SYMBOLS = ALL_25_SYMBOLS

    ok(f"Symbol count: {len(DEFAULT_SYMBOLS)}")

    if set(DEFAULT_SYMBOLS) == set(ALL_25_SYMBOLS):
        ok("Symbol list matches expected ALL_25")
    else:
        extra = set(DEFAULT_SYMBOLS) - set(ALL_25_SYMBOLS)
        missing = set(ALL_25_SYMBOLS) - set(DEFAULT_SYMBOLS)
        if extra:
            warn(f"Extra symbols in DEFAULT_SYMBOLS: {extra}")
        if missing:
            warn(f"Missing symbols from DEFAULT_SYMBOLS: {missing}")

    # Check asset class coverage
    try:
        from lib.core.asset_registry import get_asset_by_ticker

        def _get_asset_class(sym: str) -> str:
            asset = get_asset_by_ticker(sym)
            return asset.asset_class.value if asset else "unknown"

    except ImportError:
        try:
            from lib.analysis.breakout_cnn import get_asset_class_id

            def _get_asset_class(sym: str) -> str:  # type: ignore[misc]
                return str(get_asset_class_id(sym))

        except ImportError:
            skip("Cannot import asset class resolver — skipping class coverage check")
            return

    classes: dict[str, list[str]] = {}
    for sym in DEFAULT_SYMBOLS:
        try:
            cls = _get_asset_class(sym)
            classes.setdefault(str(cls), []).append(sym)
        except Exception:
            classes.setdefault("unknown", []).append(sym)

    for cls, syms in sorted(classes.items()):
        ok(f"  {cls}: {len(syms)} symbols ({', '.join(syms[:5])}{'...' if len(syms) > 5 else ''})")


def test_model_importable() -> None:
    """Quick check that the CNN model class and training function are importable."""
    header("Model Import Check")

    try:
        from lib.analysis.breakout_cnn import HybridBreakoutCNN  # noqa: F401

        ok("HybridBreakoutCNN importable")
    except ImportError as exc:
        fail(f"Cannot import HybridBreakoutCNN: {exc}")

    try:
        from lib.analysis.breakout_cnn import train_model

        if train_model is not None:
            ok("train_model available (torch installed)")
        else:
            warn("train_model is None — torch not available (GPU training won't work)")
    except ImportError as exc:
        fail(f"Cannot import train_model: {exc}")

    try:
        from lib.services.training.dataset_generator import split_dataset  # noqa: F401

        ok("split_dataset importable")
    except ImportError as exc:
        warn(f"Cannot import split_dataset: {exc}")


def test_feature_contract() -> None:
    """Validate the feature contract JSON if present."""
    header("Feature Contract")

    contract_paths = [
        _PROJECT_ROOT / "models" / "feature_contract.json",
        _PROJECT_ROOT / "feature_contract.json",
    ]

    contract_path = next((p for p in contract_paths if p.is_file()), None)
    if contract_path is None:
        info("No feature_contract.json found — will be generated during training")
        return

    try:
        with open(contract_path) as f:
            contract = json.load(f)

        version = contract.get("version") or contract.get("contract_version")
        ok(f"Feature contract found: {contract_path.name} (v{version})")

        n_features = contract.get("num_tabular_features") or contract.get("n_tabular")
        if n_features:
            info(f"  Tabular features: {n_features}")
            if n_features >= 37:
                ok(f"  v8 feature count ({n_features} >= 37)")
            else:
                warn(f"  Feature count {n_features} < 37 (v8 expects 37+)")

        if "asset_class_lookup" in contract:
            ok(f"  asset_class_lookup: {len(contract['asset_class_lookup'])} entries")
        if "asset_id_lookup" in contract:
            ok(f"  asset_id_lookup: {len(contract['asset_id_lookup'])} entries")

    except Exception as exc:
        warn(f"Cannot read feature contract: {exc}")


def estimate_full_run() -> None:
    """Estimate time and disk space for full v8 dataset generation."""
    header("Full Run Estimate")

    n_symbols = len(ALL_25_SYMBOLS)
    days_back = 120
    n_breakout_types = 13
    n_sessions = 9

    # Rough estimates based on v6 run data
    # v6 did ~20K samples with 22 symbols, 90 days, ORB-only
    # v8 with all 13 types and all 9 sessions should be 5-8x more
    # But caps (800 per type-label, 400 per session-label) limit it
    est_samples_low = 40_000
    est_samples_high = 90_000
    est_disk_gb_low = est_samples_low * 50 / 1024 / 1024  # ~50KB per image
    est_disk_gb_high = est_samples_high * 50 / 1024 / 1024

    info(f"Symbols: {n_symbols}")
    info(f"Days back: {days_back}")
    info(f"Breakout types: {n_breakout_types} (all)")
    info(f"Sessions: {n_sessions} (all)")
    info(f"Estimated samples: {est_samples_low:,} – {est_samples_high:,}")
    info(f"Estimated disk: {est_disk_gb_low:.1f} – {est_disk_gb_high:.1f} GB")
    info("Estimated time: 2-6 hours (depends on bar loading speed + GPU)")
    info("")
    info("Caps applied: max_per_type_label=800, max_per_session_label=400")
    info("These prevent ORB/US from dominating the dataset.")

    # Check disk space
    try:
        import shutil as _shutil

        usage = _shutil.disk_usage(_PROJECT_ROOT)
        free_gb = usage.free / (1024**3)
        if free_gb < est_disk_gb_high:
            warn(f"Only {free_gb:.1f} GB free — may not be enough for full dataset ({est_disk_gb_high:.1f} GB est.)")
        else:
            ok(f"Disk space: {free_gb:.1f} GB free (need ~{est_disk_gb_high:.1f} GB)")
    except Exception:
        info("Could not check disk space")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Smoke test dataset generation pipeline before full v8 run",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode — skip chart rendering and mini dataset generation",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging from dataset_generator",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=5,
        help="Days of history to load for smoke test (default: 5)",
    )
    parser.add_argument(
        "--symbol",
        type=str,
        default=None,
        help="Override smoke-test symbol (default: MES,BTC)",
    )
    args = parser.parse_args()

    if args.verbose:
        import logging

        logging.basicConfig(level=logging.DEBUG, format="%(levelname)s %(name)s: %(message)s")

    if args.symbol:
        global SMOKE_SYMBOLS
        SMOKE_SYMBOLS = [s.strip() for s in args.symbol.split(",")]

    print()
    print(f"{_BOLD}╔══════════════════════════════════════════════════════╗{_NC}")
    print(f"{_BOLD}║   Futures v8 — Dataset Generation Smoke Test        ║{_NC}")
    print(f"{_BOLD}╚══════════════════════════════════════════════════════╝{_NC}")
    print()
    info(f"Project root: {_PROJECT_ROOT}")
    info(f"Smoke symbols: {', '.join(SMOKE_SYMBOLS)}")
    info(f"Days: {args.days}")
    info(f"Mode: {'quick' if args.quick else 'full'}")

    t_start = time.monotonic()

    # --- Tests ---
    engine_url = test_engine_url()
    bars_loaded = test_bar_loading(engine_url, days=args.days)
    test_peer_bars(days=args.days)
    test_dataset_config()
    test_full_symbols_list()
    test_model_importable()
    test_feature_contract()

    if not args.quick:
        test_chart_rendering(bars_loaded)
        test_mini_dataset(bars_loaded)

    estimate_full_run()

    # --- Summary ---
    elapsed = time.monotonic() - t_start
    header("Summary")
    print()
    print(f"  {_GREEN}✓ Passed:   {_PASS}{_NC}")
    print(f"  {_YELLOW}⚠ Warnings: {_WARN}{_NC}")
    print(f"  {_RED}✗ Failed:   {_FAIL}{_NC}")
    if _SKIP:
        print(f"  {_DIM}⊘ Skipped:  {_SKIP}{_NC}")
    print(f"  ⏱  Elapsed: {elapsed:.1f}s")
    print()

    if _FAIL > 0:
        print(f"  {_RED}{_BOLD}✗ {_FAIL} critical failure(s) — fix before running full dataset generation.{_NC}")
        print()
        if not any(bars_loaded.values()):
            print(f"  {_YELLOW}Most likely fix:{_NC}")
            print("    1. Start data service: docker compose up -d postgres redis data")
            print("    2. Set ENGINE_DATA_URL: export ENGINE_DATA_URL=http://<server-tailscale-ip>:8050")
            print("    3. Re-run: python scripts/smoke_test_dataset.py")
            print()
        return 1
    elif _WARN > 0:
        print(f"  {_YELLOW}{_BOLD}⚠ All critical checks passed, but {_WARN} warning(s) to review.{_NC}")
        print(f"  {_DIM}Dataset generation will likely work but may produce fewer samples.{_NC}")
        print()
        return 2
    else:
        print(f"  {_GREEN}{_BOLD}✓ All checks passed — safe to run full v8 dataset generation!{_NC}")
        print()
        print(f"  {_CYAN}Next step:{_NC}")
        print("    python -m lib.services.training.dataset_generator \\")
        print(f"      --symbols {','.join(ALL_25_SYMBOLS)} \\")
        print("      --days 120 \\")
        print("      --breakout-type all \\")
        print("      --orb-session all")
        print()
        print(f"  {_DIM}Or trigger via the trainer API:{_NC}")
        print("    curl -X POST http://<trainer>:8200/train \\")
        print('      -H "Content-Type: application/json" \\')
        print('      -d \'{"days_back": 120, "epochs": 80, "patience": 15}\'')
        print()
        return 0


if __name__ == "__main__":
    sys.exit(main())
