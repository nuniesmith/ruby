"""
Dataset Generator — Orchestrates Chart Rendering + Auto-Labeling for CNN Training
==================================================================================
Combines the ORB simulator (auto-labeler) and chart renderer to produce a
complete labeled dataset of Ruby-style chart images suitable for training
the HybridBreakoutCNN model.

Pipeline:
  1. Load 1-minute bar data for target symbols (via Massive client or cache).
  2. Slide windows across the data, simulating ORB trades per window.
  3. Render a Ruby-style chart snapshot for each window.
  4. Write a CSV manifest (labels.csv) with image paths, labels, and tabular
     features ready for BreakoutDataset.

The generator is designed to run as an off-hours scheduled job (e.g. 02:30 ET)
via the engine scheduler, but can also be invoked manually from the CLI.

Public API:
    from dataset_generator import (
        generate_dataset,
        generate_dataset_for_symbol,
        DatasetConfig,
        DatasetStats,
    )

    stats = generate_dataset(
        symbols=["MGC", "MES", "MNQ"],
        days_back=90,
    )
    # stats.total_images → 18432
    # stats.csv_path → "dataset/labels.csv"

Dependencies:
  - rb_simulator (auto-labeling)
  - chart_renderer (image generation)
  - pandas, numpy (already in project)
  - Massive client or cached bar data (for historical bars)

Design:
  - Pure orchestration — delegates to rb_simulator and chart_renderer.
  - Resumable: skips images that already exist on disk (by filename).
  - Thread-safe: each symbol can be processed independently.
  - Produces balanced datasets by capping over-represented labels.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime
from datetime import time as dt_time_type
from typing import TYPE_CHECKING, Any
from zoneinfo import ZoneInfo

import numpy as np

# requests is optional (only needed for _load_bars_from_kraken and
# _load_bars_from_engine).  Importing it at module level makes it patchable
# in tests via `patch("lib.services.training.dataset_generator.requests", ...)`.
try:
    import requests  # type: ignore[import-untyped]
except ImportError:
    requests = None  # type: ignore[assignment]

if TYPE_CHECKING:
    from collections.abc import Sequence

    from lib.services.training.rb_simulator import ORBSimResult
import pandas as pd

logger = logging.getLogger("analysis.dataset_generator")

_EST = ZoneInfo("America/New_York")

# Sentinel object used to mark that a pre-computation was attempted but
# produced no usable result (either the function returned None or raised
# an exception).  This lets _build_row() distinguish "never attempted"
# (attribute missing → fall back to on-demand compute) from "attempted
# and failed" (attribute is _PRECOMPUTE_FAILED → skip, use defaults).
# Without this, a failed pre-compute leaves the attribute unset, and
# _build_row() falls back to recomputing the expensive operation for
# every single row (~5000×), causing the pipeline to appear stuck.
_PRECOMPUTE_FAILED = object()


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class DatasetConfig:
    """Configuration for the dataset generation pipeline."""

    # Output paths
    output_dir: str = "dataset"
    image_dir: str = "dataset/images"
    csv_filename: str = "labels.csv"

    # Window parameters (passed to rb_simulator.simulate_batch)
    window_size: int = 240  # 4 hours of 1-min bars
    step_size: int = 30  # 30-minute steps between windows
    min_window_bars: int = 60  # minimum bars to attempt simulation

    # ORB simulation bracket config
    sl_atr_mult: float = 1.5
    tp1_atr_mult: float = 2.0
    tp2_atr_mult: float = 3.0
    max_hold_bars: int = 120
    atr_period: int = 14

    # ORB session selection for dataset generation.
    # Controls which opening-range windows are simulated per bar history.
    #
    # Supported values:
    #   "us"        — US Equity Open 09:30–10:00 ET (default)
    #   "london"    — London Open 03:00–03:30 ET
    #   "all"       — All 9 sessions across the full Globex day (recommended
    #                 for maximum dataset diversity and coverage)
    #   "frankfurt" — Frankfurt/Xetra 03:00–03:30 ET
    #   "tokyo"     — Tokyo/TSE 19:00–19:30 ET (overnight)
    #   "shanghai"  — Shanghai/HK 21:00–21:30 ET (overnight)
    #   "cme"       — CME Globex re-open 18:00–18:30 ET (overnight)
    #   "sydney"    — Sydney/ASX 18:30–19:00 ET (overnight)
    #   "london_ny" — London-NY Crossover 08:00–08:30 ET
    #   "cme_settle"— CME Settlement 14:00–14:30 ET
    orb_session: str = "all"

    # Chart rendering
    chart_dpi: int = 150  # lower than live for disk space savings
    chart_figsize: tuple[float, float] = (12, 8)

    # Dataset balancing
    max_samples_per_label: int = 0  # 0 = no cap (global cap across all types/sessions)

    # Per-(label, breakout_type) cap — prevents high-frequency types (e.g. ORB)
    # from swamping rarer types (e.g. Monthly, Weekly) when --breakout-type=all.
    # v8 training recipe default: 800.
    max_samples_per_type_label: int = 800

    # Per-(label, session) cap — ensures overnight sessions (Sydney, Tokyo, Shanghai)
    # are not under-represented vs the primary London / US sessions.
    # v8 training recipe default: 400.
    max_samples_per_session_label: int = 400

    include_no_trade: bool = False  # include no_trade samples (usually not useful)

    # Resumability
    skip_existing: bool = True  # skip images that already exist on disk

    # Data source
    # "engine" — ask the engine/data service HTTP API (GET /bars/{symbol});
    #             the engine handles Redis → Postgres → external API internally
    #             and is the preferred source when the trainer is a separate
    #             machine (e.g. GPU server) without direct DB/Redis access.
    # "db"     — read from Postgres/SQLite historical_bars table directly
    # "cache"  — read from Redis directly (short-lived, limited history)
    # "massive"— call Massive REST API directly (requires MASSIVE_API_KEY)
    # "csv"    — read from local CSV files (offline/test use)
    bars_source: str = "engine"  # "engine" | "db" | "cache" | "massive" | "csv"
    csv_bars_dir: str = "data/bars"  # only used if bars_source == "csv"

    # Renderer selection
    # "mplfinance" — original chart renderer (chart_renderer.py)
    #                Rich styling with anti-aliased candles, badges, legends.
    #                Produces visually different images than the Pillow renderer —
    #                use "parity" for CNN training to keep train/inference consistent.
    # "parity"     — Pillow renderer (chart_renderer_parity.py)
    #                Pixel-precise integer coordinate math, dark theme, 224×224.
    #                RECOMMENDED for CNN training to keep images consistent.
    use_parity_renderer: bool = True

    # Breakout type(s) to simulate.
    # Controls which simulator is used and tags every CSV row with the
    # correct ``breakout_type`` / ``breakout_type_ord`` for the CNN.
    #
    # Supported values (case-insensitive):
    #   "ORB"            — Opening Range Breakout (default, sliding windows)
    #   "PrevDay"        — Previous Day High/Low breakout (one per calendar day)
    #   "InitialBalance" — First 60-min RTH range breakout (one per day)
    #   "Consolidation"  — Auto-detected tight N-bar range breakout (multiple/day)
    #   "Weekly"         — Prior week's high/low breakout (one per week)
    #   "Monthly"        — Prior month's high/low breakout (one per month)
    #   "Asian"          — Asian session range (19:00–02:00 ET) breakout
    #   "BollingerSqueeze" — BB inside KC squeeze → expansion breakout
    #   "ValueArea"      — Prior session VAH/VAL breakout (volume profile)
    #   "InsideDay"      — Inside day pattern breakout
    #   "GapRejection"   — Overnight gap fill / rejection breakout
    #   "PivotPoints"    — Classic floor pivot R1/S1 breakout
    #   "Fibonacci"      — 38.2%–61.8% Fib retracement zone breakout
    #   "all"            — Run all 13 simulators; combines their results
    #
    # When set to a specific type the corresponding ``simulate_batch_*``
    # function is called.  "ORB" still uses the existing sliding-window
    # ``simulate_batch`` for backward compatibility.
    breakout_type: str = "all"

    # Parallelism
    max_workers: int = 1  # symbols processed in parallel (1 = sequential)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class DatasetStats:
    """Statistics from a dataset generation run."""

    total_windows: int = 0
    total_trades: int = 0
    total_images: int = 0
    skipped_existing: int = 0
    render_failures: int = 0
    label_distribution: dict[str, int] = field(default_factory=dict)
    symbols_processed: list[str] = field(default_factory=list)
    # Symbols that were dropped because bar data was insufficient even after a
    # deeper fill attempt.  Callers (e.g. trainer_server) should remove these
    # from any subsequent generate_dataset / train calls.
    dropped_symbols: list[str] = field(default_factory=list)
    csv_path: str = ""
    duration_seconds: float = 0.0
    errors: list[str] = field(default_factory=list)

    # Per-(label, breakout_type) sample counters — used for max_samples_per_type_label cap.
    # Key format: "{label}__{breakout_type}", e.g. "good_long__ORB"
    _type_label_counts: dict[str, int] = field(default_factory=dict, repr=False)

    # Per-(label, session) sample counters — used for max_samples_per_session_label cap.
    # Key format: "{label}__{session_key}", e.g. "good_long__london"
    _session_label_counts: dict[str, int] = field(default_factory=dict, repr=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_windows": self.total_windows,
            "total_trades": self.total_trades,
            "total_images": self.total_images,
            "skipped_existing": self.skipped_existing,
            "render_failures": self.render_failures,
            "label_distribution": self.label_distribution,
            "symbols_processed": self.symbols_processed,
            "dropped_symbols": self.dropped_symbols,
            "csv_path": self.csv_path,
            "duration_seconds": round(self.duration_seconds, 1),
            "errors": self.errors[:20],  # cap error list
        }

    def summary(self) -> str:
        ld = ", ".join(f"{k}={v}" for k, v in sorted(self.label_distribution.items()))
        return (
            f"Dataset: {self.total_images} images from {len(self.symbols_processed)} symbols | "
            f"Trades: {self.total_trades}/{self.total_windows} windows | "
            f"Labels: [{ld}] | "
            f"Skipped: {self.skipped_existing}, Failures: {self.render_failures} | "
            f"Time: {self.duration_seconds:.0f}s | CSV: {self.csv_path}"
        )


# ---------------------------------------------------------------------------
# Bar data loading helpers
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Symbol → Yahoo-style ticker mapping
# ---------------------------------------------------------------------------
# Engine HTTP API URL — used by _load_bars_from_engine().
# Reads from ENGINE_DATA_URL env var (set by docker-compose / supervisor).
# Falls back to DATA_SERVICE_URL for backwards compatibility, then to the
# default docker-compose service name.
# ---------------------------------------------------------------------------

_ENGINE_DATA_URL: str = (os.getenv("ENGINE_DATA_URL") or os.getenv("DATA_SERVICE_URL") or "http://data:8000").rstrip(
    "/"
)

# Shared secret for inter-service authentication.
# When set, sent as X-API-Key header on all engine HTTP requests.
_API_KEY: str = os.getenv("API_KEY", "").strip()

# Timeout (seconds) for a single GET /bars/{symbol} call to the engine.
# The engine now returns immediately with existing data + filling=true when a
# background fill is in progress, so a 60-second timeout is usually enough for
# the initial call.  Set ENGINE_BARS_TIMEOUT to a higher value if the engine is
# far away on a slow link (e.g. cross-continent Tailscale tunnel).
_ENGINE_BARS_TIMEOUT: int = int(os.getenv("ENGINE_BARS_TIMEOUT", "60"))

# When the engine responds with filling=true (background fill in progress),
# the loader will poll /bars/{symbol}/fill/status and then re-fetch once the
# fill is done.  These two knobs control the polling behaviour:
#
#   ENGINE_FILL_POLL_INTERVAL  — seconds between status polls (default 10)
#   ENGINE_FILL_POLL_MAX_WAIT  — maximum seconds to wait for the fill before
#                                giving up and returning the partial data we
#                                already have (default 300 = 5 minutes).
_ENGINE_FILL_POLL_INTERVAL: int = int(os.getenv("ENGINE_FILL_POLL_INTERVAL", "10"))
_ENGINE_FILL_POLL_MAX_WAIT: int = int(os.getenv("ENGINE_FILL_POLL_MAX_WAIT", "300"))


# ---------------------------------------------------------------------------
# The backtest/dataset scripts use short names like "MGC", "MES", "MNQ"
# but the cache layer and Massive API expect Yahoo-style tickers like
# "MGC=F", "ES=F", etc.  This mapping bridges the two.

_SYMBOL_TO_TICKER: dict[str, str] = {
    # ── Micro metals ──────────────────────────────────────────────────────
    "MGC": "MGC=F",  # Micro Gold
    "SIL": "SIL=F",  # Micro Silver
    "MHG": "MHG=F",  # Micro Copper
    # ── Micro energy ──────────────────────────────────────────────────────
    "MCL": "MCL=F",  # Micro Crude Oil
    "MNG": "NG=F",  # Micro Natural Gas (data_ticker = NG=F, same bar data)
    # ── Energy (full-size) ────────────────────────────────────────────────
    "NG": "NG=F",  # Natural Gas (NYMEX)
    "CL": "CL=F",  # Crude Oil (full-size alias)
    # ── Micro equity index ────────────────────────────────────────────────
    "MES": "MES=F",  # Micro S&P 500
    "MNQ": "MNQ=F",  # Micro Nasdaq-100
    "M2K": "M2K=F",  # Micro Russell 2000
    "MYM": "MYM=F",  # Micro Dow Jones
    # ── Micro crypto ──────────────────────────────────────────────────────
    "MBT": "MBT=F",  # Micro Bitcoin (CME)
    "MET": "MET=F",  # Micro Ether (CME)
    # ── Micro FX futures (CME) ────────────────────────────────────────────
    "M6E": "6E=F",  # Micro Euro FX (data_ticker = 6E=F)
    "M6B": "6B=F",  # Micro British Pound (data_ticker = 6B=F)
    "M6J": "6J=F",  # Micro Japanese Yen (data_ticker = 6J=F)
    # ── FX futures (CME standard) ─────────────────────────────────────────
    "6E": "6E=F",  # Euro FX
    "6B": "6B=F",  # British Pound
    "6J": "6J=F",  # Japanese Yen
    "6A": "6A=F",  # Australian Dollar
    "6C": "6C=F",  # Canadian Dollar
    "6S": "6S=F",  # Swiss Franc
    # ── Interest rate futures (CBOT) ─────────────────────────────────────
    "ZN": "ZN=F",  # 10-Year T-Note
    "ZB": "ZB=F",  # 30-Year T-Bond
    "ZF": "ZF=F",  # 5-Year T-Note
    "ZT": "ZT=F",  # 2-Year T-Note
    # ── Agricultural futures (CBOT) ──────────────────────────────────────
    "ZC": "ZC=F",  # Corn
    "ZS": "ZS=F",  # Soybeans
    "ZW": "ZW=F",  # Wheat
    "ZL": "ZL=F",  # Soybean Oil
    "ZM": "ZM=F",  # Soybean Meal
    # ── Full-size contracts (data source aliases) ─────────────────────────
    "GC": "GC=F",
    "ES": "ES=F",
    "NQ": "NQ=F",
    "SI": "SI=F",
    "HG": "HG=F",
    "YM": "YM=F",
    "RTY": "RTY=F",
    # ── Crypto (CME full-size futures) ────────────────────────────────────
    "BTC_CME": "BTC=F",  # CME Bitcoin (full-size) — disambiguated alias
    "ETH_CME": "ETH=F",  # CME Ether (full-size) — disambiguated alias
    # ── Kraken spot crypto ────────────────────────────────────────────────
    # Kraken tickers use the "KRAKEN:" prefix convention.
    # These are 24/7 spot pairs — no =F suffix, no contract roll.
    "KRAKEN:XBTUSD": "KRAKEN:XBTUSD",  # Bitcoin / USD
    "KRAKEN:ETHUSD": "KRAKEN:ETHUSD",  # Ether / USD
    "KRAKEN:SOLUSD": "KRAKEN:SOLUSD",  # Solana / USD
    "KRAKEN:AVAXUSD": "KRAKEN:AVAXUSD",  # Avalanche / USD
    "KRAKEN:LINKUSD": "KRAKEN:LINKUSD",  # Chainlink / USD
    "KRAKEN:POLUSD": "KRAKEN:POLUSD",  # Polygon (POL) / USD
    "KRAKEN:DOTUSD": "KRAKEN:DOTUSD",  # Polkadot / USD
    "KRAKEN:ADAUSD": "KRAKEN:ADAUSD",  # Cardano / USD
    "KRAKEN:XRPUSD": "KRAKEN:XRPUSD",  # Ripple / USD
    # Short-form aliases — preferred for training symbol lists.
    # BTC/ETH/SOL/etc. route directly to Kraken spot (24/7 data).
    # MBT/MET continue to route to their CME futures tickers above.
    "BTC": "KRAKEN:XBTUSD",  # Bitcoin spot via Kraken
    "ETH": "KRAKEN:ETHUSD",  # Ether spot via Kraken
    "SOL": "KRAKEN:SOLUSD",  # Solana spot via Kraken
    "AVAX": "KRAKEN:AVAXUSD",  # Avalanche spot via Kraken
    "LINK": "KRAKEN:LINKUSD",  # Chainlink spot via Kraken
    "POL": "KRAKEN:POLUSD",  # Polygon (POL) spot via Kraken
    "DOT": "KRAKEN:DOTUSD",  # Polkadot spot via Kraken
    "ADA": "KRAKEN:ADAUSD",  # Cardano spot via Kraken
    "XRP": "KRAKEN:XRPUSD",  # Ripple spot via Kraken
    # Kraken pair-style aliases (no prefix)
    "XBTUSD": "KRAKEN:XBTUSD",
    "ETHUSD": "KRAKEN:ETHUSD",
    "SOLUSD": "KRAKEN:SOLUSD",
    "AVAXUSD": "KRAKEN:AVAXUSD",
    "LINKUSD": "KRAKEN:LINKUSD",
    "POLUSD": "KRAKEN:POLUSD",
    "DOTUSD": "KRAKEN:DOTUSD",
    "ADAUSD": "KRAKEN:ADAUSD",
    "XRPUSD": "KRAKEN:XRPUSD",
}

# Set of all Kraken ticker prefixes/patterns — used by loaders to route
# Kraken symbols to the dedicated Kraken bar loader instead of Massive/yfinance.
_KRAKEN_TICKERS: frozenset[str] = frozenset(t for t in _SYMBOL_TO_TICKER.values() if t.startswith("KRAKEN:"))


def _is_kraken_symbol(symbol: str) -> bool:
    """Return True if *symbol* is a Kraken spot crypto pair."""
    return symbol.upper().startswith("KRAKEN:") or _resolve_ticker(symbol).startswith("KRAKEN:")


def _load_bars_from_engine(symbol: str, days: int = 90) -> pd.DataFrame | None:
    """Load 1-minute bars from the engine/data service HTTP API.

    Calls ``GET /bars/{symbol}?interval=1m&days_back={days}&auto_fill=true``
    on the engine.  The engine handles the full resolution chain internally:
    Redis cache → Postgres DB → Massive/Kraken external API → auto-backfill.

    This is the **preferred** source when the trainer runs on a separate
    machine (GPU server) without direct access to Redis or Postgres — it only
    needs a network path to the engine's HTTP port.

    The symbol is sent as-is; the engine's bars router resolves short names
    (e.g. "MGC") and Yahoo-style tickers (e.g. "MGC=F") identically.

    Returns a DataFrame with OHLCV columns and a UTC DatetimeIndex, or None
    if the engine is unreachable, returns an error, or has no data.
    """
    try:
        import requests as _requests
    except ImportError:
        logger.debug("requests not available — cannot load bars from engine")
        return None

    url = f"{_ENGINE_DATA_URL}/bars/{symbol}"
    params: dict[str, str | int] = {
        "interval": "1m",
        "days_back": days,
        "auto_fill": "true",
    }

    # Include API key header for inter-service auth (if configured)
    headers: dict[str, str] = {}
    if _API_KEY:
        headers["X-API-Key"] = _API_KEY

    try:
        resp = _requests.get(url, params=params, headers=headers, timeout=_ENGINE_BARS_TIMEOUT)
        if resp.status_code == 404:
            logger.debug("Engine has no bars for %s (404)", symbol)
            return None
        if resp.status_code != 200:
            logger.debug(
                "Engine /bars/%s returned HTTP %s: %s",
                symbol,
                resp.status_code,
                resp.text[:200],
            )
            return None

        payload = resp.json()

        # ── Fill-poll loop ─────────────────────────────────────────────────
        # The engine now fires background fills asynchronously instead of
        # blocking.  When filling=true the response contains current
        # (possibly partial / stale) data.  We wait up to
        # ENGINE_FILL_POLL_MAX_WAIT seconds for the fill to finish and then
        # re-fetch the bars so the dataset gets the full history.
        if payload.get("filling") and _ENGINE_FILL_POLL_MAX_WAIT > 0:
            fill_status_url = payload.get("fill_status_url")
            if fill_status_url:
                fill_status_url = f"{_ENGINE_DATA_URL}{fill_status_url}"
            else:
                fill_status_url = f"{_ENGINE_DATA_URL}/bars/{symbol}/fill/status"

            waited = 0
            logger.info(
                "Engine fill in progress for %s — polling %s (max wait %ds, interval %ds)",
                symbol,
                fill_status_url,
                _ENGINE_FILL_POLL_MAX_WAIT,
                _ENGINE_FILL_POLL_INTERVAL,
            )
            while waited < _ENGINE_FILL_POLL_MAX_WAIT:
                time.sleep(_ENGINE_FILL_POLL_INTERVAL)
                waited += _ENGINE_FILL_POLL_INTERVAL
                try:
                    status_resp = _requests.get(
                        fill_status_url,
                        headers=headers,
                        timeout=_ENGINE_BARS_TIMEOUT,
                    )
                    if status_resp.status_code == 200:
                        status_data = status_resp.json()
                        fill_status = status_data.get("status", "unknown")
                        logger.debug(
                            "Fill status for %s after %ds: %s (bars_added=%s)",
                            symbol,
                            waited,
                            fill_status,
                            status_data.get("bars_added", "?"),
                        )
                        if fill_status in ("complete", "failed", "no_fill"):
                            if fill_status == "failed":
                                logger.warning(
                                    "Engine fill failed for %s: %s",
                                    symbol,
                                    status_data.get("error"),
                                )
                            break
                except Exception as poll_exc:
                    logger.debug("Fill status poll failed for %s: %s", symbol, poll_exc)

            # Re-fetch bars now that the fill has (hopefully) completed
            logger.info(
                "Re-fetching bars for %s after fill wait (%ds)",
                symbol,
                waited,
            )
            try:
                refetch_resp = _requests.get(
                    url,
                    params={**params, "auto_fill": "false"},  # don't re-trigger fill
                    headers=headers,
                    timeout=_ENGINE_BARS_TIMEOUT,
                )
                if refetch_resp.status_code == 200:
                    payload = refetch_resp.json()
            except Exception as refetch_exc:
                logger.debug(
                    "Re-fetch after fill failed for %s: %s — using original payload",
                    symbol,
                    refetch_exc,
                )

        split_data = payload.get("data")
        if not split_data:
            logger.debug("Engine /bars/%s: empty data payload", symbol)
            return None

        df = pd.DataFrame(**split_data)
        if df.empty:
            logger.debug("Engine /bars/%s: reconstructed DataFrame is empty", symbol)
            return None

        # Ensure DatetimeIndex is UTC-aware
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index, utc=True)
        elif df.index.tz is None:
            df.index = df.index.tz_localize("UTC")

        # Normalise column names to title-case (Open/High/Low/Close/Volume)
        col_map = {c: c.title() for c in df.columns if c.lower() in ("open", "high", "low", "close", "volume")}
        if col_map:
            df = df.rename(columns=col_map)

        bar_count = payload.get("bar_count", len(df))
        logger.debug(
            "Loaded %d bars for %s from engine (%s, filled=%s, filling=%s)",
            bar_count,
            symbol,
            _ENGINE_DATA_URL,
            payload.get("filled", False),
            payload.get("filling", False),
        )
        return df

    except Exception as exc:
        logger.debug("Engine bar load failed for %s: %s", symbol, exc)
        return None


def _request_deeper_fill(symbol: str, days: int) -> None:
    """Ask the engine to run a blocking deeper fill for *symbol*.

    Calls ``POST /bars/{symbol}/fill`` with the requested ``days_back`` so
    the engine fetches further back in history from Massive / yfinance.
    The call is best-effort — any network or engine error is logged at DEBUG
    level and silently swallowed so callers can always continue.

    This is used by the low-bar retry path in ``generate_dataset`` to try to
    bring under-stocked symbols up to the minimum bar threshold before
    deciding whether to drop them from the dataset run.
    """
    try:
        import requests as _requests
    except ImportError:
        logger.debug("requests not available — cannot request deeper fill for %s", symbol)
        return

    url = f"{_ENGINE_DATA_URL}/bars/{symbol}/fill"
    headers: dict[str, str] = {}
    if _API_KEY:
        headers["X-API-Key"] = _API_KEY

    try:
        resp = _requests.post(
            url,
            json={"days_back": days, "interval": "1m"},
            headers=headers,
            # Filling can be slow for large windows — use a generous timeout.
            # The /fill endpoint blocks until complete.
            timeout=max(_ENGINE_BARS_TIMEOUT * 4, 240),
        )
        if resp.status_code == 200:
            result = resp.json()
            bars_added = result.get("bars_added", 0)
            logger.info(
                "  Deeper fill for %s complete: +%d bars (total: %d)",
                symbol,
                bars_added,
                result.get("bars_after", "?"),
            )
        else:
            logger.debug(
                "Deeper fill request for %s returned HTTP %s: %s",
                symbol,
                resp.status_code,
                resp.text[:200],
            )
    except Exception as exc:
        logger.debug("Deeper fill request failed for %s: %s", symbol, exc)


def _load_bars_from_db(symbol: str, days: int = 90) -> pd.DataFrame | None:
    """Load 1-minute bars directly from the Postgres/SQLite historical_bars table.

    This is the **preferred** source for CNN dataset generation because it:
      - Contains a deep, continuous history populated by the nightly backfill.
      - Is always available (no network dependency at generation time).
      - Supports up to 365 days of 1-minute bars with Massive.

    Falls back gracefully to None if the table doesn't exist yet.
    """
    try:
        from lib.services.engine.backfill import get_stored_bars

        ticker = _resolve_ticker(symbol)
        df = get_stored_bars(ticker, days_back=days, interval="1m")
        if df is not None and not df.empty and len(df) > 50:
            logger.debug(
                "Loaded %d bars for %s (ticker=%s) from historical_bars DB",
                len(df),
                symbol,
                ticker,
            )
            return df
    except ImportError:
        logger.debug("backfill module not available — skipping DB source")
    except Exception as exc:
        logger.debug("DB load failed for %s: %s", symbol, exc)

    return None


def _resolve_ticker(symbol: str) -> str:
    """Convert a short symbol like 'MGC' to a Yahoo-style ticker like 'MGC=F'.

    If the symbol already looks like a Yahoo ticker (contains '='), returns as-is.
    Falls back to appending '=F' if not found in the explicit map.
    """
    if "=" in symbol:
        return symbol
    return _SYMBOL_TO_TICKER.get(symbol.upper(), f"{symbol.upper()}=F")


def _load_bars_from_cache(symbol: str, days: int = 90) -> pd.DataFrame | None:
    """Attempt to load 1-minute bars from Redis cache.

    Uses the standard ``get_data()`` cache layer (which stores bars under
    hashed ``futures:*`` keys) with proper Yahoo-style ticker resolution.
    Also checks legacy ``engine:bars_1m:*`` keys as a fallback.
    """
    # --- Primary path: use get_data() which handles hashed cache keys ---
    try:
        import importlib as _il

        _cache_mod = _il.import_module("cache")
        get_data = _cache_mod.get_data

        ticker = _resolve_ticker(symbol)
        # Map days to a period string for get_data()
        if days <= 5:
            period = f"{days}d"
        elif days <= 30:
            period = "1mo"
        elif days <= 90:
            period = "3mo"
        else:
            period = "6mo"

        df = get_data(ticker, interval="1m", period=period)
        if df is not None and not df.empty and len(df) > 50:
            logger.debug(
                "Loaded %d bars for %s (ticker=%s) from cache via get_data()",
                len(df),
                symbol,
                ticker,
            )
            return df
    except ImportError:
        logger.debug("Cache module not available")
    except Exception as exc:
        logger.debug("get_data() failed for %s: %s", symbol, exc)

    # --- Fallback: check legacy engine:bars_1m:* keys ---
    try:
        import importlib as _il2

        _cache_mod2 = _il2.import_module("cache")
        cache_get = _cache_mod2.cache_get
    except (ImportError, AttributeError):
        return None

    import io

    ticker = _resolve_ticker(symbol)
    for key_pattern in [
        f"engine:bars_1m_hist:{ticker}",
        f"engine:bars_1m:{ticker}",
        f"engine:bars_1m_hist:{symbol}",
        f"engine:bars_1m:{symbol}",
    ]:
        try:
            raw = cache_get(key_pattern)
            if raw:
                raw_str = raw.decode("utf-8") if isinstance(raw, bytes) else raw
                df = pd.read_json(io.StringIO(raw_str))
                if not df.empty and len(df) > 100:
                    logger.debug("Loaded %d bars for %s from cache key %s", len(df), symbol, key_pattern)
                    return df
        except Exception as exc:
            logger.debug("Cache load failed for %s key %s: %s", symbol, key_pattern, exc)

    return None


def _load_bars_from_csv(symbol: str, csv_dir: str = "data/bars") -> pd.DataFrame | None:
    """Load 1-minute bars from a local CSV file.

    Expected filename pattern: ``{csv_dir}/{symbol}_1m.csv``
    Expected columns: Date/Datetime, Open, High, Low, Close, Volume
    """
    csv_path = os.path.join(csv_dir, f"{symbol}_1m.csv")
    if not os.path.isfile(csv_path):
        # Also try lowercase
        csv_path = os.path.join(csv_dir, f"{symbol.lower()}_1m.csv")
    if not os.path.isfile(csv_path):
        logger.debug("No CSV file found for %s in %s", symbol, csv_dir)
        return None

    try:
        df = pd.read_csv(csv_path, parse_dates=True, index_col=0)
        # Ensure proper column names
        col_map = {}
        for col in df.columns:
            cl = col.lower().strip()
            if cl in ("open", "o"):
                col_map[col] = "Open"
            elif cl in ("high", "h"):
                col_map[col] = "High"
            elif cl in ("low", "l"):
                col_map[col] = "Low"
            elif cl in ("close", "c"):
                col_map[col] = "Close"
            elif cl in ("volume", "vol", "v"):
                col_map[col] = "Volume"
        if col_map:
            df = df.rename(columns=col_map)

        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        df.index.name = "Date"

        logger.debug("Loaded %d bars for %s from %s", len(df), symbol, csv_path)
        return df
    except Exception as exc:
        logger.warning("Failed to load CSV for %s: %s", symbol, exc)
        return None


def _load_bars_from_massive(symbol: str, days: int = 90) -> pd.DataFrame | None:
    """Load 1-minute bars via the Massive REST API.

    Uses ``MassiveDataProvider.get_aggs()`` with Yahoo-style ticker resolution.
    Massive only stores a limited window of 1-minute data, so for longer
    histories the effective day count may be clamped by the API.
    """
    try:
        from lib.integrations.massive_client import get_massive_provider
    except ImportError:
        logger.debug("Massive client not available")
        return None

    try:
        provider = get_massive_provider()
        if not provider.is_available:
            logger.debug("Massive provider not available (no API key?)")
            return None

        ticker = _resolve_ticker(symbol)

        # Map days to a period string that get_aggs() understands
        if days <= 1:
            period = "1d"
        elif days <= 5:
            period = "5d"
        elif days <= 10:
            period = "10d"
        elif days <= 15:
            period = "15d"
        elif days <= 30:
            period = "1mo"
        elif days <= 90:
            period = "3mo"
        else:
            period = "6mo"

        df = provider.get_aggs(ticker, interval="1m", period=period)
        if df is not None and not df.empty:
            logger.debug("Loaded %d bars for %s (ticker=%s) from Massive", len(df), symbol, ticker)
            return df
    except Exception as exc:
        logger.debug("Massive load failed for %s: %s", symbol, exc)

    return None


def _load_bars_from_kraken(symbol: str, days: int = 90) -> pd.DataFrame | None:
    """Load 1-minute bars for a Kraken spot crypto pair.

    Kraken pairs use the ``KRAKEN:`` prefix (e.g. ``KRAKEN:XBTUSD``).
    Data is fetched via the Kraken public REST API (no API key required for
    OHLC history) and normalised to the same DataFrame format used by all
    other bar loaders: DatetimeIndex + [Open, High, Low, Close, Volume].

    The Kraken OHLC endpoint returns up to 720 candles per request at any
    interval, so for ``days > 0.5`` we page backwards in time using the
    ``since`` parameter until we have enough history or hit the API limit.

    Falls back gracefully to ``None`` if the ``requests`` package is missing
    or the API returns an error — the ``load_bars`` fallback chain will then
    try the next source (Massive, cache, CSV).

    Args:
        symbol: Kraken pair in any supported form:
                ``"KRAKEN:XBTUSD"``, ``"XBTUSD"``, etc.
        days: Number of calendar days of history to request.

    Returns:
        DataFrame with DatetimeIndex (UTC) and columns
        ``[Open, High, Low, Close, Volume]``, or ``None`` on failure.
    """
    if requests is None:
        logger.debug("requests not installed — cannot load Kraken bars")
        return None

    # Resolve to canonical Kraken pair (strip "KRAKEN:" prefix for the API)
    ticker = _resolve_ticker(symbol)
    pair = ticker[7:] if ticker.upper().startswith("KRAKEN:") else ticker

    # Kraken public OHLC endpoint — 1-minute bars
    _KRAKEN_OHLC_URL = "https://api.kraken.com/0/public/OHLC"
    _INTERVAL = 1  # minutes

    import time as _time

    since_ts = int(_time.time()) - days * 86400

    all_rows: list[dict[str, float]] = []
    max_pages = 20  # guard against runaway pagination

    for _page in range(max_pages):
        try:
            req_params: dict[str, str | int] = {"pair": pair, "interval": _INTERVAL, "since": since_ts}
            resp = requests.get(
                _KRAKEN_OHLC_URL,
                params=req_params,
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            logger.debug("Kraken OHLC request failed for %s: %s", pair, exc)
            break

        if data.get("error"):
            logger.debug("Kraken API error for %s: %s", pair, data["error"])
            break

        result = data.get("result", {})
        # The result dict has one key = pair name (may differ from input)
        pair_key = next((k for k in result if k != "last"), None)
        if pair_key is None:
            break

        candles = result[pair_key]
        if not candles:
            break

        for c in candles:
            # Kraken OHLC: [time, open, high, low, close, vwap, volume, count]
            try:
                all_rows.append(
                    {
                        "ts": int(c[0]),
                        "Open": float(c[1]),
                        "High": float(c[2]),
                        "Low": float(c[3]),
                        "Close": float(c[4]),
                        "Volume": float(c[6]),
                    }
                )
            except (IndexError, ValueError, TypeError):
                continue

        # Kraken returns the ``last`` timestamp for pagination
        last_ts = result.get("last")
        if last_ts is None or int(last_ts) <= since_ts:
            break  # no more pages
        # If we've covered the full requested range, stop
        if candles and int(candles[-1][0]) >= int(_time.time()) - 60:
            break
        since_ts = int(last_ts)

    if not all_rows:
        logger.debug("No Kraken bars returned for %s", pair)
        return None

    df = pd.DataFrame(all_rows)
    df = df.drop_duplicates(subset=["ts"]).sort_values("ts")
    df.index = pd.to_datetime(df["ts"], unit="s", utc=True)
    df.index.name = "datetime"
    df = df[["Open", "High", "Low", "Close", "Volume"]]
    # Drop rows with zero OHLC (occasional Kraken data gaps)
    df = df[(df["Open"] > 0) & (df["High"] > 0) & (df["Low"] > 0) & (df["Close"] > 0)]

    logger.debug(
        "Loaded %d Kraken bars for %s (%s, pair=%s)",
        len(df),
        symbol,
        ticker,
        pair,
    )
    return pd.DataFrame(df) if len(df) > 0 else None


def load_bars(
    symbol: str,
    source: str = "engine",
    days: int = 90,
    csv_dir: str = "data/bars",
) -> pd.DataFrame | None:
    """Load 1-minute bars via the engine data client with CSV fallback.

    Primary path: EngineDataClient → GET /bars/{symbol} on the engine/data
    service. The engine resolves the full hierarchy internally:
        Redis → Postgres → Massive (futures) / Kraken REST (crypto)

    This single path replaces all the previous source-specific branches.
    The ``source`` parameter is accepted for backward compatibility but only
    ``"engine"`` and ``"csv"`` are actively used; all other values resolve
    via the engine as well.

    Fallback chain (only when engine is truly unreachable / offline):
        engine → legacy loaders → csv

    IMPORTANT: The legacy fallback chain (db, massive, cache) is only
    triggered when the EngineDataClient cannot contact the engine at all
    (connection error, timeout, import failure).  If the engine *is*
    reachable but simply has no data for a symbol, we stop here and return
    None — we do NOT fall through to the local DB or local Massive client.
    This prevents spurious "no such table: historical_bars" and
    "MASSIVE_API_KEY not set" log noise on the GPU trainer machine, which
    has no local database and no API keys (all data comes from the engine).

    Args:
        symbol: Short symbol ("MGC"), Yahoo-style ticker ("MGC=F"), or
                Kraken alias ("BTC", "KRAKEN:XBTUSD").
        source: Primary source hint — "engine" (default), "massive",
                "kraken", "db", "cache", or "csv".
        days:   Days of 1-minute bar history to request.
        csv_dir: Directory for CSV files (only used if source=="csv" or all else fails).

    Returns:
        DataFrame with OHLCV columns and UTC DatetimeIndex, or None.
    """
    # --- CSV-only path (offline/test) ------------------------------------
    if source == "csv":
        return _load_bars_from_csv(symbol, csv_dir)

    # --- Primary path: EngineDataClient ----------------------------------
    # Use client.get_bars() as the primary call (keeps the public mock
    # surface that tests patch).  When it returns a non-empty DataFrame
    # we're done.  When it returns None we need to distinguish two cases:
    #
    #   A) Engine is reachable but has no data for this symbol → return
    #      None immediately; do NOT fall through to local loaders.  The
    #      trainer machine has no local DB and no API keys, so those paths
    #      would only produce log noise ("no such table: historical_bars",
    #      "MASSIVE_API_KEY not set").
    #
    #   B) Engine is genuinely unreachable (connection refused, DNS error,
    #      import failure, etc.) → try the legacy local fallback chain so
    #      the trainer still works in offline / local-dev mode.
    #
    # We resolve A vs B by calling client.is_available() only in the
    # None-return case (one extra lightweight round-trip when data is
    # missing, zero extra round-trips on the happy path).
    try:
        from lib.services.data.engine_data_client import get_client

        client = get_client()
        df = client.get_bars(symbol, interval="1m", days_back=days, auto_fill=True)

        if df is not None and not df.empty:
            logger.debug("load_bars: %d bars for %s via EngineDataClient", len(df), symbol)
            return df

        # get_bars() returned None — check whether the engine is actually up.
        # is_available() does a fast /health probe (3 s timeout).
        if hasattr(client, "is_available") and client.is_available():
            # Engine is up but get_bars() returned nothing.  Before giving
            # up, try the Postgres-backed store directly via /api/data/bars.
            # This covers the case where the full hierarchy (Redis → PG → API)
            # returned nothing but the sync service has bars in Postgres.
            try:
                df = client.get_stored_bars(symbol, interval="1m", days_back=days)
                if df is not None and not df.empty:
                    logger.debug("load_bars: %d bars for %s via stored bars (Postgres)", len(df), symbol)
                    return df
            except Exception as exc:
                logger.debug("get_stored_bars failed for %s: %s", symbol, exc)

            # Engine is up but has no data for this symbol.  Skip local
            # fallbacks — they would only log noise on the trainer machine.
            logger.debug(
                "load_bars: engine reachable but returned no bars for %s — skipping local fallbacks",
                symbol,
            )
            return None
        # Engine appears to be down — fall through to legacy loaders below.

    except Exception as exc:
        logger.debug("EngineDataClient failed for %s: %s", symbol, exc)

    # --- Legacy fallback path (engine unreachable / offline only) --------
    # Only runs when the EngineDataClient raised an exception (connection
    # refused, DNS failure, import error, etc.).  Keeps the trainer usable
    # in local dev without a running data service.
    _is_kraken = _is_kraken_symbol(symbol)

    if _is_kraken:
        for loader_fn in [
            lambda: _load_bars_from_kraken(symbol, days),
            lambda: _load_bars_from_csv(symbol, csv_dir),
        ]:
            try:
                df = loader_fn()
                if df is not None and not df.empty:
                    return df
            except Exception:
                continue
    else:
        for loader_fn in [
            lambda: _load_bars_from_engine(symbol, days),
            lambda: _load_bars_from_db(symbol, days),
            lambda: _load_bars_from_massive(symbol, days),
            lambda: _load_bars_from_csv(symbol, csv_dir),
        ]:
            try:
                df = loader_fn()
                if df is not None and not df.empty:
                    return df
            except Exception:
                continue

    logger.warning("load_bars: no bars available for %s from any source", symbol)
    return None


def load_daily_bars(
    symbol: str,
    source: str = "engine",
    csv_dir: str = "data/bars",
) -> pd.DataFrame | None:
    """Load daily bars — resampled from 1-minute data via the engine.

    Used for NR7 detection and daily range context in the dataset generator.
    """
    # Try dedicated daily CSV first (offline use)
    daily_csv = os.path.join(csv_dir, f"{symbol}_daily.csv")
    if os.path.isfile(daily_csv):
        try:
            df = pd.read_csv(daily_csv, parse_dates=True, index_col=0)
            if not df.empty:
                return df
        except Exception:
            pass

    # Primary: engine daily bars endpoint
    try:
        from lib.services.data.engine_data_client import get_client

        client = get_client()
        df = client.get_daily_bars(symbol, days_back=365)
        if df is not None and not df.empty:
            return df
    except Exception as exc:
        logger.debug("Daily bars via EngineDataClient failed for %s: %s", symbol, exc)

    # Fallback: resample from 1-minute data
    bars_1m = load_bars(symbol, source=source, csv_dir=csv_dir, days=90)
    if bars_1m is not None and len(bars_1m) > 100:
        try:
            _resampled = (
                bars_1m.resample("1D")
                .agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"})
                .dropna()
            )
            daily = pd.DataFrame(_resampled) if not isinstance(_resampled, pd.DataFrame) else _resampled
            if len(daily) >= 7:
                return daily
        except Exception as exc:
            logger.debug("Daily resample failed for %s: %s", symbol, exc)

    return None


# ---------------------------------------------------------------------------
# Core generation logic
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Per-session BracketConfig parameters
# ---------------------------------------------------------------------------
# Maps session_key → (or_start, or_end, pm_end) as time objects (ET).
# These mirror the ORBSession definitions in engine/orb.py but are
# duplicated here to keep the dataset generator independent of the engine.
_SESSION_BRACKET_PARAMS: dict[str, tuple[dt_time_type, dt_time_type, dt_time_type]] = {}  # filled below


def _get_session_bracket_params() -> dict[str, Any]:
    """Lazily build the session → (or_start, or_end, pm_end) mapping.

    Returns a dict keyed by session_key with tuples of
    ``(or_start, or_end, pm_end)`` as ``datetime.time`` objects in ET.
    Importing ``datetime.time`` here avoids a module-level circular import.
    """
    from datetime import time as dt_time

    return {
        # CME Globex re-open  18:00–18:30 ET  (overnight, wraps_midnight)
        "cme": (dt_time(18, 0), dt_time(18, 30), dt_time(18, 0)),
        # Sydney / ASX  18:30–19:00 ET  (overnight, wraps_midnight)
        "sydney": (dt_time(18, 30), dt_time(19, 0), dt_time(18, 30)),
        # Tokyo / TSE  19:00–19:30 ET  (overnight, wraps_midnight)
        "tokyo": (dt_time(19, 0), dt_time(19, 30), dt_time(19, 0)),
        # Shanghai / HK  21:00–21:30 ET  (overnight, wraps_midnight)
        "shanghai": (dt_time(21, 0), dt_time(21, 30), dt_time(21, 0)),
        # Frankfurt / Xetra  03:00–03:30 ET
        "frankfurt": (dt_time(3, 0), dt_time(3, 30), dt_time(3, 0)),
        # London Open  03:00–03:30 ET  (primary session)
        "london": (dt_time(3, 0), dt_time(3, 30), dt_time(3, 0)),
        # London-NY Crossover  08:00–08:30 ET
        "london_ny": (dt_time(8, 0), dt_time(8, 30), dt_time(8, 0)),
        # US Equity Open  09:30–10:00 ET
        "us": (dt_time(9, 30), dt_time(10, 0), dt_time(8, 20)),
        # CME Settlement  14:00–14:30 ET
        "cme_settle": (dt_time(14, 0), dt_time(14, 30), dt_time(8, 20)),
    }


# Ordered list of all session keys for "all" mode (chronological Globex-day order)
_ALL_SESSION_KEYS: list[str] = [
    "cme",
    "sydney",
    "tokyo",
    "shanghai",
    "frankfurt",
    "london",
    "london_ny",
    "us",
    "cme_settle",
]


def _bracket_configs_for_session(
    cfg: DatasetConfig,
) -> list[tuple[str, Any]]:
    """Return ``(session_key, BracketConfig)`` pairs based on ``cfg.orb_session``.

    Supported values for ``cfg.orb_session``:

    - ``"us"``        → US Equity Open only (OR 09:30–10:00 ET)
    - ``"london"``    → London Open only (OR 03:00–03:30 ET)
    - ``"all"``       → All 9 sessions across the full Globex day
    - ``"frankfurt"`` → Frankfurt/Xetra only (OR 03:00–03:30 ET)
    - ``"tokyo"``     → Tokyo/TSE only (OR 19:00–19:30 ET)
    - ``"shanghai"``  → Shanghai/HK only (OR 21:00–21:30 ET)
    - ``"cme"``       → CME Globex re-open only (OR 18:00–18:30 ET)
    - ``"sydney"``    → Sydney/ASX only (OR 18:30–19:00 ET)
    - ``"london_ny"`` → London-NY Crossover only (OR 08:00–08:30 ET)
    - ``"cme_settle"``→ CME Settlement only (OR 14:00–14:30 ET)

    Any unknown value falls back to ``"us"``.
    """
    from lib.services.training.rb_simulator import BracketConfig

    session_params = _get_session_bracket_params()

    def _make_cfg(key: str) -> Any:
        or_start, or_end, pm_end = session_params[key]
        return BracketConfig(
            sl_atr_mult=cfg.sl_atr_mult,
            tp1_atr_mult=cfg.tp1_atr_mult,
            tp2_atr_mult=cfg.tp2_atr_mult,
            max_hold_bars=cfg.max_hold_bars,
            atr_period=cfg.atr_period,
            or_start=or_start,
            or_end=or_end,
            pm_end=pm_end,
        )

    session = cfg.orb_session.lower().strip()

    if session == "all":
        # Full Globex-day coverage — all 9 sessions
        return [(key, _make_cfg(key)) for key in _ALL_SESSION_KEYS]
    elif session in session_params:
        return [(session, _make_cfg(session))]
    else:
        logger.warning(
            "Unknown orb_session '%s' in DatasetConfig — falling back to 'us'",
            cfg.orb_session,
        )
        return [("us", _make_cfg("us"))]


def _run_simulators_for_breakout_type(
    breakout_type_str: str,
    bars_1m: pd.DataFrame,
    symbol: str,
    cfg: DatasetConfig,
    bars_daily: pd.DataFrame | None,
) -> list[Any]:
    """Dispatch to the correct simulator(s) based on *breakout_type_str*.

    Returns a flat list of :class:`~rb_simulator.ORBSimResult` objects with
    ``_session_key`` and ``_breakout_type`` already set on each result.

    Args:
        breakout_type_str: One of ``"ORB"``, ``"PrevDay"``,
                           ``"InitialBalance"``, ``"Consolidation"``,
                           ``"Weekly"``, ``"Monthly"``, ``"Asian"``,
                           ``"BollingerSqueeze"``, ``"ValueArea"``,
                           ``"InsideDay"``, ``"GapRejection"``,
                           ``"PivotPoints"``, ``"Fibonacci"``,
                           or ``"all"`` (case-insensitive).
        bars_1m:           Full 1-min OHLCV history.
        symbol:            Instrument ticker.
        cfg:               DatasetConfig (used for bracket params and session).
        bars_daily:        Optional daily bars for NR7.

    Returns:
        Combined list of simulation results tagged with breakout type.
    """
    from lib.core.breakout_types import BreakoutType
    from lib.services.training.rb_simulator import (
        BracketConfig,
        simulate_batch,
        simulate_batch_asian,
        simulate_batch_bollinger_squeeze,
        simulate_batch_consolidation,
        simulate_batch_fibonacci,
        simulate_batch_gap_rejection,
        simulate_batch_ib,
        simulate_batch_inside_day,
        simulate_batch_monthly,
        simulate_batch_pivot_points,
        simulate_batch_prev_day,
        simulate_batch_value_area,
        simulate_batch_weekly,
    )

    _bt_str = breakout_type_str.strip().lower()

    # Determine which BreakoutType(s) to run
    if _bt_str == "all":
        _types_to_run = list(BreakoutType)
    else:
        _name_map = {bt.name.lower(): bt for bt in BreakoutType}
        _bt = _name_map.get(_bt_str, BreakoutType.ORB)
        _types_to_run = [_bt]

    # Base BracketConfig from DatasetConfig scalars (used for ORB and as
    # the default for the other simulators which override or_start/or_end).
    base_bracket = BracketConfig(
        sl_atr_mult=cfg.sl_atr_mult,
        tp1_atr_mult=cfg.tp1_atr_mult,
        tp2_atr_mult=cfg.tp2_atr_mult,
        max_hold_bars=cfg.max_hold_bars,
        atr_period=cfg.atr_period,
    )

    combined: list[Any] = []

    for _bt in _types_to_run:
        if _bt == BreakoutType.ORB:
            # ORB: sliding-window simulation across all configured sessions
            session_configs = _bracket_configs_for_session(cfg)
            for session_key, bracket_cfg in session_configs:
                session_label = f"{symbol}/{session_key}"
                logger.info(
                    "Simulating ORB trades for %s (OR %s–%s, %d bars)...",
                    session_label,
                    bracket_cfg.or_start.strftime("%H:%M"),
                    bracket_cfg.or_end.strftime("%H:%M"),
                    len(bars_1m),
                )
                results = simulate_batch(
                    bars_1m=bars_1m,
                    symbol=symbol,
                    config=bracket_cfg,
                    bars_daily=bars_daily,
                    window_size=cfg.window_size,
                    step_size=cfg.step_size,
                    min_window_bars=cfg.min_window_bars,
                )
                for r in results:
                    r._session_key = session_key
                    r._breakout_type = BreakoutType.ORB
                combined.extend(results)
                logger.info(
                    "%s: %d windows → %d trades",
                    session_label,
                    len(results),
                    sum(1 for r in results if r.is_trade),
                )

        elif _bt == BreakoutType.PrevDay:
            logger.info(
                "Simulating PrevDay trades for %s (%d bars)...",
                symbol,
                len(bars_1m),
            )
            results = simulate_batch_prev_day(
                bars_1m=bars_1m,
                symbol=symbol,
                config=base_bracket,
                bars_daily=bars_daily,
            )
            for r in results:
                r._session_key = r._session_key or "us"
                r._breakout_type = BreakoutType.PrevDay
            combined.extend(results)
            logger.info(
                "PrevDay %s: %d days → %d trades",
                symbol,
                len(results),
                sum(1 for r in results if r.is_trade),
            )

        elif _bt == BreakoutType.InitialBalance:
            logger.info(
                "Simulating InitialBalance trades for %s (%d bars)...",
                symbol,
                len(bars_1m),
            )
            results = simulate_batch_ib(
                bars_1m=bars_1m,
                symbol=symbol,
                config=base_bracket,
                bars_daily=bars_daily,
            )
            for r in results:
                r._session_key = r._session_key or "us"
                r._breakout_type = BreakoutType.InitialBalance
            combined.extend(results)
            logger.info(
                "IB %s: %d days → %d trades",
                symbol,
                len(results),
                sum(1 for r in results if r.is_trade),
            )

        elif _bt == BreakoutType.Consolidation:
            logger.info(
                "Simulating Consolidation trades for %s (%d bars)...",
                symbol,
                len(bars_1m),
            )
            results = simulate_batch_consolidation(
                bars_1m=bars_1m,
                symbol=symbol,
                config=base_bracket,
                bars_daily=bars_daily,
            )
            for r in results:
                r._session_key = r._session_key or "us"
                r._breakout_type = BreakoutType.Consolidation
            combined.extend(results)
            logger.info(
                "Consolidation %s: %d boxes → %d trades",
                symbol,
                len(results),
                sum(1 for r in results if r.is_trade),
            )

        elif _bt == BreakoutType.Weekly:
            logger.info(
                "Simulating Weekly trades for %s (%d bars)...",
                symbol,
                len(bars_1m),
            )
            results = simulate_batch_weekly(
                bars_1m=bars_1m,
                symbol=symbol,
                config=base_bracket,
                bars_daily=bars_daily,
            )
            for r in results:
                r._session_key = r._session_key or "us"
                r._breakout_type = BreakoutType.Weekly
            combined.extend(results)
            logger.info(
                "Weekly %s: %d weeks → %d trades",
                symbol,
                len(results),
                sum(1 for r in results if r.is_trade),
            )

        elif _bt == BreakoutType.Monthly:
            logger.info(
                "Simulating Monthly trades for %s (%d bars)...",
                symbol,
                len(bars_1m),
            )
            results = simulate_batch_monthly(
                bars_1m=bars_1m,
                symbol=symbol,
                config=base_bracket,
                bars_daily=bars_daily,
            )
            for r in results:
                r._session_key = r._session_key or "us"
                r._breakout_type = BreakoutType.Monthly
            combined.extend(results)
            logger.info(
                "Monthly %s: %d months → %d trades",
                symbol,
                len(results),
                sum(1 for r in results if r.is_trade),
            )

        elif _bt == BreakoutType.Asian:
            logger.info(
                "Simulating Asian session trades for %s (%d bars)...",
                symbol,
                len(bars_1m),
            )
            results = simulate_batch_asian(
                bars_1m=bars_1m,
                symbol=symbol,
                config=base_bracket,
                bars_daily=bars_daily,
            )
            for r in results:
                r._session_key = r._session_key or "london"
                r._breakout_type = BreakoutType.Asian
            combined.extend(results)
            logger.info(
                "Asian %s: %d days → %d trades",
                symbol,
                len(results),
                sum(1 for r in results if r.is_trade),
            )

        elif _bt == BreakoutType.BollingerSqueeze:
            logger.info(
                "Simulating BollingerSqueeze trades for %s (%d bars)...",
                symbol,
                len(bars_1m),
            )
            results = simulate_batch_bollinger_squeeze(
                bars_1m=bars_1m,
                symbol=symbol,
                config=base_bracket,
                bars_daily=bars_daily,
            )
            for r in results:
                r._session_key = r._session_key or "us"
                r._breakout_type = BreakoutType.BollingerSqueeze
            combined.extend(results)
            logger.info(
                "BollingerSqueeze %s: %d squeezes → %d trades",
                symbol,
                len(results),
                sum(1 for r in results if r.is_trade),
            )

        elif _bt == BreakoutType.ValueArea:
            logger.info(
                "Simulating ValueArea trades for %s (%d bars)...",
                symbol,
                len(bars_1m),
            )
            results = simulate_batch_value_area(
                bars_1m=bars_1m,
                symbol=symbol,
                config=base_bracket,
                bars_daily=bars_daily,
            )
            for r in results:
                r._session_key = r._session_key or "us"
                r._breakout_type = BreakoutType.ValueArea
            combined.extend(results)
            logger.info(
                "ValueArea %s: %d days → %d trades",
                symbol,
                len(results),
                sum(1 for r in results if r.is_trade),
            )

        elif _bt == BreakoutType.InsideDay:
            logger.info(
                "Simulating InsideDay trades for %s (%d bars)...",
                symbol,
                len(bars_1m),
            )
            results = simulate_batch_inside_day(
                bars_1m=bars_1m,
                symbol=symbol,
                config=base_bracket,
                bars_daily=bars_daily,
            )
            for r in results:
                r._session_key = r._session_key or "us"
                r._breakout_type = BreakoutType.InsideDay
            combined.extend(results)
            logger.info(
                "InsideDay %s: %d days → %d trades",
                symbol,
                len(results),
                sum(1 for r in results if r.is_trade),
            )

        elif _bt == BreakoutType.GapRejection:
            logger.info(
                "Simulating GapRejection trades for %s (%d bars)...",
                symbol,
                len(bars_1m),
            )
            results = simulate_batch_gap_rejection(
                bars_1m=bars_1m,
                symbol=symbol,
                config=base_bracket,
                bars_daily=bars_daily,
            )
            for r in results:
                r._session_key = r._session_key or "us"
                r._breakout_type = BreakoutType.GapRejection
            combined.extend(results)
            logger.info(
                "GapRejection %s: %d days → %d trades",
                symbol,
                len(results),
                sum(1 for r in results if r.is_trade),
            )

        elif _bt == BreakoutType.PivotPoints:
            logger.info(
                "Simulating PivotPoints trades for %s (%d bars)...",
                symbol,
                len(bars_1m),
            )
            results = simulate_batch_pivot_points(
                bars_1m=bars_1m,
                symbol=symbol,
                config=base_bracket,
                bars_daily=bars_daily,
            )
            for r in results:
                r._session_key = r._session_key or "us"
                r._breakout_type = BreakoutType.PivotPoints
            combined.extend(results)
            logger.info(
                "PivotPoints %s: %d days → %d trades",
                symbol,
                len(results),
                sum(1 for r in results if r.is_trade),
            )

        elif _bt == BreakoutType.Fibonacci:
            logger.info(
                "Simulating Fibonacci trades for %s (%d bars)...",
                symbol,
                len(bars_1m),
            )
            results = simulate_batch_fibonacci(
                bars_1m=bars_1m,
                symbol=symbol,
                config=base_bracket,
                bars_daily=bars_daily,
            )
            for r in results:
                r._session_key = r._session_key or "us"
                r._breakout_type = BreakoutType.Fibonacci
            combined.extend(results)
            logger.info(
                "Fibonacci %s: %d days → %d trades",
                symbol,
                len(results),
                sum(1 for r in results if r.is_trade),
            )

    return combined


def generate_dataset_for_symbol(
    symbol: str,
    bars_1m: pd.DataFrame,
    bars_daily: pd.DataFrame | None = None,
    config: DatasetConfig | None = None,
    bars_by_ticker: dict[str, pd.DataFrame] | None = None,
) -> tuple[list[dict[str, Any]], DatasetStats]:
    """Generate labeled chart images for a single symbol.

    This is the workhorse function.  It:
      1. Dispatches to the correct simulator(s) based on
         ``config.breakout_type`` (ORB / PrevDay / InitialBalance /
         Consolidation / all).
      2. For each result that is a trade (or no_trade if enabled), renders
         a Ruby-style chart snapshot.
      3. Collects rows for the CSV manifest.

    When ``config.orb_session`` is ``"all"`` and ``config.breakout_type``
    is ``"ORB"`` (or ``"all"``), the function runs ORB simulation for all
    9 Globex sessions, producing training data from each.

    Args:
        symbol: Instrument symbol.
        bars_1m: 1-minute OHLCV bars.
        bars_daily: Daily bars for NR7 (optional).
        config: DatasetConfig.
        bars_by_ticker: Dict mapping ticker → 1-min bars for peer assets.
                        Used by ``_build_row()`` to compute v8-B cross-asset
                        correlation features.  If None, those features fall
                        back to neutral 0.5 defaults.

    Returns:
        (rows, stats) where rows is a list of dicts for the CSV, and
        stats is a DatasetStats for this symbol.
    """
    cfg = config or DatasetConfig()
    stats = DatasetStats()
    stats.symbols_processed.append(symbol)
    rows: list[dict[str, Any]] = []

    # Dispatch to the correct simulator(s)
    all_sim_results = _run_simulators_for_breakout_type(
        breakout_type_str=cfg.breakout_type,
        bars_1m=bars_1m,
        symbol=symbol,
        cfg=cfg,
        bars_daily=bars_daily,
    )

    stats.total_windows = len(all_sim_results)
    stats.total_trades = sum(1 for r in all_sim_results if r.is_trade)

    # ── Pre-compute crypto momentum score once for this symbol ────────────
    # compute_all_crypto_momentum() uses a single shared _all_cache keyed by
    # wall-clock time (TTL 5 min).  All 25 symbols in a training run therefore
    # share one Kraken REST fetch instead of firing 25 simultaneous requests
    # that immediately trigger EGeneral:Too many requests.
    # The score is attached to every result so _build_row() reads a plain
    # float attribute with zero network activity.
    _crypto_mom_score = 0.5
    try:
        from lib.analysis.crypto_momentum import (
            compute_all_crypto_momentum as _compute_all_cm,
        )
        from lib.analysis.crypto_momentum import (
            crypto_momentum_to_tabular as _cm_to_tabular,
        )

        _all_scores = _cm_to_tabular(_compute_all_cm())
        # _all_scores is {futures_symbol: float} in [-1, +1]; map to [0, 1]
        _raw = _all_scores.get(symbol, 0.0)
        _crypto_mom_score = max(0.0, min(1.0, (_raw + 1.0) / 2.0))
    except Exception:
        pass

    # ── Attach context data to each result for v7/v8 feature computation ──
    # _build_row() reads these private attributes to compute:
    #   - v7 features [18-23]: daily bias, weekly range, monthly trend
    #     (requires _daily_bars)
    #   - v7.1 features [26-27]: atr_trend, volume_trend
    #     (requires _bars_1m)
    #   - v8-B features [28-30]: cross-asset correlation
    #     (requires _bars_by_ticker)
    #   - v8-C features [31-36]: asset fingerprint
    #     (requires _daily_bars + _bars_1m)
    _bars_by_ticker_safe = bars_by_ticker or {}
    for r in all_sim_results:
        # Daily bars — for bias analyzer, fingerprint, and NR7
        if bars_daily is not None and not bars_daily.empty:
            r._daily_bars = bars_daily

        # 1-minute bars window — for atr_trend, volume_trend, fingerprint.
        # Attach the full bars so _build_row() can slice as needed;
        # for features like atr_trend/volume_trend the last N bars before
        # the breakout are used, so the full series is appropriate.
        if bars_1m is not None and not bars_1m.empty:
            # Use the window slice if provenance is available, otherwise
            # attach the full bars (features use tail lookbacks anyway).
            _offset = getattr(r, "_window_offset", -1)
            _wsize = getattr(r, "_window_size", 0) or cfg.window_size
            if _offset >= 0:
                _end = min(_offset + _wsize, len(bars_1m))
                r._bars_1m = bars_1m.iloc[_offset:_end]
            else:
                r._bars_1m = bars_1m

        # Peer bars dict — for cross-asset correlation features.
        # Always include the signal asset's own bars so compute_cross_asset_features
        # can find them by ticker.
        if _bars_by_ticker_safe:
            r._bars_by_ticker = _bars_by_ticker_safe

        # Pre-computed crypto momentum score — avoids per-row Kraken REST calls.
        r._crypto_momentum_score = _crypto_mom_score

    # ── Pre-compute cross-asset features once for this symbol ─────────────
    # compute_cross_asset_features() does rolling Pearson correlations over
    # the full 50k-bar peer DataFrames.  Calling it once per row (×4000+
    # rows) dominates runtime on the skip path.  Cache the result and attach
    # it to every result so _build_row() reads three plain floats.
    _cached_cross_asset: Any = None
    if _bars_by_ticker_safe:
        logger.info(
            "Pre-computing cross-asset features for %s (%d peer tickers)...",
            symbol,
            len(_bars_by_ticker_safe),
        )
        try:
            from lib.analysis.cross_asset import compute_cross_asset_features as _compute_ca

            _ca_t0 = time.monotonic()
            _cached_cross_asset = _compute_ca(symbol, _bars_by_ticker_safe)
            logger.info(
                "Pre-computed cross-asset features for %s in %.1fs",
                symbol,
                time.monotonic() - _ca_t0,
            )
        except Exception as _ca_exc:
            logger.warning(
                "Cross-asset pre-compute failed for %s: %s — features will default to 0.5",
                symbol,
                _ca_exc,
                exc_info=True,
            )
            # Mark as attempted-but-failed so _build_row() does NOT fall
            # back to the expensive per-row recomputation path.
            _cached_cross_asset = _PRECOMPUTE_FAILED

    for r in all_sim_results:
        # Always attach the result (real value, None, or _PRECOMPUTE_FAILED)
        # so _build_row() can distinguish "never attempted" from "failed".
        r._cached_cross_asset = _cached_cross_asset

    # ── Pre-compute asset fingerprint once for this symbol ────────────────
    # compute_asset_fingerprint() runs Hurst exponent, session concentration,
    # and volume profile classification — all fully deterministic per symbol.
    # Pre-slice bars_1m to the 20-day lookback window HERE, before passing
    # it in, so the function never has to touch the full 50k-row series.
    # At 1 bar/min that is at most 20 × 1440 = 28 800 rows.
    _cached_fingerprint: Any = None
    _fp_bars_count = len(bars_1m) if bars_1m is not None and not bars_1m.empty else 0
    _fp_max_bars = 20 * 1440
    logger.info(
        "Pre-computing asset fingerprint for %s (%d bars → sliced to %d)...",
        symbol,
        _fp_bars_count,
        min(_fp_bars_count, _fp_max_bars),
    )
    try:
        from lib.analysis.asset_fingerprint import compute_asset_fingerprint as _compute_fp

        _fp_bars_1m = bars_1m
        if bars_1m is not None and not bars_1m.empty and len(bars_1m) > _fp_max_bars:
            _fp_bars_1m = bars_1m.iloc[-_fp_max_bars:]

        _fp_t0 = time.monotonic()
        _cached_fingerprint = _compute_fp(
            symbol,
            bars_daily=bars_daily,
            bars_1m=_fp_bars_1m,
            lookback_days=20,
        )
        logger.info(
            "Pre-computed asset fingerprint for %s in %.1fs",
            symbol,
            time.monotonic() - _fp_t0,
        )
    except Exception as _fp_exc:
        logger.warning(
            "Asset fingerprint pre-compute failed for %s: %s — fingerprint features will default to 0.5",
            symbol,
            _fp_exc,
            exc_info=True,
        )
        # Mark as attempted-but-failed so _build_row() does NOT fall
        # back to the expensive per-row recomputation path.
        _cached_fingerprint = _PRECOMPUTE_FAILED

    for r in all_sim_results:
        # Always attach the result (real value, None, or _PRECOMPUTE_FAILED)
        # so _build_row() can distinguish "never attempted" from "failed".
        r._cached_fingerprint = _cached_fingerprint

    sim_results = all_sim_results

    # Try to import chart renderer (optional — dataset can be generated
    # without images for tabular-only models)
    _use_parity = cfg.use_parity_renderer
    _can_render = False
    _can_render_parity = False
    render_ruby_snapshot: Any = None
    RenderConfig: Any = None  # noqa: N806
    render_parity_to_file: Any = None
    dataframe_to_parity_bars: Any = None
    compute_vwap_from_bars: Any = None

    if _use_parity:
        try:
            from lib.analysis.rendering.chart_renderer_parity import (
                compute_vwap_from_bars,
                dataframe_to_parity_bars,
                render_parity_to_file,
            )

            _can_render_parity = True
            _can_render = True
            logger.info("Using Pillow renderer (chart_renderer_parity.py) for CNN training images")
        except ImportError:
            logger.warning(
                "Pillow renderer requested but chart_renderer_parity.py not found — falling back to mplfinance renderer"
            )
            _use_parity = False

    if not _use_parity:
        try:
            from lib.analysis.rendering.chart_renderer import RenderConfig, render_ruby_snapshot

            _can_render = True
        except ImportError:
            logger.warning("Chart renderer not available — generating tabular-only dataset")

    render_cfg = None
    if _can_render and not _use_parity and RenderConfig is not None:
        render_cfg = RenderConfig(
            dpi=cfg.chart_dpi,
            figsize=cfg.chart_figsize,
            output_dir=cfg.image_dir,
        )

    # Count renderable results for progress logging
    _renderable_count = sum(1 for r in sim_results if r.is_trade or cfg.include_no_trade)
    _render_t0 = time.monotonic()
    _rendered_count = 0
    # Log every ~5% but cap the interval so the first message appears within
    # a reasonable number of images even for very large runs (e.g. 5000+
    # renderable results with breakout_type=all / orb_session=all).
    # The old formula ``max(25, count // 20)`` could stay silent for 250+
    # images, making the pipeline appear stuck.
    _progress_interval = max(1, min(50, _renderable_count // 20))  # log every 1-50 or ~5%

    logger.info(
        "%s: starting image rendering — %d renderable results, progress every %d images",
        symbol,
        _renderable_count,
        _progress_interval,
    )

    for sim_idx, result in enumerate(sim_results):
        # Skip no_trade unless configured to include them
        if not result.is_trade and not cfg.include_no_trade:
            continue

        label = result.label
        stats.label_distribution[label] = stats.label_distribution.get(label, 0) + 1

        # ── Global per-label cap ──────────────────────────────────────────
        if cfg.max_samples_per_label > 0 and stats.label_distribution[label] > cfg.max_samples_per_label:
            continue

        # ── Per-(label, breakout_type) cap ────────────────────────────────
        if cfg.max_samples_per_type_label > 0:
            _bt_label_key = f"{label}__{getattr(result, '_breakout_type', 'ORB')}"
            stats._type_label_counts[_bt_label_key] = stats._type_label_counts.get(_bt_label_key, 0) + 1
            if stats._type_label_counts[_bt_label_key] > cfg.max_samples_per_type_label:
                # Don't count this toward label_distribution since we're skipping it
                stats.label_distribution[label] -= 1
                continue

        # ── Per-(label, session) cap ──────────────────────────────────────
        if cfg.max_samples_per_session_label > 0:
            _sess_label_key = f"{label}__{getattr(result, '_session_key', 'unknown')}"
            stats._session_label_counts[_sess_label_key] = stats._session_label_counts.get(_sess_label_key, 0) + 1
            if stats._session_label_counts[_sess_label_key] > cfg.max_samples_per_session_label:
                stats.label_distribution[label] -= 1
                continue

        # Determine image path
        ts_str = result.breakout_time or result.or_start_time or datetime.now(_EST).isoformat()
        # Create a safe filename component from the timestamp
        safe_ts = ts_str.replace(":", "").replace("-", "").replace(" ", "_").replace("+", "p").replace(".", "d")[:20]
        image_filename = f"{symbol}_{safe_ts}_{label}_{sim_idx}.png"
        image_path = os.path.join(cfg.image_dir, image_filename)

        # Skip if already exists (resumability)
        if cfg.skip_existing and os.path.isfile(image_path):
            stats.skipped_existing += 1
            # Still add the row to CSV
            rows.append(_build_row(result, image_path))
            stats.total_images += 1
            _rendered_count += 1
            if _rendered_count % _progress_interval == 0 or _rendered_count == _renderable_count:
                _elapsed = time.monotonic() - _render_t0
                _rate = _rendered_count / _elapsed if _elapsed > 0 else 0
                logger.info(
                    "%s: %d/%d images (%.0f%%) — %.1f img/s [skipping existing]",
                    symbol,
                    _rendered_count,
                    _renderable_count,
                    100 * _rendered_count / max(_renderable_count, 1),
                    _rate,
                )
            continue

        # Render chart image
        rendered_path = None
        if _can_render:
            # Extract the window of bars for this simulation.
            # Use _window_offset stored by simulate_batch() when available —
            # this is the authoritative start index into bars_1m.  The old
            # ``sim_idx * step_size`` calculation breaks when multiple
            # sessions are concatenated (e.g. orb_session="all") because
            # sim_idx keeps incrementing across session boundaries.
            try:
                _stored_offset = getattr(result, "_window_offset", -1)
                _stored_wsize = getattr(result, "_window_size", 0)

                if _stored_offset >= 0:
                    window_start = _stored_offset
                    window_end = min(
                        window_start + (_stored_wsize or cfg.window_size),
                        len(bars_1m),
                    )
                else:
                    # Fallback for older ORBSimResult objects without provenance
                    window_start = sim_idx * cfg.step_size
                    window_end = min(window_start + cfg.window_size, len(bars_1m))

                window_bars = bars_1m.iloc[window_start:window_end].copy()

                if len(window_bars) < 10:
                    logger.warning(
                        "Skipping render for %s window %d: only %d bars (offset=%d, end=%d, total_bars=%d)",
                        symbol,
                        sim_idx,
                        len(window_bars),
                        window_start,
                        window_end,
                        len(bars_1m),
                    )
                elif _use_parity and _can_render_parity:
                    # ── Parity renderer path ──────────────────────────
                    try:
                        assert dataframe_to_parity_bars is not None
                        assert compute_vwap_from_bars is not None
                        assert render_parity_to_file is not None
                        parity_bars = dataframe_to_parity_bars(window_bars)
                        vwap_values = compute_vwap_from_bars(parity_bars)
                        rendered_path = render_parity_to_file(
                            bars=parity_bars,
                            orb_high=result.or_high if result.or_high > 0 else 0.0,
                            orb_low=result.or_low if result.or_low > 0 else 0.0,
                            vwap_values=vwap_values if len(vwap_values) == len(parity_bars) else None,
                            direction=result.direction or "long",
                            save_path=image_path,
                        )
                        if rendered_path is None:
                            logger.warning(
                                "render_parity_to_file returned None for %s window %d",
                                symbol,
                                sim_idx,
                            )
                    except Exception as parity_exc:
                        logger.warning(
                            "Parity render exception for %s window %d: %s",
                            symbol,
                            sim_idx,
                            parity_exc,
                        )
                        rendered_path = None
                else:
                    # ── Original mplfinance renderer path ─────────────
                    assert render_ruby_snapshot is not None
                    rendered_path = render_ruby_snapshot(
                        bars=window_bars,
                        symbol=symbol,
                        orb_high=result.or_high if result.or_high > 0 else None,
                        orb_low=result.or_low if result.or_low > 0 else None,
                        direction=result.direction or None,
                        quality_pct=result.quality_pct,
                        label=label,
                        save_path=image_path,
                        config=render_cfg,
                    )
                    if rendered_path is None:
                        logger.warning(
                            "render_ruby_snapshot returned None for %s window %d (bars=%d, orb_h=%s, orb_l=%s, dir=%s)",
                            symbol,
                            sim_idx,
                            len(window_bars),
                            result.or_high,
                            result.or_low,
                            result.direction,
                        )
            except Exception as exc:
                logger.warning(
                    "Render exception for %s window %d: %s",
                    symbol,
                    sim_idx,
                    exc,
                )

        if rendered_path is None and _can_render:
            stats.render_failures += 1
            # Still create a row with empty image path for tabular-only use
            image_path = ""
        elif rendered_path:
            image_path = rendered_path

        if image_path or not _can_render:
            rows.append(_build_row(result, image_path))
            stats.total_images += 1

        _rendered_count += 1
        if _rendered_count % _progress_interval == 0 or _rendered_count == _renderable_count:
            _elapsed = time.monotonic() - _render_t0
            _rate = _rendered_count / _elapsed if _elapsed > 0 else 0
            _eta = (_renderable_count - _rendered_count) / _rate if _rate > 0 else 0
            logger.info(
                "%s: %d/%d images (%.0f%%) — %.1f img/s, ETA %.0fs",
                symbol,
                _rendered_count,
                _renderable_count,
                100 * _rendered_count / max(_renderable_count, 1),
                _rate,
                _eta,
            )

    _total_elapsed = time.monotonic() - _render_t0
    logger.info(
        "%s dataset: %d images (%d skipped, %d failures) in %.1fs",
        symbol,
        stats.total_images,
        stats.skipped_existing,
        stats.render_failures,
        _total_elapsed,
    )

    return rows, stats


def _build_row(result: ORBSimResult, image_path: str) -> dict[str, Any]:
    """Build a single CSV row from an ORBSimResult.

    The row includes all columns needed by BreakoutDataset.__getitem__ to
    compute the 37-feature tabular vector per feature_contract.json v8,
    mirroring original tabular preparation exactly:

      [0]  quality_pct_norm       ← quality_pct / 100
      [1]  volume_ratio           ← breakout_volume_ratio (log-scaled in dataset)
      [2]  atr_pct                ← atr / entry  (×100 + clamp in dataset)
      [3]  cvd_delta              ← cvd_delta [-1, 1]
      [4]  nr7_flag               ← nr7
      [5]  direction_flag         ← direction
      [6]  session_ordinal        ← derived from breakout_time in dataset
      [7]  london_overlap_flag    ← london_overlap_flag
      [8]  or_range_atr_ratio     ← range_size / atr_value  (clamp+norm in dataset)
      [9]  premarket_range_ratio  ← premarket_range / range_size (clamp+norm in dataset)
      [10] bar_of_day             ← bar_of_day_minutes / 1380  (in dataset)
      [11] day_of_week            ← day_of_week_norm  (Mon=0..Fri=4)/4
      [12] vwap_distance          ← (entry - vwap) / atr  (clamp+norm in dataset)
      [13] asset_class_id         ← get_asset_class_id(symbol)
      ── v6 additions ──
      [14] breakout_type_ord      ← BreakoutType ordinal / 12  [0, 1]
      [15] asset_volatility_class ← low=0.0 / med=0.5 / high=1.0
      [16] hour_of_day            ← ET hour / 23  [0, 1]
      [17] tp3_atr_mult_norm      ← tp3_atr_mult / 5.0  [0, 1]
      ── v7 additions (Daily Strategy layer) ──
      [18] daily_bias_direction   ← from bias_analyzer: -1→0.0, 0→0.5, +1→1.0
      [19] daily_bias_confidence  ← 0.0–1.0 scalar from bias analyzer
      [20] prior_day_pattern      ← candle pattern ordinal / 9  [0, 1]
      [21] weekly_range_position  ← price in prior week H/L range [0, 1]
      [22] monthly_trend_score    ← EMA slope [-1,+1] mapped to [0, 1]
      [23] crypto_momentum_score  ← crypto lead [-1,+1] mapped to [0, 1]
      ── v7.1 additions (Phase 4B sub-features) ──
      [24] breakout_type_category ← time=0, range=0.5, squeeze=1.0
      [25] session_overlap_flag   ← 1.0 if London+NY overlap, else 0.0
      [26] atr_trend              ← ATR expanding=1.0, contracting=0.0
      [27] volume_trend           ← 5-bar volume slope [0, 1]
      ── v8-B additions (Cross-Asset Correlation) ──
      [28] primary_peer_corr      ← Pearson r with primary peer [0, 1]
      [29] cross_class_corr       ← strongest cross-class corr [0, 1]
      [30] correlation_regime     ← broken=0, normal=0.5, elevated=1.0
      ── v8-C additions (Asset Fingerprint) ──
      [31] typical_daily_range_norm ← median daily range / ATR [0, 1]
      [32] session_concentration  ← dominant session fraction [0, 1]
      [33] breakout_follow_through ← trailing win rate [0, 1]
      [34] hurst_exponent         ← mean-reversion tendency [0, 1]
      [35] overnight_gap_tendency ← overnight gap / ATR [0, 1]
      [36] volume_profile_shape   ← volume regularity score [0, 1]
    """
    import math

    # ── [2] atr_pct — ATR as fraction of entry price ──────────────────────
    atr_pct = 0.0
    if result.entry > 0 and result.atr > 0:
        atr_pct = result.atr / result.entry

    # ── Breakout type metadata ─────────────────────────────────────────────
    try:
        from lib.core.breakout_types import BreakoutType
        from lib.core.breakout_types import breakout_type_ord as _bt_ord

        _bt = getattr(result, "_breakout_type", BreakoutType.ORB)
        if not isinstance(_bt, BreakoutType):
            _bt = BreakoutType(int(_bt))
        _bt_ord_val = _bt_ord(_bt)
        _bt_name = _bt.name
    except Exception:
        _bt_name = "ORB"
        _bt_ord_val = 0.0

    # ── [8] or_range_atr_ratio — raw ORB range / ATR ──────────────────────
    # Stored raw; BreakoutDataset normalises with clamp(0,3)/3.
    range_size = result.or_range  # ORB range in price units
    atr_value = result.atr  # ATR in price units (period 14)
    or_range_atr_raw = (range_size / atr_value) if atr_value > 0 else 0.0

    # ── [9] premarket_range_ratio — PM range / ORB range ──────────────────
    # Stored raw; BreakoutDataset normalises with clamp(0,5)/5.
    pm_high = result.pm_high
    pm_low = result.pm_low
    premarket_range = 0.0
    if (
        pm_high is not None
        and pm_low is not None
        and not math.isnan(pm_high)
        and not math.isnan(pm_low)
        and pm_high > pm_low
    ):
        premarket_range = pm_high - pm_low
    pm_range_ratio_raw = (premarket_range / range_size) if range_size > 0 else 0.0

    # ── [10] bar_of_day_minutes — minutes since Globex open (18:00 ET) ────
    # BreakoutDataset divides by 1380 to normalise to [0, 1].
    bar_of_day_minutes = 0
    try:
        bt_raw = result.breakout_time
        if bt_raw is not None:
            # breakout_time may be a datetime or an ISO string
            from datetime import datetime as _dt

            bt: _dt = _dt.fromisoformat(bt_raw) if isinstance(bt_raw, str) else bt_raw
            # Normalise to a naive ET-local time if tz-aware
            if bt.tzinfo is not None:
                from zoneinfo import ZoneInfo as _ZI

                bt = bt.astimezone(_ZI("America/New_York")).replace(tzinfo=None)
            h, m = bt.hour, bt.minute
            # Minutes since Globex open at 18:00 ET
            bar_of_day_minutes = (h - 18) * 60 + m if h >= 18 else (h + 6) * 60 + m  # +6 = 24-18
    except Exception:
        bar_of_day_minutes = 0

    # ── [11] day_of_week_norm — Mon=0..Fri=4 scaled to [0, 1] ────────────
    day_of_week_norm = 0.5  # default midweek
    try:
        bt_raw2 = result.breakout_time
        if bt_raw2 is not None:
            from datetime import datetime as _dt

            bt2: _dt = _dt.fromisoformat(bt_raw2) if isinstance(bt_raw2, str) else bt_raw2
            # weekday(): Mon=0 .. Sun=6; clamp to trading days Mon-Fri
            dow = bt2.weekday()  # 0=Mon, 4=Fri, 5/6=weekend
            day_of_week_norm = dow / 4.0 if 0 <= dow <= 4 else 0.5  # fallback for weekend bars (rare)
    except Exception:
        day_of_week_norm = 0.5

    # ── [12] vwap_distance — (entry − vwap) / ATR ─────────────────────────
    # VWAP is not always available from the simulator; use the ORB midpoint
    # as a proxy when missing.  The proxy is a reasonable estimate because
    # the ORB midpoint approximates the session VWAP at the breakout bar.
    # Stored raw; BreakoutDataset normalises with clamp(-3,3)/3.
    vwap = getattr(result, "vwap", None)
    if vwap is None or (isinstance(vwap, float) and math.isnan(vwap)):
        # Proxy: ORB midpoint
        vwap = (result.or_high + result.or_low) / 2.0 if result.or_range > 0 else result.entry
    vwap_distance_raw = ((result.entry - vwap) / atr_value) if atr_value > 0 else 0.0

    # ── [13] asset_class_id — ordinal / 4 matching asset class normalisation ─
    try:
        from lib.analysis.ml.breakout_cnn import get_asset_class_id as _get_cls

        asset_class_id = _get_cls(result.symbol)
    except Exception:
        asset_class_id = 0.0

    # ── [15] asset_volatility_class — low=0.0, med=0.5, high=1.0 ─────────
    try:
        from lib.analysis.ml.breakout_cnn import get_asset_volatility_class as _get_vol

        asset_vol_class = _get_vol(result.symbol)
    except Exception:
        asset_vol_class = 0.5

    # ── [16] hour_of_day — ET hour / 23 → [0, 1] ─────────────────────────
    hour_of_day_norm = 0.5
    try:
        bt_raw3 = result.breakout_time
        if bt_raw3 is not None:
            from datetime import datetime as _dt

            bt3: _dt = _dt.fromisoformat(bt_raw3) if isinstance(bt_raw3, str) else bt_raw3
            if bt3.tzinfo is not None:
                from zoneinfo import ZoneInfo as _ZI

                bt3 = bt3.astimezone(_ZI("America/New_York")).replace(tzinfo=None)
            hour_of_day_norm = max(0.0, min(1.0, bt3.hour / 23.0))
    except Exception:
        hour_of_day_norm = 0.5

    # ── [17] tp3_atr_mult_norm — TP3 multiplier / 5.0 → [0, 1] ──────────
    tp3_atr_mult_raw = 0.0
    try:
        _cfg = getattr(result, "_range_config", None)
        if _cfg is not None and hasattr(_cfg, "tp3_atr_mult"):
            tp3_atr_mult_raw = float(_cfg.tp3_atr_mult)
        elif hasattr(result, "tp3_atr_mult"):
            tp3_atr_mult_raw = float(result.tp3_atr_mult)  # type: ignore[union-attr]
        else:
            # Fall back to looking up the canonical config by breakout type
            try:
                from lib.core.breakout_types import get_range_config as _get_rc

                _bt = getattr(result, "_breakout_type", None)
                if _bt is not None:
                    from lib.core.breakout_types import BreakoutType as _BT

                    if not isinstance(_bt, _BT):
                        _bt = _BT(int(_bt))
                    _rc = _get_rc(_bt)
                    tp3_atr_mult_raw = float(_rc.tp3_atr_mult)
            except Exception:
                pass
    except Exception:
        tp3_atr_mult_raw = 0.0

    # ── v7 features [18–23] — Daily Strategy layer ────────────────────────
    # These features come from the daily bias analyzer and crypto momentum
    # modules.  They require daily bars which may be attached to the result
    # or loaded separately.  Falls back to neutral defaults if unavailable.

    # [18] daily_bias_direction — SHORT=0.0, NEUTRAL=0.5, LONG=1.0
    daily_bias_dir = 0.5
    # [19] daily_bias_confidence — 0.0–1.0
    daily_bias_conf = 0.0
    # [20] prior_day_pattern — candle pattern ordinal / 9 → [0, 1]
    prior_day_pat = 1.0  # neutral default
    # [21] weekly_range_position — [0, 1]
    weekly_range_pos = 0.5
    # [22] monthly_trend_score — [-1,+1] → [0, 1]
    monthly_trend = 0.5

    try:
        # Check if daily bars are attached to the result (set by the simulator)
        _daily_bars = getattr(result, "_daily_bars", None)
        _asset_name = getattr(result, "_asset_name", None) or result.symbol

        if _daily_bars is not None and not _daily_bars.empty:
            try:
                from lib.analysis.ml.breakout_cnn import (
                    get_daily_bias_confidence,
                    get_daily_bias_direction,
                    get_monthly_trend_score,
                    get_prior_day_pattern,
                    get_weekly_range_position,
                )

                daily_bias_dir = get_daily_bias_direction(_asset_name, _daily_bars)
                daily_bias_conf = get_daily_bias_confidence(_asset_name, _daily_bars)
                prior_day_pat = get_prior_day_pattern(_asset_name, _daily_bars)
                weekly_range_pos = get_weekly_range_position(_asset_name, _daily_bars)
                monthly_trend = get_monthly_trend_score(_asset_name, _daily_bars)
            except Exception:
                pass
        else:
            # Try to use a pre-computed bias cache if attached
            _bias_cache = getattr(result, "_bias_cache", None)
            if _bias_cache is not None and _asset_name in _bias_cache:
                try:
                    bias = _bias_cache[_asset_name]
                    daily_bias_dir = getattr(bias, "direction_feature", 0.5)
                    daily_bias_conf = getattr(bias, "confidence_feature", 0.0)
                    prior_day_pat = getattr(bias, "candle_pattern_feature", 1.0)
                    weekly_range_pos = getattr(bias, "weekly_range_feature", 0.5)
                    monthly_trend = getattr(bias, "monthly_trend_feature", 0.5)
                except Exception:
                    pass
    except Exception:
        pass

    # [23] crypto_momentum_score — [-1,+1] → [0, 1]
    # Read the pre-computed score attached by generate_dataset_for_symbol().
    # This avoids a Kraken REST call on every row; the score is computed once
    # per symbol and reused across all its trade rows.
    crypto_mom = float(getattr(result, "_crypto_momentum_score", 0.5))

    # ── v7.1 sub-features [24–27] — Phase 4B decomposition ───────────────
    # These sub-features enrich existing features with additional nuance
    # without replacing them.

    # [24] breakout_type_category — time-based=0, range-based=0.5, squeeze=1.0
    bt_category = 0.5
    try:
        from lib.analysis.ml.breakout_cnn import get_breakout_type_category

        bt_category = get_breakout_type_category(_bt_name)
    except Exception:
        pass

    # [25] session_overlap_flag — 1.0 if London+NY overlap, else 0.0
    session_overlap = 0.0
    try:
        from lib.analysis.ml.breakout_cnn import get_session_overlap_flag

        _session_key = getattr(result, "_session_key", "us")
        # Also check the bar hour for overlap detection
        _bar_hour_et = None
        try:
            bt_raw4 = result.breakout_time
            if bt_raw4 is not None:
                from datetime import datetime as _dt

                bt4: _dt = _dt.fromisoformat(bt_raw4) if isinstance(bt_raw4, str) else bt_raw4
                if bt4.tzinfo is not None:
                    from zoneinfo import ZoneInfo as _ZI

                    bt4 = bt4.astimezone(_ZI("America/New_York"))
                _bar_hour_et = bt4.hour
        except Exception:
            pass
        session_overlap = get_session_overlap_flag(_session_key, bar_hour_et=_bar_hour_et)
    except Exception:
        pass

    # [26] atr_trend — ATR expanding=1.0, contracting=0.0 (10-bar lookback)
    atr_trend_val = 0.5
    try:
        from lib.analysis.ml.breakout_cnn import get_atr_trend

        _bars_1m = getattr(result, "_bars_1m", None)
        if _bars_1m is not None and not _bars_1m.empty:
            atr_trend_val = get_atr_trend(_bars_1m, lookback=10)
    except Exception:
        pass

    # [27] volume_trend — 5-bar volume slope, normalised [0, 1]
    vol_trend_val = 0.5
    try:
        from lib.analysis.ml.breakout_cnn import get_volume_trend

        _bars_1m = getattr(result, "_bars_1m", None)
        if _bars_1m is not None and not _bars_1m.empty:
            vol_trend_val = get_volume_trend(_bars_1m, lookback=5)
    except Exception:
        pass

    # ── v8-B cross-asset correlation features [28–30] ─────────────────────
    # Prefer the pre-computed cached result attached by generate_dataset_for_symbol()
    # to avoid recomputing rolling Pearson correlations on the full 50k-bar
    # peer DataFrames for every row.  Falls back to computing on-demand ONLY
    # when the attribute is completely absent (standalone _build_row() calls).
    # If the attribute is _PRECOMPUTE_FAILED, skip the fallback — the
    # pre-compute already tried and failed; retrying per-row would be
    # catastrophically slow (~5000× the cost) and flood logs with warnings.
    primary_peer_corr = 0.5
    cross_class_corr = 0.5
    correlation_regime = 0.5
    try:
        _cross_sentinel = object()  # unique default to detect "not set"
        _cross_feats = getattr(result, "_cached_cross_asset", _cross_sentinel)
        if _cross_feats is _cross_sentinel:
            # Attribute not set at all — standalone call, not from generate_dataset_for_symbol.
            # Fall back to on-demand compute (expensive but necessary).
            from lib.analysis.cross_asset import compute_cross_asset_features

            _bars_by_ticker = getattr(result, "_bars_by_ticker", None)
            if _bars_by_ticker is not None and isinstance(_bars_by_ticker, dict):
                _cross_feats = compute_cross_asset_features(result.symbol, _bars_by_ticker)
        elif _cross_feats is _PRECOMPUTE_FAILED or _cross_feats is None:
            # Pre-compute was attempted but failed (or no peers available).
            # Use neutral defaults — do NOT retry the expensive computation.
            _cross_feats = None
        if _cross_feats is not None and _cross_feats is not _PRECOMPUTE_FAILED:
            from typing import cast as _cast

            from lib.analysis.cross_asset import CrossAssetFeatures as _CAF

            _cross_feats = _cast("_CAF", _cross_feats)
            primary_peer_corr = _cross_feats.primary_peer_corr
            cross_class_corr = _cross_feats.cross_class_corr
            correlation_regime = _cross_feats.correlation_regime
    except Exception:
        pass

    # ── v8-C asset fingerprint features [31–36] ──────────────────────────
    # Prefer the pre-computed cached fingerprint attached by
    # generate_dataset_for_symbol() to avoid running Hurst exponent,
    # session concentration, and volume profile classification (each doing
    # a full df.copy() of the 50k-bar 1m series) on every row.
    # Falls back to computing on-demand ONLY when the attribute is
    # completely absent (standalone _build_row() calls).  If the attribute
    # is _PRECOMPUTE_FAILED, skip — retrying per-row would be catastrophic.
    typical_daily_range_norm = 0.5
    session_concentration_val = 0.5
    breakout_follow_through_val = 0.5
    hurst_exponent_val = 0.5
    overnight_gap_tendency = 0.5
    volume_profile_shape_val = 0.5
    try:
        _fp_sentinel = object()  # unique default to detect "not set"
        _fp = getattr(result, "_cached_fingerprint", _fp_sentinel)
        if _fp is _fp_sentinel:
            # Attribute not set at all — standalone call, not from generate_dataset_for_symbol.
            # Fall back to on-demand compute (expensive but necessary).
            from lib.analysis.asset_fingerprint import compute_asset_fingerprint

            _daily_bars_fp = getattr(result, "_daily_bars", None)
            _bars_1m_fp = getattr(result, "_bars_1m", None)
            if _daily_bars_fp is not None and not _daily_bars_fp.empty:
                _fp = compute_asset_fingerprint(
                    result.symbol,
                    bars_daily=_daily_bars_fp,
                    bars_1m=_bars_1m_fp,
                    lookback_days=20,
                )
        elif _fp is _PRECOMPUTE_FAILED or _fp is None:
            # Pre-compute was attempted but failed.
            # Use neutral defaults — do NOT retry the expensive computation.
            _fp = None
        if _fp is not None and _fp is not _PRECOMPUTE_FAILED:
            from typing import cast as _cast

            from lib.analysis.asset_fingerprint import AssetFingerprint as _AFP
            from lib.analysis.asset_fingerprint import VolumeProfileShape as _VPS

            _fp = _cast("_AFP", _fp)
            # [31] typical_daily_range_norm — clamp [0.5, 2.5] → [0, 1]
            typical_daily_range_norm = max(0.0, min(1.0, (_fp.typical_daily_range_atr - 0.5) / 2.0))

            # [32] session_concentration — dominant session fraction
            session_concentration_val = max(
                _fp.session_concentration.overnight_pct,
                _fp.session_concentration.london_pct,
                _fp.session_concentration.us_pct,
                _fp.session_concentration.settle_pct,
            )

            # [33] breakout_follow_through — trailing win rate
            breakout_follow_through_val = _fp.breakout_follow_through.follow_through_rate

            # [34] hurst_exponent — mean-reversion tendency
            hurst_exponent_val = max(0.0, min(1.0, _fp.mean_reversion_tendency))

            # [35] overnight_gap_tendency — gap frequency as proxy
            overnight_gap_tendency = max(0.0, min(1.0, _fp.overnight_gap.avg_gap_atr_ratio))

            # [36] volume_profile_shape — regularity score
            _shape_scores = {
                _VPS.U_SHAPED: 0.9,
                _VPS.L_SHAPED: 0.7,
                _VPS.FRONT_LOADED: 0.6,
                _VPS.FLAT: 0.4,
                _VPS.UNKNOWN: 0.5,
            }
            volume_profile_shape_val = _shape_scores.get(_fp.volume_profile_shape, 0.5)
    except Exception:
        pass

    return {
        # ── Identity / label ──────────────────────────────────────────────
        "image_path": image_path,
        "label": result.label,
        "symbol": result.symbol,
        "direction": result.direction,
        # ── v4 tabular features (raw — normalisation applied in BreakoutDataset) ──
        "quality_pct": result.quality_pct,  # [0] → /100
        "volume_ratio": round(result.breakout_volume_ratio, 4),  # [1] → log-norm
        "atr_pct": round(atr_pct, 6),  # [2] → ×100 clamp
        "cvd_delta": round(getattr(result, "cvd_delta", 0.0), 4),  # [3]
        "nr7_flag": 1 if result.nr7 else 0,  # [4]
        # direction_flag [5] derived from "direction" column in dataset
        # session_ordinal [6] derived from "breakout_time" column in dataset
        "london_overlap_flag": getattr(result, "london_overlap_flag", 0.0),  # [7]
        "range_size": round(range_size, 6),  # [8] raw ORB range
        "atr_value": round(atr_value, 6),  # [8] ATR (price units)
        "or_range_atr_ratio": round(or_range_atr_raw, 4),  # [8] raw ratio → stored for debug
        "premarket_range": round(premarket_range, 6),  # [9] raw PM range
        "pm_range_ratio": round(pm_range_ratio_raw, 4),  # [9] raw ratio → stored for debug
        "bar_of_day_minutes": bar_of_day_minutes,  # [10] raw minutes → /1380 in dataset
        "day_of_week_norm": round(day_of_week_norm, 4),  # [11] already [0,1]
        "vwap_distance": round(vwap_distance_raw, 4),  # [12] raw → clamp+norm in dataset
        "asset_class_id": round(asset_class_id, 4),  # [13] already [0,1]
        # ── Trade geometry ────────────────────────────────────────────────
        "entry": round(result.entry, 6),
        "sl": round(result.sl, 6),
        "tp1": round(result.tp1, 6),
        "or_high": round(result.or_high, 6),
        "or_low": round(result.or_low, 6),
        "or_range": round(result.or_range, 6),
        "atr": round(result.atr, 6),
        "pnl_r": round(result.pnl_r, 4),
        "hold_bars": result.hold_bars,
        "outcome": result.outcome,
        "breakout_time": result.breakout_time,
        "pm_high": round(pm_high if (pm_high is not None and not math.isnan(pm_high)) else 0.0, 6),
        "pm_low": round(pm_low if (pm_low is not None and not math.isnan(pm_low)) else 0.0, 6),
        # ── Session / breakout-type metadata (informational + stratification) ──
        "session_key": getattr(result, "_session_key", "us"),
        "breakout_type": _bt_name,
        "breakout_type_ord": round(_bt_ord_val, 6),
        # ── v6 tabular feature columns ────────────────────────────────────
        "asset_volatility_class": round(asset_vol_class, 4),  # [15]
        "hour_of_day": round(hour_of_day_norm, 4),  # [16]
        "tp3_atr_mult": round(tp3_atr_mult_raw, 4),  # [17] raw value; dataset normalises /5.0
        # ── v7 tabular feature columns (Daily Strategy layer) ─────────────
        "daily_bias_direction": round(daily_bias_dir, 4),  # [18] already [0, 1]
        "daily_bias_confidence": round(daily_bias_conf, 4),  # [19] already [0, 1]
        "prior_day_pattern": round(prior_day_pat, 4),  # [20] already [0, 1]
        "weekly_range_position": round(weekly_range_pos, 4),  # [21] already [0, 1]
        "monthly_trend_score": round(monthly_trend, 4),  # [22] already [0, 1]
        "crypto_momentum_score": round(crypto_mom, 4),  # [23] already [0, 1]
        # ── v7.1 sub-feature columns (Phase 4B decomposition) ─────────────
        "breakout_type_category": round(bt_category, 4),  # [24] already {0, 0.5, 1.0}
        "session_overlap_flag": round(session_overlap, 4),  # [25] already {0.0, 1.0}
        "atr_trend": round(atr_trend_val, 4),  # [26] already [0, 1]
        "volume_trend": round(vol_trend_val, 4),  # [27] already [0, 1]
        # ── v8-B cross-asset correlation columns ──────────────────────────
        "primary_peer_corr": round(primary_peer_corr, 4),  # [28] already [0, 1]
        "cross_class_corr": round(cross_class_corr, 4),  # [29] already [0, 1]
        "correlation_regime": round(correlation_regime, 4),  # [30] already [0, 1]
        # ── v8-C asset fingerprint columns ────────────────────────────────
        "typical_daily_range_norm": round(typical_daily_range_norm, 4),  # [31] already [0, 1]
        "session_concentration": round(session_concentration_val, 4),  # [32] already [0, 1]
        "breakout_follow_through": round(breakout_follow_through_val, 4),  # [33] already [0, 1]
        "hurst_exponent": round(hurst_exponent_val, 4),  # [34] already [0, 1]
        "overnight_gap_tendency": round(overnight_gap_tendency, 4),  # [35] already [0, 1]
        "volume_profile_shape": round(volume_profile_shape_val, 4),  # [36] already [0, 1]
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def _resolve_peer_tickers(symbol: str) -> list[str]:
    """Return the list of peer tickers needed for cross-asset features.

    Looks up the symbol in ``cross_asset.PEER_MAP`` and returns the
    primary peer plus all cross-class peers.  Returns an empty list if
    no peer mapping exists.
    """
    try:
        from lib.analysis.cross_asset import PEER_MAP

        peer_info = PEER_MAP.get(symbol)
        if peer_info is None:
            # Try stripping =F suffix
            stripped = symbol.split("=")[0] if "=" in symbol else symbol
            peer_info = PEER_MAP.get(stripped)
        if peer_info is None:
            return []

        tickers: list[str] = []
        primary = peer_info.get("primary_peer", "")
        if primary:
            tickers.append(primary)
        cross = peer_info.get("cross_class_peers", [])
        for t in cross:
            if t and t not in tickers:
                tickers.append(t)
        return tickers
    except Exception:
        return []


def generate_dataset(
    symbols: Sequence[str],
    days_back: int = 90,
    config: DatasetConfig | None = None,
    bars_override: dict[str, pd.DataFrame] | None = None,
    fetch_only: bool = False,
) -> DatasetStats:
    """Generate a complete labeled dataset for CNN training.

    This is the main entry point called by the scheduler or CLI.

    Args:
        symbols: List of instrument symbols (e.g. ["MGC", "MES", "MNQ"]).
        days_back: Number of days of historical data to process.
        config: DatasetConfig (uses defaults if None).
        bars_override: Optional pre-loaded bars dict (symbol → DataFrame).
                       If provided, skips the data loading step.
        fetch_only: If True, only fetch and cache bar data from the engine —
                    no chart images are rendered and no labels.csv is written.
                    This is the "Load Data" step: it does all the heavy network
                    I/O on the cloud/engine side so the GPU trainer only needs
                    to render images and train, without waiting on data fetches.

    Returns:
        DatasetStats with aggregate statistics.
    """
    cfg = config or DatasetConfig()
    start_time = time.monotonic()

    # Ensure output directories exist
    os.makedirs(cfg.output_dir, exist_ok=True)
    os.makedirs(cfg.image_dir, exist_ok=True)

    aggregate_stats = DatasetStats()
    all_rows: list[dict[str, Any]] = []

    # ── Pre-load peer bars for cross-asset features (v8-B) ────────────
    # Build a shared cache of 1-minute bars keyed by short ticker.
    # Symbols that appear in the main list will be loaded anyway during
    # the per-symbol loop; peers that are NOT in the list are loaded here
    # up-front.  The cache avoids loading the same ticker twice.
    _bars_cache: dict[str, pd.DataFrame | None] = {}  # ticker → bars (or None)
    if bars_override:
        for k, v in bars_override.items():
            _bars_cache[k] = v

    # Discover all peer tickers we'll need
    _all_peer_tickers: set[str] = set()
    for symbol in symbols:
        _all_peer_tickers.update(_resolve_peer_tickers(symbol))
    # Remove tickers that are already in the primary symbol list (they'll
    # be loaded during the main loop)
    _peer_only_tickers = _all_peer_tickers - set(symbols) - set(_bars_cache.keys())

    if _peer_only_tickers:
        logger.info(
            "Pre-loading %d peer tickers for cross-asset features: %s",
            len(_peer_only_tickers),
            ", ".join(sorted(_peer_only_tickers)),
        )
        for peer_ticker in sorted(_peer_only_tickers):
            try:
                peer_bars = load_bars(
                    peer_ticker,
                    source=cfg.bars_source,
                    days=days_back,
                    csv_dir=cfg.csv_bars_dir,
                )
                _bars_cache[peer_ticker] = peer_bars
                if peer_bars is not None and not peer_bars.empty:
                    logger.info("  ✓ %s: %d bars loaded", peer_ticker, len(peer_bars))
                else:
                    logger.info("  ✗ %s: no data available", peer_ticker)
            except Exception as exc:
                logger.warning("  ✗ %s: load failed — %s", peer_ticker, exc)
                _bars_cache[peer_ticker] = None

    total_symbols = len(symbols)

    # ── fetch_only mode: just warm the bar cache, skip all image rendering ──
    # This is the "Load Data" step — it pulls all raw bars from the engine
    # (the heavy network work) so subsequent "Generate Dataset" and "Train"
    # steps only have to read local data.
    if fetch_only:
        total_bars_fetched = 0
        # symbol → bar count, recorded for low-bar analysis after the loop
        _fetched_bar_counts: dict[str, int] = {}
        all_fetch_symbols = list(symbols) + sorted(_peer_only_tickers)
        for sym_idx, symbol in enumerate(all_fetch_symbols, 1):
            logger.info("Fetching bars for %s [%d/%d]...", symbol, sym_idx, len(all_fetch_symbols))
            try:
                bars_1m = load_bars(
                    symbol,
                    source=cfg.bars_source,
                    days=days_back,
                    csv_dir=cfg.csv_bars_dir,
                )
                if bars_1m is not None and not bars_1m.empty:
                    n = len(bars_1m)
                    total_bars_fetched += n
                    _fetched_bar_counts[symbol] = n
                    aggregate_stats.symbols_processed.append(symbol)
                    logger.info("  ✓ %s: %d bars fetched", symbol, n)
                else:
                    msg = f"No bar data returned for {symbol}"
                    logger.warning("  ✗ %s", msg)
                    aggregate_stats.errors.append(msg)
            except Exception as exc:
                msg = f"Bar fetch failed for {symbol}: {exc}"
                logger.warning("  ✗ %s", msg)
                aggregate_stats.errors.append(msg)

        # ── Low-bar detection + retry ─────────────────────────────────
        # Flag symbols whose bar count is well below what we'd expect for
        # `days_back` of history.  1-minute futures trade ~23 h/day on
        # weekdays, so a conservative floor of 200 bars/day (accounts for
        # weekends, holidays, and CME maintenance windows) gives us a
        # threshold that catches genuinely sparse symbols without
        # false-positives on recently-listed contracts.
        _low_bar_threshold = days_back * 200
        _low_bar_symbols: list[tuple[str, int]] = [
            (sym, cnt)
            for sym, cnt in sorted(_fetched_bar_counts.items(), key=lambda x: x[1])
            if cnt < _low_bar_threshold
        ]
        if _low_bar_symbols:
            logger.warning(
                "⚠  %d symbol(s) have fewer bars than expected for %d days "
                "(threshold: %d bars = %d days x 200 bars/day).",
                len(_low_bar_symbols),
                days_back,
                _low_bar_threshold,
                days_back,
            )
            logger.warning("   These symbols will produce fewer training samples and may hurt model quality.")
            logger.warning("   Requesting a deeper fill then re-checking each one...")
            logger.warning("   %-10s  %8s  %8s  %7s", "Symbol", "Got", "Expected", "Cover%")
            logger.warning("   %-10s  %8s  %8s  %7s", "------", "---", "--------", "-------")
            for sym, cnt in _low_bar_symbols:
                pct = 100.0 * cnt / _low_bar_threshold if _low_bar_threshold else 0.0
                logger.warning("   %-10s  %8d  %8d  %6.1f%%", sym, cnt, _low_bar_threshold, pct)

            # ── Retry: request a deeper fill then re-fetch ────────────
            # Ask the engine to pull further back (3× the original window)
            # for each under-stocked symbol.  If the bar count still falls
            # short after the fill we drop the symbol so it doesn't produce
            # a lop-sided, under-representative slice of the dataset.
            _deeper_days = days_back * 3
            for sym, _cnt in _low_bar_symbols:
                logger.info("  → requesting deeper fill for %s (%d days)...", sym, _deeper_days)
                _request_deeper_fill(sym, _deeper_days)

                # Re-fetch to see how many bars we have now
                try:
                    refetched = load_bars(
                        sym,
                        source=cfg.bars_source,
                        days=_deeper_days,
                        csv_dir=cfg.csv_bars_dir,
                    )
                    new_cnt = len(refetched) if refetched is not None and not refetched.empty else 0
                except Exception:
                    new_cnt = 0

                if new_cnt >= _low_bar_threshold:
                    logger.info("  ✓ %s: now %d bars after deeper fill — keeping", sym, new_cnt)
                    _fetched_bar_counts[sym] = new_cnt
                else:
                    logger.warning(
                        "  ✗ %s: still only %d bars after deeper fill (need %d) — dropping from dataset",
                        sym,
                        new_cnt,
                        _low_bar_threshold,
                    )
                    aggregate_stats.dropped_symbols.append(sym)
                    aggregate_stats.errors.append(
                        f"DROPPED:{sym}: only {new_cnt} bars available "
                        f"(need {_low_bar_threshold}) after deeper fill attempt"
                    )

        aggregate_stats.duration_seconds = time.monotonic() - start_time
        logger.info(
            "Load Data complete — %d symbols fetched, %d total bars in %.1fs",
            len(aggregate_stats.symbols_processed),
            total_bars_fetched,
            aggregate_stats.duration_seconds,
        )
        return aggregate_stats

    for sym_idx, symbol in enumerate(symbols, 1):
        logger.info("Processing %s [%d/%d]...", symbol, sym_idx, total_symbols)

        # Load bar data
        if bars_override and symbol in bars_override:
            bars_1m = bars_override[symbol]
        elif symbol in _bars_cache and _bars_cache[symbol] is not None:
            bars_1m = _bars_cache[symbol]
        else:
            bars_1m = load_bars(
                symbol,
                source=cfg.bars_source,
                days=days_back,
                csv_dir=cfg.csv_bars_dir,
            )

        if bars_1m is None or bars_1m.empty:
            msg = f"No bar data available for {symbol} — skipping"
            logger.warning(msg)
            aggregate_stats.errors.append(msg)
            continue

        # Store in cache so peer lookups can find it
        _bars_cache[symbol] = bars_1m

        # Load daily bars for NR7 + v7/v8 features
        bars_daily = load_daily_bars(symbol, source=cfg.bars_source, csv_dir=cfg.csv_bars_dir)

        # ── Build bars_by_ticker for cross-asset features ─────────────
        # Include this symbol's own bars plus all its peers' bars.
        bars_by_ticker: dict[str, pd.DataFrame] = {symbol: bars_1m}
        peer_tickers = _resolve_peer_tickers(symbol)
        for pt in peer_tickers:
            # Check the cache first
            if pt in _bars_cache:
                _cached_bars = _bars_cache[pt]
                if _cached_bars is not None and not _cached_bars.empty:
                    bars_by_ticker[pt] = _cached_bars
            else:
                # Peer wasn't pre-loaded (e.g. not in PEER_MAP at planning
                # time, or newly discovered).  Try loading now.
                try:
                    pb = load_bars(
                        pt,
                        source=cfg.bars_source,
                        days=days_back,
                        csv_dir=cfg.csv_bars_dir,
                    )
                    _bars_cache[pt] = pb
                    if pb is not None and not pb.empty:
                        bars_by_ticker[pt] = pb
                except Exception:
                    _bars_cache[pt] = None

        _n_peers_loaded = len(bars_by_ticker) - 1  # exclude the symbol itself
        if _n_peers_loaded > 0:
            logger.info(
                "%s: %d/%d peer bars available for cross-asset features",
                symbol,
                _n_peers_loaded,
                len(peer_tickers),
            )

        try:
            rows, symbol_stats = generate_dataset_for_symbol(
                symbol=symbol,
                bars_1m=bars_1m,
                bars_daily=bars_daily,
                config=cfg,
                bars_by_ticker=bars_by_ticker,
            )

            all_rows.extend(rows)

            # Merge stats
            aggregate_stats.total_windows += symbol_stats.total_windows
            aggregate_stats.total_trades += symbol_stats.total_trades
            aggregate_stats.total_images += symbol_stats.total_images
            aggregate_stats.skipped_existing += symbol_stats.skipped_existing
            aggregate_stats.render_failures += symbol_stats.render_failures
            aggregate_stats.symbols_processed.append(symbol)
            for label, count in symbol_stats.label_distribution.items():
                aggregate_stats.label_distribution[label] = aggregate_stats.label_distribution.get(label, 0) + count

        except Exception as exc:
            msg = f"Dataset generation failed for {symbol}: {exc}"
            logger.error(msg, exc_info=True)
            aggregate_stats.errors.append(msg)

    # Write CSV manifest
    csv_path = os.path.join(cfg.output_dir, cfg.csv_filename)
    if all_rows:
        df = pd.DataFrame(all_rows)

        # If CSV already exists, append (for incremental builds)
        if os.path.isfile(csv_path) and cfg.skip_existing:
            try:
                existing_df = pd.read_csv(csv_path)
                # Deduplicate by image_path
                existing_paths = list(existing_df["image_path"].tolist())
                new_rows = df[~df["image_path"].isin(existing_paths)]
                if not new_rows.empty:
                    df = pd.concat([existing_df, new_rows], ignore_index=True)
                    logger.info("Appended %d new rows to existing CSV (%d total)", len(new_rows), len(df))
                else:
                    df = existing_df
                    logger.info("No new rows to append — CSV unchanged (%d rows)", len(df))
            except Exception as exc:
                logger.warning("Could not append to existing CSV, overwriting: %s", exc)

        df.to_csv(csv_path, index=False)
        aggregate_stats.csv_path = csv_path
        logger.info("Dataset CSV written: %s (%d rows)", csv_path, len(df))
    else:
        logger.warning("No data generated — CSV not written")

    aggregate_stats.duration_seconds = time.monotonic() - start_time

    # Write stats JSON alongside CSV for audit
    stats_path = os.path.join(cfg.output_dir, "dataset_stats.json")
    try:
        with open(stats_path, "w") as f:
            json.dump(aggregate_stats.to_dict(), f, indent=2)
    except Exception:
        pass

    logger.info(aggregate_stats.summary())
    return aggregate_stats


# ---------------------------------------------------------------------------
# Train/Val split helper
# ---------------------------------------------------------------------------


def split_dataset(
    csv_path: str,
    val_fraction: float = 0.15,
    output_dir: str | None = None,
    stratify: bool = True,
    random_seed: int = 42,
) -> tuple[str, str]:
    """Split a dataset CSV into train and validation sets.

    Args:
        csv_path: Path to the full dataset CSV.
        val_fraction: Fraction of data for validation (default 0.15).
        output_dir: Where to write the split CSVs (default: same dir as input).
        stratify: If True, maintain label distribution in both splits.
        random_seed: Random seed for reproducibility.

    Returns:
        (train_csv_path, val_csv_path)
    """
    df = pd.read_csv(csv_path)
    out_dir = output_dir or os.path.dirname(csv_path)

    rng = np.random.RandomState(random_seed)

    # --- Infer session from breakout_time for stratification ---
    # London session: breakout hour < 8 ET;  US session: hour >= 8 ET.
    def _infer_session(bt: Any) -> str:
        try:
            bt_str = str(bt).strip()
            if not bt_str or bt_str.lower() == "nan":
                return "unknown"
            # Parse hour from timestamp like "2026-01-29 03:30:00-05:00"
            hour = int(bt_str.split(" ")[1].split(":")[0]) if " " in bt_str else 10
            return "london" if hour < 8 else "us"
        except Exception:
            return "unknown"

    if stratify and "label" in df.columns:
        # Build a composite stratification key from (label, breakout_type, session)
        # so that every combination is proportionally represented in train and
        # val splits.  This prevents rare breakout types (Monthly, Weekly) or
        # overnight sessions from ending up entirely in one split.
        if "breakout_time" in df.columns:
            df["_session"] = df["breakout_time"].apply(_infer_session)
        else:
            df["_session"] = "unknown"

        if "breakout_type" in df.columns:
            df["_bt"] = df["breakout_type"].astype(str)
        else:
            df["_bt"] = "ORB"

        df["_strat_key"] = df["label"].astype(str) + "__" + df["_bt"] + "__" + df["_session"]

        train_parts = []
        val_parts = []
        for _key, group in df.groupby("_strat_key"):
            n_val = max(1, int(len(group) * val_fraction))
            shuffled = group.sample(frac=1, random_state=rng)
            val_parts.append(shuffled.iloc[:n_val])
            train_parts.append(shuffled.iloc[n_val:])

        train_df = pd.concat(train_parts, ignore_index=True).sample(frac=1, random_state=rng)
        val_df = pd.concat(val_parts, ignore_index=True).sample(frac=1, random_state=rng)

        # Log stratification breakdown for audit
        for split_name, split_df in [("train", train_df), ("val", val_df)]:
            session_counts = split_df["_session"].value_counts().to_dict() if "_session" in split_df.columns else {}
            label_counts = split_df["label"].value_counts().to_dict() if "label" in split_df.columns else {}
            bt_counts = split_df["_bt"].value_counts().to_dict() if "_bt" in split_df.columns else {}
            logger.info(
                "  %s split — labels: %s | breakout_types: %s | sessions: %s",
                split_name,
                label_counts,
                bt_counts,
                session_counts,
            )

        # Drop helper columns before saving
        for col in ("_session", "_bt", "_strat_key"):
            if col in train_df.columns:
                train_df = train_df.drop(columns=[col])
            if col in val_df.columns:
                val_df = val_df.drop(columns=[col])
        # Also drop from the source df to avoid leaking helper columns
        for col in ("_session", "_bt", "_strat_key"):
            if col in df.columns:
                df = df.drop(columns=[col])
    else:
        shuffled = df.sample(frac=1, random_state=rng)
        n_val = max(1, int(len(shuffled) * val_fraction))
        val_df = shuffled.iloc[:n_val]
        train_df = shuffled.iloc[n_val:]

    train_path = os.path.join(out_dir, "train.csv")
    val_path = os.path.join(out_dir, "val.csv")

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)

    logger.info(
        "Split dataset: %d train / %d val (%.1f%%) — %s, %s",
        len(train_df),
        len(val_df),
        val_fraction * 100,
        train_path,
        val_path,
    )

    return train_path, val_path


# ---------------------------------------------------------------------------
# Dataset validation helper
# ---------------------------------------------------------------------------


def validate_dataset(csv_path: str, check_images: bool = True) -> dict[str, Any]:
    """Validate a dataset CSV and optionally check that all images exist.

    Returns a report dict with counts, missing images, label distribution, etc.
    """
    if not os.path.isfile(csv_path):
        return {"valid": False, "error": f"CSV not found: {csv_path}"}

    df = pd.read_csv(csv_path)
    report: dict[str, Any] = {
        "valid": True,
        "total_rows": len(df),
        "columns": list(df.columns),
        "label_distribution": {},
        "symbols": [],
        "missing_images": 0,
        "empty_image_paths": 0,
    }

    if "label" in df.columns:
        report["label_distribution"] = df["label"].value_counts().to_dict()

    if "symbol" in df.columns:
        report["symbols"] = sorted(df["symbol"].unique().tolist())

    if check_images and "image_path" in df.columns:
        missing = 0
        empty = 0
        for path in df["image_path"]:
            if not path or (isinstance(path, float) and np.isnan(path)):
                empty += 1
                continue
            if not os.path.isfile(str(path)):
                missing += 1

        report["missing_images"] = missing
        report["empty_image_paths"] = empty
        if missing > 0:
            report["valid"] = False
            report["error"] = f"{missing} images not found on disk"

    return report


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _cli():
    """Command-line interface for dataset generation."""
    import argparse

    # Load .env so MASSIVE_API_KEY is available without manual export
    try:
        from pathlib import Path

        from dotenv import load_dotenv  # type: ignore[import-untyped]

        _env_path = Path(__file__).resolve().parents[1] / ".env"
        if _env_path.is_file():
            load_dotenv(_env_path, override=False)
    except ImportError:
        pass  # python-dotenv not installed — rely on shell env or Docker

    parser = argparse.ArgumentParser(
        description="Generate labeled chart dataset for CNN training",
    )
    sub = parser.add_subparsers(dest="command")

    # Generate
    gen_parser = sub.add_parser("generate", help="Generate dataset")
    gen_parser.add_argument(
        "--symbols",
        nargs="+",
        required=True,
        help="Symbols to process (e.g. MGC MES MNQ)",
    )
    gen_parser.add_argument("--days", type=int, default=120, help="Days of history (v8 recipe: 120)")
    gen_parser.add_argument("--output-dir", default="dataset", help="Output directory")
    gen_parser.add_argument("--image-dir", default="dataset/images", help="Image output directory")
    gen_parser.add_argument("--source", default="massive", choices=["massive", "db", "cache", "csv"])
    gen_parser.add_argument("--csv-bars-dir", default="data/bars", help="Directory for CSV bar files")
    gen_parser.add_argument("--window-size", type=int, default=240)
    gen_parser.add_argument("--step-size", type=int, default=30)
    gen_parser.add_argument("--max-per-label", type=int, default=0, help="Max samples per label (0=unlimited)")
    gen_parser.add_argument("--dpi", type=int, default=150)
    gen_parser.add_argument("--no-skip", action="store_true", help="Re-render existing images")
    gen_parser.add_argument(
        "--session",
        default="all",
        choices=[
            "us",
            "london",
            "all",
            "frankfurt",
            "tokyo",
            "shanghai",
            "cme",
            "sydney",
            "london_ny",
            "cme_settle",
        ],
        help="ORB session: 'all' (default, v8 recipe), 'us', 'london', or a specific session key",
    )
    gen_parser.add_argument(
        "--parity-renderer",
        action="store_true",
        default=True,
        help="Use the Pillow renderer (chart_renderer_parity.py) — DEFAULT. "
        "Produces consistent 224×224 images for CNN training and inference.",
    )
    gen_parser.add_argument(
        "--no-parity-renderer",
        dest="parity_renderer",
        action="store_false",
        help="Use the original mplfinance renderer instead of the Pillow renderer. "
        "Not recommended for CNN training (produces visually different images).",
    )
    gen_parser.add_argument(
        "--breakout-type",
        default="all",
        choices=[
            "ORB",
            "PrevDay",
            "InitialBalance",
            "Consolidation",
            "Weekly",
            "Monthly",
            "Asian",
            "BollingerSqueeze",
            "ValueArea",
            "InsideDay",
            "GapRejection",
            "PivotPoints",
            "Fibonacci",
            "all",
        ],
        help=(
            "Breakout type to generate (default: 'all', v8 recipe). "
            "Use 'all' to iterate all 13 types. "
            "Controls box style on chart and the breakout_type_ord tabular feature."
        ),
    )
    gen_parser.add_argument(
        "--max-per-type",
        type=int,
        default=800,
        help=(
            "Maximum samples per (label, breakout_type) bucket when --breakout-type=all. "
            "v8 recipe default: 800. Set 0 for unlimited."
        ),
    )
    gen_parser.add_argument(
        "--max-per-session",
        type=int,
        default=400,
        help=("Maximum samples per (label, session) bucket. v8 recipe default: 400. Set 0 for unlimited."),
    )

    # Split
    split_parser = sub.add_parser("split", help="Split dataset into train/val")
    split_parser.add_argument("--csv", required=True, help="Path to dataset CSV")
    split_parser.add_argument(
        "--output-dir", default=None, help="Output directory for split CSVs (default: same as input CSV)"
    )
    split_parser.add_argument("--val-frac", type=float, default=0.15)
    split_parser.add_argument("--seed", type=int, default=42)

    # Validate
    val_parser = sub.add_parser("validate", help="Validate dataset")
    val_parser.add_argument("--csv", required=True, help="Path to dataset CSV")
    val_parser.add_argument("--no-check-images", action="store_true")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    if args.command == "generate":
        from lib.core.breakout_types import BreakoutType, breakout_type_from_name

        # Resolve breakout type(s)
        _bt_arg = getattr(args, "breakout_type", "ORB")
        _breakout_types = list(BreakoutType) if _bt_arg == "all" else [breakout_type_from_name(_bt_arg)]

        # Build a single DatasetConfig using the resolved breakout type(s).
        # If the user passed "--breakout-type all" we pass "all" directly so
        # _run_simulators_for_breakout_type handles all four types in one pass.
        _bt_config_str = _bt_arg  # "all" or e.g. "PrevDay"
        _max_per_type = getattr(args, "max_per_type", 0)
        _max_per_session = getattr(args, "max_per_session", 0)

        cfg = DatasetConfig(
            output_dir=args.output_dir,
            image_dir=args.image_dir,
            window_size=args.window_size,
            step_size=args.step_size,
            max_samples_per_label=args.max_per_label,
            max_samples_per_type_label=_max_per_type,
            max_samples_per_session_label=_max_per_session,
            chart_dpi=args.dpi,
            skip_existing=not args.no_skip,
            bars_source=args.source,
            csv_bars_dir=args.csv_bars_dir,
            orb_session=args.session,
            use_parity_renderer=args.parity_renderer,
            breakout_type=_bt_config_str,
        )
        logger.info(
            "Generating dataset for breakout_type=%s max_per_type=%d max_per_session=%d ...",
            _bt_config_str,
            _max_per_type,
            _max_per_session,
        )
        stats = generate_dataset(
            symbols=args.symbols,
            days_back=args.days,
            config=cfg,
        )
        if stats:
            print(f"\n{stats.summary()}")

    elif args.command == "split":
        train_path, val_path = split_dataset(
            csv_path=args.csv,
            output_dir=args.output_dir,
            val_fraction=args.val_frac,
            random_seed=args.seed,
        )
        print(f"\nTrain: {train_path}")
        print(f"Val:   {val_path}")

    elif args.command == "validate":
        report = validate_dataset(
            csv_path=args.csv,
            check_images=not args.no_check_images,
        )
        print("\nDataset validation report:")
        for k, v in report.items():
            print(f"  {k}: {v}")

    else:
        parser.print_help()


if __name__ == "__main__":
    _cli()
