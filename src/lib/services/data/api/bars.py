"""
Historical Bars API Router
==========================
Serves stored OHLCV bars from Postgres/SQLite to any internal consumer
(engine, CNN trainer, web dashboard, backtest scripts) via clean REST
endpoints.  When data is missing or stale, the service detects the gaps
and automatically populates them from Massive before returning.

This is the **single source of truth** for bar data within the platform:
  - Engine uses it to seed ORB detection and backtesting.
  - CNN dataset generator uses it for chart image generation.
  - Web dashboard uses it for live chart display.
  - Clients never need their own Massive / yfinance connections.

Endpoints
---------
GET  /bars/{symbol}
    Return stored 1-minute bars for a symbol, ensuring data quality
    by auto-filling any detected gaps from Massive before responding.

POST /bars/bulk
    Same as above for multiple symbols in one call.

GET  /bars/{symbol}/gaps
    Return a structured gap report (missing ranges, coverage %).

POST /bars/{symbol}/fill
    Trigger an immediate incremental fill for a specific symbol and
    optional date range without running the full nightly backfill.

POST /bars/fill/all
    Trigger incremental fill for ALL enabled assets (non-blocking, runs
    in a background thread, returns a job-id that can be polled).

GET  /bars/fill/status
    Poll the status of a running fill-all job.

GET  /bars/status
    Bar counts, date ranges, and coverage for all stored symbols.

GET  /bars/assets
    List the enabled assets (from models.ASSETS) with their Massive
    product codes and bar availability.

Design Notes
------------
- **Gap detection** is done by comparing the latest stored timestamp
  against ``now – threshold``.  If the gap exceeds ``stale_minutes``
  (default 5 for 1m bars in live hours, 60 outside market hours) an
  incremental fill is triggered automatically on ``GET /bars/{symbol}``.

- **Auto-fill** uses the same chunked Massive → yfinance fallback pipeline
  as the nightly backfill, but targets only the missing window.

- **Non-blocking fills** (``fill/all``) run in a ``ThreadPoolExecutor``
  daemon thread.  The endpoint returns immediately with a job token.

- **Caching**: after a fill, the fresh bars are also written into Redis
  under ``engine:bars_1m_hist:{symbol}`` so the dataset generator and
  engine can read them without a DB round-trip.

- All response payloads use ``split`` orientation (compatible with
  ``pd.DataFrame(**payload)`` on the client side) for compact, typed
  serialisation.
"""

from __future__ import annotations

import contextlib
import logging
import os
import threading
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from typing import Any, Literal

import pandas as pd
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel

logger = logging.getLogger("api.bars")

router = APIRouter(tags=["Bars"])

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# How stale (in minutes) a symbol's latest bar must be before an auto-fill
# is triggered on a GET request.  During market hours we want fresh data
# quickly; outside hours a 60-minute threshold avoids hammering the API.
_AUTO_FILL_STALE_MINUTES = int(os.getenv("BARS_AUTO_FILL_STALE_MINUTES", "5"))

# Maximum rows returned in a single /bars/{symbol} response.  Clients that
# need more should page using start/end query params.
_MAX_ROWS = int(os.getenv("BARS_MAX_ROWS", "50000"))

# Thread pool for non-blocking fill-all jobs (1 worker: fills are sequential
# per symbol anyway, and we don't want to saturate the Massive API).
_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="bars-fill")

# Global registry of running / completed fill-all jobs.
# key = job_id (str UUID), value = _FillJob
_fill_jobs: dict[str, _FillJob] = {}
_fill_jobs_lock = threading.Lock()

# Keep at most this many completed jobs in memory to avoid unbounded growth.
_MAX_COMPLETED_JOBS = 20

# ---------------------------------------------------------------------------
# Per-symbol fill job registry
# ---------------------------------------------------------------------------
# Tracks the most recent async fill triggered by GET /bars/{symbol}.
# key = normalised symbol string, value = _SymbolFillJob
# This lets clients poll /bars/{symbol}/fill/status while a fill is running.
_symbol_fills: dict[str, _SymbolFillJob] = {}
_symbol_fills_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Internal job tracker
# ---------------------------------------------------------------------------


class _FillJob:
    """Tracks a running or completed fill-all background job."""

    def __init__(self, job_id: str, symbols: list[str]):
        self.job_id: str = job_id
        self.symbols: list[str] = symbols
        self.started_at: str = datetime.now(tz=UTC).isoformat()
        self.finished_at: str | None = None
        self.status: str = "running"  # running | complete | partial | failed
        self.progress: int = 0  # 0-100
        self.results: list[dict[str, Any]] = []
        self.errors: list[str] = []
        self.total_bars_added: int = 0
        self._future: Future[None] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "symbols": self.symbols,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "status": self.status,
            "progress": self.progress,
            "total_bars_added": self.total_bars_added,
            "symbol_count": len(self.symbols),
            "completed_count": len(self.results),
            "errors": self.errors[:20],
        }


class _SymbolFillJob:
    """Tracks the most-recent async fill triggered for a single symbol.

    Created when ``GET /bars/{symbol}`` fires a background fill instead of
    blocking.  Clients can poll ``GET /bars/{symbol}/fill/status`` to learn
    when the fill completes and then re-fetch the bars.
    """

    def __init__(self, symbol: str, days_back: int, interval: str):
        self.job_id: str = str(uuid.uuid4())
        self.symbol: str = symbol
        self.days_back: int = days_back
        self.interval: str = interval
        self.started_at: str = datetime.now(tz=UTC).isoformat()
        self.finished_at: str | None = None
        self.status: Literal["running", "complete", "failed"] = "running"
        self.bars_added: int = 0
        self.error: str | None = None
        self._future: Future[None] | None = None

    @property
    def is_running(self) -> bool:
        return self.status == "running"

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "symbol": self.symbol,
            "days_back": self.days_back,
            "interval": self.interval,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "status": self.status,
            "bars_added": self.bars_added,
            "error": self.error,
            "poll_url": f"/bars/{self.symbol}/fill/status",
        }


def _run_symbol_fill_job(job: _SymbolFillJob) -> None:
    """Worker executed in the thread pool for a single-symbol async fill.

    Updates *job* in place so callers polling ``to_dict()`` always see the
    latest state.  Registers the completed result back into
    ``_symbol_fills`` so the status endpoint can return it after the
    ``Future`` has resolved.
    """
    try:
        result = _run_incremental_fill(job.symbol, days_back=job.days_back, interval=job.interval)
        job.bars_added = result.get("bars_added", 0)
        err = result.get("error")
        if err:
            job.error = str(err)
            job.status = "failed"
            logger.warning("Async fill failed for %s: %s", job.symbol, err)
        else:
            job.status = "complete"
            logger.info(
                "Async fill complete for %s: +%d bars",
                job.symbol,
                job.bars_added,
            )
    except Exception as exc:
        job.error = str(exc)
        job.status = "failed"
        logger.error("Async fill raised for %s: %s", job.symbol, exc)
    finally:
        job.finished_at = datetime.now(tz=UTC).isoformat()


def _get_or_start_symbol_fill(symbol: str, days_back: int, interval: str) -> _SymbolFillJob:
    """Return the current running fill for *symbol*, or start a new one.

    If a fill is already running for the symbol, the existing job is
    returned so the caller can report its status without spawning a
    duplicate fill.  If the previous job finished (complete or failed) a
    new job is created and submitted to the thread pool.
    """
    with _symbol_fills_lock:
        existing = _symbol_fills.get(symbol)
        if existing is not None and existing.is_running:
            logger.debug("Fill for %s already running (job %s) — reusing", symbol, existing.job_id)
            return existing

        job = _SymbolFillJob(symbol=symbol, days_back=days_back, interval=interval)
        _symbol_fills[symbol] = job

    # Submit outside the lock so the worker can acquire it if needed
    job._future = _executor.submit(_run_symbol_fill_job, job)
    logger.info(
        "Async fill started for %s (job %s, %d days back)",
        symbol,
        job.job_id,
        days_back,
    )
    return job


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class BulkBarsRequest(BaseModel):
    """Request body for ``POST /bars/bulk``."""

    symbols: list[str]
    interval: str = "1m"
    days_back: int = 30
    auto_fill: bool = True


class FillRequest(BaseModel):
    """Request body for ``POST /bars/{symbol}/fill``."""

    days_back: int = 30
    interval: str = "1m"
    chunk_days: int = 5


class FillAllRequest(BaseModel):
    """Request body for ``POST /bars/fill/all``."""

    symbols: list[str] | None = None  # None = use ASSETS
    days_back: int = 30
    interval: str = "1m"


# ---------------------------------------------------------------------------
# Helpers: database interaction
# ---------------------------------------------------------------------------


def _get_conn():
    """Return a DB connection (Postgres or SQLite)."""
    try:
        from lib.core.models import _get_conn as _models_conn

        return _models_conn()
    except ImportError:
        import sqlite3

        db_path = os.getenv("DB_PATH", "futures_journal.db")
        conn = sqlite3.connect(db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn


def _is_postgres() -> bool:
    try:
        from lib.core.models import _is_using_postgres

        return _is_using_postgres()
    except ImportError:
        return False


def _ph() -> str:
    return "%s" if _is_postgres() else "?"


def _ensure_table() -> None:
    """Make sure historical_bars table exists (idempotent)."""
    try:
        from lib.services.engine.backfill import init_backfill_table

        init_backfill_table()
    except Exception as exc:
        logger.debug("Could not ensure backfill table: %s", exc)


# ---------------------------------------------------------------------------
# Symbol normalisation
# ---------------------------------------------------------------------------

# Kraken spot crypto symbols — these are NOT futures and must never get =F
_KRAKEN_PREFIXES = ("KRAKEN:",)
_KRAKEN_SPOT_SYMBOLS = frozenset(
    [
        "BTC",
        "ETH",
        "SOL",
        "XBT",
        "ADA",
        "AVAX",
        "DOT",
        "POL",
        "LINK",
        "UNI",
        "AAVE",
        "ALGO",
        "ATOM",
        "XRP",
        "LTC",
        "BCH",
    ]
)

# CME/CBOT futures short names that arrive without the =F suffix.
# All symbols in the Massive product map that end with =F are eligible;
# we keep an explicit set here so we never accidentally append =F to
# a Kraken spot ticker or a symbol that's already normalised.
_FUTURES_SHORT_NAMES = frozenset(
    [
        "MGC",
        "SIL",
        "MHG",
        "MCL",
        "MNG",
        "MES",
        "MNQ",
        "M2K",
        "MYM",
        "6E",
        "6B",
        "6J",
        "6A",
        "6C",
        "6S",
        "M6E",
        "M6B",
        "M6J",
        "ZN",
        "ZB",
        "ZC",
        "ZS",
        "ZW",
        "ZF",
        "ZT",
        "ZL",
        "ZM",
        "MBT",
        "MET",
        "ES",
        "NQ",
        "GC",
        "SI",
        "HG",
        "CL",
        "NG",
        "RTY",
        "YM",
        "BTC=F",
        "ETH=F",  # CME crypto futures (distinct from Kraken spot)
    ]
)


def _normalize_symbol(symbol: str) -> str:
    """Normalise a caller-supplied symbol to the canonical =F ticker form.

    The /bars endpoints accept both short names (e.g. ``MES``) and fully
    qualified Yahoo-style tickers (e.g. ``MES=F``).  All internal helpers
    (backfill, Massive resolver, DB queries) expect the ``=F`` form for
    CME/CBOT futures.  Kraken spot symbols (BTC, ETH, SOL, KRAKEN:XBTUSD)
    are returned unchanged.

    Examples
    --------
    >>> _normalize_symbol("MES")   -> "MES=F"
    >>> _normalize_symbol("MES=F") -> "MES=F"
    >>> _normalize_symbol("BTC")   -> "BTC"        (Kraken spot)
    >>> _normalize_symbol("KRAKEN:XBTUSD") -> "KRAKEN:XBTUSD"
    """
    # Already has =F — return as-is
    if "=" in symbol:
        return symbol

    # Explicit Kraken prefix — never a futures ticker
    for prefix in _KRAKEN_PREFIXES:
        if symbol.upper().startswith(prefix.upper()):
            return symbol

    # Known Kraken spot names — do not append =F
    if symbol.upper() in _KRAKEN_SPOT_SYMBOLS:
        return symbol

    # Known CME/CBOT futures short name — append =F
    if symbol.upper() in _FUTURES_SHORT_NAMES:
        return f"{symbol.upper()}=F"

    # Unknown symbol: try to detect via Massive resolver; if it resolves,
    # it's a futures ticker.  Fall back to appending =F so we at least try.
    try:
        from lib.core.cache import _get_massive_provider

        provider = _get_massive_provider()
        if provider is not None and provider.is_available and provider.resolve_from_yahoo(f"{symbol.upper()}=F"):
            return f"{symbol.upper()}=F"
    except Exception:
        pass

    # Default: assume futures
    return f"{symbol.upper()}=F"


def _fetch_stored_bars(
    symbol: str,
    interval: str = "1m",
    days_back: int = 30,
    start_dt: datetime | None = None,
    end_dt: datetime | None = None,
) -> pd.DataFrame:
    """Query historical_bars from the database.

    Returns a DataFrame with columns [Open, High, Low, Close, Volume]
    and a UTC-aware DatetimeIndex, or an empty DataFrame on failure.
    """
    ph = _ph()
    conn = None
    try:
        conn = _get_conn()
        now = datetime.now(tz=UTC)
        if end_dt is None:
            end_dt = now
        if start_dt is None:
            start_dt = now - timedelta(days=days_back)

        sql = f"""
            SELECT timestamp, open, high, low, close, volume
            FROM historical_bars
            WHERE symbol = {ph} AND interval = {ph}
              AND timestamp >= {ph} AND timestamp <= {ph}
            ORDER BY timestamp ASC
            LIMIT {_MAX_ROWS}
        """
        cur = conn.execute(sql, (symbol, interval, start_dt.isoformat(), end_dt.isoformat()))
        rows = cur.fetchall()
        if not rows:
            return pd.DataFrame()

        data = []
        for row in rows:
            if isinstance(row, (tuple, list)):
                ts, o, h, lo, c, v = row
            else:
                ts = row["timestamp"]
                o = row["open"]
                h = row["high"]
                lo = row["low"]
                c = row["close"]
                v = row["volume"]
            data.append(
                {
                    "timestamp": ts,
                    "Open": float(o),
                    "High": float(h),
                    "Low": float(lo),
                    "Close": float(c),
                    "Volume": int(v or 0),
                }
            )

        df = pd.DataFrame(data)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.set_index("timestamp").sort_index()
        return df

    except Exception as exc:
        # Downgrade "no such table" to DEBUG — this fires transiently on a
        # fresh deployment before init_backfill_table() has run, or on the
        # GPU trainer machine which has no local DB.  It is not a true error
        # in those contexts; callers handle the empty return gracefully.
        _exc_str = str(exc).lower()
        if "no such table" in _exc_str or "does not exist" in _exc_str:
            logger.debug("historical_bars table not present for %s: %s", symbol, exc)
        else:
            logger.error("Failed to fetch stored bars for %s: %s", symbol, exc)
        return pd.DataFrame()
    finally:
        if conn is not None:
            with contextlib.suppress(Exception):
                conn.close()


def _get_latest_stored_ts(symbol: str, interval: str = "1m") -> datetime | None:
    """Return the most-recent stored bar timestamp for a symbol, or None."""
    ph = _ph()
    conn = None
    try:
        conn = _get_conn()
        sql = f"SELECT MAX(timestamp) FROM historical_bars WHERE symbol = {ph} AND interval = {ph}"
        cur = conn.execute(sql, (symbol, interval))
        row = cur.fetchone()
        if row:
            val = row[0] if isinstance(row, (tuple, list)) else row["max"]
            if val:
                ts = pd.Timestamp(val)
                if ts is pd.NaT:
                    return None
                if ts.tzinfo is None:
                    ts = ts.tz_localize("UTC")
                pdt = ts.to_pydatetime()
                if isinstance(pdt, datetime):
                    return pdt
    except Exception as exc:
        logger.debug("latest_ts query failed for %s: %s", symbol, exc)
    finally:
        if conn is not None:
            with contextlib.suppress(Exception):
                conn.close()
    return None


def _bar_count(symbol: str, interval: str = "1m") -> int:
    ph = _ph()
    conn = None
    try:
        conn = _get_conn()
        sql = f"SELECT COUNT(*) FROM historical_bars WHERE symbol = {ph} AND interval = {ph}"
        cur = conn.execute(sql, (symbol, interval))
        row = cur.fetchone()
        if row:
            return int(row[0] if isinstance(row, (tuple, list)) else row["count"])
    except Exception:
        pass
    finally:
        if conn is not None:
            with contextlib.suppress(Exception):
                conn.close()
    return 0


# ---------------------------------------------------------------------------
# Helpers: staleness + gap detection
# ---------------------------------------------------------------------------


def _is_market_hours() -> bool:
    """Rough check: are we inside CME futures regular trading hours?"""
    from zoneinfo import ZoneInfo

    _EST = ZoneInfo("America/New_York")
    now = datetime.now(tz=_EST)
    # CME Globex: Sun 18:00 – Fri 17:00 ET, with a daily 17:00-18:00 break.
    # We approximate: weekday 0-4, hour not in [17].
    if now.weekday() == 6 and now.hour < 18:  # Sunday before open
        return False
    if now.weekday() == 5:  # Saturday — always closed
        return False
    return now.hour != 17  # daily maintenance break at 17:00


def _stale_minutes() -> int:
    """Return the staleness threshold in minutes based on market hours."""
    if _is_market_hours():
        return _AUTO_FILL_STALE_MINUTES
    return 60  # outside market hours, 60-min threshold is fine


def _needs_fill(symbol: str, interval: str = "1m") -> tuple[bool, int]:
    """Determine if a symbol needs a gap-fill fetch.

    Returns (needs_fill: bool, gap_minutes: int).
    ``gap_minutes`` is the number of minutes since the last stored bar.
    """
    latest = _get_latest_stored_ts(symbol, interval)
    if latest is None:
        return True, 999_999  # no data at all — definitely needs fill

    now = datetime.now(tz=UTC)
    gap_minutes = int((now - latest).total_seconds() / 60)
    threshold = _stale_minutes()
    return gap_minutes > threshold, gap_minutes


def _compute_gap_windows(
    symbol: str,
    interval: str = "1m",
    days_back: int = 30,
) -> list[dict[str, str]]:
    """Return a list of {start, end} gap windows in the stored data.

    A gap is any period > 5 minutes between consecutive bars that is not
    a weekend or the daily CME maintenance break (17:00–18:00 ET).
    """
    from zoneinfo import ZoneInfo

    _EST = ZoneInfo("America/New_York")

    df = _fetch_stored_bars(symbol, interval=interval, days_back=days_back)
    if df.empty or len(df) < 2:
        return []

    gaps: list[dict[str, Any]] = []
    prev = None
    for ts in df.index:
        if prev is not None:
            diff_min = float((ts - prev).total_seconds()) / 60.0
            if diff_min > 5:
                # Skip weekends
                prev_et = prev.astimezone(_EST)
                if prev_et.weekday() == 4 and diff_min > 2880:  # Friday → Monday
                    prev = ts
                    continue
                # Skip daily maintenance break (approx 60–65 min gap starting ~17:00 ET)
                if prev_et.hour == 17 and diff_min < 70:
                    prev = ts
                    continue
                gaps.append(
                    {
                        "start": str(prev.isoformat()),
                        "end": str(ts.isoformat()),
                        "missing_minutes": int(diff_min),
                    }
                )
        prev = ts

    return gaps


# ---------------------------------------------------------------------------
# Helpers: incremental fill (wraps backfill module)
# ---------------------------------------------------------------------------


def _run_incremental_fill(
    symbol: str,
    days_back: int = 30,
    interval: str = "1m",
    chunk_days: int = 5,
    start_dt: datetime | None = None,
    end_dt: datetime | None = None,
) -> dict[str, Any]:
    """Run an incremental gap-fill for a single symbol.

    If ``start_dt`` / ``end_dt`` are provided, only that window is fetched.
    Otherwise, the standard backfill gap-detection logic is used (starts
    from 1 minute after the latest stored bar).

    Returns a result dict compatible with ``backfill_symbol()``.
    """
    _ensure_table()

    try:
        from lib.services.engine.backfill import (
            _compute_date_range,
            _fetch_chunk_massive,
            _fetch_chunk_yfinance,
            _generate_chunks,
            _get_bar_count,
            _store_bars,
            _symbol_display_name,
        )
        from lib.services.engine.backfill import (
            _get_conn as _bf_conn,
        )
    except ImportError as exc:
        logger.error("Cannot import backfill helpers: %s", exc)
        return {"symbol": symbol, "bars_added": 0, "error": str(exc)}

    name = _symbol_display_name(symbol)
    t0 = time.monotonic()

    result: dict[str, Any] = {
        "symbol": symbol,
        "name": name,
        "bars_before": 0,
        "bars_after": 0,
        "bars_added": 0,
        "chunks_fetched": 0,
        "chunks_with_data": 0,
        "start_date": "",
        "end_date": "",
        "duration_seconds": 0.0,
        "error": "",
    }

    conn = None
    try:
        conn = _bf_conn()
        result["bars_before"] = _get_bar_count(conn, symbol, interval)

        # Determine date range
        if start_dt is not None and end_dt is not None:
            s_dt, e_dt = start_dt, end_dt
        else:
            s_dt, e_dt = _compute_date_range(symbol, conn, days_back=days_back, interval=interval)

        result["start_date"] = s_dt.strftime("%Y-%m-%d")
        result["end_date"] = e_dt.strftime("%Y-%m-%d")

        if s_dt >= e_dt:
            logger.info("%s (%s): already up to date (%d bars)", name, symbol, result["bars_before"])
            result["bars_after"] = result["bars_before"]
            return result

        chunks = _generate_chunks(s_dt, e_dt, chunk_days)
        result["chunks_fetched"] = len(chunks)

        logger.info(
            "🔄 Incremental fill: %s (%s) — %d chunks from %s to %s",
            name,
            symbol,
            len(chunks),
            s_dt.strftime("%Y-%m-%d"),
            e_dt.strftime("%Y-%m-%d"),
        )

        total_new = 0
        for i, (c_start, c_end) in enumerate(chunks):
            try:
                # Try Massive first, then yfinance
                df = _fetch_chunk_massive(symbol, c_start, c_end)
                if df.empty:
                    df = _fetch_chunk_yfinance(symbol, c_start, c_end)

                if not df.empty:
                    new_bars = _store_bars(conn, symbol, df, interval)
                    total_new += new_bars
                    result["chunks_with_data"] += 1
                    logger.debug(
                        "  chunk %d/%d: +%d new bars (%d fetched)",
                        i + 1,
                        len(chunks),
                        new_bars,
                        len(df),
                    )
            except Exception as exc:
                logger.warning("  chunk %d/%d failed for %s: %s", i + 1, len(chunks), symbol, exc)

        result["bars_added"] = total_new
        result["bars_after"] = _get_bar_count(conn, symbol, interval)

        logger.info("✅ Fill complete: %s +%d bars (total: %d)", symbol, total_new, result["bars_after"])

        # Publish fresh bars into Redis so the dataset generator can read them
        # without a DB round-trip on the next call.
        if total_new > 0:
            _warm_redis_cache(symbol, interval)

    except Exception as exc:
        result["error"] = str(exc)
        logger.error("❌ Incremental fill failed for %s: %s", symbol, exc)
    finally:
        if conn is not None:
            with contextlib.suppress(Exception):
                conn.close()
        result["duration_seconds"] = round(time.monotonic() - t0, 2)

    return result


def _warm_redis_cache(symbol: str, interval: str = "1m", days_back: int = 7) -> None:
    """Write the most-recent N days of stored bars into Redis.

    This populates ``engine:bars_1m_hist:{symbol}`` so that the dataset
    generator's ``_load_bars_from_cache`` legacy key path can find the data
    without another DB query.
    """
    try:
        from lib.core.cache import REDIS_AVAILABLE, _r

        if not REDIS_AVAILABLE or _r is None:
            return

        df = _fetch_stored_bars(symbol, interval=interval, days_back=days_back)
        if df.empty:
            return

        key = f"engine:bars_1m_hist:{symbol}"
        json_str = df.to_json(date_format="iso")
        if json_str is None:
            return
        payload = json_str.encode()
        # 8-hour TTL — the nightly backfill will refresh anyway
        _r.setex(key, 28_800, payload)
        logger.debug("Warmed Redis cache for %s: %d bars → %s", symbol, len(df), key)
    except Exception as exc:
        logger.debug("Redis cache warm failed for %s: %s", symbol, exc)


# ---------------------------------------------------------------------------
# Helpers: serialisation
# ---------------------------------------------------------------------------


def _df_to_split(df: pd.DataFrame) -> dict[str, Any] | None:
    """Serialise DataFrame to split-orientation dict (compact + typed).

    The client can reconstruct via ``pd.DataFrame(**payload)``.
    Returns None if the DataFrame is None or empty.
    """
    if df is None or df.empty:
        return None
    # Convert index to strings for JSON safety
    out = df.copy()
    out.index = out.index.astype(str)
    raw: dict[str, Any] = out.to_dict(orient="split")  # type: ignore[assignment]
    return raw


def _get_enabled_assets() -> dict[str, str]:
    """Return {name: ticker} for all enabled assets (from models.ASSETS)."""
    try:
        from lib.core.models import ASSETS

        return dict(ASSETS)
    except ImportError:
        return {
            "Gold": "MGC=F",
            "S&P": "MES=F",
            "Nasdaq": "MNQ=F",
            "Crude Oil": "MCL=F",
        }


def _get_all_symbols() -> list[str]:
    """Return unique ticker list for all enabled assets."""
    return list(set(_get_enabled_assets().values()))


# ---------------------------------------------------------------------------
# Fill-all background job runner
# ---------------------------------------------------------------------------


def _run_fill_all_job(job: _FillJob, days_back: int, interval: str) -> None:
    """Worker function executed in the thread pool for fill/all jobs."""
    n = len(job.symbols)
    job.status = "running"

    for idx, symbol in enumerate(job.symbols):
        try:
            res = _run_incremental_fill(symbol, days_back=days_back, interval=interval)
            job.results.append(res)
            job.total_bars_added += res.get("bars_added", 0)
            if res.get("error"):
                job.errors.append(f"{symbol}: {res['error']}")
        except Exception as exc:
            job.errors.append(f"{symbol}: {exc}")
            logger.warning("fill/all: error for %s — %s", symbol, exc)

        job.progress = int((idx + 1) / n * 100)

    job.finished_at = datetime.now(tz=UTC).isoformat()
    job.status = "failed" if len(job.errors) == n else ("partial" if job.errors else "complete")
    logger.info(
        "fill/all job %s %s: +%d bars across %d symbols (%d errors)",
        job.job_id,
        job.status,
        job.total_bars_added,
        n,
        len(job.errors),
    )

    # Prune old completed jobs
    with _fill_jobs_lock:
        done = [jid for jid, j in _fill_jobs.items() if j.status != "running"]
        for old_id in done[:-_MAX_COMPLETED_JOBS]:
            _fill_jobs.pop(old_id, None)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("/bars/symbols")
def list_symbols() -> JSONResponse:
    """Return a lightweight symbol list for the trainer — no DB queries.

    Unlike ``GET /bars/assets`` this endpoint does **not** hit the database
    for bar counts, so it is fast and safe to call at training-job startup.

    Response:
    ```json
    {
      "symbols": ["6A", "6B", "6C", ...],
      "assets": [
        {"name": "Gold", "ticker": "MGC=F", "symbol": "MGC"},
        ...
      ],
      "total": 25
    }
    ```

    The ``symbol`` field is the short name the trainer and dataset generator
    use internally (e.g. ``"MGC"`` rather than the Yahoo ticker ``"MGC=F"``).
    The trainer calls this endpoint to discover which symbols the engine has
    mapped in ``models.ASSETS`` without needing a local copy of the models
    module.
    """
    assets = _get_enabled_assets()

    # Build a reverse map from Yahoo ticker → shortest short symbol using the
    # dataset generator's _SYMBOL_TO_TICKER mapping.
    ticker_to_short: dict[str, str] = {}
    try:
        from lib.services.training.dataset_generator import _SYMBOL_TO_TICKER

        for sym, tkr in _SYMBOL_TO_TICKER.items():
            if tkr not in ticker_to_short or len(sym) < len(ticker_to_short[tkr]):
                ticker_to_short[tkr] = sym
    except Exception:
        pass

    result = []
    short_symbols: list[str] = []
    for name, ticker in sorted(assets.items()):
        # Derive the short symbol: prefer the reverse-map lookup, then strip
        # "=F", then leave Kraken tickers as-is.
        if ticker in ticker_to_short:
            short = ticker_to_short[ticker]
        elif ticker.startswith("KRAKEN:"):
            short = ticker
        else:
            short = ticker.replace("=F", "")

        result.append({"name": name, "ticker": ticker, "symbol": short})
        if short not in short_symbols:
            short_symbols.append(short)

    short_symbols.sort()
    return JSONResponse(
        {
            "symbols": short_symbols,
            "assets": result,
            "total": len(short_symbols),
        }
    )


@router.get("/bars/assets")
def list_assets() -> JSONResponse:
    """List all enabled assets with their tickers and stored bar counts.

    Returns a list of asset descriptors including:
    - ``name``: human-readable asset name
    - ``ticker``: Yahoo-style ticker (e.g. ``MGC=F``)
    - ``bar_count``: number of 1-minute bars currently stored in the DB
    - ``latest``: ISO timestamp of the most-recent stored bar (or null)
    - ``has_data``: whether any bars are stored
    """
    _ensure_table()
    assets = _get_enabled_assets()
    result = []
    for name, ticker in sorted(assets.items()):
        count = _bar_count(ticker, "1m")
        latest_ts = _get_latest_stored_ts(ticker, "1m")
        result.append(
            {
                "name": name,
                "ticker": ticker,
                "bar_count": count,
                "latest": latest_ts.isoformat() if latest_ts else None,
                "has_data": count > 0,
            }
        )
    return JSONResponse({"assets": result, "total_symbols": len(result)})


@router.get("/bars/status")
def bars_status() -> JSONResponse:
    """Return bar counts, date ranges, and coverage for all stored symbols.

    Proxies the backfill module's ``get_backfill_status()`` and enriches
    each entry with a staleness flag so callers can see at a glance which
    symbols need attention.
    """
    _ensure_table()
    try:
        from lib.services.engine.backfill import get_backfill_status

        status = get_backfill_status()
    except Exception as exc:
        status = {"symbols": [], "total_bars": 0, "error": str(exc)}

    # Annotate each symbol with staleness info
    threshold = _stale_minutes()
    now = datetime.now(tz=UTC)
    raw_symbols = status.get("symbols")
    symbols_list: list[dict[str, Any]] = raw_symbols if isinstance(raw_symbols, list) else []
    for sym_info in symbols_list:
        latest_str = sym_info.get("latest") if isinstance(sym_info, dict) else None
        if latest_str:
            try:
                latest = pd.Timestamp(latest_str)
                if latest.tzinfo is None:
                    latest = latest.tz_localize("UTC")
                gap_min = int((now - latest.to_pydatetime()).total_seconds() / 60)
                sym_info["gap_minutes"] = gap_min
                sym_info["is_stale"] = gap_min > threshold
            except Exception:
                sym_info["gap_minutes"] = None
                sym_info["is_stale"] = None
        else:
            sym_info["gap_minutes"] = None
            sym_info["is_stale"] = True

    return JSONResponse(
        {
            **status,
            "stale_threshold_minutes": threshold,
            "market_hours": _is_market_hours(),
            "timestamp": now.isoformat(),
        }
    )


@router.get("/bars/fill/status")
def fill_all_status(job_id: str | None = Query(default=None)) -> JSONResponse:
    """Poll the status of a fill-all background job.

    Pass ``?job_id=<uuid>`` to get a specific job.  Omit to get the most
    recent job (if any).

    Returns a job descriptor dict with ``status``, ``progress``,
    ``total_bars_added``, ``errors``, etc.
    """
    with _fill_jobs_lock:
        if job_id:
            job = _fill_jobs.get(job_id)
            if job is None:
                raise HTTPException(status_code=404, detail=f"Job {job_id!r} not found")
            return JSONResponse(job.to_dict())

        if not _fill_jobs:
            return JSONResponse({"status": "no_jobs", "message": "No fill jobs have been submitted yet"})

        # Most recent job
        latest_job = list(_fill_jobs.values())[-1]
        return JSONResponse(latest_job.to_dict())


@router.get("/bars/{symbol}/gaps")
def get_gaps(
    symbol: str,
    days_back: int = Query(default=30, ge=1, le=365),
    interval: str = Query(default="1m"),
) -> JSONResponse:
    """Return a structured gap report for a specific symbol.

    Includes:
    - ``total_bars``: bars currently stored
    - ``expected_bars``: estimated bars for the requested window
    - ``coverage_pct``: percentage of expected bars present
    - ``gaps``: list of ``{start, end, missing_minutes}`` gap windows
    - ``needs_fill``: whether auto-fill would be triggered right now

    The ``symbol`` value should be a Yahoo-style ticker, e.g. ``MGC%3DF``
    (URL-encode the ``=`` as ``%3D``).
    """
    symbol = _normalize_symbol(symbol)
    _ensure_table()
    try:
        from lib.services.engine.backfill import get_gap_report

        report = get_gap_report(symbol, days_back=days_back, interval=interval)
    except Exception as exc:
        logger.error("Gap report failed for %s: %s", symbol, exc)
        report = {
            "symbol": symbol,
            "total_bars": _bar_count(symbol, interval),
            "expected_bars": 0,
            "coverage_pct": 0.0,
            "gaps": _compute_gap_windows(symbol, interval, days_back),
        }

    needs, gap_min = _needs_fill(symbol, interval)
    report["needs_fill"] = needs
    report["gap_since_latest_minutes"] = gap_min
    report["timestamp"] = datetime.now(tz=UTC).isoformat()
    return JSONResponse(report)


@router.get("/bars/{symbol}/fill/status")
def symbol_fill_status(symbol: str) -> JSONResponse:
    """Return the status of the most-recent async fill for a single symbol.

    This endpoint is polled by dataset generators and other clients after
    receiving a ``filling: true`` flag from ``GET /bars/{symbol}``.

    Response when a fill is in progress::

        {"status": "running", "symbol": "MGC=F", "bars_added": 0, ...}

    Response when complete::

        {"status": "complete", "symbol": "MGC=F", "bars_added": 1234, ...}

    Response when no fill has been triggered yet::

        {"status": "no_fill", "symbol": "MGC=F"}
    """
    symbol = _normalize_symbol(symbol)
    with _symbol_fills_lock:
        job = _symbol_fills.get(symbol)

    if job is None:
        return JSONResponse({"status": "no_fill", "symbol": symbol})
    return JSONResponse(job.to_dict())


@router.post("/bars/{symbol}/fill")
def fill_symbol(symbol: str, req: FillRequest) -> JSONResponse:
    symbol = _normalize_symbol(symbol)
    """Trigger an immediate incremental fill for a specific symbol.

    This call **blocks** until the fill is complete and returns a summary
    of what was fetched.  Use ``POST /bars/fill/all`` for non-blocking
    bulk fills.

    The fill:
    1. Queries the latest stored bar for ``symbol``.
    2. Fetches the missing window from Massive (or yfinance fallback).
    3. Upserts the new bars into Postgres / SQLite.
    4. Warms the Redis cache so downstream consumers see fresh data.
    """
    symbol = _normalize_symbol(symbol)
    _ensure_table()
    result = _run_incremental_fill(
        symbol,
        days_back=req.days_back,
        interval=req.interval,
        chunk_days=req.chunk_days,
    )
    status_code = 200 if not result.get("error") else 500
    return JSONResponse(content=result, status_code=status_code)


@router.post("/bars/fill/all")
def fill_all(req: FillAllRequest) -> JSONResponse:
    """Trigger a non-blocking incremental fill for all (or specified) assets.

    Returns immediately with a ``job_id`` that can be polled via
    ``GET /bars/fill/status?job_id=<uuid>``.

    The job runs sequentially symbol-by-symbol in a background thread to
    avoid saturating the Massive API.
    """
    _ensure_table()

    symbols = req.symbols if req.symbols else _get_all_symbols()
    if not symbols:
        raise HTTPException(status_code=400, detail="No symbols to fill — check ASSETS configuration")

    job_id = str(uuid.uuid4())
    job = _FillJob(job_id=job_id, symbols=symbols)

    with _fill_jobs_lock:
        _fill_jobs[job_id] = job

    days_back = req.days_back
    interval = req.interval

    def _run():
        _run_fill_all_job(job, days_back=days_back, interval=interval)

    job._future = _executor.submit(_run)

    logger.info("fill/all job %s queued: %d symbols, %d days back", job_id, len(symbols), days_back)
    return JSONResponse(
        {
            "job_id": job_id,
            "status": "queued",
            "symbols": symbols,
            "symbol_count": len(symbols),
            "days_back": days_back,
            "interval": interval,
            "poll_url": f"/bars/fill/status?job_id={job_id}",
        },
        status_code=202,
    )


@router.get("/bars/{symbol}")
def get_bars(
    symbol: str,
    interval: str = Query(default="1m"),
    days_back: int = Query(default=30, ge=1, le=365),
    auto_fill: bool = Query(default=True, description="Auto-fill gaps from Massive before returning data"),
    start: str | None = Query(default=None, description="ISO start timestamp (overrides days_back)"),
    end: str | None = Query(default=None, description="ISO end timestamp (default: now)"),
) -> JSONResponse:
    """Return stored OHLCV bars for a single symbol.

    When ``auto_fill=true`` (default), the service checks whether the stored
    data is stale and triggers a targeted Massive fetch before responding.
    This guarantees the caller always gets the freshest possible data from
    Postgres without needing to manage their own Massive client.

    Response payload:
    ```json
    {
      "symbol": "MGC=F",
      "interval": "1m",
      "bar_count": 1200,
      "filled": true,
      "bars_added": 15,
      "data": { "columns": [...], "index": [...], "data": [[...]] }
    }
    ```
    The ``data`` field is in pandas ``split`` orientation and can be
    reconstructed on the client with ``pd.DataFrame(**response["data"])``.
    """
    symbol = _normalize_symbol(symbol)
    _ensure_table()

    # Parse optional explicit time bounds
    start_dt: datetime | None = None
    end_dt: datetime | None = None

    if start:
        try:
            _ts = pd.Timestamp(start)
            if _ts is pd.NaT:
                raise ValueError("NaT")
            _pdt = _ts.to_pydatetime()
            if not isinstance(_pdt, datetime):
                raise ValueError("not datetime")
            start_dt = _pdt if _pdt.tzinfo is not None else _pdt.replace(tzinfo=UTC)
        except Exception:
            raise HTTPException(status_code=400, detail=f"Invalid start timestamp: {start!r}") from None

    if end:
        try:
            _ts = pd.Timestamp(end)
            if _ts is pd.NaT:
                raise ValueError("NaT")
            _pdt = _ts.to_pydatetime()
            if not isinstance(_pdt, datetime):
                raise ValueError("not datetime")
            end_dt = _pdt if _pdt.tzinfo is not None else _pdt.replace(tzinfo=UTC)
        except Exception:
            raise HTTPException(status_code=400, detail=f"Invalid end timestamp: {end!r}") from None

    # ── Step 1: decide whether we need a gap-fill ─────────────────────────
    filled = False
    bars_added = 0
    fill_error: str | None = None

    if auto_fill:
        needs, gap_min = _needs_fill(symbol, interval)
        if needs:
            # Decide whether to block or fire-and-forget based on whether
            # the DB already has *any* data for this symbol.
            #
            # • No stored data at all → fire async and return empty immediately.
            #   The client should poll /bars/{symbol}/fill/status and retry.
            # • Stale data exists → also fire async; return the stale data
            #   straight away so the caller isn't blocked on the backfill.
            #   The ``filling`` flag signals that fresher data will arrive.
            #
            # This prevents the trainer from timing out when the engine needs
            # to run a long multi-chunk backfill (e.g. 180 days × 36 chunks).
            existing_count = _bar_count(symbol, interval)
            if existing_count > 0:
                # Stale data present — return it immediately, fill in background
                logger.info(
                    "Async auto-fill triggered for %s (gap: %d min, %d bars cached)",
                    symbol,
                    gap_min,
                    existing_count,
                )
                fill_job = _get_or_start_symbol_fill(symbol, days_back=days_back, interval=interval)
                filled = True
                fill_error = None
                _ = fill_job  # status available via /bars/{symbol}/fill/status
            else:
                # No data at all — still async; caller gets empty + filling=True
                logger.info(
                    "Async auto-fill triggered for %s (no stored data, gap: %d min)",
                    symbol,
                    gap_min,
                )
                fill_job = _get_or_start_symbol_fill(symbol, days_back=days_back, interval=interval)
                filled = True
                fill_error = None
                _ = fill_job

    # ── Step 2: fetch bars from DB ─────────────────────────────────────────
    df = _fetch_stored_bars(
        symbol,
        interval=interval,
        days_back=days_back,
        start_dt=start_dt,
        end_dt=end_dt,
    )

    # ── Step 3: if DB is still empty, try a live Massive fetch directly ───
    # (covers fresh environments where the table was just created)
    if df.empty and auto_fill:
        logger.info("DB empty after fill for %s — attempting direct Massive fetch", symbol)
        try:
            from lib.integrations.massive_client import get_massive_provider

            provider = get_massive_provider()
            if provider.is_available:
                period = f"{days_back}d" if days_back <= 365 else "1y"
                df = provider.get_aggs(symbol, interval=interval, period=period)
        except Exception as exc:
            logger.debug("Direct Massive fetch failed for %s: %s", symbol, exc)

        if df.empty:
            # Kraken spot fallback — for BTC, ETH, SOL etc. that are not in Massive.
            # Resolve short names to their Kraken internal ticker before calling.
            _KRAKEN_SPOT_TICKER_MAP: dict[str, str] = {
                "BTC": "KRAKEN:XBTUSD",
                "XBT": "KRAKEN:XBTUSD",
                "ETH": "KRAKEN:ETHUSD",
                "SOL": "KRAKEN:SOLUSD",
                "ADA": "KRAKEN:ADAUSD",
                "AVAX": "KRAKEN:AVAXUSD",
                "DOT": "KRAKEN:DOTUSD",
                "LINK": "KRAKEN:LINKUSD",
                "XRP": "KRAKEN:XRPUSD",
                "LTC": "KRAKEN:LTCUSD",
                "BCH": "KRAKEN:BCHUSD",
            }
            _kraken_ticker = _KRAKEN_SPOT_TICKER_MAP.get(
                symbol.upper(),
                symbol if symbol.upper().startswith("KRAKEN:") else None,
            )
            if _kraken_ticker:
                try:
                    from lib.integrations.kraken_client import get_kraken_ohlcv

                    df_kraken = get_kraken_ohlcv(_kraken_ticker, interval=interval, period=f"{days_back}d")
                    if df_kraken is not None and not df_kraken.empty:
                        df = df_kraken
                        logger.info(
                            "Kraken fallback supplied %d bars for %s (%s)",
                            len(df),
                            symbol,
                            _kraken_ticker,
                        )
                except Exception as exc:
                    logger.debug("Kraken fallback failed for %s (%s): %s", symbol, _kraken_ticker, exc)

        if df.empty and symbol.upper() not in _KRAKEN_SPOT_SYMBOLS:
            # Last resort: yfinance (futures only — skip for known Kraken spot symbols)
            try:
                from lib.core.cache import get_data

                period_str = f"{min(days_back, 60)}d"
                df = get_data(symbol, interval=interval, period=period_str)
            except Exception as exc:
                logger.debug("yfinance fallback failed for %s: %s", symbol, exc)

    data_payload = _df_to_split(df)

    # Determine whether a fill is currently running in the background
    with _symbol_fills_lock:
        _sym_job = _symbol_fills.get(symbol)
    filling = _sym_job is not None and _sym_job.is_running

    return JSONResponse(
        {
            "symbol": symbol,
            "interval": interval,
            "days_back": days_back,
            "bar_count": len(df) if df is not None and not df.empty else 0,
            "filled": filled,
            "filling": filling,
            "fill_status_url": f"/bars/{symbol}/fill/status" if filling else None,
            "bars_added": bars_added,
            "fill_error": fill_error,
            "data": data_payload,
            "timestamp": datetime.now(tz=UTC).isoformat(),
        }
    )


@router.post("/bars/bulk")
def get_bars_bulk(req: BulkBarsRequest) -> JSONResponse:
    """Return stored OHLCV bars for multiple symbols in a single call.

    Identical semantics to ``GET /bars/{symbol}`` but batched.  Each symbol
    is processed sequentially (to respect API rate limits).

    Response:
    ```json
    {
      "results": {
        "MGC=F": { "bar_count": 1200, "filled": true, "data": {...} },
        "MES=F": { ... }
      },
      "errors": {}
    }
    ```
    """
    req.symbols = [_normalize_symbol(s) for s in req.symbols]
    _ensure_table()

    if not req.symbols:
        raise HTTPException(status_code=400, detail="No symbols specified")
    if len(req.symbols) > 20:
        raise HTTPException(status_code=400, detail="Maximum 20 symbols per bulk request")

    results: dict[str, Any] = {}
    errors: dict[str, str] = {}

    for symbol in req.symbols:
        try:
            filled = False
            bars_added = 0

            if req.auto_fill:
                needs, gap_min = _needs_fill(symbol, req.interval)
                if needs:
                    fill_res = _run_incremental_fill(
                        symbol,
                        days_back=req.days_back,
                        interval=req.interval,
                    )
                    bars_added = fill_res.get("bars_added", 0)
                    filled = True

            df = _fetch_stored_bars(symbol, interval=req.interval, days_back=req.days_back)

            results[symbol] = {
                "bar_count": len(df) if not df.empty else 0,
                "filled": filled,
                "bars_added": bars_added,
                "data": _df_to_split(df),
            }

        except Exception as exc:
            logger.error("Bulk bars error for %s: %s", symbol, exc)
            errors[symbol] = str(exc)
            results[symbol] = {
                "bar_count": 0,
                "filled": False,
                "bars_added": 0,
                "data": None,
            }

    return JSONResponse(
        {
            "results": results,
            "errors": errors,
            "symbol_count": len(req.symbols),
            "timestamp": datetime.now(tz=UTC).isoformat(),
        }
    )


# ---------------------------------------------------------------------------
# Startup helper — called from main.py lifespan to warm caches
# ---------------------------------------------------------------------------


def startup_warm_caches(days_back: int = 7) -> None:
    """Pre-warm Redis caches from Postgres on service startup.

    Called during FastAPI lifespan startup so the dataset generator and
    engine have immediately-available bar data without waiting for a
    background fill to complete.

    Only runs if the historical_bars table already has data.
    """
    _ensure_table()
    symbols = _get_all_symbols()
    if not symbols:
        return

    warmed = 0
    for symbol in symbols:
        count = _bar_count(symbol, "1m")
        if count > 0:
            try:
                _warm_redis_cache(symbol, "1m", days_back=days_back)
                warmed += 1
            except Exception as exc:
                logger.debug("Startup warm failed for %s: %s", symbol, exc)

    if warmed:
        logger.info("Startup: warmed Redis cache for %d/%d symbols", warmed, len(symbols))
