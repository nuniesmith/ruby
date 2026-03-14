"""
Background Data Sync Service
============================
Maintains a rolling 1-year window of 1-minute OHLCV bars in Postgres for all
enabled futures and crypto symbols.

Two operating modes:
  - **Initial backfill**: When a symbol has < 365 days of data, fetch up to
    365 days of history using the existing backfill infrastructure.
  - **Incremental sync**: Every 5 minutes, fetch the latest bars for all
    enabled symbols and upsert into the ``historical_bars`` table.

After each full sync cycle the service:
  1. Enforces retention — deletes bars older than 395 days (13 months).
  2. Warms the Redis cache — loads the last 24h of bars into a sorted set
     for fast access by the engine and dashboard.

Public API (FastAPI router):
  - GET  /api/data/sync/status   — sync status for all symbols
  - POST /api/data/sync/trigger  — manually trigger a sync cycle
  - GET  /api/data/bars          — serve bars from Postgres

Integration:
  The ``DataSyncService`` is started as an ``asyncio.Task`` during the data
  service lifespan (see ``main.py``) and cancelled on shutdown.

Usage::

    from lib.services.data.sync import DataSyncService, sync_router

    svc = DataSyncService()
    task = asyncio.create_task(svc.run())
    # … on shutdown …
    await svc.stop()
"""

import asyncio
import contextlib
import json
import logging
import os
import time
from datetime import UTC, datetime, timedelta
from typing import Any

from fastapi import APIRouter, Query

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Full rolling window target: 365 calendar days of 1-min bars.
SYNC_DAYS_BACK: int = int(os.getenv("SYNC_DAYS_BACK", "365"))

# Incremental sync interval (seconds).  Every cycle fetches the latest bars
# for each symbol so the rolling window stays current.
SYNC_INTERVAL_SECONDS: int = int(os.getenv("SYNC_INTERVAL_SECONDS", "300"))  # 5 min

# Retention: keep 395 days (13 months) before pruning.  Slightly longer than
# the 365-day target so callers always have a full year even mid-cycle.
RETENTION_DAYS: int = int(os.getenv("SYNC_RETENTION_DAYS", "395"))

# Minimum bar count that constitutes "has enough data".  365 days × 23h ×
# 60 min ≈ 502 200 bars for futures.  We use a lower threshold because
# weekends/holidays reduce the actual count significantly.
_MIN_BARS_FOR_FULL_YEAR: int = 200_000

# Redis TTL for the 24-hour bar cache (sorted set): 25 hours.
_REDIS_BARS_TTL: int = 25 * 3600

# Redis TTL for per-symbol sync metadata: 48 hours.
_REDIS_SYNC_META_TTL: int = 48 * 3600

# How many hours of bars to load into Redis after each symbol sync.
_WARM_CACHE_HOURS: int = 24

# Incremental sync lookback — for the "quick" periodic sync we only need to
# fetch the last few hours since we run every 5 minutes.  Using 1 day gives
# ample overlap to cover any gaps from downtime.
_INCREMENTAL_DAYS: int = 1


# ---------------------------------------------------------------------------
# Symbol resolution
# ---------------------------------------------------------------------------


def _get_sync_symbols() -> list[str]:
    """Return all symbols that should be kept in the rolling window.

    Includes the 9+ CME micro futures data tickers plus all enabled Kraken
    crypto tickers.  Crypto symbols are appended *after* futures so they
    don't delay the critical CME data if Kraken is slow.
    """
    futures_symbols: list[str] = []
    try:
        from lib.core.models import ASSETS, CRYPTO_TICKERS

        futures_symbols = [t for t in ASSETS.values() if t not in CRYPTO_TICKERS]
    except ImportError:
        # Fallback: hard-coded core tickers
        futures_symbols = [
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

    crypto_symbols: list[str] = []
    try:
        from lib.core.models import ENABLE_KRAKEN_CRYPTO, KRAKEN_CONTRACT_SPECS

        if ENABLE_KRAKEN_CRYPTO:
            crypto_symbols = [str(spec["data_ticker"]) for spec in KRAKEN_CONTRACT_SPECS.values()]
    except ImportError:
        pass

    return futures_symbols + crypto_symbols


# ---------------------------------------------------------------------------
# Database helpers (thin wrappers around existing infra)
# ---------------------------------------------------------------------------


def _get_conn():
    """Get a database connection via the shared models helper."""
    try:
        from lib.core.models import _get_conn as models_get_conn

        return models_get_conn()
    except ImportError:
        import sqlite3

        db_path = os.getenv("DB_PATH", "futures_journal.db")
        conn = sqlite3.connect(db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn


def _is_postgres() -> bool:
    """Check if Postgres is the active backend."""
    try:
        from lib.core.models import _is_using_postgres

        return _is_using_postgres()
    except ImportError:
        return False


def _placeholder() -> str:
    return "%s" if _is_postgres() else "?"


# ---------------------------------------------------------------------------
# Retention cleanup
# ---------------------------------------------------------------------------


def _enforce_retention(days: int = RETENTION_DAYS) -> int:
    """Delete bars older than *days* from ``historical_bars``.

    Runs after each full sync cycle to keep the database from growing
    unboundedly.

    Args:
        days: Number of calendar days to retain.  Default 395 (≈ 13 months).

    Returns:
        Number of rows deleted.
    """
    cutoff = datetime.now(tz=UTC) - timedelta(days=days)
    cutoff_iso = cutoff.isoformat()

    conn = None
    try:
        conn = _get_conn()
        ph = _placeholder()
        sql = f"DELETE FROM historical_bars WHERE timestamp < {ph}"
        cur = conn.execute(sql, (cutoff_iso,))
        conn.commit()

        # Get rowcount — works for both psycopg2 cursor and sqlite3 cursor
        deleted = 0
        if hasattr(cur, "rowcount"):
            deleted = cur.rowcount if cur.rowcount and cur.rowcount > 0 else 0

        if deleted > 0:
            logger.info(
                "Retention cleanup: deleted %d bars older than %s (%d-day window)",
                deleted,
                cutoff_iso[:10],
                days,
            )
        else:
            logger.debug("Retention cleanup: no bars older than %s to delete", cutoff_iso[:10])

        return deleted
    except Exception as exc:
        logger.warning("Retention cleanup failed (non-fatal): %s", exc)
        with contextlib.suppress(Exception):
            if conn is not None:
                conn.rollback()
        return 0
    finally:
        if conn is not None:
            with contextlib.suppress(Exception):
                conn.close()


# ---------------------------------------------------------------------------
# Redis cache warming
# ---------------------------------------------------------------------------


def _warm_redis_cache(symbol: str, interval: str = "1m", hours: int = _WARM_CACHE_HOURS) -> int:
    """Load the last *hours* of bars from Postgres into a Redis sorted set.

    The sorted set key is ``bars:1m:{symbol}`` with each member scored by its
    Unix timestamp.  This gives O(log N) range queries for the engine and
    dashboard without hitting Postgres on every request.

    Also writes per-symbol sync metadata to ``data:sync:{symbol}``.

    Args:
        symbol: Ticker symbol (e.g. ``"MGC=F"``).
        interval: Bar interval.  Default ``"1m"``.
        hours: Number of hours of bars to cache.  Default 24.

    Returns:
        Number of bars loaded into Redis.
    """
    try:
        from lib.core.cache import REDIS_AVAILABLE, _r
    except ImportError:
        return 0

    if not REDIS_AVAILABLE or _r is None:
        return 0

    conn = None
    try:
        conn = _get_conn()
        now = datetime.now(tz=UTC)
        start_dt = now - timedelta(hours=hours)

        ph = _placeholder()
        sql = (
            f"SELECT timestamp, open, high, low, close, volume "
            f"FROM historical_bars "
            f"WHERE symbol = {ph} AND interval = {ph} "
            f"  AND timestamp >= {ph} "
            f"ORDER BY timestamp ASC"
        )
        cur = conn.execute(sql, (symbol, interval, start_dt.isoformat()))
        rows = cur.fetchall()

        if not rows:
            return 0

        redis_key = f"bars:{interval}:{symbol}"
        pipe = _r.pipeline()

        # Clear old data for this key, then bulk-add
        pipe.delete(redis_key)

        loaded = 0
        for row in rows:
            if isinstance(row, (tuple, list)):
                ts_str, o, h, lo, c, v = row
            else:
                ts_str = row["timestamp"]
                o = row["open"]
                h = row["high"]
                lo = row["low"]
                c = row["close"]
                v = row["volume"]

            # Compute Unix timestamp as score
            try:
                ts_dt = datetime.fromisoformat(str(ts_str))
                if ts_dt.tzinfo is None:
                    ts_dt = ts_dt.replace(tzinfo=UTC)
                score = ts_dt.timestamp()
            except Exception:
                continue

            bar_json = json.dumps(
                {
                    "t": str(ts_str),
                    "o": float(o),
                    "h": float(h),
                    "l": float(lo),
                    "c": float(c),
                    "v": int(v or 0),
                }
            )
            pipe.zadd(redis_key, {bar_json: score})
            loaded += 1

        # Set TTL on the sorted set
        pipe.expire(redis_key, _REDIS_BARS_TTL)
        pipe.execute()

        logger.debug(
            "Warmed Redis cache for %s: %d bars (%dh window)",
            symbol,
            loaded,
            hours,
        )
        return loaded

    except Exception as exc:
        logger.warning("Redis cache warm failed for %s (non-fatal): %s", symbol, exc)
        return 0
    finally:
        if conn is not None:
            with contextlib.suppress(Exception):
                conn.close()


# ---------------------------------------------------------------------------
# Per-symbol sync metadata in Redis
# ---------------------------------------------------------------------------


def _set_sync_status(
    symbol: str,
    *,
    status: str = "ok",
    bar_count: int = 0,
    error: str = "",
    last_synced: str | None = None,
) -> None:
    """Write per-symbol sync metadata to Redis.

    Key: ``data:sync:{symbol}`` → JSON dict.
    """
    try:
        from lib.core.cache import REDIS_AVAILABLE, _r
    except ImportError:
        return

    if not REDIS_AVAILABLE or _r is None:
        return

    meta = {
        "symbol": symbol,
        "status": status,
        "bar_count": bar_count,
        "last_synced": last_synced or datetime.now(tz=UTC).isoformat(),
        "error": error,
    }
    try:
        key = f"data:sync:{symbol}"
        _r.setex(key, _REDIS_SYNC_META_TTL, json.dumps(meta))
    except Exception as exc:
        logger.debug("Failed to set sync status for %s: %s", symbol, exc)


def _get_sync_status_for_symbol(symbol: str) -> dict[str, Any] | None:
    """Read per-symbol sync metadata from Redis."""
    try:
        from lib.core.cache import REDIS_AVAILABLE, _r
    except ImportError:
        return None

    if not REDIS_AVAILABLE or _r is None:
        return None

    try:
        key = f"data:sync:{symbol}"
        raw = _r.get(key)
        if raw is not None:
            return json.loads(raw)
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Public: get_sync_status
# ---------------------------------------------------------------------------


def get_sync_status() -> dict[str, Any]:
    """Return sync status for all tracked symbols.

    Reads per-symbol metadata from Redis (``data:sync:{symbol}``) and
    aggregates into a dict suitable for API consumption.

    Returns:
        Dict with keys:
          - ``symbols``: list of per-symbol status dicts
          - ``total_symbols``: int
          - ``last_checked``: ISO timestamp
    """
    symbols = _get_sync_symbols()
    results: list[dict[str, Any]] = []

    for sym in symbols:
        meta = _get_sync_status_for_symbol(sym)
        if meta is not None:
            results.append(meta)
        else:
            results.append(
                {
                    "symbol": sym,
                    "status": "unknown",
                    "bar_count": 0,
                    "last_synced": None,
                    "error": "",
                }
            )

    return {
        "symbols": results,
        "total_symbols": len(results),
        "last_checked": datetime.now(tz=UTC).isoformat(),
    }


# ---------------------------------------------------------------------------
# DataSyncService
# ---------------------------------------------------------------------------


class DataSyncService:
    """Async background service that maintains a rolling 1-year data window.

    Start via ``asyncio.create_task(svc.run())`` during the data-service
    lifespan.  Stop via ``await svc.stop()`` on shutdown.

    The service loops forever:
      1. For each symbol, check bar count.  If below threshold → full
         365-day backfill.  Otherwise → incremental 1-day sync.
      2. After syncing a symbol, warm its Redis cache (24h window).
      3. After all symbols, enforce retention (delete bars > 395 days old).
      4. Sleep for ``SYNC_INTERVAL_SECONDS`` (default 5 min), then repeat.
    """

    def __init__(self) -> None:
        self._running: bool = False
        self._task: asyncio.Task | None = None
        self._trigger_event: asyncio.Event = asyncio.Event()
        self._cycle_count: int = 0
        self._last_cycle_time: float = 0.0

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def run(self) -> None:
        """Main loop — runs until ``stop()`` is called."""
        self._running = True
        logger.info(
            "DataSyncService started (target=%d days, interval=%ds, retention=%d days)",
            SYNC_DAYS_BACK,
            SYNC_INTERVAL_SECONDS,
            RETENTION_DAYS,
        )

        # Ensure the historical_bars table exists
        try:
            from lib.services.engine.backfill import init_backfill_table

            init_backfill_table()
        except Exception as exc:
            logger.warning("Failed to ensure backfill table (non-fatal): %s", exc)

        while self._running:
            try:
                await self._run_cycle()
            except asyncio.CancelledError:
                logger.info("DataSyncService cancelled")
                break
            except Exception as exc:
                logger.error("DataSyncService cycle failed: %s", exc, exc_info=True)

            # Wait for the next cycle or a manual trigger
            try:
                self._trigger_event.clear()
                await asyncio.wait_for(
                    self._trigger_event.wait(),
                    timeout=SYNC_INTERVAL_SECONDS,
                )
                logger.info("DataSyncService: manual trigger received")
            except TimeoutError:
                pass  # Normal: interval elapsed
            except asyncio.CancelledError:
                break

        logger.info("DataSyncService stopped (completed %d cycles)", self._cycle_count)

    async def stop(self) -> None:
        """Signal the service to stop gracefully."""
        self._running = False
        self._trigger_event.set()  # wake from sleep
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task

    def trigger(self) -> None:
        """Manually trigger a sync cycle (non-blocking)."""
        self._trigger_event.set()

    @property
    def is_running(self) -> bool:
        """Whether the service loop is currently active."""
        return self._running

    @property
    def cycle_count(self) -> int:
        """Number of completed sync cycles."""
        return self._cycle_count

    # ------------------------------------------------------------------
    # Core sync cycle
    # ------------------------------------------------------------------

    async def _run_cycle(self) -> None:
        """Execute one full sync cycle across all symbols."""
        cycle_start = time.monotonic()
        symbols = _get_sync_symbols()
        logger.info(
            "Sync cycle #%d starting: %d symbols",
            self._cycle_count + 1,
            len(symbols),
        )

        successes = 0
        failures = 0

        for symbol in symbols:
            if not self._running:
                break
            try:
                await self._sync_symbol(symbol)
                successes += 1
            except Exception as exc:
                failures += 1
                logger.warning("Sync failed for %s (non-fatal): %s", symbol, exc)
                _set_sync_status(symbol, status="error", error=str(exc))

        # Retention cleanup after all symbols
        if self._running:
            try:
                await asyncio.to_thread(_enforce_retention, RETENTION_DAYS)
            except Exception as exc:
                logger.warning("Post-cycle retention cleanup failed (non-fatal): %s", exc)

        elapsed = round(time.monotonic() - cycle_start, 2)
        self._cycle_count += 1
        self._last_cycle_time = elapsed

        logger.info(
            "Sync cycle #%d complete: %d ok, %d failed, %.1fs elapsed",
            self._cycle_count,
            successes,
            failures,
            elapsed,
        )

    async def _sync_symbol(self, symbol: str) -> None:
        """Sync a single symbol — full backfill or incremental."""
        bar_count = await asyncio.to_thread(self._get_bar_count, symbol)
        is_initial = bar_count < _MIN_BARS_FOR_FULL_YEAR

        if is_initial:
            days = SYNC_DAYS_BACK
            logger.info(
                "Full backfill for %s (%d bars < %d threshold, targeting %d days)",
                symbol,
                bar_count,
                _MIN_BARS_FOR_FULL_YEAR,
                days,
            )
        else:
            days = _INCREMENTAL_DAYS
            logger.debug(
                "Incremental sync for %s (%d bars, last %d day(s))",
                symbol,
                bar_count,
                days,
            )

        # Run the backfill in a thread to avoid blocking the event loop.
        # The existing backfill_symbol() is synchronous and handles both
        # Massive (futures) and Kraken (crypto) internally.
        result = await asyncio.to_thread(self._backfill_symbol, symbol, days)

        error = result.get("error", "")
        bars_after = result.get("bars_after", bar_count)

        # Update sync status in Redis
        _set_sync_status(
            symbol,
            status="error" if error else "ok",
            bar_count=bars_after,
            error=error,
        )

        # Warm Redis cache with the last 24h of bars
        try:
            await asyncio.to_thread(_warm_redis_cache, symbol, "1m", _WARM_CACHE_HOURS)
        except Exception as exc:
            logger.debug("Cache warm failed for %s (non-fatal): %s", symbol, exc)

    # ------------------------------------------------------------------
    # Synchronous helpers (run in thread pool)
    # ------------------------------------------------------------------

    @staticmethod
    def _get_bar_count(symbol: str, interval: str = "1m") -> int:
        """Get the current bar count for a symbol from Postgres."""
        conn = None
        try:
            conn = _get_conn()
            ph = _placeholder()
            sql = f"SELECT COUNT(*) FROM historical_bars WHERE symbol = {ph} AND interval = {ph}"
            cur = conn.execute(sql, (symbol, interval))
            row = cur.fetchone()
            if row is not None:
                return int(row[0])
        except Exception as exc:
            logger.debug("Failed to get bar count for %s: %s", symbol, exc)
        finally:
            if conn is not None:
                with contextlib.suppress(Exception):
                    conn.close()
        return 0

    @staticmethod
    def _backfill_symbol(symbol: str, days_back: int) -> dict[str, Any]:
        """Call the existing backfill infrastructure for one symbol.

        Wraps ``backfill_symbol()`` from the engine backfill module,
        overriding the days_back parameter for rolling-window targets.
        """
        try:
            from lib.services.engine.backfill import backfill_symbol

            return backfill_symbol(symbol, days_back=days_back, interval="1m")
        except ImportError:
            logger.error(
                "Cannot import backfill module — backfill_symbol unavailable for %s",
                symbol,
            )
            return {"symbol": symbol, "error": "backfill module not available"}
        except Exception as exc:
            logger.error("backfill_symbol(%s) raised: %s", symbol, exc)
            return {"symbol": symbol, "error": str(exc)}


# ---------------------------------------------------------------------------
# Module-level singleton (lazily created by main.py)
# ---------------------------------------------------------------------------
_service: DataSyncService | None = None


def get_sync_service() -> DataSyncService:
    """Return (or create) the module-level DataSyncService singleton."""
    global _service
    if _service is None:
        _service = DataSyncService()
    return _service


# ---------------------------------------------------------------------------
# FastAPI router
# ---------------------------------------------------------------------------

sync_router = APIRouter(tags=["Data Sync"])


@sync_router.get("/api/data/sync/status")
def api_sync_status() -> dict[str, Any]:
    """Return sync status for all tracked symbols.

    Reads per-symbol metadata from Redis and returns an aggregate view
    showing bar counts, last sync times, and any errors.
    """
    status = get_sync_status()

    # Augment with service-level info
    svc = _service
    if svc is not None:
        status["service"] = {
            "running": svc.is_running,
            "cycles_completed": svc.cycle_count,
            "last_cycle_seconds": svc._last_cycle_time,
            "sync_interval_seconds": SYNC_INTERVAL_SECONDS,
            "target_days": SYNC_DAYS_BACK,
            "retention_days": RETENTION_DAYS,
        }
    else:
        status["service"] = {"running": False, "cycles_completed": 0}

    return status


@sync_router.post("/api/data/sync/trigger")
def api_sync_trigger() -> dict[str, str]:
    """Manually trigger a sync cycle.

    The sync runs asynchronously in the background.  This endpoint returns
    immediately with a confirmation message.
    """
    svc = _service
    if svc is None or not svc.is_running:
        return {"status": "error", "message": "DataSyncService is not running"}

    svc.trigger()
    return {"status": "ok", "message": "Sync cycle triggered"}


@sync_router.get("/api/data/bars")
def api_data_bars(
    symbol: str = Query(..., description="Ticker symbol, e.g. MGC=F or KRAKEN:XBTUSD"),
    interval: str = Query("1m", description="Bar interval (default 1m)"),
    days: int = Query(30, ge=1, le=400, description="Number of days to look back"),
    format: str = Query("json", description="Response format: 'json' (default) or 'csv'"),
):
    """Serve historical bars from Postgres via the existing backfill store.

    Returns OHLCV data as either a JSON dict (default) with ``bars`` (list of
    bar objects) and metadata fields, or as CSV text when ``format=csv``.
    """
    try:
        from lib.services.engine.backfill import get_stored_bars
    except ImportError:
        if format == "csv":
            from fastapi.responses import PlainTextResponse

            return PlainTextResponse("# error: backfill module not available\n", media_type="text/csv")
        return {
            "symbol": symbol,
            "interval": interval,
            "days": days,
            "bars": [],
            "count": 0,
            "error": "backfill module not available",
        }

    try:
        df = get_stored_bars(symbol, days_back=days, interval=interval)
    except Exception as exc:
        logger.warning("Failed to fetch bars for %s: %s", symbol, exc)
        if format == "csv":
            from fastapi.responses import PlainTextResponse

            return PlainTextResponse(f"# error: {exc}\n", media_type="text/csv")
        return {
            "symbol": symbol,
            "interval": interval,
            "days": days,
            "bars": [],
            "count": 0,
            "error": str(exc),
        }

    # --- CSV format -------------------------------------------------------
    if format == "csv":
        import io as _io

        from fastapi.responses import PlainTextResponse

        buf = _io.StringIO()
        buf.write("timestamp,open,high,low,close,volume\n")
        if not df.empty:
            for idx, row in df.iterrows():
                ts = idx.isoformat() if hasattr(idx, "isoformat") else str(idx)
                o = float(row.get("Open", row.get("open", 0)))
                h = float(row.get("High", row.get("high", 0)))
                lo = float(row.get("Low", row.get("low", 0)))
                c = float(row.get("Close", row.get("close", 0)))
                v = int(row.get("Volume", row.get("volume", 0)))
                buf.write(f"{ts},{o},{h},{lo},{c},{v}\n")
        return PlainTextResponse(buf.getvalue(), media_type="text/csv")

    # --- JSON format (default) --------------------------------------------
    bars: list[dict[str, Any]] = []
    if not df.empty:
        for idx, row in df.iterrows():
            bars.append(
                {
                    "t": idx.isoformat() if hasattr(idx, "isoformat") else str(idx),
                    "o": float(row.get("Open", row.get("open", 0))),
                    "h": float(row.get("High", row.get("high", 0))),
                    "l": float(row.get("Low", row.get("low", 0))),
                    "c": float(row.get("Close", row.get("close", 0))),
                    "v": int(row.get("Volume", row.get("volume", 0))),
                }
            )

    return {
        "symbol": symbol,
        "interval": interval,
        "days": days,
        "bars": bars,
        "count": len(bars),
        "earliest": bars[0]["t"] if bars else None,
        "latest": bars[-1]["t"] if bars else None,
    }
