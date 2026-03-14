"""
Historical Data Backfill
=====================================
Fetches and stores up to 1 year of historical 1-minute OHLCV bars for all
active micro futures contracts into the database (Postgres or SQLite).

Design decisions:
  - **Idempotent**: Uses UPSERT (INSERT OR IGNORE / ON CONFLICT DO NOTHING)
    so re-running never duplicates bars.
  - **Gap-aware**: Queries the DB for the latest stored bar per symbol and
    only fetches the missing range, minimising API calls.
  - **Chunked**: Fetches data in multi-day chunks (5 days for 1-min bars)
    to stay within yfinance and Massive API limits.
  - **Dual-backend**: Works with both SQLite (local dev) and Postgres
    (production Docker) — table DDL adapts automatically.
  - **Fault-tolerant**: Logs and continues on per-symbol or per-chunk
    failures so one bad ticker doesn't abort the entire backfill.
  - **Progress logging**: Logs per-symbol totals and a final summary.

Data source priority:
  1. Massive.com REST API (if MASSIVE_API_KEY is set)
  2. yfinance (fallback — limited to ~7 days of 1-min data)

Public API:
    from lib.services.engine.backfill import run_backfill, get_backfill_status, get_stored_bars

    # Run a full backfill for all assets (called by engine scheduler)
    summary = run_backfill()

    # Fetch stored bars for a single symbol (used by optimization/backtest)
    df = get_stored_bars("MGC=F", days_back=30)

    # Check what's stored
    status = get_backfill_status()

Usage from engine:
    The scheduler dispatches HISTORICAL_BACKFILL once per off-hours session.
    ``_handle_historical_backfill`` in ``main.py`` calls ``run_backfill()``.
"""

import contextlib
import logging
import os
import time
from datetime import UTC, datetime, timedelta
from typing import Any

import pandas as pd

# ---------------------------------------------------------------------------
# Path setup for bare imports
# ---------------------------------------------------------------------------

logger = logging.getLogger("engine.backfill")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# Maximum number of calendar days to look back for backfill.
# yfinance caps 1-min data at ~7 days; Massive supports much longer ranges.
# We default to 30 days which is useful for ATR/ORB calculations and short
# backtests.  With Massive, this can be extended to 365.
DEFAULT_DAYS_BACK = int(os.getenv("BACKFILL_DAYS_BACK", "30"))

# Chunk size in calendar days per API request.  yfinance works best with
# 5-day chunks for 1-min data.  Massive can handle larger windows.
#
# Recommended values:
#   5  (default) — safe for both yfinance and Massive; produces ~36 chunks for
#                  a 180-day backfill.  Fine for small histories but can cause
#                  Massive REST timeouts on very long initial fills because each
#                  chunk is a separate HTTP request with a short read window.
#  30             — reduces a 180-day Massive fill from ~36 chunks to ~6,
#                  dramatically cutting round-trip overhead and timeout risk.
#                  Use this when MASSIVE_API_KEY is set and yfinance is not the
#                  primary source.  Set via: BACKFILL_CHUNK_DAYS=30
#
# Note: changing this only affects *new* incremental fills; already-stored bars
# are never re-fetched (idempotent upsert).
CHUNK_DAYS = int(os.getenv("BACKFILL_CHUNK_DAYS", "5"))

# Symbols to backfill — defaults to all micro contract data tickers.
# Can be overridden via comma-separated env var.
_SYMBOLS_OVERRIDE = os.getenv("BACKFILL_SYMBOLS", "")

# Batch size for INSERT statements (number of rows per executemany call)
INSERT_BATCH_SIZE = int(os.getenv("BACKFILL_INSERT_BATCH", "5000"))


# ---------------------------------------------------------------------------
# Table DDL
# ---------------------------------------------------------------------------

_SCHEMA_SQLITE = """
CREATE TABLE IF NOT EXISTS historical_bars (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol      TEXT    NOT NULL,
    timestamp   TEXT    NOT NULL,
    open        REAL    NOT NULL,
    high        REAL    NOT NULL,
    low         REAL    NOT NULL,
    close       REAL    NOT NULL,
    volume      INTEGER NOT NULL DEFAULT 0,
    interval    TEXT    NOT NULL DEFAULT '1m',
    UNIQUE(symbol, timestamp, interval)
);
CREATE INDEX IF NOT EXISTS idx_hb_symbol_ts
    ON historical_bars (symbol, timestamp);
CREATE INDEX IF NOT EXISTS idx_hb_symbol_interval
    ON historical_bars (symbol, interval, timestamp);
"""

_SCHEMA_POSTGRES = """
CREATE TABLE IF NOT EXISTS historical_bars (
    id          SERIAL PRIMARY KEY,
    symbol      TEXT    NOT NULL,
    timestamp   TEXT    NOT NULL,
    open        DOUBLE PRECISION NOT NULL,
    high        DOUBLE PRECISION NOT NULL,
    low         DOUBLE PRECISION NOT NULL,
    close       DOUBLE PRECISION NOT NULL,
    volume      BIGINT  NOT NULL DEFAULT 0,
    interval    TEXT    NOT NULL DEFAULT '1m',
    UNIQUE(symbol, timestamp, interval)
);
CREATE INDEX IF NOT EXISTS idx_hb_symbol_ts
    ON historical_bars (symbol, timestamp);
CREATE INDEX IF NOT EXISTS idx_hb_symbol_interval
    ON historical_bars (symbol, interval, timestamp);
"""

# Upsert SQL — ignores duplicates on the unique constraint
_INSERT_SQLITE = """
INSERT OR IGNORE INTO historical_bars
    (symbol, timestamp, open, high, low, close, volume, interval)
VALUES (?, ?, ?, ?, ?, ?, ?, ?)
"""

_INSERT_POSTGRES = """
INSERT INTO historical_bars
    (symbol, timestamp, open, high, low, close, volume, interval)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (symbol, timestamp, interval) DO NOTHING
"""

# Query: latest bar timestamp for a symbol
_LATEST_BAR_SQL = """
SELECT MAX(timestamp) FROM historical_bars
WHERE symbol = {ph} AND interval = {ph}
"""

# Query: count bars for a symbol
_COUNT_BARS_SQL = """
SELECT COUNT(*) FROM historical_bars
WHERE symbol = {ph} AND interval = {ph}
"""

# Query: fetch bars for a symbol within a date range
_FETCH_BARS_SQL = """
SELECT timestamp, open, high, low, close, volume
FROM historical_bars
WHERE symbol = {ph} AND interval = {ph}
  AND timestamp >= {ph} AND timestamp <= {ph}
ORDER BY timestamp ASC
"""

# Query: all symbols with their bar counts and date ranges
_STATUS_SQL = """
SELECT symbol, interval, COUNT(*) as bar_count,
       MIN(timestamp) as earliest, MAX(timestamp) as latest
FROM historical_bars
GROUP BY symbol, interval
ORDER BY symbol, interval
"""


# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------


def _is_postgres() -> bool:
    """Check if we're using Postgres."""
    try:
        from lib.core.models import _is_using_postgres

        return _is_using_postgres()
    except ImportError:
        return False


def _get_conn():
    """Get a database connection."""
    try:
        from lib.core.models import _get_conn

        return _get_conn()
    except ImportError:
        # Fallback: direct SQLite
        import sqlite3

        db_path = os.getenv("DB_PATH", "futures_journal.db")
        conn = sqlite3.connect(db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn


def _placeholder() -> str:
    """Return the SQL placeholder for the current backend."""
    return "%s" if _is_postgres() else "?"


def _format_sql(template: str) -> str:
    """Replace {ph} placeholders with the backend-appropriate placeholder."""
    ph = _placeholder()
    return template.replace("{ph}", ph)


def init_backfill_table() -> None:
    """Create the historical_bars table if it doesn't exist.

    Safe to call multiple times (idempotent).
    """
    conn = _get_conn()
    try:
        schema = _SCHEMA_POSTGRES if _is_postgres() else _SCHEMA_SQLITE
        if _is_postgres():
            # Postgres: execute each statement separately
            for stmt in schema.strip().split(";"):
                stmt = stmt.strip()
                if stmt:
                    conn.execute(stmt)
            conn.commit()
        else:
            conn.executescript(schema)
        logger.info("historical_bars table ready")
    except Exception as exc:
        logger.error("Failed to create historical_bars table: %s", exc)
        with contextlib.suppress(Exception):
            conn.rollback()
        raise
    finally:
        with contextlib.suppress(Exception):
            conn.close()


# ---------------------------------------------------------------------------
# Symbol resolution
# ---------------------------------------------------------------------------


# Whether to include KRAKEN:* crypto tickers in the backfill run.
# Defaults to the same env var that controls the Kraken integration.
_BACKFILL_KRAKEN = os.getenv("ENABLE_KRAKEN_CRYPTO", "1").strip().lower() in ("1", "true", "yes")

# For crypto, Kraken allows up to 720 1-min candles per REST call, and the
# public OHLC endpoint only returns data up to ~12 hours back for 1m.
# We use a shorter chunk size for crypto so each chunk fits within one API
# call, and a deeper lookback that makes sense for 24/7 markets.
KRAKEN_CHUNK_HOURS = int(os.getenv("BACKFILL_KRAKEN_CHUNK_HOURS", "10"))  # < 720 min = 12h max
KRAKEN_DAYS_BACK = int(os.getenv("BACKFILL_KRAKEN_DAYS_BACK", "7"))  # Kraken 1m history is limited


def _get_backfill_symbols() -> list[str]:
    """Return the list of ticker symbols to backfill.

    Priority:
      1. BACKFILL_SYMBOLS env var (comma-separated) — explicit list, no auto-append
      2. All data tickers from models.ASSETS (futures) +
         all KRAKEN:* crypto tickers when ENABLE_KRAKEN_CRYPTO is set

    Kraken crypto tickers are always appended *after* the CME futures list so
    they don't block the critical futures backfill if Kraken is unavailable.
    """
    if _SYMBOLS_OVERRIDE:
        return [s.strip() for s in _SYMBOLS_OVERRIDE.split(",") if s.strip()]

    futures_symbols: list[str] = []
    try:
        from lib.core.models import ASSETS, CRYPTO_TICKERS

        # Only keep non-crypto tickers in the main futures list
        futures_symbols = [t for t in ASSETS.values() if t not in CRYPTO_TICKERS]
    except ImportError:
        futures_symbols = ["MGC=F", "MNQ=F", "MES=F", "MCL=F", "SI=F", "HG=F"]

    crypto_symbols: list[str] = []
    if _BACKFILL_KRAKEN:
        try:
            from lib.core.models import KRAKEN_CONTRACT_SPECS

            crypto_symbols = [str(spec["data_ticker"]) for spec in KRAKEN_CONTRACT_SPECS.values()]
        except ImportError:
            pass

    return futures_symbols + crypto_symbols


def _symbol_display_name(ticker: str) -> str:
    """Return a human-readable name for a ticker, or the ticker itself."""
    try:
        from lib.core.models import TICKER_TO_NAME

        return TICKER_TO_NAME.get(ticker, ticker)
    except ImportError:
        return ticker


# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------


def _fetch_chunk_massive(ticker: str, start_dt: datetime, end_dt: datetime) -> pd.DataFrame:
    """Fetch 1-min bars from Massive for a date range.

    Returns a DataFrame with columns [Open, High, Low, Close, Volume]
    and a DatetimeIndex, or empty DataFrame on failure.
    """
    try:
        from lib.core.cache import _get_massive_provider

        provider = _get_massive_provider()
        if provider is None or not provider.is_available:
            return pd.DataFrame()

        from lib.integrations.massive_client import (
            _dropna_ohlc,
            _parse_timestamp_index,
        )

        massive_ticker = provider.resolve_from_yahoo(ticker)
        if not massive_ticker:
            return pd.DataFrame()

        client = provider._client
        if client is None:
            return pd.DataFrame()

        start_str = start_dt.strftime("%Y-%m-%dT%H:%M:%S-05:00")
        end_str = end_dt.strftime("%Y-%m-%dT%H:%M:%S-05:00")

        aggs = list(
            client.list_futures_aggregates(
                ticker=massive_ticker,
                resolution="1min",
                window_start_gte=start_str,
                window_start_lte=end_str,
                limit=50000,
                sort="asc",
            )
        )

        if not aggs:
            return pd.DataFrame()

        rows = []
        for agg in aggs:
            rows.append(
                {
                    "Open": getattr(agg, "open", None),
                    "High": getattr(agg, "high", None),
                    "Low": getattr(agg, "low", None),
                    "Close": getattr(agg, "close", None),
                    "Volume": getattr(agg, "volume", 0) or 0,
                    "timestamp": getattr(agg, "window_start", None),
                }
            )

        df = pd.DataFrame(rows)
        if not df.empty and "timestamp" in df.columns:
            df = _parse_timestamp_index(df)
        df = _dropna_ohlc(df)

        return df

    except Exception as exc:
        logger.debug("Massive chunk fetch failed for %s: %s", ticker, exc)
        return pd.DataFrame()


def _fetch_chunk_yfinance(ticker: str, start_dt: datetime, end_dt: datetime) -> pd.DataFrame:
    """Fetch 1-min bars from yfinance for a date range.

    yfinance limits 1-min data to ~7 calendar days from today.
    Returns a DataFrame with [Open, High, Low, Close, Volume] and
    DatetimeIndex, or empty on failure.
    """
    try:
        import yfinance as yf

        from lib.core.cache import _flatten_columns

        # yfinance 1-min data only works for recent ~7 days
        now = datetime.now(tz=UTC)
        if (now - start_dt.replace(tzinfo=UTC)).days > 8:
            logger.debug(
                "yfinance: skipping chunk %s → %s for %s (>7 days ago)",
                start_dt.date(),
                end_dt.date(),
                ticker,
            )
            return pd.DataFrame()

        df = _flatten_columns(
            yf.download(
                ticker,
                interval="1m",
                start=str(start_dt.date()),
                end=str((end_dt + timedelta(days=1)).date()),
                prepost=True,
                auto_adjust=True,
                progress=False,
            )
        )

        return df if df is not None else pd.DataFrame()

    except Exception as exc:
        logger.debug("yfinance chunk fetch failed for %s: %s", ticker, exc)
        return pd.DataFrame()


def _fetch_chunk_kraken(ticker: str, start_dt: datetime, end_dt: datetime) -> pd.DataFrame:
    """Fetch 1-min bars from Kraken REST API for a KRAKEN:* ticker and date range.

    Kraken's public OHLC endpoint returns up to 720 candles per call.
    For longer date ranges we make multiple calls, stitching results together.
    Each call uses ``since_timestamp`` to page forward through the range.

    Returns a DataFrame with [Open, High, Low, Close, Volume] and a UTC
    DatetimeIndex, or empty DataFrame on failure / unavailability.
    """
    try:
        from lib.integrations.kraken_client import get_kraken_provider

        provider = get_kraken_provider()
        if provider is None or not provider.is_available:
            return pd.DataFrame()

        # Convert datetimes to UNIX timestamps for the Kraken API
        start_ts = (
            int(start_dt.replace(tzinfo=UTC).timestamp()) if start_dt.tzinfo is None else int(start_dt.timestamp())
        )
        end_ts = int(end_dt.replace(tzinfo=UTC).timestamp()) if end_dt.tzinfo is None else int(end_dt.timestamp())

        all_frames: list[pd.DataFrame] = []
        current_since = start_ts
        max_iterations = 50  # safety cap

        for _ in range(max_iterations):
            df = provider.get_ohlcv(ticker, interval="1m", since_timestamp=current_since)
            if df.empty:
                break

            # Normalise index to UTC
            try:
                dti = pd.DatetimeIndex(df.index)
                dti = dti.tz_localize("UTC") if getattr(dti, "tzinfo", None) is None else dti.tz_convert("UTC")
                df.index = dti
            except Exception:
                pass

            # Trim anything beyond end_dt
            end_cutoff = pd.Timestamp(end_ts, unit="s", tz="UTC")
            with contextlib.suppress(Exception):
                df = pd.DataFrame(df[df.index <= end_cutoff])

            if not df.empty:
                all_frames.append(df.copy())

            # Determine the next ``since`` timestamp
            if df.empty:
                break
            last_idx_raw = df.index[-1]
            try:
                last_ts_obj = pd.Timestamp(last_idx_raw)  # type: ignore[arg-type]
                next_since = int(last_ts_obj.timestamp()) + 60  # +1 minute
            except Exception:
                break
            if next_since >= end_ts:
                break
            if next_since <= current_since:
                break  # no progress — stop
            current_since = next_since

        if not all_frames:
            return pd.DataFrame()

        result: pd.DataFrame = pd.concat(all_frames)
        try:
            result = result[~result.index.duplicated(keep="last")]  # type: ignore[assignment]
            result = result.sort_index()  # type: ignore[assignment]
        except Exception:
            pass

        logger.debug(
            "Kraken: %d bars for %s (%s → %s)",
            len(result),
            ticker,
            start_dt.date(),
            end_dt.date(),
        )
        return result

    except Exception as exc:
        logger.debug("Kraken chunk fetch failed for %s: %s", ticker, exc)
        return pd.DataFrame()


def fetch_bars_chunk(ticker: str, start_dt: datetime, end_dt: datetime) -> pd.DataFrame:
    """Fetch 1-min bars for a ticker and date range.

    Routes KRAKEN:* tickers to the Kraken REST API.
    For CME futures, tries Massive first, then yfinance as fallback.

    Returns DataFrame with [Open, High, Low, Close, Volume] + DatetimeIndex.
    """
    # Kraken crypto pairs — bypass Massive/yfinance entirely
    if ticker.startswith("KRAKEN:"):
        df = _fetch_chunk_kraken(ticker, start_dt, end_dt)
        if not df.empty:
            logger.debug(
                "Kraken: %d bars for %s (%s → %s)",
                len(df),
                ticker,
                start_dt.date(),
                end_dt.date(),
            )
        return df

    # Try Massive first
    df = _fetch_chunk_massive(ticker, start_dt, end_dt)
    if not df.empty:
        logger.debug(
            "Massive: %d bars for %s (%s → %s)",
            len(df),
            ticker,
            start_dt.date(),
            end_dt.date(),
        )
        return df

    # Fallback to yfinance
    df = _fetch_chunk_yfinance(ticker, start_dt, end_dt)
    if not df.empty:
        logger.debug(
            "yfinance: %d bars for %s (%s → %s)",
            len(df),
            ticker,
            start_dt.date(),
            end_dt.date(),
        )
        return df

    return pd.DataFrame()


# ---------------------------------------------------------------------------
# Database operations
# ---------------------------------------------------------------------------


def _get_latest_stored_timestamp(conn, symbol: str, interval: str = "1m") -> str | None:
    """Query the latest stored bar timestamp for a symbol.

    Returns ISO timestamp string or None if no bars stored.
    """
    sql = _format_sql(_LATEST_BAR_SQL)
    try:
        cur = conn.execute(sql, (symbol, interval))
        row = cur.fetchone()
        if row is not None:
            # row[0] works for tuple, list, and sqlite3.Row alike
            val = row[0]
            return val
    except Exception as exc:
        logger.debug("Failed to query latest bar for %s: %s", symbol, exc)
    return None


def _get_bar_count(conn, symbol: str, interval: str = "1m") -> int:
    """Return the number of stored bars for a symbol."""
    sql = _format_sql(_COUNT_BARS_SQL)
    try:
        cur = conn.execute(sql, (symbol, interval))
        row = cur.fetchone()
        if row is not None:
            # row[0] works for tuple, list, and sqlite3.Row alike
            return int(row[0])
    except Exception:
        pass
    return 0


def _store_bars(conn, symbol: str, df: pd.DataFrame, interval: str = "1m") -> int:
    """Store bars from a DataFrame into the database.

    Uses INSERT OR IGNORE / ON CONFLICT DO NOTHING for idempotency.
    Returns the number of rows actually inserted (new bars).
    """
    if df.empty:
        return 0

    insert_sql = _INSERT_POSTGRES if _is_postgres() else _INSERT_SQLITE

    # Prepare rows
    rows = []
    for idx, row in df.iterrows():
        # Get timestamp as ISO string
        ts = idx.isoformat() if hasattr(idx, "isoformat") else str(idx)  # type: ignore[union-attr]

        try:
            o = float(row.get("Open", row.get("open", 0)))  # type: ignore[arg-type]
            h = float(row.get("High", row.get("high", 0)))  # type: ignore[arg-type]
            lo = float(row.get("Low", row.get("low", 0)))  # type: ignore[arg-type]
            c = float(row.get("Close", row.get("close", 0)))  # type: ignore[arg-type]
            v = int(row.get("Volume", row.get("volume", 0)) or 0)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue

        # Skip rows with zero OHLC (bad data)
        if o == 0 and h == 0 and lo == 0 and c == 0:
            continue

        rows.append((symbol, ts, o, h, lo, c, v, interval))

    if not rows:
        return 0

    # Count before insert
    count_before = _get_bar_count(conn, symbol, interval)

    # Insert in batches
    for i in range(0, len(rows), INSERT_BATCH_SIZE):
        batch = rows[i : i + INSERT_BATCH_SIZE]
        try:
            if _is_postgres():
                # Use executemany-style: execute each row
                for r in batch:
                    conn.execute(insert_sql, r)
            else:
                # SQLite supports executemany natively
                # But our wrapper may not, so do one at a time
                for r in batch:
                    conn.execute(insert_sql, r)
        except Exception as exc:
            logger.warning(
                "Batch insert failed for %s (batch %d-%d): %s",
                symbol,
                i,
                i + len(batch),
                exc,
            )

    try:
        conn.commit()
    except Exception as exc:
        logger.warning("Commit failed for %s: %s", symbol, exc)

    # Count after insert
    count_after = _get_bar_count(conn, symbol, interval)
    new_bars = max(0, count_after - count_before)

    return new_bars


# ---------------------------------------------------------------------------
# Core backfill logic
# ---------------------------------------------------------------------------


def _compute_date_range(
    symbol: str,
    conn,
    days_back: int = DEFAULT_DAYS_BACK,
    interval: str = "1m",
) -> tuple[datetime, datetime]:
    """Determine the date range to fetch for a symbol.

    If we already have some bars, start from the day after the latest
    stored bar.  Otherwise, go back ``days_back`` calendar days.

    Returns (start_dt, end_dt) as timezone-naive datetimes.
    """
    now = datetime.now(tz=UTC)
    end_dt = now

    latest_ts = _get_latest_stored_timestamp(conn, symbol, interval)
    if latest_ts:
        try:
            # Parse the stored timestamp
            latest = pd.Timestamp(latest_ts)
            if latest.tzinfo is None:
                latest = latest.tz_localize("UTC")
            # Start from 1 minute after the latest bar
            pdt = latest.to_pydatetime()
            if not isinstance(pdt, datetime):
                raise TypeError(f"Expected datetime, got {type(pdt)}")
            start_dt = pdt + timedelta(minutes=1)
            return start_dt, end_dt
        except Exception as exc:
            logger.debug(
                "Failed to parse latest timestamp '%s' for %s: %s",
                latest_ts,
                symbol,
                exc,
            )

    # No existing data — go back days_back
    start_dt = now - timedelta(days=days_back)
    return start_dt, end_dt


def _generate_chunks(
    start_dt: datetime,
    end_dt: datetime,
    chunk_days: int = CHUNK_DAYS,
    chunk_hours: int | None = None,
) -> list[tuple[datetime, datetime]]:
    """Split a date range into chunks of ``chunk_days`` calendar days (or
    ``chunk_hours`` hours when specified — used for Kraken 1-min data which
    is capped at 720 candles ≈ 12 hours per REST call).

    Returns a list of (chunk_start, chunk_end) tuples.
    """
    delta = timedelta(hours=chunk_hours) if chunk_hours is not None else timedelta(days=chunk_days)
    chunks = []
    current = start_dt
    while current < end_dt:
        chunk_end = min(current + delta, end_dt)
        chunks.append((current, chunk_end))
        current = chunk_end
    return chunks


def backfill_symbol(
    symbol: str,
    days_back: int | None = None,
    chunk_days: int | None = None,
    interval: str = "1m",
) -> dict[str, Any]:
    """Backfill historical bars for a single symbol.

    Returns a summary dict with keys:
      - symbol: ticker string
      - name: human-readable name
      - bars_before: number of bars before backfill
      - bars_after: number of bars after backfill
      - bars_added: number of new bars inserted
      - chunks_fetched: number of API chunks requested
      - chunks_with_data: number of chunks that returned data
      - start_date: earliest date requested
      - end_date: latest date requested
      - duration_seconds: wall-clock time for this symbol
      - error: error message if failed, empty string otherwise
    """
    name = _symbol_display_name(symbol)
    start_time = time.monotonic()

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

    # Apply per-ticker defaults: crypto uses tighter lookback + smaller chunks
    is_crypto = symbol.startswith("KRAKEN:")
    if days_back is None:
        days_back = KRAKEN_DAYS_BACK if is_crypto else DEFAULT_DAYS_BACK
    if chunk_days is None:
        chunk_days = CHUNK_DAYS
    # For Kraken, we override chunk generation to use hours instead of days
    # so each chunk fits within Kraken's 720-candle-per-call limit for 1m data.
    kraken_chunk_hours: int | None = KRAKEN_CHUNK_HOURS if is_crypto else None

    conn = None
    try:
        conn = _get_conn()

        # Count existing bars
        result["bars_before"] = _get_bar_count(conn, symbol, interval)

        # Determine date range
        start_dt, end_dt = _compute_date_range(symbol, conn, days_back=days_back, interval=interval)
        result["start_date"] = start_dt.strftime("%Y-%m-%d")
        result["end_date"] = end_dt.strftime("%Y-%m-%d")

        if start_dt >= end_dt:
            logger.info(
                "📊 %s (%s): already up to date (%d bars stored)",
                name,
                symbol,
                result["bars_before"],
            )
            result["bars_after"] = result["bars_before"]
            return result

        # Generate chunks — use hour-based chunks for Kraken crypto
        chunks = _generate_chunks(start_dt, end_dt, chunk_days, chunk_hours=kraken_chunk_hours)
        result["chunks_fetched"] = len(chunks)

        logger.info(
            "📊 %s (%s): fetching %d chunks from %s to %s (%d existing bars)",
            name,
            symbol,
            len(chunks),
            start_dt.strftime("%Y-%m-%d"),
            end_dt.strftime("%Y-%m-%d"),
            result["bars_before"],
        )

        total_new = 0
        for i, (chunk_start, chunk_end) in enumerate(chunks):
            try:
                df = fetch_bars_chunk(symbol, chunk_start, chunk_end)
                if not df.empty:
                    new_bars = _store_bars(conn, symbol, df, interval)
                    total_new += new_bars
                    result["chunks_with_data"] += 1
                    logger.debug(
                        "  Chunk %d/%d: %d bars fetched, %d new",
                        i + 1,
                        len(chunks),
                        len(df),
                        new_bars,
                    )
                else:
                    logger.debug(
                        "  Chunk %d/%d: no data",
                        i + 1,
                        len(chunks),
                    )
            except Exception as exc:
                logger.warning(
                    "  Chunk %d/%d failed for %s: %s",
                    i + 1,
                    len(chunks),
                    symbol,
                    exc,
                )

        result["bars_added"] = total_new
        result["bars_after"] = _get_bar_count(conn, symbol, interval)

        logger.info(
            "✅ %s (%s): +%d new bars (total: %d)",
            name,
            symbol,
            total_new,
            result["bars_after"],
        )

    except Exception as exc:
        result["error"] = str(exc)
        logger.error("❌ %s (%s): backfill failed — %s", name, symbol, exc)
    finally:
        if conn is not None:
            with contextlib.suppress(Exception):
                conn.close()
        result["duration_seconds"] = round(time.monotonic() - start_time, 2)

    return result


def run_backfill(
    symbols: list[str] | None = None,
    days_back: int | None = None,
    chunk_days: int | None = None,
    interval: str = "1m",
) -> dict[str, Any]:
    """Run a full historical backfill for all (or specified) symbols.

    This is the main entry point called by the engine scheduler.

    Crypto (KRAKEN:*) and futures symbols are handled with different default
    lookback windows and chunk sizes since Kraken's 1-min REST endpoint only
    exposes a limited history window.  Per-symbol overrides are applied inside
    ``backfill_symbol()`` when ``days_back`` / ``chunk_days`` are None.

    Args:
        symbols: List of tickers to backfill (None = all active assets).
        days_back: Number of calendar days to look back.  None = use
                   per-ticker defaults (7 days for crypto, 30 for futures).
        chunk_days: Number of days per API chunk.  None = per-ticker default.
        interval: Bar interval (default "1m").

    Returns:
        Summary dict with:
          - symbols: list of per-symbol result dicts
          - total_bars_added: int
          - total_duration_seconds: float
          - errors: list of error messages (if any)
          - status: "complete" or "partial" or "failed"
    """
    overall_start = time.monotonic()

    # Ensure table exists
    try:
        init_backfill_table()
    except Exception as exc:
        logger.error("Cannot initialise backfill table: %s", exc)
        return {
            "symbols": [],
            "total_bars_added": 0,
            "total_duration_seconds": 0.0,
            "errors": [f"Table init failed: {exc}"],
            "status": "failed",
        }

    if symbols is None:
        symbols = _get_backfill_symbols()

    logger.info(
        "=" * 60 + "\n"
        "  Historical Backfill Starting\n"
        "  Symbols: %s\n"
        "  Interval: %s (per-ticker defaults apply)\n" + "=" * 60,
        ", ".join(symbols),
        interval,
    )

    results: list[dict[str, Any]] = []
    errors: list[str] = []

    # Log crypto symbols separately so it's clear the Kraken path will be used
    crypto_syms = [s for s in symbols if s.startswith("KRAKEN:")]

    if crypto_syms:
        logger.info(
            "  Crypto (Kraken): %d pairs — %s",
            len(crypto_syms),
            ", ".join(crypto_syms),
        )

    for symbol in symbols:
        result = backfill_symbol(
            symbol,
            days_back=days_back,
            chunk_days=chunk_days,
            interval=interval,
        )
        results.append(result)
        if result["error"]:
            errors.append(f"{symbol}: {result['error']}")

    total_bars = sum(r["bars_added"] for r in results)
    total_duration = round(time.monotonic() - overall_start, 2)

    # Determine overall status
    if not results or (errors and len(errors) == len(results)):
        status = "failed"
    elif errors:
        status = "partial"
    else:
        status = "complete"

    summary = {
        "symbols": results,
        "total_bars_added": total_bars,
        "total_duration_seconds": total_duration,
        "errors": errors,
        "status": status,
    }

    # Publish to Redis for dashboard visibility
    _publish_backfill_status(summary)

    logger.info(
        "=" * 60 + "\n"
        "  Historical Backfill %s\n"
        "  Total new bars: %d across %d symbols\n"
        "  Duration: %.1f seconds\n"
        "  Errors: %d\n" + "=" * 60,
        status.upper(),
        total_bars,
        len(results),
        total_duration,
        len(errors),
    )

    return summary


# ---------------------------------------------------------------------------
# Query interface — used by optimization, backtesting, ORB detector
# ---------------------------------------------------------------------------


def get_stored_bars(
    symbol: str,
    days_back: int = 30,
    interval: str = "1m",
) -> pd.DataFrame:
    """Fetch stored historical bars from the database.

    Returns a DataFrame with columns [Open, High, Low, Close, Volume]
    and a DatetimeIndex, or empty DataFrame if no data.

    Args:
        symbol: Ticker symbol (e.g. "MGC=F").
        days_back: Number of calendar days to look back.
        interval: Bar interval (default "1m").

    Returns:
        DataFrame with OHLCV columns and DatetimeIndex.
    """
    conn = None
    try:
        conn = _get_conn()
        now = datetime.now(tz=UTC)
        start_dt = now - timedelta(days=days_back)

        start_str = start_dt.isoformat()
        end_str = now.isoformat()

        sql = _format_sql(_FETCH_BARS_SQL)
        cur = conn.execute(sql, (symbol, interval, start_str, end_str))
        rows = cur.fetchall()

        if not rows:
            return pd.DataFrame()

        # Convert rows to DataFrame
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

        # Set DatetimeIndex
        if not df.empty and "timestamp" in df.columns:
            try:
                df.index = pd.to_datetime(df["timestamp"], utc=True)
                df = df.drop(columns=["timestamp"])
                df = df.sort_index()
            except Exception:
                pass

        return df

    except Exception as exc:
        # Downgrade "no such table" to DEBUG — this fires on the GPU trainer
        # machine which has no local DB (all data comes from the engine over
        # the network).  It is not an error in that context; the caller's
        # fallback chain will handle it gracefully.
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


def get_backfill_status() -> dict[str, Any]:
    """Return the current backfill status: per-symbol bar counts and ranges.

    Returns a dict with:
      - symbols: list of {symbol, interval, bar_count, earliest, latest}
      - total_bars: int
    """
    conn = None
    try:
        conn = _get_conn()
        sql = _STATUS_SQL
        cur = conn.execute(sql)
        rows = cur.fetchall()

        symbols_info = []
        total = 0

        for row in rows:
            if isinstance(row, (tuple, list)):
                sym, intv, count, earliest, latest = row
            else:
                sym = row["symbol"]
                intv = row["interval"]
                count = row["bar_count"]
                earliest = row["earliest"]
                latest = row["latest"]

            count = int(count or 0)
            total += count
            symbols_info.append(
                {
                    "symbol": sym,
                    "name": _symbol_display_name(sym),
                    "interval": intv,
                    "bar_count": count,
                    "earliest": str(earliest) if earliest else None,
                    "latest": str(latest) if latest else None,
                }
            )

        return {
            "symbols": symbols_info,
            "total_bars": total,
        }

    except Exception as exc:
        logger.error("Failed to get backfill status: %s", exc)
        return {"symbols": [], "total_bars": 0, "error": str(exc)}
    finally:
        if conn is not None:
            with contextlib.suppress(Exception):
                conn.close()


def get_gap_report(
    symbol: str,
    days_back: int = 30,
    interval: str = "1m",
) -> dict[str, Any]:
    """Analyse gaps in stored historical data for a symbol.

    Returns a dict with:
      - symbol: str
      - total_bars: int
      - expected_bars: estimated count based on trading hours
      - coverage_pct: float (0-100)
      - gaps: list of {start, end, missing_minutes} for significant gaps
    """
    df = get_stored_bars(symbol, days_back=days_back, interval=interval)

    if df.empty:
        return {
            "symbol": symbol,
            "total_bars": 0,
            "expected_bars": 0,
            "coverage_pct": 0.0,
            "gaps": [],
        }

    total_bars = len(df)

    # Estimate expected bars: ~6.5 trading hours/day × 60 min × trading days
    trading_days = min(days_back, (df.index.max() - df.index.min()).days + 1)  # type: ignore[union-attr]
    # Roughly 5/7 of calendar days are trading days
    est_trading_days = max(1, int(trading_days * 5 / 7))
    # CME futures trade ~23 hours/day, but most activity in ~8 hours
    expected_bars = est_trading_days * 23 * 60  # conservative: full session

    coverage = min(100.0, (total_bars / expected_bars * 100)) if expected_bars > 0 else 0.0

    # Find gaps > 5 minutes (excluding overnight/weekend)
    gaps = []
    if len(df) > 1:
        time_diffs = df.index.to_series().diff()
        for i, diff in enumerate(time_diffs):
            if diff is not None and hasattr(diff, "total_seconds"):
                minutes = diff.total_seconds() / 60
                # Flag gaps > 30 minutes that aren't overnight (16:00-17:00 CT)
                if minutes > 30:
                    gap_start = df.index[i - 1] if i > 0 else df.index[0]
                    gap_end = df.index[i]
                    # Skip weekend gaps
                    if gap_start.weekday() == 4 and gap_end.weekday() == 6:  # type: ignore[union-attr]
                        continue
                    gaps.append(
                        {
                            "start": gap_start.isoformat(),  # type: ignore[union-attr]
                            "end": gap_end.isoformat(),  # type: ignore[union-attr]
                            "missing_minutes": int(minutes),
                        }
                    )

    return {
        "symbol": symbol,
        "total_bars": total_bars,
        "expected_bars": expected_bars,
        "coverage_pct": round(coverage, 1),
        "gaps": gaps[:50],  # Cap at 50 most recent gaps
    }


# ---------------------------------------------------------------------------
# Redis publishing
# ---------------------------------------------------------------------------


def _publish_backfill_status(summary: dict[str, Any]) -> None:
    """Publish backfill summary to Redis for dashboard consumption."""
    try:
        import json

        from lib.core.cache import REDIS_AVAILABLE, _r, cache_set

        payload = json.dumps(summary, default=str)
        cache_set("engine:backfill_status", payload.encode(), ttl=86400)  # 24h TTL

        if REDIS_AVAILABLE and _r is not None:
            with contextlib.suppress(Exception):
                _r.publish("dashboard:backfill", payload)

    except Exception as exc:
        logger.debug("Failed to publish backfill status: %s", exc)
