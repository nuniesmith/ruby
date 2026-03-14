"""
DataResolver — Unified Symbol Data Fetch with Cache-First Pipeline
===================================================================
Provides a single ``DataResolver`` class that resolves 1-minute OHLCV bars
for any supported symbol (CME futures or Kraken spot crypto) using a
four-tier cache hierarchy:

    0. Rithmic live  — real-time L1 snapshot when RITHMIC_LIVE_DATA=1
                       (single-row DataFrame; only for days <= 1 requests)
    1. Redis (hot)   — in-memory, sub-millisecond, 7-day window
    2. Postgres (warm) — durable, deep history up to 365 days
    3. API (cold)    — Massive REST (futures) or Kraken REST (crypto)

Priority chain: **Rithmic (if live) → Redis → Postgres → API**

Any data fetched from the cold path is immediately backfilled into both
Postgres and Redis so subsequent requests are served from warm/hot tiers.

Usage
-----
    from lib.services.data.resolver import DataResolver

    resolver = DataResolver()

    # Simple: resolve by symbol + days
    df = resolver.resolve("MES", days=90)

    # Explicit date range
    from datetime import datetime, UTC, timedelta
    end   = datetime.now(UTC)
    start = end - timedelta(days=30)
    df, meta = resolver.resolve_with_meta("KRAKEN:XBTUSD", start=start, end=end)
    print(meta)
    # {
    #   "symbol": "KRAKEN:XBTUSD",
    #   "source": "kraken_api",
    #   "rows": 43200,
    #   "cache_hit": False,
    #   "backfilled_redis": True,
    #   "backfilled_postgres": True,
    #   "duration_ms": 1842.3
    # }

    # Batch resolve — returns dict[symbol, DataFrame | None]
    frames = resolver.resolve_batch(["MES", "MNQ", "KRAKEN:XBTUSD"], days=90)

Architecture
------------
The resolver is stateless and thread-safe. Each ``resolve`` call is
independent. It is designed to be used from:
  - The CNN training pipeline (dataset_generator.py ``load_bars`` hot-path)
  - The ``startup_warm_caches`` background job at data-service startup
  - The engine focus computation
  - Any backfill-triggered gap-fill logic

Design goals
------------
- Cache-first: never call a remote API if Redis or Postgres has fresh data.
- Transparent: callers receive a plain DataFrame regardless of source.
- Observable: every resolve returns metadata about source + timings.
- Non-blocking backfill: Redis + Postgres writes happen inline but are
  wrapped in try/except so a Redis outage never blocks a training run.
- Kraken-aware: symbols with the ``KRAKEN:`` prefix are routed to the
  Kraken REST client; everything else goes to Massive / yfinance.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import pandas as pd

# ---------------------------------------------------------------------------
# Ticker resolution — inline copy to avoid import cycle with dataset_generator
# ---------------------------------------------------------------------------

_SYMBOL_TO_TICKER: dict[str, str] = {
    "MGC": "MGC=F",
    "SIL": "SIL=F",
    "MHG": "MHG=F",
    "MCL": "MCL=F",
    "MNG": "NG=F",
    "NG": "NG=F",
    "CL": "CL=F",
    "MES": "MES=F",
    "MNQ": "MNQ=F",
    "M2K": "M2K=F",
    "MYM": "MYM=F",
    "MBT": "MBT=F",
    "MET": "MET=F",
    "M6E": "6E=F",
    "M6B": "6B=F",
    "M6J": "6J=F",
    "6E": "6E=F",
    "6B": "6B=F",
    "6J": "6J=F",
    "6A": "6A=F",
    "6C": "6C=F",
    "6S": "6S=F",
    "ZN": "ZN=F",
    "ZB": "ZB=F",
    "ZF": "ZF=F",
    "ZT": "ZT=F",
    "ZC": "ZC=F",
    "ZS": "ZS=F",
    "ZW": "ZW=F",
    "ZL": "ZL=F",
    "ZM": "ZM=F",
    "GC": "GC=F",
    "ES": "ES=F",
    "NQ": "NQ=F",
    "SI": "SI=F",
    "HG": "HG=F",
    "YM": "YM=F",
    "RTY": "RTY=F",
    # CME crypto futures (disambiguated)
    "BTC_CME": "BTC=F",
    "ETH_CME": "ETH=F",
    # Kraken spot — short aliases route to KRAKEN: internal tickers
    "BTC": "KRAKEN:XBTUSD",
    "ETH": "KRAKEN:ETHUSD",
    "SOL": "KRAKEN:SOLUSD",
    "AVAX": "KRAKEN:AVAXUSD",
    "LINK": "KRAKEN:LINKUSD",
    "POL": "KRAKEN:POLUSD",
    "DOT": "KRAKEN:DOTUSD",
    "ADA": "KRAKEN:ADAUSD",
    "XRP": "KRAKEN:XRPUSD",
    # Kraken internal tickers
    "KRAKEN:XBTUSD": "KRAKEN:XBTUSD",
    "KRAKEN:ETHUSD": "KRAKEN:ETHUSD",
    "KRAKEN:SOLUSD": "KRAKEN:SOLUSD",
    "KRAKEN:AVAXUSD": "KRAKEN:AVAXUSD",
    "KRAKEN:LINKUSD": "KRAKEN:LINKUSD",
    "KRAKEN:POLUSD": "KRAKEN:POLUSD",
    "KRAKEN:DOTUSD": "KRAKEN:DOTUSD",
    "KRAKEN:ADAUSD": "KRAKEN:ADAUSD",
    "KRAKEN:XRPUSD": "KRAKEN:XRPUSD",
    # Kraken pair-style aliases
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


def _resolve_ticker(symbol: str) -> str:
    """Convert a short symbol (e.g. 'MGC') to a Yahoo/Kraken ticker ('MGC=F')."""
    if "=" in symbol:
        return symbol
    return _SYMBOL_TO_TICKER.get(symbol.upper(), f"{symbol.upper()}=F")


logger = logging.getLogger("data.resolver")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

KRAKEN_PREFIX = "KRAKEN:"

# Minimum bars required to consider a cached result "good enough" — avoids
# returning a near-empty Redis slice when Postgres has much more.
_MIN_BARS_REDIS = 100

# How many days of Redis bars are considered "fresh enough" to skip Postgres.
# If the caller asks for more days than this, we always go to Postgres.
_REDIS_MAX_DAYS = 7

# Kraken public REST is free; rate-limit is ~1 req/sec.  We don't need to
# be aggressive — if Postgres already has data, we won't call the API.
_KRAKEN_API_INTERVAL = "1m"


# ---------------------------------------------------------------------------
# Result metadata
# ---------------------------------------------------------------------------


@dataclass
class ResolveMetadata:
    """Metadata about a single DataResolver.resolve() call."""

    symbol: str
    source: str = "none"  # "rithmic_l1" | "redis" | "postgres" | "massive_api" | "kraken_api" | "none"
    rows: int = 0
    cache_hit: bool = False  # True when Redis satisfied the full request
    backfilled_redis: bool = False  # True when we wrote new data into Redis
    backfilled_postgres: bool = False  # True when we wrote new data into Postgres
    duration_ms: float = 0.0
    error: str | None = None
    start_dt: datetime | None = None
    end_dt: datetime | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "symbol": self.symbol,
            "source": self.source,
            "rows": self.rows,
            "cache_hit": self.cache_hit,
            "backfilled_redis": self.backfilled_redis,
            "backfilled_postgres": self.backfilled_postgres,
            "duration_ms": round(self.duration_ms, 1),
            "error": self.error,
            "start_dt": self.start_dt.isoformat() if self.start_dt else None,
            "end_dt": self.end_dt.isoformat() if self.end_dt else None,
        }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _is_kraken(symbol: str) -> bool:
    """Return True if the symbol is a Kraken spot crypto pair."""
    return symbol.upper().startswith(KRAKEN_PREFIX)


def _kraken_rest_pair(symbol: str) -> str:
    """Strip the KRAKEN: prefix to get the raw REST pair (e.g. XBTUSD)."""
    upper = symbol.upper()
    if upper.startswith(KRAKEN_PREFIX):
        return upper[len(KRAKEN_PREFIX) :]
    return upper


def _days_to_period(days: int) -> str:
    """Convert an integer day count to a Massive-compatible period string."""
    if days <= 1:
        return "1d"
    if days <= 5:
        return "5d"
    if days <= 10:
        return "10d"
    if days <= 15:
        return "15d"
    if days <= 30:
        return "1mo"
    if days <= 90:
        return "3mo"
    return "6mo"


def _try_redis(symbol: str, days: int) -> pd.DataFrame | None:
    """Attempt to read bars from Redis.

    Uses the ``engine:bars_1m_hist:{symbol}`` key written by ``_warm_redis_cache``
    in the bars router.  Returns None (not empty DataFrame) on any failure or
    miss so callers can distinguish "not available" from "zero rows".
    """
    try:
        from lib.core.cache import REDIS_AVAILABLE, _r

        if not REDIS_AVAILABLE or _r is None:
            return None

        key = f"engine:bars_1m_hist:{symbol}"
        raw = _r.get(key)
        if not raw:
            return None

        df: pd.DataFrame = pd.DataFrame(pd.read_json(raw, orient="columns"))
        if df.empty:
            return None

        # Ensure DatetimeIndex with UTC timezone
        df.index = pd.to_datetime(df.index, utc=True)

        # Filter to requested window
        if days < 365:
            cutoff = datetime.now(UTC) - timedelta(days=days)
            df = pd.DataFrame(df[df.index >= cutoff])

        if len(df) < _MIN_BARS_REDIS:
            return None

        return df

    except Exception as exc:
        logger.debug("Redis read failed for %s: %s", symbol, exc)
        return None


def _try_postgres(symbol: str, days: int) -> pd.DataFrame | None:
    """Attempt to read bars from Postgres/SQLite historical_bars table."""
    try:
        from lib.services.engine.backfill import get_stored_bars

        df = get_stored_bars(symbol, days_back=days, interval="1m")
        if df is not None and not df.empty and len(df) > 50:
            return df
    except ImportError:
        logger.debug("backfill module not available — skipping Postgres")
    except Exception as exc:
        logger.debug("Postgres read failed for %s: %s", symbol, exc)
    return None


def _try_massive(symbol: str, days: int) -> pd.DataFrame | None:
    """Attempt to fetch bars from the Massive REST API."""
    try:
        from lib.integrations.massive_client import get_massive_provider

        provider = get_massive_provider()
        if not provider.is_available:
            logger.debug("Massive provider not available (no API key)")
            return None

        ticker = _resolve_ticker(symbol)
        period = _days_to_period(days)

        df = provider.get_aggs(ticker, interval="1m", period=period)
        if df is not None and not df.empty:
            logger.info("Fetched %d bars for %s from Massive API", len(df), symbol)
            return df
    except ImportError:
        logger.debug("Massive client not available")
    except Exception as exc:
        logger.debug("Massive API fetch failed for %s: %s", symbol, exc)
    return None


def _try_kraken_api(symbol: str, days: int) -> pd.DataFrame | None:
    """Attempt to fetch OHLCV bars from Kraken REST API."""
    try:
        from lib.integrations.kraken_client import KrakenDataProvider

        provider = KrakenDataProvider()
        pair = _kraken_rest_pair(symbol)
        period = _days_to_period(days)

        df = provider.get_ohlcv_period(pair, interval=_KRAKEN_API_INTERVAL, period=period)
        if df is not None and not df.empty:
            logger.info("Fetched %d Kraken bars for %s (pair=%s)", len(df), symbol, pair)
            return df
    except ImportError:
        logger.debug("Kraken client not available")
    except Exception as exc:
        logger.debug("Kraken API fetch failed for %s: %s", symbol, exc)
    return None


def _backfill_postgres(symbol: str, df: pd.DataFrame) -> bool:
    """Write a DataFrame of bars into the historical_bars Postgres/SQLite table.

    Returns True if at least one row was inserted.
    """
    try:
        from lib.services.engine.backfill import _get_conn, _store_bars  # type: ignore[private-usage]

        conn = _get_conn()
        try:
            rows_added = _store_bars(conn, symbol, df, "1m")
            conn.commit()
            if rows_added > 0:
                logger.info("Backfilled %d new bars into Postgres for %s", rows_added, symbol)
            return rows_added > 0
        finally:
            import contextlib

            with contextlib.suppress(Exception):
                conn.close()
    except ImportError:
        logger.debug("backfill module not available — skipping Postgres backfill")
    except Exception as exc:
        logger.warning("Postgres backfill failed for %s: %s", symbol, exc)
    return False


def _backfill_redis(symbol: str, df: pd.DataFrame) -> bool:
    """Write a DataFrame of bars into Redis under the standard bars key.

    Uses the same key and TTL as ``_warm_redis_cache`` in the bars router
    so other consumers (engine, dataset generator) see the fresh data.
    Returns True on success.
    """
    try:
        from lib.core.cache import REDIS_AVAILABLE, _r

        if not REDIS_AVAILABLE or _r is None:
            return False

        key = f"engine:bars_1m_hist:{symbol}"

        # Merge with any existing Redis data to avoid losing history
        existing_raw = _r.get(key)
        if existing_raw:
            try:
                existing: pd.DataFrame = pd.read_json(existing_raw, orient="columns")
                if not existing.empty:
                    existing.index = pd.to_datetime(existing.index, utc=True)
                    combined = pd.concat([existing, df])
                    combined = combined[~combined.index.duplicated(keep="last")].sort_index()
                    df = pd.DataFrame(combined)
            except Exception:
                pass  # fall through to just write the new data

        json_str = df.to_json(date_format="iso")
        if json_str is None:
            return False

        # 8-hour TTL — same as _warm_redis_cache in bars.py
        _r.setex(key, 28_800, json_str.encode())
        logger.debug("Backfilled Redis cache for %s: %d bars", symbol, len(df))
        return True

    except Exception as exc:
        logger.debug("Redis backfill failed for %s: %s", symbol, exc)
    return False


# ---------------------------------------------------------------------------
# Rithmic Tier-0 helper (sync, uses sync Redis client)
# ---------------------------------------------------------------------------


def _try_rithmic_l1(symbol: str) -> pd.DataFrame | None:
    """Read the latest L1 snapshot from Redis ``rithmic:l1:{symbol}``.

    Returns a single-row DataFrame with columns ``open, high, low, close,
    volume`` built from the Rithmic tick snapshot, or ``None`` when:
      - ``RITHMIC_LIVE_DATA`` env var is not ``"1"``
      - The stream manager is not connected
      - The Redis key is absent or expired (2 s TTL written by ``on_tick``)

    This is intentionally synchronous so it fits cleanly into the existing
    sync ``_resolve_internal`` call path.  It uses the low-level ``_r`` Redis
    client (sync) rather than the async stream-manager methods.
    """
    if os.getenv("RITHMIC_LIVE_DATA", "0") != "1":
        return None

    try:
        # Guard: only attempt when stream manager says it's live
        from lib.integrations.rithmic_client import get_stream_manager

        if not get_stream_manager().is_live():
            return None
    except Exception:
        return None

    try:
        from lib.core.cache import REDIS_AVAILABLE, _r

        if not REDIS_AVAILABLE or _r is None:
            return None

        raw = _r.hgetall(f"rithmic:l1:{symbol}")
        if not raw:
            return None

        decoded: dict[str, float | int | None] = {}
        for k, v in raw.items():
            key_str = k.decode() if isinstance(k, bytes) else str(k)
            val_str = v.decode() if isinstance(v, bytes) else str(v)
            try:
                decoded[key_str] = float(val_str) if "." in val_str else int(val_str)
            except (ValueError, TypeError):
                decoded[key_str] = None

        last = decoded.get("last")
        if last is None:
            return None

        bid = decoded.get("bid", last)
        ask = decoded.get("ask", last)
        volume = decoded.get("volume", 0)
        ts_raw = decoded.get("ts")

        # Build a timestamp: prefer the tick ssboe, fall back to now
        if ts_raw is not None:
            try:
                ts = datetime.fromtimestamp(int(ts_raw), tz=UTC)
            except Exception:
                ts = datetime.now(UTC)
        else:
            ts = datetime.now(UTC)

        # Represent as a single-bar OHLCV row — open/high/low/close = last
        mid = (float(bid) + float(ask)) / 2.0 if bid is not None and ask is not None else float(last)
        df = pd.DataFrame(
            [
                {
                    "open": mid,
                    "high": mid,
                    "low": mid,
                    "close": mid,
                    "volume": int(volume) if volume else 0,
                }
            ],
            index=pd.DatetimeIndex([ts], tz=UTC),
        )
        logger.debug("DataResolver: Rithmic L1 hit for %s (last=%.4f)", symbol, mid)
        return df

    except Exception as exc:
        logger.debug("_try_rithmic_l1(%s) failed: %s", symbol, exc)
        return None


# ---------------------------------------------------------------------------
# DataResolver
# ---------------------------------------------------------------------------


class DataResolver:
    """Unified data resolver: Rithmic → Redis → Postgres → API with automatic backfill.

    Thread-safe and stateless — safe to share across threads or instantiate
    per-call.  All remote API calls are made synchronously (the training
    pipeline runs in a background thread where asyncio is not available).

    Parameters
    ----------
    enable_backfill_redis : bool
        Write newly API-fetched data back into Redis (default True).
    enable_backfill_postgres : bool
        Write newly API-fetched data back into Postgres (default True).
    skip_redis : bool
        Bypass the Redis tier entirely (useful for testing or when Redis is
        known to be unavailable). Default False.
    skip_postgres : bool
        Bypass the Postgres tier entirely. Default False.
    prefer_rithmic : bool
        When True, attempt Tier 0 (Rithmic live L1 snapshot) before Redis
        for requests with days <= 1.  Has no effect unless RITHMIC_LIVE_DATA=1
        and the stream manager is connected.  Default False.
    """

    def __init__(
        self,
        enable_backfill_redis: bool = True,
        enable_backfill_postgres: bool = True,
        skip_redis: bool = False,
        skip_postgres: bool = False,
        prefer_rithmic: bool = False,
    ) -> None:
        self.enable_backfill_redis = enable_backfill_redis
        self.enable_backfill_postgres = enable_backfill_postgres
        self.skip_redis = skip_redis
        self.skip_postgres = skip_postgres
        # When True, Tier 0 (Rithmic live L1) is attempted before Redis for
        # short-term requests (days <= 1).  Ignored when RITHMIC_LIVE_DATA != "1".
        self.prefer_rithmic = prefer_rithmic

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def resolve(
        self,
        symbol: str,
        days: int = 90,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> pd.DataFrame | None:
        """Resolve bars for *symbol*, returning a DataFrame or None.

        Convenience wrapper around :meth:`resolve_with_meta` that discards
        the metadata.

        Parameters
        ----------
        symbol:
            Short futures symbol (e.g. ``"MES"``) or Kraken internal ticker
            (e.g. ``"KRAKEN:XBTUSD"``).
        days:
            Number of calendar days of history to fetch.  Ignored when
            explicit *start* / *end* datetimes are provided.
        start, end:
            Optional explicit UTC-aware datetime range.  When provided,
            *days* is computed from the range and used as the API request
            window.
        """
        df, _ = self.resolve_with_meta(symbol, days=days, start=start, end=end)
        return df

    def resolve_with_meta(
        self,
        symbol: str,
        days: int = 90,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> tuple[pd.DataFrame | None, ResolveMetadata]:
        """Resolve bars for *symbol* and return (DataFrame, ResolveMetadata).

        Resolution order:
          1. Redis       — fast in-memory check (up to _REDIS_MAX_DAYS)
          2. Postgres    — durable historical table (deep history)
          3. API         — Massive (futures) or Kraken REST (crypto)
             └─ Backfill → Postgres + Redis for next time

        Parameters
        ----------
        symbol:
            Short futures symbol or Kraken internal ticker.
        days:
            History window in calendar days.
        start, end:
            Optional explicit UTC-aware datetime range.

        Returns
        -------
        (DataFrame | None, ResolveMetadata)
            DataFrame has a UTC-aware DatetimeIndex and OHLCV columns.
            Returns (None, meta) when all sources fail.
        """
        t0 = time.monotonic()
        meta = ResolveMetadata(symbol=symbol)

        # Normalise date range
        if end is None:
            end = datetime.now(UTC)
        if start is not None:
            days = max(1, int((end - start).total_seconds() / 86400) + 1)
        else:
            start = end - timedelta(days=days)

        meta.start_dt = start
        meta.end_dt = end

        try:
            df = self._resolve_internal(symbol, days, start, end, meta)
        except Exception as exc:
            logger.error("DataResolver.resolve_with_meta failed for %s: %s", symbol, exc, exc_info=True)
            meta.error = str(exc)
            df = None

        meta.duration_ms = (time.monotonic() - t0) * 1000
        if df is not None:
            meta.rows = len(df)
            # Trim to requested range
            df = pd.DataFrame(df[(df.index >= start) & (df.index <= end)])
            meta.rows = len(df)

        return df, meta

    def resolve_batch(
        self,
        symbols: list[str],
        days: int = 90,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> dict[str, pd.DataFrame | None]:
        """Resolve bars for multiple symbols, returning a dict of results.

        Symbols are processed sequentially to avoid hammering remote APIs.

        Parameters
        ----------
        symbols:
            List of symbol strings (mixed futures + Kraken OK).
        days, start, end:
            Same semantics as :meth:`resolve`.

        Returns
        -------
        dict[symbol, DataFrame | None]
        """
        results: dict[str, pd.DataFrame | None] = {}
        for sym in symbols:
            try:
                results[sym] = self.resolve(sym, days=days, start=start, end=end)
            except Exception as exc:
                logger.warning("resolve_batch: failed for %s: %s", sym, exc)
                results[sym] = None
        return results

    def resolve_batch_with_meta(
        self,
        symbols: list[str],
        days: int = 90,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> dict[str, tuple[pd.DataFrame | None, ResolveMetadata]]:
        """Same as :meth:`resolve_batch` but returns (df, meta) tuples."""
        results: dict[str, tuple[pd.DataFrame | None, ResolveMetadata]] = {}
        for sym in symbols:
            results[sym] = self.resolve_with_meta(sym, days=days, start=start, end=end)
        return results

    # ------------------------------------------------------------------
    # Internal resolution logic
    # ------------------------------------------------------------------

    def _resolve_internal(
        self,
        symbol: str,
        days: int,
        start: datetime,
        end: datetime,
        meta: ResolveMetadata,
    ) -> pd.DataFrame | None:
        """Core four-tier resolution logic (Rithmic → Redis → Postgres → API)."""
        is_crypto = _is_kraken(symbol)

        # ── Tier 0: Rithmic live L1 snapshot ─────────────────────────
        # Only attempted when:
        #   • RITHMIC_LIVE_DATA=1  (env gate)
        #   • prefer_rithmic=True  (caller opt-in)
        #   • days <= 1            (ticks cover only ~5 min of history;
        #                           longer requests fall through as normal)
        #   • non-crypto symbol    (Rithmic serves futures/equities only)
        if self.prefer_rithmic and not is_crypto and days <= 1 and os.getenv("RITHMIC_LIVE_DATA", "0") == "1":
            try:
                df = _try_rithmic_l1(symbol)
                if df is not None and not df.empty:
                    meta.source = "rithmic_l1"
                    meta.cache_hit = True
                    logger.debug(
                        "DataResolver: Rithmic L1 hit for %s (%d rows)",
                        symbol,
                        len(df),
                    )
                    return df
            except Exception as exc:
                # Never let Tier 0 block the normal resolution chain
                logger.debug("DataResolver: Rithmic Tier 0 skipped for %s: %s", symbol, exc)

        # ── Tier 1: Redis ────────────────────────────────────────────
        # Only use Redis when the request fits inside the Redis window AND
        # we have a meaningful number of bars there already.
        if not self.skip_redis and days <= _REDIS_MAX_DAYS:
            df = _try_redis(symbol, days)
            if df is not None and not df.empty:
                meta.source = "redis"
                meta.cache_hit = True
                logger.debug("DataResolver: Redis hit for %s (%d rows)", symbol, len(df))
                return df

        # ── Tier 2: Postgres ─────────────────────────────────────────
        if not self.skip_postgres:
            df = _try_postgres(symbol, days)
            if df is not None and not df.empty:
                meta.source = "postgres"
                meta.cache_hit = True
                logger.debug("DataResolver: Postgres hit for %s (%d rows)", symbol, len(df))

                # Opportunistically warm Redis for future hot-path hits
                if self.enable_backfill_redis and days <= _REDIS_MAX_DAYS:
                    try:
                        warmed = _backfill_redis(symbol, df)
                        meta.backfilled_redis = warmed
                    except Exception:
                        pass

                return df

        # ── Tier 3: Remote API ────────────────────────────────────────
        if is_crypto:
            df = _try_kraken_api(symbol, days)
            source_label = "kraken_api"
        else:
            df = _try_massive(symbol, days)
            source_label = "massive_api"

        if df is None or df.empty:
            logger.warning("DataResolver: all sources failed for %s", symbol)
            meta.source = "none"
            return None

        meta.source = source_label
        meta.cache_hit = False

        # ── Backfill Postgres ────────────────────────────────────────
        if self.enable_backfill_postgres and not self.skip_postgres:
            try:
                did_pg = _backfill_postgres(symbol, df)
                meta.backfilled_postgres = did_pg
            except Exception as exc:
                logger.warning("DataResolver: Postgres backfill failed for %s: %s", symbol, exc)

        # ── Backfill Redis ───────────────────────────────────────────
        if self.enable_backfill_redis and not self.skip_redis:
            try:
                did_redis = _backfill_redis(symbol, df)
                meta.backfilled_redis = did_redis
            except Exception as exc:
                logger.debug("DataResolver: Redis backfill failed for %s: %s", symbol, exc)

        return df


# ---------------------------------------------------------------------------
# Module-level singleton helper
# ---------------------------------------------------------------------------

_default_resolver: DataResolver | None = None


def get_resolver(
    *,
    enable_backfill_redis: bool = True,
    enable_backfill_postgres: bool = True,
) -> DataResolver:
    """Return (or create) the module-level shared DataResolver instance.

    Using a shared instance avoids re-importing heavy dependencies on every
    call while keeping the API simple for callers that don't need customisation.

    Note: the singleton is NOT reset when constructor arguments differ.
    For non-default configuration, instantiate ``DataResolver`` directly.
    """
    global _default_resolver
    if _default_resolver is None:
        _default_resolver = DataResolver(
            enable_backfill_redis=enable_backfill_redis,
            enable_backfill_postgres=enable_backfill_postgres,
        )
    return _default_resolver


def resolve(
    symbol: str,
    days: int = 90,
    start: datetime | None = None,
    end: datetime | None = None,
) -> pd.DataFrame | None:
    """Module-level shortcut: ``resolver.resolve(symbol, days)``.

    Equivalent to ``get_resolver().resolve(symbol, days)``.
    """
    return get_resolver().resolve(symbol, days=days, start=start, end=end)
