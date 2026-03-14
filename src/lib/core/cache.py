"""
Redis caching layer for market data and computed indicators.

Data source priority:
  1. Kraken REST API — for crypto pairs (KRAKEN:* tickers)
  2. Massive.com (formerly Polygon.io) — real-time futures data from CME/CBOT/NYMEX/COMEX
  3. yfinance — fallback when MASSIVE_API_KEY is not set or Massive call fails

Falls back to in-memory dict if Redis is unavailable, so the app
still works without Docker / Redis running.
"""

import contextlib
import hashlib
import json
import logging
import os
import re
from datetime import UTC, date, datetime, timedelta
from io import StringIO
from typing import Any

import pandas as pd
import yfinance as yf

logger = logging.getLogger("cache")


def _flatten_columns(df: "pd.DataFrame | None") -> pd.DataFrame:
    """Flatten MultiIndex columns returned by newer yfinance versions."""
    if df is None or df.empty:
        return pd.DataFrame()
    # At this point df is guaranteed to be a non-empty DataFrame
    result: pd.DataFrame = df.copy()
    if isinstance(result.columns, pd.MultiIndex):
        # Flatten to single level: take first level names only
        result.columns = pd.Index([col[0] if isinstance(col, tuple) else col for col in result.columns])
    # Remove duplicate columns (keep first occurrence)
    mask = ~pd.Index(result.columns).duplicated(keep="first")
    result = result.loc[:, mask]
    # Reset column names to plain strings to avoid any leftover index weirdness
    result.columns = pd.Index([str(c) for c in result.columns])
    # Drop rows with NaN in OHLCV columns (yfinance sometimes returns partial rows)
    ohlcv = [c for c in ("Open", "High", "Low", "Close", "Volume") if c in result.columns]
    if ohlcv:
        result = result.dropna(subset=ohlcv)
    return result


def _init_redis() -> "tuple[Any, bool]":
    """Attempt to connect to Redis and return (client, available) tuple."""
    try:
        import redis  # type: ignore[import-unresolved]

        _redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        client = redis.from_url(_redis_url, decode_responses=False)
        client.ping()  # type: ignore[union-attr]
        return client, True
    except Exception:
        return None, False


_r, REDIS_AVAILABLE = _init_redis()

# Fallback in-memory cache when Redis is not available
_mem_cache: dict[str, Any] = {}

# Default TTLs in seconds
TTL_MINUTE = 30  # 1-min OHLCV — short TTL for live freshness
TTL_INTRADAY = 60  # 5-min / 15-min OHLCV
TTL_DAILY = 300  # daily bars (pivots, etc.)
TTL_INDICATOR = 120  # computed indicators (ATR, EMA, VWAP)
TTL_OPTIMIZATION = 3600  # optimization results (1 hour)


def _cache_key(*parts: str) -> str:
    raw = ":".join(str(p) for p in parts)
    return "futures:" + hashlib.md5(raw.encode()).hexdigest()


def _df_to_bytes(df: pd.DataFrame) -> bytes:
    # Safety net: deduplicate columns before serialising to JSON
    if df.columns.duplicated().any():
        mask = ~pd.Index(df.columns).duplicated(keep="first")
        df = df.loc[:, mask]
    return (df.to_json(date_format="iso") or "").encode()


def _bytes_to_df(raw: bytes) -> pd.DataFrame:
    df = pd.read_json(StringIO(raw.decode()))
    # Restore DatetimeIndex if the index looks like timestamps
    if not df.empty and df.index.dtype == "int64":
        pass  # leave numeric index as-is
    elif not df.empty:
        with contextlib.suppress(Exception):
            df.index = pd.to_datetime(df.index)
    return df


# ---------------------------------------------------------------------------
# Low-level get / set
# ---------------------------------------------------------------------------


def cache_get(key: str) -> bytes | None:
    if REDIS_AVAILABLE and _r is not None:
        result = _r.get(key)
        if isinstance(result, bytes):
            return result
        return None
    entry = _mem_cache.get(key)
    if entry is None:
        return None
    if datetime.now(tz=UTC).timestamp() > entry["expires"]:
        del _mem_cache[key]
        return None
    return entry["data"]


def cache_set(key: str, data: bytes, ttl: int) -> None:
    if REDIS_AVAILABLE and _r is not None:
        _r.setex(key, ttl, data)
    else:
        _mem_cache[key] = {
            "data": data,
            "expires": datetime.now(tz=UTC).timestamp() + ttl,
        }


# ---------------------------------------------------------------------------
# Yahoo Finance interval → max period limits
# ---------------------------------------------------------------------------
# Yahoo enforces these caps on intraday data. Requesting beyond them returns
# an empty frame or a "possibly delisted" error.
_YF_MAX_PERIOD: dict[str, list[str]] = {
    # interval → ordered list of allowed periods (largest last)
    # Custom day periods (e.g. 10d, 15d) are handled via start/end dates
    # in get_data(), so they are safe to list here.
    "1m": ["1d", "5d"],
    "2m": ["1d", "5d", "10d", "15d", "1mo"],
    "5m": ["1d", "5d", "10d", "15d", "1mo"],
    "15m": ["1d", "5d", "10d", "15d", "1mo"],
    "30m": ["1d", "5d", "10d", "15d", "1mo"],
    "60m": ["1d", "5d", "10d", "15d", "1mo", "3mo", "6mo"],
    "1h": ["1d", "5d", "10d", "15d", "1mo", "3mo", "6mo"],
    "90m": ["1d", "5d", "10d", "15d", "1mo", "3mo", "6mo"],
    "1d": [
        "1d",
        "5d",
        "10d",
        "15d",
        "1mo",
        "3mo",
        "6mo",
        "1y",
        "2y",
        "5y",
        "10y",
        "max",
    ],
    "5d": ["1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "max"],
    "1wk": ["1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "max"],
    "1mo": ["3mo", "6mo", "1y", "2y", "5y", "10y", "max"],
    "3mo": ["1y", "2y", "5y", "10y", "max"],
}

# Numeric ordering so we can compare periods
_PERIOD_RANK: dict[str, int] = {
    "1d": 1,
    "5d": 5,
    "10d": 10,
    "15d": 15,
    "1mo": 30,
    "3mo": 90,
    "6mo": 180,
    "1y": 365,
    "2y": 730,
    "5y": 1825,
    "10y": 3650,
    "max": 99999,
}

# Standard Yahoo Finance periods that can be passed directly as `period=`.
# Anything NOT in this set (e.g. 10d, 15d) will be converted to explicit
# start/end dates, which Yahoo handles reliably for any day count.
_YF_NATIVE_PERIODS = {
    "1d",
    "5d",
    "1mo",
    "3mo",
    "6mo",
    "1y",
    "2y",
    "5y",
    "10y",
    "ytd",
    "max",
}


def _period_to_days(period: str) -> int | None:
    """Parse a period string like '10d' into an integer number of days.

    Returns None if the period is not a simple '<N>d' format.
    """
    m = re.match(r"^(\d+)d$", period)
    return int(m.group(1)) if m else None


def _clamp_period(interval: str, period: str) -> str:
    """Return the largest Yahoo-allowed period that is ≤ the requested one.

    If the requested period exceeds the interval's max, it is clamped down
    and a message is printed so the user knows.
    """
    allowed = _YF_MAX_PERIOD.get(interval)
    if allowed is None:
        # Unknown interval – pass through and let Yahoo decide
        return period

    req_rank = _PERIOD_RANK.get(period, 90)

    # If the requested period is within the allowed list, use it directly
    if period in allowed:
        return period

    # Otherwise find the largest allowed period that doesn't exceed the request
    best = allowed[0]  # smallest fallback
    for p in allowed:
        if _PERIOD_RANK.get(p, 0) <= req_rank:
            best = p

    if best != period:
        print(f"[cache] Clamped period {period!r} → {best!r} for interval {interval!r} (Yahoo limit)")
    return best


# ---------------------------------------------------------------------------
# Market data fetching with cache
# ---------------------------------------------------------------------------


def _yf_download(ticker: str, interval: str, period: str, **kwargs) -> pd.DataFrame:
    """Wrapper around yf.download that converts non-standard periods
    (like 10d, 15d) to explicit start/end dates.

    Yahoo Finance natively supports only a fixed set of period strings
    (1d, 5d, 1mo, 3mo, …). Custom day-based periods like '10d' or '15d'
    are unreliable when passed as `period=`. Instead, we compute the
    calendar start date and use `start=` / `end=` which Yahoo handles
    consistently for any timeframe.
    """
    days = _period_to_days(period)
    if days is not None and period not in _YF_NATIVE_PERIODS:
        # Convert to start/end dates for reliable fetching
        end_dt = date.today() + timedelta(days=1)  # include today
        start_dt = end_dt - timedelta(days=days)
        print(f"[cache] Converting period {period!r} → start={start_dt}, end={end_dt}")
        return _flatten_columns(
            yf.download(
                ticker,
                interval=interval,
                start=str(start_dt),
                end=str(end_dt),
                **kwargs,
            )
        )
    else:
        return _flatten_columns(yf.download(ticker, interval=interval, period=period, **kwargs))


# ---------------------------------------------------------------------------
# Data source: Massive.com (primary) with yfinance fallback
# ---------------------------------------------------------------------------

# Lazy import — massive_client is optional; if MASSIVE_API_KEY isn't set
# the provider reports is_available=False and we skip straight to yfinance.
_massive_provider = None
_massive_checked = False


def _get_massive_provider():
    """Lazily initialise the Massive data provider singleton."""
    global _massive_provider, _massive_checked
    if not _massive_checked:
        try:
            from lib.integrations.massive_client import get_massive_provider

            _massive_provider = get_massive_provider()
        except Exception as exc:
            logger.debug("Massive provider unavailable: %s", exc)
            _massive_provider = None
        _massive_checked = True
    return _massive_provider


def _try_massive(ticker: str, interval: str, period: str) -> pd.DataFrame:
    """Attempt to fetch OHLCV data from Massive.

    Returns a non-empty DataFrame on success, or an empty DataFrame
    if Massive is unavailable or the call fails.
    """
    provider = _get_massive_provider()
    if provider is None or not provider.is_available:
        return pd.DataFrame()

    try:
        df = provider.get_aggs(ticker, interval=interval, period=period)
        if not df.empty:
            logger.debug("Massive: got %d bars for %s %s/%s", len(df), ticker, interval, period)
        return df
    except Exception as exc:
        logger.debug("Massive fetch failed for %s: %s", ticker, exc)
        return pd.DataFrame()


def _try_massive_daily(ticker: str, period: str) -> pd.DataFrame:
    """Attempt to fetch daily bars from Massive."""
    provider = _get_massive_provider()
    if provider is None or not provider.is_available:
        return pd.DataFrame()

    try:
        df = provider.get_daily(ticker, period=period)
        if not df.empty:
            logger.debug("Massive: got %d daily bars for %s/%s", len(df), ticker, period)
        return df
    except Exception as exc:
        logger.debug("Massive daily fetch failed for %s: %s", ticker, exc)
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Data source: Kraken exchange (crypto pairs)
# ---------------------------------------------------------------------------

_kraken_provider = None
_kraken_checked = False


def _get_kraken_provider():
    """Lazily initialise the Kraken data provider singleton."""
    global _kraken_provider, _kraken_checked
    if not _kraken_checked:
        try:
            from lib.integrations.kraken_client import get_kraken_provider

            _kraken_provider = get_kraken_provider()
        except Exception as exc:
            logger.debug("Kraken provider unavailable: %s", exc)
            _kraken_provider = None
        _kraken_checked = True
    return _kraken_provider


def _is_kraken_ticker(ticker: str) -> bool:
    """Return True if *ticker* is a Kraken crypto pair (KRAKEN:* prefix)."""
    return ticker.startswith("KRAKEN:")


def _try_kraken(ticker: str, interval: str, period: str) -> pd.DataFrame:
    """Attempt to fetch OHLCV data from Kraken for a KRAKEN:* ticker.

    Returns a non-empty DataFrame on success, or an empty DataFrame
    if Kraken is unavailable or the call fails.
    """
    provider = _get_kraken_provider()
    if provider is None or not provider.is_available:
        return pd.DataFrame()

    try:
        df = provider.get_ohlcv_period(ticker, interval=interval, period=period)
        if not df.empty:
            logger.debug("Kraken: got %d bars for %s %s/%s", len(df), ticker, interval, period)
        return df
    except Exception as exc:
        logger.debug("Kraken fetch failed for %s: %s", ticker, exc)
        return pd.DataFrame()


def _try_kraken_daily(ticker: str, period: str) -> pd.DataFrame:
    """Attempt to fetch daily bars from Kraken for a KRAKEN:* ticker."""
    return _try_kraken(ticker, interval="1d", period=period)


def get_data_source(ticker: str | None = None) -> str:
    """Return the name of the active primary data source.

    If *ticker* is provided, returns the specific source that would be
    used for that ticker.  Otherwise returns the general default.

    Returns 'Kraken' for KRAKEN:* tickers, 'Massive' if the Massive API
    is configured, otherwise 'yfinance'.
    """
    if ticker and _is_kraken_ticker(ticker):
        return "Kraken"
    provider = _get_massive_provider()
    if provider is not None and provider.is_available:
        return "Massive"
    return "yfinance"


def get_data(ticker: str, interval: str, period: str) -> pd.DataFrame:
    """Fetch OHLCV data, cached in Redis for TTL_INTRADAY seconds.

    Data source priority:
      1. Kraken REST API (for KRAKEN:* crypto tickers)
      2. Massive.com REST API (if MASSIVE_API_KEY is set)
      3. yfinance (fallback)

    Automatically clamps the period to Yahoo Finance's maximum for the
    requested interval to avoid empty responses. Non-standard periods
    (e.g. 10d, 15d) are converted to start/end dates for reliability.
    """
    # Kraken crypto tickers bypass the Yahoo/Massive pipeline entirely
    if _is_kraken_ticker(ticker):
        key = _cache_key("ohlcv", ticker, interval, period)
        cached = cache_get(key)
        if cached is not None:
            return _bytes_to_df(cached)

        df = _try_kraken(ticker, interval, period)
        if not df.empty:
            ttl = TTL_MINUTE if interval == "1m" else TTL_INTRADAY
            cache_set(key, _df_to_bytes(df), ttl)
        return df

    clamped_period = _clamp_period(interval, period)
    key = _cache_key("ohlcv", ticker, interval, clamped_period)
    cached = cache_get(key)
    if cached is not None:
        return _bytes_to_df(cached)

    # Try Massive first
    df = _try_massive(ticker, interval, period)

    # Fallback to yfinance
    if df.empty:
        df = _yf_download(ticker, interval, clamped_period, prepost=True, auto_adjust=True)

    if not df.empty:
        ttl = TTL_MINUTE if interval == "1m" else TTL_INTRADAY
        cache_set(key, _df_to_bytes(df), ttl)
    return df


def get_daily(ticker: str, period: str = "10d") -> pd.DataFrame:
    """Fetch daily bars, cached longer since they change less often.

    Routes KRAKEN:* tickers to Kraken REST API.
    Tries Massive first for futures, falls back to yfinance.
    """
    key = _cache_key("daily", ticker, period)
    cached = cache_get(key)
    if cached is not None:
        return _bytes_to_df(cached)

    # Kraken crypto tickers
    if _is_kraken_ticker(ticker):
        df = _try_kraken_daily(ticker, period)
        if not df.empty:
            cache_set(key, _df_to_bytes(df), TTL_DAILY)
        return df

    # Try Massive first
    df = _try_massive_daily(ticker, period)

    # Fallback to yfinance
    if df.empty:
        df = _yf_download(ticker, "1d", period, auto_adjust=True)

    if not df.empty:
        cache_set(key, _df_to_bytes(df), TTL_DAILY)
    return df


# ---------------------------------------------------------------------------
# Indicator caching
# ---------------------------------------------------------------------------


def get_cached_indicator(name: str, ticker: str, interval: str, period: str) -> dict[str, Any] | None:
    """Return cached indicator dict or None."""
    key = _cache_key("ind", name, ticker, interval, period)
    cached = cache_get(key)
    if cached is not None:
        return json.loads(cached.decode())
    return None


def set_cached_indicator(name: str, ticker: str, interval: str, period: str, payload: dict[str, Any]) -> None:
    key = _cache_key("ind", name, ticker, interval, period)
    cache_set(key, json.dumps(payload).encode(), TTL_INDICATOR)


# ---------------------------------------------------------------------------
# Optimization results caching
# ---------------------------------------------------------------------------


def get_cached_optimization(ticker: str, interval: str, period: str) -> dict[str, Any] | None:
    key = _cache_key("opt", ticker, interval, period)
    cached = cache_get(key)
    if cached is not None:
        return json.loads(cached.decode())
    return None


def set_cached_optimization(ticker: str, interval: str, period: str, result: dict[str, Any]) -> None:
    key = _cache_key("opt", ticker, interval, period)
    cache_set(key, json.dumps(result).encode(), TTL_OPTIMIZATION)


def clear_cached_optimization(ticker: str, interval: str, period: str) -> None:
    """Remove cached optimization result so the next run re-optimizes."""
    key = _cache_key("opt", ticker, interval, period)
    if REDIS_AVAILABLE and _r is not None:
        _r.delete(key)
    elif key in _mem_cache:
        del _mem_cache[key]


def flush_all() -> None:
    """Clear all cached data (used by refresh button)."""
    if REDIS_AVAILABLE and _r is not None:
        for key in _r.scan_iter("futures:*"):
            _r.delete(key)
    else:
        _mem_cache.clear()
