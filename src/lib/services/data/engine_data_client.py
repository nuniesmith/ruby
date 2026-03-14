"""
Engine Data Client
==================
Generic data-fetching adapter for ``lib.services.data`` modules.

All analysis code (crypto_momentum, scorer, wave_analysis, volume_profile,
confluence, signal_quality, etc.) must obtain OHLCV bar data through this
module rather than calling Kraken, Massive, yfinance, Redis, or Postgres
directly.

Architecture
------------
::

    ┌──────────────────────────────────────┐
    │  Analysis / Strategy layer           │
    │  (crypto_momentum, scorer, ...)      │
    └────────────────┬─────────────────────┘
                     │  get_bars() / get_snapshot() / ...
                     ▼
    ┌──────────────────────────────────────┐
    │  EngineDataClient   (this module)    │
    │  • HTTP → engine /bars/{symbol}      │
    │  • HTTP → engine /bars/symbols       │
    │  • HTTP → engine /market_data/...    │
    │  • in-process TTL cache (5 min)      │
    └────────────────┬─────────────────────┘
                     │  REST
                     ▼
    ┌──────────────────────────────────────┐
    │  Engine / Data Service               │
    │  (cloud box)                         │
    │  Redis → Postgres → Massive/Kraken   │
    └──────────────────────────────────────┘

The client is intentionally thin:

- It carries **no business logic** — it is purely I/O.
- It has an **in-process TTL cache** (default 5 min) so analysis
  modules that call ``get_bars()`` on every candle do not hammer the
  engine on every bar.
- All methods return ``None`` / empty results on any error so callers
  can degrade gracefully.
- The engine URL, API key, and timeouts are read from environment
  variables so the same code works in Docker, local dev, and tests.

Environment variables
---------------------
ENGINE_DATA_URL
    Base URL of the engine / data service (default ``http://data:8000``).
API_KEY
    Shared secret for ``X-API-Key`` inter-service auth (default: empty).
ENGINE_BARS_TIMEOUT
    Seconds to wait for a ``/bars/`` call before giving up (default 60).
ENGINE_SNAPSHOT_TIMEOUT
    Seconds to wait for a snapshot call (default 10).
ENGINE_CACHE_TTL
    In-process cache TTL in seconds (default 300 = 5 min).

Usage
-----
::

    from lib.services.data.engine_data_client import get_client

    client = get_client()

    # 1-minute bars for the last 90 days
    df = client.get_bars("MGC", interval="1m", days_back=90)

    # 5-minute bars for the last 5 days
    df = client.get_bars("BTC", interval="5m", days_back=5)

    # Latest snapshot (price, volume, ATR)
    snap = client.get_snapshot("MES")

    # All enabled symbols from models.ASSETS
    symbols = client.get_symbols()

    # Multiple symbols at once (uses /bars/bulk if available)
    bars_dict = client.get_bars_bulk(["MGC", "MES", "MNQ"], days_back=5)

    # Convenience: bars with automatic fallback message
    df = client.require_bars("MGC", min_bars=100)
"""

from __future__ import annotations

import logging
import os
import threading
import time as _time_mod
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

logger = logging.getLogger("services.data.engine_data_client")

# ---------------------------------------------------------------------------
# Configuration (read once at module import)
# ---------------------------------------------------------------------------

_ENGINE_DATA_URL: str = (os.getenv("ENGINE_DATA_URL") or os.getenv("DATA_SERVICE_URL") or "http://data:8000").rstrip(
    "/"
)

_API_KEY: str = os.getenv("API_KEY", "").strip()

_BARS_TIMEOUT: int = int(os.getenv("ENGINE_BARS_TIMEOUT", "60"))
_SNAPSHOT_TIMEOUT: int = int(os.getenv("ENGINE_SNAPSHOT_TIMEOUT", "10"))
_CACHE_TTL: float = float(os.getenv("ENGINE_CACHE_TTL", "300"))  # 5 minutes

# ---------------------------------------------------------------------------
# In-process TTL cache
# ---------------------------------------------------------------------------
# Keyed by (symbol, interval, days_back) → (fetched_at: float, DataFrame).
# A single lock protects all entries.  Eviction is lazy (on next read).

_CacheKey = tuple[str, str, int]


@dataclass
class _CacheEntry:
    fetched_at: float
    value: Any  # pd.DataFrame | dict | list | None


_cache: dict[str, _CacheEntry] = {}
_cache_lock = threading.Lock()


def _cache_get(key: str) -> Any:
    """Return cached value if still fresh, else None."""
    with _cache_lock:
        entry = _cache.get(key)
        if entry is None:
            return None
        if _time_mod.monotonic() - entry.fetched_at > _CACHE_TTL:
            del _cache[key]
            return None
        return entry.value


def _cache_set(key: str, value: Any) -> None:
    with _cache_lock:
        _cache[key] = _CacheEntry(fetched_at=_time_mod.monotonic(), value=value)


def _cache_key(*parts: Any) -> str:
    return ":".join(str(p) for p in parts)


def clear_cache() -> None:
    """Evict all in-process cache entries (useful in tests)."""
    with _cache_lock:
        _cache.clear()


# ---------------------------------------------------------------------------
# DataFrame reconstruction helpers
# ---------------------------------------------------------------------------


def _split_to_df(split: dict[str, Any]) -> pd.DataFrame | None:
    """Convert a ``split``-orientation payload to a DataFrame.

    The engine's ``/bars/{symbol}`` endpoint returns data in pandas
    ``split`` format::

        {
            "columns": ["Open", "High", "Low", "Close", "Volume"],
            "index":   ["2025-01-06T09:30:00+00:00", ...],
            "data":    [[2712.4, 2714.1, 2711.8, 2712.9, 123], ...]
        }

    Reconstructed with ``pd.DataFrame(**split)``.
    """
    if not split or not isinstance(split, dict):
        return None

    columns = split.get("columns", [])
    index_raw = split.get("index", [])
    data = split.get("data", [])

    if not columns or data is None:
        return None

    try:
        df = pd.DataFrame(data, columns=columns)

        if index_raw:
            idx = pd.to_datetime(index_raw, utc=True)
            df.index = idx

        # Ensure OHLCV columns are float
        for col in ("Open", "High", "Low", "Close", "Volume"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        return df if not df.empty else None
    except Exception as exc:
        logger.debug("_split_to_df failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Main client class
# ---------------------------------------------------------------------------


class EngineDataClient:
    """HTTP client for the engine / data service bar endpoints.

    Designed to be used as a module-level singleton (via ``get_client()``)
    but can also be instantiated directly for testing with a custom URL.
    """

    def __init__(
        self,
        base_url: str = _ENGINE_DATA_URL,
        api_key: str = _API_KEY,
        bars_timeout: int = _BARS_TIMEOUT,
        snapshot_timeout: int = _SNAPSHOT_TIMEOUT,
        cache_ttl: float = _CACHE_TTL,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._bars_timeout = bars_timeout
        self._snapshot_timeout = snapshot_timeout
        self._cache_ttl = cache_ttl

    # ------------------------------------------------------------------
    # Internal HTTP helper
    # ------------------------------------------------------------------

    def _headers(self) -> dict[str, str]:
        if self._api_key:
            return {"X-API-Key": self._api_key}
        return {}

    def _get(self, path: str, params: dict[str, Any] | None = None, timeout: int | None = None) -> Any | None:
        """Issue a GET request and return the parsed JSON, or None on error.

        Lazily imports ``requests`` so that analysis modules that never
        call the engine (e.g. when running against pre-loaded DataFrames
        in tests) do not require ``requests`` to be installed.
        """
        try:
            import requests as _requests  # type: ignore[import-untyped]
        except ImportError:
            logger.debug("requests not available — cannot call engine")
            return None

        url = f"{self.base_url}{path}"
        _timeout = timeout if timeout is not None else self._bars_timeout

        try:
            resp = _requests.get(url, params=params, headers=self._headers(), timeout=_timeout)
            if resp.status_code == 200:
                return resp.json()
            if resp.status_code == 404:
                logger.debug("Engine 404 for %s (params=%s)", path, params)
            else:
                logger.debug(
                    "Engine returned HTTP %d for %s (params=%s): %s",
                    resp.status_code,
                    path,
                    params,
                    resp.text[:200],
                )
        except Exception as exc:
            logger.debug("Engine request failed for %s: %s", path, exc)
        return None

    def _post(self, path: str, body: dict[str, Any], timeout: int | None = None) -> Any | None:
        """Issue a POST request and return parsed JSON, or None on error."""
        try:
            import requests as _requests  # type: ignore[import-untyped]
        except ImportError:
            logger.debug("requests not available — cannot call engine")
            return None

        url = f"{self.base_url}{path}"
        _timeout = timeout if timeout is not None else self._bars_timeout

        try:
            resp = _requests.post(url, json=body, headers=self._headers(), timeout=_timeout)
            if resp.status_code == 200:
                return resp.json()
            logger.debug(
                "Engine POST %s returned HTTP %d: %s",
                path,
                resp.status_code,
                resp.text[:200],
            )
        except Exception as exc:
            logger.debug("Engine POST failed for %s: %s", path, exc)
        return None

    # ------------------------------------------------------------------
    # Bar data
    # ------------------------------------------------------------------

    def get_bars(
        self,
        symbol: str,
        interval: str = "1m",
        days_back: int = 90,
        *,
        auto_fill: bool = True,
        use_cache: bool = True,
    ) -> pd.DataFrame | None:
        """Fetch OHLCV bars for ``symbol`` from the engine.

        Routes to ``GET /bars/{symbol}?interval=…&days_back=…&auto_fill=…``.

        The engine resolves the full data hierarchy internally:
        Redis cache → Postgres DB → Massive/Kraken REST → yfinance fallback.
        Analysis code should never need to know which tier provided the data.

        Parameters
        ----------
        symbol:
            Short symbol (``"MGC"``) or Yahoo-style ticker (``"MGC=F"``).
            Kraken spot aliases (``"BTC"``, ``"KRAKEN:XBTUSD"``) are also
            accepted — the engine normalises them.
        interval:
            OHLCV bar interval: ``"1m"``, ``"5m"``, ``"15m"``, ``"1h"``,
            ``"1d"``.
        days_back:
            How many calendar days of history to request.
        auto_fill:
            Ask the engine to fill any data gaps before responding.
        use_cache:
            Whether to check / populate the in-process TTL cache.

        Returns
        -------
        pd.DataFrame | None
            OHLCV DataFrame with a UTC DatetimeIndex and columns
            ``["Open", "High", "Low", "Close", "Volume"]``, or ``None``
            if the engine is unreachable or has no data for the symbol.
        """
        key = _cache_key("bars", symbol, interval, days_back)
        if use_cache:
            cached = _cache_get(key)
            if cached is not None:
                return cached

        payload = self._get(
            f"/bars/{symbol}",
            params={
                "interval": interval,
                "days_back": days_back,
                "auto_fill": str(auto_fill).lower(),
            },
        )

        if not payload:
            return None

        bar_count = payload.get("bar_count", 0)
        if bar_count == 0:
            logger.debug("Engine returned 0 bars for %s/%s (%d days)", symbol, interval, days_back)
            return None

        split = payload.get("data")
        df = _split_to_df(split)
        if df is None or df.empty:
            return None

        if use_cache:
            _cache_set(key, df)

        logger.debug(
            "Engine: %d bars for %s/%s (%d days back)",
            len(df),
            symbol,
            interval,
            days_back,
        )
        return df

    def require_bars(
        self,
        symbol: str,
        interval: str = "1m",
        days_back: int = 90,
        min_bars: int = 30,
    ) -> pd.DataFrame | None:
        """Like ``get_bars()`` but logs a clear warning when insufficient data.

        Returns ``None`` (not an empty DataFrame) when fewer than ``min_bars``
        rows are available, so callers can branch cleanly::

            df = client.require_bars("MGC", min_bars=100)
            if df is None:
                return  # not enough data for this analysis
        """
        df = self.get_bars(symbol, interval=interval, days_back=days_back)
        if df is None:
            logger.warning(
                "No bar data available from engine for %s (%s, %d days)",
                symbol,
                interval,
                days_back,
            )
            return None
        if len(df) < min_bars:
            logger.warning(
                "Insufficient bars for %s: got %d, need %d (%s, %d days)",
                symbol,
                len(df),
                min_bars,
                interval,
                days_back,
            )
            return None
        return df

    def get_bars_bulk(
        self,
        symbols: list[str],
        interval: str = "1m",
        days_back: int = 90,
        *,
        use_cache: bool = True,
    ) -> dict[str, pd.DataFrame]:
        """Fetch bars for multiple symbols in one request (uses ``/bars/bulk``).

        Falls back to sequential ``get_bars()`` calls if the bulk endpoint
        is not available or returns an unexpected response.

        Parameters
        ----------
        symbols:
            List of symbols to fetch.
        interval:
            Bar interval (same for all symbols).
        days_back:
            Days of history (same for all symbols).

        Returns
        -------
        dict[str, pd.DataFrame]
            Mapping of symbol → DataFrame.  Missing / failed symbols are
            omitted from the result (not set to None).
        """
        # Check cache first — only request symbols we don't have yet
        result: dict[str, pd.DataFrame] = {}
        symbols_needed: list[str] = []

        for sym in symbols:
            key = _cache_key("bars", sym, interval, days_back)
            cached = _cache_get(key) if use_cache else None
            if cached is not None:
                result[sym] = cached
            else:
                symbols_needed.append(sym)

        if not symbols_needed:
            return result

        # Try the bulk endpoint
        payload = self._post(
            "/bars/bulk",
            body={
                "symbols": symbols_needed,
                "interval": interval,
                "days_back": days_back,
                "auto_fill": True,
            },
        )

        if payload and isinstance(payload.get("results"), dict):
            for sym, sym_payload in payload["results"].items():
                if not sym_payload:
                    continue
                split = sym_payload.get("data") if isinstance(sym_payload, dict) else None
                if split is None:
                    continue
                df = _split_to_df(split)
                if df is not None and not df.empty:
                    result[sym] = df
                    if use_cache:
                        _cache_set(_cache_key("bars", sym, interval, days_back), df)
            return result

        # Bulk endpoint not available — fall back to sequential fetches
        logger.debug(
            "Bulk endpoint unavailable or returned unexpected payload; falling back to %d sequential fetches",
            len(symbols_needed),
        )
        for sym in symbols_needed:
            df = self.get_bars(sym, interval=interval, days_back=days_back, use_cache=use_cache)
            if df is not None:
                result[sym] = df

        return result

    def get_daily_bars(
        self,
        symbol: str,
        days_back: int = 365,
        *,
        use_cache: bool = True,
    ) -> pd.DataFrame | None:
        """Convenience wrapper: fetch daily (1d) bars.

        Daily bars are used by ``bias_analyzer``, ``scorer``, and
        ``volatility`` for ATR averages and trend classification.
        """
        return self.get_bars(symbol, interval="1d", days_back=days_back, use_cache=use_cache)

    def get_htf_bars(
        self,
        symbol: str,
        interval: str = "15m",
        days_back: int = 30,
        *,
        use_cache: bool = True,
    ) -> pd.DataFrame | None:
        """Convenience wrapper: fetch higher-timeframe bars.

        Used by ``mtf_analyzer``, ``confluence``, and ``breakout_filters``
        for multi-timeframe EMA / MACD alignment checks.
        """
        return self.get_bars(symbol, interval=interval, days_back=days_back, use_cache=use_cache)

    # ------------------------------------------------------------------
    # Snapshot / live price
    # ------------------------------------------------------------------

    def get_snapshot(
        self,
        symbol: str,
        *,
        use_cache: bool = True,
    ) -> dict[str, Any] | None:
        """Fetch the latest price snapshot for ``symbol``.

        Routes to ``GET /market_data/snapshot/{symbol}`` on the engine.
        The engine resolves Massive snapshots for CME futures and Kraken
        ticker snapshots for spot crypto, returning a normalised dict::

            {
                "symbol":      "MGC",
                "price":       2712.4,
                "bid":         2712.3,
                "ask":         2712.5,
                "volume":      12345,
                "change_pct":  0.45,
                "timestamp":   "2025-01-06T14:30:00+00:00"
            }

        Returns ``None`` if the symbol is unknown or the engine is down.
        """
        key = _cache_key("snapshot", symbol)
        if use_cache:
            cached = _cache_get(key)
            if cached is not None:
                return cached  # type: ignore[return-value]

        payload = self._get(
            f"/market_data/snapshot/{symbol}",
            timeout=self._snapshot_timeout,
        )

        if not payload:
            return None

        if use_cache:
            _cache_set(key, payload)

        return payload  # type: ignore[return-value]

    def get_snapshots_bulk(
        self,
        symbols: list[str],
        *,
        use_cache: bool = True,
    ) -> dict[str, dict[str, Any]]:
        """Fetch snapshots for multiple symbols.

        Returns a dict keyed by symbol; missing symbols are omitted.
        """
        result: dict[str, dict[str, Any]] = {}
        symbols_needed: list[str] = []

        for sym in symbols:
            key = _cache_key("snapshot", sym)
            cached = _cache_get(key) if use_cache else None
            if cached is not None:
                result[sym] = cached  # type: ignore[assignment]
            else:
                symbols_needed.append(sym)

        for sym in symbols_needed:
            snap = self.get_snapshot(sym, use_cache=use_cache)
            if snap:
                result[sym] = snap

        return result

    # ------------------------------------------------------------------
    # Symbol / asset discovery
    # ------------------------------------------------------------------

    def get_symbols(self, *, use_cache: bool = True) -> list[str]:
        """Return the list of short symbols enabled on the engine.

        Queries ``GET /bars/symbols`` (fast, no DB overhead) with fallback
        to ``GET /bars/assets`` (slower, includes bar counts).

        Returns a sorted list of short symbols like ``["6A", "6B", "MGC", ...]``
        that can be passed directly to ``get_bars()``.
        """
        key = _cache_key("symbols")
        if use_cache:
            cached = _cache_get(key)
            if cached is not None:
                return cached  # type: ignore[return-value]

        # Fast path: /bars/symbols (no DB query)
        payload = self._get("/bars/symbols", timeout=self._snapshot_timeout)
        if payload:
            syms = payload.get("symbols", [])
            if syms and isinstance(syms, list):
                result = sorted(s for s in syms if s)
                if use_cache:
                    _cache_set(key, result)
                return result

        # Fallback: /bars/assets (includes bar counts)
        payload = self._get("/bars/assets", timeout=self._snapshot_timeout)
        if payload:
            assets = payload.get("assets", [])
            if assets and isinstance(assets, list):
                tickers = [a.get("ticker", "") for a in assets if isinstance(a, dict)]
                # Strip "=F" to get short symbols; Kraken tickers pass through
                result = sorted(set(t.replace("=F", "") if not t.startswith("KRAKEN:") else t for t in tickers if t))
                if use_cache:
                    _cache_set(key, result)
                return result

        logger.debug("Engine /bars/symbols and /bars/assets both unavailable")
        return []

    def get_assets(self, *, use_cache: bool = True) -> list[dict[str, Any]]:
        """Return the full asset list from the engine, including name/ticker/symbol.

        Queries ``GET /bars/symbols`` (preferred — includes ``symbol`` field)
        or ``GET /bars/assets`` (fallback — includes bar counts).

        Each entry is a dict with at least ``{"name": str, "ticker": str}``.
        The ``/bars/symbols`` response also includes ``"symbol"`` (short name).
        """
        key = _cache_key("assets")
        if use_cache:
            cached = _cache_get(key)
            if cached is not None:
                return cached  # type: ignore[return-value]

        payload = self._get("/bars/symbols", timeout=self._snapshot_timeout)
        if payload:
            assets = payload.get("assets", [])
            if assets and isinstance(assets, list):
                if use_cache:
                    _cache_set(key, assets)
                return assets  # type: ignore[return-value]

        payload = self._get("/bars/assets", timeout=self._snapshot_timeout)
        if payload:
            assets = payload.get("assets", [])
            if assets and isinstance(assets, list):
                if use_cache:
                    _cache_set(key, assets)
                return assets  # type: ignore[return-value]

        return []

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    def get_stored_bars(
        self,
        symbol: str,
        interval: str = "1m",
        days_back: int = 365,
    ) -> pd.DataFrame | None:
        """Fetch OHLCV bars from the Postgres-backed store via ``/api/data/bars``.

        This is a convenience method that hits the data service's
        ``GET /api/data/bars`` endpoint directly.  Unlike :meth:`get_bars`
        (which routes through Redis → Postgres → external API), this path
        reads *only* from Postgres — no external API calls are made.

        This makes it ideal for **offline training**: once the
        :class:`DataSyncService` has populated the rolling 365-day window
        in Postgres, the trainer can generate datasets without any API keys.

        Parameters
        ----------
        symbol:
            Ticker symbol, e.g. ``"MGC=F"`` or ``"KRAKEN:XBTUSD"``.
        interval:
            Bar interval (default ``"1m"``).
        days_back:
            Number of calendar days to look back (default 365).

        Returns
        -------
        pd.DataFrame | None
            OHLCV DataFrame with a UTC DatetimeIndex and columns
            ``["Open", "High", "Low", "Close", "Volume"]``, or ``None``
            if the endpoint is unreachable or returned no data.
        """
        payload = self._get(
            "/api/data/bars",
            params={
                "symbol": symbol,
                "interval": interval,
                "days": days_back,
            },
        )

        if not payload:
            return None

        # Check for server-side errors embedded in the response
        if payload.get("error"):
            logger.debug(
                "get_stored_bars: server error for %s: %s",
                symbol,
                payload["error"],
            )
            return None

        bars = payload.get("bars", [])
        if not bars:
            logger.debug("get_stored_bars: 0 bars for %s/%s (%d days)", symbol, interval, days_back)
            return None

        try:
            df = pd.DataFrame(bars)

            # Parse the ISO timestamp index
            df.index = pd.to_datetime(df["t"], utc=True)
            df.index.name = None

            # Rename compact keys to standard OHLCV column names
            df = df.rename(
                columns={
                    "o": "Open",
                    "h": "High",
                    "l": "Low",
                    "c": "Close",
                    "v": "Volume",
                }
            )

            # Drop the raw timestamp column (now in the index)
            df = df.drop(columns=["t"], errors="ignore")

            # Keep only OHLCV columns and ensure numeric types
            for col in ("Open", "High", "Low", "Close", "Volume"):
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

            # Sort by time just in case
            df = df.sort_index()

            logger.debug(
                "get_stored_bars: %d bars for %s/%s (%d days back)",
                len(df),
                symbol,
                interval,
                days_back,
            )
            return df if not df.empty else None

        except Exception as exc:
            logger.debug("get_stored_bars: DataFrame construction failed for %s: %s", symbol, exc)
            return None

    def is_available(self, timeout: int = 3) -> bool:
        """Return True if the engine health endpoint responds within ``timeout`` s."""
        try:
            import requests as _requests  # type: ignore[import-untyped]

            resp = _requests.get(
                f"{self.base_url}/health",
                headers=self._headers(),
                timeout=timeout,
            )
            return resp.status_code == 200
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"EngineDataClient(base_url={self.base_url!r}, "
            f"bars_timeout={self._bars_timeout}s, "
            f"cache_ttl={self._cache_ttl}s)"
        )


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_default_client: EngineDataClient | None = None
_client_lock = threading.Lock()


def get_client(
    base_url: str | None = None,
    api_key: str | None = None,
) -> EngineDataClient:
    """Return the shared module-level ``EngineDataClient`` instance.

    The first call creates the singleton from environment variables.
    Subsequent calls return the same instance (thread-safe).

    Pass ``base_url`` / ``api_key`` to override the defaults — useful
    in tests::

        from lib.services.data.engine_data_client import get_client
        client = get_client(base_url="http://localhost:8000")
    """
    global _default_client
    with _client_lock:
        if _default_client is None or base_url is not None or api_key is not None:
            _default_client = EngineDataClient(
                base_url=base_url or _ENGINE_DATA_URL,
                api_key=api_key or _API_KEY,
            )
    return _default_client


def reset_client() -> None:
    """Reset the singleton (useful in tests to inject a fresh client)."""
    global _default_client
    with _client_lock:
        _default_client = None
    clear_cache()


# ---------------------------------------------------------------------------
# Convenience module-level functions (delegate to singleton)
# ---------------------------------------------------------------------------


def get_bars(
    symbol: str,
    interval: str = "1m",
    days_back: int = 90,
    *,
    auto_fill: bool = True,
) -> pd.DataFrame | None:
    """Module-level shorthand: ``get_client().get_bars(symbol, ...)``.

    Intended for analysis modules that prefer a functional import style::

        from lib.services.data.engine_data_client import get_bars

        df = get_bars("MGC", interval="1m", days_back=90)
    """
    return get_client().get_bars(symbol, interval=interval, days_back=days_back, auto_fill=auto_fill)


def get_daily_bars(symbol: str, days_back: int = 365) -> pd.DataFrame | None:
    """Module-level shorthand for daily bars."""
    return get_client().get_daily_bars(symbol, days_back=days_back)


def get_htf_bars(symbol: str, interval: str = "15m", days_back: int = 30) -> pd.DataFrame | None:
    """Module-level shorthand for higher-timeframe bars."""
    return get_client().get_htf_bars(symbol, interval=interval, days_back=days_back)


def get_snapshot(symbol: str) -> dict[str, Any] | None:
    """Module-level shorthand for latest price snapshot."""
    return get_client().get_snapshot(symbol)


def get_symbols() -> list[str]:
    """Module-level shorthand: list of enabled short symbols from the engine."""
    return get_client().get_symbols()


def get_bars_bulk(
    symbols: list[str],
    interval: str = "1m",
    days_back: int = 90,
) -> dict[str, pd.DataFrame]:
    """Module-level shorthand for bulk bar fetching."""
    return get_client().get_bars_bulk(symbols, interval=interval, days_back=days_back)


# ---------------------------------------------------------------------------
# DataFrameProvider protocol
# ---------------------------------------------------------------------------
# Analysis modules that previously accepted a ``cache.get_data`` callable
# can accept an ``EngineDataClient`` instance instead. The interface is
# intentionally minimal so tests can inject a simple mock.


@dataclass
class StaticBarProvider:
    """Provides pre-loaded DataFrames — useful for testing and backtesting.

    Pass a ``bars`` dict and this object will serve ``get_bars()`` calls
    from the dict without hitting the network::

        provider = StaticBarProvider({
            "MGC": df_mgc,
            "MES": df_mes,
        })
        result = compute_wave_analysis(provider.get_bars("MGC"))
    """

    bars: dict[str, pd.DataFrame] = field(default_factory=dict)
    snapshots: dict[str, dict[str, Any]] = field(default_factory=dict)

    def get_bars(
        self,
        symbol: str,
        interval: str = "1m",
        days_back: int = 90,
        **_kwargs: Any,
    ) -> pd.DataFrame | None:
        df = self.bars.get(symbol)
        if df is None:
            # Try stripping =F or KRAKEN: prefix variants
            for candidate in (symbol.replace("=F", ""), f"{symbol}=F"):
                df = self.bars.get(candidate)
                if df is not None:
                    break
        return df

    def get_daily_bars(self, symbol: str, days_back: int = 365) -> pd.DataFrame | None:
        return self.get_bars(symbol, interval="1d", days_back=days_back)

    def get_htf_bars(self, symbol: str, interval: str = "15m", days_back: int = 30) -> pd.DataFrame | None:
        return self.get_bars(symbol, interval=interval, days_back=days_back)

    def get_snapshot(self, symbol: str, **_kwargs: Any) -> dict[str, Any] | None:
        return self.snapshots.get(symbol)

    def get_symbols(self) -> list[str]:
        return sorted(self.bars.keys())

    def get_bars_bulk(
        self,
        symbols: list[str],
        interval: str = "1m",
        days_back: int = 90,
    ) -> dict[str, pd.DataFrame]:
        return {s: df for s in symbols if (df := self.get_bars(s)) is not None}

    def is_available(self, timeout: int = 3) -> bool:
        return True

    def __repr__(self) -> str:
        return f"StaticBarProvider(symbols={sorted(self.bars.keys())!r})"


# ---------------------------------------------------------------------------
# Type alias for callers that accept either a live client or a static provider
# ---------------------------------------------------------------------------

BarProvider = EngineDataClient | StaticBarProvider


__all__ = [
    # Client class
    "EngineDataClient",
    # Static / test provider
    "StaticBarProvider",
    # Type alias
    "BarProvider",
    # Singleton management
    "get_client",
    "reset_client",
    # Module-level convenience functions
    "get_bars",
    "get_daily_bars",
    "get_htf_bars",
    "get_snapshot",
    "get_symbols",
    "get_bars_bulk",
    # Cache management
    "clear_cache",
]
