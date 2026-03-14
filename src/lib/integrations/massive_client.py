"""
Massive.com Futures Data Provider
=================================
Primary market data source for the futures trading dashboard.
Massive (formerly Polygon.io) provides real-time and historical futures data
from CME/CBOT/NYMEX/COMEX exchanges.

Features:
  - REST: Historical aggregates (OHLCV), contract resolution, snapshots
  - WebSocket: Real-time minute bars, trades (for CVD), and quotes
  - Automatic front-month contract resolution via /contracts endpoint
  - Seamless fallback to yfinance when MASSIVE_API_KEY is not set
  - Thread-safe WebSocket feed manager for background streaming

Usage:
    from lib.massive_client import get_massive_provider, MassiveFeedManager

    provider = get_massive_provider()
    if provider.is_available:
        df = provider.get_aggs("ES=F", interval="5m", period="5d")
        snapshot = provider.get_snapshot("ES")
    else:
        # Falls back to yfinance automatically via cache.py

Environment:
    MASSIVE_API_KEY  — API key from https://massive.com/dashboard
"""

import contextlib
import logging
import os
import threading
import time
from datetime import datetime, timedelta
from typing import Any, Optional
from zoneinfo import ZoneInfo

import pandas as pd

_EST = ZoneInfo("America/New_York")
logger = logging.getLogger("massive_client")
logger.setLevel(logging.INFO)
if not logger.handlers:
    _h = logging.StreamHandler()
    import datetime as _dt

    _fmt = logging.Formatter("[MASSIVE] %(asctime)s  %(message)s", "%H:%M:%S")
    _fmt.converter = lambda *_: _dt.datetime.now(_EST).timetuple()
    _h.setFormatter(_fmt)
    logger.addHandler(_h)

# ---------------------------------------------------------------------------
# Yahoo ticker → Massive product code mapping
# ---------------------------------------------------------------------------
# Maps the Yahoo Finance tickers used in models.ASSETS to the Massive
# product codes needed for contract resolution and data fetching.
YAHOO_TO_MASSIVE_PRODUCT: dict[str, str] = {
    # Full-size contracts
    "ES=F": "ES",
    "NQ=F": "NQ",
    "GC=F": "GC",
    "SI=F": "SI",
    "HG=F": "HG",
    "CL=F": "CL",
    "YM=F": "YM",
    "RTY=F": "RTY",
    # Micro contracts (data_ticker points to full-size, but we map both)
    "MES=F": "MES",
    "MNQ=F": "MNQ",
    "MGC=F": "MGC",
    "SIL=F": "SIL",
    "MHG=F": "MHG",
    "MCL=F": "MCL",
    "MYM=F": "MYM",
    "M2K=F": "M2K",
    # FX futures (CME)
    "6E=F": "6E",  # Euro FX
    "6B=F": "6B",  # British Pound
    "6J=F": "6J",  # Japanese Yen
    "6A=F": "6A",  # Australian Dollar
    "6C=F": "6C",  # Canadian Dollar
    "6S=F": "6S",  # Swiss Franc
    # Micro FX futures (CME)
    "M6E=F": "M6E",  # Micro Euro FX
    "M6B=F": "M6B",  # Micro British Pound
    "M6J=F": "M6J",  # Micro Japanese Yen
    # Crypto futures (CME)
    "MBT=F": "MBT",  # Micro Bitcoin
    "BTC=F": "BTC",  # Bitcoin (full-size)
    "MET=F": "MET",  # Micro Ether
    "ETH=F": "ETH",  # Ether (full-size)
    # Energy futures (NYMEX)
    "NG=F": "NG",  # Natural Gas
    "MNG=F": "MNG",  # Micro Natural Gas
    # Interest rate futures (CBOT)
    "ZN=F": "ZN",  # 10-Year T-Note
    "ZB=F": "ZB",  # 30-Year T-Bond
    "ZF=F": "ZF",  # 5-Year T-Note
    "ZT=F": "ZT",  # 2-Year T-Note
    # Agricultural futures (CBOT)
    "ZC=F": "ZC",  # Corn
    "ZS=F": "ZS",  # Soybeans
    "ZW=F": "ZW",  # Wheat
    "ZL=F": "ZL",  # Soybean Oil
    "ZM=F": "ZM",  # Soybean Meal
}

# Reverse: product code → Yahoo ticker (for WebSocket → cache mapping)
MASSIVE_PRODUCT_TO_YAHOO: dict[str, str] = {v: k for k, v in YAHOO_TO_MASSIVE_PRODUCT.items()}

# Interval mapping: our internal intervals → Massive resolution strings
# The Massive futures aggs endpoint uses a 'resolution' query param
# Valid resolutions: "1sec", "1min", "1day"
# For intervals >1m we fetch 1min bars and resample client-side.
INTERVAL_TO_RESOLUTION: dict[str, str] = {
    "1s": "1sec",
    "1m": "1min",
    "5m": "1min",
    "15m": "1min",
    "30m": "1min",
    "1h": "1min",
    "60m": "1min",
    "1d": "1day",
    "1wk": "1day",
    "1mo": "1day",
}

# How many raw bars to request per output bar for resampling
# e.g., 5m bars = 5x 1-minute bars
INTERVAL_MULTIPLIER: dict[str, int] = {
    "1s": 1,
    "1m": 1,
    "5m": 5,
    "15m": 15,
    "30m": 30,
    "1h": 60,
    "60m": 60,
    "1d": 1,
    "1wk": 1,
    "1mo": 1,
}

# Period string → number of calendar days
PERIOD_TO_DAYS: dict[str, int] = {
    "1d": 1,
    "2d": 2,
    "5d": 5,
    "7d": 7,
    "10d": 10,
    "15d": 15,
    "1mo": 30,
    "3mo": 90,
    "6mo": 180,
    "1y": 365,
    "2y": 730,
    "5y": 1825,
}


# ---------------------------------------------------------------------------
# Resampling helper
# ---------------------------------------------------------------------------


def _dropna_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows where any OHLC value is missing (NaN or None)."""
    ohlc = [c for c in ("Open", "High", "Low", "Close") if c in df.columns]
    if ohlc and not df.empty:
        df = df.dropna(subset=ohlc)
    return df


def _resample_to_interval(df: pd.DataFrame, interval: str) -> pd.DataFrame:
    """Resample 1-minute or 1-day bars to the requested interval.

    Needed when interval doesn't map to a native API resolution
    (e.g., 5m, 15m, 30m, 1h from 1min; 1wk from 1day).
    """
    resample_map = {
        "5m": "5min",
        "15m": "15min",
        "30m": "30min",
        "1h": "1h",
        "60m": "1h",
        "1wk": "W",
        "1mo": "ME",
    }
    rule = resample_map.get(interval)
    if rule is None or df.empty:
        return df

    # Ensure index is DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        return df

    resampled = df.resample(rule).agg(
        {
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
            "Volume": "sum",
        }
    )
    assert isinstance(resampled, pd.DataFrame)
    # Drop incomplete bars (NaN open means no data in that window)
    resampled = _dropna_ohlc(resampled)
    return resampled


# ---------------------------------------------------------------------------
# MassiveDataProvider — REST client wrapper
# ---------------------------------------------------------------------------


class MassiveDataProvider:
    """Wraps the Massive REST client for futures data fetching.

    Handles:
      - API key detection and client initialization
      - Front-month contract resolution with caching
      - OHLCV aggregate fetching with interval/period mapping
      - Real-time snapshots for current prices
      - Graceful error handling (never crashes the app)
    """

    # Cache resolved front-month tickers for 1 hour
    CONTRACT_CACHE_TTL = 3600

    def __init__(self, api_key: str | None = None):
        self._api_key: str = api_key or os.getenv("MASSIVE_API_KEY", "") or ""
        self._client = None
        self._is_available = False
        self._contract_cache: dict[str, tuple[str, float]] = {}  # product → (ticker, timestamp)
        self._lock = threading.Lock()

        if self._api_key:
            try:
                from massive import RESTClient as _RESTClient

                self._client = _RESTClient(api_key=self._api_key)
                self._is_available = True
                logger.info("Massive REST client initialized successfully")
            except Exception as exc:
                logger.warning("Failed to initialize Massive client: %s", exc)
                self._is_available = False
        else:
            logger.info("MASSIVE_API_KEY not set — Massive data provider disabled, using yfinance fallback")

    @property
    def is_available(self) -> bool:
        """Whether the Massive API client is ready to use."""
        return self._is_available and self._client is not None

    @property
    def api_key(self) -> str:
        return self._api_key

    # ----- Contract Resolution -----

    def resolve_front_month(self, product_code: str) -> str | None:
        """Resolve a product code (e.g., 'ES') to its front-month contract ticker.

        Uses a robust 3-tier fallback strategy:
          1. Active contracts on today's date (strict — original approach)
          2. All contracts filtered to future expiration (most reliable in beta)
          3. Root symbol fallback (e.g., 'ES') — works for REST aggregates

        Returns the active contract ticker (e.g., 'ESZ5') or root symbol
        as last resort.  Uses a TTL cache to avoid hammering the contracts
        endpoint.
        """
        if not self.is_available or self._client is None:
            return None

        now = time.time()

        # Check cache
        with self._lock:
            cached = self._contract_cache.get(product_code)
            if cached and (now - cached[1]) < self.CONTRACT_CACHE_TTL:
                return cached[0]

        today_str = datetime.now(tz=_EST).strftime("%Y-%m-%d")
        client = self._client  # already checked not None above

        def _is_outright(c) -> bool:
            """Filter to outright futures only — exclude spreads/combos."""
            ticker = str(getattr(c, "ticker", "") or "")
            ctype = getattr(c, "type", "") or ""
            return bool(ticker) and ctype in ("future", "single", "") and "-" not in ticker and ":" not in ticker

        def _sort_by_expiry(c) -> str:
            """Sort key: nearest last_trade_date first."""
            ltd = getattr(c, "last_trade_date", None)
            if ltd is None:
                return "9999-12-31"
            return str(ltd)

        # --- Tier 1: Strict active contracts on today's date (original) ---
        try:
            with self._lock:
                contracts = list(
                    client.list_futures_contracts(
                        product_code=product_code,
                        active=True,
                        type="single",
                        date=today_str,
                        limit=10,
                        sort="date",
                    )
                )
            active = [c for c in contracts if getattr(c, "active", True) and _is_outright(c)]
            if active:
                active.sort(key=_sort_by_expiry)
                ticker = str(getattr(active[0], "ticker", ""))
                if ticker:
                    with self._lock:
                        self._contract_cache[product_code] = (ticker, now)
                    logger.info(
                        "Resolved %s → %s (tier 1: active today, expires: %s)",
                        product_code,
                        ticker,
                        getattr(active[0], "last_trade_date", "?"),
                    )
                    return ticker
        except Exception as exc:
            logger.debug("Tier 1 (active today) failed for %s: %s", product_code, exc)

        # --- Tier 2: All contracts, filter to future expiration ---
        try:
            with self._lock:
                contracts = list(
                    client.list_futures_contracts(
                        product_code=product_code,
                        type="single",
                        limit=20,
                        sort="date",
                    )
                )
            # Keep only outrights whose last_trade_date is in the future
            future_contracts = [
                c
                for c in contracts
                if _is_outright(c) and str(getattr(c, "last_trade_date", "0000-00-00")) >= today_str
            ]
            if future_contracts:
                future_contracts.sort(key=_sort_by_expiry)
                ticker = str(getattr(future_contracts[0], "ticker", ""))
                if ticker:
                    with self._lock:
                        self._contract_cache[product_code] = (ticker, now)
                    logger.info(
                        "Resolved %s → %s (tier 2: next active, expires: %s)",
                        product_code,
                        ticker,
                        getattr(future_contracts[0], "last_trade_date", "?"),
                    )
                    return ticker
        except Exception as exc:
            logger.debug("Tier 2 (future expiration) failed for %s: %s", product_code, exc)

        # --- Tier 3: Root symbol fallback ---
        # The root symbol (e.g., "ES", "GC") works reliably for REST
        # aggregate endpoints even when the contracts endpoint is flaky.
        # For WebSocket, the broad subscription (AM.*, T.*) will catch
        # whatever contract ticker Massive broadcasts.
        logger.warning(
            "Resolved %s → %s (tier 3: root symbol fallback — contracts endpoint returned no usable results)",
            product_code,
            product_code,
        )
        with self._lock:
            # Cache with a shorter TTL (5 min) so we retry sooner
            self._contract_cache[product_code] = (
                product_code,
                now - self.CONTRACT_CACHE_TTL + 300,
            )
        return product_code

    def resolve_from_yahoo(self, yahoo_ticker: str) -> str | None:
        """Resolve a Yahoo-style ticker (e.g., 'ES=F') to its Massive front-month ticker.

        Returns the Massive contract ticker or None if resolution fails.
        """
        product_code = YAHOO_TO_MASSIVE_PRODUCT.get(yahoo_ticker)
        if product_code is None:
            logger.debug("No Massive product code mapping for Yahoo ticker %s", yahoo_ticker)
            return None
        return self.resolve_front_month(product_code)

    def get_all_front_months(self, yahoo_tickers: list[str]) -> dict[str, str]:
        """Resolve multiple Yahoo tickers to their front-month Massive tickers.

        Returns a dict: yahoo_ticker → massive_ticker (only successful resolutions).
        """
        result = {}
        for yt in yahoo_tickers:
            massive_ticker = self.resolve_from_yahoo(yt)
            if massive_ticker:
                result[yt] = massive_ticker
        return result

    # ----- Aggregates (OHLCV) -----

    def get_aggs(
        self,
        yahoo_ticker: str,
        interval: str = "5m",
        period: str = "5d",
    ) -> pd.DataFrame:
        """Fetch OHLCV aggregates for a futures contract.

        Args:
            yahoo_ticker: Yahoo-style ticker (e.g., "ES=F", "GC=F")
            interval: Bar size ("1m", "5m", "15m", "1h", "1d")
            period: Lookback period ("1d", "5d", "10d", "1mo", etc.)

        Returns:
            DataFrame with columns [Open, High, Low, Close, Volume]
            and a tz-aware DatetimeIndex, or empty DataFrame on failure.
        """
        if not self.is_available or self._client is None:
            return pd.DataFrame()

        # Resolve to Massive contract ticker
        massive_ticker = self.resolve_from_yahoo(yahoo_ticker)
        if not massive_ticker:
            return pd.DataFrame()

        # Determine the base resolution and whether we need to resample
        resolution = INTERVAL_TO_RESOLUTION.get(interval, "1min")
        needs_resample = interval in ("5m", "15m", "30m", "1h", "60m", "1wk", "1mo")

        # Calculate date range from period
        days = PERIOD_TO_DAYS.get(period)
        if days is None:
            # Try parsing "Nd" format
            try:
                days = int(period.replace("d", ""))
            except (ValueError, AttributeError):
                days = 5  # default fallback

        now = datetime.now(tz=_EST)
        end_dt = now
        start_dt = now - timedelta(days=days)

        # Format timestamps for the API
        # The API expects ISO 8601 or nanosecond timestamps
        start_str = start_dt.strftime("%Y-%m-%dT%H:%M:%S-05:00")
        end_str = end_dt.strftime("%Y-%m-%dT%H:%M:%S-05:00")

        try:
            client = self._client  # already checked not None above
            with self._lock:
                aggs = list(
                    client.list_futures_aggregates(
                        ticker=massive_ticker,
                        resolution=resolution,
                        window_start_gte=start_str,
                        window_start_lte=end_str,
                        limit=50000,
                        sort="asc",
                    )
                )

            if not aggs:
                logger.warning(
                    "No aggregates returned for %s (%s) %s/%s",
                    yahoo_ticker,
                    massive_ticker,
                    interval,
                    period,
                )
                return pd.DataFrame()

            # Convert to DataFrame
            rows = []
            for agg in aggs:
                row = {
                    "Open": getattr(agg, "open", None),
                    "High": getattr(agg, "high", None),
                    "Low": getattr(agg, "low", None),
                    "Close": getattr(agg, "close", None),
                    "Volume": getattr(agg, "volume", 0) or 0,
                    "timestamp": getattr(agg, "window_start", None),
                }
                rows.append(row)

            df = pd.DataFrame(rows)

            # Parse timestamps
            if not df.empty and "timestamp" in df.columns:
                df = _parse_timestamp_index(df)

            # Drop rows with missing OHLC
            df = _dropna_ohlc(df)

            # Resample if needed (e.g., 1m → 5m)
            if needs_resample and not df.empty:
                df = _resample_to_interval(df, interval)

            if not df.empty:
                logger.info(
                    "Fetched %d bars for %s (%s) %s/%s via Massive",
                    len(df),
                    yahoo_ticker,
                    massive_ticker,
                    interval,
                    period,
                )

            return df

        except Exception as exc:
            logger.error(
                "Failed to fetch aggs for %s (%s): %s",
                yahoo_ticker,
                massive_ticker,
                exc,
            )
            return pd.DataFrame()

    def get_daily(
        self,
        yahoo_ticker: str,
        period: str = "10d",
    ) -> pd.DataFrame:
        """Fetch daily OHLCV bars.

        Convenience wrapper around get_aggs() with interval='1d'.
        """
        return self.get_aggs(yahoo_ticker, interval="1d", period=period)

    # ----- Snapshots -----

    def get_snapshot(
        self,
        yahoo_ticker: str | None = None,
        product_code: str | None = None,
    ) -> dict | None:
        """Get a real-time snapshot for a futures contract.

        Returns a dict with keys:
            ticker, product_code, last_price, last_size, bid, ask,
            session_open, session_high, session_low, session_close,
            session_volume, change, change_percent, settlement_price

        Or None on failure.
        """
        if not self.is_available or self._client is None:
            return None

        # Determine product code
        if product_code is None and yahoo_ticker:
            product_code = YAHOO_TO_MASSIVE_PRODUCT.get(yahoo_ticker)

        if not product_code:
            return None

        try:
            client = self._client  # already checked not None above
            with self._lock:
                snapshots = list(
                    client.get_futures_snapshot(
                        product_code=product_code,
                        limit=5,
                    )
                )

            if not snapshots:
                return None

            # Use the first snapshot (front-month)
            snap = snapshots[0]

            result: dict = {
                "ticker": getattr(snap, "ticker", None),
                "product_code": getattr(snap, "product_code", product_code),
            }

            # Last trade
            lt = getattr(snap, "last_trade", None)
            if lt:
                result["last_price"] = getattr(lt, "price", None)
                result["last_size"] = getattr(lt, "size", None)

            # Quote
            lq = getattr(snap, "last_quote", None)
            if lq:
                result["bid"] = getattr(lq, "bid", None)
                result["ask"] = getattr(lq, "ask", None)
                result["bid_size"] = getattr(lq, "bid_size", None)
                result["ask_size"] = getattr(lq, "ask_size", None)

            # Session
            sess = getattr(snap, "session", None)
            if sess:
                result["session_open"] = getattr(sess, "open", None)
                result["session_high"] = getattr(sess, "high", None)
                result["session_low"] = getattr(sess, "low", None)
                result["session_close"] = getattr(sess, "close", None)
                result["session_volume"] = getattr(sess, "volume", None)
                result["change"] = getattr(sess, "change", None)
                result["change_percent"] = getattr(sess, "change_percent", None)
                result["settlement_price"] = getattr(sess, "settlement_price", None)
                result["previous_settlement"] = getattr(sess, "previous_settlement", None)

            # Last minute bar
            lm = getattr(snap, "last_minute", None)
            if lm:
                result["minute_open"] = getattr(lm, "open", None)
                result["minute_high"] = getattr(lm, "high", None)
                result["minute_low"] = getattr(lm, "low", None)
                result["minute_close"] = getattr(lm, "close", None)
                result["minute_volume"] = getattr(lm, "volume", None)

            # Details
            det = getattr(snap, "details", None)
            if det:
                result["open_interest"] = getattr(det, "open_interest", None)

            return result

        except Exception as exc:
            logger.error("Failed to get snapshot for %s: %s", product_code, exc)
            return None

    def get_all_snapshots(self, yahoo_tickers: list[str]) -> dict[str, dict]:
        """Get snapshots for multiple tickers.

        Returns dict: yahoo_ticker → snapshot_dict (only successful fetches).
        """
        results: dict[str, dict] = {}
        # Batch by product code to reduce API calls
        product_codes = set()
        ticker_to_product = {}
        for yt in yahoo_tickers:
            pc = YAHOO_TO_MASSIVE_PRODUCT.get(yt)
            if pc:
                product_codes.add(pc)
                ticker_to_product[yt] = pc

        if not self.is_available or not product_codes or self._client is None:
            return results

        # Fetch all at once using product_code_any_of
        try:
            client = self._client  # already checked not None above
            codes_str = ",".join(sorted(product_codes))
            with self._lock:
                snapshots = list(
                    client.get_futures_snapshot(
                        product_code_any_of=codes_str,
                        limit=50,
                    )
                )

            # Index by product code
            snap_by_code: dict[str, dict] = {}
            for snap in snapshots:
                pc = getattr(snap, "product_code", None)
                if pc and pc not in snap_by_code:
                    # Convert snapshot to dict (reuse get_snapshot logic)
                    snap_dict = self._snapshot_to_dict(snap, pc)
                    snap_by_code[pc] = snap_dict

            # Map back to yahoo tickers
            for yt, pc in ticker_to_product.items():
                if pc in snap_by_code:
                    results[yt] = snap_by_code[pc]

        except Exception as exc:
            logger.error("Failed to get batch snapshots: %s", exc)

        return results

    def _snapshot_to_dict(self, snap, product_code: str) -> dict:
        """Convert a FuturesSnapshot object to a plain dict."""
        result: dict = {
            "ticker": getattr(snap, "ticker", None),
            "product_code": product_code,
        }

        lt = getattr(snap, "last_trade", None)
        if lt:
            result["last_price"] = getattr(lt, "price", None)
            result["last_size"] = getattr(lt, "size", None)

        lq = getattr(snap, "last_quote", None)
        if lq:
            result["bid"] = getattr(lq, "bid", None)
            result["ask"] = getattr(lq, "ask", None)

        sess = getattr(snap, "session", None)
        if sess:
            result["session_open"] = getattr(sess, "open", None)
            result["session_high"] = getattr(sess, "high", None)
            result["session_low"] = getattr(sess, "low", None)
            result["session_close"] = getattr(sess, "close", None)
            result["session_volume"] = getattr(sess, "volume", None)
            result["change"] = getattr(sess, "change", None)
            result["change_percent"] = getattr(sess, "change_percent", None)
            result["settlement_price"] = getattr(sess, "settlement_price", None)

        return result

    # ----- Trades (for CVD) -----

    def get_recent_trades(
        self,
        yahoo_ticker: str,
        minutes_back: int = 5,
        limit: int = 5000,
    ) -> pd.DataFrame:
        """Fetch recent trades for accurate CVD calculation.

        Returns DataFrame with columns: price, size, timestamp
        """
        if not self.is_available or self._client is None:
            return pd.DataFrame()

        massive_ticker = self.resolve_from_yahoo(yahoo_ticker)
        if not massive_ticker:
            return pd.DataFrame()

        now = datetime.now(tz=_EST)
        start = now - timedelta(minutes=minutes_back)
        start_str = start.strftime("%Y-%m-%dT%H:%M:%S-05:00")

        try:
            client = self._client  # already checked not None above
            with self._lock:
                trades = list(
                    client.list_futures_trades(
                        ticker=massive_ticker,
                        timestamp_gte=start_str,
                        limit=limit,
                        sort="asc",
                    )
                )

            if not trades:
                return pd.DataFrame()

            rows = []
            for t in trades:
                rows.append(
                    {
                        "price": getattr(t, "price", None),
                        "size": getattr(t, "size", None),
                        "timestamp": getattr(t, "timestamp", None),
                    }
                )

            df = pd.DataFrame(rows)
            df = df.dropna(subset=["price", "size"])
            return df

        except Exception as exc:
            logger.error("Failed to fetch trades for %s: %s", yahoo_ticker, exc)
            return pd.DataFrame()

    # ----- Contract info -----

    def get_active_contracts(self, product_code: str, limit: int = 5) -> list[dict]:
        """List active contracts for a product code.

        Returns a list of dicts with contract details.
        """
        if not self.is_available or self._client is None:
            return []

        try:
            client = self._client  # already checked not None above
            today_str = datetime.now(tz=_EST).strftime("%Y-%m-%d")
            with self._lock:
                contracts = list(
                    client.list_futures_contracts(
                        product_code=product_code,
                        active=True,
                        date=today_str,
                        limit=limit,
                        sort="date",
                    )
                )

            result = []
            for c in contracts:
                result.append(
                    {
                        "ticker": getattr(c, "ticker", None),
                        "name": getattr(c, "name", None),
                        "product_code": getattr(c, "product_code", product_code),
                        "active": getattr(c, "active", True),
                        "first_trade_date": getattr(c, "first_trade_date", None),
                        "last_trade_date": getattr(c, "last_trade_date", None),
                        "settlement_date": getattr(c, "settlement_date", None),
                        "days_to_maturity": getattr(c, "days_to_maturity", None),
                        "trading_venue": getattr(c, "trading_venue", None),
                        "type": getattr(c, "type", None),
                    }
                )
            return result

        except Exception as exc:
            logger.error("Failed to list contracts for %s: %s", product_code, exc)
            return []

    def invalidate_contract_cache(self, product_code: str | None = None) -> None:
        """Clear the contract resolution cache.

        If product_code is None, clears the entire cache.
        """
        with self._lock:
            if product_code:
                self._contract_cache.pop(product_code, None)
            else:
                self._contract_cache.clear()
        logger.info("Contract cache cleared: %s", product_code or "all")

    # ----- Contract Overview -----

    def get_contract(self, ticker: str) -> dict | None:
        """Retrieve detailed specifications for a single futures contract by ticker.

        Maps to: GET /futures/vX/contracts/{ticker}

        Returns a dict with contract details like tick size, trading dates,
        order quantity, etc., or None on failure.
        """
        if not self.is_available or self._client is None:
            return None

        try:
            client = self._client
            with self._lock:
                contracts = list(
                    client.list_futures_contracts(
                        ticker=ticker,
                        limit=1,
                    )
                )

            if not contracts:
                return None

            c = contracts[0]
            return {
                "ticker": getattr(c, "ticker", None),
                "name": getattr(c, "name", None),
                "product_code": getattr(c, "product_code", None),
                "active": getattr(c, "active", None),
                "type": getattr(c, "type", None),
                "first_trade_date": str(getattr(c, "first_trade_date", "") or ""),
                "last_trade_date": str(getattr(c, "last_trade_date", "") or ""),
                "settlement_date": str(getattr(c, "settlement_date", "") or ""),
                "days_to_maturity": getattr(c, "days_to_maturity", None),
                "trading_venue": getattr(c, "trading_venue", None),
                "tick_size": getattr(c, "tick_size", None),
                "tick_value": getattr(c, "tick_value", None),
                "unit_of_measure": getattr(c, "unit_of_measure", None),
                "unit_of_measure_qty": getattr(c, "unit_of_measure_qty", None),
                "order_quantity_min": getattr(c, "order_quantity_min", None),
                "order_quantity_max": getattr(c, "order_quantity_max", None),
                "order_quantity_increment": getattr(c, "order_quantity_increment", None),
            }

        except Exception as exc:
            logger.error("Failed to get contract for %s: %s", ticker, exc)
            return None

    # ----- Products -----

    def get_products(
        self,
        name: str | None = None,
        product_code: str | None = None,
        trading_venue: str | None = None,
        sector: str | None = None,
        sub_sector: str | None = None,
        asset_class: str | None = None,
        asset_sub_class: str | None = None,
        limit: int = 100,
    ) -> list[dict]:
        """Filter through all available futures product specifications.

        Maps to: GET /futures/vX/products

        Returns a list of dicts with product details including asset class,
        venue, settlement method, etc.
        """
        if not self.is_available or self._client is None:
            return []

        try:
            client = self._client
            kwargs: dict = {"limit": limit, "sort": "name"}
            if name:
                kwargs["name"] = name
            if product_code:
                kwargs["product_code"] = product_code
            if trading_venue:
                kwargs["trading_venue"] = trading_venue
            if sector:
                kwargs["sector"] = sector
            if sub_sector:
                kwargs["sub_sector"] = sub_sector
            if asset_class:
                kwargs["asset_class"] = asset_class
            if asset_sub_class:
                kwargs["asset_sub_class"] = asset_sub_class

            with self._lock:
                products = list(client.list_futures_products(**kwargs))

            result = []
            for p in products:
                result.append(
                    {
                        "product_code": getattr(p, "product_code", None),
                        "name": getattr(p, "name", None),
                        "asset_class": getattr(p, "asset_class", None),
                        "asset_sub_class": getattr(p, "asset_sub_class", None),
                        "sector": getattr(p, "sector", None),
                        "sub_sector": getattr(p, "sub_sector", None),
                        "trading_venue": getattr(p, "trading_venue", None),
                        "type": getattr(p, "type", None),
                        "settlement_method": getattr(p, "settlement_method", None),
                        "currency": getattr(p, "currency", None),
                        "unit_of_measure": getattr(p, "unit_of_measure", None),
                        "tick_size": getattr(p, "tick_size", None),
                        "tick_value": getattr(p, "tick_value", None),
                    }
                )
            return result

        except Exception as exc:
            logger.error("Failed to list products: %s", exc)
            return []

    def get_product(self, product_code: str) -> dict | None:
        """Retrieve detailed information about a specific futures product.

        Maps to: GET /futures/vX/products/{product_code}

        Returns a dict with asset class, venue, name, settlement details, etc.,
        or None on failure.
        """
        if not self.is_available or self._client is None:
            return None

        try:
            results = self.get_products(product_code=product_code, limit=1)
            return results[0] if results else None
        except Exception as exc:
            logger.error("Failed to get product %s: %s", product_code, exc)
            return None

    # ----- Schedules -----

    def get_schedules(
        self,
        product_code: str | None = None,
        trading_date: str | None = None,
        trading_venue: str | None = None,
        limit: int = 100,
    ) -> list[dict]:
        """Filter through trading schedules for futures contracts.

        Maps to: GET /futures/vX/schedules  (all)
                 GET /futures/vX/products/{product_code}/schedules  (by product)

        Args:
            product_code: Filter by product (e.g. "ES", "GC"). Optional.
            trading_date: Session end date in YYYY-MM-DD format. Defaults to today.
            trading_venue: Filter by exchange (e.g. "CME", "NYMEX"). Optional.
            limit: Maximum number of results.

        Returns a list of schedule dicts.
        """
        if not self.is_available or self._client is None:
            return []

        if trading_date is None:
            trading_date = datetime.now(tz=_EST).strftime("%Y-%m-%d")

        try:
            client = self._client
            kwargs: dict = {
                "session_end_date": trading_date,
                "limit": limit,
            }
            if product_code:
                kwargs["product_code"] = product_code
            if trading_venue:
                kwargs["trading_venue"] = trading_venue

            with self._lock:
                schedules = list(client.list_futures_schedules(**kwargs))

            result = []
            for s in schedules:
                entry: dict = {
                    "product_code": getattr(s, "product_code", None),
                    "trading_venue": getattr(s, "trading_venue", None),
                    "session_end_date": str(getattr(s, "session_end_date", "") or ""),
                }
                # Session windows (pre-open, open, close, etc.)
                sessions = getattr(s, "sessions", None)
                if sessions:
                    entry["sessions"] = []
                    if isinstance(sessions, (list, tuple)):
                        for sess in sessions:
                            sess_dict: dict = {}
                            for attr in (
                                "type",
                                "start",
                                "end",
                                "status",
                                "name",
                            ):
                                val = getattr(sess, attr, None)
                                if val is not None:
                                    sess_dict[attr] = str(val)
                            entry["sessions"].append(sess_dict)
                    else:
                        entry["sessions"] = str(sessions)
                else:
                    entry["sessions"] = []

                result.append(entry)
            return result

        except Exception as exc:
            logger.error("Failed to list schedules: %s", exc)
            return []

    # ----- Quotes (top-of-book bid/ask) -----

    def get_quotes(
        self,
        yahoo_ticker: str,
        minutes_back: int = 5,
        limit: int = 5000,
    ) -> pd.DataFrame:
        """Get top-of-book bid and ask prices for a futures contract.

        Maps to: GET /futures/vX/quotes/{ticker}

        Returns DataFrame with columns: bid, bid_size, ask, ask_size, timestamp
        """
        if not self.is_available or self._client is None:
            return pd.DataFrame()

        massive_ticker = self.resolve_from_yahoo(yahoo_ticker)
        if not massive_ticker:
            return pd.DataFrame()

        now = datetime.now(tz=_EST)
        start = now - timedelta(minutes=minutes_back)
        start_str = start.strftime("%Y-%m-%dT%H:%M:%S-05:00")

        try:
            client = self._client
            with self._lock:
                quotes = list(
                    client.list_futures_quotes(
                        ticker=massive_ticker,
                        timestamp_gte=start_str,
                        limit=limit,
                        sort="asc",
                    )
                )

            if not quotes:
                return pd.DataFrame()

            rows = []
            for q in quotes:
                rows.append(
                    {
                        "bid": getattr(q, "bid", None),
                        "bid_size": getattr(q, "bid_size", None),
                        "ask": getattr(q, "ask", None),
                        "ask_size": getattr(q, "ask_size", None),
                        "timestamp": getattr(q, "timestamp", None),
                    }
                )

            df = pd.DataFrame(rows)
            df = df.dropna(subset=["bid", "ask"])

            # Parse timestamps into DatetimeIndex
            if not df.empty and "timestamp" in df.columns:
                df = _parse_timestamp_index(df)

            if not df.empty:
                logger.info(
                    "Fetched %d quotes for %s (%s) via Massive",
                    len(df),
                    yahoo_ticker,
                    massive_ticker,
                )

            return df

        except Exception as exc:
            logger.error("Failed to fetch quotes for %s: %s", yahoo_ticker, exc)
            return pd.DataFrame()

    # ----- Market Status -----

    def get_market_statuses(
        self,
        product_code: str | None = None,
    ) -> list[dict]:
        """Retrieve the current market status for futures products/exchanges.

        Maps to: GET /futures/vX/market_status

        Args:
            product_code: Optional filter by product (e.g. "ES"). If None,
                          returns status for all products.

        Returns a list of dicts with market status information.
        """
        if not self.is_available or self._client is None:
            return []

        try:
            client = self._client
            kwargs: dict = {"limit": 100}
            if product_code:
                kwargs["product_code"] = product_code

            with self._lock:
                statuses = list(client.list_futures_market_statuses(**kwargs))

            result = []
            for ms in statuses:
                result.append(
                    {
                        "product_code": getattr(ms, "product_code", None),
                        "market": getattr(ms, "market", None),
                        "status": getattr(ms, "status", None),
                        "trading_venue": getattr(ms, "trading_venue", None),
                        "session_start": str(getattr(ms, "session_start", "") or ""),
                        "session_end": str(getattr(ms, "session_end", "") or ""),
                        "early_hours": getattr(ms, "early_hours", None),
                    }
                )
            return result

        except Exception as exc:
            logger.error("Failed to get market statuses: %s", exc)
            return []

    # ----- Exchanges -----

    def get_exchanges(self) -> list[dict]:
        """Retrieve a list of supported futures exchanges.

        Maps to: GET /futures/vX/exchanges

        Returns a list of dicts with exchange codes, names, and details.
        """
        if not self.is_available or self._client is None:
            return []

        try:
            client = self._client
            with self._lock:
                exchanges = list(client.list_futures_exchanges(limit=100))

            result = []
            for ex in exchanges:
                result.append(
                    {
                        "code": getattr(ex, "code", None),
                        "name": getattr(ex, "name", None),
                        "mic": getattr(ex, "mic", None),
                        "operating_mic": getattr(ex, "operating_mic", None),
                        "asset_class": getattr(ex, "asset_class", None),
                        "country": getattr(ex, "country", None),
                        "url": getattr(ex, "url", None),
                    }
                )
            return result

        except Exception as exc:
            logger.error("Failed to list exchanges: %s", exc)
            return []


# ---------------------------------------------------------------------------
# Timestamp parsing helper
# ---------------------------------------------------------------------------


def _parse_timestamp_index(df: pd.DataFrame) -> pd.DataFrame:
    """Parse the 'timestamp' column into a proper DatetimeIndex.

    Handles multiple timestamp formats:
      - Nanosecond epoch (int)
      - Millisecond epoch (int)
      - ISO 8601 string
    """
    if "timestamp" not in df.columns:
        return df

    ts_col = df["timestamp"]
    sample = ts_col.iloc[0] if len(ts_col) > 0 else None

    if sample is None:
        return df

    try:
        if isinstance(sample, (int, float)):
            # Determine if nanoseconds, microseconds, or milliseconds
            val = int(sample)
            if val > 1e18:
                # Nanoseconds
                dt_index = pd.to_datetime(ts_col, unit="ns", utc=True)
            elif val > 1e15:
                # Microseconds
                dt_index = pd.to_datetime(ts_col, unit="us", utc=True)
            elif val > 1e12:
                # Milliseconds
                dt_index = pd.to_datetime(ts_col, unit="ms", utc=True)
            else:
                # Seconds
                dt_index = pd.to_datetime(ts_col, unit="s", utc=True)
        elif isinstance(sample, str):
            dt_index = pd.to_datetime(ts_col, utc=True)
        else:
            dt_index = pd.to_datetime(ts_col, utc=True)

        # Convert to Eastern time and assign
        df.index = pd.DatetimeIndex(dt_index).tz_convert(_EST)
    except Exception as exc:
        logger.warning("Failed to parse timestamps: %s — using raw index", exc)

    # Drop the timestamp column now that it's the index
    df = df.drop(columns=["timestamp"], errors="ignore")
    return df


# ---------------------------------------------------------------------------
# MassiveFeedManager — WebSocket live data streaming
# ---------------------------------------------------------------------------

# WebSocket endpoint for futures
_WS_FUTURES_URI = "wss://socket.massive.com/futures"


class MassiveFeedManager:
    """Manages a WebSocket connection for real-time futures data.

    Streams minute aggregates, trades, and quotes from Massive's WebSocket
    API and pushes them into the cache layer for instant dashboard updates.

    Uses the raw ``websockets`` protocol against ``wss://socket.massive.com/futures``
    which is the most reliable path for futures data.  The Massive Python SDK's
    ``WebSocketClient`` is stock-oriented; for futures we authenticate and
    subscribe manually over the raw JSON wire format.

    Wire protocol (after TLS connect):
      1. Send  ``{"action":"auth","params":"<MASSIVE_API_KEY>"}``
      2. Recv  ``[{"ev":"status","status":"auth_success",...}]``
      3. Send  ``{"action":"subscribe","params":"AM.MESZ5,T.MESZ5,..."}``
      4. Recv  stream of JSON arrays, each element having ``ev`` + ``sym`` + data keys.

    Channel prefixes (per Massive/Polygon futures docs):
      | Prefix | Description               | Message keys (short)           |
      |--------|---------------------------|--------------------------------|
      | AM     | Per-minute aggregates     | o, h, l, c, v, s, e, n, vw    |
      | A      | Per-second aggregates     | o, h, l, c, v, s, e, n, vw    |
      | T      | Trades                    | p, s (size), t, q              |
      | Q      | Quotes                    | bp, bs, ap, as, t              |

    All timestamps are **millisecond** Unix epoch in Central Time.

    Usage:
        feed = MassiveFeedManager(api_key="...", yahoo_tickers=["ES=F", "NQ=F"])
        feed.start()          # connects + subscribes in background thread
        feed.is_connected     # check status
        feed.latest_bars      # dict of latest bars by ticker
        feed.stop()           # clean shutdown
    """

    # Subscription channel prefixes — these are the actual Massive/Polygon
    # futures WebSocket channel names (NOT the stock-focused FMA/FA/FT/FQ).
    PREFIX_MINUTE_AGG = "AM"
    PREFIX_SECOND_AGG = "A"
    PREFIX_TRADE = "T"
    PREFIX_QUOTE = "Q"

    def __init__(
        self,
        api_key: str | None = None,
        yahoo_tickers: list[str] | None = None,
        provider: Optional["MassiveDataProvider"] = None,
        subscribe_trades: bool = True,
        subscribe_quotes: bool = False,
        subscribe_second_aggs: bool = False,
        use_broad_subscriptions: bool = True,
    ):
        self._api_key: str = api_key or os.getenv("MASSIVE_API_KEY", "") or ""
        self._provider = provider
        self._yahoo_tickers = yahoo_tickers or []
        self._subscribe_trades = subscribe_trades
        self._subscribe_quotes = subscribe_quotes
        self._subscribe_second_aggs = subscribe_second_aggs
        self._use_broad_subscriptions = use_broad_subscriptions

        self._ws: Any = None  # raw websockets connection or SDK client
        self._thread: threading.Thread | None = None
        self._running = False
        self._connected = False
        self._lock = threading.Lock()

        # Pending dynamic subscribe/unsubscribe requests (consumed by the
        # async loop running in the background thread).
        self._pending_subscribe: list[str] = []
        self._pending_unsubscribe: list[str] = []

        # Latest data storage (thread-safe reads via _lock)
        self._latest_bars: dict[str, dict] = {}  # massive_ticker → bar dict
        self._latest_trades: dict[str, dict] = {}  # massive_ticker → trade dict
        self._latest_quotes: dict[str, dict] = {}  # massive_ticker → quote dict
        self._trade_buffer: dict[str, list] = {}  # massive_ticker → [trade, ...]

        # Resolved ticker mapping
        self._massive_to_yahoo: dict[str, str] = {}  # massive_ticker → yahoo_ticker
        self._yahoo_to_massive: dict[str, str] = {}  # yahoo_ticker → massive_ticker

        # Callbacks
        self._on_bar_callbacks: list = []
        self._on_trade_callbacks: list = []

        # Stats
        self.msg_count = 0
        self.bar_count = 0
        self.trade_count = 0
        self.errors: list[str] = []
        self.started_at: float | None = None

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def latest_bars(self) -> dict[str, dict]:
        """Thread-safe copy of latest bars by massive ticker."""
        with self._lock:
            return dict(self._latest_bars)

    @property
    def latest_trades(self) -> dict[str, dict]:
        """Thread-safe copy of latest trades by massive ticker."""
        with self._lock:
            return dict(self._latest_trades)

    def on_bar(self, callback) -> None:
        """Register a callback for new bar events.

        Callback signature: callback(yahoo_ticker: str, bar: dict)
        bar keys: open, high, low, close, volume, start_timestamp, end_timestamp
        """
        self._on_bar_callbacks.append(callback)

    def on_trade(self, callback) -> None:
        """Register a callback for new trade events.

        Callback signature: callback(yahoo_ticker: str, trade: dict)
        trade keys: price, size, timestamp
        """
        self._on_trade_callbacks.append(callback)

    def start(self) -> bool:
        """Start the WebSocket connection in a background thread.

        Returns True if started successfully, False otherwise.
        """
        if self._running:
            logger.info("Feed manager already running")
            return True

        if not self._api_key:
            logger.warning("Cannot start feed: no MASSIVE_API_KEY")
            return False

        # Resolve tickers before connecting
        if not self._resolve_tickers():
            logger.warning("Cannot start feed: no tickers resolved")
            return False

        self._running = True
        self._thread = threading.Thread(
            target=self._run_loop,
            daemon=True,
            name="massive-ws-feed",
        )
        self._thread.start()
        self.started_at = time.time()
        logger.info("Feed manager started for %d tickers", len(self._yahoo_to_massive))
        return True

    async def stop(self) -> None:
        """Stop the WebSocket connection and background thread."""
        self._running = False
        # Signal the asyncio loop to exit.  The ws object may be a raw
        # websockets connection (with an async close) so we guard broadly.
        ws = self._ws
        if ws is not None:
            with contextlib.suppress(Exception):
                await ws.close()
        if self._thread is not None:
            self._thread.join(timeout=5)
        self._connected = False
        logger.info(
            "Feed manager stopped (msgs=%d, bars=%d, trades=%d)",
            self.msg_count,
            self.bar_count,
            self.trade_count,
        )

    def add_tickers(self, yahoo_tickers: list[str]) -> None:
        """Dynamically add new tickers to the subscription."""
        new_subs: list[str] = []
        for yt in yahoo_tickers:
            if yt not in self._yahoo_tickers:
                self._yahoo_tickers.append(yt)
            # Re-resolve
            if yt not in self._yahoo_to_massive and self._provider and self._provider.is_available:
                mt = self._provider.resolve_from_yahoo(yt)
                if mt:
                    self._yahoo_to_massive[yt] = mt
                    self._massive_to_yahoo[mt] = yt
                    new_subs.append(f"{self.PREFIX_MINUTE_AGG}.{mt}")
                    if self._subscribe_trades:
                        new_subs.append(f"{self.PREFIX_TRADE}.{mt}")
                    if self._subscribe_quotes:
                        new_subs.append(f"{self.PREFIX_QUOTE}.{mt}")

        if new_subs and self._running:
            with self._lock:
                self._pending_subscribe.extend(new_subs)
            logger.info("Queued dynamic subscribe: %s", new_subs)

    def remove_tickers(self, yahoo_tickers: list[str]) -> None:
        """Dynamically remove tickers from the subscription."""
        unsub_channels: list[str] = []
        for yt in yahoo_tickers:
            if yt in self._yahoo_tickers:
                self._yahoo_tickers.remove(yt)
            mt = self._yahoo_to_massive.pop(yt, None)
            if mt:
                self._massive_to_yahoo.pop(mt, None)
                unsub_channels.append(f"{self.PREFIX_MINUTE_AGG}.{mt}")
                if self._subscribe_trades:
                    unsub_channels.append(f"{self.PREFIX_TRADE}.{mt}")
                if self._subscribe_quotes:
                    unsub_channels.append(f"{self.PREFIX_QUOTE}.{mt}")
                if self._subscribe_second_aggs:
                    unsub_channels.append(f"{self.PREFIX_SECOND_AGG}.{mt}")

        if unsub_channels and self._running:
            with self._lock:
                self._pending_unsubscribe.extend(unsub_channels)
            logger.info("Queued dynamic unsubscribe: %s", unsub_channels)

    def upgrade_to_second_aggs(self) -> None:
        """Switch from minute to per-second aggregates (for active trading)."""
        self._subscribe_second_aggs = True
        new_subs = [f"{self.PREFIX_SECOND_AGG}.{mt}" for mt in self._massive_to_yahoo]
        if new_subs and self._running:
            with self._lock:
                self._pending_subscribe.extend(new_subs)
            logger.info("Queued upgrade to per-second aggs for %d tickers", len(new_subs))

    def downgrade_to_minute_aggs(self) -> None:
        """Switch back from per-second to per-minute aggregates."""
        self._subscribe_second_aggs = False
        unsubs = [f"{self.PREFIX_SECOND_AGG}.{mt}" for mt in self._massive_to_yahoo]
        if unsubs and self._running:
            with self._lock:
                self._pending_unsubscribe.extend(unsubs)
            logger.info("Queued downgrade from per-second aggs for %d tickers", len(unsubs))

    def get_status(self) -> dict:
        """Return current feed status for display in the UI."""
        uptime = None
        if self.started_at:
            uptime = round(time.time() - self.started_at)

        return {
            "connected": self._connected,
            "running": self._running,
            "tickers": list(self._yahoo_to_massive.keys()),
            "massive_tickers": list(self._massive_to_yahoo.keys()),
            "msg_count": self.msg_count,
            "bar_count": self.bar_count,
            "trade_count": self.trade_count,
            "uptime_seconds": uptime,
            "errors": self.errors[-5:],  # last 5 errors
        }

    def get_trade_buffer(self, yahoo_ticker: str, clear: bool = True) -> list[dict]:
        """Get buffered trades for a ticker (useful for CVD calculation).

        If clear=True, the buffer is emptied after retrieval.
        """
        mt = self._yahoo_to_massive.get(yahoo_ticker)
        if not mt:
            return []

        with self._lock:
            trades = list(self._trade_buffer.get(mt, []))
            if clear and mt in self._trade_buffer:
                self._trade_buffer[mt] = []
        return trades

    # ----- Internal methods -----

    def _resolve_tickers(self) -> bool:
        """Resolve all Yahoo tickers to Massive tickers.

        When ``use_broad_subscriptions`` is True, resolution failures are
        non-fatal — the broad ``AM.*`` / ``T.*`` wildcard channels will
        receive bars for every contract, and ``_try_reverse_map`` will
        dynamically learn ticker mappings as messages arrive.  This makes
        the WebSocket feed resilient to flaky contract-resolution endpoints.
        """
        if self._provider is None:
            # Create a temporary provider for resolution
            self._provider = MassiveDataProvider(api_key=self._api_key)

        if not self._provider.is_available:
            # With broad subs we can still proceed — reverse mapping will
            # learn tickers dynamically from the incoming message stream.
            if self._use_broad_subscriptions:
                logger.info(
                    "Massive provider unavailable but broad subscriptions enabled "
                    "— will reverse-map tickers dynamically"
                )
                return True
            return False

        resolved_any = False
        for yt in self._yahoo_tickers:
            if yt in self._yahoo_to_massive:
                resolved_any = True
                continue
            mt = self._provider.resolve_from_yahoo(yt)
            if mt:
                self._yahoo_to_massive[yt] = mt
                self._massive_to_yahoo[mt] = yt
                resolved_any = True
            else:
                logger.warning("Could not resolve %s for WebSocket", yt)

        # With broad subscriptions, always succeed — _try_reverse_map will
        # dynamically discover tickers from the incoming message stream
        if self._use_broad_subscriptions:
            if not resolved_any:
                logger.info(
                    "No tickers pre-resolved, but broad subscriptions will auto-discover contracts via reverse mapping"
                )
            return True

        return resolved_any

    def _build_subscriptions(self) -> list[str]:
        """Build the list of WebSocket subscription strings.

        Channel naming follows the Massive/Polygon futures WS docs:
          AM.{ticker}  — per-minute aggregates
          A.{ticker}   — per-second aggregates
          T.{ticker}   — trades (tick-by-tick)
          Q.{ticker}   — quotes (bid/ask)

        When ``use_broad_subscriptions`` is True, subscribes to ``AM.*``
        and ``T.*`` wildcard channels instead of per-ticker channels.
        This is much more robust when the contracts endpoint is flaky,
        because it receives bars for *every* futures contract — we then
        filter to the ones we care about in ``_handle_bar`` /
        ``_handle_trade`` using ``_try_reverse_map``.
        """
        subs: list[str] = []

        if self._use_broad_subscriptions:
            # Broad wildcard subscriptions — most reliable path
            subs.append(f"{self.PREFIX_MINUTE_AGG}.*")
            if self._subscribe_trades:
                subs.append(f"{self.PREFIX_TRADE}.*")
            if self._subscribe_quotes:
                subs.append(f"{self.PREFIX_QUOTE}.*")
            if self._subscribe_second_aggs:
                subs.append(f"{self.PREFIX_SECOND_AGG}.*")
            return subs

        # Per-ticker subscriptions (legacy path)
        for mt in self._massive_to_yahoo:
            # Always subscribe to minute aggregates
            subs.append(f"{self.PREFIX_MINUTE_AGG}.{mt}")

            # Optionally subscribe to trades (for CVD)
            if self._subscribe_trades:
                subs.append(f"{self.PREFIX_TRADE}.{mt}")

            # Optionally subscribe to quotes
            if self._subscribe_quotes:
                subs.append(f"{self.PREFIX_QUOTE}.{mt}")

            # Optionally subscribe to second-level aggs
            if self._subscribe_second_aggs:
                subs.append(f"{self.PREFIX_SECOND_AGG}.{mt}")

        return subs

    def _try_reverse_map(self, symbol: str) -> str | None:
        """Attempt to map an unknown Massive ticker back to a Yahoo ticker.

        With broad subscriptions (``AM.*``, ``T.*``) we receive bars for
        *every* futures contract on the exchange.  Most of them are not
        in our asset universe and should be ignored.

        Mapping strategy:
          1. Direct lookup (already resolved via ``_resolve_tickers``).
          2. Strip trailing month-year code to get the root product code
             (e.g., ``ESZ5`` → ``ES``, ``MESZ5`` → ``MES``, ``GCG6`` → ``GC``)
             and check against ``MASSIVE_PRODUCT_TO_YAHOO``.
          3. If still unknown, return None (symbol is outside our universe).
        """
        # 1. Already known?
        if symbol in self._massive_to_yahoo:
            return self._massive_to_yahoo[symbol]

        # 2. Derive root product code by stripping the 1-letter month + year digits
        #    CME convention: ticker = <ROOT><month_letter><year_digit(s)>
        #    e.g. ESZ5, MESZ25, GCG6, MCLM5
        import re

        m = re.match(r"^([A-Z]{1,4}?)[FGHJKMNQUVXZ]\d{1,2}$", symbol)
        if m:
            root = m.group(1)
            yahoo = MASSIVE_PRODUCT_TO_YAHOO.get(root)
            if yahoo:
                # Cache the mapping for future lookups
                self._massive_to_yahoo[symbol] = yahoo
                self._yahoo_to_massive.setdefault(yahoo, symbol)
                logger.debug("Reverse-mapped %s → root %s → Yahoo %s", symbol, root, yahoo)
                return yahoo

        return None

    # ----- Raw websockets loop (primary) -----

    def _run_loop(self) -> None:
        """Background thread entry point.

        Runs an asyncio event loop that keeps the raw WebSocket connection
        alive with automatic reconnect and exponential back-off.
        """
        import asyncio

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._async_ws_loop())
        except Exception as exc:
            logger.error("WS event loop exited unexpectedly: %s", exc)
        finally:
            loop.close()

    async def _async_ws_loop(self) -> None:
        """Async WebSocket loop with auto-reconnect."""
        import asyncio
        import json as _json

        try:
            import websockets  # type: ignore[import-untyped]
        except ImportError:
            logger.error("websockets package not installed — live feed unavailable")
            return

        reconnect_delay = 1
        max_reconnect_delay = 60
        _MIN_STABLE_SECS = 30  # only reset backoff after this many seconds of uptime
        _policy_violation_count = 0
        _MAX_POLICY_VIOLATIONS = 3  # give up after 3 consecutive 1008s

        while self._running:
            ws = None
            _connected_at = None
            try:
                logger.info("Connecting to %s …", _WS_FUTURES_URI)
                ws = await websockets.connect(
                    _WS_FUTURES_URI,
                    ping_interval=30,
                    ping_timeout=10,
                    close_timeout=5,
                )

                # --- 1. Authenticate ---
                # The server first sends a "connected" status message upon
                # TCP/TLS handshake, *before* we even send auth.  We need to
                # consume that greeting, then send our auth payload, and
                # finally wait for the "auth_success" response.

                # 1a. Consume the initial "connected" greeting
                greeting_raw = await asyncio.wait_for(ws.recv(), timeout=15)
                greeting = _json.loads(greeting_raw)
                greeting_items = greeting if isinstance(greeting, list) else [greeting]
                is_greeting = any(item.get("status") == "connected" for item in greeting_items)
                if is_greeting:
                    logger.info("WebSocket connected, sending auth …")
                else:
                    # Unexpected first message — log but continue with auth
                    logger.warning(
                        "Unexpected first WS message (expected 'connected'): %s",
                        greeting_raw[:200],
                    )

                # 1b. Send auth and wait for auth_success
                auth_payload = _json.dumps({"action": "auth", "params": self._api_key})
                await ws.send(auth_payload)
                auth_resp_raw = await asyncio.wait_for(ws.recv(), timeout=15)
                auth_resp = _json.loads(auth_resp_raw)
                # auth_resp is typically [{"ev":"status","status":"auth_success",...}]
                auth_items = auth_resp if isinstance(auth_resp, list) else [auth_resp]
                auth_ok = any(item.get("status") == "auth_success" for item in auth_items)
                if not auth_ok:
                    error_msg = f"WS auth failed: {auth_resp_raw[:300].decode() if isinstance(auth_resp_raw, bytes) else auth_resp_raw[:300]}"
                    logger.error(error_msg)
                    self.errors.append(error_msg)
                    await ws.close()
                    raise ConnectionError(error_msg)

                logger.info("WebSocket authenticated successfully")

                # --- 2. Subscribe ---
                subs = self._build_subscriptions()
                if subs:
                    sub_payload = _json.dumps({"action": "subscribe", "params": ",".join(subs)})
                    await ws.send(sub_payload)
                    logger.info(
                        "Subscribed to %d channels: %s",
                        len(subs),
                        ", ".join(subs[:10]) + (" …" if len(subs) > 10 else ""),
                    )

                self._connected = True
                self._ws = ws  # store for stop() to close
                _connected_at = asyncio.get_event_loop().time()

                # --- 3. Message loop ---
                async for raw in ws:
                    if not self._running:
                        break

                    # Process any pending dynamic sub/unsub requests
                    await self._flush_pending_subs(ws)

                    try:
                        data = _json.loads(raw)
                    except _json.JSONDecodeError:
                        continue

                    items = data if isinstance(data, list) else [data]
                    self._handle_messages(items)

            except asyncio.CancelledError:
                break
            except Exception as exc:
                error_msg = f"WebSocket error: {exc}"
                is_policy_violation = "1008" in str(exc) or "policy violation" in str(exc).lower()
                if is_policy_violation:
                    _policy_violation_count += 1
                    logger.error(error_msg)
                    if _policy_violation_count >= _MAX_POLICY_VIOLATIONS:
                        logger.warning(
                            "Massive WebSocket: received %d consecutive 1008 policy-violation "
                            "errors — your API plan may not include real-time futures WebSocket "
                            "access. Stopping WS reconnect loop. REST polling will continue. "
                            "To re-enable, restart the engine or upgrade your Massive plan.",
                            _policy_violation_count,
                        )
                        self._running = False
                else:
                    _policy_violation_count = 0  # reset counter on non-1008 error
                    logger.error(error_msg)
                self.errors.append(error_msg)
                if len(self.errors) > 100:
                    self.errors = self.errors[-50:]
            else:
                # Clean exit from message loop — reset violation counter
                _policy_violation_count = 0
            finally:
                # Only reset backoff if the connection was stable for 30+ seconds.
                # This prevents tight reconnect loops when the server keeps kicking
                # us shortly after auth (e.g. 1008 policy violation every ~8s).
                if self._connected and _connected_at is not None:
                    _uptime = asyncio.get_event_loop().time() - _connected_at
                    if _uptime >= _MIN_STABLE_SECS:
                        reconnect_delay = 1  # was genuinely stable — reset backoff
                        _policy_violation_count = 0  # stable connection — reset
                self._connected = False
                self._ws = None
                if ws is not None:
                    with contextlib.suppress(Exception):
                        await ws.close()

            if not self._running:
                break

            logger.info("Reconnecting in %ds …", reconnect_delay)
            await asyncio.sleep(reconnect_delay)
            reconnect_delay = min(reconnect_delay * 2, max_reconnect_delay)

    async def _flush_pending_subs(self, ws) -> None:
        """Send any queued subscribe / unsubscribe requests."""
        import json as _json

        with self._lock:
            to_sub = list(self._pending_subscribe)
            self._pending_subscribe.clear()
            to_unsub = list(self._pending_unsubscribe)
            self._pending_unsubscribe.clear()

        if to_sub:
            try:
                payload = _json.dumps({"action": "subscribe", "params": ",".join(to_sub)})
                await ws.send(payload)
                logger.info("Dynamic subscribe: %s", to_sub)
            except Exception as exc:
                logger.error("Failed dynamic subscribe: %s", exc)

        if to_unsub:
            try:
                payload = _json.dumps({"action": "unsubscribe", "params": ",".join(to_unsub)})
                await ws.send(payload)
                logger.info("Dynamic unsubscribe: %s", to_unsub)
            except Exception as exc:
                logger.error("Failed dynamic unsubscribe: %s", exc)

    # ----- Message dispatch (handles both SDK objects & raw JSON dicts) -----

    def _handle_messages(self, messages) -> None:
        """Process a batch of incoming WebSocket messages.

        Each *message* may be:
          - A raw JSON ``dict`` with short keys (``ev``, ``sym``, ``o``…)
            — this is the normal path when using raw ``websockets``.
          - An SDK model object with long attribute names
            (``event_type``, ``symbol``, ``open``…) — kept for
            backwards-compat if someone plugs in the SDK client.
        """
        if not messages:
            return

        self.msg_count += 1

        for msg in messages:
            # Detect whether this is a raw dict or an SDK object
            if isinstance(msg, dict):
                ev = msg.get("ev")
                sym = msg.get("sym")
            else:
                ev = getattr(msg, "event_type", None)
                sym = getattr(msg, "symbol", None)

            if ev in ("AM", "A"):
                self._handle_bar(msg, sym)
            elif ev == "T":
                self._handle_trade(msg, sym)
            elif ev == "Q":
                self._handle_quote(msg, sym)
            # "status" messages (auth ack, sub ack) are silently ignored.

    def _handle_bar(self, msg, symbol: str | None) -> None:
        """Process an aggregate bar message (AM or A channel).

        Raw JSON keys:  o, h, l, c, v, s (start ms), e (end ms), n (txns), vw
        SDK attr names: open, high, low, close, volume, start_timestamp, …

        With broad subscriptions we receive bars for every contract on the
        exchange.  ``_try_reverse_map`` filters to our asset universe and
        dynamically learns new contract tickers (e.g. after a roll).
        """
        if not symbol:
            return

        # When using broad subs, filter to our universe (and auto-learn tickers)
        if self._use_broad_subscriptions:
            yahoo = self._try_reverse_map(symbol)
            if yahoo is None:
                return  # not in our asset universe — ignore

        if isinstance(msg, dict):
            bar = {
                "open": msg.get("o"),
                "high": msg.get("h"),
                "low": msg.get("l"),
                "close": msg.get("c"),
                "volume": msg.get("v", 0) or 0,
                "transactions": msg.get("n"),
                "start_timestamp": msg.get("s"),
                "end_timestamp": msg.get("e"),
                "total_value": msg.get("vw"),
                "received_at": time.time(),
            }
        else:
            bar = {
                "open": getattr(msg, "open", None),
                "high": getattr(msg, "high", None),
                "low": getattr(msg, "low", None),
                "close": getattr(msg, "close", None),
                "volume": getattr(msg, "volume", 0) or 0,
                "transactions": getattr(msg, "transactions", None),
                "start_timestamp": getattr(msg, "start_timestamp", None),
                "end_timestamp": getattr(msg, "end_timestamp", None),
                "total_value": getattr(msg, "total_value", None),
                "received_at": time.time(),
            }

        with self._lock:
            self._latest_bars[symbol] = bar
        self.bar_count += 1

        # Fire callbacks
        yahoo_ticker = self._massive_to_yahoo.get(symbol)
        if yahoo_ticker:
            for cb in self._on_bar_callbacks:
                try:
                    cb(yahoo_ticker, bar)
                except Exception as exc:
                    logger.error("Bar callback error: %s", exc)

        # Push to cache if cache module is available
        self._push_bar_to_cache(symbol, bar)

    def _handle_trade(self, msg, symbol: str | None) -> None:
        """Process a trade message (T channel).

        Raw JSON keys:  p (price), s (size), t (timestamp ms), q (sequence)
        SDK attr names: price, size, timestamp, sequence_number

        With broad subscriptions, filters to our asset universe via
        ``_try_reverse_map``.
        """
        if not symbol:
            return

        # When using broad subs, filter to our universe
        if self._use_broad_subscriptions:
            yahoo = self._try_reverse_map(symbol)
            if yahoo is None:
                return  # not in our asset universe — ignore

        if isinstance(msg, dict):
            trade = {
                "price": msg.get("p"),
                "size": msg.get("s"),
                "timestamp": msg.get("t"),
                "sequence": msg.get("q"),
            }
        else:
            trade = {
                "price": getattr(msg, "price", None),
                "size": getattr(msg, "size", None),
                "timestamp": getattr(msg, "timestamp", None),
                "sequence": getattr(msg, "sequence_number", None),
            }

        with self._lock:
            self._latest_trades[symbol] = trade
            # Buffer trades for CVD calculation
            if symbol not in self._trade_buffer:
                self._trade_buffer[symbol] = []
            self._trade_buffer[symbol].append(trade)
            # Cap buffer size (keep last 10000 trades per ticker)
            if len(self._trade_buffer[symbol]) > 10000:
                self._trade_buffer[symbol] = self._trade_buffer[symbol][-5000:]

        self.trade_count += 1

        # Fire callbacks
        yahoo_ticker = self._massive_to_yahoo.get(symbol)
        if yahoo_ticker:
            for cb in self._on_trade_callbacks:
                try:
                    cb(yahoo_ticker, trade)
                except Exception as exc:
                    logger.error("Trade callback error: %s", exc)

    def _handle_quote(self, msg, symbol: str | None) -> None:
        """Process a quote message (Q channel).

        Raw JSON keys:  bp, bs, ap, as, t
        SDK attr names: bid_price, bid_size, ask_price, ask_size, …
        """
        if not symbol:
            return

        # When using broad subs, filter to our universe
        if self._use_broad_subscriptions:
            yahoo = self._try_reverse_map(symbol)
            if yahoo is None:
                return

        if isinstance(msg, dict):
            quote = {
                "bid": msg.get("bp"),
                "bid_size": msg.get("bs"),
                "ask": msg.get("ap"),
                "ask_size": msg.get("as"),
                "timestamp": msg.get("t"),
            }
        else:
            quote = {
                "bid": getattr(msg, "bid_price", None),
                "bid_size": getattr(msg, "bid_size", None),
                "ask": getattr(msg, "ask_price", None),
                "ask_size": getattr(msg, "ask_size", None),
                "bid_timestamp": getattr(msg, "bid_timestamp", None),
                "ask_timestamp": getattr(msg, "ask_timestamp", None),
            }

        with self._lock:
            self._latest_quotes[symbol] = quote

    def _push_bar_to_cache(self, massive_ticker: str, bar: dict) -> None:
        """Push a new bar into the Redis/memory cache for the dashboard to read."""
        try:
            from lib.core.cache import (
                TTL_INTRADAY,
                _cache_key,
                _df_to_bytes,
                cache_set,
            )

            yahoo_ticker = self._massive_to_yahoo.get(massive_ticker)
            if not yahoo_ticker:
                return

            # Create a single-row DataFrame matching the expected OHLCV format
            bar_df = pd.DataFrame(
                [
                    {
                        "Open": bar.get("open"),
                        "High": bar.get("high"),
                        "Low": bar.get("low"),
                        "Close": bar.get("close"),
                        "Volume": bar.get("volume", 0),
                    }
                ]
            )

            # Store under a special "live" key so the dashboard knows
            # there's fresh data without interfering with the historical cache
            key = _cache_key("live_bar", yahoo_ticker)
            cache_set(key, _df_to_bytes(bar_df), TTL_INTRADAY)

        except Exception:
            pass  # Non-critical — don't crash the WS loop


# ---------------------------------------------------------------------------
# Singleton provider
# ---------------------------------------------------------------------------

_provider_instance: MassiveDataProvider | None = None
_provider_lock = threading.Lock()


def get_massive_provider(api_key: str | None = None) -> MassiveDataProvider:
    """Get or create the singleton MassiveDataProvider.

    Thread-safe. On first call, initializes the client with the API key
    from the argument or MASSIVE_API_KEY environment variable.
    """
    global _provider_instance
    with _provider_lock:
        if _provider_instance is None:
            _provider_instance = MassiveDataProvider(api_key=api_key)
        return _provider_instance


def reset_provider() -> None:
    """Reset the singleton provider (useful for testing)."""
    global _provider_instance
    with _provider_lock:
        _provider_instance = None


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------


def is_massive_available() -> bool:
    """Quick check: is the Massive API configured and ready?"""
    return get_massive_provider().is_available


def get_massive_aggs(
    yahoo_ticker: str,
    interval: str = "5m",
    period: str = "5d",
) -> pd.DataFrame:
    """Convenience: fetch aggregates via the singleton provider."""
    return get_massive_provider().get_aggs(yahoo_ticker, interval, period)


def get_massive_daily(
    yahoo_ticker: str,
    period: str = "10d",
) -> pd.DataFrame:
    """Convenience: fetch daily bars via the singleton provider."""
    return get_massive_provider().get_daily(yahoo_ticker, period)


def get_massive_snapshot(
    yahoo_ticker: str,
) -> dict | None:
    """Convenience: get real-time snapshot via the singleton provider."""
    return get_massive_provider().get_snapshot(yahoo_ticker=yahoo_ticker)
