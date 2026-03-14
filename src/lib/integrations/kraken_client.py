"""
Kraken Exchange Data Provider
==============================
Provides historical OHLCV bars (REST) and real-time streaming (WebSocket)
for crypto pairs via the Kraken exchange API.

This integration plugs into the existing futures data pipeline alongside
Massive and yfinance.  Crypto pairs use a ``KRAKEN:`` prefix internally
(e.g. ``KRAKEN:XBTUSD``) so the cache layer and engine can distinguish
them from CME futures tickers.

REST API (public, no auth required for market data):
  - OHLCV candles: GET https://api.kraken.com/0/public/OHLC
  - Ticker snapshot: GET https://api.kraken.com/0/public/Ticker
  - Asset pairs info: GET https://api.kraken.com/0/public/AssetPairs
  - Server time: GET https://api.kraken.com/0/public/Time

WebSocket API v2 (public, no auth required for market data):
  - wss://ws.kraken.com/v2  (production)
  - Channels: ohlc, ticker, trade, book

Private endpoints (require KRAKEN_API_KEY + KRAKEN_API_SECRET):
  - Balance, open orders, trade history — used for future position tracking
  - Not required for the read-only market data pipeline

Environment:
    KRAKEN_API_KEY     — API key (optional, only for private endpoints)
    KRAKEN_API_SECRET  — API secret (optional, only for private endpoints)

Usage:
    from lib.integrations.kraken_client import (
        get_kraken_provider,
        KrakenFeedManager,
        KRAKEN_PAIRS,
    )

    provider = get_kraken_provider()
    df = provider.get_ohlcv("XBTUSD", interval="5m", since_hours=24)

    feed = KrakenFeedManager(pairs=["XBT/USD", "ETH/USD"])
    feed.start()
    feed.latest_bars  # dict of latest OHLCV by pair
    await feed.stop()
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import hashlib
import hmac
import json
import logging
import os
import threading
import time
import urllib.parse
from datetime import UTC, datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

_EST = ZoneInfo("America/New_York")

logger = logging.getLogger("kraken_client")
logger.setLevel(logging.INFO)
if not logger.handlers:
    _h = logging.StreamHandler()
    _fmt = logging.Formatter("[KRAKEN] %(asctime)s  %(message)s", "%H:%M:%S")
    _fmt.converter = lambda *_: datetime.now(_EST).timetuple()
    _h.setFormatter(_fmt)
    logger.addHandler(_h)


# ═══════════════════════════════════════════════════════════════════════════
# Constants & Pair Mapping
# ═══════════════════════════════════════════════════════════════════════════

# REST base URLs
KRAKEN_REST_URL = "https://api.kraken.com"
KRAKEN_REST_PUBLIC = f"{KRAKEN_REST_URL}/0/public"
KRAKEN_REST_PRIVATE = f"{KRAKEN_REST_URL}/0/private"

# WebSocket v2 URL (public market data — no auth needed)
KRAKEN_WS_URL = "wss://ws.kraken.com/v2"

# Internal ticker prefix — used throughout the pipeline to identify Kraken
# data vs CME futures.  e.g. "KRAKEN:XBTUSD" in Redis keys and ASSETS dict.
KRAKEN_PREFIX = "KRAKEN:"

# ---------------------------------------------------------------------------
# Kraken pair mapping
# ---------------------------------------------------------------------------
# Maps our internal display name → Kraken REST pair name (for /OHLC etc.)
# and the WebSocket v2 pair name (for subscriptions).
#
# Kraken REST API uses "XXBTZUSD" style for some pairs and "XBTUSD" for
# others.  The /AssetPairs endpoint returns the canonical form.  We keep
# a curated map of the most liquid pairs we want to trade.
#
# The ``ws_pair`` is the symbol format used in WebSocket v2 subscribe
# messages (e.g. "XBT/USD").
#
# The ``internal_ticker`` is what appears in models.ASSETS and Redis keys.
# ---------------------------------------------------------------------------

KRAKEN_PAIRS: dict[str, dict[str, str]] = {
    # ── Major crypto ────────────────────────────────────────────────────────
    "Bitcoin": {
        "rest_pair": "XXBTZUSD",  # Kraken REST canonical
        "ws_pair": "XBT/USD",  # WebSocket v2 format
        "internal_ticker": "KRAKEN:XBTUSD",
        "base": "XBT",
        "quote": "USD",
    },
    "Ethereum": {
        "rest_pair": "XETHZUSD",
        "ws_pair": "ETH/USD",
        "internal_ticker": "KRAKEN:ETHUSD",
        "base": "ETH",
        "quote": "USD",
    },
    "Solana": {
        "rest_pair": "SOLUSD",
        "ws_pair": "SOL/USD",
        "internal_ticker": "KRAKEN:SOLUSD",
        "base": "SOL",
        "quote": "USD",
    },
    # ── Alt-coins with decent volume ────────────────────────────────────────
    "Chainlink": {
        "rest_pair": "LINKUSD",
        "ws_pair": "LINK/USD",
        "internal_ticker": "KRAKEN:LINKUSD",
        "base": "LINK",
        "quote": "USD",
    },
    "Avalanche": {
        "rest_pair": "AVAXUSD",
        "ws_pair": "AVAX/USD",
        "internal_ticker": "KRAKEN:AVAXUSD",
        "base": "AVAX",
        "quote": "USD",
    },
    "Polkadot": {
        "rest_pair": "DOTUSD",
        "ws_pair": "DOT/USD",
        "internal_ticker": "KRAKEN:DOTUSD",
        "base": "DOT",
        "quote": "USD",
    },
    "Cardano": {
        "rest_pair": "ADAUSD",
        "ws_pair": "ADA/USD",
        "internal_ticker": "KRAKEN:ADAUSD",
        "base": "ADA",
        "quote": "USD",
    },
    "Polygon": {
        "rest_pair": "POLUSD",
        "ws_pair": "POL/USD",
        "internal_ticker": "KRAKEN:POLUSD",
        "base": "POL",
        "quote": "USD",
    },
    "XRP": {
        "rest_pair": "XXRPZUSD",
        "ws_pair": "XRP/USD",
        "internal_ticker": "KRAKEN:XRPUSD",
        "base": "XRP",
        "quote": "USD",
    },
}

# Convenience lookups
REST_PAIR_TO_NAME: dict[str, str] = {v["rest_pair"]: k for k, v in KRAKEN_PAIRS.items()}
WS_PAIR_TO_NAME: dict[str, str] = {v["ws_pair"]: k for k, v in KRAKEN_PAIRS.items()}
INTERNAL_TO_NAME: dict[str, str] = {v["internal_ticker"]: k for k, v in KRAKEN_PAIRS.items()}
INTERNAL_TO_REST: dict[str, str] = {v["internal_ticker"]: v["rest_pair"] for v in KRAKEN_PAIRS.values()}
INTERNAL_TO_WS: dict[str, str] = {v["internal_ticker"]: v["ws_pair"] for v in KRAKEN_PAIRS.values()}
NAME_TO_INTERNAL: dict[str, str] = {k: v["internal_ticker"] for k, v in KRAKEN_PAIRS.items()}
REST_TO_INTERNAL: dict[str, str] = {v["rest_pair"]: v["internal_ticker"] for v in KRAKEN_PAIRS.values()}

# All internal tickers as a frozenset for quick membership checks
ALL_KRAKEN_TICKERS: frozenset[str] = frozenset(v["internal_ticker"] for v in KRAKEN_PAIRS.values())

# ---------------------------------------------------------------------------
# Interval mapping: our internal intervals → Kraken OHLC interval (minutes)
# ---------------------------------------------------------------------------
# Kraken REST OHLC endpoint accepts interval in minutes.
# Supported: 1, 5, 15, 30, 60, 240, 1440 (1d), 10080 (1w), 21600 (15d)
INTERVAL_TO_KRAKEN_MINUTES: dict[str, int] = {
    "1m": 1,
    "5m": 5,
    "15m": 15,
    "30m": 30,
    "1h": 60,
    "60m": 60,
    "4h": 240,
    "1d": 1440,
    "1wk": 10080,
}

# WebSocket v2 OHLC interval values (same as REST, in minutes)
WS_OHLC_INTERVALS: dict[str, int] = INTERVAL_TO_KRAKEN_MINUTES.copy()


def is_kraken_ticker(ticker: str) -> bool:
    """Return True if the ticker is a Kraken crypto pair."""
    return ticker.startswith(KRAKEN_PREFIX)


def kraken_rest_pair(internal_ticker: str) -> str | None:
    """Convert an internal KRAKEN:XBTUSD ticker to the REST pair name."""
    return INTERNAL_TO_REST.get(internal_ticker)


def kraken_ws_pair(internal_ticker: str) -> str | None:
    """Convert an internal KRAKEN:XBTUSD ticker to the WS v2 pair name."""
    return INTERNAL_TO_WS.get(internal_ticker)


# ═══════════════════════════════════════════════════════════════════════════
# REST Client
# ═══════════════════════════════════════════════════════════════════════════


class KrakenDataProvider:
    """REST client for Kraken public + private API endpoints.

    Public endpoints (OHLCV, ticker, asset pairs) require no authentication.
    Private endpoints (balance, trade history) require API key + secret,
    which are read from KRAKEN_API_KEY / KRAKEN_API_SECRET env vars.

    Thread-safe: uses ``requests.Session`` with connection pooling.
    """

    # Rate limiting: Kraken allows ~1 req/sec for public endpoints,
    # ~15 req/min for private.  We enforce a small delay between calls.
    _PUBLIC_RATE_LIMIT = 0.35  # seconds between public API calls
    _PRIVATE_RATE_LIMIT = 1.0  # seconds between private API calls

    def __init__(
        self,
        api_key: str | None = None,
        api_secret: str | None = None,
    ) -> None:
        self._api_key: str = api_key or os.getenv("KRAKEN_API_KEY", "") or ""
        self._api_secret: str = api_secret or os.getenv("KRAKEN_API_SECRET", "") or ""
        self._session: Any = None
        self._lock = threading.Lock()
        self._last_public_call: float = 0.0
        self._last_private_call: float = 0.0

        self._init_session()

    def _init_session(self) -> None:
        """Lazily create a requests session with connection pooling."""
        try:
            import requests  # type: ignore[import-untyped]

            self._session = requests.Session()
            self._session.headers.update(
                {
                    "User-Agent": "FuturesCoPilot/1.0",
                    "Accept": "application/json",
                }
            )
            logger.info(
                "Kraken REST client initialized (auth=%s)",
                "yes" if self._api_key else "no",
            )
        except ImportError:
            logger.error("requests library not installed — Kraken REST client disabled")
            self._session = None

    @property
    def is_available(self) -> bool:
        """Whether the REST client is ready to make API calls."""
        return self._session is not None

    @property
    def has_auth(self) -> bool:
        """Whether API key + secret are configured for private endpoints."""
        return bool(self._api_key and self._api_secret)

    # ── Rate limiting ─────────────────────────────────────────────────────

    def _rate_limit_public(self) -> None:
        """Enforce rate limiting for public API calls."""
        with self._lock:
            elapsed = time.monotonic() - self._last_public_call
            if elapsed < self._PUBLIC_RATE_LIMIT:
                time.sleep(self._PUBLIC_RATE_LIMIT - elapsed)
            self._last_public_call = time.monotonic()

    def _rate_limit_private(self) -> None:
        """Enforce rate limiting for private API calls."""
        with self._lock:
            elapsed = time.monotonic() - self._last_private_call
            if elapsed < self._PRIVATE_RATE_LIMIT:
                time.sleep(self._PRIVATE_RATE_LIMIT - elapsed)
            self._last_private_call = time.monotonic()

    # ── Auth helpers ──────────────────────────────────────────────────────

    def _sign_request(self, url_path: str, data: dict[str, Any]) -> dict[str, str]:
        """Generate Kraken API signature headers for a private request.

        Uses the standard Kraken HMAC-SHA512 signing scheme:
          1. SHA256(nonce + POST body)
          2. HMAC-SHA512(urlpath, sha256_digest) using base64-decoded secret
        """
        nonce = str(int(time.time() * 1000))
        data["nonce"] = nonce
        post_data = urllib.parse.urlencode(data)

        # SHA256 of nonce + POST data
        sha256_digest = hashlib.sha256((nonce + post_data).encode("utf-8")).digest()

        # HMAC-SHA512 of url_path + sha256_digest
        secret_bytes = base64.b64decode(self._api_secret)
        hmac_digest = hmac.new(
            secret_bytes,
            url_path.encode("utf-8") + sha256_digest,
            hashlib.sha512,
        ).digest()

        return {
            "API-Key": self._api_key,
            "API-Sign": base64.b64encode(hmac_digest).decode("utf-8"),
        }

    # ── Public REST endpoints ─────────────────────────────────────────────

    def _public_get(self, endpoint: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        """Make a rate-limited GET to a Kraken public endpoint.

        Returns the ``result`` dict from the response, or raises on error.

        Automatically retries on ``EGeneral:Too many requests`` with
        exponential backoff (1s, 2s, 4s) before giving up.
        """
        if not self.is_available:
            raise RuntimeError("Kraken REST client not initialized")

        url = f"{KRAKEN_REST_PUBLIC}/{endpoint}"

        _MAX_RETRIES = 3
        _RETRY_DELAYS = (1.0, 2.0, 4.0)  # seconds — exponential backoff

        for attempt in range(_MAX_RETRIES + 1):
            self._rate_limit_public()

            try:
                resp = self._session.get(url, params=params, timeout=15)
                resp.raise_for_status()
                body = resp.json()
            except Exception as exc:
                logger.error("Kraken API error (%s): %s", endpoint, exc)
                raise

            errors = body.get("error", [])
            if errors:
                err_msg = "; ".join(str(e) for e in errors)
                # Retry on rate-limit errors with backoff
                if ("Too many requests" in err_msg or "EAPI:Rate limit" in err_msg) and attempt < _MAX_RETRIES:
                    delay = _RETRY_DELAYS[attempt]
                    logger.warning(
                        "Kraken rate-limit on %s (attempt %d/%d) — backing off %.1fs",
                        endpoint,
                        attempt + 1,
                        _MAX_RETRIES,
                        delay,
                    )
                    time.sleep(delay)
                    continue
                logger.error("Kraken API returned errors (%s): %s", endpoint, err_msg)
                raise RuntimeError(f"Kraken API error: {err_msg}")

            return body.get("result", {})

        # Exhausted retries
        raise RuntimeError(f"Kraken API error: Too many requests for {endpoint} after {_MAX_RETRIES} retries")

    def _private_post(self, endpoint: str, data: dict[str, Any] | None = None) -> dict[str, Any]:
        """Make a rate-limited signed POST to a Kraken private endpoint."""
        if not self.is_available:
            raise RuntimeError("Kraken REST client not initialized")
        if not self.has_auth:
            raise RuntimeError("Kraken API key and secret required for private endpoints")

        self._rate_limit_private()
        url_path = f"/0/private/{endpoint}"
        url = f"{KRAKEN_REST_URL}{url_path}"
        data = data or {}

        headers = self._sign_request(url_path, data)

        try:
            resp = self._session.post(
                url,
                data=urllib.parse.urlencode(data),
                headers={
                    **headers,
                    "Content-Type": "application/x-www-form-urlencoded",
                },
                timeout=15,
            )
            resp.raise_for_status()
            body = resp.json()
        except Exception as exc:
            logger.error("Kraken private API error (%s): %s", endpoint, exc)
            raise

        errors = body.get("error", [])
        if errors:
            err_msg = "; ".join(str(e) for e in errors)
            raise RuntimeError(f"Kraken private API error: {err_msg}")

        return body.get("result", {})

    # ── Server time (connectivity test) ───────────────────────────────────

    def get_server_time(self) -> dict[str, Any]:
        """Return Kraken server time — useful as a connectivity / auth test."""
        return self._public_get("Time")

    def ping(self) -> bool:
        """Quick connectivity check — returns True if Kraken API responds."""
        try:
            self.get_server_time()
            return True
        except Exception:
            return False

    # ── Asset pairs info ──────────────────────────────────────────────────

    def get_asset_pairs(self, pairs: list[str] | None = None) -> dict[str, Any]:
        """Return asset pair info from Kraken.

        Args:
            pairs: Optional list of pair names to query (e.g. ["XXBTZUSD"]).
                   If None, returns all available pairs.
        """
        params: dict[str, Any] = {}
        if pairs:
            params["pair"] = ",".join(pairs)
        return self._public_get("AssetPairs", params)

    # ── Ticker snapshot ───────────────────────────────────────────────────

    def get_ticker(self, pair: str) -> dict[str, Any]:
        """Return current ticker info for a pair.

        Returns dict with: a (ask), b (bid), c (last trade), v (volume),
        p (vwap), t (number of trades), l (low), h (high), o (open).
        Each value is [today, last_24h].
        """
        result = self._public_get("Ticker", {"pair": pair})
        # Result is keyed by the pair name — return the inner dict
        if result:
            return next(iter(result.values()))
        return {}

    def get_ticker_by_internal(self, internal_ticker: str) -> dict[str, Any]:
        """Get ticker for an internal ticker like ``KRAKEN:XBTUSD``."""
        rest_pair = kraken_rest_pair(internal_ticker)
        if not rest_pair:
            raise ValueError(f"Unknown Kraken ticker: {internal_ticker}")
        return self.get_ticker(rest_pair)

    def get_all_tickers(self, pairs: list[str] | None = None) -> dict[str, dict[str, Any]]:
        """Return ticker data for multiple pairs in one call.

        Args:
            pairs: List of REST pair names. If None, uses all KRAKEN_PAIRS.

        Returns:
            Dict mapping pair name → ticker data.
        """
        if pairs is None:
            pairs = [p["rest_pair"] for p in KRAKEN_PAIRS.values()]
        result = self._public_get("Ticker", {"pair": ",".join(pairs)})
        return result

    # ── OHLCV historical bars ─────────────────────────────────────────────

    def get_ohlcv(
        self,
        pair: str,
        interval: str = "5m",
        since_hours: float | None = None,
        since_timestamp: int | None = None,
    ) -> pd.DataFrame:
        """Fetch historical OHLCV candles from Kraken.

        Args:
            pair: Kraken REST pair name (e.g. "XXBTZUSD") or internal ticker
                  (e.g. "KRAKEN:XBTUSD").
            interval: Candle interval ("1m", "5m", "15m", "30m", "1h", "4h", "1d").
            since_hours: Fetch candles from this many hours ago until now.
                         Kraken returns max 720 candles per call.
            since_timestamp: Unix timestamp to start from (alternative to since_hours).

        Returns:
            DataFrame with columns: Open, High, Low, Close, Volume, VWAP, Trades
            and a tz-aware DatetimeIndex (America/New_York).
        """
        # Resolve internal ticker to REST pair if needed
        if pair.startswith(KRAKEN_PREFIX):
            rest_pair = kraken_rest_pair(pair)
            if not rest_pair:
                raise ValueError(f"Unknown Kraken ticker: {pair}")
            pair = rest_pair

        # Map interval to Kraken minutes
        kraken_interval = INTERVAL_TO_KRAKEN_MINUTES.get(interval)
        if kraken_interval is None:
            raise ValueError(
                f"Unsupported interval '{interval}'. Valid: {', '.join(INTERVAL_TO_KRAKEN_MINUTES.keys())}"
            )

        params: dict[str, Any] = {
            "pair": pair,
            "interval": kraken_interval,
        }

        if since_timestamp is not None:
            params["since"] = since_timestamp
        elif since_hours is not None:
            since_ts = int((datetime.now(tz=UTC) - timedelta(hours=since_hours)).timestamp())
            params["since"] = since_ts

        result = self._public_get("OHLC", params)

        # Result is keyed by the pair name + "last" timestamp
        # We need to find the data key (not "last")
        data_key = None
        for key in result:
            if key != "last":
                data_key = key
                break

        if data_key is None or not result.get(data_key):
            logger.warning("No OHLCV data returned for %s", pair)
            return pd.DataFrame()

        rows = result[data_key]
        # Each row: [time, open, high, low, close, vwap, volume, count]
        records = []
        for row in rows:
            records.append(
                {
                    "timestamp": int(row[0]),
                    "Open": float(row[1]),
                    "High": float(row[2]),
                    "Low": float(row[3]),
                    "Close": float(row[4]),
                    "VWAP": float(row[5]),
                    "Volume": float(row[6]),
                    "Trades": int(row[7]),
                }
            )

        if not records:
            return pd.DataFrame()

        df = pd.DataFrame(records)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
        df = df.set_index("timestamp")
        # Convert to ET for consistency with the rest of the pipeline
        assert isinstance(df.index, pd.DatetimeIndex)
        df.index = df.index.tz_convert(_EST)
        df.index.name = None

        logger.debug(
            "Kraken OHLCV: %s %s — %d bars (%s → %s)",
            pair,
            interval,
            len(df),
            str(df.index[0]) if len(df) > 0 else "—",
            str(df.index[-1]) if len(df) > 0 else "—",
        )

        return df

    def get_ohlcv_by_internal(
        self,
        internal_ticker: str,
        interval: str = "5m",
        since_hours: float | None = None,
    ) -> pd.DataFrame:
        """Convenience: fetch OHLCV using an internal ticker like KRAKEN:XBTUSD."""
        return self.get_ohlcv(internal_ticker, interval=interval, since_hours=since_hours)

    def get_ohlcv_period(
        self,
        pair: str,
        interval: str = "5m",
        period: str = "5d",
    ) -> pd.DataFrame:
        """Fetch OHLCV for a period string (e.g. "5d", "1mo").

        Translates the period to hours and calls get_ohlcv.
        Kraken returns a maximum of 720 candles per call, so for long
        periods at fine intervals, multiple calls are stitched together.
        """
        period_hours = _period_to_hours(period)
        interval_minutes = INTERVAL_TO_KRAKEN_MINUTES.get(interval, 5)
        max_candles_per_call = 720
        max_hours_per_call = (max_candles_per_call * interval_minutes) / 60.0

        if period_hours <= max_hours_per_call:
            return self.get_ohlcv(pair, interval=interval, since_hours=period_hours)

        # Need multiple calls — stitch together by paginating with `since`
        all_frames: list[pd.DataFrame] = []
        now = datetime.now(tz=UTC)
        start = now - timedelta(hours=period_hours)
        current_since = int(start.timestamp())

        while True:
            df = self.get_ohlcv(pair, interval=interval, since_timestamp=current_since)
            if df.empty:
                break

            all_frames.append(df)

            # Move `since` forward to just after the last returned candle
            assert isinstance(df.index, pd.DatetimeIndex)
            last_idx_ts: pd.Timestamp = df.index[-1]  # type: ignore[assignment]
            last_ts = int(last_idx_ts.timestamp()) + (interval_minutes * 60)
            if last_ts >= int(now.timestamp()):
                break
            if last_ts <= current_since:
                # No progress — avoid infinite loop
                break
            current_since = last_ts

            # Small delay to respect rate limits
            time.sleep(self._PUBLIC_RATE_LIMIT)

        if not all_frames:
            return pd.DataFrame()

        combined = pd.DataFrame(pd.concat(all_frames))
        # Drop duplicate timestamps (overlapping edges)
        combined = pd.DataFrame(combined[~combined.index.duplicated(keep="last")])
        combined = combined.sort_index()
        return combined

    # ── Private endpoints ─────────────────────────────────────────────────

    def get_balance(self) -> dict[str, str]:
        """Return account balances (requires auth).

        Returns:
            Dict mapping asset code → balance string (e.g. {"XXBT": "0.5000"}).
        """
        return self._private_post("Balance")

    def get_trade_balance(self, asset: str = "ZUSD") -> dict[str, Any]:
        """Return trade balance / margin info (requires auth)."""
        return self._private_post("TradeBalance", {"asset": asset})

    def get_open_orders(self) -> dict[str, Any]:
        """Return open orders (requires auth)."""
        return self._private_post("OpenOrders")

    def get_trade_history(self, start: int | None = None, end: int | None = None) -> dict[str, Any]:
        """Return trade history (requires auth)."""
        data: dict[str, Any] = {}
        if start is not None:
            data["start"] = start
        if end is not None:
            data["end"] = end
        return self._private_post("TradesHistory", data)

    # ── Health check ──────────────────────────────────────────────────────

    def health(self) -> dict[str, Any]:
        """Return a health-check dict suitable for the dashboard."""
        result: dict[str, Any] = {
            "available": self.is_available,
            "authenticated": self.has_auth,
            "connected": False,
            "server_time": None,
            "error": None,
        }
        try:
            ts = self.get_server_time()
            result["connected"] = True
            result["server_time"] = ts.get("rfc1123")
        except Exception as exc:
            result["error"] = str(exc)
        return result


# ═══════════════════════════════════════════════════════════════════════════
# Helper: period string → hours
# ═══════════════════════════════════════════════════════════════════════════

_PERIOD_HOURS: dict[str, float] = {
    "1h": 1,
    "4h": 4,
    "12h": 12,
    "1d": 24,
    "2d": 48,
    "3d": 72,
    "5d": 120,
    "7d": 168,
    "10d": 240,
    "15d": 360,
    "1mo": 720,
    "3mo": 2160,
    "6mo": 4320,
    "1y": 8760,
}


def _period_to_hours(period: str) -> float:
    """Convert a period string like '5d' or '1mo' to hours."""
    if period in _PERIOD_HOURS:
        return _PERIOD_HOURS[period]

    # Try to parse "<N>d" or "<N>h" format
    import re

    m = re.match(r"^(\d+)d$", period)
    if m:
        return float(m.group(1)) * 24

    m = re.match(r"^(\d+)h$", period)
    if m:
        return float(m.group(1))

    # Default to 5 days
    logger.warning("Unknown period '%s' — defaulting to 5d (120h)", period)
    return 120.0


# ═══════════════════════════════════════════════════════════════════════════
# WebSocket v2 Feed Manager
# ═══════════════════════════════════════════════════════════════════════════


class KrakenFeedManager:
    """WebSocket v2 feed for real-time Kraken market data.

    Streams OHLC candles and/or trades from ``wss://ws.kraken.com/v2``
    and pushes them into the cache layer and latest-data dicts.

    Wire protocol (WebSocket v2):
      1. Connect to wss://ws.kraken.com/v2
      2. Receive ``{"channel":"status","type":"update",...}``
      3. Send subscribe: ``{"method":"subscribe","params":{"channel":"ohlc","symbol":["XBT/USD"],"interval":1}}``
      4. Receive snapshots + updates as JSON messages

    Usage::

        feed = KrakenFeedManager(pairs=["XBT/USD", "ETH/USD"])
        feed.start()       # connects + subscribes in background thread
        feed.is_connected  # check status
        feed.latest_bars   # dict of latest bars by internal ticker
        await feed.stop()  # clean shutdown
    """

    def __init__(
        self,
        pairs: list[str] | None = None,
        ohlc_interval: int = 1,  # minutes (1, 5, 15, etc.)
        subscribe_trades: bool = True,
        subscribe_ohlc: bool = True,
        subscribe_spread: bool = True,
    ) -> None:
        """Initialize the feed manager.

        Args:
            pairs: List of WebSocket v2 pair names (e.g. ["XBT/USD", "ETH/USD"]).
                   If None, subscribes to all pairs in KRAKEN_PAIRS.
            ohlc_interval: OHLC candle interval in minutes.
            subscribe_trades: Whether to subscribe to the trade channel.
            subscribe_ohlc: Whether to subscribe to the OHLC channel.
            subscribe_spread: Whether to subscribe to the spread (L1) channel.
        """
        if pairs is None:
            pairs = [p["ws_pair"] for p in KRAKEN_PAIRS.values()]

        self._pairs = pairs
        self._ohlc_interval = ohlc_interval
        self._subscribe_trades = subscribe_trades
        self._subscribe_ohlc = subscribe_ohlc
        self._subscribe_spread = subscribe_spread

        self._ws: Any = None
        self._thread: threading.Thread | None = None
        self._running = False
        self._connected = False
        self._lock = threading.Lock()

        # Latest data storage
        self._latest_bars: dict[str, dict[str, Any]] = {}  # internal_ticker → bar dict
        self._latest_trades: dict[str, dict[str, Any]] = {}  # internal_ticker → trade dict

        # Pending subscribe/unsubscribe
        self._pending_subscribe: list[str] = []
        self._pending_unsubscribe: list[str] = []

        # Tick aggregation state for building 1m bars from trades
        self._tick_agg: dict[str, dict] = {}

        # Latest L1 (spread) data per internal ticker
        self._latest_l1: dict[str, dict[str, Any]] = {}

        # Callbacks
        self._on_bar_callbacks: list = []
        self._on_trade_callbacks: list = []

        # Stats
        self.msg_count = 0
        self.bar_count = 0
        self.trade_count = 0
        self.errors: list[str] = []
        self.started_at: float | None = None
        self._last_msg_at: float = 0.0
        self._reconnect_count = 0

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def latest_bars(self) -> dict[str, dict[str, Any]]:
        with self._lock:
            return dict(self._latest_bars)

    @property
    def latest_trades(self) -> dict[str, dict[str, Any]]:
        with self._lock:
            return dict(self._latest_trades)

    def on_bar(self, callback) -> None:
        """Register a callback for new OHLC bars: callback(internal_ticker, bar_dict)."""
        self._on_bar_callbacks.append(callback)

    def on_trade(self, callback) -> None:
        """Register a callback for new trades: callback(internal_ticker, trade_dict)."""
        self._on_trade_callbacks.append(callback)

    # ── Lifecycle ─────────────────────────────────────────────────────────

    def start(self) -> None:
        """Start the WebSocket connection in a background daemon thread."""
        if self._running:
            logger.warning("Kraken feed already running")
            return

        self._running = True
        self.started_at = time.time()
        self._thread = threading.Thread(
            target=self._run_loop,
            name="kraken-ws-feed",
            daemon=True,
        )
        self._thread.start()
        logger.info(
            "Kraken WebSocket feed started: %d pairs, ohlc=%dm, trades=%s",
            len(self._pairs),
            self._ohlc_interval,
            self._subscribe_trades,
        )

    async def stop(self) -> None:
        """Stop the WebSocket connection and background thread."""
        self._running = False
        if self._ws is not None:
            with contextlib.suppress(Exception):
                await self._ws.close()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=5)
        self._connected = False
        logger.info("Kraken WebSocket feed stopped")

    def stop_sync(self) -> None:
        """Synchronous stop — for use from non-async contexts."""
        self._running = False
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=5)
        self._connected = False

    def add_pairs(self, pairs: list[str]) -> None:
        """Dynamically subscribe to additional pairs."""
        with self._lock:
            for p in pairs:
                if p not in self._pairs:
                    self._pairs.append(p)
                    self._pending_subscribe.append(p)

    def remove_pairs(self, pairs: list[str]) -> None:
        """Dynamically unsubscribe from pairs."""
        with self._lock:
            for p in pairs:
                if p in self._pairs:
                    self._pairs.remove(p)
                    self._pending_unsubscribe.append(p)

    def get_status(self) -> dict[str, Any]:
        """Return feed status dict for dashboard / health checks."""
        return {
            "running": self._running,
            "connected": self._connected,
            "pairs": list(self._pairs),
            "pair_count": len(self._pairs),
            "ohlc_interval": self._ohlc_interval,
            "subscribe_trades": self._subscribe_trades,
            "msg_count": self.msg_count,
            "bar_count": self.bar_count,
            "trade_count": self.trade_count,
            "error_count": len(self.errors),
            "last_errors": self.errors[-5:] if self.errors else [],
            "started_at": datetime.fromtimestamp(self.started_at, tz=_EST).isoformat() if self.started_at else None,
            "last_msg_at": datetime.fromtimestamp(self._last_msg_at, tz=_EST).isoformat()
            if self._last_msg_at
            else None,
            "reconnect_count": self._reconnect_count,
            "uptime_seconds": round(time.time() - self.started_at, 1) if self.started_at else 0,
        }

    # ── Internal: background thread event loop ────────────────────────────

    def _run_loop(self) -> None:
        """Entry point for the background thread — runs the async event loop."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._async_ws_loop())
        except Exception as exc:
            logger.error("Kraken WS loop crashed: %s", exc, exc_info=True)
            self.errors.append(f"loop_crash: {exc}")
        finally:
            loop.close()
            self._connected = False

    async def _async_ws_loop(self) -> None:
        """Main async WebSocket loop with reconnect logic."""
        import websockets

        backoff = 1.0
        max_backoff = 60.0

        while self._running:
            try:
                logger.info("Connecting to Kraken WebSocket v2: %s", KRAKEN_WS_URL)

                async with websockets.connect(
                    KRAKEN_WS_URL,
                    ping_interval=30,
                    ping_timeout=10,
                    close_timeout=5,
                    max_size=2**20,  # 1 MB max message size
                ) as ws:
                    self._ws = ws
                    self._connected = True
                    backoff = 1.0
                    logger.info("Connected to Kraken WebSocket v2")

                    # Wait for initial status message
                    try:
                        init_msg = await asyncio.wait_for(ws.recv(), timeout=10)
                        init_data = json.loads(init_msg)
                        logger.debug("Kraken WS init: %s", init_data)
                    except TimeoutError:
                        logger.warning("No init message from Kraken WS — proceeding anyway")

                    # Subscribe to channels
                    await self._send_subscriptions(ws)

                    # Message loop
                    async for raw_msg in ws:
                        if not self._running:
                            break

                        self.msg_count += 1
                        self._last_msg_at = time.time()

                        try:
                            msg = json.loads(raw_msg)
                            self._handle_message(msg)
                        except json.JSONDecodeError:
                            logger.debug("Non-JSON message from Kraken WS: %s", raw_msg[:100])
                        except Exception as exc:
                            logger.debug("Error handling Kraken WS message: %s", exc)

                        # Process pending subscribe/unsubscribe
                        await self._flush_pending(ws)

            except asyncio.CancelledError:
                logger.info("Kraken WS cancelled")
                break
            except Exception as exc:
                self._connected = False
                self._reconnect_count += 1
                err_msg = f"reconnect_{self._reconnect_count}: {exc}"
                self.errors.append(err_msg)
                if len(self.errors) > 100:
                    self.errors = self.errors[-50:]

                if self._running:
                    logger.warning(
                        "Kraken WS disconnected (%s) — reconnecting in %.0fs",
                        exc,
                        backoff,
                    )
                    await asyncio.sleep(backoff)
                    backoff = min(backoff * 1.5, max_backoff)

        self._connected = False

    async def _send_subscriptions(self, ws) -> None:
        """Send subscribe messages for all configured channels."""
        if self._subscribe_ohlc and self._pairs:
            sub_msg = {
                "method": "subscribe",
                "params": {
                    "channel": "ohlc",
                    "symbol": list(self._pairs),
                    "interval": self._ohlc_interval,
                },
            }
            await ws.send(json.dumps(sub_msg))
            logger.info(
                "Subscribed to Kraken OHLC (%dm): %s",
                self._ohlc_interval,
                ", ".join(self._pairs[:5]) + ("..." if len(self._pairs) > 5 else ""),
            )

        if self._subscribe_trades and self._pairs:
            sub_msg = {
                "method": "subscribe",
                "params": {
                    "channel": "trade",
                    "symbol": list(self._pairs),
                },
            }
            await ws.send(json.dumps(sub_msg))
            logger.info("Subscribed to Kraken trades: %d pairs", len(self._pairs))

        if self._subscribe_spread and self._pairs:
            sub_msg = {
                "method": "subscribe",
                "params": {
                    "channel": "spread",
                    "symbol": list(self._pairs),
                },
            }
            await ws.send(json.dumps(sub_msg))
            logger.info("Subscribed to Kraken spread: %d pairs", len(self._pairs))

    async def _flush_pending(self, ws) -> None:
        """Send any pending subscribe / unsubscribe requests."""
        with self._lock:
            to_sub = list(self._pending_subscribe)
            to_unsub = list(self._pending_unsubscribe)
            self._pending_subscribe.clear()
            self._pending_unsubscribe.clear()

        if to_sub:
            if self._subscribe_ohlc:
                msg = {
                    "method": "subscribe",
                    "params": {
                        "channel": "ohlc",
                        "symbol": to_sub,
                        "interval": self._ohlc_interval,
                    },
                }
                await ws.send(json.dumps(msg))
            if self._subscribe_trades:
                msg = {
                    "method": "subscribe",
                    "params": {
                        "channel": "trade",
                        "symbol": to_sub,
                    },
                }
                await ws.send(json.dumps(msg))
            if self._subscribe_spread:
                msg = {
                    "method": "subscribe",
                    "params": {
                        "channel": "spread",
                        "symbol": to_sub,
                    },
                }
                await ws.send(json.dumps(msg))
            logger.info("Dynamically subscribed to %d new pairs", len(to_sub))

        if to_unsub:
            if self._subscribe_ohlc:
                msg = {
                    "method": "unsubscribe",
                    "params": {
                        "channel": "ohlc",
                        "symbol": to_unsub,
                        "interval": self._ohlc_interval,
                    },
                }
                await ws.send(json.dumps(msg))
            if self._subscribe_trades:
                msg = {
                    "method": "unsubscribe",
                    "params": {
                        "channel": "trade",
                        "symbol": to_unsub,
                    },
                }
                await ws.send(json.dumps(msg))
            if self._subscribe_spread:
                msg = {
                    "method": "unsubscribe",
                    "params": {
                        "channel": "spread",
                        "symbol": to_unsub,
                    },
                }
                await ws.send(json.dumps(msg))
            logger.info("Dynamically unsubscribed from %d pairs", len(to_unsub))

    # ── Message handling ──────────────────────────────────────────────────

    def _handle_message(self, msg: dict[str, Any] | list) -> None:
        """Route a parsed WebSocket v2 message to the appropriate handler."""
        # v2 messages are dicts with "channel", "type", "data" keys
        if isinstance(msg, list):
            # Shouldn't happen in v2, but handle gracefully
            return

        channel = msg.get("channel", "")
        msg_type = msg.get("type", "")

        # Skip heartbeats, status, subscription confirmations
        if channel in ("heartbeat", "status"):
            return
        if msg_type in ("subscribe", "unsubscribe"):
            logger.debug("Kraken WS subscription ack: %s", msg)
            return

        data = msg.get("data", [])
        if not data:
            return

        if channel == "ohlc":
            self._handle_ohlc(data)
        elif channel == "trade":
            self._handle_trade(data)
        elif channel == "spread":
            self._handle_spread(data)

    def _handle_ohlc(self, data: list[dict[str, Any]]) -> None:
        """Process OHLC candle data from WebSocket v2.

        v2 OHLC message data is a list of dicts, each containing:
          symbol, open, high, low, close, volume, vwap, trades,
          interval_begin, timestamp
        """
        for candle in data:
            ws_symbol = candle.get("symbol", "")

            # Map WS pair → internal ticker
            name = WS_PAIR_TO_NAME.get(ws_symbol)
            if name is None:
                continue
            internal = KRAKEN_PAIRS[name]["internal_ticker"]

            bar = {
                "open": float(candle.get("open", 0)),
                "high": float(candle.get("high", 0)),
                "low": float(candle.get("low", 0)),
                "close": float(candle.get("close", 0)),
                "volume": float(candle.get("volume", 0)),
                "vwap": float(candle.get("vwap", 0)),
                "trades": int(candle.get("trades", 0)),
                "interval_begin": candle.get("interval_begin", ""),
                "timestamp": candle.get("timestamp", ""),
                "symbol": ws_symbol,
                "internal_ticker": internal,
            }

            with self._lock:
                self._latest_bars[internal] = bar
            self.bar_count += 1

            # Fire callbacks
            for cb in self._on_bar_callbacks:
                with contextlib.suppress(Exception):
                    cb(internal, bar)

            # Push to Redis cache
            self._push_bar_to_cache(internal, bar)

    def _handle_trade(self, data: list[dict[str, Any]]) -> None:
        """Process trade data from WebSocket v2.

        v2 trade message data is a list of dicts, each containing:
          symbol, price, qty, side, ord_type, timestamp
        """
        for trade in data:
            ws_symbol = trade.get("symbol", "")

            name = WS_PAIR_TO_NAME.get(ws_symbol)
            if name is None:
                continue
            internal = KRAKEN_PAIRS[name]["internal_ticker"]

            trade_data = {
                "price": float(trade.get("price", 0)),
                "qty": float(trade.get("qty", 0)),
                "side": trade.get("side", ""),
                "ord_type": trade.get("ord_type", ""),
                "timestamp": trade.get("timestamp", ""),
                "symbol": ws_symbol,
                "internal_ticker": internal,
            }

            with self._lock:
                self._latest_trades[internal] = trade_data
            self.trade_count += 1

            # Fire callbacks
            for cb in self._on_trade_callbacks:
                with contextlib.suppress(Exception):
                    cb(internal, trade_data)

            # Publish raw tick to Redis sorted set
            try:
                from lib.core.cache import REDIS_AVAILABLE, _r

                if REDIS_AVAILABLE and _r is not None:
                    tick_key = f"kraken:ticks:{internal}"
                    ts_str = trade_data.get("timestamp", "")
                    try:
                        tick_epoch = datetime.fromisoformat(ts_str.replace("Z", "+00:00")).timestamp()
                    except Exception:
                        tick_epoch = time.time()

                    _r.zadd(tick_key, {json.dumps(trade_data): tick_epoch})

                    # Trim entries older than 5 minutes
                    cutoff = time.time() - 300
                    _r.zremrangebyscore(tick_key, "-inf", cutoff)

                    # Publish real-time event
                    _r.publish(
                        "futures:events",
                        json.dumps(
                            {
                                "event": "kraken_tick",
                                "ticker": internal,
                                "price": trade_data["price"],
                                "qty": trade_data["qty"],
                                "side": trade_data["side"],
                            }
                        ),
                    )
            except Exception:
                pass  # Non-fatal

            # Aggregate ticks into 1m bars
            self._aggregate_tick_bars(internal, trade_data)

    def _aggregate_tick_bars(self, internal_ticker: str, trade_data: dict[str, Any]) -> None:
        """Aggregate incoming ticks into 1-minute OHLCV bars."""
        try:
            ts_str = trade_data.get("timestamp", "")
            try:
                tick_dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
            except Exception:
                tick_dt = datetime.now(UTC)

            # Truncate to the current minute boundary
            minute_start = tick_dt.replace(second=0, microsecond=0)

            price = trade_data["price"]
            qty = trade_data["qty"]

            with self._lock:
                agg = self._tick_agg.get(internal_ticker)

                if agg is not None and agg["minute_start"] == minute_start:
                    # Same minute window — update OHLCV
                    agg["high"] = max(agg["high"], price)
                    agg["low"] = min(agg["low"], price)
                    agg["close"] = price
                    agg["volume"] += qty
                    agg["trade_count"] += 1
                else:
                    # New minute — emit the completed bar (if any), then start fresh
                    completed_bar = agg  # may be None on first tick
                    self._tick_agg[internal_ticker] = {
                        "open": price,
                        "high": price,
                        "low": price,
                        "close": price,
                        "volume": qty,
                        "trade_count": 1,
                        "minute_start": minute_start,
                    }

            # Emit completed bar outside the lock to avoid holding it during I/O
            if agg is not None and agg["minute_start"] != minute_start:
                ws_symbol = INTERNAL_TO_WS.get(internal_ticker, "")
                completed = {
                    "open": agg["open"],
                    "high": agg["high"],
                    "low": agg["low"],
                    "close": agg["close"],
                    "volume": agg["volume"],
                    "trade_count": agg["trade_count"],
                    "interval_begin": agg["minute_start"].isoformat(),
                    "timestamp": datetime.now(UTC).isoformat(),
                    "symbol": ws_symbol,
                    "internal_ticker": internal_ticker,
                }

                # Fire bar callbacks
                for cb in self._on_bar_callbacks:
                    with contextlib.suppress(Exception):
                        cb(internal_ticker, completed)

                # Push aggregated bar to Redis
                try:
                    from lib.core.cache import REDIS_AVAILABLE, _r

                    if REDIS_AVAILABLE and _r is not None:
                        agg_key = f"kraken:agg:{internal_ticker}"
                        _r.setex(agg_key, 120, json.dumps(completed))
                except Exception:
                    pass  # Non-fatal
        except Exception:
            pass  # Non-fatal — tick aggregation failure doesn't break anything

    def _handle_spread(self, data: list[dict[str, Any]]) -> None:
        """Process spread (L1 best bid/ask) data from WebSocket v2.

        v2 spread message data is a list of dicts, each containing:
          symbol, bid, bid_qty, ask, ask_qty, timestamp
        """
        for item in data:
            ws_symbol = item.get("symbol", "")

            name = WS_PAIR_TO_NAME.get(ws_symbol)
            if name is None:
                continue
            internal = KRAKEN_PAIRS[name]["internal_ticker"]

            l1_data = {
                "bid": float(item.get("bid", 0)),
                "bid_qty": float(item.get("bid_qty", 0)),
                "ask": float(item.get("ask", 0)),
                "ask_qty": float(item.get("ask_qty", 0)),
                "timestamp": item.get("timestamp", ""),
                "symbol": ws_symbol,
                "internal_ticker": internal,
            }

            with self._lock:
                self._latest_l1[internal] = l1_data

            # Publish L1 to Redis
            try:
                from lib.core.cache import REDIS_AVAILABLE, _r

                if REDIS_AVAILABLE and _r is not None:
                    l1_key = f"kraken:l1:{internal}"
                    _r.setex(l1_key, 30, json.dumps(l1_data))

                    _r.publish(
                        "futures:events",
                        json.dumps(
                            {
                                "event": "kraken_l1",
                                "ticker": internal,
                                "bid": l1_data["bid"],
                                "ask": l1_data["ask"],
                            }
                        ),
                    )
            except Exception:
                pass  # Non-fatal

    def _push_bar_to_cache(self, internal_ticker: str, bar: dict[str, Any]) -> None:
        """Push a live bar update into Redis for the engine / dashboard."""
        try:
            from lib.core.cache import REDIS_AVAILABLE, _r

            if not REDIS_AVAILABLE or _r is None:
                return

            cache_key = f"kraken:live:{internal_ticker}"
            _r.setex(cache_key, 120, json.dumps(bar))

            # Also publish for SSE consumers
            _r.publish(
                "futures:events",
                json.dumps(
                    {
                        "event": "kraken_bar",
                        "ticker": internal_ticker,
                        "close": bar["close"],
                        "volume": bar["volume"],
                    }
                ),
            )
        except Exception:
            pass  # Non-fatal — cache miss doesn't break anything


# ═══════════════════════════════════════════════════════════════════════════
# Module-level singleton & convenience functions
# ═══════════════════════════════════════════════════════════════════════════

_provider: KrakenDataProvider | None = None
_provider_checked = False
_feed: KrakenFeedManager | None = None


def get_kraken_provider() -> KrakenDataProvider:
    """Return the module-level KrakenDataProvider singleton.

    Always available — the provider works without auth for public
    market data (OHLCV, tickers, asset pairs).  Private endpoints
    (balance, orders) require KRAKEN_API_KEY + KRAKEN_API_SECRET.
    """
    global _provider, _provider_checked
    if not _provider_checked:
        _provider = KrakenDataProvider()
        _provider_checked = True
    return _provider  # type: ignore[return-value]


def reset_provider() -> None:
    """Reset the singleton (for testing)."""
    global _provider, _provider_checked
    _provider = None
    _provider_checked = False


def get_kraken_feed() -> KrakenFeedManager | None:
    """Return the module-level KrakenFeedManager, or None if not started."""
    return _feed


def start_kraken_feed(
    pairs: list[str] | None = None,
    ohlc_interval: int = 1,
    subscribe_trades: bool = True,
) -> KrakenFeedManager:
    """Create and start a KrakenFeedManager singleton.

    If one is already running, returns the existing instance.
    """
    global _feed
    if _feed is not None and _feed.is_running:
        return _feed

    _feed = KrakenFeedManager(
        pairs=pairs,
        ohlc_interval=ohlc_interval,
        subscribe_trades=subscribe_trades,
    )
    _feed.start()
    return _feed


async def stop_kraken_feed() -> None:
    """Stop the global Kraken feed if running."""
    global _feed
    if _feed is not None:
        await _feed.stop()
        _feed = None


def stop_kraken_feed_sync() -> None:
    """Synchronous version of stop_kraken_feed."""
    global _feed
    if _feed is not None:
        _feed.stop_sync()
        _feed = None


# ── Convenience wrappers ──────────────────────────────────────────────────


def get_kraken_ohlcv(
    internal_ticker: str,
    interval: str = "5m",
    period: str = "5d",
) -> pd.DataFrame:
    """Fetch OHLCV for a KRAKEN:* ticker via the REST API.

    This is the main entry point used by cache.py when it detects a
    Kraken ticker.
    """
    provider = get_kraken_provider()
    if not provider.is_available:
        return pd.DataFrame()

    try:
        return provider.get_ohlcv_period(internal_ticker, interval=interval, period=period)
    except Exception as exc:
        logger.error("get_kraken_ohlcv(%s) failed: %s", internal_ticker, exc)
        return pd.DataFrame()


def get_kraken_daily(internal_ticker: str, period: str = "10d") -> pd.DataFrame:
    """Fetch daily bars for a KRAKEN:* ticker."""
    return get_kraken_ohlcv(internal_ticker, interval="1d", period=period)


def get_kraken_ticker(internal_ticker: str) -> dict[str, Any]:
    """Get current ticker snapshot for a KRAKEN:* ticker."""
    provider = get_kraken_provider()
    if not provider.is_available:
        return {}
    try:
        return provider.get_ticker_by_internal(internal_ticker)
    except Exception as exc:
        logger.error("get_kraken_ticker(%s) failed: %s", internal_ticker, exc)
        return {}


def get_kraken_snapshot(internal_ticker: str) -> dict[str, Any]:
    """Get a snapshot dict compatible with the Massive snapshot format.

    Returns a dict with: last_price, bid, ask, volume_24h, high_24h,
    low_24h, open_24h, vwap_24h, trades_24h, change_pct.
    """
    raw = get_kraken_ticker(internal_ticker)
    if not raw:
        return {}

    try:
        last = float(raw.get("c", [0])[0]) if isinstance(raw.get("c"), list) else 0.0
        bid = float(raw.get("b", [0])[0]) if isinstance(raw.get("b"), list) else 0.0
        ask = float(raw.get("a", [0])[0]) if isinstance(raw.get("a"), list) else 0.0
        volume = float(raw.get("v", [0, 0])[1]) if isinstance(raw.get("v"), list) else 0.0
        high = float(raw.get("h", [0, 0])[1]) if isinstance(raw.get("h"), list) else 0.0
        low = float(raw.get("l", [0, 0])[1]) if isinstance(raw.get("l"), list) else 0.0
        open_price = float(raw.get("o", 0))
        vwap = float(raw.get("p", [0, 0])[1]) if isinstance(raw.get("p"), list) else 0.0
        trades = int(raw.get("t", [0, 0])[1]) if isinstance(raw.get("t"), list) else 0

        change_pct = 0.0
        if open_price > 0:
            change_pct = round((last - open_price) / open_price * 100, 3)

        return {
            "ticker": internal_ticker,
            "last_price": last,
            "bid": bid,
            "ask": ask,
            "spread": round(ask - bid, 6) if ask and bid else 0.0,
            "volume_24h": volume,
            "high_24h": high,
            "low_24h": low,
            "open_24h": open_price,
            "vwap_24h": vwap,
            "trades_24h": trades,
            "change_pct": change_pct,
            "source": "kraken",
            "timestamp": datetime.now(tz=_EST).isoformat(),
        }
    except Exception as exc:
        logger.error("get_kraken_snapshot parse error: %s", exc)
        return {}
