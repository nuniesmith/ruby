"""
src/lib/integrations/news_client.py
─────────────────────────────────────
News data collectors for Finnhub and Alpha Vantage.

Provides:
  - FinnhubClient   — market news + company news via proxy ETF tickers
  - AlphaVantageClient — AI-scored news sentiment + commodity prices
  - fetch_all_news() — convenience wrapper used by the scheduler

Proxy ticker map
────────────────
CME micro futures don't have news directly — we map them to liquid ETF
proxies that Finnhub and Alpha Vantage understand:

    MGC  → GLD   (gold)
    SIL  → SLV   (silver)
    MHG  → CPER  (copper)
    MCL  → USO   (crude oil)
    MNG  → UNG   (natural gas)
    MES  → SPY   (S&P 500)
    MNQ  → QQQ   (Nasdaq)
    M2K  → IWM   (Russell 2000)
    MYM  → DIA   (Dow Jones)
    M6E  → FXE   (EUR/USD)
    M6B  → FXB   (GBP/USD)
    MJY  → FXY   (JPY/USD)
    MZN  → TLT   (10Y Treasury)
    MZB  → TLT   (30Y Bond — same proxy)
    MBT  → BTC-USD (Bitcoin via Finnhub / CRYPTO:BTC via AV)
    MET  → ETH-USD (Ether via Finnhub / CRYPTO:ETH via AV)
    KRAKEN:XBTUSD → BTC-USD
    KRAKEN:ETHUSD → ETH-USD
    KRAKEN:SOLUSD → SOL-USD
    KRAKEN:LINKUSD → LINK-USD
    KRAKEN:ADAUSD  → ADA-USD
    KRAKEN:XRPUSD  → XRP-USD

Rate limits (free tiers)
────────────────────────
  Finnhub:        60 req/min  (generous — safe at 1 req/s)
  Alpha Vantage:  25 req/day  (very tight — cache aggressively, fetch ONCE/day)

Usage
─────
    from lib.integrations.news_client import FinnhubClient, AlphaVantageClient

    fh = FinnhubClient(api_key=os.getenv("FINNHUB_API_KEY"))
    articles = fh.fetch_company_news("MGC", days_back=2)

    av = AlphaVantageClient(api_key=os.getenv("ALPHA_VANTAGE_API_KEY"))
    sentiment = av.fetch_news_sentiment(["MES", "MCL", "MGC"])
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

import requests

logger = logging.getLogger("news_client")

# ---------------------------------------------------------------------------
# Proxy ticker maps
# ---------------------------------------------------------------------------

#: Maps internal futures ticker → Finnhub equity/crypto symbol
FINNHUB_PROXY: dict[str, str] = {
    "MGC=F": "GLD",
    "MGC": "GLD",
    "SIL=F": "SLV",
    "SIL": "SLV",
    "MHG=F": "CPER",
    "MHG": "CPER",
    "MCL=F": "USO",
    "MCL": "USO",
    "MNG=F": "UNG",
    "MNG": "UNG",
    "MES=F": "SPY",
    "MES": "SPY",
    "MNQ=F": "QQQ",
    "MNQ": "QQQ",
    "M2K=F": "IWM",
    "M2K": "IWM",
    "MYM=F": "DIA",
    "MYM": "DIA",
    "M6E=F": "FXE",
    "M6E": "FXE",
    "M6B=F": "FXB",
    "M6B": "FXB",
    "MJY=F": "FXY",
    "MJY": "FXY",
    "MZN=F": "TLT",
    "MZN": "TLT",
    "MZB=F": "TLT",
    "MZB": "TLT",
    "MBT=F": "BTC-USD",
    "MBT": "BTC-USD",
    "MET=F": "ETH-USD",
    "MET": "ETH-USD",
    "KRAKEN:XBTUSD": "BTC-USD",
    "KRAKEN:ETHUSD": "ETH-USD",
    "KRAKEN:SOLUSD": "SOL-USD",
    "KRAKEN:LINKUSD": "LINK-USD",
    "KRAKEN:AVAXUSD": "AVAX-USD",
    "KRAKEN:DOTUSD": "DOT-USD",
    "KRAKEN:ADAUSD": "ADA-USD",
    "KRAKEN:POLUSD": "POL-USD",
    "KRAKEN:XRPUSD": "XRP-USD",
}

#: Maps internal futures ticker → Alpha Vantage ticker + topic hint
ALPHA_VANTAGE_PROXY: dict[str, dict[str, str]] = {
    "MGC=F": {"ticker": "GLD", "topic": "financial_markets"},
    "MGC": {"ticker": "GLD", "topic": "financial_markets"},
    "SIL=F": {"ticker": "SLV", "topic": "financial_markets"},
    "SIL": {"ticker": "SLV", "topic": "financial_markets"},
    "MHG=F": {"ticker": "CPER", "topic": "financial_markets"},
    "MHG": {"ticker": "CPER", "topic": "financial_markets"},
    "MCL=F": {"ticker": "USO", "topic": "energy_transportation"},
    "MCL": {"ticker": "USO", "topic": "energy_transportation"},
    "MNG=F": {"ticker": "UNG", "topic": "energy_transportation"},
    "MNG": {"ticker": "UNG", "topic": "energy_transportation"},
    "MES=F": {"ticker": "SPY", "topic": "financial_markets"},
    "MES": {"ticker": "SPY", "topic": "financial_markets"},
    "MNQ=F": {"ticker": "QQQ", "topic": "financial_markets"},
    "MNQ": {"ticker": "QQQ", "topic": "financial_markets"},
    "M2K=F": {"ticker": "IWM", "topic": "financial_markets"},
    "M2K": {"ticker": "IWM", "topic": "financial_markets"},
    "MYM=F": {"ticker": "DIA", "topic": "financial_markets"},
    "MYM": {"ticker": "DIA", "topic": "financial_markets"},
    "M6E=F": {"ticker": "FXE", "topic": "financial_markets"},
    "M6E": {"ticker": "FXE", "topic": "financial_markets"},
    "M6B=F": {"ticker": "FXB", "topic": "financial_markets"},
    "M6B": {"ticker": "FXB", "topic": "financial_markets"},
    "MJY=F": {"ticker": "FXY", "topic": "financial_markets"},
    "MJY": {"ticker": "FXY", "topic": "financial_markets"},
    "MZN=F": {"ticker": "TLT", "topic": "financial_markets"},
    "MZN": {"ticker": "TLT", "topic": "financial_markets"},
    "MZB=F": {"ticker": "TLT", "topic": "financial_markets"},
    "MZB": {"ticker": "TLT", "topic": "financial_markets"},
    "MBT=F": {"ticker": "CRYPTO:BTC", "topic": "blockchain"},
    "MBT": {"ticker": "CRYPTO:BTC", "topic": "blockchain"},
    "MET=F": {"ticker": "CRYPTO:ETH", "topic": "blockchain"},
    "MET": {"ticker": "CRYPTO:ETH", "topic": "blockchain"},
    "KRAKEN:XBTUSD": {"ticker": "CRYPTO:BTC", "topic": "blockchain"},
    "KRAKEN:ETHUSD": {"ticker": "CRYPTO:ETH", "topic": "blockchain"},
    "KRAKEN:SOLUSD": {"ticker": "CRYPTO:SOL", "topic": "blockchain"},
    "KRAKEN:LINKUSD": {"ticker": "CRYPTO:LINK", "topic": "blockchain"},
    "KRAKEN:AVAXUSD": {"ticker": "CRYPTO:AVAX", "topic": "blockchain"},
    "KRAKEN:DOTUSD": {"ticker": "CRYPTO:DOT", "topic": "blockchain"},
    "KRAKEN:ADAUSD": {"ticker": "CRYPTO:ADA", "topic": "blockchain"},
    "KRAKEN:XRPUSD": {"ticker": "CRYPTO:XRP", "topic": "blockchain"},
}

#: Asset display names — used in Grok prompts and dashboard labels
ASSET_DISPLAY: dict[str, str] = {
    "MGC": "Gold",
    "MGC=F": "Gold",
    "SIL": "Silver",
    "SIL=F": "Silver",
    "MHG": "Copper",
    "MHG=F": "Copper",
    "MCL": "Crude Oil",
    "MCL=F": "Crude Oil",
    "MNG": "Natural Gas",
    "MNG=F": "Natural Gas",
    "MES": "S&P 500",
    "MES=F": "S&P 500",
    "MNQ": "Nasdaq",
    "MNQ=F": "Nasdaq",
    "M2K": "Russell 2000",
    "M2K=F": "Russell 2000",
    "MYM": "Dow Jones",
    "MYM=F": "Dow Jones",
    "M6E": "Euro FX",
    "M6E=F": "Euro FX",
    "M6B": "British Pound",
    "M6B=F": "British Pound",
    "MJY": "Japanese Yen",
    "MJY=F": "Japanese Yen",
    "MZN": "10Y Treasury",
    "MZN=F": "10Y Treasury",
    "MZB": "30Y Bond",
    "MZB=F": "30Y Bond",
    "MBT": "Micro Bitcoin",
    "MBT=F": "Micro Bitcoin",
    "MET": "Micro Ether",
    "MET=F": "Micro Ether",
    "KRAKEN:XBTUSD": "Bitcoin",
    "KRAKEN:ETHUSD": "Ethereum",
    "KRAKEN:SOLUSD": "Solana",
    "KRAKEN:LINKUSD": "Chainlink",
    "KRAKEN:AVAXUSD": "Avalanche",
    "KRAKEN:DOTUSD": "Polkadot",
    "KRAKEN:ADAUSD": "Cardano",
    "KRAKEN:XRPUSD": "XRP",
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class NewsArticle:
    """Normalised news article from any source."""

    source: str  # "finnhub" | "alphavantage"
    ticker: str  # internal futures ticker (e.g. "MGC=F")
    proxy: str  # ETF proxy used for the fetch
    headline: str
    summary: str
    url: str
    published_at: datetime  # UTC
    # Alpha Vantage AI scores (None for Finnhub articles until scored)
    av_overall_score: float | None = None  # 0.0–1.0 (AV: Bearish→Bullish)
    av_ticker_score: float | None = None  # ticker-specific AV score
    av_label: str | None = None  # "Bearish" | "Neutral" | "Bullish"
    raw: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Finnhub client
# ---------------------------------------------------------------------------


class FinnhubClient:
    """Fetches market news and company news from Finnhub.

    Free tier: 60 API calls/minute.  We throttle to 1 req/s by default to
    stay well inside the limit even when fetching many symbols in sequence.

    Args:
        api_key:        ``FINNHUB_API_KEY`` env var value.
        min_delay_s:    Minimum seconds between API calls (default 1.1s).
        timeout_s:      HTTP request timeout in seconds.
    """

    BASE_URL = "https://finnhub.io/api/v1"

    def __init__(
        self,
        api_key: str,
        min_delay_s: float = 1.1,
        timeout_s: int = 15,
    ) -> None:
        self._key = api_key
        self._delay = min_delay_s
        self._timeout = timeout_s
        self._last_call: float = 0.0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_call
        if elapsed < self._delay:
            time.sleep(self._delay - elapsed)
        self._last_call = time.monotonic()

    def _get(self, endpoint: str, params: dict[str, Any]) -> Any:
        """Make a throttled GET request; return parsed JSON or None."""
        self._throttle()
        params["token"] = self._key
        url = f"{self.BASE_URL}/{endpoint}"
        try:
            resp = requests.get(url, params=params, timeout=self._timeout)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.Timeout:
            logger.warning("Finnhub timeout: %s", endpoint)
            return None
        except requests.exceptions.HTTPError as e:
            logger.warning("Finnhub HTTP error %s: %s", endpoint, e)
            return None
        except Exception as e:
            logger.error("Finnhub unexpected error %s: %s", endpoint, e)
            return None

    @staticmethod
    def _parse_ts(ts: int | None) -> datetime:
        """Convert Unix timestamp to UTC datetime (fallback: now)."""
        if ts:
            try:
                return datetime.fromtimestamp(ts, tz=UTC)
            except (OSError, OverflowError, ValueError):
                pass
        return datetime.now(tz=UTC)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch_general_news(
        self,
        category: str = "general",
        min_id: int = 0,
    ) -> list[NewsArticle]:
        """Fetch top market-moving headlines (not ticker-specific).

        Args:
            category:   "general" | "forex" | "crypto" | "merger"
            min_id:     Only return articles with id > min_id (dedup helper).

        Returns:
            List of :class:`NewsArticle` objects, newest first.
        """
        data = self._get("news", {"category": category, "minId": min_id})
        if not data or not isinstance(data, list):
            return []

        articles: list[NewsArticle] = []
        for item in data:
            headline = item.get("headline") or ""
            summary = item.get("summary") or item.get("headline") or ""
            if not headline:
                continue
            articles.append(
                NewsArticle(
                    source="finnhub",
                    ticker="MARKET",
                    proxy=category,
                    headline=headline,
                    summary=summary,
                    url=item.get("url") or "",
                    published_at=self._parse_ts(item.get("datetime")),
                    raw=item,
                )
            )
        return articles

    def fetch_company_news(
        self,
        futures_ticker: str,
        days_back: int = 2,
        max_articles: int = 10,
    ) -> list[NewsArticle]:
        """Fetch news for a futures ticker via its ETF proxy.

        Args:
            futures_ticker: Internal ticker, e.g. ``"MGC=F"`` or ``"MGC"``.
            days_back:      How many calendar days to look back (max 30 for free).
            max_articles:   Cap on articles returned per ticker.

        Returns:
            List of :class:`NewsArticle` (may be empty if no proxy mapping).
        """
        proxy = FINNHUB_PROXY.get(futures_ticker)
        if not proxy:
            logger.debug("Finnhub: no proxy mapping for %s", futures_ticker)
            return []

        now = datetime.now(tz=UTC)
        date_from = (now - timedelta(days=days_back)).strftime("%Y-%m-%d")
        date_to = now.strftime("%Y-%m-%d")

        data = self._get(
            "company-news",
            {"symbol": proxy, "from": date_from, "to": date_to},
        )
        if not data or not isinstance(data, list):
            return []

        articles: list[NewsArticle] = []
        for item in data[:max_articles]:
            headline = item.get("headline") or ""
            summary = item.get("summary") or headline
            if not headline:
                continue
            articles.append(
                NewsArticle(
                    source="finnhub",
                    ticker=futures_ticker,
                    proxy=proxy,
                    headline=headline,
                    summary=summary,
                    url=item.get("url") or "",
                    published_at=self._parse_ts(item.get("datetime")),
                    raw=item,
                )
            )
        logger.debug("Finnhub: %d articles for %s (proxy=%s)", len(articles), futures_ticker, proxy)
        return articles

    def fetch_batch(
        self,
        futures_tickers: list[str],
        days_back: int = 2,
        max_per_ticker: int = 5,
    ) -> list[NewsArticle]:
        """Fetch news for multiple futures tickers (throttled).

        Args:
            futures_tickers: List of internal tickers.
            days_back:       Look-back window in calendar days.
            max_per_ticker:  Max articles per ticker to cap total volume.

        Returns:
            Flat list of all articles, deduplicated by URL.
        """
        seen_urls: set[str] = set()
        all_articles: list[NewsArticle] = []

        for ticker in futures_tickers:
            articles = self.fetch_company_news(ticker, days_back=days_back, max_articles=max_per_ticker)
            for a in articles:
                if a.url and a.url in seen_urls:
                    continue
                if a.url:
                    seen_urls.add(a.url)
                all_articles.append(a)

        logger.info(
            "Finnhub batch: %d articles for %d tickers",
            len(all_articles),
            len(futures_tickers),
        )
        return all_articles

    def is_available(self) -> bool:
        """Quick connectivity check — returns True if the API key works."""
        if not self._key:
            return False
        data = self._get("news", {"category": "general", "minId": 0})
        return isinstance(data, list)


# ---------------------------------------------------------------------------
# Alpha Vantage client
# ---------------------------------------------------------------------------


class AlphaVantageClient:
    """Fetches AI-scored news sentiment from Alpha Vantage.

    Free tier: **25 API calls/day**.  This client is designed to make at most
    1–2 calls per day:
      - One ``NEWS_SENTIMENT`` call covering all tracked tickers + topics.
      - Optionally one ``COMMODITIES`` price call.

    The caller (scheduler) is responsible for respecting the daily cap —
    results should be cached in Redis with a 12–24h TTL.

    Args:
        api_key:    ``ALPHA_VANTAGE_API_KEY`` env var value.
        timeout_s:  HTTP request timeout in seconds.
    """

    BASE_URL = "https://www.alphavantage.co/query"

    # Topics recognised by Alpha Vantage NEWS_SENTIMENT
    VALID_TOPICS = frozenset(
        {
            "blockchain",
            "earnings",
            "ipo",
            "mergers_and_acquisitions",
            "financial_markets",
            "economy_fiscal",
            "economy_monetary",
            "economy_macro",
            "energy_transportation",
            "finance",
            "life_sciences",
            "manufacturing",
            "real_estate",
            "retail_wholesale",
            "technology",
        }
    )

    # Commodity codes for COMMODITIES endpoint
    COMMODITY_CODES = {
        "WTI": "WTI",  # West Texas Intermediate crude
        "BRENT": "BRENT",  # Brent crude
        "NATURAL_GAS": "NATURAL_GAS",
        "COPPER": "COPPER",
        "ALUMINUM": "ALUMINUM",
        "WHEAT": "WHEAT",
        "CORN": "CORN",
        "COTTON": "COTTON",
        "SUGAR": "SUGAR",
        "COFFEE": "COFFEE",
    }

    def __init__(self, api_key: str, timeout_s: int = 30) -> None:
        self._key = api_key
        self._timeout = timeout_s

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get(self, params: dict[str, Any]) -> dict[str, Any] | None:
        """Execute one Alpha Vantage REST call; return JSON dict or None."""
        params["apikey"] = self._key
        try:
            resp = requests.get(self.BASE_URL, params=params, timeout=self._timeout)
            resp.raise_for_status()
            data = resp.json()
            # AV returns {"Information": "..."} on rate-limit/error
            if "Information" in data or "Note" in data:
                msg = data.get("Information") or data.get("Note", "")
                logger.warning("Alpha Vantage rate-limit/info: %s", msg[:120])
                return None
            if "Error Message" in data:
                logger.warning("Alpha Vantage error: %s", data["Error Message"])
                return None
            return data
        except requests.exceptions.Timeout:
            logger.warning("Alpha Vantage timeout")
            return None
        except requests.exceptions.HTTPError as e:
            logger.warning("Alpha Vantage HTTP error: %s", e)
            return None
        except Exception as e:
            logger.error("Alpha Vantage unexpected error: %s", e)
            return None

    @staticmethod
    def _av_score_to_float(label: str, score: float | str | None) -> float:
        """Normalise AV sentiment score to [-1, +1].

        AV provides a 0–1 score where:
            Bearish   < 0.35
            Somewhat-Bearish  0.35–0.45
            Neutral   0.45–0.55
            Somewhat-Bullish  0.55–0.65
            Bullish   > 0.65

        We map this to [-1, +1] for consistency with VADER.
        """
        try:
            s = float(score) if score is not None else 0.5
        except (TypeError, ValueError):
            s = 0.5
        # Linear remap: [0, 1] → [-1, +1]
        return round((s - 0.5) * 2.0, 4)

    @staticmethod
    def _parse_av_datetime(dt_str: str | None) -> datetime:
        """Parse AV datetime string ``"20260310T073000"`` → UTC datetime."""
        if not dt_str:
            return datetime.now(tz=UTC)
        try:
            # Format: YYYYMMDDTHHmmss
            return datetime.strptime(dt_str, "%Y%m%dT%H%M%S").replace(tzinfo=UTC)
        except ValueError:
            try:
                return datetime.fromisoformat(dt_str).replace(tzinfo=UTC)
            except ValueError:
                return datetime.now(tz=UTC)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch_news_sentiment(
        self,
        futures_tickers: list[str],
        extra_topics: list[str] | None = None,
        time_from: datetime | None = None,
        limit: int = 200,
    ) -> list[NewsArticle]:
        """Fetch AI-scored news sentiment for a list of futures tickers.

        This is the primary Alpha Vantage call.  It uses ONE API call to
        retrieve up to ``limit`` articles covering all provided tickers (via
        their proxy ETF tickers) and their associated topics.

        Args:
            futures_tickers:  Internal futures tickers (e.g. ``["MES", "MCL"]``).
            extra_topics:     Additional AV topic strings to include.
            time_from:        Only return articles after this UTC datetime.
            limit:            Max articles (AV caps at 1000; practical limit 200).

        Returns:
            List of :class:`NewsArticle` with ``av_overall_score``,
            ``av_ticker_score``, and ``av_label`` populated.
        """
        if not self._key:
            logger.warning("Alpha Vantage: no API key")
            return []

        # Build the ticker string from proxies (deduplicated)
        av_tickers: list[str] = []
        seen_av: set[str] = set()
        topics: set[str] = set()

        if extra_topics:
            for t in extra_topics:
                if t in self.VALID_TOPICS:
                    topics.add(t)

        for ft in futures_tickers:
            proxy_info = ALPHA_VANTAGE_PROXY.get(ft)
            if proxy_info:
                av_t = proxy_info["ticker"]
                topic = proxy_info.get("topic", "financial_markets")
                if av_t not in seen_av:
                    av_tickers.append(av_t)
                    seen_av.add(av_t)
                if topic in self.VALID_TOPICS:
                    topics.add(topic)

        if not av_tickers:
            logger.debug("Alpha Vantage: no proxy mapping for tickers %s", futures_tickers)
            return []

        params: dict[str, Any] = {
            "function": "NEWS_SENTIMENT",
            "tickers": ",".join(av_tickers[:10]),  # AV caps ticker list
            "limit": str(min(limit, 1000)),
            "sort": "LATEST",
        }
        if topics:
            params["topics"] = ",".join(sorted(topics))
        if time_from:
            params["time_from"] = time_from.strftime("%Y%m%dT%H%M")

        data = self._get(params)
        if not data:
            return []

        feed = data.get("feed") or []
        if not isinstance(feed, list):
            return []

        # Build a reverse map: AV ticker → list of futures tickers
        av_to_futures: dict[str, list[str]] = {}
        for ft in futures_tickers:
            proxy_info = ALPHA_VANTAGE_PROXY.get(ft)
            if proxy_info:
                av_t = proxy_info["ticker"]
                av_to_futures.setdefault(av_t, []).append(ft)

        articles: list[NewsArticle] = []
        for item in feed:
            headline = item.get("title") or ""
            summary = item.get("summary") or headline
            url = item.get("url") or ""
            published_at = self._parse_av_datetime(item.get("time_published"))

            if not headline:
                continue

            # Overall article sentiment
            overall_score_raw = item.get("overall_sentiment_score", 0.5)
            overall_label = item.get("overall_sentiment_label") or "Neutral"
            overall_score = self._av_score_to_float(overall_label, overall_score_raw)

            # Per-ticker sentiment (may be present for each ticker in the article)
            ticker_sentiments = item.get("ticker_sentiment") or []
            ts_map: dict[str, dict[str, Any]] = {}
            for ts_entry in ticker_sentiments:
                ts_map[ts_entry.get("ticker", "")] = ts_entry

            # Emit one article per matched futures ticker so the scorer can
            # attribute it precisely.
            matched_futures: list[str] = []
            for ts_entry in ticker_sentiments:
                av_t = ts_entry.get("ticker", "")
                if av_t in av_to_futures:
                    matched_futures.extend(av_to_futures[av_t])

            # Deduplicate
            matched_futures = list(dict.fromkeys(matched_futures))

            # If no specific ticker match, emit once as a generic market article
            if not matched_futures:
                articles.append(
                    NewsArticle(
                        source="alphavantage",
                        ticker="MARKET",
                        proxy="",
                        headline=headline,
                        summary=summary,
                        url=url,
                        published_at=published_at,
                        av_overall_score=overall_score,
                        av_ticker_score=None,
                        av_label=overall_label,
                        raw=item,
                    )
                )
                continue

            for ft in matched_futures:
                proxy_info = ALPHA_VANTAGE_PROXY.get(ft, {})
                av_t = proxy_info.get("ticker", "")
                ts_entry = ts_map.get(av_t, {})

                ticker_score_raw = ts_entry.get("ticker_sentiment_score")
                ticker_label = ts_entry.get("ticker_sentiment_label") or overall_label
                ticker_score = (
                    self._av_score_to_float(ticker_label, ticker_score_raw)
                    if ticker_score_raw is not None
                    else overall_score
                )

                articles.append(
                    NewsArticle(
                        source="alphavantage",
                        ticker=ft,
                        proxy=av_t,
                        headline=headline,
                        summary=summary,
                        url=url,
                        published_at=published_at,
                        av_overall_score=overall_score,
                        av_ticker_score=ticker_score,
                        av_label=ticker_label,
                        raw=item,
                    )
                )

        logger.info(
            "Alpha Vantage: %d articles from feed of %d for %d tickers",
            len(articles),
            len(feed),
            len(futures_tickers),
        )
        return articles

    def fetch_commodity_price(
        self,
        commodity: str = "WTI",
        interval: str = "daily",
    ) -> dict[str, Any] | None:
        """Fetch commodity price series from Alpha Vantage.

        Args:
            commodity:  One of the :attr:`COMMODITY_CODES` keys.
            interval:   ``"daily"`` | ``"weekly"`` | ``"monthly"``.

        Returns:
            Dict with ``"data"`` key (list of ``{date, value}`` dicts) or None.
        """
        code = self.COMMODITY_CODES.get(commodity.upper())
        if not code:
            logger.warning("Alpha Vantage: unknown commodity '%s'", commodity)
            return None

        data = self._get({"function": code, "interval": interval})
        return data

    def is_available(self) -> bool:
        """Check if the key works by fetching a small commodity dataset."""
        if not self._key:
            return False
        data = self.fetch_commodity_price("WTI", interval="monthly")
        return data is not None and "data" in data


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------


def fetch_all_news(
    futures_tickers: list[str],
    finnhub_key: str | None = None,
    alpha_key: str | None = None,
    days_back: int = 2,
    max_per_ticker: int = 5,
    include_general: bool = True,
) -> list[NewsArticle]:
    """Fetch news from all available sources and return a merged list.

    Designed for the morning scheduler run.  Both sources are optional —
    if a key is missing the corresponding source is silently skipped.

    Args:
        futures_tickers:   List of internal tickers to fetch news for.
        finnhub_key:       Finnhub API key (or None to skip).
        alpha_key:         Alpha Vantage API key (or None to skip).
        days_back:         Look-back window for Finnhub company news.
        max_per_ticker:    Max Finnhub articles per ticker.
        include_general:   Also fetch Finnhub general market headlines.

    Returns:
        Combined, deduplicated list of :class:`NewsArticle` objects sorted
        by ``published_at`` descending.
    """
    all_articles: list[NewsArticle] = []
    seen_urls: set[str] = set()

    # ── Finnhub ────────────────────────────────────────────────────────────
    if finnhub_key:
        fh = FinnhubClient(api_key=finnhub_key)
        try:
            if include_general:
                general = fh.fetch_general_news(category="general")
                for a in general[:20]:  # cap general headlines
                    if a.url not in seen_urls:
                        seen_urls.add(a.url)
                        all_articles.append(a)

            batch = fh.fetch_batch(
                futures_tickers,
                days_back=days_back,
                max_per_ticker=max_per_ticker,
            )
            for a in batch:
                if a.url not in seen_urls:
                    seen_urls.add(a.url)
                    all_articles.append(a)
        except Exception as exc:
            logger.error("fetch_all_news: Finnhub error: %s", exc)

    # ── Alpha Vantage (single batch call) ─────────────────────────────────
    if alpha_key:
        av = AlphaVantageClient(api_key=alpha_key)
        time_from = datetime.now(tz=UTC) - timedelta(days=days_back)
        try:
            av_articles = av.fetch_news_sentiment(
                futures_tickers,
                time_from=time_from,
                limit=200,
            )
            for a in av_articles:
                if a.url not in seen_urls:
                    seen_urls.add(a.url)
                    all_articles.append(a)
        except Exception as exc:
            logger.error("fetch_all_news: Alpha Vantage error: %s", exc)

    # Sort newest first
    all_articles.sort(key=lambda a: a.published_at, reverse=True)

    logger.info(
        "fetch_all_news: %d total articles (%d tickers, %d sources)",
        len(all_articles),
        len(futures_tickers),
        sum([bool(finnhub_key), bool(alpha_key)]),
    )
    return all_articles
