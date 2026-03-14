"""
src/lib/analysis/news_sentiment.py
────────────────────────────────────
Hybrid news sentiment scorer: VADER (fast, free) + Alpha Vantage AI scores
+ Grok 4.1 (context-aware, used only on ambiguous articles to minimise cost).

Architecture
────────────
  1. Each NewsArticle is scored by VADER on its headline + summary.
  2. If Alpha Vantage already provided an AI score it is incorporated directly.
  3. Articles where |vader_score| < GROK_THRESHOLD (ambiguous) are batched and
     sent to Grok in a single call — a structured JSON response scores all of
     them at once, costing ~$0.01–0.02 per morning run.
  4. Final hybrid score per article:
       hybrid = 0.40 × vader + 0.40 × av_score + 0.20 × grok_score
     When a component is unavailable its weight is redistributed to the others.
  5. Per-symbol aggregation produces a NewsSentiment dataclass with weighted
     sentiment, signal label, spike detection, and top headlines.

Futures-specific VADER lexicon extensions
──────────────────────────────────────────
Standard VADER was trained on social-media text and misses many futures-market
terms.  We extend the lexicon with domain-specific adjustments before scoring.
The full list is defined in _FUTURES_LEXICON below.

Usage
─────
    from lib.analysis.news_sentiment import compute_news_sentiment, compute_all_news_sentiments

    # Single symbol
    result = compute_news_sentiment("MGC=F", articles)

    # All symbols — used by scheduler
    results = compute_all_news_sentiments(articles_by_symbol)
"""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

logger = logging.getLogger("news_sentiment")

# ---------------------------------------------------------------------------
# Futures-specific VADER lexicon additions
# ---------------------------------------------------------------------------

_FUTURES_LEXICON: dict[str, float] = {
    # Bullish terms
    "surge": 3.0,
    "surging": 3.0,
    "rally": 2.8,
    "rallying": 2.8,
    "breakout": 2.5,
    "breakouts": 2.5,
    "squeeze": 2.2,
    "short squeeze": 3.5,
    "short-squeeze": 3.5,
    "oversold": 1.8,
    "inventory draw": 2.0,
    "draw": 1.5,
    "drawdown": -1.0,  # position drawdown is bad context
    "supply cut": 2.5,
    "output cut": 2.5,
    "hawkish": -1.5,  # bad for risk assets
    "dovish": 2.0,  # good for risk assets
    "rate cut": 2.5,
    "pivot": 2.0,
    "strong jobs": 1.5,
    "strong employment": 1.5,
    "strong gdp": 1.5,
    "beat": 1.8,
    "beats": 1.8,
    "upside surprise": 2.5,
    "higher high": 1.5,
    "new high": 2.0,
    "record high": 2.2,
    "all-time high": 2.5,
    "ath": 2.0,
    "inflows": 1.5,
    "safe haven": 1.8,
    "risk-on": 2.0,
    "risk on": 2.0,
    "reflationary": 1.5,
    "reflation": 1.5,
    "soft landing": 2.0,
    # Bearish terms
    "plunge": -3.0,
    "plunging": -3.0,
    "crash": -3.5,
    "crashing": -3.5,
    "collapse": -3.2,
    "sell-off": -2.8,
    "selloff": -2.8,
    "glut": -2.5,
    "oversupply": -2.0,
    "inventory build": -1.5,
    "build": -0.8,
    "rate hike": -2.0,
    "rate hikes": -2.0,
    "tightening": -1.8,
    "recession": -3.0,
    "stagflation": -2.5,
    "default": -2.8,
    "downgrade": -2.0,
    "miss": -1.8,
    "misses": -1.8,
    "downside surprise": -2.5,
    "lower low": -1.5,
    "new low": -2.0,
    "record low": -2.2,
    "outflows": -1.5,
    "risk-off": -2.0,
    "risk off": -2.0,
    "flight to safety": 1.0,  # bullish for treasuries/gold
    "tariff": -1.5,
    "tariffs": -1.5,
    "sanctions": -1.8,
    "war": -2.5,
    "geopolitical": -1.0,
    "opec": 0.0,  # context-dependent; Grok handles
    "opec+": 0.0,
    "fed": 0.0,  # context-dependent
    "fomc": 0.0,
    "cpi": 0.0,
    "ppi": 0.0,
    "nfp": 0.0,
    "jobs report": 0.0,
    "unemployment": -0.5,
}

# Ambiguity threshold — articles with |vader| below this go to Grok
GROK_THRESHOLD: float = 0.30

# Hybrid weight defaults (redistributed when a component is missing)
_W_VADER: float = 0.40
_W_AV: float = 0.40
_W_GROK: float = 0.20

# Maximum Grok batch size — stay well inside context limit
_GROK_BATCH_MAX: int = 40

# Redis TTL for cached sentiment results
REDIS_TTL_SECONDS: int = 7_200  # 2 hours

# Redis key template
REDIS_KEY_TEMPLATE: str = "engine:news_sentiment:{symbol}"
REDIS_SPIKE_KEY: str = "engine:news_spike"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class ScoredArticle:
    """A NewsArticle with all sentiment scores attached."""

    # Original fields (copied from NewsArticle)
    source: str
    ticker: str
    proxy: str
    headline: str
    summary: str
    url: str
    published_at: datetime

    # Scores
    vader_score: float = 0.0  # VADER compound [-1, +1]
    av_score: float | None = None  # Alpha Vantage score [-1, +1] (None if unavailable)
    grok_score: float | None = None  # Grok score [-1, +1] (None if not needed / unavailable)
    grok_label: str | None = None  # "Bullish" | "Bearish" | "Neutral"
    grok_reason: str | None = None  # Short Grok explanation
    hybrid_score: float = 0.0  # Final weighted combination

    # Metadata
    is_ambiguous: bool = False  # True if sent to Grok
    scored_at: datetime = field(default_factory=lambda: datetime.now(tz=UTC))


@dataclass
class NewsSentiment:
    """Aggregated sentiment for one symbol over the last fetch window."""

    symbol: str
    article_count: int
    finnhub_count: int
    av_count: int

    # Aggregated scores
    avg_vader: float  # Simple average VADER
    avg_av: float  # Simple average AV (0.0 if no AV articles)
    avg_grok: float  # Simple average Grok (0.0 if no Grok articles)
    weighted_hybrid: float  # Score-weighted hybrid [-1, +1]

    # Signal classification
    signal: str  # STRONG_BULL | BULL | NEUTRAL | BEAR | STRONG_BEAR
    confidence: float  # 0–1

    # Spike detection
    is_spike: bool  # True if article rate > 3× rolling avg
    articles_last_hour: int
    articles_rolling_avg: float

    # Top headlines (for dashboard)
    top_headlines: list[dict[str, Any]] = field(default_factory=list)

    # Grok narrative (set after calling get_grok_narrative)
    grok_narrative: str | None = None

    computed_at: datetime = field(default_factory=lambda: datetime.now(tz=UTC))


# ---------------------------------------------------------------------------
# VADER scoring
# ---------------------------------------------------------------------------

_vader_analyzer: Any = None


def _get_vader():
    """Lazy-load and configure the VADER analyser with futures lexicon."""
    global _vader_analyzer
    if _vader_analyzer is not None:
        return _vader_analyzer

    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

        analyzer = SentimentIntensityAnalyzer()
        # Extend the lexicon with futures-specific terms
        analyzer.lexicon.update(_FUTURES_LEXICON)
        _vader_analyzer = analyzer
        logger.debug("VADER analyser initialised with %d custom terms", len(_FUTURES_LEXICON))
        return analyzer
    except ImportError:
        logger.warning("vaderSentiment not installed — VADER scores will be 0.0")
        return None


def vader_score(text: str) -> float:
    """Score a text string using VADER + futures lexicon.

    Returns:
        Compound score in [-1, +1].  Returns 0.0 if VADER is unavailable.
    """
    if not text:
        return 0.0
    analyzer = _get_vader()
    if analyzer is None:
        return 0.0
    try:
        scores = analyzer.polarity_scores(str(text))
        return round(float(scores.get("compound", 0.0)), 4)
    except Exception as exc:
        logger.debug("VADER scoring failed: %s", exc)
        return 0.0


# ---------------------------------------------------------------------------
# Grok batch scoring
# ---------------------------------------------------------------------------


def grok_futures_sentiment_batch(
    articles: list[ScoredArticle],
    api_key: str,
    model: str = "grok-4-1-fast-reasoning",
    max_tokens: int = 1500,
) -> list[ScoredArticle]:
    """Score a batch of ambiguous articles with Grok in one API call.

    Sends all articles in a single structured prompt and parses the JSON
    response.  Each article gets a score in [-1, +1], a label, and a short
    reason.

    Args:
        articles:   List of :class:`ScoredArticle` with ``is_ambiguous=True``.
        api_key:    xAI API key (``XAI_API_KEY`` env var).
        model:      Grok model identifier.
        max_tokens: Max response tokens.

    Returns:
        The same list with ``grok_score``, ``grok_label``, and
        ``grok_reason`` populated where possible.
    """
    if not articles or not api_key:
        return articles

    # Build the numbered article list for the prompt
    items: list[str] = []
    for i, a in enumerate(articles):
        asset_name = a.ticker
        try:
            from lib.integrations.news_client import ASSET_DISPLAY

            asset_name = ASSET_DISPLAY.get(a.ticker, a.ticker)
        except ImportError:
            pass
        text = f"{a.headline}. {a.summary[:200]}" if a.summary else a.headline
        items.append(f'{i + 1}. [{asset_name}] "{text[:300]}"')

    numbered = "\n".join(items)

    system_prompt = (
        "You are a professional futures trader with 20 years of experience "
        "in commodities, equity index, FX, and crypto derivatives. "
        "You understand how macro and micro news affects futures prices intraday "
        "and over the next 1–3 days."
    )

    user_prompt = (
        f"Score the sentiment of each news article below from the perspective "
        f"of a CME micro futures trader.\n\n"
        f"For each article return a JSON object with:\n"
        f'  "id": <1-based integer>,\n'
        f'  "score": <float from -1.0 (very bearish) to +1.0 (very bullish)>,\n'
        f'  "label": <"Bullish" | "Neutral" | "Bearish">,\n'
        f'  "reason": <one sentence, ≤15 words>\n\n'
        f"Return a JSON array of objects.  No other text.\n\n"
        f"Articles:\n{numbered}"
    )

    try:
        import requests as _req

        resp = _req.post(
            "https://api.x.ai/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": 0.2,
                "max_tokens": max_tokens,
            },
            timeout=60,
        )
        resp.raise_for_status()
        content: str = resp.json()["choices"][0]["message"]["content"]
    except Exception as exc:
        logger.warning("Grok batch call failed: %s", exc)
        return articles

    # Parse JSON from response — strip any markdown fences
    content = content.strip()
    if content.startswith("```"):
        lines = content.split("\n")
        # Drop first and last fence lines
        content = "\n".join(line for line in lines if not line.strip().startswith("```"))

    try:
        scored = json.loads(content)
        if not isinstance(scored, list):
            raise ValueError("Expected JSON array")
    except (json.JSONDecodeError, ValueError) as exc:
        logger.warning("Grok batch: could not parse JSON response: %s", exc)
        return articles

    # Apply scores back to articles
    score_map: dict[int, dict[str, Any]] = {}
    for item in scored:
        if isinstance(item, dict):
            try:
                idx = int(item["id"]) - 1  # convert to 0-based
                score_map[idx] = item
            except (KeyError, TypeError, ValueError):
                continue

    for i, article in enumerate(articles):
        if i in score_map:
            entry = score_map[i]
            try:
                article.grok_score = max(-1.0, min(1.0, float(entry.get("score", 0.0))))
                article.grok_label = str(entry.get("label", "Neutral"))
                article.grok_reason = str(entry.get("reason", ""))[:120]
            except (TypeError, ValueError):
                pass

    return articles


def grok_narrative_summary(
    symbol: str,
    top_articles: list[ScoredArticle],
    weighted_score: float,
    signal: str,
    api_key: str,
    model: str = "grok-4-1-fast-reasoning",
) -> str | None:
    """Generate a 3-sentence Grok narrative for a symbol's news sentiment.

    Cheap call (~$0.005) — only made when the dashboard Research page loads
    or when the scheduler runs the morning briefing.

    Args:
        symbol:         Internal futures ticker.
        top_articles:   Up to 5 most recent / highest-impact articles.
        weighted_score: The computed weighted_hybrid score.
        signal:         Signal label (STRONG_BULL, BULL, …).
        api_key:        xAI API key.
        model:          Grok model identifier.

    Returns:
        Narrative string or None on failure.
    """
    if not api_key or not top_articles:
        return None

    asset_name = symbol
    try:
        from lib.integrations.news_client import ASSET_DISPLAY

        asset_name = ASSET_DISPLAY.get(symbol, symbol)
    except ImportError:
        pass

    headlines = "; ".join(f'"{a.headline[:100]}"' for a in top_articles[:5])
    sentiment_str = f"{signal} (score: {weighted_score:+.2f})"

    prompt = (
        f"Asset: {asset_name} ({symbol})\n"
        f"News sentiment: {sentiment_str}\n"
        f"Recent headlines: {headlines}\n\n"
        f"Write a 3-sentence trading narrative for a futures trader about "
        f"{asset_name} based on today's news. Focus on intraday and next-day "
        f"price implications. Be specific and concise."
    )

    try:
        import requests as _req

        resp = _req.post(
            "https://api.x.ai/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3,
                "max_tokens": 300,
            },
            timeout=45,
        )
        resp.raise_for_status()
        return str(resp.json()["choices"][0]["message"]["content"]).strip()
    except Exception as exc:
        logger.warning("Grok narrative failed for %s: %s", symbol, exc)
        return None


# ---------------------------------------------------------------------------
# Hybrid scoring
# ---------------------------------------------------------------------------


def _compute_hybrid(
    vader: float,
    av: float | None,
    grok: float | None,
) -> float:
    """Compute weighted hybrid score, redistributing weight for missing sources.

    When a component is missing (None), its weight is redistributed
    proportionally to the available components.

    Returns:
        Hybrid score in [-1, +1].
    """
    weights = {
        "vader": _W_VADER,
        "av": _W_AV if av is not None else 0.0,
        "grok": _W_GROK if grok is not None else 0.0,
    }
    total_w = weights["vader"] + weights["av"] + weights["grok"]

    if total_w == 0:
        return 0.0

    # Normalise weights
    wv = weights["vader"] / total_w
    wav = weights["av"] / total_w
    wg = weights["grok"] / total_w

    score = wv * vader + wav * (av or 0.0) + wg * (grok or 0.0)
    return round(max(-1.0, min(1.0, score)), 4)


def score_articles(
    articles: list[Any],  # list[NewsArticle] from news_client
    grok_api_key: str | None = None,
) -> list[ScoredArticle]:
    """Score a list of NewsArticle objects and return ScoredArticle objects.

    Pipeline:
      1. VADER score on headline + summary.
      2. Use Alpha Vantage score if present.
      3. Identify ambiguous articles (|vader| < GROK_THRESHOLD and no AV score).
      4. Batch-score ambiguous articles with Grok (if key provided).
      5. Compute hybrid score.

    Args:
        articles:       Raw NewsArticle objects from news_client.
        grok_api_key:   xAI API key.  Pass None to skip Grok (VADER + AV only).

    Returns:
        List of :class:`ScoredArticle` with all scores populated.
    """
    scored: list[ScoredArticle] = []

    for a in articles:
        # Extract fields defensively (NewsArticle dataclass or dict-like)
        headline = getattr(a, "headline", "") or ""
        summary = getattr(a, "summary", "") or ""
        text = f"{headline}. {summary}" if summary and summary != headline else headline

        vs = vader_score(text)
        av = getattr(a, "av_ticker_score", None) or getattr(a, "av_overall_score", None)
        published_at = getattr(a, "published_at", None) or datetime.now(tz=UTC)

        # An article is ambiguous if VADER is unsure AND AV hasn't scored it
        is_ambiguous = abs(vs) < GROK_THRESHOLD and av is None

        sa = ScoredArticle(
            source=getattr(a, "source", "unknown"),
            ticker=getattr(a, "ticker", "MARKET"),
            proxy=getattr(a, "proxy", ""),
            headline=headline,
            summary=summary,
            url=getattr(a, "url", ""),
            published_at=published_at,
            vader_score=vs,
            av_score=av,
            grok_score=None,
            is_ambiguous=is_ambiguous,
        )
        sa.hybrid_score = _compute_hybrid(vs, av, None)
        scored.append(sa)

    # Batch Grok scoring for ambiguous articles
    if grok_api_key:
        ambiguous = [s for s in scored if s.is_ambiguous]
        if ambiguous:
            logger.info(
                "Sending %d ambiguous articles to Grok (of %d total)",
                len(ambiguous),
                len(scored),
            )
            # Process in chunks to stay inside context window
            for i in range(0, len(ambiguous), _GROK_BATCH_MAX):
                chunk = ambiguous[i : i + _GROK_BATCH_MAX]
                grok_futures_sentiment_batch(chunk, api_key=grok_api_key)

            # Recompute hybrid scores now that Grok scores are available
            for s in scored:
                if s.is_ambiguous and s.grok_score is not None:
                    s.hybrid_score = _compute_hybrid(s.vader_score, s.av_score, s.grok_score)

    return scored


# ---------------------------------------------------------------------------
# Signal classification
# ---------------------------------------------------------------------------


def _classify_signal(weighted: float, bull_ratio: float, bear_ratio: float) -> str:
    """Classify a weighted sentiment score into a signal label."""
    if weighted >= 0.40 or (weighted >= 0.25 and bull_ratio >= 0.65):
        return "STRONG_BULL"
    if weighted >= 0.25:
        return "BULL"
    if weighted <= -0.40 or (weighted <= -0.25 and bear_ratio >= 0.65):
        return "STRONG_BEAR"
    if weighted <= -0.25:
        return "BEAR"
    return "NEUTRAL"


def _confidence(n: int) -> float:
    """Confidence grows with article count, saturating around 50 articles."""
    return round(1 - math.exp(-n / 20), 3)


# ---------------------------------------------------------------------------
# Per-symbol aggregation
# ---------------------------------------------------------------------------


def compute_news_sentiment(
    symbol: str,
    scored_articles: list[ScoredArticle],
    rolling_avg_per_hour: float = 0.0,
) -> NewsSentiment:
    """Aggregate scored articles for a single symbol into a NewsSentiment.

    Args:
        symbol:                 Internal futures ticker (e.g. ``"MGC=F"``).
        scored_articles:        Articles that matched this symbol (any ticker).
        rolling_avg_per_hour:   Historical baseline for spike detection.
                                Pass 0.0 to disable spike detection.

    Returns:
        :class:`NewsSentiment` dataclass.
    """
    relevant = [a for a in scored_articles if a.ticker == symbol or a.ticker == "MARKET"]

    if not relevant:
        return NewsSentiment(
            symbol=symbol,
            article_count=0,
            finnhub_count=0,
            av_count=0,
            avg_vader=0.0,
            avg_av=0.0,
            avg_grok=0.0,
            weighted_hybrid=0.0,
            signal="NEUTRAL",
            confidence=0.0,
            is_spike=False,
            articles_last_hour=0,
            articles_rolling_avg=rolling_avg_per_hour,
        )

    n = len(relevant)
    finnhub_count = sum(1 for a in relevant if a.source == "finnhub")
    av_count = sum(1 for a in relevant if a.source == "alphavantage")

    # Simple averages
    avg_vader = round(sum(a.vader_score for a in relevant) / n, 4)

    av_articles = [a for a in relevant if a.av_score is not None]
    avg_av = (
        round(sum(a.av_score for a in av_articles if a.av_score is not None) / len(av_articles), 4)
        if av_articles
        else 0.0
    )

    grok_articles = [a for a in relevant if a.grok_score is not None]
    avg_grok = (
        round(sum(a.grok_score for a in grok_articles if a.grok_score is not None) / len(grok_articles), 4)
        if grok_articles
        else 0.0
    )

    # Weighted hybrid aggregate — weight by recency (newer = higher weight)
    # using an exponential decay so the last 6 hours dominate
    now = datetime.now(tz=UTC)
    decay_hours = 6.0
    weighted_sum = 0.0
    weight_total = 0.0

    bull_count = 0
    bear_count = 0

    for a in relevant:
        age_hours = (now - a.published_at).total_seconds() / 3600.0
        w = math.exp(-age_hours / decay_hours)
        weighted_sum += a.hybrid_score * w
        weight_total += w
        if a.hybrid_score > 0.15:
            bull_count += 1
        elif a.hybrid_score < -0.15:
            bear_count += 1

    weighted_hybrid = round(weighted_sum / weight_total, 4) if weight_total > 0 else 0.0
    bull_ratio = bull_count / n
    bear_ratio = bear_count / n

    signal = _classify_signal(weighted_hybrid, bull_ratio, bear_ratio)
    confidence = _confidence(n)

    # Spike detection
    one_hour_ago = now.timestamp() - 3600
    articles_last_hour = sum(1 for a in relevant if a.published_at.timestamp() >= one_hour_ago)
    is_spike = rolling_avg_per_hour > 0 and articles_last_hour > rolling_avg_per_hour * 3.0

    # Top headlines — most bullish or most bearish by |hybrid_score|, then most recent
    top = sorted(relevant, key=lambda a: (abs(a.hybrid_score), a.published_at.timestamp()), reverse=True)[:5]
    top_headlines = [
        {
            "headline": a.headline,
            "source": a.source,
            "url": a.url,
            "hybrid_score": a.hybrid_score,
            "vader_score": a.vader_score,
            "av_score": a.av_score,
            "grok_score": a.grok_score,
            "grok_reason": a.grok_reason,
            "published_at": a.published_at.isoformat(),
            "label": ("bullish" if a.hybrid_score > 0.15 else "bearish" if a.hybrid_score < -0.15 else "neutral"),
        }
        for a in top
    ]

    return NewsSentiment(
        symbol=symbol,
        article_count=n,
        finnhub_count=finnhub_count,
        av_count=av_count,
        avg_vader=avg_vader,
        avg_av=avg_av,
        avg_grok=avg_grok,
        weighted_hybrid=weighted_hybrid,
        signal=signal,
        confidence=confidence,
        is_spike=is_spike,
        articles_last_hour=articles_last_hour,
        articles_rolling_avg=rolling_avg_per_hour,
        top_headlines=top_headlines,
    )


def compute_all_news_sentiments(
    scored_articles: list[ScoredArticle],
    symbols: list[str] | None = None,
    rolling_avgs: dict[str, float] | None = None,
) -> dict[str, NewsSentiment]:
    """Compute sentiment for all symbols from a flat article list.

    Args:
        scored_articles:    All scored articles from :func:`score_articles`.
        symbols:            Symbols to compute.  If None, uses all tickers found
                            in the articles (excluding ``"MARKET"``).
        rolling_avgs:       Dict of symbol → historical articles/hour baseline.

    Returns:
        Dict of symbol → :class:`NewsSentiment`.
    """
    if symbols is None:
        symbols = list({a.ticker for a in scored_articles if a.ticker != "MARKET"})

    avgs = rolling_avgs or {}
    return {sym: compute_news_sentiment(sym, scored_articles, avgs.get(sym, 0.0)) for sym in symbols}


# ---------------------------------------------------------------------------
# Redis caching
# ---------------------------------------------------------------------------


def cache_sentiments(sentiments: dict[str, NewsSentiment], redis: Any) -> None:
    """Write sentiment results to Redis for the API and dashboard to read.

    Keys:
        ``engine:news_sentiment:{symbol}``  — JSON hash, TTL 2 hours
        ``engine:news_spike``               — JSON dict of spiking symbols

    Args:
        sentiments: Dict of symbol → :class:`NewsSentiment`.
        redis:      Redis client instance.
    """
    spikes: dict[str, dict[str, Any]] = {}

    pipe = redis.pipeline()
    for sym, ns in sentiments.items():
        key = REDIS_KEY_TEMPLATE.format(symbol=sym)
        payload = {
            "symbol": ns.symbol,
            "article_count": ns.article_count,
            "finnhub_count": ns.finnhub_count,
            "av_count": ns.av_count,
            "avg_vader": ns.avg_vader,
            "avg_av": ns.avg_av,
            "avg_grok": ns.avg_grok,
            "weighted_hybrid": ns.weighted_hybrid,
            "signal": ns.signal,
            "confidence": ns.confidence,
            "is_spike": int(ns.is_spike),
            "articles_last_hour": ns.articles_last_hour,
            "grok_narrative": ns.grok_narrative or "",
            "top_headlines": json.dumps(ns.top_headlines),
            "computed_at": ns.computed_at.isoformat(),
        }
        pipe.hset(key, mapping=payload)
        pipe.expire(key, REDIS_TTL_SECONDS)

        if ns.is_spike:
            spikes[sym] = {
                "symbol": sym,
                "articles_last_hour": ns.articles_last_hour,
                "rolling_avg": ns.articles_rolling_avg,
                "weighted_hybrid": ns.weighted_hybrid,
                "signal": ns.signal,
            }

    if spikes:
        pipe.set(REDIS_SPIKE_KEY, json.dumps(spikes), ex=REDIS_TTL_SECONDS)

    try:
        pipe.execute()
    except Exception as exc:
        logger.error("cache_sentiments: Redis write failed: %s", exc)


def load_sentiment_from_cache(symbol: str, redis: Any) -> NewsSentiment | None:
    """Read a cached NewsSentiment from Redis.

    Returns:
        :class:`NewsSentiment` or None if not cached / expired.
    """
    key = REDIS_KEY_TEMPLATE.format(symbol=symbol)
    try:
        raw = redis.hgetall(key)
        if not raw:
            return None

        def _f(k: str, default: float = 0.0) -> float:
            v = raw.get(k) or raw.get(k.encode(), b"")
            if isinstance(v, bytes):
                v = v.decode()
            try:
                return float(v)
            except (TypeError, ValueError):
                return default

        def _s(k: str, default: str = "") -> str:
            v = raw.get(k) or raw.get(k.encode(), b"")
            if isinstance(v, bytes):
                v = v.decode()
            return str(v) if v else default

        top_headlines_raw = _s("top_headlines", "[]")
        try:
            top_headlines = json.loads(top_headlines_raw)
        except json.JSONDecodeError:
            top_headlines = []

        computed_at_str = _s("computed_at")
        try:
            computed_at = datetime.fromisoformat(computed_at_str)
        except ValueError:
            computed_at = datetime.now(tz=UTC)

        return NewsSentiment(
            symbol=_s("symbol", symbol),
            article_count=int(_f("article_count")),
            finnhub_count=int(_f("finnhub_count")),
            av_count=int(_f("av_count")),
            avg_vader=_f("avg_vader"),
            avg_av=_f("avg_av"),
            avg_grok=_f("avg_grok"),
            weighted_hybrid=_f("weighted_hybrid"),
            signal=_s("signal", "NEUTRAL"),
            confidence=_f("confidence"),
            is_spike=bool(int(_f("is_spike"))),
            articles_last_hour=int(_f("articles_last_hour")),
            articles_rolling_avg=_f("articles_rolling_avg"),
            top_headlines=top_headlines,
            grok_narrative=_s("grok_narrative") or None,
            computed_at=computed_at,
        )
    except Exception as exc:
        logger.error("load_sentiment_from_cache: error for %s: %s", symbol, exc)
        return None


def get_cached_sentiments(symbols: list[str], redis: Any) -> dict[str, NewsSentiment]:
    """Bulk-read cached sentiments from Redis.

    Returns only symbols that have valid cached data.
    """
    results: dict[str, NewsSentiment] = {}
    for sym in symbols:
        ns = load_sentiment_from_cache(sym, redis)
        if ns is not None:
            results[sym] = ns
    return results


# ---------------------------------------------------------------------------
# Scheduler entry point
# ---------------------------------------------------------------------------


def run_news_sentiment_pipeline(
    symbols: list[str],
    finnhub_key: str | None = None,
    alpha_key: str | None = None,
    grok_key: str | None = None,
    redis: Any = None,
    days_back: int = 2,
    max_per_ticker: int = 5,
    rolling_avgs: dict[str, float] | None = None,
) -> dict[str, NewsSentiment]:
    """Full pipeline: fetch → score → aggregate → cache.

    Designed to be called by the engine scheduler at 07:00 ET and 12:00 ET.

    Args:
        symbols:        Internal futures tickers to process.
        finnhub_key:    Finnhub API key (or None to skip).
        alpha_key:      Alpha Vantage API key (or None to skip).
        grok_key:       xAI API key (or None to skip Grok scoring).
        redis:          Redis client for caching (or None to skip cache).
        days_back:      Look-back window for news fetch.
        max_per_ticker: Max Finnhub articles per ticker.
        rolling_avgs:   Historical articles/hour baselines for spike detection.

    Returns:
        Dict of symbol → :class:`NewsSentiment`.
    """
    t0 = time.monotonic()

    # 1. Fetch
    try:
        from lib.integrations.news_client import fetch_all_news

        raw_articles = fetch_all_news(
            futures_tickers=symbols,
            finnhub_key=finnhub_key,
            alpha_key=alpha_key,
            days_back=days_back,
            max_per_ticker=max_per_ticker,
        )
    except Exception as exc:
        logger.error("run_news_sentiment_pipeline: fetch failed: %s", exc)
        raw_articles = []

    if not raw_articles:
        logger.warning("run_news_sentiment_pipeline: no articles fetched")
        return {}

    # 2. Score
    try:
        scored = score_articles(raw_articles, grok_api_key=grok_key)
    except Exception as exc:
        logger.error("run_news_sentiment_pipeline: scoring failed: %s", exc)
        return {}

    # 3. Aggregate
    sentiments = compute_all_news_sentiments(
        scored,
        symbols=symbols,
        rolling_avgs=rolling_avgs,
    )

    # 4. Cache
    if redis is not None:
        try:
            cache_sentiments(sentiments, redis)
        except Exception as exc:
            logger.error("run_news_sentiment_pipeline: cache failed: %s", exc)

    elapsed = time.monotonic() - t0
    logger.info(
        "run_news_sentiment_pipeline: %d symbols, %d articles → %d sentiments in %.1fs",
        len(symbols),
        len(raw_articles),
        len(sentiments),
        elapsed,
    )

    # Log spike alerts
    for sym, ns in sentiments.items():
        if ns.is_spike:
            logger.warning(
                "📰 NEWS SPIKE: %s — %d articles/hr (avg %.1f), sentiment %s (%.2f)",
                sym,
                ns.articles_last_hour,
                ns.articles_rolling_avg,
                ns.signal,
                ns.weighted_hybrid,
            )

    return sentiments
