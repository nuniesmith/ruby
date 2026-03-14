"""
lib.analysis.sentiment — Sentiment analysis sub-package.

Re-exports the public API from both sentiment modules so callers can use either:

    from lib.analysis.sentiment import NewsSentiment, compute_news_sentiment
    from lib.analysis.sentiment.news_sentiment import compute_news_sentiment
    from lib.analysis.sentiment.reddit_sentiment import get_full_snapshot
"""

# news_sentiment — hybrid VADER + Alpha Vantage + Grok scorer
from lib.analysis.sentiment.news_sentiment import (
    REDIS_KEY_TEMPLATE,
    REDIS_SPIKE_KEY,
    REDIS_TTL_SECONDS,
    NewsSentiment,
    ScoredArticle,
    cache_sentiments,
    compute_all_news_sentiments,
    compute_news_sentiment,
    get_cached_sentiments,
    load_sentiment_from_cache,
    run_news_sentiment_pipeline,
)

# reddit_sentiment — Reddit signal aggregation layer
from lib.analysis.sentiment.reddit_sentiment import (
    WINDOWS_MINUTES,
    RedditSignal,
    RedditSnapshot,
    aggregate_asset,
    get_asset_signal,
    get_full_snapshot,
)

__all__ = [
    # news_sentiment
    "NewsSentiment",
    "ScoredArticle",
    "REDIS_KEY_TEMPLATE",
    "REDIS_SPIKE_KEY",
    "REDIS_TTL_SECONDS",
    "cache_sentiments",
    "compute_all_news_sentiments",
    "compute_news_sentiment",
    "get_cached_sentiments",
    "load_sentiment_from_cache",
    "run_news_sentiment_pipeline",
    # reddit_sentiment
    "WINDOWS_MINUTES",
    "RedditSignal",
    "RedditSnapshot",
    "aggregate_asset",
    "get_asset_signal",
    "get_full_snapshot",
]
