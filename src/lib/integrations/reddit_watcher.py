"""
src/lib/integrations/reddit_watcher.py
───────────────────────────────────────
Reddit sentiment watcher — mirrors the pattern of grok_helper.py.

Fetches posts + top-level comments from four subreddits and produces
per-asset sentiment records that the analysis layer can query from Redis
or Postgres.

Usage (from scheduler.py or lifespan startup):
    from lib.integrations.reddit_watcher import RedditWatcher
    watcher = RedditWatcher(redis=redis_client, pg_pool=pg_pool)
    asyncio.create_task(watcher.run())

Dependencies (add to pyproject.toml):
    praw>=7.7
    vaderSentiment>=3.3
    # optional heavy path:
    # transformers>=4.40  (FinBERT — enable via USE_FINBERT=true in .env)
"""

from __future__ import annotations

import asyncio
import datetime as dt_module
import json
import logging
import os
import re
from datetime import datetime
from functools import lru_cache

import praw
import prawcore
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

logger = logging.getLogger("reddit_watcher")

# ─────────────────────────────────────────────────────────────────────────────
# Asset keyword map  (matches your CNN_RETRAIN_SYMBOLS list)
# ─────────────────────────────────────────────────────────────────────────────
ASSET_KEYWORDS: dict[str, list[str]] = {
    "GC": ["gold", "gc", "gc1!", "xau", "xauusd", "comex gold", "gold futures"],
    "NQ": ["nq", "nq1!", "nasdaq futures", "nasdaq 100", "mnq", "mnq1!", "qqq"],
    "ES": ["es", "es1!", "s&p futures", "spx futures", "spy", "sp500", "mes", "mes1!", "emini"],
    "6E": ["6e", "6e1!", "euro futures", "eurusd", "eur/usd", "euro dollar"],
    "BTC": ["btc", "bitcoin", "btcusd", "btc futures", "xbt"],
    "ETH": ["eth", "ethereum", "ethusd", "eth futures"],
    "SOL": ["sol", "solana", "solusd", "sol futures"],
}

KEYWORD_TO_ASSET: dict[str, str] = {kw.lower(): asset for asset, kws in ASSET_KEYWORDS.items() for kw in kws}

SUBREDDITS = [
    "InnerCircleTraders",
    "Daytrading",
    "wallstreetbets",
    "FuturesTrading",
]

# ─────────────────────────────────────────────────────────────────────────────
# Redis key templates  (consistent with your cache.py patterns)
# ─────────────────────────────────────────────────────────────────────────────
REDIS_SENTIMENT_ZSET = "reddit:sentiment:{asset}"  # sorted set, score=unix_ts
REDIS_MENTIONS_KEY = "reddit:mentions:{asset}:{win}"  # string, expiring counter
REDIS_SIGNAL_HASH = "reddit:signal:{asset}"  # hash, latest agg signal
REDIS_MAX_RECORDS = 500  # per-asset sorted set cap

POLL_INTERVAL = int(os.getenv("REDDIT_POLL_INTERVAL", "120"))  # seconds
USE_FINBERT = os.getenv("USE_FINBERT", "false").lower() == "true"

# Thresholds
BULL_THRESH = 0.25
BEAR_THRESH = -0.25

# ─────────────────────────────────────────────────────────────────────────────
# VADER + domain lexicon
# ─────────────────────────────────────────────────────────────────────────────
_TRADING_LEXICON = {
    "breakout": 2.5,
    "long": 1.5,
    "buy": 1.5,
    "moon": 2.0,
    "squeeze": 1.5,
    "rip": 1.8,
    "pump": 1.5,
    "bull": 2.0,
    "bullish": 2.5,
    "uptrend": 2.0,
    "accumulate": 1.5,
    "bounce": 1.2,
    "fomo": 1.0,
    "higher": 0.8,
    "bid": 1.0,
    "short": -1.5,
    "puts": -1.0,
    "dump": -1.5,
    "bear": -2.0,
    "bearish": -2.5,
    "crash": -2.5,
    "collapse": -2.5,
    "breakdown": -2.5,
    "downtrend": -2.0,
    "distribute": -1.5,
    "liquidate": -1.5,
    "capitulate": -2.0,
    "rekt": -2.5,
    "bleed": -2.0,
    "rejection": -1.2,
    # neutralise noisy words
    "futures": 0.0,
    "contract": 0.0,
    "trade": 0.0,
    "position": 0.0,
    "market": 0.0,
}


@lru_cache(maxsize=1)
def _vader() -> SentimentIntensityAnalyzer:
    sia = SentimentIntensityAnalyzer()
    sia.lexicon.update(_TRADING_LEXICON)
    return sia


def _clean(text: str) -> str:
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"\*\*?|~~|>|#+", "", text)
    return re.sub(r"\s+", " ", text).strip()


def _score(text: str) -> tuple[float, str]:
    """Return (compound, label).  label ∈ {'bullish','bearish','neutral'}"""
    if not text or not text.strip():
        return 0.0, "neutral"
    if USE_FINBERT:
        try:
            return _finbert_score(text)
        except Exception:
            pass
    compound = _vader().polarity_scores(_clean(text))["compound"]
    if compound >= BULL_THRESH:
        return compound, "bullish"
    if compound <= BEAR_THRESH:
        return compound, "bearish"
    return compound, "neutral"


@lru_cache(maxsize=1)
def _finbert_pipe():
    from transformers import pipeline as hf_pipeline

    return hf_pipeline("text-classification", model="ProsusAI/finbert")


def _finbert_score(text: str) -> tuple[float, str]:
    result = _finbert_pipe()(text[:512])[0]
    mapping = {"positive": 0.8, "negative": -0.8, "neutral": 0.0}
    compound = mapping.get(result["label"].lower(), 0.0)
    label = "bullish" if compound > 0 else ("bearish" if compound < 0 else "neutral")
    return compound, label


# ─────────────────────────────────────────────────────────────────────────────
# Asset extraction
# ─────────────────────────────────────────────────────────────────────────────
def extract_assets(text: str) -> list[str]:
    found, seen = [], set()
    for token in re.findall(r"[a-zA-Z0-9/!&]+", text.lower()):
        asset = KEYWORD_TO_ASSET.get(token)
        if asset and asset not in seen:
            found.append(asset)
            seen.add(asset)
    return found


# ─────────────────────────────────────────────────────────────────────────────
# Record normaliser
# ─────────────────────────────────────────────────────────────────────────────
def _normalise(submission, asset: str, is_comment: bool = False) -> dict:
    if is_comment:
        text = getattr(submission, "body", "") or ""
        title = ""
        url = f"https://reddit.com{submission.permalink}"
        score = submission.score
        nc, ratio = 0, None
    else:
        text = f"{submission.title or ''} {submission.selftext or ''}"
        title = submission.title or ""
        url = submission.url
        score = submission.score
        nc = submission.num_comments
        ratio = getattr(submission, "upvote_ratio", None)

    compound, label = _score(text)
    return {
        "post_id": submission.id if not is_comment else f"c_{submission.id}",
        "subreddit": str(submission.subreddit),
        "author": str(submission.author) if submission.author else "[deleted]",
        "title": title,
        "body": text[:2000],
        "url": url,
        "score": score,
        "num_comments": nc,
        "is_comment": is_comment,
        "asset": asset,
        "sentiment_score": round(compound, 4),
        "sentiment_label": label,
        "upvote_ratio": ratio,
        "created_utc": datetime.fromtimestamp(submission.created_utc, tz=dt_module.UTC),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Redis + Postgres writes
# ─────────────────────────────────────────────────────────────────────────────
async def _write_redis(record: dict, redis):
    asset = record["asset"]
    ts = record["created_utc"].timestamp()
    payload = json.dumps(
        {
            "post_id": record["post_id"],
            "compound": record["sentiment_score"],
            "label": record["sentiment_label"],
            "score": record["score"],
            "title": record["title"][:120],
            "subreddit": record["subreddit"],
            "ts": ts,
        }
    )
    key = REDIS_SENTIMENT_ZSET.format(asset=asset)
    pipe = redis.pipeline()
    pipe.zadd(key, {payload: ts})
    pipe.zremrangebyrank(key, 0, -(REDIS_MAX_RECORDS + 1))
    # rolling mention counters
    for secs, label in [(900, "15m"), (3600, "1h"), (14400, "4h"), (86400, "24h")]:
        cnt_key = REDIS_MENTIONS_KEY.format(asset=asset, win=label)
        pipe.incr(cnt_key)
        pipe.expire(cnt_key, secs)
    await pipe.execute()


async def _write_pg(record: dict, pg_pool):
    sql = """
        INSERT INTO reddit_posts (
            post_id, subreddit, author, title, body, url,
            score, num_comments, is_comment, asset,
            sentiment_score, sentiment_label, upvote_ratio, created_utc
        ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14)
        ON CONFLICT (post_id) DO NOTHING
    """
    try:
        async with pg_pool.acquire() as conn:
            await conn.execute(
                sql,
                record["post_id"],
                record["subreddit"],
                record["author"],
                record["title"],
                record["body"],
                record["url"],
                record["score"],
                record["num_comments"],
                record["is_comment"],
                record["asset"],
                record["sentiment_score"],
                record["sentiment_label"],
                record["upvote_ratio"],
                record["created_utc"],
            )
    except Exception as exc:
        logger.debug("PG insert skipped (%s): %s", record.get("post_id"), exc)


# ─────────────────────────────────────────────────────────────────────────────
# Main watcher class
# ─────────────────────────────────────────────────────────────────────────────
class RedditWatcher:
    """
    Drop-in async task.  Pass in the same redis + pg_pool already used
    by the rest of the engine.

        watcher = RedditWatcher(redis=r, pg_pool=pool)
        asyncio.create_task(watcher.run())
    """

    def __init__(
        self,
        redis,
        pg_pool,
        client_id: str | None = None,
        client_secret: str | None = None,
        user_agent: str = "FuturesCoPilot/1.0",
        mode: str = "poll",  # "poll" | "stream"
    ):
        self.redis = redis
        self.pg_pool = pg_pool
        self.mode = mode
        self._seen: set[str] = set()

        cid = client_id or os.getenv("REDDIT_CLIENT_ID", "")
        csec = client_secret or os.getenv("REDDIT_CLIENT_SECRET", "")
        if not cid or not csec:
            raise ValueError(
                "Set REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET in .env\n"
                "Free read-only app: https://www.reddit.com/prefs/apps"
            )
        self._reddit = praw.Reddit(client_id=cid, client_secret=csec, user_agent=user_agent)

    async def run(self):
        logger.info("RedditWatcher starting (mode=%s)", self.mode)
        if self.mode == "stream":
            await self._stream()
        else:
            await self._poll_loop()

    # ── streaming (best for high-volume subs) ────────────────────────────────
    async def _stream(self):
        loop = asyncio.get_running_loop()
        combined = self._reddit.subreddit("+".join(SUBREDDITS))

        def _worker():
            for sub in combined.stream.submissions(skip_existing=True, pause_after=5):
                if sub is None:
                    continue
                for asset in extract_assets(f"{sub.title} {sub.selftext}"):
                    rec = _normalise(sub, asset)
                    asyncio.run_coroutine_threadsafe(self._save(rec), loop)

        while True:
            try:
                await loop.run_in_executor(None, _worker)
            except prawcore.exceptions.ServerError as exc:
                logger.warning("Reddit server error: %s — retry in 60s", exc)
                await asyncio.sleep(60)

    # ── polling (safer rate-limit wise) ──────────────────────────────────────
    async def _poll_loop(self):
        while True:
            for sub_name in SUBREDDITS:
                try:
                    await self._poll_sub(sub_name)
                except Exception as exc:
                    logger.warning("r/%s poll error: %s", sub_name, exc)
            await asyncio.sleep(POLL_INTERVAL)

    async def _poll_sub(self, sub_name: str):
        loop = asyncio.get_running_loop()
        sub = self._reddit.subreddit(sub_name)

        def _fetch():
            results = []
            for post in [*sub.new(limit=50), *sub.hot(limit=25)]:
                if post.id in self._seen:
                    continue
                self._seen.add(post.id)
                for asset in extract_assets(f"{post.title} {post.selftext}"):
                    results.append(_normalise(post, asset))
                # top-level comments (up to 10 per post)
                try:
                    post.comments.replace_more(limit=0)
                    for comment in list(post.comments)[:10]:
                        cid = f"c_{comment.id}"
                        if cid in self._seen:
                            continue
                        self._seen.add(cid)
                        for asset in extract_assets(getattr(comment, "body", "")):
                            results.append(_normalise(comment, asset, is_comment=True))
                except Exception:
                    pass
            return results

        records = await loop.run_in_executor(None, _fetch)
        for r in records:
            await self._save(r)
        if records:
            logger.debug("r/%s → %d new records", sub_name, len(records))

        # keep seen-set bounded
        if len(self._seen) > 20_000:
            self._seen = set(list(self._seen)[-10_000:])

    async def _save(self, record: dict):
        await _write_redis(record, self.redis)
        await _write_pg(record, self.pg_pool)
