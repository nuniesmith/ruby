"""
src/lib/analysis/reddit_sentiment.py
──────────────────────────────────────
Signal aggregation layer — mirrors the pattern of volatility.py / regime.py.

Reads raw Redis sentiment data and produces a composite signal per asset
per time window.  Called by the scheduler and exposed via the API route.

Signal logic
────────────
  weighted_sentiment  = Σ(compound * log1p(abs(post_score))) / Σ(log1p(abs(post_score)))
  bull_ratio          = bullish_count / total
  bear_ratio          = bearish_count / total
  mention_velocity    = mentions_15m / (mentions_1h / 4)   > 1.5 = spike

  STRONG_BULL  w ≥ +0.40  OR  (w ≥ +0.25 AND bull_ratio ≥ 0.65)
  BULL         w ≥ +0.25
  NEUTRAL      |w| < 0.25
  BEAR         w ≤ −0.25
  STRONG_BEAR  w ≤ −0.40  OR  (w ≤ −0.25 AND bear_ratio ≥ 0.65)

  confidence  = sigmoid-like function of mention count (saturates at ~100 mentions)
"""

from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import dataclass

from lib.integrations.reddit_watcher import (
    ASSET_KEYWORDS,
    REDIS_MENTIONS_KEY,
    REDIS_SENTIMENT_ZSET,
)

logger = logging.getLogger("reddit_sentiment")

WINDOWS_MINUTES = [15, 60, 240, 1440]


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class RedditSignal:
    asset: str
    window_minutes: int
    mention_count: int
    bullish_count: int
    bearish_count: int
    neutral_count: int
    avg_sentiment: float
    weighted_sentiment: float
    bull_ratio: float
    bear_ratio: float
    mention_velocity: float  # >1.5 = spike vs normal rate
    signal: str  # STRONG_BULL | BULL | NEUTRAL | BEAR | STRONG_BEAR
    confidence: float  # 0–1
    top_posts: list[dict]  # highest-score posts for this window


@dataclass
class RedditSnapshot:
    """All assets, all windows — returned by get_full_snapshot()."""

    signals: dict[str, dict[int, RedditSignal]]  # asset → window_min → signal
    computed_at: float


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _label(w: float, bull_r: float, bear_r: float) -> str:
    if w >= 0.40 or (w >= 0.25 and bull_r >= 0.65):
        return "STRONG_BULL"
    if w >= 0.25:
        return "BULL"
    if w <= -0.40 or (w <= -0.25 and bear_r >= 0.65):
        return "STRONG_BEAR"
    if w <= -0.25:
        return "BEAR"
    return "NEUTRAL"


def _confidence(n: int) -> float:
    """0 → 0.0, 10 → ~0.28, 30 → ~0.63, 100 → ~0.96"""
    return round(1 - math.exp(-n / 30), 3)


# ─────────────────────────────────────────────────────────────────────────────
# Core aggregation
# ─────────────────────────────────────────────────────────────────────────────
async def aggregate_asset(asset: str, window_min: int, redis) -> RedditSignal:
    window_sec = window_min * 60
    since = time.time() - window_sec
    key = REDIS_SENTIMENT_ZSET.format(asset=asset)

    raw = await redis.zrangebyscore(key, since, "+inf")
    records = [json.loads(r) for r in raw]

    if not records:
        return RedditSignal(
            asset=asset,
            window_minutes=window_min,
            mention_count=0,
            bullish_count=0,
            bearish_count=0,
            neutral_count=0,
            avg_sentiment=0.0,
            weighted_sentiment=0.0,
            bull_ratio=0.0,
            bear_ratio=0.0,
            mention_velocity=0.0,
            signal="NEUTRAL",
            confidence=0.0,
            top_posts=[],
        )

    total = len(records)
    bull = sum(1 for r in records if r["label"] == "bullish")
    bear = sum(1 for r in records if r["label"] == "bearish")
    neutral = total - bull - bear
    avg_sent = round(sum(r["compound"] for r in records) / total, 4)

    # score-weighted sentiment (upvoted posts count more)
    weight_sum = sum(math.log1p(abs(r.get("score", 0))) for r in records)
    if weight_sum > 0:
        w_sent = sum(r["compound"] * math.log1p(abs(r.get("score", 0))) for r in records) / weight_sum
    else:
        w_sent = avg_sent

    w_sent = round(w_sent, 4)
    bull_r = round(bull / total, 3)
    bear_r = round(bear / total, 3)
    signal = _label(w_sent, bull_r, bear_r)
    conf = _confidence(total)

    # mention velocity — compare 15m rate to 1h rate
    m15 = int(await redis.get(REDIS_MENTIONS_KEY.format(asset=asset, win="15m")) or 0)
    m1h = int(await redis.get(REDIS_MENTIONS_KEY.format(asset=asset, win="1h")) or 0)
    expected_15m = m1h / 4 if m1h else 0
    velocity = round(m15 / expected_15m, 2) if expected_15m else 0.0

    # top 3 highest-score posts
    top = sorted(records, key=lambda r: r.get("score", 0), reverse=True)[:3]

    return RedditSignal(
        asset=asset,
        window_minutes=window_min,
        mention_count=total,
        bullish_count=bull,
        bearish_count=bear,
        neutral_count=neutral,
        avg_sentiment=avg_sent,
        weighted_sentiment=w_sent,
        bull_ratio=bull_r,
        bear_ratio=bear_r,
        mention_velocity=velocity,
        signal=signal,
        confidence=conf,
        top_posts=top,
    )


async def get_full_snapshot(redis) -> RedditSnapshot:
    """Compute signals for all 7 assets × 4 windows.  Called by scheduler."""
    signals: dict[str, dict[int, RedditSignal]] = {}
    for asset in ASSET_KEYWORDS:
        signals[asset] = {}
        for win in WINDOWS_MINUTES:
            signals[asset][win] = await aggregate_asset(asset, win, redis)

    # Cache the 1h signal in a Redis hash for the API to read instantly
    pipe = redis.pipeline()
    for asset, wins in signals.items():
        sig = wins[60]
        pipe.hset(
            f"reddit:signal:{asset}",
            mapping={
                "signal": sig.signal,
                "weighted_sentiment": sig.weighted_sentiment,
                "confidence": sig.confidence,
                "mention_count": sig.mention_count,
                "bull_ratio": sig.bull_ratio,
                "bear_ratio": sig.bear_ratio,
                "mention_velocity": sig.mention_velocity,
            },
        )
        pipe.expire(f"reddit:signal:{asset}", 600)  # 10-min TTL
    await pipe.execute()

    return RedditSnapshot(signals=signals, computed_at=time.time())


async def get_asset_signal(asset: str, redis) -> dict:
    """
    Fast path — returns the cached 1h signal hash from Redis.
    Used by dashboard SSE push and API endpoint.
    """
    key = f"reddit:signal:{asset}"
    raw = await redis.hgetall(key)
    if not raw:
        return {"signal": "NO_DATA", "confidence": 0.0}
    return {k: (float(v) if k != "signal" else v) for k, v in raw.items()}
