"""
src/lib/services/data/api/news.py
──────────────────────────────────
FastAPI router for news sentiment data.

Mirrors the pattern of reddit.py — JSON REST endpoints + HTMX fragments.

Routes
──────
    GET /api/news/sentiment?symbols=MES,MGC,MCL
        → aggregated NewsSentiment per symbol (JSON)

    GET /api/news/sentiment/{symbol}
        → single-symbol NewsSentiment (JSON)

    GET /api/news/headlines?symbol=MES&limit=10
        → recent headlines with sentiment scores (JSON)

    GET /api/news/spike
        → current spiking symbols (JSON)

    GET /htmx/news/panel
        → full news sentiment panel HTML fragment (for dashboard)

    GET /htmx/news/asset/{symbol}
        → single-asset headline card HTML fragment

Register in main app (data service):
    from lib.services.data.api.news import router as news_router
    app.include_router(news_router, tags=["News"])
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

logger = logging.getLogger("api.news")

router = APIRouter()

# Symbols the scheduler runs sentiment for — matches engine watchlist defaults.
_DEFAULT_SYMBOLS = [
    "MES",
    "MNQ",
    "MGC",
    "MCL",
    "M2K",
    "MYM",
    "M6E",
    "M6B",
    "MBT",
    "MET",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_redis(request: Request):
    """Return the Redis client from app state (or None)."""
    try:
        return request.app.state.redis
    except AttributeError:
        return None


def _load_sentiment(symbol: str, redis: Any) -> dict | None:
    """Read one cached NewsSentiment from Redis, return as plain dict."""
    if redis is None:
        return None
    try:
        from lib.analysis.sentiment.news_sentiment import load_sentiment_from_cache

        ns = load_sentiment_from_cache(symbol, redis)
        if ns is None:
            return None
        d = asdict(ns)
        # Convert datetime to ISO string for JSON serialisation
        if hasattr(d.get("computed_at"), "isoformat"):
            d["computed_at"] = d["computed_at"].isoformat()
        return d
    except Exception as exc:
        logger.debug("_load_sentiment(%s): %s", symbol, exc)
        return None


def _signal_color(signal: str) -> str:
    return {
        "STRONG_BULL": "#00c853",
        "BULL": "#69f0ae",
        "NEUTRAL": "#90a4ae",
        "BEAR": "#ff6d00",
        "STRONG_BEAR": "#d50000",
    }.get(signal, "#90a4ae")


def _signal_emoji(signal: str) -> str:
    return {
        "STRONG_BULL": "🟢🟢",
        "BULL": "🟢",
        "NEUTRAL": "⚪",
        "BEAR": "🔴",
        "STRONG_BEAR": "🔴🔴",
    }.get(signal, "⚪")


def _signal_badge(signal: str) -> str:
    """Return a small HTML badge for a signal label."""
    color = _signal_color(signal)
    emoji = _signal_emoji(signal)
    label = signal.replace("_", " ")
    return f'<span style="color:{color};font-weight:600">{emoji} {label}</span>'


def _format_score(score: float) -> str:
    """Format a [-1, +1] score as a coloured string."""
    color = "#69f0ae" if score > 0.1 else "#f87171" if score < -0.1 else "#90a4ae"
    sign = "+" if score > 0 else ""
    return f'<span style="color:{color}">{sign}{score:.2f}</span>'


# ---------------------------------------------------------------------------
# JSON / REST endpoints
# ---------------------------------------------------------------------------


@router.get("/api/news/sentiment")
async def api_news_sentiment(request: Request, symbols: str = ""):
    """Aggregated NewsSentiment for one or more symbols.

    Query params:
        symbols  — comma-separated futures tickers, e.g. ``MES,MGC,MCL``.
                   Defaults to the full engine watchlist.

    Returns a dict of ``{symbol: NewsSentiment}``.
    Missing / uncached symbols are omitted.
    """
    sym_list = [s.strip().upper() for s in symbols.split(",") if s.strip()] if symbols else _DEFAULT_SYMBOLS
    redis = _get_redis(request)

    results: dict[str, Any] = {}
    for sym in sym_list:
        data = _load_sentiment(sym, redis)
        if data is not None:
            results[sym] = data

    return {
        "symbols": sym_list,
        "cached": list(results.keys()),
        "sentiments": results,
    }


@router.get("/api/news/sentiment/{symbol}")
async def api_news_sentiment_single(symbol: str, request: Request):
    """Latest cached NewsSentiment for a single symbol.

    Returns the NewsSentiment dict, or ``{"error": "..."}`` if not cached.
    """
    symbol = symbol.upper()
    redis = _get_redis(request)
    data = _load_sentiment(symbol, redis)
    if data is None:
        return {
            "error": f"No cached sentiment for {symbol}. "
            "The engine scheduler runs the news pipeline at 07:00 ET and 12:00 ET."
        }
    return data


@router.get("/api/news/headlines")
async def api_news_headlines(request: Request, symbol: str = "", limit: int = 10):
    """Recent headlines with sentiment scores for a symbol.

    Query params:
        symbol  — futures ticker (e.g. ``MES``).  Required.
        limit   — max number of headlines to return (default 10, max 50).

    Returns a list of headline dicts with keys:
        headline, summary, url, source, published_at,
        vader_score, av_score, grok_score, hybrid_score,
        grok_label, grok_reason.
    """
    if not symbol:
        return {"error": "symbol query parameter is required"}

    symbol = symbol.upper()
    limit = max(1, min(limit, 50))
    redis = _get_redis(request)
    data = _load_sentiment(symbol, redis)

    if data is None:
        return {
            "symbol": symbol,
            "headlines": [],
            "error": "No cached data — engine news pipeline has not run yet today.",
        }

    headlines = data.get("top_headlines", [])
    return {
        "symbol": symbol,
        "total": len(headlines),
        "headlines": headlines[:limit],
        "computed_at": data.get("computed_at"),
    }


@router.get("/api/news/spike")
async def api_news_spike(request: Request):
    """Return the latest news-spike alert dict from Redis.

    The engine publishes to ``engine:news_spike`` whenever article volume
    for a symbol exceeds 3× its rolling average in one hour.

    Returns ``{"spikes": {symbol: {...}}}`` or ``{"spikes": {}}`` if quiet.
    """
    redis = _get_redis(request)
    if redis is None:
        return {"spikes": {}, "error": "Redis unavailable"}

    try:
        from lib.analysis.sentiment.news_sentiment import REDIS_SPIKE_KEY

        raw = redis.get(REDIS_SPIKE_KEY)
        if not raw:
            return {"spikes": {}}
        spikes = json.loads(raw.decode() if isinstance(raw, bytes) else raw)
        return {"spikes": spikes}
    except Exception as exc:
        logger.debug("api_news_spike: %s", exc)
        return {"spikes": {}, "error": str(exc)}


# ---------------------------------------------------------------------------
# HTMX fragment endpoints
# ---------------------------------------------------------------------------


@router.get("/htmx/news/panel", response_class=HTMLResponse)
async def htmx_news_panel(request: Request):
    """Full news sentiment panel — all watchlist assets at a glance.

    Usage in dashboard:
        hx-get="/htmx/news/panel" hx-trigger="every 120s" hx-swap="outerHTML"
    """
    redis = _get_redis(request)

    rows_html = ""
    has_data = False

    for sym in _DEFAULT_SYMBOLS:
        data = _load_sentiment(sym, redis)
        if data is None:
            continue
        has_data = True

        signal = data.get("signal", "NEUTRAL")
        score = data.get("weighted_hybrid", 0.0)
        article_count = data.get("article_count", 0)
        is_spike = data.get("is_spike", False)
        confidence = data.get("confidence", 0.0)
        color = _signal_color(signal)
        emoji = _signal_emoji(signal)

        spike_badge = '<span style="color:#fb923c;font-size:0.7rem;margin-left:4px">📰 SPIKE</span>' if is_spike else ""

        score_color = "#69f0ae" if score > 0.1 else "#f87171" if score < -0.1 else "#90a4ae"
        sign = "+" if score > 0 else ""

        rows_html += f"""
        <tr style="border-bottom:1px solid #2a2a3e">
          <td style="padding:5px 8px;font-weight:600">{sym}{spike_badge}</td>
          <td style="padding:5px 8px">
            <span style="color:{color}">{emoji} {signal.replace("_", " ")}</span>
          </td>
          <td style="padding:5px 8px;color:{score_color};font-family:monospace">
            {sign}{score:.2f}
          </td>
          <td style="padding:5px 8px;color:#888;font-size:0.8rem">
            {article_count} arts · {confidence:.0%}
          </td>
        </tr>"""

    if not has_data:
        rows_html = """
        <tr>
          <td colspan="4" style="padding:16px 8px;text-align:center;color:#666;font-size:0.85rem">
            No news data yet — engine runs pipeline at 07:00 ET &amp; 12:00 ET
          </td>
        </tr>"""

    return HTMLResponse(f"""
    <div id="news-panel"
         style="font-family:monospace;background:#1a1a2e;border-radius:8px;
                padding:12px;color:#e0e0e0">
      <div style="display:flex;justify-content:space-between;align-items:center;
                  margin-bottom:10px">
        <span style="font-size:1.1rem;font-weight:700">📰 News Sentiment</span>
        <span style="font-size:0.75rem;color:#888">VADER + AV + Grok · refreshes 2m</span>
      </div>
      <table style="width:100%;border-collapse:collapse">
        <thead>
          <tr style="color:#888;font-size:0.75rem">
            <th style="text-align:left;padding:2px 8px">Asset</th>
            <th style="text-align:left;padding:2px 8px">Signal</th>
            <th style="text-align:left;padding:2px 8px">Score</th>
            <th style="text-align:left;padding:2px 8px">Coverage</th>
          </tr>
        </thead>
        <tbody>
          {rows_html}
        </tbody>
      </table>
    </div>
    """)


@router.get("/htmx/news/asset/{symbol}", response_class=HTMLResponse)
async def htmx_news_asset_card(symbol: str, request: Request, limit: int = 8):
    """Detailed news card for a single asset — top headlines + scores.

    Usage:
        hx-get="/htmx/news/asset/MES?limit=8" hx-trigger="every 120s"
    """
    symbol = symbol.upper()
    limit = max(1, min(limit, 20))
    redis = _get_redis(request)
    data = _load_sentiment(symbol, redis)

    if data is None:
        return HTMLResponse(f"""
        <div style="font-family:monospace;background:#1a1a2e;border-radius:8px;
                    padding:14px;color:#666;font-size:0.85rem">
          No news data for {symbol} — engine runs pipeline at 07:00 ET &amp; 12:00 ET
        </div>
        """)

    signal = data.get("signal", "NEUTRAL")
    score = data.get("weighted_hybrid", 0.0)
    article_count = data.get("article_count", 0)
    is_spike = data.get("is_spike", False)
    articles_last_hour = data.get("articles_last_hour", 0)
    grok_narrative = data.get("grok_narrative") or ""
    computed_at = data.get("computed_at", "")[:16].replace("T", " ")

    color = _signal_color(signal)
    emoji = _signal_emoji(signal)
    sign = "+" if score > 0 else ""
    score_color = "#69f0ae" if score > 0.1 else "#f87171" if score < -0.1 else "#90a4ae"

    spike_banner = ""
    if is_spike:
        spike_banner = f"""
        <div style="background:#7c2d12;border-radius:4px;padding:6px 10px;
                    margin-bottom:10px;font-size:0.8rem;color:#fcd34d">
          📰 NEWS SPIKE — {articles_last_hour} articles in last hour
        </div>"""

    # Headline rows
    headlines = data.get("top_headlines", [])[:limit]
    headline_rows = ""
    for h in headlines:
        h_signal = h.get("grok_label") or (
            "Bullish" if h.get("hybrid_score", 0) > 0.1 else "Bearish" if h.get("hybrid_score", 0) < -0.1 else "Neutral"
        )
        h_color = "#69f0ae" if "bull" in h_signal.lower() else "#f87171" if "bear" in h_signal.lower() else "#90a4ae"
        h_score = h.get("hybrid_score", 0.0)
        h_sign = "+" if h_score > 0 else ""
        h_title = str(h.get("headline", ""))[:90]
        h_source = h.get("source", "")
        h_reason = h.get("grok_reason", "")
        tooltip = f' title="{h_reason}"' if h_reason else ""
        headline_rows += f"""
        <li style="margin:5px 0;font-size:0.78rem;list-style:none;
                   border-left:2px solid {h_color};padding-left:6px">
          <span style="color:{h_color};font-weight:600"
                {tooltip}>[{h_signal[:4].upper()}]</span>
          <span style="color:#ccc"> {h_title}</span>
          <span style="color:#555;margin-left:4px">
            · {h_source}
            · <span style="color:{h_color};font-family:monospace">{h_sign}{h_score:.2f}</span>
          </span>
        </li>"""

    if not headline_rows:
        headline_rows = '<li style="color:#666;font-size:0.8rem;list-style:none">No headlines cached</li>'

    grok_block = ""
    if grok_narrative:
        grok_block = f"""
        <div style="margin-top:10px;background:#12122a;border-radius:4px;
                    padding:8px 10px;font-size:0.78rem;color:#b0b8d0;
                    border-left:3px solid #6366f1">
          <div style="font-weight:600;color:#818cf8;margin-bottom:4px">🤖 Grok Narrative</div>
          {grok_narrative[:400]}
        </div>"""

    return HTMLResponse(f"""
    <div style="font-family:monospace;background:#1a1a2e;border-radius:8px;
                padding:14px;color:#e0e0e0;min-width:320px">
      <div style="display:flex;justify-content:space-between;align-items:center;
                  margin-bottom:10px">
        <span style="font-size:1.1rem;font-weight:700">{symbol} News Sentiment</span>
        <span style="font-size:0.7rem;color:#666">{computed_at}</span>
      </div>

      {spike_banner}

      <div style="display:flex;gap:16px;margin-bottom:10px;font-size:0.9rem">
        <div>
          <div style="color:#888;font-size:0.7rem">SIGNAL</div>
          <div style="color:{color};font-weight:700">{emoji} {signal.replace("_", " ")}</div>
        </div>
        <div>
          <div style="color:#888;font-size:0.7rem">HYBRID SCORE</div>
          <div style="color:{score_color};font-family:monospace;font-weight:700">
            {sign}{score:.3f}
          </div>
        </div>
        <div>
          <div style="color:#888;font-size:0.7rem">ARTICLES</div>
          <div style="color:#ccc">{article_count}</div>
        </div>
      </div>

      <div style="color:#888;font-size:0.75rem;margin-bottom:6px">
        Top Headlines
      </div>
      <ul style="margin:0;padding:0">
        {headline_rows}
      </ul>

      {grok_block}
    </div>
    """)
