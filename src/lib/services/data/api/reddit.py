"""
src/lib/services/data/api/reddit.py
─────────────────────────────────────
FastAPI router for Reddit sentiment data.
Mirrors the pattern of analysis.py / journal.py.

Register in your main app:
    from lib.services.data.api.reddit import router as reddit_router
    app.include_router(reddit_router, prefix="/reddit", tags=["reddit"])

HTMX endpoints return HTML fragments.
JSON endpoints are prefixed /api/reddit/.
"""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

from lib.analysis.sentiment.reddit_sentiment import (
    WINDOWS_MINUTES,
    aggregate_asset,
    get_asset_signal,
    get_full_snapshot,
)
from lib.integrations.reddit_watcher import ASSET_KEYWORDS

router = APIRouter()


# ─────────────────────────────────────────────────────────────────────────────
# JSON / REST endpoints
# ─────────────────────────────────────────────────────────────────────────────


@router.get("/api/reddit/signal/{asset}")
async def api_signal(asset: str, request: Request):
    """Latest 1-hour aggregated signal for a single asset."""
    asset = asset.upper()
    if asset not in ASSET_KEYWORDS:
        return {"error": f"Unknown asset {asset}"}
    redis = request.app.state.redis
    return await get_asset_signal(asset, redis)


@router.get("/api/reddit/signal/{asset}/{window_min}")
async def api_signal_window(asset: str, window_min: int, request: Request):
    """Aggregated signal for a specific time window (15, 60, 240, 1440 min)."""
    asset = asset.upper()
    if asset not in ASSET_KEYWORDS:
        return {"error": f"Unknown asset {asset}"}
    if window_min not in WINDOWS_MINUTES:
        return {"error": f"window_min must be one of {WINDOWS_MINUTES}"}
    redis = request.app.state.redis
    from dataclasses import asdict

    sig = await aggregate_asset(asset, window_min, redis)
    return asdict(sig)


@router.get("/api/reddit/snapshot")
async def api_snapshot(request: Request):
    """Full snapshot — all assets, all windows."""
    from dataclasses import asdict

    snapshot = await get_full_snapshot(request.app.state.redis)
    # serialise nested dataclasses
    return {
        "computed_at": snapshot.computed_at,
        "signals": {
            asset: {str(win): asdict(sig) for win, sig in wins.items()} for asset, wins in snapshot.signals.items()
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# HTMX fragments — drop these into your dashboard templates
# ─────────────────────────────────────────────────────────────────────────────


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


@router.get("/htmx/reddit/panel", response_class=HTMLResponse)
async def htmx_reddit_panel(request: Request):
    """
    Full Reddit Sentiment panel — all assets at a glance.
    Use with:  hx-get="/htmx/reddit/panel" hx-trigger="every 120s"
    """
    redis = request.app.state.redis
    rows = []
    for asset in ASSET_KEYWORDS:
        sig = await get_asset_signal(asset, redis)
        signal = sig.get("signal", "NO_DATA")
        color = _signal_color(signal)
        emoji = _signal_emoji(signal)
        conf = float(sig.get("confidence", 0))
        ment = int(sig.get("mention_count", 0))
        vel = float(sig.get("mention_velocity", 0))
        ws = float(sig.get("weighted_sentiment", 0))
        vel_badge = '<span style="color:#ff9800;font-size:0.75rem">🔥 spike</span>' if vel > 1.5 else ""
        rows.append(f"""
        <tr>
          <td style="font-weight:600;padding:6px 10px">{asset}</td>
          <td style="color:{color};padding:6px 10px">{emoji} {signal}</td>
          <td style="padding:6px 10px">{ws:+.3f}</td>
          <td style="padding:6px 10px">{ment} {vel_badge}</td>
          <td style="padding:6px 10px">{conf:.0%}</td>
        </tr>""")

    return HTMLResponse(f"""
    <div id="reddit-panel" style="font-family:monospace;background:#1a1a2e;
         border-radius:8px;padding:12px;color:#e0e0e0">
      <div style="display:flex;justify-content:space-between;align-items:center;
                  margin-bottom:10px">
        <span style="font-size:1.1rem;font-weight:700">📡 Reddit Sentiment</span>
        <span style="font-size:0.75rem;color:#888">1h window · refreshes 2m</span>
      </div>
      <table style="width:100%;border-collapse:collapse">
        <thead>
          <tr style="color:#888;font-size:0.8rem">
            <th style="text-align:left;padding:4px 10px">Asset</th>
            <th style="text-align:left;padding:4px 10px">Signal</th>
            <th style="text-align:left;padding:4px 10px">W.Sent</th>
            <th style="text-align:left;padding:4px 10px">Mentions</th>
            <th style="text-align:left;padding:4px 10px">Conf</th>
          </tr>
        </thead>
        <tbody>{"".join(rows)}</tbody>
      </table>
    </div>
    """)


@router.get("/htmx/reddit/asset/{asset}", response_class=HTMLResponse)
async def htmx_asset_card(asset: str, request: Request):
    """
    Detailed card for a single asset — multi-window breakdown.
    Use with:  hx-get="/htmx/reddit/asset/NQ" hx-trigger="every 120s"
    """
    asset = asset.upper()
    if asset not in ASSET_KEYWORDS:
        return HTMLResponse(f"<div>Unknown asset: {asset}</div>")

    redis = request.app.state.redis
    win_rows = []
    for win in WINDOWS_MINUTES:
        sig = await aggregate_asset(asset, win, redis)
        color = _signal_color(sig.signal)
        label = {15: "15m", 60: "1h", 240: "4h", 1440: "24h"}[win]
        win_rows.append(f"""
        <tr>
          <td style="padding:4px 8px;color:#aaa">{label}</td>
          <td style="color:{color};padding:4px 8px">{_signal_emoji(sig.signal)} {sig.signal}</td>
          <td style="padding:4px 8px">{sig.weighted_sentiment:+.3f}</td>
          <td style="padding:4px 8px">{sig.mention_count}</td>
          <td style="padding:4px 8px">{sig.bull_ratio:.0%} 🟢 / {sig.bear_ratio:.0%} 🔴</td>
        </tr>""")

    # top posts from 1h window
    sig_1h = await aggregate_asset(asset, 60, redis)
    post_rows = ""
    for p in sig_1h.top_posts:
        c = _signal_color(p.get("label", "neutral").upper().replace("BULLISH", "BULL").replace("BEARISH", "BEAR"))
        post_rows += f"""
        <li style="margin:4px 0;font-size:0.8rem">
          <span style="color:{c}">[{p.get("label", "?").upper()[:4]}]</span>
          <a href="#" style="color:#90caf9">{p.get("title", "")[:80]}</a>
          <span style="color:#666"> r/{p.get("subreddit", "")} · ↑{p.get("score", 0)}</span>
        </li>"""

    return HTMLResponse(f"""
    <div style="font-family:monospace;background:#1a1a2e;border-radius:8px;
                padding:14px;color:#e0e0e0;min-width:320px">
      <div style="font-size:1.1rem;font-weight:700;margin-bottom:10px">{asset} Reddit Sentiment</div>
      <table style="width:100%;border-collapse:collapse;margin-bottom:12px">
        <thead>
          <tr style="color:#888;font-size:0.75rem">
            <th style="text-align:left;padding:2px 8px">Win</th>
            <th style="text-align:left;padding:2px 8px">Signal</th>
            <th style="text-align:left;padding:2px 8px">W.Sent</th>
            <th style="text-align:left;padding:2px 8px">n</th>
            <th style="text-align:left;padding:2px 8px">Bull/Bear</th>
          </tr>
        </thead>
        <tbody>{"".join(win_rows)}</tbody>
      </table>
      <div style="font-size:0.8rem;color:#aaa;margin-bottom:4px">Top posts (1h)</div>
      <ul style="list-style:none;padding:0;margin:0">{post_rows or '<li style="color:#555">No posts yet</li>'}</ul>
    </div>
    """)
