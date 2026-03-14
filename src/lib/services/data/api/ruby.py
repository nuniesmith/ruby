"""
Ruby Signal Engine API Router
==============================
Exposes read-only endpoints for the Ruby Signal Engine results that are
published to Redis by ``handle_ruby_recompute()`` every 5 minutes during
the active trading session.

Endpoints
---------
GET  /api/ruby/signals
    All latest Ruby signals as a JSON map  {symbol: RubySignal}.
    Reads from ``engine:ruby_signals`` (aggregate map, TTL 15 min).

GET  /api/ruby/signal/{symbol}
    Latest Ruby signal for a single symbol.
    Reads from ``engine:ruby_signal:{symbol}`` (TTL 15 min).

GET  /api/ruby/status
    Summary of all available symbols with condensed signal info
    (direction, quality, regime, wave_ratio, breakout_detected).
    Suitable for the dashboard signal-card strip.

GET  /api/ruby/status/html
    HTMX fragment — renders the Ruby signal cards for the Live Trading
    page signal strip.  Refreshed via ``hx-get`` every 30 s.

Architecture
------------
All endpoints are read-only — they consume Redis keys written by the
engine service.  No engine imports are needed at request time, so the
data service can serve these endpoints even when the engine is not
embedded.

If Redis is unavailable, endpoints return a graceful empty response
rather than 500-ing so the dashboard degrades cleanly.

Environment Variables
---------------------
``RUBY_SIGNAL_TTL``   — Override per-signal Redis TTL in seconds (default 900)
"""

from __future__ import annotations

import contextlib
import json
import logging
from datetime import UTC, datetime

from fastapi import APIRouter
from fastapi.responses import HTMLResponse, JSONResponse

logger = logging.getLogger("api.ruby")

router = APIRouter(tags=["Ruby Signal Engine"])

# Redis key constants (must match handlers.py)
_RUBY_SIGNALS_MAP_KEY = "engine:ruby_signals"
_RUBY_SIGNAL_KEY_PREFIX = "engine:ruby_signal:"

# ---------------------------------------------------------------------------
# Redis helper — best-effort, never raises
# ---------------------------------------------------------------------------

with contextlib.suppress(Exception):
    pass


def _cache_get(key: str) -> bytes | None:
    """Best-effort Redis GET — returns None on any error."""
    try:
        from lib.core.cache import cache_get

        return cache_get(key)
    except Exception:
        return None


def _load_signals_map() -> dict[str, dict]:
    """Load the aggregate Ruby signals map from Redis.

    Returns an empty dict if the key is missing or Redis is unavailable.
    """
    raw = _cache_get(_RUBY_SIGNALS_MAP_KEY)
    if not raw:
        return {}
    with contextlib.suppress(Exception):
        decoded = raw.decode("utf-8") if isinstance(raw, bytes) else raw
        return json.loads(decoded)
    return {}


def _load_signal(symbol: str) -> dict | None:
    """Load the latest Ruby signal for a single symbol from Redis."""
    raw = _cache_get(f"{_RUBY_SIGNAL_KEY_PREFIX}{symbol}")
    if not raw:
        return None
    with contextlib.suppress(Exception):
        decoded = raw.decode("utf-8") if isinstance(raw, bytes) else raw
        return json.loads(decoded)
    return None


# ---------------------------------------------------------------------------
# GET /api/ruby/signals
# ---------------------------------------------------------------------------


@router.get("/api/ruby/signals")
async def get_ruby_signals() -> JSONResponse:
    """Return all latest Ruby signals as a JSON object keyed by symbol.

    Response shape::

        {
          "MNQ": { ...RubySignal dict... },
          "NQ":  { ...RubySignal dict... },
          ...
        }

    Returns an empty object ``{}`` when no signals have been computed yet
    (engine not yet run, Redis unavailable, or TTL expired).
    """
    signals = _load_signals_map()
    return JSONResponse(
        content={
            "signals": signals,
            "count": len(signals),
            "fetched_at": datetime.now(UTC).isoformat(),
        }
    )


# ---------------------------------------------------------------------------
# GET /api/ruby/signal/{symbol}
# ---------------------------------------------------------------------------


@router.get("/api/ruby/signal/{symbol}")
async def get_ruby_signal(symbol: str) -> JSONResponse:
    """Return the latest Ruby signal for a single symbol.

    Returns 404 with a graceful JSON body if the signal has not been
    computed yet or has expired.

    Path parameters:
        symbol: Ticker symbol (e.g. ``MNQ``, ``ES``, ``NQ``).
    """
    sig = _load_signal(symbol.upper())
    if sig is None:
        # Try the aggregate map as a fallback
        signals = _load_signals_map()
        sig = signals.get(symbol.upper())

    if sig is None:
        return JSONResponse(
            status_code=404,
            content={
                "detail": f"No Ruby signal available for {symbol.upper()}. "
                "The engine may not have processed this symbol yet.",
                "symbol": symbol.upper(),
            },
        )

    return JSONResponse(content=sig)


# ---------------------------------------------------------------------------
# GET /api/ruby/status
# ---------------------------------------------------------------------------


@router.get("/api/ruby/status")
async def get_ruby_status() -> JSONResponse:
    """Return a condensed status summary for all symbols.

    Suitable for the dashboard signal-card strip — returns only the
    fields needed to render compact signal cards without the full
    level / timestamp detail.

    Response shape::

        {
          "symbols": [
            {
              "symbol":            "MNQ",
              "direction":         "LONG",
              "quality":           75.0,
              "regime":            "TRENDING ↑",
              "wave_ratio":        1.42,
              "bull_bias":         true,
              "vol_regime":        "HIGH",
              "breakout_detected": true,
              "signal_class":      "TG_BOUNCE",
              "bar_time":          "2025-01-15T09:32:00+00:00",
            },
            ...
          ],
          "count": 3,
          "fetched_at": "..."
        }
    """
    signals = _load_signals_map()
    summaries = []

    for symbol, sig in signals.items():
        summaries.append(
            {
                "symbol": symbol,
                "direction": sig.get("direction", ""),
                "quality": sig.get("quality", 0.0),
                "regime": sig.get("regime", "NEUTRAL"),
                "phase": sig.get("phase", "NEUTRAL"),
                "wave_ratio": sig.get("wave_ratio", 1.0),
                "mkt_bias": sig.get("mkt_bias", "Neutral"),
                "bull_bias": sig.get("bull_bias", False),
                "vol_regime": sig.get("vol_regime", "MED"),
                "vol_pct": sig.get("vol_pct", 0.5),
                "ao": sig.get("ao", 0.0),
                "atr_value": sig.get("atr_value", 0.0),
                "breakout_detected": sig.get("breakout_detected", False),
                "signal_class": sig.get("signal_class", ""),
                "cnn_prob": sig.get("cnn_prob", 0.0),
                "entry": sig.get("entry", 0.0),
                "sl": sig.get("sl", 0.0),
                "tp1": sig.get("tp1", 0.0),
                "tp2": sig.get("tp2", 0.0),
                "tp3": sig.get("tp3", 0.0),
                "risk": sig.get("risk", 0.0),
                "tg_hi": sig.get("tg_hi", 0.0),
                "tg_lo": sig.get("tg_lo", 0.0),
                "orb_ready": sig.get("orb_ready", False),
                "orb_high": sig.get("orb_high", 0.0),
                "orb_low": sig.get("orb_low", 0.0),
                "sqz_on": sig.get("sqz_on", False),
                "sqz_fired": sig.get("sqz_fired", False),
                "bar_time": sig.get("bar_time"),
                "computed_at": sig.get("computed_at"),
            }
        )

    # Sort: signals first (breakout_detected), then by quality desc
    summaries.sort(key=lambda s: (-int(s["breakout_detected"]), -s["quality"]))

    return JSONResponse(
        content={
            "symbols": summaries,
            "count": len(summaries),
            "fetched_at": datetime.now(UTC).isoformat(),
        }
    )


# ---------------------------------------------------------------------------
# GET /api/ruby/status/html  — HTMX fragment
# ---------------------------------------------------------------------------

# Direction → CSS colour class (Tailwind-compatible names used by the
# trading.html stylesheet; a simple inline style fallback is also provided
# so the fragment renders even without Tailwind)
_DIR_CLASS = {
    "LONG": "ruby-long",
    "SHORT": "ruby-short",
    "": "ruby-flat",
}
_DIR_EMOJI = {
    "LONG": "📈",
    "SHORT": "📉",
    "": "⬛",
}
_REGIME_EMOJI = {
    "TRENDING ↑": "🚀",
    "TRENDING ↓": "🔻",
    "VOLATILE": "⚡",
    "RANGING": "↔️",
    "NEUTRAL": "—",
}
_QUALITY_CLASS = {
    # quality → CSS class
    "high": "ruby-quality-high",
    "mid": "ruby-quality-mid",
    "low": "ruby-quality-low",
}


def _quality_tier(q: float) -> str:
    if q >= 70:
        return "high"
    if q >= 50:
        return "mid"
    return "low"


def _render_signal_card(sig: dict) -> str:
    """Render a single Ruby signal as an HTML card fragment."""
    symbol = sig.get("symbol", "?")
    direction = sig.get("direction", "")
    quality = float(sig.get("quality", 0.0))
    regime = sig.get("regime", "NEUTRAL")
    wave_ratio = float(sig.get("wave_ratio", 1.0))
    bull_bias = bool(sig.get("bull_bias", False))
    breakout = bool(sig.get("breakout_detected", False))
    signal_class = sig.get("signal_class", "")
    vol_regime = sig.get("vol_regime", "MED")
    sqz_on = bool(sig.get("sqz_on", False))
    sqz_fired = bool(sig.get("sqz_fired", False))
    entry = float(sig.get("entry", 0.0))
    sl = float(sig.get("sl", 0.0))
    tp1 = float(sig.get("tp1", 0.0))
    risk = float(sig.get("risk", 0.0))
    bar_time_raw = sig.get("bar_time", "")
    orb_ready = bool(sig.get("orb_ready", False))

    dir_class = _DIR_CLASS.get(direction, "ruby-flat")
    dir_emoji = _DIR_EMOJI.get(direction, "⬛")
    regime_emoji = _REGIME_EMOJI.get(regime, "—")
    qt_class = _QUALITY_CLASS[_quality_tier(quality)]
    breakout_badge = ""
    if breakout:
        sc_label = signal_class.replace("_", " ") if signal_class else "SIGNAL"
        breakout_badge = f'<span class="ruby-breakout-badge">{dir_emoji} {sc_label}</span>'

    sqz_badge = ""
    if sqz_fired:
        sqz_badge = '<span class="ruby-sqz-badge" title="Squeeze fired this bar">💥 SQZ</span>'
    elif sqz_on:
        sqz_badge = '<span class="ruby-sqz-badge ruby-sqz-on" title="Squeeze building">🔲 SQZ</span>'

    # Format bar time
    bar_time_str = ""
    if bar_time_raw:
        with contextlib.suppress(Exception):
            from datetime import datetime

            dt = datetime.fromisoformat(bar_time_raw)
            bar_time_str = dt.strftime("%H:%M")

    # Levels row (only when breakout detected)
    levels_html = ""
    if breakout and entry > 0:
        levels_html = (
            f'<div class="ruby-card-levels">'
            f'<span title="Entry">E {entry:,.2f}</span>'
            f'<span title="Stop Loss" class="ruby-sl">SL {sl:,.2f}</span>'
            f'<span title="TP1">TP1 {tp1:,.2f}</span>'
            f'<span title="Risk (1R)">1R {risk:,.2f}</span>'
            f"</div>"
        )

    orb_pip = '<span class="ruby-orb-pip" title="ORB range active">ORB</span>' if orb_ready else ""

    bias_arrow = "▲" if bull_bias else "▼"
    bias_class = "ruby-bias-bull" if bull_bias else "ruby-bias-bear"

    return (
        f'<div class="ruby-signal-card {dir_class}" data-symbol="{symbol}">'
        f'  <div class="ruby-card-header">'
        f'    <span class="ruby-card-symbol">{symbol}</span>'
        f"    {breakout_badge}"
        f"    {sqz_badge}"
        f"    {orb_pip}"
        f"  </div>"
        f'  <div class="ruby-card-row">'
        f'    <span class="ruby-quality {qt_class}" title="Quality score">'
        f"      Q {quality:.0f}%</span>"
        f'    <span class="ruby-regime" title="Market regime">'
        f"      {regime_emoji} {regime}</span>"
        f"  </div>"
        f'  <div class="ruby-card-row">'
        f'    <span class="ruby-wave" title="Wave ratio">'
        f"      Wave {wave_ratio:.2f}x</span>"
        f'    <span class="{bias_class}" title="Session bias">'
        f"      {bias_arrow} {sig.get('mkt_bias', 'Neutral')}</span>"
        f'    <span class="ruby-vol" title="Volatility regime">Vol {vol_regime}</span>'
        f"  </div>"
        f"  {levels_html}"
        f'  <div class="ruby-card-footer">'
        f'    <span class="ruby-bar-time">{bar_time_str}</span>'
        f"  </div>"
        f"</div>"
    )


def _render_empty_state() -> str:
    """Return an empty-state placeholder when no signals are available."""
    return (
        '<div class="ruby-empty-state">'
        '  <span class="ruby-empty-icon">📡</span>'
        '  <span class="ruby-empty-msg">Ruby engine warming up — '
        "signals will appear once bars are processed.</span>"
        "</div>"
    )


@router.get("/api/ruby/status/html", response_class=HTMLResponse)
async def get_ruby_status_html() -> HTMLResponse:
    """HTMX fragment — Ruby signal cards for the Live Trading page.

    Returns an HTML fragment (no ``<html>`` wrapper) containing one
    ``.ruby-signal-card`` div per symbol.  The fragment is designed to
    be injected into ``#ruby-signal-strip`` via HTMX::

        <div id="ruby-signal-strip"
             hx-get="/api/ruby/status/html"
             hx-trigger="every 30s"
             hx-swap="innerHTML">
        </div>

    Inline ``<style>`` is included on the first render so the cards are
    styled even before the page CSS loads.  The styles are scoped to the
    ``.ruby-*`` namespace so they don't conflict with existing dashboard CSS.
    """
    signals = _load_signals_map()

    style_block = """
<style>
  .ruby-signal-strip { display:flex; flex-wrap:wrap; gap:8px; padding:8px 0; }
  .ruby-signal-card {
    background:#1e222d; border:1px solid #333; border-radius:6px;
    padding:8px 10px; min-width:180px; font-size:12px; color:#ccc;
    display:flex; flex-direction:column; gap:4px;
  }
  .ruby-long  { border-left:3px solid #00e676; }
  .ruby-short { border-left:3px solid #ff5252; }
  .ruby-flat  { border-left:3px solid #555; }
  .ruby-card-header { display:flex; align-items:center; gap:6px; flex-wrap:wrap; }
  .ruby-card-symbol { font-weight:700; color:#fff; font-size:13px; }
  .ruby-breakout-badge {
    background:#2a3a2a; color:#00e676; border:1px solid #00e676;
    border-radius:3px; padding:1px 5px; font-size:10px; font-weight:600;
  }
  .ruby-short .ruby-breakout-badge { background:#3a2a2a; color:#ff5252; border-color:#ff5252; }
  .ruby-sqz-badge {
    background:#3a3a1a; color:#ffd600; border:1px solid #ffd600;
    border-radius:3px; padding:1px 5px; font-size:10px;
  }
  .ruby-sqz-on { color:#ff9800; border-color:#ff9800; background:#3a2a1a; }
  .ruby-orb-pip {
    background:#1a2a3a; color:#2196f3; border:1px solid #2196f3;
    border-radius:3px; padding:1px 5px; font-size:10px;
  }
  .ruby-card-row { display:flex; gap:8px; align-items:center; flex-wrap:wrap; }
  .ruby-quality { font-weight:600; }
  .ruby-quality-high { color:#00e676; }
  .ruby-quality-mid  { color:#ffd600; }
  .ruby-quality-low  { color:#888; }
  .ruby-regime { color:#aaa; }
  .ruby-wave { color:#64b5f6; }
  .ruby-bias-bull { color:#00e676; }
  .ruby-bias-bear { color:#ff5252; }
  .ruby-vol { color:#888; font-size:11px; }
  .ruby-card-levels {
    display:flex; gap:8px; font-size:11px; flex-wrap:wrap;
    border-top:1px solid #333; padding-top:4px; margin-top:2px;
  }
  .ruby-card-levels span { color:#aaa; }
  .ruby-sl { color:#ff5252 !important; }
  .ruby-card-footer { font-size:10px; color:#555; }
  .ruby-bar-time { font-family:monospace; }
  .ruby-empty-state {
    display:flex; align-items:center; gap:8px; color:#555;
    font-size:13px; padding:12px;
  }
  .ruby-empty-icon { font-size:20px; }
</style>
"""

    if not signals:
        return HTMLResponse(content=style_block + _render_empty_state())

    # Sort: breakout signals first, then by quality descending
    sorted_sigs = sorted(
        signals.values(),
        key=lambda s: (-int(bool(s.get("breakout_detected"))), -float(s.get("quality", 0.0))),
    )

    cards_html = "\n".join(_render_signal_card(s) for s in sorted_sigs)
    fragment = (
        style_block + f'<div class="ruby-signal-strip" '
        f'     data-updated="{datetime.now(UTC).strftime("%H:%M:%S")}">\n' + cards_html + "\n</div>"
    )
    return HTMLResponse(content=fragment)
