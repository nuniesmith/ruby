"""
Kraken Exchange API Router
===========================
Provides endpoints for Kraken crypto market data, health status,
WebSocket feed management, private account data, and dashboard charts.

Crypto pairs are sourced from the Kraken exchange via REST + WebSocket
and use a ``KRAKEN:`` prefix internally (e.g. ``KRAKEN:XBTUSD``).

Endpoints:
    GET  /kraken/health            — Kraken connectivity + auth status
    GET  /kraken/status            — WebSocket feed status + pair list
    GET  /kraken/pairs             — Available Kraken pairs and their mappings
    GET  /kraken/ticker/{pair}     — Current ticker snapshot for a pair
    GET  /kraken/tickers           — All tracked pair tickers in one call
    GET  /kraken/ohlcv/{pair}      — Historical OHLCV bars for a pair
    GET  /kraken/health/html       — Dashboard HTML fragment for Kraken status
    GET  /kraken/chart/html        — Live candlestick chart HTML fragment
    GET  /kraken/account/html      — Private account (balance/orders) HTML fragment
    GET  /kraken/correlation/html  — Crypto/futures correlation panel HTML fragment
"""

from __future__ import annotations

import logging
import math
from datetime import UTC, datetime
from typing import Any
from zoneinfo import ZoneInfo

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import HTMLResponse

logger = logging.getLogger("api.kraken")

_EST = ZoneInfo("America/New_York")

router = APIRouter(tags=["Kraken"])


def _now_et() -> datetime:
    return datetime.now(tz=_EST)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_provider():
    """Return the Kraken provider singleton, or None."""
    try:
        from lib.integrations.kraken_client import get_kraken_provider

        return get_kraken_provider()
    except ImportError:
        return None


def _get_feed():
    """Return the Kraken feed manager singleton, or None."""
    try:
        from lib.integrations.kraken_client import get_kraken_feed

        return get_kraken_feed()
    except ImportError:
        return None


def _is_enabled() -> bool:
    """Return True if Kraken crypto is enabled in the config."""
    try:
        from lib.core.models import ENABLE_KRAKEN_CRYPTO

        return ENABLE_KRAKEN_CRYPTO
    except ImportError:
        return False


def _get_pairs() -> dict[str, dict[str, str]]:
    """Return the KRAKEN_PAIRS mapping."""
    try:
        from lib.integrations.kraken_client import KRAKEN_PAIRS

        return KRAKEN_PAIRS
    except ImportError:
        return {}


# ---------------------------------------------------------------------------
# JSON endpoints
# ---------------------------------------------------------------------------


@router.get("/kraken/health")
def kraken_health():
    """Kraken connectivity and configuration health check.

    Returns:
        - ``enabled``: whether ENABLE_KRAKEN_CRYPTO is set
        - ``api_available``: whether the REST client is initialized
        - ``authenticated``: whether API key + secret are configured
        - ``connected``: whether a ping to Kraken API succeeded
        - ``ws_feed``: WebSocket feed status summary
        - ``pairs_count``: number of tracked pairs
    """
    enabled = _is_enabled()
    provider = _get_provider()
    feed = _get_feed()

    result: dict[str, Any] = {
        "enabled": enabled,
        "api_available": False,
        "authenticated": False,
        "connected": False,
        "server_time": None,
        "error": None,
        "ws_feed": {
            "running": False,
            "connected": False,
            "pairs": 0,
        },
        "pairs_count": len(_get_pairs()),
        "timestamp": _now_et().isoformat(),
    }

    if not enabled:
        result["error"] = "Kraken crypto disabled (ENABLE_KRAKEN_CRYPTO=0)"
        return result

    if provider is not None:
        result["api_available"] = provider.is_available
        result["authenticated"] = provider.has_auth

        if provider.is_available:
            try:
                health = provider.health()
                result["connected"] = health.get("connected", False)
                result["server_time"] = health.get("server_time")
                result["error"] = health.get("error")
            except Exception as exc:
                result["error"] = str(exc)
    else:
        result["error"] = "Kraken client module not available"

    if feed is not None:
        result["ws_feed"] = {
            "running": feed.is_running,
            "connected": feed.is_connected,
            "pairs": len(feed._pairs),
            "msg_count": feed.msg_count,
            "bar_count": feed.bar_count,
            "trade_count": feed.trade_count,
        }

    return result


@router.get("/kraken/status")
def kraken_status():
    """Detailed Kraken WebSocket feed status.

    Returns the full feed manager status dict including per-pair
    stats, reconnect counts, uptime, and recent errors.
    """
    feed = _get_feed()
    if feed is None:
        return {
            "status": "not_running",
            "enabled": _is_enabled(),
            "message": "Kraken WebSocket feed is not running.",
            "timestamp": _now_et().isoformat(),
        }

    return {
        "status": "ok" if feed.is_connected else "disconnected",
        **feed.get_status(),
        "timestamp": _now_et().isoformat(),
    }


@router.get("/kraken/pairs")
def kraken_pairs():
    """Return all configured Kraken pairs and their internal mappings.

    Each pair entry includes:
      - ``rest_pair``: Kraken REST API pair name (e.g. "XXBTZUSD")
      - ``ws_pair``: WebSocket v2 pair name (e.g. "XBT/USD")
      - ``internal_ticker``: Internal pipeline ticker (e.g. "KRAKEN:XBTUSD")
      - ``base`` / ``quote``: Base and quote currencies
    """
    pairs = _get_pairs()
    return {
        "enabled": _is_enabled(),
        "pairs": {name: {**info} for name, info in pairs.items()},
        "count": len(pairs),
        "timestamp": _now_et().isoformat(),
    }


@router.get("/kraken/ticker/{pair}")
def kraken_ticker(pair: str):
    """Get current ticker snapshot for a Kraken pair.

    Args:
        pair: Internal ticker (e.g. "KRAKEN:XBTUSD") or display name
              (e.g. "Bitcoin", "BTC/USD").

    Returns a normalized snapshot with last_price, bid, ask, volume,
    high/low, vwap, change_pct, etc.
    """
    # Resolve the pair to an internal ticker
    internal = _resolve_internal_ticker(pair)
    if internal is None:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown Kraken pair: '{pair}'. Use /kraken/pairs for valid options.",
        )

    try:
        from lib.integrations.kraken_client import get_kraken_snapshot

        snapshot = get_kraken_snapshot(internal)
        if not snapshot:
            raise HTTPException(status_code=502, detail=f"No ticker data returned for {internal}")
        return snapshot
    except ImportError as exc:
        raise HTTPException(status_code=501, detail="Kraken client not available") from exc
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Kraken ticker error for %s: %s", pair, exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/kraken/tickers")
def kraken_tickers():
    """Get ticker snapshots for all tracked Kraken pairs in one call.

    Returns a dict mapping display name → snapshot dict.
    """
    pairs = _get_pairs()
    if not pairs:
        return {"tickers": {}, "count": 0, "timestamp": _now_et().isoformat()}

    provider = _get_provider()
    if provider is None or not provider.is_available:
        return {
            "tickers": {},
            "count": 0,
            "error": "Kraken provider not available",
            "timestamp": _now_et().isoformat(),
        }

    # Fetch all tickers in a single API call
    try:
        rest_pairs = [p["rest_pair"] for p in pairs.values()]
        raw = provider.get_all_tickers(rest_pairs)
    except Exception as exc:
        logger.error("Kraken bulk ticker error: %s", exc)
        return {
            "tickers": {},
            "count": 0,
            "error": str(exc),
            "timestamp": _now_et().isoformat(),
        }

    # Parse each result into a normalized snapshot
    try:
        from lib.integrations.kraken_client import REST_PAIR_TO_NAME
    except ImportError:
        REST_PAIR_TO_NAME = {}

    tickers: dict[str, dict[str, Any]] = {}
    for rest_key, ticker_data in raw.items():
        name = REST_PAIR_TO_NAME.get(rest_key)
        if name is None:
            # Try matching by checking if the rest_key starts with any known pair
            for pname, pinfo in pairs.items():
                if rest_key == pinfo["rest_pair"] or rest_key.startswith(pinfo["rest_pair"][:4]):
                    name = pname
                    break
        if name is None:
            continue

        try:
            last = float(ticker_data.get("c", [0])[0]) if isinstance(ticker_data.get("c"), list) else 0.0
            bid = float(ticker_data.get("b", [0])[0]) if isinstance(ticker_data.get("b"), list) else 0.0
            ask = float(ticker_data.get("a", [0])[0]) if isinstance(ticker_data.get("a"), list) else 0.0
            volume = float(ticker_data.get("v", [0, 0])[1]) if isinstance(ticker_data.get("v"), list) else 0.0
            high = float(ticker_data.get("h", [0, 0])[1]) if isinstance(ticker_data.get("h"), list) else 0.0
            low = float(ticker_data.get("l", [0, 0])[1]) if isinstance(ticker_data.get("l"), list) else 0.0
            open_price = float(ticker_data.get("o", 0))
            vwap = float(ticker_data.get("p", [0, 0])[1]) if isinstance(ticker_data.get("p"), list) else 0.0

            change_pct = 0.0
            if open_price > 0:
                change_pct = round((last - open_price) / open_price * 100, 3)

            tickers[name] = {
                "internal_ticker": pairs[name]["internal_ticker"],
                "last_price": last,
                "bid": bid,
                "ask": ask,
                "spread": round(ask - bid, 6) if ask and bid else 0.0,
                "volume_24h": volume,
                "high_24h": high,
                "low_24h": low,
                "open_24h": open_price,
                "vwap_24h": vwap,
                "change_pct": change_pct,
            }
        except Exception as exc:
            logger.debug("Failed to parse ticker for %s: %s", name, exc)

    return {
        "tickers": tickers,
        "count": len(tickers),
        "timestamp": _now_et().isoformat(),
    }


@router.get("/kraken/ohlcv/{pair}")
def kraken_ohlcv(
    pair: str,
    interval: str = Query("5m", description="Candle interval: 1m, 5m, 15m, 30m, 1h, 4h, 1d"),
    period: str = Query("5d", description="Look-back period: 1d, 5d, 10d, 1mo, etc."),
):
    """Fetch historical OHLCV candles for a Kraken pair.

    Args:
        pair: Internal ticker (e.g. "KRAKEN:XBTUSD") or display name.
        interval: Candle interval.
        period: Look-back period.

    Returns OHLCV data in split-orientation (compatible with
    ``pd.DataFrame(**payload)`` on the client side).
    """
    internal = _resolve_internal_ticker(pair)
    if internal is None:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown Kraken pair: '{pair}'. Use /kraken/pairs for valid options.",
        )

    try:
        from lib.core.cache import get_data

        df = get_data(internal, interval=interval, period=period)
    except Exception as exc:
        logger.error("Kraken OHLCV error for %s: %s", pair, exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    if df.empty:
        return {
            "pair": internal,
            "interval": interval,
            "period": period,
            "bars": 0,
            "data": None,
            "timestamp": _now_et().isoformat(),
        }

    # Convert to split orientation for compact transport
    idx = df.index
    payload = {
        "pair": internal,
        "interval": interval,
        "period": period,
        "bars": len(df),
        "data": {
            "index": [str(t) for t in idx],
            "columns": list(df.columns),
            "data": df.values.tolist(),
        },
        "first": str(idx[0]) if len(df) > 0 else None,
        "last": str(idx[-1]) if len(df) > 0 else None,
        "timestamp": _now_et().isoformat(),
    }

    return payload


# ---------------------------------------------------------------------------
# Dashboard HTML fragment
# ---------------------------------------------------------------------------


@router.get("/kraken/health/html", response_class=HTMLResponse)
def kraken_health_html():
    """Return a dashboard-ready HTML fragment for Kraken status.

    Shows connectivity, WebSocket feed status, and live price summaries
    for the tracked crypto pairs.
    """
    enabled = _is_enabled()

    if not enabled:
        return HTMLResponse(
            content="""
            <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:6px">
                <h3 class="text-sm font-semibold t-text-muted">🪙 CRYPTO (KRAKEN)</h3>
                <span class="text-[10px] t-text-faint">disabled</span>
            </div>
            <div class="text-xs t-text-faint">
                Set <span class="font-mono">ENABLE_KRAKEN_CRYPTO=1</span> to enable.
            </div>
            """
        )

    # Check provider health
    provider = _get_provider()
    feed = _get_feed()
    pairs = _get_pairs()

    # API status
    api_ok = False
    api_error = None
    if provider is not None and provider.is_available:
        try:
            provider.get_server_time()
            api_ok = True
        except Exception as exc:
            api_error = str(exc)[:60]

    api_dot = '<span style="color:#22c55e">●</span>' if api_ok else '<span style="color:#ef4444">●</span>'
    api_text = "REST API connected" if api_ok else f"REST API error: {api_error or 'unavailable'}"

    # WebSocket feed status
    ws_running = feed is not None and feed.is_running
    ws_connected = feed is not None and feed.is_connected
    if ws_connected and feed is not None:
        ws_dot = '<span style="color:#22c55e">●</span>'
        ws_text = f"WS live — {feed.bar_count} bars, {feed.trade_count} trades"
    elif ws_running and feed is not None:
        ws_dot = '<span style="color:#eab308">●</span>'
        ws_text = "WS connecting..."
    else:
        ws_dot = '<span style="color:#52525b">●</span>'
        ws_text = "WS feed not running"

    # Auth status
    auth_badge = ""
    if provider is not None and provider.has_auth:
        auth_badge = (
            '<span style="font-size:9px;color:#22c55e;background:rgba(20,83,45,0.3);'
            "border:1px solid rgba(22,163,74,0.4);border-radius:3px;padding:0 4px;"
            'margin-left:4px">auth ✓</span>'
        )

    # Live prices from the feed (if available)
    prices_html = ""
    if feed is not None and feed.is_connected:
        bars = feed.latest_bars
        if bars:
            price_rows = []
            for name, pinfo in pairs.items():
                internal = pinfo["internal_ticker"]
                bar = bars.get(internal)
                if bar is None:
                    continue
                close = bar.get("close", 0)
                if close == 0:
                    continue
                base = pinfo.get("base", name[:3])
                # Format price with appropriate precision
                if close >= 1000:
                    price_str = f"${close:,.2f}"
                elif close >= 1:
                    price_str = f"${close:,.4f}"
                else:
                    price_str = f"${close:,.6f}"

                price_rows.append(
                    f'<div style="display:flex;justify-content:space-between;padding:1px 0">'
                    f'<span style="font-size:10px;color:#d4d4d8">{base}</span>'
                    f'<span style="font-size:10px;font-family:monospace;color:#4ade80">{price_str}</span>'
                    f"</div>"
                )

            if price_rows:
                prices_html = (
                    '<div style="margin-top:6px;padding-top:4px;'
                    'border-top:1px solid var(--border-subtle)">' + "".join(price_rows[:6]) + "</div>"
                )

    return HTMLResponse(
        content=f"""
        <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:6px">
            <h3 class="text-sm font-semibold t-text-muted">🪙 CRYPTO (KRAKEN)</h3>
            <span class="text-[10px] t-text-faint">{len(pairs)} pairs</span>
        </div>
        <div class="text-xs" style="display:flex;flex-direction:column;gap:3px">
            <div style="display:flex;align-items:center;gap:6px">
                {api_dot}
                <span class="t-text-secondary">{api_text}</span>
                {auth_badge}
            </div>
            <div style="display:flex;align-items:center;gap:6px">
                {ws_dot}
                <span class="t-text-secondary">{ws_text}</span>
            </div>
        </div>
        {prices_html}
        """
    )


# ---------------------------------------------------------------------------
# Candlestick chart HTML
# ---------------------------------------------------------------------------


def _render_candle_svg(bars: list[dict], width: int = 320, height: int = 140) -> str:
    """Render a minimal inline SVG candlestick chart from OHLCV bar dicts.

    Each bar must have keys: open, high, low, close, volume (floats).
    Returns an SVG string (no <svg> wrapper — caller must provide dimensions).
    """
    if not bars:
        return f'<text x="{width // 2}" y="{height // 2}" text-anchor="middle" fill="#52525b" font-size="10">No data</text>'

    # Keep last N candles that fit nicely
    max_candles = min(len(bars), max(20, width // 12))
    bars = bars[-max_candles:]
    n = len(bars)

    highs = [b.get("high", b.get("close", 0)) for b in bars]
    lows = [b.get("low", b.get("close", 0)) for b in bars]
    price_max = max(highs) if highs else 1.0
    price_min = min(lows) if lows else 0.0
    price_range = price_max - price_min or price_max * 0.01 or 1.0

    vol_max = max((b.get("volume", 0) for b in bars), default=1.0) or 1.0

    # Layout
    pad_l, pad_r, pad_t, pad_b = 4, 4, 6, 20  # leave room for volume bars at bottom
    chart_h = height - pad_t - pad_b
    vol_h = int(chart_h * 0.18)
    candle_h = chart_h - vol_h - 2
    chart_w = width - pad_l - pad_r
    candle_w = max(1, int(chart_w / n))
    body_w = max(1, candle_w - 2)

    def py(price: float) -> float:
        """Price → SVG y-coordinate within candle area."""
        frac = (price_max - price) / price_range
        return round(pad_t + frac * candle_h, 2)

    def vy(vol: float) -> float:
        """Volume → SVG y-coordinate for volume bar."""
        frac = vol / vol_max
        return round(pad_t + candle_h + 2 + vol_h * (1 - frac), 2)

    elements: list[str] = []

    # Horizontal price guide lines (3 levels)
    for i in range(3):
        p = price_min + price_range * i / 2
        y = py(p)
        # Format price label
        if price_max >= 1000:
            lbl = f"${p:,.0f}"
        elif price_max >= 1:
            lbl = f"${p:,.2f}"
        else:
            lbl = f"${p:.4f}"
        elements.append(
            f'<line x1="{pad_l}" y1="{y}" x2="{width - pad_r}" y2="{y}" '
            f'stroke="#27272a" stroke-width="0.5"/>'
            f'<text x="{width - pad_r - 1}" y="{y - 1}" '
            f'fill="#52525b" font-size="7" text-anchor="end">{lbl}</text>'
        )

    # Candles
    for i, bar in enumerate(bars):
        o = float(bar.get("open", 0))
        h = float(bar.get("high", o))
        lo = float(bar.get("low", o))
        c = float(bar.get("close", o))
        vol = float(bar.get("volume", 0))

        is_bull = c >= o
        color = "#22c55e" if is_bull else "#ef4444"
        x_center = pad_l + i * candle_w + candle_w // 2

        # Wick
        wick_top = py(h)
        wick_bot = py(lo)
        elements.append(
            f'<line x1="{x_center}" y1="{wick_top}" x2="{x_center}" y2="{wick_bot}" '
            f'stroke="{color}" stroke-width="0.8" opacity="0.7"/>'
        )

        # Body
        body_top = py(max(o, c))
        body_bot = py(min(o, c))
        body_height = max(1, body_bot - body_top)
        x_body = pad_l + i * candle_w + (candle_w - body_w) // 2
        elements.append(
            f'<rect x="{x_body}" y="{body_top}" width="{body_w}" height="{body_height}" fill="{color}" opacity="0.85"/>'
        )

        # Volume bar
        vol_y = vy(vol)
        vol_bar_bot = pad_t + candle_h + 2 + vol_h
        vol_bar_h = max(1, vol_bar_bot - vol_y)
        elements.append(
            f'<rect x="{x_body}" y="{vol_y}" width="{body_w}" height="{vol_bar_h}" fill="{color}" opacity="0.3"/>'
        )

    return "\n".join(elements)


def _fetch_ohlcv_bars(internal_ticker: str, interval: str = "15m", period: str = "2d") -> list[dict]:
    """Fetch OHLCV bars for an internal Kraken ticker, returning a list of dicts."""
    try:
        from lib.core.cache import get_data

        df = get_data(internal_ticker, interval=interval, period=period)
        if df is None or df.empty:
            return []

        records = []
        for ts, row in df.iterrows():
            records.append(
                {
                    "ts": str(ts),
                    "open": float(row.get("Open", row.get("open", 0))),  # type: ignore[arg-type]
                    "high": float(row.get("High", row.get("high", 0))),  # type: ignore[arg-type]
                    "low": float(row.get("Low", row.get("low", 0))),  # type: ignore[arg-type]
                    "close": float(row.get("Close", row.get("close", 0))),  # type: ignore[arg-type]
                    "volume": float(row.get("Volume", row.get("volume", 0))),  # type: ignore[arg-type]
                }
            )
        return records
    except Exception as exc:
        logger.debug("Failed to fetch OHLCV bars for %s: %s", internal_ticker, exc)
        return []


@router.get("/kraken/chart/html", response_class=HTMLResponse)
def kraken_chart_html(
    pair: str = Query(default="Bitcoin", description="Pair display name or internal ticker"),
    interval: str = Query(default="15m", description="Candle interval: 1m, 5m, 15m, 1h, 4h"),
    period: str = Query(default="2d", description="Look-back period: 1d, 2d, 5d, 10d"),
):
    """Return a live candlestick chart HTML fragment for a Kraken pair.

    Renders a self-contained SVG candlestick chart with volume bars,
    current price, 24h change, and interval/period controls.
    Designed for HTMX swap into ``#kraken-chart-container``.
    """
    if not _is_enabled():
        return HTMLResponse(
            content='<div style="color:#52525b;font-size:10px;text-align:center;padding:8px">Kraken disabled</div>'
        )

    internal = _resolve_internal_ticker(pair)
    if internal is None:
        # Default to Bitcoin
        internal = "KRAKEN:XBTUSD"
        pair = "Bitcoin"

    pairs = _get_pairs()
    # Find display name for internal ticker
    display_name = pair
    base = "?"
    for name, info in pairs.items():
        if info["internal_ticker"] == internal:
            display_name = name
            base = info.get("base", name[:3])
            break

    bars = _fetch_ohlcv_bars(internal, interval=interval, period=period)

    # Current price and change
    last_price = 0.0
    change_pct = 0.0
    change_color = "#a1a1aa"
    if bars:
        last_price = bars[-1]["close"]
        first_price = bars[0]["open"] if bars[0]["open"] > 0 else bars[0]["close"]
        if first_price > 0:
            change_pct = (last_price - first_price) / first_price * 100
        change_color = "#22c55e" if change_pct >= 0 else "#ef4444"

    # Format last price
    if last_price >= 1000:
        price_str = f"${last_price:,.2f}"
    elif last_price >= 1:
        price_str = f"${last_price:,.4f}"
    elif last_price > 0:
        price_str = f"${last_price:,.6f}"
    else:
        price_str = "—"

    change_str = f"{'+' if change_pct >= 0 else ''}{change_pct:.2f}%"

    # SVG chart
    svg_content = _render_candle_svg(bars, width=300, height=130)

    # Pair selector
    pair_opts = ""
    for name in pairs:
        sel = "selected" if name == display_name else ""
        pair_opts += f'<option value="{name}" {sel}>{pairs[name].get("base", name[:3])}/USD</option>'

    # Interval selector
    intervals = [("1m", "1m"), ("5m", "5m"), ("15m", "15m"), ("30m", "30m"), ("1h", "1h"), ("4h", "4h")]
    interval_opts = "".join(
        f'<option value="{v}" {"selected" if v == interval else ""}>{label}</option>' for v, label in intervals
    )

    # Period selector
    periods = [("1d", "1D"), ("2d", "2D"), ("5d", "5D"), ("10d", "10D")]
    period_opts = "".join(
        f'<option value="{v}" {"selected" if v == period else ""}>{label}</option>' for v, label in periods
    )

    now_str = datetime.now(tz=_EST).strftime("%H:%M ET")
    bar_count = len(bars)

    return HTMLResponse(
        content=f"""
<div id="kraken-chart-inner">
    <!-- Header -->
    <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:4px">
        <div>
            <span style="font-size:12px;font-weight:700;color:var(--text-primary)">{base}/USD</span>
            <span style="font-size:13px;font-family:monospace;font-weight:700;
                         color:{change_color};margin-left:6px">{price_str}</span>
            <span style="font-size:9px;color:{change_color};margin-left:3px">{change_str}</span>
        </div>
        <span style="font-size:8px;color:var(--text-faint)">{bar_count} bars · {now_str}</span>
    </div>

    <!-- Controls -->
    <div style="display:flex;gap:4px;margin-bottom:5px;flex-wrap:wrap">
        <select style="font-size:9px;background:var(--bg-input,#27272a);color:var(--text-secondary);
                       border:1px solid var(--border-panel,#3f3f46);border-radius:3px;padding:1px 3px"
                hx-get="/kraken/chart/html"
                hx-trigger="change"
                hx-target="#kraken-chart-inner"
                hx-swap="outerHTML"
                hx-include="[name='interval'],[name='period']"
                name="pair">
            {pair_opts}
        </select>
        <select name="interval"
                style="font-size:9px;background:var(--bg-input,#27272a);color:var(--text-secondary);
                       border:1px solid var(--border-panel,#3f3f46);border-radius:3px;padding:1px 3px"
                hx-get="/kraken/chart/html"
                hx-trigger="change"
                hx-target="#kraken-chart-inner"
                hx-swap="outerHTML"
                hx-include="[name='pair'],[name='period']">
            {interval_opts}
        </select>
        <select name="period"
                style="font-size:9px;background:var(--bg-input,#27272a);color:var(--text-secondary);
                       border:1px solid var(--border-panel,#3f3f46);border-radius:3px;padding:1px 3px"
                hx-get="/kraken/chart/html"
                hx-trigger="change"
                hx-target="#kraken-chart-inner"
                hx-swap="outerHTML"
                hx-include="[name='pair'],[name='interval']">
            {period_opts}
        </select>
    </div>

    <!-- Candlestick SVG -->
    <div style="background:var(--bg-panel-inner,rgba(39,39,42,0.4));border-radius:4px;
                padding:3px;border:1px solid var(--border-subtle,#27272a)">
        {
            '<svg width="300" height="130" style="display:block;width:100%;height:auto">' + svg_content + "</svg>"
            if bars
            else '<div style="height:130px;display:flex;align-items:center;justify-content:center;'
            'color:#52525b;font-size:10px">No chart data available</div>'
        }
    </div>
</div>"""
    )


# ---------------------------------------------------------------------------
# Kraken private account panel HTML
# ---------------------------------------------------------------------------


@router.get("/kraken/account/html", response_class=HTMLResponse)
def kraken_account_html():
    """Return a dashboard HTML fragment for Kraken private account data.

    Shows account balances, open orders, and recent trade history.
    Requires KRAKEN_API_KEY + KRAKEN_API_SECRET to be set.
    If not authenticated, renders a setup prompt.
    """
    if not _is_enabled():
        return HTMLResponse(
            content='<div style="color:#52525b;font-size:10px;text-align:center;padding:8px">Kraken disabled</div>'
        )

    provider = _get_provider()
    now_str = datetime.now(tz=_EST).strftime("%I:%M %p ET")

    if provider is None or not provider.has_auth:
        return HTMLResponse(
            content=f"""
<div style="padding:6px">
    <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:6px">
        <span style="font-size:11px;font-weight:600;color:var(--text-muted)">🔐 Kraken Account</span>
        <span style="font-size:8px;color:var(--text-faint)">{now_str}</span>
    </div>
    <div style="background:rgba(251,191,36,0.08);border:1px solid rgba(251,191,36,0.3);
                border-radius:5px;padding:8px;font-size:10px;color:#fbbf24">
        Private API not configured.<br>
        <span style="font-size:9px;color:var(--text-faint);display:block;margin-top:3px">
            Set <code>KRAKEN_API_KEY</code> and <code>KRAKEN_API_SECRET</code>
            environment variables to enable position tracking, balance display,
            and order history.
        </span>
    </div>
</div>"""
        )

    # Fetch account data
    balance_html = ""
    orders_html = ""
    history_html = ""
    errors: list[str] = []

    # ── Balances ─────────────────────────────────────────────────────────────
    try:
        raw_balance = provider.get_balance()
        # Filter to non-zero balances
        nonzero = {k: v for k, v in raw_balance.items() if float(v) > 0.0001}
        if nonzero:
            rows = ""
            for asset, amount in sorted(nonzero.items()):
                amt = float(amount)
                # Map Kraken asset codes to friendly names
                friendly = {
                    "ZUSD": "USD",
                    "XXBT": "BTC",
                    "XETH": "ETH",
                    "SOL": "SOL",
                    "LINK": "LINK",
                    "AVAX": "AVAX",
                    "DOT": "DOT",
                    "ADA": "ADA",
                    "POL": "POL",
                    "XXRP": "XRP",
                }.get(asset, asset)
                if amt >= 1000:
                    amt_str = f"{amt:,.2f}"
                elif amt >= 1:
                    amt_str = f"{amt:,.4f}"
                else:
                    amt_str = f"{amt:.6f}"
                rows += (
                    f"<tr>"
                    f'<td style="padding:2px 4px;font-size:10px;color:var(--text-secondary)">{friendly}</td>'
                    f'<td style="padding:2px 4px;font-size:10px;font-family:monospace;text-align:right;'
                    f'color:var(--text-primary)">{amt_str}</td>'
                    f"</tr>"
                )
            balance_html = f"""
<div style="margin-bottom:8px">
    <div style="font-size:8px;color:var(--text-faint);margin-bottom:3px;text-transform:uppercase;
                letter-spacing:0.05em">Balances</div>
    <table style="width:100%;border-collapse:collapse">
        <thead>
            <tr style="border-bottom:1px solid var(--border-subtle,#27272a)">
                <th style="padding:2px 4px;font-size:8px;color:var(--text-faint);text-align:left;font-weight:600">Asset</th>
                <th style="padding:2px 4px;font-size:8px;color:var(--text-faint);text-align:right;font-weight:600">Amount</th>
            </tr>
        </thead>
        <tbody>{rows}</tbody>
    </table>
</div>"""
        else:
            balance_html = (
                '<div style="font-size:9px;color:var(--text-faint);margin-bottom:8px">No non-zero balances found</div>'
            )
    except Exception as exc:
        errors.append(f"Balance: {str(exc)[:60]}")

    # ── Open orders ───────────────────────────────────────────────────────────
    try:
        raw_orders = provider.get_open_orders()
        open_orders = raw_orders.get("open", {})
        if open_orders:
            rows = ""
            for _txid, order in list(open_orders.items())[:8]:  # cap at 8
                descr = order.get("descr", {})
                side = descr.get("type", "?").upper()
                pair = descr.get("pair", "?")
                price = descr.get("price", "?")
                vol = float(order.get("vol", 0))
                vol_exec = float(order.get("vol_exec", 0))
                side_color = "#22c55e" if side == "BUY" else "#ef4444"
                rows += (
                    f'<tr style="border-bottom:1px solid var(--border-subtle,#27272a)">'
                    f'<td style="padding:2px 4px;font-size:9px;color:{side_color};font-weight:600">{side}</td>'
                    f'<td style="padding:2px 4px;font-size:9px;color:var(--text-secondary)">{pair}</td>'
                    f'<td style="padding:2px 4px;font-size:9px;font-family:monospace;color:var(--text-primary)">@{price}</td>'
                    f'<td style="padding:2px 4px;font-size:9px;font-family:monospace;color:var(--text-muted)">'
                    f"{vol_exec:.2f}/{vol:.2f}</td>"
                    f"</tr>"
                )
            orders_html = f"""
<div style="margin-bottom:8px">
    <div style="font-size:8px;color:var(--text-faint);margin-bottom:3px;text-transform:uppercase;
                letter-spacing:0.05em">Open Orders ({len(open_orders)})</div>
    <div style="overflow-x:auto">
        <table style="width:100%;border-collapse:collapse">
            <thead>
                <tr style="border-bottom:1px solid var(--border-subtle,#27272a)">
                    <th style="padding:2px 4px;font-size:8px;color:var(--text-faint);text-align:left;font-weight:600">Side</th>
                    <th style="padding:2px 4px;font-size:8px;color:var(--text-faint);text-align:left;font-weight:600">Pair</th>
                    <th style="padding:2px 4px;font-size:8px;color:var(--text-faint);text-align:left;font-weight:600">Price</th>
                    <th style="padding:2px 4px;font-size:8px;color:var(--text-faint);text-align:left;font-weight:600">Filled</th>
                </tr>
            </thead>
            <tbody>{rows}</tbody>
        </table>
    </div>
</div>"""
        else:
            orders_html = '<div style="font-size:9px;color:var(--text-faint);margin-bottom:8px">No open orders</div>'
    except Exception as exc:
        errors.append(f"Orders: {str(exc)[:60]}")

    # ── Trade history (last 10) ────────────────────────────────────────────
    try:
        raw_hist = provider.get_trade_history()
        trades = raw_hist.get("trades", {})
        if trades:
            # Sort by time descending
            sorted_trades = sorted(trades.items(), key=lambda kv: float(kv[1].get("time", 0)), reverse=True)[:8]
            rows = ""
            for _txid, trade in sorted_trades:
                pair = trade.get("pair", "?")
                side = trade.get("type", "?").upper()
                price = float(trade.get("price", 0))
                vol = float(trade.get("vol", 0))
                cost = float(trade.get("cost", 0))
                fee = float(trade.get("fee", 0))
                cost - fee if side == "BUY" else -(cost + fee)
                t = float(trade.get("time", 0))
                try:
                    ts_str = datetime.fromtimestamp(t, tz=UTC).strftime("%m/%d %H:%M")
                except Exception:
                    ts_str = "—"
                side_color = "#22c55e" if side == "BUY" else "#ef4444"
                price_str = f"${price:,.2f}" if price >= 100 else f"${price:,.4f}"
                rows += (
                    f'<tr style="border-bottom:1px solid var(--border-subtle,#27272a)">'
                    f'<td style="padding:2px 4px;font-size:8px;color:var(--text-faint);white-space:nowrap">{ts_str}</td>'
                    f'<td style="padding:2px 4px;font-size:9px;color:{side_color};font-weight:600">{side}</td>'
                    f'<td style="padding:2px 4px;font-size:9px;color:var(--text-secondary)">{pair}</td>'
                    f'<td style="padding:2px 4px;font-size:9px;font-family:monospace;color:var(--text-primary)">{price_str}</td>'
                    f'<td style="padding:2px 4px;font-size:9px;font-family:monospace;color:var(--text-muted)">{vol:.4f}</td>'
                    f"</tr>"
                )
            history_html = f"""
<div>
    <div style="font-size:8px;color:var(--text-faint);margin-bottom:3px;text-transform:uppercase;
                letter-spacing:0.05em">Recent Trades</div>
    <div style="overflow-x:auto;max-height:160px;overflow-y:auto">
        <table style="width:100%;border-collapse:collapse">
            <thead>
                <tr style="border-bottom:1px solid var(--border-subtle,#27272a)">
                    <th style="padding:2px 4px;font-size:8px;color:var(--text-faint);text-align:left;font-weight:600">Time</th>
                    <th style="padding:2px 4px;font-size:8px;color:var(--text-faint);text-align:left;font-weight:600">Side</th>
                    <th style="padding:2px 4px;font-size:8px;color:var(--text-faint);text-align:left;font-weight:600">Pair</th>
                    <th style="padding:2px 4px;font-size:8px;color:var(--text-faint);text-align:left;font-weight:600">Price</th>
                    <th style="padding:2px 4px;font-size:8px;color:var(--text-faint);text-align:left;font-weight:600">Vol</th>
                </tr>
            </thead>
            <tbody>{rows}</tbody>
        </table>
    </div>
</div>"""
        else:
            history_html = '<div style="font-size:9px;color:var(--text-faint)">No recent trades found</div>'
    except Exception as exc:
        errors.append(f"History: {str(exc)[:60]}")

    error_html = ""
    if errors:
        error_html = '<div style="font-size:9px;color:#ef4444;margin-bottom:4px">' + "<br>".join(errors) + "</div>"

    return HTMLResponse(
        content=f"""
<div style="padding:2px">
    <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:8px">
        <span style="font-size:11px;font-weight:600;color:var(--text-muted)">🔐 Kraken Account</span>
        <span style="font-size:8px;color:var(--text-faint)">{now_str}</span>
    </div>
    {error_html}
    {balance_html}
    {orders_html}
    {history_html}
</div>"""
    )


# ---------------------------------------------------------------------------
# Crypto/Futures correlation panel HTML
# ---------------------------------------------------------------------------


def _pearson_corr(xs: list[float], ys: list[float]) -> float:
    """Compute Pearson correlation coefficient for two equal-length lists."""
    n = len(xs)
    if n < 3:
        return float("nan")
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=False))
    denom_x = math.sqrt(sum((x - mx) ** 2 for x in xs))
    denom_y = math.sqrt(sum((y - my) ** 2 for y in ys))
    if denom_x == 0 or denom_y == 0:
        return float("nan")
    return num / (denom_x * denom_y)


def _returns(prices: list[float]) -> list[float]:
    """Compute log returns from a price series."""
    import math as _m

    result = []
    for i in range(1, len(prices)):
        p0, p1 = prices[i - 1], prices[i]
        if p0 > 0 and p1 > 0:
            result.append(_m.log(p1 / p0))
        else:
            result.append(0.0)
    return result


def _render_corr_bar(corr: float, label: str) -> str:
    """Render a single correlation bar row."""
    if math.isnan(corr):
        return (
            f'<div style="display:flex;align-items:center;gap:6px;margin-bottom:3px">'
            f'<span style="font-size:9px;color:var(--text-muted);width:55px;flex-shrink:0">{label}</span>'
            f'<span style="font-size:9px;color:var(--text-faint);font-family:monospace">n/a</span>'
            f"</div>"
        )

    # Color: green for positive, red for negative, muted near zero
    abs_c = abs(corr)
    if abs_c >= 0.7:
        color = "#22c55e" if corr > 0 else "#ef4444"
    elif abs_c >= 0.4:
        color = "#fbbf24"
    else:
        color = "#71717a"

    # Bar: center at 50%; positive bars go right, negative go left
    bar_pct = abs_c * 48  # max 48% of half-width
    if corr >= 0:
        bar_x: float = 50.0
        bar_w = bar_pct
    else:
        bar_x = 50.0 - bar_pct
        bar_w = bar_pct

    return (
        f'<div style="display:flex;align-items:center;gap:6px;margin-bottom:3px">'
        f'<span style="font-size:9px;color:var(--text-muted);width:55px;flex-shrink:0;white-space:nowrap">{label}</span>'
        f'<div style="flex:1;height:8px;background:var(--bg-bar,#3f3f46);border-radius:2px;position:relative">'
        # center line
        f'<div style="position:absolute;left:50%;top:0;width:1px;height:100%;background:var(--border-subtle,#27272a)"></div>'
        # correlation bar
        f'<div style="position:absolute;left:{bar_x:.1f}%;top:1px;width:{bar_w:.1f}%;height:6px;'
        f'background:{color};border-radius:2px;opacity:0.85"></div>'
        f"</div>"
        f'<span style="font-size:9px;font-family:monospace;color:{color};width:38px;text-align:right;flex-shrink:0">'
        f"{corr:+.2f}</span>"
        f"</div>"
    )


@router.get("/kraken/correlation/html", response_class=HTMLResponse)
def kraken_correlation_html(
    period: str = Query(default="10d", description="Look-back period: 5d, 10d, 30d"),
    interval: str = Query(default="1h", description="Bar interval: 15m, 1h, 4h"),
):
    """Return a crypto/futures correlation panel as an HTML fragment.

    Computes rolling Pearson correlation (log returns) between BTC, ETH
    and the tracked futures instruments (MES, MGC, MNQ) over the chosen
    look-back period.  All price series are aligned on common timestamps
    before correlation is computed.

    Designed for HTMX swap into ``#correlation-container``.
    """
    now_str = datetime.now(tz=_EST).strftime("%I:%M %p ET")

    # ── Futures tickers we correlate against ───────────────────────────────
    futures_pairs = [
        ("MES", "/MES"),  # Micro E-mini S&P 500
        ("MGC", "/MGC"),  # Micro Gold
        ("MNQ", "/MNQ"),  # Micro Nasdaq
        ("MCL", "/MCL"),  # Micro Crude Oil
    ]

    # ── Crypto anchors ─────────────────────────────────────────────────────
    crypto_anchors = [
        ("BTC", "KRAKEN:XBTUSD"),
        ("ETH", "KRAKEN:ETHUSD"),
        ("SOL", "KRAKEN:SOLUSD"),
    ]

    # Period + interval selectors
    period_opts = "".join(
        f'<option value="{v}" {"selected" if v == period else ""}>{label}</option>'
        for v, label in [("5d", "5D"), ("10d", "10D"), ("30d", "30D")]
    )
    interval_opts = "".join(
        f'<option value="{v}" {"selected" if v == interval else ""}>{label}</option>'
        for v, label in [("15m", "15m"), ("1h", "1h"), ("4h", "4h")]
    )

    # ── Fetch price series ─────────────────────────────────────────────────
    price_series: dict[str, dict[str, float]] = {}  # label → {ts_str → close}

    def _fetch_series(ticker: str, label: str) -> None:
        try:
            from lib.core.cache import get_data

            df = get_data(ticker, interval=interval, period=period)
            if df is None or df.empty:
                return
            close_col = "Close" if "Close" in df.columns else "close"
            for ts, row in df.iterrows():
                ts_key = str(ts)[:16]  # round to minute
                price_series.setdefault(label, {})[ts_key] = float(row[close_col])
        except Exception as exc:
            logger.debug("Correlation fetch failed for %s: %s", ticker, exc)

    # Fetch crypto anchors
    for label, ticker in crypto_anchors:
        if _is_enabled():
            _fetch_series(ticker, label)

    # Fetch futures
    for label, ticker in futures_pairs:
        _fetch_series(ticker, label)

    # ── Compute correlations: each crypto vs each futures ──────────────────
    corr_results: dict[str, dict[str, float]] = {}  # crypto_label → {futures_label → corr}

    for c_label, _ in crypto_anchors:
        c_series = price_series.get(c_label, {})
        if not c_series:
            continue
        corr_results[c_label] = {}
        for f_label, _ in futures_pairs:
            f_series = price_series.get(f_label, {})
            if not f_series:
                corr_results[c_label][f_label] = float("nan")
                continue
            # Align on common timestamps
            common = sorted(set(c_series.keys()) & set(f_series.keys()))
            if len(common) < 5:
                corr_results[c_label][f_label] = float("nan")
                continue
            c_prices = [c_series[ts] for ts in common]
            f_prices = [f_series[ts] for ts in common]
            c_ret = _returns(c_prices)
            f_ret = _returns(f_prices)
            if len(c_ret) < 3:
                corr_results[c_label][f_label] = float("nan")
                continue
            corr_results[c_label][f_label] = round(_pearson_corr(c_ret, f_ret), 3)

    # ── Also compute crypto-to-crypto correlations ─────────────────────────
    crypto_cross: dict[str, float] = {}  # "BTC/ETH" etc.
    c_labels = [c for c, _ in crypto_anchors if c in price_series]
    for i in range(len(c_labels)):
        for j in range(i + 1, len(c_labels)):
            la, lb = c_labels[i], c_labels[j]
            s_a = price_series.get(la, {})
            s_b = price_series.get(lb, {})
            common = sorted(set(s_a.keys()) & set(s_b.keys()))
            if len(common) < 5:
                crypto_cross[f"{la}/{lb}"] = float("nan")
                continue
            r_a = _returns([s_a[ts] for ts in common])
            r_b = _returns([s_b[ts] for ts in common])
            if len(r_a) < 3:
                crypto_cross[f"{la}/{lb}"] = float("nan")
                continue
            crypto_cross[f"{la}/{lb}"] = round(_pearson_corr(r_a, r_b), 3)

    # ── Render ─────────────────────────────────────────────────────────────
    if not corr_results and not crypto_cross:
        content = f"""
<div style="color:var(--text-faint);font-size:10px;text-align:center;padding:12px">
    Insufficient data for correlation analysis.<br>
    <span style="font-size:9px">Needs {period} of {interval} bars for both crypto and futures.</span>
</div>"""
    else:
        sections = []

        # Per-crypto section
        for c_label, _ in crypto_anchors:
            if c_label not in corr_results:
                continue
            bars_html = ""
            for f_label, _ in futures_pairs:
                corr = corr_results[c_label].get(f_label, float("nan"))
                bars_html += _render_corr_bar(corr, f_label)

            sections.append(f"""
<div style="margin-bottom:8px">
    <div style="font-size:9px;font-weight:600;color:var(--text-secondary);
                margin-bottom:4px;border-bottom:1px solid var(--border-subtle,#27272a);
                padding-bottom:2px">{c_label} vs Futures</div>
    {bars_html}
</div>""")

        # Crypto cross-correlations
        if crypto_cross:
            cross_bars = ""
            for label, corr in crypto_cross.items():
                cross_bars += _render_corr_bar(corr, label)
            sections.append(f"""
<div style="margin-bottom:4px">
    <div style="font-size:9px;font-weight:600;color:var(--text-secondary);
                margin-bottom:4px;border-bottom:1px solid var(--border-subtle,#27272a);
                padding-bottom:2px">Crypto Cross</div>
    {cross_bars}
</div>""")

        content = "".join(sections)

    return HTMLResponse(
        content=f"""
<div id="correlation-inner">
    <!-- Header + controls -->
    <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:6px">
        <span style="font-size:8px;color:var(--text-faint)">{period} · {interval} · {now_str}</span>
        <div style="display:flex;gap:3px">
            <select style="font-size:9px;background:var(--bg-input,#27272a);color:var(--text-secondary);
                           border:1px solid var(--border-panel,#3f3f46);border-radius:3px;padding:1px 3px"
                    hx-get="/kraken/correlation/html"
                    hx-trigger="change"
                    hx-target="#correlation-inner"
                    hx-swap="outerHTML"
                    hx-include="[name='interval']"
                    name="period">
                {period_opts}
            </select>
            <select name="interval"
                    style="font-size:9px;background:var(--bg-input,#27272a);color:var(--text-secondary);
                           border:1px solid var(--border-panel,#3f3f46);border-radius:3px;padding:1px 3px"
                    hx-get="/kraken/correlation/html"
                    hx-trigger="change"
                    hx-target="#correlation-inner"
                    hx-swap="outerHTML"
                    hx-include="[name='period']">
                {interval_opts}
            </select>
        </div>
    </div>

    <!-- Legend -->
    <div style="display:flex;gap:8px;margin-bottom:5px;font-size:8px;color:var(--text-faint)">
        <span style="color:#22c55e">■ Strong +</span>
        <span style="color:#fbbf24">■ Moderate</span>
        <span style="color:#ef4444">■ Strong −</span>
        <span style="color:#71717a">■ Weak</span>
    </div>

    {content}

    <div style="font-size:8px;color:var(--text-faint);margin-top:4px;
                border-top:1px solid var(--border-subtle,#27272a);padding-top:3px">
        Pearson r (log returns) · |r|≥0.7 strong · |r|≥0.4 moderate
    </div>
</div>"""
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_internal_ticker(pair: str) -> str | None:
    """Resolve a user-provided pair string to an internal KRAKEN:* ticker.

    Accepts:
      - Internal ticker: "KRAKEN:XBTUSD"
      - Display name: "Bitcoin", "BTC/USD"
      - REST pair: "XXBTZUSD"
      - WS pair: "XBT/USD"
      - Shorthand: "XBTUSD", "btc", "eth"
    """
    pair_stripped = pair.strip()

    # Already an internal ticker
    if pair_stripped.startswith("KRAKEN:"):
        try:
            from lib.integrations.kraken_client import ALL_KRAKEN_TICKERS

            if pair_stripped in ALL_KRAKEN_TICKERS:
                return pair_stripped
        except ImportError:
            pass
        return None

    pairs = _get_pairs()

    # Try exact match on display name (case-insensitive)
    pair_lower = pair_stripped.lower()
    for name, info in pairs.items():
        if name.lower() == pair_lower:
            return info["internal_ticker"]

    # Try match on REST pair
    try:
        from lib.integrations.kraken_client import REST_TO_INTERNAL

        if pair_stripped in REST_TO_INTERNAL:
            return REST_TO_INTERNAL[pair_stripped]
    except ImportError:
        pass

    # Try match on WS pair (e.g. "XBT/USD")
    try:
        from lib.integrations.kraken_client import WS_PAIR_TO_NAME

        if pair_stripped in WS_PAIR_TO_NAME:
            ws_name = WS_PAIR_TO_NAME[pair_stripped]
            return pairs[ws_name]["internal_ticker"]
    except ImportError:
        pass

    # Try shorthand: "btc" → "Bitcoin", "eth" → "Ethereum", etc.
    _shorthand: dict[str, str] = {
        "btc": "Bitcoin",
        "bitcoin": "Bitcoin",
        "xbt": "Bitcoin",
        "eth": "Ethereum",
        "ethereum": "Ethereum",
        "sol": "Solana",
        "solana": "Solana",
        "link": "Chainlink",
        "chainlink": "Chainlink",
        "avax": "Avalanche",
        "avalanche": "Avalanche",
        "dot": "Polkadot",
        "polkadot": "Polkadot",
        "ada": "Cardano",
        "cardano": "Cardano",
        "pol": "Polygon",
        "polygon": "Polygon",
        "xrp": "XRP",
    }
    resolved_name = _shorthand.get(pair_lower)
    if resolved_name and resolved_name in pairs:
        return pairs[resolved_name]["internal_ticker"]

    # Try matching "XBTUSD" style (strip KRAKEN: prefix)
    for _pname, info in pairs.items():
        # Match against the suffix of internal_ticker
        suffix = info["internal_ticker"].replace("KRAKEN:", "")
        if pair_stripped.upper() == suffix:
            return info["internal_ticker"]

    return None
