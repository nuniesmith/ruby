"""
Pre-Trade Analysis API — Asset Selection & Opportunity Scoring
===============================================================
Provides endpoints for the pre-trade analysis workflow where users
review daily opportunities across crypto and futures, run analysis
pipelines, and select assets for simulation monitoring.

Endpoints:
    GET  /api/pretrade/assets         — List all assets with opportunity data
    POST /api/pretrade/analyze        — Run full analysis on selected assets
    GET  /api/pretrade/analysis/{symbol} — Get latest analysis for a symbol
    POST /api/pretrade/select         — Mark assets as selected for monitoring
    GET  /api/pretrade/selected       — Get currently selected assets
    GET  /api/pretrade/watchlist      — Live watchlist with prices + signals

Usage:
    from lib.services.data.api.pretrade import router
    app.include_router(router)
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

logger = logging.getLogger("api.pretrade")

router = APIRouter(prefix="/api/pretrade", tags=["Pre-Trade Analysis"])

# ---------------------------------------------------------------------------
# Focus futures symbols — the 9 micro contracts the system primarily trades
# ---------------------------------------------------------------------------
_FUTURES_FOCUS: list[dict[str, str]] = [
    {"name": "Gold", "symbol": "MGC=F", "exchange": "COMEX"},
    {"name": "Silver", "symbol": "SIL=F", "exchange": "COMEX"},
    {"name": "S&P 500 E-Mini", "symbol": "MES=F", "exchange": "CME"},
    {"name": "Nasdaq E-Mini", "symbol": "MNQ=F", "exchange": "CME"},
    {"name": "Russell 2000", "symbol": "M2K=F", "exchange": "CME"},
    {"name": "Dow Jones", "symbol": "MYM=F", "exchange": "CME"},
    {"name": "10Y T-Note", "symbol": "ZN=F", "exchange": "CBOT"},
    {"name": "30Y T-Bond", "symbol": "ZB=F", "exchange": "CBOT"},
    {"name": "Wheat", "symbol": "ZW=F", "exchange": "CBOT"},
]

# TTL constants
_ANALYSIS_TTL = 3600  # 1 hour
_SELECTED_TTL = 86400  # 24 hours

# Redis key prefixes
_ANALYSIS_KEY_PREFIX = "pretrade:analysis:"
_SELECTED_KEY = "pretrade:selected"


# ---------------------------------------------------------------------------
# Pydantic request / response models
# ---------------------------------------------------------------------------


class AnalyzeRequest(BaseModel):
    """Request body for POST /api/pretrade/analyze."""

    symbols: list[str] = Field(..., description="List of symbols to analyze")
    run_cnn: bool = Field(True, description="Run CNN prediction")
    run_ruby: bool = Field(True, description="Run Ruby signal engine")
    run_indicators: bool = Field(True, description="Compute technical indicators")
    run_news: bool = Field(True, description="Fetch news sentiment")


class SelectRequest(BaseModel):
    """Request body for POST /api/pretrade/select."""

    symbols: list[str] = Field(..., description="Symbols to select for monitoring")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_redis_from_cache():
    """Get the shared Redis client from the cache module, or None."""
    try:
        from lib.core.cache import REDIS_AVAILABLE, _r

        if REDIS_AVAILABLE and _r is not None:
            return _r
    except ImportError:
        pass
    return None


def _get_redis(request: Request):
    """Return the Redis client from app state or cache module."""
    try:
        r = request.app.state.redis
        if r is not None:
            return r
    except AttributeError:
        pass
    return _get_redis_from_cache()


def _safe_float(val: Any, default: float = 0.0) -> float:
    """Convert a value to float safely."""
    if val is None:
        return default
    try:
        f = float(val)
        if f != f:  # NaN check
            return default
        return f
    except (ValueError, TypeError):
        return default


def _get_kraken_specs() -> dict[str, dict[str, Any]]:
    """Load Kraken contract specs from models."""
    try:
        from lib.core.models import KRAKEN_CONTRACT_SPECS

        return dict(KRAKEN_CONTRACT_SPECS)
    except ImportError:
        return {}


def _read_kraken_price(redis_client: Any, ticker: str) -> float | None:
    """Read latest Kraken price from Redis."""
    if redis_client is None:
        return None
    try:
        raw = redis_client.get(f"kraken:live:{ticker}")
        if raw is None:
            return None
        data = json.loads(raw.decode() if isinstance(raw, bytes) else raw)
        if isinstance(data, dict):
            return _safe_float(data.get("price") or data.get("last") or data.get("c"))
        return _safe_float(data)
    except Exception:
        return None


def _read_futures_price(redis_client: Any, symbol: str) -> float | None:
    """Read latest futures price from Redis engine cache."""
    if redis_client is None:
        return None
    # Try various Redis key patterns used by the engine
    key_patterns = [
        f"kraken:live:{symbol}",
        f"futures:live:{symbol}",
        f"engine:price:{symbol}",
    ]
    for key in key_patterns:
        try:
            raw = redis_client.get(key)
            if raw is None:
                continue
            data = json.loads(raw.decode() if isinstance(raw, bytes) else raw)
            if isinstance(data, dict):
                price = data.get("price") or data.get("last") or data.get("close") or data.get("c")
                if price is not None:
                    return _safe_float(price)
            else:
                val = _safe_float(data)
                if val > 0:
                    return val
        except Exception:
            continue
    return None


def _read_analysis_cache(redis_client: Any, symbol: str) -> dict | None:
    """Read cached analysis from Redis."""
    if redis_client is None:
        return None
    try:
        raw = redis_client.get(f"{_ANALYSIS_KEY_PREFIX}{symbol}")
        if raw is None:
            return None
        return json.loads(raw.decode() if isinstance(raw, bytes) else raw)
    except Exception:
        return None


def _store_analysis_cache(redis_client: Any, symbol: str, data: dict) -> None:
    """Store analysis result in Redis with TTL."""
    if redis_client is None:
        return
    try:
        payload = json.dumps(data)
        redis_client.setex(f"{_ANALYSIS_KEY_PREFIX}{symbol}", _ANALYSIS_TTL, payload)
    except Exception as exc:
        logger.warning("Failed to cache analysis for %s: %s", symbol, exc)


def _compute_overall_score(
    cnn_confidence: float | None,
    indicator_alignment: float | None,
    news_sentiment: float | None,
) -> float:
    """Compute weighted overall score with proportional reweighting for missing components.

    Weights: CNN 0.4, indicators 0.3, news 0.3
    If a component is None, its weight is redistributed proportionally.
    Returns a score between 0 and 100.
    """
    components: list[tuple[float, float]] = []
    if cnn_confidence is not None:
        components.append((cnn_confidence, 0.4))
    if indicator_alignment is not None:
        components.append((indicator_alignment, 0.3))
    if news_sentiment is not None:
        components.append((news_sentiment, 0.3))

    if not components:
        return 50.0  # neutral default

    total_weight = sum(w for _, w in components)
    score = sum(v * (w / total_weight) for v, w in components)
    return max(0.0, min(100.0, score))


# ---------------------------------------------------------------------------
# GET /api/pretrade/assets — List all assets with opportunity data
# ---------------------------------------------------------------------------


@router.get("/assets")
async def list_assets(request: Request) -> dict:
    """List all available assets with opportunity data.

    Returns crypto (Kraken) and futures (focus 9) with latest prices,
    daily change, and analysis availability.
    """
    redis = _get_redis(request)
    kraken_specs = _get_kraken_specs()
    assets: list[dict[str, Any]] = []

    # --- Kraken crypto assets ---
    for name, spec in kraken_specs.items():
        ticker = str(spec.get("data_ticker", spec["ticker"]))
        price = _read_kraken_price(redis, ticker)
        cached = _read_analysis_cache(redis, ticker)

        assets.append(
            {
                "symbol": ticker,
                "name": name,
                "asset_class": "crypto",
                "exchange": spec.get("exchange", "kraken"),
                "latest_price": price,
                "daily_change_pct": None,
                "volume_24h": None,
                "signals_available": cached is not None,
                "last_analyzed": cached.get("analyzed_at") if cached else None,
                "margin": _safe_float(spec.get("margin", 0)),
            }
        )

    # --- Futures focus assets ---
    for item in _FUTURES_FOCUS:
        symbol = item["symbol"]
        price = _read_futures_price(redis, symbol)
        cached = _read_analysis_cache(redis, symbol)

        assets.append(
            {
                "symbol": symbol,
                "name": item["name"],
                "asset_class": "futures",
                "exchange": item["exchange"],
                "latest_price": price,
                "daily_change_pct": None,
                "volume_24h": None,
                "signals_available": cached is not None,
                "last_analyzed": cached.get("analyzed_at") if cached else None,
                "margin": None,
            }
        )

    # Try to enrich with daily change from stored bars
    for asset in assets:
        if redis is None:
            break
        try:
            raw = redis.get(f"pretrade:daily_change:{asset['symbol']}")
            if raw:
                change_data = json.loads(raw.decode() if isinstance(raw, bytes) else raw)
                asset["daily_change_pct"] = _safe_float(change_data.get("change_pct"))
                asset["volume_24h"] = _safe_float(change_data.get("volume_24h"))
        except Exception:
            pass

    return {
        "count": len(assets),
        "assets": assets,
        "timestamp": datetime.now(tz=UTC).isoformat(),
    }


# ---------------------------------------------------------------------------
# POST /api/pretrade/analyze — Run full analysis on selected assets
# ---------------------------------------------------------------------------


@router.post("/analyze")
async def analyze_assets(body: AnalyzeRequest, request: Request) -> dict:
    """Run analysis pipeline on selected assets.

    For each symbol, attempts to:
    1. Load recent bars via EngineDataClient
    2. Run technical indicators (if run_indicators=true)
    3. Get news sentiment (if run_news=true)
    4. Get CNN prediction (if run_cnn=true)

    Each step is wrapped in try/except for graceful degradation.
    Results are cached in Redis with 1-hour TTL.
    """
    redis = _get_redis(request)
    results: list[dict[str, Any]] = []

    for symbol in body.symbols:
        result: dict[str, Any] = {
            "symbol": symbol,
            "analyzed_at": datetime.now(tz=UTC).isoformat(),
            "bars_loaded": False,
            "indicators": None,
            "cnn_signal": None,
            "cnn_confidence": None,
            "ruby_signal": None,
            "news_sentiment": None,
            "news_headlines": None,
            "overall_score": None,
            "errors": [],
        }

        bars_df = None

        # Step 1: Load recent bars
        try:
            from lib.services.data.engine_data_client import get_client

            client = get_client()
            bars_df = client.get_bars(symbol, interval="5m", days_back=5)
            if bars_df is not None and not bars_df.empty:
                result["bars_loaded"] = True
                result["bar_count"] = len(bars_df)
                # Compute daily change from bars
                if len(bars_df) >= 2:
                    latest_close = _safe_float(bars_df["Close"].iloc[-1])
                    prev_close = _safe_float(bars_df["Close"].iloc[0])
                    if prev_close > 0:
                        change_pct = ((latest_close - prev_close) / prev_close) * 100
                        result["daily_change_pct"] = round(change_pct, 2)
                        # Cache the daily change for the assets endpoint
                        if redis:
                            try:
                                redis.setex(
                                    f"pretrade:daily_change:{symbol}",
                                    _ANALYSIS_TTL,
                                    json.dumps(
                                        {
                                            "change_pct": round(change_pct, 2),
                                            "volume_24h": _safe_float(bars_df["Volume"].sum())
                                            if "Volume" in bars_df.columns
                                            else None,
                                        }
                                    ),
                                )
                            except Exception:
                                pass
        except Exception as exc:
            result["errors"].append(f"bars: {exc}")
            logger.debug("Failed to load bars for %s: %s", symbol, exc)

        # Step 2: Run indicators
        indicator_alignment: float | None = None
        if body.run_indicators and bars_df is not None and not bars_df.empty:
            try:
                from lib.indicators.combined import compute_all_indicators

                indicators = compute_all_indicators(bars_df)
                if indicators and isinstance(indicators, dict):
                    result["indicators"] = {
                        k: round(v, 6) if isinstance(v, float) else v
                        for k, v in indicators.items()
                        if isinstance(v, (int, float, str, bool)) or v is None
                    }
                    # Derive indicator alignment score (0-100)
                    # Count bullish vs bearish signals from indicator values
                    bullish = 0
                    bearish = 0
                    total = 0
                    for k, v in indicators.items():
                        if isinstance(v, str):
                            low = v.lower()
                            if "bull" in low or "long" in low or "buy" in low:
                                bullish += 1
                                total += 1
                            elif "bear" in low or "short" in low or "sell" in low:
                                bearish += 1
                                total += 1
                        elif isinstance(v, (int, float)):
                            if "rsi" in k.lower() and isinstance(v, (int, float)):
                                total += 1
                                if v > 50:
                                    bullish += 1
                                else:
                                    bearish += 1
                    if total > 0:
                        indicator_alignment = (bullish / total) * 100
                    else:
                        indicator_alignment = 50.0
                    result["indicator_alignment"] = round(indicator_alignment, 1)
            except ImportError:
                result["errors"].append("indicators: module not available")
            except Exception as exc:
                result["errors"].append(f"indicators: {exc}")
                logger.debug("Indicators failed for %s: %s", symbol, exc)

        # Step 3: News sentiment
        news_score: float | None = None
        if body.run_news:
            try:
                # Try reading cached news sentiment (same pattern as news.py)
                short_sym = symbol.replace("=F", "").replace("KRAKEN:", "")
                if redis:
                    from lib.analysis.sentiment.news_sentiment import load_sentiment_from_cache

                    ns = load_sentiment_from_cache(short_sym, redis)
                    if ns is not None:
                        from dataclasses import asdict

                        ns_dict = asdict(ns)
                        score = _safe_float(ns_dict.get("weighted_hybrid", 0.0))
                        result["news_sentiment"] = round(score, 4)
                        result["news_signal"] = ns_dict.get("signal", "NEUTRAL")
                        result["news_article_count"] = ns_dict.get("article_count", 0)
                        # Normalize to 0-100 scale (sentiment is typically -1 to +1)
                        news_score = (score + 1.0) * 50.0
                        news_score = max(0.0, min(100.0, news_score))
            except ImportError:
                result["errors"].append("news: sentiment module not available")
            except Exception as exc:
                result["errors"].append(f"news: {exc}")
                logger.debug("News sentiment failed for %s: %s", symbol, exc)

        # Step 4: CNN prediction
        cnn_conf: float | None = None
        if body.run_cnn:
            try:
                from lib.services.training.trainer_server import predict_single

                prediction = predict_single(symbol)
                if prediction and isinstance(prediction, dict):
                    result["cnn_signal"] = prediction.get("signal", "NEUTRAL")
                    conf = _safe_float(prediction.get("confidence", 0.0))
                    result["cnn_confidence"] = round(conf, 4)
                    # Normalize confidence to 0-100 (already is if 0-1, scale up)
                    cnn_conf = conf * 100.0 if conf <= 1.0 else conf
                    cnn_conf = max(0.0, min(100.0, cnn_conf))
            except ImportError:
                result["errors"].append("cnn: trainer module not available")
            except Exception as exc:
                result["errors"].append(f"cnn: {exc}")
                logger.debug("CNN prediction failed for %s: %s", symbol, exc)

        # Step 5: Ruby signal (if requested)
        if body.run_ruby:
            try:
                if redis:
                    raw = redis.get(f"ruby:signal:{symbol}")
                    if raw:
                        ruby_data = json.loads(raw.decode() if isinstance(raw, bytes) else raw)
                        result["ruby_signal"] = ruby_data.get("signal", "NEUTRAL")
                        result["ruby_confidence"] = _safe_float(ruby_data.get("confidence", 0.0))
            except Exception as exc:
                result["errors"].append(f"ruby: {exc}")
                logger.debug("Ruby signal failed for %s: %s", symbol, exc)

        # Compute overall score
        result["overall_score"] = round(
            _compute_overall_score(cnn_conf, indicator_alignment, news_score),
            1,
        )

        # Cache result
        _store_analysis_cache(redis, symbol, result)
        results.append(result)

    return {
        "count": len(results),
        "results": results,
        "timestamp": datetime.now(tz=UTC).isoformat(),
    }


# ---------------------------------------------------------------------------
# GET /api/pretrade/analysis/{symbol} — Get cached analysis for a symbol
# ---------------------------------------------------------------------------


@router.get("/analysis/{symbol:path}")
async def get_analysis(symbol: str, request: Request) -> dict:
    """Get the latest cached analysis for a symbol.

    Returns the full analysis result from Redis, or 404 if not analyzed yet.
    The ``{symbol:path}`` path converter allows slashes in Kraken tickers.
    """
    redis = _get_redis(request)
    cached = _read_analysis_cache(redis, symbol)

    if cached is None:
        raise HTTPException(
            status_code=404,
            detail=f"No analysis cached for {symbol}. Run POST /api/pretrade/analyze first.",
        )

    return cached


# ---------------------------------------------------------------------------
# POST /api/pretrade/select — Mark assets as selected for monitoring
# ---------------------------------------------------------------------------


@router.post("/select")
async def select_assets(body: SelectRequest, request: Request) -> dict:
    """Mark symbols as selected for monitoring/trading.

    Stores selected symbols in a Redis set with 24-hour TTL.
    Replaces any previous selection.
    """
    redis = _get_redis(request)

    if redis is None:
        return {
            "status": "warning",
            "message": "Redis not available — selection stored in-memory only",
            "selected": body.symbols,
        }

    try:
        # Clear previous selection and store new one
        pipe = redis.pipeline()
        pipe.delete(_SELECTED_KEY)
        if body.symbols:
            pipe.sadd(_SELECTED_KEY, *body.symbols)
            pipe.expire(_SELECTED_KEY, _SELECTED_TTL)
        pipe.execute()

        logger.info("Pre-trade selection updated: %s", body.symbols)
        return {
            "status": "ok",
            "message": f"Selected {len(body.symbols)} assets for monitoring",
            "selected": body.symbols,
        }
    except Exception as exc:
        logger.error("Failed to store selection: %s", exc)
        raise HTTPException(status_code=500, detail=f"Failed to store selection: {exc}") from exc


# ---------------------------------------------------------------------------
# GET /api/pretrade/selected — Get currently selected assets
# ---------------------------------------------------------------------------


@router.get("/selected")
async def get_selected(request: Request) -> dict:
    """Get the list of currently selected symbols for monitoring."""
    redis = _get_redis(request)

    if redis is None:
        return {"selected": [], "count": 0}

    try:
        members = redis.smembers(_SELECTED_KEY)
        selected = sorted(m.decode() if isinstance(m, bytes) else m for m in (members or set()))
        return {
            "selected": selected,
            "count": len(selected),
        }
    except Exception as exc:
        logger.warning("Failed to read selection: %s", exc)
        return {"selected": [], "count": 0, "error": str(exc)}


# ---------------------------------------------------------------------------
# GET /api/pretrade/watchlist — Live watchlist with prices + signals
# ---------------------------------------------------------------------------


@router.get("/watchlist")
async def watchlist(request: Request) -> dict:
    """Live watchlist for selected assets.

    For each selected symbol returns: price, change, volume, latest
    signals, and P&L if a sim position exists.  Designed to be polled
    every 2 seconds from the frontend.
    """
    redis = _get_redis(request)

    if redis is None:
        return {"items": [], "count": 0}

    # Get selected symbols
    try:
        members = redis.smembers(_SELECTED_KEY)
        selected = sorted(m.decode() if isinstance(m, bytes) else m for m in (members or set()))
    except Exception:
        selected = []

    if not selected:
        return {"items": [], "count": 0, "message": "No assets selected"}

    items: list[dict[str, Any]] = []

    for symbol in selected:
        item: dict[str, Any] = {
            "symbol": symbol,
            "price": None,
            "daily_change_pct": None,
            "volume": None,
            "signals": {},
            "sim_position": None,
        }

        # Read live price
        if symbol.startswith("KRAKEN:"):
            item["price"] = _read_kraken_price(redis, symbol)
        else:
            item["price"] = _read_futures_price(redis, symbol)

        # Read cached daily change
        try:
            raw = redis.get(f"pretrade:daily_change:{symbol}")
            if raw:
                change_data = json.loads(raw.decode() if isinstance(raw, bytes) else raw)
                item["daily_change_pct"] = _safe_float(change_data.get("change_pct"))
                item["volume"] = _safe_float(change_data.get("volume_24h"))
        except Exception:
            pass

        # Read cached analysis for signals
        cached = _read_analysis_cache(redis, symbol)
        if cached:
            item["signals"] = {
                "cnn": cached.get("cnn_signal"),
                "cnn_confidence": cached.get("cnn_confidence"),
                "ruby": cached.get("ruby_signal"),
                "news": cached.get("news_signal"),
                "news_sentiment": cached.get("news_sentiment"),
                "overall_score": cached.get("overall_score"),
                "indicator_alignment": cached.get("indicator_alignment"),
            }
            item["last_analyzed"] = cached.get("analyzed_at")

        # Read sim position if available
        try:
            sim_raw = redis.get(f"sim:position:{symbol}")
            if sim_raw:
                sim_data = json.loads(sim_raw.decode() if isinstance(sim_raw, bytes) else sim_raw)
                item["sim_position"] = {
                    "side": sim_data.get("side"),
                    "qty": sim_data.get("qty") or sim_data.get("contracts"),
                    "entry": sim_data.get("entry") or sim_data.get("entry_price"),
                    "unrealized_pnl": sim_data.get("unrealized_pnl") or sim_data.get("pnl"),
                }
        except Exception:
            pass

        items.append(item)

    return {
        "items": items,
        "count": len(items),
        "timestamp": datetime.now(tz=UTC).isoformat(),
    }
