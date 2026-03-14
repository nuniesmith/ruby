"""
Analysis API router — exposes cached FKS analysis data.

Endpoints:
    GET  /latest/{ticker}  — Full FKS analysis dict (wave + vol + sq + regime + ict + cvd)
    GET  /status           — Engine status (refresh times, live feed, optimization progress)
    POST /force_refresh    — Trigger an immediate data refresh cycle
    GET  /data_source      — Current primary data source (Massive or yfinance)
    GET  /assets           — List of tracked assets and their tickers
    GET  /scanner          — Pre-market scanner scores for all assets
"""

import json
import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Query

from lib.core.cache import (
    _cache_key,
    cache_get,
    flush_all,
    get_cached_optimization,
    get_data_source,
)
from lib.core.models import (
    ACCOUNT_PROFILES,
    ASSETS,
    CONTRACT_SPECS,
    TICKER_TO_NAME,
)

logger = logging.getLogger("api.analysis")

router = APIRouter(tags=["analysis"])

# Reference to the engine singleton — set by main.py at startup
_engine = None


def set_engine(engine):
    """Called by main.py lifespan to inject the engine singleton."""
    global _engine
    _engine = engine


def _get_engine():
    if _engine is None:
        raise HTTPException(status_code=503, detail="Engine not started yet")
    return _engine


# ---------------------------------------------------------------------------
# GET /latest/{ticker} — Full FKS analysis bundle
# ---------------------------------------------------------------------------


@router.get("/latest/{ticker}")
def get_latest_analysis(
    ticker: str,
    interval: str = Query("5m", description="Primary interval"),
    period: str = Query("5d", description="Lookback period"),
):
    """Return the full cached FKS analysis for a ticker.

    Assembles wave analysis, volatility clusters, signal quality,
    regime detection, ICT levels, and CVD data from the Redis cache
    into a single JSON response.  Returns whatever is available —
    missing components come back as null rather than raising errors.
    """
    result: dict[str, Any] = {
        "ticker": ticker,
        "name": TICKER_TO_NAME.get(ticker, ticker),
        "interval": interval,
        "period": period,
        "data_source": get_data_source(),
    }

    # Wave analysis
    wave_raw = cache_get(_cache_key("fks_wave", ticker, interval, period))
    result["wave"] = json.loads(wave_raw) if wave_raw else None

    # Volatility clusters (K-Means)
    vol_raw = cache_get(_cache_key("fks_vol", ticker, interval, period))
    result["volatility"] = json.loads(vol_raw) if vol_raw else None

    # Signal quality (5m engine cycle)
    sq_raw = cache_get(_cache_key("fks_sq", ticker, interval, period))
    result["signal_quality"] = json.loads(sq_raw) if sq_raw else None

    # Signal quality (1m live WebSocket)
    sq_1m_raw = cache_get(_cache_key("fks_sq_1m", ticker))
    result["signal_quality_1m"] = json.loads(sq_1m_raw) if sq_1m_raw else None

    # Regime detection
    regime_raw = cache_get(_cache_key("fks_regime", ticker, interval, period))
    result["regime"] = json.loads(regime_raw) if regime_raw else None

    # ICT levels
    ict_raw = cache_get(_cache_key("fks_ict", ticker, interval, period))
    result["ict"] = json.loads(ict_raw) if ict_raw else None

    # CVD (Cumulative Volume Delta)
    cvd_raw = cache_get(_cache_key("fks_cvd", ticker, interval, period))
    result["cvd"] = json.loads(cvd_raw) if cvd_raw else None

    # Optimization results
    opt = get_cached_optimization(ticker, interval, period)
    result["optimization"] = opt

    # Confluence
    confluence_raw = cache_get(_cache_key("fks_confluence", ticker, interval, period))
    result["confluence"] = json.loads(confluence_raw) if confluence_raw else None

    return result


# ---------------------------------------------------------------------------
# GET /latest — All tickers at once
# ---------------------------------------------------------------------------


@router.get("/latest")
def get_all_latest(
    interval: str = Query("5m", description="Primary interval"),
    period: str = Query("5d", description="Lookback period"),
):
    """Return cached FKS analysis for ALL tracked assets in one call."""
    results = {}
    for name, ticker in ASSETS.items():
        try:
            results[name] = get_latest_analysis(ticker, interval, period)
        except Exception as exc:
            logger.warning("Failed to get analysis for %s: %s", name, exc)
            results[name] = {"ticker": ticker, "name": name, "error": str(exc)}
    return results


# ---------------------------------------------------------------------------
# GET /status — Engine status
# ---------------------------------------------------------------------------


@router.get("/status")
def get_engine_status():
    """Return the current engine status dict.

    Includes refresh timestamps, optimization progress, live feed
    connection state, and data source information.
    """
    engine = _get_engine()
    return engine.get_status()


# ---------------------------------------------------------------------------
# POST /force_refresh — Trigger immediate refresh
# ---------------------------------------------------------------------------


@router.post("/force_refresh")
def force_refresh():
    """Trigger an immediate data refresh + optimization cycle.

    Flushes all cached data and forces the engine to re-fetch and
    re-compute everything on the next loop iteration.
    """
    engine = _get_engine()
    flush_all()
    engine.force_refresh()
    return {
        "status": "refresh_triggered",
        "message": "Cache flushed, engine will refresh on next cycle",
    }


# ---------------------------------------------------------------------------
# GET /data_source — Current data provider
# ---------------------------------------------------------------------------


@router.get("/data_source")
def get_current_data_source():
    """Return the active primary data source name."""
    return {"data_source": get_data_source()}


# ---------------------------------------------------------------------------
# GET /assets — Tracked assets
# ---------------------------------------------------------------------------


@router.get("/assets")
def list_assets():
    """List all tracked assets with their contract specifications."""
    return {
        "assets": ASSETS,
        "contract_specs": {name: {k: v for k, v in spec.items()} for name, spec in CONTRACT_SPECS.items()},
    }


# ---------------------------------------------------------------------------
# GET /accounts — Account profiles
# ---------------------------------------------------------------------------


@router.get("/accounts")
def list_accounts():
    """List available account profiles and risk parameters."""
    return ACCOUNT_PROFILES


# ---------------------------------------------------------------------------
# GET /backtest_results — Latest backtest results
# ---------------------------------------------------------------------------


@router.get("/backtest_results")
def get_backtest_results():
    """Return the latest backtest results from the engine."""
    engine = _get_engine()
    return {"results": engine.get_backtest_results()}


# ---------------------------------------------------------------------------
# GET /strategy_history — Per-asset strategy confidence tracking
# ---------------------------------------------------------------------------


@router.get("/strategy_history")
def get_strategy_history():
    """Return the strategy history dict tracking winning strategies per asset."""
    engine = _get_engine()
    return engine.get_strategy_history()


# ---------------------------------------------------------------------------
# GET /live_feed — Live WebSocket feed status
# ---------------------------------------------------------------------------


@router.get("/live_feed")
def get_live_feed_status():
    """Return the current live feed connection status and stats."""
    engine = _get_engine()
    return engine.get_live_feed_status()
