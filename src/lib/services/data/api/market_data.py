"""
Market Data API router — proxies ALL Massive.com futures data endpoints
so clients never need to initialise their own Massive client or call
yfinance directly.

All data fetching flows through the data-service's existing cache layer
and MassiveDataProvider.

Endpoints:
    ── Aggregate Bars (OHLC) ──
    GET  /data/ohlcv              — Fetch OHLCV bars for a single ticker
    POST /data/ohlcv/bulk         — Fetch OHLCV bars for multiple tickers

    ── Daily Bars ──
    GET  /data/daily              — Fetch daily bars for a single ticker
    POST /data/daily/bulk         — Fetch daily bars for multiple tickers

    ── Contracts ──
    GET  /data/contracts          — List/filter all futures contracts
    GET  /data/contracts/{ticker} — Contract overview (specs for one ticker)
    GET  /data/contracts/resolve  — Resolve Yahoo ticker → front-month contract

    ── Products ──
    GET  /data/products           — List/filter all futures products
    GET  /data/products/{code}    — Product overview for a product code

    ── Schedules ──
    GET  /data/schedules          — Trading schedules (all or by product)

    ── Snapshots ──
    GET  /data/snapshot           — Real-time snapshot for one ticker
    POST /data/snapshot/bulk      — Snapshots for multiple tickers

    ── Trades & Quotes ──
    GET  /data/trades             — Recent trades for a ticker
    GET  /data/quotes             — Top-of-book bid/ask for a ticker

    ── Market Operations ──
    GET  /data/market_status      — Market status for products/exchanges
    GET  /data/exchanges          — List of supported futures exchanges

    ── Data Source ──
    GET  /data/source             — Current data source + Massive availability
"""

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from lib.core.cache import get_daily, get_data, get_data_source
from lib.integrations.massive_client import (
    get_massive_provider,
    is_massive_available,
)

logger = logging.getLogger("api.market_data")

router = APIRouter(tags=["Market Data"])


# ---------------------------------------------------------------------------
# Helpers — DataFrame ↔ JSON serialisation
# ---------------------------------------------------------------------------


def _df_to_json(df) -> dict | None:
    """Serialise a pandas DataFrame to a JSON-safe dict.

    Returns None if the DataFrame is None or empty.
    Uses pandas' 'split' orientation so the client can reconstruct
    the DataFrame cheaply with pd.DataFrame(**payload).
    Format: {"columns": [...], "index": [...], "data": [[...]]}
    """
    if df is None or df.empty:
        return None
    return df.to_dict(orient="split")


def _get_provider():
    """Get the Massive provider, raising 503 if unavailable."""
    provider = get_massive_provider()
    if not provider.is_available:
        raise HTTPException(
            status_code=503,
            detail="Massive API not available — set MASSIVE_API_KEY",
        )
    return provider


# ---------------------------------------------------------------------------
# Request models for bulk endpoints
# ---------------------------------------------------------------------------


class OHLCVBulkRequest(BaseModel):
    """Request body for bulk OHLCV fetch."""

    tickers: list[str]
    interval: str = "5m"
    period: str = "5d"


class DailyBulkRequest(BaseModel):
    """Request body for bulk daily bar fetch."""

    tickers: list[str]
    period: str = "10d"


class SnapshotBulkRequest(BaseModel):
    """Request body for bulk snapshot fetch."""

    tickers: list[str]


# ═══════════════════════════════════════════════════════════════════════════
# AGGREGATE BARS (OHLC)
# Maps to: GET /futures/vX/aggs/{ticker}
# ═══════════════════════════════════════════════════════════════════════════


@router.get("/ohlcv")
def get_ohlcv(
    ticker: str = Query(..., description="Futures ticker, e.g. GC=F"),
    interval: str = Query("5m", description="Bar interval: 1m, 5m, 15m, 1h, 1d"),
    period: str = Query("5d", description="Lookback period: 1d, 5d, 15d, 1mo, 3mo"),
):
    """Fetch OHLCV bars for a single ticker.

    Data source priority:
      1. Redis cache (if fresh data exists)
      2. Massive.com REST API
      3. yfinance fallback

    Returns a JSON dict in pandas 'split' format:
      {"columns": [...], "index": [...], "data": [[...]]}
    or null if no data is available.
    """
    try:
        df = get_data(ticker, interval, period)
        payload = _df_to_json(df)
        return {
            "ticker": ticker,
            "interval": interval,
            "period": period,
            "bars": len(df) if not df.empty else 0,
            "data": payload,
        }
    except Exception as exc:
        logger.error("OHLCV fetch failed for %s %s/%s: %s", ticker, interval, period, exc)
        raise HTTPException(status_code=500, detail=f"Data fetch failed: {exc}") from exc


@router.post("/ohlcv/bulk")
def get_ohlcv_bulk(request: OHLCVBulkRequest):
    """Fetch OHLCV bars for multiple tickers in a single call.

    Avoids N sequential HTTP round-trips from the client.
    Returns a dict keyed by ticker.
    """
    results: dict[str, Any] = {}
    for ticker in request.tickers:
        try:
            df = get_data(ticker, request.interval, request.period)
            results[ticker] = {
                "bars": len(df) if not df.empty else 0,
                "data": _df_to_json(df),
            }
        except Exception as exc:
            logger.warning("Bulk OHLCV fetch failed for %s: %s", ticker, exc)
            results[ticker] = {"bars": 0, "data": None, "error": str(exc)}

    return {
        "interval": request.interval,
        "period": request.period,
        "results": results,
    }


# ═══════════════════════════════════════════════════════════════════════════
# DAILY BARS
# ═══════════════════════════════════════════════════════════════════════════


@router.get("/daily")
def get_daily_bars(
    ticker: str = Query(..., description="Futures ticker, e.g. GC=F"),
    period: str = Query("10d", description="Lookback period: 5d, 10d, 1mo, 3mo"),
):
    """Fetch daily OHLCV bars for a single ticker.

    Used for pivot calculations, daily change %, and pre-market scoring.
    """
    try:
        df = get_daily(ticker, period=period)
        payload = _df_to_json(df)
        return {
            "ticker": ticker,
            "period": period,
            "bars": len(df) if not df.empty else 0,
            "data": payload,
        }
    except Exception as exc:
        logger.error("Daily fetch failed for %s/%s: %s", ticker, period, exc)
        raise HTTPException(status_code=500, detail=f"Daily data fetch failed: {exc}") from exc


@router.post("/daily/bulk")
def get_daily_bulk(request: DailyBulkRequest):
    """Fetch daily bars for multiple tickers in a single call."""
    results: dict[str, Any] = {}
    for ticker in request.tickers:
        try:
            df = get_daily(ticker, period=request.period)
            results[ticker] = {
                "bars": len(df) if not df.empty else 0,
                "data": _df_to_json(df),
            }
        except Exception as exc:
            logger.warning("Bulk daily fetch failed for %s: %s", ticker, exc)
            results[ticker] = {"bars": 0, "data": None, "error": str(exc)}

    return {
        "period": request.period,
        "results": results,
    }


# ═══════════════════════════════════════════════════════════════════════════
# CONTRACTS
# Maps to: GET /futures/vX/contracts
#           GET /futures/vX/contracts/{ticker}
# ═══════════════════════════════════════════════════════════════════════════


@router.get("/contracts/resolve")
def resolve_contract(
    ticker: str = Query(..., description="Yahoo-style ticker to resolve, e.g. ES=F"),
):
    """Resolve a Yahoo-style ticker to its Massive front-month contract ticker.

    Uses a 3-tier fallback: active contracts → future expiration → root symbol.
    """
    provider = _get_provider()
    massive_ticker = provider.resolve_from_yahoo(ticker)
    if massive_ticker is None:
        raise HTTPException(
            status_code=404,
            detail=f"Could not resolve {ticker} to a Massive contract",
        )
    return {"yahoo_ticker": ticker, "massive_ticker": massive_ticker}


@router.get("/contracts/{ticker}")
def get_contract_overview(ticker: str):
    """Retrieve detailed specifications for a futures contract by its ticker.

    Maps to: GET /futures/vX/contracts/{ticker}

    Returns tick size, trading dates, order quantity limits, etc.
    """
    provider = _get_provider()
    contract = provider.get_contract(ticker)
    if contract is None:
        raise HTTPException(status_code=404, detail=f"Contract not found: {ticker}")
    return contract


@router.get("/contracts")
def list_contracts(
    product_code: str = Query(..., description="Product code, e.g. ES, GC, CL"),
    active: bool | None = Query(None, description="Filter by active status"),
    limit: int = Query(10, description="Max results", ge=1, le=100),
):
    """List/filter all available futures contracts for a product.

    Maps to: GET /futures/vX/contracts
    """
    provider = _get_provider()
    contracts = provider.get_active_contracts(product_code, limit=limit)
    # If active filter is specified, apply it client-side
    if active is not None:
        contracts = [c for c in contracts if c.get("active") == active]
    return {
        "product_code": product_code,
        "count": len(contracts),
        "contracts": contracts,
    }


# ═══════════════════════════════════════════════════════════════════════════
# PRODUCTS
# Maps to: GET /futures/vX/products
#           GET /futures/vX/products/{product_code}
# ═══════════════════════════════════════════════════════════════════════════


@router.get("/products/{product_code}")
def get_product_overview(product_code: str):
    """Retrieve detailed information about a specific futures product.

    Maps to: GET /futures/vX/products/{product_code}

    Returns asset class, trading venue, name, settlement details, etc.
    """
    provider = _get_provider()
    product = provider.get_product(product_code)
    if product is None:
        raise HTTPException(status_code=404, detail=f"Product not found: {product_code}")
    return product


@router.get("/products")
def list_products(
    name: str | None = Query(None, description="Filter by product name"),
    product_code: str | None = Query(None, description="Filter by product code"),
    trading_venue: str | None = Query(None, description="Filter by exchange, e.g. CME, NYMEX"),
    sector: str | None = Query(None, description="Filter by sector"),
    sub_sector: str | None = Query(None, description="Filter by sub-sector"),
    asset_class: str | None = Query(None, description="Filter by asset class, e.g. futures"),
    asset_sub_class: str | None = Query(None, description="Filter by asset sub-class"),
    limit: int = Query(100, description="Max results", ge=1, le=1000),
):
    """Filter through all available futures product specifications.

    Maps to: GET /futures/vX/products

    Filter by name, venue, sector/sub-sector, settlement method, etc.
    """
    provider = _get_provider()
    products = provider.get_products(
        name=name,
        product_code=product_code,
        trading_venue=trading_venue,
        sector=sector,
        sub_sector=sub_sector,
        asset_class=asset_class,
        asset_sub_class=asset_sub_class,
        limit=limit,
    )
    return {"count": len(products), "products": products}


# ═══════════════════════════════════════════════════════════════════════════
# SCHEDULES
# Maps to: GET /futures/vX/schedules
#           GET /futures/vX/products/{product_code}/schedules
# ═══════════════════════════════════════════════════════════════════════════


@router.get("/schedules")
def list_schedules(
    product_code: str | None = Query(None, description="Filter by product code, e.g. ES"),
    trading_date: str | None = Query(
        None,
        description="Session end date YYYY-MM-DD. Defaults to today.",
    ),
    trading_venue: str | None = Query(None, description="Filter by exchange, e.g. CME"),
    limit: int = Query(100, description="Max results", ge=1, le=1000),
):
    """Filter through trading schedules for futures contracts.

    Maps to: GET /futures/vX/schedules  (all products)
             GET /futures/vX/products/{product_code}/schedules  (single product)

    Returns session windows (pre-open, open, close) for the given date.
    """
    provider = _get_provider()
    schedules = provider.get_schedules(
        product_code=product_code,
        trading_date=trading_date,
        trading_venue=trading_venue,
        limit=limit,
    )
    return {"count": len(schedules), "schedules": schedules}


# ═══════════════════════════════════════════════════════════════════════════
# SNAPSHOTS
# Maps to: GET /futures/vX/snapshot
# ═══════════════════════════════════════════════════════════════════════════


@router.get("/snapshot")
def get_snapshot(
    ticker: str | None = Query(
        None,
        description="Yahoo-style ticker, e.g. ES=F (provide ticker or product_code)",
    ),
    product_code: str | None = Query(None, description="Massive product code, e.g. ES"),
):
    """Retrieve a real-time snapshot for a futures contract.

    Maps to: GET /futures/vX/snapshot

    Returns last trade, quote, session metrics (OHLCV), and settlement prices.
    Provide either ticker (Yahoo-style) or product_code.
    """
    if not ticker and not product_code:
        raise HTTPException(
            status_code=400,
            detail="Provide either 'ticker' (e.g. ES=F) or 'product_code' (e.g. ES)",
        )
    provider = _get_provider()
    snapshot = provider.get_snapshot(yahoo_ticker=ticker, product_code=product_code)
    if snapshot is None:
        raise HTTPException(
            status_code=404,
            detail=f"No snapshot found for ticker={ticker} product_code={product_code}",
        )
    return snapshot


@router.post("/snapshot/bulk")
def get_snapshots_bulk(request: SnapshotBulkRequest):
    """Retrieve real-time snapshots for multiple tickers in one call.

    Maps to: GET /futures/vX/snapshot (batched via product_code_any_of)
    """
    provider = _get_provider()
    snapshots = provider.get_all_snapshots(request.tickers)
    return {
        "count": len(snapshots),
        "snapshots": snapshots,
    }


# ═══════════════════════════════════════════════════════════════════════════
# TRADES
# Maps to: GET /futures/vX/trades/{ticker}
# ═══════════════════════════════════════════════════════════════════════════


@router.get("/trades")
def get_trades(
    ticker: str = Query(..., description="Yahoo-style ticker, e.g. ES=F"),
    minutes_back: int = Query(5, description="Minutes of history to fetch", ge=1, le=60),
    limit: int = Query(5000, description="Max trade records", ge=1, le=50000),
):
    """Find trade records with price, size, and timestamp for a futures contract.

    Maps to: GET /futures/vX/trades/{ticker}

    Returns a JSON dict in pandas 'split' format.
    """
    provider = _get_provider()
    df = provider.get_recent_trades(yahoo_ticker=ticker, minutes_back=minutes_back, limit=limit)
    return {
        "ticker": ticker,
        "minutes_back": minutes_back,
        "count": len(df) if not df.empty else 0,
        "data": _df_to_json(df),
    }


# ═══════════════════════════════════════════════════════════════════════════
# QUOTES
# Maps to: GET /futures/vX/quotes/{ticker}
# ═══════════════════════════════════════════════════════════════════════════


@router.get("/quotes")
def get_quotes(
    ticker: str = Query(..., description="Yahoo-style ticker, e.g. ES=F"),
    minutes_back: int = Query(5, description="Minutes of history to fetch", ge=1, le=60),
    limit: int = Query(5000, description="Max quote records", ge=1, le=50000),
):
    """Get top-of-book bid and ask prices for a futures contract.

    Maps to: GET /futures/vX/quotes/{ticker}

    Returns a JSON dict in pandas 'split' format with bid, bid_size, ask, ask_size.
    """
    provider = _get_provider()
    df = provider.get_quotes(yahoo_ticker=ticker, minutes_back=minutes_back, limit=limit)
    return {
        "ticker": ticker,
        "minutes_back": minutes_back,
        "count": len(df) if not df.empty else 0,
        "data": _df_to_json(df),
    }


# ═══════════════════════════════════════════════════════════════════════════
# MARKET STATUS
# Maps to: GET /futures/vX/market_status
# ═══════════════════════════════════════════════════════════════════════════


@router.get("/market_status")
def get_market_status(
    product_code: str | None = Query(None, description="Filter by product code, e.g. ES"),
):
    """Retrieve the current market status for futures products and exchanges.

    Maps to: GET /futures/vX/market_status

    Returns status (open/closed/pre-market), session times, and exchange info.
    """
    provider = _get_provider()
    statuses = provider.get_market_statuses(product_code=product_code)
    return {"count": len(statuses), "statuses": statuses}


# ═══════════════════════════════════════════════════════════════════════════
# EXCHANGES
# Maps to: GET /futures/vX/exchanges
# ═══════════════════════════════════════════════════════════════════════════


@router.get("/exchanges")
def list_exchanges():
    """Retrieve a list of supported futures exchanges.

    Maps to: GET /futures/vX/exchanges

    Returns exchange codes, names, MICs, and other details.
    """
    provider = _get_provider()
    exchanges = provider.get_exchanges()
    return {"count": len(exchanges), "exchanges": exchanges}


# ═══════════════════════════════════════════════════════════════════════════
# DATA SOURCE
# ═══════════════════════════════════════════════════════════════════════════


@router.get("/source")
def get_source():
    """Return the name of the active primary data source and Massive availability."""
    return {
        "data_source": get_data_source(),
        "massive_available": is_massive_available(),
    }
