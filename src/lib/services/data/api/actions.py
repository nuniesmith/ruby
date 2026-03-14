"""
Actions API router — mutation endpoints that trigger engine operations.

Endpoints:
    POST /force_refresh      — Flush cache and trigger immediate data refresh
    POST /optimize_now       — Force re-optimization for all assets
    POST /update_settings    — Update engine settings (account size, interval, period)
    POST /live_feed/start    — Start the Massive WebSocket live feed
    POST /live_feed/stop     — Stop the Massive WebSocket live feed
    POST /live_feed/upgrade  — Upgrade feed to include quotes
    POST /live_feed/downgrade — Downgrade feed to bars + trades only
"""

import contextlib
import logging
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from lib.core.cache import clear_cached_optimization, flush_all
from lib.core.models import ASSETS

_EST = ZoneInfo("America/New_York")
logger = logging.getLogger("api.actions")

router = APIRouter(tags=["actions"])


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------


class UpdateSettingsRequest(BaseModel):
    """Request body for updating engine settings at runtime."""

    account_size: int | None = Field(
        None,
        description="New account size (50000, 100000, or 150000)",
    )
    interval: str | None = Field(
        None,
        description="New primary interval (e.g. '5m', '15m')",
    )
    period: str | None = Field(
        None,
        description="New lookback period (e.g. '5d', '10d')",
    )


# ---------------------------------------------------------------------------
# Engine accessor — set by main.py after lifespan starts
# ---------------------------------------------------------------------------

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
# POST /force_refresh
# ---------------------------------------------------------------------------


@router.post("/force_refresh")
def force_refresh():
    """Flush all cached data and trigger an immediate engine refresh.

    This clears the Redis/in-memory cache and tells the engine to
    re-fetch all market data, recompute indicators, and re-run
    optimizations on the next loop iteration.
    """
    engine = _get_engine()
    flush_all()
    engine.force_refresh()
    return {
        "status": "refresh_triggered",
        "message": "Cache flushed. Engine will refresh on next cycle.",
        "timestamp": datetime.now(tz=_EST).isoformat(),
    }


# ---------------------------------------------------------------------------
# POST /optimize_now
# ---------------------------------------------------------------------------


@router.post("/optimize_now")
def optimize_now():
    """Force an immediate re-optimization cycle for all assets.

    Clears cached optimization results so the engine's next optimization
    pass will run fresh Optuna trials for every asset × strategy pair.
    """
    engine = _get_engine()

    # Clear optimization cache so engine re-runs on next cycle
    for _name, ticker in ASSETS.items():
        with contextlib.suppress(Exception):
            clear_cached_optimization(ticker, engine.interval, engine.period)

    engine.force_refresh()
    return {
        "status": "optimization_triggered",
        "message": "Optimization cache cleared. Engine will re-optimize on next cycle.",
        "assets": list(ASSETS.keys()),
        "timestamp": datetime.now(tz=_EST).isoformat(),
    }


# ---------------------------------------------------------------------------
# POST /update_settings
# ---------------------------------------------------------------------------


@router.post("/update_settings")
def update_settings(req: UpdateSettingsRequest):
    """Update engine settings at runtime without restarting.

    Only provided fields are updated; omitted fields stay unchanged.
    After updating, a refresh is triggered so the engine picks up
    the new parameters immediately.
    """
    engine = _get_engine()
    changed: dict[str, Any] = {}

    kwargs: dict[str, Any] = {}
    if req.account_size is not None:
        if req.account_size not in (50_000, 100_000, 150_000):
            raise HTTPException(
                status_code=400,
                detail="account_size must be 50000, 100000, or 150000",
            )
        kwargs["account_size"] = req.account_size
        changed["account_size"] = req.account_size

    if req.interval is not None:
        allowed_intervals = {"1m", "2m", "5m", "15m", "30m", "60m", "1h", "1d"}
        if req.interval not in allowed_intervals:
            raise HTTPException(
                status_code=400,
                detail=f"interval must be one of {sorted(allowed_intervals)}",
            )
        kwargs["interval"] = req.interval
        changed["interval"] = req.interval

    if req.period is not None:
        kwargs["period"] = req.period
        changed["period"] = req.period

    if not kwargs:
        return {
            "status": "no_change",
            "message": "No settings were provided to update.",
            "timestamp": datetime.now(tz=_EST).isoformat(),
        }

    engine.update_settings(**kwargs)
    logger.info("Engine settings updated: %s", changed)

    return {
        "status": "updated",
        "changed": changed,
        "timestamp": datetime.now(tz=_EST).isoformat(),
    }


# ---------------------------------------------------------------------------
# Live feed controls
# ---------------------------------------------------------------------------


@router.post("/live_feed/start")
def start_live_feed():
    """Start the Massive WebSocket live feed for real-time 1m bars."""
    engine = _get_engine()
    success = engine.start_live_feed()
    if success:
        return {
            "status": "started",
            "message": "Massive WebSocket live feed is now running.",
            "timestamp": datetime.now(tz=_EST).isoformat(),
        }
    return {
        "status": "failed",
        "message": "Could not start live feed. Check MASSIVE_API_KEY is set.",
        "timestamp": datetime.now(tz=_EST).isoformat(),
    }


@router.post("/live_feed/stop")
async def stop_live_feed():
    """Stop the Massive WebSocket live feed."""
    engine = _get_engine()
    await engine.stop_live_feed()
    return {
        "status": "stopped",
        "message": "Live feed stopped.",
        "timestamp": datetime.now(tz=_EST).isoformat(),
    }


@router.post("/live_feed/upgrade")
def upgrade_live_feed():
    """Upgrade the live feed to include quote data (bid/ask)."""
    engine = _get_engine()
    engine.upgrade_live_feed()
    return {
        "status": "upgraded",
        "message": "Live feed upgraded to include quotes.",
        "timestamp": datetime.now(tz=_EST).isoformat(),
    }


@router.post("/live_feed/downgrade")
def downgrade_live_feed():
    """Downgrade the live feed to bars + trades only (no quotes)."""
    engine = _get_engine()
    engine.downgrade_live_feed()
    return {
        "status": "downgraded",
        "message": "Live feed downgraded (no quotes).",
        "timestamp": datetime.now(tz=_EST).isoformat(),
    }
