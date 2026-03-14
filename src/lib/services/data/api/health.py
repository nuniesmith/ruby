"""
Health check, metrics, backfill status, and system health bar API router.

Provides:
    GET /health              — Service health check (Redis, Postgres, engine, live feed)
    GET /metrics             — Lightweight operational metrics
    GET /backfill/status     — Historical data backfill status (bar counts, date ranges)
    GET /backfill/gaps/{sym} — Gap analysis for a specific symbol's stored bars
    GET /api/health          — System health status JSON (all services + broker + CNN)
    GET /api/health/html     — System health bar HTML fragment (polled by dashboard HTMX)
"""

import contextlib
import json
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from fastapi import APIRouter
from fastapi.responses import HTMLResponse, JSONResponse

_EST = ZoneInfo("America/New_York")
logger = logging.getLogger("api.health")

router = APIRouter(tags=["health"])

# Maximum age before a model is considered stale (hours)
_MODEL_STALE_HOURS = 26


def _check_model_health() -> dict[str, Any]:
    """Check CNN model existence and freshness.

    Returns a dict with:
      - ``status``: ``"ok"`` | ``"stale"`` | ``"missing"``
      - ``available``: bool
      - ``champion_path``: str | None
      - ``size_mb``: float
      - ``last_retrain``: ISO timestamp from promotion meta, or None
      - ``last_retrain_ago``: human-readable age string, or None
      - ``val_accuracy``: float from last promotion meta, or None
      - ``stale``: bool — True when model hasn't been retrained in > 26 h
    """
    result: dict[str, Any] = {
        "status": "missing",
        "available": False,
        "champion_path": None,
        "size_mb": 0.0,
        "last_retrain": None,
        "last_retrain_ago": None,
        "val_accuracy": None,
        "precision": None,
        "recall": None,
        "stale": False,
        "total_checkpoints": 0,
        # When was sync_models.sh last run? (mtime of meta.json sidecar)
        "last_sync_time": None,
        "last_sync_ago": None,
    }

    # Locate models/ directory — works both in Docker (/app/models) and bare-metal
    _model_dir_candidates = [
        Path("/app/models"),
        Path(__file__).resolve().parents[5] / "models",
        Path(__file__).resolve().parents[4] / "models",
    ]
    model_dir: Path | None = None
    for _c in _model_dir_candidates:
        if _c.is_dir():
            model_dir = _c
            break

    if model_dir is None:
        return result

    # Count all checkpoints
    all_pt = list(model_dir.glob("breakout_cnn_*.pt"))
    result["total_checkpoints"] = len(all_pt)

    # Check for the champion model
    champion = model_dir / "breakout_cnn_best.pt"
    if not champion.is_file():
        # Fall back to newest checkpoint by mtime
        if all_pt:
            champion = max(all_pt, key=lambda p: p.stat().st_mtime)
        else:
            return result  # no models at all

    result["available"] = True
    result["champion_path"] = str(champion)
    stat = champion.stat()
    result["size_mb"] = round(stat.st_size / (1024 * 1024), 1)

    now_et = datetime.now(tz=_EST)

    # last_sync_time — mtime of the meta.json sidecar written by sync_models.sh.
    # This tells us when the operator last pulled from the rb repo, which may
    # be more recent than the model's own promoted_at training timestamp.
    meta_path = model_dir / "breakout_cnn_best_meta.json"
    if meta_path.is_file():
        sync_dt = datetime.fromtimestamp(meta_path.stat().st_mtime, tz=_EST)
        result["last_sync_time"] = sync_dt.isoformat()
        sync_delta = now_et - sync_dt
        sync_hours = sync_delta.total_seconds() / 3600
        if sync_hours < 1:
            result["last_sync_ago"] = f"{int(sync_delta.total_seconds() / 60)}m ago"
        elif sync_hours < 24:
            result["last_sync_ago"] = f"{sync_hours:.1f}h ago"
        else:
            result["last_sync_ago"] = f"{sync_delta.days}d ago"

    # Load promotion metadata for accurate retrain timestamp + accuracy
    promoted_at: datetime | None = None

    if meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text())
            promoted_str = meta.get("promoted_at")
            if promoted_str:
                promoted_at = datetime.fromisoformat(promoted_str)
                if promoted_at.tzinfo is None:
                    promoted_at = promoted_at.replace(tzinfo=_EST)
                result["last_retrain"] = promoted_at.isoformat()

                delta = now_et - promoted_at
                hours = delta.total_seconds() / 3600
                if hours < 1:
                    result["last_retrain_ago"] = f"{int(delta.total_seconds() / 60)}m ago"
                elif hours < 24:
                    result["last_retrain_ago"] = f"{hours:.1f}h ago"
                else:
                    result["last_retrain_ago"] = f"{delta.days}d ago"

            val_acc = meta.get("val_accuracy")
            if val_acc is not None:
                result["val_accuracy"] = round(float(val_acc), 1)
            precision = meta.get("precision")
            if precision is not None:
                result["precision"] = round(float(precision), 3)
            recall = meta.get("recall")
            if recall is not None:
                result["recall"] = round(float(recall), 3)

        except Exception as exc:
            logger.debug("_check_model_health: could not read meta JSON: %s", exc)

    # Fall back to file mtime if no promotion metadata
    if promoted_at is None:
        promoted_at = datetime.fromtimestamp(stat.st_mtime, tz=_EST)
        result["last_retrain"] = promoted_at.isoformat()
        delta = now_et - promoted_at
        hours = delta.total_seconds() / 3600
        if hours < 1:
            result["last_retrain_ago"] = f"{int(delta.total_seconds() / 60)}m ago"
        elif hours < 24:
            result["last_retrain_ago"] = f"{hours:.1f}h ago"
        else:
            result["last_retrain_ago"] = f"{delta.days}d ago"

    # Staleness check
    stale_threshold = timedelta(hours=_MODEL_STALE_HOURS)
    is_stale = (now_et - promoted_at) > stale_threshold
    result["stale"] = is_stale
    result["status"] = "stale" if is_stale else "ok"

    return result


def _check_redis() -> dict[str, Any]:
    """Check Redis connectivity."""
    try:
        from lib.core.cache import REDIS_AVAILABLE, _r

        if REDIS_AVAILABLE and _r is not None:
            _r.ping()
            return {"status": "ok", "connected": True}
        return {"status": "unavailable", "connected": False}
    except Exception as exc:
        return {"status": "error", "connected": False, "error": str(exc)}


def _check_postgres() -> dict[str, Any]:
    """Check Postgres connectivity."""
    database_url = os.getenv("DATABASE_URL", "")
    if not database_url.startswith("postgresql"):
        return {"status": "not_configured", "connected": False}
    try:
        from lib.core.models import _get_conn

        conn = _get_conn()
        try:
            conn.execute("SELECT 1")
            return {"status": "ok", "connected": True}
        finally:
            conn.close()
    except Exception as exc:
        return {"status": "error", "connected": False, "error": str(exc)}


def _get_engine_or_none():
    """Try to get the engine singleton without raising."""
    try:
        from lib.trading.engine import get_engine

        return get_engine()
    except Exception:
        return None


def _check_kraken() -> dict[str, Any]:
    """Check Kraken exchange connectivity and WebSocket feed status.

    Returns a dict with:
      - ``enabled``: bool — whether ENABLE_KRAKEN_CRYPTO is set
      - ``status``: ``"ok"`` | ``"degraded"`` | ``"disabled"`` | ``"error"``
      - ``api_connected``: bool — REST API ping succeeded
      - ``authenticated``: bool — API key + secret configured
      - ``ws_connected``: bool — WebSocket feed is connected
      - ``ws_running``: bool — WebSocket feed thread is alive
      - ``pairs_count``: int — number of tracked crypto pairs
      - ``bar_count``: int — total OHLC bars received via WS
      - ``trade_count``: int — total trades received via WS
    """
    result: dict[str, Any] = {
        "enabled": False,
        "status": "disabled",
        "api_connected": False,
        "authenticated": False,
        "ws_connected": False,
        "ws_running": False,
        "pairs_count": 0,
        "bar_count": 0,
        "trade_count": 0,
    }

    try:
        from lib.core.models import ENABLE_KRAKEN_CRYPTO

        result["enabled"] = ENABLE_KRAKEN_CRYPTO
        if not ENABLE_KRAKEN_CRYPTO:
            return result
    except ImportError:
        return result

    # Check REST API connectivity
    try:
        from lib.integrations.kraken_client import get_kraken_provider

        provider = get_kraken_provider()
        if provider is not None and provider.is_available:
            result["authenticated"] = provider.has_auth
            try:
                provider.get_server_time()
                result["api_connected"] = True
            except Exception as exc:
                logger.debug("Kraken API ping failed: %s", exc)
    except ImportError:
        pass

    # Check WebSocket feed
    try:
        from lib.integrations.kraken_client import KRAKEN_PAIRS, get_kraken_feed

        result["pairs_count"] = len(KRAKEN_PAIRS)

        feed = get_kraken_feed()
        if feed is not None:
            result["ws_running"] = feed.is_running
            result["ws_connected"] = feed.is_connected
            result["bar_count"] = feed.bar_count
            result["trade_count"] = feed.trade_count
    except ImportError:
        pass

    # Determine overall status
    if result["api_connected"] and result["ws_connected"]:
        result["status"] = "ok"
    elif result["api_connected"]:
        result["status"] = "degraded"  # REST works but WS is down
    elif result["enabled"]:
        result["status"] = "error"
    else:
        result["status"] = "disabled"

    return result


@router.get("/health")
def health():
    """Service health check.

    Returns the status of all critical subsystems:
    - Redis cache connectivity
    - Postgres database connectivity
    - Engine running state
    - Massive WebSocket live feed
    - Data source (Massive vs yfinance vs Kraken)
    - CNN model existence, accuracy, and staleness
    - Kraken exchange connectivity + WebSocket feed
    - Database path
    """
    redis_status = _check_redis()
    postgres_status = _check_postgres()
    model_status = _check_model_health()
    kraken_status = _check_kraken()
    engine = _get_engine_or_none()

    engine_status = "not_initialized"
    live_feed_status: dict[str, Any] = {"status": "unknown"}
    data_source = "unknown"

    if engine is not None:
        try:
            status = engine.get_status()
            engine_status = status.get("engine", "unknown")
            live_feed_status = status.get("live_feed", {"status": "unknown"})
            data_source = live_feed_status.get("data_source", "unknown")
        except Exception as exc:
            engine_status = f"error: {exc}"

    db_path = os.getenv("DB_PATH", "futures_journal.db")

    # Overall status: degraded if engine not running, model missing, or model stale
    overall_ok = engine_status == "running" and model_status["available"] and not model_status["stale"]

    # --- Update Prometheus gauges ---
    with contextlib.suppress(Exception):
        from lib.services.data.api.metrics import (
            update_engine_up,
            update_model_stale,
            update_postgres_status,
            update_redis_status,
        )

        update_redis_status(redis_status.get("connected", False))
        update_postgres_status(postgres_status.get("connected", False))
        update_engine_up(engine_status == "running")
        update_model_stale(model_status.get("stale", False))

    return {
        "status": "ok" if overall_ok else "degraded",
        "timestamp": datetime.now(tz=_EST).isoformat(),
        "components": {
            "redis": redis_status,
            "postgres": postgres_status,
            "engine": {"status": engine_status},
            "live_feed": live_feed_status,
            "data_source": data_source,
            "model": model_status,
            "kraken": kraken_status,
            "database": {"path": db_path},
        },
    }


@router.get("/metrics")
def metrics():
    """Lightweight operational metrics.

    Returns counts and timing information useful for monitoring
    the data service without overwhelming detail.
    """
    engine = _get_engine_or_none()

    result = {
        "timestamp": datetime.now(tz=_EST).isoformat(),
        "engine_running": False,
        "data_refresh": {},
        "optimization": {},
        "backtest": {},
        "live_feed": {},
        "backtest_results_count": 0,
        "tracked_assets_count": 0,
    }

    if engine is not None:
        try:
            status = engine.get_status()
            result["engine_running"] = status.get("engine") == "running"
            result["data_refresh"] = status.get("data_refresh", {})
            result["optimization"] = status.get("optimization", {})
            result["backtest"] = status.get("backtest", {})
            result["live_feed"] = status.get("live_feed", {})

            with contextlib.suppress(Exception):
                result["backtest_results_count"] = len(engine.get_backtest_results())

            try:
                from lib.core.models import ASSETS

                result["tracked_assets_count"] = len(ASSETS)
            except Exception:
                pass
        except Exception as exc:
            result["error"] = str(exc)

    # Redis key count (if available)
    try:
        from lib.core.cache import REDIS_AVAILABLE, _r

        if REDIS_AVAILABLE and _r is not None:
            futures_keys = list(_r.scan_iter("futures:*", count=1000))
            result["redis_cached_keys"] = len(futures_keys)
    except Exception:
        result["redis_cached_keys"] = -1

    return result


# ---------------------------------------------------------------------------
# Backfill status endpoints (TASK-204)
# ---------------------------------------------------------------------------


@router.get("/backfill/status")
def backfill_status():
    """Return the current historical data backfill status.

    Shows per-symbol bar counts, date ranges, and total stored bars.
    Useful for monitoring whether backfill is running and how much
    data is available for optimization and backtesting.
    """
    try:
        from lib.services.engine.backfill import get_backfill_status

        status = get_backfill_status()
        return {
            "status": "ok",
            "timestamp": datetime.now(tz=_EST).isoformat(),
            **status,
        }
    except ImportError:
        return {
            "status": "unavailable",
            "timestamp": datetime.now(tz=_EST).isoformat(),
            "message": "Backfill module not available",
            "symbols": [],
            "total_bars": 0,
        }
    except Exception as exc:
        return {
            "status": "error",
            "timestamp": datetime.now(tz=_EST).isoformat(),
            "error": str(exc),
            "symbols": [],
            "total_bars": 0,
        }


@router.get("/backfill/gaps/{symbol}")
def backfill_gaps(symbol: str, days_back: int = 30):
    """Analyse gaps in stored historical data for a specific symbol.

    Args:
        symbol: Ticker symbol (e.g. ``MGC=F``). URL-encode the ``=`` as ``%3D``.
        days_back: Number of calendar days to analyse (default 30).

    Returns:
        Gap report with total bars, expected bars, coverage percentage,
        and a list of significant gaps (>30 minutes).
    """
    try:
        from lib.services.engine.backfill import get_gap_report

        report = get_gap_report(symbol, days_back=days_back)
        return {
            "status": "ok",
            "timestamp": datetime.now(tz=_EST).isoformat(),
            **report,
        }
    except ImportError:
        return {
            "status": "unavailable",
            "timestamp": datetime.now(tz=_EST).isoformat(),
            "message": "Backfill module not available",
            "symbol": symbol,
            "total_bars": 0,
        }
    except Exception as exc:
        return {
            "status": "error",
            "timestamp": datetime.now(tz=_EST).isoformat(),
            "symbol": symbol,
            "error": str(exc),
        }


# ---------------------------------------------------------------------------
# System health bar — full-stack health probe used by the dashboard header
# ---------------------------------------------------------------------------


def _cnn_model_on_disk() -> bool:
    """Return True if a usable CNN model file exists on disk."""
    override = os.getenv("CNN_MODEL_PATH", "")
    if override:
        return os.path.isfile(override)

    here = os.path.dirname(__file__)
    root = os.path.normpath(os.path.join(here, "..", "..", "..", "..", ".."))

    pt_path = os.path.join(root, "models", "breakout_cnn_best.pt")
    return os.path.isfile(pt_path)


def _compute_system_health() -> dict[str, Any]:
    """Compute full system health status from all available cache sources.

    Returns a dict covering:
        - Core service indicators (data, engine, redis, postgres)
        - Companion service indicators (charting, trainer, grafana, prometheus)
        - Broker / TradingView heartbeat fields
        - CNN model presence
    """
    result: dict[str, Any] = {
        "data_service_up": True,
        "engine_up": False,
        "redis_up": False,
        "postgres_up": False,
        "charting_up": False,
        "trainer_up": False,
        "grafana_up": False,
        "prometheus_up": False,
        "broker_connected": False,
        "broker_state": "disconnected",
        "broker_version": "",
        "broker_account": "",
        "broker_age_seconds": -1,
        "positions_count": 0,
        "risk_blocked": False,
        "last_heartbeat": None,
        "cnn_model_on_disk": False,
        "ruby_attached": False,
        "signalbus_active": False,
        "signalbus_pending": 0,
        "breakout_instruments": 0,
    }

    # Redis
    try:
        from lib.core.cache import REDIS_AVAILABLE, _r

        if REDIS_AVAILABLE and _r is not None:
            _r.ping()
            result["redis_up"] = True
    except Exception:
        pass

    # Postgres
    try:
        database_url = os.getenv("DATABASE_URL", "")
        if database_url.startswith("postgresql"):
            from lib.core.models import _get_conn

            conn = _get_conn()
            try:
                conn.execute("SELECT 1")
                result["postgres_up"] = True
            finally:
                conn.close()
    except Exception:
        pass

    # Engine (Redis-cached engine status)
    try:
        from lib.core.cache import cache_get as _cg

        raw_status = _cg("engine:status")
        if raw_status:
            eng = json.loads(raw_status)
            result["engine_up"] = eng.get("engine") == "running"
    except Exception:
        pass

    # Broker heartbeat
    heartbeat = None
    try:
        from lib.core.cache import cache_get

        raw = cache_get("futures:broker_heartbeat:current")
        if raw:
            heartbeat = json.loads(raw)
    except Exception:
        pass

    if heartbeat:
        received_at = heartbeat.get("received_at", "")
        account = heartbeat.get("account", "")
        state = heartbeat.get("state", "unknown")
        version = heartbeat.get("broker_version", "")

        result["broker_account"] = account
        result["broker_state"] = state
        result["broker_version"] = version
        result["positions_count"] = heartbeat.get("positions", 0)
        result["risk_blocked"] = heartbeat.get("riskBlocked", False)
        result["last_heartbeat"] = received_at

        if received_at:
            try:
                dt = datetime.fromisoformat(received_at)
                age = (datetime.now(tz=_EST) - dt).total_seconds()
                result["broker_age_seconds"] = round(age, 1)
                connected = age < 60
                result["broker_connected"] = connected
            except Exception:
                pass

    # CNN model
    with contextlib.suppress(Exception):
        result["cnn_model_on_disk"] = _cnn_model_on_disk()

    # Companion services (best-effort HTTP probes)
    _companion_services = [
        ("charting_up", os.getenv("CHARTING_SERVICE_URL", "http://charting:8003"), "/health"),
        ("trainer_up", os.getenv("TRAINER_SERVICE_URL", "http://trainer:8200"), "/health"),
        ("grafana_up", os.getenv("GRAFANA_URL", "http://grafana:3000"), "/api/health"),
        ("prometheus_up", os.getenv("PROMETHEUS_URL", "http://prometheus:9090"), "/-/healthy"),
    ]
    try:
        import httpx as _httpx

        for _key, _base_url, _path in _companion_services:
            try:
                with _httpx.Client(timeout=2.0) as _c:
                    _r = _c.get(f"{_base_url.rstrip('/')}{_path}")
                    result[_key] = _r.status_code < 500
            except Exception:
                result[_key] = False
    except ImportError:
        pass

    return result


def _render_health_dot(label: str, is_up: bool, title_up: str, title_down: str) -> str:
    """Render a single health indicator dot with label."""
    bg = "#22c55e" if is_up else "#ef4444"
    title = title_up if is_up else title_down
    text_color = "#d4d4d8" if is_up else "#71717a"
    return (
        f'<span style="display:inline-flex;align-items:center;gap:4px;cursor:default" title="{title}">'
        f'<span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:{bg}"></span>'
        f'<span style="font-size:11px;color:{text_color}">{label}</span>'
        f"</span>"
    )


def _render_health_bar(health: dict[str, Any]) -> str:
    """Render health indicators as a compact HTML fragment for the dashboard header."""
    data_dot = _render_health_dot(
        "Data", health.get("data_service_up", True), "Data Service: Running", "Data Service: Down"
    )
    engine_dot = _render_health_dot("Engine", health.get("engine_up", False), "Engine: Running", "Engine: Not running")
    redis_dot = _render_health_dot("Redis", health.get("redis_up", False), "Redis: Connected", "Redis: Disconnected")
    pg_dot = _render_health_dot(
        "Postgres", health.get("postgres_up", False), "Postgres: Connected", "Postgres: Disconnected"
    )
    charting_dot = _render_health_dot("Charts", health.get("charting_up", False), "Charting: Running", "Charting: Down")
    trainer_dot = _render_health_dot(
        "Trainer", health.get("trainer_up", False), "Trainer: Running", "Trainer: Down (optional)"
    )
    grafana_dot = _render_health_dot(
        "Grafana", health.get("grafana_up", False), "Grafana: Running", "Grafana: Down (optional)"
    )
    prom_dot = _render_health_dot(
        "Prom", health.get("prometheus_up", False), "Prometheus: Running", "Prometheus: Down (optional)"
    )

    cnn_on_disk = health.get("cnn_model_on_disk", False)
    if cnn_on_disk:
        cnn_title = "CNN model ready"
        cnn_bg = "rgba(88,28,135,0.5)"
        cnn_border = "rgba(126,34,206,0.6)"
        cnn_color = "#d8b4fe"
        cnn_label = "CNN \u2713"
    else:
        cnn_title = "CNN model not found \u2014 run: bash scripts/sync_models.sh"
        cnn_bg = "rgba(39,39,42,0.8)"
        cnn_border = "#3f3f46"
        cnn_color = "#71717a"
        cnn_label = "CNN \u2013"

    cnn_badge = (
        f'<span style="padding:2px 6px;background:{cnn_bg};border:1px solid {cnn_border};'
        f"border-radius:4px;font-size:10px;color:{cnn_color};font-weight:600;"
        f'letter-spacing:0.025em;cursor:default" title="{cnn_title}">{cnn_label}</span>'
    )

    divider = '<span style="width:1px;height:14px;background:#3f3f46;border-radius:1px"></span>'

    return (
        '<span style="display:inline-flex;align-items:center;gap:8px;flex-wrap:wrap">'
        f"{data_dot}{engine_dot}{redis_dot}{pg_dot}"
        f"{divider}"
        f"{charting_dot}{trainer_dot}{grafana_dot}{prom_dot}"
        f'<span style="margin-left:2px">{cnn_badge}</span>'
        "</span>"
    )


@router.get("/api/health/html", response_class=HTMLResponse)
def get_system_health_html():
    """Return system health indicators as an HTML fragment (polled by HTMX)."""
    return HTMLResponse(content=_render_health_bar(_compute_system_health()))


@router.get("/api/health")
def get_system_health():
    """Return full system health status as JSON."""
    return JSONResponse(content=_compute_system_health())
