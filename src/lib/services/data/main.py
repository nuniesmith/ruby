"""
Ruby Futures — Data Service
============================
Standalone FastAPI microservice that handles all market data, caching,
REST API, and HTMX dashboard rendering.

Architecture:
    Browser → Web (8080) → Data Service (8000) → Redis / Postgres / Massive / Kraken
                                                ↑
                                         Engine (8100) publishes focus/risk to Redis

Responsibilities:
  - REST + SSE API for the dashboard (GET /, /api/focus, /sse/dashboard, …)
  - Bar cache management (Postgres + Redis, gap-fill via Massive / Kraken)
  - Kraken WebSocket live crypto feed
  - Grok AI analyst streaming
  - Trade journal, positions, risk state reads
  - All HTMX fragment endpoints

The Data Service is STATELESS with respect to computation — it reads engine
output from Redis and never runs the DashboardEngine or engine scheduler.
The Engine service (lib.services.engine.main) runs in its own container and
publishes computed focus / risk / ORB state to Redis for the Data Service to
serve.

Usage (from project root):
    PYTHONPATH=src uvicorn lib.services.data.main:app --host 0.0.0.0 --port 8000

Docker:
    ENV PYTHONPATH="/app/src"
    CMD ["uvicorn", "lib.services.data.main:app", "--host", "0.0.0.0", "--port", "8000"]
"""

import asyncio
import json
import math
import os
from contextlib import asynccontextmanager
from typing import Any

# ---------------------------------------------------------------------------
# All imports use fully-qualified `lib.*` paths.
# PYTHONPATH only needs /app/src so that `lib` is discoverable.
# ---------------------------------------------------------------------------
from fastapi import Depends, FastAPI, Request  # noqa: E402
from fastapi.exceptions import RequestValidationError  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import JSONResponse, Response  # noqa: E402
from starlette.exceptions import HTTPException as StarletteHTTPException  # noqa: E402


# ---------------------------------------------------------------------------
# Custom JSON encoder that replaces inf / NaN with null instead of crashing.
# Backtesting and optimization routines can produce inf Sharpe ratios or
# NaN win-rates, which the stdlib json encoder rejects.
# ---------------------------------------------------------------------------
class _SafeFloatEncoder(json.JSONEncoder):
    """JSON encoder that converts inf/-inf/NaN to None."""

    def default(self, o: Any) -> Any:
        return super().default(o)

    def encode(self, o: Any) -> str:
        return super().encode(_sanitize(o))


def _sanitize(obj: Any) -> Any:
    """Recursively replace non-finite floats with None."""
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    return obj


class SafeJSONResponse(JSONResponse):
    """JSONResponse subclass that handles inf/NaN floats gracefully."""

    def render(self, content: Any) -> bytes:
        return json.dumps(
            content,
            cls=_SafeFloatEncoder,
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")


# ---------------------------------------------------------------------------
# Logging — structured via structlog
# ---------------------------------------------------------------------------
from lib.core.logging_config import (  # noqa: E402  # pylint: disable=wrong-import-position
    get_logger,
    setup_logging,
)

setup_logging(service="data-service")
logger = get_logger("data_service")

# ---------------------------------------------------------------------------
# Import routers — these live under src/services/data/api/
# Bare imports like `from cache import ...` resolve via PYTHONPATH (/app/src).
# ---------------------------------------------------------------------------
from lib.core.models import init_db  # noqa: E402
from lib.services.data.api.actions import (  # noqa: E402
    router as actions_router,
)
from lib.services.data.api.actions import (  # noqa: E402
    set_engine as actions_set_engine,
)
from lib.services.data.api.analysis import (  # noqa: E402
    router as analysis_router,
)
from lib.services.data.api.analysis import (  # noqa: E402
    set_engine as analysis_set_engine,
)
from lib.services.data.api.audit import router as audit_router  # noqa: E402
from lib.services.data.api.auth import require_api_key  # noqa: E402
from lib.services.data.api.bars import (  # noqa: E402
    router as bars_router,
)
from lib.services.data.api.bars import (  # noqa: E402
    startup_warm_caches,
)
from lib.services.data.api.charting_proxy import router as charting_proxy_router  # noqa: E402
from lib.services.data.api.chat import router as chat_router  # noqa: E402
from lib.services.data.api.chat import set_engine as chat_set_engine  # noqa: E402
from lib.services.data.api.cnn import router as cnn_router  # noqa: E402
from lib.services.data.api.dashboard import (  # noqa: E402
    router as dashboard_router,
)
from lib.services.data.api.grok import router as grok_router  # noqa: E402
from lib.services.data.api.grok import set_engine as grok_set_engine  # noqa: E402
from lib.services.data.api.health import (  # noqa: E402
    router as health_router,
)
from lib.services.data.api.journal import (  # noqa: E402
    router as journal_router,
)
from lib.services.data.api.kraken import router as kraken_router  # noqa: E402
from lib.services.data.api.live_risk import (  # noqa: E402
    router as live_risk_router,
)
from lib.services.data.api.market_data import (  # noqa: E402
    router as market_data_router,
)
from lib.services.data.api.metrics import PrometheusMiddleware  # noqa: E402
from lib.services.data.api.metrics import (  # noqa: E402
    router as metrics_router,
)
from lib.services.data.api.news import router as news_router  # noqa: E402
from lib.services.data.api.pipeline import (  # noqa: E402
    router as pipeline_router,
)
from lib.services.data.api.positions import (  # noqa: E402
    router as positions_router,
)
from lib.services.data.api.rate_limit import (  # noqa: E402
    setup_rate_limiting,
)
from lib.services.data.api.reddit import router as reddit_router  # noqa: E402
from lib.services.data.api.risk import router as risk_router  # noqa: E402
from lib.services.data.api.settings import (  # noqa: E402
    router as settings_router,
)
from lib.services.data.api.sse import router as sse_router  # noqa: E402
from lib.services.data.api.swing_actions import (  # noqa: E402
    router as swing_actions_router,
)
from lib.services.data.api.tasks import router as tasks_router  # noqa: E402
from lib.services.data.api.trades import (  # noqa: E402
    router as trades_router,
)
from lib.services.data.api.trainer import (  # noqa: E402
    router as trainer_router,
)
from lib.services.data.api.simulation_api import (  # noqa: E402
    router as sim_router,
    sse_router as sim_sse_router,
)
from lib.services.data.sync import (  # noqa: E402
    DataSyncService,
    get_sync_service,
    sync_router,
)

# ---------------------------------------------------------------------------
# Engine mode — always "remote" when running as standalone data service.
# The data service NEVER embeds the engine; it reads published state from Redis.
# ENGINE_MODE=embedded is retained only for local dev / smoke-testing without
# a separate engine container.
# ---------------------------------------------------------------------------
_ENGINE_MODE = os.getenv("ENGINE_MODE", "remote")  # default: remote (separate engine)

_engine: Any = None


class _RemoteEngineProxy:
    """Lightweight proxy that reads engine state from Redis.

    When the engine runs in a separate container, it publishes status,
    backtest results, and strategy history to Redis keys.  This proxy
    reads those keys so the API routers work without modification.
    """

    def __init__(self):
        self.interval = os.getenv("ENGINE_INTERVAL", "5m")
        self.period = os.getenv("ENGINE_PERIOD", "5d")

    def _redis_get_json(self, key: str, default: Any = None) -> Any:
        try:
            from lib.core.cache import cache_get  # noqa: E402

            raw = cache_get(key)
            if raw:
                return json.loads(raw)
        except Exception:
            pass
        return default

    def get_status(self) -> dict[str, Any]:
        return self._redis_get_json(
            "engine:status",
            {
                "engine": "remote",
                "data_refresh": {"last": None, "status": "unknown"},
                "optimization": {"last": None, "status": "unknown"},
                "backtest": {"last": None, "status": "unknown"},
                "live_feed": {"status": "unknown"},
            },
        )

    def get_backtest_results(self) -> list[Any]:
        return self._redis_get_json("engine:backtest_results", [])

    def get_strategy_history(self) -> dict[str, Any]:
        return self._redis_get_json("engine:strategy_history", {})

    def get_live_feed_status(self) -> dict[str, Any]:
        return self._redis_get_json(
            "engine:live_feed_status",
            {
                "status": "unknown",
                "connected": False,
                "data_source": "unknown",
            },
        )

    def force_refresh(self) -> None:
        try:
            from lib.core.cache import flush_all

            flush_all()
        except Exception:
            pass

    def start_live_feed(self) -> bool:
        return False

    async def stop_live_feed(self) -> None:
        pass

    def upgrade_live_feed(self) -> None:
        pass

    def downgrade_live_feed(self) -> None:
        pass

    def update_settings(self, **kwargs) -> None:
        pass

    async def stop(self) -> None:
        pass


def get_current_engine():
    """Return the running engine instance (or remote proxy). Used by health router."""
    global _engine
    if _engine is None:
        raise RuntimeError("Engine not initialised — service is still starting up")
    return _engine


# ---------------------------------------------------------------------------
# Lifespan: start engine + background tasks on startup, clean up on shutdown
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global _engine

    logger.info("=" * 60)
    logger.info("  Data Service starting up (engine_mode=%s)", _ENGINE_MODE)
    logger.info("=" * 60)

    # 1. Initialise the database
    try:
        init_db()
        logger.info(
            "Database initialised (DB_PATH=%s)",
            os.getenv("DB_PATH", "futures_journal.db"),
        )
    except Exception as exc:
        logger.error("Database init failed: %s", exc)

    # 2. Read configuration from environment
    account_size = int(os.getenv("ACCOUNT_SIZE", os.getenv("DEFAULT_ACCOUNT_SIZE", "150000")))
    interval = os.getenv("ENGINE_INTERVAL", os.getenv("DEFAULT_INTERVAL", "5m"))
    period = os.getenv("ENGINE_PERIOD", os.getenv("DEFAULT_PERIOD", "5d"))

    # 3. Start engine (embedded) or connect proxy (remote)
    if _ENGINE_MODE == "remote":
        _engine = _RemoteEngineProxy()
        logger.info("Using remote engine proxy (reads from Redis)")
    else:
        from lib.trading.engine import get_engine

        _engine = get_engine(
            account_size=account_size,
            interval=interval,
            period=period,
        )
        logger.info(
            "Embedded engine started: account=$%s  interval=%s  period=%s",
            f"{account_size:,}",
            interval,
            period,
        )

    app.state.engine = _engine

    # 4. Inject engine into routers that need it
    analysis_set_engine(_engine)
    actions_set_engine(_engine)
    grok_set_engine(_engine)
    chat_set_engine(_engine)

    # 4b. Start Reddit watcher + aggregation job (non-fatal if credentials missing)
    try:
        from lib.analysis.sentiment.reddit_sentiment import get_full_snapshot as _reddit_snapshot
        from lib.integrations.reddit_watcher import RedditWatcher

        _reddit_watcher = RedditWatcher(
            redis=app.state.redis,
            pg_pool=getattr(app.state, "pg_pool", None),
            mode=os.getenv("REDDIT_MODE", "poll"),
        )
        asyncio.create_task(_reddit_watcher.run())

        async def _reddit_aggregation_job():
            while True:
                try:
                    await _reddit_snapshot(app.state.redis)
                except Exception as exc:
                    logger.warning("Reddit aggregation error: %s", exc)
                await asyncio.sleep(300)

        asyncio.create_task(_reddit_aggregation_job())
        logger.info("Reddit watcher + aggregation job started")
    except Exception as exc:
        logger.warning("Reddit watcher startup skipped (non-fatal): %s", exc)

    # 4c. Wire LiveRiskPublisher into the live_risk API module.
    #     In embedded mode the standalone engine loop (main.py) is NOT running,
    #     so we create a LiveRiskPublisher here that the /api/live-risk/refresh
    #     endpoint and the HTMX polling strip can use.
    #     In remote mode there is no local RiskManager/PositionManager — the
    #     live_risk API falls back to reading from Redis (published by the
    #     separate engine container).
    _live_risk_publisher = None
    if _ENGINE_MODE != "remote":
        try:
            from lib.services.data.api.live_risk import set_publisher as lr_set_publisher
            from lib.services.engine.live_risk import LiveRiskPublisher

            # Try to get RiskManager and PositionManager from the engine
            _rm = getattr(_engine, "risk_manager", None) if _engine else None
            _pm = getattr(_engine, "position_manager", None) if _engine else None

            _live_risk_publisher = LiveRiskPublisher(
                risk_manager=_rm,
                position_manager=_pm,
                interval_seconds=5.0,
            )
            lr_set_publisher(_live_risk_publisher)

            # Force an initial publish so the dashboard risk strip has data
            # immediately on first load.
            _live_risk_publisher.force_publish()
            logger.info(
                "LiveRiskPublisher wired into data-service (rm=%s, pm=%s)",
                "yes" if _rm else "no",
                "yes" if _pm else "no",
            )
        except Exception as exc:
            logger.warning("LiveRiskPublisher setup failed (non-fatal): %s", exc)

    # 5. Log data source
    try:
        from lib.core.cache import get_data_source

        ds = get_data_source()
        logger.info("Primary data source: %s", ds)
    except Exception:
        logger.info("Primary data source: yfinance (default)")

    # 5b. Start Kraken WebSocket feed for live crypto data (if enabled)
    _kraken_feed = None
    try:
        from lib.core.models import ENABLE_KRAKEN_CRYPTO

        if ENABLE_KRAKEN_CRYPTO:
            from lib.integrations.kraken_client import start_kraken_feed

            _kraken_feed = start_kraken_feed()
            logger.info(
                "Kraken WebSocket feed started: %d pairs",
                len(_kraken_feed._pairs),
            )
        else:
            logger.info("Kraken crypto disabled (ENABLE_KRAKEN_CRYPTO=0)")
    except ImportError:
        logger.debug("Kraken client not available — crypto feed disabled")
    except Exception as exc:
        logger.warning("Kraken feed startup failed (non-fatal): %s", exc)

    # 6. Warm Redis bar caches from Postgres so the dataset generator and
    #    engine have data available immediately without waiting for a fill.
    try:
        startup_warm_caches(days_back=7)
    except Exception as exc:
        logger.warning("Startup cache warm failed (non-fatal): %s", exc)

    # 7. Start the background DataSyncService — maintains rolling 1-year
    #    window of 1-min bars in Postgres with incremental 5-min refreshes.
    _sync_service: DataSyncService | None = None
    try:
        _sync_service = get_sync_service()
        _sync_task = asyncio.create_task(_sync_service.run())
        _sync_service._task = _sync_task
        logger.info("DataSyncService background task started")
    except Exception as exc:
        logger.warning("DataSyncService startup failed (non-fatal): %s", exc)

    # 8. Start SimulationEngine for paper-trading with live Kraken tick data
    _sim_engine = None
    try:
        sim_enabled = os.getenv("SIM_ENABLED", "0").strip().lower() in ("1", "true", "yes")
        if sim_enabled:
            from lib.services.engine.simulation import create_sim_engine

            _sim_engine = create_sim_engine()
            _sim_engine.start()
            app.state.sim_engine = _sim_engine
            logger.info(
                "SimulationEngine started (balance=$%s, source=%s)",
                f"{_sim_engine._account_balance:,.2f}",
                os.getenv("SIM_DATA_SOURCE", "kraken"),
            )
        else:
            logger.info("SimulationEngine disabled (SIM_ENABLED=0)")
    except Exception as exc:
        logger.warning("SimulationEngine startup failed (non-fatal): %s", exc)

    logger.info("=" * 60)
    logger.info("  Data Service ready — accepting requests")
    logger.info("=" * 60)

    yield

    # --- Shutdown ---
    logger.info("=" * 60)
    logger.info("  Data Service shutting down")
    logger.info("=" * 60)

    # Stop SimulationEngine
    if _sim_engine is not None:
        try:
            _sim_engine.stop()
            logger.info("SimulationEngine stopped cleanly")
        except Exception as exc:
            logger.warning("SimulationEngine stop error (non-fatal): %s", exc)

    # Stop DataSyncService
    if _sync_service is not None:
        try:
            await _sync_service.stop()
            logger.info("DataSyncService stopped cleanly")
        except Exception as exc:
            logger.warning("DataSyncService stop error (non-fatal): %s", exc)

    # Stop Kraken WebSocket feed
    if _kraken_feed is not None:
        try:
            await _kraken_feed.stop()
            logger.info("Kraken feed stopped cleanly")
        except Exception as exc:
            logger.warning("Kraken feed stop error (non-fatal): %s", exc)

    if _engine is not None:
        try:
            await _engine.stop()
            logger.info("Engine stopped cleanly")
        except Exception as exc:
            logger.warning("Engine stop error (non-fatal): %s", exc)

    logger.info("Data Service stopped")


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Ruby Futures Data Service",
    description=(
        "Background data service for Ruby Futures. "
        "Runs the DashboardEngine, Massive WS listener, and all FKS "
        "computation modules. Exposes REST endpoints and an HTMX dashboard."
    ),
    version="1.0.0",
    lifespan=lifespan,
    default_response_class=SafeJSONResponse,
    dependencies=[Depends(require_api_key)],
)


# ---------------------------------------------------------------------------
# Structured error responses — consistent JSON shape for all error types
# ---------------------------------------------------------------------------
# Every error response follows:
#   { "error": "<short_code>", "detail": "<human message>", "status": <int> }
# This replaces FastAPI's default {"detail": ...} shape so clients can
# rely on a single schema for error handling.
# ---------------------------------------------------------------------------


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    """Handle all HTTP exceptions (404, 403, 405, 422, 500, etc.)."""
    status = exc.status_code
    detail = exc.detail if isinstance(exc.detail, str) else str(exc.detail)

    # Map common status codes to short error codes
    code_map = {
        400: "bad_request",
        401: "unauthorized",
        403: "forbidden",
        404: "not_found",
        405: "method_not_allowed",
        409: "conflict",
        422: "validation_error",
        429: "rate_limit_exceeded",
        500: "internal_error",
        502: "bad_gateway",
        503: "service_unavailable",
    }
    error_code = code_map.get(status, f"http_{status}")

    return JSONResponse(
        status_code=status,
        content={
            "error": error_code,
            "detail": detail,
            "status": status,
        },
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Handle Pydantic / query-param validation errors with structured JSON."""
    errors = exc.errors()
    # Build a human-readable summary from the validation error list
    messages = []
    for err in errors:
        loc = " → ".join(str(part) for part in err.get("loc", []))
        msg = err.get("msg", "invalid")
        messages.append(f"{loc}: {msg}" if loc else msg)

    return JSONResponse(
        status_code=422,
        content={
            "error": "validation_error",
            "detail": "; ".join(messages) if messages else "Request validation failed",
            "status": 422,
            "errors": errors,
        },
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch-all for unhandled exceptions — log and return structured 500."""
    logger.error(
        "Unhandled exception on %s %s: %s",
        request.method,
        request.url.path,
        exc,
        exc_info=True,
    )
    return JSONResponse(
        status_code=500,
        content={
            "error": "internal_error",
            "detail": f"Internal server error: {type(exc).__name__}",
            "status": 500,
        },
    )


# CORS — allow local dev origins + Tailscale IPs
# NOTE: allow_credentials must be False when using wildcard origins.
# The SSE StreamingResponse also must NOT set Access-Control-Allow-Origin
# manually — let this middleware handle it consistently to avoid the
# "wildcard + credentials" conflict that makes browsers drop EventSource.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Prometheus metrics middleware — records request count + latency (TASK-704)
app.add_middleware(PrometheusMiddleware)

# Rate limiting (TASK-703) — slowapi-based per-client rate limits
setup_rate_limiting(app)

# ---------------------------------------------------------------------------
# Register routers
# ---------------------------------------------------------------------------
# Dashboard: / (HTML page), /api/focus, /api/focus/html, /api/time, etc.
# NOTE: dashboard_router is mounted WITHOUT a prefix so GET / serves the HTML
# dashboard and /api/focus, /api/focus/html etc. are top-level paths.
app.include_router(dashboard_router, tags=["Dashboard"])

# Charting proxy: /charting-proxy/* → charting container (reverse proxy)
# Allows the Charts iframe to work from any machine by proxying through the
# data service instead of requiring direct access to charting:8003.
app.include_router(charting_proxy_router, tags=["Charting Proxy"])

# SSE: /sse/dashboard (live event stream), /sse/health
# NOTE: sse_router is mounted WITHOUT a prefix so /sse/dashboard is top-level.
app.include_router(sse_router, tags=["SSE"])

# Analysis: /analysis/latest, /analysis/latest/{ticker}, /analysis/status, etc.
app.include_router(analysis_router, prefix="/analysis", tags=["Analysis"])

# Actions: /actions/force_refresh, /actions/optimize_now, /actions/update_settings, etc.
app.include_router(actions_router, prefix="/actions", tags=["Actions"])

# Positions: /positions/, /positions/update, etc.  (broker-agnostic position management)
app.include_router(positions_router, prefix="/positions", tags=["Positions"])

# Trades: /trades, /trades/{id}/close, /log_trade, etc.  (trade CRUD)
app.include_router(trades_router, prefix="", tags=["Trades"])

# Risk: /risk/status, /risk/check, /risk/history  (risk engine API)
app.include_router(risk_router, prefix="/risk", tags=["Risk"])

# Audit: /audit/risk, /audit/orb, /audit/summary  (persistent event history)
app.include_router(audit_router, prefix="/audit", tags=["Audit"])

# Journal: /journal/save, /journal/entries, /journal/stats, /journal/today
app.include_router(journal_router, prefix="/journal", tags=["Journal"])

# Market Data: /data/ohlcv, /data/daily, /data/source  (OHLCV proxy for thin client)
app.include_router(market_data_router, prefix="/data", tags=["Market Data"])

# Bars: /bars/{symbol}, /bars/bulk, /bars/status, /bars/assets
#       /bars/{symbol}/gaps, /bars/{symbol}/fill, /bars/fill/all, /bars/fill/status
# Historical bar store — serves data from Postgres with auto gap-fill from Massive.
# Primary data source for CNN dataset generation, engine backtesting, and web charts.
app.include_router(bars_router, tags=["Bars"])

# Health: /health, /metrics  (no prefix — top-level)
app.include_router(health_router, tags=["Health"])

# Prometheus metrics: /metrics/prometheus  (TASK-704)
app.include_router(metrics_router, tags=["Metrics"])

# System Health: /api/health, /api/health/html
app.include_router(health_router, tags=["System Health"])

# CNN: /cnn/status, /cnn/retrain, /cnn/retrain/status, /cnn/history, /cnn/status/html
# NOTE: cnn_router is mounted WITHOUT a prefix — routes are defined with /cnn/ paths.
app.include_router(cnn_router, tags=["CNN"])

# Kraken: /kraken/health, /kraken/status, /kraken/pairs, /kraken/ticker/{pair},
#          /kraken/tickers, /kraken/ohlcv/{pair}, /kraken/health/html
# NOTE: kraken_router is mounted WITHOUT a prefix — routes are defined with /kraken/ paths.
app.include_router(kraken_router, tags=["Kraken"])

# Grok: /api/grok/latest, /api/grok/briefing, /api/grok/trigger/briefing,
#        /api/grok/trigger/update, /sse/grok/briefing, /sse/grok/update
# NOTE: grok_router is mounted WITHOUT a prefix — routes are defined with full paths.
app.include_router(grok_router, tags=["Grok"])


# Trainer: /trainer (HTML page), /trainer/api/* (proxy to trainer service),
#          /trainer/config, /trainer/service_status
# NOTE: trainer_router is mounted WITHOUT a prefix — routes are defined with /trainer/ paths.
app.include_router(trainer_router, tags=["Trainer"])

# Settings: /settings (HTML page)
# NOTE: settings_router is mounted WITHOUT a prefix — route is defined with /settings path.
app.include_router(settings_router, tags=["Settings"])

# Rithmic: /api/rithmic/accounts, /api/rithmic/status, /api/rithmic/status/html,
#          /api/rithmic/account/{key}, /api/rithmic/account/{key}/refresh,
#          /api/rithmic/account/{key}/config, /api/rithmic/account/{key}/remove,
#          /api/rithmic/refresh-all, /api/rithmic/config/new-key, /api/rithmic/deps,
#          /settings/rithmic/panel
# NOTE: rithmic_router is mounted WITHOUT a prefix — routes are defined with full paths.
from lib.integrations.rithmic_client import router as rithmic_router  # noqa: E402

app.include_router(rithmic_router, tags=["Rithmic"])


# Live Risk: /api/live-risk, /api/live-risk/html, /api/live-risk/summary,
#            /api/live-risk/refresh, /api/live-risk/position/{asset_name}/html
# Phase 5B/5E: Real-time risk budget state and persistent risk dashboard strip.
# NOTE: live_risk_router is mounted WITHOUT a prefix — routes are defined with /api/live-risk/ paths.
app.include_router(live_risk_router, tags=["Live Risk"])

# Swing Actions: /api/swing/accept/{asset}, /api/swing/ignore/{asset},
#                /api/swing/close/{asset}, /api/swing/stop-to-be/{asset},
#                /api/swing/update-stop/{asset}, /api/swing/pending,
#                /api/swing/active, /api/swing/detail/{asset},
#                /api/swing/history, /api/swing/status-badge/{asset}
# Phase 3D: HTMX mutation + query endpoints for swing trade management.
# NOTE: swing_actions_router is mounted WITHOUT a prefix — routes are defined with /api/swing/ paths.
app.include_router(swing_actions_router, tags=["Swing Actions"])

# Reddit Sentiment: /api/reddit/signal/{asset}, /api/reddit/signal/{asset}/{window_min},
#                   /api/reddit/snapshot, /htmx/reddit/panel, /htmx/reddit/asset/{asset}
# NOTE: reddit_router is mounted WITHOUT a prefix — routes are defined with full paths.
app.include_router(reddit_router, tags=["Reddit Sentiment"])

# Data Sync: /api/data/sync/status, /api/data/sync/trigger, /api/data/bars
# Rolling 1-year data window maintenance with background sync.
# NOTE: sync_router is mounted WITHOUT a prefix — routes are defined with full paths.
app.include_router(sync_router, tags=["Data Sync"])

# Simulation: /api/sim/status, /api/sim/order, /api/sim/close/{symbol},
#             /api/sim/close-all, /api/sim/reset, /api/sim/trades, /api/sim/pnl,
#             /sse/sim — paper trading with live Kraken tick data.
# Gated by SIM_ENABLED=1 env var at the engine level; routes always registered
# but return 503 when the engine is not running.
app.include_router(sim_router, tags=["Simulation"])
app.include_router(sim_sse_router, tags=["Simulation SSE"])

# News Sentiment: /api/news/sentiment, /api/news/sentiment/{symbol},
#                 /api/news/headlines, /api/news/spike,
#                 /htmx/news/panel, /htmx/news/asset/{symbol}
# NOTE: news_router is mounted WITHOUT a prefix — routes are defined with full paths.
app.include_router(news_router, tags=["News Sentiment"])

# Pipeline: /api/pipeline/run, /api/pipeline/status, /api/pipeline/reset,
#           /api/plan, /api/plan/confirm, /api/plan/unlock,
#           /api/live/stream, /api/market/candles, /api/market/cvd,
#           /api/journal/trades, /api/trading/settings, /trading (HTML page)
# Morning workflow pipeline — SSE-driven analysis, plan management, live trading.
# NOTE: pipeline_router is mounted WITHOUT a prefix — routes are defined with full paths.
app.include_router(pipeline_router, tags=["Pipeline"])

# Trade Executor: /api/trade/engage, /api/trade/active, /api/trade/active/{symbol},
#                 /api/trade/partial, /api/trade/close/{symbol}, /api/trade/set-stop,
#                 /api/trade/status, /api/trade/history
# Staged trade execution with stop-hunt protection and plan-aware management.
# NOTE: trade_executor_router is mounted WITHOUT a prefix — routes are defined with full paths.
from lib.services.data.api.trade_executor_routes import router as trade_executor_router  # noqa: E402

app.include_router(trade_executor_router, tags=["Trade Executor"])

# Session Reports: /api/reports/pre-session, /api/reports/post-session,
#                  /api/reports/history, /api/reports/pre-session/notes,
#                  /api/reports/post-session/notes
# Daily pre/post session reports for performance tracking and system improvement.
# NOTE: session_report_router is mounted WITHOUT a prefix — routes are defined with full paths.
from lib.services.data.api.session_report_routes import router as session_report_router  # noqa: E402

app.include_router(session_report_router, tags=["Session Reports"])

# Chat: /api/chat, /sse/chat, /api/chat/history, /api/chat/status
# RustAssistant-powered multi-turn chat with RA→Grok fallback.
# History stored in Redis per session_id. Market context auto-injected.
# NOTE: chat_router is mounted WITHOUT a prefix — routes use /api/chat/ and /sse/chat paths.
app.include_router(chat_router, tags=["Chat"])

# Tasks: /api/tasks, /api/tasks/{id}, /api/tasks/{id}/github, /api/tasks/html
# Lightweight issue/bug/note capture with RustAssistant GitHub integration.
# Stored in SQLite/Postgres tasks table. Pushes to GitHub via RA when configured.
# NOTE: tasks_router is mounted WITHOUT a prefix — routes use /api/tasks/ paths.
app.include_router(tasks_router, tags=["Tasks"])

# Copy Trade: /api/copy-trade/send, /api/copy-trade/send-from-ticker,
#             /api/copy-trade/status, /api/copy-trade/history,
#             /api/copy-trade/compliance-log, /api/copy-trade/rate,
#             /api/copy-trade/high-impact, /api/copy-trade/invalidate-cache,
#             /api/copy-trade/result/{batch_id},
#             /api/copy-trade/status/html, /api/copy-trade/history/html
# RITHMIC-F: WebUI "SEND ALL" button + prop-firm compliant copy trading.
# NOTE: copy_trade_router is mounted WITHOUT a prefix — routes use /api/copy-trade/ paths.
from lib.services.data.api.copy_trade import router as copy_trade_router  # noqa: E402
from lib.services.data.api.pine import router as pine_router  # noqa: E402
from lib.services.data.api.ruby import router as ruby_router  # noqa: E402

app.include_router(copy_trade_router, tags=["Copy Trade"])
app.include_router(ruby_router, tags=["Ruby Signal Engine"])

# Pine Script Generator: /pine (HTML page), /api/pine/modules, /api/pine/module/{name},
#     /api/pine/params, /api/pine/generate, /api/pine/output, /api/pine/download/{name},
#     /api/pine/stats, /api/pine/status/html
# NOTE: pine_router is mounted WITHOUT a prefix — routes are defined with full paths.
app.include_router(pine_router, tags=["Pine Script Generator"])

# DOM (Depth of Market): /api/dom/snapshot, /api/dom/config, /sse/dom
# NOTE: dom routers are mounted WITHOUT a prefix — routes are defined with full paths.
from lib.services.data.api.dom import router as dom_router  # noqa: E402
from lib.services.data.api.dom import sse_router as dom_sse_router  # noqa: E402

app.include_router(dom_router, tags=["DOM"])
app.include_router(dom_sse_router, tags=["DOM SSE"])

# Data Source Router: /api/sources/symbols, /api/sources/status
# NOTE: source_api_router is mounted WITHOUT a prefix — routes are defined with /api/sources/ paths.
from lib.services.data.source_router import source_api_router  # noqa: E402

app.include_router(source_api_router, tags=["Data Sources"])

# Static Pages: /chat, /dom, /journal, /pretrade — serve standalone HTML from static/
from lib.services.data.api.static_pages import router as static_pages_router  # noqa: E402

app.include_router(static_pages_router, tags=["Static Pages"])

# Pre-Trade Analysis: /api/pretrade/assets, /api/pretrade/analyze,
#                     /api/pretrade/analysis/{symbol}, /api/pretrade/select,
#                     /api/pretrade/selected, /api/pretrade/watchlist
# KRAKEN-SIM-C: Asset selection & opportunity scoring for pre-session workflow.
# NOTE: pretrade_router is mounted WITHOUT a prefix — routes are defined with /api/pretrade/ paths.
from lib.services.data.api.pretrade import router as pretrade_router  # noqa: E402

app.include_router(pretrade_router, tags=["Pre-Trade Analysis"])


# ---------------------------------------------------------------------------
# Root endpoint — now served by dashboard_router (GET / returns HTML dashboard)
# The old JSON root is moved to /api/info for programmatic consumers.
# ---------------------------------------------------------------------------
@app.get("/api/info")
def api_info():
    """Service info and links to docs (formerly GET /)."""
    return {
        "service": "futures-data-service",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "health": "/health",
        "endpoints": {
            "dashboard": "/",
            "focus_json": "/api/focus",
            "focus_html": "/api/focus/html",
            "sse_dashboard": "/sse/dashboard",
            "sse_health": "/sse/health",
            "analysis": "/analysis/latest",
            "status": "/analysis/status",
            "force_refresh": "/actions/force_refresh",
            "positions": "/positions/",
            "trades": "/trades",
            "journal": "/journal/entries",
            "market_data": "/data/ohlcv",
            "daily_data": "/data/daily",
            "data_source": "/data/source",
            "health": "/health",
            "metrics": "/metrics",
            "system_health": "/api/health",
            "system_health_html": "/api/health/html",
            "cnn_status": "/cnn/status",
            "cnn_status_html": "/cnn/status/html",
            "cnn_retrain": "/cnn/retrain",
            "cnn_retrain_status": "/cnn/retrain/status",
            "cnn_retrain_log": "/cnn/retrain/log",
            "cnn_retrain_cancel": "/cnn/retrain/cancel",
            "cnn_history": "/cnn/history",
            "cnn_sync": "/cnn/sync",
            "cnn_sync_status": "/cnn/sync/status",
            "cnn_watcher_status": "/cnn/watcher/status",
            "kraken_health": "/kraken/health",
            "kraken_status": "/kraken/status",
            "kraken_pairs": "/kraken/pairs",
            "kraken_ticker": "/kraken/ticker/{pair}",
            "kraken_tickers": "/kraken/tickers",
            "kraken_ohlcv": "/kraken/ohlcv/{pair}",
            "kraken_health_html": "/kraken/health/html",
            "trainer_page": "/trainer",
            "trainer_config": "/trainer/config",
            "trainer_service_status": "/trainer/service_status",
            "trainer_api": "/trainer/api/{path}",
            "live_risk": "/api/live-risk",
            "live_risk_html": "/api/live-risk/html",
            "live_risk_summary": "/api/live-risk/summary",
            "bars": "/bars/{symbol}",
            "bars_bulk": "/bars/bulk",
            "bars_status": "/bars/status",
            "bars_assets": "/bars/assets",
            "bars_gaps": "/bars/{symbol}/gaps",
            "bars_fill": "/bars/{symbol}/fill",
            "bars_fill_all": "/bars/fill/all",
            "bars_fill_status": "/bars/fill/status",
            "pine": "/pine",
            "pine_api": "/api/pine/modules",
        },
    }


# ---------------------------------------------------------------------------
# Favicon — return 204 No Content to suppress browser 404 errors.
# The HTML dashboard uses an inline SVG data-URI favicon in <link rel="icon">,
# but browsers still request /favicon.ico automatically on first load.
# ---------------------------------------------------------------------------
@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return Response(status_code=204)


# ---------------------------------------------------------------------------
# Run directly: python -m lib.services.data.main
# or via entrypoint: python -m entrypoints.data.main
# ---------------------------------------------------------------------------
def main() -> None:
    """Start the Data Service via uvicorn. Called by entrypoints/data/main.py."""
    import uvicorn

    host = os.getenv("DATA_SERVICE_HOST", "0.0.0.0")
    port = int(os.getenv("DATA_SERVICE_PORT", "8000"))
    log_level = os.getenv("LOG_LEVEL", "info")

    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level=log_level,
    )


if __name__ == "__main__":
    main()
