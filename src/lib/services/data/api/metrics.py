"""
Prometheus Metrics Endpoint — TASK-704
========================================
Exposes ``GET /metrics/prometheus`` returning Prometheus text-format metrics.

Tracked metrics:
  - ``http_requests_total``           — Counter: total HTTP requests by method, path, status
  - ``http_request_duration_seconds`` — Histogram: request latency by method and path
  - ``sse_connections_active``        — Gauge: currently active SSE connections
  - ``sse_events_total``              — Counter: total SSE events emitted by event type
  - ``engine_last_refresh_epoch``     — Gauge: epoch timestamp of last engine data refresh
  - ``engine_cycle_duration_seconds`` — Histogram: engine cycle duration
  - ``risk_checks_total``             — Counter: risk checks by result (allowed/blocked/advisory)
  - ``orb_detections_total``          — Counter: ORB breakout detections by direction (LONG/SHORT/none)
  - ``orb_filter_results_total``      — Counter: ORB filter gate outcomes by result (passed/rejected/error)
  - ``orb_cnn_prob``                  — Histogram: CNN P(good) probability per scored breakout
  - ``orb_cnn_signals_total``         — Counter: CNN signal outcomes by verdict (signal/no_signal/skipped)
  - ``no_trade_alerts_total``         — Counter: no-trade alerts by condition
  - ``focus_quality_gauge``           — Gauge: latest focus quality per asset symbol
  - ``positions_open_count``          — Gauge: number of currently open positions
  - ``daily_pnl_gauge``               — Gauge: current day's realised P&L in dollars
  - ``consecutive_losses_gauge``      — Gauge: number of consecutive losing trades
  - ``model_val_accuracy``            — Gauge: validation accuracy of the current champion CNN model (0–100)
  - ``model_val_precision``           — Gauge: validation precision of the current champion CNN model (0–100)
  - ``model_val_recall``              — Gauge: validation recall of the current champion CNN model (0–100)
  - ``model_train_samples``           — Gauge: number of training samples used by the current champion model
  - ``model_stale``                   — Gauge: 1 if the champion model has not been retrained in > 26 hours, 0 otherwise
  - ``model_last_retrain_epoch``      — Gauge: Unix epoch timestamp of the last successful model promotion
  - ``redis_connected``               — Gauge: 1 if Redis is connected, 0 otherwise
  - ``postgres_connected``            — Gauge: 1 if Postgres is connected, 0 otherwise
  - ``engine_up``                     — Gauge: 1 if engine is running and recently refreshed, 0 otherwise
  - ``kraken_ws_connected``           — Gauge: 1 if Kraken WebSocket feed is connected, 0 otherwise
  - ``kraken_ws_reconnect_total``     — Counter: total Kraken WebSocket reconnect attempts
  - ``kraken_ws_bars_total``          — Gauge: total OHLC bars received from Kraken WS since start
  - ``kraken_ws_errors_total``        — Gauge: total error count on the Kraken WS feed
  - ``regime_state``                  — Gauge: current HMM regime per symbol (1=trending, 2=volatile, 3=choppy)
  - ``regime_confidence``             — Gauge: regime detection confidence (0.0–1.0) per symbol
  - ``regime_position_multiplier``    — Gauge: position sizing multiplier from regime (0.25–1.0) per symbol
  - ``trainer_images_generated``      — Gauge: total chart images generated in the most recent dataset build
  - ``trainer_label_balance``         — Gauge: image count per label (``label`` = "good" / "bad") in most recent build
  - ``trainer_render_time_seconds``   — Histogram: wall-clock seconds spent rendering the full dataset

All metrics are collected in-process via ``prometheus_client`` and the ASGI
middleware automatically instruments request count + latency.

Usage:
    from lib.services.data.api.metrics import router as metrics_router, PrometheusMiddleware
    app.include_router(metrics_router)
    app.add_middleware(PrometheusMiddleware)

The ``/metrics/prometheus`` path is public (no API key required).
"""

import logging
import time

from fastapi import APIRouter, Request, Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import Response as StarletteResponse

logger = logging.getLogger("api.metrics")

# ---------------------------------------------------------------------------
# Prometheus client setup
# ---------------------------------------------------------------------------
# We use a custom CollectorRegistry so tests can create isolated instances
# without polluting the global default.
# ---------------------------------------------------------------------------
from prometheus_client import (  # noqa: E402
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

# Singleton registry for the application
_registry = CollectorRegistry()

# ---------------------------------------------------------------------------
# Metric definitions
# ---------------------------------------------------------------------------

# -- HTTP request metrics (populated by middleware) --
HTTP_REQUESTS_TOTAL = Counter(
    "http_requests_total",
    "Total HTTP requests received",
    labelnames=["method", "path", "status"],
    registry=_registry,
)

HTTP_REQUEST_DURATION = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency in seconds",
    labelnames=["method", "path"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
    registry=_registry,
)

# -- SSE metrics --
SSE_CONNECTIONS_ACTIVE = Gauge(
    "sse_connections_active",
    "Number of currently active SSE connections",
    registry=_registry,
)

SSE_EVENTS_TOTAL = Counter(
    "sse_events_total",
    "Total SSE events emitted",
    labelnames=["event_type"],
    registry=_registry,
)

# -- Engine metrics --
ENGINE_LAST_REFRESH_EPOCH = Gauge(
    "engine_last_refresh_epoch",
    "Unix epoch timestamp of the last engine data refresh",
    registry=_registry,
)

ENGINE_CYCLE_DURATION = Histogram(
    "engine_cycle_duration_seconds",
    "Duration of engine scheduler cycles",
    buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0),
    registry=_registry,
)

# -- Risk metrics --
RISK_CHECKS_TOTAL = Counter(
    "risk_checks_total",
    "Total risk checks performed",
    labelnames=["result"],  # allowed, blocked, advisory
    registry=_registry,
)

# -- ORB metrics --
ORB_DETECTIONS_TOTAL = Counter(
    "orb_detections_total",
    "Opening Range Breakout detections",
    labelnames=["direction"],  # LONG, SHORT, none
    registry=_registry,
)

ORB_FILTER_RESULTS_TOTAL = Counter(
    "orb_filter_results_total",
    "ORB filter gate outcomes",
    labelnames=["result"],  # passed, rejected, error
    registry=_registry,
)

ORB_CNN_PROB = Histogram(
    "orb_cnn_prob",
    "CNN P(good) probability score for each scored ORB breakout",
    buckets=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.80, 0.82, 0.85, 0.90, 0.95, 1.0),
    registry=_registry,
)

ORB_CNN_SIGNALS_TOTAL = Counter(
    "orb_cnn_signals_total",
    "CNN signal verdict for scored ORB breakouts",
    labelnames=["verdict"],  # signal, no_signal, skipped
    registry=_registry,
)

# -- No-trade metrics --
NO_TRADE_ALERTS_TOTAL = Counter(
    "no_trade_alerts_total",
    "No-trade alerts emitted",
    labelnames=["condition"],
    registry=_registry,
)

# -- Focus quality per asset --
FOCUS_QUALITY_GAUGE = Gauge(
    "focus_quality_gauge",
    "Latest focus quality score per asset",
    labelnames=["symbol"],
    registry=_registry,
)

# -- Positions --
POSITIONS_OPEN_COUNT = Gauge(
    "positions_open_count",
    "Number of currently open positions",
    registry=_registry,
)

# -- Risk / P&L --
DAILY_PNL_GAUGE = Gauge(
    "daily_pnl_gauge",
    "Current trading day realised P&L in dollars",
    registry=_registry,
)

CONSECUTIVE_LOSSES_GAUGE = Gauge(
    "consecutive_losses_gauge",
    "Number of consecutive losing trades",
    registry=_registry,
)

# -- CNN model performance --
MODEL_VAL_ACCURACY = Gauge(
    "model_val_accuracy",
    "Validation accuracy of the current champion CNN model (0–100)",
    registry=_registry,
)

MODEL_VAL_PRECISION = Gauge(
    "model_val_precision",
    "Validation precision of the current champion CNN model (0–100)",
    registry=_registry,
)

MODEL_VAL_RECALL = Gauge(
    "model_val_recall",
    "Validation recall of the current champion CNN model (0–100)",
    registry=_registry,
)

MODEL_TRAIN_SAMPLES = Gauge(
    "model_train_samples",
    "Number of training samples used by the current champion CNN model",
    registry=_registry,
)

MODEL_STALE = Gauge(
    "model_stale",
    "1 if the champion model has not been retrained in > 26 hours, 0 otherwise",
    registry=_registry,
)

MODEL_LAST_RETRAIN_EPOCH = Gauge(
    "model_last_retrain_epoch",
    "Unix epoch timestamp of the last successful CNN model promotion",
    registry=_registry,
)

# -- Redis connectivity --
REDIS_CONNECTED = Gauge(
    "redis_connected",
    "Whether Redis is currently connected (1=yes, 0=no)",
    registry=_registry,
)

# -- Postgres connectivity --
POSTGRES_CONNECTED = Gauge(
    "postgres_connected",
    "Whether Postgres is currently connected (1=yes, 0=no)",
    registry=_registry,
)

# -- Engine liveness --
ENGINE_UP = Gauge(
    "engine_up",
    "Whether the engine is running and recently refreshed (1=yes, 0=no)",
    registry=_registry,
)


# -- Kraken WebSocket health --
KRAKEN_WS_CONNECTED = Gauge(
    "kraken_ws_connected",
    "Whether the Kraken WebSocket feed is currently connected (1=yes, 0=no)",
    registry=_registry,
)

KRAKEN_WS_RECONNECT_TOTAL = Counter(
    "kraken_ws_reconnect_total",
    "Total number of Kraken WebSocket reconnect attempts since service start",
    registry=_registry,
)

KRAKEN_WS_BARS_TOTAL = Gauge(
    "kraken_ws_bars_total",
    "Total OHLC bars received from Kraken WebSocket since feed start",
    registry=_registry,
)

KRAKEN_WS_ERRORS_TOTAL = Gauge(
    "kraken_ws_errors_total",
    "Total error count on the Kraken WebSocket feed since feed start",
    registry=_registry,
)

# -- Regime detection --
# Encoded as an integer for Grafana state-timeline panels:
#   1 = trending  (full size)
#   2 = volatile  (half size)
#   3 = choppy    (quarter size)
REGIME_STATE = Gauge(
    "regime_state",
    "Current HMM regime state per symbol (1=trending, 2=volatile, 3=choppy)",
    labelnames=["symbol"],
    registry=_registry,
)

REGIME_CONFIDENCE = Gauge(
    "regime_confidence",
    "HMM regime detection confidence (0.0–1.0) per symbol",
    labelnames=["symbol"],
    registry=_registry,
)

REGIME_POSITION_MULTIPLIER = Gauge(
    "regime_position_multiplier",
    "Position sizing multiplier derived from HMM regime (0.25–1.0) per symbol",
    labelnames=["symbol"],
    registry=_registry,
)

_REGIME_STATE_CODES: dict[str, int] = {"trending": 1, "volatile": 2, "choppy": 3}

# -- Trainer dataset metrics --
# Updated at the end of each dataset generation run (trainer_server.py).
# All three are reset/overwritten on each run so Grafana always shows the
# most-recent build rather than a cumulative total.

TRAINER_IMAGES_GENERATED = Gauge(
    "trainer_images_generated",
    "Total chart images generated in the most recent dataset build",
    registry=_registry,
)

TRAINER_LABEL_BALANCE = Gauge(
    "trainer_label_balance",
    "Image count per label class (good/bad) in the most recent dataset build",
    labelnames=["label"],
    registry=_registry,
)

TRAINER_RENDER_TIME_SECONDS = Histogram(
    "trainer_render_time_seconds",
    "Wall-clock seconds spent rendering the full dataset in a training run",
    buckets=(10.0, 30.0, 60.0, 120.0, 300.0, 600.0, 1200.0, 1800.0, 3600.0),
    registry=_registry,
)

# Mutable single-element list — converts absolute reconnect_count to Counter increments.
_kraken_reconnect_last_seen: list[int] = [0]


# ---------------------------------------------------------------------------
# Helpers for recording metrics from other modules
# ---------------------------------------------------------------------------


def record_sse_connect() -> None:
    """Call when a new SSE client connects."""
    SSE_CONNECTIONS_ACTIVE.inc()


def record_sse_disconnect() -> None:
    """Call when an SSE client disconnects."""
    SSE_CONNECTIONS_ACTIVE.dec()


def record_sse_event(event_type: str) -> None:
    """Record an SSE event emission."""
    SSE_EVENTS_TOTAL.labels(event_type=event_type).inc()


def record_risk_check(result: str) -> None:
    """Record a risk check result ('allowed', 'blocked', or 'advisory')."""
    RISK_CHECKS_TOTAL.labels(result=result).inc()


def record_orb_detection(direction: str) -> None:
    """Record an ORB detection ('LONG', 'SHORT', or 'none')."""
    ORB_DETECTIONS_TOTAL.labels(direction=direction).inc()


def record_orb_filter_result(result: str) -> None:
    """Record an ORB filter gate outcome ('passed', 'rejected', or 'error')."""
    ORB_FILTER_RESULTS_TOTAL.labels(result=result).inc()


def record_orb_cnn_prob(prob: float) -> None:
    """Record a CNN P(good) probability score (0.0–1.0) for a breakout."""
    ORB_CNN_PROB.observe(float(prob))


def record_orb_cnn_signal(verdict: str) -> None:
    """Record a CNN signal verdict ('signal', 'no_signal', or 'skipped')."""
    ORB_CNN_SIGNALS_TOTAL.labels(verdict=verdict).inc()


def update_daily_pnl(pnl: float) -> None:
    """Update the daily P&L gauge."""
    DAILY_PNL_GAUGE.set(float(pnl))


def update_consecutive_losses(count: int) -> None:
    """Update the consecutive losses gauge."""
    CONSECUTIVE_LOSSES_GAUGE.set(int(count))


def update_model_metrics(val_accuracy: float, val_precision: float, val_recall: float, train_samples: int) -> None:
    """Update CNN model performance gauges from the champion model metadata."""
    MODEL_VAL_ACCURACY.set(float(val_accuracy))
    MODEL_VAL_PRECISION.set(float(val_precision))
    MODEL_VAL_RECALL.set(float(val_recall))
    MODEL_TRAIN_SAMPLES.set(int(train_samples))


def update_model_stale(is_stale: bool) -> None:
    """Update the model staleness gauge (1 = stale, 0 = fresh)."""
    MODEL_STALE.set(1 if is_stale else 0)


def update_model_last_retrain(epoch: float) -> None:
    """Update the last retrain epoch timestamp gauge."""
    MODEL_LAST_RETRAIN_EPOCH.set(epoch)


def record_no_trade_alert(condition: str) -> None:
    """Record a no-trade alert by condition name."""
    NO_TRADE_ALERTS_TOTAL.labels(condition=condition).inc()


def record_engine_refresh() -> None:
    """Record that the engine just refreshed data (set timestamp to now)."""
    ENGINE_LAST_REFRESH_EPOCH.set(time.time())


def record_engine_cycle(duration_seconds: float) -> None:
    """Record the duration of an engine scheduler cycle."""
    ENGINE_CYCLE_DURATION.observe(duration_seconds)


def update_focus_quality(symbol: str, quality: float) -> None:
    """Update the focus quality gauge for a specific asset."""
    FOCUS_QUALITY_GAUGE.labels(symbol=symbol).set(quality)


def update_positions_count(count: int) -> None:
    """Update the open positions count gauge."""
    POSITIONS_OPEN_COUNT.set(count)


def update_kraken_ws_status(
    connected: bool,
    reconnect_count: int = 0,
    bar_count: int = 0,
    error_count: int = 0,
) -> None:
    """Update all Kraken WebSocket health gauges/counters.

    Args:
        connected:       True if the feed is currently connected.
        reconnect_count: Cumulative reconnect attempts (monotonically increasing).
        bar_count:       Total OHLC bars received since feed start.
        error_count:     Total errors logged on the feed since start.
    """
    KRAKEN_WS_CONNECTED.set(1 if connected else 0)
    KRAKEN_WS_BARS_TOTAL.set(bar_count)
    KRAKEN_WS_ERRORS_TOTAL.set(error_count)

    # Counter can only go up — only increment by the delta since last call.
    # We store the last-seen value in a module-level variable and inc by diff.
    _delta = max(0, reconnect_count - _kraken_reconnect_last_seen[0])
    if _delta > 0:
        KRAKEN_WS_RECONNECT_TOTAL.inc(_delta)
    _kraken_reconnect_last_seen[0] = reconnect_count


def update_regime(symbol: str, regime: str, confidence: float, position_multiplier: float) -> None:
    """Update regime detection gauges for a single symbol.

    Args:
        symbol:              Asset ticker / name (used as Prometheus label).
        regime:              One of "trending", "volatile", "choppy".
        confidence:          Probability of the detected regime (0.0–1.0).
        position_multiplier: Sizing multiplier (0.25–1.0).
    """
    state_code = _REGIME_STATE_CODES.get(regime, 3)  # default choppy
    REGIME_STATE.labels(symbol=symbol).set(state_code)
    REGIME_CONFIDENCE.labels(symbol=symbol).set(float(confidence))
    REGIME_POSITION_MULTIPLIER.labels(symbol=symbol).set(float(position_multiplier))


def update_redis_status(connected: bool) -> None:
    """Update the Redis connectivity gauge."""
    REDIS_CONNECTED.set(1 if connected else 0)


def update_postgres_status(connected: bool) -> None:
    """Update the Postgres connectivity gauge."""
    POSTGRES_CONNECTED.set(1 if connected else 0)


def update_engine_up(is_up: bool) -> None:
    """Update the engine liveness gauge."""
    ENGINE_UP.set(1 if is_up else 0)


def record_trainer_dataset_stats(
    total_images: int,
    label_distribution: dict[str, int],
    render_time_seconds: float,
) -> None:
    """Update trainer dataset metrics after a generation run.

    Args:
        total_images:        Total number of chart images produced.
        label_distribution:  Mapping of label name → image count,
                             e.g. ``{"good": 1420, "bad": 580}``.
        render_time_seconds: Total wall-clock time for the render phase.
    """
    TRAINER_IMAGES_GENERATED.set(int(total_images))
    for label, count in label_distribution.items():
        TRAINER_LABEL_BALANCE.labels(label=str(label)).set(int(count))
    TRAINER_RENDER_TIME_SECONDS.observe(float(render_time_seconds))


# ---------------------------------------------------------------------------
# Path normalization for HTTP metrics
# ---------------------------------------------------------------------------

# Paths that should be collapsed to reduce cardinality
_PATH_PREFIXES_TO_NORMALIZE = [
    "/api/focus/",
    "/trades/",
    "/positions/",
    "/journal/",
    "/analysis/latest/",
    "/data/ohlcv/",
    "/data/daily/",
]


def _normalize_path(path: str) -> str:
    """Normalize request paths to reduce metric cardinality.

    For example:
        /api/focus/mgc     → /api/focus/{symbol}
        /trades/123/close  → /trades/{id}/close
        /sse/dashboard     → /sse/dashboard  (unchanged)
    """
    if not path:
        return "/"

    for prefix in _PATH_PREFIXES_TO_NORMALIZE:
        if path.startswith(prefix) and len(path) > len(prefix):
            # Collapse the next path segment to {id}
            rest = path[len(prefix) :]
            slash_pos = rest.find("/")
            if slash_pos == -1:
                return prefix + "{id}"
            else:
                return prefix + "{id}" + rest[slash_pos:]

    return path


# ---------------------------------------------------------------------------
# ASGI Middleware for automatic HTTP metrics
# ---------------------------------------------------------------------------


class PrometheusMiddleware(BaseHTTPMiddleware):
    """Starlette middleware that records HTTP request count and latency.

    Automatically instruments every request. SSE endpoints (streaming
    responses) are tracked for connection start; their duration reflects
    the full connection lifetime.
    """

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> StarletteResponse:
        method = request.method
        path = _normalize_path(request.url.path)

        start = time.perf_counter()
        status_code = "500"

        try:
            response = await call_next(request)
            status_code = str(response.status_code)
        except Exception:
            status_code = "500"
            raise
        finally:
            duration = time.perf_counter() - start
            HTTP_REQUESTS_TOTAL.labels(method=method, path=path, status=status_code).inc()
            HTTP_REQUEST_DURATION.labels(method=method, path=path).observe(duration)

        return response


# ---------------------------------------------------------------------------
# Collect live state from Redis/cache when /metrics/prometheus is hit
# ---------------------------------------------------------------------------


def _collect_live_gauges() -> None:
    """Read current state from Redis/cache and update gauges.

    Called each time the metrics endpoint is scraped so that gauges
    reflect the latest known state.
    """
    # Redis connectivity
    try:
        from lib.core.cache import REDIS_AVAILABLE, _r

        if REDIS_AVAILABLE and _r is not None:
            _r.ping()
            update_redis_status(True)
        else:
            update_redis_status(False)
    except Exception:
        update_redis_status(False)

    # Focus quality from cache
    try:
        import json as _json

        from lib.core.cache import cache_get

        raw = cache_get("engine:daily_focus")
        if raw:
            focus = _json.loads(raw)
            for asset in focus.get("assets", []):
                sym = asset.get("symbol", "")
                quality = asset.get("quality", 0)
                if sym:
                    update_focus_quality(sym, float(quality))
    except Exception:
        pass

    # Open positions count
    try:
        import json as _json

        from lib.core.cache import cache_get

        raw = cache_get("engine:positions")
        if raw:
            positions = _json.loads(raw)
            if isinstance(positions, list):
                update_positions_count(len(positions))
            elif isinstance(positions, dict):
                pos_list = positions.get("positions", [])
                update_positions_count(len(pos_list))
        else:
            update_positions_count(0)
    except Exception:
        pass

    # Engine last refresh
    try:
        import json as _json

        from lib.core.cache import cache_get

        raw = cache_get("engine:status")
        if raw:
            status = _json.loads(raw)
            last_refresh = status.get("last_refresh_epoch")
            if last_refresh:
                ENGINE_LAST_REFRESH_EPOCH.set(float(last_refresh))
    except Exception:
        pass

    # Risk / P&L gauges from RiskManager Redis key
    try:
        import json as _json

        from lib.core.cache import cache_get

        raw = cache_get("engine:risk_status")
        if raw:
            risk = _json.loads(raw)
            DAILY_PNL_GAUGE.set(float(risk.get("daily_pnl", 0.0)))
            CONSECUTIVE_LOSSES_GAUGE.set(int(risk.get("consecutive_losses", 0)))
    except Exception:
        pass

    # Kraken WebSocket feed health
    try:
        import os as _os

        _kraken_enabled = _os.getenv("ENABLE_KRAKEN_CRYPTO", "0").strip() in ("1", "true", "yes")
        if _kraken_enabled:
            from lib.integrations.kraken_client import get_kraken_feed

            feed = get_kraken_feed()
            if feed is not None:
                _status = feed.get_status()
                update_kraken_ws_status(
                    connected=bool(_status.get("connected", False)),
                    reconnect_count=int(_status.get("reconnect_count", 0)),
                    bar_count=int(_status.get("bar_count", 0)),
                    error_count=int(_status.get("error_count", 0)),
                )
            else:
                update_kraken_ws_status(connected=False)
        else:
            KRAKEN_WS_CONNECTED.set(0)
    except Exception:
        pass

    # HMM regime state — read from Redis key engine:regime:{symbol}
    try:
        import json as _json

        from lib.core.cache import cache_get

        # Read the consolidated regime map published by the engine
        raw_regime_map = cache_get("engine:regime_states")
        if raw_regime_map:
            regime_map = _json.loads(raw_regime_map)
            for sym, info in regime_map.items():
                update_regime(
                    symbol=sym,
                    regime=info.get("regime", "choppy"),
                    confidence=float(info.get("confidence", 0.0)),
                    position_multiplier=float(info.get("position_multiplier", 0.25)),
                )
    except Exception:
        pass

    # CNN model performance from meta JSON
    try:
        import json as _json
        from pathlib import Path

        _meta_candidates = [
            Path("/app/models/breakout_cnn_best_meta.json"),
            Path(__file__).resolve().parents[5] / "models" / "breakout_cnn_best_meta.json",
        ]
        for _meta_path in _meta_candidates:
            if _meta_path.exists():
                _meta = _json.loads(_meta_path.read_text())
                MODEL_VAL_ACCURACY.set(float(_meta.get("val_accuracy", 0.0)))
                MODEL_VAL_PRECISION.set(float(_meta.get("precision", 0.0)))
                MODEL_VAL_RECALL.set(float(_meta.get("recall", 0.0)))
                MODEL_TRAIN_SAMPLES.set(int(_meta.get("train_samples", 0)))

                # Model staleness: compare promoted_at to now
                try:
                    from datetime import datetime
                    from zoneinfo import ZoneInfo

                    _EST = ZoneInfo("America/New_York")
                    promoted_str = _meta.get("promoted_at")
                    if promoted_str:
                        promoted_at = datetime.fromisoformat(promoted_str)
                        if promoted_at.tzinfo is None:
                            promoted_at = promoted_at.replace(tzinfo=_EST)
                        MODEL_LAST_RETRAIN_EPOCH.set(promoted_at.timestamp())
                        age_hours = (datetime.now(tz=_EST) - promoted_at).total_seconds() / 3600
                        MODEL_STALE.set(1 if age_hours > 26 else 0)
                    else:
                        MODEL_STALE.set(0)
                except Exception:
                    MODEL_STALE.set(0)

                break
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

router = APIRouter(tags=["Metrics"])


@router.get(
    "/metrics/prometheus",
    response_class=Response,
    summary="Prometheus metrics",
    description="Returns all application metrics in Prometheus text exposition format.",
)
def prometheus_metrics():
    """Serve metrics in Prometheus text exposition format.

    Scrape target configuration for ``prometheus.yml``::

        scrape_configs:
          - job_name: 'futures-data-service'
            scrape_interval: 15s
            static_configs:
              - targets: ['data-service:8000']
            metrics_path: '/metrics/prometheus'
    """
    # Refresh gauges from live state before generating output
    _collect_live_gauges()

    # Generate Prometheus text format
    output = generate_latest(_registry)

    return Response(
        content=output,
        media_type=CONTENT_TYPE_LATEST,
    )


# ---------------------------------------------------------------------------
# Convenience: get the registry (for tests or custom collectors)
# ---------------------------------------------------------------------------


def get_registry() -> CollectorRegistry:
    """Return the application's Prometheus CollectorRegistry."""
    return _registry
