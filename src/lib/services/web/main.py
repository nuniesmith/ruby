"""
Web Service — HTMX Dashboard Frontend
========================================
Thin FastAPI frontend that serves the HTMX dashboard HTML page and
reverse-proxies all API + SSE requests to the data service backend.

Architecture:
    Browser  ──→  Web Service (port 8080)  ──→  Data Service (port 8000)
                   │                              │
                   ├─ GET /  (dashboard HTML)      ├─ GET /api/focus
                   ├─ GET /sse/dashboard ──proxy──→├─ GET /sse/dashboard
                   ├─ GET /api/* ──────proxy──────→├─ GET /api/*
                   └─ static assets (Tailwind)     └─ all API endpoints

The web service is fully stateless — it never touches Redis or Postgres
directly. All data flows through the data service API layer.

Benefits of splitting:
  - Dashboard can be scaled/restarted independently of the API.
  - CDN/caching for static assets without affecting API latency.
  - Cleaner security boundary: web faces users, data faces engine.
  - Frontend can be replaced (e.g. React/Next.js) without touching API.

Environment variables:
    DATA_SERVICE_URL   — URL of the data service (default: http://data:8000)
    WEB_HOST           — Bind host (default: 0.0.0.0)
    WEB_PORT           — Bind port (default: 8080)
    LOG_LEVEL          — Logging level (default: info)

Usage:
    PYTHONPATH=src uvicorn lib.services.web.main:app --host 0.0.0.0 --port 8080

Docker:
    ENV PYTHONPATH="/app/src"
    CMD ["uvicorn", "lib.services.web.main:app", "--host", "0.0.0.0", "--port", "8080"]
"""

import asyncio
import os
from contextlib import asynccontextmanager
from urllib.parse import urlencode

import httpx
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse

from lib.core.logging_config import get_logger, setup_logging

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_SERVICE_URL = os.getenv("DATA_SERVICE_URL", "http://data:8000").rstrip("/")
CHARTING_SERVICE_URL = os.getenv("CHARTING_SERVICE_URL", "http://charting:8003").rstrip("/")

# Shared secret for authenticating with the data service.
# Must match the API_KEY set on the data service container.
# When empty, no header is injected (data service auth is also disabled).
_DATA_SERVICE_API_KEY: str = os.getenv("API_KEY", "").strip()

WEB_HOST = os.getenv("WEB_HOST", "0.0.0.0")
WEB_PORT = int(os.getenv("WEB_PORT", "8080"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "info")

# Timeout configuration for proxied requests
PROXY_TIMEOUT_DEFAULT = 15.0  # seconds for normal API calls
PROXY_CONNECT_TIMEOUT = 5.0  # seconds to establish connection

# SSE-specific: heartbeat interval on the data service is 30s, so we
# need to tolerate gaps of at least that long without any data arriving.
# 90s gives us 3x headroom before considering the upstream dead.
SSE_READ_TIMEOUT = 90.0

setup_logging(service="web")
logger = get_logger("web")

# ---------------------------------------------------------------------------
# HTTP client — shared async httpx client for proxying to data service
# ---------------------------------------------------------------------------

_http_client: httpx.AsyncClient | None = None


# Dedicated SSE client — separate from the regular API proxy client so
# that connection-pool keepalive/expiry settings for short-lived API
# requests don't accidentally kill the long-lived SSE stream.
_sse_client: httpx.AsyncClient | None = None


def _get_client() -> httpx.AsyncClient:
    """Return the shared async HTTP client for regular API proxying."""
    global _http_client
    if _http_client is None or _http_client.is_closed:
        _http_client = httpx.AsyncClient(
            base_url=DATA_SERVICE_URL,
            timeout=httpx.Timeout(
                PROXY_TIMEOUT_DEFAULT,
                connect=PROXY_CONNECT_TIMEOUT,
            ),
            follow_redirects=True,
            limits=httpx.Limits(
                max_connections=50,
                max_keepalive_connections=20,
                keepalive_expiry=30.0,
            ),
        )
    return _http_client


# ---------------------------------------------------------------------------
# Charting HTTP client — separate pool for the charting service (port 8003)
# ---------------------------------------------------------------------------

_charting_client: httpx.AsyncClient | None = None


def _get_charting_client() -> httpx.AsyncClient:
    """Return the shared async HTTP client for charting service proxying."""
    global _charting_client
    if _charting_client is None or _charting_client.is_closed:
        _charting_client = httpx.AsyncClient(
            base_url=CHARTING_SERVICE_URL,
            timeout=httpx.Timeout(
                PROXY_TIMEOUT_DEFAULT,
                connect=PROXY_CONNECT_TIMEOUT,
            ),
            follow_redirects=True,
            limits=httpx.Limits(
                max_connections=10,
                max_keepalive_connections=5,
                keepalive_expiry=30.0,
            ),
        )
    return _charting_client


def _get_sse_client() -> httpx.AsyncClient:
    """Return a dedicated async HTTP client for SSE streaming.

    This client has NO keepalive expiry and a generous read timeout so
    that the long-lived EventSource connection isn't reaped by the
    connection pool or timed out between heartbeats.
    """
    global _sse_client
    if _sse_client is None or _sse_client.is_closed:
        _sse_client = httpx.AsyncClient(
            base_url=DATA_SERVICE_URL,
            timeout=httpx.Timeout(
                timeout=SSE_READ_TIMEOUT,
                connect=PROXY_CONNECT_TIMEOUT,
            ),
            follow_redirects=True,
            limits=httpx.Limits(
                max_connections=10,
                max_keepalive_connections=10,
                # No expiry — SSE streams are indefinitely long-lived
                keepalive_expiry=None,
            ),
        )
    return _sse_client


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown lifecycle for the web service."""
    logger.info("=" * 60)
    logger.info("  Web Service starting up")
    logger.info("  Data service backend:     %s", DATA_SERVICE_URL)
    logger.info("  Charting service backend: %s", CHARTING_SERVICE_URL)
    logger.info("  Trainer routed via:       data service /trainer/*")
    logger.info("=" * 60)

    # Pre-warm the HTTP clients
    _get_client()
    _get_charting_client()

    yield

    # Shutdown
    logger.info("Web Service shutting down...")
    global _http_client, _sse_client, _charting_client
    if _http_client is not None and not _http_client.is_closed:
        await _http_client.aclose()
        _http_client = None
    if _sse_client is not None and not _sse_client.is_closed:
        await _sse_client.aclose()
        _sse_client = None
    if _charting_client is not None and not _charting_client.is_closed:
        await _charting_client.aclose()
        _charting_client = None
    logger.info("Web Service stopped")


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Ruby Futures — Web Dashboard",
    description=(
        "HTMX dashboard frontend for Ruby Futures. "
        "Serves the dashboard HTML and proxies API/SSE requests to "
        "the data service backend."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow all origins for the dashboard (browser-facing)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Helper: build proxied headers
# ---------------------------------------------------------------------------


def _proxy_headers(request: Request) -> dict[str, str]:
    """Build headers to forward to the data service.

    Strips hop-by-hop headers and adds X-Forwarded-* headers.
    Always injects the server-side API_KEY as X-API-Key so the data
    service authenticates the web service regardless of what the
    browser sends.
    """
    # Headers that should NOT be forwarded (hop-by-hop)
    skip = {
        "host",
        "connection",
        "keep-alive",
        "transfer-encoding",
        "te",
        "trailer",
        "upgrade",
        "proxy-authorization",
        "proxy-authenticate",
        # Strip any key the browser may have sent — we always use the
        # server-side key so a missing/wrong browser key can't bypass auth.
        "x-api-key",
    }

    headers = {}
    for key, value in request.headers.items():
        if key.lower() not in skip:
            headers[key] = value

    # Add forwarding headers
    client_host = request.client.host if request.client else "unknown"
    headers["X-Forwarded-For"] = client_host
    headers["X-Forwarded-Proto"] = request.url.scheme
    headers["X-Forwarded-Host"] = request.headers.get("host", "")

    # Inject the server-side API key so the data service accepts the request.
    if _DATA_SERVICE_API_KEY:
        headers["X-API-Key"] = _DATA_SERVICE_API_KEY

    return headers


# ---------------------------------------------------------------------------
# Health check — local (no proxy needed)
# ---------------------------------------------------------------------------


@app.get("/health")
async def health():
    """Web service health check.

    Also checks connectivity to the data service backend.
    """
    backend_ok = False
    backend_status = "unknown"

    try:
        client = _get_client()
        resp = await client.get("/health", timeout=5.0)
        backend_ok = resp.status_code == 200
        backend_status = "healthy" if backend_ok else f"unhealthy (HTTP {resp.status_code})"
    except httpx.ConnectError:
        backend_status = "unreachable"
    except httpx.TimeoutException:
        backend_status = "timeout"
    except Exception as exc:
        backend_status = f"error: {exc}"

    return JSONResponse(
        status_code=200 if backend_ok else 503,
        content={
            "status": "ok" if backend_ok else "degraded",
            "service": "web",
            "backend": {
                "url": DATA_SERVICE_URL,
                "status": backend_status,
                "healthy": backend_ok,
            },
        },
    )


# ---------------------------------------------------------------------------
# Dashboard — GET / serves the full HTML page from data service
# ---------------------------------------------------------------------------


@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Serve the HTMX dashboard page.

    Fetches the full dashboard HTML from the data service and returns it.
    The dashboard's SSE and HTMX polling requests will come back to this
    web service, which proxies them to the data service.
    """
    try:
        client = _get_client()
        resp = await client.get(
            "/",
            headers=_proxy_headers(request),
            timeout=PROXY_TIMEOUT_DEFAULT,
        )
        return HTMLResponse(
            content=resp.text,
            status_code=resp.status_code,
        )
    except httpx.ConnectError:
        return HTMLResponse(
            content=_render_error_page(
                "Data Service Unavailable",
                f"Cannot connect to data service at {DATA_SERVICE_URL}. "
                "The service may be starting up — try refreshing in a few seconds.",
            ),
            status_code=503,
        )
    except httpx.TimeoutException:
        return HTMLResponse(
            content=_render_error_page(
                "Data Service Timeout",
                "The data service took too long to respond. Please try again.",
            ),
            status_code=504,
        )
    except Exception as exc:
        logger.error("Dashboard proxy error: %s", exc, exc_info=True)
        return HTMLResponse(
            content=_render_error_page("Internal Error", str(exc)),
            status_code=500,
        )


# ---------------------------------------------------------------------------
# SSE proxy — streams events from data service to browser
# ---------------------------------------------------------------------------


@app.get("/sse/dashboard")
async def sse_dashboard_proxy(request: Request):
    """Proxy the SSE dashboard stream from the data service.

    This is a long-lived streaming connection. We open a streaming
    request to the data service and forward each chunk to the browser.

    Uses a dedicated httpx client (_get_sse_client) so that the
    connection-pool keepalive settings for short-lived API requests
    don't prematurely close the SSE stream.

    If the upstream connection drops or times out (SSE_READ_TIMEOUT
    without receiving any data — including heartbeats), we send the
    browser a retry hint and close gracefully. The browser-side
    EventSource (or HTMX sse.js) will auto-reconnect.
    """

    async def _stream():
        try:
            client = _get_sse_client()
            async with client.stream(
                "GET",
                "/sse/dashboard",
                headers={
                    **_proxy_headers(request),
                    "Accept": "text/event-stream",
                    "Cache-Control": "no-cache",
                },
                timeout=httpx.Timeout(
                    # Read timeout: if no data (not even a heartbeat)
                    # arrives within this window, assume upstream is dead.
                    timeout=SSE_READ_TIMEOUT,
                    connect=PROXY_CONNECT_TIMEOUT,
                ),
            ) as resp:
                async for chunk in resp.aiter_bytes():
                    yield chunk

        except httpx.ConnectError:
            logger.warning("SSE proxy: cannot connect to data service")
            yield _sse_error_event("Cannot connect to data service")
            # Send a retry hint so EventSource reconnects quickly
            yield b"retry: 3000\n\n"
        except (httpx.ReadTimeout, httpx.TimeoutException):
            logger.warning("SSE proxy: upstream read timed out (no data for %ss)", SSE_READ_TIMEOUT)
            yield _sse_error_event("Data service stream timed out — reconnecting")
            yield b"retry: 2000\n\n"
        except asyncio.CancelledError:
            logger.debug("SSE proxy cancelled (client navigated away)")
        except Exception as exc:
            logger.error("SSE proxy error: %s", exc)
            yield _sse_error_event(f"Proxy error: {exc}")
            yield b"retry: 3000\n\n"

    return StreamingResponse(
        _stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate, max-age=0",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            # Prevent any intermediate proxy from buffering chunks
            "Content-Type": "text/event-stream; charset=utf-8",
        },
    )


@app.get("/sse/health")
async def sse_health_proxy(request: Request):
    """Proxy the SSE health endpoint."""
    return await _proxy_request(request, "/sse/health")


# ---------------------------------------------------------------------------
# Favicon — return 204 (same as data service)
# ---------------------------------------------------------------------------


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return Response(status_code=204)


# ---------------------------------------------------------------------------
# API proxy — catch-all for /api/*, /analysis/*, /actions/*, etc.
# ---------------------------------------------------------------------------

# Explicit route patterns that should be proxied to the data service.
# We list them explicitly rather than using a blanket catch-all so that
# typos/invalid routes get a clean 404 from the web service.

_PROXY_PREFIXES = (
    "/api/",
    "/analysis/",
    "/actions/",
    "/positions/",
    "/trades",
    "/log_trade",
    "/risk/",
    "/audit/",
    "/journal/",
    "/cnn/",
    "/data/",
    "/metrics",
    "/docs",
    "/openapi.json",
    "/redoc",
)


@app.api_route(
    "/api/{path:path}",
    methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
)
async def proxy_api(request: Request, path: str):
    """Proxy all /api/* requests to the data service."""
    return await _proxy_request(request, f"/api/{path}")


@app.api_route(
    "/analysis/{path:path}",
    methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
)
async def proxy_analysis(request: Request, path: str):
    """Proxy all /analysis/* requests to the data service."""
    return await _proxy_request(request, f"/analysis/{path}")


@app.api_route(
    "/actions/{path:path}",
    methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
)
async def proxy_actions(request: Request, path: str):
    """Proxy all /actions/* requests to the data service."""
    return await _proxy_request(request, f"/actions/{path}")


@app.api_route(
    "/positions/{path:path}",
    methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
)
async def proxy_positions(request: Request, path: str):
    """Proxy all /positions/* requests to the data service."""
    return await _proxy_request(request, f"/positions/{path}")


@app.api_route(
    "/trades",
    methods=["GET", "POST"],
)
async def proxy_trades_root(request: Request):
    """Proxy /trades to the data service."""
    return await _proxy_request(request, "/trades")


@app.api_route(
    "/trades/{path:path}",
    methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
)
async def proxy_trades(request: Request, path: str):
    """Proxy all /trades/* requests to the data service."""
    return await _proxy_request(request, f"/trades/{path}")


@app.api_route(
    "/log_trade",
    methods=["POST"],
)
async def proxy_log_trade(request: Request):
    """Proxy /log_trade to the data service."""
    return await _proxy_request(request, "/log_trade")


@app.api_route(
    "/risk/{path:path}",
    methods=["GET", "POST"],
)
async def proxy_risk(request: Request, path: str):
    """Proxy all /risk/* requests to the data service."""
    return await _proxy_request(request, f"/risk/{path}")


@app.api_route(
    "/audit/{path:path}",
    methods=["GET", "POST"],
)
async def proxy_audit(request: Request, path: str):
    """Proxy all /audit/* requests to the data service."""
    return await _proxy_request(request, f"/audit/{path}")


@app.api_route(
    "/journal/{path:path}",
    methods=["GET", "POST", "PUT", "DELETE"],
)
async def proxy_journal(request: Request, path: str):
    """Proxy all /journal/* requests to the data service."""
    return await _proxy_request(request, f"/journal/{path}")


@app.api_route(
    "/cnn/{path:path}",
    methods=["GET", "POST", "PUT", "DELETE"],
)
async def proxy_cnn(request: Request, path: str):
    """Proxy all /cnn/* requests to the data service (CNN model management).

    Methods include PUT and DELETE for the per-session CNN gate endpoints:
        GET    /cnn/gate                  — view all gate states
        PUT    /cnn/gate/{session_key}    — enable/disable gate for one session
        DELETE /cnn/gate/{session_key}    — remove Redis override for one session
        DELETE /cnn/gate                  — remove all overrides
        GET    /cnn/gate/html             — dashboard HTML fragment
    """
    return await _proxy_request(request, f"/cnn/{path}")


@app.api_route(
    "/kraken/{path:path}",
    methods=["GET"],
)
async def proxy_kraken(request: Request, path: str):
    """Proxy all /kraken/* requests to the data service (Kraken crypto exchange).

    Endpoints:
        GET /kraken/health          — Kraken connectivity + auth status
        GET /kraken/status          — WebSocket feed status + pair list
        GET /kraken/pairs           — Available Kraken pairs and their mappings
        GET /kraken/ticker/{pair}   — Current ticker snapshot for a pair
        GET /kraken/tickers         — All tracked pair tickers in one call
        GET /kraken/ohlcv/{pair}    — Historical OHLCV bars for a pair
        GET /kraken/health/html     — Dashboard HTML fragment for Kraken status
    """
    return await _proxy_request(request, f"/kraken/{path}")


@app.api_route(
    "/bars/{path:path}",
    methods=["GET", "POST"],
)
async def proxy_bars(request: Request, path: str):
    """Proxy all /bars/* requests to the data service (historical bar store).

    Endpoints:
        GET  /bars/{symbol}       — Historical bars for a symbol
        GET  /bars/bulk           — Bulk bar fetch for multiple symbols
        GET  /bars/status         — Bar store status + coverage summary
        GET  /bars/assets         — Available assets in the bar store
        GET  /bars/{symbol}/gaps  — Gap analysis for a symbol
        POST /bars/{symbol}/fill  — Trigger gap-fill for a symbol
        POST /bars/fill/all       — Trigger gap-fill for all symbols
        GET  /bars/fill/status    — Gap-fill job status
    """
    return await _proxy_request(request, f"/bars/{path}")


@app.api_route(
    "/data/{path:path}",
    methods=["GET"],
)
async def proxy_data(request: Request, path: str):
    """Proxy all /data/* requests to the data service."""
    return await _proxy_request(request, f"/data/{path}")


@app.get("/metrics")
async def proxy_metrics_root(request: Request):
    """Proxy /metrics to the data service."""
    return await _proxy_request(request, "/metrics")


@app.get("/metrics/{path:path}")
async def proxy_metrics(request: Request, path: str):
    """Proxy /metrics/* to the data service."""
    return await _proxy_request(request, f"/metrics/{path}")


@app.get("/docs")
async def proxy_docs(request: Request):
    """Proxy /docs to the data service."""
    return await _proxy_request(request, "/docs")


@app.get("/openapi.json")
async def proxy_openapi(request: Request):
    """Proxy /openapi.json to the data service."""
    return await _proxy_request(request, "/openapi.json")


@app.get("/orb-history", response_class=HTMLResponse)
async def proxy_orb_history(request: Request):
    """Proxy the standalone ORB Signal History page from the data service."""
    return await _proxy_request(request, "/orb-history")


@app.get("/rb-history", response_class=HTMLResponse)
async def proxy_rb_history(request: Request):
    """Canonical RB History path — proxies to /orb-history on the data service."""
    return await _proxy_request(request, "/orb-history")


@app.get("/journal/page", response_class=HTMLResponse)
async def proxy_journal_page(request: Request):
    """Proxy the standalone Journal full-page view from the data service."""
    return await _proxy_request(request, "/journal/page")


# ---------------------------------------------------------------------------
# New pages: Charts, Account, Connections
# ---------------------------------------------------------------------------


@app.get("/charts", response_class=HTMLResponse)
async def proxy_charts(request: Request):
    """Proxy the Charts page from the data service."""
    return await _proxy_request(request, "/charts")


@app.get("/account", response_class=HTMLResponse)
async def proxy_account(request: Request):
    """Proxy the Account page from the data service."""
    return await _proxy_request(request, "/account")


@app.get("/connections", response_class=HTMLResponse)
async def proxy_connections(request: Request):
    """Proxy the Connections page from the data service."""
    return await _proxy_request(request, "/connections")


@app.get("/signals", response_class=HTMLResponse)
async def proxy_signals(request: Request):
    """Proxy the Signal History page from the data service."""
    return await _proxy_request(request, "/signals")


# ---------------------------------------------------------------------------
# Health API — clean paths (proxied to data service)
# ---------------------------------------------------------------------------


@app.get("/api/health/html")
async def proxy_health_html(request: Request):
    """Proxy system health HTML fragment."""
    return await _proxy_request(request, "/api/health/html")


@app.get("/api/health")
async def proxy_health_json(request: Request):
    """Proxy system health JSON."""
    return await _proxy_request(request, "/api/health")


# ---------------------------------------------------------------------------
# Trainer proxy — forwards /trainer/* to the DATA service (port 8000)
#
# The data service hosts the full trainer dashboard page at GET /trainer and
# proxies all /trainer/api/* requests onward to the trainer service (port 8200).
# This keeps the trainer UI behind the same auth/CORS boundary as the rest of
# the dashboard, and lets the trainer service remain API-only.
#
# Route map (web → data service):
#   GET  /trainer               → data service GET /trainer    (HTML dashboard)
#   GET  /trainer/config        → data service GET /trainer/config
#   POST /trainer/config        → data service POST /trainer/config
#   GET  /trainer/service_status → data service GET /trainer/service_status
#   GET  /trainer/api/*         → data service GET /trainer/api/* (proxied to trainer:8200)
#   POST /trainer/api/*         → data service POST /trainer/api/*
# ---------------------------------------------------------------------------


@app.get("/trainer", response_class=HTMLResponse)
async def trainer_redirect(request: Request):
    """Serve the trainer dashboard page from the data service."""
    return await _proxy_request(request, "/trainer")


@app.api_route(
    "/trainer/{path:path}",
    methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
)
async def proxy_trainer(request: Request, path: str):
    """Proxy all /trainer/* requests to the data service.

    The data service hosts the trainer HTML dashboard at GET /trainer and
    forwards /trainer/api/* requests onward to the trainer service (port 8200).

    Examples:
        GET  /trainer               → data service /trainer          (HTML page)
        GET  /trainer/config        → data service /trainer/config   (JSON)
        POST /trainer/config        → data service /trainer/config   (update URL)
        GET  /trainer/service_status → data service /trainer/service_status
        GET  /trainer/api/status    → data service → trainer /status
        GET  /trainer/api/logs      → data service → trainer /logs
        POST /trainer/api/train     → data service → trainer /train
    """
    return await _proxy_request(request, f"/trainer/{path}")


# ---------------------------------------------------------------------------
# Settings proxy — forwards /settings to the DATA service
#
# The data service hosts the full settings page at GET /settings and
# sub-endpoints for services, features, risk, and API key status.
#
# Route map (web → data service):
#   GET  /settings                       → data service GET /settings  (HTML page)
#   GET  /settings/services/config       → service URL config
#   POST /settings/services/update       → save service URLs
#   GET  /settings/services/probe        → probe all service health
#   GET  /settings/features/config       → feature toggle state
#   POST /settings/features/update       → save feature toggles
#   GET  /settings/risk/config           → risk parameter config
#   POST /settings/risk/update           → save risk parameters
#   GET  /settings/keys/status           → API key status (configured/missing)
# ---------------------------------------------------------------------------


@app.get("/settings", response_class=HTMLResponse)
async def settings_page(request: Request):
    """Serve the settings page from the data service."""
    return await _proxy_request(request, "/settings")


@app.get("/settings/services/config")
async def settings_services_config(request: Request):
    return await _proxy_request(request, "/settings/services/config")


@app.post("/settings/services/update")
async def settings_services_update(request: Request):
    return await _proxy_request(request, "/settings/services/update")


@app.get("/settings/services/probe")
async def settings_services_probe(request: Request):
    return await _proxy_request(request, "/settings/services/probe")


@app.get("/settings/features/config")
async def settings_features_config(request: Request):
    return await _proxy_request(request, "/settings/features/config")


@app.post("/settings/features/update")
async def settings_features_update(request: Request):
    return await _proxy_request(request, "/settings/features/update")


@app.get("/settings/risk/config")
async def settings_risk_config(request: Request):
    return await _proxy_request(request, "/settings/risk/config")


@app.post("/settings/risk/update")
async def settings_risk_update(request: Request):
    return await _proxy_request(request, "/settings/risk/update")


@app.get("/settings/keys/status")
async def settings_keys_status(request: Request):
    return await _proxy_request(request, "/settings/keys/status")


@app.get("/settings/rithmic/panel", response_class=HTMLResponse)
async def settings_rithmic_panel(request: Request):
    """Proxy the Rithmic account settings panel fragment."""
    return await _proxy_request(request, "/settings/rithmic/panel")


# ---------------------------------------------------------------------------
# Rithmic proxy routes — read-only prop account monitoring
#
# Route map (web → data service):
#   GET  /api/rithmic/accounts              → list configured accounts (no creds)
#   GET  /api/rithmic/status                → live status JSON for all accounts
#   GET  /api/rithmic/status/html           → HTMX fragment for Connections page
#   GET  /api/rithmic/account/{key}         → single account detail JSON
#   POST /api/rithmic/account/{key}/refresh → force-refresh a single account
#   POST /api/rithmic/account/{key}/save    → save account config from UI
#   POST /api/rithmic/account/{key}/config  → save account config (API)
#   DELETE /api/rithmic/account/{key}/remove → remove account
#   POST /api/rithmic/refresh-all           → refresh all enabled accounts
#   POST /api/rithmic/config/new-key        → create blank account placeholder
#   GET  /api/rithmic/deps                  → dependency check (async-rithmic etc.)
# ---------------------------------------------------------------------------


@app.get("/api/rithmic/accounts")
async def proxy_rithmic_accounts(request: Request):
    return await _proxy_request(request, "/api/rithmic/accounts")


@app.get("/api/rithmic/status")
async def proxy_rithmic_status(request: Request):
    return await _proxy_request(request, "/api/rithmic/status")


@app.get("/api/rithmic/status/html", response_class=HTMLResponse)
async def proxy_rithmic_status_html(request: Request):
    return await _proxy_request(request, "/api/rithmic/status/html")


@app.get("/api/rithmic/account/{key}")
async def proxy_rithmic_account(request: Request, key: str):
    return await _proxy_request(request, f"/api/rithmic/account/{key}")


@app.post("/api/rithmic/account/{key}/refresh")
async def proxy_rithmic_refresh(request: Request, key: str):
    return await _proxy_request(request, f"/api/rithmic/account/{key}/refresh")


@app.post("/api/rithmic/account/{key}/save")
async def proxy_rithmic_save(request: Request, key: str):
    return await _proxy_request(request, f"/api/rithmic/account/{key}/save")


@app.post("/api/rithmic/account/{key}/config")
async def proxy_rithmic_config(request: Request, key: str):
    return await _proxy_request(request, f"/api/rithmic/account/{key}/config")


@app.delete("/api/rithmic/account/{key}/remove")
async def proxy_rithmic_remove(request: Request, key: str):
    return await _proxy_request(request, f"/api/rithmic/account/{key}/remove")


@app.post("/api/rithmic/refresh-all")
async def proxy_rithmic_refresh_all(request: Request):
    return await _proxy_request(request, "/api/rithmic/refresh-all")


@app.post("/api/rithmic/config/new-key")
async def proxy_rithmic_new_key(request: Request):
    return await _proxy_request(request, "/api/rithmic/config/new-key")


@app.get("/api/rithmic/deps")
async def proxy_rithmic_deps(request: Request):
    return await _proxy_request(request, "/api/rithmic/deps")


# ---------------------------------------------------------------------------
# Live Risk proxy routes
# Phase 5B/5E: Real-time risk budget + persistent risk dashboard strip
# ---------------------------------------------------------------------------


@app.get("/api/live-risk")
async def proxy_live_risk(request: Request):
    return await _proxy_request(request, "/api/live-risk")


@app.get("/api/live-risk/html")
async def proxy_live_risk_html(request: Request):
    """Proxy the HTML risk strip partial for HTMX injection."""
    return await _proxy_request(request, "/api/live-risk/html")


@app.get("/api/live-risk/summary")
async def proxy_live_risk_summary(request: Request):
    return await _proxy_request(request, "/api/live-risk/summary")


@app.post("/api/live-risk/refresh")
async def proxy_live_risk_refresh(request: Request):
    return await _proxy_request(request, "/api/live-risk/refresh")


@app.get("/api/live-risk/position/{asset_name}/html")
async def proxy_live_risk_position_html(request: Request, asset_name: str):
    return await _proxy_request(request, f"/api/live-risk/position/{asset_name}/html")


# ---------------------------------------------------------------------------
# SSE proxy helper (for long-lived streaming endpoints)
# ---------------------------------------------------------------------------


async def _proxy_sse_request(request: Request, path: str) -> StreamingResponse:
    """Forward an SSE request to the data service and stream the response.

    Uses the dedicated SSE client with longer timeouts for long-lived
    streaming connections (pipeline run, live price stream, etc.).
    """
    client = _get_sse_client()

    url = path
    if request.query_params:
        url = f"{path}?{request.query_params}"

    async def _stream():
        try:
            async with client.stream(
                "GET",
                url,
                headers=_proxy_headers(request),
                timeout=httpx.Timeout(
                    connect=PROXY_CONNECT_TIMEOUT,
                    read=SSE_READ_TIMEOUT,
                    write=30.0,
                    pool=30.0,
                ),
            ) as resp:
                async for chunk in resp.aiter_bytes():
                    yield chunk
        except httpx.ConnectError:
            yield b'data: {"type":"error","message":"Data service unavailable"}\n\n'
        except httpx.TimeoutException:
            yield b'data: {"type":"error","message":"Data service timeout"}\n\n'
        except Exception as exc:
            logger.error("SSE proxy error for %s: %s", path, exc)
            err_msg = str(exc).replace('"', '\\"')
            yield f'data: {{"type":"error","message":"{err_msg}"}}\n\n'.encode()

    return StreamingResponse(
        _stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate, max-age=0",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ---------------------------------------------------------------------------
# Pipeline / Trading Workflow proxy routes
# Morning pipeline, plan management, live trading, journal, trading settings
# ---------------------------------------------------------------------------


@app.get("/api/pipeline/run")
async def proxy_pipeline_run(request: Request):
    """SSE proxy for the morning analysis pipeline."""
    return await _proxy_sse_request(request, "/api/pipeline/run")


@app.get("/api/pipeline/status")
async def proxy_pipeline_status(request: Request):
    return await _proxy_request(request, "/api/pipeline/status")


@app.post("/api/pipeline/reset")
async def proxy_pipeline_reset(request: Request):
    return await _proxy_request(request, "/api/pipeline/reset")


@app.get("/api/plan")
async def proxy_plan(request: Request):
    return await _proxy_request(request, "/api/plan")


@app.post("/api/plan/confirm")
async def proxy_plan_confirm(request: Request):
    return await _proxy_request(request, "/api/plan/confirm")


@app.post("/api/plan/unlock")
async def proxy_plan_unlock(request: Request):
    return await _proxy_request(request, "/api/plan/unlock")


@app.get("/api/live/stream")
async def proxy_live_stream(request: Request):
    """SSE proxy for live price/signal stream."""
    return await _proxy_sse_request(request, "/api/live/stream")


@app.get("/api/market/candles")
async def proxy_market_candles(request: Request):
    return await _proxy_request(request, "/api/market/candles")


@app.get("/api/market/cvd")
async def proxy_market_cvd(request: Request):
    return await _proxy_request(request, "/api/market/cvd")


@app.get("/api/journal/trades")
async def proxy_journal_trades(request: Request):
    return await _proxy_request(request, "/api/journal/trades")


@app.post("/api/journal/trades/{trade_id}/grade")
async def proxy_journal_trade_grade(request: Request, trade_id: int):
    return await _proxy_request(request, f"/api/journal/trades/{trade_id}/grade")


@app.get("/api/trading/settings")
async def proxy_trading_settings_get(request: Request):
    return await _proxy_request(request, "/api/trading/settings")


@app.post("/api/trading/settings")
async def proxy_trading_settings_save(request: Request):
    return await _proxy_request(request, "/api/trading/settings")


@app.post("/api/trading/test-rithmic")
async def proxy_trading_test_rithmic(request: Request):
    return await _proxy_request(request, "/api/trading/test-rithmic")


@app.post("/api/trading/test-massive")
async def proxy_trading_test_massive(request: Request):
    return await _proxy_request(request, "/api/trading/test-massive")


@app.post("/api/trading/test-kraken")
async def proxy_trading_test_kraken(request: Request):
    return await _proxy_request(request, "/api/trading/test-kraken")


@app.get("/trading", response_class=HTMLResponse)
async def proxy_trading_page(request: Request):
    """Proxy the full trading workflow dashboard page."""
    return await _proxy_request(request, "/trading")


@app.get("/trading/app", response_class=HTMLResponse)
async def proxy_trading_app(request: Request):
    """Proxy the raw trading SPA served inside the /trading iframe."""
    return await _proxy_request(request, "/trading/app")


# ---------------------------------------------------------------------------
# Copy Trade — explicit proxy routes for RITHMIC-F WebUI cards
# ---------------------------------------------------------------------------


@app.get("/api/copy-trade/rate-alert")
async def proxy_copy_trade_rate_alert(request: Request):
    return await _proxy_request(request, "/api/copy-trade/rate-alert")


@app.get("/api/copy-trade/focus")
async def proxy_copy_trade_focus(request: Request):
    return await _proxy_request(request, "/api/copy-trade/focus")


@app.post("/api/copy-trade/pyramid")
async def proxy_copy_trade_pyramid(request: Request):
    return await _proxy_request(request, "/api/copy-trade/pyramid")


@app.get("/api/copy-trade/accounts/html", response_class=HTMLResponse)
async def proxy_copy_trade_accounts_html(request: Request):
    return await _proxy_request(request, "/api/copy-trade/accounts/html")


# ---------------------------------------------------------------------------
# Ruby Signal Engine — read-only signal data for the Live Trading page
# ---------------------------------------------------------------------------


@app.get("/api/ruby/signals")
async def proxy_ruby_signals(request: Request):
    """Proxy all latest Ruby signals map from data service."""
    return await _proxy_request(request, "/api/ruby/signals")


@app.get("/api/ruby/signal/{symbol}")
async def proxy_ruby_signal(request: Request, symbol: str):
    """Proxy single-symbol Ruby signal from data service."""
    return await _proxy_request(request, f"/api/ruby/signal/{symbol}")


@app.get("/api/ruby/status")
async def proxy_ruby_status(request: Request):
    """Proxy Ruby signal status summary from data service."""
    return await _proxy_request(request, "/api/ruby/status")


@app.get("/api/ruby/status/html", response_class=HTMLResponse)
async def proxy_ruby_status_html(request: Request):
    """Proxy Ruby signal HTMX fragment (signal cards) from data service."""
    return await _proxy_request(request, "/api/ruby/status/html")


# ---------------------------------------------------------------------------
# Chat — RustAssistant-powered multi-turn chat with RA→Grok fallback
# ---------------------------------------------------------------------------


@app.get("/chat", response_class=HTMLResponse)
async def proxy_chat_page(request: Request):
    """Proxy the standalone chat page."""
    return await _proxy_request(request, "/chat")


@app.post("/api/chat")
async def proxy_chat(request: Request):
    """Proxy non-streaming chat POST to data service."""
    return await _proxy_request(request, "/api/chat")


@app.get("/api/chat/status")
async def proxy_chat_status(request: Request):
    """Proxy chat backend status check."""
    return await _proxy_request(request, "/api/chat/status")


@app.get("/api/chat/history")
async def proxy_chat_history(request: Request):
    """Proxy chat history fetch."""
    return await _proxy_request(request, "/api/chat/history")


@app.delete("/api/chat/history")
async def proxy_chat_history_delete(request: Request):
    """Proxy chat history clear."""
    return await _proxy_request(request, "/api/chat/history")


@app.get("/sse/chat")
async def proxy_sse_chat(request: Request):
    """Proxy the streaming SSE chat endpoint from the data service.

    Uses the dedicated SSE client (no keepalive expiry) so long-running
    chat streams are not reaped by the connection pool.
    """
    return await _proxy_sse_request(request, "/sse/chat")


# ---------------------------------------------------------------------------
# Tasks — issue/bug/note capture with RustAssistant GitHub integration
# ---------------------------------------------------------------------------


@app.post("/api/tasks")
async def proxy_tasks_create(request: Request):
    """Proxy task creation."""
    return await _proxy_request(request, "/api/tasks")


@app.get("/api/tasks")
async def proxy_tasks_list(request: Request):
    """Proxy task list."""
    return await _proxy_request(request, "/api/tasks")


@app.get("/api/tasks/html")
async def proxy_tasks_html(request: Request):
    """Proxy tasks HTMX fragment."""
    return await _proxy_request(request, "/api/tasks/html")


@app.get("/api/tasks/status")
async def proxy_tasks_status(request: Request):
    """Proxy tasks subsystem status."""
    return await _proxy_request(request, "/api/tasks/status")


@app.get("/api/tasks/{task_id}")
async def proxy_task_get(request: Request, task_id: int):
    """Proxy single task fetch."""
    return await _proxy_request(request, f"/api/tasks/{task_id}")


@app.get("/api/tasks/{task_id}/html")
async def proxy_task_html(request: Request, task_id: int):
    """Proxy single task card fragment."""
    return await _proxy_request(request, f"/api/tasks/{task_id}/html")


@app.put("/api/tasks/{task_id}")
async def proxy_task_update(request: Request, task_id: int):
    """Proxy task update."""
    return await _proxy_request(request, f"/api/tasks/{task_id}")


@app.delete("/api/tasks/{task_id}")
async def proxy_task_delete(request: Request, task_id: int):
    """Proxy task deletion."""
    return await _proxy_request(request, f"/api/tasks/{task_id}")


@app.post("/api/tasks/{task_id}/github")
async def proxy_task_push_github(request: Request, task_id: int):
    """Proxy task → GitHub push via RustAssistant."""
    return await _proxy_request(request, f"/api/tasks/{task_id}/github")


# ---------------------------------------------------------------------------
# Pine Script Generator — modular indicator assembly and download
# ---------------------------------------------------------------------------


@app.get("/pine", response_class=HTMLResponse)
async def proxy_pine_page(request: Request):
    """Proxy the Pine Script Generator dashboard page."""
    return await _proxy_request(request, "/pine")


@app.get("/api/pine/modules")
async def proxy_pine_modules(request: Request):
    """Proxy Pine module list."""
    return await _proxy_request(request, "/api/pine/modules")


@app.get("/api/pine/module/{name}")
async def proxy_pine_module(request: Request, name: str):
    """Proxy single Pine module content."""
    return await _proxy_request(request, f"/api/pine/module/{name}")


@app.get("/api/pine/params")
async def proxy_pine_params(request: Request):
    """Proxy Pine params.yaml content."""
    return await _proxy_request(request, "/api/pine/params")


@app.put("/api/pine/params")
async def proxy_pine_params_update(request: Request):
    """Proxy Pine params.yaml update."""
    return await _proxy_request(request, "/api/pine/params")


@app.post("/api/pine/generate")
async def proxy_pine_generate(request: Request):
    """Proxy Pine script generation."""
    return await _proxy_request(request, "/api/pine/generate")


@app.get("/api/pine/output")
async def proxy_pine_output(request: Request):
    """Proxy Pine output files list."""
    return await _proxy_request(request, "/api/pine/output")


@app.get("/api/pine/download/{name}")
async def proxy_pine_download(request: Request, name: str):
    """Proxy Pine file download."""
    return await _proxy_request(request, f"/api/pine/download/{name}")


@app.get("/api/pine/stats")
async def proxy_pine_stats(request: Request):
    """Proxy Pine indicator statistics."""
    return await _proxy_request(request, "/api/pine/stats")


@app.get("/api/pine/status/html", response_class=HTMLResponse)
async def proxy_pine_status_html(request: Request):
    """Proxy Pine status HTMX fragment."""
    return await _proxy_request(request, "/api/pine/status/html")


# ---------------------------------------------------------------------------
# Trade Executor — staged position builder with stop-hunt protection
# ---------------------------------------------------------------------------


@app.post("/api/trade/engage")
async def proxy_trade_engage(request: Request):
    """Proxy trade engagement (SCOUT entry)."""
    return await _proxy_request(request, "/api/trade/engage")


@app.get("/api/trade/active")
async def proxy_trade_active(request: Request):
    """Proxy active trades list."""
    return await _proxy_request(request, "/api/trade/active")


@app.get("/api/trade/active/{symbol}")
async def proxy_trade_active_symbol(request: Request, symbol: str):
    """Proxy active trade for a specific symbol."""
    return await _proxy_request(request, f"/api/trade/active/{symbol}")


@app.post("/api/trade/partial")
async def proxy_trade_partial(request: Request):
    """Proxy partial profit taking."""
    return await _proxy_request(request, "/api/trade/partial")


@app.post("/api/trade/close/{symbol}")
async def proxy_trade_close(request: Request, symbol: str):
    """Proxy trade close."""
    return await _proxy_request(request, f"/api/trade/close/{symbol}")


@app.post("/api/trade/set-stop")
async def proxy_trade_set_stop(request: Request):
    """Proxy manual stop set/move."""
    return await _proxy_request(request, "/api/trade/set-stop")


@app.get("/api/trade/status")
async def proxy_trade_status(request: Request):
    """Proxy trade executor status."""
    return await _proxy_request(request, "/api/trade/status")


@app.get("/api/trade/history")
async def proxy_trade_history(request: Request):
    """Proxy completed trade history."""
    return await _proxy_request(request, "/api/trade/history")


@app.post("/api/trade/tick")
async def proxy_trade_tick(request: Request):
    """Proxy tick processing for active trades."""
    return await _proxy_request(request, "/api/trade/tick")


@app.get("/api/trade/rithmic-ready")
async def proxy_trade_rithmic_ready(request: Request):
    """Proxy Rithmic readiness check."""
    return await _proxy_request(request, "/api/trade/rithmic-ready")


# ---------------------------------------------------------------------------
# Session Reports — daily pre/post session reports for performance tracking
# ---------------------------------------------------------------------------


@app.post("/api/reports/pre-session")
async def proxy_reports_pre_session_create(request: Request):
    """Proxy pre-session report generation."""
    return await _proxy_request(request, "/api/reports/pre-session")


@app.get("/api/reports/pre-session")
async def proxy_reports_pre_session_get(request: Request):
    """Proxy pre-session report retrieval."""
    return await _proxy_request(request, "/api/reports/pre-session")


@app.post("/api/reports/post-session")
async def proxy_reports_post_session_create(request: Request):
    """Proxy post-session report generation."""
    return await _proxy_request(request, "/api/reports/post-session")


@app.get("/api/reports/post-session")
async def proxy_reports_post_session_get(request: Request):
    """Proxy post-session report retrieval."""
    return await _proxy_request(request, "/api/reports/post-session")


@app.get("/api/reports/history")
async def proxy_reports_history(request: Request):
    """Proxy report history."""
    return await _proxy_request(request, "/api/reports/history")


@app.post("/api/reports/pre-session/notes")
async def proxy_reports_pre_session_notes(request: Request):
    """Proxy pre-session notes update."""
    return await _proxy_request(request, "/api/reports/pre-session/notes")


@app.post("/api/reports/post-session/notes")
async def proxy_reports_post_session_notes(request: Request):
    """Proxy post-session notes update."""
    return await _proxy_request(request, "/api/reports/post-session/notes")


# ---------------------------------------------------------------------------
# Charting proxy — forwards /charting-proxy/* and /charting/* to charting
#   service (nginx + chart.js on port 8003)
#
# The charting container serves static TradingView Lightweight Charts assets
# (HTML, JS, CSS, images) via Nginx.  These routes let the browser access
# charting assets through the web proxy instead of connecting directly to
# port 8003.
#
# Route map (web → charting service):
#   GET /charting-proxy/                → charting /
#   GET /charting-proxy/{path}          → charting /{path}
#   GET /charting/{path}                → charting /{path}
#   Any method /charting-proxy/api/*    → charting /api/*
# ---------------------------------------------------------------------------


async def _proxy_charting_request(request: Request, upstream_path: str) -> Response:
    """Forward an HTTP request to the charting service and return the response."""
    client = _get_charting_client()

    url = upstream_path
    if request.query_params:
        url = f"{upstream_path}?{urlencode(dict(request.query_params))}"

    body = None
    if request.method in ("POST", "PUT", "PATCH"):
        body = await request.body()

    # Forward only safe headers — strip hop-by-hop and host headers
    skip = {
        "host",
        "connection",
        "keep-alive",
        "transfer-encoding",
        "te",
        "trailer",
        "upgrade",
        "content-length",
    }
    fwd_headers = {k: v for k, v in request.headers.items() if k.lower() not in skip}
    if request.client:
        fwd_headers["X-Forwarded-For"] = request.client.host
    fwd_headers["X-Forwarded-Proto"] = request.url.scheme
    fwd_headers["X-Forwarded-Host"] = request.headers.get("host", "")

    try:
        resp = await client.request(
            method=request.method,
            url=url,
            headers=fwd_headers,
            content=body,
        )

        excluded_resp = {
            "transfer-encoding",
            "content-encoding",
            "content-length",
            "connection",
            "keep-alive",
            "server",
        }
        resp_headers = {k: v for k, v in resp.headers.items() if k.lower() not in excluded_resp}

        return Response(
            content=resp.content,
            status_code=resp.status_code,
            headers=resp_headers,
            media_type=resp.headers.get("content-type"),
        )

    except httpx.ConnectError:
        return JSONResponse(
            status_code=503,
            content={
                "error": "Charting service unavailable",
                "detail": f"Cannot connect to {CHARTING_SERVICE_URL}",
                "hint": "Make sure the charting container is running: docker compose up -d charting",
            },
        )
    except httpx.TimeoutException:
        return JSONResponse(
            status_code=504,
            content={
                "error": "Charting service timeout",
                "detail": "Request to charting service timed out",
            },
        )
    except Exception as exc:
        logger.error("Charting proxy error for %s %s: %s", request.method, upstream_path, exc)
        return JSONResponse(
            status_code=502,
            content={
                "error": "Charting proxy error",
                "detail": str(exc),
            },
        )


@app.api_route("/charting-proxy/", methods=["GET", "HEAD", "OPTIONS"])
async def proxy_charting_root(request: Request):
    """Proxy charting root (index.html) from the charting service.

    HEAD and OPTIONS are included so the dashboard's fetch probe and CORS
    preflight requests don't get a 405 Method Not Allowed.
    """
    return await _proxy_charting_request(request, "/")


@app.api_route(
    "/charting-proxy/{path:path}",
    methods=["GET", "HEAD", "OPTIONS", "POST", "PUT", "PATCH", "DELETE"],
)
async def proxy_charting_path(request: Request, path: str):
    """Proxy all /charting-proxy/* requests to the charting service.

    Strips the ``/charting-proxy`` prefix before forwarding:
        GET  /charting-proxy/index.html  → charting GET /index.html
        GET  /charting-proxy/js/app.js   → charting GET /js/app.js
        GET  /charting-proxy/api/bars    → charting GET /api/bars
    """
    return await _proxy_charting_request(request, f"/{path}")


@app.api_route(
    "/charting/{path:path}",
    methods=["GET", "HEAD", "OPTIONS", "POST", "PUT", "PATCH", "DELETE"],
)
async def proxy_charting_assets(request: Request, path: str):
    """Proxy /charting/* for direct access to charting static assets.

    Strips the ``/charting`` prefix before forwarding:
        GET  /charting/index.html  → charting GET /index.html
        GET  /charting/js/app.js   → charting GET /js/app.js
    """
    return await _proxy_charting_request(request, f"/{path}")


# ---------------------------------------------------------------------------
# Generic proxy helper
# ---------------------------------------------------------------------------


async def _proxy_request(request: Request, path: str) -> Response:
    """Forward an HTTP request to the data service and return the response.

    Handles all HTTP methods, query params, body, and headers.
    Returns the upstream response with matching status code and headers.
    """
    client = _get_client()

    # Build the target URL with query parameters
    url = path
    if request.query_params:
        url = f"{path}?{urlencode(dict(request.query_params))}"

    # Read the request body for POST/PUT/PATCH
    body = None
    if request.method in ("POST", "PUT", "PATCH"):
        body = await request.body()

    headers = _proxy_headers(request)

    try:
        resp = await client.request(
            method=request.method,
            url=url,
            headers=headers,
            content=body,
            timeout=PROXY_TIMEOUT_DEFAULT,
        )

        # Filter response headers — strip hop-by-hop and server headers
        response_headers = {}
        skip_response = {
            "transfer-encoding",
            "connection",
            "keep-alive",
            "server",
        }
        for key, value in resp.headers.items():
            if key.lower() not in skip_response:
                response_headers[key] = value

        return Response(
            content=resp.content,
            status_code=resp.status_code,
            headers=response_headers,
            media_type=resp.headers.get("content-type"),
        )

    except httpx.ConnectError:
        return JSONResponse(
            status_code=503,
            content={
                "error": "Data service unavailable",
                "detail": f"Cannot connect to {DATA_SERVICE_URL}",
            },
        )
    except httpx.TimeoutException:
        return JSONResponse(
            status_code=504,
            content={
                "error": "Data service timeout",
                "detail": "Request to data service timed out",
            },
        )
    except Exception as exc:
        logger.error("Proxy error for %s %s: %s", request.method, path, exc)
        return JSONResponse(
            status_code=502,
            content={
                "error": "Proxy error",
                "detail": str(exc),
            },
        )


# ---------------------------------------------------------------------------
# Error page template
# ---------------------------------------------------------------------------


def _render_error_page(title: str, message: str) -> str:
    """Render a minimal error page with auto-refresh."""
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title} — Ruby Futures</title>
    <meta http-equiv="refresh" content="10">
    <style>
        body {{
            font-family: system-ui, -apple-system, sans-serif;
            background: #09090b;
            color: #a1a1aa;
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
            margin: 0;
        }}
        .container {{
            text-align: center;
            max-width: 480px;
            padding: 2rem;
        }}
        h1 {{
            color: #ef4444;
            font-size: 1.5rem;
            margin-bottom: 1rem;
        }}
        p {{
            line-height: 1.6;
            margin-bottom: 1.5rem;
        }}
        .retry {{
            color: #71717a;
            font-size: 0.875rem;
        }}
        .spinner {{
            display: inline-block;
            width: 1rem;
            height: 1rem;
            border: 2px solid #3f3f46;
            border-top-color: #a1a1aa;
            border-radius: 50%;
            animation: spin 0.8s linear infinite;
            margin-right: 0.5rem;
            vertical-align: middle;
        }}
        @keyframes spin {{
            to {{ transform: rotate(360deg); }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <div style="font-size: 3rem; margin-bottom: 1rem;">⚠️</div>
        <h1>{title}</h1>
        <p>{message}</p>
        <div class="retry">
            <span class="spinner"></span>
            Auto-refreshing in 10 seconds...
        </div>
    </div>
</body>
</html>"""


def _sse_error_event(message: str) -> bytes:
    """Format an SSE error event as bytes."""
    import json

    payload = json.dumps({"type": "error", "message": message})
    event = f"event: error\ndata: {payload}\n\n"
    return event.encode("utf-8")


# ---------------------------------------------------------------------------
# Run directly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=WEB_HOST,
        port=WEB_PORT,
        log_level=LOG_LEVEL,
    )
