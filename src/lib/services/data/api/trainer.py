"""
Trainer Page API Router
========================
Serves the full Trainer dashboard page at GET /trainer and proxies all
/trainer/api/* requests to the trainer service (default http://100.113.72.63:8200).

The trainer service URL is configurable at runtime and persisted in Redis
so it survives container restarts.  This lets you point the dashboard at a
remote GPU machine (e.g. a Tailscale peer) without rebuilding the image.

Endpoints:
    GET  /trainer                   — Full HTML trainer dashboard page
    GET  /trainer/api/*             — Proxy to trainer service (passthrough)
    POST /trainer/api/*             — Proxy to trainer service (passthrough)
    GET  /trainer/config            — Get current trainer URL config (JSON)
    POST /trainer/config            — Update trainer URL (JSON body: {url:...})
    GET  /trainer/service_status    — Quick JSON health check of trainer service
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
from typing import Any

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response

logger = logging.getLogger("api.trainer")

router = APIRouter(tags=["Trainer"])

# ---------------------------------------------------------------------------
# Trainer URL — default from env, overridable via Redis at runtime
# ---------------------------------------------------------------------------

_DEFAULT_TRAINER_URL = os.getenv("TRAINER_SERVICE_URL", "http://100.113.72.63:8200").rstrip("/")
_TRAINER_URL_REDIS_KEY = "futures:trainer_service_url"

# Shared secret for authenticating with the trainer service.
# Must match TRAINER_API_KEY set on the trainer container.
# When empty, no Authorization header is injected (trainer auth also disabled).
_TRAINER_API_KEY: str = os.getenv("TRAINER_API_KEY", "").strip()

# Module-level httpx client — reused across requests
_trainer_http_client: httpx.AsyncClient | None = None

# Lock to prevent concurrent recreation of the shared client
_trainer_http_client_lock: asyncio.Lock | None = None


def _trainer_auth_headers() -> dict[str, str]:
    """Return Authorization header dict for trainer requests, or empty dict."""
    if _TRAINER_API_KEY:
        return {"Authorization": f"Bearer {_TRAINER_API_KEY}"}
    return {}


def _get_trainer_url() -> str:
    """Read current trainer URL from Redis (falls back to env default)."""
    try:
        from lib.core.cache import cache_get

        val = cache_get(_TRAINER_URL_REDIS_KEY)
        if val:
            url = val.decode() if isinstance(val, bytes) else str(val)
            return url.rstrip("/")
    except Exception:
        pass
    return _DEFAULT_TRAINER_URL


def _set_trainer_url(url: str) -> None:
    """Persist trainer URL in Redis so all replicas see the update."""
    try:
        from lib.core.cache import cache_set

        cache_set(_TRAINER_URL_REDIS_KEY, url.encode(), ttl=0)  # ttl=0 → no expiry
    except Exception as exc:
        logger.warning("Could not persist trainer URL to Redis: %s", exc)


def _make_http_client(base_url: str) -> httpx.AsyncClient:
    """Create a fresh async httpx client for the trainer service."""
    return httpx.AsyncClient(
        base_url=base_url,
        timeout=httpx.Timeout(120.0, connect=5.0),
        follow_redirects=True,
        limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
    )


async def _get_http_client(base_url: str) -> httpx.AsyncClient:
    """Return a reusable async httpx client for the given base URL.

    Thread-safe: uses an asyncio lock to prevent concurrent recreation.
    """
    global _trainer_http_client, _trainer_http_client_lock

    # Lazily create the lock (must happen inside a running event loop)
    if _trainer_http_client_lock is None:
        _trainer_http_client_lock = asyncio.Lock()

    async with _trainer_http_client_lock:
        if (
            _trainer_http_client is None
            or _trainer_http_client.is_closed
            or str(_trainer_http_client.base_url).rstrip("/") != base_url.rstrip("/")
        ):
            if _trainer_http_client and not _trainer_http_client.is_closed:
                await _trainer_http_client.aclose()
            _trainer_http_client = _make_http_client(base_url)
    return _trainer_http_client


async def _invalidate_http_client() -> None:
    """Close and discard the shared client so the next call gets a fresh one.

    Called after a ReadError to avoid reusing a stale keepalive connection.
    """
    global _trainer_http_client, _trainer_http_client_lock

    if _trainer_http_client_lock is None:
        _trainer_http_client_lock = asyncio.Lock()

    async with _trainer_http_client_lock:
        if _trainer_http_client is not None and not _trainer_http_client.is_closed:
            with contextlib.suppress(Exception):
                await _trainer_http_client.aclose()
        _trainer_http_client = None


async def _probe_trainer(url: str) -> dict[str, Any]:
    """Quick health probe of the trainer service. Returns status dict."""
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(4.0, connect=3.0)) as c:
            r = await c.get(f"{url.rstrip('/')}/health", headers=_trainer_auth_headers())
            if r.status_code == 200:
                data = r.json()
                return {"online": True, "uptime_seconds": data.get("uptime_seconds"), "url": url}
            return {"online": False, "url": url, "http_status": r.status_code}
    except httpx.ConnectError:
        return {"online": False, "url": url, "error": "connection refused"}
    except httpx.TimeoutException:
        return {"online": False, "url": url, "error": "timeout"}
    except Exception as exc:
        return {"online": False, "url": url, "error": str(exc)}


# ---------------------------------------------------------------------------
# Trainer config endpoints
# ---------------------------------------------------------------------------


@router.get("/trainer/config")
async def get_trainer_config() -> JSONResponse:
    """Return current trainer service URL configuration."""
    url = _get_trainer_url()
    probe = await _probe_trainer(url)
    return JSONResponse({"trainer_url": url, "default_url": _DEFAULT_TRAINER_URL, "probe": probe})


@router.post("/trainer/config")
async def set_trainer_config(request: Request) -> JSONResponse:
    """Update trainer service URL."""
    try:
        body = await request.json()
        new_url = str(body.get("url", "")).strip().rstrip("/")
        if not new_url.startswith(("http://", "https://")):
            return JSONResponse(status_code=400, content={"error": "URL must start with http:// or https://"})
        _set_trainer_url(new_url)
        probe = await _probe_trainer(new_url)
        logger.info("Trainer URL updated to %s (online=%s)", new_url, probe.get("online"))
        return JSONResponse({"ok": True, "trainer_url": new_url, "probe": probe})
    except Exception as exc:
        return JSONResponse(status_code=400, content={"error": str(exc)})


@router.get("/trainer/service_status")
async def trainer_service_status() -> JSONResponse:
    """Quick health check — probes the trainer service and returns status."""
    url = _get_trainer_url()
    probe = await _probe_trainer(url)
    # Also pull /status if online for richer info
    if probe.get("online"):
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(5.0, connect=3.0)) as c:
                r = await c.get(f"{url}/status", headers=_trainer_auth_headers())
                if r.status_code == 200:
                    probe["status"] = r.json()
        except Exception:
            pass
    return JSONResponse(probe)


# ---------------------------------------------------------------------------
# Trainer API proxy — /trainer/api/* → trainer service /*
# ---------------------------------------------------------------------------


@router.api_route(
    "/trainer/api/{path:path}",
    methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
)
async def proxy_trainer_api(request: Request, path: str) -> Response:
    """Proxy /trainer/api/* to the trainer service.

    Strips the /trainer/api prefix before forwarding so that:
        GET /trainer/api/status        → trainer GET /status
        GET /trainer/api/logs          → trainer GET /logs
        GET /trainer/api/models        → trainer GET /models
        POST /trainer/api/train        → trainer POST /train
        POST /trainer/api/train/cancel → trainer POST /train/cancel
    """
    trainer_url = _get_trainer_url()
    client = await _get_http_client(trainer_url)

    upstream_path = f"/{path}"
    if request.url.query:
        upstream_path = f"{upstream_path}?{request.url.query}"

    body = await request.body()
    # Forward the request — strip hop-by-hop headers and any
    # Authorization the browser may have sent so we always use the
    # server-side TRAINER_API_KEY instead.
    skip = {
        "host",
        "connection",
        "keep-alive",
        "transfer-encoding",
        "te",
        "trailer",
        "upgrade",
        "content-length",
        "authorization",
    }
    fwd_headers = {k: v for k, v in request.headers.items() if k.lower() not in skip}
    # Inject the server-side Bearer token so the trainer authenticates us.
    fwd_headers.update(_trainer_auth_headers())

    # Retry once on ReadError — the shared keepalive connection may have
    # gone stale (trainer restarted, server-side close mid-stream, etc.).
    # On the first ReadError we discard the pooled client and retry with a
    # brand-new connection before giving up.
    for attempt in range(2):
        try:
            resp = await client.request(
                method=request.method,
                url=upstream_path,
                headers=fwd_headers,
                content=body or None,
            )

            excluded_resp = {"transfer-encoding", "content-encoding", "content-length"}
            resp_headers = {k: v for k, v in resp.headers.items() if k.lower() not in excluded_resp}

            return Response(
                content=resp.content,
                status_code=resp.status_code,
                headers=resp_headers,
                media_type=resp.headers.get("content-type", "application/json"),
            )

        except httpx.ReadError as exc:
            # Stale keepalive — discard the pooled client and retry once.
            await _invalidate_http_client()
            if attempt == 0:
                logger.debug(
                    "Trainer proxy ReadError for %s (attempt %d) — retrying with fresh connection: %s",
                    upstream_path,
                    attempt + 1,
                    exc,
                )
                client = await _get_http_client(trainer_url)
                continue
            # Second attempt also failed — surface as 502
            logger.warning("Trainer proxy ReadError for %s after retry: %s", upstream_path, exc)
            return JSONResponse(
                status_code=502,
                content={"error": "Trainer connection reset — please retry", "detail": str(exc)},
            )

        except httpx.ConnectError:
            return JSONResponse(
                status_code=503,
                content={
                    "error": "Trainer service unavailable",
                    "trainer_url": trainer_url,
                    "hint": "docker compose --profile training up -d trainer",
                },
            )
        except httpx.TimeoutException:
            return JSONResponse(status_code=504, content={"error": "Trainer service timed out"})
        except Exception as exc:
            logger.error("Trainer proxy error for %s: %s", upstream_path, exc, exc_info=True)
            return JSONResponse(status_code=500, content={"error": str(exc)})

    # Unreachable — loop always returns, but satisfies type checker
    return JSONResponse(status_code=500, content={"error": "Unexpected proxy loop exit"})


# ---------------------------------------------------------------------------
# Trainer dashboard HTML page
# ---------------------------------------------------------------------------

_TRAINER_PAGE_HTML = """\
<!DOCTYPE html>
<html lang="en" class="dark">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1.0,viewport-fit=cover"/>
<title>CNN Trainer — Ruby Futures</title>
<link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>🧠</text></svg>"/>
<script>(function(){var t=localStorage.getItem('theme');if(t==='light')document.documentElement.classList.remove('dark');else document.documentElement.classList.add('dark');})();</script>
<script src="https://unpkg.com/htmx.org@2.0.4"></script>
<style>
/* ── Reset & theme ── */
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
:root{
  --bg:#f4f4f5;--bg-panel:rgba(255,255,255,0.85);--bg-inner:rgba(244,244,245,0.6);
  --bg-input:#e4e4e7;--border:#d4d4d8;--border-s:#e4e4e7;
  --text:#18181b;--text2:#3f3f46;--muted:#71717a;--faint:#a1a1aa;
}
.dark{
  --bg:#09090b;--bg-panel:rgba(24,24,27,0.7);--bg-inner:rgba(39,39,42,0.5);
  --bg-input:#27272a;--border:#3f3f46;--border-s:#27272a;
  --text:#f4f4f5;--text2:#d4d4d8;--muted:#71717a;--faint:#52525b;
}
body{font-family:ui-monospace,'Cascadia Code','Fira Code',monospace;background:var(--bg);color:var(--text);min-height:100vh;font-size:13px}

/* ── Nav bar ── */
.nav{display:flex;align-items:center;gap:0;padding:0 1rem;background:var(--bg-panel);
     border-bottom:1px solid var(--border);height:44px;position:sticky;top:0;z-index:100;backdrop-filter:blur(10px)}
.nav-brand{font-weight:700;font-size:0.9rem;color:var(--text);text-decoration:none;margin-right:1.5rem;letter-spacing:-0.02em}
.nav-tab{display:inline-flex;align-items:center;gap:6px;padding:6px 14px;border-radius:6px;
         text-decoration:none;color:var(--muted);font-size:0.78rem;font-weight:500;transition:all .15s;white-space:nowrap}
.nav-tab:hover{background:var(--bg-inner);color:var(--text)}
.nav-tab.active{background:var(--bg-input);color:var(--text);font-weight:600}
.nav-right{margin-left:auto;display:flex;align-items:center;gap:8px}
.theme-btn{background:none;border:1px solid var(--border);border-radius:6px;padding:4px 8px;
           color:var(--muted);cursor:pointer;font-size:0.75rem;transition:all .15s}
.theme-btn:hover{color:var(--text);border-color:var(--text)}

/* ── Layout ── */
.page{padding:1rem;max-width:1600px;margin:0 auto}
.grid-main{display:grid;grid-template-columns:340px 1fr;gap:12px}
@media(max-width:900px){.grid-main{grid-template-columns:1fr}}

/* ── Card ── */
.card{background:var(--bg-panel);border:1px solid var(--border);border-radius:10px;padding:14px}
.card-title{font-size:0.65rem;font-weight:700;text-transform:uppercase;letter-spacing:.1em;color:var(--faint);margin-bottom:10px}

/* ── Status badge ── */
.badge{display:inline-flex;align-items:center;gap:5px;padding:2px 10px;border-radius:9999px;
       font-size:0.7rem;font-weight:700;letter-spacing:.05em;text-transform:uppercase}
.b-idle{background:#1e3a5f22;color:#60a5fa;border:1px solid #1e3a5f}
.b-run{background:#14532d22;color:#4ade80;border:1px solid #14532d}
.b-done{background:#14532d22;color:#86efac;border:1px solid #065f46}
.b-fail{background:#450a0a22;color:#f87171;border:1px solid #450a0a}
.b-cancel{background:#3b270022;color:#fb923c;border:1px solid #3b2700}
.b-online{background:#14532d22;color:#4ade80;border:1px solid #14532d}
.b-offline{background:#450a0a22;color:#f87171;border:1px solid #450a0a}
.b-unknown{background:#27272a22;color:#a1a1aa;border:1px solid #3f3f46}

/* ── Metrics ── */
.metrics-row{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin:10px 0}
.metric{text-align:center}
.metric-val{font-size:1.4rem;font-weight:800;line-height:1}
.metric-lbl{font-size:0.6rem;text-transform:uppercase;letter-spacing:.08em;color:var(--muted);margin-top:3px}

/* ── Progress ── */
.progress-wrap{height:5px;background:var(--bg-input);border-radius:3px;overflow:hidden;margin:6px 0}
.progress-fill{height:100%;background:linear-gradient(90deg,#3b82f6,#06b6d4);border-radius:3px;transition:width .5s}

/* ── Log box ── */
#log-box{background:var(--bg);border:1px solid var(--border);border-radius:8px;
         min-height:400px;max-height:600px;overflow-y:auto;padding:10px;font-size:0.75rem;line-height:1.7;
         font-family:ui-monospace,'Cascadia Code',monospace;user-select:text;white-space:pre-wrap;width:100%}
.ll-INFO{color:#93c5fd}.ll-WARNING{color:#fbbf24}.ll-ERROR{color:#f87171}
.ll-DEBUG{color:var(--faint)}.ll-ts{color:var(--faint);margin-right:5px}
.ll-name{color:#818cf8;margin-right:5px}

/* ── Form controls ── */
label.field-lbl{display:block;font-size:0.65rem;color:var(--muted);text-transform:uppercase;
                letter-spacing:.06em;margin-bottom:3px}
input[type=text],input[type=number],select,input[type=url]{
  background:var(--bg-input);border:1px solid var(--border);border-radius:6px;
  color:var(--text);padding:5px 9px;width:100%;font-size:0.8rem;outline:none;
  font-family:inherit}
input:focus,select:focus{border-color:#3b82f6;box-shadow:0 0 0 2px #3b82f620}
input[type=checkbox]{accent-color:#3b82f6;width:13px;height:13px}
.field-row{margin-bottom:9px}
.grid-2{display:grid;grid-template-columns:1fr 1fr;gap:8px}
.grid-3{display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px}

/* ── Buttons ── */
.btn{border-radius:7px;padding:7px 18px;font-size:0.8rem;font-weight:600;cursor:pointer;
     border:none;transition:opacity .15s;font-family:inherit}
.btn:hover{opacity:.85}.btn:disabled{opacity:.35;cursor:not-allowed}
.btn-primary{background:#2563eb;color:#fff}
.btn-danger{background:#dc2626;color:#fff}
.btn-neutral{background:var(--bg-input);border:1px solid var(--border);color:var(--text)}
.btn-success{background:#16a34a;color:#fff}
.btn-outline-blue{background:var(--bg-input);border:1px solid #3b82f6;color:#60a5fa}
.btn-sm{padding:4px 11px;font-size:0.74rem}
.btn-row{display:flex;gap:7px;flex-wrap:wrap;margin-top:8px}

/* ── Service status bar ── */
.svc-bar{display:flex;align-items:center;gap:10px;padding:7px 12px;
         background:var(--bg-panel);border:1px solid var(--border);border-radius:8px;
         margin-bottom:12px;flex-wrap:wrap}
.svc-pill{display:inline-flex;align-items:center;gap:4px;font-size:0.7rem;font-weight:600;
          padding:2px 9px;border-radius:6px;background:var(--bg-inner)}
.dot{width:7px;height:7px;border-radius:50%;flex-shrink:0}
.dot-green{background:#4ade80}.dot-red{background:#f87171}.dot-gray{background:#6b7280}
.dot-pulse{animation:pulse 1.4s ease-in-out infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.4}}

/* ── Archive table ── */
table{width:100%;border-collapse:collapse;font-size:0.74rem}
th{color:var(--faint);text-align:left;padding:4px 8px;border-bottom:1px solid var(--border);
   font-weight:600;text-transform:uppercase;font-size:0.62rem;letter-spacing:.05em}
td{padding:5px 8px;border-bottom:1px solid var(--border-s)}
tr:last-child td{border-bottom:none}
tr:hover td{background:var(--bg-inner)}

/* ── Contract info ── */
.contract-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(130px,1fr));gap:6px;margin-top:6px}
.contract-item{background:var(--bg-inner);border:1px solid var(--border-s);border-radius:6px;padding:6px 9px}
.contract-idx{font-size:0.6rem;color:var(--faint)}
.contract-name{font-size:0.78rem;font-weight:600;color:var(--text)}

/* ── URL config inline ── */
.url-row{display:flex;align-items:center;gap:6px;margin-top:6px}
.url-row input{flex:1}

/* ── Spinner ── */
.spin{display:inline-block;width:12px;height:12px;border:2px solid var(--border);
      border-top-color:#3b82f6;border-radius:50%;animation:spin .7s linear infinite;vertical-align:middle}
@keyframes spin{to{transform:rotate(360deg)}}

/* ── Champion ── */
.champion-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:6px;margin:8px 0}
.ch-val{font-size:1.1rem;font-weight:800}
.ch-lbl{font-size:0.6rem;text-transform:uppercase;letter-spacing:.07em;color:var(--muted);margin-top:2px}

/* ── Section divider ── */
.section-sep{border:none;border-top:1px solid var(--border-s);margin:10px 0}

/* ── Result detail panel ── */
#results-detail{display:none}
.result-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(200px,1fr));gap:8px;margin-top:8px}
.result-card{background:var(--bg-inner);border:1px solid var(--border-s);border-radius:8px;padding:10px}
.result-card-title{font-size:0.62rem;font-weight:700;text-transform:uppercase;letter-spacing:.08em;color:var(--faint);margin-bottom:6px}

/* ── Symbol pills ── */
.sym-pill{display:inline-block;background:#1e3a5f;color:#7dd3fc;border-radius:4px;
          padding:1px 5px;margin:1px;font-size:0.65rem}

/* ── Scrollbar ── */
::-webkit-scrollbar{width:5px;height:5px}
::-webkit-scrollbar-track{background:transparent}
::-webkit-scrollbar-thumb{background:var(--border);border-radius:3px}
</style>
</head>
<body>

<!-- Nav bar -->
<nav class="nav">
  <a class="nav-brand" href="/">💎 Ruby Futures</a>
  <a class="nav-tab" href="/">📊 Dashboard</a>
  <a class="nav-tab" href="/trading">🚀 Trading</a>
  <a class="nav-tab" href="/charts">📈 Charts</a>
  <a class="nav-tab" href="/account">💰 Account</a>
  <a class="nav-tab" href="/signals">📡 Signals</a>
  <a class="nav-tab" href="/journal/page">📓 Journal</a>
  <a class="nav-tab" href="/connections">🔌 Connections</a>
  <a class="nav-tab active" href="/trainer">🧠 Trainer</a>
  <a class="nav-tab" href="/settings">⚙️ Settings</a>
  <div class="nav-right">
    <div id="nav-trainer-badge" class="badge b-unknown" style="font-size:0.65rem">trainer ···</div>
    <button class="theme-btn" onclick="toggleTheme()">☀/🌙</button>
  </div>
</nav>

<div class="page">

  <!-- Service health bar -->
  <div class="svc-bar" id="svc-bar">
    <span style="font-size:0.7rem;color:var(--muted);font-weight:600;white-space:nowrap">Services:</span>
    <div class="svc-pill" id="svc-engine"><span class="dot dot-gray" id="dot-engine"></span>Engine</div>
    <div class="svc-pill" id="svc-trainer"><span class="dot dot-gray dot-pulse" id="dot-trainer"></span>Trainer</div>
    <div class="svc-pill" id="svc-redis"><span class="dot dot-gray" id="dot-redis"></span>Redis</div>
    <div class="svc-pill" id="svc-pg"><span class="dot dot-gray" id="dot-pg"></span>Postgres</div>
    <div class="svc-pill" id="svc-model" style="display:none"><span class="dot dot-gray" id="dot-model"></span>CNN Model</div>
    <div style="margin-left:auto;display:flex;align-items:center;gap:6px">
      <span style="font-size:0.68rem;color:var(--muted)">Trainer URL:</span>
      <div class="url-row" style="margin:0">
        <input type="url" id="trainer-url-input" style="width:220px;font-size:0.72rem" placeholder="http://100.113.72.63:8200"/>
        <button class="btn btn-neutral btn-sm" onclick="saveTrainerUrl()">Save</button>
        <button class="btn btn-neutral btn-sm" onclick="pollServiceStatus()">↺</button>
      </div>
    </div>
  </div>

  <!-- Main grid -->
  <div class="grid-main">

    <!-- LEFT: Status + Champion + GPU -->
    <div style="display:flex;flex-direction:column;gap:10px">

      <!-- Training status -->
      <div class="card">
        <div class="card-title" style="display:flex;align-items:center;justify-content:space-between">
          <span>Training Status</span>
          <span id="status-badge" class="badge b-idle">idle</span>
        </div>
        <div id="status-progress" style="font-size:0.8rem;color:var(--text2);min-height:20px;margin-bottom:4px">
          Idle — ready to train
        </div>
        <div class="progress-wrap"><div class="progress-fill" id="progress-fill" style="width:0%"></div></div>
        <div class="metrics-row">
          <div class="metric">
            <div class="metric-val" id="m-acc" style="color:#60a5fa">—</div>
            <div class="metric-lbl">Val Acc %</div>
          </div>
          <div class="metric">
            <div class="metric-val" id="m-prec" style="color:#a78bfa">—</div>
            <div class="metric-lbl">Precision %</div>
          </div>
          <div class="metric">
            <div class="metric-val" id="m-rec" style="color:#34d399">—</div>
            <div class="metric-lbl">Recall %</div>
          </div>
        </div>
        <div id="last-result-info" style="font-size:0.72rem;color:var(--muted);min-height:18px"></div>
        <div class="btn-row">
          <button class="btn btn-outline-blue" id="btn-load" onclick="loadData()">📥 Load Data</button>
          <button class="btn btn-neutral" id="btn-gends" onclick="generateDataset()">🗃 Generate Dataset</button>
          <button class="btn btn-primary" id="btn-train" onclick="trainModel()">🧠 Train Model</button>
          <button class="btn btn-success" id="btn-pipeline" onclick="fullPipeline()">🚀 Full Pipeline</button>
          <button class="btn btn-danger" id="btn-cancel" onclick="cancelTrain()" disabled>✕ Cancel</button>
        </div>
        <div id="timing-info" style="font-size:0.67rem;color:var(--faint);margin-top:6px;min-height:14px"></div>
      </div>

      <!-- Champion model -->
      <div class="card">
        <div class="card-title">Champion Model</div>
        <div id="champion-info">
          <div style="color:var(--faint);font-size:0.8rem">No champion model found</div>
        </div>
        <hr class="section-sep"/>
        <div class="btn-row">
          <button class="btn btn-neutral btn-sm" onclick="refreshModels()">📂 Models</button>
        </div>
      </div>

      <!-- GPU -->
      <div class="card">
        <div class="card-title">Hardware</div>
        <div id="gpu-detail" style="font-size:0.8rem;color:var(--text2)">
          <div style="color:var(--faint)">Probing trainer…</div>
        </div>
      </div>

      <!-- Feature contract -->
      <div class="card">
        <div class="card-title">Feature Contract</div>
        <div id="contract-info" style="font-size:0.78rem;color:var(--text2)">
          <div style="color:var(--faint)">Loading…</div>
        </div>
      </div>

    </div>

    <!-- MIDDLE: Config form -->
    <div style="display:flex;flex-direction:column;gap:10px">

      <div class="card" style="flex:1">
        <div class="card-title">Training Configuration</div>

        <div class="grid-2">
          <div class="field-row">
            <label class="field-lbl">Epochs</label>
            <input type="number" id="c-epochs" value="60" min="1" max="200"/>
          </div>
          <div class="field-row">
            <label class="field-lbl">Batch Size</label>
            <input type="number" id="c-batch" value="32" min="8" max="512"/>
          </div>
          <div class="field-row">
            <label class="field-lbl">Learning Rate</label>
            <input type="number" id="c-lr" value="0.0001" step="0.00001" min="0.00001" max="0.1"/>
          </div>
          <div class="field-row">
            <label class="field-lbl">Early Stop Patience</label>
            <input type="number" id="c-patience" value="12" min="1" max="50"/>
          </div>
          <div class="field-row">
            <label class="field-lbl">Days Back</label>
            <input type="number" id="c-days" value="180" min="7" max="365"/>
          </div>
          <div class="field-row">
            <label class="field-lbl">Bars Source</label>
            <select id="c-source">
              <option value="engine" selected>engine (recommended)</option>
              <option value="db">db (direct Postgres)</option>
              <option value="cache">cache (Redis only)</option>
              <option value="massive">massive (direct API)</option>
              <option value="csv">csv (offline)</option>
            </select>
          </div>
          <div class="field-row">
            <label class="field-lbl">ORB Session</label>
            <select id="c-session">
              <option value="all" selected>all</option>
              <option value="us">us</option>
              <option value="london">london</option>
              <option value="london_ny">london_ny</option>
              <option value="frankfurt">frankfurt</option>
              <option value="tokyo">tokyo</option>
              <option value="shanghai">shanghai</option>
              <option value="sydney">sydney</option>
              <option value="cme">cme</option>
              <option value="cme_settle">cme_settle</option>
            </select>
          </div>
          <div class="field-row">
            <label class="field-lbl">Breakout Type</label>
            <select id="c-btype">
              <option value="all" selected>all (recommended)</option>
              <option value="ORB">ORB</option>
              <option value="PDR">PDR — Prev Day</option>
              <option value="IB">IB — Initial Balance</option>
              <option value="CONS">CONS — Consolidation</option>
            </select>
          </div>
        </div>

        <hr class="section-sep"/>
        <div class="card-title" style="margin-bottom:8px">Validation Gates</div>
        <div class="grid-3">
          <div class="field-row">
            <label class="field-lbl">Min Accuracy %</label>
            <input type="number" id="g-acc" value="80" min="0" max="100"/>
          </div>
          <div class="field-row">
            <label class="field-lbl">Min Precision %</label>
            <input type="number" id="g-prec" value="75" min="0" max="100"/>
          </div>
          <div class="field-row">
            <label class="field-lbl">Min Recall %</label>
            <input type="number" id="g-rec" value="70" min="0" max="100"/>
          </div>
        </div>

        <div style="display:flex;align-items:center;gap:7px;margin:4px 0 10px">
          <input type="checkbox" id="c-force"/>
          <label for="c-force" style="font-size:0.75rem;color:#fb923c;cursor:pointer">
            Force promote (bypass gates — use with caution)
          </label>
        </div>

        <hr class="section-sep"/>
        <div class="card-title" style="margin-bottom:8px">Symbols</div>
        <div class="field-row">
          <label class="field-lbl">Custom symbol list (comma-separated — blank = all 22 defaults)</label>
          <input type="text" id="c-symbols" placeholder="e.g. MES,MNQ,MGC  or leave blank for all"/>
        </div>
        <div style="font-size:0.68rem;color:var(--faint);margin-top:3px" id="sym-hint">
          Defaults: MGC SIL MES MNQ M2K MYM ZN ZB ZW
        </div>

        <hr class="section-sep"/>
        <div class="card-title" style="margin-bottom:8px">Post-Training Actions</div>
        <div style="display:flex;gap:10px;flex-wrap:wrap">
          <div style="display:flex;align-items:center;gap:6px">
          </div>
          <div style="display:flex;align-items:center;gap:6px">
            <input type="checkbox" id="opt-contract" checked/>
            <label for="opt-contract" style="font-size:0.75rem;cursor:pointer">Write feature_contract.json</label>
          </div>
        </div>
      </div>

    </div>

  </div><!-- /grid-main -->

  <!-- Full-width Logs section -->
  <div class="card" style="margin-top:12px">
    <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:8px">
      <div class="card-title" style="margin:0">Live Logs</div>
      <div style="display:flex;gap:6px;align-items:center">
        <label style="font-size:0.68rem;color:var(--muted);display:flex;align-items:center;gap:4px;cursor:pointer">
          <input type="checkbox" id="log-auto" checked style="width:11px;height:11px"/> scroll
        </label>
        <button class="btn btn-neutral btn-sm" onclick="copyLogs()">📋 Copy All</button>
        <button class="btn btn-neutral btn-sm" onclick="clearLogs()">clear</button>
      </div>
    </div>
    <div id="log-box"></div>
  </div>

  <!-- Full-width Model Archive -->
  <div class="card" style="margin-top:12px">
    <div class="card-title">Model Archive</div>
    <div id="archive-table" style="overflow-x:auto">
      <div style="color:var(--faint);font-size:0.78rem">Loading…</div>
    </div>
  </div>

  <!-- Last run results (shown after training completes) -->
  <div id="results-detail" class="card" style="margin-top:12px;display:none">
    <div class="card-title">Last Training Run — Full Results</div>
    <div class="result-grid" id="results-grid"></div>
  </div>

</div><!-- /page -->

<script>
'use strict';
// ═══════════════════════════════════════════════════════
// State
// ═══════════════════════════════════════════════════════
let _logOffset = 0;
let _pollSvc   = null;
let _pollStatus = null;
let _pollLogs  = null;
let _trainerOnline = false;

// ═══════════════════════════════════════════════════════
// Boot
// ═══════════════════════════════════════════════════════
document.addEventListener('DOMContentLoaded', () => {
  loadTrainerConfig();
  pollServiceStatus();
  _pollSvc    = setInterval(pollServiceStatus, 12000);
  _pollStatus = setInterval(pollTrainerStatus, 3000);
  _pollLogs   = setInterval(pollLogs, 1800);
  loadContractInfo();
  refreshModels();
});

// ═══════════════════════════════════════════════════════
// Service health bar
// ═══════════════════════════════════════════════════════
async function pollServiceStatus() {
  // Engine health
  setDot('dot-engine', 'gray', true);
  try {
    const r = await fetch('/health', {signal: AbortSignal.timeout(5000)});
    const d = await r.json();
    const eng_ok = d.status === 'ok' || d.status === 'healthy';
    setDot('dot-engine', eng_ok ? 'green' : 'red');
    // Sub-services from engine health
    if (d.backend) {
      setDot('dot-redis',   d.redis_up   !== false ? 'green' : 'red');
      setDot('dot-pg',      d.postgres_up !== false ? 'green' : 'red');
    }
  } catch { setDot('dot-engine', 'red'); }

  // Trainer health — via our proxy
  setDot('dot-trainer', 'gray', true);
  try {
    const r = await fetch('/trainer/service_status', {signal: AbortSignal.timeout(6000)});
    const d = await r.json();
    _trainerOnline = !!d.online;
    setDot('dot-trainer', d.online ? 'green' : 'red');
    document.getElementById('nav-trainer-badge').textContent = d.online ? 'trainer ✓' : 'trainer ✗';
    document.getElementById('nav-trainer-badge').className   = 'badge ' + (d.online ? 'b-online' : 'b-offline');
    if (d.status) renderGpuFromStatus(d.status);
  } catch {
    _trainerOnline = false;
    setDot('dot-trainer', 'red');
  }

  // CNN model on disk
  try {
    const r = await fetch('/api/health', {signal: AbortSignal.timeout(4000)});
    const d = await r.json();
    const modelOk = !!d.cnn_model_on_disk;
    document.getElementById('svc-model').style.display = '';
    setDot('dot-model', modelOk ? 'green' : 'red');
  } catch {}
}

function setDot(id, color, pulse) {
  const el = document.getElementById(id);
  if (!el) return;
  el.className = 'dot dot-' + color + (pulse ? ' dot-pulse' : '');
}

// ═══════════════════════════════════════════════════════
// Trainer config
// ═══════════════════════════════════════════════════════
async function loadTrainerConfig() {
  try {
    const r = await fetch('/trainer/config');
    const d = await r.json();
    document.getElementById('trainer-url-input').value = d.trainer_url || '';
  } catch {}
}

async function saveTrainerUrl() {
  const url = document.getElementById('trainer-url-input').value.trim();
  if (!url) return;
  try {
    const r = await fetch('/trainer/config', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({url}),
    });
    const d = await r.json();
    if (d.ok) {
      addLog({ts: nowTs(), level: 'INFO', name: 'config', msg: 'Trainer URL updated → ' + url});
      pollServiceStatus();
    } else {
      addLog({ts: nowTs(), level: 'ERROR', name: 'config', msg: d.error || 'Failed to save URL'});
    }
  } catch(e) { addLog({ts: nowTs(), level: 'ERROR', name: 'config', msg: String(e)}); }
}

// ═══════════════════════════════════════════════════════
// Trainer status polling
// ═══════════════════════════════════════════════════════
const STATUS_PCT = {
  idle: 0, generating_dataset: 20, training: 52, evaluating: 82, promoting: 96, done: 100, failed: 100, cancelled: 100
};
const STATUS_MSG = {
  idle: 'Idle — ready to train',
  generating_dataset: 'Step 1/4 — Generating dataset…',
  training:           'Step 2/4 — Training model…',
  evaluating:         'Step 3/4 — Evaluating candidate…',
  promoting:          'Step 4/4 — Promoting champion…',
  done:               'Complete ✓',
  failed:             'Failed ✗',
  cancelled:          'Cancelled',
};
const BADGE_CLS = {
  idle: 'b-idle', generating_dataset: 'b-run', training: 'b-run',
  evaluating: 'b-run', promoting: 'b-run', done: 'b-done', failed: 'b-fail', cancelled: 'b-cancel'
};

async function pollTrainerStatus() {
  if (!_trainerOnline) return;
  try {
    const r = await fetch('/trainer/api/status', {signal: AbortSignal.timeout(6000)});
    if (!r.ok) return;
    const s = await r.json();
    renderStatus(s);
  } catch {}
}

function renderStatus(s) {
  const st = s.status || 'idle';
  // Badge
  const badge = document.getElementById('status-badge');
  badge.className = 'badge ' + (BADGE_CLS[st] || 'b-idle');
  badge.textContent = st.replace('_', ' ');

  // Progress
  const pct = STATUS_PCT[st] ?? 0;
  document.getElementById('progress-fill').style.width = pct + '%';

  let msg = s.progress || STATUS_MSG[st] || st;
  if (st === 'failed' && s.error) msg = '✗ ' + s.error;
  document.getElementById('status-progress').textContent = msg;

  // Buttons
  const busy = ['generating_dataset','training','evaluating','promoting'].includes(st);
  document.getElementById('btn-load').disabled     = busy;
  document.getElementById('btn-gends').disabled    = busy;
  document.getElementById('btn-train').disabled    = busy;
  document.getElementById('btn-pipeline').disabled = busy;
  document.getElementById('btn-cancel').disabled   = !busy;

  // Metrics from last result
  if (s.last_result && s.last_result.metrics) {
    const m = s.last_result.metrics;
    document.getElementById('m-acc').textContent  = m.val_accuracy  != null ? (+m.val_accuracy).toFixed(1)  : '—';
    document.getElementById('m-prec').textContent = m.val_precision != null ? (+m.val_precision).toFixed(1) : '—';
    document.getElementById('m-rec').textContent  = m.val_recall    != null ? (+m.val_recall).toFixed(1)    : '—';
  }

  // Last result info line
  if (s.last_result) {
    const r = s.last_result;
    let info = '';
    if (r.promoted)      info += '<span style="color:#4ade80">✓ Promoted to champion</span>';
    else if (r.reason)   info += '<span style="color:#fb923c">⚠ ' + esc(r.reason) + '</span>';
    if (r.dataset)       info += ' · ' + r.dataset.total_images + ' images';
    if (s.finished_at) {
      const d = new Date(s.finished_at);
      info += ' · ' + d.toLocaleTimeString();
    }
    document.getElementById('last-result-info').innerHTML = info;
    renderResults(r);
  }

  // Timing
  if (s.started_at && busy) {
    const age = Math.round((Date.now() - new Date(s.started_at)) / 1000);
    document.getElementById('timing-info').textContent = 'Elapsed: ' + fmtSecs(age);
  } else {
    document.getElementById('timing-info').textContent = '';
  }

  // Champion
  renderChampion(s.champion);

  // GPU
  renderGpuFromStatus(s);

  // Config defaults (once)
  if (s.config && !document.getElementById('c-epochs').dataset.set) {
    document.getElementById('c-epochs').value = s.config.default_epochs    || 60;
    document.getElementById('c-days').value   = s.config.default_days_back || 180;
    if (s.config.default_session) document.getElementById('c-session').value = s.config.default_session;
    document.getElementById('c-epochs').dataset.set = '1';
  }
}

function renderGpuFromStatus(s) {
  const el = document.getElementById('gpu-detail');
  if (!el) return;
  const g = s.gpu;
  if (!g) { el.innerHTML = '<div style="color:var(--faint)">GPU info unavailable</div>'; return; }
  if (g.available) {
    el.innerHTML =
      '<div style="color:#4ade80;font-weight:700;margin-bottom:4px">✓ CUDA Available</div>' +
      '<div style="font-weight:600">' + esc(g.device_name || '') + '</div>' +
      '<div style="color:var(--muted);margin-top:2px">' + (g.device_count||1) + ' device · ' + g.memory_total_gb + ' GB VRAM</div>';
    document.getElementById('nav-trainer-badge').textContent = 'trainer GPU ✓';
  } else {
    el.innerHTML = '<div style="color:#fb923c">⚠ No GPU — training on CPU</div>';
  }
}

function renderChampion(ch) {
  const el = document.getElementById('champion-info');
  if (!ch || !ch.exists) {
    el.innerHTML = '<div style="color:var(--faint);font-size:0.78rem">No champion model on disk</div>';
    return;
  }
  let html = '<div style="color:#4ade80;font-weight:700;font-size:0.82rem;margin-bottom:6px">✓ breakout_cnn_best.pt</div>';
  if (ch.metrics) {
    const m = ch.metrics;
    html += '<div class="champion-grid">';
    html += '<div><div class="ch-val" style="color:#60a5fa">' + fp(m.val_accuracy) + '%</div><div class="ch-lbl">Acc</div></div>';
    html += '<div><div class="ch-val" style="color:#a78bfa">' + fp(m.val_precision) + '%</div><div class="ch-lbl">Prec</div></div>';
    html += '<div><div class="ch-val" style="color:#34d399">' + fp(m.val_recall) + '%</div><div class="ch-lbl">Recall</div></div>';
    html += '</div>';
  }
  if (ch.trained_at) {
    html += '<div style="color:var(--muted);font-size:0.7rem;margin-top:4px">Trained: ' + new Date(ch.trained_at).toLocaleString() + '</div>';
  }
  if (ch.version) html += '<div style="color:var(--faint);font-size:0.68rem">Version: ' + esc(String(ch.version)) + '</div>';
  el.innerHTML = html;
}

function renderResults(r) {
  if (!r) return;
  document.getElementById('results-detail').style.display = '';
  const grid = document.getElementById('results-grid');
  let html = '';

  if (r.metrics) {
    const m = r.metrics;
    html += `<div class="result-card">
      <div class="result-card-title">📈 Metrics</div>
      <div>Accuracy:  <b style="color:#60a5fa">${fp(m.val_accuracy)}%</b></div>
      <div>Precision: <b style="color:#a78bfa">${fp(m.val_precision)}%</b></div>
      <div>Recall:    <b style="color:#34d399">${fp(m.val_recall)}%</b></div>
      <div style="color:var(--muted);margin-top:4px;font-size:0.7rem">
        Epochs: ${m.epochs_trained||'—'} · Best: ${m.best_epoch||'—'}
      </div></div>`;
  }

  if (r.gates) {
    const g = r.gates;
    html += `<div class="result-card">
      <div class="result-card-title">🚦 Gates</div>
      <div style="font-weight:700;color:${g.passed?'#4ade80':'#f87171'};margin-bottom:4px">
        ${g.passed ? '✓ All passed' : '✗ Failed'}
      </div>
      ${(g.failures||[]).map(f=>`<div style="color:#f87171;font-size:0.72rem">• ${esc(f)}</div>`).join('')}
    </div>`;
  }

  if (r.dataset) {
    html += `<div class="result-card">
      <div class="result-card-title">🗃 Dataset</div>
      <div>Images: <b>${r.dataset.total_images}</b></div>
      ${r.promoted ? '<div style="color:#4ade80;margin-top:4px">✓ Promoted</div>' : ''}
      ${r.feature_contract_version ? '<div style="color:var(--muted);font-size:0.7rem">contract v' + r.feature_contract_version + '</div>' : ''}
    </div>`;
  }

  if (r.config) {
    const c = r.config;
    html += `<div class="result-card">
      <div class="result-card-title">⚙ Config</div>
      <div style="font-size:0.72rem;color:var(--text2)">
        <div>Epochs ${c.epochs} · LR ${c.learning_rate} · Batch ${c.batch_size}</div>
        <div>Session: ${c.orb_session} · Source: ${c.bars_source}</div>
        <div>Days: ${c.days_back} · Type: ${c.breakout_type}</div>
        <div style="margin-top:4px;display:flex;flex-wrap:wrap">
          ${(c.symbols||[]).map(s=>`<span class="sym-pill">${s}</span>`).join('')}
        </div>
      </div>
    </div>`;
  }

  grid.innerHTML = html;
}

// ═══════════════════════════════════════════════════════
// Feature contract
// ═══════════════════════════════════════════════════════
async function loadContractInfo() {
  const el = document.getElementById('contract-info');
  try {
    const r = await fetch('/cnn/info');
    if (!r.ok) throw new Error('not ok');
    const d = await r.json();
    let html = '';
    if (d.feature_contract_version) {
      html += `<div style="margin-bottom:6px">
        <span style="color:#4ade80;font-weight:700">v${d.feature_contract_version} contract</span>
        &nbsp;·&nbsp;
        <span style="color:var(--muted)">${d.num_tabular_features||14} features</span>
      </div>`;
    }
    if (d.tabular_features) {
      html += '<div class="contract-grid">';
      d.tabular_features.forEach((f, i) => {
        html += `<div class="contract-item">
          <div class="contract-idx">[${i}]</div>
          <div class="contract-name">${esc(f)}</div>
        </div>`;
      });
      html += '</div>';
    }
    if (!html) html = '<div style="color:var(--faint)">No contract info available</div>';
    el.innerHTML = html;
  } catch {
    el.innerHTML = '<div style="color:var(--faint);font-size:0.75rem">Contract info unavailable — engine may not have a model loaded</div>';
  }
}

// ═══════════════════════════════════════════════════════
// Model archive
// ═══════════════════════════════════════════════════════
async function refreshModels() {
  const el = document.getElementById('archive-table');
  if (!_trainerOnline) {
    el.innerHTML = '<div style="color:var(--faint);font-size:0.75rem">Trainer offline — cannot list models</div>';
    return;
  }
  try {
    const r = await fetch('/trainer/api/models', {signal: AbortSignal.timeout(8000)});
    if (!r.ok) return;
    const d = await r.json();
    const models = d.models || [];
    if (!models.length) {
      el.innerHTML = '<div style="color:var(--faint);font-size:0.75rem">No model files found</div>';
      return;
    }
    let html = '<table><thead><tr><th>File</th><th>Size</th><th>Modified</th><th>Acc</th></tr></thead><tbody>';
    for (const m of models) {
      const isChamp = m.name === 'breakout_cnn_best.pt';
      const color   = isChamp ? '#4ade80' : 'var(--text)';
      const prefix  = isChamp ? '★ ' : '';
      html += `<tr>
        <td style="color:${color};font-weight:${isChamp?'700':'400'}" title="${esc(m.path||'')}">
          ${prefix}${esc(m.name)}
        </td>
        <td style="color:var(--muted)">${fmtBytes(m.size_bytes)}</td>
        <td style="color:var(--faint)">${m.modified ? new Date(m.modified*1000).toLocaleDateString() : '—'}</td>
        <td style="color:#60a5fa">${m.accuracy != null ? (+m.accuracy).toFixed(1)+'%' : '—'}</td>
      </tr>`;
    }
    html += '</tbody></table>';
    el.innerHTML = html;
  } catch(e) {
    el.innerHTML = '<div style="color:var(--faint);font-size:0.75rem">Could not load models: ' + esc(String(e)) + '</div>';
  }
}

// ═══════════════════════════════════════════════════════
// Actions
// ═══════════════════════════════════════════════════════
function buildPayload() {
  const symRaw = document.getElementById('c-symbols').value.trim();
  const symbols = symRaw ? symRaw.split(',').map(s=>s.trim()).filter(Boolean) : null;

  const payload = {
    epochs:        parseInt(document.getElementById('c-epochs').value),
    batch_size:    parseInt(document.getElementById('c-batch').value),
    learning_rate: parseFloat(document.getElementById('c-lr').value),
    patience:      parseInt(document.getElementById('c-patience').value),
    days_back:     parseInt(document.getElementById('c-days').value),
    bars_source:   document.getElementById('c-source').value,
    orb_session:   document.getElementById('c-session').value,
    breakout_type: document.getElementById('c-btype').value,
    min_accuracy:  parseFloat(document.getElementById('g-acc').value),
    min_precision: parseFloat(document.getElementById('g-prec').value),
    min_recall:    parseFloat(document.getElementById('g-rec').value),
    force_promote: document.getElementById('c-force').checked,
  };
  if (symbols) payload.symbols = symbols;
  return payload;
}

function setStepButtonsBusy(busy) {
  document.getElementById('btn-load').disabled     = busy;
  document.getElementById('btn-gends').disabled    = busy;
  document.getElementById('btn-train').disabled    = busy;
  document.getElementById('btn-pipeline').disabled = busy;
}

async function submitJob(payload, logMsg) {
  if (!_trainerOnline) {
    addLog({ts:nowTs(), level:'ERROR', name:'ui', msg:'Trainer is offline — cannot start'});
    return;
  }

  setStepButtonsBusy(true);
  document.getElementById('results-detail').style.display = 'none';

  addLog({ts:nowTs(), level:'INFO', name:'ui', msg: logMsg});

  try {
    const r = await fetch('/trainer/api/train', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify(payload),
    });
    const d = await r.json();
    if (r.status === 202) {
      addLog({ts:nowTs(), level:'INFO', name:'ui', msg:'Job accepted ✓'});
      _logOffset = 0;
    } else if (r.status === 409) {
      addLog({ts:nowTs(), level:'WARNING', name:'ui', msg:'Training already in progress'});
      setStepButtonsBusy(false);
    } else {
      addLog({ts:nowTs(), level:'ERROR', name:'ui', msg:JSON.stringify(d)});
      setStepButtonsBusy(false);
    }
  } catch(e) {
    addLog({ts:nowTs(), level:'ERROR', name:'ui', msg:String(e)});
    setStepButtonsBusy(false);
  }
  pollTrainerStatus();
}

async function loadData() {
  const payload = buildPayload();
  payload.step = 'load_data';
  await submitJob(payload, 'Loading data…');
}

async function generateDataset() {
  const payload = buildPayload();
  payload.step = 'generate_dataset';
  await submitJob(payload, 'Generating dataset…');
}

async function trainModel() {
  const payload = buildPayload();
  payload.step = 'train';
  await submitJob(payload, 'Training model…');
}

async function fullPipeline() {
  const payload = buildPayload();
  await submitJob(payload, 'Starting full pipeline…');
}

async function cancelTrain() {
  if (!confirm('Cancel the current training run?')) return;
  try {
    const r = await fetch('/trainer/api/train/cancel', {method:'POST'});
    const d = await r.json();
    addLog({ts:nowTs(), level:'WARNING', name:'ui', msg: d.message || 'Cancellation requested'});
  } catch(e) { addLog({ts:nowTs(), level:'ERROR', name:'ui', msg:String(e)}); }
  pollTrainerStatus();
}



// ═══════════════════════════════════════════════════════
// Logs
// ═══════════════════════════════════════════════════════
async function pollLogs() {
  if (!_trainerOnline) return;
  try {
    const r = await fetch('/trainer/api/logs?offset=' + _logOffset, {signal: AbortSignal.timeout(5000)});
    if (!r.ok) return;
    const d = await r.json();
    if (d.lines && d.lines.length > 0) {
      for (const l of d.lines) addLog(l);
      _logOffset = d.next_offset;
    }
  } catch {}
}

function addLog(entry) {
  const box = document.getElementById('log-box');
  const div = document.createElement('div');
  const cls = 'll-' + (entry.level || 'INFO');
  div.innerHTML =
    `<span class="ll-ts">${esc(entry.ts||'')}</span>` +
    `<span class="ll-name">[${esc(entry.name||'')}]</span>` +
    `<span class="${cls}">${esc(entry.msg||'')}</span>`;
  box.appendChild(div);
  if (document.getElementById('log-auto').checked) {
    box.scrollTop = box.scrollHeight;
  }
}

function clearLogs() {
  document.getElementById('log-box').innerHTML = '';
}

function copyLogs() {
  const box = document.getElementById('log-box');
  const text = box.innerText;
  if (navigator.clipboard && window.isSecureContext) {
    navigator.clipboard.writeText(text).then(() => {
      addLog({ts: nowTs(), level: 'INFO', name: 'ui', msg: 'Logs copied to clipboard ✓'});
    }).catch(() => { _copyFallback(text); });
  } else {
    _copyFallback(text);
  }
}
function _copyFallback(text) {
  const ta = document.createElement('textarea');
  ta.value = text;
  ta.style.cssText = 'position:fixed;left:-9999px;top:-9999px';
  document.body.appendChild(ta);
  ta.select();
  try {
    document.execCommand('copy');
    addLog({ts: nowTs(), level: 'INFO', name: 'ui', msg: 'Logs copied to clipboard ✓'});
  } catch(e) {
    addLog({ts: nowTs(), level: 'ERROR', name: 'ui', msg: 'Copy failed — please select & copy manually'});
  }
  document.body.removeChild(ta);
}

// ═══════════════════════════════════════════════════════
// Theme
// ═══════════════════════════════════════════════════════
function toggleTheme() {
  const isDark = document.documentElement.classList.toggle('dark');
  localStorage.setItem('theme', isDark ? 'dark' : 'light');
}

// ═══════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════
function fp(v) {
  if (v == null || v === '') return '—';
  return parseFloat(v).toFixed(1);
}
function fmtBytes(b) {
  if (!b) return '—';
  if (b > 1048576) return (b/1048576).toFixed(1) + ' MB';
  if (b > 1024)    return (b/1024).toFixed(0)     + ' KB';
  return b + ' B';
}
function fmtSecs(s) {
  const m = Math.floor(s/60), sec = s % 60;
  return m > 0 ? `${m}m ${sec}s` : `${sec}s`;
}
function nowTs() {
  return new Date().toTimeString().slice(0, 8);
}
function esc(s) {
  if (s == null) return '';
  return String(s)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}
</script>
</body>
</html>
"""


@router.get("/trainer", response_class=HTMLResponse)
async def trainer_page() -> HTMLResponse:
    """Serve the full Trainer dashboard HTML page."""
    return HTMLResponse(content=_TRAINER_PAGE_HTML)
