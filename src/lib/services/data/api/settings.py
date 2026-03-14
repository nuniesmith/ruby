"""
Settings Page API Router
=========================
Serves the full Settings page at GET /settings and provides HTMX fragments
for engine status, live feed controls, and service configuration.

Endpoints:
    GET  /settings                  — Full HTML settings page
    GET  /settings/services/html    — Service connectivity fragment (HTMX)
    POST /settings/services/update  — Update service URLs
    POST /settings/features/update  — Update feature toggles
    POST /settings/risk/update      — Update risk/trading parameters
"""

import json
import logging
import os
from zoneinfo import ZoneInfo

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

logger = logging.getLogger("api.settings")

router = APIRouter(tags=["Settings"])

_EST = ZoneInfo("America/New_York")

# ---------------------------------------------------------------------------
# Engine accessor (injected by main.py after lifespan starts)
# ---------------------------------------------------------------------------

_engine = None


def set_engine(engine) -> None:
    global _engine
    _engine = engine


def _get_engine():
    return _engine  # may be None — callers handle gracefully


# ---------------------------------------------------------------------------
# Redis-backed settings persistence
# ---------------------------------------------------------------------------

_SETTINGS_CACHE_KEY = "settings:overrides"


def _load_persisted_settings() -> dict:
    """Load settings overrides from Redis (or return empty dict)."""
    try:
        from lib.core.cache import cache_get

        raw = cache_get(_SETTINGS_CACHE_KEY)
        if raw:
            return json.loads(raw)
    except Exception:
        pass
    return {}


def _save_persisted_settings(data: dict) -> None:
    """Persist settings overrides to Redis."""
    try:
        from lib.core.cache import cache_set

        cache_set(_SETTINGS_CACHE_KEY, json.dumps(data).encode(), 0)  # 0 = no expiry
    except Exception as exc:
        logger.warning("Failed to persist settings: %s", exc)


# ---------------------------------------------------------------------------
# Service health probes
# ---------------------------------------------------------------------------


def _probe_service(url: str, timeout: float = 3.0) -> dict:
    """Probe a service health endpoint. Returns {ok, latency_ms, error}."""
    import time

    try:
        import httpx

        t0 = time.monotonic()
        with httpx.Client(timeout=timeout) as client:
            resp = client.get(f"{url}/health")
            latency = (time.monotonic() - t0) * 1000
            return {
                "ok": resp.status_code < 500,
                "latency_ms": round(latency, 1),
                "status": resp.status_code,
                "error": "",
            }
    except Exception as exc:
        return {"ok": False, "latency_ms": -1, "status": 0, "error": str(exc)[:120]}


# ---------------------------------------------------------------------------
# Settings page HTML
# ---------------------------------------------------------------------------

_SETTINGS_PAGE_HTML = """\
<!DOCTYPE html>
<html lang="en" class="dark">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1.0,viewport-fit=cover"/>
<title>Settings — Ruby Futures</title>
<link rel="icon" href="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><text y='.9em' font-size='90'>⚙️</text></svg>"/>
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
     border-bottom:1px solid var(--border);height:42px;position:sticky;top:0;z-index:100;backdrop-filter:blur(10px)}
.nav-brand{font-weight:700;font-size:0.9rem;color:var(--text);text-decoration:none;margin-right:1.25rem;letter-spacing:-0.02em}
.nav-tab{display:inline-flex;align-items:center;gap:5px;padding:5px 12px;border-radius:6px;
         text-decoration:none;color:var(--muted);font-size:0.78rem;font-weight:500;transition:all .12s;white-space:nowrap}
.nav-tab:hover{background:var(--bg-input);color:var(--text)}
.nav-tab.active{background:var(--bg-input);color:var(--text);font-weight:650}
.nav-right{margin-left:auto;display:flex;align-items:center;gap:8px}
.theme-btn{background:none;border:1px solid var(--border);border-radius:6px;padding:4px 8px;
           color:var(--muted);cursor:pointer;font-size:0.75rem;transition:all .12s;font-family:inherit}
.theme-btn:hover{color:var(--text);border-color:var(--text)}

/* ── Layout ── */
.page{padding:1rem;max-width:1400px;margin:0 auto}
.grid-2{display:grid;grid-template-columns:1fr 1fr;gap:12px}
.grid-3{display:grid;grid-template-columns:repeat(3,1fr);gap:8px}
@media(max-width:900px){.grid-2{grid-template-columns:1fr}}
@media(max-width:600px){.grid-3{grid-template-columns:1fr 1fr}}

/* ── Section tabs ── */
.section-tabs{display:flex;gap:4px;margin-bottom:16px;border-bottom:1px solid var(--border);padding-bottom:8px}
.section-tab{padding:6px 16px;border-radius:7px 7px 0 0;border:1px solid transparent;
             background:transparent;color:var(--muted);font-size:0.78rem;font-weight:500;
             cursor:pointer;transition:all .12s;font-family:inherit}
.section-tab:hover{color:var(--text);background:var(--bg-inner)}
.section-tab.active{background:var(--bg-panel);color:var(--text);font-weight:700;
                    border-color:var(--border);border-bottom-color:var(--bg)}
.section-content{display:none}.section-content.active{display:block}

/* ── Card ── */
.card{background:var(--bg-panel);border:1px solid var(--border);border-radius:10px;padding:16px;margin-bottom:12px}
.card-title{font-size:0.65rem;font-weight:700;text-transform:uppercase;letter-spacing:.1em;color:var(--faint);margin-bottom:12px;display:flex;align-items:center;justify-content:space-between}

/* ── Form controls ── */
label.lbl{display:block;font-size:0.65rem;color:var(--muted);text-transform:uppercase;letter-spacing:.06em;margin-bottom:3px}
input[type=text],input[type=number],select{
  background:var(--bg-input);border:1px solid var(--border);border-radius:6px;
  color:var(--text);padding:6px 10px;width:100%;font-size:0.8rem;outline:none;font-family:inherit}
input:focus,select:focus{border-color:#3b82f6;box-shadow:0 0 0 2px #3b82f620}
.field{margin-bottom:10px}
.field-hint{font-size:0.62rem;color:var(--faint);margin-top:2px}
.field-row{display:grid;grid-template-columns:1fr auto;gap:8px;align-items:end}
.field-label-row{display:flex;align-items:center;gap:5px;margin-bottom:3px}
.field-label-row label.lbl{margin-bottom:0}
.info-icon{display:inline-flex;align-items:center;justify-content:center;
           width:13px;height:13px;border-radius:50%;font-size:0.6rem;font-style:normal;
           background:var(--bg-input);border:1px solid var(--border);color:var(--muted);
           cursor:default;position:relative;flex-shrink:0;line-height:1}
.info-icon:hover{border-color:#3b82f6;color:#3b82f6}
.info-icon::after{content:attr(data-tip);position:absolute;left:50%;bottom:calc(100% + 6px);
                  transform:translateX(-50%);white-space:normal;width:210px;
                  background:#1c1c1e;color:#e4e4e7;border:1px solid #3f3f46;
                  border-radius:7px;padding:7px 9px;font-size:0.65rem;line-height:1.45;
                  pointer-events:none;opacity:0;transition:opacity .15s;z-index:200;
                  box-shadow:0 4px 16px #0008;font-style:normal}
.info-icon:hover::after{opacity:1}

/* ── Buttons ── */
.btn{border-radius:7px;padding:7px 16px;font-size:0.8rem;font-weight:600;cursor:pointer;
     border:none;transition:opacity .12s;font-family:inherit;display:inline-flex;align-items:center;gap:5px}
.btn:hover{opacity:.85}.btn:disabled{opacity:.35;cursor:not-allowed}
.btn-primary{background:#2563eb;color:#fff}
.btn-danger{background:#dc2626;color:#fff}
.btn-success{background:#16a34a;color:#fff}
.btn-warning{background:#d97706;color:#fff}
.btn-neutral{background:var(--bg-input);border:1px solid var(--border);color:var(--text)}
.btn-sm{padding:4px 11px;font-size:0.74rem}
.btn-row{display:flex;gap:8px;flex-wrap:wrap;margin-top:10px}

/* ── Status badge ── */
.badge{display:inline-flex;align-items:center;gap:5px;padding:2px 9px;border-radius:9999px;
       font-size:0.7rem;font-weight:700;letter-spacing:.05em;text-transform:uppercase}
.b-ok{background:#14532d22;color:#4ade80;border:1px solid #14532d}
.b-warn{background:#3b270022;color:#fb923c;border:1px solid #3b2700}
.b-err{background:#450a0a22;color:#f87171;border:1px solid #450a0a}
.b-info{background:#1e3a5f22;color:#60a5fa;border:1px solid #1e3a5f}
.b-gray{background:#27272a22;color:#a1a1aa;border:1px solid #3f3f46}

/* ── Status row ── */
.status-row{display:flex;align-items:center;justify-content:space-between;
            padding:7px 10px;border-radius:7px;background:var(--bg-inner);
            border:1px solid var(--border-s);margin-bottom:6px;font-size:0.78rem}
.status-key{color:var(--muted);font-size:0.7rem;text-transform:uppercase;letter-spacing:.05em}
.status-val{font-weight:600;color:var(--text);font-size:0.78rem;text-align:right}

/* ── Dot indicators ── */
.dot{width:8px;height:8px;border-radius:50%;display:inline-block;flex-shrink:0}
.dot-green{background:#4ade80}.dot-red{background:#f87171}
.dot-yellow{background:#fbbf24}.dot-gray{background:#6b7280}
.dot-pulse{animation:pulse 1.4s ease-in-out infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.4}}

/* ── Feed status block ── */
.feed-block{padding:10px 12px;border-radius:8px;background:var(--bg-inner);
            border:1px solid var(--border-s);margin-bottom:10px}
.feed-title{font-size:0.65rem;font-weight:700;text-transform:uppercase;letter-spacing:.08em;color:var(--faint);margin-bottom:6px}

/* ── Link row ── */
.link-row{display:flex;align-items:center;gap:10px;padding:8px 10px;
          border-radius:7px;background:var(--bg-inner);border:1px solid var(--border-s);
          margin-bottom:6px;font-size:0.78rem;text-decoration:none;color:var(--text);
          transition:background .12s}
.link-row:hover{background:var(--bg-input)}
.link-icon{font-size:1.1rem;width:22px;text-align:center}
.link-label{font-weight:600}
.link-desc{font-size:0.68rem;color:var(--muted);margin-top:1px}

/* ── Toggle switch ── */
.toggle-row{display:flex;align-items:center;justify-content:space-between;
            padding:8px 10px;border-radius:7px;background:var(--bg-inner);
            border:1px solid var(--border-s);margin-bottom:6px}
.toggle-label{font-size:0.78rem;font-weight:500;color:var(--text)}
.toggle-desc{font-size:0.62rem;color:var(--muted);margin-top:1px}
.toggle-switch{position:relative;width:38px;height:20px;flex-shrink:0}
.toggle-switch input{opacity:0;width:0;height:0}
.toggle-slider{position:absolute;cursor:pointer;top:0;left:0;right:0;bottom:0;
               background:var(--border);border-radius:20px;transition:.2s}
.toggle-slider:before{content:"";position:absolute;height:16px;width:16px;left:2px;bottom:2px;
                       background:var(--text);border-radius:50%;transition:.2s}
.toggle-switch input:checked + .toggle-slider{background:#3b82f6}
.toggle-switch input:checked + .toggle-slider:before{transform:translateX(18px)}

/* ── Toast ── */
#toast{position:fixed;bottom:24px;right:24px;padding:10px 18px;border-radius:8px;
       font-size:0.8rem;font-weight:600;z-index:9999;transition:opacity .3s;
       opacity:0;pointer-events:none;max-width:320px}
#toast.show{opacity:1}
#toast.ok{background:#15803d;color:#fff;border:1px solid #16a34a}
#toast.err{background:#b91c1c;color:#fff;border:1px solid #dc2626}

/* ── Section divider ── */
hr.sep{border:none;border-top:1px solid var(--border-s);margin:12px 0}

/* ── Spinner ── */
.spin{display:inline-block;width:12px;height:12px;border:2px solid var(--border);
      border-top-color:#3b82f6;border-radius:50%;animation:spin .7s linear infinite;vertical-align:middle}
@keyframes spin{to{transform:rotate(360deg)}}

/* ── Account pill selector ── */
.acct-pills{display:flex;gap:8px;flex-wrap:wrap;margin-top:4px}
.acct-pill{padding:6px 16px;border-radius:7px;border:1px solid var(--border);
           background:var(--bg-input);color:var(--muted);font-size:0.78rem;
           cursor:pointer;transition:all .12s;font-family:inherit;font-weight:500}
.acct-pill.selected{background:#1e3a5f;border-color:#3b82f6;color:#93c5fd;font-weight:700}
.acct-pill:hover:not(.selected){border-color:var(--text);color:var(--text)}

/* ── Service card ── */
.svc-card{padding:10px 12px;border-radius:8px;background:var(--bg-inner);
          border:1px solid var(--border-s);margin-bottom:8px}
.svc-header{display:flex;align-items:center;justify-content:space-between;margin-bottom:4px}
.svc-name{font-weight:700;font-size:0.78rem}
.svc-url{font-size:0.68rem;color:var(--muted);font-family:inherit;word-break:break-all}
.svc-detail{font-size:0.65rem;color:var(--faint);margin-top:3px}

/* ── API key status ── */
.key-status{display:flex;align-items:center;gap:8px;padding:8px 10px;border-radius:7px;
            background:var(--bg-inner);border:1px solid var(--border-s);margin-bottom:6px}
.key-name{font-size:0.78rem;font-weight:600;flex:1}
.key-badge{font-size:0.65rem;font-weight:700;padding:2px 8px;border-radius:9999px}
.key-set{background:#14532d22;color:#4ade80;border:1px solid #14532d}
.key-missing{background:#450a0a22;color:#f87171;border:1px solid #450a0a}
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
  <a class="nav-tab" href="/trainer">🧠 Trainer</a>
  <a class="nav-tab active" href="/settings">⚙️ Settings</a>
  <div class="nav-right">
    <button class="theme-btn" onclick="toggleTheme()">☀/🌙</button>
  </div>
</nav>

<div class="page">

  <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:12px">
    <div>
      <div style="font-size:1.2rem;font-weight:700">⚙️ Settings</div>
      <div style="font-size:0.72rem;color:var(--muted);margin-top:2px">Engine · services · features · risk · prop accounts · API keys</div>
    </div>
    <div id="engine-badge" class="badge b-gray">checking…</div>
  </div>

  <!-- Section tabs -->
  <div class="section-tabs">
    <button class="section-tab active" onclick="showSection('engine')">🔧 Engine</button>
    <button class="section-tab" onclick="showSection('services')">🌐 Services</button>
    <button class="section-tab" onclick="showSection('features')">🎛️ Features</button>
    <button class="section-tab" onclick="showSection('risk')">🛡️ Risk &amp; Trading</button>
    <button class="section-tab" onclick="showSection('propaccounts')">🏦 Prop Accounts</button>
    <button class="section-tab" onclick="showSection('keys')">🔑 API Keys</button>
  </div>

  <!-- ═══════════════════════════════════════════════════════════ -->
  <!-- ENGINE SECTION (original settings page content)           -->
  <!-- ═══════════════════════════════════════════════════════════ -->
  <div id="section-engine" class="section-content active">
    <div class="grid-2">
      <div>
        <!-- Engine runtime settings -->
        <div class="card">
          <div class="card-title">Engine Settings</div>
          <div class="field">
            <label class="lbl">Account Size</label>
            <div class="acct-pills">
              <button class="acct-pill" id="acct-50"  onclick="selectAcct(50000)">$50 K</button>
              <button class="acct-pill" id="acct-100" onclick="selectAcct(100000)">$100 K</button>
              <button class="acct-pill" id="acct-150" onclick="selectAcct(150000)">$150 K</button>
            </div>
          </div>
          <div class="grid-2" style="gap:8px">
            <div class="field">
              <div class="field-label-row">
                <label class="lbl">Primary Interval</label>
                <i class="info-icon" data-tip="1m is recommended for all intraday strategies. Change only if you have a specific need.">ⓘ</i>
              </div>
              <select id="s-interval"
                title="1m is recommended for all intraday strategies. Change only if you have a specific need.">
                <option value="1m" selected>1m</option>
                <option value="2m">2m</option>
                <option value="5m">5m</option>
                <option value="15m">15m</option>
                <option value="30m">30m</option>
                <option value="60m">60m</option>
                <option value="1h">1h</option>
                <option value="1d">1d</option>
              </select>
            </div>
            <div class="field">
              <div class="field-label-row">
                <label class="lbl">Lookback Period</label>
                <i class="info-icon" data-tip="Auto selects the optimal lookback based on your active strategy. Increase only for long-term strategies (Weekly, Monthly).">ⓘ</i>
              </div>
              <select id="s-period"
                title="Auto selects the optimal lookback based on your active strategy. Increase only for long-term strategies (Weekly, Monthly).">
                <option value="auto" selected>auto</option>
                <option value="1d">1d</option>
                <option value="3d">3d</option>
                <option value="5d">5d</option>
                <option value="7d">7d</option>
                <option value="10d">10d</option>
                <option value="14d">14d</option>
                <option value="30d">30d</option>
              </select>
            </div>
          </div>
          <div class="btn-row">
            <button class="btn btn-primary" onclick="applySettings()">💾 Apply Settings</button>
            <button class="btn btn-neutral" onclick="loadStatus()">↺ Reload</button>
          </div>
          <div id="settings-msg" style="font-size:0.72rem;color:var(--muted);margin-top:7px;min-height:16px"></div>
        </div>

        <!-- Engine actions -->
        <div class="card">
          <div class="card-title">Engine Actions</div>
          <div class="btn-row" style="margin-top:0">
            <button class="btn btn-warning btn-sm" onclick="doAction('refresh')">🔄 Force Refresh</button>
            <button class="btn btn-neutral btn-sm" onclick="doAction('optimize')">🔬 Optimize Now</button>
          </div>
          <div id="action-msg" style="font-size:0.72rem;color:var(--muted);margin-top:8px;min-height:16px"></div>
        </div>

        <!-- Live feed controls -->
        <div class="card">
          <div class="card-title">Massive Live Feed</div>
          <div class="feed-block">
            <div class="feed-title">Feed Status</div>
            <div id="feed-status-row" class="status-row" style="margin-bottom:0">
              <span class="status-key">Connection</span>
              <span id="feed-status-val" class="status-val" style="display:flex;align-items:center;gap:6px">
                <span class="dot dot-gray dot-pulse" id="feed-dot"></span>
                <span id="feed-status-text">checking…</span>
              </span>
            </div>
          </div>
          <div class="btn-row">
            <button class="btn btn-success btn-sm" onclick="feedAction('start')">▶ Start</button>
            <button class="btn btn-danger btn-sm"  onclick="feedAction('stop')">■ Stop</button>
          </div>
          <hr class="sep"/>
          <div style="font-size:0.68rem;color:var(--muted);margin-bottom:6px">Feed quality</div>
          <div class="btn-row" style="margin-top:0">
            <button class="btn btn-neutral btn-sm" onclick="feedAction('upgrade')">⬆ Upgrade (quotes)</button>
            <button class="btn btn-neutral btn-sm" onclick="feedAction('downgrade')">⬇ Downgrade (bars only)</button>
          </div>
          <div id="feed-msg" style="font-size:0.72rem;color:var(--muted);margin-top:8px;min-height:16px"></div>
        </div>
      </div>

      <div>
        <!-- Engine status panel -->
        <div class="card">
          <div class="card-title">
            <span>Engine Status</span>
            <span class="spin" id="status-spin" style="display:none"></span>
          </div>
          <div id="engine-status-panel">
            <div style="color:var(--faint);font-size:0.8rem;text-align:center;padding:20px 0">Loading…</div>
          </div>
        </div>

        <!-- Service links -->
        <div class="card">
          <div class="card-title">Quick Links</div>
          <a class="link-row" href="/" target="_self">
            <span class="link-icon">📊</span>
            <div><div class="link-label">Main Dashboard</div><div class="link-desc">ORB detection, asset focus cards, positions & P&L</div></div>
          </a>
          <a class="link-row" href="/trainer" target="_self">
            <span class="link-icon">🧠</span>
            <div><div class="link-label">CNN Trainer</div><div class="link-desc">Train, validate, and export the CNN breakout model</div></div>
          </a>
          <a class="link-row" href="/docs" target="_blank" rel="noopener">
            <span class="link-icon">📖</span>
            <div><div class="link-label">API Docs (Swagger)</div><div class="link-desc">Interactive REST API documentation</div></div>
          </a>
          <a class="link-row" href="/metrics/prometheus" target="_blank" rel="noopener">
            <span class="link-icon">📈</span>
            <div><div class="link-label">Prometheus Metrics</div><div class="link-desc">Raw metrics endpoint for Prometheus scraping</div></div>
          </a>
          <a class="link-row" href="/health" target="_blank" rel="noopener">
            <span class="link-icon">❤️</span>
            <div><div class="link-label">Health Check</div><div class="link-desc">JSON health status of all services</div></div>
          </a>
        </div>

        <!-- About -->
        <div class="card">
          <div class="card-title">About</div>
          <div class="status-row">
            <span class="status-key">Service</span>
            <span class="status-val">Ruby Futures</span>
          </div>
          <div class="status-row">
            <span class="status-key">Data Service</span>
            <span class="status-val" id="about-data">port 8000</span>
          </div>
          <div class="status-row">
            <span class="status-key">Web Service</span>
            <span class="status-val">port 8080</span>
          </div>
          <div class="status-row">
            <span class="status-key">Trainer Service</span>
            <span class="status-val">port 8200</span>
          </div>
          <div class="status-row" style="margin-bottom:0">
            <span class="status-key">Theme</span>
            <span class="status-val" id="about-theme">dark</span>
          </div>
        </div>
      </div>
    </div>
  </div>

  <!-- ═══════════════════════════════════════════════════════════ -->
  <!-- SERVICES SECTION — connectivity, URLs, Tailscale           -->
  <!-- ═══════════════════════════════════════════════════════════ -->
  <div id="section-services" class="section-content">
    <div class="grid-2">
      <div>
        <!-- Service URLs -->
        <div class="card">
          <div class="card-title">Service URLs</div>
          <div class="field">
            <label class="lbl">Data Service URL</label>
            <input type="text" id="svc-data-url" placeholder="http://100.113.72.63:8000"/>
            <div class="field-hint">Engine + data API (Pi)</div>
          </div>
          <div class="field">
            <label class="lbl">Trainer Service URL</label>
            <input type="text" id="svc-trainer-url" placeholder="http://100.122.184.58:8200"/>
            <div class="field-hint">GPU training server</div>
          </div>

          <div class="btn-row">
            <button class="btn btn-primary btn-sm" onclick="saveServiceUrls()">💾 Save URLs</button>
            <button class="btn btn-neutral btn-sm" onclick="probeAllServices()">🔍 Test Connectivity</button>
          </div>
          <div id="svc-urls-msg" style="font-size:0.72rem;color:var(--muted);margin-top:7px;min-height:16px"></div>
        </div>
      </div>

      <div>
        <!-- Tailscale network status -->
        <div class="card">
          <div class="card-title">
            <span>Network Status</span>
            <button class="btn btn-neutral btn-sm" onclick="probeAllServices()" style="font-size:0.68rem;padding:2px 8px">↺ Refresh</button>
          </div>
          <div id="services-status-panel">
            <div style="color:var(--faint);font-size:0.8rem;text-align:center;padding:20px 0">Click "Test Connectivity" to check…</div>
          </div>
        </div>


      </div>
    </div>
  </div>

  <!-- ═══════════════════════════════════════════════════════════ -->
  <!-- FEATURES SECTION — feature toggles                        -->
  <!-- ═══════════════════════════════════════════════════════════ -->
  <div id="section-features" class="section-content">
    <div class="grid-2">
      <div>
        <div class="card">
          <div class="card-title">Data & Integration</div>

          <div class="toggle-row">
            <div>
              <div class="toggle-label">Live Positions Panel</div>
              <div class="toggle-desc">Show live positions, P&L, and broker status on dashboard. Requires Rithmic broker connection.</div>
            </div>
            <label class="toggle-switch">
              <input type="checkbox" id="feat-live-positions" onchange="saveFeatures()"/>
              <span class="toggle-slider"></span>
            </label>
          </div>

          <div class="toggle-row">
            <div>
              <div class="toggle-label">Kraken Crypto Feed</div>
              <div class="toggle-desc">Enable Kraken WebSocket for live crypto data (BTC, ETH, SOL…)</div>
            </div>
            <label class="toggle-switch">
              <input type="checkbox" id="feat-kraken" onchange="saveFeatures()"/>
              <span class="toggle-slider"></span>
            </label>
          </div>

          <div class="toggle-row">
            <div>
              <div class="toggle-label">Massive Live Feed Auto-Start</div>
              <div class="toggle-desc">Start Massive WebSocket automatically on engine boot</div>
            </div>
            <label class="toggle-switch">
              <input type="checkbox" id="feat-massive-autostart" onchange="saveFeatures()"/>
              <span class="toggle-slider"></span>
            </label>
          </div>

          <div class="toggle-row">
            <div>
              <div class="toggle-label">Grok AI Analyst</div>
              <div class="toggle-desc">Enable Grok AI analysis panel on dashboard</div>
            </div>
            <label class="toggle-switch">
              <input type="checkbox" id="feat-grok" onchange="saveFeatures()"/>
              <span class="toggle-slider"></span>
            </label>
          </div>
        </div>

        <div class="card">
          <div class="card-title">Signal Filtering</div>

          <div class="toggle-row">
            <div>
              <div class="toggle-label">CNN Gate</div>
              <div class="toggle-desc">Require CNN model approval before entry (probability ≥ session threshold)</div>
            </div>
            <label class="toggle-switch">
              <input type="checkbox" id="feat-cnn-gate" onchange="saveFeatures()" checked/>
              <span class="toggle-slider"></span>
            </label>
          </div>

          <div class="toggle-row">
            <div>
              <div class="toggle-label">ORB Filter Gate</div>
              <div class="toggle-desc">Apply ORB quality filters (volume surge, VWAP, ATR ratio, etc.)</div>
            </div>
            <label class="toggle-switch">
              <input type="checkbox" id="feat-orb-filter" onchange="saveFeatures()" checked/>
              <span class="toggle-slider"></span>
            </label>
          </div>

          <div class="toggle-row">
            <div>
              <div class="toggle-label">MTF Alignment Check</div>
              <div class="toggle-desc">Require 15-min EMA/MACD alignment score ≥ threshold for entry</div>
            </div>
            <label class="toggle-switch">
              <input type="checkbox" id="feat-mtf-check" onchange="saveFeatures()" checked/>
              <span class="toggle-slider"></span>
            </label>
          </div>

          <div class="toggle-row">
            <div>
              <div class="toggle-label">SAR (Stop-and-Reverse)</div>
              <div class="toggle-desc">Enable always-in micro position reversals</div>
            </div>
            <label class="toggle-switch">
              <input type="checkbox" id="feat-sar" onchange="saveFeatures()"/>
              <span class="toggle-slider"></span>
            </label>
          </div>
        </div>
      </div>

      <div>
        <div class="card">
          <div class="card-title">Trading Modes</div>

          <div class="toggle-row">
            <div>
              <div class="toggle-label">TPT Mode (Take Profit Trader)</div>
              <div class="toggle-desc">Enable funded account rules: 4 PM ET hard flatten, fixed position sizes</div>
            </div>
            <label class="toggle-switch">
              <input type="checkbox" id="feat-tpt" onchange="saveFeatures()"/>
              <span class="toggle-slider"></span>
            </label>
          </div>

          <div class="toggle-row">
            <div>
              <div class="toggle-label">TP3 Trailing (EMA9)</div>
              <div class="toggle-desc">Enable 3-phase bracket: TP1 → breakeven → EMA9 trail to TP3</div>
            </div>
            <label class="toggle-switch">
              <input type="checkbox" id="feat-tp3-trail" onchange="saveFeatures()" checked/>
              <span class="toggle-slider"></span>
            </label>
          </div>

          <div class="toggle-row">
            <div>
              <div class="toggle-label">Auto Brackets</div>
              <div class="toggle-desc">Automatically submit SL/TP bracket orders with each entry</div>
            </div>
            <label class="toggle-switch">
              <input type="checkbox" id="feat-auto-brackets" onchange="saveFeatures()" checked/>
              <span class="toggle-slider"></span>
            </label>
          </div>

          <div class="toggle-row">
            <div>
              <div class="toggle-label">Debug Logging</div>
              <div class="toggle-desc">Verbose debug output in engine logs</div>
            </div>
            <label class="toggle-switch">
              <input type="checkbox" id="feat-debug-log" onchange="saveFeatures()"/>
              <span class="toggle-slider"></span>
            </label>
          </div>
        </div>

        <div class="card">
          <div class="card-title">Feature Status</div>
          <div id="features-msg" style="font-size:0.72rem;color:var(--muted);min-height:16px;margin-bottom:8px"></div>
          <div style="font-size:0.68rem;color:var(--faint)">
            Toggle changes are saved to Redis immediately. Engine-side toggles
            (CNN gate, ORB filter, TPT mode) are read by the engine on next refresh cycle.
            Live Positions panel is hidden until the Rithmic broker is connected and the toggle is enabled.
          </div>
        </div>
      </div>
    </div>
  </div>

  <!-- ═══════════════════════════════════════════════════════════ -->
  <!-- RISK & TRADING SECTION                                    -->
  <!-- ═══════════════════════════════════════════════════════════ -->
  <div id="section-risk" class="section-content">
    <div class="grid-2">
      <div>
        <div class="card">
          <div class="card-title">Position Sizing</div>
          <div class="field">
            <label class="lbl">Risk % Per Trade</label>
            <input type="number" id="risk-pct" value="0.5" min="0.1" max="5.0" step="0.1"/>
            <div class="field-hint">Percentage of account risked per trade (default: 0.5%)</div>
          </div>
          <div class="field">
            <label class="lbl">Max Contracts Per Trade</label>
            <input type="number" id="risk-max-contracts" value="5" min="1" max="20"/>
            <div class="field-hint">Hard cap on contracts per single entry</div>
          </div>
          <div class="field">
            <label class="lbl">Max Concurrent Positions</label>
            <input type="number" id="risk-max-positions" value="5" min="1" max="10"/>
            <div class="field-hint">Maximum open positions across all instruments</div>
          </div>
        </div>

        <div class="card">
          <div class="card-title">Stop Loss / Take Profit (ATR Multiples)</div>
          <div class="grid-3" style="gap:8px">
            <div class="field">
              <label class="lbl">SL ATR Mult</label>
              <input type="number" id="risk-sl-atr" value="1.5" min="0.5" max="5.0" step="0.1"/>
            </div>
            <div class="field">
              <label class="lbl">TP1 ATR Mult</label>
              <input type="number" id="risk-tp1-atr" value="2.0" min="0.5" max="10.0" step="0.1"/>
            </div>
            <div class="field">
              <label class="lbl">TP2 ATR Mult</label>
              <input type="number" id="risk-tp2-atr" value="3.5" min="0.5" max="10.0" step="0.1"/>
            </div>
          </div>
          <div class="grid-2" style="gap:8px">
            <div class="field">
              <label class="lbl">TP3 ATR Mult (Trail Target)</label>
              <input type="number" id="risk-tp3-atr" value="5.0" min="1.0" max="15.0" step="0.5"/>
            </div>
            <div class="field">
              <label class="lbl">Entry Cooldown (min)</label>
              <input type="number" id="risk-cooldown" value="10" min="0" max="60"/>
            </div>
          </div>
          <div class="field-hint">Per-type TP3 multiples are loaded from feature_contract.json at strategy startup</div>
        </div>

        <div class="card">
          <div class="card-title">Tick-Based Defaults (Fallback)</div>
          <div class="grid-2" style="gap:8px">
            <div class="field">
              <label class="lbl">Default SL Ticks</label>
              <input type="number" id="risk-sl-ticks" value="20" min="5" max="100"/>
            </div>
            <div class="field">
              <label class="lbl">Default TP Ticks</label>
              <input type="number" id="risk-tp-ticks" value="40" min="5" max="200"/>
            </div>
          </div>
          <div class="field-hint">Used when ATR is not yet ready (first few bars of session)</div>
        </div>

        <div class="btn-row">
          <button class="btn btn-primary" onclick="saveRiskSettings()">💾 Save Risk Settings</button>
          <button class="btn btn-neutral" onclick="loadRiskSettings()">↺ Reset to Current</button>
        </div>
        <div id="risk-msg" style="font-size:0.72rem;color:var(--muted);margin-top:7px;min-height:16px"></div>
      </div>

      <div>
        <div class="card">
          <div class="card-title">SAR Parameters</div>
          <div class="grid-2" style="gap:8px">
            <div class="field">
              <label class="lbl">Min CNN Probability</label>
              <input type="number" id="sar-min-cnn" value="0.85" min="0.5" max="1.0" step="0.01"/>
              <div class="field-hint">Minimum CNN prob for SAR reversal</div>
            </div>
            <div class="field">
              <label class="lbl">Min MTF Score</label>
              <input type="number" id="sar-min-mtf" value="0.60" min="0.0" max="1.0" step="0.05"/>
              <div class="field-hint">Minimum 15m alignment score</div>
            </div>
          </div>
          <div class="grid-2" style="gap:8px">
            <div class="field">
              <label class="lbl">Cooldown (min)</label>
              <input type="number" id="sar-cooldown" value="30" min="0" max="120"/>
            </div>
            <div class="field">
              <label class="lbl">Chase Max ATR Frac</label>
              <input type="number" id="sar-chase-atr" value="0.50" min="0.1" max="2.0" step="0.05"/>
            </div>
          </div>
          <div class="grid-2" style="gap:8px">
            <div class="field">
              <label class="lbl">Winning CNN Prob</label>
              <input type="number" id="sar-win-cnn" value="0.92" min="0.5" max="1.0" step="0.01"/>
              <div class="field-hint">CNN threshold while position is winning</div>
            </div>
            <div class="field">
              <label class="lbl">High Winner R-Mult</label>
              <input type="number" id="sar-hwinner-r" value="1.0" min="0.0" max="5.0" step="0.1"/>
            </div>
          </div>
        </div>

        <div class="card">
          <div class="card-title">CNN Filter Settings</div>
          <div class="grid-2" style="gap:8px">
            <div class="field">
              <label class="lbl">Threshold Override</label>
              <input type="number" id="cnn-threshold" value="0" min="0" max="1.0" step="0.01"/>
              <div class="field-hint">0 = use session thresholds from feature_contract.json</div>
            </div>
            <div class="field">
              <label class="lbl">Session Key</label>
              <select id="cnn-session-key">
                <option value="auto">auto (detect from time)</option>
                <option value="us" selected>us</option>
                <option value="london">london</option>
                <option value="london_ny">london_ny</option>
                <option value="frankfurt">frankfurt</option>
                <option value="tokyo">tokyo</option>
                <option value="sydney">sydney</option>
                <option value="shanghai">shanghai</option>
                <option value="cme">cme</option>
                <option value="cme_settle">cme_settle</option>
              </select>
            </div>
          </div>
          <div class="field">
            <label class="lbl">CNN Lookback Bars</label>
            <input type="number" id="cnn-lookback" value="60" min="20" max="200"/>
            <div class="field-hint">Number of bars in the chart snapshot fed to CNN</div>
          </div>
        </div>

        <div class="card">
          <div class="card-title">ORB Quality Filters</div>
          <div class="grid-2" style="gap:8px">
            <div class="field">
              <label class="lbl">Volume Surge Multiplier</label>
              <input type="number" id="orb-vol-surge" value="1.5" min="1.0" max="5.0" step="0.1"/>
            </div>
            <div class="field">
              <label class="lbl">Volume Average Period</label>
              <input type="number" id="orb-vol-period" value="20" min="5" max="50"/>
            </div>
          </div>
          <div class="grid-2" style="gap:8px">
            <div class="field">
              <label class="lbl">Min ORB/ATR Ratio</label>
              <input type="number" id="orb-min-atr" value="0.3" min="0.1" max="2.0" step="0.05"/>
            </div>
            <div class="field">
              <label class="lbl">ORB Minutes</label>
              <input type="number" id="orb-minutes" value="30" min="5" max="60"/>
            </div>
          </div>
          <div class="toggle-row" style="margin-top:8px">
            <div>
              <div class="toggle-label">Require VWAP Confirmation</div>
              <div class="toggle-desc">Price must be above/below VWAP for long/short entries</div>
            </div>
            <label class="toggle-switch">
              <input type="checkbox" id="orb-require-vwap" checked onchange="saveRiskSettings()"/>
              <span class="toggle-slider"></span>
            </label>
          </div>
        </div>
      </div>
    </div>
  </div>

  <!-- ═══════════════════════════════════════════════════════════ -->
  <!-- PROP ACCOUNTS SECTION — Rithmic read-only account monitor -->
  <!-- ═══════════════════════════════════════════════════════════ -->
  <div id="section-propaccounts" class="section-content">
    <div class="grid-2">
      <div>
        <div class="card">
          <div class="card-title">
            <span>🏦 Rithmic Prop Accounts</span>
            <button class="btn btn-neutral btn-sm"
                    onclick="loadRithmicPanel()"
                    style="font-size:0.68rem;padding:2px 8px">↺ Refresh</button>
          </div>
          <!-- Panel is loaded/reloaded via HTMX when the tab is shown -->
          <div id="rithmic-settings-panel">
            <div style="color:var(--faint);font-size:0.8rem;text-align:center;padding:20px 0">
              Loading account config…
            </div>
          </div>
        </div>
      </div>

      <div>
        <div class="card">
          <div class="card-title">ℹ️ How Rithmic Monitoring Works</div>
          <div style="font-size:0.72rem;color:var(--muted);line-height:1.7">
            <div style="margin-bottom:6px">
              Uses <code style="background:var(--bg-input);padding:1px 5px;border-radius:3px">async-rithmic</code>
              to open a short-lived, read-only session to each prop firm account.
              No orders are placed — only balances, positions, and P&amp;L are read.
            </div>
            <div style="margin-bottom:6px">
              Your Rithmic credentials are <strong>AES-256 encrypted</strong>
              before storage and are never returned to the browser.
              The UI only shows whether credentials are configured.
            </div>
            <div style="margin-bottom:6px">
              <strong>Supported firms:</strong><br/>
              Take Profit Trader (TPT) · Apex Trader Funding ·
              TopStep · TradeDay · My Funded Futures · Custom
            </div>
            <div style="margin-bottom:6px">
              <strong>One-time setup per account:</strong><br/>
              Accept market data agreements at
              <a href="https://rtraderpro.rithmic.com" target="_blank"
                 style="color:#818cf8">rtraderpro.rithmic.com</a>
              before the first Python connection.
            </div>
            <div>
              <strong>Multiple accounts:</strong> Add up to 5 accounts.
              Each account gets its own connection slot. Copy-trade
              support (Account 1 → Account 2) will be added in a
              future release once 2 funded accounts are active.
            </div>
          </div>
        </div>

        <div class="card">
          <div class="card-title">📦 Dependency Check</div>
          <div id="rithmic-dep-check">
            <div style="color:var(--faint);font-size:0.8rem;text-align:center;padding:8px 0">
              Checking…
            </div>
          </div>
          <div class="btn-row">
            <button class="btn btn-neutral btn-sm"
                    onclick="checkRithmicDeps()">↺ Check</button>
          </div>
        </div>

        <div class="card">
          <div class="card-title">🔌 Live Account Status</div>
          <div id="rithmic-live-status">
            <div style="color:var(--faint);font-size:0.8rem;text-align:center;padding:8px 0">
              Click "Refresh All" on the left to poll accounts.
            </div>
          </div>
          <div class="btn-row" style="margin-top:8px">
            <button class="btn btn-neutral btn-sm"
                    onclick="refreshAllRithmicSettings()">🔄 Refresh All Accounts</button>
          </div>
          <div id="rithmic-refresh-msg"
               style="font-size:0.72rem;color:var(--muted);margin-top:6px;min-height:14px"></div>
        </div>
      </div>
    </div>
  </div>

  <!-- ═══════════════════════════════════════════════════════════ -->
  <!-- API KEYS SECTION                                          -->
  <!-- ═══════════════════════════════════════════════════════════ -->
  <div id="section-keys" class="section-content">
    <div class="grid-2">
      <div>
        <div class="card">
          <div class="card-title">API Key Status</div>
          <div id="api-keys-panel">
            <div style="color:var(--faint);font-size:0.8rem;text-align:center;padding:20px 0">Loading…</div>
          </div>
          <div class="btn-row">
            <button class="btn btn-neutral btn-sm" onclick="loadApiKeyStatus()">↺ Refresh Status</button>
          </div>
        </div>

        <div class="card">
          <div class="card-title">⚠️ Security Note</div>
          <div style="font-size:0.72rem;color:var(--muted);line-height:1.5">
            API keys are stored as environment variables on each service host and injected via
            <code style="background:var(--bg-input);padding:1px 5px;border-radius:3px">docker-compose.yml</code>.
            They are <strong>never</strong> displayed in the UI or sent to the browser.
            This panel only shows whether each key is configured (non-empty) — use
            <code style="background:var(--bg-input);padding:1px 5px;border-radius:3px">ssh</code> to rotate keys on the host.
          </div>
        </div>
      </div>

      <div>
        <div class="card">
          <div class="card-title">Environment Variables</div>
          <div style="font-size:0.72rem;color:var(--muted);line-height:1.6">
            <div class="status-row">
              <span class="status-key">MASSIVE_API_KEY</span>
              <span class="status-val" style="font-size:0.68rem">Futures bar data (primary)</span>
            </div>
            <div class="status-row">
              <span class="status-key">KRAKEN_API_KEY</span>
              <span class="status-val" style="font-size:0.68rem">Crypto REST + WS (optional for public)</span>
            </div>
            <div class="status-row">
              <span class="status-key">KRAKEN_API_SECRET</span>
              <span class="status-val" style="font-size:0.68rem">Kraken private endpoints</span>
            </div>
            <div class="status-row">
              <span class="status-key">XAI_API_KEY</span>
              <span class="status-val" style="font-size:0.68rem">xAI Grok AI analyst</span>
            </div>
            <div class="status-row">
              <span class="status-key">DISCORD_WEBHOOK_URL</span>
              <span class="status-val" style="font-size:0.68rem">CI/CD deploy notifications</span>
            </div>
            <div class="status-row" style="margin-bottom:0">
              <span class="status-key">DATABASE_URL</span>
              <span class="status-val" style="font-size:0.68rem">Historical bar storage (Postgres)</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>

</div><!-- /page -->

<!-- Toast -->
<div id="toast"></div>

<script>
'use strict';

// ═══════════════════════════════════════════════════════
// Section tabs
// ═══════════════════════════════════════════════════════
function showSection(name) {
  document.querySelectorAll('.section-content').forEach(el => el.classList.remove('active'));
  document.querySelectorAll('.section-tab').forEach(el => el.classList.remove('active'));
  const sec = document.getElementById('section-' + name);
  if (sec) sec.classList.add('active');
  // Find the tab button that matches
  document.querySelectorAll('.section-tab').forEach(el => {
    if (el.textContent.toLowerCase().includes(name) ||
        (name === 'engine' && el.textContent.includes('Engine')) ||
        (name === 'services' && el.textContent.includes('Services')) ||
        (name === 'features' && el.textContent.includes('Features')) ||
        (name === 'risk' && el.textContent.includes('Risk')) ||
        (name === 'propaccounts' && el.textContent.includes('Prop')) ||
        (name === 'keys' && el.textContent.includes('Keys')))
      el.classList.add('active');
  });
  localStorage.setItem('settingsTab', name);
  // Load section-specific data
  if (name === 'services') { loadServiceUrls(); }
  if (name === 'features') loadFeatures();
  if (name === 'risk') loadRiskSettings();
  if (name === 'keys') loadApiKeyStatus();
  if (name === 'propaccounts') { loadRithmicPanel(); checkRithmicDeps(); loadRithmicLiveStatus(); }
}

// ═══════════════════════════════════════════════════════
// Theme
// ═══════════════════════════════════════════════════════
function toggleTheme() {
  const isDark = document.documentElement.classList.toggle('dark');
  localStorage.setItem('theme', isDark ? 'dark' : 'light');
  const el = document.getElementById('about-theme');
  if (el) el.textContent = isDark ? 'dark' : 'light';
}

// ═══════════════════════════════════════════════════════
// Toast helper
// ═══════════════════════════════════════════════════════
let _toastTimer = null;
function toast(msg, isErr) {
  const el = document.getElementById('toast');
  el.textContent = msg;
  el.className = 'show ' + (isErr ? 'err' : 'ok');
  if (_toastTimer) clearTimeout(_toastTimer);
  _toastTimer = setTimeout(() => { el.className = ''; }, 3500);
}

// ═══════════════════════════════════════════════════════
// Load engine status
// ═══════════════════════════════════════════════════════
let _selectedAcct = null;

async function loadStatus() {
  const spin = document.getElementById('status-spin');
  if (spin) spin.style.display = 'inline-block';
  try {
    const r = await fetch('/analysis/status', { signal: AbortSignal.timeout(6000) });
    if (!r.ok) throw new Error('HTTP ' + r.status);
    const d = await r.json();
    renderStatus(d);
  } catch (e) {
    document.getElementById('engine-status-panel').innerHTML =
      '<div style="color:#f87171;font-size:0.78rem;padding:12px 0">Failed to load engine status: ' + e.message + '</div>';
    document.getElementById('engine-badge').className = 'badge b-err';
    document.getElementById('engine-badge').textContent = 'error';
  } finally {
    if (spin) spin.style.display = 'none';
  }
}

function _statusRow(key, val) {
  return '<div class="status-row"><span class="status-key">' + key +
    '</span><span class="status-val">' + val + '</span></div>';
}

function renderStatus(d) {
  const badge = document.getElementById('engine-badge');
  if (d.running) {
    badge.className = 'badge b-ok'; badge.textContent = 'running';
  } else {
    badge.className = 'badge b-warn'; badge.textContent = 'idle';
  }

  const acct = d.account_size || (d.settings && d.settings.account_size);
  if (acct && !_selectedAcct) selectAcct(acct, true);

  const intv = d.interval || (d.settings && d.settings.interval);
  if (intv) {
    const sel = document.getElementById('s-interval');
    if (sel) { for (let o of sel.options) { if (o.value === intv) o.selected = true; } }
  }

  const per = d.period || (d.settings && d.settings.period);
  if (per) {
    const sel = document.getElementById('s-period');
    if (sel) {
      let matched = false;
      for (let o of sel.options) { if (o.value === per) { o.selected = true; matched = true; } }
      // If the engine returns a value not in the list, fall back to "auto"
      if (!matched) { for (let o of sel.options) { if (o.value === 'auto') o.selected = true; } }
    }
  }

  const feed = d.live_feed || {};
  const feedRunning = feed.running || feed.connected || false;
  const feedDot = document.getElementById('feed-dot');
  const feedTxt = document.getElementById('feed-status-text');
  if (feedRunning) {
    feedDot.className = 'dot dot-green';
    feedTxt.textContent = feed.mode ? 'running (' + feed.mode + ')' : 'running';
  } else {
    feedDot.className = 'dot dot-gray';
    feedTxt.textContent = 'stopped';
  }

  let rows = '';
  if (d.last_refresh) rows += _statusRow('Last Refresh', _fmtTime(d.last_refresh));
  if (d.next_refresh) rows += _statusRow('Next Refresh', _fmtTime(d.next_refresh));
  if (d.refresh_interval_minutes) rows += _statusRow('Interval', d.refresh_interval_minutes + ' min');
  if (intv)  rows += _statusRow('Bar Interval', intv);
  if (per)   rows += _statusRow('Lookback', per);
  if (acct)  rows += _statusRow('Account', '$' + Number(acct).toLocaleString());
  if (d.data_source) rows += _statusRow('Data Source', d.data_source);
  if (d.assets_loaded !== undefined) rows += _statusRow('Assets Loaded', d.assets_loaded);

  if (feed.running !== undefined) {
    rows += _statusRow('Live Feed', feedRunning
      ? '<span style="color:#4ade80">● running</span>'
      : '<span style="color:#6b7280">● stopped</span>');
  }

  if (!rows) rows = '<div style="color:var(--faint);font-size:0.78rem;padding:8px 0">No status available</div>';

  document.getElementById('engine-status-panel').innerHTML = rows;
}

function _fmtTime(iso) {
  if (!iso) return '—';
  try {
    return new Date(iso).toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit', second: '2-digit' });
  } catch { return iso; }
}

// ═══════════════════════════════════════════════════════
// Account size selector
// ═══════════════════════════════════════════════════════
function selectAcct(size, silent) {
  _selectedAcct = size;
  document.querySelectorAll('.acct-pill').forEach(p => p.classList.remove('selected'));
  const el = document.getElementById('acct-' + (size / 1000));
  if (el) el.classList.add('selected');
  if (!silent) document.getElementById('settings-msg').textContent = '';
}

// ═══════════════════════════════════════════════════════
// Apply engine settings
// ═══════════════════════════════════════════════════════
async function applySettings() {
  const body = {};
  if (_selectedAcct) body.account_size = _selectedAcct;
  const intv = document.getElementById('s-interval').value;
  const per  = document.getElementById('s-period').value;
  if (intv) body.interval = intv;
  if (per)  body.period   = per;

  if (!Object.keys(body).length) {
    document.getElementById('settings-msg').textContent = 'Nothing to update.';
    return;
  }

  const msg = document.getElementById('settings-msg');
  msg.textContent = 'Applying…';
  try {
    const r = await fetch('/actions/update_settings', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
      signal: AbortSignal.timeout(8000),
    });
    const d = await r.json();
    if (r.ok) {
      msg.style.color = '#4ade80';
      msg.textContent = '✓ ' + (d.message || 'Settings updated');
      toast('Settings applied', false);
      setTimeout(loadStatus, 600);
    } else {
      msg.style.color = '#f87171';
      msg.textContent = '✗ ' + (d.detail || 'Failed');
      toast('Failed: ' + (d.detail || r.status), true);
    }
  } catch (e) {
    msg.style.color = '#f87171';
    msg.textContent = '✗ ' + e.message;
    toast('Error: ' + e.message, true);
  }
}

// ═══════════════════════════════════════════════════════
// Engine actions (refresh / optimize)
// ═══════════════════════════════════════════════════════
async function doAction(type) {
  const urls = { refresh: '/actions/force_refresh', optimize: '/actions/optimize_now' };
  const url = urls[type];
  if (!url) return;

  const msg = document.getElementById('action-msg');
  msg.style.color = 'var(--muted)';
  msg.textContent = (type === 'refresh' ? 'Triggering refresh…' : 'Triggering optimization…');

  try {
    const r = await fetch(url, { method: 'POST', signal: AbortSignal.timeout(10000) });
    const d = await r.json();
    if (r.ok) {
      msg.style.color = '#4ade80';
      msg.textContent = '✓ ' + (d.message || d.status || 'Done');
      toast(d.message || 'Done', false);
    } else {
      msg.style.color = '#f87171';
      msg.textContent = '✗ ' + (d.detail || 'Failed');
      toast('Failed: ' + (d.detail || r.status), true);
    }
  } catch (e) {
    msg.style.color = '#f87171';
    msg.textContent = '✗ ' + e.message;
    toast('Error: ' + e.message, true);
  }
}

// ═══════════════════════════════════════════════════════
// Live feed controls
// ═══════════════════════════════════════════════════════
async function feedAction(action) {
  const urls = {
    start:     '/actions/live_feed/start',
    stop:      '/actions/live_feed/stop',
    upgrade:   '/actions/live_feed/upgrade',
    downgrade: '/actions/live_feed/downgrade',
  };
  const url = urls[action];
  if (!url) return;

  const msg = document.getElementById('feed-msg');
  msg.style.color = 'var(--muted)';
  msg.textContent = action.charAt(0).toUpperCase() + action.slice(1) + 'ing…';

  try {
    const r = await fetch(url, { method: 'POST', signal: AbortSignal.timeout(10000) });
    const d = await r.json();
    if (r.ok) {
      msg.style.color = '#4ade80';
      msg.textContent = '✓ ' + (d.message || d.status || 'Done');
      toast(d.message || 'Done', false);
      setTimeout(loadStatus, 800);
    } else {
      msg.style.color = '#f87171';
      msg.textContent = '✗ ' + (d.detail || 'Failed');
      toast('Failed: ' + (d.detail || r.status), true);
    }
  } catch (e) {
    msg.style.color = '#f87171';
    msg.textContent = '✗ ' + e.message;
    toast('Error: ' + e.message, true);
  }
}

// ═══════════════════════════════════════════════════════
// Service URLs — load / save / probe
// ═══════════════════════════════════════════════════════
async function loadServiceUrls() {
  try {
    const r = await fetch('/settings/services/config', { signal: AbortSignal.timeout(5000) });
    if (!r.ok) return;
    const d = await r.json();
    if (d.data_service_url) document.getElementById('svc-data-url').value = d.data_service_url;
    if (d.trainer_service_url) document.getElementById('svc-trainer-url').value = d.trainer_service_url;

  } catch {}
}

async function saveServiceUrls() {
  const body = {
    data_service_url: document.getElementById('svc-data-url').value.trim(),
    trainer_service_url: document.getElementById('svc-trainer-url').value.trim(),

  };
  const msg = document.getElementById('svc-urls-msg');
  msg.style.color = 'var(--muted)';
  msg.textContent = 'Saving…';
  try {
    const r = await fetch('/settings/services/update', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
      signal: AbortSignal.timeout(5000),
    });
    const d = await r.json();
    if (r.ok) {
      msg.style.color = '#4ade80';
      msg.textContent = '✓ ' + (d.message || 'Saved');
      toast('Service URLs saved', false);
    } else {
      msg.style.color = '#f87171';
      msg.textContent = '✗ ' + (d.detail || 'Failed');
    }
  } catch (e) {
    msg.style.color = '#f87171';
    msg.textContent = '✗ ' + e.message;
  }
}

async function probeAllServices() {
  const panel = document.getElementById('services-status-panel');
  panel.innerHTML = '<div style="text-align:center;padding:12px 0"><span class="spin"></span> Probing services…</div>';
  try {
    const r = await fetch('/settings/services/probe', { signal: AbortSignal.timeout(20000) });
    if (!r.ok) throw new Error('HTTP ' + r.status);
    const d = await r.json();
    let html = '';
    for (const svc of (d.services || [])) {
      const dot = svc.ok ? 'dot-green' : 'dot-red';
      const latency = svc.latency_ms > 0 ? svc.latency_ms + 'ms' : '—';
      const errHtml = svc.error ? '<div class="svc-detail" style="color:#f87171">' + svc.error + '</div>' : '';
      // Show engine state badge or latency badge
      let statusBadge = '<span style="font-size:0.7rem;color:var(--muted)">' + latency + '</span>';
      if (svc.detail) {
        const detailColor = svc.ok ? '#4ade80' : '#fbbf24';
        statusBadge = '<span style="font-size:0.68rem;color:' + detailColor + ';font-weight:600">' + svc.detail + '</span>' +
                      '<span style="font-size:0.68rem;color:var(--muted);margin-left:4px">' + latency + '</span>';
      }
      html += '<div class="svc-card">' +
        '<div class="svc-header">' +
          '<span class="svc-name">' + svc.name + '</span>' +
          '<span style="display:flex;align-items:center;gap:6px"><span class="dot ' + dot + '"></span>' +
          statusBadge + '</span>' +
        '</div>' +
        '<div class="svc-url">' + svc.url + '</div>' +
        errHtml +
      '</div>';
    }
    if (!html) html = '<div style="color:var(--faint);font-size:0.78rem">No services to probe</div>';
    panel.innerHTML = html;
  } catch (e) {
    panel.innerHTML = '<div style="color:#f87171;font-size:0.78rem">Probe failed: ' + e.message + '</div>';
  }
}



// ═══════════════════════════════════════════════════════
// Feature toggles — load / save
// ═══════════════════════════════════════════════════════
async function loadFeatures() {
  try {
    const r = await fetch('/settings/features/config', { signal: AbortSignal.timeout(5000) });
    if (!r.ok) return;
    const d = await r.json();
    const fields = {
      'feat-live-positions': 'enable_live_positions',
      'feat-kraken': 'enable_kraken_crypto',
      'feat-massive-autostart': 'massive_autostart',
      'feat-grok': 'enable_grok',
      'feat-cnn-gate': 'orb_cnn_gate',
      'feat-orb-filter': 'orb_filter_gate',
      'feat-mtf-check': 'mtf_alignment_check',
      'feat-sar': 'enable_sar',
      'feat-tpt': 'tpt_mode',
      'feat-tp3-trail': 'enable_tp3_trailing',
      'feat-auto-brackets': 'enable_auto_brackets',
      'feat-debug-log': 'enable_debug_logging',
    };
    for (const [elId, key] of Object.entries(fields)) {
      const el = document.getElementById(elId);
      if (el && d[key] !== undefined) el.checked = !!d[key];
    }
  } catch {}
}

async function saveFeatures() {
  const body = {
    enable_live_positions: document.getElementById('feat-live-positions').checked,
    enable_kraken_crypto: document.getElementById('feat-kraken').checked,
    massive_autostart: document.getElementById('feat-massive-autostart').checked,
    enable_grok: document.getElementById('feat-grok').checked,
    orb_cnn_gate: document.getElementById('feat-cnn-gate').checked,
    orb_filter_gate: document.getElementById('feat-orb-filter').checked,
    mtf_alignment_check: document.getElementById('feat-mtf-check').checked,
    enable_sar: document.getElementById('feat-sar').checked,
    tpt_mode: document.getElementById('feat-tpt').checked,
    enable_tp3_trailing: document.getElementById('feat-tp3-trail').checked,
    enable_auto_brackets: document.getElementById('feat-auto-brackets').checked,
    enable_debug_logging: document.getElementById('feat-debug-log').checked,
  };
  const msg = document.getElementById('features-msg');
  msg.style.color = 'var(--muted)';
  msg.textContent = 'Saving…';
  try {
    const r = await fetch('/settings/features/update', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
      signal: AbortSignal.timeout(5000),
    });
    const d = await r.json();
    if (r.ok) {
      msg.style.color = '#4ade80';
      msg.textContent = '✓ Features saved';
      toast('Features saved', false);
    } else {
      msg.style.color = '#f87171';
      msg.textContent = '✗ ' + (d.detail || 'Failed');
    }
  } catch (e) {
    msg.style.color = '#f87171';
    msg.textContent = '✗ ' + e.message;
  }
}

// ═══════════════════════════════════════════════════════
// Risk settings — load / save
// ═══════════════════════════════════════════════════════
async function loadRiskSettings() {
  try {
    const r = await fetch('/settings/risk/config', { signal: AbortSignal.timeout(5000) });
    if (!r.ok) return;
    const d = await r.json();
    const numFields = {
      'risk-pct': 'risk_percent_per_trade',
      'risk-max-contracts': 'max_contracts',
      'risk-max-positions': 'max_concurrent_positions',
      'risk-sl-atr': 'sl_atr_mult',
      'risk-tp1-atr': 'tp1_atr_mult',
      'risk-tp2-atr': 'tp2_atr_mult',
      'risk-tp3-atr': 'tp3_atr_mult',
      'risk-cooldown': 'entry_cooldown_minutes',
      'risk-sl-ticks': 'default_sl_ticks',
      'risk-tp-ticks': 'default_tp_ticks',
      'sar-min-cnn': 'sar_min_cnn_prob',
      'sar-min-mtf': 'sar_min_mtf_score',
      'sar-cooldown': 'sar_cooldown_minutes',
      'sar-chase-atr': 'sar_chase_max_atr_fraction',
      'sar-win-cnn': 'sar_winning_cnn_prob',
      'sar-hwinner-r': 'sar_high_winner_r_mult',
      'cnn-threshold': 'cnn_threshold_override',
      'cnn-lookback': 'cnn_lookback_bars',
      'orb-vol-surge': 'volume_surge_mult',
      'orb-vol-period': 'volume_avg_period',
      'orb-min-atr': 'min_orb_atr_ratio',
      'orb-minutes': 'orb_minutes',
    };
    for (const [elId, key] of Object.entries(numFields)) {
      const el = document.getElementById(elId);
      if (el && d[key] !== undefined) el.value = d[key];
    }
    if (d.cnn_session_key) {
      const sel = document.getElementById('cnn-session-key');
      if (sel) { for (let o of sel.options) { if (o.value === d.cnn_session_key) o.selected = true; } }
    }
    const vwapEl = document.getElementById('orb-require-vwap');
    if (vwapEl && d.require_vwap !== undefined) vwapEl.checked = d.require_vwap;
  } catch {}
}

async function saveRiskSettings() {
  const body = {
    risk_percent_per_trade: parseFloat(document.getElementById('risk-pct').value) || 0.5,
    max_contracts: parseInt(document.getElementById('risk-max-contracts').value) || 5,
    max_concurrent_positions: parseInt(document.getElementById('risk-max-positions').value) || 5,
    sl_atr_mult: parseFloat(document.getElementById('risk-sl-atr').value) || 1.5,
    tp1_atr_mult: parseFloat(document.getElementById('risk-tp1-atr').value) || 2.0,
    tp2_atr_mult: parseFloat(document.getElementById('risk-tp2-atr').value) || 3.5,
    tp3_atr_mult: parseFloat(document.getElementById('risk-tp3-atr').value) || 5.0,
    entry_cooldown_minutes: parseInt(document.getElementById('risk-cooldown').value) || 10,
    default_sl_ticks: parseInt(document.getElementById('risk-sl-ticks').value) || 20,
    default_tp_ticks: parseInt(document.getElementById('risk-tp-ticks').value) || 40,
    sar_min_cnn_prob: parseFloat(document.getElementById('sar-min-cnn').value) || 0.85,
    sar_min_mtf_score: parseFloat(document.getElementById('sar-min-mtf').value) || 0.60,
    sar_cooldown_minutes: parseInt(document.getElementById('sar-cooldown').value) || 30,
    sar_chase_max_atr_fraction: parseFloat(document.getElementById('sar-chase-atr').value) || 0.50,
    sar_winning_cnn_prob: parseFloat(document.getElementById('sar-win-cnn').value) || 0.92,
    sar_high_winner_r_mult: parseFloat(document.getElementById('sar-hwinner-r').value) || 1.0,
    cnn_threshold_override: parseFloat(document.getElementById('cnn-threshold').value) || 0,
    cnn_session_key: document.getElementById('cnn-session-key').value,
    cnn_lookback_bars: parseInt(document.getElementById('cnn-lookback').value) || 60,
    volume_surge_mult: parseFloat(document.getElementById('orb-vol-surge').value) || 1.5,
    volume_avg_period: parseInt(document.getElementById('orb-vol-period').value) || 20,
    min_orb_atr_ratio: parseFloat(document.getElementById('orb-min-atr').value) || 0.3,
    orb_minutes: parseInt(document.getElementById('orb-minutes').value) || 30,
    require_vwap: document.getElementById('orb-require-vwap').checked,
  };
  const msg = document.getElementById('risk-msg');
  msg.style.color = 'var(--muted)';
  msg.textContent = 'Saving…';
  try {
    const r = await fetch('/settings/risk/update', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
      signal: AbortSignal.timeout(5000),
    });
    const d = await r.json();
    if (r.ok) {
      msg.style.color = '#4ade80';
      msg.textContent = '✓ Risk settings saved';
      toast('Risk settings saved', false);
    } else {
      msg.style.color = '#f87171';
      msg.textContent = '✗ ' + (d.detail || 'Failed');
    }
  } catch (e) {
    msg.style.color = '#f87171';
    msg.textContent = '✗ ' + e.message;
  }
}

// ═══════════════════════════════════════════════════════
// API key status
// ═══════════════════════════════════════════════════════
async function loadApiKeyStatus() {
  try {
    const r = await fetch('/settings/keys/status', { signal: AbortSignal.timeout(5000) });
    if (!r.ok) throw new Error('HTTP ' + r.status);
    const d = await r.json();
    const panel = document.getElementById('api-keys-panel');
    let html = '';
    for (const key of (d.keys || [])) {
      const badgeClass = key.configured ? 'key-set' : 'key-missing';
      const badgeText = key.configured ? '● SET' : '○ MISSING';
      html += '<div class="key-status">' +
        '<span class="key-name">' + key.name + '</span>' +
        '<span class="key-badge ' + badgeClass + '">' + badgeText + '</span>' +
      '</div>';
    }
    if (!html) html = '<div style="color:var(--faint);font-size:0.78rem">No key status available</div>';
    panel.innerHTML = html;
  } catch (e) {
    document.getElementById('api-keys-panel').innerHTML =
      '<div style="color:#f87171;font-size:0.78rem">Failed to load key status: ' + e.message + '</div>';
  }
}

// ═══════════════════════════════════════════════════════
// Boot
// ═══════════════════════════════════════════════════════
document.addEventListener('DOMContentLoaded', () => {
  loadStatus();
  setInterval(loadStatus, 15000);

  const isDark = document.documentElement.classList.contains('dark');
  const themeEl = document.getElementById('about-theme');
  if (themeEl) themeEl.textContent = isDark ? 'dark' : 'light';

  const port = location.port || (location.protocol === 'https:' ? '443' : '80');
  const dataEl = document.getElementById('about-data');
  if (dataEl) dataEl.textContent = 'port ' + port;

  // Restore last active tab
  const savedTab = localStorage.getItem('settingsTab');
  if (savedTab && savedTab !== 'engine') showSection(savedTab);

  // If URL hash points to rithmic section, open it
  if (window.location.hash === '#rithmic') showSection('propaccounts');
});

// ═══════════════════════════════════════════════════════
// Prop Accounts (Rithmic)
// ═══════════════════════════════════════════════════════

// Defensive stubs — these are overwritten when the Rithmic panel fragment loads
window.addRithmicAccount = window.addRithmicAccount || function() { loadRithmicPanel(); };
window.saveRithmicAccount = window.saveRithmicAccount || function() { loadRithmicPanel(); };
window.removeRithmicAccount = window.removeRithmicAccount || function() {};
window.applyPropFirmPreset = window.applyPropFirmPreset || function() {};

function loadRithmicPanel() {
  const panel = document.getElementById('rithmic-settings-panel');
  if (!panel) return;
  panel.innerHTML = '<div style="color:var(--faint);font-size:0.8rem;text-align:center;padding:16px 0">Loading…</div>';
  fetch('/settings/rithmic/panel')
    .then(r => r.text())
    .then(html => {
      panel.innerHTML = html;
      // innerHTML doesn't execute <script> tags — extract and eval them
      panel.querySelectorAll('script').forEach(old => {
        const s = document.createElement('script');
        if (old.src) { s.src = old.src; }
        else { s.textContent = old.textContent; }
        old.parentNode.replaceChild(s, old);
      });
    })
    .catch(() => { panel.innerHTML = '<div style="color:#f87171;font-size:0.8rem;text-align:center;padding:8px">Failed to load account config.</div>'; });
}

function checkRithmicDeps() {
  const el = document.getElementById('rithmic-dep-check');
  if (!el) return;
  el.innerHTML = '<div style="color:var(--faint);font-size:0.8rem;text-align:center;padding:4px 0">Checking…</div>';
  fetch('/api/rithmic/deps')
    .then(r => r.json())
    .then(d => {
      const rows = Object.entries(d.packages || {}).map(([pkg, info]) => {
        const ok = info.installed;
        const dot = `<span style="display:inline-block;width:7px;height:7px;border-radius:50%;background:${ok ? '#22c55e' : '#ef4444'};margin-right:5px"></span>`;
        return `<div class="status-row">${dot}<span class="status-key">${pkg}</span><span class="status-val" style="font-size:0.68rem">${ok ? (info.version || 'installed') : 'not installed'}</span></div>`;
      }).join('');
      el.innerHTML = rows || '<div style="color:var(--faint);font-size:0.8rem">No package info.</div>';
    })
    .catch(() => { el.innerHTML = '<div style="color:#f87171;font-size:0.8rem">Dep check failed.</div>'; });
}

function loadRithmicLiveStatus() {
  const el = document.getElementById('rithmic-live-status');
  if (!el) return;
  fetch('/api/rithmic/status')
    .then(r => r.json())
    .then(d => {
      const entries = Object.entries(d.status || {});
      if (!entries.length) {
        el.innerHTML = '<div style="color:var(--faint);font-size:0.8rem;text-align:center;padding:8px">No status data yet.</div>';
        return;
      }
      const rows = entries.map(([key, st]) => {
        const ok = st.connected;
        const dot = `<span style="display:inline-block;width:7px;height:7px;border-radius:50%;background:${ok ? '#22c55e' : '#ef4444'};margin-right:5px;flex-shrink:0"></span>`;
        const pnl = st.pnl || {};
        const unreal = pnl.unrealized != null ? `$${(+pnl.unrealized).toFixed(2)}` : 'n/a';
        const posCount = (st.positions || []).length;
        return `<div style="display:flex;align-items:center;gap:4px;padding:4px 0;border-bottom:1px solid var(--border-s)">
          ${dot}
          <div style="flex:1;min-width:0">
            <span style="font-size:0.78rem;font-weight:600">${st.label || key}</span>
            <span style="font-size:0.68rem;color:var(--muted);margin-left:6px">${st.prop_firm_label || ''}</span>
            ${ok ? `<div style="font-size:0.68rem;color:var(--muted)">Unreal: ${unreal} · ${posCount} pos</div>` : `<div style="font-size:0.68rem;color:#f87171">${st.error || 'disconnected'}</div>`}
          </div>
        </div>`;
      }).join('');
      el.innerHTML = rows;
    })
    .catch(() => { el.innerHTML = '<div style="color:#f87171;font-size:0.8rem">Status fetch failed.</div>'; });
}

function refreshAllRithmicSettings() {
  const msgEl = document.getElementById('rithmic-refresh-msg');
  if (msgEl) msgEl.textContent = '🔄 Refreshing all accounts…';
  fetch('/api/rithmic/refresh-all', {method: 'POST'})
    .then(r => r.json())
    .then(() => {
      // Give the async refresh ~3s to complete, then reload status
      setTimeout(() => {
        loadRithmicLiveStatus();
        if (msgEl) msgEl.textContent = '✅ Done';
        setTimeout(() => { if (msgEl) msgEl.textContent = ''; }, 3000);
      }, 3000);
    })
    .catch(() => { if (msgEl) msgEl.textContent = '❌ Refresh failed'; });
}
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Route handlers
# ---------------------------------------------------------------------------


@router.get("/settings", response_class=HTMLResponse)
async def settings_page() -> HTMLResponse:
    """Serve the full Settings HTML page."""
    return HTMLResponse(content=_SETTINGS_PAGE_HTML)


# ---------------------------------------------------------------------------
# Services configuration endpoints
# ---------------------------------------------------------------------------


@router.get("/settings/services/config")
async def get_service_config():
    """Return current service URL configuration."""
    overrides = _load_persisted_settings()
    svc = overrides.get("services", {})
    return {
        "data_service_url": svc.get("data_service_url", os.getenv("DATA_SERVICE_URL", "http://100.113.72.63:8000")),
        "trainer_service_url": svc.get(
            "trainer_service_url", os.getenv("TRAINER_SERVICE_URL", "http://100.122.184.58:8200")
        ),
    }


@router.post("/settings/services/update")
async def update_service_config(body: dict):
    """Update service URL configuration (persisted to Redis)."""
    overrides = _load_persisted_settings()
    svc = overrides.get("services", {})

    if "data_service_url" in body:
        svc["data_service_url"] = body["data_service_url"]
    if "trainer_service_url" in body:
        svc["trainer_service_url"] = body["trainer_service_url"]
    overrides["services"] = svc
    _save_persisted_settings(overrides)

    logger.info("Service URLs updated: %s", svc)
    return {"status": "ok", "message": "Service URLs saved", "services": svc}


@router.get("/settings/services/probe")
async def probe_services():
    """Probe all configured services for health/connectivity.

    Local services (Redis, Postgres, Engine) are checked in-process using
    the same health functions the /health endpoint uses — no external HTTP
    round-trips needed and no risk of firewall / NAT false negatives.

    External services (Trainer) are probed via HTTP because they run in
    separate processes / machines.
    """
    import time

    from lib.services.data.api.health import _check_postgres, _check_redis

    overrides = _load_persisted_settings()
    svc = overrides.get("services", {})

    trainer_url = svc.get("trainer_service_url", os.getenv("TRAINER_SERVICE_URL", "http://100.122.184.58:8200"))

    # Displayed URL for the engine — whatever the operator configured, for
    # reference only (we don't HTTP-probe it since we *are* it).
    data_url = svc.get("data_service_url", os.getenv("DATA_SERVICE_URL", "http://100.113.72.63:8000"))

    services = []

    # ── Engine / Data Service — in-process check ─────────────────────────
    # We are the data service, so probe ourselves via the engine singleton
    # and the health helpers rather than making an HTTP call to our own IP.
    engine_status = "unknown"
    engine_latency = -1.0
    engine_error = ""
    try:
        from lib.trading.engine import get_engine

        t0 = time.monotonic()
        eng = get_engine()
        if eng is not None:
            st = eng.get_status()
            engine_status = st.get("engine", "unknown")
            engine_latency = round((time.monotonic() - t0) * 1000, 1)
            if engine_status not in ("running", "idle"):
                engine_error = f"engine state: {engine_status}"
        else:
            engine_error = "engine not initialised"
    except Exception as exc:
        engine_error = str(exc)[:120]

    engine_ok = engine_status in ("running", "idle") and not engine_error
    services.append(
        {
            "name": "Engine (Data Service)",
            "url": data_url,
            "ok": engine_ok,
            "latency_ms": engine_latency,
            "status": 200 if engine_ok else 0,
            "error": engine_error,
            "detail": engine_status,
        }
    )

    # ── Trainer — external HTTP probe ────────────────────────────────────
    result = _probe_service(trainer_url)
    result["name"] = "Trainer Service (GPU)"
    result["url"] = trainer_url
    services.append(result)

    # ── Redis — in-process check (ping) ──────────────────────────────────
    t0 = time.monotonic()
    redis_result = _check_redis()
    redis_latency = round((time.monotonic() - t0) * 1000, 1)
    redis_ok = redis_result.get("connected", False)
    _redis_url = os.getenv("REDIS_URL", os.getenv("REDIS_HOST", "redis://redis:6379/0"))
    services.append(
        {
            "name": "Redis",
            "url": _redis_url,
            "ok": redis_ok,
            "latency_ms": redis_latency if redis_ok else -1,
            "status": 200 if redis_ok else 0,
            "error": redis_result.get("error", "") if not redis_ok else "",
        }
    )

    # ── PostgreSQL — in-process check (SELECT 1) ─────────────────────────
    t0 = time.monotonic()
    pg_result = _check_postgres()
    pg_latency = round((time.monotonic() - t0) * 1000, 1)
    pg_ok = pg_result.get("connected", False)
    _db_url = os.getenv("DATABASE_URL", "")
    # Strip credentials from the display URL  (user:pass@host/db → host/db)
    _db_display = _db_url.split("@")[-1] if "@" in _db_url else (_db_url or "(not configured)")
    services.append(
        {
            "name": "PostgreSQL",
            "url": _db_display,
            "ok": pg_ok,
            "latency_ms": pg_latency if pg_ok else -1,
            "status": 200 if pg_ok else 0,
            "error": pg_result.get("error", "") if not pg_ok else "",
        }
    )

    return {"services": services}


# ---------------------------------------------------------------------------
# Feature toggle endpoints
# ---------------------------------------------------------------------------


@router.get("/settings/features/config")
async def get_features_config():
    """Return current feature toggle state."""
    overrides = _load_persisted_settings()
    feat = overrides.get("features", {})

    # Defaults sourced from env vars and known defaults
    return {
        "enable_live_positions": feat.get("enable_live_positions", os.getenv("ENABLE_LIVE_POSITIONS", "0") == "1"),
        "enable_kraken_crypto": feat.get("enable_kraken_crypto", os.getenv("ENABLE_KRAKEN_CRYPTO", "0") == "1"),
        "massive_autostart": feat.get("massive_autostart", os.getenv("MASSIVE_AUTOSTART", "0") == "1"),
        "enable_grok": feat.get("enable_grok", os.getenv("ENABLE_GROK", "1") == "1"),
        "orb_cnn_gate": feat.get("orb_cnn_gate", os.getenv("ORB_CNN_GATE", "1") == "1"),
        "orb_filter_gate": feat.get("orb_filter_gate", os.getenv("ORB_FILTER_GATE", "1") == "1"),
        "mtf_alignment_check": feat.get("mtf_alignment_check", True),
        "enable_sar": feat.get("enable_sar", False),
        "tpt_mode": feat.get("tpt_mode", False),
        "enable_tp3_trailing": feat.get("enable_tp3_trailing", True),
        "enable_auto_brackets": feat.get("enable_auto_brackets", True),
        "enable_debug_logging": feat.get("enable_debug_logging", False),
    }


@router.post("/settings/features/update")
async def update_features(body: dict):
    """Update feature toggles (persisted to Redis)."""
    overrides = _load_persisted_settings()
    feat = overrides.get("features", {})

    allowed_keys = {
        "enable_live_positions",
        "enable_kraken_crypto",
        "massive_autostart",
        "enable_grok",
        "orb_cnn_gate",
        "orb_filter_gate",
        "mtf_alignment_check",
        "enable_sar",
        "tpt_mode",
        "enable_tp3_trailing",
        "enable_auto_brackets",
        "enable_debug_logging",
    }

    changed = {}
    for key in allowed_keys:
        if key in body:
            feat[key] = bool(body[key])
            changed[key] = feat[key]

    overrides["features"] = feat
    _save_persisted_settings(overrides)

    logger.info("Feature toggles updated: %s", changed)
    return {"status": "ok", "message": "Features saved", "changed": changed}


# ---------------------------------------------------------------------------
# Risk & trading parameter endpoints
# ---------------------------------------------------------------------------


@router.get("/settings/risk/config")
async def get_risk_config():
    """Return current risk and trading parameters."""
    overrides = _load_persisted_settings()
    risk = overrides.get("risk", {})

    return {
        "risk_percent_per_trade": risk.get("risk_percent_per_trade", 0.5),
        "max_contracts": risk.get("max_contracts", 5),
        "max_concurrent_positions": risk.get("max_concurrent_positions", 5),
        "sl_atr_mult": risk.get("sl_atr_mult", 1.5),
        "tp1_atr_mult": risk.get("tp1_atr_mult", 2.0),
        "tp2_atr_mult": risk.get("tp2_atr_mult", 3.5),
        "tp3_atr_mult": risk.get("tp3_atr_mult", 5.0),
        "entry_cooldown_minutes": risk.get("entry_cooldown_minutes", 10),
        "default_sl_ticks": risk.get("default_sl_ticks", 20),
        "default_tp_ticks": risk.get("default_tp_ticks", 40),
        "sar_min_cnn_prob": risk.get("sar_min_cnn_prob", 0.85),
        "sar_min_mtf_score": risk.get("sar_min_mtf_score", 0.60),
        "sar_cooldown_minutes": risk.get("sar_cooldown_minutes", 30),
        "sar_chase_max_atr_fraction": risk.get("sar_chase_max_atr_fraction", 0.50),
        "sar_winning_cnn_prob": risk.get("sar_winning_cnn_prob", 0.92),
        "sar_high_winner_r_mult": risk.get("sar_high_winner_r_mult", 1.0),
        "cnn_threshold_override": risk.get("cnn_threshold_override", 0),
        "cnn_session_key": risk.get("cnn_session_key", "us"),
        "cnn_lookback_bars": risk.get("cnn_lookback_bars", 60),
        "volume_surge_mult": risk.get("volume_surge_mult", 1.5),
        "volume_avg_period": risk.get("volume_avg_period", 20),
        "min_orb_atr_ratio": risk.get("min_orb_atr_ratio", 0.3),
        "orb_minutes": risk.get("orb_minutes", 30),
        "require_vwap": risk.get("require_vwap", True),
    }


@router.post("/settings/risk/update")
async def update_risk_settings(body: dict):
    """Update risk and trading parameters (persisted to Redis)."""
    overrides = _load_persisted_settings()
    risk = overrides.get("risk", {})

    allowed_keys = {
        "risk_percent_per_trade",
        "max_contracts",
        "max_concurrent_positions",
        "sl_atr_mult",
        "tp1_atr_mult",
        "tp2_atr_mult",
        "tp3_atr_mult",
        "entry_cooldown_minutes",
        "default_sl_ticks",
        "default_tp_ticks",
        "sar_min_cnn_prob",
        "sar_min_mtf_score",
        "sar_cooldown_minutes",
        "sar_chase_max_atr_fraction",
        "sar_winning_cnn_prob",
        "sar_high_winner_r_mult",
        "cnn_threshold_override",
        "cnn_session_key",
        "cnn_lookback_bars",
        "volume_surge_mult",
        "volume_avg_period",
        "min_orb_atr_ratio",
        "orb_minutes",
        "require_vwap",
    }

    changed = {}
    for key in allowed_keys:
        if key in body:
            val = body[key]
            # Coerce types
            if key in (
                "max_contracts",
                "max_concurrent_positions",
                "entry_cooldown_minutes",
                "default_sl_ticks",
                "default_tp_ticks",
                "sar_cooldown_minutes",
                "cnn_lookback_bars",
                "volume_avg_period",
                "orb_minutes",
            ):
                val = int(val)
            elif key == "require_vwap":
                val = bool(val)
            elif key == "cnn_session_key":
                val = str(val)
            else:
                val = float(val)
            risk[key] = val
            changed[key] = val

    overrides["risk"] = risk
    _save_persisted_settings(overrides)

    logger.info("Risk settings updated: %s", changed)
    return {"status": "ok", "message": "Risk settings saved", "changed": changed}


# ---------------------------------------------------------------------------
# API key status endpoint
# ---------------------------------------------------------------------------


@router.get("/settings/keys/status")
async def get_api_key_status():
    """Return which API keys are configured (without exposing values)."""
    keys = [
        {"name": "MASSIVE_API_KEY", "configured": bool(os.getenv("MASSIVE_API_KEY", ""))},
        {"name": "KRAKEN_API_KEY", "configured": bool(os.getenv("KRAKEN_API_KEY", ""))},
        {"name": "KRAKEN_API_SECRET", "configured": bool(os.getenv("KRAKEN_API_SECRET", ""))},
        {"name": "XAI_API_KEY", "configured": bool(os.getenv("XAI_API_KEY", ""))},
        {"name": "DISCORD_WEBHOOK_URL", "configured": bool(os.getenv("DISCORD_WEBHOOK_URL", ""))},
        {"name": "DATABASE_URL", "configured": os.getenv("DATABASE_URL", "").startswith("postgresql")},
        {"name": "REDIS_URL", "configured": bool(os.getenv("REDIS_URL", ""))},
    ]
    return {"keys": keys}


# ---------------------------------------------------------------------------
# Data source switching endpoints (KRAKEN-SIM-D)
# ---------------------------------------------------------------------------

_DATA_SOURCE_KEY = "settings:data_source"
_VALID_DATA_SOURCES = ("kraken", "rithmic", "both")


@router.get("/api/settings/data-source")
async def get_data_source():
    """Return the current data source configuration.

    Reads the active source from Redis, checks Kraken and Rithmic
    connectivity, and reports which sources are available.
    """
    # --- Read active source from Redis ---
    active_source = "kraken"
    try:
        from lib.core.cache import REDIS_AVAILABLE, _r

        if REDIS_AVAILABLE and _r is not None:
            raw = _r.get(_DATA_SOURCE_KEY)
            if raw is not None:
                val = raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)
                if val in _VALID_DATA_SOURCES:
                    active_source = val
    except Exception:
        pass

    # --- Check Kraken status ---
    kraken_status = "disconnected"
    try:
        from lib.integrations.kraken_client import get_kraken_feed

        feed = get_kraken_feed()
        if feed is not None:
            if getattr(feed, "is_connected", False):
                kraken_status = "connected"
            elif getattr(feed, "is_running", False):
                kraken_status = "connecting"
    except Exception:
        pass

    # --- Check Rithmic status ---
    rithmic_status = "not_configured"
    try:
        from lib.integrations.rithmic_client import get_manager

        mgr = get_manager()
        all_status = mgr.get_all_status()
        if not all_status:
            rithmic_status = "not_configured"
        else:
            rithmic_status = "disconnected"
            for _key, st in all_status.items():
                if isinstance(st, dict) and st.get("connected"):
                    rithmic_status = "connected"
                    break
    except Exception:
        pass

    sim_enabled = os.getenv("SIM_ENABLED", "0") == "1"

    return {
        "active_source": active_source,
        "kraken_status": kraken_status,
        "rithmic_status": rithmic_status,
        "available_sources": list(_VALID_DATA_SOURCES),
        "sim_enabled": sim_enabled,
        "sim_data_source": active_source if active_source != "both" else "kraken",
    }


@router.post("/api/settings/data-source")
async def set_data_source(body: dict):
    """Switch the active data source.

    Request body: ``{"source": "kraken"}`` or ``{"source": "rithmic"}``
    or ``{"source": "both"}``.

    Stores the value in Redis and publishes a ``data_source_changed``
    event on the ``futures:events`` channel.
    """
    source = body.get("source", "").lower().strip()
    if source not in _VALID_DATA_SOURCES:
        return {
            "status": "error",
            "message": f"Invalid source '{source}'. Must be one of: {', '.join(_VALID_DATA_SOURCES)}",
        }

    try:
        from lib.core.cache import REDIS_AVAILABLE, _r

        if REDIS_AVAILABLE and _r is not None:
            _r.set(_DATA_SOURCE_KEY, source)
            # Publish event so other components can react
            _r.publish(
                "futures:events",
                json.dumps({"event": "data_source_changed", "source": source}),
            )
            logger.info("Data source switched to: %s", source)
        else:
            logger.warning("Redis unavailable — data source change not persisted")
            return {"status": "error", "message": "Redis unavailable"}
    except Exception as exc:
        logger.error("Failed to set data source: %s", exc)
        return {"status": "error", "message": f"Failed to set data source: {exc}"}

    return {"status": "ok", "message": f"Data source set to '{source}'", "active_source": source}
