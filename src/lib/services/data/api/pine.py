"""
Pine Script Generator API Router
=================================
Provides endpoints for the Pine Script Generator web UI.

The generator assembles modular ``.pine`` files from
``src/lib/integrations/pine/modules/`` into complete TradingView indicator
scripts using configuration from ``params.yaml``.

Endpoints
---------
GET  /pine                      → Full HTML dashboard page
GET  /api/pine/modules          → List of module files with metadata
GET  /api/pine/module/{name}    → Single module content
GET  /api/pine/params           → Current params.yaml
PUT  /api/pine/params           → Update params.yaml (makes backup)
POST /api/pine/generate         → Generate indicator script
GET  /api/pine/output           → List generated output files
GET  /api/pine/download/{name}  → Download generated .pine file
GET  /api/pine/stats            → Indicator statistics
GET  /api/pine/status/html      → HTMX fragment for module status
"""

from __future__ import annotations

import logging
import os
import shutil
from datetime import UTC, datetime
from pathlib import Path

import yaml  # type: ignore[import-untyped]
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

logger = logging.getLogger("api.pine")

router = APIRouter(tags=["Pine Script Generator"])

# ---------------------------------------------------------------------------
# Lazy-initialised paths & generator singleton
# ---------------------------------------------------------------------------

_generator = None
_BASE_DIR: str | None = None
_OUTPUT_DIR: str | None = None
_MODULES_DIR: str | None = None
_PARAMS_FILE: str | None = None


def _init_paths() -> None:
    """Resolve project paths for the Pine integration directory."""
    global _BASE_DIR, _OUTPUT_DIR, _MODULES_DIR, _PARAMS_FILE  # noqa: PLW0603
    if _BASE_DIR is not None:
        return
    # Navigate from  data/api/  →  lib/integrations/pine/
    _BASE_DIR = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "integrations", "pine"),
    )
    _OUTPUT_DIR = os.environ.get("PINE_OUTPUT_DIR", os.path.join(_BASE_DIR, "pine_output"))
    _MODULES_DIR = os.path.join(_BASE_DIR, "modules")
    _PARAMS_FILE = os.path.join(_BASE_DIR, "params.yaml")


def _get_generator():
    """Return a cached ``PineScriptGenerator`` instance (lazy import)."""
    global _generator  # noqa: PLW0603
    _init_paths()
    assert _BASE_DIR is not None  # guaranteed by _init_paths
    assert _OUTPUT_DIR is not None
    if _generator is None:
        try:
            from lib.integrations.pine.generate import PineScriptGenerator

            os.makedirs(_OUTPUT_DIR, exist_ok=True)
            _generator = PineScriptGenerator(_BASE_DIR)
        except Exception as exc:
            logger.error("Failed to init PineScriptGenerator: %s", exc)
            return None
    return _generator


def _reset_generator() -> None:
    """Drop the cached generator so the next call re-reads ``params.yaml``.

    Also resets the singleton held by the integration package itself so that
    any other consumer of ``get_generator()`` picks up the change.
    """
    global _generator  # noqa: PLW0603
    _generator = None
    try:
        from lib.integrations.pine.main import reset_generator

        reset_generator()
    except Exception as exc:  # noqa: BLE001
        logger.debug("Could not reset integration generator singleton: %s", exc)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _file_info(file_path: str) -> dict:
    """Return size / mtime metadata for *file_path*."""
    try:
        st = os.stat(file_path)
        return {
            "size_bytes": st.st_size,
            "size_kb": round(st.st_size / 1024, 1),
            "modified": datetime.fromtimestamp(st.st_mtime, tz=UTC).isoformat(),
            "modified_ts": st.st_mtime,
        }
    except OSError:
        return {"size_bytes": 0, "size_kb": 0, "modified": "", "modified_ts": 0}


def _read_text(path: str, max_lines: int = 0) -> str:
    """Read a text file, optionally truncated to *max_lines*."""
    try:
        with open(path, encoding="utf-8") as fh:
            if max_lines > 0:
                return "".join(fh.readline() for _ in range(max_lines))
            return fh.read()
    except OSError as exc:
        return f"// Error reading file: {exc}"


def _resolve_static_html() -> Path | None:
    """Locate ``pine.html`` across Docker + local dev paths."""
    candidates = [
        Path("/app/static/pine.html"),
        Path(__file__).resolve().parent.parent.parent.parent.parent.parent / "static" / "pine.html",
        Path(__file__).resolve().parent.parent.parent.parent.parent / "static" / "pine.html",
        Path.cwd() / "static" / "pine.html",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


# ---------------------------------------------------------------------------
# GET /pine — Full HTML dashboard page
# ---------------------------------------------------------------------------


@router.get("/pine", response_class=HTMLResponse)
async def pine_dashboard_page() -> HTMLResponse:
    """Serve the Pine Script Generator dashboard.

    The page is a self-contained HTMX single-page app read from
    ``static/pine.html`` and wrapped in the shared site nav shell.
    """
    try:
        from lib.services.data.api.dashboard import _build_page_shell

        html_file = _resolve_static_html()
        if html_file is not None:
            body_content = html_file.read_text(encoding="utf-8")
            return HTMLResponse(content=body_content, headers={"Cache-Control": "no-cache"})

        # Fallback — build a minimal placeholder via the shell.
        return HTMLResponse(
            content=_build_page_shell(
                title="Pine Generator — Ruby Futures",
                favicon_emoji="🌲",
                active_path="/pine",
                body_content=(
                    "<div style='display:flex;align-items:center;justify-content:center;"
                    "height:60vh;color:#94a3b8;font-family:monospace'>"
                    "<div style='text-align:center'>"
                    "<div style='font-size:2rem;margin-bottom:1rem'>🌲</div>"
                    "<div>pine.html not found — place it at <code>static/pine.html</code></div>"
                    "</div></div>"
                ),
            ),
            headers={"Cache-Control": "no-cache"},
        )
    except Exception:  # noqa: BLE001
        # Absolute fallback — no dashboard helper available
        html_file = _resolve_static_html()
        if html_file is not None:
            return HTMLResponse(content=html_file.read_text(encoding="utf-8"))
        return HTMLResponse(
            content=(
                "<html><body style='background:#0f1117;color:#ccc;font-family:monospace;"
                "display:flex;align-items:center;justify-content:center;height:100vh'>"
                "<div>pine.html not found</div></body></html>"
            ),
        )


# ---------------------------------------------------------------------------
# GET /api/pine/modules — List all module files
# ---------------------------------------------------------------------------


@router.get("/api/pine/modules")
async def list_modules() -> JSONResponse:
    """Return a JSON array of ``.pine`` module files with metadata and a 3-line preview."""
    _init_paths()
    assert _MODULES_DIR is not None
    modules: list[dict] = []

    if not os.path.isdir(_MODULES_DIR):
        return JSONResponse(content={"modules": [], "count": 0, "modules_dir": _MODULES_DIR})

    for fname in sorted(os.listdir(_MODULES_DIR)):
        if not fname.endswith(".pine"):
            continue
        fpath = os.path.join(_MODULES_DIR, fname)
        info = _file_info(fpath)
        preview = _read_text(fpath, max_lines=3)
        modules.append({"name": fname, "preview": preview, **info})

    return JSONResponse(content={"modules": modules, "count": len(modules)})


# ---------------------------------------------------------------------------
# GET /api/pine/module/{name} — Single module content
# ---------------------------------------------------------------------------


@router.get("/api/pine/module/{name}")
async def get_module(name: str) -> JSONResponse:
    """Return the full content of a single ``.pine`` module file."""
    _init_paths()
    assert _MODULES_DIR is not None

    if ".." in name or "/" in name or "\\" in name:
        raise HTTPException(status_code=400, detail="Invalid module name")

    fpath = os.path.join(_MODULES_DIR, name)
    if not os.path.isfile(fpath):
        raise HTTPException(status_code=404, detail=f"Module '{name}' not found")

    content = _read_text(fpath)
    info = _file_info(fpath)
    line_count = content.count("\n") + 1

    return JSONResponse(
        content={
            "name": name,
            "content": content,
            "line_count": line_count,
            **info,
        },
    )


# ---------------------------------------------------------------------------
# GET /api/pine/params — Read params.yaml
# ---------------------------------------------------------------------------


@router.get("/api/pine/params")
async def get_params() -> JSONResponse:
    """Return the current ``params.yaml`` content as JSON."""
    _init_paths()
    assert _PARAMS_FILE is not None

    if not os.path.isfile(_PARAMS_FILE):
        raise HTTPException(status_code=404, detail="params.yaml not found")

    try:
        with open(_PARAMS_FILE, encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to parse params.yaml: {exc}") from exc

    raw_text = _read_text(_PARAMS_FILE)
    return JSONResponse(content={"params": data, "raw": raw_text})


# ---------------------------------------------------------------------------
# PUT /api/pine/params — Update params.yaml (with backup)
# ---------------------------------------------------------------------------


@router.put("/api/pine/params")
async def update_params(body: dict) -> JSONResponse:
    """Overwrite ``params.yaml`` with *body["raw"]* (YAML text).

    A timestamped backup of the previous file is created automatically.
    The cached generator is reset so the next generation picks up changes.
    """
    _init_paths()
    assert _PARAMS_FILE is not None

    raw_yaml: str | None = body.get("raw")
    if raw_yaml is None:
        raise HTTPException(status_code=422, detail="Request body must contain 'raw' (YAML text)")

    # Validate the YAML is parseable before we overwrite
    try:
        yaml.safe_load(raw_yaml)
    except yaml.YAMLError as exc:
        raise HTTPException(status_code=422, detail=f"Invalid YAML: {exc}") from exc

    # Timestamped backup
    if os.path.isfile(_PARAMS_FILE):
        ts = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%S")
        backup_path = f"{_PARAMS_FILE}.bak.{ts}"
        try:
            shutil.copy2(_PARAMS_FILE, backup_path)
            logger.info("Backed up params.yaml → %s", backup_path)
        except OSError as exc:
            logger.warning("Could not create backup: %s", exc)

    try:
        with open(_PARAMS_FILE, "w", encoding="utf-8") as fh:
            fh.write(raw_yaml)
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"Failed to write params.yaml: {exc}") from exc

    _reset_generator()
    return JSONResponse(content={"ok": True, "message": "params.yaml updated"})


# ---------------------------------------------------------------------------
# POST /api/pine/generate — Generate indicator script
# ---------------------------------------------------------------------------


@router.post("/api/pine/generate")
async def generate_script(body: dict | None = None) -> JSONResponse:
    """Generate the Pine Script indicator and return result metadata.

    Accepts an optional JSON body ``{"indicator_type": "ruby"}``; defaults to
    ``"ruby"`` when omitted.
    """
    _init_paths()
    output_dir = _OUTPUT_DIR
    if output_dir is None:
        raise HTTPException(status_code=500, detail="Pine output directory not configured")

    indicator_type = "ruby"
    if body and isinstance(body.get("indicator_type"), str):
        indicator_type = body["indicator_type"]

    gen = _get_generator()
    if gen is None:
        raise HTTPException(status_code=500, detail="PineScriptGenerator failed to initialise")

    try:
        output_path = gen.save_script(indicator_type, output_dir)
    except Exception as exc:
        logger.error("Generation failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Generation failed: {exc}") from exc

    info = _file_info(output_path)

    # Gather stats for the response
    try:
        stats = gen.get_indicator_stats(indicator_type)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not collect stats after generation: %s", exc)
        stats = {}

    return JSONResponse(
        content={
            "ok": True,
            "path": output_path,
            "filename": os.path.basename(output_path),
            "stats": stats,
            **info,
        },
    )


# ---------------------------------------------------------------------------
# GET /api/pine/output — List generated output files
# ---------------------------------------------------------------------------


@router.get("/api/pine/output")
async def list_output() -> JSONResponse:
    """Return a list of ``.pine`` files in the output directory."""
    _init_paths()
    assert _OUTPUT_DIR is not None

    if not os.path.isdir(_OUTPUT_DIR):
        return JSONResponse(content={"files": [], "count": 0, "output_dir": _OUTPUT_DIR})

    files: list[dict] = []
    for fname in sorted(os.listdir(_OUTPUT_DIR)):
        fpath = os.path.join(_OUTPUT_DIR, fname)
        if not os.path.isfile(fpath):
            continue
        if not fname.endswith(".pine"):
            continue
        info = _file_info(fpath)
        files.append({"name": fname, **info})

    return JSONResponse(content={"files": files, "count": len(files), "output_dir": _OUTPUT_DIR})


# ---------------------------------------------------------------------------
# GET /api/pine/download/{filename} — Download a generated file
# ---------------------------------------------------------------------------


@router.get("/api/pine/download/{filename}")
async def download_output(filename: str) -> FileResponse:
    """Download a generated ``.pine`` file as a ``text/plain`` attachment."""
    _init_paths()
    assert _OUTPUT_DIR is not None

    if ".." in filename or "/" in filename or "\\" in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")

    fpath = os.path.join(_OUTPUT_DIR, filename)
    if not os.path.isfile(fpath):
        raise HTTPException(status_code=404, detail=f"File '{filename}' not found in output directory")

    return FileResponse(
        path=fpath,
        filename=filename,
        media_type="text/plain",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


# ---------------------------------------------------------------------------
# GET /api/pine/stats — Indicator statistics
# ---------------------------------------------------------------------------


@router.get("/api/pine/stats")
async def get_stats() -> JSONResponse:
    """Return statistics about the Ruby indicator (module count, line count, etc.)."""
    _init_paths()

    gen = _get_generator()
    if gen is None:
        raise HTTPException(status_code=500, detail="PineScriptGenerator failed to initialise")

    try:
        stats = gen.get_indicator_stats("ruby")
    except Exception as exc:
        logger.error("Stats failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Stats failed: {exc}") from exc

    return JSONResponse(content=stats)


# ---------------------------------------------------------------------------
# GET /api/pine/status/html — HTMX fragment
# ---------------------------------------------------------------------------


@router.get("/api/pine/status/html", response_class=HTMLResponse)
async def pine_status_html() -> HTMLResponse:
    """HTMX fragment — module list + generation status + generate button.

    Designed to be loaded into a container via::

        <div hx-get="/api/pine/status/html" hx-trigger="load, every 30s" hx-swap="innerHTML"></div>
    """
    _init_paths()
    modules_dir = _MODULES_DIR
    output_dir = _OUTPUT_DIR
    if modules_dir is None or output_dir is None:
        return HTMLResponse(content="<div style='color:#f55'>Pine paths not configured</div>")

    # ------------------------------------------------------------------
    # Collect module list
    # ------------------------------------------------------------------
    module_items = ""
    module_count = 0
    module_names: list[str] = []
    if os.path.isdir(modules_dir):
        for fname in sorted(os.listdir(modules_dir)):
            if not fname.endswith(".pine"):
                continue
            module_count += 1
            module_names.append(fname)
            fpath = os.path.join(modules_dir, fname)
            info = _file_info(fpath)
            module_items += (
                f'<div class="pine-mod-row" '
                f'hx-get="/api/pine/module/{fname}" hx-target="#pine-module-content" '
                f'hx-swap="innerHTML" style="cursor:pointer;">'
                f'<span class="pine-mod-name">{fname}</span>'
                f'<span class="pine-mod-size">{info["size_kb"]} KB</span>'
                f"</div>\n"
            )

    # ------------------------------------------------------------------
    # Latest output info
    # ------------------------------------------------------------------
    output_info = ""
    if os.path.isdir(output_dir):
        _od: str = output_dir  # local binding for lambda narrowing
        pine_files = [f for f in os.listdir(_od) if f.endswith(".pine") and os.path.isfile(os.path.join(_od, f))]
        if pine_files:
            latest = max(
                pine_files,
                key=lambda f, _d=_od: os.path.getmtime(os.path.join(_d, f)),  # type: ignore[misc]
            )
            info = _file_info(os.path.join(_od, latest))
            output_info = (
                f'<div class="pine-output-info">'
                f'<span style="color:#00ffff">📄 {latest}</span> '
                f'<span style="color:#888">({info["size_kb"]} KB — {info["modified"]})</span>'
                f"</div>"
            )
        else:
            output_info = '<div class="pine-output-info" style="color:#555">No output files yet</div>'
    else:
        output_info = '<div class="pine-output-info" style="color:#555">Output directory not found</div>'

    now = datetime.now(tz=UTC).strftime("%H:%M:%S UTC")

    fragment = f"""
<style>
  .pine-status-frag {{ font-family: 'JetBrains Mono', monospace; font-size: 12px; color: #ccc; }}
  .pine-mod-row {{
    display: flex; justify-content: space-between; align-items: center;
    padding: 4px 8px; border-bottom: 1px solid #1e222d; transition: background 0.15s;
  }}
  .pine-mod-row:hover {{ background: #1e222d; }}
  .pine-mod-name {{ color: #00ffff; font-weight: 600; }}
  .pine-mod-size {{ color: #555; font-size: 11px; }}
  .pine-output-info {{ padding: 6px 8px; font-size: 11px; }}
  .pine-gen-btn {{
    display: block; width: 100%; margin-top: 8px; padding: 8px 0;
    background: #00ffff; color: #0f1117; border: none; border-radius: 4px;
    font-family: 'JetBrains Mono', monospace; font-weight: 700; font-size: 12px;
    cursor: pointer; text-align: center; transition: opacity 0.15s;
  }}
  .pine-gen-btn:hover {{ opacity: 0.85; }}
  .pine-frag-header {{
    display: flex; justify-content: space-between; padding: 4px 8px;
    color: #555; font-size: 10px;
  }}
</style>
<div class="pine-status-frag">
  <div class="pine-frag-header">
    <span>🌲 {module_count} modules</span>
    <span>{now}</span>
  </div>
  {module_items}
  <div style="border-top:1px solid #222;margin-top:4px;padding-top:4px;">
    {output_info}
  </div>
  <button class="pine-gen-btn"
          hx-post="/api/pine/generate"
          hx-target="#pine-generate-result"
          hx-swap="innerHTML"
          hx-indicator="#pine-gen-spinner">
    ▶ Generate Ruby Indicator
  </button>
  <span id="pine-gen-spinner" class="htmx-indicator"
        style="color:#00ffff;font-size:11px;padding:4px 8px;">
    generating…
  </span>
</div>
"""
    return HTMLResponse(content=fragment)
