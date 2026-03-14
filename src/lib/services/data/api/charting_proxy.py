"""
Charting Service Reverse Proxy
==============================
Proxies ``/charting-proxy/*`` requests from the data service to the internal
``charting`` container so that the browser never needs direct access to port
8003.  This allows the Charts iframe to work from any machine — not just
localhost.

The charting container serves static TradingView Lightweight Charts assets
(HTML, JS, CSS, images) via Nginx.  This proxy transparently forwards every
GET request and streams the response back with the original content-type.

Environment:
    CHARTING_SERVICE_URL  — internal Docker URL of the charting container.
                            Default: ``http://charting:8003``

Routes:
    GET /charting-proxy/              — proxy root (index.html)
    GET /charting-proxy/{path:path}   — proxy any sub-path (JS, CSS, assets)
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, Response

logger = logging.getLogger("futures.data.charting_proxy")

router = APIRouter()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_CHARTING_SERVICE_URL: str = os.getenv("CHARTING_SERVICE_URL", "http://charting:8003")

# ---------------------------------------------------------------------------
# Shared httpx client — reused across requests for connection pooling
# ---------------------------------------------------------------------------

_http_client: httpx.AsyncClient | None = None
_http_client_lock: asyncio.Lock | None = None


def _make_http_client(base_url: str) -> httpx.AsyncClient:
    """Create a fresh async httpx client for the charting service."""
    return httpx.AsyncClient(
        base_url=base_url,
        timeout=httpx.Timeout(30.0, connect=5.0),
        follow_redirects=True,
        limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
    )


async def _get_http_client() -> httpx.AsyncClient:
    """Return a reusable async httpx client.

    Thread-safe: uses an asyncio lock to prevent concurrent recreation.
    """
    global _http_client, _http_client_lock

    if _http_client_lock is None:
        _http_client_lock = asyncio.Lock()

    async with _http_client_lock:
        if _http_client is None or _http_client.is_closed:
            _http_client = _make_http_client(_CHARTING_SERVICE_URL)
        return _http_client


async def _invalidate_http_client() -> None:
    """Close and discard the pooled client so the next call creates a fresh one."""
    global _http_client

    if _http_client is not None:
        with contextlib.suppress(Exception):
            await _http_client.aclose()
        _http_client = None


# ---------------------------------------------------------------------------
# Proxy routes — /charting-proxy/{path}
# ---------------------------------------------------------------------------


@router.api_route("/charting-proxy/", methods=["GET", "HEAD", "OPTIONS"])
async def proxy_charting_root(request: Request) -> Response:
    """Reverse-proxy GET/HEAD/OPTIONS for the charting root to the charting container.

    HEAD and OPTIONS are included so the dashboard's fetch probe and CORS
    preflight requests don't get a 405 Method Not Allowed.

        GET  /charting-proxy/  → charting GET /
        HEAD /charting-proxy/  → charting HEAD /  (dashboard availability probe)
    """
    return await _proxy_charting(request, "")


@router.api_route("/charting-proxy/{path:path}", methods=["GET", "HEAD", "OPTIONS"])
async def proxy_charting_path(request: Request, path: str) -> Response:
    """Reverse-proxy GET/HEAD/OPTIONS for charting sub-paths to the charting container.

    Strips the ``/charting-proxy`` prefix before forwarding so that:
        GET  /charting-proxy/index.html → charting GET /index.html
        GET  /charting-proxy/js/app.js  → charting GET /js/app.js
    """
    return await _proxy_charting(request, path)


async def _proxy_charting(request: Request, path: str) -> Response:
    """Shared implementation for all charting proxy routes."""
    client = await _get_http_client()

    upstream_path = f"/{path}" if path else "/"
    if request.url.query:
        upstream_path = f"{upstream_path}?{request.url.query}"

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
        "authorization",
    }
    fwd_headers = {k: v for k, v in request.headers.items() if k.lower() not in skip}

    for attempt in range(2):
        try:
            resp = await client.request(
                method="GET",
                url=upstream_path,
                headers=fwd_headers,
            )

            excluded_resp = {
                "transfer-encoding",
                "content-encoding",  # httpx already decompresses — browser must not re-decompress
                "content-length",  # body length changes after decompression
                "connection",
                "keep-alive",
                "server",
            }
            resp_headers = {k: v for k, v in resp.headers.items() if k.lower() not in excluded_resp}

            return Response(
                content=resp.content,
                status_code=resp.status_code,
                headers=resp_headers,
                media_type=resp.headers.get("content-type", "text/html"),
            )

        except httpx.ReadError as exc:
            await _invalidate_http_client()
            if attempt == 0:
                logger.debug(
                    "Charting proxy ReadError for %s (attempt %d) — retrying: %s",
                    upstream_path,
                    attempt + 1,
                    exc,
                )
                client = await _get_http_client()
                continue
            logger.warning("Charting proxy ReadError for %s after retry: %s", upstream_path, exc)
            return JSONResponse(
                status_code=502,
                content={"error": "Charting connection reset — please retry", "detail": str(exc)},
            )

        except httpx.ConnectError:
            return JSONResponse(
                status_code=503,
                content={
                    "error": "Charting service unavailable",
                    "charting_url": _CHARTING_SERVICE_URL,
                    "hint": "Make sure the charting container is running: docker compose up -d charting",
                },
            )
        except httpx.TimeoutException:
            return JSONResponse(
                status_code=504,
                content={"error": "Charting service timed out"},
            )
        except Exception as exc:
            logger.error("Charting proxy error for %s: %s", upstream_path, exc, exc_info=True)
            return JSONResponse(
                status_code=500,
                content={"error": str(exc)},
            )

    # Unreachable — loop always returns, but satisfies type checker
    return JSONResponse(status_code=500, content={"error": "Unexpected proxy loop exit"})
