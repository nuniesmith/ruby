"""
API key authentication for inter-service communication.

Protects data-service endpoints with a shared secret so that only
trusted clients can call them.

Configuration:
    Set ``API_KEY`` in the environment (or ``.env``).  When the variable is
    **not set or empty**, authentication is disabled (open access) so that
    local development and tests work without ceremony.

    Clients must send the key via the ``X-API-Key`` header
    (or the ``api_key`` query parameter as a fallback).

Usage in ``main.py``::

    from lib.services.data.api.auth import require_api_key

    app = FastAPI(dependencies=[Depends(require_api_key)])

Or per-router::

    router = APIRouter(dependencies=[Depends(require_api_key)])

The ``/health`` endpoint is explicitly excluded so that Docker health-checks
and load-balancers can probe the service without credentials.
"""

from __future__ import annotations

import logging
import os
import secrets

from fastapi import HTTPException, Request, Security, status
from fastapi.security import APIKeyHeader, APIKeyQuery

logger = logging.getLogger("api.auth")

# ---------------------------------------------------------------------------
# Read the shared secret from the environment.  When blank / unset,
# authentication is effectively a no-op (everything is allowed).
# ---------------------------------------------------------------------------
_API_KEY: str = os.getenv("API_KEY", "").strip()

# FastAPI security schemes — try header first, query param as fallback.
_header_scheme = APIKeyHeader(name="X-API-Key", auto_error=False)
_query_scheme = APIKeyQuery(name="api_key", auto_error=False)

# Paths that are always accessible without an API key.
_PUBLIC_PATHS: frozenset[str] = frozenset(
    {
        "/",
        "/health",
        "/metrics",
        "/metrics/prometheus",
        "/docs",
        "/openapi.json",
        "/redoc",
    }
)


def _is_public(path: str) -> bool:
    """Return ``True`` if *path* should bypass authentication."""
    return path.rstrip("/") in _PUBLIC_PATHS or path in _PUBLIC_PATHS


async def require_api_key(
    request: Request,
    header_key: str | None = Security(_header_scheme),
    query_key: str | None = Security(_query_scheme),
) -> str | None:
    """FastAPI dependency that enforces API key authentication.

    Returns the validated key (or ``None`` when auth is disabled).

    Raises ``HTTPException(403)`` when a key is configured but the
    request doesn't supply a matching one.
    """
    # 1. No key configured → auth disabled (dev / test mode)
    if not _API_KEY:
        return None

    # 2. Public endpoints are always allowed
    if _is_public(request.url.path):
        return None

    # 3. Check header, then query param
    provided = header_key or query_key

    if not provided:
        logger.warning(
            "Unauthenticated request blocked: %s %s",
            request.method,
            request.url.path,
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Missing API key. Supply via X-API-Key header or api_key query param.",
        )

    # Constant-time comparison to prevent timing attacks
    if not secrets.compare_digest(provided, _API_KEY):
        logger.warning(
            "Invalid API key from %s for %s %s",
            request.client.host if request.client else "unknown",
            request.method,
            request.url.path,
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API key.",
        )

    return provided


def get_api_key() -> str:
    """Return the configured API key (for clients that need to send it).

    Returns an empty string when no key is configured.
    """
    return _API_KEY


def is_auth_enabled() -> bool:
    """Return ``True`` if API key authentication is active."""
    return bool(_API_KEY)
