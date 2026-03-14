# =============================================================================
# Ruby Futures — Web Dashboard Service
# =============================================================================
#
# Inherits ALL Python dependencies from the shared base image.
# The base image installs the full pyproject.toml dependency set once;
# this Dockerfile only copies in the application source and static assets.
#
# Rebuild triggers:
#   pyproject.toml changed   →  rebuild base first, then this
#   src/ or static/ changed  →  rebuild this only (base cache is reused)
#
# Build:
#   docker compose build base   # if deps changed
#   docker compose build web
# =============================================================================

FROM nuniesmith/futures:base

WORKDIR /app

# Application source — the only layer that changes on normal dev cycles.
COPY src/ ./src/

# Static assets (trading dashboard HTML, JS, CSS)
COPY static/ ./static/

# Service-specific environment defaults (all overridable via compose)
ENV WEB_HOST=0.0.0.0
ENV WEB_PORT=8080
ENV DATA_SERVICE_URL=http://data:8000

EXPOSE 8080

HEALTHCHECK --interval=15s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c \
        "import urllib.request; urllib.request.urlopen('http://localhost:${WEB_PORT:-8080}/health')" \
    || exit 1

CMD ["uvicorn", "lib.services.web.main:app", "--host", "0.0.0.0", "--port", "8080", "--log-level", "info"]
