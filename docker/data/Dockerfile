# =============================================================================
# Ruby Futures — Data Service
# =============================================================================
#
# Inherits ALL Python dependencies from the shared base image.
# The base image installs the full pyproject.toml dependency set once;
# this Dockerfile only copies in the application source and config.
#
# Rebuild triggers:
#   pyproject.toml changed  →  rebuild base first, then this
#   src/ or static/ changed →  rebuild this only (base cache is reused)
#
# Build:
#   docker compose build base   # if deps changed
#   docker compose build data
# =============================================================================

FROM nuniesmith/futures:base

# Runtime-only apt packages not already in the base image.
# bash is needed by entrypoint.sh (uses bash-specific syntax).
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        bash \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Application source — the only layer that changes on normal dev cycles.
COPY src/ ./src/

# Static assets (trading dashboard HTML, JS, CSS)
COPY static/ ./static/

# Service entrypoint
COPY docker/data/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Runtime data directory (SQLite journal, local bar cache)
RUN mkdir -p /app/data

# Service-specific environment defaults (all overridable via compose)
ENV DATA_SERVICE_HOST=0.0.0.0
ENV DATA_SERVICE_PORT=8000

EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c \
        "import urllib.request; urllib.request.urlopen('http://localhost:${DATA_SERVICE_PORT:-8000}/health')" \
    || exit 1

ENTRYPOINT ["/entrypoint.sh"]
