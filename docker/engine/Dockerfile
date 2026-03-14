# =============================================================================
# Ruby Futures — Engine Worker
# =============================================================================
#
# Inherits ALL Python dependencies from the shared base image.
# The base image installs the full pyproject.toml dependency set once;
# this Dockerfile only copies in the application source and scripts.
#
# Rebuild triggers:
#   pyproject.toml changed  →  rebuild base first, then this
#   src/ or scripts/ changed →  rebuild this only (base cache is reused)
#
# Build:
#   docker compose build base   # if deps changed
#   docker compose build engine
# =============================================================================

FROM nuniesmith/futures:base

# bash is needed by entrypoint.sh
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        bash \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Application source — the only layer that changes on normal dev cycles.
COPY src/ ./src/

# Entrypoint
COPY docker/engine/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Scripts (sync_models, monitoring helpers, etc.)
COPY scripts/ ./scripts/

# Persistent data directory (journal, model cache)
RUN mkdir -p /app/data

# Health check — the engine worker writes a heartbeat file every cycle.
# No HTTP port is exposed; the data service owns port 8000.
HEALTHCHECK --interval=15s --timeout=5s --start-period=60s --retries=3 \
    CMD test -f /tmp/engine_health.json || exit 1

ENTRYPOINT ["/entrypoint.sh"]
