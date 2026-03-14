#!/bin/bash
# entrypoint.sh — start the engine worker.
# The data service (uvicorn / FastAPI) now runs in its own container.
# If the engine process exits for any reason the container exits so Docker restarts it.

set -eo pipefail

echo "[startup] PYTHONPATH=${PYTHONPATH}"
echo "[startup] Starting Engine worker..."

exec python -m lib.services.engine.main
