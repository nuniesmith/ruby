#!/bin/sh
set -e

echo "[startup] Starting Data Service (FastAPI) on port ${DATA_SERVICE_PORT:-8000}..."
echo "[startup] PYTHONPATH=${PYTHONPATH}"

exec python -m entrypoints.data.main
