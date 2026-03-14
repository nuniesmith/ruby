# =============================================================================
# Ruby Futures — Shared Python Base Image
# =============================================================================
#
# Installs all Python dependencies from pyproject.toml ONCE.
# The data, engine, and web service Dockerfiles all inherit from this image
# via:
#
#   FROM nuniesmith/futures:base
#
# This eliminates the ~700 MB pip install that was previously duplicated
# across every service build, cutting total build time by ~3-4x for the
# CPU services.
#
# Build order:
#   1. docker compose build base          ← installs deps (slow, once)
#   2. docker compose build data engine web  ← just copies src/ (fast)
#
# The base image is rebuilt only when pyproject.toml changes.
# All other source-code changes skip it entirely.
#
# Trainer has its own CUDA base and [gpu] extras so it is NOT derived
# from this image — see docker/trainer/Dockerfile.
# =============================================================================

# ---------------------------------------------------------------------------
# Stage 1 — dep-builder
# Installs everything into /usr/local so the runtime stage can COPY it in
# one layer without any build tools present.
# ---------------------------------------------------------------------------
FROM python:3.13-slim AS dep-builder

# Build-time dependencies for packages with C extensions:
#   gcc / g++       — required by some packages that build native wheels
#   libpq-dev       — psycopg builds against the Postgres headers
#   pkg-config      — used by various C-extension build scripts
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        gcc \
        g++ \
        libpq-dev \
        pkg-config \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy only the files that affect dependency resolution.
# These rarely change, so Docker's layer cache keeps this RUN cached
# across every source-only rebuild.
COPY pyproject.toml .
COPY README.md .
COPY src/lib/__init__.py src/lib/__init__.py

# Install all base dependencies (no [gpu] extras — those belong to trainer).
# --no-cache-dir keeps the image lean; the build cache is handled by
# Docker's layer cache on the COPY above instead.
RUN pip install --no-cache-dir ".[rithmic]"

# ---------------------------------------------------------------------------
# Stage 2 — runtime base
# Stripped-down image with only the runtime libs needed to actually run the
# installed packages (no gcc, no -dev headers).
# ---------------------------------------------------------------------------
FROM python:3.13-slim AS runtime

# Runtime-only native libs:
#   libpq5   — psycopg binary wheel links against this at runtime
#   libgomp1 — OpenMP, required by numpy / scikit-learn / matplotlib
LABEL org.opencontainers.image.title="futures-base" \
      org.opencontainers.image.description="Shared Python runtime base for Ruby Futures CPU services"

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libpq5 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Pull the installed packages and console-script wrappers out of the
# builder stage. Only these two directories are needed at runtime:
#   site-packages  — all Python packages
#   /usr/local/bin — pip-generated entry-points (uvicorn, etc.)
COPY --from=dep-builder /usr/local/lib/python3.13/site-packages \
                        /usr/local/lib/python3.13/site-packages
COPY --from=dep-builder /usr/local/bin \
                        /usr/local/bin

# Shared environment defaults inherited by all child images.
# Individual services can override these in their own Dockerfiles
# or via docker-compose environment: blocks.
ENV PYTHONPATH="/app/src"
ENV PYTHONUNBUFFERED=1
ENV LOG_LEVEL=info
