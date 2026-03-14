# Dockerfile.trainer — GPU Training Service (dedicated machine)
# Runs the overnight CNN retraining pipeline and/or train_gpu.py directly.
# This image is intentionally separate from the engine so that:
#   - The main stack (docker-compose.yml) runs CPU-only on any machine.

# Installs the [gpu] extras so torch/torchvision get CUDA-enabled wheels.

# ---------------------------------------------------------------------------
# Stage 1: Install dependencies
# ---------------------------------------------------------------------------
FROM nvidia/cuda:12.8.0-runtime-ubuntu24.04 AS deps

ENV DEBIAN_FRONTEND=noninteractive

# Install deadsnakes PPA for Python 3.13 + build essentials
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common \
        gpg-agent \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.13 \
        python3.13-dev \
        python3.13-venv \
        build-essential \
        libpq-dev \
        curl \
        ca-certificates \
        tzdata \
    && rm -rf /var/lib/apt/lists/*

# Make python3.13 the default python
RUN update-alternatives --install /usr/bin/python  python  /usr/bin/python3.13 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.13 1

# Bootstrap pip for 3.13
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.13

WORKDIR /app

# Copy dependency metadata + minimal package stub so hatchling can resolve
COPY pyproject.toml .
COPY README.md .
COPY src/lib/__init__.py src/lib/__init__.py

# Install base deps + [gpu,server] extras:
#   gpu    → torch, torchvision, torchaudio, tqdm  (CUDA 12.8 wheels)
#   server → fastapi, uvicorn[standard], python-multipart
RUN pip install --no-cache-dir \
    ".[gpu,server]" \
    --extra-index-url https://download.pytorch.org/whl/cu128

# ---------------------------------------------------------------------------
# Stage 2: Lean runtime image
# ---------------------------------------------------------------------------
FROM nvidia/cuda:12.8.0-runtime-ubuntu24.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive

# Python 3.13 runtime only (no build tools)
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common \
        gpg-agent \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.13 \
        python3.13-venv \
        libpq5 \
        libgomp1 \
        tzdata \
    && apt-get purge -y --auto-remove software-properties-common gpg-agent \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python  python  /usr/bin/python3.13 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.13 1

WORKDIR /app

# Copy installed packages from deps stage
COPY --from=deps /usr/local/lib/python3.13 /usr/local/lib/python3.13
COPY --from=deps /usr/lib/python3/dist-packages /usr/lib/python3/dist-packages
COPY --from=deps /usr/lib/python3.13 /usr/lib/python3.13
COPY --from=deps /usr/local/bin /usr/local/bin

# Copy application source and training scripts
COPY src/     ./src/
COPY scripts/ ./scripts/

# Persistent directories — bind-mounted from the host in compose so that
# rsync can pull models/ directly without any docker cp / volume-export.
RUN mkdir -p /app/data /app/models /app/models/archive /app/dataset

# NVIDIA runtime — exposes the full GPU to PyTorch inside the container
# Set container timezone to UTC; ZoneInfo("America/New_York") converts correctly
# at runtime via the installed tzdata package.
ENV TZ=UTC

ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

ENV PYTHONPATH="/app/src"
ENV PYTHONUNBUFFERED=1

# Trainer server config (overridable via docker-compose environment:)
ENV TRAINER_HOST=0.0.0.0
ENV TRAINER_PORT=8200

EXPOSE 8200

# ---------------------------------------------------------------------------
# Healthcheck — HTTP liveness probe against the trainer server.
# The server starts quickly and writes /tmp/trainer_health.json on startup,
# but we probe the HTTP endpoint directly for a more reliable check.
# start_period=60s gives time for torch + model imports on first boot.
# ---------------------------------------------------------------------------
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c \
        "import urllib.request; urllib.request.urlopen('http://localhost:8200/health')" \
    || exit 1

# Default: run the persistent HTTP training server.
# The server stays alive between runs and accepts POST /train requests.
#
# Override to run the one-shot pipeline directly:
#   docker compose run --rm trainer python -m lib.services.training.trainer_server
CMD ["python", "-m", "lib.services.training.trainer_server"]
