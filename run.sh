#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# Ruby Futures — Run Script
# =============================================================================
#
# Usage:
#   ./run.sh              Build, test, lint, then start Docker Compose
#   ./run.sh all          Start everything: core services + trainer + monitoring
#   ./run.sh --local      Run locally with a Python virtual environment
#   ./run.sh --down       Stop Docker Compose services (all profiles)
#   ./run.sh --test       Run tests + lint only (no compose)
#   ./run.sh --trainer    Include GPU trainer service (training profile)
#   ./run.sh --monitoring Include Prometheus + Grafana
#   ./run.sh --help       Show this help message
#
# Examples:
#   ./run.sh                        # Core services only (engine + web + db)
#   ./run.sh --trainer              # Core + GPU trainer
#   ./run.sh --monitoring           # Core + Prometheus + Grafana
#   ./run.sh --trainer --monitoring # Core + trainer + monitoring
#   ./run.sh all                    # Everything (core + trainer + monitoring)
#
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VENV_DIR=".venv"
ENV_FILE=".env"
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

log()  { echo -e "${CYAN}[run]${NC} $*"; }
ok()   { echo -e "${GREEN}[  ✓ ]${NC} $*"; }
warn() { echo -e "${YELLOW}[warn]${NC} $*"; }
err()  { echo -e "${RED}[fail]${NC} $*"; }

# Generate a cryptographically random string (URL-safe base64, no padding)
gen_secret() {
    python3 -c "import secrets; print(secrets.token_urlsafe(${1:-32}))"
}

usage() {
    echo "Usage: ./run.sh [command] [flags]"
    echo ""
    echo "Commands:"
    echo "  (no args)       Build, test, lint, then start core Docker services"
    echo "  all             Start everything: core + trainer (GPU) + monitoring"
    echo ""
    echo "Flags:"
    echo "  --local         Run locally with a Python virtual environment"
    echo "  --down          Stop all Docker Compose services (all profiles)"
    echo "  --test          Run tests + lint only (skip Docker build)"
    echo "  --trainer       Include GPU trainer service (training profile)"
    echo "  --monitoring    Include Prometheus + Grafana (monitoring profile)"
    echo "  --help          Show this help message"
    echo ""
    echo "Examples:"
    echo "  ./run.sh                        # Core services only"
    echo "  ./run.sh --trainer              # Core + GPU trainer"
    echo "  ./run.sh --monitoring           # Core + Prometheus/Grafana"
    echo "  ./run.sh --trainer --monitoring # Core + trainer + monitoring"
    echo "  ./run.sh all                    # Everything (equivalent to --trainer --monitoring)"
}

# ---------------------------------------------------------------------------
# Virtual-environment management
# ---------------------------------------------------------------------------

ensure_venv() {
    if [ ! -d "$VENV_DIR" ]; then
        log "Creating virtual environment in ${VENV_DIR} ..."
        python3 -m venv "$VENV_DIR"
        ok "Virtual environment created"
    fi

    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"

    log "Updating pip ..."
    pip install --upgrade pip -q

    log "Installing project + dev dependencies from pyproject.toml ..."
    pip install -e ".[dev]" -q
    ok "Dependencies up to date"
}

# ---------------------------------------------------------------------------
# .env generation
# ---------------------------------------------------------------------------

ensure_env() {
    if [ -f "$ENV_FILE" ]; then
        ok ".env file already exists"
    else
        log "No .env file found — generating with secure random secrets ..."

        local pg_pass
        local redis_pass
        local secret_key
        pg_pass="$(gen_secret 32)"
        redis_pass="$(gen_secret 24)"
        secret_key="$(gen_secret 48)"

        cat > "$ENV_FILE" <<EOF
# =============================================================================
# Ruby Futures — Environment
# Generated on $(date -u +"%Y-%m-%dT%H:%M:%SZ")
# =============================================================================

# ---- Postgres ----
POSTGRES_USER=futures_user
POSTGRES_PASSWORD=${pg_pass}
POSTGRES_DB=futures_db

# ---- Redis ----
REDIS_PASSWORD=${redis_pass}

# ---- App Secret Key (sessions, CSRF, etc.) ----
SECRET_KEY=${secret_key}

# ---- Grafana ----
GRAFANA_ADMIN_USER=admin
GRAFANA_ADMIN_PASSWORD=admin

# ---- API Keys (you must fill these in) ----

# Massive.com API key for real-time futures data (CME/CBOT/NYMEX/COMEX)
MASSIVE_API_KEY=
FINNHUB_API_KEY=
ALPHA_VANTAGE_API_KEY=

KRAKEN_API_KEY=
KRAKEN_API_SECRET=

# xAI Grok API key for the AI Analyst tab
XAI_API_KEY=

# URL of the copier process (set empty to disable copier integration)
COPIER_URL=http://localhost:5682

# ---- ORB / CNN Gate settings ----
ORB_FILTER_GATE=majority
ORB_CNN_GATE=0

REDDIT_CLIENT_ID=your_client_id_here
REDDIT_CLIENT_SECRET=your_client_secret_here
REDDIT_POLL_INTERVAL=120   # optional, default 120s
EOF

        ok ".env generated with secure random secrets for Postgres, Redis, etc."
        warn "You still need to set your API keys:"
        warn "  • MASSIVE_API_KEY       — https://massive.com/dashboard"
        warn "  • FINNHUB_API_KEY       — https://finnhub.io"
        warn "  • ALPHA_VANTAGE_API_KEY — https://www.alphavantage.co"
        warn "  • KRAKEN_API_KEY        — https://www.kraken.com"
        warn "  • XAI_API_KEY           — https://console.x.ai"
        warn "  • REDDIT_CLIENT_ID      — https://www.reddit.com/prefs/apps"
        echo ""
    fi

    # Always warn about placeholder API keys
    if grep -q "^MASSIVE_API_KEY=$" "$ENV_FILE" 2>/dev/null; then
        warn "MASSIVE_API_KEY is empty — real-time data disabled (yfinance fallback)"
    fi
    if grep -q "^FINNHUB_API_KEY=$" "$ENV_FILE" 2>/dev/null; then
        warn "FINNHUB_API_KEY is empty — Finnhub data source will not work"
    fi
    if grep -q "^ALPHA_VANTAGE_API_KEY=$" "$ENV_FILE" 2>/dev/null; then
        warn "ALPHA_VANTAGE_API_KEY is empty — Alpha Vantage data source will not work"
    fi
    if grep -q "^KRAKEN_API_KEY=$" "$ENV_FILE" 2>/dev/null; then
        warn "KRAKEN_API_KEY is empty — Kraken integration will not work"
    fi
    if grep -q "^XAI_API_KEY=$" "$ENV_FILE" 2>/dev/null; then
        warn "XAI_API_KEY is empty — Grok AI Analyst tab will not work"
    fi
    if grep -q "your_client_id_here" "$ENV_FILE" 2>/dev/null; then
        warn "REDDIT_CLIENT_ID is still a placeholder — Reddit sentiment will not work"
    fi
}


# ---------------------------------------------------------------------------
# Model check — verify model files are present in repo
# ---------------------------------------------------------------------------

ensure_models() {
    local meta_file="models/breakout_cnn_best_meta.json"
    local contract_file="models/feature_contract.json"

    local missing=0
    for f in "$meta_file" "$contract_file"; do
        if [ ! -f "$f" ]; then
            err "Expected model file not found: $f"
            missing=1
        fi
    done

    if [ "$missing" -eq 1 ]; then
        err "One or more model files are missing from models/ — check your git checkout"
        exit 1
    fi

    ok "CNN model files present ($(du -h "$meta_file" | awk '{print $1}') JSON)"
}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

run_tests() {
    log "Running tests ..."
    if python -m pytest src/tests/ -x -q --tb=short; then
        ok "All tests passed"
    else
        err "Tests failed — aborting"
        exit 1
    fi
}

# ---------------------------------------------------------------------------
# Lint & Type Check
# ---------------------------------------------------------------------------

run_lint_and_typecheck() {
    log "Running Ruff linter ..."
    if python -m ruff check src/ scripts/; then
        ok "Ruff lint passed"
    else
        err "Ruff lint failed — aborting"
        exit 1
    fi

    log "Running Ruff format check ..."
    if python -m ruff format --check src/ scripts/; then
        ok "Ruff format passed"
    else
        err "Ruff format check failed — aborting"
        exit 1
    fi

    log "Running mypy type checker ..."
    if python -m mypy src scripts; then
        ok "mypy passed"
    else
        err "mypy failed — aborting"
        exit 1
    fi
}

# ---------------------------------------------------------------------------
# Tailscale IP
# ---------------------------------------------------------------------------
TAILSCALE_IP=100.69.78.116
get_tailscale_ip() {
    if command -v tailscale >/dev/null 2>&1; then
        local ip
        ip=$(tailscale ip -4)
        if [ -n "$ip" ]; then
            echo "$ip"
        else
            warn "Tailscale is running but no IP found"
        fi
    else
        warn "Tailscale CLI not found; skipping Tailscale IP display"
    fi
}

# ---------------------------------------------------------------------------
# Build the docker compose profile args string
# ---------------------------------------------------------------------------

compose_profile_args() {
    local trainer="$1"
    local monitoring="$2"
    local args=""
    if [ "$trainer" = "true" ]; then
        args="$args --profile training"
    fi
    if [ "$monitoring" = "true" ]; then
        args="$args --profile monitoring"
    fi
    echo "$args"
}

# ---------------------------------------------------------------------------
# Docker Compose
# ---------------------------------------------------------------------------

run_docker() {
    local trainer="$1"
    local monitoring="$2"

    log "Checking CNN model files ..."
    ensure_models
    echo ""

    local profile_args
    profile_args=$(compose_profile_args "$trainer" "$monitoring")

    log "Building and starting Docker Compose services ..."
    if [ -n "$profile_args" ]; then
        log "Active profiles:$([ "$trainer" = "true" ] && echo " training")$([ "$monitoring" = "true" ] && echo " monitoring")"
    fi

    # Resolve Tailscale IP (prefer live lookup, fall back to hardcoded)
    local ts_ip
    ts_ip=$(get_tailscale_ip)
    ts_ip="${ts_ip:-$TAILSCALE_IP}"

    # shellcheck disable=SC2086
    docker compose $profile_args up --build -d

    echo ""
    ok "Services are running:"
    echo "    Dashboard:   http://${ts_ip}:8180"
    echo "    Data API:    http://${ts_ip}:8100"
    echo "    Postgres:    http://${ts_ip}:5433"
    echo "    Redis:       http://${ts_ip}:6380"
    echo "    Prometheus:  http://${ts_ip}:9095"
    echo "    Grafana:     http://${ts_ip}:3010"
    if [ "$trainer" = "true" ]; then
        echo "    Trainer:     http://${ts_ip}:8200"
    fi
    echo ""
    echo "  Logs:  docker compose logs -f"
    echo "  Stop:  ./run.sh --down"
}

# ---------------------------------------------------------------------------
# Local mode
# ---------------------------------------------------------------------------

run_local() {
    ensure_venv
    ensure_env

    # Resolve Tailscale IP (prefer live lookup, fall back to hardcoded)
    local ts_ip
    ts_ip=$(get_tailscale_ip)
    ts_ip="${ts_ip:-$TAILSCALE_IP}"

    log "Starting web service locally (http://${ts_ip}:8180) ..."
    log "  (data API on http://${ts_ip}:8100)"
    DATA_SERVICE_URL="http://${ts_ip}:8100" \
    PYTHONPATH=src exec uvicorn entrypoints.web.main:app \
        --host 0.0.0.0 --port 8180 --reload
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

MONITORING="false"
TRAINER="false"

# Parse flags
POSITIONAL=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --monitoring)
            MONITORING="true"
            shift
            ;;
        --trainer)
            TRAINER="true"
            shift
            ;;
        --local)
            POSITIONAL+=("local")
            shift
            ;;
        --down)
            POSITIONAL+=("down")
            shift
            ;;
        --test)
            POSITIONAL+=("test")
            shift
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        all)
            POSITIONAL+=("all")
            shift
            ;;
        *)
            echo "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Default action is "docker"
ACTION="${POSITIONAL[0]:-docker}"

case "$ACTION" in
    local)
        ensure_models
        run_local
        ;;
    down)
        log "Stopping all Docker Compose services ..."
        docker compose --profile training --profile monitoring down
        ok "All services stopped"
        ;;
    test)
        ensure_venv
        run_lint_and_typecheck
        run_tests
        ok "All checks passed"
        ;;
    all)
        # Full pipeline + every profile: core + trainer + monitoring
        ensure_venv
        ensure_env
        echo ""
        run_lint_and_typecheck
        echo ""
        run_tests
        echo ""
        run_docker "true" "true"
        ;;
    docker)
        # Full pipeline: venv → env → lint → test → build → up
        # Profiles (trainer / monitoring) activated by flags
        ensure_venv
        ensure_env
        echo ""
        run_lint_and_typecheck
        echo ""
        run_tests
        echo ""
        run_docker "$TRAINER" "$MONITORING"
        ;;
esac
