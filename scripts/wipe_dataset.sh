#!/usr/bin/env bash
# =============================================================================
# wipe_dataset.sh — Wipe the trainer dataset Docker volume for a fresh start
# =============================================================================
#
# Usage:
#   ./scripts/wipe_dataset.sh                # wipe + confirm
#   ./scripts/wipe_dataset.sh --force        # skip confirmation prompt
#   ./scripts/wipe_dataset.sh --help         # show help
#
# What it does:
#   1. Stops the trainer container (if running)
#   2. Removes the trainer_dataset named Docker volume
#   3. Restarts the trainer container (if it was running)
#
# The next training run will regenerate the full dataset from scratch
# (load bars → simulate → render images → write CSVs).
#
# This is useful when:
#   - You want a completely fresh dataset (new sessions, new symbols, etc.)
#   - The dataset is corrupted or has stale rows from old pipeline versions
#   - You changed breakout types, session configs, or chart rendering
#   - You want to reclaim disk space (dataset images can be 5–20 GB)
#
# =============================================================================

set -euo pipefail

BOLD='\033[1m'
GREEN='\033[92m'
YELLOW='\033[93m'
RED='\033[91m'
DIM='\033[2m'
RESET='\033[0m'

# ---------------------------------------------------------------------------
# Determine compose file + volume name
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# Detect which compose file is in use
if [ -f "docker-compose.trainer.yml" ] && docker compose -f docker-compose.trainer.yml ps --quiet trainer 2>/dev/null | grep -q .; then
    COMPOSE_FILE="docker-compose.trainer.yml"
elif docker compose ps --quiet trainer 2>/dev/null | grep -q .; then
    COMPOSE_FILE="docker-compose.yml"
else
    # Default to trainer-only compose if it exists, else main
    if [ -f "docker-compose.trainer.yml" ]; then
        COMPOSE_FILE="docker-compose.trainer.yml"
    else
        COMPOSE_FILE="docker-compose.yml"
    fi
fi

# Volume name follows Docker Compose convention: {project}_{volume}
# Project name defaults to the directory name (ruby)
PROJECT_NAME=$(basename "$PROJECT_ROOT")
VOLUME_NAME="${PROJECT_NAME}_trainer_dataset"

# ---------------------------------------------------------------------------
# Help
# ---------------------------------------------------------------------------
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    echo -e "${BOLD}wipe_dataset.sh${RESET} — Wipe the trainer dataset Docker volume"
    echo ""
    echo "Usage:"
    echo "  ./scripts/wipe_dataset.sh           # wipe with confirmation"
    echo "  ./scripts/wipe_dataset.sh --force    # skip confirmation"
    echo "  ./scripts/wipe_dataset.sh --help     # this help"
    echo ""
    echo "Compose file: $COMPOSE_FILE"
    echo "Volume name:  $VOLUME_NAME"
    exit 0
fi

FORCE=0
if [[ "${1:-}" == "--force" || "${1:-}" == "-f" ]]; then
    FORCE=1
fi

# ---------------------------------------------------------------------------
# Check if volume exists
# ---------------------------------------------------------------------------
echo -e "${BOLD}Dataset Volume Wipe Tool${RESET}"
echo -e "${DIM}Project root: $PROJECT_ROOT${RESET}"
echo -e "${DIM}Compose file: $COMPOSE_FILE${RESET}"
echo -e "${DIM}Volume name:  $VOLUME_NAME${RESET}"
echo ""

if ! docker volume inspect "$VOLUME_NAME" &>/dev/null; then
    echo -e "${YELLOW}⚠ Volume '$VOLUME_NAME' does not exist — nothing to wipe.${RESET}"
    echo -e "${DIM}  It will be created automatically on the next training run.${RESET}"
    exit 0
fi

# Show volume size if possible
VOLUME_MOUNT=$(docker volume inspect "$VOLUME_NAME" --format '{{ .Mountpoint }}' 2>/dev/null || echo "")
if [ -n "$VOLUME_MOUNT" ] && [ -d "$VOLUME_MOUNT" ]; then
    VOLUME_SIZE=$(sudo du -sh "$VOLUME_MOUNT" 2>/dev/null | cut -f1 || echo "unknown")
    echo -e "  Volume size: ${BOLD}$VOLUME_SIZE${RESET}"
else
    echo -e "  Volume size: ${DIM}(cannot determine — may need sudo)${RESET}"
fi

# ---------------------------------------------------------------------------
# Confirmation
# ---------------------------------------------------------------------------
if [ "$FORCE" -eq 0 ]; then
    echo ""
    echo -e "${YELLOW}⚠ This will permanently delete ALL dataset images and CSVs.${RESET}"
    echo -e "${YELLOW}  The next training run will regenerate everything from scratch.${RESET}"
    echo ""
    read -rp "Are you sure? (y/N): " confirm
    if [[ "$confirm" != "y" && "$confirm" != "Y" ]]; then
        echo -e "${DIM}Cancelled.${RESET}"
        exit 0
    fi
fi

# ---------------------------------------------------------------------------
# Stop trainer if running
# ---------------------------------------------------------------------------
TRAINER_WAS_RUNNING=0
if docker compose -f "$COMPOSE_FILE" ps --quiet trainer 2>/dev/null | grep -q .; then
    echo -e "${DIM}Stopping and removing trainer container...${RESET}"
    docker compose -f "$COMPOSE_FILE" rm -sf trainer
    TRAINER_WAS_RUNNING=1
fi

# ---------------------------------------------------------------------------
# Remove the volume
# ---------------------------------------------------------------------------
echo -e "${DIM}Removing volume '$VOLUME_NAME'...${RESET}"
docker volume rm "$VOLUME_NAME"
echo -e "${GREEN}✓ Dataset volume wiped successfully.${RESET}"

# ---------------------------------------------------------------------------
# Restart trainer if it was running
# ---------------------------------------------------------------------------
if [ "$TRAINER_WAS_RUNNING" -eq 1 ]; then
    echo -e "${DIM}Restarting trainer container...${RESET}"
    docker compose -f "$COMPOSE_FILE" up -d trainer
    echo -e "${GREEN}✓ Trainer restarted.${RESET}"
fi

echo ""
echo -e "${GREEN}Done.${RESET} The dataset volume will be recreated on the next training run."
echo -e "${DIM}  Start a fresh pipeline:  curl -X POST http://localhost:8200/train${RESET}"
