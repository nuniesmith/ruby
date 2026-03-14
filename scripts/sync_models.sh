#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# sync_models.sh — Pull latest CNN models from the orb GitHub repo
# =============================================================================
#
# Downloads the champion model files from:
#   https://github.com/nuniesmith/futures/tree/main/models
#
# Handles Git LFS files automatically:
#   1. First tries raw.githubusercontent.com (works for non-LFS files)
#   2. If the download is a Git LFS pointer, resolves the real URL via
#      the GitHub LFS batch API and re-downloads the actual content
#
# Files synced:
#   breakout_cnn_best.pt          — PyTorch checkpoint (engine inference)
#   breakout_cnn_best_meta.json   — Model metadata (dashboard display)
#   feature_contract.json         — Feature/contract mapping
#
# Usage:
#   bash scripts/sync_models.sh              # download all model files
#   bash scripts/sync_models.sh --check      # check if models are current (no download)
#   bash scripts/sync_models.sh --pt-only    # download only the .pt file
#   bash scripts/sync_models.sh --restart    # download + restart engine container
#
# After syncing, restart the engine so it picks up the new model:
#   docker compose restart engine
#
# Environment:
#   GITHUB_TOKEN   — (optional) GitHub personal access token for private repos
#                    or to avoid rate limits.  Not needed for public repos.
#
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
MODEL_DIR="$PROJECT_ROOT/models"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
DIM='\033[2m'
NC='\033[0m'

log()  { echo -e "${CYAN}[sync]${NC} $*"; }
ok()   { echo -e "${GREEN}[  ✓ ]${NC} $*"; }
warn() { echo -e "${YELLOW}[warn]${NC} $*"; }
err()  { echo -e "${RED}[fail]${NC} $*"; }

GITHUB_REPO="nuniesmith/futures"
GITHUB_BRANCH="main"
RAW_BASE="https://raw.githubusercontent.com/${GITHUB_REPO}/${GITHUB_BRANCH}/models"
LFS_API="https://github.com/${GITHUB_REPO}.git/info/lfs/objects/batch"

# Optional auth token (for private repos or rate-limit avoidance)
GITHUB_TOKEN="${GITHUB_TOKEN:-}"

# All model files to sync
ALL_FILES=(
    "breakout_cnn_best.pt"
    "breakout_cnn_best_meta.json"
    "feature_contract.json"
)

# Lightweight files (always fetched for --check / metadata)
META_FILES=(
    "breakout_cnn_best_meta.json"
    "feature_contract.json"
)

# Minimum size (bytes) for a file to be considered "real" (not an LFS pointer)
# LFS pointers are ~130-150 bytes.  Any .pt under 1KB is suspicious.
LFS_POINTER_MAX_SIZE=1024

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

usage() {
    echo "Usage: bash scripts/sync_models.sh [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  (no args)     Download all model files from rb repo"
    echo "  --check       Check if local models are current (no download)"
    echo "  --pt-only     Download only the .pt checkpoint"
    echo "  --restart     Download all + restart engine Docker container"
    echo "  --help        Show this help message"
    echo ""
    echo "Environment:"
    echo "  GITHUB_TOKEN  Optional GitHub token for private repos / rate limits"
}

ensure_model_dir() {
    mkdir -p "$MODEL_DIR"
}

# Build curl auth header if token is set
_curl_auth_args() {
    if [ -n "$GITHUB_TOKEN" ]; then
        echo "-H" "Authorization: token ${GITHUB_TOKEN}"
    fi
}

# Check if a file is a Git LFS pointer
is_lfs_pointer() {
    local filepath="$1"
    if [ ! -f "$filepath" ]; then
        return 1
    fi
    local size
    size=$(wc -c < "$filepath" | tr -d ' ')
    if [ "$size" -gt "$LFS_POINTER_MAX_SIZE" ]; then
        return 1
    fi
    # LFS pointers start with "version https://git-lfs.github.com/spec/v1"
    if head -1 "$filepath" 2>/dev/null | grep -q "^version https://git-lfs.github.com"; then
        return 0
    fi
    return 1
}

# Parse OID and size from an LFS pointer file
parse_lfs_pointer() {
    local filepath="$1"
    LFS_OID=""
    LFS_SIZE=""

    if [ ! -f "$filepath" ]; then
        return 1
    fi

    # Extract oid (sha256:xxxx)
    LFS_OID=$(grep "^oid sha256:" "$filepath" | sed 's/^oid sha256://')
    LFS_SIZE=$(grep "^size " "$filepath" | sed 's/^size //')

    if [ -z "$LFS_OID" ] || [ -z "$LFS_SIZE" ]; then
        return 1
    fi
    return 0
}

# Resolve an LFS OID to a download URL via the GitHub LFS batch API
resolve_lfs_url() {
    local oid="$1"
    local size="$2"

    # Build the JSON payload for the LFS batch API
    local payload
    payload=$(cat <<EOF
{
  "operation": "download",
  "transfer": ["basic"],
  "objects": [
    {
      "oid": "${oid}",
      "size": ${size}
    }
  ]
}
EOF
)

    local auth_args=()
    if [ -n "$GITHUB_TOKEN" ]; then
        auth_args=(-H "Authorization: token ${GITHUB_TOKEN}")
    fi

    local response
    response=$(curl -sS \
        -X POST \
        -H "Accept: application/vnd.git-lfs+json" \
        -H "Content-Type: application/vnd.git-lfs+json" \
        "${auth_args[@]+"${auth_args[@]}"}" \
        -d "$payload" \
        "$LFS_API" 2>&1)

    if [ $? -ne 0 ]; then
        err "LFS batch API request failed"
        return 1
    fi

    # Parse the download URL from the response using python (always available
    # in our environment) or fall back to grep/sed for minimal envs
    local download_url=""
    if command -v python3 >/dev/null 2>&1; then
        download_url=$(python3 -c "
import json, sys
try:
    data = json.loads(sys.stdin.read())
    obj = data.get('objects', [{}])[0]
    err = obj.get('error')
    if err:
        print('ERROR:' + err.get('message', 'unknown'), file=sys.stderr)
        sys.exit(1)
    actions = obj.get('actions', {})
    dl = actions.get('download', {})
    href = dl.get('href', '')
    if not href:
        print('ERROR:no download href in LFS response', file=sys.stderr)
        sys.exit(1)
    print(href)
except Exception as e:
    print(f'ERROR:{e}', file=sys.stderr)
    sys.exit(1)
" <<< "$response" 2>&1)
    else
        # Minimal fallback: extract href with grep
        download_url=$(echo "$response" | grep -o '"href":"[^"]*"' | head -1 | sed 's/"href":"//;s/"$//')
    fi

    if [ -z "$download_url" ] || echo "$download_url" | grep -q "^ERROR:"; then
        err "Failed to resolve LFS URL: ${download_url}"
        return 1
    fi

    # Also extract any required headers (GitHub LFS may require Authorization header)
    local lfs_headers=""
    if command -v python3 >/dev/null 2>&1; then
        lfs_headers=$(python3 -c "
import json, sys
try:
    data = json.loads(sys.stdin.read())
    obj = data.get('objects', [{}])[0]
    actions = obj.get('actions', {})
    dl = actions.get('download', {})
    headers = dl.get('header', {})
    for k, v in headers.items():
        print(f'{k}: {v}')
except Exception:
    pass
" <<< "$response" 2>/dev/null) || true
    fi

    LFS_DOWNLOAD_URL="$download_url"
    LFS_DOWNLOAD_HEADERS="$lfs_headers"
    return 0
}

# Format bytes into human-readable size
format_size() {
    local bytes="$1"
    if [ "$bytes" -ge 1073741824 ]; then
        echo "$(echo "scale=1; $bytes / 1073741824" | bc)G"
    elif [ "$bytes" -ge 1048576 ]; then
        echo "$(echo "scale=1; $bytes / 1048576" | bc)M"
    elif [ "$bytes" -ge 1024 ]; then
        echo "$(echo "scale=0; $bytes / 1024" | bc)K"
    else
        echo "${bytes}B"
    fi
}

download_file() {
    local filename="$1"
    local url="${RAW_BASE}/${filename}"
    local dest="${MODEL_DIR}/${filename}"
    local tmp="${dest}.tmp"

    local auth_args=()
    if [ -n "$GITHUB_TOKEN" ]; then
        auth_args=(-H "Authorization: token ${GITHUB_TOKEN}")
    fi

    log "Downloading ${filename}..."

    # Step 1: Download from raw.githubusercontent.com
    if ! curl -fSL --progress-bar \
        "${auth_args[@]+"${auth_args[@]}"}" \
        -o "$tmp" "$url" 2>&1; then
        rm -f "$tmp"
        err "Failed to download ${filename} from ${url}"
        return 1
    fi

    # Step 2: Check if this is an LFS pointer
    if is_lfs_pointer "$tmp"; then
        log "  → Detected Git LFS pointer, resolving actual file..."

        if ! parse_lfs_pointer "$tmp"; then
            rm -f "$tmp"
            err "Failed to parse LFS pointer for ${filename}"
            return 1
        fi

        local expected_size
        expected_size=$(format_size "$LFS_SIZE")
        log "  → LFS object: ${LFS_OID:0:12}... (${expected_size})"

        # Resolve the real download URL via LFS batch API
        if ! resolve_lfs_url "$LFS_OID" "$LFS_SIZE"; then
            rm -f "$tmp"
            err "Failed to resolve LFS download URL for ${filename}"
            return 1
        fi

        # Build header args for the LFS download
        local lfs_curl_args=()
        if [ -n "${LFS_DOWNLOAD_HEADERS:-}" ]; then
            while IFS= read -r header_line; do
                if [ -n "$header_line" ]; then
                    lfs_curl_args+=(-H "$header_line")
                fi
            done <<< "$LFS_DOWNLOAD_HEADERS"
        fi

        # Download the actual file from LFS storage
        log "  → Downloading from LFS storage..."
        if ! curl -fSL --progress-bar \
            "${lfs_curl_args[@]+"${lfs_curl_args[@]}"}" \
            -o "$tmp" "$LFS_DOWNLOAD_URL" 2>&1; then
            rm -f "$tmp"
            err "Failed to download LFS content for ${filename}"
            return 1
        fi

        # Verify the downloaded size matches the LFS pointer
        local actual_size
        actual_size=$(wc -c < "$tmp" | tr -d ' ')
        if [ "$actual_size" -ne "$LFS_SIZE" ]; then
            rm -f "$tmp"
            err "Size mismatch for ${filename}: expected ${LFS_SIZE} bytes, got ${actual_size}"
            return 1
        fi

        # Verify SHA256 if sha256sum is available
        if command -v sha256sum >/dev/null 2>&1; then
            local actual_hash
            actual_hash=$(sha256sum "$tmp" | awk '{print $1}')
            if [ "$actual_hash" != "$LFS_OID" ]; then
                rm -f "$tmp"
                err "SHA256 mismatch for ${filename}: expected ${LFS_OID:0:16}..., got ${actual_hash:0:16}..."
                return 1
            fi
            log "  → SHA256 verified ✓"
        elif command -v shasum >/dev/null 2>&1; then
            local actual_hash
            actual_hash=$(shasum -a 256 "$tmp" | awk '{print $1}')
            if [ "$actual_hash" != "$LFS_OID" ]; then
                rm -f "$tmp"
                err "SHA256 mismatch for ${filename}: expected ${LFS_OID:0:16}..., got ${actual_hash:0:16}..."
                return 1
            fi
            log "  → SHA256 verified ✓"
        fi
    fi

    # Move the verified file into place
    mv "$tmp" "$dest"
    local size
    size=$(du -h "$dest" 2>/dev/null | awk '{print $1}')
    ok "${filename} (${size})"
    return 0
}

check_file() {
    local filename="$1"
    local dest="${MODEL_DIR}/${filename}"

    if [ -f "$dest" ]; then
        local size modified
        size=$(du -h "$dest" 2>/dev/null | awk '{print $1}')
        modified=$(date -r "$dest" "+%Y-%m-%d %H:%M" 2>/dev/null || stat -c '%y' "$dest" 2>/dev/null | cut -d. -f1 || echo "unknown")

        # Check if the existing file is an LFS pointer (broken sync)
        if is_lfs_pointer "$dest"; then
            warn "${filename}  ${DIM}(LFS pointer — not actual model! Re-run sync)${NC}"
            return 1
        fi

        ok "${filename}  ${DIM}(${size}, modified ${modified})${NC}"
        return 0
    else
        warn "${filename}  — not found locally"
        return 1
    fi
}

# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

cmd_check() {
    log "Checking local model files in ${MODEL_DIR}/..."
    echo ""
    local missing=0
    for f in "${ALL_FILES[@]}"; do
        if ! check_file "$f"; then
            missing=$((missing + 1))
        fi
    done
    echo ""
    if [ "$missing" -gt 0 ]; then
        warn "${missing} file(s) missing or invalid — run: bash scripts/sync_models.sh"
        return 1
    else
        ok "All model files present and valid"
        # Show meta info if available
        local meta="${MODEL_DIR}/breakout_cnn_best_meta.json"
        if [ -f "$meta" ] && command -v python3 >/dev/null 2>&1; then
            echo ""
            echo -e "${DIM}$(python3 -c "
import json, sys
try:
    d = json.load(open('$meta'))
    acc = d.get('val_accuracy', d.get('accuracy', '?'))
    prec = d.get('val_precision', d.get('precision', '?'))
    rec = d.get('val_recall', d.get('recall', '?'))
    ep  = d.get('epochs_trained', '?')
    print(f'  Champion: acc={acc}%  precision={prec}%  recall={rec}%  epochs={ep}')
except Exception:
    pass
" 2>/dev/null)${NC}"
        fi

        # Show .pt file size to confirm it's a real model
        local pt="${MODEL_DIR}/breakout_cnn_best.pt"
        if [ -f "$pt" ]; then
            local pt_size
            pt_size=$(wc -c < "$pt" | tr -d ' ')
            local pt_human
            pt_human=$(du -h "$pt" 2>/dev/null | awk '{print $1}')
            if [ "$pt_size" -lt 1000000 ]; then
                warn "  .pt file is only ${pt_human} — may not be a real model"
            else
                echo -e "${DIM}  Model file: ${pt_human}${NC}"
            fi
        fi
        return 0
    fi
}

cmd_download() {
    local files=("$@")
    ensure_model_dir

    log "Pulling models from github.com/${GITHUB_REPO} (branch: ${GITHUB_BRANCH})..."
    if [ -n "$GITHUB_TOKEN" ]; then
        log "  Using GITHUB_TOKEN for authentication"
    fi
    echo ""

    local failed=0
    for f in "${files[@]}"; do
        if ! download_file "$f"; then
            failed=$((failed + 1))
        fi
    done

    echo ""
    if [ "$failed" -gt 0 ]; then
        err "${failed} file(s) failed to download"
        echo ""
        echo -e "  ${DIM}If downloads fail, try setting GITHUB_TOKEN:${NC}"
        echo "    export GITHUB_TOKEN=ghp_your_token_here"
        echo "    bash scripts/sync_models.sh"
        return 1
    else
        ok "All model files synced to ${MODEL_DIR}/"
        echo ""

        # Verify the .pt file is a real model, not an LFS pointer
        local pt="${MODEL_DIR}/breakout_cnn_best.pt"
        if [ -f "$pt" ] && is_lfs_pointer "$pt"; then
            err "WARNING: .pt file is still an LFS pointer — model not usable"
            echo ""
            echo -e "  ${DIM}The repo uses Git LFS. Try cloning the rb repo directly:${NC}"
            echo "    git clone https://github.com/${GITHUB_REPO}.git /tmp/orb"
            echo "    cp /tmp/orb/models/breakout_cnn_best.pt ${MODEL_DIR}/"
            return 1
        fi

        echo -e "  ${DIM}Restart the engine to pick up the new model:${NC}"
        echo "    docker compose restart engine"
        return 0
    fi
}

cmd_restart() {
    cmd_download "${ALL_FILES[@]}" || return 1
    echo ""
    log "Restarting engine container..."
    (cd "$PROJECT_ROOT" && docker compose restart engine)
    ok "Engine restarted with new model"
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

ACTION="download_all"
for arg in "$@"; do
    case "$arg" in
        --check)     ACTION="check" ;;
        --pt-only)   ACTION="pt_only" ;;
        --restart)   ACTION="restart" ;;
        --help|-h)   usage; exit 0 ;;
        *)
            err "Unknown option: $arg"
            usage
            exit 1
            ;;
    esac
done

case "$ACTION" in
    check)
        cmd_check
        ;;
    download_all)
        cmd_download "${ALL_FILES[@]}"
        ;;
    pt_only)
        cmd_download "breakout_cnn_best.pt" "${META_FILES[@]}"
        ;;
    restart)
        cmd_restart
        ;;
esac
