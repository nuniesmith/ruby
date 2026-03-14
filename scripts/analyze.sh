#!/bin/bash
# analyze.sh
# Usage: ./analyze.sh <directory_path> [--full]

TARGET_DIR=""
FULL_MODE=false
OUTPUT_FILE="codebase_summary.md"

# --- 1. Argument Parsing ---
for arg in "$@"; do
    if [[ "$arg" == "--full" ]]; then
        FULL_MODE=true
    elif [[ -d "$arg" ]]; then
        TARGET_DIR="$arg"
    else
        echo "Error: '$arg' is not a valid directory."
        echo "Usage: $0 <directory_path> [--full]"
        exit 1
    fi
done

if [[ -z "$TARGET_DIR" ]]; then
    echo "Usage: $0 <directory_path> [--full]"
    exit 1
fi

# --- 2. Directories to exclude ---
# Python:  venv, .venv, __pycache__, *.pyc
# Rust:    target
# Node/JS: node_modules, dist, .npm
# Kotlin:  build, .gradle, .kotlin, kotlin-js-store
# IDE/VCS: .git, .idea, .vscode
PRUNE_DIRS=(
    ".git"
    "__pycache__"
    "target"
    "build"
    "venv"
    ".venv"
    "dist"
    ".idea"
    ".vscode"
    ".gradle"
    ".kotlin"
    "kotlin-js-store"
    "node_modules"
    ".npm"
)

# Build the find prune args as a proper array (no eval needed)
PRUNE_ARGS=()
for i in "${!PRUNE_DIRS[@]}"; do
    if [[ $i -gt 0 ]]; then
        PRUNE_ARGS+=("-o")
    fi
    PRUNE_ARGS+=("-name" "${PRUNE_DIRS[$i]}")
done

# Helper: run find with standard prune logic, pass extra args after --
# Usage: pruned_find [-print0] [extra find expressions...]
pruned_find() {
    find "$TARGET_DIR" -type d \( "${PRUNE_ARGS[@]}" \) -prune -o "$@"
}

# Build the tree exclude pattern (pipe-separated)
TREE_EXCLUDES=$(IFS="|"; echo "${PRUNE_DIRS[*]}")
TREE_EXCLUDES="${TREE_EXCLUDES}|*.class|*.jar|*.o|*.so|*.pyc|*.pyo"

# --- 3. Setup Output ---
echo "# Codebase Analysis: $TARGET_DIR" > "$OUTPUT_FILE"
echo "Generated on: $(date)" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"

# --- 4. Statistics & Health Checks ---
echo "Collecting statistics..."
echo "## Summary Statistics" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"

# Count files and directories (excluding pruned dirs)
TOTAL_FILES=$(pruned_find -type f -print | wc -l)
TOTAL_DIRS=$(pruned_find -type d -print | wc -l)
TOTAL_SIZE=$(pruned_find -type f -print0 | xargs -0 du -ch 2>/dev/null | tail -1 | awk '{print $1}')

echo "| Metric | Count |" >> "$OUTPUT_FILE"
echo "|--------|-------|" >> "$OUTPUT_FILE"
echo "| Total files | $TOTAL_FILES |" >> "$OUTPUT_FILE"
echo "| Total directories | $TOTAL_DIRS |" >> "$OUTPUT_FILE"
echo "| Total size (source) | ${TOTAL_SIZE:-N/A} |" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"

# File count by extension
echo "### Files by Extension" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"
echo "| Extension | Count |" >> "$OUTPUT_FILE"
echo "|-----------|-------|" >> "$OUTPUT_FILE"
pruned_find -type f -print \
    | sed 's/.*\.//' \
    | grep -v '/' \
    | sort \
    | uniq -c \
    | sort -rn \
    | head -25 \
    | while read -r count ext; do
        echo "| .$ext | $count |" >> "$OUTPUT_FILE"
    done
echo "" >> "$OUTPUT_FILE"

# Lines of code by extension (top languages)
echo "### Lines of Code (Top 15)" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"
echo "| Extension | Lines |" >> "$OUTPUT_FILE"
echo "|-----------|-------|" >> "$OUTPUT_FILE"
pruned_find -type f \( \
    -name "*.rs" -o -name "*.kt" -o -name "*.kts" -o \
    -name "*.java" -o -name "*.py" -o \
    -name "*.js" -o -name "*.ts" -o -name "*.jsx" -o -name "*.tsx" -o \
    -name "*.html" -o -name "*.css" -o -name "*.scss" -o \
    -name "*.sh" -o -name "*.bash" -o \
    -name "*.toml" -o -name "*.yaml" -o -name "*.yml" -o -name "*.json" -o \
    -name "*.xml" -o -name "*.sql" -o -name "*.md" -o \
    -name "*.c" -o -name "*.cpp" -o -name "*.h" -o \
    -name "*.go" -o -name "*.rb" -o -name "*.swift" \
    \) -print \
    | while read -r f; do
        ext="${f##*.}"
        lines=$(wc -l < "$f" 2>/dev/null || echo 0)
        echo "$ext $lines"
    done \
    | awk '{a[$1]+=$2} END {for(k in a) print a[k], k}' \
    | sort -rn \
    | head -15 \
    | while read -r lines ext; do
        printf "| .%-9s | %d |\n" "$ext" "$lines" >> "$OUTPUT_FILE"
    done
echo "" >> "$OUTPUT_FILE"

# --- 5. Health Checks / Potential Issues ---
echo "### Potential Issues" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"

ISSUES_FOUND=false

# Large files (>500KB source files)
LARGE_FILES=$(pruned_find -type f -size +500k -print 2>/dev/null)
if [[ -n "$LARGE_FILES" ]]; then
    ISSUES_FOUND=true
    echo "**Large files (>500KB):** These may be binaries, data files, or generated assets that should be gitignored." >> "$OUTPUT_FILE"
    echo "" >> "$OUTPUT_FILE"
    echo '```' >> "$OUTPUT_FILE"
    while IFS= read -r f; do
        size=$(du -h "$f" 2>/dev/null | awk '{print $1}')
        echo "  $size  $f"
    done <<< "$LARGE_FILES" >> "$OUTPUT_FILE"
    echo '```' >> "$OUTPUT_FILE"
    echo "" >> "$OUTPUT_FILE"
fi

# Empty directories
EMPTY_DIRS=$(pruned_find -type d -empty -print 2>/dev/null)
if [[ -n "$EMPTY_DIRS" ]]; then
    ISSUES_FOUND=true
    echo "**Empty directories:** May indicate incomplete setup or leftover scaffolding." >> "$OUTPUT_FILE"
    echo "" >> "$OUTPUT_FILE"
    echo '```' >> "$OUTPUT_FILE"
    echo "$EMPTY_DIRS" >> "$OUTPUT_FILE"
    echo '```' >> "$OUTPUT_FILE"
    echo "" >> "$OUTPUT_FILE"
fi

# Duplicate filenames (same name in different dirs — can cause confusion)
DUPES=$(pruned_find -type f -print \
    | xargs -I{} basename {} \
    | sort \
    | uniq -cd \
    | sort -rn \
    | head -10)
if [[ -n "$DUPES" ]]; then
    ISSUES_FOUND=true
    echo "**Duplicate filenames (top 10):** Same filename in multiple directories — may cause import confusion." >> "$OUTPUT_FILE"
    echo "" >> "$OUTPUT_FILE"
    echo '```' >> "$OUTPUT_FILE"
    echo "$DUPES" >> "$OUTPUT_FILE"
    echo '```' >> "$OUTPUT_FILE"
    echo "" >> "$OUTPUT_FILE"
fi

# Files with no extension (excluding well-known convention files)
KNOWN_NO_EXT="Makefile|Dockerfile|Procfile|Gemfile|Rakefile|Vagrantfile|LICENSE|README|CHANGELOG|CODEOWNERS|Justfile|Taskfile"
NO_EXT_FILES=$(pruned_find -type f ! -name "*.*" -print 2>/dev/null \
    | grep -v -E "(${KNOWN_NO_EXT})")
if [[ -n "$NO_EXT_FILES" ]]; then
    NO_EXT_COUNT=$(echo "$NO_EXT_FILES" | wc -l)
    ISSUES_FOUND=true
    echo "**Files without extensions:** $NO_EXT_COUNT file(s) with no extension (excluding Makefile, Dockerfile, etc)." >> "$OUTPUT_FILE"
    echo "" >> "$OUTPUT_FILE"
    echo '```' >> "$OUTPUT_FILE"
    echo "$NO_EXT_FILES" | head -15 >> "$OUTPUT_FILE"
    echo '```' >> "$OUTPUT_FILE"
    echo "" >> "$OUTPUT_FILE"
fi

# TODO/FIXME/HACK count
TODO_FILES=$(pruned_find -type f -print \
    | xargs grep -rl 'TODO\|FIXME\|HACK\|XXX' 2>/dev/null)
if [[ -n "$TODO_FILES" ]]; then
    TODO_COUNT=$(echo "$TODO_FILES" | wc -l)
    ISSUES_FOUND=true
    echo "**TODO/FIXME/HACK markers:** Found in $TODO_COUNT file(s)." >> "$OUTPUT_FILE"
    echo "" >> "$OUTPUT_FILE"
    echo '```' >> "$OUTPUT_FILE"
    pruned_find -type f -print \
        | xargs grep -rn 'TODO\|FIXME\|HACK\|XXX' 2>/dev/null \
        | head -25 >> "$OUTPUT_FILE"
    echo '```' >> "$OUTPUT_FILE"
    echo "" >> "$OUTPUT_FILE"
fi

# Excluded directories that actually exist (confirms they're being skipped)
echo "### Excluded Directories Found" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"
EXCLUDED_FOUND=false
for dir in "${PRUNE_DIRS[@]}"; do
    matches=$(find "$TARGET_DIR" -type d -name "$dir" 2>/dev/null)
    if [[ -n "$matches" ]]; then
        EXCLUDED_FOUND=true
        while IFS= read -r match; do
            size=$(du -sh "$match" 2>/dev/null | awk '{print $1}')
            echo "- **$dir** — ${size:-?} at \`$match\`" >> "$OUTPUT_FILE"
        done <<< "$matches"
    fi
done
if [ "$EXCLUDED_FOUND" = false ]; then
    echo "None of the excluded directories were found in this project." >> "$OUTPUT_FILE"
fi
echo "" >> "$OUTPUT_FILE"

if [ "$ISSUES_FOUND" = false ]; then
    echo "No issues detected. 🎉" >> "$OUTPUT_FILE"
    echo "" >> "$OUTPUT_FILE"
fi

# --- 6. Generate Directory Tree ---
echo "## Project Structure" >> "$OUTPUT_FILE"
echo '```' >> "$OUTPUT_FILE"

if command -v tree &> /dev/null; then
    tree "$TARGET_DIR" -I "$TREE_EXCLUDES" --dirsfirst >> "$OUTPUT_FILE"
else
    # Fallback with exclusions applied
    echo "('tree' command not found, using 'find' fallback)" >> "$OUTPUT_FILE"
    FIND_PATH_EXCLUDES=()
    for dir in "${PRUNE_DIRS[@]}"; do
        FIND_PATH_EXCLUDES+=(-not -path "*/${dir}/*" -not -path "*/${dir}")
    done
    find "$TARGET_DIR" -maxdepth 4 "${FIND_PATH_EXCLUDES[@]}" >> "$OUTPUT_FILE"
fi

echo '```' >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"

# --- 7. Generate File Contents (Only if --full) ---
if [ "$FULL_MODE" = true ]; then
    echo "Processing file contents..."
    echo "## File Contents" >> "$OUTPUT_FILE"

    pruned_find -type f -print | while read -r file; do

        # Skip common binary extensions explicitly
        case "$file" in
            *.class|*.jar|*.o|*.so|*.pyc|*.pyo|*.wasm|*.png|*.jpg|*.jpeg|*.gif|*.ico|*.pdf|*.zip|*.tar|*.gz|*.lock)
                continue
                ;;
        esac

        # Check if file is text (skips remaining binaries/images)
        if grep -Iq . "$file" 2>/dev/null; then
            echo "### File: $file" >> "$OUTPUT_FILE"

            ext="${file##*.}"

            echo '```'"$ext" >> "$OUTPUT_FILE"
            cat "$file" >> "$OUTPUT_FILE"
            echo "" >> "$OUTPUT_FILE"
            echo '```' >> "$OUTPUT_FILE"
            echo "---" >> "$OUTPUT_FILE"
        fi
    done
fi

echo "✅ Analysis complete. Output saved to: $OUTPUT_FILE"
