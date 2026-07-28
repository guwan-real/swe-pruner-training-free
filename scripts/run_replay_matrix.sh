#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 METHOD INPUT_JSONL OUTPUT_ROOT [METHOD_CONFIG]" >&2
  exit 2
fi

METHOD="$1"
INPUT_PATH="$2"
OUTPUT_ROOT="$3"
METHOD_CONFIG="${4:-}"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
RATIOS=(1.0 0.7 0.5 0.35 0.25)

cd "$REPO_ROOT"
mkdir -p "$OUTPUT_ROOT"

CONFIG_ARGS=()
if [[ -n "$METHOD_CONFIG" ]]; then
  CONFIG_ARGS=(--config "$METHOD_CONFIG")
fi

for RATIO in "${RATIOS[@]}"; do
  "$PYTHON_BIN" -m tf_pruning.cli evaluate \
    --method "$METHOD" \
    "${CONFIG_ARGS[@]}" \
    --input "$INPUT_PATH" \
    --output-dir "$OUTPUT_ROOT/keep_$RATIO" \
    --keep-ratio "$RATIO" \
    --no-prune-below 0
done

"$PYTHON_BIN" -m evaluation.matrix "$OUTPUT_ROOT"
