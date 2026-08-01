#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LAUNCHER="$REPO_ROOT/scripts/run_posterior_history_swebench.sh"
STATE_FILE="${POSTERIOR_SWEEP_STATE_FILE:-$REPO_ROOT/.posterior_threshold_sweep.last}"
MODE="${1:-launch}"
if [[ $# -gt 0 ]]; then shift; fi

log() { printf '[posterior-threshold-sweep] %s\n' "$*" >&2; }
fail() { log "ERROR: $*"; exit 2; }

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_posterior_threshold_sweep.sh launch [THRESHOLD ...]
  bash scripts/run_posterior_threshold_sweep.sh [results|grade|status|stop]

With no thresholds, launch runs 1000 and then 500 sequentially. It always:
  - uses TASK_SLICE=0:20 unless explicitly overridden;
  - runs posterior_adaptive only;
  - skips baseline;
  - disables arm-level parallelism.

Optional environment:
  RUN_TAG_PREFIX=posterior_threshold_YYYYMMDD_HHMMSS
  POSTERIOR_THRESHOLDS="1000,500"
  TASK_SLICE=0:20
  GRADER_WORKERS=32
EOF
}

parse_thresholds() {
  local raw
  if [[ $# -gt 0 ]]; then
    THRESHOLDS=("$@")
  else
    raw="${POSTERIOR_THRESHOLDS:-1000,500}"
    raw="${raw//,/ }"
    read -r -a THRESHOLDS <<<"$raw"
  fi
  [[ ${#THRESHOLDS[@]} -gt 0 ]] || fail "no thresholds provided"
  local threshold
  for threshold in "${THRESHOLDS[@]}"; do
    [[ "$threshold" =~ ^[0-9]+$ ]] || fail "threshold must be a non-negative integer: $threshold"
  done
}

launch_sweep() {
  parse_thresholds "$@"
  local task_slice="${TASK_SLICE:-0:20}"
  local prefix="${RUN_TAG_PREFIX:-posterior_threshold_$(date +%Y%m%d_%H%M%S)}"
  local threshold run_tag
  : >"$STATE_FILE"
  for threshold in "${THRESHOLDS[@]}"; do
    run_tag="${prefix}_thr${threshold}"
    printf '%s\t%s\n' "$threshold" "$run_tag" >>"$STATE_FILE"
    log "launching threshold=$threshold run_tag=$run_tag task_slice=$task_slice"
    env \
      SKIP_BASELINE=1 \
      PARALLEL_ARMS=0 \
      POSTERIOR_HISTORY_METHODS=adaptive \
      POSTERIOR_MIN_INPUT_TOKENS="$threshold" \
      TASK_SLICE="$task_slice" \
      RUN_TAG="$run_tag" \
      bash "$LAUNCHER" launch
    log "completed threshold=$threshold run_tag=$run_tag"
  done
  log "sweep completed; state=$STATE_FILE"
}

for_each_recorded_run() {
  local action="$1"
  [[ -f "$STATE_FILE" ]] || fail "no recorded sweep: $STATE_FILE"
  local threshold run_tag count=0
  while IFS=$'\t' read -r threshold run_tag; do
    [[ -n "$threshold" && -n "$run_tag" ]] || continue
    count=$((count + 1))
    log "$action threshold=$threshold run_tag=$run_tag"
    env RUN_TAG="$run_tag" bash "$LAUNCHER" "$action"
  done <"$STATE_FILE"
  (( count > 0 )) || fail "recorded sweep is empty: $STATE_FILE"
}

case "$MODE" in
  launch) launch_sweep "$@" ;;
  results|grade|status|stop) for_each_recorded_run "$MODE" ;;
  help|-h|--help) usage ;;
  *) fail "unknown command: $MODE" ;;
esac
