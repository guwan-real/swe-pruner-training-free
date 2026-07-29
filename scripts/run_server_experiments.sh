#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

ENV_NAME="${ENV_NAME:-swepruner-training-free}"
BASE_DIR="${BASE_DIR:-/home/yuantao/futao}"
WORK_DIR="${WORK_DIR:-$BASE_DIR/swepruner_training_free_workspace}"
OLD_PROJECT_DIR="${OLD_PROJECT_DIR:-$BASE_DIR/swepruner_workspace/swepruner-structured-training}"
LIMIT="${LIMIT:-200}"
PPL_GPU="${PPL_GPU:-0}"
INFLUENCE_GPU="${INFLUENCE_GPU:-1}"
LAST_RUN_FILE="$WORK_DIR/.last_training_free_run"

MODE="launch"
DRY_RUN=0

log() {
  printf '[server-experiments] %s\n' "$*" >&2
}

fail() {
  log "ERROR: $*"
  exit 2
}

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_server_experiments.sh [launch|smoke|status|results] [--dry-run]

Commands:
  launch   Default. Prepare up to 200 replay rows and start parallel experiments.
  smoke    Run the bundled two-row IR and AST matrices in the foreground.
  status   Show the latest run's process state and recent log lines.
  results  Print every completed matrix.csv from the latest run.

Zero-configuration launch always starts:
  - ir_structural on CPU
  - execution_ast on CPU

Optional experiments are discovered automatically:
  - conditional_ppl when WORK_DIR/local_configs/conditional_ppl.json exists
  - hidden_state_similarity when WORK_DIR/replay/hidden_signals.jsonl exists
  - attention_rollout when WORK_DIR/replay/attention_signals.jsonl exists
  - influence_oracle when both its local config and oracle_50.jsonl exist

Environment overrides:
  ENV_NAME, BASE_DIR, WORK_DIR, OLD_PROJECT_DIR, LIMIT, REPLAY_PATH,
  RUN_TAG, PPL_GPU, INFLUENCE_GPU, PPL_CONFIG, INFLUENCE_CONFIG,
  HIDDEN_REPLAY, ATTENTION_REPLAY, INFLUENCE_REPLAY.

The script removes an active uv/venv from its child-process environment before
activating conda. It does not change the parent terminal after it exits.
EOF
}

if [[ $# -gt 0 && "$1" != --* ]]; then
  MODE="$1"
  shift
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN=1
      ;;
    -h | --help)
      usage
      exit 0
      ;;
    *)
      fail "unknown argument: $1"
      ;;
  esac
  shift
done

case "$MODE" in
  launch | smoke | status | results) ;;
  help)
    usage
    exit 0
    ;;
  *)
    fail "unknown command: $MODE"
    ;;
esac

remove_path_entry() {
  local target="$1"
  local entry
  local -a entries=()
  local -a kept=()

  IFS=':' read -r -a entries <<<"${PATH:-}"
  for entry in "${entries[@]}"; do
    if [[ "$entry" != "$target" ]]; then
      kept+=("$entry")
    fi
  done
  local IFS=':'
  PATH="${kept[*]}"
  export PATH
}

disable_uv_or_venv() {
  local active_venv="${VIRTUAL_ENV:-}"
  if [[ -z "$active_venv" ]]; then
    log "no active uv/venv detected"
    return
  fi

  remove_path_entry "$active_venv/bin"
  unset VIRTUAL_ENV
  unset VIRTUAL_ENV_PROMPT
  unset UV_ACTIVE
  unset UV_PROJECT_ENVIRONMENT
  unset _OLD_VIRTUAL_PATH
  unset _OLD_VIRTUAL_PS1
  hash -r
  log "disabled active uv/venv: $active_venv"
}

find_conda_base() {
  local candidate
  if command -v conda >/dev/null 2>&1; then
    conda info --base
    return
  fi
  for candidate in \
    "${CONDA_PREFIX_BASE:-}" \
    "$HOME/miniconda3" \
    "$HOME/anaconda3" \
    "/opt/conda" \
    "/root/anaconda3"; do
    if [[ -n "$candidate" && -x "$candidate/bin/conda" ]]; then
      "$candidate/bin/conda" info --base
      return
    fi
  done
  return 1
}

activate_runtime() {
  disable_uv_or_venv

  if [[ "${SKIP_CONDA:-0}" == "1" ]]; then
    PYTHON_BIN="${PYTHON_BIN:-$(command -v python3 || true)}"
    [[ -n "$PYTHON_BIN" ]] || fail "python3 was not found"
    log "SKIP_CONDA=1; using $PYTHON_BIN"
  else
    local conda_base
    conda_base="$(find_conda_base)" || fail "conda was not found"
    # shellcheck source=/dev/null
    source "$conda_base/etc/profile.d/conda.sh"
    if ! conda activate "$ENV_NAME"; then
      fail "conda env '$ENV_NAME' is missing; run scripts/create_server_conda.sh first"
    fi
    PYTHON_BIN="$CONDA_PREFIX/bin/python"
    log "activated conda env: $ENV_NAME ($PYTHON_BIN)"
  fi

  cd "$REPO_ROOT"
  "$PYTHON_BIN" -c \
    'import sys, tf_pruning; assert sys.version_info >= (3, 11), sys.version'
  log "repository import check passed"
}

RESOLVED_REPLAY=""

resolve_replay() {
  local cached_replay="$WORK_DIR/replay/replay_${LIMIT}.jsonl"
  local old_data="$OLD_PROJECT_DIR/artifacts/combined_2k/pruning_sft.jsonl"

  if [[ -n "${REPLAY_PATH:-}" ]]; then
    [[ -f "$REPLAY_PATH" ]] || fail "REPLAY_PATH does not exist: $REPLAY_PATH"
    RESOLVED_REPLAY="$REPLAY_PATH"
    log "using explicit replay: $RESOLVED_REPLAY"
    return
  fi

  if [[ -f "$cached_replay" ]]; then
    RESOLVED_REPLAY="$cached_replay"
    log "using cached replay: $RESOLVED_REPLAY"
    return
  fi

  if [[ -f "$old_data" ]]; then
    RESOLVED_REPLAY="$cached_replay"
    if [[ "$DRY_RUN" == "1" ]]; then
      log "dry-run: would convert $old_data to $RESOLVED_REPLAY"
      return
    fi
    mkdir -p "$(dirname "$cached_replay")"
    "$PYTHON_BIN" -m evaluation.convert_existing \
      --input "$old_data" \
      --output "$cached_replay" \
      --limit "$LIMIT" \
      --required-confidence 0.9
    log "converted replay: $RESOLVED_REPLAY"
    return
  fi

  RESOLVED_REPLAY="$REPO_ROOT/examples/replay/demo.jsonl"
  log "WARNING: old replay source was not found: $old_data"
  log "falling back to bundled demo replay: $RESOLVED_REPLAY"
}

resolve_latest_tag() {
  local requested_tag="${RUN_TAG:-}"
  if [[ -n "$requested_tag" ]]; then
    printf '%s\n' "$requested_tag"
    return
  fi
  [[ -f "$LAST_RUN_FILE" ]] || fail "no previous run found at $LAST_RUN_FILE"
  local stored_tag
  stored_tag="$(head -n 1 "$LAST_RUN_FILE")"
  [[ -n "$stored_tag" ]] || fail "$LAST_RUN_FILE is empty"
  printf '%s\n' "$stored_tag"
}

launch_method() {
  local method="$1"
  local input_path="$2"
  local config_path="$3"
  local visible_gpus="$4"
  local output_dir="$RUN_ROOT/$method"
  local log_path="$LOG_ROOT/$method.log"
  local pid_path="$LOG_ROOT/$method.pid"
  local -a config_args=()

  if [[ -n "$config_path" ]]; then
    config_args=("$config_path")
  fi

  nohup env \
    "CUDA_VISIBLE_DEVICES=$visible_gpus" \
    PYTHONUNBUFFERED=1 \
    "PYTHON_BIN=$PYTHON_BIN" \
    bash "$REPO_ROOT/scripts/run_replay_matrix.sh" \
    "$method" \
    "$input_path" \
    "$output_dir" \
    "${config_args[@]}" \
    >"$log_path" 2>&1 &

  local pid=$!
  printf '%s\n' "$pid" >"$pid_path"
  log "started $method pid=$pid gpu='${visible_gpus:-CPU}'"
  log "  log: $log_path"
}

print_optional_plan() {
  local ppl_config="${PPL_CONFIG:-$WORK_DIR/local_configs/conditional_ppl.json}"
  local hidden_replay="${HIDDEN_REPLAY:-$WORK_DIR/replay/hidden_signals.jsonl}"
  local attention_replay="${ATTENTION_REPLAY:-$WORK_DIR/replay/attention_signals.jsonl}"
  local influence_config="${INFLUENCE_CONFIG:-$WORK_DIR/local_configs/influence_oracle.json}"
  local influence_replay="${INFLUENCE_REPLAY:-$WORK_DIR/replay/oracle_50.jsonl}"

  [[ -f "$ppl_config" ]] \
    && log "optional: conditional_ppl ready on GPU $PPL_GPU" \
    || log "optional: conditional_ppl skipped (missing $ppl_config)"
  [[ -f "$hidden_replay" ]] \
    && log "optional: hidden_state_similarity ready on CPU" \
    || log "optional: hidden_state_similarity skipped (missing $hidden_replay)"
  [[ -f "$attention_replay" ]] \
    && log "optional: attention_rollout ready on CPU" \
    || log "optional: attention_rollout skipped (missing $attention_replay)"
  if [[ -f "$influence_config" && -f "$influence_replay" ]]; then
    log "optional: influence_oracle ready on GPU $INFLUENCE_GPU"
  else
    log "optional: influence_oracle skipped (needs config and oracle replay)"
  fi
}

launch_experiments() {
  activate_runtime
  resolve_replay

  local run_tag="${RUN_TAG:-baseline_${LIMIT}_$(date +%Y%m%d_%H%M%S)}"
  RUN_ROOT="$WORK_DIR/runs/$run_tag"
  LOG_ROOT="$WORK_DIR/logs/$run_tag"

  log "mode=launch"
  log "replay=$RESOLVED_REPLAY"
  log "run_root=$RUN_ROOT"
  print_optional_plan

  if [[ "$DRY_RUN" == "1" ]]; then
    log "dry-run complete; no directories or processes were created"
    return
  fi

  [[ ! -e "$RUN_ROOT" ]] || fail "run directory already exists: $RUN_ROOT"
  [[ ! -e "$LOG_ROOT" ]] || fail "log directory already exists: $LOG_ROOT"
  mkdir -p "$RUN_ROOT" "$LOG_ROOT" "$WORK_DIR"
  printf '%s\n' "$run_tag" >"$LAST_RUN_FILE"

  launch_method \
    ir_structural \
    "$RESOLVED_REPLAY" \
    "$REPO_ROOT/tasks/ir_structural/config.example.json" \
    ""
  launch_method \
    execution_ast \
    "$RESOLVED_REPLAY" \
    "$REPO_ROOT/tasks/execution_ast/config.example.json" \
    ""

  local ppl_config="${PPL_CONFIG:-$WORK_DIR/local_configs/conditional_ppl.json}"
  if [[ -f "$ppl_config" ]]; then
    launch_method conditional_ppl "$RESOLVED_REPLAY" "$ppl_config" "$PPL_GPU"
  fi

  local hidden_replay="${HIDDEN_REPLAY:-$WORK_DIR/replay/hidden_signals.jsonl}"
  if [[ -f "$hidden_replay" ]]; then
    launch_method \
      hidden_state_similarity \
      "$hidden_replay" \
      "$REPO_ROOT/tasks/hidden_state_similarity/config.example.json" \
      ""
  fi

  local attention_replay="${ATTENTION_REPLAY:-$WORK_DIR/replay/attention_signals.jsonl}"
  if [[ -f "$attention_replay" ]]; then
    launch_method \
      attention_rollout \
      "$attention_replay" \
      "$REPO_ROOT/tasks/attention_rollout/config.example.json" \
      ""
  fi

  local influence_config="${INFLUENCE_CONFIG:-$WORK_DIR/local_configs/influence_oracle.json}"
  local influence_replay="${INFLUENCE_REPLAY:-$WORK_DIR/replay/oracle_50.jsonl}"
  if [[ -f "$influence_config" && -f "$influence_replay" ]]; then
    launch_method \
      influence_oracle \
      "$influence_replay" \
      "$influence_config" \
      "$INFLUENCE_GPU"
  fi

  log "parallel launch complete"
  log "check status: bash scripts/run_server_experiments.sh status"
  log "show results: bash scripts/run_server_experiments.sh results"
}

run_smoke() {
  activate_runtime

  local run_tag="${RUN_TAG:-smoke_$(date +%Y%m%d_%H%M%S)}"
  local smoke_root="$WORK_DIR/runs/$run_tag"
  [[ ! -e "$smoke_root" ]] || fail "run directory already exists: $smoke_root"

  if [[ "$DRY_RUN" == "1" ]]; then
    log "dry-run: would run IR and AST smoke matrices under $smoke_root"
    return
  fi

  mkdir -p "$smoke_root"
  PYTHON_BIN="$PYTHON_BIN" bash "$REPO_ROOT/scripts/run_replay_matrix.sh" \
    ir_structural \
    "$REPO_ROOT/examples/replay/demo.jsonl" \
    "$smoke_root/ir_structural" \
    "$REPO_ROOT/tasks/ir_structural/config.example.json"
  PYTHON_BIN="$PYTHON_BIN" bash "$REPO_ROOT/scripts/run_replay_matrix.sh" \
    execution_ast \
    "$REPO_ROOT/examples/replay/demo.jsonl" \
    "$smoke_root/execution_ast" \
    "$REPO_ROOT/tasks/execution_ast/config.example.json"
  log "smoke complete: $smoke_root"
}

show_status() {
  local run_tag
  run_tag="$(resolve_latest_tag)"
  local run_root="$WORK_DIR/runs/$run_tag"
  local log_root="$WORK_DIR/logs/$run_tag"
  local pid_file
  local found=0

  [[ -d "$log_root" ]] || fail "log directory does not exist: $log_root"
  log "status for $run_tag"

  shopt -s nullglob
  for pid_file in "$log_root"/*.pid; do
    found=1
    local method
    local pid
    local state
    method="$(basename "$pid_file" .pid)"
    pid="$(head -n 1 "$pid_file")"
    if [[ -f "$run_root/$method/matrix.csv" ]]; then
      state="completed"
    elif kill -0 "$pid" 2>/dev/null; then
      state="running"
    else
      state="stopped-or-failed"
    fi
    printf '%-28s pid=%-8s %s\n' "$method" "$pid" "$state"
    if [[ -f "$log_root/$method.log" ]]; then
      tail -n 3 "$log_root/$method.log" | sed 's/^/  | /'
    fi
  done
  shopt -u nullglob
  [[ "$found" == "1" ]] || fail "no pid files found under $log_root"
}

show_results() {
  local run_tag
  run_tag="$(resolve_latest_tag)"
  local run_root="$WORK_DIR/runs/$run_tag"
  local matrix
  local found=0

  [[ -d "$run_root" ]] || fail "run directory does not exist: $run_root"
  shopt -s nullglob
  for matrix in "$run_root"/*/matrix.csv; do
    found=1
    printf '\n=== %s ===\n' "$(basename "$(dirname "$matrix")")"
    if command -v column >/dev/null 2>&1; then
      column -s, -t <"$matrix"
    else
      cat "$matrix"
    fi
  done
  shopt -u nullglob
  [[ "$found" == "1" ]] || fail "no completed matrix.csv files under $run_root"
}

case "$MODE" in
  launch)
    launch_experiments
    ;;
  smoke)
    run_smoke
    ;;
  status)
    show_status
    ;;
  results)
    show_results
    ;;
esac
