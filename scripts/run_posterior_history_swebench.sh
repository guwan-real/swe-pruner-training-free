#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SERVER_PROFILE="${SERVER_PROFILE:-$REPO_ROOT/posterior_history_server_profile.env}"

# A server profile provides defaults, while explicit command-environment
# values must win.  This matters for threshold sweeps because the local
# profile commonly contains POSTERIOR_MIN_INPUT_TOKENS=1500.
PROFILE_OVERRIDE_NAMES=(
  ENV_NAME BASE_DIR WORK_DIR POSTERIOR_HISTORY_RUNS_DIR
  VLLM_API_BASE VLLM_API_KEY VLLM_MODEL_ID MINI_SWE_PYTHON MINI_SWE_BASE_CONFIG
  MINI_EXTRA_BIN SWEBENCH_PYTHON DATASET_SUBSET DATASET_SPLIT TASK_SLICE TASK_FILTER
  AGENT_WORKERS AGENT_STEP_LIMIT GRADER_WORKERS POSTERIOR_HISTORY_METHODS POSTERIOR_HOT_OBSERVATIONS
  POSTERIOR_MIN_INPUT_TOKENS POSTERIOR_MIN_SAVINGS_TOKENS
  POSTERIOR_MAX_RETENTION_RATIO POSTERIOR_BLOCK_MAX_LINES POSTERIOR_MAX_OUTPUT_CHARS
  PARALLEL_ARMS MIN_FREE_DISK_GB RUN_TAG SKIP_BASELINE
)
PROFILE_OVERRIDE_SET_NAMES=()
PROFILE_OVERRIDE_VALUES=()
for profile_name in "${PROFILE_OVERRIDE_NAMES[@]}"; do
  if declare -p "$profile_name" >/dev/null 2>&1; then
    PROFILE_OVERRIDE_SET_NAMES+=("$profile_name")
    PROFILE_OVERRIDE_VALUES+=("${!profile_name}")
  fi
done
if [[ -f "$SERVER_PROFILE" ]]; then
  # shellcheck source=/dev/null
  source "$SERVER_PROFILE"
fi
for ((profile_index = 0; profile_index < ${#PROFILE_OVERRIDE_SET_NAMES[@]}; profile_index++)); do
  printf -v "${PROFILE_OVERRIDE_SET_NAMES[$profile_index]}" '%s' \
    "${PROFILE_OVERRIDE_VALUES[$profile_index]}"
done
unset PROFILE_OVERRIDE_NAMES PROFILE_OVERRIDE_SET_NAMES PROFILE_OVERRIDE_VALUES
unset profile_name profile_index

ENV_NAME="${ENV_NAME:-swepruner-training-free}"
BASE_DIR="${BASE_DIR:-/home/yuantao/futao}"
WORK_DIR="${WORK_DIR:-$BASE_DIR/swepruner_training_free_workspace}"
RUNS_DIR="${POSTERIOR_HISTORY_RUNS_DIR:-$WORK_DIR/posterior_history_agent_runs}"
LAST_RUN_FILE="$RUNS_DIR/.last_run"
VLLM_API_BASE="${VLLM_API_BASE:-http://127.0.0.1:8015/v1}"
VLLM_API_KEY="${VLLM_API_KEY:-EMPTY}"
DATASET_SUBSET="${DATASET_SUBSET:-verified}"
DATASET_SPLIT="${DATASET_SPLIT:-test}"
TASK_SLICE="${TASK_SLICE:-0:5}"
TASK_FILTER="${TASK_FILTER:-}"
AGENT_WORKERS="${AGENT_WORKERS:-1}"
AGENT_STEP_LIMIT="${AGENT_STEP_LIMIT:-100}"
GRADER_WORKERS="${GRADER_WORKERS:-4}"
POSTERIOR_HISTORY_METHODS="${POSTERIOR_HISTORY_METHODS:-adaptive}"
POSTERIOR_HOT_OBSERVATIONS="${POSTERIOR_HOT_OBSERVATIONS:-2}"
POSTERIOR_MIN_INPUT_TOKENS="${POSTERIOR_MIN_INPUT_TOKENS:-1500}"
POSTERIOR_MIN_SAVINGS_TOKENS="${POSTERIOR_MIN_SAVINGS_TOKENS:-256}"
POSTERIOR_MAX_RETENTION_RATIO="${POSTERIOR_MAX_RETENTION_RATIO:-0.85}"
POSTERIOR_BLOCK_MAX_LINES="${POSTERIOR_BLOCK_MAX_LINES:-16}"
POSTERIOR_MAX_OUTPUT_CHARS="${POSTERIOR_MAX_OUTPUT_CHARS:-9000}"
PARALLEL_ARMS="${PARALLEL_ARMS:-0}"
MIN_FREE_DISK_GB="${MIN_FREE_DISK_GB:-10}"
SKIP_BASELINE="${SKIP_BASELINE:-0}"

MODE="${1:-launch}"
if [[ $# -gt 0 ]]; then shift; fi
RUNTIME_ACTIVE=0

log() { printf '[posterior-history] %s\n' "$*" >&2; }
fail() { log "ERROR: $*"; exit 2; }

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_posterior_history_swebench.sh [config|preflight|smoke|launch|status|results|grade|stop]

This is an isolated posterior-history experiment. It never calls a pruning
model or an HTTP pruning service. The newest observations enter Qwen in full;
only an ephemeral view of older history is compacted after Qwen has produced a
normal follow-up action. The canonical trajectory remains full and auditable.

Default arms:
  baseline
  posterior_adaptive

Set SKIP_BASELINE=1 to run posterior arms only. For the standard 1000/500
serial sweep, use scripts/run_posterior_threshold_sweep.sh.

Important profile overrides:
  MINI_SWE_PYTHON, MINI_SWE_BASE_CONFIG, VLLM_MODEL_ID
  POSTERIOR_HISTORY_METHODS=safe,adaptive
  POSTERIOR_HOT_OBSERVATIONS=2
  AGENT_STEP_LIMIT=100       Hard per-task model-call limit (required).
  TASK_SLICE=0:5

PARALLEL_ARMS defaults to 0 because token and wall-time comparisons need a
single, uncontended vLLM. Set it to 1 only for a quality-only comparison.
EOF
}

case "$MODE" in
  config|preflight|smoke|launch|status|results|result|grade|stop) ;;
  help|-h|--help) usage; exit 0 ;;
  *) fail "unknown command: $MODE" ;;
esac

remove_path_entry() {
  local target="$1" entry
  local -a entries=() kept=()
  IFS=':' read -r -a entries <<<"${PATH:-}"
  for entry in "${entries[@]}"; do [[ "$entry" != "$target" ]] && kept+=("$entry"); done
  local IFS=':'
  PATH="${kept[*]}"
  export PATH
}

disable_uv_or_venv() {
  local active_venv="${VIRTUAL_ENV:-}"
  if [[ -z "$active_venv" ]]; then return 0; fi
  remove_path_entry "$active_venv/bin"
  unset VIRTUAL_ENV VIRTUAL_ENV_PROMPT UV_ACTIVE UV_PROJECT_ENVIRONMENT _OLD_VIRTUAL_PATH _OLD_VIRTUAL_PS1
  hash -r
  log "disabled inherited uv/venv: $active_venv"
}

resolve_mini_python_before_conda() {
  local candidate="${MINI_SWE_PYTHON:-}"
  if [[ -n "$candidate" && -x "$candidate" ]]; then printf '%s\n' "$candidate"; return; fi
  local mini_extra="${MINI_EXTRA_BIN:-$(command -v mini-extra 2>/dev/null || true)}"
  [[ -n "$mini_extra" ]] || return 1
  local mini_dir
  mini_dir="$(cd "$(dirname "$mini_extra")" && pwd)"
  for candidate in "$mini_dir/python" "$mini_dir/python3"; do
    [[ -x "$candidate" ]] && { printf '%s\n' "$candidate"; return; }
  done
  return 1
}

activate_runtime() {
  if [[ "$RUNTIME_ACTIVE" == "1" ]]; then cd "$REPO_ROOT"; return; fi
  MINI_SWE_PYTHON_BIN="$(resolve_mini_python_before_conda)" \
    || fail "cannot find mini-swe-agent Python; set MINI_SWE_PYTHON in $SERVER_PROFILE"
  local conda_base=""
  if command -v conda >/dev/null 2>&1; then
    conda_base="$(conda info --base)"
  elif [[ -x /opt/conda/bin/conda ]]; then
    conda_base="$(/opt/conda/bin/conda info --base)"
  else
    fail "conda was not found"
  fi
  disable_uv_or_venv
  # shellcheck source=/dev/null
  source "$conda_base/etc/profile.d/conda.sh"
  conda activate "$ENV_NAME" || fail "conda env '$ENV_NAME' is missing"
  PYTHON_BIN="$CONDA_PREFIX/bin/python"
  RUNTIME_ACTIVE=1
  cd "$REPO_ROOT"
  "$PYTHON_BIN" -c 'import posterior_history_pruning'
  log "runtime: conda=$ENV_NAME mini_python=$MINI_SWE_PYTHON_BIN"
}

discover_model() {
  [[ -z "${VLLM_MODEL_ID:-}" ]] || { printf '%s\n' "$VLLM_MODEL_ID"; return; }
  "$PYTHON_BIN" - "$VLLM_API_BASE" "$VLLM_API_KEY" <<'PY'
import json
import sys
import urllib.request

base, key = sys.argv[1:]
request = urllib.request.Request(base.rstrip("/") + "/models", headers={"Authorization": f"Bearer {key}"})
with urllib.request.urlopen(request, timeout=10) as response:
    payload = json.load(response)
models = [str(item.get("id", "")) for item in payload.get("data", []) if item.get("id")]
if not models:
    raise SystemExit("vLLM returned no model id")
preferred = [item for item in models if "qwen3.5" in item.lower()]
print(preferred[0] if preferred else models[0])
PY
}

discover_base_config() {
  if [[ -n "${MINI_SWE_BASE_CONFIG:-}" ]]; then
    [[ -f "$MINI_SWE_BASE_CONFIG" ]] || fail "MINI_SWE_BASE_CONFIG does not exist: $MINI_SWE_BASE_CONFIG"
    printf '%s\n' "$MINI_SWE_BASE_CONFIG"
    return
  fi
  MSWEA_SILENT_STARTUP=1 PYTHONPATH="$REPO_ROOT" "$MINI_SWE_PYTHON_BIN" - <<'PY'
from pathlib import Path
from minisweagent.config import builtin_config_dir

for candidate in (
    Path(builtin_config_dir) / "extra" / "swebench.yaml",
    Path(builtin_config_dir) / "benchmarks" / "swebench.yaml",
    Path(builtin_config_dir) / "swebench.yaml",
):
    if candidate.is_file():
        print(candidate.resolve())
        break
else:
    raise SystemExit("default config not found; set MINI_SWE_BASE_CONFIG")
PY
}

check_disk_path() {
  local label="$1" path="$2"
  local available_kb
  available_kb="$(df -Pk "$path" | awk 'NR == 2 {print $4}')"
  [[ "$available_kb" =~ ^[0-9]+$ ]] || fail "cannot determine free disk for $label"
  (( available_kb / 1024 / 1024 >= MIN_FREE_DISK_GB )) \
    || fail "$label has less than ${MIN_FREE_DISK_GB}GB free at $path"
}

split_methods() {
  local values="${POSTERIOR_HISTORY_METHODS//,/ }"
  read -r -a METHOD_VALUES <<<"$values"
  [[ ${#METHOD_VALUES[@]} -gt 0 ]] || fail "POSTERIOR_HISTORY_METHODS must not be empty"
  local method
  for method in "${METHOD_VALUES[@]}"; do
    [[ "$method" == "safe" || "$method" == "adaptive" ]] || fail "unsupported method: $method"
  done
}

preflight() {
  command -v docker >/dev/null 2>&1 || fail "Docker is required for SWE-Bench"
  docker info >/dev/null 2>&1 || fail "Docker daemon is unavailable"
  [[ "$MIN_FREE_DISK_GB" =~ ^[0-9]+$ ]] || fail "MIN_FREE_DISK_GB must be a non-negative integer"
  [[ "$PARALLEL_ARMS" == "0" || "$PARALLEL_ARMS" == "1" ]] || fail "PARALLEL_ARMS must be 0 or 1"
  [[ "$SKIP_BASELINE" == "0" || "$SKIP_BASELINE" == "1" ]] || fail "SKIP_BASELINE must be 0 or 1"
  "$PYTHON_BIN" - "$POSTERIOR_HOT_OBSERVATIONS" "$POSTERIOR_MIN_INPUT_TOKENS" \
    "$POSTERIOR_MIN_SAVINGS_TOKENS" "$POSTERIOR_MAX_RETENTION_RATIO" \
    "$POSTERIOR_BLOCK_MAX_LINES" "$POSTERIOR_MAX_OUTPUT_CHARS" "$AGENT_WORKERS" \
    "$AGENT_STEP_LIMIT" <<'PY'
import sys
hot, min_input, min_savings, retention, block, cap, workers, step_limit = sys.argv[1:]
assert int(hot) >= 1, "POSTERIOR_HOT_OBSERVATIONS must be at least 1"
assert int(min_input) >= 0, "POSTERIOR_MIN_INPUT_TOKENS must be non-negative"
assert int(min_savings) >= 1, "POSTERIOR_MIN_SAVINGS_TOKENS must be positive"
assert 0 < float(retention) < 1, "POSTERIOR_MAX_RETENTION_RATIO must be in (0, 1)"
assert int(block) >= 1, "POSTERIOR_BLOCK_MAX_LINES must be positive"
assert int(cap) >= 1000, "POSTERIOR_MAX_OUTPUT_CHARS must be at least 1000"
assert int(workers) >= 1, "AGENT_WORKERS must be positive"
assert int(step_limit) == 100, "AGENT_STEP_LIMIT must be exactly 100"
PY
  split_methods
  check_disk_path "repository" "$REPO_ROOT"
  local docker_root
  docker_root="$(docker info --format '{{.DockerRootDir}}')"
  if [[ -n "$docker_root" && -d "$docker_root" ]]; then
    check_disk_path "Docker" "$docker_root"
  fi
  PYTHONPATH="$REPO_ROOT" "$MINI_SWE_PYTHON_BIN" -m posterior_history_pruning.mini_adapter.preflight
  RESOLVED_MODEL_ID="$(discover_model)" || fail "vLLM model discovery failed"
  RESOLVED_BASE_CONFIG="$(discover_base_config)" || fail "mini base config discovery failed"
  log "preflight passed: model=$RESOLVED_MODEL_ID config=$RESOLVED_BASE_CONFIG"
}

generate_shared_config() {
  local output="$RUN_ROOT/configs/agent.yaml"
  "$PYTHON_BIN" -m posterior_history_pruning.mini_adapter.config_adapter \
    --base-config "$RESOLVED_BASE_CONFIG" --output "$output" --model-id "$RESOLVED_MODEL_ID" \
    --api-base "$VLLM_API_BASE" --api-key "$VLLM_API_KEY" --timeout 180 \
    --step-limit "$AGENT_STEP_LIMIT" >"$output.meta.json"
  SHARED_CONFIG="$output"
}

start_arm() {
  local arm="$1" method="$2"
  local arm_dir="$RUN_ROOT/arms/$arm"
  local enabled=1 allow_baseline=0
  if [[ "$arm" == "baseline" ]]; then enabled=0; allow_baseline=1; fi
  local -a args=(
    -m posterior_history_pruning.mini_adapter.swebench
    --subset "$DATASET_SUBSET" --split "$DATASET_SPLIT" --output "$arm_dir"
    --workers "$AGENT_WORKERS" --config "$SHARED_CONFIG"
  )
  [[ -z "$TASK_SLICE" ]] || args+=(--slice "$TASK_SLICE")
  [[ -z "$TASK_FILTER" ]] || args+=(--filter "$TASK_FILTER")
  local -a env_args=(
    PYTHONPATH="$REPO_ROOT" MSWEA_COST_TRACKING=ignore_errors MSWEA_MODEL_API_KEY="$VLLM_API_KEY"
    POSTERIOR_HISTORY_ENABLED="$enabled" POSTERIOR_HISTORY_ALLOW_BASELINE="$allow_baseline"
    POSTERIOR_HISTORY_METHOD="$method" POSTERIOR_HOT_OBSERVATIONS="$POSTERIOR_HOT_OBSERVATIONS"
    POSTERIOR_MIN_INPUT_TOKENS="$POSTERIOR_MIN_INPUT_TOKENS"
    POSTERIOR_MIN_SAVINGS_TOKENS="$POSTERIOR_MIN_SAVINGS_TOKENS"
    POSTERIOR_MAX_RETENTION_RATIO="$POSTERIOR_MAX_RETENTION_RATIO"
    POSTERIOR_BLOCK_MAX_LINES="$POSTERIOR_BLOCK_MAX_LINES"
    POSTERIOR_MAX_OUTPUT_CHARS="$POSTERIOR_MAX_OUTPUT_CHARS"
  )
  mkdir -p "$arm_dir"
  if [[ "$PARALLEL_ARMS" == "1" ]]; then
    nohup env "${env_args[@]}" bash "$REPO_ROOT/scripts/run_posterior_history_arm.sh" \
      "$arm_dir" "$MINI_SWE_PYTHON_BIN" "${args[@]}" >"$arm_dir/runner.log" 2>&1 &
    printf '%s\n' "$!" >"$arm_dir/pid"
    log "started arm: $arm pid=$!"
  else
    env "${env_args[@]}" bash "$REPO_ROOT/scripts/run_posterior_history_arm.sh" \
      "$arm_dir" "$MINI_SWE_PYTHON_BIN" "${args[@]}" 2>&1 | tee "$arm_dir/runner.log"
  fi
}

write_manifest() {
  "$PYTHON_BIN" - "$RUN_ROOT/manifest.json" "$VLLM_API_BASE" "$RESOLVED_MODEL_ID" \
    "$POSTERIOR_HISTORY_METHODS" "$TASK_SLICE" "$RESOLVED_BASE_CONFIG" \
    "$POSTERIOR_HOT_OBSERVATIONS" "$MINI_SWE_PYTHON_BIN" \
    "$POSTERIOR_MIN_INPUT_TOKENS" "$POSTERIOR_MIN_SAVINGS_TOKENS" \
    "$POSTERIOR_MAX_RETENTION_RATIO" "$POSTERIOR_BLOCK_MAX_LINES" \
    "$POSTERIOR_MAX_OUTPUT_CHARS" "$SKIP_BASELINE" "$AGENT_STEP_LIMIT" <<'PY'
import json
import subprocess
import sys
from datetime import datetime, timezone

(
    output,
    api_base,
    model,
    methods,
    task_slice,
    base_config,
    hot,
    mini_python,
    min_input,
    min_savings,
    max_retention,
    block_max_lines,
    max_output_chars,
    skip_baseline,
    agent_step_limit,
) = sys.argv[1:]
payload = {
    "created_at": datetime.now(timezone.utc).isoformat(),
    "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    "contract": "posterior-history-v1",
    "vllm_api_base": api_base,
    "vllm_model": model,
    "methods": methods.replace(",", " ").split(),
    "task_slice": task_slice,
    "base_config": base_config,
    "mini_swe_python": mini_python,
    "hot_observations": int(hot),
    "min_input_tokens": int(min_input),
    "min_savings_tokens": int(min_savings),
    "max_retention_ratio": float(max_retention),
    "block_max_lines": int(block_max_lines),
    "max_output_chars": int(max_output_chars),
    "baseline_included": skip_baseline != "1",
    "agent_step_limit": int(agent_step_limit),
    "timing": "full current observation, posterior-guided cold-history prompt view",
    "trained_parameters": 0,
    "pruner_model_forwards_per_observation": 0,
    "pruner_llm_tokens_per_observation": 0,
}
with open(output, "w", encoding="utf-8") as handle:
    json.dump(payload, handle, ensure_ascii=False, indent=2)
    handle.write("\n")
PY
}

launch() {
  activate_runtime
  if [[ "$MODE" == "smoke" ]]; then
    TASK_SLICE="0:1"
    POSTERIOR_HISTORY_METHODS="adaptive"
    if [[ "$SKIP_BASELINE" == "1" ]]; then
      log "smoke: one real SWE-Bench task, posterior_adaptive only"
    else
      log "smoke: one real SWE-Bench task, baseline + posterior_adaptive"
    fi
  fi
  preflight
  local run_tag="${RUN_TAG:-posterior_history_qwen35_$(date +%Y%m%d_%H%M%S)}"
  RUN_ROOT="$RUNS_DIR/$run_tag"
  [[ ! -e "$RUN_ROOT" ]] || fail "run already exists: $RUN_ROOT"
  mkdir -p "$RUN_ROOT/configs" "$RUN_ROOT/arms" "$RUNS_DIR"
  printf '%s\n' "$run_tag" >"$LAST_RUN_FILE"
  write_manifest
  generate_shared_config
  if [[ "$SKIP_BASELINE" == "1" ]]; then
    log "skipping baseline arm; posterior-only run"
  else
    start_arm baseline adaptive
  fi
  split_methods
  local method
  for method in "${METHOD_VALUES[@]}"; do start_arm "posterior_$method" "$method"; done
  [[ "$PARALLEL_ARMS" == "1" ]] && log "all arms launched; run root: $RUN_ROOT" || show_results
}

show_config() {
  printf 'SERVER_PROFILE=%s\n' "$SERVER_PROFILE"
  printf 'TASK_SLICE=%s\n' "$TASK_SLICE"
  printf 'POSTERIOR_HISTORY_METHODS=%s\n' "$POSTERIOR_HISTORY_METHODS"
  printf 'POSTERIOR_MIN_INPUT_TOKENS=%s\n' "$POSTERIOR_MIN_INPUT_TOKENS"
  printf 'AGENT_STEP_LIMIT=%s\n' "$AGENT_STEP_LIMIT"
  printf 'PARALLEL_ARMS=%s\n' "$PARALLEL_ARMS"
  printf 'SKIP_BASELINE=%s\n' "$SKIP_BASELINE"
  printf 'RUN_TAG=%s\n' "${RUN_TAG:-}"
}

resolve_run_root() {
  local tag="${RUN_TAG:-}"
  if [[ -z "$tag" ]]; then [[ -f "$LAST_RUN_FILE" ]] || fail "no previous run recorded"; tag="$(head -n 1 "$LAST_RUN_FILE")"; fi
  RUN_ROOT="$RUNS_DIR/$tag"
  [[ -d "$RUN_ROOT" ]] || fail "run does not exist: $RUN_ROOT"
}

show_status() {
  resolve_run_root
  local pid_file pid arm state
  shopt -s nullglob
  for pid_file in "$RUN_ROOT"/arms/*/pid; do
    pid="$(head -n 1 "$pid_file")"; arm="$(basename "$(dirname "$pid_file")")"
    if [[ -f "$(dirname "$pid_file")/exit_code" ]]; then state="completed(exit=$(head -n 1 "$(dirname "$pid_file")/exit_code"))"; elif kill -0 "$pid" 2>/dev/null; then state="running"; else state="stopped-or-failed"; fi
    printf 'arm  %-28s pid=%-8s %s\n' "$arm" "$pid" "$state"
  done
  shopt -u nullglob
}

show_results() {
  activate_runtime
  resolve_run_root
  "$PYTHON_BIN" -m posterior_history_pruning.agent_eval.aggregate summary --run-root "$RUN_ROOT"
  if command -v column >/dev/null 2>&1; then column -s, -t <"$RUN_ROOT/summary.csv"; else sed -n '1,20p' "$RUN_ROOT/summary.csv"; fi
}

resolve_grader_python() {
  local candidate
  for candidate in "${SWEBENCH_PYTHON:-}" "$MINI_SWE_PYTHON_BIN" "$PYTHON_BIN"; do
    [[ -n "$candidate" && -x "$candidate" ]] || continue
    "$candidate" -c 'import swebench.harness.run_evaluation' >/dev/null 2>&1 && { printf '%s\n' "$candidate"; return; }
  done
  return 1
}

grade_results() {
  activate_runtime
  resolve_run_root
  local grader_python
  grader_python="$(resolve_grader_python)" || fail "SWE-Bench harness not found; set SWEBENCH_PYTHON"
  local arm_dir arm grade_dir jsonl
  for arm_dir in "$RUN_ROOT"/arms/*; do
    [[ -f "$arm_dir/preds.json" ]] || continue
    arm="$(basename "$arm_dir")"; grade_dir="$RUN_ROOT/grade/$arm"; jsonl="$grade_dir/preds.jsonl"; mkdir -p "$grade_dir"
    "$PYTHON_BIN" -m posterior_history_pruning.agent_eval.aggregate convert-preds --input "$arm_dir/preds.json" --output "$jsonl"
    (cd "$grade_dir" && "$grader_python" -m swebench.harness.run_evaluation --dataset_name princeton-nlp/SWE-bench_Verified --predictions_path "$jsonl" --max_workers "$GRADER_WORKERS" --run_id "$(basename "$RUN_ROOT")_$arm") 2>&1 | tee "$grade_dir/grader.log"
  done
  show_results
}

stop_run() {
  resolve_run_root
  local pid_file pid command_line
  shopt -s nullglob
  for pid_file in "$RUN_ROOT"/arms/*/pid; do
    pid="$(head -n 1 "$pid_file")"; kill -0 "$pid" 2>/dev/null || continue
    command_line="$(ps -p "$pid" -o command= 2>/dev/null || true)"
    [[ "$command_line" == *posterior_history* ]] || { log "skip stale pid $pid"; continue; }
    kill "$pid"; log "stopped pid=$pid"
  done
  shopt -u nullglob
}

case "$MODE" in
  config) show_config ;;
  preflight) activate_runtime; preflight ;;
  smoke|launch) launch ;;
  status) show_status ;;
  results|result) show_results ;;
  grade) grade_results ;;
  stop) stop_run ;;
esac
