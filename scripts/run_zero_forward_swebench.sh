#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SERVER_PROFILE="${SERVER_PROFILE:-$REPO_ROOT/zero_forward_server_profile.env}"
if [[ -f "$SERVER_PROFILE" ]]; then
  # shellcheck source=/dev/null
  source "$SERVER_PROFILE"
fi

ENV_NAME="${ENV_NAME:-swepruner-training-free}"
BASE_DIR="${BASE_DIR:-/home/yuantao/futao}"
WORK_DIR="${WORK_DIR:-$BASE_DIR/swepruner_training_free_workspace}"
RUNS_DIR="${ZERO_FORWARD_RUNS_DIR:-$WORK_DIR/zero_forward_agent_runs}"
LAST_RUN_FILE="$RUNS_DIR/.last_run"

VLLM_API_BASE="${VLLM_API_BASE:-http://127.0.0.1:8015/v1}"
VLLM_API_KEY="${VLLM_API_KEY:-EMPTY}"
DATASET_SUBSET="${DATASET_SUBSET:-verified}"
DATASET_SPLIT="${DATASET_SPLIT:-test}"
TASK_SLICE="${TASK_SLICE:-0:10}"
TASK_FILTER="${TASK_FILTER:-}"
AGENT_WORKERS="${AGENT_WORKERS:-1}"
AGENT_STEP_LIMIT="${AGENT_STEP_LIMIT:-100}"
GRADER_WORKERS="${GRADER_WORKERS:-4}"
METHODS="${METHODS:-safe_rules,intent_ir,intent_structure,adaptive_evidence}"
PRUNING_THRESHOLD="${PRUNING_THRESHOLD:-0.5}"
PARALLEL_ARMS="${PARALLEL_ARMS:-1}"
ZERO_FORWARD_MIN_CHARS="${ZERO_FORWARD_MIN_CHARS:-1000}"
ZERO_FORWARD_TIMEOUT="${ZERO_FORWARD_TIMEOUT:-5}"
ZERO_FORWARD_RECOVERY_MAX_CHARS="${ZERO_FORWARD_RECOVERY_MAX_CHARS:-3000}"
MIN_INPUT_TOKENS="${MIN_INPUT_TOKENS:-1500}"
MIN_SAVINGS_TOKENS="${MIN_SAVINGS_TOKENS:-256}"
MAX_RETENTION_RATIO="${MAX_RETENTION_RATIO:-0.85}"
MAX_CPU_MS="${MAX_CPU_MS:-50}"
MAX_OUTPUT_CHARS="${MAX_OUTPUT_CHARS:-9000}"
RAW_TTL_HOURS="${RAW_TTL_HOURS:-72}"
MIN_FREE_DISK_GB="${MIN_FREE_DISK_GB:-10}"

MODE="${1:-launch}"
RUNTIME_ACTIVE=0
if [[ $# -gt 0 ]]; then
  shift
fi

log() {
  printf '[zero-forward-swebench] %s\n' "$*" >&2
}

fail() {
  log "ERROR: $*"
  exit 2
}

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_zero_forward_swebench.sh [preflight|smoke|launch|status|results|grade|stop]

This launcher is isolated from scripts/run_server_experiments.sh. It runs a
baseline and zero-forward methods against the existing mini-swe-agent, the
existing Qwen vLLM on :8015, Docker and real SWE-Bench tasks.

Default launch arms (five experiments total):
  baseline
  safe_rules
  intent_ir
  intent_structure
  adaptive_evidence

Every pruning arm performs zero LLM forwards and consumes zero pruner LLM
tokens. Raw output is compacted before the next Qwen request and can be
recovered from the local service.

Important overrides in zero_forward_server_profile.env:
  MINI_SWE_PYTHON      Python from the existing mini-swe-agent installation.
  MINI_SWE_BASE_CONFIG Existing mini-swe-agent swebench YAML.
  VLLM_MODEL_ID        Exact model id from :8015/v1/models (auto-detected otherwise).
  METHODS              Comma-separated zero-forward method names.
  PRUNING_THRESHOLD    Legacy /prune contract budget for ablation arms.
  ZERO_FORWARD_RECOVERY_MAX_CHARS
                       Maximum bounded recovery observation, default 3000.
  TASK_SLICE           mini-swe-agent slice, default 0:10.
  AGENT_STEP_LIMIT=100 Hard per-task model-call limit (required).
  PARALLEL_ARMS=1      Run all arms concurrently for quality comparison.
  PARALLEL_ARMS=0      Run sequentially for clean latency measurement.

The launcher removes an inherited uv/venv before activating the project conda.
It invokes mini-swe-agent through the isolated tool-boundary adapter. The
pruner does not import or call the Qwen/vLLM client.
EOF
}

case "$MODE" in
  preflight | smoke | launch | status | results | result | grade | stop) ;;
  help | -h | --help)
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
  unset VIRTUAL_ENV VIRTUAL_ENV_PROMPT UV_ACTIVE UV_PROJECT_ENVIRONMENT
  unset _OLD_VIRTUAL_PATH _OLD_VIRTUAL_PS1
  hash -r
  log "disabled inherited uv/venv: $active_venv"
}

resolve_mini_python_before_conda() {
  local candidate="${MINI_SWE_PYTHON:-}"
  if [[ -n "$candidate" && -x "$candidate" ]]; then
    printf '%s\n' "$candidate"
    return
  fi
  local mini_extra="${MINI_EXTRA_BIN:-$(command -v mini-extra 2>/dev/null || true)}"
  if [[ -n "$mini_extra" ]]; then
    local mini_dir
    mini_dir="$(cd "$(dirname "$mini_extra")" && pwd)"
    for candidate in "$mini_dir/python" "$mini_dir/python3"; do
      if [[ -x "$candidate" ]]; then
        printf '%s\n' "$candidate"
        return
      fi
    done
    local shebang
    shebang="$(head -n 1 "$mini_extra" 2>/dev/null || true)"
    candidate="${shebang#\#!}"
    if [[ "$candidate" = /* && -x "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return
    fi
  fi
  return 1
}

activate_runtime() {
  if [[ "$RUNTIME_ACTIVE" == "1" ]]; then
    cd "$REPO_ROOT"
    return
  fi
  MINI_SWE_PYTHON_BIN="$(resolve_mini_python_before_conda)" \
    || fail "cannot find mini-swe-agent Python; set MINI_SWE_PYTHON in $SERVER_PROFILE"
  local conda_base=""
  if command -v conda >/dev/null 2>&1; then
    conda_base="$(conda info --base)"
  elif [[ -n "${CONDA_PREFIX_BASE:-}" && -x "$CONDA_PREFIX_BASE/bin/conda" ]]; then
    conda_base="$("$CONDA_PREFIX_BASE/bin/conda" info --base)"
  elif [[ -x /opt/conda/bin/conda ]]; then
    conda_base="$(/opt/conda/bin/conda info --base)"
  else
    fail "conda was not found"
  fi
  disable_uv_or_venv
  # shellcheck source=/dev/null
  source "$conda_base/etc/profile.d/conda.sh"
  conda activate "$ENV_NAME" \
    || fail "conda env '$ENV_NAME' is missing; run scripts/create_server_conda.sh"
  PYTHON_BIN="$CONDA_PREFIX/bin/python"
  RUNTIME_ACTIVE=1
  cd "$REPO_ROOT"
  "$PYTHON_BIN" -c 'import zero_forward_pruning'
  log "runtime: conda=$ENV_NAME mini_python=$MINI_SWE_PYTHON_BIN"
}

discover_model() {
  if [[ -n "${VLLM_MODEL_ID:-}" ]]; then
    printf '%s\n' "$VLLM_MODEL_ID"
    return
  fi
  "$PYTHON_BIN" - "$VLLM_API_BASE" "$VLLM_API_KEY" <<'PY'
import json
import sys
import urllib.request

base, key = sys.argv[1:]
request = urllib.request.Request(
    base.rstrip("/") + "/models",
    headers={"Authorization": f"Bearer {key}"},
)
with urllib.request.urlopen(request, timeout=10) as response:
    payload = json.load(response)
models = [str(item.get("id", "")) for item in payload.get("data", [])]
models = [item for item in models if item]
if not models:
    raise SystemExit("vLLM returned no model id")
preferred = [item for item in models if "qwen3.5" in item.lower()]
print(preferred[0] if preferred else models[0])
PY
}

discover_base_config() {
  if [[ -n "${MINI_SWE_BASE_CONFIG:-}" ]]; then
    [[ -f "$MINI_SWE_BASE_CONFIG" ]] \
      || fail "MINI_SWE_BASE_CONFIG does not exist: $MINI_SWE_BASE_CONFIG"
    printf '%s\n' "$MINI_SWE_BASE_CONFIG"
    return
  fi
  MSWEA_SILENT_STARTUP=1 PYTHONPATH="$REPO_ROOT" "$MINI_SWE_PYTHON_BIN" - <<'PY'
from pathlib import Path
from minisweagent.config import builtin_config_dir

candidates = (
    Path(builtin_config_dir) / "extra" / "swebench.yaml",
    Path(builtin_config_dir) / "benchmarks" / "swebench.yaml",
    Path(builtin_config_dir) / "swebench.yaml",
)
for candidate in candidates:
    path = candidate.resolve()
    if path.is_file():
        print(path)
        break
else:
    raise SystemExit("default config not found; set MINI_SWE_BASE_CONFIG")
PY
}

check_disk_path() {
  local label="$1"
  local path="$2"
  local available_kb
  local available_gb
  available_kb="$(df -Pk "$path" | awk 'NR == 2 {print $4}')"
  [[ "$available_kb" =~ ^[0-9]+$ ]] \
    || fail "cannot determine free disk for $label at $path"
  available_gb=$((available_kb / 1024 / 1024))
  if ((available_gb < MIN_FREE_DISK_GB)); then
    fail "$label has ${available_gb}GB free at $path; minimum is ${MIN_FREE_DISK_GB}GB"
  fi
  log "$label disk: ${available_gb}GB free at $path"
}

preflight() {
  command -v docker >/dev/null 2>&1 || fail "Docker is required for SWE-Bench"
  docker info >/dev/null 2>&1 || fail "Docker daemon is unavailable"
  [[ "$MIN_FREE_DISK_GB" =~ ^[0-9]+$ ]] \
    || fail "MIN_FREE_DISK_GB must be a non-negative integer"
  "$PYTHON_BIN" - \
    "$PRUNING_THRESHOLD" \
    "$ZERO_FORWARD_TIMEOUT" \
    "$ZERO_FORWARD_MIN_CHARS" \
    "$ZERO_FORWARD_RECOVERY_MAX_CHARS" \
    "$MIN_INPUT_TOKENS" \
    "$MIN_SAVINGS_TOKENS" \
    "$MAX_RETENTION_RATIO" \
    "$MAX_CPU_MS" \
    "$MAX_OUTPUT_CHARS" \
    "$RAW_TTL_HOURS" \
    "$AGENT_WORKERS" \
    "$AGENT_STEP_LIMIT" <<'PY'
import sys

threshold = float(sys.argv[1])
timeout = float(sys.argv[2])
min_chars = int(sys.argv[3])
recovery_max_chars = int(sys.argv[4])
min_input = int(sys.argv[5])
min_savings = int(sys.argv[6])
max_retention = float(sys.argv[7])
max_cpu_ms = float(sys.argv[8])
max_output_chars = int(sys.argv[9])
raw_ttl_hours = float(sys.argv[10])
workers = int(sys.argv[11])
step_limit = int(sys.argv[12])
assert 0.0 <= threshold <= 1.0, "PRUNING_THRESHOLD must be in [0, 1]"
assert timeout > 0, "ZERO_FORWARD_TIMEOUT must be positive"
assert min_chars >= 0, "ZERO_FORWARD_MIN_CHARS must be non-negative"
assert recovery_max_chars >= 256, "ZERO_FORWARD_RECOVERY_MAX_CHARS must be at least 256"
assert min_input >= 0, "MIN_INPUT_TOKENS must be non-negative"
assert min_savings >= 1, "MIN_SAVINGS_TOKENS must be positive"
assert 0.0 < max_retention < 1.0, "MAX_RETENTION_RATIO must be in (0, 1)"
assert max_cpu_ms > 0, "MAX_CPU_MS must be positive"
assert max_output_chars >= 1000, "MAX_OUTPUT_CHARS must be at least 1000"
assert raw_ttl_hours > 0, "RAW_TTL_HOURS must be positive"
assert workers >= 1, "AGENT_WORKERS must be positive"
assert step_limit == 100, "AGENT_STEP_LIMIT must be exactly 100"
PY
  [[ "$PARALLEL_ARMS" == "0" || "$PARALLEL_ARMS" == "1" ]] \
    || fail "PARALLEL_ARMS must be 0 or 1"
  check_disk_path "repository" "$REPO_ROOT"
  local docker_root
  docker_root="$(docker info --format '{{.DockerRootDir}}')"
  if [[ -n "$docker_root" && -d "$docker_root" ]]; then
    check_disk_path "Docker" "$docker_root"
  else
    log "WARNING: Docker root is not directly readable: ${docker_root:-unknown}"
  fi
  PYTHONPATH="$REPO_ROOT" "$MINI_SWE_PYTHON_BIN" \
    -m zero_forward_pruning.mini_adapter.preflight
  RESOLVED_MODEL_ID="$(discover_model)" || fail "vLLM model discovery failed"
  RESOLVED_BASE_CONFIG="$(discover_base_config)" || fail "mini base config discovery failed"
  "$PYTHON_BIN" -m zero_forward_pruning.preflight --self-test
  log "preflight passed: model=$RESOLVED_MODEL_ID config=$RESOLVED_BASE_CONFIG"
}

method_port() {
  case "$1" in
    safe_rules) printf '8121\n' ;;
    intent_ir) printf '8122\n' ;;
    intent_structure) printf '8123\n' ;;
    adaptive_evidence) printf '8124\n' ;;
    *) fail "unsupported zero-forward method: $1" ;;
  esac
}

split_methods() {
  local value="${METHODS//,/ }"
  read -r -a METHOD_VALUES <<<"$value"
  [[ ${#METHOD_VALUES[@]} -gt 0 ]] || fail "METHODS must not be empty"
}

health_check() {
  "$PYTHON_BIN" - "$1" <<'PY'
import json
import sys
import time
import urllib.request

last_error = None
for _ in range(100):
    try:
        with urllib.request.urlopen(sys.argv[1], timeout=2) as response:
            payload = json.load(response)
        if payload.get("status") == "healthy":
            raise SystemExit(0)
    except Exception as exc:
        last_error = exc
        time.sleep(0.2)
raise SystemExit(f"health check failed: {last_error}")
PY
}

start_service() {
  local method="$1"
  local port
  port="$(method_port "$method")"
  local log_path="$RUN_ROOT/services/$method.log"
  local pid_path="$RUN_ROOT/services/$method.pid"
  "$PYTHON_BIN" - "$port" <<'PY' \
    || fail "port $port is already in use"
import socket
import sys

with socket.socket() as sock:
    sock.bind(("0.0.0.0", int(sys.argv[1])))
PY
  nohup "$PYTHON_BIN" -m zero_forward_pruning.http_server \
    --method "$method" \
    --host 0.0.0.0 \
    --port "$port" \
    --raw-store "$RUN_ROOT/raw/$method" \
    --raw-ttl-hours "$RAW_TTL_HOURS" \
    --public-base-url "http://host.docker.internal:$port" \
    --min-input-tokens "$MIN_INPUT_TOKENS" \
    --min-savings-tokens "$MIN_SAVINGS_TOKENS" \
    --max-retention-ratio "$MAX_RETENTION_RATIO" \
    --max-cpu-ms "$MAX_CPU_MS" \
    --max-output-chars "$MAX_OUTPUT_CHARS" \
    >"$log_path" 2>&1 &
  local pid=$!
  printf '%s\n' "$pid" >"$pid_path"
  health_check "http://127.0.0.1:$port/health" \
    || fail "zero-forward service failed: $method (see $log_path)"
  "$PYTHON_BIN" -m zero_forward_pruning.preflight \
    --url "http://127.0.0.1:$port" \
    || fail "zero-forward contract probe failed: $method (see $log_path)"
  log "started service: $method pid=$pid port=$port"
}

generate_shared_config() {
  local output="$RUN_ROOT/configs/agent.yaml"
  "$PYTHON_BIN" -m zero_forward_pruning.mini_adapter.config_adapter \
    --base-config "$RESOLVED_BASE_CONFIG" \
    --output "$output" \
    --model-id "$RESOLVED_MODEL_ID" \
    --api-base "$VLLM_API_BASE" \
    --api-key "$VLLM_API_KEY" \
    --timeout 180 \
    --step-limit "$AGENT_STEP_LIMIT" \
    >"$output.meta.json"
  SHARED_CONFIG="$output"
}

start_arm() {
  local arm="$1"
  local service_url="$2"
  local arm_dir="$RUN_ROOT/arms/$arm"
  local -a args=(
    -m zero_forward_pruning.mini_adapter.swebench
    --subset "$DATASET_SUBSET"
    --split "$DATASET_SPLIT"
    --output "$arm_dir"
    --workers "$AGENT_WORKERS"
    --config "$SHARED_CONFIG"
  )
  if [[ -n "$TASK_SLICE" ]]; then
    args+=(--slice "$TASK_SLICE")
  fi
  if [[ -n "$TASK_FILTER" ]]; then
    args+=(--filter "$TASK_FILTER")
  fi
  local zero_forward_url_value="$service_url"
  local allow_baseline=0
  if [[ "$arm" == "baseline" ]]; then
    allow_baseline=1
  fi
  local -a env_args=(
    PYTHONPATH="$REPO_ROOT"
    MSWEA_COST_TRACKING=ignore_errors
    MSWEA_MODEL_API_KEY="$VLLM_API_KEY"
    ZERO_FORWARD_PRUNER_URL="$zero_forward_url_value"
    ZERO_FORWARD_THRESHOLD="$PRUNING_THRESHOLD"
    ZERO_FORWARD_TIMEOUT="$ZERO_FORWARD_TIMEOUT"
    ZERO_FORWARD_MIN_CHARS="$ZERO_FORWARD_MIN_CHARS"
    ZERO_FORWARD_RECOVERY_MAX_CHARS="$ZERO_FORWARD_RECOVERY_MAX_CHARS"
    ZERO_FORWARD_ALLOW_BASELINE="$allow_baseline"
  )
  mkdir -p "$arm_dir"
  if [[ "$PARALLEL_ARMS" == "1" ]]; then
    nohup env "${env_args[@]}" \
      bash "$REPO_ROOT/scripts/run_zero_forward_arm.sh" \
      "$arm_dir" \
      "$MINI_SWE_PYTHON_BIN" \
      "${args[@]}" \
      >"$arm_dir/runner.log" 2>&1 &
    printf '%s\n' "$!" >"$arm_dir/pid"
    log "started arm: $arm pid=$!"
  else
    env "${env_args[@]}" \
      bash "$REPO_ROOT/scripts/run_zero_forward_arm.sh" \
      "$arm_dir" \
      "$MINI_SWE_PYTHON_BIN" \
      "${args[@]}" \
      2>&1 | tee "$arm_dir/runner.log"
  fi
}

write_manifest() {
  "$PYTHON_BIN" - \
    "$RUN_ROOT/manifest.json" \
    "$VLLM_API_BASE" \
    "$RESOLVED_MODEL_ID" \
    "$METHODS" \
    "$PRUNING_THRESHOLD" \
    "$ZERO_FORWARD_RECOVERY_MAX_CHARS" \
    "$TASK_SLICE" \
    "$RESOLVED_BASE_CONFIG" \
    "$MINI_SWE_PYTHON_BIN" \
    "$AGENT_STEP_LIMIT" <<'PY'
import json
import subprocess
import sys
from datetime import datetime, timezone

(
    output,
    api_base,
    model,
    methods,
    threshold,
    recovery_max_chars,
    task_slice,
    base_config,
    mini_python,
    agent_step_limit,
) = sys.argv[1:]
payload = {
    "created_at": datetime.now(timezone.utc).isoformat(),
    "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    "contract": "swe-pruner-compatible-v1",
    "vllm_api_base": api_base,
    "vllm_model": model,
    "methods": methods.replace(",", " ").split(),
    "contract_threshold": float(threshold),
    "recovery_output_max_chars": int(recovery_max_chars),
    "task_slice": task_slice,
    "base_config": base_config,
    "mini_swe_python": mini_python,
    "agent_step_limit": int(agent_step_limit),
    "timing": "tool output is compacted before the next agent model request",
    "trained_parameters": 0,
    "pruner_model_forwards_per_observation": 0,
    "pruner_llm_tokens_per_observation": 0,
    "raw_recovery": True,
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
    METHODS="adaptive_evidence"
    log "smoke: one real SWE-Bench task, baseline + adaptive_evidence"
  fi
  preflight
  local run_tag="${RUN_TAG:-zero_forward_qwen35_$(date +%Y%m%d_%H%M%S)}"
  RUN_ROOT="$RUNS_DIR/$run_tag"
  [[ ! -e "$RUN_ROOT" ]] || fail "run already exists: $RUN_ROOT"
  mkdir -p "$RUN_ROOT/services" "$RUN_ROOT/configs" "$RUN_ROOT/arms" "$RUN_ROOT/raw" "$RUNS_DIR"
  printf '%s\n' "$run_tag" >"$LAST_RUN_FILE"
  write_manifest
  generate_shared_config
  split_methods
  local method
  for method in "${METHOD_VALUES[@]}"; do
    start_service "$method"
  done
  start_arm baseline ""
  for method in "${METHOD_VALUES[@]}"; do
    start_arm "$method" "http://127.0.0.1:$(method_port "$method")"
  done
  if [[ "$PARALLEL_ARMS" == "1" ]]; then
    log "all arms launched; run root: $RUN_ROOT"
    log "next: bash scripts/run_zero_forward_swebench.sh status"
  else
    show_results
  fi
}

resolve_run_root() {
  local tag="${RUN_TAG:-}"
  if [[ -z "$tag" ]]; then
    [[ -f "$LAST_RUN_FILE" ]] || fail "no previous run recorded at $LAST_RUN_FILE"
    tag="$(head -n 1 "$LAST_RUN_FILE")"
  fi
  RUN_ROOT="$RUNS_DIR/$tag"
  [[ -d "$RUN_ROOT" ]] || fail "run does not exist: $RUN_ROOT"
}

show_status() {
  resolve_run_root
  local pid_file pid name state exit_file
  shopt -s nullglob
  for pid_file in "$RUN_ROOT"/arms/*/pid; do
    pid="$(head -n 1 "$pid_file")"
    name="$(basename "$(dirname "$pid_file")")"
    exit_file="$(dirname "$pid_file")/exit_code"
    if [[ -f "$exit_file" ]]; then
      state="completed(exit=$(head -n 1 "$exit_file"))"
    elif kill -0 "$pid" 2>/dev/null; then
      state="running"
    else
      state="stopped-or-failed"
    fi
    printf 'arm      %-28s pid=%-8s %s\n' "$name" "$pid" "$state"
    tail -n 1 "$(dirname "$pid_file")/runner.log" 2>/dev/null | sed 's/^/  | /' || true
  done
  for pid_file in "$RUN_ROOT"/services/*.pid; do
    pid="$(head -n 1 "$pid_file")"
    name="$(basename "$pid_file" .pid)"
    if kill -0 "$pid" 2>/dev/null; then state="running"; else state="stopped"; fi
    printf 'service  %-28s pid=%-8s %s\n' "$name" "$pid" "$state"
  done
  shopt -u nullglob
}

show_results() {
  activate_runtime
  resolve_run_root
  "$PYTHON_BIN" -m zero_forward_pruning.agent_eval.aggregate summary --run-root "$RUN_ROOT"
  if command -v column >/dev/null 2>&1; then
    column -s, -t <"$RUN_ROOT/summary.csv"
  else
    cat "$RUN_ROOT/summary.csv"
  fi
}

resolve_grader_python() {
  local candidate
  for candidate in "${SWEBENCH_PYTHON:-}" "$MINI_SWE_PYTHON_BIN" "$PYTHON_BIN"; do
    [[ -n "$candidate" && -x "$candidate" ]] || continue
    if "$candidate" -c 'import swebench.harness.run_evaluation' >/dev/null 2>&1; then
      printf '%s\n' "$candidate"
      return
    fi
  done
  return 1
}

grade_results() {
  activate_runtime
  resolve_run_root
  local grader_python
  grader_python="$(resolve_grader_python)" \
    || fail "SWE-Bench harness not found; set SWEBENCH_PYTHON"
  local arm_dir arm grade_dir jsonl
  for arm_dir in "$RUN_ROOT"/arms/*; do
    [[ -f "$arm_dir/preds.json" ]] || continue
    arm="$(basename "$arm_dir")"
    grade_dir="$RUN_ROOT/grade/$arm"
    jsonl="$grade_dir/preds.jsonl"
    mkdir -p "$grade_dir"
    "$PYTHON_BIN" -m zero_forward_pruning.agent_eval.aggregate convert-preds \
      --input "$arm_dir/preds.json" --output "$jsonl"
    (
      cd "$grade_dir"
      "$grader_python" -m swebench.harness.run_evaluation \
        --dataset_name princeton-nlp/SWE-bench_Verified \
        --predictions_path "$jsonl" \
        --max_workers "$GRADER_WORKERS" \
        --run_id "$(basename "$RUN_ROOT")_$arm"
    ) 2>&1 | tee "$grade_dir/grader.log"
  done
  show_results
}

stop_run() {
  resolve_run_root
  local pid_file pid command_line
  shopt -s nullglob
  for pid_file in "$RUN_ROOT"/arms/*/pid "$RUN_ROOT"/services/*.pid; do
    pid="$(head -n 1 "$pid_file")"
    kill -0 "$pid" 2>/dev/null || continue
    command_line="$(ps -p "$pid" -o command= 2>/dev/null || true)"
    if [[ "$command_line" != *zero_forward* ]]; then
      log "skip stale pid $pid: command does not identify zero-forward run"
      continue
    fi
    kill "$pid"
    log "stopped pid=$pid"
  done
  shopt -u nullglob
}

case "$MODE" in
  preflight)
    activate_runtime
    preflight
    ;;
  smoke | launch)
    launch
    ;;
  status)
    show_status
    ;;
  results | result)
    show_results
    ;;
  grade)
    grade_results
    ;;
  stop)
    stop_run
    ;;
esac
