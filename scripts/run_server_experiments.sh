#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SERVER_PROFILE="${SERVER_PROFILE:-$REPO_ROOT/server_profile.env}"
if [[ -f "$SERVER_PROFILE" ]]; then
  # shellcheck source=/dev/null
  source "$SERVER_PROFILE"
fi
INHERITED_MINI_EXTRA_BIN="${MINI_EXTRA_BIN:-$(command -v mini-extra 2>/dev/null || true)}"

ENV_NAME="${ENV_NAME:-swepruner-training-free}"
BASE_DIR="${BASE_DIR:-/home/yuantao/futao}"
WORK_DIR="${WORK_DIR:-$BASE_DIR/swepruner_training_free_workspace}"
AGENT_RUNS_DIR="${AGENT_RUNS_DIR:-$WORK_DIR/agent_runs}"
LAST_RUN_FILE="$AGENT_RUNS_DIR/.last_run"

VLLM_API_BASE="${VLLM_API_BASE:-http://127.0.0.1:8015/v1}"
VLLM_API_KEY="${VLLM_API_KEY:-EMPTY}"
DATASET_SUBSET="${DATASET_SUBSET:-verified}"
DATASET_SPLIT="${DATASET_SPLIT:-test}"
TASK_SLICE="${TASK_SLICE:-0:10}"
TASK_FILTER="${TASK_FILTER:-}"
AGENT_WORKERS="${AGENT_WORKERS:-4}"
GRADER_WORKERS="${GRADER_WORKERS:-4}"
KEEP_RATIOS="${KEEP_RATIOS:-0.5}"
METHODS="${METHODS:-ir_structural,execution_ast,ir_ast_hybrid}"
PARALLEL_ARMS="${PARALLEL_ARMS:-1}"
PRUNER_MIN_CHARS="${PRUNER_MIN_CHARS:-500}"
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-180}"
MIN_FREE_DISK_GB="${MIN_FREE_DISK_GB:-10}"
WARN_FREE_DISK_GB="${WARN_FREE_DISK_GB:-50}"

MODE="launch"
DRY_RUN=0

log() {
  printf '[coding-agent-experiments] %s\n' "$*" >&2
}

fail() {
  log "ERROR: $*"
  exit 2
}

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_server_experiments.sh [preflight|launch|smoke|status|results|grade|stop] [--dry-run]

This is the real coding-agent experiment launcher. It connects the installed
mini-swe-agent to the already-running OpenAI-compatible vLLM endpoint, starts
training-free /prune services, and runs SWE-Bench tasks. It never falls back to
the bundled two-row replay demo.

Commands:
  preflight  Check conda, vLLM, qwen model discovery, Docker and mini-swe-agent hooks.
  launch     Default. Launch baseline plus every method/budget arm.
  smoke      Launch a real one-task SWE-Bench Verified baseline + IR run.
  status     Show agent arm and pruner service process states.
  results    Aggregate trajectories, tokens, prune retention and grader results.
  grade      Run the official local SWE-Bench harness for every completed arm.
  stop       Stop only the agent/service PIDs recorded for the selected run.

Defaults:
  VLLM_API_BASE=http://127.0.0.1:8015/v1
  DATASET_SUBSET=verified
  DATASET_SPLIT=test
  TASK_SLICE=0:10
  AGENT_WORKERS=4
  KEEP_RATIOS=0.5
  METHODS=ir_structural,execution_ast,ir_ast_hybrid
  PARALLEL_ARMS=1

Useful overrides:
  SERVER_PROFILE        Local env file; defaults to untracked ./server_profile.env.
  VLLM_MODEL_ID        Exact id returned by GET /v1/models; otherwise auto-detected.
  MINI_EXTRA_BIN       Existing pruning-capable mini-extra executable (may be in another venv).
  MINI_SWE_PYTHON      Python executable belonging to that mini-swe-agent install.
  MINI_SWE_AGENT_ROOT  Existing mini-swe-agent source root; used to discover one template.
  MINI_SWE_BASE_CONFIG Pruning-capable mini-swe-agent swebench YAML.
  SWEBENCH_PYTHON      Python with swebench.harness installed for the grade command.
  TASK_SLICE           Same syntax as mini-extra --slice. Set to empty for full set.
  TASK_FILTER          Regex passed to mini-extra --filter.
  KEEP_RATIOS          Comma-separated retained ratios, e.g. 0.35,0.5,0.7.
  METHODS              Comma-separated methods from the default list.
  RUN_TAG               Stable output name. Required to resume a chosen run.
  PARALLEL_ARMS=0       Run experiment arms sequentially in the foreground.
  MIN_FREE_DISK_GB=10   Hard minimum on repository and Docker filesystems.
  WARN_FREE_DISK_GB=50  Print a warning below this free-space level.

The script removes an active uv/venv from its child environment before activating
the conda environment. It keeps using MINI_EXTRA_BIN through its absolute path, so
mini-swe-agent may remain in an existing separate venv. API keys are not committed.
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
  preflight | launch | smoke | status | results | grade | stop) ;;
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
    conda activate "$ENV_NAME" \
      || fail "conda env '$ENV_NAME' is missing; run scripts/create_server_conda.sh first"
    PYTHON_BIN="$CONDA_PREFIX/bin/python"
    log "activated conda env: $ENV_NAME ($PYTHON_BIN)"
  fi
  cd "$REPO_ROOT"
  "$PYTHON_BIN" -c \
    'import sys, tf_pruning; assert sys.version_info >= (3, 11), sys.version'
  local active_mini
  active_mini="$(command -v mini-extra 2>/dev/null || true)"
  MINI_EXTRA_BIN="${MINI_EXTRA_BIN:-${INHERITED_MINI_EXTRA_BIN:-$active_mini}}"
  if [[ -n "$MINI_EXTRA_BIN" && "$MINI_EXTRA_BIN" != */* ]]; then
    MINI_EXTRA_BIN="$(command -v "$MINI_EXTRA_BIN" 2>/dev/null || true)"
  fi
  if [[ -n "$MINI_EXTRA_BIN" && -e "$MINI_EXTRA_BIN" ]]; then
    MINI_EXTRA_BIN="$(
      cd "$(dirname "$MINI_EXTRA_BIN")"
      printf '%s/%s\n' "$PWD" "$(basename "$MINI_EXTRA_BIN")"
    )"
  fi
}

resolve_mini_python() {
  local candidate="${MINI_SWE_PYTHON:-}"
  if [[ -n "$candidate" ]]; then
    if [[ "$candidate" != */* ]]; then
      candidate="$(command -v "$candidate" 2>/dev/null || true)"
    fi
    [[ -x "$candidate" ]] || fail "MINI_SWE_PYTHON is not executable: ${candidate:-missing}"
    printf '%s\n' "$candidate"
    return
  fi

  local mini_dir
  mini_dir="$(dirname "$MINI_EXTRA_BIN")"
  for candidate in "$mini_dir/python" "$mini_dir/python3"; do
    if [[ -x "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return
    fi
  done

  local shebang
  shebang="$(head -n 1 "$MINI_EXTRA_BIN" 2>/dev/null || true)"
  candidate="${shebang#\#!}"
  if [[ "$candidate" = /* && -x "$candidate" ]]; then
    printf '%s\n' "$candidate"
    return
  fi
  fail "cannot resolve mini-swe-agent Python; set MINI_SWE_PYTHON explicitly"
}

discover_vllm_model() {
  if [[ -n "${VLLM_MODEL_ID:-}" ]]; then
    printf '%s\n' "$VLLM_MODEL_ID"
    return
  fi
  VLLM_REQUEST_API_KEY="$VLLM_API_KEY" \
    "$PYTHON_BIN" - "$VLLM_API_BASE" <<'PY'
import json
import os
import sys
import urllib.request

base = sys.argv[1]
key = os.environ["VLLM_REQUEST_API_KEY"]
url = base.rstrip("/") + "/models"
request = urllib.request.Request(url, headers={"Authorization": f"Bearer {key}"})
with urllib.request.urlopen(request, timeout=10) as response:
    payload = json.load(response)
models = payload.get("data", [])
if not models:
    raise SystemExit(f"no models returned by {url}")
preferred = [
    str(item.get("id", ""))
    for item in models
    if "qwen3.5" in str(item.get("id", "")).lower()
]
model_id = preferred[0] if preferred else str(models[0].get("id", ""))
if not model_id:
    raise SystemExit(f"model id missing in response from {url}")
print(model_id)
PY
}

resolve_base_config() {
  local explicit="${MINI_SWE_BASE_CONFIG:-}"
  local primary="$explicit"
  if [[ -z "$primary" ]]; then
    primary="$(
      "$MINI_SWE_PYTHON_BIN" - <<'PY' 2>/dev/null || true
from pathlib import Path
from minisweagent.config import builtin_config_dir

print((Path(builtin_config_dir) / "extra" / "swebench.yaml").resolve())
PY
    )"
  fi
  "$PYTHON_BIN" - \
    "$primary" \
    "${MINI_SWE_AGENT_ROOT:-}" \
    "$([[ -n "$explicit" ]] && printf '1' || printf '0')" <<'PY'
import sys
from pathlib import Path
from agent_eval.config_adapter import (
    resolve_pruning_base_config,
)

primary = Path(sys.argv[1]) if sys.argv[1] else None
search_root = Path(sys.argv[2]) if sys.argv[2] else None
path = resolve_pruning_base_config(
    primary,
    search_root=search_root,
    explicit=sys.argv[3] == "1",
)
print(path)
PY
}

check_disk_path() {
  local label="$1"
  local path="$2"
  local probe="$path"
  while [[ ! -e "$probe" && "$probe" != "/" ]]; do
    probe="$(dirname "$probe")"
  done
  if [[ ! -e "$probe" ]]; then
    log "WARNING: cannot locate an accessible filesystem path for $label at $path"
    return 0
  fi
  if [[ "$probe" != "$path" ]]; then
    log "WARNING: $label path is not directly accessible; checking nearest ancestor $probe"
  fi
  local available_kb
  local available_gb
  available_kb="$(df -Pk "$probe" | awk 'NR == 2 {print $4}')"
  [[ "$available_kb" =~ ^[0-9]+$ ]] \
    || fail "could not determine free disk space for $label at $probe"
  available_gb=$((available_kb / 1024 / 1024))
  if ((available_gb < MIN_FREE_DISK_GB)); then
    fail "$label filesystem has ${available_gb}GB free at $probe; minimum is ${MIN_FREE_DISK_GB}GB"
  fi
  if ((available_gb < WARN_FREE_DISK_GB)); then
    log "WARNING: $label filesystem has only ${available_gb}GB free at $probe"
  else
    log "$label disk ready: ${available_gb}GB free at $probe"
  fi
}

check_disk_capacity() {
  [[ "$MIN_FREE_DISK_GB" =~ ^[0-9]+$ ]] \
    || fail "MIN_FREE_DISK_GB must be a non-negative integer"
  [[ "$WARN_FREE_DISK_GB" =~ ^[0-9]+$ ]] \
    || fail "WARN_FREE_DISK_GB must be a non-negative integer"
  check_disk_path "workspace" "$REPO_ROOT"
  local docker_root
  docker_root="$(docker info --format '{{.DockerRootDir}}' 2>/dev/null || true)"
  if [[ -n "$docker_root" ]]; then
    check_disk_path "Docker" "$docker_root"
  else
    log "WARNING: Docker root directory could not be determined"
  fi
}

resolve_grader_python() {
  local candidate
  local resolved
  local -a candidates=(
    "${SWEBENCH_PYTHON:-}"
    "${MINI_SWE_PYTHON_BIN:-}"
    "$PYTHON_BIN"
  )
  for candidate in "${candidates[@]}"; do
    [[ -n "$candidate" ]] || continue
    resolved="$candidate"
    if [[ "$resolved" != */* ]]; then
      resolved="$(command -v "$resolved" 2>/dev/null || true)"
    fi
    [[ -x "$resolved" ]] || continue
    if "$resolved" -c 'import swebench.harness.run_evaluation' >/dev/null 2>&1; then
      printf '%s\n' "$resolved"
      return 0
    fi
  done
  return 1
}

preflight() {
  [[ "${SKIP_PREFLIGHT:-0}" != "1" ]] || {
    RESOLVED_MODEL_ID="${VLLM_MODEL_ID:-qwen3.5-27b-dry-run}"
    RESOLVED_BASE_CONFIG="${MINI_SWE_BASE_CONFIG:-/tmp/mini-swe-pruning.yaml}"
    MINI_EXTRA_BIN="${MINI_EXTRA_BIN:-mini-extra}"
    MINI_SWE_PYTHON_BIN="${MINI_SWE_PYTHON:-$PYTHON_BIN}"
    log "SKIP_PREFLIGHT=1; external checks skipped"
    return
  }
  if [[ -n "$MINI_EXTRA_BIN" && "$MINI_EXTRA_BIN" != */* ]]; then
    MINI_EXTRA_BIN="$(command -v "$MINI_EXTRA_BIN" 2>/dev/null || true)"
  fi
  [[ -x "$MINI_EXTRA_BIN" ]] \
    || fail "pruning-capable mini-extra was not found; set MINI_EXTRA_BIN to its absolute path"
  MINI_SWE_PYTHON_BIN="$(resolve_mini_python)"
  command -v docker >/dev/null 2>&1 || fail "docker is required for SWE-Bench"
  docker info >/dev/null 2>&1 \
    || fail "Docker daemon is unavailable or the current user lacks permission"
  check_disk_capacity

  local help_text
  help_text="$("$MINI_EXTRA_BIN" swebench --help 2>&1)" \
    || fail "mini-extra swebench --help failed"
  grep -q -- "--pruner-url" <<<"$help_text" \
    || fail "installed mini-swe-agent lacks --pruner-url; see docs/MINI_SWE_AGENT_ADAPTER.md"
  grep -q -- "--disable-pruner" <<<"$help_text" \
    || fail "installed mini-swe-agent lacks --disable-pruner; baseline would not be comparable"
  grep -q -- "--slice" <<<"$help_text" \
    || fail "installed mini-swe-agent lacks the required --slice option"
  if ! "$MINI_SWE_PYTHON_BIN" - <<'PY'
from minisweagent.agents.default import AgentConfig
from minisweagent.utils.pruner import PruneResponse, PrunerRequest

agent_fields = getattr(AgentConfig, "__dataclass_fields__", {})
request_fields = getattr(
    PrunerRequest,
    "model_fields",
    getattr(PrunerRequest, "__fields__", {}),
)
response_fields = getattr(
    PruneResponse,
    "model_fields",
    getattr(PruneResponse, "__fields__", {}),
)
assert "pruner" in agent_fields
assert {"query", "code", "threshold"} <= set(request_fields)
assert {
    "pruned_code",
    "origin_token_cnt",
    "left_token_cnt",
    "model_input_token_cnt",
} <= set(response_fields)
PY
  then
    fail "installed mini-swe-agent pruner request/response contract is incompatible"
  fi

  RESOLVED_MODEL_ID="$(discover_vllm_model)" \
    || fail "cannot query vLLM at $VLLM_API_BASE"
  RESOLVED_BASE_CONFIG="$(resolve_base_config)" \
    || fail "mini-swe-agent pruning config contract check failed"
  local normalized_model_id
  normalized_model_id="$(printf '%s' "$RESOLVED_MODEL_ID" | tr '[:upper:]' '[:lower:]')"
  if [[ "$normalized_model_id" != *qwen3.5* ]]; then
    log "WARNING: vLLM model id does not contain qwen3.5: $RESOLVED_MODEL_ID"
  fi
  log "vLLM ready: $VLLM_API_BASE model=$RESOLVED_MODEL_ID"
  log "mini-swe-agent CLI: $MINI_EXTRA_BIN"
  log "mini-swe-agent Python: $MINI_SWE_PYTHON_BIN"
  log "pruning config: $RESOLVED_BASE_CONFIG"
  local grader_python
  if grader_python="$(resolve_grader_python)"; then
    log "SWE-Bench grader ready: $grader_python"
  else
    log "WARNING: SWE-Bench grader Python not found; generation can run, grade requires SWEBENCH_PYTHON"
  fi
}

method_port() {
  case "$1" in
    ir_structural) printf '8111\n' ;;
    execution_ast) printf '8112\n' ;;
    ir_ast_hybrid) printf '8113\n' ;;
    *) fail "unsupported online method: $1" ;;
  esac
}

method_config() {
  case "$1" in
    ir_structural) printf '%s\n' "$REPO_ROOT/tasks/ir_structural/config.example.json" ;;
    execution_ast) printf '%s\n' "$REPO_ROOT/tasks/execution_ast/config.example.json" ;;
    ir_ast_hybrid) printf '%s\n' "$REPO_ROOT/tasks/ir_ast_hybrid/config.example.json" ;;
    *) fail "unsupported online method: $1" ;;
  esac
}

health_check() {
  local url="$1"
  "$PYTHON_BIN" - "$url" <<'PY'
import json
import sys
import time
import urllib.request

url = sys.argv[1]
last_error = None
for _ in range(50):
    try:
        with urllib.request.urlopen(url, timeout=2) as response:
            payload = json.load(response)
        if payload.get("status") == "healthy":
            raise SystemExit(0)
    except Exception as exc:
        last_error = exc
        time.sleep(0.2)
raise SystemExit(f"health check failed for {url}: {last_error}")
PY
}

split_csv() {
  local value="$1"
  value="${value//,/ }"
  read -r -a SPLIT_VALUES <<<"$value"
}

ratio_label() {
  "$PYTHON_BIN" - "$1" <<'PY'
import sys
value = float(sys.argv[1])
print(f"{round(value * 100):02d}")
PY
}

start_pruner() {
  local method="$1"
  local port
  local config
  port="$(method_port "$method")"
  config="$(method_config "$method")"
  local log_path="$RUN_ROOT/services/$method.log"
  local pid_path="$RUN_ROOT/services/$method.pid"
  if [[ "$DRY_RUN" == "1" ]]; then
    log "dry-run: pruner $method -> http://127.0.0.1:$port/prune"
    return
  fi
  if ! "$PYTHON_BIN" - "$port" <<'PY'
import socket
import sys

with socket.socket() as sock:
    sock.bind(("127.0.0.1", int(sys.argv[1])))
PY
  then
    fail "port $port is already in use; finish the previous run before launching another"
  fi
  nohup "$PYTHON_BIN" -m integrations.http_server \
    --method "$method" \
    --config "$config" \
    --host 127.0.0.1 \
    --port "$port" \
    --fail-closed \
    --no-prune-below 20 \
    >"$log_path" 2>&1 &
  local pid=$!
  printf '%s\n' "$pid" >"$pid_path"
  health_check "http://127.0.0.1:$port/health" \
    || fail "pruner failed to start: $method (see $log_path)"
  log "started pruner: $method pid=$pid port=$port"
}

generate_config() {
  local output="$1"
  local pruner_url="$2"
  local keep_ratio="$3"
  if [[ "$DRY_RUN" == "1" ]]; then
    log "dry-run: config $(basename "$output") model=$RESOLVED_MODEL_ID keep=$keep_ratio"
    return
  fi
  local config_api_key="EMPTY"
  if [[ "$VLLM_API_KEY" != "EMPTY" ]]; then
    config_api_key="ENV"
  fi
  "$PYTHON_BIN" -m agent_eval.config_adapter \
    --base-config "$RESOLVED_BASE_CONFIG" \
    --output "$output" \
    --model-id "$RESOLVED_MODEL_ID" \
    --api-base "$VLLM_API_BASE" \
    --api-key "$config_api_key" \
    --pruner-url "$pruner_url" \
    --keep-ratio "$keep_ratio" \
    --min-chars "$PRUNER_MIN_CHARS" \
    --timeout "$REQUEST_TIMEOUT" \
    >"$output.meta.json"
}

start_arm() {
  local arm="$1"
  local config="$2"
  local pruner_url="$3"
  local baseline="$4"
  local arm_dir="$RUN_ROOT/arms/$arm"
  local -a args=(
    swebench
    --subset "$DATASET_SUBSET"
    --split "$DATASET_SPLIT"
    --output "$arm_dir"
    --workers "$AGENT_WORKERS"
    --config "$config"
  )
  if [[ -n "$TASK_SLICE" ]]; then
    args+=(--slice "$TASK_SLICE")
  fi
  if [[ -n "$TASK_FILTER" ]]; then
    args+=(--filter "$TASK_FILTER")
  fi
  if [[ "$baseline" == "1" ]]; then
    args+=(--disable-pruner)
  else
    args+=(--pruner-url "$pruner_url")
  fi
  if [[ "$DRY_RUN" == "1" ]]; then
    log "dry-run: arm=$arm $MINI_EXTRA_BIN ${args[*]}"
    return
  fi
  mkdir -p "$arm_dir"
  if [[ "$PARALLEL_ARMS" == "1" ]]; then
    nohup env \
      MSWEA_COST_TRACKING=ignore_errors \
      MSWEA_MODEL_API_KEY="$VLLM_API_KEY" \
      bash "$REPO_ROOT/scripts/run_one_agent_arm.sh" \
      "$arm_dir" \
      "$MINI_EXTRA_BIN" \
      "${args[@]}" \
      >"$arm_dir/runner.log" 2>&1 &
    printf '%s\n' "$!" >"$arm_dir/pid"
    log "started agent arm: $arm pid=$!"
  else
    env \
      MSWEA_COST_TRACKING=ignore_errors \
      MSWEA_MODEL_API_KEY="$VLLM_API_KEY" \
      bash "$REPO_ROOT/scripts/run_one_agent_arm.sh" \
      "$arm_dir" \
      "$MINI_EXTRA_BIN" \
      "${args[@]}" \
      2>&1 | tee "$arm_dir/runner.log"
  fi
}

write_manifest() {
  if [[ "$DRY_RUN" != "0" ]]; then
    return 0
  fi
  "$PYTHON_BIN" - \
    "$RUN_ROOT/manifest.json" \
    "$VLLM_API_BASE" \
    "$RESOLVED_MODEL_ID" \
    "$DATASET_SUBSET" \
    "$DATASET_SPLIT" \
    "$TASK_SLICE" \
    "$TASK_FILTER" \
    "$AGENT_WORKERS" \
    "$METHODS" \
    "$KEEP_RATIOS" \
    "$RESOLVED_BASE_CONFIG" \
    "$MINI_EXTRA_BIN" \
    "$MINI_SWE_PYTHON_BIN" <<'PY'
import json
import subprocess
import sys
from datetime import datetime, timezone

(
    output,
    api_base,
    model_id,
    subset,
    split,
    task_slice,
    task_filter,
    workers,
    methods,
    keep_ratios,
    base_config,
    mini_extra_bin,
    mini_swe_python,
) = sys.argv[1:]
payload = {
    "created_at": datetime.now(timezone.utc).isoformat(),
    "git_commit": subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip(),
    "vllm_api_base": api_base,
    "vllm_model_id": model_id,
    "dataset_subset": subset,
    "dataset_split": split,
    "task_slice": task_slice,
    "task_filter": task_filter,
    "agent_workers": int(workers),
    "methods": methods.replace(",", " ").split(),
    "keep_ratios": [
        float(item) for item in keep_ratios.replace(",", " ").split()
    ],
    "mini_swe_base_config": base_config,
    "mini_extra_bin": mini_extra_bin,
    "mini_swe_python": mini_swe_python,
    "budget_semantics": "mini_threshold = 1 - training_free_keep_ratio",
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
    METHODS="ir_structural"
    KEEP_RATIOS="0.5"
    log "smoke is a real one-task SWE-Bench run; no replay/demo data is used"
  fi
  preflight

  local run_tag="${RUN_TAG:-qwen35_27b_$(date +%Y%m%d_%H%M%S)}"
  RUN_ROOT="$AGENT_RUNS_DIR/$run_tag"
  log "run root: $RUN_ROOT"
  log "dataset: $DATASET_SUBSET/$DATASET_SPLIT slice='${TASK_SLICE:-FULL}'"
  log "model: $RESOLVED_MODEL_ID via $VLLM_API_BASE"
  log "methods: $METHODS; keep ratios: $KEEP_RATIOS"
  if [[ "$DRY_RUN" == "0" ]]; then
    [[ ! -e "$RUN_ROOT" ]] || fail "run directory already exists: $RUN_ROOT"
    mkdir -p "$RUN_ROOT/services" "$RUN_ROOT/configs" "$RUN_ROOT/arms"
    mkdir -p "$AGENT_RUNS_DIR"
    printf '%s\n' "$run_tag" >"$LAST_RUN_FILE"
  fi
  write_manifest

  split_csv "$METHODS"
  local -a methods=("${SPLIT_VALUES[@]}")
  local method
  for method in "${methods[@]}"; do
    start_pruner "$method"
  done

  local baseline_url="http://127.0.0.1:8111/prune"
  if [[ ! " ${methods[*]} " =~ [[:space:]]ir_structural[[:space:]] ]]; then
    baseline_url="http://127.0.0.1:$(method_port "${methods[0]}")/prune"
  fi
  local baseline_config="$RUN_ROOT/configs/baseline.yaml"
  generate_config "$baseline_config" "$baseline_url" "1.0"
  start_arm baseline "$baseline_config" "$baseline_url" "1"

  split_csv "$KEEP_RATIOS"
  local -a ratios=("${SPLIT_VALUES[@]}")
  local ratio
  local label
  local port
  local url
  local config
  local arm
  for method in "${methods[@]}"; do
    port="$(method_port "$method")"
    url="http://127.0.0.1:$port/prune"
    for ratio in "${ratios[@]}"; do
      label="$(ratio_label "$ratio")"
      arm="${method}_keep${label}"
      config="$RUN_ROOT/configs/$arm.yaml"
      generate_config "$config" "$url" "$ratio"
      start_arm "$arm" "$config" "$url" "0"
    done
  done
  if [[ "$DRY_RUN" == "1" ]]; then
    log "dry-run complete; no files or processes were created"
  elif [[ "$PARALLEL_ARMS" == "1" ]]; then
    log "all coding-agent arms launched"
    log "next: bash scripts/run_server_experiments.sh status"
  else
    log "all coding-agent arms completed sequentially"
    show_results
  fi
}

resolve_run_root() {
  local tag="${RUN_TAG:-}"
  if [[ -z "$tag" ]]; then
    [[ -f "$LAST_RUN_FILE" ]] || fail "no previous agent run found at $LAST_RUN_FILE"
    tag="$(head -n 1 "$LAST_RUN_FILE")"
  fi
  [[ -n "$tag" ]] || fail "run tag is empty"
  RUN_ROOT="$AGENT_RUNS_DIR/$tag"
  [[ -d "$RUN_ROOT" ]] || fail "run directory does not exist: $RUN_ROOT"
}

process_state() {
  local pid_file="$1"
  local completion_file="$2"
  if [[ -f "$completion_file" ]]; then
    local code
    code="$(head -n 1 "$completion_file")"
    if [[ "$code" == "0" ]]; then
      printf 'completed\n'
    else
      printf 'failed(exit=%s)\n' "$code"
    fi
    return
  fi
  local pid
  pid="$(head -n 1 "$pid_file")"
  if kill -0 "$pid" 2>/dev/null; then
    printf 'running\n'
  else
    printf 'stopped-or-failed\n'
  fi
}

show_status() {
  resolve_run_root
  log "status for $RUN_ROOT"
  local pid_file
  local name
  local state
  shopt -s nullglob
  for pid_file in "$RUN_ROOT"/arms/*/pid; do
    name="$(basename "$(dirname "$pid_file")")"
    state="$(process_state "$pid_file" "$(dirname "$pid_file")/exit_code")"
    printf 'arm      %-34s pid=%-8s %s\n' "$name" "$(head -n 1 "$pid_file")" "$state"
    if [[ -f "$(dirname "$pid_file")/runner.log" ]]; then
      tail -n 2 "$(dirname "$pid_file")/runner.log" | sed 's/^/  | /'
    fi
  done
  for pid_file in "$RUN_ROOT"/services/*.pid; do
    name="$(basename "$pid_file" .pid)"
    if kill -0 "$(head -n 1 "$pid_file")" 2>/dev/null; then
      state="running"
    else
      state="stopped"
    fi
    printf 'service  %-34s pid=%-8s %s\n' "$name" "$(head -n 1 "$pid_file")" "$state"
  done
  shopt -u nullglob
}

show_results() {
  activate_runtime
  resolve_run_root
  "$PYTHON_BIN" -m agent_eval.aggregate summary --run-root "$RUN_ROOT"
  if command -v column >/dev/null 2>&1; then
    column -s, -t <"$RUN_ROOT/summary.csv"
  else
    cat "$RUN_ROOT/summary.csv"
  fi
}

grade_results() {
  activate_runtime
  resolve_run_root
  if [[ -n "$MINI_EXTRA_BIN" && -x "$MINI_EXTRA_BIN" ]]; then
    MINI_SWE_PYTHON_BIN="$(resolve_mini_python)"
  else
    MINI_SWE_PYTHON_BIN=""
  fi
  local grader_python
  grader_python="$(resolve_grader_python)" \
    || fail "SWE-Bench harness was not found; set SWEBENCH_PYTHON to its Python executable"
  local arm_dir
  local arm
  local predictions
  local grade_dir
  local jsonl
  shopt -s nullglob
  for arm_dir in "$RUN_ROOT"/arms/*; do
    [[ -d "$arm_dir" ]] || continue
    arm="$(basename "$arm_dir")"
    predictions="$arm_dir/preds.json"
    [[ -f "$predictions" ]] || {
      log "skip grading $arm: missing preds.json"
      continue
    }
    grade_dir="$RUN_ROOT/grade/$arm"
    jsonl="$grade_dir/preds.jsonl"
    mkdir -p "$grade_dir"
    "$PYTHON_BIN" -m agent_eval.aggregate convert-preds \
      --input "$predictions" \
      --output "$jsonl"
    log "grading $arm with official SWE-Bench harness"
    (
      cd "$grade_dir"
      "$grader_python" -m swebench.harness.run_evaluation \
        --dataset_name princeton-nlp/SWE-bench_Verified \
        --predictions_path "$jsonl" \
        --max_workers "$GRADER_WORKERS" \
        --run_id "$(basename "$RUN_ROOT")_$arm"
    ) 2>&1 | tee "$grade_dir/grader.log"
  done
  shopt -u nullglob
  show_results
}

stop_recorded_processes() {
  resolve_run_root
  local pid_file
  local pid
  local command_line
  local expected
  shopt -s nullglob
  for pid_file in "$RUN_ROOT"/arms/*/pid "$RUN_ROOT"/services/*.pid; do
    pid="$(head -n 1 "$pid_file")"
    if ! kill -0 "$pid" 2>/dev/null; then
      continue
    fi
    command_line="$(ps -p "$pid" -o command= 2>/dev/null || true)"
    if [[ "$pid_file" == */services/* ]]; then
      expected="integrations.http_server"
    else
      expected="run_one_agent_arm.sh"
    fi
    if [[ "$command_line" != *"$expected"* ]]; then
      log "skip stale pid=$pid from $pid_file; process command did not match $expected"
      continue
    fi
    kill "$pid"
    log "stopped pid=$pid from $pid_file"
  done
  shopt -u nullglob
}

case "$MODE" in
  preflight)
    activate_runtime
    preflight
    log "preflight passed"
    ;;
  launch | smoke)
    launch
    ;;
  status)
    resolve_run_root
    show_status
    ;;
  results)
    show_results
    ;;
  grade)
    grade_results
    ;;
  stop)
    stop_recorded_processes
    ;;
esac
