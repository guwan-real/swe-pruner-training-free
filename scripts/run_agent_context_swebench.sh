#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SERVER_PROFILE="${SERVER_PROFILE:-$REPO_ROOT/agent_context_server_profile.env}"

PROFILE_OVERRIDE_NAMES=(
  ENV_NAME BASE_DIR WORK_DIR AGENT_CONTEXT_RUNS_DIR
  VLLM_API_BASE VLLM_API_KEY VLLM_MODEL_ID VLLM_METRICS_URL VLLM_RESET_PREFIX_CACHE_URL
  MINI_SWE_PYTHON MINI_SWE_BASE_CONFIG MINI_EXTRA_BIN SWEBENCH_PYTHON
  DATASET_SUBSET DATASET_SPLIT TASK_SLICE TASK_FILTER AGENT_WORKERS AGENT_STEP_LIMIT
  GRADER_WORKERS AGENT_CONTEXT_ARMS AGENT_CONTEXT_ARM_CONFIG_DIR
  AGENT_CONTEXT_REFERENCE_ARM SMOKE_ARM PARALLEL_ARMS MIN_FREE_DISK_GB RUN_TAG
  SWEBENCH_DATASET_NAME
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
RUNS_DIR="${AGENT_CONTEXT_RUNS_DIR:-$WORK_DIR/agent_context_runs}"
LAST_RUN_FILE="$RUNS_DIR/.last_run"
VLLM_API_BASE="${VLLM_API_BASE:-http://127.0.0.1:8015/v1}"
VLLM_API_KEY="${VLLM_API_KEY:-EMPTY}"
VLLM_METRICS_URL="${VLLM_METRICS_URL:-http://127.0.0.1:8015/metrics}"
VLLM_API_ROOT="${VLLM_API_BASE%/}"
VLLM_API_ROOT="${VLLM_API_ROOT%/v1}"
VLLM_RESET_PREFIX_CACHE_URL="${VLLM_RESET_PREFIX_CACHE_URL:-$VLLM_API_ROOT/reset_prefix_cache}"
DATASET_SUBSET="${DATASET_SUBSET:-verified}"
DATASET_SPLIT="${DATASET_SPLIT:-test}"
TASK_SLICE="${TASK_SLICE:-0:30}"
TASK_FILTER="${TASK_FILTER:-}"
AGENT_WORKERS="${AGENT_WORKERS:-1}"
AGENT_STEP_LIMIT="${AGENT_STEP_LIMIT:-100}"
GRADER_WORKERS="${GRADER_WORKERS:-4}"
AGENT_CONTEXT_ARMS="${AGENT_CONTEXT_ARMS:-R,D,E,F,C}"
AGENT_CONTEXT_ARM_CONFIG_DIR="${AGENT_CONTEXT_ARM_CONFIG_DIR:-configs/agent_context_server_arms}"
AGENT_CONTEXT_REFERENCE_ARM="${AGENT_CONTEXT_REFERENCE_ARM:-R}"
SMOKE_ARM="${SMOKE_ARM:-}"
PARALLEL_ARMS="${PARALLEL_ARMS:-0}"
MIN_FREE_DISK_GB="${MIN_FREE_DISK_GB:-10}"
SWEBENCH_DATASET_NAME="${SWEBENCH_DATASET_NAME:-}"

MODE="${1:-launch}"
if [[ $# -gt 0 ]]; then shift; fi
RUNTIME_ACTIVE=0

log() { printf '[agent-context] %s\n' "$*" >&2; }
fail() { log "ERROR: $*"; exit 2; }

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_agent_context_swebench.sh [config|preflight|smoke|launch|status|results|grade|stop]

Default pilot arms (no baseline rerun):
  R  legacy selector candidates + global retention=0.85 + freeze-on-cold
  D  R with hot_observations=1
  E  R with signal_provider=none
  F  typed_v1 codec
  C  context-limit target + dynamic replanning (KV-cache risk)

Optional F0 is F with signal_strategy=none. Add it with
AGENT_CONTEXT_ARMS=R,D,E,F,F0,C.

The launcher requires sequential arms, saves vLLM metrics before and
after each arm, enforces agent.step_limit=100, and uses one generated mini YAML
for prompt parity. C is not an absolute hard cap: protected full views can
still produce budget_overflow_tokens, which the summary reports.
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
  unset VIRTUAL_ENV VIRTUAL_ENV_PROMPT UV_ACTIVE UV_PROJECT_ENVIRONMENT
  unset _OLD_VIRTUAL_PATH _OLD_VIRTUAL_PS1
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
  "$PYTHON_BIN" -c 'import agent_context'
  log "runtime: conda=$ENV_NAME mini_python=$MINI_SWE_PYTHON_BIN"
}

discover_model() {
  [[ -z "${VLLM_MODEL_ID:-}" ]] || { printf '%s\n' "$VLLM_MODEL_ID"; return; }
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
models = [str(item.get("id", "")) for item in payload.get("data", []) if item.get("id")]
if not models:
    raise SystemExit("vLLM returned no model id")
preferred = [item for item in models if "qwen3.5" in item.lower()]
print(preferred[0] if preferred else models[0])
PY
}

discover_base_config() {
  if [[ -n "${MINI_SWE_BASE_CONFIG:-}" ]]; then
    [[ -f "$MINI_SWE_BASE_CONFIG" ]] || fail "MINI_SWE_BASE_CONFIG does not exist"
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

resolve_config_dir() {
  if [[ "$AGENT_CONTEXT_ARM_CONFIG_DIR" = /* ]]; then
    SOURCE_CONFIG_DIR="$AGENT_CONTEXT_ARM_CONFIG_DIR"
  else
    SOURCE_CONFIG_DIR="$REPO_ROOT/$AGENT_CONTEXT_ARM_CONFIG_DIR"
  fi
  [[ -d "$SOURCE_CONFIG_DIR" ]] || fail "arm config directory is missing: $SOURCE_CONFIG_DIR"
}

split_arms() {
  local values="${AGENT_CONTEXT_ARMS//,/ }" arm seen=" "
  read -r -a ARM_VALUES <<<"$values"
  [[ ${#ARM_VALUES[@]} -gt 0 ]] || fail "AGENT_CONTEXT_ARMS must not be empty"
  for arm in "${ARM_VALUES[@]}"; do
    [[ "$arm" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || fail "invalid arm name: $arm"
    [[ "$seen" != *" $arm "* ]] || fail "duplicate arm: $arm"
    seen+="$arm "
    [[ -f "$SOURCE_CONFIG_DIR/$arm.json" ]] || fail "missing arm config: $arm.json"
  done
  [[ " $seen" == *" $AGENT_CONTEXT_REFERENCE_ARM "* ]] \
    || fail "reference arm is not selected: $AGENT_CONTEXT_REFERENCE_ARM"
}

warn_dynamic_arms() {
  local dynamic_arms
  dynamic_arms="$("$PYTHON_BIN" - "$SOURCE_CONFIG_DIR" "${ARM_VALUES[@]}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
dynamic = []
for arm in sys.argv[2:]:
    payload = json.loads((root / f"{arm}.json").read_text(encoding="utf-8"))
    if payload.get("planner", {}).get("cache_policy") == "dynamic":
        dynamic.append(arm)
print(",".join(dynamic))
PY
)"
  [[ -z "$dynamic_arms" ]] \
    || log "WARNING: dynamic replanning in arms [$dynamic_arms]; inspect cache and wall-time metrics"
}

check_metrics_endpoint() {
  local contract
  contract="$("$PYTHON_BIN" - "$VLLM_METRICS_URL" <<'PY'
import sys
import urllib.request

with urllib.request.urlopen(sys.argv[1], timeout=10) as response:
    content = response.read().decode("utf-8", errors="replace")
if not content.strip():
    raise SystemExit("vLLM metrics endpoint returned an empty response")
has_prefix_counters = (
    "prefix_cache_queries_total" in content and "prefix_cache_hits_total" in content
)
print(f"metrics=reachable prefix_cache_counters={'yes' if has_prefix_counters else 'no'}")
PY
)" || fail "vLLM metrics endpoint is unavailable: $VLLM_METRICS_URL"
  log "$contract"
  [[ "$contract" == *"prefix_cache_counters=yes"* ]] \
    || log "WARNING: vLLM does not expose prefix-cache counters; cache metrics may be incomplete"
}

check_disk_path() {
  local label="$1" path="$2" available_kb
  available_kb="$(df -Pk "$path" | awk 'NR == 2 {print $4}')"
  [[ "$available_kb" =~ ^[0-9]+$ ]] || fail "cannot determine free disk for $label"
  (( available_kb / 1024 / 1024 >= MIN_FREE_DISK_GB )) \
    || fail "$label has less than ${MIN_FREE_DISK_GB}GB free at $path"
}

preflight() {
  resolve_config_dir
  split_arms
  resolve_swebench_dataset_name >/dev/null
  [[ -z "$(git status --porcelain --untracked-files=normal)" ]] \
    || fail "repository worktree is dirty; commit the experiment code before launch"
  command -v docker >/dev/null 2>&1 || fail "Docker is required for SWE-Bench"
  docker info >/dev/null 2>&1 || fail "Docker daemon is unavailable"
  [[ "$PARALLEL_ARMS" == "0" ]] \
    || fail "PARALLEL_ARMS=1 cannot produce isolated per-arm vLLM metrics"
  [[ "$MIN_FREE_DISK_GB" =~ ^[0-9]+$ ]] || fail "MIN_FREE_DISK_GB must be an integer"
  "$PYTHON_BIN" -c 'import yaml' \
    || fail "PyYAML is missing from conda env '$ENV_NAME'"
  "$PYTHON_BIN" - "$AGENT_WORKERS" "$AGENT_STEP_LIMIT" "$GRADER_WORKERS" <<'PY'
import sys

workers, step_limit, grader_workers = map(int, sys.argv[1:])
assert workers >= 1, "AGENT_WORKERS must be positive"
assert step_limit == 100, "AGENT_STEP_LIMIT must be exactly 100"
assert grader_workers >= 1, "GRADER_WORKERS must be positive"
PY
  mkdir -p "$RUNS_DIR"
  check_disk_path "repository" "$REPO_ROOT"
  check_disk_path "runs" "$RUNS_DIR"
  local docker_root
  docker_root="$(docker info --format '{{.DockerRootDir}}')"
  if [[ -n "$docker_root" && -d "$docker_root" ]]; then check_disk_path "Docker" "$docker_root"; fi
  local -a preflight_args=(-m agent_context.adapters.preflight)
  local arm
  for arm in "${ARM_VALUES[@]}"; do preflight_args+=(--config "$SOURCE_CONFIG_DIR/$arm.json"); done
  PYTHONPATH="$REPO_ROOT" "$MINI_SWE_PYTHON_BIN" "${preflight_args[@]}"
  RESOLVED_MODEL_ID="$(discover_model)" || fail "vLLM model discovery failed"
  check_metrics_endpoint
  RESOLVED_BASE_CONFIG="$(discover_base_config)" || fail "mini base config discovery failed"
  warn_dynamic_arms
  log "preflight passed: model=$RESOLVED_MODEL_ID config=$RESOLVED_BASE_CONFIG"
}

materialize_arm_configs() {
  local arm source output
  mkdir -p "$RUN_ROOT/configs/arms"
  for arm in "${ARM_VALUES[@]}"; do
    source="$SOURCE_CONFIG_DIR/$arm.json"
    output="$RUN_ROOT/configs/arms/$arm.json"
    "$PYTHON_BIN" - "$source" "$output" <<'PY'
import json
import sys
from pathlib import Path

from agent_context.config import AgentContextConfig
from agent_context.registry import DEFAULT_COMPONENT_REGISTRY

source, output = sys.argv[1:]
payload = json.loads(Path(source).read_text(encoding="utf-8"))
config = AgentContextConfig.from_mapping(payload)
DEFAULT_COMPONENT_REGISTRY.components(config)
Path(output).write_text(
    json.dumps(config.to_dict(), ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
PY
  done
}

generate_shared_config() {
  local output="$RUN_ROOT/configs/agent.yaml"
  "$PYTHON_BIN" -m agent_context.adapters.config_adapter \
    --base-config "$RESOLVED_BASE_CONFIG" --output "$output" \
    --model-id "$RESOLVED_MODEL_ID" --api-base "$VLLM_API_BASE" \
    --api-key "$VLLM_API_KEY" --timeout 180 --step-limit "$AGENT_STEP_LIMIT" \
    >"$output.meta.json"
  SHARED_CONFIG="$output"
}

snapshot_metrics() {
  local output="$1"
  "$PYTHON_BIN" - "$VLLM_METRICS_URL" "$output" <<'PY'
import sys
import urllib.request
from pathlib import Path

try:
    with urllib.request.urlopen(sys.argv[1], timeout=5) as response:
        content = response.read()
except Exception as exc:
    Path(sys.argv[2] + ".error").write_text(str(exc) + "\n", encoding="utf-8")
    raise SystemExit(1)
Path(sys.argv[2]).write_bytes(content)
PY
}

reset_prefix_cache() {
  local output="$1"
  "$PYTHON_BIN" - "$VLLM_RESET_PREFIX_CACHE_URL" "$output" <<'PY'
import json
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

request = urllib.request.Request(sys.argv[1], data=b"", method="POST")
with urllib.request.urlopen(request, timeout=30) as response:
    body = response.read().decode("utf-8", errors="replace")
    status = response.status
if not 200 <= status < 300:
    raise SystemExit(f"prefix-cache reset returned HTTP {status}")
Path(sys.argv[2]).write_text(
    json.dumps(
        {
            "reset_at": datetime.now(timezone.utc).isoformat(),
            "url": sys.argv[1],
            "http_status": status,
            "response": body,
        },
        ensure_ascii=False,
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)
PY
}

start_arm() {
  local arm="$1" arm_dir="$RUN_ROOT/arms/$arm"
  local config_path="$RUN_ROOT/configs/arms/$arm.json"
  local -a args=(
    -m agent_context.adapters.swebench
    --subset "$DATASET_SUBSET" --split "$DATASET_SPLIT" --output "$arm_dir"
    --workers "$AGENT_WORKERS" --config "$SHARED_CONFIG"
  )
  [[ -z "$TASK_SLICE" ]] || args+=(--slice "$TASK_SLICE")
  [[ -z "$TASK_FILTER" ]] || args+=(--filter "$TASK_FILTER")
  local -a env_args=(
    PYTHONPATH="$REPO_ROOT" MSWEA_COST_TRACKING=ignore_errors
    MSWEA_MODEL_API_KEY="$VLLM_API_KEY" AGENT_CONTEXT_ENABLED=1
    AGENT_CONTEXT_CONFIG="$config_path" POSTERIOR_HISTORY_ENABLED=0
    ZERO_FORWARD_PRUNER_URL=
  )
  mkdir -p "$arm_dir"
  reset_prefix_cache "$arm_dir/prefix_cache_reset.json" \
    || fail "could not reset vLLM prefix cache before $arm"
  snapshot_metrics "$arm_dir/vllm_metrics_before.prom" \
    || log "WARNING: could not snapshot vLLM metrics before $arm"
  local arm_exit_code=0
  env "${env_args[@]}" bash "$REPO_ROOT/scripts/run_agent_context_arm.sh" \
    "$arm_dir" "$MINI_SWE_PYTHON_BIN" "${args[@]}" 2>&1 \
    | tee "$arm_dir/runner.log" || arm_exit_code=$?
  snapshot_metrics "$arm_dir/vllm_metrics_after.prom" \
    || log "WARNING: could not snapshot vLLM metrics after $arm"
  if ((arm_exit_code != 0)); then
    log "ERROR: arm $arm failed with exit code $arm_exit_code"
    return "$arm_exit_code"
  fi
}

write_manifest() {
  local dataset_name
  dataset_name="$(resolve_swebench_dataset_name)"
  "$PYTHON_BIN" - "$RUN_ROOT/manifest.json" "$VLLM_API_BASE" "$VLLM_METRICS_URL" \
    "$VLLM_RESET_PREFIX_CACHE_URL" \
    "$RESOLVED_MODEL_ID" "$AGENT_CONTEXT_ARMS" "$TASK_SLICE" "$RESOLVED_BASE_CONFIG" \
    "$MINI_SWE_PYTHON_BIN" "$AGENT_STEP_LIMIT" "$RUN_ROOT/configs/arms" \
    "$AGENT_CONTEXT_REFERENCE_ARM" "$DATASET_SUBSET" "$DATASET_SPLIT" "$dataset_name" \
    "$TASK_FILTER" "$AGENT_WORKERS" "$GRADER_WORKERS" <<'PY'
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

(
    output, api_base, metrics_url, reset_prefix_cache_url, model, arms, task_slice, base_config,
    mini_python, step_limit, config_dir, reference_arm, dataset_subset, dataset_split,
    dataset_name, task_filter, agent_workers, grader_workers,
) = sys.argv[1:]
config_root = Path(config_dir)
arm_names = arms.replace(",", " ").split()
payload = {
    "created_at": datetime.now(timezone.utc).isoformat(),
    "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    "git_worktree_clean": not bool(
        subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=normal"], text=True
        ).strip()
    ),
    "contract": "agent-context-server-v3",
    "vllm_api_base": api_base,
    "vllm_metrics_url": metrics_url,
    "vllm_reset_prefix_cache_url": reset_prefix_cache_url,
    "vllm_model": model,
    "arms": arm_names,
    "reference_arm": reference_arm,
    "task_slice": task_slice,
    "task_filter": task_filter,
    "task_ids": [],
    "dataset_subset": dataset_subset,
    "dataset_split": dataset_split,
    "dataset_name": dataset_name,
    "base_config": base_config,
    "mini_swe_python": mini_python,
    "agent_step_limit": int(step_limit),
    "agent_workers": int(agent_workers),
    "grader_workers": int(grader_workers),
    "baseline_reused": True,
    "arm_configs": {
        arm: json.loads((config_root / f"{arm}.json").read_text(encoding="utf-8"))
        for arm in arm_names
    },
}
Path(output).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PY
}

finalize_manifest() {
  "$PYTHON_BIN" -m agent_context.agent_eval.run_manifest \
    --manifest "$RUN_ROOT/manifest.json" --arms-root "$RUN_ROOT/arms"
}

launch() {
  activate_runtime
  if [[ "$MODE" == "smoke" ]]; then
    local smoke_values="${AGENT_CONTEXT_ARMS//,/ }"
    local -a smoke_arms=()
    read -r -a smoke_arms <<<"$smoke_values"
    [[ ${#smoke_arms[@]} -gt 0 ]] || fail "AGENT_CONTEXT_ARMS must not be empty"
    TASK_SLICE="0:1"
    AGENT_CONTEXT_ARMS="${SMOKE_ARM:-${smoke_arms[0]}}"
    AGENT_CONTEXT_REFERENCE_ARM="$AGENT_CONTEXT_ARMS"
    log "smoke: one real SWE-Bench task, arm=$AGENT_CONTEXT_ARMS"
  fi
  preflight
  local run_tag="${RUN_TAG:-agent_context_qwen35_$(date +%Y%m%d_%H%M%S)}"
  RUN_ROOT="$RUNS_DIR/$run_tag"
  [[ ! -e "$RUN_ROOT" ]] || fail "run already exists: $RUN_ROOT"
  mkdir -p "$RUN_ROOT/configs" "$RUN_ROOT/arms" "$RUNS_DIR"
  printf '%s\n' "$run_tag" >"$LAST_RUN_FILE"
  materialize_arm_configs
  generate_shared_config
  write_manifest
  local arm
  for arm in "${ARM_VALUES[@]}"; do start_arm "$arm"; done
  finalize_manifest
  show_results
}

resolve_run_root() {
  local tag="${RUN_TAG:-}"
  if [[ -z "$tag" ]]; then
    [[ -f "$LAST_RUN_FILE" ]] || fail "no previous run recorded"
    tag="$(head -n 1 "$LAST_RUN_FILE")"
  fi
  RUN_ROOT="$RUNS_DIR/$tag"
  [[ -d "$RUN_ROOT" ]] || fail "run does not exist: $RUN_ROOT"
}

show_status() {
  resolve_run_root
  local arm_dir pid_file pid arm state
  shopt -s nullglob
  for arm_dir in "$RUN_ROOT"/arms/*; do
    [[ -d "$arm_dir" ]] || continue
    arm="$(basename "$arm_dir")"
    pid_file="$arm_dir/pid"
    pid="-"
    [[ ! -f "$pid_file" ]] || pid="$(head -n 1 "$pid_file")"
    if [[ -f "$arm_dir/exit_code" ]]; then
      state="completed(exit=$(head -n 1 "$arm_dir/exit_code"))"
    elif [[ "$pid" != "-" ]] && kill -0 "$pid" 2>/dev/null; then
      state="running"
    elif [[ -f "$arm_dir/started_at" ]]; then
      state="stopped-or-failed"
    else
      state="pending"
    fi
    printf 'arm  %-12s pid=%-8s %s\n' "$arm" "$pid" "$state"
  done
  shopt -u nullglob
}

show_results() {
  activate_runtime
  resolve_run_root
  local reference_arm
  reference_arm="$("$PYTHON_BIN" - "$RUN_ROOT/manifest.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    print(json.load(handle).get("reference_arm", "R"))
PY
)"
  "$PYTHON_BIN" -m agent_context.agent_eval.aggregate summary \
    --run-root "$RUN_ROOT" --reference-arm "$reference_arm"
  if command -v column >/dev/null 2>&1; then
    column -s, -t <"$RUN_ROOT/summary.csv"
  else
    sed -n '1,20p' "$RUN_ROOT/summary.csv"
  fi
}

resolve_grader_python() {
  local candidate
  for candidate in "${SWEBENCH_PYTHON:-}" "$MINI_SWE_PYTHON_BIN" "$PYTHON_BIN"; do
    [[ -n "$candidate" && -x "$candidate" ]] || continue
    "$candidate" -c 'import swebench.harness.run_evaluation' >/dev/null 2>&1 \
      && { printf '%s\n' "$candidate"; return; }
  done
  return 1
}

resolve_swebench_dataset_name() {
  local dataset_name
  case "$DATASET_SUBSET" in
    verified) dataset_name='princeton-nlp/SWE-bench_Verified' ;;
    lite) dataset_name='princeton-nlp/SWE-bench_Lite' ;;
    full) dataset_name='princeton-nlp/SWE-bench' ;;
    *) fail "unsupported DATASET_SUBSET=$DATASET_SUBSET; expected verified, lite, or full" ;;
  esac
  if [[ -n "$SWEBENCH_DATASET_NAME" && "$SWEBENCH_DATASET_NAME" != "$dataset_name" ]]; then
    fail "SWEBENCH_DATASET_NAME=$SWEBENCH_DATASET_NAME does not match DATASET_SUBSET=$DATASET_SUBSET ($dataset_name)"
  fi
  printf '%s\n' "$dataset_name"
}

grade_results() {
  activate_runtime
  resolve_run_root
  local grader_python dataset_name dataset_split grader_workers
  grader_python="$(resolve_grader_python)" || fail "SWE-Bench harness not found"
  local dataset_values
  dataset_values="$("$PYTHON_BIN" - "$RUN_ROOT/manifest.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    payload = json.load(handle)
dataset_name = payload.get("dataset_name")
if not dataset_name:
    raise SystemExit("run manifest has no dataset_name")
dataset_split = payload.get("dataset_split")
if not dataset_split:
    raise SystemExit("run manifest has no dataset_split")
grader_workers = payload.get("grader_workers")
if not isinstance(grader_workers, int) or grader_workers < 1:
    raise SystemExit("run manifest has no valid grader_workers")
print(f"{dataset_name}\t{dataset_split}\t{grader_workers}")
PY
)"
  IFS=$'\t' read -r dataset_name dataset_split grader_workers <<<"$dataset_values"
  local arm_dir arm grade_dir jsonl
  for arm_dir in "$RUN_ROOT"/arms/*; do
    [[ -f "$arm_dir/preds.json" ]] || continue
    arm="$(basename "$arm_dir")"
    grade_dir="$RUN_ROOT/grade/$arm"
    jsonl="$grade_dir/preds.jsonl"
    mkdir -p "$grade_dir"
    "$PYTHON_BIN" -m agent_context.agent_eval.aggregate convert-preds \
      --input "$arm_dir/preds.json" --output "$jsonl"
    (
      cd "$grade_dir"
      "$grader_python" -m swebench.harness.run_evaluation \
        --dataset_name "$dataset_name" --split "$dataset_split" \
        --predictions_path "$jsonl" --max_workers "$grader_workers" \
        --run_id "$(basename "$RUN_ROOT")_$arm"
    ) 2>&1 | tee "$grade_dir/grader.log"
  done
  show_results
}

stop_run() {
  resolve_run_root
  local pid_file pid command_line
  shopt -s nullglob
  for pid_file in "$RUN_ROOT"/arms/*/pid; do
    pid="$(head -n 1 "$pid_file")"
    kill -0 "$pid" 2>/dev/null || continue
    command_line="$(ps -p "$pid" -o command= 2>/dev/null || true)"
    [[ "$command_line" == *agent_context* ]] || { log "skip stale pid $pid"; continue; }
    kill "$pid"
    log "stopped pid=$pid"
  done
  shopt -u nullglob
}

show_config() {
  local resolved_dataset_name
  resolved_dataset_name="$(resolve_swebench_dataset_name)"
  printf 'SERVER_PROFILE=%s\n' "$SERVER_PROFILE"
  printf 'TASK_SLICE=%s\n' "$TASK_SLICE"
  printf 'AGENT_CONTEXT_ARMS=%s\n' "$AGENT_CONTEXT_ARMS"
  printf 'AGENT_CONTEXT_ARM_CONFIG_DIR=%s\n' "$AGENT_CONTEXT_ARM_CONFIG_DIR"
  printf 'AGENT_CONTEXT_REFERENCE_ARM=%s\n' "$AGENT_CONTEXT_REFERENCE_ARM"
  printf 'SMOKE_ARM=%s\n' "$SMOKE_ARM"
  printf 'DATASET_SUBSET=%s\n' "$DATASET_SUBSET"
  printf 'DATASET_SPLIT=%s\n' "$DATASET_SPLIT"
  printf 'TASK_FILTER=%s\n' "$TASK_FILTER"
  printf 'AGENT_WORKERS=%s\n' "$AGENT_WORKERS"
  printf 'VLLM_RESET_PREFIX_CACHE_URL=%s\n' "$VLLM_RESET_PREFIX_CACHE_URL"
  printf 'SWEBENCH_DATASET_NAME=%s\n' "$SWEBENCH_DATASET_NAME"
  printf 'RESOLVED_SWEBENCH_DATASET_NAME=%s\n' "$resolved_dataset_name"
  printf 'PARALLEL_ARMS=%s\n' "$PARALLEL_ARMS"
  printf 'RUN_TAG=%s\n' "${RUN_TAG:-}"
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
