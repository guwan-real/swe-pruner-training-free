#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${1:-swepruner-training-free}"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PROFILE="${PROFILE:-agent}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
TORCH_VERSION="${TORCH_VERSION:-2.12.1}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu130}"

case "$PROFILE" in
  agent | model) ;;
  *)
    printf 'ERROR: PROFILE must be agent or model, got %s\n' "$PROFILE" >&2
    exit 2
    ;;
esac

command -v conda >/dev/null 2>&1 || {
  printf 'ERROR: conda was not found\n' >&2
  exit 2
}
CONDA_BASE="$(conda info --base)"

# New server terminals may start inside a uv-managed virtual environment.
# Resolve conda first, then remove that inherited environment before activation
# so pip and python cannot leak across the two runtimes.
if [[ -n "${VIRTUAL_ENV:-}" ]]; then
  ACTIVE_VENV="$VIRTUAL_ENV"
  NEW_PATH=""
  IFS=':' read -r -a PATH_PARTS <<<"${PATH:-}"
  for PATH_PART in "${PATH_PARTS[@]}"; do
    [[ "$PATH_PART" == "$ACTIVE_VENV/bin" ]] && continue
    if [[ -z "$NEW_PATH" ]]; then
      NEW_PATH="$PATH_PART"
    else
      NEW_PATH="$NEW_PATH:$PATH_PART"
    fi
  done
  PATH="$NEW_PATH"
  export PATH
  unset VIRTUAL_ENV VIRTUAL_ENV_PROMPT UV_ACTIVE UV_PROJECT_ENVIRONMENT
  unset _OLD_VIRTUAL_PATH _OLD_VIRTUAL_PS1
  hash -r
  printf 'Disabled inherited uv/venv: %s\n' "$ACTIVE_VENV"
fi

# shellcheck source=/dev/null
source "$CONDA_BASE/etc/profile.d/conda.sh"

if conda env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
  printf 'Reusing existing conda environment: %s\n' "$ENV_NAME"
else
  conda create -y -n "$ENV_NAME" "python=$PYTHON_VERSION" pip
fi
conda activate "$ENV_NAME"

python -m pip install --upgrade pip

if [[ "$PROFILE" == "agent" ]]; then
  # Zero-forward services are CPU-only. mini-swe-agent stays in its existing
  # separate environment and is selected with MINI_SWE_PYTHON.
  python -m pip install -e "$REPO_ROOT[agent]"
else
  # Only the second-stage PPL/hidden/attention/influence workflows need this
  # heavyweight local model stack.
  python -m pip install "torch==$TORCH_VERSION" --index-url "$TORCH_INDEX_URL"
  python -m pip install -e "$REPO_ROOT[model,syntax,agent]"
fi

python - "$PROFILE" <<'PY'
import sys

import tf_pruning
import zero_forward_pruning
import posterior_history_pruning
import yaml

print("profile:", sys.argv[1])
print("python:", sys.version.split()[0])
print("tf_pruning:", tf_pruning.__file__)
print("zero_forward_pruning:", zero_forward_pruning.__file__)
print("posterior_history_pruning:", posterior_history_pruning.__file__)
print("PyYAML:", yaml.__version__)

if sys.argv[1] == "model":
    import torch
    import transformers

    print("torch:", torch.__version__)
    print("torch CUDA:", torch.version.cuda)
    print("CUDA available:", torch.cuda.is_available())
    print("GPU count:", torch.cuda.device_count())
    print("transformers:", transformers.__version__)
    for index in range(torch.cuda.device_count()):
        print(index, torch.cuda.get_device_name(index))
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable for PROFILE=model")
PY

printf '\nEnvironment ready: %s (PROFILE=%s)\n' "$ENV_NAME" "$PROFILE"
if [[ "$PROFILE" == "agent" ]]; then
  printf 'mini-swe-agent is not installed or modified by this script.\n'
  printf 'Point the launcher at the existing mini Python with MINI_SWE_PYTHON.\n'
fi
