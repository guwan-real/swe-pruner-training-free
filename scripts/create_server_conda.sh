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
  # Online IR/AST/hybrid services are CPU-only. mini-swe-agent may stay in its
  # existing separate venv and is selected with MINI_EXTRA_BIN.
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
import yaml

print("profile:", sys.argv[1])
print("python:", sys.version.split()[0])
print("tf_pruning:", tf_pruning.__file__)
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
  printf 'Point the launcher at the existing pruning build with MINI_EXTRA_BIN.\n'
fi
