#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${1:-swepruner-training-free}"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TORCH_VERSION="${TORCH_VERSION:-2.12.1}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu130}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda create -y -n "$ENV_NAME" python=3.11 pip
conda activate "$ENV_NAME"

python -m pip install --upgrade pip
python -m pip install "torch==$TORCH_VERSION" --index-url "$TORCH_INDEX_URL"
python -m pip install -e "$REPO_ROOT[model,syntax]"

python - <<'PY'
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
    raise SystemExit("CUDA is unavailable")
PY
