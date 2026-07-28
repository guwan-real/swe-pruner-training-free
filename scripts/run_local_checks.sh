#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

if [[ -z "${PYTHON_BIN:-}" ]]; then
  for CANDIDATE in python3.14 python3.13 python3.12 python3.11 python3; do
    RESOLVED="$(command -v "$CANDIDATE" 2>/dev/null || true)"
    if [[ -n "$RESOLVED" ]] && "$RESOLVED" -c \
      'import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)' \
      2>/dev/null; then
      PYTHON_BIN="$RESOLVED"
      break
    fi
  done
fi
if [[ -z "${PYTHON_BIN:-}" ]]; then
  echo "Python 3.11+ was not found. Set PYTHON_BIN explicitly." >&2
  exit 2
fi

"$PYTHON_BIN" -c 'import sys; assert sys.version_info >= (3, 11), sys.version'
"$PYTHON_BIN" -m compileall -q tf_pruning tasks evaluation tests
"$PYTHON_BIN" -m pytest
"$PYTHON_BIN" -m tf_pruning.cli methods
