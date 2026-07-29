#!/usr/bin/env bash
set -uo pipefail

if [[ $# -lt 3 ]]; then
  printf 'Usage: run_one_agent_arm.sh ARM_DIR MINI_EXTRA_BIN ARGS...\n' >&2
  exit 2
fi

ARM_DIR="$1"
MINI_EXTRA_BIN="$2"
shift 2

mkdir -p "$ARM_DIR"
date -u +%Y-%m-%dT%H:%M:%SZ >"$ARM_DIR/started_at"
"$MINI_EXTRA_BIN" "$@"
exit_code=$?
printf '%s\n' "$exit_code" >"$ARM_DIR/exit_code"
date -u +%Y-%m-%dT%H:%M:%SZ >"$ARM_DIR/ended_at"
exit "$exit_code"
