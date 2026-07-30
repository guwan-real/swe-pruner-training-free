#!/usr/bin/env bash
set -uo pipefail

if [[ $# -lt 3 ]]; then
  printf 'Usage: run_zero_forward_arm.sh ARM_DIR COMMAND ARGS...\n' >&2
  exit 2
fi

ARM_DIR="$1"
shift

mkdir -p "$ARM_DIR"
date -u +%Y-%m-%dT%H:%M:%SZ >"$ARM_DIR/started_at"

finish() {
  local code="$1"
  printf '%s\n' "$code" >"$ARM_DIR/exit_code"
  date -u +%Y-%m-%dT%H:%M:%SZ >"$ARM_DIR/ended_at"
  exit "$code"
}

terminate() {
  kill "$child_pid" 2>/dev/null || true
  wait "$child_pid" 2>/dev/null || true
  finish 143
}

"$@" &
child_pid=$!
trap terminate TERM INT
wait "$child_pid"
exit_code=$?
trap - TERM INT
finish "$exit_code"
