#!/usr/bin/env bash

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

run_beads_hook() {
  local hook_name="$1"
  shift || true

  if ! command -v bd >/dev/null 2>&1; then
    return 0
  fi

  export BD_GIT_HOOK=1

  local timeout_seconds="${BEADS_HOOK_TIMEOUT:-300}"
  local exit_code=0

  if command -v timeout >/dev/null 2>&1; then
    timeout "${timeout_seconds}" bd hooks run "${hook_name}" "$@" || exit_code=$?
    if [[ "${exit_code}" -eq 124 ]]; then
      echo >&2 "beads: hook '${hook_name}' timed out after ${timeout_seconds}s - continuing without beads"
      return 0
    fi
  else
    bd hooks run "${hook_name}" "$@" || exit_code=$?
  fi

  if [[ "${exit_code}" -eq 3 ]]; then
    echo >&2 "beads: database not initialized - skipping hook '${hook_name}'"
    return 0
  fi

  if [[ "${exit_code}" -ne 0 ]]; then
    return "${exit_code}"
  fi
}
