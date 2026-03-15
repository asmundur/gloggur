#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GLOGGUR_WRAPPER="${SCRIPT_DIR}/gloggur"

if [[ $# -eq 0 ]]; then
  exec "${GLOGGUR_WRAPPER}" access grant .
fi

exec "${GLOGGUR_WRAPPER}" access grant "$@"
