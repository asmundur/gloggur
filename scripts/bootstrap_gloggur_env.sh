#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VENV_DIR="${REPO_ROOT}/.venv"
INSTALL_SPEC="${GLOGGUR_BOOTSTRAP_INSTALL_SPEC:-.[all,dev]}"
RECREATE=0

usage() {
  cat <<'USAGE'
Usage: scripts/bootstrap_gloggur_env.sh [--recreate]

Options:
  --recreate   Remove existing .venv and rebuild from scratch.
  --help       Show this message.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --recreate)
      RECREATE=1
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      >&2 echo "Unknown option: $1"
      usage
      exit 2
      ;;
  esac
done

if command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python3)"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python)"
else
  >&2 echo "No python interpreter found. Install Python 3.10+ and retry."
  exit 3
fi

cd "$REPO_ROOT"
if [[ $RECREATE -eq 1 && -d "$VENV_DIR" ]]; then
  rm -rf "$VENV_DIR"
fi

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

"${VENV_DIR}/bin/python" -m pip install --upgrade pip setuptools wheel
"${VENV_DIR}/bin/python" -m pip install -e "$INSTALL_SPEC"

cat <<EOF
Bootstrap complete.
- Interpreter: ${VENV_DIR}/bin/python
- Installed: pip install -e '${INSTALL_SPEC}'

Next steps:
1. scripts/gloggur status --json
2. scripts/gloggur index . --json
EOF
