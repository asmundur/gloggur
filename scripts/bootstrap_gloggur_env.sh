#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VENV_DIR="${REPO_ROOT}/.venv"
INSTALL_SPEC="${GLOGGUR_BOOTSTRAP_INSTALL_SPEC:-.[all,dev]}"
RECREATE=0
SKIP_INSTALL="${GLOGGUR_BOOTSTRAP_SKIP_INSTALL:-0}"
FORCE_INSTALL="${GLOGGUR_BOOTSTRAP_FORCE_INSTALL:-0}"
SEED_VENV_FROM="${GLOGGUR_BOOTSTRAP_VENV_SOURCE:-}"
SEED_VENV_MODE="${GLOGGUR_BOOTSTRAP_VENV_MODE:-symlink}"
SEED_VENV_DIR=""
SEED_VENV_RESULT="not_requested"
SEED_CACHE_FROM="${GLOGGUR_BOOTSTRAP_CACHE_SOURCE:-}"
SEED_CACHE_MODE="${GLOGGUR_BOOTSTRAP_CACHE_MODE:-symlink}"
SEED_CACHE_DIR=""
SEED_CACHE_RESULT="not_requested"
INSTALL_RESULT="not_run"
PYTHON_BIN=""

is_truthy() {
  local value="${1:-}"
  value="$(printf '%s' "$value" | tr '[:upper:]' '[:lower:]')"
  [[ "$value" == "1" || "$value" == "true" || "$value" == "yes" || "$value" == "on" ]]
}

validate_mode() {
  local mode="$1"
  local label="$2"
  case "$mode" in
    symlink|copy) ;;
    *)
      >&2 echo "Invalid ${label}: ${mode} (expected symlink|copy)"
      exit 2
      ;;
  esac
}

resolve_seed_venv_dir() {
  if [[ -z "$SEED_VENV_FROM" ]]; then
    return 0
  fi
  local source_path="$SEED_VENV_FROM"
  if [[ ! -e "$source_path" ]]; then
    >&2 echo "Seed venv source does not exist: ${source_path}"
    exit 2
  fi

  if [[ -d "$source_path" && "$(basename "$source_path")" == ".venv" ]]; then
    SEED_VENV_DIR="$source_path"
  else
    SEED_VENV_DIR="${source_path}/.venv"
  fi

  if [[ ! -x "${SEED_VENV_DIR}/bin/python" ]]; then
    >&2 echo "Seed venv source is missing executable python: ${SEED_VENV_DIR}/bin/python"
    exit 2
  fi

  SEED_VENV_DIR="$(cd "$SEED_VENV_DIR" && pwd -P)"
}

resolve_seed_cache_dir() {
  if [[ -z "$SEED_CACHE_FROM" ]]; then
    return 0
  fi
  local source_path="$SEED_CACHE_FROM"
  if [[ ! -e "$source_path" ]]; then
    >&2 echo "Seed cache source does not exist: ${source_path}"
    exit 2
  fi

  if [[ -d "$source_path" && "$(basename "$source_path")" == ".gloggur-cache" ]]; then
    SEED_CACHE_DIR="$source_path"
  else
    SEED_CACHE_DIR="${source_path}/.gloggur-cache"
  fi

  if [[ ! -d "$SEED_CACHE_DIR" ]]; then
    >&2 echo "Seed cache source is missing .gloggur-cache: ${source_path}"
    exit 2
  fi

  SEED_CACHE_DIR="$(cd "$SEED_CACHE_DIR" && pwd -P)"
}

seed_venv_if_requested() {
  if [[ -z "$SEED_VENV_DIR" ]]; then
    return 0
  fi
  if [[ -e "$VENV_DIR" || -L "$VENV_DIR" ]]; then
    if [[ $RECREATE -eq 1 ]]; then
      rm -rf "$VENV_DIR"
    else
      SEED_VENV_RESULT="skipped_existing:${VENV_DIR}"
      return 0
    fi
  fi

  if [[ "$SEED_VENV_MODE" == "symlink" ]]; then
    ln -s "$SEED_VENV_DIR" "$VENV_DIR"
    SEED_VENV_RESULT="symlinked:${SEED_VENV_DIR}"
    return 0
  fi

  cp -R "$SEED_VENV_DIR" "$VENV_DIR"
  SEED_VENV_RESULT="copied:${SEED_VENV_DIR}"
}

seed_cache_if_requested() {
  if [[ -z "$SEED_CACHE_DIR" ]]; then
    return 0
  fi
  local target_cache="${REPO_ROOT}/.gloggur-cache"
  if [[ -e "$target_cache" || -L "$target_cache" ]]; then
    if [[ $RECREATE -eq 1 ]]; then
      rm -rf "$target_cache"
    else
      SEED_CACHE_RESULT="skipped_existing:${target_cache}"
      return 0
    fi
  fi

  if [[ "$SEED_CACHE_MODE" == "symlink" ]]; then
    ln -s "$SEED_CACHE_DIR" "$target_cache"
    SEED_CACHE_RESULT="symlinked:${SEED_CACHE_DIR}"
    return 0
  fi

  cp -R "$SEED_CACHE_DIR" "$target_cache"
  SEED_CACHE_RESULT="copied:${SEED_CACHE_DIR}"
}

detect_python() {
  if [[ -n "$PYTHON_BIN" ]]; then
    return 0
  fi
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
    return 0
  fi
  if command -v python >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python)"
    return 0
  fi
  >&2 echo "No python interpreter found. Install Python 3.10+ and retry."
  exit 3
}

usage() {
  cat <<'USAGE'
Usage: scripts/bootstrap_gloggur_env.sh [--recreate] [--skip-install] [--force-install] [--seed-venv-from PATH] [--seed-venv-mode symlink|copy] [--seed-cache-from PATH] [--seed-cache-mode symlink|copy]

Options:
  --recreate                 Remove existing .venv and rebuild from scratch.
  --skip-install             Create/repair .venv but skip pip install steps.
  --force-install            Run pip install even when .venv was seeded.
  --seed-venv-from <path>    Seed .venv from another workspace or venv dir.
  --seed-venv-mode <mode>    Venv seed mode: symlink (default, fastest) or copy.
  --seed-cache-from <path>   Seed .gloggur-cache from another workspace or cache dir.
  --seed-cache-mode <mode>   Seed mode: symlink (default, fastest) or copy.
  --help                     Show this message.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --recreate)
      RECREATE=1
      shift
      ;;
    --skip-install)
      SKIP_INSTALL=1
      shift
      ;;
    --force-install)
      FORCE_INSTALL=1
      shift
      ;;
    --seed-venv-from)
      if [[ $# -lt 2 ]]; then
        >&2 echo "Missing value for --seed-venv-from"
        exit 2
      fi
      SEED_VENV_FROM="$2"
      shift 2
      ;;
    --seed-venv-mode)
      if [[ $# -lt 2 ]]; then
        >&2 echo "Missing value for --seed-venv-mode"
        exit 2
      fi
      SEED_VENV_MODE="$2"
      shift 2
      ;;
    --seed-cache-from)
      if [[ $# -lt 2 ]]; then
        >&2 echo "Missing value for --seed-cache-from"
        exit 2
      fi
      SEED_CACHE_FROM="$2"
      shift 2
      ;;
    --seed-cache-mode)
      if [[ $# -lt 2 ]]; then
        >&2 echo "Missing value for --seed-cache-mode"
        exit 2
      fi
      SEED_CACHE_MODE="$2"
      shift 2
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

validate_mode "$SEED_VENV_MODE" "--seed-venv-mode"
validate_mode "$SEED_CACHE_MODE" "--seed-cache-mode"
resolve_seed_venv_dir
resolve_seed_cache_dir

cd "$REPO_ROOT"
if [[ $RECREATE -eq 1 && ( -e "$VENV_DIR" || -L "$VENV_DIR" ) ]]; then
  rm -rf "$VENV_DIR"
fi

seed_venv_if_requested

if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
  detect_python
  "$PYTHON_BIN" -m venv "$VENV_DIR"
  if [[ "$SEED_VENV_RESULT" == "not_requested" ]]; then
    SEED_VENV_RESULT="created:${VENV_DIR}"
  fi
fi

if is_truthy "$SKIP_INSTALL"; then
  INSTALL_RESULT="skipped:requested"
elif [[ "$SEED_VENV_RESULT" == symlinked:* || "$SEED_VENV_RESULT" == copied:* ]] && ! is_truthy "$FORCE_INSTALL"; then
  INSTALL_RESULT="skipped:seeded_venv"
else
  "${VENV_DIR}/bin/python" -m pip install --upgrade pip setuptools wheel
  "${VENV_DIR}/bin/python" -m pip install -e "$INSTALL_SPEC"
  INSTALL_RESULT="installed:${INSTALL_SPEC}"
fi

seed_cache_if_requested

cat <<EOF
Bootstrap complete.
- Interpreter: ${VENV_DIR}/bin/python
- Venv seed: ${SEED_VENV_RESULT}
- Install step: ${INSTALL_RESULT}
- Cache seed: ${SEED_CACHE_RESULT}

Next steps:
1. scripts/gloggur status --json
2. scripts/gloggur index . --json
EOF
