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
INDEX_FRESHNESS_RESULT="not_checked"
INSTALL_GLOBAL_WRAPPER="${GLOGGUR_BOOTSTRAP_INSTALL_GLOBAL_WRAPPER:-1}"
GLOBAL_BIN_DIR="${GLOGGUR_BOOTSTRAP_GLOBAL_BIN_DIR:-}"
GLOBAL_LINK_DIRS="${GLOGGUR_BOOTSTRAP_GLOBAL_LINK_DIRS:-/opt/homebrew/bin:/usr/local/bin}"
GLOBAL_WRAPPER_RESULT="not_checked"
PYTHON_BIN=""

if [[ -z "$GLOBAL_BIN_DIR" && -n "${HOME:-}" ]]; then
  GLOBAL_BIN_DIR="${HOME}/.local/bin"
fi

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

status_needs_reindex() {
  local payload="$1"
  local flattened=""
  flattened="$(printf '%s' "$payload" | tr '\n' ' ')"
  if printf '%s' "$flattened" | grep -Eq '"needs_reindex"[[:space:]]*:[[:space:]]*true'; then
    return 0
  fi
  if printf '%s' "$flattened" | grep -Eq '"needs_reindex"[[:space:]]*:[[:space:]]*false'; then
    return 1
  fi
  return 2
}

ensure_index_is_current() {
  local wrapper="${REPO_ROOT}/scripts/gloggur"
  if [[ ! -x "$wrapper" ]]; then
    INDEX_FRESHNESS_RESULT="skipped_missing_wrapper"
    return 0
  fi

  local status_output=""
  if ! status_output="$("$wrapper" status --json 2>&1)"; then
    >&2 echo "Index freshness check failed while running: scripts/gloggur status --json"
    >&2 echo "$status_output"
    exit 1
  fi

  local initial_needs_reindex=0
  if status_needs_reindex "$status_output"; then
    initial_needs_reindex=1
  else
    local status_rc=$?
    if [[ $status_rc -eq 1 ]]; then
      INDEX_FRESHNESS_RESULT="already_current"
      return 0
    fi
    >&2 echo "Index freshness check failed: unable to parse needs_reindex from status output."
    >&2 echo "$status_output"
    exit 1
  fi

  if [[ $initial_needs_reindex -eq 0 ]]; then
    INDEX_FRESHNESS_RESULT="already_current"
    return 0
  fi

  local index_output=""
  if ! index_output="$("$wrapper" index . --json 2>&1)"; then
    >&2 echo "Index freshness check failed while running: scripts/gloggur index . --json"
    >&2 echo "$index_output"
    exit 1
  fi

  local post_status_output=""
  if ! post_status_output="$("$wrapper" status --json 2>&1)"; then
    >&2 echo "Index freshness verification failed while running: scripts/gloggur status --json"
    >&2 echo "$post_status_output"
    exit 1
  fi

  if status_needs_reindex "$post_status_output"; then
    >&2 echo "Index freshness verification failed: cache still reports needs_reindex=true after refresh."
    >&2 echo "$post_status_output"
    exit 1
  fi
  local post_status_rc=$?
  if [[ $post_status_rc -eq 2 ]]; then
    >&2 echo "Index freshness verification failed: unable to parse needs_reindex from status output."
    >&2 echo "$post_status_output"
    exit 1
  fi

  INDEX_FRESHNESS_RESULT="refreshed"
}

ensure_startup_readiness() {
  local readiness_script="${REPO_ROOT}/scripts/check_startup_readiness.py"
  local wrapper="${REPO_ROOT}/scripts/gloggur"
  if [[ ! -x "$wrapper" ]]; then
    return 0
  fi
  if [[ ! -f "$readiness_script" ]]; then
    >&2 echo "Startup readiness verification failed: missing ${readiness_script}"
    exit 1
  fi

  local readiness_output=""
  if ! readiness_output="$("${VENV_DIR}/bin/python" "$readiness_script" --format json 2>&1)"; then
    >&2 echo "Startup readiness verification failed while running: python scripts/check_startup_readiness.py --format json"
    >&2 echo "$readiness_output"
    exit 1
  fi
}

install_global_wrapper() {
  if ! is_truthy "$INSTALL_GLOBAL_WRAPPER"; then
    GLOBAL_WRAPPER_RESULT="skipped:disabled"
    return 0
  fi

  if [[ -z "$GLOBAL_BIN_DIR" ]]; then
    GLOBAL_WRAPPER_RESULT="skipped:missing_home"
    return 0
  fi

  if ! mkdir -p "$GLOBAL_BIN_DIR"; then
    GLOBAL_WRAPPER_RESULT="failed:bin_dir_unwritable:${GLOBAL_BIN_DIR}"
    return 0
  fi

  local wrapper_path="${GLOBAL_BIN_DIR}/gloggur"
  if ! cat > "$wrapper_path" <<'SH'
#!/usr/bin/env bash
set -euo pipefail

is_json=0
for arg in "$@"; do
  if [[ "$arg" == "--json" ]]; then
    is_json=1
    break
  fi
done

emit_launch_target_missing() {
  local install_root="$1"
  local launcher_path="$2"
  if [[ $is_json -eq 1 ]]; then
    printf '{\n'
    printf '  "operation": "wrapper",\n'
    printf '  "error": true,\n'
    printf '  "error_code": "wrapper_launch_target_missing",\n'
    printf '  "message": "Unable to locate executable scripts/gloggur launcher from install root.",\n'
    printf '  "remediation": [\n'
    printf '    "Set GLOGGUR_INSTALL_ROOT to a gloggur checkout with scripts/gloggur.",\n'
    printf '    "Run scripts/bootstrap_gloggur_env.sh in that checkout to repair local tooling."\n'
    printf '  ],\n'
    printf '  "detected_environment": {\n'
    printf '    "cwd": "%s",\n' "$(pwd -P)"
    printf '    "install_root": "%s",\n' "$install_root"
    printf '    "launcher_path": "%s"\n' "$launcher_path"
    printf '  }\n'
    printf '}\n'
  else
    echo "gloggur wrapper failed (wrapper_launch_target_missing): launcher not found at ${launcher_path}" >&2
    echo "Set GLOGGUR_INSTALL_ROOT to a gloggur checkout and rerun scripts/bootstrap_gloggur_env.sh." >&2
  fi
}

DEFAULT_INSTALL_ROOT="__GLOGGUR_INSTALL_ROOT__"
INSTALL_ROOT="${GLOGGUR_INSTALL_ROOT:-$DEFAULT_INSTALL_ROOT}"
LAUNCHER="${INSTALL_ROOT}/scripts/gloggur"
if [[ ! -x "$LAUNCHER" ]]; then
  emit_launch_target_missing "$INSTALL_ROOT" "$LAUNCHER"
  exit 1
fi

SHARED_VENV="${INSTALL_ROOT}/.venv"
if [[ -x "${SHARED_VENV}/bin/python" ]]; then
  export GLOGGUR_PREFLIGHT_VENV_PYTHON="${SHARED_VENV}/bin/python"
fi

export GLOGGUR_RUN_FROM_CALLER_CWD=1
exec "$LAUNCHER" "$@"
SH
  then
    GLOBAL_WRAPPER_RESULT="failed:wrapper_unwritable:${wrapper_path}"
    return 0
  fi

  if ! "${VENV_DIR}/bin/python" - "$wrapper_path" "$REPO_ROOT" <<'PY'
from pathlib import Path
import sys

wrapper_path = Path(sys.argv[1])
repo_root = sys.argv[2]
content = wrapper_path.read_text(encoding="utf8")
wrapper_path.write_text(content.replace("__GLOGGUR_INSTALL_ROOT__", repo_root), encoding="utf8")
PY
  then
    GLOBAL_WRAPPER_RESULT="failed:placeholder_substitution:${wrapper_path}"
    return 0
  fi

  if ! chmod +x "$wrapper_path"; then
    GLOBAL_WRAPPER_RESULT="failed:chmod:${wrapper_path}"
    return 0
  fi

  local linked_path=""
  local candidate=""
  local IFS=':'
  for candidate in $GLOBAL_LINK_DIRS; do
    if [[ -z "$candidate" ]]; then
      continue
    fi
    if [[ -d "$candidate" && -w "$candidate" ]]; then
      if ln -sfn "$wrapper_path" "$candidate/gloggur"; then
        linked_path="${candidate}/gloggur"
      else
        GLOBAL_WRAPPER_RESULT="installed:${wrapper_path};link_failed:${candidate}/gloggur"
        return 0
      fi
      break
    fi
  done

  if [[ -n "$linked_path" ]]; then
    GLOBAL_WRAPPER_RESULT="installed:${wrapper_path};linked:${linked_path}"
  else
    GLOBAL_WRAPPER_RESULT="installed:${wrapper_path};linked:none"
  fi
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

install_global_wrapper
seed_cache_if_requested
ensure_index_is_current
ensure_startup_readiness

cat <<EOF
Bootstrap complete.
- Interpreter: ${VENV_DIR}/bin/python
- Venv seed: ${SEED_VENV_RESULT}
- Install step: ${INSTALL_RESULT}
- Global wrapper: ${GLOBAL_WRAPPER_RESULT}
- Cache seed: ${SEED_CACHE_RESULT}
- Index freshness: ${INDEX_FRESHNESS_RESULT}

Next steps:
1. python scripts/check_startup_readiness.py --format json
2. scripts/gloggur status --json
EOF
