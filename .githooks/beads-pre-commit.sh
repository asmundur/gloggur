#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "$0")" && pwd)/_common.sh"

cd "${repo_root}"

main_git_index_path() {
  env -u GIT_INDEX_FILE git rev-parse --git-path index
}

uses_partial_commit_index() {
  local default_index=""
  default_index="$(main_git_index_path)"
  [[ "${GIT_INDEX_FILE:-${default_index}}" != "${default_index}" ]]
}

main_index_has_tracked_beads_change() {
  local default_index=""
  default_index="$(main_git_index_path)"
  ! GIT_INDEX_FILE="${default_index}" git diff --cached --quiet --no-ext-diff -- .beads/
}

current_commit_includes_tracked_beads_change() {
  ! git diff --cached --quiet --no-ext-diff -- .beads/
}

abort_partial_commit_if_beads_would_leak() {
  if uses_partial_commit_index && ( main_index_has_tracked_beads_change || current_commit_includes_tracked_beads_change ); then
    echo >&2 "beads: tracked .beads/ files are involved in this commit, but git is using an explicit pathspec or partial index."
    echo >&2 "beads: rerun 'git commit' without pathspecs so the Beads snapshot can be included in the same commit."
    echo >&2 "beads: alternatively include the relevant tracked .beads/ files explicitly in the commit."
    exit 1
  fi
}

run_beads_hook pre-commit "$@"
if [[ -f "${repo_root}/.beads/issues.jsonl" ]]; then
  bd export --no-memories -o "${repo_root}/.beads/issues.jsonl"
fi

while IFS= read -r tracked; do
  abs="${repo_root}/${tracked}"
  [[ -f "${abs}" ]] || continue
  if ! git diff --quiet --no-ext-diff -- "${tracked}" \
  || ! git diff --cached --quiet --no-ext-diff -- "${tracked}"; then
    git add "${tracked}"
  fi
done < <(git ls-files -- .beads/)

abort_partial_commit_if_beads_would_leak
