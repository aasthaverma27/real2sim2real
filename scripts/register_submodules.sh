#!/usr/bin/env bash
# Re-register the external submodules at the commits this stack was tested with.
#
# Why this exists: submodule entries are "gitlinks" - a path plus a commit SHA,
# with no content in this repository. If the external/ directories are empty on
# disk (a fresh clone, or before `git submodule update --init`), then a blanket
# `git add -A` reads them as deleted and drops the gitlinks from the index. This
# puts them back. It is idempotent and safe to run any time.
#
#   scripts/register_submodules.sh
#   git diff --cached --stat        # should list the external/ entry
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# path <TAB> pinned commit. Bump a SHA only when you have tested against it.
while IFS=$'\t' read -r path sha; do
  [ -z "$path" ] && continue
  mkdir -p "$path"
  git update-index --add --cacheinfo "160000,$sha,$path"
  printf '  registered %-45s @ %s\n' "$path" "${sha:0:10}"
done <<'ENTRIES'
external/GR00T-WholeBodyControl	b042411fae38ee4d1af9aac82a37a1f8d14d6dd0
ENTRIES

git add .gitmodules
echo
echo "populate them with:  scripts/setup_external.sh"
