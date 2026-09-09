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
#   git diff --cached --stat        # should list the four external/ entries
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
external/Isaac-GR00T	b9955401d50c92a29258732e3ad6ccd579f1bdc0
external/unitree_sdk2_python	a035adeaa6f8ea171bef9a43e8477abb87a0b35e
external/vla_for_simulation_testing	ce0252118e7595dc4785a8b031c35edf534b993a
external/simple_wbc_code	6c0d5101501c04e60cc86642be758d5ae4e78415
ENTRIES

git add .gitmodules
echo
echo "populate them with:  scripts/setup_external.sh"
