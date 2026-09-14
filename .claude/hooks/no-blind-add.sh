#!/usr/bin/env bash
# PreToolUse on Bash: refuse `git add -A` / `git add .` and friends.
#
# The repo root collects large untracked artefacts (flight*.gif, tens of MB). Staging
# everything sweeps them into history, where they are painful to remove.
set -uo pipefail

cmd=$(jq -r '.tool_input.command // empty')

# `git`, then any number of global options (some take a value: -C <path>, -c k=v), then
# `add`, then any number of its own flags, then the catch-everything pathspec.
if grep -qE 'git([[:space:]]+-[^[:space:]]+([[:space:]]+[^-[:space:]][^[:space:]]*)?)*[[:space:]]+add([[:space:]]+-[^[:space:]]+)*[[:space:]]+(-A|--all|\.|:/)([[:space:]]|$)' <<<"$cmd"; then
    cat >&2 <<'MSG'
Blocked: stage files by name, not the whole tree.

The repository root holds large untracked artefacts (flight*.gif) that must not be
committed. Run `git status --short` and `git add` the specific paths you changed.
MSG
    exit 2
fi
exit 0
