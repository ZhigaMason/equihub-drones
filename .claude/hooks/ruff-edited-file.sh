#!/usr/bin/env bash
# PostToolUse on Edit|Write: lint just the file that was edited.
#
# Advisory, not blocking: it prints ruff's findings so the agent fixes them in the same
# turn, rather than leaving the tree dirty for the next `ruff check`.
set -uo pipefail

path=$(jq -r '.tool_input.file_path // empty')
[[ $path == *.py ]] || exit 0
[[ -f $path ]] || exit 0

command -v ruff >/dev/null 2>&1 || exit 0

if ! out=$(ruff check --force-exclude --output-format concise "$path" 2>&1); then
    printf 'ruff found problems in %s:\n%s\n' "$path" "$out" >&2
    exit 2   # exit 2 feeds stderr back to the model
fi
exit 0
