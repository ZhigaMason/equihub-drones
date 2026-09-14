#!/usr/bin/env bash
# Stop: run the non-simulator test subset (~6 s) and report failures.
#
# The full suite needs the sim extra and takes about six minutes, too slow to run on every
# turn. The modules without a `pytest.importorskip` cover the control law, the safety
# layer, the drone-side missions and the web server -- the parts that fly the aircraft.
set -uo pipefail

cd "$CLAUDE_PROJECT_DIR" || exit 0

python=.venv/bin/python
[[ -x $python ]] || python=$(command -v python3) || exit 0

mapfile -t fast < <(grep -L importorskip tests/test_*.py 2>/dev/null)
[[ ${#fast[@]} -gt 0 ]] || exit 0

if ! out=$("$python" -m pytest -q --no-header -p no:cacheprovider "${fast[@]}" 2>&1); then
    printf 'Fast tests are failing:\n\n%s\n' "$(tail -n 30 <<<"$out")" >&2
    exit 2
fi
exit 0
