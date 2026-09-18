#!/usr/bin/env bash
# PreToolUse on Bash: refuse to start anything that commands real hardware.
#
# These entry points connect to a Crazyflie and spin motors next to a person. --dry-run is
# not exempt: it still opens the radio link. drones-camera spins nothing, but it connects to
# the drone's AI-deck all the same. A human starts these, never the agent.
#
# The command is split on shell separators and each segment judged on its own, so that
# *mentioning* an entry point to a read-only tool (grep drones-fly-policy README.md) is
# allowed while *running* one is not.
set -uo pipefail

cmd=$(jq -r '.tool_input.command // empty')

LIVE='drones-(fly-policy|fly-square|wall-avoid|web|fpv|camera)'
# First words that never fly anything: they read, search or print.
READONLY='grep|rg|ag|cat|bat|head|tail|sed|awk|less|more|find|fd|ls|echo|printf|wc|sort|uniq|cut|diff|git|man|which|type|jq|xargs|test|rm|cp|mv|touch|mkdir|chmod'

# One segment per line, split on ; & | && || and newlines.
while IFS= read -r segment; do
    [[ -n ${segment//[[:space:]]/} ]] || continue
    grep -qE "(^|[^-[:alnum:]_])${LIVE}([^-[:alnum:]_]|$)" <<<"$segment" || continue

    # Drop leading VAR=value assignments, then look at the command word.
    head_word=$(sed -E 's/^[[:space:]]*([A-Za-z_][A-Za-z0-9_]*=[^[:space:]]+[[:space:]]+)*//' \
                <<<"$segment" | awk '{print $1}')
    head_word=${head_word##*/}
    [[ $head_word =~ ^(${READONLY})$ ]] && continue

    cat >&2 <<'MSG'
Blocked: this command flies or connects to the real drone.

drones-fly-policy, drones-fly-square, drones-wall-avoid, drones-web and drones-fpv all open
a radio link and can spin motors. --dry-run still connects. drones-camera connects to the
AI-deck over Wi-Fi. Only the operator starts these, with the aircraft in sight.

Ask the user to run it, or work in the simulator instead (drones-eval-*, drones-render-*).
MSG
    exit 2
done < <(sed -E 's/(\|\||&&|[;&|])/\n/g' <<<"$cmd")

exit 0
