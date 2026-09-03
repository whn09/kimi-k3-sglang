#!/bin/bash
# Push scripts to both B300 hosts, and pull benchmark results back.
#
#   bash sync.sh push     # local -> hosts (scripts only)
#   bash sync.sh pull     # hosts -> local results/
#   bash sync.sh          # push then pull
#
# NEVER use a bare `rsync --delete` here: $SCRIPT_DIR_HOST also holds results/,
# which only exists on the hosts (91_bench.sh writes it there), so a --delete
# push silently wipes every benchmark log. --delete is scoped to the tracked
# script files via --include/--exclude instead.
set -euo pipefail

cd "$(dirname "$0")"
# DEFAULT DELIBERATELY NOT `P6-B300-*`. Those two ssh aliases point at a
# COLLEAGUE's us-west-2 machines (see the "别动" note in ~/.ssh/config); a bare
# `bash sync.sh` used to rsync this whole tree into them. B300-1/B300-2 are the
# aliases for our own boxes, re-pointed each time a pair is launched.
HOSTS="${HOSTS:-B300-1 B300-2}"
REMOTE="${REMOTE:-/home/ubuntu/kimi-k3-sglang}"

push() {
    for h in $HOSTS; do
        rsync -az --exclude 'results/' --exclude '.git/' ./ "$h:$REMOTE/"
        echo "pushed -> $h"
    done
}

pull() {
    mkdir -p results
    for h in $HOSTS; do
        rsync -az "$h:$REMOTE/results/" results/ 2>/dev/null && echo "pulled <- $h" \
            || echo "no results on $h"
    done
}

case "${1:-both}" in
    push) push ;;
    pull) pull ;;
    both) push; pull ;;
    *) echo "usage: $0 [push|pull]" >&2; exit 1 ;;
esac
