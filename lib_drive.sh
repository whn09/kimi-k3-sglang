#!/bin/bash
# Shared LAPTOP-side driver helpers. Sourced by 94_matched.sh.
#
# 93_matrix.sh predates this file and carries its own copies of teardown /
# wait_ready / assert_arg / assert_cap. It is deliberately NOT refactored onto
# this library: it is a working driver that produced published rows, and there is
# no machine available to re-validate a refactor of it. If you touch one, check
# the other.
#
# Everything here runs over ssh from the laptop, because the B300s have no ssh
# trust between them (tested: publickey denied both ways) -- only the laptop can
# see every host, which is also why a bench running on the prefill host cannot
# read the decode container's env.

# Set SUMMARY before calling say(); it is read at call time, so sourcing this
# file before computing the summary path is fine.
say() { echo "$@" | tee -a "${SUMMARY:-/dev/null}"; }

# Every container name this repo ever creates. Teardown must cover all of them:
# a leftover kimi-k3 (standalone) on a host that is about to run a PD decode
# holds all 8 GPUs, and the decode then dies with "memory capacity is
# unbalanced" -- which reads exactly like a colleague's job having taken the box.
OUR_CONTAINERS="kimi-k3-router kimi-k3-agg-router kimi-k3-prefill kimi-k3-decode kimi-k3 kimi-k3-bench"

# ssh wrapper with a timeout and a dry-run mode. DRY_RUN=1 prints the command
# instead of running it, which is the only way to validate a campaign's
# orchestration (arm expansion, rank assignment, router args) with no machines.
# ConnectTimeout is not optional: a host that is simply unreachable otherwise
# hangs forever with the driver's last output being a config banner, which is
# indistinguishable from a slow 1.5 TB weight load.
sshx() {
    local host="$1"; shift
    if [[ "${DRY_RUN:-0}" == "1" ]]; then
        echo "DRY  ssh $host -- $*"
        return 0
    fi
    ssh -o ConnectTimeout="${SSH_TIMEOUT:-15}" "$host" "$@"
}

# sshx for a command whose output is noise on a real run but is the WHOLE POINT of
# a dry run. `sshx ... >/dev/null 2>&1` would swallow the DRY line too, which is
# how a driver comes to have a dry-run mode that cannot show you the two commands
# you actually wanted to check (the router launch and the bench invocation).
sshq() {
    if [[ "${DRY_RUN:-0}" == "1" ]]; then sshx "$@"; else sshx "$@" >/dev/null 2>&1; fi
}

# ---- preflight ----

# P6-B300-* are a COLLEAGUE's us-west-2 machines (the "别动" note in
# ~/.ssh/config). Every driver's first real action is `docker rm -f` against
# fixed names, so a wrong host is destructive before it is obviously wrong.
refuse_colleague_hosts() {
    local h
    for h in "$@"; do
        if [[ "$h" == P6-B300* ]]; then
            say "REFUSING: '$h' is a colleague's machine (see ~/.ssh/config)."
            return 1
        fi
    done
}

preflight_reachable() {
    local h
    for h in "$@"; do
        if [[ "${DRY_RUN:-0}" == "1" ]]; then continue; fi
        if ! ssh -o ConnectTimeout=15 -o BatchMode=yes "$h" true 2>/dev/null; then
            say "REFUSING: host '$h' is unreachable."
            return 1
        fi
    done
}

# Same image ID on every host, not just the same tag. A tag is re-pointable and
# `docker pull` on three hosts at three times gives three images; a campaign
# whose rows came from different builds is not a campaign (see
# reference_pin_vs_float_upstream_ref).
preflight_same_image() {
    local img="$1"; shift
    local h id first=""
    for h in "$@"; do
        if [[ "${DRY_RUN:-0}" == "1" ]]; then continue; fi
        id=$(ssh -o ConnectTimeout=15 "$h" "docker image inspect -f '{{.Id}}' $img 2>/dev/null" 2>/dev/null | tr -d '[:space:]')
        if [[ -z "$id" ]]; then
            say "REFUSING: image '$img' is not on $h."
            return 1
        fi
        if [[ -z "$first" ]]; then first="$id"; say "  image $img on $h: ${id:7:12}"
        elif [[ "$id" != "$first" ]]; then
            say "REFUSING: '$img' on $h is ${id:7:12}, not ${first:7:12} -- different builds."
            return 1
        else
            say "  image $img on $h: ${id:7:12} (match)"
        fi
    done
}

# All 8 GPUs actually free. On 2026-09-04 a colleague's container held GPUs 0-3
# of B300-3 and my decode rank 0 died with pre_model_load_memory=74.2 GB; I had
# checked with `head -2`, which cannot see that. So: check the whole list, and
# refuse rather than launch into someone else's job.
preflight_free_gpus() {
    local h out busy
    for h in "$@"; do
        if [[ "${DRY_RUN:-0}" == "1" ]]; then continue; fi
        out=$(ssh -o ConnectTimeout=15 "$h" \
            "nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null" 2>/dev/null)
        busy=$(grep -c . <<<"$out" || true)
        if (( busy > 0 )); then
            say "REFUSING: $h has $busy live CUDA process(es) -- not ours to evict:"
            say "$(sed 's/^/    /' <<<"$out")"
            say "          (check 'docker ps' on $h; ours are: $OUR_CONTAINERS)"
            return 1
        fi
        say "  gpus free: $h"
    done
}

# ---- lifecycle ----

# `docker rm -f` returns while ~1.5 TB of model volumes are still unmapping, so
# the container lingers as Exited-but-present and the next `docker run` fails
# with "name already in use". Retry until the name is really gone.
teardown_all() {
    local h
    for h in "$@"; do
        sshx "$h" "for n in $OUR_CONTAINERS; do
            for i in \$(seq 1 30); do
                docker inspect \$n >/dev/null 2>&1 || break
                docker rm -f \$n >/dev/null 2>&1 || true
                sleep 2
            done
        done" >/dev/null 2>&1
    done
}

# 60 x 30 s = 30 min. A cold K3 is ~4 min with the JIT caches warm and ~10 min
# without, so this only trips on a real failure.
wait_ready() {
    local host="$1" port="$2" label="$3" i code
    if [[ "${DRY_RUN:-0}" == "1" ]]; then say "  (dry) would wait for $label on $host:$port"; return 0; fi
    for i in $(seq 1 "${READY_STEPS:-60}"); do
        code=$(ssh -o ConnectTimeout=10 "$host" \
            "curl -s -m 5 -o /dev/null -w '%{http_code}' http://localhost:$port/health" 2>/dev/null)
        [[ "$code" == "200" ]] && { say "  ready: $label after $((i*30))s"; return 0; }
        sleep 30
    done
    say "  TIMEOUT waiting for $label ($host:$port)"
    return 1
}

# ---- verification ----

# ONE env var out of a RUNNING container. This is the honest source for every
# axis that is passed with -e and therefore never appears in the server's own
# `server_args=` line: CAP, EP_SIZE, NNODES, TP_SIZE, DEEPEP_V2_MODE,
# MOE_A2A_BACKEND, NCCL_GIN_TYPE. 93_matrix.sh only checked CAP this way and
# consequently could not tell an ep=8 run from an ep=16 one.
read_cenv_remote() {
    local host="$1" name="$2" var="$3"
    ssh -o ConnectTimeout=15 "$host" \
        "docker inspect -f '{{range .Config.Env}}{{println .}}{{end}}' $name 2>/dev/null \
         | grep '^${var}=' | head -1 | cut -d= -f2-" 2>/dev/null | tr -d '[:space:]'
}

assert_cenv() {
    local host="$1" name="$2" var="$3" want="$4" got
    if [[ "${DRY_RUN:-0}" == "1" ]]; then say "  (dry) would assert $host/$name $var=$want"; return 0; fi
    got=$(read_cenv_remote "$host" "$name" "$var")
    if [[ "$got" != "$want" ]]; then
        say "  CONFIG MISMATCH $host/$name: want $var=$want, container has '${got:-<absent>}'"
        return 1
    fi
    say "  verified $host/$name $var=$want"
}

# Read a knob back out of the server's own server_args line -- the launcher echo
# only proves what the script meant to pass. This image prints server_args as a
# Python dict repr ("'dcp_size': 1"); older sglang printed "dcp_size=1", so both
# spellings are accepted. A key MISSING from server_args is a harness bug (rc=2)
# and a key with a different VALUE is a config bug (rc=1); collapsing those two
# into one "skip" once hid a stale pattern for two whole arms.
assert_arg() {
    local host="$1" name="$2" key="$3" want="$4" line got
    if [[ "${DRY_RUN:-0}" == "1" ]]; then say "  (dry) would assert $host/$name $key=$want"; return 0; fi
    line=$(ssh -o ConnectTimeout=15 "$host" \
        "docker logs $name 2>&1 | grep -o 'server_args=.*' | head -1" 2>/dev/null)
    if [[ -z "$line" ]]; then
        say "  HARNESS ERROR: no server_args line in $host/$name's log"
        return 2
    fi
    got=$(printf '%s' "$line" | tr ',' '\n' \
        | sed -nE "s/^[[:space:]]*'?${key}'?[[:space:]]*[:=][[:space:]]*'?([^',)]*)'?.*/\1/p" \
        | head -1)
    got="${got//[[:space:]]/}"
    if [[ -z "$got" ]]; then
        say "  HARNESS ERROR: '$key' absent from $host/$name's server_args -- pattern is stale"
        return 2
    fi
    if [[ "$got" != "$want" ]]; then
        say "  CONFIG MISMATCH $host/$name: want $key=$want, server reports '$got'"
        return 1
    fi
    say "  verified $host/$name $key=$want"
}

# rc=2 means the driver cannot see the config at all -- every remaining arm would
# be equally unverifiable, so stop rather than produce a "finished" with rows
# nobody can trust.
must_arg() {
    assert_arg "$@"; local rc=$?
    if (( rc == 2 )); then
        say "  ABORTING: config readback is broken, not benchmarking blind"
        exit 1
    fi
    return $rc
}

# Same idea as assert_arg but only REPORTS. For keys whose spelling in
# server_args is not proven on this image: worth recording, not worth aborting a
# 40-minute arm over.
note_arg() {
    local host="$1" name="$2" key="$3" line got
    if [[ "${DRY_RUN:-0}" == "1" ]]; then return 0; fi
    line=$(ssh -o ConnectTimeout=15 "$host" \
        "docker logs $name 2>&1 | grep -o 'server_args=.*' | head -1" 2>/dev/null)
    got=$(printf '%s' "$line" | tr ',' '\n' \
        | sed -nE "s/^[[:space:]]*'?${key}'?[[:space:]]*[:=][[:space:]]*'?([^',)]*)'?.*/\1/p" \
        | head -1)
    say "  note $host/$name $key=${got//[[:space:]]/}"
}

# The bench client's percentiles cannot tell "generating slower" from "queued
# behind a preemption", and the container is gone by the time anyone reads the
# summary. Snapshot EVERY node, not just rank 0: dispatch/combine time is
# node-layered and the slow node flips between runs
# (project_deepep_combine_is_node_layered).
snapshot_log() {
    local host="$1" name="$2" tag="$3" remote="$4"
    sshx "$host" "docker logs $name > $remote/results/${tag}.${host}.${name}.log 2>&1" || true
}
