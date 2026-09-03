#!/bin/bash
# Profile-matrix driver. For each configuration: tear everything down, launch it
# from scratch, wait for readiness, verify the server actually got the profile's
# args, benchmark it REPEATS times, move on.
#
# Runs on the LAPTOP, not on a B300 -- the two hosts have no ssh trust between
# them, so only the laptop can drive both. Same ssh aliases sync.sh uses.
#
#   bash sync.sh push && bash 93_matrix.sh          # all 5 configs, 2 runs each
#   CONFIGS="pd:balanced" bash 93_matrix.sh
#   REPEATS=3 TRANSFER_BACKEND=nixl CONFIGS="pd:low-latency" bash 93_matrix.sh
#
# Why a driver instead of launching by hand: every row has to come from the same
# container generation and the same script revision, or the rows are not
# comparable to each other. The first pass at this matrix was hand-launched
# across several generations, and one config had silently picked up a prefill
# mamba ratio belonging to a different profile -- which only showed up in
# server_args, not in any launcher output. So here the resolved config is read
# back out of the server's own server_args line, and a mismatch skips the config
# instead of producing a plausible-looking number.
set -uo pipefail

cd "$(dirname "$0")"
source ./env_common.sh

# B300-1/B300-2, NOT P6-B300-1/P6-B300-2. Those two aliases point at a
# COLLEAGUE's machines (the "别动" note in ~/.ssh/config), and this file used to
# default to them -- so `bash 93_matrix.sh` sent `docker rm -f kimi-k3-*` at
# someone else's box and then hung on a teardown ssh with no ConnectTimeout,
# looking exactly like a slow launch. sync.sh already carried the same warning
# for the same reason; this file's header claimed it used "the same ssh aliases
# sync.sh uses" while doing the opposite. Caught 2026-09-03 only because their
# instances happened to be stopped, so nothing reached them.
PREFILL_HOST="${PREFILL_HOST:-B300-1}"   # also the standalone + router host
DECODE_HOST="${DECODE_HOST:-B300-2}"
REMOTE="${REMOTE:-/home/ubuntu/kimi-k3-sglang}"

# Refuse to drive a colleague's box even if someone passes it explicitly. This
# driver's very first action per config is `docker rm -f` against fixed container
# names, so a wrong host is destructive before it is ever obviously wrong.
for h in "$PREFILL_HOST" "$DECODE_HOST"; do
    if [[ "$h" == P6-B300* ]]; then
        echo "REFUSING: '$h' is a colleague's machine (see ~/.ssh/config)." >&2
        echo "          Use B300-1/B300-2, or set PREFILL_HOST/DECODE_HOST." >&2
        exit 1
    fi
done
# Preflight both hosts before tearing anything down: 15 s of "unreachable" beats
# minutes of a silent hang that reads as a slow launch.
for h in "$PREFILL_HOST" "$DECODE_HOST"; do
    if ! ssh -o ConnectTimeout=15 -o BatchMode=yes "$h" true 2>/dev/null; then
        echo "REFUSING: host '$h' is unreachable." >&2
        exit 1
    fi
done

REPEATS="${REPEATS:-2}"
# A discarded first run. NOT optional politeness: deep_gemm JIT-compiles inside
# the timed region on the first bench at a given shape, and it faked a 78%
# throughput effect once already (see project_k3_pd_prefill_cap_knee). Warm runs
# measured 6-9% low across all five CAP rows. It is tagged r0 so it lands in
# results/ under its own name and cannot be mistaken for a timed replicate.
WARMUP="${WARMUP:-1}"
ISL="${ISL:-8192}"
OSL="${OSL:-1024}"
NUM_PROMPTS="${NUM_PROMPTS:-64}"
CONCURRENCY="${CONCURRENCY:-32}"
BACKEND="${TRANSFER_BACKEND:-mooncake}"
# "mode:profile" pairs, run in this order.
CONFIGS="${CONFIGS:-pd:low-latency pd:balanced pd:high-throughput standalone:low-latency standalone:balanced}"

LOCAL_RESULTS="${LOCAL_RESULTS:-./results}"
mkdir -p "$LOCAL_RESULTS"
# Same reason the per-run tag below carries the EP axis: a deepep_v2 matrix and a
# plain-TP matrix at the same isl/osl/concurrency would otherwise write the same
# summary, and this file is truncated at start (`: > "$SUMMARY"`), so the second
# sweep deletes the first sweep's summary outright. All three caps go in because
# one summary spans both pd and standalone rows. Empty when EP is off.
MTAG=""
if [[ "${MOE_A2A_BACKEND:-none}" != "none" ]]; then
    MTAG="-${MOE_A2A_BACKEND}-p${PREFILL_CAP}d${DECODE_CAP}s${STANDALONE_CAP}"
    # DECODE_MAXRUN is a second axis, not a detail of DECODE_CAP: a small CAP is
    # only legal with a small max_running_requests, so "CAP=128" and "CAP=512"
    # rows are not comparable unless MAXRUN is held equal -- which means it has to
    # be visible in the name of every file, or one arm silently overwrites the
    # other. Empty (DSPARK's own 48) leaves names unchanged.
    [[ -n "${DECODE_MAXRUN:-}" ]] && MTAG="${MTAG}-mr${DECODE_MAXRUN}"
fi

# Knobs the driver resolves locally but the launchers read from THEIR OWN
# environment: without this list `DECODE_CAP=128 bash 93_matrix.sh` verifies 128
# locally, launches 1024 remotely, and assert_cap skips the config -- a 40-minute
# no-op. Only non-empty values are forwarded so unset stays unset.
fwd() {
    local v out=""
    for v in PREFILL_CAP PREFILL_CHUNK DECODE_CAP DECODE_CHUNK DECODE_MAXRUN \
             DECODE_CGMAXBS STANDALONE_CAP MOE_A2A_BACKEND EP_SIZE \
             DEEPEP_V2_MODE MOE_RUNNER_BACKEND SPEC_BLOCK_SIZE NO_SPEC; do
        [[ -n "${!v:-}" ]] && out="$out $v=${!v}"
    done
    echo "$out"
}
FWD="$(fwd)"
SUMMARY="$LOCAL_RESULTS/matrix-isl${ISL}-osl${OSL}-c${CONCURRENCY}${MTAG}.txt"

say() { echo "$@" | tee -a "$SUMMARY"; }
: > "$SUMMARY"
say "matrix: isl=$ISL osl=$OSL n=$NUM_PROMPTS conc=$CONCURRENCY repeats=$REPEATS backend=$BACKEND"
say "configs: $CONFIGS"
say "started: $(date -u +%FT%TZ)"

# Unmapping ~1.5 TB of model volumes can outlast a single `rm -f`, so the
# container lingers as Exited-but-present and the next `docker run` fails with
# "name already in use". Retry until the name is actually gone (the launchers do
# the same for their own container).
teardown() {
    local host="$1"; shift
    local names="$*"
    # ConnectTimeout, because a host that is simply unreachable otherwise hangs
    # here indefinitely with the driver's last line of output being the config
    # banner -- indistinguishable from a slow weight load.
    ssh -o ConnectTimeout=15 "$host" "for n in $names; do
        for i in \$(seq 1 60); do
            docker inspect \$n >/dev/null 2>&1 || break
            docker rm -f \$n >/dev/null 2>&1 || true
            sleep 2
        done
    done" >/dev/null 2>&1
}

# 60 x 30 s = 30 min. A cold launch is ~4 min with the JIT caches warm, ~10 min
# without, so this only trips on a genuine failure.
wait_ready() {
    local host="$1" port="$2" label="$3" i code
    for i in $(seq 1 60); do
        code=$(ssh -o ConnectTimeout=10 "$host" \
            "curl -s -m 5 -o /dev/null -w '%{http_code}' http://localhost:$port/health" 2>/dev/null)
        [[ "$code" == "200" ]] && { say "  ready: $label after $((i*30))s"; return 0; }
        sleep 30
    done
    say "  TIMEOUT waiting for $label -- skipping this config"
    return 1
}

# Read a knob back out of the server's own server_args line. The launcher echo
# only proves what the script meant to pass; this proves what the server used.
# The pattern here is load-bearing and was WRONG until 2026-09-03: this image
# prints server_args as a Python dict repr -- "'dcp_size': 1" -- while the old
# pattern matched "dcp_size=1". Every key therefore came back <absent>, every PD
# config was skipped, and the driver reported a tidy "finished" having benchmarked
# nothing. Accept both spellings, and tell the two failure kinds apart: a key that
# is missing from server_args entirely is a HARNESS bug and aborts the run, while
# a key whose value differs is a CONFIG bug and skips that config. Collapsing
# those two into one "skip" is what hid this for two whole arms.
assert_arg() {
    local host="$1" name="$2" key="$3" want="$4" line got
    line=$(ssh -o ConnectTimeout=15 "$host" \
        "docker logs $name 2>&1 | grep -o 'server_args=.*' | head -1" 2>/dev/null)
    if [[ -z "$line" ]]; then
        say "  HARNESS ERROR: no server_args line in $name's log -- cannot verify config"
        return 2
    fi
    got=$(printf '%s' "$line" | tr ',' '\n' \
        | sed -nE "s/^[[:space:]]*'?${key}'?[[:space:]]*[:=][[:space:]]*'?([^',)]*)'?.*/\1/p" \
        | head -1)
    got="${got//[[:space:]]/}"
    if [[ -z "$got" ]]; then
        say "  HARNESS ERROR: '$key' absent from $name's server_args -- readback pattern is stale"
        return 2
    fi
    if [[ "$got" != "$want" ]]; then
        say "  CONFIG MISMATCH $name: want $key=$want, server reports '$got'"
        return 1
    fi
    say "  verified $name $key=$want"
}

# assert_arg's blind spot: CAP is passed as an ENV VAR, so it is absent from the
# server's `server_args=` line and assert_arg would report it as <absent> forever.
# A row could therefore run at the wrong capacity -- the single knob that decides
# whether the ElasticBuffer and the graph pool both fit -- and still be recorded
# as verified. Read it from the container instead.
assert_cap() {
    local host="$1" name="$2" want="$3" got
    got=$(ssh "$host" "docker inspect -f '{{range .Config.Env}}{{println .}}{{end}}' $name 2>/dev/null \
        | grep '^SGLANG_DEEPEP_V2_NUM_MAX_DISPATCH_TOKENS_PER_RANK=' | head -1 | cut -d= -f2" 2>/dev/null)
    got="${got//[[:space:]]/}"
    if [[ "$got" != "$want" ]]; then
        say "  CONFIG MISMATCH $name: want CAP=$want, container reports '${got:-<absent>}'"
        return 1
    fi
    say "  verified $name CAP=$want"
}

# rc=2 from assert_arg means the driver cannot see the config at all. That is not
# a config to skip past -- every remaining config would be equally unverifiable,
# so stop and say so instead of producing a "finished" with no rows.
must_arg() {
    assert_arg "$@"; local rc=$?
    if (( rc == 2 )); then
        say "  ABORTING: config readback is broken, not benchmarking blind"
        exit 1
    fi
    return $rc
}

run_bench() {
    local host="$1" endpoint="$2" mode="$3" profile="$4" r="$5"
    local tag="${mode}-${profile}"
    [[ "$mode" == "pd" ]] && tag="${tag}-${BACKEND}"
    # The EP axis belongs in the filename. This tag is passed to 91_bench.sh as
    # TAG=, which overrides the stamp 91_bench.sh would build for itself, so
    # without this a deepep_v2 matrix and a plain-TP matrix at the same profile
    # overwrite each other row for row -- including the .prefill.log/.decode.log
    # snapshots below. Empty when EP is off, so existing filenames are unchanged.
    if [[ "${w_a2a:-none}" != "none" ]]; then
        if [[ "$mode" == "pd" ]]; then
            tag="${tag}-${w_a2a}-p${w_pcap}d${w_dcap}"
            # See MTAG: MAXRUN is an independent axis of the decode arm.
            [[ -n "${DECODE_MAXRUN:-}" ]] && tag="${tag}-mr${DECODE_MAXRUN}"
        else
            tag="${tag}-${w_a2a}-cap${w_scap}"
        fi
    fi
    tag="${tag}-isl${ISL}-osl${OSL}-c${CONCURRENCY}-r${r}"
    say "----- run $r/$REPEATS  tag=$tag -----"
    ssh "$host" "cd $REMOTE && ISL=$ISL OSL=$OSL NUM_PROMPTS=$NUM_PROMPTS \
        CONCURRENCY=$CONCURRENCY MODE=$mode PROFILE=$profile TAG=$tag \
        ENDPOINT=$endpoint bash 91_bench.sh" 2>&1 \
        | grep -E "Successful requests|Benchmark duration|Request throughput|Output token throughput|Total token throughput|Mean TTFT|Median TTFT|Mean TPOT|Median TPOT|Mean ITL|Median ITL|Mean E2E|accept length" \
        | tee -a "$SUMMARY"

    # Snapshot the server side per run: the bench client's percentiles cannot
    # distinguish "generating slower" from "queued behind a preemption", and the
    # container is gone by the time anyone reads the summary. Both PD dcp=8
    # profiles dropped ~39% from their first run to their second (1649->1000,
    # 1676->1030) with identical token counts and unchanged median ITL, while
    # standalone dcp=8 did not -- so only the decode log's per-batch
    # #running/#queue/token-usage lines can say what happened.
    save_server_logs "$mode" "$tag"
}

save_server_logs() {
    local mode="$1" tag="$2"
    if [[ "$mode" == "pd" ]]; then
        ssh "$PREFILL_HOST" "docker logs kimi-k3-prefill > $REMOTE/results/${tag}.prefill.log 2>&1" || true
        ssh "$DECODE_HOST"  "docker logs kimi-k3-decode  > $REMOTE/results/${tag}.decode.log  2>&1" || true
    else
        ssh "$PREFILL_HOST" "docker logs kimi-k3 > $REMOTE/results/${tag}.server.log 2>&1" || true
    fi
}

for cfg in $CONFIGS; do
    mode="${cfg%%:*}"; profile="${cfg#*:}"
    say ""
    say "########## mode=$mode profile=$profile  ($(date -u +%FT%TZ)) ##########"

    # Clean slate: a leftover container from the previous config would otherwise
    # serve this config's traffic with the previous config's args.
    teardown "$PREFILL_HOST" kimi-k3-router kimi-k3-prefill kimi-k3
    teardown "$DECODE_HOST" kimi-k3-decode kimi-k3

    # Resolve the profile locally so the expected values come from the same
    # env_common.sh table the launchers read.
    eval "$(PROFILE=$profile bash -c 'source ./env_common.sh >/dev/null 2>&1
        echo "w_pdcp=$PREFILL_DCP_SIZE w_pmamba=$PREFILL_MAMBA_RATIO w_pmem=$PREFILL_MEM_FRACTION"
        echo "w_ddcp=$DECODE_DCP_SIZE w_dmamba=$DECODE_MAMBA_RATIO w_dmem=$DECODE_MEM_FRACTION"
        echo "w_sdcp=$STANDALONE_DCP_SIZE w_smamba=$STANDALONE_MAMBA_RATIO w_smem=$STANDALONE_MEM_FRACTION"
        echo "w_a2a=$MOE_A2A_BACKEND w_pcap=$PREFILL_CAP w_dcap=$DECODE_CAP w_scap=$STANDALONE_CAP"')"

    if [[ "$mode" == "pd" ]]; then
        # Launch both nodes before waiting on either: they rendezvous over the
        # bootstrap port, and each takes minutes to load 1.5 TB of weights.
        ssh "$PREFILL_HOST" "cd $REMOTE && PROFILE=$profile TRANSFER_BACKEND=$BACKEND$FWD bash 20_launch_prefill.sh" 2>&1 | tee -a "$SUMMARY"
        ssh "$DECODE_HOST"  "cd $REMOTE && PROFILE=$profile TRANSFER_BACKEND=$BACKEND$FWD bash 21_launch_decode.sh"  2>&1 | tee -a "$SUMMARY"
        wait_ready "$PREFILL_HOST" "$PORT" "prefill" || continue
        wait_ready "$DECODE_HOST"  "$PORT" "decode"  || continue

        must_arg "$PREFILL_HOST" kimi-k3-prefill dcp_size "$w_pdcp" || continue
        must_arg "$PREFILL_HOST" kimi-k3-prefill mamba_full_memory_ratio "$w_pmamba" || continue
        must_arg "$PREFILL_HOST" kimi-k3-prefill mem_fraction_static "$w_pmem" || continue
        must_arg "$DECODE_HOST"  kimi-k3-decode  dcp_size "$w_ddcp" || continue
        must_arg "$DECODE_HOST"  kimi-k3-decode  mamba_full_memory_ratio "$w_dmamba" || continue
        must_arg "$DECODE_HOST"  kimi-k3-decode  mem_fraction_static "$w_dmem" || continue
        if [[ "$w_a2a" != "none" ]]; then
            # The two sides run DIFFERENT caps on purpose -- that is the point of
            # doing DeepEP v2 under PD -- so both get checked, separately.
            assert_cap "$PREFILL_HOST" kimi-k3-prefill "$w_pcap" || continue
            assert_cap "$DECODE_HOST"  kimi-k3-decode  "$w_dcap" || continue
            # Unlike CAP, this one IS in server_args, so read it back rather than
            # trusting the flag. It decides whether the a2a budget holds at all
            # (max_running_requests * tokens_per_req <= CAP, deepep_v2.py:257), and
            # DSPARK silently rewrites it to 48 when nothing is passed -- so an
            # unforwarded DECODE_MAXRUN looks identical to a set one in the logs.
            if [[ -n "${DECODE_MAXRUN:-}" ]]; then
                must_arg "$DECODE_HOST" kimi-k3-decode max_running_requests "$DECODE_MAXRUN" || continue
            fi
        fi

        # The router is the easiest thing to forget: without it the bench still
        # "runs" and reports 0 successful requests. Gate on its /health.
        ssh "$PREFILL_HOST" "cd $REMOTE && PROFILE=$profile bash 22_launch_router.sh" >/dev/null 2>&1
        wait_ready "$PREFILL_HOST" "$ROUTER_PORT" "router" || continue
        bench_host="$PREFILL_HOST"; endpoint="localhost:$ROUTER_PORT"
    else
        ssh "$PREFILL_HOST" "cd $REMOTE && PROFILE=$profile$FWD bash 10_launch_standalone.sh" 2>&1 | tee -a "$SUMMARY"
        wait_ready "$PREFILL_HOST" "$PORT" "standalone" || continue

        must_arg "$PREFILL_HOST" kimi-k3 dcp_size "$w_sdcp" || continue
        must_arg "$PREFILL_HOST" kimi-k3 mamba_full_memory_ratio "$w_smamba" || continue
        must_arg "$PREFILL_HOST" kimi-k3 mem_fraction_static "$w_smem" || continue
        [[ "$w_a2a" == "none" ]] || assert_cap "$PREFILL_HOST" kimi-k3 "$w_scap" || continue
        bench_host="$PREFILL_HOST"; endpoint="localhost:$PORT"
    fi

    [[ "$WARMUP" == "1" ]] && { say "----- warmup (DISCARDED) -----"; \
        run_bench "$bench_host" "$endpoint" "$mode" "$profile" 0; }
    for r in $(seq 1 "$REPEATS"); do
        run_bench "$bench_host" "$endpoint" "$mode" "$profile" "$r"
    done
done

say ""
say "finished: $(date -u +%FT%TZ)"
say "raw logs: on the hosts under $REMOTE/results/ -- 'bash sync.sh pull' to fetch"
echo "summary: $SUMMARY"
