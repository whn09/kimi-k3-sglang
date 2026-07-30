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

PREFILL_HOST="${PREFILL_HOST:-P6-B300-1}"   # also the standalone + router host
DECODE_HOST="${DECODE_HOST:-P6-B300-2}"
REMOTE="${REMOTE:-/home/ubuntu/kimi-k3-sglang}"

REPEATS="${REPEATS:-2}"
ISL="${ISL:-8192}"
OSL="${OSL:-1024}"
NUM_PROMPTS="${NUM_PROMPTS:-64}"
CONCURRENCY="${CONCURRENCY:-32}"
BACKEND="${TRANSFER_BACKEND:-mooncake}"
# "mode:profile" pairs, run in this order.
CONFIGS="${CONFIGS:-pd:low-latency pd:balanced pd:high-throughput standalone:low-latency standalone:balanced}"

LOCAL_RESULTS="${LOCAL_RESULTS:-./results}"
mkdir -p "$LOCAL_RESULTS"
SUMMARY="$LOCAL_RESULTS/matrix-isl${ISL}-osl${OSL}-c${CONCURRENCY}.txt"

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
    ssh "$host" "for n in $names; do
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
assert_arg() {
    local host="$1" name="$2" key="$3" want="$4" got
    got=$(ssh "$host" "docker logs $name 2>&1 | grep -o 'server_args=.*' | head -1 \
        | tr ',' '\n' | grep -oE '^ *$key=.*' | head -1 | tr -d ' '" 2>/dev/null)
    if [[ "$got" != "$key=$want" ]]; then
        say "  CONFIG MISMATCH $name: want $key=$want, server reports '${got:-<absent>}'"
        return 1
    fi
    say "  verified $name $key=$want"
}

run_bench() {
    local host="$1" endpoint="$2" mode="$3" profile="$4" r="$5"
    local tag="${mode}-${profile}"
    [[ "$mode" == "pd" ]] && tag="${tag}-${BACKEND}"
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
        echo "w_sdcp=$STANDALONE_DCP_SIZE w_smamba=$STANDALONE_MAMBA_RATIO w_smem=$STANDALONE_MEM_FRACTION"')"

    if [[ "$mode" == "pd" ]]; then
        # Launch both nodes before waiting on either: they rendezvous over the
        # bootstrap port, and each takes minutes to load 1.5 TB of weights.
        ssh "$PREFILL_HOST" "cd $REMOTE && PROFILE=$profile TRANSFER_BACKEND=$BACKEND bash 20_launch_prefill.sh" 2>&1 | tee -a "$SUMMARY"
        ssh "$DECODE_HOST"  "cd $REMOTE && PROFILE=$profile TRANSFER_BACKEND=$BACKEND bash 21_launch_decode.sh"  2>&1 | tee -a "$SUMMARY"
        wait_ready "$PREFILL_HOST" "$PORT" "prefill" || continue
        wait_ready "$DECODE_HOST"  "$PORT" "decode"  || continue

        assert_arg "$PREFILL_HOST" kimi-k3-prefill dcp_size "$w_pdcp" || continue
        assert_arg "$PREFILL_HOST" kimi-k3-prefill mamba_full_memory_ratio "$w_pmamba" || continue
        assert_arg "$PREFILL_HOST" kimi-k3-prefill mem_fraction_static "$w_pmem" || continue
        assert_arg "$DECODE_HOST"  kimi-k3-decode  dcp_size "$w_ddcp" || continue
        assert_arg "$DECODE_HOST"  kimi-k3-decode  mamba_full_memory_ratio "$w_dmamba" || continue
        assert_arg "$DECODE_HOST"  kimi-k3-decode  mem_fraction_static "$w_dmem" || continue

        # The router is the easiest thing to forget: without it the bench still
        # "runs" and reports 0 successful requests. Gate on its /health.
        ssh "$PREFILL_HOST" "cd $REMOTE && PROFILE=$profile bash 22_launch_router.sh" >/dev/null 2>&1
        wait_ready "$PREFILL_HOST" "$ROUTER_PORT" "router" || continue
        bench_host="$PREFILL_HOST"; endpoint="localhost:$ROUTER_PORT"
    else
        ssh "$PREFILL_HOST" "cd $REMOTE && PROFILE=$profile bash 10_launch_standalone.sh" 2>&1 | tee -a "$SUMMARY"
        wait_ready "$PREFILL_HOST" "$PORT" "standalone" || continue

        assert_arg "$PREFILL_HOST" kimi-k3 dcp_size "$w_sdcp" || continue
        assert_arg "$PREFILL_HOST" kimi-k3 mamba_full_memory_ratio "$w_smamba" || continue
        assert_arg "$PREFILL_HOST" kimi-k3 mem_fraction_static "$w_smem" || continue
        bench_host="$PREFILL_HOST"; endpoint="localhost:$PORT"
    fi

    for r in $(seq 1 "$REPEATS"); do
        run_bench "$bench_host" "$endpoint" "$mode" "$profile" "$r"
    done
done

say ""
say "finished: $(date -u +%FT%TZ)"
say "raw logs: on the hosts under $REMOTE/results/ -- 'bash sync.sh pull' to fetch"
echo "summary: $SUMMARY"
