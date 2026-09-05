#!/usr/bin/env bash
# The three arms the matched-capacity campaign never ran, plus the gate that
# keeps them from destroying somebody else's job.
#
#   E  MACHINE_BUDGET=2  pd1p1d:tp            -> PD with the plain-TP MoE path.
#         Without it PD's effect cannot be separated from DeepEP v2's, because
#         every PD arm measured so far runs v2. It is a CLEAN control: at
#         isl=8192 the prefill role is single-chunk either way (A2A=none puts
#         PREFILL_CHUNK at 16384, v2 at PREFILL_CAP=8192), and the decode role
#         is cap=512 chunk=512 in both.
#      MACHINE_BUDGET=2  agg2:v2 @ CAP 8192   -> v2 at a chunk that clears
#         isl=8192 in ONE pass, like its TP counterpart's 16384. The measured
#         agg2:v2 ran chunk=1024 = 8 chunks (env_common.sh:342-346), which is
#         why its 0.20x is withdrawn. agg2:tp does NOT need rerunning: for
#         A2A=none, build_deepep_envs returns before using CAP, so CAP is a
#         no-op there and only CHUNK differs -- and 16384 vs 8192 is 1 chunk
#         vs 1 chunk at isl=8192.
#   F  4 machines        agg2x2node:v2 @ 8192 -> aggregated EP=16 across 2
#         nodes: the only cross-node row that is not also a PD row.
#   G  MACHINE_BUDGET=2  pd1p1d:v2 @ D mem-fraction 0.90/0.88/0.86 -> the fix for
#         PD's TTFT. The cause is the decode KV pool (29 resident against
#         MAXRUN=48), not admission and not Mooncake; ~14.9 GB/GPU of slack is
#         idle. Ladder steps DOWN only on a boot failure -- see the block above
#         the loop. Compared against the PUBLISHED unstamped pd1p1d:v2 row.
#
# Stage E completed on 2026-09-05 (pd1p1d:tp, salvaged) before the hosts were
# terminated; E's agg2:v2, F and G have never run. E is left in place because it
# is idempotent -- 94_matched.sh now pulls after every arm, so a repeat costs
# time, not data.
#
# WHY THE GATE: 94_matched.sh runs teardown_all on every host it uses at line
# 269, BEFORE its own preflight_free_gpus at 274. So the built-in safety cannot
# protect a foreign container -- by the time it runs, the container is gone.
# A colleague's hy4-* run occupied all four boxes when this was written.
set -uo pipefail
cd "$HOME/Documents/workspace/moonshot/kimi-k3-sglang" || exit 1

HOSTS="B300-1 B300-2 B300-3 B300-4"
LOG=/tmp/campaign2
PROG=$LOG.progress
IDLE_POLLS="${IDLE_POLLS:-5}"     # consecutive free polls before we launch
POLL="${POLL:-180}"               # seconds between polls -> 15 min of quiet
MAX_WAIT_H="${MAX_WAIT_H:-14}"    # give up rather than wait forever

: > "$PROG"
say() { echo "=== $* $(date -u +%FT%TZ)" | tee -a "$PROG"; }

# ---------------------------------------------------------------- gate --------
# "free" | "busy:<names>" | "unreachable". An ssh failure must NOT read as free.
host_state() {
    local names rc
    names=$(ssh -o ConnectTimeout=10 -o BatchMode=yes \
                -o StrictHostKeyChecking=accept-new "$1" \
                'docker ps --format "{{.Names}}"' 2>/dev/null); rc=$?
    (( rc != 0 )) && { echo "unreachable"; return; }
    # Our own containers are ours to clear; anything else means occupied.
    local foreign
    foreign=$(printf '%s\n' "$names" | grep -v '^$' | grep -v '^kimi-k3' \
              | paste -sd, -)
    [[ -n "$foreign" ]] && echo "busy:$foreign" || echo free
}

gate() {
    local streak=0 deadline=$(( $(date +%s) + MAX_WAIT_H*3600 ))
    while :; do
        local busy=""
        for h in $HOSTS; do
            local st; st=$(host_state "$h")
            [[ "$st" == free ]] || busy="$busy $h=$st"
        done
        if [[ -z "$busy" ]]; then
            streak=$((streak+1))
            say "gate: all 4 hosts free ($streak/$IDLE_POLLS)"
            (( streak >= IDLE_POLLS )) && { say "gate: OPEN"; return 0; }
        else
            (( streak > 0 )) && say "gate: RESET after $streak free polls"
            streak=0
            say "gate: occupied ->$busy"
        fi
        (( $(date +%s) > deadline )) && { say "gate: GAVE UP after ${MAX_WAIT_H}h"; return 1; }
        sleep "$POLL"
    done
}

# --------------------------------------------------------------- stages -------
stage() {  # stage <letter> <VAR=val>...
    local s="$1"; shift
    say "stage $s START  ($*)"
    env "$@" TEARDOWN_AT_END=1 bash 94_matched.sh > "$LOG.$s.log" 2>&1
    say "stage $s END rc=$?"
    # Anomalies get surfaced whether or not verify() is happy about the config.
    local n
    n=$(grep -c 'SKIP' "$LOG.$s.log"); (( n )) && {
        say "stage $s: $n SKIP lines --"; grep 'SKIP' "$LOG.$s.log" | sed 's/^/    /' | tee -a "$PROG"; }
    n=$(grep -ci 'out of memory\|OutOfMemory\|CUDA error' "$LOG.$s.log"); (( n )) && \
        say "stage $s: $n OOM/CUDA-error lines in the log -- READ IT"
}

# One arm's section of a driver log. Each arm is headed (94_matched.sh:262)
#   ##########  arm=agg2 a2a=v2 machines=2  (<ts>)  ##########
# a2a there is the SHORT kind (tp/v2), not the backend name. The trailing space
# in the pattern is load-bearing: it keeps arm=agg2 off arm=agg2x2node.
# Verified against results/matched-mixed-agg2_tp_agg2_v2_pd1p1d-cpm8_16_32.txt:
# 3 arms extract 171/183/186 lines and 9 bench results each, agg2x2node 0.
block() { awk -v pat="arm=$2 a2a=$3 " '
             /^##########  arm=/ { inb = index($0, pat) > 0 }
             inb' "$1"; }

# Did the arm launch at the capacity we asked for? assert_cenv checks cap but
# NOT chunk, and chunk is the axis that invalidated the last campaign -- so read
# it back off the launcher's own echo, scoped to this arm's block. A global grep
# would be satisfied by a DIFFERENT arm's prefill role.
verify() {  # verify <letter> <arm> <a2a> <expected launch fragment>...
    local s="$1" arm="$2" a2a="$3"; shift 3
    local b; b=$(block "$LOG.$s.log" "$arm" "$a2a")
    if [[ -z "$b" ]]; then
        say "VERIFY $arm:$a2a -- NO BLOCK IN $LOG.$s.log (arm never started)"
        return 1
    fi
    local ok=1 frag
    for frag in "$@"; do
        if grep -qF -- "$frag" <<<"$b"; then
            say "VERIFY $arm:$a2a  ok  '$frag'"
        else
            say "VERIFY $arm:$a2a  MISSING '$frag'"; ok=0
        fi
    done
    if (( ! ok )); then
        say "  launch lines actually seen in this block:"
        grep -oE 'a2a=\S+ ep=[0-9]+ mode=\S+ cap=[0-9]+ chunk=[0-9]+' <<<"$b" \
            | sort | uniq -c | sed 's/^/    /' | tee -a "$PROG"
        return 1
    fi
    # A config-correct arm that produced no numbers is still a failure.
    local nres; nres=$(grep -c 'Output token throughput' <<<"$b")
    say "VERIFY $arm:$a2a  bench results in block: $nres"
    (( nres > 0 ))
}

gate || exit 1

# ---- E: the two 2-machine arms. MACHINE_BUDGET=2 truncates ALL_HOSTS with
#         head -n (94_matched.sh:56-58), so only B300-1/B300-2 are touched.
stage E MACHINE_BUDGET=2 STANDALONE_CAP=8192 ARMS="pd1p1d:tp agg2:v2"
verify E pd1p1d tp        'a2a=none ep=8 mode=direct cap=8192 chunk=16384' \
                          'a2a=none ep=8 mode=direct cap=512 chunk=512'
agg_ok=0
verify E agg2 v2           'a2a=deepep_v2 ep=8 mode=direct cap=8192 chunk=8192' \
    && agg_ok=1

# Capacity sizes the DeepEP slab, masked_max_m AND the decode-graph capture
# pool, so 8192 can OOM where 1024 booted. Step down rather than lose the arm.
# 4096 is 2 chunks at isl=8192, not 1 -- better than 8, but say so in the doc.
if (( ! agg_ok )); then
    say "retrying agg2:v2 at STANDALONE_CAP=4096 (2 chunks, not 1 -- note it)"
    stage E2 MACHINE_BUDGET=2 STANDALONE_CAP=4096 ARMS="agg2:v2"
    verify E2 agg2 v2 'a2a=deepep_v2 ep=8 mode=direct cap=4096 chunk=4096' || {
        say "retrying agg2:v2 at STANDALONE_CAP=2048 (4 chunks)"
        stage E3 MACHINE_BUDGET=2 STANDALONE_CAP=2048 ARMS="agg2:v2"
        verify E3 agg2 v2 'a2a=deepep_v2 ep=8 mode=direct cap=2048 chunk=2048'
    }
fi

# ---- F: aggregated EP=16 across 2 nodes, all 4 machines. ep=16 forces gin=5.
stage F STANDALONE_CAP=8192 ARMS="agg2x2node:v2"
if ! verify F agg2x2node v2 'ep=16' 'cap=8192 chunk=8192'; then
    say "retrying agg2x2node:v2 at STANDALONE_CAP=4096"
    stage F2 STANDALONE_CAP=4096 ARMS="agg2x2node:v2"
    verify F2 agg2x2node v2 'ep=16' 'cap=4096 chunk=4096'
fi

# ---- G: the decode KV pool, the actual cause of PD's TTFT -------------------
# Established 2026-09-05 from pd1p1d:v2's own decode logs: TTFT is NOT the
# max_running_requests=48 wall (`#queue-req` 0 in 5067/5067 samples) and NOT
# Mooncake (`#inflight-req` flat at 2-3 across a 4x load range; ~0.6 GB/s of KV;
# the isl=128 row puts the whole PD handoff at ~684 ms, 4% of a 16.0 s TTFT).
# It is the KV pool: 266816 tokens / 9216 per request = 28.9, and `#running-req`
# peaked at exactly 29 with `full token usage` p90 = 0.94. A saturated pool stops
# pre-allocating, so decode withholds the bootstrap handshake and prefill's
# requests sit in `#bootstrap-req` (max 36, nonzero in 79% of samples at c=64) --
# a wait that lands entirely inside TTFT.
#
# The headroom is idle: ~14.9 GB per GPU is still free at steady state inside the
# 0.15 slack. Each +0.01 of mem-fraction moves ~2.66 GB into the pool (~100k
# tokens, ~11 more resident requests at this shape). 48 resident needs ~11.8 GB,
# i.e. ~0.868. So the ladder goes DOWN from 0.90: we want the HIGHEST fraction
# that still survives target-verify capture, and capture is what fails first.
#   0.90 -> ~+13.3 GB, leaves ~1.6 GB.   Expected to be the edge.
#   0.88 -> ~+8.0 GB,  leaves ~6.9 GB, ~60 resident.  The predicted answer.
#   0.86 -> ~+2.7 GB,  leaves ~12.3 GB, ~39 resident. Still above 29.
# Baseline for the comparison is the PUBLISHED pd1p1d:v2 row, which is unstamped
# (= D 0.85); 94_matched.sh stamps -dmf<val> into every tag here so this arm
# cannot overwrite it and gen_synthesis.py cannot average the two together.
# wait_ready now exits on a died container, so a rung that OOMs costs ~3 min.
#
# Do NOT expect this to fix everything: prefill is separately at 92.8% of its own
# prefill-only ceiling at c=64, so 1P1D stays prefill-bound there.
kvfound() {   # kvfound <letter> <arm> <a2a> -- report note_kv_pool's verdict
    local b; b=$(block "$LOG.$1.log" "$2" "$3")
    local lines; lines=$(grep -o 'kv-pool:.*' <<<"$b" | sort -u)
    if [[ -z "$lines" ]]; then
        say "KVPOOL $2:$3 -- no kv-pool line (note_kv_pool did not fire)"
        return 1
    fi
    say "KVPOOL $2:$3 --"; sed 's/^/    /' <<<"$lines" | tee -a "$PROG"
    ! grep -q 'POOL BINDS FIRST' <<<"$lines"
}

# The ladder steps down ONLY on a failure to boot. A rung that boots is the
# highest bootable fraction, hence the biggest pool available -- going lower from
# there would shrink the very thing being fixed, whether or not the pool cleared
# MAXRUN. So a successful rung always ends the ladder; kvfound only decides what
# the log says about it.
for mf in 0.90 0.88 0.86; do
    s="G${mf#0.}"
    stage "$s" MACHINE_BUDGET=2 DECODE_MEM_FRACTION="$mf" ARMS="pd1p1d:v2"
    if verify "$s" pd1p1d v2 "kimi-k3-decode' (profile=low-latency, mem=$mf"; then
        if kvfound "$s" pd1p1d v2; then
            say "stage $s: D=$mf boots and the pool clears MAXRUN -- ladder done"
        else
            say "stage $s: D=$mf boots but the pool STILL binds below MAXRUN."
            say "           Do not step down (that shrinks it further). The next"
            say "           lever is cuda_graph_max_bs, not mem-fraction."
        fi
        break
    fi
    say "stage $s: D=$mf produced no benchable arm -- stepping down"
done
unset mf s

# ---- collect ----
say "pull + regenerate"
bash sync.sh pull > "$LOG.pull.log" 2>&1
say "sync.sh pull rc=$?"
python3 gen_synthesis.py results  > results/SYNTHESIS.txt   2>&1
python3 audit_admission.py results > results/ADMISSION.txt  2>&1
say "ALL DONE -- read $PROG top to bottom before quoting any number"
