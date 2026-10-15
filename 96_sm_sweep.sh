#!/bin/bash
# LAPTOP-side driver: DeepEP v2 DECODE SM count, 12 vs DeepEP's own pick
# ("auto"), with decode driven to its admission cap. Same method as
# kimi-k3-vllm/94_sm_sweep.sh, so the two frameworks' answers line up.
#
#   bash 96_sm_sweep.sh                       # SMS_LIST="12 0", REPS=2
#   SMS_LIST="12 0 24" REPS=3 bash 96_sm_sweep.sh
#   CONCS=512 bash 96_sm_sweep.sh             # only the decode-full point
#   NO_SPEC=0 bash 96_sm_sweep.sh             # with DSpark (see SPEC below)
#   DECODE_ATTN=dp bash 96_sm_sweep.sh        # decode DP attention (= vLLM DP16)
#   DECODE_ATTN=dp SMS_LIST=12 REPS=1 bash 96_sm_sweep.sh   # tp-vs-dp check at SM12
#
# Topology = the pd2x2 arm of 94_matched.sh, the one that matches "2P2D, TP16 /
# EP16": ONE prefill instance TP16/EP16 on B300-1+2 and ONE decode instance
# TP16/EP16 on B300-3+4, both deepep_v2 `hybrid` over EFA (GIN type 5).
#
# - Only DECODE gets the SM knob (DECODE_DEEPEP_NUM_SMS -> SGLANG_DEEPEP_V2_NUM_SMS).
#   Prefill is booted ONCE, keeps DeepEP's choice, and is shared by every run.
# - Decode is rebooted per run: the SM count is fixed per process.
# - Order is INTERLEAVED (12, auto, 12, auto): drift hits both arms, and the
#   spread between replicates of one arm is the noise floor.
# - Speculative decoding is OFF by default (NO_SPEC=1) to match the vLLM run,
#   which follows the recipe without a draft model. With DSpark on, the a2a
#   moves (spec+1)x the tokens per step, so it is a different experiment.
# - Decode admission: no spec => tokens/forward = running, and the runtime check
#   is running <= CAP. CAP=512 is the largest CAP ever booted on a PD decode
#   node, so MAXRUN=CGMAXBS=512.
# Results are pulled after EVERY run.
set -uo pipefail
cd "$(dirname "$0")"
source ./env_common.sh
source ./lib_drive.sh

SMS_LIST="${SMS_LIST:-12 0}"
REPS="${REPS:-2}"
NO_SPEC="${NO_SPEC:-1}"
# Decode attention layout (start_decode.sh): tp (pd2x2 default) | dp (= vLLM DP16) | dcp.
# dcp needs prefill on the same dcp, so prefill is relaunched when it differs.
DECODE_ATTN="${DECODE_ATTN:-tp}"
PREFILL_DCP=$([[ "$DECODE_ATTN" == dcp ]] && echo 16 || echo 1)
SPEC=$([[ "$NO_SPEC" == 1 ]] && echo off || echo on)
IMG="${IMAGE##*:}"; IMG="${IMG%%-*}"   # image generation, e.g. sgla6a8c560
CAPD="${CAPD:-512}"
MAXRUN_D="${MAXRUN_D:-512}"
ISL="${ISL:-1024}"
OSL="${OSL:-2048}"
P0=B300-1; P1=B300-2; D0=B300-3; D1=B300-4
REMOTE="${REMOTE:-/home/ubuntu/kimi-k3-sglang}"
LOCAL_RESULTS="${LOCAL_RESULTS:-./results}"
mkdir -p "$LOCAL_RESULTS/logs"
SUMMARY="$LOCAL_RESULTS/sm_sweep_$(date +%Y%m%d-%H%M%S).summary"

if [[ "$SPEC" == on ]]; then
    tpr=$(( ${SPEC_BLOCK_SIZE:-7} + 1 ))
    (( MAXRUN_D * tpr <= CAPD )) || { say "FATAL: spec on needs MAXRUN_D*$tpr <= CAPD ($MAXRUN_D*$tpr > $CAPD)"; exit 1; }
fi

COMMON="PROFILE=${PROFILE} MOE_A2A_BACKEND=deepep_v2 DISABLE_RADIX=1 NO_SPEC=$NO_SPEC IMAGE=$IMAGE NNODES=2 TP_SIZE=16"
PENV="$COMMON DIST_INIT_ADDR=$B300_1_IP DCP_SIZE=$PREFILL_DCP"
DENV="$COMMON DIST_INIT_ADDR=$B300_3_IP DECODE_ATTN=$DECODE_ATTN DECODE_CAP=$CAPD DECODE_CHUNK=$CAPD DECODE_MAXRUN=$MAXRUN_D DECODE_CGMAXBS=$MAXRUN_D"

say "plan: attn=$DECODE_ATTN (prefill dcp=$PREFILL_DCP) sms=[$SMS_LIST] x reps=$REPS  spec=$SPEC  decode cap=$CAPD maxrun=$MAXRUN_D  ISL/OSL=$ISL/$OSL  concs=[${CONCS:-<92_sm_bench default>}]"
say "image: $IMAGE"

for h in $P0 $P1 $D0 $D1; do
    sshx "$h" "docker image inspect $IMAGE >/dev/null 2>&1" || { say "FATAL: $IMAGE not on $h -- pull it first"; exit 1; }
done

# The decode TP16 instance needs its peers' address; re-check that the static IPs
# still describe these hosts (they change on stop/start).
for h in $P0 $P1 $D0 $D1; do
    got=$(sshx "$h" "ip route get 1.1.1.1 | sed -n 's/.* src \([0-9.]*\).*/\1/p'")
    want=$(ip_of_host "$h")
    [[ "$got" == "$want" ]] || { say "FATAL: $h is $got, env_common.sh says $want -- refresh B300_*_IP"; exit 1; }
done

launch_decode() {   # launch_decode <sms>
    sshx "$P0" "docker rm -f kimi-k3-router >/dev/null 2>&1 || true"
    teardown_all $D0 $D1
    preflight_free_gpus $D0 $D1 || exit 1
    sshx "$D0" "cd $REMOTE && $DENV DECODE_DEEPEP_NUM_SMS=$1 NODE_RANK=0 bash 21_launch_decode.sh" 2>&1 | sed 's/^/    /' | tee -a "$SUMMARY"
    sshx "$D1" "cd $REMOTE && $DENV DECODE_DEEPEP_NUM_SMS=$1 NODE_RANK=1 bash 21_launch_decode.sh" 2>&1 | sed 's/^/    /' | tee -a "$SUMMARY"
}

# running <host> <container>: up AND on $IMAGE. A container left over from
# another image generation must never be reused into this one's rows.
running() {
    [[ "$(sshx "$1" "docker inspect -f '{{.State.Status}} {{.Config.Image}}' $2 2>/dev/null")" == "running $IMAGE" ]]
}

# ---- prefill (once) and the FIRST decode, booted in parallel ----
# Decode does not need prefill up to boot -- it learns the prefill address per
# request from the bootstrap info -- so the two ~10 min cold boots overlap.
# A prefill that is already healthy OR still booting (both ranks' containers
# running) is reused, never relaunched: a relaunch throws away its JIT warmup.
first_sms="${SMS_LIST%% *}"
if running "$P0" kimi-k3-prefill && running "$P1" kimi-k3-prefill \
   && [[ "$(cenv "$P0" kimi-k3-prefill DCP_SIZE)" == "$PREFILL_DCP" ]]; then
    say "### prefill containers already running on $P0+$P1 -- reusing (they must have spec=$SPEC)"
    prefill_launched=0
else
    prefill_launched=1
    teardown_all $P0 $P1
    preflight_free_gpus $P0 $P1 || exit 1
    say "### prefill TP16/EP16 on $P0+$P1"
    sshx "$P0" "cd $REMOTE && $PENV NODE_RANK=0 bash 20_launch_prefill.sh" 2>&1 | sed 's/^/    /' | tee -a "$SUMMARY"
    sshx "$P1" "cd $REMOTE && $PENV NODE_RANK=1 bash 20_launch_prefill.sh" 2>&1 | sed 's/^/    /' | tee -a "$SUMMARY"
fi
cenv() { sshx "$1" "docker inspect -f '{{range .Config.Env}}{{println .}}{{end}}' $2 2>/dev/null | sed -n 's/^$3=//p'"; }
dec_ok() { [[ "$(cenv "$1" kimi-k3-decode SGLANG_DEEPEP_V2_NUM_SMS)" == "$first_sms" \
              && "$(cenv "$1" kimi-k3-decode DECODE_ATTN)" == "$DECODE_ATTN" ]]; }
# A decode is never reused across a prefill restart: it keeps the old prefill's
# Mooncake sessions and sits in "Attempting to reconnect" (seen 2026-10-15).
# So it must have STARTED after the prefill it would serve, and prefill must not
# have been (re)launched by this invocation.
started() { sshx "$1" "docker inspect -f '{{.State.StartedAt}}' $2 2>/dev/null"; }
p_start=$(started "$P0" kimi-k3-prefill)
dec_after_prefill() { [[ -n "$p_start" && "$(started "$1" kimi-k3-decode)" > "$p_start" ]]; }
if (( ! prefill_launched )) && running "$D0" kimi-k3-decode && running "$D1" kimi-k3-decode \
   && dec_ok "$D0" && dec_ok "$D1" && dec_after_prefill "$D0" && dec_after_prefill "$D1"; then
    say "### decode containers already running on $D0+$D1 with sms=$first_sms -- reusing for the first run"
else
    say "### decode TP16/EP16 on $D0+$D1 for the first run (sms=$first_sms), in parallel with prefill"
    launch_decode "$first_sms"
fi
prelaunched=1
WAIT_CONTAINER=kimi-k3-prefill wait_ready "$P0" "$PORT" "prefill on $P0" || exit 1

for rep in $(seq 1 "$REPS"); do
    for sms in $SMS_LIST; do
        label="${DECODE_ATTN}_sms${sms}_r${rep}"
        say ""
        say "### decode TP16/EP16 on $D0+$D1, DECODE_DEEPEP_NUM_SMS=$sms  ($label, $(date -u +%FT%TZ))"
        if (( prelaunched )); then
            prelaunched=0     # booted above, alongside prefill
        else
            launch_decode "$sms"
        fi
        WAIT_CONTAINER=kimi-k3-decode wait_ready "$D0" "$PORT" "decode on $D0" || exit 1

        # The SM count every rank really uses (patches/deepep_v2_log_sms.py).
        # The first dispatch happens in graph capture, so it is logged by now.
        # grep -o, NEVER per line: 16 processes print at once and several records
        # land glued on one line, so a per-line parse sees 9/16 ranks.
        lines=$( { sshx "$D0" "docker logs kimi-k3-decode 2>&1"; sshx "$D1" "docker logs kimi-k3-decode 2>&1"; } \
            | grep -o 'dispatch num_sms=[0-9]* (SGLANG_DEEPEP_V2_NUM_SMS=[0-9]*) rank=[0-9]*')
        nranks=$(sed -n 's/.* rank=\([0-9]*\)$/\1/p' <<<"$lines" | sort -u | wc -l | tr -d ' ')
        vals=$(sed -n 's/^dispatch num_sms=\([0-9]*\) .*/\1/p' <<<"$lines" | sort -u)
        nvals=$(grep -c . <<<"$vals" || true)
        if (( nranks != 16 || nvals != 1 )); then
            say "FATAL: [k3-sms] from $nranks/16 ranks, values '$(tr '\n' ' ' <<<"$vals")'"; exit 1
        fi
        if (( sms > 0 )) && [[ "$vals" != "$sms" ]]; then
            say "FATAL: asked for $sms SMs, DeepEP is using $vals"; exit 1
        fi
        say "  DeepEP resolved num_sms=$vals on all 16 ranks (requested $sms; 0 = auto)"
        note_kv_pool "$D0" kimi-k3-decode $(( ISL + OSL )) "$MAXRUN_D"

        sshx "$P0" "cd $REMOTE && PREFILL_IPS=$B300_1_IP DECODE_IPS=$B300_3_IP ROUTER_POLICY=round_robin IMAGE=$IMAGE bash 22_launch_router.sh" 2>&1 | sed 's/^/    /' | tee -a "$SUMMARY"
        sleep 20
        sshx "$P0" "cd $REMOTE && ENDPOINT=localhost:$ROUTER_PORT bash 90_smoke_test.sh" > "$LOCAL_RESULTS/logs/smoke_${label}.txt" 2>&1
        grep -q '"content"' "$LOCAL_RESULTS/logs/smoke_${label}.txt" \
            || { say "FATAL: smoke test returned no completion -- $LOCAL_RESULTS/logs/smoke_${label}.txt"; exit 1; }
        # Numerics gate (bench/lp_check.py): first-token logprobs vs the validated
        # reference + a greedy continuation per prompt, saved for cross-arm diffs.
        # Mandatory on a new base/patch set: a patch that fails open is silent.
        sshx "$P0" "cd $REMOTE && python3 bench/lp_check.py --url http://localhost:$ROUTER_PORT --ref results/raw/k3_lp_v2.txt --out results/numerics_${label}_img-${IMG}.json" 2>&1 | tee -a "$SUMMARY" | sed 's/^/    /'
        (( ${PIPESTATUS[0]} == 0 )) || { say "FATAL: numerics gate failed on $label -- not benchmarking a wrong model"; exit 1; }

        sshx "$P0" "cd $REMOTE && ISL=$ISL OSL=$OSL CAPD=$CAPD MAXRUN_D=$MAXRUN_D SPEC=$SPEC ATTN=$DECODE_ATTN ${CONCS:+CONCS='$CONCS'} REQ_SMS=$sms RESOLVED_SMS=$vals REP=$rep IMAGE=$IMAGE bash 92_sm_bench.sh" 2>&1 | tee -a "$SUMMARY"
        rc=${PIPESTATUS[0]}
        rsync -az "$P0:$REMOTE/results/" "$LOCAL_RESULTS/" && say "  pulled <- $P0"
        for h in $D0 $D1; do
            sshx "$h" "docker logs kimi-k3-decode 2>&1" > "$LOCAL_RESULTS/logs/decode_${label}_${h}.log" 2>/dev/null || true
        done
        (( rc == 0 )) || { say "FATAL: 92_sm_bench.sh failed on $label (rc=$rc)"; exit 1; }
    done
done

python3 gen_sm_table.py "$LOCAL_RESULTS" | tee "$LOCAL_RESULTS/sm_sweep.md"
say "done. Containers left up (prefill reusable). Free with: teardown on all 4 hosts."
