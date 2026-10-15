#!/bin/bash
# LAPTOP-side driver: DeepEP v2 DECODE SM count, 12 vs DeepEP's own pick
# ("auto"), with decode driven to its admission cap. Same method as
# kimi-k3-vllm/94_sm_sweep.sh, so the two frameworks' answers line up.
#
#   bash 96_sm_sweep.sh                       # SMS_LIST="12 0", REPS=2
#   SMS_LIST="12 0 24" REPS=3 bash 96_sm_sweep.sh
#   CONCS=512 bash 96_sm_sweep.sh             # only the decode-full point
#   NO_SPEC=0 bash 96_sm_sweep.sh             # with DSpark (see SPEC below)
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
SPEC=$([[ "$NO_SPEC" == 1 ]] && echo off || echo on)
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
PENV="$COMMON DIST_INIT_ADDR=$B300_1_IP"
DENV="$COMMON DIST_INIT_ADDR=$B300_3_IP DECODE_CAP=$CAPD DECODE_CHUNK=$CAPD DECODE_MAXRUN=$MAXRUN_D DECODE_CGMAXBS=$MAXRUN_D"

say "plan: sms=[$SMS_LIST] x reps=$REPS  spec=$SPEC  decode cap=$CAPD maxrun=$MAXRUN_D  ISL/OSL=$ISL/$OSL  concs=[${CONCS:-<92_sm_bench default>}]"
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

# ---- prefill (once) ----
if sshx "$P0" "curl -sf -m 5 http://localhost:$PORT/health >/dev/null"; then
    say "### prefill already healthy on $P0 -- reusing (it must have been launched with spec=$SPEC)"
else
    teardown_all $P0 $P1 $D0 $D1
    preflight_free_gpus $P0 $P1 $D0 $D1 || exit 1
    say "### prefill TP16/EP16 on $P0+$P1"
    sshx "$P0" "cd $REMOTE && $PENV NODE_RANK=0 bash 20_launch_prefill.sh" 2>&1 | sed 's/^/    /' | tee -a "$SUMMARY"
    sshx "$P1" "cd $REMOTE && $PENV NODE_RANK=1 bash 20_launch_prefill.sh" 2>&1 | sed 's/^/    /' | tee -a "$SUMMARY"
    WAIT_CONTAINER=kimi-k3-prefill wait_ready "$P0" "$PORT" "prefill on $P0" || exit 1
fi

for rep in $(seq 1 "$REPS"); do
    for sms in $SMS_LIST; do
        label="sms${sms}_r${rep}"
        say ""
        say "### decode TP16/EP16 on $D0+$D1, DECODE_DEEPEP_NUM_SMS=$sms  ($label, $(date -u +%FT%TZ))"
        sshx "$P0" "docker rm -f kimi-k3-router >/dev/null 2>&1 || true"
        teardown_all $D0 $D1
        preflight_free_gpus $D0 $D1 || exit 1
        sshx "$D0" "cd $REMOTE && $DENV DECODE_DEEPEP_NUM_SMS=$sms NODE_RANK=0 bash 21_launch_decode.sh" 2>&1 | sed 's/^/    /' | tee -a "$SUMMARY"
        sshx "$D1" "cd $REMOTE && $DENV DECODE_DEEPEP_NUM_SMS=$sms NODE_RANK=1 bash 21_launch_decode.sh" 2>&1 | sed 's/^/    /' | tee -a "$SUMMARY"
        WAIT_CONTAINER=kimi-k3-decode wait_ready "$D0" "$PORT" "decode on $D0" || exit 1

        # The SM count every rank really uses (patches/deepep_v2_log_sms.py).
        # The first dispatch happens in graph capture, so it is logged by now.
        lines=$( { sshx "$D0" "docker logs kimi-k3-decode 2>&1"; sshx "$D1" "docker logs kimi-k3-decode 2>&1"; } | grep '\[k3-sms\]')
        nranks=$(sed -n 's/.* rank=\([0-9]*\).*/\1/p' <<<"$lines" | sort -u | wc -l | tr -d ' ')
        vals=$(sed -n 's/.*dispatch num_sms=\([0-9]*\).*/\1/p' <<<"$lines" | sort -u)
        if [[ "$nranks" != 16 || $(wc -l <<<"$vals") != 1 || -z "$vals" ]]; then
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

        sshx "$P0" "cd $REMOTE && ISL=$ISL OSL=$OSL CAPD=$CAPD MAXRUN_D=$MAXRUN_D SPEC=$SPEC ${CONCS:+CONCS='$CONCS'} REQ_SMS=$sms RESOLVED_SMS=$vals REP=$rep IMAGE=$IMAGE bash 92_sm_bench.sh" 2>&1 | tee -a "$SUMMARY"
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
