#!/bin/bash
# MATCHED-CAPACITY CAMPAIGN DRIVER.
#
# The question: at the SAME number of machines, does a PD-disaggregated K3
# deployment (DeepEP v2 + Mooncake, everything cross-node on EFA) beat the same
# machines run as N INDEPENDENT single-node servers?
#
#   4 machines:  pd2p2d / pd1p3d / pd3p1d / pd2x2   vs   agg4
#   2 machines:  pd1p1d                             vs   agg2
#
# Runs on the LAPTOP: the B300s have no ssh trust between them, so only the laptop
# can drive four hosts and read four containers' env.
#
#   bash sync.sh push
#   bash 94_matched.sh                                   # 4-machine mixed campaign
#   WL=prefill ARMS="pd3p1d agg4:tp" bash 94_matched.sh
#   WL=decode  ARMS="pd1p3d agg4:tp" bash 94_matched.sh
#   MACHINE_BUDGET=2 ARMS="pd1p1d agg2:tp agg2:v2" bash 94_matched.sh
#   DRY_RUN=1 bash 94_matched.sh                         # print the plan, touch nothing
#
# WHAT MAKES THE ROWS COMPARABLE, and it is all of these at once:
#   - same machine count per comparison (in the filename as m<N>, because it is
#     the axis the claim turns on)
#   - same client, same concurrency, same request count, same shape (WL)
#   - both arms behind a router, both routers pinned to the same policy
#   - same image ID on every host, asserted before anything is torn down
#   - every launched instance's env read back out of the container: ep, nnodes,
#     mode, gin type, cap. An arm that says "cross-node EP over EFA" and quietly
#     ran ep=8 on NVLink has happened here, and the log said `ep=8` while the
#     intent was 16.
#
# WHAT THIS DRIVER DELIBERATELY DOES NOT DO: it does not pick a winner. Read
# gen_matched_table.py's per-machine columns, and read PLAN.md for which
# workload's answer is the headline and which are diagnostics.
set -uo pipefail

cd "$(dirname "$0")"

# WL has to be exported BEFORE env_common.sh is sourced -- that is where the
# shape presets are resolved.
export WL="${WL:-mixed}"
# The per-role mem-fractions have to be captured BEFORE env_common.sh, because it
# always assigns them (from PROFILE, then two clamps). After sourcing, "was it
# the caller or the profile?" is unanswerable, and forwarding the laptop-resolved
# value would pin this laptop's NNODES=1 clamp onto a cross-node arm.
#
# The same capture also builds MFSTAMP, and that part is not optional. The run
# tag is arm-m<N>-profile-<a2atag>-isl...-c...-r..., which does NOT contain a
# mem-fraction -- so `DECODE_MEM_FRACTION=0.88 ARMS=pd1p1d` produces byte-for-byte
# the same filenames as the published 0.85 row, and `sync.sh pull` (rsync, newer
# wins) would overwrite the baseline with the variant. Stamped only when the
# caller moves a fraction off its profile value, so every existing filename stays
# valid and re-pullable; an UNSTAMPED name therefore means the PROFILE value
# (low-latency: S 0.85 / P 0.92-after-clamp / D 0.85). gen_synthesis.py reads the
# stamp into the arm label so a stamped and an unstamped run can never merge into
# one cell as extra replicates.
MF_FWD=""; MFSTAMP=""
for _v in STANDALONE_MEM_FRACTION PREFILL_MEM_FRACTION DECODE_MEM_FRACTION; do
    [[ -z "${!_v:-}" ]] && continue
    MF_FWD="$MF_FWD $_v=${!_v}"
    # smf / pmf / dmf -- the role has to be in the stamp because prefill and
    # decode want opposite values and a bare "mf0.88" would not say which moved.
    MFSTAMP="$MFSTAMP-$(echo "${_v%_MEM_FRACTION}" | cut -c1 | tr 'A-Z' 'a-z')mf${!_v}"
done
unset _v
source ./env_common.sh
source ./lib_drive.sh

REMOTE="${REMOTE:-/home/ubuntu/kimi-k3-sglang}"
# Every arm below places its first instance on B300-1, so that is the router host
# and the bench host. Keeping the client on a machine that is also serving is not
# ideal, but the PD rows already published did exactly that, and moving it for one
# arm only would introduce a difference the table cannot see.
BENCH_HOST="${BENCH_HOST:-B300-1}"
ALL_HOSTS="${ALL_HOSTS:-B300-1 B300-2 B300-3 B300-4}"
# When fewer machines are available than the alias list names, say so HERE rather
# than letting the reachability preflight abort the whole campaign. It truncates
# ALL_HOSTS to the first N and skips any arm that needs more than N machines --
# the common case is "two boxes freed up, run the 2-machine control now".
MACHINE_BUDGET="${MACHINE_BUDGET:-}"
if [[ -n "$MACHINE_BUDGET" ]]; then
    ALL_HOSTS="$(echo $ALL_HOSTS | tr ' ' '\n' | head -n "$MACHINE_BUDGET" | tr '\n' ' ')"
    ALL_HOSTS="${ALL_HOSTS% }"
fi

# 2 timed replicates after a discarded warmup. The warmup is NOT politeness:
# deep_gemm JIT-compiles inside the timed region on the first bench at a given
# shape and faked a 78% throughput effect once already. It is tagged r0 so it
# lands in results/ under its own name and cannot be mistaken for a replicate.
REPEATS="${REPEATS:-2}"
WARMUP="${WARMUP:-1}"
# CONCURRENCY SCALES WITH THE MACHINE COUNT, and this is the single most
# important decision in the whole campaign.
#
# At the published operating point (c=32) ONE standalone node already produces
# 1441-1506 out tok/s, while the 4-machine pd2x2 arm produced 1421.72 and only
# achieved 27.30 of its 32 concurrent slots. Both arms were CLIENT-limited: 32
# in-flight requests cannot keep 4 machines busy, so a c=32 comparison at 4
# machines measures the client, awards the win to whichever arm has the lower
# per-request latency, and would report "PD loses to a single node" -- a true
# sentence about a meaningless experiment.
#
# So offered load per machine is what is held constant: c = machines * CONC_PER_MACHINE.
# Both arms of a comparison have the same machine count, hence the same c, so the
# comparison stays matched; and the per-machine columns stay comparable ACROSS
# machine counts, which is what "does it scale" means.
#   8  -> below saturation, the latency reading
#   16 -> near the published knee
#   32 -> above the decode-slot ceiling on purpose (2 decode nodes at
#         DECODE_CAP=512 give 48+48=96 slots), i.e. the throughput reading
# Set CONCURRENCIES to pin absolute values instead (it then applies to every arm,
# and arms with different machine counts are no longer iso-load).
CONC_PER_MACHINE="${CONC_PER_MACHINE:-8 16 32}"
CONCURRENCIES="${CONCURRENCIES:-}"
# Radix cache OFF in every arm. --flush-cache alone leaves the question of
# whether the router broadcasts the flush to all N workers, and a cache that
# helps the baseline but not the PD arm (or the reverse) is invisible in the
# output. Turning it off costs both arms the same thing and removes the question.
# Set DISABLE_RADIX=0 to measure with prefix caching on, deliberately.
DISABLE_RADIX="${DISABLE_RADIX:-1}"

# Default = the 4-machine mixed campaign, in the order that answers the question
# fastest: the strongest baseline first, so that if PD loses we already know what
# it lost to.
ARMS="${ARMS:-agg4:tp agg4:v2 pd2p2d pd2x2}"

# Leave the last arm's containers up by default so its logs can be read and a
# follow-up point can be taken without a 6-minute reload. Set 1 to free the
# machines when the campaign ends.
TEARDOWN_AT_END="${TEARDOWN_AT_END:-0}"

# Knobs the driver resolves locally but the launchers read from THEIR OWN
# environment. Without forwarding, `PREFILL_CAP=4096 bash 94_matched.sh` verifies
# 4096 locally, launches 8192 remotely, and every arm is skipped on a mismatch --
# a multi-hour no-op (93_matrix.sh learned this the hard way).
# The derived values (PREFILL_CHUNK / STANDALONE_CHUNK) are deliberately NOT
# forwarded: they depend on MOE_A2A_BACKEND, which differs per arm, so a value
# computed on the laptop would pin the deepep_v2 chunk onto the plain-TP arm.
FWD=""
for _v in IMAGE PREFILL_CAP DECODE_CAP STANDALONE_CAP DECODE_MAXRUN DECODE_CGMAXBS \
          MOE_RUNNER_BACKEND TRANSFER_BACKEND SPEC_BLOCK_SIZE NO_SPEC MEM_FRACTION; do
    [[ -n "${!_v:-}" ]] && FWD="$FWD $_v=${!_v}"
done
unset _v
# MEM_FRACTION above is role-BLIND: in a PD arm it hits prefill and decode alike,
# and those want opposite values (prefill needs 0.92 at TP=8 or it has no KV pool
# at all; decode needs headroom for target-verify capture). The three role vars
# are forwarded instead, each read by exactly one launcher.
FWD="$FWD$MF_FWD"

if [[ -z "${WL_PPC:-}" ]]; then
    say "ERROR: WL='$WL' has no preset. This driver needs a named shape"
    say "       (mixed|prefill|decode) so that ISL/OSL/request count are not"
    say "       three separate things to get wrong per arm."
    exit 1
fi

# ---- arm layouts ----
# ROLE:HOST[+HOST...]  per INSTANCE, space-separated.
#   S = standalone (unified server)   P = PD prefill   D = PD decode
#   `+` joins hosts into ONE instance spanning them; the first host is node-rank 0
#       and is the only one that serves HTTP, so it is the address a router gets.
# TP_SIZE is 8 per node, so a 2-host instance is TP=16 / ep=16 / hybrid, which is
# the only way to get EP > 8 on 8-GPU boxes and therefore the only way the
# expert-parallel a2a touches EFA at all.
arm_spec() {
    case "$1" in
        # --- baselines: N independent single-node servers ---
        agg1)       echo "S:B300-1" ;;
        agg2)       echo "S:B300-1 S:B300-2" ;;
        agg4)       echo "S:B300-1 S:B300-2 S:B300-3 S:B300-4" ;;
        # --- unified but cross-node (the 285 tok/s arm: CHUNK is clamped to CAP,
        #     and CAP cannot rise because decode graph capture must fit) ---
        uni2node)   echo "S:B300-1+B300-2" ;;
        agg2x2node) echo "S:B300-1+B300-2 S:B300-3+B300-4" ;;
        # --- PD, ep=8 direct on both sides: EFA carries the Mooncake KV
        #     transfer, the a2a stays on NVLink ---
        pd1p1d)     echo "P:B300-1 D:B300-2" ;;
        pd2p2d)     echo "P:B300-1 P:B300-2 D:B300-3 D:B300-4" ;;
        pd1p3d)     echo "P:B300-1 D:B300-2 D:B300-3 D:B300-4" ;;
        pd3p1d)     echo "P:B300-1 P:B300-2 P:B300-3 D:B300-4" ;;
        # --- PD with cross-node EP on BOTH sides: ep=16 hybrid, gin type 5, so
        #     EFA carries the a2a AND the KV transfer. 1421.72 tok/s on
        #     2026-09-04 ---
        pd2x2)      echo "P:B300-1+B300-2 D:B300-3+B300-4" ;;
        *) return 1 ;;
    esac
}

LOCAL_RESULTS="${LOCAL_RESULTS:-./results}"
mkdir -p "$LOCAL_RESULTS"
# The summary name carries every axis of the CAMPAIGN, for the same reason each
# run's name carries every axis of the run: this file is truncated at start, so a
# second campaign with a different arm list would otherwise delete the first.
if [[ -n "$CONCURRENCIES" ]]; then
    CSTAMP="c$(echo "$CONCURRENCIES" | tr ' ' '_')"
else
    CSTAMP="cpm$(echo "$CONC_PER_MACHINE" | tr ' ' '_')"
fi
CTAG="matched-${WL}-$(echo "$ARMS" | tr ' :' '__')-${CSTAMP}${MFSTAMP}"
SUMMARY="$LOCAL_RESULTS/${CTAG}.txt"
: > "$SUMMARY"

say "campaign: wl=$WL isl=$WL_ISL osl=$WL_OSL ppc=$WL_PPC repeats=$REPEATS warmup=$WARMUP"
if [[ -n "$CONCURRENCIES" ]]; then
    say "conc    : '$CONCURRENCIES' (absolute, NOT iso-load across machine counts)"
else
    say "conc    : ${CONC_PER_MACHINE} per machine (iso-load)"
fi
say "arms    : $ARMS"
say "image   : $IMAGE   profile=$PROFILE  radix=$([[ "$DISABLE_RADIX" == 1 ]] && echo OFF || echo on)"
say "policy  : router=${ROUTER_POLICY:-round_robin (pinned by this driver)}"
say "started : $(date -u +%FT%TZ)"
[[ "${DRY_RUN:-0}" == "1" ]] && say "*** DRY RUN -- no host is touched ***"

# Pin the routing policy for both routers so it can never be the reason one arm
# won. Exported into every remote launcher invocation via FWD_ROUTER below.
ROUTER_POLICY="${ROUTER_POLICY:-round_robin}"

refuse_colleague_hosts $ALL_HOSTS || exit 1
preflight_reachable $ALL_HOSTS || exit 1
preflight_same_image "$IMAGE" $ALL_HOSTS || exit 1

for arm_entry in $ARMS; do
    arm="${arm_entry%%:*}"
    a2a_kind="${arm_entry#*:}"
    [[ "$a2a_kind" == "$arm" ]] && a2a_kind=v2

    spec="$(arm_spec "$arm")" || { say ""; say "SKIP unknown arm '$arm'"; continue; }

    case "$a2a_kind" in
        v2) A2A=deepep_v2 ;;
        tp) A2A=none ;;
        *)  say "SKIP '$arm_entry': a2a must be v2 or tp"; continue ;;
    esac

    # Parse the spec into per-instance arrays.
    I_ROLE=(); I_HOSTS=(); I_NN=(); I_RANK0=()
    ARM_HOSTS=""; machines=0; nprefill=0; ndecode=0; nstand=0
    bad=0
    for tok in $spec; do
        role="${tok%%:*}"; hostlist="${tok#*:}"
        nodes="${hostlist//+/ }"
        nn=$(echo $nodes | wc -w | tr -d ' ')
        rank0=$(echo $nodes | awk '{print $1}')
        for h in $nodes; do
            ip_of_host "$h" >/dev/null || bad=1
            ARM_HOSTS="$ARM_HOSTS $h"
        done
        I_ROLE+=("$role"); I_HOSTS+=("$nodes"); I_NN+=("$nn"); I_RANK0+=("$rank0")
        machines=$(( machines + nn ))
        case "$role" in
            P) nprefill=$(( nprefill + 1 )) ;;
            D) ndecode=$(( ndecode + 1 )) ;;
            S) nstand=$(( nstand + 1 )) ;;
            *) say "SKIP '$arm': unknown role '$role' in spec"; bad=1 ;;
        esac
    done
    (( bad )) && continue

    # Topology sanity that is cheap here and expensive later.
    if (( nprefill > 0 && ndecode == 0 )) || (( ndecode > 0 && nprefill == 0 )); then
        say "SKIP '$arm': a PD arm needs both sides (a prefill node cannot serve alone)"
        continue
    fi
    if (( nstand > 0 && nprefill + ndecode > 0 )); then
        say "SKIP '$arm': cannot mix standalone and PD instances in one arm"
        continue
    fi
    # An arm that names a host outside the available set is skipped, not attempted:
    # its rank-0 would come up and its rank-1 would never join, and the failure
    # would arrive 30 minutes later as a readiness timeout.
    for h in $ARM_HOSTS; do
        if ! grep -qw "$h" <<<"$ALL_HOSTS"; then
            say "SKIP '$arm': needs $h, which is not in ALL_HOSTS ($ALL_HOSTS)"
            bad=1
        fi
    done
    (( bad )) && continue
    if ! grep -qw "$BENCH_HOST" <<<"$ARM_HOSTS"; then
        say "SKIP '$arm': BENCH_HOST=$BENCH_HOST is not in the layout ($ARM_HOSTS)"
        continue
    fi

    if [[ "$A2A" == "none" ]] && (( nprefill > 0 )); then
        say "NOTE: '$arm_entry' runs PD with the plain-TP MoE path. That is a control,"
        say "      not the deployment -- the PD claim is about DeepEP v2 + Mooncake."
    fi

    say ""
    say "##########  arm=$arm a2a=$a2a_kind machines=$machines  ($(date -u +%FT%TZ))  ##########"
    say "layout  : $spec"

    # Clean slate on ALL hosts, not just this arm's: a leftover instance on an
    # unused host still holds its GPUs, and the next arm may need them. This has
    # to come BEFORE the free-GPU check or arm 2 would refuse to start on arm 1's
    # own containers.
    teardown_all $ALL_HOSTS

    # GPUs must be free on every host this arm uses, checked across all 8 -- a
    # colleague's job on GPUs 0-3 once killed a decode rank 0 with
    # "memory capacity is unbalanced" and it read like a launcher bug.
    preflight_free_gpus $ARM_HOSTS || { say "SKIP '$arm': hosts are busy"; continue; }

    # ---- launch ----
    launch_failed=0
    for i in "${!I_ROLE[@]}"; do
        role="${I_ROLE[$i]}"; nodes="${I_HOSTS[$i]}"; nn="${I_NN[$i]}"; rank0="${I_RANK0[$i]}"
        rank0_ip="$(ip_of_host "$rank0")" || { launch_failed=1; break; }
        case "$role" in
            S) script=10_launch_standalone.sh ;;
            P) script=20_launch_prefill.sh ;;
            D) script=21_launch_decode.sh ;;
        esac
        rank=0
        for h in $nodes; do
            # TP is 8 GPUs per node; EP_SIZE defaults to TP_SIZE, and env_common
            # upgrades `direct` to `hybrid` and clamps mem-fraction to 0.78 once
            # NNODES > 1. Nothing else has to be passed for a cross-node instance.
            env_line="PROFILE=$PROFILE MOE_A2A_BACKEND=$A2A DISABLE_RADIX=$DISABLE_RADIX"
            env_line="$env_line NNODES=$nn NODE_RANK=$rank TP_SIZE=$(( nn * 8 ))"
            (( nn > 1 )) && env_line="$env_line DIST_INIT_ADDR=$rank0_ip"
            say "  launch $role rank$rank on $h  ($env_line$FWD)"
            sshx "$h" "cd $REMOTE && $env_line$FWD bash $script" 2>&1 | sed 's/^/    /' | tee -a "$SUMMARY"
            rank=$(( rank + 1 ))
        done
    done
    (( launch_failed )) && { say "SKIP '$arm': could not resolve a host IP"; continue; }

    # ---- wait ----
    ready=1
    for i in "${!I_ROLE[@]}"; do
        case "${I_ROLE[$i]}" in
            S) WAIT_CONTAINER=kimi-k3 ;;
            P) WAIT_CONTAINER=kimi-k3-prefill ;;
            D) WAIT_CONTAINER=kimi-k3-decode ;;
        esac
        export WAIT_CONTAINER
        wait_ready "${I_RANK0[$i]}" "$PORT" "${I_ROLE[$i]}#$i on ${I_RANK0[$i]}" || ready=0
    done
    unset WAIT_CONTAINER
    (( ready )) || { say "SKIP '$arm': an instance never became ready"; continue; }

    # ---- verify what actually launched, on EVERY node ----
    ok=1
    for i in "${!I_ROLE[@]}"; do
        role="${I_ROLE[$i]}"; nodes="${I_HOSTS[$i]}"; nn="${I_NN[$i]}"
        case "$role" in
            S) cname=kimi-k3;         cap="$STANDALONE_CAP" ;;
            P) cname=kimi-k3-prefill; cap="$PREFILL_CAP" ;;
            D) cname=kimi-k3-decode;  cap="$DECODE_CAP" ;;
        esac
        want_mode=direct; want_gin=3
        if (( nn > 1 )); then want_mode=hybrid; want_gin=5; fi
        for h in $nodes; do
            assert_cenv "$h" "$cname" NNODES "$nn" || ok=0
            assert_cenv "$h" "$cname" TP_SIZE "$(( nn * 8 ))" || ok=0
            assert_cenv "$h" "$cname" MOE_A2A_BACKEND "$A2A" || ok=0
            if [[ "$A2A" != "none" ]]; then
                assert_cenv "$h" "$cname" EP_SIZE "$(( nn * 8 ))" || ok=0
                assert_cenv "$h" "$cname" DEEPEP_V2_MODE "$want_mode" || ok=0
                assert_cenv "$h" "$cname" NCCL_GIN_TYPE "$want_gin" || ok=0
                assert_cenv "$h" "$cname" \
                    SGLANG_DEEPEP_V2_NUM_MAX_DISPATCH_TOKENS_PER_RANK "$cap" || ok=0
            fi
        done
        # server_args readback on rank 0 only: proves what the SERVER used rather
        # than what the launcher meant to pass. These three keys are known to
        # exist on this image; anything less certain goes through note_arg.
        note_arg "${I_RANK0[$i]}" "$cname" chunked_prefill_size
        note_arg "${I_RANK0[$i]}" "$cname" mem_fraction_static
        note_arg "${I_RANK0[$i]}" "$cname" max_running_requests
        # The KV pool caps resident requests independently of max_running_requests
        # and is the thing mem_fraction_static actually buys, so read it back for
        # every role that decodes. tokens-per-request is ISL+OSL at this shape;
        # DECODE_MAXRUN is unset unless the caller pinned it, and DSPARK forces 48.
        if [[ "$role" != "P" ]]; then
            note_kv_pool "${I_RANK0[$i]}" "$cname" \
                "$(( WL_ISL + WL_OSL ))" "${DECODE_MAXRUN:-48}"
        fi
    done
    (( ok )) || { say "SKIP '$arm': config mismatch -- refusing to benchmark an arm that is not what its name says"; continue; }

    # ---- router ----
    if (( nprefill > 0 )); then
        pips=""; dips=""
        for i in "${!I_ROLE[@]}"; do
            rip="$(ip_of_host "${I_RANK0[$i]}")"
            [[ "${I_ROLE[$i]}" == "P" ]] && pips="$pips $rip"
            [[ "${I_ROLE[$i]}" == "D" ]] && dips="$dips $rip"
        done
        say "  router  : ${nprefill}P${ndecode}D  prefill='${pips# }' decode='${dips# }'"
        sshq "$BENCH_HOST" "cd $REMOTE && PROFILE=$PROFILE ROUTER_POLICY=$ROUTER_POLICY$FWD \
            PREFILL_IPS='${pips# }' DECODE_IPS='${dips# }' bash 22_launch_router.sh"
        wait_ready "$BENCH_HOST" "$ROUTER_PORT" "pd router" || { say "SKIP '$arm': router"; continue; }
        bench_mode=pd; endpoint="localhost:$ROUTER_PORT"
    elif (( nstand > 1 )); then
        aips=""
        for i in "${!I_ROLE[@]}"; do aips="$aips $(ip_of_host "${I_RANK0[$i]}")"; done
        say "  router  : ${nstand} independent workers  '${aips# }'"
        sshq "$BENCH_HOST" "cd $REMOTE && PROFILE=$PROFILE ROUTER_POLICY=$ROUTER_POLICY$FWD \
            AGG_IPS='${aips# }' bash 23_launch_agg_router.sh"
        wait_ready "$BENCH_HOST" "$ROUTER_PORT" "agg router" || { say "SKIP '$arm': router"; continue; }
        bench_mode=agg; endpoint="localhost:$ROUTER_PORT"
    else
        # A single instance needs no router. It is then NOT matched against the
        # multi-machine arms on hop count -- keep agg1/uni2node as reference rows,
        # not as the baseline in a claim.
        say "  router  : none (single instance; this row carries one hop fewer)"
        bench_mode=standalone; endpoint="localhost:$PORT"
    fi

    # ---- bench ----
    if [[ "$A2A" == "none" ]]; then
        a2atag="tponly"
    elif (( nprefill > 0 )); then
        a2atag="deepep_v2-p${PREFILL_CAP}d${DECODE_CAP}"
    else
        a2atag="deepep_v2-cap${STANDALONE_CAP}"
    fi

    # Iso-load per machine unless CONCURRENCIES pins absolute values.
    if [[ -n "$CONCURRENCIES" ]]; then
        arm_concs="$CONCURRENCIES"
    else
        arm_concs=""
        for cpm in $CONC_PER_MACHINE; do arm_concs="$arm_concs $(( machines * cpm ))"; done
    fi
    say "  conc    : ${arm_concs# }  ($(if [[ -n "$CONCURRENCIES" ]]; then echo "absolute"; else echo "${CONC_PER_MACHINE} x ${machines} machines"; fi))"

    for c in $arm_concs; do
        n=$(( c * WL_PPC ))
        for r in $(seq "$(( WARMUP == 1 ? 0 : 1 ))" "$REPEATS"); do
            tag="${arm}-m${machines}-${PROFILE}-${a2atag}${MFSTAMP}-isl${WL_ISL}-osl${WL_OSL}-c${c}-n${n}-r${r}"
            if (( r == 0 )); then say "----- warmup (DISCARDED) c=$c  tag=$tag -----"
            else say "----- run $r/$REPEATS c=$c  tag=$tag -----"; fi
            sshx "$BENCH_HOST" "cd $REMOTE && WL=$WL ISL=$WL_ISL OSL=$WL_OSL \
                NUM_PROMPTS=$n CONCURRENCY=$c MODE=$bench_mode PROFILE=$PROFILE \
                MACHINES=$machines TAG=$tag ENDPOINT=$endpoint bash 91_bench.sh" 2>&1 \
                | grep -E "^DRY |Successful requests|Benchmark duration|Request throughput|Input token throughput|Output token throughput|Total token throughput|Mean TTFT|Median TTFT|P99 TTFT|Mean TPOT|Median TPOT|Mean ITL|Median ITL|Concurrency|accept length|Error|error" \
                | tee -a "$SUMMARY"

            # Snapshot every node's server log per run. The client's percentiles
            # cannot tell "generating slower" from "queued behind a preemption",
            # and dispatch/combine time is node-layered with the slow node
            # flipping between runs, so one node's log is not the run.
            for i in "${!I_ROLE[@]}"; do
                case "${I_ROLE[$i]}" in
                    S) cname=kimi-k3 ;; P) cname=kimi-k3-prefill ;; D) cname=kimi-k3-decode ;;
                esac
                for h in ${I_HOSTS[$i]}; do
                    snapshot_log "$h" "$cname" "$tag" "$REMOTE"
                done
            done
        done
    done

    # PULL NOW, NOT AT THE END OF THE CAMPAIGN.
    # 91_bench.sh writes each bench JSON on the HOST. On 2026-09-05 all four
    # B300 were terminated ("User initiated") ~20 min into a multi-hour
    # follow-up campaign, and pd1p1d:tp's nine completed runs went with them --
    # only the laptop-side driver log survived, and salvage_log_json.py had to
    # rebuild the JSONs from its printed blocks. An arm is ~20 min and the pull
    # is a few KB, so there is no reason to hold results on hardware that can
    # disappear. Scoped to this arm's hosts; failures are reported, never fatal.
    if [[ "${DRY_RUN:-0}" != "1" && "${PULL_PER_ARM:-1}" == "1" ]]; then
        say "  pull    : results from$ARM_HOSTS -> $LOCAL_RESULTS"
        HOSTS="${ARM_HOSTS# }" bash sync.sh pull 2>&1 | sed 's/^/    /' | tee -a "$SUMMARY"
    fi
done

if [[ "$TEARDOWN_AT_END" == "1" ]]; then
    say ""
    say "tearing down every container on $ALL_HOSTS"
    teardown_all $ALL_HOSTS
else
    say ""
    say "NOTE: the last arm's containers are still running (TEARDOWN_AT_END=1 to free them)."
fi

say ""
say "finished: $(date -u +%FT%TZ)"
say "raw logs live on the hosts under $REMOTE/results/ -- 'bash sync.sh pull' to fetch,"
say "then 'python3 gen_matched_table.py results' for the per-machine comparison."
echo "summary: $SUMMARY"
