#!/bin/bash
# In-container: PD-disagg DECODE node (B300-2, TP=8), Mooncake/EFA KV transfer.
#
# Follows the 1P1D reference command from the SGLang cookbook
# (https://docs.sglang.io/cookbook/autoregressive/Moonshotai/Kimi-K3) with
# nixl -> mooncake. Decode-side specifics from that reference:
#   --dcp-size 8                   decode context parallel across all 8 GPUs
#   --mamba-full-memory-ratio 1.03 >1 is deliberate here (decode holds no
#                                  prefill activations, so the hybrid KDA state
#                                  cache is allowed to overcommit the ratio)
#   --disaggregation-decode-extra-slots 16
#   --enable-linear-replayssm-spec decode-only; the prefill node omits it
set -euo pipefail

source /host/kimi-k3-sglang/env_common.sh
setup_runtime_env

SPEC_ARGS=()
if [[ "${NO_SPEC:-0}" != "1" ]]; then
    SPEC_ARGS=(
        --speculative-algorithm DSPARK
        --speculative-draft-model-path "$DRAFT_MODEL_PATH"
        --speculative-dspark-block-size "${SPEC_BLOCK_SIZE:-7}"
        --enable-linear-replayssm-spec
    )
fi

AR_ARGS=()
[[ "${CUSTOM_AR:-off}" == "off" ]] && AR_ARGS=(--disable-custom-all-reduce)

# low-latency PD decode has almost no mamba cache (ratio 0.17) and relies on
# symmetric memory for the all-reduce instead.
SYMM_ARGS=()
[[ "${SYMM_MEM:-off}" == "on" ]] && SYMM_ARGS=(--enable-symm-mem)

RADIX_ARGS=()
[[ "${DISABLE_RADIX:-0}" == "1" ]] && RADIX_ARGS=(--disable-radix-cache)

# dcp-size 1 is the "no decode context parallel" case (the low-latency profile),
# where the flag is omitted rather than passed as 1.
DCP_ARGS=()
[[ "${DCP_SIZE:-8}" -gt 1 ]] && DCP_ARGS=(--dcp-size "$DCP_SIZE")

# ---- DeepEP v2 (intra-node EP) ----
EP_ARGS=()
if [[ "${MOE_A2A_BACKEND:-none}" != "none" ]]; then
    EP_ARGS=(
        --ep-size "${EP_SIZE:-$TP_SIZE}"
        --moe-a2a-backend "$MOE_A2A_BACKEND"
        --deepep-v2-mode "${DEEPEP_V2_MODE:-direct}"
        --moe-runner-backend "${MOE_RUNNER_BACKEND:-deep_gemm}"
    )
    # THE DRAFT MODEL INHERITS THIS BACKEND, AND v2 REFUSES TO BE IT.
    # moe_hook.py:validate_deepep_v2_speculative_draft() raises
    #   "DeepEP v2 MoE is not validated as a speculative draft backend"
    # when speculative_moe_a2a_backend is unset and the algorithm is not ngram:
    # it then copies moe_a2a_backend into the draft. So `--moe-a2a-backend
    # deepep_v2` + DSPARK is a hard startup failure with nothing about the draft
    # in the message -- it fails in resolve_once(), before any weight load, which
    # is at least cheap. The target model still runs v2; only the draft is pinned
    # off it. Override with SPEC_A2A_BACKEND= if a future release validates one.
    if [[ "${NO_SPEC:-0}" != "1" ]]; then
        EP_ARGS+=(--speculative-moe-a2a-backend "${SPEC_A2A_BACKEND:-none}")
    fi
fi

# THE DECODE NODE KEEPS ITS CUDA GRAPHS. Measured both ways at 8K in / 1K out,
# concurrency 16, identical 232384-token pool: graphs on gave 3.27x end-to-end
# throughput and an ITL p50 of 43.54 ms against 255.24 ms. The 255 ms was
# kernel-launch overhead, not the fixed-capacity a2a -- without graphs the step
# time is flat in batch size. Break-even against the bigger chunk graphs cost is
# ~52 output tokens, which any real answer clears.
#
# CGMAXBS does NOT save memory (the capture pool is sized by CAP, not by how many
# shapes are captured: CAP=2048 with bs=[8,16,24] still took 33.43 GiB and OOMed
# while CAP=1024 with all 13 shapes took 18.48 GB and started). It cuts the ~62 s
# capture, and it is how a small CAP becomes legal -- see the budget check below.
CG_ARGS=()
[[ -n "${CGMAXBS:-}" ]] && CG_ARGS=(--cuda-graph-max-bs-decode "$CGMAXBS")

# Bounds the decode batch. Left unset, DSPARK forces 48
# (speculative_hook.py:506-514, with a warning), and 48 is what makes a small CAP
# explode -- see the budget check below.
RUN_ARGS=()
[[ -n "${MAXRUN:-}" ]] && RUN_ARGS=(--max-running-requests "$MAXRUN")

# Fail here, not after a 10-minute weight load, and not mid-benchmark. TWO checks
# read the decode CAP and they are not the same check:
#
#   boot     moe_hook.py:400-413   graph_bs * tokens_per_req <= CAP, where
#                                  graph_bs = min(cuda_graph_max_bs_decode,
#                                  max_running_requests // attn_dp_size)
#   RUNTIME  deepep_v2.py:257      hidden_states.shape[0] > CAP -> ValueError,
#                                  evaluated on EVERY forward
#
# The runtime one is the dangerous one: it does not fall back to eager, it fails
# the request, and it fires long after startup looked clean. Passing the boot
# check is therefore NOT sufficient -- graph_bs only describes the CAPTURED
# shapes, while the runtime check sees the batch the scheduler actually built.
#
# decode.py:2609 builds that batch as min(req_to_token_pool.size,
# max_running_requests) -- and the +extra_slots+1 at pool_configurator.py:940
# sizes the POOL, not the batch -- so the sufficient rule is
#
#     max_running_requests * tokens_per_req <= CAP
#
# which also implies the boot check. SPECULATIVE DECODING IS WHAT MAKES THIS
# BITE: tokens_per_req is block_size + 1 (verified, not guessed --
# speculative_hook.py:479-494 resolves speculative_num_draft_tokens = gamma + 1,
# and overrides.py:1869 feeds exactly that to the budget check), so DSPARK
# block 7 costs 8 tokens per request per step. With spec off it is 1 and none of
# this is reachable.
#
# Worked defaults: CAP=1024 with DSPARK's 48 needs 384 <= 1024, fine, which is
# why the shipped default needs no MAXRUN at all. CAP=128 needs
# max_running_requests <= 16 and ALSO CGMAXBS >= 16 to stay captured.
if [[ "${MOE_A2A_BACKEND:-none}" != "none" ]]; then
    cap="${SGLANG_DEEPEP_V2_NUM_MAX_DISPATCH_TOKENS_PER_RANK:-128}"
    tpr=1
    [[ "${NO_SPEC:-0}" != "1" ]] && tpr=$(( ${SPEC_BLOCK_SIZE:-7} + 1 ))
    # DSPARK's own default when we do not pass one. Keep in sync with
    # speculative_hook.py:_handle_dspark.
    eff_run="${MAXRUN:-}"
    if [[ -z "$eff_run" ]]; then
        [[ "${NO_SPEC:-0}" != "1" ]] && eff_run=48 || eff_run=""
    fi
    if [[ -n "$eff_run" ]]; then
        need=$(( eff_run * tpr ))
        if (( need > cap )); then
            if [[ -n "${MAXRUN:-}" ]]; then src="from DECODE_MAXRUN"; else src="DSPARK default"; fi
            echo "ERROR: DeepEP v2 decode budget: max_running_requests ($eff_run, $src)" >&2
            echo "       x tokens_per_req ($tpr) = $need > CAP ($cap)." >&2
            echo "       deepep_v2.py:257 raises on the first forward that big." >&2
            echo "       Either raise DECODE_CAP to >= $need, or set" >&2
            echo "       DECODE_MAXRUN <= $(( cap / tpr )), or set NO_SPEC=1." >&2
            exit 1
        fi
        # Legal for the a2a but uncaptured: the quiet failure mode, ~255 ms/step.
        if [[ -n "${CGMAXBS:-}" ]] && (( CGMAXBS < eff_run )); then
            echo "WARNING: CGMAXBS ($CGMAXBS) < max_running_requests ($eff_run):" \
                 "batches above $CGMAXBS are legal but run EAGER (~255 ms/step)." >&2
            echo "         Set DECODE_CGMAXBS >= $eff_run." >&2
        fi
    else
        echo "NOTE: neither DECODE_MAXRUN nor speculative decoding is set, so the" \
             "decode batch is bounded only by the request pool; the" \
             "tokens <= CAP ($cap) budget is unchecked here." >&2
    fi
fi

echo "=== Kimi-K3 DECODE: TP=${TP_SIZE} ep=${EP_SIZE:-1} a2a=${MOE_A2A_BACKEND:-none}" \
     "cap=${SGLANG_DEEPEP_V2_NUM_MAX_DISPATCH_TOKENS_PER_RANK:-n/a}" \
     "gin=${NCCL_GIN_TYPE:-unset} graphs=ON max-bs=${CGMAXBS:-<sglang default>}" \
     "max-run=${MAXRUN:-<DSPARK 48>} dcp=${DCP_SIZE:-8} mem=${MEM_FRACTION:-0.85} ==="

exec python3 -m sglang.launch_server \
    --model-path "$MODEL_PATH" \
    --served-model-name "$SERVED_MODEL_NAME" \
    --trust-remote-code \
    --tp-size "$TP_SIZE" \
    "${DCP_ARGS[@]}" \
    --mem-fraction-static "${MEM_FRACTION:-0.85}" \
    --chunked-prefill-size "${CHUNKED_PREFILL:-1024}" \
    --disaggregation-decode-extra-slots "${DECODE_EXTRA_SLOTS:-16}" \
    --reasoning-parser kimi_k3 \
    --tool-call-parser kimi_k3 \
    --disaggregation-mode decode \
    --disaggregation-transfer-backend "${TRANSFER_BACKEND:-mooncake}" \
    --mamba-full-memory-ratio "${MAMBA_RATIO:-1.03}" \
    "${SPEC_ARGS[@]}" \
    "${RADIX_ARGS[@]}" \
    "${AR_ARGS[@]}" \
    "${SYMM_ARGS[@]}" \
    "${EP_ARGS[@]}" \
    "${CG_ARGS[@]}" \
    "${RUN_ARGS[@]}" \
    --host 0.0.0.0 \
    --port "$PORT" \
    --decode-log-interval 1 \
    --watchdog-timeout 1000000 \
    --dist-timeout 7200
