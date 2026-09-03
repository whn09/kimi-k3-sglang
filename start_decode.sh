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

# Fail here, not after a 10-minute weight load. The decode-side half of
# validate_deepep_v2_dispatch_token_budget (moe_hook.py:375-410) is
#     graph_bs * tokens_per_req <= CAP
# and SPECULATIVE DECODING IS WHAT MAKES THIS BITE. With spec off, tokens_per_req
# is 1, so even CAP=256 admits a batch of 256 and the check is invisible. With
# DSPARK block size 7 each request carries ~8 tokens per step, so the largest
# captured batch must be <= CAP/8: CAP=1024 -> 128 (fine at sglang's defaults),
# but CAP=256 -> 32, which the default captured list blows straight past.
#
# So the "smaller CAP is faster" experiment (on DSV4/B300, 2048 -> 256 was -16%
# step / +18% tok/s) is only reachable on this model by ALSO pinning CGMAXBS.
# Untested on K3; run it as DECODE_CAP=256 DECODE_CGMAXBS=32.
if [[ "${MOE_A2A_BACKEND:-none}" != "none" && -n "${CGMAXBS:-}" ]]; then
    cap="${SGLANG_DEEPEP_V2_NUM_MAX_DISPATCH_TOKENS_PER_RANK:-128}"
    # block_size + 1 is the conservative reading of DSPARK's tokens per step; if
    # sglang turns out to use block_size itself this check is one step strict,
    # which is the harmless direction.
    tpr=1
    [[ "${NO_SPEC:-0}" != "1" ]] && tpr=$(( ${SPEC_BLOCK_SIZE:-7} + 1 ))
    need=$(( CGMAXBS * tpr ))
    if (( need > cap )); then
        echo "ERROR: DeepEP v2 budget: cuda-graph-max-bs-decode ($CGMAXBS)" >&2
        echo "       x tokens_per_req ($tpr) = $need > CAP ($cap)." >&2
        echo "       Either raise DECODE_CAP to >= $need, or lower DECODE_CGMAXBS" >&2
        echo "       to <= $(( cap / tpr )), or set NO_SPEC=1." >&2
        exit 1
    fi
fi
# The check above can only run when CGMAXBS is pinned. Left unset, sglang picks
# the captured list itself and this script cannot pre-compute graph_bs, so the
# same budget is enforced only by sglang -- as an abort AFTER the weight load.
# It has fit so far (the observed default list tops out at bs=104, and
# 104 x 8 = 832 <= 1024), but that is a property of a default we do not control.
if [[ "${MOE_A2A_BACKEND:-none}" != "none" && -z "${CGMAXBS:-}" && "${NO_SPEC:-0}" != "1" ]]; then
    echo "NOTE: CGMAXBS unset with speculative decoding on, so the" \
         "graph_bs*tokens_per_req <= CAP budget is unchecked here." >&2
    echo "      If startup aborts on the DeepEP v2 token budget, pin" \
         "DECODE_CGMAXBS <= $(( ${SGLANG_DEEPEP_V2_NUM_MAX_DISPATCH_TOKENS_PER_RANK:-128} / (${SPEC_BLOCK_SIZE:-7} + 1) ))." >&2
fi

echo "=== Kimi-K3 DECODE: TP=${TP_SIZE} ep=${EP_SIZE:-1} a2a=${MOE_A2A_BACKEND:-none}" \
     "cap=${SGLANG_DEEPEP_V2_NUM_MAX_DISPATCH_TOKENS_PER_RANK:-n/a}" \
     "gin=${NCCL_GIN_TYPE:-unset} graphs=ON max-bs=${CGMAXBS:-<sglang default>}" \
     "dcp=${DCP_SIZE:-8} mem=${MEM_FRACTION:-0.85} ==="

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
    --host 0.0.0.0 \
    --port "$PORT" \
    --decode-log-interval 1 \
    --watchdog-timeout 1000000 \
    --dist-timeout 7200
