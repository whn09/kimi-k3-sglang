#!/bin/bash
# In-container: single-node Kimi-K3 on 8x B300 (TP=8).
#
# Mirrors the reference command from the SGLang cookbook
# (https://docs.sglang.io/cookbook/autoregressive/Moonshotai/Kimi-K3), adapted
# for local weights on NVMe instead of an HF-hub pull. Launched by
# 10_launch_standalone.sh.
#
# Set NO_SPEC=1 to drop DSPARK speculative decoding — do this for the first
# bring-up so a draft-model problem cannot be confused with a base-model one.
set -euo pipefail

source /host/kimi-k3-sglang/env_common.sh
setup_runtime_env

SPEC_ARGS=()
if [[ "${NO_SPEC:-0}" != "1" ]]; then
    SPEC_ARGS=(
        --speculative-algorithm DSPARK
        --speculative-draft-model-path "$DRAFT_MODEL_PATH"
        --speculative-dspark-block-size 7
        --enable-linear-replayssm-spec
    )
fi

# Radix cache off makes seeded/random benchmark runs repeatable (no cross-run
# prefix reuse). Leave it ON for real serving.
DCP_ARGS=()
[[ "${DCP_SIZE:-1}" -gt 1 ]] && DCP_ARGS=(--dcp-size "$DCP_SIZE")

AR_ARGS=()
[[ "${CUSTOM_AR:-on}" == "off" ]] && AR_ARGS=(--disable-custom-all-reduce)

RADIX_ARGS=()
[[ "${DISABLE_RADIX:-0}" == "1" ]] && RADIX_ARGS=(--disable-radix-cache)

# ---- DeepEP v2 (intra-node EP) ----
# The unified server is the arm where ONE capacity has to serve both the prefill
# chunk and the decode graph workspace, so it is stuck at CAP=1024 with graphs on.
# It is kept as the baseline that the PD split is measured against, not because
# it is a good config.
EP_ARGS=()
if [[ "${MOE_A2A_BACKEND:-none}" != "none" ]]; then
    EP_ARGS=(
        --ep-size "${EP_SIZE:-$TP_SIZE}"
        --moe-a2a-backend "$MOE_A2A_BACKEND"
        --deepep-v2-mode "${DEEPEP_V2_MODE:-direct}"
        --moe-runner-backend "${MOE_RUNNER_BACKEND:-deep_gemm}"
    )
    cap="${SGLANG_DEEPEP_V2_NUM_MAX_DISPATCH_TOKENS_PER_RANK:-128}"
    chunk="${CHUNKED_PREFILL:-1024}"
    if (( chunk > cap )); then
        echo "ERROR: DeepEP v2 budget: chunked-prefill-size ($chunk) > CAP ($cap)." >&2
        exit 1
    fi
fi

CG_ARGS=()
[[ -n "${CGMAXBS:-}" ]] && CG_ARGS=(--cuda-graph-max-bs-decode "$CGMAXBS")

echo "=== Kimi-K3 standalone: TP=${TP_SIZE}, spec=$([[ ${NO_SPEC:-0} == 1 ]] && echo off || echo DSPARK)," \
     "a2a=${MOE_A2A_BACKEND:-none} ep=${EP_SIZE:-1}" \
     "cap=${SGLANG_DEEPEP_V2_NUM_MAX_DISPATCH_TOKENS_PER_RANK:-n/a}" \
     "chunk=${CHUNKED_PREFILL:-1024} gin=${NCCL_GIN_TYPE:-unset} ==="

exec python3 -m sglang.launch_server \
    --model-path "$MODEL_PATH" \
    --served-model-name "$SERVED_MODEL_NAME" \
    --trust-remote-code \
    --tp-size "$TP_SIZE" \
    "${DCP_ARGS[@]}" \
    --mem-fraction-static "${MEM_FRACTION:-0.85}" \
    --mamba-full-memory-ratio "${MAMBA_RATIO:-0.86}" \
    --chunked-prefill-size "${CHUNKED_PREFILL:-1024}" \
    --max-prefill-tokens "${MAX_PREFILL_TOKENS:-${CHUNKED_PREFILL:-1024}}" \
    --reasoning-parser kimi_k3 \
    --tool-call-parser kimi_k3 \
    "${SPEC_ARGS[@]}" \
    "${RADIX_ARGS[@]}" \
    "${AR_ARGS[@]}" \
    "${EP_ARGS[@]}" \
    "${CG_ARGS[@]}" \
    --host 0.0.0.0 \
    --port "$PORT" \
    --decode-log-interval 1 \
    --watchdog-timeout 1000000 \
    --dist-timeout 7200
