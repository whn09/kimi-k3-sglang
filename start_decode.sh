#!/bin/bash
# In-container: PD-disagg DECODE node (B300-2, TP=8), Mooncake/EFA KV transfer.
#
# Follows the upstream lmsys 1P1D reference command with nixl -> mooncake.
# Decode-side specifics from that reference:
#   --dcp-size 8                   decode context parallel across all 8 GPUs
#   --mamba-full-memory-ratio 1.03 >1 is deliberate here (decode holds no
#                                  prefill activations, so the hybrid KDA state
#                                  cache is allowed to overcommit the ratio)
#   --disaggregation-decode-extra-slots 16
#   --enable-linear-replayssm-spec decode-only; the prefill node omits it
set -euo pipefail

source /host/kimi-k3-aws/env_common.sh
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

echo "=== Kimi-K3 DECODE: TP=${TP_SIZE} dcp=${DCP_SIZE:-8} mem=${MEM_FRACTION:-0.85} ==="

exec python3 -m sglang.launch_server \
    --model-path "$MODEL_PATH" \
    --served-model-name "$SERVED_MODEL_NAME" \
    --trust-remote-code \
    --tp-size "$TP_SIZE" \
    "${DCP_ARGS[@]}" \
    --mem-fraction-static "${MEM_FRACTION:-0.85}" \
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
    --host 0.0.0.0 \
    --port "$PORT" \
    --decode-log-interval 1 \
    --watchdog-timeout 1000000 \
    --dist-timeout 7200
