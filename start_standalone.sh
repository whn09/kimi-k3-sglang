#!/bin/bash
# In-container: single-node Kimi-K3 on 8x B300 (TP=8).
#
# Mirrors the upstream lmsys reference command, adapted for local weights on
# NVMe instead of an HF-hub pull. Launched by 10_launch_standalone.sh.
#
# Set NO_SPEC=1 to drop DSPARK speculative decoding — do this for the first
# bring-up so a draft-model problem cannot be confused with a base-model one.
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

# Radix cache off makes seeded/random benchmark runs repeatable (no cross-run
# prefix reuse). Leave it ON for real serving.
DCP_ARGS=()
[[ "${DCP_SIZE:-1}" -gt 1 ]] && DCP_ARGS=(--dcp-size "$DCP_SIZE")

AR_ARGS=()
[[ "${CUSTOM_AR:-on}" == "off" ]] && AR_ARGS=(--disable-custom-all-reduce)

RADIX_ARGS=()
[[ "${DISABLE_RADIX:-0}" == "1" ]] && RADIX_ARGS=(--disable-radix-cache)

echo "=== Kimi-K3 standalone: TP=${TP_SIZE}, spec=$([[ ${NO_SPEC:-0} == 1 ]] && echo off || echo DSPARK) ==="

exec python3 -m sglang.launch_server \
    --model-path "$MODEL_PATH" \
    --served-model-name "$SERVED_MODEL_NAME" \
    --trust-remote-code \
    --tp-size "$TP_SIZE" \
    "${DCP_ARGS[@]}" \
    --mem-fraction-static "${MEM_FRACTION:-0.85}" \
    --mamba-full-memory-ratio "${MAMBA_RATIO:-0.86}" \
    --reasoning-parser kimi_k3 \
    --tool-call-parser kimi_k3 \
    "${SPEC_ARGS[@]}" \
    "${RADIX_ARGS[@]}" \
    "${AR_ARGS[@]}" \
    --host 0.0.0.0 \
    --port "$PORT" \
    --decode-log-interval 1 \
    --watchdog-timeout 1000000 \
    --dist-timeout 7200
