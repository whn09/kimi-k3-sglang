#!/bin/bash
# In-container: PD-disagg PREFILL node (B300-1, TP=8), Mooncake/EFA KV transfer.
#
# Follows the 1P1D reference command from the SGLang cookbook
# (https://docs.sglang.io/cookbook/autoregressive/Moonshotai/Kimi-K3), with the
# transfer backend switched from nixl to mooncake. Note the asymmetry with the
# decode side, which is intentional and taken from that reference:
#   - prefill gets --enable-symm-mem, chunked/max prefill tokens
#   - decode gets --dcp-size 8, extra slots, a >1 mamba ratio, and
#     --enable-linear-replayssm-spec
set -euo pipefail

source /host/kimi-k3-sglang/env_common.sh
setup_runtime_env

SPEC_ARGS=()
if [[ "${NO_SPEC:-0}" != "1" ]]; then
    SPEC_ARGS=(
        --speculative-algorithm DSPARK
        --speculative-draft-model-path "$DRAFT_MODEL_PATH"
        --speculative-dspark-block-size 7
    )
fi

# Radix cache off makes seeded/random benchmark runs repeatable, but
# 91_bench.sh's --flush-cache already handles that, so this stays off by
# default and the server runs a production config.
RADIX_ARGS=()
[[ "${DISABLE_RADIX:-0}" == "1" ]] && RADIX_ARGS=(--disable-radix-cache)

# Off by default: the upstream reference gives dcp to decode only. But with
# prefill dcp=1 and decode dcp=8, prefill's bootstrap_thread dies with
#   PD DCP source/destination KV geometry differs: src=[1152 x24, 256 x10],
#   dst=[1152 x34]
# because prepare_dcp_token_item_lens() (srt/disaggregation/common/conn.py)
# builds the destination geometry as one item length replicated across all
# layers, which never matches K3's hybrid stack (24 full-MLA layers at 1152 B,
# KDA layers at 256 B). That kills the balanced and high-throughput PD profiles,
# both of which set decode dcp=8. requires_dcp_relayout() returns False when
# dcp_size == dst_dcp_size, so matching prefill's dcp to decode's should bypass
# the broken path -- set DCP_SIZE=8 here to try it.
DCP_ARGS=()
[[ -n "${DCP_SIZE:-}" && "${DCP_SIZE}" != "1" ]] && DCP_ARGS=(--dcp-size "$DCP_SIZE")

echo "=== Kimi-K3 PREFILL: TP=${TP_SIZE} bootstrap=${BOOTSTRAP_PORT} dcp=${DCP_SIZE:-1} ==="

exec python3 -m sglang.launch_server \
    --model-path "$MODEL_PATH" \
    --served-model-name "$SERVED_MODEL_NAME" \
    --trust-remote-code \
    --tp-size "$TP_SIZE" \
    --disable-custom-all-reduce \
    --enable-symm-mem \
    --mem-fraction-static "${MEM_FRACTION:-0.85}" \
    --chunked-prefill-size "${CHUNKED_PREFILL:-16384}" \
    --max-prefill-tokens "${MAX_PREFILL_TOKENS:-16384}" \
    --reasoning-parser kimi_k3 \
    --tool-call-parser kimi_k3 \
    --disaggregation-mode prefill \
    --disaggregation-transfer-backend "${TRANSFER_BACKEND:-mooncake}" \
    --disaggregation-bootstrap-port "$BOOTSTRAP_PORT" \
    --mamba-full-memory-ratio "${MAMBA_RATIO:-0.86}" \
    "${SPEC_ARGS[@]}" \
    "${RADIX_ARGS[@]}" \
    "${DCP_ARGS[@]}" \
    --host 0.0.0.0 \
    --port "$PORT" \
    --decode-log-interval 1 \
    --watchdog-timeout 1000000 \
    --dist-timeout 7200
