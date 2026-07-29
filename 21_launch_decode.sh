#!/bin/bash
# HOST-side launcher: DECODE container. Run on P6-B300-2.
#
#   bash 21_launch_decode.sh
#   NO_SPEC=1 bash 21_launch_decode.sh
set -euo pipefail

cd "$(dirname "$0")"
source ./env_common.sh

NAME="${NAME:-kimi-k3-decode}"
# symm_allocator: --enable-symm-mem JIT-compiles torch's NCCL allocator into
# /tmp/symm_allocator, which is container-local and so rebuilt on every launch
# (~1-2 min). Persisting it on the host makes restarts reuse the .so.
mkdir -p "$HOST_CACHE_DIR/deep_gemm" "$HOST_CACHE_DIR/torch" \
         "$HOST_CACHE_DIR/flashinfer" "$HOST_CACHE_DIR/symm_allocator"

# Unmapping ~1.5 TB of model volumes can outlast a single `rm -f`, leaving the
# container Exited-but-present and the next `docker run` failing with "container
# name already in use". Retry the removal (not just an inspect — inspect
# succeeds on an Exited container) until the name is actually free.
for _ in $(seq 1 60); do
    docker inspect "$NAME" >/dev/null 2>&1 || break
    docker rm -f "$NAME" >/dev/null 2>&1 || true
    sleep 2
done
if docker inspect "$NAME" >/dev/null 2>&1; then
    echo "ERROR: could not remove existing container '$NAME'" >&2
    exit 1
fi

require_efa_image "$IMAGE"

docker run -d --name "$NAME" \
    --gpus all \
    --net=host --ipc=host \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    --device=/dev/infiniband --privileged \
    --shm-size=600g \
    -v "$HOST_MODEL_DIR/Kimi-K3:/models/Kimi-K3:ro" \
    -v "$HOST_MODEL_DIR/Kimi-K3-DSpark:/models/Kimi-K3-DSpark:ro" \
    -v "$HOST_CACHE_DIR/deep_gemm:/root/.cache/deep_gemm" \
    -v "$HOST_CACHE_DIR/torch:/root/.cache/torch" \
    -v "$HOST_CACHE_DIR/flashinfer:/root/.cache/flashinfer" \
    -v "$HOST_CACHE_DIR/symm_allocator:/tmp/symm_allocator" \
    -v "$SCRIPT_DIR_HOST:/host/kimi-k3-aws:ro" \
    -e NO_SPEC="${NO_SPEC:-0}" \
    -e MEM_FRACTION="${MEM_FRACTION:-$DECODE_MEM_FRACTION}" \
    -e MAMBA_RATIO="${MAMBA_RATIO:-$DECODE_MAMBA_RATIO}" \
    -e DCP_SIZE="${DCP_SIZE:-$DECODE_DCP_SIZE}" \
    -e CUSTOM_AR="${CUSTOM_AR:-$DECODE_CUSTOM_AR}" \
    -e SYMM_MEM="${SYMM_MEM:-$DECODE_SYMM_MEM}" \
    -e DECODE_EXTRA_SLOTS="${DECODE_EXTRA_SLOTS:-16}" \
    -e TRANSFER_BACKEND="${TRANSFER_BACKEND:-mooncake}" \
    -e NCCL_DEBUG="${NCCL_DEBUG:-WARN}" \
    -e TP_SIZE="$TP_SIZE" -e PORT="$PORT" \
    --entrypoint bash \
    "$IMAGE" \
    /host/kimi-k3-aws/start_decode.sh

echo "launched '$NAME' (profile=$PROFILE, mem=${MEM_FRACTION:-$DECODE_MEM_FRACTION}, dcp=${DCP_SIZE:-$DECODE_DCP_SIZE}, mamba=${MAMBA_RATIO:-$DECODE_MAMBA_RATIO}, symm=${SYMM_MEM:-$DECODE_SYMM_MEM})  ->  docker logs -f $NAME"
