#!/bin/bash
# HOST-side launcher: start the single-node Kimi-K3 container on one B300.
#
#   bash 10_launch_standalone.sh            # with DSPARK spec decoding
#   NO_SPEC=1 bash 10_launch_standalone.sh  # base model only (first bring-up)
#
# Follow with: docker logs -f kimi-k3
set -euo pipefail

cd "$(dirname "$0")"
source ./env_common.sh

NAME="${NAME:-kimi-k3}"
build_cache_args "/tmp/symm_allocator=symm_allocator"

docker rm -f "$NAME" 2>/dev/null || true

# --net=host: the 18 EFA rails and the ENA interface must be visible as-is for
#   NCCL/Mooncake device discovery; a bridge network breaks rail selection.
# --device=/dev/infiniband + memlock=-1: required for EFA RDMA registration.
# --shm-size=600g: TP=8 loading 1.5 TB of shards moves a lot through /dev/shm.
docker run -d --name "$NAME" \
    --gpus all \
    --net=host --ipc=host \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    --device=/dev/infiniband \
    --shm-size=600g \
    -v "$HOST_MODEL_DIR/Kimi-K3:/models/Kimi-K3:ro" \
    -v "$HOST_MODEL_DIR/Kimi-K3-DSpark:/models/Kimi-K3-DSpark:ro" \
    "${CACHE_ARGS[@]}" \
    -v "$SCRIPT_DIR_HOST:/host/kimi-k3-sglang:ro" \
    -e NO_SPEC="${NO_SPEC:-0}" \
    -e MEM_FRACTION="${MEM_FRACTION:-$STANDALONE_MEM_FRACTION}" \
    -e MAMBA_RATIO="${MAMBA_RATIO:-$STANDALONE_MAMBA_RATIO}" \
    -e DCP_SIZE="$STANDALONE_DCP_SIZE" \
    -e CUSTOM_AR="$STANDALONE_CUSTOM_AR" \
    -e NCCL_DEBUG="${NCCL_DEBUG:-WARN}" \
    -e TP_SIZE="$TP_SIZE" \
    -e PORT="$PORT" \
    --entrypoint bash \
    "$IMAGE" \
    /host/kimi-k3-sglang/start_standalone.sh

echo "launched '$NAME' (profile=$PROFILE, dcp=$STANDALONE_DCP_SIZE, mamba=$STANDALONE_MAMBA_RATIO, mem=$STANDALONE_MEM_FRACTION)  ->  docker logs -f $NAME"
echo "health: curl -s localhost:${PORT}/health_generate"
