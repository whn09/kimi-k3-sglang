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
build_cache_args "/tmp/symm_allocator=symm_allocator"
build_gdr_args
# Decode's OWN capacity -- smaller than prefill's, because this side spends the
# memory on CUDA graph capture instead. See the CAP block in env_common.sh.
build_deepep_envs "${CAP:-$DECODE_CAP}"

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
    --device=/dev/infiniband ${GDR_ARGS[@]+"${GDR_ARGS[@]}"} --privileged \
    --shm-size=600g \
    -v "$HOST_MODEL_DIR/Kimi-K3:/models/Kimi-K3:ro" \
    -v "$HOST_MODEL_DIR/Kimi-K3-DSpark:/models/Kimi-K3-DSpark:ro" \
    "${CACHE_ARGS[@]}" \
    -v "$SCRIPT_DIR_HOST:/host/kimi-k3-sglang:ro" \
    -e NO_SPEC="${NO_SPEC:-0}" \
    -e MEM_FRACTION="${MEM_FRACTION:-$DECODE_MEM_FRACTION}" \
    -e MAMBA_RATIO="${MAMBA_RATIO:-$DECODE_MAMBA_RATIO}" \
    -e DCP_SIZE="${DCP_SIZE:-$DECODE_DCP_SIZE}" \
    -e CUSTOM_AR="${CUSTOM_AR:-$DECODE_CUSTOM_AR}" \
    -e SYMM_MEM="${SYMM_MEM:-$DECODE_SYMM_MEM}" \
    -e DECODE_EXTRA_SLOTS="${DECODE_EXTRA_SLOTS:-16}" \
    -e TRANSFER_BACKEND="${TRANSFER_BACKEND:-mooncake}" \
    ${PYTORCH_CUDA_ALLOC_CONF+-e PYTORCH_CUDA_ALLOC_CONF} \
    -e NCCL_DEBUG="${NCCL_DEBUG:-WARN}" \
    -e MOE_A2A_BACKEND="$MOE_A2A_BACKEND" \
    -e EP_SIZE="$EP_SIZE" \
    -e DEEPEP_V2_MODE="$DEEPEP_V2_MODE" \
    -e MOE_RUNNER_BACKEND="$MOE_RUNNER_BACKEND" \
    -e CHUNKED_PREFILL="${CHUNK:-$DECODE_CHUNK}" \
    -e CGMAXBS="${CGMAXBS:-$DECODE_CGMAXBS}" \
    -e MAXRUN="${MAXRUN:-$DECODE_MAXRUN}" \
    -e SPEC_BLOCK_SIZE="${SPEC_BLOCK_SIZE:-7}" \
    ${DEEPEP_ENVS[@]+"${DEEPEP_ENVS[@]}"} \
    -e TP_SIZE="$TP_SIZE" -e PORT="$PORT" \
    --entrypoint bash \
    "$IMAGE" \
    /host/kimi-k3-sglang/start_decode.sh

echo "launched '$NAME' (profile=$PROFILE, mem=${MEM_FRACTION:-$DECODE_MEM_FRACTION}, dcp=${DCP_SIZE:-$DECODE_DCP_SIZE}, mamba=${MAMBA_RATIO:-$DECODE_MAMBA_RATIO}, symm=${SYMM_MEM:-$DECODE_SYMM_MEM})  ->  docker logs -f $NAME"
# CAP is an env var, not a flag, so it is invisible in `docker inspect .Args`.
echo "  a2a=$MOE_A2A_BACKEND ep=$EP_SIZE cap=${CAP:-$DECODE_CAP} chunk=${CHUNK:-$DECODE_CHUNK} decode-graphs=ON max-bs=${CGMAXBS:-${DECODE_CGMAXBS:-<sglang default>}} max-run=${MAXRUN:-${DECODE_MAXRUN:-<DSPARK 48>}}"
