#!/bin/bash
# HOST-side launcher: PREFILL container. Run on B300-1.
#
#   bash 20_launch_prefill.sh
#   NO_SPEC=1 bash 20_launch_prefill.sh
#
# Cross-node EP: one prefill instance spanning two hosts (TP=16, ep=16, hybrid).
# Run this on BOTH hosts, same command except NODE_RANK, and DIST_INIT_ADDR is
# rank 0's IP on both. Only rank 0 serves HTTP, so that is the address the router
# gets as --prefill.
#   B300-1: NNODES=2 NODE_RANK=0 TP_SIZE=16 DIST_INIT_ADDR=$B300_1_IP bash 20_launch_prefill.sh
#   B300-2: NNODES=2 NODE_RANK=1 TP_SIZE=16 DIST_INIT_ADDR=$B300_1_IP bash 20_launch_prefill.sh
set -euo pipefail

cd "$(dirname "$0")"
# K3_ROLE keeps env_common.sh's mem-fraction resolution lines scoped to THIS
# role; without it a decode container's boot log carries prefill's clamp.
K3_ROLE=prefill
source ./env_common.sh

NAME="${NAME:-kimi-k3-prefill}"
build_cache_args
build_gdr_args
# Prefill's OWN capacity -- larger than decode's, which is the whole reason to
# run PD with DeepEP v2. See the CAP block in env_common.sh.
build_deepep_envs "${CAP:-$PREFILL_CAP}"

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

# --init: reaps the TP workers. Without it `docker rm -f` fails on a zombie
#   PID 1, which makes the `docker rm -f` above abort the NEXT launch.
docker run -d --name "$NAME" \
    --init \
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
    -e MEM_FRACTION="${MEM_FRACTION:-$PREFILL_MEM_FRACTION}" \
    -e MAMBA_RATIO="${MAMBA_RATIO:-$PREFILL_MAMBA_RATIO}" \
    -e DCP_SIZE="${DCP_SIZE:-$PREFILL_DCP_SIZE}" \
    -e TRANSFER_BACKEND="${TRANSFER_BACKEND:-mooncake}" \
    ${PYTORCH_CUDA_ALLOC_CONF+-e PYTORCH_CUDA_ALLOC_CONF} \
    -e NCCL_DEBUG="${NCCL_DEBUG:-WARN}" \
    -e DISABLE_RADIX="${DISABLE_RADIX:-0}" \
    -e MOE_A2A_BACKEND="$MOE_A2A_BACKEND" \
    -e EP_SIZE="$EP_SIZE" \
    -e DEEPEP_V2_MODE="$DEEPEP_V2_MODE" \
    -e MOE_RUNNER_BACKEND="$MOE_RUNNER_BACKEND" \
    -e CHUNKED_PREFILL="${CHUNK:-$PREFILL_CHUNK}" \
    -e PREFILL_CUDA_GRAPH="${PREFILL_CUDA_GRAPH:-0}" \
    ${DEEPEP_ENVS[@]+"${DEEPEP_ENVS[@]}"} \
    -e TP_SIZE="$TP_SIZE" -e PORT="$PORT" -e BOOTSTRAP_PORT="$BOOTSTRAP_PORT" \
    -e NNODES="$NNODES" -e NODE_RANK="$NODE_RANK" \
    -e DIST_INIT_ADDR="$DIST_INIT_ADDR" -e DIST_INIT_PORT="$DIST_INIT_PORT" \
    --entrypoint bash \
    "$IMAGE" \
    /host/kimi-k3-sglang/start_prefill.sh

echo "launched '$NAME' (profile=$PROFILE, mem=${MEM_FRACTION:-$PREFILL_MEM_FRACTION}, dcp=${DCP_SIZE:-$PREFILL_DCP_SIZE}, mamba=${MAMBA_RATIO:-$PREFILL_MAMBA_RATIO}, backend=${TRANSFER_BACKEND:-mooncake})  ->  docker logs -f $NAME"
# CAP is an env var, not a flag, so it is invisible in `docker inspect .Args`.
# Print it here or a run's most important axis goes unrecorded.
echo "  a2a=$MOE_A2A_BACKEND ep=$EP_SIZE mode=$DEEPEP_V2_MODE cap=${CAP:-$PREFILL_CAP} chunk=${CHUNK:-$PREFILL_CHUNK} decode-graphs=$([ "${PREFILL_CUDA_GRAPH:-0}" = 1 ] && echo on || echo OFF)"
if (( NNODES > 1 )); then
    echo "  tp=$TP_SIZE over $NNODES nodes, this host is node-rank $NODE_RANK, rendezvous ${DIST_INIT_ADDR}:${DIST_INIT_PORT}"
    echo "  (rank != 0 never binds :$PORT -- do not wait for 'server is fired up' there)"
fi
