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
build_gdr_args
build_deepep_envs "${CAP:-$STANDALONE_CAP}"

docker rm -f "$NAME" 2>/dev/null || true

# Standalone did not used to check the image, but with DeepEP v2 it must: an
# image missing the kimi_k3.py patch serves WRONG NUMERICS rather than failing.
require_efa_image "$IMAGE"

# --net=host: the 18 EFA rails and the ENA interface must be visible as-is for
#   NCCL/Mooncake device discovery; a bridge network breaks rail selection.
# --device=/dev/infiniband + memlock=-1: required for EFA RDMA registration, and
#   also for DeepEP v2 -- sgl-deep-ep aborts on a NONE GIN type, and without this
#   device NCCL sees no network at all.
# --device=/dev/gdrdrv (via GDR_ARGS): GDRCopy, otherwise NCCL falls back.
# --privileged: matches what the two PD launchers and every measured deepep_v2
#   run used. Standalone was the odd one out, which made it the only arm where a
#   GIN init failure could be a permissions artefact rather than the real thing.
# --shm-size=600g: TP=8 loading 1.5 TB of shards moves a lot through /dev/shm.
# --init: TP=8 leaves 8 unreaped children, and without an init as PID 1 a
#   `docker rm -f` fails with "PID ... is zombie and can not be killed". That is
#   not just an untidy teardown -- the `docker rm -f` at the top of this script
#   would then fail under `set -e` and abort the NEXT launch. Hit 2026-09-04.
docker run -d --name "$NAME" \
    --init \
    --gpus all \
    --net=host --ipc=host \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    --device=/dev/infiniband \
    ${GDR_ARGS[@]+"${GDR_ARGS[@]}"} \
    --privileged \
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
    -e MOE_A2A_BACKEND="$MOE_A2A_BACKEND" \
    -e EP_SIZE="$EP_SIZE" \
    -e DEEPEP_V2_MODE="$DEEPEP_V2_MODE" \
    -e MOE_RUNNER_BACKEND="$MOE_RUNNER_BACKEND" \
    -e CHUNKED_PREFILL="${CHUNK:-$STANDALONE_CHUNK}" \
    -e CGMAXBS="${CGMAXBS:-}" \
    ${DEEPEP_ENVS[@]+"${DEEPEP_ENVS[@]}"} \
    -e TP_SIZE="$TP_SIZE" \
    -e PORT="$PORT" \
    --entrypoint bash \
    "$IMAGE" \
    /host/kimi-k3-sglang/start_standalone.sh

echo "launched '$NAME' (profile=$PROFILE, dcp=$STANDALONE_DCP_SIZE, mamba=$STANDALONE_MAMBA_RATIO, mem=$STANDALONE_MEM_FRACTION)  ->  docker logs -f $NAME"
# CAP is an env var, not a flag, so it is invisible in `docker inspect .Args`.
echo "  a2a=$MOE_A2A_BACKEND ep=$EP_SIZE cap=${CAP:-$STANDALONE_CAP} chunk=${CHUNK:-$STANDALONE_CHUNK}"
echo "health: curl -s localhost:${PORT}/health_generate"
