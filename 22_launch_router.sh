#!/bin/bash
# HOST-side launcher: sglang-router fronting the prefill+decode pair.
# Run on P6-B300-1 (or anywhere that can reach both nodes).
#
#   bash 22_launch_router.sh
#
# Clients then talk to http://<this-host>:8080/v1/... only.
set -euo pipefail

cd "$(dirname "$0")"
source ./env_common.sh

NAME="${NAME:-kimi-k3-router}"

docker rm -f "$NAME" 2>/dev/null || true

echo "router  : 0.0.0.0:${ROUTER_PORT}"
echo "prefill : http://${PREFILL_IP}:${PORT} (bootstrap ${BOOTSTRAP_PORT})"
echo "decode  : http://${DECODE_IP}:${PORT}"

# The model must be mounted: the router asks each worker for its model path,
# gets back /models/Kimi-K3, and tries to load a tokenizer from it. Without the
# mount it treats that path as an HF repo id and fails registration with
# "404 Not Found for https://huggingface.co/api/models//models/Kimi-K3".
docker run -d --name "$NAME" \
    --net=host \
    -v "$HOST_MODEL_DIR/Kimi-K3:/models/Kimi-K3:ro" \
    -v "$SCRIPT_DIR_HOST:/host/kimi-k3-aws:ro" \
    --entrypoint python3 \
    "$IMAGE" \
    -m sglang_router.launch_router \
    --pd-disaggregation \
    --prefill "http://${PREFILL_IP}:${PORT}" "${BOOTSTRAP_PORT}" \
    --decode "http://${DECODE_IP}:${PORT}" \
    --host 0.0.0.0 \
    --port "${ROUTER_PORT}"

echo "launched '$NAME'  ->  docker logs -f $NAME"
