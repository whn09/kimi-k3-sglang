#!/bin/bash
# HOST-side launcher: sglang-router fronting the prefill and decode nodes.
# Run on B300-1 (or anywhere that can reach every node).
#
#   bash 22_launch_router.sh                                   # 1P1D, the defaults
#   PREFILL_IPS="$B300_1_IP $B300_2_IP" \
#   DECODE_IPS="$B300_3_IP $B300_4_IP" bash 22_launch_router.sh # 2P2D
#
# The topology lives entirely here: sglang_router takes --prefill/--decode any
# number of times, and the 20_/21_ launchers do not know how many peers exist.
#
# Clients then talk to http://<this-host>:8080/v1/... only.
set -euo pipefail

cd "$(dirname "$0")"
source ./env_common.sh

NAME="${NAME:-kimi-k3-router}"

docker rm -f "$NAME" 2>/dev/null || true

# --prefill takes TWO words (url + bootstrap port) and --decode takes one, so the
# flags are built into an array rather than a string: an unquoted string would be
# re-split by IFS and an quoted one would arrive as a single argv entry.
ROUTE_ARGS=()
for ip in $PREFILL_IPS; do
    ROUTE_ARGS+=(--prefill "http://${ip}:${PORT}" "${BOOTSTRAP_PORT}")
done
for ip in $DECODE_IPS; do
    ROUTE_ARGS+=(--decode "http://${ip}:${PORT}")
done

np=$(echo $PREFILL_IPS | wc -w); nd=$(echo $DECODE_IPS | wc -w)
echo "router  : 0.0.0.0:${ROUTER_PORT}   (${np}P${nd}D)"
for ip in $PREFILL_IPS; do echo "prefill : http://${ip}:${PORT} (bootstrap ${BOOTSTRAP_PORT})"; done
for ip in $DECODE_IPS;  do echo "decode  : http://${ip}:${PORT}"; done

# A node that is not up yet fails registration and the router exits, so probe the
# HTTP ports first -- otherwise the only symptom is a router that vanished.
for ip in $PREFILL_IPS $DECODE_IPS; do
    (exec 3<>"/dev/tcp/${ip}/${PORT}") 2>/dev/null \
      || { echo "FATAL: ${ip}:${PORT} is not accepting connections."; \
           echo "       Start 20_/21_ on every node and wait for 'server is fired up'"; \
           echo "       (a cold K3 needs ~6 min: 214 GB of weights + graph capture)."; \
           exit 1; }
done

# The model must be mounted: the router asks each worker for its model path,
# gets back /models/Kimi-K3, and tries to load a tokenizer from it. Without the
# mount it treats that path as an HF repo id and fails registration with
# "404 Not Found for https://huggingface.co/api/models//models/Kimi-K3".
docker run -d --name "$NAME" \
    --net=host \
    -v "$HOST_MODEL_DIR/Kimi-K3:/models/Kimi-K3:ro" \
    -v "$SCRIPT_DIR_HOST:/host/kimi-k3-sglang:ro" \
    --entrypoint python3 \
    "$IMAGE" \
    -m sglang_router.launch_router \
    --pd-disaggregation \
    "${ROUTE_ARGS[@]}" \
    --host 0.0.0.0 \
    --port "${ROUTER_PORT}"

echo "launched '$NAME'  ->  docker logs -f $NAME"
