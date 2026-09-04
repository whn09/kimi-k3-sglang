#!/bin/bash
# HOST-side launcher: a PLAIN (non-PD) sglang-router in front of N INDEPENDENT
# single-node servers. This is the BASELINE, not a deployment we are proposing:
# the claim under test is "PD disaggregation beats N standalone servers at the
# same machine count", and this arm is the N standalone servers.
#
#   AGG_IPS="$B300_1_IP $B300_2_IP" bash 23_launch_agg_router.sh          # 2 machines
#   AGG_IPS="$B300_1_IP $B300_2_IP $B300_3_IP $B300_4_IP" \
#       ROUTER_POLICY=round_robin bash 23_launch_agg_router.sh            # 4 machines
#
# Run 10_launch_standalone.sh on every one of those hosts first.
#
# WHY THE BASELINE GETS A ROUTER AT ALL. It would be easier to fire N bench
# clients at N servers and add up the throughputs, and it would also be wrong
# twice over: the runs have different durations so the per-second numbers do not
# add, and the PD rows would carry a router hop (plus its tokenizer and its
# queueing) that the baseline rows do not. Same client, same concurrency, same
# number of hops -- then the only difference left is the topology.
#
# One entry per INSTANCE, and for a multi-node instance it is the rank-0 address:
# ranks other than 0 never bind :$PORT, so listing them would fail the probe
# below.
set -euo pipefail

cd "$(dirname "$0")"
source ./env_common.sh

NAME="${NAME:-kimi-k3-agg-router}"

docker rm -f "$NAME" 2>/dev/null || true

# round_robin, not the router's cache_aware default. With --random-range-ratio
# 1.0 every request is the same size and the prompts are unique random tokens, so
# a prefix-affinity policy has nothing to exploit and can only imbalance the
# split -- while round_robin gives each of N identical machines exactly 1/N of
# the traffic, which is what "N independent servers" means. power_of_two
# (least-load) is the variant worth one run if a queueing artefact is ever
# suspected; it can only make this baseline stronger.
POLICY="${ROUTER_POLICY:-round_robin}"

WORKER_URLS=()
for ip in $AGG_IPS; do
    WORKER_URLS+=("http://${ip}:${PORT}")
done
nw=${#WORKER_URLS[@]}
if (( nw == 0 )); then
    echo "ERROR: AGG_IPS is empty -- nothing to route to." >&2
    exit 1
fi

echo "agg router: 0.0.0.0:${ROUTER_PORT}   (${nw} independent workers, policy=${POLICY})"
for u in "${WORKER_URLS[@]}"; do echo "worker    : $u"; done

# A worker that is not up yet fails registration and the router exits, so probe
# every HTTP port first -- otherwise the only symptom is a router that vanished.
for ip in $AGG_IPS; do
    (exec 3<>"/dev/tcp/${ip}/${PORT}") 2>/dev/null \
      || { echo "FATAL: ${ip}:${PORT} is not accepting connections."; \
           echo "       Start 10_launch_standalone.sh on every worker host and wait for"; \
           echo "       'server is fired up' (a cold K3 needs ~6 min)."; \
           exit 1; }
done

# The model mount is not optional: the router asks each worker for its model path,
# gets back /models/Kimi-K3, and tries to load a tokenizer from it. Without the
# mount it treats that path as an HF repo id and fails registration with
# "404 Not Found for https://huggingface.co/api/models//models/Kimi-K3".
#
# --worker-urls is nargs='*', so it goes LAST: any flag after it would be
# swallowed as another URL.
docker run -d --name "$NAME" \
    --net=host \
    -v "$HOST_MODEL_DIR/Kimi-K3:/models/Kimi-K3:ro" \
    -v "$SCRIPT_DIR_HOST:/host/kimi-k3-sglang:ro" \
    --entrypoint python3 \
    "$IMAGE" \
    -m sglang_router.launch_router \
    --host 0.0.0.0 \
    --port "${ROUTER_PORT}" \
    --policy "${POLICY}" \
    --worker-urls "${WORKER_URLS[@]}"

echo "launched '$NAME'  ->  docker logs -f $NAME"
echo "health: curl -s localhost:${ROUTER_PORT}/health"
