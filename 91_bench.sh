#!/bin/bash
# Throughput / latency benchmark via sglang.bench_serving.
#
#   bash 91_bench.sh                                  # standalone, 1k/1k, 32 concurrent
#   ENDPOINT=localhost:8080 bash 91_bench.sh          # via the PD router
#   ISL=4096 OSL=512 NUM_PROMPTS=128 CONCURRENCY=64 bash 91_bench.sh
#
# Raw output is tee'd to $RESULTS_DIR/<TAG>.log and the JSON summary written
# alongside it, so a whole sweep leaves an auditable trail.
set -uo pipefail

cd "$(dirname "$0")"
source ./env_common.sh

MODE="${MODE:-standalone}"
# Under PD the default endpoint MUST be the router, not $PORT. Benching
# localhost:30000 in PD mode hits the PREFILL worker directly, which answers a
# completion request with a body that has no "choices" key, and bench_serving
# reports that as `Warmup failed ... KeyError: 'choices'` -- an error that says
# nothing about the actual mistake. Pass ENDPOINT= explicitly to override.
if [[ "$MODE" == "pd" ]]; then
    ENDPOINT="${ENDPOINT:-localhost:$ROUTER_PORT}"
else
    ENDPOINT="${ENDPOINT:-localhost:$PORT}"
fi
HOST="${ENDPOINT%%:*}"
BPORT="${ENDPOINT##*:}"
ISL="${ISL:-1024}"
OSL="${OSL:-1024}"
NUM_PROMPTS="${NUM_PROMPTS:-64}"
CONCURRENCY="${CONCURRENCY:-32}"
NAME="${NAME:-kimi-k3-bench}"

# Raw-log capture. TAG identifies the experiment; MODE is standalone|pd.
RESULTS_DIR="${RESULTS_DIR:-$SCRIPT_DIR_HOST/results}"

# The EP axis has to be IN THE FILENAME. Without it a deepep_v2 run and a
# plain-TP run at the same profile/ISL/OSL/concurrency write the same .log and
# .json, and the second silently deletes the first -- and under PD the two sides
# have DIFFERENT caps, so "cap" is two numbers, not one. CAP comes from the
# container's env rather than from this script's variables because it is an env
# var: it never appears in the server's `server_args=` line, so nothing else in
# this harness can see it.
#
# The stamp is empty when EP is off, so existing plain-TP filenames are unchanged.
EPTAG=""
if [[ "${MOE_A2A_BACKEND:-none}" != "none" ]]; then
    if [[ "$MODE" == "pd" ]]; then
        EPTAG="-${MOE_A2A_BACKEND}-p$(read_cap kimi-k3-prefill "${CAP_PREFILL:-}")d$(read_cap kimi-k3-decode "${CAP_DECODE:-}")"
    else
        EPTAG="-${MOE_A2A_BACKEND}-cap$(read_cap kimi-k3)"
    fi
fi
# NUM_PROMPTS belongs in the name too. It was missing, and a c16 run at 32
# requests then wrote the same file as a c16 run at 64 -- two different
# denominators, one filename, second overwrites the first.
TAG="${TAG:-${MODE}-${PROFILE}${EPTAG}-isl${ISL}-osl${OSL}-c${CONCURRENCY}-n${NUM_PROMPTS}}"
mkdir -p "$RESULTS_DIR"
LOG="$RESULTS_DIR/${TAG}.log"
JSON="$RESULTS_DIR/${TAG}.json"

echo "bench: ${ENDPOINT}  isl=${ISL} osl=${OSL} n=${NUM_PROMPTS} conc=${CONCURRENCY}"
echo "log  : ${LOG}"

{
  echo "### tag=${TAG}"
  echo "### mode=${MODE} profile=${PROFILE} endpoint=${ENDPOINT}"
  echo "### isl=${ISL} osl=${OSL} num_prompts=${NUM_PROMPTS} concurrency=${CONCURRENCY}"
  # Recorded even when EP is off, so a log can never be ambiguous about which
  # MoE path produced it. "unknown" means the container was not reachable from
  # here (e.g. benching a remote endpoint) -- treat such a row as unlabelled.
  if [[ "$MODE" == "pd" ]]; then
      echo "### a2a=${MOE_A2A_BACKEND} ep=${EP_SIZE} cap_prefill=$(read_cap_src kimi-k3-prefill "${CAP_PREFILL:-}") cap_decode=$(read_cap_src kimi-k3-decode "${CAP_DECODE:-}")"
  else
      echo "### a2a=${MOE_A2A_BACKEND} ep=${EP_SIZE} cap=$(read_cap kimi-k3)"
  fi
  echo "### started=$(date -u +%FT%TZ)"
} > "$LOG"

# --flush-cache: the random dataset is seeded (seed=42), so a second run replays
# the same prompts and hits the radix cache from the first — inflating output
# throughput and collapsing TTFT. For a fully cache-free measurement also start
# the server with DISABLE_RADIX=1.
#
# --tokenizer must point at the local weights: the server reports its model_path
# as /models/Kimi-K3, and bench_serving would otherwise try to resolve that as an
# HF repo id ("Repo id must be in the form 'repo_name' or 'namespace/repo_name'").
docker run --rm --name "$NAME" --net=host \
    -v "$HOST_MODEL_DIR/Kimi-K3:/models/Kimi-K3:ro" \
    -v "$RESULTS_DIR:/results" \
    --entrypoint python3 "$IMAGE" \
    -m sglang.bench_serving \
    --backend sglang-oai \
    --host "$HOST" --port "$BPORT" \
    --model "$SERVED_MODEL_NAME" \
    --tokenizer "$MODEL_PATH" \
    --dataset-name random \
    --random-input-len "$ISL" \
    --random-output-len "$OSL" \
    --random-range-ratio 1.0 \
    --num-prompts "$NUM_PROMPTS" \
    --max-concurrency "$CONCURRENCY" \
    --flush-cache \
    --output-file "/results/${TAG}.json" 2>&1 | tee -a "$LOG"

rc=${PIPESTATUS[0]}
echo "### finished=$(date -u +%FT%TZ) rc=${rc}" >> "$LOG"
[[ -f "$JSON" ]] && echo "json : ${JSON}"
exit "$rc"
