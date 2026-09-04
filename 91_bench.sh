#!/bin/bash
# Throughput / latency benchmark via sglang.bench_serving.
#
#   bash 91_bench.sh                                  # standalone, 1k/1k, 32 concurrent
#   ENDPOINT=localhost:8080 bash 91_bench.sh          # via the PD router
#   ISL=4096 OSL=512 NUM_PROMPTS=128 CONCURRENCY=64 bash 91_bench.sh
#   WL=mixed MODE=pd bash 91_bench.sh                 # named shape (env_common.sh)
#   WL=decode MODE=agg MACHINES=4 bash 91_bench.sh    # the aggregated baseline
#
# Raw output is tee'd to $RESULTS_DIR/<TAG>.log and the JSON summary written
# alongside it, so a whole sweep leaves an auditable trail.
set -uo pipefail

cd "$(dirname "$0")"
source ./env_common.sh

# standalone = one server on this host; pd = prefill/decode split behind
# 22_launch_router.sh; agg = N independent standalone servers behind
# 23_launch_agg_router.sh (the matched-capacity baseline).
MODE="${MODE:-standalone}"
# Under PD the default endpoint MUST be the router, not $PORT. Benching
# localhost:30000 in PD mode hits the PREFILL worker directly, which answers a
# completion request with a body that has no "choices" key, and bench_serving
# reports that as `Warmup failed ... KeyError: 'choices'` -- an error that says
# nothing about the actual mistake. Pass ENDPOINT= explicitly to override.
# `agg` is the same story for a different reason: hitting one worker directly
# would measure one machine and label it N.
case "$MODE" in
    pd|agg) ENDPOINT="${ENDPOINT:-localhost:$ROUTER_PORT}" ;;
    *)      ENDPOINT="${ENDPOINT:-localhost:$PORT}" ;;
esac
HOST="${ENDPOINT%%:*}"
BPORT="${ENDPOINT##*:}"
# WL (env_common.sh) names a shape; unset keeps the historical 1k/1k defaults so
# no published filename changes meaning. NUM_PROMPTS scales with concurrency only
# under a WL preset, for the same reason 92_sweep.sh scales it: a high-concurrency
# point at a fixed request count spends its wall clock ramping and draining. The
# WL_PPC=2 presets reproduce n=64 at c=32 exactly.
ISL="${ISL:-${WL_ISL:-1024}}"
OSL="${OSL:-${WL_OSL:-1024}}"
CONCURRENCY="${CONCURRENCY:-32}"
if [[ -n "${WL_PPC:-}" ]]; then
    NUM_PROMPTS="${NUM_PROMPTS:-$(( CONCURRENCY * WL_PPC ))}"
else
    NUM_PROMPTS="${NUM_PROMPTS:-64}"
fi
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
#
# The EP SIZE and the NODE COUNT belong in the name for the same reason, and they
# are read from the CONTAINER, not from this shell: a 2-node ep=16 run and a
# 1-node ep=8 run at the same profile/ISL/OSL/c/n otherwise write the same file
# and the second deletes the first -- and `bash 91_bench.sh` without the launch's
# TP_SIZE= would label the ep=16 server "ep8".
EPTAG=""
if [[ "${MOE_A2A_BACKEND:-none}" != "none" ]]; then
    if [[ "$MODE" == "pd" ]]; then
        SRV=kimi-k3-prefill
        EPTAG="-${MOE_A2A_BACKEND}-p$(read_cap kimi-k3-prefill "${CAP_PREFILL:-}")d$(read_cap kimi-k3-decode "${CAP_DECODE:-}")"
    else
        SRV=kimi-k3
        EPTAG="-${MOE_A2A_BACKEND}-cap$(read_cap kimi-k3)"
    fi
    RUN_EP="$(read_cenv "$SRV" EP_SIZE "${EP_SIZE:-}")"
    RUN_NN="$(read_cenv "$SRV" NNODES "${NNODES:-1}")"
    RUN_MODE="$(read_cenv "$SRV" DEEPEP_V2_MODE "${DEEPEP_V2_MODE:-}")"
    EPTAG="-ep${RUN_EP}x${RUN_NN}node${EPTAG}"
fi
# NUM_PROMPTS belongs in the name too. It was missing, and a c16 run at 32
# requests then wrote the same file as a c16 run at 64 -- two different
# denominators, one filename, second overwrites the first.
#
# THE MACHINE COUNT IS THE AXIS THIS WHOLE COMPARISON TURNS ON, so it cannot be
# the one that is missing from the name: a 2-machine agg run and a 4-machine agg
# run are identical in every other field and would overwrite each other. Under
# MODE=agg it is derived from AGG_IPS (one entry per INSTANCE, so a 2x2-node
# layout would under-count -- 94_matched.sh passes MACHINES= explicitly and is
# the only thing that builds such a layout). Empty leaves names unchanged.
if [[ "$MODE" == "agg" && -z "${MACHINES:-}" ]]; then
    MACHINES="$(echo $AGG_IPS | wc -w | tr -d ' ')"
fi
MTAG=""
[[ -n "${MACHINES:-}" ]] && MTAG="-m${MACHINES}"
TAG="${TAG:-${MODE}${MTAG}-${PROFILE}${EPTAG}-isl${ISL}-osl${OSL}-c${CONCURRENCY}-n${NUM_PROMPTS}}"
mkdir -p "$RESULTS_DIR"
LOG="$RESULTS_DIR/${TAG}.log"
JSON="$RESULTS_DIR/${TAG}.json"

echo "bench: ${ENDPOINT}  isl=${ISL} osl=${OSL} n=${NUM_PROMPTS} conc=${CONCURRENCY}"
echo "log  : ${LOG}"

{
  echo "### tag=${TAG}"
  echo "### mode=${MODE} profile=${PROFILE} endpoint=${ENDPOINT}"
  echo "### wl=${WL:-custom} machines=${MACHINES:-unstated}"
  echo "### isl=${ISL} osl=${OSL} num_prompts=${NUM_PROMPTS} concurrency=${CONCURRENCY}"
  # Recorded even when EP is off, so a log can never be ambiguous about which
  # MoE path produced it. "unknown" means the container was not reachable from
  # here (e.g. benching a remote endpoint) -- treat such a row as unlabelled.
  if [[ "$MODE" == "pd" ]]; then
      echo "### a2a=${MOE_A2A_BACKEND} ep=${RUN_EP:-$EP_SIZE} mode=${RUN_MODE:-$DEEPEP_V2_MODE} nnodes=${RUN_NN:-$NNODES} cap_prefill=$(read_cap_src kimi-k3-prefill "${CAP_PREFILL:-}") cap_decode=$(read_cap_src kimi-k3-decode "${CAP_DECODE:-}")"
  else
      echo "### a2a=${MOE_A2A_BACKEND} ep=${RUN_EP:-$EP_SIZE} mode=${RUN_MODE:-$DEEPEP_V2_MODE} nnodes=${RUN_NN:-$NNODES} cap=$(read_cap kimi-k3)"
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
