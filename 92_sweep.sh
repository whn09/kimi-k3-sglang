#!/bin/bash
# Concurrency sweep driver. Runs 91_bench.sh at each concurrency against an
# already-running server, saving one raw log + JSON per point.
#
#   MODE=standalone PROFILE=low-latency bash 92_sweep.sh
#   MODE=pd PROFILE=low-latency ENDPOINT=localhost:8080 bash 92_sweep.sh
#   CONCURRENCIES="1 8 32" bash 92_sweep.sh
#
# NUM_PROMPTS scales with concurrency (PROMPTS_PER_CONC rounds per point) so
# every point measures a comparable number of steady-state batches instead of
# spending most of its time ramping up and draining.
set -uo pipefail

cd "$(dirname "$0")"
source ./env_common.sh

ISL="${ISL:-8192}"
OSL="${OSL:-1024}"
MODE="${MODE:-standalone}"
CONCURRENCIES="${CONCURRENCIES:-1 8 16 32 64}"
PROMPTS_PER_CONC="${PROMPTS_PER_CONC:-3}"
MIN_PROMPTS="${MIN_PROMPTS:-8}"
RESULTS_DIR="${RESULTS_DIR:-$SCRIPT_DIR_HOST/results}"
mkdir -p "$RESULTS_DIR"

SUMMARY="$RESULTS_DIR/sweep-${MODE}-${PROFILE}-isl${ISL}-osl${OSL}.txt"
echo "sweep: mode=$MODE profile=$PROFILE isl=$ISL osl=$OSL conc='$CONCURRENCIES'" | tee "$SUMMARY"

for c in $CONCURRENCIES; do
    n=$(( c * PROMPTS_PER_CONC ))
    [[ "$n" -lt "$MIN_PROMPTS" ]] && n="$MIN_PROMPTS"
    echo "" | tee -a "$SUMMARY"
    echo "===== concurrency=$c num_prompts=$n =====" | tee -a "$SUMMARY"
    ISL="$ISL" OSL="$OSL" NUM_PROMPTS="$n" CONCURRENCY="$c" \
        MODE="$MODE" PROFILE="$PROFILE" RESULTS_DIR="$RESULTS_DIR" \
        bash ./91_bench.sh 2>&1 \
        | grep -E "Successful requests|Benchmark duration|Request throughput|Output token throughput|Total token throughput|Mean TTFT|Median TTFT|P99 TTFT|Mean TPOT|Median TPOT|Mean ITL|Median ITL|accept length|Concurrency" \
        | tee -a "$SUMMARY"
done

echo "" | tee -a "$SUMMARY"
echo "summary: $SUMMARY" | tee -a "$SUMMARY"
