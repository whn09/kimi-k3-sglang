#!/bin/bash
# One sglang.bench_serving run against an already-running K3 server.
#
#   bash bench/bench_k3.sh                      # all defaults, tag auto-generated
#   ISL=8192 OSL=1 NP=16 CONC=4 bash bench/bench_k3.sh
#   TAG=v2_p8k_c2048 ISL=8192 OSL=1 NP=16 CONC=4 bash bench/bench_k3.sh
#
# TAG IS THE CONTRACT WITH gen_bench_table.py. That script has a hard-coded ARMS
# list of *semantic* tags (v2_p8k_c2048, v1_d256, ...) and reads
# /opt/dlami/nvme/k3_bench_<TAG>.txt. So:
#   - to (re)fill a published table row, pass that row's TAG explicitly;
#   - otherwise leave TAG alone and get an auto tag that encodes every axis this
#     script varies. An auto tag deliberately does NOT match ARMS -- a fresh run
#     should not silently overwrite or impersonate a published arm.
#
# The auto tag cannot encode the axes that live on the SERVER, and those dominate:
# CAP and chunked-prefill-size moved prefill throughput 3.94x between two runs of
# the same backend. So instead of pretending, this script records the server's
# real launch args in the log header -- every log is self-describing even when the
# filename is not. Read the header before trusting any number.
set -u

C="${C:-k3-v2-unified}"     # server container to bench (also where the client runs)
PORT="${PORT:-30000}"       # 30000 unified / 30001 prefill / 30002 decode / 30010 v1
# Defaults match 91_bench.sh so the two bench entry points agree.
ISL="${ISL:-1024}"
OSL="${OSL:-1024}"
NP="${NP:-64}"
CONC="${CONC:-32}"
TAG="${TAG:-auto-isl${ISL}-osl${OSL}-n${NP}-c${CONC}}"

OUT="/opt/dlami/nvme/k3_bench_${TAG}.txt"

docker inspect "$C" >/dev/null 2>&1 || {
    echo "no such container: $C (start it with run_k3_v2.sh / run_k3_v1.sh)" >&2
    exit 1
}

# Provenance header. `docker inspect .Args` is the server's actual argv, so it
# survives someone editing the launcher afterwards, and the CAP env var is the one
# knob that is passed by environment rather than on the command line.
{
    echo "### tag=${TAG}"
    echo "### started=$(date -u +%FT%TZ)"
    echo "### client: isl=${ISL} osl=${OSL} num_prompts=${NP} concurrency=${CONC}"
    echo "### container=${C} port=${PORT}"
    echo "### server argv: $(docker inspect -f '{{range .Args}}{{.}} {{end}}' "$C" 2>/dev/null)"
    echo "### server CAP:  $(docker inspect -f '{{range .Config.Env}}{{println .}}{{end}}' "$C" 2>/dev/null \
                            | grep -E 'DISPATCH_TOKENS_PER_RANK' || echo '(unset)')"
} > "$OUT"

# --random-range-ratio 1.0 fixes every prompt at exactly ISL tokens; without it
# bench_serving samples a range and the input-throughput denominator drifts.
docker exec "$C" python3 -m sglang.bench_serving \
  --backend sglang --host 127.0.0.1 --port "$PORT" \
  --dataset-name random --random-input-len "$ISL" --random-output-len "$OSL" \
  --random-range-ratio 1.0 --num-prompts "$NP" --max-concurrency "$CONC" \
  --warmup-requests 2 \
  2>&1 | tee -a "$OUT" | tail -40
rc=${PIPESTATUS[0]}

echo "### finished=$(date -u +%FT%TZ) rc=${rc}" >> "$OUT"
echo "log: $OUT"
exit "$rc"
