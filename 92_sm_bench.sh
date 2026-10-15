#!/bin/bash
# HOST-side (B300-1): one decode-saturation bench per concurrency for the SM
# sweep, each wrapped by decode_sampler.py. Called by 96_sm_sweep.sh; usable by
# hand once the PD pair + router are up.
#
#   REQ_SMS=12 RESOLVED_SMS=12 REP=1 bash 92_sm_bench.sh
#
# REQ_SMS / RESOLVED_SMS / CAP / MAXRUN / SPEC are LABELS: the decode containers
# live on B300-3/4, which this host cannot inspect, so 96_sm_sweep.sh reads them
# from the decode side and passes them in. Every axis goes into the filename.
#
# Writes per concurrency: <tag>.json (bench_serving), <tag>.log, and
# <tag>.decode.json (decode tok/s over the window where decode was FULL).
set -euo pipefail
cd "$(dirname "$0")"
source ./env_common.sh

ISL="${ISL:-1024}"
OSL="${OSL:-2048}"
CAPD="${CAPD:-512}"
MAXRUN_D="${MAXRUN_D:-512}"
SPEC="${SPEC:-off}"
# 1/4, 1/2 and all of the decode admission cap. The top point is the
# "decode full" row; the sampler proves whether it got there.
CONCS="${CONCS:-$(( MAXRUN_D / 4 )) $(( MAXRUN_D / 2 )) $MAXRUN_D}"
NPROMPT_MULT="${NPROMPT_MULT:-2}"          # two full waves per point
REP="${REP:-1}"
REQ_SMS="${REQ_SMS:?REQ_SMS (requested decode SMs, 0 = auto)}"
RESOLVED_SMS="${RESOLVED_SMS:-unknown}"
DECODE_URL="${DECODE_URL:-http://${B300_3_IP}:${PORT}}"
OUT="$SCRIPT_DIR_HOST/results"
mkdir -p "$OUT"

run_bench() {   # run_bench <conc> <n> <tag>
    MODE=pd MOE_A2A_BACKEND=deepep_v2 CAP_PREFILL="$PREFILL_CAP" CAP_DECODE="$CAPD" \
    ISL="$ISL" OSL="$OSL" CONCURRENCY="$1" NUM_PROMPTS="$2" TAG="$3" \
        bash 91_bench.sh
}

echo "== warmup (not saved as a result)"
run_bench 16 32 "warmup_sms${REQ_SMS}_$(date +%s)" >/dev/null 2>&1 || true
rm -f "$OUT"/warmup_sms*

for c in $CONCS; do
    n=$(( c * NPROMPT_MULT )); (( n < 64 )) && n=64
    tag="k3s_pd2x2_dec-tp16ep16_sms${REQ_SMS}_spec${SPEC}_cap${CAPD}_run${MAXRUN_D}_isl${ISL}_osl${OSL}_c${c}_n${n}_r${REP}_$(date +%Y%m%d-%H%M%S)"
    echo "== $tag  (resolved sms=$RESOLVED_SMS)"
    python3 decode_sampler.py --out "$OUT/$tag.decode.json" --target "$c" "$DECODE_URL" &
    spid=$!
    run_bench "$c" "$n" "$tag" 2>&1 | grep -E "Successful requests|Benchmark duration|Output token throughput|Median TTFT|Median TPOT|P99 TPOT|Median ITL" || true
    kill -INT "$spid"; wait "$spid" || true
    # A bench that dies before sending a request still "finishes"; no JSON = failed.
    if [[ ! -s "$OUT/$tag.json" ]]; then
        echo "FATAL: bench_serving wrote no $tag.json -- tail of its log:" >&2
        tail -8 "$OUT/$tag.log" >&2 || true
        exit 1
    fi
    python3 - "$OUT/$tag.decode.json" <<PY
import json, sys
p = sys.argv[1]; d = json.load(open(p))
d.update(requested_sms="$REQ_SMS", resolved_sms="$RESOLVED_SMS", rep=$REP, spec="$SPEC",
         cap_decode=$CAPD, maxrun_decode=$MAXRUN_D, image="$IMAGE")
json.dump(d, open(p, "w"))
PY
done
