#!/bin/bash
# bench_k3.sh <container> <port> <isl> <osl> <numprompts> <conc> <tag>
set -u
C=$1; PORT=$2; ISL=$3; OSL=$4; NP=$5; CONC=$6; TAG=$7
docker exec $C python3 -m sglang.bench_serving \
  --backend sglang --host 127.0.0.1 --port $PORT \
  --dataset-name random --random-input-len $ISL --random-output-len $OSL \
  --random-range-ratio 1.0 --num-prompts $NP --max-concurrency $CONC \
  --warmup-requests 2 \
  2>&1 | tee /opt/dlami/nvme/k3_bench_${TAG}.txt | tail -40
