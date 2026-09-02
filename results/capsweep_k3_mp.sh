#!/bin/bash
# Sweep MC_MAX_CONCURRENT_REG_MR over the 8-process K3 replay.
#
# The cap is per process, so on a tp8 node the global concurrency is 8x the cap.
# That is the whole reason a single-process sweep gave the opposite answer, so
# print both numbers.
set -u
NRANKS=${NRANKS:-8}
SCALE=${SCALE:-1.0}
CAPS=${CAPS:-"unset 128 64 32 16 8 4"}
OUT=${OUT:-/tmp/mpsweep.txt}
# TAG keeps log files from different orders apart.
TAG=${TAG:-}
export REGBENCH_ORDER=${REGBENCH_ORDER:-desc}

: > "$OUT"
for cap in $CAPS; do
    # Wait for the previous round's GPU memory to actually drain, otherwise a
    # later round can fail to allocate and look like a registration problem.
    for _ in $(seq 60); do
        used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
               | awk '{s+=$1} END {print s+0}')
        [ "$used" -lt 2000 ] && break
        sleep 5
    done

    /tmp/hostrun_k3_mp.sh "$cap" "$NRANKS" "$SCALE" "/tmp/mp_${TAG}${cap}" \
        > /dev/null 2>&1
    rc=$?

    # Wall clock for the node = slowest rank, since all ranks start together and
    # the model server cannot serve until every rank has registered.
    read -r slowest mean n <<< "$(grep -h "^RESULT" /tmp/mp_${TAG}${cap}.r*.log 2>/dev/null \
        | awk '{if($6+0>m) m=$6+0; s+=$6; c++} END {printf "%d %d %d", m, (c?s/c:0), c}')"
    if [ "$cap" = "unset" ]; then glob="~unbounded"; else glob=$((cap * NRANKS)); fi
    printf "order=%-6s cap=%-6s global=%-11s ranks=%s/%s  slowest=%sms  mean=%sms  rc=%s\n" \
        "$REGBENCH_ORDER" "$cap" "$glob" "$n" "$NRANKS" "$slowest" "$mean" "$rc" | tee -a "$OUT"
done
echo "=== done ===" | tee -a "$OUT"
