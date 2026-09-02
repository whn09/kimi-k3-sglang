#!/bin/bash
# Launch the 8-process K3 registration replay against the host-built engine.
# Usage: hostrun_k3_mp.sh <cap|unset> <nranks> <scale> <logprefix>
#
# "unset" removes MC_MAX_CONCURRENT_REG_MR from the environment entirely, which
# is the case that must reproduce the historical unbounded behavior.
set -u
CAP=$1; NRANKS=$2; SCALE=$3; PREFIX=$4
IP=$(hostname -I | awk '{print $1}')

source /opt/pytorch/bin/activate
export CUDA_HOME=/opt/pytorch/lib/python3.13/site-packages/nvidia/cu13
export LD_LIBRARY_PATH=$CUDA_HOME/lib:/opt/amazon/efa/lib:/opt/dlami/nvme/mcsrc/Mooncake/build/mooncake-common:${LD_LIBRARY_PATH:-}
export PYTHONPATH=/opt/dlami/nvme/mcsrc/Mooncake/build/mooncake-integration:${PYTHONPATH:-}
export GLOG_logtostderr=1
# REGBENCH_ORDER / MC_LOG_LEVEL pass through from the caller.
export REGBENCH_ORDER="${REGBENCH_ORDER:-desc}"

if [ "$CAP" = "unset" ]; then
    unset MC_MAX_CONCURRENT_REG_MR
else
    export MC_MAX_CONCURRENT_REG_MR="$CAP"
fi

BDIR=$(mktemp -d /tmp/k3bar.XXXXXX)
trap 'rm -rf "$BDIR"' EXIT

pids=()
for r in $(seq 0 $((NRANKS - 1))); do
    python3 /opt/dlami/nvme/mcsrc/regbench_k3_mp.py \
        "$IP" "$r" "$NRANKS" "$BDIR" "$SCALE" > "${PREFIX}.r${r}.log" 2>&1 &
    pids+=($!)
done

rc=0
for p in "${pids[@]}"; do
    wait "$p" || rc=1
done
exit $rc
