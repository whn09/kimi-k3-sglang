#!/bin/bash
# v1 BASELINE: unpatched image, --moe-a2a-backend deepep. Same everything else as
# run_k3_v2.sh so a greedy-output diff attributes to the a2a backend, not config.
set -u
NAME=k3-v1
IMG=lmsysorg/sglang:nightly-dev-cu13-20260901-07c8f729
CHUNK="${CHUNK:-2048}"
MEMFRAC="${MEMFRAC:-0.85}"
MAXRUN="${MAXRUN:-32}"
# v1 has no DeepEP-v2 ElasticBuffer, so it can afford a bigger CHUNK than v2 --
# which is exactly why the comparison must be run at a MATCHED CHUNK. Prefill
# throughput is ~linear in CHUNK on this model (v2: 512 -> 2048 gave 3.94x), so an
# unmatched chunk would swamp any real backend difference.
#
# THE DEFAULTS OF THE TWO SCRIPTS ARE NOT MATCHED, ON PURPOSE. run_k3_v2.sh
# defaults to MAXRUN=0 (unlimited) because it has to be a usable *server* on pure
# defaults; this script keeps MAXRUN=32 because that is the arm the numbers in
# results/deepep_v2_on_k3_b300.md were measured at. MAXRUN changes both the
# concurrency cap and max_total_num_tokens (measured on v2 at CAP=CHUNK=2048,
# graphs off, everything else equal: MAXRUN 0 -> 232384, 32 -> 364416), so for any
# v1/v2 comparison pass the SAME MAXRUN and the SAME CHUNK to both explicitly. Do
# not compare two default launches.
RUNARGS=(); [ "$MAXRUN" -gt 0 ] && RUNARGS=(--max-running-requests "$MAXRUN")
RADIXARGS=(); [ "${NORADIX:-1}" = 1 ] && RADIXARGS=(--disable-radix-cache)
CGARGS=(); [ "${DISCG:-0}" = 1 ] && CGARGS=(--disable-cuda-graph)
docker rm -f "$NAME" >/dev/null 2>&1
# See run_k3_v2.sh: `docker rm -f` returns both before the container is gone (it
# can linger as Exited 137, and then `docker run` fails "name is already in use")
# and before the GPUs are released. Wait for both, in that order.
for _ in $(seq 1 30); do
  docker inspect "$NAME" >/dev/null 2>&1 || break
  echo "waiting for $NAME to be removed"; docker rm -f "$NAME" >/dev/null 2>&1; sleep 2
done
for _ in $(seq 1 60); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | paste -sd+ | bc)
  [ "${used:-1}" -lt 2048 ] && break
  echo "waiting for GPU drain: ${used} MiB still in use"; sleep 5
done
docker run -d --name "$NAME" --gpus all --net=host --ipc=host --shm-size 32g \
  -v /opt/dlami/nvme/models:/models:ro \
  -v /opt/dlami/nvme/cache/deep_gemm:/root/.cache/deep_gemm \
  -v /opt/dlami/nvme/cache/torch:/root/.cache/torch \
  -v /opt/dlami/nvme/cache/flashinfer:/root/.cache/flashinfer \
  -v /opt/dlami/nvme/cache/tvm-ffi:/root/.cache/tvm-ffi \
  -v /opt/dlami/nvme/cache/sglang:/root/.cache/sglang \
  -v /opt/dlami/nvme/cache/triton:/root/.triton \
  -v /opt/dlami/nvme/cache/nv_compute:/root/.nv/ComputeCache \
  --device=/dev/gdrdrv --device=/dev/infiniband \
  --cap-add IPC_LOCK --ulimit memlock=-1 \
  -e NCCL_IB_HCA="${HCA:-ibp198s0f0}" \
  -e SGLANG_LOAD_TIMEOUT=7200 -e PYTHONUNBUFFERED=1 \
  --entrypoint python3 "$IMG" \
  -m sglang.launch_server --model-path /models/Kimi-K3 --trust-remote-code \
  --tp-size 8 --ep-size 8 --moe-a2a-backend deepep --deepep-mode auto \
  --moe-runner-backend deep_gemm --disable-prefill-cuda-graph \
  --chunked-prefill-size "$CHUNK" --max-prefill-tokens "$CHUNK" \
  "${RUNARGS[@]}" "${RADIXARGS[@]}" "${CGARGS[@]}" \
  --mem-fraction-static "$MEMFRAC" --mamba-full-memory-ratio 0.86 \
  --host 0.0.0.0 --port 30010 --decode-log-interval 1 \
  --watchdog-timeout 1000000 --dist-timeout 7200 >/dev/null
echo "launched $NAME on port 30010 chunk=$CHUNK memfrac=$MEMFRAC"
