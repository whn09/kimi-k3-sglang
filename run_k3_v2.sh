#!/bin/bash
# Launch PATCHED K3 + deepep_v2.  $1 = unified|prefill|decode
#
# Five files are patched and bind-mounted over the image (see /opt/dlami/nvme/patch;
# per-blocker root cause in patches/README.md):
#   moe_hook.py     : KimiK3ForConditionalGeneration added to the deepep_v2 whitelist
#   kimi_k3.py x2   : is_deepep_v2() added to the two hand-rolled _ep_a2a lists, which
#                     listed every other EP-a2a backend but not v2. Without this the
#                     MoE region would keep its DP-gather / TP-reduce while the v2
#                     dispatcher already did the a2a -- the exact hazard the upstream
#                     whitelist error message warns about.
#   fmt_layer.py    : the FP4/MXFP4 quant-method gate, which is stricter than the
#                     deep_gemm kernels behind it
#   mr_deep_gemm.py : the v2 pre-permute's `activation == "silu"` assert, widened to
#                     accept K3's "situ" (sglang PR #37514)
#   ep_moe_kernels.py: ep_scatter_from_psum missing kernel args (sglang PR #37211)
# EXPERIMENT ONLY: correctness must be checked against a deepep(v1) baseline.
# Numerics were checked -- first-token top-5 logprobs are byte-identical to v1.
#
# HARDWARE NOTE: this B300 box is *not* EFA. It has 2x Mellanox ConnectX-7
# (MT2910, link_layer InfiniBand, PORT_ACTIVE); the efa kernel module is loaded
# but exposes no device. So the GIN backend here is GDAKI (NCCL_GIN_TYPE=3,
# IB/DOCA), not EFA_GDA(5). sgl-deep-ep asserts ginType != NONE even for a
# single-node `direct` run (csrc/kernels/backend/nccl.cu:87), and without
# --device=/dev/infiniband NCCL sees no network at all -> NONE.
#
# The two HCAs sit on two IB planes with NO path between them: ibv_rc_pingpong
# loopback inside ibp198s0f0 works (11 us/iter) but ibp198s0f0 <-> ibp199s0f0
# fails with "transport retry counter exceeded (12)" -- the same
# IBV_WC_RETRY_EXC_ERR that GDAKI context creation died on
# (ncclGinIbGdakiCreateContext -> gin.cc:288). Hence NCCL_IB_HCA pins ONE plane.
set -u
MODE="${1:-unified}"
NAME="k3-v2-$MODE"
IMG=lmsysorg/sglang:nightly-dev-cu13-20260901-07c8f729
P=/opt/dlami/nvme/patch
# THE DEFAULTS BELOW ARE THE MEASURED-WORKING CONFIG. `bash run_k3_v2.sh` with no
# env overrides reaches READY in ~165 s with max_total_num_tokens=232384 and serves.
# (MAXRUN=32 instead gives a bigger pool, 364416, at the cost of capping
# concurrency at 32 -- that is the prefill-measurement arm, not a better default.)
# Everything here is coupled, so change one knob and you are off the measured
# path -- the failure is always a late OOM, never a clear error:
#   CAP=2048   is the ceiling. The v2 ElasticBuffer costs exactly 10.5 GiB per
#              1024 of CAP, there are ~39.6 GB free after the KV pool, and
#              CAP=3072 asks 31.50 GiB but still OOMs.
#   CHUNK=CAP  is forced: validate_deepep_v2_dispatch_token_budget
#              (moe_hook.py:381) enforces chunked_prefill_size/dp_size <= CAP.
#   DP=1       DP attention is NOT an escape hatch here, despite making the
#              budget check trivially pass. It stops TP-sharding the attention
#              weights and replicates them per DP rank: weights go 214.88 ->
#              265.87 GiB of 267.68 and it OOMs inside _initialize_model, before
#              the KV pool even exists. Measured, not theorised -- DP=8 is what
#              this script used to default to, and it could never start.
#   DISCG=1    decode CUDA graphs are OFF by default. Capture takes a 33.43 GiB
#              private pool, which does not fit alongside a 21 GiB
#              ElasticBuffer: at CAP=2048 with graphs on, capture OOMs asking
#              6.12 GiB. Graphs on caps CAP at 1024 (verified: READY,
#              max_total=232384) -- i.e. graphs cost half the prefill chunk, and
#              prefill throughput here is nearly linear in chunk (512 -> 2048 is
#              3.94x on the same backend). For a decode-latency measurement set
#              DISCG=0 CAP=1024 CHUNK=1024 and accept the smaller chunk.
CAP="${CAP:-2048}"
CHUNK="${CHUNK:-$CAP}"
MEMFRAC="${MEMFRAC:-0.85}"
DP="${DP:-1}"          # DP attention degree; 1 disables it. See above: >1 OOMs.
DPARGS=()
[ "$DP" -gt 1 ] && DPARGS=(--enable-dp-attention --dp-size "$DP")
MAXRUN="${MAXRUN:-0}"   # 0 = leave sglang's default
RUNARGS=()
[ "$MAXRUN" -gt 0 ] && RUNARGS=(--max-running-requests "$MAXRUN")

# --- the knob that actually buys ElasticBuffer headroom: --dcp-size ---
# Budget per B300 (267.68 GiB): K3 weights are 1.5 TB / 8 = ~187.5 GiB per rank,
# and --mem-fraction-static covers weights + pool. So MEMFRAC=0.85 -> 227.5 GiB
# static -> ~40 GiB pool and ~40 GiB left outside for the v2 ElasticBuffer.
# Measured ElasticBuffer cost is exactly 10.5 GiB per 1024 of CAP (CAP=3072 asked
# 31.50 GiB and OOMed; CAP=4096 asked 42.00 GiB). To afford a bigger CAP the pool
# must shrink, but plain MEMFRAC=0.75 dies with "Not enough GPU memory for hybrid
# (mamba/linear-attention) state cache".
# --dcp-size 8 shards the decode KV/state across all 8 GPUs, which is what lets a
# smaller pool still hold the required states -- upstream's own "balanced" profile
# spends that on --mamba-full-memory-ratio 5.13 (vs 0.86 at dcp=1) and turns the
# custom all-reduce off. See env_common.sh for the cookbook profile table.
DCP="${DCP:-1}"
MAMBA="${MAMBA:-0.86}"
DCPARGS=()
[ "$DCP" -gt 1 ] && DCPARGS=(--dcp-size "$DCP" --disable-custom-all-reduce)
# Repeatable benchmarks: radix cache off stops cross-request prefix reuse from
# flattering prefill numbers (upstream's start_standalone.sh does the same).
RADIXARGS=()
[ "${NORADIX:-1}" = 1 ] && RADIXARGS=(--disable-radix-cache)

# MEASURED allocation order and cost per B300 (267.68 GiB), MEMFRAC=0.85:
#   Load weight end   -> mem usage 214.88 GB, avail 51.11 GB
#   KV Cache          -> 5.99 GB / 232384 tokens, avail 39.63 GB
#   decode CUDA graph -> capture begins at avail 38.99 GB and takes a 33.43 GiB
#                        private pool
#   DeepEP v2 buffer  -> LAST, 10.5 GiB per 1024 of CAP
# So the competition is ElasticBuffer vs CUDA graph, not vs the KV pool, and 33.43
# + 21 does not fit in 39.63: at CAP=2048 with graphs on it is *capture* that OOMs
# (asking 6.12 GiB), before the ElasticBuffer is ever allocated. Off by default.
CGARGS=()
[ "${DISCG:-1}" = 1 ] && CGARGS=(--disable-cuda-graph)

DIS=(); PORTX=30000
case "$MODE" in
  prefill) DIS=(--disaggregation-mode prefill --disaggregation-transfer-backend mooncake); PORTX=30001 ;;
  decode)  DIS=(--disaggregation-mode decode  --disaggregation-transfer-backend mooncake); PORTX=30002 ;;
esac

docker rm -f "$NAME" >/dev/null 2>&1
# `docker rm -f` returns before it has finished, in two independent ways, and both
# break the very next launch:
#   1. the container can still EXIST (killed, Exited 137) when `docker run` runs,
#      which fails outright with "container name is already in use";
#   2. even once it is gone, the 8 scheduler processes may not have released their
#      GPU memory. Relaunching then OOMs during *weight load* on a tiny alloc
#      (seen: "Tried to allocate 588.00 MiB ... free: 18022400"), which looks
#      exactly like a mem-fraction problem and is not.
# So wait for both, in that order.
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
  -v "$P/kimi_k3.py:/sgl-workspace/sglang/python/sglang/srt/models/kimi_k3.py:ro" \
  -v "$P/moe_hook.py:/sgl-workspace/sglang/python/sglang/srt/arg_groups/moe_hook.py:ro" \
  -v "$P/fmt_layer.py:/sgl-workspace/sglang/python/sglang/srt/layers/moe/fused_moe_triton/layer.py:ro" \
  -v "$P/mr_deep_gemm.py:/sgl-workspace/sglang/python/sglang/srt/layers/moe/moe_runner/deep_gemm.py:ro" \
  -v "$P/ep_moe_kernels.py:/sgl-workspace/sglang/python/sglang/kernels/ops/moe/ep_moe_kernels.py:ro" \
  -v /opt/dlami/nvme/cache/deep_gemm:/root/.cache/deep_gemm \
  -v /opt/dlami/nvme/cache/torch:/root/.cache/torch \
  -v /opt/dlami/nvme/cache/flashinfer:/root/.cache/flashinfer \
  -v /opt/dlami/nvme/cache/tvm-ffi:/root/.cache/tvm-ffi \
  -v /opt/dlami/nvme/cache/sglang:/root/.cache/sglang \
  -v /opt/dlami/nvme/cache/triton:/root/.triton \
  -v /opt/dlami/nvme/cache/nv_compute:/root/.nv/ComputeCache \
  --device=/dev/gdrdrv --device=/dev/infiniband \
  --privileged --ulimit memlock=-1 \
  -e NCCL_GIN_TYPE="${GIN:-3}" \
  -e NCCL_IB_HCA="${HCA:-ibp198s0f0}" \
  -e NCCL_DEBUG="${NCCL_DEBUG:-INFO}" -e NCCL_DEBUG_SUBSYS="${NCCL_DEBUG_SUBSYS:-INIT,NET,GIN}" \
  -e SGLANG_LOAD_TIMEOUT=7200 -e PYTHONUNBUFFERED=1 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e SGLANG_DEEPEP_V2_NUM_MAX_DISPATCH_TOKENS_PER_RANK="$CAP" \
  --entrypoint python3 "$IMG" \
  -m sglang.launch_server --model-path /models/Kimi-K3 --trust-remote-code \
  --tp-size 8 --ep-size 8 --moe-a2a-backend deepep_v2 --deepep-v2-mode direct \
  "${DPARGS[@]}" \
  --moe-runner-backend deep_gemm \
  --chunked-prefill-size "$CHUNK" --max-prefill-tokens "$CHUNK" \
  "${DIS[@]}" "${RUNARGS[@]}" "${DCPARGS[@]}" "${RADIXARGS[@]}" "${CGARGS[@]}" \
  --mem-fraction-static "$MEMFRAC" --mamba-full-memory-ratio "$MAMBA" \
  --host 0.0.0.0 --port "$PORTX" --decode-log-interval 1 \
  --watchdog-timeout 1000000 --dist-timeout 7200 >/dev/null
echo "launched $NAME  mode=$MODE port=$PORTX cap=$CAP chunk=$CHUNK dp=$DP dcp=$DCP mamba=$MAMBA maxrun=$MAXRUN memfrac=$MEMFRAC gin=${GIN:-3} hca=${HCA:-ibp198s0f0}"
