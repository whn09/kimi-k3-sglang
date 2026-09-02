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
# THE DEFAULTS BELOW ARE THE MEASURED-BEST CONFIG for serving. `bash run_k3_v2.sh`
# with no env overrides reaches READY in ~230 s (of which 61.59 s is CUDA graph
# capture) with max_total_num_tokens=232384, and on 8K in / 1K out at concurrency 16
# it is 3.27x the end-to-end throughput of the previous defaults. See DISCG below
# for the both-ways measurement.
# (MAXRUN=32 instead gives a bigger pool, 364416, at the cost of capping
# concurrency at 32 -- that is the prefill-measurement arm, not a better default.)
# Everything here is coupled, so change one knob and you are off the measured
# path -- the failure is always a late OOM, never a clear error:
#   CAP        is NOT an sglang flag -- there is no --cap. It is this script's name
#              for the env var SGLANG_DEEPEP_V2_NUM_MAX_DISPATCH_TOKENS_PER_RANK
#              (environ.py:1085, sglang's own default is 128, so 2048 is 16x
#              upstream). It is DeepEP v2's per-rank communication buffer capacity
#              in tokens. Because it is passed by environment it does not appear in
#              `docker inspect .Args` -- bench_k3.sh reads it out of .Config.Env
#              separately for exactly that reason.
#
#              ONE capacity is bound by TWO unrelated constraints
#              (validate_deepep_v2_dispatch_token_budget, moe_hook.py:375-410):
#                prefill:      CHUNK / dp_size  <= CAP   -> wants 2048
#                decode graph: graph_bs * tokens_per_req <= CAP -> wants only 104
#              So raising CAP to buy a big prefill chunk also inflates the decode
#              graph's workspace by the same 16x. That is where the 33.43 GiB
#              capture pool comes from, and why trimming the captured bs list does
#              not help at all (measured: bs=[8,16,24] still took 33.43 GiB and
#              OOMed, while CAP=1024 with all 13 sizes took 18.48 GB and started).
#              The two needs should not share one number; that is worth an upstream
#              issue.
#   CAP=1024   is the ceiling WITH decode graphs, and it is the default because
#              decode graphs are worth far more than the chunk they cost -- see
#              DISCG below for the measurement. The arithmetic is tight: capture
#              starts at avail 38.99 GB and takes 18.48 GB, leaving 20.51 GB for a
#              10.5 GiB ElasticBuffer. CAP=1536 would need ~27.7 + 15.75 = 43 GB of
#              38.99 and cannot work, so do not bother trying 1536 or 2048 with
#              graphs on. (Graphs OFF, the ceiling is CAP=2048: ElasticBuffer is
#              10.5 GiB per 1024 of CAP and CAP=3072 asks 31.50 GiB and OOMs.)
#   CHUNK=CAP  is forced by the prefill constraint above, at dp_size=1.
#   DP=1       DP attention is NOT an escape hatch here, despite making the
#              budget check trivially pass. It stops TP-sharding the attention
#              weights and replicates them per DP rank: weights go 214.88 ->
#              265.87 GiB of 267.68 and it OOMs inside _initialize_model, before
#              the KV pool even exists. Measured, not theorised -- DP=8 is what
#              this script used to default to, and it could never start.
#   DISCG=0    decode CUDA graphs are ON by default, which is what forces CAP down
#              to 1024. This trade was MEASURED both ways, 8K in / 1K out at
#              concurrency 16, same 232384-token pool on both sides
#              (results/deepep_v2_on_k3_b300.md 4.2):
#
#                                  graphs OFF/CAP=2048   graphs ON/CAP=1024
#                end-to-end (32 req)      565.55 s            172.77 s   3.27x
#                output tok/s               57.94              189.67    3.27x
#                ITL p50                   255.24 ms           43.54 ms  5.86x
#                TTFT p50                11735.66 ms        22571.74 ms  0.52x
#
#              Graphs cost half the prefill chunk, so TTFT doubles -- but each
#              token after the first arrives 5.86x sooner. Break-even is at ~52
#              output tokens (11.7 + 0.255N = 22.6 + 0.0435N), so anything that
#              generates a real answer wins, by a lot.
#              SET DISCG=1 CAP=2048 CHUNK=2048 for a prefill-only measurement
#              (OSL=1) or a workload whose outputs are shorter than ~52 tokens --
#              there the big chunk wins and graphs are dead weight.
#              Do NOT try to keep both by trimming the captured batch sizes: the
#              capture pool is sized by CAP, not by how many shapes are captured
#              (see CAP above for the measurement that settles this).
CAP="${CAP:-1024}"
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
#   decode CUDA graph -> capture begins at avail 38.99 GB; 18.48 GB at CAP=1024,
#                        33.43 GiB at CAP=2048 (which then OOMs asking 6.12 GiB)
#   DeepEP v2 buffer  -> LAST, 10.5 GiB per 1024 of CAP
# So the competition is ElasticBuffer vs CUDA graph, never the KV pool, and at
# CAP=2048 it is *capture* that dies, before the ElasticBuffer is ever allocated.
# CAP=1024 is the largest CAP where both fit: 18.48 + 10.5 <= 38.99.
#
# CGMAXBS trims the captured batch-size list. It does NOT save memory -- that was
# the hypothesis and it is now disproved. Two measurements at avail 38.99 GB:
#   CAP=2048, bs=[8,16,24]                  -> 33.43 GiB, OOM
#   CAP=1024, bs=[8,16,...,104] (13 shapes) ->  18.48 GB, starts
# Four times fewer shapes cost MORE memory; the only variable that moved the number
# is CAP. The capture pool is sized by the per-rank capacity, not by how many shapes
# are captured, so there is no way to keep CAP=2048 and graphs at the same time.
# What CGMAXBS is still good for is startup time: capture takes 61.59 s for 13
# shapes, and this server cannot reach a decode batch above ~25 anyway (the KV pool
# caps live requests at max_total_num_tokens/(ISL+OSL) = 232384/9216 = 25 at 8K/1K),
# so capturing up to 104 is wasted time. UNMEASURED as a default, and keep it above
# the pool cap or large batches silently fall out of the graph and back to 255 ms.
# NOTE the flag was renamed: --cuda-graph-max-bs is now a deprecated alias for
# --cuda-graph-max-bs-decode. Only meaningful with DISCG=0.
CGARGS=()
if [ "${DISCG:-0}" = 1 ]; then
  CGARGS=(--disable-cuda-graph)
elif [ -n "${CGMAXBS:-}" ]; then
  CGARGS=(--cuda-graph-max-bs-decode "$CGMAXBS")
fi

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
# Print every axis that decides whether this boots and how fast it decodes --
# especially DISCG/CGMAXBS, which were missing here and are the difference between
# a 256 ms and a ~35 ms decode step. `cap` is spelled out: it is an env var, not a
# flag, so it is invisible in `docker inspect .Args`.
echo "launched $NAME  mode=$MODE port=$PORTX chunk=$CHUNK dp=$DP dcp=$DCP mamba=$MAMBA maxrun=$MAXRUN memfrac=$MEMFRAC gin=${GIN:-3} hca=${HCA:-ibp198s0f0}"
echo "  SGLANG_DEEPEP_V2_NUM_MAX_DISPATCH_TOKENS_PER_RANK=$CAP  (env var, not a flag; upstream default 128)"
echo "  decode cuda graph: $([ "${DISCG:-0}" = 1 ] && echo "DISABLED (DISCG=1) -- expect ~255 ms ITL" || echo "ON, max-bs-decode=${CGMAXBS:-<sglang default>} -- expect ~44 ms ITL")"
