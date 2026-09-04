#!/bin/bash
# Shared config for all Kimi-K3 B300 scripts. Sourced by both host launchers and
# in-container start scripts, so keep it POSIX-ish and side-effect free.

# ---- paths (host) ----
NVME="${NVME:-/opt/dlami/nvme}"
HOST_MODEL_DIR="${HOST_MODEL_DIR:-$NVME/models}"
HOST_CACHE_DIR="${HOST_CACHE_DIR:-$NVME/cache}"
SCRIPT_DIR_HOST="${SCRIPT_DIR_HOST:-/home/ubuntu/kimi-k3-sglang}"

# ---- JIT / autotune caches to persist on the host ----
# Every launch otherwise recompiles from scratch: on a cold container 100% of the
# files in these dirs are newly written (verified with find -newermt), costing
# ~140 s inside "Load weight" (the three "Precompiled ..." K3 kernels) plus a
# ~4.4 min FlashInfer autotune. Note that the three dirs an SGLang guide usually
# tells you to mount — deep_gemm, torch, flashinfer — stay at 4-12 KB on this
# image: the real caches moved to tvm-ffi / .triton / .nv / .cache/sglang.
#
# "container path=host subdir". Keys inside these caches include the arch and
# the library version (e.g. ..._arch_10.3a__tvmffi_0.1.11), so they are safe
# across image rebuilds; wipe $HOST_CACHE_DIR if a rebuild ever misbehaves.
CACHE_MOUNTS=(
    "/root/.cache/deep_gemm=deep_gemm"
    "/root/.cache/torch=torch"
    "/root/.cache/flashinfer=flashinfer"
    "/root/.cache/tvm-ffi=tvm-ffi"          # sgl_kernel JIT (K3 attn/MoE), ~58 MB
    "/root/.cache/sglang=sglang"            # FlashInfer autotune results
    "/root/.triton=triton"                  # Triton kernels
    "/root/.nv/ComputeCache=nv_compute"     # CUDA/PTX JIT, ~800 MB
)

# Fills the CACHE_ARGS array with docker -v flags, creating the host dirs first.
# Extra args are further "container=host" entries (decode adds symm_allocator).
build_cache_args() {
    local entry cpath hsub
    CACHE_ARGS=()
    for entry in "${CACHE_MOUNTS[@]}" "$@"; do
        cpath="${entry%%=*}"; hsub="${entry#*=}"
        mkdir -p "$HOST_CACHE_DIR/$hsub"
        CACHE_ARGS+=(-v "$HOST_CACHE_DIR/$hsub:$cpath")
    done
}

# Fills GDR_ARGS with the /dev/gdrdrv device flag when the host has the gdrdrv
# module loaded. Without it aws-ofi-nccl logs "NET/OFI Failed to initialize
# GDRCopy: Failed to open gdr handle" and falls back to a slower host-memory
# path. The PD launchers happened to get the device for free because
# --privileged bind-mounts the host's whole /dev; standalone does not, so it was
# the only one actually running without GDRCopy. Conditional because the module
# is not loaded on every host (see feedback_gdrdrv_after_kernel_upgrade: DKMS
# has to be rebuilt after a kernel upgrade, and there is no udev rule), and a
# missing --device makes docker run fail outright.
build_gdr_args() {
    GDR_ARGS=()
    if [[ -c /dev/gdrdrv ]]; then
        GDR_ARGS+=(--device=/dev/gdrdrv)
    else
        echo "WARN: /dev/gdrdrv missing -- NCCL will run without GDRCopy." >&2
        echo "      Check: lsmod | grep gdrdrv ; sudo mknod /dev/gdrdrv c \$(awk '/gdrdrv/{print \$1}' /proc/devices) 0" >&2
    fi
}

# ---- paths (inside container) ----
MODEL_PATH="${MODEL_PATH:-/models/Kimi-K3}"
DRAFT_MODEL_PATH="${DRAFT_MODEL_PATH:-/models/Kimi-K3-DSpark}"

# Locally built image (see Dockerfile): sglang nightly + EFA + gdrcopy + the
# EFA/CUDA13 Mooncake wheel + the five DeepEP v2 patches. Required for PD
# disaggregation; harmless for standalone.
#
# The TAG CHANGED from kimi-k3-efa:latest deliberately. The old tag was built on
# lmsysorg/sglang:kimi-k3, which predates the v2 dispatcher and carries none of
# the patches -- and an unpatched image does not fail, it serves WRONG NUMERICS
# (see the kimi_k3.py note in the Dockerfile). Reusing the tag would have made
# that indistinguishable from a good run, so a stale image now fails to be found
# instead. require_efa_image() re-checks the patches at launch for the same
# reason.
IMAGE="${IMAGE:-kimi-k3-efa-v2:latest}"

# ---- serving profile ----
# The SGLang cookbook page for this model
# (https://docs.sglang.io/cookbook/autoregressive/Moonshotai/Kimi-K3) publishes
# low-latency / balanced / high-throughput variants of both the standalone server
# and the PD decode node. Several knobs move together, and
# the standalone and decode tables are NOT the same — notably the custom
# all-reduce, which standalone/low-latency keeps but every PD decode disables.
#
#   standalone       dcp   custom-AR   symm-mem   mamba   mem-fraction
#   low-latency      --    on          --         0.86    0.85
#   balanced         8     off         --         5.13    0.85
#   high-throughput  8     off         --         5.13    0.92
#
#   PD decode        dcp   custom-AR   symm-mem   mamba   mem-fraction
#   low-latency      --    off         ON         0.17    0.85
#   balanced         8     off         --         1.03    0.85
#   high-throughput  8     off         --         1.03    0.92
#
# --dcp-size 8 shards the decode KV/state across all 8 GPUs, freeing the memory
# that lets the mamba (KDA state) cache ratio rise — 5.13 standalone, 1.03 for
# PD decode. Without dcp the state cache must fit per-GPU: 0.86 standalone, and
# only 0.17 for PD decode, which instead leans on --enable-symm-mem.
# Only mem-fraction separates balanced from high-throughput.
#
# Upstream runs the PD prefill node identically across all three profiles, i.e.
# without dcp. On K3 that only works for low-latency, because a decode dcp of 8
# against a prefill dcp of 1 kills prefill's bootstrap thread (see the DCP_ARGS
# comment in start_prefill.sh). PREFILL_DCP_SIZE therefore mirrors the decode
# value for balanced/high-throughput as a workaround; override it to 1 to
# reproduce the upstream config and the failure.
#
# PREFILL_MAMBA_RATIO must move with PREFILL_DCP_SIZE. It was previously
# hardcoded to 0.86 in 20_launch_prefill.sh while dcp came from the profile, so
# PROFILE=balanced silently launched prefill as dcp=8 + mamba=0.86 -- a pairing
# no profile defines: dcp shards the KV cache but the state cache stays sized for
# the unsharded case. That invalidated the first balanced/high-throughput
# measurements. Prefill mirrors the decode ratio for the same reason it mirrors
# dcp; mem-fraction stays 0.85 because only the decode node raises it to 0.92.
#
# Select with PROFILE=low-latency|balanced|high-throughput.
PROFILE="${PROFILE:-low-latency}"
case "$PROFILE" in
    low-latency)
        # standalone
        STANDALONE_DCP_SIZE=1; STANDALONE_CUSTOM_AR=on
        STANDALONE_MAMBA_RATIO=0.86; STANDALONE_MEM_FRACTION=0.85
        # PD decode
        DECODE_DCP_SIZE=1; DECODE_CUSTOM_AR=off; DECODE_SYMM_MEM=on
        DECODE_MAMBA_RATIO=0.17; DECODE_MEM_FRACTION=0.85
        # PD prefill
        PREFILL_DCP_SIZE=1
        PREFILL_MAMBA_RATIO=0.86; PREFILL_MEM_FRACTION=0.85 ;;
    balanced)
        STANDALONE_DCP_SIZE=8; STANDALONE_CUSTOM_AR=off
        STANDALONE_MAMBA_RATIO=5.13; STANDALONE_MEM_FRACTION=0.85
        DECODE_DCP_SIZE=8; DECODE_CUSTOM_AR=off; DECODE_SYMM_MEM=off
        DECODE_MAMBA_RATIO=1.03; DECODE_MEM_FRACTION=0.85
        PREFILL_DCP_SIZE=8
        PREFILL_MAMBA_RATIO=1.03; PREFILL_MEM_FRACTION=0.85 ;;
    high-throughput)
        STANDALONE_DCP_SIZE=8; STANDALONE_CUSTOM_AR=off
        STANDALONE_MAMBA_RATIO=5.13; STANDALONE_MEM_FRACTION=0.92
        DECODE_DCP_SIZE=8; DECODE_CUSTOM_AR=off; DECODE_SYMM_MEM=off
        DECODE_MAMBA_RATIO=1.03; DECODE_MEM_FRACTION=0.92
        PREFILL_DCP_SIZE=8
        # high-throughput raises mem-fraction on the decode node only; the
        # prefill node still holds prefill activations, so it keeps 0.85.
        PREFILL_MAMBA_RATIO=1.03; PREFILL_MEM_FRACTION=0.85 ;;
    *) echo "unknown PROFILE '$PROFILE' (low-latency|balanced|high-throughput)" >&2; exit 1 ;;
esac

# ---- model / serving ----
TP_SIZE="${TP_SIZE:-8}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-moonshotai/Kimi-K3}"
PORT="${PORT:-30000}"

# ---- DeepEP v2 expert parallelism ----
# K3's MoE on `--moe-a2a-backend deepep_v2` + `--moe-runner-backend deep_gemm`.
# Set MOE_A2A_BACKEND=none to get the old plain-TP flashinfer_mxfp4 path back as
# a baseline; everything else in this block is then ignored.
#
# EP DEFAULTS TO INTRA-NODE, AND THAT DEFAULT MEASURES NOTHING ABOUT EFA.
# ep_size == tp_size == 8 with `--deepep-v2-mode direct` keeps the whole a2a on
# NVLink, so in a PD run the only thing crossing the wire is the Mooncake KV
# transfer. For cross-node EP set NNODES/TP_SIZE (see the multi-node block in the
# cluster section) -- `direct` is then upgraded to `hybrid` automatically below,
# because `direct` cannot reach another node and the failure is not obvious.
MOE_A2A_BACKEND="${MOE_A2A_BACKEND:-deepep_v2}"
EP_SIZE="${EP_SIZE:-$TP_SIZE}"
# Recorded before the default is applied, so the NNODES>1 upgrade below can tell
# "the caller asked for direct" from "nobody said anything".
DEEPEP_V2_MODE_SET="${DEEPEP_V2_MODE+1}"
DEEPEP_V2_MODE="${DEEPEP_V2_MODE:-direct}"
MOE_RUNNER_BACKEND="${MOE_RUNNER_BACKEND:-deep_gemm}"

# CAP = SGLANG_DEEPEP_V2_NUM_MAX_DISPATCH_TOKENS_PER_RANK (environ.py:1085,
# upstream default 128). It is an ENV VAR, not a flag, so it never shows up in
# `docker inspect .Args` -- the launchers echo it explicitly for that reason.
#
# It is v2's per-rank dispatch buffer capacity in tokens, and in a UNIFIED server
# one number has to satisfy two unrelated constraints at once
# (validate_deepep_v2_dispatch_token_budget, moe_hook.py:375-410):
#     prefill:      chunked_prefill_size / dp_size   <= CAP   -> wants it LARGE
#     decode graph: graph_bs * tokens_per_req        <= CAP   -> wants it SMALL
# and CAP also sizes two allocations that compete for the same ~39 GB:
#     ElasticBuffer      10.5 GiB per 1024 of CAP
#     decode graph pool  18.48 GB at CAP=1024, 33.43 GiB at CAP=2048
# In a unified server that forces CAP=1024 with graphs on, and the big prefill
# chunk is simply unaffordable.
#
# *** PD DISAGGREGATION IS WHY THIS FILE HAS TWO CAPS. *** Prefill and decode are
# separate processes, so they get separate values of an env var, and each side can
# take the end of the trade it actually wants. That is not a compromise -- it is
# strictly better than either unified setting.
#
# PREFILL: no decode graphs at all, so the whole capture pool becomes headroom
# and CAP can go up. 2048 is the measured graphs-off ceiling (CAP=3072 asks a
# 31.50 GiB ElasticBuffer and OOMs; 4096 asks 42.00 GiB). Prefill is near-linear
# in chunk: 512 -> 2048 was 3.94x.
#   CAVEAT, and it is a real one: 2048 was measured on a UNIFIED server with no
#   draft model and no --enable-symm-mem. The PD prefill node loads Kimi-K3-DSpark
#   and enables symm-mem, so it has less headroom and CAP=2048 is UNTESTED there.
#   If it OOMs after "Load weight", drop to 1024 -- that costs prefill throughput,
#   not correctness.
#   2026-09-03 UPDATE, and it changes the default: at the README operating point
#   (ISL 8192 / OSL 1024 / 64 prompts / c32, PD low-latency, 2x p6-b300) CAP=2048
#   costs 2.6x. Measured, 2 timed runs each after a discarded warmup:
#       2048 -> 688 tok/s, mean TTFT 31.4 s
#       4096 -> 1244             14.1 s
#       8192 -> 1767              6.6 s     <-- +158% over 2048
#      16384 -> 1790              6.5 s     (+0.9%, i.e. nothing)
#   The knee is exactly ISL: once chunk >= ISL the request stops being chunked and
#   more CAP only grows the ElasticBuffer. 16384 booted fine, so the OOM caveat
#   below is a standalone-arm concern, not a PD-prefill one -- prefill runs
#   --disable-cuda-graph and has no capture pool to compete with.
#   SO: SET THIS TO AT LEAST THE ISL YOU SERVE. 8192 is the default because the
#   documented workload is ISL 8192; it is NOT a value to sweep.
#   Caveat that stays: validated on PROFILE=low-latency (dcp=1) only. The dcp=8
#   profiles have a different memory split and are untested at 8192.
PREFILL_CAP="${PREFILL_CAP:-8192}"
# CHUNK must satisfy CHUNK/dp_size <= CAP, and dp_size is 1 here, so CHUNK==CAP.
# Note this is a big cut from the pre-DeepEP default of 16384: that is the price
# of the v2 dispatcher, not a tuning mistake -- so the plain-TP baseline keeps the
# old 16384 rather than being handicapped into a flattering comparison.
if [[ "$MOE_A2A_BACKEND" == "none" ]]; then
    PREFILL_CHUNK="${PREFILL_CHUNK:-16384}"
else
    PREFILL_CHUNK="${PREFILL_CHUNK:-$PREFILL_CAP}"
fi

# DECODE: graphs ON -- they are worth far more than the chunk they cost (measured
# 8K/1K conc 16: 3.27x end-to-end, ITL p50 43.54 vs 255.24 ms; break-even is ~52
# output tokens).
#
# *** 1024 DOES NOT BOOT ON A PD DECODE NODE. *** It was the default here for a
# while on the strength of a UNIFIED-server measurement (capture pool 18.48 GB +
# ElasticBuffer 10.5 <= 38.99 GB free). The PD decode node also loads
# Kimi-K3-DSpark and enables symm-mem, so the same arithmetic does not hold: it
# dies in capture with `Tried to allocate 10.50 GiB ... 22.90 GiB allocated in
# private pools` (measured 2026-09-03, PROFILE=low-latency, and again worked
# around by hand during the 2P2D bring-up on 2026-09-04). 512 is the largest
# value observed to boot there, so it is the default.
#
# Smaller is also FASTER, and on K3 that is now measured, not inherited from
# DSV4: at ISL 8192 / OSL 1024 / c32 with max_running_requests pinned to 16 in
# both arms, CAP=128 beat CAP=512 by +6.3% output tok/s / -6.2% TPOT, and left
# 8.57 GB more free after capture. It is not the default anyway, because
#     max_running_requests * (SPEC_BLOCK_SIZE + 1) <= CAP
# (see DECODE_MAXRUN) makes CAP=128 imply MAXRUN <= 16, i.e. only 16 decode
# slots per node -- and at the README point that admission limit, not the a2a, is
# what caps end-to-end throughput. 512 leaves DSPARK's own choice of 48 legal
# (48 * 8 = 384 <= 512) and needs no MAXRUN. To chase the -6.2% TPOT instead of
# slots, set DECODE_CAP=128 DECODE_MAXRUN=16 DECODE_CGMAXBS=16 together.
#   Residual gap: 512 has only ever been booted with MAXRUN pinned to 16. The
#   capture pool is sized by CAP and not by how many shapes are captured
#   (measured), so MAXRUN=48 should be free, but it is inference, not a reading.
DECODE_CAP="${DECODE_CAP:-512}"
# Decode still needs a chunked-prefill size, even though it does no real prefill.
# 2026-09-03: the "is the prefill-budget check gated on --disaggregation-mode"
# question is now ANSWERED -- moe_hook.py:376 is `if view.disaggregation_mode !=
# "decode":`, so a decode node's chunk is NOT checked against CAP and this may be
# set freely. It still defaults to CAP so the two never drift by accident, but
# WHEN COMPARING TWO DECODE CAPs, PIN THIS TO THE SAME VALUE IN BOTH ARMS --
# otherwise the arms differ on two axes and the chunk (inert here) gets the credit.
DECODE_CHUNK="${DECODE_CHUNK:-$DECODE_CAP}"
# THE BOUND THAT ACTUALLY DECIDES WHETHER A SMALL DECODE CAP WORKS. Two separate
# checks read the same CAP, and only the first one is a boot check:
#   boot    moe_hook.py:400-413    graph_bs * tokens_per_req <= CAP
#   RUNTIME deepep_v2.py:257       tokens_this_forward > CAP -> raise ValueError
# The runtime one is per-forward and it does NOT degrade to eager -- it kills the
# request. decode.py:2609 bounds the PD decode batch at
# `min(req_to_token_pool.size, max_running_requests)` (the +extra_slots+1 in
# pool_configurator.py:940 grows the POOL, not the batch), so the safe rule is
#     max_running_requests * (SPEC_BLOCK_SIZE + 1) <= DECODE_CAP
# and it implies the boot check, because moe_hook.py clamps graph_bs by
# max_running_requests // attn_dp_size anyway.
# Empty = let DSPARK pick, and DSPARK picks 48 (speculative_hook.py:506-514).
# 48 * 8 = 384 <= 1024, which is exactly why the default CAP needs no MAXRUN --
# and why CAP=128 cannot work without one (384 > 128 would raise on step one).
DECODE_MAXRUN="${DECODE_MAXRUN:-}"
# Caps the captured decode batch-size list (--cuda-graph-max-bs-decode; the old
# --cuda-graph-max-bs is a deprecated alias). It does NOT save memory -- the
# capture pool is sized by CAP, not by how many shapes are captured (measured:
# CAP=2048 with bs=[8,16,24] still took 33.43 GiB and OOMed, while CAP=1024 with
# all 13 shapes took 18.48 GB and started). What it does is (a) cut the ~62 s
# capture, and (b) satisfy `graph_bs * tokens_per_req <= CAP`, which is what makes
# a small DECODE_CAP legal. Empty = sglang's default list.
# Keep it ABOVE the achievable decode batch or large batches silently fall out of
# the graph and back to a ~255 ms step -- i.e. set it >= DECODE_MAXRUN. Setting it
# BELOW DECODE_MAXRUN is the one combination that is silently slow rather than
# loud: the batch is legal for the a2a but uncaptured.
DECODE_CGMAXBS="${DECODE_CGMAXBS:-}"

# STANDALONE (unified server) is the arm that has to satisfy BOTH constraints
# with one number, so it gets the compromise: 1024 with graphs on. That is not a
# tuning preference, it is the ceiling -- 18.48 GB of capture + a 10.5 GiB
# ElasticBuffer just fits the 38.99 GB available, and 1536 would need
# ~27.7 + 15.75 = 43 GB. Keep it here as the PD-vs-unified baseline; the point of
# the PD split above is that neither side has to accept this.
STANDALONE_CAP="${STANDALONE_CAP:-1024}"
if [[ "$MOE_A2A_BACKEND" == "none" ]]; then
    STANDALONE_CHUNK="${STANDALONE_CHUNK:-16384}"
else
    STANDALONE_CHUNK="${STANDALONE_CHUNK:-$STANDALONE_CAP}"
fi

# ---- NCCL GIN backend (DeepEP v2's transport) ----
# sgl-deep-ep asserts ginType != NONE even for a single-node `direct` run
# (csrc/kernels/backend/nccl.cu:87), so a GIN backend must initialize even though
# nothing crosses the wire. Without --device=/dev/infiniband NCCL sees no network
# at all and reports NONE.
#
# The two backends this repo has run on:
#   3 = GDAKI     InfiniBand/DOCA. The earlier B300-KR box had ONLY this (2x
#                 ConnectX-7 MT2910, link_layer InfiniBand, no EFA device at
#                 all). Its two HCAs sat on two IB planes with NO path between
#                 them (ibv_rc_pingpong across them failed with "transport retry
#                 counter exceeded"), which is why NCCL_IB_HCA must pin exactly
#                 one plane.
#   5 = EFA_GDA   AWS EFA. Confirmed working on p5en (gen-2 EFA, 0xEFA2).
#
# PREFER 5 WHEREVER EFA RAILS EXIST. Type 3 routes every scale-out byte through
# a CPU proxy thread; type 5 has the GPU write the WQE and ring the doorbell
# itself (reference_efa_gda_mechanism), and on b300 the 16 EFA rails carry 100
# GB/s per GPU while the 2 ConnectX-7 planes are a side channel. Using type 3 on
# this box means the EFA fabric is idle.
#
# TYPE 5 NEEDS THREE THINGS TOGETHER, and omitting any one of them looks like a
# different bug. This ordering used to be reversed here on the strength of a test
# that had only the first of the three:
#
#  1. NCCL >= 2.31. With the base image's 2.30.7 the plugin loads and is even
#     assigned ("Loaded gin plugin Libfabric_GDAKI (v14)", "Assigned plugin
#     Libfabric_GDAKI type 5 to comm") and then device-comm setup rejects it:
#         gin/gin_host.cc:247 (ncclGinDevCommSetup) NCCL WARN
#           Cannot get backend version for invalid GIN type 5
#         RuntimeError: NCCL exception (csrc/kernels/backend/nccl.cu:188): 3
#     This is an IMAGE property, not a knob: prior to 2.31 DeepEP demands an
#     exact compile/runtime NCCL match, so upgrading NCCL inside a running
#     container just trades that error for the nccl.cu:184 assert (measured
#     2026-09-04, and EP_SUPPRESS_NCCL_CHECK=1 does not cover it). The Dockerfile
#     now pins nvidia-nccl-cu13 2.31.2 BEFORE the DeepEP build for this reason.
#  2. NCCL_SYM_GIN_KERNELS_ENABLE=0, always paired with NCCL_GIN_TYPE=5.
#     NCCL's symmetric-memory GIN kernels require strong signals, which the
#     libfabric GIN plugin does not implement, so leaving them enabled crashes.
#     Both or neither -- this is the pair the EFA microbenchmark kit ships
#     (ep-benchmarks-efa/deepep-v2-efa-official/run_test_ep.sh).
#  3. NCCL_IB_HCA=rdmap on a MIXED device list. b300 shows 16 rdmap* EFA +
#     2 ibp* ConnectX-7; without the prefix pin NCCL builds only 2 GIN GDAKI
#     NICs off the ibp* pair and a rank dies at nccl.cu:185. The success line is
#         NET/Libfabric_GDAKI : GPU Direct RDMA Enabled for HCA 0..7 'rdmap*'
#     On a PURE-rdmap box (p5en) the pin is unnecessary and is not set, because
#     there it would only be a chance to get the prefix wrong.
#
# RESIDUAL, be honest about it: the 2026-09-03 type-5 attempt on this box died at
# DECODE CUDA GRAPH CAPTURE with
#   RuntimeError: NCCL exception (csrc/kernels/backend/nccl.cu:108): 5
#     (GIN strong signals are required, but the GIN plugin does not support them.)
#   ... decode_cuda_graph_runner.py:491 in __init__
# That run had none of (1) or (2). If it still fires with all three in place, the
# escape hatches in order of preference are GIN_TYPE=3 on the DECODE side only
# (build_deepep_envs feeds both sides, so pass it per launcher), then
# DECODE_CUDA_GRAPH=0. Do NOT go chasing memory: sglang wraps this as "Capture
# cuda graph failed" and appends its generic OOM "Possible solutions" list
# (mem-fraction / cuda-graph-max-bs-decode / disable the decode graph), all of
# which are irrelevant here.
#
# NCCL_DEBUG_SUBSYS=GIN prints the backend that was actually selected -- check it
# rather than assuming, because a wrong type is a performance bug, not an error.
detect_gin() {
    if [[ -n "${GIN_TYPE:-}" ]]; then
        echo "$GIN_TYPE"; return
    fi
    local devs=""
    [[ -d /sys/class/infiniband ]] && devs="$(ls /sys/class/infiniband 2>/dev/null)"
    if grep -q '^rdmap\|^efa' <<<"$devs"; then
        echo 5
    elif grep -q '^ibp\|^mlx' <<<"$devs"; then
        # IB-only box (the old B300-KR pair). GDAKI is the only backend there.
        echo 3
    else
        # No RDMA device visible at all. DeepEP v2 will abort on the GIN assert,
        # so say why now instead of letting it look like a v2 bug. Note the host
        # is what matters here: inside a container this also fires when
        # --device=/dev/infiniband was forgotten.
        echo "WARN: no EFA or IB device under /sys/class/infiniband -- DeepEP v2" >&2
        echo "      will abort on the ginType != NONE assert. Guessing type 5." >&2
        echo "      Override with GIN_TYPE=, or MOE_A2A_BACKEND=none to skip EP." >&2
        echo 5
    fi
}
# One-plane pin for GIN type 3; harmless otherwise. Auto-picks the first
# PORT_ACTIVE IB device rather than hardcoding p6-b300's ibp198s0f0, because the
# two planes have no path between them and NCCL must not straddle them.
pick_ib_hca() {
    local d st
    for d in /sys/class/infiniband/ibp* /sys/class/infiniband/mlx*; do
        [[ -e "$d" ]] || continue
        st="$(cat "$d"/ports/1/state 2>/dev/null || true)"
        [[ "$st" == *ACTIVE* ]] && { basename "$d"; return; }
    done
    echo ibp198s0f0   # p6-b300's first plane; also the "nothing found" fallback
}
IB_HCA="${IB_HCA:-$(pick_ib_hca)}"

# For GIN type 5: the NCCL_IB_HCA value that keeps NCCL on the EFA rails, or
# empty if no pin is needed. Only a MIXED device list needs one -- see reason (3)
# in the GIN block. `rdmap` is a PREFIX, not a device: NCCL matches it against
# every device name, so all 16 rails are included with one word.
pick_efa_hca() {
    local devs=""
    [[ -d /sys/class/infiniband ]] && devs="$(ls /sys/class/infiniband 2>/dev/null)"
    grep -q '^rdmap' <<<"$devs" || return 0          # no EFA rails: nothing to pin
    grep -qv '^rdmap' <<<"$devs" || return 0         # pure-rdmap box: pin unneeded
    echo rdmap
}
EFA_HCA="${EFA_HCA-$(pick_efa_hca)}"

# ---- cluster ----
# Primary ENA interface (the other 16 enpXX are EFA-only rails).
PRIMARY_IFACE="${PRIMARY_IFACE:-enp71s0}"
# These are re-assigned on every instance restart -- re-check with
# `ssh B300-N hostname -I` before a PD run, or bootstrap silently times out.
B300_1_IP="${B300_1_IP:-172.31.30.164}"  # B300-1 / i-04f949e0b7779c0a6 (2026-09-04 CB)
B300_2_IP="${B300_2_IP:-172.31.28.101}"  # B300-2 / i-0b5b67d579231b60b (2026-09-04 CB)
B300_3_IP="${B300_3_IP:-172.31.30.41}"   # B300-3 / i-0f58d1d2ed885a1d4 (2026-09-04 CB)
B300_4_IP="${B300_4_IP:-172.31.21.43}"   # B300-4 / i-0f5aef3e6a9f5f43e (2026-09-04 CB)

# PD disaggregation.
#
# PREFILL_IPS / DECODE_IPS are space-separated LISTS -- one entry per node on that
# side -- and are the only thing that decides the topology. 20_/21_ are already
# node-agnostic (each just serves one TP=8 instance on whatever host it runs on,
# and neither references the other side's address: prefill only publishes
# BOOTSTRAP_PORT, and decode learns the prefill address from the bootstrap info
# the router attaches to each request). So scaling 1P1D -> 2P2D is entirely a
# router-side change plus running 20_/21_ on more hosts.
#
# PREFILL_IP / DECODE_IP are kept as the single-node names so an existing
# `PREFILL_IP=... bash 22_launch_router.sh` still means 1P1D.
PREFILL_IP="${PREFILL_IP:-$B300_1_IP}"
DECODE_IP="${DECODE_IP:-$B300_2_IP}"
PREFILL_IPS="${PREFILL_IPS:-$PREFILL_IP}"
DECODE_IPS="${DECODE_IPS:-$DECODE_IP}"
# Same port on every node of a side: each node is a separate host, so there is no
# collision, and a per-node port would have to be threaded into 20_/21_ too.
BOOTSTRAP_PORT="${BOOTSTRAP_PORT:-8998}"
ROUTER_PORT="${ROUTER_PORT:-8080}"

# ---- multi-node instances (one sglang instance spanning >1 host) ----
# ORTHOGONAL TO PD. PREFILL_IPS/DECODE_IPS above scale the NUMBER of independent
# instances; these three scale ONE instance across hosts, which is the only way to
# get EP > 8 on 8-GPU boxes. Both can be used at once: 1P1D where each side is a
# 2-node TP=16 instance uses all four B300s.
#
# WHY IT MATTERS AND WHAT WAS WRONG BEFORE. With the default TP_SIZE=8 /
# EP_SIZE=8 / DEEPEP_V2_MODE=direct, every expert-parallel byte stays on NVLink:
# `direct` is the intra-node path and 8 ranks fit in one box, so DeepEP v2 never
# touches EFA, GIN, or the network at all -- the a2a is real but it is NVLink a2a.
# The 2P2D run on 2026-09-04 was exactly this: it validated PD and the Mooncake
# KV transfer and told us NOTHING about the EFA EP path, while the log said
# `ep=8`. A cross-node EP measurement needs all three of
#     NNODES=2  TP_SIZE=16  EP_SIZE=16  DEEPEP_V2_MODE=hybrid
# and `hybrid` is the mode that lets the kernels use both NVLink and the fabric.
#
# NODE_RANK is PER HOST -- 0 on the host named by DIST_INIT_ADDR, then 1, 2...
# Rank 0 is also the only node that serves HTTP, so that is the host the router
# (or a client) talks to.
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
# Deliberately unset by default, and validated rather than defaulted to B300_1_IP:
# a silently-wrong rendezvous address does not fail, it HANGS for --dist-timeout
# (7200 s here), which is the single most expensive way to get this wrong.
DIST_INIT_ADDR="${DIST_INIT_ADDR:-}"
DIST_INIT_PORT="${DIST_INIT_PORT:-29500}"

# Fills MULTINODE_ARGS with the sglang flags for a multi-host instance. Empty at
# NNODES=1, so single-node call sites are unaffected.
build_multinode_args() {
    MULTINODE_ARGS=()
    if (( NNODES <= 1 )); then
        return
    fi
    if [[ -z "$DIST_INIT_ADDR" ]]; then
        echo "ERROR: NNODES=$NNODES needs DIST_INIT_ADDR=<rank-0 host IP>." >&2
        echo "       e.g. DIST_INIT_ADDR=\$B300_1_IP" >&2
        return 1
    fi
    if (( TP_SIZE % NNODES != 0 )); then
        echo "ERROR: TP_SIZE=$TP_SIZE is not divisible by NNODES=$NNODES." >&2
        return 1
    fi
    # 8 GPUs per box. A TP that is not exactly NNODES*8 leaves GPUs idle, which is
    # legal but is never what these runs mean, so say so loudly rather than
    # producing a quietly half-sized measurement.
    if (( TP_SIZE != NNODES * 8 )); then
        echo "WARN: TP_SIZE=$TP_SIZE over $NNODES nodes = $((TP_SIZE / NNODES)) GPUs/node," \
             "not 8. $(( NNODES * 8 - TP_SIZE )) GPUs will sit idle." >&2
    fi
    MULTINODE_ARGS=(
        --nnodes "$NNODES"
        --node-rank "$NODE_RANK"
        --dist-init-addr "${DIST_INIT_ADDR}:${DIST_INIT_PORT}"
    )
}

# `direct` is the intra-node a2a path; on a multi-node instance it cannot reach
# the other node's experts. Upgrade it here rather than making every caller
# remember, but only when the caller left the default alone -- an explicit
# DEEPEP_V2_MODE is honoured so the broken combination stays testable.
if (( NNODES > 1 )) && [[ "$MOE_A2A_BACKEND" != "none" \
      && "$DEEPEP_V2_MODE" == "direct" && -z "$DEEPEP_V2_MODE_SET" ]]; then
    echo "NNODES=$NNODES: deepep_v2 mode direct -> hybrid (direct is intra-node only)" >&2
    DEEPEP_V2_MODE=hybrid
fi

# Custom all-reduce is an NVLink/P2P path and does not span hosts. PROFILE
# low-latency turns it on for the standalone server, which is right on one node
# and wrong on two, so force it off here rather than making the caller notice.
# (Every PD profile already has it off on both sides.)
if (( NNODES > 1 )) && [[ "${STANDALONE_CUSTOM_AR:-off}" == "on" ]]; then
    echo "NNODES=$NNODES: custom all-reduce off (it is an intra-node P2P path)" >&2
    STANDALONE_CUSTOM_AR=off
fi

# ---- runtime env applied inside the container ----
setup_runtime_env() {
    export PYTHONUNBUFFERED=1
    export LD_LIBRARY_PATH="/usr/local/lib/python3.12/dist-packages/torch/lib:/usr/local/cuda/lib64:/opt/amazon/efa/lib:/usr/local/lib:${LD_LIBRARY_PATH:-}"

    # EFA / NCCL. p6-b300 exposes 18 EFA HCAs; exclude loopback + docker bridge
    # so NCCL bootstrap picks the ENA interface rather than 172.17.x.
    export FI_PROVIDER=efa
    export FI_EFA_USE_DEVICE_RDMA=1
    export NCCL_SOCKET_IFNAME="^lo,docker"
    export GLOO_SOCKET_IFNAME="${PRIMARY_IFACE}"
    export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

    # Mooncake KV transfer over EFA (PD disagg only; harmless otherwise).
    export MOONCAKE_PROTOCOL="${MOONCAKE_PROTOCOL:-efa}"

    # Bounds Mooncake's MR-registration thread fan-out. K3 registers 1376
    # buffers, and unbounded this was 138.7 s of startup at a 138-thread peak.
    # FEWER threads is faster, because registration serializes on the EFA
    # provider's per-domain lock: 128 -> 130.1 s, 32 -> 51.0 s, 8 -> 20.7 s,
    # 4 -> 24.8 s. Re-sweep on a different instance type; the optimum tracks
    # core count / ranks, not a universal constant.
    export MC_MAX_CONCURRENT_REG_MR="${MC_MAX_CONCURRENT_REG_MR:-8}"

    # DeepEP v2's ElasticBuffer is allocated LAST, after weights + KV pool +
    # graph capture, so it is the allocation that meets a fragmented heap, and
    # expandable_segments is what lets it fit.
    #
    # BUT IT BREAKS MOONCAKE PD. Measured 2026-09-03 on 2x p6-b300: with
    # expandable_segments:True every single KV block transfer fails
    #
    #   efa_context.cpp:1175] fi_read/fi_write failed: Invalid argument
    #     (source=0x1bd24dcc000, len=147456, dest=0x1be4f059c00, rkey=59)
    #
    # 5168 of them in one instant = 152 transfers x K3's 34 layers
    # (147456 B x 24 full-MLA + 32768 B x 10 KDA), i.e. a 100% failure rate, and
    # prefill then reports the misleading "Decode instance could be dead, remote
    # mooncake session ... is not alive" because one failure blacklists the
    # session. Registration is NOT the problem -- fi_mr_regattr failures are 0
    # and all 16 rails come up.
    #
    # It is expandable_segments, not EFA: transfer_engine_bench at the SAME
    # 147456 B block between these two nodes does 105.43 GB/s over protocol=efa.
    # The tell is the address range -- 0x1bd/0x1be... are cuMem VMM addresses.
    # expandable_segments backs a tensor with a growable VA reservation whose
    # physical mapping is remapped as it grows, so the dmabuf-derived MR
    # registered at startup no longer describes the memory at transfer time and
    # libfabric rejects the RMA with EINVAL. This is also why PD worked on
    # 2026-08-14 and broke now: the var was added for DeepEP v2, after that run.
    #
    # So: only PD sets TRANSFER_BACKEND (10_launch_standalone.sh does not), and
    # PD is exactly the case that must not have it. Standalone keeps it.
    if [[ -n "${TRANSFER_BACKEND:-}" && "${TRANSFER_BACKEND}" != "none" ]]; then
        export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:False}"
    else
        export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
    fi

    # NIXL alternative to Mooncake (TRANSFER_BACKEND=nixl). The image ships
    # libplugin_LIBFABRIC.so, which is the EFA-capable NIXL backend; SGLang
    # selects it via this env (srt/disaggregation/nixl/conn.py:420).
    if [[ "${TRANSFER_BACKEND:-mooncake}" == "nixl" ]]; then
        export SGLANG_DISAGGREGATION_NIXL_BACKEND="${SGLANG_DISAGGREGATION_NIXL_BACKEND:-LIBFABRIC}"
    fi

    # 96 shards / 1.5 TB of weights: give the loader plenty of time.
    export SGLANG_LOAD_TIMEOUT="${SGLANG_LOAD_TIMEOUT:-7200}"
}

# Echo the CAP a RUNNING container was actually given, or "unknown".
#
# This has to come from .Config.Env, not from the profile variables above and not
# from the server's own `server_args=` line: CAP is an env var, not a flag, so it
# is absent from server_args entirely -- which is exactly why 93_matrix.sh's
# assert_arg() cannot check it and needs assert_env() instead. Reading the
# container is the only way to learn what the server really got rather than what
# a script meant to pass.
# Under PD the two sides run on DIFFERENT hosts and there is no ssh between them
# (tested: publickey denied both ways), so a bench running on the prefill host can
# only inspect the prefill container -- the decode CAP comes back "unknown" and the
# filename loses an axis. $2 is an optional asserted value for that case (pass
# CAP_DECODE=). It goes into the filename bare, because a filename is a join key
# and decorating it would split one arm across two names; the provenance is
# recorded in the .log header instead, where read_cap_src() marks it "(asserted)".
# One env var out of a RUNNING container's config. Everything a benchmark tag
# needs is passed with -e and never appears in the server's `server_args=` line,
# so this is the only honest source: reading this shell's own $EP_SIZE would label
# a TP=16 server "ep=8" whenever 91_bench.sh is invoked without the same
# TP_SIZE= that launched it, which is a mislabelled result, not a cosmetic bug.
read_cenv() {
    local name="$1" var="$2" fallback="${3:-}" v
    v=$(docker inspect -f \
        '{{range .Config.Env}}{{println .}}{{end}}' "$name" 2>/dev/null \
        | grep "^${var}=" | head -1 | cut -d= -f2-)
    echo "${v:-${fallback:-unknown}}"
}

read_cap() {
    local name="$1" asserted="${2:-}"
    read_cenv "$name" SGLANG_DEEPEP_V2_NUM_MAX_DISPATCH_TOKENS_PER_RANK "$asserted"
}

# Same value, annotated with where it came from. For the log header only.
read_cap_src() {
    local name="$1" asserted="${2:-}" v
    v=$(docker inspect -f \
        '{{range .Config.Env}}{{println .}}{{end}}' "$name" 2>/dev/null \
        | grep '^SGLANG_DEEPEP_V2_NUM_MAX_DISPATCH_TOKENS_PER_RANK=' \
        | head -1 | cut -d= -f2)
    if [[ -n "$v" ]]; then
        echo "$v"
    elif [[ -n "$asserted" ]]; then
        echo "${asserted}(asserted: container not on this host)"
    else
        echo "unknown(container not on this host and no CAP_* override)"
    fi
}

# Assert the image actually has an EFA-capable Mooncake before a 10-minute load.
# The upstream lmsysorg/sglang:kimi-k3 image ships a pip mooncake wheel built
# WITHOUT EFA, which does not fail at startup: it passes SGLang's PD warmup and
# then kills the first real request. Bind-mounting the host's EFA libs does not
# help either, since the missing support is compiled into the wheel.
require_efa_image() {
    local img="$1"
    if ! docker image inspect "$img" >/dev/null 2>&1; then
        echo "ERROR: image '$img' not found. Build it first:" >&2
        echo "  docker build -t kimi-k3-efa-v2:latest -f Dockerfile ." >&2
        exit 1
    fi
    # All the mooncake checks in ONE container start, each with its own message.
    #
    # THREE THINGS HERE WERE WRONG AND EVERY ONE OF THEM MADE THIS GATE LIE.
    # Measured against the mooncake-transfer-engine-efa-cuda13 wheel (0.3.13.post1)
    # on 2026-09-03, which is what the Dockerfile installs now:
    #
    #  1. The extension is `engine.so`, NOT `engine.cpython-<abi>.so`. That name
    #     came from the old cmake source build. The glob matched nothing, so
    #     `strings` errored on a literal asterisk and BOTH checks below failed --
    #     which is how a known-good image got rejected as "no EFA support".
    #  2. `grep -ciE efa` is vacuous: "efa" is a substring of "d-efa-ult". It
    #     returns 389 on the NON-EFA wheel, i.e. it passes on exactly the build
    #     this check exists to reject. The real discriminator is that the EFA
    #     build LINKS libfabric and carries its entry points; the non-EFA build
    #     has zero of both (ibverbs symbols are in both, so they prove nothing).
    #  3. `bindCudaContextIfNeeded` was the FORK's function name. The upstream
    #     wheel that carries the same fix does not use it. What it does carry is
    #     the driver-API calls the fix is made of, including the distinctive
    #     "cuCtxSetCurrent (restore)" log string from its restore path.
    #
    # So: grep for what the wheel actually contains, not for what our fork was
    # called. Do not "simplify" any of these back to a name or an `efa` substring.
    local rc=0 out
    out=$(docker run --rm --entrypoint bash "$img" -c '
        set -eu
        MC=$(python3 -c "import mooncake,os;print(os.path.dirname(mooncake.__file__))")
        E="$MC/engine.so"
        test -f "$E"                                              || { echo "NO_ENGINE_SO $MC"; exit 1; }
        ldd "$E" | grep -q libfabric                              || { echo "NO_LIBFABRIC";    exit 1; }
        test "$(strings "$E" | grep -cE "^fi_(getinfo|mr_reg)" || true)" -ge 2 \
                                                                  || { echo "NO_FI_SYMS";      exit 1; }
        strings "$E" | grep -q cuDevicePrimaryCtxRetain           || { echo "NO_CTXFIX";       exit 1; }
        strings "$E" | grep -q "cuCtxSetCurrent (restore)"        || { echo "NO_CTXFIX";       exit 1; }
        strings "$E" | grep -q MC_MAX_CONCURRENT_REG_MR           || { echo "NO_REGCAP";       exit 1; }
        echo OK
    ' 2>&1) || rc=$?
    case "$out" in
        *OK*) : ;;
        *NO_ENGINE_SO*)
            echo "ERROR: '$img' has no mooncake engine.so; the wheel layout changed." >&2
            echo "       $out" >&2; exit 1 ;;
        *NO_LIBFABRIC*)
            echo "ERROR: mooncake in '$img' does not link libfabric -- this is the" >&2
            echo "       NON-EFA wheel. Rebuild with MOONCAKE_PKG naming an" >&2
            echo "       ...-efa-... distribution." >&2; exit 1 ;;
        *NO_FI_SYMS*)
            echo "ERROR: mooncake in '$img' links libfabric but carries no libfabric" >&2
            echo "       entry points. Suspect a stripped or partial build." >&2; exit 1 ;;
        *NO_CTXFIX*)
            echo "ERROR: mooncake in '$img' lacks the GPU-MR CUDA-context fix." >&2
            echo "       Without it GPU KV registration fails 'Operation not" >&2
            echo "       supported', startup and PD warmup still pass, and the" >&2
            echo "       FIRST REAL REQUEST dies. Rebuild from Dockerfile with a" >&2
            echo "       current mooncake-transfer-engine-efa-cuda13 wheel." >&2; exit 1 ;;
        *NO_REGCAP*)
            echo "ERROR: mooncake in '$img' has no MC_MAX_CONCURRENT_REG_MR, so the" >&2
            echo "       MR-registration fan-out is unbounded (138 s of startup)." >&2; exit 1 ;;
        *)
            echo "ERROR: mooncake EFA check on '$img' failed (rc=$rc):" >&2
            echo "       $out" >&2; exit 1 ;;
    esac

    # Same class of trap, and the worst one: an image WITHOUT the kimi_k3.py
    # patch runs deepep_v2 happily and returns wrong logits, because the MoE
    # region keeps its DP-gather / TP-reduce on top of the a2a the v2 dispatcher
    # already did. There is no error to notice. So refuse to launch on an image
    # that cannot prove the patch is in it. Skipped when EP is off, since the
    # patches are only reachable through the v2 path.
    if [[ "${MOE_A2A_BACKEND:-deepep_v2}" == "deepep_v2" ]]; then
        if ! docker run --rm --entrypoint bash "$img" -c \
            'SGL=$(python3 -c "import sglang,os;print(os.path.dirname(sglang.__file__))");
             grep -q KimiK3ForConditionalGeneration "$SGL/srt/arg_groups/moe_hook.py" &&
             grep -q is_deepep_v2 "$SGL/srt/models/kimi_k3.py"' 2>/dev/null; then
            echo "ERROR: '$img' is missing the DeepEP v2 patches for K3." >&2
            echo "       This image would SERVE WRONG NUMERICS, not fail." >&2
            echo "       Rebuild: docker build -t kimi-k3-efa-v2:latest -f Dockerfile ." >&2
            echo "       (or set MOE_A2A_BACKEND=none to run the plain-TP baseline)" >&2
            exit 1
        fi
    fi
}

# Fills DEEPEP_ENVS with the docker -e flags DeepEP v2 needs in the CONTAINER
# environment. $1 = the CAP for THIS side; prefill and decode pass different
# values, which is the whole point of doing this under PD (see the CAP block).
# Comes back empty when MOE_A2A_BACKEND=none.
#
# Only the env side lives here. The server FLAGS are assembled in the start_*.sh
# scripts alongside every other flag group, from the plain vars forwarded with -e.
build_deepep_envs() {
    local cap="$1"
    DEEPEP_ENVS=()
    if [[ "$MOE_A2A_BACKEND" == "none" ]]; then
        return
    fi
    local gin; gin="$(detect_gin)"
    DEEPEP_ENVS=(
        -e SGLANG_DEEPEP_V2_NUM_MAX_DISPATCH_TOKENS_PER_RANK="$cap"
        -e NCCL_GIN_TYPE="$gin"
    )
    local hca=""
    if [[ "$gin" == "3" ]]; then
        # One-plane pin. The IB-only box's two planes have no path between them,
        # so NCCL must not straddle them. Never a bare device name on EFA: that
        # would hide 15 of the 16 rails.
        hca="$IB_HCA"; DEEPEP_ENVS+=(-e NCCL_IB_HCA="$IB_HCA")
    else
        # Type 5 is a PAIR -- see reason (2) in the GIN block. Setting
        # NCCL_GIN_TYPE=5 without this crashes in NCCL's symmetric GIN kernels.
        DEEPEP_ENVS+=(-e NCCL_SYM_GIN_KERNELS_ENABLE=0)
        # Prefix pin, and only on a mixed device list -- reason (3).
        if [[ -n "$EFA_HCA" ]]; then
            hca="$EFA_HCA"; DEEPEP_ENVS+=(-e NCCL_IB_HCA="$EFA_HCA")
        fi
    fi
    echo "DeepEP v2: ep=$EP_SIZE mode=$DEEPEP_V2_MODE runner=$MOE_RUNNER_BACKEND cap=$cap gin=$gin${hca:+ hca=$hca}" >&2
}
