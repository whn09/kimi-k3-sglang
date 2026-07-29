#!/bin/bash
# Shared config for all Kimi-K3 B300 scripts. Sourced by both host launchers and
# in-container start scripts, so keep it POSIX-ish and side-effect free.

# ---- paths (host) ----
NVME="${NVME:-/opt/dlami/nvme}"
HOST_MODEL_DIR="${HOST_MODEL_DIR:-$NVME/models}"
HOST_CACHE_DIR="${HOST_CACHE_DIR:-$NVME/cache}"
SCRIPT_DIR_HOST="${SCRIPT_DIR_HOST:-/home/ubuntu/kimi-k3-aws}"

# ---- paths (inside container) ----
MODEL_PATH="${MODEL_PATH:-/models/Kimi-K3}"
DRAFT_MODEL_PATH="${DRAFT_MODEL_PATH:-/models/Kimi-K3-DSpark}"

# Locally built image (see Dockerfile): kimi-k3 base + EFA + gdrcopy + Mooncake
# built -DUSE_EFA=ON. Required for PD disaggregation; harmless for standalone.
IMAGE="${IMAGE:-kimi-k3-efa:latest}"

# ---- serving profile ----
# Upstream publishes low-latency / balanced / high-throughput variants of both
# the standalone server and the PD decode node. Several knobs move together, and
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
        PREFILL_DCP_SIZE=1 ;;
    balanced)
        STANDALONE_DCP_SIZE=8; STANDALONE_CUSTOM_AR=off
        STANDALONE_MAMBA_RATIO=5.13; STANDALONE_MEM_FRACTION=0.85
        DECODE_DCP_SIZE=8; DECODE_CUSTOM_AR=off; DECODE_SYMM_MEM=off
        DECODE_MAMBA_RATIO=1.03; DECODE_MEM_FRACTION=0.85
        PREFILL_DCP_SIZE=8 ;;
    high-throughput)
        STANDALONE_DCP_SIZE=8; STANDALONE_CUSTOM_AR=off
        STANDALONE_MAMBA_RATIO=5.13; STANDALONE_MEM_FRACTION=0.92
        DECODE_DCP_SIZE=8; DECODE_CUSTOM_AR=off; DECODE_SYMM_MEM=off
        DECODE_MAMBA_RATIO=1.03; DECODE_MEM_FRACTION=0.92
        PREFILL_DCP_SIZE=8 ;;
    *) echo "unknown PROFILE '$PROFILE' (low-latency|balanced|high-throughput)" >&2; exit 1 ;;
esac

# ---- model / serving ----
TP_SIZE="${TP_SIZE:-8}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-moonshotai/Kimi-K3}"
PORT="${PORT:-30000}"

# ---- cluster ----
# Primary ENA interface (the other 16 enpXX are EFA-only rails).
PRIMARY_IFACE="${PRIMARY_IFACE:-enp71s0}"
B300_1_IP="${B300_1_IP:-172.31.60.28}"   # P6-B300-1
B300_2_IP="${B300_2_IP:-172.31.51.133}"  # P6-B300-2

# PD disaggregation
PREFILL_IP="${PREFILL_IP:-$B300_1_IP}"
DECODE_IP="${DECODE_IP:-$B300_2_IP}"
BOOTSTRAP_PORT="${BOOTSTRAP_PORT:-8998}"
ROUTER_PORT="${ROUTER_PORT:-8080}"

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

    # NIXL alternative to Mooncake (TRANSFER_BACKEND=nixl). The image ships
    # libplugin_LIBFABRIC.so, which is the EFA-capable NIXL backend; SGLang
    # selects it via this env (srt/disaggregation/nixl/conn.py:420).
    if [[ "${TRANSFER_BACKEND:-mooncake}" == "nixl" ]]; then
        export SGLANG_DISAGGREGATION_NIXL_BACKEND="${SGLANG_DISAGGREGATION_NIXL_BACKEND:-LIBFABRIC}"
    fi

    # 96 shards / 1.5 TB of weights: give the loader plenty of time.
    export SGLANG_LOAD_TIMEOUT="${SGLANG_LOAD_TIMEOUT:-7200}"
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
        echo "  docker build -t kimi-k3-efa:latest -f Dockerfile ." >&2
        exit 1
    fi
    if ! docker run --rm --entrypoint bash "$img" -c \
        'MC=$(python3 -c "import mooncake,os;print(os.path.dirname(mooncake.__file__))");
         test "$(strings "$MC"/engine.cpython-*.so | grep -ciE efa)" -gt 0' 2>/dev/null; then
        echo "ERROR: mooncake in '$img' has no EFA support; rebuild from Dockerfile." >&2
        exit 1
    fi
    # Same class of trap: an image built from upstream main cannot register the
    # GPU KV cache (see the Mooncake section of the Dockerfile) yet still starts
    # and passes PD warmup, so catch it here rather than 10 minutes later.
    if ! docker run --rm --entrypoint bash "$img" -c \
        'MC=$(python3 -c "import mooncake,os;print(os.path.dirname(mooncake.__file__))");
         strings "$MC"/engine.cpython-*.so | grep -q bindCudaContextIfNeeded' 2>/dev/null; then
        echo "ERROR: mooncake in '$img' lacks the GPU-MR CUDA-context fix." >&2
        echo "       Rebuild from Dockerfile (MOONCAKE_REF=fix/efa-gpu-mr-cuda-context)." >&2
        exit 1
    fi
}
