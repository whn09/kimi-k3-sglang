# Kimi-K3 on AWS p6-b300 (8x B300, sm_103), with EFA-capable PD disaggregation.
#
# Layers on top of the lmsys Kimi-K3 image:
#   - EFA:      aws-efa-installer (libfabric 2.x + EFA provider + aws-ofi-nccl)
#   - GDRCopy:  GPU memory registration for EFA VRAM transfer
#   - Mooncake: kvcache-ai @ main, built -DUSE_EFA=ON (KV transfer for PD-disagg)
#
# NVSHMEM/DeepEP are deliberately NOT built: K3's MoE runs on the
# flashinfer_mxfp4 runner with plain TP here, so there is no expert-parallel
# all-to-all to accelerate.
#
# Why this image instead of bind-mounting the host's EFA stack: the base image's
# `mooncake` is a pip wheel built WITHOUT EFA support (`strings engine.*.so |
# grep -c efa` == 0), so no amount of library mounting gives PD-disagg an EFA KV
# path. It also ships an rdma-core older than the host's, so host libfabric 2.4
# mounted in fails with "libefa.so.1: version `EFA_1.4' not found".
#
# Build (no GPU needed):
#   docker build -t kimi-k3-efa:latest -f Dockerfile .
ARG SGLANG_BASE=lmsysorg/sglang:kimi-k3
FROM ${SGLANG_BASE}

USER root
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
        git cmake build-essential wget curl \
        libgflags-dev autoconf automake libtool ninja-build
# NOTE: do NOT purge /var/lib/apt/lists here — the EFA installer below runs its
# own `apt-get install` (pciutils, tcl, environment-modules, ...) and needs the
# package index. It is purged at the end of the EFA step instead.

# ---- GDRCopy (GPU memreg for EFA) ----
ARG GDRCOPY_VERSION=2.5.2
RUN cd /tmp && \
    wget -q https://github.com/NVIDIA/gdrcopy/archive/refs/tags/v${GDRCOPY_VERSION}.tar.gz && \
    tar xf v${GDRCOPY_VERSION}.tar.gz && cd gdrcopy-${GDRCOPY_VERSION} && \
    make -j$(nproc) lib lib_install CUDA=/usr/local/cuda PREFIX=/usr/local && \
    rm -rf /tmp/gdrcopy* /tmp/v${GDRCOPY_VERSION}.tar.gz

# ---- AWS EFA installer (libfabric + EFA provider + aws-ofi-nccl) ----
# Bundled rdma-core + --skip-rdma-core so the build does not depend on whether
# the build host has EFA devices. This also replaces the base image's older
# rdma-core, which is the actual fix for the EFA_1.4 symbol mismatch.
ARG EFA_INSTALLER_VERSION=latest
RUN cd /tmp && \
    curl -O https://efa-installer.amazonaws.com/aws-efa-installer-${EFA_INSTALLER_VERSION}.tar.gz && \
    tar xzf aws-efa-installer-${EFA_INSTALLER_VERSION}.tar.gz && \
    cd aws-efa-installer && \
    . /etc/os-release && \
    case "$VERSION_ID" in \
      24.04) DEBS_DIR=UBUNTU2404 ;; \
      22.04) DEBS_DIR=UBUNTU2204 ;; \
      *)     DEBS_DIR=UBUNTU2404 ;; \
    esac && \
    echo "Using EFA DEBS dir: $DEBS_DIR (Ubuntu $VERSION_ID)" && \
    (apt-get install -y ./DEBS/${DEBS_DIR}/x86_64/rdma-core/*.deb || true) && \
    ./efa_installer.sh -y --skip-kmod -g --no-verify --skip-rdma-core && \
    rm -rf /tmp/aws-efa-installer* /var/lib/apt/lists/*

# ---- Mooncake transfer engine (EFA + CUDA) ----
# The mooncake already in the base image is a pip wheel built WITHOUT EFA
# support, so it is replaced with a -DUSE_EFA=ON source build.
#
# Built from a branch, not upstream main, because upstream cannot register the
# GPU KV cache on this hardware: `fi_mr_regattr` fails with "Operation not
# supported" (8 failures, one per GPU), startup and SGLang's PD warmup still
# pass, and the first real request dies while blaming the peer. Cause is a
# missing CUDA context on the registering thread — libfabric's
# cuda_get_dmabuf_fd() calls cuMemGetHandleForAddressRange() without binding
# one, and the EFA provider masks the resulting CUDA_ERROR_INVALID_CONTEXT as
# ENOTSUP (`efa_mr.c`: `if (!errno) errno = ENOTSUP`). The branch below binds
# the primary context before registering; verified 1456/1456 registrations and
# 1P1D end to end on 2x p6-b300.
#
# Two things this failure is NOT, both checked, so they aren't re-chased: a
# stale main missing the FI_HMEM-in-caps change (`objdump -dC` of
# EfaContext::construct shows `movabs $0x800000003306` — bit 47 is set), and a
# VMM/mempool allocator (the failing buffers are plain cudaMalloc).
#
# MOONCAKE_REF is unpinned by default so rebuilds pick up branch updates; set it
# to a commit sha for a reproducible image. `git log --oneline -1` below records
# what an image was actually built from — check it before debugging a KV failure.
ARG MOONCAKE_REPO=https://github.com/whn09/Mooncake.git
ARG MOONCAKE_REF=fix/efa-gpu-mr-cuda-context
RUN pip uninstall -y mooncake-transfer-engine mooncake 2>/dev/null || true
RUN cd /tmp && \
    git clone --depth 1 --branch "$MOONCAKE_REF" "$MOONCAKE_REPO" Mooncake || \
      { git clone "$MOONCAKE_REPO" Mooncake && git -C Mooncake checkout "$MOONCAKE_REF"; } && \
    cd Mooncake && git log --oneline -1 && \
    bash dependencies.sh -y && \
    apt-get install -y libgflags-dev && \
    mkdir build && cd build && \
    PYBIN=$(which python3) && \
    PYINC=$(python3 -c "import sysconfig; print(sysconfig.get_path('include'))") && \
    cmake .. -DUSE_EFA=ON -DUSE_CUDA=ON -DCMAKE_BUILD_TYPE=RelWithDebInfo \
      -DPython3_EXECUTABLE="$PYBIN" \
      -DPython3_INCLUDE_DIR="$PYINC" \
      -DPython_EXECUTABLE="$PYBIN" && \
    make -j$(nproc) && \
    cp mooncake-integration/engine.cpython-*.so ../mooncake-wheel/mooncake/ && \
    cp mooncake-integration/store.cpython-*.so ../mooncake-wheel/mooncake/ 2>/dev/null || true && \
    cp mooncake-common/libasio.so ../mooncake-wheel/mooncake/ && \
    pip install ../mooncake-wheel --no-build-isolation && \
    MCDIR=$(python3 -c "import mooncake, os; print(os.path.dirname(mooncake.__file__))") && \
    cp -v mooncake-integration/engine.cpython-*.so "$MCDIR/" && \
    cp -v mooncake-integration/store.cpython-*.so "$MCDIR/" 2>/dev/null || true && \
    cp -v mooncake-common/libasio.so "$MCDIR/" && \
    rm -rf /tmp/Mooncake

# ---- EFA runtime env ----
ENV FI_PROVIDER=efa
ENV FI_EFA_USE_DEVICE_RDMA=1

# On NGC-based images the EFA installer lays down libnccl-ofi-ngc-v3, which ships
# ONLY libnccl-net-ofi.so -- NOT the default name libnccl-net.so that NCCL
# auto-loads. Without this, NCCL logs "NET/Plugin: Could not find: libnccl-net.so"
# and falls back to TCP sockets (~14 GB/s vs ~400 GB/s over EFA), which shows up
# as 3-5x slower prefill TTFT rather than as an error.
# Use the SHORT name "ofi": NCCL templates it into libnccl-net-<value>.so and
# resolves that through ldconfig, where the EFA installer registered
# /opt/amazon/ofi-nccl/lib. Do NOT use an absolute path -- NCCL 2.27.x applies
# the same templating to it, yielding a bogus doubled path -> silent TCP
# fallback. Verify: NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET should log
# "NET/OFI Selected provider is efa" -- not "NET/Socket".
ENV NCCL_NET_PLUGIN=ofi

ENV PATH="/opt/amazon/efa/bin:${PATH}"
ENV LD_LIBRARY_PATH="/opt/amazon/efa/lib:/usr/local/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"
RUN TORCH_LIB=$(python3 -c "import torch, os; print(os.path.dirname(torch.__file__) + '/lib')") && \
    printf '%s\n' "$TORCH_LIB" > /etc/ld.so.conf.d/zz-torch.conf && \
    ldconfig && \
    echo "registered torch lib dir with ldconfig: $TORCH_LIB"

# ---- build-time sanity checks ----
# libcuda.so.1 is absent at build time (no GPU), so use the CUDA stub to let the
# import of the CUDA-linked mooncake engine resolve.
RUN ln -sf /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1 && \
    LD_LIBRARY_PATH="/usr/local/cuda/lib64/stubs:${LD_LIBRARY_PATH}" \
      python3 -c "from mooncake.engine import TransferEngine; print('mooncake TransferEngine OK')" && \
    MC=$(python3 -c "import mooncake, os; print(os.path.dirname(mooncake.__file__))") && \
    echo "EFA symbols in mooncake engine:" && \
    strings "$MC"/engine.cpython-*.so | grep -ciE "efa" && \
    test "$(strings "$MC"/engine.cpython-*.so | grep -ciE 'efa')" -gt 0 \
      || (echo "FATAL: mooncake built without EFA support" && exit 1)

WORKDIR /workspace
