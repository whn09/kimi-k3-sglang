# Kimi-K3 on AWS p6-b300 (8x B300, sm_103), with EFA-capable PD disaggregation
# and DeepEP v2 expert parallelism.
#
# Layers on top of the lmsys sglang nightly:
#   - EFA:      aws-efa-installer (libfabric 2.x + EFA provider + aws-ofi-nccl)
#   - GDRCopy:  GPU memory registration for EFA VRAM transfer
#   - Mooncake: kvcache-ai fork, built -DUSE_EFA=ON (KV transfer for PD-disagg)
#   - 5 K3/DeepEP-v2 source patches (see patches/README.md)
#
# NVSHMEM/DeepEP v1 are still deliberately NOT built, but the reason changed.
# It used to be "K3's MoE runs flashinfer_mxfp4 with plain TP, so there is no
# a2a to accelerate". K3 now runs on **DeepEP v2** (`--moe-a2a-backend
# deepep_v2`), which is not NVSHMEM-based at all: sgl-deep-ep's v2 kernels ship
# prebuilt in the sglang nightly and talk to the fabric through NCCL's GIN
# backend. So there is still nothing to compile here -- there is just an
# expert-parallel a2a now, and the patches below are what let K3 reach it.
#
# EP IS INTRA-NODE ONLY (`--deepep-v2-mode direct`, ep-size == tp-size == 8).
# Cross-node EP needs more than an image change and is out of scope; the only
# thing crossing the wire in a PD run is the Mooncake KV transfer.
#
# Why this image instead of bind-mounting the host's EFA stack: the base image's
# `mooncake` is a pip wheel built WITHOUT EFA support (`strings engine.*.so |
# grep -c efa` == 0), so no amount of library mounting gives PD-disagg an EFA KV
# path. It also ships an rdma-core older than the host's, so host libfabric 2.4
# mounted in fails with "libefa.so.1: version `EFA_1.4' not found".
#
# Build (no GPU needed):
#   docker build -t kimi-k3-efa-v2:latest -f Dockerfile .
#
# BASE IMAGE: this is the nightly that every deepep_v2-on-K3 measurement in
# results/deepep_v2_on_k3_b300.md was taken on, NOT lmsysorg/sglang:kimi-k3.
# The old kimi-k3 tag predates the v2 dispatcher, so the five patches below have
# nothing to attach to there. Pinned by digest-bearing tag on purpose: the
# patches are line-context diffs against this exact tree, and a floating
# `nightly-dev` would break them silently-looking-loudly at build time.
ARG SGLANG_BASE=lmsysorg/sglang:nightly-dev-cu13-20260901-07c8f729
FROM ${SGLANG_BASE}

USER root
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
        git cmake build-essential wget curl patch \
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

# ---- Mooncake transfer engine (EFA + CUDA 13) ----
# The mooncake in the base image is a pip wheel built WITHOUT EFA support, so it
# has to be replaced. This used to be a ~15-minute cmake build of a fork; both
# fixes that fork carried are now upstream and published in this wheel, which
# ships prebuilt with USE_EFA=ON + USE_CUDA=ON against CUDA 13:
#   - GPU KV-cache registration binds the CUDA primary context before
#     fi_mr_regattr (without it, registration fails "Operation not supported",
#     startup and PD warmup still pass, and the first real request dies).
#   - registerLocalMemoryBatch()'s thread fan-out is bounded; the launchers set
#     MC_MAX_CONCURRENT_REG_MR=8, which took K3's 1376-buffer registration from
#     138.7 s to 20.7 s (it serializes on the EFA per-domain lock, so fewer
#     threads is faster: 128 -> 130.1 s, 32 -> 51.0 s, 8 -> 20.7 s, 4 -> 24.8 s).
#
# The uninstall is BY DISCOVERED NAME, not by a guessed list. The base image
# ships the distribution `mooncake-transfer-engine-cuda13` -- not
# `mooncake-transfer-engine` and not `mooncake` -- so a hard-coded uninstall list
# is a silent no-op and BOTH wheels end up installed at once. They own the same
# `mooncake/` package directory, so pip just overwrites what overlaps and leaves
# the rest: a 161 MB libmooncake_pg.so and four pg_*.so from the non-EFA build
# survive next to the EFA build's files, and both `.libs` dirs sit side by side
# for RPATH to choose from. One package dir, two provenances. So: enumerate every
# installed mooncake* distribution, uninstall them all, then delete the package
# dir and any *.libs dirs outright before installing.
#
# Consequence of that clean-out, on purpose: the EFA wheel ships the Python shims
# ep.py / pg.py / mooncake_ep_buffer.py but NOT their compiled extensions
# (_ep*.so, libmooncake_ep_device.so, pg_*.so), which only the non-EFA wheel had.
# So `import mooncake.ep` now fails -- loudly, with "Mooncake EP was not built".
# That is correct for this kit: mooncake is the KV-transfer engine here, EP is
# DeepEP v2, and sglang's only mooncake.pg imports are gated on the torch
# distributed backend being "mooncake" (parallel_state.py:336, :2283) or on
# elastic EP, neither of which any launcher selects. Do not "restore" those files
# by skipping the uninstall -- that resurrects the mixed-provenance directory,
# where the EP device library is the one built WITHOUT EFA.
ARG MOONCAKE_PKG=mooncake-transfer-engine-efa-cuda13
RUN set -eu; \
    SP=$(python3 -c "import sysconfig; print(sysconfig.get_paths()['purelib'])"); \
    old=$(pip list --format=freeze 2>/dev/null | sed -n 's/^\(mooncake[^=]*\)==.*/\1/p' | tr '\n' ' '); \
    echo "removing pre-installed mooncake distributions: ${old:-<none>}"; \
    if [ -n "$old" ]; then pip uninstall -y $old; fi; \
    rm -rf "$SP/mooncake" "$SP"/mooncake*.libs; \
    pip install --no-cache-dir "$MOONCAKE_PKG"; \
    python3 -c "import importlib.metadata as m; print('mooncake wheel:', '$MOONCAKE_PKG', m.version('$MOONCAKE_PKG'))"; \
    n=$(pip list --format=freeze 2>/dev/null | grep -c '^mooncake' || true); \
    test "$n" -eq 1 || { echo "FATAL: $n mooncake distributions installed, expected exactly 1:"; \
                         pip list --format=freeze | grep '^mooncake'; exit 1; }

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

# ---- DeepEP v2 source patches ----
# K3 + `--moe-a2a-backend deepep_v2` needs five one-to-eight-line source fixes.
# They used to be five bind-mounted whole files under /opt/dlami/nvme/patch on
# one specific host, which meant the image alone could not reproduce a run and a
# second machine silently got the UNPATCHED behaviour. Baking them in fixes that;
# per-blocker root cause is in patches/README.md. Numerics were checked against a
# deepep(v1) baseline -- first-token top-5 logprobs are byte-identical.
#
#   moe_hook.py       KimiK3ForConditionalGeneration added to the deepep_v2
#                     model whitelist (otherwise v2 refuses to engage at all)
#   kimi_k3.py        is_deepep_v2() added to the two hand-rolled _ep_a2a lists,
#                     which listed every other EP-a2a backend but not v2.
#                     WITHOUT THIS THE MODEL IS SILENTLY WRONG, not broken: the
#                     MoE region keeps its DP-gather / TP-reduce while the v2
#                     dispatcher has already done the a2a -- exactly the hazard
#                     upstream's whitelist error message warns about.
#   fmt_layer.py      the FP4/MXFP4 quant-method gate, stricter than the
#                     deep_gemm kernels behind it
#   mr_deep_gemm.py   the v2 pre-permute's `activation == "silu"` assert, widened
#                     to accept K3's "situ" (sglang PR #37514)
#   ep_moe_kernels.py ep_scatter_from_psum missing kernel args (sglang PR #37211)
#
# Applied by TARGET PATH rather than -p<n>: the diffs were captured against
# /opt/dlami/nvme/patch/orig/*.py, so no strip level reaches the real tree.
#
# The loop is idempotent AND fail-closed, which matters more than it looks. If
# one of these lands upstream, a plain `patch` would fail and (with `|| true`)
# leave a half-patched image whose only symptom is wrong logits. So: try forward,
# else detect already-applied by a successful REVERSE dry-run and skip, else
# FAIL THE BUILD. A build break here is the intended signal to go check whether
# the corresponding PR merged and drop that diff.
COPY patches/ /opt/k3-patches/
RUN set -eu; \
    SGL=$(python3 -c "import sglang, os; print(os.path.dirname(sglang.__file__))"); \
    echo "sglang tree: $SGL"; \
    apply_one() { \
      diff_file="/opt/k3-patches/$1"; target="$SGL/$2"; \
      test -f "$target" || { echo "FATAL: patch target missing: $target"; exit 1; }; \
      if patch --forward --dry-run "$target" "$diff_file" >/dev/null 2>&1; then \
        patch --forward "$target" "$diff_file" >/dev/null && echo "  applied  $1 -> $2"; \
      elif patch --reverse --dry-run "$target" "$diff_file" >/dev/null 2>&1; then \
        echo "  SKIP     $1 -> $2 (already present in this base image)"; \
      else \
        echo "FATAL: $1 does not apply to $target and is not already applied."; \
        echo "       The base image moved. Check whether the upstream PR merged"; \
        echo "       (see patches/README.md) and re-cut the diff or delete it."; \
        exit 1; \
      fi; \
    }; \
    apply_one moe_hook.diff        srt/arg_groups/moe_hook.py; \
    apply_one kimi_k3.diff         srt/models/kimi_k3.py; \
    apply_one fmt_layer.diff       srt/layers/moe/fused_moe_triton/layer.py; \
    apply_one mr_deep_gemm.diff    srt/layers/moe/moe_runner/deep_gemm.py; \
    apply_one ep_moe_kernels.diff  kernels/ops/moe/ep_moe_kernels.py; \
    echo "all five DeepEP v2 patches accounted for"

# ---- build-time sanity checks ----
# libcuda.so.1 is absent at build time (no GPU), so use the CUDA stub to let the
# import of the CUDA-linked mooncake engine resolve.
#
# Three things this rewrite fixes, each of which made the check lie:
#
#  1. The extension is named `engine.so`, NOT `engine.cpython-<abi>.so`. That
#     naming came from the old cmake source build; the wheel does not use it, so
#     the glob matched nothing and `strings` failed on a literal asterisk.
#  2. `grep -ciE efa` matches the substring in "d-efa-ult". It returned 389 on the
#     base image's NON-EFA wheel, i.e. it would have passed on exactly the build
#     this stage exists to reject. The discriminator is that the EFA build LINKS
#     libfabric: `ldd engine.so` names libfabric.so.1 and the binary carries
#     fi_getinfo/fi_mr_reg, while the non-EFA build has zero of all of those
#     (both have ibv_reg_mr, so ibverbs proves nothing either).
#  3. Every command shared one trailing `||`, so any failure anywhere in the
#     chain -- including the import -- printed "built without EFA support". The
#     first real failure here was an import error reported as a link-config error.
#     Each check now fails with its own message.
RUN set -eu; \
    ln -sf /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1; \
    export LD_LIBRARY_PATH="/usr/local/cuda/lib64/stubs:${LD_LIBRARY_PATH}"; \
    python3 -c "from mooncake.engine import TransferEngine; print('mooncake TransferEngine OK')" \
      || { echo "FATAL: cannot import mooncake.engine -- see the traceback above."; exit 1; }; \
    MC=$(python3 -c "import mooncake, os; print(os.path.dirname(mooncake.__file__))"); \
    E="$MC/engine.so"; \
    test -f "$E" || { echo "FATAL: $E missing; the wheel layout changed."; \
                      echo "       found instead:"; ls -1 "$MC" | grep -i engine; exit 1; }; \
    ldd "$E" | grep -q libfabric \
      || { echo "FATAL: $E does not link libfabric -- this is the NON-EFA wheel."; \
           echo "       Check that MOONCAKE_PKG names an ...-efa-... distribution."; \
           echo "       ldd says:"; ldd "$E" | grep -iE "fabric|ibverbs" || true; exit 1; }; \
    fi_syms=$(strings "$E" | grep -cE "^fi_(getinfo|mr_reg)" || true); \
    test "$fi_syms" -ge 2 \
      || { echo "FATAL: $E links libfabric but carries only $fi_syms libfabric"; \
           echo "       entry points (expected >= 2). Suspect a stripped/partial build."; exit 1; }; \
    echo "mooncake EFA verified: links libfabric, $fi_syms libfabric entry points"

# DeepEP v2 reachability, checked here rather than 10 minutes into a weight load.
# Two independent things can be missing and neither raises at import time:
#   1. sgl-deep-ep's v2 extension is absent from the base image, so
#      `--moe-a2a-backend deepep_v2` dies only once the MoE layer is built;
#   2. the whitelist patch did not take, in which case v2 refuses K3 by name.
# The whitelist grep is on the PATCHED file, so it also proves the COPY+patch
# step above actually landed in the same layer stack the server will import.
#
# deep_ep is probed with find_spec, NOT `import deep_ep`. Do not "simplify" that
# back: deep_ep/__init__.py runs _check_prerequisites(), which needs a live CUDA
# device and raises "The NVIDIA driver does not expose a usable CUDA device".
# docker build never has a GPU, so an import here fails 100% of builds on a
# perfectly good image. find_spec answers the only question this stage can
# answer -- is the package present -- without initialising CUDA.
RUN set -eu; \
    SGL=$(python3 -c "import sglang, os; print(os.path.dirname(sglang.__file__))"); \
    grep -q "KimiK3ForConditionalGeneration" "$SGL/srt/arg_groups/moe_hook.py" \
      || { echo "FATAL: K3 is not in the deepep_v2 whitelist -- moe_hook patch did not land"; exit 1; }; \
    grep -q "is_deepep_v2" "$SGL/srt/models/kimi_k3.py" \
      || { echo "FATAL: kimi_k3.py _ep_a2a lists do not mention deepep_v2 -- WRONG NUMERICS if run"; exit 1; }; \
    python3 -c "import importlib.util as u, sys; s = u.find_spec('deep_ep'); \
      print('deep_ep OK:', s.origin) if s else sys.exit(1)" \
      || { echo "FATAL: no deep_ep in this base image; deepep_v2 cannot run."; exit 1; }; \
    echo "deepep_v2 preflight OK"

WORKDIR /workspace
