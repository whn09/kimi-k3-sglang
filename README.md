# Kimi-K3 on AWS p6-b300

Serving `moonshotai/Kimi-K3` on p6-b300 nodes (8x B300 275 GB, sm_103), first
single-node then 1P1D prefill/decode disaggregation over EFA.

## Model

| | |
|---|---|
| Params | 2.78 T total, MXFP4-packed MoE (`mxfp4-pack-quantized`, group 32) |
| On-disk | 1561 GB / 96 shards → ~195 GB per GPU at TP=8 |
| Layers | 93 + 1: **69 KDA** (linear attn) + **24 full MLA** — hybrid, so it needs mamba-style state cache |
| Context | 1 048 576 |
| Multimodal | yes (vision tower; `has_image_understanding: true`) |
| Draft | `RadixArk/Kimi-K3-DSpark`, 2.25 B BF16 / 4.5 GB |

Both repos are public — no `HF_TOKEN` needed.

## Machines

| Host | Internal IP | Role |
|---|---|---|
| P6-B300-1 | 172.31.60.28 | standalone / prefill |
| P6-B300-2 | 172.31.51.133 | decode |

`enp71s0` is the ENA interface; the other 16 `enpXX` are EFA rails (18 HCAs).
Models live on `/opt/dlami/nvme` (27 TB); host python venv is `/opt/pytorch`.

## Layout

| File | Runs on | Purpose |
|---|---|---|
| `Dockerfile` | host | kimi-k3 base + EFA + gdrcopy + Mooncake `-DUSE_EFA=ON` |
| `env_common.sh` | both | shared paths/IPs/env, `PROFILE` table, `require_efa_image` |
| `00_download_models.sh` | host | fetch both models to NVMe |
| `10_launch_standalone.sh` | host | single-node container |
| `20_launch_prefill.sh` / `21_launch_decode.sh` | host | PD containers |
| `22_launch_router.sh` | host | `sglang_router` in front of the P/D pair |
| `start_standalone.sh` / `start_prefill.sh` / `start_decode.sh` | container | the actual `sglang.launch_server` invocations |
| `90_smoke_test.sh` / `91_bench.sh` | host | health + chat + streaming; `bench_serving` |
| `92_sweep.sh` | host | concurrency sweep driver over `91_bench.sh` |

`PROFILE=low-latency|balanced|high-throughput` selects the upstream serving
variant; `env_common.sh` documents which knobs each one moves, and they differ
between standalone and PD decode.

## Quick start

```bash
# once per machine (~1.5 TB, and ~8 min to build the image)
bash 00_download_models.sh
docker build -t kimi-k3-efa:latest -f Dockerfile .

# --- single node (on P6-B300-1) ---
NO_SPEC=1 bash 10_launch_standalone.sh   # first bring-up: base model only
bash 10_launch_standalone.sh             # then with DSPARK
bash 90_smoke_test.sh

# --- 1P1D (prefill on B300-1, decode on B300-2) ---
bash 20_launch_prefill.sh    # on B300-1
bash 21_launch_decode.sh     # on B300-2
bash 22_launch_router.sh     # on B300-1
ENDPOINT=localhost:8080 bash 90_smoke_test.sh

# benchmark
MODE=pd PROFILE=low-latency ENDPOINT=localhost:8080 bash 91_bench.sh
MODE=pd PROFILE=low-latency ENDPOINT=localhost:8080 bash 92_sweep.sh
```

The Dockerfile builds Mooncake from `whn09/Mooncake@fix/efa-gpu-mr-cuda-context`
(upstream PR
[#3177](https://github.com/kvcache-ai/Mooncake/pull/3177)) — override with
`--build-arg MOONCAKE_REPO=... --build-arg MOONCAKE_REF=...` once that lands.

Startup takes ~11 min: ~6 min to load 1.5 TB of weights, then FlashInfer
autotune + CUDA graph capture. **GPU utilisation reads 0% for almost all of
it** — the bottleneck is disk→H2D, then kernel JIT. Watch
`nvidia-smi --query-gpu=memory.used` climb (~200 GB/GPU after load, ~232 GB
once the KV/mamba cache is allocated) rather than utilisation.

## Results

Single node TP=8, `mem-fraction-static 0.85`, `mamba-full-memory-ratio 0.86`:

- **base (`NO_SPEC=1`)**: correct output, ~164 tok/s single-stream
- **DSPARK, block size 7**: same output, accept len 2–5 (rate 0.14–0.57),
  **330–430 tok/s** single-stream — roughly 2.5x

SGLang auto-selects `trtllm_mla` for decode/verify, pins
`--linear-attn-verify-backend nv_cutedsl` (fused Kimi-K3/DSPARK kernel), and
defaults the draft to `trtllm_mha`. The MoE runs on `flashinfer_mxfp4`.

### 1P1D over Mooncake/EFA

ISL 8192 / OSL 1024, 64 prompts, concurrency 32, DSPARK on, via the router.

| mode | profile | out tok/s | mean TTFT | mean TPOT | median ITL |
|---|---|---|---|---|---|
| PD | low-latency | 1683 | 10.06 s | 7.19 ms | 52.6 ms |

That row was measured on a container carrying the GPU-MR fix as a bind-mounted
`.so` rather than from a clean image, so treat it as indicative; the matrix below
is being re-run from `kimi-k3-efa:latest` built from the fix branch. Still to
fill in: PD balanced / high-throughput (both need `PREFILL_DCP_SIZE=8`, see
below) and standalone low-latency / balanced.

Earlier NIXL (`TRANSFER_BACKEND=nixl`, `SGLANG_DISAGGREGATION_NIXL_BACKEND=LIBFABRIC`)
concurrency-sweep points at the same ISL/OSL are in `results/` for comparison.

## Notes and pitfalls

**The upstream image cannot do PD over EFA.** `lmsysorg/sglang:kimi-k3` ships a
pip `mooncake` wheel with **zero** EFA symbols, plus generic libfabric 1.20 (no
EFA provider) and no `aws-ofi-nccl`. This does not fail loudly: it passes
SGLang's PD warmup and then kills the first real request. Hence the `Dockerfile`
and the `require_efa_image` guard in the launchers.

**Bind-mounting the host EFA stack does not fix it.** Besides the wheel problem,
the base image's rdma-core is older than the host's, so host libfabric 2.4
mounted in dies with ``libefa.so.1: version `EFA_1.4' not found`` and
`fi_info -p efa` enumerates 0 providers. Getting it to enumerate needs
`libfabric` + `libefa` + `libibverbs` + the whole ABI-tagged
`libibverbs/` provider dir from the host as one matched set. Installing
aws-efa-installer into the image is cleaner.

**`NCCL_NET_PLUGIN=ofi`** (set in the Dockerfile) — the EFA installer only lays
down `libnccl-net-ofi.so`, not the `libnccl-net.so` NCCL auto-loads. Without it
NCCL silently falls back to TCP (~14 GB/s vs ~400 GB/s), which reads as slow
TTFT rather than an error. Verify with
`NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET` → expect
`NET/OFI Selected provider is efa`.

**Benign startup warnings**: `Failed to load generation config for
.../Kimi-K3-DSpark` (the draft repo has no `generation_config.json`);
`DeepGemm is enabled but the scale_fmt of checkpoint is not ue8m0`;
`Acceleration for non-quantized schemes is not supported by Compressed
Tensors` (the `ignore` list in `quantization_config` keeps attn / shared-experts
/ lm_head in BF16 by design).

**Mooncake could not register the GPU KV cache** on upstream `main`, which is why
the Dockerfile builds from a branch. `fi_mr_regattr` on `FI_HMEM_CUDA` memory
failed with a bare `Operation not supported` for a few buffers per rank;
`registerLocalMemory` rolls back on error, so those buffers ended up on **no NIC
at all**, and the first real KV transfer died with the misleading `remote
mooncake session ... is not alive`. The cause is a missing CUDA context on the
registering thread: `registerLocalMemoryBatch` spawns one `std::async` thread per
buffer, and libfabric's `cuda_get_dmabuf_fd()` calls the *driver* API
`cuMemGetHandleForAddressRange()` without binding one. K3 surfaces it because it
registers 300+ GPU buffers at once (24 MLA layers x 8 ranks at 953 MB, plus
aux/state), so the fan-out outruns anything that would bind a context. Fix:
retain and set the device's primary context before registering device memory —
[Mooncake#3177](https://github.com/kvcache-ai/Mooncake/pull/3177). Failures went
1448 ok / 8 failed → 1456 ok / 0 failed, and the first request 6.9 s-fail →
2.9 s-ok. `require_efa_image` greps the built `.so` for the fix so a stale image
fails in seconds instead of 10 minutes in.

**PD decode `--dcp-size 8` needs a matching prefill dcp on K3.** The upstream
reference gives dcp to decode only, which is fine for low-latency (dcp=1) but
kills the balanced and high-throughput profiles: prefill's `bootstrap_thread`
dies with `PD DCP source/destination KV geometry differs: src=[1152 x24, 256
x10], dst=[1152 x34]`. `prepare_dcp_token_item_lens()`
(`srt/disaggregation/common/conn.py`) builds the destination geometry as a single
item length replicated across every layer, which can never match K3's hybrid
stack (24 full-MLA layers at 1152 B, KDA layers at 256 B). Since
`requires_dcp_relayout()` returns False when `dcp_size == dst_dcp_size`, setting
prefill's dcp equal to decode's bypasses the broken path — that is what
`PREFILL_DCP_SIZE` in the profile table does. Note the RuntimeError is fatal to
that thread, so **prefill must be restarted** after any geometry mismatch; it
will not recover on its own.

**Single node needs no EFA** — TP=8 is all NVLink (53 GB/s per link, Fabric
Manager must be `active`). EFA only matters for the cross-node KV transfer in
PD mode.
