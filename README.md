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
| `env_common.sh` | both | shared paths/IPs/env, `PROFILE` table, `CACHE_MOUNTS`, `require_efa_image` |
| `00_download_models.sh` | host | fetch both models to NVMe |
| `10_launch_standalone.sh` | host | single-node container |
| `20_launch_prefill.sh` / `21_launch_decode.sh` | host | PD containers |
| `22_launch_router.sh` | host | `sglang_router` in front of the P/D pair |
| `start_standalone.sh` / `start_prefill.sh` / `start_decode.sh` | container | the actual `sglang.launch_server` invocations |
| `90_smoke_test.sh` / `91_bench.sh` | host | health + chat + streaming; `bench_serving` |
| `92_sweep.sh` | host | concurrency sweep driver over `91_bench.sh` |
| `sync.sh` | laptop | push scripts to both hosts, pull `results/` back |

`PROFILE=low-latency|balanced|high-throughput` selects the upstream serving
variant; `env_common.sh` documents which knobs each one moves, and they differ
between standalone and PD decode.

The launch commands follow the SGLang cookbook recipe for this model —
[docs.sglang.io/cookbook/autoregressive/Moonshotai/Kimi-K3](https://docs.sglang.io/cookbook/autoregressive/Moonshotai/Kimi-K3)
— which is also where the three profiles and the 1P1D reference command come
from. The deliberate departures from it are: `--disaggregation-transfer-backend
mooncake` instead of nixl, and a prefill `--dcp-size` matching decode's for the
balanced / high-throughput profiles (see the DCP note below).

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

| profile | out tok/s | total tok/s | mean TTFT | mean TPOT | median ITL | mean E2E |
|---|---|---|---|---|---|---|
| low-latency (dcp 1/1, symm on, mamba 0.17) | **1598** | 14384 | 10.96 s | **7.12 ms** | **52.5 ms** | **18.25 s** |
| balanced (dcp 8/8, mamba 1.03) | 1354 | 13540 | **7.23 s** | 11.62 ms | 58.4 ms | 19.12 s |
| high-throughput (= balanced + decode mem 0.92) | 1436 | 12924 | 8.76 s | 11.24 ms | 57.1 ms | 20.26 s |

64/64 requests succeeded in every run, 0 MR / KV-transfer / geometry errors on
both sides. Measured from `kimi-k3-efa:latest` built from the fix branch, no
bind-mounted `.so`. An earlier low-latency run on a hot-patched container gave
1683 tok/s / 10.06 s TTFT — within run-to-run noise, i.e. the packaged fix
behaves like the hot patch.

**The profile names invert at this workload point.** `balanced` buys a 34 %
better TTFT (10.96 → 7.23 s) by paying 63 % on TPOT (7.12 → 11.62 ms). At OSL
1024 the accumulated per-token cost dwarfs the one-off TTFT saving, so
`low-latency` wins on *throughput* too, by 18 %. `high-throughput` only raises
decode `mem-fraction-static` 0.85 → 0.92, which lands between the two and inside
noise of `balanced` — at concurrency 32 decode is not KV-capacity-bound, so the
extra memory buys nothing. Expect the ordering to change at higher concurrency
or shorter OSL; these are single runs, not averaged.

Still to fill in: standalone low-latency / balanced (needs the 1P1D pair torn
down). NIXL (`TRANSFER_BACKEND=nixl`,
`SGLANG_DISAGGREGATION_NIXL_BACKEND=LIBFABRIC`) is the comparison baseline.

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

**Startup is ~10 min and most of it is JIT, not weight I/O.** A cold prefill
container spends 32 s on distributed init, **140 s** inside `Load weight` (three
"Precompiled the Kimi-K3 KDA / vision RoPE / vision FA4 kernel" steps plus
CUTLASS DSL codegen — the 1.5 TB read itself is a minor part), **262 s** in
FlashInfer autotune, and ~2 min on CUDA-graph capture and warmup. None of it
depends on the serving flags, so changing only `--mem-fraction-static` still
costs the full 10 minutes.

Persisting the JIT output is what shortens a restart, and the three cache dirs an
SGLang guide typically tells you to mount are the *wrong* ones on this image —
`deep_gemm`, `torch` and `flashinfer` stay at 4–12 KB. The caches that actually
get written live in `/root/.cache/tvm-ffi` (sgl_kernel JIT, ~58 MB),
`/root/.cache/sglang` (FlashInfer autotune results), `/root/.triton` (~5 MB) and
`/root/.nv/ComputeCache` (CUDA/PTX JIT, ~800 MB). `CACHE_MOUNTS` in
`env_common.sh` maps all seven to `$HOST_CACHE_DIR`. Confirm a mount is doing
something with

```bash
docker exec kimi-k3-prefill find /root/.cache/tvm-ffi -type f -newermt '-20 min' | wc -l
```

— on a launch that reused the cache this is ~0; when it equals the total file
count, everything was recompiled.

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
