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
| `93_matrix.sh` | laptop | profile-matrix driver: relaunch + verify + bench each config |
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

# --- whole profile matrix, unattended (from the laptop) ---
bash sync.sh push && bash 93_matrix.sh    # ~1 h: 5 configs x 2 runs
```

`93_matrix.sh` relaunches every configuration from scratch so all rows share one
container generation, and gates each on `/health` plus a read-back of `dcp_size`,
`mamba_full_memory_ratio` and `mem_fraction_static` from the server's own
`server_args` — a config mismatch skips the row instead of producing a
plausible-looking number. It runs on the laptop because the two hosts have no ssh
trust between them.

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

### Profile matrix

ISL 8192 / OSL 1024, 64 prompts, concurrency 32, DSPARK on. PD runs go through
the router and move KV over Mooncake/EFA. Every row below was produced by one
unattended `93_matrix.sh` run — one container generation, one script revision,
each config relaunched from scratch and its `dcp_size` /
`mamba_full_memory_ratio` / `mem_fraction_static` read back out of `server_args`
before benchmarking. All ten runs did identical work: 64/64 successful, 524288
input tokens, 65536 generated, 0 errors. Both runs are shown because the spread
turned out to be the most interesting result — see the accept-length collapse
below, which is why the dcp=8 rows should be read as first-run numbers.

| mode | profile | out tok/s (r1 / r2) | total tok/s | mean TTFT | median TTFT | mean TPOT |
|---|---|---|---|---|---|---|
| PD | low-latency (dcp 1/1, symm on, mamba 0.17) | **2162.7 / 2158.1** | 19465 / 19423 | 6.12 / 6.09 s | 5.03 / 5.27 s | 7.11 / 7.07 ms |
| PD | balanced (dcp 8/8, mamba 1.03) | 1649.1 / *999.7* | 14842 / 8998 | 5.42 / 4.88 s | 4.16 / 2.73 s | 11.36 / 19.03 ms |
| PD | high-throughput (= balanced + decode mem 0.92) | 1675.6 / *1029.5* | 15081 / 9265 | 5.30 / 4.83 s | 4.41 / 2.82 s | 11.40 / 19.05 ms |
| standalone | low-latency (dcp 1, custom-AR on, mamba 0.86) | 1441.2 / 1506.5 | 12971 / 13559 | 4.83 / 3.99 s | 1.94 / 1.91 s | 16.32 / 16.29 ms |
| standalone | balanced (dcp 8, mamba 5.13) | 1333.3 / 1395.2 | 12000 / 12557 | 5.15 / 4.13 s | 2.36 / 1.96 s | 17.52 / 17.52 ms |

**PD low-latency is the best configuration here, and it is the only one that is
reproducible to within noise.** 2162.7 / 2158.1 across these two runs, and
2159 / 2164 in the backend A/B below — four runs inside 0.3 %. It beats the best
standalone profile by 43 % on output throughput and 2.3× on TPOT: two nodes'
worth of hardware, but more than the 2× decode-side FLOPs alone would give,
because the decode node never interleaves prefill chunks into its batches.

**Both PD dcp=8 profiles lose ~39 % on their second run** (1649 → 1000,
1676 → 1030) with identical token counts and unchanged median ITL
(57.97 → 57.62 ms). It is not a one-off cliff: a 4-repeat re-run decays
monotonically and then saturates, and a fresh container reproduces the whole
curve, so it is deterministic rather than noise.

| run (fresh containers) | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| 4-repeat matrix run | 1660 | 1038 | 962 | 878 | 876 (after 12 min idle) |
| independent repeat | 1668 | 1009 | 934 | — | — |

**The mechanism is DSPARK accept length collapsing, not queueing.** Aligning the
decode log by decode step:

| run | start | peak | tail plateau | steps for 1024 tok | accept rate |
|---|---|---|---|---|---|
| 1 | 2.08 | **5.15** | no decay; 5.5 → 7.0 | ~880 | ~0.5 |
| 2 | 1.85 | 4.28 | 1.59 | ~1400 | ↓ |
| 3 | 4.77 | 4.11 | 1.35–1.4 | ~1440 | 0.05 |
| 4 | 5.69 | 4.11 | **1.07–1.2** | ~1640 | **0.01** |

Same 1024 output tokens, but run 4 needs 1640 decode steps where run 1 needs 880
— that ratio alone accounts for 1660 → 878. Median ITL is flat at 56–58 ms
throughout, i.e. each step costs the same and there are simply more of them.
Runs 3 and 4 even *start* higher (4.77, 5.69) before collapsing, so the server is
not broken from the outset; it degrades within each run, earlier each time.

What this rules out:

- **not preemption or admission pressure** — `#retracted-req` and `#queue-req`
  are 0 for every batch of every run, and mean/median TTFT do not worsen
- **not radix-cache reuse** — `91_bench.sh` passes `--flush-cache` and the prefill
  log shows 0 % hit and ~530 K new tokens every run, so prefill does identical work
- **not the prompts** — the random dataset is seeded with `--random-range-ratio 1.0`
- **not wrong output** — spot checks return correct text with
  `finish_reason: length`; K3 is a reasoner, so most of a short budget goes to
  `reasoning_content`
- **not KV-pool or mamba-slot pressure** — at the *same* mamba usage 0.5, run 1
  gets accept len 4.32 and run 5 only 3.32; the state is per-run-history, not
  per-load
- **not `mem-fraction-static`** — `balanced` and `high-throughput` differ only in
  that knob and decay identically
- **not dcp=8 by itself, and not DSPARK by itself** — standalone at dcp=8 with
  DSPARK on reports a rock-steady accept length of **6.40** across all six runs
  (6.39–6.41). Only the PD × DSPARK combination degrades.

Reading the DSPARK implementation in the image eliminates two more candidates:
`HostConfidenceBudgetPlanner`'s carry ring is generation-guarded (stale rows
return `ones`, which is optimistic, so it cannot depress the budget), and the
verify-budget scheduler is inert here — no `--speculative-dspark-sps-table-path`
means `build_uninitialized_sps_table()`, and the server itself warns that the
budget then "degenerates to verify-all (zero scheduling gain)".

The remaining suspect is decode-side `--enable-linear-replayssm-spec` (on by
default here, with `mamba_ssm_dtype=float32`). KDA's rollback path
(`commit_kda_replayssm_after_verify` in `srt/speculative/spec_utils.py`) replays
the accepted window into an fp32 `temporal` checkpoint on every commit, and in PD
mode the KDA state starts from a cross-node KV transfer — the one ingredient
standalone lacks. Slot reuse is not the issue: the ring cursors are zeroed on
alloc (`mem_cache/memory_pool.py:1305-1312`). The working hypothesis is that the
folded SSM checkpoint drifts from the target's true recurrent state, so the draft
mispredicts progressively more.

**Treat the dcp=8 rows as first-run numbers.** A restart fully recovers them, so
the workaround is to restart decode between measurements, or to run
`NO_SPEC=1`. It also explains the earlier unexplained 1598 vs 2159 for what
should have been the same config: that measurement landed in the degraded state.

**The profile names do not describe this workload point.** `low-latency` wins on
throughput too — by 31 % over `balanced` even on balanced's *good* run — because
at OSL 1024 the accumulated per-token cost (7.11 vs 11.36 ms TPOT) dwarfs
balanced's one-off TTFT saving (6.12 → 5.42 s). `high-throughput` only raises
decode `mem-fraction-static` 0.85 → 0.92 and lands within 1.6 % of `balanced`: at
concurrency 32 decode is not KV-capacity-bound, so the extra memory buys nothing.
Standalone's two profiles are likewise nearly interchangeable, with `low-latency`
ahead by 8 %. Expect the ordering to move at higher concurrency or shorter OSL.

**`mamba_full_memory_ratio` dominates the standalone token pool.** `balanced`'s
ratio of 5.13 puts ~17 GB into the KDA state cache
(`ssm_state size: 15.67 GB`, 309 slots), leaving
`max_total_num_tokens=63744` — versus 475776 for `low-latency` at ratio 0.86.
That is a 7.5× smaller pool, and at ISL 8192 × 32 concurrent the working set
(~262 K tokens) exceeds it 4×, so `balanced` runs KV-bound and recomputes. Its
309 state slots do raise `max_running_requests` 35 → 48, which nearly cancels the
loss; the cancellation will not hold at longer ISL.

An earlier pass at this matrix was thrown out because of a launcher bug:
`20_launch_prefill.sh` hardcoded `MAMBA_RATIO=0.86` while `DCP_SIZE` came from the
profile, so `PROFILE=balanced` launched the prefill node as **dcp=8 with
mamba=0.86** — a pairing no profile defines, where dcp shards the KV cache but the
state cache stays sized for the unsharded case. `PREFILL_MAMBA_RATIO` and
`PREFILL_MEM_FRACTION` are now profile-derived in `env_common.sh`; the launcher
echoes its resolved config the way `21_launch_decode.sh` always did; and
`93_matrix.sh` asserts the values against `server_args`, so the same class of
mistake now aborts the row instead of publishing a number. For reference, fixing
it moved PD balanced 1354 → 1649 tok/s.

### Mooncake vs NIXL: KV transfer is at parity, startup is not

Same workload point, PD low-latency, two runs each, all four on one container
generation. PD low-latency is the right config for a backend A/B precisely
because it is the reproducible one: its four independent runs here and in the
matrix above span 2158–2164 tok/s, so a real backend difference of more than
~1 % would be visible.

| backend | out tok/s | total tok/s | mean TTFT | median TTFT | mean TPOT | median ITL |
|---|---|---|---|---|---|---|
| Mooncake/EFA | 2159 / **2164** | 19428 / **19472** | 6.04 / 6.09 s | 5.05 / 5.31 s | 7.09 / 7.07 ms | 52.3 / 51.1 ms |
| NIXL/LIBFABRIC | 2101 / 2141 | 18912 / 18912 | 6.26 / 6.07 s | 5.21 / 5.21 s | 6.99 / 7.10 ms | 51.0 / 51.0 ms |

**No meaningful difference in serving performance.** Mooncake is nominally 1–3 %
ahead, but run-to-run spread within each backend is 2–3 %, so the two are
indistinguishable here; TTFT, TPOT and ITL all overlap. 64/64 in every run, 0
errors. NIXL genuinely used EFA — both sides log
`Backend LIBFABRIC was instantiated`, not a TCP fallback. Verify that before
trusting any NIXL number.

**Mooncake costs ~2 extra minutes of startup, though.** Measured from the end of
FlashInfer autotune to the server accepting connections — the phase where the
only remaining work is KV-cache memory registration:

| backend | autotune done → Uvicorn ready |
|---|---|
| Mooncake/EFA | 07:40:44 → 07:42:50 = **125 s** |
| NIXL/LIBFABRIC | 07:29:32 → 07:29:36 = **4 s** |

Those 125 s are 1456 `fi_mr_reg` calls (the `efa_transport.cpp:483` log lines
bracket the window exactly), fanned out through an unbounded
`std::async(std::launch::async)` in `registerLocalMemoryBatch` — peak 1104
concurrent registrations on a 192-core box, so the `duration=` each line reports
is queueing delay, not registration time. That is the one place Mooncake is 30×
behind NIXL on K3, and it is a startup cost only, not a serving cost.

Fix in progress on `whn09/Mooncake@fix/efa-mr-reg-throttle`: bound the fan-out to
a fixed pool (`MC_MAX_CONCURRENT_REG_MR`, default `nproc/4` clamped to [4, 32]),
demote the 1456 per-chunk `LOG(WARNING)` lines to the existing trace gate, and log
one batch total instead. The numbers above are from before that change; rebuild
the image against that branch to re-measure the startup column.

Measured against the PR #3177 branch as of image build 05:42 UTC, which predates
that PR's last two (non-perf) commits.

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
`/root/.cache/sglang` (FlashInfer autotune results), `/root/.triton` and
`/root/.nv/ComputeCache` (CUDA/PTX JIT). `CACHE_MOUNTS` in `env_common.sh` maps
all seven to `$HOST_CACHE_DIR`; after one launch the host side holds ~971 MB
(nv_compute 867 M, tvm-ffi 65 M, triton 39 M, sglang 92 K) that every previous
restart had thrown away, while the three conventional dirs are still 4–12 KB.
Confirm a mount is doing something with

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
