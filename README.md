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
# once per machine (~1.5 TB, and ~2 min to build the image on a warm base)
bash 00_download_models.sh
docker build -t kimi-k3-efa-v2:latest -f Dockerfile .

# ...or pull it prebuilt instead of building. Published in BOTH regions p6-b300
# has actually been obtainable in -- pick the one this host is in, because a
# cross-region pull of 14.9 GB is slower and bills egress:
#   ap-northeast-2  (where the spot grabber hunts, AZ ap-northeast-2c)
#   us-west-2       (where the capacity-block/on-demand attempts have run)
REGION=$(curl -s -H "X-aws-ec2-metadata-token: $(curl -sX PUT \
  http://169.254.169.254/latest/api/token \
  -H 'X-aws-ec2-metadata-token-ttl-seconds: 60')" \
  http://169.254.169.254/latest/meta-data/placement/region)
ECR=579019700964.dkr.ecr.$REGION.amazonaws.com/kimi-k3-sglang-b300
aws ecr get-login-password --region $REGION \
  | docker login --username AWS --password-stdin "${ECR%%/*}"
docker pull $ECR:deepep-v2-20260902-07c8f729
docker tag  $ECR:deepep-v2-20260902-07c8f729 kimi-k3-efa-v2:latest

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

The ECR tag names the base sglang commit (`07c8f729`) and the build date, and the
pull retags it to `kimi-k3-efa-v2:latest` because that is what `env_common.sh`
defaults `IMAGE` to. It is pushed to two regions on purpose: p6-b300 capacity is
scarce enough that the instance shows up wherever it shows up, and an image that
lives in the wrong region is an extra 15 minutes at exactly the moment a spot
instance is finally in hand. **Pull the dated tag, not `:latest`** — `:latest` moves, so a
run recorded against it cannot be reproduced later. `deepep-v2-20260902-07c8f729`
was built and verified on a p5.4xlarge: mooncake resolves to exactly one
distribution (`mooncake-transfer-engine-efa-cuda13` 0.3.13.post1) whose
`engine.so` links libfabric, all five K3 patches applied cleanly, and `deep_ep` is
importable. It has **not** been run on a B300 yet — the image is verified, the
model is not.

`93_matrix.sh` relaunches every configuration from scratch so all rows share one
container generation, and gates each on `/health` plus a read-back of `dcp_size`,
`mamba_full_memory_ratio` and `mem_fraction_static` from the server's own
`server_args` — a config mismatch skips the row instead of producing a
plausible-looking number. It runs on the laptop because the two hosts have no ssh
trust between them.

The Dockerfile installs Mooncake from the published
`mooncake-transfer-engine-efa-cuda13` wheel, which is prebuilt `USE_EFA=ON` +
`USE_CUDA=ON` against CUDA 13 and now contains both fixes this repo used to build
a fork for: the GPU-MR CUDA-context fix (upstream PR
[#3177](https://github.com/kvcache-ai/Mooncake/pull/3177)) and the bounded
registration fan-out described [below](#bounding-the-fan-out-1387-s--207-s-on-the-dominant-batch).
Override with `--build-arg MOONCAKE_PKG=...`.

### DeepEP v2

K3's MoE runs on `--moe-a2a-backend deepep_v2` + `--moe-runner-backend deep_gemm`,
with **EP inside one node** (`ep_size == tp_size == 8`, `--deepep-v2-mode direct`).
In a PD run the only thing crossing the wire is the Mooncake KV transfer;
cross-node EP is out of scope for these scripts. The image bakes in five source
patches without which v2 either refuses K3 by name or — worse, in the case of
`kimi_k3.py` — **serves wrong numerics silently**, so `require_efa_image` refuses
to launch on an image that cannot prove they are present. Root cause per patch is
in [`patches/README.md`](patches/README.md).

The one knob that matters is `SGLANG_DEEPEP_V2_NUM_MAX_DISPATCH_TOKENS_PER_RANK`,
called `CAP` here. It is an **env var, not a flag**, so it never appears in
`docker inspect .Args`; the launchers echo it explicitly for that reason. One
capacity has to satisfy two unrelated constraints, and it also sizes two
allocations competing for the same ~39 GB:

| | wants CAP | cost of a bigger CAP |
|---|---|---|
| prefill chunk (`chunk/dp_size <= CAP`) | large | — |
| decode graph (`graph_bs*tokens_per_req <= CAP`) | small | — |
| ElasticBuffer | — | 10.5 GiB per 1024 of CAP |
| decode graph capture pool | — | 18.48 GB @1024, 33.43 GiB @2048 |

**PD disaggregation is what dissolves that conflict**: prefill and decode are
separate processes, so each gets its own value of an env var and each takes the
end of the trade it actually wants. Hence two caps in `env_common.sh`:

| | CAP | chunk | decode CUDA graphs |
|---|---|---|---|
| `PREFILL_*` | 2048 | 2048 | **off** — frees the whole capture pool |
| `DECODE_*` | 1024 | 1024 | on |
| `STANDALONE_*` (unified baseline) | 1024 | 1024 | on |

Prefill gives up graphs it would never use and buys twice the chunk (prefill is
near-linear in chunk: 512 → 2048 was 3.94x). Decode keeps graphs, which are worth
3.27x end-to-end and an ITL p50 of 43.54 ms against 255.24 ms — the 255 ms was
kernel-launch overhead, not the a2a. Break-even between the two is ~52 output
tokens.

Two things to know before changing these:

- **`PREFILL_CAP=2048` is untested on the PD prefill node.** 2048 is the measured
  graphs-off ceiling (3072 asks a 31.50 GiB ElasticBuffer and OOMs), but that was
  a unified server with no draft model and no `--enable-symm-mem`, both of which
  the PD prefill node has. If it OOMs after "Load weight", drop to 1024.
- **Speculative decoding is what makes a small `DECODE_CAP` illegal.** With spec
  off `tokens_per_req` is 1 and the decode constraint is invisible; with DSPARK
  block 7 each request carries ~8 tokens per step, so the largest captured batch
  must be ≤ CAP/8. The "smaller CAP is faster" experiment (on DSV4/B300, 2048 →
  256 was −16% step / +18% tok/s, untested on K3) therefore needs
  `DECODE_CAP=256 DECODE_CGMAXBS=32`. Both `start_*.sh` pre-check this and fail in
  a second rather than after a 10-minute weight load.

`MOE_A2A_BACKEND=none` restores the old plain-TP `flashinfer_mxfp4` path as a
baseline, and puts the chunk back to 16384 so the comparison is not rigged.

Which NCCL GIN backend DeepEP v2 uses is auto-detected (`detect_gin`), because
sgl-deep-ep aborts on a `NONE` gin type even for a single-node `direct` run:
type **5** (EFA_GDA) on an EFA box, type **3** (GDAKI) plus a one-plane
`NCCL_IB_HCA` pin on the InfiniBand-only B300. Type 5 on p6-b300 is **unverified**
— it is confirmed on p5en's gen-2 EFA, not on this generation's factory stack, so
it is the first thing to suspect if startup aborts on the GIN assert
(`NCCL_DEBUG_SUBSYS=GIN` prints which backend was selected).

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
defaults the draft to `trtllm_mha`. The MoE ran on `flashinfer_mxfp4` when these
numbers were taken; it now runs on DeepEP v2 + `deep_gemm` by default (above), so
treat this section as the plain-TP baseline — reproduce it with
`MOE_A2A_BACKEND=none`.

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

**Turning DSPARK off removes the decay entirely.** Same config otherwise (PD
balanced, dcp 8/8, mamba 1.03, Mooncake/EFA), `NO_SPEC=1`, three runs on one
container generation — `speculative_algorithm=None` verified in `server_args`:

| | run 1 | run 2 | run 3 | run 4 |
|---|---|---|---|---|
| DSPARK on | 1667.8 | 1009.3 | 933.8 | 878 |
| DSPARK off | 938.5 | 939.2 | 940.3 | — |

Spread across the three no-spec runs is 0.19 %, and median ITL is 28.18 / 28.19 /
28.19 ms. Two things follow. First, **only the first run benefits from
speculation** (+78 %). Second, the degraded plateau (~878) is *below* the no-spec
throughput (~939): once accept rate reaches 0.01 the draft is pure overhead, which
the ITL confirms — 28.2 ms/step without speculation versus 57–58 ms/step with it,
because each step drafts 7 tokens and verifies them. So at accept len ≈ 1 the
per-token cost is roughly 2× the non-speculative baseline.

That isolates the cause to **DSPARK's cross-request state on a PD decode node**.
The prime suspect is decode-side `--enable-linear-replayssm-spec` (on by default
here, with `mamba_ssm_dtype=float32`). KDA's rollback path
(`commit_kda_replayssm_after_verify` in `srt/speculative/spec_utils.py`) replays
the accepted window into an fp32 `temporal` checkpoint on every commit, and in PD
mode the KDA state starts from a cross-node KV transfer — the one ingredient
standalone lacks. Slot reuse is not the issue: the ring cursors are zeroed on
alloc (`mem_cache/memory_pool.py:1305-1312`). The working hypothesis is that the
folded SSM checkpoint drifts from the target's true recurrent state, so the draft
mispredicts progressively more.

Note that the cookbook's own no-DSPARK decode command pairs `--dcp-size 8` with
`--mamba-full-memory-ratio 1.44` rather than 1.03, since no draft model has to be
budgeted for. This control deliberately held mamba at 1.03 so that speculation was
the only variable, so 939 tok/s understates what a properly-tuned no-spec config
would reach — it does not affect the conclusion that the decay disappears.

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

### Prefill CAP: the DeepEP v2 default costs 2.6× at ISL 8192

Same operating point as the profile matrix above (ISL 8192 / OSL 1024, 64
prompts, concurrency 32, DSPARK on, PD through the router, Mooncake/EFA), 2x
p6-b300, DeepEP v2 `direct` ep=tp=8, decode fixed at `DECODE_CAP=512
DECODE_CGMAXBS=64`. Only `PREFILL_CAP` varies, and `PREFILL_CHUNK` tracks it
(`CHUNK/dp_size <= CAP` is a boot check, so chunked-prefill-size *is* CAP here).
Each arm ran a **discarded warmup** plus two timed runs — the warmup is not
optional, it is worth 6–9 % on the first run.

| prefill CAP = chunk | out tok/s (r1 / r2) | total tok/s | mean TTFT | mean TPOT | ITL p50 |
|---|---|---|---|---|---|
| 2048 (the current default) | 688.98 / 687.73 | 6201 / 6190 | 31.4 s | 5.91 ms | 43.0 ms |
| 4096 | 1238.20 / 1249.13 | 11144 / 11242 | 14.1 s | 7.63 ms | 56.9 ms |
| **8192** | **1775.03 / 1759.55** | 15975 / 15836 | **6.6 s** | 8.87 ms | 66.2 ms |
| 16384 | 1785.01 / 1794.43 | 16065 / 16150 | 6.5 s | 8.70 ms | 66.1 ms |
| plain TP (`MOE_A2A_BACKEND=none`, chunk 16384) | 2291.05 / 2291.76 | 20619 / 20626 | 7.0 s | 5.38 ms | 42.0 ms |

**Raise `PREFILL_CAP` to at least the ISL you serve.** 2048 → 8192 is **+158 %**
output throughput and cuts mean TTFT 31.4 → 6.6 s (4.8×), and every arm is
reproducible to 0.9 %. The default of 2048 was carried over from a *standalone*
measurement where one number had to satisfy the decode capture pool as well; a PD
prefill node runs `--disable-cuda-graph`, has no capture pool, and boots fine at
16384.

**The knee is exactly ISL.** 8192 → 16384 buys 0.9 %, because an 8192-token
request is no longer chunked once chunk ≥ 8192; past that, CAP only enlarges the
ElasticBuffer. So this is not a knob to sweep — set it from the workload.

**TPOT gets worse as prefill gets faster (5.91 → 8.87 ms), and that is not a
regression.** The decode node is byte-identical across these five rows. At
CAP=2048 prefill is the bottleneck and starves decode, so decode runs small
batches at low per-token cost and low throughput; ITL p50 moves the same way
(43.0 → 66.2 ms). Read the throughput column, and read TPOT against the
concurrency actually achieved.

**DeepEP v2 at its best is still 78 % of plain TP here** (1794 vs 2291) with 1.6×
the TPOT. That is the price of the v2 dispatcher at ep=8 on two nodes, and it is
not something CAP recovers — v2 is for large-EP shapes. Today's plain-TP arm also
reproduces the profile-matrix reference row above (2162.7 / 2158.1) to +6 %,
which is what validates the whole comparison as same-口径.

### Decode CAP: what makes a small one legal (and what silently breaks)

Untested territory — this section is the constraint, not a result. On DSV4/B300
dropping the decode capacity 2048 → 256 was −16 % step time / +18 % tok/s, so a
small decode CAP is worth measuring on K3 too. Getting there needs one more knob
than it looks, because **two different checks read the decode CAP**:

| where | check | when it fires |
|---|---|---|
| `moe_hook.py:400-413` | `graph_bs * tokens_per_req <= CAP` | at startup, before weight load |
| `deepep_v2.py:257` | `tokens_this_forward > CAP` → `ValueError` | **every forward, at runtime** |

Passing the startup check is *not* sufficient. `graph_bs` only describes the
captured shapes; the runtime check sees the batch the scheduler actually built.
And it does not degrade to eager — it fails the request, minutes into a run that
started clean. `decode.py:2609` builds that batch as
`min(req_to_token_pool.size, max_running_requests)` (the `+extra_slots+1` in
`pool_configurator.py:940` sizes the *pool*, not the batch), so the sufficient
rule — which also implies the startup check, since `moe_hook.py` clamps `graph_bs`
by `max_running_requests` anyway — is:

```
max_running_requests * (SPEC_BLOCK_SIZE + 1) <= DECODE_CAP
```

**Speculative decoding is what makes this bite.** `tokens_per_req` is
`block_size + 1` — `speculative_hook.py:479-494` resolves
`speculative_num_draft_tokens = gamma + 1`, and `overrides.py:1869` feeds exactly
that to the budget check — so DSPARK block 7 costs 8 tokens per request per step.
With `NO_SPEC=1` it is 1 and none of this is reachable.

The trap: **DSPARK rewrites `max_running_requests` to 48 when nothing is passed**
(`speculative_hook.py:506-514`). 48 × 8 = 384, which fits the shipped `CAP=1024`
and is why the default needs no tuning — and which makes `CAP=128` fail on step
one no matter what `--cuda-graph-max-bs-decode` says. Hence `DECODE_MAXRUN`.

| DECODE_CAP | DECODE_MAXRUN | DECODE_CGMAXBS | outcome |
|---|---|---|---|
| 1024 | *(unset → 48)* | *(unset)* | 384 ≤ 1024 — the shipped default |
| 128 | *(unset → 48)* | 16 | aborts at launch, 384 > 128 |
| 128 | 16 | 16 | legal: 16 × 8 = 128 ≤ 128, captured to bs=16 |
| 128 | 16 | *(unset)* | legal — sglang's list is clamped to 16 anyway |
| 128 | 32 | 32 | aborts at launch, 256 > 128 |

`start_decode.sh` now enforces the rule before the weight load and warns on the
one combination that is quiet rather than loud: `CGMAXBS < MAXRUN`, where the
batch is legal for the a2a but runs uncaptured at ~255 ms/step.

Two things to hold equal when comparing two decode CAPs, or the arms differ on
more than one axis: `DECODE_MAXRUN` (a small CAP forces a small one, so the
larger-CAP control must be given the same value rather than DSPARK's 48), and
`DECODE_CHUNK` (it defaults to CAP, and although a decode node's chunk is never
checked against CAP — `moe_hook.py:376` skips the prefill half when
`disaggregation_mode == "decode"` — letting it track CAP moves a second knob).
`DECODE_MAXRUN` is in the filename for that reason. Note that a lower
`max_running_requests` caps server-side concurrency, so a `CAP=128` row measures
CAP *and* an admission limit unless the control shares it.

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

Measured against the PR #3177 branch as of image build 05:42 UTC, which predates
that PR's last two (non-perf) commits.

#### Bounding the fan-out: 138.7 s → 20.7 s on the dominant batch

`whn09/Mooncake@fix/efa-mr-reg-throttle` replaces both batch fan-outs with a
fixed pool (`MC_MAX_CONCURRENT_REG_MR`), demotes the per-chunk `LOG(WARNING)` to
the existing trace gate, and logs one batch total instead. Verified by building
that branch and bind-mounting the resulting `engine.cpython-312-*.so` over the
image's copy, so nothing but Mooncake changed:

| | baseline (unbounded) | bounded, cap 32 |
|---|---|---|
| decode registration window | 09:34:00.7 → 09:36:19.3 = **138.7 s** | 10:11:18.0 → 10:12:12.0 = **54.1 s** |
| prefill registration window | — | 10:15:48.6 → 10:16:38.2 = **49.6 s** |
| buffers registered | 1376 | 1376 |
| peak threads (largest batch) | 138 — one per buffer | 32 |
| slowest self-reported chunk | 134.9 s | 52.1 s ≈ the batch wall time |
| `WARNING` lines per startup | 1376 | 0 |

Output throughput is unchanged: **939.72** tok/s against the 938.5–940.3 baseline
band (`NO_SPEC=1`, PD balanced, ISL 8192 / OSL 1024 / c32, 64/64 successful),
median ITL 28.16 vs 28.18–28.19 ms.

**Fewer threads are faster.** Sweeping the cap over the dominant 138-buffer batch
(everything else held fixed) is not monotonic in the direction you would expect:

| `MC_MAX_CONCURRENT_REG_MR` | 138-buffer batch |
|---|---|
| unbounded (138 threads) | 138.7 s |
| 128 | 130.1 s |
| 32 | 51.0 s |
| **8** | **20.7 s** |
| 4 | 24.8 s |

It falls from 128 down to 8 and then turns back up, so 8 is a real optimum rather
than the end of a slope. Registration is not CPU-bound — it pins pages and
serializes on the EFA provider's per-domain lock — so extra threads buy
contention, not parallelism. Note the practical consequence: the branch's first
revision defaulted to `nproc/4` clamped to [4, 32], which on these 192-core nodes
picks exactly 32, the worst admissible value. It now defaults to a flat 8.

#### Why it is still ~5x NIXL, and where the rest is

Even at cap 8 the window is far above NIXL's 4 s, and the cause is upstream of
the cap. `efa_transport.cpp` only takes the NIC-parallel registration path when
`parallel_reg_mr == -1` (the default) **and** the buffer was pre-touched — and
pre-touch is skipped for VRAM, because a CPU-side store into a `cudaMalloc`
pointer segfaults:

```cpp
bool is_host_mem = resolved_name.rfind("cpu", 0) == 0;
bool do_pre_touch = is_host_mem && ...;                       // false for the KV cache
use_parallel_reg = assigned_nics.size() > 1 && do_pre_touch;  // so: 0
```

The KV cache is GPU memory, so **every buffer registers on all 16 NICs
serially**. That is 138 × 16 = 2208 `fi_mr_regattr` calls in the big batch, ~700
ms each (220 MB per buffer at 4 KB pages, re-pinned once per NIC). Two levers,
neither tested yet: force the NIC-parallel path with
`MC_ENABLE_PARALLEL_REG_MR=1` (which would push concurrency to 8 × 16 = 128 —
and the sweep above says that direction backfires), or shrink the per-buffer NIC
fan-out so there is simply less to register.

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
