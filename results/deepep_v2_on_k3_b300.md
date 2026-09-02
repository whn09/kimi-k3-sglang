# Kimi-K3 on `--moe-a2a-backend deepep_v2`, 1 x p6-b300 — results

Date 2026-09-01. Host `B300-KR` (8 x B300 SXM6, sm_103, **InfiniBand not EFA** —
2 x ConnectX-7, `NCCL_GIN_TYPE=3`/GDAKI, one plane pinned via
`NCCL_IB_HCA=ibp198s0f0`). Image
`lmsysorg/sglang:nightly-dev-cu13-20260901-07c8f729`. Five local patches, see
`../patches/README.md`.

Raw `bench_serving` logs in `raw/`; the table below is produced by
`../gen_bench_table.py raw/`, not transcribed.

## 1. Does deepep_v2 run K3 / does it support FP4?

**Yes, and the FP4 hypothesis was wrong about the cause.** deep_gemm 0.1.7 already
ships `m_grouped_fp8_fp4_gemm_nt_masked` and both v2 runner paths already carry
`is_fp4_experts -> recipe_a=(1,128)/recipe_b=(1,32)`. What blocks K3+v2 is four
over-strict *validations* plus one real upstream bug (the user's own PR #37211) —
not a missing kernel. Details per blocker in `../patches/README.md`.

## 2. Numerics — v2 is bit-identical to v1

First-token top-5 logprobs, greedy, three fixed prompts (`raw/k3_lp_v1.txt` vs
`raw/k3_lp_v2.txt`): **all 15 tokens and all 15 logprob values byte-identical.**

```
The capital of France is   ' Paris' -0.295707  ' Berlin' -3.983207  ' {' -3.983207 ...
2 + 2 =                    ' '      -0.149297  ' -'      -4.211796  ' \\' -4.961796 ...
def fibonacci(n):          ' \n'    -1.517094  ' #'      -2.267094  ' '   -2.392094 ...
```

Long greedy completions (`raw/k3_probe_*.txt`) agree exactly on 2 of 4 prompts and
diverge after ~60 tokens on the others — expected from a different reduction order
once sampling amplifies a tie, and irrelevant given the identical logprobs.

## 3. Throughput

| tag | backend | workload | chunk | cudagraph | reqs | dur (s) | in tok/s | out tok/s | conc | TTFT p50 (ms) | TTFT mean (ms) | TPOT p50 (ms) | note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `v2_p8k` | deepep_v2 | prefill ISL=8K OSL=1 conc4 | 512 | on | 16 | 87.45 | 1498.77 | 0.18 | 3.64 | 21794.28 | 19879.54 | 0.00 | CAP=512 |
| `v2_p8k_c2048` | deepep_v2 | prefill ISL=8K OSL=1 conc4 | 2048 | off | 16 | 22.20 | 5903.74 | 0.72 | 3.63 | 5477.49 | 5042.35 | 0.00 | CAP=2048 |
| `v1_p8k_c2048` | deepep | prefill ISL=8K OSL=1 conc4 | 2048 | on | 16 | 23.66 | 5540.26 | 0.68 | 3.67 | 5829.51 | 5430.68 | 0.00 | no ElasticBuffer |
| `v2_d256` | deepep_v2 | decode ISL=256 OSL=1K conc32 | 512 | on | 32 | 87.49 | 93.64 | 374.55 | 22.15 | 3443.80 | 18138.40 | 42.05 | CAP=512 |
| `v1_d256_c512` | deepep | decode ISL=256 OSL=1K conc32 | 512 | on | 32 | 45.92 | 178.40 | 713.58 | 31.97 | 3363.22 | 3358.49 | 41.56 | matched to v2_d256 |
| `v1_d256` | deepep | decode ISL=256 OSL=1K conc32 | 2048 | on | 32 | 42.39 | 193.25 | 772.98 | 31.96 | 1590.06 | 1641.05 | 39.84 | UNMATCHED chunk |

### Prefill (ISL=8K, OSL=1) — v2 wins by 6.6% at matched chunk

`v2_p8k_c2048` 5903.74 vs `v1_p8k_c2048` 5540.26 tok/s, median TTFT 5477 vs 5830 ms.

**Read `v2_p8k` (1498.77 tok/s) as a config artifact, not a v2 result.** Going
CHUNK 512 -> 2048 gave 3.94x on the *same* backend, i.e. prefill throughput here is
essentially linear in chunk. Any v1/v2 comparison at unmatched chunk is worthless —
which is why `v1_d256` is kept in the table only as a labelled trap.

### Decode (ISL=256, OSL=1K, conc32) — same per-token latency, heavy TTFT tail

At matched CHUNK=512, cudagraph on, default `max_running_requests`:

- **TPOT p50 42.05 (v2) vs 41.56 ms (v1) — parity.** The a2a itself is not slower.
- **TTFT p50 3443.80 (v2) vs 3363.22 ms (v1) — also parity.**
- **TTFT mean 18138.40 (v2) vs 3358.49 ms (v1) — 5.4x, and 5.3x its own median.**

So v2 is not uniformly slower: half the requests behave exactly like v1, and a tail
stalls hard. That tail drops achieved concurrency to 22.15 (v1: 31.97) and halves
end-to-end output throughput, 374.55 vs 713.58 tok/s, despite identical TPOT.

**Mechanism not identified.** Candidates worth one profiling pass: the fixed-capacity
dispatch (`masked_max_m = CAP x ep_size` = 4096 rows of masked GEMM per forward
regardless of real tokens, see `reference_sglang_deepep_v2_capacity`), and v2's
prefill/decode interleaving under a full `max_running_requests`. One run per arm, no
replicates — treat the tail as reproducible-pending, the TPOT/median parity as solid.

## 4. Memory budget — why CHUNK=8192 is not reachable on one node

Measured on this box (`Load weight end` / `KV Cache is allocated` / `Memory pool end`
log lines, MEMFRAC=0.84):

| item | cost per B300 (of 267.68 GiB) |
|---|---|
| K3 weights + init overhead | **214.88 GB** (80%), leaves 51.11 GB |
| KV pool @ MEMFRAC 0.84 | 6.73 GB / 261120 tokens, leaves 42.54 GB |
| decode CUDA graph capture | **> 21 GB** (begins at avail 41.90 GB) |
| DeepEP v2 ElasticBuffer | **10.5 GiB per 1024 of CAP** — allocated **last** |

The ElasticBuffer figure is exact and linear: CAP=2048 asks 21.00 GiB, CAP=3072
31.50 GiB, CAP=4096 42.00 GiB. Since `chunked_prefill_size / dp_size <= CAP`
(`moe_hook.py:381`), CHUNK=8192 at dp=1 needs CAP=8192 = 84 GiB. Not available.

Two escape routes, both measured and both closed:

- **DP attention** (`--dp-size 8` would make CHUNK=8192 need only CAP=1024): the
  attention weights stop being TP-sharded and are replicated per DP rank, pushing
  weights to **265.87 GiB** — it OOMs inside `_initialize_model`, before the KV pool
  exists. DP=1 loads the same weights fine, so the delta is the replication.
- **CAP=4096 with the CUDA graph off**: frees the >21 GB of graph memory, and the
  42.00 GiB request *still* fails with ~48.7 GB nominally free — it needs one
  contiguous block.

**Working configs.** Prefill arm: `CAP=2048 CHUNK=2048 MEMFRAC=0.85 MAXRUN=32
DISCG=1` -> READY 210 s, `max_total_num_tokens=364416`. Decode arm: `CAP=512
CHUNK=512 MEMFRAC=0.85` with graphs on -> `max_total_num_tokens=256576`. There is no
single config that is right for both; the launcher takes both as env knobs.

Two traps that cost several boots and look like memory bugs but are not:

- `docker rm -f` returns **before** the 8 scheduler processes release the GPUs.
  Relaunching immediately OOMs during *weight load* on a tiny allocation ("Tried to
  allocate 588.00 MiB ... free: 18022400"). `run_k3_v2.sh` now waits for the drain.
- Lowering MEMFRAC is the **wrong** direction. The ElasticBuffer is charged against
  the same physical memory as `--mem-fraction-static`, so a too-low MEMFRAC makes the
  pool sizing go negative first: `total_rest_memory=-2.56 GB` at 0.80. That number is
  what pins the weight footprint at 214.88 GB.

## 5. Not done

- **PD-disaggregated arms** (`--disaggregation-mode prefill` / `decode`, ports
  30001/30002) still unrun with the working knobs. Only the unified server is proven.
- `NCCL_GIN_TYPE=5` (EFA_GDA) cannot run here at all — no EFA device on this host.
- No replicates on any row; the decode TTFT tail in particular needs a second run.
