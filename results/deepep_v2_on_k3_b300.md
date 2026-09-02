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

| tag | backend | workload | chunk | cudagraph | reqs | dur (s) | in tok/s | out tok/s | conc | TTFT p50 (ms) | TTFT mean (ms) | TPOT p50 (ms) | ITL p50 (ms) | note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `v2_p8k` | deepep_v2 | prefill ISL=8K OSL=1 conc4 | 512 | on | 16 | 87.45 | 1498.77 | 0.18 | 3.64 | 21794.28 | 19879.54 | 0.00 | 0.00 | CAP=512 |
| `v2_p8k_c2048` | deepep_v2 | prefill ISL=8K OSL=1 conc4 | 2048 | off | 16 | 22.20 | 5903.74 | 0.72 | 3.63 | 5477.49 | 5042.35 | 0.00 | 0.00 | CAP=2048 |
| `v1_p8k_c2048` | deepep | prefill ISL=8K OSL=1 conc4 | 2048 | on | 16 | 23.66 | 5540.26 | 0.68 | 3.67 | 5829.51 | 5430.68 | 0.00 | 0.00 | no ElasticBuffer |
| `v2_d256` | deepep_v2 | decode ISL=256 OSL=1K conc32 | 512 | on | 32 | 87.49 | 93.64 | 374.55 | 22.15 | 3443.80 | 18138.40 | 42.05 | 41.51 | CAP=512 |
| `v1_d256_c512` | deepep | decode ISL=256 OSL=1K conc32 | 512 | on | 32 | 45.92 | 178.40 | 713.58 | 31.97 | 3363.22 | 3358.49 | 41.56 | 39.24 | matched to v2_d256 |
| `v1_d256` | deepep | decode ISL=256 OSL=1K conc32 | 2048 | on | 32 | 42.39 | 193.25 | 772.98 | 31.96 | 1590.06 | 1641.05 | 39.84 | 39.49 | UNMATCHED chunk |
| `8k1k_nograph_cap2048` | deepep_v2 | 8K/1K conc16 | 2048 | off | 32 | 565.55 | 463.52 | 57.94 | 16.00 | 11735.66 | 11733.14 | 264.94 | 255.24 | CAP=2048 |
| `8k1k_graph_cap1024` | deepep_v2 | 8K/1K conc16 | 1024 | on | 32 | 172.77 | 1517.32 | 189.67 | 16.00 | 22571.74 | 22537.76 | 62.34 | 43.54 | CAP=1024, the default |

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

### 4.1 The launcher defaults (2026-09-02) — and what the CAP ceiling really is

The launcher used to default to `DP=8 CHUNK=8192 CAP=1024`, which **could never
start**: `DP=8` is the measured-and-closed DP-attention route above, so
`bash run_k3_v2.sh` OOMed on a 588 MiB alloc at 265.60 GiB allocated. Defaults are
now the measured-working config; `bash run_k3_v2.sh` with no env overrides reaches
READY in ~165 s (warm caches) at `max_total_num_tokens=232384` and answers.

Three launches on `B300-KR` today, `MEMFRAC=0.85 DP=1 MAXRUN=0`, pin the ceiling:

| CAP | CHUNK | decode graphs | result |
|---|---|---|---|
| 1024 | 1024 | **on** | READY, `max_total=232384`, serves ← **the default, as of §4.2** |
| 2048 | 2048 | **on** | **OOM in `Capture cuda graph`**, asking 6.12 GiB |
| 2048 | 2048 | off | READY, `max_total=232384`, serves — was the default for one day |

**This corrects §4's allocation story.** With graphs on it is *not* the
ElasticBuffer that OOMs — capture takes a **33.43 GiB private pool** and dies
before the ElasticBuffer is ever allocated (`Memory pool end avail=39.63 GB` ->
`capture begin avail=38.99 GB` -> OOM; "33.43 GiB allocated in private pools").
33.43 + 21.00 does not fit in 39.63. So the real trade is **decode CUDA graphs
cost half the prefill chunk**: graphs on caps CAP at 1024, graphs off allows 2048.

> **Both conclusions in this paragraph were wrong — see §4.2, which measured the
> trade end to end.** (a) "Graphs off is the better default": no, graphs on is
> **3.27x** faster end to end on 8K/1K, because the chunk only costs TTFT while
> graphs cost every token. (b) "`--cuda-graph-max-bs` should shrink the 33.43 GiB
> enough to hold both": no, the capture pool is sized by CAP and is completely
> insensitive to the captured shape count — 3 shapes at CAP=2048 still took 33.43
> GiB. The original reasoning below is kept because it shows the trap: prefill
> throughput really is near-linear in chunk (512 -> 2048 is 3.94x, §3), and
> generalising from that one axis alone is what produced the wrong default.

Note `max_total_num_tokens` is **232384 in all three rows** — it is set by MEMFRAC
and MAXRUN, not by CAP, because the ElasticBuffer is allocated after the KV pool.
The 364416 figure above belongs to `MAXRUN=32`, which buys a bigger pool by
capping concurrency at 32; that is the prefill-measurement arm, not a better
default.

### 4.2 The CAP ceiling is the wrong trade — decode CUDA graphs win 3.27x (2026-09-02)

§4.1 concluded "graphs off is the better default for a unified server" from the
*prefill* linearity alone. Measured both ways end to end, that is **wrong**, and the
launcher default is now graphs **on**.

Same box, same 232384-token pool on both sides, `bench_k3.sh` defaults
(8K in / 1K out, 32 requests, concurrency 16, `--random-range-ratio 1.0`):

| | graphs OFF, CAP=CHUNK=2048 | graphs ON, CAP=CHUNK=1024 | ratio |
|---|---|---|---|
| end-to-end duration, 32 req | 565.55 s | **172.77 s** | **3.27x** |
| output token throughput | 57.94 tok/s | **189.67 tok/s** | **3.27x** |
| input token throughput | 463.52 tok/s | **1517.32 tok/s** | 3.27x |
| ITL p50 | 255.24 ms | **43.54 ms** | **5.86x** |
| TPOT p50 | 264.94 ms | **62.34 ms** | 4.25x |
| TTFT p50 | **11735.66 ms** | 22571.74 ms | 0.52x |
| achieved concurrency | 16.00 | 16.00 | — |
| GPU util (`nvidia-smi`) | 30-49% | **25-38%** | — |
| power draw | 311-321 W | **419-441 W** | +36% |

Logs: `k3_bench_8k1k_nograph_cap2048.txt`, `k3_bench_8k1k_graph_cap1024.txt`.

**TTFT doubles, everything else improves ~4-6x.** Break-even on output length:
`11.7 + 0.255N = 22.6 + 0.0435N` -> **N ≈ 52 tokens**. Any request that generates a
real answer wins. Keep `DISCG=1 CAP=2048` only for a prefill-only measurement
(OSL=1) or outputs shorter than ~52 tokens.

**Three mechanisms, all measured:**

1. **The 255 ms step is kernel-launch overhead, not the fixed-capacity dispatch.**
   Without graphs the decode step time is flat in batch size — batch 2 -> 16 moved it
   280 -> 256 ms while total throughput went 7.15 -> 62.6 tok/s (near-linear). A
   per-step cost independent of batch is launch overhead, not compute, not HBM. And
   it is not CAP's fixed-capacity a2a either. Compared at the same metric (ITL p50,
   from the generator): **41.51 ms** at CAP=512 / ISL=256 (§3 `v2_d256`) vs **43.54
   ms** at CAP=1024 / ISL=8192 (here) — +4.9% across a 2x CAP and a 32x context. So
   neither the a2a padding nor the context length is what costs the 255 ms.
   (Do not compare against `v2_d256`'s *TPOT* of 42.05 ms as if it were the same
   quantity: TPOT includes prefill interleaving, which is why the graphs-on/off gap
   is 5.86x in ITL but only 4.25x in TPOT.)

2. **The capture pool is sized by CAP, not by how many shapes are captured.** This
   kills the obvious "keep both" idea. `--cuda-graph-max-bs` is a deprecated alias
   for `--cuda-graph-max-bs-decode` in this image; trimming the list works (the log
   prints `bs=[8, 16, 24]`) and saves nothing:

   | CAP | captured shapes | capture memory | outcome |
   |---|---|---|---|
   | 2048 | 3 — `[8,16,24]` | **33.43 GiB** | OOM asking 6.12 GiB |
   | 1024 | 13 — `[8,16,...,104]` | **18.48 GB** | starts, 61.59 s |

   4x fewer shapes cost *more* memory. Only CAP moved the number. So CAP=1024 is the
   hard ceiling with graphs on: 18.48 + 10.5 <= 38.99 GB, while CAP=1536 would need
   ~27.7 + 15.75 = 43 GB. `--cuda-graph-max-bs-decode` is still worth setting to cut
   the 61.59 s capture, since the pool caps live requests at 232384/9216 = 25 anyway.

3. **Why one knob controls both** (`validate_deepep_v2_dispatch_token_budget`,
   `moe_hook.py:375-410`). `SGLANG_DEEPEP_V2_NUM_MAX_DISPATCH_TOKENS_PER_RANK`
   (`environ.py:1085`, upstream default **128**) is checked against two unrelated
   requirements: `CHUNK / dp_size <= CAP` wants 2048, and
   `graph_bs * tokens_per_req <= CAP` wants only 104. Raising CAP to buy prefill
   chunk inflates the decode graph's workspace by the same factor. **These two needs
   should not share one capacity — worth an upstream issue.** Note CAP is an env var,
   not a flag, so it is invisible in `docker inspect .Args`; `bench_k3.sh` reads it
   from `.Config.Env` into every log header for this reason.

**Methodology warning: `nvidia-smi utilization.gpu` inverted the conclusion here.**
It *fell* (30-49% -> 25-38%) while throughput rose 6.2x. It only reports whether a
kernel was resident at sample time, not SM occupancy or work done — CUDA graphs
collapse thousands of small launches into one replay, so the sampler hits fewer busy
instants while doing far more work. Power tracked reality (315 W -> 430 W). Use
`gen throughput` from the server log and power draw; do not diagnose from
`utilization.gpu`.

Also fixed in the launcher: `docker rm -f` can return with the container still
present (killed, `Exited 137`), so the next `docker run` failed with "container
name is already in use" — a second, independent way a relaunch could not start.
The script now waits for removal *and* for the GPU drain.

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
