# Experiment 1: why does `MC_MAX_CONCURRENT_REG_MR` have an optimum?

P5-1 (p5.48xlarge, H100 x8, 32 EFA NICs, 192 cores), 2026-07-31.
Raw data: `mcreg-exp1-mechanism.txt`, `mcreg-exp1b-cap-vs-count.txt`,
`mcreg-exp1c-cap-curve.txt`.

## Scope limit — read this first

**Everything below is single-process, and single-process is the wrong regime for
the question.** 48 registration threads on a 192-core node never contend for CPU,
so "more threads is faster" is guaranteed and says nothing about why the real
8-process sweep peaks at cap=16 (= 128 threads on 192 cores). The interesting
mechanism only appears when 8 x cap threads oversubscribe the cores.

Kept because two of the four candidate mechanisms die on this data regardless of
regime, and the per-registration cost curve is a useful input. The actual
mechanism test is experiment 4 (`mcreg-exp4-core-budget.md`).

## Setup

`regbench_mech.py <ip> count <bytes> <nbuf>` — one process, GPU 0, one
`batch_register_memory()` of `nbuf` identical GPU buffers. Uniform sizes make
input order meaningless by construction, so any cap effect here is pure
concurrency rather than scheduling. v1 of this harness held total *bytes* fixed
at 8 GiB, which made the 2.7 MB cell 3076 buffers and unrunnable at cap=1; it
holds buffer *count* fixed instead, since count is what the cap schedules.

## 1a. Cost per registration is linear in size, not superlinear

48 buffers per cell, cap=1 (fully serial), time / 48:

| buffer size | total (cap=1) | per registration | per MB |
|---|---|---|---|
| 2.79 MB | 4.24 s | 88 ms | 31.7 ms |
| 79.4 MB | 29.06 s | 606 ms | 7.6 ms |
| 230.8 MB | 71.92 s | 1498 ms | 6.5 ms |
| 410.3 MB | 122.00 s | 2542 ms | 6.2 ms |

Per-MB cost *falls* with size and flattens at ~6.2 ms/MB, i.e. a fixed
per-registration overhead (~80 ms) plus a term linear in bytes. **Rules out
"cost superlinear in size"**, which was one candidate explanation for
largest-first being slow.

## 1b. Registration parallelizes well, so there is no size-proportional global lock

410 MB x 48, sweeping the cap:

| cap | time | speedup vs cap=1 |
|---|---|---|
| 1 | 122.0 s | 1.0x |
| 4 | 46.4 s | 2.6x |
| 16 | 31.1 / 30.5 s | 3.9x |
| 24 | 21.9 / 23.4 s | 5.4x |
| 32 | 16.9 / 14.8 s | 7.7x |
| 40 | 10.1 s | 12.1x |
| 44 | 6.7 s | 18.2x |
| 46 | 6.2 s | 19.7x |
| 48 | 5.4 / 5.2 s | 23.0x |

**Rules out "shared provider lock with a critical section proportional to
size"** — that predicts no speedup from the cap at all. 23x on 48 buffers is
close to linear.

The curve is smooth. An earlier reading of only cap=16/24/47/48 suggested a
cliff between 24 and 47; the intermediate points show a continuous ramp, so
there is no threshold effect to explain.

## 1c. The fast point tracks `cap == count`, not a fixed thread number

Doubling the buffer count moves the whole curve, so nothing here is a property
of "~48 threads":

| count | cap=16 | cap=32 | cap=48 | cap=96 |
|---|---|---|---|---|
| 48 (18.8 GiB) | 30.5 s | 14.8 s | **5.2 s** | — |
| 96 (39.4 GiB) | 85.4 s | 71.2 s | 54.8 s | **10.2 s** |

Which is the expected shape for work that parallelizes: with no CPU pressure,
the fastest cap is always "as many threads as there are buffers" — which is what
`MC_MAX_CONCURRENT_REG_MR=0` (unbounded, the default) already does.

## What survives

Of the four candidates:

- ❌ cost superlinear in size — refuted by 1a
- ❌ shared provider lock, critical section ∝ size — refuted by 1b
- ❓ page-pinning / DMA-BUF bandwidth saturation — not tested here; would show
  as time flat in cap but equal across size classes at equal bytes. 1b shows
  time is *not* flat in cap, so if a bandwidth ceiling exists it sits above the
  ~3.6 GB/s that 48x410 MB in 5.2 s reaches on one domain.
- ✅ per-registration fixed overhead — confirmed at ~80 ms by 1a

None of these explains the 8-process optimum, because none of them is about CPU
contention. That is what experiment 4 tests.
