# EFA batch MR registration: cap x input order, 8-process K3 replay

Host p5.48xlarge (H100, 32 EFA NICs, 192 cores), Mooncake built on the host with
`USE_EFA=ON USE_CUDA=ON` so GPU buffers take the real `fi_mr_regattr` /
`FI_HMEM_CUDA` DMA-BUF path. Replay is `regbench_k3_mp.py`: 8 processes, one
`TransferEngine` and one GPU each, 182 buffers / 14.3 GiB per rank (1456 /
114 GiB per node), barrier-synchronized so all 8 domains hit the NICs at once.

## Why 8 processes

SGLang's `MooncakeTransferEngine` is a module-level singleton initialized per
`gpu_id` from `ModelRunner.init_shared_mooncake_transfer_engine()`, so tp_size=8
means **8 scheduler processes, 8 engines, 8 libfabric domains** — not one shared
engine. Confirmed in a real K3 prefill log: 8 distinct pids each enumerate all 18
RDMA devices and listen on their own RPC port, and every bucket of the size
histogram divides by 8 exactly.

This matters because `MC_MAX_CONCURRENT_REG_MR` caps **each process**. On a tp8
node `cap=16` means 128 registrations in flight globally. An earlier
single-process replay (1456 buffers in one domain) gave the opposite answer for
this reason and should be disregarded.

Ground truth from the B300 log: the kv batch has 1376 buffers, peaks at **1104
registrations in flight**, and takes **127.4 s** wall. The 8-process replay
unbounded lands at 98-128 s. The single-process one gave 239 s.

## Results (slowest rank, ms — the server cannot serve until every rank is done)

| per-proc cap | global | desc (size-descending) | layer (what SGLang passes) | asc (size-ascending) |
|---|---|---|---|---|
| unset | ~1100 | 108.0 / 98.6 / 128.1 s | 112.4 s | 119.9 s |
| 128 | 1024 | 186.3 s | — | — |
| 64 | 512 | 183.8 / 208.5 s | 100.6 s | 60.0 s |
| 32 | 256 | 129.7 s | — | — |
| 16 | 128 | 95.2 / 97.9 / 96.7 s | **43.1 s** | **36.2 / 37.4 s** |
| 8 | 64 | 106.6 s | — | 44.1 s |
| 4 | 32 | 157.3 s | — | — |

Repeats are separate full sweeps, not reruns within one sweep, so they include
process startup and GPU allocation each time. The asc cap=16 repeat (36.2 then
37.4 s) puts the run-to-run spread at ~3%, well inside the 2.6-3.3x effect. By
contrast unbounded spans 98.6-128.1 s (~30%), which is why the earlier
"cap=16 beats unbounded by 13%" reading was not a result.

Raw sweep output for every row: `mcreg-cap-order-p5-8proc-sweeps.txt`.

## Findings

1. **Input order dominates the cap.** At cap=16 the same workload takes 96.7 s
   (desc) or 36.2 s (asc) — 2.7x from ordering alone. At cap=64 it is 208 s vs
   60 s, 3.5x.

2. **Order only matters when a cap is set.** Unbounded is 98-128 s regardless of
   order, because every buffer gets its own thread and there is no queue to
   schedule. This is exactly what a fixed pool pulling tasks by index predicts.

3. **Size-descending (longest-first) is the WORST order, not the best.** This is
   backwards from LPT scheduling intuition and is the single most surprising
   result here. Hypothesis (untested): the 391 MB and 220 MB registrations
   saturate some shared resource — the per-domain provider lock, page-pinning
   bandwidth, or PTE installation — so filling every worker slot with a large
   buffer first serializes on that resource, whereas starting with small buffers
   lets many complete per unit of contention. Needs verification before being
   stated as fact.

4. **There is an optimum, and it is near cap=16 for both orders.** On asc the
   sweep is 60.0 s (64) -> 36.2 s (16) -> 44.1 s (8); on desc it is 183.8 s (64)
   -> 95.2 s (16) -> 106.6 s (8) -> 157.3 s (4). Both curves bottom out at the
   same cap, so the cap and the order are largely independent effects: the cap
   sets how much the provider can overlap, and the order sets how efficiently
   that overlap is used. cap=16 x 8 processes = 128 in flight per node, i.e.
   4 per NIC on 32 NICs.

5. **The best configuration is a cap plus a favorable order** (36-43 s), which is
   2.6-3.3x faster than unbounded. But a cap with the *unfavorable* order
   (96-208 s) is at best a wash and at worst 2x slower than unbounded. The
   earlier "cap=16 helps by 13%" reading on the desc order was inside the noise
   band of unbounded.

## Implication for MC_MAX_CONCURRENT_REG_MR

The knob is real but incomplete: on its own, its effect ranges from 2.6x speedup
to 2x slowdown depending on an input order the caller controls and Mooncake never
documents. Keeping the default unbounded (current behavior) is right — a
compiled-in cap would inherit whatever order the caller happened to use.

**For SGLang specifically the knob is worth setting today.** The `layer` column
is the order SGLang really passes (`get_contiguous_buf_infos()` groups by pool,
not by size), and there `MC_MAX_CONCURRENT_REG_MR=16` is 112.4 s -> 43.1 s, a
2.6x cut in registration time at startup, with no code change on either side.
That is the one recommendation this data supports without further work.

The larger finding is that `runBoundedParallel` should not consume tasks in
caller order when a cap is set. Sorting ascending by length inside the batch
entry point would make the good case the default and remove the 2.7-3.5x order
penalty, without asking operators to know any of this. Two caveats before acting:
the mechanism behind "large-first is bad" is still unverified (finding 3), and
the B300 table that motivated the original default of 8 was collected at a
different NIC count (16 vs 32) and, being a full K3 server run, used the layer
order — so it needs re-measuring with this harness before the two platforms can
be reconciled.

## Reproduce

`regbench_k3_mp.py`, `hostrun_k3_mp.sh`, `capsweep_k3_mp.sh` (saved alongside
this file). Raw per-rank logs in `mcreg-cap-order-p5-8proc-logs.tgz`, raw sweep
output in `mcreg-cap-order-p5-8proc-sweeps.txt`.

The host-memory comparison quoted above (single process, 128 x 2 GiB of 4 KB-page
host memory, `fi_mr_reg`) is a different harness; its raw output is in
`mcreg-cap-p5-hostmem-sweep.txt`. Note the sweep ran caps out of order
(8, 128, 32, 4, 64, 16) specifically so the curve could not be an artifact of
monotonically warming or fragmenting state, and cap=16 still won.

```
REGBENCH_ORDER=asc TAG=ord_asc_ SCALE=1.0 NRANKS=8 \
  CAPS="64 16 unset" OUT=/tmp/sweep.txt ./capsweep_k3_mp.sh
```
