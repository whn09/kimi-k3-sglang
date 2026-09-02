# Experiment 7: the NIXL gap is the NIC fan-out, and so is the accumulation

P5-1 (p5.48xlarge, H100 x8, 32 EFA NICs), 2026-07-31.
Raw: `mcreg-exp7-nic-fanout.txt`, `mcreg-exp6-host-vs-gpu.txt`.

## Question

Why is NIXL's GPU registration so much faster than Mooncake's, when both call
`fi_mr_reg` through libfabric on the same hardware?

## What the two codebases actually do

Reading NIXL's libfabric plugin:

- `nixlLibfabricRailManager::selectRailsForMemory()`
  (`src/utils/libfabric/libfabric_rail_manager.cpp:735`) branches on memory
  type. For `VRAM_SEG` it queries the buffer's PCI bus ID via
  `cudaQueryAddr()` and returns **only that GPU's topology-local rails**
  (`getEfaDevicesForPci`). For `DRAM_SEG` it returns all rails.
- Mooncake's `EfaTransport::registerLocalMemoryInternal()` puts a single-chunk
  buffer on **every** context: `for (n = 0; n < num_nics; ++n)
  nic_assignments[0].push_back(n)`.

On p5.48xlarge that is 32 NICs vs 4 (32 NICs / 8 GPUs). So for the same KV
cache, Mooncake issues 8x as many `fi_mr_reg` calls as NIXL — and it issues
them serially per buffer, because `use_parallel_reg` is 0 for VRAM (pre-touch
is skipped on device memory, and the NIC-parallel path requires it).

## Measurement: 48 x 391 MB GPU buffers, strictly serial

Restricting Mooncake's device list to emulate NIXL's rail selection:

| NICs/buffer | total | 1st reg | 48th reg | avg |
|---|---|---|---|---|
| 32 (Mooncake today) | **123.7 s** | 128.1 ms | 5013.6 ms | 2577.9 ms |
| 4 (what NIXL does for VRAM) | **17.6 s** | 17.7 ms | 595.3 ms | 367.2 ms |
| 1 | 4.4 s | 5.8 ms | 150.1 ms | 91.1 ms |

**7.0x from the fan-out alone**, and it scales almost exactly linearly with the
NIC count (123.7/4.4 = 28x for 32x the NICs). This is a much bigger lever than
either the concurrency cap (2.6x) or the sort (3.1x).

It also reframes experiment 5: the ~260 ms/prior-GiB accumulation is **per
domain**, so registering on 32 domains pays it 32 times. At 4 NICs the same
curve is still there (17.7 ms first vs 595 ms last, 34x) but every term is 8x
cheaper. The accumulation is not separate from the fan-out — the fan-out
multiplies it.

## Host memory does not accumulate at all (experiment 6)

Experiment 5 used `cuMemAlloc` buffers only, so "cost grows with bytes already
registered" was so far a claim about device memory. Repeating it on `mmap`'d
host memory (`regbench_serial_host.py`, same 48-buffer sequences):

| | GPU (exp 5) | host (exp 6) |
|---|---|---|
| 391 MB, 1st of 48 | 143 ms | 696 ms |
| 391 MB, 48th of 48 | 5022 ms | ~580 ms |
| 2.79 MB after 0 GiB | 70 ms | 25 ms |
| 2.79 MB after 9.17 GiB | 2505 ms | 130 ms |
| `s2l` (asc) total | 35.5 s | 16.6 s |
| `l2s` (desc) total | 93.3 s | 14.0 s |

Host is **flat**: a 391 MB buffer costs the same whether it is 1st or 48th, and
the two orders come out within run-to-run spread — in the first pass descending
was even marginally faster. A control that allocates the identical sequence in
both modes and varies **only** the registration order (`regbench_serial_host2.py`,
2 reps) confirms there is no order effect on host memory:

| rep | asc | desc |
|---|---|---|
| 1 | 15.3 s | 13.2 s |
| 2 | 14.2 s | 15.7 s |

Direction flips between reps, so the difference is noise.

Consequences:

- The accumulation is a **device-memory** property, most likely the nvidia
  peer-memory / dmabuf path rebuilding per-domain state, not a generic
  libfabric `fi_mr_reg` cost. A libfabric issue should say "CUDA/HMEM
  registration" specifically.
- Sorting ascending is a **no-op** for host memory rather than a regression,
  which is why the sort in `perf/efa-sort-mr-reg-by-size` is applied
  unconditionally instead of only for `VRAM`.
- CPU-to-CPU registration is ~8x cheaper per byte to begin with (host 48 x
  391 MB = 28.0 s vs GPU 121.2 s), so neither the sort nor the cap matters much
  there.

## Where this leaves the two open PRs

The fan-out is the dominant term and is not addressed by either PR. Ranked by
measured effect on GPU registration:

| lever | effect | status |
|---|---|---|
| per-GPU NIC selection for VRAM (NIXL's approach) | **7.0x** | not implemented; see `project_sliding_window_mr` |
| `MC_MAX_CONCURRENT_REG_MR` | 2.6x on SGLang's order | PR #3210, draft |
| ascending sort | 3.1x on descending input, ~20% on SGLang's order | `perf/efa-sort-mr-reg-by-size`, uncommitted upstream |

The cap and the sort compose with the fan-out fix rather than being replaced by
it, but a 7x lever left on the table is worth saying out loud in #3210 before it
goes ready-for-review.

Caveat: registering a VRAM buffer on only 4 of 32 NICs changes which NICs can
serve a transfer for it, so it is a **throughput** decision, not a free win —
that is why it needs its own measurement (transfer bandwidth, not just
registration time) rather than being folded into either PR.
