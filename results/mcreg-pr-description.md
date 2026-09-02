# PR description draft: MC_MAX_CONCURRENT_REG_MR

Detail moved out of `docs/source/design/transfer-engine/index.md`, which keeps a
one-paragraph bullet matching its neighbors. Paste the relevant parts into the PR
body when opening it. Data source: `mcreg-cap-order-p5-8proc.md`.

## Problem

`registerLocalMemoryBatch()` / `unregisterLocalMemoryBatch()` spawn one
`std::async(std::launch::async)` per buffer with no cap, which libstdc++ takes
literally — one fresh thread each. Kimi-K3 registers ~180 KV buffers per TP rank
in a single batch, and with one `TransferEngine` per rank an 8-rank node peaks at
~1100 threads all inside `fi_mr_reg` / `fi_mr_regattr`.

## Change

Both fan-outs become a fixed pool pulling from a shared index, sized by
`MC_MAX_CONCURRENT_REG_MR`. Default 0 = unbounded, so behavior is unchanged
unless an operator sets the knob: with no cap the pool spawns `count-1` threads
and runs the caller as a worker, matching what `std::async` did. Error semantics
unchanged — every item is still attempted, first non-zero return propagates.

## Measurements

p5.48xlarge (32 NICs), replaying K3's KV registration the way SGLang issues it:
8 processes, one engine and one GPU each, 182 GPU buffers of 2.5 KB to 391 MB per
process, barrier-synchronized. Times are the slowest rank, since nothing serves
until every rank has registered.

| per-process cap | in flight per node | size-descending | grouped by pool (what SGLang passes) | size-ascending |
|---|---|---|---|---|
| unset (unbounded) | ~1100 | 108 / 99 / 128 s | 112 s | 120 s |
| 64 | 512 | 184 / 208 s | 101 s | 60 s |
| 16 | 128 | 95 / 98 / 97 s | **43 s** | **36 / 37 s** |
| 8 | 64 | 107 s | — | 44 s |
| 4 | 32 | 157 s | — | — |

Repeated entries are separate sweeps (~3% run-to-run; unbounded's own spread is
~30%, which is why an earlier "cap helps by 13%" reading was not a result).

- A cap of 16 cuts registration **2.6×** on the order SGLang really uses
  (112 s → 43 s), but only 1.1× if buffers arrive largest-first.
- **Order alone** moves the capped result by 2.7× (cap 16) to 3.5× (cap 64).
- Unbounded is insensitive to order (99–128 s) because every buffer gets its own
  thread and nothing queues.
- The optimum is cap≈16 for *both* orders, so cap and order are largely
  independent: the cap sets how much the provider can overlap, the order sets how
  efficiently that overlap is used.

Largest-first being the *worst* order is the opposite of what
longest-processing-time scheduling predicts. **Unexplained** — hypothesis is that
the 391 MB / 220 MB registrations saturate a shared resource (per-domain provider
lock, page-pinning bandwidth, or PTE installation), so filling every worker slot
with a large buffer serializes on it. Not verified, so this PR does **not**
reorder the caller's buffers.

## Why no compiled-in default

The optimum tracks the platform:

- p5 host memory (`fi_mr_reg`, 128 × 2 GiB of 4 KB pages): 274 s unbounded,
  120 s @64, 89.6 s @32, **59.1 s @16**, 70.9 s @8, 257 s @4. (Sweep ran caps out
  of order — 8, 128, 32, 4, 64, 16 — so the curve is not a warming artifact.)
- 2× p6-b300, 16 NICs/rank, full K3 server run: 138.7 s unbounded, 130.1 s @128,
  51.0 s @32, **20.7 s @8**, 24.8 s @4.

The B300 run used a different NIC count *and* a different buffer order, so read it
as "the optimum is platform-specific", not as a second opinion on the same setup.

## Caveats to disclose in the PR

1. The mechanism behind "large-first is bad" is unverified.
2. The B300 numbers have not been re-measured with this 8-process harness, so the
   two platforms are not yet reconciled.
3. Follow-up worth considering: sorting ascending by length inside the batch entry
   point would make the good case the default and remove the order penalty
   entirely — but it should wait on (1).

## Housekeeping before opening

- Branch is behind `origin/main` (27 commits as of 2026-07-27) — rebase first.
- Use `.github/pull_request_template.md`; Module: Transfer Engine; disclose AI
  assistance.
- Checked for overlapping PRs: #2962 touches a different file, no other `reg_mr`
  PRs.
