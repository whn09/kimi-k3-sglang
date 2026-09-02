# Experiment 5: registration cost is a function of what is already registered

P5-1 (p5.48xlarge, H100, 32 EFA NICs), 2026-07-31.
Raw: `mcreg-exp5-accumulation.txt` (192 timed registrations).

**This closes gap #1 of PR #3210** — the "largest-first is the worst order"
mystery, open since 2026-07-30.

## Hypothesis

One mechanism would explain both open observations at once: **the cost of
registering a buffer grows with the total bytes already registered on that
domain.** Largest-first would then be worst because the big buffers go in early
and inflate every registration after them; and a worker's second registration
would be dearer than its first because by then the domain holds much more.

## Method

`regbench_serial.py <ip> <mode>` — one buffer per `batch_register_memory()`
call, **strictly sequential, single thread**. No concurrency means no lock
contention and no queueing, so any growth in the curve is a property of the
domain's state rather than of scheduling. Needs no instrumentation and no code
change.

Four modes separate cumulative bytes from cumulative count:

| mode | sequence |
|---|---|
| `large` | 48 × 391 MB |
| `small` | 48 × 2.79 MB |
| `l2s` | 24 × 391 MB, then 24 × 2.79 MB |
| `s2l` | 24 × 2.79 MB, then 24 × 391 MB |

`l2s` vs `s2l` is the control: same 48 buffers, same 9.23 GiB total, only the
order differs. It isolates *position* from *size*.

## Result: the same buffer costs 36× more when registered late

The identical 2.79 MB buffer, timed in both positions:

| | prior bytes | avg cost | n |
|---|---|---|---|
| `s2l` idx 0–23 (small first) | 0.00 GiB | **70.0 ms** | 24 |
| `l2s` idx 24–47 (small last) | 9.17 GiB | **2505.2 ms** | 24 |

Meanwhile the identical 391 MB buffer costs the same in both positions, because
in both it is preceded by ~0 bytes:

| | prior bytes | avg cost | n |
|---|---|---|---|
| `l2s` idx 0–23 (big first) | 0.00 GiB | 1382.4 ms | 24 |
| `s2l` idx 24–47 (big after smalls) | 0.06 GiB | 1407.5 ms | 24 |

So cost is **not** a function of the buffer's own size in any dominant way — it
is a function of the bytes already registered. Confirming that the two orders are
otherwise identical: both end at exactly 9913540608 cumulative bytes, and the
final registration costs 2421 ms (`l2s`) vs 2422 ms (`s2l`).

## The growth is linear in prior bytes

`large` mode, cost against cumulative bytes:

| idx | cumulative | cost | cost / prior GiB |
|---|---|---|---|
| 0 | 0.38 GiB | 143.5 ms | 375.6 |
| 8 | 3.44 GiB | 897.6 ms | 261.0 |
| 16 | 6.50 GiB | 1681.2 ms | 258.8 |
| 24 | 9.55 GiB | 2458.4 ms | 257.4 |
| 32 | 12.61 GiB | 3344.8 ms | 265.3 |
| 40 | 15.67 GiB | 4430.6 ms | 282.8 |

~260 ms per already-registered GiB, flat across the range, i.e. **cost ≈ k ×
prior_bytes** with k ≈ 260 ms/GiB. The first registration is 143 ms and the 48th
is 5022 ms — a 35× spread for identical work.

`small` mode stays at 44–70 ms throughout (only 128 MiB accumulates in total),
confirming it is bytes and not *count* that drives the cost. One 451 ms outlier
at idx 40 is unexplained but isolated.

## Why this makes ascending order optimal, and it is not a scheduling effect

Total serial time for the same 48 buffers, purely from ordering:

| mode | total |
|---|---|
| `small` (128 MiB) | 3.7 s |
| `s2l` (ascending-ish) | **35.5 s** |
| `l2s` (descending-ish) | **93.3 s** |
| `large` (18.8 GiB) | 121.2 s |

`s2l` vs `l2s` is **2.6× on identical work**, with one thread and no queue. This
is the same effect the 8-process runs saw as "order matters more than the cap"
(desc 101.3 s vs asc 32.8 s, 3.1×) — so that was never about longest-processing-
time scheduling at all, which is why LPT intuition gave the wrong prediction.

With cost = k × prior_bytes, the total for a fixed set of buffers is minimized by
registering in **ascending size order**: each large buffer's expensive
registration then sits behind as few prior bytes as possible. Descending order
does the opposite. This is a well-defined optimum, not a heuristic.

## Consistency with earlier experiments

- Explains the cap=24 (23.0 s) vs cap=47 (5.5 s) gap that queue depth alone
  could not (4.2× where only 2× was available): a worker's second registration
  happens when the domain is far fuller, so it is much more expensive than its
  first. Not a separate phenomenon.
- Consistent with experiment 1a's "cost linear in bytes ~6.2 ms/MB". That
  measurement swept size at fixed count, so prior-bytes rose with size and the
  apparent per-byte cost of a buffer was really the accumulation term.
- Does **not** conflict with experiment 4 (`cap × nranks ≈ cores`). Accumulation
  sets the cost of each registration; the CPU budget sets how many can proceed at
  once. They compose, which is why capping and sorting were measured as
  independent effects.

## Consequence for PR #3210: the fix is the sort, not the knob

The PR's stated reason for not sorting was that the mechanism was unknown. It is
now known, and it directly implies ascending order is optimal. Recommended:

1. **Sort ascending by length inside `registerLocalMemoryBatch()`.** Already
   measured out-of-tree at 3.1× on descending input, landing level with
   pre-sorted input. It removes the order penalty for every caller instead of
   only those who opt in, and needs no configuration.
2. Keep `MC_MAX_CONCURRENT_REG_MR` as the opt-in CPU-budget knob with the
   `cores / nranks` guidance — it is an orthogonal, smaller, and still real win.
3. The deeper fix is upstream: ~260 ms per prior GiB suggests the provider walks
   or rebuilds per-domain state on each registration. Worth an `efa`/libfabric
   issue, but out of scope here.
