# Experiment 4: the cap optimum is set by the CPU budget

P5-1 (p5.48xlarge, H100 x8, 32 EFA NICs, **192 cores**), 2026-07-31.
Raw: `mcreg-exp4-core-budget.txt`, logs `mcreg-exp4-core-budget-logs.tgz`.

## Question

The 8-process K3 replay peaks at `MC_MAX_CONCURRENT_REG_MR=16`. Since the cap is
per process, that is 8 x 16 = **128 registration threads on a 192-core node**,
while cap=32 gives 256 threads and is slower. Is the optimum a property of the
CPU budget, or of something per process?

Testable prediction: if CPU oversubscription is the mechanism, the optimum obeys
`cap_opt x nranks ~= cores` and must **move** when either side changes.

Experiment 1 could not answer this — it was single-process, so 48 threads on 192
cores never contended, and "more threads is faster" was guaranteed there.

## Method

`hostrun_k3_cores.sh` = the existing 8-process replay plus an optional
`taskset -c 0-(N-1)` applied to **all ranks sharing one cpuset**, so they contend
with each other exactly as they do for the whole machine, just with a smaller
budget. (Giving each rank a disjoint slice would have removed the contention
being measured.) Real K3 per-rank mix, `REGBENCH_ORDER=desc`, scale 1.0,
slowest rank = the node's wall clock.

## Result

Optimum in bold. `threads` = cap x nranks = global registration concurrency.

| config | nranks | cores | cap | threads | slowest | mean |
|---|---|---|---|---|---|---|
| A | 8 | 192 | 8 | 64 | 106.7 s | 101.7 s |
| A | 8 | 192 | **16** | **128** | **99.8 s** | 92.7 s |
| A | 8 | 192 | 32 | 256 | 127.8 s | 118.7 s |
| A | 8 | 192 | 64 | 512 | 182.6 s | 177.1 s |
| B | 4 | 192 | 8 | 32 | 109.1 s | 105.0 s |
| B | 4 | 192 | 16 | 64 | 100.4 s | 98.7 s |
| B | 4 | 192 | **32** | **128** | **89.2 s** | 85.4 s |
| B | 4 | 192 | 64 | 256 | 120.6 s | 117.2 s |
| C | 8 | 64 | 4 | 32 | 155.1 s | 153.2 s |
| C | 8 | 64 | **8** | **64** | **121.3 s** | 113.5 s |
| C | 8 | 64 | 16 | 128 | 164.7 s | 155.2 s |
| C | 8 | 64 | 32 | 256 | 263.8 s | 258.7 s |

## Conclusion: confirmed, and the invariant is global threads ≈ cores

The optimal *cap* is different in all three configs — 16, 32, 8 — so it is not a
per-process constant. Convert to global threads and the three optima collapse:

| config | cap_opt | **threads_opt** | cores |
|---|---|---|---|
| A | 16 | **128** | 192 |
| B | 32 | **128** | 192 |
| C | 8 | **64** | 64 |

- **A → B**: halving the process count *doubled* the optimal cap, holding
  threads_opt at 128. The cap moved; the thread count did not.
- **A → C**: cutting cores 192 → 64 moved threads_opt 128 → 64, i.e. it tracked
  the cores, and dropped the optimal cap from 16 to 8.

`threads_opt ≈ cores` in C and `≈ 0.67 x cores` in A/B. Oversubscription is
punished hard and superlinearly — A cap=64 (512 threads, 2.7x oversubscribed) is
1.8x slower than the optimum, and C cap=32 (4x oversubscribed) is 2.2x slower.
Registration threads sit in page-pinning, which is CPU-bound kernel work, so
past the core count they only add scheduler pressure.

## Consequences for PR #3210

1. **The knob's correct value is derivable, not a guess.** The current docs say
   the best value "depends on the platform and on the order the caller passes
   buffers in" and tell operators to measure. That understates what is known:
   `cap ≈ cores / nranks`, i.e. ~0.7-1.0 x cores globally. On a tp8 192-core node
   that is 16-24, which is exactly what was measured.

2. **"Do not core-scale the value" is wrong and must be corrected.** That
   sentence was written from the single-process data and says the bottleneck is
   the provider lock rather than CPU. Experiment 1b refuted the lock (23x speedup
   from the cap), and this experiment shows the value *is* core-scaled — it is
   the one thing it should scale with. This is the one docs change the data
   forces.

3. **A default is now defensible, but needs the process count.** `cores / nranks`
   is only computable if the transport knows how many peer processes share the
   node, which a single `TransferEngine` does not. Absent that, unbounded remains
   the only safe default and the knob stays opt-in — but the docs can now tell
   operators how to compute it instead of asking them to sweep.

4. Orthogonal to ordering. This is all at `order=desc`; experiment 2's internal
   ascending sort (desc 101.3 → 32.8 s) is a separate and larger effect, and the
   two compose — B's cap=32 optimum was found at desc, so an internal sort should
   move all of these down further.
