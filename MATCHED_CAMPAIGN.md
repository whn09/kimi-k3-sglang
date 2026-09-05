# Matched-capacity campaign, 2026-09-05 (4x B300, ap-northeast-2)

**All numbers live in `results/SYNTHESIS.txt`. Regenerate, never quote from
here:**

```
bash sync.sh pull                        # raw logs from the 4 hosts
python3 gen_synthesis.py results > results/SYNTHESIS.txt
```

The question: at the **same machine count**, does a PD-disaggregated K3 with
DeepEP v2 beat the same boxes running independent aggregated instances?

Driver `94_matched.sh` behind `/tmp/campaign.sh`. Ran 06:28:59 wall clock,
four stages, all `rc=0`, zero errors in any stage log. Every arm x workload has
1 discarded warmup + 2 timed replicates at 3 iso-load points
(`c = machines x {8,16,32}`). Both arms sit behind a pinned `round_robin`
router; `assert_cenv` reads each instance's env back out of its container and
SKIPs the arm on mismatch.

Read `SYNTHESIS.txt` top-down: the **inventory** block first (an arm absent
there was not measured), then the per-workload tables, then the **per working
box, matched load** block.

## What each table can and cannot answer

`SYNTHESIS.txt` has two kinds of ratio and they answer different questions.

The **deployment-level** ratio (the `x<arm>` columns) is what you deploy on. It
charges PD for the machine it hands to prefill, which is correct: that machine
is part of the deployment.

The **per working box, matched load** block is what measures the *backend*. Two
corrections it applies, both of which changed a conclusion here:

1. Divide by the boxes doing the bottleneck role, not by the machine count. A
   deployment-level ratio attributes a whole idle-ish prefill box to the decode
   result.
2. Compare at matched **concurrency per box**. PD concentrates the same total
   concurrency onto fewer boxes, so its boxes run fuller; comparing at equal
   *total* concurrency flatters it. The generator interpolates the baseline
   curve and refuses to extrapolate -- cells outside the baseline's measured
   range print `n/a`, they are not estimates.

## Retracted

**`agg*:v2` is not a measurement of "DeepEP v2 vs no EP".** The v2 aggregated
arms ran chunked-prefill **1024** while their TP counterparts ran **16384** --
at isl=8192 that is 8 chunks vs 1. The cause is in `env_common.sh:342-346`:
`STANDALONE_CHUNK` defaults to 16384 for `MOE_A2A_BACKEND=none` but to
`STANDALONE_CAP` (default 1024) for v2, because v2 requires `CHUNK <= CAP`.
That is a default-value choice, not an EP cost.

Two conclusions drawn from those arms are therefore withdrawn -- they are the
same artifact seen from both sides:

- ~~v2 is ~0.20x of plain TP~~
- ~~PD separation is worth ~3.3x over aggregated~~

Rerun with `STANDALONE_CAP=8192` (which drags `STANDALONE_CHUNK` with it) to
get either number honestly. Watch for OOM on decode-graph capture: capacity
sizes the capture pool, not just the slab.

Also withdrawn: ~~"v2's decode path is ~0.6x the compute efficiency of TP"~~ as
originally derived. The 0.6 came from a deployment-level ratio with no per-box
normalisation -- it charged v2 for the box PD gives to prefill. The per-box
matched-load block reaches a similar figure by the correct route, so use that
block's number and its reasoning, not the original.

## Known-degenerate cells

- **osl=1 (prefill stage):** no inter-token interval exists, so `median_tpot_ms`
  is identically 0 and `out tok/s` is just the request rate. The generator
  refuses to print both rather than showing plausible-looking noise.
- **`agg4:tp` at c=32 in the decode stage:** 13.8% replicate spread. With
  `ppc=2` that bench is only ~2 waves. Judge that arm at c=64 / c=128.

## Not measured -- do not infer these

- `pd*:tp` -- PD with the plain-TP MoE path. `94_matched.sh:256` supports it
  and labels it a control. Without it, **PD's effect cannot be separated from
  v2's effect**, because every PD arm here runs v2.
- `agg*:v2` at a chunk matched to its TP counterpart (see Retracted).
- `agg2x2node:v2` -- aggregated EP=16 across 2 nodes. The layout exists in
  `94_matched.sh:146`. Without it there is no aggregated cross-node row.

The cross-node cost that *is* measured (`pd2x2` vs `pd2p2d`, same 4 machines,
same PD shape, same cap/chunk) conflates crossing a node boundary with
doubling EP degree 8 -> 16, and its mem_fraction differs (0.78 vs 0.85). Those
are inseparable in deployment -- a box has 8 GPUs, so EP=16 is necessarily
cross-node -- so the ratio is the right number to deploy on, but it is not a
clean measurement of the wire.
