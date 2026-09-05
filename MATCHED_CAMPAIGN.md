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

## Stage E, 2026-09-05: one arm survived, the hosts did not

`95_matched_followup.sh` launched at 10:17Z. `pd1p1d:tp` completed all 9 runs.
`agg2:v2 @ STANDALONE_CAP=8192` **launched successfully** -- `cap=8192
chunk=8192` on both boxes, no OOM, so the capture pool does fit at 8192 -- and
then all four B300 were terminated (`StateTransitionReason: User initiated`)
before a single bench ran. `agg2x2node:v2` never started.

The bench JSONs live on the hosts until `sync.sh pull`, so `pd1p1d:tp`'s JSONs
died with them. `salvage_log_json.py results/campaign2.E.log results` rebuilds
them from the driver log, which prints each block under its full tag -- the same
string as the JSON stem. Those rows are marked **SALVAGED** in the inventory:
real measurements, but not re-pullable and not re-checkable.

**`pd1p1d:tp` answers Q1: PD separation itself costs 13-18% of throughput, and
buys a 2.6x better TPOT.** Against `agg2:tp` its out tok/s ratio is
0.869 / 0.867 / 0.817 at c=16/32/64. The decomposition against the published
`pd1p1d:v2` is exact -- 0.817 x 0.855 = 0.698, and likewise at the other two
loads -- so of the 0.698 that `pd1p1d:v2` scores, **PD owns -18% and DeepEP v2
owns -15%**. PD's implementation is not the problem.

It also gives the cleanest v2-decode number in the campaign, because only the
MoE backend changes: same PD shape, same `cap=512 chunk=512` decode role, same
mem 0.92/0.85, same ep=8. Median TPOT `pd1p1d:tp` 4.7/5.1/5.6 ms vs
`pd1p1d:v2` 7.6/9.0/9.1 ms => v2 is **1.6-1.76x slower per token**, inverse
0.57-0.62, independently consistent with the per-box block's 0.629/0.649/0.665.

Still open after the termination: Q2 (`agg2:v2` at a matched chunk, and its
2-machine counterpart) and the aggregated cross-node row.

## PD's TTFT is the decode KV pool, not admission and not Mooncake

Asked and answered 2026-09-05 from `pd1p1d:v2`'s own decode/prefill server logs
(616 `.log` files in `results/`; Mooncake is byte-identical between the `:tp` and
`:v2` arms, so the finding covers both).

**Not the `max_running_requests=48` wall.** Decode `#queue-req` was 0 in
**5067/5067** samples at c=64, `#running-req` never exceeded **29**, and
`#retracted-req` was 0.

**Not Mooncake.** Prefill `#inflight-req` -- requests with a KV transfer actually
in flight -- has max 2/2/3 and mean 1.8/1.8/1.9 at c=16/32/64. A 4x load range
moves it by one request; a transport that were the bottleneck would back up.
Bandwidth agrees: KV is 6.87 GB / 266816 tokens ~ 26 KB/token, so a 8192-token
request is ~232 MB and 2.47 req/s is **~0.6 GB/s** aggregate over 16 EFA NICs.
The fixed cost is separately measured: at isl=128, where KV is 3.2 MB and
transfer time is noise, PD's TTFT is still 831.6 ms against `agg4:tp`'s 147.2 ms
-- **~684 ms of protocol**, which is **~4%** of the 16.0 s TTFT at isl=8192.

**It is the KV pool.** `#tokens: 266816` / 9216 per request (8192 in + 1024 out)
= **28.9**, exactly the observed `#running-req` ceiling of 29, with
`full token usage` p90 = **0.94** at both c=32 and c=64 (0.52 at c=16). A
saturated pool cannot pre-allocate, so decode withholds the bootstrap handshake
and prefill's requests wait in `#bootstrap-req`: nonzero in 3% / 56% / **79%** of
samples at c=16/32/64, mean 0.0 / 1.4 / **14.5**, max 36, mirrored by decode's
`#prealloc-req` (max 36). That wait is upstream of any KV transfer and lands
entirely inside TTFT.

**Compounded by prefill saturation.** The prefill-only ceiling is ~21.8k in tok/s
per box (65150.2/3 = 21717; 43699.9/2 = 21850). `pd1p1d` mixed at c=64 reaches
20241 -- **92.8%** of its single prefill box's own ceiling -- and prefill
`#queue-req` is nonzero in 61% of samples (max 28). So even a perfect pool leaves
1P1D prefill-bound at c=64.

**The headroom is idle.** Decode ledger, per GPU, all lines from one log:

```
265.83 avail -> weights 213.79 -> 52.04 -> draft model 1.06 -> 50.98
KV 6.87 + mamba 4.32           -> 38.26   (Memory pool end)
target-verify CUDA graph 15.79  -> 20.37
DeepEP symmetric pool 4 GiB, draft graph 0.03 -> 14.93 avail at steady state
```

~14.9 GB of the 0.15 slack is never touched. Each +0.01 of `mem_fraction_static`
moves ~2.66 GB into the pool (~100k tokens, ~11 more resident requests at this
shape); 48 resident needs ~11.8 GB, i.e. ~0.868. Mamba is not binding (usage max
0.47 of 64 slots).

`DECODE_CAP` cannot buy the same memory: 512 -> 128 frees 8.57 GB of capture pool
but `max_running_requests x (SPEC_BLOCK_SIZE+1) <= DECODE_CAP` then forces
MAXRUN ~ 16, **below** the 29 we already have. The levers are mem-fraction and
`cuda_graph_max_bs`.

`95_matched_followup.sh` **stage G** tests it: `pd1p1d:v2` at D mem-fraction
0.90 / 0.88 / 0.86, stepping down only on a failure to boot, against the
published (unstamped = 0.85) row. Not yet run -- it needs machines.

## Harness changes made alongside, 2026-09-05

All motivated by something that actually went wrong here.

- **`94_matched.sh` pulls after every arm** (`PULL_PER_ARM=0` to disable). Stage E
  lost nine completed runs because JSONs sat on hosts until an end-of-campaign
  pull. The pull is scoped to that arm's hosts and non-fatal.
- **Mem-fraction is per-role and in the filename.** `env_common.sh` honoured
  `MEM_FRACTION` (role-blind: hits prefill and decode alike, and they want
  opposite values) but silently discarded `DECODE_MEM_FRACTION`, because the
  PROFILE table assigns it with a plain `=`. Now a caller's value wins, is
  validated as a decimal in (0,1), and is forwarded to the one launcher that
  reads it. Critically, the run tag gains `-dmf0.88`/`-pmf`/`-smf` when a
  fraction leaves its profile value -- **without that, a rerun at 0.88 writes the
  published 0.85 row's exact filenames and `sync.sh pull` overwrites the
  baseline.** `gen_synthesis.py` reads the stamp into the arm label
  (`pd1p1d:v2/d0.88`), so a stamped and an unstamped run cannot merge into one
  cell as extra replicates. An unstamped name means the profile value.
- **`note_kv_pool`** prints, per decoding instance, the pool in tokens and the
  residency it implies against MAXRUN, flagging `POOL BINDS FIRST`. This is the
  admission limit `max_running_requests` does not show, and it took six days of
  log archaeology to find once.
- **`wait_ready` exits on a died container** (`WAIT_CONTAINER`, `docker inspect
  -f '{{.State.Status}}'`, plus the last 15 log lines). An instance that OOMs in
  capture at t=3min used to burn the full 30-minute timeout -- unaffordable for a
  ladder whose whole point is to find where OOM starts.
- **`snapshot_log` does `mkdir -p`** on the remote `results/`. `sync.sh push`
  excludes `results/`, so on a fresh host the shell redirect failed before docker
  ran and every snapshot of the first arm silently produced nothing.
- **Role-scoped resolution echoes** (`K3_ROLE`): a decode container's boot log no
  longer carries `NNODES=1: PREFILL_MEM_FRACTION 0.85 -> 0.92`.
- `gen_synthesis.py` column widths now follow the longest arm label, which also
  fixes a pre-existing 2-char-per-column drift between header and data.

## Not measured -- do not infer these

`95_matched_followup.sh` runs exactly these three arms. It gates on host
idleness first, because `94_matched.sh` calls `teardown_all` on every host it
uses at line 269, **before** its own `preflight_free_gpus` at 274 -- so that
preflight cannot protect a foreign container, and the gate has to be outside.
It also reads the chunk back out of each launcher's echo, scoped to that arm's
`##########  arm=... a2a=...` block: `assert_cenv` checks cap but **not chunk**,
and chunk is the axis that invalidated the arms below.

- `pd*:tp` -- PD with the plain-TP MoE path. `94_matched.sh:256` supports it
  and labels it a control. Without it, **PD's effect cannot be separated from
  v2's effect**, because every PD arm here runs v2. `pd1p1d:tp` is a clean
  control: at isl=8192 its prefill role is single-chunk too (`A2A=none` puts
  `PREFILL_CHUNK` at 16384, v2 at `PREFILL_CAP`=8192) and both decode roles are
  `cap=512 chunk=512`.
- `agg*:v2` at a chunk matched to its TP counterpart (see Retracted). Fixed
  with `STANDALONE_CAP=8192`, which drags `STANDALONE_CHUNK` with it.
  **`agg2:tp` does not need rerunning:** `build_deepep_envs` returns before it
  ever reads `cap` when `MOE_A2A_BACKEND=none`, so `STANDALONE_CAP` is a no-op
  on a TP arm, and 16384 vs 8192 is one chunk either way at isl=8192.
- `agg2x2node:v2` -- aggregated EP=16 across 2 nodes. The layout exists in
  `94_matched.sh:146`. Without it there is no aggregated cross-node row.

Capacity sizes the DeepEP slab, `masked_max_m` **and** the decode-graph capture
pool, so `STANDALONE_CAP=8192` can OOM where 1024 booted. The follow-up driver
steps down 8192 -> 4096 -> 2048; if it lands below 8192, say so with the number,
because 4096 is 2 chunks at isl=8192 and 2048 is 4.

## Admission

`audit_admission.py results` names the cells where an instance **sat** at its
`max_running_requests` ceiling, read from the scheduler's own `#running-req`.
Do not use the bench JSON's `concurrency` for this: it is
`sum(e2e_latency)/duration`, a time average that ramp-up and drain depress on
every short bench, so it flags healthy cells (0.66-0.75 on the 15-55 s decode
benches). Across all 33 cells of this campaign exactly one is contaminated --
`isl128/osl1024 pd2p2d c=128` -- and `gen_synthesis.py` already drops it from
every per-box ratio as outside the baseline's measured range.

The cross-node cost that *is* measured (`pd2x2` vs `pd2p2d`, same 4 machines,
same PD shape, same cap/chunk) conflates crossing a node boundary with
doubling EP degree 8 -> 16, and its mem_fraction differs (0.78 vs 0.85). Those
are inseparable in deployment -- a box has 8 GPUs, so EP=16 is necessarily
cross-node -- so the ratio is the right number to deploy on, but it is not a
clean measurement of the wire.
