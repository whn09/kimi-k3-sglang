# Matched-capacity plan: does PD disaggregation beat the same machines run separately?

The claim we have to be able to defend, in one sentence:

> At the same number of B300 machines, a PD-disaggregated Kimi-K3 deployment
> (DeepEP v2 + Mooncake, everything cross-node on EFA) delivers more input **and**
> output throughput than those machines run as independent single-node servers —
> or, if it does not, it loses by a stated and small margin.

Everything below exists to make that a falsifiable statement rather than a
comparison of two numbers taken on different days.

---

## 0. Why the numbers already in the README cannot answer it

| what we have | why it does not settle the claim |
|---|---|
| pd2x2 (4 machines): **1421.72** out tok/s @ c32 | one standalone node at c32 does **1441-1506**. Read literally, 4 machines lose to 1. |
| 2P2D (4 machines): **1272.94** out tok/s @ c32, achieved conc **29.61** | decode-admission-bound at 32 slots (`DECODE_CAP=128`), and still client-limited |
| standalone low-latency: 1441.2 / 1506.5 @ c32 | measured on ONE machine, no router hop, older MoE path (`flashinfer_mxfp4`) |

Three separate defects, and each one alone is enough to invert the answer:

1. **32 in-flight requests cannot load 4 machines.** Both arms were limited by the
   client, so the win goes to whichever arm has the lower per-request latency,
   which is a latency result wearing a throughput label.
2. **The baseline was never actually run as a baseline.** "4x a single-node number"
   is not a measurement: four independent runs have four different durations, so
   their per-second numbers do not add, and the PD row carries a router hop the
   single-node row does not.
3. **The arms differ on more than P/D.** Different image generation, different MoE
   backend, different day.

`94_matched.sh` fixes all three by construction. What is left to do is run it.

---

## 1. What "matched" means here (the harness enforces every line)

- **Same machine count per comparison**, stamped in every filename as `m<N>`.
  A 2-machine and a 4-machine run are otherwise identical in every field and
  would overwrite each other.
- **Load scales with machines**: `c = machines * CONC_PER_MACHINE`, default
  `8 16 32` per machine. Both arms of a comparison have the same machine count,
  so they get the same `c`; per-machine columns stay comparable across machine
  counts. `c = 32 * machines` is deliberately *above* the decode-slot ceiling
  (2 decode nodes at `DECODE_CAP=512` = 48+48 = 96 slots) so at least one point
  is a saturation reading and not a latency reading.
- **Both arms behind a router**, both routers pinned to `round_robin`
  (`23_launch_agg_router.sh` is new, and exists only so the baseline pays the
  same hop). A single-instance arm (`agg1`, `uni2node`) carries one hop fewer and
  is therefore a *reference row*, never the baseline in a claim.
- **Same image ID** — not tag — asserted on every host before anything is torn
  down. `IMAGE` now defaults to `kimi-k3-efa-v2:nccl2312`; `:latest` is the
  NCCL-2.30.7 build where GIN type 5 cannot initialise at all.
- **Radix cache off in every arm** (`DISABLE_RADIX=1`, newly reachable — see §6).
  A cache that helps one arm and not the other is invisible in the output.
- **Every launched instance's env read back out of the container**: `NNODES`,
  `TP_SIZE`, `MOE_A2A_BACKEND`, and for v2 also `EP_SIZE`, `DEEPEP_V2_MODE`,
  `NCCL_GIN_TYPE`, dispatch cap. A mismatch **skips the arm** rather than
  benchmarking it — an arm that says "cross-node EP over EFA" and quietly ran
  ep=8 on NVLink has already happened here, and the log said `ep=8` while the
  intent was 16.
- **Warmup replicate r0 is discarded**, always. deep_gemm JIT-compiles inside the
  timed region on the first bench at a given shape and faked a 78% effect once.
- **The table is generated, never transcribed** (`gen_matched_table.py`): the
  PD-vs-baseline ratio is computed from the JSON against a baseline row selected
  by the same `(isl, osl, conc, machines)` key.

## 2. Baselines, and why the strongest one runs first

The single-node baseline family is **two** arms, not one:

| arm | MoE path | chunked prefill | why it is in the campaign |
|---|---|---|---|
| `agg4:tp` | `MOE_A2A_BACKEND=none` (plain TP) | **16384** | the strongest single-node adversary, and the honest one |
| `agg4:v2` | DeepEP v2, ep=8 `direct` | 1024 | apples-to-apples on the MoE path with the PD arms |

A unified v2 server is stuck at `CAP=CHUNK=1024` because decode graph capture has
to fit, while plain TP gets `CHUNK=16384`. If we only ran `agg4:v2` we would be
beating a straw man; if we only ran `agg4:tp` we could not separate "PD wins" from
"v2 wins". `agg4:tp` runs **first** so that if PD loses, we already know what it
lost to. `gen_matched_table.py` picks the **stronger** of the two as the baseline
unless `--baseline` pins one.

## 3. The arm x workload matrix

Shapes (`WL`, resolved in `env_common.sh`; `n = c * ppc`):

| `WL` | ISL / OSL | ppc | what it isolates | how to read it |
|---|---|---|---|---|
| `mixed` | 8192 / 1024 | 2 | the deployment | **the headline.** both throughput axes |
| `prefill` | 8192 / 1 | 4 | prefill capacity only | input tok/s + TTFT. **TPOT/ITL are undefined at OSL=1** — ignore those columns, they are noise or absent |
| `decode` | 128 / 1024 | 2 | decode capacity only | output tok/s + TPOT. input tok/s is meaninglessly small |

### 4 machines (the headline experiment)

| order | arm | layout | a2a | machines |
|---|---|---|---|---|
| 1 | `agg4:tp` | 4x standalone TP=8 | none | 4 |
| 2 | `agg4:v2` | 4x standalone TP=8 | v2 ep=8 `direct`, gin 3 | 4 |
| 3 | `pd2p2d` | 2P + 2D, all TP=8 | v2 ep=8 `direct` + Mooncake/EFA KV | 4 |
| 4 | `pd2x2` | 1P over 2 nodes + 1D over 2 nodes | v2 **ep=16 `hybrid`, gin 5** + Mooncake/EFA | 4 |

`pd2x2` is the arm the user's "must use DeepEP v2 and maximize EFA" points at:
it is the only 4-machine layout where EFA carries **both** the expert a2a and the
KV transfer. `pd2p2d` keeps the a2a on NVLink and is the control that says how
much of the result is the cross-node a2a.

### 2 machines (the scaling control)

| arm | layout | machines |
|---|---|---|
| `agg2:tp` | 2x standalone | 2 |
| `agg2:v2` | 2x standalone | 2 |
| `pd1p1d` | 1P + 1D, TP=8 each | 2 |

This is the missing **matched-image 1P1D control**: the 1683 tok/s 1P1D figure in
the README was taken on the previous image generation and cannot be compared to
anything current.

### P:D ratio sweep (only after the headline is in)

The mixed shape at 8192/1024 puts ~8x more tokens through prefill than decode, but
prefill is ~8x faster per token, so 2P2D is the a-priori guess and the measured
2P2D run was decode-starved (prefill idle ~77%). The ratio is therefore a real
knob, and each stress shape names its own:

| shape | expected best ratio | arm |
|---|---|---|
| `prefill` (8192/1) | prefill-heavy | `pd3p1d` |
| `decode` (128/1024) | decode-heavy | `pd1p3d` |
| `mixed` (8192/1024) | between | `pd2p2d`, `pd2x2` |

## 4. Run order and wall clock

Per arm: teardown + 4 parallel launches + readiness + env asserts ~= 15 min, then
3 concurrency points x (1 warmup + 2 timed) = 9 benches.

| stage | command | est. |
|---|---|---|
| A. headline, 4 machines, mixed | `bash 94_matched.sh` | **~2.5-3 h** |
| B. 2-machine control, mixed | `MACHINE_BUDGET=2 ARMS="agg2:tp agg2:v2 pd1p1d" bash 94_matched.sh` | ~1.5 h |
| C. prefill stress | `WL=prefill ARMS="agg4:tp pd3p1d pd2p2d" bash 94_matched.sh` | ~2 h |
| D. decode stress | `WL=decode ARMS="agg4:tp pd1p3d pd2p2d" bash 94_matched.sh` | ~1.5 h |
| E. table | `bash sync.sh pull && python3 gen_matched_table.py results` | minutes |

**If there is only time for one thing, run stage A.** It is the user's question
verbatim. Stage B is what makes A a *scaling* statement rather than a single point.

Each stage is independently resumable: `ARMS=` takes any subset, and a re-run
overwrites only the tags it produces.

## 5. Prerequisites, in order (5 minutes, and skipping one costs an hour)

```bash
# 1. Four IPs. They survive a stop/start, the ssh aliases do not always.
#    B300-1 172.31.30.164  B300-2 172.31.28.101
#    B300-3 172.31.30.41   B300-4 172.31.21.43     (ap-northeast-2)
for h in B300-1 B300-2 B300-3 B300-4; do ssh $h hostname -I; done

# 2. Scripts up to date on all four (skips unreachable hosts, never --delete)
bash sync.sh push

# 3. The SAME image ID on all four, and it must be the nccl2312 one
for h in B300-1 B300-2 B300-3 B300-4; do
  echo -n "$h "; ssh $h docker images --format '{{.ID}} {{.Repository}}:{{.Tag}}' \
    kimi-k3-efa-v2:nccl2312; done

# 4. A stop/start EMPTIES /opt/dlami/nvme -- weights and the JIT cache both.
#    Cold JIT looks exactly like a bootstrap timeout / security-group problem.
for h in B300-1 B300-2 B300-3 B300-4; do echo -n "$h "; ssh $h du -sh /opt/dlami/nvme/models 2>/dev/null; done

# 5. Nobody else's job on the GPUs (the driver refuses to start otherwise, but
#    check before booking 3 hours)
for h in B300-1 B300-2 B300-3 B300-4; do echo "== $h"; ssh $h nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader; done

# 6. Rehearse the orchestration without touching a host
DRY_RUN=1 bash 94_matched.sh
```

`P6-B300*` are a **colleague's** machines in us-west-2. The driver refuses them
outright; do not "fix" that.

## 6. Three latent bugs this work turned up (all fixed, none measured yet)

1. **`DISABLE_RADIX` was forwarded by no launcher.** All three `start_*.sh`
   honored it; `10_/20_/21_launch_*.sh` never passed it into the container. So
   `91_bench.sh`'s own advice ("also start the server with `DISABLE_RADIX=1`")
   was a dead letter and `--flush-cache` was the only cache control we had.
2. **`IMAGE` defaulted to a build that cannot run the campaign.** `:latest` is
   NCCL 2.30.7; GIN type 5 reports "Cannot get backend version for invalid GIN
   type 5" and it cannot be repaired at runtime (DeepEP asserts an exact
   compile/runtime NCCL match below 2.31). Default is now `:nccl2312`.
3. **Single-node v2 arms did not boot at all on the current image.** `detect_gin()`
   now returns 5 on b300, but type 5 needs more than one node, so a stock
   `ep=8 direct` server died in `_C.ElasticBuffer(...)` with
   `GIN: DevComm setup failed on all available backends`, surfaced as
   `Capture cuda graph failed:`. `env_common.sh` now forces `GIN_TYPE=3` when
   `NNODES == 1`. **`agg4:v2` and `agg2:v2` have therefore never run** — expect
   the first attempt at them to be the real test of this fix.

Also automated, having been manual and campaign-breaking before: the multi-node
`mem_fraction` clamp to **0.78** (TP=16 halves per-GPU weights, the KV pool grows,
and 0.85 OOMs decode capture).

## 7. Reading rules — decided now, before any data exists

- **Per-machine first.** Total throughput at 4 machines beating total at 2 says
  nothing. `out/mach` and `in/mach` are the claim's axes.
- **Both throughput axes, always.** They can move in opposite directions here: a
  PD split that fixes prefill raises input tok/s while decode admission caps
  output tok/s. A table with one of them supports either conclusion.
- **Quote time next to rate.** A GB/s- or tok/s-only table has inverted a
  conclusion in this project before; TTFT p50/p99 and TPOT are in every row.
- **n=2 is not statistics.** The spread block prints min..max and flags any row
  whose replicates differ by >5%. Do not read a <5% difference off a flagged row;
  add replicates instead.
- **If PD loses, say by how much and to what.** "PD is 0.93x of `agg4:tp` on
  output tok/s at c=128, and 1.4x on input tok/s" is a publishable result. So is
  a loss traced to an admission ceiling. What is not publishable is a loss traced
  to a client-limited operating point — that is why §1 exists.
- **State the constraint the arm actually hit**, from the server logs snapshotted
  per run: `#queue-req`, `#inflight-req`, observed `max_running_requests`, prefill
  batch tok/s. The 2P2D run's real answer was "32 decode slots", and no client
  percentile could have said that.

## 8. Known open items that will shape the next round

- `DECODE_CAP=512` gives 48 slots/node by DSPARK's own arithmetic
  (`MAXRUN * (SPEC_BLOCK_SIZE+1) <= CAP`) but has only ever been booted with
  `MAXRUN` pinned to 16. If the decode side plateaus at 16 slots/node, that is
  the first thing to check, not the wire.
- **PD + DSPARK degrades within a run** at `dcp=8` (accept length collapses,
  1660 -> 878 tok/s over 4 runs). The campaign runs `PROFILE=low-latency`
  (`dcp=1/1`), which does not show it — but it is the reason each replicate is a
  separate tagged file rather than an average.
- `pd2x2`'s decode side is one TP=16 instance, so its slot ceiling is a single
  `max_running_requests`, not two. That is a real structural difference from
  `pd2p2d` and may well be why `pd2p2d` wins at high concurrency.
