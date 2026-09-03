#!/usr/bin/env python3
"""Generate the decode-CAP comparison table, and verify it is a one-axis comparison.

    python3 gen_decode_cap_table.py                       # ./results, caps 128 512
    python3 gen_decode_cap_table.py --caps 128 512 1024
    python3 gen_decode_cap_table.py --results path/to/results

Two things this does that a hand-written table cannot:

1.  It reads the numbers out of the .json 91_bench.sh wrote, so no figure is ever
    transcribed. 91_bench.sh *appends* to that file, so a tag that gets re-run
    ends up with several records in one file -- this takes the LAST and says so,
    because silently taking the first would report a stale replicate under a
    fresh filename.

2.  It asserts the comparison is one-axis. A small decode CAP is only legal with
    a small --max-running-requests, so "CAP=128 vs CAP=512" moves two knobs
    unless MAXRUN is pinned; and DECODE_CHUNK defaults to CAP in env_common.sh,
    which would move a third. Those live in the decode server's own log, so they
    are read back from the .decode.log snapshot rather than assumed. Any arm
    that disagrees with the others is reported as a BLOCKER and the table is
    still printed, marked untrustworthy -- a comparison that quietly moved an
    axis is worse than no table.

r0 is the discarded warmup (deep_gemm JIT compiles inside the timed region on
the first bench at a given shape) and is excluded from every statistic, but it
is printed so a warmup that is far off the timed runs stays visible.
"""
import argparse
import glob
import json
import os
import re
import sys

# Read back out of the decode server's server_args / pool lines. Key -> regex.
# These are the axes that MUST match across arms for the CAP delta to mean
# anything; CAP itself is deliberately absent, since that is the axis under test.
INVARIANTS = {
    "max_running_requests": r"'max_running_requests': (\d+)",
    "chunked_prefill_size": r"'chunked_prefill_size': (\d+)",
    "dcp_size": r"'dcp_size': (\d+)",
    "cuda_graph_max_bs_decode": r"'cuda_graph_max_bs_decode': (\d+)",
    "spec_num_draft_tokens": r"'speculative_num_draft_tokens': (\d+)",
    "kv_pool_tokens": r"KV Cache is allocated.*?#tokens: (\d+)",
    "mem_fraction_static": r"'mem_fraction_static': ([\d.]+)",
}

METRICS = [
    ("output_throughput", "out tok/s", "%.1f"),
    ("duration", "dur s", "%.2f"),
    ("mean_ttft_ms", "TTFT mean ms", "%.0f"),
    ("median_ttft_ms", "TTFT p50 ms", "%.0f"),
    ("mean_tpot_ms", "TPOT mean ms", "%.2f"),
    ("median_itl_ms", "ITL p50 ms", "%.2f"),
    ("accept_length", "accept len", "%.2f"),
]


def load_last(path):
    """91_bench.sh appends; the last record is the current run."""
    recs = [json.loads(l) for l in open(path) if l.strip()]
    return (recs[-1], len(recs)) if recs else (None, 0)


def scan(results, caps, prefix, suffix):
    """arms[cap] = {rep: record}, plus how many records each file held."""
    arms, stale = {}, []
    for cap in caps:
        pat = os.path.join(results, f"{prefix}d{cap}{suffix}-r*.json")
        reps = {}
        for path in sorted(glob.glob(pat)):
            m = re.search(r"-r(\d+)\.json$", path)
            if not m:
                continue
            rec, n = load_last(path)
            if rec is None:
                continue
            reps[int(m.group(1))] = rec
            if n > 1:
                stale.append((os.path.basename(path), n))
        if reps:
            arms[cap] = reps
    return arms, stale


def invariants(results, cap, prefix, suffix, reps):
    """Read the pinned axes out of each replicate's decode-log snapshot."""
    seen = {}
    for r in sorted(reps):
        path = os.path.join(results, f"{prefix}d{cap}{suffix}-r{r}.decode.log")
        if not os.path.exists(path):
            continue
        txt = open(path, errors="replace").read()
        for key, rx in INVARIANTS.items():
            m = re.search(rx, txt, re.S)
            if m:
                seen.setdefault(key, set()).add(m.group(1))
    return {k: sorted(v) for k, v in seen.items()}


def mean(xs):
    return sum(xs) / len(xs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="./results")
    ap.add_argument("--caps", nargs="+", type=int, default=[128, 512])
    ap.add_argument("--prefix", default="pd-low-latency-mooncake-deepep_v2-p8192")
    ap.add_argument("--suffix", default="-mr16-isl8192-osl1024-c32")
    a = ap.parse_args()

    arms, stale = scan(a.results, a.caps, a.prefix, a.suffix)
    if not arms:
        sys.exit(f"no results matched {a.prefix}d<cap>{a.suffix}-r*.json under {a.results}")

    print(f"decode CAP sweep  {a.prefix}d<CAP>{a.suffix}")
    print("r0 = discarded warmup (deep_gemm JIT in the timed region), excluded from means\n")

    # ---- per-replicate table ----
    hdr = f"{'CAP':>6} {'rep':>4} " + " ".join(f"{lbl:>13}" for _, lbl, _ in METRICS)
    print(hdr)
    print("-" * len(hdr))
    for cap in sorted(arms):
        for r in sorted(arms[cap]):
            rec = arms[cap][r]
            cells = []
            for key, _, fmt in METRICS:
                v = rec.get(key)
                cells.append(f"{(fmt % v) if isinstance(v, (int, float)) else '-':>13}")
            mark = "r%d%s" % (r, "*" if r == 0 else "")
            print(f"{cap:>6} {mark:>4} " + " ".join(cells))
        print()

    # ---- means over timed replicates ----
    timed = {c: [rec for r, rec in arms[c].items() if r != 0] for c in arms}
    print(f"{'CAP':>6} {'n':>4} " + " ".join(f"{lbl:>13}" for _, lbl, _ in METRICS))
    print("-" * len(hdr))
    means = {}
    for cap in sorted(timed):
        recs = timed[cap]
        row, means[cap] = [], {}
        for key, _, fmt in METRICS:
            xs = [r[key] for r in recs if isinstance(r.get(key), (int, float))]
            if xs:
                means[cap][key] = mean(xs)
                row.append(f"{(fmt % mean(xs)):>13}")
            else:
                row.append(f"{'-':>13}")
        print(f"{cap:>6} {len(recs):>4} " + " ".join(row))

    # ---- deltas against the largest CAP as baseline ----
    base = max(means)
    print(f"\ndelta vs CAP={base} (negative = smaller CAP is better for time metrics)")
    for cap in sorted(means):
        if cap == base:
            continue
        parts = []
        for key, lbl, _ in METRICS:
            if key in means[cap] and key in means[base] and means[base][key]:
                d = 100.0 * (means[cap][key] - means[base][key]) / means[base][key]
                parts.append(f"{lbl} {d:+.1f}%")
        print(f"  CAP={cap}: " + ", ".join(parts))

    # ---- spread, so a delta can be read against replicate noise ----
    print("\nreplicate spread (max-min as % of mean, timed runs only)")
    for cap in sorted(timed):
        parts = []
        for key, lbl, _ in METRICS:
            xs = [r[key] for r in timed[cap] if isinstance(r.get(key), (int, float))]
            if len(xs) > 1 and mean(xs):
                parts.append(f"{lbl} {100.0*(max(xs)-min(xs))/mean(xs):.1f}%")
        print(f"  CAP={cap} (n={len(timed[cap])}): " + (", ".join(parts) or "n/a"))

    # ---- the one-axis check ----
    print("\none-axis check (read back from each arm's .decode.log)")
    per_arm = {c: invariants(a.results, c, a.prefix, a.suffix, arms[c]) for c in arms}
    blockers = []
    keys = sorted({k for v in per_arm.values() for k in v})
    for key in keys:
        vals = {c: per_arm[c].get(key, ["<absent>"]) for c in sorted(per_arm)}
        within = [c for c, v in vals.items() if len(v) > 1]
        across = {tuple(v) for v in vals.values()}
        cells = ", ".join(f"CAP={c}:{'/'.join(v)}" for c, v in vals.items())
        if within:
            blockers.append(f"{key} varies BETWEEN REPLICATES of CAP={within} -> {cells}")
            print(f"  BLOCKER {key}: {cells}")
        elif len(across) > 1:
            blockers.append(f"{key} differs across arms -> {cells}")
            print(f"  BLOCKER {key}: {cells}")
        else:
            print(f"  ok      {key}: {vals[sorted(vals)[0]][0]}")

    if stale:
        print("\nnote: these files held more than one record (91_bench.sh appends);")
        print("      the LAST was used. Re-run history, not an error:")
        for name, n in sorted(set(stale)):
            print(f"      {name}: {n} records")

    if blockers:
        print("\n*** NOT A ONE-AXIS COMPARISON -- do not publish the deltas above ***")
        for b in blockers:
            print(f"    {b}")
        return 1
    print("\nall pinned axes agree: the only difference between arms is the decode CAP.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
