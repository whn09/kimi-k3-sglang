#!/usr/bin/env python3
"""Four-stage synthesis table for the matched-capacity campaign.

Reads the bench JSONs written by 94_matched.sh and prints, per workload, the
arms side by side at iso-load (c = machines x cpm).  Never hand-transcribe a
number out of a log: run this.

    python3 gen_synthesis.py results
"""
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

STEM = re.compile(
    r"^(?P<arm>[a-z0-9]+)-m(?P<mach>\d+)-.*?"
    r"isl(?P<isl>\d+)-osl(?P<osl>\d+)-c(?P<c>\d+)-n\d+-r(?P<rep>\d+)\.json$"
)

WORKLOAD = {
    (8192, 1024): "mixed    isl=8192 osl=1024",
    (8192, 1): "prefill  isl=8192 osl=1",
    (128, 1024): "decode   isl=128  osl=1024",
}

# metric label -> (json key, format, lower_is_better)
METRICS = [
    ("out tok/s", "output_throughput", "%9.1f", False),
    ("in tok/s", "input_throughput", "%9.1f", False),
    ("TPOT ms", "median_tpot_ms", "%9.1f", True),
    ("TTFT med ms", "median_ttft_ms", "%9.1f", True),
    ("TTFT p99 ms", "p99_ttft_ms", "%9.1f", True),
]


# Cells whose numbers came from a driver log rather than a pulled JSON, via
# salvage_log_json.py. They are real measurements, but the hosts they ran on no
# longer exist, so they cannot be re-pulled or re-checked -- the reader has to
# be told which rows those are.
SALVAGED = set()


def load(root):
    """{(workload, machines, cpm, arm): {metric: [values over replicates]}}"""
    cells = defaultdict(lambda: defaultdict(list))
    for path in sorted(Path(root).glob("*.json")):
        m = STEM.match(path.name)
        if not m or int(m["rep"]) == 0:  # r0 is the discarded warmup
            continue
        isl, osl, mach, c = (int(m[k]) for k in ("isl", "osl", "mach", "c"))
        wl = WORKLOAD.get((isl, osl))
        if wl is None:
            continue
        a2a = "tp" if "tponly" in path.name else "v2"
        arm = f"{m['arm']}:{a2a}"
        cpm = c // mach
        d = json.loads(path.read_text())
        if d.get("_salvaged_from"):
            SALVAGED.add((wl, mach, cpm, arm))
        for _, key, _, _ in METRICS:
            if d.get(key) is not None:
                cells[(wl, mach, cpm, arm)][key].append(d[key])
    return cells


# At osl=1 there is no inter-token interval, so TPOT is identically 0 and
# out tok/s is just request rate -- neither cell is readable.
DEGENERATE = {"prefill  isl=8192 osl=1": {"median_tpot_ms", "output_throughput"}}


def spread(vals):
    m = mean(vals)
    return 0.0 if len(vals) < 2 or m == 0 else (max(vals) - min(vals)) / m * 100


# How many machines of each arm actually carry the workload's bottleneck role.
# A deployment-level ratio charges PD for the box it hands to prefill; that is
# the right number to deploy on, but it does NOT measure the backend. To ask
# "is v2 worse than plain TP", divide by the boxes doing the work AND compare
# at matched concurrency-per-box -- PD's fewer-but-fuller boxes flatter it
# otherwise. At isl=128 prefill is trivial, so an aggregated box counts as a
# decoder; at osl=1 there is no decode at all.
WORKING_BOXES = {
    "decode   isl=128  osl=1024": {"agg4:tp": 4, "pd1p3d:v2": 3, "pd2p2d:v2": 2},
    "prefill  isl=8192 osl=1": {"agg4:tp": 4, "pd3p1d:v2": 3, "pd2p2d:v2": 2},
}
# An arm cannot admit more than this many concurrent requests per box.
# DSPARK pins max_running_requests=48, which caps pd2p2d at 2x48=96.
ADMIT_CAP = {"pd2p2d:v2": 48, "pd1p3d:v2": 48, "pd1p1d:v2": 48}


def per_box(cells, wl, key):
    """[(arm, req/box, metric/box)] for the arms whose box count we know."""
    boxes = WORKING_BOXES.get(wl)
    if not boxes:
        return None
    out = []
    for (w, mach, cpm, arm), d in cells.items():
        if w != wl or arm not in boxes or key not in d:
            continue
        n = boxes[arm]
        admitted = min(mach * cpm, n * ADMIT_CAP.get(arm, 10**9))
        out.append((arm, admitted / n, mean(d[key]) / n))
    return sorted(out)


def interp(curve, r):
    """Linear interpolation on the baseline curve. Never extrapolates."""
    ks = sorted(curve)
    if not ks or r < ks[0] or r > ks[-1]:
        return None
    for a, b in zip(ks, ks[1:]):
        if a <= r <= b:
            return curve[a] if a == b else curve[a] + (r-a)/(b-a)*(curve[b]-curve[a])


def print_per_box(cells):
    for wl, boxes in WORKING_BOXES.items():
        key = "input_throughput" if wl.endswith("osl=1") else "output_throughput"
        rows = per_box(cells, wl, key)
        if not rows:
            continue
        base = next(a for a in boxes if a.endswith(":tp"))
        curve = {r: v for a, r, v in rows if a == base}
        print(f"\n=== per working box, matched load -- {wl}")
        print(f"    metric={key}, baseline={base} "
              f"({', '.join(f'{r:.0f} req/box -> {v:.1f}' for r, v in sorted(curve.items()))})")
        print(f"{'arm':<12}{'req/box':>9}{'per box':>10}{'base@same':>11}{'ratio':>8}")
        for arm, r, v in rows:
            if arm == base:
                continue
            b = interp(curve, r)
            tail = (f"{b:>11.1f}{v/b:>8.3f}" if b
                    else f"{'-':>11}{'  n/a: outside baseline range':>8}")
            print(f"{arm:<12}{r:>9.1f}{v:>10.1f}" + tail)


def print_inventory(cells):
    """What was actually run -- so the reader never has to trust a claim about
    which arms exist. An arm absent here was NOT measured."""
    print("=== inventory: arms actually measured (timed replicates only)")
    print(f"{'workload':<26}{'machines':>9}{'arm':>13}{'c':>18}{'reps':>6}{'source':>10}")
    seen = defaultdict(set)
    reps = defaultdict(int)
    for (wl, mach, cpm, arm), d in cells.items():
        seen[(wl, mach, arm)].add(mach * cpm)
        reps[(wl, mach, arm)] = max(reps[(wl, mach, arm)],
                                    max(len(v) for v in d.values()))
    for k in sorted(seen):
        wl, mach, arm = k
        cs = ",".join(str(c) for c in sorted(seen[k]))
        src = "SALVAGED" if any((wl, mach, c // mach, arm) in SALVAGED
                                for c in seen[k]) else "pulled"
        print(f"{wl:<26}{mach:>9}{arm:>13}{cs:>18}{reps[k]:>6}{src:>10}")


def main(root):
    cells = load(root)
    print_inventory(cells)
    for wl in WORKLOAD.values():
        keys = [k for k in cells if k[0] == wl]
        if not keys:
            continue
        for mach in sorted({k[1] for k in keys}):
            arms = sorted({k[3] for k in keys if k[1] == mach})
            cpms = sorted({k[2] for k in keys if k[1] == mach})
            base = next((a for a in arms if a.endswith(":tp")), arms[0])
            print(f"\n=== {wl}   {mach} machines   baseline={base}")
            head = f"{'metric':<13}{'c':>5}" + "".join(f"{a:>17}" for a in arms)
            # Full arm name, a2a kind included: stripping it printed two
            # identical 'xpd1p1d' columns once pd1p1d:tp joined pd1p1d:v2.
            print(head + "".join(f"{'  x'+a:>13}"
                                 for a in arms if a != base))
            for label, key, fmt, lower_better in METRICS:
                if key in DEGENERATE.get(wl, ()):
                    print(f"{label:<13}{'':>5}  -- degenerate at osl=1, omitted")
                    print()
                    continue
                for cpm in cpms:
                    row = f"{label:<13}{mach*cpm:>5}"
                    vals = {}
                    for a in arms:
                        v = cells[(wl, mach, cpm, a)].get(key)
                        if not v:
                            row += f"{'-':>17}"
                            continue
                        vals[a] = mean(v)
                        sp = spread(v)
                        flag = "!" if sp > 5 else " "
                        tail = " n=1" if len(v) < 2 else f"{sp:4.1f}%"
                        row += (fmt % vals[a]) + flag + tail
                    for a in arms:
                        if a == base:
                            continue
                        if a in vals and base in vals and vals[base]:
                            row += f"{vals[a]/vals[base]:13.3f}"
                        else:
                            row += f"{'-':>13}"
                    print(row)
                print()
    print_per_box(cells)
    print()
    print("'!' marks replicate spread > 5% -- that cell is not readable.")
    if SALVAGED:
        print("SALVAGED rows were rebuilt from the driver log by")
        print("salvage_log_json.py because the hosts were terminated before")
        print("sync.sh pull ran. Real measurements, but not re-checkable.")
    print("Ratio columns are arm/baseline: >1 is better for throughput,")
    print("worse for TPOT/TTFT.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results")
