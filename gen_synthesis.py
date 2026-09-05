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


def main(root):
    cells = load(root)
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
            print(head + "".join(f"{'  x'+a.split(':')[0]:>12}"
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
                            row += f"{vals[a]/vals[base]:12.3f}"
                        else:
                            row += f"{'-':>12}"
                    print(row)
                print()
    print("'!' marks replicate spread > 5% -- that cell is not readable.")
    print("Ratio columns are arm/baseline: >1 is better for throughput,")
    print("worse for TPOT/TTFT.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results")
