#!/usr/bin/env python3
"""Matched-capacity comparison table: PD vs N independent single-node servers.

    python3 gen_matched_table.py                 # ./results
    python3 gen_matched_table.py path/to/results
    python3 gen_matched_table.py results --baseline agg4:tponly

Reads the JSON that 91_bench.sh writes for runs tagged by 94_matched.sh:

    <arm>-m<MACHINES>-<PROFILE>-<a2a>-isl<ISL>-osl<OSL>-c<CONC>-n<N>-r<REP>

THE POINT OF THIS FILE is that the claim -- "PD beats the same machines run as N
standalone servers" -- is a RATIO between two rows, and a ratio computed by hand
from two tables is how a conclusion gets inverted. So the ratio is computed here,
from the JSON, against a baseline row selected by the same (workload, machines,
concurrency) key. Nothing is transcribed.

Two rules it enforces, both learned the expensive way:

  - Replicates are AGGREGATED, not cherry-picked: the median of the timed runs is
    the row, and the spread is printed. r0 is the discarded JIT warmup and is
    dropped (deep_gemm compiles inside the timed region on the first bench at a
    given shape).
  - Per-machine columns come first. Total throughput at 4 machines beating total
    throughput at 2 machines says nothing; out tok/s PER MACHINE is the axis the
    claim lives on, and it is the one that can move the other way.

Both throughput axes are always shown. Input and output throughput can and do
scale with opposite signs here (a PD split that fixes prefill can raise input
tok/s while decode admission caps output tok/s), so a table with one of them is a
table that can support either conclusion.
"""
import json
import pathlib
import re
import statistics
import sys

# <arm>-m<N>-<profile>-<a2a...>-isl<N>-osl<N>-c<N>-n<N>[-r<N>]
NAME = re.compile(
    r"^(?P<arm>.+?)-m(?P<machines>\d+)-(?P<profile>low-latency|balanced|high-throughput)"
    r"-(?P<a2a>.+?)-isl(?P<isl>\d+)-osl(?P<osl>\d+)-c(?P<conc>\d+)-n(?P<n>\d+)"
    r"(?:-r(?P<rep>\d+))?$"
)

# label, json key, aggregation-friendly format, "higher is better"
COLS = [
    ("out tok/s",   "output_throughput", "{:.1f}", True),
    ("in tok/s",    "input_throughput",  "{:.1f}", True),
    ("dur s",       "duration",          "{:.1f}", False),
    ("TTFT p50 ms", "median_ttft_ms",    "{:.0f}", False),
    ("TTFT p99 ms", "p99_ttft_ms",       "{:.0f}", False),
    ("TPOT ms",     "mean_tpot_ms",      "{:.2f}", False),
    ("ITL p50 ms",  "median_itl_ms",     "{:.2f}", False),
]


def med(vals):
    vals = [v for v in vals if isinstance(v, (int, float))]
    return statistics.median(vals) if vals else None


def fmt(v, f):
    return "-" if v is None else f.format(v)


def input_throughput(d):
    """bench_serving does not always emit input_throughput; derive it when absent.

    total_throughput - output_throughput is exact when both are present, because
    they share the one denominator (the benchmark duration). Deriving it from
    total_input_tokens / duration is the fallback, and is the same number.
    """
    if isinstance(d.get("input_throughput"), (int, float)):
        return d["input_throughput"]
    tot, out = d.get("total_throughput"), d.get("output_throughput")
    if isinstance(tot, (int, float)) and isinstance(out, (int, float)):
        return tot - out
    ti, dur = d.get("total_input_tokens"), d.get("duration")
    if isinstance(ti, (int, float)) and isinstance(dur, (int, float)) and dur:
        return ti / dur
    return None


def main():
    args = [a for a in sys.argv[1:]]
    baseline_pref = None
    if "--baseline" in args:
        i = args.index("--baseline")
        baseline_pref = args[i + 1]
        del args[i:i + 2]
    root = pathlib.Path(args[0] if args else "results")

    # key -> list of per-replicate dicts
    runs, unparsed, warmups = {}, [], 0
    for p in sorted(root.glob("*.json")):
        m = NAME.match(p.stem)
        if not m:
            unparsed.append(p.name)
            continue
        g = m.groupdict()
        if g["rep"] == "0":            # the discarded JIT warmup
            warmups += 1
            continue
        try:
            d = json.loads(p.read_text())
        except Exception as e:         # noqa: BLE001
            unparsed.append(f"{p.name} ({e})")
            continue
        d = dict(d)
        d["input_throughput"] = input_throughput(d)
        key = (int(g["isl"]), int(g["osl"]), int(g["conc"]), int(g["machines"]),
               f'{g["arm"]}:{g["a2a"]}', g["profile"])
        runs.setdefault(key, []).append(d)

    if not runs:
        print(f"no matched-campaign JSON under {root}/")
        if unparsed:
            print("NOT PARSED (shown, not dropped): " + ", ".join(unparsed))
        return

    # Baseline per (isl, osl, conc, machines): the agg* arm, i.e. the same
    # machines run as independent single-node servers. When several agg variants
    # exist (tponly and deepep_v2), the STRONGEST one is the baseline -- picking
    # the weaker would be arguing against a straw man. --baseline pins one.
    base = {}
    for key, ds in runs.items():
        isl, osl, conc, mach, arm, prof = key
        if not arm.startswith("agg"):
            continue
        if baseline_pref and arm != baseline_pref:
            continue
        cur = med([d.get("output_throughput") for d in ds]) or 0
        slot = (isl, osl, conc, mach)
        if slot not in base or cur > base[slot][1]:
            base[slot] = (key, cur)

    head = ["wl", "mach", "arm", "conc", "reps",
            "out/mach", "in/mach", "vs base out", "vs base in"] + [c[0] for c in COLS]
    print("| " + " | ".join(head) + " |")
    print("|" + "---|" * len(head))

    for key in sorted(runs, key=lambda k: (-k[0], k[1], k[3], k[2], k[4])):
        isl, osl, conc, mach, arm, prof = key
        ds = runs[key]
        vals = {k: med([d.get(k) for d in ds]) for _, k, _, _ in COLS}
        out_pm = vals["output_throughput"] / mach if vals["output_throughput"] else None
        in_pm = vals["input_throughput"] / mach if vals["input_throughput"] else None

        slot = (isl, osl, conc, mach)
        vs_out = vs_in = "-"
        if slot in base and base[slot][0] != key:
            bds = runs[base[slot][0]]
            b_out = med([d.get("output_throughput") for d in bds])
            b_in = med([d.get("input_throughput") for d in bds])
            if b_out and vals["output_throughput"]:
                vs_out = f'{vals["output_throughput"] / b_out:.2f}x'
            if b_in and vals["input_throughput"]:
                vs_in = f'{vals["input_throughput"] / b_in:.2f}x'
        elif slot in base and base[slot][0] == key:
            vs_out = vs_in = "(base)"

        row = [f"{isl}/{osl}", str(mach), arm, str(conc), str(len(ds)),
               fmt(out_pm, "{:.1f}"), fmt(in_pm, "{:.1f}"), vs_out, vs_in]
        row += [fmt(vals[k], f) for _, k, f, _ in COLS]
        print("| " + " | ".join(row) + " |")

    # Spread across replicates, separately, because a 2-replicate "median" is a
    # mean and a range over n=2 is not a confidence interval -- printing it as a
    # column would dress up two numbers as statistics.
    print()
    print("replicate spread (out tok/s), min..max per row:")
    for key in sorted(runs, key=lambda k: (-k[0], k[1], k[3], k[2], k[4])):
        ds = runs[key]
        v = sorted(d.get("output_throughput") or 0 for d in ds)
        flag = ""
        if len(v) >= 2 and v[-1] and (v[-1] - v[0]) / v[-1] > 0.05:
            flag = "   <-- >5% spread, do not read a <5% difference off this row"
        print(f'  {key[4]:<28} m{key[3]} c{key[2]} isl{key[0]}: '
              f'{v[0]:.1f}..{v[-1]:.1f} (n={len(v)}){flag}')

    if warmups:
        print(f"\ndropped {warmups} r0 warmup run(s) (JIT compiles inside the timed region)")
    if unparsed:
        print("\nNOT PARSED (shown, not dropped -- these are probably pre-campaign "
              "runs, use gen_pd_table.py for them):")
        for u in unparsed:
            print(f"  {u}")


if __name__ == "__main__":
    main()
