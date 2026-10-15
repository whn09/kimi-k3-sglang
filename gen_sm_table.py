#!/usr/bin/env python3
"""DeepEP v2 decode SM sweep table (96_sm_sweep.sh output).

Ported from kimi-k3-vllm/gen_sm_table.py; same columns, same verdict rule.

    python3 gen_sm_table.py results/ > results/sm_sweep.md

Generated, never hand-copied. Per (SM arm, concurrency): every replicate is
listed as mean [min-max], so the spread of an arm against ITSELF is on the page
next to the gap between arms. The verdict column only calls a difference real
when it is larger than both arms' own spread.

Primary metric is decode-side tok/s over the window where decode was full
(decode_sampler.py); a row whose decode never reached SAT_FRAC x target says
"not full" instead of quoting a number. TPOT is the time axis.
"""
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

PAT = re.compile(
    r"k3s_pd2x2_dec-(?P<attn>tp|dp|dcp)16ep16_sms(?P<sms>\d+)_spec(?P<spec>on|off)_cap\d+_run(?P<seqs>\d+)"
    r"_isl(?P<isl>\d+)_osl(?P<osl>\d+)_c(?P<conc>\d+)_n(?P<n>\d+)_r(?P<rep>\d+)(?:_img-(?P<img>[a-z0-9]+))?_(?P<ts>[\d-]+)\.json$"
)


def fmt(xs, nd=1):
    xs = [x for x in xs if x is not None]
    if not xs:
        return "n/a"
    if len(xs) == 1:
        return f"{xs[0]:.{nd}f}"
    return f"{mean(xs):.{nd}f} [{min(xs):.{nd}f}-{max(xs):.{nd}f}]"


def spread(xs):
    xs = [x for x in xs if x is not None]
    return (max(xs) - min(xs)) / mean(xs) if len(xs) > 1 and mean(xs) else 0.0


def main(root):
    # (kv, seqs, isl, osl, conc, sms) -> rep -> latest (ts, client, decode)
    runs = defaultdict(dict)
    bad = []
    for p in sorted(Path(root).glob("k3s_pd2x2_*.json")):
        if p.name.endswith(".decode.json"):
            continue
        m = PAT.search(p.name)
        if not m:
            bad.append(p.name)
            continue
        dpath = p.with_name(p.name[:-5] + ".decode.json")
        dec = json.loads(dpath.read_text()) if dpath.exists() else {}
        # Files from before the image was stamped are the 07c8f729 generation.
        key = (f"{m['img'] or 'sgl07c8f729'}/{m['attn']}/spec-{m['spec']}", int(m["seqs"]), int(m["isl"]), int(m["osl"]), int(m["conc"]), int(m["sms"]))
        rep = int(m["rep"])
        if rep not in runs[key] or m["ts"] > runs[key][rep][0]:
            runs[key][rep] = (m["ts"], json.loads(p.read_text()), dec)

    print("| image/attn/spec | ISL/OSL | conc | SM (req -> used) | reps | decode-full tok/s | "
          "max running / target | TPOT p50 (ms) | ITL p50 (ms) | TPOT p99 (ms) | client out tok/s | dur (s) |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    groups = defaultdict(dict)  # (kv, seqs, isl, osl, conc) -> sms -> metrics
    for key in sorted(runs, key=lambda k: (k[0], k[2], k[3], k[4], k[5] == 0, k[5])):
        kv, seqs, isl, osl, conc, sms = key
        reps = [runs[key][r] for r in sorted(runs[key])]
        cl = [c for _, c, _ in reps]
        dc = [d for _, _, d in reps]
        used = sorted({str(d.get("resolved_sms", "?")) for d in dc})
        sat = [d.get("sat_gen_tok_s") for d in dc]
        full = all(s is not None for s in sat)
        mx = [d.get("max_running", 0) for d in dc]
        tpot = [c["median_tpot_ms"] for c in cl]
        groups[key[:5]][sms] = {"sat": sat if full else None, "tpot": tpot}
        print(f"| {kv} | {isl}/{osl} | {conc} | {'auto' if sms == 0 else sms} -> {'/'.join(used)} | "
              f"{len(reps)} | {fmt(sat, 0) if full else 'NOT FULL'} | "
              f"{max(mx):.0f} / {conc} | {fmt(tpot, 2)} | {fmt([c['median_itl_ms'] for c in cl], 2)} | "
              f"{fmt([c['p99_tpot_ms'] for c in cl], 2)} | {fmt([c['output_throughput'] for c in cl], 0)} | "
              f"{fmt([c['duration'] for c in cl], 0)} |")

    print("\n**SM12 vs auto** (positive tok/s and negative TPOT favour SM12; "
          "noise = larger of the two arms' own max-min spread)\n")
    print("| image/attn/spec | ISL/OSL | conc | Δ decode-full tok/s | Δ TPOT p50 | noise (tok/s / TPOT) | verdict |")
    print("|---|---|---|---|---|---|---|")
    for g, arms in sorted(groups.items(), key=lambda kv_: (kv_[0][0], kv_[0][2], kv_[0][3], kv_[0][4])):
        if 12 not in arms or 0 not in arms:
            continue
        a, b = arms[12], arms[0]
        dt = mean(a["tpot"]) / mean(b["tpot"]) - 1
        nt = max(spread(a["tpot"]), spread(b["tpot"]))
        if a["sat"] and b["sat"]:
            ds = mean(a["sat"]) / mean(b["sat"]) - 1
            ns = max(spread(a["sat"]), spread(b["sat"]))
            ds_s, ns_s = f"{ds:+.1%}", f"{ns:.1%}"
        else:
            ds, ns, ds_s, ns_s = None, None, "n/a (not full)", "n/a"
        real = abs(dt) > nt and (ds is None or abs(ds) > ns)
        reps_ok = min(len(a["tpot"]), len(b["tpot"])) >= 2
        verdict = ("beats noise" if real else "within noise") if reps_ok else "1 rep: no noise floor"
        print(f"| {g[0]} | {g[2]}/{g[3]} | {g[4]} | {ds_s} | {dt:+.1%} | {ns_s} / {nt:.1%} | {verdict} |")

    if bad:
        print(f"\nUNPARSED ({len(bad)}): " + ", ".join(bad), file=sys.stderr)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results")
