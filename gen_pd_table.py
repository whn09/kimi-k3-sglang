#!/usr/bin/env python3
"""Generate the PD (1P1D) benchmark table from results/*.json.

Companion to gen_bench_table.py, which parses the older standalone
k3_bench_<tag>.txt logs. This one reads the JSON that 91_bench.sh writes:

    python3 gen_pd_table.py                # ./results
    python3 gen_pd_table.py path/to/results

Every axis lives in the FILENAME, and every axis in the filename becomes a
column -- that is the whole point. Two rows that differ in an axis the table
does not show are not comparable, and a request count that is not in the name
silently overwrote a row once already (see the -n suffix). So: parse the name,
never hand-copy the numbers, and print anything that does not parse instead of
dropping it.

Filename grammar written by 91_bench.sh:

    <MODE>-<PROFILE><EPTAG>-isl<ISL>-osl<OSL>-c<CONCURRENCY>-n<NUM_PROMPTS>[-<rep>]

EPTAG is free-form and carries the arm (e.g. `-deepep_v2-p2048d512`,
`-tponly`); it is printed verbatim rather than decoded, because inventing a
schema for it would just be another place to get an axis wrong. Older runs
predate the `-n` suffix or append a hand-written replicate tag (`-r1`, `-ab2`);
both are handled, and a guessed request count is marked with `?`.

Parsing anchors on the `-isl<N>-osl<N>-c<N>` middle rather than matching the
whole name left-to-right: `low-latency` contains the same separator as the
fields around it, so a left-anchored regex splits it as mode=`standalone`,
profile=`b`, arm=`alanced`. Anchoring on the part with fixed syntax avoids that.
"""
import json
import pathlib
import re
import sys

ANCHOR = re.compile(r"-isl(?P<isl>\d+)-osl(?P<osl>\d+)-c(?P<conc>\d+)")
PROFILES = ("low-latency", "high-throughput", "balanced")

# label, json key, format
COLS = [
    ("reqs",        "completed",         "{:.0f}"),
    ("dur s",       "duration",          "{:.2f}"),
    ("out tok/s",   "output_throughput", "{:.2f}"),
    ("tot tok/s",   "total_throughput",  "{:.2f}"),
    ("TTFT mean",   "mean_ttft_ms",      "{:.0f}"),
    ("TTFT p50",    "median_ttft_ms",    "{:.0f}"),
    ("TTFT p99",    "p99_ttft_ms",       "{:.0f}"),
    ("TPOT mean",   "mean_tpot_ms",      "{:.2f}"),
    ("ITL p50",     "median_itl_ms",     "{:.2f}"),
    ("acc len",     "accept_length",     "{:.2f}"),
]


def cell(v, fmt):
    if v is None:
        return "-"
    try:
        return fmt.format(v)
    except (TypeError, ValueError):
        return str(v)


def split_name(stem):
    """stem -> dict(mode, profile, arm, isl, osl, conc, n, rep) or None."""
    m = ANCHOR.search(stem)
    if not m:
        return None
    g = dict(isl=m["isl"], osl=m["osl"], conc=m["conc"])

    head = stem[:m.start()]
    for prof in PROFILES:
        mode, sep, rest = head.partition(f"-{prof}")
        if sep:
            g.update(mode=mode, profile=prof, arm=rest.lstrip("-"))
            break
    else:
        mode, _, rest = head.partition("-")
        g.update(mode=mode, profile=rest or "-", arm="")

    tail = stem[m.end():].lstrip("-")
    n, rep = None, ""
    for tok in filter(None, tail.split("-")):
        if re.fullmatch(r"n\d+", tok) and n is None:
            n = tok[1:]
        else:
            rep = tok if not rep else f"{rep}-{tok}"
    g.update(n=n, rep=rep)
    return g


def main():
    root = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "results")
    rows, unparsed = [], []
    for p in sorted(root.glob("*.json")):
        g = split_name(p.stem)
        if g is None:
            unparsed.append(p.name)
            continue
        try:
            d = json.loads(p.read_text())
        except Exception as e:                        # noqa: BLE001
            unparsed.append(f"{p.name} ({e})")
            continue
        # An older run may predate the -n suffix. Fall back to the measured
        # request count, and MARK IT, so a row on a guessed denominator is
        # never mistaken for one on a stamped denominator.
        g["n"] = g["n"] or f'{d.get("completed", "?")}?'
        rows.append((g, d))

    head = ["mode", "profile", "arm", "isl", "osl", "conc", "n", "rep"] + [c[0] for c in COLS]
    print("| " + " | ".join(head) + " |")
    print("|" + "---|" * len(head))
    # Sort by shape then arm then concurrency so a CAP sweep reads down the page.
    for g, d in sorted(rows, key=lambda r: (int(r[0]["isl"]), int(r[0]["osl"]),
                                            r[0]["arm"], int(r[0]["conc"]), r[0]["rep"])):
        cells = [g["mode"], g["profile"], g["arm"] or "-",
                 g["isl"], g["osl"], g["conc"], g["n"], g["rep"] or "-"]
        cells += [cell(d.get(k), f) for _, k, f in COLS]
        print("| " + " | ".join(cells) + " |")

    if unparsed:
        print("\nNOT PARSED (shown, not dropped): " + ", ".join(unparsed))


if __name__ == "__main__":
    main()
