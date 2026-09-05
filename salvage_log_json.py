#!/usr/bin/env python3
"""Rebuild missing bench JSONs from a driver log's printed blocks.

The bench JSONs are written on the HOSTS and only reach the laptop via
`sync.sh pull` at the end of a campaign. If the hosts die first -- on
2026-09-05 all four B300 were terminated "User initiated" mid-stage-E -- the
only surviving record of a completed run is the driver log, which prints each
block under its full tag:

    ----- run 1/2 c=64  tag=pd1p1d-m2-low-latency-tponly-isl8192-osl1024-c64-n128-r1 -----
    Output token throughput (tok/s):         2529.05

That tag IS the JSON stem gen_synthesis.py parses, so reconstructing the file
keeps the generator as the single source of every published number instead of
hand-copying survivors into a table.

Every rebuilt file carries "_salvaged_from" so a salvaged row can never be
mistaken for a pulled one, and existing files are never overwritten.

    python3 salvage_log_json.py results/campaign2.E.log results
"""
import json
import re
import sys
from pathlib import Path

TAG = re.compile(r"^-----.*\btag=(?P<tag>\S+)\s+-----")

# printed label -> json key. Only the keys anything downstream reads, plus
# enough context to sanity-check a block (a run that died mid-way prints a
# partial block, and a partial block must not become a JSON).
FIELDS = {
    "Successful requests": "completed",
    "Benchmark duration (s)": "duration",
    "Request throughput (req/s)": "request_throughput",
    "Input token throughput (tok/s)": "input_throughput",
    "Output token throughput (tok/s)": "output_throughput",
    "Total token throughput (tok/s)": "total_token_throughput",
    "Concurrency": "concurrency",
    "Mean TTFT (ms)": "mean_ttft_ms",
    "Median TTFT (ms)": "median_ttft_ms",
    "P99 TTFT (ms)": "p99_ttft_ms",
    "Mean TPOT (ms)": "mean_tpot_ms",
    "Median TPOT (ms)": "median_tpot_ms",
    "Mean ITL (ms)": "mean_itl_ms",
    "Median ITL (ms)": "median_itl_ms",
}
# A block missing any of these is incomplete -- refuse it rather than emit a
# JSON with holes that reads as a real measurement.
REQUIRED = {"completed", "duration", "output_throughput", "input_throughput",
            "median_ttft_ms", "p99_ttft_ms", "median_tpot_ms"}

NUM = re.compile(r"^(?P<label>[A-Za-z][^:]*?):\s+(?P<val>-?[\d.]+)\s*$")


def blocks(log):
    """[(tag, {key: float})] in file order."""
    out, tag, cur = [], None, {}
    for line in Path(log).read_text(errors="replace").splitlines():
        m = TAG.match(line)
        if m:
            if tag:
                out.append((tag, cur))
            tag, cur = m["tag"], {}
            continue
        if tag is None:
            continue
        n = NUM.match(line.strip())
        if n and n["label"].strip() in FIELDS:
            cur[FIELDS[n["label"].strip()]] = float(n["val"])
    if tag:
        out.append((tag, cur))
    return out


def main(log, root):
    root = Path(root)
    wrote = skipped = partial = 0
    for tag, d in blocks(log):
        missing = REQUIRED - d.keys()
        if missing:
            print(f"PARTIAL  {tag}  missing {sorted(missing)} -- not written")
            partial += 1
            continue
        path = root / f"{tag}.json"
        if path.exists():
            print(f"exists   {path.name}")
            skipped += 1
            continue
        # max_concurrency is not printed; it is the c<N> in the tag, which is
        # what the driver asked for.
        c = re.search(r"-c(\d+)-", tag)
        d["max_concurrency"] = int(c[1]) if c else None
        d["_salvaged_from"] = str(log)
        path.write_text(json.dumps(d, indent=2) + "\n")
        print(f"wrote    {path.name}")
        wrote += 1
    print(f"\n{wrote} written, {skipped} already present, {partial} partial/refused")
    if wrote:
        print("Salvaged rows carry _salvaged_from. Say so next to any table that\n"
              "uses them: they are the driver's printed output, not a pulled JSON.")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2])
