#!/usr/bin/env python3
"""Generate the K3 deepep_v2-vs-v1 benchmark table from raw bench_serving logs.

Never hand-copy these numbers: parse the logs. Run on the host (or after scp'ing
/opt/dlami/nvme/k3_bench_*.txt somewhere local):

    python3 gen_bench_table.py /opt/dlami/nvme

Each log is one `python3 -m sglang.bench_serving` run written by bench_k3.sh as
k3_bench_<TAG>.txt. TAG encodes the arm; ARMS below maps TAG -> (row label, the
axes that were actually varied). Every axis that differs between two rows must
appear in the label, or the table silently compares different configs.
"""
import re
import sys
import pathlib

# tag -> (backend, workload, chunk, cudagraph, note)
ARMS = [
    ("v2_p8k",       "deepep_v2", "prefill ISL=8K OSL=1 conc4",  512,  "on",  "CAP=512"),
    ("v2_p8k_c2048", "deepep_v2", "prefill ISL=8K OSL=1 conc4",  2048, "off", "CAP=2048"),
    ("v1_p8k_c2048", "deepep",    "prefill ISL=8K OSL=1 conc4",  2048, "on",  "no ElasticBuffer"),
    ("v2_d256",      "deepep_v2", "decode ISL=256 OSL=1K conc32", 512, "on",  "CAP=512"),
    ("v1_d256_c512", "deepep",    "decode ISL=256 OSL=1K conc32", 512, "on",  "matched to v2_d256"),
    ("v1_d256",      "deepep",    "decode ISL=256 OSL=1K conc32", 2048, "on", "UNMATCHED chunk"),
]

FIELDS = [
    ("dur",     r"Benchmark duration \(s\):\s+([\d.]+)"),
    ("in_tps",  r"Input token throughput \(tok/s\):\s+([\d.]+)"),
    ("out_tps", r"Output token throughput \(tok/s\):\s+([\d.]+)"),
    ("conc",    r"Concurrency:\s+([\d.]+)"),
    ("ttft_p50", r"Median TTFT \(ms\):\s+([\d.]+)"),
    ("ttft_avg", r"Mean TTFT \(ms\):\s+([\d.]+)"),
    ("tpot_p50", r"Median TPOT \(ms\):\s+([\d.]+)"),
    ("ok",      r"Successful requests:\s+(\d+)"),
]


def parse(path):
    txt = path.read_text(errors="replace")
    out = {}
    for name, pat in FIELDS:
        m = re.search(pat, txt)
        out[name] = float(m.group(1)) if m else None
    return out


def cell(v, fmt="{:.2f}"):
    return "-" if v is None else fmt.format(v)


def main():
    root = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "/opt/dlami/nvme")
    rows, missing = [], []
    for tag, backend, workload, chunk, cg, note in ARMS:
        p = root / f"k3_bench_{tag}.txt"
        if not p.exists():
            missing.append(tag)
            continue
        d = parse(p)
        rows.append((tag, backend, workload, chunk, cg, note, d))

    print("| tag | backend | workload | chunk | cudagraph | reqs | dur (s) | in tok/s | out tok/s | conc | TTFT p50 (ms) | TTFT mean (ms) | TPOT p50 (ms) | note |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for tag, backend, workload, chunk, cg, note, d in rows:
        print("| `{}` | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |".format(
            tag, backend, workload, chunk, cg,
            cell(d["ok"], "{:.0f}"), cell(d["dur"]),
            cell(d["in_tps"]), cell(d["out_tps"]), cell(d["conc"]),
            cell(d["ttft_p50"]), cell(d["ttft_avg"]), cell(d["tpot_p50"]), note))
    if missing:
        print("\nMISSING logs (row omitted, not zero): " + ", ".join(missing))


if __name__ == "__main__":
    main()
