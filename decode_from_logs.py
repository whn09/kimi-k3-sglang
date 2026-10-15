#!/usr/bin/env python3
"""Decode-side 'full window' throughput recomputed from sglang decode logs.

    python3 decode_from_logs.py --target 512 --start 2026-10-15T07:43:00 --dur 363 \
        results/logs/decode_dp_sms12_r1_B300-3.log results/logs/decode_dp_sms12_r1_B300-4.log

Fallback for runs whose live sampler aggregated wrongly (DP attention: each DP
rank reports its own gauges and they live on two nodes' /metrics). Every
"Decode batch" line carries a timestamp, the DP rank, #running-req and the
rank's gen throughput, so: bucket into 5 s windows, take each rank's mean in
the window, SUM over ranks (MAX for a TP instance, whose ranks all report the
same batch), and average the windows whose total running >= 90% of target.
"""
import argparse
import re
from collections import defaultdict
from datetime import datetime, timedelta

LINE = re.compile(
    r"\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d)(?: DP(\d+))? TP(\d+)[^\]]*\] Decode batch,"
    r".*?#running-req: (\d+).*?gen throughput \(token/s\): ([\d.]+)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, required=True)
    ap.add_argument("--start", required=True, help="UTC, ISO")
    ap.add_argument("--dur", type=float, required=True)
    ap.add_argument("--bucket", type=float, default=5.0)
    ap.add_argument("--sat-frac", type=float, default=0.9)
    ap.add_argument("logs", nargs="+")
    a = ap.parse_args()
    t0 = datetime.fromisoformat(a.start)
    t1 = t0 + timedelta(seconds=a.dur)
    # bucket -> rank -> [(running, tput)]
    b = defaultdict(lambda: defaultdict(list))
    dp_mode = False
    for path in a.logs:
        for m in LINE.finditer(open(path, errors="replace").read()):
            ts = datetime.fromisoformat(m.group(1).replace(" ", "T"))
            if not (t0 <= ts <= t1):
                continue
            dp_mode |= m.group(2) is not None
            rank = m.group(2) if m.group(2) is not None else m.group(3)
            k = int((ts - t0).total_seconds() // a.bucket)
            b[k][rank].append((int(m.group(4)), float(m.group(5))))
    full, tot_run = [], []
    for k in sorted(b):
        per = {r: (sum(x for x, _ in v) / len(v), sum(y for _, y in v) / len(v)) for r, v in b[k].items()}
        if dp_mode:
            run = sum(x for x, _ in per.values()); tput = sum(y for _, y in per.values())
        else:
            run = max(x for x, _ in per.values()); tput = max(y for _, y in per.values())
        tot_run.append(run)
        if run >= a.sat_frac * a.target:
            full.append(tput)
    mode = "dp(sum over ranks)" if dp_mode else "tp(max over ranks)"
    if full:
        print(f"{mode}: full windows {len(full)}x{a.bucket:.0f}s, max running {max(tot_run):.0f}/{a.target}, "
              f"decode-full tok/s {sum(full)/len(full):.0f}")
    else:
        print(f"{mode}: NOT FULL (max running {max(tot_run, default=0):.0f}/{a.target})")


if __name__ == "__main__":
    main()
