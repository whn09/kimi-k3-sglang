#!/usr/bin/env python3
"""Find cells where an instance sat at its admission cap, from the SERVER logs.

Every K3 instance boots with max_running_requests=48 (DSPARK picks it). An arm
with fewer instances has fewer slots; past that, requests queue before their
first token, which lands inside TTFT and depresses throughput. Comparing a
capped arm against an uncapped one measures the cap, not the backend.

Read the server's own counters, not the client's. `concurrency` in the bench
JSON is sum(e2e_latency)/duration -- a time average that ramp-up and drain
depress on every short bench, so it flags healthy cells as shortfalls. The
scheduler prints what actually happened:

    #running-req: N   requests resident in this instance
    #queue-req:   N   requests admitted-but-waiting

An instance is cap-bound when it SITS at #running-req == MAXRUN, measured as
the share of scheduler samples at the ceiling. A single transient touch of 48
is not binding; sitting there is. Do not require a local #queue-req -- with
prefill/decode split, a saturated decode instance's backlog forms at the
prefill instance instead, so the local queue reads 0 while the cap is fully
binding (pd2x2 c=128 does exactly this).

    python3 audit_admission.py results
"""
import re
import sys
from collections import defaultdict
from pathlib import Path

MAXRUN = 48          # verified in every server log: max_running_requests=48
AT_CAP_FRAC = 0.05   # share of samples sitting at the ceiling

# results/<tag>.<host>.<container>.log
LOGNAME = re.compile(
    r"^(?P<arm>[a-z0-9]+)-m(?P<mach>\d+)-.*?isl(?P<isl>\d+)-osl(?P<osl>\d+)"
    r"-c(?P<c>\d+)-n\d+-r(?P<rep>[1-9])\.(?P<host>[\w-]+)\.(?P<cont>[\w-]+)\.log$"
)
RUN = re.compile(rb"#running-req: (\d+)")
QUE = re.compile(rb"#queue-req: (\d+)")


def scan(path):
    blob = path.read_bytes()
    runs = [int(m[1]) for m in RUN.finditer(blob)]
    ques = [int(m[1]) for m in QUE.finditer(blob)]
    if not runs:
        return None
    return (max(runs), max(ques, default=0),
            sum(1 for r in runs if r >= MAXRUN), len(runs))


def main(root):
    cells = defaultdict(list)
    for p in sorted(Path(root).glob("*.log")):
        m = LOGNAME.match(p.name)
        if not m:
            continue
        s = scan(p)
        if s:
            key = (f"isl{m['isl']}/osl{m['osl']}", m["arm"], int(m["c"]))
            cells[key].append((m["host"], m["cont"].replace("kimi-k3", "").strip("-")
                               or "unified", *s))

    print(f"{'workload':<16}{'arm':<10}{'c':>5}  {'instance':<22}"
          f"{'maxrun':>7}{'maxq':>6}{'at cap':>10}{'verdict':>10}")
    flagged = []
    for key in sorted(cells):
        wl, arm, c = key
        for host, cont, mr, mq, nz, tot in sorted(cells[key]):
            frac = nz / tot if tot else 0
            capped = frac > AT_CAP_FRAC
            if capped:
                flagged.append((wl, arm, c))
            print(f"{wl:<16}{arm:<10}{c:>5}  {host+'/'+cont:<22}"
                  f"{mr:>7}{mq:>6}{f'{nz}/{tot}':>10}"
                  f"{'CAPPED' if capped else 'ok':>10}")

    print(f"\nCAPPED = the instance sat at #running-req == {MAXRUN} for more than")
    print(f"         {AT_CAP_FRAC:.0%} of scheduler samples. 'at cap' is that count.")
    print("maxrun=0 on a prefill-role instance is a log-format artifact, not a")
    print("reading: prefill batches do not print #running-req.")
    if flagged:
        print("\nDo not compare these cells against an uncapped arm:")
        for k in sorted(set(flagged)):
            print(f"  {k[0]}  {k[1]}  c={k[2]}")
    else:
        print("\nNo cell sat at its admission cap.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results")
