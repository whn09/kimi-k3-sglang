#!/usr/bin/env python3
"""Sample the decode server's Prometheus gauges while a benchmark runs.

    python3 decode_sampler.py --out X.decode.json --target 512 http://172.31.0.212:30000 &
    ... benchmark ...
    kill -INT $!            # writes X.decode.json on exit

Same method as kimi-k3-vllm/decode_sampler.py, adapted to sglang. The client's
output tok/s averages over the ramp-up (every request's prefill lands at once)
and the drain, so it under-reports what decode sustains once it is FULL. The
headline number here is decode throughput over the window where
`sglang:num_running_reqs` >= SAT_FRAC x target, i.e. decode actually saturated;
if that window is empty the run reports None instead of a number.

Differences from the vLLM sampler, both forced by sglang:
- Throughput is the scheduler's own `sglang:gen_throughput` gauge (decode tok/s
  over its last log interval; the launcher sets --decode-log-interval 1), not a
  counter delta: sglang's `generation_tokens_total` is incremented when a
  request FINISHES, so over a 5 s window it is a step function.
- Series are combined with MAX, not SUM: a TP instance's ranks all report the
  same batch, so summing would multiply by the TP degree. (A DP-attention decode
  would need SUM; this kit's decode is TP.)
"""
import argparse
import json
import re
import signal
import sys
import time
import urllib.request

def gauge(name):
    return re.compile(r'^' + re.escape(name) + r'(?:\{[^}]*\})?\s+([0-9.e+-]+)$', re.M)

RUN = gauge("sglang:num_running_reqs")
QUEUE = gauge("sglang:num_queue_reqs")
TPUT = gauge("sglang:gen_throughput")
USAGE = gauge("sglang:token_usage")


def scrape(urls):
    run, queue, tput, usage = [], [], [], []
    for u in urls:
        txt = urllib.request.urlopen(u + "/metrics", timeout=5).read().decode()
        run += [float(x) for x in RUN.findall(txt)]
        queue += [float(x) for x in QUEUE.findall(txt)]
        tput += [float(x) for x in TPUT.findall(txt)]
        usage += [float(x) for x in USAGE.findall(txt)]
    if not run:
        raise RuntimeError("no sglang:num_running_reqs in /metrics -- is --enable-metrics on?")
    return max(run), max(queue, default=0.0), max(tput, default=None), max(usage, default=None)


def summarize(samples, target, sat_frac):
    sat = [s for s in samples if s["running"] >= sat_frac * target and s["tput"] is not None]
    out = {
        "target_running": target,
        "sat_frac": sat_frac,
        "n_samples": len(samples),
        "max_running": max((s["running"] for s in samples), default=0),
        "max_queue": max((s["queue"] for s in samples), default=0),
        "max_token_usage": max((s["usage"] or 0 for s in samples), default=0),
        "sat_samples": len(sat),
    }
    if len(sat) >= 2:
        out["sat_window_s"] = sat[-1]["t"] - sat[0]["t"]
        out["sat_gen_tok_s"] = sum(s["tput"] for s in sat) / len(sat)
        out["sat_mean_running"] = sum(s["running"] for s in sat) / len(sat)
    else:
        out["sat_window_s"] = 0
        out["sat_gen_tok_s"] = None   # decode never full: do not quote a number
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--target", type=int, required=True, help="expected running reqs when full")
    ap.add_argument("--interval", type=float, default=5.0)
    ap.add_argument("--sat-frac", type=float, default=0.9)
    ap.add_argument("urls", nargs="+")
    a = ap.parse_args()

    samples, t0, stop = [], time.time(), False

    def _stop(*_):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)
    while not stop:
        try:
            r, q, tp, us = scrape(a.urls)
            samples.append({"t": time.time() - t0, "running": r, "queue": q, "tput": tp, "usage": us})
        except Exception as e:  # a missed scrape is a gap, not a failure
            print(f"sampler: scrape failed: {e}", file=sys.stderr)
        time.sleep(a.interval)

    res = summarize(samples, a.target, a.sat_frac)
    res["samples"] = samples
    with open(a.out, "w") as f:
        json.dump(res, f)
    print(f"sampler: max_running={res['max_running']:.0f}/{a.target} "
          f"sat_window={res['sat_window_s']:.0f}s sat_gen_tok_s={res['sat_gen_tok_s']}")


if __name__ == "__main__":
    main()
