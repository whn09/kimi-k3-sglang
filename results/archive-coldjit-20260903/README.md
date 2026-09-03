# Discarded decode-CAP arms, 2026-09-03

Two arms that must not be read as results. Kept because a discarded run is only
credible if it is still inspectable, and because both failure modes are ones we
expect to hit again.

## `...p8192d1024s1024-mr16.txt` — CAP=1024 does not fit

Ends abruptly after `ready: prefill after 210s`. The decode node OOMed during
decode CUDA graph capture:

    torch.OutOfMemoryError: Tried to allocate 10.50 GiB ... 8.02 GiB is free ...
    22.90 GiB allocated in private pools (e.g. CUDA Graphs)
    decode_cuda_graph_runner.py:491, via capture_decode_graph

Not fixed by `--cuda-graph-max-bs-decode`: the capture pool is sized by the CAP,
not by how many shapes are captured. Lowering `mem_fraction_static` would have
moved a second axis, so the arm was dropped and the sweep run as 128 vs 512.

## `...p8192d128s1024-mr16.txt` + the raw logs — CAP=128, cold JIT

The prefill host had just been restored from a stop/start, which wipes
`/opt/dlami/nvme` — weights *and* the host JIT cache tree. Consequences:

- prefill took 420 s to become ready instead of 210 s, so the warmup run r0 spent
  its life in `KVPoll.Bootstrapping` and produced no metrics at all;
- r1 therefore absorbed the deep_gemm JIT compile **inside the timed region**:
  54.08 s / TTFT 15.8 s / 1211.9 tok/s, against 1684 tok/s warm. Do not average it
  with anything;
- only r2 (1690.9 tok/s) is clean, i.e. n=1.

The arm was re-run once the caches were warm and gave 1687.5 / 1684.1 / 1678.7 —
and this arm's surviving r2 agrees with those to within 0.5 %, which is the only
reason it appears here as corroboration rather than as data.

The raw `.log`/`.json`/`.decode.log` files in this directory are gitignored
(`results/**/*.log`); they exist on the laptop only.
