#!/usr/bin/env python3
"""
Replay Kimi-K3's KV-cache registration the way SGLang actually issues it.

The topology matters more than the sizes, and an earlier single-process version
of this benchmark got it wrong. SGLang's MooncakeTransferEngine is a *module*
level singleton (mooncake_transfer_engine.py), initialized from
ModelRunner.init_shared_mooncake_transfer_engine() per gpu_id. With tp_size=8
that is 8 scheduler *processes*, so 8 independent TransferEngines and 8
independent libfabric domains on one node -- not one engine shared by 8 ranks.

Confirmed against a real K3 prefill log (2x p6-b300, 16 NICs): 8 distinct pids
each enumerate all 18 RDMA devices and listen on their own RPC port, and every
bucket of the size histogram divides by 8 exactly:

      2560 B  x  7      (aux / state scalars)
      5120 B  x  1
     20480 B  x  2
   2792448 B  x 69      (2.7 MiB)
  79429632 B  x 69      (75.8 MiB)
 230842368 B  x 24      (220.1 MiB)
 410271744 B  x 10      (391.3 MiB)
                 182 buffers / 14.3 GiB per rank, 1456 / 114 GiB per node

This is what makes MC_MAX_CONCURRENT_REG_MR hard to reason about: it caps each
process independently, so "cap 8" on a tp8 node means 64 registrations in flight
globally, not 8. The real log peaks at 1104 in flight (computed from the
per-buffer durations), which is 8 x ~138, not 1456.

Each rank registers on its own GPU because that is where its KV pool lives, and
issues the calls the way register_buffer_to_engine() does: one
batch_register_memory() for kv_data, a second for aux_data.

Ranks rendezvous on a barrier directory before registering, so the contention
being measured is the real thing -- 8 domains hitting the same NICs at once --
rather than 8 staggered runs.

Usage: regbench_k3_mp.py <ip> <rank> <nranks> <barrier_dir> [scale]
Prints: RESULT <rank> <cap> <nbuffers> <bytes> <total_ms> <kv_ms> <aux_ms>
"""

import ctypes
import os
import sys
import time

cuda = ctypes.CDLL("libcuda.so.1")

# Without explicit argtypes ctypes passes size_t as 32-bit, silently truncating
# any allocation >= 4 GiB.
cuda.cuMemAlloc_v2.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
cuda.cuMemsetD8_v2.argtypes = [ctypes.c_void_p, ctypes.c_ubyte, ctypes.c_size_t]
cuda.cuDeviceGet.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
cuda.cuDevicePrimaryCtxRetain.argtypes = [ctypes.POINTER(ctypes.c_void_p),
                                          ctypes.c_int]
cuda.cuCtxSetCurrent.argtypes = [ctypes.c_void_p]
cuda.cuGetErrorString.argtypes = [ctypes.c_int,
                                  ctypes.POINTER(ctypes.c_char_p)]

# Per-rank counts, i.e. the log histogram divided by 8.
KV_DATA = [(410271744, 10), (230842368, 24), (79429632, 69), (2792448, 69)]
AUX_DATA = [(20480, 2), (5120, 1), (2560, 7)]

BARRIER_TIMEOUT_S = 900


def chk(rc, what):
    if rc != 0:
        err = ctypes.c_char_p()
        cuda.cuGetErrorString(rc, ctypes.byref(err))
        msg = err.value.decode() if err.value else "?"
        raise RuntimeError(f"{what} failed: {rc} ({msg})")


def alloc(spec):
    addrs, sizes = [], []
    for size, count in spec:
        for _ in range(count):
            ptr = ctypes.c_void_p()
            chk(cuda.cuMemAlloc_v2(ctypes.byref(ptr), size),
                f"cuMemAlloc({size})")
            chk(cuda.cuMemsetD8_v2(ptr, 0, size), "cuMemsetD8")
            addrs.append(ptr.value)
            sizes.append(size)
    return addrs, sizes


def barrier(bdir, rank, nranks, tag):
    """Wait until every rank has reached `tag`. Fails loudly on timeout rather
    than letting one slow rank turn the run into a staggered one."""
    open(os.path.join(bdir, f"{tag}.{rank}"), "w").close()
    deadline = time.time() + BARRIER_TIMEOUT_S
    while True:
        n = len([f for f in os.listdir(bdir) if f.startswith(f"{tag}.")])
        if n >= nranks:
            return
        if time.time() > deadline:
            raise RuntimeError(f"barrier {tag}: only {n}/{nranks} after "
                               f"{BARRIER_TIMEOUT_S}s")
        time.sleep(0.02)


def main():
    ip, rank, nranks, bdir = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), \
        sys.argv[4]
    scale = float(sys.argv[5]) if len(sys.argv) > 5 else 1.0
    cap = os.environ.get("MC_MAX_CONCURRENT_REG_MR", "unset")

    kv = [(s, max(1, int(c * scale))) for s, c in KV_DATA]
    aux = [(s, max(1, int(c * scale))) for s, c in AUX_DATA]
    order = os.environ.get("REGBENCH_ORDER", "desc")

    # One GPU per rank, like a real TP rank's KV pool.
    chk(cuda.cuInit(0), "cuInit")
    dev = ctypes.c_int()
    chk(cuda.cuDeviceGet(ctypes.byref(dev), rank), f"cuDeviceGet({rank})")
    ctx = ctypes.c_void_p()
    chk(cuda.cuDevicePrimaryCtxRetain(ctypes.byref(ctx), dev),
        f"cuDevicePrimaryCtxRetain({rank})")
    chk(cuda.cuCtxSetCurrent(ctx), "cuCtxSetCurrent")

    from mooncake.engine import TransferEngine

    engine = TransferEngine()
    if engine.initialize(f"{ip}:{12001 + rank}", "P2PHANDSHAKE", "efa", "") != 0:
        raise RuntimeError(f"rank {rank}: engine init failed")

    kv_addrs, kv_sizes = alloc(kv)
    aux_addrs, aux_sizes = alloc(aux)
    chk(cuda.cuCtxSynchronize(), "cuCtxSynchronize")

    # Reorder the kv call only; aux is 10 tiny buffers and cannot matter.
    #
    # runBoundedParallel hands out tasks by index, so with a cap the input order
    # decides the schedule. KV_DATA is size-descending, which is longest-first --
    # the classic *good* order for a fixed pool. "asc" is the adversarial one: a
    # 391 MB registration only starts after every 2.7 MB one is done, leaving a
    # tail nothing can overlap with. "layer" is what SGLang actually passes,
    # grouped by pool (c4 -> indexer -> c128) rather than sorted by size, per
    # deepseek_v4_memory_pool.get_contiguous_buf_infos().
    if order != "desc":
        pairs = list(zip(kv_addrs, kv_sizes))
        if order == "asc":
            pairs.sort(key=lambda p: p[1])
        elif order == "layer":
            groups = {}
            for a, sz in pairs:
                groups.setdefault(sz, []).append((a, sz))
            pairs = [x for sz in (79429632, 2792448, 230842368, 410271744)
                     for x in groups.get(sz, [])]
            assert len(pairs) == len(kv_addrs), "layer order dropped buffers"
        else:
            raise RuntimeError(f"unknown REGBENCH_ORDER={order}")
        kv_addrs = [a for a, _ in pairs]
        kv_sizes = [s for _, s in pairs]

    total = sum(kv_sizes) + sum(aux_sizes)
    n = len(kv_sizes) + len(aux_sizes)
    print(f"rank={rank} cap={cap} order={order} buffers={n} "
          f"total={total/2**30:.1f}GiB gpu={rank} ready", flush=True)

    barrier(bdir, rank, nranks, "ready")

    t0 = time.perf_counter()
    r1 = engine.batch_register_memory(kv_addrs, kv_sizes)
    t1 = time.perf_counter()
    r2 = engine.batch_register_memory(aux_addrs, aux_sizes)
    t2 = time.perf_counter()

    if r1 or r2:
        print(f"RESULT {rank} {cap} {n} {total} FAILED_{r1}_{r2}", flush=True)
        os._exit(1)
    print(f"RESULT {rank} {cap} {n} {total} {(t2-t0)*1000:.0f} "
          f"{(t1-t0)*1000:.0f} {(t2-t1)*1000:.0f}", flush=True)
    # _exit: skip unregistering 182 MRs on 32 NICs, which would dominate the run.
    os._exit(0)


if __name__ == "__main__":
    main()
