# Building Mooncake with EFA + CUDA on the AWS Deep Learning AMI

Verified 2026-07-31 on a freshly restarted p5.48xlarge (Ubuntu 24.04 DLAMI,
32 EFA NICs, H100). The instance store (`/opt/dlami/nvme`) is wiped on stop/start,
so this is the from-scratch path.

On the DLAMI both Python 3.13 and the CUDA toolkit live inside the `/opt/pytorch`
virtualenv — there is **no `/usr/local/cuda`**. Everything below follows from that.

## Recipe

```bash
sudo bash dependencies.sh -y          # needs root, else it fails silently

source /opt/pytorch/bin/activate
export CUDA_HOME=/opt/pytorch/lib/python3.13/site-packages/nvidia/cu13
export PATH=$CUDA_HOME/bin:$PATH
export CPLUS_INCLUDE_PATH=$CUDA_HOME/include:$CPLUS_INCLUDE_PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib:$LD_LIBRARY_PATH
export LIBRARY_PATH=$CUDA_HOME/lib:$LIBRARY_PATH

cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DUSE_EFA=ON -DUSE_CUDA=ON \
  -DWITH_STORE=OFF -DWITH_STORE_RUST=OFF -DWITH_P2P_STORE=OFF \
  -DBUILD_UNIT_TESTS=ON
cmake --build build -j $(nproc)
```

`/opt/pytorch/cuda` is a symlink to the same `nvidia/cu13` directory, so either
path works as `CUDA_HOME`.

## Why `CPLUS_INCLUDE_PATH` and not a `/usr/local/cuda` symlink

Both make the build succeed, but the env var is the right one:

`mooncake-transfer-engine/nvlink-allocator` is driven by its own `build.sh`
invoked from CMake, so it never sees `-DCUDAToolkit_ROOT`, `-DCMAKE_CXX_FLAGS`,
or any other CMake-level include path. It calls `g++` directly. `CPLUS_INCLUDE_PATH`
is read by `g++` itself, so it reaches that subprocess; CMake variables do not.

Verified by A/B with no `/usr/local/cuda` present:

```
$ g++ -fsyntax-only -std=gnu++20 -DUSE_CUDA -I../include nvlink_allocator.cpp
  with CPLUS_INCLUDE_PATH set   -> exit 0
  with CPLUS_INCLUDE_PATH unset -> fatal error: cuda.h: No such file or directory
```

A `sudo ln -s /opt/pytorch/cuda /usr/local/cuda` also works but needs root and
leaves a system-wide change behind for a per-build problem. Prefer the env vars.

## Failure modes and what they mean

| Symptom | Cause |
|---|---|
| `Could not find nvcc, please set CUDAToolkit_ROOT` | venv not activated / `$CUDA_HOME/bin` not on `PATH` |
| `fatal error: cuda.h: No such file or directory` | `CPLUS_INCLUDE_PATH` not set (this is the `nvlink-allocator` failure) |
| `cannot find -lcudart` | `LIBRARY_PATH` / `LD_LIBRARY_PATH` not set |
| `ModuleNotFoundError: No module named 'mooncake.engine'` | `.so` built against the wrong Python, **or** see below |
| `WITH_STORE_RUST=ON requires WITH_STORE=ON` | turning off `WITH_STORE` requires turning off `WITH_STORE_RUST` too |
| `✗ ERROR: Require root permission` from `dependencies.sh` | run it with `sudo` |

### The `mooncake.engine` import error has a second cause

Even with a correct build, the extension lands *outside* the package directory:

```
build/mooncake-integration/engine.cpython-313-x86_64-linux-gnu.so   <- built here
build/mooncake-integration/mooncake/__init__.py                     <- package here
```

so `PYTHONPATH=build/mooncake-integration` finds `mooncake/` but not
`mooncake.engine`. Link it in:

```bash
B=$PWD/build/mooncake-integration
ln -sfn $B/engine.cpython-313-x86_64-linux-gnu.so $B/mooncake/
```

## Always verify the GPU registration path

A build that misses the CUDA headers can still link and run, but registers GPU
pointers through `fi_mr_reg` instead of `fi_mr_regattr` + `FI_HMEM_CUDA`. It looks
*faster*, and every registration measurement taken on it is wrong. Check:

```bash
nm -C build/mooncake-transfer-engine/src/libtransfer_engine.a \
  | grep -c "cuPointerGetAttribute\|cuMemGetAddressRange"   # expect >= 2
```

Also confirm EFA itself came back after a restart — `fi_info` is not on `PATH`:

```bash
ls /sys/class/infiniband/ | wc -l              # expect 32 on p5.48xlarge
/opt/amazon/efa/bin/fi_info -p efa | grep -c "provider: efa"
lsmod | grep -E "^efa|efa_nv_peermem"
```
