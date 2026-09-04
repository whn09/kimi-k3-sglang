# Kimi-K3 on sglang `--moe-a2a-backend deepep_v2` — the three blockers and the patch

Image: `lmsysorg/sglang:nightly-dev-cu13-20260901-07c8f729`
(sglang `0.0.0.dev1+g07c8f7294`, torch 2.13.0+cu130, nvcc 13.0.88, sgl-deep-gemm 0.1.7)
Host: 1 × p6-b300 (8 × B300 SXM6, sm_103), `ap-northeast-2`, **spot**.

Out of the box, `--moe-a2a-backend deepep_v2` on Kimi-K3 fails at **five**
independent points, plus one host/fabric issue. Four of the five are validation
that is stricter than the kernels behind it; the fifth is a real upstream bug
(the user's own PR). With all five patched the server reaches READY in 160 s and
answers correctly. The diffs are marked `LOCAL EXPERIMENT ONLY`: passing a gate
is not a correctness proof.

| # | file | what | kind |
|---|---|---|---|
| 1 | `moe_hook.diff` | architecture whitelist | validation |
| 2 | `kimi_k3.diff` | K3's two hand-rolled EP-a2a lists omit v2 | **real wiring gap** |
| 3 | `fmt_layer.diff` | FP4/MXFP4 quant-method gate | validation |
| 4 | `mr_deep_gemm.diff` | v2 pre-permute rejects the `situ` activation | validation |
| 5 | `ep_moe_kernels.diff` | `ep_scatter_from_psum` missing kernel args | **real bug — sglang PR #37211** |

## Blocker 1 — architecture whitelist (`moe_hook.diff`)

`srt/arg_groups/moe_hook.py:416-446` `validate_deepep_v2_model_architecture`:

```
ValueError: DeepEP v2 MoE is not validated for 'KimiK3ForConditionalGeneration';
supported architectures are ['DeepseekV3ForCausalLM', 'DeepseekV4ForCausalLM',
'Qwen3MoeForCausalLM']. Other model workflows may require an all-reduce after
A2A combine. Use --moe-a2a-backend deepep.
```

Fires in `run_resolution_pipeline` (`arg_groups/pipeline.py:306`) **before any
weight load**, and it is **independent of `--disaggregation-mode`** — verified
identical under `unified`, `prefill` and `decode`, so a PD split cannot route
around it.

Note the error's own warning: "may require an all-reduce after A2A combine".
That is blocker 2.

## Blocker 2 — K3's hand-rolled EP-a2a lists omit v2 (`kimi_k3.diff`)

`srt/models/kimi_k3.py` builds its own list of "is this an EP all-to-all
backend" at two sites (~`:542` and `:2193`) and enumerates megamoe / deepep /
mooncake / ascend_fuseep / mori — **but not `deepep_v2`**. The central helper
`layers/moe/utils.py:577` `is_deepep_class_backend()` *does* include v2;
`kimi_k3.py` does not use it.

Consequence if unpatched: `self._ep_a2a` stays `False`, so the MoE region keeps
its DP-gather / TP-reduce while the v2 dispatcher has already done the a2a —
exactly the hazard blocker 1's message describes. This is why patching only the
whitelist is not enough.

## Blocker 3 — the FP4 quant gate (`fmt_layer.diff`) ← the user's hypothesis

`srt/layers/moe/fused_moe_triton/layer.py:240-263`
`_validate_deepep_v2_quant_method`, called per MoE layer from `FusedMoE.__init__`:

```
ValueError: --moe-a2a-backend deepep_v2 requires 128x128 blockwise FP8 experts
with dynamic activation scaling, but this layer selected Mxfp4MoEMethod.
Use a compatible checkpoint or --moe-a2a-backend deepep.
```

It rejects five things: any non-`Fp8MoEMethod`, `use_mxfp8`, `is_fp4_expert`,
`weight_block_size != [128,128]`, and `activation_scheme != "dynamic"`.

**So yes — sglang refuses FP4 experts on deepep_v2. But it is NOT because
deep_gemm can't do it:**

| | evidence |
|---|---|
| deep_gemm 0.1.7 ships the masked W4A8 GEMM | `m_grouped_fp8_fp4_gemm_nt_masked`, `m_grouped_fp8_fp4_gemm_nt_contiguous` |
| the v2 **masked** runner already has an FP4 branch | `moe_runner/deep_gemm.py:646` → `recipe_a=(1,128), recipe_b=(1,32)` |
| the v2 **contiguous** runner already has an FP4 branch | `moe_runner/deep_gemm.py:343`, same recipes |
| K3's MXFP4 method asks for exactly that | `quantization/mxfp4.py:1443-1452` → `DeepGemmMoeQuantInfo(use_fp8=True, block_shape=[128,128], is_fp4_experts=True)` |
| the v2 dispatcher's activation quant matches `recipe_a` | `token_dispatcher/deepep_v2.py` `_SCALE_BLOCK_SIZE = 128`, `sglang_per_token_group_quant_fp8` |

i.e. the activation side the v2 dispatcher produces (per-token 128-group FP8)
*is* `recipe_a=(1,128)`, and the FP4 weight side is `recipe_b=(1,32)`. The gate
is stricter than the code path behind it. The patch lets `Mxfp4MoEMethod`
through when `--moe-runner-backend deep_gemm`.

**This is a hypothesis under test, not a result.** Scale-format details
(`DEEPGEMM_SCALE_UE8M0`, TMA-aligned scales, `transform_sf_into_required_layout`)
are not proven to line up. Any run that boots must be validated numerically
against the v1 baseline below; a coherent-looking completion is not enough.

## Blocker 4 — the v2 pre-permute rejects K3's activation (`mr_deep_gemm.diff`)

`srt/layers/moe/moe_runner/deep_gemm.py:1595`, inside
`pre_permute_deepep_v2_to_deep_gemm`:

```python
assert runner_config.activation == "silu"      # AssertionError, K3 is "situ"
```

K3's `hidden_act` is `situ` (with `activation_situ_beta` / `_linear_beta`, which
`kimi_k3.py:473-475` passes as `gemm1_alpha` / `gemm1_clamp_limit`).

The same file shows this is an oversight, not a limitation — **everything else on
the v2 path already handles `situ`**:

| site | code |
|---|---|
| v1 pre-permute, one screen up | `:1244` `assert runner_config.activation in ("silu", "situ")` |
| the runner constructor | `:272` same pair, comment: "needs the masked-gemm activation site to branch" |
| v2 **contiguous** activation | `:378` `if self.config.activation == "situ":` → `_situ_mul_quant_contig_kernel` |
| v2 **masked** activation | `:702` → `_varlen_deep_gemm_situ_mul_quant` → `sglang.kernels.ops.kimi_k3.situ_and_mul_masked_post_quant` |

The last row is a **K3-specific CUDA kernel that already lives on the deepep_v2
masked path**. Only the one assert was not widened.

## Blocker 5 — `ep_scatter_from_psum` missing kernel args (`ep_moe_kernels.diff`)

This is **sglang PR #37211** (author `whn09`, OPEN as of 2026-09-01), applied
verbatim: `ep_scatter_from_psum` launches `_fwd_kernel_ep_scatter_2` without the
`expert_start` / `num_experts` parameters the kernel has required since
`5f216fc3`, so every rank dies with
`TypeError: dynamic_func() missing 2 required positional arguments`.

Verified still missing in this nightly (`07c8f729`). Reached only from
`pre_permute_deepep_v2_to_deep_gemm`'s **non-masked** branch, i.e. only from
deepep_v2 prefill — which is exactly the path K3 takes once blockers 1-4 are out
of the way. `scale_ue8m0` is passed by keyword at `deep_gemm.py:1687`, so
inserting `expert_start: int = 0` ahead of it is safe.

## Host issue — this B300 box is InfiniBand, not EFA

`lspci`: 1 × ENA + **2 × Mellanox ConnectX-7 (MT2910)**, zero EFA devices; the
`efa` module is loaded (EFA installer 1.49.0) but binds nothing. So:

- the GIN backend here is **GDAKI, `NCCL_GIN_TYPE=3`** — `EFA_GDA` (5) has no
  device to drive.
- DeepEP asserts `ginType != NONE` even for a single-node `direct` run
  (`csrc/kernels/backend/nccl.cu:87`), so the NICs cannot simply be excluded;
  without `--device=/dev/infiniband` NCCL sees no network and the assert fires.
- **the two HCAs are on two IB planes with no path between them.** GDAKI context
  creation died with `IBV_WC_RETRY_EXC_ERR(12)` on `ibp199s0f0`
  (`ncclGinIbGdakiCreateContext` → `gin.cc:288`). Isolated in seconds without
  touching the server:

  ```
  ibv_rc_pingpong -d ibp198s0f0 ... localhost   ->  11.21 usec/iter   OK
  server on ibp198s0f0, client on ibp199s0f0    ->  "transport retry counter
                                                     exceeded (12)"
  ```

  Fix: `NCCL_IB_HCA=ibp198s0f0` pins one plane. (DeepEP's own assert text names
  this case: "in multi-plane network".)

Both ports are PORT_ACTIVE, 100 Gb/s 2X HDR, `active_mtu 512`, LIDs 1 and 4 under
one SM.

## Memory: capacity 4096 does not fit

`CAP=4096` makes the ElasticBuffer ask for **42.00 GiB** per rank on top of
243.6 GiB already in use → `torch.OutOfMemoryError`. `CAP=512 CHUNK=512
MEMFRAC=0.85` boots (`max_total_num_tokens=256576`). `MEMFRAC=0.75` is too low
the other way: `RuntimeError: Not enough GPU memory for hybrid (mamba/
linear-attention) state cache`.

## The v1 baseline (works unpatched, use it as the reference)

```
--moe-a2a-backend deepep --deepep-mode auto \
--moe-runner-backend deep_gemm --disable-prefill-cuda-graph
```

READY in 220 s, real `/v1/chat/completions` returns coherent text. Two traps:

- `--moe-runner-backend auto` resolves to `flashinfer_mxfp4`, which leaves
  `deprecate_flag=False` and dies in
  `AssertionError: forward_deepgemm_masked is deprecated`
  (`ep_moe/layer.py:280-310`). MXFP4 + DeepEP is only wired on the `deep_gemm`
  runner — `ep_moe/layer.py:129-137` names Kimi K3 in its comment.
- without `--disable-prefill-cuda-graph`: `cudaErrorStreamCaptureInvalidated`
  in `capture_prefill_graph`, first raised at `kimi_k3.py:1111`
  `torch.cuda.current_stream().wait_event(shared_event)` — a cross-stream wait
  that is not capturable. (deepep_v2 disables the prefill graph itself, per
  `moe_hook.py:182-243`.)

## Reproducing

**These five diffs are now applied inside the image** — see the "DeepEP v2 source
patches" step in `../Dockerfile`, which `patch`es each one by target path and
*fails the build* if a diff neither applies nor is already present. So the normal
way to reproduce is:

```bash
docker build -t kimi-k3-efa-v2:latest -f ../Dockerfile ..
bash ../10_launch_standalone.sh          # or 20_/21_ for 1P1D
```

That replaced the older flow, in which `../run_k3_v2.sh {unified|prefill|decode}`
bind-mounted patched whole-file copies from `/opt/dlami/nvme/patch` over the
image. `run_k3_v2.sh` still works and is still the fastest way to iterate on a
diff on a live host, but it only ever worked on the one host that had that
directory — a second machine silently got the **unpatched** behaviour, which for
`kimi_k3.diff` means wrong numerics rather than an error. `require_efa_image()`
in `../env_common.sh` now refuses to launch on an image that cannot prove the
patches are in it.

To re-cut a diff against a new base image:

```bash
P=/opt/dlami/nvme/patch; mkdir -p $P/orig
cid=$(docker create --entrypoint true lmsysorg/sglang:nightly-dev-cu13-20260901-07c8f729)
B=/sgl-workspace/sglang/python/sglang
docker cp $cid:$B/srt/models/kimi_k3.py                    $P/orig/kimi_k3.py
docker cp $cid:$B/srt/arg_groups/moe_hook.py               $P/orig/moe_hook.py
docker cp $cid:$B/srt/layers/moe/fused_moe_triton/layer.py $P/orig/fmt_layer.py
docker cp $cid:$B/srt/layers/moe/moe_runner/deep_gemm.py   $P/orig/mr_deep_gemm.py
docker cp $cid:$B/kernels/ops/moe/ep_moe_kernels.py        $P/orig/ep_moe_kernels.py
docker rm -f $cid
for f in kimi_k3 moe_hook fmt_layer mr_deep_gemm ep_moe_kernels; do
  cp $P/orig/$f.py $P/$f.py
  patch $P/$f.py < patches/$f.diff
done
```

`validate_deepep_v2_dispatch_token_budget` (`moe_hook.py:381`) rejects the
upstream default capacity of 128 against any real prefill budget, so a CAP has to
be set. **It is not 4096** — an earlier version of this file said `CAP=4096
CHUNK=4096` were "required", and that config cannot start: the ElasticBuffer costs
10.5 GiB per 1024 of CAP, so 4096 asks 42.00 GiB of the ~39 GB available and OOMs
(3072 asks 31.50 GiB and also OOMs). The measured ceilings are **2048 with CUDA
graphs off** and **1024 with them on**; `../env_common.sh` sets prefill and decode
separately for exactly this reason.

## Status (2026-09-01, updated)

With all five patches the **unified** server serves and is **numerically
bit-identical to the v1 deepep baseline**: first-token top-5 logprobs match to the
last digit on all three probe prompts. Prefill (ISL=8K) is 6.6% faster than v1 at
matched chunk; decode TPOT is at parity but shows a 5.4x TTFT *mean/median* gap
(a tail stall) that halves end-to-end decode throughput.

Full numbers, the memory budget that caps CHUNK at 2048 on one node, and the
launcher knobs: `../results/deepep_v2_on_k3_b300.md`.

The PD-disaggregated arms (`--disaggregation-mode prefill|decode`) get past every
gate and reach `Load weight begin` on all 8 ranks, but have **not** been run to
READY or benchmarked.
