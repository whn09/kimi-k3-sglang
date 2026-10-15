"""Kimi-K3 on --moe-a2a-backend deepep_v2, for sglang main (validated on a6a8c560).

The 2026-09-01 base needed five diffs (patches/*.diff, see patches/README.md).
On a6a8c560 one of them is fixed upstream (ep_scatter_from_psum now passes
expert_start/num_experts -- the bug PR #37211 was opened for) and K3 was
refactored onto stage boundaries, so the other four are re-cut here against
the new code. Same four semantics as before:

  1. registry   configs/moe_model_registry.py -- deepep_v2 is now gated on a
                per-architecture registry (was a whitelist in moe_hook.py) and
                K3 is not in it. Default policy: no FP32 silu-mul (K3 is SiTU,
                handled in the runner) and the default prefill sizing ("the
                ceiling split across the ranks each per-DP chunk is scattered
                over"), which is K3's SP-MoE layout.
  2. kimi_k3    models/kimi_k3.py:_uses_ep_a2a() -- the list of a2a backends
                that move rows to their experts themselves. WITHOUT deepep_v2
                here the MoE region keeps its DP gather / TP reduce on top of
                the v2 a2a and serves WRONG NUMERICS with no error.
  3. quant gate layers/moe/fused_moe_triton/layer.py -- the v2 validator only
                admits Fp8MoEMethod; K3's experts are Mxfp4MoEMethod, whose
                deep_gemm branch builds DeepGemmMoeQuantInfo(is_fp4_experts=
                True), and both v2 runner paths carry the fp4 recipe. Admitted
                only with the deep_gemm runner.
  4. situ       layers/moe/moe_runner/deep_gemm.py -- the v2 pre-permute still
                asserts activation == "silu", while the runner ctor and the v1
                pre-permute accept ("silu", "situ") and both v2 activation
                sites have a situ branch.

Usage (image build): python3 k3_deepep_v2_main.py <sglang package dir>
Each anchor must match exactly once, else the build fails -- a moved base must
be re-read, never silently shipped. Re-applying is a no-op (SKIP).
Numerics MUST be re-validated on every new base (first-token logprobs vs the
--moe-a2a-backend deepep / none baseline), because patch 2 failing open is
silent.
"""
import sys
from pathlib import Path

root = Path(sys.argv[1])
TAG = "K3-DEEPEP-V2"


def patch(rel, old, new):
    p = root / rel
    s = p.read_text()
    if TAG in s and new in s:
        print(f"  SKIP    {rel} (already patched)")
        return
    n = s.count(old)
    assert n == 1, f"{rel}: expected exactly 1 anchor, found {n}:\n{old}"
    p.write_text(s.replace(old, new))
    print(f"  patched {rel}")


patch(
    "srt/configs/moe_model_registry.py",
    '    "MiMoV2FlashForCausalLM": _DeepEPv2ModelPolicy(),\n',
    '    "MiMoV2FlashForCausalLM": _DeepEPv2ModelPolicy(),\n'
    f'    "KimiK3ForConditionalGeneration": _DeepEPv2ModelPolicy(),  # {TAG}\n',
)

patch(
    "srt/models/kimi_k3.py",
    """        _a2a_backend.is_megamoe()
        or _a2a_backend.is_deepep()
        or _a2a_backend.is_mooncake()
""",
    f"""        _a2a_backend.is_megamoe()
        or _a2a_backend.is_deepep()
        or _a2a_backend.is_deepep_v2()  # {TAG}
        or _a2a_backend.is_mooncake()
""",
)

patch(
    "srt/layers/moe/fused_moe_triton/layer.py",
    """    if isinstance(quant_method, UnquantizedFusedMoEMethod):
        return

    config = (
        quant_method.quant_config if isinstance(quant_method, Fp8MoEMethod) else None
    )
""",
    f"""    if isinstance(quant_method, UnquantizedFusedMoEMethod):
        return

    # {TAG}: MXFP4 experts through deep_gemm (is_fp4_experts recipe).
    from sglang.srt.layers.quantization.mxfp4 import Mxfp4MoEMethod

    if isinstance(quant_method, Mxfp4MoEMethod) and get_moe_runner_backend().is_deep_gemm():
        return

    config = (
        quant_method.quant_config if isinstance(quant_method, Fp8MoEMethod) else None
    )
""",
)

# The v2 pre-permute is the only `== "silu"` assert left in the runner.
p = root / "srt/layers/moe/moe_runner/deep_gemm.py"
s = p.read_text()
head = s.index("def pre_permute_deepep_v2_to_deep_gemm(")
body = s[head:]
old = '    assert runner_config.activation == "silu"\n'
new = f'    assert runner_config.activation in ("silu", "situ")  # {TAG}\n'
if new in body:
    print("  SKIP    srt/layers/moe/moe_runner/deep_gemm.py (already patched)")
else:
    first = body.index(old)  # first assert inside the v2 pre-permute
    nxt = body.find("\ndef ", 1)
    assert first < nxt, "the silu assert is no longer inside pre_permute_deepep_v2_to_deep_gemm"
    p.write_text(s[:head] + body[:first] + new + body[first + len(old):])
    print("  patched srt/layers/moe/moe_runner/deep_gemm.py")

print("K3 deepep_v2 patches for sglang main: all four accounted for")
