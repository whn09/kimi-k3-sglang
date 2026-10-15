"""Log the SM count DeepEP v2 dispatch actually uses, once per rank.

sglang already exposes the knob (SGLANG_DEEPEP_V2_NUM_SMS, passed straight to
ElasticBuffer.dispatch; combine keeps num_sms=0 = "reuse the dispatch handle's
count"). What it never says is what 0 ("auto") RESOLVES to: DeepEP's
get_theoretical_num_sms(), which returns max(theory, 64) because sglang builds
the buffer with prefer_overlap_with_compute=False. A 12-vs-auto comparison has
to record the auto number, so this prints, on every rank's first dispatch:

    [k3-sms] DeepEP v2 dispatch num_sms=<n> (SGLANG_DEEPEP_V2_NUM_SMS=<env>) rank=<r>

No behaviour change: the value printed is computed with the same arguments
dispatch() itself uses (get_theoretical_num_sms is cached), and dispatch still
receives the env value unchanged.

Applied at image build: python3 deepep_v2_log_sms.py <sglang package dir>.
Asserts its anchor matched exactly once, so a moved base fails the build.
"""
import sys
from pathlib import Path

p = Path(sys.argv[1]) / "srt/layers/moe/token_dispatcher/deepep_v2.py"
s = p.read_text()
MARK = "[k3-sms]"
if MARK in s:
    print(f"  SKIP  {p} (already patched)")
    sys.exit(0)

old = """        buffer = self._get_buffer()
        recv_x, recv_topk_idx, recv_topk_weights, handle, event = buffer.dispatch(
"""
new = """        buffer = self._get_buffer()
        if not getattr(buffer, "_k3_sms_logged", False):
            _req = envs.SGLANG_DEEPEP_V2_NUM_SMS.get()
            _used = _req or buffer.get_theoretical_num_sms(
                self.num_experts, topk_ids.shape[1]
            )
            print(
                f"[k3-sms] DeepEP v2 dispatch num_sms={_used} "
                f"(SGLANG_DEEPEP_V2_NUM_SMS={_req}) rank={dist.get_rank()}",
                flush=True,
            )
            buffer._k3_sms_logged = True
        recv_x, recv_topk_idx, recv_topk_weights, handle, event = buffer.dispatch(
"""
n = s.count(old)
assert n == 1, f"{p}: expected 1 anchor, found {n}"
p.write_text(s.replace(old, new))
print(f"  patched {p}")
