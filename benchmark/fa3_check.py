"""Is an exact FA3 split faster than flex for the Wan-Animate-2 in-context attention?

  modal run fa3_check.py    1x H100, ~10 min, 30 min cap

Split: dense FA3 over the generation keys (+ a separate chunk for generation frame 1 when
log_scale != 0, whose LSE gets +log_scale), per-frame FA3 over the reference keys, merged by LSE.
Shape = the full run: 944x528 -> 33x59 = 1947 tokens/frame, 21 gen frames, 20 reference frames.
"""

import json

import modal

T = "<path to trees>"
app = modal.App("wan-animate2-fa3-check")
image = modal.Image.from_registry("lmsysorg/sglang:dev").add_local_dir(
    f"{T}/headfull/python/sglang", "/root/head/sglang", copy=True, ignore=["**/__pycache__", "**/.claude"]
)

CODE = r'''
import json, statistics, torch
from types import SimpleNamespace
from torch.nn.attention.flex_attention import create_block_mask
from sglang.multimodal_gen.runtime.models.dits import wan_animate_2_block as blk
from sglang.kernels.ops.attention.flash_attention_v3 import flash_attn_varlen_func as fa3

dev, dt, H, D = "cuda", torch.bfloat16, 40, 128
f, h, w = 21, 33, 59
hw = h * w
layout = blk._InContextLayout((f, h, w), (f - 1, h, w), (f, h, w))
N, R = layout.generation_video_num_tokens, layout.reference_video_num_tokens
mask = create_block_mask(layout.block_mask_fn(), B=None, H=None, Q_LEN=layout.padded_generation_video_len,
                         KV_LEN=layout.padded_kv_len, device=dev, _compile=True)
g = torch.Generator(device=dev).manual_seed(0)
q, k, v = (torch.randn(1, N, H, D, device=dev, dtype=dt, generator=g) for _ in range(3))
rk, rv = (torch.randn(1, R, H, D, device=dev, dtype=dt, generator=g) for _ in range(2))
pk, pv = blk._pad_reference_video_kv(rk, layout), blk._pad_reference_video_kv(rv, layout)

def i32(xs):
    return torch.tensor(xs, dtype=torch.int32, device=dev)

def dense(qq, kk, vv):
    r = fa3(qq, kk, vv, i32([0, qq.shape[0]]), i32([0, kk.shape[0]]), qq.shape[0], kk.shape[0],
            return_softmax_lse=True)
    return r[0], r[1]  # [Nq, H, D], [H, Nq]

# Per-frame reference: queries of gen frames 1..f-1 attend reference frame (i-1); frame 0 attends none.
cu_q = i32([j * hw for j in range(f - 1 + 1)])
cu_k = i32([j * hw for j in range(f - 1 + 1)])

def split(log_scale):
    qq, kk, vv = q[0], k[0], v[0]
    parts = []
    if log_scale == 0.0:
        parts.append(dense(qq, kk, vv))
    else:
        # frame-1 keys biased by log_scale: softmax weight * e^c  <=>  lse + c
        o0, l0 = dense(qq, torch.cat([kk[:hw], kk[2 * hw:]]), torch.cat([vv[:hw], vv[2 * hw:]]))
        o1, l1 = dense(qq, kk[hw:2 * hw], vv[hw:2 * hw])
        parts += [(o0, l0), (o1, l1 + log_scale)]
    o_ref, l_ref = fa3(qq[hw:], rk[0], rv[0], cu_q, cu_k, hw, hw, return_softmax_lse=True)[:2]
    o_ref = torch.cat([torch.zeros(hw, H, D, device=dev, dtype=o_ref.dtype), o_ref])
    l_ref = torch.cat([torch.full((H, hw), float("-inf"), device=dev), l_ref.float()], dim=1)
    parts.append((o_ref, l_ref))
    lses = torch.stack([l.float() for _, l in parts])  # [P, H, N]
    m = lses.max(0).values
    ws = torch.exp(lses - m)  # [P, H, N]
    num = sum(o.float() * wgt.transpose(0, 1).unsqueeze(-1) for (o, _), wgt in zip(parts, ws))
    return (num / ws.sum(0).transpose(0, 1).unsqueeze(-1)).to(dt).unsqueeze(0)

def flex(log_scale):
    return blk.WanAnimate2TransformerBlock._in_context_attention(
        SimpleNamespace(log_scale=log_scale), q, k, v, pk, pv, mask, layout)

def timeit(fn, iters=10):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        a, b = torch.cuda.Event(True), torch.cuda.Event(True)
        a.record(); fn(); b.record(); torch.cuda.synchronize()
        ts.append(a.elapsed_time(b))
    return round(statistics.median(ts), 2), round(statistics.pstdev(ts), 2)

out = {"shape": {"gen_tokens": N, "ref_tokens": R, "heads": H, "head_dim": D},
       "gpu": torch.cuda.get_device_name()}
for ls in (0.0, -1.3):
    ref, alt = flex(ls).float(), split(ls).float()
    d = alt - ref
    out[f"log_scale={ls}"] = {
        "flex_ms": timeit(lambda: flex(ls)), "fa3_split_ms": timeit(lambda: split(ls)),
        "rel_l2_vs_flex": (d.norm() / ref.norm()).item(), "max_abs_vs_flex": d.abs().max().item(),
    }
out["fa3_dense_only_ms"] = timeit(lambda: dense(q[0], k[0], v[0]))
out["fa3_ref_per_frame_only_ms"] = timeit(lambda: fa3(q[0, hw:], rk[0], rv[0], cu_q, cu_k, hw, hw, return_softmax_lse=True))
# Reference point for the error scale: flex vs an fp32 SDPA over the same mask is too big at this
# size, so compare both against each other only; bf16 rounding alone gives rel_l2 ~ 3e-3.
print(json.dumps(out))
'''


@app.function(image=image, gpu="H100", cpu=8, memory=65536, timeout=1800)
def check():
    import os
    import subprocess
    import sys

    p = subprocess.run([sys.executable, "-c", CODE], env=dict(os.environ, PYTHONPATH="/root/head"),
                       cwd="/root/head", capture_output=True, text=True)
    return p.stdout.strip().splitlines()[-1] if p.returncode == 0 else p.stderr[-4000:]


@app.local_entrypoint()
def main():
    out = check.remote()
    print(out)
    open(__file__.replace("fa3_check.py", "fa3_out.json"), "w").write(out)
