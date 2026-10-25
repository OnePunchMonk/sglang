# Micro-benchmark for LingBot-World bidirectional camera conditioning (P4).
#
# Measures how much of one block's time the per-block camera scale/shift takes,
# at the 14B shape for 480p, 81 frames (S = 21 * 30 * 52 tokens).
#
# The block cost is a proxy: self-attention QKV/out and FFN only (no cross-attn,
# no norms). The camera path uses compute_scale_shift plus a plain-torch modulate
# (the fused JIT modulate kernel needs nvcc).
#
# Run on one H100 with sglang's multimodal_gen dependencies installed, from the
# repo root:  PYTHONPATH=python python scripts/experiments/p4_camera_bench.py

import os

os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("MASTER_PORT", "29511")
os.environ.setdefault("RANK", "0")
os.environ.setdefault("WORLD_SIZE", "1")
os.environ.setdefault("LOCAL_RANK", "0")

import torch
import torch.nn as nn
import torch.nn.functional as F

from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    maybe_init_distributed_environment_and_model_parallel,
)
from sglang.multimodal_gen.runtime.models.dits.lingbot_world import (
    LingBotWorldCamConditioner,
)

assert torch.cuda.is_available(), "needs a GPU"
maybe_init_distributed_environment_and_model_parallel(tp_size=1, sp_size=1)

dev = "cuda"
dtype = torch.bfloat16
D, FFN, HEADS = 5120, 13824, 40
HD = D // HEADS
S = 21 * 30 * 52

cam = LingBotWorldCamConditioner(D).to(dev, dtype).eval()
x = torch.randn(1, S, D, device=dev, dtype=dtype)
c2ws = torch.randn(1, S, D, device=dev, dtype=dtype)

qkv = nn.Linear(D, 3 * D, device=dev, dtype=dtype)
out = nn.Linear(D, D, device=dev, dtype=dtype)
ffn_in = nn.Linear(D, FFN, device=dev, dtype=dtype)
ffn_out = nn.Linear(FFN, D, device=dev, dtype=dtype)


def block_proxy():
    q, k, v = qkv(x).view(1, S, 3, HEADS, HD).permute(2, 0, 3, 1, 4)
    a = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(1, S, D)
    h = out(a)
    return ffn_out(F.gelu(ffn_in(h), approximate="tanh"))


def cam_full():
    s, sh = cam.compute_scale_shift(c2ws)
    return x * (1 + s) + sh


def cam_scale_shift_only():
    return cam.compute_scale_shift(c2ws)


def timeit(fn, iters=10, warm=3):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        ts.append(start.elapsed_time(end))
    ts.sort()
    return ts[len(ts) // 2]


with torch.no_grad():
    t_block = timeit(block_proxy)
    t_cam = timeit(cam_full)
    t_ss = timeit(cam_scale_shift_only)

print(f"S={S} D={D} FFN={FFN}")
print(f"block_proxy_ms={t_block:.3f} cam_full_ms={t_cam:.3f} cam_scale_shift_ms={t_ss:.3f}")
print(f"cam share of block (proxy) = {100 * t_cam / (t_cam + t_block):.1f}%")
