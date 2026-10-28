# P4: request-static camera conditioning (LingBot-World bidirectional)

Tried and measured, not added to sgl-project/sglang.

## Claim

The bidirectional LingBot-World block recomputes `compute_scale_shift(c2ws_plucker_emb)` on every call. The claim was that this is about 14% of FLOPs per step at 480p, and that computing it once per request (or sharing it across CFG branches) would recover that.

## What was measured

Single H100, 14B shape (dim 5120, FFN 13824, 40 heads), S = 21 x 30 x 52 = 32760 tokens (480p, 81 frames). Median of 10 runs after warmup.

| Part | Time per block |
|---|---|
| Block proxy (self-attn QKV/out + FFN) | 85.1 ms |
| Camera path (scale/shift + modulate) | 9.3 ms |
| Camera scale/shift only | 8.5 ms |

The camera path is about 10% of the block proxy. The proxy leaves out cross-attention and norms, so the real share is likely a little lower.

## Why it was not added

- **The saving is small.** Scale/shift is per block, since every block has its own camera weights, so nothing is shared across blocks. The only sharing is across CFG branches. If the pipeline runs two forwards per step, reusing the scale/shift saves about 4.5% of a CFG step. Without CFG there is no saving. This assumes two forwards per step, which was not verified.
- **The memory cost is large.** Holding scale/shift for all 40 blocks for the whole request is 2 x S x d per block, about 27 GB at this shape. That is too much on top of a 14B model.
- **No pipeline serves this path.** Only the two causal DMD LingBot configs are registered on `origin/main`.

## Caveats

- The fused JIT modulate kernel did not build in the test image (it needs nvcc). The camera path here uses plain torch for that step, which is a small share of the total.
- The GPU run was on Modal with the dependencies installed by hand. Numbers are for one run on one H100.

Benchmark: `scripts/experiments/p4_camera_bench.py`
