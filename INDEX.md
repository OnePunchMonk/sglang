# Wan-Animate-2 ref K/V + FA3 split: every run and output

Draft PR: OnePunchMonk/sglang#21. All GPU runs were on 1x H100 80GB HBM3 (driver 610.57.04) on Modal, image `lmsysorg/sglang:dev`, with the base or head tree overlaid via PYTHONPATH. Checkpoint `Wan-AI/Wan2.2-Animate-2-14B-Diffusers` @ `7d48412d`.

Trees: main = upstream `41fb719cf2`; head = main + PR commits. Early CPU tests used base `e7c55f6959`, which has no Wan-Animate-2 changes relative to `41fb719cf2`.

Every run is listed below, including stopped, failed and discarded ones.

| # | Run | Status | Outputs here |
|---|---|---|---|
| 1 | Weight prefetch to Volume (CPU) | done | `raw/prefetch_out.json`, `raw/logs/prefetch.log` |
| 2 | Unit tests, Modal CPU: base 29 passed; head 30 passed; head tests on base fail at import (expected) | done | `raw/unit_out.json`, `raw/logs/unit.log` |
| 3 | CI-shape micro + e2e benchmark (`benchmark/bench.py`, `worker.py`), 3 configs | **stopped by me twice before producing results**, superseded by the two-run plan | `raw/logs/bench.log` (startup only). No numbers. |
| 4 | Full run: main vs head with FP8 (commit `b9848ef` only), 40 steps | done | `full_out.json`, `psnr_out.json`, `main_41fb719.mp4`, `head_b9848ef_fp8.mp4`, `side_by_side.gif`, `raw/logs/full.log` |
| 5 | Profile, 12 steps (2 profiled): main offload, head FP8 offload, head FP8 no offload, main no offload (OOM) | done | `opt_out.json`, `raw/profile_traces/*.trace.json.gz` (Chrome traces; `main_no_offload` is the stub left by the OOM), `raw/logs/opt.log` |
| 6 | Kernel check: flex vs exact FA3 split, single call, random inputs | done (first attempt failed on a harness unpack bug, rerun) | `fa3_out.json`, `benchmark/fa3_check.py` |
| 7 | FA3 e2e attempt 1: H100 unit tests segfaulted (11 passed, then -11) | failed, no e2e | not saved; recorded below |
| 8 | Debug: per-op probe on H100 | done | not saved; recorded below |
| 9 | H100 tests after fix: 38 passed | done | not saved; recorded below |
| 10 | FA3 e2e attempt 2: main vs head, split gated to full-length clips only | **discarded**: the demo clip is short, so head fell back to flex | `raw/discarded_fa3_fallback_run/{main,head}.mp4`, numbers below |
| 11 | H100 tests after short-clip fix: 43 passed | done | in run 12's JSON (`unit_tests_h100`) |
| 12 | **Final**: head (FA3 split + FP8, `44db7c9`) first on a cold cache, then main, 40 steps | done | `fa3/fa3_full_out.json`, `fa3/psnr_fa3_out.json`, `fa3/main_41fb719.mp4`, `fa3/head_fa3_fp8.mp4`, `fa3/side_by_side.gif`, `raw/logs/fa3_full.log` |

| 13 | Follow-up: distilled checkpoint, head (FP8 off) then main, 10 steps, guidance 1.0; base head FA3 only (FP8 off) | done; exposed the +6.3 GB cache-padding regression | `followup/followup_out.json`, `raw/logs/followup.log` |
| 14 | Fix `3412e3b616` (cache sized to the clip): H100 tests 43 passed (after one test-assertion fix), FA3-only rerun | done | `followup/followup_fix_out.json`, `raw/logs/followup_fix.log` |
| 15 | FP8 rerun with the fix | done | `followup/followup_fp8_out.json`, `raw/logs/followup_fp8.log` |
| 16 | PSNR (CPU): FA3-only (FP8 off) vs main, base checkpoint; distilled head vs distilled main. main mp4 is bitwise identical across two runs (md5 `6335ac1f`) | done | `followup/psnr_fa3_only_out.json`, `followup/psnr_distilled_out.json` |
| 17 | Multi-GPU, 2x H100: CFG-parallel and Ulysses SP=2, head (FA3, FP8 off) then main per mode, 40 steps | done | `multigpu/multigpu_out.json`, `multigpu/*.mp4`, `raw/logs/multigpu.log` |
| 18 | Multi-clip, 1x H100: demo video looped 3x (318 frames at 30 fps, 169 output frames, 3 clips), head then main, 40 steps, FP8 off | done | `multiclip/multiclip_out.json`, `multiclip/{main,head}.mp4`, `raw/logs/multiclip.log` |

## Runs whose output was not saved to a file (recorded from the session)

- **Run 7.** `pytest test_wan_animate_2_parallelism.py` on H100 exited -11 after 11 passing tests. The first failing test was `test_split_attention_matches_flex[False-0.0]`.
- **Run 8.** On H100:
  - `sglang.kernels.ops.attention.flash_attention_v3.flash_attn_varlen_func(..., return_softmax_lse=True)` works.
  - `sglang.multimodal_gen.runtime.layers.attention.backends.flash_attn.flash_attn_varlen_func_op_lse(q, k, v, cu, cu, 384, 384, ver=3)` segfaults (-11). The likely cause is the op's `window_size=None` default; the dispatcher defaults to `(-1, -1)`.
  - `sglang.srt.layers.attention.merge_state.merge_state` works and returns `(out [N,H,D], lse [N,H])`.
  - Fix: call `sglang.kernels.ops.attention.flash_attention.flash_attn_varlen_func(..., ver=3)`.
- **Run 9.** 38 passed in 21.6 s, H100, full test file.
- **Run 10.** The FA3 path fell back to flex, so this is effectively main vs FP8-only again:

  | | main | head |
  |---|---|---|
  | load (s) | 124.8 | 74.2 |
  | request wall (s) | 467.6 | 429.4 |
  | s/it at step 40 (running avg) | 10.60 | 10.03 |
  | peak reserved / allocated (MB) | 58,458 / 32,198 | 40,286 / 25,109 |
  | flex AUTOTUNE lines | 2 | 0 (head ran second, warm cache, so this is not evidence) |

  Cause: the 56 output frames pad to one 57-frame clip, shorter than the 77-frame full-clip layout, which the gate excluded at the time.

## Not run

- **P13 import timing.** Its only harness was run 3, which was stopped. Unmeasured; P13 was dropped on code inspection (`torch.compile` at import only builds a lazy wrapper).
- **CI-shape repeated A/B with variance.** Run 3 was stopped. All e2e numbers are single runs.
