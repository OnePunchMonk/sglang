"""Runs inside one tree (PYTHONPATH = base or head). Modes: import | micro | e2e <name> | psnr."""

import json
import os
import statistics
import sys
import time


def _import():
    t = []
    import subprocess

    for _ in range(5):
        code = (
            "import time,torch;t=time.perf_counter();"
            "import torch.nn.attention.flex_attention;t1=time.perf_counter();"
            "import sglang.multimodal_gen.runtime.models.dits.wan_animate_2_block;"
            "print(t1-t, time.perf_counter()-t1)"
        )
        so = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True).stdout
        t.append([float(x) for x in so.split()])
        code2 = (
            "import time,torch;from torch.nn.attention.flex_attention import flex_attention;"
            "t=time.perf_counter();torch.compile(flex_attention,dynamic=False,mode='max-autotune',"
            "fullgraph=True,backend='inductor');print(time.perf_counter()-t)"
        )
        t[-1].append(float(subprocess.run([sys.executable, "-c", code2], capture_output=True, text=True).stdout))
    print(json.dumps({"flex_import_s": [r[0] for r in t], "block_module_import_s": [r[1] for r in t],
                      "torch_compile_wrapper_call_s": [r[2] for r in t]}))


def _micro():
    from types import SimpleNamespace

    import torch
    from torch.nn.attention.flex_attention import create_block_mask

    from sglang.multimodal_gen.runtime.models.dits import wan_animate_2_block as blk

    head = hasattr(blk, "_pad_reference_video_kv")
    fp8 = os.environ.get("FP8") == "1"
    dev, dt, H, D = "cuda", torch.bfloat16, 40, 128
    out = {"tree": "head" if head else "base", "fp8": fp8}
    # CI: 576x1024, clip 37 -> 10 latent frames + reference-image frame; 720p clip 77 -> 20 + 1.
    for tag, (f, h, w) in (("ci_576x1024_37f", (11, 36, 64)), ("720p_77f", (21, 45, 80))):
        layout = blk._InContextLayout((f, h, w), (f - 1, h, w), (f, h, w))
        mask = create_block_mask(layout.block_mask_fn(), B=None, H=None, Q_LEN=layout.padded_generation_video_len,
                                 KV_LEN=layout.padded_kv_len, device=dev, _compile=True)
        g = torch.Generator(device=dev).manual_seed(0)
        n = layout.generation_video_num_tokens
        q, k, v = (torch.randn(1, n, H, D, device=dev, dtype=dt, generator=g) for _ in range(3))
        rk, rv = (torch.randn(1, layout.reference_video_num_tokens, H, D, device=dev, dtype=dt, generator=g) for _ in range(2))
        args = [q, k, v, rk, rv, mask, layout]
        cache_bytes = rk.numel() * rk.element_size() * 2
        if head:
            pk, pv = blk._pad_reference_video_kv(rk, layout), blk._pad_reference_video_kv(rv, layout)
            args = [q, k, v, pk, pv, mask, layout]
            cache_bytes = pk.numel() * pk.element_size() * 2
            if fp8:
                (k8, ks), (v8, vs) = blk._quantize_fp8_per_token_head(pk), blk._quantize_fp8_per_token_head(pv)
                args = [q, k, v, k8, v8, mask, layout, ks, vs]
                cache_bytes = sum(t.numel() * t.element_size() for t in (k8, ks, v8, vs))
        stub = SimpleNamespace(log_scale=0.0)
        call = lambda: blk.WanAnimate2TransformerBlock._in_context_attention(stub, *args)

        def timeit(fn, iters=20):
            for _ in range(3):
                fn()
            torch.cuda.synchronize()
            ts = []
            for _ in range(iters):
                a, b = torch.cuda.Event(True), torch.cuda.Event(True)
                a.record(); fn(); b.record(); torch.cuda.synchronize()
                ts.append(a.elapsed_time(b))
            return statistics.median(ts), statistics.pstdev(ts)

        full = timeit(call)
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); before = torch.cuda.memory_allocated()
        o = call(); torch.cuda.synchronize()
        transient = torch.cuda.max_memory_allocated() - before
        # Packing alone: the same call with the attention kernel replaced by a no-op.
        real = blk._flex_attention_compiled
        blk._flex_attention_compiled = lambda q, k, v, **kw: q
        pack = timeit(call)
        blk._flex_attention_compiled = real
        # Accuracy against an fp32 dense reference of the same padded inputs.
        torch.save(o.float().cpu(), f"/tmp/micro_out_{tag}_{out['tree']}_{int(fp8)}.pt")
        out[tag] = {"call_ms_median": full[0], "call_ms_std": full[1], "pack_ms_median": pack[0],
                    "pack_ms_std": pack[1], "transient_MB": transient / 2**20,
                    "ref_cache_MB_per_block": cache_bytes / 2**20,
                    "ref_cache_GB_40_blocks": cache_bytes * 40 / 2**30}
        if head and fp8:
            ref = torch.load(f"/tmp/micro_out_{tag}_head_0.pt")
            d = o.float().cpu() - ref
            out[tag]["fp8_vs_bf16_rel_l2"] = (d.norm() / ref.norm()).item()
            out[tag]["fp8_vs_bf16_max_abs"] = d.abs().max().item()
            out[tag]["finite"] = bool(torch.isfinite(o).all())
        del args, q, k, v, rk, rv, o, mask
        torch.cuda.empty_cache()
    print(json.dumps(out))


def _e2e(name):
    import urllib.request

    import numpy as np
    import torch

    from sglang.multimodal_gen import DiffGenerator

    ci = os.environ["CI"]
    for f in ("reference_short.jpg", "driving_video_short.mp4"):
        if not os.path.exists(f"/tmp/{f}"):
            urllib.request.urlretrieve(f"{ci}/{f}", f"/tmp/{f}")
    t0 = time.perf_counter()
    gen = DiffGenerator.from_pretrained(model_path=os.environ["MODEL"], num_gpus=1, dit_layerwise_offload=True,
                                        text_encoder_cpu_offload=True, layerwise_offload_components=["transformer"])
    load_s = time.perf_counter() - t0
    params = dict(
        prompt=("Character appearance description: a smiling young woman with shoulder-length dark hair, a "
                "mustard-yellow knit sweater, blue jeans and white sneakers. Background description: a plain "
                "light-gray studio backdrop."),
        image_path="/tmp/reference_short.jpg", video_path="/tmp/driving_video_short.mp4",
        width=576, height=1024, fps=16, clip_len=37, num_inference_steps=8, seed=42,
        save_output=False, return_frames=True,
    )
    runs = []
    # run 0 = warmup (flex compile), run 1 also writes the mp4 (excluded from timing), 2-4 timed.
    for i in range(5):
        p = dict(params, save_output=True, output_path="/vol/videos", output_file_name=f"{name}.mp4") if i == 1 else params
        t = time.perf_counter()
        r = gen.generate(sampling_params_kwargs=p)
        r = r[0] if isinstance(r, list) else r
        wall = time.perf_counter() - t
        runs.append({"wall_s": wall, "gen_s": r.generation_time, "peak_MB": r.peak_memory_mb,
                     "metrics": {k: v for k, v in (r.metrics or {}).items() if isinstance(v, (int, float, dict))}})
        if i == 1:
            np.save(f"/tmp/frames_{name}.npy", np.asarray(r.frames))
    timed = [x["wall_s"] for x in runs[2:]]
    print(json.dumps({"name": name, "load_s": load_s, "warmup_wall_s": runs[0]["wall_s"],
                      "wall_s_median": statistics.median(timed), "wall_s_all": timed,
                      "peak_MB": max(x["peak_MB"] for x in runs), "runs": runs,
                      "torch_peak_reserved_MB": torch.cuda.max_memory_reserved() / 2**20}, default=str))


def _psnr():
    import numpy as np

    base = np.load("/tmp/frames_base.npy").astype(np.float64)
    out = {}
    for n in ("p10", "p10_fp8"):
        x = np.load(f"/tmp/frames_{n}.npy").astype(np.float64)
        mse = ((x - base) ** 2).mean()
        per = [10 * np.log10(255**2 / max(((a - b) ** 2).mean(), 1e-12)) for a, b in zip(x, base)]
        out[n] = {"shape": list(x.shape), "bitwise_equal": bool((x == base).all()),
                  "psnr_db": 10 * np.log10(255**2 / mse) if mse else float("inf"),
                  "psnr_min_frame_db": min(per), "max_abs": float(np.abs(x - base).max())}
    print(json.dumps(out))


if __name__ == "__main__":
    mode = sys.argv[1]
    {"import": _import, "micro": _micro, "psnr": _psnr}.get(mode, lambda: _e2e(sys.argv[2]))()
