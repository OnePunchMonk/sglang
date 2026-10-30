"""Multi-GPU check of the FA3 split on 2x H100: CFG-parallel (the CI case) and Ulysses SP=2.

  modal run multigpu.py

Base checkpoint, demo inputs, 40 steps, guidance 3.0, seed 42, FP8 off, layerwise offload as in CI.
Each output is compared with the 1-GPU outputs already on the volume:
  main 1-GPU = /vol/videos_fa3b/main.mp4, head 1-GPU (FA3, FP8 off) = /vol/videos_followup/base_head_fa3_only.mp4
CI notes CFG-parallel is bit-identical to 1-GPU, so cfg runs should match their 1-GPU mp4 byte for byte.
"""

import json

import modal

T = "<path to trees>"
BASE = "Wan-AI/Wan2.2-Animate-2-14B-Diffusers"

app = modal.App("wan-animate2-multigpu")
vol = modal.Volume.from_name("wan-animate2-refkv")
image = (
    modal.Image.from_registry("lmsysorg/sglang:dev")
    .add_local_dir(f"{T}/main/python/sglang", "/root/main/sglang", copy=True, ignore=["**/__pycache__", "**/.claude"])
    .add_local_dir(f"{T}/headfull/python/sglang", "/root/head/sglang", copy=True, ignore=["**/__pycache__", "**/.claude"])
    .add_local_dir(f"{T}/inputs", "/root/inputs", copy=True)
)

RUN = r'''
import json, os, sys, time
from sglang.multimodal_gen import DiffGenerator
name, mode = sys.argv[1], sys.argv[2]
kw = dict(model_path=os.environ["MODEL"], num_gpus=2, dit_layerwise_offload=True, text_encoder_cpu_offload=True,
          layerwise_offload_components=["transformer"], encoder_parallel="replicate")
if mode == "cfg":
    kw["enable_cfg_parallel"] = True
else:
    kw.update(sp_degree=2, ulysses_degree=2)
t0 = time.perf_counter()
gen = DiffGenerator.from_pretrained(**kw)
load_s = time.perf_counter() - t0
t = time.perf_counter()
r = gen.generate(sampling_params_kwargs=dict(
    prompt="视频中的人在做动作", image_path="/root/inputs/image.jpeg", video_path="/root/inputs/video.mp4",
    width=640, height=800, num_inference_steps=40, guidance_scale=3.0, clip_len=77, seed=42,
    save_output=True, output_path="/vol/videos_multigpu", output_file_name=f"{name}.mp4"))
r = r[0] if isinstance(r, list) else r
snap = (r.metrics or {}).get("memory_snapshots", {}).get("after_forward", {})
print(json.dumps({"name": name, "load_s": load_s, "wall_s": time.perf_counter() - t,
                  "peak_reserved_MB": snap.get("peak_reserved_mb"), "peak_allocated_MB": snap.get("peak_allocated_mb")}))
'''

COMPARE = r'''
import hashlib, json, av, numpy as np
def frames(p):
    return np.stack([f.to_ndarray(format="rgb24") for f in av.open(p).decode(video=0)]).astype(np.float64)
def md5(p):
    return hashlib.md5(open(p, "rb").read()).hexdigest()
def psnr(a, b):
    n = min(len(a), len(b))
    per = [10 * np.log10(255**2 / max(((x - y) ** 2).mean(), 1e-12)) for x, y in zip(a[:n], b[:n])]
    return {"mean_db": float(np.mean(per)), "min_db": float(min(per)), "bitwise_decoded": bool((a[:n] == b[:n]).all())}
ref = {"main": "/vol/videos_fa3b/main.mp4", "head": "/vol/videos_followup/base_head_fa3_only.mp4"}
out = {}
for name in ("main_cfg", "head_cfg", "main_sp2", "head_sp2"):
    p = f"/vol/videos_multigpu/{name}.mp4"
    try:
        x = frames(p)
    except Exception as e:
        out[name] = {"error": str(e)}
        continue
    tree = name.split("_")[0]
    out[name] = {"md5": md5(p), "md5_same_as_1gpu_" + tree: md5(p) == md5(ref[tree]),
                 "vs_1gpu_same_tree": psnr(x, frames(ref[tree])), "vs_1gpu_main": psnr(x, frames(ref["main"]))}
for mode in ("cfg", "sp2"):
    a, b = f"/vol/videos_multigpu/main_{mode}.mp4", f"/vol/videos_multigpu/head_{mode}.mp4"
    try:
        out[f"head_vs_main_{mode}"] = psnr(frames(b), frames(a))
    except Exception as e:
        out[f"head_vs_main_{mode}"] = {"error": str(e)}
print(json.dumps(out))
'''


@app.function(image=image, gpu="H100:2", cpu=16, memory=196608, timeout=4500, volumes={"/vol": vol})
def multigpu():
    import os
    import re
    import subprocess
    import sys

    res = {}
    # head first in each mode: on a cold cache, a flex fallback in head would print AUTOTUNE lines.
    for name, root, mode in (("head_cfg", "/root/head", "cfg"), ("main_cfg", "/root/main", "cfg"),
                             ("head_sp2", "/root/head", "sp2"), ("main_sp2", "/root/main", "sp2")):
        env = dict(os.environ, PYTHONPATH=root, HF_HOME="/vol/hf", MODEL=BASE,
                   SGLANG_DIFFUSION_WAN_ANIMATE_2_FP8_REFERENCE_KV="false")
        p = subprocess.run([sys.executable, "-c", RUN, name, mode], env=env, cwd=root,
                           capture_output=True, text=True, timeout=1800)
        out = {"rc": p.returncode, "flex_autotune_lines": p.stderr.count("AUTOTUNE"),
               "s_per_it": re.findall(r"40/40 \[[^,]*,\s*([\d.]+)s/it\]", p.stderr)[-2:]}
        if p.returncode == 0:
            out.update(json.loads(p.stdout.strip().splitlines()[-1]))
        else:
            out["error_tail"] = p.stderr[-3000:]
        out["log_tail"] = p.stderr[-1500:]
        res[name] = out
        vol.commit()
    p = subprocess.run([sys.executable, "-c", COMPARE], capture_output=True, text=True)
    res["compare"] = p.stdout.strip() or p.stderr[-2000:]
    return res


@app.local_entrypoint()
def main():
    out = multigpu.remote()
    print(json.dumps({k: ({kk: vv for kk, vv in v.items() if kk != "log_tail"} if isinstance(v, dict) else v)
                      for k, v in out.items()}, indent=1))
    with open(__file__.replace("multigpu.py", "multigpu_out.json"), "w") as f:
        json.dump(out, f, indent=1)
