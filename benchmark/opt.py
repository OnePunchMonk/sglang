"""Step 1 + 2 of the time plan: profile main vs head, and head FP8 without layerwise offload.

  modal run opt.py::gifs    CPU: side-by-side GIF of the two full-run videos -> volume
  modal run opt.py          1x H100: 4 short runs (12 steps, 2 profiled), GPU-time breakdown

Same inputs / seed / size as full.py; fewer steps since only per-step time is compared.
"""

import json

import modal

T = "<path to trees: main/ and headfull/ python/sglang + inputs/>"
MODEL_ID = "Wan-AI/Wan2.2-Animate-2-14B-Diffusers"

app = modal.App("wan-animate2-opt")
vol = modal.Volume.from_name("wan-animate2-refkv")
image = (
    modal.Image.from_registry("lmsysorg/sglang:dev")
    .add_local_dir(f"{T}/main/python/sglang", "/root/main/sglang", copy=True, ignore=["**/__pycache__", "**/.claude"])
    .add_local_dir(f"{T}/headfull/python/sglang", "/root/head/sglang", copy=True, ignore=["**/__pycache__", "**/.claude"])
    .add_local_dir(f"{T}/inputs", "/root/inputs", copy=True)
)

RUN = r'''
import json, os, re, sys, time, torch
from sglang.multimodal_gen import DiffGenerator
name, offload = sys.argv[1], sys.argv[2] == "1"
t0 = time.perf_counter()
kw = dict(model_path=os.environ["MODEL"], num_gpus=1, text_encoder_cpu_offload=True, dit_layerwise_offload=offload)
if offload:
    kw["layerwise_offload_components"] = ["transformer"]
gen = DiffGenerator.from_pretrained(**kw)
load_s = time.perf_counter() - t0
t = time.perf_counter()
r = gen.generate(sampling_params_kwargs=dict(
    prompt="视频中的人在做动作", image_path="/root/inputs/image.jpeg", video_path="/root/inputs/video.mp4",
    width=640, height=800, num_inference_steps=12, clip_len=77, seed=42, save_output=False,
    profile=True, num_profiled_timesteps=2))
r = r[0] if isinstance(r, list) else r
snap = (r.metrics or {}).get("memory_snapshots", {}).get("after_forward", {})
print(json.dumps({"name": name, "offload": offload, "load_s": load_s, "wall_s": time.perf_counter() - t,
                  "peak_reserved_MB": snap.get("peak_reserved_mb"), "peak_allocated_MB": snap.get("peak_allocated_mb")}))
'''


def _steady_s_per_it(log: str):
    import re

    its = re.findall(r"(\d+)/12 \[[^,]*,\s*([\d.]+)s/it\]", log)
    # per-step deltas are not printed; the running average after the last step is the closest number.
    return float(its[-1][1]) if its else None


def _breakdown(trace_dir: str):
    import glob
    import gzip

    files = sorted(glob.glob(f"{trace_dir}/**/*.json*", recursive=True))
    if not files:
        return {"error": f"no trace in {trace_dir}"}
    f = files[-1]
    data = json.load(gzip.open(f) if f.endswith(".gz") else open(f))
    ev = data["traceEvents"] if isinstance(data, dict) else data
    cats = {}
    lo, hi, busy = None, None, []
    for e in ev:
        c = e.get("cat", "")
        if c not in ("kernel", "gpu_memcpy", "gpu_memset") or "dur" not in e:
            continue
        n = e.get("name", "")
        ln = n.lower()
        if c == "gpu_memcpy":
            k = "memcpy_" + ("HtoD" if "htod" in ln else "DtoH" if "dtoh" in ln else "DtoD")
        elif c == "gpu_memset":
            k = "memset"
        elif "flex" in ln or "triton_tem" in ln or "flash" in ln or "fmha" in ln or "attention" in ln:
            k = "attention"
        elif any(s in ln for s in ("gemm", "nvjet", "sm90_", "cutlass", "cublas", "xmma", "splitk")):
            k = "gemm"
        elif any(s in ln for s in ("copy", "fill", "cat", "zero", "elementwise", "unrolled")):
            k = "copy_fill_elementwise"
        elif "triton" in ln:
            k = "triton_other"
        else:
            k = "other"
        cats[k] = cats.get(k, 0.0) + e["dur"]
        lo = e["ts"] if lo is None else min(lo, e["ts"])
        hi = e["ts"] + e["dur"] if hi is None else max(hi, e["ts"] + e["dur"])
        busy.append((e["ts"], e["ts"] + e["dur"]))
    busy.sort()
    covered, cur_s, cur_e = 0.0, None, None
    for s, t in busy:
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                covered += cur_e - cur_s
            cur_s, cur_e = s, t
        else:
            cur_e = max(cur_e, t)
    if cur_e is not None:
        covered += cur_e - cur_s
    span = (hi - lo) if lo is not None else 0
    return {"trace": f.split("/")[-1], "span_ms": span / 1e3, "gpu_busy_ms": covered / 1e3,
            "gpu_idle_pct": 100 * (1 - covered / span) if span else None,
            "kernel_ms_by_category": {k: round(v / 1e3, 1) for k, v in sorted(cats.items(), key=lambda x: -x[1])}}


@app.function(image=image, gpu="H100", cpu=8, memory=131072, timeout=3600, volumes={"/vol": vol})
def opt():
    import os
    import subprocess
    import sys

    res = {}
    for name, root, fp8, offload in (
        ("main_offload", "/root/main", "false", "1"),
        ("head_fp8_offload", "/root/head", "true", "1"),
        ("head_fp8_no_offload", "/root/head", "true", "0"),
        ("main_no_offload", "/root/main", "false", "0"),
    ):
        pdir = f"/vol/prof/{name}"
        env = dict(os.environ, PYTHONPATH=root, HF_HOME="/vol/hf", MODEL=MODEL_ID,
                   SGLANG_DIFFUSION_WAN_ANIMATE_2_FP8_REFERENCE_KV=fp8, SGLANG_TORCH_PROFILER_DIR=pdir)
        p = subprocess.run([sys.executable, "-c", RUN, name, offload], env=env, cwd=root,
                           capture_output=True, text=True, timeout=1500)
        out = {"rc": p.returncode, "s_per_it_running_avg": _steady_s_per_it(p.stderr)}
        if p.returncode == 0:
            out.update(json.loads(p.stdout.strip().splitlines()[-1]))
            out["profile"] = _breakdown(pdir)
        else:
            tail = p.stderr[-3000:]
            out["oom"] = "out of memory" in p.stderr.lower()
            out["error_tail"] = tail
        res[name] = out
        vol.commit()
    return res


@app.function(image=image, cpu=4, memory=8192, timeout=900, volumes={"/vol": vol})
def gifs():
    import subprocess

    os_dir = "/vol/videos_full"
    vf = ("[0:v]drawtext=text='main 41fb719':x=10:y=10:fontsize=24:fontcolor=white:box=1:boxcolor=black@0.6[a];"
          "[1:v]drawtext=text='this PR (FP8 ref KV)':x=10:y=10:fontsize=24:fontcolor=white:box=1:boxcolor=black@0.6[b];"
          "[a][b]hstack,fps=8,scale=960:-1:flags=lanczos,split[x][y];[y]palettegen=max_colors=128[p];[x][p]paletteuse")
    p = subprocess.run(["ffmpeg", "-y", "-i", f"{os_dir}/main.mp4", "-i", f"{os_dir}/head.mp4",
                        "-filter_complex", vf, f"{os_dir}/side_by_side.gif"], capture_output=True, text=True)
    vol.commit()
    return p.returncode, p.stderr[-1500:]


@app.local_entrypoint()
def main():
    out = opt.remote()
    print(json.dumps(out, indent=1, default=str))
    with open(__file__.replace("opt.py", "opt_out.json"), "w") as f:
        json.dump(out, f, indent=1, default=str)
