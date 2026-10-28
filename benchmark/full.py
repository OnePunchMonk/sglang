"""Two full end-to-end runs on 1x H100: current main vs main + our commit (FP8 reference K/V on).

  modal run full.py

Official Wan2.2 demo inputs (examples/wan_animate/animate), model defaults (40 steps,
guidance 3.0, clip_len 77), 640x800 area budget, seed 42, layerwise offload as in CI.
"""

import json

import modal

S = "<path to trees: main/ and headfull/ python/sglang + inputs/>"
MODEL_ID = "Wan-AI/Wan2.2-Animate-2-14B-Diffusers"

app = modal.App("wan-animate2-full")
vol = modal.Volume.from_name("wan-animate2-refkv")
image = (
    modal.Image.from_registry("lmsysorg/sglang:dev")
    .add_local_dir(f"{S}/main/python/sglang", "/root/main/sglang", copy=True, ignore=["**/__pycache__", "**/.claude"])
    .add_local_dir(f"{S}/headfull/python/sglang", "/root/head/sglang", copy=True, ignore=["**/__pycache__", "**/.claude"])
    .add_local_dir(f"{S}/inputs", "/root/inputs", copy=True)
)

RUN = r'''
import json, os, sys, time, torch
from sglang.multimodal_gen import DiffGenerator
name = sys.argv[1]
t0 = time.perf_counter()
gen = DiffGenerator.from_pretrained(model_path=os.environ["MODEL"], num_gpus=1, dit_layerwise_offload=True,
                                    text_encoder_cpu_offload=True, layerwise_offload_components=["transformer"])
load_s = time.perf_counter() - t0
t = time.perf_counter()
r = gen.generate(sampling_params_kwargs=dict(
    prompt="视频中的人在做动作",  # official demo prompt
    image_path="/root/inputs/image.jpeg", video_path="/root/inputs/video.mp4",
    width=640, height=800, num_inference_steps=40, clip_len=77, seed=42,
    save_output=True, output_path="/vol/videos_full", output_file_name=f"{name}.mp4", return_frames=True))
r = r[0] if isinstance(r, list) else r
wall = time.perf_counter() - t
import numpy as np
np.save(f"/tmp/frames_{name}.npy", np.asarray(r.frames))
print(json.dumps({"name": name, "load_s": load_s, "wall_s": wall, "gen_s": r.generation_time,
                  "peak_MB": r.peak_memory_mb, "size": r.size,
                  "torch_max_reserved_MB": torch.cuda.max_memory_reserved() / 2**20,
                  "metrics": {k: v for k, v in (r.metrics or {}).items() if isinstance(v, (int, float, dict))}},
                 default=str))
'''

PSNR = r'''
import json, numpy as np
a = np.load("/tmp/frames_main.npy").astype(np.float64); b = np.load("/tmp/frames_head.npy").astype(np.float64)
per = [10*np.log10(255**2/max(((x-y)**2).mean(), 1e-12)) for x, y in zip(a, b)]
print(json.dumps({"shape": list(a.shape), "psnr_mean_db": float(np.mean(per)), "psnr_min_frame_db": float(min(per)),
                  "max_abs": float(np.abs(a-b).max()), "bitwise_equal": bool((a == b).all())}))
'''


@app.function(image=image, gpu="H100", cpu=8, memory=131072, timeout=7200, volumes={"/vol": vol})
def full():
    import os
    import subprocess
    import sys

    res = {}
    p = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                        "stream=width,height,r_frame_rate,nb_frames,duration", "-of", "json", "/root/inputs/video.mp4"],
                       capture_output=True, text=True)
    res["driving_video"] = p.stdout or p.stderr
    res["gpu"] = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
                                capture_output=True, text=True).stdout.strip()
    for name, root, fp8 in (("main", "/root/main", "false"), ("head", "/root/head", "true")):
        env = dict(os.environ, PYTHONPATH=root, HF_HOME="/vol/hf", MODEL=MODEL_ID,
                   SGLANG_DIFFUSION_WAN_ANIMATE_2_FP8_REFERENCE_KV=fp8)
        p = subprocess.run([sys.executable, "-c", RUN, name], env=env, cwd=root, capture_output=True, text=True)
        res[name] = p.stdout.strip().splitlines()[-1] if p.returncode == 0 else p.stderr[-5000:]
        res[f"{name}_log_tail"] = p.stderr[-2500:]
    p = subprocess.run([sys.executable, "-c", PSNR], capture_output=True, text=True)
    res["quality"] = p.stdout.strip() or p.stderr[-2000:]
    vol.commit()
    return res


@app.local_entrypoint()
def main():
    out = full.remote()
    print(json.dumps(out, indent=1, default=str))
    with open(__file__.replace("full.py", "full_out.json"), "w") as f:
        json.dump(out, f, indent=1, default=str)
