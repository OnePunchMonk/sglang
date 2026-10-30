"""Multi-clip check on 1x H100: demo driving video looped 3x (~10.6 s -> ~170 frames at 16 fps,
2 full 77-frame clips + a short last clip). main vs head (FA3 split, FP8 off), 40 steps, seed 42.

  modal run multiclip.py
"""

import json

import modal

T = "<path to trees>"
BASE = "Wan-AI/Wan2.2-Animate-2-14B-Diffusers"

app = modal.App("wan-animate2-multiclip")
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
name = sys.argv[1]
t0 = time.perf_counter()
gen = DiffGenerator.from_pretrained(model_path=os.environ["MODEL"], num_gpus=1, dit_layerwise_offload=True,
                                    text_encoder_cpu_offload=True, layerwise_offload_components=["transformer"])
load_s = time.perf_counter() - t0
t = time.perf_counter()
r = gen.generate(sampling_params_kwargs=dict(
    prompt="视频中的人在做动作", image_path="/root/inputs/image.jpeg", video_path="/tmp/video_x3.mp4",
    width=640, height=800, num_inference_steps=40, guidance_scale=3.0, clip_len=77, seed=42,
    save_output=True, output_path="/vol/videos_multiclip", output_file_name=f"{name}.mp4"))
r = r[0] if isinstance(r, list) else r
snap = (r.metrics or {}).get("memory_snapshots", {}).get("after_forward", {})
print(json.dumps({"name": name, "load_s": load_s, "wall_s": time.perf_counter() - t,
                  "peak_reserved_MB": snap.get("peak_reserved_mb"), "peak_allocated_MB": snap.get("peak_allocated_mb")}))
'''

COMPARE = r'''
import json, av, numpy as np
def frames(p):
    return np.stack([f.to_ndarray(format="rgb24") for f in av.open(p).decode(video=0)]).astype(np.float64)
a, b = frames("/vol/videos_multiclip/main.mp4"), frames("/vol/videos_multiclip/head.mp4")
n = min(len(a), len(b))
per = [10 * np.log10(255**2 / max(((x - y) ** 2).mean(), 1e-12)) for x, y in zip(a[:n], b[:n])]
# 76 new frames per clip after the first (one conditioning frame carried over)
clips = [per[i:i + 76] for i in range(0, n, 76)]
print(json.dumps({"frames_main": len(a), "frames_head": len(b), "psnr_mean_db": float(np.mean(per)),
                  "psnr_min_db": float(min(per)), "per_clip_mean_db": [float(np.mean(c)) for c in clips],
                  "temporal_mad_main": float(np.abs(np.diff(a, axis=0)).mean()),
                  "temporal_mad_head": float(np.abs(np.diff(b, axis=0)).mean())}))
'''


@app.function(image=image, gpu="H100", cpu=8, memory=131072, timeout=4500, volumes={"/vol": vol})
def multiclip():
    import os
    import re
    import subprocess
    import sys

    res = {}
    p = subprocess.run(["ffmpeg", "-y", "-stream_loop", "2", "-i", "/root/inputs/video.mp4", "-c", "copy",
                        "/tmp/video_x3.mp4"], capture_output=True, text=True)
    q = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                        "stream=nb_frames,duration,r_frame_rate", "-of", "json", "/tmp/video_x3.mp4"],
                       capture_output=True, text=True)
    res["driving_video"] = q.stdout or p.stderr[-1500:]
    for name, root in (("head", "/root/head"), ("main", "/root/main")):
        env = dict(os.environ, PYTHONPATH=root, HF_HOME="/vol/hf", MODEL=BASE,
                   SGLANG_DIFFUSION_WAN_ANIMATE_2_FP8_REFERENCE_KV="false")
        p = subprocess.run([sys.executable, "-c", RUN, name], env=env, cwd=root,
                           capture_output=True, text=True, timeout=3000)
        out = {"rc": p.returncode, "flex_autotune_lines": p.stderr.count("AUTOTUNE"),
               "per_clip_s_per_it": re.findall(r"40/40 \[[^,]*,\s*([\d.]+)s/it\]", p.stderr)}
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
    out = multiclip.remote()
    print(json.dumps({k: ({kk: vv for kk, vv in v.items() if kk != "log_tail"} if isinstance(v, dict) else v)
                      for k, v in out.items()}, indent=1))
    with open(__file__.replace("multiclip.py", "multiclip_out.json"), "w") as f:
        json.dump(out, f, indent=1)
