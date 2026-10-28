"""Follow-ups 1 + 2 for sgl-project/sglang#47121.

  modal run followup.py::prefetch   CPU: distilled weights -> Volume
  modal run followup.py             1x H100 (60 min cap):
     a) distilled, this PR (FP8 off) first on a cold cache, then distilled main: 10 steps, guidance 1.0
     b) base checkpoint, this PR with FP8 off (FA3 only), 40 steps; compare with the main run in fa3/fa3_full_out.json

Same demo inputs, size and seed as full.py. Trees: main = 41fb719cf2, head = main + PR diff.
"""

import json

import modal

T = "<path to trees>"
BASE = "Wan-AI/Wan2.2-Animate-2-14B-Diffusers"
DISTILLED = "Wan-AI/Wan2.2-Animate-2-14B-Distilled-Diffusers"

app = modal.App("wan-animate2-followup")
vol = modal.Volume.from_name("wan-animate2-refkv")
image = (
    modal.Image.from_registry("lmsysorg/sglang:dev")
    .add_local_dir(f"{T}/main/python/sglang", "/root/main/sglang", copy=True, ignore=["**/__pycache__", "**/.claude"])
    .add_local_dir(f"{T}/headfull/python/sglang", "/root/head/sglang", copy=True, ignore=["**/__pycache__", "**/.claude"])
    .add_local_dir(f"{T}/inputs", "/root/inputs", copy=True)
)

RUN = r'''
import json, os, sys, time, torch
from sglang.multimodal_gen import DiffGenerator
name, steps, guidance = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
t0 = time.perf_counter()
gen = DiffGenerator.from_pretrained(model_path=os.environ["MODEL"], num_gpus=1, dit_layerwise_offload=True,
                                    text_encoder_cpu_offload=True, layerwise_offload_components=["transformer"])
load_s = time.perf_counter() - t0
t = time.perf_counter()
r = gen.generate(sampling_params_kwargs=dict(
    prompt="视频中的人在做动作", image_path="/root/inputs/image.jpeg", video_path="/root/inputs/video.mp4",
    width=640, height=800, num_inference_steps=steps, guidance_scale=guidance, clip_len=77, seed=42,
    save_output=True, output_path="/vol/videos_followup", output_file_name=f"{name}.mp4"))
r = r[0] if isinstance(r, list) else r
snap = (r.metrics or {}).get("memory_snapshots", {}).get("after_forward", {})
print(json.dumps({"name": name, "load_s": load_s, "wall_s": time.perf_counter() - t,
                  "peak_reserved_MB": snap.get("peak_reserved_mb"), "peak_allocated_MB": snap.get("peak_allocated_mb")}))
'''


@app.function(image=image, cpu=4, memory=8192, timeout=3600, volumes={"/vol": vol})
def prefetch():
    import os

    os.environ["HF_HOME"] = "/vol/hf"
    from huggingface_hub import snapshot_download

    path = snapshot_download(DISTILLED)
    vol.commit()
    return path


@app.function(image=image, gpu="H100", cpu=8, memory=131072, timeout=3600, volumes={"/vol": vol},
              secrets=[modal.Secret.from_dict({"JOBS": "base_head_fa3_fp8"})])
def followup():
    import os
    import re
    import subprocess
    import sys

    res = {}
    p = subprocess.run([sys.executable, "-m", "pytest", "-q", "-W", "ignore", "--tb=short", "-p", "no:cacheprovider",
                        "sglang/multimodal_gen/test/unit/test_wan_animate_2_parallelism.py"],
                       env=dict(os.environ, PYTHONPATH="/root/head"), cwd="/root/head", capture_output=True, text=True)
    res["unit_tests_h100"] = {"rc": p.returncode, "out": p.stdout[-1500:], "log_tail": ""}
    if p.returncode:
        return res
    jobs = os.environ.get("JOBS", "distilled_head,distilled_main,base_head_fa3_only").split(",")
    for name, root, model, steps, guidance in [j for j in (
        ("distilled_head", "/root/head", DISTILLED, 10, 1.0),
        ("distilled_main", "/root/main", DISTILLED, 10, 1.0),
        ("base_head_fa3_only", "/root/head", BASE, 40, 3.0),
        ("base_head_fa3_fp8", "/root/head", BASE, 40, 3.0),
    ) if j[0] in jobs]:
        env = dict(os.environ, PYTHONPATH=root, HF_HOME="/vol/hf", MODEL=model,
                   SGLANG_DIFFUSION_WAN_ANIMATE_2_FP8_REFERENCE_KV="true" if name.endswith("fp8") else "false")
        p = subprocess.run([sys.executable, "-c", RUN, name, str(steps), str(guidance)], env=env, cwd=root,
                           capture_output=True, text=True, timeout=1500)
        out = {"rc": p.returncode, "flex_autotune_lines": p.stderr.count("AUTOTUNE"),
               "s_per_it": re.findall(rf"{steps}/{steps} \[[^,]*,\s*([\d.]+)s/it\]", p.stderr)[-2:]}
        if p.returncode == 0:
            out.update(json.loads(p.stdout.strip().splitlines()[-1]))
        else:
            out["error_tail"] = p.stderr[-3000:]
        out["log_tail"] = p.stderr[-1500:]
        res[name] = out
        vol.commit()
    return res


@app.local_entrypoint()
def main():
    out = followup.remote()
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "log_tail"} for k, v in out.items()}, indent=1))
    with open(__file__.replace("followup.py", "followup_fp8_out.json"), "w") as f:
        json.dump(out, f, indent=1)
