"""End-to-end ComfyUI check for native batching (PR 44105) with Z-Image-Turbo.

Real ComfyUI, the plugin copied into custom_nodes, the shipped
z-image_sgld.json workflow submitted over ComfyUI's /prompt API. Only
SGLDOptions.native_batching changes between runs.
"""

import modal

SHA = "c0584cf9afefdb4a3004a57af8b1aec501d44a4a"
WT = "/path/to/sglang"  # a checkout of PR 44105 (feat/comfyui-native-batching)
PKG = "python/sglang/multimodal_gen"
CHANGED = [
    f"{PKG}/apps/ComfyUI_SGLDiffusion/core/generator.py",
    f"{PKG}/apps/ComfyUI_SGLDiffusion/executors/adapter.py",
    f"{PKG}/apps/ComfyUI_SGLDiffusion/executors/base.py",
    f"{PKG}/apps/ComfyUI_SGLDiffusion/executors/flux.py",
    f"{PKG}/apps/ComfyUI_SGLDiffusion/executors/zimage.py",
    f"{PKG}/apps/ComfyUI_SGLDiffusion/nodes.py",
    f"{PKG}/runtime/pipelines_core/schedule_batch.py",
    f"{PKG}/runtime/pipelines_core/stages/denoising.py",
]
FILES = {
    "diffusion_models": "split_files/diffusion_models/z_image_turbo_bf16.safetensors",
    "text_encoders": "split_files/text_encoders/qwen_3_4b.safetensors",
    "vae": "split_files/vae/ae.safetensors",
}

image = (
    modal.Image.from_registry("lmsysorg/sglang:latest")
    .run_commands(
        "git init -q /sglang-pr",
        "cd /sglang-pr && git remote add origin https://github.com/sgl-project/sglang.git"
        f" && git fetch -q --depth 1 origin {SHA} && git checkout -q FETCH_HEAD",
        "cd /sglang-pr/python && pip install -e '.[diffusion]'",
    )
    .run_commands("pip uninstall -y flashinfer-cubin flashinfer-jit-cache || true")
    .run_commands(
        "git clone -q --depth 1 https://github.com/comfyanonymous/ComfyUI.git /ComfyUI",
        # Keep the sglang image's torch stack; install the rest of ComfyUI's deps.
        "cd /ComfyUI && grep -v -E '^(torch|torchvision|torchaudio)\\b' requirements.txt"
        " > /tmp/req.txt && pip install -r /tmp/req.txt",
        "cd /ComfyUI && git log -1 --format='ComfyUI %h %cd'",
    )
    .env({"HF_HOME": "/hf"})
)
for path in CHANGED:
    image = image.add_local_file(f"{WT}/{path}", f"/sglang-pr/{path}", copy=True)
# The plugin as a user installs it: the directory copied into custom_nodes.
image = image.run_commands(
    "cp -r /sglang-pr/python/sglang/multimodal_gen/apps/ComfyUI_SGLDiffusion"
    " /ComfyUI/custom_nodes/ComfyUI_SGLDiffusion"
)

hf_cache = modal.Volume.from_name("sgld-bugfix-hf-cache")
app = modal.App("sgld-comfyui-e2e", image=image)


@app.function(gpu="L40S", volumes={"/hf": hf_cache}, timeout=60 * 60)
def run() -> str:
    import base64
    import copy
    import io
    import json
    import os
    import subprocess
    import time

    import numpy as np
    import requests
    from huggingface_hub import hf_hub_download
    from PIL import Image

    res = {
        "comfyui": subprocess.run(
            ["git", "-C", "/ComfyUI", "log", "-1", "--format=%h %cd"],
            capture_output=True,
            text=True,
        ).stdout.strip()
    }
    for folder, remote in FILES.items():
        local = hf_hub_download("Comfy-Org/z_image_turbo", remote)
        dest = f"/ComfyUI/models/{folder}/{os.path.basename(remote)}"
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        if not os.path.exists(dest):
            os.symlink(local, dest)
    hf_cache.commit()

    log = open("/tmp/comfyui.log", "w")
    server = subprocess.Popen(
        ["python", "main.py", "--listen", "127.0.0.1", "--port", "8188",
         "--disable-auto-launch"],
        cwd="/ComfyUI", stdout=log, stderr=subprocess.STDOUT,
    )
    base = "http://127.0.0.1:8188"
    try:
        t0 = time.time()
        while True:
            if server.poll() is not None:
                raise RuntimeError(open("/tmp/comfyui.log").read()[-4000:])
            try:
                if requests.get(f"{base}/object_info/SGLDOptions", timeout=2).ok:
                    break
            except requests.RequestException:
                pass
            if time.time() - t0 > 600:
                raise TimeoutError(open("/tmp/comfyui.log").read()[-4000:])
            time.sleep(2)
        info = requests.get(f"{base}/object_info/SGLDOptions").json()
        res["native_batching_input"] = info["SGLDOptions"]["input"]["optional"][
            "native_batching"
        ]

        wf_path = (
            "/ComfyUI/custom_nodes/ComfyUI_SGLDiffusion/workflows/z-image_sgld.json"
        )
        shipped = json.load(open(wf_path))
        by_type = {node["class_type"]: key for key, node in shipped.items()}

        def workflow(mode, prefix, seed=1234):
            wf = copy.deepcopy(shipped)
            latent = wf[by_type["EmptySD3LatentImage"]]["inputs"]
            latent.update(width=1024, height=1024, batch_size=2)
            sampler = wf[by_type["KSampler"]]["inputs"]
            # cfg > 1 so ComfyUI batches cond and uncond: 4 rows per DiT call.
            sampler.update(cfg=2.0, seed=seed, steps=9)
            # The shipped 3-word negative is too short for ComfyUI to concat
            # with the long positive (lcm padding > 4x), so cond and uncond
            # would run as separate calls. A zeroed-out positive has the same
            # shape, so ComfyUI batches cond and uncond together.
            wf["901"] = {
                "class_type": "ConditioningZeroOut",
                "inputs": {"conditioning": sampler["positive"]},
            }
            sampler["negative"] = ["901", 0]
            wf["900"] = {
                "class_type": "SGLDOptions",
                "inputs": {"native_batching": mode},
            }
            wf[by_type["SGLDUNETLoader"]]["inputs"]["sgld_options"] = ["900", 0]
            wf[by_type["SaveImage"]]["inputs"]["filename_prefix"] = prefix
            return wf

        def submit(mode, prefix, seed):
            r = requests.post(
                f"{base}/prompt", json={"prompt": workflow(mode, prefix, seed)}
            )
            if not r.ok:
                return {"error": r.text[:1500]}
            pid = r.json()["prompt_id"]
            start = time.time()
            while True:
                h = requests.get(f"{base}/history/{pid}").json()
                if pid in h:
                    break
                if time.time() - start > 1200:
                    return {"error": "timeout"}
                time.sleep(0.5)
            entry = h[pid]
            status = entry.get("status", {})
            out = {"status": status.get("status_str"), "wall_s": round(time.time() - start, 2)}
            msgs = status.get("messages", [])
            times = {m[0]: m[1].get("timestamp") for m in msgs if isinstance(m, list)}
            if "execution_start" in times and "execution_success" in times:
                out["exec_s"] = round(
                    (times["execution_success"] - times["execution_start"]) / 1000, 2
                )
            errors = [m for m in msgs if m[0] == "execution_error"]
            if errors:
                out["error"] = json.dumps(errors[0][1])[:1500]
            images = []
            for node_out in entry.get("outputs", {}).values():
                for img in node_out.get("images", []):
                    path = os.path.join("/ComfyUI/output", img.get("subfolder", ""), img["filename"])
                    images.append(np.asarray(Image.open(path).convert("RGB"), dtype=np.float64))
            out["images"] = images
            return out

        def psnr(a, b):
            mse = float(((a - b) ** 2).mean())
            return float("inf") if mse == 0 else round(10 * np.log10(255.0**2 / mse), 2)

        runs = {}
        for name, mode, seed in (
            ("warmup_off", "off", 1),  # loads models; other seed so off_a runs
            ("off_a", "off", 1234),
            ("cfg_only", "cfg_only", 1234),
            ("full", "full", 1234),
        ):
            runs[name] = submit(mode, name, seed)
        ref = runs["off_a"].get("images") or []
        summary = {}
        for name, r in runs.items():
            imgs = r.pop("images", [])
            r["num_images"] = len(imgs)
            if imgs and ref and len(imgs) == len(ref):
                r["psnr_vs_off_a_db"] = [psnr(a, b) for a, b in zip(imgs, ref)]
                r["max_abs_vs_off_a"] = [float(np.abs(a - b).max()) for a, b in zip(imgs, ref)]
            summary[name] = r
        res["runs"] = summary

        # Contact sheet: image 0 and 1 of off_a, cfg_only, full.
        rows = []
        for name in ("off_a", "cfg_only", "full"):
            prefix_files = sorted(
                f for f in os.listdir("/ComfyUI/output") if f.startswith(name)
            )
            rows.append(
                np.concatenate(
                    [np.asarray(Image.open(f"/ComfyUI/output/{f}").convert("RGB").resize((384, 384)))
                     for f in prefix_files[:2]],
                    axis=1,
                )
            )
        buf = io.BytesIO()
        Image.fromarray(np.concatenate(rows, axis=0)).save(buf, "PNG")
        res["sheet_png_b64"] = base64.b64encode(buf.getvalue()).decode()

        text = open("/tmp/comfyui.log").read()
        res["log_warnings"] = [
            line for line in text.splitlines()
            if "native_batching" in line or "out of memory" in line.lower()
            or "Traceback" in line
        ][:20]
        res["log_prompt_times"] = [
            line for line in text.splitlines() if "Prompt executed in" in line
        ]
        out = json.dumps(res, default=float)
        print("CONTAINER_RESULTS " + json.dumps({k: v for k, v in res.items() if k != "sheet_png_b64"}, default=float))
        return out
    finally:
        server.terminate()
        try:
            server.wait(30)
        except subprocess.TimeoutExpired:
            server.kill()
        log.close()
        print(open("/tmp/comfyui.log").read()[-6000:])


@app.local_entrypoint()
def main(out: str = "pr44105_comfyui_e2e.png"):
    import base64
    import json

    r = json.loads(run.remote())
    sheet = r.pop("sheet_png_b64", None)
    print("RESULTS_JSON " + json.dumps(r))
    print(json.dumps(r, indent=2))
    if sheet:
        open(out, "wb").write(base64.b64decode(sheet))
        print(f"sheet saved to {out}")
