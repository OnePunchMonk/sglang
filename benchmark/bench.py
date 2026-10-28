"""Wan-Animate-2 reference K/V (P10 padded cache, P11 FP8 cache) A/B on Modal.

  modal run bench.py::prefetch   CPU: weights -> Volume
  modal run bench.py::unit       CPU: targeted unit tests, base vs head
  modal run bench.py::bench      1x H100: micro + e2e + import timing (60 min cap)
"""

import json

import modal

SCRATCH = "<scratch>"
BASE_PY = f"{SCRATCH}/base/python/sglang"
HEAD_PY = "<path to the PR checkout>/python/sglang"
MODEL_ID = "Wan-AI/Wan2.2-Animate-2-14B-Diffusers"
CI = (
    "https://raw.githubusercontent.com/sgl-project/ci-data-diffusion/"
    "c7258eb56e9951878c66fc0fb3673d9f4775b0b2/diffusion-ci/inputs/wan_animate_2"
)
TEST = "sglang/multimodal_gen/test/unit/test_wan_animate_2_parallelism.py"

app = modal.App("wan-animate2-refkv")
vol = modal.Volume.from_name("wan-animate2-refkv", create_if_missing=True)
image = (
    modal.Image.from_registry("lmsysorg/sglang:dev")
    .add_local_dir(BASE_PY, "/root/base/sglang", copy=True, ignore=["**/__pycache__"])
    .add_local_dir(HEAD_PY, "/root/head/sglang", copy=True, ignore=["**/__pycache__"])
    .add_local_file(__file__.replace("bench.py", "worker.py"), "/root/worker.py", copy=True)
)
ENV = {"HF_HOME": "/vol/hf"}


def _run(args, root, extra_env=None, timeout=None):
    import os
    import subprocess
    import sys

    env = dict(os.environ, PYTHONPATH=root, **ENV, **(extra_env or {}))
    p = subprocess.run(
        [sys.executable, *args], env=env, capture_output=True, text=True, cwd=root, timeout=timeout
    )
    return p.returncode, p.stdout, p.stderr


@app.function(image=image, cpu=4, memory=8192, timeout=3600, volumes={"/vol": vol})
def prefetch():
    import os

    os.environ.update(ENV)
    from huggingface_hub import snapshot_download

    path = snapshot_download(MODEL_ID)
    vol.commit()
    return path


@app.function(image=image, cpu=8, memory=32768, timeout=1800)
def unit():
    out = {}
    for name, root in (("base", "/root/base"), ("head", "/root/head")):
        rc, so, se = _run(["-m", "pytest", "-q", "-p", "no:cacheprovider", "-W", "ignore", "--tb=short", TEST], root)
        out[name] = (rc, so[-3000:] + se[-1500:])
    # The head test file against base code: the new FP8 / padded-cache tests must fail there.
    import shutil

    shutil.copy(f"/root/head/{TEST}", f"/root/base/{TEST}")
    rc, so, se = _run(["-m", "pytest", "-q", "-p", "no:cacheprovider", "-W", "ignore", "--tb=short", TEST], "/root/base")
    out["head_tests_on_base"] = (rc, so[-2000:] + se[-1000:])
    return out


@app.function(image=image, gpu="H100", cpu=8, memory=131072, timeout=3600, volumes={"/vol": vol})
def bench():
    res = {}
    rc, so, se = _run(["-c", "import torch;print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name())"], "/root/head")
    res["env"] = so.strip()
    # P13: cold import of the block module (the torch.compile wrapper is built here).
    rc, so, se = _run(["/root/worker.py", "import"], "/root/base")
    res["import"] = so.strip() or se[-1500:]
    # Attention micro-benchmark.
    for name, root, fp8 in (("base", "/root/base", "0"), ("p10", "/root/head", "0"), ("p10_fp8", "/root/head", "1")):
        rc, so, se = _run(["/root/worker.py", "micro"], root, {"FP8": fp8}, timeout=900)
        res[f"micro_{name}"] = so.strip().splitlines()[-1] if rc == 0 else se[-3000:]
    # End-to-end.
    for name, root, fp8 in (("base", "/root/base", "false"), ("p10", "/root/head", "false"), ("p10_fp8", "/root/head", "true")):
        rc, so, se = _run(
            ["/root/worker.py", "e2e", name], root,
            {"SGLANG_DIFFUSION_WAN_ANIMATE_2_FP8_REFERENCE_KV": fp8, "CI": CI, "MODEL": MODEL_ID},
            timeout=1200,
        )
        res[f"e2e_{name}"] = so.strip().splitlines()[-1] if rc == 0 else se[-4000:]
    rc, so, se = _run(["/root/worker.py", "psnr"], "/root/head")
    res["quality"] = so.strip() or se[-2000:]
    vol.commit()
    return res


@app.local_entrypoint()
def main(job: str = "bench"):
    fn = {"prefetch": prefetch, "unit": unit, "bench": bench}[job]
    out = fn.remote()
    print(json.dumps(out, indent=1, default=str))
    with open(__file__.replace("bench.py", f"{job}_out.json"), "w") as f:
        json.dump(out, f, indent=1, default=str)
