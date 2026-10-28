import modal
T = "<path to trees>"
app = modal.App("wan-animate2-dbg")
image = modal.Image.from_registry("lmsysorg/sglang:dev").add_local_dir(
    f"{T}/headfull/python/sglang", "/root/head/sglang", copy=True, ignore=["**/__pycache__", "**/.claude"])

@app.function(image=image, gpu="H100", cpu=8, memory=65536, timeout=1200)
def dbg():
    import os, subprocess, sys
    f = "sglang/multimodal_gen/test/unit/test_wan_animate_2_parallelism.py"
    env = dict(os.environ, PYTHONPATH="/root/head", PYTHONFAULTHANDLER="1")
    out = []
    p = subprocess.run([sys.executable, "-m", "pytest", "-v", "-W", "ignore", "--tb=short", "-p", "no:cacheprovider", f],
                       env=env, cwd="/root/head", capture_output=True, text=True)
    out.append(("pytest_full_file", p.returncode, p.stdout[-2500:], p.stderr[-800:]))
    return out

@app.local_entrypoint()
def main():
    for step, rc, so, se in dbg.remote():
        print("==", step, "RC", rc); print(so); print("STDERR", se)
