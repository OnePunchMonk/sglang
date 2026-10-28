import json, modal
app = modal.App("wan-animate2-psnr")
vol = modal.Volume.from_name("wan-animate2-refkv")
image = modal.Image.from_registry("lmsysorg/sglang:dev")

@app.function(image=image, cpu=4, memory=16384, timeout=900, volumes={"/vol": vol})
def psnr():
    import av, numpy as np
    def frames(p):
        c = av.open(p); s = c.streams.video[0]
        meta = {"fps": float(s.average_rate), "w": s.width, "h": s.height}
        return np.stack([f.to_ndarray(format="rgb24") for f in c.decode(video=0)]).astype(np.float64), meta
    a, ma = frames("/vol/videos_fa3b/main.mp4"); b, mb = frames("/vol/videos_fa3b/head.mp4")
    n = min(len(a), len(b))
    per = [10 * np.log10(255**2 / max(((x - y) ** 2).mean(), 1e-12)) for x, y in zip(a[:n], b[:n])]
    # Temporal consistency proxy: mean abs frame-to-frame change, per video.
    tc = lambda v: float(np.abs(np.diff(v, axis=0)).mean())
    return {"main": dict(ma, frames=len(a), temporal_mad=tc(a)), "head": dict(mb, frames=len(b), temporal_mad=tc(b)),
            "psnr_mean_db": float(np.mean(per)), "psnr_min_frame_db": float(min(per)),
            "psnr_max_frame_db": float(max(per)), "mean_abs_diff": float(np.abs(a[:n] - b[:n]).mean()),
            "bitwise_equal_decoded": bool((a[:n] == b[:n]).all())}

@app.local_entrypoint()
def main():
    out = psnr.remote(); print(json.dumps(out, indent=1)); json.dump(out, open("psnr_fa3_out.json", "w"), indent=1)
