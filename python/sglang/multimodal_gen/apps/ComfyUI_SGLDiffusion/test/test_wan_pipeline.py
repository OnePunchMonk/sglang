"""Wan 2.1 / 2.2 through the ComfyUI worker, driven by the real WanAdapter.

Each case launches its own worker process: the scheduler client is a process
singleton, so two generators cannot coexist in one interpreter. Tiny random
checkpoints exercise every variant; set ``SGLANG_TEST_WAN_T2V_PATH`` to a real
original-layout file (e.g. ``wan2.1_t2v_1.3B_bf16.safetensors``) for one
full-size sanity step.
"""

import json
import os
import subprocess
import sys
import tempfile

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs a CUDA device"
)

PIPELINES = {
    "t2v": "WanComfyUIT2VPipeline",
    "i2v": "WanComfyUII2VPipeline",
    "ti2v": "WanComfyUITI2VPipeline",
}
NOISE_CHANNELS = {"t2v": 16, "i2v": 16, "ti2v": 48}
IN_DIM = {"t2v": 16, "i2v": 36, "ti2v": 48}
T, H, W, TEXT_LEN = 3, 8, 8, 5


def _worker(spec: dict, out_path: str) -> None:
    from sglang.multimodal_gen.apps.ComfyUI_SGLDiffusion.executors.wan import (
        WanAdapter,
    )
    from sglang.multimodal_gen.apps.ComfyUI_SGLDiffusion.test.passthrough import (
        prepare_passthrough_request,
    )
    from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams
    from sglang.multimodal_gen.runtime.entrypoints.diffusion_generator import (
        DiffGenerator,
    )
    from sglang.multimodal_gen.runtime.entrypoints.utils import prepare_request

    variant = spec["variant"]
    kwargs = {}
    if spec.get("expert_2"):
        kwargs["component_weights_paths"] = {"transformer_2": spec["expert_2"]}
    if spec.get("boundary_ratio") is not None:
        kwargs["boundary_ratio"] = spec["boundary_ratio"]
    generator = DiffGenerator.from_pretrained(
        model_path=spec["model_path"],
        pipeline_class_name=PIPELINES[variant],
        num_gpus=1,
        sp_degree=1,
        comfyui_mode=True,
        dit_cpu_offload=False,
        **kwargs,
    )

    torch.manual_seed(0)
    x = torch.randn(1, IN_DIM[variant], T, H, W, device="cuda", dtype=torch.bfloat16)
    if spec.get("cond_shift"):
        x[:, NOISE_CHANNELS[variant] :] += spec["cond_shift"]
    context = torch.randn(
        1, TEXT_LEN, spec.get("text_dim", 32), device="cuda", dtype=torch.bfloat16
    )
    clip = (
        torch.randn(1, 257, 1280, device="cuda", dtype=torch.bfloat16)
        if spec.get("clip")
        else None
    )
    timestep = torch.tensor(spec["timestep"], device="cuda")

    adapter = WanAdapter(noise_channels=NOISE_CHANNELS[variant])
    packed = adapter.pack(x, timestep, context, clip_fea=clip)
    sampling_params = SamplingParams.from_user_sampling_params_args(
        generator.server_args.model_path,
        server_args=generator.server_args,
        prompt=" ",
        guidance_scale=1.0,
        height=packed.height,
        width=packed.width,
        num_frames=packed.unpack_ctx["num_frames"],
        num_inference_steps=1,
        seed=0,
        save_output=False,
    )
    req = prepare_request(
        server_args=generator.server_args, sampling_params=sampling_params
    )
    adapter.fill_req(req, packed)
    prepare_passthrough_request(req)
    output = generator._send_to_scheduler_and_wait_for_response([req])
    torch.save(adapter.unpack(output.noise_pred, packed, x).float().cpu(), out_path)
    generator.shutdown()


def _run(tmp_path, **spec) -> torch.Tensor:
    spec_path = tmp_path / "spec.json"
    out_path = tmp_path / f"out_{len(list(tmp_path.iterdir()))}.pt"
    spec_path.write_text(json.dumps(spec))
    proc = subprocess.run(
        [sys.executable, __file__, "--worker", str(spec_path), str(out_path)],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, (proc.stdout + proc.stderr)[-4000:]
    return torch.load(out_path)


@pytest.fixture(scope="module")
def checkpoints():
    from safetensors.torch import save_file

    from sglang.multimodal_gen.test.unit.test_comfyui_wan import (
        _original_wan_state_dict,
    )

    with tempfile.TemporaryDirectory() as root:
        paths = {}
        for name, (in_dim, out_dim, i2v) in {
            "t2v": (16, 16, False),
            "t2v_low": (16, 16, False),
            "i2v": (36, 16, True),
            "ti2v": (48, 48, False),
        }.items():
            torch.manual_seed(len(paths))
            tensors = _original_wan_state_dict(in_dim=in_dim, out_dim=out_dim, i2v=i2v)
            path = os.path.join(root, f"{name}.safetensors")
            save_file(tensors, path)
            paths[name] = path
        yield paths


@pytest.mark.parametrize("variant", ["t2v", "i2v", "ti2v"])
def test_worker_step_returns_finite_noise_pred(tmp_path, checkpoints, variant):
    out = _run(
        tmp_path,
        variant=variant,
        model_path=checkpoints[variant],
        timestep=[500.0],
        clip=variant == "i2v",
    )
    assert out.shape == (1, NOISE_CHANNELS[variant], T, H, W)
    assert torch.isfinite(out).all() and out.abs().max() > 0


def test_i2v_start_image_changes_the_prediction(tmp_path, checkpoints):
    """The concatenated mask+image channels must reach the DiT, not be dropped."""
    common = dict(
        variant="i2v", model_path=checkpoints["i2v"], timestep=[500.0], clip=True
    )
    base = _run(tmp_path, **common)
    shifted = _run(tmp_path, cond_shift=1.0, **common)
    assert (base - shifted).abs().max() > 1e-3


def test_ti2v_conditioned_frame_timestep_changes_the_prediction(tmp_path, checkpoints):
    common = dict(variant="ti2v", model_path=checkpoints["ti2v"])
    uniform = _run(tmp_path, timestep=[[700.0, 700.0, 700.0]], **common)
    masked = _run(tmp_path, timestep=[[0.0, 700.0, 700.0]], **common)
    assert (uniform - masked).abs().max() > 1e-3


def test_experts_switch_by_timestep_against_boundary(tmp_path, checkpoints):
    """High-noise steps use `transformer`, low-noise steps use `transformer_2`."""
    high_only = _run(
        tmp_path, variant="t2v", model_path=checkpoints["t2v"], timestep=[900.0]
    )
    low_only = _run(
        tmp_path, variant="t2v", model_path=checkpoints["t2v_low"], timestep=[100.0]
    )
    dual = dict(
        variant="t2v",
        model_path=checkpoints["t2v"],
        expert_2=checkpoints["t2v_low"],
        boundary_ratio=0.5,
    )
    torch.testing.assert_close(
        _run(tmp_path, timestep=[900.0], **dual), high_only, atol=2e-2, rtol=2e-2
    )
    torch.testing.assert_close(
        _run(tmp_path, timestep=[100.0], **dual), low_only, atol=2e-2, rtol=2e-2
    )


@pytest.mark.skipif(
    "SGLANG_TEST_WAN_T2V_PATH" not in os.environ, reason="no real Wan checkpoint"
)
def test_real_t2v_checkpoint_step(tmp_path):
    out = _run(
        tmp_path,
        variant="t2v",
        model_path=os.environ["SGLANG_TEST_WAN_T2V_PATH"],
        timestep=[800.0],
        text_dim=4096,
    )
    assert out.shape == (1, 16, T, H, W) and torch.isfinite(out).all()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        _worker(json.load(open(sys.argv[2])), sys.argv[3])
    else:
        sys.exit(pytest.main([__file__, "-v"]))
