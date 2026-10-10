# SPDX-License-Identifier: Apache-2.0
"""Pack / unpack contract for ComfyUI model adapters."""

from types import SimpleNamespace

import pytest
import torch

from sglang.multimodal_gen.apps.ComfyUI_SGLDiffusion.executors.adapter import (
    get_adapter_class,
    registered_model_types,
)
from sglang.multimodal_gen.apps.ComfyUI_SGLDiffusion.executors.flux import FluxAdapter
from sglang.multimodal_gen.apps.ComfyUI_SGLDiffusion.executors.wan import WanAdapter
from sglang.multimodal_gen.apps.ComfyUI_SGLDiffusion.executors.zimage import (
    ZImageAdapter,
)
from sglang.multimodal_gen.runtime.loader.comfyui_checkpoints.wan import (
    WAN_I2V_PIPELINE,
    WAN_T2V_PIPELINE,
    WAN_TI2V_PIPELINE,
)


def test_registered_comfyui_model_types() -> None:
    types = registered_model_types()
    assert "flux" in types
    assert "lumina2" in types
    assert get_adapter_class("lumina2") is ZImageAdapter
    assert get_adapter_class("flux") is FluxAdapter
    assert get_adapter_class("lumina2").pipeline_class_name == "ZImagePipeline"


def test_zimage_pack_sets_seq_lens_and_time_dim() -> None:
    adapter = ZImageAdapter()
    x = torch.ones(1, 16, 90, 160)
    timestep = torch.tensor([1.0])
    context = torch.ones(1, 19, 2560)
    packed = adapter.pack(x, timestep, context)
    assert packed.latents.shape == (1, 16, 1, 90, 160)
    assert packed.prompt_embeds[0].shape == (19, 2560)
    assert packed.prompt_seq_lens == [[19]]
    assert packed.height == 720
    assert packed.width == 1280
    assert torch.equal(packed.timesteps, timestep * 1000.0)

    pred = torch.ones(1, 16, 1, 90, 160)
    out = adapter.unpack(pred, packed, x)
    assert out.shape == x.shape


def test_flux_pack_and_unpack_roundtrip() -> None:
    adapter = FluxAdapter()
    x = torch.arange(1 * 16 * 8 * 8, dtype=torch.float32).reshape(1, 16, 8, 8)
    timestep = torch.tensor([0.5])
    context = torch.ones(1, 8, 4096)
    y = torch.ones(1, 768)
    packed = adapter.pack(x, timestep, context, y=y, guidance=torch.tensor([1.0]))
    assert packed.latents.ndim == 3
    assert packed.pooled_embeds[0] is y
    assert packed.guidance_scale == 1.0
    out = adapter.unpack(packed.latents, packed, x)
    assert out.shape == x.shape
    assert torch.equal(out, x)

    default = adapter.pack(x, timestep, context, y=y)
    assert default.guidance_scale == 3.5


@pytest.mark.parametrize(
    "unet_config, pipeline",
    [
        ({"model_type": "t2v", "in_dim": 16, "out_dim": 16}, WAN_T2V_PIPELINE),
        ({"model_type": "i2v", "in_dim": 36, "out_dim": 16}, WAN_I2V_PIPELINE),
        ({"model_type": "t2v", "in_dim": 48, "out_dim": 48}, WAN_TI2V_PIPELINE),
    ],
)
def test_wan_detected_variant_selects_pipeline(unet_config, pipeline) -> None:
    assert get_adapter_class("wan2.1") is WanAdapter
    assert WanAdapter.pipeline_class_for(SimpleNamespace(unet_config=unet_config)) == (
        pipeline
    )


@pytest.mark.parametrize(
    "extra",
    [
        {"model_type": "vace"},
        {"model_type": "camera"},
        {"model_type": "t2v", "flf_pos_embed_token_number": 514},
        {"model_type": "t2v", "in_dim_ref_conv": 16},
        {"model_type": "t2v", "causal_ar": True},
    ],
)
def test_wan_subfamilies_without_an_sglang_dit_are_rejected(extra) -> None:
    """ComfyUI reports every Wan sub-family as ``wan2.1``; only t2v / i2v can run."""
    config = SimpleNamespace(unet_config={"in_dim": 16, "out_dim": 16, **extra})
    with pytest.raises(ValueError, match="Unsupported Wan variant"):
        WanAdapter.pipeline_class_for(config)


def test_wan_i2v_pack_splits_conditioning_channels_from_noise() -> None:
    """ComfyUI hands the DiT noise+mask+image as one tensor; the model predicts noise only."""
    adapter = WanAdapter(noise_channels=16)
    x = torch.randn(1, 36, 3, 6, 8)
    context = torch.randn(1, 11, 4096)
    clip = torch.randn(1, 257, 1280)
    packed = adapter.pack(x, torch.tensor([800.0]), context, clip_fea=clip)

    assert torch.equal(packed.latents, x[:, :16])
    assert torch.equal(packed.extra_req["image_latent"], x[:, 16:])
    assert packed.extra_req["image_embeds"] == [clip]
    assert packed.prompt_embeds == [context]
    assert (packed.height, packed.width) == (48, 64)
    assert packed.unpack_ctx["num_frames"] == 9
    assert "comfyui_frame_timesteps" not in packed.extra_req

    pred = torch.randn(1, 16, 3, 6, 8, dtype=torch.float64)
    out = adapter.unpack(pred, packed, x)
    assert out.shape == (1, 16, 3, 6, 8) and out.dtype == x.dtype

    adapter.drop_cached_fields(packed)
    assert "image_embeds" not in packed.extra_req and not packed.prompt_embeds


def test_wan_t2v_pack_sends_no_conditioning_and_one_shared_sigma() -> None:
    packed = WanAdapter().pack(
        torch.randn(2, 16, 2, 4, 4),
        torch.tensor([500.0, 500.0]),
        torch.randn(2, 5, 4096),
    )
    assert packed.extra_req == {}
    assert packed.timesteps.shape == (1,)


def test_wan_ti2v_frame_timesteps_only_sent_when_frames_differ() -> None:
    """A conditioned first frame carries its own (lower) timestep in TI2V masking."""
    adapter = WanAdapter(noise_channels=48)
    x = torch.randn(1, 48, 2, 4, 4)
    context = torch.randn(1, 5, 4096)

    uniform = adapter.pack(x, torch.tensor([[700.0, 700.0]]), context)
    assert "comfyui_frame_timesteps" not in uniform.extra_req

    masked = adapter.pack(x, torch.tensor([[0.0, 700.0]]), context)
    assert torch.equal(
        masked.extra_req["comfyui_frame_timesteps"], torch.tensor([[0.0, 700.0]])
    )
    assert masked.timesteps.item() == 700.0

    req = SimpleNamespace(extra=None, latents=None)
    adapter.fill_req(req, masked)
    assert "comfyui_frame_timesteps" in req.extra
    assert not hasattr(req, "comfyui_frame_timesteps")


def test_wan_rejects_conditioning_it_would_silently_drop() -> None:
    with pytest.raises(NotImplementedError, match="reference_latent"):
        WanAdapter().pack(
            torch.randn(1, 16, 2, 4, 4),
            torch.tensor([500.0]),
            torch.randn(1, 5, 4096),
            reference_latent=torch.randn(1, 16, 1, 4, 4),
        )


def test_wan_cond_key_separates_start_images_with_identical_text() -> None:
    """Same prompt, different I2V start image: cached conds must not be reused."""
    from sglang.multimodal_gen.apps.ComfyUI_SGLDiffusion.executors.wan import (
        WanExecutor,
    )

    executor = WanExecutor.__new__(WanExecutor)
    adapter = WanAdapter()
    context = torch.randn(1, 5, 4096)
    x_a = torch.randn(1, 36, 2, 4, 4)
    x_b = x_a.clone()
    x_b[:, 16:] += 1.0
    t = torch.tensor([500.0])

    key_a = executor._cond_key(adapter.pack(x_a, t, context))
    assert key_a == executor._cond_key(adapter.pack(x_a, t, context))
    assert key_a != executor._cond_key(adapter.pack(x_b, t, context))
