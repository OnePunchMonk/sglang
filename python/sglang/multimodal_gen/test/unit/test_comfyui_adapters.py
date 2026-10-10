# SPDX-License-Identifier: Apache-2.0
"""Pack / unpack contract for ComfyUI model adapters."""

import pytest
import torch

from sglang.multimodal_gen.apps.ComfyUI_SGLDiffusion.executors.adapter import (
    get_adapter_class,
    registered_model_types,
)
from sglang.multimodal_gen.apps.ComfyUI_SGLDiffusion.executors.base import (
    SGLDiffusionExecutor,
)
from sglang.multimodal_gen.apps.ComfyUI_SGLDiffusion.executors.flux import FluxAdapter
from sglang.multimodal_gen.apps.ComfyUI_SGLDiffusion.executors.zimage import (
    ZImageAdapter,
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


def _flux_step(**kwargs):
    x = torch.zeros(2, 16, 8, 8)
    return FluxAdapter().pack(
        x, torch.tensor([0.5]), torch.ones(2, 8, 4096), y=torch.ones(2, 768), **kwargs
    )


def test_flux_pack_forwards_control_residuals_with_gaps() -> None:
    seq = (8 // 2) * (8 // 2)
    double = [torch.ones(2, seq, 3072), None, torch.ones(2, seq, 3072)]
    single = [torch.ones(2, seq, 3072)]
    packed = _flux_step(control={"input": double, "output": single})
    assert packed.control["input"][1] is None
    assert len(packed.control["input"]) == 3
    assert len(packed.control["output"]) == 1
    assert _flux_step().control is None
    assert _flux_step(control={"input": [None]}).control is None


def test_flux_pack_rejects_misshaped_control() -> None:
    with pytest.raises(ValueError, match=r"control input\[0\]"):
        _flux_step(control={"input": [torch.ones(2, 3, 3072)]})


def test_control_is_rejected_for_adapters_that_do_not_apply_it() -> None:
    class _Executor(SGLDiffusionExecutor):
        adapter_cls = ZImageAdapter

        def __init__(self):
            torch.nn.Module.__init__(self)
            self.adapter = ZImageAdapter()

    executor = _Executor()
    control = {"input": [torch.ones(1, 4, 8)]}
    with pytest.raises(ValueError, match="does not apply control"):
        executor(
            torch.ones(1, 16, 8, 8),
            torch.tensor([1.0]),
            torch.ones(1, 8, 16),
            control=control,
        )
    assert FluxAdapter.applied_conditioning == ("control",)
    assert ZImageAdapter.applied_conditioning == ()
