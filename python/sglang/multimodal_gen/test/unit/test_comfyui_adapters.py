# SPDX-License-Identifier: Apache-2.0
"""Pack / unpack contract for ComfyUI model adapters."""

import copy
import pickle
from types import SimpleNamespace

import pytest
import torch

from sglang.multimodal_gen.apps.ComfyUI_SGLDiffusion.executors.adapter import (
    get_adapter_class,
    registered_model_types,
)
from sglang.multimodal_gen.apps.ComfyUI_SGLDiffusion.executors.cache_options import (
    CACHE_OPTIONS_KEY,
    build_cache_options,
    read_cache_options,
    with_cache_options,
)
from sglang.multimodal_gen.apps.ComfyUI_SGLDiffusion.executors.flux import (
    FluxAdapter,
    FluxExecutor,
)
from sglang.multimodal_gen.apps.ComfyUI_SGLDiffusion.executors.minimax_h3 import (
    MiniMaxH3Executor,
    worker_transformer_options,
)
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


class _FakePatcher:
    def __init__(self, model_options, executor):
        self.model_options = model_options
        self.model = SimpleNamespace(diffusion_model=executor)

    def clone(self):
        return _FakePatcher(
            copy.deepcopy(self.model_options), self.model.diffusion_model
        )


def _bare_executor(cls, enable_cache_dit=None):
    executor = cls.__new__(cls)
    torch.nn.Module.__init__(executor)
    executor.adapter = cls.adapter_cls()
    executor.enable_cache_dit = enable_cache_dit
    return executor


def test_cache_options_node_returns_clone_and_keeps_input() -> None:
    patcher = _FakePatcher(
        {"transformer_options": {"sample_sigmas": [1.0, 0.0]}},
        _bare_executor(MiniMaxH3Executor),
    )
    options = build_cache_options(True, {"max_warmup_steps": 2})
    out = with_cache_options(patcher, options)
    assert out is not patcher
    assert patcher.model_options["transformer_options"] == {"sample_sigmas": [1.0, 0.0]}
    stored = out.model_options["transformer_options"]
    assert stored["sample_sigmas"] == [1.0, 0.0]
    assert read_cache_options(stored) == options
    pickle.dumps(options)


def test_cache_options_reject_bad_requests() -> None:
    with pytest.raises(ValueError, match="Unknown cache_dit_params"):
        build_cache_options(True, {"not_a_knob": 1})
    with pytest.raises(ValueError, match="enable_cache_dit is not on"):
        build_cache_options(None, {"max_warmup_steps": 2})


def test_cache_dit_on_unsupported_model_fails_clearly() -> None:
    flux = _bare_executor(FluxExecutor)
    on = build_cache_options(True, None)
    with pytest.raises(ValueError, match="FluxExecutor does not support Cache-DiT"):
        with_cache_options(_FakePatcher({}, flux), on)
    # A workflow that bypasses the node still fails before any request is sent.
    with pytest.raises(ValueError, match="does not support Cache-DiT"):
        flux.forward(
            torch.ones(1, 16, 8, 8),
            torch.tensor([0.5]),
            torch.ones(1, 8, 4096),
            y=torch.ones(1, 768),
            transformer_options={CACHE_OPTIONS_KEY: on},
        )
    # Off and default are valid everywhere.
    with_cache_options(_FakePatcher({}, flux), build_cache_options(False, None))
    with_cache_options(_FakePatcher({}, flux), build_cache_options(None, None))


def _h3_kwargs(executor, cache_options):
    packed = SimpleNamespace(
        guidance_scale=1.0, height=8, width=8, cache_options=cache_options
    )
    return executor._sampling_params_kwargs(packed, torch.tensor([0.5]))


def test_h3_per_run_cache_dit_overrides_loader_flag() -> None:
    loader_on = _bare_executor(MiniMaxH3Executor, enable_cache_dit=True)
    run_off = build_cache_options(False, None)
    assert _h3_kwargs(loader_on, run_off)["enable_cache_dit"] is False
    # Unset per-run value falls back to the loader option (backward compatible).
    assert _h3_kwargs(loader_on, build_cache_options(None, None))["enable_cache_dit"]
    assert "enable_cache_dit" not in _h3_kwargs(_bare_executor(MiniMaxH3Executor), {})
    params = {"max_warmup_steps": 3}
    kwargs = _h3_kwargs(loader_on, build_cache_options(True, params))
    assert kwargs["cache_dit_params"] == params


def test_cache_options_not_sent_in_h3_worker_transformer_options() -> None:
    opts = {CACHE_OPTIONS_KEY: build_cache_options(True, None), "sample_sigmas": [1.0]}
    assert worker_transformer_options(opts) == {"sample_sigmas": [1.0]}
