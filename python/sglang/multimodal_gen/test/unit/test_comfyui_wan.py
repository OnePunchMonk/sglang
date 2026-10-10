# SPDX-License-Identifier: Apache-2.0
"""ComfyUI integrated mode for Wan 2.1 / 2.2: checkpoint spec and loader."""

from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from sglang.multimodal_gen.runtime.loader.comfyui_checkpoints import spec
from sglang.multimodal_gen.runtime.loader.comfyui_checkpoints.wan import (
    WAN_I2V_PIPELINE,
    WAN_T2V_PIPELINE,
    WAN_TI2V_PIPELINE,
    infer_wan_geometry,
    read_safetensors_shapes,
)
from sglang.multimodal_gen.runtime.loader.utils import get_param_names_mapping
from sglang.multimodal_gen.runtime.server_args import set_global_server_args

DIM, FFN, LAYERS, TEXT, FREQ, CLIP = 128, 256, 2, 32, 256, 1280


def _original_wan_state_dict(*, in_dim, out_dim, i2v=False) -> dict[str, torch.Tensor]:
    """Tensor names and shapes as ComfyUI's WanModel (original release layout)."""
    shapes = {
        "patch_embedding.weight": (DIM, in_dim, 1, 2, 2),
        "patch_embedding.bias": (DIM,),
        "text_embedding.0.weight": (DIM, TEXT),
        "text_embedding.0.bias": (DIM,),
        "text_embedding.2.weight": (DIM, DIM),
        "text_embedding.2.bias": (DIM,),
        "time_embedding.0.weight": (DIM, FREQ),
        "time_embedding.0.bias": (DIM,),
        "time_embedding.2.weight": (DIM, DIM),
        "time_embedding.2.bias": (DIM,),
        "time_projection.1.weight": (6 * DIM, DIM),
        "time_projection.1.bias": (6 * DIM,),
        "head.modulation": (1, 2, DIM),
        "head.head.weight": (4 * out_dim, DIM),
        "head.head.bias": (4 * out_dim,),
    }
    for i in range(LAYERS):
        block = f"blocks.{i}."
        shapes[block + "modulation"] = (1, 6, DIM)
        for proj in ("q", "k", "v", "o"):
            for attn in ("self_attn", "cross_attn"):
                shapes[f"{block}{attn}.{proj}.weight"] = (DIM, DIM)
                shapes[f"{block}{attn}.{proj}.bias"] = (DIM,)
        for attn in ("self_attn", "cross_attn"):
            shapes[f"{block}{attn}.norm_q.weight"] = (DIM,)
            shapes[f"{block}{attn}.norm_k.weight"] = (DIM,)
        shapes[block + "norm3.weight"] = (DIM,)
        shapes[block + "norm3.bias"] = (DIM,)
        shapes[block + "ffn.0.weight"] = (FFN, DIM)
        shapes[block + "ffn.0.bias"] = (FFN,)
        shapes[block + "ffn.2.weight"] = (DIM, FFN)
        shapes[block + "ffn.2.bias"] = (DIM,)
        if i2v:
            for proj in ("k_img", "v_img"):
                shapes[f"{block}cross_attn.{proj}.weight"] = (DIM, DIM)
                shapes[f"{block}cross_attn.{proj}.bias"] = (DIM,)
            shapes[block + "cross_attn.norm_k_img.weight"] = (DIM,)
    if i2v:
        shapes.update(
            {
                "img_emb.proj.0.weight": (CLIP,),
                "img_emb.proj.0.bias": (CLIP,),
                "img_emb.proj.1.weight": (CLIP, CLIP),
                "img_emb.proj.1.bias": (CLIP,),
                "img_emb.proj.3.weight": (DIM, CLIP),
                "img_emb.proj.3.bias": (DIM,),
                "img_emb.proj.4.weight": (DIM,),
                "img_emb.proj.4.bias": (DIM,),
            }
        )
    return {name: torch.randn(*shape) for name, shape in shapes.items()}


_VARIANTS = {
    "t2v": dict(in_dim=16, out_dim=16, pipeline=WAN_T2V_PIPELINE),
    "i2v": dict(in_dim=36, out_dim=16, i2v=True, pipeline=WAN_I2V_PIPELINE),
    "ti2v": dict(in_dim=48, out_dim=48, pipeline=WAN_TI2V_PIPELINE),
}


def _server_args(model_path, **overrides):
    # Only the fields the Wan DiT constructor and the loader read.
    args = SimpleNamespace(
        pipeline_config=SimpleNamespace(dit_config=None, dit_precision="fp32"),
        model_paths={},
        transformer_weights_path=None,
        component_weights_paths={},
        model_path=str(model_path),
        dit_precision="fp32",
        component_precisions={},
        quantization=None,
        nunchaku_config=None,
        hsdp_replicate_dim=1,
        hsdp_shard_dim=1,
        pin_cpu_memory=False,
        attention_backend=None,
        attention_backend_config=None,
        comfyui_mode=True,
        disable_autocast=True,
        boundary_ratio=None,
        kv_gather_degree=1,
        should_start_component_on_cpu=lambda _: False,
        should_use_fsdp_for_component=lambda _: False,
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


@pytest.fixture
def cpu_loader(single_process_model_parallel, monkeypatch):
    monkeypatch.setattr(spec, "get_local_torch_device", lambda: torch.device("cpu"))


def _load(tmp_path, variant, **server_overrides):
    cfg = _VARIANTS[variant]
    tensors = _original_wan_state_dict(
        in_dim=cfg["in_dim"], out_dim=cfg["out_dim"], i2v=cfg.get("i2v", False)
    )
    path = tmp_path / f"{variant}.safetensors"
    save_file(tensors, path)
    args = _server_args(path, **server_overrides)
    set_global_server_args(args)
    pipeline = SimpleNamespace(
        pipeline_name=cfg["pipeline"], model_path=str(path), get_module=lambda _: None
    )
    return tensors, args, spec.load_comfyui_transformer(pipeline, args)


@pytest.mark.parametrize("variant", sorted(_VARIANTS))
def test_original_layout_checkpoint_fills_every_sglang_parameter(
    cpu_loader, tmp_path, variant
):
    """Every checkpoint tensor must land, unchanged, on its own SGLang parameter.

    A name-mapping gap would either fail the strict load or leave a parameter
    at its random init, which only shows up as garbage video on a GPU.
    """
    tensors, args, modules = _load(tmp_path, variant)
    model = modules["transformer"]
    mapping = get_param_names_mapping(
        args.pipeline_config.dit_config.arch_config.param_names_mapping
    )
    params = dict(model.state_dict())

    targets = {name: mapping(name)[0] for name in tensors}
    assert len(set(targets.values())) == len(tensors) == len(params)
    for name, target in targets.items():
        assert torch.equal(params[target].float(), tensors[name].float()), name
    assert "transformer_2" not in modules


def test_variant_geometry_selects_pipeline_and_channels(tmp_path):
    for variant, cfg in _VARIANTS.items():
        path = tmp_path / f"{variant}.safetensors"
        save_file(
            _original_wan_state_dict(
                in_dim=cfg["in_dim"], out_dim=cfg["out_dim"], i2v=cfg.get("i2v", False)
            ),
            path,
        )
        geometry = infer_wan_geometry(read_safetensors_shapes(str(path)))
        assert (geometry["in_dim"], geometry["out_dim"]) == (
            cfg["in_dim"],
            cfg["out_dim"],
        )
        assert (geometry["image_dim"] is not None) == cfg.get("i2v", False)


def test_unsupported_checkpoints_are_rejected_before_loading():
    base = {
        name: tuple(t.shape)
        for name, t in _original_wan_state_dict(in_dim=16, out_dim=16).items()
    }
    with pytest.raises(ValueError, match="fp8-scaled"):
        infer_wan_geometry({**base, "blocks.0.self_attn.q.scale_weight": (1,)})
    with pytest.raises(ValueError, match="first-last-frame"):
        infer_wan_geometry({**base, "img_emb.emb_pos": (1, 514, CLIP)})
    with pytest.raises(ValueError, match="original-layout"):
        infer_wan_geometry({"blocks.0.attn1.to_q.weight": (DIM, DIM)})


def test_second_expert_loads_into_transformer_2_with_boundary(cpu_loader, tmp_path):
    """Wan 2.2 A14B: the low-noise file becomes transformer_2 and enables the switch."""
    low_noise = tmp_path / "low_noise.safetensors"
    save_file(_original_wan_state_dict(in_dim=16, out_dim=16), low_noise)
    _, args, modules = _load(
        tmp_path,
        "t2v",
        component_weights_paths={"transformer_2": str(low_noise)},
    )
    assert modules["transformer"] is not modules["transformer_2"]
    assert args.pipeline_config.dit_config.arch_config.boundary_ratio == 0.875

    low_noise_i2v = tmp_path / "low_noise_i2v.safetensors"
    save_file(_original_wan_state_dict(in_dim=36, out_dim=16, i2v=True), low_noise_i2v)
    _, args, _ = _load(
        tmp_path,
        "i2v",
        component_weights_paths={"transformer_2": str(low_noise_i2v)},
    )
    assert args.pipeline_config.dit_config.arch_config.boundary_ratio == 0.9

    with pytest.raises(ValueError, match="does not match the architecture"):
        _load(
            tmp_path,
            "i2v",
            component_weights_paths={"transformer_2": str(low_noise)},
        )

    _, args, _ = _load(
        tmp_path,
        "t2v",
        component_weights_paths={"transformer_2": str(low_noise)},
        boundary_ratio=0.5,
    )
    assert args.pipeline_config.dit_config.arch_config.boundary_ratio == 0.5


def test_prebuilt_modules_keep_the_second_expert(cpu_loader):
    experts = {"transformer": object(), "transformer_2": object()}
    pipeline = SimpleNamespace(pipeline_name="x", get_module=lambda _: "scheduler")
    modules = spec.load_comfyui_transformer(pipeline, None, experts)
    assert modules["transformer_2"] is experts["transformer_2"]


def test_single_expert_checkpoint_keeps_boundary_switch_off(cpu_loader, tmp_path):
    _, args, _ = _load(tmp_path, "t2v")
    assert args.pipeline_config.dit_config.arch_config.boundary_ratio is None


def test_frame_timesteps_expand_frame_major_and_split_across_sp_ranks(monkeypatch):
    """Tokens are frame-major, and each SP rank keeps only the frames it owns."""
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages import (
        wan_ti2v,
    )

    monkeypatch.setattr(wan_ti2v, "get_local_torch_device", lambda: torch.device("cpu"))
    # 4 frames, 4x4 latent with 2x2 patches -> 4 tokens per frame.
    batch = SimpleNamespace(
        raw_latent_shape=(1, 48, 4, 4, 4), did_sp_shard_latents=False
    )
    frames = torch.tensor([[0.0, 10.0, 20.0, 30.0]])

    full = wan_ti2v.expand_comfyui_frame_timestep(
        batch, frames, torch.float32, (1, 2, 2)
    )
    assert full.shape == (1, 16)
    assert full[0].tolist() == [0.0] * 4 + [10.0] * 4 + [20.0] * 4 + [30.0] * 4

    # Token-shard mode (or a T not divisible by SP): latents stay whole, so the
    # full timestep goes to the DiT, which pads and shards it itself.
    monkeypatch.setattr(wan_ti2v, "get_sp_world_size", lambda: 2)
    monkeypatch.setattr(wan_ti2v, "get_sp_parallel_rank", lambda: 1)
    unsharded = wan_ti2v.expand_comfyui_frame_timestep(
        batch, frames, torch.float32, (1, 2, 2)
    )
    assert unsharded.shape == (1, 16)

    batch.did_sp_shard_latents = True
    monkeypatch.setattr(wan_ti2v, "get_sp_parallel_rank", lambda: 1)
    local = wan_ti2v.expand_comfyui_frame_timestep(
        batch, frames, torch.float32, (1, 2, 2)
    )
    assert local[0].tolist() == [20.0] * 4 + [30.0] * 4


def test_denoise_stage_uses_frame_timesteps_when_comfyui_sends_them(monkeypatch):
    from sglang.multimodal_gen.runtime.pipelines_core.stages.denoising import (
        DenoisingStage,
    )

    server_args = SimpleNamespace(
        pipeline_config=SimpleNamespace(
            dit_config=SimpleNamespace(
                arch_config=SimpleNamespace(patch_size=(1, 2, 2))
            ),
            task_type=None,
        )
    )
    batch = SimpleNamespace(
        raw_latent_shape=(1, 48, 2, 4, 4),
        did_sp_shard_latents=False,
        condition_image=None,
        extra={"comfyui_frame_timesteps": torch.tensor([[0.0, 5.0]])},
    )
    from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages import (
        wan_ti2v,
    )

    monkeypatch.setattr(wan_ti2v, "get_local_torch_device", lambda: torch.device("cpu"))
    out = DenoisingStage.expand_timestep_before_forward(
        None, batch, server_args, torch.tensor(5.0), torch.float32, None, None
    )
    assert out.shape == (1, 8) and out[0, 0] == 0.0 and out[0, -1] == 5.0
