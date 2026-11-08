# SPDX-License-Identifier: Apache-2.0
"""Tests for LTX-2.3 single-file checkpoint registration."""

from types import SimpleNamespace

from sglang.multimodal_gen.configs.models.dits.ltx_2 import LTX2Config
from sglang.multimodal_gen.runtime.loader.comfyui_checkpoints import (
    get_comfyui_checkpoint_spec,
)
from sglang.multimodal_gen.runtime.loader.utils import get_param_names_mapping


def test_ltx23_comfyui_spec_builds_the_right_architecture():
    spec = get_comfyui_checkpoint_spec("LTX2Pipeline")
    config = LTX2Config()
    server_args = SimpleNamespace(pipeline_config=SimpleNamespace(dit_config=config))

    assert spec is not None
    assert spec.dit_cls_name == "LTX2VideoTransformer3DModel"
    assert spec.build_dit_config(server_args) is config
    assert config.arch_config.ltx_variant == "ltx_2_3"
    assert config.arch_config.caption_proj_before_connector
    assert config.arch_config.cross_attention_adaln
    assert config.arch_config.apply_gated_attention


def test_ltx23_comfyui_spec_keeps_only_the_transformer_weights():
    spec = get_comfyui_checkpoint_spec("LTX2Pipeline")

    assert spec is not None
    assert spec.checkpoint_key_filter("model.diffusion_model.transformer_blocks.0")
    assert not spec.checkpoint_key_filter("vae.decoder.conv_in.weight")
    assert not spec.checkpoint_key_filter(
        "model.diffusion_model.audio_embeddings_connector.learnable_registers"
    )
    assert not spec.checkpoint_key_filter(
        "model.diffusion_model.video_embeddings_connector.learnable_registers"
    )


def test_ltx23_comfyui_spec_maps_diffusers_checkpoint_names():
    spec = get_comfyui_checkpoint_spec("LTX2Pipeline")
    config = LTX2Config()
    server_args = SimpleNamespace(pipeline_config=SimpleNamespace(dit_config=config))
    spec.build_dit_config(server_args)
    mapping = get_param_names_mapping(config.arch_config.param_names_mapping)

    assert (
        mapping("model.diffusion_model.transformer_blocks.0.ff.net.0.proj.weight")[0]
        == "transformer_blocks.0.ff.proj_in.weight"
    )


def test_sulphur_resolves_to_ltx23_configs():
    from sglang.multimodal_gen.configs.pipeline_configs.ltx_2 import LTX23PipelineConfig
    from sglang.multimodal_gen.registry import (
        _get_config_info,
        get_non_diffusers_pipeline_name,
    )
    from sglang.multimodal_gen.runtime.server_args import ServerArgs

    assert (
        _get_config_info("SulphurAI/Sulphur-2-base").pipeline_config_cls
        is LTX23PipelineConfig
    )
    assert get_non_diffusers_pipeline_name("SulphurAI/Sulphur-2-base") == "LTX2Pipeline"
    assert ServerArgs._is_ltx23_model_path("SulphurAI/Sulphur-2-base")
