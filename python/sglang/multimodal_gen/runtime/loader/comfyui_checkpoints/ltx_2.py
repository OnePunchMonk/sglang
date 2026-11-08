# SPDX-License-Identifier: Apache-2.0
"""ComfyUI single-file checkpoint support for LTX-2.3 fine-tunes."""

from sglang.multimodal_gen.configs.models.dits.ltx_2 import LTX2Config
from sglang.multimodal_gen.runtime.loader.comfyui_checkpoints.spec import (
    ComfyUICheckpointSpec,
    register_comfyui_checkpoint,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs


def _build_dit_config(server_args: ServerArgs) -> LTX2Config:
    dit_config = getattr(server_args.pipeline_config, "dit_config", None)
    if not isinstance(dit_config, LTX2Config):
        dit_config = LTX2Config()
        server_args.pipeline_config.dit_config = dit_config

    # Sulphur 2 is an LTX-2.3 fine-tune. These flags select the 2.3
    # transformer blocks and Gemma connector behavior in the shared model.
    arch = dit_config.arch_config
    arch.ltx_variant = "ltx_2_3"
    arch.caption_proj_before_connector = True
    arch.cross_attention_adaln = True
    arch.apply_gated_attention = True
    return dit_config


register_comfyui_checkpoint(
    "LTX2Pipeline",
    ComfyUICheckpointSpec(
        dit_cls_name="LTX2VideoTransformer3DModel",
        build_dit_config=_build_dit_config,
        # Keep the transformer portion of ComfyUI's all-in-one checkpoint;
        # LTX-2.3's text encoder and VAEs are loaded by the existing pipeline.
        checkpoint_key_filter=lambda key: key.startswith("model.diffusion_model."),
    ),
)
