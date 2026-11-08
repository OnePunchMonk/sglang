# SPDX-License-Identifier: Apache-2.0
"""ComfyUI single-file checkpoint support for LTX-2.3 fine-tunes."""

from sglang.multimodal_gen.configs.models.dits.ltx_2 import LTX2Config
from sglang.multimodal_gen.runtime.loader.comfyui_checkpoints.spec import (
    ComfyUICheckpointSpec,
    register_comfyui_checkpoint,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs

_COMFYUI_ONLY_CONNECTOR_PREFIXES = (
    "model.diffusion_model.audio_embeddings_connector.",
    "model.diffusion_model.video_embeddings_connector.",
)


def _is_ltx2_dit_weight(key: str) -> bool:
    """Keep transformer weights; ComfyUI runs text connectors on its side."""
    return key.startswith("model.diffusion_model.") and not key.startswith(
        _COMFYUI_ONLY_CONNECTOR_PREFIXES
    )


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
        # Keep the DiT portion of ComfyUI's all-in-one checkpoint. ComfyUI
        # runs the LTX text connectors itself and sends their output embeddings.
        checkpoint_key_filter=_is_ltx2_dit_weight,
    ),
)
