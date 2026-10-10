# SPDX-License-Identifier: Apache-2.0
"""ComfyUI Wan 2.1 / 2.2 checkpoint spec.

ComfyUI stores Wan with the original release's parameter names, so the DiT
geometry is read from tensor shapes instead of a ``config.json``. Wan 2.2
A14B ships as two files (high-noise and low-noise experts) with identical
geometry; both go through the same spec.
"""

from __future__ import annotations

import math

from safetensors import safe_open

from sglang.multimodal_gen.configs.models.dits.wanvideo import WanVideoConfig
from sglang.multimodal_gen.runtime.loader.comfyui_checkpoints.spec import (
    ComfyUICheckpointSpec,
    ParamNamesMapping,
    register_comfyui_checkpoint,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs

WAN_HEAD_DIM = 128
WAN_PATCH_SIZE = (1, 2, 2)
# Wan 2.2 TI2V-5B uses the 48-channel VAE; every other Wan DiT emits 16.
WAN_TI2V_LATENT_CHANNELS = 48

WAN_T2V_PIPELINE = "WanComfyUIT2VPipeline"
WAN_I2V_PIPELINE = "WanComfyUII2VPipeline"
WAN_TI2V_PIPELINE = "WanComfyUITI2VPipeline"

# Default high-noise/low-noise switch point for the two-expert checkpoints,
# as sigma * 1000. Matches the official Wan 2.2 A14B configs.
WAN_DEFAULT_BOUNDARY_RATIO = {
    WAN_T2V_PIPELINE: 0.875,
    WAN_I2V_PIPELINE: 0.9,
}

_FP8_SCALED_MARKERS = (".scale_weight", ".scale_input", "scaled_fp8")


def read_safetensors_shapes(path: str) -> dict[str, tuple[int, ...]]:
    with safe_open(path, framework="pt", device="cpu") as handle:
        return {
            name: tuple(handle.get_slice(name).get_shape()) for name in handle.keys()
        }


def _strip_prefix(shapes: dict[str, tuple[int, ...]]) -> dict[str, tuple[int, ...]]:
    prefix = "model.diffusion_model."
    return {
        (name[len(prefix) :] if name.startswith(prefix) else name): shape
        for name, shape in shapes.items()
    }


def infer_wan_geometry(shapes: dict[str, tuple[int, ...]]) -> dict[str, int | bool]:
    """Derive the DiT geometry from tensor shapes of an original-layout Wan file."""
    shapes = _strip_prefix(shapes)
    for name in shapes:
        if name.endswith(_FP8_SCALED_MARKERS) or name == "scaled_fp8":
            raise ValueError(
                "fp8-scaled Wan checkpoints are not supported in ComfyUI integrated "
                "mode; use the BF16/FP16 file or a GGUF via transformer_weights_path"
            )
    if "img_emb.emb_pos" in shapes:
        raise ValueError(
            "Wan first-last-frame checkpoints (img_emb.emb_pos) are not supported"
        )
    try:
        out_channels_dim, in_dim = shapes["patch_embedding.weight"][:2]
        head_out = shapes["head.head.weight"][0]
        dim = shapes["blocks.0.self_attn.q.weight"][0]
        ffn_dim = shapes["blocks.0.ffn.0.weight"][0]
        text_dim = shapes["text_embedding.0.weight"][1]
        freq_dim = shapes["time_embedding.0.weight"][1]
    except KeyError as exc:
        raise ValueError(
            f"Not an original-layout Wan checkpoint: missing tensor {exc.args[0]!r}"
        ) from exc
    if dim % WAN_HEAD_DIM != 0 or out_channels_dim != dim:
        raise ValueError(
            f"Unexpected Wan hidden size {dim} (patch embed {out_channels_dim})"
        )
    num_layers = 1 + max(
        int(name.split(".")[1]) for name in shapes if name.startswith("blocks.")
    )
    image_dim = (
        shapes["img_emb.proj.0.weight"][0]
        if "img_emb.proj.0.weight" in shapes
        else None
    )
    return {
        "dim": dim,
        "in_dim": in_dim,
        "out_dim": head_out // math.prod(WAN_PATCH_SIZE),
        "ffn_dim": ffn_dim,
        "num_layers": num_layers,
        "text_dim": text_dim,
        "freq_dim": freq_dim,
        "image_dim": image_dim,
        "qk_norm": "blocks.0.self_attn.norm_q.weight" in shapes,
        "cross_attn_norm": "blocks.0.norm3.weight" in shapes,
    }


def wan_pipeline_name(geometry: dict[str, int | bool]) -> str:
    if geometry["out_dim"] == WAN_TI2V_LATENT_CHANNELS:
        return WAN_TI2V_PIPELINE
    if geometry["in_dim"] > geometry["out_dim"]:
        return WAN_I2V_PIPELINE
    return WAN_T2V_PIPELINE


def _build_dit_config(server_args: ServerArgs) -> WanVideoConfig:
    geometry = infer_wan_geometry(read_safetensors_shapes(server_args.model_path))
    dit_config = WanVideoConfig()
    arch = dit_config.arch_config
    arch.num_attention_heads = geometry["dim"] // WAN_HEAD_DIM
    arch.attention_head_dim = WAN_HEAD_DIM
    arch.in_channels = geometry["in_dim"]
    arch.out_channels = geometry["out_dim"]
    arch.ffn_dim = geometry["ffn_dim"]
    arch.num_layers = geometry["num_layers"]
    arch.text_dim = geometry["text_dim"]
    arch.freq_dim = geometry["freq_dim"]
    arch.qk_norm = "rms_norm_across_heads" if geometry["qk_norm"] else None
    arch.cross_attn_norm = geometry["cross_attn_norm"]
    arch.image_dim = geometry["image_dim"]
    arch.added_kv_proj_dim = geometry["dim"] if geometry["image_dim"] else None
    arch.__post_init__()

    pipeline_name = wan_pipeline_name(geometry)
    if server_args.component_weights_paths.get("transformer_2"):
        arch.boundary_ratio = (
            server_args.boundary_ratio
            if server_args.boundary_ratio is not None
            else WAN_DEFAULT_BOUNDARY_RATIO.get(pipeline_name)
        )
    server_args.pipeline_config.dit_config = dit_config
    return dit_config


def _attn(target: str) -> str:
    return rf"blocks.\1.{target}.\2"


_PARAM_NAMES_MAPPING: ParamNamesMapping = {
    r"^model\.diffusion_model\.(.*)$": (r"\1", None, None),
    r"^patch_embedding\.(.*)$": (r"patch_embedding.proj.\1", None, None),
    r"^text_embedding\.0\.(.*)$": (
        r"condition_embedder.text_embedder.fc_in.\1",
        None,
        None,
    ),
    r"^text_embedding\.2\.(.*)$": (
        r"condition_embedder.text_embedder.fc_out.\1",
        None,
        None,
    ),
    r"^time_embedding\.0\.(.*)$": (
        r"condition_embedder.time_embedder.mlp.fc_in.\1",
        None,
        None,
    ),
    r"^time_embedding\.2\.(.*)$": (
        r"condition_embedder.time_embedder.mlp.fc_out.\1",
        None,
        None,
    ),
    r"^time_projection\.1\.(.*)$": (
        r"condition_embedder.time_modulation.linear.\1",
        None,
        None,
    ),
    r"^img_emb\.proj\.0\.(.*)$": (
        r"condition_embedder.image_embedder.norm1.\1",
        None,
        None,
    ),
    r"^img_emb\.proj\.1\.(.*)$": (
        r"condition_embedder.image_embedder.ff.fc_in.\1",
        None,
        None,
    ),
    r"^img_emb\.proj\.3\.(.*)$": (
        r"condition_embedder.image_embedder.ff.fc_out.\1",
        None,
        None,
    ),
    r"^img_emb\.proj\.4\.(.*)$": (
        r"condition_embedder.image_embedder.norm2.\1",
        None,
        None,
    ),
    r"^blocks\.(\d+)\.self_attn\.q\.(.*)$": (_attn("to_q"), None, None),
    r"^blocks\.(\d+)\.self_attn\.k\.(.*)$": (_attn("to_k"), None, None),
    r"^blocks\.(\d+)\.self_attn\.v\.(.*)$": (_attn("to_v"), None, None),
    r"^blocks\.(\d+)\.self_attn\.o\.(.*)$": (_attn("to_out"), None, None),
    r"^blocks\.(\d+)\.self_attn\.norm_q\.(.*)$": (_attn("norm_q"), None, None),
    r"^blocks\.(\d+)\.self_attn\.norm_k\.(.*)$": (_attn("norm_k"), None, None),
    r"^blocks\.(\d+)\.cross_attn\.q\.(.*)$": (_attn("attn2.to_q"), None, None),
    r"^blocks\.(\d+)\.cross_attn\.k\.(.*)$": (_attn("attn2.to_k"), None, None),
    r"^blocks\.(\d+)\.cross_attn\.v\.(.*)$": (_attn("attn2.to_v"), None, None),
    r"^blocks\.(\d+)\.cross_attn\.o\.(.*)$": (_attn("attn2.to_out"), None, None),
    r"^blocks\.(\d+)\.cross_attn\.norm_q\.(.*)$": (_attn("attn2.norm_q"), None, None),
    r"^blocks\.(\d+)\.cross_attn\.norm_k\.(.*)$": (_attn("attn2.norm_k"), None, None),
    r"^blocks\.(\d+)\.cross_attn\.k_img\.(.*)$": (
        _attn("attn2.add_k_proj"),
        None,
        None,
    ),
    r"^blocks\.(\d+)\.cross_attn\.v_img\.(.*)$": (
        _attn("attn2.add_v_proj"),
        None,
        None,
    ),
    r"^blocks\.(\d+)\.cross_attn\.norm_k_img\.(.*)$": (
        _attn("attn2.norm_added_k"),
        None,
        None,
    ),
    # Original norm3 is the pre-cross-attention LayerNorm, which SGLang fuses
    # into the self-attention residual norm.
    r"^blocks\.(\d+)\.norm3\.(.*)$": (
        _attn("self_attn_residual_norm.norm"),
        None,
        None,
    ),
    r"^blocks\.(\d+)\.ffn\.0\.(.*)$": (_attn("ffn.fc_in"), None, None),
    r"^blocks\.(\d+)\.ffn\.2\.(.*)$": (_attn("ffn.fc_out"), None, None),
    r"^blocks\.(\d+)\.modulation$": (r"blocks.\1.scale_shift_table", None, None),
    r"^head\.head\.(.*)$": (r"proj_out.\1", None, None),
    r"^head\.modulation$": (r"scale_shift_table", None, None),
}


for _pipeline_name in (WAN_T2V_PIPELINE, WAN_I2V_PIPELINE, WAN_TI2V_PIPELINE):
    register_comfyui_checkpoint(
        _pipeline_name,
        ComfyUICheckpointSpec(
            dit_cls_name="WanTransformer3DModel",
            build_dit_config=_build_dit_config,
            param_names_mapping=_PARAM_NAMES_MAPPING,
            # Both mappings claim the diffusers-style names this one also
            # rewrites (patch_embedding.*), so layering would double-apply.
            inherit_config_mapping=False,
        ),
    )
