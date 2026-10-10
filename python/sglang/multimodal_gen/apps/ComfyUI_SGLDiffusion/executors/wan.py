# SPDX-License-Identifier: Apache-2.0
"""Wan 2.1 / 2.2 adapter for the ComfyUI DiT-forward contract.

ComfyUI concatenates the I2V conditioning (4 mask + 16 image latent channels)
onto the noisy latent before calling the DiT, so ``x`` can carry more channels
than the model predicts. The adapter splits them: SGLang takes the noisy
latent as ``latents`` and the conditioning as ``image_latent``.
"""

from __future__ import annotations

import torch

from sglang.multimodal_gen.runtime.loader.comfyui_checkpoints.wan import (
    WAN_T2V_PIPELINE,
    wan_pipeline_name,
)

from .adapter import ComfyUIModelAdapter, PackedForward
from .base import SGLDiffusionExecutor

# Spatial VAE stride: 8 for Wan 2.1 / 2.2-A14B, 16 for the 2.2 TI2V-5B VAE.
_SPATIAL_STRIDE = {16: 8, 48: 16}
_TEMPORAL_STRIDE = 4

_SUPPORTED_MODEL_TYPES = ("t2v", "i2v")
# ComfyUI conds that only exist on Wan sub-families SGLang has no DiT for.
_UNSUPPORTED_CONDS = (
    "reference_latent",
    "context_latents",
    "time_dim_concat",
    "vace_context",
    "camera_conditions",
    "audio_embed",
)


def _validate_unet_config(unet_config: dict) -> None:
    model_type = unet_config.get("model_type")
    unsupported = [
        key
        for key in ("flf_pos_embed_token_number", "in_dim_ref_conv", "causal_ar")
        if unet_config.get(key)
    ]
    if model_type not in _SUPPORTED_MODEL_TYPES or unsupported:
        raise ValueError(
            "Unsupported Wan variant for SGLang integrated mode: "
            f"model_type={model_type!r}, unsupported={unsupported}. "
            f"Supported: Wan 2.1 / 2.2 {', '.join(_SUPPORTED_MODEL_TYPES)} "
            "(including the A14B two-expert and TI2V-5B releases)."
        )


class WanAdapter(ComfyUIModelAdapter):
    model_types = ("wan2.1",)
    pipeline_class_name = WAN_T2V_PIPELINE

    def __init__(self, noise_channels: int = 16) -> None:
        self.noise_channels = noise_channels

    @classmethod
    def pipeline_class_for(cls, model_config) -> str:
        unet_config = model_config.unet_config
        _validate_unet_config(unet_config)
        return wan_pipeline_name(
            {
                "in_dim": unet_config["in_dim"],
                "out_dim": unet_config["out_dim"],
            }
        )

    def pack(self, x, timestep, context, clip_fea=None, **kwargs) -> PackedForward:
        for key in _UNSUPPORTED_CONDS:
            if kwargs.get(key) is not None:
                raise NotImplementedError(
                    f"Wan conditioning {key!r} is not supported in SGLang integrated mode"
                )
        noise = x[:, : self.noise_channels]
        cond = x[:, self.noise_channels :]
        stride = _SPATIAL_STRIDE[self.noise_channels]
        num_frames = (noise.shape[2] - 1) * _TEMPORAL_STRIDE + 1

        extra_req = {}
        # ComfyUI already passes sigma * 1000 for flow models (Flux alone uses 0..1).
        # [B] normally; [B, T] when TI2V gives conditioned frames their own timestep.
        timesteps = timestep.reshape(timestep.shape[0], -1).to(torch.float32)
        if timesteps.shape[1] > 1 and not torch.all(timesteps == timesteps[:, :1]):
            extra_req["comfyui_frame_timesteps"] = timesteps
        if cond.shape[1] > 0:
            extra_req["image_latent"] = cond
        if clip_fea is not None:
            extra_req["image_embeds"] = [clip_fea]
        return PackedForward(
            latents=noise,
            # Rows share one sigma; the largest frame value drives expert choice.
            timesteps=timesteps.max().reshape(1),
            prompt_embeds=[context],
            height=noise.shape[-2] * stride,
            width=noise.shape[-1] * stride,
            extra_req=extra_req,
            unpack_ctx={"num_frames": num_frames},
        )

    def fill_req(self, req, packed: PackedForward) -> None:
        extra_req = dict(packed.extra_req)
        frame_timesteps = extra_req.pop("comfyui_frame_timesteps", None)
        packed.extra_req = extra_req
        super().fill_req(req, packed)
        if frame_timesteps is not None:
            req.extra = {
                **(req.extra or {}),
                "comfyui_frame_timesteps": frame_timesteps,
            }

    def unpack(self, noise_pred, packed, x):
        return noise_pred.to(device=x.device, dtype=x.dtype)

    def drop_cached_fields(self, packed: PackedForward) -> None:
        super().drop_cached_fields(packed)
        packed.extra_req.pop("image_embeds", None)


class WanExecutor(SGLDiffusionExecutor):
    adapter_cls = WanAdapter

    def __init__(self, generator, model_path, model, config):
        super().__init__(generator, model_path, model, config)
        self.adapter = WanAdapter(noise_channels=config.unet_config["out_dim"])

    def _cond_key(self, packed) -> tuple | None:
        key = super()._cond_key(packed)
        image_latent = packed.extra_req.get("image_latent")
        if key is None or image_latent is None:
            return key
        flat = image_latent.reshape(-1)
        # Same text with a different start image must not reuse cached I2V conds.
        return key + (
            tuple(int(dim) for dim in image_latent.shape),
            float(flat[0].item()),
            float(flat[-1].item()),
        )

    def _sampling_params_kwargs(self, packed, timestep) -> dict:
        kwargs = super()._sampling_params_kwargs(packed, timestep)
        kwargs["num_frames"] = packed.unpack_ctx["num_frames"]
        return kwargs
