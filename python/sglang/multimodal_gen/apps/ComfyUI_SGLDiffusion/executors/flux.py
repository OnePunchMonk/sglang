"""Flux adapter for the ComfyUI DiT-forward contract."""

import torch

from .adapter import ComfyUIModelAdapter, PackedForward
from .base import SGLDiffusionExecutor


def _flux_guidance_scale(guidance) -> float:
    if guidance is None:
        return 3.5
    if torch.is_tensor(guidance):
        return float(guidance.detach().reshape(-1)[0].item())
    return float(guidance)


def _pack_control(control, batch_size: int, seq_len: int) -> dict[str, list] | None:
    """Validate ComfyUI's {"input": double-block, "output": single-block} residuals.

    None entries are kept so list positions still match the block index.
    """
    if not control:
        return None
    packed = {}
    for key in ("input", "output"):
        residuals = []
        for index, residual in enumerate(control.get(key) or []):
            if residual is not None:
                if residual.ndim != 3 or residual.shape[:2] != (batch_size, seq_len):
                    raise ValueError(
                        f"Flux control {key}[{index}] has shape {tuple(residual.shape)}, "
                        f"expected ({batch_size}, {seq_len}, hidden)"
                    )
                residual = residual.detach()
            residuals.append(residual)
        if any(r is not None for r in residuals):
            packed[key] = residuals
    return packed or None


class FluxAdapter(ComfyUIModelAdapter):
    model_types = ("flux",)
    pipeline_class_name = "FluxPipeline"
    applied_conditioning = ("control",)

    def pack(
        self, x, timestep, context, y=None, guidance=None, **kwargs
    ) -> PackedForward:
        packed = self._pack_latents(x)
        t5_seq = int(context.shape[-2]) if context.ndim >= 2 else int(context.shape[0])
        clip_batch = int(y.shape[0]) if y is not None else 1
        return PackedForward(
            latents=packed,
            timesteps=timestep * 1000.0,
            prompt_embeds=[y, context],
            prompt_seq_lens=[[clip_batch], [t5_seq]],
            pooled_embeds=[y],
            height=x.shape[-2] * 8,
            width=x.shape[-1] * 8,
            guidance_scale=_flux_guidance_scale(guidance),
            control=_pack_control(
                kwargs.get("control"), packed.shape[0], packed.shape[1]
            ),
            unpack_ctx={
                "height": x.shape[-2],
                "width": x.shape[-1],
                "channels": x.shape[1],
            },
        )

    def unpack(self, noise_pred, packed, x):
        ctx = packed.unpack_ctx
        return self._unpack_latents(
            noise_pred, ctx["height"], ctx["width"], ctx["channels"]
        ).to(x.device)

    @staticmethod
    def _unpack_latents(latents, height, width, channels):
        batch_size = latents.shape[0]
        latents = latents.view(batch_size, height // 2, width // 2, channels, 2, 2)
        latents = latents.permute(0, 3, 1, 4, 2, 5)
        return latents.reshape(batch_size, channels, height, width)

    @staticmethod
    def _pack_latents(latents):
        batch_size, num_channels_latents, height, width = latents.shape
        latents = latents.view(
            batch_size, num_channels_latents, height // 2, 2, width // 2, 2
        )
        latents = latents.permute(0, 2, 4, 1, 3, 5)
        return latents.reshape(
            batch_size, (height // 2) * (width // 2), num_channels_latents * 4
        )


class FluxExecutor(SGLDiffusionExecutor):
    adapter_cls = FluxAdapter
