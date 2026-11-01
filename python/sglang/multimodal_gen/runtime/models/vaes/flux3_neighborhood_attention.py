# SPDX-License-Identifier: Apache-2.0
"""Neighborhood attention of the FLUX 3 video VAE without NATTEN.

NATTEN is an optional dependency; when it is missing, the VAE runs the same
windows through a compiled FlexAttention block mask. The windows follow
NATTEN's ``na2d`` / ``na3d`` (stride 1, no dilation): on a non-causal axis the
window of query ``i`` is ``[start, start + k)`` with
``start = clamp(i - k // 2, 0, L - k)`` (shifted inward at the borders); on a
causal axis it is ``[max(0, i - k + 1), i]``.
"""

from __future__ import annotations

import importlib.util
from collections import OrderedDict
from functools import lru_cache

import torch

from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

# One block mask per (grid, kernel, causal, device); a server sees a handful.
# LRU: an OrderedDict, hits move to the end, the oldest entry is evicted.
_BLOCK_MASK_CACHE_MAX = 16
_block_masks: OrderedDict[tuple, object] = OrderedDict()


@lru_cache(maxsize=1)
def natten_available() -> bool:
    available = importlib.util.find_spec("natten") is not None
    if not available:
        logger.warning(
            "NATTEN is not installed; the FLUX 3 video VAE uses the FlexAttention "
            "fallback, compiled on the first request. Install a wheel matching "
            "your torch/CUDA build from https://whl.natten.org/ to use NATTEN."
        )
    return available


@lru_cache(maxsize=1)
def _compiled_flex_attention():
    from torch.nn.attention.flex_attention import flex_attention

    # Uncompiled, flex_attention materializes the dense score matrix. dynamic=True
    # keeps one graph across grid shapes; dynamic=False recompiles per shape and
    # falls back to uncompiled after torch's recompile_limit (8).
    return torch.compile(flex_attention, dynamic=True)


@lru_cache(maxsize=1)
def _compiled_create_block_mask():
    from torch.nn.attention.flex_attention import create_block_mask

    # Compiled so mask construction stays sparse (eager is O(S^2) memory).
    return torch.compile(create_block_mask, dynamic=True)


def _make_mask_mod(params: torch.Tensor, causal: tuple[bool, ...]):
    """Build the NATTEN window mask_mod.

    ``params`` is an int32 ``(ndim, 3)`` tensor of ``(n, k, stride)`` per axis.
    Sizes are read from the tensor, not closed over as Python ints, so a new
    grid shape does not retrace the compiled function. Only the causal flags
    stay Python values.
    """

    def mask_mod(batch_idx, head_idx, q_idx, kv_idx):
        inside = None
        for axis, is_causal in enumerate(causal):
            n, k, stride = params[axis, 0], params[axis, 1], params[axis, 2]
            q_pos = (q_idx // stride) % n
            k_pos = (kv_idx // stride) % n
            if is_causal:
                axis_ok = (k_pos <= q_pos) & (k_pos > q_pos - k)
            else:
                start = torch.minimum(torch.clamp(q_pos - k // 2, min=0), n - k)
                axis_ok = (k_pos >= start) & (k_pos < start + k)
            inside = axis_ok if inside is None else inside & axis_ok
        return inside

    return mask_mod


def _neighborhood_block_mask(
    grid: tuple[int, ...],
    kernel: tuple[int, ...],
    causal: tuple[bool, ...],
    device: torch.device,
):
    kernel = tuple(min(k, n) for k, n in zip(kernel, grid))
    key = (grid, kernel, causal, str(device))
    mask = _block_masks.get(key)
    if mask is not None:
        _block_masks.move_to_end(key)
        return mask
    strides = [1] * len(grid)
    for axis in range(len(grid) - 2, -1, -1):
        strides[axis] = strides[axis + 1] * grid[axis + 1]
    params = torch.tensor(
        list(zip(grid, kernel, strides)), dtype=torch.int32, device=device
    )

    seq_len = 1
    for n in grid:
        seq_len *= n
    mask = _compiled_create_block_mask()(
        _make_mask_mod(params, causal),
        B=None,
        H=None,
        Q_LEN=seq_len,
        KV_LEN=seq_len,
        device=device,
    )
    if len(_block_masks) >= _BLOCK_MASK_CACHE_MAX:
        _block_masks.popitem(last=False)
    _block_masks[key] = mask
    return mask


def neighborhood_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    kernel_size: list[int],
    is_causal: list[bool] | None = None,
) -> torch.Tensor:
    """NATTEN-layout ``(B, *grid, heads, head_dim)`` in and out, like ``na2d`` / ``na3d``."""
    batch, *grid, heads, head_dim = q.shape
    assert len(kernel_size) == len(grid), (
        f"kernel_size {list(kernel_size)} does not match grid {grid}"
    )
    if is_causal is not None:
        assert len(is_causal) == len(grid), (
            f"is_causal {list(is_causal)} does not match grid {grid}"
        )
    causal = tuple(is_causal or [False] * len(grid))
    mask = _neighborhood_block_mask(tuple(grid), tuple(kernel_size), causal, q.device)

    def to_flex(x: torch.Tensor) -> torch.Tensor:
        return x.reshape(batch, -1, heads, head_dim).transpose(1, 2)

    out = _compiled_flex_attention()(
        to_flex(q), to_flex(k), to_flex(v), block_mask=mask
    )
    return out.transpose(1, 2).reshape(q.shape)
