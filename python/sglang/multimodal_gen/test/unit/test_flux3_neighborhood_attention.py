# SPDX-License-Identifier: Apache-2.0
"""FlexAttention fallback mask vs. a brute-force NATTEN window reference."""

import itertools

import pytest
import torch

pytest.importorskip("torch.nn.attention.flex_attention")
from torch.nn.attention.flex_attention import create_mask

from sglang.multimodal_gen.runtime.models.vaes import flux3_neighborhood_attention as na


def _window(i: int, n: int, k: int, causal: bool) -> range:
    if causal:
        return range(max(0, i - k + 1), i + 1)
    start = min(max(i - k // 2, 0), n - k)
    return range(start, start + k)


def _reference(grid, kernel, causal) -> torch.Tensor:
    coords = list(itertools.product(*[range(n) for n in grid]))
    ref = torch.zeros(len(coords), len(coords), dtype=torch.bool)
    for qi, q in enumerate(coords):
        wins = [
            set(_window(p, n, k, c)) for p, n, k, c in zip(q, grid, kernel, causal)
        ]
        for ki, kv in enumerate(coords):
            ref[qi, ki] = all(p in w for p, w in zip(kv, wins))
    return ref


def _dense_mask(grid, kernel, causal) -> torch.Tensor:
    kernel = tuple(min(k, n) for k, n in zip(kernel, grid))
    strides = [1] * len(grid)
    for a in range(len(grid) - 2, -1, -1):
        strides[a] = strides[a + 1] * grid[a + 1]
    params = torch.tensor(list(zip(grid, kernel, strides)), dtype=torch.int32)
    s = 1
    for n in grid:
        s *= n
    mod = na._make_mask_mod(params, tuple(causal))
    return create_mask(mod, 1, 1, s, s, device="cpu")[0, 0]


CASES = [
    ((5, 6), (3, 3), (False, False)),
    ((5, 6), (4, 2), (False, False)),  # even kernels
    ((5, 6), (3, 4), (True, False)),
    ((5, 6), (3, 3), (True, True)),
    ((4, 3), (7, 9), (False, False)),  # kernel larger than grid
    ((4, 3), (7, 2), (True, False)),
    ((3, 4, 5), (3, 3, 3), (False, False, False)),
    ((3, 4, 5), (2, 4, 3), (True, False, False)),
    ((3, 4, 5), (5, 2, 9), (True, True, True)),
    ((3, 4, 5), (3, 1, 3), (False, True, False)),
]


@pytest.mark.parametrize("grid,kernel,causal", CASES)
def test_mask_matches_natten_window(grid, kernel, causal):
    ckernel = tuple(min(k, n) for k, n in zip(kernel, grid))
    assert torch.equal(_dense_mask(grid, kernel, causal), _reference(grid, ckernel, causal))


def test_rank_mismatch_asserts():
    x = torch.zeros(1, 4, 4, 2, 8)
    with pytest.raises(AssertionError):
        na.neighborhood_attention(x, x, x, kernel_size=[3, 3, 3])
    with pytest.raises(AssertionError):
        na.neighborhood_attention(x, x, x, kernel_size=[3, 3], is_causal=[True])


def test_block_mask_cache_is_lru(monkeypatch):
    calls = []

    class _Fake:
        def __call__(self, *a, **kw):
            calls.append(1)
            return object()

    monkeypatch.setattr(na, "_compiled_create_block_mask", lambda: _Fake())
    monkeypatch.setattr(na, "_BLOCK_MASK_CACHE_MAX", 2)
    na._block_masks.clear()
    dev = torch.device("cpu")
    a = na._neighborhood_block_mask((4, 4), (3, 3), (False, False), dev)
    na._neighborhood_block_mask((5, 5), (3, 3), (False, False), dev)
    na._neighborhood_block_mask((4, 4), (3, 3), (False, False), dev)  # hit: a is MRU
    na._neighborhood_block_mask((6, 6), (3, 3), (False, False), dev)  # evicts (5, 5)
    assert len(calls) == 3
    assert na._neighborhood_block_mask((4, 4), (3, 3), (False, False), dev) is a
    assert len(calls) == 3
    na._block_masks.clear()


def test_block_mask_key_clamps_kernel(monkeypatch):
    monkeypatch.setattr(
        na, "_compiled_create_block_mask", lambda: lambda *a, **k: object()
    )
    na._block_masks.clear()
    dev = torch.device("cpu")
    a = na._neighborhood_block_mask((4, 4), (4, 4), (False, False), dev)
    assert na._neighborhood_block_mask((4, 4), (9, 9), (False, False), dev) is a
    na._block_masks.clear()
