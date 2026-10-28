import pytest
import torch
from safetensors.torch import safe_open, save_file

from sglang.multimodal_gen.runtime.loader.parallel_safetensors import (
    ParallelPreadReader,
    ParallelSliceHandle,
)


@pytest.fixture
def checkpoint(tmp_path):
    g = torch.Generator().manual_seed(0)
    tensors = {
        "w_f32": torch.randn(64, 48, generator=g),
        "w_bf16": torch.randn(32, 96, generator=g).bfloat16(),
        "w_3d": torch.randn(8, 16, 12, generator=g).half(),
        "scalar": torch.tensor(3.5),
        "bias": torch.randn(7, generator=g),
        # Large enough to span several pread chunks.
        "big": torch.randn(1024, 5000, generator=g),
    }
    path = str(tmp_path / "m.safetensors")
    save_file(tensors, path)
    return path, tensors


@pytest.mark.parametrize("name", ["w_f32", "w_bf16", "w_3d", "scalar", "bias", "big"])
def test_whole_tensor_matches_safe_open(checkpoint, name):
    path, _ = checkpoint
    with ParallelPreadReader("cpu", num_threads=4) as reader:
        got = reader.read(path, name)
    with safe_open(path, framework="pt") as handle:
        assert torch.equal(got, handle.get_tensor(name))


@pytest.mark.parametrize(
    "name,slices",
    [
        ("w_f32", (slice(16, 48), slice(None))),
        ("w_f32", (slice(None), slice(12, 36))),
        ("w_bf16", (slice(None), slice(48, 96))),
        ("w_3d", (slice(2, 6), slice(None), slice(None))),
        ("w_3d", (slice(None), slice(4, 12), slice(None))),
        ("w_3d", (slice(None), slice(None), slice(6, 12))),
        ("w_3d", (slice(2, 4), slice(None), slice(3, 9))),
        ("w_f32", (slice(5, 5), slice(None))),
        ("big", (slice(100, 900), slice(None))),
        ("big", (slice(None), slice(2500, 5000))),
    ],
)
def test_slice_matches_safe_open(checkpoint, name, slices):
    path, _ = checkpoint
    with ParallelPreadReader("cpu", num_threads=4) as reader:
        got = ParallelSliceHandle(path, reader).get_slice(name)[slices]
    with safe_open(path, framework="pt") as handle:
        want = handle.get_slice(name)[slices].contiguous()
    assert got.shape == want.shape
    assert torch.equal(got, want)


def test_inflight_bound_forces_buffer_reuse(checkpoint):
    path, _ = checkpoint
    with ParallelPreadReader("cpu", num_threads=2, max_inflight_bytes=1 << 20) as r:
        for _ in range(5):
            r.read(path, "big")
        assert r._inflight_bytes == 0
