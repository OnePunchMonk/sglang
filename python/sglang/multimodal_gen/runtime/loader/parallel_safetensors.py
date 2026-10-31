# SPDX-License-Identifier: Apache-2.0
"""Threaded `pread` of safetensors byte ranges into pinned host buffers.

`safe_open` reads one tensor at a time through mmap page faults on a single
thread, which is CPU-bound well below storage bandwidth even from a warm page
cache. This module parses the safetensors header itself, reads byte ranges with
a thread pool, and copies them to the device asynchronously.

A slice along dim 0 is one contiguous byte range, so only that range is read.
A slice along any other dim reads the whole tensor and narrows on the device:
twice the bytes read in parallel beats a strided mmap gather.
"""

import json
import os
import struct
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import torch

from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

_DTYPES: dict[str, torch.dtype] = {
    "F64": torch.float64,
    "F32": torch.float32,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "I64": torch.int64,
    "I32": torch.int32,
    "I16": torch.int16,
    "I8": torch.int8,
    "U8": torch.uint8,
    "BOOL": torch.bool,
    "F8_E4M3": torch.float8_e4m3fn,
    "F8_E5M2": torch.float8_e5m2,
}

_NUM_THREADS = 12
_MAX_INFLIGHT_BYTES = 2 << 30
# Arbitrary; large enough to amortize a syscall, small enough to spread one tensor.
_CHUNK_BYTES = 16 << 20
_MIN_BUFFER_BYTES = 1 << 20


class _TensorInfo:
    __slots__ = ("dtype", "shape", "begin", "end")

    def __init__(
        self, dtype: torch.dtype, shape: tuple[int, ...], begin: int, end: int
    ):
        self.dtype = dtype
        self.shape = shape
        self.begin = begin
        self.end = end


def parse_safetensors_header(path: str) -> dict[str, _TensorInfo]:
    """Tensor name -> dtype, shape and absolute file byte range."""
    with open(path, "rb") as f:
        (header_len,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(header_len))
    data_start = 8 + header_len
    infos: dict[str, _TensorInfo] = {}
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        begin, end = meta["data_offsets"]
        infos[name] = _TensorInfo(
            _DTYPES[meta["dtype"]],
            tuple(meta["shape"]),
            data_start + begin,
            data_start + end,
        )
    return infos


def _round_buffer_bytes(nbytes: int) -> int:
    size = _MIN_BUFFER_BYTES
    while size < nbytes:
        size <<= 1
    return size


class ParallelPreadReader:
    """Reads tensors or dim slices of safetensors files onto `device`.

    Host staging buffers are pinned, pooled, and bounded by `max_inflight_bytes`.
    Device copies are asynchronous on the current stream, so returned tensors are
    ready for use on that stream; `close()` synchronizes and frees the buffers.
    """

    def __init__(
        self,
        device: torch.device | str,
        *,
        num_threads: int = _NUM_THREADS,
        max_inflight_bytes: int = _MAX_INFLIGHT_BYTES,
    ):
        self._device = torch.device(device)
        self._to_cuda = self._device.type == "cuda"
        self._max_inflight_bytes = max_inflight_bytes
        self._pool = ThreadPoolExecutor(num_threads, thread_name_prefix="pread")
        self._headers: dict[str, dict[str, _TensorInfo]] = {}
        self._fds: dict[str, int] = {}
        self._free_buffers: dict[int, list[torch.Tensor]] = {}
        self._pending: deque[tuple[torch.cuda.Event, torch.Tensor]] = deque()
        self._inflight_bytes = 0
        self.bytes_read = 0

    def _info(self, path: str, name: str) -> _TensorInfo:
        if path not in self._headers:
            self._headers[path] = parse_safetensors_header(path)
            self._fds[path] = os.open(path, os.O_RDONLY)
        return self._headers[path][name]

    def _acquire_buffer(self, nbytes: int) -> torch.Tensor:
        size = _round_buffer_bytes(nbytes)
        while self._pending and self._inflight_bytes + size > self._max_inflight_bytes:
            self._retire_oldest()
        free = self._free_buffers.get(size)
        if free:
            buf = free.pop()
        else:
            buf = torch.empty(size, dtype=torch.uint8, pin_memory=self._to_cuda)
        self._inflight_bytes += size
        return buf

    def _retire_oldest(self) -> None:
        event, buf = self._pending.popleft()
        event.synchronize()
        self._release_buffer(buf)

    def _release_buffer(self, buf: torch.Tensor) -> None:
        self._inflight_bytes -= buf.numel()
        self._free_buffers.setdefault(buf.numel(), []).append(buf)

    def _pread_into(self, fd: int, buf: torch.Tensor, begin: int, nbytes: int) -> None:
        view = memoryview(buf.numpy())
        futures = []
        for offset in range(0, nbytes, _CHUNK_BYTES):
            length = min(_CHUNK_BYTES, nbytes - offset)
            futures.append(
                self._pool.submit(
                    _pread_exact, fd, view[offset : offset + length], begin + offset
                )
            )
        for future in futures:
            future.result()
        self.bytes_read += nbytes

    def read(
        self,
        path: str,
        name: str,
        *,
        dim: int = 0,
        start: int | None = None,
        stop: int | None = None,
    ) -> torch.Tensor:
        """Tensor `name`, optionally narrowed to [start, stop) along `dim`."""
        info = self._info(path, name)
        shape = info.shape
        narrow = start is not None and (
            start != 0 or stop != (shape[dim] if shape else 0)
        )
        if narrow and (not shape or dim >= len(shape)):
            raise ValueError(f"cannot narrow dim {dim} of {name} with shape {shape}")
        if narrow and dim == 0:
            row_bytes = (info.end - info.begin) // shape[0] if shape[0] else 0
            begin = info.begin + start * row_bytes
            nbytes = (stop - start) * row_bytes
            read_shape = (stop - start, *shape[1:])
            tensor = self._read_range(path, info.dtype, read_shape, begin, nbytes)
            return tensor
        tensor = self._read_range(
            path, info.dtype, shape, info.begin, info.end - info.begin
        )
        if narrow:
            tensor = tensor.narrow(dim, start, stop - start).contiguous()
        return tensor

    def _read_range(
        self,
        path: str,
        dtype: torch.dtype,
        shape: tuple[int, ...],
        begin: int,
        nbytes: int,
    ) -> torch.Tensor:
        if nbytes == 0:
            return torch.empty(shape, dtype=dtype, device=self._device)
        buf = self._acquire_buffer(nbytes)
        self._pread_into(self._fds[path], buf, begin, nbytes)
        host = buf[:nbytes].view(dtype).reshape(shape)
        if not self._to_cuda:
            # Detach from the pooled buffer so it can be reused.
            out = host.clone()
            self._release_buffer(buf)
            return out.to(self._device)
        out = torch.empty(shape, dtype=dtype, device=self._device)
        out.copy_(host, non_blocking=True)
        event = torch.cuda.Event()
        event.record()
        self._pending.append((event, buf))
        return out

    def close(self) -> None:
        if self._to_cuda:
            torch.cuda.current_stream(self._device).synchronize()
        while self._pending:
            self._release_buffer(self._pending.popleft()[1])
        self._free_buffers.clear()
        self._pool.shutdown()
        for fd in self._fds.values():
            os.close(fd)
        self._fds.clear()

    def __enter__(self) -> "ParallelPreadReader":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


def _pread_exact(fd: int, view: memoryview, offset: int) -> None:
    done = 0
    total = len(view)
    while done < total:
        n = os.preadv(fd, [view[done:]], offset + done)
        if n <= 0:
            raise OSError(f"short read at offset {offset + done}")
        done += n


class ParallelSliceHandle:
    """Stands in for a `safe_open` handle: `get_tensor` and `get_slice(...)[...]`."""

    def __init__(self, path: str, reader: ParallelPreadReader):
        self._path = path
        self._reader = reader

    def get_tensor(self, name: str) -> torch.Tensor:
        return self._reader.read(self._path, name)

    def get_slice(self, name: str) -> "_ParallelSlice":
        return _ParallelSlice(self._reader, self._path, name)


class _ParallelSlice:
    def __init__(self, reader: ParallelPreadReader, path: str, name: str):
        self._reader = reader
        self._path = path
        self._name = name

    def __getitem__(self, slices: tuple[slice, ...]) -> torch.Tensor:
        shape = self._reader._info(self._path, self._name).shape
        bounds = [s.indices(n) for s, n in zip(slices, shape, strict=True)]
        partial = [
            d for d, (a, b, _) in enumerate(bounds) if (a, b) != (0, shape[d])
        ]
        if any(step != 1 for _, _, step in bounds):
            raise ValueError("strided slices are not supported")
        if len(partial) > 1:
            # Rare; read whole, then slice on the device.
            full = self._reader.read(self._path, self._name)
            return full[tuple(slice(a, b) for a, b, _ in bounds)].contiguous()
        if not partial:
            return self._reader.read(self._path, self._name)
        dim = partial[0]
        return self._reader.read(
            self._path, self._name, dim=dim, start=bounds[dim][0], stop=bounds[dim][1]
        )
