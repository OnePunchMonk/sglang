# SPDX-License-Identifier: Apache-2.0
"""Threaded `pread` into pinned buffers, then an async copy to the GPU.

Faster than `safe_open` for a local checkpoint headed to the GPU, but the tensors
it yields are anonymous device memory with no file mapping behind them, so a
host-bound load keeps the mmap reader (see weight_utils).
"""

import time
from typing import Callable, ClassVar, Iterator

import torch
from tqdm.auto import tqdm

from sglang.multimodal_gen.runtime.loader.parallel_safetensors import (
    ParallelPreadReader,
    parse_safetensors_header,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)

_BAR_FORMAT = "{desc}: {percentage:.0f}%|{bar}| {n_fmt}/{total_fmt}"


class ParallelPreadWeightReader:
    name: ClassVar[str] = "parallel_pread"
    supports_key_filter: ClassVar[bool] = True
    retains_file_mapping: ClassVar[bool] = False

    @classmethod
    def is_available(cls) -> bool:
        return torch.cuda.is_available()

    def iter_weights(
        self,
        files: list[str],
        *,
        device: str,
        to_cpu: bool,
        key_filter: Callable[[str], bool] | None = None,
        clone_tensors: bool = True,
        show_progress: bool = True,
    ) -> Iterator[tuple[str, torch.Tensor]]:
        start_time = time.perf_counter()
        with ParallelPreadReader(device) as reader:
            for path in tqdm(
                files,
                desc="Loading safetensors checkpoint shards",
                disable=not show_progress,
                bar_format=_BAR_FORMAT,
            ):
                for name in parse_safetensors_header(path):
                    if key_filter is not None and not key_filter(name):
                        continue
                    yield name, reader.read(path, name)
            elapsed = time.perf_counter() - start_time
            logger.info(
                "Parallel pread: read=%.2f GiB in %.1fs (%.2f GB/s)",
                reader.bytes_read / (1024**3),
                elapsed,
                reader.bytes_read / 1e9 / max(elapsed, 1e-9),
            )
