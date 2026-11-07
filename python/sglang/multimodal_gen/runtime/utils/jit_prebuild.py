# SPDX-License-Identifier: Apache-2.0
"""Build a pipeline's JIT kernels in the background while its weights load.

A JIT kernel variant is otherwise compiled the first time a forward reaches it,
which for a cold cache means one nvcc run per variant, back to back, inside the
server warmup (Qwen-Image 2.1: 11 variants at ~7 s each on an H100 host). The
builders run concurrently on a side thread instead, so the compiles overlap
each other and the weight load. A forward that reaches a variant still being
built waits on that variant's build lock and then loads the published module;
a builder that fails only leaves its variant to the usual compile on first use.
"""

import threading
import time
from typing import Any, Callable, Iterable

from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


def prebuild_jit_modules_in_background(
    label: str, builders: Iterable[Callable[[], Any]]
) -> threading.Thread | None:
    """Start building ``builders`` on a daemon thread; return it, or None if there is nothing to build."""
    builders = list(builders)
    if not builders:
        return None

    def run() -> None:
        from sglang.kernels.jit.utils import parallel_compile

        start = time.perf_counter()
        try:
            parallel_compile(builders)
        except Exception as e:
            logger.warning(
                "Background JIT prebuild for %s failed; the remaining kernels "
                "compile on first use: %s",
                label,
                e,
            )
            return
        logger.info(
            "Prebuilt %d JIT kernel variants for %s in %.1fs",
            len(builders),
            label,
            time.perf_counter() - start,
        )

    thread = threading.Thread(target=run, name="jit-prebuild", daemon=True)
    thread.start()
    return thread
