# SPDX-License-Identifier: Apache-2.0

"""Import-safe specifications and entry points for diffusion child processes.

``multiprocessing`` unpickles a target's arguments before calling the target.
Runtime objects therefore cannot cross this boundary directly: merely passing
``ServerArgs`` used to import the diffusion configuration graph before worker
bootstrap began. ``ServerArgsPayload`` keeps that object graph opaque until the
child reaches the explicit runtime-activation phase.

``spawn`` re-executes the launching script's module scope earlier still, before
any argument is unpickled, so an offline script may bind only the
``DiffGenerator`` proxy and ``_PRE_ACTIVATION_MODULES`` at module scope; every
other diffusion import belongs inside its ``if __name__ == "__main__":`` guard.
A violation is reported rather than silently tolerated.

Keep module scope limited to the standard library and import-neutral types.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import os
import pickle
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from multiprocessing.connection import Connection
    from multiprocessing.process import BaseProcess

    from sglang.multimodal_gen.runtime.server_args import ServerArgs

_DIFFUSION_PREFIX = "sglang.multimodal_gen."
_RUNTIME_NAMESPACES = (
    "sglang.multimodal_gen.runtime",
    "sglang.multimodal_gen.runtime.managers",
)
# What may legitimately be imported this early: bootstrap's own imports, and the
# platform and plugin modules that every plugin loads and the contract keeps
# import-safe. Listing these rather than their complement keeps the check
# complete as subpackages are added.
_PRE_ACTIVATION_MODULES = (
    "sglang.multimodal_gen.envs",
    "sglang.multimodal_gen.runtime.platforms",
    "sglang.multimodal_gen.runtime.utils",
    "sglang.multimodal_gen.runtime.managers.worker_bootstrap",
)
_MAX_REPORTED_MODULES = 5


def _warn_if_runtime_imported_early() -> None:
    """Name the modules that this child imported ahead of its own lifecycle."""
    early = sorted(
        name
        for name in list(sys.modules)
        if name.startswith(_DIFFUSION_PREFIX)
        and name not in _RUNTIME_NAMESPACES
        and not name.startswith(_PRE_ACTIVATION_MODULES)
    )
    if not early:
        return

    listed = ", ".join(early[:_MAX_REPORTED_MODULES])
    if len(early) > _MAX_REPORTED_MODULES:
        listed += f" (+{len(early) - _MAX_REPORTED_MODULES} more)"
    # In a spawned child __main__ is the re-executed launching script, which is
    # the file whose imports have to move.
    script = vars(sys.modules["__main__"]).get("__file__", "the launching script")
    logging.getLogger(__name__).warning(
        "Diffusion runtime modules were imported before this worker initialized "
        "its platform: %s. spawn re-executes %s at module scope in every child, "
        "so these were built ahead of platform initialization and hook "
        "application, and the classes and registrations they created are "
        'already past reach. Move the import inside if __name__ == "__main__": '
        "or into the function that uses it.",
        listed,
        script,
    )


@dataclass(frozen=True, slots=True)
class ServerArgsPayload:
    """A deferred ``ServerArgs`` snapshot safe to unpickle before bootstrap."""

    _pickle: bytes = field(repr=False)

    @classmethod
    def capture(cls, server_args: ServerArgs) -> ServerArgsPayload:
        return cls(pickle.dumps(server_args, protocol=pickle.HIGHEST_PROTOCOL))

    def materialize(self) -> ServerArgs:
        # Importing ServerArgs pulls in pipeline configuration modules. This
        # method must only be called after the process lifecycle is initialized.
        from sglang.multimodal_gen.runtime.server_args import ServerArgs

        server_args = pickle.loads(self._pickle)
        if not isinstance(server_args, ServerArgs):
            raise TypeError("Bootstrap payload did not contain diffusion ServerArgs")
        return server_args


@dataclass(frozen=True, slots=True)
class SchedulerProcessSpec:
    """Everything a scheduler child needs, without eagerly importing runtime state."""

    local_rank: int
    rank: int
    server_args: ServerArgsPayload
    pipe_writer: Connection


def _initialize_runtime(rank: int) -> None:
    """Arm the child and bring up platform and plugins, in dependency order."""
    # Arm PDEATHSIG before any vendor code runs: everything below can block,
    # and a child that hangs there would outlive a dead launcher.
    from sglang.multimodal_gen.runtime.utils.process import (
        kill_itself_when_parent_died,
    )

    kill_itself_when_parent_died()

    # Every rank re-executes the same script, so one rank reporting is enough.
    if rank == 0:
        _warn_if_runtime_imported_early()

    # Platform initialization is the first extensible runtime action. In
    # particular it precedes plugin callbacks and hook target resolution, both
    # of which may import arbitrary runtime modules.
    from sglang.multimodal_gen.runtime.platforms import initialize_current_platform

    initialize_current_platform()

    from sglang.multimodal_gen.runtime.platforms.plugins import (
        apply_plugin_hooks,
        load_plugins,
    )

    load_plugins()
    apply_plugin_hooks()


def _run_scheduler(spec: SchedulerProcessSpec) -> None:
    """Materialize the arguments and hand the child to its worker."""
    server_args = spec.server_args.materialize()

    # Resolve the function from its module after hook application. A ``from``
    # binding created earlier would retain the unpatched callable.
    from sglang.multimodal_gen.runtime.managers import gpu_worker

    gpu_worker.run_scheduler_process(
        local_rank=spec.local_rank,
        rank=spec.rank,
        server_args=server_args,
        pipe_writer=spec.pipe_writer,
    )


def bootstrap_scheduler_process(spec: SchedulerProcessSpec) -> None:
    """Initialize a child in dependency order, then invoke its worker."""
    _initialize_runtime(spec.rank)
    _run_scheduler(spec)


# Variables the launcher may set after pre-starting workers and that workers
# read only after activation (NCCL at communicator init, the CUDA allocator at
# first allocation). Any other change cannot be replayed into a process that
# has already imported torch, so the pre-started workers are discarded.
_ACTIVATION_ENV_PREFIXES = ("NCCL_", "PYTORCH_CUDA_ALLOC_CONF")


def _environ_changes(before: dict[str, str]) -> dict[str, str | None]:
    """Variables added, changed (new value) or removed (None) since ``before``."""
    now = dict(os.environ)
    changes: dict[str, str | None] = {
        name: value for name, value in now.items() if before.get(name) != value
    }
    changes.update({name: None for name in before if name not in now})
    return changes


def _prestarted_scheduler_main(
    spec_reader: Connection,
    rank: int,
    target: Callable[[SchedulerProcessSpec], None] | None,
) -> None:
    _initialize_runtime(rank)

    # The expensive import of a scheduler child, done before its spec exists.
    from sglang.multimodal_gen.runtime.managers import gpu_worker  # noqa: F401

    # materialize() imports this after activation; plugin hooks are applied.
    from sglang.multimodal_gen.runtime.server_args import ServerArgs  # noqa: F401

    import torch

    if torch.cuda.is_initialized():
        raise RuntimeError(
            "A pre-started scheduler worker touched CUDA before activation; "
            "the launcher's environment can no longer reach it."
        )
    spec, environ_changes = spec_reader.recv()
    spec_reader.close()
    for name, value in environ_changes.items():
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = value
    (target or _run_scheduler)(spec)


class PrestartedWorkers:
    """Scheduler children that finished importing and are waiting for a spec."""

    def __init__(
        self,
        processes: list[BaseProcess],
        spec_writers: list[Connection],
        rank_offset: int,
        environ_snapshot: dict[str, str],
    ):
        self.processes = processes
        self._spec_writers = spec_writers
        self.rank_offset = rank_offset
        self._environ_snapshot = environ_snapshot

    @property
    def num_workers(self) -> int:
        return len(self.processes)

    def can_activate(self, *, num_workers: int, rank_offset: int) -> bool:
        if num_workers != self.num_workers or rank_offset != self.rank_offset:
            return False
        unsafe = [
            name
            for name in _environ_changes(self._environ_snapshot)
            if not name.startswith(_ACTIVATION_ENV_PREFIXES)
        ]
        if unsafe:
            logging.getLogger(__name__).warning(
                "Not using pre-started workers: environment changed after they "
                "started in ways they cannot pick up: %s",
                ", ".join(sorted(unsafe)),
            )
        return not unsafe

    def activate(
        self,
        *,
        server_args: ServerArgsPayload,
        pipe_writers: list[Connection],
        local_ranks: list[int],
    ) -> list[BaseProcess]:
        """Send each worker its spec; ``pipe_writers[i]`` reports worker ``i``."""
        changes = _environ_changes(self._environ_snapshot)
        for i, writer in enumerate(self._spec_writers):
            spec = SchedulerProcessSpec(
                local_rank=local_ranks[i],
                rank=self.rank_offset + i,
                server_args=server_args,
                pipe_writer=pipe_writers[i],
            )
            writer.send((spec, changes))
            writer.close()
        return self.processes

    def terminate(self) -> None:
        for process in self.processes:
            if process.is_alive():
                process.terminate()
        for process in self.processes:
            process.join(timeout=10)


def prestart_scheduler_workers(
    num_local_workers: int,
    rank_offset: int,
    target: Callable[[SchedulerProcessSpec], None] | None = None,
) -> PrestartedWorkers:
    """Spawn scheduler children now so their imports overlap the launcher's own.

    Call this before importing anything heavy, then pass the result to
    ``launch_server(..., prestarted=...)``. ``target`` replaces the final step
    (materialize ``ServerArgs`` and run the worker) and receives the spec; the
    platform, plugin and ``gpu_worker`` initialization has already run by then.
    """
    context = mp.get_context("spawn")
    snapshot = dict(os.environ)
    processes = []
    spec_writers = []
    for i in range(num_local_workers):
        rank = rank_offset + i
        reader, writer = context.Pipe(duplex=False)
        process = context.Process(
            target=_prestarted_scheduler_main,
            args=(reader, rank, target),
            name=f"sglang-diffusionWorker-{rank}",
            daemon=True,
        )
        process.start()
        reader.close()
        processes.append(process)
        spec_writers.append(writer)
    return PrestartedWorkers(processes, spec_writers, rank_offset, snapshot)


def bootstrap_http_server_process(server_args: ServerArgsPayload) -> None:
    from sglang.multimodal_gen.runtime.utils.process import (
        kill_itself_when_parent_died,
    )

    kill_itself_when_parent_died()

    _warn_if_runtime_imported_early()

    # No initialize_current_platform() here: this child serves HTTP and never
    # touches the device, so it has no reason to bring up a vendor backend.
    from sglang.multimodal_gen.runtime.platforms.plugins import (
        apply_plugin_hooks,
        load_plugins,
    )

    load_plugins()
    apply_plugin_hooks()

    from sglang.multimodal_gen.runtime import launch_server

    launch_server.launch_http_server_only(server_args.materialize())
