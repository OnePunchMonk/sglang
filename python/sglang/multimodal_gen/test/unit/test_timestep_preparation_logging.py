import logging
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sglang.multimodal_gen.configs.pipeline_configs.flux import (
    Flux2KleinBasePipelineConfig,
)
from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams
from sglang.multimodal_gen.runtime.models.schedulers.scheduling_flow_match_euler_discrete import (
    FlowMatchEulerDiscreteScheduler,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages import (
    base,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages import (
    timestep_preparation as module,
)


@pytest.mark.parametrize("warmup", [False, True])
@pytest.mark.parametrize("debug", [False, True])
def test_timestep_logging_preserves_scheduler_and_skips_unused_copy(warmup, debug):
    args = SimpleNamespace(pipeline_config=Flux2KleinBasePipelineConfig())
    scheduler = FlowMatchEulerDiscreteScheduler()
    with patch.object(base, "get_global_server_args", return_value=args):
        stage = module.TimestepPreparationStage(scheduler)
    batch = Req(sampling_params=SamplingParams(num_inference_steps=4))
    batch.is_warmup = warmup
    records = []

    class Capture(logging.Handler):
        def emit(self, record):
            records.append(record)
            self.format(record)

    test_logger = logging.Logger(
        "timestep-test", logging.DEBUG if debug else logging.INFO
    )
    test_logger.addHandler(Capture())
    with (
        patch.object(module, "logger", test_logger),
        patch.object(
            module, "get_local_torch_device", return_value=torch.device("cpu")
        ),
    ):
        if debug and not warmup:
            result = stage.forward(batch, args)
        else:
            with patch.object(
                torch.Tensor, "detach", side_effect=AssertionError("unused log copy")
            ):
                result = stage.forward(batch, args)
    assert result is batch
    assert batch.scheduler is scheduler
    assert batch.timesteps is scheduler.timesteps
    assert len(records) == int(debug and not warmup)
    if records:
        value = records[0].args[-1]
        assert value.device.type == "cpu"
        assert torch.equal(value, batch.timesteps)
        assert "TimestepPreparationStage" in records[0].getMessage()


def _grouped_timestep_stage(prepare_extra_kwargs=None):
    args = SimpleNamespace(
        pipeline_config=Flux2KleinBasePipelineConfig(),
        comfyui_mode=True,
        enable_layerwise_nvtx_marker=False,
    )
    with patch.object(base, "get_global_server_args", return_value=args):
        stage = module.TimestepPreparationStage(
            FlowMatchEulerDiscreteScheduler(), prepare_extra_kwargs
        )
    return stage, args


def _run_grouped(stage, args, reqs):
    with patch.object(
        module, "get_local_torch_device", return_value=torch.device("cpu")
    ):
        return stage.run_grouped_requests(reqs, args)


def test_grouped_timesteps_keep_per_group_scheduler_state():
    stage, args = _grouped_timestep_stage()
    reqs = [
        Req(sampling_params=SamplingParams(num_inference_steps=steps))
        for steps in (4, 8)
    ]

    _run_grouped(stage, args, reqs)

    for req in reqs:
        assert len(req.scheduler.timesteps) == req.num_inference_steps
        assert torch.equal(req.scheduler.timesteps, req.timesteps)


def test_grouped_timesteps_fingerprint_covers_extra_kwargs_inputs():
    stage, args = _grouped_timestep_stage([lambda b, _: ("mu", b.extra["m"])])
    reqs = [Req(sampling_params=SamplingParams(num_inference_steps=4)) for _ in "ab"]
    reqs[0].extra["m"], reqs[1].extra["m"] = 0.5, 1.5

    _run_grouped(stage, args, reqs)

    assert [req.extra["mu"] for req in reqs] == [0.5, 1.5]
    rollout = Req(sampling_params=SamplingParams(num_inference_steps=4, rollout=True))
    rollout.extra["m"] = 0.5
    assert stage.build_dedup_fingerprint(
        rollout, args
    ) != stage.build_dedup_fingerprint(reqs[0], args)
