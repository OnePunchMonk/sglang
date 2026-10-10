# SPDX-License-Identifier: Apache-2.0
"""Wan pipelines for ComfyUI integrated mode.

ComfyUI keeps the sampler, text/image encoders and VAE, and sends one DiT
forward per step, so these pipelines only load the transformer(s). One class
per Wan variant because the pipeline config (task type, latent geometry) and
the checkpoint spec are selected by pipeline name.
"""

from sglang.multimodal_gen.configs.pipeline_configs.wan import (
    Wan2_2_TI2V_5B_Config,
    WanI2V720PConfig,
    WanT2V480PConfig,
)
from sglang.multimodal_gen.configs.sample.wan import (
    Wan2_2_TI2V_5B_SamplingParam,
    WanI2V_14B_720P_SamplingParam,
    WanT2V_14B_SamplingParams,
)
from sglang.multimodal_gen.runtime.loader.comfyui_checkpoints.wan import (
    WAN_I2V_PIPELINE,
    WAN_T2V_PIPELINE,
    WAN_TI2V_PIPELINE,
)
from sglang.multimodal_gen.runtime.pipelines.wan_pipeline import WanPipeline


class WanComfyUIT2VPipeline(WanPipeline):
    pipeline_name = WAN_T2V_PIPELINE
    pipeline_config_cls = WanT2V480PConfig
    sampling_params_cls = WanT2V_14B_SamplingParams


class WanComfyUII2VPipeline(WanPipeline):
    pipeline_name = WAN_I2V_PIPELINE
    pipeline_config_cls = WanI2V720PConfig
    sampling_params_cls = WanI2V_14B_720P_SamplingParam


class WanComfyUITI2VPipeline(WanPipeline):
    pipeline_name = WAN_TI2V_PIPELINE
    pipeline_config_cls = Wan2_2_TI2V_5B_Config
    sampling_params_cls = Wan2_2_TI2V_5B_SamplingParam


EntryClass = [WanComfyUIT2VPipeline, WanComfyUII2VPipeline, WanComfyUITI2VPipeline]
