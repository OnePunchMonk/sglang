# SPDX-License-Identifier: Apache-2.0
import copy
import os

from sglang.multimodal_gen.configs.pipeline_configs.qwen_image21 import (
    QwenImage21PipelineConfig,
)
from sglang.multimodal_gen.configs.sample.qwenimage21 import QwenImage21SamplingParams
from sglang.multimodal_gen.runtime.disaggregation.roles import RoleType
from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
from sglang.multimodal_gen.runtime.models.dits.qwen_image21 import (
    qwen_image21_dit_jit_prebuilds,
)
from sglang.multimodal_gen.runtime.pipelines_core import LoRAPipeline
from sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base import (
    ComposedPipelineBase,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages import (
    ComfyUILatentPreparationStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.qwen_image21 import (
    QwenImage21ComfyUIConditionStage,
    QwenImage21DenoisingStage,
    QwenImage21EncodingStage,
    QwenImage21InputValidationStage,
    prepare_qwen21_mu,
)
from sglang.multimodal_gen.runtime.utils.hf_diffusers_utils import (
    get_diffusers_component_config,
)
from sglang.multimodal_gen.runtime.utils.jit_prebuild import (
    prebuild_jit_modules_in_background,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.runtime.utils.precision import resolve_component_precision

logger = init_logger(__name__)


class QwenImage21Pipeline(LoRAPipeline, ComposedPipelineBase):
    pipeline_name = "QwenImage21Pipeline"
    # Config for a ComfyUI single-file DiT, which has no model_index.json.
    pipeline_config_cls = QwenImage21PipelineConfig
    sampling_params_cls = QwenImage21SamplingParams
    _required_config_modules = [
        "processor",
        "text_encoder",
        "transformer",
        "vae",
        "scheduler",
    ]

    def _load_config(self):
        model_index = super()._load_config()
        sample_sigmas = model_index.pop("sample_sigmas", None)
        if self.server_args.pipeline_config.sample_sigmas is None:
            self.server_args.pipeline_config.sample_sigmas = sample_sigmas
        return model_index

    def load_modules(self, server_args, loaded_modules=None):
        # nvcc runs on the host while the weights load; warmup then finds the kernels built
        try:
            builders = self._jit_prebuilds(server_args)
        except Exception as e:
            logger.warning("Skipping the Qwen-Image 2.1 JIT prebuild: %s", e)
            builders = []
        prebuild_jit_modules_in_background("Qwen-Image 2.1", builders)
        return super().load_modules(server_args, loaded_modules)

    def _jit_prebuilds(self, server_args):
        builders = []
        if "transformer" in self.required_config_modules:
            dit_config = server_args.pipeline_config.dit_config
            builders += qwen_image21_dit_jit_prebuilds(
                device=get_local_torch_device(),
                dtype=resolve_component_precision(server_args, "transformer"),
                head_dim=dit_config.arch_config.attention_head_dim,
            )
        # a ComfyUI single-file model has no vae/config.json to read
        if "vae" in self.required_config_modules and not server_args.comfyui_mode:
            from sglang.multimodal_gen.runtime.models.vaes.qwen_image21_vae_cuda_opt import (
                qwen_image21_vae_jit_prebuilds,
            )

            vae_path = server_args.component_paths.get("vae") or os.path.join(
                self.model_path, "vae"
            )
            # the loader applies the checkpoint's arch later; a private copy sees it now
            vae_config = copy.deepcopy(server_args.pipeline_config.vae_config)
            vae_arch = get_diffusers_component_config(component_path=vae_path)
            vae_arch.pop("_class_name", None)
            vae_arch.pop("_diffusers_version", None)
            vae_config.update_model_arch(vae_arch)
            vae_config.post_init()
            builders += qwen_image21_vae_jit_prebuilds(vae_config)
        return builders

    def create_pipeline_stages(self, server_args):
        self.add_stage(QwenImage21InputValidationStage())
        self.add_stage_factory(
            RoleType.ENCODER,
            lambda: QwenImage21EncodingStage(
                self.get_module("text_encoder"),
                self.get_module("processor"),
                self.get_module("vae"),
                self.get_module("scheduler"),
            ),
            "conditioning_stage",
        )
        self.add_standard_latent_preparation_stage()
        self.add_standard_timestep_preparation_stage(
            prepare_extra_kwargs=[prepare_qwen21_mu]
        )
        self.add_stage_factory(
            RoleType.DENOISER,
            lambda: QwenImage21DenoisingStage(
                transformer=self.get_module("transformer"),
                scheduler=self.get_module("scheduler"),
            ),
            "denoising_stage",
        )
        self.add_standard_decoding_stage()

    def create_comfyui_stages(self, server_args):
        if server_args.enable_cfg_parallel:
            # ComfyUI owns CFG; each CFG rank would recompute the same DiT call.
            raise ValueError(
                "enable_cfg_parallel is not supported for Qwen-Image 2.1 in "
                "--comfyui-mode; use sp_degree or tp_size"
            )
        transformer = self.get_module("transformer")
        scheduler = self.get_module("scheduler")
        self.add_stages(
            [
                ComfyUILatentPreparationStage(scheduler, transformer),
                QwenImage21ComfyUIConditionStage(),
                QwenImage21DenoisingStage(transformer, scheduler),
            ]
        )


EntryClass = QwenImage21Pipeline
