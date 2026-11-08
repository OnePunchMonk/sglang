# SPDX-License-Identifier: Apache-2.0
"""Early validation for unsupported ComfyUI model patches."""

import sys
from types import ModuleType

import pytest


def _load_model_patcher(monkeypatch):
    """Load the patcher without requiring a full ComfyUI checkout."""
    comfy = ModuleType("comfy")
    comfy.__path__ = []
    model_patcher = ModuleType("comfy.model_patcher")

    class ModelPatcher:
        pass

    model_patcher.ModelPatcher = ModelPatcher
    monkeypatch.setitem(sys.modules, "comfy", comfy)
    monkeypatch.setitem(sys.modules, "comfy.model_patcher", model_patcher)
    sys.modules.pop(
        "sglang.multimodal_gen.apps.ComfyUI_SGLDiffusion.core.model_patcher", None
    )
    from sglang.multimodal_gen.apps.ComfyUI_SGLDiffusion.core.model_patcher import (
        SGLDModelPatcher,
    )
    monkeypatch.setitem(
        sys.modules,
        "sglang.multimodal_gen.apps.ComfyUI_SGLDiffusion.core.model_patcher",
        sys.modules[
            "sglang.multimodal_gen.apps.ComfyUI_SGLDiffusion.core.model_patcher"
        ],
    )

    return SGLDModelPatcher


def test_patch_model_rejects_controlnet_before_comfyui_applies_it(monkeypatch):
    SGLDModelPatcher = _load_model_patcher(monkeypatch)
    patcher = SGLDModelPatcher.__new__(SGLDModelPatcher)
    patcher.model_options = {
        "transformer_options": {"patches_replace": {"dit": {"block": object()}}}
    }

    with pytest.raises(ValueError, match="cannot apply ComfyUI model patches"):
        patcher.patch_model()


def test_patch_model_keeps_noop_for_unpatched_model(monkeypatch):
    SGLDModelPatcher = _load_model_patcher(monkeypatch)
    patcher = SGLDModelPatcher.__new__(SGLDModelPatcher)
    patcher.model_options = {"transformer_options": {}}

    assert patcher.patch_model() is None
