# SPDX-License-Identifier: Apache-2.0
"""Server-mode ComfyUI video nodes: the returned video must be a local file."""

import os
import sys
import types
from unittest import mock

import pytest

VIDEO_BYTES = b"\x00\x00\x00\x18ftypmp42 fake video"


class _VideoFromFile:
    """Stands in for comfy_api.input_impl.VideoFromFile."""

    def __init__(self, file):
        self.file = file


def _load(temp_dir):
    """Import the real nodes.py with the ComfyUI modules it needs stubbed."""
    folder_paths = types.ModuleType("folder_paths")
    folder_paths.folder_names_and_paths = {}
    folder_paths.get_temp_directory = lambda: str(temp_dir)
    comfy_api = types.ModuleType("comfy_api")
    comfy_api_input = types.ModuleType("comfy_api.input")
    comfy_api_input.VideoInput = type("VideoInput", (), {})
    comfy_api_input_impl = types.ModuleType("comfy_api.input_impl")
    comfy_api_input_impl.VideoFromFile = _VideoFromFile
    comfy_api.input = comfy_api_input
    comfy_api.input_impl = comfy_api_input_impl
    stubs = {
        "folder_paths": folder_paths,
        "comfy_api": comfy_api,
        "comfy_api.input": comfy_api_input,
        "comfy_api.input_impl": comfy_api_input_impl,
    }
    import sglang.multimodal_gen.apps.ComfyUI_SGLDiffusion as plugin

    with mock.patch.dict(sys.modules, stubs):
        # Fresh import per test so nodes/utils bind this test's stubs.
        for name in ("nodes", "utils"):
            sys.modules.pop(f"{plugin.__name__}.{name}", None)
            plugin.__dict__.pop(name, None)
        from sglang.multimodal_gen.apps.ComfyUI_SGLDiffusion import nodes
        from sglang.multimodal_gen.apps.ComfyUI_SGLDiffusion.core.server_api import (
            SGLDiffusionServerAPI,
        )
    return nodes, SGLDiffusionServerAPI


class _Download:
    def __init__(self, body=VIDEO_BYTES, status=200, drop_midway=False):
        self.body, self.status, self.drop_midway = body, status, drop_midway

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def raise_for_status(self):
        if self.status >= 400:
            import requests

            raise requests.exceptions.HTTPError(f"{self.status}")

    def iter_content(self, chunk_size=None):
        yield self.body
        if self.drop_midway:
            import requests

            raise requests.exceptions.ChunkedEncodingError("connection dropped")


def _run_video_node(tmp_path, job, gets):
    """Run SGLDiffusionGenerateVideo against a server that returns ``job``."""
    nodes, api_cls = _load(tmp_path / "comfy_temp")
    os.makedirs(tmp_path / "comfy_temp", exist_ok=True)
    client = api_cls(base_url="http://sgld-host:30010")
    client.generate_video = lambda **params: dict(job)
    requested = []

    def fake_get(url, headers=None, stream=False, timeout=None):
        requested.append((url, headers))
        return gets.get(url, _Download(status=404))

    node = nodes.SGLDiffusionGenerateVideo()
    with mock.patch("requests.get", side_effect=fake_get):
        video, video_path = getattr(node, node.FUNCTION)(
            client, positive_prompt="a fox", width=64, height=64
        )
    return video, video_path, requested, client


def _assert_local_video(video, video_path):
    assert video_path and os.path.isfile(video_path)
    with open(video_path, "rb") as f:
        assert f.read() == VIDEO_BYTES
    assert isinstance(video, _VideoFromFile) and video.file == video_path


def test_local_server_output_is_used_directly(tmp_path) -> None:
    local = tmp_path / "out.mp4"
    local.write_bytes(VIDEO_BYTES)
    job = {"id": "v0", "status": "completed", "file_path": str(local)}
    video, video_path, requested, _ = _run_video_node(tmp_path, job, gets={})
    assert video_path == str(local)
    assert requested == []
    _assert_local_video(video, video_path)


def test_cloud_storage_output_is_downloaded_without_api_key(tmp_path) -> None:
    # /v1/videos sets file_path=None once the video is uploaded to cloud storage.
    url = "https://bucket.example/v1.mp4"
    job = {"id": "v1", "status": "completed", "file_path": None, "url": url}
    video, video_path, requested, _ = _run_video_node(
        tmp_path, job, gets={url: _Download()}
    )
    _assert_local_video(video, video_path)
    assert requested == [(url, None)]  # never send the server's API key elsewhere


def test_remote_server_output_is_fetched_from_content_endpoint(tmp_path) -> None:
    # file_path is a path on the server host, not on the ComfyUI machine.
    content = "http://sgld-host:30010/v1/videos/v2/content"
    job = {"id": "v2", "status": "completed", "file_path": "/srv/outputs/v2.mp4"}
    video, video_path, requested, client = _run_video_node(
        tmp_path, job, gets={content: _Download()}
    )
    _assert_local_video(video, video_path)
    assert requested == [(content, client.headers)]


def test_download_name_ignores_server_job_id(tmp_path) -> None:
    # The job id comes from the server; it must not choose the local path.
    url = "https://bucket.example/v4.mp4"
    job = {"id": "../../escape", "status": "completed", "file_path": None, "url": url}
    _, video_path, _, _ = _run_video_node(tmp_path, job, gets={url: _Download()})
    temp_dir = os.path.realpath(tmp_path / "comfy_temp")
    assert os.path.dirname(os.path.realpath(video_path)) == temp_dir
    assert "escape" not in os.path.basename(video_path)


def test_failed_download_leaves_no_partial_file(tmp_path) -> None:
    url = "https://bucket.example/v5.mp4"
    job = {"id": "v5", "status": "completed", "file_path": None, "url": url}
    with pytest.raises(RuntimeError, match="v5"):
        _run_video_node(tmp_path, job, gets={url: _Download(drop_midway=True)})
    assert os.listdir(tmp_path / "comfy_temp") == []


def test_missing_video_fails_in_the_generate_node(tmp_path) -> None:
    job = {"id": "v3", "status": "completed", "file_path": None}
    with pytest.raises(RuntimeError, match="v3"):
        _run_video_node(tmp_path, job, gets={})
