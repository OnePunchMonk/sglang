# PR 44105 end-to-end ComfyUI check (Z-Image-Turbo, native batching)

- `modal_comfyui_e2e.py`: the Modal harness used (set `WT` to a checkout of the PR branch). It installs ComfyUI, copies the plugin into `custom_nodes`, downloads the Comfy-Org/z_image_turbo files, starts ComfyUI headless and submits the workflow over `/prompt`.
- `run2_zeroed_negative_api.json`: API workflow for run 2 (this is what the script builds). Only node `900` (`SGLDOptions.native_batching`) changes between runs: `off`, `cfg_only`, `full`. Warmup run used `seed=1`, measured runs `seed=1234`.
- `run1_shipped_negative_api.json`: run 2 without node `901` (the shipped "blurry ugly bad" negative). Run 1 used `seed=1234` for the warmup too, so its repeated `off` runs came from ComfyUI's cache.
- `*_sheet.png`: rows are `off`, `cfg_only`, `full`; columns are image 0 and 1 of the batch, resized to 384x384.
