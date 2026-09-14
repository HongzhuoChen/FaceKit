# Vendored StyleGAN3

Subset of https://github.com/NVlabs/stylegan3 (commit on `main`, copied
2026-09-13), licensed under the **NVIDIA Source Code License** in
`LICENSE.txt`, which permits non-commercial use only. This directory is not
covered by FaceKit's own license.

Kept: `dnnlib/`, `torch_utils/`, `training/`, `metrics/`, `legacy.py`,
`dataset_tool.py`, `train.py`, `gen_images.py`.
Dropped: `gui_utils/`, `viz/`, `visualizer.py`, `gen_video.py`,
`avg_spectra.py`, `docs/`, `Dockerfile`, `environment.yml`.

Local modifications (keep this list current):

1. `torch_utils/ops/conv2d_gradfix.py`, `torch_utils/ops/grid_sample_gradfix.py`:
   `from pkg_resources import parse_version` replaced by
   `from packaging.version import parse as parse_version`. `pkg_resources` is
   deprecated and scheduled for removal from setuptools.
2. `training/training_loop.py`, `train.py`: `--lr-schedule {constant,cosine}`
   (default `constant`, the original behaviour). `cosine` multiplies both
   optimizers' base learning rates by `0.5 * (1 + cos(pi * t / T))`, with `t`
   the images seen so far and `T = total_kimg`, updated every batch. This is
   the schedule the manuscript's generators were trained with; `facekit train`
   turns it on by default. Every edit is marked `FaceKit modification`.
3. `torch_utils/misc.py`: `InfiniteSampler.__init__` calls
   `super().__init__()` without the dataset; `torch.utils.data.Sampler` stopped
   accepting a `data_source` argument in PyTorch 2.2.

The modules must stay importable under their original top-level names
(`dnnlib`, `torch_utils`, ...) because trained network pickles refer to them
that way. `facekit.vendor.ensure_stylegan3_on_path()` puts this directory on
`sys.path`.
