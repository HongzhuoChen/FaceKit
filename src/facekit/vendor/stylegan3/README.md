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

The modules must stay importable under their original top-level names
(`dnnlib`, `torch_utils`, ...) because trained network pickles refer to them
that way. `facekit.vendor.ensure_stylegan3_on_path()` puts this directory on
`sys.path`.
