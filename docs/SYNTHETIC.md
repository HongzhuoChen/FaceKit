# Synthetic faces: `enhance`, `pack`, `train`, `generate`

This document records what each command in the generation pipeline does,
every default and where it comes from, and what changed in the vendored
StyleGAN3. The README gives the overview; this is the reference.

## Pipeline

```
raw/<cohort>/*.jpg
  │  facekit enhance      (optional: colorize grayscale, restore small faces)
  ▼
prepared/<cohort>/*.png
  │  facekit pack         (face crop -> 256x256 -> StyleGAN3 zip)
  ▼
datasets/<cohort>/*.png  +  datasets/<cohort>.zip
  │  facekit train        (one generator per cohort, GPU)
  ▼
runs/<cohort>/<NNNNN>-stylegan3-t-.../network-snapshot-<kimg>.pkl
  │  facekit generate
  ▼
synthetic/<cohort>/seedNNNN.png
```

Split the patients of each cohort into a training and a held-out partition
**before** `pack`, and pack only the training partition. The held-out crops
are what `facekit privacy` compares against later; without them the audit
has no calibration set.

## `enhance`

Two enhancement steps, applied only where a test says they are needed,
colorization first (GFPGAN was trained on colour faces).

| Test | Default | Action |
| --- | --- | --- |
| mean HSV saturation < `--saturation` (0.03) or single-channel file | grayscale | DDColor colorization |
| shorter side of the MediaPipe landmark box < `--min-face` (128 px) | small face | GFPGAN v1.4 restoration, output `--upscale` (2) times the input |

`--force-colorize` / `--force-restore` skip the tests; `--no-colorize` /
`--no-restore` skip the steps. Faded or yellowed prints have low saturation
and are recolorized; that is intended. The face-size test uses the face box
rather than the image size because GFPGAN restores faces: a large group
photograph with an 80-pixel face is the case that needs it.

Outputs mirror the input layout, every image as PNG whether processed or
not, plus `enhance_log.csv`:

| column | meaning |
| --- | --- |
| `cohort`, `image` | input folder and file name |
| `grayscale` | result of the saturation test (`True`/`False`; empty if unreadable) |
| `face_px` | shorter side of the landmark box in pixels (empty if no face) |
| `colorized`, `restored` | whether each step ran |
| `status` | `ok` or `unreadable` |

Models: DDColor-L (`piddnad/ddcolor_modelscope` on Hugging Face; `--ddcolor-size
tiny` selects `ddcolor_paper_tiny`), GFPGAN v1.4 from the GFPGAN GitHub
release, facexlib's RetinaFace-ResNet50 detector and ParseNet for alignment
and paste-back. All download on first use to `~/.cache/facekit`
(`FACEKIT_CACHE_DIR`). The DDColor model code and GFPGAN's clean
architecture are vendored under `src/facekit/vendor/` because the pip
packages depend on `basicsr`, which does not install on Python 3.13 and
whose DDColor fork collides with GFPGAN's.

## `pack`

For each image the MediaPipe landmark bounding box is made square, widened
by `--margin` (0.3, i.e. 30 % of the box on every side), clamped into the
image with edge replication where it overhangs, and resized to
`--resolution` (256; StyleGAN3 needs a power of two). Crops are written to
`<out>/<cohort>/<stem>.png`, then StyleGAN3's `dataset_tool.py` packs them
into `<out>/<cohort>.zip` (uncompressed PNGs plus `dataset.json`).
`pack_log.csv` lists every input with status `ok`, `no_face` or
`unreadable`.

The crop folder is kept on purpose: it is the "prepared real images" that
`extract-features` and `privacy` read, with the same framing as the
synthetic output.

## `train`

A thin wrapper around the vendored `train.py`. Defaults are the settings the
FaceKit generators were trained with, as reported in the manuscript:

| option | default | note |
| --- | --- | --- |
| `--cfg` | `stylegan3-t` | StyleGAN3 translational-equivariant config |
| `--gamma` | 2 | R1 regularization weight |
| `--mirror` | on | horizontal-flip augmentation |
| `--kimg` | 5000 | training length, thousands of real images shown |
| `--glr` / `--dlr` | 0.0025 / 0.002 | generator / discriminator learning rates |
| `--lr-schedule` | `cosine` | both rates decay as $\tfrac{1}{2}(1+\cos(\pi t/T))$ over `--kimg`; `constant` is stock StyleGAN3 |
| `--map-depth` | 8 | mapping-network layers (stock stylegan3-t uses 2) |
| `--batch` / `--batch-gpu` | 32 / batch÷gpus | images per step; the remainder of `--batch` is accumulated |
| `--snap` | 50 ticks (200 kimg) | snapshot interval; each snapshot is about 350 MB |
| `--metrics` | `none` | `fid50k_full` is meaningless on cohorts of ~100 images |
| `--extra "..."` | | any native `train.py` option, passed through verbatim |

`--dry-run` prints the resolved StyleGAN3 options without a GPU. Runs are
numbered by StyleGAN3 (`00000-…`, `00001-…`). Adaptive discriminator
augmentation (ADA, target 0.6) is on, as in stock StyleGAN3.

Measured on one NVIDIA B200 MIG slice (45 GB, 6 CPU threads): 85 s per
kimg at 256 × 256 with batch 32, 5.7 GB of GPU memory. A 5,000 kimg run is
therefore about five days on such a slice; use a full GPU for real training.

## `generate`

Loads `G_ema` from a network pickle and writes one PNG per seed to
`<out>/<name>/seedNNNN.png` (`--name` defaults to the pickle's file stem).
Latents are drawn with `numpy.random.RandomState(seed)`, exactly as in
StyleGAN3's `gen_images.py`, so a seed names an image independent of device
or batch. `--n N` uses seeds `--first-seed … --first-seed+N-1`; `--seeds
0,7,40-49` gives an explicit list. `--trunc` is the truncation ψ (1.0 = none).
Conditional generators need `--class-idx`. Runs on CPU (about 40 s per image
at 256 × 256) when no GPU is present.

## Vendored StyleGAN3

`src/facekit/vendor/stylegan3/` is a subset of
[NVlabs/stylegan3](https://github.com/NVlabs/stylegan3) under the NVIDIA
Source Code License (non-commercial). Its modules keep their original
top-level names (`dnnlib`, `torch_utils`, `training`, `legacy`) because
network pickles refer to them that way; `facekit.vendor.ensure_stylegan3_on_path()`
puts the directory on `sys.path`.

Six local modifications, each marked `FaceKit modification` in the source
and listed in `src/facekit/vendor/stylegan3/README.md`:

1. `pkg_resources.parse_version` → `packaging.version.parse` (deprecated import).
2. `--lr-schedule {constant,cosine}` in `train.py` / `training_loop.py`.
3. `InfiniteSampler.__init__` no longer passes the dataset to
   `torch.utils.data.Sampler` (removed in PyTorch 2.2).
4. `custom_ops.get_plugin` uses the module returned by
   `torch.utils.cpp_extension.load()`; PyTorch 2.x no longer registers it in
   `sys.modules`.
5. Adam `betas=[0.0, 0.99]` (recent PyTorch rejects a mixed int/float pair).
6. A failed custom-op build now warns and falls back to the reference
   implementation instead of aborting.

Modifications 3 to 6 are what a 2021 codebase needs to run on PyTorch 2.x;
none depends on the CUDA version. `tests/test_stylegan3_compat.py` replays
one full single-GPU training iteration on CPU (dataset, sampler, networks,
augmentation, every loss phase, optimizers, EMA, snapshot pickle and reload)
so that a future PyTorch change is caught without a GPU.
