# FaceKit

Rare-disease facial phenotype analysis toolkit.

FaceKit takes face images of patients with rare genetic syndromes and produces:

1. **MediaPipe landmarks** (478 3D points, optional blendshapes / head pose).
2. **Average face** images per cohort.
3. **Geometric phenotype features** (~120 columns derived from the landmarks)
   suitable for downstream classification or HPO-aligned reporting.
4. **Synthetic faces** from cohort-specific StyleGAN3 generators, with the
   image preparation and training commands to build such a generator.
5. **A privacy audit** of those synthetic faces against the real images that
   trained the generator.

It is designed to operate either on a directory tree of images
(`images/<cohort>/*.jpg`) or on a pre-computed JSONL of landmarks.

## Installation

```bash
pip install -e .[all]       # everything
pip install -e .[morph]     # only landmarks + average face
pip install -e .[geometric] # only geometric features
pip install -e .[synth]     # pack / train / generate (StyleGAN3; needs a CUDA GPU to train)
pip install -e .[enhance]   # enhance (DDColor + GFPGAN)
pip install -e .[privacy]   # privacy (InsightFace, LPIPS)
```

Python ≥ 3.10. The geometric module depends on
[MediaPipe](https://github.com/google/mediapipe) and
[oaklib](https://github.com/INCATools/ontology-access-kit) for HPO/MONDO
resolution. The `synth`, `enhance` and `privacy` extras pull in PyTorch; model
weights (ArcFace, DDColor, GFPGAN) download on first use to `~/.cache/facekit`
or the InsightFace cache.

## CLI

A single `facekit` entry point exposes all commands:

| Command | What it does |
| --- | --- |
| `facekit extract-landmarks` | Run MediaPipe over an image folder; write JSON or JSONL landmarks (optionally with blendshapes, pose, mesh viz). |
| `facekit average-face` | Generate a per-cohort average face image. |
| `facekit extract-features` | Compute the full ~120-column geometric phenotype CSV from images or a landmark JSONL. Supports `--mode disease-specific` and `--frontal-check`. |
| `facekit extract-features-custom` | Same extractor, but driven by a user JSON that maps cohort folder names to a chosen subset of feature groups (no MONDO needed). Optionally loads a user Python file with extra `@register_feature` formulas. |
| `facekit score` | Convert a phenotype CSV into feature-level z-scores against a normative reference (a packaged FairFace control reference by default). |
| `facekit resolve-diseases` | Helper that resolves disease names to MONDO IDs and caches the results. |
| `facekit enhance` | Prepare photographs for generator training: DDColor colorization where an image is grayscale, GFPGAN restoration where the face is small. |
| `facekit pack` | Crop each face to a fixed square from its landmarks and pack a cohort into a StyleGAN3 dataset zip. |
| `facekit train` | Train a StyleGAN3 generator on a packed cohort with the FaceKit defaults (GPU). |
| `facekit generate` | Sample synthetic faces from a generator pickle into the cohort folder layout. |
| `facekit privacy` | Audit synthetic faces for identity (ArcFace / AdaFace / LVFace) and appearance (LPIPS) leakage from the training images. |
| 🚧 HPO prediction | _Coming soon_ — predict per-patient HPO phenotype terms directly from the geometric features. |

Run any command with `--help` for the full flag list.

## What each command does

> Example faces are derived from the [GestaltMatcher Database (GMDB)](https://db.gestaltmatcher.org) and are shown with consent.

One real example per command, produced by `scripts/make_figures.py`.

### `average-face`

![average-face](figures/average-face.png)

Average face of the Williams syndrome cohort (GMDB).

### `extract-landmarks`

![extract-landmarks](figures/extract-landmarks.png)

478-point MediaPipe mesh on an example face.

### `extract-features`

![extract-features](figures/extract-features.png)

Four geometric measurements (IPD, inter-canthal, nasal base width, mouth width) drawn on the face.

### `score`

`extract-features` gives each measurement in its own units, which says nothing
about whether a value is unusual. `score` divides that question out by
expressing every measurement as a z-score against a control population:

```
z = (measurement - reference mean) / reference SD
```

The packaged reference (`src/facekit/data/reference_fairface.csv`) holds the
per-feature mean, SD and n of the 886 FairFace control images that pass the
frontal-pose gate, sampled across three ancestry groups. Pass `--reference` a
phenotype CSV of your own controls to build a reference from them instead; the
same pose gate is applied, since a reference measured under looser acquisition
conditions carries an inflated SD. Rebuild the packaged table with
`scripts/build_reference.py`.

A reference and the faces scored against it must come from the same feature
definitions: the pose correction fixes the meaning of all 120 measurements at
once, so tables extracted under different conventions must not be combined.

### Quick start

```bash
# 1. Extract landmarks to a JSONL (cohort-aware: subfolders become cohorts)
facekit extract-landmarks -i images/ -o outputs/ --format jsonl --transform

# 2. Compute geometric features from that JSONL
facekit extract-features -i outputs/images_landmarks.jsonl -o results/

# 3. Express every measurement in reference SD units
facekit score -i results/phenotypes_all.csv -o results/

# 4. Average-face image per cohort
facekit average-face -i images/ -o results/
```

### Synthetic faces and the privacy audit

```bash
# 1. Optional: colorize grayscale photographs, restore small faces
facekit enhance -i raw/ -o prepared/

# 2. Face crops at 256x256 -> prepared/<cohort>/ and one StyleGAN3 zip per cohort
facekit pack -i prepared/ -o datasets/

# 3. Train one generator per cohort (GPU node)
facekit train --data datasets/noonan.zip -o runs/noonan --gpus 1 --batch 32 --batch-gpu 8

# 4. Sample 500 faces; the output is already a cohort folder for extract-features
facekit generate --network runs/noonan/00000-*/network-snapshot-005000.pkl \
    -o synthetic/ --name noonan --n 500

# 5. Audit the generator. Before step 3, hold out some patients per cohort:
#    train/<cohort>/ went into the zip, heldout/<cohort>/ never did.
facekit privacy --train train/ --heldout heldout/ --synthetic synthetic/ -o privacy/
```

`train` runs the vendored StyleGAN3 with the settings the FaceKit generators
were trained with (stylegan3-t, γ = 2, x-flips, 5,000 kimg, an 8-layer mapping
network and a cosine learning-rate decay); every native StyleGAN3 option can be
passed through with `--extra`. `privacy` writes `flagging.csv` (percentage of
synthetic images closer to a training image than a held-out image is, at
calibration percentiles 1, 2, 5, 10 and 20), `nnaa.csv` (nearest-neighbour
adversarial accuracy and privacy loss with bootstrap confidence intervals),
and two plots. The identity axis needs ArcFace only; AdaFace and LVFace are
used when their model files are passed.

### Custom disease → feature mapping

`extract-features-custom` takes a JSON of the form:

```json
{
  "by_disease": {
    "22q11.2 deletion syndrome": {
      "feature_groups": ["EYE_FISSURE_SLANT", "NOSE_TIP_SHAPE", "PHILTRUM_LENGTH"]
    }
  }
}
```

See `custom_mapping.json` for a working example.

User-defined feature formulas can be registered at import time:

```python
# my_features.py
from facekit.api import register_feature

@register_feature(
    group="MY_NEW_GROUP",
    csv_columns=["my_metric"],
    hpo_terms=[("HP:0001234", "Example phenotype")],
)
def my_metric(lm, scales):
    return {"my_metric": float(lm[0, 0])}
```

```bash
facekit extract-features-custom \
    -i images/ -o results/ \
    --user-mapping custom_mapping.json \
    --user-features my_features.py
```

Plugins are sandboxed against the base 120 columns and the HPO direction
codes; collisions raise at registration time.

## Roadmap

- 🚧 **HPO prediction** — predict per-patient [HPO](https://hpo.jax.org/) phenotype
  terms directly from the extracted geometric features, turning the 120-column
  representation into a ranked list of candidate facial phenotypes.
- 🚧 **Pre-trained generators** — `generate --cohort <name>` will download the
  generators trained on the ten GMDB rare-disease cohorts once their
  redistribution is approved. Until then `generate` takes a `--network` pickle.

## Project layout

```
src/facekit/
├── cli.py                # typer entry point
├── api.py                # public plugin registry
├── commands/             # one CLI command per file
├── core/
│   ├── morph/            # landmarks, alignment, averaging, warping
│   ├── geometric/        # 120-column extractor + batch drivers
│   ├── synth/            # face crop -> dataset zip, sampling from a generator
│   ├── enhance/          # grayscale / small-face tests, DDColor, GFPGAN
│   └── privacy/          # embeddings, LPIPS, flagging, NNAA
└── vendor/               # third-party code under its own licenses
    ├── stylegan3/        # NVIDIA StyleGAN3 (non-commercial), + --lr-schedule
    ├── ddcolor/          # DDColor model (Apache-2.0)
    └── gfpgan/           # GFPGAN v1 clean architecture (Apache-2.0)
scripts/
└── download_mondo.sh     # pin a dated mondo.obo for HPO/MONDO lookups
tests/                    # pytest suite
```

## License

FaceKit's own code is under `LICENSE`. The code under `src/facekit/vendor/` keeps
its upstream license: StyleGAN3 is under the NVIDIA Source Code License, which
permits **non-commercial use only**; DDColor and GFPGAN are Apache-2.0.
