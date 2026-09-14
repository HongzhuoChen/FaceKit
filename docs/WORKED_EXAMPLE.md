# FaceKit Worked End-to-End Example

> **⚠️ STALE — regenerate before publication.** The transcripts below were
> recorded with the pre-fix extractor and still show **125 feature columns and
> 6 metadata columns**. The current extractor emits **120 feature columns and
> 7 metadata columns** (`hairline_height`, `nostril_region_area`,
> `mouth_upper_fit_residual`, `lip_outer_to_inner_area` and
> `cupid_bow_peak_asym` were removed; a `derotated` column was added), and
> several columns changed their normalizer. The numbers here have deliberately
> **not** been hand-edited — they are pasted terminal output, and editing them
> would make this a fabricated log rather than a record of a real run.
> Re-run the walkthrough and replace this file wholesale.

This walkthrough takes you from a fresh install to a finished feature CSV,
using **real commands** from an actual FaceKit run. Every command shown here
matches the run logged in
`facekit_project/logs/facekit_run.log`; only the file paths have been kept
verbatim so you can see exactly what a real invocation looks like.

The example processes a folder of face images for one cohort
(`Crouzon_syndrome`, 41 images) and produces a 125-column phenotype CSV.

---

## Step 0 — Install FaceKit

FaceKit is a normal Python package (Python ≥ 3.10). Install it in editable
mode from the repository root:

```bash
# everything: landmarks + geometric features + average-face
pip install -e ".[all]"

# or, just the parts needed for this walkthrough:
pip install -e ".[geometric]"
```

This puts a single `facekit` command on your `PATH`. Check it:

```bash
facekit --help
facekit extract-landmarks --help
facekit extract-features --help
```

The MediaPipe Face Landmarker model file is downloaded automatically the
first time you run a command that needs it.

---

## Step 1 — Arrange the input images

FaceKit reads a folder of images. Two layouts are supported:

```
images/                          images/Crouzon_syndrome/
├── Crouzon_syndrome/    OR       ├── img17541.jpg
│   ├── img001.jpg               ├── img17560.jpg
│   └── ...                      └── ...
└── Williams_syndrome/
    └── ...
```

- **Multi-cohort**: each subfolder is one cohort; the subfolder name becomes
  the `disease` label.
- **Single folder**: a flat folder of images is treated as one unnamed
  cohort.

In the worked run, each cohort was processed from its own single folder, e.g.
`images/Crouzon_syndrome/` containing 41 `.jpg` files.

---

## Step 2 — Extract landmarks

Run MediaPipe over the image folder and write the 478 landmarks per face to a
single JSONL file. The `--transform` flag also stores the head-pose matrix,
which the next step needs for `--frontal-check`.

**Real command (from `facekit_run.log`):**

```bash
facekit extract-landmarks \
  --input  images/Crouzon_syndrome \
  --output results/landmarks/Crouzon_syndrome \
  --format jsonl \
  --transform
```

**What it does, flag by flag:**

| Flag | Effect |
| --- | --- |
| `--input` | Folder of images to process. |
| `--output` | Directory the JSONL file is written into. |
| `--format jsonl` | Write **one** `*_landmarks.jsonl` file (one image per line) instead of one JSON per image. This is the format `extract-features` consumes. |
| `--transform` | Also store the 4×4 facial transformation matrix (head pose) on each line. Needed for the frontal-pose check later. |

**What it prints:**

```
[FaceKit] No subfolders detected; treating .../images/Crouzon_syndrome as a single cohort
[FaceKit] Loading MediaPipe Face Landmarker...
[FaceKit] Outputs: landmarks + transformation matrix
[FaceKit] Crouzon_syndrome: processing 41 images
[FaceKit]   ✓ 41/41 -> Crouzon_syndrome_landmarks.jsonl
[FaceKit] Done: 41/41 images processed across 1 cohort(s) -> .../Crouzon_syndrome_landmarks.jsonl
```

**Output file:**

```
features/landmarks/Crouzon_syndrome/Crouzon_syndrome_landmarks.jsonl
```

The file name is `<input_folder_name>_landmarks.jsonl`. Each line is one JSON
object — here is one line with the landmark list shortened:

```json
{
  "image_id": "img16983",
  "disease": "",
  "landmarks_3d": [[0.5156, 0.7740, -0.0914], [0.5105, 0.6679, -0.1669], ...478 triples...],
  "image_size": [w, h],
  "transformation_matrix": [[...], [...], [...], [...]]
}
```

| Field | Meaning |
| --- | --- |
| `image_id` | Image file name without extension. |
| `disease` | Cohort label — empty here because the input was a single flat folder. |
| `landmarks_3d` | 478 `[x, y, z]` points in **normalized [0,1]** coordinates. |
| `image_size` | `[width, height]` of the source image, in pixels. |
| `transformation_matrix` | 4×4 head-pose matrix (present because `--transform` was set). |

See `OUTPUT_FORMAT.md` §1.2 for the full schema.

---

## Step 3 — Extract geometric features

Feed the JSONL into the geometric extractor to produce the phenotype CSV.

**Real command (from `facekit_run.log`):**

```bash
facekit extract-features \
  --input  results/landmarks/Crouzon_syndrome/Crouzon_syndrome_landmarks.jsonl \
  --output results/per_cohort/Crouzon_syndrome \
  --mode all \
  --frontal-check
```

**What it does, flag by flag:**

| Flag | Effect |
| --- | --- |
| `--input` | The JSONL landmark file from Step 2. (Could also be an image folder — then MediaPipe runs here instead.) |
| `--output` | Directory the CSV is written into. |
| `--mode all` | Compute **all 125** feature columns. (`disease-specific` would mask to a per-cohort subset.) |
| `--frontal-check` | Reject faces whose head pose is out of range. Requires the transformation matrix from Step 2. |

**What it prints:**

```
[FaceKit] extract-features: mode=all
[FaceKit] Input    : .../Crouzon_syndrome_landmarks.jsonl
[FaceKit] Output   : .../features/per_cohort/Crouzon_syndrome
[FaceKit] Done: 41 rows, 125 feature columns -> .../features/per_cohort/Crouzon_syndrome
```

**Output file:**

```
features/per_cohort/Crouzon_syndrome/phenotypes_all.csv
```

The file is auto-named `phenotypes_all.csv` because `--mode all` was used. It
has 41 rows (one per image) and 6 metadata columns + 125 feature columns:

```
disease, image_id, frontal_ok, pose_yaw, pose_pitch, pose_roll,
eb_thickness_r, eb_thickness_l, ... (125 feature columns) ...
```

| Column group | Meaning |
| --- | --- |
| `disease`, `image_id` | Identifiers carried from the JSONL. |
| `frontal_ok` | `True`/`False`/`NaN` — whether the pose passed the frontal check. |
| `pose_yaw`, `pose_pitch`, `pose_roll` | Head angles in degrees. |
| 125 feature columns | The geometric phenotype. See `FEATURE_GLOSSARY.md`. |

A row that fails `--frontal-check` keeps its identifier and pose columns but
has `NaN` for all 125 feature columns. See `OUTPUT_FORMAT.md` §2 for the full
schema.

---

## Step 4 — Repeat per cohort and combine

For a multi-cohort study you run Steps 2–3 once per cohort. The logged run
processed six cohorts the same way:

| Cohort | Images |
| --- | --- |
| `Crouzon_syndrome` | 41 |
| `Williams_syndrome` | 169 |
| `Noonan_syndrome` | 131 |
| `Hutchinson-Gilford_progeria` | 40 |
| `Sotos_syndrome` | 61 |
| `healthy` | 358 |

Each cohort produces its own `phenotypes_all.csv`. Concatenating the six CSVs
(and adding a `cohort` column from the folder name) gives one combined table
of 800 rows × 125 features, ready for downstream analysis.

> **Tip — one command for many cohorts.** If you arrange the images as
> `images/<cohort>/*.jpg` (multi-cohort layout) and point `--input` at the
> parent `images/` folder, a single `extract-landmarks` call produces one
> JSONL covering every cohort, with the `disease` field set per line. The
> logged run processed cohorts individually instead, which is equally valid.

---

## Quick-reference: the whole pipeline

```bash
# 0. install
pip install -e ".[geometric]"

# 1. (arrange images/ — your data)

# 2. images  ->  landmarks JSONL
facekit extract-landmarks -i images/Crouzon_syndrome \
                          -o features/landmarks/Crouzon_syndrome \
                          --format jsonl --transform

# 3. landmarks JSONL  ->  phenotype CSV (125 features)
facekit extract-features -i features/landmarks/Crouzon_syndrome/Crouzon_syndrome_landmarks.jsonl \
                         -o features/per_cohort/Crouzon_syndrome \
                         --mode all --frontal-check
```

| Stage | Input | Output |
| --- | --- | --- |
| `extract-landmarks` | folder of images | `*_landmarks.jsonl` (478 points/face) |
| `extract-features` | `*_landmarks.jsonl` | `phenotypes_all.csv` (125 features/face) |

For field-by-field schemas see `OUTPUT_FORMAT.md`; for what each feature
column means see `FEATURE_GLOSSARY.md`.
