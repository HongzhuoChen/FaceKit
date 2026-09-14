# FaceKit Output-Format Specification

This document describes exactly what the two main FaceKit commands write to
disk: file types, file names, and column/field schemas. It reflects the code
in `src/facekit/commands/extract_landmarks.py`,
`src/facekit/commands/extract_features.py`, and
`src/facekit/core/geometric/`. Section 4 lists the log and result files of
the generation and privacy commands.

---

## 1. `facekit extract-landmarks`

Runs MediaPipe Face Landmarker over an image folder and writes the detected
landmarks. Two output formats are selectable with `--format`.

### Input layouts
- **Multi-cohort**: `images/<cohort>/*.jpg` — each subfolder is one cohort.
- **Single folder**: `images/*.jpg` — treated as one unnamed cohort.

### 1.1 `--format json` (default)

One JSON file **per image**, written next to a per-cohort subfolder:

```
<output_dir>/<cohort>/<image_stem>_landmarks.json
```

(For single-folder input the files go directly under `<output_dir>/`.)
If `--visualize` is set, an annotated `<image_stem>_vis.png` is written
alongside each JSON.

The JSON object is produced by `MediaPipeLandmarkExtractor.extract_full` and
contains the landmark array plus any optional fields enabled by flags
(`--blendshapes`, `--transform`).

### 1.2 `--format jsonl` (recommended for the feature pipeline)

A **single** file:

```
<output_dir>/<input_dir_name>_landmarks.jsonl
```

One JSON object per line, one line per successfully detected image. This is
the format consumed directly by `facekit extract-features`. `--visualize` is
not allowed in this mode.

**Per-line schema** (from `_process_image_jsonl`):

| Field | Type | Always present? | Meaning |
| --- | --- | --- | --- |
| `image_id` | string | yes | The image file name without extension. |
| `disease` | string | yes | The cohort subfolder name; empty string `""` for single-folder input. |
| `landmarks_3d` | list of 478 `[x, y, z]` triples | yes | MediaPipe landmarks in **normalized [0,1] coordinates** (x/y relative to image width/height; z is relative depth). Feature extraction reads x/y only — z is accepted and ignored, see `derotated` below. |
| `image_size` | `[width, height]` integers | yes | Pixel size of the source image. |
| `transformation_matrix` | 4×4 list of floats | only with `--transform` | MediaPipe facial transformation matrix; encodes head pose. **Required for `extract-features --frontal-check`**, and without it pose correction degrades to roll-only (`derotated = False`). |
| `blendshapes` | list of 52 floats | only with `--blendshapes` | Facial-expression blendshape scores. |

Images where no face is detected are skipped (not written) and reported on
stderr.

> **Coordinate note.** The JSONL stores **normalized** coordinates. The
> feature extractor multiplies `landmarks_3d` by `image_size` internally to
> recover pixel coordinates before computing geometry.

---

## 2. `facekit extract-features`

Runs the geometric feature extractor (120 columns) and writes one phenotype
CSV. Input can be **either** an image directory **or** a `.jsonl` landmark
file produced by `extract-landmarks --format jsonl`.

### 2.1 Output file

A single CSV in `<output_dir>/`, auto-named by `--mode`:

| `--mode` | Output file |
| --- | --- |
| `all` (default) | `phenotypes_all.csv` |
| `disease-specific` | `phenotypes_disease_specific.csv` |

In `disease-specific` mode an extra side file
`disease_specific_selection_report.csv` is written, listing which feature
columns were kept for each cohort.

### 2.2 CSV schema

The CSV has **7 metadata columns followed by the 120 feature columns**, in
this fixed order:

```
disease, image_id, frontal_ok, derotated, pose_yaw, pose_pitch, pose_roll, <120 feature columns>
```

| Column | Type | Meaning |
| --- | --- | --- |
| `disease` | string | Cohort label carried from the JSONL `disease` field (or the image subfolder name). May be empty for single-folder input. |
| `image_id` | string | Image file name without extension. |
| `frontal_ok` | tri-state | `True` = head pose within the frontal thresholds; `False` = out of range; `NaN` = pose unknown (no transformation matrix). **Always reports the real pose gate**, whether or not `--frontal-check` is on — the flag only controls whether out-of-range rows still get features computed. |
| `derotated` | bool | `True` = yaw/pitch/roll were undone with the transformation matrix and the canonical depth table (the normal path). `False` = fell back to roll-only correction because the matrix was missing. A run mixing both values is mixing two feature definitions and should not be pooled. |
| `pose_yaw` | float (deg) | Head turn left/right, from the transformation matrix. `NaN` if no matrix. |
| `pose_pitch` | float (deg) | Head tilt up/down. `NaN` if no matrix. |
| `pose_roll` | float (deg) | Head tilt sideways. `NaN` if no matrix. |
| 120 feature columns | float | The geometric features. See `FEATURE_GLOSSARY.md`. |

The 120 feature columns appear in a **fixed canonical order**, set once by
`GeometricFeatureExtractor._init_feature_columns`, so the schema is identical
across runs and across cohorts. Plugin columns (if any) are appended after
the base 120.

### 2.3 The `--frontal-check` flag

When `--frontal-check` is passed, each face is tested against the head-pose
thresholds (extractor defaults: `|yaw| ≤ 15°`, `|pitch| ≤ 15°`,
`|roll| ≤ 10°`). A face that fails gets `NaN` for all 120 feature columns;
the row is still written.

`frontal_ok` itself does **not** depend on this flag — it always reports the
pose gate whenever the pose is known. Without the flag you get every row's
features plus an honest quality column, so downstream code can filter on
`df[df.frontal_ok]` itself. (Before this was fixed, `frontal_ok` was
hardcoded to `True` whenever a matrix existed, which silently turned every
such downstream filter into a no-op.)

With a JSONL input, every row **must** carry a `transformation_matrix` when
`--frontal-check` is on, otherwise the run fails. That is why
`extract-landmarks` should be run with `--transform` when you intend to use
`--frontal-check` downstream.

> Projects that need a **stricter** frontal criterion can filter on the
> `pose_yaw / pose_pitch / pose_roll` columns of the output CSV themselves —
> the columns are always written when a transformation matrix is present.

### 2.4 Missing values

- A row with `frontal_ok = False` has `NaN` in all feature columns.
- An individual feature can be `NaN` if its geometry is degenerate for that
  face (e.g. a division by a near-zero length). The extractor adds small
  epsilon terms to most denominators to avoid this, so `NaN` feature cells
  are rare in practice.

---

## 3. Format summary

| Command | Format flag | On-disk artifact | One row/object per |
| --- | --- | --- | --- |
| `extract-landmarks` | `--format json` | `<cohort>/<stem>_landmarks.json` (+ optional `_vis.png`) | image |
| `extract-landmarks` | `--format jsonl` | `<input_dir_name>_landmarks.jsonl` | image (line) |
| `extract-features` | `--mode all` | `phenotypes_all.csv` | image (row) |
| `extract-features` | `--mode disease-specific` | `phenotypes_disease_specific.csv` + `disease_specific_selection_report.csv` | image (row) |

---

## 4. Generation and privacy commands

### 4.1 `facekit enhance`

```
<output_dir>/<cohort>/<image_stem>.png   every input image, processed or not
<output_dir>/enhance_log.csv
```

| column | type | meaning |
| --- | --- | --- |
| `cohort` | str | input subfolder (or the input folder's name) |
| `image` | str | input file name |
| `grayscale` | bool / empty | saturation test result; empty when the file was unreadable |
| `face_px` | int / empty | shorter side of the landmark box; empty when no face was found or `--no-restore` |
| `colorized` | bool | DDColor ran |
| `restored` | bool | GFPGAN ran |
| `status` | `ok` / `unreadable` | |

### 4.2 `facekit pack`

```
<output_dir>/<cohort>/<image_stem>.png   square crops at --resolution
<output_dir>/<cohort>.zip                StyleGAN3 dataset archive (PNGs + dataset.json)
<output_dir>/pack_log.csv                cohort, image, status (ok / no_face / unreadable)
```

### 4.3 `facekit train`

StyleGAN3's own run directory, `<output_dir>/<NNNNN>-<cfg>-<dataset>-gpus<N>-batch<B>-gamma<G>/`,
with `training_options.json`, `log.txt`, `stats.jsonl`, `reals.png`,
`fakes<kimg>.png` and `network-snapshot-<kimg>.pkl` (a pickle holding
`G`, `D`, `G_ema`, `augment_pipe`, `training_set_kwargs`).

### 4.4 `facekit generate`

```
<output_dir>/<name>/seed<NNNN>.png       one PNG per seed, --name defaults to the pickle stem
```

### 4.5 `facekit privacy`

```
<output_dir>/flagging.csv
<output_dir>/nnaa.csv
<output_dir>/privacy_results.json
<output_dir>/below_threshold.png
<output_dir>/nnaa.png
<output_dir>/cache/embeddings/<backbone>/<split>__<cohort>.{npy,json}
<output_dir>/cache/lpips/*.npy
```

`flagging.csv`:

| column | meaning |
| --- | --- |
| `metric` | `arcface`, `adaface`, `lvface` or `lpips` |
| `percentile` | calibration percentile p |
| `threshold` | p-th percentile of the held-out nearest-training distances |
| `synthetic_pct`, `heldout_pct` | percentage of synthetic / held-out images below the threshold (`heldout_pct` ≈ p by construction) |
| `n_synthetic`, `n_heldout` | images entering the comparison after dropping undetected faces |

`nnaa.csv`:

| column | meaning |
| --- | --- |
| `metric` | as above |
| `aa_train_synth`, `aa_heldout_synth` | bootstrap medians of the two adversarial accuracies |
| `privacy_loss`, `ci_lower`, `ci_upper` | `aa_heldout_synth - aa_train_synth` with its 95 % bootstrap interval |
| `n_train`, `n_heldout`, `n_synthetic`, `n_bootstrap` | set sizes after capping, bootstrap draws |
