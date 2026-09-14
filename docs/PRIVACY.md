# Privacy audit: `facekit privacy`

`privacy` checks whether the faces sampled from a generator carry residual
information about the real photographs that trained it. It is the toolkit
form of the assessment in the FaceKit manuscript (Methods, "Privacy
assessment of the pre-trained generators"), ported from the standalone
`face_audit` package used there.

## Inputs

Three roots with one subfolder per cohort; cohort folder names must match
across the three. A cohort present in only one or two roots is skipped with a
message.

```
train/<cohort>/*.png      the images the generator was trained on
heldout/<cohort>/*.png    real images of the same cohort never used in training
synthetic/<cohort>/*.png  images sampled from that cohort's generator
```

The held-out partition must be disjoint from the training partition **at the
patient level**; two photographs of one patient split across the partitions
would make held-out images look like training images and shift every
threshold. FaceKit cannot check this, since it knows nothing about patients.

Use the crops written by `facekit pack` for the two real partitions, so that
real and synthetic images share one framing. The audit compares images by
nearest-neighbour distance, and a systematic framing difference between
partitions shows up as distance.

## Two axes

| axis | metric | what a hit means |
| --- | --- | --- |
| identity | cosine distance between recognition embeddings; ArcFace (InsightFace `buffalo_l`) by default, plus AdaFace (`--adaface-dir`, `--adaface-ckpt`) and LVFace (`--lvface-onnx`) when given | the synthetic face resembles a training patient as a face-recognition system sees them |
| appearance | LPIPS (AlexNet, 256 px) between whole images | the synthetic image resembles a training photograph pixel-wise: lighting, framing, background, colour cast |

Faces are detected with InsightFace's RetinaFace and aligned to 112 × 112
through five keypoints; images in which no face is found are dropped and
counted in `privacy_results.json`.

## Statistics

**Flagging at calibrated thresholds.** For each synthetic and each held-out
image, the distance to the nearest training image *of the same cohort* is
computed. The threshold at percentile $p$ is the $p$-th percentile of the
held-out distances, pooled over cohorts, so the held-out flagging rate is $p$
by construction (the audit's built-in check) and the synthetic rate is the
quantity of interest. A synthetic rate at or below $p$ means a sample from
the generator is no more likely to land close to a training patient than
another real photograph of the same syndrome is. Percentiles default to
1, 2, 5, 10, 20 (`--percentiles`).

**Nearest-neighbour adversarial accuracy (NNAA).** For two sets $A$, $B$,
$\mathrm{AA}(A,B)$ is the fraction of points whose nearest neighbour within
their own set is closer than their nearest neighbour in the other set,
averaged over both directions; 0.5 means the sets cannot be told apart by
proximity. The audit reports $\mathrm{AA}(\text{train},\text{synthetic})$,
$\mathrm{AA}(\text{held-out},\text{synthetic})$ and their difference, the
**privacy loss**
$\mathrm{AA}(\text{held-out},\text{synth}) - \mathrm{AA}(\text{train},\text{synth})$.
A privacy loss near zero says the synthetic images lie no closer to the
training partition than held-out real images do. Confidence intervals come
from resampling images with replacement (`--bootstrap`, default 1000
draws). Sets are capped at `--max-samples` (1000) for embeddings and
`--lpips-max-samples` (200) for LPIPS, whose cost is quadratic.

The absolute AA level carries no privacy meaning on its own: synthetic faces
are globally separable from real ones in both embedding and LPIPS space, so
both AA values sit well above 0.5. Only their difference is informative.

## Outputs

| file | content |
| --- | --- |
| `flagging.csv` | one row per metric and percentile: `threshold`, `synthetic_pct`, `heldout_pct`, `n_synthetic`, `n_heldout` |
| `nnaa.csv` | one row per metric: `aa_train_synth`, `aa_heldout_synth`, `privacy_loss` with `ci_lower`, `ci_upper`, set sizes, `n_bootstrap` |
| `privacy_results.json` | everything above plus per-cohort no-face counts |
| `below_threshold.png` | synthetic and held-out flagging rate against $p$; identity backbones left, LPIPS right |
| `nnaa.png` | AA bars with bootstrap intervals |
| `cache/` | embeddings per backbone/split/cohort and LPIPS distance arrays; a re-run with other percentiles or bootstrap counts reuses them |

## Reading the manuscript's result

On the ten GMDB generators, synthetic images were flagged below the held-out
rate for all three recognition models at every operating point, and the
embedding privacy loss was indistinguishable from zero. On LPIPS the
synthetic images were flagged well above the held-out rate, with a privacy
loss whose interval excludes zero. The two axes disagree because they
measure different things: LPIPS cannot separate two photographs of one
patient from photographs of two patients, so what it detects is the
acquisition style of the training set (colour cast, framing, background),
which the generator learns along with facial morphology. The identity axis is
the one that speaks to re-identification.

The audit is empirical: it bounds what these particular models and this
metric can detect. A stronger attack or a better embedding could in
principle detect more.

## Practical notes

- **CPU runs.** `--threads` (default 8) caps onnxruntime and torch threads.
  Without it onnxruntime spins one thread per core, and on a busy shared node
  InsightFace slows to tens of seconds per image.
- **AdaFace** needs a clone of the AdaFace repository (for `net.py`) and an
  IR-50 checkpoint; **LVFace** needs the ONNX model from Hugging Face.
  Neither is downloaded automatically.
- **Metric sanity check.** The manuscript also verified each recognition
  model's ordinal behaviour on same-patient pairs. That check needs patient
  identifiers in the file names and is not part of `facekit privacy` yet.
