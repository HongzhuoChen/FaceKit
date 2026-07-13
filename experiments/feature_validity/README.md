# Feature-validity audit

Does each geometric feature actually measure the HPO term it is mapped to?

## Why the obvious experiment does not work

Gold-positive vs "not annotated" is meaningless: only 26% of CdLS patients carry a
`HP:0000664 Synophrys` annotation against a clinical prevalence above 95%, and the
whole GMDB gold set contains **4** explicit `absent` annotations for it. The negative
arm would be ~70% false negatives.

Swapping in patients from *other* syndromes fixes the labels but buys a worse problem:
two syndromes differ in everything (face shape, age, ancestry, and the `bizyg` /
`face_h` denominators themselves), so any separation is uninterpretable.

## The two contrasts

Neither contrast is sound alone; their failure modes are opposite, so both are run.

| | label contamination | cohort confound | fails by |
|---|---|---|---|
| **CROSS** gold-positives pooled over all syndromes (capped at 30/syndrome so no single disease dominates) vs patients of syndromes where the term is absent from HPOA *and* observed at <5% | low | **high** | **false positives** |
| **WITHIN** gold-positive vs gold-silent patients *inside the same syndrome* (gold-silent = has ≥1 other present HPO but not this one), z-scored per syndrome then pooled | **high** | none | **false negatives** |

WITHIN can only dilute an effect, never fabricate one, so a hit there is trustworthy
and a miss is not a refutation. CROSS is judged against a **placebo null**: the same
contrast recomputed over the ~110 features of *unrelated* anatomical regions. The
question is never "is the AUC high" but "is it higher than unrelated features get for
free on the same split".

## Calibration

The run is void unless these land as expected:

- **positive** — `eye_fissure_slant_mean` for *both* Down- and Up-slanted palpebral
  fissures, i.e. the same feature with opposite expected directions. A cohort effect
  pushes a feature one way and cannot satisfy both. Both come out VALIDATED.
- **negative** — High forehead → `forehead_height`, which misses in three separate
  diseases in the earlier `facekit_project` validation. Comes out NO_SIGNAL.

## Scripts

| | |
|---|---|
| `run_validity.py` | the two contrasts + placebo null → `results/feature_validity.csv` |
| `run_renorm.py` | re-expresses every `bizyg`/`face_h`-normalized feature per inter-pupillary distance → `results/renorm.csv` |
| `run_remap.py` | held-out confirmation of re-mapping candidates: the best-in-region feature is picked on a discovery half and scored on the held-out half, over 5 splits → `results/remap_heldout.csv` |

Patient-level throughout (median over that patient's frontal images); splits are by
patient, never by image.

## Findings

Of 100 vocabulary terms, 53 are UNDERPOWERED (<30 gold-positive patients — half the
vocabulary cannot be evaluated at all). Of the 47 testable: **11 VALIDATED**, 6
LIKELY_COHORT, 3 CROSS_ONLY, 27 NO_SIGNAL.

**Regions where no feature works at all** — eyebrow (0/3), cheek/midface (0/2, best
held-out feature scores 0.364, i.e. *anti*-correlated), forehead (1/4). These are
exactly the regions whose MediaPipe landmarks have no image evidence to lock onto:
brow points are anchored to the canonical mesh rather than to the hair boundary,
cheek/maxilla points sit in the face interior where there is no edge, and TRICHION is
a hairline hidden under hair.

`eb_inter_distance` for synophrys: AUC 0.545, at the **45th percentile** of unrelated
features — no signal at all. The best feature in the whole eyebrow region reaches
0.557 on held-out data. Synophrys is not mis-mapped; the eyebrow landmarks cannot see it.

**The normalizer was not the culprit.** Re-expressing 44 features per inter-pupillary
distance rescues nothing (Full cheeks 0.484 → 0.523; most deltas negative). The
denominators are not what is broken — the landmarks are.

**Two re-mappings survived held-out confirmation** and are applied in
`labels/facial_hpo_vocab.csv`:

| HPO | was | held-out | now | held-out |
|---|---|---|---|---|
| Anteverted nares | `nasolabial_angle` +1 | 0.543 | `nose_length` **−1** | **0.702** |
| Epicanthus | `eye_epicanthus_angle_mean` −1 | 0.432 (inverted) | `gaze_asym_x` **+1** | **0.686** |

The epicanthal fold displaces the detected inner canthus laterally in *both* eyes,
which is antisymmetric in image x, so `iris_offset_x_r - iris_offset_x_l` accumulates
rather than cancelling. The originally mapped angle is inverted because the fold makes
MediaPipe misplace the very landmark the angle is built from.

Five other candidates that looked strong on full data (broad nasal tip, short
columella, depressed nasal tip, micrognathia, thick eyebrow) did **not** survive the
held-out split — they were winner's curse, a max taken over 10-20 features.

Six terms pass CROSS at AUC 0.65–0.77 but are refused by the zero-confound WITHIN
contrast (short nose, long face, narrow mouth, low hanging columella, thick vermilion,
everted lower lip). Their validity claims do not stand.

## Source of truth

`facial_hpo_vocab.csv` is *generated*, not authored: `hpo_predict/step2_build_labels.py`
rebuilds it from `hpo_direction_codes.csv`. That file used to live outside the repo, so
any rerun of step2 on a machine with a stale copy would have silently reverted the two
re-mappings above. It is now tracked at `labels/hpo_direction_codes.csv` and is what
`defaults.py` and both `hpo_predict*/config.py` read by default (still overridable via
`FACEKIT_DIRECTION_CODES_CSV`). Re-running step2 now reproduces `facial_hpo_vocab.csv`
byte-for-byte.
