# Packaged reference distributions

`reference_fairface.csv` — per-feature `mean`, `sd` and `n` over the FairFace
control images used as the normative reference throughout the FaceKit paper:
2,051 images sampled across three ancestry groups (white / black / asian), of
which 886 pass the frontal-pose gate and enter the table. `facekit score` uses
it by default.

The values are tied to the current feature definitions. Pose correction takes
depth from a fixed canonical-face table, which fixes the meaning of all 120
measurements at once; a reference built under a different convention must not
be used to score faces extracted under this one.

Rebuild with:

```bash
python scripts/build_reference.py CONTROLS_phenotypes_all.csv
```
