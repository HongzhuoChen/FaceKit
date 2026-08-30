"""Per-feature sensitivity of the 120 measurements to ancestry, age and resolution.

Three columns, one row per feature:

  race_eta2   partial eta^2 of ancestry in the healthy FairFace controls,
              adjusted for age band, gender and the three pose angles
  age_eta2    partial eta^2 of age band, adjusted for ancestry, gender and pose
  res_drift   median |change| of the feature, in reference SD, when the same
              face is measured at 224 px instead of its native 448 px

The first two are mutually adjusted, so a feature that looks ancestry-sensitive
only because the ancestry groups differ in age composition is not credited to
ancestry. The third is a paired within-image quantity and involves no model.

This table is used twice: as a descriptive result, and as the ordering variable
for the concentration test in ``compare_arms.py`` (a conditioning that works
should help most where the feature is most sensitive to what is conditioned on).

Usage:
    PYTHONPATH=src python experiments/reference_sensitivity/feature_sensitivity.py
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

HERE = Path(__file__).resolve().parent
CHOP = Path(os.environ.get(
    "FACEKIT_CHOP_DIR",
    "/vast/projects/kai/multimodal-machine-learn/hongzhuo/chop_22q_analysis"))

NON_FEATURE = {"disease", "image_id", "frontal_ok", "derotated",
               "pose_yaw", "pose_pitch", "pose_roll", "age", "gender", "race"}

FULL = "y ~ C(age) + C(race) + C(gender) + pose_yaw + pose_pitch + pose_roll"
DROP = {"race": "y ~ C(age) + C(gender) + pose_yaw + pose_pitch + pose_roll",
        "age":  "y ~ C(race) + C(gender) + pose_yaw + pose_pitch + pose_roll"}


def partial_eta2(data: pd.DataFrame, term: str) -> tuple[float, float]:
    full = smf.ols(FULL, data=data).fit()
    reduced = smf.ols(DROP[term], data=data).fit()
    table = sm.stats.anova_lm(reduced, full)
    ss = table.ss_diff.iloc[1]
    return float(ss / (ss + full.ssr)), float(table["Pr(>F)"].iloc[1])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--resolution-drift", type=Path,
                    default=HERE / "results" / "resolution_drift.csv")
    ap.add_argument("--out", type=Path, default=HERE / "results" / "feature_sensitivity.csv")
    args = ap.parse_args()

    manifest = pd.read_csv(CHOP / "fairface_controls" / "manifest.csv")
    controls = pd.read_csv(CHOP / "feats_fairface_v2" / "phenotypes_all.csv")
    controls = controls[controls.frontal_ok == True]  # noqa: E712
    controls = controls.merge(manifest[["image_id", "age", "gender", "race"]],
                              on="image_id")
    features = [c for c in controls.columns if c not in NON_FEATURE]

    rows = []
    for col in features:
        data = controls[[col, "race", "age", "gender",
                         "pose_yaw", "pose_pitch", "pose_roll"]].dropna()
        data = data.rename(columns={col: "y"})
        race_eta2, race_p = partial_eta2(data, "race")
        age_eta2, age_p = partial_eta2(data, "age")
        rows.append(dict(feature=col, race_eta2=race_eta2, race_p=race_p,
                         age_eta2=age_eta2, age_p=age_p))

    out = pd.DataFrame(rows)
    out["race_q"] = multipletests(out.race_p, method="fdr_bh")[1]
    out["age_q"] = multipletests(out.age_p, method="fdr_bh")[1]

    if args.resolution_drift.exists():
        drift = pd.read_csv(args.resolution_drift)
        out = out.merge(drift[["feature", "res_drift_224"]], on="feature", how="left")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)

    print(f"{len(out)} features -> {args.out}\n")
    for term in ("race", "age"):
        sig = out[out[f"{term}_q"] < 0.05]
        print(f"{term:5s}: q<0.05 for {len(sig)}/{len(out)} features | "
              f"median partial eta^2 among those {sig[f'{term}_eta2'].median():.3f} | "
              f"max {out[f'{term}_eta2'].max():.3f}")
    print("\nmost ancestry-sensitive:")
    print(out.nlargest(6, "race_eta2")[["feature", "race_eta2", "age_eta2"]]
          .to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\nmost age-sensitive:")
    print(out.nlargest(6, "age_eta2")[["feature", "age_eta2", "race_eta2"]]
          .to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
