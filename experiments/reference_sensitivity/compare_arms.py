"""Paired comparison of a pooled versus a conditioned reference.

The two arms score the same patients on the same terms, so the comparison is
paired term by term, not a contrast of two independent rates.

Four readings, in increasing order of how much they use:

  headline    how many terms move between "reproduced" and "not", tested with
              an exact McNemar. Uses only the terms that cross the threshold,
              so it is the least powerful reading and rarely reaches
              significance on a catalogue this size.
  effect      the signed z of every term, tested with a Wilcoxon. Uses all
              terms but treats them as interchangeable.
  mechanism   whether the gain concentrates on the features that are most
              sensitive to the variable being conditioned on. A conditioning
              that works should help where it is supposed to help; a gain
              unrelated to sensitivity is indistinguishable from drift. This
              is the reading that separated ancestry from age.
  artifact    conditioning on anything shrinks the reference SD, which inflates
              every |z| mechanically and pushes terms over the threshold
              without improving calibration. The within-group to pooled SD
              ratio bounds how much of any gain that can explain.

Usage:
    python experiments/reference_sensitivity/compare_arms.py --arm ancestry
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, wilcoxon, spearmanr, mannwhitneyu

HERE = Path(__file__).resolve().parent
CHOP = Path(os.environ.get(
    "FACEKIT_CHOP_DIR",
    "/vast/projects/kai/multimodal-machine-learn/hongzhuo/chop_22q_analysis"))
SENSITIVITY_COLUMN = {"ancestry": "race_eta2", "age": "age_eta2",
                      "resolution": "res_drift_224"}
GROUPING_COLUMN = {"ancestry": "race", "age": "age"}


def load(path: Path, tag: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    keep = ["hpo_id", "hpo_name", "feature", "arm", "n_patients"]
    stats = ["z", "q", "rank", "passed", "specific"]
    return df[keep + stats].rename(columns={c: f"{c}_{tag}" for c in stats})


def sd_ratio(arm: str) -> float | None:
    """Median within-group / pooled reference SD, the mechanical inflation."""
    if arm not in GROUPING_COLUMN:
        return None
    manifest = pd.read_csv(CHOP / "fairface_controls" / "manifest.csv")
    ref = pd.read_csv(CHOP / "feats_fairface_v2" / "phenotypes_all.csv")
    ref = ref[ref.frontal_ok == True].merge(  # noqa: E712
        manifest[["image_id", GROUPING_COLUMN[arm]]], on="image_id")
    non = {"disease", "image_id", "frontal_ok", "derotated",
           "pose_yaw", "pose_pitch", "pose_roll", GROUPING_COLUMN[arm]}
    feats = [c for c in ref.columns if c not in non]
    within = ref.groupby(GROUPING_COLUMN[arm])[feats].std(ddof=1).mean()
    return float((within / ref[feats].std(ddof=1)).median())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["ancestry", "age", "resolution"])
    ap.add_argument("--results", type=Path, default=None)
    args = ap.parse_args()
    base = args.results or (HERE / "results" / args.arm)

    merged = load(base / "pooled" / "reference_validity.csv", "pool").merge(
        load(base / "conditioned" / "reference_validity.csv", "cond"),
        on=["hpo_id", "hpo_name", "feature", "arm", "n_patients"])
    gold = merged[(merged.arm == "GOLD") & merged.q_pool.notna()].copy()
    gold["gain"] = gold.z_cond - gold.z_pool

    gained = (~gold.passed_pool) & gold.passed_cond
    lost = gold.passed_pool & (~gold.passed_cond)
    n_g, n_l = int(gained.sum()), int(lost.sum())

    print(f"=== arm: {args.arm} ===")
    print(f"terms compared (GOLD, testable): {len(gold)}")
    print(f"patients: {int(gold.n_patients.max())} max per term\n")

    print("headline")
    print(f"  reproduced: {int(gold.passed_pool.sum())} -> {int(gold.passed_cond.sum())}")
    print(f"  not reproduced -> reproduced : {n_g}")
    print(f"  reproduced -> not reproduced : {n_l}")
    p = binomtest(n_g, n_g + n_l, 0.5).pvalue if (n_g + n_l) else float("nan")
    print(f"  exact McNemar p = {p:.4f}")

    print("\neffect")
    up = int((gold.gain > 0).sum())
    print(f"  signed z increased for {up}/{len(gold)} terms "
          f"(binomial p = {binomtest(up, len(gold), 0.5).pvalue:.4f})")
    print(f"  median change {gold.gain.median():+.4f} "
          f"(Wilcoxon p = {wilcoxon(gold.gain).pvalue:.4f})")
    print(f"  mean |z|: {gold.z_pool.abs().mean():.3f} -> {gold.z_cond.abs().mean():.3f}")
    print(f"  specificity (mapped feature in top 30): "
          f"{int(gold.specific_pool.sum())} -> {int(gold.specific_cond.sum())}")

    sens_path = HERE / "results" / "feature_sensitivity.csv"
    column = SENSITIVITY_COLUMN[args.arm]
    if sens_path.exists():
        sens = pd.read_csv(sens_path)
        if column in sens.columns:
            m = gold.merge(sens[["feature", column]], on="feature").dropna(subset=[column])
            rho, p_rho = spearmanr(m[column], m.gain)
            hi = m[m[column] > m[column].median()]
            lo = m[m[column] <= m[column].median()]
            print("\nmechanism")
            print(f"  spearman(feature sensitivity, gain) rho = {rho:+.3f}  p = {p_rho:.4f}")
            print(f"  more sensitive half (n={len(hi)}): median gain {hi.gain.median():+.4f}")
            print(f"  less sensitive half (n={len(lo)}): median gain {lo.gain.median():+.4f}")
            print(f"  Mann-Whitney p = {mannwhitneyu(hi.gain, lo.gain).pvalue:.4f}")
        else:
            print(f"\nmechanism: {column} not in feature_sensitivity.csv, skipped")
    else:
        print("\nmechanism: run feature_sensitivity.py first, skipped")

    ratio = sd_ratio(args.arm)
    if ratio is not None:
        print("\nartifact")
        print(f"  median within-group / pooled reference SD = {ratio:.3f}  "
              f"(|z| inflates by ~{100*(1/ratio - 1):.1f}% mechanically)")

    out = base / "comparison.csv"
    gold.sort_values("gain", ascending=False).to_csv(out, index=False)
    print(f"\nper-term detail -> {out}")
    if n_g:
        print("\nnot reproduced -> reproduced:")
        print(gold[gained][["hpo_id", "hpo_name", "feature", "n_patients",
                            "z_pool", "z_cond", "q_pool", "q_cond"]]
              .to_string(index=False, float_format=lambda v: f"{v:.3g}"))
    if n_l:
        print("\nreproduced -> not reproduced:")
        print(gold[lost][["hpo_id", "hpo_name", "feature", "n_patients",
                          "z_pool", "z_cond", "q_pool", "q_cond"]]
              .to_string(index=False, float_format=lambda v: f"{v:.3g}"))


if __name__ == "__main__":
    main()
