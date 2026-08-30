"""Does the choice of normative reference change the validity conclusions?

Section 3.1 scores every patient against one pooled FairFace reference. That
reference differs from the patient population in ways that are known to move
the geometric measurements:

  ancestry    the reference is 51% white, the patients 77% European
  age         the reference is a single age<20 pool, the cohorts' median ages
              span 1.1 to 15.0 years
  resolution  the patients are 224 px images, the reference 448 px ones

Each arm re-scores the same patients against a reference conditioned on one of
those variables and asks whether the verdicts move. Nothing else changes:
patients, features, statistics and thresholds all come from
``feature_validity/run_reference.py`` unchanged.

How the substitution works. ``run_reference.main`` computes

    z = (X_patients - mu_pooled) / sd_pooled

in one line. To feed it a conditioned z we hand it

    X' = mu_pooled + sd_pooled * z_conditioned

which that same line turns back into ``z_conditioned`` exactly. The scoring
loop, the t tests, the BH correction, the specificity ranks and the thresholds
are therefore the original code, not a reimplementation.

Both arms of a comparison always use the SAME patients: an arm that can only
score part of the cohort (ancestry labels are missing for 23% of patients, a
usable age for 44%) restricts the pooled arm to that same subset, so the only
thing that varies is the reference.

Usage:
    PYTHONPATH=src python experiments/reference_sensitivity/run_arms.py --arm ancestry
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "feature_validity"))

HERE = Path(__file__).resolve().parent
CHOP = Path(os.environ.get(
    "FACEKIT_CHOP_DIR",
    "/vast/projects/kai/multimodal-machine-learn/hongzhuo/chop_22q_analysis"))
GMDB_HF = "HzChen20/GMDB_enhanced_new_top100_filtered"

# GMDB ethnicity_category -> the three FairFace control groups. Unknown and
# Others have no counterpart in the reference and are dropped from both arms.
ETHNICITY_TO_GROUP = {"European": "white", "Asian": "asian", "African": "black"}

# FairFace records age as a band, so the patient side is bucketed to match.
AGE_BANDS = [(0, 3, "0-2"), (3, 10, "3-9"), (10, 20, "10-19")]


def gmdb_metadata() -> pd.DataFrame:
    """Patient-level age and ancestry from the GMDB HuggingFace release.

    age_year = 0 together with age_month = 0 is the dataset's missing-age
    sentinel, not a newborn: those rows are indistinguishable from the overall
    age mixture on every age-dependent measurement, whereas rows with
    age_year = 0 and age_month > 0 track the 1-year-olds as real infants would.
    """
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from datasets import load_dataset

    ds = load_dataset(GMDB_HF)["train"]
    keep = ["image_id", "patient_id", "age_year", "age_month", "ethnicity_category"]
    df = ds.remove_columns([c for c in ds.column_names if c not in keep]).to_pandas()

    missing_age = df.age_year.isna() | (
        (df.age_year == 0) & ((df.age_month == 0) | df.age_month.isna()))
    df["age"] = np.where(missing_age, np.nan,
                         df.age_year.fillna(0) + df.age_month.fillna(0) / 12)
    df["group"] = df.ethnicity_category.map(ETHNICITY_TO_GROUP)
    return df


def to_band(age: float) -> str | None:
    for lo, hi, name in AGE_BANDS:
        if lo <= age < hi:
            return name
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["ancestry", "age", "resolution"])
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--reference-224", type=Path,
                    default=HERE / "results" / "reference_224.csv")
    args = ap.parse_args()
    outdir = args.out or (HERE / "results" / args.arm)

    import run_reference as RR
    original_load = RR.load

    # --- what defines each patient's reference subgroup ------------------
    manifest = pd.read_csv(CHOP / "fairface_controls" / "manifest.csv")
    if args.arm == "ancestry":
        ref_key = manifest[["image_id", "race"]].rename(columns={"race": "key"})
        meta = gmdb_metadata()
        patient_key = (meta.dropna(subset=["group"]).groupby("patient_id").group
                       .agg(lambda s: s.mode().iloc[0] if len(s.mode()) else None))
    elif args.arm == "age":
        ref_key = manifest[["image_id", "age"]].rename(columns={"age": "key"})
        meta = gmdb_metadata()
        age = meta.dropna(subset=["age"]).groupby("patient_id").age.median()
        patient_key = age.map(to_band).dropna()
    else:
        ref_key = patient_key = None

    def make_load(conditioned: bool):
        def load():
            (pat, pat_disease, gold_pos, ref,
             vocab, freq, feat_cols) = original_load()

            if args.arm == "resolution":
                if not conditioned:
                    return pat, pat_disease, gold_pos, ref, vocab, freq, feat_cols
                alt = pd.read_csv(args.reference_224)
                alt = alt[alt.frontal_ok == True]  # noqa: E712
                return (pat, pat_disease, gold_pos, alt[feat_cols],
                        vocab, freq, feat_cols)

            # Restrict BOTH arms to patients whose subgroup is known.
            pat.index = pat.index.astype(patient_key.index.dtype)
            key = pd.Series(pat.index.map(patient_key), index=pat.index)
            known = key.notna()
            pat, pat_disease, key = pat[known], pat_disease[known], key[known]
            if not conditioned:
                return pat, pat_disease, gold_pos, ref, vocab, freq, feat_cols

            refdf = pd.read_csv(RR.REFERENCE)
            refdf = refdf[refdf.frontal_ok == True].merge(ref_key, on="image_id")  # noqa: E712
            mu_pooled, sd_pooled = ref.mean(), ref.std(ddof=1)
            out = pat.copy()
            for group in key.unique():
                subset = refdf[refdf.key == group][feat_cols]
                idx = key.index[key == group]
                z = (pat.loc[idx, feat_cols] - subset.mean()) / subset.std(ddof=1)
                out.loc[idx, feat_cols] = mu_pooled + sd_pooled * z
            return out, pat_disease, gold_pos, ref, vocab, freq, feat_cols
        return load

    for name, conditioned in [("pooled", False), ("conditioned", True)]:
        RR.load = make_load(conditioned)
        RR.OUT = outdir / name
        print("=" * 72)
        print(f"arm = {args.arm}   reference = {name}")
        print("=" * 72)
        RR.main()
        print()
    RR.load = original_load


if __name__ == "__main__":
    main()
