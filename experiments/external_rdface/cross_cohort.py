"""Do independent patients with the same disease show the same deviation profile?

Three diseases appear in both RDFace and the 50 GMDB cohorts: Angelman,
Joubert and Mowat-Wilson. If the 120 measurements capture something real about
a disease rather than something about the GMDB collection, then a handful of
RDFace images of one of those diseases should look like the GMDB cohort of the
same disease, and not like the other 49.

The design avoids needing a precise estimate from 3 to 5 images. Each disease
is summarised by its mean z profile against the FairFace reference, and the
RDFace profile is correlated against all 50 GMDB cohort profiles. What is
tested is the RANK of the correct cohort, which is uniform on 1..50 under the
null however noisy the profile is.

RDFace carries no per-image age, so age cannot enter the comparison directly.
Its dataset-level average is 6.36 years against a GMDB median of 6.0, so the
two cohorts are close enough overall that a difference in age composition is
unlikely to drive the agreement; per-disease age is still uncontrolled.

Every patient group sits at a common offset from the healthy reference, because
patient photographs differ from Flickr controls in ways that have nothing to do
with the diagnosis. Left in, that offset makes every correlation high and the
ranking meaningless, so profiles are centred on each side: a GMDB cohort
against the mean of the 50, an RDFace disease against the mean of all RDFace
diseases. What remains is what distinguishes one disease from the average
patient, which is what the ranking should be based on.

Usage:
    PYTHONPATH=src python experiments/external_rdface/cross_cohort.py \
        --rdface <phenotypes_all.csv>
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
CHOP = Path(os.environ.get(
    "FACEKIT_CHOP_DIR",
    "/vast/projects/kai/multimodal-machine-learn/hongzhuo/chop_22q_analysis"))
GMDB = Path(os.environ.get(
    "FACEKIT_GMDB_PHENOTYPES",
    "/vast/projects/kai/multimodal-machine-learn/hongzhuo/mm_fusion_top50/"
    "combined_data/phenotypes_v2.csv"))

OVERLAP = {"ANG": "Angelman_syndrome",
           "JOUA": "Joubert_syndrome_1",
           "MOW": "Mowat-Wilson_syndrome"}
NON_FEATURE = {"disease", "image_id", "frontal_ok", "derotated",
               "pose_yaw", "pose_pitch", "pose_roll", "patient_id"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rdface", required=True, type=Path)
    ap.add_argument("--out-dir", type=Path, default=HERE / "results")
    args = ap.parse_args()

    ref = pd.read_csv(CHOP / "feats_fairface_v2" / "phenotypes_all.csv")
    ref = ref[ref.frontal_ok == True]  # noqa: E712
    features = [c for c in ref.columns if c not in NON_FEATURE]
    mu, sd = ref[features].mean(), ref[features].std(ddof=1)

    gm = pd.read_csv(GMDB)
    gm = gm[gm.frontal_ok == True]  # noqa: E712
    gm_profiles = ((gm[features] - mu) / sd).groupby(gm.disease.values).mean()

    rd = pd.read_csv(args.rdface)
    rd = rd[rd.frontal_ok == True]  # noqa: E712
    rd_profiles = ((rd[features] - mu) / sd).groupby(rd.disease.values).mean()

    rows = []
    for centred in (False, True):
        G = gm_profiles - gm_profiles.mean() if centred else gm_profiles
        R = rd_profiles - rd_profiles.mean() if centred else rd_profiles
        for abbr, cohort in OVERLAP.items():
            if abbr not in R.index or cohort not in G.index:
                continue
            corr = G.apply(lambda row: np.corrcoef(row.values, R.loc[abbr].values)[0, 1],
                           axis=1).sort_values(ascending=False)
            rank = int(list(corr.index).index(cohort)) + 1
            rows.append(dict(centred=centred, rdface=abbr, gmdb=cohort,
                             n_rdface=int((rd.disease == abbr).sum()),
                             n_gmdb=int((gm.disease == cohort).sum()),
                             r_true=float(corr[cohort]),
                             r_best=float(corr.iloc[0]), best=corr.index[0],
                             rank=rank, n_cohorts=len(corr)))

    out = pd.DataFrame(rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_dir / "cross_cohort.csv", index=False)

    for centred in (False, True):
        sub = out[out.centred == centred]
        if sub.empty:
            continue
        label = "centred on the average patient" if centred else "raw profiles"
        print(f"\n=== {label} ===")
        print(f"{'RDFace':>6} {'n':>3} {'GMDB cohort':>26} {'n':>4} "
              f"{'r':>7} {'rank':>6}  best match")
        for r in sub.itertuples():
            print(f"{r.rdface:>6} {r.n_rdface:>3} {r.gmdb:>26} {r.n_gmdb:>4} "
                  f"{r.r_true:+7.3f} {r.rank:>3}/{r.n_cohorts}  {r.best[:34]}")
        ranks = sub["rank"].to_numpy()
        n = int(sub.n_cohorts.iloc[0])
        # Under the null each rank is uniform on 1..n and the three diseases are
        # independent, so the chance of all three landing this high is the
        # product of their individual tail probabilities.
        p = float(np.prod(ranks / n))
        print(f"  ranks {list(ranks)} of {n}   combined p = {p:.4g}")


if __name__ == "__main__":
    main()
