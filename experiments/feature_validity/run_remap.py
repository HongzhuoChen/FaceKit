"""Held-out confirmation of the re-mapping candidates.

run_validity.py reports, for every HPO term, the best-performing feature in the
term's anatomical region. Those AUCs are a max over 10-20 features, so they are
inflated by the winner's curse and cannot be trusted as-is.

Here the selection and the evaluation are separated:

  DISCOVERY half  -> pick the best feature in the region (the "candidate")
  HELD-OUT half   -> score that pre-committed feature, plus the currently mapped
                     feature, plus the placebo null. No selection happens here,
                     so the AUC is unbiased.

A re-mapping is CONFIRMED only if, on held-out data, the candidate beats the
mapped feature and clears the placebo null. Repeated over several splits so a
single lucky partition cannot carry a claim.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

import run_validity as V

HERE = Path(__file__).resolve().parent
N_SPLITS = 5
CONFIRM_AUC = 0.65     # held-out AUC the candidate must clear
CONFIRM_NULL = 95.0    # and it must beat this % of unrelated-region features
MIN_ARM = 15           # per split half, per arm


def split(pat: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Patient-level 50/50 split, stratified by disease."""
    rng = np.random.default_rng(seed)
    disc = []
    for _, sub in pat.groupby("disease"):
        idx = rng.permutation(len(sub))
        disc.append(sub.iloc[idx[: len(sub) // 2]])
    d = pd.concat(disc)
    return d, pat.drop(index=d.index)


def arms(src: pd.DataFrame, hpo: str, neg_ok: dict, seed: int):
    is_pos = src["present"].map(lambda s: hpo in s)
    rng = np.random.default_rng(seed)
    pos = V.cap(src[is_pos], rng)
    neg = V.cap(
        src[(~is_pos) & (src["n_present"] >= 1) & src["disease"].isin(neg_ok[hpo])],
        np.random.default_rng(seed),
    )
    return pos, neg


def best_in_region(pos, neg, region, feat_cols):
    best, best_auc = None, 0.5
    for c in feat_cols:
        if V.region_of(c) != region:
            continue
        for sgn in (1, -1):
            p = pos[c].to_numpy(dtype=float) * sgn
            n = neg[c].to_numpy(dtype=float) * sgn
            p, n = p[np.isfinite(p)], n[np.isfinite(n)]
            a = V.auc(p, n)
            if np.isfinite(a) and a > best_auc:
                best, best_auc, best_sgn = c, a, sgn
    return best, (best_sgn if best else 1)


def main() -> None:
    pat, vocab, fc, neg_ok = V.load()
    feat_cols = fc["feat"].tolist()

    # The clinical canthal index (ICD/OCD) is not among the 125 columns; both
    # terms carry the same bizyg denominator, so the ratio is scale-free.
    pat["canthal_index"] = pat["inter_canthal_distance"] / pat["outer_canthal_distance"]
    feat_cols = feat_cols + ["canthal_index"]

    rows = []
    for _, v in vocab.iterrows():
        hpo, mapped, d = v["hpo_id"], v["csv_column"], int(v["expected_direction"])
        if mapped not in feat_cols:
            continue
        region = V.region_of(mapped)

        per_split = []
        for s in range(N_SPLITS):
            disc, held = split(pat, 1000 + s)
            pd_, nd_ = arms(disc, hpo, neg_ok, 1000 + s)
            ph_, nh_ = arms(held, hpo, neg_ok, 2000 + s)
            if min(len(pd_), len(nd_), len(ph_), len(nh_)) < MIN_ARM:
                continue

            cand, sgn = best_in_region(pd_, nd_, region, feat_cols)   # selection: DISCOVERY only
            if cand is None:
                continue
            ch = V.contrast(ph_, nh_, cand, sgn, feat_cols)           # evaluation: HELD-OUT only
            mh = V.contrast(ph_, nh_, mapped, d, feat_cols)
            per_split.append(dict(cand=cand, auc_cand=ch["auc"], null_cand=ch["null_pct"],
                                  auc_mapped=mh["auc"], n_pos=len(ph_)))

        if not per_split:
            continue
        df = pd.DataFrame(per_split)
        top = df["cand"].mode().iloc[0]
        agree = (df["cand"] == top).mean()
        rows.append(dict(
            hpo_name=v["hpo_name"], region=region, mapped=mapped,
            candidate=top, cand_stability=round(agree, 2),
            auc_mapped_held=df["auc_mapped"].mean(),
            auc_cand_held=df["auc_cand"].mean(),
            null_cand_held=df["null_cand"].mean(),
            gain=df["auc_cand"].mean() - df["auc_mapped"].mean(),
            n_splits=len(df), n_pos_held=int(df["n_pos"].mean()),
        ))

    res = pd.DataFrame(rows)
    res["confirmed"] = (
        (res["auc_cand_held"] >= CONFIRM_AUC)
        & (res["null_cand_held"] >= CONFIRM_NULL)
        & (res["gain"] > 0.05)
        & (res["cand_stability"] >= 0.6)
    )
    res = res.sort_values("gain", ascending=False)
    res.to_csv(HERE / "results" / "remap_heldout.csv", index=False)

    pd.set_option("display.width", 220)
    cols = ["hpo_name", "mapped", "auc_mapped_held", "candidate", "cand_stability",
            "auc_cand_held", "null_cand_held", "gain", "n_pos_held"]
    print(f"terms evaluated: {len(res)}  |  CONFIRMED re-mappings: {res['confirmed'].sum()}\n")
    print("######## CONFIRMED")
    print(res[res["confirmed"]][cols].to_string(index=False))
    print("\n######## NOT CONFIRMED (top 12 by held-out gain)")
    print(res[~res["confirmed"]].head(12)[cols].to_string(index=False))


if __name__ == "__main__":
    main()
