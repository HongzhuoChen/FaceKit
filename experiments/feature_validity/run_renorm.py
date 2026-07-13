"""Normalizer swap: is a failing feature broken because of its LANDMARKS, or
because of its DENOMINATOR?

Nearly every FaceKit feature is divided by bizygomatic width (`bizyg`) or face
height (`face_h`). Both denominators are themselves disease-dependent: a
syndrome that widens the face inflates `bizyg`, which silently cancels any
numerator that grows with it. And `face_h` is anchored on TRICHION (MediaPipe
index 10), a hairline point with no image evidence at all.

Both denominators can be recovered from the published feature CSV, so the swap
to an inter-ocular reference needs no re-extraction:

    inter_pupillary_distance = IPD / bizyg          =>  bizyg / IPD = 1 / ipd
    face_aspect_ratio        = bizyg / face_h       =>  face_h / IPD = 1 / (far * ipd)

Hence  X/bizyg -> X/IPD  is  f / ipd ;  X/bizyg^2 -> f / ipd^2 ;
       X/face_h -> X/IPD is  f / (far * ipd).

Run after run_validity.py; reuses its contrast machinery.
"""
from __future__ import annotations

import json
import numpy as np
import pandas as pd
from pathlib import Path

import run_validity as V

HERE = Path(__file__).resolve().parent


def renormalize(pat: pd.DataFrame, norm: dict[str, str]) -> tuple[pd.DataFrame, list[str]]:
    """Return a copy with bizyg/face_h-normalized features re-expressed per IPD."""
    out = pat.copy()
    ipd = out["inter_pupillary_distance"].replace(0, np.nan)
    far = out["face_aspect_ratio"].replace(0, np.nan)
    changed = []
    for col, kind in norm.items():
        if col not in out.columns:
            continue
        # IPD normalized by IPD is the constant 1 -- meaningless, leave it alone.
        if col in ("inter_pupillary_distance", "face_aspect_ratio"):
            continue
        if kind == "bizyg":
            out[col] = out[col] / ipd
        elif kind == "bizyg2":
            out[col] = out[col] / ipd**2
        elif kind == "face_h":
            out[col] = out[col] / (far * ipd)
        else:
            continue
        changed.append(col)
    return out, changed


def main() -> None:
    rng = np.random.default_rng(V.SEED)
    pat, vocab, fc, neg_ok = V.load()
    feat_cols = fc["feat"].tolist()
    norm = json.loads((HERE / "normalizers.json").read_text())

    # Hypertelorism cannot be expressed per-IPD (self-reference). The clinical
    # standard is the canthal index ICD/OCD, which is scale-free because both
    # terms already carry the same bizyg denominator.
    for d in (pat,):
        d["canthal_index"] = d["inter_canthal_distance"] / d["outer_canthal_distance"]
    feat_cols_ci = feat_cols + ["canthal_index"]

    ren, changed = renormalize(pat, norm)
    print(f"re-expressed {len(changed)} features per inter-pupillary distance")

    rows = []
    for _, v in vocab.iterrows():
        hpo, col, d = v["hpo_id"], v["csv_column"], int(v["expected_direction"])
        if col not in feat_cols or norm.get(col) not in ("bizyg", "bizyg2", "face_h"):
            continue

        def arm(src: pd.DataFrame, cols: list[str], target: str) -> dict:
            is_pos = src["present"].map(lambda s: hpo in s)
            pos = V.cap(src[is_pos], np.random.default_rng(V.SEED))
            neg = V.cap(
                src[(~is_pos) & (src["n_present"] >= 1) & src["disease"].isin(neg_ok[hpo])],
                np.random.default_rng(V.SEED),
            )
            if len(src[is_pos]) < V.MIN_POS or len(neg) < V.MIN_POS:
                return {}
            return V.contrast(pos, neg, target, d, cols)

        old = arm(pat, feat_cols, col)
        new = arm(ren, feat_cols, col)
        if not old or not new:
            continue
        r = dict(
            hpo_name=v["hpo_name"], feature=col, denom=norm[col],
            auc_bizyg=old["auc"], null_bizyg=old["null_pct"],
            auc_iod=new["auc"], null_iod=new["null_pct"],
            delta=new["auc"] - old["auc"],
        )
        if hpo == "HP:0000316":  # hypertelorism: also try the canthal index
            ci = arm(pat, feat_cols_ci, "canthal_index")
            r["canthal_index_auc"] = ci.get("auc")
        rows.append(r)

    res = pd.DataFrame(rows).sort_values("delta", ascending=False)
    res.to_csv(HERE / "results" / "renorm.csv", index=False)
    pd.set_option("display.width", 200)
    print()
    print(res.drop(columns=["canthal_index_auc"], errors="ignore").to_string(index=False))

    ci = res.get("canthal_index_auc")
    if ci is not None and ci.notna().any():
        row = res[res["canthal_index_auc"].notna()].iloc[0]
        print(f"\nHypertelorism, clinical canthal index (ICD/OCD): "
              f"AUC={row['canthal_index_auc']:.3f}  (mapped IPD/bizyg was {row['auc_bizyg']:.3f})")


if __name__ == "__main__":
    main()
