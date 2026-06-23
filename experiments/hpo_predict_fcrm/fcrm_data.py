"""Shared FCRM data + PU-metric helpers.

Reuses `hpo_predict/common.py` for pooling / gold / freq tables (imported, not
copied). Everything here operates per MEAN-POOLED PATIENT (OPEN-5). PU framing
for all TUNE sweeps: pos = gold-present; neg = gold-absent OR disease-freq <= F_LOW.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

import common
import config


def load_pooled() -> tuple[pd.DataFrame, list[str]]:
    """One row per patient (all 3208): patient_id, disease, omim, present/absent sets, feats."""
    pheno = common.load_phenotypes()
    feats = common.feature_columns(pheno)
    pooled = common.pool_patients(pheno, gold=common.load_gold())
    pooled = pooled.rename(columns={"present_facial_hpo": "present", "absent_facial_hpo": "absent"})
    pooled["omim"] = pooled["disease"].map(common.disease_omim_map())
    return pooled, feats


def zscore_base_stats(pooled: pd.DataFrame, test_ids: set[int], feats: list[str]):
    """Population mean/std fit on ALL non-test patients (skipna); std floored to avoid /0."""
    base = pooled[~pooled["patient_id"].isin(test_ids)]
    mean = base[feats].mean(skipna=True)
    std = base[feats].std(ddof=0, skipna=True).replace(0.0, 1.0)
    return mean, std


def compute_z(pooled: pd.DataFrame, mean: pd.Series, std: pd.Series, feats: list[str]) -> pd.DataFrame:
    """Per-patient z-scored features, indexed by patient_id."""
    z = (pooled.set_index("patient_id")[feats] - mean) / std
    return z


def freq_of(freq_table: dict, omim, hpo: str) -> float:
    return float(freq_table.get(omim, {}).get(hpo, 0.0))


def build_long(sub: pd.DataFrame, z: pd.DataFrame, dc: pd.DataFrame, freq_table: dict) -> pd.DataFrame:
    """Long table over (patient in `sub`) x (filtered HPO in `dc`).

    Columns: patient_id, omim, hpo_id, signed_z, freq, pos, neg.
      signed_z = expected_direction * z[csv_column]
      pos = gold-present; neg = (not present) AND (gold-absent OR freq <= F_LOW)
    """
    base = sub[["patient_id", "omim", "present", "absent"]].reset_index(drop=True)
    pids = base["patient_id"].to_numpy()
    parts = []
    for r in dc.itertuples(index=False):
        col, direction, hpo = r.csv_column, int(r.expected_direction), r.hpo_id
        zc = z[col].reindex(pids).to_numpy(dtype=float)
        part = base.copy()
        part["hpo_id"] = hpo
        part["signed_z"] = direction * zc
        part["freq"] = [freq_of(freq_table, o, hpo) for o in base["omim"]]
        part["pos"] = [hpo in s for s in base["present"]]
        part["is_absent"] = [hpo in s for s in base["absent"]]
        parts.append(part)
    long = pd.concat(parts, ignore_index=True)
    long["neg"] = (~long["pos"]) & (long["is_absent"] | (long["freq"] <= config.F_LOW))
    return long.drop(columns=["present", "absent", "is_absent"])


def macro_f1(long: pd.DataFrame, pred_col: str) -> float:
    """Mean over HPO terms (>=1 tune positive) of the per-term F1 on PU pos/neg cells."""
    f1s = []
    for _, g in long.groupby("hpo_id"):
        gg = g[(g["pos"] | g["neg"])]
        if gg["pos"].sum() == 0:
            continue
        pred = gg[pred_col].to_numpy(bool)
        pos = gg["pos"].to_numpy(bool)
        neg = gg["neg"].to_numpy(bool)
        tp = int((pos & pred).sum())
        fp = int((neg & pred).sum())
        fn = int((pos & ~pred).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return float(np.mean(f1s)) if f1s else 0.0


def macro_ap(long: pd.DataFrame, score_col: str) -> float:
    """Mean over HPO terms (>=1 pos AND >=1 neg) of per-term average precision."""
    aps = []
    for _, g in long.groupby("hpo_id"):
        gg = g[(g["pos"] | g["neg"])]
        sc = gg[score_col].to_numpy(dtype=float)
        fin = np.isfinite(sc)
        y = gg["pos"].to_numpy(int)[fin]
        sc = sc[fin]
        if y.sum() > 0 and (y == 0).sum() > 0:
            aps.append(average_precision_score(y, sc))
    return float(np.mean(aps)) if aps else 0.0
