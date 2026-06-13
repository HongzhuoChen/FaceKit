"""Step 3: patient-grouped train/val/test split, stratified by disease.

Split unit = patient (OPEN-5). Uses a LOCAL seeded numpy Generator (never global
RNG). Within each disease the patients are shuffled and partitioned by SPLIT_RATIOS,
so no patient (hence no image) leaks across folds.
"""
import os

import numpy as np
import pandas as pd

import common
import config


def make_split(patients: pd.DataFrame, seed: int) -> pd.DataFrame:
    """patients: one row per patient with `patient_id`, `disease`. Returns +`split`."""
    rng = np.random.default_rng(seed)
    r_train, r_val, _ = config.SPLIT_RATIOS
    assignments = []
    for disease, grp in patients.groupby("disease"):
        ids = grp["patient_id"].to_numpy().copy()
        rng.shuffle(ids)  # local generator
        n = len(ids)
        n_train = int(round(n * r_train))
        n_val = int(round(n * r_val))
        # guarantee at least one train patient when the disease is non-empty
        n_train = max(n_train, 1) if n > 0 else 0
        n_val = min(n_val, n - n_train)
        for i, pid in enumerate(ids):
            if i < n_train:
                s = "train"
            elif i < n_train + n_val:
                s = "val"
            else:
                s = "test"
            assignments.append((int(pid), disease, s))
    return pd.DataFrame(assignments, columns=["patient_id", "disease", "split"])


def run(run_dir) -> dict:
    pheno = common.load_phenotypes()
    gold = common.load_gold()
    patients = common.pool_patients(pheno, gold)[["patient_id", "disease"]]

    split = make_split(patients, config.SEED)
    out_path = run_dir / "split.csv"
    split.to_csv(out_path, index=False)

    # verify: no patient-id overlap across splits
    sets = {s: set(g["patient_id"]) for s, g in split.groupby("split")}
    inter = (sets.get("train", set()) & sets.get("val", set())) | \
            (sets.get("train", set()) & sets.get("test", set())) | \
            (sets.get("val", set()) & sets.get("test", set()))
    assert not inter, f"patient-id leak across splits: {sorted(inter)[:5]}"

    hist = split.groupby(["split", "disease"]).size().unstack(fill_value=0)
    counts = split["split"].value_counts().to_dict()
    n_disease_in_all = int((hist > 0).all(axis=0).sum()) if hist.shape[0] == 3 else 0
    stats = {
        "patients_total": int(len(split)),
        "split_counts": counts,
        "diseases_present_in_all_splits": n_disease_in_all,
        "n_diseases": int(split["disease"].nunique()),
        "patient_leak": len(inter),
    }
    print("[step3] verify:", stats)
    print("[step3] per-split disease histogram (head):")
    print(hist.T.head(10).to_string())
    print(f"[step3] wrote {out_path}")
    return stats


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from pathlib import Path
    d = Path("/tmp/hpo_split_debug"); d.mkdir(exist_ok=True)
    run(d)
