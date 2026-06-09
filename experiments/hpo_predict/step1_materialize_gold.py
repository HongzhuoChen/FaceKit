"""Step 1: materialize the static gold table from the HF dataset (read ONCE, offline).

Reconstructs the join key pat{patient_id}_img{image_id}, intersects
present_features / absent_features with the 100 facial direction-code HPO, and
writes labels/gold_hpo_facial.csv plus labels/disease_omim_map.csv.

All downstream code reads these CSVs; nothing else depends on `datasets`.
"""
import os

import pandas as pd

import common
import config


def run() -> dict:
    config.LABELS_DIR.mkdir(parents=True, exist_ok=True)

    facial = set(common.facial_vocab())  # 100 facial HPO ids

    # --- read HF gold once, offline cache is fine ---
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    from datasets import load_dataset

    hf = load_dataset(config.GOLD_HF_DATASET)["train"].to_pandas()
    hf["key"] = "pat" + hf["patient_id"].astype(int).astype(str) + "_img" + hf["image_id"].astype(int).astype(str)
    if hf["key"].duplicated().any():
        raise RuntimeError("Duplicate join keys in HF gold dataset; join would be ambiguous.")

    def _facial_codes(cell):
        codes = common.parse_hpo_list(cell)
        return sorted(set(codes) & facial)

    hf["present_facial"] = hf["present_features"].map(_facial_codes)
    hf["absent_facial"] = hf["absent_features"].map(_facial_codes)

    # --- restrict gold to the rows present in phenotypes.csv (join unit) ---
    pheno = pd.read_csv(config.PHENOTYPES_CSV, usecols=["disease", "image_id"], dtype={"image_id": str})
    merged = pheno.merge(
        hf[["key", "patient_id", "OMIM", "present_facial", "absent_facial"]],
        left_on="image_id", right_on="key", how="left",
    )
    n_join = int(merged["key"].notna().sum())
    if n_join != len(pheno):
        raise RuntimeError(
            f"Gold join incomplete: {n_join}/{len(pheno)} phenotypes rows matched HF gold. "
            "Cannot materialize gold reliably."
        )

    gold = pd.DataFrame({
        "image_id": merged["image_id"],
        "patient_id": merged["patient_id"].astype(int),
        "present_facial_hpo": merged["present_facial"].map(lambda v: ";".join(v)),
        "absent_facial_hpo": merged["absent_facial"].map(lambda v: ";".join(v)),
    })
    gold.to_csv(config.GOLD_CSV, index=False)

    # --- disease_name -> OMIM (verified 1:1) so downstream needs no HF ---
    omap = merged.dropna(subset=["OMIM"]).copy()
    omap["omim"] = "OMIM:" + omap["OMIM"].astype(int).astype(str)
    dmap = omap.groupby("disease")["omim"].agg(lambda s: sorted(set(s)))
    bad = dmap[dmap.map(len) != 1]
    if len(bad):
        raise RuntimeError(f"{len(bad)} disease(s) map to !=1 OMIM; cannot build disease_omim_map.")
    pd.DataFrame({"disease": dmap.index, "omim": dmap.map(lambda s: s[0]).values}).to_csv(
        config.DISEASE_OMIM_CSV, index=False
    )

    # --- verify ---
    n_present = int((merged["present_facial"].map(len) > 0).sum())
    n_absent = int((merged["absent_facial"].map(len) > 0).sum())
    attested = sorted({c for lst in merged["present_facial"] for c in lst})
    stats = {
        "join": f"{n_join}/{len(pheno)}",
        "rows_with_present_facial": n_present,
        "present_coverage_pct": round(100 * n_present / len(pheno), 1),
        "rows_with_absent_facial": n_absent,
        "distinct_facial_terms_attested": len(attested),
        "n_diseases": int(dmap.shape[0]),
    }
    print("[step1] verify:", stats)
    print(f"[step1] wrote {config.GOLD_CSV} and {config.DISEASE_OMIM_CSV}")
    return stats


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    run()
