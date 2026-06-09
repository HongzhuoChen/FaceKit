"""Step 2: build the facial label space and the disease->HPO frequency table.

Outputs (committed label artifacts):
  - labels/facial_hpo_vocab.csv : the 100-term vocab + attested/zero-gold flag + counts
  - labels/disease_hpo_freq.csv : (omim, disease, hpo_id, freq_value) soft-label source
"""
import os

import pandas as pd

import common
import config


def run() -> dict:
    config.LABELS_DIR.mkdir(parents=True, exist_ok=True)
    dc = common.load_direction_codes()
    vocab = common.facial_vocab()  # 100 ids

    gold = common.load_gold()
    # per-term gold-positive (image-level) and absent (negative) counts
    pos_counts = {h: 0 for h in vocab}
    neg_counts = {h: 0 for h in vocab}
    for lst in gold["present_facial_hpo"]:
        for h in lst:
            pos_counts[h] += 1
    for lst in gold["absent_facial_hpo"]:
        for h in lst:
            neg_counts[h] += 1

    vdf = dc[["hpo_id", "hpo_name", "feature_group", "csv_column", "expected_direction", "confidence"]].copy()
    vdf = vdf.sort_values("hpo_id").reset_index(drop=True)
    vdf["gold_positive_images"] = vdf["hpo_id"].map(pos_counts)
    vdf["absent_negative_images"] = vdf["hpo_id"].map(neg_counts)
    vdf["attested"] = vdf["gold_positive_images"] > 0
    vdf.to_csv(config.VOCAB_CSV, index=False)

    # --- disease -> facial-HPO frequency table from hpoa ---
    omim_set = set(common.disease_omim_map().values())
    facial_set = set(vocab)
    hpoa = pd.read_csv(config.HPOA_PATH, sep="\t", comment="#", low_memory=False)
    sub = hpoa[hpoa["database_id"].isin(omim_set) & hpoa["hpo_id"].isin(facial_set)].copy()
    sub["freq_value"] = sub["frequency"].map(common.parse_frequency)
    # one disease may annotate an HPO several times -> keep the most informative (max)
    freq = sub.groupby(["database_id", "hpo_id"], as_index=False)["freq_value"].max()
    freq = freq.rename(columns={"database_id": "omim"})
    # attach disease name for readability
    inv = {v: k for k, v in common.disease_omim_map().items()}
    freq["disease"] = freq["omim"].map(inv)
    freq[["omim", "disease", "hpo_id", "freq_value"]].to_csv(config.DISEASE_FREQ_CSV, index=False)

    stats = {
        "vocab_size": len(vocab),
        "attested_terms": int(vdf["attested"].sum()),
        "zero_gold_terms": int((~vdf["attested"]).sum()),
        "diseases_covered_in_freq": int(freq["omim"].nunique()),
        "facial_hpo_in_freq": int(freq["hpo_id"].nunique()),
        "top_gold_positive": vdf.sort_values("gold_positive_images", ascending=False)
        .head(5).set_index("hpo_name")["gold_positive_images"].to_dict(),
        "terms_with_absent_negatives": int((vdf["absent_negative_images"] > 0).sum()),
    }
    print("[step2] verify:", stats)
    print(f"[step2] wrote {config.VOCAB_CSV} and {config.DISEASE_FREQ_CSV}")
    return stats


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    run()
