"""Step 7: compare rule vs learned under the identical protocol + figures.

Outputs in the run dir:
  - comparison.csv : key metrics, rule vs learned, on the test split
  - per_family.csv : per-feature-group gold-positive support + per-term macro-AP
  - calibration_learned.png : reliability curve + ECE for the learned model
  - bar_compare.png : rule vs learned ranking metrics
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import common
import config
import eval_protocol


def run(run_dir) -> dict:
    metrics = json.loads((run_dir / "metrics.json").read_text())

    # ---------- comparison table (test split) ----------
    rows = []
    for meth in ("rule", "learned"):
        m = metrics[meth]["test"]
        rows.append({
            "method": meth,
            "n_eval_patients": m["n_eval_patients"],
            "n_covered_patients": m["n_covered_patients"],
            "macro_ap": m["ranking_primary"]["macro_ap"],
            "random_macro_ap": m["ranking_primary"]["random_macro_ap"],
            "macro_auroc": m["ranking_primary"]["macro_auroc"],
            "micro_auroc": m["per_term"]["micro_auroc"],
            "mrr_gold": m["ranking_primary"]["mrr_gold"],
            "naive_precision": m["arbitration"]["naive_precision"],
            "arbitrated_precision": m["arbitration"]["arbitrated_precision"],
            "missed_label_hits": m["arbitration"]["missed_label_hits"],
            "naive_fp": m["arbitration"]["naive_fp"],
            "true_fp": m["arbitration"]["true_fp"],
            "trusted_f1": m["trusted_subset_strict"]["f1"],
            "trusted_precision": m["trusted_subset_strict"]["precision"],
            "trusted_recall": m["trusted_subset_strict"]["recall"],
            "calibration_ece": m.get("calibration", {}).get("ece", float("nan")),
        })
    comp = pd.DataFrame(rows)
    comp.to_csv(run_dir / "comparison.csv", index=False)

    # ---------- per-family breakdown (gold support by feature group) ----------
    vocab_df = pd.read_csv(config.VOCAB_CSV)  # already carries feature_group (step 2)
    per_family = vocab_df.groupby("feature_group").agg(
        n_terms=("hpo_id", "nunique"),
        gold_positive_images=("gold_positive_images", "sum"),
        attested_terms=("attested", "sum"),
    ).sort_values("gold_positive_images", ascending=False).reset_index()
    per_family.to_csv(run_dir / "per_family.csv", index=False)

    # ---------- calibration plot (learned) ----------
    table, feats = common.load_patient_table(run_dir)
    freq_table = common.disease_freq_table()
    conf = common.direction_conf_map()
    vocab = common.facial_vocab()
    test = table[table["split"] == "test"]
    preds = pd.read_csv(run_dir / "learned_predictions_test.csv")
    sc = preds.pivot(index="patient_id", columns="hpo_id", values="score").reindex(columns=vocab)
    meta = test.set_index("patient_id").loc[sc.index]
    ys, ps = [], []
    for pid, row in sc.iterrows():
        present = meta.at[pid, "present"]
        omim = meta.at[pid, "omim"]
        for j, h in enumerate(vocab):
            if h in present:
                ys.append(1); ps.append(row.iloc[j])
            elif freq_table.get(omim, {}).get(h, 0.0) <= config.F_LOW:
                ys.append(0); ps.append(row.iloc[j])
    ys, ps = np.array(ys, float), np.array(ps, float)
    bins = np.linspace(0, 1, 11)
    idx = np.clip(np.digitize(ps, bins) - 1, 0, 9)
    bx, by = [], []
    for b in range(10):
        m = idx == b
        if m.any():
            bx.append(ps[m].mean()); by.append(ys[m].mean())
    ece = metrics["learned"]["test"].get("calibration", {}).get("ece", float("nan"))
    plt.figure(figsize=(4, 4))
    plt.plot([0, 1], [0, 1], "--", color="gray", label="perfect")
    plt.plot(bx, by, "o-", label=f"learned (ECE={ece:.3f})")
    plt.xlabel("mean predicted prob"); plt.ylabel("empirical positive rate")
    plt.title("Learned model calibration (test, PU pairs)"); plt.legend(); plt.tight_layout()
    plt.savefig(run_dir / "calibration_learned.png", dpi=120); plt.close()

    # ---------- ranking comparison bar ----------
    plt.figure(figsize=(6, 4))
    labels = ["macro_ap", "macro_auroc", "mrr_gold"]
    x = np.arange(len(labels)); w = 0.35
    rule_v = [metrics["rule"]["test"]["ranking_primary"][k] for k in labels]
    learn_v = [metrics["learned"]["test"]["ranking_primary"][k] for k in labels]
    rand_v = [metrics["rule"]["test"]["ranking_primary"]["random_macro_ap"], 0.5, np.nan]
    plt.bar(x - w / 2, rule_v, w, label="rule")
    plt.bar(x + w / 2, learn_v, w, label="learned")
    plt.plot(x, rand_v, "kx", markersize=10, label="random ref")
    plt.xticks(x, labels); plt.ylabel("score"); plt.title("Rule vs learned (test)")
    plt.legend(); plt.tight_layout()
    plt.savefig(run_dir / "bar_compare.png", dpi=120); plt.close()

    print("[step7] comparison (test):")
    print(comp.to_string(index=False))
    print(f"[step7] wrote comparison.csv, per_family.csv, calibration_learned.png, bar_compare.png")
    return {"comparison_rows": len(comp), "families": len(per_family)}


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from pathlib import Path
    run(Path("/tmp/hpo_split_debug"))
