"""Step 4: rule baseline (no training).

Per feature column: z-score fit on TRAIN patients only. For each direction code
(hpo, csv_column, expected_direction): signed_z = direction * z[column].
Firing (OPEN-2, any-over-threshold): pred_pos = signed_z >= tau. Ranking score =
signed_z (directional z magnitude). Each facial HPO maps to exactly one column, so
"any" is over that single column. tau is swept on VAL (trusted-subset F1) over
TAU_GRID; ranking scores are tau-independent.
"""
import os

import numpy as np
import pandas as pd

import common
import config
import eval_protocol


def build_predictions(table, mean, std, dc, tau):
    """Long DataFrame [patient_id, hpo_id, score, pred_pos] for the given rows."""
    feats = list(mean.index)
    z = (table[feats] - mean) / std  # (P, F)
    rows = []
    for _, r in dc.iterrows():
        hpo, col, direction = r["hpo_id"], r["csv_column"], int(r["expected_direction"])
        signed = direction * z[col].to_numpy()  # (P,)
        for pid, sc in zip(table["patient_id"].to_numpy(), signed):
            rows.append((int(pid), hpo, float(sc), bool(sc >= tau)))
    return pd.DataFrame(rows, columns=["patient_id", "hpo_id", "score", "pred_pos"])


def run(run_dir) -> dict:
    table, feats = common.load_patient_table(run_dir)
    dc = common.load_direction_codes()
    freq_table = common.disease_freq_table()
    conf = common.direction_conf_map()
    vocab = common.facial_vocab()

    train = table[table["split"] == "train"]
    mean, std = common.zscore_stats(train[feats])
    pd.DataFrame({"mean": mean, "std": std}).to_csv(run_dir / "zscore_stats.csv")

    val = table[table["split"] == "val"]
    test = table[table["split"] == "test"]
    val_meta = val[["patient_id", "omim", "present", "absent"]]

    # --- sweep tau on val by trusted-subset F1 (threshold-dependent metric) ---
    sweep = []
    for tau in config.TAU_GRID:
        preds = build_predictions(val, mean, std, dc, tau)
        m = eval_protocol.evaluate(preds, val_meta, freq_table, conf, vocab)
        f1 = m["trusted_subset_strict"]["f1"]
        sweep.append((tau, f1, m["trusted_subset_strict"]["precision"], m["trusted_subset_strict"]["recall"]))
    sweep_df = pd.DataFrame(sweep, columns=["tau", "trusted_f1", "trusted_precision", "trusted_recall"])
    sweep_df.to_csv(run_dir / "rule_tau_sweep.csv", index=False)
    best = sweep_df.sort_values(["trusted_f1", "tau"], ascending=[False, True]).iloc[0]
    best_tau = float(best["tau"])

    # --- emit predictions at best tau (deterministic) ---
    build_predictions(val, mean, std, dc, best_tau).to_csv(run_dir / "rule_predictions_val.csv", index=False)
    build_predictions(test, mean, std, dc, best_tau).to_csv(run_dir / "rule_predictions_test.csv", index=False)
    pd.Series({"best_tau": best_tau, "selected_by": "val trusted-subset F1"}).to_json(run_dir / "rule_tau.json")

    stats = {
        "best_tau": best_tau,
        "val_trusted_f1": float(best["trusted_f1"]),
        "tau_grid": config.TAU_GRID,
        "n_val": int(len(val)), "n_test": int(len(test)),
    }
    print("[step4] verify:", stats)
    print("[step4] tau sweep:\n", sweep_df.to_string(index=False))
    return stats


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from pathlib import Path
    run(Path("/tmp/hpo_split_debug"))
