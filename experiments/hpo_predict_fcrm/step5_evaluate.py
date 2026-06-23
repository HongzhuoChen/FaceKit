"""Step 5 [HOT]: evaluate the 3 methods on the 100 test patients.

All three are scored through the EXISTING, unchanged hpo_predict/eval_protocol.evaluate.
Methods (long preds [patient_id, hpo_id, score, pred_pos]):
  - disease-prior : score = disease-freq ; pred_pos = freq >= f_thresh
  - Rule+perHPO   : score = signed_z     ; pred_pos = signed_z >= tau_h
  - FCRM          : score = signed_z*freq^beta ; pred_pos = (signed_z>=tau_h) AND (freq>=f_min)
"""
import json

import numpy as np
import pandas as pd

import common
import eval_protocol
import fcrm_data
import config


def _disease_prior_preds(test_long, f_thresh):
    df = test_long[["patient_id", "hpo_id", "freq"]].copy()
    df = df.rename(columns={"freq": "score"})
    df["pred_pos"] = df["score"] >= f_thresh
    return df


def _rule_perhpo_preds(test_long, tau_map):
    df = test_long[["patient_id", "hpo_id", "signed_z"]].copy()
    df = df.rename(columns={"signed_z": "score"})
    tau = test_long["hpo_id"].map(tau_map).to_numpy(dtype=float)
    df["pred_pos"] = test_long["signed_z"].to_numpy(dtype=float) >= tau
    return df


def _fcrm_preds(test_long, tau_map, beta, fmin):
    sz = test_long["signed_z"].to_numpy(dtype=float)
    freq = test_long["freq"].to_numpy(dtype=float)
    tau = test_long["hpo_id"].map(tau_map).to_numpy(dtype=float)
    df = test_long[["patient_id", "hpo_id"]].copy()
    df["score"] = sz * np.power(freq, beta)
    df["pred_pos"] = (sz >= tau) & (freq >= fmin)
    return df


def run(run_dir, test_pooled, test_long, tune_long, dc, vocab,
        f_thresh, tau_map, perhpo_df, beta, fmin):
    freq_table = common.disease_freq_table()
    conf = common.direction_conf_map()
    meta = test_pooled[["patient_id", "omim", "present", "absent"]].copy()

    methods = {
        "disease_prior": _disease_prior_preds(test_long, f_thresh),
        "rule_perhpo": _rule_perhpo_preds(test_long, tau_map),
        "fcrm": _fcrm_preds(test_long, tau_map, beta, fmin),
    }

    comparison = []
    for name, preds in methods.items():
        preds.to_csv(run_dir / f"predictions_{name}_test.csv", index=False)
        m = eval_protocol.evaluate(preds, meta, freq_table, conf, vocab)
        with open(run_dir / f"metrics_{name}.json", "w") as f:
            json.dump(m, f, indent=2)
        rp, ts = m["ranking_primary"], m["trusted_subset_strict"]
        comparison.append({
            "method": name,
            "macro_ap": rp["macro_ap"],
            "macro_auroc": rp["macro_auroc"],
            "mrr_gold": rp["mrr_gold"],
            "trusted_f1": ts["f1"],
            "trusted_precision": ts["precision"],
            "trusted_recall": ts["recall"],
            "n_covered_patients": m["n_covered_patients"],
        })

    comp_df = pd.DataFrame(comparison)
    comp_df.to_csv(run_dir / "comparison.csv", index=False)

    # FCRM per-HPO TUNE F1 at the final config
    tau = tune_long["hpo_id"].map(tau_map).to_numpy(dtype=float)
    tune_long = tune_long.copy()
    tune_long["_pred"] = (tune_long["signed_z"].to_numpy(dtype=float) >= tau) & \
                         (tune_long["freq"].to_numpy(dtype=float) >= fmin)
    perf = []
    for hpo, g in tune_long.groupby("hpo_id"):
        gg = g[(g["pos"] | g["neg"])]
        pos = gg["pos"].to_numpy(bool)
        pred = gg["_pred"].to_numpy(bool)
        neg = gg["neg"].to_numpy(bool)
        tp = int((pos & pred).sum())
        fp = int((neg & pred).sum())
        fn = int((pos & ~pred).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        perf.append((hpo, f1, prec, rec))
    perf_df = pd.DataFrame(perf, columns=["hpo_id", "tune_f1", "tune_precision", "tune_recall"])
    perf_df = perf_df.merge(perhpo_df, on="hpo_id", how="left")
    perf_df.to_csv(run_dir / "per_hpo_performance.csv", index=False)

    print("[step5] comparison (test, 100 patients):")
    print(comp_df.to_string(index=False))
    print(f"[step5] wrote comparison.csv, per-method predictions+metrics, per_hpo_performance.csv")
    return comp_df
