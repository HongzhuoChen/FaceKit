"""Phase 1.5 steps 11-12: dual-口径 comparison + coverage cost + artifact recheck.

Reads metrics_p15.json and the prediction CSVs and emits:
  - comparison_p15.csv     : rule/learned x ungated/gated test metrics, + Phase-1 reference.
  - coverage_delta.csv      : per-term gold-positive support ungated vs gated (test) + totals.
  - artifact_recheck.csv    : before (Phase-1 raw z) vs after (gated) for the two named FPs.
  - top_fp_recheck.csv       : top rule FPs Phase-1-raw vs gated (artifacts should be gone).
  - bar_compare_p15.png      : ungated vs gated macro-AP / trusted-F1 for both methods.
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import common
import config


def _summ(m):
    return {
        "n_eval_patients": m["n_eval_patients"],
        "n_covered_patients": m["n_covered_patients"],
        "macro_ap": round(m["ranking_primary"]["macro_ap"], 4),
        "macro_auroc": round(m["ranking_primary"]["macro_auroc"], 4),
        "mrr_gold": round(m["ranking_primary"]["mrr_gold"], 4),
        "naive_precision": round(m["arbitration"]["naive_precision"], 4),
        "arbitrated_precision": round(m["arbitration"]["arbitrated_precision"], 4),
        "trusted_precision": round(m["trusted_subset_strict"]["precision"], 4),
        "trusted_recall": round(m["trusted_subset_strict"]["recall"], 4),
        "trusted_f1": round(m["trusted_subset_strict"]["f1"], 4),
    }


def _phase1_reference():
    """Latest Phase-1 metrics.json test summaries (rule/learned), for head-to-head."""
    root = config.RESULTS_ROOT / config.EXP_NAME
    cands = sorted(root.glob("*/metrics.json")) if root.exists() else []
    if not cands:
        return {}
    m = json.loads(cands[-1].read_text())
    return {meth: _summ(m[meth]["test"]) for meth in ("rule", "learned")}


def _raw_rule_fp(test_table, mean, std, dc, vocab, freq_table, conf, tau=1.0):
    """Phase-1-style raw rule FPs (no z-clip, no down-weight): long FP table by score."""
    feats = list(mean.index)
    z = (test_table[feats] - mean) / std
    meta = test_table.set_index("patient_id")
    rows = []
    for r in dc.itertuples(index=False):
        hpo, col, direction = r.hpo_id, r.csv_column, int(r.expected_direction)
        signed = direction * z[col].to_numpy()
        for pid, sc in zip(test_table["patient_id"].to_numpy(), signed):
            if np.isnan(sc) or sc < tau:
                continue
            present = meta.at[pid, "present"]
            if hpo in present:
                continue  # gold positive -> not an FP
            rows.append((int(pid), hpo, conf.get(hpo), round(float(sc), 3)))
    return pd.DataFrame(rows, columns=["patient_id", "hpo_id", "confidence", "raw_z"]).sort_values(
        "raw_z", ascending=False).reset_index(drop=True)


def run(run_dir, ung_table, gat_table, feats, dc, vocab, freq_table, conf):
    metrics = json.loads((run_dir / "metrics_p15.json").read_text())

    # ---------- comparison table ----------
    rows = []
    for meth in ("rule", "learned"):
        for kou in ("ungated", "gated"):
            rows.append({"method": meth, "kou": kou, **_summ(metrics[meth][kou]["test"])})
    ref = _phase1_reference()
    for meth, s in ref.items():
        rows.append({"method": meth, "kou": "phase1_ref", **s})
    comp = pd.DataFrame(rows)
    comp.to_csv(run_dir / "comparison_p15.csv", index=False)

    # ---------- coverage delta (test) ----------
    hpo_col = dict(zip(dc["hpo_id"], dc["csv_column"]))
    ung_test = ung_table[ung_table["split"] == "test"].set_index("patient_id")
    gat_test = gat_table[gat_table["split"] == "test"].set_index("patient_id")
    cov_rows = []
    for h in vocab:
        col = hpo_col[h]
        n_ung = int(sum(h in p for p in ung_test["present"]))
        # gated positive support: patient in gated table, gold-present, source feature observed
        n_gat = 0
        for pid, prow in gat_test.iterrows():
            if h in prow["present"] and not pd.isna(prow[col]):
                n_gat += 1
        cov_rows.append((h, col, n_ung, n_gat, n_ung - n_gat))
    cov = pd.DataFrame(cov_rows, columns=["hpo_id", "csv_column", "n_pos_ungated", "n_pos_gated", "delta"])
    cov.to_csv(run_dir / "coverage_delta.csv", index=False)
    coverage_totals = {
        "n_test_patients_ungated": int(len(ung_test)),
        "n_test_patients_gated": int(len(gat_test)),
        "n_test_patients_dropped": int(len(ung_test) - len(gat_test)),
        "n_covered_rule_ungated": metrics["rule"]["ungated"]["test"]["n_covered_patients"],
        "n_covered_rule_gated": metrics["rule"]["gated"]["test"]["n_covered_patients"],
        "n_covered_learned_ungated": metrics["learned"]["ungated"]["test"]["n_covered_patients"],
        "n_covered_learned_gated": metrics["learned"]["gated"]["test"]["n_covered_patients"],
        "total_pos_support_ungated": int(cov["n_pos_ungated"].sum()),
        "total_pos_support_gated": int(cov["n_pos_gated"].sum()),
    }

    # ---------- artifact recheck (the two named Phase-1 top-FP cases) ----------
    mean_u, std_u = common.zscore_stats(ung_table[ung_table["split"] == "train"][feats])
    rule_gated = pd.read_csv(run_dir / "rule_gated_predictions_test.csv")
    art_rows = []
    for c in config.ARTIFACT_CASES:
        pid, hpo = c["patient_id"], c["hpo_id"]
        col = hpo_col[hpo]
        direction = int(dc.set_index("hpo_id").at[hpo, "expected_direction"])
        # before: Phase-1 ungated raw signed z (no clip / no down-weight)
        before_z = float("nan")
        if pid in ung_table["patient_id"].values:
            xu = ung_table.set_index("patient_id").at[pid, col]
            before_z = direction * (xu - mean_u[col]) / std_u[col]
        # after: gated value + whether the cell survived (abstained if masked)
        gat_idx = gat_table.set_index("patient_id")
        after_val = gat_idx.at[pid, col] if pid in gat_idx.index else float("nan")
        abstained = bool(pd.isna(after_val))
        g = rule_gated[(rule_gated["patient_id"] == pid) & (rule_gated["hpo_id"] == hpo)]
        after_score = float("nan") if abstained or g.empty else float(g.iloc[0]["score"])
        after_pred = (not g.empty) and bool(g.iloc[0]["pred_pos"])
        art_rows.append({
            "patient_id": pid, "hpo_id": hpo, "hpo_name": c["hpo_name"],
            "csv_column": col, "gate": c["gate"],
            "before_raw_z": round(before_z, 3), "before_pred_pos": bool(before_z >= 1.0),
            "after_gated_abstained": abstained,
            "after_score": round(after_score, 3) if not np.isnan(after_score) else None,
            "after_pred_pos": after_pred,
        })
    art = pd.DataFrame(art_rows)
    art.to_csv(run_dir / "artifact_recheck.csv", index=False)

    # ---------- top-FP recheck: Phase-1-raw rule FPs vs gated rule FPs ----------
    raw_fp = _raw_rule_fp(ung_table[ung_table["split"] == "test"], mean_u, std_u, dc, vocab, freq_table, conf)
    raw_fp.head(15).to_csv(run_dir / "top_fp_phase1_raw.csv", index=False)
    art_pairs = {(c["patient_id"], c["hpo_id"]) for c in config.ARTIFACT_CASES}
    top10_pairs = set(zip(raw_fp.head(10)["patient_id"], raw_fp.head(10)["hpo_id"]))
    artifacts_in_top10_raw = sorted(art_pairs & top10_pairs)

    # gated rule FPs ranked by transformed score
    gmeta = gat_table[gat_table["split"] == "test"].set_index("patient_id")
    gfp = rule_gated[rule_gated["pred_pos"]].copy()
    gfp = gfp[[h not in gmeta.at[pid, "present"] for pid, h in zip(gfp["patient_id"], gfp["hpo_id"])]]
    gfp = gfp.sort_values("score", ascending=False).reset_index(drop=True)
    gfp.head(15).to_csv(run_dir / "top_fp_gated.csv", index=False)
    gated_top_pairs = set(zip(gfp.head(15)["patient_id"], gfp.head(15)["hpo_id"]))
    artifacts_in_top_gated = sorted(art_pairs & gated_top_pairs)

    # ---------- figure ----------
    plt.figure(figsize=(7, 4))
    labels, x, w = ["macro_ap", "trusted_f1"], np.arange(2), 0.2
    for k, (meth, kou) in enumerate([("rule", "ungated"), ("rule", "gated"),
                                     ("learned", "ungated"), ("learned", "gated")]):
        s = _summ(metrics[meth][kou]["test"])
        plt.bar(x + (k - 1.5) * w, [s["macro_ap"], s["trusted_f1"]], w, label=f"{meth}/{kou}")
    plt.xticks(x, labels); plt.ylabel("score (test)")
    plt.title("Phase 1.5 — ungated vs gated"); plt.legend(fontsize=7); plt.tight_layout()
    plt.savefig(run_dir / "bar_compare_p15.png", dpi=120); plt.close()

    print("\n[report] comparison (test):")
    print(comp.to_string(index=False))
    print("\n[report] coverage totals:", coverage_totals)
    print("\n[report] artifact recheck:")
    print(art.to_string(index=False))
    print(f"\n[report] artifacts in Phase-1-raw top-10 FP: {artifacts_in_top10_raw}")
    print(f"[report] artifacts in gated top-15 FP:        {artifacts_in_top_gated}")
    (run_dir / "report_summary.json").write_text(json.dumps({
        "coverage_totals": coverage_totals,
        "artifacts_in_top10_raw_fp": [list(p) for p in artifacts_in_top10_raw],
        "artifacts_in_top15_gated_fp": [list(p) for p in artifacts_in_top_gated],
    }, indent=2))
    return {"coverage_totals": coverage_totals,
            "artifacts_gone_under_gating": len(artifacts_in_top_gated) == 0}


if __name__ == "__main__":
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
