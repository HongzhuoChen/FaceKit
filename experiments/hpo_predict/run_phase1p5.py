"""Phase 1.5 orchestrator (steps 8-12).

Reuses the Phase-1 labels/ (steps 1-2, unchanged) and the SAME patient split (step 3,
re-derived deterministically from config.SEED -> asserted identical to Phase-1). Then,
through ONE shared gating + per-HPO-threshold + eval path, runs BOTH the rule baseline
and the learned model in TWO口径:
  - ungated  : Phase-1 features; apply 1.5-B/C/D only (head-to-head vs Phase 1).
  - gated    : additionally apply 1.5-A per-image gating (with its coverage cost).

Outputs under results/hpo_predict_phase1p5/<YYYY-MM-DD_HH-MM-SS>/.
"""
import json
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import common  # noqa: E402
import config  # noqa: E402
import eval_protocol  # noqa: E402
import gating  # noqa: E402
import report_p15  # noqa: E402
import step3_split  # noqa: E402
import step4_rule_baseline as step4  # noqa: E402
import step5_train_model as step5  # noqa: E402
import thresholds  # noqa: E402


def _assert_split_identity(run_dir):
    """Guard: the re-derived split must equal an existing Phase-1 split (fair before/after)."""
    import pandas as pd
    p1_root = config.RESULTS_ROOT / config.EXP_NAME
    cands = sorted(p1_root.glob("*/split.csv")) if p1_root.exists() else []
    if not cands:
        print("[split] no Phase-1 split found to compare against (deterministic re-derive only)")
        return
    ref = pd.read_csv(cands[-1])
    cur = pd.read_csv(run_dir / "split.csv")
    same = ref.sort_values("patient_id").reset_index(drop=True).equals(
        cur.sort_values("patient_id").reset_index(drop=True))
    assert same, f"re-derived split differs from Phase-1 {cands[-1]}"
    print(f"[split] identity asserted vs {cands[-1].parent.name}")


def _finalize(run_dir, method, kou, vp, tp, val_meta, test_meta, vocab, freq_table, conf, grid):
    """Shared per-HPO threshold tuning + apply + eval for one (method,口径)."""
    per_hpo, sweep_df, global_thr = thresholds.tune_per_hpo_thresholds(
        vp, val_meta, freq_table, conf, vocab, grid)
    sweep_df.to_csv(run_dir / f"{method}_{kou}_per_hpo_thresholds.csv", index=False)
    assert sweep_df["threshold"].notna().all() and len(sweep_df) == len(vocab), "missing per-HPO threshold"

    vp2 = thresholds.apply_per_hpo(vp, per_hpo)
    tp2 = thresholds.apply_per_hpo(tp, per_hpo)
    vp2.to_csv(run_dir / f"{method}_{kou}_predictions_val.csv", index=False)
    tp2.to_csv(run_dir / f"{method}_{kou}_predictions_test.csv", index=False)

    is_prob = method == "learned"
    mval = eval_protocol.evaluate(vp2, val_meta, freq_table, conf, vocab, is_prob=is_prob)
    mtest = eval_protocol.evaluate(tp2, test_meta, freq_table, conf, vocab, is_prob=is_prob)
    return {"val": mval, "test": mtest, "global_fallback_threshold": global_thr}


def main():
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = config.RESULTS_ROOT / config.EXP_NAME_P15 / ts
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"=== Phase-1.5 run: {run_dir} (seed={config.SEED}) ===")

    for f in (config.GOLD_CSV, config.VOCAB_CSV, config.DISEASE_FREQ_CSV, config.DISEASE_OMIM_CSV):
        assert f.exists(), f"missing Phase-1 label artifact (run Phase 1 first): {f}"

    # --- same split as Phase 1 (deterministic re-derive + identity guard) ---
    step3_split.run(run_dir)
    _assert_split_identity(run_dir)

    # --- shared resources ---
    dc = common.load_direction_codes()
    vocab = common.facial_vocab()
    freq_table = common.disease_freq_table()
    conf = common.direction_conf_map()

    # --- step 8: gating (verify + artifacts) ---
    gating.run(run_dir)

    # --- feature tables: ungated (Phase-1 identical pooling) + gated ---
    ung_table, feats = common.load_patient_table(run_dir)
    gat_table, _, _, _, _ = gating.load_gated_patient_table(run_dir)

    def meta_for(table, split):
        s = table[table["split"] == split]
        return s[["patient_id", "omim", "present", "absent"]]

    results: dict = {"rule": {}, "learned": {}}
    for kou, table in (("ungated", ung_table), ("gated", gat_table)):
        train = table[table["split"] == "train"]
        mean, std = common.zscore_stats(train[feats])
        val, test = table[table["split"] == "val"], table[table["split"] == "test"]
        val_meta, test_meta = meta_for(table, "val"), meta_for(table, "test")

        # rule baseline (1.5-C/D applied in build_rule_scores; abstain on NaN)
        vp = step4.build_rule_scores(val, mean, std, dc, conf)
        tp = step4.build_rule_scores(test, mean, std, dc, conf)
        results["rule"][kou] = _finalize(run_dir, "rule", kou, vp, tp,
                                         val_meta, test_meta, vocab, freq_table, conf, config.TAU_GRID)

        # learned model (NaN-capable; ungated has no NaN -> mask channel all-zero)
        lvp, ltp = step5.run_p15(run_dir, table, feats, vocab, freq_table, conf, dc, tag=kou)
        results["learned"][kou] = _finalize(run_dir, "learned", kou, lvp, ltp,
                                            val_meta, test_meta, vocab, freq_table, conf,
                                            config.PRED_POS_THRESHOLD_GRID)

    (run_dir / "metrics_p15.json").write_text(json.dumps(results, indent=2))
    print(f"[eval] wrote {run_dir / 'metrics_p15.json'}")

    # --- steps 11-12: dual-口径 comparison, coverage cost, artifact recheck, figures ---
    report_p15.run(run_dir, ung_table, gat_table, feats, dc, vocab, freq_table, conf)
    print(f"=== done -> {run_dir} ===")


if __name__ == "__main__":
    main()
