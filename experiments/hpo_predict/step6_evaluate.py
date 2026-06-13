"""Step 6: run BOTH methods through the single eval_protocol path.

Reads the prediction CSVs emitted by steps 4 (rule) and 5 (learned) and scores
them with the identical `eval_protocol.evaluate`. Writes one metrics JSON holding
naive-vs-arbitrated, per-term + macro ranking, trusted-subset strict, and (for the
learned model) calibration -- side by side for rule and learned.
"""
import json
import os

import pandas as pd

import common
import config
import eval_protocol


def run(run_dir) -> dict:
    table, _ = common.load_patient_table(run_dir)
    freq_table = common.disease_freq_table()
    conf = common.direction_conf_map()
    vocab = common.facial_vocab()

    def meta_for(split):
        s = table[table["split"] == split]
        return s[["patient_id", "omim", "present", "absent"]]

    jobs = [
        ("rule", "val", "rule_predictions_val.csv", False),
        ("rule", "test", "rule_predictions_test.csv", False),
        ("learned", "val", "learned_predictions_val.csv", True),
        ("learned", "test", "learned_predictions_test.csv", True),
    ]
    results: dict = {}
    for method, split, fname, is_prob in jobs:
        preds = pd.read_csv(run_dir / fname)
        m = eval_protocol.evaluate(preds, meta_for(split), freq_table, conf, vocab, is_prob=is_prob)
        results.setdefault(method, {})[split] = m

    (run_dir / "metrics.json").write_text(json.dumps(results, indent=2))

    def _summ(m):
        return {
            "macro_ap": round(m["ranking_primary"]["macro_ap"], 4),
            "random_ap": round(m["ranking_primary"]["random_macro_ap"], 4),
            "macro_auroc": round(m["ranking_primary"]["macro_auroc"], 4),
            "mrr": round(m["ranking_primary"]["mrr_gold"], 4),
            "naive_prec": round(m["arbitration"]["naive_precision"], 4),
            "arb_prec": round(m["arbitration"]["arbitrated_precision"], 4),
            "missed_hits": m["arbitration"]["missed_label_hits"],
            "trusted_f1": round(m["trusted_subset_strict"]["f1"], 4),
        }
    stats = {f"{meth}_test": _summ(results[meth]["test"]) for meth in ("rule", "learned")}
    print("[step6] verify (test split):")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    print(f"[step6] wrote {run_dir / 'metrics.json'}")
    return stats


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from pathlib import Path
    run(Path("/tmp/hpo_split_debug"))
