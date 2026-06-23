"""Orchestrator: FCRM study steps 0-5 on a 100-patient gold test set.

Runs end-to-end, writing every artifact under a single timestamped run dir:
    results/hpo_predict_fcrm/<YYYY-MM-DD_HH-MM-SS>/

Reuses hpo_predict/{common,eval_protocol}.py (imported, not copied). Pure
numpy/pandas/sklearn. Deterministic given config.SEED. Operates per mean-pooled
PATIENT throughout; prints every actual count it computes.

    python experiments/hpo_predict_fcrm/run_fcrm.py
"""
import os
import sys
from datetime import datetime
from pathlib import Path

# this dir first (so `import config` -> OUR superset config), then the reused hpo_predict dir
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.append(str(_HERE.parent / "hpo_predict"))

import config  # noqa: E402
import fcrm_data  # noqa: E402
import step0_filter, step1_split, step2_disease_prior  # noqa: E402
import step3_perhpo, step4_beta_fmin, step5_evaluate  # noqa: E402


def main():
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = config.RESULTS_ROOT / config.EXP_NAME / ts
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[run] seed={config.SEED}  run_dir={run_dir}")

    # Step 0 — feature filter -> 83-HPO vocab
    filtered_dc, vocab = step0_filter.run(run_dir)

    # shared data (per patient)
    pooled, feats = fcrm_data.load_pooled()
    print(f"[run] pooled patients={len(pooled)}  feature cols={len(feats)}")

    # Step 1 — re-split + z-score base
    split, test_set, tune_ids, mean, std, z = step1_split.run(
        run_dir, pooled, feats, config.SEED
    )

    freq_table = __import__("common").disease_freq_table()
    test_pooled = pooled[pooled["patient_id"].isin(test_set)]
    tune_pooled = pooled[pooled["patient_id"].isin(set(tune_ids))]
    tune_long = fcrm_data.build_long(tune_pooled, z, filtered_dc, freq_table)
    test_long = fcrm_data.build_long(test_pooled, z, filtered_dc, freq_table)

    # Step 2 — disease-prior threshold
    f_thresh = step2_disease_prior.run(run_dir, tune_long)

    # Step 3 — per-HPO tau
    tau_map, perhpo_df = step3_perhpo.run(run_dir, tune_long)

    # Step 4 — FCRM beta / f_min
    beta, fmin = step4_beta_fmin.run(run_dir, tune_long, tau_map)

    # Step 5 — evaluate 3 methods on the 100 test patients
    comp_df = step5_evaluate.run(
        run_dir, test_pooled, test_long, tune_long, filtered_dc, vocab,
        f_thresh, tau_map, perhpo_df, beta, fmin,
    )

    print("\n[run] ===== FINAL SUMMARY =====")
    print(f"[run] vocab={len(vocab)} HPO | test={len(test_set)} | tune={len(tune_ids)} patients")
    print(f"[run] disease-prior f_thresh={f_thresh} | FCRM beta={beta}, f_min={fmin}")
    print(comp_df.to_string(index=False))
    print(f"[run] artifacts in {run_dir}")


if __name__ == "__main__":
    main()
