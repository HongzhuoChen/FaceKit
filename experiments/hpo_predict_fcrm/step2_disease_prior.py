"""Step 2: disease-prior baseline threshold.

score(patient, hpo) = disease-level freq (0 if unannotated), no geometry.
Tune a single f_thresh over F_THRESH_GRID on TUNE by macro-F1 (PU pos/neg).
"""
import pandas as pd

import fcrm_data
import config


def run(run_dir, tune_long: pd.DataFrame):
    sweep = []
    for ft in config.F_THRESH_GRID:
        col = f"_pred_{ft}"
        tune_long[col] = tune_long["freq"] >= ft
        f1 = fcrm_data.macro_f1(tune_long, col)
        sweep.append((ft, f1))
        tune_long.drop(columns=[col], inplace=True)
    sweep_df = pd.DataFrame(sweep, columns=["f_thresh", "macro_f1"])
    sweep_df.to_csv(run_dir / "disease_prior_sweep.csv", index=False)

    best = sweep_df.sort_values(["macro_f1", "f_thresh"], ascending=[False, True]).iloc[0]
    best_ft = float(best["f_thresh"])
    print("[step2] disease-prior f_thresh sweep (macro-F1 on TUNE):")
    print(sweep_df.to_string(index=False))
    print(f"[step2] chosen f_thresh = {best_ft} (macro-F1 {best['macro_f1']:.4f})")
    return best_ft
