"""Step 4: FCRM beta / f_min sweep.

score = signed_z * freq^beta ; pred_pos = (signed_z >= tau_h) AND (freq >= f_min).
Sweep BETA_GRID x FMIN_GRID on TUNE (15 cells). Selection: primary macro-AP
(a function of beta via the score), tie-broken by macro-F1 (a function of f_min
via the prediction gate). beta=0, f_min=0 reduces to Rule+perHPO; this sweep
searches the f_min>0 / beta>0 cells.
"""
import numpy as np
import pandas as pd

import fcrm_data
import config


def run(run_dir, tune_long: pd.DataFrame, tau_map: dict):
    long = tune_long.copy()
    long["tau"] = long["hpo_id"].map(tau_map)
    fires = long["signed_z"].to_numpy(dtype=float) >= long["tau"].to_numpy(dtype=float)
    freq = long["freq"].to_numpy(dtype=float)
    sz = long["signed_z"].to_numpy(dtype=float)

    rows = []
    for beta in config.BETA_GRID:
        long["_score"] = sz * np.power(freq, beta)
        ap = fcrm_data.macro_ap(long, "_score")
        for fmin in config.FMIN_GRID:
            long["_pred"] = fires & (freq >= fmin)
            f1 = fcrm_data.macro_f1(long, "_pred")
            rows.append((beta, fmin, ap, f1))
    long.drop(columns=["_score", "_pred"], inplace=True, errors="ignore")

    df = pd.DataFrame(rows, columns=["beta", "f_min", "macro_ap", "macro_f1"])
    df.to_csv(run_dir / "beta_fmin_sweep.csv", index=False)

    best = df.sort_values(
        ["macro_ap", "macro_f1", "beta", "f_min"], ascending=[False, False, True, True]
    ).iloc[0]
    beta, fmin = float(best["beta"]), float(best["f_min"])
    print("[step4] beta x f_min sweep on TUNE (15 cells):")
    print(df.to_string(index=False))
    print(f"[step4] chosen beta={beta}, f_min={fmin} "
          f"(macro-AP {best['macro_ap']:.4f}, macro-F1 {best['macro_f1']:.4f})")
    print("[step4] note: beta=0, f_min=0 == Rule+perHPO")
    return beta, fmin
