"""Step 3 [HOT]: per-HPO threshold tuning.

For each filtered HPO h on TUNE: signed_z = expected_direction * z[csv_column];
pos = h present; neg = h absent OR disease-freq(h) <= F_LOW. Sweep tau over
TAU_GRID, pick the tau maximizing that term's F1 (PU pos/neg). Fall back to
FALLBACK_TAU when the term has fewer than MIN_TUNE_POS tune positives.
"""
import numpy as np
import pandas as pd

import config


def run(run_dir, tune_long: pd.DataFrame):
    rows = []
    n_fallback = 0
    for hpo, g in tune_long.groupby("hpo_id"):
        gg = g[(g["pos"] | g["neg"])]
        pos = gg["pos"].to_numpy(bool)
        neg = gg["neg"].to_numpy(bool)
        sz = gg["signed_z"].to_numpy(dtype=float)
        n_pos = int(pos.sum())

        if n_pos < config.MIN_TUNE_POS:
            tau = config.FALLBACK_TAU
            used_fallback = True
            n_fallback += 1
        else:
            best_tau, best_f1 = config.FALLBACK_TAU, -1.0
            for tau in config.TAU_GRID:
                pred = sz >= tau
                tp = int((pos & pred).sum())
                fp = int((neg & pred).sum())
                fn = int((pos & ~pred).sum())
                prec = tp / (tp + fp) if tp + fp else 0.0
                rec = tp / (tp + fn) if tp + fn else 0.0
                f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
                if f1 > best_f1:
                    best_f1, best_tau = f1, tau
            tau = best_tau
            used_fallback = False
        rows.append((hpo, float(tau), n_pos, used_fallback))

    df = pd.DataFrame(rows, columns=["hpo_id", "tau", "n_pos_tune", "used_fallback"])
    df.to_csv(run_dir / "per_hpo_thresholds.csv", index=False)
    tau_map = dict(zip(df["hpo_id"], df["tau"]))

    print(f"[step3] tuned per-HPO tau for {len(df)} HPO; "
          f"{n_fallback} used fallback tau={config.FALLBACK_TAU} (<{config.MIN_TUNE_POS} tune pos)")
    print(f"[step3] wrote per_hpo_thresholds.csv")
    return tau_map, df
