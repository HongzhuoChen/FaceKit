"""Phase 1.5 step 10 core: per-HPO thresholds + confidence weighting + z-clip.

Shared by BOTH methods so rule and learned stay on one decision path:
  * 1.5-C confidence down-weighting: MEDIUM-confidence direction codes have their score
    multiplied by config.MEDIUM_CONF_WEIGHT (weak proxies fire less).
  * 1.5-D extreme-z clip: a rule directional z with |z| > config.Z_CLIP is treated as a
    likely artifact and clipped to +/-Z_CLIP before ranking / thresholding. (The learned
    model clips its STANDARDIZED INPUTS to +/-Z_CLIP instead, in step5.)
  * 1.5-B per-HPO threshold: replace the single global tau / prob cutoff with one decision
    threshold per HPO, tuned on val by that HPO's F1 (gold-present vs freq-screened clean
    negatives). HPO with no val signal fall back to the global trusted-subset-F1 threshold.

Abstained (patient, HPO) cells are simply absent from the prediction frame; nothing here
re-introduces them.
"""
import numpy as np
import pandas as pd

import config


def apply_score_transforms(preds: pd.DataFrame, conf: dict, is_prob: bool) -> pd.DataFrame:
    """Apply 1.5-D z-clip (rule scores only) and 1.5-C MEDIUM down-weight. Returns a copy."""
    out = preds.copy()
    if not is_prob:  # rule directional z: clip magnitude (learned clips its inputs in step5)
        out["score"] = out["score"].clip(lower=-config.Z_CLIP, upper=config.Z_CLIP)
    medium = {h for h, c in conf.items() if c == "MEDIUM"}
    is_med = out["hpo_id"].isin(medium)
    out.loc[is_med, "score"] = out.loc[is_med, "score"] * config.MEDIUM_CONF_WEIGHT
    return out


def _f1_at(scores, labels, thr):
    pred = scores >= thr
    tp = int((pred & (labels == 1)).sum())
    fp = int((pred & (labels == 0)).sum())
    fn = int((~pred & (labels == 1)).sum())
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    return (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0


def tune_per_hpo_thresholds(val_preds, val_meta, freq_table, conf, vocab, grid):
    """Tune one threshold per HPO on val. Returns (per_hpo dict, sweep_df, global_thr).

    per-HPO labels (PU): gold-present = 1; not-present & disease-freq <= F_LOW = 0; the
    rest are unlabeled and dropped. HPO lacking >=1 positive AND >=1 negative on val
    inherit `global_thr` (the threshold that maximizes the pooled trusted-subset F1).
    """
    meta = val_meta.set_index("patient_id")
    f_high, f_low = config.F_HIGH, config.F_LOW

    # attach per-cell (label, trusted?) to the prediction rows present (non-abstained)
    rows = []
    for r in val_preds.itertuples(index=False):
        pid, h, sc = r.patient_id, r.hpo_id, r.score
        if pid not in meta.index:
            continue
        present = meta.at[pid, "present"]
        omim = meta.at[pid, "omim"]
        fr = freq_table.get(omim, {}).get(h, 0.0)
        if h in present:
            lab = 1
        elif fr <= f_low:
            lab = 0
        else:
            lab = -1  # unlabeled
        trusted = (conf.get(h) == "HIGH") and (fr >= f_high)
        rows.append((pid, h, float(sc), lab, trusted))
    cell = pd.DataFrame(rows, columns=["patient_id", "hpo_id", "score", "label", "trusted"])

    # global fallback: pooled trusted-subset F1 (same criterion Phase-1 used for tau)
    tr = cell[cell["trusted"] & cell["label"].isin([0, 1])]
    if len(tr) and (tr["label"] == 1).any() and (tr["label"] == 0).any():
        g_scores, g_labels = tr["score"].to_numpy(), tr["label"].to_numpy()
        global_thr = max(grid, key=lambda t: (_f1_at(g_scores, g_labels, t), -t))
    else:
        global_thr = float(min(grid))

    sweep, per_hpo = [], {}
    for h in vocab:
        sub = cell[(cell["hpo_id"] == h) & cell["label"].isin([0, 1])]
        s, lab = sub["score"].to_numpy(), sub["label"].to_numpy()
        n_pos, n_neg = int((lab == 1).sum()), int((lab == 0).sum())
        if n_pos > 0 and n_neg > 0:
            thr = max(grid, key=lambda t: (_f1_at(s, lab, t), -t))
            f1 = _f1_at(s, lab, thr)
            source = "per_hpo"
        else:
            thr, f1, source = global_thr, float("nan"), "global_fallback"
        per_hpo[h] = float(thr)
        sweep.append((h, float(thr), f1, n_pos, n_neg, source))
    sweep_df = pd.DataFrame(sweep, columns=["hpo_id", "threshold", "val_f1", "n_pos", "n_neg", "source"])
    return per_hpo, sweep_df, float(global_thr)


def apply_per_hpo(preds: pd.DataFrame, per_hpo: dict) -> pd.DataFrame:
    """Set pred_pos = score >= per-HPO threshold. Returns a copy."""
    out = preds.copy()
    thr = out["hpo_id"].map(per_hpo)
    out["pred_pos"] = (out["score"] >= thr).fillna(False)
    return out
