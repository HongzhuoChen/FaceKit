"""Phase 1.5 step 10 core: per-HPO thresholds + confidence weighting + z-clip.

Shared by BOTH methods so rule and learned stay on one decision path:
  * 1.5-C confidence down-weighting: MEDIUM-confidence direction codes have their score
    multiplied by config.MEDIUM_CONF_WEIGHT (weak proxies fire less).
  * 1.5-D extreme-z clip: a rule directional z with |z| > config.Z_CLIP is treated as a
    likely artifact and clipped to +/-Z_CLIP before ranking / thresholding. (The learned
    model clips its STANDARDIZED INPUTS to +/-Z_CLIP instead, in step5.)
  * 1.5-B per-HPO threshold: replace the single global tau / prob cutoff with one decision
    threshold per HPO, tuned on val by that HPO's TRUSTED-SUBSET F1 -- the same objective
    (HIGH-confidence direction codes intersected with high-freq HPO) the trusted-F1 headline
    reports, so tuning and metric agree. HPO with too few positive trusted val cells fall
    back to the pooled-trusted-subset-F1 global threshold.

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
    """Tune one threshold per HPO on val by the TRUSTED-SUBSET F1. Returns
    (per_hpo dict, sweep_df, global_thr).

    Objective matches the headline exactly (eval_protocol.trusted_subset_strict): the
    trusted subset is the set of cells where the HPO is HIGH-confidence AND the patient's
    disease annotates it at freq >= F_HIGH; within that subset positive = gold-present,
    negative = not-present. Cells outside the trusted subset are not used for tuning.

    Per-HPO threshold is the val-trusted-F1 argmax over `grid`, but only for HPO with
    >= config.MIN_TRUSTED_SUPPORT positive trusted val cells; sparser HPO fall back to
    `global_thr` (the threshold maximizing the POOLED trusted-subset F1) rather than a
    degenerate 0/1.
    """
    meta = val_meta.set_index("patient_id")
    f_high = config.F_HIGH

    # collect the trusted-subset cells (non-abstained val prediction rows only), labeled
    # gold-present(1)/not-present(0) -- identical membership to the trusted-F1 headline.
    rows = []
    for r in val_preds.itertuples(index=False):
        pid, h, sc = r.patient_id, r.hpo_id, r.score
        if pid not in meta.index:
            continue
        if conf.get(h) != "HIGH":
            continue  # only HIGH-confidence direction codes enter the trusted subset
        omim = meta.at[pid, "omim"]
        if freq_table.get(omim, {}).get(h, 0.0) < f_high:
            continue  # below high-freq -> not in the trusted subset
        lab = 1 if h in meta.at[pid, "present"] else 0
        rows.append((pid, h, float(sc), lab))
    cell = pd.DataFrame(rows, columns=["patient_id", "hpo_id", "score", "label"])

    # global fallback: pooled trusted-subset F1
    if len(cell) and (cell["label"] == 1).any() and (cell["label"] == 0).any():
        g_scores, g_labels = cell["score"].to_numpy(), cell["label"].to_numpy()
        global_thr = max(grid, key=lambda t: (_f1_at(g_scores, g_labels, t), -t))
    else:
        global_thr = float(min(grid))

    sweep, per_hpo = [], {}
    for h in vocab:
        sub = cell[cell["hpo_id"] == h]
        s, lab = sub["score"].to_numpy(), sub["label"].to_numpy()
        n_pos, n_neg = int((lab == 1).sum()), int((lab == 0).sum())
        if n_pos >= config.MIN_TRUSTED_SUPPORT:
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
