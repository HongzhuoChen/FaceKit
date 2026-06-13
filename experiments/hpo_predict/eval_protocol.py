"""Step 6 core: the SINGLE frequency-aware arbitration + metric path.

Both the rule baseline and the learned model are scored by `evaluate(...)` so the
metric logic is never duplicated. PU framing throughout:
  - gold `present` terms are trusted positives.
  - clean negatives = terms not present whose disease-level freq <= f_low
    (non-annotated terms have freq 0 -> clean negatives). Plausible-but-unlabeled
    terms (freq > f_low, not present) are DROPPED from negatives, not assumed false.
  - predicted-positive terms not in gold are arbitrated against disease freq:
    freq >= f_high & HIGH-confidence -> "gold likely-missing" (excluded from FP);
    freq <= f_low -> true FP; in-between -> grey zone (reported, not in headline).

Inputs
------
predictions : long DataFrame [patient_id, hpo_id, score, pred_pos(bool)]
patients_meta : DataFrame [patient_id, omim, present(set), absent(set)]
freq_table : {omim: {hpo: freq}}
direction_conf : {hpo: "HIGH"/"MEDIUM"/"LOW"}
vocab : list[str] (the 100 facial HPO, fixed column order)
is_prob : scores are calibrated probabilities -> also compute ECE
"""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

import config


def _freq(freq_table, omim, hpo):
    return freq_table.get(omim, {}).get(hpo, 0.0)


def _pivot(predictions, vocab, value, fill):
    p = predictions.pivot(index="patient_id", columns="hpo_id", values=value)
    return p.reindex(columns=vocab).fillna(fill)


def _ece(y_true, y_prob, n_bins=10):
    """Expected calibration error over a flat array of {0,1} labels and probs."""
    if len(y_true) == 0:
        return float("nan")
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(y_prob, bins) - 1, 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            continue
        ece += (m.mean()) * abs(y_true[m].mean() - y_prob[m].mean())
    return float(ece)


def evaluate(predictions, patients_meta, freq_table, direction_conf, vocab, is_prob=False):
    meta = patients_meta.set_index("patient_id")
    scores = _pivot(predictions, vocab, "score", fill=float("-inf"))
    preds = _pivot(predictions, vocab, "pred_pos", fill=False).astype(bool)
    # abstention (Phase 1.5): a (patient, HPO) cell with NO prediction row is abstained —
    # excluded from ranking, arbitration and trusted-subset scoring (never scored as a
    # negative). Phase-1 predictions cover every cell -> abstain all-False -> unchanged.
    ind = predictions.assign(_has=1).pivot(index="patient_id", columns="hpo_id", values="_has")
    abstain = ind.reindex(index=scores.index, columns=vocab).isna()
    # align meta to the patients we have predictions for
    meta = meta.loc[scores.index]

    f_high, f_low = config.F_HIGH, config.F_LOW

    # ---------- per-patient ranking (primary, PU-robust) ----------
    per_pt_auroc, per_pt_ap, recip_ranks, rand_ap = [], [], [], []
    # accumulators for per-term and micro
    micro_y, micro_s = [], []
    term_pos = {h: [] for h in vocab}   # per-term scores at positive patients
    term_neg = {h: [] for h in vocab}
    # calibration accumulators (PU-labeled pairs)
    cal_y, cal_p = [], []
    n_covered = 0

    for pid, row in scores.iterrows():
        omim = meta.at[pid, "omim"]
        present = meta.at[pid, "present"]
        s = row.to_numpy(dtype=float)
        ab = abstain.loc[pid].to_numpy()
        labels = np.full(len(vocab), -1, dtype=int)  # -1 = dropped/unlabeled/abstained
        for j, h in enumerate(vocab):
            if ab[j]:
                continue  # abstained cell -> not scored
            if h in present:
                labels[j] = 1
            elif _freq(freq_table, omim, h) <= f_low:
                labels[j] = 0
        pos_mask = labels == 1
        neg_mask = labels == 0
        if pos_mask.sum() == 0:
            continue  # uncovered patient -> excluded from ranking metrics
        n_covered += 1
        keep = pos_mask | neg_mask
        y, sc = labels[keep], s[keep]
        finite = np.isfinite(sc)
        y, sc = y[finite], sc[finite]
        if y.sum() > 0 and (y == 0).sum() > 0:
            per_pt_auroc.append(roc_auc_score(y, sc))
            per_pt_ap.append(average_precision_score(y, sc))
            rand_ap.append(y.mean())  # expected AP of a random ranker
        # MRR of gold positives among ranked vocab (all finite scores)
        order = np.argsort(-s)
        ranked = [vocab[k] for k in order if np.isfinite(s[k])]
        rr = 0.0
        for rank, h in enumerate(ranked, 1):
            if h in present:
                rr = 1.0 / rank
                break
        recip_ranks.append(rr)
        # per-term + micro + calibration accumulation
        for j, h in enumerate(vocab):
            if labels[j] == 1:
                term_pos[h].append(s[j]); micro_y.append(1); micro_s.append(s[j])
            elif labels[j] == 0:
                term_neg[h].append(s[j]); micro_y.append(0); micro_s.append(s[j])
            if is_prob and labels[j] in (0, 1):
                cal_y.append(labels[j]); cal_p.append(s[j])

    # macro per-term AUROC/AP
    term_auroc, term_ap = [], []
    for h in vocab:
        pos, neg = term_pos[h], term_neg[h]
        if len(pos) > 0 and len(neg) > 0:
            y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
            sc = np.r_[pos, neg]
            fin = np.isfinite(sc)
            if y[fin].sum() > 0 and (y[fin] == 0).sum() > 0:
                term_auroc.append(roc_auc_score(y[fin], sc[fin]))
                term_ap.append(average_precision_score(y[fin], sc[fin]))
    micro_y = np.array(micro_y); micro_s = np.array(micro_s)
    micro_fin = np.isfinite(micro_s)

    # ---------- arbitration of predicted positives (FP analysis) ----------
    tp = naive_fp = true_fp = missed = grey = fn = absent_fp = 0
    for pid, row in preds.iterrows():
        omim = meta.at[pid, "omim"]
        present = meta.at[pid, "present"]
        absent = meta.at[pid, "absent"]
        ab = abstain.loc[pid].to_numpy()
        pred_set = set()
        for j, h in enumerate(vocab):
            if ab[j]:
                continue  # abstained cell -> not scored
            if row.iloc[j]:  # predicted positive
                pred_set.add(h)
                if h in present:
                    tp += 1
                else:
                    naive_fp += 1
                    fr = _freq(freq_table, omim, h)
                    # gold ABSENT vetoes the "gold likely-missing" branch: a term gold
                    # explicitly marked absent is a true FP, never a missed-label hit.
                    if h in absent:
                        true_fp += 1
                        absent_fp += 1
                    elif fr >= f_high and direction_conf.get(h) == "HIGH":
                        missed += 1
                    elif fr <= f_low:
                        true_fp += 1
                    else:
                        grey += 1
        # FN over gold-present terms that were NOT abstained (abstained gold isn't scored)
        present_observed = {h for j, h in enumerate(vocab) if (h in present) and not ab[j]}
        fn += len(present_observed - pred_set)

    def _safe(a, b):
        return float(a / b) if b else float("nan")

    # ---------- trusted-subset strict eval (HIGH conf & freq>=f_high) ----------
    t_tp = t_fp = t_fn = 0
    for pid, row in preds.iterrows():
        omim = meta.at[pid, "omim"]
        present = meta.at[pid, "present"]
        ab = abstain.loc[pid].to_numpy()
        for j, h in enumerate(vocab):
            if ab[j]:
                continue  # abstained cell -> not scored
            if direction_conf.get(h) != "HIGH" or _freq(freq_table, omim, h) < f_high:
                continue
            pp, gp = bool(row.iloc[j]), (h in present)
            if pp and gp:
                t_tp += 1
            elif pp and not gp:
                t_fp += 1
            elif (not pp) and gp:
                t_fn += 1
    t_prec, t_rec = _safe(t_tp, t_tp + t_fp), _safe(t_tp, t_tp + t_fn)

    metrics = {
        "n_eval_patients": int(len(scores)),
        "n_covered_patients": n_covered,
        "ranking_primary": {
            "macro_auroc": _safe(sum(per_pt_auroc), len(per_pt_auroc)),
            "macro_ap": _safe(sum(per_pt_ap), len(per_pt_ap)),
            "random_macro_ap": _safe(sum(rand_ap), len(rand_ap)),
            "mrr_gold": _safe(sum(recip_ranks), len(recip_ranks)),
            "n_patients_scored": len(per_pt_auroc),
        },
        "per_term": {
            "macro_auroc": _safe(sum(term_auroc), len(term_auroc)),
            "macro_ap": _safe(sum(term_ap), len(term_ap)),
            "n_terms_scored": len(term_auroc),
            "micro_auroc": float(roc_auc_score(micro_y[micro_fin], micro_s[micro_fin]))
            if micro_fin.any() and micro_y[micro_fin].sum() > 0 and (micro_y[micro_fin] == 0).sum() > 0 else float("nan"),
            "micro_ap": float(average_precision_score(micro_y[micro_fin], micro_s[micro_fin]))
            if micro_fin.any() and micro_y[micro_fin].sum() > 0 else float("nan"),
        },
        "arbitration": {
            "tp": tp, "fn": fn,
            "naive_fp": naive_fp,
            "true_fp": true_fp,
            "missed_label_hits": missed,
            "grey_zone": grey,
            "naive_precision": _safe(tp, tp + naive_fp),
            "arbitrated_precision": _safe(tp, tp + true_fp),
            "recall": _safe(tp, tp + fn),
        },
        "absent_specificity": {
            "absent_anchor_fp": absent_fp,
            "note": "predicted-positive terms that gold explicitly marks ABSENT (clean FP, small N)",
        },
        "trusted_subset_strict": {
            "tp": t_tp, "fp": t_fp, "fn": t_fn,
            "precision": t_prec, "recall": t_rec,
            "f1": _safe(2 * t_prec * t_rec, t_prec + t_rec) if not (np.isnan(t_prec) or np.isnan(t_rec)) else float("nan"),
        },
    }
    if is_prob:
        metrics["calibration"] = {
            "ece": _ece(np.array(cal_y), np.array(cal_p)),
            "n_pairs": len(cal_y),
        }
    return metrics
