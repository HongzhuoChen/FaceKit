"""Distance-based privacy statistics (pure numpy / scipy).

Ported from ``face_audit.calibrate_and_flag`` and ``face_audit.nnaa``.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
from scipy.spatial.distance import cdist


def drop_zero_rows(emb: np.ndarray, min_norm: float = 0.1) -> np.ndarray:
    """Remove all-zero rows, which mark images where no face was detected."""
    if len(emb) == 0:
        return emb
    return emb[np.linalg.norm(emb, axis=1) > min_norm]


def min_cross_distances(query: np.ndarray, ref: np.ndarray, metric: str = "cosine") -> np.ndarray:
    """Distance from each query row to its nearest ``ref`` row."""
    if len(query) == 0 or len(ref) == 0:
        return np.zeros(0)
    return cdist(query, ref, metric=metric).min(axis=1)


def percentile_rows(syn_min: np.ndarray, ho_min: np.ndarray, percentiles: List[int]) -> List[Dict]:
    """Flagging table: the threshold at ``p`` is the p-th percentile of the
    held-out nearest-neighbour distances, so ``heldout_pct`` is ``p`` by
    construction and ``synthetic_pct`` is the quantity of interest."""
    rows = []
    for p in percentiles:
        thr = float(np.percentile(ho_min, p))
        n_syn_below = int(np.sum(syn_min < thr))
        n_ho_below = int(np.sum(ho_min < thr))
        rows.append({
            "percentile": p,
            "threshold": thr,
            "synthetic_below": n_syn_below,
            "synthetic_total": int(len(syn_min)),
            "synthetic_pct": round(100.0 * n_syn_below / len(syn_min), 2) if len(syn_min) else 0.0,
            "heldout_below": n_ho_below,
            "heldout_total": int(len(ho_min)),
            "heldout_pct": round(100.0 * n_ho_below / len(ho_min), 2) if len(ho_min) else 0.0,
        })
    return rows


def adversarial_accuracy(A: np.ndarray, B: np.ndarray, metric: str = "cosine") -> float:
    """Nearest-neighbour adversarial accuracy AA(A, B) (Steier et al., 2025).

    For each point, is its nearest neighbour within its own set closer than
    its nearest neighbour in the other set? Averaged over both directions.
    0.5 means the sets are indistinguishable by proximity; below 0.5 means
    they are closer to each other than to themselves (memorization).
    """
    if len(A) < 2 or len(B) < 2:
        return float("nan")
    d_aa = cdist(A, A, metric=metric)
    np.fill_diagonal(d_aa, np.inf)
    d_bb = cdist(B, B, metric=metric)
    np.fill_diagonal(d_bb, np.inf)
    d_ab = cdist(A, B, metric=metric)
    a_score = np.mean(d_ab.min(axis=1) > d_aa.min(axis=1))
    b_score = np.mean(d_ab.min(axis=0) > d_bb.min(axis=1))
    return float(0.5 * (a_score + b_score))


def aa_from_min_dists(q2r: np.ndarray, q2q: np.ndarray, r2q: np.ndarray, r2r: np.ndarray) -> float:
    """AA from precomputed nearest-neighbour distances (used for LPIPS)."""
    return float(0.5 * (np.mean(q2r > q2q) + np.mean(r2q > r2r)))


def bootstrap_aa(A: np.ndarray, B: np.ndarray, metric: str, n_bootstrap: int,
                 rng: np.random.Generator) -> np.ndarray:
    """Resample both sets with replacement and recompute AA each time."""
    out = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        ia = rng.choice(len(A), size=len(A), replace=True)
        ib = rng.choice(len(B), size=len(B), replace=True)
        out[i] = adversarial_accuracy(A[ia], B[ib], metric=metric)
    return out


def bootstrap_aa_from_min_dists(q2r, q2q, r2q, r2r, n_bootstrap: int,
                                rng: np.random.Generator) -> np.ndarray:
    out = np.empty(n_bootstrap)
    n_q, n_r = len(q2r), len(r2q)
    for i in range(n_bootstrap):
        iq = rng.choice(n_q, size=n_q, replace=True)
        ir = rng.choice(n_r, size=n_r, replace=True)
        out[i] = 0.5 * (np.mean(q2r[iq] > q2q[iq]) + np.mean(r2q[ir] > r2r[ir]))
    return out


def ci_from_samples(samples: np.ndarray, alpha: float = 0.05) -> Dict[str, float]:
    return {
        "mean": float(np.mean(samples)),
        "median": float(np.median(samples)),
        "std": float(np.std(samples)),
        "ci_lower": float(np.percentile(samples, 100 * alpha / 2)),
        "ci_upper": float(np.percentile(samples, 100 * (1 - alpha / 2))),
    }


def nnaa_summary(bs_train: np.ndarray, bs_test: np.ndarray) -> Dict[str, Dict[str, float]]:
    """AA(train, synth), AA(test, synth) and privacy loss = AA(test) - AA(train)
    with bootstrap confidence intervals."""
    return {
        "aa_train_synth": ci_from_samples(bs_train),
        "aa_test_synth": ci_from_samples(bs_test),
        "privacy_loss": ci_from_samples(bs_test - bs_train),
    }
