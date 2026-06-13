"""Step 5: weak-supervision learned model.

Small MLP head: standardized geometric features -> per-HPO probability over the
100-term facial vocab. Soft targets = disease-level hpoa freq_value of the patient's
disease (0 if the disease does not annotate the term). Loss = soft BCE (with a global
pos_weight to counter the sparse/low-frequency targets). All patients train (incl.
the 63.4% with no facial gold positive — they still carry disease soft labels).

Reproducibility: local seeded torch + numpy generators for init and the loader; no
global RNG reliance. Device/dtype derived from availability and applied explicitly.
"""
import json
import math
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import common
import config
import eval_protocol


class MLPHead(nn.Module):
    def __init__(self, in_dim: int, n_out: int, gen: torch.Generator, hidden: int = 128, p_drop: float = 0.3):
        super().__init__()
        self.p_drop = p_drop
        self._gen = gen  # local RNG for init + dropout (no global torch RNG touched)
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, n_out)

    def reset_parameters(self):
        # replicate default nn.Linear init (U(-1/sqrt(fan_in), 1/sqrt(fan_in))) via the
        # local generator so weight init draws no global RNG state.
        for lin in (self.fc1, self.fc2):
            bound = 1.0 / math.sqrt(lin.in_features)
            with torch.no_grad():
                lin.weight.uniform_(-bound, bound, generator=self._gen)
                lin.bias.uniform_(-bound, bound, generator=self._gen)

    def forward(self, x):  # x: (B, in_dim) -> logits (B, n_out)
        h = torch.relu(self.fc1(x))
        if self.training and self.p_drop > 0.0:
            keep = 1.0 - self.p_drop
            mask = torch.rand(h.shape, generator=self._gen, device=h.device, dtype=h.dtype) < keep
            h = h * mask / keep
        return self.fc2(h)


def _soft_targets(table, vocab, freq_table):
    """(P, |vocab|) float target = freq_value of patient's disease per term, else 0."""
    Y = np.zeros((len(table), len(vocab)), dtype=np.float32)
    col = {h: j for j, h in enumerate(vocab)}
    for i, omim in enumerate(table["omim"].to_numpy()):
        for h, fr in freq_table.get(omim, {}).items():
            Y[i, col[h]] = fr
    return Y


def run_p15(run_dir, table, feats, vocab, freq_table, conf, dc, tag,
            n_epochs: int = 300, lr: float = 1e-3, weight_decay: float = 1e-4):
    """Phase 1.5 NaN-capable learned model. Returns (val_preds, test_preds) long frames
    with transformed `score` (prob, MEDIUM down-weighted); pred_pos set later by the
    per-HPO thresholds. `tag` in {"ungated","gated"} only labels saved artifacts.

    NaN strategy (PLAN 1.5-B):
      - standardize on TRAIN NaN-aware mean/std; clip standardized inputs to +/-Z_CLIP
        (1.5-D); impute NaN -> 0 (= train z-mean) AND append a per-feature missingness
        channel so the model knows a value was gated, not observed. Input dim = 2*F.
      - loss is masked at (patient, HPO) cells whose source column is NaN (never supervise
        an abstained cell); inference ABSTAINS on those cells (no row emitted).
    """
    import thresholds

    loader_gen = torch.Generator().manual_seed(config.SEED)
    train = table[table["split"] == "train"].reset_index(drop=True)
    val = table[table["split"] == "val"].reset_index(drop=True)
    test = table[table["split"] == "test"].reset_index(drop=True)

    mean, std = common.zscore_stats(train[feats])  # NaN-aware (pandas skipna)
    # each vocab HPO maps 1:1 to a single csv_column -> its feature index for abstention
    hpo_col = dict(zip(dc["hpo_id"], dc["csv_column"]))
    feat_idx = {c: k for k, c in enumerate(feats)}
    hpo_fidx = np.array([feat_idx[hpo_col[h]] for h in vocab])  # (V,)

    def _build(df):
        z = ((df[feats] - mean) / std).clip(lower=-config.Z_CLIP, upper=config.Z_CLIP)
        miss = z.isna().to_numpy().astype(np.float32)        # (P, F) 1=gated/missing
        zf = z.fillna(0.0).to_numpy(dtype=np.float32)        # impute to train z-mean (0)
        X = torch.tensor(np.concatenate([zf, miss], axis=1)) # (P, 2F)
        abstain = miss[:, hpo_fidx].astype(bool)             # (P, V) cell abstained?
        return X, abstain

    Xtr, ab_tr = _build(train)
    Xval, ab_val = _build(val)
    Xte, ab_te = _build(test)
    Ytr = torch.tensor(_soft_targets(train, vocab, freq_table))
    cellmask_tr = torch.tensor((~ab_tr).astype(np.float32))  # 1 = supervise this cell

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    init_gen = torch.Generator(device=device).manual_seed(config.SEED)
    model = MLPHead(2 * len(feats), len(vocab), init_gen).to(device)
    model.reset_parameters()
    pos_rate = float(Ytr.mean().clamp(min=1e-4))
    pos_weight = torch.tensor((1.0 - pos_rate) / pos_rate, device=device).clamp(max=50.0)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    ds = TensorDataset(Xtr, Ytr, cellmask_tr)
    loader = DataLoader(ds, batch_size=256, shuffle=True, generator=loader_gen)
    val_meta = val[["patient_id", "omim", "present", "absent"]]

    def _preds(X, df, abstain):
        model.eval()
        with torch.no_grad():
            probs = torch.sigmoid(model(X.to(device))).cpu().numpy()  # (P, V)
        rows = []
        pids = df["patient_id"].to_numpy()
        for i, pid in enumerate(pids):
            for j, h in enumerate(vocab):
                if abstain[i, j]:
                    continue  # abstain: HPO's source feature gated for this patient
                rows.append((int(pid), h, float(probs[i, j])))
        preds = pd.DataFrame(rows, columns=["patient_id", "hpo_id", "score"])
        return thresholds.apply_score_transforms(preds, conf, is_prob=True)

    loss_hist, best_val_ap, best_state = [], float("-inf"), None
    for epoch in range(n_epochs):
        model.train()
        ep_loss = 0.0
        for xb, yb, mb in loader:
            xb, yb, mb = xb.to(device), yb.to(device), mb.to(device)
            opt.zero_grad()
            loss = (criterion(model(xb), yb) * mb).sum() / mb.sum().clamp(min=1.0)
            loss.backward()
            opt.step()
            ep_loss += loss.item() * len(xb)
        ep_loss /= len(ds)
        loss_hist.append(ep_loss)
        if epoch % 10 == 0 or epoch == n_epochs - 1:
            vp = _preds(Xval, val, ab_val)
            vap = eval_protocol.evaluate(vp.assign(pred_pos=False), val_meta, freq_table,
                                         conf, vocab, is_prob=True)["ranking_primary"]["macro_ap"]
            if best_state is None or vap > best_val_ap:
                best_val_ap = vap
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    val_preds = _preds(Xval, val, ab_val)
    test_preds = _preds(Xte, test, ab_te)

    cfg = {"arch": "MLPHead", "in_dim": 2 * len(feats), "n_out": len(vocab),
           "missingness_channel": True, "z_clip": config.Z_CLIP, "tag": tag,
           "lr": lr, "weight_decay": weight_decay, "n_epochs": n_epochs,
           "pos_weight": float(pos_weight.item()), "seed": config.SEED,
           "best_val_macro_ap": round(best_val_ap, 4),
           "first_epoch_loss": round(loss_hist[0], 4), "last_epoch_loss": round(loss_hist[-1], 4)}
    torch.save({"state_dict": best_state, "config": cfg, "feature_cols": feats, "vocab": vocab,
                "zscore_mean": mean.to_dict(), "zscore_std": std.to_dict()},
               run_dir / f"learned_{tag}_model.pt")
    (run_dir / f"learned_{tag}_config.json").write_text(json.dumps(cfg, indent=2))
    print(f"[step9:{tag}] loss {cfg['first_epoch_loss']}->{cfg['last_epoch_loss']}; "
          f"best_val_macro_ap {cfg['best_val_macro_ap']}; "
          f"abstained val cells {int(ab_val.sum())}, test cells {int(ab_te.sum())}")
    return val_preds, test_preds


def run(run_dir, n_epochs: int = 300, lr: float = 1e-3, weight_decay: float = 1e-4) -> dict:
    loader_gen = torch.Generator().manual_seed(config.SEED)

    table, feats = common.load_patient_table(run_dir)
    vocab = common.facial_vocab()
    freq_table = common.disease_freq_table()
    conf = common.direction_conf_map()

    train = table[table["split"] == "train"].reset_index(drop=True)
    val = table[table["split"] == "val"].reset_index(drop=True)
    test = table[table["split"] == "test"].reset_index(drop=True)

    mean, std = common.zscore_stats(train[feats])

    def _X(df):
        return torch.tensor(((df[feats] - mean) / std).to_numpy(dtype=np.float32))

    Xtr, Xval, Xte = _X(train), _X(val), _X(test)
    Ytr = torch.tensor(_soft_targets(train, vocab, freq_table))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    init_gen = torch.Generator(device=device).manual_seed(config.SEED)
    model = MLPHead(len(feats), len(vocab), init_gen).to(device)
    model.reset_parameters()  # params live on `device` -> generator must match device
    # global pos_weight from soft-target mass: up-weights the (rare) positive signal
    pos_rate = float(Ytr.mean().clamp(min=1e-4))
    pos_weight = torch.tensor((1.0 - pos_rate) / pos_rate, device=device).clamp(max=50.0)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    ds = TensorDataset(Xtr, Ytr)
    loader = DataLoader(ds, batch_size=256, shuffle=True, generator=loader_gen)

    val_meta = val[["patient_id", "omim", "present", "absent"]]

    def _predict(X, df, threshold):
        model.eval()
        with torch.no_grad():
            probs = torch.sigmoid(model(X.to(device))).cpu().numpy()  # (P, V)
        rows = []
        for i, pid in enumerate(df["patient_id"].to_numpy()):
            for j, h in enumerate(vocab):
                rows.append((int(pid), h, float(probs[i, j]), bool(probs[i, j] >= threshold)))
        return pd.DataFrame(rows, columns=["patient_id", "hpo_id", "score", "pred_pos"])

    # checkpoint selection is on threshold-free val ranking AP (macro_ap); the decision
    # threshold is tuned separately below. best_val_ap=-inf + the `best_state is None`
    # guard make the first eval always set a state, even if its macro_ap is NaN.
    loss_hist, val_ap_hist, best_val_ap, best_state = [], [], float("-inf"), None
    for epoch in range(n_epochs):
        model.train()
        ep_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            opt.step()
            ep_loss += loss.item() * len(xb)
        ep_loss /= len(ds)
        loss_hist.append(ep_loss)
        if epoch % 10 == 0 or epoch == n_epochs - 1:
            vm = eval_protocol.evaluate(_predict(Xval, val, 0.5), val_meta, freq_table, conf, vocab, is_prob=True)
            vap = vm["ranking_primary"]["macro_ap"]
            val_ap_hist.append((epoch, vap))
            if best_state is None or vap > best_val_ap:
                best_val_ap = vap
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)

    # --- tune the prob->pred-positive threshold on val by trusted-subset F1 (the SAME
    # criterion step4 uses for tau); ranking scores are threshold-independent. ---
    sweep = []
    for thr in config.PRED_POS_THRESHOLD_GRID:
        m = eval_protocol.evaluate(_predict(Xval, val, thr), val_meta, freq_table, conf, vocab, is_prob=True)
        ts = m["trusted_subset_strict"]
        sweep.append((thr, ts["f1"], ts["precision"], ts["recall"]))
    sweep_df = pd.DataFrame(sweep, columns=["threshold", "trusted_f1", "trusted_precision", "trusted_recall"])
    sweep_df.to_csv(run_dir / "learned_threshold_sweep.csv", index=False)
    best_thr = float(sweep_df.sort_values(["trusted_f1", "threshold"], ascending=[False, True]).iloc[0]["threshold"])

    _predict(Xval, val, best_thr).to_csv(run_dir / "learned_predictions_val.csv", index=False)
    _predict(Xte, test, best_thr).to_csv(run_dir / "learned_predictions_test.csv", index=False)

    cfg = {
        "arch": "MLPHead", "hidden": 128, "dropout": 0.3,
        "in_dim": len(feats), "n_out": len(vocab),
        "lr": lr, "weight_decay": weight_decay, "n_epochs": n_epochs,
        "batch_size": 256, "pos_weight": float(pos_weight.item()),
        "seed": config.SEED, "device": str(device),
        "pred_pos_threshold": best_thr,
        "pred_pos_threshold_selected_by": "val trusted-subset F1",
    }
    torch.save({"state_dict": best_state, "config": cfg, "feature_cols": feats, "vocab": vocab,
                "zscore_mean": mean.to_dict(), "zscore_std": std.to_dict()},
               run_dir / "learned_model.pt")
    (run_dir / "learned_config.json").write_text(json.dumps(cfg, indent=2))
    pd.DataFrame({"epoch": range(len(loss_hist)), "train_loss": loss_hist}).to_csv(
        run_dir / "learned_train_loss.csv", index=False)

    # random ranking reference on val
    rand_ap = eval_protocol.evaluate(_predict(Xval, val, best_thr), val_meta, freq_table, conf, vocab)["ranking_primary"]["random_macro_ap"]
    stats = {
        "first_epoch_loss": round(loss_hist[0], 4),
        "last_epoch_loss": round(loss_hist[-1], 4),
        "loss_decreased": loss_hist[-1] < loss_hist[0],
        "best_val_macro_ap": round(best_val_ap, 4),
        "val_random_macro_ap": round(rand_ap, 4),
        "beats_random": best_val_ap > rand_ap,
        "best_pred_pos_threshold": best_thr,
        "val_trusted_f1_at_threshold": float(sweep_df.sort_values(["trusted_f1", "threshold"], ascending=[False, True]).iloc[0]["trusted_f1"]),
    }
    print("[step5] verify:", stats)
    return stats


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from pathlib import Path
    run(Path("/tmp/hpo_split_debug"))
