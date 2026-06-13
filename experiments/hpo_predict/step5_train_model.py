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
    def __init__(self, in_dim: int, n_out: int, hidden: int = 128, p_drop: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(p_drop),
            nn.Linear(hidden, n_out),
        )

    def forward(self, x):  # x: (B, in_dim) -> logits (B, n_out)
        return self.net(x)


def _soft_targets(table, vocab, freq_table):
    """(P, |vocab|) float target = freq_value of patient's disease per term, else 0."""
    Y = np.zeros((len(table), len(vocab)), dtype=np.float32)
    col = {h: j for j, h in enumerate(vocab)}
    for i, omim in enumerate(table["omim"].to_numpy()):
        for h, fr in freq_table.get(omim, {}).items():
            Y[i, col[h]] = fr
    return Y


def run(run_dir, n_epochs: int = 300, lr: float = 1e-3, weight_decay: float = 1e-4) -> dict:
    torch.manual_seed(config.SEED)
    np_gen = np.random.default_rng(config.SEED)
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
    Yval_soft = _soft_targets(val, vocab, freq_table)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLPHead(len(feats), len(vocab)).to(device)
    # global pos_weight from soft-target mass: up-weights the (rare) positive signal
    pos_rate = float(Ytr.mean().clamp(min=1e-4))
    pos_weight = torch.tensor((1.0 - pos_rate) / pos_rate, device=device).clamp(max=50.0)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    ds = TensorDataset(Xtr, Ytr)
    loader = DataLoader(ds, batch_size=256, shuffle=True, generator=loader_gen)

    val_meta = val[["patient_id", "omim", "present", "absent"]]

    def _predict(X, df):
        model.eval()
        with torch.no_grad():
            probs = torch.sigmoid(model(X.to(device))).cpu().numpy()  # (P, V)
        rows = []
        for i, pid in enumerate(df["patient_id"].to_numpy()):
            for j, h in enumerate(vocab):
                rows.append((int(pid), h, float(probs[i, j]), bool(probs[i, j] >= config.PRED_POS_THRESHOLD)))
        return pd.DataFrame(rows, columns=["patient_id", "hpo_id", "score", "pred_pos"])

    loss_hist, val_ap_hist, best_val_ap, best_state = [], [], -1.0, None
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
            vm = eval_protocol.evaluate(_predict(Xval, val), val_meta, freq_table, conf, vocab, is_prob=True)
            vap = vm["ranking_primary"]["macro_ap"]
            val_ap_hist.append((epoch, vap))
            if vap > best_val_ap:
                best_val_ap = vap
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    _predict(Xval, val).to_csv(run_dir / "learned_predictions_val.csv", index=False)
    _predict(Xte, test).to_csv(run_dir / "learned_predictions_test.csv", index=False)

    cfg = {
        "arch": "MLPHead", "hidden": 128, "dropout": 0.3,
        "in_dim": len(feats), "n_out": len(vocab),
        "lr": lr, "weight_decay": weight_decay, "n_epochs": n_epochs,
        "batch_size": 256, "pos_weight": float(pos_weight.item()),
        "seed": config.SEED, "device": str(device),
        "pred_pos_threshold": config.PRED_POS_THRESHOLD,
    }
    torch.save({"state_dict": best_state, "config": cfg, "feature_cols": feats, "vocab": vocab,
                "zscore_mean": mean.to_dict(), "zscore_std": std.to_dict()},
               run_dir / "learned_model.pt")
    (run_dir / "learned_config.json").write_text(json.dumps(cfg, indent=2))
    pd.DataFrame({"epoch": range(len(loss_hist)), "train_loss": loss_hist}).to_csv(
        run_dir / "learned_train_loss.csv", index=False)

    # random ranking reference on val
    rand_ap = eval_protocol.evaluate(_predict(Xval, val), val_meta, freq_table, conf, vocab)["ranking_primary"]["random_macro_ap"]
    stats = {
        "first_epoch_loss": round(loss_hist[0], 4),
        "last_epoch_loss": round(loss_hist[-1], 4),
        "loss_decreased": loss_hist[-1] < loss_hist[0],
        "best_val_macro_ap": round(best_val_ap, 4),
        "val_random_macro_ap": round(rand_ap, 4),
        "beats_random": best_val_ap > rand_ap,
    }
    print("[step5] verify:", stats)
    return stats


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from pathlib import Path
    run(Path("/tmp/hpo_split_debug"))
