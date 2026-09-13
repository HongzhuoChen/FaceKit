"""Orchestrates the privacy audit: embeddings -> flagging -> NNAA, then LPIPS.

Inputs are three roots with one subfolder per cohort. A cohort is audited
when it is present in all three; the training and held-out partitions must
be disjoint at the patient level (the caller's responsibility). Everything
expensive is cached under ``<out>/cache`` so the command can be re-run with
different percentiles or bootstrap counts for free.
"""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
from tqdm import tqdm

from facekit.core.privacy import embeddings as E
from facekit.core.privacy import lpips_dist as L
from facekit.core.privacy.metrics import (
    bootstrap_aa,
    bootstrap_aa_from_min_dists,
    drop_zero_rows,
    min_cross_distances,
    nnaa_summary,
    percentile_rows,
)

SPLITS = ("train", "heldout", "synthetic")


@dataclass
class AuditConfig:
    train_dir: Path
    heldout_dir: Path
    synthetic_dir: Path
    out_dir: Path
    backbones: List[str] = field(default_factory=lambda: ["arcface"])
    adaface_dir: Optional[Path] = None
    adaface_ckpt: Optional[Path] = None
    lvface_onnx: Optional[Path] = None
    percentiles: List[int] = field(default_factory=lambda: [1, 2, 5, 10, 20])
    n_bootstrap: int = 1000
    max_samples: int = 1000          # cap per set for embedding NNAA
    lpips: bool = True
    lpips_net: str = "alex"
    lpips_size: int = 256
    lpips_max_samples: int = 200     # cap per set for LPIPS NNAA (quadratic cost)
    metric: str = "cosine"
    device: str = "cpu"
    threads: Optional[int] = None    # CPU thread cap (onnxruntime + torch)
    seed: int = 0


def find_cohorts(cfg: AuditConfig) -> Dict[str, List[str]]:
    """``{"common": [...], "train_only": [...], ...}`` from subfolder names."""
    roots = {"train": cfg.train_dir, "heldout": cfg.heldout_dir, "synthetic": cfg.synthetic_dir}
    names = {k: {d.name for d in v.iterdir() if d.is_dir() and not d.name.startswith(".")}
             for k, v in roots.items()}
    common = set.intersection(*names.values())
    out = {"common": sorted(common)}
    for k, s in names.items():
        out[f"{k}_only"] = sorted(s - common)
    return out


def _build_backbone(name: str, cfg: AuditConfig, face_analysis) -> E.Backbone:
    if name == "arcface":
        return E.arcface_backbone(face_analysis)
    if name == "adaface":
        if not (cfg.adaface_dir and cfg.adaface_ckpt):
            raise ValueError("adaface needs --adaface-dir and --adaface-ckpt")
        return E.adaface_backbone(face_analysis, cfg.adaface_dir, cfg.adaface_ckpt, cfg.device)
    if name == "lvface":
        if not cfg.lvface_onnx:
            raise ValueError("lvface needs --lvface-onnx")
        return E.lvface_backbone(face_analysis, cfg.lvface_onnx, cfg.device)
    raise ValueError(f"unknown backbone {name!r}")


def _split_dir(cfg: AuditConfig, split: str, cohort: str) -> Path:
    root = {"train": cfg.train_dir, "heldout": cfg.heldout_dir, "synthetic": cfg.synthetic_dir}[split]
    return root / cohort


def _subsample(x, n: int, rng: np.random.Generator):
    if n and len(x) > n:
        idx = np.sort(rng.choice(len(x), n, replace=False))
        return [x[i] for i in idx] if isinstance(x, list) else x[idx]
    return x


def audit_embeddings(cfg: AuditConfig, cohorts: List[str], echo: Callable[[str], None]) -> Dict:
    face_analysis = E.make_face_analysis(cfg.device, threads=cfg.threads)
    results = {}
    for name in cfg.backbones:
        backbone = _build_backbone(name, cfg, face_analysis)
        echo(f"[FaceKit] embeddings: {name}")
        cache = cfg.out_dir / "cache" / "embeddings" / name
        pooled = {s: [] for s in SPLITS}
        syn_min, ho_min = [], []
        no_face = {}
        for cohort in cohorts:
            emb = {}
            for split in SPLITS:
                folder = _split_dir(cfg, split, cohort)
                n = len(E.list_images(folder))
                with tqdm(total=n, desc=f"  {name} {split}/{cohort}", leave=False) as bar:
                    arr, names = E.embed_folder(backbone, folder, cache / f"{split}__{cohort}",
                                                progress=bar.update)
                no_face[f"{split}/{cohort}"] = int(np.sum(np.linalg.norm(arr, axis=1) <= 0.1))
                emb[split] = drop_zero_rows(arr)
                pooled[split].append(emb[split])
            syn_min.extend(min_cross_distances(emb["synthetic"], emb["train"], cfg.metric))
            ho_min.extend(min_cross_distances(emb["heldout"], emb["train"], cfg.metric))
        syn_min, ho_min = np.asarray(syn_min), np.asarray(ho_min)
        rows = percentile_rows(syn_min, ho_min, cfg.percentiles)
        for r in rows:
            echo(f"  p={r['percentile']:>2}%  thr={r['threshold']:.4f}  "
                 f"synthetic {r['synthetic_pct']:.2f}%  held-out {r['heldout_pct']:.2f}%")

        rng = np.random.default_rng(cfg.seed)
        pool = {s: _subsample(np.vstack(v), cfg.max_samples, rng) for s, v in pooled.items()}
        echo(f"  NNAA on train={len(pool['train'])} heldout={len(pool['heldout'])} "
             f"synthetic={len(pool['synthetic'])}, {cfg.n_bootstrap} bootstrap draws")
        bs_train = bootstrap_aa(pool["train"], pool["synthetic"], cfg.metric, cfg.n_bootstrap, rng)
        bs_test = bootstrap_aa(pool["heldout"], pool["synthetic"], cfg.metric, cfg.n_bootstrap, rng)
        nnaa = nnaa_summary(bs_train, bs_test)
        pl = nnaa["privacy_loss"]
        echo(f"  privacy loss = {pl['median']:+.4f} [{pl['ci_lower']:+.4f}, {pl['ci_upper']:+.4f}]")

        results[name] = {
            "flagging": rows, "n_synthetic": int(len(syn_min)), "n_heldout": int(len(ho_min)),
            "nnaa": nnaa, "nnaa_n": {s: int(len(v)) for s, v in pool.items()},
            "no_face": no_face,
        }
    return results


def audit_lpips(cfg: AuditConfig, cohorts: List[str], echo: Callable[[str], None]) -> Dict:
    loss_fn = L.make_lpips(cfg.lpips_net, cfg.device)
    cache = cfg.out_dir / "cache" / "lpips"
    echo(f"[FaceKit] LPIPS ({cfg.lpips_net}, {cfg.lpips_size}px): nearest training image")
    syn_min, ho_min = [], []
    for cohort in cohorts:
        train_paths = E.list_images(_split_dir(cfg, "train", cohort))
        train_t = None
        for split, store in (("synthetic", syn_min), ("heldout", ho_min)):
            path = cache / f"min_{split}__{cohort}.npy"
            if not path.exists():
                if train_t is None:
                    train_t = L.load_tensors(train_paths, cfg.lpips_size, cfg.device)
                q_paths = E.list_images(_split_dir(cfg, split, cohort))
                q_t = L.load_tensors(q_paths, cfg.lpips_size, cfg.device)
                with tqdm(total=len(q_paths), desc=f"  lpips {split}/{cohort}", leave=False) as bar:
                    arr = L.min_dists_cross(q_t, train_t, loss_fn, progress=bar.update)
                path.parent.mkdir(parents=True, exist_ok=True)
                np.save(path, arr)
            store.extend(np.load(path))
    syn_min = np.asarray([d for d in syn_min if not np.isnan(d)])
    ho_min = np.asarray([d for d in ho_min if not np.isnan(d)])
    rows = percentile_rows(syn_min, ho_min, cfg.percentiles)
    for r in rows:
        echo(f"  p={r['percentile']:>2}%  thr={r['threshold']:.4f}  "
             f"synthetic {r['synthetic_pct']:.2f}%  held-out {r['heldout_pct']:.2f}%")

    # NNAA on a pooled, capped sample of each set.
    rng = np.random.default_rng(cfg.seed)
    paths = {s: _subsample(sorted(p for c in cohorts for p in E.list_images(_split_dir(cfg, s, c))),
                           cfg.lpips_max_samples, rng) for s in SPLITS}
    echo(f"  NNAA on train={len(paths['train'])} heldout={len(paths['heldout'])} "
         f"synthetic={len(paths['synthetic'])}")
    tensors = {s: L.load_tensors(p, cfg.lpips_size, cfg.device) for s, p in paths.items()}
    tag = f"n{cfg.lpips_max_samples}_seed{cfg.seed}"

    def within(s):
        return L.cached(cache / f"nnaa_{tag}_{s}_to_{s}.npy",
                        lambda: L.min_dists_within(tensors[s], loss_fn))

    def cross(a, b):
        return L.cached(cache / f"nnaa_{tag}_{a}_to_{b}.npy",
                        lambda: L.min_dists_cross(tensors[a], tensors[b], loss_fn))

    t2t, s2s, h2h = within("train"), within("synthetic"), within("heldout")
    t2s, s2t = cross("train", "synthetic"), cross("synthetic", "train")
    h2s, s2h = cross("heldout", "synthetic"), cross("synthetic", "heldout")

    def clean(*arrs):
        m = np.all([~np.isnan(a) for a in arrs], axis=0)
        return [a[m] for a in arrs]

    t2s_c, t2t_c = clean(t2s, t2t)
    s2t_c, s2s_a = clean(s2t, s2s)
    h2s_c, h2h_c = clean(h2s, h2h)
    s2h_c, s2s_b = clean(s2h, s2s)
    bs_train = bootstrap_aa_from_min_dists(t2s_c, t2t_c, s2t_c, s2s_a, cfg.n_bootstrap, rng)
    bs_test = bootstrap_aa_from_min_dists(h2s_c, h2h_c, s2h_c, s2s_b, cfg.n_bootstrap, rng)
    nnaa = nnaa_summary(bs_train, bs_test)
    pl = nnaa["privacy_loss"]
    echo(f"  privacy loss = {pl['median']:+.4f} [{pl['ci_lower']:+.4f}, {pl['ci_upper']:+.4f}]")
    return {
        "flagging": rows, "n_synthetic": int(len(syn_min)), "n_heldout": int(len(ho_min)),
        "nnaa": nnaa, "nnaa_n": {s: int(len(p)) for s, p in paths.items()},
    }


def write_outputs(cfg: AuditConfig, cohorts: List[str], results: Dict[str, Dict]) -> None:
    out = cfg.out_dir
    with open(out / "flagging.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["metric", "percentile", "threshold", "synthetic_pct", "heldout_pct",
                    "n_synthetic", "n_heldout"])
        for metric, r in results.items():
            for row in r["flagging"]:
                w.writerow([metric, row["percentile"], f"{row['threshold']:.6f}",
                            row["synthetic_pct"], row["heldout_pct"],
                            r["n_synthetic"], r["n_heldout"]])
    with open(out / "nnaa.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["metric", "aa_train_synth", "aa_heldout_synth", "privacy_loss",
                    "ci_lower", "ci_upper", "n_train", "n_heldout", "n_synthetic", "n_bootstrap"])
        for metric, r in results.items():
            n = r["nnaa"]
            w.writerow([metric, f"{n['aa_train_synth']['median']:.4f}",
                        f"{n['aa_test_synth']['median']:.4f}",
                        f"{n['privacy_loss']['median']:+.4f}",
                        f"{n['privacy_loss']['ci_lower']:+.4f}",
                        f"{n['privacy_loss']['ci_upper']:+.4f}",
                        r["nnaa_n"]["train"], r["nnaa_n"]["heldout"], r["nnaa_n"]["synthetic"],
                        cfg.n_bootstrap])
    with open(out / "privacy_results.json", "w") as fh:
        json.dump({"cohorts": cohorts, "percentiles": cfg.percentiles,
                   "n_bootstrap": cfg.n_bootstrap, "results": results}, fh, indent=2)
    plot_results(cfg, results, out)


def plot_results(cfg: AuditConfig, results: Dict[str, Dict], out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    identity = {k: v for k, v in results.items() if k != "lpips"}
    panels = [("Identity (recognition embeddings)", identity)]
    if "lpips" in results:
        panels.append(("Appearance (LPIPS)", {"lpips": results["lpips"]}))
    fig, axes = plt.subplots(1, len(panels), figsize=(6 * len(panels), 4.2), squeeze=False)
    for ax, (title, group) in zip(axes[0], panels):
        ho_done = False
        for metric, r in group.items():
            x = [row["percentile"] for row in r["flagging"]]
            ax.plot(x, [row["synthetic_pct"] for row in r["flagging"]], "o-", lw=2, ms=6,
                    label=f"synthetic, {metric} (n={r['n_synthetic']})")
            if not ho_done:
                ax.plot(x, [row["heldout_pct"] for row in r["flagging"]], "s--", color="0.4",
                        lw=1.5, ms=5, label=f"held-out (n={r['n_heldout']})")
                ho_done = True
        ax.set_xlabel("calibration percentile p (threshold = p-th %ile of held-out NN distance)")
        ax.set_ylabel("% below threshold")
        ax.set_xticks(cfg.percentiles)
        ax.set_xticklabels([f"{p}%" for p in cfg.percentiles])
        ax.set_ylim(bottom=0)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "below_threshold.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(1.6 * len(results) + 3, 4))
    names = list(results)
    x = np.arange(len(names))
    for k, (key, label) in enumerate((("aa_train_synth", "AA(train, synthetic)"),
                                      ("aa_test_synth", "AA(held-out, synthetic)"))):
        med = [results[n]["nnaa"][key]["median"] for n in names]
        lo = [med[i] - results[n]["nnaa"][key]["ci_lower"] for i, n in enumerate(names)]
        hi = [results[n]["nnaa"][key]["ci_upper"] - med[i] for i, n in enumerate(names)]
        ax.bar(x + (k - 0.5) * 0.38, med, 0.36, yerr=[lo, hi], capsize=3, label=label)
    ax.axhline(0.5, color="0.3", ls=":", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("adversarial accuracy")
    ax.set_title("Nearest-neighbour adversarial accuracy (0.5 = indistinguishable)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "nnaa.png", dpi=150)
    plt.close(fig)


def run_audit(cfg: AuditConfig, echo: Callable[[str], None] = print) -> Dict:
    cohorts = find_cohorts(cfg)
    for k, v in cohorts.items():
        if k != "common" and v:
            echo(f"[FaceKit] skipped ({k.replace('_', ' ')}): {', '.join(v)}")
    common = cohorts["common"]
    if not common:
        raise ValueError("no cohort folder is present in all of --train, --heldout, --synthetic")
    echo(f"[FaceKit] {len(common)} cohort(s): {', '.join(common)}")
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    if cfg.threads:
        import torch
        torch.set_num_threads(cfg.threads)

    results = audit_embeddings(cfg, common, echo)
    if cfg.lpips:
        results["lpips"] = audit_lpips(cfg, common, echo)
    write_outputs(cfg, common, results)
    return results
