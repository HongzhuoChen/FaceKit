"""
FaceKit CLI command: privacy

Analyse synthetic faces for residual similarity to the real photographs that
trained the generator, along two separate axes:

  identity   -- recognition embeddings (ArcFace by default; AdaFace and LVFace
                when their model files are supplied), cosine distance
  appearance -- LPIPS perceptual distance

For each axis every synthetic and every held-out image is characterized by
its distance to the nearest training image of the same cohort. Thresholds
are read off the held-out distribution at percentiles p, so held-out images
are flagged at rate p by construction and the synthetic rate is the quantity
of interest. Nearest-neighbour adversarial accuracy (NNAA) and its privacy
loss, AA(held-out, synthetic) - AA(train, synthetic), summarize each axis
with bootstrap confidence intervals.

Inputs are three roots with one subfolder per cohort; cohort folder names
must match across the three. The held-out partition must be disjoint from
the training partition at the patient level.

  train/<cohort>/*.png      images the generator was trained on
  heldout/<cohort>/*.png    real images never used in training
  synthetic/<cohort>/*.png  images sampled from the cohort's generator

Outputs, under ``--output``:
  flagging.csv, nnaa.csv, privacy_results.json,
  below_threshold.png, nnaa.png, cache/ (embeddings and LPIPS distances)

Requires ``facekit[privacy]``. The ArcFace weights (InsightFace buffalo_l)
download automatically on first use.

Examples
--------
    facekit privacy --train prepared/train --heldout prepared/heldout \
        --synthetic synthetic/ -o privacy_out/

    facekit privacy ... --backbone arcface --backbone lvface \
        --lvface-onnx models/LVFace-L_Glint360K.onnx --bootstrap 2000
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import typer

_DIR = dict(exists=True, file_okay=False, dir_okay=True)
_FILE = dict(exists=True, file_okay=True, dir_okay=False)


def privacy(
    train_dir: Path = typer.Option(..., "--train", **_DIR,
                                   help="Training images, one subfolder per cohort."),
    heldout_dir: Path = typer.Option(..., "--heldout", **_DIR,
                                     help="Held-out real images, same cohort subfolders."),
    synthetic_dir: Path = typer.Option(..., "--synthetic", **_DIR,
                                       help="Synthetic images, same cohort subfolders."),
    output_dir: Path = typer.Option(..., "--output", "-o", file_okay=False, dir_okay=True,
                                    help="Output directory."),
    backbones: List[str] = typer.Option(
        ["arcface"], "--backbone",
        help="Recognition model(s): arcface, adaface, lvface. Repeat to use several.",
    ),
    adaface_dir: Optional[Path] = typer.Option(None, "--adaface-dir", **_DIR,
                                               help="Clone of the AdaFace repository."),
    adaface_ckpt: Optional[Path] = typer.Option(None, "--adaface-ckpt", **_FILE,
                                                help="AdaFace checkpoint (.ckpt)."),
    lvface_onnx: Optional[Path] = typer.Option(None, "--lvface-onnx", **_FILE,
                                               help="LVFace ONNX model."),
    percentiles: str = typer.Option("1,2,5,10,20", "--percentiles",
                                    help="Calibration percentiles, comma-separated."),
    bootstrap: int = typer.Option(1000, "--bootstrap", min=0,
                                  help="Bootstrap draws for the NNAA confidence intervals."),
    max_samples: int = typer.Option(1000, "--max-samples", min=2,
                                    help="Cap per set for embedding NNAA."),
    lpips: bool = typer.Option(True, "--lpips/--no-lpips",
                               help="Also run the LPIPS appearance assessment."),
    lpips_max_samples: int = typer.Option(200, "--lpips-max-samples", min=2,
                                          help="Cap per set for LPIPS NNAA (quadratic cost)."),
    device: str = typer.Option("auto", "--device", help="'auto', 'cuda' or 'cpu'."),
    threads: int = typer.Option(8, "--threads", min=1,
                                help="CPU threads for onnxruntime and torch (shared nodes)."),
    seed: int = typer.Option(0, "--seed", min=0, help="Seed for subsampling and bootstrap."),
):
    """Analyse synthetic faces for identity and appearance leakage from the training images."""
    try:
        pcts = sorted({int(p) for p in percentiles.split(",") if p.strip()})
    except ValueError:
        typer.secho(f"[FaceKit] bad --percentiles {percentiles!r}", fg=typer.colors.RED, err=True)
        raise typer.Exit(2)
    for b in backbones:
        if b not in ("arcface", "adaface", "lvface"):
            typer.secho(f"[FaceKit] unknown --backbone {b!r}", fg=typer.colors.RED, err=True)
            raise typer.Exit(2)

    import torch
    from facekit.core.privacy.audit import AuditConfig, run_audit

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg = AuditConfig(
        train_dir=train_dir.expanduser(), heldout_dir=heldout_dir.expanduser(),
        synthetic_dir=synthetic_dir.expanduser(), out_dir=output_dir.expanduser(),
        backbones=list(dict.fromkeys(backbones)),
        adaface_dir=adaface_dir, adaface_ckpt=adaface_ckpt, lvface_onnx=lvface_onnx,
        percentiles=pcts, n_bootstrap=bootstrap, max_samples=max_samples,
        lpips=lpips, lpips_max_samples=lpips_max_samples, device=device, threads=threads, seed=seed,
    )
    typer.echo(f"[FaceKit] privacy: device={device} backbones={cfg.backbones} "
               f"lpips={'on' if lpips else 'off'}")
    try:
        run_audit(cfg, echo=typer.echo)
    except ValueError as e:
        typer.secho(f"[FaceKit] {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(2)
    typer.echo(f"[FaceKit] Wrote flagging.csv, nnaa.csv, privacy_results.json and plots to {output_dir}")
