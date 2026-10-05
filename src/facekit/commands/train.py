"""
FaceKit CLI command: train

Train a StyleGAN3 generator on a dataset zip written by ``facekit pack``.
This is a thin wrapper around the vendored ``train.py`` whose defaults are
the settings the FaceKit generators were trained with:

  --cfg stylegan3-t  --gamma 2  --mirror  --kimg 5000
  --glr 0.0025  --dlr 0.002  --map-depth 8  --lr-schedule cosine

Anything else StyleGAN3 accepts can be appended verbatim with ``--extra``.
Snapshots land in ``<output>/<NNNNN>-<desc>/network-snapshot-*.pkl`` and
feed ``facekit generate --network``. Requires a CUDA GPU (``facekit[synth]``).

Examples
--------
    facekit train --data datasets/noonan.zip -o runs/ --gpus 1 --batch 32 --batch-gpu 8

    # Print the resolved StyleGAN3 options without training
    facekit train --data datasets/noonan.zip -o runs/ --dry-run

    # Pass native StyleGAN3 options through
    facekit train --data d.zip -o runs/ --extra "--snap 10 --resume ffhq256.pkl"
"""
from __future__ import annotations

import shlex
import subprocess
import sys
from pathlib import Path
from typing import Optional

import typer

from facekit.vendor import STYLEGAN3_DIR


def train(
    data: Path = typer.Option(
        ...,
        "--data",
        exists=True, dir_okay=True, file_okay=True,
        help="Dataset zip from 'facekit pack' (or a folder of same-size square PNGs).",
    ),
    output_dir: Path = typer.Option(
        ...,
        "--output", "-o",
        file_okay=False, dir_okay=True,
        help="Root for training runs; StyleGAN3 creates one numbered subfolder per run.",
    ),
    gpus: int = typer.Option(1, "--gpus", min=1, help="Number of GPUs."),
    batch: int = typer.Option(32, "--batch", min=1, help="Total batch size."),
    batch_gpu: Optional[int] = typer.Option(
        None, "--batch-gpu", min=1,
        help="Images per GPU per step; the rest of --batch is accumulated. Default: batch/gpus.",
    ),
    cfg: str = typer.Option(
        "stylegan3-t", "--cfg",
        help="StyleGAN3 base configuration: stylegan3-t, stylegan3-r or stylegan2.",
    ),
    gamma: float = typer.Option(2.0, "--gamma", min=0.0, help="R1 regularization weight."),
    mirror: bool = typer.Option(True, "--mirror/--no-mirror", help="Horizontal-flip augmentation."),
    kimg: int = typer.Option(5000, "--kimg", min=1, help="Training length in thousands of images."),
    glr: float = typer.Option(0.0025, "--glr", min=0.0, help="Generator learning rate."),
    dlr: float = typer.Option(0.002, "--dlr", min=0.0, help="Discriminator learning rate."),
    map_depth: int = typer.Option(8, "--map-depth", min=1, help="Mapping network layers."),
    lr_schedule: str = typer.Option(
        "cosine", "--lr-schedule", help="'cosine' decay to zero over --kimg, or 'constant'.",
    ),
    snap: int = typer.Option(50, "--snap", min=1, help="Snapshot every N ticks (a tick is 4 kimg)."),
    metrics: str = typer.Option(
        "none", "--metrics",
        help="StyleGAN3 quality metrics per snapshot, e.g. 'fid50k_full'. Off by default.",
    ),
    seed: int = typer.Option(0, "--seed", min=0, help="Random seed."),
    extra: Optional[str] = typer.Option(
        None, "--extra", help="Additional train.py options, quoted as one string.",
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Print the resolved options and exit without training.",
    ),
):
    """Train a StyleGAN3 generator with the FaceKit defaults."""
    if cfg not in ("stylegan3-t", "stylegan3-r", "stylegan2"):
        typer.secho(f"[FaceKit] unknown --cfg {cfg!r}", fg=typer.colors.RED, err=True)
        raise typer.Exit(2)
    if lr_schedule not in ("cosine", "constant"):
        typer.secho(f"[FaceKit] unknown --lr-schedule {lr_schedule!r}",
                    fg=typer.colors.RED, err=True)
        raise typer.Exit(2)

    if not dry_run:
        import torch
        n_cuda = torch.cuda.device_count()
        if n_cuda < gpus:
            typer.secho(f"[FaceKit] {gpus} GPU(s) requested but {n_cuda} visible; "
                        "run on a GPU node or use --dry-run",
                        fg=typer.colors.RED, err=True)
            raise typer.Exit(2)

    output_dir = output_dir.expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    argv = [
        sys.executable, str(STYLEGAN3_DIR / "train.py"),
        f"--outdir={output_dir}", f"--data={data.expanduser()}",
        f"--cfg={cfg}", f"--gpus={gpus}", f"--batch={batch}", f"--gamma={gamma}",
        f"--mirror={int(mirror)}", f"--kimg={kimg}", f"--glr={glr}", f"--dlr={dlr}",
        f"--map-depth={map_depth}", f"--lr-schedule={lr_schedule}",
        f"--snap={snap}", f"--metrics={metrics}", f"--seed={seed}",
    ]
    if batch_gpu is not None:
        argv.append(f"--batch-gpu={batch_gpu}")
    if extra:
        argv.extend(shlex.split(extra))
    if dry_run:
        argv.append("--dry-run")

    typer.echo(f"[FaceKit] train: {data} -> {output_dir}")
    typer.echo("[FaceKit] " + " ".join(shlex.quote(a) for a in argv[2:]))
    proc = subprocess.run(argv, cwd=str(STYLEGAN3_DIR))
    if proc.returncode != 0:
        typer.secho(f"[FaceKit] train.py exited with {proc.returncode}",
                    fg=typer.colors.RED, err=True)
        raise typer.Exit(proc.returncode)
