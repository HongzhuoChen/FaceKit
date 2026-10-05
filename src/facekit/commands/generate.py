"""
FaceKit CLI command: generate

Sample synthetic faces from a StyleGAN3 generator. Images land in
``<output>/<name>/seed0000.png ...``, one folder per generator, which is the
multi-cohort layout that ``facekit extract-features`` and
``facekit privacy`` read directly.

Requires ``facekit[synth]``. Runs on CPU when no GPU is available (slowly).

Examples
--------
    # 200 images, seeds 0-199, from a generator you trained
    facekit generate --network runs/noonan/network-snapshot-005000.pkl \
        -o synthetic/ --name noonan --n 200

    # Reproduce specific seeds with truncation
    facekit generate --network G.pkl -o synthetic/ --seeds 0,7,40-49 --trunc 0.7
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer


def generate(
    network: Path = typer.Option(
        ...,
        "--network",
        exists=True, dir_okay=False, file_okay=True,
        help="StyleGAN3 network pickle (network-snapshot-*.pkl).",
    ),
    output_dir: Path = typer.Option(
        ...,
        "--output", "-o",
        file_okay=False, dir_okay=True,
        help="Output root. Images go to <output>/<name>/.",
    ),
    name: Optional[str] = typer.Option(
        None, "--name",
        help="Subfolder name (cohort). Defaults to the pickle's file stem.",
    ),
    n: Optional[int] = typer.Option(
        None, "--n", min=1,
        help="Number of images; uses seeds --first-seed .. --first-seed+n-1.",
    ),
    first_seed: int = typer.Option(
        0, "--first-seed", min=0, help="First seed when --n is given.",
    ),
    seeds: Optional[str] = typer.Option(
        None, "--seeds",
        help="Explicit seed list, e.g. '0,1,4-6'. Mutually exclusive with --n.",
    ),
    trunc: float = typer.Option(
        1.0, "--trunc", help="Truncation psi (1.0 = no truncation).",
    ),
    class_idx: Optional[int] = typer.Option(
        None, "--class-idx",
        help="Class label for a conditional generator; omit for unconditional.",
    ),
    device: str = typer.Option(
        "auto", "--device", help="'auto', 'cuda', 'cuda:1' or 'cpu'.",
    ),
):
    """Sample synthetic faces from a StyleGAN3 generator pickle."""
    if (n is None) == (seeds is None):
        typer.secho("[FaceKit] give exactly one of --n or --seeds",
                    fg=typer.colors.RED, err=True)
        raise typer.Exit(2)

    from facekit.core.synth.generate import (
        generate as _generate,
        load_generator,
        parse_seeds,
        resolve_device,
    )

    seed_list = list(range(first_seed, first_seed + n)) if n else parse_seeds(seeds)
    dev = resolve_device(device)
    out = output_dir.expanduser() / (name or network.stem)

    typer.echo(f"[FaceKit] generate: {network}")
    typer.echo(f"[FaceKit] Device: {dev}  Seeds: {len(seed_list)}  trunc={trunc}")
    G = load_generator(network, dev)
    typer.echo(f"[FaceKit] Generator: {G.img_resolution}x{G.img_resolution}, "
               f"c_dim={G.c_dim}")

    try:
        with typer.progressbar(
            _generate(G, seed_list, out, trunc, class_idx, dev),
            length=len(seed_list), label="[FaceKit] Sampling",
        ) as bar:
            for _ in bar:
                pass
    except ValueError as e:
        typer.secho(f"[FaceKit] {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(2)

    typer.echo(f"[FaceKit] Wrote {len(seed_list)} images to {out}")
