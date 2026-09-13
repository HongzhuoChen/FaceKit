"""
FaceKit CLI command: enhance

Prepare heterogeneous photographs for generator training. Two steps, each
applied only where a test says it is needed, colorization first:

  colorize  DDColor, when the image is grayscale (mean HSV saturation below
            --saturation, or a single-channel file)
  restore   GFPGAN v1.4, when the face is small (MediaPipe landmark box
            shorter side below --min-face pixels)

Handles the same two input layouts as ``facekit pack``. Every input image is
written to ``--output`` as PNG, processed or not, so the output folder is a
drop-in replacement for the input; ``enhance_log.csv`` records the decision
per image. Model weights download on first use to ``~/.cache/facekit``
(override with FACEKIT_CACHE_DIR).

Requires ``facekit[enhance]``. Slow on CPU; use a GPU for whole cohorts.

Examples
--------
    facekit enhance -i raw_images/ -o prepared/
    facekit enhance -i raw_images/ -o prepared/ --no-colorize --min-face 160
    facekit enhance -i raw_images/ -o prepared/ --force-restore
"""
from __future__ import annotations

from pathlib import Path
from typing import List

import typer

from facekit.commands.pack import _find_cohort_dirs, _has_images
from facekit.core.morph import list_imgpaths


def enhance(
    input_dir: Path = typer.Option(
        ..., "--input", "-i", exists=True, file_okay=False, dir_okay=True,
        help="Image folder: one subfolder per cohort, or a flat folder of images.",
    ),
    output_dir: Path = typer.Option(
        ..., "--output", "-o", file_okay=False, dir_okay=True,
        help="Output directory, mirroring the input layout; plus enhance_log.csv.",
    ),
    colorize: bool = typer.Option(True, "--colorize/--no-colorize",
                                  help="Colorize grayscale images with DDColor."),
    restore: bool = typer.Option(True, "--restore/--no-restore",
                                 help="Restore small faces with GFPGAN."),
    force_colorize: bool = typer.Option(False, "--force-colorize",
                                        help="Colorize every image, skipping the grayscale test."),
    force_restore: bool = typer.Option(False, "--force-restore",
                                       help="Restore every image, skipping the face-size test."),
    min_face: int = typer.Option(128, "--min-face", min=1,
                                 help="Restore when the face box's shorter side is below this."),
    saturation: float = typer.Option(0.03, "--saturation", min=0.0, max=1.0,
                                     help="Mean HSV saturation (0-1) below which an image is grayscale."),
    upscale: int = typer.Option(2, "--upscale", min=1,
                                help="GFPGAN output size relative to the input."),
    ddcolor_size: str = typer.Option("large", "--ddcolor-size",
                                     help="DDColor model: 'large' (default) or 'tiny'."),
    device: str = typer.Option("auto", "--device", help="'auto', 'cuda' or 'cpu'."),
    threads: int = typer.Option(8, "--threads", min=1, help="CPU threads for torch."),
):
    """Colorize grayscale photographs and restore small faces before training."""
    if ddcolor_size not in ("large", "tiny"):
        typer.secho("[FaceKit] --ddcolor-size must be 'large' or 'tiny'", fg=typer.colors.RED, err=True)
        raise typer.Exit(2)
    input_dir, output_dir = input_dir.expanduser(), output_dir.expanduser()
    cohort_dirs = _find_cohort_dirs(input_dir)
    if cohort_dirs and _has_images(input_dir):
        typer.secho("[FaceKit] input has both images and image subfolders; use one layout",
                    fg=typer.colors.RED, err=True)
        raise typer.Exit(2)
    flat = not cohort_dirs
    if flat:
        if not _has_images(input_dir):
            typer.secho(f"[FaceKit] no images under {input_dir}", fg=typer.colors.RED, err=True)
            raise typer.Exit(2)
        cohort_dirs = [input_dir]

    import torch
    from facekit.core.enhance.run import EnhanceOptions, Enhancer, enhance_folder, write_log
    from facekit.core.morph import MediaPipeLandmarkExtractor

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if not device.startswith("cuda"):
        torch.set_num_threads(threads)

    def make_colorizer(dev):
        from facekit.core.enhance.colorize import Colorizer
        typer.echo(f"[FaceKit] loading DDColor ({ddcolor_size})")
        return Colorizer(dev, model_size=ddcolor_size)

    def make_restorer(dev):
        from facekit.core.enhance.restore import Restorer
        typer.echo("[FaceKit] loading GFPGAN v1.4")
        return Restorer(dev, upscale=upscale)

    opts = EnhanceOptions(colorize=colorize, restore=restore, force_colorize=force_colorize,
                          force_restore=force_restore, min_face=min_face,
                          saturation_threshold=saturation)
    enhancer = Enhancer(opts, device, MediaPipeLandmarkExtractor(), make_colorizer, make_restorer)

    typer.echo(f"[FaceKit] enhance: {input_dir} -> {output_dir}  device={device}")
    typer.echo(f"[FaceKit] colorize={'on' if colorize else 'off'} restore={'on' if restore else 'off'} "
               f"min_face={min_face} saturation<{saturation}")
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for cohort_dir in cohort_dirs:
        cohort = cohort_dir.name
        paths = [Path(p) for p in list_imgpaths(cohort_dir)]
        out = output_dir if flat else output_dir / cohort
        with typer.progressbar(paths, label=f"[FaceKit] {cohort}") as bar:
            records.extend(enhance_folder(enhancer, bar, out, cohort))
    write_log(records, output_dir / "enhance_log.csv")
    n_c = sum(r.colorized for r in records)
    n_r = sum(r.restored for r in records)
    n_bad = sum(r.status != "ok" for r in records)
    typer.echo(f"[FaceKit] {len(records)} images: {n_c} colorized, {n_r} restored, "
               f"{n_bad} unreadable -> {output_dir} (enhance_log.csv)")
