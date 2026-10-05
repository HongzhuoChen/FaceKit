"""
FaceKit CLI command: pack

Prepare a StyleGAN3 training set from face photographs: locate each face
from its MediaPipe landmarks, crop a square around it, resize to a fixed
resolution, and pack the crops into the zip that ``facekit train`` reads.

Handles the same two input layouts as ``facekit extract-landmarks``:

  Layout A (multi-cohort):  images/<cohort>/*.jpg  -> one zip per cohort
  Layout B (single folder): images/*.jpg           -> one zip named after the folder

Outputs, under ``--output``:
  <cohort>/<stem>.png   the prepared crops (usable by extract-features / privacy)
  <cohort>.zip          the StyleGAN3 dataset archive
  pack_log.csv          one row per input image: ok / no_face / unreadable

Run ``facekit enhance`` first if the photographs need colorization or
restoration; ``pack`` itself only crops and resizes.

Examples
--------
    facekit pack -i images/ -o datasets/
    facekit pack -i images/noonan -o datasets/ --resolution 256 --margin 0.3
"""
from __future__ import annotations

from pathlib import Path
from typing import List

import typer

from facekit.core.morph import MediaPipeLandmarkExtractor, list_imgpaths
from facekit.core.synth.pack import (
    PackRecord,
    build_dataset_zip,
    prepare_cohort,
    write_log,
)


def _has_images(folder: Path) -> bool:
    if not folder.is_dir():
        return False
    try:
        return any(list_imgpaths(folder))
    except (FileNotFoundError, NotADirectoryError):
        return False


def _find_cohort_dirs(input_dir: Path) -> List[Path]:
    return sorted(d for d in input_dir.iterdir() if d.is_dir() and _has_images(d))


def pack(
    input_dir: Path = typer.Option(
        ...,
        "--input", "-i",
        exists=True, file_okay=False, dir_okay=True,
        help="Image folder: one subfolder per cohort, or a flat folder of images.",
    ),
    output_dir: Path = typer.Option(
        ...,
        "--output", "-o",
        file_okay=False, dir_okay=True,
        help="Output directory for <cohort>/ crops, <cohort>.zip and pack_log.csv.",
    ),
    resolution: int = typer.Option(
        256, "--resolution", min=32,
        help="Side of the square output images (StyleGAN3 needs a power of two).",
    ),
    margin: float = typer.Option(
        0.3, "--margin", min=0.0,
        help="Border around the landmark box, as a fraction of its size, per side.",
    ),
):
    """Crop faces to a fixed square and pack them into a StyleGAN3 dataset zip."""
    input_dir = input_dir.expanduser()
    output_dir = output_dir.expanduser()

    if resolution & (resolution - 1):
        typer.secho(f"[FaceKit] --resolution {resolution} is not a power of two",
                    fg=typer.colors.RED, err=True)
        raise typer.Exit(2)

    cohort_dirs = _find_cohort_dirs(input_dir)
    if cohort_dirs and _has_images(input_dir):
        typer.secho("[FaceKit] input has both images and image subfolders; "
                    "use one layout", fg=typer.colors.RED, err=True)
        raise typer.Exit(2)
    if not cohort_dirs:
        if not _has_images(input_dir):
            typer.secho(f"[FaceKit] no images under {input_dir}",
                        fg=typer.colors.RED, err=True)
            raise typer.Exit(2)
        cohort_dirs = [input_dir]

    typer.echo(f"[FaceKit] pack: {input_dir} -> {output_dir}")
    typer.echo(f"[FaceKit] {len(cohort_dirs)} cohort(s), {resolution}x{resolution}, "
               f"margin={margin}")
    output_dir.mkdir(parents=True, exist_ok=True)
    extractor = MediaPipeLandmarkExtractor()

    records: List[PackRecord] = []
    for cohort_dir in cohort_dirs:
        cohort = cohort_dir.name
        paths = [Path(p) for p in list_imgpaths(cohort_dir)]
        crop_dir = output_dir / cohort
        with typer.progressbar(paths, label=f"[FaceKit] {cohort}") as bar:
            recs = prepare_cohort(bar, crop_dir, extractor, resolution, margin, cohort)
        n_ok = sum(r.status == "ok" for r in recs)
        records.extend(recs)
        if n_ok == 0:
            typer.secho(f"[FaceKit] {cohort}: no faces found, no zip written",
                        fg=typer.colors.YELLOW, err=True)
            continue
        zip_path = output_dir / f"{cohort}.zip"
        build_dataset_zip(crop_dir, zip_path)
        typer.echo(f"[FaceKit] {cohort}: {n_ok}/{len(paths)} faces -> {zip_path}")

    write_log(records, output_dir / "pack_log.csv")
    skipped = [r for r in records if r.status != "ok"]
    if skipped:
        typer.secho(f"[FaceKit] {len(skipped)} image(s) skipped; see pack_log.csv",
                    fg=typer.colors.YELLOW)
