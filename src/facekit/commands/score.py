"""
FaceKit CLI command: score

Convert a phenotype CSV produced by ``extract-features`` into feature-level
z-scores against a normative reference, so that every measurement is expressed
on the same scale: how many reference standard deviations a face departs from
the control population.

Examples
--------
    # Against the packaged FairFace reference
    facekit score -i results/phenotypes_all.csv -o results/

    # Against a reference built from your own control phenotypes
    facekit score -i results/phenotypes_all.csv -o results/ \
        --reference controls/phenotypes_all.csv
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import typer

from facekit.core.geometric.batch import LEADING_COLUMNS
from facekit.core.geometric.defaults import DEFAULT_REFERENCE

REFERENCE_COLUMNS = ("feature", "mean", "sd", "n")


def _load_reference(path: Path) -> pd.DataFrame:
    """Return a ``feature, mean, sd, n`` table.

    ``path`` is either such a table already, or a phenotype CSV of controls
    from which one is derived. The pose gate is applied when deriving, because
    a reference measured under looser acquisition conditions than the faces
    scored against it contributes an inflated SD.
    """
    df = pd.read_csv(path)
    if "feature" in df.columns:
        missing = [c for c in REFERENCE_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"reference table {path} is missing columns: {missing}")
        return df

    if "frontal_ok" not in df.columns:
        raise ValueError(
            f"{path} is neither a reference table (needs {list(REFERENCE_COLUMNS)}) "
            "nor a phenotype CSV (needs 'frontal_ok')"
        )
    controls = df[df["frontal_ok"] == True]  # noqa: E712
    if controls.empty:
        raise ValueError(f"no frontal control images in {path}")
    feat = [c for c in df.columns if c not in set(LEADING_COLUMNS)]
    return pd.DataFrame({
        "feature": feat,
        "mean": controls[feat].mean().values,
        "sd": controls[feat].std(ddof=1).values,
        "n": controls[feat].notna().sum().values,
    })


def score(
    input_path: Path = typer.Option(
        ...,
        "--input", "-i",
        exists=True, dir_okay=False, file_okay=True,
        help="Phenotype CSV written by 'facekit extract-features'.",
    ),
    output_dir: Path = typer.Option(
        ...,
        "--output", "-o",
        file_okay=False, dir_okay=True,
        help="Output directory. CSV is auto-named 'phenotypes_z.csv'.",
    ),
    reference: Path = typer.Option(
        DEFAULT_REFERENCE, "--reference",
        exists=True, dir_okay=False, file_okay=True,
        help=(
            "Reference: a 'feature,mean,sd,n' table, or a phenotype CSV of "
            "control images to derive one from. Defaults to the packaged "
            "FairFace reference."
        ),
    ),
):
    """Express each measurement as a z-score against a normative reference.

    Output schema: the leading columns of the input, followed by one z-score
    column per feature under the feature's own name, so the table is a drop-in
    replacement for the raw phenotype CSV in downstream analysis. A feature
    absent from the reference, or with a zero or missing reference SD, is
    written as NaN rather than dropped, keeping the column order stable.

    ``frontal_ok`` is carried through unchanged and is not used to filter the
    scored faces: rows that failed the pose gate are scored and left for the
    caller to exclude.
    """
    output_dir = output_dir.expanduser()
    reference = reference.expanduser()

    typer.echo(f"[FaceKit] score: {input_path}")
    typer.echo(f"[FaceKit] Reference: {reference}")

    try:
        ref = _load_reference(reference)
        df = pd.read_csv(input_path)
    except (ValueError, FileNotFoundError) as e:
        typer.secho(f"[FaceKit] {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(2)

    lead = [c for c in LEADING_COLUMNS if c in df.columns]
    feat_cols = [c for c in df.columns if c not in set(LEADING_COLUMNS)]
    if not feat_cols:
        typer.secho(f"[FaceKit] no feature columns in {input_path}",
                    fg=typer.colors.RED, err=True)
        raise typer.Exit(2)

    mu = ref.set_index("feature")["mean"]
    sd = ref.set_index("feature")["sd"].where(lambda s: s > 0)

    z = {}
    scored = []
    for c in feat_cols:
        if c in mu.index and pd.notna(sd.get(c)):
            z[c] = (df[c] - mu[c]) / sd[c]
            scored.append(c)
        else:
            z[c] = pd.Series(pd.NA, index=df.index)
    out = pd.concat([df[lead], pd.DataFrame(z, index=df.index)], axis=1)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_csv = output_dir / "phenotypes_z.csv"
    out.to_csv(out_csv, index=False)

    unscored = len(feat_cols) - len(scored)
    typer.echo(
        f"[FaceKit] Done: {len(out)} rows, {len(scored)} features z-scored"
        + (f", {unscored} without a reference SD" if unscored else "")
        + f" -> {out_csv}"
    )
