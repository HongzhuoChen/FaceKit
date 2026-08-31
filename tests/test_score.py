"""Tests for the `facekit score` command."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import typer

from facekit.commands.score import _load_reference, score
from facekit.core.geometric.batch import LEADING_COLUMNS
from facekit.core.geometric.defaults import DEFAULT_REFERENCE


def _phenotypes(values, frontal=True):
    """Minimal phenotype table: the leading columns plus two features."""
    n = len(values)
    return pd.DataFrame({
        "disease": ["d"] * n,
        "image_id": [f"i{i}" for i in range(n)],
        "frontal_ok": [frontal] * n,
        "derotated": [True] * n,
        "pose_yaw": [0.0] * n,
        "pose_pitch": [0.0] * n,
        "pose_roll": [0.0] * n,
        "feat_a": values,
        "feat_b": [1.0] * n,
    })


def test_packaged_reference_covers_every_feature():
    ref = pd.read_csv(DEFAULT_REFERENCE)
    assert list(ref.columns) == ["feature", "mean", "sd", "n"]
    assert len(ref) == 120
    assert ref.feature.is_unique
    assert (ref.sd > 0).all()
    assert (ref.n > 0).all()


def test_z_scores_are_exact(tmp_path):
    ref = pd.DataFrame({"feature": ["feat_a", "feat_b"],
                        "mean": [10.0, 0.0], "sd": [2.0, 1.0], "n": [100, 100]})
    ref_csv = tmp_path / "ref.csv"
    ref.to_csv(ref_csv, index=False)

    pheno = tmp_path / "pheno.csv"
    _phenotypes([10.0, 12.0, 6.0]).to_csv(pheno, index=False)

    score(input_path=pheno, output_dir=tmp_path, reference=ref_csv)

    out = pd.read_csv(tmp_path / "phenotypes_z.csv")
    assert list(out.columns) == LEADING_COLUMNS + ["feat_a", "feat_b"]
    np.testing.assert_allclose(out.feat_a, [0.0, 1.0, -2.0])


def test_non_frontal_rows_are_scored_not_dropped(tmp_path):
    """The pose gate is the caller's to apply; score must not filter silently."""
    ref = pd.DataFrame({"feature": ["feat_a", "feat_b"],
                        "mean": [10.0, 0.0], "sd": [2.0, 1.0], "n": [100, 100]})
    ref_csv = tmp_path / "ref.csv"
    ref.to_csv(ref_csv, index=False)

    pheno = tmp_path / "pheno.csv"
    _phenotypes([10.0, 12.0], frontal=False).to_csv(pheno, index=False)

    score(input_path=pheno, output_dir=tmp_path, reference=ref_csv)
    assert len(pd.read_csv(tmp_path / "phenotypes_z.csv")) == 2


def test_feature_without_a_reference_becomes_nan(tmp_path):
    ref = pd.DataFrame({"feature": ["feat_a"], "mean": [10.0], "sd": [2.0], "n": [100]})
    ref_csv = tmp_path / "ref.csv"
    ref.to_csv(ref_csv, index=False)

    pheno = tmp_path / "pheno.csv"
    _phenotypes([10.0, 12.0]).to_csv(pheno, index=False)

    score(input_path=pheno, output_dir=tmp_path, reference=ref_csv)
    out = pd.read_csv(tmp_path / "phenotypes_z.csv")
    assert out.feat_b.isna().all()
    np.testing.assert_allclose(out.feat_a, [0.0, 1.0])


def test_reference_from_a_phenotype_csv_applies_the_pose_gate(tmp_path):
    """Controls that failed the pose gate carry inflated SD and must be excluded."""
    controls = pd.concat([
        _phenotypes([8.0, 10.0, 12.0], frontal=True),
        _phenotypes([100.0, -100.0], frontal=False),
    ], ignore_index=True)
    ctl_csv = tmp_path / "controls.csv"
    controls.to_csv(ctl_csv, index=False)

    ref = _load_reference(ctl_csv).set_index("feature")
    assert ref.loc["feat_a", "n"] == 3
    assert ref.loc["feat_a", "mean"] == pytest.approx(10.0)
    assert ref.loc["feat_a", "sd"] == pytest.approx(2.0)


def test_unusable_reference_is_rejected(tmp_path):
    bad = tmp_path / "bad.csv"
    pd.DataFrame({"foo": [1], "bar": [2]}).to_csv(bad, index=False)
    with pytest.raises(ValueError, match="neither a reference table"):
        _load_reference(bad)

    partial = tmp_path / "partial.csv"
    pd.DataFrame({"feature": ["feat_a"], "mean": [1.0]}).to_csv(partial, index=False)
    with pytest.raises(ValueError, match="missing columns"):
        _load_reference(partial)


def test_input_without_feature_columns_exits(tmp_path):
    pheno = tmp_path / "pheno.csv"
    _phenotypes([1.0]).drop(columns=["feat_a", "feat_b"]).to_csv(pheno, index=False)
    with pytest.raises(typer.Exit):
        score(input_path=pheno, output_dir=tmp_path, reference=DEFAULT_REFERENCE)
