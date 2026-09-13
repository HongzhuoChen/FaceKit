"""`facekit privacy`: the distance statistics and the cohort matching.

The recognition models and LPIPS are not exercised here (they need the
``privacy`` extra and downloaded weights); the end-to-end run on real
images is documented in the commit that added the command.
"""
from __future__ import annotations

import numpy as np
import pytest
from typer.testing import CliRunner

from facekit.core.privacy.metrics import (
    adversarial_accuracy,
    aa_from_min_dists,
    bootstrap_aa,
    ci_from_samples,
    drop_zero_rows,
    min_cross_distances,
    percentile_rows,
)


def test_heldout_rate_equals_percentile_by_construction():
    rng = np.random.default_rng(0)
    ho = rng.random(1000)
    syn = rng.random(300)
    rows = percentile_rows(syn, ho, [1, 2, 5, 10, 20])
    for r in rows:
        assert abs(r["heldout_pct"] - r["percentile"]) <= 0.2
        assert r["synthetic_total"] == 300 and r["heldout_total"] == 1000


def test_flagging_detects_copies_of_training_images():
    rng = np.random.default_rng(1)
    train = rng.normal(size=(50, 8))
    ho = rng.normal(size=(40, 8))
    syn = train[:20] + 1e-3 * rng.normal(size=(20, 8))  # near-copies
    rows = percentile_rows(min_cross_distances(syn, train), min_cross_distances(ho, train), [5])
    assert rows[0]["synthetic_pct"] == 100.0


def test_drop_zero_rows():
    emb = np.array([[1.0, 0.0], [0.0, 0.0], [0.0, 2.0]])
    assert drop_zero_rows(emb).shape == (2, 2)


def test_adversarial_accuracy_extremes():
    rng = np.random.default_rng(2)
    A = rng.normal(size=(60, 5))
    far = A + 100.0
    assert adversarial_accuracy(A, far, metric="euclidean") == 1.0  # separated
    B = rng.normal(size=(60, 5))
    assert 0.3 < adversarial_accuracy(A, B, metric="euclidean") < 0.7  # same distribution
    assert np.isnan(adversarial_accuracy(A[:1], B))


def test_aa_from_min_dists_matches_direct():
    rng = np.random.default_rng(3)
    A, B = rng.normal(size=(30, 4)), rng.normal(size=(25, 4))
    from scipy.spatial.distance import cdist
    d_aa = cdist(A, A); np.fill_diagonal(d_aa, np.inf)
    d_bb = cdist(B, B); np.fill_diagonal(d_bb, np.inf)
    d_ab = cdist(A, B)
    direct = adversarial_accuracy(A, B, metric="euclidean")
    via = aa_from_min_dists(d_ab.min(1), d_aa.min(1), d_ab.min(0), d_bb.min(1))
    assert direct == pytest.approx(via)


def test_bootstrap_ci_brackets_point_estimate():
    rng = np.random.default_rng(4)
    A, B = rng.normal(size=(40, 3)), rng.normal(size=(40, 3))
    bs = bootstrap_aa(A, B, "euclidean", 50, rng)
    ci = ci_from_samples(bs)
    assert ci["ci_lower"] <= ci["median"] <= ci["ci_upper"]


def test_find_cohorts(tmp_path):
    from facekit.core.privacy.audit import AuditConfig, find_cohorts

    for split, names in (("train", ["a", "b", "c"]), ("heldout", ["a", "b"]), ("synthetic", ["a", "b", "d"])):
        for n in names:
            (tmp_path / split / n).mkdir(parents=True)
    cfg = AuditConfig(tmp_path / "train", tmp_path / "heldout", tmp_path / "synthetic", tmp_path / "out")
    c = find_cohorts(cfg)
    assert c["common"] == ["a", "b"]
    assert c["train_only"] == ["c"] and c["synthetic_only"] == ["d"] and c["heldout_only"] == []


def test_cli_rejects_bad_backbone(tmp_path):
    from facekit.cli import app

    for s in ("train", "heldout", "synthetic"):
        (tmp_path / s).mkdir()
    r = CliRunner().invoke(app, ["privacy", "--train", str(tmp_path / "train"),
                                 "--heldout", str(tmp_path / "heldout"),
                                 "--synthetic", str(tmp_path / "synthetic"),
                                 "-o", str(tmp_path / "out"), "--backbone", "facenet"])
    assert r.exit_code == 2 and "backbone" in r.output
