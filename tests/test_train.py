"""`facekit train`: option resolution through the vendored train.py (dry run)
and the FaceKit cosine learning-rate schedule. No GPU is used."""
from __future__ import annotations

import numpy as np
import pytest
from typer.testing import CliRunner

torch = pytest.importorskip("torch")
pytest.importorskip("psutil")

from facekit.cli import app  # noqa: E402
from facekit.core.synth.pack import build_dataset_zip  # noqa: E402
from facekit.vendor import ensure_stylegan3_on_path  # noqa: E402


@pytest.fixture(scope="module")
def tiny_zip(tmp_path_factory):
    import cv2

    root = tmp_path_factory.mktemp("ds")
    crops = root / "crops"
    crops.mkdir()
    for i in range(8):
        cv2.imwrite(str(crops / f"{i}.png"), np.full((32, 32, 3), 30 * i, np.uint8))
    zip_path = root / "tiny.zip"
    build_dataset_zip(crops, zip_path)
    return zip_path


def test_lr_factor():
    ensure_stylegan3_on_path()
    from training.training_loop import lr_factor  # vendored

    assert lr_factor("constant", 12345, 10) == 1.0
    assert lr_factor("cosine", 0, 100) == pytest.approx(1.0)
    assert lr_factor("cosine", 50_000, 100) == pytest.approx(0.5)
    assert lr_factor("cosine", 100_000, 100) == pytest.approx(0.0)
    assert lr_factor("cosine", 200_000, 100) == pytest.approx(0.0)  # clamps past the end
    with pytest.raises(ValueError):
        lr_factor("linear", 0, 1)


def test_infinite_sampler_constructs_on_current_torch():
    ensure_stylegan3_on_path()
    from torch_utils import misc  # vendored

    sampler = misc.InfiniteSampler(dataset=list(range(5)), rank=0, num_replicas=1, seed=0)
    it = iter(sampler)
    assert {next(it) for _ in range(50)} == set(range(5))


def test_dry_run_resolves_facekit_defaults(tiny_zip, tmp_path):
    r = CliRunner().invoke(app, [
        "train", "--data", str(tiny_zip), "-o", str(tmp_path / "runs"),
        "--batch", "8", "--batch-gpu", "4", "--dry-run",
    ])
    assert r.exit_code == 0, r.output
    # The wrapper echoes the resolved train.py argv; check the manuscript settings.
    for flag in ("--cfg=stylegan3-t", "--gamma=2.0", "--mirror=1", "--kimg=5000",
                 "--glr=0.0025", "--dlr=0.002", "--map-depth=8",
                 "--lr-schedule=cosine", "--metrics=none", "--dry-run"):
        assert flag in r.output, flag
    assert not any((tmp_path / "runs").iterdir())  # dry run creates no run folder


def test_bad_cfg_rejected(tiny_zip, tmp_path):
    r = CliRunner().invoke(app, ["train", "--data", str(tiny_zip), "-o", str(tmp_path),
                                 "--cfg", "stylegan4", "--dry-run"])
    assert r.exit_code == 2 and "--cfg" in r.output


@pytest.mark.skipif(torch.cuda.is_available(), reason="checks the no-GPU error path")
def test_refuses_to_train_without_gpu(tiny_zip, tmp_path):
    r = CliRunner().invoke(app, ["train", "--data", str(tiny_zip), "-o", str(tmp_path)])
    assert r.exit_code == 2 and "GPU" in r.output
