"""`facekit pack`: crop geometry, zip building via the vendored dataset_tool,
and the CLI's layout checks. MediaPipe is not exercised here; the real-image
run is documented in the commit that added the command."""
from __future__ import annotations

import zipfile

import numpy as np
import pytest
from typer.testing import CliRunner

from facekit.core.synth.pack import (
    build_dataset_zip,
    crop_square,
    face_square,
    resize_square,
)


def test_face_square_is_square_and_centered():
    lm = np.array([[10, 20], [110, 20], [10, 60], [110, 60]])  # 100 x 40 box
    x0, y0, side = face_square(lm, margin=0.0)
    assert side == 100
    assert (x0, y0) == (10, -10)  # centred on (60, 40)
    _, _, side_m = face_square(lm, margin=0.3)
    assert side_m == 160


def test_crop_square_pads_past_border():
    img = np.zeros((50, 50, 3), dtype=np.uint8)
    img[:, -1] = 255  # bright right edge, should be replicated
    crop = crop_square(img, x0=30, y0=-10, side=40)
    assert crop.shape == (40, 40, 3)
    assert crop[:, -1].min() == 255


def test_resize_square():
    assert resize_square(np.zeros((300, 300, 3), np.uint8), 256).shape == (256, 256, 3)
    assert resize_square(np.zeros((100, 100, 3), np.uint8), 256).shape == (256, 256, 3)


def test_build_dataset_zip(tmp_path):
    pytest.importorskip("torch")
    import cv2

    crops = tmp_path / "crops"
    crops.mkdir()
    for i in range(3):
        cv2.imwrite(str(crops / f"{i}.png"), np.full((64, 64, 3), i * 40, np.uint8))
    zip_path = tmp_path / "c.zip"
    build_dataset_zip(crops, zip_path)
    names = zipfile.ZipFile(zip_path).namelist()
    assert sum(n.endswith(".png") for n in names) == 3
    assert "dataset.json" in names


def test_cli_rejects_mixed_layout_and_bad_resolution(tmp_path):
    from facekit.cli import app
    import cv2

    root = tmp_path / "imgs"
    (root / "a").mkdir(parents=True)
    blank = np.zeros((8, 8, 3), np.uint8)
    cv2.imwrite(str(root / "x.png"), blank)
    cv2.imwrite(str(root / "a" / "y.png"), blank)
    r = CliRunner().invoke(app, ["pack", "-i", str(root), "-o", str(tmp_path / "o")])
    assert r.exit_code == 2 and "one layout" in r.output
    r = CliRunner().invoke(app, ["pack", "-i", str(root / "a"), "-o", str(tmp_path / "o"),
                                 "--resolution", "200"])
    assert r.exit_code == 2 and "power of two" in r.output
