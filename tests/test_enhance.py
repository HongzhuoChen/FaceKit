"""`facekit enhance`: the two gating tests and the per-image decision logic.

DDColor and GFPGAN are replaced by stubs; the real-model run is documented
in the commit that added the command.
"""
from __future__ import annotations

import numpy as np
import pytest

from facekit.core.enhance.detect import face_short_side, is_grayscale
from facekit.core.enhance.run import EnhanceOptions, Enhancer, enhance_folder


def test_is_grayscale():
    gray3 = np.full((40, 40, 3), 120, np.uint8)
    assert is_grayscale(gray3)
    assert is_grayscale(np.full((40, 40), 120, np.uint8))
    colour = np.zeros((40, 40, 3), np.uint8)
    colour[:, :, 2] = 200  # saturated red
    assert not is_grayscale(colour)
    faded = np.full((40, 40, 3), 150, np.uint8)
    faded[:, :, 2] += 3  # barely tinted: still "grayscale"
    assert is_grayscale(faded)


class _FakeExtractor:
    def __init__(self, lm):
        self.lm = lm

    def extract_from_array(self, rgb):
        return self.lm


def test_face_short_side():
    lm = np.array([[10, 20], [110, 20], [10, 60], [110, 60]])
    assert face_short_side(np.zeros((100, 200, 3), np.uint8), _FakeExtractor(lm)) == 40
    assert face_short_side(np.zeros((100, 200, 3), np.uint8), _FakeExtractor(None)) is None


class _Stub:
    def __init__(self):
        self.calls = 0

    def colorize(self, bgr):
        self.calls += 1
        out = bgr.copy()
        out[:, :, 2] = 255
        return out

    def restore(self, bgr):
        self.calls += 1
        return np.ascontiguousarray(bgr[::-1])


def _write(tmp_path, name, bgr):
    import cv2
    p = tmp_path / name
    cv2.imwrite(str(p), bgr)
    return p


def test_decisions(tmp_path):
    import cv2

    gray = _write(tmp_path, "gray.png", np.full((60, 60, 3), 100, np.uint8))
    colour = np.zeros((60, 60, 3), np.uint8)
    colour[:, :, 1] = 200
    col = _write(tmp_path, "col.png", colour)
    bad = tmp_path / "bad.png"
    bad.write_bytes(b"not an image")

    stub = _Stub()
    big = np.array([[0, 0], [200, 0], [0, 200], [200, 200]])
    small = np.array([[0, 0], [50, 0], [0, 50], [50, 50]])

    # grayscale + big face -> colorized only
    e = Enhancer(EnhanceOptions(), "cpu", _FakeExtractor(big), lambda d: stub, lambda d: stub)
    recs = enhance_folder(e, [gray, col, bad], tmp_path / "out", "c")
    by = {r.image: r for r in recs}
    assert by["gray.png"].colorized and not by["gray.png"].restored and by["gray.png"].face_px == 200
    assert not by["col.png"].colorized and not by["col.png"].restored
    assert by["bad.png"].status == "unreadable"
    assert (tmp_path / "out" / "gray.png").exists() and (tmp_path / "out" / "col.png").exists()
    assert cv2.imread(str(tmp_path / "out" / "gray.png"))[0, 0, 2] == 255  # stub colour applied

    # small face -> restored; --no-colorize leaves gray untouched
    stub2 = _Stub()
    e = Enhancer(EnhanceOptions(colorize=False), "cpu", _FakeExtractor(small), lambda d: stub2, lambda d: stub2)
    recs = {r.image: r for r in enhance_folder(e, [gray, col], tmp_path / "out2", "c")}
    assert recs["gray.png"].restored and not recs["gray.png"].colorized
    assert recs["col.png"].restored
    assert stub2.calls == 2

    # models are built lazily: nothing needed -> factories never called
    built = []
    e = Enhancer(EnhanceOptions(), "cpu", _FakeExtractor(big),
                 lambda d: built.append("c"), lambda d: built.append("r"))
    enhance_folder(e, [col], tmp_path / "out3", "c")
    assert built == []


def test_force_flags(tmp_path):
    colour = np.zeros((60, 60, 3), np.uint8)
    colour[:, :, 0] = 200
    col = _write(tmp_path, "col.png", colour)
    stub = _Stub()
    big = np.array([[0, 0], [200, 0], [0, 200], [200, 200]])
    e = Enhancer(EnhanceOptions(force_colorize=True, force_restore=True), "cpu",
                 _FakeExtractor(big), lambda d: stub, lambda d: stub)
    r = enhance_folder(e, [col], tmp_path / "out", "c")[0]
    assert r.colorized and r.restored and stub.calls == 2
