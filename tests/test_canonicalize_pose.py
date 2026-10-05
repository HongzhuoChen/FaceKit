"""Pose canonicalization: the properties the four-arm audit (5.7) bought us.

These lock in behaviour that is easy to regress silently, because every one of
these changes the *definition* of all 120 features without raising anything.
"""
from __future__ import annotations

import numpy as np
import pytest

from facekit.core.geometric.canonical_depth import CANONICAL_DEPTH
from facekit.core.geometric.extractor import GeometricFeatureExtractor


@pytest.fixture(autouse=True)
def _reset_state():
    GeometricFeatureExtractor.FEATURE_COLUMNS = None
    yield
    GeometricFeatureExtractor.FEATURE_COLUMNS = None


def _rot(pitch_deg: float = 0.0, yaw_deg: float = 0.0) -> np.ndarray:
    p, y = np.radians(pitch_deg), np.radians(yaw_deg)
    rx = np.array([[1, 0, 0], [0, np.cos(p), -np.sin(p)], [0, np.sin(p), np.cos(p)]])
    ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
    return ry @ rx


def _synthetic_face(rng: np.random.Generator) -> np.ndarray:
    """478 head-frame points whose depths follow the canonical table."""
    pts = np.zeros((478, 3))
    pts[:, :2] = rng.uniform(-100.0, 100.0, size=(478, 2))
    # Fix the zygomatic pair so the bizygomatic width is a known 200.
    pts[234, :2], pts[454, :2] = [-100.0, 0.0], [100.0, 0.0]
    pts[133, :2], pts[362, :2] = [-20.0, -30.0], [20.0, -30.0]
    pts[:, 2] = CANONICAL_DEPTH * 200.0
    return pts


def _project(pts: np.ndarray, rot: np.ndarray) -> np.ndarray:
    return (pts @ rot.T)[:, :2]


def test_canonical_depth_table_shape():
    assert CANONICAL_DEPTH.shape == (478,)
    assert np.isfinite(CANONICAL_DEPTH).all()


def test_pose_correction_recovers_the_frontal_shape():
    """A face whose depths match the table is recovered under real head pose.

    Compared up to a similarity transform, since global scale is normalized
    away by every feature anyway.
    """
    rng = np.random.default_rng(0)
    pts = _synthetic_face(rng)
    ex = GeometricFeatureExtractor()
    front, _, _ = ex._canonicalize(_project(pts, np.eye(3)), np.eye(4))

    for pitch, yaw in [(15.0, 0.0), (0.0, 20.0), (20.0, 20.0)]:
        rot = _rot(pitch, yaw)
        matrix = np.eye(4)
        matrix[:3, :3] = rot
        got, _, derotated = ex._canonicalize(_project(pts, rot), matrix)
        assert derotated
        k = (got * front).sum() / (got * got).sum()
        err = np.linalg.norm(k * got - front, axis=1).mean() / 200.0
        assert err < 0.01, f"pitch={pitch} yaw={yaw}: shape error {err:.3%}"


def test_features_do_not_read_mediapipe_z():
    """A z column must not change any feature: depth comes from the table."""
    rng = np.random.default_rng(1)
    lm = rng.uniform(0.0, 224.0, size=(478, 2))
    matrix = np.eye(4)
    matrix[:3, :3] = _rot(12.0, 8.0)

    ex = GeometricFeatureExtractor()
    without = ex.extract_from_landmarks(lm, matrix)
    with_junk = ex.extract_from_landmarks(
        np.c_[lm, rng.normal(0.0, 50.0, 478)], matrix
    )
    for key, value in without.items():
        other = with_junk[key]
        if isinstance(value, float) and np.isfinite(value):
            assert other == pytest.approx(value), key


def test_matrix_alone_is_enough_for_full_correction():
    """2-column landmarks + a matrix must NOT degrade to the roll-only path."""
    rng = np.random.default_rng(2)
    lm = rng.uniform(0.0, 224.0, size=(478, 2))
    ex = GeometricFeatureExtractor()
    assert ex.extract_from_landmarks(lm, np.eye(4))["derotated"] is True
    assert ex.extract_from_landmarks(lm)["derotated"] is False


def test_roll_only_fallback_when_matrix_missing():
    rng = np.random.default_rng(3)
    lm = rng.uniform(0.0, 224.0, size=(478, 2))
    feats = GeometricFeatureExtractor().extract_from_landmarks(lm)
    assert feats["derotated"] is False
    assert np.isnan(feats["frontal_ok"])
