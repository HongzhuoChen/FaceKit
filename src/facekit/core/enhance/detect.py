"""The two tests that gate the enhancement steps."""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np


def is_grayscale(bgr: np.ndarray, saturation_threshold: float = 0.03) -> bool:
    """True for single-channel files and for colour files whose mean HSV
    saturation is below ``saturation_threshold`` (scale 0-1). Faded or
    yellowed prints fall under the threshold and are recolorized, which is
    the intended behaviour."""
    if bgr.ndim == 2 or bgr.shape[2] == 1:
        return True
    sat = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[:, :, 1]
    return float(sat.mean()) / 255.0 < saturation_threshold


def face_short_side(bgr: np.ndarray, extractor) -> Optional[int]:
    """Shorter side of the MediaPipe landmark bounding box in pixels, or None
    when no face is found. Used instead of the image size because GFPGAN
    restores faces: a large group photograph with an 80-pixel face is what
    needs restoring, not a 300-pixel portrait."""
    lm = extractor.extract_from_array(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    if lm is None:
        return None
    w = int(lm[:, 0].max() - lm[:, 0].min())
    h = int(lm[:, 1].max() - lm[:, 1].min())
    return min(w, h)
