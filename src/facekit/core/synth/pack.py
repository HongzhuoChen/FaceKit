"""Turn a folder of face photographs into a StyleGAN3 training set.

For each image the MediaPipe landmarks give a face box; the box is made
square, widened by ``margin`` on every side, clamped into the image (with
edge replication where it still overhangs), and resized to ``resolution``.
The crops are written as PNG, then the vendored ``dataset_tool.py`` packs
them into the uncompressed zip that ``train.py`` reads.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import cv2
import numpy as np

from facekit.vendor import ensure_stylegan3_on_path


def face_square(landmarks_xy: np.ndarray, margin: float) -> tuple[int, int, int]:
    """(x0, y0, side) of the square crop around a (N, 2) pixel landmark array."""
    x_min, y_min = landmarks_xy.min(axis=0)
    x_max, y_max = landmarks_xy.max(axis=0)
    cx, cy = (x_min + x_max) / 2.0, (y_min + y_max) / 2.0
    side = max(x_max - x_min, y_max - y_min) * (1.0 + 2.0 * margin)
    side = max(int(round(side)), 1)
    return int(round(cx - side / 2.0)), int(round(cy - side / 2.0)), side


def crop_square(img: np.ndarray, x0: int, y0: int, side: int) -> np.ndarray:
    """Crop ``img[y0:y0+side, x0:x0+side]``, replicating edges past the border."""
    h, w = img.shape[:2]
    pad_l, pad_t = max(0, -x0), max(0, -y0)
    pad_r, pad_b = max(0, x0 + side - w), max(0, y0 + side - h)
    if pad_l or pad_t or pad_r or pad_b:
        img = cv2.copyMakeBorder(img, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_REPLICATE)
        x0, y0 = x0 + pad_l, y0 + pad_t
    return img[y0:y0 + side, x0:x0 + side]


def resize_square(img: np.ndarray, resolution: int) -> np.ndarray:
    interp = cv2.INTER_AREA if img.shape[0] > resolution else cv2.INTER_CUBIC
    return cv2.resize(img, (resolution, resolution), interpolation=interp)


@dataclass
class PackRecord:
    cohort: str
    image: str
    status: str  # "ok" | "no_face" | "unreadable"


def prepare_cohort(
    image_paths: Iterable[Path],
    crop_dir: Path,
    extractor,
    resolution: int,
    margin: float,
    cohort: str,
) -> List[PackRecord]:
    """Write one ``<stem>.png`` crop per detected face into ``crop_dir``."""
    crop_dir.mkdir(parents=True, exist_ok=True)
    records: List[PackRecord] = []
    for path in image_paths:
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            records.append(PackRecord(cohort, path.name, "unreadable"))
            continue
        lm = extractor.extract_from_array(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        if lm is None:
            records.append(PackRecord(cohort, path.name, "no_face"))
            continue
        x0, y0, side = face_square(lm, margin)
        crop = resize_square(crop_square(bgr, x0, y0, side), resolution)
        cv2.imwrite(str(crop_dir / f"{path.stem}.png"), crop)
        records.append(PackRecord(cohort, path.name, "ok"))
    return records


def build_dataset_zip(crop_dir: Path, zip_path: Path) -> None:
    """Pack a folder of equal-size square PNGs with StyleGAN3's dataset_tool."""
    ensure_stylegan3_on_path()
    from dataset_tool import convert_dataset  # noqa: E402  (vendored, click command)

    if zip_path.exists():
        zip_path.unlink()
    convert_dataset.main(
        ["--source", str(crop_dir), "--dest", str(zip_path)],
        standalone_mode=False,
    )


def write_log(records: List[PackRecord], path: Path) -> None:
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["cohort", "image", "status"])
        w.writerows((r.cohort, r.image, r.status) for r in records)
