"""Walk a folder, decide per image, and apply colorization then restoration."""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, List, Optional

import cv2

from facekit.core.enhance.detect import face_short_side, is_grayscale


@dataclass
class EnhanceRecord:
    cohort: str
    image: str
    grayscale: Optional[bool]     # None when the file could not be read
    face_px: Optional[int]        # None when no face was found
    colorized: bool
    restored: bool
    status: str                   # "ok" | "unreadable"


@dataclass
class EnhanceOptions:
    colorize: bool = True
    restore: bool = True
    force_colorize: bool = False
    force_restore: bool = False
    min_face: int = 128
    saturation_threshold: float = 0.03


class Enhancer:
    """Holds the (lazily built) models so a run pays for them only if needed."""

    def __init__(self, opts: EnhanceOptions, device: str, extractor,
                 make_colorizer: Callable, make_restorer: Callable):
        self.opts = opts
        self.device = device
        self.extractor = extractor
        self._make_colorizer, self._make_restorer = make_colorizer, make_restorer
        self._colorizer = self._restorer = None

    @property
    def colorizer(self):
        if self._colorizer is None:
            self._colorizer = self._make_colorizer(self.device)
        return self._colorizer

    @property
    def restorer(self):
        if self._restorer is None:
            self._restorer = self._make_restorer(self.device)
        return self._restorer

    def process(self, path: Path, out_path: Path, cohort: str) -> EnhanceRecord:
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            return EnhanceRecord(cohort, path.name, None, None, False, False, "unreadable")
        o = self.opts
        gray = is_grayscale(bgr, o.saturation_threshold)
        do_color = o.colorize and (gray or o.force_colorize)
        if do_color:
            bgr = self.colorizer.colorize(bgr)
        face_px = face_short_side(bgr, self.extractor) if o.restore else None
        do_restore = o.restore and (o.force_restore or (face_px is not None and face_px < o.min_face))
        if do_restore:
            bgr = self.restorer.restore(bgr)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), bgr)
        return EnhanceRecord(cohort, path.name, gray, face_px, do_color, do_restore, "ok")


def enhance_folder(enhancer: Enhancer, image_paths: Iterable[Path], out_dir: Path,
                   cohort: str) -> List[EnhanceRecord]:
    return [enhancer.process(p, out_dir / f"{p.stem}.png", cohort) for p in image_paths]


def write_log(records: List[EnhanceRecord], path: Path) -> None:
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["cohort", "image", "grayscale", "face_px", "colorized", "restored", "status"])
        for r in records:
            w.writerow([r.cohort, r.image, r.grayscale, r.face_px, r.colorized, r.restored, r.status])
