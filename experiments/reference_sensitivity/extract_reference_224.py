"""Re-extract the FairFace reference at the patient working resolution.

The GMDB patient images are 224x224; the FairFace control images are 448x448.
Every z-score in the validity analysis therefore takes its numerator from a
224 px measurement and its denominator from a 448 px one. This script rebuilds
the reference from the same images downsampled to a 224 px short side, so the
two sides of the z-score are measured under the same conditions.

Only the input resolution changes: same images, same extractor, same feature
formulas. Output matches the schema of feats_fairface_v2/phenotypes_all.csv.

Note: mediapipe 0.10.33's mp.Image.create_from_file returns GRAY8 for JPEG and
SRGBA for PNG, neither of which the landmarker accepts, so every image would
error and the run would silently produce no rows. We therefore feed the
extractor an SRGB image built with PIL. See docs note in the module header of
run_arms.py.

Usage:
    PYTHONPATH=src python experiments/reference_sensitivity/extract_reference_224.py \
        --images <fairface_controls dir> --out results/reference_224.csv [--short-side 224]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import mediapipe as mp
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

_current = {"image": None}
mp.Image.create_from_file = staticmethod(lambda path: _current["image"])

from facekit.core.geometric.extractor import GeometricFeatureExtractor  # noqa: E402


def to_srgb(pil: Image.Image) -> "mp.Image":
    arr = np.asarray(pil.convert("RGB"), dtype=np.uint8)
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(arr))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--short-side", type=int, default=224)
    args = ap.parse_args()

    cohorts = sorted(d for d in args.images.iterdir() if d.is_dir())
    extractor = GeometricFeatureExtractor()
    records = []
    for cohort in cohorts:
        files = sorted(f for f in cohort.iterdir()
                       if f.suffix.lower() in {".jpg", ".jpeg", ".png"})
        for path in tqdm(files, desc=cohort.name, leave=False):
            try:
                im = Image.open(path)
            except Exception:
                continue
            scale = args.short_side / min(im.size)
            if scale < 1.0:
                im = im.resize((max(1, round(im.width * scale)),
                                max(1, round(im.height * scale))), Image.LANCZOS)
            _current["image"] = to_srgb(im)
            try:
                feats = extractor.extract(str(path), frontal_check=False)
            except Exception as exc:
                print(f"skip {path.name}: {exc}", file=sys.stderr)
                continue
            if feats is None:
                continue
            records.append({"disease": cohort.name, "image_id": path.stem, **feats})

    df = pd.DataFrame(records)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"{len(df)} rows -> {args.out}")
    if "frontal_ok" in df:
        print(f"frontal_ok: {int(df.frontal_ok.sum())}")


if __name__ == "__main__":
    main()
