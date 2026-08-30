"""How much does a measurement change when the same face is measured smaller?

Paired within-image: the native-resolution value of a face is the ground truth
for the downsampled values of that same face, so nothing here depends on an
annotation or on a reference population. Drift is expressed in FairFace
reference SD, the unit every effect size in the paper is reported in, so it can
be read directly against them.

Two quantities per feature and resolution:

  bias      the median signed drift. A systematic offset of the feature's mean
            is what biases a z-score, because it does not average out over
            patients.
  spread    the interquartile range of the drift. Random error, which widens
            confidence intervals but does not shift an estimate.

The median is used rather than the mean: a few features (the malar_bulge group)
have a near-zero denominator and produce drifts of hundreds of SD on individual
images, which dominates any mean.

Usage:
    PYTHONPATH=src python experiments/reference_sensitivity/resolution_drift.py --n 400
"""
from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import mediapipe as mp
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
CHOP = Path(os.environ.get(
    "FACEKIT_CHOP_DIR",
    "/vast/projects/kai/multimodal-machine-learn/hongzhuo/chop_22q_analysis"))

_current = {"image": None}
mp.Image.create_from_file = staticmethod(lambda path: _current["image"])
from facekit.core.geometric.extractor import GeometricFeatureExtractor  # noqa: E402

NATIVE = 448
LADDER = [224, 160, 128, 96, 64]
NON_FEATURE = {"disease", "image_id", "frontal_ok", "derotated",
               "pose_yaw", "pose_pitch", "pose_roll"}


def to_srgb(pil: Image.Image) -> "mp.Image":
    arr = np.asarray(pil.convert("RGB"), dtype=np.uint8)
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(arr))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--seed", type=int, default=20260830)
    ap.add_argument("--out-dir", type=Path, default=HERE / "results")
    args = ap.parse_args()

    ref = pd.read_csv(CHOP / "feats_fairface_v2" / "phenotypes_all.csv")
    ref = ref[ref.frontal_ok == True]  # noqa: E712
    features = [c for c in ref.columns if c not in NON_FEATURE]
    ref_sd = ref[features].std(ddof=1)

    paths = [p for p in (CHOP / "fairface_controls").rglob("*")
             if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    random.Random(args.seed).shuffle(paths)

    extractor = GeometricFeatureExtractor()
    rows, used = [], 0
    for path in tqdm(paths, desc="images"):
        if used >= args.n:
            break
        try:
            im = Image.open(path)
        except Exception:
            continue
        if min(im.size) < NATIVE:
            continue
        measured = {}
        for side in [NATIVE] + LADDER:
            scale = side / min(im.size)
            pil = im if side == NATIVE else im.resize(
                (max(1, round(im.width * scale)), max(1, round(im.height * scale))),
                Image.LANCZOS)
            _current["image"] = to_srgb(pil)
            try:
                measured[side] = extractor.extract(str(path), frontal_check=False)
            except Exception:
                measured[side] = None
        if any(measured[s] is None for s in [NATIVE] + LADDER):
            continue
        base = measured[NATIVE]
        for side in LADDER:
            row = {"image_id": path.stem, "resolution": side}
            for col in features:
                b, v = base.get(col), measured[side].get(col)
                row[col] = ((v - b) / ref_sd[col]
                            if b is not None and v is not None and ref_sd[col] > 0
                            else np.nan)
            rows.append(row)
        used += 1

    long = pd.DataFrame(rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    long.to_csv(args.out_dir / "resolution_drift_long.csv", index=False)

    summary = []
    for col in features:
        entry = {"feature": col}
        for side in LADDER:
            d = long.loc[long.resolution == side, col]
            entry[f"bias_{side}"] = d.median()
            entry[f"iqr_{side}"] = d.quantile(.75) - d.quantile(.25)
        entry["res_drift_224"] = abs(entry["bias_224"])
        summary.append(entry)
    out = pd.DataFrame(summary)
    out.to_csv(args.out_dir / "resolution_drift.csv", index=False)

    print(f"\nimages used: {used} (native {NATIVE} px)")
    print(f"\n{'res':>5} {'|bias| med':>11} {'>0.10 SD':>9} {'>0.25 SD':>9} {'>0.50 SD':>9}")
    for side in LADDER:
        b = out[f"bias_{side}"].abs()
        print(f"{side:>5} {b.median():11.3f} {int((b > 0.10).sum()):>9} "
              f"{int((b > 0.25).sum()):>9} {int((b > 0.50).sum()):>9}")
    print("\nreference: the flagship Crouzon hypertelorism effect is 1.04 SD; "
          "the mean |z| over all terms is 0.469 SD")
    print("\nmost biased at 224 px (the patient working resolution):")
    top = out.reindex(out.bias_224.abs().sort_values(ascending=False).index).head(10)
    print(top[["feature", "bias_224", "iqr_224"]]
          .to_string(index=False, float_format=lambda v: f"{v:+.3f}"))


if __name__ == "__main__":
    main()
