"""Does the geometric representation carry disease information outside GMDB?

RDFace is an independent rare-disease image collection: 103 diseases, 456
images, only three of its diseases overlap the 50 GMDB cohorts, and its images
are far more heterogeneous (short side 41 to 837 px, a quarter effectively
grayscale). Its images were already screened as frontal portraits by the
curators under clinical supervision, so the share FaceKit's pose gate rejects
measures disagreement with a human frontality judgement rather than the share
of unusable images. It carries no HPO annotations and no patient identifiers, so it
cannot test the term-level claims of Section 3.1. What it can test is whether
the 120 measurements separate diseases at all on a corpus the pipeline was
never tuned against.

The test is a permutation on the disease labels: are two images of the same
disease closer in the 120-dimensional feature space than two images of
different diseases? Two confounds are controlled.

  acquisition   images of one disease are often scraped from one source, so a
                folder tends to be homogeneous in resolution and colour. The
                labels are therefore permuted only WITHIN strata of image
                resolution and colour, so acquisition similarity cannot
                produce the effect.
  duplicates    RDFace ships no patient identifiers, so several images in a
                folder could be the same patient. Near-duplicates are detected
                with a difference hash and one image of each pair is dropped.
                This bounds the threat; it cannot remove it, because two
                different photographs of one patient are not near-duplicates.
  resolution    RDFace short sides run from 41 px, and resolution_drift.py
                shows the catalogue degrading below about 128 px. The test is
                therefore repeated behind a ladder of resolution gates, up to
                the 224 px the GMDB patients are measured at, to check that the
                separation is not produced by the unreliable tail.

Usage:
    PYTHONPATH=src python experiments/external_rdface/run_rdface.py \
        --features <phenotypes_all.csv> --images <rd_images dir>
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy.spatial.distance import pdist, squareform

HERE = Path(__file__).resolve().parent
NON_FEATURE = {"disease", "image_id", "frontal_ok", "derotated",
               "pose_yaw", "pose_pitch", "pose_roll"}
MIN_PER_DISEASE = 3


def dhash(path: Path, size: int = 8) -> int | None:
    """Difference hash: 1 bit per adjacent-pixel comparison on a 9x8 grey grid."""
    try:
        im = Image.open(path).convert("L").resize((size + 1, size), Image.LANCZOS)
    except Exception:
        return None
    a = np.asarray(im, dtype=np.int16)
    bits = (a[:, 1:] > a[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def image_properties(images: Path) -> pd.DataFrame:
    rows = []
    for path in sorted(images.rglob("*")):
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        try:
            im = Image.open(path)
        except Exception:
            continue
        px = np.asarray(im.convert("RGB").resize((32, 32)), dtype=int)
        grey = int(np.abs(px[:, :, 0] - px[:, :, 1]).max() < 8
                   and np.abs(px[:, :, 1] - px[:, :, 2]).max() < 8)
        rows.append(dict(image_id=path.stem, short_side=min(im.size),
                         grey=grey, dhash=dhash(path)))
    return pd.DataFrame(rows)


def separation(X: np.ndarray, labels: np.ndarray) -> float:
    """Mean within-label distance minus mean between-label distance."""
    d = squareform(pdist(X))
    iu = np.triu_indices(len(labels), 1)
    same = (labels[:, None] == labels[None, :])[iu]
    dist = d[iu]
    return float(dist[same].mean() - dist[~same].mean())


def permutation_test(X, labels, strata, rng, n=2000):
    observed = separation(X, labels)
    null = np.empty(n)
    for i in range(n):
        shuffled = labels.copy()
        for s in np.unique(strata):
            m = strata == s
            shuffled[m] = rng.permutation(shuffled[m])
        null[i] = separation(X, shuffled)
    p = ((null <= observed).sum() + 1) / (n + 1)
    return observed, p, float(null.mean()), float(null.std())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True, type=Path)
    ap.add_argument("--images", required=True, type=Path)
    ap.add_argument("--dhash-threshold", type=int, default=6,
                    help="Hamming distance below which two images are near-duplicates")
    ap.add_argument("--seed", type=int, default=20260830)
    ap.add_argument("--out-dir", type=Path, default=HERE / "results")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    feats_df = pd.read_csv(args.features)
    props = image_properties(args.images)
    df = feats_df.merge(props, on="image_id", how="left")

    print(f"landmarked images     : {len(feats_df)}")
    rejected = len(feats_df) - int(df.frontal_ok.sum())
    print(f"passing the pose gate : {int(df.frontal_ok.sum())} "
          f"({100 * df.frontal_ok.mean():.1f}%)")
    print(f"  rejected as non-frontal, though curated as frontal portraits: "
          f"{rejected} ({100 * rejected / len(feats_df):.1f}%)")
    print(f"diseases retaining >=1: {df[df.frontal_ok == True].disease.nunique()}")  # noqa: E712

    df = df[df.frontal_ok == True].copy()  # noqa: E712

    # --- near-duplicate detection, within each disease folder ------------
    drop = set()
    pairs = []
    for disease, group in df.groupby("disease"):
        rows = group.dropna(subset=["dhash"]).reset_index()
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                dist = hamming(int(rows.dhash[i]), int(rows.dhash[j]))
                if dist <= args.dhash_threshold:
                    pairs.append(dict(disease=disease, a=rows.image_id[i],
                                      b=rows.image_id[j], hamming=dist))
                    drop.add(rows.image_id[j])
    args.out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(pairs).to_csv(args.out_dir / "near_duplicates.csv", index=False)
    print(f"\nnear-duplicate pairs (dhash <= {args.dhash_threshold}): {len(pairs)}"
          f" -> {len(drop)} images dropped")

    features = [c for c in feats_df.columns if c not in NON_FEATURE]
    subsets = [("all frontal images", df),
               ("near-duplicates removed", df[~df.image_id.isin(drop)])]
    subsets += [(f"short side >= {t} px", df[df.short_side >= t])
                for t in (128, 160, 224)]

    results = []
    for label, subset in subsets:
        sub = subset.groupby("disease").filter(lambda g: len(g) >= MIN_PER_DISEASE)
        X = sub[features].to_numpy(float)
        X = (X - X.mean(0)) / X.std(0)
        labels = sub.disease.to_numpy()
        bins = min(4, sub.short_side.nunique())
        strata = (pd.qcut(np.log10(sub.short_side), bins, labels=False,
                          duplicates="drop").astype(str)
                  + "_" + sub.grey.astype(str)).to_numpy()
        obs, p, mu, sd = permutation_test(X, labels, strata, rng)
        z = (obs - mu) / sd
        results.append(dict(subset=label, n_images=len(sub),
                            n_diseases=sub.disease.nunique(),
                            separation=obs, null_mean=mu, null_sd=sd,
                            z_vs_null=z, p=p))
        print(f"\n{label}: {len(sub)} images, {sub.disease.nunique()} diseases")
        print(f"  within - between distance = {obs:+.4f}")
        print(f"  stratified null: mean {mu:+.4f} sd {sd:.4f}  ->  "
              f"{z:.1f} SD below null, p = {p:.4f}")
    pd.DataFrame(results).to_csv(args.out_dir / "disease_separation.csv", index=False)


if __name__ == "__main__":
    main()
