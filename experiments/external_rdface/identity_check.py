"""Are several images in one RDFace folder photographs of the same patient?

RDFace ships no patient identifiers, and its paper neither states that the
images within a disease class are distinct individuals nor describes any
deduplication step. If a folder held several photographs of one person, "images
of the same disease resemble each other" would be trivially true, so the
separation result in run_rdface.py needs this checked.

A difference hash cannot do it: it settles that no image appears twice and has
no discriminative power beyond that. This uses ArcFace identity embeddings
instead, with the same recipe as the manuscript's privacy assessment
(InsightFace buffalo_l, the escalating-resize detection fallback from
face_audit/compute_embeddings.py, cosine distance).

The threshold is calibrated the way the privacy assessment calibrates its
operating points: from a distribution of pairs known to be different people.
Two images in different disease folders cannot be the same patient, so the
threshold is set at a low percentile of the between-folder distances and the
false positive rate among known-different pairs is fixed by construction.

Requires insightface and onnxruntime, which are not FaceKit dependencies:

    python -m venv --system-site-packages ~/.venvs/face_audit
    ~/.venvs/face_audit/bin/pip install insightface onnxruntime
    ~/.venvs/face_audit/bin/python experiments/external_rdface/identity_check.py \
        --images <rd_images dir>
"""
from __future__ import annotations

import argparse
import itertools
import os
from pathlib import Path

os.environ.setdefault("ORT_DISABLE_CPU_AFFINITY", "1")

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

HERE = Path(__file__).resolve().parent


def resize_to_safe_dimensions(image_bgr: np.ndarray, target_size: int = 224) -> np.ndarray:
    """face_audit/utils.py: scale the long side, keep both sides a multiple of 32."""
    h, w = image_bgr.shape[:2]
    scale = target_size / max(h, w)
    return cv2.resize(image_bgr, (max(32, (int(w * scale) // 32) * 32),
                                  max(32, (int(h * scale) // 32) * 32)))


def arcface_embedding(app, image_bgr: np.ndarray) -> np.ndarray | None:
    """face_audit/compute_embeddings.py: try 224, then native, then larger."""
    for attempt in (lambda: app.get(resize_to_safe_dimensions(image_bgr, 224)),
                    lambda: app.get(image_bgr),
                    *[(lambda t=t: app.get(resize_to_safe_dimensions(image_bgr, t)))
                      for t in (320, 512, 800)]):
        try:
            faces = attempt()
        except Exception:
            continue
        if faces:
            return faces[0].embedding
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, type=Path)
    ap.add_argument("--fpr-percentile", type=float, default=0.1,
                    help="false positive rate, in percent, among known-different pairs")
    ap.add_argument("--out-dir", type=Path, default=HERE / "results")
    args = ap.parse_args()

    from insightface.app import FaceAnalysis
    app = FaceAnalysis(name="buffalo_l")
    app.prepare(ctx_id=-1, det_size=(224, 224))

    paths = sorted(p for p in args.images.rglob("*")
                   if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    records = []
    for path in tqdm(paths, desc="embedding"):
        image = cv2.imread(str(path))
        if image is None:
            continue
        emb = arcface_embedding(app, image)
        if emb is None:
            continue
        records.append((path.parent.name, path.stem, emb / np.linalg.norm(emb)))

    disease = np.array([r[0] for r in records])
    image_id = np.array([r[1] for r in records])
    X = np.vstack([r[2] for r in records])
    print(f"\nimages with an ArcFace embedding: {len(X)} / {len(paths)}")

    cosine = 1.0 - X @ X.T
    iu = np.triu_indices(len(X), 1)
    same_folder = (disease[:, None] == disease[None, :])[iu]
    d = cosine[iu]
    within, between = d[same_folder], d[~same_folder]

    threshold = float(np.percentile(between, args.fpr_percentile))
    flagged = within <= threshold
    print(f"\nwithin-folder pairs  : {len(within)}   median cosine distance {np.median(within):.3f}")
    print(f"between-folder pairs : {len(between)}   median cosine distance {np.median(between):.3f}")
    print(f"threshold at the {args.fpr_percentile}th percentile of between-folder "
          f"distances = {threshold:.4f}")
    print(f"within-folder pairs below it: {int(flagged.sum())} "
          f"({100 * flagged.mean():.2f}%), against {args.fpr_percentile}% expected "
          f"if every folder held distinct patients")

    a_idx, b_idx = iu[0][same_folder], iu[1][same_folder]
    pairs = pd.DataFrame(dict(disease=disease[a_idx], a=image_id[a_idx],
                              b=image_id[b_idx], cosine=within,
                              flagged=flagged)).sort_values("cosine")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(args.out_dir / "identity_pairs.csv", index=False)

    drop = sorted({b for b in pairs.loc[pairs.flagged, "b"]})
    pd.Series(drop, name="image_id").to_csv(
        args.out_dir / "identity_duplicates.csv", index=False)
    print(f"\none image of each flagged pair -> {len(drop)} to drop "
          f"-> {args.out_dir / 'identity_duplicates.csv'}")
    if len(pairs):
        print("\nclosest within-folder pairs:")
        print(pairs.head(10).to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
