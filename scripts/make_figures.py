#!/usr/bin/env python3
"""Regenerate the three FaceKit illustration figures in ``figures/``.

All three figures are built from real GestaltMatcher Database (GMDB) source
images that live OUTSIDE this repository; the raw GMDB jpgs are never copied
in, only the rendered PNG outputs. NOTE: average-face.png is an aggregate,
de-identified average; extract-landmarks.png and extract-features.png overlay
annotations on an identifiable patient face -- confirm GMDB consent/DUA before
publishing these. The public cannot re-run this script (it needs GMDB access)
-- that is expected, the PNGs are committed.

Exact command used to generate the committed figures::

    python scripts/make_figures.py \
        --crouzon-image /vast/projects/kai/multimodal-machine-learn/hongzhuo/mm_fusion_top50/visualization_images/pat8721_img13755.jpg \
        --williams-dir  /vast/projects/kai/multimodal-machine-learn/hongzhuo/mm_fusion_top50/combined_data/images/Williams_syndrome

Produces:
    figures/phenotyping.png       one 3-panel figure, identically framed: the 478
                                  landmarks as dots on a Crouzon face, four
                                  measurements on the same face, and the
                                  ``average-face`` of the Williams cohort
    figures/enhance.png           ``facekit enhance`` before/after on that face
                                  (grayscale -> DDColor; 80 px -> GFPGAN); needs
                                  ``facekit[enhance]``

With ``--panels`` it also rasterizes four manuscript figures (needs pymupdf):
    figures/pose-correction.png   manuscript/image/posecorr_geom-1.pdf
    figures/privacy-identity.png  manuscript/image/privacy_c_far_identity.pdf
    figures/privacy-lpips.png     manuscript/image/privacy_d_far_lpips.pdf

``--only enhance`` / ``--only panels`` skip the GMDB-dependent figures.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent.parent
FIGURES = ROOT / "figures"

DEFAULT_CROUZON = Path(
    "/vast/projects/kai/multimodal-machine-learn/hongzhuo/mm_fusion_top50"
    "/visualization_images/pat8721_img13755.jpg"
)
DEFAULT_WILLIAMS = Path(
    "/vast/projects/kai/multimodal-machine-learn/hongzhuo/mm_fusion_top50"
    "/combined_data/images/Williams_syndrome"
)

# Landmark-pair segments for the four facekit features, with indices copied
# verbatim from src/facekit/core/geometric/extractor.py. Each feature value is
# euclidean(pair) / euclidean(BIZYG), i.e. normalized by bizygomatic width.
FEATURES = [
    ("inter_pupillary_distance", 468, 473, "#e6194b"),  # iris centers
    ("inter_canthal_distance", 133, 362, "#3cb44b"),    # inner canthi
    ("nose_base_width", 98, 327, "#4363d8"),            # alae
    ("mouth_width", 61, 291, "#f58231"),                # mouth commissures
]
BIZYG = (234, 454)
DPI = 200



def run(cmd: list[str]) -> None:
    """Run a facekit command with the interpreter running this script."""
    assert cmd[0] == "facekit"
    subprocess.run([sys.executable, "-m", "facekit.cli", *cmd[1:]], check=True, capture_output=True, text=True)


def run_average_face(williams_dir: Path, tmp: Path) -> Path:
    """Run ``facekit average-face`` on the Williams cohort ONLY; return the PNG."""
    cohorts = tmp / "cohorts"
    cohorts.mkdir()
    (cohorts / "Williams_syndrome").symlink_to(williams_dir)
    out = tmp / "avg_out"
    run(["facekit", "average-face", "-i", str(cohorts), "-o", str(out)])
    return next(out.rglob("Williams_syndrome*avg*.png"))


def detect_landmarks(image_path: Path, model_path: Path | None):
    """Return (rgb_image, landmarks_px (N, 2)) via MediaPipe Face Landmarker.

    Resolves the model through facekit's ``ensure_model`` (auto-download)."""
    from facekit.core.morph.landmarks import ensure_model
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision

    model = ensure_model(model_path)
    base = mp_python.BaseOptions(model_asset_path=str(model))
    opts = mp_vision.FaceLandmarkerOptions(base_options=base, num_faces=1)
    detector = mp_vision.FaceLandmarker.create_from_options(opts)
    image = mp.Image.create_from_file(str(image_path))
    res = detector.detect(image)
    detector.close()  # explicit close avoids MediaPipe's teardown error at interpreter exit
    if not res.face_landmarks:
        raise RuntimeError(f"No face detected in {image_path}")
    rgb = image.numpy_view()[:, :, :3]
    h, w = rgb.shape[:2]
    lm = np.array([[p.x * w, p.y * h] for p in res.face_landmarks[0]], dtype=np.float64)
    return rgb, lm


def square_crop(rgb: np.ndarray, lm: np.ndarray, margin: float = 0.12):
    """Square crop around the landmark box, widened by ``margin`` per side.
    Returns (crop, landmarks shifted into crop coordinates)."""
    h, w = rgb.shape[:2]
    x0, y0 = lm.min(axis=0)
    x1, y1 = lm.max(axis=0)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    side = max(x1 - x0, y1 - y0) * (1 + 2 * margin)
    left, top = int(round(cx - side / 2)), int(round(cy - side / 2))
    side = int(round(side))
    pad = max(0, -left, -top, left + side - w, top + side - h)
    if pad:
        rgb = np.pad(rgb, ((pad, pad), (pad, pad), (0, 0)), mode="edge")
        left, top = left + pad, top + pad
    crop = rgb[top:top + side, left:left + side]
    return crop, lm + pad - np.array([left, top])


def make_phenotyping(crouzon_image: Path, williams_dir: Path, model_path: Path | None, tmp: Path) -> None:
    """Three identically framed panels: landmarks, measurements, average face."""
    rgb, lm = detect_landmarks(crouzon_image, model_path)
    crop, lm_c = square_crop(rgb, lm)
    avg_png = run_average_face(williams_dir, tmp)
    avg_rgb, avg_lm = detect_landmarks(avg_png, model_path)
    avg_crop, _ = square_crop(avg_rgb, avg_lm)

    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.6), dpi=DPI)
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)

    axes[0].imshow(crop)
    axes[0].scatter(lm_c[:, 0], lm_c[:, 1], s=2.2, color="#1f77b4", alpha=0.9, linewidths=0)
    axes[0].set_title("extract-landmarks: 478 points", fontsize=8.5)

    axes[1].imshow(crop)
    bz = np.linalg.norm(lm[BIZYG[0]] - lm[BIZYG[1]])
    handles = []
    for name, a, b, color in FEATURES:
        pa, pb = lm_c[a], lm_c[b]
        axes[1].plot([pa[0], pb[0]], [pa[1], pb[1]], color=color, lw=2.0, solid_capstyle="round", zorder=4)
        axes[1].scatter([pa[0], pb[0]], [pa[1], pb[1]], s=16, color=color, edgecolors="white", linewidths=0.6, zorder=5)
        val = np.linalg.norm(lm[a] - lm[b]) / bz
        handles.append(Line2D([0], [0], color=color, lw=2.0, label=f"{name.replace('_', ' ')}  {val:.2f}"))
    axes[1].legend(handles=handles, loc="lower center", fontsize=6.2, framealpha=0.92,
                   handlelength=1.2, borderpad=0.4, labelspacing=0.3, edgecolor="#999999", ncol=1)
    axes[1].set_title("extract-features: 4 of 120 measurements", fontsize=8.5)

    axes[2].imshow(avg_crop)
    axes[2].set_title("average-face: Williams syndrome (n=257)", fontsize=8.5)

    fig.tight_layout(pad=0.4)
    fig.savefig(FIGURES / "phenotyping.png", bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def make_enhance(crouzon_image: Path, tmp: Path) -> None:
    """Before/after panel for ``facekit enhance``: a grayscale copy of the face
    is colorized, an 80-px copy is restored and upscaled 2x."""
    import cv2

    src = cv2.imread(str(crouzon_image))
    inp = tmp / "enh_in"
    inp.mkdir()
    gray = cv2.cvtColor(cv2.cvtColor(src, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    cv2.imwrite(str(inp / "grayscale.png"), gray)
    h, w = src.shape[:2]
    s = 80 / max(h, w)
    small = cv2.resize(src, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(inp / "low_resolution.png"), small)
    out = tmp / "enh_out"
    run(["facekit", "enhance", "-i", str(inp), "-o", str(out), "--device", "cpu"])

    panels = [
        (gray, "input: grayscale"),
        (cv2.imread(str(out / "grayscale.png")), "DDColor colorized"),
        (cv2.resize(small, (small.shape[1] * 2, small.shape[0] * 2), interpolation=cv2.INTER_NEAREST),
         f"input: {small.shape[1]} px face (shown 2x)"),
        (cv2.imread(str(out / "low_resolution.png")), "GFPGAN restored (2x)"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(5.2, 5.6), dpi=DPI)
    for ax, (img, title) in zip(axes.ravel(), panels):
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title(title, fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
    fig.tight_layout(pad=0.4)
    fig.savefig(FIGURES / "enhance.png", bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


MANUSCRIPT_PANELS = {
    "manuscript/image/posecorr_geom-1.pdf": "pose-correction.png",
}


def make_manuscript_panels(dpi: int = 220) -> None:
    """Rasterize manuscript PDF panels that the README reuses (pymupdf)."""
    import pymupdf

    for src, dst in MANUSCRIPT_PANELS.items():
        page = pymupdf.open(ROOT / src)[0]
        page.get_pixmap(dpi=dpi, alpha=False).save(FIGURES / dst)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--crouzon-image", type=Path, default=DEFAULT_CROUZON)
    p.add_argument("--williams-dir", type=Path, default=DEFAULT_WILLIAMS)
    p.add_argument("--model-path", type=Path, default=None,
                   help="MediaPipe face_landmarker.task (auto-downloaded if absent).")
    p.add_argument("--panels", action="store_true",
                   help="Also rasterize the manuscript PDF panels (needs pymupdf).")
    p.add_argument("--only", default=None,
                   help="Comma-separated subset of: phenotyping, enhance, panels.")
    args = p.parse_args()
    only = set(args.only.split(",")) if args.only else {"phenotyping", "enhance"} | ({"panels"} if args.panels else set())

    FIGURES.mkdir(exist_ok=True)
    if only & {"phenotyping", "enhance"}:
        for path in (args.crouzon_image, args.williams_dir):
            if not path.exists():
                raise SystemExit(f"missing source asset: {path}")

    with tempfile.TemporaryDirectory() as tmp_str:
        tmp = Path(tmp_str)
        if "phenotyping" in only:
            make_phenotyping(args.crouzon_image, args.williams_dir, args.model_path, tmp)
        if "enhance" in only:
            make_enhance(args.crouzon_image, tmp)
    if "panels" in only:
        make_manuscript_panels()
    print("wrote figures for", sorted(only), "to", FIGURES)


if __name__ == "__main__":
    main()
