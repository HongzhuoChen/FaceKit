"""Figure: what the canonical-depth correction moves on a real photograph.

The paper version of ``slides/make_pose_case_figures.py``'s ``arm_c.png``. That
one is drawn for a slide and carries its explanation inside the panel; here the
explanation lives in the LaTeX caption, so the only text left in the figure is
the colour-bar, which has nowhere else to go. The numbers the caption quotes
are printed on each run rather than copied.

    python manuscript/make_posecorr_mechanism.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "slides"))
sys.path.insert(0, str(REPO / "src"))

from make_pose_case_figures import (                                 # noqa: E402
    ACCENT, IPD, SIDE, WARN, crop_to_face, detect, similarity_fit,
)
from facekit.core.geometric.canonical_depth import CANONICAL_DEPTH   # noqa: E402
from facekit.core.geometric.extractor import (                       # noqa: E402
    GeometricFeatureExtractor, L_ZYGO, R_ZYGO,
)

OUT = REPO / "manuscript" / "image"
MODEL = REPO.parent / "facekit_project" / "face_landmarker.task"


def main():
    lm, M, im, pose = detect(SIDE)
    ex = GeometricFeatureExtractor(model_path=MODEL)
    lm_c, sc_c, derot = ex._canonicalize(lm, M)
    assert derot
    lm_a, sc_a, _ = ex._canonicalize(lm, None)
    f_a, f_c = ex._compute_all(lm_a, sc_a), ex._compute_all(lm_c, sc_c)

    # Every feature is a ratio, so a global similarity changes nothing; what is
    # left after the best-fit similarity is the part that moves a measurement.
    disp = similarity_fit(lm_c, lm) - lm
    biz = np.linalg.norm(lm[R_ZYGO] - lm[L_ZYGO])
    mag_pct = 100 * np.linalg.norm(disp, axis=1) / biz

    fig, axes = plt.subplots(1, 2, figsize=(8.8, 4.15), dpi=300)
    fig.subplots_adjust(left=0.02, right=0.845, top=0.98, bottom=0.02,
                        wspace=0.04)

    # The table is negative toward the camera, so plot its negation: red is
    # then the part of the face that sticks out.
    protrusion = -CANONICAL_DEPTH
    lim = float(np.quantile(np.abs(protrusion), 0.96))

    ax = axes[0]
    ax.imshow(im)
    crop_to_face(ax, lm, im.shape)
    scat = ax.scatter(lm[:, 0], lm[:, 1], c=protrusion, cmap="coolwarm",
                      s=6.5, vmin=-lim, vmax=lim, zorder=4, linewidths=0)

    cax = fig.add_axes([0.858, 0.20, 0.016, 0.60])
    cb = fig.colorbar(scat, cax=cax)
    cb.set_label("protrusion toward the camera, $-Z$\n(bizygomatic units)",
                 fontsize=9, labelpad=10, linespacing=1.4)
    cb.ax.tick_params(labelsize=8)

    ax = axes[1]
    ax.imshow(im, alpha=0.5)
    crop_to_face(ax, lm, im.shape)
    mag, step = 2.0, 5
    idx = np.arange(0, len(lm), step)
    for i in idx:
        d = disp[i] * mag
        if np.hypot(*d) < 1.5:
            continue
        ax.add_patch(FancyArrowPatch(tuple(lm[i]), tuple(lm[i] + d),
                                     arrowstyle="-|>", mutation_scale=6.5,
                                     lw=1.0, color=WARN, zorder=5))
    ax.scatter(lm[idx, 0], lm[idx, 1], s=3.5, color=ACCENT, zorder=4,
               linewidths=0)

    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"F_posecorr_mechanism.{ext}", facecolor="white")

    # Summarise over the measurements whose baseline is large enough for a
    # percentage to mean anything; the asymmetry columns sit near zero and
    # blow up otherwise.
    changed = {k: 100 * (f_c[k] - f_a[k]) / abs(f_a[k])
               for k in f_a
               if np.isfinite(f_a[k]) and np.isfinite(f_c[k])
               and abs(f_a[k]) > 0.05}
    d_ipd = 100 * (f_c[IPD] - f_a[IPD]) / abs(f_a[IPD])
    print(f"WROTE: {OUT/'F_posecorr_mechanism.pdf'} (+ .png)")
    print(f"caption numbers: yaw {pose[0]:+.1f} deg, arrows x{mag:g}, "
          f"median displacement {np.median(mag_pct):.1f}%, "
          f"95th percentile {np.percentile(mag_pct, 95):.1f}%, "
          f"IPD/bizygomatic {d_ipd:+.2f}%, "
          f"face_asymmetry {changed['face_asymmetry']:+.0f}%, "
          f"eye_fissure_slant_mean {changed['eye_fissure_slant_mean']:+.0f}%, "
          f"median over measurements {np.median(np.abs(list(changed.values()))):.1f}%")


if __name__ == "__main__":
    main()
