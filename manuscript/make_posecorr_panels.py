"""Build the pose-correction figure panels for the Results section.

The slide version (`slides/make_pose_arms_figure.py`) stacks both panels into one
raster image, which cannot carry LaTeX subfigure labels. This script reuses that
script's two panel functions unchanged and writes each panel as its own vector
PDF, so the manuscript can reference them as (a) and (b).

  posecorr_a_geometry.pdf   orthographic decomposition of a rotated face
  posecorr_b_roundtrip.pdf  synthetic round trip, arms A / B / C

    python manuscript/make_posecorr_panels.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "slides"))

from make_pose_arms_figure import panel_geometry, panel_roundtrip  # noqa: E402

OUT = HERE / "image"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for stem, draw, size in (
        ("posecorr_a_geometry", panel_geometry, (5.1, 3.5)),
        ("posecorr_b_roundtrip", panel_roundtrip, (5.1, 3.0)),
    ):
        fig = plt.figure(figsize=size, dpi=260)
        draw(fig.add_subplot(1, 1, 1))
        fig.tight_layout()
        for ext in ("pdf", "png"):
            fig.savefig(OUT / f"{stem}.{ext}", facecolor="white")
        plt.close(fig)
        print(f"wrote {OUT / stem}.pdf")


if __name__ == "__main__":
    main()
