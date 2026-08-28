"""Extract the privacy-assessment plot panels from the summary deck.

The face_audit run that produced these plots is not in this repository, so the
panels are taken from the figures embedded in `results/Privacy Analysis
Summary.pptx`. Each plot carries a matplotlib title that duplicates the
manuscript caption; it is cropped off so the panels read as subfigures. Nothing
inside the axes is altered.

The panels are written as PDF, but the content stays raster: it is the deck's
own bitmap wrapped in a PDF page at its native resolution. Re-export as true
vector art from the face_audit outputs (kde_density_*.png, below_threshold_*.png)
once the raw result files are available.

    python manuscript/make_privacy_panels.py
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
DECK = REPO / "results" / "Privacy Analysis Summary.pptx"

# deck media file -> (manuscript panel name, how much of the header to remove)
#   "all"      drop every text row above the axes
#   "suptitle" drop only the topmost text block, keeping the subplot titles
PANELS = {
    "image11": ("privacy_a_kde_arcface", "all"),   # slide 5
    "image8": ("privacy_b_kde_lpips", "all"),      # slide 11
    "image14": ("privacy_c_far_identity", "all"),  # slide 15
    "image17": ("privacy_d_far_lpips", "all"),     # slide 12
}

# panels are placed at roughly half the 6.5 in text width, so this keeps the
# effective print resolution near 300 dpi
DPI = 300


def flatten(im: Image.Image) -> Image.Image:
    """Composite RGBA onto white."""
    if im.mode != "RGBA":
        return im.convert("RGB")
    bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
    return Image.alpha_composite(bg, im).convert("RGB")


def crop_row(im: Image.Image, mode: str) -> int:
    """Row to crop the image at, so the title is gone but no glyph is clipped."""
    grey = np.asarray(im.convert("L"))
    ink = (grey < 250).any(axis=1)

    if mode == "suptitle":
        # leading margin, then the title block, then the gap below it
        top = int(np.flatnonzero(ink)[0])
        gap = np.flatnonzero(~ink[top:])
        return top + int(gap[0]) if len(gap) else 0

    # "all": back off from the topmost axes spine to the top of the white gap
    spines = np.flatnonzero((grey < 128).sum(axis=1) > 0.45 * im.width)
    if not len(spines):
        return 0
    row = int(spines[0])
    while row > 0 and not ink[row - 1]:
        row -= 1
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--deck", type=Path, default=DECK, help="privacy summary .pptx")
    ap.add_argument("--out", type=Path, default=HERE / "image", help="output directory")
    args = ap.parse_args()

    if not args.deck.exists():
        raise SystemExit(f"deck not found: {args.deck}")
    args.out.mkdir(parents=True, exist_ok=True)

    tmp = Path(tempfile.mkdtemp(prefix="privacy_panels_"))
    try:
        with zipfile.ZipFile(args.deck) as zf:
            zf.extractall(tmp)
        media = tmp / "ppt" / "media"
        for stem, (name, mode) in PANELS.items():
            im = flatten(Image.open(media / f"{stem}.png"))
            cut = crop_row(im, mode)
            out = im.crop((0, cut, im.width, im.height))
            out.save(args.out / f"{name}.pdf", resolution=DPI)
            print(f"{stem}.png -> {name}.pdf  (cut {cut} rows, {mode}) {out.size}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
