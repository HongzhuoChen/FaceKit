"""Extract the 120 geometric features from the RDFace corpus.

mediapipe 0.10.33's mp.Image.create_from_file returns GRAY8 for JPEG and
SRGBA for PNG; the landmarker needs SRGB, so every image errors and
run_batch silently returns 0 rows. We patch only the image loader.
"""
import sys, os, zipfile, shutil
sys.path.insert(0, "src")
import numpy as np, mediapipe as mp
from PIL import Image
from pathlib import Path

SP = Path(sys.argv[1]); RAW = Path(sys.argv[2])
work = SP/"rdface"; work.mkdir(parents=True, exist_ok=True)
if not (work/"rd_images").exists():
    with zipfile.ZipFile(RAW) as z: z.extractall(work)
    with zipfile.ZipFile(work/"RDFace/rd_images.zip") as z:
        for n in z.namelist():
            if not n.startswith("__MACOSX/"): z.extract(n, work)
    for p in work.rglob(".DS_Store"): p.unlink()

def _load_srgb(path):
    arr = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(arr))
mp.Image.create_from_file = staticmethod(_load_srgb)

from facekit.core.geometric.batch import run_batch
df = run_batch(work/"rd_images", SP/"rdface_out", mode="all", frontal_check=False)
print("ROWS:", len(df))
