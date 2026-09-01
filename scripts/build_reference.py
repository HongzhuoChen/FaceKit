"""Rebuild the packaged normative reference from a control phenotype CSV.

    python scripts/build_reference.py CONTROLS.csv

Writes ``src/facekit/data/reference_fairface.csv``. The controls must come from
the same feature definitions as the faces that will be scored against them: the
pose correction fixes the meaning of all 120 measurements at once, so a
reference and a patient table extracted under different conventions must not be
combined.

The shipped table was built from 2,051 FairFace images sampled across three
ancestry groups (white/black/asian), of which 886 pass the frontal-pose gate.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from facekit.core.geometric.batch import LEADING_COLUMNS  # noqa: E402
from facekit.core.geometric.defaults import DEFAULT_REFERENCE  # noqa: E402

df = pd.read_csv(sys.argv[1])
controls = df[df["frontal_ok"] == True]  # noqa: E712
feat = [c for c in df.columns if c not in set(LEADING_COLUMNS)]

pd.DataFrame({
    "feature": feat,
    "mean": controls[feat].mean().values,
    "sd": controls[feat].std(ddof=1).values,
    "n": controls[feat].notna().sum().values,
}).to_csv(DEFAULT_REFERENCE, index=False, float_format="%.10g")

print(f"{len(feat)} features from {len(controls)} frontal of {len(df)} "
      f"controls -> {DEFAULT_REFERENCE}")
