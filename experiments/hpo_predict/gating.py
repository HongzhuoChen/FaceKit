"""Phase 1.5 step 8: per-image input-quality gating + NaN-aware pooling.

Gating runs PER IMAGE, BEFORE per-patient mean-pooling: a 6-8 sigma artifact only
dilutes (does not vanish) under pooling, so the bad value must be removed at the image
level. Two gates (thresholds in config, all adjustable):

  * pose gate (two-tier, per axis, degrees):
      - extreme (|ang| > POSE_EXCLUDE_DEG) on ANY axis -> DROP the whole image.
      - moderate yaw  (|yaw|  > POSE_MASK_DEG["yaw"])   -> NaN-mask the YAW family.
      - moderate pitch(|pitch|> POSE_MASK_DEG["pitch"]) -> NaN-mask the PITCH family.
      - roll: corrected in canonicalization -> masks no family (extreme roll still drops).
  * mouth gate: mouth_opening > MOUTH_OPEN_THRESH -> NaN-mask the LIP_*/MOUTH_* family.

A patient whose images are ALL dropped abstains entirely (excluded from features and
from eval -> counted as a coverage cost). NaN-masked (not dropped) images still pool
via NaN-aware groupby.mean (skipna).

The directional pose->family map is defined EXPLICITLY in config.py.
"""
import numpy as np
import pandas as pd

import common
import config


def feature_families(feature_cols: list[str]) -> dict[str, list[str]]:
    """Resolve the config family rules against the actual feature columns.

    Returns {"yaw": [...], "pitch": [...], "mouth": [...]} (sorted). Asserts every
    explicitly-listed column exists so a typo cannot silently drop a column from a gate.
    """
    cols = set(feature_cols)

    def _check(names):
        missing = [c for c in names if c not in cols]
        assert not missing, f"family columns absent from feature set: {missing}"
        return list(names)

    yaw = set()
    yaw.update(c for c in feature_cols if config.YAW_ASYM_KEYWORD in c)
    yaw.update(c for c in feature_cols if c.endswith(config.YAW_LR_SUFFIXES))
    yaw.update(_check(config.YAW_WIDTH_COLS))
    pitch = set(_check(config.PITCH_VERTICAL_COLS))
    mouth = set(_check(config.MOUTH_FAMILY_COLS))
    return {"yaw": sorted(yaw), "pitch": sorted(pitch), "mouth": sorted(mouth)}


def gate_images(pheno: pd.DataFrame, families: dict[str, list[str]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply per-image gates. Returns (gated_pheno, manifest).

    gated_pheno: pheno with dropped images removed and masked feature cells set to NaN.
    manifest: one row per GATED image (dropped or masked) with the triggering gate(s).
    """
    feats = common.feature_columns(pheno)
    df = pheno.copy()
    yaw_c, pitch_c, mouth_c = families["yaw"], families["pitch"], families["mouth"]

    yaw_a = df["pose_yaw"].abs()
    pitch_a = df["pose_pitch"].abs()
    roll_a = df["pose_roll"].abs()
    extreme = (
        (yaw_a > config.POSE_EXCLUDE_DEG["yaw"])
        | (pitch_a > config.POSE_EXCLUDE_DEG["pitch"])
        | (roll_a > config.POSE_EXCLUDE_DEG["roll"])
    )
    mask_yaw = ~extreme & (yaw_a > config.POSE_MASK_DEG["yaw"])
    mask_pitch = ~extreme & (pitch_a > config.POSE_MASK_DEG["pitch"])
    mask_mouth = ~extreme & (df["mouth_opening"] > config.MOUTH_OPEN_THRESH)

    manifest = []
    for i in df.index:
        # capture pose/mouth BEFORE any masking (mouth_opening is itself a mouth-family col)
        rec = (df.at[i, "image_id"], int(df.at[i, "patient_id"]),
               round(float(df.at[i, "pose_yaw"]), 2), round(float(df.at[i, "pose_pitch"]), 2),
               round(float(df.at[i, "pose_roll"]), 2), round(float(df.at[i, "mouth_opening"]), 3))
        if extreme[i]:
            manifest.append((rec[0], rec[1], "dropped", "pose_extreme", "", *rec[2:]))
            continue
        gates, fams = [], set()
        if mask_yaw[i]:
            gates.append("pose_yaw"); fams.update(yaw_c)
        if mask_pitch[i]:
            gates.append("pose_pitch"); fams.update(pitch_c)
        if mask_mouth[i]:
            gates.append("mouth_open"); fams.update(mouth_c)
        if gates:
            cols = sorted(fams)
            df.loc[i, cols] = np.nan
            manifest.append((rec[0], rec[1], "masked", "+".join(gates), f"{len(cols)} cols", *rec[2:]))

    gated = df[~extreme.to_numpy()].copy()
    manifest_df = pd.DataFrame(
        manifest,
        columns=["image_id", "patient_id", "action", "gates", "n_cols_masked",
                 "pose_yaw", "pose_pitch", "pose_roll", "mouth_opening"],
    )
    return gated, manifest_df


def pool_gated(gated_pheno: pd.DataFrame, gold: pd.DataFrame) -> pd.DataFrame:
    """NaN-aware per-patient mean-pool of the gated images (skipna), union gold.

    Mirrors common.pool_patients but tolerates NaN-masked cells; a patient with no
    surviving images simply does not appear in the output (full abstention).
    """
    feats = common.feature_columns(gated_pheno)
    agg_feats = gated_pheno.groupby("patient_id")[feats].mean()  # skipna=True by default
    disease = gated_pheno.groupby("patient_id")["disease"].first()

    g = gold.copy()
    g["patient_id"] = g["image_id"].map(common._parse_patient_id)

    def _union(series):
        s: set[str] = set()
        for lst in series:
            s.update(lst)
        return s

    present = g.groupby("patient_id")["present_facial_hpo"].apply(_union)
    absent = g.groupby("patient_id")["absent_facial_hpo"].apply(_union)
    out = agg_feats.join(disease).join(present).join(absent)
    out["present_facial_hpo"] = out["present_facial_hpo"].apply(lambda v: v if isinstance(v, set) else set())
    out["absent_facial_hpo"] = out["absent_facial_hpo"].apply(lambda v: v if isinstance(v, set) else set())
    return out.reset_index()


def load_gated_patient_table(run_dir):
    """Gated counterpart of common.load_patient_table.

    Returns (table, feats, families, manifest, coverage_stats). `table` is the gated
    per-patient frame (NaN where every contributing image masked that feature) joined
    to the Phase-1 split.
    """
    pheno = common.load_phenotypes()
    feats = common.feature_columns(pheno)
    families = feature_families(feats)
    gated, manifest = gate_images(pheno, families)
    pooled = pool_gated(gated, gold=common.load_gold())
    pooled = pooled.rename(columns={"present_facial_hpo": "present", "absent_facial_hpo": "absent"})
    pooled["omim"] = pooled["disease"].map(common.disease_omim_map())
    split = pd.read_csv(run_dir / "split.csv")[["patient_id", "split"]]
    table = pooled.merge(split, on="patient_id", how="inner")

    # coverage / NaN bookkeeping
    n_patients_all = pheno["patient_id"].nunique()
    n_patients_gated = int(table["patient_id"].nunique())
    nan_rate = float(table[feats].isna().to_numpy().mean())
    coverage = {
        "n_images_total": int(len(pheno)),
        "n_images_dropped": int((manifest["action"] == "dropped").sum()),
        "n_images_masked": int((manifest["action"] == "masked").sum()),
        "n_patients_ungated": int(n_patients_all),
        "n_patients_gated": n_patients_gated,
        "n_patients_dropped": int(n_patients_all - n_patients_gated),
        "per_patient_feature_nan_rate": round(nan_rate, 4),
    }
    return table, feats, families, manifest, coverage


def run(run_dir) -> dict:
    """Step 8 verify: emit families + manifest; confirm the two named artifact images."""
    table, feats, families, manifest, coverage = load_gated_patient_table(run_dir)

    fam_rows = [(fam, c) for fam, cols in families.items() for c in cols]
    pd.DataFrame(fam_rows, columns=["family", "csv_column"]).to_csv(
        run_dir / "feature_families.csv", index=False)
    manifest.to_csv(run_dir / "gated_image_manifest.csv", index=False)

    art = manifest[manifest["patient_id"].isin([c["patient_id"] for c in config.ARTIFACT_CASES])]
    print("[step8] family sizes:", {k: len(v) for k, v in families.items()})
    print("[step8] coverage:", coverage)
    print("[step8] artifact images in manifest:")
    print(art.to_string(index=False) if len(art) else "  (none — pose-near-frontal; handled by z-clip)")
    return {"families": {k: len(v) for k, v in families.items()}, "coverage": coverage,
            "n_manifest": int(len(manifest))}


if __name__ == "__main__":
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from pathlib import Path
    run(Path("/vast/projects/kai/multimodal-machine-learn/hongzhuo/facekit/results/hpo_predict_phase1/2026-06-13_11-01-04"))
