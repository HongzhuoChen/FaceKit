"""Central config for the FCRM study (hpo_predict_fcrm).

This is a SUPERSET config: it defines every attribute that the reused
`hpo_predict/common.py` and `hpo_predict/eval_protocol.py` read (so that their
`import config` resolves cleanly to THIS module), plus the FCRM-specific
hyperparameters. All filesystem paths are env-overridable; no path is baked
into logic. Defaults mirror the sibling `hpo_predict` study (shared source data
and label artifacts).
"""
import os
from pathlib import Path

# facekit/experiments/hpo_predict_fcrm/config.py -> facekit/
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(
    os.environ.get(
        "FACEKIT_HPO_DATA_ROOT",
        "/vast/projects/kai/multimodal-machine-learn/hongzhuo",
    )
)

# --- source data (read-only); shared with hpo_predict ---
PHENOTYPES_CSV = Path(
    os.environ.get("FACEKIT_PHENOTYPES_CSV", DATA_ROOT / "mm_fusion_top50/combined_data/phenotypes.csv")
)
DIRECTION_CODES_CSV = Path(
    os.environ.get("FACEKIT_DIRECTION_CODES_CSV", DATA_ROOT / "mm_fusion_top50/hpo_direction_codes.csv")
)
HPOA_PATH = Path(os.environ.get("FACEKIT_HPOA", DATA_ROOT / "mm_fusion_top10/phenotype.hpoa"))

# --- materialized label artifacts (hpo_predict step 1-2 outputs, committed) ---
LABELS_DIR = Path(os.environ.get("FACEKIT_LABELS_DIR", REPO_ROOT / "labels"))
GOLD_CSV = LABELS_DIR / "gold_hpo_facial.csv"
DISEASE_OMIM_CSV = LABELS_DIR / "disease_omim_map.csv"
VOCAB_CSV = LABELS_DIR / "facial_hpo_vocab.csv"
DISEASE_FREQ_CSV = LABELS_DIR / "disease_hpo_freq.csv"

# --- run outputs (timestamped) ---
RESULTS_ROOT = Path(os.environ.get("FACEKIT_RESULTS_ROOT", REPO_ROOT / "results"))
EXP_NAME = "hpo_predict_fcrm"

# --- reproducibility ---
SEED = int(os.environ.get("FACEKIT_SEED", 20260623))

# phenotypes columns that are NOT geometric features (read by common.feature_columns)
NON_FEATURE_COLS = ("disease", "image_id", "pose_yaw", "pose_pitch", "pose_roll", "frontal_ok")

# arbitration thresholds read by eval_protocol.evaluate (shared, unchanged protocol)
F_HIGH = 0.5
F_LOW = 0.1
# soft-label default for an annotated (disease, hpo) pair with blank hpoa frequency
DEFAULT_FREQ = 0.5

# =====================================================================================
# FCRM experiment hyperparameters
# =====================================================================================
N_TEST = 100  # gold-positive patients held out as the test set (stratified by disease)

# --- Step 0 feature filter ---
# Drop direction codes that are (a) expected_direction == 0 (no signed proxy),
# (b) confidence == 'LOW', or (c) in DROP_FEATURE_GROUPS below.
#
# !!! ASSUMPTION (flagged in QUESTIONS.md) !!!
# PLAN step 0 says to drop "the 6 listed groups" but never enumerates them; it only
# pins the target count 100 -> 83. After dropping dir==0 / LOW (2 unique rows) 98
# remain, so the 6 groups must remove exactly 15 terms. The set below is a documented,
# rationale-based default (transient / asymmetry / pose-confounded / presence-binary
# families that are unreliable from a single mean-pooled frontal image) that yields
# exactly 83. CONFIRM the intended 6 groups; correcting is a one-line edit here.
DROP_FEATURE_GROUPS = (
    "EYE_GAZE_ASYMMETRY",  # 4 - gaze direction is transient, not anatomy
    "LIP_CLEFT",           # 5 - presence/binary; documented Phase-1 artifact source (pat9656)
    "FACE_ASPECT_RATIO",   # 3 - strongly pitch/pose sensitive
    "FACE_ASYMMETRY",      # 1 - asymmetry pose-confounded
    "EYE_ASYMMETRY",       # 1 - asymmetry pose-confounded
    "HEAD_FRONTAL_TAPER",  # 1 - global head-shape, pose sensitive
)

# --- Step 2 disease-prior threshold sweep ---
F_THRESH_GRID = (0.1, 0.2, 0.3, 0.4, 0.5)

# --- Step 3 per-HPO threshold tuning ---
TAU_GRID = (0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0)
FALLBACK_TAU = 1.0
MIN_TUNE_POS = 5  # < this many tune positives -> fall back to FALLBACK_TAU

# --- Step 4 FCRM beta / f_min sweep ---
BETA_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
FMIN_GRID = (0.05, 0.1, 0.2)
