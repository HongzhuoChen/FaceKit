"""Central config for the Phase-1 per-patient facial-HPO prediction study.

All filesystem paths are env-overridable (no hardcoded paths baked into logic).
Defaults point at the sibling data trees under the project root.
"""
import os
from pathlib import Path

# facekit/experiments/hpo_predict/config.py -> facekit/
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(
    os.environ.get(
        "FACEKIT_HPO_DATA_ROOT",
        "/vast/projects/kai/multimodal-machine-learn/hongzhuo",
    )
)

# --- source data (read-only) ---
PHENOTYPES_CSV = Path(
    os.environ.get("FACEKIT_PHENOTYPES_CSV", DATA_ROOT / "mm_fusion_top50/combined_data/phenotypes.csv")
)
DIRECTION_CODES_CSV = Path(
    os.environ.get("FACEKIT_DIRECTION_CODES_CSV", DATA_ROOT / "mm_fusion_top50/hpo_direction_codes.csv")
)
HPOA_PATH = Path(os.environ.get("FACEKIT_HPOA", DATA_ROOT / "mm_fusion_top10/phenotype.hpoa"))
GOLD_HF_DATASET = os.environ.get("FACEKIT_GOLD_HF", "HzChen20/GMDB_enhanced_new_top100_filtered")

# --- materialized label artifacts (step 1-2 outputs, committed) ---
LABELS_DIR = REPO_ROOT / "labels"
GOLD_CSV = LABELS_DIR / "gold_hpo_facial.csv"
DISEASE_OMIM_CSV = LABELS_DIR / "disease_omim_map.csv"
VOCAB_CSV = LABELS_DIR / "facial_hpo_vocab.csv"
DISEASE_FREQ_CSV = LABELS_DIR / "disease_hpo_freq.csv"

# --- run outputs (seeded / timestamped) ---
RESULTS_ROOT = REPO_ROOT / "results"
EXP_NAME = "hpo_predict_phase1"
EXP_NAME_P15 = "hpo_predict_phase1p5"  # Phase 1.5 run dir (does not overwrite Phase 1)

# --- reproducibility / hyperparameters ---
SEED = 20260609
SPLIT_RATIOS = (0.70, 0.15, 0.15)  # train / val / test (by patient, within disease)

# arbitration thresholds (OPEN-3 RESOLVED)
F_HIGH = 0.5
F_LOW = 0.1
# rule-baseline z threshold swept on val (OPEN-3 RESOLVED)
TAU_GRID = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
# learned-model prob->pred-positive threshold, swept on val by trusted-subset F1
# (same criterion the rule baseline uses for tau); pos_weight inflates probs, so 0.5
# over-fires -- the threshold must be tuned, not hardcoded.
PRED_POS_THRESHOLD_GRID = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
# soft-label default for an annotated (disease, hpo) pair whose hpoa frequency is blank
DEFAULT_FREQ = 0.5

# phenotypes columns that are NOT geometric features
NON_FEATURE_COLS = ("disease", "image_id", "pose_yaw", "pose_pitch", "pose_roll", "frontal_ok")

# =====================================================================================
# Phase 1.5 — input-quality gating + precision improvement (all values adjustable here)
# =====================================================================================
# Gating runs PER IMAGE, before per-patient mean-pooling (6-8 sigma artifacts survive
# pooling, so they must be removed at the image level). `frontal_ok` is True for all
# 4559 rows -> useless; the pose gates use the angle columns (degrees) instead.
POSE_MASK_DEG = {"yaw": 15.0, "pitch": 15.0, "roll": 12.0}     # moderate -> NaN-mask families
POSE_EXCLUDE_DEG = {"yaw": 25.0, "pitch": 25.0, "roll": 20.0}  # extreme  -> drop whole image
MOUTH_OPEN_THRESH = 0.30  # mouth_opening above this -> NaN-mask LIP_*/MOUTH_* families on that image
MEDIUM_CONF_WEIGHT = 0.5  # MEDIUM-confidence direction codes: multiply score by this (1.5-C)
Z_CLIP = 5.0              # clip |directional z| (rule score / learned standardized input) (1.5-D)

# -------------------------------------------------------------------------------------
# Directional pose -> feature-family map (1.5-A). EXPLICIT and documented; the reviewer
# checks this. A column may sit in MORE THAN ONE family and is NaN-masked if ANY of its
# families' gate fires. `roll` is already corrected in canonicalization -> it masks no
# family (extreme roll still drops the image via POSE_EXCLUDE_DEG).
#
# YAW-sensitive (masked when |yaw| > POSE_MASK_DEG["yaw"]): lateral asymmetry + single-
# side L/R measures + frontal-plane WIDTH measures (all distorted by out-of-plane yaw).
#   - asymmetry  : any column whose name contains this keyword (covers *_asym, gaze_asym_*,
#                  face_asymmetry).
YAW_ASYM_KEYWORD = "asym"
#   - L/R paired : any column ending in one of these suffixes (single-side measures).
YAW_LR_SUFFIXES = ("_l", "_r")
#   - width      : horizontal distances / width-ratios measured in the frontal plane.
YAW_WIDTH_COLS = (
    "eb_inter_distance", "inter_canthal_distance", "inter_pupillary_distance",
    "outer_canthal_distance", "canthal_to_pupillary_ratio", "nose_bridge_width",
    "nose_tip_width", "nose_base_width", "philtrum_width", "mouth_width", "chin_width",
    "jaw_bigonial_width", "forehead_width_upper", "forehead_width_mid",
    "face_width_uniformity", "jaw_to_forehead_ratio", "midface_width", "face_aspect_ratio",
)
# PITCH-sensitive (masked when |pitch| > POSE_MASK_DEG["pitch"]): vertical extents and
# length / vertical-ratio measures (foreshortened by chin-up / chin-down pitch).
PITCH_VERTICAL_COLS = (
    "philtrum_length", "nose_length", "nose_bridge_length", "nose_tip_elevation",
    "nose_tip_to_bridge_ratio", "nasolabial_angle", "columella_length",
    "columella_hang_below_ala", "nose_ala_height_mean", "eye_fissure_height_mean",
    "eye_fissure_aspect_mean", "eye_upper_lid_to_iris_mean", "eye_lower_lid_to_iris_mean",
    "chin_height", "forehead_height", "hairline_height", "forehead_taper_ratio",
    "face_aspect_ratio", "mouth_opening", "upper_vermilion_height",
    "lower_vermilion_height", "vermilion_total",
)
# MOUTH/expression family (masked when mouth_opening > MOUTH_OPEN_THRESH): lip- and
# mouth-derived measures (the LIP_* / MOUTH_* direction-code groups + their relatives).
MOUTH_FAMILY_COLS = (
    "mouth_width", "mouth_opening", "upper_vermilion_height", "lower_vermilion_height",
    "vermilion_total", "cupid_bow_drop", "mouth_corner_drop", "mouth_triangularity",
    "mouth_tenting", "mouth_upper_curvature_c2", "mouth_upper_fit_residual",
    "upper_lip_eversion", "lower_lip_eversion", "lip_outer_to_inner_area",
    "lip_midline_x_std", "cupid_bow_peak_asym",
)

# the two Phase-1 top-FP artifact cases to re-check under gating (PLAN-named)
ARTIFACT_CASES = (
    {"patient_id": 9656, "hpo_id": "HP:0000202", "hpo_name": "Orofacial cleft", "gate": "mouth_open"},
    {"patient_id": 6514, "hpo_id": "HP:0000343", "hpo_name": "Long philtrum", "gate": "z_clip"},
)
