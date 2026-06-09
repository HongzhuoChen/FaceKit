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

# --- reproducibility / hyperparameters ---
SEED = 20260609
SPLIT_RATIOS = (0.70, 0.15, 0.15)  # train / val / test (by patient, within disease)

# arbitration thresholds (OPEN-3 RESOLVED)
F_HIGH = 0.5
F_LOW = 0.1
# rule-baseline z threshold swept on val (OPEN-3 RESOLVED)
TAU_GRID = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
# learned-model decision threshold for converting probs -> predicted-positive
PRED_POS_THRESHOLD = 0.5
# soft-label default for an annotated (disease, hpo) pair whose hpoa frequency is blank
DEFAULT_FREQ = 0.5

# phenotypes columns that are NOT geometric features
NON_FEATURE_COLS = ("disease", "image_id", "pose_yaw", "pose_pitch", "pose_roll", "frontal_ok")
