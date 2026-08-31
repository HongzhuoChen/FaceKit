"""
Default file-system paths for the geometric / disease-specific pipeline.

Each can be overridden on the CLI. The defaults reflect the user's cluster
layout documented in PLAN.md; production deployments should pin them to
project-tracked copies.
"""
from pathlib import Path

# Local MONDO ontology (downloaded by ``scripts/download_mondo.sh``).
DEFAULT_MONDO_OBO = Path("~/data/ontologies/mondo.obo").expanduser()

# 49-disease mapping JSON (feature_group lists per disease).
DEFAULT_FEATURE_MAPPING = Path(
    "/vast/projects/kai/multimodal-machine-learn/hongzhuo/"
    "mm_fusion_top50/feature_disease_mapping.json"
)

# HPO term -> feature_group / csv_column codes. Project-tracked: the file encodes
# which feature is the sensor for each phenotype, and two of those mappings were
# corrected by experiments/feature_validity. An out-of-tree copy would silently
# revert them on the next run of hpo_predict/step2_build_labels.py.
DEFAULT_HPO_CODES = Path(__file__).resolve().parents[3].parent / "labels" / "hpo_direction_codes.csv"

# Normative reference distributions shipped with the package: per-feature mean,
# SD and n over the 886 FairFace control images that pass the frontal-pose gate
# (of 2,051 sampled across three ancestry groups). Built by
# ``scripts/build_reference.py``; this is the same reference the manuscript
# scores patients against.
DEFAULT_REFERENCE = Path(__file__).resolve().parents[2] / "data" / "reference_fairface.csv"
