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

# feature_group -> csv_column table shipped with the package. The toolkit reads
# only these two columns of hpo_direction_codes.csv, so this is that file's
# projection; tests/test_feature_groups.py keeps the two in sync when the full
# file is present. Any file with those two columns can be passed via --hpo-codes.
DEFAULT_HPO_CODES = Path(__file__).resolve().parents[2] / "data" / "feature_groups.csv"

# Normative reference distributions shipped with the package: per-feature mean,
# SD and n over the 886 FairFace control images that pass the frontal-pose gate
# (of 2,051 sampled across three ancestry groups). Built by
# ``scripts/build_reference.py``; this is the same reference the manuscript
# scores patients against.
DEFAULT_REFERENCE = Path(__file__).resolve().parents[2] / "data" / "reference_fairface.csv"
