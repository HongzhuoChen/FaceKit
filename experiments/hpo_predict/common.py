"""Shared data helpers: facial vocab, gold, per-patient pooling, disease->freq table.

These are read by every downstream step so that the gold/label semantics live in
one place. None of this depends on the HuggingFace `datasets` library (that is read
exactly once, in step 1, to materialize labels/gold_hpo_facial.csv).
"""
import re

import numpy as np
import pandas as pd

import config

_PAT_RE = re.compile(r"^pat(\d+)_img(\d+)$")

# HPO frequency-class terms -> representative point estimate (HPO standard buckets).
_FREQ_TERM = {
    "HP:0040280": 1.00,   # Obligate
    "HP:0040281": 0.90,   # Very frequent (80-99%)
    "HP:0040282": 0.55,   # Frequent (30-79%)
    "HP:0040283": 0.17,   # Occasional (5-29%)
    "HP:0040284": 0.025,  # Very rare (1-4%)
    "HP:0040285": 0.00,   # Excluded
}


def load_direction_codes() -> pd.DataFrame:
    """The 100 facial direction-code HPO: rule-baseline KB and canonical label space."""
    return pd.read_csv(config.DIRECTION_CODES_CSV)


def facial_vocab() -> list[str]:
    """Sorted list of the 100 facial HPO ids (the canonical, fixed label space)."""
    return sorted(load_direction_codes()["hpo_id"].unique().tolist())


def feature_columns(phenotypes: pd.DataFrame) -> list[str]:
    """The 125 geometric feature columns (everything that is not metadata/pose)."""
    return [c for c in phenotypes.columns if c not in config.NON_FEATURE_COLS]


def _parse_patient_id(image_id: str) -> int:
    m = _PAT_RE.match(str(image_id))
    if m is None:
        raise ValueError(f"image_id does not match pat<NN>_img<MM>: {image_id!r}")
    return int(m.group(1))


def load_phenotypes() -> pd.DataFrame:
    """Per-image geometric features + disease + parsed patient_id."""
    df = pd.read_csv(config.PHENOTYPES_CSV)
    df["patient_id"] = df["image_id"].map(_parse_patient_id)
    return df


def parse_hpo_list(cell) -> list[str]:
    if pd.isna(cell) or cell == "":
        return []
    return [t for t in str(cell).split(";") if t]


def load_gold() -> pd.DataFrame:
    """Static gold table (step-1 output): per-image facial present/absent HPO."""
    g = pd.read_csv(config.GOLD_CSV, dtype={"image_id": str})
    g["present_facial_hpo"] = g["present_facial_hpo"].map(parse_hpo_list)
    g["absent_facial_hpo"] = g["absent_facial_hpo"].map(parse_hpo_list)
    return g


def pool_patients(phenotypes: pd.DataFrame, gold: pd.DataFrame) -> pd.DataFrame:
    """Mean-pool features per patient (OPEN-5); union gold present/absent across images.

    Returns one row per patient with the 125 mean-pooled feature columns plus
    `disease`, `present_facial_hpo` (set), `absent_facial_hpo` (set).
    """
    feats = feature_columns(phenotypes)
    agg_feats = phenotypes.groupby("patient_id")[feats].mean()
    disease = phenotypes.groupby("patient_id")["disease"].first()

    gold = gold.copy()
    gold["patient_id"] = gold["image_id"].map(_parse_patient_id)

    def _union(series):
        s: set[str] = set()
        for lst in series:
            s.update(lst)
        return s

    present = gold.groupby("patient_id")["present_facial_hpo"].apply(_union)
    absent = gold.groupby("patient_id")["absent_facial_hpo"].apply(_union)

    out = agg_feats.join(disease).join(present).join(absent)
    out["present_facial_hpo"] = out["present_facial_hpo"].apply(lambda v: v if isinstance(v, set) else set())
    out["absent_facial_hpo"] = out["absent_facial_hpo"].apply(lambda v: v if isinstance(v, set) else set())
    return out.reset_index()


def parse_frequency(value) -> float:
    """hpoa frequency -> float in [0,1]. Handles k/n, percentages, HPO terms, blank."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return config.DEFAULT_FREQ
    s = str(value).strip()
    if s == "":
        return config.DEFAULT_FREQ
    if s in _FREQ_TERM:
        return _FREQ_TERM[s]
    if s.endswith("%"):
        return float(s[:-1]) / 100.0
    if "/" in s:
        k, n = s.split("/")
        n = float(n)
        return float(k) / n if n > 0 else config.DEFAULT_FREQ
    try:
        return float(s)
    except ValueError:
        return config.DEFAULT_FREQ


def disease_freq_table() -> dict[str, dict[str, float]]:
    """{OMIM_id: {facial_hpo: freq_value}} for the diseases in disease_omim_map.csv.

    Reads the materialized labels/disease_hpo_freq.csv produced in step 2.
    """
    df = pd.read_csv(config.DISEASE_FREQ_CSV)
    table: dict[str, dict[str, float]] = {}
    for omim, sub in df.groupby("omim"):
        table[omim] = dict(zip(sub["hpo_id"], sub["freq_value"]))
    return table


def disease_omim_map() -> dict[str, str]:
    """{disease_name: OMIM_id} (step-1 output)."""
    df = pd.read_csv(config.DISEASE_OMIM_CSV)
    return dict(zip(df["disease"], df["omim"]))
