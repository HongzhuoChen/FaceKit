"""The packaged feature_groups.csv must equal the (feature_group, csv_column)
projection of hpo_direction_codes.csv, so a corrected mapping cannot drift.
Skipped where the full file is absent."""
import csv
from pathlib import Path

import pytest

from facekit.core.geometric.defaults import DEFAULT_HPO_CODES

FULL = Path(__file__).resolve().parents[1] / "labels" / "hpo_direction_codes.csv"


def _pairs(path):
    with open(path, newline="") as fh:
        return {
            (r["feature_group"].strip(), r["csv_column"].strip())
            for r in csv.DictReader(fh)
            if r["feature_group"].strip() and r["csv_column"].strip()
        }


def test_packaged_table_exists():
    assert DEFAULT_HPO_CODES.exists()
    assert _pairs(DEFAULT_HPO_CODES)


@pytest.mark.skipif(not FULL.exists(), reason="hpo_direction_codes.csv not in this checkout")
def test_packaged_table_matches_full_codes():
    assert _pairs(DEFAULT_HPO_CODES) == _pairs(FULL)
