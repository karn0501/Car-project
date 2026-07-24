"""
Unit tests for Phase 6: Cross-Source Entity Resolution & RapidFuzz Deduplication Engine.
"""

import os
import sys
import pytest
import pandas as pd

# Add project root to python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.deduplication import CrossSourceDeduplicator


def test_normalize_title():
    dedup = CrossSourceDeduplicator()
    assert dedup.normalize_title("Maruti-Suzuki Swift VXi 2021!!") == "maruti suzuki swift vxi 2021"


def test_find_and_remove_duplicates():
    dedup = CrossSourceDeduplicator(similarity_threshold=80.0)

    df_test = pd.DataFrame([
        {
            "company_name": "Maruti",
            "model_name": "Swift",
            "variant_name": "VXi",
            "manufacture_year": 2021,
            "km_driven": 45000,
            "asking_price": 550000,
            "city": "Mumbai"
        },
        {
            "company_name": "Maruti Suzuki",
            "model_name": "Swift",
            "variant_name": "VXi Manual",
            "manufacture_year": 2021,
            "km_driven": 45300,
            "asking_price": 540000,
            "city": "Mumbai"
        },
        {
            "company_name": "Hyundai",
            "model_name": "Creta",
            "variant_name": "SX",
            "manufacture_year": 2022,
            "km_driven": 20000,
            "asking_price": 1250000,
            "city": "Delhi"
        }
    ])

    df_res = dedup.find_duplicates(df_test)
    assert df_res.loc[1, "is_duplicate"] == True
    assert df_res.loc[0, "is_duplicate"] == False

    df_clean = dedup.deduplicate_dataframe(df_test)
    assert len(df_clean) == 2
    assert "Creta" in df_clean["model_name"].values
