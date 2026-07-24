"""
Database data exporter module for used car price prediction.
Extracts normalized relational records (companies, models, variants, listings)
into a structured Pandas DataFrame for ML/DL modeling pipelines.
"""

import os
import sys
import pandas as pd
from datetime import datetime

# Add project root to python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from db.database import engine, SessionLocal
from db.models import Company, Model, Variant, Listing


def load_dataset_from_db():
    """
    Query database tables and return a clean, flat Pandas DataFrame.
    Returns:
        pd.DataFrame: Contains all structured car resale records and metadata.
    """
    db = SessionLocal()
    try:
        query = (
            db.query(
                Listing.id.label("listing_id"),
                Company.name.label("company_name"),
                Company.country.label("company_country"),
                Model.name.label("model_name"),
                Model.body_type.label("body_type"),
                Model.launch_year.label("model_launch_year"),
                Model.discontinued_year.label("model_discontinued_year"),
                Variant.name.label("variant_name"),
                Variant.fuel_type.label("fuel_type"),
                Variant.transmission.label("transmission"),
                Variant.engine_cc.label("engine_cc"),
                Variant.seating_capacity.label("seating_capacity"),
                Variant.ex_showroom_price.label("ex_showroom_price"),
                Listing.source_platform.label("source_platform"),
                Listing.manufacture_year.label("manufacture_year"),
                Listing.km_driven.label("km_driven"),
                Listing.owner_count.label("owner_count"),
                Listing.city.label("city"),
                Listing.insurance_valid.label("insurance_valid"),
                Listing.accident_history.label("accident_history"),
                Listing.asking_price.label("asking_price"),
                Listing.scraped_at.label("scraped_at")
            )
            .join(Variant, Listing.variant_id == Variant.id)
            .join(Model, Variant.model_id == Model.id)
            .join(Company, Model.company_id == Company.id)
        )

        records = [r._asdict() for r in query.all()]
        df = pd.DataFrame(records)
        return df
    finally:
        db.close()


if __name__ == "__main__":
    df_sample = load_dataset_from_db()
    print("=" * 80)
    print("DATASET EXPORT SUMMARY:")
    print(f"Total Rows Extracted : {len(df_sample)}")
    print(f"Total Features       : {len(df_sample.columns)}")
    print(f"Columns              : {list(df_sample.columns)}")
    print("=" * 80)
