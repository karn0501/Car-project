"""
Pipeline runner script for Phase 1 Data Pipeline Foundation.
Populates initial validated used car listings into database tables,
and logs scraper metadata in scraper_logs.
"""

import os
import sys

# Add project root to python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from db.database import init_db, SessionLocal
from db.models import Company, Model, Variant, Listing, ScraperLog
from scrapers.car_scraper.items import CarListingItem
from scrapers.car_scraper.pipelines import DatabasePipeline


def run_sample_ingestion():
    """Ingests a foundational set of validated listings into the relational database."""
    print("Initializing database schema...")
    init_db()

    pipeline = DatabasePipeline()
    pipeline.open_spider(type("MockSpider", (), {"name": "cardekho"})())

    sample_raw_listings = [
        {
            "company_name": "Maruti Suzuki",
            "model_name": "Swift",
            "variant_name": "VXI",
            "source_platform": "CarDekho",
            "source_url": "https://www.cardekho.com/used-car-details/maruti-swift-vxi-delhi-1.htm",
            "manufacture_year": 2019,
            "km_driven": 42000.0,
            "owner_count": 1,
            "city": "Delhi",
            "asking_price": 540000.0,
            "fuel_type": "Petrol",
            "transmission": "Manual",
            "insurance_valid": True,
            "accident_history": False,
            "description": "Well maintained single owner Maruti Swift VXI in Delhi.",
        },
        {
            "company_name": "Hyundai",
            "model_name": "Creta",
            "variant_name": "SX Petrol",
            "source_platform": "CarDekho",
            "source_url": "https://www.cardekho.com/used-car-details/hyundai-creta-sx-mumbai-2.htm",
            "manufacture_year": 2021,
            "km_driven": 28000.0,
            "owner_count": 1,
            "city": "Mumbai",
            "asking_price": 1180000.0,
            "fuel_type": "Petrol",
            "transmission": "Manual",
            "insurance_valid": True,
            "accident_history": False,
            "description": "Top condition Hyundai Creta SX with full company service history.",
        },
        {
            "company_name": "Honda",
            "model_name": "City",
            "variant_name": "VX CVT",
            "source_platform": "CarDekho",
            "source_url": "https://www.cardekho.com/used-car-details/honda-city-vx-cvt-bangalore-3.htm",
            "manufacture_year": 2018,
            "km_driven": 55000.0,
            "owner_count": 2,
            "city": "Bangalore",
            "asking_price": 790000.0,
            "fuel_type": "Petrol",
            "transmission": "Automatic",
            "insurance_valid": True,
            "accident_history": False,
            "description": "Automatic Honda City VX in Bangalore, non-accidental.",
        },
        {
            "company_name": "Tata",
            "model_name": "Nexon",
            "variant_name": "XZ Plus",
            "source_platform": "CarDekho",
            "source_url": "https://www.cardekho.com/used-car-details/tata-nexon-xz-plus-pune-4.htm",
            "manufacture_year": 2020,
            "km_driven": 36000.0,
            "owner_count": 1,
            "city": "Pune",
            "asking_price": 750000.0,
            "fuel_type": "Diesel",
            "transmission": "Manual",
            "insurance_valid": True,
            "accident_history": False,
            "description": "Tata Nexon XZ Plus Diesel in immaculate condition.",
        },
        {
            "company_name": "Mahindra",
            "model_name": "Thar",
            "variant_name": "LX Hard Top",
            "source_platform": "CarDekho",
            "source_url": "https://www.cardekho.com/used-car-details/mahindra-thar-lx-gurgaon-5.htm",
            "manufacture_year": 2022,
            "km_driven": 18000.0,
            "owner_count": 1,
            "city": "Gurgaon",
            "asking_price": 1350000.0,
            "fuel_type": "Diesel",
            "transmission": "Automatic",
            "insurance_valid": True,
            "accident_history": False,
            "description": "Mahindra Thar LX Diesel Automatic 4WD.",
        },
    ]

    for raw in sample_raw_listings:
        item = CarListingItem()
        for k, v in raw.items():
            item[k] = v
        try:
            pipeline.process_item(item, type("MockSpider", (), {"name": "cardekho"})())
            print(f"Processed: {raw['manufacture_year']} {raw['company_name']} {raw['model_name']} {raw['variant_name']} (INR {raw['asking_price']:,.0f})")
        except Exception as err:
            print(f"Skipped item due to validation/anomaly: {err}")

    pipeline.close_spider(type("MockSpider", (), {"name": "cardekho"})())

    # Verify rows in DB
    db = SessionLocal()
    companies = db.query(Company).all()
    models = db.query(Model).all()
    variants = db.query(Variant).all()
    listings = db.query(Listing).all()
    logs = db.query(ScraperLog).all()

    print("\n--- Phase 1 Pipeline Database Summary ---")
    print(f"Companies registered : {len(companies)}")
    print(f"Models registered    : {len(models)}")
    print(f"Variants registered  : {len(variants)}")
    print(f"Listings stored      : {len(listings)}")
    print(f"Scraper logs count   : {len(logs)}")

    db.close()


if __name__ == "__main__":
    run_sample_ingestion()
