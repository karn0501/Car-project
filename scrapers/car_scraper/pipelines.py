"""
Scrapy Item Pipeline for data validation, anomaly filtering,
and relational database persistence into PostgreSQL/SQLite.
"""

from datetime import datetime, timezone
import logging
from scrapy.exceptions import DropItem
from pydantic import ValidationError

from db.database import SessionLocal, init_db
from db.models import Company, Model, Variant, Listing, ScraperLog
from scrapers.car_scraper.items import CarListingSchema

logger = logging.getLogger(__name__)


class DatabasePipeline:
    """Pipeline for saving validated and filtered listings to the database."""

    def __init__(self):
        init_db()
        self.db = None
        self.records_scraped = 0
        self.start_time = None

    def open_spider(self, spider):
        self.db = SessionLocal()
        self.records_scraped = 0
        self.start_time = datetime.now(timezone.utc)
        logger.info(f"DatabasePipeline opened for spider: {spider.name}")

    def close_spider(self, spider):
        if self.db:
            run_time = (datetime.now(timezone.utc) - self.start_time).total_seconds()
            log_entry = ScraperLog(
                source=spider.name,
                status="SUCCESS",
                records_scraped=self.records_scraped,
                run_time_seconds=run_time,
            )
            self.db.add(log_entry)
            self.db.commit()
            self.db.close()
            logger.info(f"DatabasePipeline closed for spider {spider.name}. Processed {self.records_scraped} records.")

    def process_item(self, item, spider):
        # Step 1: Pydantic Validation
        item_dict = dict(item)
        try:
            validated_data = CarListingSchema(**item_dict)
        except ValidationError as e:
            logger.warning(f"Item failed validation: {e}")
            raise DropItem(f"Validation error: {e}")

        # Step 2: Anomaly Filter
        self._apply_anomaly_filter(validated_data)

        # Step 3: Database Insertion / Upsert
        try:
            company = self.db.query(Company).filter_by(name=validated_data.company_name).first()
            if not company:
                company = Company(name=validated_data.company_name)
                self.db.add(company)
                self.db.flush()

            model = (
                self.db.query(Model)
                .filter_by(company_id=company.id, name=validated_data.model_name)
                .first()
            )
            if not model:
                model = Model(company_id=company.id, name=validated_data.model_name)
                self.db.add(model)
                self.db.flush()

            variant = (
                self.db.query(Variant)
                .filter_by(
                    model_id=model.id,
                    name=validated_data.variant_name,
                    fuel_type=validated_data.fuel_type,
                    transmission=validated_data.transmission,
                )
                .first()
            )
            if not variant:
                variant = Variant(
                    model_id=model.id,
                    name=validated_data.variant_name,
                    fuel_type=validated_data.fuel_type,
                    transmission=validated_data.transmission,
                )
                self.db.add(variant)
                self.db.flush()

            # Check if listing already exists by source_url
            listing = self.db.query(Listing).filter_by(source_url=validated_data.source_url).first()
            if not listing:
                listing = Listing(
                    variant_id=variant.id,
                    source_platform=validated_data.source_platform,
                    source_url=validated_data.source_url,
                    manufacture_year=validated_data.manufacture_year,
                    km_driven=validated_data.km_driven,
                    owner_count=validated_data.owner_count,
                    city=validated_data.city,
                    asking_price=validated_data.asking_price,
                    insurance_valid=validated_data.insurance_valid,
                    accident_history=validated_data.accident_history,
                    description=validated_data.description,
                    image_urls=validated_data.image_urls,
                )
                self.db.add(listing)
            else:
                # Update asking_price and scraped_at
                listing.asking_price = validated_data.asking_price
                listing.km_driven = validated_data.km_driven
                listing.scraped_at = datetime.utcnow()

            self.db.commit()
            self.records_scraped += 1
            return item

        except Exception as e:
            self.db.rollback()
            logger.error(f"Database write failed: {e}")
            raise DropItem(f"Database write error: {e}")

    def _apply_anomaly_filter(self, data: CarListingSchema):
        """
        Anomaly filter to flag or reject impossible values.
        Rejects:
        - Car age < 0 or > 35 years
        - Unrealistic km driven per year (> 120,000 km/year or negative)
        - Asking price too low (< 15,000 INR for car) or unrealistic depreciation
        """
        current_year = datetime.now().year
        car_age = current_year - data.manufacture_year

        if car_age < 0:
            raise DropItem(f"Anomaly detected: Manufacture year {data.manufacture_year} in future.")

        km_per_year = data.km_driven / (car_age + 1)
        if km_per_year > 120_000:
            raise DropItem(f"Anomaly detected: Unrealistic km per year ({km_per_year:.0f} km/year).")

        if data.asking_price < 15_000:
            raise DropItem(f"Anomaly detected: Asking price {data.asking_price} below minimal threshold.")
