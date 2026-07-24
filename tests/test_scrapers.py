"""
Unit tests for Scraper Spider Parsing, Anomaly Filtering, and Item Pipeline logic.
"""

import pytest
from datetime import datetime
from scrapy.exceptions import DropItem
from scrapy.selector import Selector

from scrapers.car_scraper.items import CarListingItem, CarListingSchema
from scrapers.car_scraper.spiders.cardekho_spider import CarDekhoSpider
from scrapers.car_scraper.pipelines import DatabasePipeline


def test_clean_price_parser():
    """Test price cleaning utility function for various Indian currency representations."""
    assert CarDekhoSpider.clean_price("₹ 5.25 Lakh") == 525000.0
    assert CarDekhoSpider.clean_price("₹ 1.2 Crore") == 12000000.0
    assert CarDekhoSpider.clean_price("₹ 4,50,000") == 450000.0
    assert CarDekhoSpider.clean_price("450000") == 450000.0
    assert CarDekhoSpider.clean_price("") == 0.0


def test_spider_card_parsing():
    """Test HTML selector card parsing in CarDekho spider."""
    html_snippet = """
    <div class="carCard">
        <h3 class="title"><a href="/used-car/123">2020 Hyundai Creta SX</a></h3>
        <span class="price">₹ 10.5 Lakh</span>
        <div class="overviewDetail">40,000 kms • Diesel • Automatic</div>
    </div>
    """
    selector = Selector(text=html_snippet)
    item = CarDekhoSpider.parse_card_element(selector, "https://www.cardekho.com")

    assert item is not None
    assert item["company_name"] == "Hyundai"
    assert item["model_name"] == "Creta"
    assert item["variant_name"] == "SX"
    assert item["manufacture_year"] == 2020
    assert item["asking_price"] == 1050000.0
    assert item["km_driven"] == 40000.0
    assert item["fuel_type"] == "Diesel"
    assert item["transmission"] == "Automatic"


def test_item_pydantic_schema_validation():
    """Test Pydantic schema validation rules."""
    valid_data = {
        "company_name": "Honda",
        "model_name": "City",
        "variant_name": "VX",
        "source_platform": "CarDekho",
        "source_url": "https://example.com/honda-city-1",
        "manufacture_year": 2018,
        "km_driven": 50000.0,
        "owner_count": 1,
        "city": "Mumbai",
        "asking_price": 650000.0,
        "fuel_type": "Petrol",
        "transmission": "Manual",
    }
    schema = CarListingSchema(**valid_data)
    assert schema.company_name == "Honda"
    assert schema.asking_price == 650000.0


def test_anomaly_filter_rejections():
    """Test anomaly filter rules in DatabasePipeline."""
    pipeline = DatabasePipeline()

    # 1. Future manufacture year
    future_item = CarListingSchema.model_construct(
        company_name="Maruti",
        model_name="Swift",
        variant_name="VXI",
        source_platform="CarDekho",
        source_url="https://example.com/future",
        manufacture_year=datetime.now().year + 5,
        km_driven=1000.0,
        city="Delhi",
        asking_price=500000.0,
        fuel_type="Petrol",
        transmission="Manual",
    )
    with pytest.raises(DropItem, match=r"(?i)manufacture year"):
        pipeline._apply_anomaly_filter(future_item)

    # 2. Unrealistic km per year (> 120,000 km/year)
    high_km_item = CarListingSchema.model_construct(
        company_name="Maruti",
        model_name="Swift",
        variant_name="VXI",
        source_platform="CarDekho",
        source_url="https://example.com/highkm",
        manufacture_year=datetime.now().year - 1,
        km_driven=300000.0,
        city="Delhi",
        asking_price=500000.0,
        fuel_type="Petrol",
        transmission="Manual",
    )
    with pytest.raises(DropItem, match="Unrealistic km"):
        pipeline._apply_anomaly_filter(high_km_item)

    # 3. Unrealistic low price (< 15,000 INR)
    low_price_item = CarListingSchema.model_construct(
        company_name="Maruti",
        model_name="Swift",
        variant_name="VXI",
        source_platform="CarDekho",
        source_url="https://example.com/lowprice",
        manufacture_year=2015,
        km_driven=50000.0,
        city="Delhi",
        asking_price=5000.0,
        fuel_type="Petrol",
        transmission="Manual",
    )
    with pytest.raises(DropItem, match="below minimal threshold"):
        pipeline._apply_anomaly_filter(low_price_item)

