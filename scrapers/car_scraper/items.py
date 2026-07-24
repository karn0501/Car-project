"""
Scrapy Item definitions and Pydantic validation schema for scraped used car listings.
"""

from typing import Optional
from datetime import datetime
import scrapy
from pydantic import BaseModel, Field, field_validator


class CarListingSchema(BaseModel):
    """Pydantic validation schema for a single car listing."""
    company_name: str = Field(..., min_length=1)
    model_name: str = Field(..., min_length=1)
    variant_name: str = Field(..., min_length=1)
    source_platform: str = Field(..., min_length=1)
    source_url: str = Field(..., min_length=5)
    manufacture_year: int = Field(..., ge=1990, le=datetime.now().year + 1)
    km_driven: float = Field(..., ge=0.0, le=1_000_000.0)
    owner_count: int = Field(default=1, ge=1, le=10)
    city: str = Field(..., min_length=1)
    asking_price: float = Field(..., ge=10_000.0, le=50_000_000.0)
    fuel_type: str = Field(default="Petrol")
    transmission: str = Field(default="Manual")
    insurance_valid: bool = Field(default=True)
    accident_history: bool = Field(default=False)
    description: Optional[str] = None
    image_urls: Optional[str] = None

    @field_validator("company_name", "model_name", "variant_name", "city")
    @classmethod
    def clean_strings(cls, v: str) -> str:
        return v.strip().title() if v else v


class CarListingItem(scrapy.Item):
    """Scrapy Item class for transporting scraped car listing data through pipelines."""
    company_name = scrapy.Field()
    model_name = scrapy.Field()
    variant_name = scrapy.Field()
    source_platform = scrapy.Field()
    source_url = scrapy.Field()
    manufacture_year = scrapy.Field()
    km_driven = scrapy.Field()
    owner_count = scrapy.Field()
    city = scrapy.Field()
    asking_price = scrapy.Field()
    fuel_type = scrapy.Field()
    transmission = scrapy.Field()
    insurance_valid = scrapy.Field()
    accident_history = scrapy.Field()
    description = scrapy.Field()
    image_urls = scrapy.Field()
