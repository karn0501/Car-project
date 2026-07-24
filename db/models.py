"""
SQLAlchemy ORM models representing the Used Car Price Prediction schema.
Reflects Section 4.1 of the Project Documentation.
"""

from datetime import datetime, timezone
from sqlalchemy import (
    Column, Integer, String, Float, Boolean, DateTime, ForeignKey, Text
)
from sqlalchemy.orm import relationship
from db.database import Base


class Company(Base):
    __tablename__ = "companies"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), unique=True, nullable=False, index=True)
    logo_url = Column(String(255), nullable=True)
    country = Column(String(50), nullable=True)

    models = relationship("Model", back_populates="company", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Company(name='{self.name}')>"


class Model(Base):
    __tablename__ = "models"

    id = Column(Integer, primary_key=True, index=True)
    company_id = Column(Integer, ForeignKey("companies.id"), nullable=False)
    name = Column(String(100), nullable=False, index=True)
    body_type = Column(String(50), nullable=True)  # Hatchback, Sedan, SUV, MPV, etc.
    launch_year = Column(Integer, nullable=True)
    discontinued_year = Column(Integer, nullable=True)

    company = relationship("Company", back_populates="models")
    variants = relationship("Variant", back_populates="model", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Model(name='{self.name}')>"


class Variant(Base):
    __tablename__ = "variants"

    id = Column(Integer, primary_key=True, index=True)
    model_id = Column(Integer, ForeignKey("models.id"), nullable=False)
    name = Column(String(150), nullable=False, index=True)  # e.g., LXI, VXI, ZXI(O)
    fuel_type = Column(String(30), nullable=False)  # Petrol, Diesel, CNG, Electric, Hybrid
    transmission = Column(String(30), nullable=False)  # Manual, Automatic
    engine_cc = Column(Integer, nullable=True)
    seating_capacity = Column(Integer, nullable=True)
    ex_showroom_price = Column(Float, nullable=True)
    launch_date = Column(String(30), nullable=True)

    model = relationship("Model", back_populates="variants")
    listings = relationship("Listing", back_populates="variant", cascade="all, delete-orphan")
    price_histories = relationship("PriceHistory", back_populates="variant", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Variant(name='{self.name}', fuel='{self.fuel_type}')>"


class Listing(Base):
    __tablename__ = "listings"

    id = Column(Integer, primary_key=True, index=True)
    variant_id = Column(Integer, ForeignKey("variants.id"), nullable=False)
    source_platform = Column(String(50), nullable=False)  # CarDekho, Spinny, Cars24, etc.
    source_url = Column(String(500), unique=True, nullable=False)
    manufacture_year = Column(Integer, nullable=False)
    km_driven = Column(Float, nullable=False)
    owner_count = Column(Integer, nullable=False, default=1)
    city = Column(String(100), nullable=False, index=True)
    asking_price = Column(Float, nullable=False)
    insurance_valid = Column(Boolean, default=True)
    accident_history = Column(Boolean, default=False)
    description = Column(Text, nullable=True)
    image_urls = Column(Text, nullable=True)  # JSON or comma-separated URLs
    scraped_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    is_active = Column(Boolean, default=True)

    variant = relationship("Variant", back_populates="listings")

    def __repr__(self):
        return f"<Listing(price={self.asking_price}, year={self.manufacture_year}, city='{self.city}')>"


class PriceHistory(Base):
    __tablename__ = "price_history"

    id = Column(Integer, primary_key=True, index=True)
    variant_id = Column(Integer, ForeignKey("variants.id"), nullable=False)
    city = Column(String(100), nullable=False)
    avg_price = Column(Float, nullable=False)
    recorded_date = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    variant = relationship("Variant", back_populates="price_histories")


class ScraperLog(Base):
    __tablename__ = "scraper_logs"

    id = Column(Integer, primary_key=True, index=True)
    source = Column(String(50), nullable=False)
    status = Column(String(30), nullable=False)  # SUCCESS, FAILED, WARNING
    records_scraped = Column(Integer, default=0)
    run_time_seconds = Column(Float, default=0.0)
    error_message = Column(Text, nullable=True)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class Feedback(Base):
    __tablename__ = "feedback"

    id = Column(Integer, primary_key=True, index=True)
    variant_name = Column(String(150), nullable=True)
    manufacture_year = Column(Integer, nullable=True)
    actual_sold_price = Column(Float, nullable=False)
    predicted_price = Column(Float, nullable=True)
    user_comments = Column(Text, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
