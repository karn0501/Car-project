"""
Unit tests for Database ORM Models and Session Management.
"""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from db.models import Base, Company, Model, Variant, Listing, ScraperLog


@pytest.fixture
def db_session():
    """Create in-memory SQLite database session for unit testing."""
    engine = create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(bind=engine)
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = TestingSessionLocal()
    yield session
    session.close()


def test_company_model_variant_hierarchy(db_session):
    """Test relational hierarchy: Company -> Model -> Variant -> Listing."""
    company = Company(name="Maruti Suzuki", country="India")
    db_session.add(company)
    db_session.commit()

    model = Model(company_id=company.id, name="Swift", body_type="Hatchback", launch_year=2005)
    db_session.add(model)
    db_session.commit()

    variant = Variant(
        model_id=model.id,
        name="VXI",
        fuel_type="Petrol",
        transmission="Manual",
        engine_cc=1197,
        ex_showroom_price=600000.0,
    )
    db_session.add(variant)
    db_session.commit()

    listing = Listing(
        variant_id=variant.id,
        source_platform="CarDekho",
        source_url="https://example.com/listing/1",
        manufacture_year=2020,
        km_driven=35000.0,
        owner_count=1,
        city="Delhi",
        asking_price=520000.0,
    )
    db_session.add(listing)
    db_session.commit()

    # Assertions
    fetched_company = db_session.query(Company).filter_by(name="Maruti Suzuki").first()
    assert fetched_company is not None
    assert len(fetched_company.models) == 1
    assert fetched_company.models[0].name == "Swift"

    fetched_model = fetched_company.models[0]
    assert len(fetched_model.variants) == 1
    assert fetched_model.variants[0].name == "VXI"

    fetched_variant = fetched_model.variants[0]
    assert len(fetched_variant.listings) == 1
    assert fetched_variant.listings[0].asking_price == 520000.0


def test_scraper_log_creation(db_session):
    """Test ScraperLog table persistence."""
    log = ScraperLog(
        source="cardekho",
        status="SUCCESS",
        records_scraped=150,
        run_time_seconds=12.5,
    )
    db_session.add(log)
    db_session.commit()

    fetched_log = db_session.query(ScraperLog).filter_by(source="cardekho").first()
    assert fetched_log is not None
    assert fetched_log.records_scraped == 150
    assert fetched_log.status == "SUCCESS"
