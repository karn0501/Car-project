"""
Database connection and session management module.
Defaults to PostgreSQL database (postgresql://postgres:CarPassword123!@localhost:5432/car_prediction_db),
with automatic fallback to local SQLite database if PostgreSQL server is offline.
"""

import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

POSTGRES_DB_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:CarPassword123!@localhost:5432/car_prediction_db"
)

if POSTGRES_DB_URL.startswith("postgres://"):
    POSTGRES_DB_URL = POSTGRES_DB_URL.replace("postgres://", "postgresql://", 1)

SQLITE_FALLBACK_URL = f"sqlite:///{os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'car_prediction.db'))}"

try:
    connect_args = {"check_same_thread": False} if POSTGRES_DB_URL.startswith("sqlite") else {}
    engine = create_engine(POSTGRES_DB_URL, connect_args=connect_args, echo=False)
    # Test connection to ensure server is reachable
    with engine.connect() as conn:
        pass
except Exception:
    # Graceful fallback to SQLite if PostgreSQL service is offline
    engine = create_engine(SQLITE_FALLBACK_URL, connect_args={"check_same_thread": False}, echo=False)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    """Dependency for obtaining database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Initializes tables in the target database."""
    Base.metadata.create_all(bind=engine)
