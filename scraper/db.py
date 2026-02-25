"""
UmaEdge — Database connection utility.
Provides SQLAlchemy engine and session for the horsebet schema.
"""

import os
from contextlib import contextmanager
from dotenv import load_dotenv
from sqlalchemy import create_engine, event, text
from sqlalchemy.orm import sessionmaker, Session

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise RuntimeError(
        "DATABASE_URL not set. Copy .env.example to .env and fill in your Supabase credentials."
    )

engine = create_engine(
    DATABASE_URL,
    pool_size=5,
    max_overflow=10,
    pool_pre_ping=True,
)


@event.listens_for(engine, "connect")
def _set_search_path(dbapi_conn, connection_record):
    """Set search_path on every new raw connection (works through poolers)."""
    cursor = dbapi_conn.cursor()
    cursor.execute("SET search_path TO horsebet, public")
    cursor.close()
    dbapi_conn.commit()


SessionLocal = sessionmaker(bind=engine)


@contextmanager
def get_session() -> Session:
    """Context manager that yields a SQLAlchemy session with auto-commit/rollback."""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def test_connection():
    """Quick test to verify DB connectivity and schema access."""
    with get_session() as session:
        result = session.execute(
            text("SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'horsebet'")
        )
        count = result.scalar()
        print(f"✅ Connected to Supabase. Found {count} tables in horsebet schema.")
        return count


if __name__ == "__main__":
    test_connection()
