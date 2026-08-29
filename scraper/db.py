"""
UmaEdge — Database connection utility.

Provides a SQLAlchemy engine and session for the local SQLite database.
All UmaEdge tables (races, entries, horses, etc.) are stored in horsebet.db.

DATABASE_URL is read from .env, e.g.:
    DATABASE_URL=sqlite:///horsebet.db
"""

import os
from contextlib import contextmanager
from dotenv import load_dotenv
from sqlalchemy import create_engine, text, event
from sqlalchemy.orm import sessionmaker, Session

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    raise RuntimeError(
        "DATABASE_URL not set. Add DATABASE_URL=sqlite:///horsebet.db to your .env file."
    )

from sqlalchemy.pool import NullPool

connect_args = {"check_same_thread": False}
if DATABASE_URL.startswith("sqlite"):
    connect_args["timeout"] = 30

engine = create_engine(
    DATABASE_URL,
    poolclass=NullPool,
    connect_args=connect_args,
)

@event.listens_for(engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    if DATABASE_URL and DATABASE_URL.startswith("sqlite"):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.execute("PRAGMA busy_timeout=30000")
        cursor.close()


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
    """Quick test to verify DB connectivity."""
    with get_session() as session:
        count = session.execute(text("SELECT COUNT(*) FROM races")).scalar()
        print(f"✅ Connected to SQLite. Found {count:,} races.")
        return count


if __name__ == "__main__":
    test_connection()
