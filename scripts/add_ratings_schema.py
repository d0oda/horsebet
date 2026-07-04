from scraper.db import get_session
from sqlalchemy import text

def add_schema():
    with get_session() as session:
        session.execute(text("""
        CREATE TABLE IF NOT EXISTS horse_ratings (
            horse_id INTEGER PRIMARY KEY,
            ability_rating REAL,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """))
        session.execute(text("""
        CREATE TABLE IF NOT EXISTS horse_rating_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            horse_id INTEGER,
            race_id INTEGER,
            race_date TEXT,
            rating_before REAL,
            race_score REAL,
            rating_after REAL,
            components TEXT
        );
        """))
        session.execute(text("CREATE INDEX IF NOT EXISTS idx_horse_rating_hist_horse ON horse_rating_history(horse_id);"))
        session.execute(text("CREATE INDEX IF NOT EXISTS idx_horse_rating_hist_date ON horse_rating_history(race_date);"))
        print("Successfully created horse_ratings and horse_rating_history tables.")

if __name__ == "__main__":
    add_schema()
