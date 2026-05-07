from scraper.db import get_session
from sqlalchemy import text

with get_session() as session:
    res = session.execute(text("""
        UPDATE horsebet.entries e
        SET finish_pos = r.finish_pos,
            time_secs = r.time_secs,
            last_3f_secs = r.last_3f_secs,
            corner_positions = r.corner_positions,
            margin = r.margin
        FROM horsebet.results r
        WHERE e.id = r.entry_id
          AND (e.finish_pos IS NULL OR e.time_secs IS NULL)
    """))
    print(f"Updated {res.rowcount} entries with result data.")
