from scraper.db import get_session
from sqlalchemy import text

with get_session() as session:
    rows = session.execute(text("SELECT id, netkeiba_id, post_time FROM horsebet.races WHERE date='2026-05-23' LIMIT 5")).fetchall()
    for row in rows:
        print(row)
