from scraper.db import get_session
from sqlalchemy import text
with get_session() as session:
    res = session.execute(text("SELECT DISTINCT bet_type FROM odds_snapshots;"))
    for row in res:
        print(row)
