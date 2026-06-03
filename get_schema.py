from scraper.db import get_session
from sqlalchemy import text
with get_session() as session:
    res = session.execute(text("SELECT table_name, column_name, data_type FROM information_schema.columns WHERE table_schema = 'horsebet' AND table_name IN ('results', 'odds_snapshots', 'races', 'entries');"))
    for row in res:
        print(row)
