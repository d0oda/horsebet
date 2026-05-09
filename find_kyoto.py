from sqlalchemy import text
from scraper.db import get_session
import sys

with get_session() as session:
    res = session.execute(text("SELECT netkeiba_id FROM horsebet.races WHERE date = '2026-05-09' AND course_id = 8 AND race_number = 7")).fetchone()
    if res:
        print(f"NETKEIBA_ID={res[0]}")
    else:
        print("Not found")
