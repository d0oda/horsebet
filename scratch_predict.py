import sys
sys.path.insert(0, '/Users/ryfei.wang/Documents/horsebet')
from pipeline import step_predict
from scraper.db import get_session
from scraper.odds_watcher import fetch_win_odds
from sqlalchemy import text

date = '2026-05-09'

with get_session() as session:
    res = session.execute(text("SELECT r.id, r.netkeiba_id FROM horsebet.races r JOIN horsebet.courses c ON r.course_id = c.id WHERE r.date = '2026-05-09' AND c.name ILIKE '%tokyo%' AND r.race_number = 8")).fetchone()
    if not res:
        print("Race not found!")
        sys.exit(1)
    db_id = res.id
    nk_id = res.netkeiba_id

print(f"Fetching odds for Tokyo R8 (nk_id: {nk_id}, db_id: {db_id})")
odds = fetch_win_odds(nk_id)
if odds:
    with get_session() as session:
        for o in odds:
            session.execute(
                text("UPDATE horsebet.entries SET odds_win = :odds WHERE race_id = :race_id AND post_position = :pp"),
                {"odds": o["odds_value"], "race_id": db_id, "pp": int(o["combination"])}
            )
        session.commit()
    print("Odds updated.")

print("Running prediction...")
output = step_predict(date, "retrain_20260507_1645", 0.25, 30.0, 2.0, [db_id])
print("Prediction complete. Output path:", output)
