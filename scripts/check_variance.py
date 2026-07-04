import pandas as pd
import numpy as np
from sqlalchemy import text
from scraper.db import get_session

def check_winners():
    print("Reading features and odds...")
    # Read the same races
    with get_session() as session:
        dates = session.execute(
            text("SELECT DISTINCT date FROM races WHERE date >= '2026-05-01' AND date <= '2026-05-23' ORDER BY date")
        ).fetchall()
        dates = [d[0] for d in dates]

        race_ids = []
        for d in dates:
            rows = session.execute(
                text("SELECT id FROM races WHERE date = :d ORDER BY race_number"),
                {"d": d}
            ).fetchall()
            race_ids.extend([r[0] for r in rows])
            
        entries = pd.read_sql(
            text("SELECT id as entry_id, race_id, finish_pos as finish_pos_actual, odds_win as odds_win_actual FROM entries WHERE race_id = ANY(:rids)"),
            session.bind,
            params={"rids": race_ids}
        )
        
    print(f"Loaded {len(entries)} entries.")
    
    # Check what happens if we just randomly bet on a horse with odds between 30 and 100
    longshots = entries[(entries["odds_win_actual"] >= 30.0) & (entries["odds_win_actual"] <= 100.0)]
    print(f"\nTotal longshots (30-100 odds): {len(longshots)}")
    winners = longshots[longshots["finish_pos_actual"] == 1]
    print(f"Winners among them: {len(winners)}")
    print("Winning odds:")
    print(winners["odds_win_actual"].values)
    
if __name__ == "__main__":
    check_winners()
