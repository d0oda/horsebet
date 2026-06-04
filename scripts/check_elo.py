import pandas as pd
from sqlalchemy import text
from scraper.db import get_session

def main():
    with get_session() as session:
        query = """
        WITH todays_horses AS (
            SELECT r.id as race_id, e.horse_id, h.name as horse_name, e.finish_pos
            FROM races r
            JOIN entries e ON r.id = e.race_id
            JOIN horses h ON e.horse_id = h.id
            WHERE r.date = '2026-05-23'
        ),
        latest_ratings AS (
            SELECT hrh.horse_id, hrh.rating_after,
                   ROW_NUMBER() OVER(PARTITION BY hrh.horse_id ORDER BY hrh.race_date DESC) as rn
            FROM horse_rating_history hrh
            JOIN todays_horses th ON hrh.horse_id = th.horse_id
            WHERE hrh.race_date < '2026-05-23'
        )
        SELECT th.horse_name, th.finish_pos, COALESCE(lr.rating_after, 55.0) as pre_race_elo
        FROM todays_horses th
        LEFT JOIN latest_ratings lr ON th.horse_id = lr.horse_id AND lr.rn = 1
        ORDER BY pre_race_elo DESC
        """
        
        df = pd.read_sql(text(query), session.bind)
        
    print("=== TOP 10 HIGHEST ELO HORSES TODAY ===")
    print(df.head(10).to_string(index=False))
    
    print("\n=== ELO STATS BY FINISH POSITION ===")
    print(df.groupby('finish_pos')['pre_race_elo'].mean().sort_index().head(5))
    print(f"\nAverage Elo overall: {df['pre_race_elo'].mean():.1f}")
    
if __name__ == "__main__":
    main()
