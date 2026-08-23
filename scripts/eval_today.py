import pandas as pd
from sqlalchemy import text, bindparam
from scraper.db import get_session
from models.predict_final import predict_with_filters

def main():
    with get_session() as session:
        rows = session.execute(
            text("SELECT id FROM races WHERE date = '2026-05-23' ORDER BY race_number")
        ).fetchall()
    race_ids = [r[0] for r in rows]
    
    # Predict all horses for today
    df = predict_with_filters(
        race_ids=race_ids,
        model_version="retrain_20260523_2115",
        ev_threshold=-999.0,  # No EV filter
        max_odds=9999.0,      # No odds filter
        min_odds=1.0,         
        bankroll=100000,
        kelly_fraction=0.25,
        flat=True
    )
    
    # df has: race_id, entry_id, combined_prob, horse_name
    
    # Fetch actual finish positions
    with get_session() as session:
        entries = pd.read_sql(
            text("SELECT id as entry_id, finish_pos FROM entries WHERE race_id IN :rids").bindparams(bindparam("rids", expanding=True)),
            session.bind,
            params={"rids": race_ids}
        )
        
    df = df.merge(entries, on="entry_id")
    
    # Accuracy: Did the model's top predicted horse win?
    df["rank"] = df.groupby("race_id")["combined_prob"].rank("dense", ascending=False)
    
    winners = df[df["finish_pos"] == 1]
    
    top1_wins = len(winners[winners["rank"] == 1])
    top2_wins = len(winners[winners["rank"] <= 2])
    top3_wins = len(winners[winners["rank"] <= 3])
    
    print(f"Total Races: {len(race_ids)}")
    print(f"Top 1 Predicted Horse Won: {top1_wins} ({top1_wins/len(race_ids):.1%})")
    print(f"Top 2 Predicted Horse Won: {top2_wins} ({top2_wins/len(race_ids):.1%})")
    print(f"Top 3 Predicted Horse Won: {top3_wins} ({top3_wins/len(race_ids):.1%})")

    # Let's see the average rank of the winner
    avg_winner_rank = winners["rank"].mean()
    print(f"Average Rank of Winner: {avg_winner_rank:.2f}")

if __name__ == "__main__":
    main()
