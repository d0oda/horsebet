import pandas as pd
from sqlalchemy import text, bindparam
from scraper.db import get_session
from models.predict_final import predict_with_filters

def main():
    with get_session() as session:
        dates = session.execute(
            text("SELECT DISTINCT date FROM races WHERE date >= '2026-05-01' AND date <= '2026-05-23' ORDER BY date")
        ).fetchall()
        dates = [d[0] for d in dates]

    all_dfs = []
    total_races = 0
    for d in dates:
        with get_session() as session:
            rows = session.execute(
                text("SELECT id FROM races WHERE date = :d ORDER BY race_number"),
                {"d": d}
            ).fetchall()
            race_ids = [r[0] for r in rows]
        
        print(f"Running predictions on {len(race_ids)} races for {d}...")
        total_races += len(race_ids)
        
        # Predict all horses for this day
        df = predict_with_filters(
            race_ids=race_ids,
            model_version="20260605_113654",
            ev_threshold=0.50,  
            max_odds=60.0,      
            min_odds=1.5,         
            bankroll=100000,
            kelly_fraction=0.0,
            flat=True
        )
        all_dfs.append(df)
        
    df = pd.concat(all_dfs, ignore_index=True)
    
    value_bets = df[df["is_value_bet"] == True].copy()
    
    # Fetch actual finish positions and payouts
    with get_session() as session:
        all_race_ids = df["race_id"].unique().tolist()
        entries = pd.read_sql(
            text("SELECT id as entry_id, finish_pos, odds_win FROM entries WHERE race_id IN :rids").bindparams(bindparam("rids", expanding=True)),
            session.bind,
            params={"rids": all_race_ids}
        )
        
    value_bets = value_bets.merge(entries, on="entry_id", suffixes=('', '_actual'))
    
    # Calculate returns
    flat_stake = 1000
    total_staked = len(value_bets) * flat_stake
    
    # Profit calculation
    # If finish_pos == 1, payout = odds_win * stake. 
    # Net profit = payout - stake. If loss, net profit = -stake
    value_bets['payout'] = value_bets.apply(
        lambda row: row['odds_win'] * flat_stake if row['finish_pos'] == 1 else 0.0, 
        axis=1
    )
    value_bets['net_profit'] = value_bets['payout'] - flat_stake
    
    total_payout = value_bets['payout'].sum()
    net_profit = value_bets['net_profit'].sum()
    roi = (net_profit / total_staked) if total_staked > 0 else 0.0
    
    winners = value_bets[value_bets['finish_pos'] == 1]
    
    print("\n" + "="*50)
    print("      UMAEDGE PERFORMANCE — MAY 2026 (SO FAR)      ")
    print("="*50)
    print(f"Total Races Evaluated : {total_races}")
    print(f"Value Bets Placed     : {len(value_bets)}")
    print(f"Winners Caught        : {len(winners)}")
    print(f"Hit Rate              : {(len(winners)/len(value_bets) if len(value_bets) > 0 else 0.0):.1%}")
    print("-"*50)
    print(f"Total Staked          : ¥{total_staked:,.0f}")
    print(f"Total Returned        : ¥{total_payout:,.0f}")
    print(f"Net Profit            : ¥{net_profit:,.0f}")
    print(f"ROI                   : {roi:+.2%}")
    print("="*50)
    
    if len(winners) > 0:
        print("\nNotable Winners Caught:")
        top_winners = winners.sort_values(by='odds_win', ascending=False).head(5)
        for _, w in top_winners.iterrows():
            print(f" - {w['horse_name']} (Odds: {w['odds_win']:.1f}x) -> +¥{w['net_profit']:,.0f}")

if __name__ == "__main__":
    main()
