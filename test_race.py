from scraper.netkeiba import scrape_race
from scraper.db import get_session
from models.predict import predict_and_store
from sqlalchemy import text

netkeiba_id = "202605020301"
print(f"Scraping race {netkeiba_id}...")
result = scrape_race(netkeiba_id)

if result:
    print(f"Successfully scraped: {result.race_name_jp} ({len(result.entries)} entries)")
else:
    print("Scraping failed or no result returned.")
    exit(1)

# Look up internal ID
with get_session() as session:
    race = session.execute(
        text("SELECT id FROM races WHERE netkeiba_id = :jid"), 
        {"jid": netkeiba_id}
    ).fetchone()

if not race:
    print("Could not find race in database after scraping.")
    exit(1)

internal_id = race[0]
print(f"Internal race ID: {internal_id}")

# Predict
print("Running predictions...")
df = predict_and_store(
    race_id=internal_id,
    model_version="retrain_20260507_1645",
    store_to_db=False  # just testing
)

if df.empty:
    print("No predictions generated.")
else:
    print(f"\n🎯 Predictions — Race #{internal_id}")
    print("-" * 75)
    print(f"{'Horse':<16} {'Model':>6} {'Pace':>6} {'Combined':>8} {'Odds':>5} {'MktP':>5} {'EV':>6} {'Value':>5}")
    print("-" * 75)
    for _, row in df.iterrows():
        name = str(row.get("horse_name", "?"))[:14]
        is_val = "✅" if row.get("is_value") else ""
        print(
            f"{name:<16} "
            f"{row['model_win_prob']:>5.1%} "
            f"{row['pace_win_prob']:>5.1%} "
            f"{row['combined_win_prob']:>7.1%} "
            f"{row['odds']:>5.1f} "
            f"{row['market_prob']:>5.1%} "
            f"{row['ev']:>+5.2f} "
            f"{is_val:>5}"
        )
