import json
import sys

def print_bets(filepath):
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Could not find {filepath}")
        return

    print("=" * 60)
    print(f"🏆 BETTING SLIP FOR: {data.get('date', 'Unknown Date')}")
    print(f"📈 Strategy: {data.get('strategy', 'Unknown')}")
    print("=" * 60)
    
    total_bets = 0
    
    for race in data.get('races', []):
        race_num = race.get('race_number')
        venue = race.get('venue')
        
        bets_in_race = [e for e in race.get('entries', []) if e.get('is_bet')]
        
        if bets_in_race:
            print(f"\n🏇 Race {race_num} @ {venue}")
            for bet in bets_in_race:
                post = bet.get('post_position', '-')
                name = bet.get('horse_name', 'Unknown')
                odds = bet.get('odds', 0.0)
                ev = bet.get('ev', 0.0)
                prob = bet.get('prob_combined', 0.0)
                
                print(f"   ► Bet: #{post} {name}")
                print(f"      Odds: {odds}x | Win Prob: {prob}% | EV: +{ev}%")
                total_bets += 1

    print("\n" + "=" * 60)
    if total_bets == 0:
        print("❌ No value bets found for this session based on your thresholds.")
    else:
        print(f"💰 Total Recommended Bets: {total_bets}")
    print("=" * 60)

if __name__ == "__main__":
    file_path = "data/data.json"
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    print_bets(file_path)
