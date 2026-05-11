import pytest
from models.pace_sim import PaceSimulator

def _make_entries(n):
    return [
        {"horse_id": i+1, "avg_first_corner": i+1, "avg_last_3f": 34.0, "career_win_pct": 0.1, "odds_win": 10.0, "last3_avg_finish": 5.0}
        for i in range(n)
    ]

sim = PaceSimulator(n_simulations=10, seed=42)
res = sim.simulate_race(_make_entries(5), 2000)
print(res.keys())
