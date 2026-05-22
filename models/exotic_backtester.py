import logging
import multiprocessing
from dataclasses import dataclass
from collections import defaultdict
from itertools import permutations, combinations
import numpy as np
import pandas as pd
from tqdm import tqdm

from agents.exotic_bets import TicketConstructor, ExoticBetType, TicketRecommendation
from models.pace_sim import PaceSimulator, classify_running_style, STYLE_FRONT

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("exotic_backtest")


def harville_exacta_prob(probs: dict[int, float], a: int, b: int) -> float:
    """P(a finishes 1st AND b finishes 2nd) under Harville model."""
    pa, pb = probs[a], probs[b]
    if pa >= 1.0 or pa <= 0:
        return 0.0
    return pa * (pb / (1.0 - pa))


def harville_trifecta_prob(probs: dict[int, float], a: int, b: int, c: int) -> float:
    """P(a 1st, b 2nd, c 3rd) under Harville model."""
    pa, pb, pc = probs[a], probs[b], probs[c]
    denom1 = 1.0 - pa
    denom2 = 1.0 - pa - pb
    if denom1 <= 0 or denom2 <= 0:
        return 0.0
    return pa * (pb / denom1) * (pc / denom2)


def harville_market_probs(
    market_probs: dict[int, float],
    bet_type: ExoticBetType,
) -> dict[str, float]:
    """
    Compute deterministic Harville combination probabilities for a given bet type.
    This replaces the noisy Monte Carlo approach.
    """
    horse_ids = sorted(market_probs.keys())
    result = {}

    if bet_type == ExoticBetType.EXACTA:
        for a, b in permutations(horse_ids, 2):
            key = f"{a}-{b}"
            result[key] = harville_exacta_prob(market_probs, a, b)

    elif bet_type == ExoticBetType.QUINELLA:
        for a, b in combinations(horse_ids, 2):
            key = f"{a}-{b}"
            result[key] = (
                harville_exacta_prob(market_probs, a, b)
                + harville_exacta_prob(market_probs, b, a)
            )

    elif bet_type == ExoticBetType.TRIFECTA:
        for a, b, c in permutations(horse_ids, 3):
            key = f"{a}-{b}-{c}"
            result[key] = harville_trifecta_prob(market_probs, a, b, c)

    elif bet_type == ExoticBetType.TRIO:
        for combo in combinations(horse_ids, 3):
            prob = sum(
                harville_trifecta_prob(market_probs, a, b, c)
                for a, b, c in permutations(combo)
            )
            key = "-".join(str(h) for h in sorted(combo))
            result[key] = prob

    elif bet_type == ExoticBetType.WIDE:
        # Wide = any 2 that both finish in the top 3.
        # P(a,b both in top 3) = sum over all c != a,b of all 6 trifecta
        # permutations that include a, b, and c.
        for a, b in combinations(horse_ids, 2):
            prob = 0.0
            for c in horse_ids:
                if c == a or c == b:
                    continue
                for perm in permutations([a, b, c]):
                    prob += harville_trifecta_prob(market_probs, *perm)
            key = f"{a}-{b}"
            result[key] = prob

    return result


class BacktestTicketConstructor(TicketConstructor):
    def __init__(self, win_probs: dict, horse_names: dict, entries_cache: list, market_probs: dict):
        super().__init__()
        self._mock_win_probs = win_probs
        self._mock_horse_names = horse_names
        self._entries_cache = entries_cache
        self._market_probs = market_probs

    def prepare_race_data(self, race_id: int) -> tuple[dict, dict, list]:
        finishes = self.build_finish_distribution(race_id, self._mock_win_probs)
        return self._mock_win_probs, self._mock_horse_names, finishes

    def build_finish_distribution(
        self,
        race_id: int,
        win_probs: dict,
        n_sims: int = None,
    ) -> np.ndarray:
        n_sims = n_sims or self.n_simulations
        horse_ids = sorted(win_probs.keys())
        n_horses = len(horse_ids)

        if n_horses < 2:
            return np.zeros((n_sims, n_horses), dtype=int)

        styles = []
        for horse_num in horse_ids:
            entry = next((e for e in self._entries_cache if e["post_position"] == horse_num), {})
            style = classify_running_style(
                avg_first_corner=entry.get("avg_first_corner"),
                field_size=n_horses,
                explicit_style=entry.get("running_style"),
            )
            styles.append(style)

        # Determine pace scenario weights
        n_front = sum(1 for s in styles if s == STYLE_FRONT)
        pace_fast_prob = min(0.2 + n_front * 0.15, 0.7)
        pace_slow_prob = max(0.3 - n_front * 0.1, 0.1)
        pace_mod_prob = 1.0 - pace_fast_prob - pace_slow_prob
        pace_probs = [pace_fast_prob, pace_mod_prob, pace_slow_prob]
        pace_scenarios = ["fast", "moderate", "slow"]
        
        sim_engine = PaceSimulator()
        probs = np.array([win_probs[horse_num] for horse_num in horse_ids])
        probs = np.maximum(probs, 1e-6)
        probs = probs / probs.sum()
        log_probs = np.log(probs)

        expected_modifiers = np.zeros(n_horses)
        for j in range(n_horses):
            for k, p_scenario in enumerate(pace_scenarios):
                mod = sim_engine._pace_modifier(styles[j], p_scenario)
                expected_modifiers[j] += pace_probs[k] * mod

        finishes = np.zeros((n_sims, n_horses), dtype=int)
        for i in range(n_sims):
            pace = self.rng.choice(pace_scenarios, p=pace_probs)
            gumbels = self.rng.gumbel(loc=0, scale=1, size=n_horses)
            scores = log_probs + gumbels
            for j in range(n_horses):
                raw_modifier = sim_engine._pace_modifier(styles[j], pace)
                centered_modifier = raw_modifier / expected_modifiers[j]
                scores[j] += np.log(centered_modifier)
            order = np.argsort(-scores)
            for pos, horse_idx in enumerate(order):
                finishes[i, horse_idx] = pos

        return finishes

    def evaluate_pool_tickets(
        self,
        race_id: int,
        bet_type: ExoticBetType,
        budget: int,
        win_probs: dict,
        horse_names: dict,
        finishes: np.ndarray,
    ) -> list:
        """
        Override to use Harville on both sides:
        - Model side: extract pace-adjusted marginal win probs from MC, then Harville.
        - Market side: Harville on raw market probs.
        This eliminates MC noise in combo probabilities while preserving the 
        structural pace adjustments from the simulation.
        """
        horse_ids = sorted(win_probs.keys())
        n_sims = finishes.shape[0]

        # Extract pace-adjusted marginal win probabilities from the MC simulation.
        # Count how often each horse won (position 0) across all simulations.
        model_marginals = {}
        for i, hid in enumerate(horse_ids):
            wins = np.sum(finishes[:, i] == 0)
            model_marginals[hid] = max(wins / n_sims, 1e-8)

        # Normalize to sum to 1.0
        total = sum(model_marginals.values())
        for hid in model_marginals:
            model_marginals[hid] /= total

        # Deterministic combo probs: Harville applied to pace-adjusted marginals
        combo_probs = harville_market_probs(model_marginals, bet_type)

        # Deterministic market odds: Harville applied to raw market probs
        pool_odds = self.get_pool_odds(race_id, bet_type)

        # Optimise
        tickets = self.find_optimal_tickets(combo_probs, pool_odds, budget)

        # Fill in horse names
        for t in tickets:
            ids = t.combination.split("-")
            t.horse_names = [horse_names.get(int(hid), str(hid)) for hid in ids]

        return tickets

    def get_pool_odds(self, race_id: int, bet_type: ExoticBetType) -> dict[str, float]:
        """
        Derive perfectly efficient market pool odds using the deterministic
        Harville formula applied to the market's implied win probabilities.
        No Monte Carlo noise — exact mathematical probabilities.
        """
        market_combo_probs = harville_market_probs(self._market_probs, bet_type)
        return self.estimate_odds(market_combo_probs)


def evaluate_race(race_df_tuple):
    race_id, df, entries_cache = race_df_tuple
    budget = 5000
    
    # 1. Prepare data
    win_probs = {}
    market_probs = {}
    horse_names = {}
    actual_finishes = {}
    
    for _, r in df.iterrows():
        horse_num = int(str(r["entry_id"])[-2:])
        win_probs[horse_num] = r["win_prob"]
        
        # Calculate market win prob (normalized roughly to 1.0)
        odds = float(r.get("odds_win", 0))
        if odds > 0:
            market_probs[horse_num] = 1.0 / odds
        else:
            # Fallback to model's probability so EV = 0 for missing odds data
            market_probs[horse_num] = r["win_prob"]
            
        horse_names[horse_num] = r.get("horse_name", str(horse_num))
        
        try:
            pos = float(r["finish_pos"])
            if not np.isnan(pos):
                actual_finishes[horse_num] = int(pos)
        except (ValueError, TypeError):
            pass

    sorted_finishes = sorted(actual_finishes.items(), key=lambda x: x[1])
    top_3_actual = [h for h, p in sorted_finishes[:3]]
    
    if len(top_3_actual) < 3:
        return None

    # Normalize market probs exactly to 1.0 to reflect a perfect market with 0 takeout
    market_prob_sum = sum(market_probs.values())
    for k in market_probs:
        market_probs[k] /= market_prob_sum

    # Filter cache for this race
    race_entries = [e for e in entries_cache if e["race_id"] == race_id]

    constructor = BacktestTicketConstructor(win_probs, horse_names, race_entries, market_probs)
    
    try:
        best_type, tickets = constructor.recommend_best_pool(race_id, budget=budget, store_to_db=False)
    except Exception as e:
        log.error(f"Error in recommend_best_pool for race {race_id}: {e}")
        return None
        
    if not tickets:
        return None
        
    total_cost = sum(t.cost for t in tickets)
    total_payout = 0
    hit = False
    
    for t in tickets:
        is_winner = False
        combo = [int(x) for x in t.combination.split("-")]
        
        if best_type == ExoticBetType.TRIO:
            is_winner = set(combo) == set(top_3_actual)
        elif best_type == ExoticBetType.TRIFECTA:
            is_winner = tuple(combo) == tuple(top_3_actual)
        elif best_type == ExoticBetType.EXACTA:
            is_winner = tuple(combo) == tuple(top_3_actual[:2])
        elif best_type == ExoticBetType.QUINELLA:
            is_winner = set(combo) == set(top_3_actual[:2])
        elif best_type == ExoticBetType.WIDE:
            is_winner = set(combo).issubset(set(top_3_actual))
            
        if is_winner:
            hit = True
            total_payout += t.pool_odds * t.cost
            
    return {
        "race_id": race_id,
        "bet_type": best_type.value,
        "cost": total_cost,
        "payout": total_payout,
        "profit": total_payout - total_cost,
        "hit": hit
    }

class ExoticBacktester:
    def __init__(self, pred_df: pd.DataFrame):
        self.pred_df = pred_df
        
    def run(self):
        log.info(f"Starting Exotic Backtest on {self.pred_df['race_id'].nunique()} races (Sequential execution)...")
        
        # Pre-fetch HISTORICAL running styles (from prior races, NOT the target race)
        race_ids = tuple(int(x) for x in self.pred_df["race_id"].unique())
        log.info(f"Pre-fetching historical stats for {len(race_ids)} races...")
        
        from scraper.db import get_session
        from sqlalchemy import text
        entries_cache = []
        if race_ids:
            with get_session() as session:
                # Get each horse's most recent prior running style and corner positions.
                # We join entries → horse_id → previous entries → previous results,
                # filtering to only races BEFORE the target race date.
                rows = session.execute(text("""
                    SELECT 
                        e.race_id,
                        e.post_position,
                        prev_res.running_style,
                        prev_res.corner_positions
                    FROM entries e
                    JOIN races r ON r.id = e.race_id
                    LEFT JOIN LATERAL (
                        SELECT res2.running_style, res2.corner_positions
                        FROM entries e2
                        JOIN races r2 ON r2.id = e2.race_id
                        JOIN results res2 ON res2.entry_id = e2.id
                        WHERE e2.horse_id = e.horse_id
                          AND r2.date < r.date
                        ORDER BY r2.date DESC
                        LIMIT 1
                    ) prev_res ON true
                    WHERE e.race_id IN :rids
                """), {"rids": race_ids}).fetchall()
                
                for r in rows:
                    avg_first_corner = None
                    if r.corner_positions:
                        corners = [int(c) for c in r.corner_positions.split("-") if c.isdigit()]
                        if corners:
                            avg_first_corner = corners[0]  # Use first corner position only
                            
                    entries_cache.append({
                        "race_id": r.race_id,
                        "post_position": r.post_position,
                        "running_style": r.running_style,
                        "avg_first_corner": avg_first_corner
                    })

        log.info(f"Pre-fetched {len(entries_cache)} entries (historical styles).")
        
        # Prepare iterables
        races = []
        for race_id, df in self.pred_df.groupby("race_id"):
            races.append((race_id, df, entries_cache))
        
        results = []
        for res in tqdm(map(evaluate_race, races), total=len(races)):
            if res:
                results.append(res)
                    
        # Summary
        if not results:
            log.warning("No exotic tickets placed during backtest.")
            return pd.DataFrame()
            
        results_df = pd.DataFrame(results)
        total_races_bet = len(results_df)
        total_cost = results_df["cost"].sum()
        total_payout = results_df["payout"].sum()
        total_profit = results_df["profit"].sum()
        total_hits = results_df["hit"].sum()
        roi = (total_profit / total_cost * 100) if total_cost > 0 else 0.0
        hit_rate = (total_hits / total_races_bet * 100) if total_races_bet > 0 else 0.0
        
        print("\n" + "="*50)
        print("EXOTIC WALK-FORWARD BACKTEST RESULTS")
        print("="*50)
        print(f"Races Evaluated : {len(races):,}")
        print(f"Races Bet       : {total_races_bet:,}")
        print(f"Total Hits      : {total_hits:,} ({hit_rate:.1f}%)")
        print(f"Total Staked    : ¥{total_cost:,.0f}")
        print(f"Total Payout    : ¥{total_payout:,.0f}")
        print(f"Total Profit    : ¥{total_profit:,.0f}")
        print(f"ROI             : {roi:+.2f}%")
        print("="*50)
        
        # Breakdown by bet type
        print("\nBreakdown by Bet Type:")
        print("-" * 50)
        breakdown = results_df.groupby("bet_type").agg({
            "race_id": "count",
            "cost": "sum",
            "payout": "sum",
            "profit": "sum",
            "hit": "sum"
        }).reset_index()
        
        for _, row in breakdown.iterrows():
            bt = row["bet_type"].capitalize()
            b_roi = (row["profit"] / row["cost"] * 100) if row["cost"] > 0 else 0
            b_hit = (row["hit"] / row["race_id"] * 100) if row["race_id"] > 0 else 0
            print(f"{bt:<10}: {int(row['race_id']):>4} races | ROI: {b_roi:>+6.1f}% | Hit: {b_hit:>5.1f}% | Profit: ¥{int(row['profit']):>7,}")
            
        return results_df
