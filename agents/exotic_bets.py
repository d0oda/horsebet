"""
UmaEdge — Exotic Bet Constructor.

Optimises multi-leg bet ticket construction (三連複, 三連単, ワイド)
using integer linear programming (PuLP). Finds the set of tickets
that maximises expected profit under a budget constraint.

Usage:
    # Trio (三連複) tickets for a race with ¥5,000 budget
    python -m agents.exotic_bets --race-id 1 --budget 5000 --type trio

    # Trifecta (三連単) tickets
    python -m agents.exotic_bets --race-id 1 --budget 5000 --type trifecta

    # Wide (ワイド) tickets
    python -m agents.exotic_bets --race-id 1 --budget 3000 --type wide
"""

import argparse
import json
import logging
from dataclasses import dataclass
from enum import Enum
from itertools import combinations, permutations
from typing import Optional

import numpy as np
from sqlalchemy import text

from scraper.db import get_session
from models.simulate_race import load_race_entries
from models.pace_sim import PaceSimulator, classify_running_style, STYLE_FRONT

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("exotic_bets")

UNIT_COST = 100  # ¥100 per ticket in JRA


# ---------------------------------------------------------------------------
# Bet Types
# ---------------------------------------------------------------------------

class ExoticBetType(Enum):
    TRIO = "trio"           # 三連複 — top 3 in any order
    TRIFECTA = "trifecta"   # 三連単 — top 3 in exact order
    WIDE = "wide"           # ワイド — any 2 of top 3
    EXACTA = "exacta"       # 馬単 — 1st and 2nd in exact order
    QUINELLA = "quinella"   # 馬連 — 1st and 2nd in any order


# ---------------------------------------------------------------------------
# Ticket Recommendation
# ---------------------------------------------------------------------------

@dataclass
class TicketRecommendation:
    """A single optimised ticket."""
    combination: str          # e.g. "3-5-7" or "3-7-5"
    horse_names: list[str]
    model_prob: float
    pool_odds: float          # from odds_snapshots or estimated
    expected_payout: float    # prob × odds × 100
    cost: int = UNIT_COST     # ¥100
    ev: float = 0.0           # expected_payout - cost


# ---------------------------------------------------------------------------
# Ticket Constructor
# ---------------------------------------------------------------------------

class TicketConstructor:
    """
    Builds optimal exotic bet tickets using Integer Linear Programming.

    Given model probabilities and pool odds, selects the set of tickets
    that maximises expected profit under a budget constraint.
    """

    def __init__(self, n_simulations: int = 10_000, seed: int = 42):
        self.n_simulations = n_simulations
        self.rng = np.random.default_rng(seed)

    # -------------------------------------------------------------------
    # Probability Matrix
    # -------------------------------------------------------------------

    def build_finish_distribution(
        self,
        race_id: int,
        win_probs: dict[int, float],
        n_sims: int = None,
    ) -> np.ndarray:
        """
        Build a joint finish-position distribution via simulation.

        Given P(win) per horse, simulates races by sampling from
        a Dirichlet distribution anchored on win probabilities.

        Args:
            win_probs: {horse_id: win_probability}
            n_sims: Number of simulations (default: self.n_simulations)

        Returns:
            2D array of shape (n_sims, n_horses) where each row is a
            simulated finish order (0=first, 1=second, ...).
            Horse IDs are in sorted order.
        """
        n_sims = n_sims or self.n_simulations
        horse_ids = sorted(win_probs.keys())
        n_horses = len(horse_ids)

        if n_horses < 2:
            return np.zeros((n_sims, n_horses), dtype=int)

        # 1. Load running styles from db to model pace correlations
        entries = load_race_entries(race_id)
        styles = []
        for horse_num in horse_ids:
            entry = next((e for e in entries if e["post_position"] == horse_num), {})
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
        
        # We need the pace_modifier from PaceSimulator
        sim_engine = PaceSimulator()

        probs = np.array([win_probs[horse_num] for horse_num in horse_ids])
        probs = np.maximum(probs, 1e-6)  # prevent log(0)
        probs = probs / probs.sum()  # normalise
        log_probs = np.log(probs)

        # Pre-calculate expected modifier for each horse to preserve ML marginals
        expected_modifiers = np.zeros(n_horses)
        for j in range(n_horses):
            for k, p_scenario in enumerate(pace_scenarios):
                mod = sim_engine._pace_modifier(styles[j], p_scenario)
                expected_modifiers[j] += pace_probs[k] * mod

        finishes = np.zeros((n_sims, n_horses), dtype=int)
        for i in range(n_sims):
            # Sample pace scenario
            pace = self.rng.choice(pace_scenarios, p=pace_probs)
            
            # Gumbel-max trick perfectly preserves Plackett-Luce probabilities
            gumbels = self.rng.gumbel(loc=0, scale=1, size=n_horses)
            scores = log_probs + gumbels
            
            # Apply structural pace correlation to scores
            # Divide by expected_modifier to center the bias and preserve the baseline ML marginal
            for j in range(n_horses):
                raw_modifier = sim_engine._pace_modifier(styles[j], pace)
                centered_modifier = raw_modifier / expected_modifiers[j]
                scores[j] += np.log(centered_modifier)
                
            # argsort descending → best horse gets position 0 (first)
            order = np.argsort(-scores)
            for pos, horse_idx in enumerate(order):
                finishes[i, horse_idx] = pos

        return finishes

    def compute_combo_probabilities(
        self,
        finishes: np.ndarray,
        horse_ids: list[int],
        bet_type: ExoticBetType,
    ) -> dict[str, float]:
        """
        From the simulated finish matrix, compute P(combination) for each
        possible ticket.

        Returns:
            {combination_string: probability}
        """
        n_sims = finishes.shape[0]
        n_horses = len(horse_ids)
        combo_counts: dict[str, int] = {}

        for sim in range(n_sims):
            # Get finish positions for this simulation
            positions = finishes[sim]  # positions[i] = finish pos of horse i

            if bet_type == ExoticBetType.TRIO:
                # Top 3 in any order
                top3_indices = np.argsort(positions)[:3]
                top3_ids = sorted([horse_ids[i] for i in top3_indices])
                key = "-".join(str(h) for h in top3_ids)
                combo_counts[key] = combo_counts.get(key, 0) + 1

            elif bet_type == ExoticBetType.TRIFECTA:
                # Top 3 in exact order
                top3_indices = np.argsort(positions)[:3]
                top3_ids = [horse_ids[i] for i in top3_indices]
                key = "-".join(str(h) for h in top3_ids)
                combo_counts[key] = combo_counts.get(key, 0) + 1

            elif bet_type == ExoticBetType.WIDE:
                # Any 2 of the top 3 (3 possible combinations per race)
                top3_indices = np.argsort(positions)[:3]
                top3_ids = [horse_ids[i] for i in top3_indices]
                pairs = list(combinations(top3_ids, 2))
                for p in pairs:
                    key = "-".join(str(h) for h in sorted(p))
                    combo_counts[key] = combo_counts.get(key, 0) + 1
                    
            elif bet_type == ExoticBetType.EXACTA:
                # Top 2 in exact order
                top2_indices = np.argsort(positions)[:2]
                top2_ids = [horse_ids[i] for i in top2_indices]
                key = "-".join(str(h) for h in top2_ids)
                combo_counts[key] = combo_counts.get(key, 0) + 1
                
            elif bet_type == ExoticBetType.QUINELLA:
                # Top 2 in any order
                top2_indices = np.argsort(positions)[:2]
                top2_ids = sorted([horse_ids[i] for i in top2_indices])
                key = "-".join(str(h) for h in top2_ids)
                combo_counts[key] = combo_counts.get(key, 0) + 1

        # Normalise to probabilities
        return {k: v / n_sims for k, v in combo_counts.items()}

    # -------------------------------------------------------------------
    # Pool Odds
    # -------------------------------------------------------------------

    def get_pool_odds(
        self,
        race_id: int,
        bet_type: ExoticBetType,
    ) -> dict[str, float]:
        """
        Load exotic odds from the odds_snapshots table.
        Falls back to estimated odds if no snapshot exists.
        """
        type_map = {
            ExoticBetType.TRIO: "trio",
            ExoticBetType.TRIFECTA: "trifecta",
            ExoticBetType.WIDE: "wide",
            ExoticBetType.EXACTA: "exacta",
            ExoticBetType.QUINELLA: "quinella",
        }

        try:
            with get_session() as session:
                rows = session.execute(text("""
                    SELECT combination, odds_value
                    FROM odds_snapshots
                    WHERE race_id = :race_id
                      AND bet_type = :bt
                    ORDER BY captured_at DESC
                """), {
                    "race_id": race_id,
                    "bt": type_map[bet_type],
                }).fetchall()

            if rows:
                # Keep only latest odds per combination
                odds_map = {}
                for combo, odds_val in rows:
                    if combo not in odds_map:
                        odds_map[combo] = float(odds_val)
                return odds_map

        except Exception as e:
            log.warning(f"Could not load pool odds: {e}")

        return {}

    def estimate_odds(
        self,
        combo_probs: dict[str, float],
        take_rate: float = 0.25,
        max_odds: float = 30000.0,
    ) -> dict[str, float]:
        """
        Estimate odds from probabilities if no market odds available.
        Uses simple 1/prob with take-rate adjustment, capped at max_odds to prevent
        infinite liquidity assumptions on rare combinations.
        """
        estimated = {}
        for combo, prob in combo_probs.items():
            if prob > 0:
                fair_odds = 1.0 / prob
                pool_odds = fair_odds * (1 - take_rate)
                estimated[combo] = min(pool_odds, max_odds)
            else:
                estimated[combo] = 0
        return estimated

    # -------------------------------------------------------------------
    # Optimisation
    # -------------------------------------------------------------------

    def find_optimal_tickets(
        self,
        combo_probs: dict[str, float],
        combo_odds: dict[str, float],
        budget: int,
        min_ev: float = 0.0,
    ) -> list[TicketRecommendation]:
        """
        Solve ILP to find the optimal set of tickets that maximises
        expected profit under a budget constraint.

        Decision variable x_i ∈ {0, 1} for each combination i.
        Objective: maximise Σ (model_prob_i × odds_i × 100 - 100) × x_i
        Constraint: Σ 100 × x_i ≤ budget
        """
        import pulp

        # Filter to combos that have both probs and odds
        valid_combos = [
            c for c in combo_probs
            if c in combo_odds and combo_odds[c] > 0 and combo_probs[c] > 0
        ]

        if not valid_combos:
            log.warning("No valid combinations found for optimisation")
            return []

        # Calculate expected value for each combo
        ev_map = {}
        for combo in valid_combos:
            prob = combo_probs[combo]
            odds = combo_odds[combo]
            expected_payout = prob * odds * UNIT_COST
            ev_map[combo] = expected_payout - UNIT_COST

        # Filter to positive EV only (or above threshold)
        candidates = [c for c in valid_combos if ev_map[c] >= min_ev]

        if not candidates:
            log.info("No positive-EV combinations found")
            return []

        max_tickets = budget // UNIT_COST

        # ILP formulation
        prob = pulp.LpProblem("exotic_bets", pulp.LpMaximize)

        # Decision variables: buy or not
        x = {c: pulp.LpVariable(f"x_{c}", cat="Binary") for c in candidates}

        # Objective: maximise expected profit
        prob += pulp.lpSum(ev_map[c] * x[c] for c in candidates)

        # Budget constraint
        prob += pulp.lpSum(UNIT_COST * x[c] for c in candidates) <= budget

        # Solve
        prob.solve(pulp.PULP_CBC_CMD(msg=0))

        if prob.status != pulp.constants.LpStatusOptimal:
            log.warning(f"Optimisation not optimal: status={prob.status}")
            # Fall back to greedy selection
            return self._greedy_selection(candidates, ev_map, combo_probs, combo_odds, budget)

        # Extract selected tickets
        tickets = []
        for combo in candidates:
            if x[combo].value() and x[combo].value() > 0.5:
                p = combo_probs[combo]
                o = combo_odds[combo]
                tickets.append(TicketRecommendation(
                    combination=combo,
                    horse_names=[],  # filled later
                    model_prob=p,
                    pool_odds=o,
                    expected_payout=p * o * UNIT_COST,
                    ev=ev_map[combo],
                ))

        # Sort by EV descending
        tickets.sort(key=lambda t: -t.ev)
        return tickets

    def _greedy_selection(
        self,
        candidates: list[str],
        ev_map: dict[str, float],
        combo_probs: dict[str, float],
        combo_odds: dict[str, float],
        budget: int,
    ) -> list[TicketRecommendation]:
        """Fallback greedy selection: pick by EV until budget exhausted."""
        sorted_by_ev = sorted(candidates, key=lambda c: -ev_map[c])
        tickets = []
        spent = 0

        for combo in sorted_by_ev:
            if spent + UNIT_COST > budget:
                break
            if ev_map[combo] <= 0:
                continue
            p = combo_probs[combo]
            o = combo_odds[combo]
            tickets.append(TicketRecommendation(
                combination=combo,
                horse_names=[],
                model_prob=p,
                pool_odds=o,
                expected_payout=p * o * UNIT_COST,
                ev=ev_map[combo],
            ))
            spent += UNIT_COST

        return tickets

    # -------------------------------------------------------------------
    # High-Level Pipeline
    # -------------------------------------------------------------------

    def prepare_race_data(self, race_id: int) -> tuple[dict, dict, list]:
        """
        Step 1 & 2: Load win probabilities and simulate finish distribution.
        Run this once per race.
        """
        log.info(f"Loading predictions for race {race_id}...")
        win_probs = {}
        horse_names = {}

        # We must use RAW model_win_prob for the Plackett-Luce simulation.
        from models.predict import predict_and_store
        pred_df = predict_and_store(race_id, store_to_db=False)
        
        if pred_df.empty:
            log.error("Could not generate predictions")
            return {}, {}, []

        for _, r in pred_df.iterrows():
            with get_session() as session:
                entry = session.execute(text(
                    "SELECT post_position FROM entries WHERE id = :eid"
                ), {"eid": r["entry_id"]}).fetchone()
                horse = session.execute(text(
                    "SELECT h.name_jp FROM horses h JOIN entries e ON e.horse_id = h.id WHERE e.id = :eid"
                ), {"eid": r["entry_id"]}).fetchone() if entry else None

            horse_num = entry[0] if entry else int(str(r["entry_id"])[-2:]) # Fallback to last 2 digits of entry_id
            win_probs[horse_num] = r["model_win_prob"]
            horse_names[horse_num] = horse[0] if horse else str(horse_num)

        if len(win_probs) < 3:
            log.error(f"Need at least 3 horses, got {len(win_probs)}")
            return {}, {}, []

        log.info(f"Simulating {self.n_simulations:,} pace-correlated race finishes...")
        finishes = self.build_finish_distribution(race_id, win_probs)
        return win_probs, horse_names, finishes

    def evaluate_pool_tickets(
        self,
        race_id: int,
        bet_type: ExoticBetType,
        budget: int,
        win_probs: dict,
        horse_names: dict,
        finishes: list,
    ) -> list[TicketRecommendation]:
        """
        Step 3-5: Calculate combinations, odds, and optimize for a specific pool.
        """
        horse_ids = sorted(win_probs.keys())
        
        # 3. Compute combination probabilities
        combo_probs = self.compute_combo_probabilities(finishes, horse_ids, bet_type)

        # 4. Load or estimate pool odds
        pool_odds = self.get_pool_odds(race_id, bet_type)
        if not pool_odds:
            pool_odds = self.estimate_odds(combo_probs)

        # 5. Optimise
        tickets = self.find_optimal_tickets(combo_probs, pool_odds, budget)

        # Fill in horse names
        for t in tickets:
            ids = t.combination.split("-")
            t.horse_names = [horse_names.get(int(hid), str(hid)) for hid in ids]

        return tickets

    def construct_tickets(
        self,
        race_id: int,
        bet_type: ExoticBetType,
        budget: int = 5000,
        store_to_db: bool = True,
        model_version: str = "latest",
    ) -> list[TicketRecommendation]:
        """End-to-end ticket construction for a single pool."""
        win_probs, horse_names, finishes = self.prepare_race_data(race_id)
        if len(finishes) == 0:
            return []
            
        log.info(f"Evaluating {bet_type.value} pool...")
        tickets = self.evaluate_pool_tickets(race_id, bet_type, budget, win_probs, horse_names, finishes)
        
        # 6. Store to DB
        if store_to_db and tickets:
            self._store_tickets(race_id, bet_type, budget, tickets, model_version)

        return tickets

    def recommend_best_pool(
        self, 
        race_id: int, 
        budget: int,
        store_to_db: bool = True,
        model_version: str = "latest",
    ) -> tuple[ExoticBetType, list[TicketRecommendation]]:
        """Evaluate all pools and return the one with the highest Total EV."""
        win_probs, horse_names, finishes = self.prepare_race_data(race_id)
        if len(finishes) == 0:
            return ExoticBetType.TRIO, []

        log.info(f"Evaluating ALL exotic pools to find highest Total EV (Budget: ¥{budget:,})...")
        best_type = ExoticBetType.TRIO
        best_ev = -float('inf')
        best_tickets = []
        
        pool_types = [
            ExoticBetType.TRIO, 
            ExoticBetType.TRIFECTA, 
            ExoticBetType.EXACTA, 
            ExoticBetType.QUINELLA, 
            ExoticBetType.WIDE
        ]
        
        results = []
        for bt in pool_types:
            tickets = self.evaluate_pool_tickets(race_id, bt, budget, win_probs, horse_names, finishes)
            total_ev = sum(t.ev for t in tickets)
            total_cost = sum(t.cost for t in tickets)
            roi = (total_ev / total_cost * 100) if total_cost > 0 else 0
            
            results.append((bt, tickets, total_ev, total_cost, roi))
            
            if total_ev > best_ev:
                best_ev = total_ev
                best_type = bt
                best_tickets = tickets
                
        # Print summary
        print(f"\n📊 Pool Evaluation Summary - Race #{race_id}")
        print("-" * 65)
        for bt, tkts, ev, cost, roi in sorted(results, key=lambda x: x[2], reverse=True):
            type_name = bt.value.capitalize()
            # If cost is 0, print "No +EV tickets"
            if cost == 0:
                print(f" {type_name:<10}: {'No +EV tickets found':<48}")
            else:
                marker = "⭐ BEST " if bt == best_type else "        "
                print(f"{marker}{type_name:<10}: {len(tkts):>2} tkts | Cost: ¥{cost:>5,} | EV: {ev:>+7,.0f} | ROI: {roi:>+5.1f}%")
        print("-" * 65)
        
        if store_to_db and best_tickets:
            self._store_tickets(race_id, best_type, budget, best_tickets, model_version)
            
        return best_type, best_tickets

    def _store_tickets(
        self,
        race_id: int,
        bet_type: ExoticBetType,
        budget: int,
        tickets: list[TicketRecommendation],
        model_version: str,
    ):
        """Store ticket recommendations to the exotic_tickets table."""
        combos_json = [
            {
                "combination": t.combination,
                "horse_names": t.horse_names,
                "model_prob": round(t.model_prob, 6),
                "odds": round(t.pool_odds, 1),
                "cost": t.cost,
                "ev": round(t.ev, 2),
            }
            for t in tickets
        ]

        total_cost = sum(t.cost for t in tickets)
        total_ev = sum(t.ev for t in tickets)
        expected_roi = (total_ev / total_cost * 100) if total_cost > 0 else 0

        with get_session() as session:
            session.execute(text("""
                INSERT INTO exotic_tickets
                    (race_id, bet_type, budget, combinations, total_cost,
                     expected_profit, expected_roi, model_version)
                VALUES
                    (:race_id, :bt, :budget, :combos, :cost,
                     :profit, :roi, :version)
            """), {
                "race_id": race_id,
                "bt": bet_type.value,
                "budget": budget,
                "combos": json.dumps(combos_json),
                "cost": total_cost,
                "profit": total_ev,
                "roi": expected_roi,
                "version": model_version,
            })

        log.info(f"💾 Stored {len(tickets)} tickets (cost: ¥{total_cost:,}, expected ROI: {expected_roi:+.1f}%)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="UmaEdge — Exotic Bet Constructor")
    parser.add_argument("--race-id", type=int, required=True, help="Database race ID")
    parser.add_argument(
        "--type",
        type=str,
        choices=["trio", "trifecta", "wide", "exacta", "quinella", "best"],
        default="best",
        help="Exotic bet type (or 'best' to evaluate all and pick the highest EV)",
    )
    parser.add_argument("--budget", type=int, default=5000, help="Budget in yen (default: 5000)")
    parser.add_argument("--sims", type=int, default=10000, help="Simulations (default: 10000)")
    parser.add_argument("--no-store", action="store_true", help="Don't store to DB")
    args = parser.parse_args()

    constructor = TicketConstructor(n_simulations=args.sims)
    store = not args.no_store

    if args.type == "best":
        bet_type, tickets = constructor.recommend_best_pool(
            race_id=args.race_id,
            budget=args.budget,
            store_to_db=store,
            model_version="latest", # Usually queried dynamically inside predict_and_store but we'll use latest here
        )
    else:
        bet_type = ExoticBetType(args.type)
        tickets = constructor.construct_tickets(
            race_id=args.race_id,
            bet_type=bet_type,
            budget=args.budget,
            store_to_db=store,
        )

    if not tickets:
        print(f"No profitable tickets found for race #{args.race_id}")
        return

    type_names = {
        ExoticBetType.TRIO: "三連複 (Trio)",
        ExoticBetType.TRIFECTA: "三連単 (Trifecta)",
        ExoticBetType.WIDE: "ワイド (Wide)",
        ExoticBetType.EXACTA: "馬単 (Exacta)",
        ExoticBetType.QUINELLA: "馬連 (Quinella)",
    }

    print(f"\n🎰 {type_names[bet_type]} — Race #{args.race_id}")
    print(f"Budget: ¥{args.budget:,} | Tickets: {len(tickets)}")
    print("-" * 75)
    print(f"{'#':>3}  {'Combination':<20} {'Prob':>6}  {'Odds':>7}  {'E[Pay]':>7}  {'EV':>7}")
    print("-" * 75)

    total_cost = 0
    total_ev = 0
    for i, t in enumerate(tickets, 1):
        names = " / ".join(n[:6] for n in t.horse_names)
        combo_display = f"{t.combination} ({names})"
        print(
            f"{i:>3}  "
            f"{combo_display:<20} "
            f"{t.model_prob:>5.2%}  "
            f"{t.pool_odds:>7.1f}  "
            f"¥{t.expected_payout:>6,.0f}  "
            f"¥{t.ev:>+6,.0f}"
        )
        total_cost += t.cost
        total_ev += t.ev

    print("-" * 75)
    roi = (total_ev / total_cost * 100) if total_cost > 0 else 0
    print(f"     Total: ¥{total_cost:,} cost | ¥{total_ev:+,.0f} expected profit | {roi:+.1f}% expected ROI")


if __name__ == "__main__":
    main()
