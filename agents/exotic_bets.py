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

        # Use win probs as concentration parameters for Dirichlet
        probs = np.array([win_probs[hid] for hid in horse_ids])
        probs = np.maximum(probs, 0.001)  # prevent zeros
        probs = probs / probs.sum()  # normalise

        # Scale to make Dirichlet sharper (respects relative probs)
        alpha = probs * n_horses * 2

        finishes = np.zeros((n_sims, n_horses), dtype=int)
        for i in range(n_sims):
            # Sample "ability scores" from Dirichlet, then rank
            scores = self.rng.dirichlet(alpha)
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
                # Any 2 of top 3
                top3_indices = np.argsort(positions)[:3]
                top3_ids = sorted([horse_ids[i] for i in top3_indices])
                for pair in combinations(top3_ids, 2):
                    key = "-".join(str(h) for h in pair)
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
    ) -> dict[str, float]:
        """
        Estimate odds from probabilities if no market odds available.
        Uses simple 1/prob with take-rate adjustment.
        """
        estimated = {}
        for combo, prob in combo_probs.items():
            if prob > 0:
                fair_odds = 1.0 / prob
                estimated[combo] = fair_odds * (1 - take_rate)
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

    def construct_tickets(
        self,
        race_id: int,
        bet_type: ExoticBetType,
        budget: int = 5000,
        store_to_db: bool = True,
        model_version: str = "latest",
    ) -> list[TicketRecommendation]:
        """
        End-to-end ticket construction for a race.

        1. Load predictions (win probabilities)
        2. Simulate joint finish distribution
        3. Compute combination probabilities
        4. Load or estimate pool odds
        5. Optimise ticket selection
        6. Store to DB
        """
        # 1. Load win probabilities
        log.info(f"Loading predictions for race {race_id}...")
        win_probs = {}
        horse_names = {}

        with get_session() as session:
            rows = session.execute(text("""
                SELECT p.entry_id, e.horse_id, h.name_jp, p.win_prob
                FROM predictions p
                JOIN entries e ON e.id = p.entry_id
                JOIN horses h ON h.id = e.horse_id
                WHERE p.race_id = :race_id
                ORDER BY p.win_prob DESC
            """), {"race_id": race_id}).fetchall()

        if not rows:
            # Fallback: run predictions
            log.info("No stored predictions — running prediction pipeline...")
            from models.predict import predict_and_store
            pred_df = predict_and_store(race_id, store_to_db=True)
            if pred_df.empty:
                log.error("Could not generate predictions")
                return []

            for _, r in pred_df.iterrows():
                with get_session() as session:
                    entry = session.execute(text(
                        "SELECT horse_id FROM entries WHERE id = :eid"
                    ), {"eid": r["entry_id"]}).fetchone()
                    horse = session.execute(text(
                        "SELECT name_jp FROM horses WHERE id = :hid"
                    ), {"hid": entry[0]}).fetchone() if entry else None

                hid = entry[0] if entry else r["entry_id"]
                win_probs[hid] = r["combined_win_prob"]
                horse_names[hid] = horse[0] if horse else str(hid)
        else:
            for row in rows:
                win_probs[row[1]] = float(row[3])
                horse_names[row[1]] = row[2] or str(row[1])

        if len(win_probs) < 3:
            log.error(f"Need at least 3 horses, got {len(win_probs)}")
            return []

        # 2. Simulate finish distribution
        log.info(f"Simulating {self.n_simulations:,} race finishes...")
        finishes = self.build_finish_distribution(win_probs)
        horse_ids = sorted(win_probs.keys())

        # 3. Compute combination probabilities
        combo_probs = self.compute_combo_probabilities(finishes, horse_ids, bet_type)
        log.info(f"Generated {len(combo_probs)} unique combinations")

        # 4. Load or estimate pool odds
        pool_odds = self.get_pool_odds(race_id, bet_type)
        if not pool_odds:
            log.info("No pool odds available — estimating from probabilities")
            pool_odds = self.estimate_odds(combo_probs)

        # 5. Optimise
        log.info(f"Optimising ticket selection (budget: ¥{budget:,})...")
        tickets = self.find_optimal_tickets(combo_probs, pool_odds, budget)

        # Fill in horse names
        for t in tickets:
            ids = t.combination.split("-")
            t.horse_names = [horse_names.get(int(hid), str(hid)) for hid in ids]

        # 6. Store to DB
        if store_to_db and tickets:
            self._store_tickets(race_id, bet_type, budget, tickets, model_version)

        return tickets

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
    parser.add_argument("--type", choices=["trio", "trifecta", "wide"], default="trio",
                        help="Bet type (default: trio)")
    parser.add_argument("--budget", type=int, default=5000, help="Budget in yen (default: 5000)")
    parser.add_argument("--sims", type=int, default=10000, help="Simulations (default: 10000)")
    parser.add_argument("--no-store", action="store_true", help="Don't store to DB")
    args = parser.parse_args()

    bet_type = ExoticBetType(args.type)
    constructor = TicketConstructor(n_simulations=args.sims)

    tickets = constructor.construct_tickets(
        race_id=args.race_id,
        bet_type=bet_type,
        budget=args.budget,
        store_to_db=not args.no_store,
    )

    if not tickets:
        print(f"No profitable tickets found for race #{args.race_id}")
        return

    type_names = {
        ExoticBetType.TRIO: "三連複 (Trio)",
        ExoticBetType.TRIFECTA: "三連単 (Trifecta)",
        ExoticBetType.WIDE: "ワイド (Wide)",
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
