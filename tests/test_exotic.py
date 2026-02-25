"""
Unit tests for the Exotic Bet Constructor.
Tests probability matrix generation, combination counting,
ILP optimisation, and budget constraints.
"""

import numpy as np
import pytest

from agents.exotic_bets import (
    ExoticBetType,
    TicketConstructor,
    TicketRecommendation,
    UNIT_COST,
)


# ---------------------------------------------------------------------------
# Finish Distribution
# ---------------------------------------------------------------------------

class TestFinishDistribution:
    def setup_method(self):
        self.tc = TicketConstructor(n_simulations=5000, seed=42)

    def test_shape(self):
        probs = {1: 0.4, 2: 0.3, 3: 0.2, 4: 0.1}
        finishes = self.tc.build_finish_distribution(probs, n_sims=1000)
        assert finishes.shape == (1000, 4)

    def test_valid_positions(self):
        """Each row should be a valid permutation of positions 0..n-1."""
        probs = {1: 0.4, 2: 0.3, 3: 0.2, 4: 0.1}
        finishes = self.tc.build_finish_distribution(probs, n_sims=500)
        for row in finishes:
            assert set(row) == {0, 1, 2, 3}

    def test_favourite_wins_more(self):
        """The horse with highest probability should win most often."""
        probs = {1: 0.6, 2: 0.1, 3: 0.1, 4: 0.1, 5: 0.1}
        finishes = self.tc.build_finish_distribution(probs, n_sims=5000)
        horse_ids = sorted(probs.keys())

        # Horse 1 (idx 0) should finish 1st (position 0) more than any other
        horse1_wins = np.sum(finishes[:, 0] == 0)
        for i in range(1, len(horse_ids)):
            other_wins = np.sum(finishes[:, i] == 0)
            assert horse1_wins > other_wins

    def test_single_horse(self):
        probs = {1: 1.0}
        finishes = self.tc.build_finish_distribution(probs, n_sims=100)
        assert finishes.shape == (100, 1)

    def test_two_horses(self):
        probs = {1: 0.7, 2: 0.3}
        finishes = self.tc.build_finish_distribution(probs, n_sims=1000)
        assert finishes.shape == (1000, 2)
        # Each row should have positions {0, 1}
        for row in finishes:
            assert set(row) == {0, 1}


# ---------------------------------------------------------------------------
# Combination Probabilities
# ---------------------------------------------------------------------------

class TestComboProbabilities:
    def setup_method(self):
        self.tc = TicketConstructor(n_simulations=10000, seed=42)
        self.probs = {1: 0.3, 2: 0.25, 3: 0.2, 4: 0.15, 5: 0.1}
        self.finishes = self.tc.build_finish_distribution(self.probs)
        self.horse_ids = sorted(self.probs.keys())

    def test_trio_probs_sum_to_one(self):
        combo_probs = self.tc.compute_combo_probabilities(
            self.finishes, self.horse_ids, ExoticBetType.TRIO
        )
        total = sum(combo_probs.values())
        assert total == pytest.approx(1.0, abs=0.01)

    def test_trifecta_probs_sum_to_one(self):
        combo_probs = self.tc.compute_combo_probabilities(
            self.finishes, self.horse_ids, ExoticBetType.TRIFECTA
        )
        total = sum(combo_probs.values())
        assert total == pytest.approx(1.0, abs=0.01)

    def test_wide_probs_sum_to_three(self):
        """Each simulation produces 3 wide pairs (C(3,2)=3)."""
        combo_probs = self.tc.compute_combo_probabilities(
            self.finishes, self.horse_ids, ExoticBetType.WIDE
        )
        total = sum(combo_probs.values())
        assert total == pytest.approx(3.0, abs=0.1)

    def test_trio_has_fewer_combos_than_trifecta(self):
        trio = self.tc.compute_combo_probabilities(
            self.finishes, self.horse_ids, ExoticBetType.TRIO
        )
        trifecta = self.tc.compute_combo_probabilities(
            self.finishes, self.horse_ids, ExoticBetType.TRIFECTA
        )
        # Trio = C(5,3) = 10 max, Trifecta = P(5,3) = 60 max
        assert len(trio) <= len(trifecta)

    def test_favourite_trio_most_likely(self):
        """Trio containing top 3 favourites should be the most likely."""
        combo_probs = self.tc.compute_combo_probabilities(
            self.finishes, self.horse_ids, ExoticBetType.TRIO
        )
        # Top 3 by prob: horses 1, 2, 3 → key "1-2-3"
        top3_key = "1-2-3"
        assert top3_key in combo_probs
        assert combo_probs[top3_key] == max(combo_probs.values())


# ---------------------------------------------------------------------------
# Odds Estimation
# ---------------------------------------------------------------------------

class TestOddsEstimation:
    def setup_method(self):
        self.tc = TicketConstructor()

    def test_estimated_odds_positive(self):
        probs = {"1-2-3": 0.10, "1-2-4": 0.05, "2-3-4": 0.02}
        odds = self.tc.estimate_odds(probs)
        for combo, o in odds.items():
            assert o > 0

    def test_take_rate_applied(self):
        probs = {"1-2-3": 0.10}
        fair = self.tc.estimate_odds(probs, take_rate=0.0)
        with_take = self.tc.estimate_odds(probs, take_rate=0.25)
        assert with_take["1-2-3"] < fair["1-2-3"]

    def test_zero_prob_zero_odds(self):
        probs = {"1-2-3": 0.0}
        odds = self.tc.estimate_odds(probs)
        assert odds["1-2-3"] == 0


# ---------------------------------------------------------------------------
# ILP Optimisation
# ---------------------------------------------------------------------------

class TestOptimisation:
    def setup_method(self):
        self.tc = TicketConstructor()

    def test_budget_constraint(self):
        """Total cost should not exceed budget."""
        combo_probs = {
            "1-2-3": 0.10,
            "1-2-4": 0.08,
            "1-3-4": 0.06,
            "2-3-4": 0.04,
        }
        combo_odds = {
            "1-2-3": 12.0,
            "1-2-4": 18.0,
            "1-3-4": 25.0,
            "2-3-4": 40.0,
        }
        tickets = self.tc.find_optimal_tickets(combo_probs, combo_odds, budget=300)
        total_cost = sum(t.cost for t in tickets)
        assert total_cost <= 300

    def test_selects_positive_ev_only(self):
        """Should only select tickets with positive expected value."""
        combo_probs = {
            "1-2-3": 0.10,   # EV = 0.1*12*100 - 100 = +20
            "2-3-4": 0.001,  # EV = 0.001*40*100 - 100 = -96
        }
        combo_odds = {
            "1-2-3": 12.0,
            "2-3-4": 40.0,
        }
        tickets = self.tc.find_optimal_tickets(combo_probs, combo_odds, budget=300)
        for t in tickets:
            assert t.ev > 0

    def test_empty_when_no_positive_ev(self):
        combo_probs = {"1-2-3": 0.001}
        combo_odds = {"1-2-3": 5.0}  # EV = 0.001*5*100 - 100 = -99.5
        tickets = self.tc.find_optimal_tickets(combo_probs, combo_odds, budget=1000)
        assert len(tickets) == 0

    def test_all_tickets_when_budget_allows(self):
        """If budget is large enough, all positive-EV tickets should be selected."""
        combo_probs = {
            "1-2-3": 0.20,  # EV = 0.2*8*100 - 100 = +60
            "1-2-4": 0.15,  # EV = 0.15*10*100 - 100 = +50
        }
        combo_odds = {
            "1-2-3": 8.0,
            "1-2-4": 10.0,
        }
        tickets = self.tc.find_optimal_tickets(combo_probs, combo_odds, budget=10000)
        assert len(tickets) == 2

    def test_ticket_has_correct_attributes(self):
        combo_probs = {"1-2-3": 0.15}
        combo_odds = {"1-2-3": 10.0}
        tickets = self.tc.find_optimal_tickets(combo_probs, combo_odds, budget=1000)
        assert len(tickets) == 1
        t = tickets[0]
        assert t.combination == "1-2-3"
        assert t.model_prob == 0.15
        assert t.pool_odds == 10.0
        assert t.expected_payout == pytest.approx(0.15 * 10.0 * 100, rel=0.01)
        assert t.cost == UNIT_COST

    def test_ev_calculation(self):
        combo_probs = {"1-2-3": 0.15}
        combo_odds = {"1-2-3": 10.0}
        tickets = self.tc.find_optimal_tickets(combo_probs, combo_odds, budget=1000)
        t = tickets[0]
        expected_ev = 0.15 * 10.0 * 100 - 100  # = 50
        assert t.ev == pytest.approx(expected_ev, rel=0.01)


# ---------------------------------------------------------------------------
# Greedy Fallback
# ---------------------------------------------------------------------------

class TestGreedyFallback:
    def setup_method(self):
        self.tc = TicketConstructor()

    def test_greedy_respects_budget(self):
        combo_probs = {f"1-2-{i}": 0.10 for i in range(3, 20)}
        combo_odds = {f"1-2-{i}": 15.0 for i in range(3, 20)}
        ev_map = {k: combo_probs[k] * combo_odds[k] * 100 - 100 for k in combo_probs}
        candidates = list(combo_probs.keys())

        tickets = self.tc._greedy_selection(
            candidates, ev_map, combo_probs, combo_odds, budget=500
        )
        total_cost = sum(t.cost for t in tickets)
        assert total_cost <= 500

    def test_greedy_picks_highest_ev_first(self):
        combo_probs = {"A": 0.20, "B": 0.10}
        combo_odds = {"A": 8.0, "B": 15.0}
        ev_map = {k: combo_probs[k] * combo_odds[k] * 100 - 100 for k in combo_probs}
        candidates = list(combo_probs.keys())

        tickets = self.tc._greedy_selection(
            candidates, ev_map, combo_probs, combo_odds, budget=100
        )
        assert len(tickets) == 1
        # Should pick the one with higher EV
        expected_best = max(ev_map, key=ev_map.get)
        assert tickets[0].combination == expected_best
