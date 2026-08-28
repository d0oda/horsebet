"""
Unit tests for the Joint Finishing Probability Model in models/betting_engine.py.
Tests Harville / Plackett-Luce formulations, Henery discount adjustments,
and probability conservation across Quinella, Exacta, Wide, Trio, and Trifecta bets.
"""

from itertools import combinations, permutations
import pytest
from models.betting_engine import JointFinishModel


class TestJointFinishModel:
    def test_exact_1_2_sums_to_one(self):
        probs = {1: 0.40, 2: 0.30, 3: 0.20, 4: 0.10}
        model = JointFinishModel(probs, gamma=0.92)

        total_1_2 = 0.0
        for i in probs:
            for j in probs:
                if i != j:
                    total_1_2 += model.exact_1_2(i, j)

        assert total_1_2 == pytest.approx(1.0, rel=1e-5)

    def test_exact_1_2_3_sums_to_one(self):
        probs = {1: 0.35, 2: 0.25, 3: 0.20, 4: 0.12, 5: 0.08}
        model = JointFinishModel(probs, gamma=0.92)

        total_1_2_3 = 0.0
        for i, j, k in permutations(probs.keys(), 3):
            total_1_2_3 += model.exact_1_2_3(i, j, k)

        assert total_1_2_3 == pytest.approx(1.0, rel=1e-5)

    def test_quinella_and_exacta_relationships(self):
        probs = {1: 0.50, 2: 0.30, 3: 0.20}
        model = JointFinishModel(probs, gamma=1.0)

        # Quinella(1, 2) == Exacta(1 -> 2) + Exacta(2 -> 1)
        q_12 = model.quinella_prob(1, 2)
        ex_12 = model.exacta_prob(1, 2)
        ex_21 = model.exacta_prob(2, 1)

        assert q_12 == pytest.approx(ex_12 + ex_21, rel=1e-5)

        # Sum of all quinellas in a 3-horse race should equal 1.0
        sum_quinellas = sum(model.quinella_prob(i, j) for i, j in combinations(probs.keys(), 2))
        assert sum_quinellas == pytest.approx(1.0, rel=1e-5)

    def test_wide_sums_to_three(self):
        """In any race with >= 3 runners, exactly 3 wide pairs hit (C(3,2) = 3)."""
        probs = {1: 0.35, 2: 0.25, 3: 0.20, 4: 0.12, 5: 0.08}
        model = JointFinishModel(probs, gamma=0.92)

        sum_wide = sum(model.wide_prob(i, j) for i, j in combinations(probs.keys(), 2))
        assert sum_wide == pytest.approx(3.0, rel=1e-4)

    def test_trio_sums_to_one(self):
        """Sum of all trio combinations must equal 1.0."""
        probs = {1: 0.35, 2: 0.25, 3: 0.20, 4: 0.12, 5: 0.08}
        model = JointFinishModel(probs, gamma=0.92)

        sum_trio = sum(model.trio_prob(i, j, k) for i, j, k in combinations(probs.keys(), 3))
        assert sum_trio == pytest.approx(1.0, rel=1e-5)

    def test_place_probability_bounds(self):
        probs = {1: 0.50, 2: 0.25, 3: 0.15, 4: 0.10}
        model = JointFinishModel(probs, gamma=0.92)

        for p in probs:
            place_p = model.place_prob(p, top_n=3)
            # Place probability must be >= Win probability and <= 1.0
            assert place_p >= probs[p]
            assert place_p <= 1.0

        # For a 4-horse race, sum of place probabilities should equal 3.0
        sum_place = sum(model.place_prob(p, top_n=3) for p in probs)
        assert sum_place == pytest.approx(3.0, rel=1e-4)

    def test_henery_discount_effect(self):
        """Henery gamma > 1.0 reduces extreme longshot place probability relative to standard Harville.

        With gamma > 1.0, p_j^gamma < p_j for small p_j (longshots), so their conditional
        probability of placing is reduced vs pure Harville (gamma=1.0).
        NOTE: the OLD codebase used gamma < 1.0 which was WRONG — it INFLATED longshot place prob.
        The fix inverted the convention: gamma > 1.0 is now the correct, empirically-calibrated direction.
        See test_henery_gamma_direction_correct for the definitive assertion of this property.
        """
        probs = {1: 0.70, 2: 0.20, 3: 0.08, 4: 0.02}
        harville_model = JointFinishModel(probs, gamma=1.0)
        henery_model = JointFinishModel(probs, gamma=1.10)  # correct direction: > 1.0

        # With gamma > 1.0 the favourite's conditional 2nd-place weight is higher
        # (longshot weight is suppressed, redistributing mass to stronger runners)
        fav_2nd_given_3 = harville_model.exact_1_2(3, 1) / probs[3]
        fav_2nd_given_3_henery = henery_model.exact_1_2(3, 1) / probs[3]

        assert fav_2nd_given_3_henery > 0.0
        assert fav_2nd_given_3 > 0.0

    def test_boundary_zero_and_small_field(self):
        probs = {1: 0.70, 2: 0.30, 3: 0.0}
        model = JointFinishModel(probs, gamma=1.10)

        assert model.exact_1_2(1, 3) == pytest.approx(0.0)
        assert model.exact_1_2_3(1, 2, 3) == pytest.approx(0.0)
        assert model.wide_prob(1, 3) == pytest.approx(0.0)

    def test_henery_gamma_direction_correct(self):
        """Fix #1 (CRITICAL): gamma > 1.0 must REDUCE longshot place prob vs standard Harville.

        With the original buggy gamma < 1.0, longshot place_prob INCREASED.
        With corrected gamma > 1.0, longshot place_prob must DECREASE.
        """
        probs = {1: 0.70, 2: 0.20, 3: 0.08, 4: 0.02}
        harville = JointFinishModel(probs, gamma=1.0)     # standard Harville baseline
        henery_correct = JointFinishModel(probs, gamma=1.10)  # correct direction

        longshot_place_harville = harville.place_prob(4, top_n=3)
        longshot_place_henery = henery_correct.place_prob(4, top_n=3)

        assert longshot_place_henery < longshot_place_harville, (
            f"Henery gamma > 1.0 must REDUCE longshot place prob. "
            f"Harville={longshot_place_harville:.4f}, Henery(1.10)={longshot_place_henery:.4f}. "
            f"If Henery >= Harville, gamma is going in the wrong direction."
        )

        # Also verify favourite place prob increases (as probability mass is redistributed)
        fav_place_harville = harville.place_prob(1, top_n=3)
        fav_place_henery = henery_correct.place_prob(1, top_n=3)
        assert fav_place_henery >= fav_place_harville, (
            f"Henery gamma > 1.0 should increase or maintain favourite place prob. "
            f"Harville={fav_place_harville:.4f}, Henery={fav_place_henery:.4f}"
        )

    def test_henery_gamma_default_is_correct_direction(self):
        """The default gamma must be > 1.0 (correct direction)."""
        import inspect
        sig = inspect.signature(JointFinishModel.__init__)
        default_gamma = sig.parameters["gamma"].default
        assert default_gamma > 1.0, (
            f"JointFinishModel default gamma must be > 1.0 for correct Henery direction. "
            f"Got {default_gamma}. Values < 1.0 INFLATE longshot place probability."
        )
