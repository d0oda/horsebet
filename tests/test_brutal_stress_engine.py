import math
import pytest
from typing import List, Dict
from models.betting_engine import (
    HorsePricing,
    StakingPlan,
    BetTicket,
    JointFinishModel,
    select_adaptive_strategy,
    construct_staking_plan,
    analyze_race_betting,
    EV_PASS,
    EV_TOP_VALUE,
    EV_SECONDARY_VALUE,
)

class TestBettingEngineBrutalStress:
    """Brutal stress tests covering extreme edge cases in the betting engine."""

    def test_extreme_field_sizes_and_probs(self):
        # 1-horse field
        single_horse = [
            HorsePricing(
                post_position=1,
                horse_name="Solo",
                horse_name_jp="ソロ",
                win_prob=1.0,
                fair_odds=1.0,
                market_odds=1.1,
                ev=0.1,
                verdict="Top Value",
                is_top_value=True,
                is_favorite=True,
            )
        ]
        strategy, reason = select_adaptive_strategy(single_horse)
        assert strategy in ["pure_win", "pass"]
        plan = construct_staking_plan(single_horse, budget=1000, strategy_mode="pure_win")
        assert plan.total_staked <= 1000

        # 18-horse uniform field (1/18 prob each)
        p = 1.0 / 18.0
        uniform_18 = [
            HorsePricing(
                post_position=i,
                horse_name=f"Horse {i}",
                horse_name_jp=f"ホース{i}",
                win_prob=p,
                fair_odds=18.0,
                market_odds=25.0,  # EV = 25 * (1/18) - 1 = +0.388
                ev=0.388,
                verdict="Top Value" if i == 1 else "Value",
                is_top_value=(i == 1),
                is_favorite=(i == 1),
            )
            for i in range(1, 19)
        ]
        strat, reason = select_adaptive_strategy(uniform_18)
        assert strat in ["dutching", "pure_win", "hybrid"]
        for mode in ["auto", "pure_win", "hybrid", "dutching"]:
            plan = construct_staking_plan(uniform_18, budget=1000, strategy_mode=mode)
            assert plan.total_staked <= 1000
            assert plan.total_staked >= 0
            # Ensure tickets are all multiples of 100 yen (JRA standard)
            for ticket in plan.tickets:
                assert ticket.stake % 100 == 0
                assert ticket.stake > 0

    def test_extreme_odds_and_zero_probs(self):
        # Extreme longshot with microscopic win prob (should NOT divide by zero or overflow)
        extreme_pricing = [
            HorsePricing(
                post_position=1,
                horse_name="SuperFav",
                horse_name_jp="スーパー本命",
                win_prob=0.999,
                fair_odds=1.001,
                market_odds=1.01,
                ev=-0.01,
                verdict="Fair",
                is_favorite=True,
                is_top_value=False,
            ),
            HorsePricing(
                post_position=2,
                horse_name="Ghost",
                horse_name_jp="ゴースト",
                win_prob=0.000001,
                fair_odds=1000000.0,
                market_odds=9999.0,
                ev=-0.99,
                verdict="Too Short",
                is_favorite=False,
                is_top_value=False,
            ),
            HorsePricing(
                post_position=3,
                horse_name="ZeroProb",
                horse_name_jp="ゼロ",
                win_prob=0.0,
                fair_odds=9999.0,
                market_odds=None,
                ev=None,
                verdict="Too Short",
                is_favorite=False,
                is_top_value=False,
            ),
        ]
        strat, reason = select_adaptive_strategy(extreme_pricing)
        assert strat == "pass"

    def test_joint_finish_exotic_permutations_edge_cases(self):
        # 3 horses where one has 0 probability
        probs = {1: 0.6, 2: 0.4, 3: 0.0}
        model = JointFinishModel(probs)
        assert model.exact_1_2(1, 2) > 0.0
        assert model.exact_1_2(1, 3) == 0.0
        assert model.exact_1_2(3, 1) == 0.0
        assert model.quinella_prob(1, 2) > 0.0
        assert model.quinella_prob(1, 3) == 0.0
        assert model.trio_prob(1, 2, 3) == 0.0

        # Empty dictionary handling
        m_empty = JointFinishModel({})
        assert m_empty.probs == {}
        assert m_empty.exact_1_2(1, 2) == 0.0

    def test_dutching_rounding_odd_budgets(self):
        # Odd budgets like 100, 200, 300, 400, 500, 700 yen across 2 horses
        horses = [
            HorsePricing(
                post_position=1,
                horse_name="H1",
                horse_name_jp="H1",
                win_prob=0.25,
                fair_odds=4.0,
                market_odds=6.0,
                ev=0.5,
                verdict="Top Value",
                is_top_value=True,
                is_favorite=False,
            ),
            HorsePricing(
                post_position=2,
                horse_name="H2",
                horse_name_jp="H2",
                win_prob=0.20,
                fair_odds=5.0,
                market_odds=8.0,
                ev=0.6,
                verdict="Value",
                is_top_value=False,
                is_favorite=False,
            ),
            HorsePricing(
                post_position=3,
                horse_name="H3",
                horse_name_jp="H3",
                win_prob=0.10,
                fair_odds=10.0,
                market_odds=3.0,
                ev=-0.7,
                verdict="Too Short",
                is_top_value=False,
                is_favorite=True,
            ),
        ]
        for b in [100, 200, 300, 400, 500, 700, 1000, 2500, 10000]:
            plan = construct_staking_plan(horses, budget=b, strategy_mode="dutching")
            assert plan.total_staked <= b
            for t in plan.tickets:
                assert t.stake % 100 == 0
                assert t.stake >= 100
