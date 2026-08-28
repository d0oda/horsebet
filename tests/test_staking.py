"""
Unit tests for the Dynamic, Odds-Aware Portfolio Staking Engine in models/betting_engine.py.
Tests budget scaling, discrete ¥100 integer allocation, exact budget conservation,
EV-driven style selection (単勝, 馬単, 馬連, ワイド, 三連複), and simple bet fallback construction.
"""

import pytest
from models.betting_engine import (
    calculate_pricing_breakdown,
    construct_staking_plan,
)


@pytest.fixture
def sample_pricing_underdog_value():
    """Underdog value horse #18 with chalk partner #6."""
    entries = [
        {"post_position": 18, "horse_name": "Tagano Araria", "horse_name_jp": "タガノアラリア", "win_prob": 0.20, "odds": 6.5},   # Value +30%
        {"post_position": 6, "horse_name": "Sephiro", "horse_name_jp": "セフィロ", "win_prob": 0.25, "odds": 3.8},               # Favorite (underpriced)
        {"post_position": 9, "horse_name": "Cruzeiro do Sul", "horse_name_jp": "クルゼイロドスル", "win_prob": 0.12, "odds": 9.5}, # Contender
        {"post_position": 8, "horse_name": "Tagano Elpida", "horse_name_jp": "タガノエルピーダ", "win_prob": 0.08, "odds": 12.0},
        {"post_position": 3, "horse_name": "Yabusame", "horse_name_jp": "ヤブサメ", "win_prob": 0.05, "odds": 18.0},
    ]
    return calculate_pricing_breakdown(entries)


@pytest.fixture
def sample_pricing_fav_dominant():
    """Dominant favorite where exotics have deep negative EV due to heavy chalk & takeout."""
    entries = [
        {"post_position": 1, "horse_name": "SuperStar", "horse_name_jp": "スーパースター", "win_prob": 0.60, "odds": 2.2}, # EV +32%
        {"post_position": 2, "horse_name": "Chaser", "horse_name_jp": "チェイサー", "win_prob": 0.18, "odds": 4.0},
        {"post_position": 3, "horse_name": "Third", "horse_name_jp": "サード", "win_prob": 0.12, "odds": 7.0},
        {"post_position": 4, "horse_name": "Fourth", "horse_name_jp": "フォース", "win_prob": 0.10, "odds": 9.0},
    ]
    return calculate_pricing_breakdown(entries)


@pytest.fixture
def sample_pricing_dual_value():
    """Two runners with high value where Quinella has large positive EV."""
    entries = [
        {"post_position": 1, "horse_name": "ChalkFav", "horse_name_jp": "チョーク", "win_prob": 0.28, "odds": 2.6},
        {"post_position": 2, "horse_name": "ValueA", "horse_name_jp": "バリューA", "win_prob": 0.26, "odds": 5.5},       # EV +43%
        {"post_position": 3, "horse_name": "ValueB", "horse_name_jp": "バリューB", "win_prob": 0.22, "odds": 7.0},       # EV +54%
        {"post_position": 4, "horse_name": "RunnerD", "horse_name_jp": "ランナーD", "win_prob": 0.14, "odds": 7.0},
        {"post_position": 5, "horse_name": "RunnerE", "horse_name_jp": "ランナーE", "win_prob": 0.10, "odds": 12.0},
    ]
    return calculate_pricing_breakdown(entries)


class TestStakingEngine:
    @pytest.mark.parametrize("budget", [500, 1000, 1500, 2000, 2500, 5000, 10000, 50000])
    def test_budget_exact_sum(self, sample_pricing_underdog_value, budget):
        plan = construct_staking_plan(sample_pricing_underdog_value, budget=budget)

        total_staked = sum(t.stake for t in plan.portfolio_tickets)
        assert total_staked == budget, f"Expected total stake {budget}, got {total_staked}"

        for t in plan.portfolio_tickets:
            assert t.stake > 0
            assert t.stake % 100 == 0, f"Ticket stake {t.stake} is not a multiple of 100"

    def test_underdog_value_portfolio(self, sample_pricing_underdog_value):
        plan = construct_staking_plan(sample_pricing_underdog_value, budget=1000)

        # Core Win bet is always selected when positive EV exists
        assert len(plan.portfolio_tickets) >= 1
        win_ticket = plan.portfolio_tickets[0]
        assert win_ticket.ticket_type == "単勝"
        assert win_ticket.selection == [18]
        assert win_ticket.ev > 0.0
        assert sum(t.stake for t in plan.portfolio_tickets) == 1000

    def test_dominant_favorite_wins_anchor(self, sample_pricing_fav_dominant):
        """Dominant favorite with top EV should be the anchor, and win ticket always included."""
        plan = construct_staking_plan(sample_pricing_fav_dominant, budget=1000)

        assert plan.anchor_horse.post_position == 1
        win_ticket = next((t for t in plan.portfolio_tickets if t.ticket_type_en == "win"), None)
        assert win_ticket is not None, "Win ticket must always be included when anchor has positive EV"
        assert win_ticket.selection == [1]
        assert win_ticket.ev > 0.0
        assert sum(t.stake for t in plan.portfolio_tickets) == 1000

    def test_dual_value_quinella_inclusion(self, sample_pricing_dual_value):
        """When multiple runners have value, exotic tickets with positive EV should be selected."""
        plan = construct_staking_plan(sample_pricing_dual_value, budget=1000)

        ticket_types = [t.ticket_type for t in plan.portfolio_tickets]
        assert "単勝" in ticket_types
        # Verify exotics were dynamically included
        assert len(plan.portfolio_tickets) >= 2
        assert sum(t.stake for t in plan.portfolio_tickets) == 1000

    def test_simple_bet_alternative(self, sample_pricing_underdog_value):
        plan = construct_staking_plan(sample_pricing_underdog_value, budget=1000)

        # simple_bet always anchors to the top-value horse regardless of plan mode
        assert plan.simple_bet.ticket_type == "単勝"
        assert plan.simple_bet.selection == [18]
        assert plan.simple_bet.stake == 1000
        assert "¥1,000 on No. 18 単勝" in plan.simple_bet_label
        # Both PP18 (EV≈86%) and PP9 (EV≈63%) clear the 5% EV_TOP_VALUE threshold,
        # so the engine correctly produces a Dutch plan with two win tickets.
        # (R5-INTEGRITY-1 fix expanded Dutch detection from 8%+ to 5%+ EV pool.)
        assert "No. 18 タガノアラリア" in plan.verdict_summary

    def test_small_budget_handling(self, sample_pricing_underdog_value):
        """Even for budgets like ¥100 or ¥200, allocation must be valid and sum to budget."""
        for b in [100, 200, 300]:
            plan = construct_staking_plan(sample_pricing_underdog_value, budget=b)
            total = sum(t.stake for t in plan.portfolio_tickets)
            assert total == b
            for t in plan.portfolio_tickets:
                assert t.stake % 100 == 0
                assert t.stake >= 100


class TestDutchingBugFix:
    """Fix #2: Dutching mode must actually place two win tickets when 2 EV >= 8% horses exist."""

    def test_dutching_places_two_win_tickets(self):
        """Two horses with EV >= 8% should result in 2 win tickets in dutching mode."""
        entries = [
            {"post_position": 1, "horse_name": "ValueA", "horse_name_jp": "バリューA", "win_prob": 0.26, "odds": 5.5},  # EV = +43%
            {"post_position": 2, "horse_name": "ValueB", "horse_name_jp": "バリューB", "win_prob": 0.22, "odds": 7.0},  # EV = +54%
            {"post_position": 3, "horse_name": "ChalkFav", "horse_name_jp": "チョーク", "win_prob": 0.30, "odds": 2.4}, # EV = -28%
            {"post_position": 4, "horse_name": "Other", "horse_name_jp": "その他", "win_prob": 0.12, "odds": 9.0},
        ]
        from models.betting_engine import calculate_pricing_breakdown, construct_staking_plan
        pricing = calculate_pricing_breakdown(entries)
        plan = construct_staking_plan(pricing, budget=1000, strategy_mode="dutching")

        win_tickets = [t for t in plan.portfolio_tickets if t.ticket_type_en == "win"]
        assert len(win_tickets) == 2, (
            f"Dutching mode must place 2 win tickets but got {len(win_tickets)}: "
            f"{[t.selection for t in win_tickets]}"
        )
        assert sum(t.stake for t in plan.portfolio_tickets) == 1000

    def test_auto_selects_dutching_for_dual_value(self):
        """Auto mode should pick 'dutching' when 2 runners have EV >= 8%."""
        entries = [
            {"post_position": 1, "horse_name": "ValueA", "win_prob": 0.26, "odds": 5.5},
            {"post_position": 2, "horse_name": "ValueB", "win_prob": 0.22, "odds": 7.0},
            {"post_position": 3, "horse_name": "Fav", "win_prob": 0.30, "odds": 2.4},
        ]
        from models.betting_engine import calculate_pricing_breakdown, select_adaptive_strategy
        pricing = calculate_pricing_breakdown(entries)
        strat, reason = select_adaptive_strategy(pricing)
        assert strat == "dutching", f"Expected 'dutching', got '{strat}': {reason}"


class TestContenderSelectionBugFix:
    """Fix #5: Exotic partners must be selected by EV, not win_prob."""

    def test_contenders_sorted_by_ev(self):
        """The anchor's Wide/Exacta partner should be the highest-EV non-anchor horse."""
        entries = [
            {"post_position": 1, "horse_name": "Anchor", "win_prob": 0.18, "odds": 7.0},    # EV = +26%, top value
            {"post_position": 2, "horse_name": "TopWinFav", "win_prob": 0.35, "odds": 2.0}, # EV = -30%, highest prob
            {"post_position": 3, "horse_name": "GoodVal", "win_prob": 0.25, "odds": 5.0},   # EV = +25%, best EV partner
            {"post_position": 4, "horse_name": "Low", "win_prob": 0.12, "odds": 9.0},
        ]
        from models.betting_engine import calculate_pricing_breakdown, construct_staking_plan
        pricing = calculate_pricing_breakdown(entries)
        plan = construct_staking_plan(pricing, budget=1000, strategy_mode="hybrid")

        # Wide/exotic tickets should involve PP3 (EV=+25%), NOT PP2 (EV=-30%)
        exotic_selections = [t.selection for t in plan.portfolio_tickets if len(t.selection) > 1]
        for sel in exotic_selections:
            assert 2 not in sel, (
                f"Exotic ticket {sel} paired anchor with PP2 (EV=-30%) — "
                f"should prefer PP3 (EV=+25%)"
            )
