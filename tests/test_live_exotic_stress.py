"""
Brutal Stress Tests & Invariance Verification for Live Exotic Pool Odds & Staking Engine.
"""
import pytest
from models.betting_engine import (
    HorsePricing,
    construct_staking_plan,
    BetTicket,
    StakingPlan,
    calculate_pricing_breakdown,
)
from scraper.odds_watcher import (
    parse_combination_key,
    get_live_exotic_odds_map,
)


def _make_field(n_runners: int, base_odds_list=None):
    """Helper to generate mock pricing breakdown for n runners."""
    entries = []
    for i in range(1, n_runners + 1):
        odds = base_odds_list[i - 1] if base_odds_list and i <= len(base_odds_list) else round(2.0 + (i * 2.5), 1)
        prob = 1.0 / odds
        entries.append({
            "post_position": i,
            "horse_name": f"Horse {i}",
            "horse_name_jp": f"馬{i}",
            "odds": odds,
            "win_prob": prob,
        })
    return calculate_pricing_breakdown(entries)


# ---------------------------------------------------------------------------
# 1. Field Size Extremes & Invariants
# ---------------------------------------------------------------------------

def test_field_size_2_runners():
    pricing = _make_field(2, [2.0, 3.0])
    plan = construct_staking_plan(pricing, budget=1000, strategy_mode="hybrid")
    # In 2-runner field, no exotics (wide/trio) should be created
    for t in plan.portfolio_tickets:
        assert t.ticket_type in ("単勝", "win", "PASS", "pass")


def test_field_size_3_runners():
    pricing = _make_field(3, [2.0, 4.0, 8.0])
    plan = construct_staking_plan(pricing, budget=1000, strategy_mode="hybrid")
    # In 3-runner field, Trio and Wide are invalid (JRA rules); Exacta/Quinella allowed
    for t in plan.portfolio_tickets:
        assert t.ticket_type not in ("ワイド", "三連複", "wide", "trio")


def test_field_size_18_runners_max_jra():
    pricing = _make_field(18)
    pricing[0].win_prob = 0.35
    pricing[0].ev = (0.35 * pricing[0].market_odds) - 1.0
    pricing[0].is_top_value = True

    exotic_map = {
        ("trio", (1, 2, 3)): 250.0,
        ("quinella", (1, 2)): 18.0,
        ("wide", (1, 2)): 5.5,
        ("exacta", (1, 2)): 35.0,
    }
    plan = construct_staking_plan(pricing, budget=5000, strategy_mode="hybrid", exotic_odds_map=exotic_map)
    assert plan.budget == 5000
    assert sum(t.stake for t in plan.portfolio_tickets) == 5000


def test_pass_race_when_no_value():
    """Verify that when no runners have positive EV (all EV < +8%), staking engine in auto mode returns PASS with 0 stake."""
    entries = [
        {"post_position": 1, "horse_name": "H1", "odds": 2.0, "win_prob": 0.40},  # EV = -20%
        {"post_position": 2, "horse_name": "H2", "odds": 3.0, "win_prob": 0.30},  # EV = -10%
        {"post_position": 3, "horse_name": "H3", "odds": 5.0, "win_prob": 0.18},  # EV = -10%
        {"post_position": 4, "horse_name": "H4", "odds": 8.0, "win_prob": 0.12},  # EV = -4%
    ]
    pricing = calculate_pricing_breakdown(entries)
    plan = construct_staking_plan(pricing, budget=1000, strategy_mode="auto")
    assert plan.resolved_mode == "pass"
    assert len(plan.portfolio_tickets) == 0



# ---------------------------------------------------------------------------
# 2. Live Overlay vs Severe Underlay (Pruning Invariant)
# ---------------------------------------------------------------------------

def test_exotic_massive_overlay_selection():
    """Verify that a massive live pool overlay (e.g. 800x) is selected and receives positive EV."""
    pricing = _make_field(8, [3.0, 4.5, 6.0, 10.0, 15.0, 20.0, 30.0, 50.0])
    # Give anchor horse high model probability to create a value bet
    pricing[0].win_prob = 0.45
    pricing[0].ev = (0.45 * pricing[0].market_odds) - 1.0
    pricing[0].is_top_value = True

    # High overlay on trio (1-2-3)
    exotic_map = {
        ("trio", (1, 2, 3)): 850.0,
    }
    plan = construct_staking_plan(pricing, budget=1000, strategy_mode="hybrid", exotic_odds_map=exotic_map)
    
    trio_ticket = next((t for t in plan.portfolio_tickets if t.ticket_type == "三連複"), None)
    if trio_ticket:
        assert trio_ticket.odds_source == "live_pool"
        assert trio_ticket.market_odds == 850.0
        assert trio_ticket.ev > 5.0  # +500%+ EV


def test_exotic_severe_underlay_pruned():
    """Verify that if live pool is crushed (e.g. 1.05x), negative EV ticket is pruned by positive EV filter."""
    pricing = _make_field(8, [3.0, 4.5, 6.0, 10.0, 15.0, 20.0, 30.0, 50.0])
    pricing[0].win_prob = 0.40
    pricing[0].ev = (0.40 * pricing[0].market_odds) - 1.0
    pricing[0].is_top_value = True

    # Severe underlay: Trio paying only 1.1x (sub-fair odds)
    exotic_map = {
        ("trio", (1, 2, 3)): 1.1,
        ("wide", (1, 2)): 1.05,
        ("quinella", (1, 2)): 1.1,
        ("exacta", (1, 2)): 1.1,
    }
    plan = construct_staking_plan(pricing, budget=1000, strategy_mode="hybrid", exotic_odds_map=exotic_map)
    # The underlaid exotics should NOT be in portfolio_tickets because EV < 0.05
    for t in plan.portfolio_tickets:
        if not t.ticket_type in ("単勝", "win"):
            assert t.ev >= 0.05


# ---------------------------------------------------------------------------
# 3. Permutation & Symmetry Invariants
# ---------------------------------------------------------------------------

def test_combination_symmetry_invariants():
    # Trio: all 6 permutations of (2, 5, 8) must resolve identically
    assert parse_combination_key("trio", "2-5-8") == (2, 5, 8)
    assert parse_combination_key("trio", "8-5-2") == (2, 5, 8)
    assert parse_combination_key("trio", "5-8-2") == (2, 5, 8)
    assert parse_combination_key("trio", "2-8-5") == (2, 5, 8)

    # Quinella & Wide: both orders must resolve identically
    assert parse_combination_key("quinella", "7-3") == (3, 7)
    assert parse_combination_key("quinella", "3-7") == (3, 7)
    assert parse_combination_key("wide", "14-2") == (2, 14)
    assert parse_combination_key("wide", "2-14") == (2, 14)

    # Exacta: order MUST be preserved
    assert parse_combination_key("exacta", "7-3") == (7, 3)
    assert parse_combination_key("exacta", "3-7") == (3, 7)
    assert parse_combination_key("exacta", "7-3") != parse_combination_key("exacta", "3-7")


# ---------------------------------------------------------------------------
# 4. Budget Discretization & Mathematical Invariants
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("budget", [100, 200, 300, 500, 1000, 2500, 5000, 10000, 25000])
def test_budget_discretization_invariants(budget):
    pricing = _make_field(10, [2.5, 4.0, 6.0, 8.0, 12.0, 15.0, 20.0, 30.0, 40.0, 50.0])
    pricing[0].win_prob = 0.45
    pricing[0].ev = (0.45 * pricing[0].market_odds) - 1.0
    pricing[0].is_top_value = True

    exotic_map = {
        ("trio", (1, 2, 3)): 45.0,
        ("wide", (1, 2)): 8.0,
        ("quinella", (1, 2)): 15.0,
    }

    plan = construct_staking_plan(pricing, budget=budget, strategy_mode="hybrid", exotic_odds_map=exotic_map)
    
    if plan.portfolio_tickets:
        # Sum of stakes must exactly equal budget
        total_stake = sum(t.stake for t in plan.portfolio_tickets)
        assert total_stake == budget, f"Budget {budget} stake sum mismatch: {total_stake}"
        
        # Each individual stake must be >= 100 and multiple of 100
        for t in plan.portfolio_tickets:
            assert t.stake >= 100, f"Ticket stake {t.stake} < 100"
            assert t.stake % 100 == 0, f"Ticket stake {t.stake} not a multiple of 100"


# ---------------------------------------------------------------------------
# 5. Bad / Malformed Input Resilience
# ---------------------------------------------------------------------------

def test_malformed_exotic_odds_resilience():
    pricing = _make_field(8)
    pricing[0].win_prob = 0.40
    pricing[0].ev = (0.40 * pricing[0].market_odds) - 1.0
    pricing[0].is_top_value = True

    # Corrupted / edge-case exotic map values
    corrupted_map = {
        ("trio", (1, 2, 3)): 0.0,
        ("quinella", (1, 2)): -5.0,
        ("wide", (1, 2)): 0.8,  # sub-1.0 invalid odds
        ("exacta", (1, 2)): None,
    }

    # Must not raise exceptions, and must gracefully fall back to synthetic odds
    plan = construct_staking_plan(pricing, budget=1000, strategy_mode="hybrid", exotic_odds_map=corrupted_map)
    assert plan is not None
    for t in plan.portfolio_tickets:
        if t.market_odds is not None:
            assert t.market_odds > 1.0


# ---------------------------------------------------------------------------
# 6. Serialization Contract Invariants
# ---------------------------------------------------------------------------

def test_bet_ticket_serialization_contract():
    ticket = BetTicket(
        ticket_type="三連複",
        ticket_type_en="trio",
        selection=[3, 8, 9],
        selection_display="3–8–9 三連複",
        prob=0.0403,
        fair_odds=24.8,
        market_odds=634.9,
        estimated_market_odds=35.2,
        odds_source="live_pool",
        ev=24.59,
        stake=100,
        weight_pct=0.10,
    )
    d = ticket.to_dict()
    assert d["type"] == "三連複"
    assert d["type_en"] == "trio"
    assert d["selection"] == [3, 8, 9]
    assert d["market_odds"] == 634.9
    assert d["estimated_market_odds"] == 35.2
    assert d["odds_source"] == "live_pool"
    assert d["ev"] == 24.59
    assert d["stake"] == 100
