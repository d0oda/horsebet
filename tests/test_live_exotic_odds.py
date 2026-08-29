"""
Unit and integration tests for Live JRA Exotic Pool Odds Upgrade.
"""
import pytest
from models.betting_engine import (
    HorsePricing,
    construct_staking_plan,
    BetTicket,
)
from scraper.odds_watcher import (
    parse_combination_key,
    get_live_exotic_odds_map,
)


def test_parse_combination_key():
    # Symmetric bets should sort combination parts
    assert parse_combination_key("trio", "3-8-9") == (3, 8, 9)
    assert parse_combination_key("trio", "9-3-8") == (3, 8, 9)
    assert parse_combination_key("quinella", "8-3") == (3, 8)
    assert parse_combination_key("wide", "12-4") == (4, 12)

    # Asymmetric bets should preserve order
    assert parse_combination_key("exacta", "8-3") == (8, 3)
    assert parse_combination_key("exacta", "3-8") == (3, 8)
    assert parse_combination_key("trifecta", "8-3-9") == (8, 3, 9)

    # Invalid input handling
    assert parse_combination_key("trio", "abc") is None
    assert parse_combination_key("trio", "") is None


def test_staking_plan_with_live_exotic_odds():
    pricing = [
        HorsePricing(post_position=3, horse_name="Horse A", horse_name_jp="ホースA", win_prob=0.35, fair_odds=2.9, market_odds=3.2, ev=0.12, verdict="Top pick & clear value", is_favorite=True, is_top_value=True, is_market_favorite=True),
        HorsePricing(post_position=8, horse_name="Horse B", horse_name_jp="ホースB", win_prob=0.20, fair_odds=5.0, market_odds=8.5, ev=0.70, verdict="Best value", is_favorite=False, is_top_value=False, is_market_favorite=False),
        HorsePricing(post_position=9, horse_name="Horse C", horse_name_jp="ホースC", win_prob=0.15, fair_odds=6.7, market_odds=14.0, ev=1.10, verdict="Secondary value", is_favorite=False, is_top_value=False, is_market_favorite=False),
        HorsePricing(post_position=1, horse_name="Horse D", horse_name_jp="ホースD", win_prob=0.10, fair_odds=10.0, market_odds=12.0, ev=0.20, verdict="Slight value", is_favorite=False, is_top_value=False, is_market_favorite=False),
        HorsePricing(post_position=2, horse_name="Horse E", horse_name_jp="ホースE", win_prob=0.10, fair_odds=10.0, market_odds=15.0, ev=0.50, verdict="Moderate value", is_favorite=False, is_top_value=False, is_market_favorite=False),
        HorsePricing(post_position=4, horse_name="Horse F", horse_name_jp="ホースF", win_prob=0.10, fair_odds=10.0, market_odds=20.0, ev=1.00, verdict="Secondary value", is_favorite=False, is_top_value=False, is_market_favorite=False),
    ]

    # Mock real live pool odds for 4-8-9 trio and 4-9 quinella
    exotic_odds_map = {
        ("trio", (4, 8, 9)): 634.9,
        ("quinella", (4, 9)): 45.0,
        ("wide", (4, 9)): 15.2,
        ("exacta", (9, 4)): 85.0,
    }

    plan = construct_staking_plan(pricing, budget=1000, strategy_mode="hybrid", exotic_odds_map=exotic_odds_map)
    assert len(plan.portfolio_tickets) > 0

    trio_ticket = None
    for t in plan.portfolio_tickets:
        if t.ticket_type == "三連複" or t.ticket_type_en == "trio":
            trio_ticket = t
            break

    assert trio_ticket is not None
    assert trio_ticket.odds_source == "live_pool"
    assert trio_ticket.market_odds == 634.9
    assert trio_ticket.estimated_market_odds is not None
    # EV should be recalculated using 634.9
    expected_ev = (trio_ticket.prob * 634.9) - 1.0
    assert pytest.approx(trio_ticket.ev, rel=1e-2) == expected_ev
    
    # Test serialization to_dict
    d = trio_ticket.to_dict()
    assert d["odds_source"] == "live_pool"
    assert d["market_odds"] == 634.9
    assert "estimated_market_odds" in d


def test_staking_plan_synthetic_fallback():
    # Same pricing breakdown without exotic_odds_map (fallback mode)
    pricing = [
        HorsePricing(post_position=3, horse_name="Horse A", horse_name_jp="ホースA", win_prob=0.35, fair_odds=2.9, market_odds=3.2, ev=0.12, verdict="Top pick & clear value", is_favorite=True, is_top_value=True, is_market_favorite=True),
        HorsePricing(post_position=8, horse_name="Horse B", horse_name_jp="ホースB", win_prob=0.20, fair_odds=5.0, market_odds=8.5, ev=0.70, verdict="Best value", is_favorite=False, is_top_value=False, is_market_favorite=False),
        HorsePricing(post_position=9, horse_name="Horse C", horse_name_jp="ホースC", win_prob=0.15, fair_odds=6.7, market_odds=14.0, ev=1.10, verdict="Secondary value", is_favorite=False, is_top_value=False, is_market_favorite=False),
        HorsePricing(post_position=1, horse_name="Horse D", horse_name_jp="ホースD", win_prob=0.10, fair_odds=10.0, market_odds=12.0, ev=0.20, verdict="Slight value", is_favorite=False, is_top_value=False, is_market_favorite=False),
        HorsePricing(post_position=2, horse_name="Horse E", horse_name_jp="ホースE", win_prob=0.10, fair_odds=10.0, market_odds=15.0, ev=0.50, verdict="Moderate value", is_favorite=False, is_top_value=False, is_market_favorite=False),
    ]

    plan = construct_staking_plan(pricing, budget=1000, strategy_mode="hybrid", exotic_odds_map=None)
    for t in plan.portfolio_tickets:
        if t.ticket_type not in ("単勝", "win"):
            assert t.odds_source == "synthetic"
            assert t.estimated_market_odds == t.market_odds
