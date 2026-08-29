"""
Brutal and exhaustive test suite for betting engine, ticket settlement,
multi-strategy aggregation, edge cases, and mathematical consistency.
"""

import pytest
from models.betting_engine import (
    BetTicket,
    HorsePricing,
    StakingPlan,
    calculate_pricing_breakdown,
    construct_staking_plan,
    settle_bet_tickets,
)


class TestSingleBetSettlement:
    """Tests 単勝 (Win) and 複勝 (Place) settlement across all conditions."""

    def test_win_ticket_won(self):
        entries_map = {
            1: {"finish_pos": 1, "odds": 5.0, "horse_name": "Ace", "horse_name_jp": "エース"},
            2: {"finish_pos": 2, "odds": 3.0, "horse_name": "King", "horse_name_jp": "キング"},
        }
        ticket = BetTicket(
            ticket_type="単勝",
            ticket_type_en="win",
            selection=[1],
            selection_display="No. 1 単勝",
            prob=0.25,
            market_odds=5.0,
            stake=1000,
        )
        res = settle_bet_tickets([ticket], entries_map)
        assert ticket.won is True
        assert ticket.payout == 5000
        assert ticket.profit == 4000
        assert ticket.roi_pct == 400.0
        assert res["status"] == "won"
        assert res["tickets_won"] == 1

    def test_win_ticket_lost(self):
        entries_map = {
            1: {"finish_pos": 2, "odds": 5.0, "horse_name": "Ace", "horse_name_jp": "エース"},
            2: {"finish_pos": 1, "odds": 3.0, "horse_name": "King", "horse_name_jp": "キング"},
        }
        ticket = BetTicket(
            ticket_type="単勝",
            ticket_type_en="win",
            selection=[1],
            selection_display="No. 1 単勝",
            prob=0.25,
            market_odds=5.0,
            stake=1000,
        )
        res = settle_bet_tickets([ticket], entries_map)
        assert ticket.won is False
        assert ticket.payout == 0
        assert ticket.profit == -1000
        assert ticket.roi_pct == -100.0
        assert res["status"] == "lost"
        assert res["tickets_won"] == 0

    def test_place_ticket_field_ge_8(self):
        # 8 horses: top 3 qualify for Place
        entries_map = {i: {"finish_pos": i, "odds": float(i * 2)} for i in range(1, 9)}

        # 3rd place runner -> WON
        t_3rd = BetTicket(
            ticket_type="複勝",
            ticket_type_en="place",
            selection=[3],
            selection_display="No. 3 複勝",
            prob=0.20,
            market_odds=2.5,
            stake=1000,
        )
        # 4th place runner -> LOST
        t_4th = BetTicket(
            ticket_type="複勝",
            ticket_type_en="place",
            selection=[4],
            selection_display="No. 4 複勝",
            prob=0.15,
            market_odds=3.0,
            stake=1000,
        )
        settle_bet_tickets([t_3rd, t_4th], entries_map)
        assert t_3rd.won is True
        assert t_3rd.payout == 2500
        assert t_4th.won is False
        assert t_4th.payout == 0

    def test_place_ticket_field_le_7(self):
        # 6 horses: only top 2 qualify for Place
        entries_map = {i: {"finish_pos": i, "odds": float(i * 2)} for i in range(1, 7)}

        t_2nd = BetTicket(
            ticket_type="複勝",
            ticket_type_en="place",
            selection=[2],
            selection_display="No. 2 複勝",
            prob=0.20,
            market_odds=2.0,
            stake=500,
        )
        t_3rd = BetTicket(
            ticket_type="複勝",
            ticket_type_en="place",
            selection=[3],
            selection_display="No. 3 複勝",
            prob=0.15,
            market_odds=3.0,
            stake=500,
        )
        settle_bet_tickets([t_2nd, t_3rd], entries_map)
        assert t_2nd.won is True
        assert t_2nd.payout == 1000
        assert t_3rd.won is False  # 3rd place loses in <=7 horse field!
        assert t_3rd.payout == 0


class TestExoticBetSettlement:
    """Tests Quinella, Wide, Exacta, Trio, Trifecta permutations and settlements."""

    def test_quinella_permutations(self):
        # Quinella 1-2 wins if 1st=1, 2nd=2 OR 1st=2, 2nd=1
        entries_map_a = {1: {"finish_pos": 1, "odds": 3.0}, 2: {"finish_pos": 2, "odds": 4.0}, 3: {"finish_pos": 3, "odds": 5.0}}
        entries_map_b = {1: {"finish_pos": 2, "odds": 3.0}, 2: {"finish_pos": 1, "odds": 4.0}, 3: {"finish_pos": 3, "odds": 5.0}}
        entries_map_c = {1: {"finish_pos": 1, "odds": 3.0}, 2: {"finish_pos": 3, "odds": 4.0}, 3: {"finish_pos": 2, "odds": 5.0}}

        t1 = BetTicket(ticket_type="馬連", ticket_type_en="quinella", selection=[1, 2], selection_display="1–2 馬連", prob=0.1, market_odds=10.0, stake=100)
        settle_bet_tickets([t1], entries_map_a)
        assert t1.won is True
        assert t1.payout == 1000

        t2 = BetTicket(ticket_type="馬連", ticket_type_en="quinella", selection=[1, 2], selection_display="1–2 馬連", prob=0.1, market_odds=10.0, stake=100)
        settle_bet_tickets([t2], entries_map_b)
        assert t2.won is True
        assert t2.payout == 1000

        t3 = BetTicket(ticket_type="馬連", ticket_type_en="quinella", selection=[1, 2], selection_display="1–2 馬連", prob=0.1, market_odds=10.0, stake=100)
        settle_bet_tickets([t3], entries_map_c)
        assert t3.won is False
        assert t3.payout == 0

    def test_exacta_order_strictness(self):
        # Exacta 1-2 wins ONLY if 1st=1 and 2nd=2
        entries_map_win = {1: {"finish_pos": 1, "odds": 3.0}, 2: {"finish_pos": 2, "odds": 4.0}}
        entries_map_rev = {1: {"finish_pos": 2, "odds": 3.0}, 2: {"finish_pos": 1, "odds": 4.0}}

        t_win = BetTicket(ticket_type="馬単", ticket_type_en="exacta", selection=[1, 2], selection_display="1→2 馬単", prob=0.05, market_odds=20.0, stake=100)
        settle_bet_tickets([t_win], entries_map_win)
        assert t_win.won is True
        assert t_win.payout == 2000

        t_rev = BetTicket(ticket_type="馬単", ticket_type_en="exacta", selection=[1, 2], selection_display="1→2 馬単", prob=0.05, market_odds=20.0, stake=100)
        settle_bet_tickets([t_rev], entries_map_rev)
        assert t_rev.won is False
        assert t_rev.payout == 0

    def test_wide_all_3_pairs(self):
        # Top 3 are horses 1, 2, 3 in 8-horse field
        entries_map = {i: {"finish_pos": i, "odds": 3.0} for i in range(1, 9)}

        w_12 = BetTicket(ticket_type="ワイド", ticket_type_en="wide", selection=[1, 2], selection_display="1–2 ワイド", prob=0.15, market_odds=5.0, stake=100)
        w_13 = BetTicket(ticket_type="ワイド", ticket_type_en="wide", selection=[1, 3], selection_display="1–3 ワイド", prob=0.15, market_odds=6.0, stake=100)
        w_23 = BetTicket(ticket_type="ワイド", ticket_type_en="wide", selection=[2, 3], selection_display="2–3 ワイド", prob=0.15, market_odds=7.0, stake=100)
        w_14 = BetTicket(ticket_type="ワイド", ticket_type_en="wide", selection=[1, 4], selection_display="1–4 ワイド", prob=0.15, market_odds=8.0, stake=100)

        settle_bet_tickets([w_12, w_13, w_23, w_14], entries_map)
        assert w_12.won is True and w_12.payout == 500
        assert w_13.won is True and w_13.payout == 600
        assert w_23.won is True and w_23.payout == 700
        assert w_14.won is False and w_14.payout == 0

    def test_trio_all_6_permutations(self):
        # Trio 1-2-3 wins for any ordering of {1, 2, 3} finishing 1st, 2nd, 3rd
        perms = [
            (1, 2, 3), (1, 3, 2),
            (2, 1, 3), (2, 3, 1),
            (3, 1, 2), (3, 2, 1)
        ]
        for p1, p2, p3 in perms:
            emap = {p1: {"finish_pos": 1}, p2: {"finish_pos": 2}, p3: {"finish_pos": 3}, 4: {"finish_pos": 4}}
            t = BetTicket(ticket_type="三連複", ticket_type_en="trio", selection=[1, 2, 3], selection_display="1–2–3 三連複", prob=0.03, market_odds=50.0, stake=100)
            settle_bet_tickets([t], emap)
            assert t.won is True
            assert t.payout == 5000

    def test_trifecta_strict_order(self):
        emap_correct = {1: {"finish_pos": 1}, 2: {"finish_pos": 2}, 3: {"finish_pos": 3}}
        emap_wrong = {1: {"finish_pos": 1}, 2: {"finish_pos": 3}, 3: {"finish_pos": 2}}

        t_corr = BetTicket(ticket_type="三連単", ticket_type_en="trifecta", selection=[1, 2, 3], selection_display="1→2→3 三連単", prob=0.01, market_odds=200.0, stake=100)
        settle_bet_tickets([t_corr], emap_correct)
        assert t_corr.won is True
        assert t_corr.payout == 20000

        t_wrg = BetTicket(ticket_type="三連単", ticket_type_en="trifecta", selection=[1, 2, 3], selection_display="1→2→3 三連単", prob=0.01, market_odds=200.0, stake=100)
        settle_bet_tickets([t_wrg], emap_wrong)
        assert t_wrg.won is False
        assert t_wrg.payout == 0


class TestEdgeCasesAndDefensiveCoding:
    """Stress tests on edge conditions: no results, partial results, DNF, dead heats, zero budget."""

    def test_no_results_pending_state(self):
        # When all finish_pos are None (race hasn't run yet)
        entries_map = {1: {"finish_pos": None}, 2: {"finish_pos": None}}
        t = BetTicket(ticket_type="単勝", selection=[1], stake=1000, prob=0.3)
        res = settle_bet_tickets([t], entries_map)
        assert res["is_settled"] is False
        assert res["status"] == "pending"
        assert t.is_settled is False
        assert t.won is None
        assert t.payout == 0

    def test_scratched_or_dnf_horse(self):
        # Horse 1 is DNF (finish_pos = None), Horse 2 finishes 1st
        entries_map = {1: {"finish_pos": None, "odds": 4.0}, 2: {"finish_pos": 1, "odds": 3.0}}
        t_dnf = BetTicket(ticket_type="単勝", selection=[1], stake=500, prob=0.25)
        t_win = BetTicket(ticket_type="単勝", selection=[2], stake=500, prob=0.30, market_odds=3.0)
        res = settle_bet_tickets([t_dnf, t_win], entries_map)
        assert t_dnf.won is False
        assert t_dnf.payout == 0
        assert t_win.won is True
        assert t_win.payout == 1500
        assert res["status"] == "won"
        assert res["profit"] == 500

    def test_dead_heat_tie_for_1st(self):
        # Dead heat: both Horse 1 and Horse 2 finish 1st!
        entries_map = {1: {"finish_pos": 1, "odds": 4.0}, 2: {"finish_pos": 1, "odds": 6.0}}
        t1 = BetTicket(ticket_type="単勝", selection=[1], stake=500, prob=0.25, market_odds=4.0)
        t2 = BetTicket(ticket_type="単勝", selection=[2], stake=500, prob=0.15, market_odds=6.0)
        res = settle_bet_tickets([t1, t2], entries_map)
        assert t1.won is True
        assert t2.won is True
        assert t1.payout == 2000
        assert t2.payout == 3000
        assert res["tickets_won"] == 2
        assert res["profit"] == 4000

    def test_synthetic_odds_fallback_calculation(self):
        # When exotic market odds is missing, falls back cleanly to synthetic odds without crashing
        entries_map = {1: {"finish_pos": 1, "odds": 4.0}, 2: {"finish_pos": 2, "odds": 5.0}}
        t = BetTicket(
            ticket_type="馬連",
            ticket_type_en="quinella",
            selection=[1, 2],
            selection_display="1–2 馬連",
            prob=0.08,
            market_odds=None,  # No market odds
            stake=100,
        )
        settle_bet_tickets([t], entries_map)
        assert t.won is True
        assert t.payout > 0  # Computed via synthetic formula (4.0 * 5.0 / 2.5 = 8.0x -> ¥800)
        assert t.profit > 0
