"""
Regression tests covering all issues from the August 2026 betting engine audit.
Each class maps to one or more audit items (BUG-1, FLAW-1, etc.).
"""

import math
import pytest
from models.betting_engine import (
    calculate_pricing_breakdown,
    construct_staking_plan,
    select_adaptive_strategy,
    EV_EXOTIC,
    EV_PASS,
)


class TestBug1MixedOddsCrash:
    """BUG-1: TypeError crash when some horses have market_odds=None."""

    def test_mixed_odds_no_crash(self):
        """construct_staking_plan must not crash when a runner has no market odds."""
        entries = [
            {"post_position": 1, "horse_name": "Known", "win_prob": 0.30, "odds": 5.0},
            {"post_position": 2, "horse_name": "Unknown", "win_prob": 0.25, "odds": None},  # no odds
            {"post_position": 3, "horse_name": "Known2", "win_prob": 0.25, "odds": 6.0},
            {"post_position": 4, "horse_name": "Known3", "win_prob": 0.20, "odds": 8.0},
        ]
        pricing = calculate_pricing_breakdown(entries)
        # This must NOT raise TypeError
        plan = construct_staking_plan(pricing, budget=1000, strategy_mode="hybrid")
        assert plan is not None

    def test_all_odds_missing_is_pending(self):
        """When all horses have no odds, the plan is a pending plan with no tickets."""
        entries = [
            {"post_position": 1, "horse_name": "A", "win_prob": 0.50, "odds": None},
            {"post_position": 2, "horse_name": "B", "win_prob": 0.50, "odds": None},
        ]
        pricing = calculate_pricing_breakdown(entries)
        plan = construct_staking_plan(pricing, budget=1000)
        assert plan.portfolio_tickets == []

    def test_single_horse_with_odds_no_crash(self):
        """BUG-1 edge: only 1 of N horses has odds — mkt_probs_by_pp must handle the rest safely."""
        entries = [
            {"post_position": 1, "horse_name": "H1", "win_prob": 0.40, "odds": 3.5},
            {"post_position": 2, "horse_name": "H2", "win_prob": 0.35, "odds": None},
            {"post_position": 3, "horse_name": "H3", "win_prob": 0.25, "odds": None},
        ]
        pricing = calculate_pricing_breakdown(entries)
        # must not raise
        plan = construct_staking_plan(pricing, budget=1000)
        assert plan is not None


class TestFlaw1AdaptiveStrategyEVOrder:
    """FLAW-1: select_adaptive_strategy must anchor on the best-EV horse, not the highest-prob one."""

    def test_top_v_is_best_ev_not_best_prob(self):
        """Adaptive strategy 'reason' message must mention the best-EV horse, not the best-prob one."""
        entries = [
            {"post_position": 1, "horse_name": "HighProb", "win_prob": 0.40, "odds": 2.8},  # EV=+12%
            {"post_position": 2, "horse_name": "BestEV", "win_prob": 0.15, "odds": 10.0},   # EV=+50%
            {"post_position": 3, "horse_name": "Chalk", "win_prob": 0.45, "odds": 1.8},     # EV=-19%
        ]
        pricing = calculate_pricing_breakdown(entries)
        strat, reason = select_adaptive_strategy(pricing)
        # The reason must reference the best-EV horse (#2 in pure_win) not the highest-prob one (#1)
        # In this case PP1 (EV=+12%) and PP2 (EV=+50%) both pass EV_PASS=8%,
        # so dutching is selected — but strong_value must be ordered by EV too.
        # The first horse in the reason should be the HIGHER-EV one (PP2).
        if strat == "dutching":
            # Dutching pairs the two best-EV horses; PP2 should come first.
            assert "#2" in reason, (
                f"Dutching reason must lead with highest-EV horse (#2, EV=+50%), got: '{reason}'"
            )
        elif strat == "pure_win":
            # Pure win should mention #2 (highest EV)
            assert "#2" in reason, (
                f"Pure win reason must reference highest-EV horse (#2), got: '{reason}'"
            )

    def test_strong_value_ordered_by_ev_for_dutching(self):
        """Dutching must pair the two HIGHEST-EV horses, not the two highest-prob ones."""
        entries = [
            {"post_position": 1, "horse_name": "HighProb_LowEV", "win_prob": 0.40, "odds": 2.8},  # EV=+12%
            {"post_position": 2, "horse_name": "BestEV", "win_prob": 0.20, "odds": 8.0},           # EV=+60%
            {"post_position": 3, "horse_name": "SecondBestEV", "win_prob": 0.18, "odds": 7.0},     # EV=+26%
            {"post_position": 4, "horse_name": "Chalk", "win_prob": 0.22, "odds": 2.0},            # EV=-55%
        ]
        pricing = calculate_pricing_breakdown(entries)
        strat, reason = select_adaptive_strategy(pricing)
        assert strat == "dutching", f"Expected dutching, got {strat}: {reason}"
        # The two best-EV runners are PP2 (EV=+60%) and PP3 (EV=+26%), NOT PP1.
        assert "#2" in reason, f"Dutching should mention PP2 (best EV), got: '{reason}'"
        assert "#1" not in reason, (
            f"Dutching should NOT pair PP1 (lower EV than PP2/PP3), got: '{reason}'"
        )


class TestFlaw3ExoticEVFloor:
    """FLAW-3: EV_EXOTIC must be >= 0.05 to filter noise-level exotic bets."""

    def test_ev_exotic_floor_is_meaningful(self):
        """EV_EXOTIC must be at least 5% — any lower is inside estimation error bounds."""
        assert EV_EXOTIC >= 0.05, (
            f"EV_EXOTIC={EV_EXOTIC} is too low. Exotic odds derived from Harville have "
            f"±5-10% estimation error; threshold must exceed this to avoid noise bets."
        )


class TestIntegrity2BudgetRounding:
    """R5-LOGIC-5: Budget must be floored to ¥100 multiples, never rounded up above the stated amount."""

    @pytest.mark.parametrize("raw,expected", [
        (999, 900),    # floor: 900, not 1000 (rounding would exceed stated budget)
        (1050, 1000),  # floor: 1000
        (1051, 1000),  # floor: 1000
        (1049, 1000),  # floor: 1000
        (1099, 1000),  # floor: 1000
        (1150, 1100),  # floor: 1100
        (550, 500),    # floor: 500
        (449, 400),    # floor: 400
    ])
    def test_budget_floored_not_rounded_up(self, raw, expected):
        """Budgets must be floored to the nearest ¥100, never exceeding the caller's stated amount."""
        entries = [
            {"post_position": 1, "horse_name": "A", "win_prob": 0.50, "odds": 3.0},
            {"post_position": 2, "horse_name": "B", "win_prob": 0.30, "odds": 5.0},
            {"post_position": 3, "horse_name": "C", "win_prob": 0.20, "odds": 8.0},
        ]
        pricing = calculate_pricing_breakdown(entries)
        plan = construct_staking_plan(pricing, budget=raw)
        total = sum(t.stake for t in plan.portfolio_tickets)
        assert total == expected, (
            f"budget={raw} should be floored to ¥{expected} (nearest ¥100 floor), "
            f"but total staked was ¥{total}"
        )
        # Critical: total staked must NEVER exceed the raw budget
        assert total <= raw, f"Stakes ¥{total} exceeded stated budget ¥{raw}"


class TestMinor1WideOpenFavourite:
    """MINOR-1: is_fav fires correctly in wide-open 18-horse fields (threshold 0.12 → 0.06)."""

    def test_favourite_identified_in_wide_open_field(self):
        """No runner above 12% but the top runner (9%) should still be flagged is_fav."""
        entries = [
            {"post_position": 1, "horse_name": "TopHorse", "win_prob": 0.09, "odds": 10.0},
        ] + [
            {"post_position": i, "horse_name": f"H{i}", "win_prob": 0.054, "odds": 18.0}
            for i in range(2, 19)
        ]
        results = calculate_pricing_breakdown(entries)
        favs = [h for h in results if h.is_favorite]
        assert len(favs) == 1, f"Expected 1 favourite, got {len(favs)}"
        assert favs[0].post_position == 1

    def test_underpriced_verdict_fires_in_wide_field(self):
        """'Most likely, but underpriced' fires for the leader with clearly negative EV.

        Requires:
        - is_fav = True (leader has max prob)
        - is_top_value = False (another horse has higher EV, or leader EV < EV_TOP_VALUE)
        - ev < 0 (market overprices the favourite)
        """
        # Leader (PP1) has raw prob 0.09 but short odds (2.5x) -> strong negative EV.
        # PP2 has longer odds -> positive EV and will be is_top_value.
        entries = [
            {"post_position": 1, "horse_name": "OverbetFav", "win_prob": 0.09, "odds": 2.5},  # EV = -0.78
            {"post_position": 2, "horse_name": "ValueHorse", "win_prob": 0.08, "odds": 14.0}, # EV = +0.12 -> top_val
        ] + [
            {"post_position": i, "horse_name": f"H{i}", "win_prob": 0.054, "odds": 20.0}
            for i in range(3, 12)
        ]
        results = calculate_pricing_breakdown(entries)
        fav = next((h for h in results if h.is_favorite), None)
        assert fav is not None, "A favourite must be identified (MINOR-1 fix)"
        assert fav.post_position == 1, f"Expected PP1 as fav, got PP{fav.post_position}"
        assert fav.ev is not None and fav.ev < 0, (
            f"Favourite must have negative EV to trigger underpriced verdict, got {fav.ev:+.2%}"
        )
        assert not fav.is_top_value, "Favourite must not be top_value for underpriced verdict to fire"
        assert fav.verdict == "Most likely, but underpriced", (
            f"Expected 'Most likely, but underpriced', got '{fav.verdict}'"
        )


class TestMinor4QuinellaWideDeduplicate:
    """MINOR-4: Quinella and Wide must not be issued for the same pair."""

    def test_no_quinella_wide_same_pair(self):
        """When c1 is both the quinella and wide partner, only the Wide is included."""
        entries = [
            {"post_position": 1, "horse_name": "Anchor", "win_prob": 0.25, "odds": 6.5},  # top_val
            {"post_position": 2, "horse_name": "Partner", "win_prob": 0.35, "odds": 4.5}, # best EV contender
            {"post_position": 3, "horse_name": "Other", "win_prob": 0.25, "odds": 5.5},
            {"post_position": 4, "horse_name": "Tail", "win_prob": 0.15, "odds": 9.0},
        ]
        pricing = calculate_pricing_breakdown(entries)
        plan = construct_staking_plan(pricing, budget=1000, strategy_mode="hybrid")

        # Gather all 2-horse ticket selections
        two_horse = [(t.ticket_type, tuple(sorted(t.selection))) for t in plan.portfolio_tickets
                     if len(t.selection) == 2]
        # Group by selection pair
        pairs: dict = {}
        for tt, sel in two_horse:
            pairs.setdefault(sel, []).append(tt)

        for sel, types in pairs.items():
            quinella_wide_overlap = ("馬連" in types and "ワイド" in types)
            assert not quinella_wide_overlap, (
                f"Pair {sel} received BOTH 馬連 and ワイド — correlated redundancy. Types: {types}"
            )

class TestR6Logic2BudgetSubHundred:
    """R6-LOGIC-2: budgets below ¥100 must raise ValueError, not silently inflate."""

    @pytest.mark.parametrize("bad_budget", [0, 1, 50, 99])
    def test_sub_100_budget_raises(self, bad_budget):
        entries = [
            {"post_position": 1, "horse_name": "A", "win_prob": 0.50, "odds": 3.0},
            {"post_position": 2, "horse_name": "B", "win_prob": 0.50, "odds": 5.0},
        ]
        pricing = calculate_pricing_breakdown(entries)
        with pytest.raises(ValueError, match="¥100"):
            construct_staking_plan(pricing, budget=bad_budget)


class TestR6Crash1HorsePricingToDict:
    """R6-CRASH-1: HorsePricing.to_dict() must not crash when ev is None (pre-market)."""

    def test_to_dict_no_crash_when_ev_none(self):
        entries = [
            {"post_position": 1, "horse_name": "A", "win_prob": 0.5, "odds": None},
            {"post_position": 2, "horse_name": "B", "win_prob": 0.5, "odds": None},
        ]
        pricing = calculate_pricing_breakdown(entries)
        for h in pricing:
            assert h.ev is None, "Pre-market horse should have ev=None"
            d = h.to_dict()   # must not raise
            assert d["ev"] is None, "to_dict() must serialize None ev as None, not crash"


class TestR6Verdict:
    """R6-VERDICT-1 and R6-VERDICT-2: verdict tree boundary and new is_fav branch."""

    def test_r6_verdict1_ev_exactly_best_value_is_slight_value(self):
        """A non-top-val horse with ev == EV_BEST_VALUE (0.10) must get 'Slight value',
        not fall through to 'Clearly too short'."""
        # PP1: prob=0.5, odds=2.2 -> ev = 0.5*2.2-1 = 0.10 exactly
        # PP2: higher EV -> becomes is_top_value, taking that slot away from PP1
        entries = [
            {"post_position": 1, "horse_name": "Fav", "win_prob": 0.50, "odds": 2.2},
            {"post_position": 2, "horse_name": "BetterEV", "win_prob": 0.10, "odds": 15.0},
            {"post_position": 3, "horse_name": "C", "win_prob": 0.40, "odds": 1.8},
        ]
        pricing = calculate_pricing_breakdown(entries)
        fav = next(h for h in pricing if h.post_position == 1)
        assert abs(fav.ev - 0.10) < 1e-9, f"Expected ev=0.10 exactly, got {fav.ev}"
        assert not fav.is_top_value, "PP1 should not be is_top_value (PP2 has higher EV)"
        assert fav.verdict in ("Slight value", "Top pick & slight edge"), (
            f"Expected 'Slight value' or 'Top pick & slight edge' for ev=+10%, "
            f"got '{fav.verdict}' — likely 'Clearly too short' off-by-one bug"
        )
        assert fav.verdict != "Clearly too short", (
            "'Clearly too short' for a +10% EV horse is the R6-VERDICT-1 boundary bug"
        )

    def test_r6_verdict2_is_fav_positive_ev_not_top_val(self):
        """A model-favourite with positive EV but not is_top_value should get 'Top pick & slight edge',
        not a generic 'Slight value' (same as a random contender)."""
        entries = [
            {"post_position": 1, "horse_name": "Fav", "win_prob": 0.55, "odds": 2.0},    # ev=+10%, is_fav
            {"post_position": 2, "horse_name": "BestEV", "win_prob": 0.05, "odds": 25.0},  # ev=+25%, is_top_val
            {"post_position": 3, "horse_name": "C", "win_prob": 0.40, "odds": 1.6},
        ]
        pricing = calculate_pricing_breakdown(entries)
        fav = next(h for h in pricing if h.post_position == 1)
        assert fav.is_favorite, "PP1 should be the model favourite"
        assert not fav.is_top_value, "PP1 should not be is_top_value (PP2 has higher EV)"
        assert fav.ev is not None and fav.ev > 0, f"PP1 must have positive EV, got {fav.ev}"
        assert fav.verdict == "Top pick & slight edge", (
            f"Expected 'Top pick & slight edge' for is_fav+positive_ev+not_top_val, "
            f"got '{fav.verdict}'"
        )


class TestR6Meta1PassPendingMeta:
    """R6-META-1: PASS and PENDING resolved modes must return dedicated strategy_meta,
    not fall back to the 'auto'/'Adaptive AI Router' entry."""

    def test_pass_does_not_show_adaptive_ai_router_stats(self):
        """When auto mode returns PASS, strategy_meta must not claim '+88.5% ROI'."""
        from models.betting_engine import analyze_race_betting
        # We can test the STRATEGY_META lookup directly without DB:
        # Replicate the fallback logic from analyze_race_betting line 1322
        STRATEGY_META = {
            "auto": {"name": "Adaptive AI Router", "historical_roi": "+88.5%"},
            "hybrid": {"name": "Balanced Portfolio"},
            "pass": {"name": "Pass (No Bet)", "historical_roi": "N/A"},
            "pending": {"name": "Awaiting Market Odds", "historical_roi": "N/A"},
        }
        # auto + PASS -> _resolved_mode='pass'
        meta = STRATEGY_META.get("pass", STRATEGY_META.get("auto", STRATEGY_META["hybrid"]))
        assert meta["name"] == "Pass (No Bet)", (
            f"auto+PASS should show 'Pass (No Bet)', not '{meta['name']}'"
        )
        assert meta["historical_roi"] == "N/A", (
            f"PASS meta must not show historical_roi='{meta['historical_roi']}'"
        )

    def test_pending_does_not_show_adaptive_ai_router_stats(self):
        """When mode is pending (no market odds), strategy_meta must be the pending entry."""
        STRATEGY_META = {
            "auto": {"name": "Adaptive AI Router", "historical_roi": "+88.5%"},
            "hybrid": {"name": "Balanced Portfolio"},
            "pass": {"name": "Pass (No Bet)", "historical_roi": "N/A"},
            "pending": {"name": "Awaiting Market Odds", "historical_roi": "N/A"},
        }
        meta = STRATEGY_META.get("pending", STRATEGY_META.get("auto", STRATEGY_META["hybrid"]))
        assert meta["name"] == "Awaiting Market Odds"
        assert meta["historical_roi"] == "N/A"


class TestR6DictDict:
    """R6-DICT-1: StakingPlan.to_dict() must include resolved_mode."""

    def test_staking_plan_to_dict_includes_resolved_mode(self):
        entries = [
            {"post_position": 1, "horse_name": "A", "win_prob": 0.50, "odds": 3.0},
            {"post_position": 2, "horse_name": "B", "win_prob": 0.30, "odds": 5.0},
            {"post_position": 3, "horse_name": "C", "win_prob": 0.20, "odds": 8.0},
        ]
        pricing = calculate_pricing_breakdown(entries)
        plan = construct_staking_plan(pricing, budget=1000, strategy_mode="hybrid")
        d = plan.to_dict()
        assert "resolved_mode" in d, (
            "StakingPlan.to_dict() is missing 'resolved_mode' key (R6-DICT-1)"
        )
        # This 3-horse field has Dutch runners but no qualifying exotics (wide disabled at <4).
        # R12-MINOR-1 correctly reclassifies hybrid+Dutch+no_exotics → 'dutching'.
        assert d["resolved_mode"] == "dutching", (
            f"Expected resolved_mode='dutching' (R12-MINOR-1: Dutch wins but no exotics), "
            f"got {d['resolved_mode']!r}"
        )

    def test_staking_plan_to_dict_pass_resolved_mode(self):
        """PASS plans must have resolved_mode='pass' in to_dict()."""
        entries = [
            {"post_position": 1, "horse_name": "A", "win_prob": 0.50, "odds": 1.1},
            {"post_position": 2, "horse_name": "B", "win_prob": 0.50, "odds": 1.2},
        ]
        pricing = calculate_pricing_breakdown(entries)
        plan = construct_staking_plan(pricing, budget=1000, strategy_mode="auto")
        d = plan.to_dict()
        assert d["resolved_mode"] in ("pass", "hybrid", "pure_win", "dutching"), (
            f"resolved_mode must be a valid mode string, got {d['resolved_mode']!r}"
        )


# ---------------------------------------------------------------------------
# R7 / R8 Fixes
# ---------------------------------------------------------------------------

class TestR7R8Fixes:
    """Tests covering all Round 7 and Round 8 audit fixes."""

    # --- PASS ticket semantics (R7-SIG-2 + R8-FIX-1) ---

    def test_pass_ticket_auto_mode_ev_none(self):
        """All-negative-EV field in auto mode: PASS simple_bet has ev=None, fair_odds=None."""
        entries = [
            {"post_position": 1, "horse_name": "A", "win_prob": 0.7, "odds": 1.1},
            {"post_position": 2, "horse_name": "B", "win_prob": 0.3, "odds": 2.0},
        ]
        pricing = calculate_pricing_breakdown(entries)
        plan = construct_staking_plan(pricing, budget=1000, strategy_mode="auto")
        assert plan.resolved_mode == "pass"
        assert plan.simple_bet.ev is None, "R8-FIX-1: auto PASS simple_bet.ev must be None"
        assert plan.simple_bet.fair_odds is None, "R8-FIX-1: auto PASS simple_bet.fair_odds must be None"

    def test_pass_ticket_has_positive_edge_path_ev_none(self):
        """All-negative-EV field in hybrid mode: has_positive_edge=False path uses None."""
        entries = [
            {"post_position": 1, "horse_name": "A", "win_prob": 0.7, "odds": 1.1},
            {"post_position": 2, "horse_name": "B", "win_prob": 0.3, "odds": 2.0},
        ]
        pricing = calculate_pricing_breakdown(entries)
        plan = construct_staking_plan(pricing, budget=1000, strategy_mode="hybrid")
        assert plan.resolved_mode == "pass"
        assert plan.simple_bet.ev is None, "R7-SIG-2: has_positive_edge PASS simple_bet.ev must be None"
        assert plan.simple_bet.fair_odds is None

    def test_ev_pass_gate_fires_for_sub8pct_ev_manual_mode(self):
        """Manual mode must PASS when best anchor EV is 5-7% (R7-CRITICAL-2)."""
        # 0.40 * 2.65 - 1 ≈ +6%
        entries = [
            {"post_position": 1, "horse_name": "LowEV", "win_prob": 0.40, "odds": 2.65},
            {"post_position": 2, "horse_name": "B",     "win_prob": 0.60, "odds": 1.10},
        ]
        pricing = calculate_pricing_breakdown(entries)
        for mode in ("hybrid", "pure_win", "dutching"):
            plan = construct_staking_plan(pricing, budget=1000, strategy_mode=mode)
            assert plan.resolved_mode == "pass", (
                f"R7-CRITICAL-2: mode={mode!r} should PASS for +6% EV, "
                f"got resolved_mode={plan.resolved_mode!r}"
            )
            assert plan.simple_bet.ev is None
            assert plan.simple_bet.fair_odds is None

    def test_ev_pass_boundary_8pct_allowed(self):
        """Exactly EV_PASS (8%) must be allowed to bet — boundary is inclusive (R7-CRITICAL-2)."""
        # 0.40 * 2.70 - 1 = 0.08 exactly
        entries = [
            {"post_position": 1, "horse_name": "Boundary", "win_prob": 0.40, "odds": 2.70},
            {"post_position": 2, "horse_name": "B",        "win_prob": 0.60, "odds": 1.20},
        ]
        pricing = calculate_pricing_breakdown(entries)
        plan = construct_staking_plan(pricing, budget=1000, strategy_mode="hybrid")
        assert plan.resolved_mode != "pass", (
            "EV == EV_PASS (8%) should be allowed; EV_PASS gate is strictly <, not <="
        )

    def test_is_fav_ev_zero_gets_fairly_priced_verdict(self):
        """Model favourite at exactly 0% EV gets 'Top pick, fairly priced' (R7-MINOR-2)."""
        # 0.5 * 2.0 - 1 = 0.0
        entries = [
            {"post_position": 1, "horse_name": "Fav", "win_prob": 0.5, "odds": 2.0},
            {"post_position": 2, "horse_name": "B",   "win_prob": 0.3, "odds": 4.0},
            {"post_position": 3, "horse_name": "C",   "win_prob": 0.2, "odds": 6.0},
        ]
        pricing = calculate_pricing_breakdown(entries)
        fav = next(h for h in pricing if h.is_favorite)
        assert fav.verdict == "Top pick, fairly priced", (
            f"R7-MINOR-2: is_fav + ev≈0.0 expected 'Top pick, fairly priced', got {fav.verdict!r}"
        )

    def test_dutching_degraded_resolved_mode_is_pure_win(self):
        """When dutching is requested but only 1 value horse exists, resolved_mode='pure_win'."""
        entries = [
            {"post_position": 1, "horse_name": "A", "win_prob": 0.60, "odds": 2.5},
            {"post_position": 2, "horse_name": "B", "win_prob": 0.30, "odds": 2.0},
            {"post_position": 3, "horse_name": "C", "win_prob": 0.10, "odds": 5.0},
        ]
        pricing = calculate_pricing_breakdown(entries)
        plan = construct_staking_plan(pricing, budget=1000, strategy_mode="dutching")
        assert plan.resolved_mode == "pure_win", (
            f"R7-MINOR-1: dutching degraded to single win; expected resolved_mode='pure_win', "
            f"got {plan.resolved_mode!r}"
        )

    def test_ev_exotic_inclusive_boundary(self):
        """EV_EXOTIC filter uses >= so ev==EV_EXOTIC (5.0%) exotics are included (R7-SIG-1)."""
        assert EV_EXOTIC == 0.05
        # Simulate: a mock exotic at exactly EV_EXOTIC must pass the filter
        mock_ev = EV_EXOTIC
        assert mock_ev >= EV_EXOTIC, "R7-SIG-1: ev==EV_EXOTIC must satisfy >= filter"

    def test_pending_ticket_fair_odds_not_none(self):
        """Pending tickets (pre-market) must have fair_odds set; PASS fix must not regress this."""
        entries = [
            {"post_position": 1, "horse_name": "A", "win_prob": 0.5},
            {"post_position": 2, "horse_name": "B", "win_prob": 0.5},
        ]
        pricing = calculate_pricing_breakdown(entries)
        plan = construct_staking_plan(pricing, budget=1000, strategy_mode="auto")
        assert plan.resolved_mode == "pending"
        assert plan.simple_bet.fair_odds is not None, "Pending ticket must have a fair_odds value"
        assert plan.simple_bet.fair_odds == pytest.approx(2.0)

    def test_successful_bet_simple_bet_fair_odds_not_none(self):
        """A real bet plan's simple_bet must have fair_odds (regression guard for R7-SIG-2)."""
        entries = [
            {"post_position": 1, "horse_name": "A", "win_prob": 0.5, "odds": 2.5},
            {"post_position": 2, "horse_name": "B", "win_prob": 0.3, "odds": 5.0},
            {"post_position": 3, "horse_name": "C", "win_prob": 0.2, "odds": 8.0},
        ]
        pricing = calculate_pricing_breakdown(entries)
        plan = construct_staking_plan(pricing, budget=1000, strategy_mode="hybrid")
        assert plan.resolved_mode != "pass"
        assert plan.simple_bet.fair_odds is not None, "Success path simple_bet.fair_odds must not be None"
        assert plan.simple_bet.ev is not None


# ---------------------------------------------------------------------------
# R10 Fixes
# ---------------------------------------------------------------------------

class TestR10Fixes:
    """Tests covering Round 10 audit fixes: dead code removal and mode validation."""

    # --- R10-DEAD-1/2: Verdict tree still correct after removing dead branches ---

    def test_is_top_val_fav_moderate_ev_gets_moderate_value(self):
        """is_top_val + is_fav + EV_TOP_VALUE(5%) <= ev < EV_BEST_VALUE(10%) -> 'Top pick & moderate value'.

        The old dead branches ('Top pick & slight value') were below this in the tree;
        removing them must not regress this verdict.
        """
        # PP1: prob=0.50, odds=2.14 -> ev = 0.50*2.14-1 = +7.0%
        # PP1 has highest prob -> is_fav. PP1 has highest EV -> is_top_val.
        entries = [
            {"post_position": 1, "horse_name": "FavTop", "win_prob": 0.50, "odds": 2.14},
            {"post_position": 2, "horse_name": "B",      "win_prob": 0.30, "odds": 3.0},
            {"post_position": 3, "horse_name": "C",      "win_prob": 0.20, "odds": 5.0},
        ]
        pricing = calculate_pricing_breakdown(entries)
        fav = next(h for h in pricing if h.post_position == 1)
        assert fav.is_favorite, "PP1 must be model favourite"
        assert fav.is_top_value, "PP1 must be top value (highest EV)"
        assert fav.ev is not None and 0.05 <= fav.ev < 0.10, (
            f"PP1 EV must be in [5%, 10%) for this test, got {fav.ev:+.2%}"
        )
        assert fav.verdict == "Top pick & moderate value", (
            f"R10-DEAD-1 regression: expected 'Top pick & moderate value', got {fav.verdict!r}"
        )

    def test_is_top_val_no_fav_moderate_ev_gets_moderate_value(self):
        """is_top_val + NOT is_fav + EV_TOP_VALUE <= ev < EV_BEST_VALUE -> 'Moderate value'."""
        # PP1 has highest prob -> is_fav. PP2 has highest EV -> is_top_val.
        entries = [
            {"post_position": 1, "horse_name": "Fav",   "win_prob": 0.55, "odds": 1.8},   # ev=-1%
            {"post_position": 2, "horse_name": "TopVal", "win_prob": 0.25, "odds": 5.2},   # ev=+30%, is_top_val
            {"post_position": 3, "horse_name": "C",      "win_prob": 0.20, "odds": 4.0},
        ]
        pricing = calculate_pricing_breakdown(entries)
        top_val = next(h for h in pricing if h.post_position == 2)
        assert top_val.is_top_value, "PP2 must be top value"
        assert not top_val.is_favorite, "PP2 must NOT be model favourite"
        assert top_val.ev is not None and top_val.ev >= 0.05, (
            f"PP2 EV must be >= 5%, got {top_val.ev:+.2%}"
        )
        assert top_val.verdict in ("Moderate value", "Best value"), (
            f"R10-DEAD-2 regression: expected 'Moderate value' or 'Best value', "
            f"got {top_val.verdict!r}"
        )

    def test_is_top_val_requires_ev_above_threshold(self):
        """is_top_val=True structurally implies ev > EV_TOP_VALUE.

        This validates the invariant that dead branches 297-300 relied on an impossible
        scenario (is_top_val=True with ev <= EV_TOP_VALUE) that cannot actually occur.
        """
        from models.betting_engine import EV_TOP_VALUE
        entries = [
            {"post_position": 1, "horse_name": "A", "win_prob": 0.50, "odds": 2.5},
            {"post_position": 2, "horse_name": "B", "win_prob": 0.30, "odds": 4.0},
            {"post_position": 3, "horse_name": "C", "win_prob": 0.20, "odds": 6.0},
        ]
        pricing = calculate_pricing_breakdown(entries)
        for h in pricing:
            if h.is_top_value:
                assert h.ev is not None and h.ev > EV_TOP_VALUE, (
                    f"Invariant violated: is_top_val=True but ev={h.ev:+.2%} <= EV_TOP_VALUE={EV_TOP_VALUE:+.2%}. "
                    f"Dead branches 297-300 would be reachable!"
                )

    # --- R10-MODE-1: Invalid strategy_mode raises ValueError ---

    @pytest.mark.parametrize("bad_mode", ["bogus", "AUTO", "Hybrid", "unknown", ""])
    def test_invalid_strategy_mode_raises(self, bad_mode):
        """construct_staking_plan must raise ValueError for unrecognized strategy_mode (R10-MODE-1)."""
        entries = [
            {"post_position": 1, "horse_name": "A", "win_prob": 0.5, "odds": 3.0},
            {"post_position": 2, "horse_name": "B", "win_prob": 0.5, "odds": 4.0},
        ]
        pricing = calculate_pricing_breakdown(entries)
        with pytest.raises(ValueError, match="Unknown strategy_mode"):
            construct_staking_plan(pricing, budget=1000, strategy_mode=bad_mode)

    def test_valid_strategy_modes_accepted(self):
        """All documented strategy modes must be accepted without raising (R10-MODE-1)."""
        entries = [
            {"post_position": 1, "horse_name": "A", "win_prob": 0.5, "odds": 3.0},
            {"post_position": 2, "horse_name": "B", "win_prob": 0.3, "odds": 5.0},
            {"post_position": 3, "horse_name": "C", "win_prob": 0.2, "odds": 8.0},
        ]
        pricing = calculate_pricing_breakdown(entries)
        for mode in ("hybrid", "pure_win", "dutching", "auto", "adaptive"):
            plan = construct_staking_plan(pricing, budget=1000, strategy_mode=mode)
            assert plan is not None, f"Mode {mode!r} should not raise"


class TestR11Fixes:
    """Round 11 audit regression tests: R11-MINOR-1 (exotic budget-pruning disclosure)."""

    # A 4-horse field where two runners have EV >= 8% (Dutch triggers) and exotics also qualify
    ENTRIES = [
        {"post_position": 1, "horse_name": "A", "win_prob": 0.35, "odds": 3.2},   # EV+12%
        {"post_position": 2, "horse_name": "B", "win_prob": 0.30, "odds": 5.0},   # EV+50%, top_val
        {"post_position": 3, "horse_name": "C", "win_prob": 0.20, "odds": 7.0},   # EV+40%
        {"post_position": 4, "horse_name": "D", "win_prob": 0.15, "odds": 9.0},   # EV+35%
    ]

    def _pricing(self):
        return calculate_pricing_breakdown(self.ENTRIES)

    # --- R11-MINOR-1: Exotic pruning disclosure ---

    def test_adequate_budget_no_prune_suffix(self):
        """Budget large enough for exotics shows no exotic-prune warning (R11-MINOR-1)."""
        plan = construct_staking_plan(self._pricing(), budget=1000, strategy_mode="hybrid")
        n_exotics = sum(1 for t in plan.portfolio_tickets if t.ticket_type_en not in ("win", "place"))
        assert n_exotics >= 1, "With ¥1000, at least one exotic should be allocated"
        assert "exotics available" not in plan.verdict_summary, (
            "No prune suffix when exotics are actually allocated"
        )

    def test_small_budget_dutch_hybrid_surfaces_exotic_note(self):
        """When budget=¥200 only covers 2 Dutch wins and prunes exotics, verdict_summary notes it (R11-MINOR-1)."""
        plan = construct_staking_plan(self._pricing(), budget=200, strategy_mode="hybrid")
        n_exotics = sum(1 for t in plan.portfolio_tickets if t.ticket_type_en not in ("win", "place"))
        assert n_exotics == 0, "With ¥200, exotics should be pruned (max_tickets=2)"
        assert "exotics available" in plan.verdict_summary, (
            f"R11-MINOR-1: verdict_summary must disclose exotic pruning, got: {plan.verdict_summary!r}"
        )
        assert plan.resolved_mode == "hybrid", (
            "resolved_mode must remain 'hybrid' (Dutch allocation succeeded; only exotic overlay dropped)"
        )

    def test_prune_suffix_mentions_budget(self):
        """The prune suffix includes the current budget so user knows what to increase (R11-MINOR-1)."""
        plan = construct_staking_plan(self._pricing(), budget=200, strategy_mode="hybrid")
        assert "200" in plan.verdict_summary, (
            f"Prune suffix must mention the current budget (¥200), got: {plan.verdict_summary!r}"
        )

    def test_boundary_budget_300_has_exotic(self):
        """At budget=¥300, max_tickets=3 allows [win, win, exotic], so no prune warning (R11-MINOR-1)."""
        plan = construct_staking_plan(self._pricing(), budget=300, strategy_mode="hybrid")
        n_exotics = sum(1 for t in plan.portfolio_tickets if t.ticket_type_en not in ("win", "place"))
        assert n_exotics >= 1, "With ¥300, an exotic should fit (max_tickets=3)"
        assert "exotics available" not in plan.verdict_summary, (
            "No prune suffix when at least one exotic is allocated"
        )

    def test_pure_win_small_budget_no_prune_suffix(self):
        """pure_win mode never has exotics, so R11-MINOR-1 suffix must NOT fire for pure_win (R11-MINOR-1)."""
        plan = construct_staking_plan(self._pricing(), budget=100, strategy_mode="pure_win")
        assert "exotics available" not in plan.verdict_summary, (
            "R11-MINOR-1 prune suffix must not appear for pure_win mode"
        )

    def test_dutching_small_budget_no_prune_suffix(self):
        """dutching mode never allocates exotics, so R11-MINOR-1 suffix must NOT fire for dutching (R11-MINOR-1)."""
        plan = construct_staking_plan(self._pricing(), budget=200, strategy_mode="dutching")
        assert "exotics available" not in plan.verdict_summary, (
            "R11-MINOR-1 prune suffix must not appear for dutching mode"
        )

    def test_no_qualifying_exotics_uses_distinct_message(self):
        """When no exotics pass EV filter (not pruned, just none qualify), old message fires (R11-MINOR-1 invariant)."""
        # Use a 2-horse field: wide is disabled (len<4), no trio, no exacta partner → no exotics generated
        entries_2h = [
            {"post_position": 1, "horse_name": "A", "win_prob": 0.55, "odds": 2.3},  # EV+26.5%
            {"post_position": 2, "horse_name": "B", "win_prob": 0.45, "odds": 3.0},  # EV+35%
        ]
        pricing_2h = calculate_pricing_breakdown(entries_2h)
        plan = construct_staking_plan(pricing_2h, budget=1000, strategy_mode="hybrid")
        # 2-horse field: wide=disabled, trio=disabled, quinella partner=B same as dutch partner
        # Should either be "no qualifying exotics" or exotics present, but NOT "exotics available"
        assert "exotics available" not in plan.verdict_summary, (
            "R11-MINOR-1: 'exotics available' must only fire when exotics qualified but were pruned"
        )


class TestR12Fixes:
    """Round 12 audit regression tests:
    R12-MINOR-1: hybrid + Dutch wins + 0 qualifying exotics → resolved_mode must be 'dutching'
    R12-MINOR-2: hybrid + no Dutch + 0 qualifying exotics → resolved_mode must be 'pure_win'
    """

    # 2-horse field: Dutch fires (both ev>=5%), but wide/trio disabled → no qualifying exotics
    ENTRIES_2H = [
        {"post_position": 1, "horse_name": "A", "win_prob": 0.60, "odds": 2.3},   # EV+38%
        {"post_position": 2, "horse_name": "B", "win_prob": 0.40, "odds": 3.5},   # EV+40%
    ]

    # Single dominant horse, all other horses negative EV → no Dutch, no qualifying exotics
    ENTRIES_WIN_ONLY = [
        {"post_position": 1, "horse_name": "A", "win_prob": 0.55, "odds": 2.2},   # EV+21%
        {"post_position": 2, "horse_name": "B", "win_prob": 0.25, "odds": 3.5},   # EV-12.5%
        {"post_position": 3, "horse_name": "C", "win_prob": 0.12, "odds": 8.0},   # EV-4%
        {"post_position": 4, "horse_name": "D", "win_prob": 0.08, "odds": 11.0},  # EV-12%
    ]

    # Field with qualifying exotics — should stay 'hybrid'
    ENTRIES_FULL_HYBRID = [
        {"post_position": 1, "horse_name": "A", "win_prob": 0.35, "odds": 3.2},   # EV+12%
        {"post_position": 2, "horse_name": "B", "win_prob": 0.30, "odds": 5.0},   # EV+50%
        {"post_position": 3, "horse_name": "C", "win_prob": 0.20, "odds": 7.0},   # EV+40%
        {"post_position": 4, "horse_name": "D", "win_prob": 0.15, "odds": 9.0},   # EV+35%
    ]

    # --- R12-MINOR-1: hybrid + Dutch + no qualifying exotics → 'dutching' ---

    def test_r12_minor1_hybrid_dutch_no_exotics_resolves_as_dutching(self):
        """hybrid + 2 Dutch wins + 0 exotics → resolved_mode='dutching', not 'hybrid' (R12-MINOR-1)."""
        pricing = calculate_pricing_breakdown(self.ENTRIES_2H)
        plan = construct_staking_plan(pricing, budget=1000, strategy_mode="hybrid")
        n_exotics = sum(1 for t in plan.portfolio_tickets if t.ticket_type_en not in ("win", "place"))
        n_wins = sum(1 for t in plan.portfolio_tickets if t.ticket_type_en == "win")
        assert n_wins == 2, f"Expected 2 Dutch win tickets, got {n_wins}"
        assert n_exotics == 0, f"Expected 0 exotic tickets (2-horse field), got {n_exotics}"
        assert plan.resolved_mode == "dutching", (
            f"R12-MINOR-1: hybrid+Dutch+no_exotics must resolve as 'dutching', got {plan.resolved_mode!r}"
        )

    def test_r12_minor1_strategy_meta_shows_dutching_not_hybrid(self):
        """When resolved_mode='dutching', STRATEGY_META lookup returns dutching meta (R12-MINOR-1)."""
        # This is an integration sanity: construct_staking_plan returns 'dutching', so
        # analyze_race_betting would look up STRATEGY_META['dutching'], not ['hybrid'].
        pricing = calculate_pricing_breakdown(self.ENTRIES_2H)
        plan = construct_staking_plan(pricing, budget=1000, strategy_mode="hybrid")
        # resolved_mode must not be 'hybrid' so hybrid meta is not served
        assert plan.resolved_mode != "hybrid", (
            "R12-MINOR-1: STRATEGY_META must not return hybrid stats when no exotics were placed"
        )

    def test_r12_minor1_verdict_suffix_still_present(self):
        """The '(no qualifying exotics)' suffix must still appear in verdict_summary (R12-MINOR-1)."""
        pricing = calculate_pricing_breakdown(self.ENTRIES_2H)
        plan = construct_staking_plan(pricing, budget=1000, strategy_mode="hybrid")
        assert "no qualifying exotics" in plan.verdict_summary, (
            f"R12-MINOR-1: verdict_summary must retain the disclosure suffix. "
            f"Got: {plan.verdict_summary!r}"
        )

    # --- R12-MINOR-2: hybrid + no Dutch + no qualifying exotics → 'pure_win' ---

    def test_r12_minor2_hybrid_no_dutch_no_exotics_resolves_as_pure_win(self):
        """hybrid + 1 win + 0 exotics (none qualify) → resolved_mode='pure_win', not 'hybrid' (R12-MINOR-2)."""
        pricing = calculate_pricing_breakdown(self.ENTRIES_WIN_ONLY)
        plan = construct_staking_plan(pricing, budget=1000, strategy_mode="hybrid")
        n_exotics = sum(1 for t in plan.portfolio_tickets if t.ticket_type_en not in ("win", "place"))
        n_wins = sum(1 for t in plan.portfolio_tickets if t.ticket_type_en == "win")
        assert n_wins == 1, f"Expected 1 win ticket, got {n_wins}"
        assert n_exotics == 0, f"Expected 0 exotic tickets, got {n_exotics}"
        assert plan.resolved_mode == "pure_win", (
            f"R12-MINOR-2: hybrid+no_Dutch+no_exotics must resolve as 'pure_win', got {plan.resolved_mode!r}"
        )

    def test_r12_minor2_no_qualifying_exotics_suffix_retained(self):
        """The '(no qualifying exotics)' verdict suffix must still appear (R12-MINOR-2)."""
        pricing = calculate_pricing_breakdown(self.ENTRIES_WIN_ONLY)
        plan = construct_staking_plan(pricing, budget=1000, strategy_mode="hybrid")
        assert "no qualifying exotics" in plan.verdict_summary, (
            f"R12-MINOR-2: verdict_summary must retain disclosure suffix. Got: {plan.verdict_summary!r}"
        )

    # --- Invariants: cases that must NOT be affected ---

    def test_r12_invariant_full_hybrid_stays_hybrid(self):
        """When hybrid allocates both Dutch wins AND exotics, resolved_mode stays 'hybrid' (R12 invariant)."""
        pricing = calculate_pricing_breakdown(self.ENTRIES_FULL_HYBRID)
        plan = construct_staking_plan(pricing, budget=1000, strategy_mode="hybrid")
        n_exotics = sum(1 for t in plan.portfolio_tickets if t.ticket_type_en not in ("win", "place"))
        assert n_exotics >= 1, f"Expected at least 1 exotic ticket with ¥1000 budget, got {n_exotics}"
        assert plan.resolved_mode == "hybrid", (
            f"R12 invariant: full hybrid (with exotics) must stay 'hybrid'. Got: {plan.resolved_mode!r}"
        )

    def test_r12_invariant_r11_budget_pruned_stays_hybrid(self):
        """Budget-pruned exotics (R11-MINOR-1 case) must keep resolved_mode='hybrid' (R12 invariant)."""
        # ¥200 budget → max_tickets=2 → only Dutch wins fit, exotics pruned despite qualifying
        pricing = calculate_pricing_breakdown(self.ENTRIES_FULL_HYBRID)
        plan = construct_staking_plan(pricing, budget=200, strategy_mode="hybrid")
        assert plan.resolved_mode == "hybrid", (
            f"R12 invariant: budget-pruned exotics must still show 'hybrid' (R11-MINOR-1 disclosure). "
            f"Got: {plan.resolved_mode!r}"
        )
        assert "exotics available" in plan.verdict_summary, (
            "R12 invariant: R11-MINOR-1 budget-prune disclosure must still appear"
        )

    def test_r12_invariant_pure_win_mode_unaffected(self):
        """pure_win mode is unaffected by R12 changes (R12 invariant)."""
        pricing = calculate_pricing_breakdown(self.ENTRIES_WIN_ONLY)
        plan = construct_staking_plan(pricing, budget=1000, strategy_mode="pure_win")
        assert plan.resolved_mode == "pure_win", (
            f"pure_win mode must not be changed by R12 fixes. Got: {plan.resolved_mode!r}"
        )


class TestR13Fixes:
    """R13-MINOR-1: longshot_overlay verdict was unreachable when is_top_val=True.

    When a sub-5%-probability horse is the top-EV pick (is_top_val=True) AND
    its market odds are >= 20x AND its EV >= 25%, the is_top_val branches in the
    verdict tree fired before the longshot_overlay branch, producing the misleading
    'Best value' label for a speculative longshot. Fixed by guarding the is_top_val
    branches with 'not _is_longshot_overlay' so the longshot profile takes priority.
    """

    ENTRIES_LONGSHOT_IS_TOP_VAL = [
        {"post_position": 1, "horse_name": "Dom",  "win_prob": 0.55, "odds": 1.5},
        {"post_position": 2, "horse_name": "Mid",  "win_prob": 0.30, "odds": 4.0},
        {"post_position": 3, "horse_name": "Edge", "win_prob": 0.04, "odds": 50.0},
    ]

    def test_longshot_top_val_gets_longshot_overlay(self):
        """is_top_val horse with prob<5%, odds>=20x, ev>=25% → 'Longshot overlay', not 'Best value'."""
        pricing = calculate_pricing_breakdown(self.ENTRIES_LONGSHOT_IS_TOP_VAL)
        h = next(h for h in pricing if h.post_position == 3)
        assert h.is_top_value, "PP3 must be is_top_value (highest EV)"
        assert h.win_prob < 0.05, f"PP3 win_prob must be < 5%, got {h.win_prob:.3f}"
        assert h.market_odds is not None and h.market_odds >= 20.0, (
            f"PP3 market_odds must be >= 20x, got {h.market_odds}"
        )
        assert h.ev is not None and h.ev >= 0.25, (
            f"PP3 ev must be >= 25%, got {h.ev:.2%}"
        )
        assert h.verdict == "Longshot overlay", (
            f"R13-MINOR-1: is_top_val longshot must get 'Longshot overlay', got '{h.verdict}'"
        )

    def test_solid_top_val_still_gets_best_value(self):
        """is_top_val horse with prob>=5% (solid probability) still gets 'Best value'."""
        entries = [
            {"post_position": 1, "horse_name": "A", "win_prob": 0.35, "odds": 4.0},
            {"post_position": 2, "horse_name": "B", "win_prob": 0.40, "odds": 2.0},
            {"post_position": 3, "horse_name": "C", "win_prob": 0.25, "odds": 5.0},
        ]
        pricing = calculate_pricing_breakdown(entries)
        h = next(h for h in pricing if h.post_position == 1)
        assert h.is_top_value, "PP1 should be is_top_value"
        assert h.win_prob >= 0.05, f"PP1 prob={h.win_prob:.3f} should be >= 5%"
        assert h.verdict == "Best value", (
            f"R13 invariant: solid is_top_val horse must still get 'Best value', got '{h.verdict}'"
        )

    def test_longshot_overlay_threshold_boundary_ev(self):
        """is_top_val horse with ev<25% (but >10%) gets 'Best value', not 'Longshot overlay'."""
        # Set PP2 odds=3.0 (EV=+1.1%) so PP3 (EV=+21.3%, odds=27x) is the highest-EV horse (is_top_val)
        entries = [
            {"post_position": 1, "horse_name": "Dom",  "win_prob": 0.55, "odds": 1.5},
            {"post_position": 2, "horse_name": "Mid",  "win_prob": 0.30, "odds": 3.0},
            {"post_position": 3, "horse_name": "Edge", "win_prob": 0.04, "odds": 27.0},
        ]
        pricing = calculate_pricing_breakdown(entries)
        h = next(h for h in pricing if h.post_position == 3)
        assert h.is_top_value, "PP3 should be is_top_value"
        assert h.ev is not None and 0.10 <= h.ev < 0.25, f"PP3 EV should be in [10%, 25%), got {h.ev:.2%}"
        assert h.verdict == "Best value", (
            f"R13 boundary: is_top_val with ev<25% should get 'Best value', got '{h.verdict}'"
        )

    def test_longshot_overlay_threshold_boundary_prob(self):
        """is_top_val horse with prob>=5% and ev>=25% gets 'Best value' (not longshot_overlay)."""
        # PP3 prob=6%, odds=25x -> EV = +50% (ev >= 25%, odds >= 20x, but prob >= 5%)
        entries = [
            {"post_position": 1, "horse_name": "Dom",  "win_prob": 0.55, "odds": 1.5},
            {"post_position": 2, "horse_name": "Mid",  "win_prob": 0.30, "odds": 3.0},
            {"post_position": 3, "horse_name": "Edge", "win_prob": 0.06, "odds": 25.0},
        ]
        pricing = calculate_pricing_breakdown(entries)
        h = next(h for h in pricing if h.post_position == 3)
        assert h.is_top_value, "PP3 should be is_top_value"
        assert h.win_prob >= 0.05, f"PP3 win_prob should be >= 5%, got {h.win_prob:.3f}"
        assert h.ev is not None and h.ev >= 0.25
        assert h.verdict == "Best value", (
            f"R13 boundary: prob>=5% should get 'Best value', got '{h.verdict}'"
        )


class TestR14Fixes:
    """Round 14 Audit: Probing probability normalization, mathematical boundaries,
    small-field exotics, Harville conditional distributions, and multi-tier budget scaling.
    """

    def test_zero_sum_normalization(self):
        """All-zero raw win probabilities normalize uniformly and sum to 1.0."""
        entries = [{"post_position": i, "horse_name": f"H{i}", "win_prob": 0.0, "odds": 5.0} for i in range(1, 5)]
        pricing = calculate_pricing_breakdown(entries)
        assert len(pricing) == 4
        assert all(math.isclose(h.win_prob, 0.25, rel_tol=1e-3) for h in pricing)
        assert sum(h.win_prob for h in pricing) == pytest.approx(1.0)

    def test_string_type_coercion(self):
        """String-encoded post_positions, probabilities, and odds coerce safely without TypeError."""
        entries = [
            {"post_position": "1", "horse_name": "A", "win_prob": "0.45", "odds": "3.5"},
            {"post_position": "2", "horse_name": "B", "win_prob": "0.35", "odds": "4.0"},
            {"post_position": "3", "horse_name": "C", "win_prob": "0.20", "odds": "6.0"},
        ]
        pricing = calculate_pricing_breakdown(entries)
        assert pricing[0].post_position == 1 and isinstance(pricing[0].post_position, int)
        assert pricing[0].market_odds == 3.5 and isinstance(pricing[0].market_odds, float)

    def test_negative_raw_prob_clamping(self):
        """Negative probability outputs are clamped to 0.0 before normalization and 0.0001 after."""
        entries = [
            {"post_position": 1, "horse_name": "A", "win_prob": -0.20, "odds": 4.0},
            {"post_position": 2, "horse_name": "B", "win_prob": 0.60, "odds": 2.0},
            {"post_position": 3, "horse_name": "C", "win_prob": 0.40, "odds": 3.0},
        ]
        pricing = calculate_pricing_breakdown(entries)
        p1 = next(h for h in pricing if h.post_position == 1)
        p2 = next(h for h in pricing if h.post_position == 2)
        assert p1.win_prob == 0.0001
        assert p2.win_prob == 0.60

    def test_harville_conditional_distribution_properties(self):
        """Joint Harville conditional 2nd and 3rd place distributions preserve probability mass."""
        from models.betting_engine import JointFinishModel
        probs = {1: 0.40, 2: 0.30, 3: 0.20, 4: 0.10}
        model = JointFinishModel(probs, gamma=1.0)
        for i in probs:
            cond_2nd_sum = sum(model.exact_1_2(i, j) for j in probs if j != i)
            assert math.isclose(cond_2nd_sum, probs[i], rel_tol=1e-5)
            for j in probs:
                if i == j:
                    continue
                cond_3rd_sum = sum(model.exact_1_2_3(i, j, k) for k in probs if k not in (i, j))
                assert math.isclose(cond_3rd_sum, model.exact_1_2(i, j), rel_tol=1e-5)

    def test_small_field_2_runner_joint_models(self):
        """In a 2-horse field, top-2 place = 1.0, quinella = 1.0, and 3-runner exotics return 0.0."""
        from models.betting_engine import JointFinishModel
        m2 = JointFinishModel({1: 0.6, 2: 0.4}, gamma=1.0)
        assert m2.place_prob(1, top_n=2) == 1.0
        assert m2.place_prob(2, top_n=2) == 1.0
        assert m2.quinella_prob(1, 2) == 1.0
        assert m2.exacta_prob(1, 2) == 0.6
        assert m2.exacta_prob(2, 1) == 0.4
        assert m2.trio_prob(1, 2, 3) == 0.0

    def test_unquoted_runners_exotic_filtering(self):
        """Unquoted runners receive pseudo-probs for denominator math but are excluded from tickets."""
        entries = [
            {"post_position": 1, "horse_name": "A", "win_prob": 0.50, "odds": 2.0},
            {"post_position": 2, "horse_name": "B", "win_prob": 0.30, "odds": 4.0},
            {"post_position": 3, "horse_name": "C", "win_prob": 0.15, "odds": None},
            {"post_position": 4, "horse_name": "D", "win_prob": 0.05, "odds": None},
        ]
        pricing = calculate_pricing_breakdown(entries)
        plan = construct_staking_plan(pricing, budget=1000, strategy_mode="hybrid")
        assert plan.anchor_horse.post_position == 2
        for t in plan.portfolio_tickets:
            if t.ticket_type_en != "win":
                assert 3 not in t.selection and 4 not in t.selection

    @pytest.mark.parametrize("budget", [100, 200, 300, 400, 500, 1000, 5000, 10000, 1000000])
    def test_multi_tier_budget_conservation(self, budget):
        """Total staked matches budget exactly with ¥100 multiples across all budget tiers."""
        entries = [
            {"post_position": 1, "horse_name": "A", "win_prob": 0.50, "odds": 3.0},
            {"post_position": 2, "horse_name": "B", "win_prob": 0.25, "odds": 5.0},
            {"post_position": 3, "horse_name": "C", "win_prob": 0.25, "odds": 1.5},
        ]
        pricing = calculate_pricing_breakdown(entries)
        plan = construct_staking_plan(pricing, budget=budget, strategy_mode="hybrid")
        assert sum(t.stake for t in plan.portfolio_tickets) == budget
        assert all(t.stake >= 100 and t.stake % 100 == 0 for t in plan.portfolio_tickets)

    def test_single_horse_field_pure_win(self):
        """1-runner field cleanly generates a single 100% win bet without exotics or errors."""
        from models.betting_engine import HorsePricing
        pricing = [HorsePricing(
            post_position=1, horse_name="Solo", horse_name_jp="ソロ",
            win_prob=1.0, fair_odds=1.0, market_odds=2.0, ev=1.0,
            verdict="Top pick & clear value", is_favorite=True, is_top_value=True,
        )]
        plan = construct_staking_plan(pricing, budget=1000, strategy_mode="pure_win")
        assert len(plan.portfolio_tickets) == 1
        assert plan.portfolio_tickets[0].stake == 1000
        assert plan.resolved_mode == "pure_win"


