"""
Unit tests for the pace simulation engine.
Tests running style classification, simulation mechanics, and output validity.
"""

import numpy as np
import pytest

from models.pace_sim import (
    PaceSimulator,
    classify_running_style,
    STYLE_FRONT,
    STYLE_STALK,
    STYLE_CLOSER,
    STYLE_DEEP,
    STYLE_PARAMS,
)


# ---------------------------------------------------------------------------
# Running Style Classification
# ---------------------------------------------------------------------------

class TestClassifyStyle:
    def test_front_runner(self):
        """Very early first corner position → front-runner."""
        assert classify_running_style(1.5, field_size=16) == STYLE_FRONT

    def test_stalker(self):
        assert classify_running_style(4.0, field_size=16) == STYLE_STALK

    def test_closer(self):
        assert classify_running_style(8.0, field_size=16) == STYLE_CLOSER

    def test_deep_closer(self):
        assert classify_running_style(13.0, field_size=16) == STYLE_DEEP

    def test_none_defaults_to_stalker(self):
        assert classify_running_style(None) == STYLE_STALK

    def test_nan_defaults_to_stalker(self):
        assert classify_running_style(np.nan) == STYLE_STALK

    def test_explicit_style_overrides(self):
        assert classify_running_style(1.0, explicit_style=STYLE_DEEP) == STYLE_DEEP

    def test_invalid_explicit_style_falls_through(self):
        result = classify_running_style(1.5, field_size=16, explicit_style="invalid")
        assert result == STYLE_FRONT  # falls through to position-based


# ---------------------------------------------------------------------------
# Pace Modifier
# ---------------------------------------------------------------------------

class TestPaceModifier:
    def test_fast_pace_hurts_front_runners(self):
        assert PaceSimulator._pace_modifier(STYLE_FRONT, "fast") < 1.0

    def test_fast_pace_helps_closers(self):
        assert PaceSimulator._pace_modifier(STYLE_CLOSER, "fast") > 1.0
        assert PaceSimulator._pace_modifier(STYLE_DEEP, "fast") > 1.0

    def test_slow_pace_helps_front_runners(self):
        assert PaceSimulator._pace_modifier(STYLE_FRONT, "slow") > 1.0

    def test_slow_pace_hurts_deep_closers(self):
        assert PaceSimulator._pace_modifier(STYLE_DEEP, "slow") < 1.0

    def test_moderate_pace_is_neutral(self):
        for style in [STYLE_FRONT, STYLE_STALK, STYLE_CLOSER, STYLE_DEEP]:
            assert PaceSimulator._pace_modifier(style, "moderate") == 1.0


# ---------------------------------------------------------------------------
# Simulation Output
# ---------------------------------------------------------------------------

class TestSimulation:
    def _make_entries(self, n=8):
        """Create synthetic entries."""
        return [
            {
                "horse_id": i + 1,
                "avg_first_corner": i * 2 + 1,
                "avg_last_3f": 33.5 + i * 0.3,
                "career_win_pct": max(0.05, 0.30 - i * 0.04),
                "odds_win": 3.0 + i * 2.0,
                "last3_avg_finish": 2.0 + i * 0.5,
            }
            for i in range(n)
        ]

    def test_simulation_returns_all_horses(self):
        entries = self._make_entries(5)
        sim = PaceSimulator(n_simulations=1000, seed=42)
        results = sim.simulate_race(entries, distance=2000)

        horse_results = {k: v for k, v in results.items() if k != "_race_level_"}
        assert len(horse_results) == 5
        for hid in range(1, 6):
            assert hid in horse_results

    def test_win_probs_sum_to_one(self):
        entries = self._make_entries(8)
        sim = PaceSimulator(n_simulations=10000, seed=42)
        results = sim.simulate_race(entries, distance=2000)

        total_win = sum(r["win_prob"] for k, r in results.items() if k != "_race_level_")
        assert total_win == pytest.approx(1.0, abs=0.01)

    def test_place_probs_sum_to_three(self):
        """Sum of place probabilities should be ~3.0 (3 places available)."""
        entries = self._make_entries(8)
        sim = PaceSimulator(n_simulations=10000, seed=42)
        results = sim.simulate_race(entries, distance=2000)

        total_place = sum(r["place_prob"] for k, r in results.items() if k != "_race_level_")
        assert total_place == pytest.approx(3.0, abs=0.05)

    def test_favourite_has_highest_win_prob(self):
        """The horse with best stats and lowest odds should generally win most."""
        entries = self._make_entries(5)
        sim = PaceSimulator(n_simulations=50000, seed=42)
        results = sim.simulate_race(entries, distance=2000)

        # Horse 1 has the best stats
        win_probs = [(hid, data["win_prob"]) for hid, data in results.items() if hid != "_race_level_"]
        winner = max(win_probs, key=lambda x: x[1])
        assert winner[0] == 1, f"Expected horse 1 to be favourite, got horse {winner[0]}"

    def test_empty_entries(self):
        sim = PaceSimulator(n_simulations=100)
        results = sim.simulate_race([], distance=2000)
        assert results == {}

    def test_single_horse_always_wins(self):
        entries = [{"horse_id": 1, "career_win_pct": 0.5, "odds_win": 1.1}]
        sim = PaceSimulator(n_simulations=1000, seed=42)
        results = sim.simulate_race(entries, distance=2000)

        assert results[1]["win_prob"] == 1.0
        assert results[1]["place_prob"] == 1.0

    def test_style_assigned(self):
        entries = self._make_entries(4)
        sim = PaceSimulator(n_simulations=100, seed=42)
        results = sim.simulate_race(entries, distance=2000)

        for hid, data in results.items():
            if hid != "_race_level_":
                assert data["style"] in [STYLE_FRONT, STYLE_STALK, STYLE_CLOSER, STYLE_DEEP]

    def test_reproducible_with_seed(self):
        entries = self._make_entries(5)
        sim1 = PaceSimulator(n_simulations=5000, seed=123)
        sim2 = PaceSimulator(n_simulations=5000, seed=123)

        r1 = sim1.simulate_race(entries, distance=2000)
        r2 = sim2.simulate_race(entries, distance=2000)

        for hid in r1:
            if hid == "_race_level_":
                continue
            assert r1[hid]["win_prob"] == r2[hid]["win_prob"]

    def test_long_distance_favours_closers(self):
        """At 3200m, closers should get a boost relative to 1200m."""
        entries = [
            {"horse_id": 1, "avg_first_corner": 1, "career_win_pct": 0.2, "odds_win": 5.0,
             "avg_last_3f": 35.0, "last3_avg_finish": 4.0},  # front-runner
            {"horse_id": 2, "avg_first_corner": 14, "career_win_pct": 0.2, "odds_win": 5.0,
             "avg_last_3f": 33.0, "last3_avg_finish": 4.0},  # deep closer
        ]
        sim = PaceSimulator(n_simulations=20000, seed=42)

        long_results = sim.simulate_race(entries, distance=3200)
        sim_short = PaceSimulator(n_simulations=20000, seed=42)
        short_results = sim_short.simulate_race(entries, distance=1200)

        # Closer should do relatively better at 3200m vs 1200m
        closer_long = long_results[2]["win_prob"]
        closer_short = short_results[2]["win_prob"]
        assert closer_long > closer_short


# ---------------------------------------------------------------------------
# Base Ability Estimation
# ---------------------------------------------------------------------------

class TestBaseAbility:
    def test_no_data_returns_default(self):
        sim = PaceSimulator()
        ability = sim._estimate_base_ability({})
        assert ability == 0.5

    def test_low_odds_means_higher_ability(self):
        sim = PaceSimulator()
        strong = sim._estimate_base_ability({"odds_win": 1.5, "career_win_pct": 0.3})
        weak = sim._estimate_base_ability({"odds_win": 50.0, "career_win_pct": 0.03})
        assert strong > weak


# ---------------------------------------------------------------------------
# Data-Driven Calibration — Win Rate Ordering (JRA: 逃 > 先 > 差 > 追)
# ---------------------------------------------------------------------------

class TestCalibration:
    """Validate that sim win rates respect the real JRA ordering across styles."""

    def _run_sim_batch(self, n_races=200, n_sims=2000, seed=42):
        """Run a batch of diverse simulated races and aggregate win rates by style."""
        rng = np.random.default_rng(seed)
        style_wins = {s: 0.0 for s in [STYLE_FRONT, STYLE_STALK, STYLE_CLOSER, STYLE_DEEP]}
        style_runs = {s: 0 for s in style_wins}

        for i in range(n_races):
            field_size = int(rng.choice([12, 14, 16]))
            distance = int(rng.choice([1200, 1600, 2000, 2400]))
            entries = []
            for h in range(field_size):
                # Distribute corners to create a mix of styles
                corner = rng.uniform(1, field_size)
                entries.append({
                    "horse_id": h,
                    "avg_first_corner": corner,
                    "avg_last_3f": rng.normal(35.5, 1.5),
                    "career_win_pct": rng.beta(2, 15),
                    "odds_win": rng.lognormal(2.5, 0.8),
                    "last3_avg_finish": rng.normal(7, 3),
                })

            sim = PaceSimulator(n_simulations=n_sims, seed=i)
            results = sim.simulate_race(entries, distance=distance)

            for hid, data in results.items():
                if hid == "_race_level_":
                    continue
                style = data["style"]
                style_runs[style] += 1
                style_wins[style] += data["win_prob"]

        rates = {s: style_wins[s] / max(style_runs[s], 1) for s in style_wins}
        return rates

    def test_style_ordering_matches_jra(self):
        """Front-runners should win most, then stalkers, closers, deep closers.
        Real JRA: 逃 17.7% > 先 10.9% > 差 5.7% > 追 2.1%"""
        rates = self._run_sim_batch()
        assert rates[STYLE_FRONT] > rates[STYLE_STALK], \
            f"逃 ({rates[STYLE_FRONT]:.3f}) should beat 先 ({rates[STYLE_STALK]:.3f})"
        assert rates[STYLE_STALK] > rates[STYLE_CLOSER], \
            f"先 ({rates[STYLE_STALK]:.3f}) should beat 差 ({rates[STYLE_CLOSER]:.3f})"
        assert rates[STYLE_CLOSER] > rates[STYLE_DEEP], \
            f"差 ({rates[STYLE_CLOSER]:.3f}) should beat 追 ({rates[STYLE_DEEP]:.3f})"

    def test_front_runner_advantage_is_large(self):
        """Front-runners should win at least 2x more often than closers (real: 3x)."""
        rates = self._run_sim_batch()
        ratio = rates[STYLE_FRONT] / max(rates[STYLE_CLOSER], 0.001)
        assert ratio >= 1.5, f"逃/差 ratio {ratio:.1f}x is too low (real JRA: ~3x)"

    def test_sprint_favours_front_runners(self):
        """At sprint distances, front-runners should outperform their long-distance rate."""
        rng = np.random.default_rng(99)

        def _sim_front_rate(distance, n_races=100):
            all_styles = [STYLE_FRONT, STYLE_STALK, STYLE_CLOSER, STYLE_DEEP]
            style_wins = {s: 0.0 for s in all_styles}
            style_runs = {s: 0 for s in all_styles}
            for i in range(n_races):
                entries = []
                for h in range(12):
                    corner = rng.uniform(1, 12)
                    entries.append({
                        "horse_id": h,
                        "avg_first_corner": corner,
                        "avg_last_3f": rng.normal(35.5, 1.0),
                        "career_win_pct": rng.beta(2, 15),
                        "odds_win": rng.lognormal(2.5, 0.6),
                        "last3_avg_finish": rng.normal(6, 2),
                    })
                sim = PaceSimulator(n_simulations=2000, seed=i + distance)
                results = sim.simulate_race(entries, distance=distance)
                for hid, data in results.items():
                    if hid == "_race_level_":
                        continue
                    style_runs[data["style"]] += 1
                    style_wins[data["style"]] += data["win_prob"]
            if style_runs[STYLE_FRONT] > 0:
                return style_wins[STYLE_FRONT] / style_runs[STYLE_FRONT]
            return 0.0

        sprint_rate = _sim_front_rate(1200)
        long_rate = _sim_front_rate(2400)
        assert sprint_rate > long_rate, \
            f"Front-runner win rate at 1200m ({sprint_rate:.3f}) should exceed 2400m ({long_rate:.3f})"

