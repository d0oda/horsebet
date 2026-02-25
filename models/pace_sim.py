"""
UmaEdge — Monte Carlo Pace Simulation Engine.

Simulates race outcomes by modelling running styles, pace tempo,
and energy depletion. Outputs P(win) and P(top-3) per horse
conditioned on thousands of simulated race scenarios.

Usage:
    from models.pace_sim import PaceSimulator

    sim = PaceSimulator()
    probs = sim.simulate_race(race_entries)
    # probs = {horse_id: {"win_prob": 0.15, "place_prob": 0.42}, ...}
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("pace_sim")

# ---------------------------------------------------------------------------
# Running Style Classification
# ---------------------------------------------------------------------------

STYLE_FRONT = "逃"   # Front-runner (逃げ)
STYLE_STALK = "先"   # Stalker (先行)
STYLE_CLOSER = "差"  # Closer (差し)
STYLE_DEEP = "追"    # Deep closer (追込)

STYLE_PARAMS = {
    # style: (early_speed_mean, early_speed_std, closing_power_mean, closing_power_std)
    STYLE_FRONT:  (0.90, 0.04, 0.70, 0.08),
    STYLE_STALK:  (0.80, 0.05, 0.80, 0.07),
    STYLE_CLOSER: (0.65, 0.06, 0.90, 0.06),
    STYLE_DEEP:   (0.55, 0.07, 0.95, 0.05),
}


def classify_running_style(
    avg_first_corner: Optional[float],
    field_size: int = 16,
    explicit_style: Optional[str] = None,
) -> str:
    """
    Classify a horse's running style based on their average first corner position.
    If an explicit style is given (from past results), use that.
    """
    if explicit_style and explicit_style in STYLE_PARAMS:
        return explicit_style

    if avg_first_corner is None or np.isnan(avg_first_corner):
        return STYLE_STALK  # default

    # Normalise corner position to relative position (0-1 in field)
    relative_pos = avg_first_corner / max(field_size, 1)

    if relative_pos <= 0.15:
        return STYLE_FRONT
    elif relative_pos <= 0.35:
        return STYLE_STALK
    elif relative_pos <= 0.65:
        return STYLE_CLOSER
    else:
        return STYLE_DEEP


# ---------------------------------------------------------------------------
# Pace Simulation Engine
# ---------------------------------------------------------------------------

class PaceSimulator:
    """Monte Carlo race simulator based on running styles and pace dynamics."""

    def __init__(self, n_simulations: int = 10000, seed: int = 42):
        self.n_simulations = n_simulations
        self.rng = np.random.default_rng(seed)

    def simulate_race(
        self,
        entries: list[dict],
        distance: int = 2000,
    ) -> dict:
        """
        Simulate a race N times and return win/place probabilities.

        Args:
            entries: List of dicts with keys:
                - horse_id (int)
                - avg_first_corner (float): Average first corner position
                - avg_last_3f (float): Average last 3F time in seconds
                - best_last_3f (float): Best last 3F time
                - last3_avg_finish (float): Average finish position in last 3
                - career_win_pct (float): Career win percentage
                - odds_win (float): Market odds
                - running_style (str, optional): Explicit style
            distance: Race distance in metres

        Returns:
            Dict of {horse_id: {"win_prob": float, "place_prob": float, "style": str}}
        """
        n_horses = len(entries)
        if n_horses == 0:
            return {}

        # Classify styles and build parameter arrays
        styles = []
        base_abilities = []

        for e in entries:
            style = classify_running_style(
                avg_first_corner=e.get("avg_first_corner"),
                field_size=n_horses,
                explicit_style=e.get("running_style"),
            )
            styles.append(style)

            # Base ability from career stats + market odds
            ability = self._estimate_base_ability(e)
            base_abilities.append(ability)

        base_abilities = np.array(base_abilities)

        # Determine pace scenario weights
        n_front = sum(1 for s in styles if s == STYLE_FRONT)
        pace_fast_prob = min(0.2 + n_front * 0.15, 0.7)  # More front-runners → faster pace
        pace_slow_prob = max(0.3 - n_front * 0.1, 0.1)
        pace_mod_prob = 1.0 - pace_fast_prob - pace_slow_prob

        # Run simulations
        win_counts = np.zeros(n_horses)
        place_counts = np.zeros(n_horses)

        for _ in range(self.n_simulations):
            # Sample pace scenario
            pace = self.rng.choice(
                ["fast", "moderate", "slow"],
                p=[pace_fast_prob, pace_mod_prob, pace_slow_prob],
            )

            # Compute performance for each horse
            performances = np.zeros(n_horses)
            for i, (style, ability) in enumerate(zip(styles, base_abilities)):
                performances[i] = self._simulate_horse_performance(
                    style=style,
                    base_ability=ability,
                    pace=pace,
                    distance=distance,
                )

            # Rank (higher performance = better finish)
            rankings = np.argsort(-performances)  # descending
            win_counts[rankings[0]] += 1
            for j in range(min(3, n_horses)):
                place_counts[rankings[j]] += 1

        # Convert to probabilities
        results = {}
        for i, entry in enumerate(entries):
            results[entry["horse_id"]] = {
                "win_prob": win_counts[i] / self.n_simulations,
                "place_prob": place_counts[i] / self.n_simulations,
                "style": styles[i],
            }

        return results

    def _estimate_base_ability(self, entry: dict) -> float:
        """
        Estimate a horse's base ability from available metrics.
        Returns a normalised ability score (higher = better).
        """
        signals = []

        # Career win percentage (strong signal)
        win_pct = entry.get("career_win_pct")
        if win_pct is not None and not np.isnan(win_pct):
            signals.append(("win_pct", win_pct * 3.0, 2.0))  # (name, value, weight)

        # Recent form
        last3 = entry.get("last3_avg_finish")
        if last3 is not None and not np.isnan(last3):
            # Lower finish = better, invert and normalise
            signals.append(("form", max(0, 1.0 - last3 / 18.0), 1.5))

        # Closing speed (lower 3F = better)
        last_3f = entry.get("avg_last_3f")
        if last_3f is not None and not np.isnan(last_3f) and last_3f > 0:
            # Normalise: 33s = great, 38s = poor
            speed = max(0, (38.0 - last_3f) / 5.0)
            signals.append(("speed", speed, 1.5))

        # Market odds (lower = market thinks better)
        odds = entry.get("odds_win")
        if odds is not None and not np.isnan(odds) and odds > 0:
            market_signal = 1.0 / odds  # implied probability
            signals.append(("market", market_signal * 5.0, 2.0))

        if not signals:
            return 0.5  # no data → average

        weighted_sum = sum(v * w for _, v, w in signals)
        total_weight = sum(w for _, _, w in signals)
        return weighted_sum / total_weight

    def _simulate_horse_performance(
        self,
        style: str,
        base_ability: float,
        pace: str,
        distance: int,
    ) -> float:
        """
        Simulate a single horse's performance in a race.
        Returns a performance score (higher = finishes closer to first).
        """
        params = STYLE_PARAMS.get(style, STYLE_PARAMS[STYLE_STALK])
        early_mean, early_std, close_mean, close_std = params

        # Sample early speed and closing power
        early_speed = self.rng.normal(early_mean, early_std)
        closing_power = self.rng.normal(close_mean, close_std)

        # Pace effect
        pace_modifier = self._pace_modifier(style, pace)

        # Distance effect — longer races favour closers
        dist_modifier = 1.0
        if distance >= 2400:
            if style in (STYLE_CLOSER, STYLE_DEEP):
                dist_modifier = 1.05
            elif style == STYLE_FRONT:
                dist_modifier = 0.92
        elif distance <= 1200:
            if style in (STYLE_FRONT, STYLE_STALK):
                dist_modifier = 1.05
            elif style == STYLE_DEEP:
                dist_modifier = 0.90

        # Energy depletion — front-runners tire more
        energy = 1.0
        if style == STYLE_FRONT:
            energy = self.rng.normal(0.88, 0.06)
        elif style == STYLE_STALK:
            energy = self.rng.normal(0.93, 0.04)
        else:
            energy = self.rng.normal(0.97, 0.03)

        # Final performance
        performance = (
            base_ability
            * (early_speed * 0.3 + closing_power * 0.7)
            * pace_modifier
            * dist_modifier
            * energy
            + self.rng.normal(0, 0.03)  # random noise
        )

        return max(0, performance)

    @staticmethod
    def _pace_modifier(style: str, pace: str) -> float:
        """
        How pace scenario affects different running styles.
        Fast pace benefits closers; slow pace benefits front-runners.
        """
        modifiers = {
            ("fast", STYLE_FRONT):   0.85,  # Front-runners struggle in fast pace
            ("fast", STYLE_STALK):   0.95,
            ("fast", STYLE_CLOSER):  1.10,  # Closers benefit
            ("fast", STYLE_DEEP):    1.15,
            ("moderate", STYLE_FRONT):  1.00,
            ("moderate", STYLE_STALK):  1.00,
            ("moderate", STYLE_CLOSER): 1.00,
            ("moderate", STYLE_DEEP):   1.00,
            ("slow", STYLE_FRONT):   1.15,  # Front-runners benefit
            ("slow", STYLE_STALK):   1.05,
            ("slow", STYLE_CLOSER):  0.90,
            ("slow", STYLE_DEEP):    0.85,  # Deep closers can't catch up
        }
        return modifiers.get((pace, style), 1.0)


# ---------------------------------------------------------------------------
# CLI — Run simulation on a race
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Example with mock data
    test_entries = [
        {"horse_id": 1, "avg_first_corner": 2, "avg_last_3f": 34.0, "career_win_pct": 0.25, "odds_win": 3.5, "last3_avg_finish": 3.0},
        {"horse_id": 2, "avg_first_corner": 5, "avg_last_3f": 33.5, "career_win_pct": 0.20, "odds_win": 5.0, "last3_avg_finish": 4.0},
        {"horse_id": 3, "avg_first_corner": 10, "avg_last_3f": 33.0, "career_win_pct": 0.30, "odds_win": 2.5, "last3_avg_finish": 2.5},
        {"horse_id": 4, "avg_first_corner": 14, "avg_last_3f": 33.2, "career_win_pct": 0.15, "odds_win": 8.0, "last3_avg_finish": 5.0},
        {"horse_id": 5, "avg_first_corner": 1, "avg_last_3f": 35.0, "career_win_pct": 0.18, "odds_win": 12.0, "last3_avg_finish": 6.0},
    ]

    sim = PaceSimulator(n_simulations=50000)
    results = sim.simulate_race(test_entries, distance=2000)

    print("\n🏇 Pace Simulation Results (50k iterations)")
    print("-" * 55)
    print(f"{'Horse':>6}  {'Style':>4}  {'Win%':>6}  {'Place%':>7}  {'Odds':>5}")
    print("-" * 55)

    for hid, data in sorted(results.items(), key=lambda x: -x[1]["win_prob"]):
        entry = next(e for e in test_entries if e["horse_id"] == hid)
        print(
            f"  #{hid:>3}   {data['style']:>4}  "
            f"{data['win_prob']*100:>5.1f}%  "
            f"{data['place_prob']*100:>6.1f}%  "
            f"{entry['odds_win']:>5.1f}"
        )
