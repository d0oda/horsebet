import pytest
import pandas as pd
import numpy as np
from models.ability_rating import AbilityRatingEngine, _parse_margin, DEFAULT_RATING, RATING_MIN, RATING_MAX

class TestAbilityRatingAndSectionalsBrutal:
    """Stress tests ability rating computations, margin parsing edge cases, and EWMA stability."""

    def test_margin_parser_all_formats(self):
        # Winner (None / NaN / empty)
        assert _parse_margin(None) == 0.0
        assert _parse_margin(float("nan")) == 0.0
        assert _parse_margin("") == 0.0

        # Japanese and English special notations
        assert _parse_margin("ハナ") == 0.05
        assert _parse_margin("クビ") == 0.25
        assert _parse_margin("アタマ") == 0.1
        assert _parse_margin("大") == 10.0
        assert _parse_margin("同着") == 0.0
        assert _parse_margin("NS") == 0.05
        assert _parse_margin("NK") == 0.25
        assert _parse_margin("HD") == 0.1
        assert _parse_margin("DS") == 10.0
        assert _parse_margin("DH") == 0.0

        # Fractions and mixed numbers
        assert _parse_margin("1/2") == 0.5
        assert _parse_margin("3/4") == 0.75
        assert _parse_margin("1 1/2") == 1.5
        assert _parse_margin("2.1/2") == 2.5
        assert _parse_margin("3 1/4") == 3.25
        assert _parse_margin("5") == 5.0
        assert _parse_margin("2.5") == 2.5
        assert _parse_margin("1/2 +") == 0.5

        # Garbage input fallback
        assert _parse_margin("INVALID_TEXT") == 0.0

    def test_ability_engine_rating_bounds(self):
        engine = AbilityRatingEngine()
        
        # Test performance score calculation with extreme times
        # Extremely fast time (world record)
        score_fast = engine.compute_race_score(
            time_secs=50.0,
            distance=1200,
            course_id=1,
            going="good",
            race_class="G1",
            weight_carried=57.0,
            field_avg_weight=57.0,
            margin="0",
            finish_pos=1,
            field_size=16,
            par_time=70.0,
            par_std=2.0,
        )
        assert "total" in score_fast
        assert score_fast["total"] <= RATING_MAX

        # Extremely slow time
        score_slow = engine.compute_race_score(
            time_secs=200.0,
            distance=1200,
            course_id=1,
            going="yielding",
            race_class="未勝利",
            weight_carried=57.0,
            field_avg_weight=57.0,
            margin="50",
            finish_pos=16,
            field_size=16,
            par_time=70.0,
            par_std=2.0,
        )
        assert "total" in score_slow
        assert score_slow["total"] >= RATING_MIN

    def test_ewma_update_mathematical_bounds(self):
        # 1. New horse (None initial rating)
        r1 = AbilityRatingEngine.update_rating(None, 85.0)
        assert r1 == 85.0

        # 2. Sequential convergence: horse runs 100 consecutive 100-rating races
        curr = 50.0
        for _ in range(100):
            curr = AbilityRatingEngine.update_rating(curr, 100.0)
        assert abs(curr - 100.0) < 1e-4

        # 3. Rating clamping
        r_high = AbilityRatingEngine.update_rating(140.0, 300.0)
        assert r_high <= RATING_MAX
        r_low = AbilityRatingEngine.update_rating(30.0, -100.0)
        assert r_low >= RATING_MIN
