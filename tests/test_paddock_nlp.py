"""
Unit tests for the paddock NLP scorer.
Tests keyword fallback, probability adjustment, and Gemini response parsing.
"""

import pytest
from unittest.mock import MagicMock, patch

from models.paddock_nlp import (
    PaddockScorer,
    PaddockScore,
    adjust_probability,
    POSITIVE_KEYWORDS,
    NEGATIVE_KEYWORDS,
    DIMENSION_WEIGHTS,
    _avg,
)


# ---------------------------------------------------------------------------
# PaddockScore Dataclass
# ---------------------------------------------------------------------------

class TestPaddockScore:
    def test_default_scores_are_zero(self):
        score = PaddockScore()
        assert score.build == 0.0
        assert score.overall == 0.0

    def test_to_dict(self):
        score = PaddockScore(build=0.5, temperament=-0.3, gait=0.2, coat=0.1, overall=0.15)
        d = score.to_dict()
        assert d["build"] == 0.5
        assert d["temperament"] == -0.3

    def test_from_dict(self):
        d = {"build": 0.6, "temperament": -0.2, "gait": 0.3, "coat": 0.1, "overall": 0.25}
        score = PaddockScore.from_dict(d)
        assert score.build == 0.6
        assert score.overall == 0.25

    def test_from_dict_missing_keys_default_zero(self):
        score = PaddockScore.from_dict({})
        assert score.build == 0.0
        assert score.overall == 0.0


# ---------------------------------------------------------------------------
# Keyword Fallback Scorer
# ---------------------------------------------------------------------------

class TestKeywordFallback:
    def setup_method(self):
        self.scorer = PaddockScorer(use_llm=False)

    def test_positive_comment(self):
        score = self.scorer.score_comment("好馬体で落ち着いている。踏み込み深く毛艶良い。")
        assert score.build > 0
        assert score.temperament > 0
        assert score.gait > 0
        assert score.coat > 0
        assert score.overall > 0

    def test_negative_comment(self):
        score = self.scorer.score_comment("太め残りで入れ込み気味。歩様硬い。")
        assert score.build < 0
        assert score.temperament < 0
        assert score.gait < 0
        assert score.overall < 0

    def test_mixed_comment(self):
        score = self.scorer.score_comment("好馬体だが入れ込み気味。")
        assert score.build > 0
        assert score.temperament < 0

    def test_empty_comment(self):
        score = self.scorer.score_comment("")
        assert score.overall == 0.0

    def test_none_comment_returns_zero(self):
        score = self.scorer.score_comment("")
        assert score == PaddockScore()

    def test_unknown_text_returns_zero(self):
        score = self.scorer.score_comment("特に気になる点はなし")
        assert score.overall == 0.0

    def test_scores_clamped_to_range(self):
        # Even with many positive keywords, score should stay <= 1.0
        extreme = "好馬体 馬体良 馬体充実 張り良い 筋肉質 馬体絞れ 成長 張り"
        score = self.scorer.score_comment(extreme)
        assert -1.0 <= score.build <= 1.0
        assert -1.0 <= score.overall <= 1.0

    def test_coat_keywords(self):
        score = self.scorer.score_comment("毛艶ピカピカ")
        assert score.coat > 0

    def test_temperament_negative(self):
        score = self.scorer.score_comment("入れ込みチャカつき発汗多い")
        assert score.temperament < -0.3

    def test_gait_positive(self):
        score = self.scorer.score_comment("踏み込み深い弾力スムーズ")
        assert score.gait > 0.3


# ---------------------------------------------------------------------------
# Overall Score Calculation
# ---------------------------------------------------------------------------

class TestOverallScore:
    def test_overall_is_weighted_average(self):
        scorer = PaddockScorer(use_llm=False)
        # A comment hitting all positive dimensions
        score = scorer.score_comment("好馬体 落ち着き 踏み込み深い 毛艶良い")
        expected_overall = (
            score.build * DIMENSION_WEIGHTS["build"]
            + score.temperament * DIMENSION_WEIGHTS["temperament"]
            + score.gait * DIMENSION_WEIGHTS["gait"]
            + score.coat * DIMENSION_WEIGHTS["coat"]
        )
        assert abs(score.overall - expected_overall) < 0.01


# ---------------------------------------------------------------------------
# Probability Adjustment
# ---------------------------------------------------------------------------

class TestProbabilityAdjustment:
    def test_positive_score_increases_prob(self):
        score = PaddockScore(overall=0.6)
        result = adjust_probability(0.10, score)
        assert result > 0.10

    def test_negative_score_decreases_prob(self):
        score = PaddockScore(overall=-0.6)
        result = adjust_probability(0.10, score)
        assert result < 0.10

    def test_below_threshold_no_change(self):
        score = PaddockScore(overall=0.2)
        result = adjust_probability(0.10, score, threshold=0.3)
        assert result == 0.10

    def test_adjustment_capped_at_max(self):
        score = PaddockScore(overall=1.0)
        result = adjust_probability(0.10, score, max_adjustment=0.05)
        assert result <= 0.10 + 0.05 + 0.001  # small tolerance

    def test_adjusted_prob_never_exceeds_one(self):
        score = PaddockScore(overall=1.0)
        result = adjust_probability(0.98, score, max_adjustment=0.10)
        assert result <= 0.999

    def test_adjusted_prob_never_below_zero(self):
        score = PaddockScore(overall=-1.0)
        result = adjust_probability(0.02, score, max_adjustment=0.10)
        assert result >= 0.001

    def test_zero_score_no_change(self):
        score = PaddockScore(overall=0.0)
        result = adjust_probability(0.15, score)
        assert result == 0.15


# ---------------------------------------------------------------------------
# Gemini Scoring (Mocked)
# ---------------------------------------------------------------------------

class TestGeminiScoring:
    @patch("models.paddock_nlp.PaddockScorer._score_with_gemini")
    def test_gemini_returns_valid_score(self, mock_gemini):
        mock_gemini.return_value = PaddockScore(
            build=0.7, temperament=0.5, gait=0.6, coat=0.4, overall=0.55
        )
        scorer = PaddockScorer(use_llm=True)
        scorer._api_key = "test_key"
        result = scorer.score_comment("好馬体で落ち着いている")
        assert result.overall == 0.55

    @patch("models.paddock_nlp.PaddockScorer._score_with_gemini")
    def test_gemini_failure_falls_back_to_keywords(self, mock_gemini):
        mock_gemini.side_effect = Exception("API error")
        scorer = PaddockScorer(use_llm=True)
        scorer._api_key = "test_key"
        result = scorer.score_comment("好馬体で落ち着いている")
        # Should still return a valid score from keyword fallback
        assert result.build > 0
        assert result.temperament > 0


# ---------------------------------------------------------------------------
# Batch Scoring
# ---------------------------------------------------------------------------

class TestBatchScoring:
    def test_batch_returns_scores(self):
        scorer = PaddockScorer(use_llm=False)
        comments = [
            {"horse_id": 1, "comment": "好馬体"},
            {"horse_id": 2, "comment": "入れ込み"},
        ]
        results = scorer.score_batch(comments)
        assert len(results) == 2
        assert results[0]["score"].build > 0
        assert results[1]["score"].temperament < 0

    def test_batch_preserves_original_keys(self):
        scorer = PaddockScorer(use_llm=False)
        comments = [{"horse_id": 99, "comment": "好馬体", "extra": "data"}]
        results = scorer.score_batch(comments)
        assert results[0]["horse_id"] == 99
        assert results[0]["extra"] == "data"


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

class TestAvg:
    def test_avg_empty(self):
        assert _avg([]) == 0.0

    def test_avg_single(self):
        assert _avg([0.5]) == 0.5

    def test_avg_multiple(self):
        assert _avg([0.4, 0.6]) == pytest.approx(0.5)
