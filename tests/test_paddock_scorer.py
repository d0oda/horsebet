"""
Unit tests for the paddock comment NLP scorer.
Tests prompt construction, score validation, and DB operations.
"""

import json
from unittest.mock import patch, MagicMock

import pytest

from models.paddock_scorer import (
    _build_prompt,
    _validate_scores,
    score_comment,
    SCORE_FIELDS,
    SCORE_WEIGHTS,
    SYSTEM_PROMPT,
)


# ---------------------------------------------------------------------------
# Prompt Construction
# ---------------------------------------------------------------------------

class TestBuildPrompt:
    def test_single_horse(self):
        prompt = _build_prompt({1: "馬体良好、落ち着きあり"})
        assert "馬番1" in prompt
        assert "馬体良好" in prompt
        assert SYSTEM_PROMPT in prompt

    def test_multiple_horses_sorted(self):
        prompt = _build_prompt({3: "歩様硬い", 1: "好馬体", 2: "テンション高い"})
        # Horse 1 should appear before horse 3
        idx_1 = prompt.index("馬番1")
        idx_2 = prompt.index("馬番2")
        idx_3 = prompt.index("馬番3")
        assert idx_1 < idx_2 < idx_3

    def test_prompt_includes_scoring_instructions(self):
        prompt = _build_prompt({1: "テスト"})
        assert "score_build" in prompt
        assert "score_temperament" in prompt
        assert "score_gait" in prompt
        assert "score_coat" in prompt
        assert "-1.0" in prompt
        assert "+1.0" in prompt


# ---------------------------------------------------------------------------
# Score Validation
# ---------------------------------------------------------------------------

class TestValidateScores:
    def test_valid_scores_pass_through(self):
        raw = [{"post_position": 1, "score_build": 0.5, "score_temperament": 0.3,
                "score_gait": -0.2, "score_coat": 0.1}]
        result = _validate_scores(raw)
        assert len(result) == 1
        assert result[0]["score_build"] == 0.5
        assert result[0]["score_temperament"] == 0.3

    def test_scores_clamped_to_range(self):
        raw = [{"post_position": 1, "score_build": 1.5, "score_temperament": -2.0,
                "score_gait": 0.0, "score_coat": 0.0}]
        result = _validate_scores(raw)
        assert result[0]["score_build"] == 1.0
        assert result[0]["score_temperament"] == -1.0

    def test_overall_computed(self):
        raw = [{"post_position": 1, "score_build": 1.0, "score_temperament": 1.0,
                "score_gait": 1.0, "score_coat": 1.0}]
        result = _validate_scores(raw)
        assert result[0]["score_overall"] == pytest.approx(1.0, abs=0.01)

    def test_overall_weighted_correctly(self):
        raw = [{"post_position": 1, "score_build": 1.0, "score_temperament": 0.0,
                "score_gait": 0.0, "score_coat": 0.0}]
        result = _validate_scores(raw)
        # build weight = 0.30
        assert result[0]["score_overall"] == pytest.approx(0.30, abs=0.01)

    def test_missing_post_position_skipped(self):
        raw = [{"score_build": 0.5, "score_temperament": 0.3,
                "score_gait": 0.0, "score_coat": 0.0}]
        result = _validate_scores(raw)
        assert len(result) == 0

    def test_missing_score_defaults_to_zero(self):
        raw = [{"post_position": 1}]
        result = _validate_scores(raw)
        assert result[0]["score_build"] == 0.0

    def test_invalid_score_type_defaults_to_zero(self):
        raw = [{"post_position": 1, "score_build": "excellent",
                "score_temperament": None, "score_gait": 0.5, "score_coat": 0.5}]
        result = _validate_scores(raw)
        assert result[0]["score_build"] == 0.0
        assert result[0]["score_temperament"] == 0.0

    def test_weights_sum_to_one(self):
        total = sum(SCORE_WEIGHTS.values())
        assert total == pytest.approx(1.0, abs=0.001)

    def test_multiple_horses(self):
        raw = [
            {"post_position": 1, "score_build": 0.8, "score_temperament": 0.5,
             "score_gait": 0.3, "score_coat": 0.6},
            {"post_position": 5, "score_build": -0.5, "score_temperament": -0.7,
             "score_gait": -0.3, "score_coat": -0.4},
        ]
        result = _validate_scores(raw)
        assert len(result) == 2
        assert result[0]["post_position"] == 1
        assert result[1]["post_position"] == 5
        # Horse 1 should have positive overall, horse 5 negative
        assert result[0]["score_overall"] > 0
        assert result[1]["score_overall"] < 0


# ---------------------------------------------------------------------------
# Score Comment (with mocked Gemini)
# ---------------------------------------------------------------------------

class TestScoreComment:
    @patch("models.paddock_scorer._call_gemini")
    def test_returns_scores_on_success(self, mock_gemini):
        mock_gemini.return_value = [
            {"post_position": 1, "score_build": 0.7, "score_temperament": 0.5,
             "score_gait": 0.4, "score_coat": 0.6}
        ]
        result = score_comment("馬体良好、落ち着きあり", post_position=1)
        assert result is not None
        assert result["score_build"] == 0.7
        assert "score_overall" in result

    @patch("models.paddock_scorer._call_gemini")
    def test_returns_none_on_api_failure(self, mock_gemini):
        mock_gemini.return_value = None
        result = score_comment("テスト")
        assert result is None

    @patch("models.paddock_scorer._call_gemini")
    def test_returns_none_on_empty_response(self, mock_gemini):
        mock_gemini.return_value = []
        result = score_comment("テスト")
        assert result is None


# ---------------------------------------------------------------------------
# Score Semantics — Positive vs Negative Comments
# ---------------------------------------------------------------------------

class TestScoreSemantics:
    """Verify that known-good/bad comments produce expected score directions
    when using the real prompt structure (tested via validation only)."""

    def test_positive_comment_format(self):
        """A manually-scored positive comment should have positive overall."""
        scores = _validate_scores([{
            "post_position": 1,
            "score_build": 0.8,
            "score_temperament": 0.7,
            "score_gait": 0.6,
            "score_coat": 0.5,
        }])
        assert scores[0]["score_overall"] > 0.5

    def test_negative_comment_format(self):
        """A manually-scored negative comment should have negative overall."""
        scores = _validate_scores([{
            "post_position": 1,
            "score_build": -0.6,
            "score_temperament": -0.8,
            "score_gait": -0.5,
            "score_coat": -0.3,
        }])
        assert scores[0]["score_overall"] < -0.4
