"""
UmaEdge — Paddock Comment NLP Scorer.

Scores Japanese paddock comments (パドックコメント) on 5 dimensions
using Google Gemini LLM, with a keyword dictionary fallback.

Usage:
    from models.paddock_nlp import PaddockScorer

    scorer = PaddockScorer()
    result = scorer.score_comment("好馬体で落ち着いている。踏み込み深く毛艶も良い。")
    print(result)  # PaddockScore(build=0.7, temperament=0.5, gait=0.6, coat=0.5, overall=0.58)

    adjusted = adjust_probability(0.12, result)
    print(adjusted)  # ~0.135 (boosted by positive condition)
"""

import json
import logging
import os
from dataclasses import asdict, dataclass
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("paddock_nlp")


# ---------------------------------------------------------------------------
# Score Data Class
# ---------------------------------------------------------------------------

@dataclass
class PaddockScore:
    """Paddock condition score on -1.0 to +1.0 scale per dimension."""
    build: float = 0.0          # 馬体 (physique)
    temperament: float = 0.0    # 気配 (demeanour)
    gait: float = 0.0           # 歩様 (movement quality)
    coat: float = 0.0           # 毛艶 (coat shine)
    overall: float = 0.0        # weighted composite

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "PaddockScore":
        return cls(
            build=float(d.get("build", 0)),
            temperament=float(d.get("temperament", 0)),
            gait=float(d.get("gait", 0)),
            coat=float(d.get("coat", 0)),
            overall=float(d.get("overall", 0)),
        )


# ---------------------------------------------------------------------------
# Keyword Dictionary (Fallback Scorer)
# ---------------------------------------------------------------------------

# Positive keywords → score contribution
POSITIVE_KEYWORDS = {
    # Build (馬体)
    "好馬体": ("build", 0.6),
    "馬体良": ("build", 0.5),
    "筋肉質": ("build", 0.5),
    "張り良い": ("build", 0.5),
    "張り": ("build", 0.3),
    "身体つき": ("build", 0.3),
    "成長": ("build", 0.3),
    "馬体充実": ("build", 0.6),
    "馬体絞れ": ("build", 0.5),
    "ふっくら": ("build", 0.3),

    # Temperament (気配)
    "落ち着き": ("temperament", 0.5),
    "落ち着い": ("temperament", 0.5),
    "リラックス": ("temperament", 0.6),
    "集中": ("temperament", 0.4),
    "穏やか": ("temperament", 0.4),
    "気合い十分": ("temperament", 0.4),
    "気合い乗り": ("temperament", 0.4),
    "堂々": ("temperament", 0.5),

    # Gait (歩様)
    "踏み込み深い": ("gait", 0.6),
    "踏み込み良い": ("gait", 0.5),
    "踏み込み": ("gait", 0.3),
    "弾力": ("gait", 0.5),
    "柔らか": ("gait", 0.4),
    "スムーズ": ("gait", 0.5),
    "キビキビ": ("gait", 0.5),
    "力強い": ("gait", 0.5),
    "軽快": ("gait", 0.4),

    # Coat (毛艶)
    "毛艶良い": ("coat", 0.5),
    "毛艶": ("coat", 0.3),
    "ピカピカ": ("coat", 0.6),
    "艶良い": ("coat", 0.4),
    "艶": ("coat", 0.2),
}

NEGATIVE_KEYWORDS = {
    # Build
    "太め残り": ("build", -0.5),
    "太め": ("build", -0.4),
    "太い": ("build", -0.3),
    "細め": ("build", -0.4),
    "細い": ("build", -0.3),
    "腹回り": ("build", -0.3),
    "余裕残し": ("build", -0.3),
    "馬体減": ("build", -0.4),
    "緩い": ("build", -0.3),

    # Temperament
    "入れ込み": ("temperament", -0.7),
    "入れ込む": ("temperament", -0.6),
    "チャカつき": ("temperament", -0.6),
    "チャカつく": ("temperament", -0.5),
    "テンション高い": ("temperament", -0.5),
    "テンション": ("temperament", -0.3),
    "発汗多い": ("temperament", -0.6),
    "発汗": ("temperament", -0.4),
    "汗": ("temperament", -0.3),
    "煩い": ("temperament", -0.4),
    "うるさい": ("temperament", -0.4),
    "暴れ": ("temperament", -0.7),
    "イレ込み": ("temperament", -0.6),

    # Gait
    "硬い": ("gait", -0.4),
    "硬め": ("gait", -0.3),
    "左前甘い": ("gait", -0.5),
    "右前甘い": ("gait", -0.5),
    "甘い": ("gait", -0.3),
    "ぎこちない": ("gait", -0.5),
    "重い": ("gait", -0.4),
    "トモ甘い": ("gait", -0.4),

    # Coat
    "毛艶冴えない": ("coat", -0.5),
    "冴えない": ("coat", -0.4),
    "粗い": ("coat", -0.3),
    "フケ": ("coat", -0.4),
}

# Dimension weights for overall score
DIMENSION_WEIGHTS = {
    "build": 0.30,
    "temperament": 0.30,
    "gait": 0.25,
    "coat": 0.15,
}


# ---------------------------------------------------------------------------
# Gemini Prompt
# ---------------------------------------------------------------------------

SCORING_PROMPT = """あなたは日本競馬のパドック解説エキスパートです。
以下のパドックコメントを読み、馬のコンディションを5つの観点で-1.0〜+1.0のスコアで評価してください。

観点:
- build (馬体): 筋肉の張り、体型、仕上がり。+1.0=理想的な馬体、-1.0=太め残りや細すぎ
- temperament (気配): 落ち着き、精神状態。+1.0=リラックスして集中、-1.0=入れ込みや発汗
- gait (歩様): 歩き方の質、踏み込み。+1.0=弾力あり深い踏み込み、-1.0=硬くぎこちない
- coat (毛艶): 毛の艶。+1.0=ピカピカ、-1.0=冴えない
- overall: 上記の加重平均（build 30%, temperament 30%, gait 25%, coat 15%）

パドックコメント:
{comment}

JSONのみで回答してください（説明不要）:
{{"build": 0.0, "temperament": 0.0, "gait": 0.0, "coat": 0.0, "overall": 0.0}}"""


# ---------------------------------------------------------------------------
# Scorer
# ---------------------------------------------------------------------------

class PaddockScorer:
    """Score paddock comments using Gemini LLM with keyword fallback."""

    def __init__(self, use_llm: bool = True):
        """
        Args:
            use_llm: If True, try Gemini first, fall back to keywords on failure.
                     If False, always use keyword-based scoring.
        """
        self.use_llm = use_llm
        self._api_key = os.getenv("GOOGLE_API_KEY")

    def score_comment(self, comment: str) -> PaddockScore:
        """
        Score a single paddock comment.

        Args:
            comment: Japanese paddock comment text.

        Returns:
            PaddockScore with -1.0 to +1.0 scores per dimension.
        """
        if not comment or not comment.strip():
            return PaddockScore()

        # Try LLM first
        if self.use_llm and self._api_key and self._api_key != "your_gemini_api_key_here":
            try:
                return self._score_with_gemini(comment)
            except Exception as e:
                log.warning(f"Gemini scoring failed, falling back to keywords: {e}")

        # Keyword fallback
        return self._keyword_fallback(comment)

    def score_batch(
        self,
        comments: list[dict],
    ) -> list[dict]:
        """
        Score a batch of paddock comments.

        Args:
            comments: List of dicts with keys:
                - horse_id or entry_id
                - comment (str): The paddock comment text

        Returns:
            List of dicts with original keys + 'score' (PaddockScore).
        """
        results = []
        for item in comments:
            text = item.get("comment", "")
            score = self.score_comment(text)
            results.append({**item, "score": score})
        return results

    def _score_with_gemini(self, comment: str) -> PaddockScore:
        """Score using Google Gemini API."""
        from google import genai

        client = genai.Client(api_key=self._api_key)
        prompt = SCORING_PROMPT.format(comment=comment)

        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
            config={
                "temperature": 0.1,  # Low temp for consistent scoring
                "max_output_tokens": 256,
            },
        )

        text_response = response.text.strip()

        # Strip markdown code fences if present
        if text_response.startswith("```"):
            lines = text_response.split("\n")
            text_response = "\n".join(lines[1:-1])

        data = json.loads(text_response)

        # Validate and clamp scores
        score = PaddockScore(
            build=max(-1.0, min(1.0, float(data.get("build", 0)))),
            temperament=max(-1.0, min(1.0, float(data.get("temperament", 0)))),
            gait=max(-1.0, min(1.0, float(data.get("gait", 0)))),
            coat=max(-1.0, min(1.0, float(data.get("coat", 0)))),
            overall=max(-1.0, min(1.0, float(data.get("overall", 0)))),
        )

        log.info(f"Gemini scored: overall={score.overall:+.2f}")
        return score

    def _keyword_fallback(self, comment: str) -> PaddockScore:
        """Score using keyword dictionary matching."""
        dimension_scores: dict[str, list[float]] = {
            "build": [],
            "temperament": [],
            "gait": [],
            "coat": [],
        }

        # Check positive keywords (longest match first)
        for keyword, (dim, value) in sorted(
            POSITIVE_KEYWORDS.items(), key=lambda x: -len(x[0])
        ):
            if keyword in comment:
                dimension_scores[dim].append(value)

        # Check negative keywords
        for keyword, (dim, value) in sorted(
            NEGATIVE_KEYWORDS.items(), key=lambda x: -len(x[0])
        ):
            if keyword in comment:
                dimension_scores[dim].append(value)

        # Average per dimension, default 0.0
        build = _avg(dimension_scores["build"])
        temperament = _avg(dimension_scores["temperament"])
        gait = _avg(dimension_scores["gait"])
        coat = _avg(dimension_scores["coat"])

        # Weighted overall
        overall = (
            build * DIMENSION_WEIGHTS["build"]
            + temperament * DIMENSION_WEIGHTS["temperament"]
            + gait * DIMENSION_WEIGHTS["gait"]
            + coat * DIMENSION_WEIGHTS["coat"]
        )

        return PaddockScore(
            build=max(-1.0, min(1.0, build)),
            temperament=max(-1.0, min(1.0, temperament)),
            gait=max(-1.0, min(1.0, gait)),
            coat=max(-1.0, min(1.0, coat)),
            overall=max(-1.0, min(1.0, overall)),
        )


def _avg(values: list[float]) -> float:
    """Average a list of floats, returning 0.0 if empty."""
    return sum(values) / len(values) if values else 0.0


# ---------------------------------------------------------------------------
# Probability Adjustment
# ---------------------------------------------------------------------------

def adjust_probability(
    win_prob: float,
    score: PaddockScore,
    max_adjustment: float = 0.05,
    threshold: float = 0.3,
) -> float:
    """
    Adjust a win probability based on paddock condition score.

    The adjustment is proportional to the overall score and capped
    at ±max_adjustment (default 5%).

    Args:
        win_prob: Original win probability (0-1).
        score: PaddockScore from the scorer.
        max_adjustment: Maximum absolute probability adjustment.
        threshold: Minimum |overall| score to trigger an adjustment.

    Returns:
        Adjusted win probability, clamped to [0.001, 0.999].
    """
    if abs(score.overall) < threshold:
        return win_prob

    # Scale: map overall [-1, +1] to adjustment [-max_adj, +max_adj]
    # Only apply beyond threshold
    effective = score.overall - threshold if score.overall > 0 else score.overall + threshold
    adjustment = effective / (1.0 - threshold) * max_adjustment

    adjusted = win_prob + adjustment
    return max(0.001, min(0.999, adjusted))


# ---------------------------------------------------------------------------
# DB — Store / Retrieve
# ---------------------------------------------------------------------------

def store_paddock_score(
    entry_id: int,
    comment: str,
    score: PaddockScore,
    source: str = "manual",
) -> Optional[int]:
    """Store a paddock comment and score to the database."""
    try:
        from database import get_session
        from sqlalchemy import text

        with get_session() as session:
            result = session.execute(
                text("""
                    INSERT INTO horsebet.paddock_comments
                        (entry_id, source, comment_text,
                         score_build, score_temperament, score_gait,
                         score_coat, score_overall)
                    VALUES (:entry_id, :source, :comment,
                            :build, :temperament, :gait, :coat, :overall)
                    RETURNING id
                """),
                {
                    "entry_id": entry_id,
                    "source": source,
                    "comment": comment,
                    "build": score.build,
                    "temperament": score.temperament,
                    "gait": score.gait,
                    "coat": score.coat,
                    "overall": score.overall,
                },
            )
            row = result.fetchone()
            session.commit()
            return row[0] if row else None
    except Exception as e:
        log.error(f"Failed to store paddock score: {e}")
        return None


def get_paddock_score(entry_id: int) -> Optional[PaddockScore]:
    """Retrieve the most recent paddock score for an entry."""
    try:
        from database import get_session
        from sqlalchemy import text

        with get_session() as session:
            row = session.execute(
                text("""
                    SELECT score_build, score_temperament, score_gait,
                           score_coat, score_overall
                    FROM horsebet.paddock_comments
                    WHERE entry_id = :entry_id
                    ORDER BY scored_at DESC
                    LIMIT 1
                """),
                {"entry_id": entry_id},
            ).fetchone()

            if row:
                return PaddockScore(
                    build=row[0] or 0.0,
                    temperament=row[1] or 0.0,
                    gait=row[2] or 0.0,
                    coat=row[3] or 0.0,
                    overall=row[4] or 0.0,
                )
            return None
    except Exception as e:
        log.error(f"Failed to retrieve paddock score: {e}")
        return None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Score paddock comments")
    parser.add_argument("comment", nargs="?", help="Paddock comment text to score")
    parser.add_argument("--no-llm", action="store_true", help="Use keyword fallback only")
    parser.add_argument("--examples", action="store_true", help="Run example comments")
    args = parser.parse_args()

    scorer = PaddockScorer(use_llm=not args.no_llm)

    if args.examples:
        examples = [
            "好馬体で落ち着いている。踏み込み深く毛艶も良い。仕上がり万全。",
            "太め残りで入れ込み気味。発汗多く硬い歩様。",
            "まずまずの馬体。やや気合い乗り良好。",
            "細めだが毛艶ピカピカ。リラックスしてキビキビ歩く。",
            "入れ込みチャカつき暴れる。歩様ぎこちなく毛艶冴えない。",
        ]
        print("\n🐴 Paddock Comment Scoring Examples")
        print("=" * 70)
        for comment in examples:
            score = scorer.score_comment(comment)
            print(f"\n  Comment: {comment}")
            print(f"  Build:       {score.build:+.2f}")
            print(f"  Temperament: {score.temperament:+.2f}")
            print(f"  Gait:        {score.gait:+.2f}")
            print(f"  Coat:        {score.coat:+.2f}")
            print(f"  Overall:     {score.overall:+.2f}")

            # Show probability adjustment
            base_prob = 0.10
            adj_prob = adjust_probability(base_prob, score)
            if adj_prob != base_prob:
                print(f"  Win probability: {base_prob:.1%} → {adj_prob:.1%}")
            else:
                print(f"  Win probability: {base_prob:.1%} (no adjustment)")
        print("=" * 70)

    elif args.comment:
        score = scorer.score_comment(args.comment)
        print(f"\n  Build:       {score.build:+.2f}")
        print(f"  Temperament: {score.temperament:+.2f}")
        print(f"  Gait:        {score.gait:+.2f}")
        print(f"  Coat:        {score.coat:+.2f}")
        print(f"  Overall:     {score.overall:+.2f}")
    else:
        parser.print_help()
