"""
UmaEdge — Paddock Comment NLP Scorer.

Uses Google Gemini to score Japanese paddock observation comments on four
dimensions (build, temperament, gait, coat) and compute a composite
condition score.  Scores are stored in the paddock_comments table and
can be used as a real-time prediction adjustment on race day.

Usage:
    from models.paddock_scorer import score_race_paddock, get_paddock_scores

    # Score all unscored comments for a race
    score_race_paddock(race_id=12345)

    # Get composite scores for prediction adjustment
    scores = get_paddock_scores(race_id=12345)
    # → {entry_id: 0.45, entry_id2: -0.3, ...}

CLI:
    # Score comments for a specific race
    python -m models.paddock_scorer --race-id 12345

    # Insert + score a single comment
    python -m models.paddock_scorer --entry-id 5678 --comment "馬体良好、落ち着きあり"
"""

import argparse
import json
import logging
import os
from typing import Optional

from sqlalchemy import text

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger("paddock_scorer")


# ---------------------------------------------------------------------------
# Gemini Prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an expert Japanese horse racing paddock analyst.
You receive paddock observation comments (パドックコメント) for horses before a race.
Your job is to score each horse's physical condition on 4 dimensions.

For each horse, output a JSON object with these scores (each -1.0 to +1.0):

- score_build: Physique / muscle tone / body condition (馬体)
  +1.0 = exceptional build, muscular, powerful
  0.0 = normal / average
  -1.0 = poor build, thin, dull

- score_temperament: Demeanour / calmness / focus (気配)
  +1.0 = calm, focused, relaxed
  0.0 = normal
  -1.0 = sweating, agitated, fighting handler

- score_gait: Walk / stride quality (歩様)
  +1.0 = smooth, powerful stride
  0.0 = normal gait
  -1.0 = stiff, limping, short stride

- score_coat: Coat shine / sweat (毛ヅヤ)
  +1.0 = gleaming coat, healthy sheen
  0.0 = normal
  -1.0 = dull coat, excessive sweat

Return a JSON array of objects. Each object must have:
{"post_position": <int>, "score_build": <float>, "score_temperament": <float>, "score_gait": <float>, "score_coat": <float>}

Only return the JSON array, no explanation."""


def _build_prompt(comments: dict[int, str]) -> str:
    """Build the scoring prompt for a batch of horses.

    Args:
        comments: {post_position: comment_text}
    """
    lines = ["Score the following paddock comments:\n"]
    for pp, text_str in sorted(comments.items()):
        lines.append(f"馬番{pp}: {text_str}")
    return SYSTEM_PROMPT + "\n\n" + "\n".join(lines)


# ---------------------------------------------------------------------------
# Gemini API
# ---------------------------------------------------------------------------

def _call_gemini(prompt: str) -> Optional[list[dict]]:
    """Call Gemini and parse the JSON response."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key or api_key in ("your_gemini_api_key_here", ""):
        log.warning("GOOGLE_API_KEY not set — cannot score paddock comments")
        return None

    try:
        from google import genai

        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=prompt,
            config={
                "temperature": 0.1,  # Low temp for consistent scoring
                "max_output_tokens": 2048,
            },
        )

        text_response = response.text.strip()

        # Strip markdown code fences if present
        if text_response.startswith("```"):
            lines = text_response.split("\n")
            text_response = "\n".join(lines[1:-1])

        return json.loads(text_response)

    except json.JSONDecodeError as e:
        log.error(f"Failed to parse Gemini JSON: {e}")
        log.debug(f"Raw response: {text_response[:500]}")
        return None
    except Exception as e:
        log.error(f"Gemini API call failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Score Validation
# ---------------------------------------------------------------------------

SCORE_FIELDS = ["score_build", "score_temperament", "score_gait", "score_coat"]
SCORE_WEIGHTS = {
    "score_build": 0.30,
    "score_temperament": 0.30,
    "score_gait": 0.25,
    "score_coat": 0.15,
}


def _validate_scores(scores: list[dict]) -> list[dict]:
    """Validate and clamp returned scores to [-1.0, +1.0]."""
    validated = []
    for item in scores:
        pp = item.get("post_position")
        if pp is None:
            continue
        clean = {"post_position": int(pp)}
        for field in SCORE_FIELDS:
            val = item.get(field, 0.0)
            try:
                val = float(val)
            except (TypeError, ValueError):
                val = 0.0
            clean[field] = max(-1.0, min(1.0, val))

        # Compute weighted composite
        clean["score_overall"] = sum(
            clean[f] * SCORE_WEIGHTS[f] for f in SCORE_FIELDS
        )
        clean["score_overall"] = max(-1.0, min(1.0, clean["score_overall"]))
        validated.append(clean)
    return validated


# ---------------------------------------------------------------------------
# Score a single comment (without DB)
# ---------------------------------------------------------------------------

def score_comment(comment_text: str, post_position: int = 1) -> Optional[dict]:
    """Score a single paddock comment. Returns a dict with all score fields."""
    prompt = _build_prompt({post_position: comment_text})
    raw = _call_gemini(prompt)
    if not raw:
        return None
    validated = _validate_scores(raw)
    if not validated:
        return None
    return validated[0]


# ---------------------------------------------------------------------------
# Database Operations
# ---------------------------------------------------------------------------

def score_race_paddock(race_id: int) -> int:
    """
    Score all unscored paddock comments for a race.

    Finds comments in paddock_comments that have comment_text but no
    score_overall, sends them to Gemini in a batch, and saves the scores.

    Returns:
        Number of comments scored.
    """
    from scraper.db import get_session

    with get_session() as session:
        # Get unscored comments with their entry's post_position
        rows = session.execute(
            text("""
                SELECT pc.id, pc.entry_id, e.post_position, pc.comment_text
                FROM horsebet.paddock_comments pc
                JOIN horsebet.entries e ON e.id = pc.entry_id
                WHERE e.race_id = :race_id
                  AND pc.comment_text IS NOT NULL
                  AND pc.comment_text != ''
                  AND pc.score_overall IS NULL
                ORDER BY e.post_position
            """),
            {"race_id": race_id},
        ).fetchall()

    if not rows:
        log.info(f"No unscored paddock comments for race {race_id}")
        return 0

    # Build comment map: post_position → text
    comments = {}
    row_map = {}  # post_position → (pc_id, entry_id)
    for r in rows:
        pp = r.post_position or 0
        comments[pp] = r.comment_text
        row_map[pp] = (r.id, r.entry_id)

    log.info(f"Scoring {len(comments)} paddock comments for race {race_id}")

    # Call Gemini
    prompt = _build_prompt(comments)
    raw = _call_gemini(prompt)
    if not raw:
        log.error(f"Gemini returned no scores for race {race_id}")
        return 0

    validated = _validate_scores(raw)

    # Save scores to DB
    scored = 0
    with get_session() as session:
        for item in validated:
            pp = item["post_position"]
            if pp not in row_map:
                continue
            pc_id, _ = row_map[pp]
            session.execute(
                text("""
                    UPDATE horsebet.paddock_comments
                    SET score_build = :sb, score_temperament = :st,
                        score_gait = :sg, score_coat = :sc,
                        score_overall = :so, scored_at = NOW()
                    WHERE id = :id
                """),
                {
                    "sb": item["score_build"],
                    "st": item["score_temperament"],
                    "sg": item["score_gait"],
                    "sc": item["score_coat"],
                    "so": item["score_overall"],
                    "id": pc_id,
                },
            )
            scored += 1
        session.commit()

    log.info(f"✅ Scored {scored}/{len(comments)} paddock comments for race {race_id}")
    return scored


def get_paddock_scores(race_id: int) -> dict[int, float]:
    """
    Load paddock scores for a race.

    Returns:
        {entry_id: score_overall} for all scored entries in the race.
    """
    from scraper.db import get_session

    with get_session() as session:
        rows = session.execute(
            text("""
                SELECT pc.entry_id, pc.score_overall
                FROM horsebet.paddock_comments pc
                JOIN horsebet.entries e ON e.id = pc.entry_id
                WHERE e.race_id = :race_id
                  AND pc.score_overall IS NOT NULL
            """),
            {"race_id": race_id},
        ).fetchall()

    return {r.entry_id: r.score_overall for r in rows}


def insert_comment(entry_id: int, comment_text: str, source: str = "manual") -> int:
    """Insert a paddock comment into the DB. Returns the new row ID."""
    from scraper.db import get_session

    with get_session() as session:
        result = session.execute(
            text("""
                INSERT INTO horsebet.paddock_comments (entry_id, comment_text, source)
                VALUES (:eid, :text, :src)
                RETURNING id
            """),
            {"eid": entry_id, "text": comment_text, "src": source},
        )
        session.commit()
        return result.fetchone().id


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="UmaEdge — Paddock Comment NLP Scorer",
    )
    parser.add_argument("--race-id", type=int, help="Score all unscored comments for this race")
    parser.add_argument("--entry-id", type=int, help="Insert + score a single comment")
    parser.add_argument("--comment", type=str, help="Comment text (with --entry-id)")
    args = parser.parse_args()

    if args.entry_id and args.comment:
        pc_id = insert_comment(args.entry_id, args.comment)
        log.info(f"Inserted paddock comment (id={pc_id})")

        # Score it
        scores = score_comment(args.comment)
        if scores:
            print("\n🐴 Paddock Score:")
            for field in SCORE_FIELDS:
                label = field.replace("score_", "").capitalize()
                val = scores[field]
                bar = "█" * int((val + 1) * 5) + "░" * (10 - int((val + 1) * 5))
                print(f"  {label:<13} [{bar}] {val:+.2f}")
            print(f"  {'Overall':<13} [{'':<10}] {scores['score_overall']:+.2f}")

    elif args.race_id:
        scored = score_race_paddock(args.race_id)
        if scored == 0:
            print("No comments to score (or all already scored)")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
