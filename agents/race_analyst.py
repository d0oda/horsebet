"""
UmaEdge — Race Analyst Agent.

LLM-powered agent (Google Gemini) that generates structured,
bilingual (JP + EN) race analysis including pace forecasts,
value plays, horses to watch, and risk factors.

Usage:
    # Generate analysis for a race
    python -m agents.race_analyst --race-id 1

    # Japanese only
    python -m agents.race_analyst --race-id 1 --lang ja

    # Don't store to DB
    python -m agents.race_analyst --race-id 1 --no-store
"""

import argparse
import json
import logging
import os
from typing import Optional

from dotenv import load_dotenv
from sqlalchemy import text

from scraper.db import get_session

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("race_analyst")


# ---------------------------------------------------------------------------
# Analysis Schema
# ---------------------------------------------------------------------------

ANALYSIS_SCHEMA = {
    "type": "object",
    "properties": {
        "race_summary": {
            "type": "string",
            "description": "1-2 sentence race overview"
        },
        "pace_forecast": {
            "type": "object",
            "properties": {
                "likely_tempo": {"type": "string", "enum": ["fast", "moderate", "slow"]},
                "leaders": {"type": "array", "items": {"type": "string"}},
                "scenario": {"type": "string"},
            },
        },
        "horses_to_watch": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "horse_name": {"type": "string"},
                    "post_position": {"type": "integer"},
                    "rating": {"type": "string", "enum": ["★★★", "★★", "★"]},
                    "reason": {"type": "string"},
                },
            },
        },
        "value_plays": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "horse_name": {"type": "string"},
                    "model_prob": {"type": "number"},
                    "market_prob": {"type": "number"},
                    "ev": {"type": "number"},
                    "odds": {"type": "number"},
                    "reasoning": {"type": "string"},
                },
            },
        },
        "risk_factors": {
            "type": "array",
            "items": {"type": "string"},
        },
        "recommended_strategy": {"type": "string"},
    },
}


# ---------------------------------------------------------------------------
# Data Loader
# ---------------------------------------------------------------------------

def load_race_context(race_id: int) -> dict:
    """
    Load full race context from the DB: race info, entries,
    predictions, pace sim results, and odds.
    """
    context = {"race_id": race_id}

    with get_session() as session:
        # Race metadata
        race = session.execute(text("""
            SELECT r.race_name_jp, r.race_name, r.date, r.distance, r.surface,
                   r.going, r.class, r.grade, r.weather, r.field_size,
                   c.name_jp AS course_name
            FROM races r
            LEFT JOIN courses c ON c.id = r.course_id
            WHERE r.id = :rid
        """), {"rid": race_id}).fetchone()

        if not race:
            return context

        context["race"] = {
            "name_jp": race[0],
            "name": race[1],
            "date": str(race[2]),
            "distance": race[3],
            "surface": race[4],
            "going": race[5],
            "class": race[6],
            "grade": race[7],
            "weather": race[8],
            "field_size": race[9],
            "course": race[10],
        }

        # Entries with predictions
        entries = session.execute(text("""
            SELECT
                e.post_position, h.name_jp, h.name, h.sex, h.birth_year,
                j.name_jp AS jockey, t.name_jp AS trainer,
                e.weight_carried, e.horse_weight, e.horse_weight_change,
                e.odds_win, e.popularity, e.draw,
                p.win_prob, p.place_prob,
                res.finish_pos, res.last_3f_secs, res.running_style,
                res.corner_positions
            FROM entries e
            JOIN horses h ON h.id = e.horse_id
            LEFT JOIN jockeys j ON j.id = e.jockey_id
            LEFT JOIN trainers t ON t.id = h.trainer_id
            LEFT JOIN predictions p ON p.entry_id = e.id
            LEFT JOIN results res ON res.entry_id = e.id
            WHERE e.race_id = :rid
            ORDER BY e.post_position
        """), {"rid": race_id}).fetchall()

        context["entries"] = []
        for e in entries:
            context["entries"].append({
                "post": e[0],
                "name_jp": e[1],
                "name": e[2],
                "sex": e[3],
                "birth_year": e[4],
                "jockey": e[5],
                "trainer": e[6],
                "weight_carried": e[7],
                "horse_weight": e[8],
                "weight_change": e[9],
                "odds": e[10],
                "popularity": e[11],
                "draw": e[12],
                "model_win_prob": float(e[13]) if e[13] else None,
                "model_place_prob": float(e[14]) if e[14] else None,
                "finish_pos": e[15],
                "last_3f": e[16],
                "running_style": e[17],
                "corner_positions": e[18],
            })

        # Value bets
        vb = session.execute(text("""
            SELECT entry_id, model_prob, market_prob, ev, kelly_fraction, recommended_stake
            FROM value_bets
            WHERE race_id = :rid
            ORDER BY ev DESC
        """), {"rid": race_id}).fetchall()

        context["value_bets"] = [
            {
                "entry_id": v[0],
                "model_prob": float(v[1]),
                "market_prob": float(v[2]),
                "ev": float(v[3]),
                "kelly": float(v[4]) if v[4] else 0,
                "stake": v[5],
            }
            for v in vb
        ]

    return context


# ---------------------------------------------------------------------------
# Prompt Builder
# ---------------------------------------------------------------------------

def build_prompt(context: dict, language: str = "en") -> str:
    """Build the structured prompt for the race analyst LLM."""
    race = context.get("race", {})
    entries = context.get("entries", [])
    value_bets = context.get("value_bets", [])

    lang_instruction = (
        "Respond entirely in Japanese (日本語で回答してください)."
        if language == "ja"
        else "Respond in English."
    )

    # Format entries table
    entries_table = []
    for e in entries:
        prob_str = f"{e['model_win_prob']:.1%}" if e.get("model_win_prob") else "N/A"
        odds_str = f"{e['odds']:.1f}" if e.get("odds") else "N/A"
        entries_table.append(
            f"  #{e['post']} {e.get('name_jp', e.get('name', '?'))} "
            f"(J: {e.get('jockey', '?')}, T: {e.get('trainer', '?')}) "
            f"Odds: {odds_str} | Model P(win): {prob_str} | "
            f"Style: {e.get('running_style', '?')} | Last 3F: {e.get('last_3f', '?')}"
        )

    # Format value bets
    vb_lines = []
    for vb in value_bets:
        entry = next((e for e in entries if e.get("post") == vb.get("entry_id")), None)
        name = "?"
        if entry:
            name = entry.get("name_jp", entry.get("name", "?"))
        vb_lines.append(
            f"  {name}: EV={vb['ev']:+.3f}, model={vb['model_prob']:.1%}, "
            f"market={vb['market_prob']:.1%}, Kelly={vb['kelly']:.3f}"
        )

    prompt = f"""You are an expert Japanese horse racing analyst (競馬予想家).
Analyse the following race and provide a structured preview.

{lang_instruction}

## Race Information
- Name: {race.get('name_jp', race.get('name', 'Unknown'))}
- Date: {race.get('date')}
- Course: {race.get('course', 'Unknown')}
- Distance: {race.get('distance')}m {race.get('surface', '')}
- Going: {race.get('going', 'Unknown')}
- Weather: {race.get('weather', 'Unknown')}
- Class: {race.get('class', '')} {race.get('grade', '')}
- Field Size: {race.get('field_size', len(entries))} runners

## Entries (with model predictions)
{chr(10).join(entries_table)}

## Model Value Bets (positive EV plays)
{chr(10).join(vb_lines) if vb_lines else "  None identified"}

## Your Task
Provide a structured race analysis in this exact JSON format:
{{
  "race_summary": "1-2 sentence overview",
  "pace_forecast": {{
    "likely_tempo": "fast|moderate|slow",
    "leaders": ["horse names likely to lead"],
    "scenario": "detailed pace scenario description"
  }},
  "horses_to_watch": [
    {{
      "horse_name": "name",
      "post_position": 1,
      "rating": "★★★|★★|★",
      "reason": "why to watch this horse"
    }}
  ],
  "value_plays": [
    {{
      "horse_name": "name",
      "model_prob": 0.15,
      "market_prob": 0.08,
      "ev": 0.07,
      "odds": 12.5,
      "reasoning": "why this is a value play"
    }}
  ],
  "risk_factors": ["list of risk factors"],
  "recommended_strategy": "overall recommendation"
}}

Focus on:
1. Pace dynamics (展開) — who leads, how it affects closers
2. Value opportunities where model probability diverges from market
3. Specific risk factors: first-time distance, surface change, draw bias, weather impact
4. Be honest about uncertainty — don't overfit to the model

Return ONLY valid JSON, no markdown fences or extra text.
"""
    return prompt


# ---------------------------------------------------------------------------
# LLM Integration
# ---------------------------------------------------------------------------

def call_gemini(prompt: str) -> dict:
    """Call Google Gemini API and parse JSON response."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key or api_key == "your_gemini_api_key_here":
        log.warning("GOOGLE_API_KEY not set — returning mock analysis")
        return _mock_analysis()

    try:
        from google import genai

        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model="gemini-1.5-flash",
            contents=prompt,
            config={
                "temperature": 0.3,
                "max_output_tokens": 4096,
            },
        )

        text_response = response.text.strip()

        # Strip markdown code fences if present
        if text_response.startswith("```"):
            lines = text_response.split("\n")
            text_response = "\n".join(lines[1:-1])

        return json.loads(text_response)

    except json.JSONDecodeError as e:
        log.error(f"Failed to parse Gemini JSON response: {e}")
        log.debug(f"Raw response: {text_response[:500]}")
        return _mock_analysis()
    except Exception as e:
        log.error(f"Gemini API call failed: {e}")
        return _mock_analysis()


def _mock_analysis() -> dict:
    """Return a placeholder analysis when Gemini is unavailable."""
    return {
        "race_summary": "Analysis unavailable — Gemini API key not configured.",
        "pace_forecast": {
            "likely_tempo": "moderate",
            "leaders": [],
            "scenario": "Unable to generate pace forecast without LLM.",
        },
        "horses_to_watch": [],
        "value_plays": [],
        "risk_factors": ["LLM analysis not available — rely on model probabilities and pace sim"],
        "recommended_strategy": "Use model EV flags and bankroll manager for sizing.",
    }


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------

def analyse_race(
    race_id: int,
    language: str = "en",
    store_to_db: bool = True,
    model_version: str = "latest",
) -> dict:
    """
    Generate a full race analysis using the LLM agent.

    1. Load race context from DB (predictions, entries, value bets)
    2. Build structured prompt
    3. Call Gemini for analysis
    4. Store to agent_analyses table

    Returns:
        Parsed analysis dict
    """
    # 1. Load context
    log.info(f"Loading race context for race {race_id}...")
    context = load_race_context(race_id)

    if not context.get("race"):
        log.error(f"Race {race_id} not found in database")
        return {}

    # Ensure predictions exist
    if not any(e.get("model_win_prob") for e in context.get("entries", [])):
        log.info("No predictions found — running prediction pipeline first...")
        from models.predict import predict_and_store
        predict_and_store(race_id, model_version=model_version, store_to_db=True)
        context = load_race_context(race_id)

    # 2. Build prompt
    prompt = build_prompt(context, language)

    # 3. Call LLM
    log.info(f"Calling Gemini for {language.upper()} analysis...")
    analysis = call_gemini(prompt)

    # 4. Store
    if store_to_db:
        _store_analysis(race_id, analysis, language, model_version)

    return analysis


def _store_analysis(
    race_id: int,
    analysis: dict,
    language: str,
    model_version: str,
):
    """Store analysis to DB."""
    # Build plain-text version
    raw_parts = [analysis.get("race_summary", "")]

    pace = analysis.get("pace_forecast", {})
    if pace:
        raw_parts.append(f"\n展開予想: {pace.get('scenario', 'N/A')}")

    for hw in analysis.get("horses_to_watch", []):
        raw_parts.append(f"\n{hw.get('rating', '★')} {hw.get('horse_name', '?')}: {hw.get('reason', '')}")

    for vp in analysis.get("value_plays", []):
        raw_parts.append(
            f"\n💰 {vp.get('horse_name', '?')} "
            f"(EV: {vp.get('ev', 0):+.3f}, Odds: {vp.get('odds', 0):.1f}): "
            f"{vp.get('reasoning', '')}"
        )

    for rf in analysis.get("risk_factors", []):
        raw_parts.append(f"\n⚠️ {rf}")

    raw_parts.append(f"\n📋 {analysis.get('recommended_strategy', '')}")
    raw_text = "\n".join(raw_parts)

    with get_session() as session:
        session.execute(text("""
            INSERT INTO agent_analyses
                (race_id, model_version, language, analysis, raw_text)
            VALUES
                (:race_id, :version, :lang, :analysis, :raw_text)
            ON CONFLICT (race_id, model_version, language) DO UPDATE SET
                analysis = EXCLUDED.analysis,
                raw_text = EXCLUDED.raw_text,
                created_at = now()
        """), {
            "race_id": race_id,
            "version": model_version,
            "lang": language,
            "analysis": json.dumps(analysis),
            "raw_text": raw_text,
        })

    log.info(f"💾 Stored {language.upper()} analysis for race {race_id}")


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def print_analysis(analysis: dict):
    """Pretty-print the analysis to stdout."""
    if not analysis:
        print("No analysis available.")
        return

    print("\n🏇 UmaEdge — Race Analysis")
    print("=" * 60)

    print(f"\n📝 {analysis.get('race_summary', 'N/A')}")

    # Pace forecast
    pace = analysis.get("pace_forecast", {})
    if pace:
        print(f"\n🏃 Pace: {pace.get('likely_tempo', '?').upper()}")
        leaders = pace.get("leaders", [])
        if leaders:
            print(f"   Leaders: {', '.join(leaders)}")
        print(f"   {pace.get('scenario', '')}")

    # Horses to watch
    htw = analysis.get("horses_to_watch", [])
    if htw:
        print("\n⭐ Horses to Watch:")
        for h in htw:
            print(f"   {h.get('rating', '★')} #{h.get('post_position', '?')} {h.get('horse_name', '?')}")
            print(f"      {h.get('reason', '')}")

    # Value plays
    vp = analysis.get("value_plays", [])
    if vp:
        print("\n💰 Value Plays:")
        for v in vp:
            print(
                f"   {v.get('horse_name', '?')} — "
                f"EV: {v.get('ev', 0):+.3f} | "
                f"Odds: {v.get('odds', 0):.1f} | "
                f"Model: {v.get('model_prob', 0):.1%} vs Market: {v.get('market_prob', 0):.1%}"
            )
            print(f"      {v.get('reasoning', '')}")

    # Risk factors
    risks = analysis.get("risk_factors", [])
    if risks:
        print("\n⚠️ Risk Factors:")
        for r in risks:
            print(f"   • {r}")

    # Strategy
    strategy = analysis.get("recommended_strategy", "")
    if strategy:
        print(f"\n📋 Strategy: {strategy}")

    print("\n" + "=" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="UmaEdge — Race Analyst Agent")
    parser.add_argument("--race-id", type=int, required=True, help="Database race ID")
    parser.add_argument("--lang", choices=["en", "ja"], default="en", help="Language (default: en)")
    parser.add_argument("--no-store", action="store_true", help="Don't store to DB")
    parser.add_argument("--version", type=str, default="latest", help="Model version label")
    args = parser.parse_args()

    analysis = analyse_race(
        race_id=args.race_id,
        language=args.lang,
        store_to_db=not args.no_store,
        model_version=args.version,
    )

    print_analysis(analysis)


if __name__ == "__main__":
    main()
