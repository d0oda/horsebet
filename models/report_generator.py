"""
UmaEdge — Structured Race Report Generator.

Generates markdown race reports with ability ratings, condition fit,
value analysis, and decision labels for each runner.

Usage:
    from models.report_generator import generate_race_report
    report = generate_race_report(predictions_df, race_id)
"""

import logging
import os
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
from sqlalchemy import text

from scraper.db import get_session

log = logging.getLogger("report_generator")


def generate_race_report(
    predictions: pd.DataFrame,
    race_id: int,
    output_dir: str = "outputs/race_reports",
) -> str:
    """
    Generate a structured markdown race report for a single race.

    Args:
        predictions: DataFrame from predict_and_store() with columns:
            horse_name, combined_win_prob, odds, fair_odds, fair_market_prob,
            edge, ability_rating, condition_fit, bounce_risk, decision
        race_id: Database race ID.
        output_dir: Directory to write the report to.

    Returns:
        The markdown report string.
    """
    # Load race metadata
    with get_session() as session:
        race = session.execute(
            text("""
                SELECT r.date, r.race_number, c.name_jp AS venue,
                       r.distance, r.surface, r.going, r.class, r.field_size
                FROM races r
                LEFT JOIN courses c ON c.id = r.course_id
                WHERE r.id = :rid
            """),
            {"rid": race_id},
        ).fetchone()

    if not race:
        log.warning(f"Race {race_id} not found in database")
        return ""

    date_str = str(race[0]) if race[0] else "Unknown"
    race_num = race[1] or "?"
    venue = race[2] or "Unknown"
    distance = race[3] or "?"
    surface = race[4] or "?"
    going = race[5] or "?"
    race_class = race[6] or "?"
    field_size = race[7] or len(predictions)

    # Sort by combined probability
    df = predictions.sort_values("combined_win_prob", ascending=False).reset_index(drop=True)

    # Summary stats
    top_ability = df.loc[df["ability_rating"].idxmax()] if "ability_rating" in df.columns and df["ability_rating"].notna().any() else None
    top_prob = df.iloc[0] if len(df) > 0 else None
    value_candidates = df[df["decision"].isin(["Strong Value", "Value"])] if "decision" in df.columns else pd.DataFrame()
    best_value = value_candidates.sort_values("edge", ascending=False).iloc[0] if len(value_candidates) > 0 else None

    # Favourite vulnerability check
    if len(df) >= 2:
        fav = df.sort_values("odds").iloc[0]
        fav_vuln = []
        if fav.get("condition_fit") and fav["condition_fit"] < 75:
            fav_vuln.append(f"low condition fit ({fav['condition_fit']:.0f})")
        if fav.get("bounce_risk", 0) == 1:
            fav_vuln.append("bounce risk")
        fav_vuln_text = f"{fav['horse_name']}: {', '.join(fav_vuln)}" if fav_vuln else "None detected"
    else:
        fav_vuln_text = "N/A"

    # Race confidence: gap between #1 and #2 probabilities
    if len(df) >= 2:
        gap = df.iloc[0]["combined_win_prob"] - df.iloc[1]["combined_win_prob"]
        if gap > 0.10:
            confidence = "High"
        elif gap > 0.04:
            confidence = "Medium"
        else:
            confidence = "Low"
    else:
        confidence = "N/A"

    overround = df.iloc[0].get("overround", 1.0) if len(df) > 0 else 1.0

    # Build markdown
    lines = []
    lines.append(f"# Race Report: {venue} {race_num}R")
    lines.append(f"**{date_str}** | {distance}m {surface} | Going: {going} | Class: {race_class} | Field: {field_size}")
    lines.append(f"Overround: {overround*100:.1f}%")
    lines.append("")

    # Summary
    lines.append("## Summary")
    if top_ability is not None and pd.notna(top_ability.get("ability_rating")):
        lines.append(f"- **Top Absolute Ability:** {top_ability['horse_name']} (rating: {top_ability['ability_rating']:.0f})")
    if top_prob is not None:
        lines.append(f"- **Top Race-Relative:** {top_prob['horse_name']} ({top_prob['combined_win_prob']:.1%})")
    if best_value is not None:
        lines.append(f"- **Best Value Candidate:** {best_value['horse_name']} (edge: +{best_value['edge']:.1%})")
    lines.append(f"- **Favourite Vulnerability:** {fav_vuln_text}")
    lines.append(f"- **Overall Race Confidence:** {confidence}")
    lines.append("")

    # Runner table
    lines.append("## Runner Table")
    lines.append("| # | Horse | Ability | Cond. Fit | Odds | Model P | Fair Odds | Edge | Decision |")
    lines.append("|--:|-------|--------:|----------:|-----:|--------:|----------:|-----:|----------|")

    for i, (_, row) in enumerate(df.iterrows()):
        ability = f"{row['ability_rating']:.0f}" if pd.notna(row.get("ability_rating")) else "-"
        cfit = f"{row['condition_fit']:.0f}" if pd.notna(row.get("condition_fit")) else "-"
        edge_str = f"+{row['edge']:.1%}" if row.get("edge", 0) >= 0 else f"{row['edge']:.1%}"
        decision = row.get("decision", "")
        icon = {"Strong Value": "🔥", "Value": "✅", "Lean": "👀", "Watch Only": "⚠️"}.get(decision, "")

        lines.append(
            f"| {i+1} | {row['horse_name']} | {ability} | {cfit} | "
            f"{row['odds']:.1f} | {row['combined_win_prob']:.1%} | "
            f"{row.get('fair_odds', 0):.1f} | {edge_str} | {icon} {decision} |"
        )
    lines.append("")

    # Value candidates detail
    if len(value_candidates) > 0:
        lines.append("## Value Candidates")
        for _, row in value_candidates.iterrows():
            ability_rank = (df["ability_rating"].rank(ascending=False, method="min").loc[df["horse_name"] == row["horse_name"]]).values
            ab_rank = int(ability_rank[0]) if len(ability_rank) > 0 and pd.notna(ability_rank[0]) else "?"

            lines.append(f"### {row['horse_name']}")
            if pd.notna(row.get("ability_rating")):
                lines.append(f"- Absolute ability: {row['ability_rating']:.0f} (rank: {ab_rank}/{field_size})")
            if pd.notna(row.get("condition_fit")):
                lines.append(f"- Condition fit: {row['condition_fit']:.0f}/100")
            lines.append(f"- Current odds: {row['odds']:.1f}")
            lines.append(f"- Model probability: {row['combined_win_prob']:.1%}")
            lines.append(f"- Fair odds: {row.get('fair_odds', 0):.1f}")
            lines.append(f"- Edge: +{row['edge']:.1%}")
            lines.append(f"- Decision: **{row['decision']}**")

            # Reason
            reasons = []
            if pd.notna(row.get("ability_rating")) and row["ability_rating"] >= 90:
                reasons.append("elite ability rating")
            if pd.notna(row.get("condition_fit")) and row["condition_fit"] >= 80:
                reasons.append("strong condition fit for today")
            if row.get("edge", 0) >= 0.05:
                reasons.append(f"significant market edge ({row['edge']:.1%})")
            lines.append(f"- **Reason:** {'; '.join(reasons) if reasons else 'Model edge over market probability'}")

            # Risk
            risks = []
            if row.get("bounce_risk", 0) == 1:
                risks.append("bounce risk after career-best last run")
            if pd.notna(row.get("condition_fit")) and row["condition_fit"] < 80:
                risks.append(f"moderate condition fit ({row['condition_fit']:.0f})")
            if row["odds"] >= 15:
                risks.append("longshot odds (higher variance)")
            lines.append(f"- **Risk:** {'; '.join(risks) if risks else 'Standard betting risk'}")
            lines.append("")

    # Avoid section
    avoid = df[df["decision"] == "Avoid"] if "decision" in df.columns else pd.DataFrame()
    notable_avoids = avoid[avoid["ability_rating"].fillna(0) >= 80] if "ability_rating" in avoid.columns else pd.DataFrame()
    if len(notable_avoids) > 0:
        lines.append("## Notable Avoids")
        for _, row in notable_avoids.iterrows():
            lines.append(f"### {row['horse_name']}")
            reasons = []
            if pd.notna(row.get("condition_fit")) and row["condition_fit"] < 70:
                reasons.append(f"poor condition fit ({row['condition_fit']:.0f})")
            if row.get("edge", 0) < 0:
                reasons.append(f"negative edge ({row['edge']:.1%})")
            lines.append(f"- **Reason:** {'; '.join(reasons) if reasons else 'No value at current odds'}")
            lines.append("")

    report = "\n".join(lines)

    # Write to file
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{date_str}_{venue}_R{race_num}.md"
    filepath = os.path.join(output_dir, filename)
    with open(filepath, "w") as f:
        f.write(report)
    log.info(f"📝 Race report: {filepath}")

    return report
