# UmaEdge — Missing Features from Implementation Blueprint

Everything below is **not yet implemented** in UmaEdge. Each item is extracted from `horse_racing_betting_assistant_implementation.md` and cross-referenced against the current codebase (`models/features.py`, `models/predict.py`, `models/explain.py`, `pipeline.py`, etc.).

Items marked with ⭐ are the highest-impact additions.

---

## 1. ⭐ Persistent Absolute Ability Rating (EWMA)

**Blueprint ref:** §2 — Score Layer A (the core concept)

**What we have:** Per-run `speed_figure_last/best/avg3` — a snapshot metric computed fresh for each prediction. No persistent horse-level rating that accumulates across races.

**What we need:**

A stored, EWMA-updated **absolute ability score** per horse (scale 20–100), combining:

```
absolute_ability_score =
    base_speed_figure
    + class_adj
    + field_strength_adj
    + pace_adj
    + weight_adj
    + going_adj
    + margin_adj
```

After each race result, update persistently:

```python
new_rating = old_rating * 0.70 + race_performance * 0.30
```

### Implementation requirements:
- New DB column: `horses.ability_rating` (float)
- New DB table: `horse_rating_history` (horse_id, race_id, date, rating_before, race_score, rating_after)
- Batch job: `scripts/backfill_ratings.py` — compute ratings from 2018 forward in chronological order
- Updater: `scripts/update_ratings.py` — run after each race day's results are scraped
- New features for XGBoost:
  - `horse_ability_rating` — current EWMA rating
  - `rating_trend_slope` — slope of last 5 ratings
  - `rating_vs_field_avg` — horse rating minus field average rating
  - `field_avg_ability_rating` — mean of all runners' ratings in this race
  - `rating_percentile_in_field` — where this horse ranks within today's field

### Sub-components to build:

#### 1a. `class_adj` (scored, not just a rank)
Currently `CLASS_RANK` maps classes to ordinal 1–10. Blueprint wants a numeric **score** adjustment:

```python
CLASS_SCORE = {
    "G1": 115, "G2": 108, "G3": 103, "L": 98, "OP": 94,
    "3勝": 90, "2勝": 84, "1勝": 78, "未勝利": 70, "新馬": 65,
}
class_adj = (CLASS_SCORE[race_class] - 90) * 0.25  # scaled contribution
```

#### 1b. `field_strength_adj` (using ability ratings)
Currently `field_avg_career_win_pct` uses career win% as a proxy. Blueprint wants:

```python
field_strength = mean(ability_rating of all runners)
field_strength_adj = (field_strength - global_average_field_strength) * 0.35
```

This creates a circular dependency (field strength depends on individual ratings, which depend on field strength). Resolve by:
- First pass: compute ratings without field strength adj
- Second pass: re-compute with field strength adj using first-pass ratings
- Or: use a 1-race-lagged field strength (each horse's rating from their *previous* race)

#### 1c. `margin_adj` (scored, not just a feature)
Currently `beaten_lengths_avg3` and `class_adjusted_margin` exist as raw features. Blueprint wants an explicit scored adjustment:

```python
margin_adj = -min(beaten_lengths * 1.5, 12)
# Forgiveness in strong fields:
if field_strength_adj > 5 and beaten_lengths <= 5:
    margin_adj *= 0.75
```

#### 1d. `going_adj` (ability-layer, not suitability)
Currently going features exist for condition fit. Blueprint separately wants a going adjustment in the ability layer — to prevent false downgrades when a horse ran on unsuitable going:

```python
if horse_preferred_going != race_going and performance_was_poor:
    going_adj = +2  # forgive the bad run
else:
    going_adj = 0
```

Requires knowing `horse_preferred_going` — derivable from going with best historical win rate.

---

## 2. ⭐ Condition Fit Composite Score (0–100)

**Blueprint ref:** §4 — Score Layer C

**What we have:** Individual component features (`win_pct_at_dist`, `win_pct_on_surface`, `runs_at_course`, etc.) that XGBoost uses independently.

**What we need:** An explicit, interpretable **composite score** (0–100 scale) combining:

```python
condition_fit_score =
    0.25 * distance_fit     # 0-100
    + 0.20 * going_fit      # 0-100
    + 0.15 * surface_fit    # 0-100
    + 0.10 * track_fit      # 0-100
    + 0.10 * pace_fit       # 0-100
    + 0.08 * draw_fit       # 0-100
    + 0.07 * class_fit      # 0-100
    + 0.05 * rest_days_fit  # 0-100
```

### Each sub-component (0–100 scale):

**`distance_fit`** — win/place rate at today's distance band vs overall:
```python
bands = {"sprint": <=1400, "mile": 1401-1800, "middle": 1801-2200, "staying": >2200}
distance_fit = clamp(horse_band_win_rate / overall_win_rate * 100, 60, 115)
```
(If no history at band → 75 default)

**`going_fit`** — performance on today's going vs career average:
```python
going_fit = clamp(horse_going_win_rate / overall_win_rate * 100, 60, 115)
```

**`surface_fit`** — turf vs dirt suitability from `win_pct_on_surface`

**`track_fit`** — course-specific from `win_pct_at_course`

**`pace_fit`** — does expected pace pattern match running style:
```python
if expected_pace == "fast" and style == "closer": pace_fit = 85
if expected_pace == "slow" and style == "front_runner": pace_fit = 90
# etc.
```

**`draw_fit`** — from `draw_bias_at_course` / `draw_bias_90d`

**`class_fit`** — is the horse stepping up, staying level, or dropping:
```python
if class_change < 0: class_fit = 90   # dropping (easier)
if class_change == 0: class_fit = 80  # same level
if class_change > 0: class_fit = 65   # stepping up
```

**`rest_days_fit`** — from layoff buckets:
```python
if 14 <= days <= 35: rest_fit = 90
if 36 <= days <= 60: rest_fit = 80
if days > 90: rest_fit = 60
if days < 14: rest_fit = 70
```

### Purpose:
- Used as input to the Race-Relative Score
- Used in Final Decision Layer (threshold: `condition_fit >= 80` for Strong Value)
- Displayed in race reports
- Fed as an additional XGBoost feature

---

## 3. ⭐ Race-Relative Composite Score

**Blueprint ref:** §3 — Score Layer B

**What we have:** XGBoost does implicit relative ranking via per-race z-score normalisation. But there's no explicit, interpretable weighted composite.

**What we need:**

```python
race_relative_score =
    0.35 * recent_absolute_ability_score   # from §1
    + 0.20 * best_recent_absolute_ability_score
    + 0.15 * condition_fit_score           # from §2
    + 0.10 * pace_setup_score
    + 0.08 * form_trajectory_score
    + 0.07 * jockey_trainer_score
    + 0.05 * draw_score
```

### Outputs:
```
rank                 # 1st to Nth in field
relative_score       # raw weighted composite
gap_to_top           # score difference from top-ranked
gap_to_market_rank   # rank from score vs rank from odds
```

### Purpose:
- A human-readable "who should win this race" ranking
- Divergence between `gap_to_market_rank` and actual rank = potential value detection
- Displayed in race reports alongside XGBoost probabilities

---

## 4. ⭐ Form Trajectory Score

**Blueprint ref:** §5 — Score Layer D

**What we have:** `last3_avg_finish`, `last5_win_pct`, `speed_figure_last/best/avg3` as raw features. No composite trajectory score.

**What we need:**

```python
form_trajectory_score =
    0.40 * average_last_3_ability_scores
    + 0.30 * best_last_5_ability_scores
    + 0.20 * slope_last_5_scores_scaled
    + 0.10 * consistency_score
```

### New features required:

| Feature | Description | Status |
|---|---|---|
| `avg_last_3_ability` | Mean of last 3 absolute ability scores | 🔴 New — needs ability rating |
| `best_last_5_ability` | Max of last 5 absolute ability scores | 🔴 New — needs ability rating |
| `ability_slope_last5` | Linear regression slope of last 5 ability scores | 🔴 New |
| `ability_consistency` | StdDev of last 5 ability scores (lower = more consistent) | 🔴 New |
| `second_up_flag` | Is this the horse's 2nd run back from a spell? | 🔴 New |
| `third_up_flag` | Is this the horse's 3rd run back from a spell? | 🔴 New |
| `new_peak_flag` | Did last run set a new career-best ability score? | 🔴 New |
| `bounce_risk_flag` | Last score > previous best + 8 points → regression risk | 🔴 New |

### Spell detection logic:
```python
# If days_since_last > 60, the horse was "spelling" (on a break)
# Count runs since last spell to determine 1st-up, 2nd-up, 3rd-up
spell_threshold = 60  # days
run_number_since_spell = count(runs after last gap > spell_threshold)

second_up_flag = 1 if run_number_since_spell == 2 else 0
third_up_flag = 1 if run_number_since_spell == 3 else 0
```

### Bounce risk logic:
```python
if speed_figure_last > previous_best_figure + 8:
    bounce_risk_flag = 1
else:
    bounce_risk_flag = 0
```

---

## 5. ⭐ Final Decision Layer (Labelled Categories)

**Blueprint ref:** §7 — Final Betting Decision Layer

**What we have:** Binary `is_value` flag based on EV threshold + odds range. No granular categories.

**What we need:**

```python
if edge >= 0.07 and condition_fit >= 80 and confidence >= 0.70:
    decision = "Strong Value"
elif edge >= 0.03 and condition_fit >= 75:
    decision = "Value"
elif race_relative_rank <= 3 and edge >= 0:
    decision = "Lean"
elif absolute_ability_rank <= 2 and condition_fit < 70:
    decision = "Watch Only"
else:
    decision = "Avoid"
```

### Also needs:
- `favourite_vulnerability` — flag if the favourite has condition_fit < 75 or is bouncing
- `overall_race_confidence` — how separable the field is (gap between top 2 scores)
- Every recommendation includes: reason + risk (mandatory pair)

### Where used:
- `bets.txt` output
- Frontend predictions display
- Race report markdown

---

## 6. Structured Race Reports (Markdown)

**Blueprint ref:** §8, §16 — Explanation Template + Report Format

**What we have:** `models/explain.py` does SHAP-based feature importance. No structured markdown race reports with the blueprint's format.

**What we need:** For each race, generate `outputs/race_reports/{race_id}.md` with:

```markdown
# Race Report: {venue} {race_number}R

## Summary
- Top Absolute Ability Horse: {name} (rating: {score})
- Top Race-Relative Horse: {name} (score: {score})
- Best Value Candidate: {name} (edge: +{edge}%)
- Favourite Vulnerability: {description or "None"}
- Overall Race Confidence: {High/Medium/Low}

## Runner Table
| Horse | Ability | Rel. Rank | Cond. Fit | Odds | Model P | Fair Odds | Edge | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---|

## Value Candidates
### {Horse Name}
- Absolute ability rank: {rank}/{field_size}
- Race-relative rank: {rank}/{field_size}
- Condition fit: {score}/100
- Current odds: {odds}
- Model probability: {prob}%
- Fair odds: {fair_odds}
- Edge: +{edge}%
- Decision: {label}

Reason: {natural language explanation}
Risk: {natural language risk}

## Avoid / Oppose
### {Horse Name}
Reason: {why this horse is flagged as Avoid}
```

### Implementation:
- New module: `models/report_generator.py`
- Called from `pipeline.py` after predictions are generated
- Writes to `outputs/race_reports/` (create dir)
- Requires: ability rating, condition fit score, race-relative score, decision labels

---

## 7. Overround-Adjusted Fair Market Probability

**Blueprint ref:** §6 — Market Value Score

**What we have:** `market_prob = 1 / odds` (raw implied probability, includes overround).

**What we need:** Strip overround to get *fair* market probability:

```python
raw_probs = [1/odds for odds in all_horses_odds]
overround = sum(raw_probs)
fair_market_prob = (1 / odds) / overround
```

Then:

```python
edge = model_prob - fair_market_prob  # instead of model_prob - (1/odds)
fair_odds = 1 / model_prob
```

### Impact:
- Currently the overround (~120% for JRA) inflates `market_prob`, which *understates* the edge for all horses
- After correction, EV calculations will be more accurate
- Requires all runners' odds to be available at prediction time (they are, from `race_df`)

---

## 8. Going-Specific Speed Profile

**Blueprint ref:** §18 — Future Extensions (going-specific performance profiles)

**What we have:** `horse_going_win_pct` (binary win/loss), `horse_wet_track_advantage` (performance on wet vs good). No going-specific *speed figure* profile.

**What we need:** For each horse, compute average speed figure by going condition:

```python
speed_on_good = avg(speed_figures where going == "良")
speed_on_yielding = avg(speed_figures where going == "稍重")
speed_on_soft = avg(speed_figures where going == "重")
speed_on_heavy = avg(speed_figures where going == "不良")
```

Features:
- `speed_fig_on_todays_going` — avg speed figure on today's going
- `speed_fig_going_delta` — speed on today's going minus overall avg speed fig
- `best_going_code` — the going where horse has best average speed figure

---

## 9. Stable Form Indicators

**Blueprint ref:** §18 — Future Extensions (stable form indicators)

**What we have:** `trainer_win_pct_10`, `trainer_14d_win_pct` — trainer form. No stable-level (training center) aggregation.

**What we need:**

```python
# Is the trainer's stable "in form"?
stable_form = trainer_14d_win_pct weighted by recency

# Are horses from this stable running well THIS WEEK?
stable_last7d_runners = count(trainer's horses that raced in last 7 days)
stable_last7d_win_pct = wins / runners in last 7 days
stable_last7d_place_pct = placed / runners in last 7 days
```

Also:
- `stable_hot_streak` — trainer has 3+ wins in last 7 days
- `stable_cold_streak` — trainer has 0 wins in last 15 runners

---

## 10. Trip Notes / In-Running Trouble Parser

**Blueprint ref:** §18 — Future Extensions (trip notes / trouble-in-running parser)

**What we have:** Nothing. No in-running comments or trouble indicators.

**What we need:**

If `in_running_comment` or equivalent data becomes available (from netkeiba race comments):

```python
# Parse for trouble indicators
trouble_keywords = ["不利", "挟まれ", "出遅れ", "掛かり", "詰まり"]
had_trouble = any(kw in comment for kw in trouble_keywords)

# Feature: was the horse's last run affected by trouble?
last_run_trouble = 1 if had_trouble else 0
# Feature: excuse factor — did trouble cost them placings?
trouble_excuse = 1 if had_trouble and finish_pos > 3 and beaten_lengths < 3 else 0
```

This is a data acquisition challenge — would need to scrape race comments from netkeiba.

---

## Priority Summary

| # | Feature | Impact | Effort | Priority |
|---|---|---|---|---|
| 1 | Persistent Ability Rating (EWMA) | 🔴 Very High — unlocks §2-5 | High | ⭐ Do first |
| 2 | Condition Fit Composite Score | 🔴 High — needed for decisions | Medium | ⭐ Do second |
| 3 | Race-Relative Composite Score | 🟡 Medium — interpretability | Medium | Third |
| 4 | Form Trajectory Score + Flags | 🟡 Medium — new features for XGB | Medium | Third |
| 5 | Final Decision Labels | 🟡 Medium — UX / reporting | Low | After 1-4 |
| 6 | Structured Race Reports | 🟡 Medium — UX / reporting | Medium | After 1-5 |
| 7 | Overround-Adjusted Fair Probs | 🟢 Quick win — accuracy fix | Low | Anytime |
| 8 | Going-Specific Speed Profile | 🟡 Medium — new features | Low | Anytime |
| 9 | Stable Form Indicators | 🟢 Low — marginal | Low | Later |
| 10 | Trip Notes Parser | 🟢 Low — data dependency | High | Later |

---

## Dependency Chain

```
1. Ability Rating  ──→  2. Condition Fit Score  ──→  3. Race-Relative Score
        │                        │                           │
        └── 4. Form Trajectory ──┘                           │
                                                             ▼
                                                    5. Decision Labels
                                                             │
                                                             ▼
                                                    6. Race Reports

7. Overround Fix ──→ standalone (can do anytime)
8. Going Speed Profile ──→ standalone (can do anytime)
9. Stable Form ──→ standalone (can do anytime)
10. Trip Notes ──→ standalone (needs data scraping)
```
