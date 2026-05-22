# Horse Racing Betting Assistant: Cross-Race Ability Scoring Implementation

## Purpose

Build a betting-assistant scoring layer inspired by the distinction between:

1. **Race-relative scoring**: how strong a horse looks within the current race field.
2. **Absolute / cross-race ability scoring**: how strong the horse’s underlying performance level is across races, after normalising for class, pace, distance, going, track, draw, field strength, and race shape.

The goal is not to blindly predict winners. The goal is to estimate when the market may be mispricing a horse because its recent results are misleading relative to its true ability.

This document should be used as an implementation brief for the coding agent.

---

## Core Design Principle

Do not use one monolithic “score”.

Implement separate score layers:

```text
Horse Evaluation
├── A. Absolute Ability Score
├── B. Race-Relative Score
├── C. Condition Fit Score
├── D. Form / Trajectory Score
├── E. Market Value Score
└── F. Final Betting Decision Layer
```

The assistant should always explain which layer is driving a recommendation.

Example:

> “Horse A is only 4th on race-relative score, but has the highest absolute ability figure in the field and returns to its optimal distance. Current odds imply 12% win probability, while the model estimates 18%, so it is a value candidate.”

---

## 1. Data Model

Create a structured data model that separates horse-level, race-level, and run-level data.

### 1.1 Horse Table

Suggested file: `data/horses.csv`

Required columns:

```text
horse_id
horse_name
sex
age
trainer
owner
sire
dam
```

Optional columns:

```text
country
preferred_distance_min
preferred_distance_max
preferred_going
running_style
```

---

### 1.2 Race Table

Suggested file: `data/races.csv`

Required columns:

```text
race_id
race_date
track
race_class
distance_m
going
surface
field_size
prize_money
race_type
```

Optional columns:

```text
rail_position
weather
track_bias_note
sectional_available
```

---

### 1.3 Run / Entry Table

Suggested file: `data/runs.csv`

Each row is one horse in one race.

Required columns:

```text
race_id
horse_id
draw
carried_weight
jockey
finish_position
beaten_lengths
starting_price
official_rating
```

Optional but strongly preferred:

```text
time_seconds
last_600m_seconds
last_400m_seconds
last_200m_seconds
position_early
position_mid
position_turn
position_finish
pace_pressure
in_running_comment
```

---

### 1.4 Odds Table

Suggested file: `data/odds.csv`

```text
race_id
horse_id
timestamp
bookmaker
decimal_odds
exchange_odds
market_percentage
```

Use this for market-implied probability and value detection.

---

## 2. Score Layer A: Absolute Ability Score

### Purpose

Estimate how good a horse’s performance was in absolute terms, not merely relative to that specific field.

This is the most important learning from the source idea: a horse can look strong in a weak race or ordinary in a strong race. The assistant should try to distinguish those cases.

### Required Concept

Create an `absolute_ability_score` for every past run.

This should be calculated from:

```text
raw_performance
+ class_adjustment
+ time_adjustment
+ pace_adjustment
+ going_adjustment
+ weight_adjustment
+ field_strength_adjustment
+ margin_adjustment
```

### Suggested Formula

Start with an interpretable weighted score before building ML models.

```python
absolute_ability_score =
    base_speed_figure
    + class_adj
    + field_strength_adj
    + pace_adj
    + weight_adj
    + going_adj
    + margin_adj
```

### Component Definitions

#### `base_speed_figure`

Measures how fast the horse ran relative to historical par for:

```text
track + distance + surface + going
```

Implementation:

```python
base_speed_figure = 100 + ((par_time_seconds - actual_time_seconds) / par_time_std) * 10
```

If lower time is better, a horse faster than par gets a higher score.

If exact time data is unavailable, approximate using:

```text
race class
finish position
beaten lengths
field size
historical race strength
```

---

#### `class_adj`

Adjust for race class.

Example mapping:

```python
CLASS_RATING = {
    "G1": 115,
    "G2": 108,
    "G3": 103,
    "Listed": 98,
    "Open": 94,
    "3-win": 90,
    "2-win": 84,
    "1-win": 78,
    "Maiden": 70,
}
```

Use the class value as a contextual prior, not as the full score.

---

#### `field_strength_adj`

Estimate how strong the opponents were.

For each race, compute:

```python
field_strength = mean(previous_absolute_ability_score of all runners)
```

Then:

```python
field_strength_adj = (field_strength - global_average_field_strength) * 0.35
```

This prevents weak-field wins from being overvalued and strong-field losses from being undervalued.

---

#### `pace_adj`

Correct for whether the race shape helped or hurt the horse.

Examples:

```text
Fast early pace + closer finished well = upgrade
Slow early pace + front-runner won easily = possible downgrade
Fast early pace + front-runner held on = strong upgrade
Slow pace + closer failed = forgive partially
```

Represent running styles as:

```text
front_runner
prominent
midfield
closer
```

Suggested initial logic:

```python
if pace == "fast" and running_style == "front_runner" and finish_position <= 3:
    pace_adj = +5
elif pace == "fast" and running_style == "closer":
    pace_adj = -1
elif pace == "slow" and running_style == "closer" and beaten_lengths <= 3:
    pace_adj = +3
elif pace == "slow" and running_style == "front_runner" and finish_position == 1:
    pace_adj = -2
else:
    pace_adj = 0
```

Later replace this with learned weights.

---

#### `weight_adj`

Correct for carried weight.

Simple version:

```python
weight_adj = (average_weight_in_race - carried_weight) * -0.8
```

Explanation:

- Carrying more weight than the field average and still performing well should upgrade the run.
- Carrying less weight should not be over-rewarded.

You may need to tune this by jurisdiction.

---

#### `going_adj`

Adjust for ground/surface suitability.

Keep two concepts separate:

1. The horse’s **actual performance on that going**.
2. Whether that going is likely to be optimal in a future race.

For absolute ability, going adjustment should mostly prevent false downgrades on unsuitable ground.

Example:

```python
if horse_preferred_going != race_going and poor_performance:
    going_adj = +2
else:
    going_adj = 0
```

---

#### `margin_adj`

Use beaten lengths carefully.

A horse beaten 6 lengths in a Group 1 may have run better than a horse winning a weak race.

Suggested:

```python
margin_adj = -min(beaten_lengths * 1.5, 12)
```

Add protection against over-penalising:

```python
if field_strength_adj > 5 and beaten_lengths <= 5:
    margin_adj *= 0.75
```

---

## 3. Score Layer B: Race-Relative Score

### Purpose

Rank horses within the current race.

This answers:

> “Compared with today’s opponents, who is strongest?”

Unlike absolute ability, this is explicitly field-specific.

### Inputs

For each declared runner:

```text
recent_absolute_ability_score
best_absolute_ability_score
weighted_form_score
condition_fit_score
pace_setup_score
jockey_trainer_score
draw_score
market_score
```

### Suggested Formula

```python
race_relative_score =
    0.35 * recent_absolute_ability_score
    + 0.20 * best_recent_absolute_ability_score
    + 0.15 * condition_fit_score
    + 0.10 * pace_setup_score
    + 0.08 * form_trajectory_score
    + 0.07 * jockey_trainer_score
    + 0.05 * draw_score
```

Then normalise within race:

```python
relative_rank_percentile = rank_percentile(race_relative_score within race)
```

Output:

```text
rank
score
gap_to_top
gap_to_market_rank
```

---

## 4. Score Layer C: Condition Fit Score

### Purpose

Estimate whether today’s race conditions fit the horse.

This should be separate from raw ability. A very good horse in the wrong race should be marked as risky.

### Components

```text
distance_fit
going_fit
surface_fit
track_fit
class_fit
pace_fit
draw_fit
weight_fit
rest_days_fit
```

### Suggested Formula

```python
condition_fit_score =
    0.25 * distance_fit
    + 0.20 * going_fit
    + 0.15 * surface_fit
    + 0.10 * track_fit
    + 0.10 * pace_fit
    + 0.08 * draw_fit
    + 0.07 * class_fit
    + 0.05 * rest_days_fit
```

Each component should be 0 to 100.

### Distance Fit

Estimate from historical performance by distance band:

```text
sprint: <= 1400m
mile: 1401-1800m
middle: 1801-2200m
staying: > 2200m
```

Example:

```python
distance_fit = mean_score_for_distance_band / overall_mean_score * 100
```

Clamp:

```python
distance_fit = min(max(distance_fit, 60), 115)
```

---

## 5. Score Layer D: Form / Trajectory Score

### Purpose

Detect whether a horse is improving, declining, inconsistent, or returning to form.

Do not just average recent placings.

Use the absolute ability figures.

### Suggested Features

```text
last_run_score
average_last_3_score
best_last_5_score
slope_last_5_scores
days_since_last_run
layoff_flag
second_up_flag
third_up_flag
new_peak_flag
bounce_risk_flag
```

### Example Logic

```python
form_trajectory_score =
    0.40 * average_last_3_score
    + 0.30 * best_last_5_score
    + 0.20 * slope_last_5_scores_scaled
    + 0.10 * consistency_score
```

### Bounce Risk

If a horse just ran a career-best figure by a large margin, flag possible regression.

```python
if last_run_score > previous_best_score + 8:
    bounce_risk_flag = True
```

Do not automatically downgrade; just surface as risk.

---

## 6. Score Layer E: Market Value Score

### Purpose

Convert model rankings into betting decisions.

A horse is not a good bet just because it has the highest score. It is a good bet only when the offered odds are bigger than the model-implied fair odds.

### Convert Score to Probability

For each race, transform final model scores into probabilities using softmax.

```python
import numpy as np

def softmax(scores, temperature=8):
    scores = np.array(scores)
    adjusted = scores / temperature
    exp_scores = np.exp(adjusted - np.max(adjusted))
    return exp_scores / exp_scores.sum()
```

Lower temperature = more aggressive probabilities.
Higher temperature = more conservative probabilities.

### Market Implied Probability

```python
market_prob = 1 / decimal_odds
```

Adjust for overround:

```python
fair_market_prob = market_prob / sum(all_market_probs_in_race)
```

### Edge

```python
edge = model_prob - fair_market_prob
```

### Value Flag

```python
if edge >= 0.03 and decimal_odds >= 3.0:
    value_flag = True
else:
    value_flag = False
```

The threshold should be configurable.

---

## 7. Final Betting Decision Layer

### Output Categories

Use conservative labels:

```text
Strong Value
Value
Lean
Watch Only
Avoid
```

### Suggested Rules

```python
if edge >= 0.07 and condition_fit_score >= 80 and confidence >= 0.70:
    decision = "Strong Value"
elif edge >= 0.03 and condition_fit_score >= 75:
    decision = "Value"
elif race_relative_rank <= 3 and edge >= 0:
    decision = "Lean"
elif absolute_ability_rank <= 2 and condition_fit_score < 70:
    decision = "Watch Only"
else:
    decision = "Avoid"
```

### Never Recommend a Bet Without Explaining

Every recommendation must include:

```text
- Absolute ability rank
- Race-relative rank
- Condition fit
- Market odds
- Model probability
- Fair odds
- Edge
- Main reason
- Main risk
```

---

## 8. Explanation Template

For each race, generate a structured explanation.

Example output:

```markdown
## Race: Tokyo 11R

### Top Value Candidate: Horse A

- Absolute Ability Rank: 1/16
- Race-Relative Rank: 3/16
- Condition Fit: 86/100
- Current Odds: 8.0
- Model Probability: 18.2%
- Fair Odds: 5.49
- Market-Implied Probability: 12.5%
- Estimated Edge: +5.7%

Horse A is interesting because its last two runs rate better on absolute ability than the finishing positions suggest. Both came in stronger-than-average fields, and the most recent race had an unfavourable slow pace for a closer. Today’s larger field and expected stronger tempo should suit better.

Main risk: the horse is stepping back in distance, and its best figures have come over 2000m rather than today’s 1800m.
```

---

## 9. Architecture

Suggested project structure:

```text
betting-assistant/
├── data/
│   ├── horses.csv
│   ├── races.csv
│   ├── runs.csv
│   └── odds.csv
├── src/
│   ├── data_loader.py
│   ├── feature_engineering.py
│   ├── ability_score.py
│   ├── condition_fit.py
│   ├── race_relative.py
│   ├── market_value.py
│   ├── explanations.py
│   ├── config.py
│   └── main.py
├── notebooks/
│   └── score_validation.ipynb
├── tests/
│   ├── test_ability_score.py
│   ├── test_condition_fit.py
│   └── test_market_value.py
├── outputs/
│   └── race_reports/
└── README.md
```

---

## 10. Module Responsibilities

### `data_loader.py`

Responsibilities:

```text
- Load horses, races, runs, odds
- Validate required columns
- Standardise date formats
- Merge tables into modelling dataset
- Handle missing values safely
```

---

### `feature_engineering.py`

Responsibilities:

```text
- Build horse historical features
- Calculate recent averages
- Calculate best recent figures
- Calculate field strength
- Calculate pace indicators
- Calculate running style
- Calculate distance/going/surface history
```

---

### `ability_score.py`

Responsibilities:

```text
- Calculate base speed figure
- Calculate class adjustment
- Calculate field strength adjustment
- Calculate pace adjustment
- Calculate margin adjustment
- Calculate final absolute ability score
```

---

### `condition_fit.py`

Responsibilities:

```text
- Score today’s distance suitability
- Score going suitability
- Score surface suitability
- Score track suitability
- Score class suitability
- Score draw suitability
- Combine into condition fit score
```

---

### `race_relative.py`

Responsibilities:

```text
- Compare declared runners in a race
- Rank by recent ability
- Rank by best ability
- Combine with condition fit and pace setup
- Produce relative field ranking
```

---

### `market_value.py`

Responsibilities:

```text
- Convert model score to probability
- Remove bookmaker overround
- Calculate fair odds
- Calculate edge
- Assign value labels
```

---

### `explanations.py`

Responsibilities:

```text
- Produce human-readable race summaries
- Explain why a horse is upgraded or downgraded
- Separate ability, suitability, and price reasoning
- Include main risk for each recommendation
```

---

## 11. Configuration

Create `src/config.py`.

Example:

```python
VALUE_EDGE_THRESHOLD = 0.03
STRONG_VALUE_EDGE_THRESHOLD = 0.07

MIN_ODDS_FOR_VALUE = 3.0

SOFTMAX_TEMPERATURE = 8

ABILITY_WEIGHTS = {
    "base_speed": 1.00,
    "class": 0.25,
    "field_strength": 0.35,
    "pace": 1.00,
    "weight": 1.00,
    "going": 1.00,
    "margin": 1.00,
}

RACE_RELATIVE_WEIGHTS = {
    "recent_absolute_ability": 0.35,
    "best_recent_absolute_ability": 0.20,
    "condition_fit": 0.15,
    "pace_setup": 0.10,
    "form_trajectory": 0.08,
    "jockey_trainer": 0.07,
    "draw": 0.05,
}
```

---

## 12. Validation Requirements

The assistant must not just generate scores. It must test whether they are useful.

### Backtesting

Implement a backtest that checks:

```text
- Win strike rate by score rank
- Place strike rate by score rank
- ROI by value bucket
- Calibration of model probability
- Performance by race class
- Performance by distance type
- Performance by odds band
```

### Calibration Buckets

Group runners by model probability:

```text
0-5%
5-10%
10-15%
15-20%
20-30%
30%+
```

For each bucket, compare:

```text
average predicted probability
actual win rate
number of runners
```

If horses predicted at 20% only win 10%, the model is overconfident.

---

## 13. Important Anti-Bugs

### Do not leak future data

When calculating previous ability scores, only use races before the target race date.

Bad:

```python
horse_average_score = mean(all_runs_for_horse)
```

Good:

```python
horse_average_score = mean(runs_before_today)
```

---

### Do not overvalue finishing position

A horse’s finishing position is not the same as its performance.

Bad:

```python
score = 100 - finish_position
```

Better:

```python
score = function(time, margin, class, pace, field_strength, weight)
```

---

### Do not make odds part of ability

Odds can be used in the final value layer, but not in the absolute ability layer.

Reason:

```text
Odds are market opinion. If odds are used too early, the assistant may simply reproduce the market instead of finding mispricing.
```

---

### Keep ability and suitability separate

A horse can be high ability but low suitability today.

Example:

```text
High ability + unsuitable distance = Watch Only
Medium ability + ideal conditions + overpriced odds = Value
```

---

## 14. Minimum Viable Product

Build in this order:

### Phase 1: Deterministic scoring

```text
1. Load data
2. Calculate absolute ability score
3. Calculate condition fit score
4. Calculate race-relative score
5. Convert scores into model probabilities
6. Compare with odds
7. Generate markdown report
```

### Phase 2: Backtesting

```text
1. Run historical race simulation
2. Calculate ROI by decision label
3. Check probability calibration
4. Tune score weights
```

### Phase 3: Model upgrade

Replace or supplement manual weights with:

```text
- Logistic regression
- Gradient boosting
- XGBoost / LightGBM
- Bayesian hierarchical model
```

Still keep the explanation layers. Do not turn it into a black box.

---

## 15. First Coding Task for Antigravity

Implement the deterministic MVP.

### Task

Create the project structure and implement the following pipeline:

```text
Input:
- data/horses.csv
- data/races.csv
- data/runs.csv
- data/odds.csv

Output:
- outputs/race_reports/{race_id}.md
```

For a given `race_id`, the program should:

```text
1. Load declared runners.
2. Pull each horse’s historical runs before the race date.
3. Calculate absolute ability score for historical runs.
4. Calculate recent and best ability summaries.
5. Calculate condition fit for today’s conditions.
6. Calculate race-relative score.
7. Convert final scores to probabilities.
8. Compare probabilities against odds.
9. Generate a markdown betting report.
```

Command:

```bash
python -m src.main --race-id RACE_ID
```

Expected output:

```text
outputs/race_reports/RACE_ID.md
```

---

## 16. Report Format

Generated report should follow this structure:

```markdown
# Race Report: {race_name or race_id}

## Summary

- Top Absolute Ability Horse:
- Top Race-Relative Horse:
- Best Value Candidate:
- Favourite Vulnerability:
- Overall Race Confidence:

## Runner Table

| Horse | Abs Ability | Relative Rank | Condition Fit | Odds | Model Prob | Fair Odds | Edge | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---|

## Value Candidates

### {Horse Name}

- Absolute ability:
- Race-relative rank:
- Condition fit:
- Current odds:
- Model probability:
- Fair odds:
- Edge:
- Decision:

Reason:
Risk:

## Avoid / Oppose

### {Horse Name}

Reason:
```

---

## 17. Development Standards

Use:

```text
Python 3.11+
pandas
numpy
pydantic or dataclasses
pytest
```

Rules:

```text
- Prefer transparent functions over hidden side effects.
- Each score component must be inspectable.
- Store component scores, not just final scores.
- Write unit tests for every scoring function.
- Never overwrite raw data.
- Never use future races when scoring historical examples.
- Every generated recommendation must include at least one reason and one risk.
```

---

## 18. Future Extensions

Potential upgrades:

```text
- Add sectional timing analysis
- Add expected pace map
- Add trip notes / trouble-in-running parser
- Add jockey-trainer interaction scores
- Add track bias detection
- Add going-specific performance profiles
- Add stable form indicators
- Add odds movement / steam detection
- Add Monte Carlo race simulation
- Add Kelly staking with conservative fractional Kelly
```

Do not implement staking until probability calibration has been tested.

---

## 19. Betting Safety Rule

The assistant should avoid language that guarantees outcomes.

Use:

```text
value candidate
model edge
overpriced
underpriced
positive expected value
risk
uncertainty
```

Avoid:

```text
sure win
lock
guaranteed
can’t lose
```

---

## 20. Success Criteria

The implementation is successful if:

```text
1. It produces transparent ability, suitability, and value scores.
2. It separates cross-race ability from within-race ranking.
3. It explains recommendations in human-readable terms.
4. It supports backtesting without future-data leakage.
5. It can be tuned without rewriting the whole system.
```
