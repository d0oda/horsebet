# UmaEdge — Next Steps (Post-2025 Evaluation)

> Generated 2025-02-25 based on the 2025 holdout backtest results.

---

## Diagnosis: Why the Model Lost Money

The backtest returned **−18.0% ROI** across 32 bets on 100 races. Three root causes explain the result:

### 1. Market-Mirror Problem (Critical)
`odds_win` alone accounts for **6.7×** more feature importance than the second feature. The model is not discovering independent signal — it is just re-stating what the crowd already believes, then trying to bet against that same crowd. This is structurally incapable of generating alpha.

### 2. Data Starvation
Only **181 training races / 2,332 entries** to learn **96 features**. At ~24 entries per feature, the model cannot reliably separate signal from noise, especially for interaction effects (e.g., sire × surface, draw × distance).

### 3. Calibration Drift in the Value Zone
The model is well-calibrated at low probabilities (4% predicted → 3.2% actual), but **overconfident** in the 20–40% range where most value bets are flagged (25% predicted → 34.5% actual and 44% predicted → 33.3% actual). This means EV calculations are systematically wrong for the bets that matter most.

---

## Iteration Roadmap

### Sprint 1 — Fix the Data & Features (1–2 days)

| # | Action | Detail | Success Metric |
|---|--------|--------|----------------|
| 1.1 | **Expand training data to 2+ years** | Scrape 2022–2024 in full using `batch_scrape`. Target ≥ 1,000 races / 12,000+ entries. | ≥ 1,000 training races |
| 1.2 | **Fix NaN odds entries** | Debug scraper to handle missing odds gracefully; exclude entries with no valid odds from backtesting to stop flat ¥1,000 dilution bets. | 0 NaN-odds bets in backtest |
| 1.3 | **Add rolling jockey/trainer form features** | Compute 5-race and 10-race rolling win%, place%, and ROI for each jockey and trainer at query time. | 4+ new features added |
| 1.4 | **Integrate pace simulation output** | The Monte Carlo pace sim in `models/pace_sim.py` already exists but isn't wired into the feature pipeline. Feed its P(win\|pace) and running-style distribution into `features.py`. | Pace features appear in feature importance |

### Sprint 2 — Model Architecture Changes (1–2 days)

| # | Action | Detail | Success Metric |
|---|--------|--------|----------------|
| 2.1 | **Train an odds-free model** | Build a second model that **excludes all odds-derived features** (`odds_win`, `log_odds`, `odds_win_z`, `log_odds_z`, `popularity`, `popularity_z`). This forces the model to learn fundamental signals (speed, form, jockey, pace). | Odds features removed; model still achieves AUC ≥ 0.72 |
| 2.2 | **Create a hybrid ensemble** | Combine the odds-free model (fundamental view) with the odds-aware model (market view). The EV edge comes from cases where the fundamental model disagrees with the market model. | Ensemble defined; divergence metric created |
| 2.3 | **Improve calibration** | Apply isotonic regression (instead of Platt scaling) or use Venn-ABERS prediction for the 20–40% range. Re-check calibration table after. | Mid-range predicted vs actual within ±5% |
| 2.4 | **Walk-forward cross-validation** | Replace single time-split with expanding-window walk-forward CV (e.g., train on months 1–6, test on 7; train on 1–7, test on 8; etc.) for more robust metric estimates. | CV log-loss variance < 0.01 |

### Sprint 3 — Backtest & Strategy Tuning (1 day)

| # | Action | Detail | Success Metric |
|---|--------|--------|----------------|
| 3.1 | **Raise EV threshold to 10–15%** | Current 5% threshold is too loose and lets through marginal bets that are negative after JRA's ~25% take. | Fewer bets, higher hit rate |
| 3.2 | **Backtest the odds-free model separately** | Run the full backtest on the odds-free model to see if it finds genuine alpha independent of market. | Positive or breakeven ROI on 2025 holdout |
| 3.3 | **Analyse losing bets** | For the 27 losing bets (of 32), categorise failure mode: (a) model overconfident, (b) odds already reflected true value, (c) bad luck / variance. | Failure mode breakdown documented |
| 3.4 | **Experiment with exotic markets** | Use `agents/exotic_bets.py` to construct 三連複 (trio) tickets from model probabilities. Exotic pools are less efficient → more alpha potential. | Trio backtest ROI computed |

### Sprint 4 — Productionise & Monitor (ongoing)

| # | Action | Detail |
|---|--------|--------|
| 4.1 | **Weekly retrain pipeline** | Automate: scrape weekend results → retrain → deploy updated model → log calibration metrics. |
| 4.2 | **Model drift dashboard** | Track log-loss, AUC, and calibration plot on the frontend Backtest page, updated weekly. |
| 4.3 | **Paper trading** | Run 4 weekends of live paper trading with the improved model before any real stakes. |
| 4.4 | **Odds movement features** | Once `scraper/odds_watcher.py` is collecting time-series odds, add slope and late-money features to capture smart-money signal. |

---

## Key Hypotheses to Test

| # | Hypothesis | How to Test | Pass Condition |
|---|-----------|-------------|----------------|
| H1 | An odds-free model can achieve AUC ≥ 0.72 on 2025 holdout | Sprint 2.1 | AUC ≥ 0.72 |
| H2 | Fundamental-vs-market disagreement identifies positive-EV bets | Sprint 2.2, 3.2 | ROI ≥ 0% on holdout (breakeven or better) |
| H3 | Pace simulation adds unique signal not captured by other features | Sprint 1.4 | Pace features in top-20 importance; AUC improves ≥ 0.5% |
| H4 | Exotic (trio) markets offer better ROI than win-only | Sprint 3.4 | Trio ROI > win ROI |
| H5 | More training data (1,000+ races) reduces calibration error | Sprint 1.1 | Mid-range calibration gap < 5% |

---

## Immediate Next Command

```bash
# Step 1: Scrape 2 more years of data
python -m scraper.batch_scrape --year 2023 --max-races 500
python -m scraper.batch_scrape --year 2022 --max-races 500

# Step 2: Re-run evaluation with odds features removed
python -m models.test_2025 --exclude-odds --ev-threshold 0.10
```
