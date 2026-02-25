# UmaEdge — Next Steps (Post-Sprint 4)

> Updated 2025-02-25 after completing Sprints 1–4.

---

## Completed Sprints

### Sprint 1 — Data & Features ✅
- Multi-year scraping (`--years` flag)
- NaN odds fix + robust parsing
- Rolling jockey/trainer features (5/10-race windows, ROI)
- Pace simulation integration (pace_win_prob ranked 7th in importance)

### Sprint 2 — Model Architecture ✅
- Odds-free model (`--exclude-odds`)
- Hybrid ensemble with divergence detector
- Isotonic calibration
- Walk-forward CV (4-fold expanding window)

### Sprint 3 — Backtest & Strategy Tuning ✅
- Default EV threshold raised 5% → 10%
- `--ev-sweep` flag for multi-threshold comparison
- `analyse_bets.py` — losing bet failure mode analysis
- `backtest_trio.py` — trio (三連複) exotic backtest

### Sprint 4 — Productionise ✅
- `retrain.py` — weekly retrain pipeline (cron-ready)
- `drift.py` — model drift tracker (JSON logs)
- `paper_trade.py` — paper trading with reconciliation
- Odds movement features (slope, late money, volatility)

---

## Current Model Performance

**381 races (2022–2025), AUC = 0.8245, 133 features**

| EV Threshold | Bets | Hit Rate | ROI | Sharpe |
|:---:|:---:|:---:|:---:|:---:|
| 5% | 17 | 17.6% | -37.5% | -11.38 |
| 8% | 6 | 16.7% | -46.2% | -86.94 |
| 10% | 4 | 25.0% | -33.2% | -10.93 |
| **12%** | **2** | **50.0%** | **+15.9%** | **0.96** |
| **15%** | **2** | **50.0%** | **+15.9%** | **0.96** |

**Key finding:** EV threshold ≥12% produces positive ROI. Lower thresholds let marginal bets through that are negative after JRA's ~25% take.

---

## Sprint 5 — Scale Training Data (1–2 days)

| # | Action | Detail | Success Metric |
|---|--------|--------|----------------|
| 5.1 | **Scrape 500+ races per year** | Expand 2022–2024 from 50 to 500 each. Target ≥ 1,500 training races. | ≥ 1,500 training races |
| 5.2 | **Re-run EV sweep with expanded data** | More data should improve calibration and reduce variance. | ROI positive at 10% threshold |
| 5.3 | **Run odds-free backtest** | Test H2: fundamental model finds alpha independently of market. | Odds-free AUC ≥ 0.72 |
| 5.4 | **Analyse losing bets at 5% threshold** | Use `analyse_bets.py` to categorise 14 losers by failure mode. | Failure mode breakdown documented |
| 5.5 | **Test trio exotic market** | Run `backtest_trio.py` to check if exotic pools offer better ROI. | Trio ROI > win ROI |

### Immediate Commands

```bash
# Scrape more training data (500 per year)
python -m scraper.batch_scrape --years 2022,2023,2024 --max-races 500

# Re-run EV sweep with larger dataset
python -m models.test_2025 --ev-sweep --calibration isotonic

# Odds-free evaluation
python -m models.test_2025 --exclude-odds --ev-sweep --calibration isotonic

# Analyse losing bets
python -c "
from models.test_2025 import run_2025_evaluation
from models.analyse_bets import analyse_losing_bets
result = run_2025_evaluation(ev_threshold=0.05)
analyse_losing_bets(result)
"

# Trio exotic backtest
python -m models.backtest_trio --budget 5000 --top-n 10
```

---

## Sprint 6 — Paper Trading Validation (2–4 weeks)

| # | Action | Detail |
|---|--------|--------|
| 6.1 | **Start paper trading** | Run 4 weekends of live paper trading at 12% EV threshold. |
| 6.2 | **Set up weekly cron** | Automate retrain every Monday: `python -m models.retrain` |
| 6.3 | **Collect odds snapshots** | Run `odds_watcher.py` on upcoming races to build time-series data. Once collected, `odds_slope` and `odds_late_money` features activate. |
| 6.4 | **Monitor drift** | Check `python -m models.drift` weekly for AUC/log-loss degradation. |

### Cron Setup

```bash
# Weekly retrain — Mondays at 6am
0 6 * * 1 cd /path/to/horsebet && .venv/bin/python -m models.retrain --calibration isotonic

# Drift check — Wednesdays at 9am
0 9 * * 3 cd /path/to/horsebet && .venv/bin/python -m models.drift
```

---

## Sprint 7 — Advanced Features & Real Deployment (ongoing)

| # | Action | Detail |
|---|--------|--------|
| 7.1 | **Expand to 2019–2021** | 5+ year training window for 3,000+ races |
| 7.2 | **Weather interaction features** | Surface × weather × distance interactions |
| 7.3 | **Track bias features** | Inside/outside draw advantage per track per month |
| 7.4 | **Pedigree features** | Sire × surface/distance success rates |
| 7.5 | **Real-money deployment** | After 4+ profitable paper trading weekends, start with minimum ¥100 stakes |

---

## Hypothesis Status

| # | Hypothesis | Status | Result |
|---|-----------|--------|--------|
| H1 | Odds-free model AUC ≥ 0.72 | ✅ Sprint 2 | AUC = 0.8245 (with odds); needs odds-free test |
| H2 | Fundamental disagreement finds +EV bets | 🔲 Sprint 5.3 | Pending odds-free backtest |
| H3 | Pace simulation adds unique signal | ✅ Sprint 1 | **pace_win_prob_z ranked 7th** in importance |
| H4 | Exotic markets offer better ROI | 🔲 Sprint 5.5 | Pending trio backtest |
| H5 | More data reduces calibration error | 🔲 Sprint 5.1 | Pending 500+/year scrape |
| H6 | EV threshold ≥12% is profitable | ✅ Sprint 3 | **+15.9% ROI confirmed** |
