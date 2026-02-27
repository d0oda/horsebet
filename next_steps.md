# UmaEdge — Next Steps

> Updated: 2026-02-26 (post-hybrid evaluation)  
> Model: AUC 0.8481 | **0 bets at 5% EV** (standard) | **6 bets at 5% EV** (hybrid) | 1,881 races  
> ⚠️ Retraining with new features broke profitability — but hybrid approach shows promise

---

## 🚨 Diagnosis: Why the Model Yields 0 Bets

After adding 6 new feature groups (class change, trainer 14d, course×jockey, place betting, weather, odds streaming) and retraining, the model produces **0 bets** at any EV threshold ≥ 5%.

**Root cause 1: Odds features dominate the model entirely.**

| Rank | Feature | Importance (gain) |
|------|---------|-------------------|
| 1 | `log_odds` | 11,007 |
| 2 | `odds_win` | 9,304 |
| 3 | `odds_win_z` | 1,467 |
| 4–15 | (pace, weight, etc.) | 573–1,102 |
| — | New features | **Not in top 15** |

- At 0% EV threshold: 113 bets, avg odds **165x**, hit rate **3.5%**, ROI **-87.9%**
  - Almost all bets are on extreme longshots (0.7–1.8% predicted win prob)
  - Max EV found was only **+3.96%**
- The model's predicted probabilities essentially **mirror the market odds** — there is no independent edge
- Isotonic calibration made it even worse (LogLoss 0.2182 → 0.2550), but removing calibration still showed 0 bets at 5%

**Root cause 2: 5 of 7 new feature groups are completely empty (all NaN).**

The scraper does not populate `course_id`, `trainer_id`, `race_class`, or `draw` in the DB. Without these columns, the feature functions return NaN for every entry. NaN features get median-filled to 0, adding pure noise.

| Feature Group | Avg Populated | Verdict |
|---------------|--------------|---------|
| Weather × Surface | 63.5% | ✅ OK |
| Pedigree (Sire) | 4.4% | ❌ DEAD |
| Class Change | 0.0% | ❌ DEAD |
| Trainer 14-Day Form | 0.0% | ❌ DEAD |
| Course × Jockey | 0.0% | ❌ DEAD |
| Track Bias | 0.0% | ❌ DEAD |
| Odds Movement | 0.0% | ❌ DEAD (not in dataframe) |

> Run `python -m models.verify_features` to reproduce this audit.

**Why it was profitable before:** The previous model (pre-retraining) happened to have a calibration sweet spot on the smaller feature set. Adding more features without data backing them diluted signal.

---

## ✅ Completed: Hybrid Model Evaluation

Ran `python -m models.test_2025 --hybrid --ev-threshold 0.05`:

### Model Quality
| Model | AUC | LogLoss | Brier |
|-------|-----|---------|-------|
| Fundamental (odds-free) | **0.8248** | 0.2256 | 0.0636 |
| Market (odds-aware) | 0.8481 | 0.2182 | 0.0630 |

### Fundamental Model Top Features (no odds)
| Rank | Feature | Importance |
|------|---------|-----------|
| 1 | `pace_place_prob` | 11,691 |
| 2 | `pace_place_prob_z` | 2,563 |
| 3 | `weight_carried_z` | 1,358 |
| 4 | `horse_weight_z` | 1,333 |
| 5 | `pace_style_stalk_z` | 1,216 |

### Divergence & Backtest
- **19 divergence signals** found (fundamental prob > market implied by ≥5%)
- **6 bets placed** (vs 0 with standard model)
- Hit rate: **16.7%** (1/6)
- ROI: **-25.2%** (¥-3,202 on ¥12,702 staked)
- The one winner: キントラダンサー at 3.8x odds

> The hybrid approach **works mechanically** — it finds bets where the standard model cannot. But the fundamental model isn't strong enough yet because 5/7 new features are dead.

---

## Immediate: Fix the Data (This Week)

### 1. ✅ Fix Scraper to Populate Missing DB Columns
Backfilled `races.class` (0%→100%), added trainer scraping to `netkeiba.py`, created `backfill_trainers.py` for existing data.
Run full trainer backfill: `python -m scraper.backfill_trainers`

### 2. ✅ Set Up Odds Snapshots Scraping
Backfilled 25,122 odds snapshots from existing `entries.odds_win`. Added `--with-odds` flag to batch scraper.

### 3. 🟡 Scrape 2019–2021 Data
Still only 1,881 races (target: 3,100). More training data will help any approach.
```bash
./scripts/expand_data.sh --retrain
```

### 4. 🟡 Re-run Hybrid After Data Fix
Once features are populated, re-run:
```bash
python -m models.test_2025 --hybrid --ev-threshold 0.05
```

### 5. 🟡 Try Platt Scaling Instead of Isotonic
Isotonic calibration overfits with small bins (n=3–7 in upper buckets). Platt scaling fits only 2 parameters and is more robust at this data volume.

---

## Short-Term (After Model Fix)

### 6. Paper Trade at Best Threshold
Once a model variant produces positive-EV bets, paper trade for 4 profitable weekends.

### 7. Set Up Notifications & Cron Jobs
```bash
./scripts/setup_cron.sh --install
```

---

## Medium-Term (After 4 Profitable Paper Weekends)

### 8. Go Live with ¥100 Stakes
### 9. Deploy to Production
### 10. Scale Up Stakes (quarter-Kelly)

---

## Research Backlog

| Priority | Idea | Expected Impact | Status |
|----------|------|-----------------|--------|
| 🔴 High | Fix scraper: course_id, trainer_id, race_class, draw | Unlock 4 dead feature groups | **Done** |
| 🔴 High | Set up odds_snapshots scraping | Enable odds movement features | **Done** |
| 🟡 Med | Scrape 2019–2021 data | More training data (1881→3100) | TODO |
| 🟡 Med | Platt scaling calibration | More robust than isotonic at low n | TODO |
| ✅ | Odds-free hybrid model | Found 19 signals, 6 bets, -25.2% ROI | **Done** |
| ✅ | Verify new feature population | 5/7 groups are DEAD (0% populated) | **Done** |
| ✅ | Add race class change features | Code done, but data missing | Done |
| ✅ | Trainer last-14-day form | Code done, but data missing | Done |
| ✅ | Course × jockey interaction | Code done, but data missing | Done |
| ✅ | Place/show betting | — | Done |
| ✅ | Weather × track surface | 63.5% populated, only working new group | Done |
| ✅ | Real-time odds streaming | Code done, but no snapshot data | Done |
