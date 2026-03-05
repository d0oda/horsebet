# UmaEdge — TODO

> Updated: 2026-03-05
> DB: 8,962 races | 31,835 horses | 122,164 entries | 122,164 results
> Data: **100% coverage** on all columns. Feature engineering complete (Sprint 8+9).

---

## 🔴 Model — Retrain & Validate

- [ ] Retrain model with cleaned features (dead odds movement features removed, margin parser fixed)
- [ ] Run full backtest (`models/backtest.py`) and compare ROI vs previous model
- [ ] Run multi-day backtest (`models/multiday_backtest.py`) to validate across 2021–2025
- [ ] Check feature importances — confirm new Sprint 8/9 features are getting picked up
- [ ] Evaluate ensemble model (`models/ensemble.py`) vs single model
- [ ] Run drift detection (`models/drift.py`) on recent vs historical data

---

## 🟡 Race Day Pipeline

- [ ] End-to-end test of `pipeline.py` for an upcoming race day
- [ ] Wire up odds streaming (`scraper/odds_watcher.py`) into pipeline for live race days
- [ ] Verify results scraping step (`step_results`) works post-race
- [ ] Set up cron / scheduled task for automated race day runs

---

## 🟡 Agents

- [ ] Test exotic bets agent (`agents/exotic_bets.py`) on real race data
- [ ] Test bankroll agent (`agents/bankroll.py`) Kelly sizing on backtest results
- [ ] Test race analyst agent (`agents/race_analyst.py`) output quality

---

## 🔵 Frontend

- [ ] Per-venue ROI breakdown charts
- [ ] Odds range slider on results page
- [ ] Backtest visualization improvements
- [ ] Bankroll tracking dashboard polish

---

## 🔵 Notifications

- [ ] Configure and test LINE notifications (`notifications/line.py`)
- [ ] Configure and test Telegram notifications (`notifications/telegram.py`)
- [ ] Set up pre-race alerts for positive EV bets

---

## ⚪ Nice to Have / Research

- [x] ~~Paddock comment NLP signal~~ — Gemini-powered scorer integrated into pipeline, 17/17 tests pass
- [ ] Race replay CV analysis (stride, traffic, running line)
- [x] ~~Pace simulation tuning~~ — recalibrated from 122k JRA results, 27/27 tests pass
- [ ] Live paper trading log to track theoretical P&L before real money

---

## 🔴 After Each Year Scrape — Re-run These

```bash
python -m scraper.backfill_running_style
python -m scraper.backfill_weights_netkeiba --workers 3
python -m scraper.backfill_sectionals --workers 3
python -m scraper.backfill_race_metadata --workers 2
```
