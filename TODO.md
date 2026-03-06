# UmaEdge — TODO

> Updated: 2026-03-05
> DB: 8,962 races | 31,835 horses | 122,164 entries | 122,164 results
> Data: **100% coverage** on all columns. Feature engineering complete (Sprint 8+9).

---

## 🔴 Model — Retrain & Validate

- [x] ~~Retrain model with cleaned features~~ — rebuilt 122k entries, 251 features, cached
- [x] ~~Run full backtest~~ — LGB AUC 0.8245, XGB AUC 0.8251, model ROI -2.5% vs fav -21.3%
- [x] ~~Run multi-day backtest~~ — 78 days, 41% profitable, +18.8pp edge over favorites
- [x] ~~Check feature importances~~ — Sprint 8/9 features (class_rank, sire, draw_bias, trainer) in top 15
- [x] ~~Evaluate ensemble model~~ — Hybrid Combined AUC 0.8385 beats LGB-only 0.8245 (+1.4pp)
- [x] ~~Run drift detection~~ — raw features drift as expected (cumulative), z-scored stable, win rate stable (7.39→7.12%)

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
