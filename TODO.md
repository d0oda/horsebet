# UmaEdge — To-Do

> Updated: 2026-02-28
> Model: Hybrid Ensemble (Platt, 60/40 fund/mkt blend)
> Feb 28 results: Favs (1-3x) +15.9% ROI ✅ | Longshots (30x+) -100% ❌ | Overall -73.8%

---

## 🔴 Priority 1 — Fix Longshot Bias

The model assigns ~1.5% floor probability to every horse, creating fake "value" at high odds.
206 bets at 30x+ → 0 winners → accounts for nearly all losses.

- [x] Recalibrate Platt scaling for low-probability region (< 5%)
- [x] Evaluate isotonic regression as alternative (better in tails)
- [x] Add `is_extreme_longshot` feature (odds > 50x) so model learns to discount
- [x] Hard-cap bettable odds at 30x (or sweep 20x/30x/50x on historical data)

## 🟡 Priority 2 — Market Model Reweighting

Market model should suppress longshot bets but isn't doing enough.
`log_odds` feature not influential enough in combined prediction.

- [x] Increase market model weight for high-odds horses
- [x] Add odds-aware blending (weight market more when odds diverge from fundamental)
- [x] Investigate why `log_odds` isn't suppressing longshot probabilities

## 🟡 Priority 3 — Bet Sizing & Selection

Flat ¥100/bet on 284 bets is poor bankroll management.
Favs-only subset (22 bets) was +15.9% — lean into strength.

- [x] Implement Kelly criterion position sizing
- [x] Add configurable EV + odds filters to `predict_final.py`
- [x] Backtest optimal odds ceiling on full historical data (not just 1 day)

## 🔵 Priority 4 — Evaluation & Data

Single-day results are noisy. Need multi-day validation.

- [ ] Run multi-day backtests to validate the fav-vs-longshot ROI split
- [ ] Scrape more 2025–2026 races for out-of-sample testing
- [ ] Track favorites-only baseline vs full model
- [ ] Build Sharpe ratio tracking across race days

## ⚪ Priority 5 — Nice to Have

- [ ] Add odds range slider to results webpage
- [ ] Per-venue ROI breakdown charts
- [ ] Auto-generate results page after each prediction run
- [ ] Pedigree (sire) feature backfill — still at ~5%
- [ ] Weather interaction features

---

## ✅ Done

| Item | Result |
|------|--------|
| Fix scraper (encoding, draw/PP, dead API) | 36 races scraped clean |
| Repair Feb 28 data | 520 entries, 519 w/ odds, 515 w/ results |
| Run predictions | 36/36 races, model works end-to-end |
| Results webpage | `results/` — dark theme, venue tabs, sortable value bets |
| Hybrid ensemble + Platt | AUC 0.82 fund / 0.85 mkt |
| Feature engineering | 203 features, ~90 entries/s build speed |
| Scraper optimisation | Hierarchical probing, multi-worker |
