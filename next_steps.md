# UmaEdge — Next Steps

> Updated: 2026-02-28  
> Model: Hybrid Ensemble (Platt calibration, 60/40 fundamental/market blend)  
> Latest eval: 38 bets at 5% EV → +13.3% | 33 bets at 8% EV → +11.1%  
> ⚠️ IP currently blocked from netkeiba — scraping paused

---

## ✅ Completed

| Item | Detail |
|------|--------|
| Fix scraper columns | `course_id`, `trainer_id`, `race_class`, `draw` all populated |
| Odds snapshots | 25,122 backfilled from existing data, `--with-odds` flag added |
| Platt scaling | Replaced isotonic calibration — saved in `2025_hybrid/metadata.json` |
| Hybrid ensemble | Fundamental (odds-free, AUC 0.8248) + Market (odds-aware, AUC 0.8481) |
| Eval at 5% EV | 38 bets, 11 wins, hit rate ~29%, final balance ¥113,254 |
| Eval at 8% EV | 33 bets, 10 wins, final balance ¥111,104 |
| Frontend updated | Dashboard, backtest page, insights page all reflect 2025 results |
| Ensemble tests | `test_ensemble.py` — divergence detector, alpha signal, odds features |
| Scraper optimisation | Hierarchical probing, multi-worker, CSS class fix |
| Feature engineering perf | Vectorised history lookups, reduced pace sim count |

---

## 🟡 In Progress

### 1. Scrape 2019–2021 Data (IP Blocked)
Batch scrape was running (`--years 2019,2020,2021 --with-odds`) but netkeiba blocked the IP.  
**Resume when:** IP unblocked (likely 24–48h), or via VPN/mobile hotspot.  
**Target:** 3,100+ races (currently ~1,881).

---

## 🔴 Next Up (No Scraping Needed)

### 2. Re-evaluate with Current Data
The scraper fixes (class, trainer, draw, odds snapshots) may have added enough data to improve results even without 2019–2021. Retrain + eval to see:
```bash
python -m models.test_2025 --hybrid --ev-threshold 0.05
```

### 3. EV Threshold Sweep
Systematically test thresholds (3%, 5%, 8%, 10%, 12%) and compare ROI, bet count, and Sharpe ratio to find the optimal operating point.

### 4. Improve Feature Population
Run `python -m models.verify_features` to check which feature groups are now populated after the scraper fixes. Specifically:
- Pedigree (sire) — was at 4.4%, needs `backfill_sires` script
- Track bias — should now work with `draw` column populated
- Odds movement — should now work with odds_snapshots table

### 5. Update Frontend
Refresh the insights page with Platt calibration data and latest hybrid model results (currently shows isotonic + old decision matrix with "0 bets").

### 6. Experiment with Ensemble Methods
- Try CatBoost as a third base learner
- Tune fundamental/market weight ratio (currently 60/40)
- Stacking instead of weighted average

---

## Short-Term (After Model Improvement)

### 7. Paper Trade at Best Threshold
Once ROI is positive at the chosen EV threshold, paper trade for 4 profitable weekends.
```bash
python -m models.run_sprint6 --paper-trade <RACE_IDS>
```

### 8. Set Up Cron & Notifications
```bash
./scripts/setup_cron.sh --install
```

---

## Medium-Term (After 4 Profitable Paper Weekends)

### 9. Go Live with ¥100 Stakes
### 10. Deploy to Production
### 11. Scale Up Stakes (quarter-Kelly)
