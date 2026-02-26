# UmaEdge — Post-Scrape Evaluation Plan

> Run these steps after `batch_scrape --years 2022,2023,2024 --max-races 500` completes.
> All evaluations use the full ~148 feature set (Sprints 1–4 + Sprint 7).

---

## 1. Verify Data Volume

```bash
python -c "
from scraper.db import get_supabase
sb = get_supabase()
for y in [2022, 2023, 2024, 2025]:
    rows = sb.schema('horsebet').table('races').select('id', count='exact').gte('date', f'{y}-01-01').lt('date', f'{y+1}-01-01').execute()
    print(f'{y}: {rows.count} races')
"
```

**Target:** ≥500 races/year for 2022–2024, ≥100 for 2025. Total ≥1,600 training races.

---

## 2. Full EV Sweep (with odds)

```bash
python -m models.test_2025 --ev-sweep --calibration isotonic
```

| What to check | Pass criteria |
|---|---|
| AUC | ≥ 0.82 (maintain or improve from 381-race baseline) |
| ROI at 10% EV | **Positive** (currently only positive at ≥12%) |
| Calibration | Mid-range (20–40% predicted) closer to actual |
| Sharpe at 12% | > 1.0 |

> More data + Sprint 7 features should improve calibration and lower the profitable EV threshold from 12% → 10%.

---

## 3. Odds-Free Evaluation (Hypothesis H2)

```bash
python -m models.test_2025 --exclude-odds --ev-sweep --calibration isotonic
```

| What to check | Pass criteria |
|---|---|
| Odds-free AUC | ≥ 0.72 |
| Top features without odds | Horse/jockey/pace features dominating, not just weight/draw |

> This confirms the model finds **independent alpha** rather than echoing the market.
> If AUC < 0.72, fundamental features aren't strong enough — rethink feature engineering before going live.

---

## 4. Losing Bet Analysis

```bash
python -m models.analyse_bets --ev-threshold 0.05
```

Categorise the losers by failure mode:
- **Calibration error** — model probability was too high
- **Pace collapse** — front-runner faded, model missed pace scenario
- **Surface/weather** — wrong-footed by going change
- **Class jump** — horse promoted beyond ability

Use this to prioritise the next feature iteration.

---

## 5. Trio Exotic Backtest (Hypothesis H4)

```bash
python -m models.backtest_trio --budget 5000 --top-n 10
```

| What to check | Pass criteria |
|---|---|
| Trio ROI | > Win ROI |
| Trio hit rate | Meaningfully higher than win hit rate |

> Exotic pools are less efficient — if trio ROI > win ROI, add trio to paper trading.

---

## Decision Matrix

| Result | Action |
|---|---|
| ROI positive at 10% | ✅ Lower live threshold from 12% → 10% |
| ROI only positive at 12%+ | Keep 12%, focus on adding more data (2019–2021) |
| Odds-free AUC ≥ 0.72 | ✅ Confirms fundamental edge — safe to trade |
| Odds-free AUC < 0.72 | 🔴 Model is echoing odds — need better fundamental features |
| Trio ROI > Win ROI | ✅ Add trio to paper trading portfolio |
| Trio ROI ≤ Win ROI | Stick with win bets only |
| Losing bets dominated by one failure mode | Target that mode in next feature sprint |

---

## After Evaluation → Sprint 6 (Paper Trading)

Once the numbers look good:

1. **Paper trade 4 weekends** at chosen EV threshold
   ```bash
   python -m models.paper_trade --ev-threshold 0.12
   ```
2. **Set up weekly retrain cron**
   ```bash
   # Mondays at 6am
   0 6 * * 1 cd /path/to/horsebet && .venv/bin/python -m models.retrain --calibration isotonic
   ```
3. **Start odds watcher** on upcoming races to activate movement features
   ```bash
   python -m scraper.odds_watcher --race-ids <upcoming>
   ```
4. **Monitor drift weekly**
   ```bash
   python -m models.drift
   ```

---

## Remaining Buildout

| Item | Status |
|---|---|
| Sprint 7.1 — Expand to 2019–2021 data | 🔲 After Sprint 5 confirms more data helps |
| Sprint 7.5 — Real-money deployment | 🔲 After 4+ profitable paper weekends |
| Notifications — LINE/Telegram alerts | 🔲 Not yet built |
| Deployment — Vercel + Render/Railway | 🔲 Not yet configured |
