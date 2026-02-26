# UmaEdge — Next Steps

> Based on evaluation results from 2026-02-26.  
> Model: AUC 0.8476 | ROI +23.6% at 5% EV | 1,881 races

---

## Immediate (This Week)

### 1. Paper Trade at 5% EV Threshold
The model is profitable. Start accumulating the required 4 profitable weekends.
```bash
# Place paper bets on upcoming race weekend
python -m models.run_sprint6 --paper-trade <RACE_IDS> --ev-threshold 0.05

# After results come in, reconcile
python -m models.run_sprint6 --reconcile

# Check progress
python -m models.run_sprint6 --summary
```

### 2. Install Automated Cron Jobs
```bash
./scripts/setup_cron.sh --install
```
This sets up weekly retrain + drift monitoring.

---

## Short-Term (Next 2–4 Weeks)

### 3. Scrape 2019–2021 Data
The evaluation shows 1,881 races (target: 3,100). More data should improve model robustness, especially for the "no_edge" failure mode.
```bash
./scripts/expand_data.sh --retrain
```

### 4. Fix the "No Edge" Failure Mode
73% of losing bets had `model_prob ≈ market_prob` — the model overestimates winners by ~8pp. Potential fixes:

- **Sharper EV filter**: Raise minimum EV from 5% → 8% (reduces bets from 18→8 but ROI improves 23.6%→28.4%)
- **Add features**: Race class changes, trainer form last 14 days, course-specific jockey stats
- **Calibration tuning**: Current isotonic calibration slightly worsens LogLoss (0.2174→0.2270) — try Platt scaling or reduce calibration aggression
- **Min-odds filter**: Add a `min_odds ≥ 2.5` filter to avoid tight-margin bets

### 5. Set Up Notifications
```bash
# Add to .env:
TELEGRAM_BOT_TOKEN=<your_token>
TELEGRAM_CHAT_ID=<your_chat_id>

# Test:
curl -X POST http://localhost:8000/api/notifications/test
```

---

## Medium-Term (After 4 Profitable Paper Weekends)

### 6. Go Live with ¥100 Stakes
```bash
# Check guardrails first
python -m models.live_trade --check

# Place real bets (¥100/bet, requires --confirm)
python -m models.run_sprint6 --live <RACE_IDS> --confirm

# Monitor
python -m models.run_sprint6 --live-summary
```

### 7. Deploy to Production
```bash
./scripts/deploy.sh --all
```
- Frontend → Vercel
- API → Render (or Railway)

### 8. Scale Up Stakes
After 4+ profitable live weekends at ¥100, consider:
- Increase to ¥200–500 (stay under ¥500 max guardrail)
- Adopt quarter-Kelly sizing (already implemented in `live_trade.py`)
- Adjust `MAX_DAILY_LOSS` based on bankroll growth

---

## Research Backlog

| Priority | Idea | Expected Impact |
|----------|------|-----------------|
| 🔴 High | Add race class change features | Reduce no_edge failures |
| 🔴 High | Trainer last-14-day form | Better short-term signals |
| 🟡 Med | Course × jockey interaction | Exploit specialist jockeys |
| 🟡 Med | Place/show betting (2nd/3rd) | 27% of losses finished 2nd/3rd |
| 🟢 Low | Weather × track surface interaction | Already in features, refine |
| 🟢 Low | Real-time odds streaming | Better entry timing |
