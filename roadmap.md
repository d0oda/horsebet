# UmaEdge — Roadmap

> Updated 2026-02-26 — all sprints complete.

---

## Completed

| Sprint | What | Key Result |
|--------|------|------------|
| 1 | Data & Features | Multi-year scraping, rolling jockey/trainer features, pace simulation |
| 2 | Model Architecture | Odds-free model, hybrid ensemble, isotonic calibration, walk-forward CV |
| 3 | Backtest & Strategy | EV sweep, losing bet analysis, trio exotic backtest |
| 4 | Productionise | Retrain pipeline, drift tracker, paper trading, odds movement features |
| 5 | Evaluation Tooling | `run_evaluation.py` (steps 1–5), decision matrix, analyse_bets CLI |
| 6 | Paper Trading Infra | `run_sprint6.py` orchestrator, cron setup, weekend workflow |
| 7.1 | Data Expansion | `scripts/expand_data.sh` — scrape 2019–2021, targets updated to 3,100+ |
| 7.2–7.4 | Advanced Features | Weather interactions, track bias, pedigree features (~148 total) |
| 7.5 | Real-Money Trading | `live_trade.py` with guardrails (¥100 stake, ¥5k/day limit, --confirm) |
| 8 | Notifications | Telegram + LINE backends, dispatcher, API test endpoint |
| 9 | Deployment | Dockerfile, Vercel, Render blueprint, docker-compose, deploy script |

---

## Next: Run Evaluation

```bash
python -m models.run_evaluation
```

This runs all 5 post-scrape steps and prints a decision matrix:

| Result | Action |
|--------|--------|
| ROI positive at 10% EV | ✅ Lower live threshold 12% → 10% |
| ROI only positive at 12%+ | Scrape more data (2019–2021) |
| Odds-free AUC ≥ 0.72 | ✅ Fundamental edge confirmed — safe to trade |
| Odds-free AUC < 0.72 | 🔴 Model echoing odds — rethink features |
| Trio ROI > Win ROI | ✅ Add trio to paper trading |

---

## Next: Paper Trade 4 Weekends

```bash
# Place paper bets on upcoming races
python -m models.run_sprint6 --paper-trade <RACE_IDS>

# Reconcile after results are in
python -m models.run_sprint6 --reconcile

# Install weekly retrain + drift cron
./scripts/setup_cron.sh --install
```

**Target:** 4 profitable weekends before real money.

---

## Next: Go Live (Sprint 7.5)

```bash
# Check guardrails
python -m models.live_trade --check

# Place live bets (¥100/bet, requires --confirm)
python -m models.run_sprint6 --live <RACE_IDS> --confirm

# Reconcile live trades
python -m models.run_sprint6 --live-reconcile

# View live P&L
python -m models.run_sprint6 --live-summary
```

**Guardrails:** ¥100 flat stake, ¥500 max/bet, ¥5,000/day loss limit, 4 profitable paper weekends required.

---

## Deploy

```bash
# Frontend → Vercel
./scripts/deploy.sh --frontend

# Backend → Render/Railway
./scripts/deploy.sh --backend

# Local development
docker compose up
```

---

## Notifications

```bash
# Test (set TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID in .env first)
curl -X POST http://localhost:8000/api/notifications/test
```

Bets and reconciliation results are auto-sent to all configured backends (Telegram, LINE).
