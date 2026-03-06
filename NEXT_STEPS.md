# UmaEdge — Audit & Next Steps

> Generated: 2026-03-05
> Basis: Full audit of `TODO.md`, project files, recent conversation history, and idea.txt roadmap.

---

## Audit Summary

### ✅ Completed (Model & Data — Phase 1 Complete)

| Area | Status | Evidence |
|------|--------|----------|
| Data pipeline | **Done** | 122k entries, 31k horses, 100% column coverage |
| Feature engineering | **Done** | Sprint 8+9 complete, 251 features, verified |
| Model training | **Done** | Hybrid ensemble AUC 0.8385 |
| Backtesting | **Done** | 78-day multiday backtest, +18.8pp edge over favorites |
| Drift detection | **Done** | Z-scored features stable, win rate stable |
| Paddock NLP | **Done** | Gemini-powered scorer, 17/17 tests pass |
| Pace simulation | **Done** | Recalibrated from 122k results, 27/27 tests pass |

### 🟡 Built But Untested (Code Exists, Never Run on Live Data)

| Module | File | Size | Notes |
|--------|------|------|-------|
| Race day pipeline | `pipeline.py` | 527 lines | 6 steps (scrape → odds → paddock → predict → frontend → results), never run end-to-end on a real race day |
| Odds watcher | `scraper/odds_watcher.py` | 229 lines | Stream mode + single-race mode, fetches from netkeiba API |
| Paper trader | `models/paper_trade.py` | 276 lines | `PaperTrader` class with place/reconcile/summary, saves to JSON |
| Live trader | `models/live_trade.py` | 510 lines | Full guardrails (¥100 stake, ¥5k daily max, 4 profitable paper weekends required) |
| Bankroll agent | `agents/bankroll.py` | 20k | Kelly criterion sizing |
| Exotic bets agent | `agents/exotic_bets.py` | 20k | Trio/trifecta combination constructor |
| Race analyst agent | `agents/race_analyst.py` | 17k | Natural-language previews |
| LINE notifications | `notifications/line.py` | 3k | LineNotifier class (needs `LINE_CHANNEL_TOKEN`) |
| Telegram notifications | `notifications/telegram.py` | 3k | TelegramNotifier class (needs `TELEGRAM_BOT_TOKEN`) |
| Notification dispatcher | `notifications/dispatcher.py` | 3.6k | Auto-detects configured backends |

### 🔵 Frontend Wishlist (Low Priority)

Per-venue ROI charts, odds slider, backtest viz improvements, bankroll dashboard polish — all nice-to-have.

---

## 🎯 Recommended Next Steps (Priority Order)

The core model is strong (AUC 0.8385, +18.8pp edge). The gap is between **"model that backtests well"** and **"system that makes money on race day."** Every step below is designed to close that gap.

---

### Step 1 — Dry-Run the Pipeline on a Real Race Day  
**Priority: 🔴 Critical** | **Effort: 1 session** | **Risk: Low**

The single highest-value thing you can do. `pipeline.py` has never been run end-to-end.

```bash
# Pick an upcoming JRA race day
python pipeline.py --date 2026-03-08
```

**What to check:**
- [ ] `step_scrape` discovers and ingests race cards correctly
- [ ] `step_odds` fetches live odds from netkeiba (even stale ones are fine)
- [ ] `step_paddock` runs Gemini scoring on any available comments
- [ ] `step_predict` generates a predictions JSON with EV flags
- [ ] `step_frontend` builds `data.json` for the dashboard
- [ ] `step_results` scrapes post-race results and reconciles

**Expected outcome:** A `predictions_YYYY-MM-DD.json` file with ranked bets, and a working dashboard.

---

### Step 2 — Start Paper Trading  
**Priority: 🔴 Critical** | **Effort: 2–3 weekends** | **Risk: None**

The live trader **requires** 4 profitable paper weekends (`REQUIRED_PAPER_WEEKENDS = 4` in `live_trade.py`). Start accumulating that track record now.

```bash
# On race day, after predictions exist:
python -m models.paper_trade --date 2026-03-08 --place

# After races finish:
python -m models.paper_trade --reconcile
python -m models.paper_trade --summary
```

**What to track:** Keep a running log of:
- Hit rate vs predicted probability
- Actual ROI per weekend
- Calibration drift (are 30% shots winning ~30% of the time?)

---

### Step 3 — Wire Up Notifications  
**Priority: 🟡 Medium** | **Effort: 30 min** | **Risk: None**

The code is 100% done. You just need API tokens in `.env`:

```env
# Choose one or both:
TELEGRAM_BOT_TOKEN=<from @BotFather>
TELEGRAM_CHAT_ID=<your chat ID>

LINE_CHANNEL_TOKEN=<from LINE Developers console>
LINE_USER_ID=<your LINE user ID>
```

Then the dispatcher auto-detects configured backends. Test with:
```python
from notifications.dispatcher import notify_message, get_status
print(get_status())
notify_message("UmaEdge test notification 🐴")
```

**Why now:** You want notifications before you start paper trading so you get push alerts for positive-EV bets as they appear.

---

### Step 4 — Validate the Agents on Real Data  
**Priority: 🟡 Medium** | **Effort: 1 session per agent** | **Risk: Low**

All three agents exist but haven't been tested with real predictions.

| Agent | Test Command | What to Validate |
|-------|-------------|-----------------|
| Race Analyst | `python -m agents.race_analyst --race <ID>` | Is the NL preview coherent? Does it reference the right horses? |
| Bankroll | `python -m agents.bankroll --backtest results/data.json` | Are Kelly fractions sane (typically 1–5% of bankroll)? |
| Exotic Bets | `python -m agents.exotic_bets --race <ID>` | Do trio/trifecta combinations cover reasonable horses? Is the budget constraint working? |

---

### Step 5 — Set Up Cron for Automated Race Days  
**Priority: 🟡 Medium** | **Effort: 1 hour** | **Risk: Low (after Steps 1–2 prove stable)**

Once the pipeline works manually, automate it:

```bash
# Example crontab entry (run at 6 AM JST on Sat/Sun)
0 6 * * 6,0 cd /Users/ryfei.wang/Documents/horsebet && python pipeline.py --date $(date +\%Y-\%m-\%d) >> logs/pipeline_$(date +\%Y\%m\%d).log 2>&1
```

Consider adding:
- A `--notify` flag to pipeline.py that calls the notification dispatcher at key steps
- A post-race reconciliation cron at 6 PM

---

### Step 6 — Go Live (After 4 Profitable Paper Weekends)  
**Priority: ⚪ Gated** | **Effort: Minimal (guardrails built in)** | **Risk: Real money**

`live_trade.py` already enforces:
- ¥100 default flat stake
- ¥500 max per bet
- ¥5,000 daily loss cap
- 4 profitable paper weekends prerequisite

```bash
python -m models.live_trade --race <ID> --confirm
```

> [!CAUTION]
> Only proceed after 4+ profitable paper weekends. The model has a theoretical edge, but slippage, odds movement on entry, and real-market dynamics can erode it.

---

## 🔮 Research Backlog (After Live Trading Stabilizes)

These are from `idea.txt` and the TODO nice-to-have section:

| Item | Impact | Effort | Notes |
|------|--------|--------|-------|
| Race replay CV analysis | 🔴 High | 🔴 High | Stride/traffic/running line — nobody prices this. Biggest potential edge. |
| Odds microstructure features | 🟡 Med | 🟡 Med | Late smart money vs early fan money — need odds_snapshots history first |
| Continuous online learning | 🟡 Med | 🟡 Med | Retrain on rolling window, detect regime changes |
| Content engine (race previews) | 🟡 Med | 🟢 Low | Race analyst agent is already 90% of this — add a web frontend |
| Exotic bet market inefficiency | 🟡 Med | 🟡 Med | Trio/trifecta pools have higher take but more mispricing |

---

## Summary: Where You Are on the Roadmap

From `idea.txt`:

| Phase | Status |
|-------|--------|
| Phase 1 — Research tool (no betting) | ✅ **Complete** |
| Phase 2 — Value detector + paper trade | 🟡 **Code ready, needs live testing** |
| Phase 3 — Bankroll agent + real money | ⚪ **Gated on paper trading results** |

**Bottom line:** The model and infrastructure are done. The gap is execution — running the pipeline on real race days, building paper-trading history, and then going live with guardrails. Steps 1–3 can all happen this weekend.
