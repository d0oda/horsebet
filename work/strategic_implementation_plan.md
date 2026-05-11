# Strategic Roadmap: Overcoming the 20% JRA Rake

## User Review Required
> [!IMPORTANT]
> Please review the prioritized options below. Option 1 is the fastest path to positive ROI, while Options 2 and 3 require more R&D but offer theoretically higher absolute edges. Let me know which path you'd like to actively pursue.

## Priority 1: Re-Embrace the Hybrid "Odds-Aware" Model (Production Readiness)
The public market is highly efficient. Instead of ignoring it to prove a point about pure alpha, we should use `odds_win` as the baseline probability anchor, and let our new Path 3 Pace/Shape features adjust those probabilities. We already know this configuration yields highly positive ROI.

### Implementation Steps
1. **Remove `--odds-free` constraints:** Ensure `odds_win` and its z-score derivatives are fed into the training matrix.
2. **Backtest Calibration:** Run a full walk-forward backtest on the Hybrid model using a realistic EV threshold (e.g., 0.15) to generate sustainable betting volume.
3. **Kelly Criterion Tuning:** Apply fractional Kelly sizing (e.g., `kelly_fraction = 0.25`) to limit variance while maximizing the compounding bankroll growth.
4. **Deploy to Pipeline:** Update `scripts/cron_pipeline.sh` to use the optimized hybrid model for upcoming race cards, ensuring the `frontend/app/page.tsx` displays the correct hybrid EV predictions.

---

## Priority 2: Switch to LambdaRank (Algorithmic Edge)
Horse racing is a zero-sum, cross-sectional contest ("who beats whom"), not an independent binary classification problem ("will this horse win? Yes/No"). Since our new Path 3 features are all relative field rankings (e.g., `best_class_rank_z`, `speed_figure_last_z`), a Pairwise Ranking model (LambdaRank) is perfectly suited to extract alpha from this data.

### Implementation Steps
1. **Label Transformation:** Utilize the existing `_finish_to_relevance` logic in `walk_forward_backtest.py` to convert finish positions (1st, 2nd, 3rd) to graded relevance labels (e.g., 3, 2, 1, 0).
2. **Backtest Evaluation:** Execute `walk_forward_backtest.py` combining the `--ranker` flag with the `--odds-free` flag.
3. **Softmax Conversion:** Monitor the `scores_to_probs` function to ensure raw LamdaRank output scores are being correctly calibrated into bounded win probabilities per race.
4. **Compare Baselines:** Compare the LambdaRank ROI against the standard Binary Classification ROI (-7.3%) to see if the algorithmic optimization alone can overcome the 20% rake without market data.

---

## Priority 3: Attack the Exotic Pools (Structural Edge)
The Win pool is ruthlessly efficient. Exotic pools (Exactas, Trifectas) are much harder for the public to price correctly, especially regarding correlated outcomes like pace-meltdowns. Our fundamental model is highly predictive of pace shape (Path 3), making it ideal for simulating and exploiting exotic combinations.

### Implementation Steps
1. **Monte Carlo Engine:** Leverage the existing `models/simulate_race.py` to run 10,000 simulations per race based on the fundamental model's win/place probabilities and pace correlations.
2. **Scrape Exotic Odds:** Enhance the `scraper` suite to fetch live Exacta/Trifecta matrix payouts from Netkeiba.
3. **EV Identification:** Compare the simulated hit rate of an Exacta combination (e.g., 12.5% chance) against the public exotic odds (e.g., payout of 20.0x) to identify massive structural EV (+150%).
4. **Betting Strategy:** Integrate the exotic EV calculations into `agents/exotic_bets.py` for automated bet slip generation.
