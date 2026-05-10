# Advanced Betting Strategies: Implementation Plan

This document outlines the roadmap for expanding the UmaEdge betting engine beyond single-horse Win bets, introducing mathematical support for **Win Dutching** (multiple bets per race) and **Exotic Bets** (Exactas, Trifectas).

As requested, we will leave the current `Backtester` strictly evaluating single bets until the core alpha is fully verified, but this plan serves as the blueprint for our next evolutionary step.

---

## Phase 1: Win Dutching (Simultaneous Kelly Criterion)

Currently, the backtester is hardcoded to find the single best EV candidate per race. Dutching allows us to bet on multiple positive-EV horses in the same race, which smooths variance and captures more value.

**Implementation Steps:**
1. **Remove the 1-Bet Limit:** Refactor the loop in `models/backtest.py` to evaluate all horses in a race that meet the minimum `ev_threshold` and `min_model_prob`.
2. **Implement Mutually Exclusive Kelly:** We cannot use the independent Kelly formula because only one horse can win a race. The stakes must be sized together.
   * We will implement the **Simultaneous Kelly Strategy** for mutually exclusive outcomes.
   * We calculate the optimal set of horses to bet on by sorting them by expected return, adding them to the bet set one by one, and stopping when adding the next horse decreases the overall expected growth rate of the bankroll.
3. **Backtest Config Update:** Add a flag like `allow_dutching=True` to `BacktestConfig` to toggle this behavior.

---

## Phase 2: Exotic Probability Modeling (Harville Formula)

To bet on Exactas (picking 1st and 2nd in order) and Trifectas (1st, 2nd, and 3rd), we need to estimate the probability of those specific sequences occurring.

**Implementation Steps:**
1. **Probability Normalization:** First, we must normalize our model's win probabilities across the entire field so that they perfectly sum to 1.0 (100%).
2. **The Harville Formula:** We will implement the standard quantitative racing model for sequence probabilities:
   * **Exacta $P(A, B)$:** The probability that A wins and B is second is $P(A) \times \frac{P(B)}{1 - P(A)}$.
   * **Trifecta $P(A, B, C)$:** The probability of A, B, C finishing 1-2-3 is $P(A) \times \frac{P(B)}{1 - P(A)} \times \frac{P(C)}{1 - P(A) - P(B)}$.
3. **Discount Factors (Optional):** The Harville formula traditionally overestimates the probability of longshots finishing in 2nd/3rd. We may need to implement discounting parameters (like the Benter or conditional logit models) to shrink the place probabilities of heavy favorites and boost them for longshots.

---

## Phase 3: Data Acquisition & Schema Expansion

Our database schema (`entries` table) currently only stores `odds_win` and `odds_place`. Exotics have massive combinatorial pools, and we need this historical payout data to backtest our edge.

**Implementation Steps:**
1. **Database Schema:** Create a new `race_payouts` table to store the historical winning combinations and their final odds/payouts for Exacta, Quinella, and Trifecta pools.
2. **Scraper Updates:** Update `scraper/netkeiba.py` to parse the exotic payout tables from the race results pages.
3. **Backfill:** Run a targeted backfill job to populate the new `race_payouts` table for our 2018-2025 training/validation set.

---

## Phase 4: Engine Integration

Once we have the probabilities and the payout data, we combine them in the backtester.

**Implementation Steps:**
1. **Combinatorial Generation:** In `models/backtest.py`, if `bet_type="exacta"`, generate all possible pairs ($N \times (N-1)$ combinations).
2. **EV Evaluation:** Calculate the EV of every combination using our Harville probabilities vs. the scraped Exacta odds.
3. **Fractional Kelly for Exotics:** Apply Kelly sizing to the positive EV combinations, mindful of the fact that multiple exotic combinations (e.g., betting three different Exactas) are mutually exclusive.

---

## Summary
By keeping this modular, we can finish validating our single-bet Win alpha using the clean, odds-free model we are currently training. Once you are satisfied with that baseline Sharpe ratio, we can flip the switch on Phase 1 to immediately boost volume through Win Dutching, followed by the heavy data engineering required for Phase 3/4 Exotics.
