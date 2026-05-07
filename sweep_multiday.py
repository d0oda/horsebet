import logging
import sys
from models.multiday_backtest import run_multiday_backtest

# Suppress debug/info logs from the backtester so we just see the summary
logging.getLogger("multiday_backtest").setLevel(logging.WARNING)
logging.getLogger("test_2025").setLevel(logging.WARNING)

thresholds = [0.05, 0.08, 0.10, 0.12, 0.15, 0.20]
calibrators = ['platt', 'isotonic']

print("=== EV Sweep: Platt vs Isotonic (Min Odds: 2.0, Max Odds: 30.0) ===")
print("Calibrator | EV Thresh | Bets | Staked  | Profit  | ROI")
print("-" * 65)

for cal in calibrators:
    for ev in thresholds:
        try:
            day_results = run_multiday_backtest(
                ev_threshold=ev,
                max_odds=30.0,
                min_odds=2.0,
                use_kelly=False,
                calibration_method=cal,
                output_path=None,
                rebuild=False
            )
            if not day_results:
                continue
                
            total_bets = sum(d.bets for d in day_results)
            total_staked = sum(d.staked for d in day_results)
            total_profit = sum(d.profit for d in day_results)
            roi = (total_profit / total_staked * 100) if total_staked > 0 else 0.0
            
            print(f"{cal:<10} | {ev:>9.0%} | {total_bets:>4} | ¥{total_staked:>6,} | ¥{total_profit:>6,} | {roi:>+6.2f}%")
        except Exception as e:
            print(f"Error on {cal} at {ev}: {e}")
