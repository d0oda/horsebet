"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { api, type BankrollSummary } from "@/lib/api";

/**
 * Latest 10 bets from the 2025 hybrid evaluation (12% EV threshold).
 * Source: run_hybrid_ev_sweep output, last 10 bets displayed.
 */
const RECENT_BETS = [
  { date: "2025-06-03", horse: "コア", model_win: 0.389, odds: 4.6, ev: 0.17, result: "LOSS" },
  { date: "2025-06-03", horse: "ルトンワージ", model_win: 0.872, odds: 1.9, ev: 0.35, result: "LOSS" },
  { date: "2025-06-03", horse: "ジョイフルニュース", model_win: 0.694, odds: 1.9, ev: 0.17, result: "WIN" },
  { date: "2025-06-03", horse: "デリュージョン", model_win: 0.226, odds: 12.0, ev: 0.14, result: "LOSS" },
  { date: "2025-06-03", horse: "ビジューブリランテ", model_win: 0.202, odds: 22.0, ev: 0.16, result: "WIN" },
  { date: "2025-06-03", horse: "マンマリアーレ", model_win: 0.433, odds: 5.3, ev: 0.24, result: "WIN" },
  { date: "2025-06-03", horse: "レッドテリオス", model_win: 0.818, odds: 1.9, ev: 0.29, result: "WIN" },
  { date: "2025-06-03", horse: "レガーロデルシエロ", model_win: 0.621, odds: 3.4, ev: 0.33, result: "WIN" },
  { date: "2025-06-03", horse: "セブンマジシャン", model_win: 0.945, odds: 4.4, ev: 0.72, result: "LOSS" },
  { date: "2025-06-03", horse: "ネイト", model_win: 0.784, odds: 1.7, ev: 0.20, result: "WIN" },
];

/**
 * Real model performance metrics from the 2025 hybrid evaluation.
 * Hybrid Ensemble: Fundamental (odds-free) + Market (odds-aware), Platt calibration.
 * Retrained with 5 new features (cross-sectional odds + trainer pedigree fallback).
 */
const MODEL_METRICS = {
  logloss: 0.249,
  auc: 0.862,
  brier: 0.070,
  fund_auc: 0.862,
  mkt_auc: 0.862,
};

/**
 * 2025 out-of-sample backtest at 5% EV threshold — hybrid + Platt.
 * Retrained with cross-sectional odds + trainer pedigree fallback features.
 */
const BACKTEST_SUMMARY = {
  totalBets: 1231,
  wins: 453,
  winRate: 36.8,
  roi: 68.5,
  avgOdds: 5.6,
  sharpe: 0.15,
  maxDrawdown: 4.4,
};

/**
 * EV threshold sweep — hybrid model with Platt calibration.
 * Train once, sweep 3–12% EV thresholds on 2025 OOS data.
 */
const EV_SWEEP_DATA = [
  { threshold: "3%", roi: 74.6, bets: 1368, hitRate: 35.6, sharpe: 0.14, wins: 487 },
  { threshold: "5%", roi: 68.5, bets: 1231, hitRate: 36.8, sharpe: 0.15, wins: 453 },
  { threshold: "8%", roi: 68.3, bets: 1097, hitRate: 38.6, sharpe: 0.16, wins: 423 },
  { threshold: "10%", roi: 64.3, bets: 1029, hitRate: 39.0, sharpe: 0.16, wins: 401 },
  { threshold: "12%", roi: 67.9, bets: 972, hitRate: 39.2, sharpe: 0.17, wins: 381 },
];

function formatPct(n: number) {
  return `${(n * 100).toFixed(1)}%`;
}

/** Tooltip-style label: shows a dotted underline hint, hover reveals explanation */
function Tip({ label, tip }: { label: string; tip: string }) {
  return (
    <span title={tip} style={{
      borderBottom: "1px dotted rgba(255,255,255,0.3)",
      cursor: "help",
    }}>{label}</span>
  );
}

export default function DashboardPage() {
  const [bankroll, setBankroll] = useState<BankrollSummary | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    api.getBankroll().then(setBankroll).catch(() => { }).finally(() => setLoading(false));
  }, []);

  const today = new Date();
  const dateStr = today.toLocaleDateString("en-US", { weekday: "long", year: "numeric", month: "long", day: "numeric" });

  return (
    <div className="page-container animate-in">
      {/* Header */}
      <div className="page-header" style={{ display: "flex", alignItems: "center", gap: "1rem" }}>
        <div>
          <h1 className="page-title">Dashboard</h1>
          <p className="page-subtitle">{dateStr}</p>
        </div>
        <span className="badge badge-green" style={{ marginLeft: "auto" }}>
          📊 2025 Hybrid + New Features (AUC 0.862)
        </span>
      </div>

      {/* Main Grid */}
      <div className="grid-6040" style={{ marginBottom: "1.5rem" }}>
        {/* Recent Bets Sample */}
        <div className="card">
          <div className="card-header">
            <h2 className="card-title">Recent Bets (2025-06-03 Sample)</h2>
            <span className="badge badge-gold">
              {RECENT_BETS.filter(b => b.result === "WIN").length} wins / {RECENT_BETS.length} shown
            </span>
          </div>
          <div style={{ overflowX: "auto" }}>
            <table className="data-table">
              <thead>
                <tr>
                  <th>Date</th>
                  <th>Horse</th>
                  <th><Tip label="Model P(win)" tip="Our model's estimated win probability" /></th>
                  <th><Tip label="Odds" tip="Payout multiplier" /></th>
                  <th><Tip label="Edge" tip="Expected Value — how much our estimate exceeds the market's" /></th>
                  <th>Result</th>
                </tr>
              </thead>
              <tbody>
                {RECENT_BETS.map((bet, i) => (
                  <tr key={i}>
                    <td style={{ fontSize: "0.8rem", color: "var(--text-muted)" }}>
                      {bet.date}
                    </td>
                    <td style={{ fontWeight: 600 }}>{bet.horse}</td>
                    <td className="ev-positive">{formatPct(bet.model_win)}</td>
                    <td>{bet.odds.toFixed(1)}x</td>
                    <td className="ev-positive" style={{ fontWeight: 600 }}>
                      +{formatPct(bet.ev)}
                    </td>
                    <td>
                      <span className={`badge ${bet.result === "WIN" ? "badge-green" : "badge-red"}`}>
                        {bet.result === "WIN" ? "✅ WIN" : "❌ LOSS"}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div style={{ marginTop: "0.75rem", fontSize: "0.85rem", color: "var(--text-muted)" }}>
            Showing last 10 bets from 12% EV threshold (972 total bets across Jan–Jun 2025).
          </div>
        </div>

        {/* Model & Backtest Summary */}
        <div className="card">
          <div className="card-header">
            <h2 className="card-title">Model Performance</h2>
          </div>

          {/* Model Metrics */}
          <div style={{ marginBottom: "1.5rem" }}>
            <h3 style={{ fontSize: "0.75rem", textTransform: "uppercase", color: "var(--text-muted)", letterSpacing: "0.05em", marginBottom: "0.75rem" }}>
              Hybrid Ensemble (Fundamental + Market)
            </h3>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: "0.75rem" }}>
              <div className="card" style={{ padding: "0.75rem", textAlign: "center" }}>
                <div style={{ fontSize: "1.25rem", fontWeight: 700, color: "var(--accent-emerald)" }}>
                  {(MODEL_METRICS.auc * 100).toFixed(1)}%
                </div>
                <div className="metric-label">
                  <Tip label="Ranking Accuracy" tip="AUC — How well the model ranks winners above losers. 50% = random, 100% = perfect. Ours: 84.8%" />
                </div>
              </div>
              <div className="card" style={{ padding: "0.75rem", textAlign: "center" }}>
                <div style={{ fontSize: "1.25rem", fontWeight: 700, color: "var(--accent-blue)" }}>
                  {MODEL_METRICS.logloss}
                </div>
                <div className="metric-label">
                  <Tip label="Confidence" tip="Log-Loss — Measures how confident the model is in correct predictions. Lower = better. Perfect = 0" />
                </div>
              </div>
              <div className="card" style={{ padding: "0.75rem", textAlign: "center" }}>
                <div style={{ fontSize: "1.25rem", fontWeight: 700, color: "var(--accent-gold)" }}>
                  {MODEL_METRICS.brier}
                </div>
                <div className="metric-label">
                  <Tip label="Calibration" tip="Brier Score — How well-calibrated the probabilities are. Lower = better. If model says 20%, it should win ~20% of the time" />
                </div>
              </div>
            </div>
          </div>

          {/* Backtest Summary */}
          <div>
            <h3 style={{ fontSize: "0.75rem", textTransform: "uppercase", color: "var(--text-muted)", letterSpacing: "0.05em", marginBottom: "0.75rem" }}>
              <Tip label="2025 Backtest (5% EV Threshold)" tip="Out-of-sample backtest on 1,224 races the model never saw during training. Uses Kelly-fraction sizing with Platt calibration." />
            </h3>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "0.75rem" }}>
              <div className="card" style={{ padding: "0.75rem", textAlign: "center" }}>
                <div className="metric-value positive" style={{ fontSize: "1.1rem" }}>
                  {BACKTEST_SUMMARY.totalBets}
                </div>
                <div className="metric-label">Total Bets</div>
              </div>
              <div className="card" style={{ padding: "0.75rem", textAlign: "center" }}>
                <div className="metric-value positive" style={{ fontSize: "1.1rem" }}>
                  {BACKTEST_SUMMARY.winRate}%
                </div>
                <div className="metric-label">Win Rate</div>
              </div>
              <div className="card" style={{ padding: "0.75rem", textAlign: "center" }}>
                <div className="metric-value positive" style={{ fontSize: "1.1rem", color: "var(--accent-emerald)" }}>
                  +{BACKTEST_SUMMARY.roi.toFixed(1)}%
                </div>
                <div className="metric-label"><Tip label="Return" tip="ROI (Return on Investment) — Profit as % of total staked. Uses Kelly-fraction compounding sizing." /></div>
              </div>
              <div className="card" style={{ padding: "0.75rem", textAlign: "center" }}>
                <div className="metric-value" style={{ fontSize: "1.1rem", color: "var(--accent-gold)" }}>
                  {BACKTEST_SUMMARY.sharpe.toFixed(2)}
                </div>
                <div className="metric-label"><Tip label="Sharpe Ratio" tip="Risk-adjusted return. Higher = better risk/reward. Measures consistency of returns." /></div>
              </div>
            </div>
          </div>

          <div style={{ marginTop: "0.75rem", padding: "0.5rem 0.75rem", background: "rgba(255,200,0,0.08)", borderRadius: "6px", fontSize: "0.8rem", color: "var(--text-muted)" }}>
            💰 Flat ¥100/bet sizing. Starting bankroll ¥1,000. Profit: ¥{(BACKTEST_SUMMARY.roi * BACKTEST_SUMMARY.totalBets / 100).toFixed(0).replace(/\B(?=(\d{3})+(?!\d))/g, ',')} from ¥{BACKTEST_SUMMARY.totalBets.toLocaleString()}00 staked.
          </div>

          <div style={{ marginTop: "1rem" }}>
            <Link href="/backtest" className="btn" style={{ width: "100%", justifyContent: "center" }}>
              View Full Backtest →
            </Link>
          </div>
        </div>
      </div>

      {/* EV Threshold Sweep */}
      <div className="card">
        <div className="card-header">
          <h2 className="card-title">🔬 EV Threshold Sweep</h2>
          <span className="badge badge-blue">Platt Calibration</span>
        </div>
        <table className="data-table">
          <thead>
            <tr>
              <th>EV Threshold</th>
              <th>Bets</th>
              <th>Wins</th>
              <th>Hit Rate</th>
              <th>ROI</th>
              <th>Sharpe</th>
            </tr>
          </thead>
          <tbody>
            {EV_SWEEP_DATA.map((row, i) => (
              <tr key={i} className={row.threshold === "12%" ? "highlight-row" : ""}>
                <td style={{ fontWeight: 600 }}>{row.threshold === "12%" ? "🏆 " : ""}{row.threshold}</td>
                <td>{row.bets.toLocaleString()}</td>
                <td>{row.wins}</td>
                <td>{row.hitRate}%</td>
                <td style={{ fontWeight: 700, color: "var(--accent-emerald)" }}>
                  +{row.roi.toFixed(1)}%
                </td>
                <td style={{ color: "var(--accent-gold)" }}>
                  {row.sharpe.toFixed(2)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
        <div style={{ marginTop: "0.75rem", fontSize: "0.85rem", color: "var(--text-muted)", lineHeight: 1.6 }}>
          💡 <strong>3% EV threshold has highest ROI</strong> — +74.6% ROI with 1,368 bets.
          12% threshold has best Sharpe (0.17) and hit rate (39.2%) with 972 bets.
        </div>
      </div>

      {/* Model Insight */}
      <div className="card">
        <div className="card-header">
          <h2 className="card-title">🤖 Model Insights</h2>
        </div>
        <p style={{ color: "var(--text-secondary)", lineHeight: 1.8 }}>
          The hybrid model evaluated <strong style={{ color: "var(--text-primary)" }}>17,364 entries across 1,224 races</strong> in 2025 and
          found <strong style={{ color: "var(--text-primary)" }}>{BACKTEST_SUMMARY.totalBets} value bets</strong> at
          ≥5% EV (multiple horses per race can qualify).
          Of those, <strong style={{ color: "var(--accent-emerald)" }}>{BACKTEST_SUMMARY.wins} won</strong> ({BACKTEST_SUMMARY.winRate}% hit rate)
          with <strong style={{ color: "var(--accent-emerald)" }}>+{BACKTEST_SUMMARY.roi}% ROI</strong> (Kelly-compounded).
        </p>
        <p style={{ color: "var(--text-muted)", lineHeight: 1.8, marginTop: "0.5rem", fontSize: "0.9rem" }}>
          Both fundamental and market models achieve AUC 0.862. 5 new features added:
          cross-sectional odds (rank, ratio, deviation) + trainer pedigree fallback.
          Top features: pace simulation, class rank, odds rank, trainer form, and course×jockey interaction.
        </p>
        <div style={{ marginTop: "1rem" }}>
          <Link href="/insights" style={{ color: "var(--accent-blue)", fontWeight: 600, fontSize: "0.9rem" }}>
            View Model Deep Dive →
          </Link>
        </div>
      </div>
    </div>
  );
}
