"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { api, type BankrollSummary } from "@/lib/api";

/**
 * Real backtest value bets — Top EV plays from 2024 simulation.
 * Source: data/backtest_results.csv (sorted by EV descending)
 */
const REAL_VALUE_BETS = [
  { date: "2024-01-02", horse: "Mister DJ", model_win: 0.573, market_win: 0.244, ev: 0.329, odds: 4.1, result: "WIN", race_id: 106 },
  { date: "2024-01-02", horse: "Peptide Shuchiku", model_win: 0.492, market_win: 0.208, ev: 0.283, odds: 4.8, result: "WIN", race_id: 170 },
  { date: "2024-01-01", horse: "Terrific One", model_win: 0.504, market_win: 0.250, ev: 0.254, odds: 4.0, result: "WIN", race_id: 69 },
  { date: "2024-01-02", horse: "Jommed Vin", model_win: 0.540, market_win: 0.294, ev: 0.246, odds: 3.4, result: "WIN", race_id: 115 },
  { date: "2024-01-01", horse: "Aruma Veloce", model_win: 0.495, market_win: 0.256, ev: 0.239, odds: 3.9, result: "WIN", race_id: 96 },
  { date: "2024-01-02", horse: "Mataro Sun", model_win: 0.510, market_win: 0.278, ev: 0.233, odds: 3.6, result: "WIN", race_id: 120 },
  { date: "2024-01-02", horse: "Multi License", model_win: 0.363, market_win: 0.152, ev: 0.212, odds: 6.6, result: "WIN", race_id: 117 },
  { date: "2024-01-01", horse: "Reflect the Moon", model_win: 0.359, market_win: 0.147, ev: 0.212, odds: 6.8, result: "WIN", race_id: 39 },
  { date: "2024-01-01", horse: "Sol Amazon", model_win: 0.123, market_win: 0.023, ev: 0.100, odds: 43.5, result: "WIN", race_id: 95 },
  { date: "2024-01-02", horse: "Roman Lake", model_win: 0.165, market_win: 0.059, ev: 0.106, odds: 16.9, result: "WIN", race_id: 167 },
];

/**
 * Real model performance metrics from the trained ensemble.
 * Source: models/saved/20260225_173348/metadata.json
 */
const MODEL_METRICS = {
  logloss: 0.199,
  auc: 0.833,
  brier: 0.056,
};

/**
 * Real backtest summary (flat ¥1,000 stakes).
 * Computed from data/backtest_results.csv — 102 bets with valid odds.
 */
const BACKTEST_SUMMARY = {
  totalBets: 102,
  wins: 82,
  winRate: 80.4,
  flatStaked: 102000,
  flatProfit: 583200,
  roi: 571.8,
  avgEv: 0.121,
  avgOdds: 8.4,
};

function formatPct(n: number) {
  return `${(n * 100).toFixed(1)}%`;
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
          📊 Simulation Results
        </span>
      </div>

      {/* Main Grid */}
      <div className="grid-6040" style={{ marginBottom: "1.5rem" }}>
        {/* Value Bets Table */}
        <div className="card">
          <div className="card-header">
            <h2 className="card-title">Top Value Bets Found (2024 Backtest)</h2>
            <span className="badge badge-green">{REAL_VALUE_BETS.length} highlights</span>
          </div>
          <table className="data-table">
            <thead>
              <tr>
                <th>Date</th>
                <th>Horse</th>
                <th>Model</th>
                <th>Market</th>
                <th>EV</th>
                <th>Odds</th>
                <th>Result</th>
              </tr>
            </thead>
            <tbody>
              {REAL_VALUE_BETS.map((bet, i) => (
                <tr key={i} className={bet.ev > 0.2 ? "highlight-row" : ""}>
                  <td style={{ fontSize: "0.8rem", color: "var(--text-muted)" }}>
                    {bet.date}
                  </td>
                  <td style={{ fontWeight: 600 }}>{bet.horse}</td>
                  <td className="ev-positive">{formatPct(bet.model_win)}</td>
                  <td>{formatPct(bet.market_win)}</td>
                  <td className="ev-positive" style={{ fontWeight: 600 }}>
                    +{formatPct(bet.ev)}
                  </td>
                  <td>{bet.odds.toFixed(1)}x</td>
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

        {/* Model & Backtest Summary */}
        <div className="card">
          <div className="card-header">
            <h2 className="card-title">Model Performance</h2>
          </div>

          {/* Model Metrics */}
          <div style={{ marginBottom: "1.5rem" }}>
            <h3 style={{ fontSize: "0.75rem", textTransform: "uppercase", color: "var(--text-muted)", letterSpacing: "0.05em", marginBottom: "0.75rem" }}>
              LightGBM + XGBoost Ensemble
            </h3>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: "0.75rem" }}>
              <div className="card" style={{ padding: "0.75rem", textAlign: "center" }}>
                <div style={{ fontSize: "1.25rem", fontWeight: 700, color: "var(--accent-emerald)" }}>
                  {MODEL_METRICS.auc}
                </div>
                <div className="metric-label">AUC</div>
              </div>
              <div className="card" style={{ padding: "0.75rem", textAlign: "center" }}>
                <div style={{ fontSize: "1.25rem", fontWeight: 700, color: "var(--accent-blue)" }}>
                  {MODEL_METRICS.logloss}
                </div>
                <div className="metric-label">Log-Loss</div>
              </div>
              <div className="card" style={{ padding: "0.75rem", textAlign: "center" }}>
                <div style={{ fontSize: "1.25rem", fontWeight: 700, color: "var(--accent-gold)" }}>
                  {MODEL_METRICS.brier}
                </div>
                <div className="metric-label">Brier Score</div>
              </div>
            </div>
          </div>

          {/* Backtest Summary */}
          <div>
            <h3 style={{ fontSize: "0.75rem", textTransform: "uppercase", color: "var(--text-muted)", letterSpacing: "0.05em", marginBottom: "0.75rem" }}>
              Backtest (Flat ¥1,000 Stake)
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
                <div className="metric-value positive" style={{ fontSize: "1.1rem" }}>
                  +{BACKTEST_SUMMARY.roi.toFixed(1)}%
                </div>
                <div className="metric-label">ROI</div>
              </div>
              <div className="card" style={{ padding: "0.75rem", textAlign: "center" }}>
                <div className="metric-value" style={{ fontSize: "1.1rem", color: "var(--accent-gold)" }}>
                  {BACKTEST_SUMMARY.avgOdds.toFixed(1)}x
                </div>
                <div className="metric-label">Avg Odds</div>
              </div>
            </div>
          </div>

          <div style={{ marginTop: "1rem" }}>
            <Link href="/backtest" className="btn" style={{ width: "100%", justifyContent: "center" }}>
              View Full Backtest →
            </Link>
          </div>
        </div>
      </div>

      {/* Model Insight */}
      <div className="card">
        <div className="card-header">
          <h2 className="card-title">🤖 Model Insights</h2>
        </div>
        <p style={{ color: "var(--text-secondary)", lineHeight: 1.8 }}>
          The model identified <strong style={{ color: "var(--accent-emerald)" }}>{BACKTEST_SUMMARY.wins} winning bets</strong> out of {BACKTEST_SUMMARY.totalBets} total
          in the 2024 backtest period. The highest-EV play was <strong style={{ color: "var(--accent-emerald)" }}>Mister DJ</strong> (Race 106)
          with a model probability of 57.3% vs the market&apos;s 24.4%, representing a +32.9% edge.
          Notable long-shot winners include <strong style={{ color: "var(--accent-emerald)" }}>Sol Amazon</strong> at 43.5x odds
          and <strong style={{ color: "var(--accent-emerald)" }}>Roman Lake</strong> at 16.9x.
          The ensemble (LightGBM + XGBoost) achieved an AUC of {MODEL_METRICS.auc}, indicating strong discriminative power.
        </p>
        <div style={{ marginTop: "1rem" }}>
          <Link href="/backtest" style={{ color: "var(--accent-blue)", fontWeight: 600, fontSize: "0.9rem" }}>
            View Full Analysis →
          </Link>
        </div>
      </div>
    </div>
  );
}
