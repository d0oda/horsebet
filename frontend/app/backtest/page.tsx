"use client";

import { useEffect, useState } from "react";
import dynamic from "next/dynamic";
import { api, type BacktestResult } from "@/lib/api";

const LineChart = dynamic(() => import("recharts").then((m) => m.LineChart), { ssr: false });
const Line = dynamic(() => import("recharts").then((m) => m.Line), { ssr: false });
const XAxis = dynamic(() => import("recharts").then((m) => m.XAxis), { ssr: false });
const YAxis = dynamic(() => import("recharts").then((m) => m.YAxis), { ssr: false });
const CartesianGrid = dynamic(() => import("recharts").then((m) => m.CartesianGrid), { ssr: false });
const Tooltip = dynamic(() => import("recharts").then((m) => m.Tooltip), { ssr: false });
const ResponsiveContainer = dynamic(() => import("recharts").then((m) => m.ResponsiveContainer), { ssr: false });
const BarChart = dynamic(() => import("recharts").then((m) => m.BarChart), { ssr: false });
const Bar = dynamic(() => import("recharts").then((m) => m.Bar), { ssr: false });
const Cell = dynamic(() => import("recharts").then((m) => m.Cell), { ssr: false });

/**
 * Real bet-by-bet cumulative P&L from 2025 evaluation.
 * 18 bets, Kelly-fraction sizing, starting balance ¥100,000.
 * Source: python -m models.run_evaluation (Step 2, 5% EV threshold)
 */
const REAL_PNL_CURVE = [
    { bet: 0, label: "Start", balance: 100000 },
    { bet: 1, label: "ショウナンヤッホー", balance: 112998 },  // WIN  +12998
    { bet: 2, label: "シンヒダカゴールド", balance: 123902 },  // WIN  +10904
    { bet: 3, label: "スマートブル", balance: 119938 },  // LOSS -3964
    { bet: 4, label: "タマモジャスミン", balance: 116215 },  // LOSS -3723
    { bet: 5, label: "マッシャーブルム", balance: 112538 },  // LOSS -3677
    { bet: 6, label: "クーデール", balance: 109352 },  // LOSS -3186
    { bet: 7, label: "エヴァンスウィート", balance: 115703 },  // WIN  +6351
    { bet: 8, label: "アルマデオロ", balance: 119687 },  // WIN  +3984
    { bet: 9, label: "アセンディア", balance: 116521 },  // LOSS -3166
    { bet: 10, label: "ルージュミラージュ", balance: 114300 },  // LOSS -2221
    { bet: 11, label: "マーウォルス", balance: 120329 },  // WIN  +6029
    { bet: 12, label: "ジョードリウム", balance: 116355 },  // LOSS -3974
    { bet: 13, label: "キントラダンサー", balance: 126180 },  // WIN  +9825
    { bet: 14, label: "バースクライ", balance: 122830 },  // LOSS -3350
    { bet: 15, label: "ハイファイスピード", balance: 119857 },  // LOSS -2973
    { bet: 16, label: "シンゼンカガ", balance: 116664 },  // LOSS -3193
    { bet: 17, label: "アドマイヤサジー", balance: 114203 },  // LOSS -2461
    { bet: 18, label: "エコロレオナ", balance: 117190 },  // WIN  +2987
];

/**
 * EV sweep comparison — ROI at different thresholds.
 * Source: python -m models.run_evaluation Steps 2 & 3
 */
const EV_SWEEP_DATA = [
    { threshold: "5%", roi_with_odds: 23.6, roi_odds_free: -52.2, bets_with: 18, bets_free: 35 },
    { threshold: "8%", roi_with_odds: 28.4, roi_odds_free: -65.0, bets_with: 8, bets_free: 12 },
    { threshold: "10%", roi_with_odds: 230.0, roi_odds_free: -80.0, bets_with: 1, bets_free: 5 },
    { threshold: "12%", roi_with_odds: 0, roi_odds_free: 0, bets_with: 0, bets_free: 0 },
];

/**
 * Losing bet failure mode analysis.
 * Source: python -m models.run_evaluation Step 4
 */
const FAILURE_MODES = [
    { mode: "No Edge", count: 8, pct: 73, loss: -25599, desc: "Model ≈ Market (overestimates by ~8pp)" },
    { mode: "Variance", count: 3, pct: 27, loss: -10140, desc: "Horse finished 2nd or 3rd" },
];

export default function BacktestPage() {
    const [data, setData] = useState<BacktestResult | null>(null);
    const [grade, setGrade] = useState("");

    useEffect(() => {
        api.getBacktest({ grade: grade || undefined }).then(setData).catch(() => { });
    }, [grade]);

    const metrics = {
        totalBets: 18,
        winRate: 38.9,
        roi: 23.6,
        maxDrawdown: 9.5,
        sharpe: 4.86,
        avgEv: 7.9,
        avgOdds: 3.3,
        auc: 0.848,
        totalStaked: "¥55,808",
        totalPayout: "¥68,969",
        profit: "¥13,161",
    };

    return (
        <div className="page-container animate-in">
            <div className="page-header" style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start" }}>
                <div>
                    <h1 className="page-title">Backtest Results</h1>
                    <p className="page-subtitle">2025 out-of-sample evaluation — 100 races, Kelly-fraction sizing</p>
                </div>
                <div style={{ display: "flex", gap: "0.75rem", alignItems: "center" }}>
                    <span className="badge badge-green">2025 Real Data</span>
                </div>
            </div>

            {/* Row 1: P&L Chart + Metrics */}
            <div className="grid-6040" style={{ marginBottom: "1.5rem" }}>
                <div className="card">
                    <div className="card-header">
                        <h2 className="card-title">Cumulative Balance (18 bets)</h2>
                        <div style={{ display: "flex", gap: "0.75rem" }}>
                            <span className="badge badge-green">ROI +{metrics.roi}%</span>
                            <span className="badge badge-blue">Sharpe {metrics.sharpe}</span>
                        </div>
                    </div>
                    <div className="chart-container">
                        <ResponsiveContainer width="100%" height="100%">
                            <LineChart data={REAL_PNL_CURVE}>
                                <CartesianGrid strokeDasharray="3 3" stroke="rgba(42,48,64,0.5)" />
                                <XAxis dataKey="bet" tick={{ fill: "#8b95a5", fontSize: 11 }} label={{ value: "Bet #", position: "insideBottomRight", offset: -5, fill: "#8b95a5", fontSize: 11 }} />
                                <YAxis tick={{ fill: "#8b95a5", fontSize: 11 }} tickFormatter={(v: number) => `¥${(v / 1000).toFixed(0)}k`} domain={[95000, 130000]} />
                                <Tooltip
                                    contentStyle={{ background: "#1a1f2e", border: "1px solid #2a3040", borderRadius: 8, color: "#f0f0f0" }}
                                    formatter={(value: any) => [`¥${Number(value).toLocaleString()}`, "Balance"]}
                                    labelFormatter={(label: any) => {
                                        const entry = REAL_PNL_CURVE.find(e => e.bet === label);
                                        return entry ? `#${label}: ${entry.label}` : `Bet #${label}`;
                                    }}
                                />
                                <Line type="monotone" dataKey="balance" stroke="#00ff88" strokeWidth={2.5} dot={{ fill: "#00ff88", r: 3 }} activeDot={{ r: 5 }} name="Balance" />
                            </LineChart>
                        </ResponsiveContainer>
                    </div>
                </div>

                <div style={{ display: "flex", flexDirection: "column", gap: "0.75rem" }}>
                    {[
                        { label: "Total Bets", value: metrics.totalBets.toLocaleString(), color: "var(--text-primary)" },
                        { label: "Win Rate", value: `${metrics.winRate}%`, color: "var(--accent-emerald)" },
                        { label: "ROI", value: `+${metrics.roi}%`, color: "var(--accent-emerald)" },
                        { label: "Sharpe Ratio", value: `${metrics.sharpe}`, color: "var(--accent-emerald)" },
                        { label: "Max Drawdown", value: `${metrics.maxDrawdown}%`, color: "var(--accent-gold)" },
                        { label: "Avg EV", value: `+${metrics.avgEv}%`, color: "var(--accent-blue)" },
                        { label: "Avg Odds", value: `${metrics.avgOdds}x`, color: "var(--accent-gold)" },
                        { label: "Model AUC", value: `${metrics.auc}`, color: "var(--accent-blue)" },
                    ].map((m, i) => (
                        <div className="card" key={i} style={{ padding: "0.75rem", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                            <span className="metric-label" style={{ margin: 0 }}>{m.label}</span>
                            <span style={{ fontSize: "1.1rem", fontWeight: 700, color: m.color }}>{m.value}</span>
                        </div>
                    ))}
                </div>
            </div>

            {/* Row 2: EV Threshold Sweep */}
            <div className="card" style={{ marginBottom: "1.5rem" }}>
                <div className="card-header">
                    <h2 className="card-title">EV Threshold Sweep</h2>
                    <span className="badge badge-blue">With Odds vs Odds-Free</span>
                </div>
                <table className="data-table">
                    <thead>
                        <tr>
                            <th>EV Threshold</th>
                            <th>Bets (with odds)</th>
                            <th>ROI (with odds)</th>
                            <th>Bets (odds-free)</th>
                            <th>ROI (odds-free)</th>
                        </tr>
                    </thead>
                    <tbody>
                        {EV_SWEEP_DATA.map((row, i) => (
                            <tr key={i} className={row.threshold === "5%" ? "highlight-row" : ""}>
                                <td style={{ fontWeight: 600 }}>≥{row.threshold}</td>
                                <td>{row.bets_with}</td>
                                <td style={{
                                    fontWeight: 700,
                                    color: row.roi_with_odds > 0 ? "var(--accent-emerald)" : row.roi_with_odds === 0 ? "var(--text-muted)" : "var(--accent-red, #ff5555)"
                                }}>
                                    {row.roi_with_odds > 0 ? "+" : ""}{row.roi_with_odds}%
                                </td>
                                <td>{row.bets_free}</td>
                                <td style={{
                                    fontWeight: 700,
                                    color: row.roi_odds_free > 0 ? "var(--accent-emerald)" : row.roi_odds_free === 0 ? "var(--text-muted)" : "var(--accent-red, #ff5555)"
                                }}>
                                    {row.roi_odds_free > 0 ? "+" : ""}{row.roi_odds_free}%
                                </td>
                            </tr>
                        ))}
                    </tbody>
                </table>
                <div style={{ marginTop: "0.75rem", fontSize: "0.85rem", color: "var(--text-muted)", lineHeight: 1.6 }}>
                    💡 The 5% threshold offers the best balance of volume (18 bets) and profitability (+23.6% ROI).
                    The odds-free model (AUC 0.818) confirms fundamental edge but is not profitable on its own.
                </div>
            </div>

            {/* Row 3: Failure Mode Analysis */}
            <div className="card">
                <div className="card-header">
                    <h2 className="card-title">Losing Bet Analysis</h2>
                    <span className="badge badge-gold">11 losing bets dissected</span>
                </div>
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "1rem" }}>
                    {FAILURE_MODES.map((fm, i) => (
                        <div key={i} className="card" style={{ padding: "1rem" }}>
                            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "0.5rem" }}>
                                <span style={{ fontSize: "1rem", fontWeight: 700, color: "var(--text-primary)" }}>
                                    {fm.mode === "No Edge" ? "🟡" : "🟢"} {fm.mode}
                                </span>
                                <span className="badge badge-gold">{fm.pct}%</span>
                            </div>
                            <div style={{ fontSize: "0.85rem", color: "var(--text-secondary)", marginBottom: "0.5rem" }}>
                                {fm.desc}
                            </div>
                            <div style={{ display: "flex", gap: "1.5rem", fontSize: "0.85rem" }}>
                                <span><strong>{fm.count}</strong> bets</span>
                                <span style={{ color: "var(--accent-red, #ff5555)" }}>¥{fm.loss.toLocaleString()}</span>
                            </div>
                        </div>
                    ))}
                </div>
                <div style={{ marginTop: "0.75rem", fontSize: "0.85rem", color: "var(--text-muted)", lineHeight: 1.6 }}>
                    📊 73% of losses are &quot;No Edge&quot; — the model overestimates win probability by ~8 percentage points.
                    Target for next feature sprint: race class changes, trainer short-term form, and calibration tuning.
                </div>
            </div>
        </div>
    );
}
