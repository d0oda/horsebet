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
 * Real bet-by-bet cumulative P&L from 2025 hybrid evaluation (Platt scaling).
 * 41 bets, Kelly-fraction sizing, starting balance ¥100,000.
 * Source: data/all_bets_platt.json
 */
const REAL_PNL_CURVE = [
    { bet: 0, label: "Start", balance: 100000 },
    { bet: 1, label: "メイショウコシュウ", balance: 95000 },
    { bet: 2, label: "ハイファイスピード", balance: 90250 },
    { bet: 3, label: "ウインマクシマム", balance: 86932 },
    { bet: 4, label: "バロンドール", balance: 82586 },
    { bet: 5, label: "ダークメモリー", balance: 78457 },
    { bet: 6, label: "カフェブーケット", balance: 83163 },
    { bet: 7, label: "ピティロディア", balance: 79005 },
    { bet: 8, label: "ラルフテソーロ", balance: 85720 },
    { bet: 9, label: "リアライズカミオン", balance: 88291 },
    { bet: 10, label: "パーフェクトパール", balance: 92122 },
    { bet: 11, label: "ザハント", balance: 101334 },
    { bet: 12, label: "ストロベリーツリー", balance: 97527 },
    { bet: 13, label: "マンゲタック", balance: 94597 },
    { bet: 14, label: "スマートブル", balance: 89868 },
    { bet: 15, label: "フェアリーライク", balance: 94688 },
    { bet: 16, label: "スムースベルベット", balance: 93458 },
    { bet: 17, label: "アルマデオロ", balance: 102334 },
    { bet: 18, label: "テーオーシュターデ", balance: 99989 },
    { bet: 19, label: "ヒシアムルーズ", balance: 105987 },
    { bet: 20, label: "アリスメティーク", balance: 118704 },
    { bet: 21, label: "メイショウキンタイ", balance: 112769 },
    { bet: 22, label: "サクラファレル", balance: 114460 },
    { bet: 23, label: "タイセイアビリティ", balance: 108737 },
    { bet: 24, label: "プルパレイ", balance: 107256 },
    { bet: 25, label: "イムホテプ", balance: 110473 },
    { bet: 26, label: "ハイエンドモデル", balance: 104950 },
    { bet: 27, label: "グリオンヴール", balance: 105999 },
    { bet: 28, label: "メリディアンスター", balance: 102582 },
    { bet: 29, label: "ベイラム", balance: 97453 },
    { bet: 30, label: "アイスグリーン", balance: 95655 },
    { bet: 31, label: "アーロンイメル", balance: 102828 },
    { bet: 32, label: "リーゼノアール", balance: 97687 },
    { bet: 33, label: "リフレックス", balance: 92803 },
    { bet: 34, label: "レッドスティンガー", balance: 97604 },
    { bet: 35, label: "ヤマニンヒストリア", balance: 92724 },
    { bet: 36, label: "リスレジャンデール", balance: 88088 },
    { bet: 37, label: "ディニトーソ", balance: 83684 },
    { bet: 38, label: "シンゼンカガ", balance: 79500 },
    { bet: 39, label: "ピンクジン", balance: 114082 },
    { bet: 40, label: "コマチチャン", balance: 108378 },
    { bet: 41, label: "ノチェセラーダ", balance: 113254 },
];

/**
 * Calibration comparison — ROI by calibration method, all at 5% EV threshold.
 * Source: python -m models.test_2025 --hybrid --calibration {none,platt,isotonic}
 */
const CALIBRATION_DATA = [
    { method: "Platt", roi: 7.6, bets: 41, hitRate: 39.0, sharpe: 12.43, avgOdds: 12.0, profit: 13254 },
    { method: "Isotonic", roi: 2.1, bets: 49, hitRate: 32.7, sharpe: 1.40, avgOdds: 11.9, profit: 3564 },
    { method: "None", roi: -25.2, bets: 26, hitRate: 7.7, sharpe: -1.89, avgOdds: 52.9, profit: -13061 },
];

/**
 * Key insights from the calibration comparison.
 */
const CALIBRATION_INSIGHTS = [
    { insight: "Platt", desc: "Best overall: +7.6% ROI, 39% hit rate, highest Sharpe (12.43). Fits 2 parameters — robust with small validation sets." },
    { insight: "Isotonic", desc: "Also profitable (+2.1%) but more bets (49) at lower conviction. Overfits slightly with small bins (n=1–7 in upper buckets)." },
    { insight: "None", desc: "Disastrous: bets on extreme longshots (52.9x avg odds), only 7.7% hit rate. Uncalibrated probabilities are overconfident." },
];

export default function BacktestPage() {
    const [data, setData] = useState<BacktestResult | null>(null);
    const [grade, setGrade] = useState("");

    useEffect(() => {
        api.getBacktest({ grade: grade || undefined }).then(setData).catch(() => { });
    }, [grade]);

    const metrics = {
        totalBets: 41,
        winRate: 39.0,
        roi: 7.6,
        maxDrawdown: 33.0,
        sharpe: 12.43,
        avgEv: 21.1,
        avgOdds: 12.0,
        auc: 0.821,
        totalStaked: "¥175,474",
        totalPayout: "¥188,728",
        profit: "¥13,254",
    };

    return (
        <div className="page-container animate-in">
            <div className="page-header" style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start" }}>
                <div>
                    <h1 className="page-title">Backtest Results</h1>
                    <p className="page-subtitle">2025 OOS — Hybrid Ensemble with Platt Scaling, 193 features, +7.6% ROI</p>
                </div>
                <div style={{ display: "flex", gap: "0.75rem", alignItems: "center" }}>
                    <span className="badge badge-green">2025 Hybrid</span>
                    <span className="badge badge-blue">193 Features</span>
                </div>
            </div>

            {/* Row 1: P&L Chart + Metrics */}
            <div className="grid-6040" style={{ marginBottom: "1.5rem" }}>
                <div className="card">
                    <div className="card-header">
                        <h2 className="card-title">Cumulative Balance (41 bets)</h2>
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
                                <YAxis tick={{ fill: "#8b95a5", fontSize: 11 }} tickFormatter={(v: number) => `¥${(v / 1000).toFixed(0)}k`} domain={[75000, 125000]} />
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

            {/* Row 2: Calibration Comparison */}
            <div className="card" style={{ marginBottom: "1.5rem" }}>
                <div className="card-header">
                    <h2 className="card-title">🔬 Calibration Method Comparison</h2>
                    <span className="badge badge-blue">5% EV Threshold</span>
                </div>
                <table className="data-table">
                    <thead>
                        <tr>
                            <th>Method</th>
                            <th>Bets</th>
                            <th>Hit Rate</th>
                            <th>ROI</th>
                            <th>Sharpe</th>
                            <th>Avg Odds</th>
                            <th>Profit</th>
                        </tr>
                    </thead>
                    <tbody>
                        {CALIBRATION_DATA.map((row, i) => (
                            <tr key={i} className={row.method === "Platt" ? "highlight-row" : ""}>
                                <td style={{ fontWeight: 600 }}>{row.method === "Platt" ? "✅ " : ""}{row.method}</td>
                                <td>{row.bets}</td>
                                <td>{row.hitRate}%</td>
                                <td style={{ fontWeight: 700, color: row.roi > 0 ? "var(--accent-emerald)" : "var(--accent-red, #ff5555)" }}>
                                    {row.roi > 0 ? "+" : ""}{row.roi}%
                                </td>
                                <td style={{ color: row.sharpe > 2 ? "var(--accent-emerald)" : row.sharpe > 0 ? "var(--accent-gold)" : "var(--accent-red, #ff5555)" }}>
                                    {row.sharpe.toFixed(2)}
                                </td>
                                <td>{row.avgOdds}x</td>
                                <td style={{ fontWeight: 600, color: row.profit > 0 ? "var(--accent-emerald)" : "var(--accent-red, #ff5555)" }}>
                                    {row.profit > 0 ? "+" : ""}¥{row.profit.toLocaleString()}
                                </td>
                            </tr>
                        ))}
                    </tbody>
                </table>
                <div style={{ marginTop: "0.75rem", fontSize: "0.85rem", color: "var(--text-muted)", lineHeight: 1.6 }}>
                    💡 <strong>Platt scaling is the clear winner</strong> — best risk-adjusted returns with a Sharpe ratio of 12.43.
                </div>
            </div>

            {/* Row 3: Key Insights */}
            <div className="card">
                <div className="card-header">
                    <h2 className="card-title">Calibration Insights</h2>
                    <span className="badge badge-gold">Why Platt wins</span>
                </div>
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: "1rem" }}>
                    {CALIBRATION_INSIGHTS.map((item, i) => (
                        <div key={i} className="card" style={{ padding: "1rem" }}>
                            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "0.5rem" }}>
                                <span style={{ fontSize: "1rem", fontWeight: 700, color: "var(--text-primary)" }}>
                                    {item.insight === "Platt" ? "🏆" : item.insight === "Isotonic" ? "🟡" : "🔴"} {item.insight}
                                </span>
                            </div>
                            <div style={{ fontSize: "0.85rem", color: "var(--text-secondary)" }}>
                                {item.desc}
                            </div>
                        </div>
                    ))}
                </div>
            </div>
        </div>
    );
}
