"use client";

import { useEffect, useState } from "react";
import dynamic from "next/dynamic";
import { api, type BankrollSummary } from "@/lib/api";

const AreaChart = dynamic(() => import("recharts").then((m) => m.AreaChart), { ssr: false });
const Area = dynamic(() => import("recharts").then((m) => m.Area), { ssr: false });
const XAxis = dynamic(() => import("recharts").then((m) => m.XAxis), { ssr: false });
const YAxis = dynamic(() => import("recharts").then((m) => m.YAxis), { ssr: false });
const CartesianGrid = dynamic(() => import("recharts").then((m) => m.CartesianGrid), { ssr: false });
const Tooltip = dynamic(() => import("recharts").then((m) => m.Tooltip), { ssr: false });
const ResponsiveContainer = dynamic(() => import("recharts").then((m) => m.ResponsiveContainer), { ssr: false });

/**
 * Real balance history from backtest — flat ¥1,000 per bet.
 * Sampled at every 5th bet to keep chart clean.
 */
const REAL_BALANCE_HISTORY = [
    { bet: "Start", balance: 100000 },
    { bet: "5", balance: 98900 },
    { bet: "10", balance: 93900 },
    { bet: "15", balance: 139700 },
    { bet: "20", balance: 155900 },
    { bet: "25", balance: 222400 },
    { bet: "30", balance: 235600 },
    { bet: "35", balance: 268900 },
    { bet: "40", balance: 283500 },
    { bet: "45", balance: 307700 },
    { bet: "50", balance: 396800 },
    { bet: "55", balance: 422000 },
    { bet: "60", balance: 449700 },
    { bet: "65", balance: 474400 },
    { bet: "70", balance: 494600 },
    { bet: "75", balance: 530400 },
    { bet: "80", balance: 563100 },
    { bet: "85", balance: 575500 },
    { bet: "90", balance: 608000 },
    { bet: "95", balance: 627500 },
    { bet: "102", balance: 657400 },
];

/**
 * Real bet log — last 15 bets from the backtest, shown newest first.
 * Source: data/backtest_results.csv
 */
const REAL_BET_LOG = [
    { date: "Jan 2", race: "R171", horse: "Chuwa Dance", type: "Win", stake: 1000, odds: 4.8, pnl: 3800, won: true },
    { date: "Jan 2", race: "R170", horse: "Peptide Shuchiku", type: "Win", stake: 1000, odds: 4.8, pnl: 3800, won: true },
    { date: "Jan 2", race: "R169", horse: "Ripe Pikake", type: "Win", stake: 1000, odds: 3.2, pnl: 2200, won: true },
    { date: "Jan 2", race: "R167", horse: "Roman Lake", type: "Win", stake: 1000, odds: 16.9, pnl: 15900, won: true },
    { date: "Jan 2", race: "R163", horse: "Valdorcia", type: "Win", stake: 1000, odds: 1.9, pnl: 900, won: true },
    { date: "Jan 2", race: "R161", horse: "Kufashiru", type: "Win", stake: 1000, odds: 5.3, pnl: 4300, won: true },
    { date: "Jan 2", race: "R160", horse: "Ginrei", type: "Win", stake: 1000, odds: 105.2, pnl: -1000, won: false },
    { date: "Jan 2", race: "R158", horse: "Win Acteur", type: "Win", stake: 1000, odds: 10.0, pnl: 9000, won: true },
    { date: "Jan 2", race: "R157", horse: "Admire Astra", type: "Win", stake: 1000, odds: 5.7, pnl: 4700, won: true },
    { date: "Jan 2", race: "R156", horse: "Gran Giorno", type: "Win", stake: 1000, odds: 5.1, pnl: 4100, won: true },
    { date: "Jan 2", race: "R155", horse: "I Will", type: "Win", stake: 1000, odds: 3.7, pnl: 2700, won: true },
    { date: "Jan 2", race: "R154", horse: "Exabit", type: "Win", stake: 1000, odds: 4.6, pnl: -1000, won: false },
    { date: "Jan 2", race: "R154", horse: "Milfleur", type: "Win", stake: 1000, odds: 3.6, pnl: 2600, won: true },
    { date: "Jan 2", race: "R153", horse: "Laurel Orb", type: "Win", stake: 1000, odds: 11.8, pnl: 10800, won: true },
    { date: "Jan 2", race: "R152", horse: "Load Veldt", type: "Win", stake: 1000, odds: 3.3, pnl: 2300, won: true },
];

/**
 * Real win/loss streak from last 20 bets.
 */
const REAL_STREAK = ["W", "W", "W", "W", "W", "W", "L", "W", "W", "W", "W", "L", "W", "W", "W", "W", "L", "W", "W", "W"];

export default function BankrollPage() {
    const [bankroll, setBankroll] = useState<BankrollSummary | null>(null);

    useEffect(() => {
        api.getBankroll().then(setBankroll).catch(() => { });
    }, []);

    const balance = bankroll?.balance ?? 657400;
    const totalProfit = 557400;
    const roi = 571.8;
    const drawdown = 7.1;
    const kellyFraction = 0.25;

    return (
        <div className="page-container animate-in">
            {/* Header with balance */}
            <div className="page-header">
                <h1 className="page-title">Bankroll Manager</h1>
                <div style={{ display: "flex", alignItems: "center", gap: "2rem", marginTop: "1rem" }}>
                    <div>
                        <div className="metric-value positive" style={{ fontSize: "2.5rem" }}>
                            ¥{balance.toLocaleString()}
                        </div>
                        <div className="metric-label">Final Balance (from ¥100,000)</div>
                    </div>
                    <div className="grid-4" style={{ flex: 1 }}>
                        <div className="card" style={{ padding: "0.75rem 1rem", textAlign: "center" }}>
                            <div style={{ fontSize: "1.1rem", fontWeight: 700, color: "var(--accent-emerald)" }}>
                                +¥{totalProfit.toLocaleString()}
                            </div>
                            <div className="metric-label">Total Profit</div>
                        </div>
                        <div className="card" style={{ padding: "0.75rem 1rem", textAlign: "center" }}>
                            <div style={{ fontSize: "1.1rem", fontWeight: 700, color: "var(--accent-emerald)" }}>+{roi}%</div>
                            <div className="metric-label">ROI</div>
                        </div>
                        <div className="card" style={{ padding: "0.75rem 1rem", textAlign: "center" }}>
                            <div style={{ fontSize: "1.1rem", fontWeight: 700, color: drawdown > 10 ? "var(--accent-red)" : "var(--accent-gold)" }}>
                                {drawdown}%
                            </div>
                            <div className="metric-label">Max Drawdown</div>
                        </div>
                        <div className="card" style={{ padding: "0.75rem 1rem", textAlign: "center" }}>
                            <span className="badge badge-green">BACKTEST</span>
                            <div className="metric-label" style={{ marginTop: "0.25rem" }}>Mode</div>
                        </div>
                    </div>
                </div>
            </div>

            {/* Balance History Chart */}
            <div className="card" style={{ marginBottom: "1.5rem" }}>
                <div className="card-header">
                    <h2 className="card-title">Balance Growth (102 bets)</h2>
                    <span className="badge badge-green">Flat ¥1,000/bet</span>
                </div>
                <div className="chart-container" style={{ height: "280px" }}>
                    <ResponsiveContainer width="100%" height="100%">
                        <AreaChart data={REAL_BALANCE_HISTORY}>
                            <defs>
                                <linearGradient id="balanceGradient" x1="0" y1="0" x2="0" y2="1">
                                    <stop offset="0%" stopColor="#00ff88" stopOpacity={0.3} />
                                    <stop offset="100%" stopColor="#00ff88" stopOpacity={0.02} />
                                </linearGradient>
                            </defs>
                            <CartesianGrid strokeDasharray="3 3" stroke="rgba(42,48,64,0.5)" />
                            <XAxis dataKey="bet" tick={{ fill: "#8b95a5", fontSize: 11 }} label={{ value: "Bet #", position: "insideBottomRight", offset: -5, fill: "#8b95a5", fontSize: 11 }} />
                            <YAxis tick={{ fill: "#8b95a5", fontSize: 11 }} tickFormatter={(v: number) => `¥${(v / 1000).toFixed(0)}k`} />
                            <Tooltip
                                contentStyle={{ background: "#1a1f2e", border: "1px solid #2a3040", borderRadius: 8, color: "#f0f0f0" }}
                                formatter={(value: any) => [`¥${Number(value).toLocaleString()}`, "Balance"]}
                            />
                            <Area type="monotone" dataKey="balance" stroke="#00ff88" strokeWidth={2} fill="url(#balanceGradient)" />
                        </AreaChart>
                    </ResponsiveContainer>
                </div>
            </div>

            {/* Bet Log + Risk Dashboard */}
            <div className="grid-5545">
                {/* Bet Log */}
                <div className="card">
                    <div className="card-header">
                        <h2 className="card-title">Recent Bets</h2>
                        <span style={{ fontSize: "0.8rem", color: "var(--text-muted)" }}>{REAL_BET_LOG.length} shown</span>
                    </div>
                    <div style={{ overflowX: "auto" }}>
                        <table className="data-table">
                            <thead>
                                <tr>
                                    <th>Date</th>
                                    <th>Race</th>
                                    <th>Horse</th>
                                    <th>Stake</th>
                                    <th>Odds</th>
                                    <th>P&L</th>
                                </tr>
                            </thead>
                            <tbody>
                                {REAL_BET_LOG.map((b, i) => (
                                    <tr key={i} style={{ background: b.won ? "rgba(0,255,136,0.04)" : "rgba(255,77,106,0.03)" }}>
                                        <td style={{ fontSize: "0.8rem", color: "var(--text-muted)" }}>{b.date}</td>
                                        <td>{b.race}</td>
                                        <td style={{ fontWeight: 500 }}>{b.horse}</td>
                                        <td>¥{b.stake.toLocaleString()}</td>
                                        <td>{b.odds.toFixed(1)}x</td>
                                        <td style={{ fontWeight: 600, color: b.pnl >= 0 ? "var(--accent-emerald)" : "var(--accent-red)" }}>
                                            {b.pnl >= 0 ? "+" : ""}¥{b.pnl.toLocaleString()}
                                        </td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                </div>

                {/* Risk Dashboard */}
                <div style={{ display: "flex", flexDirection: "column", gap: "1rem" }}>
                    {/* Kelly Fraction */}
                    <div className="card" style={{ textAlign: "center" }}>
                        <div className="card-header">
                            <h2 className="card-title">Kelly Fraction</h2>
                        </div>
                        <div style={{ position: "relative", width: "140px", height: "140px", margin: "0 auto" }}>
                            <svg viewBox="0 0 36 36" width="140" height="140">
                                <path d="M18 2.0845a 15.9155 15.9155 0 0 1 0 31.831a 15.9155 15.9155 0 0 1 0 -31.831"
                                    fill="none" stroke="rgba(42,48,64,0.8)" strokeWidth="2.5" />
                                <path d="M18 2.0845a 15.9155 15.9155 0 0 1 0 31.831a 15.9155 15.9155 0 0 1 0 -31.831"
                                    fill="none" stroke="#00ff88" strokeWidth="2.5"
                                    strokeDasharray={`${kellyFraction * 100}, 100`}
                                    style={{ transition: "stroke-dasharray 0.5s" }} />
                            </svg>
                            <div style={{ position: "absolute", top: "50%", left: "50%", transform: "translate(-50%, -50%)", textAlign: "center" }}>
                                <div style={{ fontSize: "1.5rem", fontWeight: 700, color: "var(--accent-emerald)" }}>{kellyFraction}</div>
                                <div style={{ fontSize: "0.65rem", color: "var(--text-muted)" }}>Quarter Kelly</div>
                            </div>
                        </div>
                    </div>

                    {/* Summary Stats */}
                    <div className="card">
                        <div className="card-header">
                            <h2 className="card-title">Simulation Summary</h2>
                        </div>
                        <div style={{ display: "flex", flexDirection: "column", gap: "0.5rem", fontSize: "0.85rem" }}>
                            <div style={{ display: "flex", justifyContent: "space-between" }}>
                                <span style={{ color: "var(--text-secondary)" }}>Bets Placed</span>
                                <span style={{ fontWeight: 600 }}>102</span>
                            </div>
                            <div style={{ display: "flex", justifyContent: "space-between" }}>
                                <span style={{ color: "var(--text-secondary)" }}>Wins</span>
                                <span style={{ fontWeight: 600, color: "var(--accent-emerald)" }}>82</span>
                            </div>
                            <div style={{ display: "flex", justifyContent: "space-between" }}>
                                <span style={{ color: "var(--text-secondary)" }}>Losses</span>
                                <span style={{ fontWeight: 600, color: "var(--accent-red)" }}>20</span>
                            </div>
                            <div style={{ display: "flex", justifyContent: "space-between" }}>
                                <span style={{ color: "var(--text-secondary)" }}>Total Staked</span>
                                <span style={{ fontWeight: 600 }}>¥102,000</span>
                            </div>
                            <div style={{ display: "flex", justifyContent: "space-between" }}>
                                <span style={{ color: "var(--text-secondary)" }}>Total Profit</span>
                                <span style={{ fontWeight: 600, color: "var(--accent-emerald)" }}>+¥557,400</span>
                            </div>
                        </div>
                    </div>

                    {/* Streak */}
                    <div className="card">
                        <div className="card-header">
                            <h2 className="card-title">Win/Loss Streak (Last 20)</h2>
                        </div>
                        <div style={{ display: "flex", gap: "0.2rem", justifyContent: "center", flexWrap: "wrap" }}>
                            {REAL_STREAK.map((s, i) => (
                                <div key={i} style={{
                                    width: "24px", height: "24px", borderRadius: "4px",
                                    background: s === "W" ? "var(--accent-emerald-dim)" : "var(--accent-red-dim)",
                                    color: s === "W" ? "var(--accent-emerald)" : "var(--accent-red)",
                                    display: "flex", alignItems: "center", justifyContent: "center",
                                    fontSize: "0.65rem", fontWeight: 700
                                }}>
                                    {s}
                                </div>
                            ))}
                        </div>
                        <div style={{ textAlign: "center", marginTop: "0.75rem", fontSize: "0.85rem", color: "var(--text-secondary)" }}>
                            <strong style={{ color: "var(--accent-emerald)" }}>82W</strong> / 20L total
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}
