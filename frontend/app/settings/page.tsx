"use client";

import { useState } from "react";

export default function SettingsPage() {
    const [settings, setSettings] = useState({
        modelVersion: "v1.0",
        evThreshold: 10,
        minOdds: 3.0,
        confidence: 60,
        initialBankroll: 100000,
        kellyFraction: 0.25,
        maxDailyExposure: 5,
        maxSingleBet: 50000,
        drawdownThreshold: 15,
        minBet: 100,
        maxBet: 50000,
        enableLine: false,
        enableTelegram: false,
        telegramToken: "",
        telegramChatId: "",
        alertHighEV: true,
        alertRaceDay: true,
        alertResults: false,
        scrapeSchedule: "race_days",
    });

    const update = (key: string, value: unknown) => setSettings((s) => ({ ...s, [key]: value }));

    return (
        <div className="page-container animate-in">
            <div className="page-header">
                <h1 className="page-title">Settings</h1>
                <p className="page-subtitle">Configure model parameters, bankroll rules, and notifications</p>
            </div>

            <div style={{ display: "grid", gap: "1.5rem", maxWidth: "800px" }}>
                {/* Model Parameters */}
                <div className="card">
                    <div className="card-header">
                        <h2 className="card-title">📊 Model Parameters</h2>
                    </div>
                    <div className="form-group">
                        <label className="form-label">Model Version</label>
                        <select className="form-select" value={settings.modelVersion} onChange={(e) => update("modelVersion", e.target.value)}>
                            <option value="v1.0">v1.0 (Stable)</option>
                            <option value="v1.1-beta">v1.1-beta (Experimental)</option>
                        </select>
                    </div>
                    <div className="form-group">
                        <label className="form-label">EV Threshold: {settings.evThreshold}%</label>
                        <input type="range" min="5" max="30" value={settings.evThreshold} onChange={(e) => update("evThreshold", +e.target.value)}
                            style={{ width: "100%", accentColor: "var(--accent-emerald)" }} />
                        <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.7rem", color: "var(--text-muted)" }}>
                            <span>5%</span><span>30%</span>
                        </div>
                    </div>
                    <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "1rem" }}>
                        <div className="form-group">
                            <label className="form-label">Min Odds</label>
                            <input type="number" className="form-input" value={settings.minOdds} step="0.5"
                                onChange={(e) => update("minOdds", +e.target.value)} />
                        </div>
                        <div className="form-group">
                            <label className="form-label">Confidence: {settings.confidence}%</label>
                            <input type="range" min="0" max="100" value={settings.confidence} onChange={(e) => update("confidence", +e.target.value)}
                                style={{ width: "100%", accentColor: "var(--accent-emerald)" }} />
                        </div>
                    </div>
                </div>

                {/* Bankroll Configuration */}
                <div className="card">
                    <div className="card-header">
                        <h2 className="card-title">💰 Bankroll Configuration</h2>
                    </div>
                    <div className="form-group">
                        <label className="form-label">Initial Bankroll (¥)</label>
                        <input type="number" className="form-input" value={settings.initialBankroll}
                            onChange={(e) => update("initialBankroll", +e.target.value)} />
                    </div>
                    <div className="form-group">
                        <label className="form-label">Kelly Fraction: {settings.kellyFraction} ({settings.kellyFraction === 0.25 ? "Quarter-Kelly" : settings.kellyFraction === 0.5 ? "Half-Kelly" : "Custom"})</label>
                        <input type="range" min="0.1" max="1.0" step="0.05" value={settings.kellyFraction}
                            onChange={(e) => update("kellyFraction", +e.target.value)}
                            style={{ width: "100%", accentColor: "var(--accent-emerald)" }} />
                        <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.7rem", color: "var(--text-muted)" }}>
                            <span>0.1 (Conservative)</span><span>1.0 (Full Kelly)</span>
                        </div>
                    </div>
                    <div className="form-group">
                        <label className="form-label">Max Daily Exposure: {settings.maxDailyExposure}%</label>
                        <input type="range" min="1" max="10" value={settings.maxDailyExposure}
                            onChange={(e) => update("maxDailyExposure", +e.target.value)}
                            style={{ width: "100%", accentColor: "var(--accent-emerald)" }} />
                    </div>
                    <div className="form-group">
                        <label className="form-label">Drawdown Circuit Breaker: {settings.drawdownThreshold}%</label>
                        <input type="range" min="5" max="30" value={settings.drawdownThreshold}
                            onChange={(e) => update("drawdownThreshold", +e.target.value)}
                            style={{ width: "100%", accentColor: "var(--accent-red)" }} />
                    </div>
                    <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "1rem" }}>
                        <div className="form-group">
                            <label className="form-label">Min Bet (¥)</label>
                            <input type="number" className="form-input" value={settings.minBet}
                                onChange={(e) => update("minBet", +e.target.value)} />
                        </div>
                        <div className="form-group">
                            <label className="form-label">Max Bet (¥)</label>
                            <input type="number" className="form-input" value={settings.maxBet}
                                onChange={(e) => update("maxBet", +e.target.value)} />
                        </div>
                    </div>
                </div>

                {/* Notifications */}
                <div className="card">
                    <div className="card-header">
                        <h2 className="card-title">🔔 Notifications</h2>
                    </div>
                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "1rem" }}>
                        <span style={{ color: "var(--text-primary)" }}>Enable LINE Notifications</span>
                        <input type="checkbox" className="form-toggle" checked={settings.enableLine}
                            onChange={(e) => update("enableLine", e.target.checked)} />
                    </div>
                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "1rem" }}>
                        <span style={{ color: "var(--text-primary)" }}>Enable Telegram Notifications</span>
                        <input type="checkbox" className="form-toggle" checked={settings.enableTelegram}
                            onChange={(e) => update("enableTelegram", e.target.checked)} />
                    </div>
                    {settings.enableTelegram && (
                        <div style={{ paddingLeft: "1rem", borderLeft: "2px solid var(--border-default)" }}>
                            <div className="form-group">
                                <label className="form-label">Bot Token</label>
                                <input type="password" className="form-input" value={settings.telegramToken} placeholder="Enter token..."
                                    onChange={(e) => update("telegramToken", e.target.value)} />
                            </div>
                            <div className="form-group">
                                <label className="form-label">Chat ID</label>
                                <input type="text" className="form-input" value={settings.telegramChatId} placeholder="Enter chat ID..."
                                    onChange={(e) => update("telegramChatId", e.target.value)} />
                            </div>
                        </div>
                    )}
                    <div style={{ marginTop: "1rem" }}>
                        <div className="form-label">Alert Types</div>
                        {[
                            { key: "alertHighEV", label: "High EV alerts (EV > threshold)" },
                            { key: "alertRaceDay", label: "Race day morning summary" },
                            { key: "alertResults", label: "Result updates" },
                        ].map((a) => (
                            <label key={a.key} style={{ display: "flex", alignItems: "center", gap: "0.75rem", marginBottom: "0.5rem", cursor: "pointer" }}>
                                <input type="checkbox" checked={settings[a.key as keyof typeof settings] as boolean}
                                    onChange={(e) => update(a.key, e.target.checked)}
                                    style={{ accentColor: "var(--accent-emerald)" }} />
                                <span style={{ fontSize: "0.875rem", color: "var(--text-primary)" }}>{a.label}</span>
                            </label>
                        ))}
                    </div>
                </div>

                {/* Data & Scraper */}
                <div className="card">
                    <div className="card-header">
                        <h2 className="card-title">🔄 Data & Scraper</h2>
                    </div>
                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "1rem" }}>
                        <div>
                            <div style={{ fontSize: "0.85rem", color: "var(--text-primary)" }}>Last Scrape</div>
                            <div style={{ fontSize: "0.8rem", color: "var(--text-muted)" }}>2024-02-25 06:30:00 UTC</div>
                        </div>
                        <button className="btn">Run Manual Scrape</button>
                    </div>
                    <div className="form-group">
                        <label className="form-label">Auto-Scrape Schedule</label>
                        <select className="form-select" value={settings.scrapeSchedule} onChange={(e) => update("scrapeSchedule", e.target.value)}>
                            <option value="daily">Daily</option>
                            <option value="weekly">Weekly</option>
                            <option value="race_days">Race Days Only</option>
                        </select>
                    </div>
                </div>

                {/* Save Button */}
                <button className="btn btn-primary" style={{ width: "100%", justifyContent: "center", padding: "0.875rem", fontSize: "1rem" }}>
                    Save Settings
                </button>
            </div>
        </div>
    );
}
