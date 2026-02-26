"use client";

/**
 * Insights Page — Feature importance, model explainability, and evaluation findings.
 * Updated with real 2025 evaluation results.
 * Source: python -m models.run_evaluation
 */

/** Real feature importance from the trained LightGBM model (gain-based) */
const TOP_FEATURES = [
    { feature: "odds_win", gain: 15016, category: "Odds" },
    { feature: "log_odds", gain: 5861, category: "Odds" },
    { feature: "odds_win_z", gain: 1561, category: "Odds" },
    { feature: "horse_weight_change_z", gain: 1051, category: "Physical" },
    { feature: "weight_carried_z", gain: 1036, category: "Physical" },
    { feature: "sex_code_z", gain: 932, category: "Physical" },
    { feature: "horse_weight_z", gain: 868, category: "Physical" },
    { feature: "pace_style_closer_z", gain: 857, category: "Pace" },
    { feature: "pace_win_prob_z", gain: 781, category: "Pace" },
    { feature: "pace_place_prob_z", gain: 699, category: "Pace" },
    { feature: "horse_weight", gain: 696, category: "Physical" },
    { feature: "log_odds_z", gain: 666, category: "Odds" },
    { feature: "pace_style_stalk_z", gain: 610, category: "Pace" },
    { feature: "weight_trend_z", gain: 558, category: "Physical" },
    { feature: "pace_style_deep_z", gain: 520, category: "Pace" },
];

const SPRINT7_FEATURES = [
    { name: "weather_code", category: "Weather", desc: "Numeric weather encoding (sunny→rain)" },
    { name: "going_x_surface", category: "Weather", desc: "Going × surface interaction" },
    { name: "going_x_distance", category: "Weather", desc: "Going × distance interaction" },
    { name: "horse_going_win_pct", category: "Weather", desc: "Horse's win% on this going" },
    { name: "horse_wet_track_advantage", category: "Weather", desc: "Win% differential: wet − dry" },
    { name: "draw_bias_at_course", category: "Track Bias", desc: "Avg finish for this draw at course" },
    { name: "draw_low_win_pct", category: "Track Bias", desc: "Inner draw (1-4) win rate" },
    { name: "draw_high_win_pct", category: "Track Bias", desc: "Outer draw (9+) win rate" },
    { name: "draw_bias_score", category: "Track Bias", desc: "Signed advantage for horse's draw" },
    { name: "course_month_bias", category: "Track Bias", desc: "Seasonal draw bias by month" },
    { name: "sire_runners", category: "Pedigree", desc: "Sire offspring race count" },
    { name: "sire_win_pct", category: "Pedigree", desc: "Sire offspring overall win rate" },
    { name: "sire_win_pct_surface", category: "Pedigree", desc: "Sire × surface affinity" },
    { name: "sire_win_pct_distance", category: "Pedigree", desc: "Sire × distance affinity" },
    { name: "sire_avg_finish", category: "Pedigree", desc: "Sire offspring avg finish" },
];

const CATEGORY_COLORS: Record<string, string> = {
    "Odds": "#ff6b6b",
    "Physical": "#4da6ff",
    "Pace": "#00ff88",
    "Weather": "#4da6ff",
    "Track Bias": "#00ff88",
    "Pedigree": "#ffaa33",
};

/** Decision matrix from evaluation */
const DECISION_MATRIX = [
    { check: "ROI positive at 5% EV", status: "✅", result: "+23.6%", action: "Safe to paper trade at 5% threshold" },
    { check: "ROI positive at 8% EV", status: "✅", result: "+28.4%", action: "Consider 8% for concentrated portfolio" },
    { check: "Odds-free AUC ≥ 0.72", status: "✅", result: "0.819", action: "Fundamental edge confirmed" },
    { check: "Odds-free profitable", status: "❌", result: "-52.2%", action: "Odds features are critical" },
    { check: "Trio ROI > Win ROI", status: "❌", result: "0%", action: "Stick with win bets only" },
    { check: "Dominant failure mode", status: "⚠️", result: "73% no_edge", action: "Target in next feature sprint" },
];

/** Calibration curve from evaluation */
const CALIBRATION_DATA = [
    { predicted: 2, actual: 3.5, n: 960 },
    { predicted: 14, actual: 12.8, n: 78 },
    { predicted: 24, actual: 22.1, n: 68 },
    { predicted: 33, actual: 26.7, n: 30 },
    { predicted: 44, actual: 33.3, n: 30 },
    { predicted: 59, actual: 47.1, n: 17 },
    { predicted: 62, actual: 50.0, n: 4 },
    { predicted: 85, actual: 75.0, n: 12 },
    { predicted: 92, actual: 66.7, n: 6 },
];

export default function InsightsPage() {
    const maxGain = TOP_FEATURES[0]?.gain || 1;

    return (
        <div className="page-container animate-in">
            <div className="page-header">
                <h1 className="page-title">Model Insights</h1>
                <p className="page-subtitle">2025 evaluation — LightGBM + XGBoost ensemble, 126 features, 163 total columns</p>
            </div>

            {/* Feature Importance Bar Chart */}
            <div className="card" style={{ marginBottom: "1.5rem" }}>
                <div className="card-header">
                    <h2 className="card-title">📊 Feature Importance (Top 15 by Gain)</h2>
                    <span className="badge badge-blue">LightGBM gain values</span>
                </div>
                <div style={{ display: "flex", flexDirection: "column", gap: "0.4rem" }}>
                    {TOP_FEATURES.map((f, i) => {
                        const barWidth = (f.gain / maxGain) * 100;
                        const color = CATEGORY_COLORS[f.category] || "#4da6ff";
                        return (
                            <div key={i} style={{ display: "flex", alignItems: "center", gap: "0.75rem" }}>
                                <span style={{
                                    width: "200px",
                                    fontSize: "0.78rem",
                                    fontWeight: 600,
                                    color: color,
                                    textAlign: "right",
                                    fontFamily: "monospace",
                                }}>
                                    {f.feature}
                                </span>
                                <div style={{ flex: 1, position: "relative", height: "22px" }}>
                                    <div style={{
                                        width: `${barWidth}%`,
                                        height: "100%",
                                        borderRadius: "4px",
                                        background: `linear-gradient(90deg, ${color}, ${color}66)`,
                                        transition: "width 0.5s ease",
                                    }} />
                                    <span style={{
                                        position: "absolute",
                                        right: "8px",
                                        top: "50%",
                                        transform: "translateY(-50%)",
                                        fontSize: "0.72rem",
                                        fontWeight: 600,
                                        color: "var(--text-primary)",
                                    }}>
                                        {f.gain.toLocaleString()}
                                    </span>
                                </div>
                                <span style={{
                                    width: "60px",
                                    fontSize: "0.7rem",
                                    color: "var(--text-muted)",
                                    fontStyle: "italic",
                                }}>
                                    {f.category}
                                </span>
                            </div>
                        );
                    })}
                </div>
                <div style={{ marginTop: "1rem", fontSize: "0.85rem", color: "var(--text-muted)", lineHeight: 1.6 }}>
                    💡 <strong style={{ color: "#ff6b6b" }}>Odds features</strong> dominate (22,104 total gain) — the model
                    leverages market pricing heavily. <strong style={{ color: "#4da6ff" }}>Physical</strong> features (weight changes,
                    sex) and <strong style={{ color: "#00ff88" }}>pace</strong> features (running style, pace probability) provide
                    the model&apos;s fundamental edge beyond odds.
                </div>
            </div>

            {/* Calibration Curve */}
            <div className="card" style={{ marginBottom: "1.5rem" }}>
                <div className="card-header">
                    <h2 className="card-title">📈 Model Calibration</h2>
                    <span className="badge badge-green">Isotonic regression</span>
                </div>
                <div style={{ display: "flex", flexDirection: "column", gap: "0.35rem" }}>
                    {CALIBRATION_DATA.map((c, i) => {
                        const barWidth = (c.actual / 100) * 100;
                        const isCalibrated = Math.abs(c.predicted - c.actual) < 10;
                        return (
                            <div key={i} style={{ display: "flex", alignItems: "center", gap: "0.75rem" }}>
                                <span style={{
                                    width: "80px",
                                    fontSize: "0.78rem",
                                    fontWeight: 600,
                                    color: "var(--text-secondary)",
                                    textAlign: "right",
                                    fontFamily: "monospace",
                                }}>
                                    P={c.predicted}%
                                </span>
                                <div style={{ flex: 1, position: "relative", height: "20px", background: "rgba(42,48,64,0.3)", borderRadius: "4px" }}>
                                    {/* Predicted marker */}
                                    <div style={{
                                        position: "absolute",
                                        left: `${c.predicted}%`,
                                        top: 0,
                                        bottom: 0,
                                        width: "2px",
                                        background: "rgba(255,255,255,0.2)",
                                    }} />
                                    {/* Actual bar */}
                                    <div style={{
                                        width: `${barWidth}%`,
                                        height: "100%",
                                        borderRadius: "4px",
                                        background: isCalibrated
                                            ? "linear-gradient(90deg, #00ff88, #00ff8866)"
                                            : "linear-gradient(90deg, #ffaa33, #ffaa3366)",
                                        transition: "width 0.5s ease",
                                    }} />
                                    <span style={{
                                        position: "absolute",
                                        right: "8px",
                                        top: "50%",
                                        transform: "translateY(-50%)",
                                        fontSize: "0.7rem",
                                        fontWeight: 600,
                                        color: "var(--text-primary)",
                                    }}>
                                        actual={c.actual}% (n={c.n})
                                    </span>
                                </div>
                            </div>
                        );
                    })}
                </div>
                <div style={{ marginTop: "0.75rem", fontSize: "0.85rem", color: "var(--text-muted)" }}>
                    The model is well-calibrated for low probabilities (P=2% → actual 3.5%) but slightly over-confident for mid-range predictions.
                </div>
            </div>

            {/* Decision Matrix */}
            <div className="card" style={{ marginBottom: "1.5rem" }}>
                <div className="card-header">
                    <h2 className="card-title">🎯 Decision Matrix</h2>
                    <span className="badge badge-gold">Evaluation outcomes</span>
                </div>
                <table className="data-table">
                    <thead>
                        <tr>
                            <th>Check</th>
                            <th>Status</th>
                            <th>Result</th>
                            <th>Action</th>
                        </tr>
                    </thead>
                    <tbody>
                        {DECISION_MATRIX.map((row, i) => (
                            <tr key={i}>
                                <td style={{ fontWeight: 600 }}>{row.check}</td>
                                <td style={{ fontSize: "1.1rem" }}>{row.status}</td>
                                <td style={{
                                    fontWeight: 700,
                                    fontFamily: "monospace",
                                    color: row.status === "✅" ? "var(--accent-emerald)" :
                                        row.status === "❌" ? "var(--accent-red, #ff5555)" : "var(--accent-gold)"
                                }}>
                                    {row.result}
                                </td>
                                <td style={{ color: "var(--text-secondary)" }}>{row.action}</td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>

            {/* Sprint 7 Feature Reference */}
            <div className="card">
                <div className="card-header">
                    <h2 className="card-title">🆕 Sprint 7 Features (Active)</h2>
                    <span className="badge badge-green">15 features · 163 total columns</span>
                </div>
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: "1rem" }}>
                    {["Weather", "Track Bias", "Pedigree"].map(cat => (
                        <div key={cat} className="card" style={{ padding: "1rem" }}>
                            <h3 style={{
                                fontSize: "0.8rem",
                                fontWeight: 700,
                                color: CATEGORY_COLORS[cat],
                                marginBottom: "0.75rem",
                                textTransform: "uppercase",
                                letterSpacing: "0.05em",
                            }}>
                                {cat === "Weather" ? "🌧️" : cat === "Track Bias" ? "📐" : "🧬"} {cat}
                            </h3>
                            <div style={{ display: "flex", flexDirection: "column", gap: "0.5rem" }}>
                                {SPRINT7_FEATURES.filter(f => f.category === cat).map((f, i) => (
                                    <div key={i} style={{ fontSize: "0.8rem" }}>
                                        <div style={{ fontWeight: 600, color: "var(--text-primary)", fontFamily: "monospace" }}>
                                            {f.name}
                                        </div>
                                        <div style={{ color: "var(--text-muted)", fontSize: "0.75rem" }}>
                                            {f.desc}
                                        </div>
                                    </div>
                                ))}
                            </div>
                        </div>
                    ))}
                </div>
            </div>
        </div>
    );
}
