"use client";

/**
 * Insights Page — Feature importance, model explainability, and evaluation findings.
 * Updated with real 2025 evaluation results.
 * Source: python -m models.run_evaluation
 */

/** Real feature importance from the retrained LightGBM model (gain-based, 130 features) */
const TOP_FEATURES = [
    { feature: "log_odds", gain: 11007, category: "Odds" },
    { feature: "odds_win", gain: 9304, category: "Odds" },
    { feature: "odds_win_z", gain: 1467, category: "Odds" },
    { feature: "horse_weight_change_z", gain: 1102, category: "Physical" },
    { feature: "popularity", gain: 1083, category: "Odds" },
    { feature: "horse_weight_z", gain: 1049, category: "Physical" },
    { feature: "pace_win_prob_z", gain: 914, category: "Pace" },
    { feature: "pace_style_closer_z", gain: 885, category: "Pace" },
    { feature: "sex_code_z", gain: 850, category: "Physical" },
    { feature: "weight_carried_z", gain: 845, category: "Physical" },
    { feature: "horse_weight", gain: 681, category: "Physical" },
    { feature: "pace_place_prob_z", gain: 671, category: "Pace" },
    { feature: "pace_style_stalk_z", gain: 611, category: "Pace" },
    { feature: "pace_style_deep_z", gain: 603, category: "Pace" },
    { feature: "post_position_z", gain: 573, category: "Physical" },
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
    // Research Backlog features
    { name: "class_change", category: "Class Change", desc: "Direction: dropped (−) or rising (+) in class" },
    { name: "class_drops_last5", category: "Class Change", desc: "# of class drops in last 5 races" },
    { name: "class_rises_last5", category: "Class Change", desc: "# of class rises in last 5 races" },
    { name: "class_at_last_win", category: "Class Change", desc: "Class rank at most recent win" },
    { name: "trainer_14d_runs", category: "Trainer Form", desc: "Trainer runners in last 14 days" },
    { name: "trainer_14d_win_pct", category: "Trainer Form", desc: "Trainer 14-day win rate" },
    { name: "trainer_14d_place_pct", category: "Trainer Form", desc: "Trainer 14-day place rate" },
    { name: "jockey_course_runs", category: "Course×Jockey", desc: "Jockey rides at this specific course" },
    { name: "jockey_course_win_pct", category: "Course×Jockey", desc: "Jockey win rate at this course" },
    { name: "jockey_course_place_pct", category: "Course×Jockey", desc: "Jockey place rate at this course" },
    { name: "horse_heavy_speed_diff", category: "Weather+", desc: "Speed delta: wet vs dry races" },
    { name: "going_x_dist_x_surface", category: "Weather+", desc: "3-way: going × distance × surface" },
];

const CATEGORY_COLORS: Record<string, string> = {
    "Odds": "#ff6b6b",
    "Physical": "#4da6ff",
    "Pace": "#00ff88",
    "Weather": "#4da6ff",
    "Track Bias": "#00ff88",
    "Pedigree": "#ffaa33",
    "Class Change": "#ff88cc",
    "Trainer Form": "#bb88ff",
    "Course×Jockey": "#88ffdd",
    "Weather+": "#66bbff",
};

/** Decision matrix from evaluation (retrained with research backlog features) */
const DECISION_MATRIX = [
    { check: "ROI positive at 5% EV", status: "⚠️", result: "0 bets", action: "Isotonic calibration too conservative — needs tuning" },
    { check: "ROI positive at 8% EV", status: "⚠️", result: "0 bets", action: "Same: calibrator compresses probabilities" },
    { check: "OOS AUC ≥ 0.84", status: "✅", result: "0.850", action: "Model discriminative power improved" },
    { check: "LightGBM AUC", status: "✅", result: "0.851", action: "Best single model" },
    { check: "XGBoost AUC", status: "✅", result: "0.840", action: "Ensemble diversity maintained" },
    { check: "Features expanded", status: "✅", result: "130 feats", action: "+4 class, +3 trainer, +3 jockey, +2 weather" },
];

/** Calibration curve from retrained model evaluation (OOS 2025) */
const CALIBRATION_DATA = [
    { predicted: 2, actual: 3.7, n: 984 },
    { predicted: 14, actual: 26.8, n: 56 },
    { predicted: 23, actual: 9.8, n: 51 },
    { predicted: 34, actual: 26.7, n: 45 },
    { predicted: 43, actual: 40.0, n: 30 },
    { predicted: 55, actual: 21.1, n: 19 },
    { predicted: 60, actual: 66.7, n: 3 },
    { predicted: 75, actual: 71.4, n: 7 },
    { predicted: 86, actual: 66.7, n: 12 },
];

export default function InsightsPage() {
    const maxGain = TOP_FEATURES[0]?.gain || 1;

    return (
        <div className="page-container animate-in">
            <div className="page-header">
                <h1 className="page-title">Model Insights</h1>
                <p className="page-subtitle">Retrained 2025 evaluation — LightGBM + XGBoost ensemble, 130 features (+12 research backlog), 187 columns</p>
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
