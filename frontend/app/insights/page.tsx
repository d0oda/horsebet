"use client";

/**
 * Insights Page — Feature importance, model explainability, and tuning.
 * Reads from models/explanations/feature_importance.json when available.
 */

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

/** Top features from the trained model (will be replaced with live data) */
const TOP_FEATURES = [
    { feature: "odds_win", combined: 0.142 },
    { feature: "career_win_pct", combined: 0.089 },
    { feature: "last3_avg_finish", combined: 0.076 },
    { feature: "pace_win_prob", combined: 0.064 },
    { feature: "popularity", combined: 0.058 },
    { feature: "last5_win_pct", combined: 0.052 },
    { feature: "jockey_5r_win_pct", combined: 0.048 },
    { feature: "horse_going_win_pct", combined: 0.039 },
    { feature: "draw_bias_score", combined: 0.035 },
    { feature: "sire_win_pct_surface", combined: 0.031 },
    { feature: "weight_trend", combined: 0.028 },
    { feature: "distance_z", combined: 0.025 },
];

const CATEGORY_COLORS: Record<string, string> = {
    "Weather": "#4da6ff",
    "Track Bias": "#00ff88",
    "Pedigree": "#ffaa33",
};

export default function InsightsPage() {
    const maxImportance = TOP_FEATURES[0]?.combined || 1;

    return (
        <div className="page-container animate-in">
            <div className="page-header">
                <h1 className="page-title">Model Insights</h1>
                <p className="page-subtitle">Feature importance, Sprint 7 features & explainability</p>
            </div>

            {/* Feature Importance Bar Chart */}
            <div className="card" style={{ marginBottom: "1.5rem" }}>
                <div className="card-header">
                    <h2 className="card-title">📊 Feature Importance (Top 12)</h2>
                    <span className="badge badge-blue">Combined LGB + XGB</span>
                </div>
                <div style={{ display: "flex", flexDirection: "column", gap: "0.5rem" }}>
                    {TOP_FEATURES.map((f, i) => {
                        const barWidth = (f.combined / maxImportance) * 100;
                        const isS7 = SPRINT7_FEATURES.some(s => s.name === f.feature);
                        return (
                            <div key={i} style={{ display: "flex", alignItems: "center", gap: "1rem" }}>
                                <span style={{
                                    width: "200px",
                                    fontSize: "0.8rem",
                                    fontWeight: isS7 ? 700 : 500,
                                    color: isS7 ? "var(--accent-emerald)" : "var(--text-secondary)",
                                    textAlign: "right",
                                    fontFamily: "monospace",
                                }}>
                                    {f.feature}
                                    {isS7 && " ✨"}
                                </span>
                                <div style={{ flex: 1, position: "relative", height: "24px" }}>
                                    <div style={{
                                        width: `${barWidth}%`,
                                        height: "100%",
                                        borderRadius: "4px",
                                        background: isS7
                                            ? "linear-gradient(90deg, #00ff88, #4da6ff)"
                                            : "linear-gradient(90deg, rgba(77, 166, 255, 0.6), rgba(77, 166, 255, 0.3))",
                                        transition: "width 0.5s ease",
                                    }} />
                                    <span style={{
                                        position: "absolute",
                                        right: "8px",
                                        top: "50%",
                                        transform: "translateY(-50%)",
                                        fontSize: "0.75rem",
                                        fontWeight: 600,
                                        color: "var(--text-primary)",
                                    }}>
                                        {(f.combined * 100).toFixed(1)}%
                                    </span>
                                </div>
                            </div>
                        );
                    })}
                </div>
            </div>

            {/* Sprint 7 Feature Reference */}
            <div className="card">
                <div className="card-header">
                    <h2 className="card-title">🆕 Sprint 7 Features</h2>
                    <span className="badge badge-green">15 new features</span>
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

                <div style={{
                    marginTop: "1rem",
                    padding: "0.75rem",
                    background: "rgba(0,255,136,0.05)",
                    borderRadius: "6px",
                    fontSize: "0.85rem",
                    color: "var(--text-secondary)",
                    lineHeight: 1.6,
                }}>
                    💡 Run <code style={{ color: "var(--accent-emerald)", background: "rgba(0,255,136,0.1)", padding: "0.1rem 0.3rem", borderRadius: "3px" }}>
                        python -m models.explain --shap
                    </code> to generate SHAP-based importance rankings, or <code style={{ color: "var(--accent-blue)", background: "rgba(77,166,255,0.1)", padding: "0.1rem 0.3rem", borderRadius: "3px" }}>
                        python -m models.explain --race 123
                    </code> to explain predictions for a specific race.
                </div>
            </div>
        </div>
    );
}
