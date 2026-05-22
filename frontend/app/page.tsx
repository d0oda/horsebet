"use client";

import { useState, useEffect } from "react";
import { api, type Race, type Entry, type ValueBet, type Prediction } from "@/lib/api";

export default function RacedayDashboard() {
  const [races, setRaces] = useState<Race[]>([]);
  const [selectedRace, setSelectedRace] = useState<Race | null>(null);
  
  // Detail state
  const [entries, setEntries] = useState<Entry[]>([]);
  const [valueBets, setValueBets] = useState<ValueBet[]>([]);
  const [predictions, setPredictions] = useState<Prediction[]>([]);
  const [loading, setLoading] = useState(false);
  const [refreshing, setRefreshing] = useState(false);

  useEffect(() => {
    fetchRaces();
  }, []);

  const fetchRaces = async () => {
    try {
      const res = await api.getRaces({ limit: 12 });
      const sorted = res.races.sort((a, b) => a.race_number - b.race_number);
      setRaces(sorted);
      if (sorted.length > 0) {
        selectRace(sorted[0].id);
      }
    } catch (e) {
      console.error(e);
    }
  };

  const selectRace = async (id: number) => {
    setLoading(true);
    try {
      const race = races.find(r => r.id === id);
      if (race) setSelectedRace(race);
      
      const res = await api.getRace(id);
      setEntries(res.entries || []);
      setValueBets(res.value_bets || []);
      setPredictions(res.predictions || []);
    } catch (e) {
      console.error(e);
    } finally {
      setLoading(false);
    }
  };

  const handleRefresh = async () => {
    if (!selectedRace) return;
    setRefreshing(true);
    try {
      await api.refreshRace(selectedRace.id);
      await selectRace(selectedRace.id);
    } catch (e) {
      console.error(e);
      alert("Failed to refresh: " + String(e));
    } finally {
      setRefreshing(false);
    }
  };

  const getDecisionBadge = (decision: string) => {
    switch (decision) {
      case "Strong Value": return { label: "Strong Value", color: "bg-primary text-background" };
      case "Value": return { label: "Value", color: "bg-tertiary text-on-tertiary" };
      case "Lean": return { label: "Lean", color: "bg-secondary text-on-secondary" };
      case "Watch Only": return { label: "Watch Only", color: "bg-surface-container-highest text-on-surface-variant font-bold text-yellow-600 dark:text-yellow-400" };
      case "Avoid":
      default: return { label: "Avoid", color: "bg-error-container text-on-error-container" };
    }
  };

  return (
    <>
      <aside className="flex flex-col h-full w-60 bg-surface-container-low border-r border-outline-variant py-md overflow-y-auto shrink-0">
        <div className="px-md mb-lg">
          <h1 className="font-title-md text-title-md text-on-surface mb-xs">Raceday Select</h1>
          <p className="font-label-caps text-label-caps text-on-surface-variant opacity-70">
            {races.length > 0 ? races[0].courses?.name_jp || "All Tracks" : "Loading..."}
          </p>
        </div>
        <nav className="flex-1 px-sm space-y-1">
          {races.map((r) => {
            const isActive = selectedRace?.id === r.id;
            return (
              <button 
                key={r.id}
                onClick={() => selectRace(r.id)}
                className={`w-full flex items-center px-md py-sm rounded-lg transition-all ${
                  isActive 
                  ? "text-primary font-bold bg-surface-container-high border-l-2 border-primary scale-95" 
                  : "text-on-surface-variant hover:text-on-surface hover:bg-surface-container group"
                }`}
              >
                <span className="material-symbols-outlined mr-sm text-[20px]" data-icon="shuttle_run">airport_shuttle</span>
                <span className="font-body-sm text-body-sm">{r.race_number}R {r.courses?.name_jp || r.courses?.name || "Track"}</span>
              </button>
            )
          })}
        </nav>
      </aside>
      
      <main className="flex-1 flex flex-col min-w-0 bg-background relative">
        <header className="flex justify-between items-center w-full px-lg h-16 bg-surface border-b border-outline-variant shrink-0">
          <div className="flex items-center gap-md">
            <div className="font-headline-lg text-headline-lg font-bold text-primary dark:text-primary">UmaEdge Pro</div>
            <div className="h-6 w-[1px] bg-outline-variant mx-sm"></div>
            <div className="font-title-md text-title-md text-on-surface">
              {selectedRace ? `${selectedRace.courses?.name_jp || "Track"} ${selectedRace.race_number}R — ${selectedRace.race_name_jp}` : "Loading..."}
            </div>
          </div>
          <div className="flex items-center gap-md">
            <button 
              onClick={handleRefresh}
              disabled={refreshing}
              className="bg-primary-container text-on-primary-container px-lg py-2 rounded-lg font-label-caps text-label-caps hover:opacity-80 transition-all flex items-center gap-sm disabled:opacity-50"
            >
              <span className={`material-symbols-outlined text-[18px] ${refreshing ? "animate-spin" : ""}`} data-icon="refresh">refresh</span>
              {refreshing ? "Refreshing..." : "Refresh Live Odds"}
            </button>
          </div>
        </header>
        
        <div className="flex-1 overflow-y-auto p-lg space-y-lg">
          <section className="bg-surface-container-low border border-outline-variant rounded-xl overflow-hidden">
            <div className="px-lg py-md border-b border-outline-variant bg-surface-container flex justify-between items-center">
              <h3 className="font-label-caps text-label-caps text-on-surface">Live Market Depth & Analytics</h3>
            </div>
            <div className="overflow-x-auto">
              {loading ? (
                <div className="p-10 text-center text-on-surface-variant">Loading Data...</div>
              ) : (
              <table className="w-full text-left border-collapse">
                <thead>
                  <tr className="bg-surface-container-high">
                    <th className="px-md py-sm font-label-caps text-label-caps text-on-surface-variant border-b border-outline-variant">#</th>
                    <th className="px-md py-sm font-label-caps text-label-caps text-on-surface-variant border-b border-outline-variant">Horse Name</th>
                    <th className="px-md py-sm font-label-caps text-label-caps text-on-surface-variant border-b border-outline-variant text-right">Current Odds</th>
                    <th className="px-md py-sm font-label-caps text-label-caps text-on-surface-variant border-b border-outline-variant text-center">Weight</th>
                    <th className="px-md py-sm font-label-caps text-label-caps text-on-surface-variant border-b border-outline-variant text-right">Fair Odds</th>
                    <th className="px-md py-sm font-label-caps text-label-caps text-on-surface-variant border-b border-outline-variant text-right">Ability</th>
                    <th className="px-md py-sm font-label-caps text-label-caps text-on-surface-variant border-b border-outline-variant text-right">Cond. Fit</th>
                    <th className="px-md py-sm font-label-caps text-label-caps text-on-surface-variant border-b border-outline-variant text-right">EV Edge</th>
                    <th className="px-md py-sm font-label-caps text-label-caps text-on-surface-variant border-b border-outline-variant text-right">Decision</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-outline-variant/50">
                  {entries.map(entry => {
                    const vb = valueBets.find(v => v.entry_id === entry.id && v.bet_type === "win");
                    const pred = predictions.find(p => p.entry_id === entry.id);
                    const ev = pred?.edge ?? (vb ? vb.ev : 0);
                    const decision = pred?.decision || "Avoid";
                    const badge = getDecisionBadge(decision);
                    
                    return (
                      <tr key={entry.id} className="hover:bg-surface-container transition-colors group">
                        <td className="px-md py-sm tabular-nums font-bold text-on-surface">{entry.draw || entry.post_position}</td>
                        <td className="px-md py-sm">
                          <div className="flex flex-col">
                            <span className="font-title-md text-body-md font-semibold">{entry.horses?.name_jp || entry.horses?.name || "Unknown"}</span>
                            <span className="text-[10px] text-on-surface-variant uppercase tracking-widest">{entry.jockeys?.name_jp || entry.jockeys?.name}</span>
                          </div>
                        </td>
                        <td className="px-md py-sm text-right tabular-nums font-bold text-secondary">{entry.odds_win?.toFixed(1) || "-"}x</td>
                        <td className="px-md py-sm text-center">
                          <div className="flex flex-col items-center">
                            <span className="tabular-nums">{entry.horse_weight || "?"}</span>
                            <span className="text-[10px] text-on-surface-variant">({entry.horse_weight_change > 0 ? '+' : ''}{entry.horse_weight_change || 0})</span>
                          </div>
                        </td>
                        <td className="px-md py-sm text-right tabular-nums font-bold">
                          {pred?.fair_odds ? `${pred.fair_odds.toFixed(1)}x` : "-"}
                        </td>
                        <td className="px-md py-sm text-right tabular-nums font-bold text-primary">
                          {pred?.ability_rating ? pred.ability_rating.toFixed(0) : "-"}
                        </td>
                        <td className="px-md py-sm text-right tabular-nums font-bold text-tertiary">
                          <div className="flex items-center justify-end gap-1">
                            {pred?.bounce_risk && <span title="Bounce Risk" className="text-yellow-500 text-sm">⚠️</span>}
                            {pred?.condition_fit ? pred.condition_fit.toFixed(0) : "-"}
                          </div>
                        </td>
                        <td className="px-md py-sm text-right tabular-nums font-bold" style={{ color: ev > 0 ? "var(--primary)" : "" }}>
                          {(pred?.edge !== undefined || vb) ? `${ev > 0 ? '+' : ''}${(ev * 100).toFixed(1)}%` : "-"}
                        </td>
                        <td className="px-md py-sm text-right">
                          <span className={`px-3 py-1 rounded-full font-label-caps text-[10px] font-bold ${badge.color}`}>
                            {badge.label}
                          </span>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
              )}
            </div>
          </section>
        </div>
      </main>
    </>
  );
}
