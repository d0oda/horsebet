/**
 * UmaEdge — API Client
 * Fetch wrapper for the FastAPI backend.
 */

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

async function fetchAPI<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });

  if (!res.ok) {
    throw new Error(`API error: ${res.status} ${res.statusText}`);
  }

  return res.json();
}

// --- Types ---

export interface Race {
  id: number;
  netkeiba_id: string;
  date: string;
  race_number: number;
  distance: number;
  surface: string;
  going: string;
  class: string;
  grade: string;
  race_name: string;
  race_name_jp: string;
  weather: string;
  field_size: number;
  courses?: { name: string; name_jp: string; surface: string };
}

export interface Entry {
  id: number;
  race_id: number;
  draw: number;
  post_position: number;
  weight_carried: number;
  horse_weight: number;
  horse_weight_change: number;
  odds_win: number;
  popularity: number;
  horses?: { name: string; name_jp: string; sex: string; birth_year: number };
  jockeys?: { name: string; name_jp: string };
  results?: {
    finish_pos: number;
    margin: string;
    time_secs: number;
    last_3f_secs: number;
    corner_positions: string;
    running_style: string;
  };
}

export interface Prediction {
  id: number;
  race_id: number;
  entry_id: number;
  win_prob: number;
  place_prob: number;
  top3_prob: number;
  model_version: string;
  fair_odds?: number;
  edge?: number;
  ability_rating?: number;
  condition_fit?: number;
  bounce_risk?: boolean;
  decision?: string;
  entries?: Entry;
}

export interface ValueBet {
  race_id: number;
  entry_id: number;
  bet_type: string;
  model_prob: number;
  market_prob: number;
  ev: number;
  kelly_fraction: number;
  recommended_stake: number;
}

export interface BankrollSummary {
  balance: number;
  today_pnl: number;
  today_staked: number;
  peak_balance: number;
  drawdown_pct: number;
  total_bets: number;
  winning_bets: number;
  win_rate: number;
  roi_pct: number;
  total_staked: number;
  total_profit: number;
  history: { date: string; running_balance: number; profit: number; stake: number }[];
}

export interface BacktestResult {
  total_bets: number;
  wins: number;
  win_rate: number;
  total_staked: number;
  total_profit: number;
  roi_pct: number;
  balance_curve: { date: string; balance: number; profit: number }[];
  roi_by_grade: Record<string, { staked: number; profit: number; count: number; roi: number }>;
  bets: unknown[];
}

export interface Analysis {
  id: number;
  race_id: number;
  analysis: {
    race_summary?: string;
    pace_forecast?: { scenario: string; front_runners: string[]; likely_tempo: string };
    horses_to_watch?: { name: string; reasoning: string }[];
    value_plays?: { name: string; model_prob: number; market_prob: number; ev: number; odds: number; reasoning: string }[];
    risk_factors?: string[];
    recommended_strategy?: string;
  };
  language: string;
}

// --- API Functions ---

export const api = {
  // Races
  getRaces: (params?: { upcoming?: boolean; limit?: number; offset?: number }) => {
    const qs = new URLSearchParams();
    if (params?.upcoming) qs.set("upcoming", "true");
    if (params?.limit) qs.set("limit", String(params.limit));
    if (params?.offset) qs.set("offset", String(params.offset));
    return fetchAPI<{ races: Race[]; count: number }>(`/api/races?${qs}`);
  },

  getRace: (raceId: number) =>
    fetchAPI<{ race: Race; entries: Entry[]; predictions: Prediction[]; value_bets: ValueBet[] }>(
      `/api/races/${raceId}`
    ),

  // Predictions
  getPredictions: (raceId: number) =>
    fetchAPI<{ predictions: Prediction[]; value_bets: ValueBet[] }>(
      `/api/predictions/${raceId}`
    ),

  // Analysis
  getAnalysis: (raceId: number, language = "en") =>
    fetchAPI<{ analysis: Analysis | null }>(`/api/analysis/${raceId}?language=${language}`),

  // Bankroll
  getBankroll: () => fetchAPI<BankrollSummary>("/api/bankroll"),

  getBankrollLog: (limit = 20, offset = 0) =>
    fetchAPI<{ log: unknown[]; count: number }>(`/api/bankroll/log?limit=${limit}&offset=${offset}`),

  // Exotic tickets
  getExoticTickets: (raceId: number, betType?: string) => {
    const qs = betType ? `?bet_type=${betType}` : "";
    return fetchAPI<{ tickets: unknown[] }>(`/api/exotic/${raceId}${qs}`);
  },

  // Backtest
  getBacktest: (params?: { model_version?: string; grade?: string; course?: string }) => {
    const qs = new URLSearchParams();
    if (params?.model_version) qs.set("model_version", params.model_version);
    if (params?.grade) qs.set("grade", params.grade);
    if (params?.course) qs.set("course", params.course);
    return fetchAPI<BacktestResult>(`/api/backtest?${qs}`);
  },

  // Odds
  getOdds: (raceId: number, betType = "win") =>
    fetchAPI<{ odds: unknown[] }>(`/api/odds/${raceId}?bet_type=${betType}`),

  // Automation / Refresh
  scrapeDate: (dateStr: string) => 
    fetchAPI<{ status: string; date: string; races_scraped: number }>(`/api/scraper/date/${dateStr}`, { method: 'POST' }),

  refreshRace: (raceId: number) =>
    fetchAPI<{ status: string; race_id: number; entries_predicted: number }>(`/api/races/${raceId}/refresh`, { method: 'POST' }),

  // Health
  health: () => fetchAPI<{ status: string }>("/api/health"),
};
