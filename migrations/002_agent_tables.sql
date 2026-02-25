-- ==========================================
-- UmaEdge: Agent Layer Tables
-- Phase 3 — Race Analyst, Bankroll, Exotic Bets
-- ==========================================

-- ------------------------------------------
-- Agent Analyses (LLM-generated race previews)
-- ------------------------------------------

CREATE TABLE IF NOT EXISTS horsebet.agent_analyses (
    id SERIAL PRIMARY KEY,
    race_id INTEGER NOT NULL REFERENCES horsebet.races(id) ON DELETE CASCADE,
    model_version TEXT,
    language VARCHAR(5) NOT NULL DEFAULT 'en',   -- 'en' or 'ja'
    analysis JSONB NOT NULL,                      -- structured analysis object
    raw_text TEXT,                                 -- plain-text version for display
    created_at TIMESTAMPTZ DEFAULT now(),
    UNIQUE(race_id, model_version, language)
);

CREATE INDEX idx_analyses_race ON horsebet.agent_analyses(race_id);

-- ------------------------------------------
-- Exotic Bet Tickets (optimised ticket sets)
-- ------------------------------------------

CREATE TABLE IF NOT EXISTS horsebet.exotic_tickets (
    id SERIAL PRIMARY KEY,
    race_id INTEGER NOT NULL REFERENCES horsebet.races(id) ON DELETE CASCADE,
    bet_type VARCHAR(20) NOT NULL,                -- 'trio', 'trifecta', 'wide'
    budget INTEGER NOT NULL,                       -- budget in yen
    combinations JSONB NOT NULL,                   -- array of {combo, odds, model_prob, cost, ev}
    total_cost INTEGER NOT NULL,
    expected_profit REAL,
    expected_roi REAL,
    model_version TEXT,
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX idx_exotic_race ON horsebet.exotic_tickets(race_id);
