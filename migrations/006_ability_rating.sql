-- ==========================================
-- UmaEdge: Ability Rating System
-- ==========================================
-- Adds persistent EWMA ability ratings per horse,
-- and a history table for tracking rating evolution.

-- Add ability_rating column to horses table
ALTER TABLE horsebet.horses ADD COLUMN IF NOT EXISTS ability_rating REAL;

-- History table for tracking rating evolution per run
CREATE TABLE IF NOT EXISTS horsebet.horse_rating_history (
    id SERIAL PRIMARY KEY,
    horse_id INTEGER NOT NULL REFERENCES horsebet.horses(id),
    race_id INTEGER NOT NULL REFERENCES horsebet.races(id),
    race_date DATE NOT NULL,
    rating_before REAL,         -- rating before this race (NULL for first run)
    race_score REAL NOT NULL,   -- raw ability score for this single run
    rating_after REAL NOT NULL, -- EWMA-updated rating after this race
    components JSONB,           -- breakdown: {base_speed, class_adj, weight_adj, margin_adj, going_adj}
    created_at TIMESTAMPTZ DEFAULT now(),
    UNIQUE(horse_id, race_id)
);

CREATE INDEX IF NOT EXISTS idx_rating_history_horse_date
    ON horsebet.horse_rating_history(horse_id, race_date);
CREATE INDEX IF NOT EXISTS idx_rating_history_race
    ON horsebet.horse_rating_history(race_id);
CREATE INDEX IF NOT EXISTS idx_rating_history_date
    ON horsebet.horse_rating_history(race_date);
