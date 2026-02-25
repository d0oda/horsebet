-- ==========================================
-- UmaEdge: Horse Racing Intelligence Schema
-- ==========================================

CREATE SCHEMA IF NOT EXISTS horsebet;

-- ------------------------------------------
-- Core Entities
-- ------------------------------------------

CREATE TABLE horsebet.trainers (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    name_jp TEXT,
    stable TEXT,
    win_rate REAL,
    place_rate REAL,
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE horsebet.jockeys (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    name_jp TEXT,
    win_rate REAL,
    place_rate REAL,
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE horsebet.horses (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    name_jp TEXT,
    sex VARCHAR(10),
    birth_year INTEGER,
    sire_id INTEGER REFERENCES horsebet.horses(id),
    dam_id INTEGER REFERENCES horsebet.horses(id),
    broodmare_sire_id INTEGER REFERENCES horsebet.horses(id),
    trainer_id INTEGER REFERENCES horsebet.trainers(id),
    owner TEXT,
    netkeiba_id TEXT UNIQUE,
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE horsebet.courses (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    name_jp TEXT,
    surface VARCHAR(10),           -- 'turf', 'dirt'
    direction VARCHAR(10),         -- 'right', 'left', 'straight'
    inner_outer VARCHAR(10),       -- 'inner', 'outer', NULL
    location TEXT,
    created_at TIMESTAMPTZ DEFAULT now()
);

-- ------------------------------------------
-- Race Data
-- ------------------------------------------

CREATE TABLE horsebet.races (
    id SERIAL PRIMARY KEY,
    netkeiba_id TEXT UNIQUE,
    date DATE NOT NULL,
    course_id INTEGER REFERENCES horsebet.courses(id),
    race_number INTEGER,
    distance INTEGER NOT NULL,     -- metres
    surface VARCHAR(10),           -- 'turf', 'dirt'
    going VARCHAR(20),             -- '良', '稍重', '重', '不良'
    class VARCHAR(50),             -- e.g. 'G1', 'G2', 'G3', 'OP', '3勝', '2勝', '1勝', '未勝利', '新馬'
    grade VARCHAR(10),             -- 'G1', 'G2', 'G3', 'L', NULL
    race_name TEXT,
    race_name_jp TEXT,
    weather VARCHAR(20),
    field_size INTEGER,
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE horsebet.entries (
    id SERIAL PRIMARY KEY,
    race_id INTEGER NOT NULL REFERENCES horsebet.races(id) ON DELETE CASCADE,
    horse_id INTEGER NOT NULL REFERENCES horsebet.horses(id),
    jockey_id INTEGER REFERENCES horsebet.jockeys(id),
    draw INTEGER,                  -- 枠番 / gate number
    post_position INTEGER,         -- 馬番
    weight_carried REAL,           -- 斤量 (kg)
    horse_weight INTEGER,          -- 馬体重 (kg)
    horse_weight_change INTEGER,   -- 体重増減
    odds_win REAL,                 -- final win odds
    popularity INTEGER,            -- 人気 (popularity rank)
    created_at TIMESTAMPTZ DEFAULT now(),
    UNIQUE(race_id, horse_id)
);

CREATE TABLE horsebet.results (
    id SERIAL PRIMARY KEY,
    entry_id INTEGER NOT NULL REFERENCES horsebet.entries(id) ON DELETE CASCADE UNIQUE,
    finish_pos INTEGER,            -- 着順 (NULL = DNF/DQ)
    margin TEXT,                   -- 着差 (e.g. '1/2', 'クビ', 'アタマ')
    time_secs REAL,                -- 走破タイム in seconds
    last_3f_secs REAL,             -- 上がり3F in seconds
    first_3f_secs REAL,            -- 前半3F (if available)
    corner_positions TEXT,         -- e.g. '3-3-2-1' (comma-separated corner positions)
    running_style VARCHAR(10),     -- 逃/先/差/追
    created_at TIMESTAMPTZ DEFAULT now()
);

-- ------------------------------------------
-- Odds Time-Series
-- ------------------------------------------

CREATE TABLE horsebet.odds_snapshots (
    id SERIAL PRIMARY KEY,
    race_id INTEGER NOT NULL REFERENCES horsebet.races(id) ON DELETE CASCADE,
    captured_at TIMESTAMPTZ NOT NULL,
    bet_type VARCHAR(20) NOT NULL, -- 'win', 'place', 'exacta', 'trifecta', 'trio', 'wide'
    combination TEXT NOT NULL,     -- e.g. '5' for win, '5-3' for exacta, '5-3-1' for trifecta
    odds_value REAL NOT NULL,
    pool_size BIGINT,              -- total pool in yen (if available)
    created_at TIMESTAMPTZ DEFAULT now()
);

-- ------------------------------------------
-- Model Outputs
-- ------------------------------------------

CREATE TABLE horsebet.predictions (
    id SERIAL PRIMARY KEY,
    race_id INTEGER NOT NULL REFERENCES horsebet.races(id) ON DELETE CASCADE,
    entry_id INTEGER NOT NULL REFERENCES horsebet.entries(id) ON DELETE CASCADE,
    model_version TEXT NOT NULL,
    win_prob REAL NOT NULL,
    place_prob REAL,
    top3_prob REAL,
    shap_values JSONB,             -- feature importance for this prediction
    created_at TIMESTAMPTZ DEFAULT now(),
    UNIQUE(race_id, entry_id, model_version)
);

CREATE TABLE horsebet.value_bets (
    id SERIAL PRIMARY KEY,
    race_id INTEGER NOT NULL REFERENCES horsebet.races(id) ON DELETE CASCADE,
    entry_id INTEGER REFERENCES horsebet.entries(id) ON DELETE CASCADE,
    bet_type VARCHAR(20) NOT NULL, -- 'win', 'place', 'exacta', etc.
    combination TEXT,              -- for exotic bets
    model_prob REAL NOT NULL,
    market_prob REAL NOT NULL,
    ev REAL NOT NULL,              -- expected value = model_prob / market_prob - 1
    kelly_fraction REAL,
    recommended_stake INTEGER,     -- in yen
    model_version TEXT,
    created_at TIMESTAMPTZ DEFAULT now()
);

-- ------------------------------------------
-- Bankroll Tracking
-- ------------------------------------------

CREATE TABLE horsebet.bankroll_log (
    id SERIAL PRIMARY KEY,
    user_id UUID REFERENCES auth.users(id),
    date DATE NOT NULL,
    race_id INTEGER REFERENCES horsebet.races(id),
    bet_type VARCHAR(20),
    combination TEXT,
    stake INTEGER NOT NULL,        -- in yen
    odds_at_bet REAL,
    payout INTEGER DEFAULT 0,
    profit INTEGER GENERATED ALWAYS AS (payout - stake) STORED,
    running_balance INTEGER,
    notes TEXT,
    created_at TIMESTAMPTZ DEFAULT now()
);

-- Enable RLS on bankroll (private data)
ALTER TABLE horsebet.bankroll_log ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can only see own bankroll"
    ON horsebet.bankroll_log
    FOR ALL
    USING (auth.uid() = user_id);

-- ------------------------------------------
-- Indexes for performance
-- ------------------------------------------

CREATE INDEX idx_races_date ON horsebet.races(date);
CREATE INDEX idx_races_course_date ON horsebet.races(course_id, date);
CREATE INDEX idx_entries_race ON horsebet.entries(race_id);
CREATE INDEX idx_entries_horse ON horsebet.entries(horse_id);
CREATE INDEX idx_entries_race_horse ON horsebet.entries(race_id, horse_id);
CREATE INDEX idx_results_entry ON horsebet.results(entry_id);
CREATE INDEX idx_odds_race_time ON horsebet.odds_snapshots(race_id, captured_at);
CREATE INDEX idx_predictions_race ON horsebet.predictions(race_id);
CREATE INDEX idx_value_bets_race ON horsebet.value_bets(race_id);
CREATE INDEX idx_horses_netkeiba ON horsebet.horses(netkeiba_id);
CREATE INDEX idx_bankroll_user ON horsebet.bankroll_log(user_id, date);
