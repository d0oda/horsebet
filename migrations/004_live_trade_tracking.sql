-- ==========================================
-- UmaEdge: Sprint 7.5 — Live Trade Tracking
-- ==========================================

CREATE TABLE IF NOT EXISTS horsebet.live_trades (
    id SERIAL PRIMARY KEY,
    race_id INTEGER NOT NULL REFERENCES horsebet.races(id) ON DELETE CASCADE,
    entry_id INTEGER NOT NULL REFERENCES horsebet.entries(id) ON DELETE CASCADE,
    horse_name TEXT,
    date DATE NOT NULL,
    bet_type VARCHAR(20) NOT NULL DEFAULT 'win',
    model_prob REAL NOT NULL,
    market_prob REAL NOT NULL,
    ev REAL NOT NULL,
    odds REAL NOT NULL,
    stake INTEGER NOT NULL,           -- yen
    status VARCHAR(20) NOT NULL DEFAULT 'pending',  -- pending, won, lost, cancelled
    finish_pos INTEGER,
    payout INTEGER DEFAULT 0,
    profit INTEGER GENERATED ALWAYS AS (payout - stake) STORED,
    kelly_fraction REAL,
    model_version TEXT,
    confirmed_at TIMESTAMPTZ,         -- when user confirmed the bet
    reconciled_at TIMESTAMPTZ,        -- when result was checked
    created_at TIMESTAMPTZ DEFAULT now()
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_live_trades_race ON horsebet.live_trades(race_id);
CREATE INDEX IF NOT EXISTS idx_live_trades_date ON horsebet.live_trades(date);
CREATE INDEX IF NOT EXISTS idx_live_trades_status ON horsebet.live_trades(status);

-- Daily loss tracking view
CREATE OR REPLACE VIEW horsebet.daily_live_summary AS
SELECT
    date,
    COUNT(*) AS total_bets,
    SUM(stake) AS total_staked,
    SUM(payout) AS total_payout,
    SUM(payout - stake) AS total_profit,
    SUM(CASE WHEN status = 'won' THEN 1 ELSE 0 END) AS wins,
    SUM(CASE WHEN status = 'pending' THEN 1 ELSE 0 END) AS pending
FROM horsebet.live_trades
GROUP BY date
ORDER BY date DESC;
