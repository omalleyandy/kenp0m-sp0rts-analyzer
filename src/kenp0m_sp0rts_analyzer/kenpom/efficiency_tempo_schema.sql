-- KenPom Efficiency & Tempo Schema
-- Stores team efficiency and tempo statistics from kenpom.com/summary.php
-- Designed for historical analysis from 2015-present

-- ============================================================================
-- EFFICIENCY TEMPO TABLE
-- Comprehensive team statistics including offensive/defensive efficiency and tempo
-- ============================================================================

CREATE TABLE IF NOT EXISTS efficiency_tempo (
    -- Primary Key
    efficiency_tempo_id INTEGER PRIMARY KEY AUTOINCREMENT,

    -- Team Identity
    team_id INTEGER,              -- FK to teams table
    team_name TEXT NOT NULL,
    conference TEXT,

    -- Season Context
    season INTEGER NOT NULL,
    snapshot_date DATE NOT NULL,  -- Date of data collection

    -- Rankings
    rank_overall INTEGER,         -- Overall KenPom rank

    -- Adjusted Efficiency Margin (AdjEM)
    adj_em REAL,                  -- Adjusted Efficiency Margin
    rank_adj_em INTEGER,

    -- Offensive Efficiency (AdjO)
    adj_oe REAL,                  -- Adjusted Offensive Efficiency (points per 100 poss)
    rank_adj_oe INTEGER,

    -- Defensive Efficiency (AdjD)
    adj_de REAL,                  -- Adjusted Defensive Efficiency (points allowed per 100 poss)
    rank_adj_de INTEGER,

    -- Tempo (AdjT)
    adj_tempo REAL,               -- Adjusted Tempo (possessions per 40 min)
    rank_adj_tempo INTEGER,

    -- Strength of Schedule (SOS)
    sos_adj_em REAL,              -- Strength of schedule (adj EM)
    rank_sos INTEGER,
    sos_adj_oe REAL,              -- SOS offensive component
    rank_sos_oe INTEGER,
    sos_adj_de REAL,              -- SOS defensive component
    rank_sos_de INTEGER,

    -- Non-Conference Strength of Schedule (NCSOS)
    nc_sos_adj_em REAL,           -- Non-conference SOS
    rank_nc_sos INTEGER,

    -- Luck & Consistency
    luck REAL,                    -- Luck rating (deviation from expected W/L)
    rank_luck INTEGER,
    consistency REAL,             -- Consistency rating
    rank_consistency INTEGER,

    -- Record
    wins INTEGER,
    losses INTEGER,
    win_pct REAL,

    -- Four Factors - Offense
    efg_pct_offense REAL,         -- Effective FG% (offense)
    rank_efg_offense INTEGER,
    to_pct_offense REAL,          -- Turnover % (offense)
    rank_to_offense INTEGER,
    or_pct_offense REAL,          -- Offensive Rebound %
    rank_or_offense INTEGER,
    ft_rate_offense REAL,         -- Free Throw Rate (offense)
    rank_ft_rate_offense INTEGER,

    -- Four Factors - Defense
    efg_pct_defense REAL,         -- Effective FG% allowed (defense)
    rank_efg_defense INTEGER,
    to_pct_defense REAL,          -- Turnover % forced (defense)
    rank_to_defense INTEGER,
    or_pct_defense REAL,          -- Offensive Rebound % allowed (defense)
    rank_or_defense INTEGER,
    ft_rate_defense REAL,         -- Free Throw Rate allowed (defense)
    rank_ft_rate_defense INTEGER,

    -- Three Point Shooting
    three_pt_pct REAL,            -- 3-point shooting %
    rank_three_pt_pct INTEGER,
    three_pt_pct_defense REAL,    -- 3-point defense %
    rank_three_pt_defense INTEGER,
    three_pt_rate REAL,           -- % of FGA that are 3s
    rank_three_pt_rate INTEGER,

    -- Two Point Shooting
    two_pt_pct REAL,              -- 2-point shooting %
    rank_two_pt_pct INTEGER,
    two_pt_pct_defense REAL,      -- 2-point defense %
    rank_two_pt_defense INTEGER,

    -- Free Throws
    ft_pct REAL,                  -- Free throw %
    rank_ft_pct INTEGER,

    -- Blocks & Steals
    block_pct REAL,               -- Block %
    rank_block_pct INTEGER,
    steal_pct REAL,               -- Steal %
    rank_steal_pct INTEGER,

    -- Experience & Size
    avg_experience REAL,          -- Average years of experience
    rank_experience INTEGER,
    avg_height REAL,              -- Average height (inches)
    rank_height INTEGER,
    effective_height REAL,        -- Effective height (weighted by minutes)
    rank_effective_height INTEGER,

    -- Bench Performance
    bench_minutes_pct REAL,       -- % of minutes from bench
    rank_bench_minutes INTEGER,

    -- Metadata
    scraped_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    data_source TEXT DEFAULT 'kenpom.com/summary.php',

    -- Constraints
    FOREIGN KEY (team_id) REFERENCES teams(team_id),
    UNIQUE (team_name, season, snapshot_date)
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_efficiency_tempo_season ON efficiency_tempo(season DESC);
CREATE INDEX IF NOT EXISTS idx_efficiency_tempo_team ON efficiency_tempo(team_id, season);
CREATE INDEX IF NOT EXISTS idx_efficiency_tempo_rank ON efficiency_tempo(rank_overall, season);
CREATE INDEX IF NOT EXISTS idx_efficiency_tempo_date ON efficiency_tempo(snapshot_date DESC);
CREATE INDEX IF NOT EXISTS idx_efficiency_tempo_adj_em ON efficiency_tempo(adj_em DESC);

-- ============================================================================
-- VIEWS FOR ANALYSIS
-- ============================================================================

-- Current Season Top 25 by Adjusted Efficiency Margin
CREATE VIEW IF NOT EXISTS v_top_25_current AS
SELECT
    rank_overall,
    team_name,
    conference,
    adj_em,
    adj_oe,
    adj_de,
    adj_tempo,
    wins,
    losses
FROM efficiency_tempo
WHERE season = (SELECT MAX(season) FROM efficiency_tempo)
  AND snapshot_date = (
      SELECT MAX(snapshot_date)
      FROM efficiency_tempo
      WHERE season = (SELECT MAX(season) FROM efficiency_tempo)
  )
ORDER BY rank_overall ASC
LIMIT 25;

-- Historical Performance by Season
CREATE VIEW IF NOT EXISTS v_team_historical_performance AS
SELECT
    team_name,
    season,
    rank_overall,
    adj_em,
    adj_oe,
    adj_de,
    wins,
    losses,
    ROUND(CAST(wins AS REAL) / (wins + losses), 3) as win_pct
FROM efficiency_tempo
WHERE snapshot_date = (
    SELECT MAX(snapshot_date)
    FROM efficiency_tempo e2
    WHERE e2.season = efficiency_tempo.season
)
ORDER BY team_name, season DESC;

-- Elite Offenses (AdjO >= 120)
CREATE VIEW IF NOT EXISTS v_elite_offenses AS
SELECT
    season,
    team_name,
    rank_adj_oe,
    adj_oe,
    efg_pct_offense,
    to_pct_offense,
    or_pct_offense,
    adj_tempo
FROM efficiency_tempo
WHERE adj_oe >= 120.0
  AND snapshot_date = (
      SELECT MAX(snapshot_date)
      FROM efficiency_tempo e2
      WHERE e2.season = efficiency_tempo.season
  )
ORDER BY season DESC, adj_oe DESC;

-- Elite Defenses (AdjD <= 90)
CREATE VIEW IF NOT EXISTS v_elite_defenses AS
SELECT
    season,
    team_name,
    rank_adj_de,
    adj_de,
    efg_pct_defense,
    to_pct_defense,
    block_pct,
    steal_pct
FROM efficiency_tempo
WHERE adj_de <= 90.0
  AND snapshot_date = (
      SELECT MAX(snapshot_date)
      FROM efficiency_tempo e2
      WHERE e2.season = efficiency_tempo.season
  )
ORDER BY season DESC, adj_de ASC;

-- Tempo Extremes
CREATE VIEW IF NOT EXISTS v_tempo_extremes AS
SELECT
    'Fast' as tempo_type,
    season,
    team_name,
    adj_tempo,
    rank_adj_tempo,
    adj_oe,
    adj_de
FROM efficiency_tempo
WHERE adj_tempo >= 75.0
  AND snapshot_date = (
      SELECT MAX(snapshot_date)
      FROM efficiency_tempo e2
      WHERE e2.season = efficiency_tempo.season
  )
UNION ALL
SELECT
    'Slow' as tempo_type,
    season,
    team_name,
    adj_tempo,
    rank_adj_tempo,
    adj_oe,
    adj_de
FROM efficiency_tempo
WHERE adj_tempo <= 60.0
  AND snapshot_date = (
      SELECT MAX(snapshot_date)
      FROM efficiency_tempo e2
      WHERE e2.season = efficiency_tempo.season
  )
ORDER BY season DESC, adj_tempo DESC;

-- Conference Strength by Season
CREATE VIEW IF NOT EXISTS v_conference_strength AS
SELECT
    season,
    conference,
    COUNT(*) as team_count,
    ROUND(AVG(adj_em), 2) as avg_adj_em,
    ROUND(AVG(adj_oe), 2) as avg_adj_oe,
    ROUND(AVG(adj_de), 2) as avg_adj_de,
    ROUND(AVG(rank_overall), 1) as avg_rank
FROM efficiency_tempo
WHERE conference IS NOT NULL
  AND conference != ''
  AND snapshot_date = (
      SELECT MAX(snapshot_date)
      FROM efficiency_tempo e2
      WHERE e2.season = efficiency_tempo.season
  )
GROUP BY season, conference
HAVING team_count >= 5  -- Only conferences with 5+ teams
ORDER BY season DESC, avg_adj_em DESC;
