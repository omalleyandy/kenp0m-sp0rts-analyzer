-- KenPom Raw Player Stats Schema
-- Stores player statistics scraped from kenpom.com/playerstats.php
-- Designed for machine learning (XGBoost) and advanced analytics

-- ============================================================================
-- KENPOM PLAYER STATS (Raw Data)
-- Direct mapping from scraped data with all available statistics
-- ============================================================================

CREATE TABLE IF NOT EXISTS kenpom_player_stats (
    -- Primary Key
    player_stat_id INTEGER PRIMARY KEY AUTOINCREMENT,

    -- Identity
    player_name TEXT NOT NULL,
    team_name TEXT NOT NULL,
    conference TEXT,  -- Not available from player stats page
    position TEXT,    -- Not available from player stats page

    -- Physical Attributes
    height TEXT,              -- Format: "6-11"
    height_inches INTEGER,    -- Converted to total inches
    weight_lbs INTEGER,       -- Weight in pounds
    class_year TEXT,          -- "Fr", "So", "Jr", "Sr", "GR"

    -- Season Context
    season INTEGER NOT NULL,
    rank_overall INTEGER,     -- Player's rank in ORtg leaderboard

    -- Tempo-Free Offensive Stats (per 100 possessions)
    offensive_rating REAL,    -- ORtg: Points produced per 100 possessions
    pct_possessions REAL,     -- %Poss: % of team possessions used

    -- Shooting Efficiency
    efg_pct REAL,            -- eFG%: Effective Field Goal %
    ts_pct REAL,             -- TS%: True Shooting %
    three_pt_rate REAL,      -- %Shots: % of FGA that are 3-pointers
    ft_rate REAL,            -- FTRate: Free throw attempts per FGA

    -- Per-Game Stats (not directly available, placeholder for future)
    ppg REAL,                -- Points per game
    rpg REAL,                -- Rebounds per game
    apg REAL,                -- Assists per game

    -- Advanced Metrics
    assist_rate REAL,        -- ARate: % of teammate FGs assisted while on court
    turnover_rate REAL,      -- TO%: Turnovers per 100 plays
    offensive_reb_pct REAL,  -- OR%: % of available offensive rebounds grabbed
    defensive_reb_pct REAL,  -- DR%: % of available defensive rebounds grabbed

    -- Fouling & Minutes
    fouls_committed_per_40 REAL,  -- FC/40: Fouls committed per 40 minutes
    minutes_pct REAL,             -- %Min: % of team minutes played

    -- Scraped Metadata
    scraped_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    data_source TEXT DEFAULT 'kenpom.com/playerstats.php',

    -- Constraints
    UNIQUE (player_name, team_name, season)
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_kenpom_player_season ON kenpom_player_stats(season, player_name);
CREATE INDEX IF NOT EXISTS idx_kenpom_player_team ON kenpom_player_stats(team_name, season);
CREATE INDEX IF NOT EXISTS idx_kenpom_player_ortg ON kenpom_player_stats(offensive_rating DESC);
CREATE INDEX IF NOT EXISTS idx_kenpom_player_class ON kenpom_player_stats(class_year, season);

-- ============================================================================
-- KENPOM PLAYER STATS HISTORY
-- Track changes over time (weekly snapshots during season)
-- ============================================================================

CREATE TABLE IF NOT EXISTS kenpom_player_stats_history (
    history_id INTEGER PRIMARY KEY AUTOINCREMENT,
    player_stat_id INTEGER NOT NULL,
    snapshot_date DATE NOT NULL,

    -- All stats from main table (denormalized for ML training)
    player_name TEXT NOT NULL,
    team_name TEXT NOT NULL,
    season INTEGER NOT NULL,

    offensive_rating REAL,
    pct_possessions REAL,
    efg_pct REAL,
    ts_pct REAL,
    three_pt_rate REAL,
    ft_rate REAL,
    assist_rate REAL,
    turnover_rate REAL,
    offensive_reb_pct REAL,
    defensive_reb_pct REAL,
    fouls_committed_per_40 REAL,
    minutes_pct REAL,

    FOREIGN KEY (player_stat_id) REFERENCES kenpom_player_stats(player_stat_id),
    UNIQUE (player_stat_id, snapshot_date)
);

CREATE INDEX IF NOT EXISTS idx_player_history_date ON kenpom_player_stats_history(snapshot_date DESC);
CREATE INDEX IF NOT EXISTS idx_player_history_player ON kenpom_player_stats_history(player_stat_id, snapshot_date);

-- ============================================================================
-- VIEWS FOR ANALYSIS
-- ============================================================================

-- Top 50 Players by Offensive Rating
CREATE VIEW IF NOT EXISTS v_top_players_ortg AS
SELECT
    rank_overall,
    player_name,
    team_name,
    class_year,
    height,
    offensive_rating,
    pct_possessions,
    efg_pct,
    ts_pct,
    minutes_pct
FROM kenpom_player_stats
WHERE season = (SELECT MAX(season) FROM kenpom_player_stats)
ORDER BY offensive_rating DESC
LIMIT 50;

-- Freshmen Standouts (High impact first-year players)
CREATE VIEW IF NOT EXISTS v_freshman_standouts AS
SELECT
    player_name,
    team_name,
    offensive_rating,
    pct_possessions,
    efg_pct,
    minutes_pct
FROM kenpom_player_stats
WHERE class_year = 'Fr'
  AND season = (SELECT MAX(season) FROM kenpom_player_stats)
  AND offensive_rating >= 110.0  -- Elite threshold
ORDER BY offensive_rating DESC;

-- High Usage Players (Primary options)
CREATE VIEW IF NOT EXISTS v_high_usage_players AS
SELECT
    player_name,
    team_name,
    class_year,
    offensive_rating,
    pct_possessions,
    minutes_pct,
    ROUND(offensive_rating * pct_possessions / 100, 2) as impact_score
FROM kenpom_player_stats
WHERE season = (SELECT MAX(season) FROM kenpom_player_stats)
  AND pct_possessions >= 25.0  -- High usage threshold
ORDER BY impact_score DESC
LIMIT 100;

-- Elite Efficiency (Top shooters)
CREATE VIEW IF NOT EXISTS v_elite_efficiency AS
SELECT
    player_name,
    team_name,
    efg_pct,
    ts_pct,
    three_pt_rate,
    ft_rate,
    offensive_rating
FROM kenpom_player_stats
WHERE season = (SELECT MAX(season) FROM kenpom_player_stats)
  AND efg_pct >= 60.0  -- Elite efficiency
ORDER BY ts_pct DESC;
