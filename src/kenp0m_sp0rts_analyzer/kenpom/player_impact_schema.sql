-- Player Impact & Injury Tracking Schema
-- Extends KenPom database for player-level analysis

-- ============================================================================
-- PLAYERS TABLE
-- Core player information and advanced metrics
-- ============================================================================

CREATE TABLE IF NOT EXISTS players (
    player_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    team_id INTEGER NOT NULL,
    jersey_number INTEGER,
    position TEXT CHECK(position IN ('PG', 'SG', 'SF', 'PF', 'C', 'G', 'F')),
    class_year TEXT CHECK(class_year IN ('FR', 'SO', 'JR', 'SR', 'GR')),
    height_inches INTEGER,
    weight_lbs INTEGER,

    -- Season Context
    season INTEGER NOT NULL,
    games_played INTEGER DEFAULT 0,
    games_started INTEGER DEFAULT 0,
    minutes_per_game REAL,

    -- Basic Statistics
    ppg REAL,  -- Points per game
    rpg REAL,  -- Rebounds per game
    apg REAL,  -- Assists per game
    spg REAL,  -- Steals per game
    bpg REAL,  -- Blocks per game
    topg REAL, -- Turnovers per game
    fpg REAL,  -- Fouls per game

    -- Shooting Percentages
    fg_pct REAL CHECK(fg_pct BETWEEN 0 AND 1),
    three_pt_pct REAL CHECK(three_pt_pct BETWEEN 0 AND 1),
    ft_pct REAL CHECK(ft_pct BETWEEN 0 AND 1),
    ts_pct REAL CHECK(ts_pct BETWEEN 0 AND 1),  -- True Shooting %
    efg_pct REAL CHECK(efg_pct BETWEEN 0 AND 1), -- Effective FG%

    -- Advanced Metrics
    usage_rate REAL CHECK(usage_rate BETWEEN 0 AND 100),
    offensive_rating REAL,
    defensive_rating REAL,
    net_rating REAL,  -- ORtg - DRtg

    -- Box Plus/Minus (points per 100 poss above average)
    bpm REAL,   -- Total BPM
    obpm REAL,  -- Offensive BPM
    dbpm REAL,  -- Defensive BPM

    win_shares REAL,  -- Wins contributed
    per REAL,  -- Player Efficiency Rating

    -- Usage & Pace
    pace REAL,  -- Possessions per 40 minutes
    assist_rate REAL,  -- % of teammate FGs assisted
    turnover_rate REAL,  -- TO per 100 plays
    rebound_rate REAL,  -- % of available rebounds

    -- Impact Tier (calculated)
    tier INTEGER CHECK(tier BETWEEN 1 AND 5),
    tier_justification TEXT,

    -- Calculated Impact Estimates
    spread_impact REAL,  -- Points impact on spread when OUT
    total_impact REAL,   -- Points impact on total
    win_prob_impact REAL CHECK(win_prob_impact BETWEEN 0 AND 1),
    confidence_score REAL CHECK(confidence_score BETWEEN 0 AND 100),

    -- Replacement Player
    replacement_player_id INTEGER,  -- Who plays when this player is out?
    replacement_quality_delta REAL,  -- Skill gap to replacement

    -- NBA Draft Projection (proxy for talent)
    draft_projection TEXT CHECK(
        draft_projection IN ('Lottery', '1st Round', '2nd Round', 'Undrafted', 'Unknown')
    ),
    draft_rank INTEGER,  -- Projected pick number

    -- Current Status
    is_active BOOLEAN DEFAULT 1,
    injury_status TEXT CHECK(
        injury_status IN ('Healthy', 'Questionable', 'Doubtful', 'Out', 'Day-to-Day')
    ),
    injury_type TEXT,  -- 'Ankle', 'Knee', 'Illness', etc.
    expected_return DATE,

    -- Metadata
    source_url TEXT,  -- Sports-Reference profile URL
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (team_id) REFERENCES teams(team_id),
    FOREIGN KEY (replacement_player_id) REFERENCES players(player_id),
    UNIQUE (name, team_id, season)
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_players_tier ON players(tier);
CREATE INDEX IF NOT EXISTS idx_players_team ON players(team_id, season);
CREATE INDEX IF NOT EXISTS idx_players_impact ON players(spread_impact DESC);
CREATE INDEX IF NOT EXISTS idx_players_status ON players(injury_status);
CREATE INDEX IF NOT EXISTS idx_players_season ON players(season, is_active);


-- ============================================================================
-- PLAYER GAME LOG
-- Individual game performance tracking
-- ============================================================================

CREATE TABLE IF NOT EXISTS player_game_log (
    log_id INTEGER PRIMARY KEY AUTOINCREMENT,
    player_id INTEGER NOT NULL,
    game_id INTEGER,  -- NULL if game not in our system yet
    team_id INTEGER NOT NULL,
    opponent_id INTEGER NOT NULL,
    game_date DATE NOT NULL,
    home_game BOOLEAN,

    -- Participation
    played BOOLEAN NOT NULL DEFAULT 1,
    dnp_reason TEXT CHECK(
        dnp_reason IN ('Injury', 'Suspension', 'Illness', 'Coach Decision', 'Personal', NULL)
    ),
    minutes_played INTEGER CHECK(minutes_played BETWEEN 0 AND 50),
    started BOOLEAN,

    -- Scoring
    points INTEGER CHECK(points >= 0),
    fgm INTEGER CHECK(fgm >= 0),
    fga INTEGER CHECK(fga >= 0),
    three_pm INTEGER CHECK(three_pm >= 0),
    three_pa INTEGER CHECK(three_pa >= 0),
    ftm INTEGER CHECK(ftm >= 0),
    fta INTEGER CHECK(fta >= 0),

    -- Other Stats
    rebounds INTEGER CHECK(rebounds >= 0),
    offensive_rebounds INTEGER CHECK(offensive_rebounds >= 0),
    defensive_rebounds INTEGER CHECK(defensive_rebounds >= 0),
    assists INTEGER CHECK(assists >= 0),
    steals INTEGER CHECK(steals >= 0),
    blocks INTEGER CHECK(blocks >= 0),
    turnovers INTEGER CHECK(turnovers >= 0),
    fouls INTEGER CHECK(fouls >= 0),

    -- Advanced
    plus_minus INTEGER,  -- Team's margin while player on court
    game_score REAL,  -- John Hollinger's Game Score metric

    -- Team Performance (for WITH/WITHOUT analysis)
    team_score INTEGER,
    opponent_score INTEGER,
    team_offensive_eff REAL,
    team_defensive_eff REAL,

    -- Metadata
    source_url TEXT,
    scraped_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (player_id) REFERENCES players(player_id),
    FOREIGN KEY (team_id) REFERENCES teams(team_id),
    FOREIGN KEY (opponent_id) REFERENCES teams(team_id),
    UNIQUE (player_id, game_date, opponent_id)
);

CREATE INDEX IF NOT EXISTS idx_game_log_player ON player_game_log(player_id);
CREATE INDEX IF NOT EXISTS idx_game_log_date ON player_game_log(game_date DESC);
CREATE INDEX IF NOT EXISTS idx_game_log_dnp ON player_game_log(player_id, played);


-- ============================================================================
-- TEAM PERFORMANCE SPLITS (WITH vs WITHOUT Player)
-- ============================================================================

CREATE TABLE IF NOT EXISTS team_performance_splits (
    split_id INTEGER PRIMARY KEY AUTOINCREMENT,
    team_id INTEGER NOT NULL,
    player_id INTEGER NOT NULL,
    season INTEGER NOT NULL,

    -- WITH Player Statistics
    games_with INTEGER DEFAULT 0,
    wins_with INTEGER DEFAULT 0,
    losses_with INTEGER DEFAULT 0,
    avg_margin_with REAL,
    avg_points_with REAL,
    avg_points_allowed_with REAL,
    avg_adj_em_with REAL,
    avg_offensive_eff_with REAL,
    avg_defensive_eff_with REAL,
    avg_tempo_with REAL,
    win_pct_with REAL,

    -- Opponent Context WITH
    avg_opponent_adj_em_with REAL,
    avg_opponent_rank_with REAL,

    -- WITHOUT Player Statistics
    games_without INTEGER DEFAULT 0,
    wins_without INTEGER DEFAULT 0,
    losses_without INTEGER DEFAULT 0,
    avg_margin_without REAL,
    avg_points_without REAL,
    avg_points_allowed_without REAL,
    avg_adj_em_without REAL,
    avg_offensive_eff_without REAL,
    avg_defensive_eff_without REAL,
    avg_tempo_without REAL,
    win_pct_without REAL,

    -- Opponent Context WITHOUT
    avg_opponent_adj_em_without REAL,
    avg_opponent_rank_without REAL,

    -- Calculated Impact (Raw)
    raw_spread_impact REAL,  -- margin_with - margin_without
    raw_total_impact REAL,   -- (points_with - points_without)
    raw_tempo_impact REAL,
    raw_win_prob_impact REAL,

    -- Opponent-Adjusted Impact
    adjusted_spread_impact REAL,
    adjusted_total_impact REAL,
    opponent_adjustment_factor REAL,

    -- Statistical Confidence
    sample_size_score REAL CHECK(sample_size_score BETWEEN 0 AND 100),
    impact_confidence TEXT CHECK(
        impact_confidence IN ('Very High', 'High', 'Medium', 'Low', 'Very Low')
    ),

    -- Regression to Mean
    regressed_spread_impact REAL,  -- Adjusted for small sample
    regression_weight REAL,  -- How much we trust this vs statistical model

    -- Metadata
    last_calculated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    calculation_method TEXT,  -- 'Empirical', 'Statistical', 'Hybrid'

    FOREIGN KEY (team_id) REFERENCES teams(team_id),
    FOREIGN KEY (player_id) REFERENCES players(player_id),
    UNIQUE (team_id, player_id, season)
);

CREATE INDEX IF NOT EXISTS idx_splits_team ON team_performance_splits(team_id, season);
CREATE INDEX IF NOT EXISTS idx_splits_player ON team_performance_splits(player_id);
CREATE INDEX IF NOT EXISTS idx_splits_impact ON team_performance_splits(adjusted_spread_impact DESC);


-- ============================================================================
-- INJURY ALERTS & MONITORING
-- ============================================================================

CREATE TABLE IF NOT EXISTS injury_alerts (
    alert_id INTEGER PRIMARY KEY AUTOINCREMENT,
    player_id INTEGER NOT NULL,
    team_id INTEGER NOT NULL,
    game_id INTEGER,  -- NULL if pre-game alert
    alert_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    game_date DATE,
    opponent_id INTEGER,

    -- Alert Source
    source_type TEXT CHECK(
        source_type IN ('Twitter', 'ESPN', 'Team Official', 'Beat Reporter', 'Other')
    ),
    source_account TEXT,  -- '@KUHoops', 'ESPN.com'
    source_url TEXT,
    source_reliability_score REAL CHECK(source_reliability_score BETWEEN 0 AND 100),

    -- Injury Details
    injury_type TEXT,  -- 'Ankle', 'Knee', 'Concussion', 'Illness', 'Suspension'
    injury_severity TEXT CHECK(
        injury_severity IN ('Minor', 'Moderate', 'Severe', 'Unknown')
    ),
    status TEXT CHECK(
        status IN ('Out', 'Doubtful', 'Questionable', 'Probable', 'Game-Time Decision')
    ),
    expected_return TEXT,  -- '1-2 weeks', 'Season', 'Next game'
    is_late_scratch BOOLEAN,  -- Within 2 hours of tip-off

    -- Player Context
    player_tier INTEGER,
    player_ppg REAL,
    player_bpm REAL,

    -- Impact Estimates
    estimated_spread_impact REAL,
    estimated_total_impact REAL,
    confidence_in_estimate REAL CHECK(confidence_in_estimate BETWEEN 0 AND 100),

    -- Betting Context (if available)
    spread_at_alert REAL,  -- Line when alert issued
    total_at_alert REAL,
    spread_30min_later REAL,  -- Track market reaction
    total_30min_later REAL,
    line_movement_spread REAL,  -- How much line moved
    line_movement_total REAL,

    -- Verification
    verified BOOLEAN DEFAULT 0,
    verified_at TIMESTAMP,
    verified_source TEXT,
    actual_status TEXT,  -- What actually happened

    -- Alert Action Taken
    alerted_user BOOLEAN DEFAULT 0,
    alert_method TEXT,  -- 'Email', 'SMS', 'Dashboard'
    user_response TEXT,  -- 'Placed bet', 'Ignored', 'Saved for later'

    FOREIGN KEY (player_id) REFERENCES players(player_id),
    FOREIGN KEY (team_id) REFERENCES teams(team_id),
    FOREIGN KEY (opponent_id) REFERENCES teams(team_id)
);

CREATE INDEX IF NOT EXISTS idx_alerts_time ON injury_alerts(alert_timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_alerts_player ON injury_alerts(player_id);
CREATE INDEX IF NOT EXISTS idx_alerts_game ON injury_alerts(game_date, team_id);
CREATE INDEX IF NOT EXISTS idx_alerts_verified ON injury_alerts(verified);
CREATE INDEX IF NOT EXISTS idx_alerts_late_scratch ON injury_alerts(is_late_scratch, verified);


-- ============================================================================
-- TWITTER/X BEAT REPORTERS
-- Tracking list for automated monitoring
-- ============================================================================

CREATE TABLE IF NOT EXISTS beat_reporters (
    reporter_id INTEGER PRIMARY KEY AUTOINCREMENT,
    twitter_handle TEXT NOT NULL UNIQUE,
    reporter_name TEXT,
    affiliated_team_id INTEGER,
    affiliated_conference TEXT,

    -- Reliability Metrics
    reliability_score REAL CHECK(reliability_score BETWEEN 0 AND 100) DEFAULT 50,
    false_positive_rate REAL,  -- % of alerts that were wrong
    avg_scoop_lead_time_minutes INTEGER,  -- How early they report vs others
    total_alerts_issued INTEGER DEFAULT 0,
    alerts_verified INTEGER DEFAULT 0,

    -- Monitoring
    is_active BOOLEAN DEFAULT 1,
    last_checked TIMESTAMP,
    check_frequency_minutes INTEGER DEFAULT 5,

    -- Source Classification
    source_tier TEXT CHECK(source_tier IN ('Tier 1', 'Tier 2', 'Tier 3')) DEFAULT 'Tier 2',

    FOREIGN KEY (affiliated_team_id) REFERENCES teams(team_id)
);

CREATE INDEX IF NOT EXISTS idx_reporters_active ON beat_reporters(is_active, affiliated_team_id);


-- ============================================================================
-- PLAYER MATCHUP HISTORY
-- Track individual player vs opponent performance
-- ============================================================================

CREATE TABLE IF NOT EXISTS player_matchup_history (
    matchup_id INTEGER PRIMARY KEY AUTOINCREMENT,
    player_id INTEGER NOT NULL,
    opponent_team_id INTEGER NOT NULL,
    season INTEGER NOT NULL,

    games_played INTEGER DEFAULT 0,
    avg_points REAL,
    avg_minutes REAL,
    avg_fg_pct REAL,
    avg_plus_minus REAL,

    -- Performance vs this opponent
    performance_rating TEXT CHECK(
        performance_rating IN ('Excellent', 'Good', 'Average', 'Below Average', 'Poor')
    ),

    FOREIGN KEY (player_id) REFERENCES players(player_id),
    FOREIGN KEY (opponent_team_id) REFERENCES teams(team_id),
    UNIQUE (player_id, opponent_team_id, season)
);


-- ============================================================================
-- VIEWS FOR COMMON QUERIES
-- ============================================================================

-- Top Players by Impact
CREATE VIEW IF NOT EXISTS v_top_impact_players AS
SELECT
    p.player_id,
    p.name,
    t.name as team_name,
    p.tier,
    p.spread_impact,
    p.ppg,
    p.bpm,
    p.win_shares,
    p.injury_status,
    CASE
        WHEN p.tier = 1 THEN 'Elite Game-Changer'
        WHEN p.tier = 2 THEN 'All-Conference Star'
        WHEN p.tier = 3 THEN 'Key Rotation'
        WHEN p.tier = 4 THEN 'Specialist'
        WHEN p.tier = 5 THEN 'Bench Depth'
    END as tier_description
FROM players p
JOIN teams t ON p.team_id = t.team_id
WHERE p.is_active = 1
  AND p.season = (SELECT MAX(season) FROM players)
ORDER BY p.spread_impact DESC;


-- Current Injury Report
CREATE VIEW IF NOT EXISTS v_current_injuries AS
SELECT
    p.name as player_name,
    t.name as team_name,
    p.injury_status,
    p.injury_type,
    p.tier,
    p.spread_impact,
    p.expected_return,
    p.last_updated
FROM players p
JOIN teams t ON p.team_id = t.team_id
WHERE p.injury_status IN ('Out', 'Doubtful', 'Questionable')
  AND p.is_active = 1
ORDER BY p.tier ASC, p.spread_impact DESC;


-- Recent Alerts Summary
CREATE VIEW IF NOT EXISTS v_recent_alerts AS
SELECT
    ia.alert_timestamp,
    p.name as player_name,
    t.name as team_name,
    ia.status,
    ia.injury_type,
    ia.estimated_spread_impact,
    ia.is_late_scratch,
    ia.verified,
    ia.source_account,
    ia.line_movement_spread
FROM injury_alerts ia
JOIN players p ON ia.player_id = p.player_id
JOIN teams t ON ia.team_id = t.team_id
WHERE ia.alert_timestamp >= datetime('now', '-7 days')
ORDER BY ia.alert_timestamp DESC;
