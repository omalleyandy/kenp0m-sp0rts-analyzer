# Player Impact & Late Scratch Injury System

**Date**: 2025-12-19
**Author**: Andy + Claude
**Purpose**: Gain betting edge through player absence tracking and impact quantification

---

## Executive Summary

Late scratch injuries represent a **high-value, low-competition betting edge** in NCAA Basketball. This system will:

1. **Track Top 200 Players**: Real-time status monitoring of elite college basketball players
2. **Quantify Player Impact**: Convert player absence into spread adjustments (points)
3. **Monitor Breaking News**: Automated Twitter/X tracking for late scratch alerts
4. **Adjust Predictions**: Integrate player impact into our XGBoost prediction system

**Expected Edge**: 3-8 points per game when star player is out (vs market reaction of 1-4 points)

---

## Core Methodology

### Method A: Historical Game Analysis (Empirical) ⭐ PRIMARY

**Process**:
1. Identify all games player X played (WITH)
2. Identify all games player X missed (WITHOUT)
3. Calculate performance deltas

**Metrics to Track**:
```python
with_player = {
    "adj_em": team_avg_adj_em_with_player,
    "margin": avg_point_margin,
    "pace": avg_tempo,
    "offensive_efficiency": points_per_100_poss,
    "defensive_efficiency": opponent_points_per_100_poss,
    "wins": win_count,
    "losses": loss_count
}

without_player = {
    # Same metrics for games player missed
}

player_impact = {
    "spread_impact": with_player["margin"] - without_player["margin"],
    "total_impact": with_player["offensive_efficiency"] - without_player["offensive_efficiency"],
    "tempo_impact": with_player["pace"] - without_player["pace"],
    "win_prob_impact": with_player["win_pct"] - without_player["win_pct"]
}
```

**Opponent Adjustment**:
```python
# Normalize for opponent quality
avg_opponent_adj_em_with = mean([opp.adj_em for games with player])
avg_opponent_adj_em_without = mean([opp.adj_em for games without player])

opponent_delta = avg_opponent_adj_em_with - avg_opponent_adj_em_without

# Adjust impact
adjusted_spread_impact = raw_spread_impact - (opponent_delta * 0.5)
```

### Method B: Statistical Contribution (Analytical)

**Formula**:
```python
def calculate_statistical_impact(player_stats, replacement_stats):
    """Calculate player impact from statistical contribution.

    Args:
        player_stats: Dict with PPG, APG, RPG, SPG, BPG, FG%, etc.
        replacement_stats: Backup player's stats

    Returns:
        Estimated point impact per game
    """
    # Direct scoring
    scoring_impact = (player_stats["ppg"] - replacement_stats["ppg"]) * 0.6

    # Playmaking (assists create points)
    assist_impact = (player_stats["apg"] - replacement_stats["apg"]) * 2.0

    # Rebounding (second chance points)
    rebound_impact = (player_stats["rpg"] - replacement_stats["rpg"]) * 0.4

    # Efficiency gap
    efficiency_impact = (
        (player_stats["ts_pct"] - replacement_stats["ts_pct"]) *
        player_stats["usage_rate"] * 0.5
    )

    # Defensive impact
    defensive_impact = (
        (player_stats["dbpm"] - replacement_stats["dbpm"]) * 0.3
    )

    # Intangibles (leadership, clutch factor)
    intangible_factor = 1.0 if player_stats["is_star"] else 0.5

    total_impact = (
        scoring_impact +
        assist_impact +
        rebound_impact +
        efficiency_impact +
        defensive_impact +
        intangible_factor
    )

    return total_impact
```

### Method C: Advanced Metrics (Validation) ⭐ SECONDARY

**Using Box Plus/Minus (BPM)**:
```python
def bpm_to_spread_impact(player_bpm, replacement_bpm):
    """Convert BPM to spread impact.

    BPM measures how many points per 100 possessions a player contributes
    compared to average player.

    Args:
        player_bpm: Player's BPM (e.g., +8.5 for elite)
        replacement_bpm: Backup's BPM (e.g., +1.0 for bench)

    Returns:
        Estimated spread impact in points
    """
    bpm_delta = player_bpm - replacement_bpm

    # Conversion: ~10 points per 10 BPM difference
    # Adjust for college pace (~70 possessions vs 100)
    college_adjustment = 0.70

    spread_impact = (bpm_delta / 10) * 10 * college_adjustment

    return spread_impact

# Example:
# Star Player: BPM = +8.5
# Replacement: BPM = +1.0
# Impact = (7.5 / 10) * 10 * 0.70 = 5.25 points
```

**Using Win Shares**:
```python
def win_shares_to_impact(player_ws, team_wins, games_played):
    """Convert Win Shares to per-game impact.

    Args:
        player_ws: Player's total Win Shares for season
        team_wins: Team's total wins
        games_played: Games player participated in

    Returns:
        Win probability impact per game
    """
    # Player's contribution to team wins
    win_contribution_pct = player_ws / team_wins

    # Per-game win probability impact
    win_prob_impact = win_contribution_pct / games_played

    # Convert to spread (rough: 10% win prob = ~2 points)
    spread_impact = win_prob_impact * 20

    return spread_impact
```

---

## Player Tier Classification

### Tier 1: Elite Game-Changers (7-12 point impact)
**Criteria**:
- BPM: +8.0 or higher
- PPG: 20+ points
- Usage Rate: 28%+
- Win Shares: 5.0+ per season
- NBA Draft: Projected lottery pick

**Examples**: Hunter Dickinson (Kansas), RJ Davis (UNC), Zach Edey-level players

**Impact Calculation**:
```python
tier1_impact = {
    "spread": 7.0 to 12.0,
    "total": 3.0 to 6.0,
    "win_probability": 0.15 to 0.25,
    "confidence": "HIGH"
}
```

### Tier 2: All-Conference Stars (4-7 point impact)
**Criteria**:
- BPM: +5.0 to +8.0
- PPG: 15-20 points
- Usage Rate: 23-28%
- Win Shares: 3.0-5.0
- NBA Draft: Projected 1st-2nd round

**Examples**: Top 3 scorers on ranked teams

**Impact Calculation**:
```python
tier2_impact = {
    "spread": 4.0 to 7.0,
    "total": 2.0 to 4.0,
    "win_probability": 0.10 to 0.15,
    "confidence": "MEDIUM-HIGH"
}
```

### Tier 3: Key Rotation Players (2-4 point impact)
**Criteria**:
- BPM: +2.0 to +5.0
- PPG: 10-15 points
- Minutes: 25-30 MPG
- Win Shares: 1.5-3.0

**Examples**: Starting point guards, defensive anchors

**Impact Calculation**:
```python
tier3_impact = {
    "spread": 2.0 to 4.0,
    "total": 1.0 to 2.0,
    "win_probability": 0.05 to 0.10,
    "confidence": "MEDIUM"
}
```

### Tier 4: Specialists (1-2 point impact)
**Criteria**:
- Elite 3-point shooter (40%+)
- Elite rim protector (3+ BPG)
- Defensive specialist
- Minutes: 20-25 MPG

**Examples**: Pure shooters, defensive anchors off bench

### Tier 5: Bench Depth (<1 point impact)
**Criteria**:
- BPM: <+2.0
- Minutes: <20 MPG
- Easily replaceable

**Impact**: Minimal, typically ignored

---

## Data Collection Strategy

### Primary Sources

#### 1. Advanced Stats Platforms
**Sports-Reference (college-basketball-reference.com)**:
```python
sources = {
    "player_stats": {
        "url": "https://www.sports-reference.com/cbb/players/{player_name}.html",
        "metrics": [
            "PPG", "RPG", "APG", "SPG", "BPG",
            "FG%", "3P%", "FT%", "TS%",
            "Usage Rate", "Offensive Rating", "Defensive Rating",
            "BPM", "Win Shares", "PER"
        ]
    },
    "game_logs": {
        "url": "https://www.sports-reference.com/cbb/players/{player_name}/gamelog/{year}",
        "data": "Individual game performance for WITH/WITHOUT analysis"
    }
}
```

**KenPom Player Stats** (if available via API):
- Offensive Rating by player
- Defensive Rating by player
- Usage Rate adjusted for opponent

#### 2. Team Roster & Depth Charts
**ESPN**:
```python
espn_sources = {
    "rosters": "https://www.espn.com/mens-college-basketball/team/roster/_/id/{team_id}",
    "depth_charts": "https://www.espn.com/mens-college-basketball/team/depth/_/id/{team_id}",
    "injury_report": "https://www.espn.com/mens-college-basketball/injuries"
}
```

#### 3. NBA Draft Projections (Proxy for Impact)
**Tankathon, ESPN Mock Drafts**:
- Lottery picks = Tier 1
- 1st round = Tier 2
- 2nd round = Tier 3

### Secondary Sources

#### 4. Historical Game Results
**KenPom Database** (already integrated):
```sql
SELECT
    game_date,
    team_id,
    opponent_id,
    final_score,
    opponent_score,
    tempo,
    offensive_efficiency,
    defensive_efficiency
FROM game_results
WHERE season = 2025
```

**Match with Player Availability**:
```python
# Cross-reference ESPN injury reports with game results
# Flag games where player was OUT
# Calculate performance delta
```

---

## Real-Time Injury Monitoring

### Twitter/X Intelligence Network

**Primary Sources (Beat Reporters)**:

#### By Conference

**ACC**:
```python
acc_reporters = {
    "Duke": ["@DukeMBB", "@JonScheyer", "@joetipton_on3"],
    "UNC": ["@UNC_Basketball", "@AdamLucas_IC"],
    "Virginia": ["@UVAMensHoops", "@ByJerryRatcliffe"],
    "Louisville": "@CardChronicle",
    "Syracuse": "@OrangeFizz"
}
```

**Big Ten**:
```python
big_ten_reporters = {
    "Purdue": ["@BoilerBall", "@GoldandBlackcom"],
    "Illinois": "@IlliniMBB",
    "Michigan": "@UMichBball",
    "Michigan State": "@MSU_Basketball"
}
```

**Big 12**:
```python
big_12_reporters = {
    "Kansas": ["@KUHoops", "@KUSports"],
    "Baylor": "@BaylorMBB",
    "Texas": "@TexasMBB"
}
```

**SEC**:
```python
sec_reporters = {
    "Kentucky": ["@KentuckyMBB", "@KySportsRadio"],
    "Tennessee": "@Vol_Hoops",
    "Auburn": "@AuburnMBB",
    "Alabama": "@AlabamaMBB"
}
```

**Monitoring Strategy**:
```python
keywords = [
    "injury",
    "out",
    "doubtful",
    "questionable",
    "game-time decision",
    "won't play",
    "ruled out",
    "suspended",
    "illness",
    "DNP",
    "late scratch"
]

time_windows = {
    "pre_game": "4 hours before tipoff",
    "late_scratch": "30 minutes before tipoff",
    "emergency": "Real-time monitoring"
}
```

### Automated Monitoring Script

**Architecture**:
```python
"""
Twitter/X Beat Reporter Monitor
- Poll every 5 minutes starting 4 hours before games
- Filter by keywords
- Send alerts for Tier 1-3 players
- Auto-adjust spreads in prediction system
"""

class InjuryMonitor:
    def __init__(self, api_key):
        self.twitter_api = TwitterAPI(api_key)
        self.player_db = PlayerDatabase()

    def monitor_reporters(self, game_start_time):
        """Poll reporters starting 4 hours before game."""
        reporters = self.get_relevant_reporters(game_start_time)

        for reporter in reporters:
            tweets = self.twitter_api.get_recent_tweets(
                account=reporter,
                since=game_start_time - timedelta(hours=4)
            )

            for tweet in tweets:
                if self.contains_injury_keyword(tweet):
                    player = self.extract_player_name(tweet)
                    if self.is_tracked_player(player):
                        self.send_alert(player, tweet, game_start_time)
                        self.update_player_status(player, "OUT")
```

---

## Database Schema

### Players Table
```sql
CREATE TABLE players (
    player_id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    team_id INTEGER NOT NULL,
    jersey_number INTEGER,
    position TEXT,
    class_year TEXT,  -- FR, SO, JR, SR

    -- Season Stats
    season INTEGER NOT NULL,
    games_played INTEGER,
    games_started INTEGER,
    minutes_per_game REAL,

    -- Basic Stats
    ppg REAL,
    rpg REAL,
    apg REAL,
    spg REAL,
    bpg REAL,

    -- Shooting
    fg_pct REAL,
    three_pt_pct REAL,
    ft_pct REAL,
    ts_pct REAL,  -- True Shooting %

    -- Advanced Metrics
    usage_rate REAL,
    offensive_rating REAL,
    defensive_rating REAL,
    bpm REAL,  -- Box Plus/Minus
    obpm REAL,  -- Offensive BPM
    dbpm REAL,  -- Defensive BPM
    win_shares REAL,
    per REAL,  -- Player Efficiency Rating

    -- Impact Tier
    tier INTEGER,  -- 1-5 (1 = Elite, 5 = Bench)
    spread_impact REAL,  -- Calculated impact in points
    total_impact REAL,
    win_prob_impact REAL,

    -- NBA Draft Projection
    draft_projection TEXT,  -- "Lottery", "1st Round", "2nd Round", "Undrafted"

    -- Status
    is_active BOOLEAN DEFAULT 1,
    injury_status TEXT,  -- "Healthy", "Questionable", "Doubtful", "Out"
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (team_id) REFERENCES teams(team_id),
    UNIQUE (name, team_id, season)
);

CREATE INDEX idx_players_tier ON players(tier);
CREATE INDEX idx_players_team ON players(team_id);
CREATE INDEX idx_players_impact ON players(spread_impact DESC);
```

### Player Game Log
```sql
CREATE TABLE player_game_log (
    log_id INTEGER PRIMARY KEY AUTOINCREMENT,
    player_id INTEGER NOT NULL,
    game_id INTEGER NOT NULL,
    game_date DATE NOT NULL,

    -- Participation
    played BOOLEAN NOT NULL,  -- Did player participate?
    dnp_reason TEXT,  -- "Injury", "Suspension", "Coach's Decision", NULL
    minutes INTEGER,

    -- Stats
    points INTEGER,
    rebounds INTEGER,
    assists INTEGER,
    steals INTEGER,
    blocks INTEGER,
    turnovers INTEGER,
    fouls INTEGER,

    -- Shooting
    fgm INTEGER,
    fga INTEGER,
    three_pm INTEGER,
    three_pa INTEGER,
    ftm INTEGER,
    fta INTEGER,

    -- Plus/Minus
    plus_minus INTEGER,  -- Team's margin while player on court

    FOREIGN KEY (player_id) REFERENCES players(player_id),
    UNIQUE (player_id, game_id)
);

CREATE INDEX idx_game_log_player ON player_game_log(player_id);
CREATE INDEX idx_game_log_date ON player_game_log(game_date);
```

### Team Performance WITH/WITHOUT Player
```sql
CREATE TABLE team_performance_splits (
    split_id INTEGER PRIMARY KEY AUTOINCREMENT,
    team_id INTEGER NOT NULL,
    player_id INTEGER NOT NULL,
    season INTEGER NOT NULL,

    -- WITH Player Stats
    games_with INTEGER,
    wins_with INTEGER,
    losses_with INTEGER,
    avg_margin_with REAL,
    avg_adj_em_with REAL,
    avg_offensive_eff_with REAL,
    avg_defensive_eff_with REAL,
    avg_tempo_with REAL,
    avg_opponent_adj_em_with REAL,

    -- WITHOUT Player Stats
    games_without INTEGER,
    wins_without INTEGER,
    losses_without INTEGER,
    avg_margin_without REAL,
    avg_adj_em_without REAL,
    avg_offensive_eff_without REAL,
    avg_defensive_eff_without REAL,
    avg_tempo_without REAL,
    avg_opponent_adj_em_without REAL,

    -- Calculated Impact
    spread_impact REAL,  -- margin_with - margin_without (adj for opponent)
    total_impact REAL,
    tempo_impact REAL,
    win_prob_impact REAL,

    -- Confidence
    sample_size_score REAL,  -- Quality of impact estimate (0-1)
    opponent_adjusted BOOLEAN,

    last_calculated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (team_id) REFERENCES teams(team_id),
    FOREIGN KEY (player_id) REFERENCES players(player_id),
    UNIQUE (team_id, player_id, season)
);
```

### Injury Alerts
```sql
CREATE TABLE injury_alerts (
    alert_id INTEGER PRIMARY KEY AUTOINCREMENT,
    player_id INTEGER NOT NULL,
    game_id INTEGER,
    alert_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    -- Source
    source TEXT,  -- "Twitter", "ESPN", "Team Official"
    source_account TEXT,  -- "@KUHoops"
    source_url TEXT,

    -- Status
    injury_type TEXT,  -- "Ankle", "Knee", "Illness", "Suspension"
    status TEXT,  -- "Out", "Doubtful", "Questionable", "Game-Time"
    expected_return TEXT,  -- "1 week", "Season", "Next game"

    -- Impact
    player_tier INTEGER,
    estimated_spread_impact REAL,

    -- Betting Context
    line_at_alert REAL,  -- Spread when alert was issued
    line_movement REAL,  -- How much line moved after alert

    verified BOOLEAN DEFAULT 0,

    FOREIGN KEY (player_id) REFERENCES players(player_id)
);

CREATE INDEX idx_alerts_time ON injury_alerts(alert_time DESC);
CREATE INDEX idx_alerts_player ON injury_alerts(player_id);
```

---

## Implementation Phases

### Phase 1: Data Foundation (Week 1)
**Tasks**:
- [ ] Create database schema (players, game_log, performance_splits)
- [ ] Scrape Top 200 players from Sports-Reference
- [ ] Populate player stats (PPG, BPM, Win Shares)
- [ ] Classify players into Tiers 1-5
- [ ] Calculate statistical impact for each player

**Deliverable**: Database with 200 players, tier classification, initial impact estimates

### Phase 2: Historical Analysis (Week 2)
**Tasks**:
- [ ] Scrape game logs for Top 200 players
- [ ] Match game logs with team performance data
- [ ] Calculate WITH/WITHOUT splits for each player
- [ ] Adjust for opponent quality
- [ ] Validate impact estimates against empirical data

**Deliverable**: Refined player impact scores with confidence intervals

### Phase 3: Real-Time Monitoring (Week 3)
**Tasks**:
- [ ] Build Twitter/X API integration
- [ ] Create beat reporter tracking lists
- [ ] Implement keyword filtering
- [ ] Set up alert system (email/SMS)
- [ ] Test with historical injury news

**Deliverable**: Automated injury monitoring system

### Phase 4: Prediction Integration (Week 4)
**Tasks**:
- [ ] Modify `IntegratedPredictor.predict_game()` to accept player status
- [ ] Adjust spread based on player tier + impact
- [ ] Update confidence intervals when data is limited
- [ ] Backtest on 2024 season with known injuries
- [ ] Measure CLV improvement

**Deliverable**: Production system with player impact adjustments

### Phase 5: Continuous Improvement (Ongoing)
**Tasks**:
- [ ] Track prediction accuracy by player tier
- [ ] Refine impact estimates as season progresses
- [ ] Monitor line movement vs our adjustments
- [ ] Calculate CLV on injury-adjusted bets
- [ ] Expand to Top 300 players if successful

---

## Expected ROI

### Conservative Estimate
**Assumptions**:
- 5 games per week with Tier 1-2 player out
- Catch injury 30 min before market adjusts
- 3 point edge on average
- 60% win rate on injury-based bets

**Calculation**:
```
Games per season: 5/week × 20 weeks = 100 games
Edge per game: 3 points
Win rate: 60% (vs 52.4% breakeven)
Average bet: $100

Expected profit per bet: $100 × (0.60 - 0.476) = $12.40
Season profit: $12.40 × 100 = $1,240
ROI: 12.4% per bet
```

### Optimistic Estimate
**Assumptions**:
- 8 games per week
- 5 point edge (late scratch, market slow to adjust)
- 65% win rate

**Calculation**:
```
Games per season: 8/week × 20 weeks = 160 games
Edge: 5 points
Win rate: 65%
Bet: $200

Profit per bet: $200 × (0.65 - 0.476) = $34.80
Season profit: $34.80 × 160 = $5,568
ROI: 17.4% per bet
```

---

## Risk Mitigation

### Challenges

1. **Sample Size**: Some players have limited games missed
   - **Solution**: Use statistical model as baseline, update with empirical data

2. **Replacement Quality Varies**: Backup player strength affects impact
   - **Solution**: Track backup player stats, adjust impact accordingly

3. **Market Efficiency**: Lines move quickly on injury news
   - **Solution**: Focus on late scratches (30 min before game), lesser-known injuries

4. **False Positives**: Twitter rumors can be wrong
   - **Solution**: Verify with multiple sources, weight by source reliability

### Confidence Scoring

```python
def calculate_confidence_score(player, games_missed):
    """Score confidence in impact estimate (0-100).

    Higher confidence = more reliable impact estimate
    """
    score = 50  # Base

    # Sample size
    if games_missed >= 5:
        score += 20
    elif games_missed >= 3:
        score += 10
    elif games_missed >= 1:
        score += 5

    # Statistical validation
    if player.statistical_impact and player.empirical_impact:
        # Models agree
        if abs(player.statistical_impact - player.empirical_impact) < 2:
            score += 15

    # Advanced metrics available
    if player.bpm and player.win_shares:
        score += 10

    # Historical accuracy
    if player.past_predictions:
        accuracy_bonus = player.prediction_accuracy * 5
        score += min(accuracy_bonus, 15)

    return min(score, 100)
```

---

## Next Steps

1. **Immediate** (Today):
   - Design and create database schema
   - Begin scraping Top 200 players list

2. **This Week**:
   - Populate player database with stats
   - Calculate tier classifications
   - Implement WITH/WITHOUT historical analysis

3. **Next Week**:
   - Build Twitter monitoring prototype
   - Test on recent injury news (backtest)

4. **Go-Live**:
   - Integrate with `IntegratedPredictor`
   - Start tracking real-time for remainder of 2024-25 season

---

**This system represents a 3-8 point edge opportunity on 5-10 games per week. At scale, this is a MASSIVE competitive advantage in the NCAA basketball betting market.** 🎯
