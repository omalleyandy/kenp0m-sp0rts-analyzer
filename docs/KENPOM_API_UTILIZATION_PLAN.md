# KenPom API Utilization Plan

**Date**: 2025-12-19
**Author**: Andy + Claude
**Purpose**: Comprehensive strategy for leveraging all KenPom API endpoints to enhance prediction accuracy and achieve 54%+ ATS win rate

---

## Executive Summary

The KenPom API provides 9 endpoints with rich basketball analytics data. Currently, only **3 of 9 endpoints** (33%) are actively integrated into the prediction system. This plan outlines a phased approach to leverage the remaining **6 unused endpoints** to improve prediction accuracy, feature engineering, and betting edge detection.

**Key Opportunity**: The `fanmatch` endpoint provides KenPom's own game predictions with win probabilities - perfect for ensemble modeling and edge detection.

---

## Current State Assessment

### ✅ Currently Integrated (3 endpoints)

| Endpoint | Usage | Data Stored | Last Sync |
|----------|-------|-------------|-----------|
| **ratings** | Core team efficiency metrics | ratings_snapshots table | Daily via BatchScheduler |
| **four-factors** | Dean Oliver's Four Factors | four_factors table | Daily via BatchScheduler |
| **pointdist** | Point distribution (FT/2pt/3pt) | point_distribution table | Daily via BatchScheduler |

**Storage**: All data persists in `data/kenpom.db` SQLite database (8 tables)
**Sync**: Automated daily sync at 6 AM via `batch_scheduler.py`
**Usage**: Powers XGBoost feature engineering (27 features)

### ❌ Not Currently Integrated (6 endpoints)

| Endpoint | Opportunity | Priority | Complexity |
|----------|-------------|----------|------------|
| **fanmatch** | Game predictions w/ win probabilities | 🔥 HIGH | Low |
| **misc-stats** | Advanced shooting/assist metrics | 🔥 HIGH | Medium |
| **height** | Team height, experience, bench strength | 🟡 MEDIUM | Low |
| **archive** | Historical ratings for trend analysis | 🟡 MEDIUM | Medium |
| **teams** | Team metadata lookup automation | 🟢 LOW | Low |
| **conferences** | Conference validation | 🟢 LOW | Low |

---

## Phased Implementation Plan

### Phase 1: High-Value Predictions & Features (Weeks 1-2)

#### 1.1 Fanmatch Integration (🔥 HIGHEST PRIORITY)

**Objective**: Leverage KenPom's own game predictions for ensemble modeling and edge detection

**Endpoints Used**:
- `GET /api.php?endpoint=fanmatch&d=YYYY-MM-DD`

**Data Fields**:
- `HomePred` / `VisitorPred`: Predicted scores
- `HomeWP`: Home win probability (%)
- `PredTempo`: Predicted tempo
- `ThrillScore`: Expected game excitement

**Implementation**:

1. **Database Schema** (`fanmatch_predictions` table):
```sql
CREATE TABLE fanmatch_predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    game_id INTEGER NOT NULL,
    game_date DATE NOT NULL,
    season INTEGER NOT NULL,
    home_team_id INTEGER NOT NULL,
    visitor_team_id INTEGER NOT NULL,
    home_pred REAL NOT NULL,
    visitor_pred REAL NOT NULL,
    home_wp REAL NOT NULL,  -- Win probability
    pred_tempo REAL NOT NULL,
    thrill_score REAL,
    fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (home_team_id) REFERENCES teams(team_id),
    FOREIGN KEY (visitor_team_id) REFERENCES teams(team_id),
    UNIQUE (game_id, game_date)
);
```

2. **KenPomService Methods**:
```python
def sync_daily_games(self, game_date: date | None = None) -> SyncResult:
    """Sync daily game predictions from fanmatch endpoint."""

def get_kenpom_prediction(self, home_team_id: int, visitor_team_id: int, game_date: date) -> dict:
    """Get KenPom's prediction for a specific game."""
```

3. **Ensemble Model Enhancement** (`integrated_predictor.py`):
```python
class IntegratedPredictor:
    def predict_game(self, team1: str, team2: str, ...) -> PredictionResult:
        # Current: XGBoost prediction only
        xgb_margin = self.xgboost_predictor.predict(...)

        # NEW: Blend with KenPom prediction
        kenpom_margin = self.kenpom_service.get_kenpom_prediction(...)

        # Ensemble: Weight 70% XGBoost, 30% KenPom
        final_margin = (0.7 * xgb_margin) + (0.3 * kenpom_margin)

        # Edge detection: Compare consensus to Vegas
        edge = final_margin - vegas_spread
```

**Value Proposition**:
- KenPom predictions are highly respected in basketball analytics community
- Ensemble modeling typically outperforms single models
- Provides second opinion for edge validation
- `ThrillScore` helps identify high-variance games (avoid betting)

**Automated Workflow**:
```python
# scripts/daily_kenpom_sync.py
def main():
    service = KenPomService()
    today = date.today()

    # Sync ratings, factors, point dist (existing)
    service.sync_all(year=2025)

    # NEW: Sync daily game predictions
    service.sync_daily_games(game_date=today)
```

---

#### 1.2 Misc-Stats Integration (🔥 HIGH PRIORITY)

**Objective**: Add advanced shooting and playmaking metrics for richer feature engineering

**Endpoints Used**:
- `GET /api.php?endpoint=misc-stats&y=2025`

**Data Fields** (Offense):
- `FG3Pct`, `FG2Pct`, `FTPct`: Shooting percentages
- `BlockPct`: Offensive block percentage
- `StlRate`: Steal rate (offensive steals)
- `ARate`: Assist rate
- `F3GRate`: 3-point attempt rate

**Data Fields** (Defense):
- `OppFG3Pct`, `OppFG2Pct`, `OppFTPct`: Opponent shooting allowed
- `OppBlockPct`, `OppStlRate`, `OppARate`, `OppF3GRate`

**Implementation**:

1. **Database Schema** (`misc_stats` table):
```sql
CREATE TABLE misc_stats (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    snapshot_date DATE NOT NULL,
    season INTEGER NOT NULL,
    team_id INTEGER NOT NULL,
    -- Offensive metrics
    fg3_pct REAL, rank_fg3_pct INTEGER,
    fg2_pct REAL, rank_fg2_pct INTEGER,
    ft_pct REAL, rank_ft_pct INTEGER,
    block_pct REAL, rank_block_pct INTEGER,
    stl_rate REAL, rank_stl_rate INTEGER,
    nst_rate REAL, rank_nst_rate INTEGER,
    a_rate REAL, rank_a_rate INTEGER,
    f3g_rate REAL, rank_f3g_rate INTEGER,
    -- Defensive metrics
    opp_fg3_pct REAL, rank_opp_fg3_pct INTEGER,
    opp_fg2_pct REAL, rank_opp_fg2_pct INTEGER,
    opp_ft_pct REAL, rank_opp_ft_pct INTEGER,
    opp_block_pct REAL, rank_opp_block_pct INTEGER,
    opp_stl_rate REAL, rank_opp_stl_rate INTEGER,
    opp_nst_rate REAL, rank_opp_nst_rate INTEGER,
    opp_a_rate REAL, rank_opp_a_rate INTEGER,
    opp_f3g_rate REAL, rank_opp_f3g_rate INTEGER,
    FOREIGN KEY (team_id) REFERENCES teams(team_id),
    UNIQUE (snapshot_date, team_id)
);
```

2. **XGBoost Feature Expansion** (Add 8 new features):
```python
# Current: 27 features
# NEW: 35 features (27 + 8 misc-stats)

features.update({
    # Shooting efficiency differentials
    "three_pct_diff": misc1.fg3_pct - misc2.fg3_pct,
    "two_pct_diff": misc1.fg2_pct - misc2.fg2_pct,
    "ft_pct_diff": misc1.ft_pct - misc2.ft_pct,

    # Playmaking differentials
    "assist_rate_diff": misc1.a_rate - misc2.a_rate,
    "three_attempt_rate_diff": misc1.f3g_rate - misc2.f3g_rate,

    # Defensive pressure differentials
    "steal_rate_diff": misc1.stl_rate - misc2.stl_rate,
    "block_pct_diff": misc1.block_pct - misc2.block_pct,

    # Defensive matchup advantages
    "three_pct_matchup_adv": misc1.fg3_pct - misc2.opp_fg3_pct,
})
```

**Value Proposition**:
- Current model uses basic shooting stats from point_distribution
- Misc-stats provides actual shooting percentages (not just point distribution)
- Assist rate correlates with offensive efficiency
- Three-point attempt rate identifies style matchups
- Defensive metrics help predict opponent's scoring struggles

---

### Phase 2: Team Context & Experience (Weeks 3-4)

#### 2.1 Height & Experience Integration (🟡 MEDIUM PRIORITY)

**Objective**: Add team composition metrics for matchup analysis

**Endpoints Used**:
- `GET /api.php?endpoint=height&y=2025`

**Data Fields**:
- `AvgHgt`, `HgtEff`: Average and effective height
- `Hgt1` through `Hgt5`: Position-specific heights
- `Exp`: Experience rating
- `Bench`: Bench strength rating
- `Continuity`: Team continuity rating

**Implementation**:

1. **Database Schema** (`height_experience` table - ALREADY EXISTS!):
```sql
-- Table already exists in database schema
-- Just need to populate it via sync
```

2. **KenPomService Enhancement**:
```python
def sync_height_experience(self, year: int | None = None) -> SyncResult:
    """Sync height and experience data from API."""
    response = self.api.get_height(year=year)
    count = self.repository.save_height_experience(snapshot_date, data)
    return SyncResult(...)
```

3. **Feature Engineering** (Add 4 new features):
```python
features.update({
    "height_diff": height1.avg_hgt - height2.avg_hgt,
    "experience_diff": height1.exp - height2.exp,
    "bench_strength_diff": height1.bench - height2.bench,
    "continuity_diff": height1.continuity - height2.continuity,
})
```

**Value Proposition**:
- Height advantages matter in rebounding and interior defense
- Experienced teams handle pressure better (March Madness)
- Bench strength predicts late-game performance
- Continuity correlates with chemistry and execution

**Use Cases**:
- Tournament games: Experience matters more in high-pressure situations
- Injury situations: Bench strength becomes critical
- Style matchups: Height vs speed (tall slow teams vs small fast teams)

---

#### 2.2 Archive Integration for Trend Analysis (🟡 MEDIUM PRIORITY)

**Objective**: Track rating changes over time to identify momentum and regression

**Endpoints Used**:
- `GET /api.php?endpoint=archive&d=YYYY-MM-DD`
- `GET /api.php?endpoint=archive&preseason=true&y=2025`

**Data Fields**:
- `AdjEM`, `AdjOE`, `AdjDE`, `AdjTempo`: Ratings on archive date
- `AdjEMFinal`, `AdjOEFinal`, `AdjDEFinal`: Final season ratings
- `RankChg`, `AdjEMChg`, `AdjTChg`: Changes from archive to final

**Implementation**:

1. **Use Case: Preseason to Current Comparison**:
```python
# Identify overachievers vs underachievers
preseason = api.get_archive(preseason=True, year=2025)
current = api.get_ratings(year=2025)

for team in teams:
    preseason_rank = team['RankAdjEM']  # Expected ranking
    current_rank = current[team_id]['RankAdjEM']  # Actual ranking

    if current_rank < preseason_rank - 20:
        # Team exceeding expectations - bet against (regression risk)
    elif current_rank > preseason_rank + 20:
        # Team underperforming - potential buy-low opportunity
```

2. **Use Case: Weekly Rating Changes**:
```python
# Track momentum by comparing weekly snapshots
def get_team_momentum(team_id: int, weeks: int = 4) -> float:
    """Calculate efficiency rating change over last N weeks."""
    snapshots = repository.get_rating_history(team_id, weeks)

    # Rising teams: +2.0+ AdjEM in 4 weeks
    # Falling teams: -2.0+ AdjEM in 4 weeks
    momentum = snapshots[0].adj_em - snapshots[-1].adj_em
    return momentum
```

3. **Feature Engineering** (Add 3 momentum features):
```python
features.update({
    "momentum_diff_4wk": get_momentum(team1, 4) - get_momentum(team2, 4),
    "rank_change_diff": (preseason_rank1 - current_rank1) - (preseason_rank2 - current_rank2),
    "regression_risk_diff": calculate_regression_risk(team1) - calculate_regression_risk(team2),
})
```

**Value Proposition**:
- Momentum matters: Hot teams cover spreads better than cold teams
- Regression to mean: Teams significantly exceeding preseason expectations tend to regress
- Injury tracking: Sharp drops in ratings signal key injuries
- Schedule strength: Teams with front-loaded schedules show inflated early ratings

**Data Storage Strategy**:
- Weekly snapshots: Store every Monday to track weekly changes
- Preseason baseline: Store once per season (October)
- Historical lookback: Maintain 2-3 seasons for year-over-year comparisons

---

### Phase 3: Infrastructure & Metadata (Week 5)

#### 3.1 Teams & Conferences Metadata (🟢 LOW PRIORITY)

**Objective**: Automate team lookups and validate conference affiliations

**Endpoints Used**:
- `GET /api.php?endpoint=teams&y=2025`
- `GET /api.php?endpoint=conferences&y=2025`

**Implementation**:

1. **Auto-populate Teams Table**:
```python
def sync_teams_metadata(year: int = 2025) -> int:
    """Populate teams table with coach, arena, conference data."""
    teams = api.get_teams(year=year)

    for team in teams.data:
        repository.upsert_team(
            team_id=team['TeamID'],
            team_name=team['TeamName'],
            coach=team['Coach'],
            arena=team['Arena'],
            arena_city=team['ArenaCity'],
            arena_state=team['ArenaState'],
            conference=team['ConfShort'],
        )
```

2. **Conference-Level Analytics**:
```python
def get_conference_strength(conference: str, year: int) -> dict:
    """Calculate average conference strength metrics."""
    teams = api.get_ratings(year=year, conference=conference)

    return {
        "avg_adj_em": mean(t['AdjEM'] for t in teams),
        "avg_sos": mean(t['SOS'] for t in teams),
        "top_25_count": sum(1 for t in teams if t['RankAdjEM'] <= 25),
    }
```

**Value Proposition**:
- Automates team ID lookups (no more manual searches)
- Conference strength matters for tournament seeding
- Conference-only stats available via `conf_only=true` parameter
- Coach changes tracked year-over-year

---

## Feature Engineering Summary

### Current XGBoost Features (27)

**Core Efficiency** (6):
- adj_em_diff, adj_oe_diff, adj_de_diff, adj_tempo_diff
- pythag_diff, luck_diff

**Strength of Schedule** (2):
- sos_diff, ncsos_diff

**Four Factors** (6):
- efg_diff, to_diff, or_diff, ft_rate_diff
- efg_adv_t1, efg_adv_t2

**Point Distribution** (4):
- three_pct_diff, two_pct_diff, ft_pct_diff, three_reliance_diff

**Rankings** (4):
- rank_diff, rank_oe_diff, rank_de_diff, win_pct_diff

**Tempo/Context** (5):
- home_advantage, apl_off_diff, apl_def_diff, avg_tempo

---

### Proposed New Features (+19 features)

**From misc-stats** (8):
- three_pct_diff, two_pct_diff, ft_pct_diff (actual shooting %, not distribution)
- assist_rate_diff, three_attempt_rate_diff
- steal_rate_diff, block_pct_diff, three_pct_matchup_adv

**From height** (4):
- height_diff, experience_diff, bench_strength_diff, continuity_diff

**From archive** (3):
- momentum_diff_4wk, rank_change_diff, regression_risk_diff

**From fanmatch** (4):
- kenpom_spread_diff (their spread vs ours)
- kenpom_wp_diff (their win prob vs ours)
- kenpom_tempo_agreement (their tempo vs our predicted tempo)
- thrill_score (high variance indicator - avoid betting)

**Total**: 27 → 46 features (+70% expansion)

---

## Edge Detection Enhancement

### Current Edge Detection Logic

```python
# From integrated_predictor.py
if result.has_spread_edge:
    edge = result.predicted_margin - vegas_spread
    if abs(edge) >= 2.0:
        # Bet signal
```

**Issues**:
- Single model opinion (XGBoost only)
- No confidence intervals
- No ensemble validation

---

### Enhanced Edge Detection with Fanmatch

```python
class EdgeDetector:
    def detect_edge(self, game: Game, vegas_spread: float) -> EdgeResult:
        # Get predictions from multiple sources
        xgb_margin = self.xgboost_predictor.predict(game)
        kenpom_margin = self.get_kenpom_fanmatch(game)

        # Ensemble prediction (weighted average)
        ensemble_margin = (0.7 * xgb_margin) + (0.3 * kenpom_margin)

        # Calculate edge
        edge = ensemble_margin - vegas_spread

        # Confidence checks
        model_agreement = abs(xgb_margin - kenpom_margin) < 3.0
        thrill_score = self.get_thrill_score(game)

        # Edge qualification
        if abs(edge) >= 2.0 and model_agreement and thrill_score < 7.0:
            return EdgeResult(
                edge=edge,
                confidence="HIGH",
                ensemble_margin=ensemble_margin,
                model_agreement=model_agreement,
                reason="Both XGBoost and KenPom agree on value"
            )
        elif abs(edge) >= 3.0 and not model_agreement:
            return EdgeResult(
                edge=edge,
                confidence="MEDIUM",
                reason="Strong edge but models disagree - proceed with caution"
            )
        else:
            return EdgeResult(
                edge=edge,
                confidence="LOW",
                reason="Edge too small or high variance game"
            )
```

**Value Adds**:
- **Model Agreement**: Only bet when both models see value
- **Thrill Score Filter**: Avoid high-variance "coin flip" games
- **Confidence Tiers**: HIGH/MEDIUM/LOW for bet sizing
- **Ensemble Stability**: Reduces overfitting to single model

---

## Implementation Roadmap

### Week 1: Fanmatch Foundation
- [ ] Create `fanmatch_predictions` table schema
- [ ] Add `sync_daily_games()` to KenPomService
- [ ] Add `get_kenpom_prediction()` method
- [ ] Update `batch_scheduler.py` to sync fanmatch daily
- [ ] Test with recent games data

### Week 2: Ensemble Integration
- [ ] Modify `IntegratedPredictor` to blend XGBoost + KenPom
- [ ] Implement edge detection with model agreement checks
- [ ] Add thrill score filtering
- [ ] Backtest ensemble vs XGBoost-only on historical data
- [ ] Validate CLV improvement

### Week 3: Misc-Stats Expansion
- [ ] Create `misc_stats` table schema
- [ ] Add `sync_misc_stats()` to KenPomService
- [ ] Expand XGBoost features from 27 → 35
- [ ] Retrain model with new features
- [ ] Validate performance improvement

### Week 4: Height & Momentum
- [ ] Populate `height_experience` table (already exists)
- [ ] Implement archive snapshots (weekly + preseason)
- [ ] Add momentum calculation functions
- [ ] Expand features to 46 total
- [ ] Final model retraining

### Week 5: Polish & Automation
- [ ] Auto-populate teams metadata
- [ ] Add conference-level analytics
- [ ] Document all new features in FEATURES.md
- [ ] Update model versioning to v2.0
- [ ] Deploy to production with monitoring

---

## Success Metrics

### Model Performance
- **Current Baseline**: ~52% ATS (XGBoost only)
- **Phase 1 Target**: 53%+ ATS (Ensemble with fanmatch)
- **Phase 2 Target**: 54%+ ATS (Full 46-feature model)
- **Gold Standard**: Positive CLV on 60%+ of bets

### Feature Importance
- Track which new features have highest importance scores
- Remove features with importance < 0.01 (noise)
- Iteratively refine feature set

### Edge Detection Accuracy
- **Model Agreement Rate**: 70%+ (when to trust edge)
- **False Positive Rate**: < 30% (edges that don't hit)
- **High Confidence Bets**: Win rate 55%+ on HIGH confidence edges

---

## Risk Mitigation

### Overfitting Risk
- Use cross-validation with temporal splits (not random)
- Validate on hold-out season (2023 data)
- Monitor feature importance for stability
- Remove correlated features (VIF analysis)

### Data Freshness
- Automated daily sync at 6 AM (before 12 PM ET games)
- Freshness checks: Reject data > 24 hours old
- Fallback: Use yesterday's snapshot if sync fails

### API Rate Limits
- Unknown official rate limits
- Implement exponential backoff (3 retries)
- Cache responses for 6 hours (ratings don't change intraday)
- Monitor 429 responses, adjust if needed

---

## Future Enhancements (Beyond Initial Scope)

### Conference Tournament Context
- Track teams playing on "home court" in conference tournaments
- Identify rest advantages (playing 3 games in 3 days)
- Fatigue modeling for late-round games

### Injury Intelligence
- Scrape injury reports from ESPN/TeamRankings
- Correlate rating drops with player absences
- Build player impact model (points above replacement)

### Line Movement Tracking
- Track opening vs closing spreads
- Identify sharp money movement
- Bet earlier when you agree with sharps

### Tournament Simulation
- Use fanmatch predictions for bracket simulations
- Monte Carlo tournament outcomes
- Expected value calculations for futures bets

---

## Appendix: API Endpoint Quick Reference

| Endpoint | Primary Use | Sync Frequency | Storage Table |
|----------|-------------|----------------|---------------|
| **ratings** | Core efficiency metrics | Daily | ratings_snapshots |
| **four-factors** | Dean Oliver factors | Daily | four_factors |
| **pointdist** | Point distribution | Daily | point_distribution |
| **fanmatch** | Game predictions | Daily | fanmatch_predictions |
| **misc-stats** | Advanced shooting/assists | Daily | misc_stats |
| **height** | Team composition | Weekly | height_experience |
| **archive** | Historical ratings | Weekly | ratings_snapshots (dated) |
| **teams** | Team metadata | Seasonal | teams |
| **conferences** | Conference list | Seasonal | conferences |

---

## Conclusion

The KenPom API provides a wealth of untapped data that can significantly enhance prediction accuracy. By implementing this phased approach, we can:

1. **Immediately improve** with ensemble modeling (fanmatch)
2. **Expand feature richness** with misc-stats and height data
3. **Add context** with trend analysis and momentum tracking
4. **Automate infrastructure** for maintainability

**Expected Outcome**: Achieve 54%+ ATS win rate through better feature engineering, ensemble modeling, and disciplined edge detection.

**Timeline**: 5 weeks to full implementation
**Next Steps**: Start with Phase 1 (fanmatch integration) this week
