# KenPom Database Schema Updates Plan

## Overview
Incremental schema updates to align with KenPom API documentation while preserving existing data and functionality.

## Phase 1: Add Missing Tables

### 1.1 Conferences Table
```sql
CREATE TABLE IF NOT EXISTS conferences (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    season INTEGER NOT NULL,
    conf_id INTEGER NOT NULL,
    conf_short TEXT NOT NULL,
    conf_long TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(season, conf_id)
);

CREATE INDEX IF NOT EXISTS idx_conferences_season ON conferences(season);
CREATE INDEX IF NOT EXISTS idx_conferences_short ON conferences(conf_short);
```

**API Mapping:** Direct from `/api.php?endpoint=conferences&y=2025`

### 1.2 Archive Ratings Table
```sql
CREATE TABLE IF NOT EXISTS archive_ratings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    archive_date DATE NOT NULL,
    season INTEGER NOT NULL,
    is_preseason BOOLEAN DEFAULT 0,
    team_id INTEGER NOT NULL,
    team_name TEXT NOT NULL,
    conference TEXT,
    seed INTEGER,
    event TEXT,

    -- Archive date metrics
    adj_em REAL NOT NULL,
    adj_oe REAL NOT NULL,
    adj_de REAL NOT NULL,
    adj_tempo REAL NOT NULL,
    rank_adj_em INTEGER,
    rank_adj_oe INTEGER,
    rank_adj_de INTEGER,
    rank_adj_tempo INTEGER,

    -- Final season metrics (for comparison)
    adj_em_final REAL,
    adj_oe_final REAL,
    adj_de_final REAL,
    adj_tempo_final REAL,
    rank_adj_em_final INTEGER,
    rank_adj_oe_final INTEGER,
    rank_adj_de_final INTEGER,
    rank_adj_tempo_final INTEGER,

    -- Changes (archive to final)
    rank_change INTEGER,
    adj_em_change REAL,
    adj_tempo_change REAL,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (team_id) REFERENCES teams(team_id),
    UNIQUE(archive_date, team_id, is_preseason)
);

CREATE INDEX IF NOT EXISTS idx_archive_date ON archive_ratings(archive_date);
CREATE INDEX IF NOT EXISTS idx_archive_team ON archive_ratings(team_id);
CREATE INDEX IF NOT EXISTS idx_archive_season ON archive_ratings(season);
CREATE INDEX IF NOT EXISTS idx_archive_preseason ON archive_ratings(is_preseason);
```

**API Mapping:** From `/api.php?endpoint=archive&d=2025-02-15` or `&preseason=true&y=2025`

**Use Case:** Track rating evolution throughout season, identify improvement/decline trends

### 1.3 Update Teams Table
```sql
-- Add missing columns to teams table
ALTER TABLE teams ADD COLUMN season INTEGER;
ALTER TABLE teams ADD COLUMN arena TEXT;
ALTER TABLE teams ADD COLUMN arena_city TEXT;
ALTER TABLE teams ADD COLUMN arena_state TEXT;

-- Add index for season-based queries
CREATE INDEX IF NOT EXISTS idx_teams_season ON teams(season);
```

**API Mapping:** From `/api.php?endpoint=teams&y=2025`

## Phase 2: Handle Incomplete Tables

### 2.1 Efficiency Tempo Table Decision

**Current Status:** Incomplete table with ~181 lines truncated in schema output

**Options:**
1. **Complete it** - Finish the schema and populate (if it serves a unique purpose)
2. **Remove it** - Data appears duplicative with `ratings_snapshots` and `four_factors`
3. **Merge it** - Consolidate into existing tables

**Recommendation:** Remove it. Data is already covered by:
- `ratings_snapshots` (adj_em, adj_oe, adj_de, adj_tempo, sos, luck)
- `four_factors` (efg_pct, to_pct, or_pct, ft_rate)
- `misc_stats` (shooting percentages, assist rates)

```sql
DROP TABLE IF EXISTS efficiency_tempo;
```

## Phase 3: Historical Data Collection Strategy

### 3.1 Data Availability

**KenPom Archive Coverage:**
- Full historical data: 2002-present (22 seasons)
- Archive snapshots: Multiple dates per season
- Preseason ratings: Available for all recent seasons

### 3.2 Rate Limiting Strategy

**Conservative Approach (Recommended):**
```python
RATE_LIMITS = {
    "requests_per_minute": 10,
    "delay_between_requests": 6.0,  # seconds
    "daily_request_cap": 1000,
    "retry_attempts": 3,
    "retry_backoff": 30,  # seconds
    "respect_429_errors": True
}
```

**Request Calculation:**
- ~360 teams per season
- ~8 endpoints to sync per team/date
- ~20 archive dates per season (key dates: preseason, weekly, postseason)
- **Total for 1 season:** 360 teams × 20 dates = 7,200 requests
- **10 seasons:** 72,000 requests
- **At 10 req/min:** 120 hours = 5 days of continuous collection
- **Recommended:** Spread over 14 days = 8.5 hours/day

### 3.3 Collection Priority

**Tier 1 (Critical for XGBoost):**
1. `ratings` endpoint → `ratings_snapshots` (2015-2024)
2. `four-factors` endpoint → `four_factors` (2015-2024)
3. `archive` endpoint → `archive_ratings` (preseason + 5 key dates per season)

**Tier 2 (Enhanced Features):**
4. `pointdist` endpoint → `point_distribution`
5. `misc-stats` endpoint → `misc_stats`
6. `height` endpoint → `height_experience`

**Tier 3 (Reference):**
7. `teams` endpoint → `teams` (all seasons)
8. `conferences` endpoint → `conferences` (all seasons)

### 3.4 Historical Collection Schedule

**Week 1-2: Foundation**
- Collect `teams` for all seasons (2015-2024)
- Collect `conferences` for all seasons
- **Requests:** ~10,000

**Week 3-6: Current Season Data**
- Daily sync of all endpoints for current season (2024-25)
- Build up current season baseline
- **Requests:** ~2,000/week = 8,000 total

**Week 7-10: Historical Ratings (Most Important)**
- Backfill `ratings_snapshots` season by season (2015-2024)
- **Requests:** ~40,000

**Week 11-14: Historical Four Factors**
- Backfill `four_factors` season by season
- **Requests:** ~40,000

**Ongoing: Archive Snapshots**
- Collect key archive dates (preseason, mid-season, postseason)
- **Requests:** ~20,000 over time

### 3.5 Implementation Approach

**Daily Sync (Current Season):**
```python
# Run at 6 AM daily
def daily_sync():
    endpoints = ["ratings", "four-factors", "pointdist", "misc-stats", "height"]
    for endpoint in endpoints:
        sync_endpoint_for_current_season(endpoint)
        time.sleep(6)  # Rate limit
```

**Historical Backfill (Background):**
```python
# Run as separate process, spread over weeks
def backfill_historical_data():
    seasons = range(2015, 2025)
    for season in seasons:
        for endpoint in PRIORITY_ENDPOINTS:
            backfill_season_endpoint(season, endpoint)
            # Respect rate limits aggressively
            time.sleep(6)
```

## Phase 4: Data Validation

### 4.1 Validation Rules

After collection, validate:
1. **Completeness:** All teams present for each date
2. **Consistency:** Rankings sum correctly (1 to N)
3. **Reasonableness:** Metrics within expected ranges
4. **Relationships:** Team IDs match across tables

### 4.2 Validation Queries

```sql
-- Check for missing teams on a date
SELECT COUNT(DISTINCT team_id) as team_count, snapshot_date
FROM ratings_snapshots
GROUP BY snapshot_date
HAVING team_count < 350;  -- Expect ~360 teams

-- Check for rank gaps
SELECT season, snapshot_date,
       COUNT(*) as teams_with_rank,
       MAX(rank_adj_em) as highest_rank
FROM ratings_snapshots
WHERE rank_adj_em IS NOT NULL
GROUP BY season, snapshot_date
HAVING highest_rank != teams_with_rank;

-- Check for unreasonable values
SELECT * FROM ratings_snapshots
WHERE adj_em > 50 OR adj_em < -50  -- Extreme values
   OR adj_oe > 150 OR adj_oe < 60
   OR adj_de > 150 OR adj_de < 60;
```

## Migration Script

```python
# src/kenp0m_sp0rts_analyzer/kenpom/migrate_schema.py
"""
Database schema migration to add missing tables and fields.
Run once to update existing database.
"""

def apply_schema_updates(db_path: str = "data/kenpom.db"):
    """Apply schema updates to existing database."""
    updates = [
        # Add conferences table
        CONFERENCES_TABLE_SQL,

        # Add archive_ratings table
        ARCHIVE_RATINGS_TABLE_SQL,

        # Update teams table
        """
        ALTER TABLE teams ADD COLUMN IF NOT EXISTS season INTEGER;
        ALTER TABLE teams ADD COLUMN IF NOT EXISTS arena TEXT;
        ALTER TABLE teams ADD COLUMN IF NOT EXISTS arena_city TEXT;
        ALTER TABLE teams ADD COLUMN IF NOT EXISTS arena_state TEXT;
        """,

        # Remove incomplete table
        "DROP TABLE IF EXISTS efficiency_tempo;",
    ]

    with DatabaseManager(db_path).transaction() as conn:
        for sql in updates:
            conn.executescript(sql)

    print("Schema updates applied successfully!")
```

## XGBoost Feature Enhancement

With historical archive data, we can add powerful features:

### New Features from Archive Data
1. **Trend Features:**
   - `rating_momentum`: Change in AdjEM over last 30 days
   - `rating_volatility`: Standard deviation of ratings
   - `peak_rating_gap`: Current rating vs season peak

2. **Schedule Context:**
   - `recent_opponent_quality`: Avg opponent AdjEM last 5 games
   - `fatigue_factor`: Games played in last 7 days

3. **Historical Performance:**
   - `preseason_vs_current`: Rating improvement since preseason
   - `consistency_score`: Variance in performance

## Summary

**Key Changes:**
- ✅ Add `conferences` table
- ✅ Add `archive_ratings` table
- ✅ Update `teams` table with arena fields
- ✅ Remove incomplete `efficiency_tempo` table
- ✅ Implement respectful rate-limited historical collection
- ✅ Prioritize data critical for XGBoost training
- ✅ Validate data quality at each step

**Timeline:**
- Schema updates: 1 day
- Current season sync: Ongoing daily
- Historical backfill: 14 weeks (conservative, respectful)

**Estimated Requests:**
- Total: ~100,000 requests over 14 weeks
- Average: ~1,000 requests/day
- Rate: ~10 requests/minute (very conservative)

**Benefits:**
- Preserve existing data and code
- Add powerful historical features for XGBoost
- Maintain data integrity
- Respect KenPom's servers
