# KenPom Module Architecture

## Overview

The `kenpom` module provides a complete solution for integrating KenPom basketball analytics data into the ML betting system. It follows a layered architecture with clear separation of concerns.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    Kenpom API                               │
│                 (api_client.py)                             │
└────────────────────────┬────────────────────────────────────┘
                         │
        ┌────────────────┼────────────────┐
        │                │                │
    ┌───▼────┐      ┌───▼────┐      ┌───▼────┐
    │ Batch  │      │ Real-  │      │Archive │
    │ Jobs   │      │ time   │      │Loader  │
    │(Daily) │      │Monitor │      │(2023+) │
    │        │      │        │      │        │
    │batch_  │      │realtime│      │archive_│
    │scheduler│      │_monitor│      │loader  │
    └───┬────┘      └───┬────┘      └───┬────┘
        │                │                │
        └────────────────┼────────────────┘
                         │
        ┌────────────────▼────────────────┐
        │      Data Validation &          │
        │    Error Handling Layer         │
        │   (validators.py, exceptions.py)│
        └────────────────┬────────────────┘
                         │
        ┌────────────────▼────────────────┐
        │      Repository Layer           │
        │       (repository.py)           │
        └────────────────┬────────────────┘
                         │
        ┌────────────────▼────────────────┐
        │      SQLite3 Database           │
        │  (Teams, Ratings, Historical,   │
        │   Predictions, Game Stats)      │
        │       (database.py)             │
        └────────────────┬────────────────┘
                         │
        ┌────────────────▼────────────────┐
        │    Unified Service Interface    │
        │   (api.py - KenPomService)      │
        │                                 │
        │    ML Feature Engineering       │
        │   & Predictive Modeling         │
        │      (models.py)                │
        └─────────────────────────────────┘
```

## Module Structure

```
src/kenp0m_sp0rts_analyzer/kenpom/
├── __init__.py          # Package exports & factory functions
├── api.py               # KenPomService - unified interface
├── archive_loader.py    # Historical data (2023+)
├── batch_scheduler.py   # Daily sync orchestration
├── database.py          # SQLite schema & management
├── exceptions.py        # Custom exception hierarchy
├── models.py            # Pydantic data models
├── realtime_monitor.py  # Live change monitoring
├── repository.py        # Data access layer (CRUD)
└── validators.py        # Data validation & sanitization
```

## Layer Descriptions

### 1. API Layer (`api_client.py`)
**Location**: `src/kenp0m_sp0rts_analyzer/api_client.py` (existing)

The low-level HTTP client for KenPom API:
- Authentication handling
- Request/response management
- Rate limiting
- Error handling

### 2. Data Collection Layer

#### BatchScheduler (`batch_scheduler.py`)
Orchestrates daily data synchronization:
- Priority-based task execution
- Retry logic with configurable attempts
- Callback system for monitoring
- CLI interface for cron integration

```python
from kenp0m_sp0rts_analyzer.kenpom import BatchScheduler

scheduler = BatchScheduler()
result = scheduler.run_daily_workflow()
```

#### ArchiveLoader (`archive_loader.py`)
Handles historical data from 2023+:
- Date range backfilling
- Season-at-a-time loading
- Incremental updates
- Rate limiting for API protection

```python
from kenp0m_sp0rts_analyzer.kenpom import ArchiveLoader

loader = ArchiveLoader()
result = loader.backfill_season(2024)
```

#### RealtimeMonitor (`realtime_monitor.py`)
Tracks rating changes in near real-time:
- Decorator-based callback registration
- Background thread polling
- Configurable change thresholds
- Significant move detection

```python
from kenp0m_sp0rts_analyzer.kenpom import RealtimeMonitor

monitor = RealtimeMonitor()

@monitor.on_significant_move(threshold=2.0)
def handle_move(event):
    print(f"{event.team_name}: {event.change_amount:+.1f}")

monitor.start(poll_interval=300)
```

### 3. Validation Layer (`validators.py`)
Ensures data quality throughout the pipeline:
- Range validation for all metrics
- Required field checks
- Consistency validation (AdjEM = AdjOE - AdjDE)
- Statistical anomaly detection (3σ thresholds)
- Data freshness checks

### 4. Error Handling (`exceptions.py`)
Structured exception hierarchy:
```
KenPomError (base)
├── DataValidationError
├── DatabaseError
├── SyncError
├── RateLimitError
├── ArchiveNotAvailableError
├── TeamNotFoundError
└── ConfigurationError
```

### 5. Repository Layer (`repository.py`)
Clean data access abstraction:
- CRUD operations for all entities
- Transaction management
- Query optimization
- Matchup data aggregation

### 6. Database Layer (`database.py`)
SQLite schema and management:
- 8 core tables with indexes
- Schema versioning for migrations
- Backup/restore operations
- Connection pooling

### 7. Data Models (`models.py`)
Pydantic models with validation:
- `TeamRating` - Core efficiency metrics
- `FourFactors` - Dean Oliver's factors
- `PointDistribution` - Scoring breakdown
- `GamePrediction` - Prediction tracking
- `AccuracyReport` - Performance metrics

### 8. Service Interface (`api.py`)
Unified entry point - `KenPomService`:
- All operations in one place
- Lazy API initialization
- ML feature generation
- Accuracy tracking

## Database Schema

### Tables

| Table | Purpose | Key Fields |
|-------|---------|------------|
| teams | Team info | team_id, team_name, conference |
| ratings_snapshots | Daily ratings | UNIQUE(snapshot_date, team_id) |
| four_factors | Four Factors | efg_pct_off/def, to_pct_off/def |
| point_distribution | Scoring % | ft_pct, two_pct, three_pct |
| height_experience | Physical stats | avg_height, experience |
| game_predictions | Prediction tracking | predicted_margin, actual_margin |
| accuracy_metrics | Daily summaries | mae, rmse, ats_percentage |
| sync_history | Sync logs | endpoint, status, records |

### Indexes (15+)
- `idx_ratings_date` - Fast date lookup
- `idx_ratings_team` - Team history queries
- `idx_predictions_pending` - Unresolved games
- etc.

## Usage Patterns

### Daily Workflow
```python
from kenp0m_sp0rts_analyzer.kenpom import BatchScheduler

scheduler = BatchScheduler()

# Check if sync is needed
should_run, reason = scheduler.check_should_run()

if should_run:
    result = scheduler.run_daily_workflow()
    print(f"Synced {result.total_records} records")
```

### ML Feature Generation
```python
from kenp0m_sp0rts_analyzer.kenpom import KenPomService

service = KenPomService()

# Generate features for XGBoost model
features = service.get_features_for_game(
    team1_id=73,  # Duke
    team2_id=152,  # UNC
    home_team_id=73
)

# Features include:
# - adj_em_diff, adj_oe_diff, adj_de_diff
# - efg_diff, to_diff, or_diff
# - three_pct_diff, pythag_diff
# - home_advantage
# ... (27 features total)
```

### Accuracy Tracking
```python
from kenp0m_sp0rts_analyzer.kenpom import KenPomService

service = KenPomService()

# Save prediction
pred_id = service.save_prediction(prediction)

# After game completes
service.update_prediction_result(pred_id, result)

# Generate report
report = service.get_accuracy_report(days=30)
print(f"ATS: {report.ats_percentage:.1%}")
print(f"CLV: {report.avg_clv:+.1f} points")
```

## Integration with Existing Code

### With XGBoostFeatureEngineer
The `get_features_for_game()` method produces features compatible with the existing 27-feature XGBoost model:

```python
# In prediction.py
features = service.get_features_for_game(team1_id, team2_id)
prediction = model.predict(features)
```

### With Vegas Lines
```python
# Save prediction with Vegas lines
pred = GamePrediction(
    game_date=date.today(),
    team1_id=73,
    team2_id=152,
    predicted_margin=5.5,
    vegas_spread=-3.5,  # Store opening line
    vegas_total=145.5,
    ...
)
```

## Configuration

### Environment Variables
- `KENPOM_API_KEY` - Required for API access

### Database Path
Default: `data/kenpom.db`

Override in any factory function:
```python
service = KenPomService(db_path="custom/path/kenpom.db")
```

## CLI Usage

### Batch Scheduler
```bash
# Check if sync needed
python -m kenp0m_sp0rts_analyzer.kenpom.batch_scheduler --check

# Run sync
python -m kenp0m_sp0rts_analyzer.kenpom.batch_scheduler

# Force sync
python -m kenp0m_sp0rts_analyzer.kenpom.batch_scheduler --force

# Status
python -m kenp0m_sp0rts_analyzer.kenpom.batch_scheduler --status
```

### Cron Integration
```bash
# Add to crontab for 6 AM daily sync
0 6 * * * cd /path/to/project && python -m kenp0m_sp0rts_analyzer.kenpom.batch_scheduler
```

## Testing

```bash
# Run module tests
pytest tests/test_kenpom/ -v

# With coverage
pytest tests/test_kenpom/ --cov=kenp0m_sp0rts_analyzer.kenpom
```

## Next Steps

1. **Write Unit Tests** - Cover each component
2. **Backfill 2023-2025** - Load historical data
3. **Deploy Daily Sync** - Set up cron/systemd
4. **Monitor Accuracy** - Track CLV over time
5. **Optimize Queries** - Add more indexes if needed
