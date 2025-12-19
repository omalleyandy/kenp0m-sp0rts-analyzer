# KenPom Sports Analyzer - Claude Context

## Project Overview

NCAA Division I Men's Basketball analytics system combining KenPom statistical analysis with XGBoost machine learning for game predictions and betting edge detection.

**Primary Objective**: Achieve 54%+ against-the-spread win rate (profitable after vig)  
**Gold Standard Metric**: Closing Line Value (CLV)

## Project Structure

```
kenp0m-sp0rts-analyzer/
├── src/kenp0m_sp0rts_analyzer/
│   ├── __init__.py              # Package exports
│   ├── integrated_predictor.py  # Unified KenPom + XGBoost
│   ├── api_client.py            # KenPom API client (40 KB)
│   ├── prediction.py            # XGBoost prediction (45 KB, 27 features)
│   ├── analysis.py              # Basic matchup analysis
│   ├── client.py                # Legacy kenpompy client
│   ├── comprehensive_matchup_analysis.py
│   ├── four_factors_matchup.py  # Dean Oliver Four Factors
│   ├── tournament_simulator.py  # Monte Carlo bracket simulation
│   ├── report_generator.py      # Formatted reports
│   ├── luck_regression.py       # Luck adjustment analysis
│   ├── historical_odds_db.py    # Vegas lines database
│   ├── overtime_scraper.py      # Overtime.ag scraper
│   ├── mcp_server.py            # MCP server integration
│   ├── models.py                # Pydantic models
│   └── kenpom/                  # KenPom data module (10 files, 155 KB)
│       ├── api.py               # KenPomService
│       ├── batch_scheduler.py   # Daily sync automation
│       ├── archive_loader.py    # Historical data (2023+)
│       ├── realtime_monitor.py  # Live rating changes
│       ├── database.py          # SQLite management (8 tables)
│       ├── repository.py        # Data access layer (30 KB)
│       ├── validators.py        # Data validation
│       ├── models.py            # Pydantic models
│       └── exceptions.py        # Custom exceptions
├── scripts/
│   ├── fetch_and_train_xgboost.py  # Train XGBoost model
│   ├── predict_game.py             # CLI game prediction
│   └── collect_daily_data.py       # Daily data collection
├── examples/
│   ├── basic_usage.py
│   └── integrated_prediction_demo.py
├── docs/
│   ├── KENPOM_API.md               # API documentation
│   ├── KENPOM_MODULE_ARCHITECTURE.md  # System architecture
│   └── SETUP_GUIDE.md              # Installation guide
├── tests/
├── data/                           # Data files (kenpom.db, models/)
└── analyze_todays_games.py         # Main daily analysis script
```

## Key Components

### IntegratedPredictor (Primary Interface)
```python
from kenp0m_sp0rts_analyzer import IntegratedPredictor

predictor = IntegratedPredictor()

# Predict game
result = predictor.predict_game("Duke", "North Carolina")
print(f"Margin: {result.predicted_margin:+.1f}")

# With Vegas lines for edge detection
result = predictor.predict_game(
    "Duke", "UNC",
    vegas_spread=-3.5,
    vegas_total=145.0
)
if result.has_spread_edge:
    print(f"Edge: {result.edge_vs_spread:+.1f} points")
```

### XGBoost Model (27 Features)
- Core: adj_em_diff, adj_oe_diff, adj_de_diff, adj_tempo_diff
- Advanced: pythag_diff, luck_diff, sos_diff, ncsos_diff
- Four Factors: efg_diff, to_diff, or_diff, ft_rate_diff
- Shooting: three_pct_diff, two_pct_diff, ft_pct_diff
- Context: home_advantage, avg_tempo

### KenPom Module
- **KenPomService**: Unified data access
- **BatchScheduler**: Daily sync (6 AM recommended)
- **ArchiveLoader**: Historical data backfill (2023+)
- **RealtimeMonitor**: Live rating change detection

### Database Schema
8 tables: teams, ratings_snapshots, four_factors, point_distribution, 
height_experience, game_predictions, accuracy_metrics, sync_history

## Commands

```bash
# Daily workflow
python -m kenp0m_sp0rts_analyzer.kenpom.batch_scheduler

# Train XGBoost model
python scripts/fetch_and_train_xgboost.py

# Predict specific game
python scripts/predict_game.py "Duke" "North Carolina"

# Analyze today's games
python analyze_todays_games.py

# Run tests
uv run pytest tests/ -v
```

## Environment

```env
KENPOM_API_KEY=your_api_key_here  # Required for KenPom access
```

## Performance Targets

- **ATS Win Rate**: 54%+ (profitable after -110 vig)
- **Edge Threshold**: ≥2 points spread, ≥3 points total
- **CLV**: Primary metric for long-term success
