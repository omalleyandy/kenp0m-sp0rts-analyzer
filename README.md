# KenPom Sports Analyzer

Advanced NCAA Division I Men's Basketball analytics system combining KenPom statistical analysis with XGBoost machine learning for game predictions and betting edge detection.

## Features

- **KenPom Data Integration**: Complete API client with historical archiving (2023+)
- **XGBoost Predictions**: 27-feature enhanced model for spread and total predictions  
- **Edge Detection**: Compare predictions against Vegas lines to find value
- **Daily Automation**: Batch scheduler for automated data sync
- **Real-time Monitoring**: Track rating changes and trigger alerts
- **Accuracy Tracking**: CLV-based performance metrics

## Quick Start

```python
from kenp0m_sp0rts_analyzer import IntegratedPredictor

# Initialize predictor
predictor = IntegratedPredictor()

# Predict a game
result = predictor.predict_game("Duke", "North Carolina")
print(f"Predicted margin: {result.predicted_margin:+.1f}")
print(f"Win probability: {result.win_probability:.1%}")

# With Vegas lines for edge detection
result = predictor.predict_game(
    "Duke", "UNC",
    vegas_spread=-3.5,
    vegas_total=145.0
)
if result.has_spread_edge:
    print(f"Edge vs spread: {result.edge_vs_spread:+.1f} points")
```

## Installation

```bash
# Clone repository
git clone https://github.com/omalleyandy/kenp0m-sp0rts-analyzer.git
cd kenp0m-sp0rts-analyzer

# Install with uv
uv sync

# Set up environment
cp .env.example .env
# Add your KENPOM_API_KEY to .env
```

## Core Components

### IntegratedPredictor
Unified interface combining KenPom data + XGBoost ML:

```python
from kenp0m_sp0rts_analyzer import IntegratedPredictor

predictor = IntegratedPredictor()

# Single game prediction
result = predictor.predict_game("Kansas", "Kentucky")

# Batch predictions
games = [("Duke", "UNC"), ("Kansas", "Kentucky")]
results = predictor.predict_batch(games)

# Find edges vs Vegas
edges = predictor.find_edges(games_with_lines, min_spread_edge=2.0)
```

### KenPom Module
Complete data management system:

```python
from kenp0m_sp0rts_analyzer.kenpom import (
    KenPomService,
    BatchScheduler,
    ArchiveLoader,
)

# Get latest ratings
service = KenPomService()
ratings = service.get_latest_ratings()

# Daily sync
scheduler = BatchScheduler()
result = scheduler.run_daily_workflow()

# Historical backfill
loader = ArchiveLoader()
loader.backfill_season(season=2024)
```

### XGBoost Prediction
27-feature enhanced model:

```python
from kenp0m_sp0rts_analyzer import XGBoostPredictor

predictor = XGBoostPredictor()
predictor.load_model("models/xgboost_model.json")

result = predictor.predict(features)
print(f"Margin: {result.margin:+.1f}")
print(f"Total: {result.total:.1f}")
```

## Project Structure

```
kenp0m-sp0rts-analyzer/
├── src/kenp0m_sp0rts_analyzer/
│   ├── __init__.py              # Package exports
│   ├── integrated_predictor.py  # Unified KenPom + XGBoost
│   ├── api_client.py            # KenPom API client
│   ├── prediction.py            # XGBoost prediction system
│   ├── luck_regression.py       # Luck regression analysis
│   ├── kenpom/                  # KenPom data module
│   │   ├── api.py               # KenPomService
│   │   ├── batch_scheduler.py   # Daily sync
│   │   ├── archive_loader.py    # Historical data
│   │   ├── realtime_monitor.py  # Change detection
│   │   ├── database.py          # SQLite management
│   │   ├── repository.py        # Data access layer
│   │   ├── validators.py        # Data validation
│   │   └── models.py            # Pydantic models
│   └── ...
├── scripts/                     # Utility scripts
├── examples/                    # Usage examples
├── tests/                       # Test suite
├── docs/                        # Documentation
└── data/                        # Data files
```

## Key Metrics

The system focuses on **Closing Line Value (CLV)** as the gold standard for long-term betting success:

- **Target**: 54%+ ATS win rate (profitable after vig)
- **Edge Detection**: ≥2 point spread edge, ≥3 point total edge
- **Confidence Intervals**: Track prediction uncertainty

## Documentation

See `docs/` for detailed documentation:

- `KENPOM_API.md` - API client documentation
- `KENPOM_MODULE_ARCHITECTURE.md` - System architecture
- `KENPOM_ANALYTICS_GUIDE.md` - Analytics methodology
- `EDGE_VALIDATION_GUARDRAILS.md` - Edge validation rules
- `MATCHUP_ANALYSIS_FRAMEWORK.md` - Matchup analysis
- `SETUP_GUIDE.md` - Installation guide
- `QUICK_START_PREDICTIONS.md` - Prediction quickstart

## Configuration

Create `.env` file:

```env
KENPOM_API_KEY=your_api_key_here
```

## Testing

```bash
# Run all tests
uv run pytest tests/ -v

# With coverage
uv run pytest tests/ --cov=kenp0m_sp0rts_analyzer
```

## License

MIT License - see LICENSE file.

## Author

Andy O'Malley
