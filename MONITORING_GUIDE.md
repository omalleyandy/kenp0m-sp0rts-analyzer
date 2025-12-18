# Comprehensive Monitoring & Analysis System

## Overview

This system implements the **Billy Walters approach** to sports betting:
1. **Analyze games statistically BEFORE lines are posted** (KenPom analysis)
2. **Identify edges when lines appear** (compare to Vegas)
3. **Track line movement** from opening to close
4. **Calculate Closing Line Value (CLV)** for performance tracking

## System Components

### 1. Continuous Monitoring (`monitor_and_analyze.py`)
**Purpose**: Automatically checks overtime.ag for college basketball availability and triggers analysis

**Features**:
- Checks every X minutes (configurable, default 30)
- Captures opening lines when games first appear
- Tracks line movement throughout the day
- Stores all data in SQLite database
- Generates comprehensive reports

**Usage**:
```bash
# Start monitoring (runs indefinitely)
uv run python monitor_and_analyze.py

# Check every 15 minutes (faster)
uv run python monitor_and_analyze.py --interval 15

# Run one check and exit
uv run python monitor_and_analyze.py --check-once
```

### 2. KenPom Pre-Game Analyzer (`scripts/analysis/kenpom_pregame_analyzer.py`)
**Purpose**: Analyze games statistically BEFORE Vegas lines post

**Features**:
- Uses KenPom API to get today's game schedule
- Runs comprehensive matchup analysis for each game
- Identifies statistical edges (Four Factors, tempo, size, experience)
- Generates predictions with confidence levels
- Exports analysis for edge detection

**Usage**:
```bash
# Analyze today's games
uv run python scripts/analysis/kenpom_pregame_analyzer.py

# Analyze specific date
uv run python scripts/analysis/kenpom_pregame_analyzer.py --date 2025-12-19

# Save analysis for later comparison
uv run python scripts/analysis/kenpom_pregame_analyzer.py -o data/kenpom_analysis_2025-12-19.json

# Analyze single game
uv run python scripts/analysis/kenpom_pregame_analyzer.py --game "Duke" "North Carolina"
```

**Output**:
```json
{
  "date": "2025-12-19",
  "game_count": 50,
  "games": [
    {
      "matchup": "Duke @ North Carolina",
      "kenpom_prediction": {
        "home_win_prob": 0.68,
        "predicted_margin": 4.5,
        "predicted_total": 152.3,
        "confidence": "high"
      },
      "four_factors": {
        "efg_advantage": 0.08,
        "to_advantage": -0.02,
        "or_advantage": 0.05,
        "ft_advantage": 0.03
      },
      "edges": [
        {
          "type": "four_factors",
          "factor": "eFG% Edge",
          "magnitude": 0.08,
          "significance": "high"
        }
      ]
    }
  ],
  "summary": {
    "high_confidence_picks": 8,
    "total_edges": 24
  }
}
```

### 3. Edge Detector (`scripts/analysis/edge_detector.py`)
**Purpose**: Compare KenPom predictions to Vegas lines and identify betting opportunities

**Features**:
- Fuzzy team name matching (handles different naming conventions)
- Calculates spread edge and total edge
- Generates betting recommendations with confidence levels
- Filters by minimum edge threshold
- Exports markdown reports

**Usage**:
```bash
# Detect edges for today
uv run python scripts/analysis/edge_detector.py

# Specific date
uv run python scripts/analysis/edge_detector.py --date 2025-12-19

# Only show games with 3+ point edge
uv run python scripts/analysis/edge_detector.py --min-edge 3.0

# Export report
uv run python scripts/analysis/edge_detector.py -o reports/edge_report_2025-12-19.md
```

**Example Output**:
```markdown
# College Basketball Edge Detection Report

Date: 2025-12-19

## Recommended Bets

### 1. Duke @ North Carolina

**BET: North Carolina -6.5**
- Edge: 2.5 points
- Confidence: HIGH
- Reason: KenPom predicts -4.0, Vegas has -6.5

**Analysis:**
- KenPom Margin: +4.5
- Vegas Spread: +6.5
- Spread Edge: 2.0
- KenPom Total: 152.3
- Vegas Total: 148.5
- Total Edge: 3.8
```

### 4. Timing Database (`overtime_timing.py`)
**Purpose**: Track when odds are first posted and analyze timing patterns

**Features**:
- Captures all API responses from overtime.ag
- Tracks when games first appear
- Discovers timestamp fields in API responses
- Calculates lead times (time between post and tip-off)
- Identifies optimal capture windows

**Usage**:
```bash
# Analyze collected timing data
uv run python scripts/scrapers/monitor_overtime_timing.py --analyze

# Quick summary
uv run python scripts/scrapers/monitor_overtime_timing.py --summary

# Continuous monitoring (24 hours, 30 min intervals)
uv run python scripts/scrapers/monitor_overtime_timing.py -i 30 -d 24
```

## Complete Workflow

### Phase 1: Morning - Pre-Game Analysis (8:00 AM)
```bash
# 1. Run KenPom analysis for today's games
uv run python scripts/analysis/kenpom_pregame_analyzer.py \
  -o data/kenpom_analysis_$(date +%Y-%m-%d).json

# Output: Your predictions BEFORE Vegas sets lines
```

### Phase 2: Midday - Monitor for Lines (11:00 AM - 6:00 PM)
```bash
# 2. Start continuous monitoring for line availability
uv run python monitor_and_analyze.py --interval 15

# This will:
# - Check overtime.ag every 15 minutes
# - When college basketball appears, capture OPENING lines
# - Continue tracking line movement every 15 minutes
# - Store all data in database
```

### Phase 3: Afternoon - Edge Detection (When Lines Post)
```bash
# 3. Compare KenPom to Vegas (automatic in monitor, or manual)
uv run python scripts/analysis/edge_detector.py \
  --min-edge 2.5 \
  -o reports/edge_report_$(date +%Y-%m-%d).md

# Output: Betting recommendations with edges
```

### Phase 4: Pre-Game - Final Analysis (30 min before games)
```bash
# 4. Capture closing lines
uv run python capture_odds_today.py
# Move file: vegas_lines.json -> vegas_lines_close.json

# 5. Generate final edge report
uv run python scripts/analysis/edge_detector.py -o reports/final_edges.md
```

### Phase 5: Post-Game - CLV Analysis (Next Day)
```bash
# 6. Calculate closing line value
uv run python scripts/analysis/compare_predictions.py --clv

# Shows:
# - Which bets had positive CLV
# - Average edge captured
# - Win rate vs CLV correlation
```

## Data Storage

### Directory Structure
```
data/
├── overtime_monitoring/
│   └── overtime_odds.db           # Timing & API capture database
├── historical_odds.db              # Long-term odds tracking
├── vegas_lines_open.json          # Opening lines
├── vegas_lines_current.json       # Current lines
├── vegas_lines_close.json         # Closing lines
├── kenpom_analysis_YYYY-MM-DD.json  # Pre-game analysis
└── screenshots/                   # Navigation diagnostics

reports/
└── edge_report_YYYY-MM-DD.md      # Daily edge reports
```

### Database Schema

**overtime_odds.db**:
- `api_snapshots`: All API responses captured
- `game_tracking`: When games first appeared
- `timestamp_fields`: Discovered timestamp fields

**historical_odds.db**:
- `odds_snapshots`: All line captures (open/current/close)
- `games`: Game results
- `predictions`: Your predictions for CLV tracking

## Key Metrics

### Edge Quality
- **High Edge**: 3.0+ points → High confidence bet
- **Medium Edge**: 2.0-2.9 points → Consider bet
- **Low Edge**: 1.0-1.9 points → Monitor only
- **No Edge**: <1.0 points → Pass

### CLV Targets
- **Excellent**: +2.0 average CLV
- **Good**: +1.0 to +2.0 CLV
- **Acceptable**: +0.5 to +1.0 CLV
- **Poor**: <+0.5 CLV

### Win Rate vs CLV
- **Positive CLV with 52%+ win rate** = Long-term profitable
- **Positive CLV with <52% win rate** = Variance, keep betting
- **Negative CLV** = Strategy needs adjustment

## Troubleshooting

### College Basketball Not Available
**Issue**: overtime.ag doesn't show college basketball option

**Solutions**:
1. Check site manually: https://overtime.ag/sports
2. Try different time of day (lines post 12-24 hours before games)
3. Use alternative odds source (see below)

### Alternative Odds Sources
If overtime.ag doesn't have college basketball:

1. **Covers.com** (already have scraper)
   ```bash
   uv run python scripts/scrapers/scrape_covers_injuries.py
   ```

2. **OddsAPI.io** (paid API)
   ```python
   # Add to .env
   ODDS_API_KEY=your_key
   ```

3. **Action Network, ESPN BET, DraftKings**

### Team Name Matching Issues
**Issue**: Edge detector can't match teams between KenPom and Vegas

**Solution**: Check team name mapping in `edge_detector.py`:
```python
# Add custom mappings
TEAM_ALIASES = {
    'St. Mary's': "Saint Mary's",
    'UConn': 'Connecticut',
    # Add more as needed
}
```

## Performance Tracking

### Daily Routine
1. **Morning**: Run KenPom analysis
2. **Afternoon**: Start monitoring
3. **When lines post**: Check edges, place bets
4. **Pre-game**: Capture closing lines
5. **Next day**: Calculate CLV

### Weekly Review
```bash
# Generate weekly CLV report
uv run python scripts/analysis/weekly_clv_report.py --week 2025-W51

# Shows:
# - Total bets placed
# - Average CLV
# - Win rate
# - ROI
```

## Advanced Features

### Automated Bet Placement (Future)
```python
# Integration with sportsbooks API
from kenp0m_sp0rts_analyzer.automated_betting import place_bet

# Only place bets with high confidence + 3+ point edge
for edge in high_confidence_edges:
    if edge['edge'] >= 3.0:
        place_bet(edge, unit_size=1.0)
```

### Machine Learning Enhancement
```python
# Train predictor on historical data
from kenp0m_sp0rts_analyzer.prediction import GamePredictor

predictor = GamePredictor()
predictor.fit(historical_games, margins, totals)

# Use in edge detection
ml_prediction = predictor.predict(duke_stats, unc_stats)
```

## Next Steps

1. **Start monitoring**:
   ```bash
   uv run python monitor_and_analyze.py
   ```

2. **Wait for college basketball to appear** on overtime.ag

3. **When games are available**:
   - Opening lines captured automatically
   - Edge report generated automatically
   - Line movement tracked continuously

4. **Review reports and place bets** based on edges

5. **Track CLV** to measure success

---

**Questions?** Check the project documentation:
- `docs/PROJECT_SCOPE.md` - Project boundaries
- `docs/KENPOM_ANALYTICS_GUIDE.md` - Analytics methodology
- `docs/EDGE_VALIDATION_GUARDRAILS.md` - Betting framework
- `CLAUDE.md` - Development guidelines
