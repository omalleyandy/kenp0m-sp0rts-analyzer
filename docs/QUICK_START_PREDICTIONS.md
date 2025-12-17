# Quick Start: Making Predictions with KenPom Data

**Created**: 2025-12-16
**Purpose**: Get started with KenPom predictions in 5 minutes

---

## Prerequisites

1. **KenPom API Key**: Set environment variable
   ```bash
   # Windows (PowerShell)
   $env:KENPOM_API_KEY = "your-api-key-here"

   # Or add to .env file
   echo "KENPOM_API_KEY=your-api-key-here" >> .env
   ```

2. **Install dependencies**:
   ```bash
   uv sync
   ```

---

## Step 1: Collect Data (Run Once Daily)

```bash
# Collect all KenPom data for 2025 season
uv run python scripts/collect_daily_data.py --season 2025

# Output:
# - Creates data/kenpom_2025_YYYYMMDD_HHMMSS.parquet
# - Creates data/kenpom_2025_latest.parquet (symlink to latest)
# - Shows top 10 teams and momentum leaders
```

**What you get**:
- 360+ teams with 100+ features each
- Core metrics: AdjEM, AdjO, AdjD, AdjTempo
- Four Factors: eFG%, TO%, OR%, FT_Rate
- Point Distribution: 3pt%, 2pt%, FT%
- Size/Experience: Height, years of experience
- Momentum: 30-day change in AdjEM and ranking

**Example Output**:
```
======================================================================
KenPom Data Collection - 2025 Season
======================================================================

[1/6] Fetching team ratings...
      Retrieved 363 teams
[2/6] Fetching Four Factors...
      Retrieved 363 teams
[3/6] Fetching point distribution...
      Retrieved 363 teams
[4/6] Fetching height/experience data...
      Retrieved 363 teams
[5/6] Fetching miscellaneous stats...
      Retrieved 363 teams
[6/6] Fetching historical data (30 days ago)...
      Retrieved 363 teams from 2024-11-16

Merged dataset:
  Teams: 363
  Features: 127
  Memory: 5.32 MB

Saved data to: data/kenpom_2025_20241216_143022.parquet
Saved latest copy to: data/kenpom_2025_latest.parquet

======================================================================
DATA SUMMARY
======================================================================

Top 10 Teams (by AdjEM):
   1. Duke                       AdjEM: +32.18  Record: 10-2
   2. Auburn                     AdjEM: +30.45  Record: 11-1
   3. Tennessee                  AdjEM: +29.87  Record: 11-0
   4. Iowa State                 AdjEM: +27.91  Record: 9-2
   5. Kentucky                   AdjEM: +27.23  Record: 10-2
   6. Gonzaga                    AdjEM: +26.54  Record: 10-3
   7. Florida                    AdjEM: +25.89  Record: 11-0
   8. Alabama                    AdjEM: +25.12  Record: 9-2
   9. Houston                    AdjEM: +24.76  Record: 8-3
  10. Kansas                     AdjEM: +24.31  Record: 9-2

Key Metrics:
  Highest AdjEM: +32.18 (Duke)
  Lowest AdjEM: -24.56 (Mississippi Valley State)
  Avg Tempo: 68.4 possessions/game
  Fastest: 77.2 (North Carolina Central)
  Slowest: 60.1 (Virginia)

Momentum (30-day change):
  Biggest Riser: Tennessee (+5.23 AdjEM)
  Biggest Faller: North Carolina (-3.45 AdjEM)

Data collection complete for 2025 season!
```

---

## Step 2: Make Predictions

### Basic Prediction (Simple)

```bash
uv run python scripts/predict_game.py Duke "North Carolina"
```

**Output**:
```
================================================================================
GAME PREDICTION - 2025 Season
================================================================================

Duke vs North Carolina
Location: Neutral Site

--------------------------------TEAM STATS--------------------------------------

Duke                                     | North Carolina
--------------------------------------------------------------------------------
AdjEM: +32.18 (#1  )                     | AdjEM: +20.15 (#18 )
AdjO:  125.35 (#2  )                     | AdjO:  117.82 (#25 )
AdjD:   93.17 (#5  )                     | AdjD:   97.67 (#42 )
Tempo:  68.20 (#85 )                     | Tempo:  70.15 (#45 )
Record: 10-2                             | Record: 8-4

--------------------------------PREDICTION--------------------------------------

Winner: Duke
Margin: 12.0 points
Confidence: High (85.0%)

Expected Score: 86.5 - 74.5
Predicted Total: 161.0 points
Expected Tempo: 69.2 possessions/game

================================================================================
```

### Home Game Prediction

```bash
uv run python scripts/predict_game.py Duke Virginia --home team1
```

**Output**:
```
Location: Duke (Home)
Winner: Duke
Margin: 13.8 points  # +3.5 for home court
Confidence: High (85.0%)
```

### Detailed Breakdown

```bash
uv run python scripts/predict_game.py Duke Kentucky --detailed
```

**Output includes**:
```
----------------------------------DETAILS---------------------------------------

Efficiency Margin Difference: +4.95
Offensive Advantage: Duke (7.53 pts/100 poss)
Defensive Advantage: Duke (2.58 pts allowed/100 poss)
Pace Control: Kentucky (2.3 poss/game faster)
```

---

## Step 3: Understand the Prediction Formula

### Simple Model (70% accuracy)

```python
Predicted_Margin = Team1_AdjEM - Team2_AdjEM + Home_Court_Advantage
                 = 32.18 - 20.15 + 0.0
                 = 12.0 points
```

**Why this works**:
- **AdjEM** (Adjusted Efficiency Margin) is the single best predictor of team quality
- It represents expected point differential per 100 possessions
- Home court advantage is ~3.5 points in college basketball

### Advanced Model (75-80% accuracy with tempo)

```python
# Expected tempo (average of both teams)
Expected_Tempo = (Duke_Tempo + UNC_Tempo) / 2
               = (68.2 + 70.1) / 2
               = 69.2 possessions/game

# Team scores
Duke_Score = Duke_AdjO × (UNC_AdjD / 100) × (Expected_Tempo / 100)
           = 125.35 × (97.67 / 100) × (69.2 / 100)
           = 86.5 points

UNC_Score = UNC_AdjO × (Duke_AdjD / 100) × (Expected_Tempo / 100)
          = 117.82 × (93.17 / 100) × (69.2 / 100)
          = 74.5 points

Predicted_Total = 86.5 + 74.5 = 161.0 points
```

---

## Step 4: Automate Daily Collection

### Windows (Task Scheduler)

```powershell
# Create scheduled task to run daily at 6:00 AM
$action = New-ScheduledTaskAction -Execute "uv" -Argument "run python scripts/collect_daily_data.py --season 2025" -WorkingDirectory "C:\Users\omall\Documents\python_projects\kenp0m-sp0rts-analyzer"
$trigger = New-ScheduledTaskTrigger -Daily -At 6am
Register-ScheduledTask -Action $action -Trigger $trigger -TaskName "KenPom Daily Data Collection"
```

### Linux/Mac (Cron)

```bash
# Add to crontab
crontab -e

# Run daily at 6:00 AM
0 6 * * * cd /path/to/kenp0m-sp0rts-analyzer && python scripts/collect_daily_data.py --season 2025
```

---

## Step 5: Explore Advanced Features

### Using Your Existing Modules

```python
from kenp0m_sp0rts_analyzer.api_client import KenPomAPI
from kenp0m_sp0rts_analyzer.comprehensive_matchup_analysis import (
    ComprehensiveMatchupAnalyzer
)
from kenp0m_sp0rts_analyzer.report_generator import MatchupReportGenerator

# Get data
api = KenPomAPI()
duke = api.get_team_by_name("Duke", 2025)
unc = api.get_team_by_name("North Carolina", 2025)

# Comprehensive analysis (uses all 7 analytical modules)
analyzer = ComprehensiveMatchupAnalyzer()
analysis = analyzer.analyze_matchup(
    team1_id=duke['TeamID'],
    team2_id=unc['TeamID'],
    season=2025
)

# Generate formatted report
report_gen = MatchupReportGenerator()
report = report_gen.generate_text_report(analysis)
print(report)
```

### Tournament Simulation

```python
from kenp0m_sp0rts_analyzer.tournament_simulator import TournamentSimulator

simulator = TournamentSimulator()
results = simulator.simulate_tournament(
    teams=tournament_teams,  # DataFrame from collect_daily_data.py
    num_simulations=10000
)

print("Championship Probabilities:")
for team, prob in results.championship_probabilities.head(10).items():
    print(f"  {team}: {prob:.1%}")
```

---

## What Each Metric Means

### Core Efficiency Metrics

| Metric | Description | Good Value | Elite Value |
|--------|-------------|------------|-------------|
| **AdjEM** | Efficiency Margin (AdjO - AdjD) | +15 to +20 | +25+ |
| **AdjO** | Offensive Efficiency (pts/100 poss) | 110-115 | 120+ |
| **AdjD** | Defensive Efficiency (pts allowed/100 poss) | 95-100 | <93 |
| **AdjTempo** | Pace (possessions per game) | 68-72 | 75+ (fast) / <65 (slow) |
| **Pythag** | Expected win % based on scoring | 0.75-0.85 | 0.90+ |
| **SOS** | Strength of Schedule | 5-10 | 1-3 |

### Four Factors (Dean Oliver)

| Factor | Offensive | Defensive | Weight | Description |
|--------|-----------|-----------|--------|-------------|
| **Shooting** | eFG_Pct | DeFG_Pct | 40% | Most important factor |
| **Turnovers** | TO_Pct | DTO_Pct | 25% | Lower is better (offense) |
| **Rebounding** | OR_Pct | DR_Pct | 20% | Possession control |
| **Free Throws** | FT_Rate | DFT_Rate | 15% | Getting to the line |

**Good Values** (Offense):
- eFG_Pct: >54% (elite: >56%)
- TO_Pct: <17% (elite: <15%)
- OR_Pct: >32% (elite: >35%)
- FT_Rate: >36% (elite: >40%)

---

## Common Use Cases

### 1. Daily Betting Edge Finder

```python
import pandas as pd

# Load today's predictions
data = pd.read_parquet("data/kenpom_2025_latest.parquet")

# Get today's games from fanmatch endpoint
api = KenPomAPI()
games = api.get_fanmatch("2025-03-15").to_dataframe()

# Find edges where KenPom disagrees with market
for _, game in games.iterrows():
    kenpom_margin = game['PredictedMargin']
    # Compare with sportsbook lines (you'd fetch these from odds API)
    # if abs(kenpom_margin - market_line) > 3:
    #     print(f"EDGE: {game['HomeTeam']} vs {game['AwayTeam']}")
```

### 2. Conference Power Rankings

```python
# Load data
data = pd.read_parquet("data/kenpom_2025_latest.parquet")

# Top conferences by average AdjEM
conf_power = data.groupby('ConfShort')['AdjEM'].agg(['mean', 'count']).sort_values('mean', ascending=False)
print(conf_power.head(10))
```

### 3. Upset Alert Finder

```python
# Find teams with high AdjEM but low rank (undervalued)
data = pd.read_parquet("data/kenpom_2025_latest.parquet")

# Teams with momentum (rising fast)
if 'AdjEM_Momentum' in data.columns:
    risers = data.nlargest(10, 'AdjEM_Momentum')[['TeamName', 'AdjEM_Momentum', 'RankAdjEM']]
    print("Teams with momentum (upset potential):")
    print(risers)
```

---

## Step 6: Validate Betting Edges

**IMPORTANT**: Always validate predictions against market lines before betting!

### Why Validation Matters

KenPom models can be wrong when they disagree with actual game results. Always check:
- Model prediction vs historical averages
- Both should agree on direction (OVER/UNDER)
- Edge should be significant (>5 points for totals)

### Validation Workflow

```bash
# 1. Make prediction
uv run python scripts/predict_game.py "Montana St." "Cal Poly" --home team2

# 2. Validate against market lines
uv run python scripts/validate_edge.py "Montana St." "Cal Poly" \
  --market-spread -3 \
  --market-total 160 \
  --home team2
```

### Example Output

```
================================RECOMMENDATIONS=================================

Spread: PASS (edge too small)
Total: CONFLICT - Model and historical disagree (>10 points)

====================================WARNINGS====================================

WARNING: Manual validation required - check actual game results
```

### When to Bet

| Scenario | Action |
|----------|--------|
| Model + Historical agree, edge >5 pts | ✅ BET |
| Model + Historical agree, edge <5 pts | ⚠️ PASS (edge too small) |
| Model disagrees with historical >10 pts | ❌ CONFLICT - PASS |
| Direction mismatch (OVER vs UNDER) | ❌ PASS |

**Golden Rule**: When in doubt, pass. No bet is better than a bad bet.

### Detailed Guardrails

See `docs/EDGE_VALIDATION_GUARDRAILS.md` for:
- Complete validation workflow
- Data source rules (KenPom only!)
- Edge thresholds and decision trees
- Common pitfalls to avoid

---

## Next Steps

1. ✅ **Data Collection**: Run `collect_daily_data.py` to get fresh data
2. ✅ **Make Predictions**: Use `predict_game.py` for any matchup
3. ✅ **Validate Edges**: Use `validate_edge.py` before betting
4. 📊 **Build Features**: Extend with Four Factors analysis (see `docs/KENPOM_ANALYTICS_GUIDE.md`)
5. 🤖 **Train ML Model**: Use `prediction.py` to train on historical games
6. 📈 **Backtest**: Validate accuracy using archive endpoint
7. 🎯 **Optimize**: Find betting edges and tournament picks

---

## Troubleshooting

### "Team not found" Error

```bash
# Use partial match
python scripts/predict_game.py "North Carolina" Virginia

# Or check exact name
python -c "from kenp0m_sp0rts_analyzer.api_client import KenPomAPI; api = KenPomAPI(); print(api.get_ratings(year=2025).to_dataframe()['TeamName'].sort_values())"
```

### API Key Issues

```bash
# Check if API key is set
echo $env:KENPOM_API_KEY  # Windows PowerShell
echo $KENPOM_API_KEY      # Linux/Mac

# Test API connection
python -c "from kenp0m_sp0rts_analyzer.api_client import KenPomAPI; api = KenPomAPI(); print(api.get_ratings(year=2025))"
```

### Data Too Old

```bash
# Re-collect fresh data
python scripts/collect_daily_data.py --season 2025

# Check data age
python -c "import pandas as pd; df = pd.read_parquet('data/kenpom_2025_latest.parquet'); print(df['DataThrough'].iloc[0])"
```

---

## Resources

- **Full Analytics Guide**: `docs/KENPOM_ANALYTICS_GUIDE.md`
- **API Documentation**: `docs/KENPOM_API.md`
- **Reverse Engineering Findings**: `docs/API_REVERSE_ENGINEERING_FINDINGS.md`
- **Examples Directory**: `examples/` (14 working demos)

---

**Ready to make predictions? Start with Step 1! 🏀**
