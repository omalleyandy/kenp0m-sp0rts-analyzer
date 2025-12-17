# CLAUDE.md - AI Assistant Guidelines for KenPom Sports Analyzer

## Project Overview

This is a Python project for NCAA Division I Men's Basketball analytics using KenPom data. It provides comprehensive tools for analyzing college basketball through KenPom's advanced efficiency metrics, machine learning predictions, and statistical modeling.

## Quick Reference

### Build & Test Commands
```bash
# Install dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run tests with coverage
pytest --cov=src/kenp0m_sp0rts_analyzer

# Type checking
mypy src/

# Linting
ruff check src/ tests/
ruff format src/ tests/
```

### Project Structure
```
kenp0m-sp0rts-analyzer/
├── src/kenp0m_sp0rts_analyzer/     # 22 modules
│   ├── Core API & Data Access:
│   │   ├── api_client.py           # Official KenPom API (recommended)
│   │   ├── browser.py / scraper.py # Web scraping (fallback)
│   │   └── client.py               # kenpompy wrapper
│   ├── Analytics Modules:
│   │   ├── TIER 1: four_factors_matchup.py, point_distribution_analysis.py,
│   │   │          defensive_analysis.py, tempo_analysis.py
│   │   ├── TIER 2: size_athleticism_analysis.py, experience_chemistry_analysis.py
│   │   └── conference_analytics.py, player_impact.py
│   ├── Prediction & Simulation:
│   │   ├── prediction.py           # ML prediction + backtesting
│   │   └── tournament_simulator.py # Bracket simulation
│   ├── Integration:
│   │   ├── comprehensive_matchup_analysis.py  # Combines all TIER modules
│   │   └── report_generator.py     # Generates matchup reports
│   └── Core Infrastructure:
│       ├── models.py, utils.py, mcp_server.py
│       └── api_docs_reverse_engineer.py
├── scripts/                         # 7 operational scripts
│   ├── collect_daily_data.py, predict_game.py, validate_edge.py
│   └── reverse_engineer_api_docs.py, validate_tempo_features.py
├── examples/                        # 14 demo scripts
│   ├── comprehensive_integration_demo.py, tournament_simulator_demo.py
│   ├── TIER demos: four_factors_matchup_demo.py, defensive_matchup_demo.py
│   └── stealth_scraper.py, basic_usage.py
├── docs/                            # 17 documentation files
│   ├── API: KENPOM_API.md, API_QUICK_REFERENCE.md
│   ├── Analytics: KENPOM_ANALYTICS_GUIDE.md, MATCHUP_ANALYSIS_FRAMEWORK.md
│   ├── Implementation: TIER1_IMPLEMENTATION_PLAN.md, TIER2_IMPLEMENTATION_PLAN.md
│   └── Validation: EDGE_VALIDATION_GUARDRAILS.md, PREVENTING_FALSE_EDGES.md
├── tests/                           # 11 test files (matching all major modules)
├── data/                            # Cached KenPom data (parquet files)
├── reports/                         # Generated analysis reports
├── analyze_todays_games.py         # Standalone game analyzer script
└── pyproject.toml, CLAUDE.md       # Project configuration
```

### Key Modules Reference

#### Analytics Pipeline
| Module | Purpose | TIER |
|--------|---------|------|
| `comprehensive_matchup_analysis.py` | All-in-one matchup analyzer | Integration |
| `four_factors_matchup.py` | Four Factors analysis | 1 |
| `point_distribution_analysis.py` | Scoring breakdown | 1 |
| `defensive_analysis.py` | Defensive matchups | 1 |
| `tempo_analysis.py` | Pace/tempo analysis | 1 |
| `size_athleticism_analysis.py` | Height/athleticism | 2 |
| `experience_chemistry_analysis.py` | Experience analysis | 2 |
| `prediction.py` | ML predictions + backtesting | Advanced |
| `tournament_simulator.py` | Bracket simulation | Advanced |

#### Data Access (Choose One)
1. **api_client.py** - Official API (recommended, requires separate key)
2. **scraper.py + browser.py** - Web scraping (subscription login required)
3. **client.py** - kenpompy wrapper (subscription login required)

### Execution Strategy
- **Prefer parallel**: Run independent operations (file reads, searches, API calls) in parallel for speed
- **Sequential when needed**: Use sequential execution only for stability-critical operations or when there are dependencies

## KenPom Data Source

### Authentication Requirements
- **KenPom subscription required** for full data access
- Store credentials in environment variables:
  - `KENPOM_EMAIL` - Your kenpom.com email
  - `KENPOM_PASSWORD` - Your kenpom.com password
- Never commit credentials to version control

### kenpompy Library Usage
```python
from kenpompy.utils import login

# Authentication
browser = login(email, password)

# Available modules:
# - kenpompy.summary: Efficiency, four factors, height, team stats
# - kenpompy.misc: Arenas, HCA, ratings, trends, refs
# - kenpompy.team: Schedules, scouting reports
# - kenpompy.FanMatch: Game predictions
```

### Key Data Functions
| Module | Function | Description |
|--------|----------|-------------|
| summary | `get_efficiency()` | Adjusted offensive/defensive efficiency |
| summary | `get_fourfactors()` | Four Factors analysis |
| summary | `get_teamstats()` | Miscellaneous team statistics |
| misc | `get_pomeroy_ratings()` | Full Pomeroy ratings table |
| misc | `get_hca()` | Home court advantage data |
| misc | `get_trends()` | Statistical trends |
| team | `get_schedule()` | Team schedules |
| team | `get_scouting_report()` | Team statistics dictionary |

## Official KenPom API Reference

See `docs/KENPOM_API.md` for complete API documentation including:
- All 9 endpoint specifications (ratings, archive, four-factors, pointdist, height, misc-stats, fanmatch, teams, conferences)
- Parameter requirements and aliases
- Response field definitions
- Example requests

The API client in `src/kenp0m_sp0rts_analyzer/api_client.py` implements all documented endpoints with full parameter alias support for backward compatibility.

### Parameter Aliases

The Python API client supports both Python-style and official API parameter names:

| Python Style | Official API | Endpoints |
|--------------|--------------|-----------|
| `year` | `y` | ratings, four-factors, misc-stats, height, pointdist, archive, teams, conferences |
| `conference` | `c` | ratings, four-factors, misc-stats, height, pointdist |
| `archive_date` | `d` | archive |
| `game_date` | `d` | fanmatch |

**Example**:
```python
from kenp0m_sp0rts_analyzer.api_client import KenPomAPI

api = KenPomAPI()

# Both work identically
ratings1 = api.get_ratings(year=2025, conference="ACC")
ratings2 = api.get_ratings(y=2025, c="ACC")
```

### String-Boolean Conversion

The API automatically converts string boolean fields to Python booleans:
- `Preseason` (archive endpoint): "true"/"false" → True/False
- `ConfOnly` (four-factors, pointdist, misc-stats): "true"/"false" → True/False

## Code Style Guidelines

### Python Standards
- Python 3.11+ required
- Use type hints on all functions
- Follow PEP 8 style (enforced by ruff)
- Max line length: 88 characters
- Use pathlib for file paths

### Naming Conventions
- Classes: `PascalCase`
- Functions/variables: `snake_case`
- Constants: `UPPER_SNAKE_CASE`
- Private methods: `_leading_underscore`

### Data Handling
- Use Pydantic models for data validation
- Return pandas DataFrames for tabular data
- Use numpy for numerical computations
- Handle missing data gracefully with proper defaults

### AI Development Approach
- **Read before edit**: Always read and understand existing code before proposing changes
- **No speculation**: Never speculate about code behavior without inspecting it first
- **Simplicity first**: Avoid over-engineering; keep solutions minimal and focused
- **Clean workspace**: Remove any temporary files created during development

## Key Metrics Reference

### KenPom Efficiency Metrics
- **AdjO** (Adjusted Offensive Efficiency): Points scored per 100 possessions, adjusted for opponent
- **AdjD** (Adjusted Defensive Efficiency): Points allowed per 100 possessions, adjusted for opponent
- **AdjEM** (Adjusted Efficiency Margin): AdjO - AdjD
- **AdjT** (Adjusted Tempo): Possessions per 40 minutes, adjusted for opponent

### Four Factors (Dean Oliver)
1. **eFG%** - Effective Field Goal Percentage
2. **TO%** - Turnover Percentage
3. **OR%** - Offensive Rebound Percentage
4. **FTRate** - Free Throw Rate

### Research Methodology
When conducting complex analysis:
- **Form hypotheses**: State initial hypotheses before gathering data
- **Track confidence**: Maintain confidence levels and adjust based on evidence
- **Reflect on results**: After each data retrieval, reflect on findings before proceeding
- **Systematic approach**: Break complex questions into sub-hypotheses

## Testing Requirements

- Write tests for all new functionality
- Use pytest fixtures for common test data
- Mock external API calls in unit tests
- Maintain >80% code coverage
- Write general solutions; avoid test-specific hacks or hardcoded workarounds

## Data Sources

- **Primary**: https://kenpom.com/ (requires subscription)
- **Library Docs**: https://kenpompy.readthedocs.io/
- **Historical data**: Available from 1999 season onwards

## Documentation Index

Core guides in `docs/` directory:

| Document | Purpose |
|----------|---------|
| `KENPOM_API.md` | Official API reference (9 endpoints) |
| `KENPOM_ANALYTICS_GUIDE.md` | Analytics methodology and KenPom metrics |
| `MATCHUP_ANALYSIS_FRAMEWORK.md` | Matchup analysis framework |
| `TIER1_IMPLEMENTATION_PLAN.md` | TIER 1 features (Four Factors, Point Distribution, Defense, Tempo) |
| `TIER2_IMPLEMENTATION_PLAN.md` | TIER 2 features (Size/Athleticism, Experience/Chemistry) |
| `EDGE_VALIDATION_GUARDRAILS.md` | Edge validation framework |
| `PREVENTING_FALSE_EDGES.md` | False edges prevention guide |
| `QUICK_START_PREDICTIONS.md` | Quick start for predictions |
| `SETUP_GUIDE.md` | Setup and installation guide |
| `API_QUICK_REFERENCE.md` | API quick reference |
| `API_REVERSE_ENGINEERING_FINDINGS.md` | API reverse engineering findings |
| `TEMPO_PACE_DEEP_DIVE.md` | Tempo and pace deep dive |
| `KENPOM_DATA_COVERAGE.md` | Data coverage documentation |

Full documentation index: See `docs/README.md`

## Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `KENPOM_API_KEY` | Official KenPom API key (for api_client.py) | For API |
| `KENPOM_EMAIL` | KenPom subscription email (for scraper) | For scraper |
| `KENPOM_PASSWORD` | KenPom subscription password (for scraper) | For scraper |
| `KENPOM_CACHE_DIR` | Directory for cached data | No |
| `LOG_LEVEL` | Logging level (DEBUG, INFO, etc.) | No |

## Common Tasks

### Comprehensive Matchup Analysis (All TIERs)
```python
from kenp0m_sp0rts_analyzer.comprehensive_matchup_analysis import ComprehensiveMatchupAnalyzer

# Initialize analyzer with API key
analyzer = ComprehensiveMatchupAnalyzer(api_key="your-api-key")

# Analyze matchup with all TIER 1 + TIER 2 features
report = analyzer.analyze_matchup("Duke", "North Carolina", neutral_site=True)

# Generate detailed markdown report
print(report.generate_markdown())
# Includes: Four Factors, Point Distribution, Defensive Matchups, Tempo,
#           Size/Athleticism, Experience/Chemistry, Predictions

# Access specific analysis components
print(f"Predicted margin: {report.prediction.predicted_margin}")
print(f"Tempo edge: {report.tempo_analysis.pace_advantage}")
print(f"Size advantage: {report.size_analysis.height_advantage}")
```

### Tournament Bracket Simulation
```python
from kenp0m_sp0rts_analyzer.tournament_simulator import TournamentSimulator

# Initialize simulator
sim = TournamentSimulator(api_key="your-api-key")

# Run Monte Carlo simulation
results = sim.simulate_tournament(num_simulations=10000)

# Champion probabilities
print("Championship Odds:")
for team, prob in results.champion_probabilities.items():
    print(f"{team}: {prob:.1%}")

# Upset picks (lower seed with >50% win probability)
print("\nUpset Alerts:")
for upset in results.upset_picks:
    print(f"{upset.lower_seed} over {upset.higher_seed} ({upset.probability:.1%})")

# Expected value for brackets
print(f"\nExpected bracket score: {results.expected_value}")
```

### Fetching Team Efficiency Data
```python
from kenp0m_sp0rts_analyzer.client import KenPomClient

client = KenPomClient()
efficiency = client.get_efficiency(season=2024)
```

### Generating Scouting Reports
```python
report = client.get_scouting_report(team="Duke", season=2024)
```

### Analyzing Matchups (Basic)
```python
from kenp0m_sp0rts_analyzer.analysis import analyze_matchup

result = analyze_matchup(team1="Duke", team2="North Carolina", season=2024)
```

### Predictive Modeling with Machine Learning
```python
from kenp0m_sp0rts_analyzer.prediction import GamePredictor, BacktestingFramework

# Train predictor on historical data
predictor = GamePredictor()
predictor.fit(historical_games_df, margins, totals)

# Make predictions with confidence intervals
duke_stats = {'AdjEM': 24.5, 'AdjO': 118.3, 'AdjD': 93.8, 'AdjT': 68.2, 'Pythag': 0.88, 'SOS': 6.5}
unc_stats = {'AdjEM': 20.1, 'AdjO': 115.7, 'AdjD': 95.6, 'AdjT': 70.1, 'Pythag': 0.82, 'SOS': 5.8}

result = predictor.predict_with_confidence(duke_stats, unc_stats, neutral_site=True)
print(f"Margin: {result.predicted_margin} ({result.confidence_interval})")
print(f"Win probability: {result.team1_win_prob:.1%}")

# Backtest model performance
framework = BacktestingFramework()
metrics = framework.run_backtest(historical_games_df, train_split=0.8)
print(f"MAE: {metrics.mae_margin} points")
print(f"Accuracy: {metrics.accuracy:.1%}")
print(f"ATS Record: {metrics.ats_record[0]}-{metrics.ats_record[1]}")
```

### Stealth Browser Scraping (Advanced)
```python
import asyncio
from kenp0m_sp0rts_analyzer import KenPomScraper

async def scrape_data():
    # headless=False shows the browser window
    async with KenPomScraper(headless=False) as scraper:
        await scraper.login()
        ratings = await scraper.get_ratings()

        # Chrome DevTools Protocol access
        cdp = await scraper.get_cdp_session()
        await cdp.send("Network.enable")

asyncio.run(scrape_data())
```

## Official KenPom API (Recommended)

The official KenPom API provides direct JSON access to KenPom data. It requires a separate API key purchase from https://kenpom.com/register-api.php.

### API Authentication
- **Base URL**: `https://kenpom.com/api.php`
- **Auth Method**: `Authorization: Bearer <API_KEY>` header
- Store your API key in `KENPOM_API_KEY` environment variable

### Available Endpoints

| Endpoint | Parameters | Description |
|----------|------------|-------------|
| `ratings` | `y` (year), `team_id`, `c` (conference) | Team ratings (AdjEM, AdjO, AdjD, AdjT, SOS, etc.) |
| `archive` | `d` (date), `preseason`, `y` (year), `team_id`, `c` | Historical ratings from specific dates or preseason |
| `teams` | `y` (year), `c` (conference) | Team list with TeamID, coach, arena info |
| `conferences` | `y` (year) | Conference list with IDs |
| `fanmatch` | `d` (date: YYYY-MM-DD) | Game predictions with win probability |
| `four-factors` | `y` (year), `team_id`, `c`, `conf_only` | Four Factors (eFG%, TO%, OR%, FT Rate) |
| `misc-stats` | `y` (year), `team_id`, `c`, `conf_only` | Shooting %, blocks, steals, assists |
| `height` | `y` (year), `team_id`, `c` | Team height and experience data |
| `pointdist` | `y` (year), `team_id`, `c`, `conf_only` | Point distribution breakdown |

### API Client Usage
```python
from kenp0m_sp0rts_analyzer.api_client import KenPomAPI

# Initialize (uses KENPOM_API_KEY env var)
api = KenPomAPI()

# Get 2025 ratings
ratings = api.get_ratings(year=2025)
print(f"#1 team: {ratings.data[0]['TeamName']}")

# Get team historical data
duke_history = api.get_ratings(team_id=73)  # Duke's TeamID

# Filter by conference
big12 = api.get_ratings(year=2025, conference="B12")

# Get archived ratings from a specific date
archive = api.get_archive(archive_date="2025-02-15")

# Get preseason ratings
preseason = api.get_archive(preseason=True, year=2025)

# Get game predictions
games = api.get_fanmatch("2025-03-15")
close_games = [g for g in games.data if 40 <= g['HomeWP'] <= 60]

# Get Four Factors (conference-only stats)
four_factors = api.get_four_factors(year=2025, conf_only=True)
df = four_factors.to_dataframe()

# Find team by name
duke = api.get_team_by_name("Duke", 2025)
print(f"Duke TeamID: {duke['TeamID']}")  # 73
```

### API Response Fields (Ratings)

| Field | Description |
|-------|-------------|
| `AdjEM` | Adjusted Efficiency Margin |
| `AdjOE` | Adjusted Offensive Efficiency |
| `AdjDE` | Adjusted Defensive Efficiency |
| `AdjTempo` | Adjusted Tempo |
| `Pythag` | Pythagorean win expectation |
| `Luck` | Luck factor |
| `SOS` / `SOSO` / `SOSD` | Strength of schedule metrics |
| `NCSOS` | Non-conference SOS |
| `APL_Off` / `APL_Def` | Average Possession Length |
| `Seed` | NCAA Tournament seed |

### Raw API Examples (curl)
```bash
# Get 2025 ratings
curl "https://kenpom.com/api.php?endpoint=ratings&y=2025" \
  -H "Authorization: Bearer $KENPOM_API_KEY"

# Get Duke's historical data (team_id=73)
curl "https://kenpom.com/api.php?endpoint=ratings&team_id=73" \
  -H "Authorization: Bearer $KENPOM_API_KEY"

# Filter by conference
curl "https://kenpom.com/api.php?endpoint=ratings&y=2025&c=SEC" \
  -H "Authorization: Bearer $KENPOM_API_KEY"

# Get archived ratings from a specific date
curl "https://kenpom.com/api.php?endpoint=archive&d=2025-02-15" \
  -H "Authorization: Bearer $KENPOM_API_KEY"

# Get preseason ratings
curl "https://kenpom.com/api.php?endpoint=archive&preseason=true&y=2025" \
  -H "Authorization: Bearer $KENPOM_API_KEY"

# Get game predictions for a date
curl "https://kenpom.com/api.php?endpoint=fanmatch&d=2025-03-15" \
  -H "Authorization: Bearer $KENPOM_API_KEY"

# Get Four Factors (conference-only stats)
curl "https://kenpom.com/api.php?endpoint=four-factors&y=2025&conf_only=true" \
  -H "Authorization: Bearer $KENPOM_API_KEY"
```

## Stealth Browser Module

### Installation
```bash
pip install -e ".[browser]"
playwright install chromium
```

### Key Components
| Component | Description |
|-----------|-------------|
| `StealthBrowser` | Playwright browser with stealth configuration |
| `BrowserConfig` | Configuration for viewport, user agent, CDP |
| `KenPomScraper` | High-level scraper with login and data extraction |

### Stealth Features
- Removes `navigator.webdriver` detection flag
- Randomizes viewport sizes and user agents
- Provides Chrome DevTools Protocol (CDP) access
- Supports persistent sessions via `user_data_dir`
- Human-like delays and interactions

## MCP Server (Claude Integration)

The MCP (Model Context Protocol) server exposes KenPom analytics tools for use with Claude and other MCP clients.

### Installation
```bash
pip install -e ".[mcp]"
```

### Running the Server
```bash
python -m kenp0m_sp0rts_analyzer.mcp_server
```

### Configuration with Claude Code
Add to `.claude/mcp.json`:
```json
{
  "mcpServers": {
    "kenpom": {
      "command": "python",
      "args": ["-m", "kenp0m_sp0rts_analyzer.mcp_server"],
      "env": {
        "KENPOM_API_KEY": "your-api-key"
      }
    }
  }
}
```

### Available Tools

| Tool | Description |
|------|-------------|
| `get_team_efficiency` | Adjusted offensive/defensive efficiency ratings |
| `get_four_factors` | Four Factors analysis (eFG%, TO%, OR%, FTRate) |
| `get_team_schedule` | Team schedule with results and opponent metrics |
| `get_scouting_report` | Comprehensive team scouting report |
| `get_pomeroy_ratings` | Full KenPom ratings table |
| `analyze_matchup` | Head-to-head matchup analysis with predictions |
| `get_home_court_advantage` | Home court advantage data |
| `get_game_predictions` | Game predictions for a specific date |

### Authentication Priority
1. **Official API** (`KENPOM_API_KEY`) - Recommended, faster, more reliable
2. **Scraper** (`KENPOM_EMAIL`/`KENPOM_PASSWORD`) - Fallback if no API key
