# CLAUDE.md - AI Assistant Guidelines for KenPom Sports Analyzer

## Project Overview

This is a Python project for NCAA Division I Men's Basketball analytics using KenPom data. It integrates the kenpompy library for data retrieval with Billy Walters betting methodology for college basketball research.

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
├── src/kenp0m_sp0rts_analyzer/
│   ├── __init__.py          # Package init with version
│   ├── api_client.py        # Official KenPom API client (recommended)
│   ├── client.py            # KenPom client wrapper (kenpompy)
│   ├── browser.py           # Stealth browser automation (Playwright)
│   ├── scraper.py           # KenPom web scraper
│   ├── models.py            # Pydantic data models
│   ├── analysis.py          # Analytics functions
│   └── utils.py             # Utility functions
├── tests/                   # Test suite
├── examples/                # Usage examples
├── .claude/                 # Claude Code settings
├── pyproject.toml          # Project configuration
└── CLAUDE.md               # This file
```

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

### Billy Walters Methodology Integration
- Focus on line movement analysis
- Track sharp money vs public money
- Identify value opportunities via efficiency differentials
- Consider home court advantage adjustments

## Testing Requirements

- Write tests for all new functionality
- Use pytest fixtures for common test data
- Mock external API calls in unit tests
- Maintain >80% code coverage

## Data Sources

- **Primary**: https://kenpom.com/ (requires subscription)
- **Library Docs**: https://kenpompy.readthedocs.io/
- **Historical data**: Available from 1999 season onwards

## Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `KENPOM_API_KEY` | Official KenPom API key (for api_client.py) | For API |
| `KENPOM_EMAIL` | KenPom subscription email (for scraper) | For scraper |
| `KENPOM_PASSWORD` | KenPom subscription password (for scraper) | For scraper |
| `KENPOM_CACHE_DIR` | Directory for cached data | No |
| `LOG_LEVEL` | Logging level (DEBUG, INFO, etc.) | No |

## Common Tasks

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

### Analyzing Matchups
```python
from kenp0m_sp0rts_analyzer.analysis import analyze_matchup

result = analyze_matchup(team1="Duke", team2="North Carolina", season=2024)
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
| `ratings` | `y` (year) or `team_id` | Team ratings (AdjEM, AdjO, AdjD, AdjT, SOS, etc.) |
| `teams` | `y` (year) | Team list with TeamID, coach, arena info |
| `conferences` | `y` (year) | Conference list with IDs |
| `fanmatch` | `d` (date: YYYY-MM-DD) | Game predictions with win probability |
| `four-factors` | `y` (year) | Four Factors (eFG%, TO%, OR%, FT Rate) |
| `misc-stats` | `y` (year) | Shooting %, blocks, steals, assists |
| `height` | `y` (year) | Team height and experience data |
| `pointdist` | `y` (year) | Point distribution breakdown |

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

# Get game predictions
games = api.get_fanmatch("2025-03-15")
close_games = [g for g in games.data if 40 <= g['HomeWP'] <= 60]

# Get Four Factors
four_factors = api.get_four_factors(year=2025)
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

# Get game predictions for a date
curl "https://kenpom.com/api.php?endpoint=fanmatch&d=2025-03-15" \
  -H "Authorization: Bearer $KENPOM_API_KEY"

# Get Four Factors
curl "https://kenpom.com/api.php?endpoint=four-factors&y=2025" \
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
