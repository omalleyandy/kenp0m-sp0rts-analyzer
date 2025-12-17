# KenPom Sports Analyzer

Advanced NCAA Division I Men's Basketball analytics using KenPom data for research and analysis.

## Overview

This project provides tools for analyzing college basketball using Ken Pomeroy's advanced analytics system with machine learning-based predictions and confidence intervals.

## Features

- **Official KenPom API**: Direct JSON access to KenPom data (recommended)
- **MCP Server**: Claude integration for AI-assisted basketball analytics
- **KenPom Data Integration**: Full access to efficiency ratings, four factors, and tempo metrics
- **Team Analysis**: Comprehensive scouting reports and schedule analysis
- **Matchup Analysis**: Head-to-head comparisons with predicted outcomes
- **Historical/Archive Data**: Track team ratings from specific dates or preseason
- **Conference Filtering**: Filter data by conference for focused analysis
- **Machine Learning Predictions**: Gradient Boosting models with confidence intervals
- **Stealth Browser Scraping**: Playwright-based browser with stealth techniques for reliable data access

## Requirements

- Python 3.11+
- **KenPom subscription** (required for data access)

## Installation

### From Source

```bash
git clone https://github.com/omalleyandy/kenp0m-sp0rts-analyzer.git
cd kenp0m-sp0rts-analyzer
pip install -e ".[dev]"
```

### With Browser Automation (Stealth Scraping)

```bash
# Install with browser dependencies
pip install -e ".[browser]"

# Install Chromium for Playwright
playwright install chromium
```

### With MCP Server (Claude Integration)

```bash
# Install with MCP dependencies
pip install -e ".[mcp]"
```

### All Dependencies

```bash
# Install everything
pip install -e ".[all]"
playwright install chromium
```

### Core Dependencies

```bash
pip install kenpompy pandas numpy pydantic httpx
```

## Configuration

### Environment Variables

Create a `.env` file in your project root:

```bash
# Official API (recommended)
KENPOM_API_KEY=your-api-key

# Scraper credentials (alternative)
KENPOM_EMAIL=your-email@example.com
KENPOM_PASSWORD=your-password
```

> **Security Note**: Never commit credentials to version control. The `.env` file is included in `.gitignore`.

## Quick Start

### Authentication

```python
from kenpompy.utils import login

# Login to KenPom (requires subscription)
browser = login("your-email@example.com", "your-password")
```

### Fetch Team Efficiency Data

```python
from kenpompy.summary import get_efficiency

# Get current season efficiency ratings
efficiency_df = get_efficiency(browser)

# Get specific season
efficiency_2023 = get_efficiency(browser, season=2023)
```

### Get Four Factors Analysis

```python
from kenpompy.summary import get_fourfactors

# Four Factors: eFG%, TO%, OR%, FTRate
four_factors = get_fourfactors(browser)
```

### Team Scouting Reports

```python
from kenpompy.team import get_scouting_report, get_schedule

# Get team statistics
duke_report = get_scouting_report(browser, team="Duke", season=2024)

# Get team schedule
duke_schedule = get_schedule(browser, team="Duke", season=2024)
```

### Miscellaneous Data

```python
from kenpompy.misc import (
    get_pomeroy_ratings,
    get_hca,
    get_trends,
    get_arenas
)

# Full Pomeroy ratings table
ratings = get_pomeroy_ratings(browser)

# Home court advantage data
hca = get_hca(browser)

# Statistical trends
trends = get_trends(browser)

# Arena information
arenas = get_arenas(browser)
```

## KenPom Metrics Reference

### Efficiency Metrics

| Metric | Description |
|--------|-------------|
| **AdjO** | Adjusted Offensive Efficiency - Points scored per 100 possessions |
| **AdjD** | Adjusted Defensive Efficiency - Points allowed per 100 possessions |
| **AdjEM** | Adjusted Efficiency Margin (AdjO - AdjD) |
| **AdjT** | Adjusted Tempo - Possessions per 40 minutes |

### Four Factors (Dean Oliver)

| Factor | Description |
|--------|-------------|
| **eFG%** | Effective Field Goal Percentage (accounts for 3-pointers) |
| **TO%** | Turnover Percentage (turnovers per possession) |
| **OR%** | Offensive Rebound Percentage |
| **FTRate** | Free Throw Rate (FTA/FGA) |

### Additional Metrics

| Metric | Description |
|--------|-------------|
| **SOS** | Strength of Schedule |
| **NCSOS** | Non-Conference Strength of Schedule |
| **Luck** | Close game performance vs expected outcomes |
| **NetRtg** | Net Efficiency Rating |

## Predictive Modeling

This package includes machine learning models for game predictions with confidence intervals and validation.

### Features

- **Gradient Boosting**: Non-linear predictions using team efficiency metrics
- **Confidence Intervals**: 50% confidence bands via quantile regression (25th-75th percentile)
- **Backtesting**: Validate predictions against historical outcomes
- **Feature Engineering**: 9 engineered features including efficiency differentials and tempo interactions

### Quick Start

```python
from kenp0m_sp0rts_analyzer.prediction import GamePredictor, BacktestingFramework
import pandas as pd

# Train model on historical data
predictor = GamePredictor()
predictor.fit(historical_games_df, margins, totals)

# Predict game outcome
duke_stats = {'AdjEM': 24.5, 'AdjO': 118.3, 'AdjD': 93.8, 'AdjT': 68.2, 'Pythag': 0.88, 'SOS': 6.5}
unc_stats = {'AdjEM': 20.1, 'AdjO': 115.7, 'AdjD': 95.6, 'AdjT': 70.1, 'Pythag': 0.82, 'SOS': 5.8}

result = predictor.predict_with_confidence(duke_stats, unc_stats, neutral_site=True)

print(f"Predicted margin: {result.predicted_margin} points")
print(f"Confidence interval: {result.confidence_interval}")
print(f"Duke win probability: {result.team1_win_prob:.1%}")
print(f"Predicted total: {result.predicted_total}")
```

### Backtesting Performance

```python
# Validate model with historical data
framework = BacktestingFramework()
metrics = framework.run_backtest(historical_games_df, train_split=0.8)

print(f"MAE (margin): {metrics.mae_margin} points")
print(f"RMSE (margin): {metrics.rmse_margin} points")
print(f"Accuracy: {metrics.accuracy:.1%}")
print(f"Brier score: {metrics.brier_score:.3f}")
print(f"R² score: {metrics.r2_margin:.3f}")
```

### Expected Model Performance

With sufficient training data (100+ games), typical backtesting metrics:

- **MAE (Mean Absolute Error)**: 8-10 points
- **RMSE (Root Mean Squared Error)**: 10-12 points
- **Accuracy** (correct winner): 65-72%
- **Brier Score** (probability calibration): < 0.20
- **R² Score**: 0.35-0.50

### Cross-Validation

```python
# K-fold cross-validation for robust evaluation
metrics_list = framework.cross_validate(games_df, n_folds=5)

for i, metrics in enumerate(metrics_list):
    print(f"Fold {i+1}: MAE={metrics.mae_margin}, Accuracy={metrics.accuracy:.1%}")
```

## Stealth Browser Scraping

For more reliable data access, the package includes a Playwright-based stealth browser that mimics real user behavior.

### Basic Stealth Scraper Usage

```python
import asyncio
from kenp0m_sp0rts_analyzer import KenPomScraper

async def main():
    # headless=False shows the browser window
    async with KenPomScraper(headless=False) as scraper:
        # Login with credentials from environment
        await scraper.login()

        # Get ratings data
        ratings = await scraper.get_ratings()
        print(ratings.head(10))

        # Take a screenshot
        await scraper.screenshot("kenpom.png")

        # Access Chrome DevTools Protocol
        cdp = await scraper.get_cdp_session()
        await cdp.send("Network.enable")

asyncio.run(main())
```

### Stealth Browser Features

| Feature | Description |
|---------|-------------|
| **Visible Browser** | `headless=False` opens a visible Chrome window |
| **Stealth Mode** | Removes automation detection flags |
| **CDP Access** | Full Chrome DevTools Protocol access |
| **Session Persistence** | Save login sessions across runs |
| **Randomization** | Random viewport sizes and user agents |

### Low-Level Browser API

```python
from kenp0m_sp0rts_analyzer import create_stealth_browser

async with create_stealth_browser(headless=False) as browser:
    page = await browser.new_page()
    await page.goto("https://kenpom.com")

    # CDP session for advanced control
    cdp = await browser.get_cdp_session(page)
    await cdp.send("Network.enable")
```

### Stealth Configuration

```python
from kenp0m_sp0rts_analyzer.browser import BrowserConfig, StealthBrowser

config = BrowserConfig(
    headless=False,              # Show browser window
    slow_mo=100,                 # Slow down actions (ms)
    timeout=30000,               # Default timeout (ms)
    randomize_viewport=True,     # Random screen size
    randomize_user_agent=True,   # Random user agent
    enable_cdp=True,             # Chrome DevTools access
    disable_webdriver=True,      # Remove webdriver flag
)

browser = StealthBrowser(config=config)
```

## Official KenPom API (Recommended)

The official KenPom API provides direct JSON access to KenPom data. It requires a separate API key purchase from https://kenpom.com/register-api.php.

### API Setup

```bash
export KENPOM_API_KEY="your-api-key"
```

### API Client Usage

```python
from kenp0m_sp0rts_analyzer.api_client import KenPomAPI

api = KenPomAPI()

# Get 2025 ratings
ratings = api.get_ratings(year=2025)

# Filter by conference
sec = api.get_ratings(year=2025, conference="SEC")

# Get archived ratings from a specific date
archive = api.get_archive(archive_date="2025-02-15")

# Get preseason ratings
preseason = api.get_archive(preseason=True, year=2025)

# Get game predictions
games = api.get_fanmatch("2025-03-15")

# Get Four Factors (conference-only stats)
four_factors = api.get_four_factors(year=2025, conf_only=True)
df = four_factors.to_dataframe()

# Get team by name
duke = api.get_team_by_name("Duke", 2025)
print(f"Duke TeamID: {duke['TeamID']}")
```

### API Endpoints

| Endpoint | Description |
|----------|-------------|
| `ratings` | Team ratings (AdjEM, AdjO, AdjD, AdjT, SOS) |
| `archive` | Historical ratings from specific dates or preseason |
| `teams` | Team list with IDs, coaches, arenas |
| `conferences` | Conference list with IDs |
| `fanmatch` | Game predictions with win probability |
| `four-factors` | Four Factors (eFG%, TO%, OR%, FT Rate) |
| `misc-stats` | Shooting %, blocks, steals, assists |
| `height` | Team height and experience data |
| `pointdist` | Point distribution breakdown |

### API Documentation

See **[docs/KENPOM_API.md](docs/KENPOM_API.md)** for complete API documentation including:
- All 9 endpoint specifications with parameters and response fields
- Official API parameter names and Python-style aliases
- Example requests and usage patterns
- String-boolean conversion details

The Python API client supports both naming conventions:
```python
# Python-style parameters
ratings = api.get_ratings(year=2025, conference="ACC")

# Official API parameters (both work identically)
ratings = api.get_ratings(y=2025, c="ACC")
```

## MCP Server (Claude Integration)

The MCP (Model Context Protocol) server exposes KenPom analytics tools for use with Claude and other MCP clients.

### Running the MCP Server

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

### Available MCP Tools

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

## kenpompy Module Reference

### Summary Module (`kenpompy.summary`)

| Function | Description |
|----------|-------------|
| `get_efficiency()` | Efficiency and tempo statistics |
| `get_fourfactors()` | Four Factors analysis |
| `get_height()` | Height/Experience data |
| `get_kpoy()` | Player of the Year leaders |
| `get_playerstats()` | Player leader metrics |
| `get_pointdist()` | Team points distribution |
| `get_teamstats()` | Miscellaneous team statistics |

### Misc Module (`kenpompy.misc`)

| Function | Description |
|----------|-------------|
| `get_arenas()` | Arena information |
| `get_current_season()` | Latest available season |
| `get_gameattribs()` | Game attributes (Excitement, Tension, etc.) |
| `get_hca()` | Home court advantage data |
| `get_pomeroy_ratings()` | Full Pomeroy ratings table |
| `get_program_ratings()` | Historical program ratings |
| `get_refs()` | Officials rankings |
| `get_trends()` | Statistical trends |

### Team Module (`kenpompy.team`)

| Function | Description |
|----------|-------------|
| `get_schedule()` | Team schedules with results |
| `get_scouting_report()` | Team statistics dictionary |
| `get_valid_teams()` | Available teams for a season |

### FanMatch Module (`kenpompy.FanMatch`)

Provides game prediction data including predicted scores, margins of victory, and favorite records.

## Data Availability

- Historical data available from **1999** season onwards
- Season parameter is optional (defaults to current season)
- All functions return pandas DataFrames for easy analysis

## Development

### Setup Development Environment

```bash
pip install -e ".[dev]"
```

### Run Tests

```bash
pytest
pytest --cov=src/kenp0m_sp0rts_analyzer
```

### Code Quality

```bash
# Linting
ruff check src/ tests/

# Formatting
ruff format src/ tests/

# Type checking
mypy src/
```

### API Reverse Engineering Tools

The project includes development tools for reverse engineering and validating the KenPom API documentation:

```bash
# Reverse engineer API documentation
uv run python scripts/reverse_engineer_api_docs.py --email your-email@example.com --password your-password

# Compare discovered endpoints with implementation
uv run python scripts/compare_api_docs.py
```

**Tools:**
- `scripts/reverse_engineer_api_docs.py` - Extracts endpoint information from API documentation page
- `scripts/compare_api_docs.py` - Compares discovered endpoints with `api_client.py` implementation

**Output:**
- `reports/api_reverse_engineering/` - Generated reports (gitignored)
- `docs/API_REVERSE_ENGINEERING_FINDINGS.md` - Findings documentation

See **[scripts/README.md](scripts/README.md)** for detailed usage instructions.

## Project Structure

```
kenp0m-sp0rts-analyzer/
├── src/kenp0m_sp0rts_analyzer/
│   ├── __init__.py        # Package initialization
│   ├── api_client.py      # Official KenPom API client (recommended)
│   ├── mcp_server.py      # MCP server for Claude integration
│   ├── client.py          # KenPom client wrapper (kenpompy)
│   ├── browser.py         # Stealth browser automation
│   ├── scraper.py         # KenPom web scraper
│   ├── models.py          # Pydantic data models
│   ├── analysis.py        # Analytics functions
│   └── utils.py           # Utility functions
├── tests/                 # Test suite
│   ├── test_api_client.py # API client tests
│   └── test_mcp_server.py # MCP server tests
├── examples/
│   ├── basic_usage.py     # Getting started
│   ├── matchup_analysis.py # CLI matchup tool
│   └── stealth_scraper.py # Stealth browser example
├── .claude/               # Claude Code settings
├── pyproject.toml        # Project configuration
├── CLAUDE.md             # AI assistant guidelines
└── README.md             # This file
```

## Resources

- **KenPom Website**: https://kenpom.com/
- **kenpompy Documentation**: https://kenpompy.readthedocs.io/
- **kenpompy Source**: https://github.com/j-andrews7/kenpompy

## License

MIT License - See [LICENSE](LICENSE) file for details.

## Disclaimer

This project is for educational and research purposes only. A valid KenPom subscription is required for data access. The responsibility to use this package in a reasonable manner falls upon the user. Please respect KenPom's terms of service and rate limits.
