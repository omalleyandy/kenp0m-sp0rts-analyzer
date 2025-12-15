# KenPom Sports Analyzer

Advanced NCAA Division I Men's Basketball analytics using KenPom data, integrated with Billy Walters betting methodology for college basketball research.

## Overview

This project provides tools for analyzing college basketball using Ken Pomeroy's advanced analytics system. It combines statistical analysis with proven sports betting methodologies to identify value opportunities and generate insights.

## Features

- **KenPom Data Integration**: Full access to efficiency ratings, four factors, and tempo metrics
- **Team Analysis**: Comprehensive scouting reports and schedule analysis
- **Matchup Analysis**: Head-to-head comparisons with predicted outcomes
- **Historical Trends**: Track team and conference performance over time
- **Billy Walters Integration**: Sharp money analysis and value identification
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

## Project Structure

```
kenp0m-sp0rts-analyzer/
├── src/kenp0m_sp0rts_analyzer/
│   ├── __init__.py        # Package initialization
│   ├── client.py          # KenPom client wrapper (kenpompy)
│   ├── browser.py         # Stealth browser automation
│   ├── scraper.py         # KenPom web scraper
│   ├── models.py          # Pydantic data models
│   ├── analysis.py        # Analytics functions
│   └── utils.py           # Utility functions
├── tests/                 # Test suite
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
