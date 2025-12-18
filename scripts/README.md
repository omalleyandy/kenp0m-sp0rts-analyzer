# Scripts Directory

This directory contains utility scripts for the KenPom Sports Analyzer project, organized by function.

## Directory Structure

```
scripts/
├── analysis/              # Prediction analysis and performance tracking
│   ├── analyze_fanmatch.py
│   ├── analyze_performance.py
│   ├── compare_predictions.py
│   └── kenpom_vs_vegas_analysis.py
├── scrapers/              # Data collection scripts
│   ├── monitor_overtime_timing.py
│   ├── scrape_covers_injuries.py
│   ├── scrape_overtime_lines.py
│   └── scrape_results.py
├── collect_daily_data.py  # Daily data collection workflow
├── compare_api_docs.py    # API documentation comparison
├── predict_game.py        # Single game prediction
├── reverse_engineer_api_docs.py  # API reverse engineering
├── test_beat_reporters.py # Twitter beat reporter testing
├── test_twitter_connection.py    # Twitter API testing
├── validate_edge.py       # Edge validation
└── validate_tempo_features.py    # Tempo feature validation
```

---

## Analysis Scripts (`analysis/`)

### `analyze_fanmatch.py`
Analyzes KenPom FanMatch predictions for a specific date.

```bash
uv run python scripts/analysis/analyze_fanmatch.py
```

### `analyze_performance.py`
Generates performance analytics from the historical odds database.

```bash
# Show overall model performance
uv run python scripts/analysis/analyze_performance.py

# Performance for specific date range
uv run python scripts/analysis/analyze_performance.py --start 2025-12-01 --end 2025-12-31

# CLV (Closing Line Value) analysis
uv run python scripts/analysis/analyze_performance.py --clv

# ATS record by conference
uv run python scripts/analysis/analyze_performance.py --by-conference
```

### `compare_predictions.py`
Compares KenPom FanMatch predictions with actual game results.

```bash
uv run python scripts/analysis/compare_predictions.py
```

### `kenpom_vs_vegas_analysis.py`
Complete analysis comparing KenPom predictions vs Vegas lines vs actual results.

```bash
uv run python scripts/analysis/kenpom_vs_vegas_analysis.py
```

---

## Scraper Scripts (`scrapers/`)

### `scrape_overtime_lines.py`
Scrapes Vegas lines from overtime.ag for college basketball games.

```bash
# Fetch today's lines
uv run python scripts/scrapers/scrape_overtime_lines.py

# Fetch tomorrow's lines
uv run python scripts/scrapers/scrape_overtime_lines.py --tomorrow

# Fetch lines for a specific date
uv run python scripts/scrapers/scrape_overtime_lines.py --date 2025-12-18

# Run in headless mode
uv run python scripts/scrapers/scrape_overtime_lines.py --headless

# Mark as opening/closing lines
uv run python scripts/scrapers/scrape_overtime_lines.py --open
uv run python scripts/scrapers/scrape_overtime_lines.py --close
```

### `monitor_overtime_timing.py`
Monitors overtime.ag to discover when college basketball odds are released.

```bash
# Run continuous monitoring (24 hours, every 30 minutes)
uv run python scripts/scrapers/monitor_overtime_timing.py

# Custom interval and duration
uv run python scripts/scrapers/monitor_overtime_timing.py -i 15 -d 48

# Show browser for debugging
uv run python scripts/scrapers/monitor_overtime_timing.py --show-browser

# Single capture (for testing)
uv run python scripts/scrapers/monitor_overtime_timing.py --single

# Analyze collected data
uv run python scripts/scrapers/monitor_overtime_timing.py --analyze

# Quick summary
uv run python scripts/scrapers/monitor_overtime_timing.py --summary

# Export report to file
uv run python scripts/scrapers/monitor_overtime_timing.py --analyze -o timing_report.md
```

**Data stored in:** `data/overtime_monitoring/`

### `scrape_results.py`
Scrapes game results from ESPN and updates the historical database.

```bash
# Update results for today's games
uv run python scripts/scrapers/scrape_results.py

# Update results for a specific date
uv run python scripts/scrapers/scrape_results.py --date 2025-12-17

# Show pending games
uv run python scripts/scrapers/scrape_results.py --pending

# Calculate prediction results
uv run python scripts/scrapers/scrape_results.py --calculate
```

### `scrape_covers_injuries.py`
Scrapes injury information from Covers.com.

```bash
uv run python scripts/scrapers/scrape_covers_injuries.py
```

---

## Daily Workflow Scripts

### `collect_daily_data.py`
Runs the complete daily data collection workflow.

```bash
uv run python scripts/collect_daily_data.py
```

### `predict_game.py`
Makes a prediction for a single game matchup.

```bash
uv run python scripts/predict_game.py "Duke" "North Carolina"
```

---

## API Reverse Engineering Tools

### `reverse_engineer_api_docs.py`

Reverse engineers the KenPom API documentation page using Chrome DevTools Protocol.

```bash
# Using environment variables
export KENPOM_EMAIL=your-email@example.com
export KENPOM_PASSWORD=your-password
uv run python scripts/reverse_engineer_api_docs.py

# With credentials
uv run python scripts/reverse_engineer_api_docs.py --email user@example.com --password pass

# Headless mode
uv run python scripts/reverse_engineer_api_docs.py --headless
```

**Output:** `reports/api_reverse_engineering/`

### `compare_api_docs.py`

Compares discovered endpoints with the existing `api_client.py` implementation.

```bash
# Use latest results
uv run python scripts/compare_api_docs.py

# Specify results file
uv run python scripts/compare_api_docs.py reports/api_reverse_engineering/api_reverse_engineering_*.json
```

---

## Validation Scripts

### `validate_edge.py`
Validates edge detection and betting edge calculations.

```bash
uv run python scripts/validate_edge.py
```

### `validate_tempo_features.py`
Validates tempo analysis features and data consistency.

```bash
uv run python scripts/validate_tempo_features.py
```

---

## Testing Scripts

### `test_beat_reporters.py`
Tests Twitter beat reporter configuration and connectivity.

```bash
uv run python scripts/test_beat_reporters.py
```

### `test_twitter_connection.py`
Tests Twitter API connection and authentication.

```bash
uv run python scripts/test_twitter_connection.py
```

---

## Requirements

- Python 3.11+
- KenPom subscription credentials
- For browser scripts: `pip install kenp0m-sp0rts-analyzer[browser]`
- Playwright: `playwright install chromium`

## Troubleshooting

### Import Errors
```bash
pip install -e ".[browser]"
playwright install chromium
```

### Cloudflare Protection
- Scripts use stealth browser techniques automatically
- Try `--headless=false` to debug
- Verify KenPom credentials are valid
