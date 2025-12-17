# Scripts Directory

This directory contains utility scripts for the KenPom Sports Analyzer project.

## API Reverse Engineering Tools

### `reverse_engineer_api_docs.py`

Reverse engineers the KenPom API documentation page using Chrome DevTools Protocol to discover endpoints, parameters, and response fields.

**Usage:**
```bash
# Using uv (recommended)
uv run python scripts/reverse_engineer_api_docs.py --email user@example.com --password pass

# Using environment variables with uv
export KENPOM_EMAIL=your-email@example.com
export KENPOM_PASSWORD=your-password
uv run python scripts/reverse_engineer_api_docs.py

# Run in headless mode
uv run python scripts/reverse_engineer_api_docs.py --headless --email user@example.com --password pass

# Don't save output files
uv run python scripts/reverse_engineer_api_docs.py --no-save --email user@example.com --password pass

# Or with regular python (if in virtual environment)
python scripts/reverse_engineer_api_docs.py --email user@example.com --password pass
```

**What it does:**
1. Logs into KenPom (if required)
2. Navigates to the API documentation page
3. Monitors all network requests using Chrome DevTools Protocol
4. Extracts endpoint information from HTML documentation
5. Saves results to `reports/api_reverse_engineering/`

**Output files:**
- `api_docs_[timestamp].html` - Full HTML of documentation page
- `api_docs_[timestamp].txt` - Text content
- `api_reverse_engineering_[timestamp].json` - Structured results
- `network_requests_[timestamp].json` - All captured API requests

### `compare_api_docs.py`

Compares discovered endpoints from reverse engineering with the existing `api_client.py` implementation to identify gaps.

**Usage:**
```bash
# Use latest results file automatically
uv run python scripts/compare_api_docs.py

# Specify a results file
uv run python scripts/compare_api_docs.py reports/api_reverse_engineering/api_reverse_engineering_20250101_120000.json

# Use custom reports directory
uv run python scripts/compare_api_docs.py --reports-dir custom/path
```

**What it does:**
1. Loads reverse engineering results
2. Compares with implemented endpoints in `api_client.py`
3. Identifies:
   - Missing endpoints
   - Missing parameters
   - Extra parameters (implemented but not in docs)
   - Fully/partially covered endpoints
4. Generates a comparison report

**Output:**
- Prints comparison report to console
- Saves `comparison_report.json` to results directory

## Workflow

### Complete Reverse Engineering Workflow

1. **Run reverse engineering:**
   ```bash
   uv run python scripts/reverse_engineer_api_docs.py --email your-email --password your-password
   ```

2. **Compare with implementation:**
   ```bash
   uv run python scripts/compare_api_docs.py
   ```

3. **Review findings:**
   - Check `docs/API_REVERSE_ENGINEERING_FINDINGS.md`
   - Review comparison report JSON
   - Examine captured network requests

4. **Update implementation:**
   - Add missing endpoints to `api_client.py`
   - Update documentation in `docs/KENPOM_API.md`
   - Add tests for new features

## Requirements

- Python 3.11+
- KenPom subscription credentials
- Browser dependencies: `pip install kenp0m-sp0rts-analyzer[browser]`
- Playwright: `playwright install chromium`

## Troubleshooting

### Cloudflare Protection
If you encounter Cloudflare challenges:
- The script uses stealth browser techniques automatically
- Try running with `--headless=false` to see what's happening
- Ensure you have valid KenPom credentials

### Network Monitoring Issues
If network requests aren't being captured:
- Check that CDP is enabled (it is by default)
- Verify browser is not in headless mode for debugging
- Check browser console for errors

### Missing Dependencies
If you get import errors:
```bash
pip install -e ".[browser]"
playwright install chromium
```

## Other Scripts

### `validate_tempo_features.py`
Validates tempo analysis features and data consistency.

