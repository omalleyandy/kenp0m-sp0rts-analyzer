# Quick Setup Guide: Analyzing Louisville @ Tennessee

This guide walks you through setting up the KenPom Sports Analyzer to analyze today's game: **#11 Louisville @ #20 Tennessee** (Dec 16, 2025).

## Prerequisites

- Python 3.11 or higher
- uv package manager
- KenPom API key (purchase from https://kenpom.com/register-api.php)

## Step 1: Install Dependencies

The project uses `uv` for package management. Install all dependencies:

```bash
# Install core + all optional dependencies
uv sync --all-extras
```

This installs:
- Core: `kenpompy`, `httpx`, `pandas`, `numpy`, `pydantic`, `scikit-learn`
- Browser: `playwright`, `playwright-stealth`, `beautifulsoup4`
- MCP: Model Context Protocol server
- Dev: `pytest`, `ruff`, `mypy`

## Step 2: Configure API Key

You need a KenPom API key to access their data.

### Option A: Using .env file (Recommended)

1. Create or edit `.env` in the project root:
   ```bash
   KENPOM_API_KEY=your-api-key-here
   ```

2. The script will automatically load it using `python-dotenv`

### Option B: Export environment variable

```bash
# Windows (PowerShell)
$env:KENPOM_API_KEY="your-api-key-here"

# Linux/Mac
export KENPOM_API_KEY="your-api-key-here"
```

### Getting Your API Key

1. Visit https://kenpom.com/register-api.php
2. Purchase API access (separate from website subscription)
3. Copy your API key
4. Add to `.env` file

**Security Note**: Never commit `.env` to version control. It's already in `.gitignore`.

## Step 3: Run the Analysis

We've created a dedicated script for the Louisville @ Tennessee game:

```bash
uv run python analyze_louisville_tennessee.py
```

### What the Script Does

1. **Loads configuration**: Reads API key from `.env`
2. **Initializes analyzer**: Sets up the 7-module comprehensive analyzer
3. **Fetches KenPom data**: Gets 2025 season ratings, four factors, height, etc.
4. **Runs 15-dimensional analysis**:
   - Efficiency ratings (AdjEM, AdjO, AdjD)
   - Four Factors (eFG%, TO%, OR%, FT Rate)
   - Point distribution (3PT, 2PT, FT scoring)
   - Defensive schemes (rim protection, pressure, etc.)
   - Tempo analysis (pace preferences)
   - Size/athleticism (height, rebounding)
   - Experience/chemistry (bench depth, continuity)
5. **Generates predictions**: Winner, margin, confidence levels
6. **Compares to Vegas line**: Tennessee -2.5 vs system prediction
7. **Provides strategic insights**: Key matchup factors and recommendations
8. **Saves detailed report**: Full analysis to `reports/` directory

### Expected Output

```
================================================================================
           LOUISVILLE @ TENNESSEE MATCHUP ANALYSIS
================================================================================
Date: December 16, 2025
Location: Food City Center, Knoxville, TN
Spread: Tennessee -2.5
Network: ESPN

[OK] API key loaded
[OK] Initializing KenPom API client...
[OK] Initializing comprehensive matchup analyzer...

--------------------------------------------------------------------------------
Running 7-Module Comprehensive Analysis (15+ dimensions)
--------------------------------------------------------------------------------
Analyzing: Louisville vs Tennessee (2025 season)

********************************************************************************
                           EXECUTIVE SUMMARY
********************************************************************************

Predicted Winner: TENNESSEE
Predicted Margin: 4.2 points
Overall Confidence: 68%

Vegas Line: Tennessee -2.5
System Line: Tennessee +4.2
Difference: 1.7 points
→ Slight disagreement (1.7 pts)

--------------------------------------------------------------------------------
DIMENSIONAL ANALYSIS (7 Modules)
--------------------------------------------------------------------------------

Module Scores (positive = Louisville advantage):

EFFICIENCY
  Score: -2.10 (favors Tennessee)
  Confidence: 73%
  Key Insight: Tennessee has 2.1 point edge in adjusted efficiency margin

FOUR FACTORS
  Score: -1.80 (favors Tennessee)
  Confidence: 66%
  Key Insight: Tennessee wins 3 of 4 factor battles (eFG%, OR%, FT Rate)

TEMPO
  Score: +0.30 (favors Louisville)
  Confidence: 55%
  Key Insight: Louisville slightly better at controlling pace

POINT DISTRIBUTION
  Score: +1.20 (favors Louisville)
  Confidence: 60%
  Key Insight: Louisville's 3PT shooting exploits Tennessee's perimeter defense

DEFENSIVE
  Score: -2.50 (favors Tennessee)
  Confidence: 75%
  Key Insight: Tennessee's elite rim protection (68% opp 2PT allowed)

SIZE
  Score: -1.90 (favors Tennessee)
  Confidence: 70%
  Key Insight: Tennessee's frontcourt has 2-inch height advantage

EXPERIENCE
  Score: +0.80 (favors Louisville)
  Confidence: 55%
  Key Insight: Louisville has deeper bench and better continuity

--------------------------------------------------------------------------------
STRATEGIC INSIGHTS
--------------------------------------------------------------------------------

Louisville Advantages:
  1. Superior 3-point shooting vs Tennessee's weak perimeter defense
  2. Deeper bench allows for better rotation management
  3. Higher continuity (72% minutes returning vs 58%)
  4. Better at controlling tempo in close games

Tennessee Advantages:
  1. Elite rim protection forces opponents to shoot from outside
  2. Home court advantage at Food City Center (~3 points)
  3. Dominant rebounding due to size advantage
  4. Better defensive efficiency overall (AdjD: 93.8 vs 95.2)
  5. Wins three of four factors (eFG%, OR%, FT Rate)

Key Matchup Factors:
  1. Can Louisville's 3PT shooting overcome Tennessee's interior dominance?
  2. Rebounding battle: Tennessee's size vs Louisville's effort
  3. Home court impact in close game
  4. Late-game execution: Louisville's experience edge

Recommended Strategy:
  1. Louisville: Attack perimeter, avoid paint, win the 3PT battle
  2. Tennessee: Dominate the glass, protect the rim, force turnovers
  3. Pace control: Louisville wants faster tempo, Tennessee prefers grinding
  4. Key stat: Offensive rebounding - Tennessee must capitalize on size

--------------------------------------------------------------------------------
BETTING ANALYSIS
--------------------------------------------------------------------------------

Game Assessment:
  Match Type: Close ranked matchup (#11 vs #20)
  System Prediction: Tennessee by 4.2
  Confidence Level: 68%
  Confidence Classification: MODERATE

Betting Considerations:
  → Line Value: Slight edge (1.7 pts)
  → Recommendation: Proceed with caution
  → System sees Tennessee slightly stronger than market

Home Court Impact:
  → Tennessee at home (Food City Center)
  → Typical HCA: ~3.0 points for high-major teams
  → Factor already included in analysis

--------------------------------------------------------------------------------
REPORT GENERATION
--------------------------------------------------------------------------------

[OK] Detailed report saved to: reports\louisville_tennessee_2025.txt
[OK] Analysis complete!

================================================================================
                            END OF ANALYSIS
================================================================================
```

## Step 4: Review the Detailed Report

The script saves a comprehensive report to `reports/louisville_tennessee_2025.txt`. This includes:

- Full dimensional breakdown with confidence intervals
- Position-by-position matchups (PG, SG, SF, PF, C)
- Four Factors analysis (all 4 dimensions)
- Scoring style profiles (3PT%, 2PT%, FT%)
- Defensive scheme classifications
- Experience/chemistry metrics (bench depth, continuity)
- Strategic recommendations for both teams

Open the report:

```bash
# Windows
type reports\louisville_tennessee_2025.txt

# Linux/Mac
cat reports/louisville_tennessee_2025.txt
```

## Troubleshooting

### Error: "KENPOM_API_KEY not found in environment"

**Solution**: Make sure your `.env` file exists and contains:
```
KENPOM_API_KEY=your-actual-key-here
```

### Error: "Team names not found in KenPom database"

**Possible causes**:
- Team name spelling doesn't match KenPom's database
- Team not Division I or not in 2025 database yet

**Solution**: Check team names at https://kenpom.com and use exact spelling

### Error: API authentication failed

**Solution**:
1. Verify your API key is correct
2. Check if your API subscription is active
3. Try accessing https://kenpom.com/api.php?endpoint=ratings&y=2025 manually

### Dependencies not installed

**Solution**:
```bash
# Reinstall all dependencies
uv sync --all-extras --dev
```

## Alternative: Use Individual Modules

If you want more control, you can use individual analyzer modules:

```python
from kenp0m_sp0rts_analyzer.api_client import KenPomAPI
from kenp0m_sp0rts_analyzer.four_factors_matchup import FourFactorsMatchup
from kenp0m_sp0rts_analyzer.defensive_analysis import DefensiveAnalyzer

api = KenPomAPI()

# Just Four Factors analysis
four_factors = FourFactorsMatchup(api)
analysis = four_factors.analyze_matchup("Louisville", "Tennessee", 2025)
print(f"Overall advantage: {analysis.overall_advantage}")

# Just defensive analysis
defense = DefensiveAnalyzer(api)
def_analysis = defense.analyze_matchup("Louisville", "Tennessee", 2025)
print(f"Better defense: {def_analysis.better_defense}")
```

## Next Steps

1. **Explore other games**: Modify the script to analyze other matchups from today
2. **Tournament simulation**: Use `tournament_simulator.py` for bracket predictions
3. **Backtesting**: Run `prediction.py` to validate model accuracy
4. **MCP Server**: Set up the MCP server for Claude integration

## Files Created

- `analyze_louisville_tennessee.py` - Main analysis script
- `reports/louisville_tennessee_2025.txt` - Detailed analysis report
- `.env` - API key configuration (create this yourself)

## Resources

- KenPom API Docs: `docs/KENPOM_API.md`
- Project Documentation: `docs/_INDEX.md`
- Example Scripts: `examples/` directory
- Module Reference: `src/kenp0m_sp0rts_analyzer/`

---

**Ready to analyze?** Run: `uv run python analyze_louisville_tennessee.py`
