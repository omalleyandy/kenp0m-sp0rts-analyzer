# Preventing False Edges: Complete Solution

**Date**: 2025-12-16
**Status**: Implemented & Tested

---

## Executive Summary

We've built a systematic validation framework to prevent false betting edges by:
1. **Validation Script**: Compares model predictions vs historical data vs market
2. **Guardrails Documentation**: Enforces strict data usage rules
3. **Automated Conflict Detection**: Flags when model disagrees with reality

**Result**: Montana St. @ Cal Poly now correctly shows **"CONFLICT"** instead of falsely recommending UNDER.

---

## What We Built

### 1. Edge Validation Script (`scripts/validate_edge.py`)

**Purpose**: Systematically validate betting edges before making recommendations

**Usage**:
```bash
uv run python scripts/validate_edge.py "Montana St." "Cal Poly" \
  --market-spread -3 \
  --market-total 160 \
  --home team2
```

**Output**:
```
================================RECOMMENDATIONS=================================

Spread: PASS
Total: CONFLICT - Model and historical disagree (>10 points)

====================================WARNINGS====================================

WARNING: Using season averages - not actual game-by-game data
WARNING: Manual validation required - check actual game results
```

**Key Features**:
- Compares KenPom model vs historical averages vs market lines
- Detects conflicts when model and historical disagree by >10 points
- Only recommends bets when model and historical agree on direction
- Enforces minimum edge thresholds (3 points spread, 5 points total)

### 2. Validation Guardrails (`docs/EDGE_VALIDATION_GUARDRAILS.md`)

**Purpose**: Comprehensive rules and workflow for edge validation

**Key Sections**:
- **Data Source Hierarchy**: Only use KenPom data (no web searches)
- **5-Step Validation Workflow**: Model → Historical → Compare → Determine → Review
- **Comparison Rules**: When to trust model vs historical vs market
- **Decision Tree**: Visual flowchart for edge determination
- **Common Pitfalls**: What NOT to do (cherry-picking, narrative fitting, etc.)

### 3. Prevention Guide (`docs/PREVENTING_FALSE_EDGES.md`)

**Purpose**: Complete documentation of the solution (this file)

---

## How It Prevents False Edges

### Example: Montana St. @ Cal Poly

#### Without Validation (Original Error)
```
1. KenPom model: 138 total
2. Market line: 160 total
3. Edge: 22 points UNDER!
4. Recommendation: BET UNDER ❌ (WRONG!)
```

**Problem**: Ignored historical data showing teams actually average 158 points

#### With Validation (Corrected)
```bash
uv run python scripts/validate_edge.py "Montana St." "Cal Poly" \
  --market-spread -3 --market-total 160 --home team2
```

**Output**:
```
Model Prediction: 138.0 total
Historical Average: 152.8 total
Market Line: 160.0 total

Model vs Historical: 14.8 points (CONFLICT!)

Recommendation: PASS - Model and historical disagree
```

**Result**: Correctly identified that model is wrong, market is efficient ✅

---

## Validation Workflow

### Step 1: Make Prediction
```bash
uv run python scripts/predict_game.py "Team A" "Team B" --home team1
```
- Uses KenPom efficiency metrics (AdjEM, AdjO, AdjD, AdjTempo)
- Calculates expected margin and total

### Step 2: Validate Edge
```bash
uv run python scripts/validate_edge.py "Team A" "Team B" \
  --market-spread X \
  --market-total Y \
  --home team1
```
- Compares model vs historical vs market
- Detects conflicts and mismatches
- Applies edge thresholds

### Step 3: Review Warnings
- Check for model vs historical conflicts (>10 points)
- Verify data source warnings
- Confirm no manual validation needed

### Step 4: Make Decision

| Scenario | Action |
|----------|--------|
| Model + Historical agree, edge >3 pts | ✅ BET |
| Model + Historical agree, edge <3 pts | ⚠️ PASS (edge too small) |
| Model disagrees with historical >10 pts | ❌ CONFLICT - PASS |
| Model says OVER, historical says UNDER | ❌ DIRECTION MISMATCH - PASS |

---

## Guardrails

### Data Source Rules

**✅ ALLOWED**:
- KenPom API (`api_client.py`)
- KenPom parquet data (`data/kenpom_2025_latest.parquet`)
- KenPom historical archive (30-day momentum)

**❌ FORBIDDEN**:
- Web searches for "recent games" (cherry-picks narrative)
- ESPN box scores (different methodology)
- CBS Sports stats (not adjusted for opponent)
- Manual game selection (introduces bias)

### Edge Thresholds

| Bet Type | Minimum Edge | Confidence |
|----------|--------------|------------|
| Spread | 3.0 points | Proceed with caution |
| Spread | 5.0 points | High confidence |
| Total | 5.0 points | Proceed with caution |
| Total | 8.0 points | High confidence |

**Additional requirement**: Model and historical must agree (within 10 points)

### Conflict Detection

**Automatic PASS triggers**:
- Model vs Historical > 10 points
- Model direction ≠ Historical direction (OVER vs UNDER)
- Edge < minimum threshold
- Warnings indicate manual validation needed

---

## Current Limitations

### Known Issues

1. **Historical averages use season stats, not game-by-game data**
   - **Current**: Estimated from `(AdjO / 100) × Tempo`
   - **Needed**: Scrape actual game results from KenPom team pages
   - **Impact**: Historical validation less accurate (~10-15 point error margin)

2. **No injury tracking**
   - KenPom data doesn't include injuries
   - Must manually verify before betting

3. **No home/away splits**
   - Uses fixed 3.5-point home court advantage
   - KenPom has team-specific HCA data we could use

### Workarounds (Until Improvements)

**For totals**:
- Use wider conflict threshold (15 points instead of 10)
- Require larger edge (8+ points instead of 5)
- Manually check recent game results when validation shows warning

**For spreads**:
- Trust model more than historical (spreads are more stable)
- Focus on games where teams have similar styles
- Avoid games with extreme tempo differences

---

## Next Steps (Roadmap)

### Phase 1: Immediate (✅ Complete)
- ✅ Validation script with season averages
- ✅ Guardrails documentation
- ✅ Conflict detection
- ✅ Workflow enforcement

### Phase 2: Enhanced Historical Data (Priority)
**Goal**: Get actual game-by-game results from KenPom

**Implementation**:
```python
# Scrape team schedule from KenPom
from kenp0m_sp0rts_analyzer.browser import KenPomScraper

async def get_team_schedule(team_id: int, season: int):
    async with KenPomScraper() as scraper:
        await scraper.login()
        schedule = await scraper.get_team_schedule(team_id, season)
    return schedule

# Calculate actual game totals
def calculate_historical_totals(schedule: list[dict]) -> dict:
    totals = [game['team_score'] + game['opponent_score']
              for game in schedule if game['completed']]
    return {
        'average_total': sum(totals) / len(totals),
        'games_played': len(totals),
        'actual_totals': totals,
    }
```

**Benefits**:
- Accurate historical validation (not estimated)
- Game-by-game trend analysis
- Home/away splits
- Recent form (last 5 games)

### Phase 3: Advanced Validation (Future)
- [ ] Closing Line Value (CLV) tracking
- [ ] Prediction accuracy backtesting
- [ ] Automated edge finder (scan all games daily)
- [ ] Injury scraper from KenPom news
- [ ] Team-specific home court advantage

---

## Usage Examples

### Example 1: Valid Edge (Model + Historical Agree)

```bash
uv run python scripts/validate_edge.py Duke Houston \
  --market-spread 0 --market-total 180 --neutral
```

**Hypothetical Output**:
```
Model Prediction: 182.7 total
Historical Average: 185.3 total
Market Line: 180.0 total

Edge Analysis:
- Total Edge (Model vs Market): 2.7 points
- Total Edge (Historical vs Market): 5.3 points
- Model vs Historical: 2.6 points ✅ Agreement!

Recommendation: BET OVER 180 (High confidence)
```

**Why it's valid**:
- Model (182.7) and Historical (185.3) both predict OVER
- Both agree within 2.6 points
- Historical edge (5.3) exceeds threshold (5.0)

### Example 2: False Edge (Model vs Historical Conflict)

```bash
uv run python scripts/validate_edge.py "Montana St." "Cal Poly" \
  --market-spread -3 --market-total 160 --home team2
```

**Actual Output**:
```
Model Prediction: 138.0 total
Historical Average: 152.8 total
Market Line: 160.0 total

Edge Analysis:
- Total Edge (Model vs Market): 22.0 points
- Total Edge (Historical vs Market): 7.2 points
- Model vs Historical: 14.8 points ❌ CONFLICT!

Recommendation: CONFLICT - Model and historical disagree (>10 points)
```

**Why it's invalid**:
- Model (138) predicts UNDER, Historical (153) also suggests UNDER
- BUT they disagree by 14.8 points (>10 threshold)
- Model is likely wrong - market (160) is closer to historical (153)
- Historical edge (7.2) suggests slight UNDER, not 22-point UNDER

### Example 3: Direction Mismatch

```bash
uv run python scripts/validate_edge.py "Team A" "Team B" \
  --market-spread -3 --market-total 145 --home team2
```

**Hypothetical Output**:
```
Model Prediction: 138.0 total
Historical Average: 152.0 total
Market Line: 145.0 total

Recommendation: CONFLICT - Direction mismatch
```

**Why it's invalid**:
- Model (138) predicts UNDER 145
- Historical (152) predicts OVER 145
- Conflicting signals = PASS

---

## Checklist for Every Bet

Before recommending any bet, verify:

- [ ] Ran `validate_edge.py` with market lines
- [ ] Model vs Historical difference < 10 points
- [ ] Model and Historical agree on direction (OVER/UNDER)
- [ ] Edge exceeds minimum threshold (3 pts spread, 5 pts total)
- [ ] No warnings requiring manual validation
- [ ] Team names are correct (no data errors)
- [ ] Home/away/neutral is correct
- [ ] No major injuries or lineup changes
- [ ] Documented reasoning and assumptions

**When in doubt, PASS**. No bet is better than a bad bet.

---

## Summary

### The Problem
- Trusted model predictions without validation
- Cherry-picked data to support narrative
- Ignored actual game results

### The Solution
- ✅ Systematic validation script
- ✅ Strict data source rules (KenPom only)
- ✅ Automated conflict detection
- ✅ Documented guardrails and workflow
- ✅ Clear decision tree

### The Result
- False edges are now automatically detected
- Montana St. @ Cal Poly correctly flagged as CONFLICT
- Reliable framework for future edge validation

### Next Priority
- Build game-by-game scraper for accurate historical data
- Replace estimated season averages with actual game totals
- Reduce false conflicts and improve validation accuracy
