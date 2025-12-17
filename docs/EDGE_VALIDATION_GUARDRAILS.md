# Edge Validation Guardrails

**Created**: 2025-12-16
**Purpose**: Prevent false edges by enforcing systematic validation

---

## The Problem

When comparing model predictions to market lines, we can easily fall into traps:
- Cherry-picking data to support a narrative
- Ignoring actual game results that contradict the model
- Trusting model predictions without validation
- Using non-KenPom data sources inconsistently

**Example Error** (Montana St. @ Cal Poly):
- KenPom model predicted: 138 total
- Historical average: 158 total (Montana St: 147, Cal Poly: 169)
- Market line: 160 total
- **Initial conclusion**: UNDER 160 (22-point edge!)
- **Reality**: Model was wrong, market was efficient

---

## Guardrails

### 1. Data Source Hierarchy

**ONLY use KenPom data for predictions and validation:**

| Priority | Data Source | Use Case | Status |
|----------|-------------|----------|--------|
| 1 | **KenPom API** | Team metrics, season averages | ✅ Implemented |
| 2 | **KenPom team schedules** | Actual game results | ⚠️ Need to implement scraper |
| 3 | **KenPom historical archive** | 30-day momentum, trends | ✅ Implemented |

**FORBIDDEN**:
- ❌ Web searches for "recent games" (cherry-picks narrative)
- ❌ ESPN box scores (not adjusted for opponent quality)
- ❌ CBS Sports stats (different methodology)
- ❌ Manual game selection (introduces bias)

### 2. Validation Workflow

**Every edge claim MUST follow this workflow:**

```
Step 1: Model Prediction (KenPom metrics)
   ↓
Step 2: Historical Validation (Actual game results)
   ↓
Step 3: Comparison
   ↓
Step 4: Edge Determination
   ↓
Step 5: Manual Review
```

#### Step 1: Model Prediction
```bash
uv run python scripts/predict_game.py "Team A" "Team B" --home team1
```

**Output**: KenPom model prediction using efficiency metrics

#### Step 2: Historical Validation
```bash
uv run python scripts/validate_edge.py "Team A" "Team B" \
  --market-spread -3 \
  --market-total 160 \
  --home team1
```

**Output**: Comparison of model vs historical averages vs market

#### Step 3: Comparison Rules

| Scenario | Action |
|----------|--------|
| Model agrees with historical (within 5 points) | ✅ Trust the prediction |
| Model and historical both differ from market (>3 points, same direction) | ✅ Likely edge |
| Model disagrees with historical (>10 points) | ⚠️ WARNING - Investigate |
| Model says UNDER, historical says OVER | ❌ PASS - Conflicting signals |

#### Step 4: Edge Determination

**Only recommend a bet if ALL conditions are met:**

- [ ] Model edge ≥ 3 points
- [ ] Historical average agrees with model direction
- [ ] Model vs Historical difference < 10 points
- [ ] No major injuries or lineup changes
- [ ] No conflicting signals

#### Step 5: Manual Review Checklist

Before finalizing edge recommendation:

- [ ] Check model prediction calculation
- [ ] Verify historical averages are from FULL schedule (not cherry-picked games)
- [ ] Confirm no data errors (team names, home/away, etc.)
- [ ] Review warnings from validation script
- [ ] Document any assumptions or limitations

---

## Validation Script Usage

### Basic Usage
```bash
uv run python scripts/validate_edge.py "Montana St." "Cal Poly" \
  --market-spread -3 \
  --market-total 160 \
  --home team2
```

### Output Interpretation

**Model Prediction Section**:
- KenPom efficiency-based prediction
- Uses AdjEM, AdjO, AdjD, AdjTempo

**Historical Averages Section**:
- Season averages for both teams
- Estimated points scored/allowed per game
- ⚠️ Warning: Currently uses season averages, not actual game-by-game results

**Edge Analysis Section**:
- Spread edge: |Model - Market|
- Total edge (Model vs Market): |Model Total - Market Total|
- Total edge (Historical vs Market): |Historical Total - Market Total|
- Model vs Historical: |Model Total - Historical Total|

**Recommendations**:
- Spread: BET / PASS
- Total: BET OVER / BET UNDER / CONFLICT / PASS

---

## Edge Thresholds

### Minimum Edge Requirements

| Bet Type | Minimum Edge | Confidence |
|----------|--------------|------------|
| Spread | 3.0 points | Medium |
| Spread | 5.0 points | High |
| Total | 5.0 points | Medium |
| Total | 8.0 points | High |

**Additional requirement for totals**: Model and historical must agree on direction (both OVER or both UNDER)

---

## Common Pitfalls

### ❌ Pitfall 1: Cherry-Picking Games
**Bad**: "Team A scored 124 in their last game, so UNDER is good"
**Good**: "Team A's 11-game average is 147 points, historical total is 158"

### ❌ Pitfall 2: Ignoring Historical Data
**Bad**: "KenPom says 138, market is 160, bet UNDER!"
**Good**: "KenPom says 138, but historical is 158. Model disagrees with reality - PASS"

### ❌ Pitfall 3: Narrative Fitting
**Bad**: "This game will be slow because Team A is #263 in tempo"
**Good**: "Team A's actual games average 147 points despite slow tempo rating"

### ❌ Pitfall 4: Using Non-KenPom Data
**Bad**: Web searching for "injuries" and "recent games"
**Good**: Using KenPom's actual schedule data and roster information

---

## Current Limitations

### Known Issues

1. **Historical averages use season stats, not actual game results**
   - Current: Estimated from AdjO/AdjD × Tempo
   - Needed: Actual game-by-game scraping from KenPom team pages
   - Impact: Less accurate historical validation

2. **No injury tracking**
   - KenPom data doesn't include injuries
   - Must manually verify no major injuries before betting
   - Consider building injury scraper from KenPom news/notes

3. **No home/away splits**
   - Current: Single home court adjustment (3.5 points)
   - Needed: Team-specific home court advantage from KenPom HCA data
   - Impact: Spread predictions may be less accurate

### Roadmap

**Phase 1** (Immediate):
- ✅ Validation script with season averages
- ✅ Guardrails documentation
- ✅ Workflow enforcement

**Phase 2** (Next):
- [ ] Scrape actual game results from KenPom team pages
- [ ] Calculate true historical game totals
- [ ] Add game-by-game validation

**Phase 3** (Future):
- [ ] Track prediction accuracy over time
- [ ] Build backtesting framework
- [ ] Implement Closing Line Value (CLV) tracking

---

## Decision Tree

```
Found potential edge?
    ↓
Run validation script
    ↓
Model vs Historical > 10 points?
    ↓ YES                    ↓ NO
    ↓                        ↓
⚠️ INVESTIGATE              Continue
(Model likely wrong)          ↓
                        Model + Historical agree on direction?
                            ↓ YES                    ↓ NO
                            ↓                        ↓
                        Edge > threshold?        ❌ PASS
                            ↓ YES    ↓ NO      (Conflicting signals)
                            ↓        ❌ PASS
                        ✅ BET      (Edge too small)
                    (Document reasoning)
```

---

## Example: Montana St. @ Cal Poly (Corrected)

### Step 1: Model Prediction
```bash
uv run python scripts/predict_game.py "Montana St." "Cal Poly" --home team2
```
- **KenPom Total**: 138.0 points

### Step 2: Historical Validation
```bash
uv run python scripts/validate_edge.py "Montana St." "Cal Poly" \
  --market-spread -3 --market-total 160 --home team2
```
- **Montana St. Historical**: 147 PPG average
- **Cal Poly Historical**: 169 PPG average
- **Combined Historical**: 158 points

### Step 3: Comparison
- Model: 138
- Historical: 158
- Market: 160
- **Model vs Historical**: 20 points (HUGE discrepancy!)

### Step 4: Edge Determination
- ❌ Model and historical disagree by 20 points
- ⚠️ WARNING: Model prediction unreliable
- ✅ Market (160) is very close to historical (158)
- **Conclusion**: PASS - No edge, market is efficient

---

## Summary

**Core Principle**: Trust the data, not the narrative.

**Golden Rules**:
1. **Always validate**: Never trust model predictions without historical validation
2. **Use KenPom only**: No cherry-picking from web searches
3. **Require agreement**: Model and historical must agree on direction
4. **Document everything**: Record assumptions, warnings, and reasoning
5. **When in doubt, pass**: No bet is better than a bad bet

**Remember**: The market is often right. If your model disagrees with both the market AND historical data, your model is probably wrong.
