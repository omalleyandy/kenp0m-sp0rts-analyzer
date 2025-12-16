# KenPom Data Coverage Analysis

**Comprehensive review of KenPom stats categories and implementation status**

Generated: 2025-12-16

---

## Executive Summary

This analysis identifies which KenPom statistics are:
1. ✅ **Fully Implemented** - API access + analyzer module
2. 🟡 **Partially Implemented** - API access but no dedicated analyzer
3. ❌ **Not Implemented** - Requires web scraping (not available via API)

---

## KenPom Official API Coverage

The official KenPom API provides **9 endpoints** with comprehensive data access:

| Endpoint | Status | Analyzer Module | Priority |
|----------|--------|----------------|----------|
| `ratings` | ✅ Complete | tempo_analysis.py, analysis.py, prediction.py | Core |
| `four-factors` | ✅ Complete | four_factors_matchup.py | High |
| `fanmatch` | ✅ Complete | analysis.py, prediction.py | High |
| `archive` | ✅ Complete | Multiple modules | Medium |
| `teams` | ✅ Complete | Multiple modules | Core |
| `conferences` | ✅ Complete | conference_analytics.py | Medium |
| **`pointdist`** | 🟡 **API Only** | **MISSING** | **HIGH** |
| **`height`** | 🟡 **API Only** | **MISSING** | **MEDIUM** |
| **`misc-stats`** | 🟡 **Partial** | **INCOMPLETE** | **HIGH** |

---

## 1. Point Distribution (pointdist) 🟡 HIGH PRIORITY

### Current Status
- ✅ API endpoint available: `api.get_point_distribution()`
- ❌ No dedicated analyzer module
- ❌ Not integrated into matchup analysis

### Available Data
```python
# Offense
OffFt: float       # % of points from free throws
OffFg2: float      # % of points from 2-point FGs
OffFg3: float      # % of points from 3-point FGs

# Defense
DefFt: float       # % of points allowed from free throws
DefFg2: float      # % of points allowed from 2-point FGs
DefFg3: float      # % of points allowed from 3-point FGs

# All with corresponding ranks
```

### Strategic Value
- **Scoring Style Identification**: Inside vs perimeter teams
- **Defensive Vulnerabilities**: Which shot types defense struggles with
- **Matchup Advantages**: 3pt-heavy offense vs poor 3pt defense
- **Adjustment Opportunities**: Exploit opponent's weakest defensive area

### Recommended Implementation
**File**: `src/kenp0m_sp0rts_analyzer/point_distribution_analysis.py`

```python
class PointDistributionMatchup:
    """Analyze scoring style matchups and defensive vulnerabilities."""

    def analyze_matchup(self, team1: str, team2: str, season: int) -> dict:
        """
        Identify scoring style advantages:

        1. Three-Point Battle:
           - Team1 3pt offense (OffFg3) vs Team2 3pt defense (DefFg3)
           - Identify if 3pt shooting team faces weak 3pt defense

        2. Paint Battle:
           - Team1 2pt offense vs Team2 2pt defense
           - Size advantages for inside scoring

        3. Free Throw Battle:
           - FT drawing ability vs foul-prone defense
           - FT shooting % matchups

        4. Style Mismatch Score:
           - Quantify offensive style vs defensive vulnerability
           - Recommend game plan adjustments
        """

    def identify_scoring_vulnerabilities(self, team: str) -> dict:
        """Identify which shot types team allows most."""

    def recommend_offensive_strategy(self, offense: str, defense: str) -> str:
        """Generate strategic recommendation based on point distribution."""
```

**Integration Points**:
- Add to `FourFactorsMatchup` as complementary analysis
- Include in `analyze_matchup()` comprehensive reports
- Enhance `GamePredictor` feature engineering

---

## 2. Height & Experience (height) 🟡 MEDIUM PRIORITY

### Current Status
- ✅ API endpoint available: `api.get_height()`
- ❌ No dedicated analyzer module
- ❌ Not used in matchup predictions

### Available Data
```python
# Height Metrics
AvgHgt: float      # Average team height in inches
HgtEff: float      # Effective height (weighted by minutes)
Hgt5: float        # Center position height
Hgt4: float        # Power forward height
Hgt3: float        # Small forward height
Hgt2: float        # Shooting guard height
Hgt1: float        # Point guard height

# Team Composition
Exp: float         # Experience rating
Bench: float       # Bench strength rating
Continuity: float  # Team continuity rating

# All with corresponding ranks
```

### Strategic Value
- **Size Matchups**: Identify height advantages at each position
- **Rebounding Predictions**: Correlate height with OR%/DR%
- **Experience Edge**: Tournament performance, late-game execution
- **Bench Depth**: Fatigue resistance, rotation adjustments
- **Chemistry Factor**: Continuity impact on team performance

### Recommended Implementation
**File**: `src/kenp0m_sp0rts_analyzer/height_experience_analysis.py`

```python
class HeightExperienceAnalyzer:
    """Analyze height matchups and experience advantages."""

    def analyze_size_matchup(self, team1: str, team2: str, season: int) -> dict:
        """
        Position-by-position height analysis:
        - PG, SG, SF, PF, C height comparisons
        - Overall height advantage
        - Effective height (minutes-weighted)
        """

    def predict_rebounding_advantage(self, team1: str, team2: str) -> dict:
        """Correlate height metrics with rebounding percentages."""

    def analyze_experience_edge(self, team1: str, team2: str) -> dict:
        """
        Experience advantages:
        - Tournament games (clutch factor)
        - Late-season execution
        - Coaching adjustments
        """

    def evaluate_bench_strength(self, team1: str, team2: str) -> dict:
        """Bench depth and rotation advantages."""
```

**Integration Points**:
- Enhance rebounding predictions in `FourFactorsMatchup`
- Add experience factor to tournament simulations
- Include in comprehensive scouting reports

---

## 3. Miscellaneous Stats (misc-stats) 🟡 HIGH PRIORITY

### Current Status
- ✅ API endpoint available: `api.get_misc_stats()`
- 🟡 Partially used in existing modules
- ❌ No dedicated defensive analyzer

### Available Data
```python
# Shooting Percentages
FG3Pct: float          # 3-point FG %
FG2Pct: float          # 2-point FG %
FTPct: float           # Free throw %

# Advanced Metrics
BlockPct: float        # Block percentage
StlRate: float         # Steal rate
NSTRate: float         # Non-steal turnover rate
ARate: float           # Assist rate
F3GRate: float         # 3-point attempt rate

# Defensive (Opp prefix)
OppFG3Pct: float       # Opponent 3pt %
OppFG2Pct: float       # Opponent 2pt %
OppBlockPct: float     # Opponent blocks
OppStlRate: float      # Opponent steals
OppARate: float        # Opponent assists

# All with corresponding ranks
```

### Strategic Value
- **Shot Selection Analysis**: 3pt attempt rate trends
- **Defensive Pressure**: Steal rates, forced turnovers
- **Interior Defense**: Block percentages
- **Ball Movement**: Assist rates (offensive flow)
- **Defensive Versatility**: Multi-dimensional defense metrics

### Recommended Implementation
**File**: `src/kenp0m_sp0rts_analyzer/defensive_analysis.py`

```python
class DefensiveAnalyzer:
    """Advanced defensive matchup analysis."""

    def analyze_perimeter_defense(self, team1: str, team2: str) -> dict:
        """
        3-point defense analysis:
        - OppFG3Pct (defense ability)
        - F3GRate (opponent's 3pt frequency)
        - Matchup: 3pt-heavy offense vs 3pt defense
        """

    def analyze_interior_defense(self, team1: str, team2: str) -> dict:
        """
        Paint defense:
        - OppFG2Pct (2pt defense)
        - BlockPct (shot alteration)
        - Rim protection effectiveness
        """

    def analyze_pressure_defense(self, team1: str, team2: str) -> dict:
        """
        Turnover creation:
        - StlRate (steal pressure)
        - Opponent ball security
        - Press vulnerability
        """

    def identify_defensive_scheme(self, team: str) -> dict:
        """
        Classify defensive identity:
        - Block-heavy (rim protection)
        - Steal-heavy (pressure/gambling)
        - Balanced (versatile)
        """
```

**Integration Points**:
- Enhance `FourFactorsMatchup` with defensive scheme analysis
- Add to `analyze_matchup()` for comprehensive breakdowns
- Improve `GamePredictor` with defensive style features

---

## 4. Player Stats (playerstats.php) ❌ NOT AVAILABLE VIA API

### Current Status
- ❌ Not available in official KenPom API
- ❌ Would require web scraping via `KenPomScraper`
- 🟡 Basic player impact modeling exists (`player_impact.py`)

### Potential Data (via Scraping)
- Individual player efficiency ratings
- Usage rates and shot distribution
- Defensive impact metrics
- Minutes played and rotation patterns

### Recommendation
**SKIP FOR NOW** - Focus on team-level analytics first

**Rationale**:
1. Official API doesn't provide player stats
2. Web scraping is fragile (requires auth, page structure changes)
3. Team-level analysis already comprehensive
4. KenPom primarily focuses on team metrics

**Future Consideration**:
- If user requests player-level analysis, implement scraper
- Integrate with existing `player_impact.py` module
- Use for injury impact quantification

---

## 5. Summary Page (summary.php) ✅ FULLY COVERED

### Current Status
✅ **Data already accessible via combined endpoints**

The summary page aggregates data from multiple endpoints:
- Efficiency metrics → `ratings` endpoint
- Four Factors → `four-factors` endpoint
- Shooting stats → `misc-stats` endpoint

**No additional implementation needed** - all data points available through existing API endpoints.

---

## Implementation Priority Roadmap

### 🔴 TIER 1: High Impact, Quick Implementation

| Module | Effort | Impact | Status | Next Sprint |
|--------|--------|--------|--------|-------------|
| **Point Distribution Analyzer** | Low | High | 🟡 Missing | ✅ **YES** |
| **Defensive Analyzer (misc-stats)** | Medium | High | 🟡 Partial | ✅ **YES** |

**Why These First?**:
- Complement existing `FourFactorsMatchup` perfectly
- Low implementation effort (API already available)
- High strategic value for matchup analysis
- Direct enhancement to game predictions

---

### 🟡 TIER 2: Medium Impact, Moderate Effort

| Module | Effort | Impact | Status | Later Sprint |
|--------|--------|--------|--------|--------------|
| **Height/Experience Analyzer** | Medium | Medium | 🟡 Missing | Later |

**Why Later?**:
- Useful but less critical than scoring/defense analysis
- Moderate complexity (position-by-position comparisons)
- Best integrated after core matchup analytics complete

---

### 🟢 TIER 3: Lower Priority

| Module | Effort | Impact | Status | Future |
|--------|--------|--------|--------|--------|
| **Player Stats Scraper** | High | Low-Med | ❌ Missing | TBD |

**Why Lowest?**:
- Not available via official API (requires scraping)
- High maintenance burden (website changes)
- Team-level analysis already comprehensive

---

## Recommended Next Steps

### Immediate (This Session):
1. ✅ Create `point_distribution_analysis.py`
   - Implement `PointDistributionMatchup` class
   - Analyze scoring style vs defensive vulnerabilities
   - Generate strategic recommendations

2. ✅ Create `defensive_analysis.py`
   - Implement `DefensiveAnalyzer` class
   - Perimeter defense (3pt), interior defense (2pt/blocks), pressure defense (steals)
   - Classify defensive schemes

3. ✅ Create demo scripts showing new analysis capabilities

### Short-Term (Next Session):
4. Integrate new analyzers into existing workflows:
   - Add to `analyze_matchup()` comprehensive reports
   - Enhance `FourFactorsMatchup` with complementary insights
   - Update `GamePredictor` feature engineering

5. Create `height_experience_analysis.py` when ready

---

## Current System Strengths

### What We Already Have ✅

1. **Core Efficiency Analysis** (ratings endpoint)
   - `tempo_analysis.py`: Comprehensive pace analysis
   - `analysis.py`: Basic matchup predictions
   - `prediction.py`: ML-based predictions with confidence intervals

2. **Four Factors Strategic Analysis** (four-factors endpoint)
   - `four_factors_matchup.py`: Dean Oliver's weighted analysis
   - Offense vs defense matchup identification

3. **Conference Analytics** (conferences endpoint)
   - `conference_analytics.py`: Power ratings, head-to-head

4. **Game Predictions** (fanmatch endpoint)
   - Used in `analysis.py` and `prediction.py`

5. **Player Impact Modeling** (derived metrics)
   - `player_impact.py`: Injury impact quantification

---

## Data Completeness by KenPom Category

### kenpom.com/summary.php → ✅ FULLY COVERED
- **AdjEM, AdjO, AdjD**: `ratings` endpoint → Used everywhere
- **Four Factors**: `four-factors` endpoint → `four_factors_matchup.py`
- **Tempo/Pace**: `ratings` (AdjTempo, APL) → `tempo_analysis.py`

### kenpom.com/stats.php → ✅ FULLY COVERED
- **Four Factors page**: `four-factors` endpoint → `four_factors_matchup.py`

### kenpom.com/teamstats.php → 🟡 PARTIAL (misc-stats)
- **Shooting stats**: `misc-stats` endpoint → Used in predictions
- **Blocks/Steals**: `misc-stats` endpoint → **Needs dedicated analyzer**

### kenpom.com/pointdist.php → 🟡 API ONLY
- **Point distribution**: `pointdist` endpoint → **Needs analyzer module**

### kenpom.com/height.php → 🟡 API ONLY
- **Height/Experience**: `height` endpoint → **Needs analyzer module**

### kenpom.com/playerstats.php → ❌ NOT AVAILABLE
- **Player stats**: Not in API → Would require scraping

---

## Summary

**Current Coverage**: 6/9 API endpoints fully utilized with analyzer modules

**Quick Wins** (TIER 1):
1. Point Distribution Analyzer (scoring style matchups)
2. Defensive Analyzer (misc-stats deep-dive)

**Medium-Term** (TIER 2):
3. Height/Experience Analyzer (size matchups, bench depth)

**Low Priority** (TIER 3):
4. Player Stats Scraper (only if specifically requested)

**Expected Outcome**: Near-complete coverage of team-level KenPom analytics with strategic matchup insights across all dimensions (efficiency, pace, four factors, scoring style, defense, size).
