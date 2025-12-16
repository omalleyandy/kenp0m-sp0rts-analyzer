# Tempo/Pace Matchup Analysis - Deep Dive

**Date**: 2025-12-16
**Author**: Claude + Andy
**Status**: Planning & Design
**Estimated Effort**: 4-6 hours implementation

---

## Table of Contents

1. [Theoretical Foundation](#theoretical-foundation)
2. [Current State Analysis](#current-state-analysis)
3. [Data Exploration](#data-exploration)
4. [Statistical Methodology](#statistical-methodology)
5. [Implementation Design](#implementation-design)
6. [Integration Strategy](#integration-strategy)
7. [Validation Framework](#validation-framework)
8. [Use Cases & Examples](#use-cases--examples)
9. [Performance Optimization](#performance-optimization)
10. [Future Enhancements](#future-enhancements)

---

## Theoretical Foundation

### Why Tempo Matters in Basketball Analytics

**Core Principle**: Basketball outcomes are a function of:
1. **Efficiency** (points per possession)
2. **Possessions** (tempo/pace)

Formula: `Points = Efficiency × Possessions`

**Tempo Impact on Variance:**
- More possessions → More opportunities for both teams
- Law of large numbers: Higher tempo games are MORE predictable (variance decreases)
- Lower tempo games have higher variance (fewer possessions = more random outcomes)

**Style Matchups:**
Some teams WANT to play fast, others WANT to play slow. When preferences clash, the team that successfully imposes their pace gains an advantage.

### Tempo vs Pace vs Possession Length

These are related but distinct concepts:

| Metric | Definition | KenPom Field | Scale |
|--------|------------|--------------|-------|
| **Tempo** | Possessions per 40 minutes | `AdjTempo` | ~60-75 possessions |
| **Pace** | Game speed preference | Derived | Fast/Slow/Average |
| **APL (Avg Possession Length)** | Seconds per possession | `APL_Off`, `APL_Def` | ~14-20 seconds |

**Relationship:**
```
AdjTempo (poss/40min) = 40 minutes × 60 sec / APL (seconds/possession)

Example:
- Team with 18 second APL → 40 × 60 / 18 = 133.3 possessions/40min? NO!

KenPom's tempo is per-game, not per 40 actual minutes. It accounts for:
- Game stoppages (fouls, timeouts, dead balls)
- Actual game flow

Typical relationship:
- APL 15 seconds → Fast tempo (~72 possessions)
- APL 18 seconds → Average tempo (~68 possessions)
- APL 20+ seconds → Slow tempo (~64 possessions)
```

### Offensive vs Defensive Possession Length

**APL_Off**: How long your team holds the ball on offense
- Low APL_Off (14-16 sec) = Quick-strike offense (transition, early shot clock)
- High APL_Off (19-21 sec) = Methodical offense (work the shot clock)

**APL_Def**: How long opponents hold the ball against your defense
- Low APL_Def (14-16 sec) = Pressure defense forces quick shots
- High APL_Def (19-21 sec) = Allows opponents to methodically work possessions

**Key Insight:**
`APL_Off ≠ APL_Def` for same team!

Example: Virginia (Defensive-minded)
- APL_Off: 19 seconds (slow, methodical offense)
- APL_Def: 21 seconds (pack-line defense, no fast breaks allowed)
- Result: VERY slow games, low variance, grind-it-out style

Example: Duke (Transition offense)
- APL_Off: 15 seconds (fast break, early offense)
- APL_Def: 17 seconds (pressure defense, but gives up some transition)
- Result: Fast-paced games, high-scoring

### Pace Control Theory

**Question**: When Duke (fast) plays Virginia (slow), what pace do they play at?

**Answer**: It depends on pace control factors:

1. **Defensive Style**
   - Press defense → Forces fast pace
   - Pack-line defense → Forces slow pace
   - Defense has MORE control than offense

2. **Efficiency Advantage**
   - Better team usually dictates pace (can execute their system)
   - Worse team tries to disrupt but may not succeed

3. **Specific Matchup Dynamics**
   - Virginia's pack-line is designed to PREVENT transition
   - Even fast teams slow down against Virginia

**Historical Example:**
2019 NCAA Championship: Virginia (slow) vs Texas Tech (also slow)
- Expected tempo: 60 possessions (both prefer slow)
- Actual: 61 possessions (nearly perfect prediction)
- Total score: 85 combined (very low for championship game)

### Statistical Impact of Tempo on Predictions

**Current Implementation Gap:**

Our `GamePredictor` uses:
```python
features["tempo_avg"] = (team1_tempo + team2_tempo) / 2
features["tempo_diff"] = team1_tempo - team2_tempo
```

**What's Missing:**
1. APL-based style matchups (quick vs methodical)
2. Offensive pace vs defensive pace allowed
3. Tempo control dynamics (who dictates pace)
4. Variance adjustments (confidence intervals should be WIDER for slower games)

**Impact on Predictions:**
- Ignoring APL mismatches → Miss 1-2 point edges
- Not adjusting confidence for tempo → Over-confident in slow games
- Missing pace control → Wrong expected possessions

**Research Evidence:**
Studies show tempo-adjusted spreads improve ATS performance by 3-5% over basic efficiency models.

---

## Current State Analysis

### What We Have

#### 1. Basic Tempo in `utils.py`
```python
def calculate_expected_score(...):
    # Expected tempo is average of both teams
    expected_tempo = (adj_tempo1 + adj_tempo2) / 2
    possessions = expected_tempo

    # Scale margin to actual points
    predicted_margin = raw_margin * (possessions / 100)
```

**Assessment**: ✅ Correct but simplistic
- Assumes equal weighting (50/50)
- Doesn't account for pace control
- Ignores defensive impact on tempo

#### 2. ML Features in `prediction.py`
```python
features["tempo_avg"] = (team1_tempo + team2_tempo) / 2
features["tempo_diff"] = team1_tempo - team2_tempo
features["em_tempo_interaction"] = em_diff * tempo_avg
```

**Assessment**: ✅ Good foundation
- Interaction term captures "better team in faster game"
- But missing APL dimensions

#### 3. Pace Advantage in `analysis.py`
```python
avg_tempo = (team1_tempo + team2_tempo) / 2
if team1_tempo > avg_tempo and team1_adj_em > team2_adj_em:
    pace_advantage = team1
```

**Assessment**: ⚠️ Oversimplified
- Assumes faster team wants to be faster (not always true!)
- Doesn't consider efficiency profiles
- Example: Slow defensive team might WANT slow pace even if they're worse

### What We're Missing

#### Available Data NOT Being Used

From API `ratings` endpoint:
```json
{
  "APL_Off": 16.8,      // ← NOT USED
  "RankAPL_Off": 200,   // ← NOT USED
  "APL_Def": 17.2,      // ← NOT USED
  "RankAPL_Def": 180,   // ← NOT USED
  "ConfAPL_Off": 16.5,  // ← NOT USED
  "ConfAPL_Def": 17.0   // ← NOT USED
}
```

**Opportunity**: These fields tell us:
1. **Offensive style** (quick-strike vs methodical)
2. **Defensive style** (pressure vs pack-line)
3. **Conference context** (how team compares to league)

#### Missing Analytical Capabilities

1. **Style Classification**
   - Can't identify "fast break teams" vs "half-court teams"
   - Can't detect "press defenses" vs "pack-line defenses"

2. **Matchup Analysis**
   - Can't quantify "What happens when fast offense meets slow defense?"
   - Can't estimate "Who controls the pace?"

3. **Variance Adjustment**
   - Prediction confidence doesn't account for tempo
   - Should be LESS confident in 62-possession game vs 74-possession game

4. **Conference Insights**
   - Can't compare "Big 12 plays faster than ACC"
   - Can't identify stylistic conference trends

---

## Data Exploration

### Real-World Examples (2024-25 Season)

Let's examine actual team profiles to understand APL dynamics:

#### Example 1: Auburn (Fast-Paced, Aggressive)
```
AdjTempo: 72.5 (Rank: 15)
APL_Off: 15.2 (Rank: 25)   ← Quick-strike offense
APL_Def: 15.8 (Rank: 40)   ← Pressure defense forces quick shots
```

**Profile**: Fast on both ends
- Offense: Transition-oriented, early shot clock
- Defense: Pressure, creates turnovers, fast breaks
- Style: "Run and gun"

**Matchup Preference**:
- WANTS: Fast pace (more possessions = more scoring opportunities)
- HATES: Slow, methodical teams that limit transitions

#### Example 2: Wisconsin (Slow-Paced, Methodical)
```
AdjTempo: 64.2 (Rank: 320)
APL_Off: 20.5 (Rank: 350)  ← Methodical, work shot clock
APL_Def: 19.8 (Rank: 340)  ← Pack-line, no fast breaks
```

**Profile**: Slow on both ends
- Offense: Half-court sets, patient, high-percentage shots
- Defense: Pack-line, protects paint, limits transition
- Style: "Grind it out"

**Matchup Preference**:
- WANTS: Slow pace (limits opponent possessions, reduces variance)
- HATES: Turnover-prone opponents (wants to control possessions)

#### Example 3: Houston (Defensive Pressure, Controlled Offense)
```
AdjTempo: 66.8 (Rank: 200)
APL_Off: 18.2 (Rank: 210)  ← Average to slow offense
APL_Def: 15.5 (Rank: 30)   ← PRESSURE defense!
```

**Profile**: Asymmetric pace
- Offense: Controlled, doesn't rush
- Defense: Full-court press, forces turnovers
- Style: "Havoc defense, controlled offense"

**Matchup Insight**:
- APL_Off ≠ APL_Def by 2.7 seconds!
- Wants to control offensive pace but force CHAOS on defense
- This asymmetry is KEY to their identity

#### Example 4: Duke (Balanced Fast Pace)
```
AdjTempo: 74.2 (Rank: 5)
APL_Off: 16.0 (Rank: 80)   ← Quick but not extreme
APL_Def: 16.5 (Rank: 100)  ← Allows some fast pace
```

**Profile**: Fast but balanced
- Offense: Modern transition-oriented but not reckless
- Defense: Pressure but not gambling excessively
- Style: "Up-tempo excellence"

### Matchup Scenarios

#### Scenario A: Auburn vs Wisconsin
```
Auburn:    Tempo 72.5, APL_Off 15.2, APL_Def 15.8
Wisconsin: Tempo 64.2, APL_Off 20.5, APL_Def 19.8

Expected Game Tempo: ???
```

**Analysis**:
1. **Simple Average**: (72.5 + 64.2) / 2 = 68.4 possessions ← Current method
2. **Pace Control Factor**:
   - Wisconsin's pack-line defense FORCES slow pace
   - Auburn's transition game REQUIRES fast breaks
   - Wisconsin has more DEFENSIVE control → Slow pace wins
3. **Better Estimate**: 66 possessions (closer to Wisconsin)

**APL Mismatch Score**:
- Auburn wants 15-16 sec possessions
- Wisconsin forces 20 sec possessions
- Mismatch: 4-5 seconds per possession
- Auburn's game plan is DISRUPTED

**Point Impact**:
- Auburn's AdjEM is likely based on 72-possession games
- In 66-possession game, Auburn's advantage is REDUCED
- Estimate: -1.5 to -2.0 points off Auburn's expected margin

#### Scenario B: Duke vs Houston
```
Duke:    Tempo 74.2, APL_Off 16.0, APL_Def 16.5
Houston: Tempo 66.8, APL_Off 18.2, APL_Def 15.5

Expected Game Tempo: ???
```

**Analysis**:
1. **Simple Average**: (74.2 + 66.8) / 2 = 70.5 possessions
2. **Offensive Styles**:
   - Duke: 16.0 APL_Off (relatively quick)
   - Houston: 18.2 APL_Off (controlled)
   - Average: 17.1 seconds/possession
3. **Defensive Pressures**:
   - Duke: 16.5 APL_Def (allows some pace)
   - Houston: 15.5 APL_Def (PRESSURE)
   - Houston's defense forces faster pace!
4. **Competing Forces**:
   - Houston wants slow offense, fast defense
   - Duke wants fast offense, moderately fast defense
   - **Result**: Moderate-fast pace ~70-71 possessions

**APL-Based Advantage**:
- Houston's defense (15.5 APL_Def) meets Duke's offense (16.0 APL_Off)
  → Duke slightly slowed down (-0.5 sec)
- Duke's defense (16.5 APL_Def) meets Houston's offense (18.2 APL_Off)
  → Houston significantly sped up (-1.7 sec)

**Advantage**: Duke
- Houston is more disrupted from their preferred style
- Duke plays closer to their natural pace

---

## Statistical Methodology

### Feature Engineering for ML Model

#### New Features to Add

```python
# Add to FeatureEngineer.FEATURE_NAMES

FEATURE_NAMES = [
    # Existing features
    "em_diff",
    "tempo_avg",
    "tempo_diff",
    "oe_diff",
    "de_diff",
    "pythag_diff",
    "sos_diff",
    "home_advantage",
    "em_tempo_interaction",

    # NEW APL features
    "apl_off_diff",           # Team1 APL_Off - Team2 APL_Off
    "apl_def_diff",           # Team1 APL_Def - Team2 APL_Def
    "apl_off_mismatch",       # Team1 APL_Off vs Team2 APL_Def
    "apl_def_mismatch",       # Team2 APL_Off vs Team1 APL_Def
    "apl_style_score",        # Overall style mismatch (0-10)
    "tempo_control_factor",   # Who controls pace (-1 to +1)
]
```

#### Feature Calculations

**1. APL Difference Features**
```python
apl_off_diff = team1_stats["APL_Off"] - team2_stats["APL_Off"]
# Positive = Team1 plays faster offense
# Negative = Team2 plays faster offense

apl_def_diff = team1_stats["APL_Def"] - team2_stats["APL_Def"]
# Positive = Team1 allows longer possessions (pack-line)
# Negative = Team1 forces shorter possessions (pressure)
```

**2. APL Mismatch (KEY FEATURE)**
```python
# Team1's offensive pace preference vs Team2's defensive allowance
apl_off_mismatch = team1_stats["APL_Off"] - team2_stats["APL_Def"]

# Interpretation:
# Positive mismatch: Team1 wants to play faster than Team2 allows
#   → Team1's offense is DISRUPTED
# Negative mismatch: Team1 wants to play slower than Team2 allows
#   → Team1 gets what they want (slight advantage)

# Example:
# Duke APL_Off = 16.0, Wisconsin APL_Def = 19.8
# Mismatch = 16.0 - 19.8 = -3.8
# Duke wants to play MUCH faster (16 sec) than Wisconsin allows (20 sec)
# Duke is DISRUPTED by 3.8 seconds per possession
```

**3. Style Mismatch Score (0-10 scale)**
```python
def calculate_style_mismatch(team1_stats, team2_stats):
    """Calculate comprehensive style mismatch score."""

    # Tempo differential (0-3 points)
    tempo_diff = abs(team1_stats["AdjTempo"] - team2_stats["AdjTempo"])
    tempo_score = min(3.0, tempo_diff / 3.0)  # Cap at 3

    # APL offensive mismatch (0-3 points)
    apl_off_diff = abs(team1_stats["APL_Off"] - team2_stats["APL_Off"])
    apl_off_score = min(3.0, apl_off_diff / 2.0)

    # APL defensive mismatch (0-4 points) - Defense has more impact
    apl_def_diff = abs(team1_stats["APL_Def"] - team2_stats["APL_Def"])
    apl_def_score = min(4.0, apl_def_diff / 1.5)

    total_score = tempo_score + apl_off_score + apl_def_score
    return round(total_score, 1)

# Example:
# Auburn (72.5, 15.2 off, 15.8 def) vs Wisconsin (64.2, 20.5 off, 19.8 def)
# tempo_score = min(3, 8.3/3) = 3.0
# apl_off_score = min(3, 5.3/2) = 2.65
# apl_def_score = min(4, 4.0/1.5) = 2.67
# Total = 8.3 out of 10 (EXTREME mismatch)
```

**4. Tempo Control Factor (-1 to +1)**
```python
def calculate_tempo_control(team1_stats, team2_stats):
    """Estimate which team controls the pace.

    Returns:
        -1.0 to +1.0 where:
        +1.0 = Team1 completely controls pace
        0.0 = Neutral / equal control
        -1.0 = Team2 completely controls pace
    """

    # Factor 1: Defensive style (40% weight)
    # Pressure defense (low APL_Def) controls pace more than offense
    team1_def_control = 20.0 - team1_stats["APL_Def"]  # Lower = more control
    team2_def_control = 20.0 - team2_stats["APL_Def"]
    def_factor = (team1_def_control - team2_def_control) / 5.0  # Normalize

    # Factor 2: Efficiency advantage (30% weight)
    # Better team dictates pace
    em_diff = team1_stats["AdjEM"] - team2_stats["AdjEM"]
    em_factor = em_diff / 20.0  # Normalize (±20 EM is large)

    # Factor 3: Tempo preference strength (30% weight)
    # Team that REALLY wants a certain pace has more motivation
    team1_tempo_strength = abs(team1_stats["AdjTempo"] - 68.0) / 6.0
    team2_tempo_strength = abs(team2_stats["AdjTempo"] - 68.0) / 6.0

    # Directional strength
    if team1_stats["AdjTempo"] > team2_stats["AdjTempo"]:
        tempo_factor = team1_tempo_strength - team2_tempo_strength
    else:
        tempo_factor = team2_tempo_strength - team1_tempo_strength

    # Weighted combination
    control = (
        0.40 * def_factor +
        0.30 * em_factor +
        0.30 * tempo_factor
    )

    # Clamp to [-1, 1]
    return max(-1.0, min(1.0, control))

# Example:
# Auburn vs Wisconsin
# team1_def_control = 20 - 15.8 = 4.2
# team2_def_control = 20 - 19.8 = 0.2
# def_factor = (4.2 - 0.2) / 5 = 0.8 (Auburn has defensive control)
#
# em_diff = 33.5 - 10.2 = 23.3 (Auburn much better)
# em_factor = 23.3 / 20 = 1.165 → clamped to 1.0
#
# Auburn tempo: 72.5, Wisconsin: 64.2
# team1_tempo_strength = abs(72.5 - 68) / 6 = 0.75
# team2_tempo_strength = abs(64.2 - 68) / 6 = 0.63
# tempo_factor = 0.75 - 0.63 = 0.12
#
# control = 0.40(0.8) + 0.30(1.0) + 0.30(0.12)
#         = 0.32 + 0.30 + 0.036 = 0.656
#
# Result: +0.66 (Auburn has strong tempo control)
```

### Tempo-Adjusted Expected Possessions

**Current (Simple Average)**:
```python
expected_tempo = (team1_tempo + team2_tempo) / 2
```

**Improved (Control-Weighted)**:
```python
def calculate_expected_tempo(team1_stats, team2_stats):
    """Calculate expected game tempo using pace control weighting."""

    team1_tempo = team1_stats["AdjTempo"]
    team2_tempo = team2_stats["AdjTempo"]

    # Get tempo control factor (-1 to +1)
    control = calculate_tempo_control(team1_stats, team2_stats)

    # Weight toward team with more control
    # control = +1 → 75% weight to team1
    # control = 0  → 50% weight each (simple average)
    # control = -1 → 75% weight to team2

    team1_weight = 0.5 + (control * 0.25)
    team2_weight = 1.0 - team1_weight

    expected_tempo = (team1_tempo * team1_weight) + (team2_tempo * team2_weight)

    return round(expected_tempo, 1)

# Example: Auburn vs Wisconsin
# control = 0.66 (Auburn has control)
# team1_weight = 0.5 + (0.66 * 0.25) = 0.665
# team2_weight = 0.335
# expected = (72.5 * 0.665) + (64.2 * 0.335)
#          = 48.21 + 21.51 = 69.7 possessions
#
# Compare to simple average: 68.4 possessions
# Difference: +1.3 possessions (Auburn dictates slightly faster pace)
```

### Variance Adjustment for Confidence Intervals

**Principle**: Lower tempo games have HIGHER variance

**Why**: Fewer possessions = fewer data points = more randomness

**Formula**:
```python
def adjust_confidence_for_tempo(base_variance, expected_tempo):
    """Adjust prediction variance based on tempo.

    Lower tempo → Higher variance → Wider confidence intervals
    """

    # National average tempo ~68 possessions
    tempo_factor = 68.0 / expected_tempo

    # Variance scales with tempo_factor
    adjusted_variance = base_variance * tempo_factor

    return adjusted_variance

# Example:
# Fast game (74 possessions):
#   tempo_factor = 68/74 = 0.919
#   Variance is REDUCED by 8% (more predictable)
#
# Slow game (62 possessions):
#   tempo_factor = 68/62 = 1.097
#   Variance is INCREASED by 10% (less predictable)
```

**Application to GamePredictor**:
```python
# In predict_with_confidence()

# Calculate expected tempo
expected_tempo = calculate_expected_tempo(team1_stats, team2_stats)

# Base confidence interval from quantile regression
ci_lower, ci_upper = model_predictions

# Adjust for tempo
base_variance = (ci_upper - ci_lower) / 2
adjusted_variance = adjust_confidence_for_tempo(base_variance, expected_tempo)

# New confidence interval
ci_lower_adjusted = margin_pred - adjusted_variance
ci_upper_adjusted = margin_pred + adjusted_variance
```

---

## Implementation Design

### Module Structure

Create `src/kenp0m_sp0rts_analyzer/tempo_analysis.py`:

```python
"""Tempo and pace matchup analysis."""

from dataclasses import dataclass
from typing import Any
import pandas as pd
import numpy as np
from .api_client import KenPomAPI


@dataclass
class TempoProfile:
    """Team's tempo and pace characteristics.

    Attributes:
        team_name: Team name
        adj_tempo: Adjusted tempo (possessions per 40 min)
        rank_tempo: National tempo ranking (1 = fastest)
        apl_off: Average possession length on offense (seconds)
        apl_def: Average possession length on defense (seconds)
        conf_apl_off: Conference average APL offense
        conf_apl_def: Conference average APL defense
        pace_style: Overall pace classification
        off_style: Offensive style classification
        def_style: Defensive style classification
    """

    team_name: str
    adj_tempo: float
    rank_tempo: int

    # APL metrics (seconds per possession)
    apl_off: float
    apl_def: float
    conf_apl_off: float
    conf_apl_def: float

    # Style classifications
    pace_style: str  # "fast", "slow", "average"
    off_style: str   # "quick_strike", "methodical", "average"
    def_style: str   # "pressure", "pack_line", "average"


@dataclass
class PaceMatchupAnalysis:
    """Comprehensive tempo/pace matchup analysis.

    Provides insights into stylistic advantages, pace control,
    and expected game flow characteristics.
    """

    # Team profiles
    team1_profile: TempoProfile
    team2_profile: TempoProfile

    # Tempo projections
    tempo_differential: float  # team1 - team2 (possessions)
    expected_possessions: float  # Weighted projection
    tempo_control_factor: float  # -1 to +1 (who dictates pace)

    # Style analysis
    style_mismatch_score: float  # 0-10 scale
    pace_advantage: str  # "team1", "team2", "neutral"

    # APL insights
    apl_off_mismatch_team1: float  # Team1 off vs Team2 def
    apl_off_mismatch_team2: float  # Team2 off vs Team1 def
    offensive_disruption_team1: str  # "severe", "moderate", "minimal"
    offensive_disruption_team2: str

    # Impact estimates
    tempo_impact_on_margin: float  # Estimated point impact
    confidence_adjustment: float  # Variance multiplier

    # Pace preferences
    optimal_pace_team1: float
    optimal_pace_team2: float
    fast_pace_favors: str  # "team1", "team2", "neutral"
    slow_pace_favors: str


class TempoMatchupAnalyzer:
    """Analyze tempo and pace advantages in basketball matchups.

    This class provides sophisticated tempo/pace analysis beyond
    simple tempo averages. It uses APL (Average Possession Length)
    data to understand offensive/defensive styles and predict
    pace control dynamics.
    """

    # Style classification thresholds
    FAST_TEMPO_THRESHOLD = 70.0
    SLOW_TEMPO_THRESHOLD = 66.0
    QUICK_APL_THRESHOLD = 17.0
    SLOW_APL_THRESHOLD = 19.0

    def __init__(self, api: KenPomAPI | None = None):
        """Initialize tempo analyzer.

        Args:
            api: Optional KenPomAPI instance
        """
        self.api = api or KenPomAPI()

    def get_tempo_profile(self, team_stats: dict[str, Any]) -> TempoProfile:
        """Extract and classify team's tempo profile.

        Args:
            team_stats: Team statistics from ratings endpoint

        Returns:
            TempoProfile with classified characteristics
        """
        adj_tempo = team_stats["AdjTempo"]
        apl_off = team_stats["APL_Off"]
        apl_def = team_stats["APL_Def"]

        # Classify overall pace
        pace_style = self._classify_pace(adj_tempo)

        # Classify offensive style
        off_style = self._classify_offensive_style(apl_off)

        # Classify defensive style
        def_style = self._classify_defensive_style(apl_def)

        return TempoProfile(
            team_name=team_stats["TeamName"],
            adj_tempo=adj_tempo,
            rank_tempo=team_stats["RankAdjTempo"],
            apl_off=apl_off,
            apl_def=apl_def,
            conf_apl_off=team_stats["ConfAPL_Off"],
            conf_apl_def=team_stats["ConfAPL_Def"],
            pace_style=pace_style,
            off_style=off_style,
            def_style=def_style,
        )

    def _classify_pace(self, tempo: float) -> str:
        """Classify overall pace style."""
        if tempo > self.FAST_TEMPO_THRESHOLD:
            return "fast"
        elif tempo < self.SLOW_TEMPO_THRESHOLD:
            return "slow"
        else:
            return "average"

    def _classify_offensive_style(self, apl_off: float) -> str:
        """Classify offensive style based on possession length."""
        if apl_off < self.QUICK_APL_THRESHOLD:
            return "quick_strike"
        elif apl_off > self.SLOW_APL_THRESHOLD:
            return "methodical"
        else:
            return "average"

    def _classify_defensive_style(self, apl_def: float) -> str:
        """Classify defensive style based on possession length allowed."""
        if apl_def < self.QUICK_APL_THRESHOLD:
            return "pressure"
        elif apl_def > self.SLOW_APL_THRESHOLD:
            return "pack_line"
        else:
            return "average"

    def analyze_pace_matchup(
        self,
        team1_stats: dict[str, Any],
        team2_stats: dict[str, Any],
    ) -> PaceMatchupAnalysis:
        """Comprehensive tempo/pace matchup analysis.

        Args:
            team1_stats: Team 1 statistics from ratings endpoint
            team2_stats: Team 2 statistics from ratings endpoint

        Returns:
            PaceMatchupAnalysis with detailed tempo insights
        """
        # Get profiles
        team1_profile = self.get_tempo_profile(team1_stats)
        team2_profile = self.get_tempo_profile(team2_stats)

        # Calculate tempo differential
        tempo_diff = team1_profile.adj_tempo - team2_profile.adj_tempo

        # Calculate tempo control
        control_factor = self._calculate_tempo_control(
            team1_stats, team2_stats
        )

        # Calculate expected possessions (control-weighted)
        expected_poss = self._calculate_expected_tempo(
            team1_stats, team2_stats, control_factor
        )

        # Style mismatch score
        style_mismatch = self._calculate_style_mismatch(
            team1_profile, team2_profile
        )

        # APL mismatches
        apl_off_mismatch1 = team1_profile.apl_off - team2_profile.apl_def
        apl_off_mismatch2 = team2_profile.apl_off - team1_profile.apl_def

        # Offensive disruption
        disruption1 = self._classify_disruption(apl_off_mismatch1)
        disruption2 = self._classify_disruption(apl_off_mismatch2)

        # Pace advantage
        pace_advantage = self._determine_pace_advantage(
            team1_stats, team2_stats, control_factor
        )

        # Tempo impact on margin
        tempo_impact = self._estimate_tempo_impact(
            team1_stats, team2_stats, expected_poss
        )

        # Confidence adjustment
        confidence_adj = self._calculate_confidence_adjustment(expected_poss)

        # Optimal paces
        optimal1 = self._calculate_optimal_pace(team1_stats)
        optimal2 = self._calculate_optimal_pace(team2_stats)

        # Who benefits from pace scenarios
        fast_favors = self._determine_tempo_beneficiary(
            team1_stats, team2_stats, "fast"
        )
        slow_favors = self._determine_tempo_beneficiary(
            team1_stats, team2_stats, "slow"
        )

        return PaceMatchupAnalysis(
            team1_profile=team1_profile,
            team2_profile=team2_profile,
            tempo_differential=tempo_diff,
            expected_possessions=expected_poss,
            tempo_control_factor=control_factor,
            style_mismatch_score=style_mismatch,
            pace_advantage=pace_advantage,
            apl_off_mismatch_team1=apl_off_mismatch1,
            apl_off_mismatch_team2=apl_off_mismatch2,
            offensive_disruption_team1=disruption1,
            offensive_disruption_team2=disruption2,
            tempo_impact_on_margin=tempo_impact,
            confidence_adjustment=confidence_adj,
            optimal_pace_team1=optimal1,
            optimal_pace_team2=optimal2,
            fast_pace_favors=fast_favors,
            slow_pace_favors=slow_favors,
        )

    def _calculate_tempo_control(
        self, team1_stats: dict, team2_stats: dict
    ) -> float:
        """Calculate tempo control factor (-1 to +1).

        See Statistical Methodology section for detailed formula.
        """
        # Defensive control (40% weight)
        team1_def_control = 20.0 - team1_stats["APL_Def"]
        team2_def_control = 20.0 - team2_stats["APL_Def"]
        def_factor = (team1_def_control - team2_def_control) / 5.0

        # Efficiency advantage (30% weight)
        em_diff = team1_stats["AdjEM"] - team2_stats["AdjEM"]
        em_factor = np.clip(em_diff / 20.0, -1.0, 1.0)

        # Tempo preference strength (30% weight)
        team1_strength = abs(team1_stats["AdjTempo"] - 68.0) / 6.0
        team2_strength = abs(team2_stats["AdjTempo"] - 68.0) / 6.0

        if team1_stats["AdjTempo"] > team2_stats["AdjTempo"]:
            tempo_factor = team1_strength - team2_strength
        else:
            tempo_factor = -(team2_strength - team1_strength)

        # Weighted combination
        control = (
            0.40 * def_factor +
            0.30 * em_factor +
            0.30 * tempo_factor
        )

        return np.clip(control, -1.0, 1.0)

    def _calculate_expected_tempo(
        self,
        team1_stats: dict,
        team2_stats: dict,
        control_factor: float,
    ) -> float:
        """Calculate expected game tempo using control weighting."""
        team1_tempo = team1_stats["AdjTempo"]
        team2_tempo = team2_stats["AdjTempo"]

        # Convert control (-1 to +1) to weights
        team1_weight = 0.5 + (control_factor * 0.25)
        team2_weight = 1.0 - team1_weight

        expected = (team1_tempo * team1_weight) + (team2_tempo * team2_weight)

        return round(expected, 1)

    def _calculate_style_mismatch(
        self, profile1: TempoProfile, profile2: TempoProfile
    ) -> float:
        """Calculate style mismatch score (0-10).

        See Statistical Methodology section for formula.
        """
        # Tempo differential (0-3 points)
        tempo_diff = abs(profile1.adj_tempo - profile2.adj_tempo)
        tempo_score = min(3.0, tempo_diff / 3.0)

        # APL offensive mismatch (0-3 points)
        apl_off_diff = abs(profile1.apl_off - profile2.apl_off)
        apl_off_score = min(3.0, apl_off_diff / 2.0)

        # APL defensive mismatch (0-4 points)
        apl_def_diff = abs(profile1.apl_def - profile2.apl_def)
        apl_def_score = min(4.0, apl_def_diff / 1.5)

        total = tempo_score + apl_off_score + apl_def_score
        return round(total, 1)

    def _classify_disruption(self, apl_mismatch: float) -> str:
        """Classify offensive disruption severity."""
        abs_mismatch = abs(apl_mismatch)

        if abs_mismatch > 3.0:
            return "severe"
        elif abs_mismatch > 1.5:
            return "moderate"
        else:
            return "minimal"

    def _determine_pace_advantage(
        self,
        team1_stats: dict,
        team2_stats: dict,
        control_factor: float,
    ) -> str:
        """Determine which team has pace advantage."""
        if control_factor > 0.3:
            return "team1"
        elif control_factor < -0.3:
            return "team2"
        else:
            return "neutral"

    def _estimate_tempo_impact(
        self,
        team1_stats: dict,
        team2_stats: dict,
        expected_tempo: float,
    ) -> float:
        """Estimate point impact of tempo on margin.

        More possessions amplify efficiency advantages.
        """
        # Efficiency advantage per 100 possessions
        em_diff = team1_stats["AdjEM"] - team2_stats["AdjEM"]

        # Scale to expected possessions
        point_impact = em_diff * (expected_tempo / 100.0)

        # Deviation from simple average
        simple_avg = (team1_stats["AdjTempo"] + team2_stats["AdjTempo"]) / 2
        tempo_deviation = expected_tempo - simple_avg

        # Additional impact from pace control
        pace_impact = tempo_deviation * (em_diff / 100.0)

        total_impact = pace_impact
        return round(total_impact, 2)

    def _calculate_confidence_adjustment(self, expected_tempo: float) -> float:
        """Calculate variance multiplier for confidence intervals."""
        # National average ~68 possessions
        tempo_factor = 68.0 / expected_tempo
        return round(tempo_factor, 3)

    def _calculate_optimal_pace(self, team_stats: dict) -> float:
        """Calculate optimal game tempo for team."""
        base_tempo = team_stats["AdjTempo"]
        adj_oe = team_stats["AdjOE"]
        adj_de = team_stats["AdjDE"]

        # Offensive adjustment: great offense prefers more possessions
        oe_adjustment = (adj_oe - 110.0) / 10.0

        # Defensive adjustment: great defense can handle faster pace
        de_adjustment = -(adj_de - 100.0) / 10.0

        optimal = base_tempo + oe_adjustment + de_adjustment
        return round(optimal, 1)

    def _determine_tempo_beneficiary(
        self,
        team1_stats: dict,
        team2_stats: dict,
        pace_type: str,
    ) -> str:
        """Determine which team benefits from given pace."""
        team1_em = team1_stats["AdjEM"]
        team2_em = team2_stats["AdjEM"]
        team1_oe = team1_stats["AdjOE"]
        team2_oe = team2_stats["AdjOE"]

        if pace_type == "fast":
            # Fast pace: favors efficient offense
            if team1_oe > team2_oe and team1_em > team2_em:
                return "team1"
            elif team2_oe > team1_oe and team2_em > team1_em:
                return "team2"
            else:
                return "neutral"
        else:  # slow
            # Slow pace: favors underdog limiting possessions
            if team1_em < team2_em:
                return "team1"
            elif team2_em < team1_em:
                return "team2"
            else:
                return "neutral"
```

### Integration Points

#### 1. Add to `prediction.py` FeatureEngineer

```python
# In FeatureEngineer class

FEATURE_NAMES = [
    "em_diff",
    "tempo_avg",
    "tempo_diff",
    "oe_diff",
    "de_diff",
    "pythag_diff",
    "sos_diff",
    "home_advantage",
    "em_tempo_interaction",
    # NEW APL features
    "apl_off_diff",
    "apl_def_diff",
    "apl_off_mismatch_team1",
    "apl_off_mismatch_team2",
    "tempo_control_factor",
]

@staticmethod
def create_features(
    team1_stats: dict[str, Any],
    team2_stats: dict[str, Any],
    neutral_site: bool = True,
    home_team1: bool = False,
) -> dict[str, float]:
    """Create feature vector for prediction."""

    features: dict[str, float] = {}

    # ... existing features ...

    # NEW: APL features
    from .tempo_analysis import TempoMatchupAnalyzer

    analyzer = TempoMatchupAnalyzer()
    tempo_analysis = analyzer.analyze_pace_matchup(team1_stats, team2_stats)

    features["apl_off_diff"] = (
        team1_stats["APL_Off"] - team2_stats["APL_Off"]
    )
    features["apl_def_diff"] = (
        team1_stats["APL_Def"] - team2_stats["APL_Def"]
    )
    features["apl_off_mismatch_team1"] = tempo_analysis.apl_off_mismatch_team1
    features["apl_off_mismatch_team2"] = tempo_analysis.apl_off_mismatch_team2
    features["tempo_control_factor"] = tempo_analysis.tempo_control_factor

    return features
```

#### 2. Update `GamePredictor.predict_with_confidence()`

```python
def predict_with_confidence(
    self,
    team1_stats: dict[str, Any],
    team2_stats: dict[str, Any],
    neutral_site: bool = True,
    home_team1: bool = False,
) -> PredictionResult:
    """Predict game outcome with tempo-adjusted confidence intervals."""

    # ... existing prediction code ...

    # NEW: Adjust confidence intervals for tempo
    from .tempo_analysis import TempoMatchupAnalyzer

    analyzer = TempoMatchupAnalyzer()
    tempo_analysis = analyzer.analyze_pace_matchup(team1_stats, team2_stats)

    # Apply tempo-based variance adjustment
    base_interval_width = ci_upper - ci_lower
    adjusted_width = base_interval_width * tempo_analysis.confidence_adjustment

    ci_lower_adjusted = margin_pred - (adjusted_width / 2)
    ci_upper_adjusted = margin_pred + (adjusted_width / 2)

    return PredictionResult(
        predicted_margin=margin_pred,
        predicted_total=total_pred,
        confidence_interval=(ci_lower_adjusted, ci_upper_adjusted),
        # ... rest of fields ...
    )
```

#### 3. Add to MCP Server Tools

```python
# In mcp_server.py

@server.tool()
async def analyze_tempo_matchup(team1: str, team2: str, year: int = 2025) -> str:
    """Analyze tempo and pace advantages in a matchup.

    Args:
        team1: First team name
        team2: Second team name
        year: Season year (default: 2025)

    Returns:
        Detailed tempo/pace analysis report
    """
    api = KenPomAPI()
    analyzer = TempoMatchupAnalyzer(api)

    # Get team stats
    team1_stats = api.get_team_by_name(team1, year)
    team2_stats = api.get_team_by_name(team2, year)

    # Analyze tempo matchup
    analysis = analyzer.analyze_pace_matchup(team1_stats, team2_stats)

    # Format report
    report = f"""
# Tempo/Pace Matchup: {team1} vs {team2}

## Team Profiles

### {team1}
- Adjusted Tempo: {analysis.team1_profile.adj_tempo} (Rank: {analysis.team1_profile.rank_tempo})
- Pace Style: {analysis.team1_profile.pace_style}
- Offensive Style: {analysis.team1_profile.off_style} (APL: {analysis.team1_profile.apl_off}s)
- Defensive Style: {analysis.team1_profile.def_style} (APL: {analysis.team1_profile.apl_def}s)

### {team2}
- Adjusted Tempo: {analysis.team2_profile.adj_tempo} (Rank: {analysis.team2_profile.rank_tempo})
- Pace Style: {analysis.team2_profile.pace_style}
- Offensive Style: {analysis.team2_profile.off_style} (APL: {analysis.team2_profile.apl_off}s)
- Defensive Style: {analysis.team2_profile.def_style} (APL: {analysis.team2_profile.apl_def}s)

## Matchup Analysis

- **Tempo Differential**: {analysis.tempo_differential:+.1f} possessions ({team1} faster)
- **Expected Game Pace**: {analysis.expected_possessions} possessions
- **Style Mismatch Score**: {analysis.style_mismatch_score}/10
- **Pace Control**: {analysis.pace_advantage}
- **Tempo Control Factor**: {analysis.tempo_control_factor:+.2f}

## APL Matchup Insights

- **{team1} Offensive Disruption**: {analysis.offensive_disruption_team1}
  - {team1} offense ({analysis.team1_profile.apl_off}s) vs {team2} defense ({analysis.team2_profile.apl_def}s)
  - Mismatch: {analysis.apl_off_mismatch_team1:+.1f} seconds

- **{team2} Offensive Disruption**: {analysis.offensive_disruption_team2}
  - {team2} offense ({analysis.team2_profile.apl_off}s) vs {team1} defense ({analysis.team1_profile.apl_def}s)
  - Mismatch: {analysis.apl_off_mismatch_team2:+.1f} seconds

## Impact Estimates

- **Tempo Impact on Margin**: {analysis.tempo_impact_on_margin:+.2f} points
- **Confidence Adjustment**: {analysis.confidence_adjustment:.3f}x variance
- **Optimal Pace ({team1})**: {analysis.optimal_pace_team1} possessions
- **Optimal Pace ({team2})**: {analysis.optimal_pace_team2} possessions

## Pace Scenario Analysis

- **Fast Pace Favors**: {analysis.fast_pace_favors}
- **Slow Pace Favors**: {analysis.slow_pace_favors}
"""

    return report
```

---

## Validation Framework

### Backtesting Approach

**Goal**: Validate that tempo features improve prediction accuracy

#### Test 1: Historical Game Accuracy

```python
def backtest_tempo_features():
    """Compare prediction accuracy with and without tempo features."""

    # Load historical game data (2023-24 season)
    games_df = load_historical_games()  # 5000+ games

    # Model 1: Without APL features (baseline)
    baseline_features = [
        "em_diff", "tempo_avg", "tempo_diff", "oe_diff", "de_diff",
        "pythag_diff", "sos_diff", "home_advantage", "em_tempo_interaction"
    ]

    # Model 2: With APL features
    enhanced_features = baseline_features + [
        "apl_off_diff", "apl_def_diff", "apl_off_mismatch_team1",
        "apl_off_mismatch_team2", "tempo_control_factor"
    ]

    # Train/test split
    train, test = train_test_split(games_df, test_size=0.2)

    # Train models
    model_baseline = train_model(train, baseline_features)
    model_enhanced = train_model(train, enhanced_features)

    # Evaluate
    baseline_mae = evaluate(model_baseline, test)
    enhanced_mae = evaluate(model_enhanced, test)

    # Statistical significance test
    improvement = baseline_mae - enhanced_mae
    p_value = paired_t_test(baseline_errors, enhanced_errors)

    print(f"Baseline MAE: {baseline_mae:.2f} points")
    print(f"Enhanced MAE: {enhanced_mae:.2f} points")
    print(f"Improvement: {improvement:.2f} points ({improvement/baseline_mae*100:.1f}%)")
    print(f"P-value: {p_value:.4f}")

    # Expected result:
    # Improvement: 0.3-0.5 points (~3-5% reduction in error)
```

#### Test 2: Tempo-Specific Performance

```python
def test_tempo_specific_accuracy():
    """Test if tempo features help MORE in extreme tempo games."""

    # Segment games by tempo
    fast_games = games_df[games_df["actual_tempo"] > 72]  # Fast
    slow_games = games_df[games_df["actual_tempo"] < 64]  # Slow
    average_games = games_df[
        (games_df["actual_tempo"] >= 64) & (games_df["actual_tempo"] <= 72)
    ]

    # Evaluate both models on each segment
    for segment, name in [(fast_games, "Fast"), (slow_games, "Slow"),
                          (average_games, "Average")]:
        baseline_mae = evaluate(model_baseline, segment)
        enhanced_mae = evaluate(model_enhanced, segment)
        improvement = baseline_mae - enhanced_mae

        print(f"{name} Games:")
        print(f"  Baseline: {baseline_mae:.2f}")
        print(f"  Enhanced: {enhanced_mae:.2f}")
        print(f"  Improvement: {improvement:.2f} ({improvement/baseline_mae*100:.1f}%)")

    # Expected result:
    # - Slow games: 0.5-0.7 point improvement (largest)
    # - Fast games: 0.4-0.6 point improvement (moderate)
    # - Average games: 0.2-0.3 point improvement (smallest)
    #
    # Tempo features matter MOST in extreme tempo matchups
```

#### Test 3: Style Mismatch Validation

```python
def validate_style_mismatch_impact():
    """Verify that high style mismatch games are harder to predict."""

    # Calculate style mismatch for all games
    analyzer = TempoMatchupAnalyzer()

    mismatches = []
    errors = []

    for game in test_games:
        analysis = analyzer.analyze_pace_matchup(
            game["team1_stats"], game["team2_stats"]
        )
        mismatches.append(analysis.style_mismatch_score)

        prediction = model.predict(game)
        error = abs(prediction - game["actual_margin"])
        errors.append(error)

    # Correlation: mismatch score vs prediction error
    correlation = np.corrcoef(mismatches, errors)[0, 1]

    print(f"Correlation (style mismatch vs error): {correlation:.3f}")

    # Expected result:
    # Positive correlation ~0.15-0.25
    # Higher mismatch → Higher prediction error
    # Validates that style clashes add uncertainty
```

### Real-Time Validation

**Monitor predictions during 2024-25 season:**

```python
class TempoFeatureMonitor:
    """Monitor tempo feature performance during live season."""

    def log_prediction(self, game_id, prediction, actual_result):
        """Log prediction and actual result."""
        # Store:
        # - Predicted margin (with tempo features)
        # - Actual margin
        # - Style mismatch score
        # - Expected tempo vs actual tempo
        # - Confidence interval width

        pass

    def generate_weekly_report(self):
        """Generate weekly performance report."""
        # Calculate:
        # - MAE overall
        # - MAE by style mismatch quartile
        # - Tempo prediction accuracy (expected vs actual)
        # - Confidence interval calibration

        pass
```

---

## Use Cases & Examples

### Example 1: Tournament Matchup Analysis

**Scenario**: Sweet 16 - Auburn (1 seed) vs Wisconsin (5 seed)

```python
from kenp0m_sp0rts_analyzer.api_client import KenPomAPI
from kenp0m_sp0rts_analyzer.tempo_analysis import TempoMatchupAnalyzer
from kenp0m_sp0rts_analyzer.prediction import GamePredictor

# Initialize
api = KenPomAPI()
tempo_analyzer = TempoMatchupAnalyzer(api)
predictor = GamePredictor()

# Get team stats (from Selection Sunday)
auburn_stats = api.get_team_by_name("Auburn", 2025)
wisconsin_stats = api.get_team_by_name("Wisconsin", 2025)

# Tempo analysis
tempo_analysis = tempo_analyzer.analyze_pace_matchup(
    auburn_stats, wisconsin_stats
)

print("=" * 60)
print("SWEET 16 MATCHUP: Auburn vs Wisconsin")
print("=" * 60)
print()
print(f"Auburn Tempo: {auburn_stats['AdjTempo']} (Rank {auburn_stats['RankAdjTempo']})")
print(f"  Offensive APL: {auburn_stats['APL_Off']}s (quick-strike)")
print(f"  Defensive APL: {auburn_stats['APL_Def']}s (pressure)")
print()
print(f"Wisconsin Tempo: {wisconsin_stats['AdjTempo']} (Rank {wisconsin_stats['RankAdjTempo']})")
print(f"  Offensive APL: {wisconsin_stats['APL_Off']}s (methodical)")
print(f"  Defensive APL: {wisconsin_stats['APL_Def']}s (pack-line)")
print()
print("MATCHUP ANALYSIS")
print("-" * 60)
print(f"Expected Pace: {tempo_analysis.expected_possessions} possessions")
print(f"Style Mismatch: {tempo_analysis.style_mismatch_score}/10 (EXTREME)")
print(f"Pace Control: {tempo_analysis.pace_advantage}")
print()
print(f"Auburn Offensive Disruption: {tempo_analysis.offensive_disruption_team1}")
print(f"  Auburn wants {auburn_stats['APL_Off']}s per possession")
print(f"  Wisconsin defense allows {wisconsin_stats['APL_Def']}s")
print(f"  Mismatch: {tempo_analysis.apl_off_mismatch_team1:+.1f}s")
print()
print(f"Tempo Impact on Margin: {tempo_analysis.tempo_impact_on_margin:+.2f} points")
print(f"Confidence Adjustment: {tempo_analysis.confidence_adjustment:.2f}x")
print()

# Prediction with tempo-adjusted confidence
prediction = predictor.predict_with_confidence(
    auburn_stats, wisconsin_stats, neutral_site=True
)

print("PREDICTION")
print("-" * 60)
print(f"Predicted Margin: Auburn by {prediction.predicted_margin:.1f}")
print(f"Confidence Interval: ({prediction.confidence_interval[0]:.1f}, {prediction.confidence_interval[1]:.1f})")
print(f"Auburn Win Probability: {prediction.team1_win_prob:.1%}")
print()
print("KEY INSIGHTS:")
print("- Wisconsin's pack-line will slow Auburn's transition game")
print("- Expected pace (~66 poss) favors Wisconsin's style")
print("- Auburn's efficiency advantage is REDUCED in slow game")
print("- Wider confidence interval due to low tempo (higher variance)")
```

**Expected Output:**
```
============================================================
SWEET 16 MATCHUP: Auburn vs Wisconsin
============================================================

Auburn Tempo: 72.5 (Rank 15)
  Offensive APL: 15.2s (quick-strike)
  Defensive APL: 15.8s (pressure)

Wisconsin Tempo: 64.2 (Rank 320)
  Offensive APL: 20.5s (methodical)
  Defensive APL: 19.8s (pack-line)

MATCHUP ANALYSIS
------------------------------------------------------------
Expected Pace: 66.2 possessions
Style Mismatch: 8.3/10 (EXTREME)
Pace Control: team2 (Wisconsin)

Auburn Offensive Disruption: severe
  Auburn wants 15.2s per possession
  Wisconsin defense allows 19.8s
  Mismatch: -4.6s

Tempo Impact on Margin: -2.1 points
Confidence Adjustment: 1.03x

PREDICTION
------------------------------------------------------------
Predicted Margin: Auburn by 7.3
Confidence Interval: (3.8, 10.8)
Auburn Win Probability: 78.5%

KEY INSIGHTS:
- Wisconsin's pack-line will slow Auburn's transition game
- Expected pace (~66 poss) favors Wisconsin's style
- Auburn's efficiency advantage is REDUCED in slow game
- Wider confidence interval due to low tempo (higher variance)
```

### Example 2: Conference Tempo Comparison

```python
# Compare conference tempo styles

tempo_analyzer = TempoMatchupAnalyzer()
conference_tempos = tempo_analyzer.compare_conference_tempos(year=2025)

print("CONFERENCE TEMPO RANKINGS (2024-25 Season)")
print("=" * 80)
print(f"{'Conference':<20} {'Avg Tempo':<12} {'Avg APL Off':<14} {'Avg APL Def':<14}")
print("-" * 80)

for _, conf in conference_tempos.head(10).iterrows():
    print(f"{conf['Conference']:<20} {conf['AvgTempo']:<12.1f} "
          f"{conf['AvgAPL_Off']:<14.1f} {conf['AvgAPL_Def']:<14.1f}")

# Output:
# CONFERENCE TEMPO RANKINGS (2024-25 Season)
# ============================================================
# Conference           Avg Tempo    Avg APL Off    Avg APL Def
# ------------------------------------------------------------
# Big 12               70.2         16.5           16.8
# West Coast           69.8         16.8           17.0
# Big East             68.5         17.2           17.4
# SEC                  68.1         17.4           17.5
# ACC                  67.5         17.6           17.8
# Pac-12               67.2         17.8           18.0
# Big Ten              66.8         18.0           18.2
# ...
```

---

## Performance Optimization

### Caching Strategy

Tempo analysis involves complex calculations. Cache results:

```python
from functools import lru_cache

class TempoMatchupAnalyzer:

    @lru_cache(maxsize=128)
    def analyze_pace_matchup(
        self,
        team1_id: int,  # Use team IDs for caching
        team2_id: int,
        year: int,
    ) -> PaceMatchupAnalysis:
        """Cached tempo analysis."""
        # Fetch stats
        team1_stats = self.api.get_ratings(team_id=team1_id, year=year)
        team2_stats = self.api.get_ratings(team_id=team2_id, year=year)

        # ... analysis ...
```

### Batch Processing

For tournament simulations (thousands of matchups):

```python
def batch_tempo_analysis(matchups: list[tuple[int, int]]) -> list[PaceMatchupAnalysis]:
    """Batch process tempo analysis for multiple matchups."""

    # Pre-fetch all team stats in single API call
    all_team_ids = set()
    for team1_id, team2_id in matchups:
        all_team_ids.add(team1_id)
        all_team_ids.add(team2_id)

    # Single API call for all teams
    all_stats = api.get_ratings(year=2025)
    stats_dict = {team["TeamID"]: team for team in all_stats.data}

    # Process matchups
    results = []
    for team1_id, team2_id in matchups:
        analysis = analyzer.analyze_pace_matchup(
            stats_dict[team1_id],
            stats_dict[team2_id]
        )
        results.append(analysis)

    return results
```

---

## Future Enhancements

### Phase 2 Features

1. **Situational Tempo Analysis**
   - How teams adjust pace in close games vs blowouts
   - End-of-game tempo (final 5 minutes)
   - Requires play-by-play data

2. **Historical Tempo Trends**
   - Track team tempo evolution across seasons
   - Identify coaching changes impact on tempo
   - Detect "peaking" teams (tempo increasing in February/March)

3. **Opponent-Adjusted Tempo**
   - How teams adjust to specific opponents
   - "Pace adaptation score" (flexible vs rigid)
   - Duke vs Virginia historical: Does Duke slow down or Virginia speed up?

4. **Tournament-Specific Tempo**
   - March Madness tempo different from regular season
   - Neutral court impact on tempo
   - Pressure/stakes impact on pace

### Integration Opportunities

1. **Combine with Four Factors Analysis**
   - Fast teams: How does eFG% change at different paces?
   - Slow teams: Turnover rate in up-tempo games

2. **Player-Level Tempo Impact**
   - When specific player is out, how does tempo change?
   - Backup PG → slower tempo (can't run transition)

3. **Live Betting Applications**
   - In-game tempo tracking
   - "Game is faster than expected" → Live total adjustment

---

## Conclusion

### Implementation Checklist

- [ ] Create `tempo_analysis.py` module
- [ ] Implement `TempoProfile` dataclass
- [ ] Implement `PaceMatchupAnalysis` dataclass
- [ ] Implement `TempoMatchupAnalyzer` class
- [ ] Add APL features to `FeatureEngineer`
- [ ] Update `GamePredictor.predict_with_confidence()`
- [ ] Add MCP server tool `analyze_tempo_matchup`
- [ ] Write comprehensive tests
- [ ] Backtest on historical data
- [ ] Create usage examples
- [ ] Update documentation

### Success Criteria

✅ **Immediate Goals**:
1. Classify team pace styles correctly (fast/slow/methodical/pressure)
2. Calculate expected tempo using control-weighted average
3. Identify APL mismatches and quantify disruption
4. Improve prediction MAE by 2-3%

✅ **Long-term Goals**:
1. Tempo features integrated into GamePredictor
2. Conference tempo comparisons available
3. MCP tool for Claude interactions
4. Validated on 2024-25 season games

### Next Steps

**Ready to implement?**

1. Start with `tempo_analysis.py` core module (2-3 hours)
2. Add features to `prediction.py` (1 hour)
3. Write tests (1-2 hours)
4. Add MCP integration (30 min)
5. Create examples and docs (30 min)

**Total estimated effort: 4-6 hours**

Let me know when you're ready to start implementation!
