# KenPom Matchup Analysis Framework

**Complete Multi-Dimensional Basketball Analytics System**

---

## Overview

This framework provides comprehensive basketball matchup analysis across 3 tiers and 15 dimensions, combining statistical metrics, scheme analysis, and intangible factors.

### Framework Status

| Tier | Modules | Status | Dimensions | Impact |
|------|---------|--------|------------|--------|
| **TIER 1** | Four Factors, Scoring Styles, Defense | ✅ Complete | 10 | High |
| **TIER 2** | Size/Athleticism, Experience/Chemistry | 📋 Planned | 5 | High |
| **TIER 3** | Tempo/Pace, Home Court Advantage | 🔮 Future | 3+ | Medium |

**Total Framework**: 15+ analytical dimensions for complete matchup assessment

---

## TIER 1: Statistical & Scheme Analysis (✅ COMPLETE)

### Module 1: Four Factors Analysis
**Purpose**: Fundamental basketball statistics (Dean Oliver framework)

**Dimensions Analyzed** (4):
1. **eFG%** - Shooting efficiency (40% weight - most important)
2. **TO%** - Ball security vs pressure (25% weight)
3. **OR%** - Offensive rebounding (20% weight)
4. **FT Rate** - Drawing fouls (15% weight)

**Key Outputs**:
- Factor-by-factor matchup breakdown
- Weighted advantage score (-10 to +10)
- Strategic insights (who wins each battle)
- Key matchup battles to watch

**Strategic Value**: Foundation of basketball analysis - identifies fundamental statistical edges

---

### Module 2: Point Distribution Analysis
**Purpose**: Scoring style matchups and defensive vulnerabilities

**Dimensions Analyzed** (3):
1. **3-Point Scoring** - Perimeter offense vs perimeter defense
2. **2-Point Scoring** - Interior offense vs interior defense
3. **Free Throw Scoring** - Foul drawing vs foul prevention

**Scoring Style Classifications**:
- **Perimeter**: >40% of points from 3-pointers
- **Interior**: >55% of points from 2-pointers
- **Balanced**: Neither threshold met

**Key Outputs**:
- Scoring style profiles (perimeter/interior/balanced)
- Style mismatch score (0-10 scale)
- Exploitable areas with specific percentages
- Strategic recommendations for game plans

**Strategic Value**: Identifies how teams score and where defensive weaknesses exist

---

### Module 3: Defensive Analysis
**Purpose**: Defensive scheme identification and matchup evaluation

**Dimensions Analyzed** (3):
1. **Perimeter Defense** - Opponent 3PT% allowed
2. **Interior Defense** - Opponent 2PT% allowed, blocks
3. **Pressure Defense** - Steal rate, forced turnovers

**Defensive Scheme Classifications**:
- **Rim Protection**: High block rate + low opponent 2PT%
- **Pressure**: High steal rate (>9%)
- **Versatile**: Elite at multiple dimensions
- **Balanced**: Solid fundamental defense

**Key Outputs**:
- Defensive scheme identification
- Dimensional advantages (perimeter/interior/pressure)
- Defensive advantage score (0-10 scale)
- Strategic defensive keys

**Strategic Value**: Reveals defensive identity and exploitable weaknesses

---

## TIER 2: Physical & Intangible Factors (📋 PLANNED)

### Module 4: Size & Athleticism Analysis
**Purpose**: Physical matchups and position-specific height advantages

**Dimensions Analyzed** (1 overall + 5 positions = effectively 2):
1. **Overall Size** - Effective height advantage
2. **Position Matchups** - PG, SG, SF, PF, C height comparisons

**Size Classifications**:
- **Elite Size**: AvgHgt > 79"
- **Above Average**: AvgHgt > 77.5"
- **Average**: 76" < AvgHgt < 77.5"
- **Undersized**: AvgHgt < 76"

**Key Outputs**:
- Position-by-position height analysis
- Frontcourt vs backcourt advantages
- Rebounding battle predictions
- Paint scoring predictions
- Size-based strategic recommendations

**Strategic Value**: Physical advantages matter - size impacts rebounding, interior scoring, rim protection

**Key Correlations**:
- Height → Rebounding: ~2% OR per inch of effective height
- Height → Blocks: Taller frontcourts alter more shots
- Height → Paint Scoring: Interior size enables post-ups

---

### Module 5: Experience & Chemistry Analysis
**Purpose**: Intangible factors (experience, bench depth, continuity)

**Dimensions Analyzed** (3):
1. **Experience** - Veteran presence (0-4 scale: Fr=0, Sr=3)
2. **Bench Depth** - Depth beyond starting five
3. **Continuity** - Returning minutes percentage

**Experience Classifications**:
- **Very Experienced**: >2.5 rating (mostly juniors/seniors)
- **Experienced**: >2.0 rating
- **Average**: 1.5-2.0 rating
- **Young**: <1.5 rating (mostly freshmen/sophomores)

**Key Outputs**:
- Experience profile and intangibles score
- Late-game execution predictions
- Tournament readiness assessment
- Adversity handling predictions
- Experience-based strategic keys

**Strategic Value**: Intangibles win March Madness - experience, chemistry, and depth matter in pressure situations

**Key Insights**:
- Experience → Tournament Success: Experienced teams significantly outperform in March
- Continuity → Chemistry: High continuity (>70% minutes) indicates system familiarity
- Bench Depth → Endurance: Elite depth allows aggressive substitution, maintains intensity

---

## TIER 3: Situational & Environmental Factors (🔮 FUTURE)

### Potential Module 6: Tempo & Pace Analysis
**Purpose**: Game pace preferences and strategic tempo control

**Dimensions**:
- Adjusted Tempo preferences
- Pace differential impacts
- Fast break vs half-court efficiency
- Strategic tempo control

**Strategic Value**: Tempo dictates shot opportunities and fatigue

---

### Potential Module 7: Home Court Advantage & Travel
**Purpose**: Environmental and logistical factors

**Dimensions**:
- Home court advantage by venue
- Travel distance impacts
- Neutral site adjustments
- Altitude/climate factors

**Strategic Value**: Venue and logistics affect performance

---

## Complete Framework: 15-Dimensional Analysis

### Dimensional Breakdown

**TIER 1: Statistical & Scheme (10 dimensions)**
1. eFG% Matchup
2. TO% Matchup
3. OR% Matchup
4. FT Rate Matchup
5. 3-Point Scoring Matchup
6. 2-Point Scoring Matchup
7. Free Throw Matchup
8. Perimeter Defense Matchup
9. Interior Defense Matchup
10. Pressure Defense Matchup

**TIER 2: Physical & Intangible (5 dimensions)**
11. Overall Size/Height Matchup
12. Position-Specific Size Matchups (composite)
13. Experience Matchup
14. Bench Depth Matchup
15. Team Continuity/Chemistry Matchup

**TIER 3: Situational (3+ dimensions, future)**
16. Tempo/Pace Matchup
17. Home Court Advantage
18. Travel/Logistics Factors

---

## Analysis Workflow

### Step 1: Run All Analyzers
```python
from kenp0m_sp0rts_analyzer.api_client import KenPomAPI
from kenp0m_sp0rts_analyzer.four_factors_matchup import FourFactorsMatchup
from kenp0m_sp0rts_analyzer.point_distribution_analysis import PointDistributionAnalyzer
from kenp0m_sp0rts_analyzer.defensive_analysis import DefensiveAnalyzer
# TIER 2 (when implemented):
# from kenp0m_sp0rts_analyzer.size_athleticism_analysis import SizeAthleticismAnalyzer
# from kenp0m_sp0rts_analyzer.experience_chemistry_analysis import ExperienceChemistryAnalyzer

api = KenPomAPI()

# TIER 1 Analysis
ff = FourFactorsMatchup(api)
pd = PointDistributionAnalyzer(api)
da = DefensiveAnalyzer(api)

ff_analysis = ff.analyze_matchup("Duke", "North Carolina", 2025)
pd_analysis = pd.analyze_matchup("Duke", "North Carolina", 2025)
def_analysis = da.analyze_matchup("Duke", "North Carolina", 2025)

# TIER 2 Analysis (future):
# sa = SizeAthleticismAnalyzer(api)
# ec = ExperienceChemistryAnalyzer(api)
# size_analysis = sa.analyze_matchup("Duke", "North Carolina", 2025)
# exp_analysis = ec.analyze_matchup("Duke", "North Carolina", 2025)
```

### Step 2: Aggregate "Battles Won"
Track advantages across all dimensions:

```python
team1_advantages = 0
team2_advantages = 0

# TIER 1: Four Factors (4 battles)
for factor in [ff_analysis.efg_matchup, ff_analysis.to_matchup,
               ff_analysis.or_matchup, ff_analysis.ft_matchup]:
    if factor.predicted_winner == "team1":
        team1_advantages += 1
    elif factor.predicted_winner == "team2":
        team2_advantages += 1

# TIER 1: Scoring (3 battles)
if pd_analysis.three_point_advantage > 1.0:
    team1_advantages += 1
elif pd_analysis.three_point_advantage < -1.0:
    team2_advantages += 1
# ... repeat for 2pt and FT

# TIER 1: Defense (3 battles)
if def_analysis.perimeter_defense_advantage == "team1":
    team1_advantages += 1
else:
    team2_advantages += 1
# ... repeat for interior and pressure

# TIER 2: Size (2 battles - overall + frontcourt/backcourt composite)
# TIER 2: Intangibles (3 battles - experience, bench, continuity)

total_battles = 15  # (10 TIER 1 + 5 TIER 2)
print(f"Team 1: {team1_advantages}/{total_battles}")
print(f"Team 2: {team2_advantages}/{total_battles}")
```

### Step 3: Generate Comprehensive Summary
```python
if team1_advantages > team2_advantages + 3:
    prediction = "Team 1 has significant advantages across multiple dimensions"
elif team2_advantages > team1_advantages + 3:
    prediction = "Team 2 has significant advantages across multiple dimensions"
else:
    prediction = "Evenly matched - execution and intangibles will decide"
```

---

## Strategic Decision Framework

### When to Emphasize Each Tier

**TIER 1 (Statistical/Scheme)**: Always foundational
- Use for: Regular season matchups, neutral site games, initial scouting
- Weight: 60% of overall assessment
- Reliability: Highest - based on season-long statistical evidence

**TIER 2 (Physical/Intangible)**: Critical for high-stakes games
- Use for: Tournament games, rivalry games, late-season matchups
- Weight: 30% of overall assessment
- Reliability: High - physical advantages are real, intangibles matter in pressure situations

**TIER 3 (Situational)**: Context-dependent modifiers
- Use for: Home/away splits, travel impacts, environmental factors
- Weight: 10% of overall assessment
- Reliability: Moderate - situational factors add context but vary by team

---

## Interpretation Guide

### Advantage Thresholds

**Four Factors**:
- Massive: >8.0% advantage
- Significant: 5.0-8.0%
- Moderate: 2.5-5.0%
- Minimal: 1.0-2.5%
- Neutral: <1.0%

**Scoring Styles**:
- Strong Mismatch: >5.0% distribution advantage
- Moderate Mismatch: 2.5-5.0%
- Slight Mismatch: 1.0-2.5%
- Neutral: <1.0%

**Defense**:
- Elite Edge: >6.5/10 advantage score
- Significant Edge: 5.5-6.5/10
- Neutral: 4.5-5.5/10
- Disadvantage: <4.5/10

**Size** (when implemented):
- Massive: >3.0" height advantage
- Significant: 2.0-3.0"
- Moderate: 1.0-2.0"
- Minimal: 0.5-1.0"
- Neutral: <0.5"

**Experience** (when implemented):
- Major Edge: >1.0 rating advantage
- Significant Edge: 0.6-1.0
- Moderate Edge: 0.3-0.6
- Neutral: <0.3

---

## Use Cases

### 1. Pre-Game Scouting Report
**Workflow**: Run all analyzers → Generate comprehensive summary → Identify 3-5 key factors

**Output**:
- Overall matchup assessment (team1/team2/neutral)
- Top 3 advantages for each team
- Strategic keys for winning
- Key battles to watch

---

### 2. Tournament Bracket Analysis
**Workflow**: Assess tournament readiness (TIER 2) → Identify bracket position advantages

**Focus Areas**:
- Experience & continuity (March matters)
- Bench depth (6 games in 3 weeks)
- Size advantages (rebounding critical)
- Four Factors consistency

---

### 3. In-Game Adjustments
**Workflow**: Real-time matchup monitoring → Identify underutilized advantages

**Adjustments**:
- Exploit size mismatches at specific positions
- Attack defensive weaknesses (perimeter vs interior)
- Leverage depth advantage with aggressive substitutions
- Adjust tempo based on physical advantages

---

### 4. Historical Analysis & Validation
**Workflow**: Backtest framework on past games → Validate predictions → Refine weights

**Metrics**:
- Prediction accuracy (winner selection)
- Margin prediction (within ±5 points)
- Key factor identification (most impactful dimension)

---

## Integration Examples

### Example 1: Duke vs North Carolina (2025)

**TIER 1 Summary**:
- Four Factors: Duke wins eFG% + OR%, UNC wins TO%
- Scoring: Duke perimeter team (42% from 3pt), UNC balanced
- Defense: UNC pressure defense, Duke rim protection

**TIER 2 Summary** (when implemented):
- Size: Duke +1.5" effective height (frontcourt advantage)
- Experience: UNC +0.8 rating (veteran backcourt)
- Bench: Duke elite depth, UNC good depth

**Overall**: Duke 9/15 battles, UNC 6/15 battles
**Prediction**: Duke has slight edge, but UNC's experience could matter in close game

---

### Example 2: Tournament Sweet 16 Matchup

**Scenario**: #1 seed (statistical powerhouse, young) vs #5 seed (balanced stats, veteran)

**TIER 1**: #1 seed dominates (8/10 dimensions)
**TIER 2**: #5 seed edges experience and chemistry (3/5 dimensions)

**Key Insight**: In tournament pressure, TIER 2 factors become more important
**Adjusted Prediction**: #1 seed favored, but #5 seed has upset potential if game is tight late

---

## Framework Evolution

### Current State (TIER 1 Complete)
- ✅ 10-dimensional statistical and scheme analysis
- ✅ Three specialized analyzers working together
- ✅ Comprehensive demo showing integrated analysis
- ✅ All modules tested and production-ready

### Next Phase (TIER 2 Implementation)
- 📋 Size & Athleticism Analyzer (2-3 hours)
- 📋 Experience & Chemistry Analyzer (2-3 hours)
- 📋 Integration into comprehensive demo (1-2 hours)
- 📋 Testing and validation

### Future Enhancements (TIER 3)
- 🔮 Tempo/Pace Analysis
- 🔮 Home Court Advantage Modeling
- 🔮 Travel & Logistics Factors
- 🔮 Referee Tendencies (advanced)
- 🔮 Injury Impact Modeling (advanced)

---

## Technical Architecture

### Module Structure (Consistent Across Tiers)
```
analyzer_module.py:
├── Profile Dataclasses (team-specific metrics)
├── Matchup Dataclasses (head-to-head analysis)
├── Analyzer Class
│   ├── __init__(api_client)
│   ├── get_profile(team, season)
│   ├── analyze_matchup(team1, team2, season)
│   └── _helper_methods (classification, prediction, etc.)
```

### Design Principles
1. **Consistency**: All modules follow same patterns
2. **Composability**: Modules work independently or together
3. **Type Safety**: Full type hints with Literal types
4. **Documentation**: Comprehensive docstrings
5. **Testability**: Pure functions, mockable API calls

---

## Getting Started

### Quick Start (TIER 1 Only)
```python
from kenp0m_sp0rts_analyzer.api_client import KenPomAPI
from kenp0m_sp0rts_analyzer.four_factors_matchup import FourFactorsMatchup

api = KenPomAPI()
analyzer = FourFactorsMatchup(api)
analysis = analyzer.analyze_matchup("Duke", "North Carolina", 2025)

print(f"Advantage: {analysis.overall_advantage}")
print(f"Key Factor: {analysis.most_important_factor}")
for insight in analysis.strategic_insights:
    print(f"- {insight}")
```

### Comprehensive Analysis (Run All Demos)
```bash
# Four Factors only
python examples/four_factors_matchup_demo.py

# Point Distribution only
python examples/point_distribution_demo.py

# Defensive Analysis only
python examples/defensive_matchup_demo.py

# All TIER 1 combined (15-20 second runtime)
python examples/comprehensive_matchup_demo.py
```

---

## Conclusion

This framework provides a **systematic, multi-dimensional approach** to basketball matchup analysis. By combining statistical edges (TIER 1), physical advantages (TIER 2), and situational factors (TIER 3), it delivers comprehensive scouting reports for any matchup.

**Key Strengths**:
- **Comprehensive**: 15+ dimensions capture full picture
- **Actionable**: Clear strategic recommendations
- **Validated**: Based on proven KenPom metrics
- **Extensible**: Easy to add new analyzers
- **Production-Ready**: TIER 1 complete and tested

**Next Steps**: Implement TIER 2 modules to add physical and intangible factors, creating the most complete basketball matchup analysis system available.
