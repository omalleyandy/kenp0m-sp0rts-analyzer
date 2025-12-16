# KenPom Sports Analyzer - NCAA Basketball Analytics Roadmap

**Advanced Analytics for NCAA Division I Men's Basketball**

## Executive Summary

This roadmap focuses on implementing sophisticated basketball-specific analytics using KenPom data. The repository has a strong foundation with:
- ✅ Official KenPom API integration
- ✅ Machine learning predictions with confidence intervals
- ✅ Comprehensive tempo/pace analysis system
- ✅ Player impact modeling for injury analysis
- ✅ Conference power ratings and analytics

**Focus Areas:**
1. Tournament probability modeling and simulation
2. Advanced tempo/pace matchup analysis
3. Four Factors matchup breakdowns
4. Historical performance backtesting
5. Interactive visualization and dashboards

---

## 🎯 TIER 1: Core Basketball Analytics

### 1. Tournament Probability Engine ✅ HIGH PRIORITY

**Gap**: No tournament simulation capability
**Value**: Critical for March Madness analysis

```python
# src/kenp0m_sp0rts_analyzer/tournament.py

import numpy as np
from scipy import stats

class TournamentSimulator:
    """Monte Carlo simulation for NCAA tournament."""

    def __init__(self, api_client):
        self.api = api_client

    def simulate_tournament(
        self,
        selection_sunday_date: str,
        num_simulations: int = 10000
    ) -> dict:
        """
        Run Monte Carlo simulation of tournament.

        Returns probability of reaching each round for all teams.
        """
        # Get ratings from Selection Sunday
        ratings = self.api.get_archive(archive_date=selection_sunday_date)

        results = {
            'round_64': {},   # First round probabilities
            'round_32': {},   # Second round
            'sweet_16': {},   # Sweet 16
            'elite_8': {},    # Elite 8
            'final_4': {},    # Final Four
            'championship': {},  # Championship game
            'winner': {}      # Tournament winner
        }

        for sim in range(num_simulations):
            bracket = self._simulate_single_bracket(ratings)
            self._accumulate_results(bracket, results)

        # Normalize to probabilities
        for round_name in results:
            for team in results[round_name]:
                results[round_name][team] /= num_simulations

        return results

    def _simulate_game(self, team1: dict, team2: dict) -> str:
        """Simulate single game with variance."""
        # Use efficiency margins + uncertainty
        em_diff = team1['AdjEM'] - team2['AdjEM']

        # Add variance based on tempo (more possessions = more predictable)
        tempo = (team1['AdjTempo'] + team2['AdjTempo']) / 2
        variance = self._calculate_game_variance(tempo)

        # Sample from normal distribution
        predicted_margin = np.random.normal(em_diff, variance)

        return team1['TeamName'] if predicted_margin > 0 else team2['TeamName']

    def _calculate_game_variance(self, tempo: float) -> float:
        """Higher tempo = lower variance (law of large numbers)."""
        # Empirical: ~11 point standard deviation for average tempo
        return 11.0 * (68.0 / tempo) ** 0.5

    def analyze_upset_probability(
        self,
        higher_seed: int,
        lower_seed: int
    ) -> float:
        """Calculate upset probability by seed matchup."""
        # Historical seed vs seed performance
        # Compare to efficiency-based prediction
```

**Features to Implement:**
- Bracket optimization for pool scoring systems
- Identify high-probability upset picks
- Conference tournament simulation for bubble teams
- Historical seed performance validation

---

### 2. Tempo/Pace Deep-Dive Enhancements

**Current State**: ✅ Comprehensive `tempo_analysis.py` module implemented
**Gap**: Visualization, historical tracking, situational analysis

#### Already Implemented (temp_analysis.py):
- ✅ Tempo profile classification (fast/slow/methodical)
- ✅ APL (Average Possession Length) analysis
- ✅ Tempo control calculation (defensive/efficiency/preference weighting)
- ✅ Style mismatch scoring (0-10 scale)
- ✅ Offensive disruption classification
- ✅ Confidence interval adjustments for pace
- ✅ Optimal pace calculations
- ✅ Conference tempo comparisons

#### Proposed Enhancements:

**A. Situational Tempo Analysis**
```python
class SituationalTempoAnalyzer:
    """Analyze how teams adjust pace in different situations."""

    def analyze_tempo_by_game_state(
        self,
        team: str,
        season: int
    ) -> dict:
        """
        Analyze how team's pace changes based on:
        - Leading vs trailing
        - Close game vs blowout
        - Conference vs non-conference
        - Home vs away vs neutral
        - Early season vs tournament time
        """

    def identify_tempo_adjustments(
        self,
        team: str,
        opponent_style: str  # "fast", "slow", "average"
    ) -> dict:
        """
        How does team adjust pace against different opponent styles?

        Returns:
            - Historical pace vs fast teams
            - Historical pace vs slow teams
            - Adaptability score (0-10)
        """
```

**B. Tempo Trend Tracking**
```python
class TempoTrendAnalyzer:
    """Track tempo evolution throughout season."""

    def calculate_tempo_trends(
        self,
        team: str,
        season: int,
        window_size: int = 5
    ) -> pd.DataFrame:
        """
        Rolling window analysis:
        - Is team getting faster/slower?
        - APL trends (becoming more methodical?)
        - Opponent-adjusted trends
        """

    def detect_tempo_inflection_points(
        self,
        team: str,
        season: int
    ) -> list:
        """
        Identify games where team significantly changed pace:
        - Coaching adjustments
        - Personnel changes
        - Style pivots
        """
```

**C. Visualization Dashboard**
```python
# scripts/tempo_dashboard.py

import streamlit as st
import plotly.express as px

def create_tempo_dashboard():
    """Interactive tempo analysis dashboard."""

    # Scatter plot: AdjTempo vs AdjEM (are faster teams better?)
    # APL offensive vs defensive (style quadrants)
    # Conference tempo heatmap
    # Historical tempo trends by team
    # Matchup-specific tempo predictions
```

---

### 3. Four Factors Matchup Breakdown

**Gap**: Using Four Factors data but not analyzing matchup interactions
**Value**: Identify specific strategic advantages

```python
# src/kenp0m_sp0rts_analyzer/four_factors_matchup.py

class FourFactorsMatchup:
    """Analyze Four Factors matchup advantages."""

    def analyze_matchup(
        self,
        team1: str,
        team2: str,
        season: int
    ) -> dict:
        """
        Breakdown by factor:

        1. eFG% Battle:
           - Team1 shooting vs Team2 defense
           - Team2 shooting vs Team1 defense

        2. Turnover Battle:
           - Team1 ball security vs Team2 pressure
           - Team2 ball security vs Team1 pressure

        3. Rebounding Battle:
           - Team1 offensive boards vs Team2 defensive boards
           - Team2 offensive boards vs Team1 defensive boards

        4. Free Throw Battle:
           - Team1 drawing fouls vs Team2 avoiding fouls
           - Team2 drawing fouls vs Team1 avoiding fouls
        """

    def identify_key_matchup_factors(
        self,
        analysis: dict
    ) -> list:
        """
        Rank factors by impact on game outcome.

        Example:
        1. "Turnover battle critical - Team A forces TOs, Team B protects ball"
        2. "Rebounding advantage to Team A (+8% OR differential)"
        3. "eFG% neutral - both elite shooting offenses"
        """
```

---

### 4. Historical Performance Backtesting

**Current State**: ✅ Backtesting framework in `prediction.py`
**Gap**: Historical trend analysis, model evolution tracking

```python
class HistoricalAnalyzer:
    """Analyze historical KenPom prediction performance."""

    def backtest_predictions_by_season(
        self,
        start_year: int,
        end_year: int
    ) -> pd.DataFrame:
        """
        Test prediction accuracy across multiple seasons.

        Metrics by season:
        - Accuracy (correct winner %)
        - MAE (mean absolute error in margins)
        - Brier score (probability calibration)
        - Upset prediction accuracy
        """

    def analyze_model_blind_spots(
        self,
        predictions: pd.DataFrame,
        actuals: pd.DataFrame
    ) -> dict:
        """
        Identify systematic errors:
        - Does model overvalue high seeds?
        - Does it struggle with certain conferences?
        - Tempo matchup prediction accuracy
        - Style clash scenarios
        """
```

---

## 🎯 TIER 2: Advanced Features

### 5. Conference Strength Dynamics

**Current State**: ✅ Conference power ratings implemented
**Enhancement**: Temporal dynamics and head-to-head analysis

```python
class ConferenceDynamics:
    """Track conference strength evolution."""

    def track_conference_evolution(
        self,
        conference: str,
        start_year: int,
        end_year: int
    ) -> pd.DataFrame:
        """
        Multi-year conference strength trends:
        - Average AdjEM by year
        - Tournament success rate
        - Non-conference performance
        - Top team quality vs depth
        """

    def compare_conferences_head_to_head(
        self,
        conf1: str,
        conf2: str,
        season: int
    ) -> dict:
        """
        Non-conference matchup analysis:
        - Win percentage
        - Average margin
        - Quality of wins/losses
        """
```

---

### 6. Player Impact Extension

**Current State**: ✅ Player impact modeling implemented
**Enhancement**: Lineup combinations, rotation patterns

```python
class LineupAnalyzer:
    """Analyze lineup combinations and rotation impacts."""

    def analyze_rotation_depth(
        self,
        team: str,
        season: int
    ) -> dict:
        """
        Rotation analysis:
        - Bench contribution to AdjEM
        - Starter vs bench efficiency differential
        - Fatigue risk (minutes distribution)
        """

    def estimate_lineup_synergy(
        self,
        team: str,
        season: int
    ) -> dict:
        """
        Position combinations:
        - Guard-heavy vs big lineups
        - Shooting vs size trade-offs
        - Defensive versatility
        """
```

---

### 7. Home Court Advantage Deep-Dive

**Current State**: Basic HCA data from API
**Enhancement**: Venue-specific analysis

```python
class HomeCourtAnalyzer:
    """Detailed home court advantage analysis."""

    def analyze_venue_advantages(
        self,
        season: int
    ) -> pd.DataFrame:
        """
        Venue-specific factors:
        - Crowd size impact (capacity vs actual)
        - Altitude adjustments (Colorado, Wyoming, etc.)
        - Court dimensions (non-standard venues)
        - Travel distance impact
        """

    def identify_road_warriors(
        self,
        season: int,
        min_road_games: int = 5
    ) -> pd.DataFrame:
        """
        Teams that perform well on road:
        - Road AdjEM vs Home AdjEM
        - Experience factor
        - Coaching adjustments
        """
```

---

## 🎯 TIER 3: Visualization & Reporting

### 8. Interactive Dashboards

**Tool**: Streamlit + Plotly

**Dashboards to Build:**

1. **Team Comparison Dashboard**
   - Side-by-side efficiency metrics
   - Four Factors radar charts
   - Tempo profile visualizations
   - Strength of schedule comparisons

2. **Conference Analytics Dashboard**
   - Power ratings table
   - Inter-conference performance
   - Tempo/pace distributions
   - Tournament bid projections

3. **Tournament Simulator Dashboard**
   - Bracket visualization
   - Round-by-round probabilities
   - Upset likelihood heatmap
   - Pool optimizer

4. **Tempo Matchup Analyzer**
   - Interactive tempo scatter plots
   - APL mismatch visualizations
   - Historical pace trends
   - Style clash identification

---

### 9. Automated Reporting

```python
class ReportGenerator:
    """Generate comprehensive game/matchup reports."""

    def generate_matchup_report(
        self,
        team1: str,
        team2: str,
        game_date: str,
        format: str = "markdown"
    ) -> str:
        """
        Comprehensive scouting report:

        1. Executive Summary
        2. Efficiency Comparison
        3. Four Factors Breakdown
        4. Tempo/Pace Analysis
        5. Key Player Matchups
        6. Historical Head-to-Head
        7. Prediction with Confidence Intervals
        """
```

---

## 📊 Implementation Priority Matrix

| Feature | Priority | Effort | Impact | Status |
|---------|----------|--------|--------|--------|
| **Tournament Simulator** | 🔴 HIGHEST | High | Very High | ⏳ Not Started |
| **Tempo Dashboard** | 🔴 HIGH | Medium | High | ⏳ Not Started |
| **Four Factors Matchup** | 🟡 MEDIUM | Low | High | ⏳ Not Started |
| **Historical Backtesting** | 🟡 MEDIUM | Medium | Medium | ✅ **Implemented** |
| **Situational Tempo** | 🟡 MEDIUM | Medium | Medium | ⏳ Not Started |
| **Conference Dynamics** | 🟢 LOW | Low | Medium | ✅ **Partially Done** |
| **Lineup Analysis** | 🟢 LOW | High | Low | ⏳ Not Started |
| **HCA Deep-Dive** | 🟢 LOW | Medium | Low | ⏳ Not Started |
| **Automated Reports** | 🟢 LOW | Medium | Medium | ⏳ Not Started |

---

## 🎓 KenPom Methodology Alignment

### Core Principles to Maintain:

1. **Possession-Based Metrics**
   - All analysis uses per-100-possession normalization
   - Tempo-adjusted comparisons

2. **Strength-of-Schedule Adjusted**
   - Leverage KenPom's SOS calculations
   - Avoid naive comparisons of raw stats

3. **Probabilistic Thinking**
   - Win probabilities, not binary predictions
   - Confidence intervals for all estimates

4. **Methodological Transparency**
   - Document all assumptions
   - Validate against historical data
   - Report model limitations honestly

---

## 📝 Next Steps

### Immediate (This Sprint):
1. ✅ Implement Tournament Simulator with Monte Carlo
2. ⏳ Create Tempo Analysis Dashboard (Streamlit)
3. ⏳ Build Four Factors Matchup Breakdown

### Short-Term (Next Month):
4. ⏳ Add Situational Tempo Analysis
5. ⏳ Conference Dynamics Tracking
6. ⏳ Matchup Report Generator

### Medium-Term (Next Quarter):
7. ⏳ Historical Performance Backtesting Extension
8. ⏳ Player/Lineup Impact Analysis
9. ⏳ Home Court Advantage Deep-Dive

---

## 🔗 Data Requirements

### KenPom API Endpoints (Already Integrated):
- ✅ `ratings` - Team efficiency metrics
- ✅ `four-factors` - Dean Oliver's four factors
- ✅ `archive` - Historical ratings snapshots
- ✅ `fanmatch` - Game predictions
- ✅ `teams` - Team metadata
- ✅ `conferences` - Conference listings
- ✅ `misc-stats` - Additional team statistics
- ✅ `height` - Team height/experience
- ✅ `pointdist` - Point distribution

### Additional Data Sources (Optional):
- **Bart Torvik** - Complementary college hoops ratings
- **HoopMath** - Shot chart data (if available)
- **TeamRankings** - Trends and situational stats

---

## 💡 Innovation Opportunities

### Novel Analytics Not in Standard KenPom:

1. **Recency-Weighted Ratings**
   - Weight recent games more heavily (momentum)
   - Detect teams "peaking" at tournament time
   - Identify late-season slumps

2. **Injury Impact Quantification**
   - Model drop in AdjEM based on missing players
   - Practice report integration (if available)
   - Rotation depth analysis

3. **Coaching Adjustments**
   - Track coaches' tournament performance vs seed
   - Identify "underdog specialists"
   - Timeout usage patterns

4. **Schedule Difficulty Timing**
   - Not just SOS average, but "when" tough games occur
   - Rest disadvantage quantification
   - Conference tournament fatigue

5. **Style Evolution Tracking**
   - How teams adjust tempo throughout season
   - Pre-conference vs conference play differences
   - Tournament pace adjustments

---

## 📚 Documentation Needs

**To Add:**

1. **`docs/TEMPO_ANALYSIS_GUIDE.md`**
   - Explain APL metrics and their importance
   - How tempo control is calculated
   - Style mismatch interpretation

2. **`docs/PREDICTION_METHODOLOGY.md`**
   - Document all model assumptions
   - Feature engineering details
   - Validation procedures

3. **`docs/TOURNAMENT_SIMULATION.md`**
   - Monte Carlo methodology
   - Variance calculations
   - Upset probability interpretation

---

## 🎯 Success Metrics

**How We'll Measure Progress:**

1. **Prediction Accuracy**
   - Tournament bracket performance (percentile vs public)
   - Regular season game predictions (accuracy %)
   - Upset identification rate

2. **Model Performance**
   - Calibration (Brier score < 0.20)
   - MAE for margins (< 10 points)
   - R² for efficiency predictions (> 0.40)

3. **User Value**
   - Dashboards built and functional
   - Reports generated automatically
   - Analysis insights actionable

---

## Summary

**Current State**: Strong foundation with API integration, ML predictions, and tempo analysis

**Focus**: NCAA Division I Men's Basketball analytics using KenPom's rigorous methodology

**Top 3 Priorities**:
1. Tournament Monte Carlo simulator
2. Interactive tempo/pace dashboard
3. Four Factors matchup breakdown

**Expected Outcome**: Comprehensive basketball analytics platform with tournament simulation, advanced tempo analysis, and data-driven matchup insights.
