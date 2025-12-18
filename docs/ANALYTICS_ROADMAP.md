# KenPom Sports Analyzer - NCAA Basketball Analytics Roadmap

**Advanced Analytics for NCAA Division I Men's Basketball**

## Executive Summary

This roadmap focuses on implementing sophisticated basketball-specific analytics using KenPom data. The repository has a strong foundation with:
- ✅ Official KenPom API integration
- ✅ Machine learning predictions with confidence intervals
- ✅ Comprehensive tempo/pace analysis system
- ✅ Player impact modeling for injury analysis
- ✅ Conference power ratings and analytics
- ✅ Historical odds database infrastructure

**Focus Areas:**
1. Tournament probability modeling and simulation
2. Advanced tempo/pace matchup analysis
3. Four Factors matchup breakdowns
4. Historical performance backtesting
5. Vegas line comparison and CLV tracking
6. Interactive visualization and dashboards

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
        em_diff = team1['AdjEM'] - team2['AdjEM']
        tempo = (team1['AdjTempo'] + team2['AdjTempo']) / 2
        variance = self._calculate_game_variance(tempo)
        predicted_margin = np.random.normal(em_diff, variance)
        return team1['TeamName'] if predicted_margin > 0 else team2['TeamName']

    def _calculate_game_variance(self, tempo: float) -> float:
        """Higher tempo = lower variance (law of large numbers)."""
        return 11.0 * (68.0 / tempo) ** 0.5
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

#### Already Implemented (tempo_analysis.py):
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

    def analyze_tempo_by_game_state(self, team: str, season: int) -> dict:
        """
        Analyze how team's pace changes based on:
        - Leading vs trailing
        - Close game vs blowout
        - Conference vs non-conference
        - Home vs away vs neutral
        - Early season vs tournament time
        """

    def identify_tempo_adjustments(self, team: str, opponent_style: str) -> dict:
        """How does team adjust pace against different opponent styles?"""
```

**B. Tempo Trend Tracking**
```python
class TempoTrendAnalyzer:
    """Track tempo evolution throughout season."""

    def calculate_tempo_trends(self, team: str, season: int, window_size: int = 5) -> pd.DataFrame:
        """Rolling window analysis of tempo changes."""

    def detect_tempo_inflection_points(self, team: str, season: int) -> list:
        """Identify games where team significantly changed pace."""
```

---

### 3. Four Factors Matchup Breakdown

**Gap**: Using Four Factors data but not analyzing matchup interactions
**Value**: Identify specific strategic advantages

```python
class FourFactorsMatchup:
    """Analyze Four Factors matchup advantages."""

    def analyze_matchup(self, team1: str, team2: str, season: int) -> dict:
        """
        Breakdown by factor:
        1. eFG% Battle: Shooting vs defense
        2. Turnover Battle: Ball security vs pressure
        3. Rebounding Battle: Offensive vs defensive boards
        4. Free Throw Battle: Drawing fouls vs avoiding fouls
        """

    def identify_key_matchup_factors(self, analysis: dict) -> list:
        """Rank factors by impact on game outcome."""
```

---

### 4. Historical Performance Backtesting

**Current State**: ✅ Backtesting framework in `prediction.py`
**Gap**: Historical trend analysis, model evolution tracking

```python
class HistoricalAnalyzer:
    """Analyze historical KenPom prediction performance."""

    def backtest_predictions_by_season(self, start_year: int, end_year: int) -> pd.DataFrame:
        """Test prediction accuracy across multiple seasons."""

    def analyze_model_blind_spots(self, predictions: pd.DataFrame, actuals: pd.DataFrame) -> dict:
        """Identify systematic errors in predictions."""
```

---

## 🎯 TIER 2: Vegas Lines & Model Validation

### 5. Historical Odds Infrastructure ✅ COMPLETE

**Status**: Database schema and collection infrastructure implemented.

#### Database Schema
- `games` - Core game info (teams, date, scores)
- `odds_snapshots` - Time-series odds (opening, current, closing)
- `predictions` - Our model predictions
- `prediction_results` - Outcomes and CLV tracking

#### Automated Data Collection
- **Pre-game**: Scrape opening lines (morning) and closing lines (before tip)
- **Post-game**: Scrape final scores from ESPN
- **Daily job**: Update prediction results, calculate CLV

---

### 6. Closing Line Value (CLV) Analysis

**Why it matters**: CLV is the gold standard for measuring prediction skill.
If you consistently beat the closing line, you're finding +EV.

```
CLV = Your Line - Closing Line

Example:
- You bet Duke -5.5 (your model said -7)
- Line closes at Duke -7
- CLV = -7 - (-5.5) = -1.5 points of value
```

**Metrics to Track**:
- Average CLV per pick
- CLV by confidence level
- CLV by conference
- CLV by game type (rivalry, conference, etc.)

---

### 7. Against The Spread (ATS) Performance

Track model ATS record with breakdowns:
- By conference
- By spread size (favorites vs underdogs)
- By total (high-scoring vs low-scoring games)
- By day of week
- By month (early season vs late season)
- Home vs Away

---

### 8. Totals Performance

Over/Under analysis breakdowns:
- By pace matchups (fast vs slow)
- By defensive efficiency
- By altitude/location

---

### 9. Line Movement Correlation

Study how line movement predicts outcomes:
- Games where line moves toward our prediction = higher confidence
- Reverse line movement (sharp action) signals
- Steam moves and their predictive value

---

## 🎯 TIER 3: Model Calibration & Improvement

### 10. Identify Systematic Biases

**Questions to Answer**:
1. Do we overrate home court advantage?
2. Do we handle pace mismatches correctly?
3. Are we accurate on conference games vs non-conference?
4. How do we perform on back-to-backs / rest advantages?
5. Are we calibrated on big favorites vs small favorites?

```python
# Example: Check if we're biased on big favorites
big_favorites = games.where(spread <= -10)
our_error = our_prediction - actual_result
avg_error = big_favorites.our_error.mean()
# If avg_error > 0, we're overrating big favorites
```

---

### 11. Feature Importance Analysis

Track which factors best predict when we're right/wrong:
- Tempo differential
- Experience gap
- Conference strength
- Rest days
- Travel distance
- Injury impact

---

### 12. Confidence Calibration

Ensure confidence scores are well-calibrated:
- When we say 70% confident, do we win ~70%?
- Build calibration curves
- Adjust confidence thresholds for picks

---

## 🎯 TIER 4: Advanced Analytics

### 13. Market Efficiency by Segment

Find where Vegas is consistently off:
- Specific conferences (MAC? Sun Belt?)
- Early season games (less data = more edge)
- Games with key injuries not yet priced in
- Revenge games / rivalry games
- Teams after bye weeks

---

### 14. Situational Spots Database

Build a database of profitable situations:
- Team A after a blowout loss
- Team B on short rest
- Team C playing 3rd road game in a row
- Conference tournament trends

---

### 15. Ensemble Prediction Model

Combine multiple signals:
```python
final_prediction = (
    w1 * kenpom_prediction +
    w2 * tempo_adjusted_prediction +
    w3 * situational_model +
    w4 * historical_matchup_factor
)
```

---

### 16. Neural Network for Edge Detection

Train a model to predict WHEN our base model will be accurate:
- Input: game features + our prediction + Vegas line
- Output: probability our pick covers
- Use this to size bets / filter picks

---

## 🎯 TIER 5: Visualization & Reporting

### 17. Interactive Dashboards

**Tool**: Streamlit + Plotly

**Dashboards to Build:**
1. **Team Comparison Dashboard** - Side-by-side efficiency metrics
2. **Conference Analytics Dashboard** - Power ratings and tournament projections
3. **Tournament Simulator Dashboard** - Bracket visualization
4. **Tempo Matchup Analyzer** - Interactive tempo scatter plots
5. **CLV Tracking Dashboard** - Performance vs closing lines

---

### 18. Automated Reporting

```python
class ReportGenerator:
    """Generate comprehensive game/matchup reports."""

    def generate_matchup_report(self, team1: str, team2: str, game_date: str) -> str:
        """
        Comprehensive scouting report:
        1. Executive Summary
        2. Efficiency Comparison
        3. Four Factors Breakdown
        4. Tempo/Pace Analysis
        5. Vegas Line Comparison
        6. Prediction with Confidence Intervals
        """
```

---

## 📊 Implementation Priority Matrix

| Feature | Priority | Effort | Impact | Status |
|---------|----------|--------|--------|--------|
| **Historical Odds DB** | 🔴 HIGHEST | Medium | Very High | ✅ **Complete** |
| **Tournament Simulator** | 🔴 HIGHEST | High | Very High | ⏳ Not Started |
| **CLV Tracking** | 🔴 HIGH | Medium | High | ⏳ Not Started |
| **ATS Performance** | 🔴 HIGH | Low | High | ⏳ Not Started |
| **Tempo Dashboard** | 🟡 MEDIUM | Medium | High | ⏳ Not Started |
| **Four Factors Matchup** | 🟡 MEDIUM | Low | High | ⏳ Not Started |
| **Systematic Bias Analysis** | 🟡 MEDIUM | Medium | High | ⏳ Not Started |
| **Confidence Calibration** | 🟡 MEDIUM | Medium | Medium | ⏳ Not Started |
| **Situational Spots** | 🟢 LOW | High | Medium | ⏳ Not Started |
| **Ensemble Model** | 🟢 LOW | High | Medium | ⏳ Not Started |

---

## 📅 Implementation Timeline

### Immediate (This Week)
1. ✅ Create historical odds database
2. ⏳ Integrate scraper to auto-store odds
3. ⏳ Build results scraper (ESPN)
4. ⏳ Create daily update job

### Short-term (This Month)
5. Build CLV tracking dashboard
6. Create ATS record breakdown reports
7. Identify first systematic biases

### Medium-term (This Season)
8. Build situational spots database
9. Implement confidence calibration
10. Create ensemble prediction model

### Long-term (Next Season)
11. Full backtesting framework
12. Live line movement integration
13. Automated pick generation with Kelly sizing

---

## 🎯 Success Metrics

### Prediction Accuracy
- Tournament bracket performance (percentile vs public)
- Regular season game predictions (accuracy %)
- Upset identification rate

### Model Performance
- Calibration (Brier score < 0.20)
- MAE for margins (< 10 points)
- R² for efficiency predictions (> 0.40)

### Vegas Performance
- **Target**: 54%+ ATS win rate (profitable after vig)
- Consistent positive CLV
- 2-3 profitable situational spots identified

---

## 🔗 Data Sources

| Source | Data | Status |
|--------|------|--------|
| KenPom API | Predictions, ratings | ✅ Integrated |
| Overtime.ag | Vegas lines | ✅ Integrated |
| ESPN | Final scores | ✅ Integrated |
| Covers.com | Injury reports | ✅ Integrated |

---

## 📝 Notes

- Need ~500+ games for statistically significant analysis
- Conference play (Jan-Mar) has less variance than early season
- Tournament games are unique situations
- Always track confidence level with picks

---

## Summary

**Current State**: Strong foundation with API integration, ML predictions, tempo analysis, and historical odds infrastructure

**Focus**: NCAA Division I Men's Basketball analytics using KenPom's rigorous methodology combined with Vegas line comparison

**Top Priorities**:
1. CLV tracking and ATS performance analysis
2. Tournament Monte Carlo simulator
3. Interactive dashboards for all analytics

**Expected Outcome**: Comprehensive basketball analytics platform with tournament simulation, advanced tempo analysis, Vegas line comparison, and data-driven matchup insights.
