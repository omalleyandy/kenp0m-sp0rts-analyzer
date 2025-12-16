# KenPom Sports Analyzer - Advanced Analytics Roadmap

**Based on KenPom Blog Analysis & Current Repo Gap Assessment**

## Executive Summary

> **Note**: Billy Walters methodology is for NFL and NCAAF (college football) analysis, not basketball. This roadmap focuses on basketball-specific analytics using KenPom data. Betting-related features (CLV tracking, line movement analysis) are reference implementations and may not be directly applicable to basketball research.

The repository has a **strong data access foundation** with recent additions to machine learning-based predictions. This roadmap prioritizes implementations based on:
1. KenPom's blog methodologies
2. Modern sports analytics best practices
3. Advanced statistical modeling for basketball

---

## 🎯 TIER 1: Critical Missing Functionality (Implement First)

### 1. Billy Walters Methodology - Closing Line Value (CLV) Tracker

**Current Gap**: Claimed but not implemented
**Priority**: HIGHEST - Core to sports betting analytics

```python
# src/kenp0m_sp0rts_analyzer/betting.py

from dataclasses import dataclass
from datetime import datetime

@dataclass
class LineMovement:
    """Track line movement from opening to close"""
    game_id: str
    team: str
    opening_spread: float
    closing_spread: float
    opening_total: float
    closing_total: float
    movement: float
    reverse_line_move: bool  # Line moves against public betting %

class ClosingLineValueTracker:
    """Track CLV - the CORE of Billy Walters methodology"""

    def calculate_clv(self,
        your_bet_line: float,
        closing_line: float,
        bet_amount: float
    ) -> float:
        """
        CLV = (Closing Line - Your Line) * Bet Amount

        Positive CLV = You beat the closing line (sharp bet)
        Negative CLV = You got worse than closing (square bet)
        """
        return (closing_line - your_bet_line) * bet_amount

    def track_historical_clv(self,
        predictions: dict,
        actual_closing_lines: dict,
        results: dict
    ) -> dict:
        """Measure prediction accuracy vs closing lines"""
        # Compare kenpom predictions to closing lines
        # Track where model finds value vs market
        # Calculate CLV over time

    def detect_sharp_moves(self,
        line_history: list[LineMovement]
    ) -> list[str]:
        """Identify when sharp money moves lines"""
        # Line moves against public betting %
        # Large line moves on low volume
        # Professional bettor tracking
```

**Data Requirements**:
- Multiple sportsbook APIs (Pinnacle, DraftKings, FanDuel, BetMGM)
- Historical line movement data
- Public betting percentages (Action Network, Sports Insights)

---

### 2. Tournament Probability Engine

**Current Gap**: No tournament simulation
**Priority**: HIGH - KenPom publishes this every March

```python
# src/kenp0m_sp0rts_analyzer/tournament.py

import numpy as np
from scipy import stats

class TournamentSimulator:
    """Monte Carlo simulation for NCAA tournament"""

    def __init__(self, api_client):
        self.api = api_client

    def simulate_tournament(self,
        selection_sunday_date: str,
        num_simulations: int = 10000
    ) -> dict:
        """Run Monte Carlo simulation of tournament"""

        # Get ratings from Selection Sunday
        ratings = self.api.get_archive(d=selection_sunday_date)

        results = {
            'round_64': {},  # First round probabilities
            'round_32': {},
            'sweet_16': {},
            'elite_8': {},
            'final_4': {},
            'championship': {},
            'winner': {}
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
        """Simulate single game with variance"""
        # Use efficiency margins + uncertainty
        em_diff = team1['AdjEM'] - team2['AdjEM']

        # Add variance based on tempo (more possessions = more predictable)
        tempo = (team1['AdjTempo'] + team2['AdjTempo']) / 2
        variance = self._calculate_game_variance(tempo)

        # Sample from normal distribution
        predicted_margin = np.random.normal(em_diff, variance)

        return team1['TeamName'] if predicted_margin > 0 else team2['TeamName']

    def _calculate_game_variance(self, tempo: float) -> float:
        """Higher tempo = lower variance (law of large numbers)"""
        # Empirical: ~11 point standard deviation for average tempo
        return 11.0 * (68.0 / tempo) ** 0.5

    def analyze_upset_probability(self,
        higher_seed: int,
        lower_seed: int
    ) -> float:
        """Calculate upset probability by seed matchup"""
        # Historical seed vs seed performance
        # Compare to efficiency-based prediction
```

**Extension Ideas**:
- Bracket optimization for pools (max expected points)
- Identify "value" upset picks (high upset % but low public pick %)
- Simulate conference tournaments for auto-bids

---

### 3. Predictive Model with Uncertainty Quantification

**Current Gap**: Linear prediction only, no confidence intervals
**Priority**: HIGH - Essential for decision making

```python
# src/kenp0m_sp0rts_analyzer/models/predictive.py

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
import numpy as np

class GamePredictor:
    """Advanced prediction with uncertainty"""

    def __init__(self):
        self.margin_model = None  # Gradient Boosting for margin
        self.total_model = None   # Ridge regression for total

    def train_model(self, historical_data: pd.DataFrame):
        """Train on historical game results"""

        features = self._engineer_features(historical_data)
        # Features: AdjEM_diff, AdjTempo_diff, AdjO_diff, AdjD_diff,
        #           Home, Conf_game, Days_rest, Momentum, etc.

        X = features
        y_margin = historical_data['margin']
        y_total = historical_data['total_score']

        # Train margin predictor
        self.margin_model = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            loss='quantile',  # Enable quantile prediction for CI
            alpha=0.5
        )
        self.margin_model.fit(X, y_margin)

        # Train total predictor
        self.total_model = Ridge(alpha=1.0)
        self.total_model.fit(X, y_total)

    def predict_with_confidence(self,
        team1: dict,
        team2: dict,
        confidence_level: float = 0.90
    ) -> dict:
        """Predict margin with confidence intervals"""

        features = self._engineer_features_for_game(team1, team2)

        # Point prediction
        margin_pred = self.margin_model.predict(features)[0]
        total_pred = self.total_model.predict(features)[0]

        # Confidence intervals via quantile regression
        ci_lower = self._predict_quantile(features, (1-confidence_level)/2)
        ci_upper = self._predict_quantile(features, 1-(1-confidence_level)/2)

        return {
            'predicted_margin': margin_pred,
            'predicted_total': total_pred,
            'confidence_interval': (ci_lower, ci_upper),
            'confidence_level': confidence_level,
            'win_probability': self._margin_to_probability(margin_pred),
            'uncertainty': ci_upper - ci_lower  # Prediction uncertainty
        }

    def backtest_predictions(self,
        test_data: pd.DataFrame
    ) -> dict:
        """Validate model accuracy"""
        predictions = self.predict(test_data)
        actuals = test_data['margin']

        return {
            'mae': np.mean(np.abs(predictions - actuals)),
            'rmse': np.sqrt(np.mean((predictions - actuals)**2)),
            'accuracy': np.mean((predictions > 0) == (actuals > 0)),
            'calibration': self._calculate_calibration(predictions, actuals),
            'ats_record': self._against_spread_record(predictions, actuals, test_data['spread'])
        }
```

---

## 🎯 TIER 2: Advanced Analytics Features

### 4. Tempo-Pace Matchup Analysis

**KenPom Insight**: Possession length (APL) affects game outcomes
**Gap**: Not using APL data despite API providing it

```python
class TempoMatchupAnalyzer:
    """Analyze tempo/pace advantages in matchups"""

    def analyze_pace_advantage(self, team1: dict, team2: dict) -> dict:
        """Identify tempo-based stylistic advantages"""

        # Fast-paced team vs slow-paced team
        tempo_diff = team1['AdjTempo'] - team2['AdjTempo']

        # Average possession length analysis
        apl_diff = team1['APL_Off'] - team2['APL_Def']

        # Offensive style matching
        team1_3pt_rate = self._get_three_point_rate(team1)
        team2_3pt_defense = self._get_three_point_defense(team2)

        return {
            'tempo_advantage': 'team1' if tempo_diff > 3 else 'team2' if tempo_diff < -3 else 'neutral',
            'pace_impact_on_margin': self._estimate_pace_impact(tempo_diff),
            'style_mismatch': self._calculate_style_advantage(team1, team2),
            'three_point_edge': team1_3pt_rate - team2_3pt_defense
        }
```

### 5. Conference Strength Tracker

**KenPom Data**: Conference filtering, SOS metrics
**Gap**: No conference power ratings or comparison tools

```python
class ConferenceAnalytics:
    """Conference strength and comparison analysis"""

    def calculate_conference_power_ratings(self, year: int) -> pd.DataFrame:
        """Aggregate team ratings by conference"""

        conferences = self.api.get_conferences(year=year)

        ratings = []
        for conf in conferences.data:
            conf_teams = self.api.get_ratings(year=year, conference=conf['ConfShort'])

            ratings.append({
                'conference': conf['ConfLong'],
                'avg_adjEM': np.mean([t['AdjEM'] for t in conf_teams.data]),
                'top_team_adjEM': max([t['AdjEM'] for t in conf_teams.data]),
                'depth': len([t for t in conf_teams.data if t['AdjEM'] > 0]),  # Teams above avg
                'tournament_bids': self._estimate_bids(conf_teams.data)
            })

        return pd.DataFrame(ratings).sort_values('avg_adjEM', ascending=False)

    def compare_conferences_head_to_head(self,
        conf1: str,
        conf2: str,
        year: int
    ) -> dict:
        """Compare two conferences in non-conference games"""
        # Parse schedule data for inter-conference games
        # Calculate win %, avg margin, SOS comparison
```

### 6. Player Impact Modeling (Future)

**Gap**: API provides `playerstats` but repo doesn't use it
**Opportunity**: Extend beyond team-level to player-level

```python
class PlayerImpactAnalyzer:
    """Player-level analytics (future expansion)"""

    def calculate_player_value(self, team: str, season: int):
        """Estimate player contribution to team efficiency"""
        # Use kenpompy.summary.get_playerstats()
        # Calculate contribution to AdjEM
        # Injury impact modeling
```

---

## 🎯 TIER 3: Visualization & Reporting

### 7. Interactive Dashboard

**Current Gap**: Zero visualization capabilities
**Priority**: MEDIUM - Essential for insights

```python
# Option 1: Streamlit Dashboard
# scripts/dashboard.py

import streamlit as st
import plotly.express as px

def main():
    st.title("KenPom Advanced Analytics Dashboard")

    # Team comparison scatter
    fig = px.scatter(ratings,
        x='AdjOE',
        y='AdjDE',
        color='ConfShort',
        hover_data=['TeamName', 'AdjEM'],
        title='Offensive vs Defensive Efficiency'
    )
    st.plotly_chart(fig)

    # Tournament probability table
    # Matchup predictor
    # Conference power ratings
```

### 8. Report Generator

```python
class MatchupReportGenerator:
    """Generate comprehensive scouting reports"""

    def generate_pdf_report(self, team1: str, team2: str, game_date: str):
        """Create PDF with charts, predictions, key factors"""
        # Efficiency comparison charts
        # Four Factors breakdown
        # Historical head-to-head
        # Betting line analysis
        # Key player matchups
```

---

## 🎯 TIER 4: Data Engineering & Infrastructure

### 9. Real-Time Data Pipeline

**Gap**: All data is historical/manual fetch
**Opportunity**: Automated data refresh

```python
class DataRefreshScheduler:
    """Automated data updates"""

    def schedule_daily_refresh(self):
        """Pull latest ratings every morning"""
        # Use APScheduler or Celery
        # Update ratings, archive historical data
        # Trigger prediction updates
```

### 10. Betting Line Integration

**Critical Gap**: No sportsbook data ingestion
**Opportunity**: Core of Billy Walters methodology

```python
class SportsbookAPIClient:
    """Integrate multiple sportsbook APIs"""

    def get_current_lines(self, game_id: str) -> dict:
        """Fetch lines from multiple books"""
        # Pinnacle (sharp book - closing line benchmark)
        # DraftKings, FanDuel, BetMGM (public books)
        # Return best available lines

    def track_line_movement(self, game_id: str, interval_minutes: int = 15):
        """Record line changes over time"""
        # Store in database for CLV calculation
```

---

## 📊 Implementation Priority Matrix

| Feature | Priority | Effort | Impact | KenPom Blog Connection |
|---------|----------|--------|--------|------------------------|
| CLV Tracker | 🔴 HIGHEST | Medium | High | Billy Walters methodology (football) |
| Tournament Simulator | 🔴 HIGH | High | High | "2023 tourney forecast" |
| **Predictive Models** | ✅ **IMPLEMENTED** | High | High | Core analytics improvement |
| Uncertainty Quantification | 🟡 MEDIUM | Medium | Medium | Statistical rigor |
| Tempo/Pace Analysis | 🟡 MEDIUM | Low | Medium | KenPom efficiency focus |
| Conference Analytics | 🟡 MEDIUM | Low | Medium | "kenpom vs. the world" |
| Dashboard/Viz | 🟡 MEDIUM | High | High | User experience |
| Pairwise Rankings | 🟢 LOW | Medium | Low | "H.U.M.A.N. poll" |
| Player Analytics | 🟢 LOW | High | Medium | Future expansion |
| Real-Time Pipeline | 🟢 LOW | High | Medium | Infrastructure |

---

## 🎓 Learning from KenPom Blog

### Key Themes to Implement:

1. **Probabilistic Thinking**:
   - KenPom emphasizes win probabilities, not binary predictions
   - Implement: Tournament probability tables, confidence intervals

2. **Methodological Rigor**:
   - "kenpom vs. the world" shows importance of comparing prediction methods
   - Implement: Backtesting framework, method comparison tools

3. **Crowdsourced Intelligence**:
   - "H.U.M.A.N. poll" explores crowd wisdom
   - Implement: Bradley-Terry pairwise ranking aggregation

4. **Transparency**:
   - KenPom publishes methodologies openly
   - Implement: Document all model assumptions, feature engineering

---

## 📝 Recommended Next Steps

### Immediate (Next Sprint):
1. ⚠️  Implement `ClosingLineValueTracker` class (Note: For football, not basketball)
2. ⏳ Add `TournamentSimulator` with Monte Carlo
3. ✅ **COMPLETE**: Create `PredictiveModel` with confidence intervals
   - **Implementation**: `src/kenp0m_sp0rts_analyzer/prediction.py`
   - **Features**: Gradient Boosting, quantile regression for confidence intervals
   - **Backtesting**: Full validation framework with MAE, RMSE, accuracy, Brier score
   - **Tests**: 26 comprehensive test cases (100% passing)

### Short-Term (Next Month):
4. ✅ Build Streamlit dashboard for team/matchup viz
5. ✅ Add tempo/pace matchup analysis
6. ✅ Integrate at least one sportsbook API (start with Odds API)

### Medium-Term (Next Quarter):
7. ✅ Full Billy Walters CLV tracking system
8. ✅ Conference strength analytics module
9. ✅ Backtesting framework with historical validation

### Long-Term (Next 6 Months):
10. ✅ Player-level impact modeling
11. ✅ Real-time data pipeline with automated refresh
12. ✅ ML ensemble models (Gradient Boosting, Random Forest)

---

## 🔗 Data Source Requirements

To fully implement Billy Walters methodology, you need:

### Betting Data:
- **The Odds API** (free tier: 500 requests/month) - Historical lines
- **Pinnacle API** - Sharp book for CLV benchmarking
- **Action Network API** - Public betting percentages
- **Sports Insights** - Steam moves, line movement alerts

### Enhanced Basketball Data:
- **Synergy Sports** (if budget allows) - Play-by-play data
- **Bart Torvik** - Complementary college hoops ratings
- **HoopMath** - Shot chart data
- **TeamRankings** - Trends and betting angles

---

## 💡 Innovation Opportunities

### Novel Analytics Not in KenPom Blog:

1. **Recency-Weighted Ratings**:
   - Weight recent games more heavily (momentum)
   - Detect teams "peaking" at tournament time

2. **Injury Impact Quantification**:
   - Scrape practice reports (already in your plan!)
   - Model drop in AdjEM based on missing players

3. **Coaching Adjustments**:
   - Track coaches' tournament performance vs seed
   - Identify "underdog specialists"

4. **Schedule Difficulty Modeling**:
   - Not just SOS average, but "when" tough games occur
   - Rest disadvantage quantification

5. **Market Inefficiency Detection**:
   - Find systematic biases in betting markets
   - Public perception vs analytics (Duke always overvalued?)

---

## 📚 Documentation Needs

The repo claims Billy Walters methodology but doesn't explain it. Add:

1. **`docs/BILLY_WALTERS_METHODOLOGY.md`**
   - Explain CLV, sharp money, line movement
   - How kenpom fits into betting workflow

2. **`docs/PREDICTION_METHODOLOGY.md`**
   - Document all model assumptions
   - Feature engineering details
   - Validation procedures

3. **`docs/API_SPORTS_BOOKS.md`**
   - Document sportsbook API integration
   - Authentication, rate limits, data formats

---

## ⚠️ Ethical Considerations

Since this is sports betting-focused, document:

1. **Responsible Gaming**:
   - Tool is for research/education
   - Gambling disclaimers

2. **Data Usage**:
   - Respect KenPom's ToS
   - Rate limit API calls
   - Don't redistribute raw data

3. **Model Limitations**:
   - Document prediction accuracy honestly
   - Explain uncertainty bounds
   - No guarantees on betting outcomes

---

## 🎯 Summary

**Current State**: Solid data access foundation, minimal analytics
**Billy Walters Gap**: Methodology claimed but not implemented
**KenPom Blog Insights**: Tournament probabilities, methodological rigor, crowdsourced rankings

**Top 3 Priorities**:
1. Implement Closing Line Value (CLV) tracking system
2. Build Tournament Monte Carlo simulator
3. Add predictive models with confidence intervals

**Expected Outcome**: Transform from data wrapper into comprehensive sports analytics platform with legitimate Billy Walters betting methodology implementation.
