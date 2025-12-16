# High-Value Features: Implementation Plan

**Date**: 2025-12-16
**Status**: Planning & Brainstorming
**Priority**: TIER 2 Advanced Analytics

## Executive Summary

This document outlines the design, implementation approach, and technical specifications for four high-value analytical features:

1. **Tempo/Pace Matchup Analysis** - Leverage APL (Average Possession Length) data
2. **Conference Power Ratings** - Aggregate strength comparisons and rankings
3. **Pairwise Ranking System** - KenPom's H.U.M.A.N. poll methodology (Bradley-Terry)
4. **Player Impact Modeling** - Injury impact quantification and player value

Each feature builds on existing infrastructure (API client, prediction models, analysis functions) while adding sophisticated analytical capabilities not currently available in the repository.

---

## Feature 1: Tempo/Pace Matchup Analysis

### Business Value
- **Impact**: HIGH - Tempo mismatches significantly affect game outcomes
- **Complexity**: LOW - Data already available in API (APL fields)
- **Effort**: 4-6 hours
- **Dependencies**: None (uses existing API client)

### Current State
- ✅ Basic tempo data used in predictions (`AdjTempo`)
- ✅ Simple tempo difference calculated in matchups
- ❌ APL (Average Possession Length) data NOT used despite being available
- ❌ No analysis of offensive/defensive pace advantages
- ❌ No style mismatch detection (fast-paced vs slow-paced)

### Available Data (from KenPom API)
```
Ratings endpoint provides:
- AdjTempo: Adjusted Tempo (possessions per 40 minutes)
- RankAdjTempo: National tempo ranking
- APL_Off: Average Possession Length on Offense (seconds)
- APL_Def: Average Possession Length on Defense (seconds)
- ConfAPL_Off: Conference Average Possession Length (Offense)
- ConfAPL_Def: Conference Average Possession Length (Defense)
```

### Implementation Design

#### Module Structure
```python
# src/kenp0m_sp0rts_analyzer/tempo_analysis.py

from dataclasses import dataclass
from typing import Any
import pandas as pd
from .api_client import KenPomAPI

@dataclass
class TempoProfile:
    """Team's tempo and pace characteristics."""

    team_name: str
    adj_tempo: float
    rank_tempo: int
    apl_off: float  # Average possession length offense
    apl_def: float  # Average possession length defense
    conf_apl_off: float
    conf_apl_def: float
    pace_style: str  # "fast", "slow", "average"
    off_style: str  # "quick_strike", "methodical", "average"
    def_style: str  # "press", "halfcourt", "average"

@dataclass
class PaceMatchupAnalysis:
    """Analysis of tempo/pace advantage in a matchup."""

    team1_profile: TempoProfile
    team2_profile: TempoProfile

    # Tempo advantages
    tempo_differential: float  # team1 - team2
    expected_possessions: float  # Predicted game pace
    pace_advantage: str  # "team1", "team2", "neutral"

    # Style matchups
    style_mismatch_score: float  # 0-10 scale
    fast_favors: str  # Which team benefits from faster pace
    slow_favors: str  # Which team benefits from slower pace

    # APL-based insights
    off_pace_edge: float  # Team1's offensive pace vs Team2's defensive pace
    def_pace_edge: float  # Team1's defensive pace vs Team2's offensive pace

    # Projected impact
    tempo_impact_on_margin: float  # Estimated point impact
    optimal_pace_team1: float  # Ideal tempo for team1
    optimal_pace_team2: float  # Ideal tempo for team2
    pace_control: str  # Which team likely controls pace


class TempoMatchupAnalyzer:
    """Analyze tempo and pace advantages in basketball matchups."""

    def __init__(self, api: KenPomAPI | None = None):
        self.api = api or KenPomAPI()

    def get_tempo_profile(self, team_stats: dict[str, Any]) -> TempoProfile:
        """Extract tempo profile from team stats.

        Args:
            team_stats: Team statistics from ratings endpoint

        Returns:
            TempoProfile with classified pace characteristics
        """
        adj_tempo = team_stats["AdjTempo"]
        apl_off = team_stats["APL_Off"]
        apl_def = team_stats["APL_Def"]

        # Classify pace style (based on national averages)
        # Average tempo ~68 possessions, APL ~18 seconds
        pace_style = "fast" if adj_tempo > 70 else "slow" if adj_tempo < 66 else "average"

        # Offensive style based on possession length
        off_style = (
            "quick_strike" if apl_off < 17
            else "methodical" if apl_off > 19
            else "average"
        )

        # Defensive style
        def_style = (
            "press" if apl_def < 17
            else "halfcourt" if apl_def > 19
            else "average"
        )

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

    def analyze_pace_matchup(
        self,
        team1_stats: dict[str, Any],
        team2_stats: dict[str, Any],
    ) -> PaceMatchupAnalysis:
        """Analyze tempo/pace advantages between two teams.

        Key Insights:
        1. Tempo Differential: Who plays faster/slower
        2. Style Mismatch: Fast vs slow, methodical vs quick-strike
        3. APL Advantages: Offensive pace vs defensive pace allowed
        4. Pace Control: Which team likely dictates game flow
        5. Impact: How tempo affects expected margin

        Args:
            team1_stats: Team 1 statistics from ratings endpoint
            team2_stats: Team 2 statistics from ratings endpoint

        Returns:
            PaceMatchupAnalysis with comprehensive tempo insights
        """
        # Get tempo profiles
        team1_profile = self.get_tempo_profile(team1_stats)
        team2_profile = self.get_tempo_profile(team2_stats)

        # Calculate tempo differential
        tempo_diff = team1_profile.adj_tempo - team2_profile.adj_tempo

        # Expected possessions (average of both teams' tempos)
        expected_poss = (team1_profile.adj_tempo + team2_profile.adj_tempo) / 2

        # Determine pace advantage
        pace_advantage = (
            "team1" if tempo_diff > 3
            else "team2" if tempo_diff < -3
            else "neutral"
        )

        # APL-based edges
        # Team1's offensive pace vs Team2's defensive pace allowed
        off_pace_edge = team2_profile.apl_def - team1_profile.apl_off

        # Team2's offensive pace vs Team1's defensive pace allowed
        def_pace_edge = team1_profile.apl_def - team2_profile.apl_off

        # Calculate style mismatch score (0-10)
        style_mismatch = self._calculate_style_mismatch(
            team1_profile, team2_profile
        )

        # Determine which team benefits from tempo
        fast_favors = self._determine_tempo_beneficiary(
            team1_stats, team2_stats, "fast"
        )
        slow_favors = self._determine_tempo_beneficiary(
            team1_stats, team2_stats, "slow"
        )

        # Estimate tempo impact on margin
        tempo_impact = self._estimate_tempo_impact(
            team1_stats, team2_stats, tempo_diff
        )

        # Determine pace control
        pace_control = self._determine_pace_control(
            team1_profile, team2_profile, team1_stats, team2_stats
        )

        # Calculate optimal paces
        optimal_pace_team1 = self._calculate_optimal_pace(team1_stats)
        optimal_pace_team2 = self._calculate_optimal_pace(team2_stats)

        return PaceMatchupAnalysis(
            team1_profile=team1_profile,
            team2_profile=team2_profile,
            tempo_differential=tempo_diff,
            expected_possessions=expected_poss,
            pace_advantage=pace_advantage,
            style_mismatch_score=style_mismatch,
            fast_favors=fast_favors,
            slow_favors=slow_favors,
            off_pace_edge=off_pace_edge,
            def_pace_edge=def_pace_edge,
            tempo_impact_on_margin=tempo_impact,
            optimal_pace_team1=optimal_pace_team1,
            optimal_pace_team2=optimal_pace_team2,
            pace_control=pace_control,
        )

    def _calculate_style_mismatch(
        self, profile1: TempoProfile, profile2: TempoProfile
    ) -> float:
        """Calculate style mismatch score (0-10).

        Higher scores indicate greater style contrast:
        - Fast vs slow pace
        - Quick-strike vs methodical offense
        - Press vs halfcourt defense
        """
        score = 0.0

        # Pace style mismatch (0-3 points)
        if profile1.pace_style != profile2.pace_style:
            if "fast" in [profile1.pace_style, profile2.pace_style] and \
               "slow" in [profile1.pace_style, profile2.pace_style]:
                score += 3  # Maximum mismatch
            else:
                score += 1.5  # One average, one extreme

        # Offensive style mismatch (0-3 points)
        if profile1.off_style != profile2.off_style:
            if "quick_strike" in [profile1.off_style, profile2.off_style] and \
               "methodical" in [profile1.off_style, profile2.off_style]:
                score += 3
            else:
                score += 1.5

        # Defensive style mismatch (0-4 points)
        # Defense has more impact on pace control
        if profile1.def_style != profile2.def_style:
            if "press" in [profile1.def_style, profile2.def_style] and \
               "halfcourt" in [profile1.def_style, profile2.def_style]:
                score += 4
            else:
                score += 2

        return round(score, 1)

    def _determine_tempo_beneficiary(
        self,
        team1_stats: dict[str, Any],
        team2_stats: dict[str, Any],
        pace_type: str,
    ) -> str:
        """Determine which team benefits from a given pace.

        Logic:
        - Fast pace benefits teams with:
          - Higher AdjO (more possessions = more scoring)
          - Lower AdjD (prevents opponent from exploiting possessions)
          - Higher efficiency margin
        - Slow pace benefits teams that want to limit possessions
        """
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
            # Slow pace: favors limiting possessions when outmatched
            if team1_em < team2_em:
                return "team1"  # Underdog wants slow pace
            elif team2_em < team1_em:
                return "team2"
            else:
                return "neutral"

    def _estimate_tempo_impact(
        self,
        team1_stats: dict[str, Any],
        team2_stats: dict[str, Any],
        tempo_diff: float,
    ) -> float:
        """Estimate point impact of tempo differential.

        Formula:
        - Each possession is worth ~1.05 points (average efficiency)
        - Tempo diff translates to possession diff
        - Multiply by efficiency advantage

        Returns:
            Estimated point impact (positive favors team1)
        """
        team1_oe = team1_stats["AdjOE"]
        team2_oe = team2_stats["AdjOE"]

        # Efficiency advantage (points per 100 possessions)
        eff_advantage = (team1_oe - team2_oe) / 100

        # Tempo impact: more possessions = more impact
        # Rough estimate: 1 tempo point = 0.5 possessions per game
        possession_impact = tempo_diff * 0.5

        # Point impact
        point_impact = possession_impact * eff_advantage

        return round(point_impact, 2)

    def _determine_pace_control(
        self,
        profile1: TempoProfile,
        profile2: TempoProfile,
        team1_stats: dict[str, Any],
        team2_stats: dict[str, Any],
    ) -> str:
        """Determine which team likely controls the game pace.

        Pace control factors:
        1. Defensive style (press forces fast, halfcourt forces slow)
        2. Efficiency margin (better team dictates pace)
        3. Tempo preference strength
        """
        # Defensive style has most impact
        if profile1.def_style == "press" and profile2.def_style != "press":
            return "team1"
        elif profile2.def_style == "press" and profile1.def_style != "press":
            return "team2"

        # Better team usually controls pace
        team1_em = team1_stats["AdjEM"]
        team2_em = team2_stats["AdjEM"]

        if abs(team1_em - team2_em) > 10:
            return "team1" if team1_em > team2_em else "team2"

        # Otherwise, higher tempo team slightly favored
        if profile1.adj_tempo > profile2.adj_tempo + 5:
            return "team1"
        elif profile2.adj_tempo > profile1.adj_tempo + 5:
            return "team2"

        return "neutral"

    def _calculate_optimal_pace(self, team_stats: dict[str, Any]) -> float:
        """Calculate optimal game tempo for a team.

        Optimal pace based on:
        - Team's natural tempo preference
        - Offensive efficiency (higher = prefer more possessions)
        - Defensive efficiency (better = can handle more possessions)
        """
        adj_tempo = team_stats["AdjTempo"]
        adj_oe = team_stats["AdjOE"]
        adj_de = team_stats["AdjDE"]

        # Offensive adjustment: great offense wants more possessions
        # National average ~110 OE
        oe_adjustment = (adj_oe - 110) / 10  # ±1 tempo per 10 points

        # Defensive adjustment: great defense can handle faster pace
        # National average ~100 DE (lower is better)
        de_adjustment = -(adj_de - 100) / 10

        optimal = adj_tempo + oe_adjustment + de_adjustment

        return round(optimal, 1)

    def compare_conference_tempos(self, year: int) -> pd.DataFrame:
        """Compare average tempo across all conferences.

        Returns:
            DataFrame with conference tempo statistics
        """
        conferences = self.api.get_conferences(year=year)

        results = []
        for conf in conferences.data:
            conf_teams = self.api.get_ratings(year=year, conference=conf["ConfShort"])

            tempos = [team["AdjTempo"] for team in conf_teams.data]
            apl_offs = [team["APL_Off"] for team in conf_teams.data]
            apl_defs = [team["APL_Def"] for team in conf_teams.data]

            results.append({
                "Conference": conf["ConfLong"],
                "ConfShort": conf["ConfShort"],
                "AvgTempo": round(sum(tempos) / len(tempos), 2),
                "MaxTempo": max(tempos),
                "MinTempo": min(tempos),
                "AvgAPL_Off": round(sum(apl_offs) / len(apl_offs), 2),
                "AvgAPL_Def": round(sum(apl_defs) / len(apl_defs), 2),
                "NumTeams": len(conf_teams.data),
            })

        df = pd.DataFrame(results)
        return df.sort_values("AvgTempo", ascending=False)
```

### Integration Points

#### 1. Enhance `GamePredictor` with tempo features
```python
# src/kenp0m_sp0rts_analyzer/prediction.py

# Add to FeatureEngineer.FEATURE_NAMES:
FEATURE_NAMES = [
    # ... existing features
    "apl_off_diff",  # APL_Off team1 - team2
    "apl_def_diff",  # APL_Def team1 - team2
    "apl_mismatch",  # Team1's APL_Off vs Team2's APL_Def
    "tempo_style_score",  # Tempo mismatch score
]
```

#### 2. Add to MCP Server tools
```python
# src/kenp0m_sp0rts_analyzer/mcp_server.py

@server.tool()
async def analyze_tempo_matchup(
    team1: str, team2: str, year: int = 2025
) -> dict:
    """Analyze tempo and pace advantages in a matchup."""
    # ... implementation
```

#### 3. Add to analysis.py
```python
# src/kenp0m_sp0rts_analyzer/analysis.py

def analyze_matchup(team1: str, team2: str, ...) -> MatchupAnalysis:
    # Add tempo analysis to existing matchup function
    from .tempo_analysis import TempoMatchupAnalyzer

    tempo_analyzer = TempoMatchupAnalyzer(api)
    tempo_analysis = tempo_analyzer.analyze_pace_matchup(
        team1_stats, team2_stats
    )

    # Include in MatchupAnalysis result
```

### Testing Strategy

```python
# tests/test_tempo_analysis.py

class TestTempoProfile:
    def test_fast_paced_team(self):
        """Test classification of fast-paced team."""
        team_stats = {
            "TeamName": "Gonzaga",
            "AdjTempo": 72.5,
            "RankAdjTempo": 15,
            "APL_Off": 16.5,
            "APL_Def": 16.8,
            "ConfAPL_Off": 17.2,
            "ConfAPL_Def": 17.5,
        }

        analyzer = TempoMatchupAnalyzer()
        profile = analyzer.get_tempo_profile(team_stats)

        assert profile.pace_style == "fast"
        assert profile.off_style == "quick_strike"
        assert profile.def_style == "press"

class TestPaceMatchupAnalysis:
    def test_tempo_mismatch_fast_vs_slow(self):
        """Test analysis of fast vs slow tempo matchup."""
        # Fast team
        team1_stats = {...}  # AdjTempo 72, APL_Off 16
        # Slow team
        team2_stats = {...}  # AdjTempo 64, APL_Off 20

        analyzer = TempoMatchupAnalyzer()
        analysis = analyzer.analyze_pace_matchup(team1_stats, team2_stats)

        assert analysis.pace_advantage == "team1"
        assert analysis.style_mismatch_score > 5
        assert analysis.tempo_differential == 8

class TestConferenceTempo:
    def test_compare_conference_tempos(self):
        """Test conference tempo comparison."""
        analyzer = TempoMatchupAnalyzer()
        df = analyzer.compare_conference_tempos(year=2025)

        assert len(df) > 0
        assert "AvgTempo" in df.columns
        assert df["AvgTempo"].min() > 60
        assert df["AvgTempo"].max() < 80
```

### Documentation & Examples

```python
# examples/tempo_analysis_demo.py

from kenp0m_sp0rts_analyzer.api_client import KenPomAPI
from kenp0m_sp0rts_analyzer.tempo_analysis import TempoMatchupAnalyzer

# Initialize
api = KenPomAPI()
tempo_analyzer = TempoMatchupAnalyzer(api)

# Get team stats
duke_stats = api.get_team_by_name("Duke", 2025)
unc_stats = api.get_team_by_name("North Carolina", 2025)

# Analyze tempo matchup
analysis = tempo_analyzer.analyze_pace_matchup(duke_stats, unc_stats)

print(f"Tempo Differential: {analysis.tempo_differential}")
print(f"Expected Possessions: {analysis.expected_possessions}")
print(f"Pace Advantage: {analysis.pace_advantage}")
print(f"Style Mismatch: {analysis.style_mismatch_score}/10")
print(f"Fast Pace Favors: {analysis.fast_favors}")
print(f"Tempo Impact on Margin: {analysis.tempo_impact_on_margin} points")

# Conference comparison
conf_tempos = tempo_analyzer.compare_conference_tempos(year=2025)
print("\nFastest Conferences:")
print(conf_tempos.head())
```

### Performance Considerations
- ✅ All data available in single API call (ratings endpoint)
- ✅ No additional API requests needed
- ✅ Calculations are lightweight (no ML training)
- ⚠️ Conference comparison requires multiple API calls (one per conference)

### Future Enhancements
1. **Historical Tempo Trends**: Track team tempo evolution across seasons
2. **Situational Tempo**: Tempo in close games vs blowouts
3. **Tournament Tempo**: How tempo changes in March Madness
4. **Opponent Adjustment**: How teams adjust pace against different opponents

---

## Feature 2: Conference Power Ratings

### Business Value
- **Impact**: MEDIUM - Useful for tournament selection, conference strength debates
- **Complexity**: LOW - Aggregation of existing data
- **Effort**: 3-4 hours
- **Dependencies**: None

### Current State
- ✅ Conference filtering in API (`get_ratings(conference=...)`)
- ✅ Basic conference standings in `analysis.py`
- ❌ No aggregate conference strength metrics
- ❌ No head-to-head conference comparisons
- ❌ No conference depth analysis

### Implementation Design

```python
# src/kenp0m_sp0rts_analyzer/conference_analytics.py

from dataclasses import dataclass
import pandas as pd
from .api_client import KenPomAPI

@dataclass
class ConferencePowerRating:
    """Aggregate power rating for a conference."""

    conference: str
    conf_short: str
    num_teams: int

    # Aggregate metrics
    avg_adj_em: float
    median_adj_em: float
    top_team_adj_em: float
    bottom_team_adj_em: float

    # Depth metrics
    top_25_teams: int  # Teams in top 25 by AdjEM
    top_50_teams: int
    above_average: int  # Teams with AdjEM > 0

    # Quality metrics
    avg_adj_oe: float
    avg_adj_de: float
    avg_tempo: float

    # Strength of schedule
    avg_sos: float
    avg_ncsos: float

    # Projected tournament bids
    estimated_bids: int
    bid_stealers: int  # Teams likely to win conf tournament despite low rating

    # Power rating score (0-100)
    power_score: float
    power_rank: int

@dataclass
class ConferenceHeadToHead:
    """Head-to-head comparison between two conferences."""

    conf1: str
    conf2: str
    games_played: int
    conf1_wins: int
    conf2_wins: int
    win_percentage: float  # conf1's win %
    avg_margin: float  # conf1's average margin

    # Quality of matchups
    top_vs_top: str  # Record when both teams top 50
    best_win_conf1: str  # Best win by conf1 team
    worst_loss_conf1: str  # Worst loss by conf1 team

class ConferenceAnalytics:
    """Advanced conference strength and comparison analytics."""

    def __init__(self, api: KenPomAPI | None = None):
        self.api = api or KenPomAPI()

    def calculate_conference_power_ratings(
        self, year: int
    ) -> pd.DataFrame:
        """Calculate comprehensive power ratings for all conferences.

        Power Rating Formula:
        1. Average AdjEM (40% weight)
        2. Top team strength (20% weight)
        3. Depth - teams above average (20% weight)
        4. Top 25 representation (10% weight)
        5. Non-conference SOS (10% weight)

        Returns:
            DataFrame with conference power ratings ranked
        """
        conferences = self.api.get_conferences(year=year)

        all_ratings = []
        for conf in conferences.data:
            rating = self._calculate_single_conference_rating(
                conf["ConfShort"], year
            )
            all_ratings.append(rating)

        # Convert to DataFrame
        df = pd.DataFrame([vars(r) for r in all_ratings])

        # Calculate power scores
        df = self._calculate_power_scores(df)

        # Rank conferences
        df["power_rank"] = df["power_score"].rank(ascending=False)

        return df.sort_values("power_score", ascending=False)

    def _calculate_single_conference_rating(
        self, conf_short: str, year: int
    ) -> ConferencePowerRating:
        """Calculate rating for a single conference."""
        teams = self.api.get_ratings(year=year, conference=conf_short)

        adj_ems = [t["AdjEM"] for t in teams.data]
        adj_oes = [t["AdjOE"] for t in teams.data]
        adj_des = [t["AdjDE"] for t in teams.data]
        tempos = [t["AdjTempo"] for t in teams.data]
        sos_vals = [t["SOS"] for t in teams.data]
        ncsos_vals = [t["NCSOS"] for t in teams.data]

        # Depth metrics
        top_25 = sum(1 for em in adj_ems if em >= 15)  # Rough top 25 threshold
        top_50 = sum(1 for em in adj_ems if em >= 10)
        above_avg = sum(1 for em in adj_ems if em > 0)

        # Tournament bid estimation
        estimated_bids = self._estimate_tournament_bids(teams.data)

        return ConferencePowerRating(
            conference=teams.data[0]["ConfShort"],  # Get full name from first team
            conf_short=conf_short,
            num_teams=len(teams.data),
            avg_adj_em=sum(adj_ems) / len(adj_ems),
            median_adj_em=sorted(adj_ems)[len(adj_ems) // 2],
            top_team_adj_em=max(adj_ems),
            bottom_team_adj_em=min(adj_ems),
            top_25_teams=top_25,
            top_50_teams=top_50,
            above_average=above_avg,
            avg_adj_oe=sum(adj_oes) / len(adj_oes),
            avg_adj_de=sum(adj_des) / len(adj_des),
            avg_tempo=sum(tempos) / len(tempos),
            avg_sos=sum(sos_vals) / len(sos_vals),
            avg_ncsos=sum(ncsos_vals) / len(ncsos_vals),
            estimated_bids=estimated_bids,
            bid_stealers=0,  # Calculated separately
            power_score=0.0,  # Calculated after all conferences
            power_rank=0,
        )

    def _estimate_tournament_bids(self, teams: list[dict]) -> int:
        """Estimate NCAA tournament bids for conference.

        Estimation logic:
        - AdjEM > 15: Likely tournament team
        - AdjEM 10-15: Bubble team (50% chance)
        - AdjEM < 10: Unlikely unless auto-bid
        - Always include at least 1 (auto-bid)
        """
        likely = sum(1 for t in teams if t["AdjEM"] > 15)
        bubble = sum(1 for t in teams if 10 <= t["AdjEM"] <= 15)

        estimated = likely + (bubble // 2)
        return max(estimated, 1)  # At least auto-bid

    def _calculate_power_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate normalized power scores (0-100) for conferences."""
        # Normalize each metric to 0-100 scale
        df["em_score"] = self._normalize_column(df["avg_adj_em"]) * 40
        df["top_score"] = self._normalize_column(df["top_team_adj_em"]) * 20
        df["depth_score"] = self._normalize_column(df["above_average"]) * 20
        df["top25_score"] = self._normalize_column(df["top_25_teams"]) * 10
        df["ncsos_score"] = self._normalize_column(df["avg_ncsos"]) * 10

        # Total power score
        df["power_score"] = (
            df["em_score"] +
            df["top_score"] +
            df["depth_score"] +
            df["top25_score"] +
            df["ncsos_score"]
        )

        return df

    @staticmethod
    def _normalize_column(series: pd.Series) -> pd.Series:
        """Normalize series to 0-1 range."""
        min_val = series.min()
        max_val = series.max()
        if max_val == min_val:
            return pd.Series([1.0] * len(series))
        return (series - min_val) / (max_val - min_val)

    def compare_conferences_head_to_head(
        self, conf1: str, conf2: str, year: int
    ) -> ConferenceHeadToHead:
        """Compare two conferences in non-conference games.

        Note: Requires schedule data which is not in current API.
        This is a FUTURE ENHANCEMENT placeholder.
        """
        # TODO: Implement when schedule endpoint is available
        raise NotImplementedError(
            "Head-to-head comparison requires schedule data. "
            "Use kenpompy library for this feature."
        )

    def get_conference_tournament_outlook(
        self, conf_short: str, year: int
    ) -> pd.DataFrame:
        """Analyze conference tournament implications.

        Returns teams ranked with tournament odds.
        """
        teams = self.api.get_ratings(year=year, conference=conf_short)

        results = []
        for team in teams.data:
            # Estimate tournament probability
            adj_em = team["AdjEM"]
            if adj_em > 15:
                ncaa_prob = 0.95
            elif adj_em > 10:
                ncaa_prob = 0.7
            elif adj_em > 5:
                ncaa_prob = 0.4
            elif adj_em > 0:
                ncaa_prob = 0.15
            else:
                ncaa_prob = 0.05

            results.append({
                "Team": team["TeamName"],
                "AdjEM": adj_em,
                "RankAdjEM": team["RankAdjEM"],
                "NCAA_Probability": ncaa_prob,
                "Bubble": "Lock" if ncaa_prob > 0.8
                         else "Bubble" if ncaa_prob > 0.3
                         else "NIT/CBI",
            })

        df = pd.DataFrame(results)
        return df.sort_values("AdjEM", ascending=False)
```

### Testing Strategy

```python
# tests/test_conference_analytics.py

class TestConferencePowerRatings:
    def test_calculate_conference_ratings(self):
        """Test conference power rating calculation."""
        analytics = ConferenceAnalytics()
        df = analytics.calculate_conference_power_ratings(year=2025)

        assert len(df) > 0
        assert "power_score" in df.columns
        assert df["power_score"].max() <= 100
        assert df["power_score"].min() >= 0

    def test_top_conference_metrics(self):
        """Test that top conferences have expected characteristics."""
        analytics = ConferenceAnalytics()
        df = analytics.calculate_conference_power_ratings(year=2025)

        # Top conference should have high average AdjEM
        top_conf = df.iloc[0]
        assert top_conf["avg_adj_em"] > 5
        assert top_conf["top_25_teams"] > 0
```

---

## Feature 3: Pairwise Ranking System

### Business Value
- **Impact**: MEDIUM - Alternative to standard rankings, crowdsourced wisdom
- **Complexity**: MEDIUM - Requires Bradley-Terry model implementation
- **Effort**: 6-8 hours
- **Dependencies**: scipy, pandas

### Background: KenPom's H.U.M.A.N. Poll

KenPom's H.U.M.A.N. poll aggregates pairwise comparisons using the Bradley-Terry model. Instead of ranking teams 1-N, users make head-to-head comparisons, which are then converted to rankings.

**Advantages**:
- More intuitive than absolute rankings
- Captures transitive relationships
- Aggregates crowd wisdom effectively

### Implementation Design

```python
# src/kenp0m_sp0rts_analyzer/pairwise_rankings.py

import pandas as pd
import numpy as np
from scipy.optimize import minimize
from dataclasses import dataclass

@dataclass
class PairwiseComparison:
    """A single pairwise comparison between two teams."""

    team_a: str
    team_b: str
    winner: str  # "team_a" or "team_b"
    confidence: float  # 1-5 scale
    voter_id: str | None = None

@dataclass
class BradleyTerryRating:
    """Bradley-Terry model rating for a team."""

    team: str
    rating: float  # Log-scale rating
    rank: int
    win_probability_vs_average: float

class PairwiseRankingSystem:
    """Implement Bradley-Terry model for pairwise rankings."""

    def __init__(self):
        self.teams: list[str] = []
        self.ratings: dict[str, float] = {}
        self.comparisons: list[PairwiseComparison] = []

    def add_comparison(
        self,
        team_a: str,
        team_b: str,
        winner: str,
        confidence: float = 3.0,
        voter_id: str | None = None,
    ) -> None:
        """Add a pairwise comparison.

        Args:
            team_a: First team in comparison
            team_b: Second team in comparison
            winner: "team_a" or "team_b"
            confidence: 1-5 scale (5 = very confident)
            voter_id: Optional identifier for vote tracking
        """
        comparison = PairwiseComparison(
            team_a=team_a,
            team_b=team_b,
            winner=winner,
            confidence=confidence,
            voter_id=voter_id,
        )
        self.comparisons.append(comparison)

        # Track teams
        if team_a not in self.teams:
            self.teams.append(team_a)
        if team_b not in self.teams:
            self.teams.append(team_b)

    def fit_bradley_terry_model(self) -> dict[str, BradleyTerryRating]:
        """Fit Bradley-Terry model to pairwise comparisons.

        Bradley-Terry Model:
        P(A beats B) = exp(r_A) / (exp(r_A) + exp(r_B))

        Where r_A and r_B are team ratings.

        We solve for ratings by maximum likelihood estimation.

        Returns:
            Dictionary mapping team names to BradleyTerryRating objects
        """
        if len(self.comparisons) == 0:
            raise ValueError("No comparisons to fit model")

        n_teams = len(self.teams)
        team_to_idx = {team: i for i, team in enumerate(self.teams)}

        # Build comparison matrix
        # wins[i][j] = number of times team i beat team j
        wins = np.zeros((n_teams, n_teams))

        for comp in self.comparisons:
            i = team_to_idx[comp.team_a]
            j = team_to_idx[comp.team_b]

            # Weight by confidence (1-5 scale)
            weight = comp.confidence

            if comp.winner == "team_a":
                wins[i][j] += weight
            else:
                wins[j][i] += weight

        # Maximum likelihood estimation
        initial_ratings = np.zeros(n_teams)

        def negative_log_likelihood(ratings):
            """Negative log-likelihood for Bradley-Terry model."""
            ll = 0.0
            for i in range(n_teams):
                for j in range(n_teams):
                    if wins[i][j] > 0:
                        # P(i beats j) = exp(r_i) / (exp(r_i) + exp(r_j))
                        prob = np.exp(ratings[i]) / (
                            np.exp(ratings[i]) + np.exp(ratings[j])
                        )
                        ll += wins[i][j] * np.log(prob + 1e-10)
            return -ll

        # Optimize
        result = minimize(
            negative_log_likelihood,
            initial_ratings,
            method='BFGS',
        )

        optimal_ratings = result.x

        # Normalize ratings (set average to 0)
        optimal_ratings -= optimal_ratings.mean()

        # Create BradleyTerryRating objects
        self.ratings = {}
        for team, idx in team_to_idx.items():
            rating_value = optimal_ratings[idx]

            # Win probability vs average team (rating = 0)
            win_prob = 1 / (1 + np.exp(-rating_value))

            self.ratings[team] = BradleyTerryRating(
                team=team,
                rating=rating_value,
                rank=0,  # Will be set after sorting
                win_probability_vs_average=win_prob,
            )

        # Rank teams
        sorted_teams = sorted(
            self.ratings.items(),
            key=lambda x: x[1].rating,
            reverse=True
        )
        for rank, (team, rating_obj) in enumerate(sorted_teams, 1):
            rating_obj.rank = rank

        return self.ratings

    def get_rankings_dataframe(self) -> pd.DataFrame:
        """Get rankings as a DataFrame."""
        if not self.ratings:
            raise ValueError("Model not fit yet. Call fit_bradley_terry_model() first.")

        data = []
        for team, rating in self.ratings.items():
            data.append({
                "Rank": rating.rank,
                "Team": team,
                "Rating": round(rating.rating, 3),
                "WinProb_vs_Avg": round(rating.win_probability_vs_average, 3),
            })

        df = pd.DataFrame(data)
        return df.sort_values("Rank")

    def predict_matchup(self, team_a: str, team_b: str) -> dict[str, float]:
        """Predict outcome of a matchup using Bradley-Terry ratings.

        Returns:
            Dictionary with win probabilities for each team
        """
        if team_a not in self.ratings or team_b not in self.ratings:
            raise ValueError("Teams not in rating system")

        rating_a = self.ratings[team_a].rating
        rating_b = self.ratings[team_b].rating

        # Bradley-Terry win probability
        prob_a = 1 / (1 + np.exp(-(rating_a - rating_b)))
        prob_b = 1 - prob_a

        return {
            team_a: prob_a,
            team_b: prob_b,
        }

    def compare_to_kenpom(self, kenpom_ratings: pd.DataFrame) -> pd.DataFrame:
        """Compare pairwise rankings to KenPom rankings.

        Args:
            kenpom_ratings: DataFrame from API with TeamName and RankAdjEM

        Returns:
            DataFrame with both rankings and differences
        """
        pairwise_df = self.get_rankings_dataframe()

        # Merge with KenPom
        comparison = pairwise_df.merge(
            kenpom_ratings[["TeamName", "RankAdjEM", "AdjEM"]],
            left_on="Team",
            right_on="TeamName",
            how="inner"
        )

        comparison["Rank_Diff"] = comparison["Rank"] - comparison["RankAdjEM"]
        comparison = comparison.sort_values("Rank")

        return comparison[
            ["Rank", "Team", "Rating", "RankAdjEM", "AdjEM", "Rank_Diff"]
        ]

def create_pairwise_poll_from_kenpom(
    year: int,
    num_matchups: int = 100,
    api: KenPomAPI | None = None,
) -> PairwiseRankingSystem:
    """Create synthetic pairwise poll based on KenPom ratings.

    This generates pairwise comparisons by sampling matchups and
    using KenPom AdjEM to determine winners.

    Args:
        year: Season year
        num_matchups: Number of pairwise comparisons to generate
        api: Optional KenPomAPI instance

    Returns:
        PairwiseRankingSystem with fitted model
    """
    if api is None:
        api = KenPomAPI()

    ratings = api.get_ratings(year=year)
    teams = ratings.data[:68]  # Top 68 teams (tournament field)

    system = PairwiseRankingSystem()

    # Generate random matchups
    for _ in range(num_matchups):
        team_a = np.random.choice(teams)
        team_b = np.random.choice(teams)

        if team_a["TeamName"] == team_b["TeamName"]:
            continue

        # Determine winner based on AdjEM
        em_diff = team_a["AdjEM"] - team_b["AdjEM"]

        # Add noise to simulate voter disagreement
        noise = np.random.normal(0, 3)  # ±3 points std dev
        em_diff_noisy = em_diff + noise

        winner = "team_a" if em_diff_noisy > 0 else "team_b"

        # Confidence based on margin
        confidence = min(5.0, max(1.0, abs(em_diff) / 5))

        system.add_comparison(
            team_a=team_a["TeamName"],
            team_b=team_b["TeamName"],
            winner=winner,
            confidence=confidence,
        )

    # Fit model
    system.fit_bradley_terry_model()

    return system
```

### Testing Strategy

```python
# tests/test_pairwise_rankings.py

class TestPairwiseRankingSystem:
    def test_add_comparison(self):
        """Test adding pairwise comparisons."""
        system = PairwiseRankingSystem()
        system.add_comparison("Duke", "UNC", "Duke", confidence=5.0)

        assert len(system.comparisons) == 1
        assert "Duke" in system.teams
        assert "UNC" in system.teams

    def test_bradley_terry_fit(self):
        """Test Bradley-Terry model fitting."""
        system = PairwiseRankingSystem()

        # Duke beats UNC
        system.add_comparison("Duke", "UNC", "Duke")
        # Duke beats Virginia
        system.add_comparison("Duke", "Virginia", "Duke")
        # UNC beats Virginia
        system.add_comparison("UNC", "Virginia", "UNC")

        ratings = system.fit_bradley_terry_model()

        # Duke should be ranked #1
        assert ratings["Duke"].rank == 1
        assert ratings["UNC"].rank == 2
        assert ratings["Virginia"].rank == 3

    def test_predict_matchup(self):
        """Test matchup prediction."""
        system = PairwiseRankingSystem()
        system.add_comparison("Duke", "UNC", "Duke")
        system.fit_bradley_terry_model()

        probs = system.predict_matchup("Duke", "UNC")

        assert probs["Duke"] > 0.5
        assert probs["UNC"] < 0.5
        assert abs(probs["Duke"] + probs["UNC"] - 1.0) < 0.01
```

---

## Feature 4: Player Impact Modeling

### Business Value
- **Impact**: HIGH - Injury impact is crucial for betting and analysis
- **Complexity**: HIGH - Requires player-level data and statistical modeling
- **Effort**: 12-16 hours
- **Dependencies**: kenpompy library (for player stats), sklearn

### Current State
- ❌ No player-level data ingestion
- ❌ No injury impact modeling
- ❌ KenPom API has `playerstats` endpoint but not documented
- ⚠️ Requires kenpompy library for player data

### Available Data (kenpompy library)
```python
from kenpompy.summary import get_playerstats

# Returns DataFrame with:
# Player, Team, Conf, ht, yr, Poss, Poss%,
# ORtg, Shot%, eFG%, TS%, ORB%, DRB%, TRB%,
# AST%, TO%, STL%, BLK%, FC/40, FD/40, FTRate
```

### Implementation Design

```python
# src/kenp0m_sp0rts_analyzer/player_impact.py

from dataclasses import dataclass
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge

@dataclass
class PlayerValue:
    """Player's estimated value and impact."""

    player_name: str
    team: str
    position: str

    # Usage metrics
    possession_pct: float  # % of team possessions used
    minutes_pct: float  # % of available minutes

    # Efficiency metrics
    offensive_rating: float
    effective_fg_pct: float
    true_shooting_pct: float

    # Impact metrics
    estimated_value: float  # Points of AdjEM contribution
    offensive_contribution: float
    defensive_contribution: float

    # Replaceability
    replacement_level: float  # Expected performance of backup
    value_over_replacement: float  # Value - Replacement

@dataclass
class InjuryImpact:
    """Estimated impact of player absence."""

    player: PlayerValue
    team_adj_em_baseline: float

    # Impact estimates
    estimated_adj_em_loss: float
    confidence_interval: tuple[float, float]

    # Adjusted ratings
    adjusted_adj_em: float
    adjusted_adj_oe: float
    adjusted_adj_de: float

    # Severity classification
    severity: str  # "Minor", "Moderate", "Major", "Devastating"

class PlayerImpactModel:
    """Model player contributions and injury impact."""

    def __init__(self):
        self.value_model: Ridge | None = None
        self.player_values: dict[str, PlayerValue] = {}

    def calculate_player_value(
        self,
        player_stats: dict,
        team_stats: dict,
    ) -> PlayerValue:
        """Calculate player's estimated value to team.

        Value estimation:
        1. Usage: possession % and minutes %
        2. Efficiency: ORtg, eFG%, TS%
        3. Team context: How player fits in team's system
        4. Contribution: Estimated AdjEM points contributed

        Args:
            player_stats: Player statistics from kenpompy
            team_stats: Team statistics from API

        Returns:
            PlayerValue object
        """
        # Usage metrics
        poss_pct = player_stats["Poss%"]

        # Estimate minutes % (not directly available, use possession %)
        # Rough approximation: high usage = more minutes
        minutes_pct = min(100, poss_pct * 1.2)

        # Efficiency
        ortg = player_stats["ORtg"]
        efg = player_stats["eFG%"]
        ts = player_stats["TS%"]

        # Estimate offensive contribution
        # Player's ORtg above team ORtg, weighted by usage
        team_oe = team_stats["AdjOE"]
        oe_above_team = (ortg - team_oe) * (poss_pct / 100)

        # Defensive contribution (harder to estimate from stats)
        # Use rebounds, steals, blocks as proxies
        trb_pct = player_stats["TRB%"]
        stl_pct = player_stats.get("STL%", 0)
        blk_pct = player_stats.get("BLK%", 0)

        defensive_impact = (trb_pct + stl_pct * 2 + blk_pct * 2) / 10

        # Total value (points of AdjEM)
        estimated_value = oe_above_team + defensive_impact

        # Replacement level (assume backup is 70% as efficient)
        replacement_level = estimated_value * 0.7
        vor = estimated_value - replacement_level

        return PlayerValue(
            player_name=player_stats["Player"],
            team=player_stats["Team"],
            position="",  # Not in data
            possession_pct=poss_pct,
            minutes_pct=minutes_pct,
            offensive_rating=ortg,
            effective_fg_pct=efg,
            true_shooting_pct=ts,
            estimated_value=estimated_value,
            offensive_contribution=oe_above_team,
            defensive_contribution=defensive_impact,
            replacement_level=replacement_level,
            value_over_replacement=vor,
        )

    def estimate_injury_impact(
        self,
        player: PlayerValue,
        team_stats: dict,
        injury_severity: str = "out",  # "questionable", "doubtful", "out"
    ) -> InjuryImpact:
        """Estimate impact of player absence on team ratings.

        Injury Impact Model:
        - "questionable": 50% of value lost
        - "doubtful": 75% of value lost
        - "out": 100% of value lost (replaced by backup)

        Confidence intervals based on:
        - Player value variability
        - Replacement player uncertainty
        - Team system resilience

        Args:
            player: PlayerValue object
            team_stats: Team's current statistics
            injury_severity: Injury status

        Returns:
            InjuryImpact with adjusted ratings
        """
        team_adj_em = team_stats["AdjEM"]
        team_adj_oe = team_stats["AdjOE"]
        team_adj_de = team_stats["AdjDE"]

        # Value lost based on severity
        if injury_severity == "out":
            value_multiplier = 1.0
        elif injury_severity == "doubtful":
            value_multiplier = 0.75
        elif injury_severity == "questionable":
            value_multiplier = 0.5
        else:
            value_multiplier = 0.0

        # Value over replacement (what we actually lose)
        value_lost = player.value_over_replacement * value_multiplier

        # Adjust team ratings
        adjusted_em = team_adj_em - value_lost
        adjusted_oe = team_adj_oe - (player.offensive_contribution * value_multiplier)

        # Defensive impact is smaller (harder for one player to affect)
        adjusted_de = team_adj_de + (player.defensive_contribution * value_multiplier * 0.5)

        # Confidence interval
        # Uncertainty based on:
        # 1. Player value estimation error (±20%)
        # 2. Replacement player variability (±30%)
        uncertainty = abs(value_lost) * 0.35
        ci_lower = adjusted_em - uncertainty
        ci_upper = adjusted_em + uncertainty

        # Classify severity
        if abs(value_lost) < 1:
            severity = "Minor"
        elif abs(value_lost) < 3:
            severity = "Moderate"
        elif abs(value_lost) < 6:
            severity = "Major"
        else:
            severity = "Devastating"

        return InjuryImpact(
            player=player,
            team_adj_em_baseline=team_adj_em,
            estimated_adj_em_loss=value_lost,
            confidence_interval=(ci_lower, ci_upper),
            adjusted_adj_em=adjusted_em,
            adjusted_adj_oe=adjusted_oe,
            adjusted_adj_de=adjusted_de,
            severity=severity,
        )

    def get_team_depth_chart(
        self,
        team_name: str,
        player_stats_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Create depth chart with player values.

        Args:
            team_name: Team to analyze
            player_stats_df: DataFrame from kenpompy.get_playerstats()

        Returns:
            DataFrame with players ranked by value
        """
        team_players = player_stats_df[
            player_stats_df["Team"] == team_name
        ].copy()

        # Calculate values
        # Note: Requires team stats, simplified here
        team_players["EstimatedValue"] = (
            (team_players["ORtg"] - 110) * (team_players["Poss%"] / 100)
        )

        team_players = team_players.sort_values("EstimatedValue", ascending=False)

        return team_players[
            ["Player", "yr", "ht", "Poss%", "ORtg", "eFG%", "EstimatedValue"]
        ]
```

### Integration with GamePredictor

```python
# Enhance prediction with injury adjustments

def predict_with_injuries(
    team1_stats: dict,
    team2_stats: dict,
    team1_injuries: list[InjuryImpact],
    team2_injuries: list[InjuryImpact],
) -> PredictionResult:
    """Predict game outcome accounting for injuries."""

    # Adjust team stats for injuries
    team1_adjusted = team1_stats.copy()
    team2_adjusted = team2_stats.copy()

    # Apply injury impacts
    for injury in team1_injuries:
        team1_adjusted["AdjEM"] = injury.adjusted_adj_em
        team1_adjusted["AdjOE"] = injury.adjusted_adj_oe
        team1_adjusted["AdjDE"] = injury.adjusted_adj_de

    for injury in team2_injuries:
        team2_adjusted["AdjEM"] = injury.adjusted_adj_em
        team2_adjusted["AdjOE"] = injury.adjusted_adj_oe
        team2_adjusted["AdjDE"] = injury.adjusted_adj_de

    # Run standard prediction with adjusted stats
    predictor = GamePredictor()
    result = predictor.predict_with_confidence(
        team1_adjusted, team2_adjusted
    )

    return result
```

### Data Requirements & Challenges

**Challenge**: Official KenPom API doesn't provide player stats
**Solution**: Use kenpompy library for player data

```python
# Data ingestion workflow

from kenpompy.summary import get_playerstats
from kenpompy.utils import login

# Login to kenpompy
browser = login(email, password)

# Get player stats
player_stats = get_playerstats(browser, year=2025)

# Calculate player values
player_impact_model = PlayerImpactModel()
# ... process player_stats
```

**Limitations**:
1. Player stats require separate kenpompy scraping (slower)
2. Defensive impact is hard to quantify from box score stats
3. Replacement player estimation is uncertain
4. Injuries are not tracked in KenPom (need external source)

### Future Enhancements

1. **Injury Database**: Scrape practice reports (already planned in TIER 1!)
2. **Synergy Integration**: Advanced player metrics if budget allows
3. **Lineup Analysis**: Five-man lineup efficiency (requires play-by-play)
4. **Transfer Portal Impact**: Model impact of roster changes

---

## Implementation Priority & Roadmap

### Priority Order (Recommended)

1. **Tempo/Pace Analysis** (Feature 1)
   - **Why First**: Data already available, low complexity, immediate value
   - **Effort**: 4-6 hours
   - **Impact**: HIGH - directly improves predictions

2. **Conference Power Ratings** (Feature 2)
   - **Why Second**: Builds on tempo work, helps with tournament analysis
   - **Effort**: 3-4 hours
   - **Impact**: MEDIUM - useful for tournament selection

3. **Pairwise Rankings** (Feature 3)
   - **Why Third**: Interesting but lower practical value
   - **Effort**: 6-8 hours
   - **Impact**: MEDIUM - alternative ranking system

4. **Player Impact Modeling** (Feature 4)
   - **Why Last**: Most complex, requires additional data sources
   - **Effort**: 12-16 hours
   - **Impact**: HIGH - but highest complexity

### Sprint Plan

#### Sprint 1: Tempo/Pace Analysis (Week 1)
- [ ] Create `tempo_analysis.py` module
- [ ] Implement `TempoProfile` and `PaceMatchupAnalysis`
- [ ] Add tempo features to `GamePredictor`
- [ ] Write comprehensive tests
- [ ] Add examples and documentation

#### Sprint 2: Conference Power Ratings (Week 1-2)
- [ ] Create `conference_analytics.py` module
- [ ] Implement power rating calculation
- [ ] Add conference comparison tools
- [ ] Write tests
- [ ] Create conference dashboard example

#### Sprint 3: Pairwise Rankings (Week 2-3)
- [ ] Create `pairwise_rankings.py` module
- [ ] Implement Bradley-Terry model
- [ ] Add KenPom comparison tools
- [ ] Write tests
- [ ] Create H.U.M.A.N. poll example

#### Sprint 4: Player Impact (Week 3-5)
- [ ] Create `player_impact.py` module
- [ ] Integrate kenpompy for player stats
- [ ] Implement value estimation model
- [ ] Implement injury impact model
- [ ] Write tests
- [ ] Create injury analysis examples

### Testing Strategy (All Features)

```python
# Comprehensive test coverage for each feature

# Unit tests
- Test individual functions with mock data
- Test edge cases (empty data, invalid inputs)
- Test validation logic

# Integration tests
- Test feature integration with existing modules
- Test API data flow
- Test ML model integration

# Performance tests
- Benchmark calculation times
- Test with large datasets (all 363 teams)

# Regression tests
- Ensure existing tests still pass
- Validate predictions don't degrade
```

### Documentation Requirements

1. **Module Docstrings**: Comprehensive docstrings for all classes/functions
2. **Examples Directory**:
   - `examples/tempo_analysis_demo.py`
   - `examples/conference_rankings_demo.py`
   - `examples/pairwise_poll_demo.py`
   - `examples/injury_impact_demo.py`
3. **Update ANALYTICS_ROADMAP.md**: Mark features as completed
4. **Update README.md**: Add feature documentation
5. **Update CLAUDE.md**: Add usage examples for Claude integration

---

## Success Metrics

### Feature 1: Tempo/Pace Analysis
- ✅ Successfully classify team pace styles (fast/slow/average)
- ✅ Calculate APL-based matchup advantages
- ✅ Improve prediction accuracy by 2-3% (measure via backtesting)

### Feature 2: Conference Power Ratings
- ✅ Generate power ratings for all 32 conferences
- ✅ Correctly identify top 5 conferences (vs expert consensus)
- ✅ Estimate tournament bids within ±2 of actuals

### Feature 3: Pairwise Rankings
- ✅ Bradley-Terry model converges successfully
- ✅ Rankings correlate >0.85 with KenPom rankings
- ✅ Identify interesting ranking disagreements

### Feature 4: Player Impact
- ✅ Calculate player values for top 100 players
- ✅ Injury impact estimates within ±2 points of actual
- ✅ Improve prediction accuracy for games with injuries by 5%+

---

## Questions & Decisions

### Technical Decisions
1. **Tempo Analysis**: Use APL data directly or derive from tempo?
   - **Decision**: Use APL data - it's more granular and already available

2. **Conference Power**: Use weighted or simple averages?
   - **Decision**: Use weighted formula (40% avg EM, 20% top team, 20% depth, etc.)

3. **Pairwise Rankings**: Use Bradley-Terry or Elo?
   - **Decision**: Bradley-Terry matches KenPom's H.U.M.A.N. poll methodology

4. **Player Impact**: Estimate replacement value or assume zero?
   - **Decision**: Estimate replacement at 70% of player value

### Data Decisions
1. **Player Stats**: Use kenpompy scraping or wait for API?
   - **Decision**: Use kenpompy now, refactor when API available

2. **Injury Data**: Scrape practice reports or use external API?
   - **Decision**: Start with manual input, add scraping in Phase 2

### Integration Decisions
1. **Add to GamePredictor or separate modules?**
   - **Decision**: Separate modules, but integrate tempo into GamePredictor features

2. **MCP Server tools or analysis functions?**
   - **Decision**: Both - analysis functions with MCP wrappers

---

## Risks & Mitigation

### Risk 1: Player Data Availability
- **Risk**: Player stats not in official API, requires scraping
- **Mitigation**: Use kenpompy library, plan for future API integration
- **Impact**: MEDIUM

### Risk 2: Injury Data Accuracy
- **Risk**: Injury reports incomplete or inaccurate
- **Mitigation**: Use multiple sources, confidence intervals
- **Impact**: HIGH

### Risk 3: Model Complexity
- **Risk**: Player impact model may be too uncertain
- **Mitigation**: Wide confidence intervals, conservative estimates
- **Impact**: MEDIUM

### Risk 4: API Rate Limits
- **Risk**: Conference analysis requires many API calls
- **Mitigation**: Cache results, batch requests
- **Impact**: LOW

---

## Conclusion

These four features represent significant analytical enhancements to the KenPom Sports Analyzer:

1. **Tempo/Pace Analysis** adds sophisticated game flow insights
2. **Conference Power Ratings** provides conference strength metrics
3. **Pairwise Rankings** offers alternative ranking methodology
4. **Player Impact** enables injury quantification

**Recommended Start**: Begin with Tempo/Pace Analysis for immediate impact, then Conference Power Ratings. Pairwise Rankings and Player Impact can follow based on user demand and available time.

**Total Effort**: 25-34 hours across all four features
**Expected Value**: HIGH - transforms analytical capabilities significantly
**Dependencies**: Minimal - mostly uses existing infrastructure

Ready to implement upon approval.
