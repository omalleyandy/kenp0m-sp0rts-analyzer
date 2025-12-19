"""Tempo and pace matchup analysis for basketball games.

This module provides sophisticated tempo/pace analysis beyond simple tempo averages.
It uses APL (Average Possession Length) data to understand offensive/defensive styles
and predict pace control dynamics in matchups.

Example:
    ```python
    from kenp0m_sp0rts_analyzer.api_client import KenPomAPI
    from kenp0m_sp0rts_analyzer.tempo_analysis import TempoMatchupAnalyzer

    api = KenPomAPI()
    analyzer = TempoMatchupAnalyzer(api)

    # Get team stats
    duke_stats = api.get_team_by_name("Duke", 2025)
    unc_stats = api.get_team_by_name("North Carolina", 2025)

    # Analyze tempo matchup
    analysis = analyzer.analyze_pace_matchup(duke_stats, unc_stats)

    print(f"Expected pace: {analysis.expected_possessions} possessions")
    print(f"Style mismatch: {analysis.style_mismatch_score}/10")
    print(f"Pace control: {analysis.pace_advantage}")
    ```
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

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
    off_style: str  # "quick_strike", "methodical", "average"
    def_style: str  # "pressure", "pack_line", "average"


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

    Key capabilities:
    - Classify team pace styles (fast/slow/methodical/pressure)
    - Calculate control-weighted expected tempo
    - Identify APL mismatches and offensive disruption
    - Estimate tempo impact on game outcomes
    - Adjust prediction confidence for tempo variance
    """

    # Style classification thresholds
    FAST_TEMPO_THRESHOLD = 70.0
    SLOW_TEMPO_THRESHOLD = 66.0
    QUICK_APL_THRESHOLD = 17.0
    SLOW_APL_THRESHOLD = 19.0

    # Tempo control calculation constants
    NATIONAL_AVG_TEMPO = 68.0  # Average possessions per game
    NATIONAL_AVG_APL = 17.5  # Average possession length in seconds
    APL_DEF_BASELINE = 20.0  # Baseline for defensive control calculation
    DEFENSIVE_CONTROL_WEIGHT = 0.40  # Weight for defensive style in pace control
    EFFICIENCY_WEIGHT = 0.30  # Weight for efficiency advantage in pace control
    TEMPO_PREFERENCE_WEIGHT = 0.30  # Weight for tempo preference in pace control
    TEMPO_STRENGTH_DIVISOR = 6.0  # Divisor for tempo preference strength
    DEF_CONTROL_DIVISOR = 5.0  # Divisor for defensive control normalization
    EM_NORMALIZATION = 20.0  # Divisor for efficiency margin normalization

    # Style mismatch scoring weights
    TEMPO_MISMATCH_WEIGHT = 3.0  # Max points for tempo differential
    TEMPO_MISMATCH_DIVISOR = 3.0  # Divisor for tempo score calculation
    APL_OFF_MISMATCH_WEIGHT = 3.0  # Max points for offensive APL mismatch
    APL_OFF_MISMATCH_DIVISOR = 2.0  # Divisor for offensive APL score
    APL_DEF_MISMATCH_WEIGHT = 4.0  # Max points for defensive APL mismatch (highest)
    APL_DEF_MISMATCH_DIVISOR = 1.5  # Divisor for defensive APL score

    # Disruption classification thresholds (seconds)
    SEVERE_DISRUPTION_THRESHOLD = 3.0
    MODERATE_DISRUPTION_THRESHOLD = 1.5

    # Optimal pace calculation constants
    NATIONAL_AVG_OE = 110.0  # National average offensive efficiency
    NATIONAL_AVG_DE = 100.0  # National average defensive efficiency
    PACE_ADJUSTMENT_DIVISOR = 10.0  # Divisor for pace adjustments

    # Confidence adjustment
    IQR_TO_STDDEV_RATIO = 1.35  # Conversion from IQR to standard deviation

    # Expected tempo weighting
    MAX_TEMPO_CONTROL_WEIGHT = 0.25  # Maximum shift from 50/50 weighting

    def __init__(self, api: KenPomAPI | None = None):
        """Initialize tempo analyzer.

        Args:
            api: Optional KenPomAPI instance. Creates one if not provided.
        """
        self.api = api or KenPomAPI()

    def get_tempo_profile(self, team_stats: dict[str, Any]) -> TempoProfile:
        """Extract and classify team's tempo profile.

        Args:
            team_stats: Team statistics from ratings endpoint with fields:
                - TeamName, AdjTempo, RankAdjTempo
                - APL_Off, APL_Def, ConfAPL_Off, ConfAPL_Def

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
        """Classify overall pace style.

        Args:
            tempo: Adjusted tempo (possessions per 40 min)

        Returns:
            "fast", "slow", or "average"
        """
        if tempo > self.FAST_TEMPO_THRESHOLD:
            return "fast"
        elif tempo < self.SLOW_TEMPO_THRESHOLD:
            return "slow"
        else:
            return "average"

    def _classify_offensive_style(self, apl_off: float) -> str:
        """Classify offensive style based on possession length.

        Args:
            apl_off: Average possession length on offense (seconds)

        Returns:
            "quick_strike", "methodical", or "average"
        """
        if apl_off < self.QUICK_APL_THRESHOLD:
            return "quick_strike"
        elif apl_off > self.SLOW_APL_THRESHOLD:
            return "methodical"
        else:
            return "average"

    def _classify_defensive_style(self, apl_def: float) -> str:
        """Classify defensive style based on possession length allowed.

        Args:
            apl_def: Average possession length on defense (seconds)

        Returns:
            "pressure", "pack_line", or "average"
        """
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

        Analyzes:
        1. Team tempo profiles and style classifications
        2. Tempo control dynamics (who dictates pace)
        3. Expected game tempo (control-weighted)
        4. Style mismatches and offensive disruption
        5. Tempo impact on predicted margin
        6. Confidence interval adjustments for variance

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
        control_factor = self.calculate_tempo_control(team1_stats, team2_stats)

        # Calculate expected possessions (control-weighted)
        expected_poss = self._calculate_expected_tempo(
            team1_stats, team2_stats, control_factor
        )

        # Style mismatch score
        style_mismatch = self._calculate_style_mismatch(team1_profile, team2_profile)

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

    def calculate_tempo_control(self, team1_stats: dict, team2_stats: dict) -> float:
        """Calculate tempo control factor (-1 to +1).

        Factors:
        1. Defensive style (40% weight) - Defense controls pace more than offense
        2. Efficiency advantage (30% weight) - Better team dictates pace
        3. Tempo preference strength (30% weight) - Extreme tempo teams have more control

        Args:
            team1_stats: Team 1 statistics
            team2_stats: Team 2 statistics

        Returns:
            Control factor from -1 (team2 controls) to +1 (team1 controls)
        """
        # Defensive control (40% weight)
        # Lower APL_Def = more defensive pressure = more pace control
        team1_def_control = self.APL_DEF_BASELINE - team1_stats["APL_Def"]
        team2_def_control = self.APL_DEF_BASELINE - team2_stats["APL_Def"]
        def_factor = (team1_def_control - team2_def_control) / self.DEF_CONTROL_DIVISOR

        # Efficiency advantage (30% weight)
        em_diff = team1_stats["AdjEM"] - team2_stats["AdjEM"]
        em_factor = np.clip(em_diff / self.EM_NORMALIZATION, -1.0, 1.0)

        # Tempo preference strength (30% weight)
        team1_strength = (
            abs(team1_stats["AdjTempo"] - self.NATIONAL_AVG_TEMPO)
            / self.TEMPO_STRENGTH_DIVISOR
        )
        team2_strength = (
            abs(team2_stats["AdjTempo"] - self.NATIONAL_AVG_TEMPO)
            / self.TEMPO_STRENGTH_DIVISOR
        )

        if team1_stats["AdjTempo"] > team2_stats["AdjTempo"]:
            tempo_factor = team1_strength - team2_strength
        else:
            tempo_factor = -(team2_strength - team1_strength)

        # Weighted combination
        control = (
            self.DEFENSIVE_CONTROL_WEIGHT * def_factor
            + self.EFFICIENCY_WEIGHT * em_factor
            + self.TEMPO_PREFERENCE_WEIGHT * tempo_factor
        )

        return float(np.clip(control, -1.0, 1.0))

    def _calculate_expected_tempo(
        self,
        team1_stats: dict,
        team2_stats: dict,
        control_factor: float,
    ) -> float:
        """Calculate expected game tempo using control weighting.

        Instead of simple average, weights toward team with more pace control.

        Args:
            team1_stats: Team 1 statistics
            team2_stats: Team 2 statistics
            control_factor: Tempo control factor (-1 to +1)

        Returns:
            Expected game tempo (possessions)
        """
        team1_tempo = team1_stats["AdjTempo"]
        team2_tempo = team2_stats["AdjTempo"]

        # Convert control (-1 to +1) to weights
        # control = +1 → 75% weight to team1
        # control = 0  → 50% weight each (simple average)
        # control = -1 → 75% weight to team2
        team1_weight = 0.5 + (control_factor * self.MAX_TEMPO_CONTROL_WEIGHT)
        team2_weight = 1.0 - team1_weight

        expected = (team1_tempo * team1_weight) + (team2_tempo * team2_weight)

        return round(expected, 1)

    def _calculate_style_mismatch(
        self, profile1: TempoProfile, profile2: TempoProfile
    ) -> float:
        """Calculate style mismatch score (0-10).

        Higher scores indicate greater style contrast:
        - Tempo differential (0-3 points)
        - APL offensive mismatch (0-3 points)
        - APL defensive mismatch (0-4 points)

        Args:
            profile1: Team 1 tempo profile
            profile2: Team 2 tempo profile

        Returns:
            Mismatch score from 0 (similar styles) to 10 (extreme mismatch)
        """
        # Tempo differential (0-3 points)
        tempo_diff = abs(profile1.adj_tempo - profile2.adj_tempo)
        tempo_score = min(
            self.TEMPO_MISMATCH_WEIGHT, tempo_diff / self.TEMPO_MISMATCH_DIVISOR
        )

        # APL offensive mismatch (0-3 points)
        apl_off_diff = abs(profile1.apl_off - profile2.apl_off)
        apl_off_score = min(
            self.APL_OFF_MISMATCH_WEIGHT, apl_off_diff / self.APL_OFF_MISMATCH_DIVISOR
        )

        # APL defensive mismatch (0-4 points)
        apl_def_diff = abs(profile1.apl_def - profile2.apl_def)
        apl_def_score = min(
            self.APL_DEF_MISMATCH_WEIGHT, apl_def_diff / self.APL_DEF_MISMATCH_DIVISOR
        )

        total = tempo_score + apl_off_score + apl_def_score
        return round(total, 1)

    def _classify_disruption(self, apl_mismatch: float) -> str:
        """Classify offensive disruption severity.

        Args:
            apl_mismatch: Team's APL_Off - Opponent's APL_Def (seconds)

        Returns:
            "severe", "moderate", or "minimal"
        """
        abs_mismatch = abs(apl_mismatch)

        if abs_mismatch > self.SEVERE_DISRUPTION_THRESHOLD:
            return "severe"
        elif abs_mismatch > self.MODERATE_DISRUPTION_THRESHOLD:
            return "moderate"
        else:
            return "minimal"

    def _determine_pace_advantage(
        self,
        team1_stats: dict,
        team2_stats: dict,
        control_factor: float,
    ) -> str:
        """Determine which team has pace advantage.

        Args:
            team1_stats: Team 1 statistics
            team2_stats: Team 2 statistics
            control_factor: Tempo control factor

        Returns:
            "team1", "team2", or "neutral"
        """
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

        Calculates how deviation from simple average tempo affects
        the expected margin based on efficiency advantages.

        Args:
            team1_stats: Team 1 statistics
            team2_stats: Team 2 statistics
            expected_tempo: Expected game tempo

        Returns:
            Estimated point impact (positive favors team1)
        """
        # Efficiency advantage per 100 possessions
        em_diff = team1_stats["AdjEM"] - team2_stats["AdjEM"]

        # Simple average tempo
        simple_avg = (team1_stats["AdjTempo"] + team2_stats["AdjTempo"]) / 2

        # Deviation from simple average
        tempo_deviation = expected_tempo - simple_avg

        # Impact: tempo deviation amplifies efficiency advantage
        pace_impact = tempo_deviation * (em_diff / 100.0)

        return round(pace_impact, 2)

    def _calculate_confidence_adjustment(self, expected_tempo: float) -> float:
        """Calculate variance multiplier for confidence intervals.

        Lower tempo games have higher variance (fewer possessions = more randomness).

        Args:
            expected_tempo: Expected game tempo

        Returns:
            Variance multiplier (>1 means wider CI, <1 means narrower CI)
        """
        # National average ~68 possessions
        tempo_factor = self.NATIONAL_AVG_TEMPO / expected_tempo
        return round(tempo_factor, 3)

    def _calculate_optimal_pace(self, team_stats: dict) -> float:
        """Calculate optimal game tempo for team.

        Considers:
        - Team's natural tempo preference
        - Offensive efficiency (higher = prefer more possessions)
        - Defensive efficiency (better = can handle faster pace)

        Args:
            team_stats: Team statistics

        Returns:
            Optimal game tempo (possessions)
        """
        base_tempo = team_stats["AdjTempo"]
        adj_oe = team_stats["AdjOE"]
        adj_de = team_stats["AdjDE"]

        # Offensive adjustment: great offense prefers more possessions
        oe_adjustment = (adj_oe - self.NATIONAL_AVG_OE) / self.PACE_ADJUSTMENT_DIVISOR

        # Defensive adjustment: great defense can handle faster pace
        de_adjustment = (
            -(adj_de - self.NATIONAL_AVG_DE) / self.PACE_ADJUSTMENT_DIVISOR
        )

        optimal = base_tempo + oe_adjustment + de_adjustment
        return round(optimal, 1)

    def _determine_tempo_beneficiary(
        self,
        team1_stats: dict,
        team2_stats: dict,
        pace_type: str,
    ) -> str:
        """Determine which team benefits from given pace.

        Args:
            team1_stats: Team 1 statistics
            team2_stats: Team 2 statistics
            pace_type: "fast" or "slow"

        Returns:
            "team1", "team2", or "neutral"
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
            # Slow pace: favors underdog limiting possessions
            if team1_em < team2_em:
                return "team1"
            elif team2_em < team1_em:
                return "team2"
            else:
                return "neutral"

    def compare_conference_tempos(self, year: int) -> pd.DataFrame:
        """Compare average tempo across all conferences.

        Args:
            year: Season year

        Returns:
            DataFrame with conference tempo statistics sorted by average tempo
        """
        conferences = self.api.get_conferences(year=year)

        results = []
        for conf in conferences.data:
            conf_teams = self.api.get_ratings(year=year, conference=conf["ConfShort"])

            if len(conf_teams.data) == 0:
                continue

            tempos = [team["AdjTempo"] for team in conf_teams.data]
            apl_offs = [team["APL_Off"] for team in conf_teams.data]
            apl_defs = [team["APL_Def"] for team in conf_teams.data]

            results.append(
                {
                    "Conference": conf["ConfLong"],
                    "ConfShort": conf["ConfShort"],
                    "AvgTempo": round(sum(tempos) / len(tempos), 2),
                    "MaxTempo": max(tempos),
                    "MinTempo": min(tempos),
                    "AvgAPL_Off": round(sum(apl_offs) / len(apl_offs), 2),
                    "AvgAPL_Def": round(sum(apl_defs) / len(apl_defs), 2),
                    "NumTeams": len(conf_teams.data),
                }
            )

        df = pd.DataFrame(results)
        return df.sort_values("AvgTempo", ascending=False)
