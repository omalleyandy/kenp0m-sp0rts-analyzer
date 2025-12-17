"""Four Factors matchup analysis for basketball games.

This module provides detailed Four Factors (Dean Oliver) analysis for head-to-head
matchups, identifying offensive/defensive advantages and strategic edges.

The Four Factors:
1. eFG% - Effective Field Goal Percentage (most important)
2. TO% - Turnover Percentage
3. OR% - Offensive Rebound Percentage
4. FT Rate - Free Throw Rate (least important)

Example:
    ```python
    from kenp0m_sp0rts_analyzer.api_client import KenPomAPI
    from kenp0m_sp0rts_analyzer.four_factors_matchup import FourFactorsMatchup

    api = KenPomAPI()
    analyzer = FourFactorsMatchup(api)

    # Analyze Duke vs UNC matchup
    analysis = analyzer.analyze_matchup("Duke", "North Carolina", 2025)

    print(f"Key factor: {analysis.most_important_factor}")
    print(f"Advantage: {analysis.overall_advantage}")

    for insight in analysis.strategic_insights:
        print(f"- {insight}")
    ```
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from .api_client import KenPomAPI
from .utils import normalize_team_name


@dataclass
class FactorMatchup:
    """Analysis of a single Four Factor matchup.

    Compares team's offensive capability vs opponent's defensive capability
    for one of the four factors.

    Attributes:
        factor_name: Name of the factor (eFG%, TO%, OR%, FT Rate)
        team1_offense: Team 1's offensive rating for this factor
        team2_defense: Team 2's defensive rating for this factor
        team2_offense: Team 2's offensive rating for this factor
        team1_defense: Team 1's defensive rating for this factor
        team1_advantage: Net advantage for team 1 (positive = advantage)
        team2_advantage: Net advantage for team 2 (positive = advantage)
        advantage_magnitude: Absolute advantage (0-100 scale)
        advantage_classification: "massive", "significant", "moderate", "minimal", "neutral"
        predicted_winner: Which team has advantage in this factor
        importance_weight: Relative importance (eFG% > TO% > OR% > FT Rate)
    """

    factor_name: str
    team1_offense: float
    team2_defense: float
    team2_offense: float
    team1_defense: float
    team1_advantage: float
    team2_advantage: float
    advantage_magnitude: float
    advantage_classification: str
    predicted_winner: str  # "team1", "team2", "neutral"
    importance_weight: float


@dataclass
class FourFactorsAnalysis:
    """Complete Four Factors matchup analysis.

    Provides comprehensive breakdown of all four factors with strategic
    insights and overall assessment.
    """

    # Team information
    team1_name: str
    team2_name: str
    season: int

    # Individual factor matchups
    efg_matchup: FactorMatchup
    to_matchup: FactorMatchup
    or_matchup: FactorMatchup
    ft_matchup: FactorMatchup

    # Overall assessment
    overall_advantage: str  # "team1", "team2", "neutral"
    advantage_score: float  # Weighted advantage (-10 to +10, positive favors team1)
    most_important_factor: str  # Which factor matters most in this matchup

    # Strategic insights
    strategic_insights: list[str]
    key_matchup_battles: list[str]

    # Efficiency context
    team1_adj_oe: float
    team1_adj_de: float
    team2_adj_oe: float
    team2_adj_de: float


class FourFactorsMatchup:
    """Analyze Four Factors matchups between teams.

    This class provides detailed Four Factors analysis for basketball matchups,
    identifying specific offensive/defensive advantages and strategic edges.

    The Four Factors importance weights (Dean Oliver):
    - eFG%: 40% (most important - shooting efficiency)
    - TO%: 25% (ball security)
    - OR%: 20% (rebounding)
    - FT Rate: 15% (least important - free throws)

    Key capabilities:
    - Compare offensive vs defensive capabilities for each factor
    - Identify strategic advantages and key battles
    - Rank factors by impact for specific matchup
    - Provide actionable scouting insights
    """

    # Dean Oliver's Four Factors importance weights
    EFG_WEIGHT = 0.40  # Shooting is most important
    TO_WEIGHT = 0.25   # Turnovers second
    OR_WEIGHT = 0.20   # Rebounding third
    FT_WEIGHT = 0.15   # Free throws least important

    # Advantage classification thresholds (percentage points)
    MASSIVE_THRESHOLD = 8.0
    SIGNIFICANT_THRESHOLD = 5.0
    MODERATE_THRESHOLD = 2.5
    MINIMAL_THRESHOLD = 1.0

    def __init__(self, api: KenPomAPI | None = None):
        """Initialize Four Factors analyzer.

        Args:
            api: Optional KenPomAPI instance. Creates one if not provided.
        """
        self.api = api or KenPomAPI()

    def analyze_matchup(
        self,
        team1: str,
        team2: str,
        season: int,
        conf_only: bool = False
    ) -> FourFactorsAnalysis:
        """Comprehensive Four Factors matchup analysis.

        Analyzes all four factors to identify strategic advantages:
        1. eFG%: Shooting efficiency battle
        2. TO%: Ball security vs pressure
        3. OR%: Offensive rebounding battle
        4. FT Rate: Drawing fouls vs avoiding fouls

        Args:
            team1: First team name
            team2: Second team name
            season: Season year (e.g., 2025 for 2024-25 season)
            conf_only: Whether to use conference-only stats

        Returns:
            FourFactorsAnalysis with detailed matchup breakdown

        Raises:
            ValueError: If team not found
        """
        # Normalize team names for consistent lookups
        team1 = normalize_team_name(team1)
        team2 = normalize_team_name(team2)

        # Get Four Factors data
        four_factors = self.api.get_four_factors(year=season, conf_only=conf_only)
        df = four_factors.to_dataframe()

        # Find teams
        team1_data = df[df['TeamName'] == team1]
        team2_data = df[df['TeamName'] == team2]

        if team1_data.empty:
            raise ValueError(f"Team '{team1}' not found in {season} season")
        if team2_data.empty:
            raise ValueError(f"Team '{team2}' not found in {season} season")

        team1_stats = team1_data.iloc[0].to_dict()
        team2_stats = team2_data.iloc[0].to_dict()

        # Analyze each factor
        efg = self._analyze_efg_matchup(team1_stats, team2_stats, team1, team2)
        to = self._analyze_to_matchup(team1_stats, team2_stats, team1, team2)
        or_factor = self._analyze_or_matchup(team1_stats, team2_stats, team1, team2)
        ft = self._analyze_ft_matchup(team1_stats, team2_stats, team1, team2)

        # Calculate overall advantage
        advantage_score = self._calculate_overall_advantage(efg, to, or_factor, ft)

        # Determine overall winner
        if advantage_score > 2.0:
            overall_advantage = "team1"
        elif advantage_score < -2.0:
            overall_advantage = "team2"
        else:
            overall_advantage = "neutral"

        # Identify most important factor for this matchup
        most_important = self._identify_key_factor([efg, to, or_factor, ft])

        # Generate strategic insights
        insights = self._generate_strategic_insights(
            efg, to, or_factor, ft, team1, team2
        )

        # Identify key battles
        battles = self._identify_key_battles(efg, to, or_factor, ft, team1, team2)

        return FourFactorsAnalysis(
            team1_name=team1,
            team2_name=team2,
            season=season,
            efg_matchup=efg,
            to_matchup=to,
            or_matchup=or_factor,
            ft_matchup=ft,
            overall_advantage=overall_advantage,
            advantage_score=round(advantage_score, 2),
            most_important_factor=most_important,
            strategic_insights=insights,
            key_matchup_battles=battles,
            team1_adj_oe=team1_stats['AdjOE'],
            team1_adj_de=team1_stats['AdjDE'],
            team2_adj_oe=team2_stats['AdjOE'],
            team2_adj_de=team2_stats['AdjDE'],
        )

    def _analyze_efg_matchup(
        self,
        team1_stats: dict[str, Any],
        team2_stats: dict[str, Any],
        team1_name: str,
        team2_name: str
    ) -> FactorMatchup:
        """Analyze eFG% matchup (shooting efficiency)."""
        # Team 1 shooting vs Team 2 defense
        team1_off = team1_stats['eFG_Pct']
        team2_def = team2_stats['DeFG_Pct']  # Lower is better for defense

        # Team 2 shooting vs Team 1 defense
        team2_off = team2_stats['eFG_Pct']
        team1_def = team1_stats['DeFG_Pct']

        # Calculate advantages
        # Team 1 advantage = how much better they shoot than opponent allows
        team1_advantage = team1_off - team2_def
        team2_advantage = team2_off - team1_def

        # Net advantage (positive = team1 has edge)
        net_advantage = team1_advantage - team2_advantage
        magnitude = abs(net_advantage)

        # Classify advantage
        classification = self._classify_advantage(magnitude)

        # Determine winner
        if net_advantage > 1.0:
            winner = "team1"
        elif net_advantage < -1.0:
            winner = "team2"
        else:
            winner = "neutral"

        return FactorMatchup(
            factor_name="eFG%",
            team1_offense=round(team1_off, 1),
            team2_defense=round(team2_def, 1),
            team2_offense=round(team2_off, 1),
            team1_defense=round(team1_def, 1),
            team1_advantage=round(team1_advantage, 2),
            team2_advantage=round(team2_advantage, 2),
            advantage_magnitude=round(magnitude, 2),
            advantage_classification=classification,
            predicted_winner=winner,
            importance_weight=self.EFG_WEIGHT
        )

    def _analyze_to_matchup(
        self,
        team1_stats: dict[str, Any],
        team2_stats: dict[str, Any],
        team1_name: str,
        team2_name: str
    ) -> FactorMatchup:
        """Analyze TO% matchup (ball security vs pressure)."""
        # Lower TO% is better for offense, higher DTO% is better for defense
        team1_off = team1_stats['TO_Pct']  # Lower is better
        team2_def = team2_stats['DTO_Pct']  # Higher is better (forces TOs)

        team2_off = team2_stats['TO_Pct']
        team1_def = team1_stats['DTO_Pct']

        # Team 1 advantage = they protect ball better than opponent forces TOs
        # Inverse because lower TO% is better
        team1_advantage = team2_def - team1_off  # Team 2 forces TOs vs Team 1 gives them up
        team2_advantage = team1_def - team2_off

        net_advantage = team1_advantage - team2_advantage
        magnitude = abs(net_advantage)

        classification = self._classify_advantage(magnitude)

        if net_advantage > 1.0:
            winner = "team1"
        elif net_advantage < -1.0:
            winner = "team2"
        else:
            winner = "neutral"

        return FactorMatchup(
            factor_name="TO%",
            team1_offense=round(team1_off, 1),
            team2_defense=round(team2_def, 1),
            team2_offense=round(team2_off, 1),
            team1_defense=round(team1_def, 1),
            team1_advantage=round(team1_advantage, 2),
            team2_advantage=round(team2_advantage, 2),
            advantage_magnitude=round(magnitude, 2),
            advantage_classification=classification,
            predicted_winner=winner,
            importance_weight=self.TO_WEIGHT
        )

    def _analyze_or_matchup(
        self,
        team1_stats: dict[str, Any],
        team2_stats: dict[str, Any],
        team1_name: str,
        team2_name: str
    ) -> FactorMatchup:
        """Analyze OR% matchup (offensive rebounding battle)."""
        # Higher OR% is better for offense
        # Higher DOR% is better for defense (defensive rebounding)
        team1_off = team1_stats['OR_Pct']
        team2_def = team2_stats['DOR_Pct']

        team2_off = team2_stats['OR_Pct']
        team1_def = team1_stats['DOR_Pct']

        # Team 1 advantage = their offensive rebounding vs opponent's defensive rebounding
        # Since we want to compare offensive boards vs defensive boards, we invert defense
        team1_advantage = team1_off - (100 - team2_def)
        team2_advantage = team2_off - (100 - team1_def)

        net_advantage = team1_advantage - team2_advantage
        magnitude = abs(net_advantage)

        classification = self._classify_advantage(magnitude)

        if net_advantage > 1.0:
            winner = "team1"
        elif net_advantage < -1.0:
            winner = "team2"
        else:
            winner = "neutral"

        return FactorMatchup(
            factor_name="OR%",
            team1_offense=round(team1_off, 1),
            team2_defense=round(team2_def, 1),
            team2_offense=round(team2_off, 1),
            team1_defense=round(team1_def, 1),
            team1_advantage=round(team1_advantage, 2),
            team2_advantage=round(team2_advantage, 2),
            advantage_magnitude=round(magnitude, 2),
            advantage_classification=classification,
            predicted_winner=winner,
            importance_weight=self.OR_WEIGHT
        )

    def _analyze_ft_matchup(
        self,
        team1_stats: dict[str, Any],
        team2_stats: dict[str, Any],
        team1_name: str,
        team2_name: str
    ) -> FactorMatchup:
        """Analyze FT Rate matchup (drawing fouls vs avoiding fouls)."""
        # Higher FT_Rate is better for offense (getting to the line)
        # Lower DFT_Rate is better for defense (not fouling)
        team1_off = team1_stats['FT_Rate']
        team2_def = team2_stats['DFT_Rate']  # Lower is better for defense

        team2_off = team2_stats['FT_Rate']
        team1_def = team1_stats['DFT_Rate']

        # Team 1 advantage = they draw fouls better than opponent avoids fouls
        team1_advantage = team1_off - team2_def
        team2_advantage = team2_off - team1_def

        net_advantage = team1_advantage - team2_advantage
        magnitude = abs(net_advantage)

        classification = self._classify_advantage(magnitude)

        if net_advantage > 2.0:  # FT Rate has wider variance
            winner = "team1"
        elif net_advantage < -2.0:
            winner = "team2"
        else:
            winner = "neutral"

        return FactorMatchup(
            factor_name="FT Rate",
            team1_offense=round(team1_off, 1),
            team2_defense=round(team2_def, 1),
            team2_offense=round(team2_off, 1),
            team1_defense=round(team1_def, 1),
            team1_advantage=round(team1_advantage, 2),
            team2_advantage=round(team2_advantage, 2),
            advantage_magnitude=round(magnitude, 2),
            advantage_classification=classification,
            predicted_winner=winner,
            importance_weight=self.FT_WEIGHT
        )

    def _classify_advantage(self, magnitude: float) -> str:
        """Classify advantage magnitude."""
        if magnitude >= self.MASSIVE_THRESHOLD:
            return "massive"
        elif magnitude >= self.SIGNIFICANT_THRESHOLD:
            return "significant"
        elif magnitude >= self.MODERATE_THRESHOLD:
            return "moderate"
        elif magnitude >= self.MINIMAL_THRESHOLD:
            return "minimal"
        else:
            return "neutral"

    def _calculate_overall_advantage(
        self,
        efg: FactorMatchup,
        to: FactorMatchup,
        or_factor: FactorMatchup,
        ft: FactorMatchup
    ) -> float:
        """Calculate weighted overall advantage score.

        Returns:
            Score from -10 to +10 (positive favors team1)
        """
        # Weight each factor by importance
        score = (
            efg.team1_advantage * self.EFG_WEIGHT +
            to.team1_advantage * self.TO_WEIGHT +
            or_factor.team1_advantage * self.OR_WEIGHT +
            ft.team1_advantage * self.FT_WEIGHT
        ) - (
            efg.team2_advantage * self.EFG_WEIGHT +
            to.team2_advantage * self.TO_WEIGHT +
            or_factor.team2_advantage * self.OR_WEIGHT +
            ft.team2_advantage * self.FT_WEIGHT
        )

        return score

    def _identify_key_factor(self, factors: list[FactorMatchup]) -> str:
        """Identify which factor is most important for this matchup."""
        # Score each factor: magnitude × importance weight
        scores = [
            (f.factor_name, f.advantage_magnitude * f.importance_weight)
            for f in factors
        ]

        # Return factor with highest weighted magnitude
        key_factor = max(scores, key=lambda x: x[1])
        return key_factor[0]

    def _generate_strategic_insights(
        self,
        efg: FactorMatchup,
        to: FactorMatchup,
        or_factor: FactorMatchup,
        ft: FactorMatchup,
        team1: str,
        team2: str
    ) -> list[str]:
        """Generate strategic insights from Four Factors analysis."""
        insights = []

        # eFG% insights (most important)
        if efg.advantage_classification in ["massive", "significant"]:
            winner = team1 if efg.predicted_winner == "team1" else team2
            loser = team2 if winner == team1 else team1
            insights.append(
                f"SHOOTING ADVANTAGE: {winner} has {efg.advantage_classification} "
                f"eFG% edge ({abs(efg.team1_advantage - efg.team2_advantage):.1f}% net). "
                f"This is the most important factor."
            )

        # TO% insights
        if to.advantage_classification in ["massive", "significant"]:
            if to.predicted_winner == "team1":
                insights.append(
                    f"TURNOVER BATTLE: {team1} protects ball better ({team1}:{to.team1_offense:.1f}% TO) "
                    f"vs {team2}'s pressure ({team2}:{to.team2_defense:.1f}% DTO). "
                    f"Ball security could be decisive."
                )
            elif to.predicted_winner == "team2":
                insights.append(
                    f"TURNOVER BATTLE: {team2} protects ball better ({team2}:{to.team2_offense:.1f}% TO) "
                    f"vs {team1}'s pressure ({team1}:{to.team1_defense:.1f}% DTO). "
                    f"Ball security could be decisive."
                )

        # OR% insights
        if or_factor.advantage_classification in ["massive", "significant"]:
            winner = team1 if or_factor.predicted_winner == "team1" else team2
            insights.append(
                f"REBOUNDING EDGE: {winner} has {or_factor.advantage_classification} "
                f"offensive rebounding advantage. Second-chance points critical."
            )

        # FT Rate insights
        if ft.advantage_classification in ["massive", "significant"]:
            if ft.predicted_winner == "team1":
                insights.append(
                    f"FREE THROW BATTLE: {team1} draws fouls well ({team1}:{ft.team1_offense:.1f} FT Rate) "
                    f"vs {team2}'s foul-prone defense ({team2}:{ft.team2_defense:.1f} DFT Rate). "
                    f"Getting to the line will be key."
                )
            elif ft.predicted_winner == "team2":
                insights.append(
                    f"FREE THROW BATTLE: {team2} draws fouls well ({team2}:{ft.team2_offense:.1f} FT Rate) "
                    f"vs {team1}'s foul-prone defense ({team1}:{ft.team1_defense:.1f} DFT Rate). "
                    f"Getting to the line will be key."
                )

        # Overall assessment
        factors = [efg, to, or_factor, ft]
        team1_wins = sum(1 for f in factors if f.predicted_winner == "team1")
        team2_wins = sum(1 for f in factors if f.predicted_winner == "team2")

        if team1_wins >= 3:
            insights.append(f"OVERALL: {team1} wins {team1_wins}/4 factor battles. Strong statistical edge.")
        elif team2_wins >= 3:
            insights.append(f"OVERALL: {team2} wins {team2_wins}/4 factor battles. Strong statistical edge.")
        else:
            insights.append("OVERALL: Balanced matchup. No team dominates all four factors.")

        return insights

    def _identify_key_battles(
        self,
        efg: FactorMatchup,
        to: FactorMatchup,
        or_factor: FactorMatchup,
        ft: FactorMatchup,
        team1: str,
        team2: str
    ) -> list[str]:
        """Identify key matchup battles to watch."""
        battles = []

        # eFG% battle
        if efg.predicted_winner != "neutral":
            winner = team1 if efg.predicted_winner == "team1" else team2
            loser = team2 if winner == team1 else team1
            battles.append(
                f"Can {loser} slow down {winner}'s elite shooting? "
                f"({winner}:{efg.team1_offense if winner==team1 else efg.team2_offense:.1f}% eFG)"
            )

        # TO% battle
        if to.predicted_winner != "neutral":
            battles.append(
                f"Turnover battle: Ball protection vs defensive pressure will determine possessions"
            )

        # OR% battle
        if or_factor.predicted_winner != "neutral":
            battles.append(
                f"Glass battle: Second-chance points could swing close game"
            )

        # FT Rate battle
        if ft.predicted_winner != "neutral":
            battles.append(
                f"Foul game: Free throws and bonus situations will matter late"
            )

        return battles
