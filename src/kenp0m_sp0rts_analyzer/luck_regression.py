"""Luck Regression Analysis Module.

This module implements luck regression analysis to identify overvalued/undervalued
teams based on their performance in close games.

Based on research showing that close game performance (Luck) always regresses to the mean,
providing 2-5 point betting edges when exploited correctly.

Author: Andy O'Malley & Claude
Date: 2025-12-18
"""

from dataclasses import dataclass
from typing import Literal


@dataclass
class LuckAnalysis:
    """Results of luck regression analysis for a team."""

    team_name: str
    current_adjEM: float
    luck_factor: float
    luck_category: Literal["very_lucky", "lucky", "neutral", "unlucky", "very_unlucky"]
    luck_adjusted_adjEM: float
    regression_adjustment: float
    games_remaining: int
    betting_recommendation: Literal["FADE", "BACK", "NEUTRAL"]
    edge_magnitude: float
    confidence: Literal["high", "medium", "low"]
    reasoning: str

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "team_name": self.team_name,
            "current_adjEM": self.current_adjEM,
            "luck_factor": self.luck_factor,
            "luck_category": self.luck_category,
            "luck_adjusted_adjEM": self.luck_adjusted_adjEM,
            "regression_adjustment": self.regression_adjustment,
            "games_remaining": self.games_remaining,
            "betting_recommendation": self.betting_recommendation,
            "edge_magnitude": self.edge_magnitude,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
        }


@dataclass
class MatchupLuckEdge:
    """Luck-based edge analysis for a matchup."""

    team1_name: str
    team2_name: str
    team1_luck: float
    team2_luck: float
    raw_margin: float
    luck_adjusted_margin: float
    luck_edge: float
    combined_luck_impact: float
    betting_recommendation: str
    confidence: Literal["high", "medium", "low"]
    expected_clv: float
    reasoning: str

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "team1_name": self.team1_name,
            "team2_name": self.team2_name,
            "team1_luck": self.team1_luck,
            "team2_luck": self.team2_luck,
            "raw_margin": self.raw_margin,
            "luck_adjusted_margin": self.luck_adjusted_margin,
            "luck_edge": self.luck_edge,
            "combined_luck_impact": self.combined_luck_impact,
            "betting_recommendation": self.betting_recommendation,
            "confidence": self.confidence,
            "expected_clv": self.expected_clv,
            "reasoning": self.reasoning,
        }


class LuckRegressionAnalyzer:
    """Analyzes teams for luck regression opportunities.

    This analyzer identifies teams that are overperforming (lucky) or underperforming
    (unlucky) based on their close game record, and calculates expected regression.

    Example:
        ```python
        analyzer = LuckRegressionAnalyzer()

        # Analyze single team
        duke_analysis = analyzer.analyze_team_luck(
            team_name="Duke",
            adjEM=24.5,
            luck=0.18,
            games_remaining=15
        )

        # Analyze matchup
        edge = analyzer.analyze_matchup_luck(
            team1_name="Duke",
            team1_adjEM=24.5,
            team1_luck=0.18,
            team2_name="UNC",
            team2_adjEM=22.0,
            team2_luck=-0.12,
            games_remaining=15
        )
        ```
    """

    # Luck category thresholds
    VERY_LUCKY_THRESHOLD = 0.15
    LUCKY_THRESHOLD = 0.08
    UNLUCKY_THRESHOLD = -0.08
    VERY_UNLUCKY_THRESHOLD = -0.15

    # Regression parameters
    REGRESSION_RATE = 0.50  # 50% of luck regresses over remaining games
    LUCK_TO_POINTS_MULTIPLIER = 10.0  # Luck converts to points per game

    # Edge thresholds for betting recommendations
    STRONG_EDGE_THRESHOLD = 2.5  # 2.5+ points = strong recommendation
    MEDIUM_EDGE_THRESHOLD = 1.5  # 1.5-2.5 points = medium recommendation

    def categorize_luck(self, luck: float) -> str:
        """Categorize luck factor.

        Args:
            luck: Luck factor from KenPom

        Returns:
            Category string: "very_lucky", "lucky", "neutral", "unlucky", "very_unlucky"
        """
        if luck >= self.VERY_LUCKY_THRESHOLD:
            return "very_lucky"
        elif luck >= self.LUCKY_THRESHOLD:
            return "lucky"
        elif luck <= self.VERY_UNLUCKY_THRESHOLD:
            return "very_unlucky"
        elif luck <= self.UNLUCKY_THRESHOLD:
            return "unlucky"
        else:
            return "neutral"

    def calculate_regression_adjustment(
        self,
        luck: float,
        games_remaining: int,
        season_total: int = 30,
    ) -> float:
        """Calculate expected luck regression in AdjEM points.

        Formula:
            adjustment = luck × luck_to_points × regression_rate × (games_remaining / season_total)

        Args:
            luck: Current luck factor
            games_remaining: Games remaining in season
            season_total: Total games in season (default: 30)

        Returns:
            Expected regression in AdjEM points
        """
        # Convert luck to points per game
        luck_points = luck * self.LUCK_TO_POINTS_MULTIPLIER

        # Calculate regression weight based on games remaining
        regression_weight = games_remaining / season_total

        # Expected regression
        adjustment = luck_points * self.REGRESSION_RATE * regression_weight

        return adjustment

    def analyze_team_luck(
        self,
        team_name: str,
        adjEM: float,
        luck: float,
        games_remaining: int,
        season_total: int = 30,
    ) -> LuckAnalysis:
        """Analyze luck for a single team.

        Args:
            team_name: Team name
            adjEM: Current Adjusted Efficiency Margin
            luck: Current luck factor
            games_remaining: Games remaining in season
            season_total: Total games in season

        Returns:
            LuckAnalysis object with complete analysis
        """
        # Categorize luck
        category = self.categorize_luck(luck)

        # Calculate regression adjustment
        regression_adjustment = self.calculate_regression_adjustment(
            luck, games_remaining, season_total
        )

        # Adjust AdjEM
        luck_adjusted_adjEM = adjEM - regression_adjustment

        # Determine betting recommendation
        edge_magnitude = abs(regression_adjustment)

        if category in ["very_lucky", "lucky"]:
            recommendation = "FADE"
            reasoning = (
                f"{team_name} is {category.replace('_', ' ')} (Luck = {luck:+.3f}). "
                f"Expected regression: {regression_adjustment:+.1f} points over next {games_remaining} games. "
                f"Team is overvalued by ~{edge_magnitude:.1f} points."
            )
        elif category in ["very_unlucky", "unlucky"]:
            recommendation = "BACK"
            reasoning = (
                f"{team_name} is {category.replace('_', ' ')} (Luck = {luck:+.3f}). "
                f"Expected improvement: {abs(regression_adjustment):+.1f} points over next {games_remaining} games. "
                f"Team is undervalued by ~{edge_magnitude:.1f} points."
            )
        else:
            recommendation = "NEUTRAL"
            reasoning = f"{team_name} has neutral luck (Luck = {luck:+.3f}). No significant regression expected."

        # Determine confidence
        if edge_magnitude >= self.STRONG_EDGE_THRESHOLD:
            confidence = "high"
        elif edge_magnitude >= self.MEDIUM_EDGE_THRESHOLD:
            confidence = "medium"
        else:
            confidence = "low"

        return LuckAnalysis(
            team_name=team_name,
            current_adjEM=adjEM,
            luck_factor=luck,
            luck_category=category,
            luck_adjusted_adjEM=luck_adjusted_adjEM,
            regression_adjustment=regression_adjustment,
            games_remaining=games_remaining,
            betting_recommendation=recommendation,
            edge_magnitude=edge_magnitude,
            confidence=confidence,
            reasoning=reasoning,
        )

    def analyze_matchup_luck(
        self,
        team1_name: str,
        team1_adjEM: float,
        team1_luck: float,
        team2_name: str,
        team2_adjEM: float,
        team2_luck: float,
        games_remaining: int = 15,
        season_total: int = 30,
        neutral_site: bool = True,
        home_court_advantage: float = 3.5,
    ) -> MatchupLuckEdge:
        """Analyze luck-based edge for a matchup.

        Args:
            team1_name: Team 1 name (home team if not neutral)
            team1_adjEM: Team 1 Adjusted Efficiency Margin
            team1_luck: Team 1 luck factor
            team2_name: Team 2 name (away team if not neutral)
            team2_adjEM: Team 2 Adjusted Efficiency Margin
            team2_luck: Team 2 luck factor
            games_remaining: Games remaining in season
            season_total: Total games in season
            neutral_site: Whether game is at neutral site
            home_court_advantage: Home court advantage in points

        Returns:
            MatchupLuckEdge object with complete analysis
        """
        # Analyze individual teams
        team1_analysis = self.analyze_team_luck(
            team1_name, team1_adjEM, team1_luck, games_remaining, season_total
        )
        team2_analysis = self.analyze_team_luck(
            team2_name, team2_adjEM, team2_luck, games_remaining, season_total
        )

        # Calculate raw margin
        raw_margin = team1_adjEM - team2_adjEM
        if not neutral_site:
            raw_margin += home_court_advantage

        # Calculate luck-adjusted margin
        luck_adjusted_margin = (
            team1_analysis.luck_adjusted_adjEM - team2_analysis.luck_adjusted_adjEM
        )
        if not neutral_site:
            luck_adjusted_margin += home_court_advantage

        # Calculate luck edge
        luck_edge = luck_adjusted_margin - raw_margin

        # Combined luck impact
        combined_luck_impact = team2_analysis.regression_adjustment - team1_analysis.regression_adjustment

        # Generate betting recommendation
        if abs(luck_edge) >= self.STRONG_EDGE_THRESHOLD:
            if luck_edge > 0:
                recommendation = f"STRONG BACK {team1_name}"
                expected_clv = abs(luck_edge) * 0.7  # Conservative CLV estimate
            else:
                recommendation = f"STRONG BACK {team2_name}"
                expected_clv = abs(luck_edge) * 0.7
            confidence = "high"
        elif abs(luck_edge) >= self.MEDIUM_EDGE_THRESHOLD:
            if luck_edge > 0:
                recommendation = f"LEAN {team1_name}"
                expected_clv = abs(luck_edge) * 0.5
            else:
                recommendation = f"LEAN {team2_name}"
                expected_clv = abs(luck_edge) * 0.5
            confidence = "medium"
        else:
            recommendation = "NO SIGNIFICANT LUCK EDGE"
            expected_clv = 0.0
            confidence = "low"

        # Generate reasoning
        reasoning_parts = []
        reasoning_parts.append(f"{team1_name}: {team1_analysis.reasoning}")
        reasoning_parts.append(f"{team2_name}: {team2_analysis.reasoning}")
        reasoning_parts.append(
            f"\nLuck-adjusted prediction: {team1_name} by {luck_adjusted_margin:+.1f} "
            f"(vs raw {raw_margin:+.1f})"
        )
        reasoning_parts.append(f"Luck edge: {luck_edge:+.1f} points")

        if abs(luck_edge) >= self.MEDIUM_EDGE_THRESHOLD:
            reasoning_parts.append(
                f"\n💰 BETTING OPPORTUNITY: {recommendation} "
                f"(Expected CLV: +{expected_clv:.1f} points)"
            )

        reasoning = "\n".join(reasoning_parts)

        return MatchupLuckEdge(
            team1_name=team1_name,
            team2_name=team2_name,
            team1_luck=team1_luck,
            team2_luck=team2_luck,
            raw_margin=raw_margin,
            luck_adjusted_margin=luck_adjusted_margin,
            luck_edge=luck_edge,
            combined_luck_impact=combined_luck_impact,
            betting_recommendation=recommendation,
            confidence=confidence,
            expected_clv=expected_clv,
            reasoning=reasoning,
        )


def identify_luck_opportunities(
    teams: list[dict],
    min_edge: float = 1.5,
) -> list[LuckAnalysis]:
    """Identify teams with significant luck regression opportunities.

    Args:
        teams: List of team dicts with 'TeamName', 'AdjEM', 'Luck' fields
        min_edge: Minimum edge magnitude to include

    Returns:
        List of LuckAnalysis objects, sorted by edge magnitude
    """
    analyzer = LuckRegressionAnalyzer()
    opportunities = []

    for team in teams:
        analysis = analyzer.analyze_team_luck(
            team_name=team['TeamName'],
            adjEM=team['AdjEM'],
            luck=team['Luck'],
            games_remaining=15,  # Default mid-season
        )

        if analysis.edge_magnitude >= min_edge:
            opportunities.append(analysis)

    # Sort by edge magnitude (highest first)
    opportunities.sort(key=lambda x: x.edge_magnitude, reverse=True)

    return opportunities


# Convenience functions for quick analysis
def quick_luck_check(adjEM: float, luck: float) -> str:
    """Quick luck check for a team.

    Args:
        adjEM: Team's Adjusted Efficiency Margin
        luck: Team's luck factor

    Returns:
        Quick recommendation string
    """
    analyzer = LuckRegressionAnalyzer()
    analysis = analyzer.analyze_team_luck(
        team_name="Team",
        adjEM=adjEM,
        luck=luck,
        games_remaining=15,
    )

    return f"{analysis.betting_recommendation}: {analysis.reasoning}"


def calculate_luck_edge(
    team1_adjEM: float,
    team1_luck: float,
    team2_adjEM: float,
    team2_luck: float,
) -> float:
    """Quick calculation of luck edge in a matchup.

    Args:
        team1_adjEM: Team 1's AdjEM
        team1_luck: Team 1's luck factor
        team2_adjEM: Team 2's AdjEM
        team2_luck: Team 2's luck factor

    Returns:
        Luck edge in points (positive favors team1, negative favors team2)
    """
    analyzer = LuckRegressionAnalyzer()

    # Calculate adjustments
    team1_adj = analyzer.calculate_regression_adjustment(team1_luck, games_remaining=15)
    team2_adj = analyzer.calculate_regression_adjustment(team2_luck, games_remaining=15)

    # Edge is the difference in adjustments
    luck_edge = team2_adj - team1_adj

    return luck_edge
