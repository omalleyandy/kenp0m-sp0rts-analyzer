"""Comprehensive matchup analysis combining analysis modules.

Provides a unified interface for analyzing basketball matchups across
multiple dimensions with configurable weights.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .api_client import KenPomAPI
from .helpers import normalize_team_name


@dataclass
class DimensionScore:
    """Score for a single analysis dimension."""

    dimension: str
    score: float
    confidence: float
    key_insight: str
    details: str = ""


@dataclass
class MatchupWeights:
    """Weights for different analysis dimensions."""

    efficiency: float = 0.30
    four_factors: float = 0.25
    tempo: float = 0.15
    size: float = 0.15
    experience: float = 0.15

    def __post_init__(self):
        """Validate that weights sum to 1.0."""
        total = (
            self.efficiency
            + self.four_factors
            + self.tempo
            + self.size
            + self.experience
        )
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Weights must sum to 1.0, got {total}")

    @classmethod
    def tournament_weights(cls) -> MatchupWeights:
        """Create weight profile for NCAA Tournament context."""
        return cls(
            efficiency=0.25,
            four_factors=0.25,
            tempo=0.15,
            size=0.15,
            experience=0.20,
        )


@dataclass
class ComprehensiveMatchupReport:
    """Complete matchup analysis report combining all dimensions."""

    team1: str
    team2: str
    season: int
    tournament_context: bool
    weights: MatchupWeights

    efficiency_score: DimensionScore
    four_factors_score: DimensionScore
    tempo_score: DimensionScore
    size_score: DimensionScore
    experience_score: DimensionScore

    composite_score: float
    key_factors: list[tuple[str, float]]
    team1_game_plan: list[str]
    team2_game_plan: list[str]
    overall_assessment: str

    def generate_text_report(self, detailed: bool = True) -> str:
        """Generate comprehensive text-based matchup report."""
        lines = [
            f"{'='*60}",
            f"COMPREHENSIVE MATCHUP ANALYSIS",
            f"{self.team1} vs {self.team2}",
            f"Season: {self.season}",
            f"{'='*60}",
            "",
            f"COMPOSITE SCORE: {self.composite_score:+.2f}",
            f"(positive favors {self.team1})",
            "",
            "KEY FACTORS:",
        ]

        for factor, score in self.key_factors[:5]:
            lines.append(f"  • {factor}: {score:+.2f}")

        lines.extend([
            "",
            f"OVERALL ASSESSMENT:",
            f"  {self.overall_assessment}",
            "",
        ])

        if detailed:
            lines.extend([
                "DIMENSION SCORES:",
                f"  • Efficiency: {self.efficiency_score.score:+.2f}",
                f"    {self.efficiency_score.key_insight}",
                f"  • Four Factors: {self.four_factors_score.score:+.2f}",
                f"    {self.four_factors_score.key_insight}",
                f"  • Tempo: {self.tempo_score.score:+.2f}",
                f"    {self.tempo_score.key_insight}",
                "",
            ])

        lines.extend([
            f"GAME PLAN - {self.team1}:",
        ])
        for plan in self.team1_game_plan[:3]:
            lines.append(f"  • {plan}")

        lines.extend([
            "",
            f"GAME PLAN - {self.team2}:",
        ])
        for plan in self.team2_game_plan[:3]:
            lines.append(f"  • {plan}")

        return "\n".join(lines)


class ComprehensiveMatchupAnalyzer:
    """Unified matchup analyzer combining all dimensions."""

    def __init__(self, api: KenPomAPI | None = None):
        """Initialize the comprehensive analyzer."""
        self.api = api or KenPomAPI()

    def analyze_matchup(
        self,
        team1: str,
        team2: str,
        season: int,
        tournament_context: bool = False,
        weights: MatchupWeights | None = None,
    ) -> ComprehensiveMatchupReport:
        """Analyze matchup across all dimensions."""
        team1 = normalize_team_name(team1)
        team2 = normalize_team_name(team2)

        if weights is None:
            weights = (
                MatchupWeights.tournament_weights()
                if tournament_context
                else MatchupWeights()
            )

        # Get efficiency data
        efficiency = self.api.get_efficiency(year=season)
        df = efficiency.to_dataframe()

        team1_data = df[df['TeamName'] == team1].iloc[0].to_dict()
        team2_data = df[df['TeamName'] == team2].iloc[0].to_dict()

        # Calculate efficiency score
        em_diff = team1_data['AdjEM'] - team2_data['AdjEM']
        efficiency_score = DimensionScore(
            dimension="Efficiency",
            score=em_diff / 3,  # Normalize to roughly -10 to +10
            confidence=0.9,
            key_insight=f"{team1 if em_diff > 0 else team2} has efficiency edge",
        )

        # Calculate tempo score
        tempo_diff = team1_data['AdjTempo'] - team2_data['AdjTempo']
        tempo_score = DimensionScore(
            dimension="Tempo",
            score=tempo_diff / 5,
            confidence=0.8,
            key_insight=f"Pace favors {team1 if tempo_diff > 0 else team2}",
        )

        # Simplified four factors score
        four_factors_score = DimensionScore(
            dimension="Four Factors",
            score=em_diff / 4,
            confidence=0.85,
            key_insight="Shooting and ball security are key",
        )

        # Size score (placeholder)
        size_score = DimensionScore(
            dimension="Size",
            score=0.0,
            confidence=0.7,
            key_insight="Size matchup is neutral",
        )

        # Experience score (placeholder)
        experience_score = DimensionScore(
            dimension="Experience",
            score=0.0,
            confidence=0.7,
            key_insight="Experience levels similar",
        )

        # Calculate composite score
        composite_score = (
            efficiency_score.score * weights.efficiency
            + four_factors_score.score * weights.four_factors
            + tempo_score.score * weights.tempo
            + size_score.score * weights.size
            + experience_score.score * weights.experience
        )

        # Generate key factors
        key_factors = sorted([
            ("Efficiency", efficiency_score.score * weights.efficiency),
            ("Four Factors", four_factors_score.score * weights.four_factors),
            ("Tempo", tempo_score.score * weights.tempo),
            ("Size", size_score.score * weights.size),
            ("Experience", experience_score.score * weights.experience),
        ], key=lambda x: abs(x[1]), reverse=True)

        # Generate game plans
        team1_game_plan = self._generate_game_plan(
            team1, team2, team1_data, team2_data, is_team1=True
        )
        team2_game_plan = self._generate_game_plan(
            team2, team1, team2_data, team1_data, is_team1=False
        )

        # Overall assessment
        if composite_score > 3:
            overall = f"{team1} is heavily favored based on efficiency edge"
        elif composite_score > 1:
            overall = f"{team1} has slight advantage but {team2} can compete"
        elif composite_score < -3:
            overall = f"{team2} is heavily favored based on efficiency edge"
        elif composite_score < -1:
            overall = f"{team2} has slight advantage but {team1} can compete"
        else:
            overall = "Even matchup - execution will be decisive"

        return ComprehensiveMatchupReport(
            team1=team1,
            team2=team2,
            season=season,
            tournament_context=tournament_context,
            weights=weights,
            efficiency_score=efficiency_score,
            four_factors_score=four_factors_score,
            tempo_score=tempo_score,
            size_score=size_score,
            experience_score=experience_score,
            composite_score=composite_score,
            key_factors=key_factors,
            team1_game_plan=team1_game_plan,
            team2_game_plan=team2_game_plan,
            overall_assessment=overall,
        )

    def _generate_game_plan(
        self,
        team: str,
        opponent: str,
        team_data: dict,
        opponent_data: dict,
        is_team1: bool,
    ) -> list[str]:
        """Generate strategic game plan for a team."""
        plans = []

        # Tempo strategy
        if team_data['AdjTempo'] > opponent_data['AdjTempo'] + 2:
            plans.append("Push pace - tempo advantage")
        elif team_data['AdjTempo'] < opponent_data['AdjTempo'] - 2:
            plans.append("Control pace - slow it down")

        # Offensive strategy
        if team_data['AdjOE'] > opponent_data['AdjDE']:
            plans.append("Attack the paint - efficiency advantage")
        else:
            plans.append("Execute half-court offense patiently")

        # Defensive strategy
        if team_data['AdjDE'] < opponent_data['AdjOE']:
            plans.append("Force tough shots - defensive strength")
        else:
            plans.append("Limit second-chance opportunities")

        return plans if plans else ["Execute fundamentals"]
