"""Matchup report generator for basketball analysis.

Generates formatted reports from analysis results for various output formats.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from .api_client import KenPomAPI
from .utils import normalize_team_name


@dataclass
class MatchupReport:
    """Generated matchup report."""

    team1: str
    team2: str
    season: int
    generated_at: datetime
    summary: str
    detailed_analysis: str
    key_factors: list[str]
    prediction: dict[str, Any]
    betting_insights: list[str]


class MatchupReportGenerator:
    """Generate formatted matchup reports."""

    def __init__(self, api: KenPomAPI | None = None):
        """Initialize report generator."""
        self.api = api or KenPomAPI()

    def generate_report(
        self,
        team1: str,
        team2: str,
        season: int,
        include_betting: bool = True,
        format_type: str = "text",
    ) -> MatchupReport:
        """Generate a comprehensive matchup report.

        Args:
            team1: First team name
            team2: Second team name
            season: Season year
            include_betting: Include betting insights
            format_type: Output format (text, markdown, html)

        Returns:
            MatchupReport with formatted analysis
        """
        team1 = normalize_team_name(team1)
        team2 = normalize_team_name(team2)

        # Get team data
        efficiency = self.api.get_efficiency(year=season)
        df = efficiency.to_dataframe()

        team1_data = df[df['TeamName'] == team1].iloc[0].to_dict()
        team2_data = df[df['TeamName'] == team2].iloc[0].to_dict()

        # Calculate prediction
        em_diff = team1_data['AdjEM'] - team2_data['AdjEM']
        avg_tempo = (team1_data['AdjTempo'] + team2_data['AdjTempo']) / 2
        tempo_factor = avg_tempo / 68.0

        # Add HCA for non-neutral games
        hca = 3.5  # Assume home game for team1

        predicted_margin = em_diff * tempo_factor + hca

        # Calculate total
        avg_oe = (team1_data['AdjOE'] + team2_data['AdjOE']) / 2
        avg_de = (team1_data['AdjDE'] + team2_data['AdjDE']) / 2
        predicted_total = (avg_oe + (200 - avg_de)) / 100 * avg_tempo * 2

        # Generate summary
        if predicted_margin > 10:
            summary = f"{team1} is heavily favored by {predicted_margin:.1f} points"
        elif predicted_margin > 3:
            summary = f"{team1} is favored by {predicted_margin:.1f} points"
        elif predicted_margin > -3:
            summary = f"Close matchup, slight edge to {team1 if predicted_margin > 0 else team2}"
        elif predicted_margin > -10:
            summary = f"{team2} is favored by {abs(predicted_margin):.1f} points"
        else:
            summary = f"{team2} is heavily favored by {abs(predicted_margin):.1f} points"

        # Generate detailed analysis
        detailed = self._generate_detailed_analysis(
            team1, team2, team1_data, team2_data, format_type
        )

        # Key factors
        key_factors = self._identify_key_factors(team1, team2, team1_data, team2_data)

        # Betting insights
        betting_insights = []
        if include_betting:
            betting_insights = self._generate_betting_insights(
                team1, team2, predicted_margin, predicted_total
            )

        return MatchupReport(
            team1=team1,
            team2=team2,
            season=season,
            generated_at=datetime.now(),
            summary=summary,
            detailed_analysis=detailed,
            key_factors=key_factors,
            prediction={
                "margin": round(predicted_margin, 1),
                "total": round(predicted_total, 1),
                "winner": team1 if predicted_margin > 0 else team2,
                "confidence": min(0.95, 0.5 + abs(predicted_margin) / 30),
            },
            betting_insights=betting_insights,
        )

    def _generate_detailed_analysis(
        self,
        team1: str,
        team2: str,
        team1_data: dict,
        team2_data: dict,
        format_type: str,
    ) -> str:
        """Generate detailed analysis section."""
        lines = []

        # Header
        if format_type == "markdown":
            lines.append(f"## {team1} vs {team2}")
            lines.append("")
        else:
            lines.append(f"=== {team1} vs {team2} ===")
            lines.append("")

        # Efficiency comparison
        if format_type == "markdown":
            lines.append("### Efficiency Comparison")
        else:
            lines.append("EFFICIENCY COMPARISON:")

        lines.append(f"  {team1}: AdjEM {team1_data['AdjEM']:+.1f} (#{team1_data['Rank']:.0f})")
        lines.append(f"  {team2}: AdjEM {team2_data['AdjEM']:+.1f} (#{team2_data['Rank']:.0f})")
        lines.append("")

        # Offensive analysis
        if format_type == "markdown":
            lines.append("### Offensive Analysis")
        else:
            lines.append("OFFENSIVE ANALYSIS:")

        lines.append(f"  {team1}: {team1_data['AdjOE']:.1f} pts/100 poss (#{team1_data.get('ORank', 'N/A')})")
        lines.append(f"  {team2}: {team2_data['AdjOE']:.1f} pts/100 poss (#{team2_data.get('ORank', 'N/A')})")
        lines.append("")

        # Defensive analysis
        if format_type == "markdown":
            lines.append("### Defensive Analysis")
        else:
            lines.append("DEFENSIVE ANALYSIS:")

        lines.append(f"  {team1}: {team1_data['AdjDE']:.1f} pts/100 poss (#{team1_data.get('DRank', 'N/A')})")
        lines.append(f"  {team2}: {team2_data['AdjDE']:.1f} pts/100 poss (#{team2_data.get('DRank', 'N/A')})")
        lines.append("")

        # Tempo
        if format_type == "markdown":
            lines.append("### Tempo")
        else:
            lines.append("TEMPO:")

        lines.append(f"  {team1}: {team1_data['AdjTempo']:.1f} poss/game")
        lines.append(f"  {team2}: {team2_data['AdjTempo']:.1f} poss/game")
        lines.append(f"  Expected pace: {(team1_data['AdjTempo'] + team2_data['AdjTempo'])/2:.1f}")

        return "\n".join(lines)

    def _identify_key_factors(
        self,
        team1: str,
        team2: str,
        team1_data: dict,
        team2_data: dict,
    ) -> list[str]:
        """Identify key factors for the matchup."""
        factors = []

        # Efficiency gap
        em_gap = abs(team1_data['AdjEM'] - team2_data['AdjEM'])
        if em_gap > 10:
            better = team1 if team1_data['AdjEM'] > team2_data['AdjEM'] else team2
            factors.append(f"Large efficiency gap favors {better} ({em_gap:.1f} pts)")

        # Tempo mismatch
        tempo_gap = abs(team1_data['AdjTempo'] - team2_data['AdjTempo'])
        if tempo_gap > 5:
            faster = team1 if team1_data['AdjTempo'] > team2_data['AdjTempo'] else team2
            factors.append(f"Tempo mismatch - {faster} wants to push pace")

        # Offensive vs defensive matchup
        if team1_data['AdjOE'] > team2_data['AdjDE'] + 5:
            factors.append(f"{team1} offense should exploit {team2} defense")
        if team2_data['AdjOE'] > team1_data['AdjDE'] + 5:
            factors.append(f"{team2} offense should exploit {team1} defense")

        if not factors:
            factors.append("Balanced matchup - execution will be key")

        return factors

    def _generate_betting_insights(
        self,
        team1: str,
        team2: str,
        predicted_margin: float,
        predicted_total: float,
    ) -> list[str]:
        """Generate betting-relevant insights."""
        insights = []

        # Spread insight
        if abs(predicted_margin) > 15:
            insights.append(f"Large spread expected - look for value if line is off")
        elif abs(predicted_margin) < 3:
            insights.append(f"Toss-up game - spread likely around pick'em")

        # Total insight
        if predicted_total > 155:
            insights.append(f"High-scoring game expected ({predicted_total:.0f}) - OVER potential")
        elif predicted_total < 130:
            insights.append(f"Low-scoring game expected ({predicted_total:.0f}) - UNDER potential")

        # Model confidence
        confidence = min(0.95, 0.5 + abs(predicted_margin) / 30)
        if confidence > 0.75:
            insights.append(f"High model confidence ({confidence:.0%}) on {team1 if predicted_margin > 0 else team2}")

        return insights

    def format_as_markdown(self, report: MatchupReport) -> str:
        """Format report as Markdown."""
        lines = [
            f"# Matchup Analysis: {report.team1} vs {report.team2}",
            f"*Season: {report.season} | Generated: {report.generated_at.strftime('%Y-%m-%d %H:%M')}*",
            "",
            "## Summary",
            report.summary,
            "",
            "## Prediction",
            f"- **Winner**: {report.prediction['winner']}",
            f"- **Margin**: {report.prediction['margin']:+.1f}",
            f"- **Total**: {report.prediction['total']:.1f}",
            f"- **Confidence**: {report.prediction['confidence']:.0%}",
            "",
            "## Key Factors",
        ]

        for factor in report.key_factors:
            lines.append(f"- {factor}")

        if report.betting_insights:
            lines.extend([
                "",
                "## Betting Insights",
            ])
            for insight in report.betting_insights:
                lines.append(f"- {insight}")

        lines.extend([
            "",
            "## Detailed Analysis",
            report.detailed_analysis,
        ])

        return "\n".join(lines)
