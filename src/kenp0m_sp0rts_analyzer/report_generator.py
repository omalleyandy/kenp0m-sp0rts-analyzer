"""Automated Matchup Report Generator.

Generates comprehensive, actionable matchup reports by synthesizing insights
from all 7 analytical modules into multiple output formats.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Literal, Optional

from .api_client import KenPomAPI
from .comprehensive_matchup_analysis import ComprehensiveMatchupAnalyzer


@dataclass
class MatchupReport:
    """Complete matchup analysis report."""

    # Metadata
    team1: str
    team2: str
    season: int
    generated_at: datetime

    # Executive Summary
    executive_summary: str
    overall_advantage: str  # "Team A by X.X points" or "Even matchup"
    confidence_level: str  # "High", "Medium", "Low"

    # Dimensional Analysis
    dimensional_scores: dict[str, float]  # dimension -> score (0-10, 5=neutral)
    team1_advantages: list[str]  # Top advantages for team1
    team2_advantages: list[str]  # Top advantages for team2

    # Strategic Insights
    key_factors: list[str]  # What will decide the game
    strategic_recommendations: dict[str, list[str]]  # team -> recommendations

    # Prediction
    predicted_margin: float
    win_probability: dict[str, float]  # team -> probability
    scoring_prediction: dict[str, float]  # team -> predicted points

    # Raw Data
    raw_analysis: dict  # Full comprehensive analysis data


ReportFormat = Literal["markdown", "html", "json"]


class MatchupReportGenerator:
    """Generate comprehensive matchup reports in multiple formats."""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize report generator.

        Args:
            api_key: KenPom API key (uses env var if not provided)
        """
        self.api = KenPomAPI(api_key)
        self.analyzer = ComprehensiveMatchupAnalyzer(self.api)

    def generate_report(
        self,
        team1: str,
        team2: str,
        season: int = 2025,
        format: ReportFormat = "markdown"
    ) -> str:
        """Generate comprehensive matchup report.

        Args:
            team1: First team name
            team2: Second team name
            season: Season year
            format: Output format (markdown, html, json)

        Returns:
            Formatted report string
        """
        # Run comprehensive analysis
        analysis = self.analyzer.analyze_matchup(team1, team2, season)

        # Build structured report
        report = self._build_report_data(team1, team2, season, analysis)

        # Format output
        if format == "markdown":
            return self._format_markdown(report)
        elif format == "html":
            return self._format_html(report)
        elif format == "json":
            return self._format_json(report)
        else:
            raise ValueError(f"Unknown format: {format}")

    def _build_report_data(
        self,
        team1: str,
        team2: str,
        season: int,
        analysis: dict
    ) -> MatchupReport:
        """Build structured report data from analysis."""
        # Extract key metrics
        overall_score = analysis["overall_advantage"]
        dimensions = analysis["dimension_scores"]

        # Determine advantage and confidence
        advantage_text = self._format_advantage(team1, team2, overall_score)
        confidence = self._calculate_confidence(analysis)

        # Identify top advantages for each team
        team1_advs = self._identify_advantages(dimensions, favor_team1=True)
        team2_advs = self._identify_advantages(dimensions, favor_team1=False)

        # Generate executive summary
        exec_summary = self._generate_executive_summary(
            team1, team2, overall_score, team1_advs, team2_advs
        )

        # Extract strategic insights
        key_factors = self._extract_key_factors(analysis)
        recommendations = self._extract_recommendations(analysis)

        # Build prediction
        predicted_margin = overall_score - 5.0  # Convert from 0-10 to margin
        win_prob = self._calculate_win_probability(predicted_margin)

        return MatchupReport(
            team1=team1,
            team2=team2,
            season=season,
            generated_at=datetime.now(),
            executive_summary=exec_summary,
            overall_advantage=advantage_text,
            confidence_level=confidence,
            dimensional_scores=dimensions,
            team1_advantages=team1_advs,
            team2_advantages=team2_advs,
            key_factors=key_factors,
            strategic_recommendations=recommendations,
            predicted_margin=predicted_margin,
            win_probability=win_prob,
            scoring_prediction={},  # TODO: Add scoring prediction
            raw_analysis=analysis
        )

    def _format_advantage(
        self,
        team1: str,
        team2: str,
        overall_score: float
    ) -> str:
        """Format overall advantage text."""
        margin = overall_score - 5.0

        if abs(margin) < 0.5:
            return "Even matchup"
        elif margin > 0:
            return f"{team1} advantage (+{margin:.1f})"
        else:
            return f"{team2} advantage ({margin:.1f})"

    def _calculate_confidence(self, analysis: dict) -> str:
        """Calculate confidence level based on consistency across dimensions."""
        scores = list(analysis["dimension_scores"].values())

        # High confidence if dimensions agree
        std_dev = sum((s - 5.0) ** 2 for s in scores) ** 0.5 / len(scores)

        if std_dev < 1.0:
            return "Low"  # Dimensions disagree
        elif std_dev < 2.0:
            return "Medium"
        else:
            return "High"  # Strong agreement

    def _identify_advantages(
        self,
        dimensions: dict[str, float],
        favor_team1: bool,
        top_n: int = 3
    ) -> list[str]:
        """Identify top advantages for a team."""
        # Sort dimensions by advantage (descending if team1, ascending if team2)
        sorted_dims = sorted(
            dimensions.items(),
            key=lambda x: x[1] if favor_team1 else -x[1],
            reverse=True
        )

        # Filter to only significant advantages (>6 for team1, <4 for team2)
        if favor_team1:
            filtered = [(d, s) for d, s in sorted_dims if s > 6.0]
        else:
            filtered = [(d, s) for d, s in sorted_dims if s < 4.0]

        # Return top N with descriptions
        advantages = []
        for dim, score in filtered[:top_n]:
            advantage_text = self._describe_advantage(dim, score, favor_team1)
            advantages.append(advantage_text)

        return advantages

    def _describe_advantage(
        self,
        dimension: str,
        score: float,
        favor_team1: bool
    ) -> str:
        """Create human-readable advantage description."""
        magnitude = abs(score - 5.0)

        if magnitude > 2.5:
            strength = "Major"
        elif magnitude > 1.5:
            strength = "Moderate"
        else:
            strength = "Slight"

        return f"{strength} {dimension.replace('_', ' ').title()} advantage"

    def _generate_executive_summary(
        self,
        team1: str,
        team2: str,
        overall_score: float,
        team1_advs: list[str],
        team2_advs: list[str]
    ) -> str:
        """Generate executive summary paragraph."""
        margin = overall_score - 5.0

        if abs(margin) < 0.5:
            summary = (
                f"This is an evenly matched contest between {team1} and {team2}. "
            )
        elif margin > 0:
            summary = (
                f"{team1} holds a meaningful advantage over {team2} in this matchup. "
            )
        else:
            summary = (
                f"{team2} holds a meaningful advantage over {team1} in this matchup. "
            )

        # Add top advantages
        if team1_advs:
            summary += f"{team1} strengths: {', '.join(team1_advs[:2])}. "
        if team2_advs:
            summary += f"{team2} strengths: {', '.join(team2_advs[:2])}. "

        summary += (
            "This analysis synthesizes 15 dimensions of basketball analytics "
            "including efficiency, four factors, scoring style, defense, size, "
            "and experience."
        )

        return summary

    def _extract_key_factors(self, analysis: dict) -> list[str]:
        """Extract key factors that will decide the game."""
        factors = []

        # Add top 3 most polarized dimensions
        dimensions = analysis["dimension_scores"]
        sorted_by_magnitude = sorted(
            dimensions.items(),
            key=lambda x: abs(x[1] - 5.0),
            reverse=True
        )

        for dim, score in sorted_by_magnitude[:3]:
            factor_text = self._describe_key_factor(dim, score)
            factors.append(factor_text)

        return factors

    def _describe_key_factor(self, dimension: str, score: float) -> str:
        """Describe a key factor in the matchup."""
        dim_name = dimension.replace('_', ' ').title()
        margin = score - 5.0

        if abs(margin) < 0.5:
            return f"{dim_name}: Evenly matched, could swing either way"
        elif margin > 0:
            return f"{dim_name}: Clear advantage, must be neutralized"
        else:
            return f"{dim_name}: Significant deficit, must overcome"

    def _extract_recommendations(self, analysis: dict) -> dict[str, list[str]]:
        """Extract strategic recommendations from analysis."""
        # TODO: Implement extraction from individual analyzer recommendations
        return {
            "team1": ["Control tempo", "Exploit size advantage", "Limit turnovers"],
            "team2": ["Push pace", "Attack in transition", "Crash offensive glass"]
        }

    def _calculate_win_probability(self, margin: float) -> dict[str, float]:
        """Calculate win probability from predicted margin."""
        # Simple logistic curve: 1 / (1 + exp(-margin / 11))
        # 11 points = ~1 standard deviation in college basketball
        import math

        team1_prob = 1.0 / (1.0 + math.exp(-margin / 11.0))
        team2_prob = 1.0 - team1_prob

        return {
            "team1": team1_prob,
            "team2": team2_prob
        }

    def _format_markdown(self, report: MatchupReport) -> str:
        """Format report as Markdown."""
        md = f"""# Matchup Analysis: {report.team1} vs {report.team2}

**Season**: {report.season} | **Generated**: {report.generated_at.strftime('%Y-%m-%d %H:%M')}

---

## Executive Summary

{report.executive_summary}

**Overall Assessment**: {report.overall_advantage} (Confidence: {report.confidence_level})

**Prediction**: {report.team1} {report.predicted_margin:+.1f} | Win Probability: {report.win_probability['team1']:.1%}

---

## Dimensional Analysis

### {report.team1} Advantages
"""

        if report.team1_advantages:
            for adv in report.team1_advantages:
                md += f"- ✅ {adv}\n"
        else:
            md += "- No significant advantages identified\n"

        md += f"\n### {report.team2} Advantages\n"

        if report.team2_advantages:
            for adv in report.team2_advantages:
                md += f"- ✅ {adv}\n"
        else:
            md += "- No significant advantages identified\n"

        md += "\n### Dimension Scores (0-10 scale, 5 = neutral)\n\n"
        md += "| Dimension | Score | Assessment |\n"
        md += "|-----------|-------|------------|\n"

        for dim, score in sorted(report.dimensional_scores.items()):
            assessment = self._score_assessment(score)
            dim_name = dim.replace('_', ' ').title()
            md += f"| {dim_name} | {score:.1f} | {assessment} |\n"

        md += "\n---\n\n## Key Factors\n\n"

        for i, factor in enumerate(report.key_factors, 1):
            md += f"{i}. {factor}\n"

        md += "\n---\n\n## Strategic Recommendations\n\n"
        md += f"### {report.team1}\n"
        for rec in report.strategic_recommendations.get("team1", []):
            md += f"- {rec}\n"

        md += f"\n### {report.team2}\n"
        for rec in report.strategic_recommendations.get("team2", []):
            md += f"- {rec}\n"

        md += "\n---\n\n"
        md += "*This report synthesizes 15 dimensions of basketball analytics "
        md += "using KenPom efficiency metrics, Four Factors, scoring style, "
        md += "defensive schemes, size/athleticism, and experience/chemistry.*\n"

        return md

    def _score_assessment(self, score: float) -> str:
        """Convert score to text assessment."""
        if score >= 7.0:
            return f"Strong {self._team_label(1)} advantage"
        elif score >= 6.0:
            return f"Moderate {self._team_label(1)} advantage"
        elif score >= 5.5:
            return f"Slight {self._team_label(1)} edge"
        elif score >= 4.5:
            return "Neutral"
        elif score >= 4.0:
            return f"Slight {self._team_label(2)} edge"
        elif score >= 3.0:
            return f"Moderate {self._team_label(2)} advantage"
        else:
            return f"Strong {self._team_label(2)} advantage"

    def _team_label(self, team_num: int) -> str:
        """Get team label (Team 1 or Team 2)."""
        return f"Team {team_num}"

    def _format_html(self, report: MatchupReport) -> str:
        """Format report as HTML."""
        # Convert markdown to HTML (basic implementation)
        md = self._format_markdown(report)

        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Matchup Analysis: {report.team1} vs {report.team2}</title>
    <style>
        body {{ font-family: Arial, sans-serif; max-width: 900px; margin: 0 auto; padding: 20px; }}
        h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; }}
        h2 {{ color: #34495e; margin-top: 30px; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background-color: #3498db; color: white; }}
        .summary {{ background-color: #ecf0f1; padding: 15px; border-radius: 5px; }}
        .advantage {{ color: #27ae60; font-weight: bold; }}
        .disadvantage {{ color: #e74c3c; font-weight: bold; }}
    </style>
</head>
<body>
    <h1>Matchup Analysis: {report.team1} vs {report.team2}</h1>
    <p><strong>Season:</strong> {report.season} | <strong>Generated:</strong> {report.generated_at.strftime('%Y-%m-%d %H:%M')}</p>

    <div class="summary">
        <h2>Executive Summary</h2>
        <p>{report.executive_summary}</p>
        <p><strong>Overall Assessment:</strong> {report.overall_advantage}</p>
        <p><strong>Confidence:</strong> {report.confidence_level}</p>
        <p><strong>Prediction:</strong> {report.team1} {report.predicted_margin:+.1f} points | Win Probability: {report.win_probability['team1']:.1%}</p>
    </div>

    <h2>Dimensional Analysis</h2>
    <h3>{report.team1} Advantages</h3>
    <ul>
"""

        for adv in report.team1_advantages or ["No significant advantages"]:
            html += f"        <li class='advantage'>{adv}</li>\n"

        html += f"    </ul>\n    <h3>{report.team2} Advantages</h3>\n    <ul>\n"

        for adv in report.team2_advantages or ["No significant advantages"]:
            html += f"        <li class='advantage'>{adv}</li>\n"

        html += """    </ul>

    <h2>Key Factors</h2>
    <ol>
"""

        for factor in report.key_factors:
            html += f"        <li>{factor}</li>\n"

        html += """    </ol>
</body>
</html>"""

        return html

    def _format_json(self, report: MatchupReport) -> str:
        """Format report as JSON."""
        import json

        report_dict = {
            "metadata": {
                "team1": report.team1,
                "team2": report.team2,
                "season": report.season,
                "generated_at": report.generated_at.isoformat()
            },
            "summary": {
                "executive_summary": report.executive_summary,
                "overall_advantage": report.overall_advantage,
                "confidence_level": report.confidence_level
            },
            "prediction": {
                "margin": report.predicted_margin,
                "win_probability": report.win_probability,
                "scoring": report.scoring_prediction
            },
            "analysis": {
                "dimensional_scores": report.dimensional_scores,
                "team1_advantages": report.team1_advantages,
                "team2_advantages": report.team2_advantages,
                "key_factors": report.key_factors,
                "strategic_recommendations": report.strategic_recommendations
            },
            "raw_data": report.raw_analysis
        }

        return json.dumps(report_dict, indent=2)
