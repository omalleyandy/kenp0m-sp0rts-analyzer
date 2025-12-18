"""Luck Regression Report Generator.

Generates comprehensive reports on luck regression opportunities across all teams.
Identifies overvalued (lucky) and undervalued (unlucky) teams for betting edges.

Usage:
    # Generate report for all teams
    uv run python scripts/analysis/luck_regression_reporter.py

    # Filter by conference
    uv run python scripts/analysis/luck_regression_reporter.py --conference ACC

    # Set minimum edge threshold
    uv run python scripts/analysis/luck_regression_reporter.py --min-edge 2.0

    # Export to file
    uv run python scripts/analysis/luck_regression_reporter.py -o reports/luck_regression.md
"""
import argparse
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from kenp0m_sp0rts_analyzer.api_client import KenPomAPI
from kenp0m_sp0rts_analyzer.luck_regression import (
    LuckRegressionAnalyzer,
    identify_luck_opportunities,
)


class LuckRegressionReporter:
    """Generates comprehensive luck regression reports."""

    def __init__(self, api_key: str | None = None):
        """Initialize reporter.

        Args:
            api_key: KenPom API key (reads from env if not provided)
        """
        self.api = KenPomAPI(api_key=api_key)
        self.analyzer = LuckRegressionAnalyzer()

    def get_all_teams(self, year: int = 2025, conference: str | None = None) -> list[dict]:
        """Get all teams with luck ratings.

        Args:
            year: Season year
            conference: Optional conference filter

        Returns:
            List of team dicts with ratings
        """
        print(f"[KENPOM] Fetching {year} ratings...")

        try:
            response = self.api.get_ratings(year=year, conference=conference)

            if not response.data:
                print(f"  [WARNING] No ratings found for {year}")
                return []

            teams = []
            for team in response.data:
                teams.append({
                    'TeamName': team.get('TeamName'),
                    'AdjEM': team.get('AdjEM', 0.0),
                    'Luck': team.get('Luck', 0.0),
                    'Rank': team.get('Rank', 999),
                    'Conference': team.get('Conf', 'Unknown'),
                    'Record': team.get('Record', 'N/A'),
                })

            print(f"  [OK] Found {len(teams)} teams")
            return teams

        except Exception as e:
            print(f"  [ERROR] Failed to fetch ratings: {e}")
            return []

    def generate_opportunities_report(
        self,
        teams: list[dict],
        min_edge: float = 1.5,
    ) -> str:
        """Generate markdown report of luck regression opportunities.

        Args:
            teams: List of team dicts
            min_edge: Minimum edge magnitude to include

        Returns:
            Markdown report text
        """
        # Identify opportunities
        opportunities = identify_luck_opportunities(teams, min_edge=min_edge)

        # Separate by recommendation type
        fade_teams = [o for o in opportunities if o.betting_recommendation == "FADE"]
        back_teams = [o for o in opportunities if o.betting_recommendation == "BACK"]

        # Generate report
        lines = []
        lines.append("# Luck Regression Opportunities Report")
        lines.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"Minimum Edge Threshold: {min_edge} points")
        lines.append("\n" + "=" * 70)

        # Summary
        lines.append("\n## Summary")
        lines.append(f"\n- Total teams analyzed: {len(teams)}")
        lines.append(f"- Teams with significant luck edge: {len(opportunities)}")
        lines.append(f"  - Overvalued (FADE): {len(fade_teams)}")
        lines.append(f"  - Undervalued (BACK): {len(back_teams)}")

        # Key insights
        if opportunities:
            highest_edge = max(opportunities, key=lambda x: x.edge_magnitude)
            lines.append(f"\n**Highest Edge Opportunity:**")
            lines.append(f"- {highest_edge.team_name}: {highest_edge.betting_recommendation}")
            lines.append(f"- Edge Magnitude: {highest_edge.edge_magnitude:.1f} points")
            lines.append(f"- Category: {highest_edge.luck_category.replace('_', ' ').title()}")

        # FADE opportunities (overvalued teams)
        if fade_teams:
            lines.append("\n" + "=" * 70)
            lines.append("\n## FADE Opportunities (Overvalued Teams)")
            lines.append("\nThese teams are lucky and expected to regress. Fade them (bet against).")
            lines.append("\n" + "-" * 70)

            for i, opp in enumerate(fade_teams, 1):
                lines.append(f"\n### {i}. {opp.team_name}")
                lines.append(f"\n**Recommendation: {opp.betting_recommendation}**")
                lines.append(f"- Edge Magnitude: {opp.edge_magnitude:.1f} points")
                lines.append(f"- Confidence: {opp.confidence.upper()}")
                lines.append(f"- Luck Category: {opp.luck_category.replace('_', ' ').title()}")
                lines.append(f"\n**Current Metrics:**")
                lines.append(f"- AdjEM: {opp.current_adjEM:.1f}")
                lines.append(f"- Luck Factor: {opp.luck_factor:+.3f}")
                lines.append(f"\n**Adjusted Metrics:**")
                lines.append(f"- Luck-Adjusted AdjEM: {opp.luck_adjusted_adjEM:.1f}")
                lines.append(f"- Regression Adjustment: {opp.regression_adjustment:+.1f} points")
                lines.append(f"\n**Analysis:**")
                lines.append(f"{opp.reasoning}")

        # BACK opportunities (undervalued teams)
        if back_teams:
            lines.append("\n" + "=" * 70)
            lines.append("\n## BACK Opportunities (Undervalued Teams)")
            lines.append("\nThese teams are unlucky and expected to improve. Back them (bet on).")
            lines.append("\n" + "-" * 70)

            for i, opp in enumerate(back_teams, 1):
                lines.append(f"\n### {i}. {opp.team_name}")
                lines.append(f"\n**Recommendation: {opp.betting_recommendation}**")
                lines.append(f"- Edge Magnitude: {opp.edge_magnitude:.1f} points")
                lines.append(f"- Confidence: {opp.confidence.upper()}")
                lines.append(f"- Luck Category: {opp.luck_category.replace('_', ' ').title()}")
                lines.append(f"\n**Current Metrics:**")
                lines.append(f"- AdjEM: {opp.current_adjEM:.1f}")
                lines.append(f"- Luck Factor: {opp.luck_factor:+.3f}")
                lines.append(f"\n**Adjusted Metrics:**")
                lines.append(f"- Luck-Adjusted AdjEM: {opp.luck_adjusted_adjEM:.1f}")
                lines.append(f"- Regression Adjustment: {opp.regression_adjustment:+.1f} points")
                lines.append(f"\n**Analysis:**")
                lines.append(f"{opp.reasoning}")

        # No opportunities found
        if not opportunities:
            lines.append("\n## No Significant Opportunities Found")
            lines.append(f"\nNo teams meet the minimum edge threshold of {min_edge} points.")
            lines.append("\nConsider lowering the threshold or waiting for more games to be played.")

        # Methodology
        lines.append("\n" + "=" * 70)
        lines.append("\n## Methodology")
        lines.append("\n**Luck Regression Theory:**")
        lines.append("- Teams with high luck (>0.15) have won more close games than expected")
        lines.append("- Teams with low luck (<-0.15) have lost more close games than expected")
        lines.append("- Close game performance ALWAYS regresses to the mean")
        lines.append("- This creates 2-5 point edges when exploited correctly")
        lines.append("\n**Calculation:**")
        lines.append("- Luck converts to ~10 points per game impact")
        lines.append("- 50% of luck regresses over remaining games")
        lines.append("- Adjustment weighted by games remaining / season total")
        lines.append("\n**Confidence Levels:**")
        lines.append("- HIGH: Edge magnitude >= 2.5 points")
        lines.append("- MEDIUM: Edge magnitude >= 1.5 points")
        lines.append("- LOW: Edge magnitude < 1.5 points")

        return "\n".join(lines)

    def generate_team_rankings_report(
        self,
        teams: list[dict],
    ) -> str:
        """Generate luck-adjusted team rankings.

        Args:
            teams: List of team dicts

        Returns:
            Markdown report text
        """
        # Analyze all teams
        analyses = []
        for team in teams:
            analysis = self.analyzer.analyze_team_luck(
                team_name=team['TeamName'],
                adjEM=team['AdjEM'],
                luck=team['Luck'],
                games_remaining=15,  # Mid-season default
            )
            analyses.append((team, analysis))

        # Sort by luck-adjusted AdjEM
        analyses.sort(key=lambda x: x[1].luck_adjusted_adjEM, reverse=True)

        # Generate report
        lines = []
        lines.append("# Luck-Adjusted Team Rankings")
        lines.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("\n" + "=" * 70)

        lines.append("\n## Rankings")
        lines.append("\nTeams ranked by luck-adjusted efficiency margin (AdjEM).")
        lines.append("\n")
        lines.append(f"{'Rank':<6} {'Team':<25} {'AdjEM':>8} {'Luck':>8} {'Adjusted':>8} {'Rec':<8}")
        lines.append("-" * 70)

        for i, (team, analysis) in enumerate(analyses, 1):
            rec_short = analysis.betting_recommendation[:4]  # FADE or BACK or NEUT
            lines.append(
                f"{i:<6} {team['TeamName']:<25} "
                f"{analysis.current_adjEM:>8.1f} {analysis.luck_factor:>+8.3f} "
                f"{analysis.luck_adjusted_adjEM:>8.1f} {rec_short:<8}"
            )

        # Biggest movers
        lines.append("\n" + "=" * 70)
        lines.append("\n## Biggest Movers")

        # Most overvalued
        most_overvalued = max(
            [a for t, a in analyses if a.regression_adjustment > 0],
            key=lambda x: x.regression_adjustment,
            default=None
        )
        if most_overvalued:
            lines.append("\n**Most Overvalued Team:**")
            lines.append(f"- {most_overvalued.team_name}")
            lines.append(f"- Overvalued by {most_overvalued.regression_adjustment:.1f} points")
            lines.append(f"- Luck: {most_overvalued.luck_factor:+.3f}")

        # Most undervalued
        most_undervalued = min(
            [a for t, a in analyses if a.regression_adjustment < 0],
            key=lambda x: x.regression_adjustment,
            default=None
        )
        if most_undervalued:
            lines.append("\n**Most Undervalued Team:**")
            lines.append(f"- {most_undervalued.team_name}")
            lines.append(f"- Undervalued by {abs(most_undervalued.regression_adjustment):.1f} points")
            lines.append(f"- Luck: {most_undervalued.luck_factor:+.3f}")

        return "\n".join(lines)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Luck regression report generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--year", "-y",
        type=int,
        default=2025,
        help="Season year (default: 2025)",
    )

    parser.add_argument(
        "--conference", "-c",
        type=str,
        help="Filter by conference (e.g., ACC, B12, SEC)",
    )

    parser.add_argument(
        "--min-edge",
        type=float,
        default=1.5,
        help="Minimum edge threshold in points (default: 1.5)",
    )

    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Output report path (markdown format)",
    )

    parser.add_argument(
        "--rankings",
        action="store_true",
        help="Generate luck-adjusted rankings instead of opportunities",
    )

    args = parser.parse_args()

    # Initialize reporter
    print("=" * 70)
    print("LUCK REGRESSION REPORT GENERATOR")
    print("=" * 70)
    print(f"Year: {args.year}")
    if args.conference:
        print(f"Conference: {args.conference}")
    print("=" * 70)

    reporter = LuckRegressionReporter()

    # Get teams
    teams = reporter.get_all_teams(year=args.year, conference=args.conference)

    if not teams:
        print("\n[ERROR] No teams found")
        return 1

    # Generate report
    if args.rankings:
        report = reporter.generate_team_rankings_report(teams)
    else:
        report = reporter.generate_opportunities_report(teams, min_edge=args.min_edge)

    # Save or display
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            f.write(report)

        print(f"\n[SAVED] {output_path}")
    else:
        print("\n" + report)

    return 0


if __name__ == "__main__":
    sys.exit(main())
