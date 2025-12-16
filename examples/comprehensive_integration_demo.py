"""Demo of comprehensive matchup analysis integrating all 7 analyzer modules.

This example shows how to use the ComprehensiveMatchupAnalyzer to perform
multi-dimensional matchup analysis with weighted composite scoring and
strategic synthesis.

Run with: uv run python examples/comprehensive_integration_demo.py
"""

import os

from kenp0m_sp0rts_analyzer.api_client import KenPomAPI
from kenp0m_sp0rts_analyzer.comprehensive_matchup_analysis import (
    ComprehensiveMatchupAnalyzer,
    MatchupWeights,
)


def main():
    """Run comprehensive matchup analysis examples."""
    # Initialize API
    api_key = os.getenv("KENPOM_API_KEY")
    if not api_key:
        print("Error: KENPOM_API_KEY environment variable not set")
        print(
            "Please set your API key: export KENPOM_API_KEY='your-key-here'"
        )
        return

    api = KenPomAPI(api_key=api_key)
    analyzer = ComprehensiveMatchupAnalyzer(api)

    # Example matchups for 2025 season
    matchups = [
        ("Duke", "North Carolina"),
        ("Purdue", "Arizona"),
        ("Houston", "Kansas"),
    ]

    print("\n" + "=" * 80)
    print("COMPREHENSIVE MATCHUP ANALYSIS - REGULAR SEASON")
    print("=" * 80)

    for team1, team2 in matchups:
        print(f"\n\nAnalyzing: {team1} vs {team2}")
        print("-" * 80)

        try:
            # Regular season analysis
            report = analyzer.analyze_matchup(team1, team2, 2025)

            # Generate and display text report
            print(report.generate_text_report(detailed=True))

        except ValueError as e:
            print(f"Error: {e}")
            continue

    # NCAA Tournament context example
    print("\n\n" + "=" * 80)
    print("NCAA TOURNAMENT CONTEXT - DIFFERENT WEIGHT PROFILE")
    print("=" * 80)

    print("\nAnalyzing: Duke vs North Carolina (Tournament Setting)")
    print("-" * 80)

    try:
        # Tournament analysis with different weights
        report = analyzer.analyze_matchup(
            "Duke", "North Carolina", 2025, tournament_context=True
        )

        print(report.generate_text_report(detailed=True))

        # Show weight differences
        print("\n" + "=" * 80)
        print("WEIGHT PROFILE COMPARISON")
        print("=" * 80)

        regular_weights = MatchupWeights()
        tourney_weights = MatchupWeights.tournament_weights()

        print(f"\n{'Dimension':<25} {'Regular':>10} {'Tournament':>12} {'Change':>10}")
        print("-" * 60)

        dimensions = [
            ("Efficiency", "efficiency"),
            ("Four Factors", "four_factors"),
            ("Tempo", "tempo"),
            ("Point Distribution", "point_distribution"),
            ("Defensive", "defensive"),
            ("Size", "size"),
            ("Experience", "experience"),
        ]

        for name, attr in dimensions:
            reg_val = getattr(regular_weights, attr)
            tour_val = getattr(tourney_weights, attr)
            change = tour_val - reg_val

            print(
                f"{name:<25} {reg_val:>9.0%} {tour_val:>11.0%} "
                f"{change:>+9.1%}"
            )

    except ValueError as e:
        print(f"Error: {e}")

    # Custom weights example
    print("\n\n" + "=" * 80)
    print("CUSTOM WEIGHT PROFILE - DEFENSIVE-FOCUSED ANALYSIS")
    print("=" * 80)

    # Create custom weights emphasizing defense
    defensive_weights = MatchupWeights(
        efficiency=0.20,  # Slightly reduced
        four_factors=0.20,
        tempo=0.08,
        point_distribution=0.12,
        defensive=0.25,  # Significantly increased
        size=0.10,
        experience=0.05,
    )

    print("\nAnalyzing: Houston vs Kansas (Defensive emphasis)")
    print("-" * 80)

    try:
        report = analyzer.analyze_matchup(
            "Houston",
            "Kansas",
            2025,
            custom_weights=defensive_weights,
        )

        print(report.generate_text_report(detailed=False))

        print("\nCustom Weight Profile:")
        print(f"  Defensive emphasis: {defensive_weights.defensive:.0%} (vs 15% default)")

    except ValueError as e:
        print(f"Error: {e}")

    print("\n" + "=" * 80)
    print("Demo complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
