"""Luck Regression Demo - See it in action!

This script demonstrates luck regression analysis with real-world examples.

Usage:
    uv run python examples/luck_regression_demo.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from kenp0m_sp0rts_analyzer.luck_regression import (
    LuckRegressionAnalyzer,
    quick_luck_check,
    calculate_luck_edge,
)


def print_separator(title: str = ""):
    """Print a section separator."""
    print("\n" + "=" * 70)
    if title:
        print(f" {title}")
        print("=" * 70)


def demo_single_team_analysis():
    """Demonstrate single team luck analysis."""
    print_separator("EXAMPLE 1: Single Team Analysis")

    analyzer = LuckRegressionAnalyzer()

    # Example: Very lucky team
    print("\n[TEAM A - Very Lucky]")
    print("Record: 12-2, Luck = +0.18")
    print("Won 5-0 in close games (should be ~2.5-2.5)")

    analysis = analyzer.analyze_team_luck(
        team_name="Team A (Very Lucky)",
        adjEM=24.5,
        luck=0.18,
        games_remaining=15,
    )

    print(f"\nAdjEM: {analysis.current_adjEM:.1f}")
    print(f"Luck Category: {analysis.luck_category}")
    print(f"Regression Adjustment: {analysis.regression_adjustment:+.1f} points")
    print(f"Luck-Adjusted AdjEM: {analysis.luck_adjusted_adjEM:.1f}")
    print(f"\nBetting Recommendation: {analysis.betting_recommendation}")
    print(f"Edge Magnitude: {analysis.edge_magnitude:.1f} points")
    print(f"Confidence: {analysis.confidence.upper()}")
    print(f"\nReasoning: {analysis.reasoning}")

    # Example: Very unlucky team
    print("\n" + "-" * 70)
    print("\n[TEAM B - Very Unlucky]")
    print("Record: 8-4, Luck = -0.15")
    print("Lost 1-4 in close games (should be ~2.5-2.5)")

    analysis = analyzer.analyze_team_luck(
        team_name="Team B (Very Unlucky)",
        adjEM=22.0,
        luck=-0.15,
        games_remaining=15,
    )

    print(f"\nAdjEM: {analysis.current_adjEM:.1f}")
    print(f"Luck Category: {analysis.luck_category}")
    print(f"Regression Adjustment: {analysis.regression_adjustment:+.1f} points")
    print(f"Luck-Adjusted AdjEM: {analysis.luck_adjusted_adjEM:.1f}")
    print(f"\nBetting Recommendation: {analysis.betting_recommendation}")
    print(f"Edge Magnitude: {analysis.edge_magnitude:.1f} points")
    print(f"Confidence: {analysis.confidence.upper()}")
    print(f"\nReasoning: {analysis.reasoning}")


def demo_matchup_analysis():
    """Demonstrate matchup luck analysis."""
    print_separator("EXAMPLE 2: Matchup Analysis - Duke vs UNC")

    analyzer = LuckRegressionAnalyzer()

    print("\nSCENARIO:")
    print("  Duke: 12-2 record, Luck = +0.18 (very lucky)")
    print("  UNC:  9-5 record, Luck = -0.12 (unlucky)")
    print("  Vegas Line: Duke -7.5")
    print("  Neutral Site")

    matchup = analyzer.analyze_matchup_luck(
        team1_name="Duke",
        team1_adjEM=24.5,
        team1_luck=0.18,
        team2_name="UNC",
        team2_adjEM=22.0,
        team2_luck=-0.12,
        games_remaining=15,
        neutral_site=True,
    )

    print(f"\n{'='*70}")
    print("ANALYSIS RESULTS")
    print(f"{'='*70}")

    print(f"\nRAW PREDICTION (No Luck Adjustment):")
    print(f"  Duke by {matchup.raw_margin:+.1f}")

    print(f"\nLUCK-ADJUSTED PREDICTION:")
    print(f"  Duke by {matchup.luck_adjusted_margin:+.1f}")

    print(f"\nLUCK EDGE: {matchup.luck_edge:+.1f} points")
    print(f"  Duke Luck: {matchup.team1_luck:+.3f} (overvalued)")
    print(f"  UNC Luck: {matchup.team2_luck:+.3f} (undervalued)")

    print(f"\nVS VEGAS LINE: Duke -7.5")
    print(f"  True Line: Duke {matchup.luck_adjusted_margin:+.1f}")
    print(f"  Edge: {7.5 - abs(matchup.luck_adjusted_margin):.1f} points on UNC")

    print(f"\nBETTING RECOMMENDATION:")
    print(f"  {matchup.betting_recommendation}")
    print(f"  Confidence: {matchup.confidence.upper()}")
    print(f"  Expected CLV: +{matchup.expected_clv:.1f} points")

    print(f"\n{'='*70}")
    print("REASONING")
    print(f"{'='*70}")
    print(matchup.reasoning)


def demo_extreme_cases():
    """Demonstrate extreme luck scenarios."""
    print_separator("EXAMPLE 3: Extreme Luck Scenarios")

    # Scenario 1: Both teams very lucky
    print("\n[SCENARIO 1: Both Teams Very Lucky]")
    print("Team X: Luck = +0.20, Team Y: Luck = +0.18")

    edge = calculate_luck_edge(
        team1_adjEM=25.0,
        team1_luck=0.20,
        team2_adjEM=24.0,
        team2_luck=0.18,
    )

    print(f"Luck Edge: {edge:+.1f} points")
    print("Both overvalued, small relative edge")
    print("PASS - Look for better opportunities")

    # Scenario 2: Huge luck differential
    print("\n" + "-" * 70)
    print("\n[SCENARIO 2: Huge Luck Differential]")
    print("Team X: Luck = +0.22 (very lucky)")
    print("Team Y: Luck = -0.18 (very unlucky)")

    edge = calculate_luck_edge(
        team1_adjEM=23.0,
        team1_luck=0.22,
        team2_adjEM=21.0,
        team2_luck=-0.18,
    )

    print(f"Luck Edge: {edge:+.1f} points")
    print("MASSIVE EDGE - Team X overvalued, Team Y undervalued")
    print("STRONG BACK Team Y (Expected CLV: +3-4 points)")

    # Scenario 3: Neutral luck
    print("\n" + "-" * 70)
    print("\n[SCENARIO 3: Neutral Luck]")
    print("Team X: Luck = +0.02, Team Y: Luck = -0.03")

    edge = calculate_luck_edge(
        team1_adjEM=24.0,
        team1_luck=0.02,
        team2_adjEM=22.0,
        team2_luck=-0.03,
    )

    print(f"Luck Edge: {edge:+.1f} points")
    print("No significant luck edge - use normal analysis")


def demo_quick_functions():
    """Demonstrate quick helper functions."""
    print_separator("EXAMPLE 4: Quick Helper Functions")

    print("\n[Quick Luck Check]")
    print("\nTeam: AdjEM = 26.0, Luck = +0.19")
    result = quick_luck_check(adjEM=26.0, luck=0.19)
    print(f"Result: {result}")

    print("\n" + "-" * 70)
    print("\n[Quick Edge Calculation]")
    print("Duke (24.5, +0.18) vs UNC (22.0, -0.12)")
    edge = calculate_luck_edge(24.5, 0.18, 22.0, -0.12)
    print(f"Luck Edge: {edge:+.1f} points")


def demo_real_world_workflow():
    """Demonstrate real-world betting workflow."""
    print_separator("EXAMPLE 5: Real-World Betting Workflow")

    analyzer = LuckRegressionAnalyzer()

    print("\n[MORNING - Pre-Game Analysis]")
    print("You analyze all games before Vegas posts lines...")

    # Analyze multiple teams
    teams = [
        ("Duke", 24.5, 0.18),
        ("Kansas", 26.0, 0.21),
        ("UNC", 22.0, -0.12),
        ("Virginia", 20.5, -0.16),
        ("Gonzaga", 25.5, 0.03),
    ]

    print("\nTeam Rankings (Luck-Adjusted):")
    print(f"{'TEAM':<15} {'AdjEM':>8} {'Luck':>8} {'Adjusted':>8} {'Recommendation':<15}")
    print("-" * 70)

    for team_name, adjEM, luck in teams:
        analysis = analyzer.analyze_team_luck(team_name, adjEM, luck, 15)
        print(
            f"{team_name:<15} {adjEM:>8.1f} {luck:>+8.3f} "
            f"{analysis.luck_adjusted_adjEM:>8.1f} {analysis.betting_recommendation:<15}"
        )

    print("\n[AFTERNOON - Vegas Lines Post]")
    print("Duke -7.5 vs UNC")
    print("Kansas -9.5 vs Virginia")

    print("\n[EDGE DETECTION]")

    # Matchup 1: Duke vs UNC
    m1 = analyzer.analyze_matchup_luck(
        "Duke", 24.5, 0.18, "UNC", 22.0, -0.12, neutral_site=True
    )

    print(f"\nDuke vs UNC:")
    print(f"  Luck-adjusted line: Duke {m1.luck_adjusted_margin:+.1f}")
    print(f"  Vegas line: Duke -7.5")
    print(f"  Edge: {7.5 - abs(m1.luck_adjusted_margin):.1f} points on UNC")
    print(f"  {m1.betting_recommendation}")

    # Matchup 2: Kansas vs Virginia
    m2 = analyzer.analyze_matchup_luck(
        "Kansas", 26.0, 0.21, "Virginia", 20.5, -0.16, neutral_site=True
    )

    print(f"\nKansas vs Virginia:")
    print(f"  Luck-adjusted line: Kansas {m2.luck_adjusted_margin:+.1f}")
    print(f"  Vegas line: Kansas -9.5")
    print(f"  Edge: {abs(m2.luck_adjusted_margin) - 9.5:+.1f} points")
    print(f"  {m2.betting_recommendation}")

    print("\n[BETTING DECISIONS]")
    print("BET 1: UNC +7.5 (Expected CLV: +3.0)")
    print("BET 2: Virginia +9.5 (Expected CLV: +2.5)")


def main():
    """Run all demos."""
    print("\n" + "=" * 70)
    print(" LUCK REGRESSION ANALYSIS - LIVE DEMO")
    print("=" * 70)
    print("\n This demo shows how luck regression identifies betting edges")
    print(" by detecting overvalued (lucky) and undervalued (unlucky) teams.")

    demo_single_team_analysis()
    demo_matchup_analysis()
    demo_extreme_cases()
    demo_quick_functions()
    demo_real_world_workflow()

    print("\n" + "=" * 70)
    print(" DEMO COMPLETE")
    print("=" * 70)
    print("\n Key Takeaways:")
    print(" 1. Lucky teams (Luck > 0.15) are overvalued by 2-3 points")
    print(" 2. Unlucky teams (Luck < -0.15) are undervalued by 2-3 points")
    print(" 3. Luck always regresses to the mean over next 10-20 games")
    print(" 4. This creates 6-10 point edges when exploited correctly")
    print("\n LUCK REGRESSION IS NOW INTEGRATED INTO YOUR SYSTEM!")
    print(" Run: uv run python scripts/analysis/kenpom_pregame_analyzer.py")
    print("=" * 70)


if __name__ == "__main__":
    main()
