#!/usr/bin/env python3
"""Matchup analysis example for predicting game outcomes.

This script demonstrates how to use the KenPom data to analyze
head-to-head matchups and predict game outcomes based on
efficiency metrics.

Requirements:
    - KenPom subscription
    - Set environment variables:
        - KENPOM_EMAIL
        - KENPOM_PASSWORD
"""

import argparse
import os
import sys

from dotenv import load_dotenv


def main():
    """Run matchup analysis between two teams."""
    load_dotenv()

    parser = argparse.ArgumentParser(description="Analyze a matchup between two teams")
    parser.add_argument("team1", help="First team name (e.g., 'Duke')")
    parser.add_argument("team2", help="Second team name (e.g., 'North Carolina')")
    parser.add_argument(
        "--neutral",
        action="store_true",
        default=True,
        help="Game is at neutral site (default)",
    )
    parser.add_argument(
        "--home",
        type=str,
        help="Home team (if not neutral site)",
    )
    parser.add_argument(
        "--season",
        type=int,
        help="Season year (defaults to current)",
    )

    args = parser.parse_args()

    # Verify credentials
    if not os.getenv("KENPOM_EMAIL") or not os.getenv("KENPOM_PASSWORD"):
        print("Error: KENPOM_EMAIL and KENPOM_PASSWORD environment variables required")
        sys.exit(1)

    from kenp0m_sp0rts_analyzer import KenPomClient, analyze_matchup

    # Initialize client
    print("Connecting to KenPom...")
    client = KenPomClient()

    # Determine site
    neutral_site = args.neutral and not args.home
    home_team = args.home if args.home else None

    # Run analysis
    print(f"\nAnalyzing: {args.team1} vs {args.team2}")
    if neutral_site:
        print("Location: Neutral site")
    else:
        print(f"Location: {home_team} home court")

    try:
        matchup = analyze_matchup(
            team1=args.team1,
            team2=args.team2,
            season=args.season,
            neutral_site=neutral_site,
            home_team=home_team,
            client=client,
        )
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Display results
    print("\n" + "=" * 50)
    print("MATCHUP ANALYSIS")
    print("=" * 50)

    print(f"\n{matchup.team1} (#{matchup.team1_rank})")
    print(f"  Adjusted Efficiency Margin: {matchup.team1_adj_em:+.2f}")
    print(f"  Adjusted Tempo: {matchup.team1_tempo:.1f}")

    print(f"\n{matchup.team2} (#{matchup.team2_rank})")
    print(f"  Adjusted Efficiency Margin: {matchup.team2_adj_em:+.2f}")
    print(f"  Adjusted Tempo: {matchup.team2_tempo:.1f}")

    print("\n" + "-" * 50)
    print("PREDICTION")
    print("-" * 50)

    print(f"\n  Efficiency Margin Difference: {matchup.em_difference:+.2f}")
    print(f"  Expected Tempo: {matchup.expected_tempo:.1f} possessions")
    print(f"  Pace Advantage: {matchup.pace_advantage}")

    print(f"\n  >>> PREDICTED WINNER: {matchup.predicted_winner}")
    print(f"  >>> PREDICTED MARGIN: {matchup.predicted_margin:.1f} points")
    print(f"  >>> PREDICTED TOTAL:  {matchup.predicted_total:.1f} points")

    # Calculate implied scores
    if matchup.predicted_total:
        avg_score = matchup.predicted_total / 2
        winner_score = avg_score + (matchup.predicted_margin / 2)
        loser_score = avg_score - (matchup.predicted_margin / 2)

        if matchup.predicted_winner == matchup.team1:
            t1_score, t2_score = winner_score, loser_score
        else:
            t1_score, t2_score = loser_score, winner_score

        print(f"\n  Projected Score: {matchup.team1} {t1_score:.0f}, {matchup.team2} {t2_score:.0f}")


if __name__ == "__main__":
    main()
