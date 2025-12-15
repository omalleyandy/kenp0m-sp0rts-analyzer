#!/usr/bin/env python3
"""Basic usage examples for the KenPom Sports Analyzer.

This script demonstrates the fundamental operations available
through the kenpompy library and the kenp0m_sp0rts_analyzer package.

Requirements:
    - KenPom subscription
    - Set environment variables:
        - KENPOM_EMAIL
        - KENPOM_PASSWORD
"""

import os
from pprint import pprint

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Verify credentials are set
if not os.getenv("KENPOM_EMAIL") or not os.getenv("KENPOM_PASSWORD"):
    print("Error: Please set KENPOM_EMAIL and KENPOM_PASSWORD environment variables")
    print("Create a .env file with:")
    print("  KENPOM_EMAIL=your-email@example.com")
    print("  KENPOM_PASSWORD=your-password")
    exit(1)


def example_direct_kenpompy_usage():
    """Demonstrate direct usage of the kenpompy library."""
    print("\n" + "=" * 60)
    print("Direct kenpompy Usage")
    print("=" * 60)

    from kenpompy.misc import get_pomeroy_ratings
    from kenpompy.summary import get_efficiency, get_fourfactors
    from kenpompy.team import get_schedule, get_scouting_report
    from kenpompy.utils import login

    # Step 1: Authenticate
    email = os.getenv("KENPOM_EMAIL")
    password = os.getenv("KENPOM_PASSWORD")
    browser = login(email, password)
    print("Successfully logged in to KenPom!")

    # Step 2: Get efficiency ratings
    print("\n--- Top 10 Teams by Efficiency ---")
    efficiency = get_efficiency(browser)
    print(efficiency.head(10).to_string())

    # Step 3: Get Four Factors
    print("\n--- Four Factors (Top 5) ---")
    four_factors = get_fourfactors(browser)
    print(four_factors.head(5).to_string())

    # Step 4: Get Pomeroy ratings
    print("\n--- Pomeroy Ratings (Top 5) ---")
    ratings = get_pomeroy_ratings(browser)
    print(ratings.head(5).to_string())

    # Step 5: Get a team's schedule
    print("\n--- Duke Schedule ---")
    duke_schedule = get_schedule(browser, team="Duke")
    print(duke_schedule.head(10).to_string())

    # Step 6: Get scouting report
    print("\n--- Duke Scouting Report ---")
    duke_report = get_scouting_report(browser, team="Duke")
    pprint(duke_report)

    return browser


def example_client_wrapper_usage():
    """Demonstrate usage of the KenPomClient wrapper."""
    print("\n" + "=" * 60)
    print("KenPomClient Wrapper Usage")
    print("=" * 60)

    from kenp0m_sp0rts_analyzer import KenPomClient

    # Initialize client (auto-authenticates)
    client = KenPomClient()
    print(f"Current season: {client.current_season}")

    # Get efficiency data
    print("\n--- Top 10 Teams ---")
    efficiency = client.get_efficiency()
    print(efficiency.head(10)[["Team", "Conf", "Rk", "AdjEM", "AdjO", "AdjD"]].to_string())

    # Compare two teams
    print("\n--- Duke vs North Carolina ---")
    comparison = client.compare_teams("Duke", "UNC")
    print(comparison.to_string())

    # Get team rank
    duke_rank = client.get_team_rank("Duke")
    print(f"\nDuke's current KenPom rank: {duke_rank}")

    # Get home court advantage
    print("\n--- Home Court Advantage (Top 10) ---")
    hca = client.get_hca()
    print(hca.head(10).to_string())

    return client


def example_analysis_functions():
    """Demonstrate the analysis functions."""
    print("\n" + "=" * 60)
    print("Analysis Functions")
    print("=" * 60)

    from kenp0m_sp0rts_analyzer import (
        KenPomClient,
        analyze_matchup,
        get_conference_standings,
    )

    client = KenPomClient()

    # Analyze a matchup
    print("\n--- Matchup Analysis: Duke vs North Carolina ---")
    matchup = analyze_matchup(
        team1="Duke",
        team2="North Carolina",
        neutral_site=True,
        client=client,
    )
    print(f"  Team 1: {matchup.team1} (Rank #{matchup.team1_rank})")
    print(f"  Team 2: {matchup.team2} (Rank #{matchup.team2_rank})")
    print(f"  Efficiency Margin Diff: {matchup.em_difference}")
    print(f"  Predicted Winner: {matchup.predicted_winner}")
    print(f"  Predicted Margin: {matchup.predicted_margin} points")
    print(f"  Predicted Total: {matchup.predicted_total} points")
    print(f"  Expected Tempo: {matchup.expected_tempo} possessions")

    # Get conference standings
    print("\n--- ACC Standings (KenPom) ---")
    acc = get_conference_standings("ACC", client=client)
    print(acc[["Team", "Rk", "AdjEM", "AdjO", "AdjD"]].to_string())

    # Find value games
    print("\n--- SEC Standings (KenPom) ---")
    sec = get_conference_standings("SEC", client=client)
    print(sec[["Team", "Rk", "AdjEM", "AdjO", "AdjD"]].to_string())


if __name__ == "__main__":
    print("KenPom Sports Analyzer - Basic Usage Examples")
    print("=" * 60)

    # Run examples
    example_direct_kenpompy_usage()
    example_client_wrapper_usage()
    example_analysis_functions()

    print("\n" + "=" * 60)
    print("Examples completed!")
    print("=" * 60)
