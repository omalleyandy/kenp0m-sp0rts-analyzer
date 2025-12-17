"""Batch analysis of all NCAA Basketball games from December 16, 2025.

This script analyzes all major matchups from today's schedule.
"""

import os
from dotenv import load_dotenv
from kenp0m_sp0rts_analyzer.analysis import analyze_matchup
from kenp0m_sp0rts_analyzer.api_client import KenPomAPI
from kenp0m_sp0rts_analyzer.four_factors_matchup import FourFactorsMatchup
from kenp0m_sp0rts_analyzer.point_distribution_analysis import PointDistributionAnalyzer
from kenp0m_sp0rts_analyzer.defensive_analysis import DefensiveAnalyzer
from kenp0m_sp0rts_analyzer.size_athleticism_analysis import SizeAthleticismAnalyzer
from kenp0m_sp0rts_analyzer.experience_chemistry_analysis import ExperienceChemistryAnalyzer

# Load environment variables
load_dotenv()

# Today's games (December 16, 2025)
GAMES = [
    # Already analyzed:
    # ("Louisville", "Tennessee", -2.5, "ESPN"),
    # ("Lipscomb", "Duke", -33.5, "ACC Network"),
    # ("Toledo", "Michigan St.", -23.5, "Peacock"),

    # Remaining games:
    ("East Tennessee St.", "North Carolina", -15.5, "ACC Network"),
    ("Butler", "Connecticut", -15.5, "Peacock"),
    ("Pacific", "BYU", -23.5, "Peacock"),
    ("DePaul", "St. John's", -19.5, "ESPN"),
    ("Northern Colorado", "Texas Tech", -24.5, "ESPN+"),
    ("Abilene Christian", "Arizona", -33.5, "ESPN+"),
    ("Queens", "Arkansas", -25.5, "SEC Network"),
    ("Towson", "Kansas", -18.5, "ESPN2"),
]

SEASON = 2025

def print_separator():
    print("\n" + "=" * 80)
    print("=" * 80 + "\n")

def analyze_game(team1, team2, spread, network):
    """Analyze a single game and return key insights."""
    print(f"\nAnalyzing: {team1} @ {team2} (Spread: {team2} {spread}) [{network}]")
    print("-" * 80)

    try:
        # Initialize analyzers
        api = KenPomAPI()

        # Basic analysis
        basic = analyze_matchup(team1, team2, SEASON, neutral_site=False, home_team=team2)

        # Four Factors
        ff = FourFactorsMatchup(api)
        ff_analysis = ff.analyze_matchup(team1, team2, SEASON)

        # Point Distribution
        pd = PointDistributionAnalyzer(api)
        pd_analysis = pd.analyze_matchup(team1, team2, SEASON)

        # Defense
        defense = DefensiveAnalyzer(api)
        def_analysis = defense.analyze_matchup(team1, team2, SEASON)

        # Size
        size = SizeAthleticismAnalyzer(api)
        size_analysis = size.analyze_matchup(team1, team2, SEASON)

        # Experience
        exp = ExperienceChemistryAnalyzer(api)
        exp_analysis = exp.analyze_matchup(team1, team2, SEASON)

        # Count dimensional wins
        team1_wins = 0
        team2_wins = 0

        # Basic efficiency
        if basic.em_difference > 0:
            team1_wins += 1
        else:
            team2_wins += 1

        # Four Factors
        if ff_analysis.overall_advantage == team1.lower():
            team1_wins += 1
        else:
            team2_wins += 1

        # Defense
        if def_analysis.better_defense == team1:
            team1_wins += 1
        else:
            team2_wins += 1

        # Size
        if size_analysis.better_size_team == team1:
            team1_wins += 1
        else:
            team2_wins += 1

        # Experience
        if exp_analysis.better_intangibles == team1:
            team1_wins += 1
        elif exp_analysis.better_intangibles == team2:
            team2_wins += 1

        # Print results
        print(f"\nRESULTS:")
        print(f"  System Prediction: {basic.predicted_winner} by {basic.predicted_margin:.1f}")
        print(f"  Vegas Line: {team2} {spread}")
        print(f"  Dimensional Battles: {team2} {team2_wins}-{team1_wins} {team1}")

        # Four Factors highlights
        print(f"\n  Four Factors Advantage: {ff_analysis.overall_advantage.upper()}")
        print(f"  Advantage Score: {ff_analysis.advantage_score:+.2f} (+ = {team1})")

        # Key advantages
        if pd_analysis.three_point_advantage > 2.0:
            print(f"  -> {team1} has STRONG 3PT advantage ({pd_analysis.three_point_advantage:+.1f}%)")
        elif pd_analysis.three_point_advantage < -2.0:
            print(f"  -> {team2} has STRONG 3PT advantage ({pd_analysis.three_point_advantage:+.1f}%)")

        if size_analysis.overall_height_advantage > 1.5:
            print(f"  -> {team1} has BIG size advantage ({size_analysis.overall_height_advantage:+.1f}\")")
        elif size_analysis.overall_height_advantage < -1.5:
            print(f"  -> {team2} has BIG size advantage ({size_analysis.overall_height_advantage:+.1f}\")")

        # Betting recommendation
        system_spread = basic.predicted_margin if basic.predicted_winner == team2 else -basic.predicted_margin
        spread_diff = abs(system_spread - spread)

        print(f"\n  BETTING:")
        if spread_diff < 2.0:
            print(f"    -> System agrees with Vegas (diff: {spread_diff:.1f})")
            print(f"    -> PASS - no edge")
        elif spread_diff >= 5.0:
            if system_spread > spread:
                print(f"    -> System sees {team2} stronger (diff: {spread_diff:.1f})")
                print(f"    -> VALUE: {team2} {spread}")
            else:
                print(f"    -> System sees {team1} stronger (diff: {spread_diff:.1f})")
                print(f"    -> VALUE: {team1} +{abs(spread)}")
        else:
            print(f"    -> Slight disagreement (diff: {spread_diff:.1f})")
            print(f"    -> Proceed with caution")

        return {
            'team1': team1,
            'team2': team2,
            'spread': spread,
            'system_winner': basic.predicted_winner,
            'system_margin': basic.predicted_margin,
            'team1_wins': team1_wins,
            'team2_wins': team2_wins,
            'ff_advantage': ff_analysis.overall_advantage,
            'value': spread_diff >= 5.0
        }

    except Exception as e:
        print(f"  [ERROR] {e}")
        return None

def main():
    print("=" * 80)
    print("NCAA BASKETBALL - DECEMBER 16, 2025 MATCHUP ANALYSIS")
    print("=" * 80)

    api_key = os.getenv("KENPOM_API_KEY")
    if not api_key:
        print("\n[ERROR] KENPOM_API_KEY not found")
        return

    print(f"\n[OK] API key loaded")
    print(f"[OK] Analyzing {len(GAMES)} remaining games...\n")

    results = []

    for i, (team1, team2, spread, network) in enumerate(GAMES, 1):
        print(f"\n[{i}/{len(GAMES)}] ", end="")
        result = analyze_game(team1, team2, spread, network)
        if result:
            results.append(result)
        print_separator()

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY - VALUE PICKS")
    print("=" * 80)

    value_games = [r for r in results if r and r.get('value')]

    if value_games:
        print(f"\nFound {len(value_games)} games with betting value:\n")
        for r in value_games:
            print(f"  {r['team1']} @ {r['team2']} ({r['team2']} {r['spread']})")
            print(f"    -> System: {r['system_winner']} by {r['system_margin']:.1f}")
            print(f"    -> Value opportunity detected\n")
    else:
        print("\nNo significant value opportunities detected.")
        print("All lines are efficient relative to system predictions.\n")

    print("=" * 80)
    print(f"[OK] Analysis complete! Analyzed {len(results)} games.")
    print("=" * 80)

if __name__ == "__main__":
    main()
