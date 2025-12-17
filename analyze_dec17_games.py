"""Batch analysis of key NCAA Basketball games from December 17, 2025.

Focus on ranked teams, major conference matchups, and competitive games.
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

# December 17, 2025 - Key Games
# Format: (away_team, home_team, spread (home team), network)
GAMES = [
    # RANKED TEAMS
    ("Vanderbilt", "Memphis", -2.5, "ESPN2"),  # #13 Vanderbilt @ Memphis
    ("Saint Francis", "Florida", -28.5, "SEC Network"),  # @ #23 Florida
    ("South Florida", "Alabama", -19.5, "SECN+"),  # @ #16 Alabama
    ("Campbell", "Gonzaga", -32.5, "ESPN+"),  # @ #7 Gonzaga

    # BIG EAST MATCHUPS
    ("Creighton", "Xavier", -3.5, "FS1"),  # Big East battle
    ("Georgetown", "Marquette", -8.5, "FS1"),  # Big East rivalry

    # ACC MATCHUPS
    ("Longwood", "Wake Forest", -18.5, "ACC Extra"),
    ("Mercyhurst", "Syracuse", -26.5, "ACC Extra"),
    ("Binghamton", "Pittsburgh", -21.5, "ACC Extra"),
    ("Texas Southern", "NC State", -24.5, "ACC Extra"),

    # OTHER COMPETITIVE GAMES
    ("Northern Iowa", "UIC", -4.5, "ESPN+"),  # MVC vs Horizon
    ("Mercer", "UCF", -12.5, "ESPN+"),  # SoCon vs Big 12
    ("UTSA", "USC", -8.5, "BTN"),  # Conference USA vs Big Ten
    ("Air Force", "San Diego State", -12.5, "CBSSN"),  # Mountain West
]

SEASON = 2025

def print_separator():
    print("\n" + "=" * 80)
    print("=" * 80 + "\n")

def analyze_game(team1, team2, spread, network):
    """Analyze a single game and return key insights."""
    print(f"\nAnalyzing: {team1} @ {team2} (Spread: {team2} {spread:+.1f}) [{network}]")
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
        print(f"  Vegas Line: {team2} {spread:+.1f}")
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
        # Spread calculation logic:
        # - Vegas spread is from HOME team perspective (negative = home favored)
        # - System spread: negative if home wins, positive if away wins
        # - spread_diff = system_spread - vegas_spread (KEEP THE SIGN!)
        #   - Positive diff: System sees home team WEAKER → Value on underdog
        #   - Negative diff: System sees home team STRONGER → Value on favorite
        system_spread = -basic.predicted_margin if basic.predicted_winner == team2 else basic.predicted_margin
        spread_diff = system_spread - spread  # Don't use abs() - direction matters!

        print(f"\n  BETTING:")
        print(f"    System: {system_spread:+.1f} | Vegas: {spread:+.1f} | Diff: {spread_diff:+.1f}")

        if abs(spread_diff) < 2.0:
            print(f"    -> System agrees with Vegas")
            print(f"    -> PASS - no edge")
        elif abs(spread_diff) >= 5.0:
            if spread_diff > 0:
                # System sees home team weaker than Vegas → Take underdog
                print(f"    -> System sees {team2} WEAKER than Vegas (won't cover)")
                print(f"    -> VALUE: {team1} {abs(spread):+.1f} (underdog)")
            else:
                # System sees home team stronger than Vegas → Take favorite
                print(f"    -> System sees {team2} STRONGER than Vegas (will cover)")
                print(f"    -> VALUE: {team2} {spread:+.1f} (favorite)")
        else:
            print(f"    -> Slight disagreement ({abs(spread_diff):.1f} pts)")
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
            'value': abs(spread_diff) >= 5.0,
            'spread_diff': abs(spread_diff)
        }

    except Exception as e:
        print(f"  [ERROR] {e}")
        return None

def main():
    print("=" * 80)
    print("NCAA BASKETBALL - DECEMBER 17, 2025 MATCHUP ANALYSIS")
    print("=" * 80)

    api_key = os.getenv("KENPOM_API_KEY")
    if not api_key:
        print("\n[ERROR] KENPOM_API_KEY not found")
        return

    print(f"\n[OK] API key loaded")
    print(f"[OK] Analyzing {len(GAMES)} key games from tomorrow's schedule...\n")

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
        print(f"\nFound {len(value_games)} games with significant betting value (5+ point edge):\n")
        for r in value_games:
            print(f"  {r['team1']} @ {r['team2']} ({r['team2']} {r['spread']:+.1f})")
            print(f"    -> System: {r['system_winner']} by {r['system_margin']:.1f}")
            print(f"    -> Edge: {r['spread_diff']:.1f} points")
            print(f"    -> Value opportunity detected\n")
    else:
        print("\nNo significant value opportunities detected (5+ point edge).")
        print("All lines appear efficient relative to system predictions.\n")

    # Interesting games (2-5 point edge)
    moderate_games = [r for r in results if r and 2.0 <= r.get('spread_diff', 0) < 5.0]
    if moderate_games:
        print("\n" + "-" * 80)
        print("MODERATE DISAGREEMENTS (2-5 point edge - proceed with caution):")
        print("-" * 80 + "\n")
        for r in moderate_games:
            print(f"  {r['team1']} @ {r['team2']} ({r['team2']} {r['spread']:+.1f})")
            print(f"    -> System: {r['system_winner']} by {r['system_margin']:.1f}")
            print(f"    -> Edge: {r['spread_diff']:.1f} points\n")

    print("=" * 80)
    print(f"[OK] Analysis complete! Analyzed {len(results)} games.")
    print("=" * 80)

if __name__ == "__main__":
    main()
