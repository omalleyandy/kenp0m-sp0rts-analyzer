"""Comprehensive analysis of ALL NCAA Division I Basketball games from December 17, 2025.

Analyzes 50+ games from the full schedule.
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

# December 17, 2025 - FULL SCHEDULE
# Format: (away_team, home_team, spread (home team), network)
# Note: Spread = 0.0 means no line available
GAMES = [
    # Early Games
    ("Kansas Christian", "Central Arkansas", 0.0, "ESPN+"),
    ("Southwestern Christian", "Rice", 0.0, "ESPN+"),
    ("Champion Christian", "Arkansas Pine Bluff", 0.0, "None"),

    # Afternoon/Evening Games (5:00 PM - 7:30 PM ET)
    ("Northern Iowa", "Illinois Chicago", -4.5, "ESPN+"),  # UIC -> Illinois Chicago
    ("Stonehill", "New Hampshire", 0.0, "ESPN+"),
    ("Saint Francis", "Florida", -28.5, "SEC Network"),
    ("Creighton", "Xavier", -3.5, "FS1"),
    ("Vanderbilt", "Memphis", -2.5, "ESPN2"),
    ("Mid-Atlantic Christian", "North Carolina Central", 0.0, "ESPN+"),
    ("UC Santa Barbara", "Green Bay", 0.0, "ESPN+"),
    ("Eastern Kentucky", "Jacksonville St.", 0.0, "ESPN+"),
    ("Albany", "Stony Brook", 0.0, "None"),  # UAlbany -> Albany
    ("The Citadel", "Charleston", 0.0, "None"),
    ("Southern Wesleyan", "South Carolina Upstate", 0.0, "ESPN+"),
    ("Brewton-Parker", "West Georgia", 0.0, "ESPN+"),
    ("Mercer", "UCF", -12.5, "ESPN+"),
    ("Olivet", "Central Michigan", 0.0, "ESPN+"),
    ("Jacksonville", "Florida A&M", 0.0, "None"),
    ("Chattanooga", "Bellarmine", 0.0, "ESPN+"),
    ("Maryland Eastern Shore", "Wagner", 0.0, "None"),
    ("Siena", "Vermont", 0.0, "ESPN+"),
    ("Presbyterian", "East Carolina", 0.0, "ESPN+"),
    ("Quinnipiac", "Monmouth", 0.0, "None"),
    ("Alabama St.", "Cincinnati", 0.0, "ESPN+"),
    ("Youngstown St.", "Robert Morris", 0.0, "ESPN+"),
    ("Oakland", "Northern Kentucky", 0.0, "ESPN+"),
    ("Longwood", "Wake Forest", -18.5, "ACC Extra"),
    ("Mercyhurst", "Syracuse", -26.5, "ACC Extra"),
    ("Binghamton", "Pittsburgh", -21.5, "ACC Extra"),
    ("Texas Southern", "North Carolina St.", -24.5, "ACC Extra"),  # NC State -> North Carolina St.
    ("Richmond", "Elon", 0.0, "None"),
    ("Wofford", "Wichita St.", 0.0, "ESPN+"),
    ("Cleveland St.", "UAB", 0.0, "ESPN+"),
    ("Louisiana Tech", "Tulane", 0.0, "ESPN+"),
    ("Kennesaw St.", "Middle Tennessee", 0.0, "ESPN+"),
    ("Houston Christian", "Nicholls", 0.0, "ESPN+"),
    ("South Alabama", "Louisiana Monroe", 0.0, "ESPN+"),  # UL Monroe -> Louisiana Monroe

    # Late Evening Games (8:00 PM - 10:00 PM ET)
    ("South Florida", "Alabama", -19.5, "SECN+"),
    ("Alabama A&M", "Mississippi", 0.0, "SECN+"),  # Ole Miss -> Mississippi
    ("Weber St.", "Utah Valley", 0.0, "ESPN+"),
    ("Bethune Cookman", "Saint Louis", 0.0, "ESPN+"),  # Bethune-Cookman -> Bethune Cookman
    ("Dakota St.", "South Dakota", 0.0, "None"),
    ("James Madison", "Old Dominion", 0.0, "ESPN+"),
    ("Georgetown", "Marquette", -8.5, "FS1"),
    ("Arkansas St.", "Texas St.", 0.0, "ESPN+"),
    ("Campbell", "Gonzaga", -32.5, "ESPN+"),
    ("Montana Tech", "Montana", 0.0, "ESPN+"),
    ("UTSA", "Southern California", -8.5, "BTN"),  # USC -> Southern California
    ("Seattle", "UC Davis", 0.0, "ESPN+"),  # Seattle U -> Seattle
    ("North Texas", "Santa Clara", 0.0, "None"),
    ("Portland St.", "Colorado", 0.0, "ESPN+"),
    ("Texas A&M Corpus Chris", "Stephen F. Austin", 0.0, "ESPN+"),  # Texas A&M-CC -> Texas A&M Corpus Chris
    ("Sam Houston St.", "Oregon St.", 0.0, "ESPN+"),
    ("Eastern Washington", "Washington St.", 0.0, "YouTube"),
    ("Air Force", "San Diego St.", -12.5, "CBSSN"),
    ("UT Arlington", "Stanford", 0.0, "ACC Extra"),
    ("Loyola Chicago", "San Francisco", 0.0, "ESPN+"),
]

SEASON = 2025

def print_separator():
    print("\n" + "=" * 80)

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
        if spread != 0.0:
            print(f"  Vegas Line: {team2} {spread:+.1f}")
        else:
            print(f"  Vegas Line: No line available")
        print(f"  Dimensional Battles: {team2} {team2_wins}-{team1_wins} {team1}")

        # Four Factors highlights
        print(f"\n  Four Factors Advantage: {ff_analysis.overall_advantage.upper()}")

        # Key advantages
        if abs(pd_analysis.three_point_advantage) > 2.0:
            adv_team = team1 if pd_analysis.three_point_advantage > 0 else team2
            print(f"  -> {adv_team} has 3PT advantage ({pd_analysis.three_point_advantage:+.1f}%)")

        if abs(size_analysis.overall_height_advantage) > 1.5:
            adv_team = team1 if size_analysis.overall_height_advantage > 0 else team2
            print(f"  -> {adv_team} has size advantage ({size_analysis.overall_height_advantage:+.1f}\")")

        # Betting recommendation (only if spread available)
        result = {
            'team1': team1,
            'team2': team2,
            'spread': spread,
            'system_winner': basic.predicted_winner,
            'system_margin': basic.predicted_margin,
            'team1_wins': team1_wins,
            'team2_wins': team2_wins,
            'ff_advantage': ff_analysis.overall_advantage,
            'value': False,
            'spread_diff': 0.0
        }

        if spread != 0.0:
            # When team2 (home) wins, spread is negative (e.g., Wake Forest -12.7)
            # When team1 (away) wins, spread is positive (e.g., +12.7 for away team)
            system_spread = -basic.predicted_margin if basic.predicted_winner == team2 else basic.predicted_margin
            spread_diff = abs(system_spread - spread)
            result['spread_diff'] = spread_diff

            print(f"\n  BETTING:")
            if spread_diff < 2.0:
                print(f"    -> System agrees with Vegas (diff: {spread_diff:.1f})")
                print(f"    -> PASS - no edge")
            elif spread_diff >= 5.0:
                result['value'] = True
                if system_spread > spread:
                    print(f"    -> System sees {team2} stronger (diff: {spread_diff:.1f})")
                    print(f"    -> VALUE: {team2} {spread:+.1f}")
                else:
                    print(f"    -> System sees {team1} stronger (diff: {spread_diff:.1f})")
                    print(f"    -> VALUE: {team1} +{abs(spread):.1f}")
            else:
                print(f"    -> Slight disagreement (diff: {spread_diff:.1f})")
                print(f"    -> Proceed with caution")

        return result

    except Exception as e:
        print(f"  [ERROR] {e}")
        return None

def main():
    print("=" * 80)
    print("NCAA DIVISION I BASKETBALL - FULL SCHEDULE ANALYSIS")
    print("December 17, 2025 - ALL GAMES")
    print("=" * 80)

    api_key = os.getenv("KENPOM_API_KEY")
    if not api_key:
        print("\n[ERROR] KENPOM_API_KEY not found")
        return

    print(f"\n[OK] API key loaded")
    print(f"[OK] Analyzing {len(GAMES)} games from tomorrow's full schedule...")
    print(f"[OK] This may take several minutes...\n")

    results = []
    errors = []

    for i, (team1, team2, spread, network) in enumerate(GAMES, 1):
        print(f"\n[{i}/{len(GAMES)}] ", end="")
        result = analyze_game(team1, team2, spread, network)
        if result:
            results.append(result)
        else:
            errors.append((team1, team2))
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

    # Moderate disagreements
    moderate_games = [r for r in results if r and 2.0 <= r.get('spread_diff', 0) < 5.0]
    if moderate_games:
        print("\n" + "-" * 80)
        print("MODERATE DISAGREEMENTS (2-5 point edge - proceed with caution):")
        print("-" * 80 + "\n")
        for r in moderate_games:
            print(f"  {r['team1']} @ {r['team2']} ({r['team2']} {r['spread']:+.1f})")
            print(f"    -> System: {r['system_winner']} by {r['system_margin']:.1f}")
            print(f"    -> Edge: {r['spread_diff']:.1f} points\n")

    # Errors
    if errors:
        print("\n" + "-" * 80)
        print(f"ERRORS - {len(errors)} games could not be analyzed:")
        print("-" * 80 + "\n")
        for team1, team2 in errors:
            print(f"  {team1} @ {team2}")

    # Statistics
    print("\n" + "=" * 80)
    print("ANALYSIS STATISTICS")
    print("=" * 80)
    print(f"  Total games scheduled: {len(GAMES)}")
    print(f"  Successfully analyzed: {len(results)}")
    print(f"  Errors/team not found: {len(errors)}")
    print(f"  Games with Vegas lines: {len([r for r in results if r['spread'] != 0.0])}")
    print(f"  Value opportunities (5+ edge): {len(value_games)}")
    print(f"  Moderate edges (2-5): {len(moderate_games)}")
    print("=" * 80)

if __name__ == "__main__":
    main()
