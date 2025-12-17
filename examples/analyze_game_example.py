"""NCAA Basketball Game Analysis Example.

This script demonstrates how to perform comprehensive matchup analysis using the
5 fully-implemented analyzer modules:
1. Basic Efficiency Analysis
2. Four Factors (Dean Oliver Framework)
3. Scoring Styles & Point Distribution
4. Defensive Schemes & Matchups
5. Size & Athleticism
6. Experience & Chemistry

Example Game (December 16, 2025):
- #11 Louisville @ #20 Tennessee
- Spread: Tennessee -2.5
- Network: ESPN
- Location: Food City Center, Knoxville, TN

Usage:
    # Run with default teams (Louisville vs Tennessee)
    uv run python examples/analyze_game_example.py

    # Or modify TEAM1, TEAM2, SEASON variables below for any matchup

Requirements:
    - KENPOM_API_KEY environment variable set
    - KenPom API access (purchase from https://kenpom.com/register-api.php)
"""

import os
from pathlib import Path

from dotenv import load_dotenv

from kenp0m_sp0rts_analyzer.analysis import analyze_matchup
from kenp0m_sp0rts_analyzer.api_client import KenPomAPI
from kenp0m_sp0rts_analyzer.defensive_analysis import DefensiveAnalyzer
from kenp0m_sp0rts_analyzer.experience_chemistry_analysis import (
    ExperienceChemistryAnalyzer,
)
from kenp0m_sp0rts_analyzer.four_factors_matchup import FourFactorsMatchup
from kenp0m_sp0rts_analyzer.point_distribution_analysis import (
    PointDistributionAnalyzer,
)
from kenp0m_sp0rts_analyzer.size_athleticism_analysis import SizeAthleticismAnalyzer

# Load environment variables
load_dotenv()

# Game parameters
TEAM1 = "Toledo"
TEAM2 = "Michigan St."
SEASON = 2025


def print_header(text: str, width: int = 80, char: str = "=") -> None:
    """Print formatted header."""
    print(f"\n{char * width}")
    print(text.center(width))
    print(char * width)


def print_section(text: str, width: int = 80, char: str = "-") -> None:
    """Print formatted section header."""
    print(f"\n{char * width}")
    print(text)
    print(char * width)


def main() -> None:
    """Run focused analysis of Louisville vs Tennessee."""
    print_header("LOUISVILLE @ TENNESSEE MATCHUP ANALYSIS")
    print(f"Date: December 16, 2025")
    print(f"Location: Food City Center, Knoxville, TN (Tennessee Home)")
    print(f"Spread: Tennessee -2.5")
    print(f"Network: ESPN")

    # Check API key
    api_key = os.getenv("KENPOM_API_KEY")
    if not api_key:
        print("\n[ERROR] KENPOM_API_KEY not found in environment")
        print("\nTo set up your API key:")
        print("1. Get your API key from https://kenpom.com/register-api.php")
        print("2. Add to .env file: KENPOM_API_KEY=your-key-here")
        return

    print(f"\n[OK] API key loaded")
    print(f"[OK] Initializing analyzers...")

    # Initialize API and analyzers
    api = KenPomAPI(api_key=api_key)
    four_factors = FourFactorsMatchup(api)
    point_dist = PointDistributionAnalyzer(api)
    defensive = DefensiveAnalyzer(api)
    size_analysis = SizeAthleticismAnalyzer(api)
    experience = ExperienceChemistryAnalyzer(api)

    try:
        # =====================================================================
        # SECTION 1: BASIC MATCHUP ANALYSIS
        # =====================================================================
        print_section("1. BASIC EFFICIENCY ANALYSIS")

        basic = analyze_matchup(TEAM1, TEAM2, SEASON, neutral_site=False, home_team=TEAM2)

        print(f"\nTeam Rankings:")
        print(f"  {TEAM1:20s}: #{basic.team1_rank} (AdjEM: {basic.team1_adj_em:+.2f})")
        print(f"  {TEAM2:20s}: #{basic.team2_rank} (AdjEM: {basic.team2_adj_em:+.2f})")

        print(f"\nEfficiency Margin Difference: {basic.em_difference:+.2f}")
        print(f"  -> Advantage: {TEAM1 if basic.em_difference > 0 else TEAM2}")

        print(f"\nBasic Prediction:")
        print(f"  Winner: {basic.predicted_winner}")
        print(f"  Margin: {basic.predicted_margin:.1f} points")
        print(f"  Total: {basic.predicted_total:.1f} points")

        print(f"\nTempo Analysis:")
        print(f"  {TEAM1} AdjT: {basic.team1_tempo:.1f}")
        print(f"  {TEAM2} AdjT: {basic.team2_tempo:.1f}")
        print(f"  Pace Advantage: {basic.pace_advantage}")

        # =====================================================================
        # SECTION 2: FOUR FACTORS ANALYSIS
        # =====================================================================
        print_section("2. FOUR FACTORS ANALYSIS (Dean Oliver Framework)")

        ff = four_factors.analyze_matchup(TEAM1, TEAM2, SEASON)

        print(f"\nOverall Advantage: {ff.overall_advantage.upper()}")
        print(f"Advantage Score: {ff.advantage_score:+.2f} (positive = {TEAM1})")

        print(f"\nFactor-by-Factor Winners:")
        factors = [
            ("eFG%", ff.efg_matchup),
            ("TO%", ff.to_matchup),
            ("OR%", ff.or_matchup),
            ("FT Rate", ff.ft_matchup),
        ]

        for factor_name, matchup in factors:
            winner = TEAM1 if matchup.predicted_winner == "team1" else TEAM2
            print(f"  {factor_name:10s}: {winner:20s} ({matchup.advantage_classification})")

        print(f"\nKey Strategic Insights:")
        for i, insight in enumerate(ff.strategic_insights[:3], 1):
            print(f"  {i}. {insight}")

        # =====================================================================
        # SECTION 3: SCORING STYLES ANALYSIS
        # =====================================================================
        print_section("3. SCORING STYLES & POINT DISTRIBUTION")

        pd = point_dist.analyze_matchup(TEAM1, TEAM2, SEASON)

        print(f"\nScoring Styles:")
        print(f"  {TEAM1:20s}: {pd.team1_profile.style.upper()} ({pd.team1_profile.primary_strength})")
        print(f"  {TEAM2:20s}: {pd.team2_profile.style.upper()} ({pd.team2_profile.primary_strength})")

        print(f"\nStyle Mismatch Score: {pd.style_mismatch_score:.1f}/10")
        print(f"Key Matchup Factor: {pd.key_matchup_factor}")

        print(f"\nPoint Distribution Advantages (positive = {TEAM1}):")
        print(f"  3-Point: {pd.three_point_advantage:+.2f}%")
        print(f"  2-Point: {pd.two_point_advantage:+.2f}%")
        print(f"  Free Throw: {pd.free_throw_advantage:+.2f}%")

        print(f"\n{TEAM1} Can Exploit:")
        for exploit in pd.team1_exploitable_areas:
            print(f"  - {exploit}")

        print(f"\n{TEAM2} Can Exploit:")
        for exploit in pd.team2_exploitable_areas:
            print(f"  - {exploit}")

        # =====================================================================
        # SECTION 4: DEFENSIVE ANALYSIS
        # =====================================================================
        print_section("4. DEFENSIVE SCHEMES & MATCHUPS")

        defense = defensive.analyze_matchup(TEAM1, TEAM2, SEASON)

        print(f"\nBetter Defense: {defense.better_defense.upper()}")
        print(f"Defensive Advantage Score: {defense.defensive_advantage_score:.1f}/10")

        print(f"\nDefensive Schemes:")
        print(f"  {TEAM1:20s}: {defense.team1_defense.defensive_scheme.upper().replace('_', ' ')}")
        print(f"    Primary Strength: {defense.team1_defense.primary_strength}")
        print(f"    Weakness: {defense.team1_defense.primary_weakness}")

        print(f"  {TEAM2:20s}: {defense.team2_defense.defensive_scheme.upper().replace('_', ' ')}")
        print(f"    Primary Strength: {defense.team2_defense.primary_strength}")
        print(f"    Weakness: {defense.team2_defense.primary_weakness}")

        print(f"\nDimensional Advantages:")
        print(f"  Perimeter Defense: {defense.perimeter_defense_advantage}")
        print(f"  Interior Defense: {defense.interior_defense_advantage}")
        print(f"  Pressure Defense: {defense.pressure_defense_advantage}")

        # =====================================================================
        # SECTION 5: SIZE & ATHLETICISM
        # =====================================================================
        print_section("5. SIZE & ATHLETICISM MATCHUPS")

        size = size_analysis.analyze_matchup(TEAM1, TEAM2, SEASON)

        print(f"\nBetter Size: {size.better_size_team.upper()}")
        print(f"Size Advantage Score: {size.size_advantage_score:.1f}/10")
        print(f'Overall Height Advantage: {size.overall_height_advantage:+.2f}" (positive = {TEAM1})')

        print(f"\nSize Profiles:")
        print(f"  {TEAM1:20s}: {size.team1_profile.size_profile.upper().replace('_', ' ')} (Eff Hgt: {size.team1_profile.eff_height:.1f}\")")
        print(f"  {TEAM2:20s}: {size.team2_profile.size_profile.upper().replace('_', ' ')} (Eff Hgt: {size.team2_profile.eff_height:.1f}\")")

        print(f"\nPosition-by-Position:")
        for pos_matchup in [size.pg_matchup, size.sg_matchup, size.sf_matchup, size.pf_matchup, size.c_matchup]:
            winner = TEAM1 if pos_matchup.height_advantage > 0.5 else (TEAM2 if pos_matchup.height_advantage < -0.5 else "EVEN")
            print(f"  {pos_matchup.position}: {winner:20s} ({pos_matchup.advantage_classification})")

        print(f"\nRebounding: {size.rebounding_prediction}")
        print(f"Paint Scoring: {size.paint_scoring_prediction}")

        # =====================================================================
        # SECTION 6: EXPERIENCE & CHEMISTRY
        # =====================================================================
        print_section("6. EXPERIENCE & CHEMISTRY")

        exp = experience.analyze_matchup(TEAM1, TEAM2, SEASON)

        print(f"\nBetter Intangibles: {exp.better_intangibles.upper()}")
        print(f"Intangibles Score: {exp.intangibles_advantage_score:.1f}/10")

        print(f"\nExperience Levels:")
        print(f"  {TEAM1:20s}: {exp.team1_profile.experience_level.upper().replace('_', ' ')} (Rating: {exp.team1_profile.experience_rating:.2f})")
        print(f"  {TEAM2:20s}: {exp.team2_profile.experience_level.upper().replace('_', ' ')} (Rating: {exp.team2_profile.experience_rating:.2f})")

        print(f"\nBench Depth:")
        print(f"  {TEAM1:20s}: {exp.team1_profile.bench_classification.upper().replace('_', ' ')}")
        print(f"  {TEAM2:20s}: {exp.team2_profile.bench_classification.upper().replace('_', ' ')}")

        print(f"\nSituational Predictions:")
        print(f"  Late-Game Execution: {exp.late_game_execution}")
        print(f"  Tournament Readiness: {exp.tournament_readiness}")

        # =====================================================================
        # SECTION 7: OVERALL ASSESSMENT
        # =====================================================================
        print_section("7. OVERALL ASSESSMENT & BETTING ANALYSIS")

        # Count advantages
        team1_wins = 0
        team2_wins = 0

        # Basic efficiency
        if basic.em_difference > 0:
            team1_wins += 1
        else:
            team2_wins += 1

        # Four Factors
        if ff.overall_advantage == TEAM1.lower():
            team1_wins += 1
        else:
            team2_wins += 1

        # Defense
        if defense.better_defense == TEAM1:
            team1_wins += 1
        else:
            team2_wins += 1

        # Size
        if size.better_size_team == TEAM1:
            team1_wins += 1
        else:
            team2_wins += 1

        # Experience
        if exp.better_intangibles == TEAM1:
            team1_wins += 1
        else:
            team2_wins += 1

        print(f"\nDimensional Battle Count:")
        print(f"  {TEAM1}: {team1_wins}/5 key dimensions")
        print(f"  {TEAM2}: {team2_wins}/5 key dimensions")

        print(f"\nVegas Line Comparison:")
        vegas_spread = 2.5
        system_spread = basic.predicted_margin if basic.predicted_winner == TEAM2 else -basic.predicted_margin
        diff = abs(system_spread - vegas_spread)

        print(f"  Vegas: Tennessee -{vegas_spread}")
        print(f"  System: {basic.predicted_winner} by {basic.predicted_margin:.1f}")
        print(f"  Difference: {diff:.1f} points")

        if diff < 1.0:
            print(f"  -> Assessment: System agrees with market")
        elif diff < 3.0:
            print(f"  -> Assessment: Slight disagreement")
        else:
            print(f"  -> Assessment: Significant value opportunity")

        print(f"\nFinal Recommendation:")
        if team2_wins > team1_wins:
            print(f"  -> {TEAM2} has the edge across multiple dimensions")
            print(f"  -> Key advantages: {team2_wins}/{team1_wins + team2_wins} battles won")
            print(f"  -> Home court at Food City Center adds ~3 points")
        elif team1_wins > team2_wins:
            print(f"  -> {TEAM1} has the edge despite being on the road")
            print(f"  -> Key advantages: {team1_wins}/{team1_wins + team2_wins} battles won")
        else:
            print(f"  -> Evenly matched contest - coin flip game")
            print(f"  -> Intangibles and execution will decide winner")

        # Save report
        print_section("REPORT SAVED")
        output_dir = Path("reports")
        output_dir.mkdir(exist_ok=True)

        report_path = output_dir / f"louisville_tennessee_{SEASON}_detailed.txt"
        # In a real implementation, would write full report to file
        print(f"\n[OK] Analysis complete! Would save to: {report_path}")

        print_header("END OF ANALYSIS")

    except ValueError as e:
        print(f"\n[ERROR] Analysis failed: {e}")
    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
