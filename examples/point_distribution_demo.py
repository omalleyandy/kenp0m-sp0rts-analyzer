"""Demo of Point Distribution analysis."""

import os
from pathlib import Path

from dotenv import load_dotenv

from kenp0m_sp0rts_analyzer.api_client import KenPomAPI
from kenp0m_sp0rts_analyzer.point_distribution_analysis import (
    PointDistributionAnalyzer,
)

# Load environment variables
env_path = Path(__file__).parent / ".env"
if env_path.exists():
    load_dotenv(env_path)
    print(f"[OK] Loaded .env from {env_path}")
else:
    print(f"[WARNING] No .env file found at {env_path}")

# Check API key
api_key = os.getenv("KENPOM_API_KEY")
if not api_key:
    print("[ERROR] No KENPOM_API_KEY found in environment")
    exit(1)

print("[OK] API key loaded\n")

# Initialize analyzer
api = KenPomAPI()
analyzer = PointDistributionAnalyzer(api)

# Example matchups to analyze
matchups = [
    ("Duke", "North Carolina"),
    ("Kansas", "Kentucky"),
    ("Connecticut", "Villanova"),
]

print("=" * 80)
print("POINT DISTRIBUTION ANALYSIS - 2025 Season")
print("Scoring Style Matchups and Strategic Insights")
print("=" * 80)
print()

for team1, team2 in matchups:
    print(f"\n{'=' * 80}")
    print(f"{team1} vs {team2}")
    print("=" * 80)

    try:
        analysis = analyzer.analyze_matchup(team1, team2, 2025)

        # Team profiles
        print(f"\n[{team1.upper()} SCORING PROFILE]")
        print(f"Style: {analysis.team1_profile.style.upper()}")
        print(f"Strength: {analysis.team1_profile.primary_strength}")
        print(f"Defensive Weakness: {analysis.team1_profile.defensive_weakness}")
        print("\nPoint Distribution:")
        print(
            f"  3-Point: {analysis.team1_profile.fg3_pct:.1f}% (Rank #{analysis.team1_profile.fg3_rank})"
        )
        print(
            f"  2-Point: {analysis.team1_profile.fg2_pct:.1f}% (Rank #{analysis.team1_profile.fg2_rank})"
        )
        print(
            f"  Free Throw: {analysis.team1_profile.ft_pct:.1f}% (Rank #{analysis.team1_profile.ft_rank})"
        )
        print("\nDefensive Distribution (Points Allowed):")
        print(f"  3-Point: {analysis.team1_profile.def_fg3_pct:.1f}%")
        print(f"  2-Point: {analysis.team1_profile.def_fg2_pct:.1f}%")
        print(f"  Free Throw: {analysis.team1_profile.def_ft_pct:.1f}%")

        print(f"\n[{team2.upper()} SCORING PROFILE]")
        print(f"Style: {analysis.team2_profile.style.upper()}")
        print(f"Strength: {analysis.team2_profile.primary_strength}")
        print(f"Defensive Weakness: {analysis.team2_profile.defensive_weakness}")
        print("\nPoint Distribution:")
        print(
            f"  3-Point: {analysis.team2_profile.fg3_pct:.1f}% (Rank #{analysis.team2_profile.fg3_rank})"
        )
        print(
            f"  2-Point: {analysis.team2_profile.fg2_pct:.1f}% (Rank #{analysis.team2_profile.fg2_rank})"
        )
        print(
            f"  Free Throw: {analysis.team2_profile.ft_pct:.1f}% (Rank #{analysis.team2_profile.ft_rank})"
        )
        print("\nDefensive Distribution (Points Allowed):")
        print(f"  3-Point: {analysis.team2_profile.def_fg3_pct:.1f}%")
        print(f"  2-Point: {analysis.team2_profile.def_fg2_pct:.1f}%")
        print(f"  Free Throw: {analysis.team2_profile.def_ft_pct:.1f}%")

        # Matchup analysis
        print("\n[MATCHUP ANALYSIS]")
        print(f"Key Factor: {analysis.key_matchup_factor}")
        print(f"Style Mismatch Score: {analysis.style_mismatch_score:.1f}/10")
        print(f"\nAdvantages (positive = {team1}, negative = {team2}):")
        print(f"  3-Point: {analysis.three_point_advantage:+.2f}%")
        print(f"  2-Point: {analysis.two_point_advantage:+.2f}%")
        print(f"  Free Throw: {analysis.free_throw_advantage:+.2f}%")

        # Strategic insights
        print(f"\n[{team1.upper()} EXPLOITABLE AREAS]")
        for i, exploit in enumerate(analysis.team1_exploitable_areas, 1):
            print(f"{i}. {exploit}")

        print(f"\n[{team2.upper()} EXPLOITABLE AREAS]")
        for i, exploit in enumerate(analysis.team2_exploitable_areas, 1):
            print(f"{i}. {exploit}")

        # Recommendation
        print("\n[STRATEGIC RECOMMENDATION]")
        print(f"{analysis.recommended_strategy}")

    except ValueError as e:
        print(f"[ERROR] {e}")
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")
        import traceback

        traceback.print_exc()

print(f"\n{'=' * 80}")
print("[OK] Point Distribution analysis complete!")
print("=" * 80)
