"""Demo of Defensive matchup analysis."""

import os
from pathlib import Path

from dotenv import load_dotenv

from kenp0m_sp0rts_analyzer.api_client import KenPomAPI
from kenp0m_sp0rts_analyzer.defensive_analysis import DefensiveAnalyzer

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
analyzer = DefensiveAnalyzer(api)

# Example matchups to analyze
matchups = [
    ("Duke", "North Carolina"),
    ("Kansas", "Kentucky"),
    ("Connecticut", "Villanova"),
]

print("=" * 80)
print("DEFENSIVE MATCHUP ANALYSIS - 2025 Season")
print("Defensive Scheme Identification and Strategic Keys")
print("=" * 80)
print()

for team1, team2 in matchups:
    print(f"\n{'=' * 80}")
    print(f"{team1} vs {team2}")
    print("=" * 80)

    try:
        analysis = analyzer.analyze_matchup(team1, team2, 2025)

        # Team 1 defensive profile
        print(f"\n[{team1.upper()} DEFENSIVE PROFILE]")
        print(
            f"Scheme: {analysis.team1_defense.defensive_scheme.upper().replace('_', ' ')}"
        )
        print(f"Strength: {analysis.team1_defense.primary_strength}")
        print(f"Weakness: {analysis.team1_defense.primary_weakness}")
        print("\nPerimeter Defense:")
        print(
            f"  Opp 3PT%: {analysis.team1_defense.opp_fg3_pct:.1f}% (Rank #{analysis.team1_defense.opp_fg3_rank})"
        )
        print("\nInterior Defense:")
        print(
            f"  Opp 2PT%: {analysis.team1_defense.opp_fg2_pct:.1f}% (Rank #{analysis.team1_defense.opp_fg2_rank})"
        )
        print(
            f"  Block%: {analysis.team1_defense.block_pct:.1f}% (Rank #{analysis.team1_defense.block_rank})"
        )
        print("\nPressure Defense:")
        print(
            f"  Steal Rate: {analysis.team1_defense.stl_rate:.1f}% (Rank #{analysis.team1_defense.stl_rank})"
        )
        print(
            f"  Opp NST Rate: {analysis.team1_defense.opp_nst_rate:.1f}% (Rank #{analysis.team1_defense.nst_rank})"
        )
        print("\nBall Movement Allowed:")
        print(
            f"  Opp Assist Rate: {analysis.team1_defense.opp_assist_rate:.1f}% (Rank #{analysis.team1_defense.assist_rank})"
        )

        # Team 2 defensive profile
        print(f"\n[{team2.upper()} DEFENSIVE PROFILE]")
        print(
            f"Scheme: {analysis.team2_defense.defensive_scheme.upper().replace('_', ' ')}"
        )
        print(f"Strength: {analysis.team2_defense.primary_strength}")
        print(f"Weakness: {analysis.team2_defense.primary_weakness}")
        print("\nPerimeter Defense:")
        print(
            f"  Opp 3PT%: {analysis.team2_defense.opp_fg3_pct:.1f}% (Rank #{analysis.team2_defense.opp_fg3_rank})"
        )
        print("\nInterior Defense:")
        print(
            f"  Opp 2PT%: {analysis.team2_defense.opp_fg2_pct:.1f}% (Rank #{analysis.team2_defense.opp_fg2_rank})"
        )
        print(
            f"  Block%: {analysis.team2_defense.block_pct:.1f}% (Rank #{analysis.team2_defense.block_rank})"
        )
        print("\nPressure Defense:")
        print(
            f"  Steal Rate: {analysis.team2_defense.stl_rate:.1f}% (Rank #{analysis.team2_defense.stl_rank})"
        )
        print(
            f"  Opp NST Rate: {analysis.team2_defense.opp_nst_rate:.1f}% (Rank #{analysis.team2_defense.nst_rank})"
        )
        print("\nBall Movement Allowed:")
        print(
            f"  Opp Assist Rate: {analysis.team2_defense.opp_assist_rate:.1f}% (Rank #{analysis.team2_defense.assist_rank})"
        )

        # Matchup analysis
        print("\n[DEFENSIVE MATCHUP ANALYSIS]")
        print(f"Better Defense: {analysis.better_defense.upper()}")
        print(
            f"Defensive Advantage Score: {analysis.defensive_advantage_score:.1f}/10 (5 = neutral)"
        )
        print("\nDimensional Advantages:")
        print(f"  Perimeter Defense: {analysis.perimeter_defense_advantage}")
        print(f"  Interior Defense: {analysis.interior_defense_advantage}")
        print(f"  Pressure Defense: {analysis.pressure_defense_advantage}")

        # Strategic keys
        print(f"\n[{team1.upper()} DEFENSIVE KEYS]")
        for i, key in enumerate(analysis.team1_defensive_keys, 1):
            print(f"{i}. {key}")

        print(f"\n[{team2.upper()} DEFENSIVE KEYS]")
        for i, key in enumerate(analysis.team2_defensive_keys, 1):
            print(f"{i}. {key}")

        # Recommendation
        print("\n[MATCHUP RECOMMENDATION]")
        print(f"{analysis.matchup_recommendation}")

    except ValueError as e:
        print(f"[ERROR] {e}")
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")
        import traceback

        traceback.print_exc()

print(f"\n{'=' * 80}")
print("[OK] Defensive matchup analysis complete!")
print("=" * 80)
