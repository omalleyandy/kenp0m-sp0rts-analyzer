"""Demo of Size & Athleticism matchup analysis."""

import os
from pathlib import Path

from dotenv import load_dotenv

from kenp0m_sp0rts_analyzer.api_client import KenPomAPI
from kenp0m_sp0rts_analyzer.size_athleticism_analysis import SizeAthleticismAnalyzer

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
analyzer = SizeAthleticismAnalyzer(api)

# Example matchups to analyze
matchups = [
    ("Duke", "North Carolina"),
    ("Kansas", "Kentucky"),
    ("Connecticut", "Villanova"),
]

print("=" * 80)
print("SIZE & ATHLETICISM MATCHUP ANALYSIS - 2025 Season")
print("Physical Advantages and Position-Specific Size Matchups")
print("=" * 80)
print()

for team1, team2 in matchups:
    print(f"\n{'=' * 80}")
    print(f"{team1} vs {team2}")
    print("=" * 80)

    try:
        analysis = analyzer.analyze_matchup(team1, team2, 2025)

        # Team 1 size profile
        print(f"\n[{team1.upper()} SIZE PROFILE]")
        print(
            f"Size Profile: {analysis.team1_profile.size_profile.upper().replace('_', ' ')}"
        )
        print(
            f'Average Height: {analysis.team1_profile.avg_height:.1f}" '
            f"(Rank #{analysis.team1_profile.avg_height_rank})"
        )
        print(
            f'Effective Height: {analysis.team1_profile.eff_height:.1f}" '
            f"(Rank #{analysis.team1_profile.eff_height_rank})"
        )
        print(f"Biggest Advantage: {analysis.team1_profile.biggest_advantage}")
        print(f"Biggest Weakness: {analysis.team1_profile.biggest_weakness}")
        print("\nPosition-Specific Heights:")
        print(
            f'  PG: {analysis.team1_profile.pg_height:.1f}" '
            f"(Rank #{analysis.team1_profile.pg_height_rank})"
        )
        print(
            f'  SG: {analysis.team1_profile.sg_height:.1f}" '
            f"(Rank #{analysis.team1_profile.sg_height_rank})"
        )
        print(
            f'  SF: {analysis.team1_profile.sf_height:.1f}" '
            f"(Rank #{analysis.team1_profile.sf_height_rank})"
        )
        print(
            f'  PF: {analysis.team1_profile.pf_height:.1f}" '
            f"(Rank #{analysis.team1_profile.pf_height_rank})"
        )
        print(
            f'  C: {analysis.team1_profile.c_height:.1f}" '
            f"(Rank #{analysis.team1_profile.c_height_rank})"
        )

        # Team 2 size profile
        print(f"\n[{team2.upper()} SIZE PROFILE]")
        print(
            f"Size Profile: {analysis.team2_profile.size_profile.upper().replace('_', ' ')}"
        )
        print(
            f'Average Height: {analysis.team2_profile.avg_height:.1f}" '
            f"(Rank #{analysis.team2_profile.avg_height_rank})"
        )
        print(
            f'Effective Height: {analysis.team2_profile.eff_height:.1f}" '
            f"(Rank #{analysis.team2_profile.eff_height_rank})"
        )
        print(f"Biggest Advantage: {analysis.team2_profile.biggest_advantage}")
        print(f"Biggest Weakness: {analysis.team2_profile.biggest_weakness}")
        print("\nPosition-Specific Heights:")
        print(
            f'  PG: {analysis.team2_profile.pg_height:.1f}" '
            f"(Rank #{analysis.team2_profile.pg_height_rank})"
        )
        print(
            f'  SG: {analysis.team2_profile.sg_height:.1f}" '
            f"(Rank #{analysis.team2_profile.sg_height_rank})"
        )
        print(
            f'  SF: {analysis.team2_profile.sf_height:.1f}" '
            f"(Rank #{analysis.team2_profile.sf_height_rank})"
        )
        print(
            f'  PF: {analysis.team2_profile.pf_height:.1f}" '
            f"(Rank #{analysis.team2_profile.pf_height_rank})"
        )
        print(
            f'  C: {analysis.team2_profile.c_height:.1f}" '
            f"(Rank #{analysis.team2_profile.c_height_rank})"
        )

        # Overall matchup analysis
        print("\n[SIZE MATCHUP ANALYSIS]")
        print(f"Better Size Team: {analysis.better_size_team.upper()}")
        print(
            f"Size Advantage Score: {analysis.size_advantage_score:.1f}/10 (5 = neutral)"
        )
        print(
            f'Overall Height Advantage: {analysis.overall_height_advantage:+.2f}" '
            f"(positive = {team1})"
        )

        # Court-specific advantages
        print("\nCourt Advantages:")
        print(f"  Frontcourt (SF/PF/C): {analysis.frontcourt_advantage}")
        print(f"  Backcourt (PG/SG): {analysis.backcourt_advantage}")

        # Position-by-position breakdown
        print("\n[POSITION-BY-POSITION MATCHUPS]")
        for matchup in [
            analysis.pg_matchup,
            analysis.sg_matchup,
            analysis.sf_matchup,
            analysis.pf_matchup,
            analysis.c_matchup,
        ]:
            print(f"\n{matchup.position_name} ({matchup.position}):")
            print(
                f'  {team1}: {matchup.team1_height:.1f}" vs '
                f'{team2}: {matchup.team2_height:.1f}"'
            )
            print(
                f'  Height Advantage: {matchup.height_advantage:+.2f}" '
                f'({matchup.advantage_inches:.2f}" absolute)'
            )
            print(f"  Classification: {matchup.advantage_classification.upper()}")
            print(f"  Impact: {matchup.predicted_impact}")

        # Predictions
        print("\n[PREDICTIONS]")
        print(f"Rebounding Battle: {analysis.rebounding_prediction}")
        print(f"Paint Scoring: {analysis.paint_scoring_prediction}")

        # Strategic recommendation
        print("\n[STRATEGIC RECOMMENDATION]")
        print(analysis.strategic_recommendation)

    except ValueError as e:
        print(f"[ERROR] {e}")
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")
        import traceback

        traceback.print_exc()

print(f"\n{'=' * 80}")
print("[OK] Size & Athleticism matchup analysis complete!")
print("=" * 80)
