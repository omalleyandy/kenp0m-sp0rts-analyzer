"""Demo of Experience & Chemistry matchup analysis."""

import os
from pathlib import Path

from dotenv import load_dotenv

from kenp0m_sp0rts_analyzer.api_client import KenPomAPI
from kenp0m_sp0rts_analyzer.experience_chemistry_analysis import (
    ExperienceChemistryAnalyzer,
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
analyzer = ExperienceChemistryAnalyzer(api)

# Example matchups to analyze
matchups = [
    ("Duke", "North Carolina"),
    ("Kansas", "Kentucky"),
    ("Connecticut", "Villanova"),
]

print("=" * 80)
print("EXPERIENCE & CHEMISTRY MATCHUP ANALYSIS - 2025 Season")
print("Intangibles: Experience, Bench Depth, Continuity, Tournament Readiness")
print("=" * 80)
print()

for team1, team2 in matchups:
    print(f"\n{'=' * 80}")
    print(f"{team1} vs {team2}")
    print("=" * 80)

    try:
        analysis = analyzer.analyze_matchup(team1, team2, 2025)

        # Team 1 experience profile
        print(f"\n[{team1.upper()} EXPERIENCE PROFILE]")
        print(
            f"Experience Level: {analysis.team1_profile.experience_level.upper().replace('_', ' ')}"
        )
        print(
            f"Experience Rating: {analysis.team1_profile.experience_rating:.2f} "
            f"(Rank #{analysis.team1_profile.experience_rank})"
        )
        print(
            f"Bench Strength: {analysis.team1_profile.bench_strength:+.2f} "
            f"(Rank #{analysis.team1_profile.bench_rank}) - "
            f"{analysis.team1_profile.bench_classification.upper().replace('_', ' ')}"
        )
        print(
            f"Continuity: {analysis.team1_profile.continuity:.1f}% minutes returning "
            f"(Rank #{analysis.team1_profile.continuity_rank}) - "
            f"{analysis.team1_profile.continuity_level.upper().replace('_', ' ')}"
        )
        print(f"Intangibles Score: {analysis.team1_profile.intangibles_score:.1f}/10")
        print(f"Primary Strength: {analysis.team1_profile.primary_strength}")
        print(f"Primary Weakness: {analysis.team1_profile.primary_weakness}")

        # Team 1 tournament readiness
        t1_readiness = analyzer.assess_tournament_readiness(team1, 2025)
        print("\nTournament Readiness:")
        print(
            f"  Overall: {t1_readiness.tournament_readiness_score:.1f}/10 "
            f"({t1_readiness.readiness_tier.upper()})"
        )
        print(f"  Experience Score: {t1_readiness.experience_score:.1f}/10")
        print(f"  Late-Game Poise: {t1_readiness.late_game_poise:.1f}/10")
        print(f"  Depth for Neutral Site: {t1_readiness.depth_for_neutral_site:.1f}/10")
        print(f"  Biggest Concern: {t1_readiness.biggest_concern}")

        # Team 2 experience profile
        print(f"\n[{team2.upper()} EXPERIENCE PROFILE]")
        print(
            f"Experience Level: {analysis.team2_profile.experience_level.upper().replace('_', ' ')}"
        )
        print(
            f"Experience Rating: {analysis.team2_profile.experience_rating:.2f} "
            f"(Rank #{analysis.team2_profile.experience_rank})"
        )
        print(
            f"Bench Strength: {analysis.team2_profile.bench_strength:+.2f} "
            f"(Rank #{analysis.team2_profile.bench_rank}) - "
            f"{analysis.team2_profile.bench_classification.upper().replace('_', ' ')}"
        )
        print(
            f"Continuity: {analysis.team2_profile.continuity:.1f}% minutes returning "
            f"(Rank #{analysis.team2_profile.continuity_rank}) - "
            f"{analysis.team2_profile.continuity_level.upper().replace('_', ' ')}"
        )
        print(f"Intangibles Score: {analysis.team2_profile.intangibles_score:.1f}/10")
        print(f"Primary Strength: {analysis.team2_profile.primary_strength}")
        print(f"Primary Weakness: {analysis.team2_profile.primary_weakness}")

        # Team 2 tournament readiness
        t2_readiness = analyzer.assess_tournament_readiness(team2, 2025)
        print("\nTournament Readiness:")
        print(
            f"  Overall: {t2_readiness.tournament_readiness_score:.1f}/10 "
            f"({t2_readiness.readiness_tier.upper()})"
        )
        print(f"  Experience Score: {t2_readiness.experience_score:.1f}/10")
        print(f"  Late-Game Poise: {t2_readiness.late_game_poise:.1f}/10")
        print(f"  Depth for Neutral Site: {t2_readiness.depth_for_neutral_site:.1f}/10")
        print(f"  Biggest Concern: {t2_readiness.biggest_concern}")

        # Matchup analysis
        print("\n[EXPERIENCE & CHEMISTRY MATCHUP]")
        print(f"Better Intangibles: {analysis.better_intangibles.upper()}")
        print(
            f"Intangibles Advantage Score: {analysis.intangibles_advantage_score:.1f}/10 "
            f"(5 = neutral)"
        )

        print("\nDimensional Advantages:")
        print(f"  Experience: {analysis.experience_advantage}")
        print(f"    Gap: {analysis.experience_gap:+.2f}")
        print(f"    Impact: {analysis.experience_impact}")

        print(f"  Bench Depth: {analysis.bench_advantage}")
        print(f"    Gap: {analysis.bench_gap:+.2f}")
        print(f"    Impact: {analysis.bench_impact}")

        print(f"  Continuity: {analysis.continuity_advantage}")
        print(f"    Gap: {analysis.continuity_gap:+.1f}%")
        print(f"    Impact: {analysis.continuity_impact}")

        # Situational predictions
        print("\n[SITUATIONAL PREDICTIONS]")
        print(f"Late-Game Execution: {analysis.late_game_execution}")
        print(f"Tournament Readiness: {analysis.tournament_readiness}")
        print(f"Adversity Handling: {analysis.adverse_conditions}")

        # Strategic keys
        print(f"\n[{team1.upper()} STRATEGIC KEYS]")
        for i, key in enumerate(analysis.team1_keys, 1):
            print(f"{i}. {key}")

        print(f"\n[{team2.upper()} STRATEGIC KEYS]")
        for i, key in enumerate(analysis.team2_keys, 1):
            print(f"{i}. {key}")

        # Recommendation
        print("\n[MATCHUP RECOMMENDATION]")
        print(analysis.matchup_recommendation)

    except ValueError as e:
        print(f"[ERROR] {e}")
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")
        import traceback

        traceback.print_exc()

print(f"\n{'=' * 80}")
print("[OK] Experience & Chemistry matchup analysis complete!")
print("=" * 80)
