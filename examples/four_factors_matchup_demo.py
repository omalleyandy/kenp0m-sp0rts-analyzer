"""Demo of Four Factors matchup analysis."""

import os
from pathlib import Path

from dotenv import load_dotenv

from kenp0m_sp0rts_analyzer.api_client import KenPomAPI
from kenp0m_sp0rts_analyzer.four_factors_matchup import FourFactorsMatchup

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

print(f"[OK] API key loaded\n")

# Initialize analyzer
api = KenPomAPI()
analyzer = FourFactorsMatchup(api)

# Example matchups to analyze
matchups = [
    ("Duke", "North Carolina"),
    ("Kansas", "Kentucky"),
    ("Connecticut", "Villanova"),
]

print("=" * 80)
print("FOUR FACTORS MATCHUP ANALYSIS - 2025 Season")
print("=" * 80)
print()

for team1, team2 in matchups:
    print(f"\n{'=' * 80}")
    print(f"{team1} vs {team2}")
    print("=" * 80)

    try:
        analysis = analyzer.analyze_matchup(team1, team2, 2025)

        # Overall assessment
        print(f"\n[OVERALL ASSESSMENT]")
        print(f"Advantage: {analysis.overall_advantage.upper()}")
        print(f"Advantage Score: {analysis.advantage_score:+.2f} (positive = {team1})")
        print(f"Key Factor: {analysis.most_important_factor}")

        # Efficiency context
        print(f"\n[EFFICIENCY CONTEXT]")
        print(f"{team1:20s}: AdjOE={analysis.team1_adj_oe:.1f}, AdjDE={analysis.team1_adj_de:.1f}")
        print(f"{team2:20s}: AdjOE={analysis.team2_adj_oe:.1f}, AdjDE={analysis.team2_adj_de:.1f}")

        # Factor-by-factor breakdown
        print(f"\n[FOUR FACTORS BREAKDOWN]")
        print()

        for factor_name, matchup in [
            ("eFG%", analysis.efg_matchup),
            ("TO%", analysis.to_matchup),
            ("OR%", analysis.or_matchup),
            ("FT Rate", analysis.ft_matchup)
        ]:
            print(f"{factor_name} ({matchup.importance_weight:.0%} importance):")
            print(f"  {team1} Offense: {matchup.team1_offense:.1f}%")
            print(f"  {team2} Defense: {matchup.team2_defense:.1f}%")
            print(f"  {team2} Offense: {matchup.team2_offense:.1f}%")
            print(f"  {team1} Defense: {matchup.team1_defense:.1f}%")
            print(f"  -> {team1} Advantage: {matchup.team1_advantage:+.2f}")
            print(f"  -> {team2} Advantage: {matchup.team2_advantage:+.2f}")
            print(f"  -> Winner: {matchup.predicted_winner.upper()} ({matchup.advantage_classification})")
            print()

        # Strategic insights
        print(f"[STRATEGIC INSIGHTS]")
        for i, insight in enumerate(analysis.strategic_insights, 1):
            print(f"{i}. {insight}")

        # Key battles
        print(f"\n[KEY MATCHUP BATTLES]")
        for i, battle in enumerate(analysis.key_matchup_battles, 1):
            print(f"{i}. {battle}")

    except ValueError as e:
        print(f"[ERROR] {e}")
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")
        import traceback
        traceback.print_exc()

print(f"\n{'=' * 80}")
print("[OK] Four Factors matchup analysis complete!")
print("=" * 80)
