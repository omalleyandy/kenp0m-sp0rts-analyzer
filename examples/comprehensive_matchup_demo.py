"""Comprehensive matchup analysis demo combining all TIER 1 + TIER 2 analyzers.

This demo showcases the complete 15-dimensional analytical framework:

TIER 1 (10 dimensions):
1. Four Factors Analysis - Fundamental basketball statistics (4 dimensions)
2. Point Distribution Analysis - Scoring style and defensive vulnerabilities (3 dimensions)
3. Defensive Analysis - Defensive scheme and strategic opportunities (3 dimensions)

TIER 2 (5 dimensions):
4. Size & Athleticism Analysis - Physical matchups and height advantages (2 dimensions)
5. Experience & Chemistry Analysis - Intangibles and tournament readiness (3 dimensions)

Together, these provide a complete 15-dimensional picture of team matchups.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

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

# Initialize all analyzers (TIER 1 + TIER 2)
api = KenPomAPI()
four_factors = FourFactorsMatchup(api)
point_dist = PointDistributionAnalyzer(api)
defensive = DefensiveAnalyzer(api)
size_athleticism = SizeAthleticismAnalyzer(api)
experience = ExperienceChemistryAnalyzer(api)

# Example matchups to analyze
matchups = [
    ("Duke", "North Carolina"),
    ("Kansas", "Kentucky"),
]

print("=" * 80)
print("COMPREHENSIVE MATCHUP ANALYSIS - 2025 Season")
print("Complete 15-Dimensional Framework: TIER 1 + TIER 2")
print("=" * 80)
print()

for team1, team2 in matchups:
    print(f"\n{'=' * 100}")
    print(f"{team1.upper()} vs {team2.upper()} - COMPLETE MATCHUP BREAKDOWN")
    print("=" * 100)

    try:
        # Run all analyses (TIER 1 + TIER 2)
        ff_analysis = four_factors.analyze_matchup(team1, team2, 2025)
        pd_analysis = point_dist.analyze_matchup(team1, team2, 2025)
        def_analysis = defensive.analyze_matchup(team1, team2, 2025)
        size_analysis = size_athleticism.analyze_matchup(team1, team2, 2025)
        exp_analysis = experience.analyze_matchup(team1, team2, 2025)

        # =====================================================================
        # SECTION 1: FOUR FACTORS ANALYSIS
        # =====================================================================
        print("\n[1] FOUR FACTORS ANALYSIS")
        print("-" * 100)
        print(f"Overall Advantage: {ff_analysis.overall_advantage.upper()}")
        print(
            f"Advantage Score: {ff_analysis.advantage_score:+.2f} (positive = {team1})"
        )
        print(f"Most Important Factor: {ff_analysis.most_important_factor}")

        print("\nEfficiency Ratings:")
        print(
            f"  {team1:20s}: AdjOE={ff_analysis.team1_adj_oe:.1f}, "
            f"AdjDE={ff_analysis.team1_adj_de:.1f}"
        )
        print(
            f"  {team2:20s}: AdjOE={ff_analysis.team2_adj_oe:.1f}, "
            f"AdjDE={ff_analysis.team2_adj_de:.1f}"
        )

        print("\nFactor Winners:")
        for factor_name, matchup in [
            ("eFG%", ff_analysis.efg_matchup),
            ("TO%", ff_analysis.to_matchup),
            ("OR%", ff_analysis.or_matchup),
            ("FT Rate", ff_analysis.ft_matchup),
        ]:
            winner_name = (
                team1
                if matchup.predicted_winner == "team1"
                else (team2 if matchup.predicted_winner == "team2" else "NEUTRAL")
            )
            print(
                f"  {factor_name:10s}: {winner_name:20s} "
                f"({matchup.advantage_classification})"
            )

        # =====================================================================
        # SECTION 2: SCORING STYLES ANALYSIS
        # =====================================================================
        print("\n[2] SCORING STYLES ANALYSIS")
        print("-" * 100)
        print(f"Matchup Factor: {pd_analysis.key_matchup_factor}")
        print(f"Style Mismatch Score: {pd_analysis.style_mismatch_score:.1f}/10")

        print("\nScoring Styles:")
        print(
            f"  {team1:20s}: {pd_analysis.team1_profile.style.upper()} "
            f"({pd_analysis.team1_profile.primary_strength})"
        )
        print(
            f"  {team2:20s}: {pd_analysis.team2_profile.style.upper()} "
            f"({pd_analysis.team2_profile.primary_strength})"
        )

        print("\nPoint Distribution Breakdown:")
        print(f"  {team1} Offense:")
        print(
            f"    3PT: {pd_analysis.team1_profile.fg3_pct:.1f}% | "
            f"2PT: {pd_analysis.team1_profile.fg2_pct:.1f}% | "
            f"FT: {pd_analysis.team1_profile.ft_pct:.1f}%"
        )
        print(f"  {team2} Defense Allowed:")
        print(
            f"    3PT: {pd_analysis.team2_profile.def_fg3_pct:.1f}% | "
            f"2PT: {pd_analysis.team2_profile.def_fg2_pct:.1f}% | "
            f"FT: {pd_analysis.team2_profile.def_ft_pct:.1f}%"
        )
        print(f"  Advantages (positive = {team1} edge):")
        print(
            f"    3PT: {pd_analysis.three_point_advantage:+.2f}% | "
            f"2PT: {pd_analysis.two_point_advantage:+.2f}% | "
            f"FT: {pd_analysis.free_throw_advantage:+.2f}%"
        )
        print()
        print(f"  {team2} Offense:")
        print(
            f"    3PT: {pd_analysis.team2_profile.fg3_pct:.1f}% | "
            f"2PT: {pd_analysis.team2_profile.fg2_pct:.1f}% | "
            f"FT: {pd_analysis.team2_profile.ft_pct:.1f}%"
        )
        print(f"  {team1} Defense Allowed:")
        print(
            f"    3PT: {pd_analysis.team1_profile.def_fg3_pct:.1f}% | "
            f"2PT: {pd_analysis.team1_profile.def_fg2_pct:.1f}% | "
            f"FT: {pd_analysis.team1_profile.def_ft_pct:.1f}%"
        )

        print("\nExploitable Areas:")
        print(f"  {team1} can exploit:")
        for exploit in pd_analysis.team1_exploitable_areas:
            print(f"    - {exploit}")
        print(f"  {team2} can exploit:")
        for exploit in pd_analysis.team2_exploitable_areas:
            print(f"    - {exploit}")

        # =====================================================================
        # SECTION 3: DEFENSIVE ANALYSIS
        # =====================================================================
        print("\n[3] DEFENSIVE ANALYSIS")
        print("-" * 100)
        print(f"Better Defense: {def_analysis.better_defense.upper()}")
        print(
            f"Defensive Advantage: {def_analysis.defensive_advantage_score:.1f}/10 "
            f"(5 = neutral)"
        )

        print("\nDefensive Schemes:")
        print(
            f"  {team1:20s}: "
            f"{def_analysis.team1_defense.defensive_scheme.upper().replace('_', ' ')} "
            f"({def_analysis.team1_defense.primary_strength})"
        )
        print(
            f"  {team2:20s}: "
            f"{def_analysis.team2_defense.defensive_scheme.upper().replace('_', ' ')} "
            f"({def_analysis.team2_defense.primary_strength})"
        )

        print("\nDimensional Advantages:")
        print(
            f"  Perimeter Defense: {def_analysis.perimeter_defense_advantage} "
            f"(Opp 3PT%: {team1}={def_analysis.team1_defense.opp_fg3_pct:.1f}%, "
            f"{team2}={def_analysis.team2_defense.opp_fg3_pct:.1f}%)"
        )
        print(
            f"  Interior Defense: {def_analysis.interior_defense_advantage} "
            f"(Opp 2PT%: {team1}={def_analysis.team1_defense.opp_fg2_pct:.1f}%, "
            f"{team2}={def_analysis.team2_defense.opp_fg2_pct:.1f}%)"
        )
        print(
            f"  Pressure Defense: {def_analysis.pressure_defense_advantage} "
            f"(Steal Rate: {team1}={def_analysis.team1_defense.stl_rate:.1f}%, "
            f"{team2}={def_analysis.team2_defense.stl_rate:.1f}%)"
        )

        # =====================================================================
        # SECTION 4: STRATEGIC SUMMARY
        # =====================================================================
        print("\n[4] STRATEGIC SUMMARY")
        print("-" * 100)

        print("\nFour Factors Strategic Insights:")
        for i, insight in enumerate(ff_analysis.strategic_insights, 1):
            print(f"  {i}. {insight}")

        print("\nScoring Style Strategy:")
        print(f"  {pd_analysis.recommended_strategy}")

        print("\nDefensive Matchup:")
        print(f"  {def_analysis.matchup_recommendation}")

        print("\nKey Matchup Battles:")
        for i, battle in enumerate(ff_analysis.key_matchup_battles, 1):
            print(f"  {i}. {battle}")

        print(f"\n{team1} Defensive Keys:")
        for i, key in enumerate(def_analysis.team1_defensive_keys, 1):
            print(f"  {i}. {key}")

        print(f"\n{team2} Defensive Keys:")
        for i, key in enumerate(def_analysis.team2_defensive_keys, 1):
            print(f"  {i}. {key}")

        # =====================================================================
        # SECTION 5: SIZE & ATHLETICISM ANALYSIS
        # =====================================================================
        print("\n[5] SIZE & ATHLETICISM ANALYSIS")
        print("-" * 100)
        print(f"Better Size: {size_analysis.better_size_team.upper()}")
        print(
            f"Size Advantage Score: {size_analysis.size_advantage_score:.1f}/10 "
            f"(5 = neutral)"
        )
        print(
            f'Overall Height Advantage: {size_analysis.overall_height_advantage:+.2f}" '
            f"(positive = {team1})"
        )

        print("\nTeam Size Profiles:")
        print(
            f"  {team1:20s}: {size_analysis.team1_profile.size_profile.upper().replace('_', ' ')} "
            f'(Eff Hgt: {size_analysis.team1_profile.eff_height:.1f}")'
        )
        print(
            f"  {team2:20s}: {size_analysis.team2_profile.size_profile.upper().replace('_', ' ')} "
            f'(Eff Hgt: {size_analysis.team2_profile.eff_height:.1f}")'
        )

        print("\nCourt Advantages:")
        print(f"  Frontcourt (SF/PF/C): {size_analysis.frontcourt_advantage}")
        print(f"  Backcourt (PG/SG): {size_analysis.backcourt_advantage}")

        print("\nPosition-by-Position Matchups:")
        for pos_matchup in [
            size_analysis.pg_matchup,
            size_analysis.sg_matchup,
            size_analysis.sf_matchup,
            size_analysis.pf_matchup,
            size_analysis.c_matchup,
        ]:
            winner = (
                team1
                if pos_matchup.height_advantage > 0.5
                else (team2 if pos_matchup.height_advantage < -0.5 else "EVEN")
            )
            print(
                f"  {pos_matchup.position}: {winner:20s} "
                f"({pos_matchup.advantage_classification})"
            )

        print(f"\nRebounding Prediction: {size_analysis.rebounding_prediction}")
        print(f"Paint Scoring: {size_analysis.paint_scoring_prediction}")
        print(f"\nStrategy: {size_analysis.strategic_recommendation}")

        # =====================================================================
        # SECTION 6: EXPERIENCE & CHEMISTRY ANALYSIS
        # =====================================================================
        print("\n[6] EXPERIENCE & CHEMISTRY ANALYSIS")
        print("-" * 100)
        print(f"Better Intangibles: {exp_analysis.better_intangibles.upper()}")
        print(
            f"Intangibles Advantage Score: {exp_analysis.intangibles_advantage_score:.1f}/10 "
            f"(5 = neutral)"
        )

        print("\nExperience Profiles:")
        print(
            f"  {team1:20s}: {exp_analysis.team1_profile.experience_level.upper().replace('_', ' ')} "
            f"(Rating: {exp_analysis.team1_profile.experience_rating:.2f})"
        )
        print(
            f"  {team2:20s}: {exp_analysis.team2_profile.experience_level.upper().replace('_', ' ')} "
            f"(Rating: {exp_analysis.team2_profile.experience_rating:.2f})"
        )

        print("\nBench Depth:")
        print(
            f"  {team1:20s}: {exp_analysis.team1_profile.bench_classification.upper().replace('_', ' ')} "
            f"({exp_analysis.team1_profile.bench_strength:+.2f})"
        )
        print(
            f"  {team2:20s}: {exp_analysis.team2_profile.bench_classification.upper().replace('_', ' ')} "
            f"({exp_analysis.team2_profile.bench_strength:+.2f})"
        )

        print("\nContinuity:")
        print(
            f"  {team1:20s}: {exp_analysis.team1_profile.continuity:.1f}% minutes returning "
            f"({exp_analysis.team1_profile.continuity_level.upper().replace('_', ' ')})"
        )
        print(
            f"  {team2:20s}: {exp_analysis.team2_profile.continuity:.1f}% minutes returning "
            f"({exp_analysis.team2_profile.continuity_level.upper().replace('_', ' ')})"
        )

        print("\nSituational Predictions:")
        print(f"  Late-Game Execution: {exp_analysis.late_game_execution}")
        print(f"  Tournament Readiness: {exp_analysis.tournament_readiness}")
        print(f"  Adversity Handling: {exp_analysis.adverse_conditions}")

        # =====================================================================
        # SECTION 7: OVERALL ASSESSMENT (15 DIMENSIONS)
        # =====================================================================
        print("\n[7] OVERALL ASSESSMENT (15-DIMENSIONAL FRAMEWORK)")
        print("-" * 100)

        # Count advantages across all dimensions
        team1_advantages = 0
        team2_advantages = 0

        # Four Factors (4 battles)
        factors = [
            ff_analysis.efg_matchup,
            ff_analysis.to_matchup,
            ff_analysis.or_matchup,
            ff_analysis.ft_matchup,
        ]
        for factor in factors:
            if factor.predicted_winner == "team1":
                team1_advantages += 1
            elif factor.predicted_winner == "team2":
                team2_advantages += 1

        # Scoring advantages (3 battles: 3pt, 2pt, FT)
        if pd_analysis.three_point_advantage > 1.0:
            team1_advantages += 1
        elif pd_analysis.three_point_advantage < -1.0:
            team2_advantages += 1

        if pd_analysis.two_point_advantage > 1.0:
            team1_advantages += 1
        elif pd_analysis.two_point_advantage < -1.0:
            team2_advantages += 1

        if pd_analysis.free_throw_advantage > 1.0:
            team1_advantages += 1
        elif pd_analysis.free_throw_advantage < -1.0:
            team2_advantages += 1

        # Defensive battles (3 dimensions: perimeter, interior, pressure)
        if def_analysis.perimeter_defense_advantage == team1:
            team1_advantages += 1
        else:
            team2_advantages += 1

        if def_analysis.interior_defense_advantage == team1:
            team1_advantages += 1
        else:
            team2_advantages += 1

        if def_analysis.pressure_defense_advantage == team1:
            team1_advantages += 1
        else:
            team2_advantages += 1

        # TIER 2: Size battles (2 dimensions: overall size + frontcourt/backcourt composite)
        if size_analysis.better_size_team == team1:
            team1_advantages += 1
        elif size_analysis.better_size_team == team2:
            team2_advantages += 1

        # Composite frontcourt + backcourt advantage (1 dimension)
        fc_bc_advantages = 0
        if size_analysis.frontcourt_advantage == team1:
            fc_bc_advantages += 1
        elif size_analysis.frontcourt_advantage == team2:
            fc_bc_advantages -= 1

        if size_analysis.backcourt_advantage == team1:
            fc_bc_advantages += 1
        elif size_analysis.backcourt_advantage == team2:
            fc_bc_advantages -= 1

        if fc_bc_advantages > 0:
            team1_advantages += 1
        elif fc_bc_advantages < 0:
            team2_advantages += 1

        # TIER 2: Experience battles (3 dimensions: experience, bench, continuity)
        if exp_analysis.experience_advantage == team1:
            team1_advantages += 1
        elif exp_analysis.experience_advantage == team2:
            team2_advantages += 1

        if exp_analysis.bench_advantage == team1:
            team1_advantages += 1
        elif exp_analysis.bench_advantage == team2:
            team2_advantages += 1

        if exp_analysis.continuity_advantage == team1:
            team1_advantages += 1
        elif exp_analysis.continuity_advantage == team2:
            team2_advantages += 1

        total_battles = 15  # TIER 1 (10) + TIER 2 (5)
        print(f"\nBattles Won (out of {total_battles} key matchups):")
        print(f"  {team1}: {team1_advantages}/{total_battles}")
        print(f"  {team2}: {team2_advantages}/{total_battles}")

        if team1_advantages > team2_advantages + 2:
            print(
                f"\nPREDICTION: {team1} has significant statistical advantages "
                f"across multiple dimensions"
            )
        elif team2_advantages > team1_advantages + 2:
            print(
                f"\nPREDICTION: {team2} has significant statistical advantages "
                f"across multiple dimensions"
            )
        else:
            print(
                "\nPREDICTION: Evenly matched contest. "
                "Execution and intangibles will decide the outcome."
            )

    except ValueError as e:
        print(f"[ERROR] {e}")
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")
        import traceback

        traceback.print_exc()

print(f"\n{'=' * 100}")
print("[OK] Comprehensive matchup analysis complete!")
print("=" * 100)
