"""Demo: NCAA Tournament Monte Carlo Simulation.

This script demonstrates how to use the TournamentSimulator to:
1. Load tournament bracket data from Selection Sunday
2. Run 10,000+ Monte Carlo simulations
3. Calculate round-by-round advancement probabilities
4. Identify high-value upset opportunities
5. Generate bracket recommendations

The simulator uses KenPom efficiency ratings to model game outcomes
with appropriate variance, providing probabilistic predictions for
every team's tournament performance.
"""

import os
from pathlib import Path

import pandas as pd

from kenp0m_sp0rts_analyzer.tournament_simulator import TournamentSimulator


def main():
    """Run basic tournament simulation demo."""
    # Check for API key
    api_key = os.getenv("KENPOM_API_KEY")
    if not api_key:
        print("Error: KENPOM_API_KEY environment variable not set")
        print("Get your API key from: https://kenpom.com/register-api.php")
        return

    print("NCAA Tournament Monte Carlo Simulator")
    print("=" * 70)
    print()

    # Initialize simulator
    print("[1/5] Initializing simulator...")
    simulator = TournamentSimulator(api_key=api_key, random_seed=42)
    print("✓ Simulator initialized with reproducible random seed")

    # Load bracket from Selection Sunday
    print("\n[2/5] Loading bracket from Selection Sunday 2024...")
    selection_sunday = "2024-03-17"  # Example: 2024 Selection Sunday

    # Note: In production, you would provide actual bracket data
    # For demo, we'll use auto-assignment of top 64 teams
    print(f"  Fetching KenPom ratings from {selection_sunday}...")

    try:
        bracket = simulator.load_bracket_from_kenpom(
            selection_sunday_date=selection_sunday,
            regions=None,  # Auto-assign top 64 teams
        )

        # Display bracket structure
        print(f"✓ Loaded {sum(len(teams) for teams in bracket.values())} teams")
        print("\nRegion breakdown:")
        for region, teams in bracket.items():
            print(f"  {region}: {len(teams)} teams (seeds 1-16)")
            # Show top 4 seeds
            top_4 = sorted(teams, key=lambda t: t.seed)[:4]
            for team in top_4:
                print(f"    [{team.seed}] {team.name} (AdjEM: {team.adj_em:.2f})")

    except Exception as e:
        print(f"✗ Error loading bracket: {e}")
        print("\nNote: You may need to use a more recent Selection Sunday date")
        print("      or provide manual bracket data. See docstring for details.")
        return

    # Run simulations
    print("\n[3/5] Running Monte Carlo simulations...")
    num_sims = 10000
    print(f"  Simulating {num_sims:,} tournaments...")

    probabilities = simulator.simulate_tournament(
        bracket=bracket, num_simulations=num_sims, verbose=True
    )

    print(f"✓ Completed {num_sims:,} simulations")

    # Display champion probabilities
    print("\n[4/5] Championship Probabilities (Top 10)")
    print("-" * 70)

    champion_probs = sorted(
        probabilities.winner.items(), key=lambda x: x[1], reverse=True
    )

    for i, (team, prob) in enumerate(champion_probs[:10], 1):
        # Find team seed
        team_obj = None
        for region_teams in bracket.values():
            for t in region_teams:
                if t.name == team:
                    team_obj = t
                    break

        if team_obj:
            bar = "█" * int(prob * 50)
            print(f"{i:2d}. [{team_obj.seed:2d}] {team:20s} {prob:5.1%} {bar}")

    # Identify upset picks
    print("\n[5/5] High-Value Upset Picks (Top 10)")
    print("-" * 70)

    upsets = simulator.identify_upset_picks(
        probabilities=probabilities,
        bracket=bracket,
        min_upset_prob=0.30,  # At least 30% chance
        max_seed_diff=4,  # Maximum 4-seed difference
    )

    if upsets:
        print(f"{'Matchup':<15} {'Underdog':<20} {'Prob':>6} {'Value':>7}")
        print("-" * 70)

        for upset in upsets[:10]:
            print(
                f"{upset.matchup:<15} "
                f"{upset.underdog:<20} "
                f"{upset.upset_probability:5.1%} "
                f"{upset.expected_value:6.3f}"
            )
    else:
        print("No high-value upsets identified")

    # Generate bracket recommendation
    print("\n" + "=" * 70)
    print("BRACKET RECOMMENDATION")
    print("=" * 70)

    recommendation = simulator.generate_bracket_recommendation(
        probabilities=probabilities, bracket=bracket
    )

    print(f"\nChampion: {recommendation.champion}")
    print(f"Champion Probability: {recommendation.champion_probability:.1%}")
    print(f"Confidence Score: {recommendation.confidence_score:.1%}")

    print(f"\nFinal Four:")
    for i, team in enumerate(recommendation.final_four, 1):
        # Find seed
        for region_teams in bracket.values():
            for t in region_teams:
                if t.name == team:
                    print(f"  {i}. [{t.seed}] {team} ({t.region})")
                    break

    print(f"\nTop Upset Picks:")
    for i, upset in enumerate(recommendation.upset_picks, 1):
        print(
            f"  {i}. {upset.matchup}: "
            f"{upset.underdog} ({upset.upset_probability:.0%} chance)"
        )

    # Export results
    print("\n" + "=" * 70)
    print("Exporting Results")
    print("=" * 70)

    output_dir = Path("examples/tournament_results")
    output_dir.mkdir(exist_ok=True)

    # Export round probabilities
    print("\nExporting round-by-round probabilities...")
    prob_data = []
    for team_name, prob in probabilities.round_32.items():
        team_obj = None
        for region_teams in bracket.values():
            for t in region_teams:
                if t.name == team_name:
                    team_obj = t
                    break

        if team_obj:
            prob_data.append(
                {
                    "team": team_name,
                    "seed": team_obj.seed,
                    "region": team_obj.region,
                    "adj_em": team_obj.adj_em,
                    "round_32": prob,
                    "sweet_16": probabilities.sweet_16.get(team_name, 0.0),
                    "elite_8": probabilities.elite_8.get(team_name, 0.0),
                    "final_4": probabilities.final_4.get(team_name, 0.0),
                    "championship": probabilities.championship.get(team_name, 0.0),
                    "champion": probabilities.winner.get(team_name, 0.0),
                }
            )

    df_probs = pd.DataFrame(prob_data)
    df_probs = df_probs.sort_values("champion", ascending=False)

    probs_file = output_dir / f"probabilities_{num_sims}sims.csv"
    df_probs.to_csv(probs_file, index=False)
    print(f"✓ Saved: {probs_file}")

    # Export upset picks
    print("\nExporting upset picks...")
    upset_data = [
        {
            "matchup": u.matchup,
            "favorite": u.favorite,
            "underdog": u.underdog,
            "favorite_seed": u.favorite_seed,
            "underdog_seed": u.underdog_seed,
            "upset_probability": u.upset_probability,
            "expected_value": u.expected_value,
        }
        for u in upsets
    ]

    df_upsets = pd.DataFrame(upset_data)
    upsets_file = output_dir / f"upset_picks_{num_sims}sims.csv"
    df_upsets.to_csv(upsets_file, index=False)
    print(f"✓ Saved: {upsets_file}")

    # Summary
    print("\n" + "=" * 70)
    print("Simulation Complete!")
    print("=" * 70)
    print(f"Simulations run: {num_sims:,}")
    print(f"Teams analyzed: {len(prob_data)}")
    print(f"Upset picks identified: {len(upsets)}")
    print(f"\nResults saved to: {output_dir}")
    print("\nNext steps:")
    print(f"  - View probabilities: open {probs_file}")
    print(f"  - Review upsets: open {upsets_file}")
    print("  - Adjust simulation parameters in code for different scenarios")


def analyze_seed_performance():
    """Analyze historical seed performance in simulations."""
    api_key = os.getenv("KENPOM_API_KEY")
    if not api_key:
        print("Error: KENPOM_API_KEY environment variable not set")
        return

    print("Seed Performance Analysis")
    print("=" * 70)

    # Initialize simulator
    simulator = TournamentSimulator(api_key=api_key, random_seed=42)

    # Load bracket
    selection_sunday = "2024-03-17"
    print(f"Loading bracket from {selection_sunday}...")

    try:
        bracket = simulator.load_bracket_from_kenpom(
            selection_sunday_date=selection_sunday
        )
    except Exception as e:
        print(f"Error loading bracket: {e}")
        return

    # Run simulations
    print("Running 10,000 simulations...")
    probabilities = simulator.simulate_tournament(
        bracket=bracket, num_simulations=10000, verbose=False
    )

    # Analyze by seed
    print("\nAnalyzing seed performance...")
    seed_performance = simulator.analyze_historical_seed_performance(
        probabilities=probabilities, bracket=bracket
    )

    # Display results
    print("\n" + "=" * 70)
    print("Average Advancement Rates by Seed")
    print("=" * 70)
    print(f"{'Seed':<6} {'R32':>6} {'S16':>6} {'E8':>6} {'F4':>6} {'Champ':>7}")
    print("-" * 70)

    for seed in sorted(seed_performance.keys()):
        rates = seed_performance[seed]
        print(
            f"{seed:<6} "
            f"{rates.get('round_32', 0.0):5.1%} "
            f"{rates.get('sweet_16', 0.0):5.1%} "
            f"{rates.get('elite_8', 0.0):5.1%} "
            f"{rates.get('final_4', 0.0):5.1%} "
            f"{rates.get('winner', 0.0):6.2%}"
        )

    # Save results
    output_dir = Path("examples/tournament_results")
    output_dir.mkdir(exist_ok=True)

    seed_data = []
    for seed, rates in seed_performance.items():
        seed_data.append(
            {
                "seed": seed,
                "round_32": rates.get("round_32", 0.0),
                "sweet_16": rates.get("sweet_16", 0.0),
                "elite_8": rates.get("elite_8", 0.0),
                "final_4": rates.get("final_4", 0.0),
                "champion": rates.get("winner", 0.0),
            }
        )

    df = pd.DataFrame(seed_data).sort_values("seed")
    output_file = output_dir / "seed_performance.csv"
    df.to_csv(output_file, index=False)

    print(f"\n✓ Results saved to: {output_file}")


def compare_bracket_scenarios():
    """Compare multiple bracket scenarios."""
    api_key = os.getenv("KENPOM_API_KEY")
    if not api_key:
        print("Error: KENPOM_API_KEY environment variable not set")
        return

    print("Bracket Scenario Comparison")
    print("=" * 70)

    # Test different random seeds to see variability
    scenarios = [42, 123, 456, 789, 2024]

    results = []

    for i, seed in enumerate(scenarios, 1):
        print(f"\n[{i}/{len(scenarios)}] Scenario {seed}")

        simulator = TournamentSimulator(api_key=api_key, random_seed=seed)

        try:
            bracket = simulator.load_bracket_from_kenpom(
                selection_sunday_date="2024-03-17"
            )

            probabilities = simulator.simulate_tournament(
                bracket=bracket,
                num_simulations=5000,  # Faster for comparison
                verbose=False,
            )

            # Get top champion
            top_champion = max(probabilities.winner.items(), key=lambda x: x[1])

            results.append(
                {
                    "scenario": seed,
                    "champion": top_champion[0],
                    "champion_prob": top_champion[1],
                }
            )

            print(f"  Top pick: {top_champion[0]} ({top_champion[1]:.1%})")

        except Exception as e:
            print(f"  Error: {e}")

    # Display comparison
    print("\n" + "=" * 70)
    print("Scenario Comparison")
    print("=" * 70)
    print(f"{'Scenario':<12} {'Champion':<20} {'Probability':>12}")
    print("-" * 70)

    for result in results:
        print(
            f"{result['scenario']:<12} "
            f"{result['champion']:<20} "
            f"{result['champion_prob']:11.1%}"
        )

    # Check consistency
    champions = [r["champion"] for r in results]
    most_common = max(set(champions), key=champions.count)
    consistency = champions.count(most_common) / len(champions)

    print(f"\nMost frequent champion: {most_common}")
    print(f"Consistency: {consistency:.0%}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        mode = sys.argv[1]

        if mode == "seed-analysis":
            analyze_seed_performance()
        elif mode == "compare":
            compare_bracket_scenarios()
        else:
            print(f"Unknown mode: {mode}")
            print("Usage: python tournament_simulator_demo.py [seed-analysis|compare]")
    else:
        main()
