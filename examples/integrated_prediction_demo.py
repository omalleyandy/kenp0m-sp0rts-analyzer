"""Comprehensive example demonstrating integrated prediction workflow.

This example shows the full pipeline from data sync to game prediction
with edge detection against Vegas lines.
"""

from datetime import date

# Import the integrated predictor
from kenp0m_sp0rts_analyzer import IntegratedPredictor


def main():
    """Run comprehensive prediction example."""
    print("=" * 60)
    print("KenPom Sports Analyzer - Integrated Prediction Demo")
    print("=" * 60)

    # Initialize predictor
    predictor = IntegratedPredictor()

    # Example 1: Simple game prediction
    print("\n1. Simple Game Prediction")
    print("-" * 40)
    try:
        result = predictor.predict_game(
            home_team="Duke",
            away_team="North Carolina"
        )
        print(f"   {result.home_team} vs {result.away_team}")
        print(f"   Predicted margin: {result.predicted_margin:+.1f}")
        print(f"   Predicted total: {result.predicted_total:.1f}")
        print(f"   Win probability: {result.win_probability:.1%}")
    except Exception as e:
        print(f"   (Demo mode - actual prediction requires KenPom data)")
        print(f"   Error: {e}")

    # Example 2: Prediction with Vegas lines
    print("\n2. Edge Detection vs Vegas")
    print("-" * 40)
    try:
        result = predictor.predict_game(
            home_team="Kansas",
            away_team="Kentucky",
            vegas_spread=-3.5,
            vegas_total=145.0
        )
        print(f"   {result.home_team} vs {result.away_team}")
        print(f"   Our prediction: {result.predicted_margin:+.1f}")
        print(f"   Vegas spread: {result.vegas_spread}")
        if result.edge_vs_spread:
            print(f"   Edge vs spread: {result.edge_vs_spread:+.1f}")
            if result.has_spread_edge:
                print("   >>> BETTING VALUE DETECTED! <<<")
    except Exception as e:
        print(f"   (Demo mode - actual prediction requires KenPom data)")

    # Example 3: Team comparison
    print("\n3. Team Comparison")
    print("-" * 40)
    try:
        comparison = predictor.compare_teams("Duke", "North Carolina")
        print(f"   {comparison['team1']['name']}:")
        print(f"      AdjEM: {comparison['team1']['adj_em']:.1f}")
        print(f"      Rank: #{comparison['team1']['rank']}")
        print(f"   {comparison['team2']['name']}:")
        print(f"      AdjEM: {comparison['team2']['adj_em']:.1f}")
        print(f"      Rank: #{comparison['team2']['rank']}")
        print(f"   Differential: {comparison['differential']['adj_em']:+.1f}")
    except Exception as e:
        print(f"   (Demo mode - comparison requires KenPom data)")

    # Example 4: Top teams
    print("\n4. Top 10 Teams by AdjEM")
    print("-" * 40)
    try:
        top_teams = predictor.get_top_teams(n=10)
        for i, team in enumerate(top_teams, 1):
            print(f"   {i:2}. {team.team_name}: {team.adj_em:+.1f}")
    except Exception as e:
        print(f"   (Demo mode - requires KenPom data)")

    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
