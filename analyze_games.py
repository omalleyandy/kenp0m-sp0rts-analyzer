#!/usr/bin/env python3
"""Analyze today's top NCAA basketball games with predictions and edge detection.

Uses the integrated prediction system with XGBoost and KenPom ensemble.
"""

from datetime import date
from kenp0m_sp0rts_analyzer import IntegratedPredictor


def print_game_analysis(result, game_num: int):
    """Print formatted analysis for a single game."""
    print(f"\n{'='*90}")
    print(f"GAME {game_num}: {result.away_team} @ {result.home_team}")
    print(f"{'='*90}")

    print(f"\nPREDICTION:")
    print(f"  Margin: {result.predicted_margin:+.1f} (Home team favored)" if result.predicted_margin > 0
          else f"  Margin: {abs(result.predicted_margin):.1f} (Away team favored)")
    print(f"  Total: {result.predicted_total:.1f} points")
    print(f"  Win Probability: {result.win_probability:.1%}")
    print(f"  Confidence Interval: ({result.confidence_interval[0]:+.1f}, {result.confidence_interval[1]:+.1f})")

    print(f"\nTEAM RATINGS:")
    print(f"  {result.home_team}:")
    print(f"    AdjEM: {result.home_rating.adj_em:+.2f} (#{result.home_rating.rank_adj_em})")
    print(f"    AdjO: {result.home_rating.adj_oe:.2f} (#{result.home_rating.rank_adj_oe})")
    print(f"    AdjD: {result.home_rating.adj_de:.2f} (#{result.home_rating.rank_adj_de})")

    print(f"  {result.away_team}:")
    print(f"    AdjEM: {result.away_rating.adj_em:+.2f} (#{result.away_rating.rank_adj_em})")
    print(f"    AdjO: {result.away_rating.adj_oe:.2f} (#{result.away_rating.rank_adj_oe})")
    print(f"    AdjD: {result.away_rating.adj_de:.2f} (#{result.away_rating.rank_adj_de})")

    if result.vegas_spread is not None:
        print(f"\nVEGAS LINES & EDGE DETECTION:")
        print(f"  Vegas Spread: {-result.vegas_spread:+.1f} (home team)")
        print(f"  Our Edge: {result.edge_vs_spread:+.1f} points")
        print(f"  Significant Edge: {'YES' if result.has_spread_edge else 'NO'}")

        if result.vegas_total is not None:
            print(f"  Vegas Total: {result.vegas_total:.1f}")
            print(f"  Total Edge: {result.edge_vs_total:+.1f}")
            print(f"  Significant Total Edge: {'YES' if result.has_total_edge else 'NO'}")

        if result.has_spread_edge:
            print(f"\n  [BETTING EDGE DETECTED]")
            if result.edge_vs_spread > 0:
                print(f"  -> Bet HOME {result.home_team} (Our line: {result.predicted_margin:+.1f}, Vegas: {-result.vegas_spread:+.1f})")
            else:
                print(f"  -> Bet AWAY {result.away_team} (Our line: {result.predicted_margin:+.1f}, Vegas: {-result.vegas_spread:+.1f})")


def main():
    """Analyze top games for today."""
    print("\n" + "="*90)
    print("KENPOM SPORTS ANALYZER - LIVE GAME PREDICTIONS")
    print(f"Date: {date.today()}")
    print("="*90)

    # Initialize predictor
    print("\nInitializing prediction system...")
    predictor = IntegratedPredictor()
    print("[OK] System ready with XGBoost + KenPom ensemble")

    # Today's featured games (examples - adjust for actual games)
    games = [
        # Top-25 matchups
        ("Auburn", "Duke", -6.5, 155.5),  # Example Vegas lines
        ("Kansas", "Kentucky", -3.0, 148.0),
        ("UConn", "Gonzaga", -4.5, 152.0),

        # Conference games
        ("Purdue", "Michigan", -8.5, 142.0),
        ("Arizona", "UCLA", -5.0, 150.0),
        ("Marquette", "Creighton", -2.5, 145.0),

        # Upset potential
        ("Villanova", "Xavier", -3.5, 138.0),
        ("Baylor", "Houston", -1.0, 140.0),
    ]

    print(f"\nAnalyzing {len(games)} games...")
    print(f"(Note: Using example games - adjust for today's actual schedule)")

    edges_found = []

    for i, (away, home, vegas_spread, vegas_total) in enumerate(games, 1):
        try:
            result = predictor.predict_game(
                home_team=home,
                away_team=away,
                vegas_spread=vegas_spread,
                vegas_total=vegas_total
            )

            print_game_analysis(result, i)

            if result.has_spread_edge or result.has_total_edge:
                edges_found.append((i, result))

        except Exception as e:
            print(f"\n[ERROR] Game {i} ({away} @ {home}): {e}")

    # Summary
    print(f"\n{'='*90}")
    print(f"SUMMARY: Found {len(edges_found)} game(s) with significant betting edges")
    print(f"{'='*90}\n")

    if edges_found:
        for game_num, result in edges_found:
            print(f"Game {game_num}: {result.away_team} @ {result.home_team}")
            if result.has_spread_edge:
                print(f"  -> Spread Edge: {result.edge_vs_spread:+.1f} points")
            if result.has_total_edge:
                print(f"  -> Total Edge: {result.edge_vs_total:+.1f} points")
        print()
    else:
        print("No significant edges detected in these games.\n")


if __name__ == "__main__":
    main()
