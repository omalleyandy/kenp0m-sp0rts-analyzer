"""Compare KenPom FanMatch predictions with actual game results.

This script analyzes prediction accuracy by comparing FanMatch predictions
with actual game outcomes.
"""

from datetime import date

import pandas as pd

from kenp0m_sp0rts_analyzer.api_client import KenPomAPI


def compare_predictions_with_actuals(
    game_date: str | date, actual_results: dict[str, tuple[int, int]]
) -> None:
    """Compare FanMatch predictions with actual results.

    Args:
        game_date: Date in YYYY-MM-DD format or date object.
        actual_results: Dictionary mapping "Visitor @ Home" to (visitor_score,
            home_score) tuple.
    """
    print(f"\n{'='*80}")
    print(f"KenPom FanMatch Prediction Accuracy Analysis - {game_date}")
    print(f"{'='*80}\n")

    # Initialize API client
    try:
        api = KenPomAPI()
    except Exception as e:
        print(f"Error: {e}")
        return

    # Fetch FanMatch predictions
    print("Fetching FanMatch predictions...\n")
    try:
        response = api.get_fanmatch(game_date)
    except Exception as e:
        print(f"Error fetching data: {e}")
        return

    # Convert to DataFrame
    df = response.to_dataframe()

    if df.empty:
        print(f"No games found for {game_date}")
        return

    # Track accuracy metrics
    comparisons = []

    for _, game in df.iterrows():
        visitor = game["Visitor"]
        home = game["Home"]
        game_key = f"{visitor} @ {home}"

        # Check if we have actual results for this game
        if game_key not in actual_results:
            continue

        visitor_actual, home_actual = actual_results[game_key]

        # Predicted values
        visitor_pred = game["VisitorPred"]
        home_pred = game["HomePred"]
        home_wp = game["HomeWP"]
        pred_tempo = game["PredTempo"]
        thrill = game["ThrillScore"]

        # Calculate metrics
        pred_margin = home_pred - visitor_pred  # Positive = home favored
        actual_margin = home_actual - visitor_actual  # Positive = home won by more
        margin_error = abs(pred_margin - actual_margin)

        pred_total = visitor_pred + home_pred
        actual_total = visitor_actual + home_actual
        total_error = abs(pred_total - actual_total)

        # Did the favorite win?
        home_was_favorite = home_wp > 50
        home_actually_won = home_actual > visitor_actual
        favorite_correct = home_was_favorite == home_actually_won

        # Was it an upset?
        is_upset = (home_wp > 70 and not home_actually_won) or (
            home_wp < 30 and home_actually_won
        )

        comparisons.append(
            {
                "game": game_key,
                "visitor": visitor,
                "home": home,
                "visitor_pred": visitor_pred,
                "home_pred": home_pred,
                "visitor_actual": visitor_actual,
                "home_actual": home_actual,
                "pred_margin": pred_margin,
                "actual_margin": actual_margin,
                "margin_error": margin_error,
                "pred_total": pred_total,
                "actual_total": actual_total,
                "total_error": total_error,
                "home_wp": home_wp,
                "favorite_correct": favorite_correct,
                "is_upset": is_upset,
                "thrill": thrill,
            }
        )

    if not comparisons:
        print("No matching games found for comparison.")
        api.close()
        return

    comp_df = pd.DataFrame(comparisons)

    # Print detailed comparison
    print("=" * 80)
    print("GAME-BY-GAME COMPARISON")
    print("=" * 80 + "\n")

    for _, row in comp_df.iterrows():
        print(f"{row['game']}")
        print(
            f"  Predicted: {row['visitor']} {row['visitor_pred']:.1f}, "
            f"{row['home']} {row['home_pred']:.1f} "
            f"(spread: {row['home']} by {abs(row['pred_margin']):.1f}, "
            f"WP: {row['home_wp']:.1f}%)"
        )
        print(
            f"  Actual:    {row['visitor']} {row['visitor_actual']}, "
            f"{row['home']} {row['home_actual']} "
            f"(margin: {row['home']} by {row['actual_margin']:.1f})"
        )
        print(f"  Margin Error: {row['margin_error']:.1f} points")
        print(
            f"  Total Points: Predicted {row['pred_total']:.1f}, "
            f"Actual {row['actual_total']:.0f} (error: {row['total_error']:.1f})"
        )

        if row["is_upset"]:
            print("  *** UPSET ALERT! ***")
        if not row["favorite_correct"]:
            print("  Favorite did NOT win as predicted")

        print()

    # Summary statistics
    print("=" * 80)
    print("ACCURACY SUMMARY")
    print("=" * 80 + "\n")

    print(f"Games Analyzed: {len(comp_df)}")
    print()

    # Favorite/Underdog accuracy
    favorites_correct = comp_df["favorite_correct"].sum()
    accuracy_pct = (favorites_correct / len(comp_df)) * 100
    print(f"Favorite Prediction Accuracy: {favorites_correct}/{len(comp_df)} "
          f"({accuracy_pct:.1f}%)")
    print()

    # Upsets
    upsets = comp_df["is_upset"].sum()
    if upsets > 0:
        print(f"Upsets (predictions with >70% or <30% WP that were wrong): {upsets}")
        upset_games = comp_df[comp_df["is_upset"]]
        for _, game in upset_games.iterrows():
            print(f"  {game['game']} (predicted {game['home_wp']:.1f}% home win)")
        print()

    # Margin accuracy
    avg_margin_error = comp_df["margin_error"].mean()
    median_margin_error = comp_df["margin_error"].median()
    print("Margin of Victory Prediction:")
    print(f"  Average Error: {avg_margin_error:.2f} points")
    print(f"  Median Error: {median_margin_error:.2f} points")
    print(
        f"  Within 5 points: {len(comp_df[comp_df['margin_error'] <= 5])} "
        f"({len(comp_df[comp_df['margin_error'] <= 5]) / len(comp_df) * 100:.1f}%)"
    )
    print(
        f"  Within 10 points: {len(comp_df[comp_df['margin_error'] <= 10])} "
        f"({len(comp_df[comp_df['margin_error'] <= 10]) / len(comp_df) * 100:.1f}%)"
    )
    print()

    # Total points accuracy
    avg_total_error = comp_df["total_error"].mean()
    median_total_error = comp_df["total_error"].median()
    print("Total Points Prediction:")
    print(f"  Average Error: {avg_total_error:.2f} points")
    print(f"  Median Error: {median_total_error:.2f} points")
    print(
        f"  Within 5 points: {len(comp_df[comp_df['total_error'] <= 5])} "
        f"({len(comp_df[comp_df['total_error'] <= 5]) / len(comp_df) * 100:.1f}%)"
    )
    print(
        f"  Within 10 points: {len(comp_df[comp_df['total_error'] <= 10])} "
        f"({len(comp_df[comp_df['total_error'] <= 10]) / len(comp_df) * 100:.1f}%)"
    )
    print()

    # Best and worst predictions
    print("Most Accurate Margin Prediction:")
    best_margin = comp_df.nsmallest(1, "margin_error").iloc[0]
    print(
        f"  {best_margin['game']}: Error of {best_margin['margin_error']:.1f} points"
    )
    print()

    print("Least Accurate Margin Prediction:")
    worst_margin = comp_df.nlargest(1, "margin_error").iloc[0]
    print(
        f"  {worst_margin['game']}: Error of {worst_margin['margin_error']:.1f} points"
    )
    print()

    # Correlation analysis
    print("=" * 80)
    print("CORRELATION INSIGHTS")
    print("=" * 80 + "\n")

    # Did thrill score correlate with close games?
    close_games = comp_df[comp_df["margin_error"] <= 10]
    avg_thrill_close = close_games["thrill"].mean() if len(close_games) > 0 else 0
    avg_thrill_blowout = (
        comp_df[comp_df["margin_error"] > 10]["thrill"].mean()
        if len(comp_df[comp_df["margin_error"] > 10]) > 0
        else 0
    )
    print(f"Average Thrill Score for Close Games (<=10 pt margin error): "
          f"{avg_thrill_close:.1f}")
    print(f"Average Thrill Score for Blowouts (>10 pt margin error): "
          f"{avg_thrill_blowout:.1f}")
    print()

    # Win probability calibration
    print("Win Probability Calibration:")
    high_confidence_home = comp_df[comp_df["home_wp"] >= 80]
    if len(high_confidence_home) > 0:
        home_wins_80plus = high_confidence_home["favorite_correct"].sum()
        print(
            f"  80%+ Home WP: {home_wins_80plus}/{len(high_confidence_home)} "
            f"({home_wins_80plus / len(high_confidence_home) * 100:.1f}%) correct"
        )

    moderate_confidence_home = comp_df[
        (comp_df["home_wp"] >= 60) & (comp_df["home_wp"] < 80)
    ]
    if len(moderate_confidence_home) > 0:
        home_wins_60_80 = moderate_confidence_home["favorite_correct"].sum()
        print(
            f"  60-80% Home WP: {home_wins_60_80}/{len(moderate_confidence_home)} "
            f"({home_wins_60_80 / len(moderate_confidence_home) * 100:.1f}%) correct"
        )

    tossup_games = comp_df[(comp_df["home_wp"] >= 45) & (comp_df["home_wp"] <= 55)]
    if len(tossup_games) > 0:
        home_wins_tossup = (
            tossup_games["home_actual"] > tossup_games["visitor_actual"]
        ).sum()
        print(
            f"  45-55% Toss-ups: {home_wins_tossup}/{len(tossup_games)} "
            f"({home_wins_tossup / len(tossup_games) * 100:.1f}%) home wins"
        )

    api.close()


if __name__ == "__main__":
    # Actual results from December 16, 2025 (from ESPN)
    actual_results = {
        "Abilene Christian @ Arizona": (62, 96),
        "Lipscomb @ Duke": (73, 97),
        "Butler @ Connecticut": (60, 79),
        "Toledo @ Michigan St.": (69, 92),
        "Pacific @ BYU": (57, 93),
        "Louisville @ Tennessee": (62, 83),
        "East Tennessee St. @ North Carolina": (58, 77),
        "Queens @ Arkansas": (80, 108),
        "Towson @ Kansas": (49, 73),
        "Northern Colorado @ Texas Tech": (90, 101),
        "DePaul @ St. John's": (66, 79),
    }

    compare_predictions_with_actuals("2025-12-16", actual_results)
