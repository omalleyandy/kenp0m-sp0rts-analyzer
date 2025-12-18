"""Analyze KenPom FanMatch predictions vs actual results.

This script fetches FanMatch predictions for a specific date and compares them
with actual game outcomes to evaluate prediction accuracy.
"""

from datetime import date

import pandas as pd

from kenp0m_sp0rts_analyzer.api_client import KenPomAPI


def analyze_fanmatch(game_date: str | date) -> None:
    """Analyze FanMatch predictions for a given date.

    Args:
        game_date: Date in YYYY-MM-DD format or date object.
    """
    print(f"\n{'='*80}")
    print(f"KenPom FanMatch Analysis - {game_date}")
    print(f"{'='*80}\n")

    # Initialize API client
    try:
        api = KenPomAPI()
    except Exception as e:
        print(f"Error: {e}")
        print("\nTo use this script, set your KENPOM_API_KEY environment variable:")
        print("  export KENPOM_API_KEY='your-api-key'  # Mac/Linux")
        print("  $env:KENPOM_API_KEY='your-api-key'   # Windows PowerShell")
        return

    # Fetch FanMatch predictions
    print("Fetching FanMatch predictions...\n")
    try:
        response = api.get_fanmatch(game_date)
    except Exception as e:
        print(f"Error fetching data: {e}")
        return

    # Convert to DataFrame for easier analysis
    df = response.to_dataframe()

    if df.empty:
        print(f"No games found for {game_date}")
        return

    print(f"Found {len(df)} games scheduled for {game_date}\n")

    # Explain the columns
    print("=" * 80)
    print("FANMATCH COLUMN DEFINITIONS")
    print("=" * 80)
    print(
        """
Field Definitions:
------------------
Season:       Ending year of the season (e.g., 2025 = 2024-25 season)
GameID:       Unique identifier for this game
DateOfGame:   Game date (YYYY-MM-DD format)

Teams:
  Visitor:      Visiting team name
  Home:         Home team name
  VisitorRank:  Visitor's KenPom ranking on game day
  HomeRank:     Home team's KenPom ranking on game day

Predictions:
  VisitorPred:  Predicted final score for visitor
  HomePred:     Predicted final score for home team
  HomeWP:       Home team win probability (0-100%)

  How to interpret HomeWP:
    - 50%: Toss-up game
    - 60-70%: Moderate favorite
    - 70-80%: Strong favorite
    - 80%+: Heavy favorite

Game Metrics:
  PredTempo:    Predicted game tempo (total possessions)
  ThrillScore:  Expected excitement level of the game (0-100)
                Higher = more competitive/exciting expected finish
    """
    )

    # Display all predictions
    print("\n" + "=" * 80)
    print("ALL GAME PREDICTIONS")
    print("=" * 80 + "\n")

    for _, game in df.iterrows():
        visitor = game["Visitor"]
        home = game["Home"]
        visitor_rank = game["VisitorRank"]
        home_rank = game["HomeRank"]
        visitor_pred = game["VisitorPred"]
        home_pred = game["HomePred"]
        home_wp = game["HomeWP"]
        pred_tempo = game["PredTempo"]
        thrill = game["ThrillScore"]

        # Calculate spread
        spread = home_pred - visitor_pred

        print(f"{visitor} (#{visitor_rank}) @ {home} (#{home_rank})")
        print(f"  Predicted Score: {visitor} {visitor_pred:.1f}, {home} {home_pred:.1f}")
        print(f"  Spread: {home} by {abs(spread):.1f}")
        print(f"  Win Probability: {home} {home_wp:.1f}%")
        print(f"  Predicted Tempo: {pred_tempo:.1f} possessions")
        print(f"  Thrill Score: {thrill:.1f}")

        # Categorize the game
        if 45 <= home_wp <= 55:
            category = "TOSS-UP"
        elif 55 < home_wp <= 65 or 35 <= home_wp < 45:
            category = "SLIGHT FAVORITE"
        elif 65 < home_wp <= 75 or 25 <= home_wp < 35:
            category = "MODERATE FAVORITE"
        elif 75 < home_wp <= 85 or 15 <= home_wp < 25:
            category = "STRONG FAVORITE"
        else:
            category = "HEAVY FAVORITE"

        print(f"  Category: {category}\n")

    # Summary statistics
    print("=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    print(f"\nTotal Games: {len(df)}")
    print(f"Average Home Win Probability: {df['HomeWP'].mean():.1f}%")
    print(f"Average Predicted Tempo: {df['PredTempo'].mean():.1f} possessions")
    print(f"Average Thrill Score: {df['ThrillScore'].mean():.1f}")
    print(f"\nPredicted Total Points Range: {df['VisitorPred'] + df['HomePred']}")
    total_min = (df["VisitorPred"] + df["HomePred"]).min()
    total_max = (df["VisitorPred"] + df["HomePred"]).max()
    total_avg = (df["VisitorPred"] + df["HomePred"]).mean()
    print(f"  Min: {total_min:.1f}")
    print(f"  Max: {total_max:.1f}")
    print(f"  Average: {total_avg:.1f}")

    # Competitive games
    close_games = df[(df["HomeWP"] >= 40) & (df["HomeWP"] <= 60)]
    print(f"\nClose Games (40-60% win probability): {len(close_games)}")
    if not close_games.empty:
        for _, game in close_games.iterrows():
            print(
                f"  {game['Visitor']} @ {game['Home']} ({game['HomeWP']:.1f}% home win)"
            )

    # High tempo games
    high_tempo = df[df["PredTempo"] >= df["PredTempo"].quantile(0.75)]
    print(f"\nHigh Tempo Games (top 25%): {len(high_tempo)}")
    if not high_tempo.empty:
        for _, game in high_tempo.iterrows():
            print(
                f"  {game['Visitor']} @ {game['Home']} "
                f"({game['PredTempo']:.1f} poss)"
            )

    # High thrill games
    high_thrill = df[df["ThrillScore"] >= df["ThrillScore"].quantile(0.75)]
    print(f"\nHigh Excitement Games (top 25% thrill): {len(high_thrill)}")
    if not high_thrill.empty:
        for _, game in high_thrill.iterrows():
            print(
                f"  {game['Visitor']} @ {game['Home']} "
                f"(thrill: {game['ThrillScore']:.1f})"
            )

    print("\n" + "=" * 80)
    print("ACTUAL RESULTS COMPARISON")
    print("=" * 80)
    print(
        "\nNote: To compare with actual results, we would need to fetch "
        "completed game data."
    )
    print(
        "The KenPom API doesn't provide live scores, so you would need to "
        "cross-reference"
    )
    print(
        "with another source like ESPN API, NCAA stats, or manually from "
        "game results."
    )
    print(
        "\nFor a full accuracy analysis, you would need to calculate:\n"
        "  1. Point spread accuracy (predicted margin vs actual margin)\n"
        "  2. Win probability calibration (predicted WP vs actual outcomes)\n"
        "  3. Total points accuracy (predicted total vs actual total)\n"
        "  4. Tempo accuracy (predicted tempo vs actual tempo)\n"
    )

    api.close()


if __name__ == "__main__":
    # Analyze December 16, 2025 games
    analyze_fanmatch("2025-12-16")
